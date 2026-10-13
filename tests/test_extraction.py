import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from fsl_trainer.preparation import extraction as e
from fsl_trainer.preparation.pipeline import prepare
from fsl_trainer.preparation.dataset import inspect_dataset


def result(labels):
    return NS(hand_landmarks=[[NS(x=value, y=value+0.1, z=-0.01) for _ in range(21)]
                              for _, _, value in labels],
              handedness=[[NS(category_name=label, score=score)]
                          for label, score, _ in labels])


class ExtractionTests(unittest.TestCase):
    def test_slot_order_flags_and_collision(self):
        x = e.pack_result(result([("Right", 0.9, 0.2), ("Left", 0.95, 0.4), ("Left", 0.6, 0.8)]))
        self.assertAlmostEqual(float(x[0]), 0.4)
        self.assertAlmostEqual(float(x[63]), 0.2)
        np.testing.assert_array_equal(x[126:], [1, 1])

    def test_absence_and_swap(self):
        x = e.pack_result(result([("Right", 0.9, 0.2)]), swap_hands=True)
        np.testing.assert_array_equal(x[63:126], 0)
        np.testing.assert_array_equal(x[126:], [1, 0])
        np.testing.assert_array_equal(e.pack_result(result([])), np.zeros(128))

    def test_nested_extensions_and_source_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"nested").mkdir()
            for name in ("photo.JPG", "clip.MP4", "points.npy", "ignore.txt"):
                (root/"nested"/name).touch()
            self.assertEqual(len(e.source_files(root)), 3)
        self.assertEqual(e.default_group("A/one_jpg.rf.abc.jpg"), e.default_group("A/one_jpg.rf.xyz.jpg"))
        self.assertEqual(e.default_group("A/one_jpg.rf.abc.jpg"), e.default_group("A/one.jpg"))
        self.assertNotEqual(e.default_group("A/one.jpg"), e.default_group("B/one.jpg"))

    def test_raw_prepare_preserves_provenance_and_reports_rejections(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root/"raw"
            samples = {"A": ["one.rf.aaa.jpg", "one.rf.bbb.jpg", "two.jpg", "three.jpg"],
                       "J": ["one.mp4", "two.mp4", "three.mp4"]}
            for label, names in samples.items():
                (raw/label).mkdir(parents=True)
                for i, name in enumerate(names):
                    (raw/label/name).write_text(str(i+1))
            (raw/"A"/"bad.jpg").write_text("bad")
            (raw/"J"/"unknown.npz").write_bytes(b"unknown")
            asset = root/"hand.task"
            asset.write_bytes(b"fixture")
            class FakeExtractor:
                def __init__(self, **kwargs):
                    self.model, self.mp = asset, NS(__version__="fixture")
                    self.confidence, self.swap_hands = 0.5, False
                def extract(self, path):
                    if path.name == "bad.jpg":
                        raise ValueError("No hand detected")
                    frames = 1 if path.suffix == ".jpg" else 7
                    x = np.zeros((frames, 128), dtype=np.float32)
                    x[:, :63] = int(path.read_text()) * 0.1
                    x[:, 126] = 1
                    if frames > 1:
                        x[:, 0] += np.linspace(0, .05, frames)
                    return x, {"kind": "image" if frames == 1 else "video",
                               "frames": frames, "detected_frames": frames}
                def close(self):
                    pass
            args = argparse.Namespace(source=str(raw), output=str(root/"out"), groups=None,
                                      validation=0.15, test=0.15, seed=42, static_variants=8, jitter=0.0075)
            with patch("fsl_trainer.preparation.extraction.Extractor", FakeExtractor), contextlib.redirect_stdout(io.StringIO()):
                prepare(args)
            manifest, _, counts = inspect_dataset(root/"out")
            self.assertEqual(len(manifest), 35)
            self.assertEqual(sum(counts[s]["A"] for s in counts), 32)
            first = [r for r in manifest if "one.rf." in r["source"]]
            self.assertEqual(len({r["split"] for r in first}), 1)
            self.assertTrue(all(r["variant_of"] == r["source"] for r in manifest))
            report = json.loads((root/"out"/"extraction_report.json").read_text())
            self.assertEqual(len(report["rejected"]), 1)
            self.assertEqual(report["ignored"][0]["source"], "J/unknown.npz")
            self.assertEqual(report["rejected"][0]["source"], "A/bad.jpg")
            self.assertTrue((root/"out"/"extracted_sources.json").is_file())
            self.assertEqual(len(list((root/"out"/"extracted").glob("*/*.npy"))), 7)
            args.output = str(root / "augmented")
            args.video_extra_samples, args.video_time_warp = 3, .05
            with patch("extraction.Extractor", FakeExtractor), contextlib.redirect_stdout(io.StringIO()):
                t.prepare(args)
            augmented, _, _ = t.inspect_dataset(root / "augmented")
            self.assertEqual(len(augmented), 38)
            video_report = json.loads((root / "augmented" / "video_augmentation_report.json").read_text())
            self.assertEqual(video_report["generated_extra_samples"], 3)
            self.assertTrue(all(r["source"] in {"J/one.mp4", "J/two.mp4", "J/three.mp4"}
                                for r in video_report["sources"]))
            for row in augmented:
                self.assertIn(row["source"], {r["source"] for r in manifest})
                self.assertEqual(row["variant_of"], row["source"])
            for file in (root / "out" / "extracted").glob("*/*.npy"):
                np.testing.assert_array_equal(np.load(file), np.load(root / "augmented" / "extracted" / file.relative_to(root / "out" / "extracted")))



if __name__ == "__main__":
    unittest.main()
