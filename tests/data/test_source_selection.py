"""Source filtering is identical in discovery and actual preparation."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import extraction as e
import trainer as t
from fsl_trainer.data.source_selection import folder_selection


class SourceSelectionTests(unittest.TestCase):
    def test_selection_matches_immediate_subfolders_and_nested_clips(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("S1/clip.mp4", "S4/nested/clip.MOV", "PUB/clip.mp4",
                         "root.mp4", "S1/photo.jpg", "S1/.hidden/clip.mp4", "PUB/S1/clip.mp4"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            paths = e.source_files(root, ["S1", "S4"], True)
            self.assertEqual([p.relative_to(root).as_posix() for p in paths],
                             ["S1/clip.mp4", "S4/nested/clip.MOV"])
            self.assertEqual(len(e.source_files(root)), 6)
            self.assertEqual(folder_selection(["S4", "S1", "S1"]), ["S1", "S4"])
            for value in ([], "S1", [".."], ["S1/nested"], [True]):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    folder_selection(value)

    def test_selected_video_preparation_keeps_provenance_and_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, output = root / "raw", root / "out"
            selected_paths, calls = [], []
            for label in ("blue", "red", "violet"):
                for name in ("S1", "S2", "S4", "PUB"):
                    folder = raw / label / name
                    folder.mkdir(parents=True)
                    if label != "violet" or name == "PUB":
                        path = folder / "clip.mp4"
                        path.touch()
                        if name != "PUB":
                            selected_paths.append(path)
                    (folder / "photo.jpg").touch()
            asset = root / "hand.task"
            asset.write_bytes(b"fixture")

            class FakeExtractor:
                def __init__(self, **kwargs):
                    self.model, self.mp = asset, NS(__version__="fixture")
                    self.confidence, self.swap_hands = .5, False

                def extract(self, path):
                    calls.append(path)
                    if path not in selected_paths:
                        raise AssertionError(f"Excluded source reached extraction: {path}")
                    x = np.zeros((9, 128), dtype=np.float32)
                    x[:, :63] = (selected_paths.index(path) + 1) * .1
                    x[:, 0] += np.linspace(0, .05, len(x))
                    x[:, 126] = 1
                    return x, {"kind": "video", "frames": len(x), "detected_frames": len(x)}

                def close(self):
                    pass

            groups = root / "groups.csv"
            with groups.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["path", "group"])
                writer.writerows((p.relative_to(raw).as_posix(), p.parent.name) for p in selected_paths)
            args = argparse.Namespace(source=str(raw), output=str(output), groups=str(groups),
                                      source_folders=["S1", "S2", "S4"], videos_only=True,
                                      validation=.15, test=.15, seed=42, static_variants=8, jitter=.0075,
                                      video_extra_samples=3, video_time_warp=.05)
            with patch("extraction.Extractor", FakeExtractor), contextlib.redirect_stdout(io.StringIO()):
                t.prepare(args)
            manifest, classes, _ = t.inspect_dataset(output)
            self.assertEqual(classes, ["blue", "red"])
            self.assertEqual(set(calls), set(selected_paths))
            self.assertEqual(len(manifest), 12)
            self.assertEqual({r["source"] for r in manifest},
                             {p.relative_to(raw).as_posix() for p in selected_paths})
            for group in ("S1", "S2", "S4"):
                self.assertEqual(len({r["split"] for r in manifest if r["group"] == group}), 1)
            report = json.loads((output / "extraction_report.json").read_text())
            metadata = json.loads((output / "preprocessing.json").read_text())
            self.assertEqual(report["source_selection"], metadata["source_selection"])
            self.assertEqual(report["source_selection"]["excluded_classes"], ["violet"])
            self.assertEqual(report["source_selection"]["selected_files"], 6)

    def test_nested_landmark_selection_needs_no_media_extractor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for label in ("A", "B", "empty"):
                for folder in ("S4", "PUB"):
                    target = root / "raw" / label / folder
                    target.mkdir(parents=True)
                    for index in range(3 if label != "empty" or folder == "PUB" else 0):
                        x = np.zeros((5, 128), dtype=np.float32)
                        x[:, :63] = (index + 1 + (10 if label == "B" else 0)) * .01
                        x[:, 126] = 1
                        np.save(target / f"{index}.npy", x)
            args = argparse.Namespace(source=str(root / "raw"), output=str(root / "out"), groups=None,
                                      source_folders=["S4"], videos_only=False,
                                      validation=.15, test=.15, seed=42, static_variants=8, jitter=.0075)
            with patch("extraction.Extractor", side_effect=AssertionError("Unexpected media dependency")), \
                    contextlib.redirect_stdout(io.StringIO()):
                t.prepare(args)
            manifest, classes, _ = t.inspect_dataset(root / "out")
            self.assertEqual(classes, ["A", "B"])
            self.assertEqual(len(manifest), 6)
            self.assertTrue(all("/S4/" in row["source"] for row in manifest))
            metadata = json.loads((root / "out" / "preprocessing.json").read_text())
            self.assertEqual(metadata["source_selection"]["excluded_classes"], ["empty"])


if __name__ == "__main__":
    unittest.main()
