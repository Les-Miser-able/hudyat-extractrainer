"""Video augmentation through the dataset preparation and inspection interfaces."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from fsl_trainer.paths import PROJECT_ROOT

import numpy as np
import trainer as t


class VideoAugmentationTests(unittest.TestCase):
    def sources(self, root, static=False, length=90):
        source = root / "raw"
        for label_index, label in enumerate(("A", "J")):
            folder = source / label
            folder.mkdir(parents=True)
            for index in range(3):
                frames = 1 if static and label == "A" else length
                x = np.zeros((frames, 128), np.float32)
                x[:, :63] = .2 + label_index * .1 + index * .01
                x[:, 0] = np.linspace(.1 + index * .01, .8 + index * .01, frames)
                x[:, 126] = 1
                np.save(folder / f"{index}.npy", x)
        return source

    def prepare(self, source, output, **options):
        values = dict(source=str(source), output=str(output), groups=None,
                      validation=.15, test=.15, seed=42, static_variants=8,
                      jitter=.0075, video_extra_samples=0, video_time_warp=.05)
        legacy = options.pop("legacy", False)
        values.update(options)
        if legacy:
            del values["video_extra_samples"], values["video_time_warp"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            t.prepare(argparse.Namespace(**values))
            return t.inspect_dataset(output)

    def test_training_sources_keep_originals_and_gain_distinct_timing_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.sources(root)
            baseline, _, _ = self.prepare(source, root / "baseline")
            manifest, _, counts = self.prepare(source, root / "augmented", video_extra_samples=3)
            self.assertEqual(len(manifest), 12)
            self.assertEqual({s: sum(c.values()) for s, c in counts.items()},
                             {"train": 8, "validation": 2, "test": 2})
            for original in baseline:
                siblings = [r for r in manifest if r["source"] == original["source"]]
                self.assertEqual(siblings[0], original)
                self.assertEqual(len(siblings), 4 if original["split"] == "train" else 1)
                base = np.load(root / "baseline" / original["path"])
                np.testing.assert_array_equal(np.load(root / "augmented" / original["path"]), base)
                seen = {base.tobytes()}
                for row in siblings[1:]:
                    sample = t.load_sequence(root / "augmented" / row["path"], fixed=True)
                    self.assertEqual(sample.dtype, np.float32)
                    self.assertTrue(row["synthetic"])
                    self.assertFalse(row["static"])
                    self.assertEqual(row["group"], original["group"])
                    self.assertEqual(row["variant_of"], original["variant_of"])
                    np.testing.assert_array_equal(sample[[0, -1]], base[[0, -1]])
                    self.assertTrue((np.diff(sample[:, 0]) >= 0).all())
                    positions = (sample[:, 0] - base[0, 0]) / (base[-1, 0] - base[0, 0])
                    self.assertLessEqual(float(np.max(np.abs(positions - np.linspace(0, 1, 32)))), .050001)
                    self.assertNotIn(sample.tobytes(), seen)
                    seen.add(sample.tobytes())


    def test_motionless_sources_report_shortfalls_instead_of_duplicate_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.sources(root, length=32)
            for file in source.glob("*/*.npy"):
                x = np.load(file)
                x[:, 0] = x[0, 0]
                # Signed zeros are numerically equal, not distinct gestures.
                x[:, 1] = np.where(np.arange(32) % 2, -0.0, 0.0)
                np.save(file, x)
            manifest, _, _ = self.prepare(source, root / "dataset", video_extra_samples=3)
            self.assertEqual(len(manifest), 6)
            report = json.loads((root / "dataset" / "video_augmentation_report.json").read_text())
            self.assertEqual(report["requested_extra_samples"], 6)
            self.assertEqual(report["generated_extra_samples"], 0)
            self.assertEqual(report["unfilled_extra_samples"], 6)
            self.assertEqual(report["duplicate_candidates"], 60)
            self.assertEqual(report["invalid_candidates"], 0)
            self.assertEqual(len(report["sources"]), 2)
            self.assertTrue(all(r["shortfall_reasons"] for r in report["sources"]))
            metadata = json.loads((root / "dataset" / "preprocessing.json").read_text())
            self.assertEqual(metadata["video_augmentation"]["extra_samples"], 3)
            self.assertEqual(metadata["video_augmentation"]["max_attempts_per_sample"], 10)


    def test_mixed_sources_preserve_legacy_outputs_and_report_augmentation_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.sources(root, static=True)
            baseline, _, _ = self.prepare(source, root / "baseline", legacy=True)
            disabled, _, _ = self.prepare(source, root / "disabled")
            self.assertEqual(baseline, disabled)
            for name in ("manifest.json", "preprocessing.json", "augmentation_summary.json"):
                self.assertEqual((root / "baseline" / name).read_bytes(), (root / "disabled" / name).read_bytes())
            self.assertFalse((root / "disabled" / "video_augmentation_report.json").exists())
            augmented, _, _ = self.prepare(source, root / "augmented", video_extra_samples=3)
            repeated, _, _ = self.prepare(source, root / "repeated", video_extra_samples=3)
            self.assertEqual(augmented, repeated)
            for row in baseline:
                self.assertIn(row, augmented)
                np.testing.assert_array_equal(np.load(root / "baseline" / row["path"]),
                                              np.load(root / "augmented" / row["path"]))
            for row in augmented:
                np.testing.assert_array_equal(np.load(root / "augmented" / row["path"]),
                                              np.load(root / "repeated" / row["path"]))
            summary = json.loads((root / "augmented" / "augmentation_summary.json").read_text())
            self.assertEqual(summary["original_sources"], 6)
            self.assertEqual(summary["augmented_static_sequences"], 24)
            self.assertEqual(summary["resampled_sequences"], 3)
            self.assertEqual(summary["augmented_sequence_samples"], 3)
            self.assertEqual(summary["requested_extra_samples"], 3)
            self.assertEqual(summary["unfilled_extra_samples"], 0)
            self.assertEqual(summary["total_output_sequences"], 30)
            for split in ("validation", "test"):
                self.assertEqual(summary["by_split"][split]["augmented_sequence_samples"], 0)


    def test_cli_prepares_extra_samples_and_rejects_invalid_controls_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.sources(root)
            command = [sys.executable, "-B", str(PROJECT_ROOT / "trainer.py"), "prepare",
                       "--source", str(source), "--output", str(root / "dataset")]
            result = subprocess.run(command + ["--video-extra-samples", "3", "--video-time-warp", ".05"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads((root / "dataset" / "manifest.json").read_text())), 12)
            self.assertIn("6 of 6 requested", result.stdout)
            for field, value in [("--video-extra-samples", "-1"), ("--video-extra-samples", "1.5"),
                                 ("--video-extra-samples", "True"), ("--video-extra-samples", "1001"),
                                 ("--video-time-warp", "0"), ("--video-time-warp", "nan"),
                                 ("--video-time-warp", ".11")]:
                output = root / "invalid"
                bad = subprocess.run(command[:-1] + [str(output), field, value],
                                     capture_output=True, text=True, timeout=30)
                self.assertNotEqual(bad.returncode, 0, (field, value))
                self.assertFalse(output.exists())


    def test_category_cli_forwards_video_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.sources(root / "data")
            command = [sys.executable, "-B", str(PROJECT_ROOT / "categories.py"), "prepare",
                       "--data", str(root / "data"), "--category", "raw",
                       "--datasets", str(root / "datasets"), "--dataset-name", "timing_v2", "--video-extra-samples", "3",
                       "--video-time-warp", ".05"]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((root / "datasets" / "raw" / "timing_v2" / "preprocessing.json").read_text())
            self.assertEqual(metadata["video_augmentation"]["extra_samples"], 3)
            self.assertEqual(metadata["video_augmentation"]["max_displacement"], .05)
            self.assertEqual(len(json.loads((root / "datasets" / "raw" / "timing_v2" / "manifest.json").read_text())), 12)

    def test_named_dataset_and_original_keep_independent_manifests_and_caches(self):
        from fsl_trainer.data.dataset_paths import prepared_datasets
        from fsl_trainer.training.cache import open_cache

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.sources(root / 'data')
            original = root / 'datasets' / 'raw'
            self.prepare(source, original)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                original_cache = open_cache(original)[0]
            before = {p.relative_to(original): p.read_bytes() for p in original.rglob('*') if p.is_file()}
            command = [sys.executable, '-B', str(PROJECT_ROOT / 'categories.py'), 'prepare',
                       '--data', str(root / 'data'), '--category', 'raw', '--datasets', str(root / 'datasets'),
                       '--dataset-name', 'augmented v2', '--video-extra-samples', '3']
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            named = original / 'augmented v2'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(len(t.inspect_dataset(original)[0]), 6)
                self.assertEqual(len(t.inspect_dataset(named)[0]), 12)
                named_cache = open_cache(named)[0]
                self.assertEqual(open_cache(original)[0], original_cache)
            self.assertEqual(original_cache.parent, original / '_training_cache')
            self.assertEqual(named_cache.parent, named / '_training_cache')
            self.assertEqual(prepared_datasets(original), [original, named])
            for path, contents in before.items():
                self.assertEqual((original / path).read_bytes(), contents, str(path))


    def test_two_frame_inputs_keep_hand_slots_and_binary_presence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.sources(root, length=2)
            for file in source.glob("*/*.npy"):
                x = np.load(file)
                x[1, 63:126] = x[1, :63]
                x[1, :63] = 0
                x[1, 126:] = [0, 1]
                np.save(file, x)
            manifest, _, _ = self.prepare(source, root / "dataset", video_extra_samples=3)
            self.assertEqual(len(manifest), 12)
            for row in manifest:
                sample = t.load_sequence(root / "dataset" / row["path"], fixed=True)
                np.testing.assert_array_equal(sample[[0, -1], 126:], [[1, 0], [0, 1]])
                self.assertTrue(np.isin(sample[:, 126:], [0, 1]).all())
                self.assertTrue((sample[sample[:, 126] == 0, :63] == 0).all())
                self.assertTrue((sample[sample[:, 127] == 0, 63:126] == 0).all())

    def test_sparse_detection_skips_invalid_candidates_and_keeps_valid_originals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.sources(root, length=256)
            for file in source.glob("*/*.npy"):
                x = np.load(file)
                kept = x[123].copy()
                x[:] = 0
                x[123] = kept
                np.save(file, x)
            manifest, _, _ = self.prepare(source, root / "dataset", video_extra_samples=3)
            report = json.loads((root / "dataset" / "video_augmentation_report.json").read_text())
            self.assertGreater(report["invalid_candidates"], 0)
            self.assertEqual(report["generated_extra_samples"], len(manifest) - 6)
            self.assertEqual(report["generated_extra_samples"] + report["unfilled_extra_samples"], 6)
            for row in manifest:
                self.assertTrue(t.load_sequence(root / "dataset" / row["path"], fixed=True)[:, 126:].any())

    def test_shared_source_groups_stay_together_with_all_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.sources(root)
            groups = root / "groups.csv"
            groups.write_text("path,group\n" + "".join(
                f"{label}/{index}.npy,signer{index}\n" for label in ("A", "J") for index in range(3)))
            manifest, _, _ = self.prepare(source, root / "dataset", groups=str(groups), video_extra_samples=3)
            self.assertEqual(len(manifest), 12)
            for group in {r["group"] for r in manifest}:
                rows = [r for r in manifest if r["group"] == group]
                self.assertEqual(len({r["split"] for r in rows}), 1)
                self.assertEqual({r["label"] for r in rows}, {"A", "J"})

    def test_source_streams_are_repeatable_independent_of_other_sources_and_change_with_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.sources(root)
            for folder in source.iterdir():
                for index in range(3, 6):
                    x = np.load(folder / "0.npy")
                    x[:, 0] += index * .01
                    np.save(folder / f"{index}.npy", x)
            first, _, _ = self.prepare(source, root / "first", video_extra_samples=3)
            other_seed, _, _ = self.prepare(source, root / "other_seed", video_extra_samples=3, seed=43)
            extra = source / "0"
            extra.mkdir()
            for index in range(3):
                x = np.zeros((1, 128), np.float32)
                x[:, 0], x[:, 126] = .9 + index * .01, 1
                np.save(extra / f"{index}.npy", x)
            reordered, _, _ = self.prepare(source, root / "reordered", video_extra_samples=3)
            def variants(rows):
                return {(r["source"], r["variant"]): r for r in rows if r.get("augmentation")}
            a, b, c = variants(first), variants(reordered), variants(other_seed)
            common = a.keys() & b.keys()
            self.assertTrue(common)
            for key in common:
                np.testing.assert_array_equal(np.load(root / "first" / a[key]["path"]),
                                              np.load(root / "reordered" / b[key]["path"]))
            common_seed = a.keys() & c.keys()
            self.assertTrue(common_seed)
            self.assertTrue(any(not np.array_equal(np.load(root / "first" / a[key]["path"]),
                                                   np.load(root / "other_seed" / c[key]["path"]))
                                for key in common_seed))

    def test_invalid_programmatic_controls_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.sources(root)
            cases = [{"video_extra_samples": value} for value in (True, False, -1, .5, 1001)]
            cases += [{"video_time_warp": value} for value in (True, 0, -.01, .11, float("nan"))]
            for options in cases:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    self.prepare(source, root / "invalid", **options)
                self.assertFalse((root / "invalid").exists())

    def test_packed_cache_covers_augmented_samples_and_tracks_settings(self):
        import training_cache as cache
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.sources(root)
            manifest, classes, _ = self.prepare(source, root / "dataset", video_extra_samples=3)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                packed, _, _, _ = cache.open_cache(root / "dataset")
                for split in cache.SPLITS:
                    rows = [r for r in manifest if r["split"] == split]
                    parts = list(cache.batches(packed, split, 3))
                    arrays = np.concatenate([x for x, _ in parts])
                    labels = np.concatenate([y for _, y in parts])
                    self.assertEqual(len(arrays), len(rows))
                    for index, row in enumerate(rows):
                        np.testing.assert_array_equal(arrays[index], np.load(root / "dataset" / row["path"]))
                        self.assertEqual(labels[index], classes.index(row["label"]))
                metadata_path = root / "dataset" / "preprocessing.json"
                metadata = json.loads(metadata_path.read_text())
                metadata["video_augmentation"]["max_displacement"] = .06
                t.save_json(metadata_path, metadata)
                changed, _, _, _ = cache.open_cache(root / "dataset")
                self.assertNotEqual(packed, changed)


if __name__ == "__main__":
    unittest.main()
