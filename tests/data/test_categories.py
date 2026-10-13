import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import categories as c


class CategoryTests(unittest.TestCase):
    def options(self, root, **extra):
        options = dict(command="prepare", data=str(root/"data"), datasets=str(root/"datasets"),
                       runs=str(root/"runs"), category=None, all=True, seed=42,
                       validation=0.15, test=0.15, static_variants=8, jitter=0.0075,
                       groups_dir=None, run="first", epochs=1, batch_size=8,
                       learning_rate=0.001, patience=12, class_weights=False)
        options.update(extra)
        return argparse.Namespace(**options)

    def source(self, root, category):
        folder = root / "data" / category / "gesture"
        folder.mkdir(parents=True)
        (folder/"one.npy").touch()

    def test_discovery_and_isolated_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("numbers", "alphabet", "custom_category"):
                self.source(root, name)
            (root/"data"/"empty").mkdir()
            jobs, skipped = c.build_jobs(self.options(root))
            self.assertEqual(skipped, ["empty"])
            self.assertEqual([j[0] for j in jobs], ["alphabet", "custom_category", "numbers"])
            for name, target, command in jobs:
                self.assertEqual(target, (root/"datasets"/name).resolve())
                self.assertIn(str((root/"data"/name).resolve()), command)
            selected, _ = c.build_jobs(self.options(root, category=["numbers"]))
            self.assertEqual([j[0] for j in selected], ["numbers"])

    def test_train_has_separate_run_and_dataset_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("numbers", "alphabet"):
                folder = root/"datasets"/name
                folder.mkdir(parents=True)
                (folder/"manifest.json").write_text("[]")
            jobs, _ = c.build_jobs(self.options(root, command="train"))
            for name, target, command in jobs:
                self.assertEqual(target, (root/"runs"/name/"first").resolve())
                self.assertIn(str((root/"datasets"/name).resolve()), command)

    def test_empty_unknown_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.source(root, "alphabet")
            (root/"data"/"empty").mkdir()
            for name in ("missing", "empty"):
                with self.assertRaises(ValueError):
                    c.build_jobs(self.options(root, category=[name]))
            target = root/"datasets"/"alphabet"
            target.mkdir(parents=True)
            (target/"keep.txt").write_text("keep")
            with self.assertRaises(ValueError):
                c.build_jobs(self.options(root))
            self.assertEqual((target/"keep.txt").read_text(), "keep")

    def test_failures_reported_and_other_jobs_continue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.source(root, "alphabet")
            self.source(root, "numbers")
            with patch("categories.subprocess.run", side_effect=[
                argparse.Namespace(returncode=1), argparse.Namespace(returncode=0)]) as run:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "alphabet"):
                        c.execute(self.options(root))
                self.assertEqual(run.call_count, 2)

    def test_path_names_rejected(self):
        for value in ("..", "../escape", "a/b", "a\\b"):
            with self.assertRaises(argparse.ArgumentTypeError):
                c.category_name(value)


if __name__ == "__main__":
    unittest.main()
