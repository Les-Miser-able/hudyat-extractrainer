"""Public imports, script entry points, and project-relative paths survive moves."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fsl_trainer.paths import PROJECT_ROOT


class CompatibilityTests(unittest.TestCase):
    def test_legacy_imports_share_the_implementation_module(self):
        modules = {
            "trainer": "fsl_trainer.training.trainer",
            "training_cache": "fsl_trainer.training.cache",
            "extraction": "fsl_trainer.data.extraction",
            "categories": "fsl_trainer.data.categories",
            "confusion_matrix": "fsl_trainer.reporting.confusion_matrix",
            "progress": "fsl_trainer.progress",
            "dashboard": "fsl_trainer.dashboard.server",
        }
        for old, new in modules.items():
            with self.subTest(module=old):
                self.assertIs(importlib.import_module(old), importlib.import_module(new))

    def test_project_defaults_and_dashboard_assets_still_resolve(self):
        import categories
        import dashboard
        import extraction

        self.assertEqual(categories.BASE, PROJECT_ROOT)
        self.assertEqual(dashboard.BASE, PROJECT_ROOT)
        self.assertEqual(extraction.DEFAULT_MODEL, PROJECT_ROOT / "models" / "hand_landmarker.task")
        for asset in ("index.html", "app.js", "style.css"):
            self.assertTrue((dashboard.UI / asset).is_file(), asset)
            self.assertEqual(dashboard.local_path(asset, dashboard.UI), dashboard.UI / asset)

    def test_script_help_works_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            for script in ("trainer.py", "categories.py", "dashboard.py", "confusion_matrix.py"):
                with self.subTest(script=script):
                    result = subprocess.run(
                        [sys.executable, "-B", str(PROJECT_ROOT / script), "--help"],
                        cwd=folder, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_category_listing_works_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "data"
            gesture = data / "alphabet" / "A"
            gesture.mkdir(parents=True)
            (gesture / "sample.npy").touch()
            result = subprocess.run(
                [sys.executable, "-B", str(PROJECT_ROOT / "categories.py"),
                 "list", "--data", str(data)],
                cwd=folder, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["alphabet"]["source_files"], 1)


if __name__ == "__main__":
    unittest.main()
