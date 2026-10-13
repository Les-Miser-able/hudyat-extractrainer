"""Dashboard contracts; no ML dependencies or training required."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import dashboard


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.base_patch = patch.object(dashboard, "BASE", self.root)
        self.base_patch.start()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.base_patch.stop)
        self.source = self.root / "data" / "alphabet" / "A"
        self.source.mkdir(parents=True)
        (self.source / "sample.jpg").write_bytes(b"fixture")
        self.dataset = self.root / "datasets_augmented" / "alphabet"
        self.dataset.mkdir(parents=True)
        (self.dataset / "manifest.json").write_text("[]")
        self.run = self.root / "runs" / "alphabet" / "first"
        self.run.mkdir(parents=True)
        (self.run / "best.keras").write_bytes(b"fixture")
        (self.run / "classes.json").write_text('["A", "B"]')
        (self.run / "test_metrics.json").write_text('{"accuracy": 0.9, "macro_f1": 0.89}')

    def test_inventory_and_results_are_real_artifacts(self):
        inventory = dashboard.inventory()
        self.assertEqual(inventory[0]["samples"], 1)
        self.assertTrue(inventory[0]["prepared"])
        runs = dashboard.runs()
        self.assertEqual(runs[0]["accuracy"], .9)
        self.assertEqual(runs[0]["classes"], 2)

    def test_paths_and_invalid_parameters_are_rejected(self):
        for path in ["../outside", str(self.root.parent / "outside")]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                dashboard.local_path(path, self.root)
        for name in ["..", "../alphabet", "a/b", "a\\b", "a:run", "name."]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                dashboard.folder_name(name)
        for value in ["NaN", "inf", 1.5, 0, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                dashboard.build_command({"action": "train", "category": "alphabet",
                                         "run": "fresh", "epochs": value})

    def test_commands_use_existing_cli_and_protect_outputs(self):
        command = dashboard.build_command({"action": "train", "category": "alphabet", "run": "fresh"})
        self.assertEqual(command[:3], [sys.executable, "-u", "-B"])
        self.assertIn(str(self.dataset), command)
        for action, extra in [("train", {"run": "first"}), ("prepare", {})]:
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, "already exists"):
                dashboard.build_command({"action": action, "category": "alphabet", **extra})
        with self.assertRaisesRegex(ValueError, "sum"):
            dashboard.build_command({"action": "prepare", "category": "alphabet",
                                     "datasets": "fresh_datasets", "validation": .7, "test": .4})
        command = dashboard.build_command({"action": "predict", "run": "runs/alphabet/first",
                                           "sample": "data/alphabet/A/sample.jpg"})
        self.assertIn(str(self.source / "sample.jpg"), command)
        self.assertFalse((self.root / "runs" / "alphabet" / "fresh").exists())

    def wait_job(self, jobs):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            snapshot = jobs.snapshot()
            if snapshot and snapshot[0]["exit_code"] is not None:
                return snapshot[0]
            time.sleep(.02)
        self.fail("Worker did not finish")

    def test_empty_sources_are_rejected_before_a_worker_starts(self):
        empty = self.root / "data" / "empty"
        empty.mkdir()
        (empty / ".gitkeep").write_text("")
        row = next(c for c in dashboard.inventory() if c["name"] == "empty")
        self.assertEqual(row["samples"], 0)
        self.assertIn("No gesture class folders", row["preparation_issue"])
        jobs = dashboard.Jobs()
        with patch.object(dashboard.subprocess, "Popen") as worker:
            with self.assertRaisesRegex(ValueError, "data/empty/A/"):
                jobs.start({"action": "prepare", "category": "empty", "datasets": "fresh_datasets"})
            worker.assert_not_called()
        self.assertFalse((self.root / "fresh_datasets").exists())

    def test_preparation_requires_two_classes_and_enough_sources(self):
        request = {"action": "prepare", "category": "alphabet", "datasets": "fresh_datasets"}
        with self.assertRaisesRegex(ValueError, "second class"):
            dashboard.build_command(request)
        second = self.source.parent / "B"
        second.mkdir()
        with self.assertRaisesRegex(ValueError, "three source files"):
            dashboard.build_command(request)
        for folder in [self.source, second]:
            for index in range(3):
                (folder / f"source-{index}.jpg").write_bytes(b"fixture")
        command = dashboard.build_command(request)
        self.assertIn("prepare", command)
        self.assertIsNone(dashboard.inventory()[0]["preparation_issue"])

    def test_worker_streams_logs_and_reports_failures(self):
        jobs = dashboard.Jobs()
        self.addCleanup(jobs.shutdown)
        command = [sys.executable, "-u", "-c", "print('sample output'); raise SystemExit(3)"]
        with patch.object(dashboard, "build_command", return_value=command):
            jobs.start({"action": "inspect", "category": "alphabet"})
        job = self.wait_job(jobs)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["exit_code"], 3)
        self.assertIn("sample output", job["logs"])

    def test_single_worker_and_cancellation(self):
        jobs = dashboard.Jobs()
        self.addCleanup(jobs.shutdown)
        command = [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(30)"]
        with patch.object(dashboard, "build_command", return_value=command):
            job_id = jobs.start({"action": "train", "category": "alphabet"})
            with self.assertRaisesRegex(ValueError, "already running"):
                jobs.start({"action": "train", "category": "alphabet"})
        jobs.cancel(job_id)
        self.assertEqual(self.wait_job(jobs)["status"], "cancelled")


class DashboardHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # HTTP checks must not scan the developer's potentially large datasets.
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base_patch = patch.object(dashboard, "BASE", Path(cls.temp.name).resolve())
        cls.base_patch.start()
        cls.addClassCleanup(cls.base_patch.stop)
        ui = Path(cls.temp.name) / "ui"
        shutil.copytree(dashboard.UI, ui)
        cls.ui_patch = patch.object(dashboard, "UI", ui)
        cls.ui_patch.start()
        cls.addClassCleanup(cls.ui_patch.stop)
        cls.server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
        cls.server.token = "test-token"
        cls.server.jobs = dashboard.Jobs()
        cls.origin = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, path, body=None, headers=None):
        request = Request(self.origin + path, data=json.dumps(body).encode() if body is not None else None,
                          headers=headers or {})
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def test_pages_and_results_are_served(self):
        for path in ["/", "/style.css", "/app.js", "/api/jobs"]:
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 200)
        status, body = self.request("/api/state")
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertIn("runs", result)
        self.assertEqual(result["token"], "test-token")

    def test_requests_cannot_escape_or_launch_cross_origin_jobs(self):
        self.assertEqual(self.request("/%2e%2e/dashboard.py")[0], 400)
        self.assertEqual(self.request("/api/jobs", {"action": "train"})[0], 403)
        headers = {"X-Dashboard-Token": "test-token", "Origin": "https://example.com"}
        self.assertEqual(self.request("/api/jobs", {"action": "train"}, headers)[0], 403)
        self.assertEqual(self.request("/", headers={"Host": "example.com"})[0], 403)
        self.assertEqual(self.request("/api/jobs", [], {"X-Dashboard-Token": "test-token"})[0], 400)
        self.assertEqual(self.server.jobs.snapshot(), [])


if __name__ == "__main__":
    unittest.main()
