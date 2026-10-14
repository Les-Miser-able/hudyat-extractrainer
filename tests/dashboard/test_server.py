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
        existing = self.dataset / "existing"
        existing.mkdir()
        (existing / "manifest.json").write_text("[]")
        for action, extra in [("train", {"run": "first"}), ("prepare", {"dataset_name": "existing"})]:
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, "already exists"):
                dashboard.build_command({"action": action, "category": "alphabet", "dataset": "datasets_augmented/alphabet", **extra})
        with self.assertRaisesRegex(ValueError, "sum"):
            dashboard.build_command({"action": "prepare", "dataset_name": "fresh", "category": "alphabet",
                                     "datasets": "fresh_datasets", "validation": .7, "test": .4})
        command = dashboard.build_command({"action": "predict", "run": "runs/alphabet/first",
                                           "sample": "data/alphabet/A/sample.jpg"})
        self.assertIn(str(self.source / "sample.jpg"), command)
        self.assertFalse((self.root / "runs" / "alphabet" / "fresh").exists())

    def test_video_controls_forward_and_reject_invalid_values(self):
        second = self.source.parent / "B"
        second.mkdir()
        for folder in (self.source, second):
            for index in range(3):
                (folder / f"source-{index}.mp4").write_bytes(b"fixture")
        request = {"action": "prepare", "dataset_name": "fresh", "category": "alphabet", "datasets": "fresh_datasets"}
        baseline = dashboard.build_command(request)
        self.assertNotIn("--video-extra-samples", baseline)
        command = dashboard.build_command({**request, "video_extra_samples": 3, "video_time_warp": .05})
        self.assertIn("--video-extra-samples", command)
        self.assertEqual(command[command.index("--video-extra-samples") + 1], "3")
        self.assertEqual(command[command.index("--video-time-warp") + 1], "0.05")
        for key, values in (("video_extra_samples", (-1, 1.5, True, False, 1001, "NaN")),
                            ("video_time_warp", (0, -.01, .11, True, "NaN", "inf"))):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    dashboard.build_command({**request, key: value})
        self.assertFalse((self.root / "fresh_datasets").exists())

    def wait_job(self, jobs):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            snapshot = jobs.snapshot()
            if snapshot and snapshot[0]["exit_code"] is not None:
                return snapshot[0]
            time.sleep(.02)
        self.fail("Worker did not finish")

    def test_folder_selection_counts_and_worker_command(self):
        for label in ("blue", "red", "violet"):
            for name in ("S1", "S2", "S4", "PUB"):
                folder = self.root / "data" / "colors" / label / name
                folder.mkdir(parents=True)
                if label != "violet" or name == "PUB":
                    (folder / "clip.MP4").touch()
            (self.root / "data" / "colors" / label / "PUB" / "photo.jpg").touch()
        row = next(c for c in dashboard.inventory() if c["name"] == "colors")
        self.assertEqual(row["samples"], 12)
        blue = next(c for c in row["classes"] if c["name"] == "blue")
        self.assertEqual(blue["folders"]["S4"], {"samples": 1, "videos": 1})
        self.assertEqual(blue["folders"]["PUB"], {"samples": 2, "videos": 1})
        request = {"action": "prepare", "dataset_name": "fresh", "category": "colors", "datasets": "fresh",
                   "source_folders": ["S1", "S2", "S4"], "videos_only": True}
        command = dashboard.build_command(request)
        self.assertEqual([command[i + 1] for i, value in enumerate(command) if value == "--source-folder"],
                         ["S1", "S2", "S4"])
        self.assertIn("--videos-only", command)
        self.assertNotIn("PUB", command)
        for folders in ([], ["PUB"], ["missing"], ["../PUB"], "S1"):
            with self.subTest(folders=folders), self.assertRaises(ValueError):
                dashboard.build_command({**request, "source_folders": folders})
        self.assertFalse((self.root / "fresh").exists())

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
                jobs.start({"action": "prepare", "dataset_name": "fresh", "category": "empty", "datasets": "fresh_datasets"})
            worker.assert_not_called()
        self.assertFalse((self.root / "fresh_datasets").exists())

    def test_preparation_requires_two_classes_and_enough_sources(self):
        request = {"action": "prepare", "dataset_name": "fresh", "category": "alphabet", "datasets": "fresh_datasets"}
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
        cls.server.live = dashboard.LiveInference(cls.server.jobs, Path(cls.temp.name))
        cls.origin = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.live.shutdown()
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
            with error:
                return error.code, error.read()

    def test_pages_and_results_are_served(self):
        for path in ["/", "/style.css", "/app.js", "/api/jobs"]:
            with self.subTest(path=path):
                status, body = self.request(path)
                self.assertEqual(status, 200)
                if path == "/":
                    self.assertIn(b'name="video_extra_samples"', body)
                    self.assertIn(b'name="video_time_warp"', body)
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

    def test_live_timing_updates_are_validated_before_worker_exchange(self):
        headers = {"X-Dashboard-Token": "test-token"}
        image = "data:image/jpeg;base64,fixture"
        request = {"session": "fixture", "image": image, "timestamp": 50,
                   "timing": {"settle_ms": 250, "missing_ms": 900}}
        with patch.object(self.server.live, "frame", return_value={"status": "capturing"}) as frame:
            self.assertEqual(self.request('/api/live/frame', request, headers)[0], 200)
            frame.assert_called_once_with("fixture", image, 50., timing=request['timing'])
            frame.reset_mock()
            for invalid in ({"settle_ms": True}, {"missing_ms": 99}, {"settle_ms": 2001},
                            {"missing_ms": 200.5}, {"missing_ms": "400"}, {"unknown": 100}, []):
                with self.subTest(invalid=invalid):
                    self.assertEqual(self.request('/api/live/frame', {**request, 'timing': invalid}, headers)[0], 400)
            frame.assert_not_called()

    def test_model_record_endpoint_persists_manual_choices_and_rejects_unknown_runs(self):
        run = Path(self.temp.name) / 'runs/greetings/record_fixture'
        run.mkdir(parents=True, exist_ok=True)
        (run / 'classes.json').write_text('["good_afternoon", "good_evening"]')
        (run / 'best.keras').write_bytes(b'checkpoint')
        headers = {'X-Dashboard-Token': 'test-token'}
        payload = {'run': 'runs/greetings/record_fixture', 'decision': 'Preferred',
                   'settle_ms': 700, 'missing_ms': 450, 'frame_interval_ms': 50, 'notes': 'Live-tested'}
        status, body = self.request('/api/model-record', payload, headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['model_records'][payload['run']]['notes'], 'Live-tested')
        state = json.loads(self.request('/api/state')[1])
        self.assertEqual(state['model_records'][payload['run']]['decision'], 'Preferred')
        self.assertEqual((run / 'best.keras').read_bytes(), b'checkpoint')
        self.assertEqual(self.request('/api/model-record', payload)[0], 403)
        self.assertEqual(self.request('/api/model-record', {**payload, 'run': '../outside'}, headers)[0], 400)
        self.assertEqual(self.request('/api/model-record', {**payload, 'missing_ms': False}, headers)[0], 400)

    def test_live_endpoints_validate_paths_and_forward_frames(self):
        root = Path(self.temp.name)
        run = root / "runs" / "colors" / "live_fixture"
        run.mkdir(parents=True, exist_ok=True)
        (run / "best.keras").write_bytes(b"fixture")
        (run / "preprocessing.json").write_text('{"extraction":{"model_sha256":"fixture"}}')
        headers = {"X-Dashboard-Token": "test-token"}
        with patch.object(self.server.live, "start", return_value={"session": "fixture"}) as start:
            status, body = self.request('/api/live/start', {"run": "runs/colors/live_fixture", "mode": "motion"}, headers)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["session"], "fixture")
            start.assert_called_once_with(run, None, "motion")
        with patch.object(self.server.live, "start", return_value={"session": "fixture"}) as start:
            self.assertEqual(self.request('/api/live/start', {"run": "runs/colors/live_fixture", "mode": "gesture"}, headers)[0], 200)
            start.assert_called_once_with(run, None, "gesture")
        with patch.object(self.server.live, "frame", return_value={"status": "no_hand"}) as frame:
            image = "data:image/jpeg;base64,fixture"
            self.assertEqual(self.request('/api/live/frame', {"session": "fixture", "image": image, "timestamp": 50}, headers)[0], 200)
            frame.assert_called_once_with("fixture", image, 50.)
            for data in ({"image": "invalid", "timestamp": 50}, {"image": image, "timestamp": True}):
                self.assertEqual(self.request('/api/live/frame', data, headers)[0], 400)
        with patch.object(self.server.live, "stop") as stop:
            self.assertEqual(self.request('/api/live/stop', {"session": "fixture"}, headers)[0], 200)
            stop.assert_called_once_with("fixture")
        with patch.object(self.server.live, "reset", return_value={"status": "waiting"}) as reset:
            self.assertEqual(self.request('/api/live/reset', {"session": "fixture"}, headers)[0], 200)
            reset.assert_called_once_with("fixture")
            self.assertEqual(self.request('/api/live/reset', {}, headers)[0], 400)
        self.assertEqual(self.request('/api/live/start', {"run": "../outside"}, headers)[0], 400)
        self.assertEqual(self.request('/api/live/start', {"run": "runs/colors/live_fixture", "mode": "invalid"}, headers)[0], 400)
        self.assertEqual(self.request('/api/live/start', {"run": "runs/colors/live_fixture"})[0], 403)


if __name__ == "__main__":
    unittest.main()
