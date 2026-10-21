"""Dashboard contracts; no ML dependencies or training required."""
import json
import csv
import hashlib
import io
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
from zipfile import ZipFile

import dashboard


class DashboardTests(unittest.TestCase):
    def test_preprocessing_defaults_are_isolated_to_alphabet_and_overridable(self):
        from fsl_trainer.data.preprocessing import HAND_RELATIVE
        for category in ['alphabet', 'colors', 'numbers', 'greetings', 'family', 'shapes']:
            for label in ['A', 'B']:
                folder = self.root / 'data' / category / label
                folder.mkdir(parents=True, exist_ok=True)
                for index in range(3):
                    (folder / f'{index}.jpg').touch()
            request = {'action': 'prepare', 'category': category, 'dataset_name': 'fresh', 'jitter': ''}
            command = dashboard.build_command(request)
            if category == 'alphabet':
                self.assertEqual(command[command.index('--normalization') + 1], HAND_RELATIVE)
                self.assertEqual(command[command.index('--jitter') + 1], '0.03')
            else:
                self.assertNotIn('--normalization', command)
                self.assertEqual(command[command.index('--jitter') + 1], '0.0075')
            command = dashboard.build_command({**request, 'normalization': 'image'})
            self.assertNotIn('--normalization', command)
            self.assertEqual(command[command.index('--jitter') + 1], '0.0075')
        with self.assertRaisesRegex(ValueError, 'normalization'):
            dashboard.build_command({**request, 'normalization': 'unknown'})

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

    def test_paper_report_jobs_use_saved_data_without_retraining(self):
        command = dashboard.build_command({'action': 'paper-report', 'dataset': 'datasets_augmented/alphabet'})
        self.assertIn('paper-report', command)
        self.assertIn(str(self.dataset), command)
        command = dashboard.build_command({'action': 'paper-report', 'run': 'runs/alphabet/first'})
        self.assertIn('--run', command)
        self.assertNotIn('train', command)
        command = dashboard.build_command({'action': 'paper-report'})
        self.assertIn('--all', command)
        for data in ({'dataset': '../outside'}, {'run': 'runs/alphabet/missing'}):
            with self.assertRaises(ValueError):
                dashboard.build_command({'action': 'paper-report', **data})

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

    def test_camera_timing_job_reads_hudyat_but_writes_under_trainer(self):
        (self.run / 'preprocessing.json').write_text(json.dumps({'extraction': {'model_sha256': 'a' * 64}}))
        command = dashboard.build_command({'action': 'hudyat-timing', 'run': 'runs/alphabet/first',
                                           'hudyat_project': '../HUDYAT', '_job_id': '0011223344556677'})
        self.assertEqual(command[3], str(self.root / 'tools/prepare_hudyat_timing.py'))
        self.assertEqual(command[command.index('--project') + 1], str(self.root.parent / 'HUDYAT'))
        self.assertEqual(command[command.index('--run') + 1], 'runs/alphabet/first')

    def test_benchmark_command_uses_selected_run_without_retraining(self):
        (self.run / 'training_config.json').write_text('{}')
        (self.run / 'split_manifest.json').write_text('[]')
        command = dashboard.build_command({'action': 'benchmark', 'run': 'runs/alphabet/first'})
        self.assertEqual(command[-3:], ['benchmark', '--run', str(self.run)])
        for run_id in ('../outside', 'runs/alphabet/missing'):
            with self.assertRaises(ValueError):
                dashboard.build_command({'action': 'benchmark', 'run': run_id})
        (self.run / 'training_config.json').unlink()
        with self.assertRaisesRegex(ValueError, 'training config'):
            dashboard.build_command({'action': 'benchmark', 'run': 'runs/alphabet/first'})

    def test_hudyat_export_command_uses_selected_model_and_verified_detector(self):
        detector = self.root / 'assets/hand_landmarker.task'
        detector.parent.mkdir()
        detector.write_bytes(b'detector')
        (self.run / 'preprocessing.json').write_text(json.dumps({'extraction': {
            'model_sha256': hashlib.sha256(b'detector').hexdigest()}}))
        request = {'action': 'export', 'run': 'runs/alphabet/first', '_job_id': '0011223344556677'}
        command = dashboard.build_command(request)
        self.assertEqual(command[:3], [sys.executable, '-u', '-B'])
        self.assertEqual(command[3], str(self.root / 'tools/export_hudyat_model.py'))
        self.assertEqual(command[command.index('--run') + 1], request['run'])
        self.assertEqual(command[command.index('--output') + 1], str(self.root / 'exports/hudyat/0011223344556677.zip'))
        self.assertEqual(command[command.index('--hand-model') + 1], str(detector))
        for changes in ({'run': '../outside'}, {'_job_id': '../outside'}, {'hand_model': '../outside.task'},
                        {'hand_model': True}, {'hand_model': ['assets/hand_landmarker.task']}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dashboard.build_command({**request, **changes})
        detector.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'hand detector'):
            dashboard.build_command(request)

    def test_video_controls_forward_and_reject_invalid_values(self):
        second = self.source.parent / "B"
        second.mkdir()
        for folder in (self.source, second):
            for index in range(3):
                (folder / f"source-{index}.mp4").write_bytes(b"fixture")
        request = {"action": "prepare", "dataset_name": "fresh", "category": "alphabet", "datasets": "fresh_datasets"}
        baseline = dashboard.build_command(request)
        self.assertNotIn("--video-extra-samples", baseline)
        self.assertNotIn('--video-mirror-hands', baseline)
        mirrored = dashboard.build_command({**request, 'video_mirror_hands': True})
        self.assertIn('--video-mirror-hands', mirrored)
        self.assertNotIn('--video-extra-samples', mirrored)
        for value in (1, 0, 'true', None):
            with self.assertRaises(ValueError):
                dashboard.build_command({**request, 'video_mirror_hands': value})
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

    def test_camera_timing_page_and_save_endpoint_are_bound_to_prepared_session(self):
        session = 'aabbccddeeff0011'
        run_id = 'runs/colors/camera_timing_fixture'
        base = Path(self.temp.name)
        run = base / run_id
        run.mkdir(parents=True, exist_ok=True)
        classes = ['blue', 'green', 'orange', 'red', 'violet', 'yellow']
        (run / 'classes.json').write_text(json.dumps(classes))
        (run / 'best.keras').write_bytes(b'model')
        folder = base / 'exports/hudyat-timing' / session
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'session.json').write_text(json.dumps({'session': session, 'run': run_id, 'classes': classes,
            'checkpoint_sha256': hashlib.sha256(b'model').hexdigest(), 'runtime_sources': {'runtime': 'hash'}}))
        (folder / 'index.html').write_text('<title>Camera timing</title>')
        self.assertEqual(self.request('/hudyat-timing/' + session + '/')[0], 200)
        self.assertEqual(self.request('/hudyat-timing/' + session + '/%2e%2e/%2e%2e/model.json')[0], 400)
        payload = {'session': session, 'device': {'user_agent': 'test browser'},
                   'trials': [{'target': 'blue', 'predicted': 'blue', 'correct': True, 'timeout': False,
                               'elapsed_ms': 2200, 'response_ms': 2200}]}
        headers = {'X-Dashboard-Token': 'test-token'}
        status, body = self.request('/api/hudyat-timing', payload, headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['benchmark']['response_time_ms'], 2200)
        self.assertEqual(self.request('/api/hudyat-timing', {**payload, 'session': '../outside'}, headers)[0], 400)

    def test_benchmark_csv_matches_category_and_preferred_filters(self):
        from fsl_trainer.reporting.benchmark import save_benchmark
        run_id = 'runs/colors/benchmark_fixture'
        run = Path(self.temp.name) / run_id
        run.mkdir(parents=True, exist_ok=True)
        classes = ['blue', 'red', 'green', 'yellow', 'black', 'white']
        metrics = {'accuracy': .9, 'macro_f1': .9, 'per_class': {
            label: {'precision': .9, 'recall': .9, 'f1': .9, 'support': 10} for label in classes}}
        (run / 'classes.json').write_text(json.dumps(classes))
        (run / 'test_metrics.json').write_text(json.dumps(metrics))
        (run / 'camera_timing.json').write_text(json.dumps({'timing': {'scope': 'hudyat_camera_to_result', 'median_ms': 100, 'samples': 30, 'tested_signs': classes, 'timeouts': 0}}))
        save_benchmark(run, metrics, classes, None)
        dashboard.save_record(Path(self.temp.name), {'run': run_id, 'decision': 'Preferred'}, dashboard.runs())
        with urlopen(self.origin + '/api/benchmarks/export?category=colors&preferred=1', timeout=5) as response:
            self.assertEqual(response.headers['Content-Type'], 'text/csv; charset=utf-8')
            self.assertIn('model-benchmarks.csv', response.headers['Content-Disposition'])
            rows = list(csv.DictReader(io.StringIO(response.read().decode('utf-8-sig'))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['Run'], 'benchmark_fixture')
        self.assertEqual(rows[0]['Macro precision (>=85%)'], '90.000000%')
        self.assertEqual(rows[0]['Met benchmark?'], 'Met')
        detail = json.loads(self.request('/api/run?id=' + run_id)[1])
        self.assertEqual(detail['benchmark']['lowest_sign_recall'], .9)

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

    def test_paper_report_pages_and_tables_are_served_without_exposing_source_files(self):
        from fsl_trainer.reporting.paper import Report
        folder = Path(self.temp.name) / 'reports' / 'paper'
        report = Report(folder, 'Study', 'Saved data')
        report.table('counts', 'Counts', ['Name', 'Count'], [['A', 3]])
        report.finish()
        self.assertEqual(self.request('/paper-report/reports/paper/index.html')[0], 200)
        status, body = self.request('/paper-report/reports/paper/counts.csv')
        self.assertEqual(status, 200)
        self.assertIn(b'A,3', body)
        self.assertEqual(self.request('/paper-report/%2e%2e/outside.csv')[0], 400)
        self.assertEqual(self.request('/paper-report/runs/prediction_tests.json')[0], 404)

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

    def test_prediction_tests_and_downloads_persist_model_details_with_authorization(self):
        run_id = 'runs/greetings/test_history_fixture'
        run = Path(self.temp.name) / run_id
        run.mkdir(parents=True, exist_ok=True)
        (run / 'classes.json').write_text('["afternoon", "evening"]')
        (run / 'training_config.json').write_text('{"epochs":50,"seed":42}')
        headers = {'X-Dashboard-Token': 'test-token'}
        payload = {'run': run_id, 'notes': 'Camera test', 'live_trials': 10, 'live_correct': 8}
        self.assertEqual(self.request('/api/prediction-test', payload)[0], 403)
        self.assertEqual(self.request('/api/prediction-test', {**payload, 'run': '../outside'}, headers)[0], 400)
        status, body = self.request('/api/prediction-test', payload, headers)
        self.assertEqual(status, 200)
        test = json.loads(body)['test']
        self.assertEqual(test['name'], 'Test 1')
        self.assertEqual(test['model']['training_config']['seed'], 42)
        state = json.loads(self.request('/api/state')[1])
        self.assertEqual(state['prediction_tests'][test['id']]['record']['live_correct'], 8)
        with urlopen(self.origin + '/api/model-record/export?run=' + run_id, timeout=5) as response:
            self.assertIn('attachment;', response.headers['Content-Disposition'])
            exported = json.loads(response.read())
            self.assertEqual(exported['model']['id'], run_id)
            self.assertEqual(exported['prediction_tests'], [test])
        with urlopen(self.origin + '/api/prediction-test/export?id=' + test['id'], timeout=5) as response:
            self.assertEqual(response.headers['Content-Disposition'], 'attachment; filename="prediction-test-1.json"')
            self.assertEqual(json.loads(response.read())['test'], test)
        self.assertEqual(self.request('/api/model-record/export?run=../outside')[0], 400)
        self.assertEqual(self.request('/api/prediction-test/export?id=missing')[0], 400)

    def test_native_model_zip_and_completed_hudyat_bundle_are_downloadable(self):
        run_id = 'runs/greetings/download_fixture'
        run = Path(self.temp.name) / run_id
        run.mkdir(parents=True, exist_ok=True)
        (run / 'best.keras').write_bytes(b'actual model')
        (run / 'classes.json').write_text('["afternoon", "evening"]')
        (run / 'preprocessing.json').write_text('{}')
        (run / 'benchmark.json').write_text('{}')
        (run / 'benchmark.csv').write_text('Model,Signs')
        with urlopen(self.origin + '/api/model/download?run=' + run_id, timeout=5) as response:
            self.assertEqual(response.headers['Content-Type'], 'application/zip')
            self.assertIn('model.zip', response.headers['Content-Disposition'])
            with ZipFile(io.BytesIO(response.read())) as archive:
                self.assertEqual(archive.read('best.keras'), b'actual model')
                self.assertIn('model-record.json', archive.namelist())
                self.assertIn('benchmark.json', archive.namelist())
                self.assertIn('benchmark.csv', archive.namelist())
        self.assertEqual(self.request('/api/model/download?run=../outside')[0], 400)
        job_id = '1122334455667788'
        bundle = Path(self.temp.name) / 'exports/hudyat' / (job_id + '.zip')
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(bundle, 'w') as archive:
            archive.writestr('model-assets/catalog.json', '{}')
        job = {'id': job_id, 'action': 'export', 'status': 'completed', 'run': run_id}
        with patch.object(self.server.jobs, 'snapshot', return_value=[job]):
            status, payload = self.request('/api/model/export/download?id=' + job_id)
            self.assertEqual(status, 200)
            with ZipFile(io.BytesIO(payload)) as archive:
                self.assertIn('model-assets/catalog.json', archive.namelist())
            for status in ('running', 'failed', 'cancelled'):
                job['status'] = status
                self.assertEqual(self.request('/api/model/export/download?id=' + job_id)[0], 400)
        self.assertEqual(self.request('/api/model/export/download?id=../outside')[0], 400)

    def test_automatic_trials_are_authorized_persistent_and_exportable_as_csv(self):
        run_id = 'runs/colors/automatic_fixture'
        run = Path(self.temp.name) / run_id
        run.mkdir(parents=True, exist_ok=True)
        (run / 'classes.json').write_text('["blue", "red"]')
        headers = {'X-Dashboard-Token': 'test-token'}
        status, body = self.request('/api/prediction-test', {'run': run_id, 'automated': True}, headers)
        self.assertEqual(status, 200)
        test = json.loads(body)['test']
        payload = {'id': test['id'], 'trial': {'id': '00112233445566778899aabbccddeeff',
            'expected': 'blue', 'outcome': 'prediction', 'probabilities': {'blue': .8, 'red': .2},
            'elapsed_ms': 700, 'response_ms': 500, 'captured_at': '2026-10-17T10:00:00Z',
            'settings': {'mode': 'gesture', 'settle_ms': 600, 'missing_ms': 400, 'frame_interval_ms': 50}}}
        self.assertEqual(self.request('/api/prediction-test/trial', payload)[0], 403)
        self.assertEqual(self.request('/api/prediction-test/trial', payload, headers)[0], 200)
        status, body = self.request('/api/prediction-test/trial', payload, headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['test']['measurements']['trials'], 1)
        state = json.loads(self.request('/api/state')[1])
        self.assertEqual(state['prediction_tests'][test['id']]['record']['live_correct'], 1)
        self.assertEqual(self.request('/api/prediction-test/trial', {**payload, 'id': 'missing'}, headers)[0], 400)
        for format in ('metrics', 'trials'):
            with urlopen(self.origin + '/api/prediction-test/export?id=' + test['id'] + '&format=' + format) as response:
                self.assertEqual(response.headers['Content-Type'], 'text/csv; charset=utf-8')
                self.assertIn(f'-{format}.csv', response.headers['Content-Disposition'])
                rows = list(csv.DictReader(io.StringIO(response.read().decode('utf-8-sig'))))
            if format == 'metrics':
                self.assertEqual(rows[0]['Accuracy (%)'], '100.00%')
                self.assertEqual(rows[0]['Median response (ms)'], '500.0')
                self.assertEqual(rows[0]['Coverage complete'], 'False')
            else:
                self.assertEqual(rows[0]['Expected gesture'], 'blue')
                self.assertEqual(rows[0]['Predicted gesture'], 'blue')
        exported = json.loads(self.request('/api/prediction-test/export?id=' + test['id'])[1])
        self.assertEqual(len(exported['test']['trials']), 1)
        self.assertEqual(self.request('/api/prediction-test/export?id=' + test['id'] + '&format=bad')[0], 400)

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
