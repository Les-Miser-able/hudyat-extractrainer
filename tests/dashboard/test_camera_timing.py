"""Camera timing outcomes, session provenance, persistence, and incomplete evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from fsl_trainer.dashboard.camera_timing import camera_session, save_camera_timing
from fsl_trainer.reporting.benchmark import read_benchmark, save_benchmark


class CameraTimingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.run_id = 'runs/colors/model'
        self.run = self.base / self.run_id
        self.run.mkdir(parents=True)
        self.classes = ['blue', 'green', 'orange', 'red', 'violet', 'yellow']
        self.metrics = {'accuracy': .9, 'macro_f1': .9, 'per_class': {
            label: {'precision': .9, 'recall': .9, 'f1': .9, 'support': 10} for label in self.classes}}
        (self.run / 'classes.json').write_text(json.dumps(self.classes))
        (self.run / 'test_metrics.json').write_text(json.dumps(self.metrics))
        (self.run / 'best.keras').write_bytes(b'checkpoint')
        self.known = [{'id': self.run_id, 'category': 'colors', 'class_labels': self.classes}]
        self.session = '0011223344556677'
        self.folder = self.base / 'exports/hudyat-timing' / self.session
        self.folder.mkdir(parents=True)
        self.config = {'session': self.session, 'run': self.run_id, 'classes': self.classes,
                       'checkpoint_sha256': hashlib.sha256(b'checkpoint').hexdigest(),
                       'runtime_sources': {'category-recognition.tsx': 'source hash'}}
        (self.folder / 'session.json').write_text(json.dumps(self.config))
        (self.folder / 'index.html').write_text('test page')

    def payload(self, milliseconds=1500):
        return {'session': self.session, 'device': {'user_agent': 'test browser', 'backend': 'webgl'},
                'trials': [{'target': label, 'predicted': label, 'correct': True,
                            'response_ms': milliseconds, 'elapsed_ms': milliseconds, 'timeout': False}
                           for label in self.classes]}

    def test_full_camera_results_persist_and_local_timing_cannot_replace_them(self):
        local = {'scope': 'local_model_inference', 'median_ms': 10, 'samples': 30}
        save_benchmark(self.run, self.metrics, self.classes, local)
        before = (self.run / 'test_metrics.json').read_bytes()
        report = save_camera_timing(self.base, self.payload(), self.known)['benchmark']
        self.assertEqual(report['response_time_ms'], 1500)
        self.assertEqual(report['timing_scope'], 'hudyat_camera_to_result')
        self.assertEqual(report['status'], 'Not met')
        self.assertEqual(report['local_inference']['median_ms'], 10)
        stored = json.loads((self.run / 'camera_timing.json').read_text())
        self.assertEqual(len(stored['sessions'][self.session]['trials']), 6)
        self.assertEqual(stored['timing']['runtime_sources'], self.config['runtime_sources'])
        self.assertEqual((self.run / 'test_metrics.json').read_bytes(), before)
        self.assertEqual(read_benchmark(self.run, self.metrics, self.classes, 'colors')['response_time_ms'], 1500)

    def test_partial_coverage_is_incomplete_and_timeouts_are_not_omitted(self):
        data = self.payload(900)
        data['trials'] = data['trials'][:1]
        report = save_camera_timing(self.base, data, self.known)['benchmark']
        self.assertEqual(report['status'], 'Incomplete')
        data = self.payload(900)
        data['trials'][0].update(predicted=None, correct=False, response_ms=None, elapsed_ms=15000, timeout=True)
        report = save_camera_timing(self.base, data, self.known)['benchmark']
        self.assertEqual(report['timing']['samples'], 5)
        self.assertEqual(report['timing']['timeouts'], 1)
        self.assertEqual(report['status'], 'Not met')

    def test_wrong_confirmations_are_retained_and_count_toward_response_times(self):
        data = self.payload(900)
        data['trials'][0].update(predicted='red', correct=False)
        report = save_camera_timing(self.base, data, self.known)['benchmark']
        self.assertEqual(report['timing']['samples'], 6)
        self.assertEqual(report['timing']['correct_trials'], 5)
        self.assertEqual(report['status'], 'Met')

    def test_changed_model_and_invalid_trial_never_replace_saved_results(self):
        save_camera_timing(self.base, self.payload(), self.known)
        path = self.run / 'camera_timing.json'
        before = path.read_bytes()
        for change in ({'response_ms': float('nan')}, {'response_ms': True}, {'correct': False},
                       {'target': 'unknown'}, {'elapsed_ms': -1}, {'timeout': True}):
            data = self.payload()
            data['trials'][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                save_camera_timing(self.base, data, self.known)
            self.assertEqual(path.read_bytes(), before)
        (self.run / 'best.keras').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            save_camera_timing(self.base, self.payload(), self.known)
        for session in ('../outside', 'bad'):
            with self.assertRaises(ValueError):
                camera_session(self.base, session)


if __name__ == '__main__':
    unittest.main()
