"""Independent live/sample evaluations and portable model records, without ML."""
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fsl_trainer.dashboard.model_records import read_records, save_record
from fsl_trainer.dashboard.prediction_tests import read_tests, save_test, export_model, append_trial
from fsl_trainer.dashboard.prediction_trials import metrics_csv, trials_csv


class PredictionTestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.known = [{'id': f'runs/greetings/{name}', 'category': 'greetings', 'name': name,
                       'class_labels': ['afternoon', 'evening'], 'accuracy': .9,
                       'metadata': {'dataset': 'dataset-v1', 'training': {'epochs': 50}}}
                      for name in ('a', 'b')]
        for model in self.known:
            run = self.root / model['id']
            run.mkdir(parents=True)
            (run / 'best.keras').write_bytes(b'checkpoint')
            (run / 'training_config.json').write_text('{"epochs":50,"seed":42}')
            (run / 'preprocessing.json').write_text('{"normalization":"image"}')
            (run / 'test_metrics.json').write_text('{"accuracy":0.9,"macro_f1":0.89}')

    def save(self, **values):
        return save_test(self.root, {'run': self.known[0]['id'], **values}, self.known)['test']

    def test_tests_are_numbered_per_model_and_survive_reload_without_copying_checkpoints(self):
        a1 = self.save()
        a2 = self.save()
        b1 = self.save(run=self.known[1]['id'])
        self.assertEqual([a1['name'], a2['name'], b1['name']], ['Test 1', 'Test 2', 'Test 1'])
        self.assertEqual(len(read_tests(self.root)), 3)
        self.assertEqual(len(list(self.root.rglob('best.keras'))), 2)
        self.assertEqual((self.root / self.known[0]['id'] / 'best.keras').read_bytes(), b'checkpoint')

    def test_inherits_description_but_starts_with_fresh_measurements(self):
        save_record(self.root, {'run': self.known[0]['id'], 'decision': 'Preferred',
                               'notes': 'Model description', 'device': 'Laptop', 'settle_ms': 700,
                               'live_trials': 50, 'live_correct': 45, 'response_ms': 500}, self.known)
        original = read_records(self.root)
        test = self.save(missing_ms=450)
        self.assertEqual(test['model']['record'], original[self.known[0]['id']])
        self.assertEqual(test['model']['training_config']['seed'], 42)
        self.assertEqual(test['record']['notes'], 'Model description')
        self.assertEqual(test['record']['settle_ms'], 700)
        self.assertEqual(test['record']['missing_ms'], 450)
        for key in ('live_trials', 'live_correct', 'response_ms'):
            self.assertIsNone(test['record'][key])
        self.assertEqual(test['record']['decision'], 'Testing')
        self.assertEqual(read_records(self.root), original)

    def test_edits_preserve_the_original_model_snapshot_and_other_tests(self):
        first = self.save(live_trials=10, live_correct=0)
        second = self.save(name='Bright light')
        self.known[0]['metadata']['training']['epochs'] = 100
        (self.root / self.known[0]['id'] / 'training_config.json').write_text('{"epochs":100}')
        edited = self.save(id=first['id'], notes='All attempts missed', response_ms=700)
        self.assertEqual(edited['model'], first['model'])
        self.assertEqual(edited['record']['live_correct'], 0)
        self.assertEqual(edited['record']['live_trials'], 10)
        self.assertEqual(edited['record']['notes'], 'All attempts missed')
        self.assertEqual(read_tests(self.root)[second['id']], second)
        self.assertEqual(len(read_tests(self.root)), 2)

    def test_exports_include_current_record_and_historical_snapshots(self):
        test = self.save(prediction={'source': 'sample', 'sample': 'data/greetings/a.jpg',
                                    'captured_at': '2026-10-15T10:00:00',
                                    'probabilities': {'afternoon': .8, 'evening': .2}})
        save_record(self.root, {'run': self.known[0]['id'], 'notes': 'Current notes'}, self.known)
        exported = export_model(self.root, self.known[0]['id'], self.known)
        self.assertEqual(exported['model']['record']['notes'], 'Current notes')
        self.assertEqual(exported['model']['metrics']['macro_f1'], .89)
        self.assertEqual(exported['model']['preprocessing']['normalization'], 'image')
        self.assertEqual(exported['prediction_tests'], [test])
        self.assertEqual(exported['prediction_tests'][0]['model']['record'], {})
        json.dumps(exported, allow_nan=False)

    def test_invalid_input_does_not_replace_saved_tests(self):
        test = self.save()
        path = self.root / 'runs/prediction_tests.json'
        before = path.read_bytes()
        for values in ({'run': '../outside'}, {'id': 'missing'}, {'name': '  '},
                       {'live_trials': 2, 'live_correct': 3}, {'response_ms': float('nan')},
                       {'id': test['id'], 'run': self.known[1]['id']},
                       {'prediction': {'source': 'live', 'probabilities': {'wrong model': .9}}},
                       {'prediction': {'source': 'live', 'probabilities': {'afternoon': True}}}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.save(**values)
            self.assertEqual(path.read_bytes(), before)

    def test_atomic_failure_and_corrupt_files_preserve_saved_work(self):
        self.save()
        path = self.root / 'runs/prediction_tests.json'
        before = path.read_bytes()
        with patch('fsl_trainer.dashboard.prediction_tests.os.replace', side_effect=PermissionError('locked')):
            with self.assertRaises(PermissionError):
                self.save()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.glob('.prediction-tests-*.tmp')), [])
        path.write_text('broken JSON')
        with self.assertRaisesRegex(ValueError, 'Could not read'):
            self.save()
        self.assertEqual(path.read_text(), 'broken JSON')

    def test_simultaneous_saves_receive_distinct_test_numbers(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            tests = list(executor.map(lambda _: self.save(), range(8)))
        self.assertEqual(sorted(t['number'] for t in tests), list(range(1, 9)))
        self.assertEqual(len(read_tests(self.root)), 8)

    def trial(self, number, expected='afternoon', predicted='afternoon', response=100, **values):
        return {'id': f'{number:032x}', 'expected': expected, 'outcome': 'prediction',
                'probabilities': {label: .8 if label == predicted else .2 for label in ('afternoon', 'evening')},
                'response_ms': response, 'elapsed_ms': 600,
                'captured_at': '2026-10-17T10:00:00Z', 'frames': 12, 'duration_ms': 400,
                'settings': {'mode': 'gesture', 'settle_ms': 600, 'missing_ms': 400, 'frame_interval_ms': 50},
                **values}

    def append(self, test, trial):
        return append_trial(self.root, {'id': test['id'], 'trial': trial})['test']

    def test_automatic_metrics_include_wrong_predictions_and_unrecognized_attempts(self):
        test = self.save(automated=True, device_details={'browser': 'fixture'})
        for trial in (self.trial(1), self.trial(2, predicted='evening', response=400),
                      self.trial(3, expected='evening', predicted='evening', response=300),
                      self.trial(4, expected='evening', outcome='timeout', response_ms=None,
                                 probabilities={}, elapsed_ms=15000)):
            test = self.append(test, trial)
        report = test['measurements']
        self.assertEqual(report['trials'], 4)
        self.assertEqual(report['correct'], 2)
        self.assertEqual(report['accuracy'], .5)
        self.assertEqual(report['macro_precision'], .75)
        self.assertEqual(report['macro_recall'], .5)
        self.assertAlmostEqual(report['macro_f1'], 7 / 12)
        self.assertAlmostEqual(report['per_class']['afternoon']['f1'], 2 / 3)
        self.assertEqual(report['per_class']['evening']['unrecognized'], 1)
        self.assertEqual(report['lowest_sign_recall'], .5)
        self.assertEqual(report['lowest_recall_signs'], ['afternoon', 'evening'])
        self.assertEqual(report['confusion_matrix'], [[1, 1], [0, 1]])
        self.assertTrue(report['coverage_complete'])
        self.assertEqual(report['timing']['median_ms'], 300)
        self.assertEqual(report['timing']['p95_ms'], 400)
        self.assertEqual(report['timing']['samples'], 3)
        self.assertEqual(test['record']['live_correct'], 2)
        self.assertEqual(test['record']['live_trials'], 4)
        self.assertEqual(test['record']['response_ms'], 300)
        self.assertEqual(read_tests(self.root)[test['id']], test)
        self.assertEqual(test['model']['metrics'], {'accuracy': .9, 'macro_f1': .89})
        self.assertFalse((self.root / test['run'] / 'camera_timing.json').exists())
        self.assertEqual(read_records(self.root), {})

    def test_empty_partial_and_all_failed_sessions_keep_missing_evidence_explicit(self):
        test = self.save(automated=True)
        self.assertIsNone(test['measurements']['accuracy'])
        self.assertIsNone(test['record']['live_trials'])
        test = self.append(test, self.trial(1, outcome='incomplete', probabilities={}, response_ms=None))
        self.assertEqual(test['measurements']['accuracy'], 0)
        self.assertIsNone(test['measurements']['macro_recall'])
        self.assertIsNone(test['measurements']['lowest_sign_recall'])
        self.assertEqual(test['measurements']['missing_signs'], ['evening'])
        test = self.append(test, self.trial(2, expected='evening', outcome='stopped', probabilities={}, response_ms=None))
        self.assertEqual(test['measurements']['macro_precision'], 0)
        self.assertEqual(test['measurements']['macro_recall'], 0)
        self.assertEqual(test['measurements']['macro_f1'], 0)
        self.assertIsNone(test['measurements']['timing']['median_ms'])
        self.assertIsNone(test['prediction'])

    def test_retry_is_idempotent_and_completed_trials_cannot_be_replaced(self):
        test = self.save(automated=True)
        trial = {**self.trial(1), 'correct': False, 'predicted': 'evening'}
        first = self.append(test, trial)
        second = self.append(test, trial)
        self.assertEqual(first, second)
        self.assertTrue(first['trials'][0]['correct'], 'correctness must be derived from probabilities')
        with self.assertRaisesRegex(ValueError, 'cannot be replaced'):
            self.append(test, {**trial, 'expected': 'evening'})
        self.assertEqual(len(read_tests(self.root)[test['id']]['trials']), 1)

    def test_manual_edits_cannot_override_automatic_measurements_or_model_snapshot(self):
        test = self.append(self.save(automated=True), self.trial(1))
        before = test['model']
        self.known[0]['accuracy'] = 0
        edited = self.save(id=test['id'], live_trials=900, live_correct=800, response_ms=999, notes='New note')
        self.assertEqual(edited['record']['live_trials'], 1)
        self.assertEqual(edited['record']['live_correct'], 1)
        self.assertEqual(edited['record']['response_ms'], 100)
        self.assertEqual(edited['record']['notes'], 'New note')
        self.assertEqual(edited['model'], before)
        self.assertEqual(edited['trials'], test['trials'])

    def test_invalid_trials_preserve_saved_results_and_atomic_save_can_be_retried(self):
        test = self.save(automated=True)
        path = self.root / 'runs/prediction_tests.json'
        before = path.read_bytes()
        invalid = [{'expected': 'wrong'}, {'outcome': 'unknown'}, {'id': '../outside'},
                   {'response_ms': True}, {'response_ms': float('nan')}, {'elapsed_ms': -1},
                   {'response_ms': 700}, {'probabilities': {'afternoon': .8}},
                   {'probabilities': {'afternoon': .9, 'evening': .9}},
                   {'probabilities': {'afternoon': True, 'evening': 0}},
                   {'outcome': 'timeout'}, {'outcome': 'incomplete', 'probabilities': {}},
                   {'settings': {'mode': 'pose', 'frame_interval_ms': 0}}, {'frames': False}]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.append(test, self.trial(1, **values))
            self.assertEqual(path.read_bytes(), before)
        with patch('fsl_trainer.dashboard.prediction_tests.os.replace', side_effect=PermissionError('locked')):
            with self.assertRaises(PermissionError):
                self.append(test, self.trial(1))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.glob('.prediction-tests-*.tmp')), [])
        self.assertEqual(len(self.append(test, self.trial(1))['trials']), 1)
        with self.assertRaises(ValueError):
            self.append(self.save(), self.trial(2))

    def test_concurrent_appends_do_not_lose_trials(self):
        test = self.save(automated=True)
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda n: self.append(test, self.trial(n)), range(8)))
        stored = read_tests(self.root)[test['id']]
        self.assertEqual(stored['measurements']['trials'], 8)
        self.assertEqual(len({t['id'] for t in stored['trials']}), 8)

    def test_csv_exports_retain_trial_evidence_and_per_gesture_metrics(self):
        test = self.append(self.save(automated=True, name='=test', device='+camera'), self.trial(1))
        rows = list(csv.DictReader(io.StringIO(trials_csv(test))))
        self.assertEqual(rows[0]['Test'], "'=test")
        self.assertEqual(rows[0]['Device / session'], "'+camera")
        self.assertEqual(rows[0]['Expected gesture'], 'afternoon')
        self.assertEqual(rows[0]['Predicted gesture'], 'afternoon')
        self.assertEqual(rows[0]['Correct'], 'True')
        self.assertEqual(rows[0]['Response (ms)'], '100.0')
        metrics = list(csv.DictReader(io.StringIO(metrics_csv(test))))
        self.assertEqual(metrics[0]['Accuracy'], '1.0')
        self.assertEqual(metrics[0]['Precision'], '', 'macro evidence is incomplete')
        self.assertEqual(metrics[0]['Coverage complete'], 'False')
        self.assertEqual(metrics[1]['Gesture'], 'afternoon')
        self.assertEqual(metrics[1]['Recall'], '1.0')
        self.assertEqual(metrics[2]['Trials'], '0')


if __name__ == '__main__':
    unittest.main()
