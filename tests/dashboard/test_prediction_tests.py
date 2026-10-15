"""Independent live/sample evaluations and portable model records, without ML."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fsl_trainer.dashboard.model_records import read_records, save_record
from fsl_trainer.dashboard.prediction_tests import read_tests, save_test, export_model


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


if __name__ == '__main__':
    unittest.main()
