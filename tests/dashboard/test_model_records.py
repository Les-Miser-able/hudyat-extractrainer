"""Manual model selection, provenance, and durable notes without ML dependencies."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fsl_trainer.dashboard.model_records import read_records, save_record, run_metadata


class ModelRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.known = [{'id': 'runs/greetings/a', 'category': 'greetings'},
                      {'id': 'runs/greetings/b', 'category': 'greetings'},
                      {'id': 'runs/colors/a', 'category': 'colors'}]
        self.run = self.root / 'runs/greetings/a'
        self.run.mkdir(parents=True)
        (self.run / 'best.keras').write_bytes(b'unchanged checkpoint')
        (self.run / 'test_metrics.json').write_text('{"accuracy": 0.9}')

    def save(self, run='runs/greetings/a', **values):
        return save_record(self.root, {'run': run, **values}, self.known)

    def test_preferences_notes_and_timings_survive_reload_without_changing_models(self):
        self.save(decision='Preferred', notes='Afternoon works with slower signing.', settle_ms=700,
                  missing_ms=450, frame_interval_ms=50, live_trials=10, live_correct=0)
        self.save('runs/colors/a', decision='Preferred', notes='Different category')
        self.save('runs/greetings/b', decision='Preferred')
        records = read_records(self.root)
        self.assertEqual(records['runs/greetings/b']['decision'], 'Preferred')
        self.assertEqual(records['runs/greetings/a']['decision'], 'Testing')
        self.assertEqual(records['runs/colors/a']['decision'], 'Preferred')
        self.assertEqual(records['runs/greetings/a']['settle_ms'], 700)
        self.assertEqual(records['runs/greetings/a']['live_correct'], 0)
        self.assertIn('Afternoon', records['runs/greetings/a']['notes'])
        self.assertEqual((self.run / 'best.keras').read_bytes(), b'unchanged checkpoint')
        self.assertEqual((self.run / 'test_metrics.json').read_text(), '{"accuracy": 0.9}')

    def test_unrecorded_timings_stay_blank(self):
        record = self.save(notes='Not tested yet')['runs/greetings/a']
        for key in ('settle_ms', 'missing_ms', 'frame_interval_ms', 'response_ms', 'live_trials'):
            self.assertIsNone(record[key])

    def test_invalid_record_never_replaces_saved_notes(self):
        self.save(notes='Keep this note')
        before = (self.root / 'runs/model_records.json').read_bytes()
        for values in ({'run': '../outside'}, {'decision': 'Best'}, {'settle_ms': True},
                       {'missing_ms': 99}, {'frame_interval_ms': 501},
                       {'live_trials': 2, 'live_correct': 3}, {'response_ms': float('nan')},
                       {'tested_on': '2026-02-31'}, {'notes': ['not text']}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.save(**values)
            self.assertEqual((self.root / 'runs/model_records.json').read_bytes(), before)

    def test_corrupt_records_are_not_silently_overwritten(self):
        path = self.root / 'runs/model_records.json'
        path.write_text('incomplete JSON')
        with self.assertRaisesRegex(ValueError, 'Could not read'):
            self.save(notes='replacement')
        self.assertEqual(path.read_text(), 'incomplete JSON')

    def test_failed_atomic_save_preserves_original_file_and_removes_temporary_file(self):
        self.save(notes='Keep this note')
        before = (self.root / 'runs/model_records.json').read_bytes()
        with patch('fsl_trainer.dashboard.model_records.os.replace', side_effect=PermissionError('locked')):
            with self.assertRaises(PermissionError):
                self.save(notes='New note')
        self.assertEqual((self.root / 'runs/model_records.json').read_bytes(), before)
        self.assertEqual(list((self.root / 'runs').glob('.model-records-*.tmp')), [])

    def test_metadata_reports_recorded_data_and_distinct_originals(self):
        (self.run / 'training_config.json').write_text(json.dumps({
            'dataset': 'old-machine/datasets/greetings', 'epochs': 100, 'batch_size': 32,
            'split_counts': {'train': {'a': 9, 'b': 9}, 'test': {'a': 2, 'b': 2}}}))
        (self.run / 'preprocessing.json').write_text(json.dumps({
            'source_selection': {'folders': ['S1', 'S2'], 'selected_files': 3},
            'video_augmentation': {'extra_samples': 8, 'max_displacement': .05, 'mirror_hands': True}}))
        (self.run / 'split_manifest.json').write_text(json.dumps([
            {'source': 'a/S1/one.mp4'}, {'source': 'a/S1/one.mp4'}, {'source': 'b/S2/two.mp4'}]))
        (self.run / 'history.csv').write_text('epoch,accuracy\n0,.5\n1,.7\n')
        metadata = run_metadata(self.run)
        self.assertEqual(metadata['dataset'], 'old-machine/datasets/greetings')
        self.assertEqual(metadata['original_sources'], 2)
        self.assertEqual(metadata['selected_sources'], 3)
        self.assertEqual(metadata['split_counts']['train'], 18)
        self.assertEqual(metadata['epochs_recorded'], 2)
        self.assertEqual(metadata['preparation']['video_extra_samples'], 8)
        self.assertTrue(metadata['preparation']['video_mirror_hands'])
        self.assertEqual(metadata['input_kind'], 'Video')


if __name__ == '__main__':
    unittest.main()
