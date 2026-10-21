"""Deletion is exact, recoverable, and serialized with jobs and live prediction."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fsl_trainer.dashboard import server
from fsl_trainer.dashboard.deletion import delete_artifact
from fsl_trainer.dashboard.model_records import read_records, save_record


class DeletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        patcher = patch.object(server, 'BASE', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.source = self.root / 'data/colors/blue/S1/clip.mp4'
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b'original recording')
        self.run = self.root / 'runs/colors/first'
        self.run.mkdir(parents=True)
        (self.run / 'classes.json').write_text('["blue", "red"]')
        (self.run / 'best.keras').write_bytes(b'checkpoint')
        self.dataset = self.prepared('v1')

    def prepared(self, name=None):
        target = self.root / 'datasets_augmented/colors'
        if name:
            target /= name
        target.mkdir(parents=True, exist_ok=True)
        (target / 'manifest.json').write_text('[]')
        (target / 'preprocessing.json').write_text('{}')
        for split in ('train', 'validation', 'test'):
            (target / split).mkdir()
            (target / split / 'fixture.npy').write_bytes(b'prepared')
        return target

    def test_model_deletion_keeps_sources_datasets_and_frozen_test_evidence(self):
        run_id = 'runs/colors/first'
        save_record(self.root, {'run': run_id, 'decision': 'Preferred', 'notes': 'review'},
                    [{'id': run_id, 'category': 'colors'}])
        evidence = self.root / 'runs/prediction_tests.json'
        evidence.write_text('{"version": 1, "tests": {"old": {"run": "runs/colors/first"}}}')
        original = evidence.read_bytes()
        result = server.Jobs().delete({'kind': 'model', 'path': run_id})
        self.assertFalse(self.run.exists())
        self.assertEqual(server.runs(), [])
        trash = self.root / result['trash']
        self.assertEqual((trash / 'files/first/best.keras').read_bytes(), b'checkpoint')
        self.assertEqual(json.loads((trash / 'model_record.json').read_text())['notes'], 'review')
        self.assertNotIn(run_id, read_records(self.root))
        self.assertEqual(evidence.read_bytes(), original)
        self.assertTrue(self.dataset.exists())
        self.assertEqual(self.source.read_bytes(), b'original recording')
        self.assertEqual(json.loads((trash / 'deletion.json').read_text())['original_path'], run_id)

    def test_named_and_legacy_deletions_preserve_other_datasets(self):
        legacy = self.prepared()
        second = self.prepared('v2')
        result = delete_artifact(self.root, {'kind': 'dataset', 'path': 'datasets_augmented/colors/v1'})
        self.assertFalse(self.dataset.exists())
        self.assertTrue((self.root / result['trash'] / 'files/v1/manifest.json').exists())
        self.assertTrue((legacy / 'manifest.json').exists())
        self.assertTrue((second / 'manifest.json').exists())
        result = delete_artifact(self.root, {'kind': 'dataset', 'path': 'datasets_augmented/colors'})
        self.assertFalse((legacy / 'manifest.json').exists())
        self.assertFalse((legacy / 'train').exists())
        self.assertTrue((second / 'manifest.json').exists())
        self.assertTrue((self.root / result['trash'] / 'files/manifest.json').exists())
        self.assertEqual(self.source.read_bytes(), b'original recording')

    def test_invalid_or_linked_targets_cannot_remove_sources_or_parent_directories(self):
        requests = [
            {'kind': 'model', 'path': 'runs'}, {'kind': 'model', 'path': 'runs/colors'},
            {'kind': 'model', 'path': 'data/colors'}, {'kind': 'dataset', 'path': 'datasets_augmented'},
            {'kind': 'dataset', 'path': '../outside'}, {'kind': 'dataset', 'path': 'data/colors', 'datasets': 'data'},
            {'kind': 'dataset', 'path': 'datasets_augmented/colors/v1/train'},
            {'kind': 'dataset', 'path': 'runs/colors/first', 'datasets': 'runs'},
            {'kind': 'unknown', 'path': 'datasets_augmented/colors/v1'},
            {'kind': 'dataset', 'path': 'datasets_augmented/colors/missing'},
        ]
        for data in requests:
            with self.subTest(data=data), self.assertRaises(ValueError):
                delete_artifact(self.root, data)
        self.assertTrue(self.source.is_file())
        self.assertTrue(self.dataset.is_dir())
        self.assertTrue(self.run.is_dir())
        with patch('pathlib.Path.is_symlink', return_value=True), self.assertRaises(ValueError):
            delete_artifact(self.root, {'kind': 'model', 'path': 'runs/colors/first'})

    def test_deletion_blocks_during_jobs_live_inference_and_shutdown(self):
        jobs = server.Jobs()
        data = {'kind': 'model', 'path': 'runs/colors/first'}
        jobs.process = Mock()
        jobs.process.poll.return_value = None
        with self.assertRaisesRegex(ValueError, 'Stop the running job'):
            jobs.delete(data)
        jobs.process = None
        jobs.live_active = True
        with self.assertRaises(ValueError):
            jobs.delete(data)
        jobs.live_active = False
        jobs.closed = True
        with self.assertRaises(ValueError):
            jobs.delete(data)
        self.assertTrue(self.run.exists())

    def test_failed_move_rolls_back_legacy_files(self):
        from fsl_trainer.dashboard import deletion
        target = self.prepared()
        actual = deletion.os.replace
        calls = 0
        def move(source, dest):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise PermissionError('fixture locked file')
            return actual(source, dest)
        with patch.object(deletion.os, 'replace', side_effect=move), self.assertRaises(PermissionError):
            delete_artifact(self.root, {'kind': 'dataset', 'path': 'datasets_augmented/colors'})
        self.assertTrue((target / 'manifest.json').exists())
        self.assertTrue((target / 'preprocessing.json').exists())
        self.assertTrue((target / 'train').is_dir())

    def test_valid_names_cannot_collide_with_the_trash_receipt(self):
        special = self.run.with_name('deletion.json')
        self.run.rename(special)
        result = delete_artifact(self.root, {'kind': 'model', 'path': 'runs/colors/deletion.json'})
        trash = self.root / result['trash']
        self.assertTrue((trash / 'deletion.json').is_file())
        self.assertTrue((trash / 'files/deletion.json/best.keras').is_file())


if __name__ == '__main__':
    unittest.main()
