"""Portable native backups and HUDYAT imports without training or a database."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from fsl_trainer.dashboard.model_exports import (native_archive, write_hudyat_bundle, export_options,
                                                export_folder_name, hudyat_export_path)
from fsl_trainer.dashboard.model_records import save_record
from integrations.hudyat.install_hudyat import import_bundle
from integrations.hudyat.runtime_support import runtime_changes, apply_changes

ROOT = Path(__file__).resolve().parents[2]


class ModelExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_id = 'runs/alphabet/test model'
        self.run = self.root / self.run_id
        self.run.mkdir(parents=True)
        self.hand = self.root / 'assets/hand_landmarker.task'
        self.hand.parent.mkdir()
        self.hand.write_bytes(b'hand detector')
        self.pre = {'normalization': 'hand-relative-sequence-v1', 'extraction': {
            'model_sha256': hashlib.sha256(self.hand.read_bytes()).hexdigest(), 'detection_confidence': .5}}
        (self.run / 'best.keras').write_bytes(b'trained weights')
        (self.run / 'classes.json').write_text('["A", "B"]')
        (self.run / 'preprocessing.json').write_text(json.dumps(self.pre))
        (self.run / 'test_metrics.json').write_text('{"accuracy":0.9}')
        (self.run / 'private-source.npy').write_bytes(b'not part of the model')
        self.known = [{'id': self.run_id, 'name': self.run.name, 'category': 'alphabet', 'class_labels': ['A', 'B']}]
        self.output = self.root / 'exports/hudyat/bundle.zip'
        shutil.copytree(ROOT / 'integrations/hudyat', self.root / 'integrations/hudyat',
                        ignore=shutil.ignore_patterns('__pycache__'))

    def fake_converter(self, script, command):
        site = Path(command[command.index('--site') + 1])
        version = 'aabbccddeeff0011'
        model = site / 'models/alphabet' / version
        model.mkdir(parents=True)
        hand = site / 'shared' / (self.pre['extraction']['model_sha256'] + '.task')
        hand.parent.mkdir()
        hand.write_bytes(self.hand.read_bytes())
        weights = b'browser weights'
        meta = json.dumps({'format': 'fsl-conv1d-bilstm-v1', 'inputShape': [32, 128],
                           'category': 'alphabet', 'version': version, 'classes': ['A', 'B'],
                           'preprocessing': self.pre, 'weightSha256': hashlib.sha256(weights).hexdigest()}).encode()
        (model / 'model.json').write_bytes(meta)
        (model / 'weights.bin').write_bytes(weights)
        (model / 'parity.json').write_text('{}')
        (model / 'camera-preprocessing-parity.json').write_text('{}')
        entry = {'id': 'alphabet', 'name': 'Alphabet', 'available': True, 'version': version,
                 'modelUrl': './models/alphabet/' + version + '/model.json',
                 'weightsUrl': './models/alphabet/' + version + '/weights.bin',
                 'handUrl': './shared/' + hand.name, 'modelSha256': hashlib.sha256(meta).hexdigest(),
                 'weightsSha256': hashlib.sha256(weights).hexdigest(), 'handSha256': self.pre['extraction']['model_sha256']}
        (site / 'catalog.json').write_text(json.dumps({'categories': [entry, {'id': 'colors', 'available': False}]}))

    def bundle(self):
        with patch('fsl_trainer.dashboard.model_exports._convert', side_effect=self.fake_converter) as convert:
            write_hudyat_bundle(self.root, self.run_id, self.known, self.output)
        return convert

    def hudyat(self):
        project = self.root / 'HUDYAT'
        runtime = project / 'client/src/features/fsl-recognition/category-runtime.ts'
        runtime.parent.mkdir(parents=True)
        shutil.copy2(ROOT / 'tests/fixtures/hudyat-category-runtime.ts', runtime)
        (project / 'client/package.json').write_text(json.dumps({'scripts': {'test': 'node --test tests/old.test.mjs'}}))
        assets = project / 'server/model-assets'
        assets.mkdir(parents=True)
        (project / 'server/package.json').write_text(json.dumps({'scripts': {'models:register': 'tsx registerModels.ts'}}))
        (assets / 'catalog.json').write_text(json.dumps({'categories': [
            {'id': 'colors', 'name': 'Colors', 'available': True, 'version': 'existing-colors'},
            {'id': 'alphabet', 'name': 'Alphabet', 'available': True, 'version': 'old-alphabet'}]}))
        (assets / 'existing.bin').write_bytes(b'keep other categories')
        return project

    def extracted(self):
        self.bundle()
        folder = self.root / 'extracted'
        with ZipFile(self.output) as archive:
            archive.extractall(folder)
        return folder

    def test_native_zip_keeps_trained_files_and_records_without_source_data(self):
        save_record(self.root, {'run': self.run_id, 'notes': 'Preferred model details'}, self.known)
        before = {p.name: p.read_bytes() for p in self.run.iterdir()}
        with native_archive(self.root, self.run_id, self.known) as (handle, filename):
            self.assertEqual(filename, 'alphabet-test-model-model.zip')
            with ZipFile(handle) as archive:
                self.assertEqual(archive.read('best.keras'), b'trained weights')
                self.assertEqual(json.loads(archive.read('model-record.json'))['model']['record']['notes'], 'Preferred model details')
                self.assertNotIn('private-source.npy', archive.namelist())
                self.assertTrue({'classes.json', 'preprocessing.json', 'README.txt'} <= set(archive.namelist()))
        self.assertTrue(handle.closed)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.run.iterdir()})

    def test_export_folder_names_are_validated_before_creating_outputs(self):
        for name in ('Alphabet v2', 'alphabet-v2', '_alphabet', 'release.2', 'x' * 100):
            with self.subTest(name=name):
                self.assertEqual(export_folder_name(name), name)
                path = hudyat_export_path(self.root, '1122334455667788', name)
                self.assertEqual(path, self.root / 'exports/hudyat' / name / (name + '.zip'))
        for name in ('', '.', '..', '.hidden', ' leading', 'trailing ', 'trailing.',
                     '../outside', 'a/b', 'a\\b', 'a:b', 'a"b', 'a\nb', 'x' * 101,
                     'CON', 'con.txt', 'CON .txt', 'nul', 'COM1', 'lpt9.txt', None, True, []):
            with self.subTest(name=name), self.assertRaises(ValueError):
                export_folder_name(name)
        self.assertFalse((self.root / 'exports').exists())

    def test_missing_artifacts_unknown_runs_and_landmark_only_browser_exports_are_rejected(self):
        with self.assertRaises(ValueError):
            with native_archive(self.root, '../outside', self.known):
                pass
        (self.run / 'preprocessing.json').write_text('{}')
        self.assertTrue(export_options(self.run)['download'])
        self.assertFalse(export_options(self.run)['hudyat'])
        with self.assertRaisesRegex(ValueError, 'extraction metadata'):
            write_hudyat_bundle(self.root, self.run_id, self.known, self.output)
        (self.run / 'classes.json').unlink()
        with self.assertRaisesRegex(ValueError, 'best.keras'):
            with native_archive(self.root, self.run_id, self.known):
                pass

    def test_hudyat_bundle_uses_the_training_detector_and_contains_one_portable_release(self):
        converter = self.bundle()
        command = converter.call_args.args[1]
        self.assertEqual(command[command.index('--hand-model') + 1], str(self.hand.resolve()))
        with ZipFile(self.output) as archive:
            catalog = json.loads(archive.read('model-assets/catalog.json'))
            self.assertEqual([e['id'] for e in catalog['categories']], ['alphabet'])
            self.assertIn('install_hudyat.py', archive.namelist())
            self.assertIn('model-preprocessing.js', archive.namelist())
            self.assertIn('HUDYAT_IMPORT.md', archive.namelist())
            self.assertNotIn('best.keras', archive.namelist())
        self.assertEqual(list(self.output.parent.glob('.hudyat-export-*')), [])

    def test_failed_conversion_wrong_detector_and_existing_output_preserve_work(self):
        with patch('fsl_trainer.dashboard.model_exports._convert', side_effect=SystemExit(2)):
            with self.assertRaises(SystemExit):
                write_hudyat_bundle(self.root, self.run_id, self.known, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob('.hudyat-export-*')), [])
        self.hand.write_bytes(b'wrong detector')
        with self.assertRaisesRegex(ValueError, 'hand detector'):
            write_hudyat_bundle(self.root, self.run_id, self.known, self.output)
        self.hand.write_bytes(b'hand detector')
        self.output.write_bytes(b'existing bundle')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            write_hudyat_bundle(self.root, self.run_id, self.known, self.output)
        self.assertEqual(self.output.read_bytes(), b'existing bundle')

    def test_import_merges_one_category_and_installs_idempotent_camera_support(self):
        self.output = hudyat_export_path(self.root, '1122334455667788', 'Alphabet v2')
        folder, project = self.extracted(), self.hudyat()
        self.assertTrue(self.output.is_file())
        import_bundle(folder, project)
        assets = project / 'server/model-assets'
        entries = {e['id']: e for e in json.loads((assets / 'catalog.json').read_text())['categories']}
        self.assertEqual(entries['alphabet']['version'], 'aabbccddeeff0011')
        self.assertEqual(entries['colors']['version'], 'existing-colors')
        self.assertEqual((assets / 'existing.bin').read_bytes(), b'keep other categories')
        source = (project / 'client/src/features/fsl-recognition/category-runtime.ts').read_text()
        self.assertIn('resample(preprocessSequence(frames,frameWidth,frameHeight,metadata.preprocessing))', source)
        self.assertIn('frameWidth=video.videoWidth', source)
        before = {p.relative_to(project): p.read_bytes() for p in project.rglob('*') if p.is_file()}
        import_bundle(folder, project)
        self.assertEqual(before, {p.relative_to(project): p.read_bytes() for p in project.rglob('*') if p.is_file()})

    def test_corrupt_or_escaping_bundle_is_rejected_before_modifying_hudyat(self):
        folder, project = self.extracted(), self.hudyat()
        before = {p.relative_to(project): p.read_bytes() for p in project.rglob('*') if p.is_file()}
        weights = next((folder / 'model-assets/models').rglob('weights.bin'))
        weights.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            import_bundle(folder, project)
        self.assertEqual(before, {p.relative_to(project): p.read_bytes() for p in project.rglob('*') if p.is_file()})
        catalog = folder / 'model-assets/catalog.json'
        data = json.loads(catalog.read_text())
        data['categories'][0]['modelUrl'] = '../../outside/model.json'
        catalog.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'artifact path'):
            import_bundle(folder, project)
        self.assertEqual(before, {p.relative_to(project): p.read_bytes() for p in project.rglob('*') if p.is_file()})

    def test_unrecognized_runtime_is_rejected_before_any_camera_file_is_changed(self):
        project = self.hudyat()
        runtime = project / 'client/src/features/fsl-recognition/category-runtime.ts'
        runtime.write_text('different application')
        with self.assertRaisesRegex(ValueError, 'camera code has changed'):
            apply_changes(runtime_changes(project))
        self.assertEqual(runtime.read_text(), 'different application')
        self.assertFalse((runtime.parent / 'model-preprocessing.js').exists())

    def test_failed_helper_install_leaves_the_existing_camera_imports_intact(self):
        project = self.hudyat()
        runtime = project / 'client/src/features/fsl-recognition/category-runtime.ts'
        before = runtime.read_bytes()
        with patch('integrations.hudyat.runtime_support.atomic_write', side_effect=PermissionError('locked')):
            with self.assertRaises(PermissionError):
                apply_changes(runtime_changes(project))
        self.assertEqual(runtime.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
