"""Exercise signer preparation, provenance, cache auditing, and command routing."""
import argparse
import contextlib
import copy
import csv
import io
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fsl_trainer.data.signer_splits import split_signers, options, validate_protocol
from fsl_trainer.training import trainer


class SignerSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'data/colors'
        rng = np.random.default_rng(71)
        for label in ('blue', 'red'):
            for signer in ('S1', 'S2', 'S3', 'S4'):
                folder = self.source / label / signer
                folder.mkdir(parents=True)
                for index, frames in enumerate((1, 6)):
                    data = np.zeros((frames, 128), dtype=np.float32)
                    data[:, 63:126] = rng.uniform(.1, .8, (frames, 63))
                    data[:, 127] = 1
                    np.save(folder / f'{index}.npy', data)
        self.args = argparse.Namespace(source=str(self.source), output=str(self.root / 'prepared'),
            groups=None, validation=.15, test=.15, seed=42, static_variants=4, jitter=.0075,
            split_strategy='loso', test_signer='S1', validation_signer=None,
            video_extra_samples=1, video_time_warp=.05, video_mirror_hands=True)

    def prepare(self, reports=False):
        from fsl_trainer.reporting.paper import save_dataset_report
        with patch('fsl_trainer.reporting.paper.save_dataset_report',
                   side_effect=save_dataset_report if reports else lambda *a, **kw: None), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            trainer.prepare(self.args)
            return trainer.inspect_dataset(self.args.output)[0]

    def test_preparation_keeps_signers_disjoint_and_evaluation_clean_with_tables_and_charts(self):
        rows = self.prepare(reports=True)
        root = Path(self.args.output)
        metadata = json.loads((root / 'preprocessing.json').read_text())
        protocol = metadata['evaluation_protocol']
        self.assertEqual(protocol['test_signer'], 'S1')
        self.assertEqual(protocol['validation_signer'], 'S2')
        self.assertEqual(protocol['train_signers'], ['S3', 'S4'])
        self.assertIsNone(metadata['test_fraction'])
        for row in rows:
            expected = {'S1': 'test', 'S2': 'validation', 'S3': 'train', 'S4': 'train'}[row['signer']]
            self.assertEqual(row['split'], expected)
            self.assertEqual(row['source'].split('/')[1], row['signer'])
            if expected != 'train':
                self.assertFalse(row['synthetic'])
                self.assertEqual(row['variant'], 0)
        self.assertEqual(sum(r['split'] == 'test' for r in rows), 4)
        self.assertEqual(sum(r['split'] == 'validation' for r in rows), 4)
        self.assertTrue(any(r['synthetic'] and r['static'] for r in rows))
        self.assertTrue(any(r.get('augmentation', {}).get('kind') == 'hand_mirror' for r in rows))
        for stem in ('signer_splits', 'signer_coverage'):
            for suffix in ('.csv', '.png', '.svg'):
                self.assertTrue((root / 'paper_report' / (stem + suffix)).is_file())
        with (root / 'paper_report/signer_splits.csv').open(encoding='utf-8-sig') as handle:
            table = list(csv.DictReader(handle))
        self.assertEqual(sum(int(r['Synthetic outputs']) for r in table if r['Split'] != 'train'), 0)
        self.assertEqual({r['Signer'] for r in table}, {'S1', 'S2', 'S3', 'S4'})
        from fsl_trainer.training.cache import open_cache
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            first = open_cache(root)
            cached = open_cache(root)
        self.assertEqual(first[0], cached[0])
        validate_protocol(cached[1], protocol)

    def test_all_test_signers_can_be_rotated_without_overlap(self):
        for signer in ('S1', 'S2', 'S3', 'S4'):
            self.args.test_signer = signer
            self.args.output = str(self.root / signer)
            rows = self.prepare()
            self.assertEqual({r['signer'] for r in rows if r['split'] == 'test'}, {signer})
        self.args.output = str(self.root / 'explicit')
        self.args.test_signer, self.args.validation_signer = 'S2', 'S4'
        rows = self.prepare()
        self.assertEqual({r['signer'] for r in rows if r['split'] == 'validation'}, {'S4'})

    def test_rejects_missing_signers_labels_and_ambiguous_groups_before_output(self):
        records = [{'source': f'{label}/{s}/clip.npy', 'label': label, 'group': f'{label}/{s}', 'signer': s}
                   for label in ('A', 'B') for s in ('S1', 'S2', 'S3')]
        for test, val in [('missing', None), ('S1', 'S1'), ('S1', 'unknown')]:
            with self.assertRaises(ValueError):
                split_signers(records, test, val)
        with self.assertRaisesRegex(ValueError, 'at least three'):
            split_signers(records[:2], 'S1')
        with self.assertRaisesRegex(ValueError, 'missing gestures'):
            split_signers([r for r in records if r['source'] != 'B/S1/clip.npy'], 'S1')
        records[1]['group'] = records[0]['group']
        with self.assertRaisesRegex(ValueError, 'group crosses'):
            split_signers(records, 'S1')
        for request in [('other', None, None), ('loso', None, None), ('loso', '../S1', None),
                        ('grouped', 'S1', None), ('loso', False, None)]:
            with self.assertRaises(ValueError):
                options(*request)
        (self.source / 'blue/direct.npy').write_bytes((self.source / 'blue/S1/0.npy').read_bytes())
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(Path(self.args.output).exists())

    def test_training_rejects_corrupt_protocol_before_tensorflow_for_both_loaders(self):
        rows = self.prepare()
        root = Path(self.args.output)
        protocol = json.loads((root / 'preprocessing.json').read_text())['evaluation_protocol']
        for mutation in ('split', 'signer', 'synthetic', 'missing_metadata'):
            bad = copy.deepcopy(rows)
            if mutation == 'split':
                bad[0]['split'] = 'train'
            elif mutation == 'signer':
                bad[0]['signer'] = 'S99'
            elif mutation == 'synthetic':
                bad[0]['synthetic'] = True
            with self.assertRaises(ValueError):
                validate_protocol(bad, None if mutation == 'missing_metadata' else protocol)
        bad = copy.deepcopy(rows)
        bad[0]['split'] = 'train'
        (root / 'manifest.json').write_text(json.dumps(bad))
        for loader in ('files', 'packed'):
            args = argparse.Namespace(dataset=str(root), output=str(self.root / ('run-' + loader)), loader=loader)
            with patch.object(trainer, 'tensorflow') as tf, \
                    patch('fsl_trainer.training.cache.open_cache', return_value=(None, bad, ['blue', 'red'], {})), \
                    self.assertRaises(ValueError):
                trainer.train(args)
            tf.assert_not_called()

    def test_raw_extraction_retains_signer_identity_and_original_paths(self):
        from fsl_trainer.data import extraction
        raw = self.root / 'raw'
        for source in self.source.rglob('*.npy'):
            path = raw / source.relative_to(self.source).with_suffix('.mp4')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(source))
        asset = self.root / 'hand.task'
        asset.write_bytes(b'fixture')
        class Extractor:
            def __init__(self, **kwargs):
                self.model, self.mp = asset, SimpleNamespace(__version__='fixture')
                self.confidence, self.swap_hands = .5, False
            def extract(self, path):
                data = np.load(path.read_text())
                return data, {'kind': 'video', 'frames': len(data), 'detected_frames': len(data), 'fps': 30}
            def close(self):
                pass
        self.args.source = str(raw)
        with patch.object(extraction, 'Extractor', Extractor):
            rows = self.prepare()
        self.assertTrue(all(r['source'].endswith('.mp4') for r in rows))
        self.assertTrue(all(r['signer'] == r['source'].split('/')[1] for r in rows))
        self.assertTrue((Path(self.args.output) / 'extracted_sources.json').is_file())

    def test_dashboard_and_both_clis_forward_the_same_signer_protocol(self):
        from fsl_trainer.dashboard import server
        from fsl_trainer.data import categories
        with patch.object(server, 'BASE', self.root):
            command = server.build_command({'action': 'prepare', 'category': 'colors', 'dataset_name': 'loso-S2',
                                            'split_strategy': 'loso', 'test_signer': 'S2', 'validation_signer': 'S4'})
        self.assertEqual(command[command.index('--test-signer') + 1], 'S2')
        self.assertEqual(command[command.index('--validation-signer') + 1], 'S4')
        argv = ['categories.py', 'prepare', '--category', 'colors', '--data', str(self.source.parent),
                '--datasets', str(self.root / 'datasets'), '--split-strategy', 'loso',
                '--test-signer', 'S2', '--validation-signer', 'S4']
        with patch('sys.argv', argv), patch.object(categories, 'execute') as execute:
            categories.main()
        jobs, _ = categories.build_jobs(execute.call_args.args[0])
        self.assertEqual(jobs[0][2][jobs[0][2].index('--test-signer') + 1], 'S2')
        argv = ['trainer.py', 'prepare', '--source', str(self.source), '--output', str(self.root / 'cli'),
                '--split-strategy', 'loso', '--test-signer', 'S2', '--validation-signer', 'S4']
        with patch('sys.argv', argv), patch.object(trainer, 'prepare') as prepare:
            trainer.main()
        self.assertEqual(prepare.call_args.args[0].test_signer, 'S2')
        self.assertEqual(prepare.call_args.args[0].validation_signer, 'S4')

    def test_all_fold_jobs_prepare_and_train_fresh_models_and_reject_incomplete_or_mixed_studies(self):
        from fsl_trainer.data import categories
        from fsl_trainer.data.signer_splits import fold_datasets
        argv = ['categories.py', 'prepare', '--category', 'colors', '--data', str(self.source.parent),
                '--datasets', str(self.root / 'datasets'), '--dataset-name', 'study', '--loso-all']
        with patch('sys.argv', argv), patch.object(categories, 'execute') as execute:
            categories.main()
        prepare_args = execute.call_args.args[0]
        jobs, _ = categories.build_jobs(prepare_args)
        self.assertEqual(len(jobs), 4)
        self.assertEqual({command[command.index('--test-signer') + 1] for _, _, command in jobs},
                         {'S1', 'S2', 'S3', 'S4'})
        for _, target, command in jobs:
            self.args.output = str(target)
            self.args.test_signer = command[command.index('--test-signer') + 1]
            self.prepare()
        paths = fold_datasets(self.root / 'datasets/colors', 'study')
        self.assertEqual(len(paths), 4)
        argv = ['categories.py', 'train', '--category', 'colors', '--datasets', str(self.root / 'datasets'),
                '--runs', str(self.root / 'runs'), '--dataset-name', 'study', '--run', 'experiment', '--loso-all']
        with patch('sys.argv', argv), patch.object(categories, 'execute') as execute:
            categories.main()
        train_args = execute.call_args.args[0]
        training, _ = categories.build_jobs(train_args)
        self.assertEqual({t.name for _, t, _ in training}, {f'experiment-loso-S{i}' for i in range(1, 5)})
        self.assertTrue(all('--resume-from' not in cmd for _, _, cmd in training))
        with patch.object(categories.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run, \
                patch('fsl_trainer.reporting.paper.save_loso_report') as report, \
                contextlib.redirect_stdout(io.StringIO()):
            categories.execute(train_args)
        self.assertEqual(run.call_count, 4)
        self.assertEqual(len(report.call_args.args[0]), 4)
        for _, target, _ in training:
            target.mkdir(parents=True)
            (target / 'classes.json').write_text('["blue", "red"]')
        train_args.report_only = True
        with patch.object(categories.subprocess, 'run') as run, \
                patch('fsl_trainer.reporting.paper.save_loso_report') as report, \
                contextlib.redirect_stdout(io.StringIO()):
            categories.execute(train_args)
        run.assert_not_called()
        self.assertEqual(len(report.call_args.args[0]), 4)
        train_args.report_only = False
        # Remove the stand-in outputs before checking the other preflight errors.
        for _, target, _ in training:
            (target / 'classes.json').unlink()
            target.rmdir()
        train_args.resume_from = 'other-fold/best.keras'
        with self.assertRaisesRegex(ValueError, 'cannot be shared'):
            categories.build_jobs(train_args)
        train_args.resume_from = None
        metadata_path = paths[0] / 'preprocessing.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['seed'] = 55
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'different sources or preprocessing'):
            categories.build_jobs(train_args)
        metadata['seed'] = 42
        metadata_path.write_text(json.dumps(metadata))
        paths[-1].rename(paths[-1].with_name('other-study'))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            categories.build_jobs(train_args)

    @unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow is not installed')
    def test_real_training_saves_held_out_signer_and_evaluates_only_clean_test_sources(self):
        self.args.static_variants = 1
        self.args.video_extra_samples, self.args.video_mirror_hands = 0, False
        manifest = self.prepare()
        output = self.root / 'runs/colors/loso-S1'
        args = argparse.Namespace(dataset=self.args.output, output=str(output), seed=42,
            epochs=1, batch_size=8, learning_rate=.001, patience=1, class_weights=False, loader='packed')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            trainer.train(args)
        config = json.loads((output / 'training_config.json').read_text())
        self.assertEqual(config['evaluation_protocol']['test_signer'], 'S1')
        self.assertEqual(config['evaluation_protocol']['validation_signer'], 'S2')
        self.assertEqual(json.loads((output / 'split_manifest.json').read_text()), manifest)
        metrics = json.loads((output / 'test_metrics.json').read_text())
        self.assertEqual(sum(v['support'] for v in metrics['per_class'].values()), 4)
        self.assertTrue((output / 'best.keras').is_file())
        self.assertTrue((output / 'paper_report/signer_splits.csv').is_file())
        self.assertTrue((output / 'paper_report/signer_splits.png').is_file())


if __name__ == '__main__':
    unittest.main()
