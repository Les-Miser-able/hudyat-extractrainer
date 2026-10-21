"""Reports must preserve original counts, missing evidence, and timing scopes."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from fsl_trainer.reporting.paper import (Report, dataset_counts, save_dataset_report,
                                         save_model_report, save_study_report, save_loso_report)
from fsl_trainer.dashboard.model_exports import native_archive
from fsl_trainer.reporting.formatting import percentage
from fsl_trainer.reporting.confusion_matrix import save_confusion_tables


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def read_table(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


class PaperReportTests(unittest.TestCase):
    def test_loso_report_formats_fold_scores_and_requires_complete_rotation_for_aggregates(self):
        folds = []
        for index, signer in enumerate(('S1', 'S2', 'S3')):
            dataset, run = self.root / ('dataset-' + signer), self.root / ('run-' + signer)
            write_json(dataset / 'preprocessing.json', {'evaluation_protocol': {
                'strategy': 'leave_one_signer_out', 'test_signer': signer,
                'validation_signer': ('S1', 'S2', 'S3')[(index + 1) % 3], 'all_signers': ['S1', 'S2', 'S3']}})
            write_json(run / 'classes.json', ['A', 'B'])
            value = (.7, .8, .9)[index]
            write_json(run / 'test_metrics.json', {'accuracy': value, 'macro_precision': value,
                'macro_recall': value, 'macro_f1': value, 'lowest_sign_recall': value,
                'per_class': {c: {'precision': value, 'recall': value, 'f1': value, 'support': 5} for c in ('A', 'B')}})
            folds.append((dataset, run))
        output = self.root / 'loso'
        save_loso_report(folds, output)
        table = read_table(output / 'loso_folds.csv')
        self.assertEqual([r['Accuracy (%)'] for r in table], ['70.00%', '80.00%', '90.00%'])
        summary = read_table(output / 'loso_summary.csv')
        self.assertEqual(summary[0]['Mean (%)'], '80.00%')
        self.assertEqual(summary[0]['Sample SD (percentage points)'], '10.00%')
        for name in ('loso_folds', 'loso_summary', 'loso_variation'):
            self.assertTrue((output / (name + '.png')).is_file())
        save_loso_report(folds[:-1], output)
        self.assertFalse((output / 'loso_summary.csv').exists())
        self.assertFalse((output / 'loso_variation.png').exists())
        save_loso_report(folds[:-1], self.root / 'partial-loso')
        self.assertFalse((self.root / 'partial-loso/loso_summary.csv').exists())
        self.assertIn('incomplete', (self.root / 'partial-loso/index.html').read_text())

    def test_percentage_units_are_explicit_and_do_not_change_other_values(self):
        self.assertEqual(percentage(.987), '98.70%')
        self.assertEqual(percentage(0), '0.00%')
        self.assertEqual(percentage(1), '100.00%')
        self.assertEqual(percentage(75, scale=1), '75.00%')
        for value in (None, True, float('nan'), float('inf')):
            self.assertIsNone(percentage(value))
        report = Report(self.root / 'formatting', 'Formatting', 'Saved measurements')
        report.settings({'learning_rate': .001, 'jitter_std': .03, 'class_weights': False,
                         'validation_fraction': .15, 'extraction': {'detection_confidence': .5},
                         'pose_augmentation': {'mirror_probability': .5}})
        settings = {r['Setting']: r['Value'] for r in read_table(report.output / 'settings.csv')}
        self.assertEqual(settings['learning_rate'], '0.001')
        self.assertEqual(settings['jitter_std'], '0.03')
        self.assertEqual(settings['class_weights'], 'False')
        self.assertEqual(settings['validation_fraction (%)'], '15.00%')
        self.assertEqual(settings['extraction / detection_confidence (%)'], '50.00%')
        self.assertEqual(settings['pose_augmentation / mirror_probability (%)'], '50.00%')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'datasets_augmented' / 'colors' / 'example'
        self.run = self.root / 'runs' / 'colors' / 'seed_17'
        self.manifest = []
        for split in ('train', 'validation', 'test'):
            for label in ('A', 'B'):
                source = f'{label}/{split}.mp4'
                row = {'source': source, 'variant_of': source, 'label': label,
                       'group': source, 'split': split, 'static': label == 'A',
                       'synthetic': False, 'path': f'{split}/{label}/clean.npy'}
                self.manifest.append(row)
                if split == 'train':
                    self.manifest.append({**row, 'synthetic': True, 'path': f'{split}/{label}/extra.npy'})
        write_json(self.dataset / 'manifest.json', self.manifest)
        write_json(self.dataset / 'preprocessing.json', {'seed': 42, 'frames': 32, 'features': 128})
        write_json(self.run / 'classes.json', ['A', 'B'])
        write_json(self.run / 'preprocessing.json', {'frames': 32, 'features': 128})
        write_json(self.run / 'training_config.json', {'seed': 17, 'dataset': str(self.dataset)})

    def test_standalone_confusion_tables_use_percent_signs_only_for_normalized_values(self):
        save_confusion_tables({'classes': ['A', 'B'],
                               'confusion_matrix_rows_true_columns_predicted': [[3, 1], [0, 0]]}, self.run)
        self.assertEqual(read_table(self.run / 'confusion_matrix.csv')[0]['A'], '3')
        normalized = read_table(self.run / 'confusion_matrix_normalized.csv')
        self.assertEqual(normalized[0]['A'], '75.00%')
        self.assertEqual(normalized[0]['B'], '25.00%')
        self.assertEqual(normalized[1]['A'], '0.00%')

    def test_preparation_counts_and_real_figures_agree_with_manifest(self):
        accepted = [{'source': row['source'], 'kind': 'video', 'frames': 60,
                     'detected_frames': 45, 'fps': 30} for row in self.manifest if not row['synthetic']]
        write_json(self.dataset / 'extraction_report.json',
                   {'accepted': accepted, 'rejected': [{'reason': 'No hand'}], 'ignored': ['notes.txt']})
        path = save_dataset_report(self.dataset)
        output = path.parent
        overview = {r['Measurement']: r['Count'] for r in read_table(output / 'dataset_overview.csv')}
        self.assertEqual(overview['Original sources'], '6')
        self.assertEqual(overview['Total output sequences'], '8')
        splits = read_table(output / 'split_counts.csv')
        self.assertEqual([r['Total output sequences'] for r in splits], ['4', '2', '2'])
        self.assertEqual([r['Original sources'] for r in splits], ['2', '2', '2'])
        media = read_table(output / 'recording_quality.csv')
        self.assertEqual(media[0]['Mean video duration (s)'], '2.0')
        self.assertEqual(media[0]['Detected / decoded frames (%)'], '75.00%')
        manifest = json.loads((output / 'report_manifest.json').read_text())
        for filename in manifest['artifacts']:
            self.assertTrue((output / filename).is_file(), filename)
        self.assertTrue((output / 'split_counts.png').read_bytes().startswith(b'\x89PNG'))
        self.assertIn('<svg', (output / 'split_counts.svg').read_text())
        chart = output / 'split_counts.png'
        original = (chart.read_bytes(), chart.stat().st_mtime_ns)
        with patch.object(Report, 'figure', side_effect=AssertionError('Existing figures should be reused')):
            save_dataset_report(self.dataset, tables_only=True)
        self.assertEqual((chart.read_bytes(), chart.stat().st_mtime_ns), original)

    def test_source_groups_and_synthetic_holdout_are_reported_without_changing_evidence(self):
        rows = [dict(row) for row in self.manifest]
        rows[-1]['group'] = rows[0]['group']
        rows[-1]['synthetic'] = True
        original = (self.dataset / 'manifest.json').read_bytes()
        with patch.object(Report, 'chart'):
            path = save_dataset_report(self.dataset, manifest=rows)
        audit = {r['Identity']: r['Identities spanning multiple splits']
                 for r in read_table(path.parent / 'split_isolation.csv')}
        self.assertEqual(audit['group'], '1')
        self.assertIn('synthetic validation/test', path.read_text())
        self.assertEqual((self.dataset / 'manifest.json').read_bytes(), original)
        counts = dataset_counts(rows)
        self.assertEqual(counts['all', 'all']['originals'], 6)
        del rows[0]['synthetic']
        self.assertIsNone(dataset_counts(rows)['all', 'all']['clean'])

    def test_model_report_keeps_unknown_scores_and_incomplete_live_coverage_unknown(self):
        write_json(self.run / 'test_metrics.json', {'accuracy': .8, 'loss': .3,
                   'confusion_matrix_rows_true_columns_predicted': [[3, 1], [0, 2]]})
        write_json(self.run / 'history.json', {'accuracy': [.7, .8], 'val_accuracy': [.6, .75],
                                              'loss': [.5, .3], 'val_loss': [.6, .4]})
        write_json(self.run / 'epoch_times.json', [2, 3])
        measured = {'accuracy': 1., 'macro_precision': None, 'macro_recall': None, 'macro_f1': None,
                    'coverage_complete': False, 'per_class': {
                        'A': {'precision': 1., 'recall': 1., 'f1': 1., 'support': 1},
                        'B': {'precision': 0., 'recall': 0., 'f1': 0., 'support': 0}},
                    'timing': {'scope': 'trainer_camera_first_hand_to_prediction_received',
                               'median_ms': 123, 'p95_ms': 123, 'max_ms': 123}}
        test = {'automated': True, 'run': 'runs/colors/seed_17', 'name': '<script>bad()</script>',
                'measurements': measured, 'trials': []}
        write_json(self.root / 'runs' / 'prediction_tests.json', {'tests': {'saved': test}})
        path = save_model_report(self.run)
        metrics = {r['Metric']: r['Value'] for r in read_table(path.parent / 'metrics.csv')}
        self.assertEqual(metrics['Accuracy'], '80.00%')
        self.assertEqual(metrics['Loss'], '0.3')
        self.assertEqual(metrics['Macro precision'], '')
        live = {r['Metric']: r['Value'] for r in read_table(path.parent / 'live_test_1_metrics.csv')}
        self.assertEqual(live['Accuracy'], '100.00%')
        self.assertEqual(live['Macro recall'], '')
        self.assertIn('trainer_camera_first_hand_to_prediction_received', path.read_text())
        self.assertNotIn('<script>bad()', path.read_text())
        self.assertTrue((path.parent / 'learning_accuracy.png').is_file())
        summary = {r['Measurement']: r['Value'] for r in read_table(path.parent / 'training_summary.csv')}
        self.assertEqual(summary['Lowest recorded validation loss epoch'], '2')
        self.assertEqual(summary['Total recorded epoch time (s)'], '5')
        history = read_table(path.parent / 'learning_history.csv')
        self.assertEqual(history[0]['accuracy (%)'], '70.00%')
        self.assertEqual(history[0]['loss'], '0.5')
        checks = {r['Check']: r['Threshold'] for r in read_table(path.parent / 'benchmark_checks.csv')}
        self.assertEqual(checks['accuracy'], '85.00%')
        self.assertEqual(checks['response_time_ms'], '1000')
        normalized = read_table(path.parent / 'confusion_matrix_normalized.csv')
        self.assertEqual(normalized[0]['A'], '75.00%')
        self.assertEqual(normalized[0]['B'], '25.00%')
        self.assertEqual(read_table(path.parent / 'confusion_matrix.csv')[0]['A'], '3')
        self.assertEqual(json.loads((self.run / 'test_metrics.json').read_text())['accuracy'], .8)

    def test_study_comparison_preserves_missing_legacy_data_and_exports_figures_with_models(self):
        write_json(self.run / 'test_metrics.json', {'accuracy': .9})
        write_json(self.root / 'datasets_augmented' / 'colors' / 'broken' / 'manifest.json', [])
        (self.run / 'best.keras').write_bytes(b'checkpoint')
        (self.run / 'test_predictions.csv').write_text('true,predicted\nA,B\n')
        with patch.object(Report, 'chart'), patch.object(Report, 'matrix'):
            page = save_study_report(self.root)
            save_model_report(self.run)
        models = read_table(page.parent / 'models.csv')
        self.assertEqual(models[0]['Training seed'], '17')
        self.assertEqual(models[0]['Accuracy (%)'], '90.00%')
        self.assertEqual(models[0]['Macro recall (%)'], '')
        unavailable = read_table(page.parent / 'unavailable.csv')
        self.assertEqual(len(unavailable), 1)
        self.assertIn('broken', unavailable[0]['Artifact'])
        run_id = 'runs/colors/seed_17'
        (self.run / 'paper_report' / 'obsolete.csv').write_text('old,values\n')
        known = [{'id': run_id, 'category': 'colors', 'name': 'seed_17', 'class_labels': ['A', 'B']}]
        with native_archive(self.root, run_id, known) as (handle, _), ZipFile(handle) as archive:
            self.assertIn('paper_report/index.html', archive.namelist())
            self.assertIn('paper_report/metrics.csv', archive.namelist())
            self.assertIn('test_predictions.csv', archive.namelist())
            self.assertNotIn('paper_report/obsolete.csv', archive.namelist())


if __name__ == '__main__':
    unittest.main()
