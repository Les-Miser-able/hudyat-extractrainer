"""Metrics, benchmark boundaries, legacy reports, and real single-sample timing."""
import argparse
import contextlib
import importlib.util
import os
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from fsl_trainer.training.trainer import evaluation
from fsl_trainer.training.benchmark import measure_latency, benchmark_run
from fsl_trainer.reporting.benchmark import benchmark_summary, benchmark_csv, read_benchmark, save_benchmark


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.classes = ['blue', 'green', 'red', 'yellow', 'black', 'white']
        self.metrics = {'accuracy': .85, 'macro_f1': .85, 'per_class': {
            label: {'precision': .85, 'recall': .85, 'f1': .85, 'support': 10}
            for label in self.classes}}
        self.timing = {'scope': 'hudyat_camera_to_result', 'median_ms': 999.99, 'samples': 30, 'tested_signs': self.classes, 'timeouts': 0}

    def test_macro_scores_and_lowest_recall_use_all_classes_equally(self):
        report = evaluation(np.array([0, 1, 1]), np.array([[1, 0], [1, 0], [0, 1]]), ['A', 'J'])
        self.assertAlmostEqual(report['macro_precision'], .75)
        self.assertAlmostEqual(report['macro_recall'], .75)
        self.assertAlmostEqual(report['macro_f1'], 2 / 3)
        self.assertEqual(report['lowest_sign_recall'], .5)
        self.assertEqual(report['lowest_recall_signs'], ['J'])

    def test_score_threshold_is_inclusive_but_latency_is_strict(self):
        report = benchmark_summary(self.metrics, self.classes, 'colors', self.timing)
        self.assertTrue(report['met_benchmark'])
        self.assertEqual(report['test_samples'], 60)
        for ms, passed in [(0, True), (999.99, True), (1000, False), (1001, False)]:
            with self.subTest(ms=ms):
                result = benchmark_summary(self.metrics, self.classes, 'colors', {**self.timing, 'median_ms': ms})
                self.assertEqual(result['checks']['response_time_ms'], passed)
        self.metrics['per_class']['blue']['recall'] = .70
        result = benchmark_summary(self.metrics, self.classes, 'colors', self.timing)
        self.assertTrue(result['checks']['lowest_sign_recall'])
        self.assertFalse(result['checks']['macro_recall'])
        self.metrics['per_class']['blue']['recall'] = .699999
        result = benchmark_summary(self.metrics, self.classes, 'colors', self.timing)
        self.assertFalse(result['checks']['lowest_sign_recall'])

    def test_missing_timing_cannot_pass_and_partial_models_fail_coverage(self):
        report = benchmark_summary(self.metrics, self.classes, 'colors')
        self.assertEqual(report['status'], 'Incomplete')
        self.assertIsNone(report['met_benchmark'])
        report = benchmark_summary(self.metrics, self.classes[:5], 'colors', self.timing)
        self.assertFalse(report['checks']['sign_count'])
        self.metrics['per_class']['blue']['support'] = 0
        report = benchmark_summary(self.metrics, self.classes, 'colors', self.timing)
        self.assertFalse(report['checks']['test_coverage'])
        self.assertFalse(report['met_benchmark'])
        report = benchmark_summary({'accuracy': .9}, self.classes, 'colors', self.timing)
        self.assertIsNone(report['checks']['test_coverage'])
        self.assertEqual(report['status'], 'Incomplete')

    def test_nonfinite_scores_and_wrong_timing_scope_stay_unknown(self):
        self.metrics['accuracy'] = float('nan')
        report = benchmark_summary(self.metrics, self.classes, 'colors', {**self.timing, 'scope': 'local_model_inference'})
        self.assertIsNone(report['accuracy'])
        self.assertIsNone(report['response_time_ms'])
        self.assertEqual(report['status'], 'Incomplete')

    def test_local_inference_cannot_substitute_for_camera_timing(self):
        local = {'scope': 'local_model_inference', 'median_ms': 10, 'samples': 30}
        report = benchmark_summary(self.metrics, self.classes, 'colors', local)
        self.assertIsNone(report['response_time_ms'])
        self.assertEqual(report['status'], 'Incomplete')
        report = benchmark_summary(self.metrics, self.classes, 'colors', {**self.timing, 'tested_signs': self.classes[:5]})
        self.assertIsNone(report['checks']['camera_sign_coverage'])
        report = benchmark_summary(self.metrics, self.classes, 'colors', {**self.timing, 'timeouts': 1})
        self.assertFalse(report['checks']['camera_response_completion'])

    def test_legacy_report_is_derived_without_overwriting_saved_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / 'colors' / 'model, first'
            run.mkdir(parents=True)
            path = run / 'test_metrics.json'
            path.write_text(json.dumps(self.metrics))
            before = path.read_bytes()
            report = read_benchmark(run, self.metrics, self.classes, 'colors')
            self.assertEqual(report['macro_precision'], .85)
            self.assertEqual(report['lowest_sign_recall'], .85)
            self.assertEqual(report['status'], 'Incomplete')
            (run / 'camera_timing.json').write_text(json.dumps({'timing': self.timing}))
            saved = save_benchmark(run, self.metrics, self.classes, None)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(read_benchmark(run, self.metrics, self.classes, 'colors')['status'], 'Met')
            rows = list(csv.DictReader(io.StringIO((run / 'benchmark.csv').read_text())))
            self.assertEqual(rows[0]['Run'], 'model, first')
            self.assertEqual(rows[0]['Met benchmark?'], 'Met')
            self.assertIn('measured_at', saved)

    def test_latency_is_synchronized_batch_one_with_warmup_excluded(self):
        calls = []
        class Tensor:
            def __array__(self, dtype=None, copy=None):
                calls.append('synchronized')
                return np.array([[.2, .8]], dtype=dtype)
        def model(batch, training):
            self.assertEqual(batch.shape, (1, 32, 128))
            self.assertFalse(training)
            calls.append('model')
            return Tensor()
        with patch('fsl_trainer.training.benchmark.time.perf_counter', side_effect=[0, .02, 1, 1.04, 2, 2.06]):
            timing = measure_latency(model, np.zeros((2, 32, 128)), ['A', 'B'], repetitions=3, warmup=2)
        self.assertEqual(calls, ['model', 'synchronized'] * 5)
        self.assertAlmostEqual(timing['median_ms'], 40)
        self.assertEqual(timing['unique_test_samples'], 2)
        self.assertEqual(timing['batch_size'], 1)
        self.assertEqual(timing['samples'], 3)
        self.assertIn('hand extraction', timing['excludes'])
        with self.assertRaisesRegex(ValueError, 'shape'):
            measure_latency(model, np.zeros((1, 1, 128)), ['A', 'B'])


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow is not installed')
class TrainingBenchmarkIntegrationTests(unittest.TestCase):
    def test_training_writes_all_scores_and_keeps_full_camera_timing_pending(self):
        from fsl_trainer.training import trainer
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, dataset, run = base / 'source', base / 'dataset', base / 'runs/colors/smoke'
            for label_id, label in enumerate(['blue', 'green', 'orange', 'red', 'violet', 'yellow']):
                folder = source / label
                folder.mkdir(parents=True)
                for index in range(4):
                    frame = np.zeros((1, 128), dtype=np.float32)
                    frame[:, :63] = np.linspace(.1, .8, 63) + label_id * .001 + index * .0001
                    frame[:, 126] = 1
                    np.save(folder / f'{index}.npy', frame)
            prep = argparse.Namespace(source=str(source), output=str(dataset), groups=None, validation=.2,
                test=.2, seed=42, static_variants=1, jitter=.0075, hand_model=None,
                detection_confidence=.5, swap_hands=False)
            fit = argparse.Namespace(dataset=str(dataset), output=str(run), epochs=1, batch_size=12,
                learning_rate=.001, patience=1, seed=42, class_weights=False, loader='files', resume_from=None)
            with patch.dict(os.environ, {'MPLCONFIGDIR': str(base / 'mpl')}), contextlib.redirect_stdout(io.StringIO()):
                trainer.prepare(prep)
                trainer.train(fit)
                benchmark_run(argparse.Namespace(run=str(run)))
            metrics = json.loads((run / 'test_metrics.json').read_text())
            for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'lowest_sign_recall'):
                self.assertTrue(0 <= metrics[key] <= 1)
            benchmark = json.loads((run / 'benchmark.json').read_text())
            self.assertIsNone(benchmark['response_time_ms'])
            self.assertIsNone(benchmark['checks']['camera_sign_coverage'])
            self.assertEqual(benchmark['local_inference']['scope'], 'local_model_inference')
            self.assertEqual(benchmark['local_inference']['samples'], 30)
            self.assertTrue((run / 'benchmark.csv').is_file())
            self.assertTrue((run / 'best.keras').is_file())


if __name__ == '__main__':
    unittest.main()
