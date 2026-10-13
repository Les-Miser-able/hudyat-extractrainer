<<<<<<<< HEAD:tests/training/test_trainer.py
"""Run: python -m unittest -v tests.training.test_trainer
========
"""Run: python -m unittest -v tests.test_trainer
>>>>>>>> 46d0e1656159055e1137771beb6aeb8d123f107d:tests/test_trainer.py
TensorFlow tests are skipped only when TensorFlow is not installed.
"""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fsl_trainer.common import save_json
from fsl_trainer.preparation.pipeline import prepare
from fsl_trainer.preparation.dataset import inspect_dataset
from fsl_trainer.preparation.sequences import load_sequence, resample, synthetic_sequence
from fsl_trainer.preparation.split import split_sources
from fsl_trainer.training.evaluation import evaluation
from fsl_trainer.models.network import tensorflow, build_model


def pose(length=1, offset=0.0, both=False):
    x = np.zeros((length, 128), dtype=np.float32)
    x[:, :63] = np.linspace(0.1, 0.8, 63) + offset
    x[:, 126] = 1
    if both:
        x[:, 63:126] = np.linspace(0.2, 0.9, 63) + offset
        x[:, 127] = 1
    return x


class PreprocessingTests(unittest.TestCase):
    def test_interpolation_short_and_long(self):
        for length in (2, 7, 32, 57):
            x = pose(length)
            x[:, 0] = np.linspace(0, 1, length)
            out = resample(x)
            self.assertEqual(out.shape, (32, 128))
            self.assertEqual(out.dtype, np.float32)
            np.testing.assert_allclose(out[:, 0], np.linspace(0, 1, 32), atol=1e-6)
            np.testing.assert_array_equal(out[[0, -1]], x[[0, -1]])

    def test_presence_stays_binary_and_absent_zero(self):
        x = pose(2)
        x[1] = 0
        out = resample(x)
        self.assertTrue(np.isin(out[:, 126:], [0, 1]).all())
        self.assertTrue((out[out[:, 126] == 0, :63] == 0).all())
        self.assertTrue((out[:, 63:126] == 0).all())

    def test_static_repeat(self):
        x = pose()
        np.testing.assert_array_equal(resample(x), np.repeat(x, 32, axis=0))

    def test_independent_gaussian_jitter(self):
        x = pose(both=True)
        out = synthetic_sequence(x, np.random.default_rng(12))
        noise = out[:, :126] - x[:, :126]
        self.assertGreater(noise.std(), 0.0065)
        self.assertLess(noise.std(), 0.0085)
        self.assertLess(abs(float(noise.mean())), 0.0005)
        self.assertTrue((noise.std(axis=0) > 0).all())
        self.assertLess(abs(float(np.corrcoef(noise[:-1].ravel(), noise[1:].ravel())[0, 1])), 0.08)
        np.testing.assert_array_equal(out[:, 126:], np.ones((32, 2)))
        np.testing.assert_array_equal(out, synthetic_sequence(x, np.random.default_rng(12)))

    def test_jitter_preserves_missing_hand(self):
        out = synthetic_sequence(pose(), np.random.default_rng(5))
        self.assertTrue((out[:, 63:126] == 0).all())
        self.assertTrue((out[:, 127] == 0).all())
        self.assertTrue((out[:, 126] == 1).all())

    def test_validation_rejects_invalid_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.npy"
            cases = [np.zeros((0, 128)), np.zeros((32, 126)), np.zeros((32, 128))]
            invalid = pose(); invalid[0, 0] = np.nan; cases.append(invalid)
            invalid = pose(); invalid[0, 127] = 0.5; cases.append(invalid)
            invalid = pose(); invalid[0, 63] = 0.1; cases.append(invalid)
            for value in cases:
                np.save(path, value)
                with self.assertRaises(ValueError):
                    load_sequence(path)

    def test_group_splits(self):
        records = [{"label": label, "group": f"signer{i}", "source": f"{label}/{i}.npy"}
                   for label in ("A", "J") for i in range(10)]
        split = split_sources(records, 0.15, 0.15, 42)
        self.assertEqual(set(split.values()), {"train", "validation", "test"})
        self.assertEqual(split, split_sources(records, 0.15, 0.15, 42))
        with self.assertRaises(ValueError):
            split_sources([{"label": "A", "group": "one"}], 0.15, 0.15, 42)

    def test_prepare_end_to_end_and_leakage_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for label in ("A", "J"):
                (root / "raw" / label).mkdir(parents=True)
                for i in range(5):
                    x = pose(1 if label == "A" else 9, i * 0.02)
                    if label == "J":
                        x[:, 0] = np.linspace(0, 0.7 + i * 0.02, 9)
                    np.save(root / "raw" / label / f"{i}.npy", x)
            args = argparse.Namespace(source=str(root / "raw"), output=str(root / "dataset"),
                                      groups=None, validation=0.15, test=0.15,
                                      seed=42, static_variants=8, jitter=0.0075)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(args)
            manifest, classes, stats = inspect_dataset(root / "dataset")
            self.assertEqual(classes, ["A", "J"])
            self.assertEqual(len(manifest), 45)
            summary = json.loads((root / "dataset" / "augmentation_summary.json").read_text())
            self.assertEqual(summary["original_sources"], 10)
            self.assertEqual(summary["static_sources"], 5)
            self.assertEqual(summary["augmented_static_sequences"], 40)
            self.assertEqual(summary["resampled_sequences"], 5)
            self.assertEqual(summary["total_output_sequences"], 45)
            self.assertEqual(summary["by_gesture"]["J"]["augmented_static_sequences"], 0)
            self.assertEqual(summary["by_gesture"]["A"]["total_output_sequences"], 40)
            self.assertEqual(sum(v["total_output_sequences"] for v in summary["by_split"].values()), 45)
            self.assertEqual(sum(stats[s]["A"] for s in stats), 40)
            for source in {r["source"] for r in manifest}:
                siblings = [r for r in manifest if r["source"] == source]
                self.assertEqual(len({r["split"] for r in siblings}), 1)
                self.assertEqual({r["variant_of"] for r in siblings}, {source})
                self.assertEqual(len(siblings), 8 if source.startswith("A/") else 1)
            with self.assertRaises(ValueError):
                prepare(args)
            # Corrupt source grouping across splits: inspection must reject it.
            first = manifest[0]
            other = next(r for r in manifest if r["split"] != first["split"])
            other["variant_of"] = first["variant_of"]
            save_json(root / "dataset" / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Data leakage"):
                inspect_dataset(root / "dataset")

    def test_metrics(self):
        report = evaluation(np.array([0, 1, 1]), np.array([[1, 0], [1, 0], [0, 1]]), ["A", "J"])
        self.assertAlmostEqual(report["accuracy"], 2/3)
        self.assertEqual(report["confusion_matrix_rows_true_columns_predicted"], [[1, 0], [1, 1]])


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
class TensorFlowTests(unittest.TestCase):
    def test_architecture_training_and_export(self):
        tf = tensorflow()
        tf.keras.utils.set_random_seed(42)
        model = build_model(3)
        self.assertEqual(model.input_shape, (None, 32, 128))
        self.assertEqual(model.output_shape, (None, 3))
        self.assertEqual(model.count_params(), 486083)
        conv = [x for x in model.layers if isinstance(x, tf.keras.layers.Conv1D)]
        self.assertEqual([x.filters for x in conv], [64, 128])
        self.assertTrue(all(x.kernel_size == (3,) for x in conv))
        bn = [x for x in model.layers if isinstance(x, tf.keras.layers.BatchNormalization)]
        self.assertTrue(all(x.momentum == 0.99 and x.epsilon == 0.0001 for x in bn))
        dropout = [x for x in model.layers if isinstance(x, tf.keras.layers.Dropout)]
        self.assertEqual(len(dropout), 4)
        self.assertTrue(all(x.rate == 0.3 for x in dropout))
        rnn = [x for x in model.layers if isinstance(x, tf.keras.layers.Bidirectional)]
        self.assertEqual([x.forward_layer.units for x in rnn], [128, 64])
        self.assertTrue(rnn[0].return_sequences)
        self.assertFalse(rnn[1].return_sequences)
        x = np.stack([synthetic_sequence(pose(both=True), np.random.default_rng(i)) for i in range(3)])
        result = model.train_on_batch(x, np.array([0, 1, 2]))
        self.assertTrue(np.isfinite(result).all())
        predicted = model(x, training=False).numpy()
        np.testing.assert_allclose(predicted.sum(axis=1), 1, atol=1e-6)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "model.keras"
            model.save(target)
            restored = tf.keras.models.load_model(target)
            np.testing.assert_allclose(restored(x, training=False).numpy(), predicted, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
