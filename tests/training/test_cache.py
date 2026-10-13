import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import trainer as t
import training_cache as c


class CacheTests(unittest.TestCase):
    def make_dataset(self, root):
        for label_index, label in enumerate(("A", "B")):
            folder = root / "raw" / label
            folder.mkdir(parents=True)
            for i in range(3):
                x = np.zeros((1,128), np.float32)
                x[:, :63] = 0.1 + label_index*0.2 + i*0.02
                x[:,126] = 1
                np.save(folder/f"{i}.npy", x)
        args = argparse.Namespace(source=str(root/"raw"), output=str(root/"dataset"),
                                  groups=None, validation=0.15, test=0.15, seed=42,
                                  static_variants=8, jitter=0.0075)
        t.prepare(args)
        return root/"dataset"

    def test_exact_values_labels_coverage_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = self.make_dataset(Path(temp))
            cache, manifest, classes, stats = c.open_cache(root)
            for split in c.SPLITS:
                rows = [r for r in manifest if r["split"] == split]
                parts = list(c.batches(cache, split, 7))
                x = np.concatenate([part[0] for part in parts])
                y = np.concatenate([part[1] for part in parts])
                self.assertEqual(len(x), len(rows))
                for i, row in enumerate(rows):
                    np.testing.assert_array_equal(x[i], t.load_sequence(root/row["path"], fixed=True))
                    self.assertEqual(y[i], classes.index(row["label"]))
                shuffled = list(c.batches(cache, split, 7, np.random.default_rng(42)))
                pairs = [(a.tobytes(), int(b)) for xx, yy in shuffled for a,b in zip(xx,yy)]
                self.assertCountEqual(pairs, [(a.tobytes(), int(b)) for a,b in zip(x,y)])
            # A cache hit does not open or validate individual sequence arrays again.
            with patch("trainer.inspect_dataset", side_effect=AssertionError("unexpected validation")):
                reused, _, _, _ = c.open_cache(root)
            self.assertEqual(reused, cache)

    def test_source_change_invalidates_cache_and_bad_data_rejected(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = self.make_dataset(Path(temp))
            first, manifest, _, _ = c.open_cache(root)
            path = root/manifest[0]["path"]
            x = np.load(path)
            x[0,0] += 0.02
            np.save(path,x)
            second, _, _, _ = c.open_cache(root)
            self.assertNotEqual(first,second)
            x[0,0] = np.nan
            np.save(path,x)
            with self.assertRaises(ValueError):
                c.open_cache(root)
            self.assertFalse(list((root/"_training_cache").glob("building-*")))

    def test_manifest_leakage_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = self.make_dataset(Path(temp))
            _, manifest, _, _ = c.open_cache(root)
            first = manifest[0]
            other = next(r for r in manifest if r["split"] != first["split"])
            other["group"] = first["group"]
            t.save_json(root/"manifest.json",manifest)
            with self.assertRaisesRegex(ValueError, "Data leakage"):
                c.open_cache(root)


if __name__ == "__main__":
    unittest.main()
