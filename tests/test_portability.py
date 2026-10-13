"""Regression checks for byte-identical preparation and relocatable launchers."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from fsl_trainer import paths
from fsl_trainer.preparation.pipeline import prepare
from fsl_trainer.training.cache import open_cache
from tests.test_trainer import pose


class PortabilityTests(unittest.TestCase):
    def test_preparation_matches_original_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in ('A', 'J'):
                folder = root / 'raw' / label
                folder.mkdir(parents=True)
                for i in range(5):
                    value = pose(1 if label == 'A' else 9, i * 0.02)
                    if label == 'J':
                        value[:, 0] = np.linspace(0, 0.7 + i * 0.02, 9)
                    np.save(folder / f'{i}.npy', value)
            args = argparse.Namespace(source=str(root/'raw'), output=str(root/'dataset'),
                                      groups=None, validation=0.15, test=0.15,
                                      seed=42, static_variants=8, jitter=0.0075)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                prepare(args)
                cache, _, _, _ = open_cache(root/'dataset', cache_root=root/'cache')
            self.assertEqual(cache.parent, root/'cache')
            self.assertFalse((root/'dataset'/'_training_cache').exists())
            actual = {p.relative_to(root/'dataset').as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (root/'dataset').rglob('*') if p.is_file()}
            expected = json.loads((Path(__file__).parent/'fixtures'/'preparation_hashes.json').read_text())
            self.assertEqual(actual, expected)

    def test_relocated_project_and_category_subprocess(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()/'copied trainer with spaces'
            root.mkdir()
            shutil.copytree(paths.ROOT/'src'/'fsl_trainer', root/'src'/'fsl_trainer',
                            ignore=shutil.ignore_patterns('__pycache__'))
            for name in ('categories.py', 'trainer.py', 'confusion_matrix.py'):
                shutil.copy2(paths.ROOT/name, root/name)
            for label in ('A', 'B'):
                folder = root/'data'/'raw'/'fixture'/label
                folder.mkdir(parents=True)
                for i in range(3):
                    np.save(folder/f'{i}.npy', pose(offset=i*0.02 + (0.2 if label == 'B' else 0)))
            env = os.environ.copy()
            env['PYTHONPATH'] = str(root/'src')
            def run(*args):
                result = subprocess.run([sys.executable, *map(str, args)], cwd=temporary,
                                        env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout
            self.assertIn('fixture', run(root/'categories.py', 'list'))
            run(root/'categories.py', 'prepare', '--category', 'fixture')
            dataset = root/'data'/'processed'/'default'/'fixture'
            self.assertTrue((dataset/'manifest.json').is_file())
            run('-m', 'fsl_trainer', 'inspect', '--dataset', dataset)
            self.assertIn('fixture', run('-m', 'fsl_trainer.categories', 'list'))
            run(root/'trainer.py', '--help')
            run(root/'confusion_matrix.py', '--help')
            resolved = json.loads(run('-c', 'import json; from fsl_trainer.paths import ROOT, HAND_MODEL, CACHE; '
                                     'print(json.dumps([str(ROOT), str(HAND_MODEL), str(CACHE)]))'))
            self.assertEqual(resolved, [str(root), str(root/'assets'/'hand_landmarker.task'), str(root/'data'/'cache')])
            # Explicit relative paths continue to resolve against the caller's cwd.
            self.assertIn('fixture', run(root/'categories.py', 'list', '--data',
                                        Path(root.name)/'data'/'raw'))


if __name__ == '__main__':
    unittest.main()
