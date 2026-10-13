"""Integration smoke check in an installed temporary copy, leaving real runs intact.

Run with the project's Python environment. Requires the included alphabet/first
checkpoint, detector asset, TensorFlow, pip, and setuptools. No network is used.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='fsl-portability-') as temporary:
        scratch = Path(temporary).resolve()
        copied = scratch/'copied trainer with spaces'
        copied.mkdir()
        shutil.copytree(ROOT/'src'/'fsl_trainer', copied/'src'/'fsl_trainer',
                        ignore=shutil.ignore_patterns('__pycache__'))
        for relative in ('pyproject.toml', 'requirements.txt', 'trainer.py', 'categories.py',
                         'confusion_matrix.py', 'assets/hand_landmarker.task',
                         'pwa-sample/tools/export_model.py'):
            target = copied/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, target)
        existing = copied/'runs'/'alphabet'/'first'
        existing.mkdir(parents=True)
        for name in ('best.keras', 'classes.json', 'preprocessing.json'):
            shutil.copy2(ROOT/'runs'/'alphabet'/'first'/name, existing/name)
        env = os.environ.copy()
        env['PIP_NO_INDEX'] = '1'
        env['TF_CPP_MIN_LOG_LEVEL'] = '2'
        env['PYTHONIOENCODING'] = 'utf-8'
        env.pop('PYTHONPATH', None)

        def run(*arguments):
            result = subprocess.run([sys.executable, *map(str, arguments)], cwd=scratch,
                                    env=env, capture_output=True, text=True,
                                    encoding='utf-8', timeout=180)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            return result.stdout

        installed = scratch/'installed'
        run('-m', 'pip', 'install', '--no-deps', '--no-build-isolation',
            '--target', installed, '-e', copied)
        # Activate only the copied editable installation, retaining dependency paths.
        bootstrap = scratch/'installed_entry.py'
        bootstrap.write_text(
            'import runpy, site, sys\n'
            f'sys.path = [p for p in sys.path if p != {str(ROOT / "src")!r}]\n'
            f'site.addsitedir({str(installed)!r})\n'
            'from fsl_trainer.paths import ROOT\n'
            f'assert str(ROOT) == {str(copied)!r}, str(ROOT)\n'
            'module = sys.argv.pop(1)\n'
            'runpy.run_module(module, run_name="__main__")\n', encoding='utf-8')
        for label_index, label in enumerate(('A', 'B')):
            folder = copied/'data'/'raw'/'fixture'/label
            folder.mkdir(parents=True)
            for i in range(3):
                x = np.zeros((1, 128), dtype=np.float32)
                x[:, :63] = np.linspace(0.1, 0.6, 63) + label_index*0.1 + i*0.01
                x[:, 126] = 1
                np.save(folder/f'{i}.npy', x)
        print('Editable installation in relocated directory: passed', flush=True)
        run(bootstrap, 'fsl_trainer.categories', 'prepare', '--category', 'fixture', '--static-variants', '2')
        dataset = copied/'data'/'processed'/'default'/'fixture'
        run(bootstrap, 'fsl_trainer', 'inspect', '--dataset', dataset)
        for name in ('first', 'resumed'):
            args = ['train', '--category', 'fixture', '--run', name, '--epochs', '1', '--batch-size', '4']
            if name == 'resumed':
                args += ['--resume-from', str(copied/'runs'/'fixture'/'first'/'best.keras')]
            run(bootstrap, 'fsl_trainer.categories', *args)
            metrics = json.loads((copied/'runs'/'fixture'/name/'test_metrics.json').read_text())
            assert np.isfinite(metrics['loss'])
        assert list((copied/'data'/'cache').glob('*/cache.json'))
        sample = copied/'data'/'raw'/'fixture'/'A'/'0.npy'
        for checkpoint in (copied/'runs'/'fixture'/'resumed', existing):
            output = run(bootstrap, 'fsl_trainer', 'predict', '--run', checkpoint, '--sample', sample)
            probabilities = json.loads(output)
            assert abs(sum(probabilities.values()) - 1) < 1e-5
        print('Preparation, inspection, packed training, resume, and prediction: passed', flush=True)
        site = copied/'pwa-sample'/'dist'
        run(copied/'pwa-sample'/'tools'/'export_model.py', '--category', 'alphabet',
            '--run', existing, '--site', site)
        catalog = json.loads((site/'catalog.json').read_text())
        assert any(entry['id'] == 'alphabet' and entry['available'] for entry in catalog['categories'])
        print('Existing checkpoint prediction and browser export after relocation: passed', flush=True)
    print('Temporary fixture removed; project data, runs, and browser exports untouched.', flush=True)


if __name__ == '__main__':
    main()
