"""Prepare the small HUDYAT camera compatibility update before writing files."""
import json
import os
from pathlib import Path
import tempfile

SUPPORT = Path(__file__).resolve().parent


def inside(root, relative):
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('Import paths must stay inside the selected HUDYAT folder.')
    return path


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.hudyat-import-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def runtime_changes(project):
    runtime = inside(project, 'client/src/features/fsl-recognition/category-runtime.ts')
    source = runtime.read_text(encoding='utf-8')
    package_path = inside(project, 'client/package.json')
    package = json.loads(package_path.read_text(encoding='utf-8'))
    if "from './model-preprocessing.js'" not in source:
        replacements = [
            ("import {buildModel,loadWeights,packResult,resample,sha256} from './category-core.js';",
             "import {buildModel,loadWeights,packResult,resample,sha256} from './category-core.js';\n"
             "import {normalizationFor,preprocessSequence} from './model-preprocessing.js';"),
            ('const model=buildModel(tf,metadata.classes.length);',
             'normalizationFor(metadata.preprocessing);\n const model=buildModel(tf,metadata.classes.length);'),
            ('let disposed=false;', 'let disposed=false;\n  let frameWidth=0,frameHeight=0;'),
            ('return packResult(release.id',
             'frameWidth=video.videoWidth;frameHeight=video.videoHeight;return packResult(release.id'),
            ('resample(frames),[1,32,128]',
             'resample(preprocessSequence(frames,frameWidth,frameHeight,metadata.preprocessing)),[1,32,128]'),
        ]
        for old, new in replacements:
            if source.count(old) != 1:
                raise ValueError('HUDYAT camera code has changed. Review category-runtime.ts before applying this update.')
            source = source.replace(old, new, 1)
    elif 'resample(preprocessSequence(frames,frameWidth,frameHeight,metadata.preprocessing))' not in source:
        raise ValueError('HUDYAT has a different preprocessing integration. Review it before importing.')
    changes = {}
    folder = 'client/src/features/fsl-recognition/'
    for name in ('model-preprocessing.js', 'model-preprocessing.d.ts'):
        changes[inside(project, folder + name)] = (SUPPORT / name).read_bytes()
    test = (SUPPORT / 'model-preprocessing.test.mjs').read_text(encoding='utf-8').replace(
        "from './model-preprocessing.js'", "from '../src/features/fsl-recognition/model-preprocessing.js'")
    changes[inside(project, 'client/tests/model-preprocessing.test.mjs')] = test.encode('utf-8')
    test_path = 'tests/model-preprocessing.test.mjs'
    current = package.get('scripts', {}).get('test', '')
    if test_path not in current:
        if not current.startswith('node --test '):
            raise ValueError('HUDYAT uses a different test command. Review client/package.json before importing.')
        package['scripts']['test'] = current + ' ' + test_path
        changes[package_path] = (json.dumps(package, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    changes[runtime] = source.encode('utf-8')
    return changes


def apply_changes(changes):
    for path, payload in changes.items():
        if not path.is_file() or path.read_bytes() != payload:
            atomic_write(path, payload)
            print('Updated ' + str(path))
