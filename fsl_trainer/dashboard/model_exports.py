"""Model backups and HUDYAT browser bundles; conversion runs in a worker."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

from ..data.preprocessing import NORMALIZATIONS
from .model_records import _read
from .prediction_tests import export_model

REQUIRED = ('best.keras', 'classes.json', 'preprocessing.json')
OPTIONAL = ('training_config.json', 'model_summary.txt', 'history.json', 'history.csv',
            'split_manifest.json', 'test_predictions.csv', 'video_augmentation_report.json',
            'test_metrics.json', 'benchmark.json', 'benchmark.csv', 'camera_timing.json', 'epoch_times.json', 'confusion_matrix.png',
            'confusion_matrix_normalized.png', 'confusion_matrix.csv',
            'confusion_matrix_normalized.csv')


def download_name(value):
    return re.sub(r'[^a-zA-Z0-9_-]', '-', value)[:100] or 'model'


def saved_run(base, run_id, known_runs):
    row = next((r for r in known_runs if r['id'] == run_id), None)
    if row is None:
        raise ValueError('Select an existing saved model.')
    run = (base / run_id).resolve()
    if base.resolve() not in run.parents or (base / 'runs').resolve() not in run.parents:
        raise ValueError('Model files must stay inside the runs folder.')
    for name in REQUIRED:
        path = run / name
        if not path.is_file() or run not in path.resolve().parents:
            raise ValueError('This model needs best.keras, classes.json, and preprocessing.json before export.')
    return run, row


def export_options(run):
    download = all((run / name).is_file() for name in REQUIRED)
    pre = _read(run / 'preprocessing.json', {})
    normalization = pre.get('normalization', 'image')
    extraction = pre.get('extraction')
    reason = ('Missing trained model files.' if not download else
              'HUDYAT camera use needs extraction metadata from raw-media training.'
              if not isinstance(extraction, dict) or not extraction else
              'HUDYAT export needs the training hand detector checksum.'
              if not isinstance(extraction.get('model_sha256'), str)
              or not re.fullmatch(r'[a-f0-9]{64}', extraction['model_sha256']) else
              'Unsupported model preprocessing.' if normalization not in NORMALIZATIONS else
              'Use lowercase letters, digits, hyphens, or underscores for the category.'
              if not re.fullmatch(r'[a-z0-9_-]+', run.parent.name) else '')
    return {'download': download, 'hudyat': not reason, 'reason': reason, 'normalization': normalization}


def matching_hand_model(base, run, override=None):
    if override is not None and not isinstance(override, str):
        raise ValueError('Enter the hand detector as a project-relative .task path.')
    extraction = _read(run / 'preprocessing.json', {}).get('extraction')
    expected = extraction.get('model_sha256') if isinstance(extraction, dict) else None
    if not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{64}', expected):
        raise ValueError('HUDYAT camera export needs the training hand detector checksum.')
    candidates = [base / override] if override else [base / 'models/hand_landmarker.task', base / 'assets/hand_landmarker.task']
    for candidate in candidates:
        path = candidate.resolve()
        if base.resolve() not in path.parents:
            raise ValueError('The hand detector must stay inside the trainer project.')
        if path.is_file() and path.suffix == '.task' and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return path
    raise ValueError('The hand detector must match the one used for training. Enter its project-relative .task path.')


def _convert(script, arguments):
    # Run in the dashboard's worker process so Stop job also stops conversion.
    spec = importlib.util.spec_from_file_location('_fsl_browser_export', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(arguments)


@contextmanager
def native_archive(base, run_id, known_runs):
    run, row = saved_run(base, run_id, known_runs)
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode='w+b') as handle:
        with ZipFile(handle, 'w', compression=ZIP_DEFLATED) as archive:
            for name in (*REQUIRED, *OPTIONAL):
                path = run / name
                if not path.is_file():
                    continue
                if run not in path.resolve().parents:
                    raise ValueError('Model artifacts must stay inside their saved run.')
                archive.write(path, name, compress_type=ZIP_STORED if name == 'best.keras' else ZIP_DEFLATED)
            report = run / 'paper_report'
            manifest = _read(report / 'report_manifest.json', {})
            if manifest:
                for name in sorted(set([*manifest.get('artifacts', []), 'report_manifest.json'])):
                    path = report / name
                    if path.is_file():
                        if run not in path.resolve().parents or report.resolve() not in path.resolve().parents:
                            raise ValueError('Paper report artifacts must stay inside their saved run.')
                        archive.write(path, 'paper_report/' + name)
            archive.writestr('model-record.json', json.dumps(export_model(base, run_id, known_runs),
                             indent=2, ensure_ascii=False, allow_nan=False))
            archive.writestr('README.txt', 'Trained model: ' + row['category'] + ' / ' + row['name'] + '\n\n'
                             'Keep best.keras, classes.json, and preprocessing.json together.\n'
                             'For raw camera/media input, also use the same MediaPipe .task detector as training.\n'
                             'Use Export for HUDYAT in the dashboard to create browser-ready weights.\n')
        handle.seek(0)
        yield handle, download_name(row['category'] + '-' + row['name']) + '-model.zip'


def write_hudyat_bundle(base, run_id, known_runs, output, hand_model=None):
    run, row = saved_run(base, run_id, known_runs)
    options = export_options(run)
    if not options['hudyat']:
        raise ValueError(options['reason'])
    detector = matching_hand_model(base, run, hand_model)
    if output.exists():
        raise ValueError('An export already exists at this output path. Choose a new filename.')
    snapshot = export_model(base, run_id, known_runs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.hudyat-export-', dir=output.parent) as temporary:
        staging = Path(temporary)
        site = staging / 'model-assets'
        _convert(base / 'pwa-sample/tools/export_model.py',
                 ['--category', row['category'], '--run', str(run), '--hand-model', str(detector), '--site', str(site)])
        catalog = json.loads((site / 'catalog.json').read_text(encoding='utf-8'))
        catalog['categories'] = [e for e in catalog['categories'] if e['id'] == row['category'] and e.get('available')]
        if len(catalog['categories']) != 1:
            raise ValueError('The browser exporter did not produce the selected model release.')
        (site / 'catalog.json').write_text(json.dumps(catalog, indent=2), encoding='utf-8')
        bundle = staging / 'bundle.zip'
        with ZipFile(bundle, 'w', compression=ZIP_DEFLATED) as archive:
            for path in sorted(site.rglob('*')):
                if path.is_file():
                    if site.resolve() not in path.resolve().parents:
                        raise ValueError('Exported artifacts must stay inside the bundle.')
                    archive.write(path, path.relative_to(staging).as_posix())
            for name in ('install_hudyat.py', 'runtime_support.py', 'model-preprocessing.js',
                         'model-preprocessing.d.ts', 'model-preprocessing.test.mjs'):
                archive.write(base / 'integrations/hudyat' / name, name)
            for name in ('benchmark.json', 'benchmark.csv', 'camera_timing.json'):
                path = run / name
                if path.is_file():
                    if run not in path.resolve().parents:
                        raise ValueError('Benchmark artifacts must stay inside their saved run.')
                    archive.write(path, name)
            archive.writestr('model-record.json', json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False))
            archive.writestr('HUDYAT_IMPORT.md',
                '# Import ' + row['category'] + ' / ' + row['name'] + '\n\n'
                'Extract this ZIP, then run from its extracted folder:\n\n'
                '```powershell\npython install_hudyat.py --project "C:/Users/Russel/Desktop/HUDYAT"\n```\n\n'
                'The importer verifies file hashes, merges this category into server/model-assets/catalog.json, '
                'and adds camera preprocessing support when needed. Other category entries are kept.\n\n'
                'To activate the imported release, run in HUDYAT/server:\n\n'
                '```powershell\nnpm run models:register\n```\n\n'
                'Then rebuild/restart HUDYAT and re-enter the category online. '
                'Exporting and importing do not change the database until you run models:register.\n'
                'Alphabet still uses HUDYAT\'s held-pose capture; this export does not establish motion accuracy for J/Z.\n')
        os.replace(bundle, output)
    return output
