"""Durable prediction tests with a snapshot of their trained model."""
from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from uuid import uuid4

from ..reporting.benchmark import read_benchmark
from .model_records import _read, read_records, validate_record
from .prediction_trials import summarize_trials, validate_trial

_LOCK = threading.RLock()


def read_tests(base):
    with _LOCK:
        path = base / 'runs' / 'prediction_tests.json'
        if not path.exists():
            return {}
        try:
            stored = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise ValueError('Could not read saved prediction tests. Check runs/prediction_tests.json.') from error
        if (not isinstance(stored, dict) or stored.get('version') != 1
                or not isinstance(stored.get('tests'), dict)
                or any(not isinstance(v, dict) or v.get('id') != k
                       or not isinstance(v.get('run'), str)
                       or not isinstance(v.get('number'), int) or v['number'] < 1
                       or not isinstance(v.get('model'), dict)
                       or not isinstance(v.get('record'), dict)
                       or ('automated' in v and not isinstance(v['automated'], bool))
                       or (v.get('automated') and (not isinstance(v.get('trials'), list)
                           or len(v['trials']) > 1000 or not isinstance(v.get('measurements'), dict)))
                       for k, v in stored['tests'].items())):
            raise ValueError('Saved prediction tests have an unsupported format.')
        return stored['tests']


def model_snapshot(base, run_id, known_runs):
    run = next((row for row in known_runs if row['id'] == run_id), None)
    if run is None:
        raise ValueError('Select an existing saved run.')
    path = (base / run_id).resolve()
    if base.resolve() not in path.parents:
        raise ValueError('Paths must stay inside the project folder.')
    return deepcopy({**run, 'record': read_records(base).get(run_id, {}),
                     'training_config': _read(path / 'training_config.json', {}),
                     'preprocessing': _read(path / 'preprocessing.json', {}),
                     'metrics': _read(path / 'test_metrics.json', {}),
                     'camera_timing': _read(path / 'camera_timing.json', {}),
                     'benchmark': read_benchmark(path, _read(path / 'test_metrics.json', {}),
                                                 run.get('class_labels', []), run['category'])})


def _prediction(value, model):
    if value is None:
        return None
    if not isinstance(value, dict) or value.get('source') not in ('sample', 'live'):
        raise ValueError('Choose a sample or live prediction result.')
    probabilities = value.get('probabilities')
    labels = model.get('class_labels', [])
    if (not isinstance(probabilities, dict) or not probabilities
            or any(label not in labels or isinstance(score, bool)
                   or not isinstance(score, (int, float)) or not math.isfinite(score)
                   or not 0 <= score <= 1 for label, score in probabilities.items())):
        raise ValueError('Prediction scores must match this model and be between 0 and 1.')
    sample = value.get('sample', '')
    if not isinstance(sample, str) or len(sample) > 2000:
        raise ValueError('Sample path must be text of at most 2000 characters.')
    captured_at = value.get('captured_at', '')
    if not isinstance(captured_at, str) or len(captured_at) > 100:
        raise ValueError('Invalid prediction date.')
    return {'source': value['source'], 'sample': sample,
            'captured_at': captured_at, 'probabilities': dict(probabilities)}


def save_test(base, data, known_runs):
    """Create an independent test, or edit its observations without changing its model."""
    with _LOCK:
        tests = read_tests(base)
        test_id = data.get('id')
        if 'automated' in data and not isinstance(data['automated'], bool):
            raise ValueError('Choose whether this is an automated test.')
        if 'trials' in data or 'measurements' in data:
            raise ValueError('Measurements are calculated from individually recorded trials.')
        if test_id is not None:
            if not isinstance(test_id, str) or test_id not in tests:
                raise ValueError('Prediction test not found.')
            previous = tests[test_id]
            if data.get('run', previous['run']) != previous['run']:
                raise ValueError('A saved test cannot be moved to another model.')
            test = deepcopy(previous)
            if 'automated' in data and data['automated'] != test.get('automated', False):
                raise ValueError('A saved test cannot change its recording type.')
        else:
            run_id = data.get('run')
            model = model_snapshot(base, run_id, known_runs)
            number = max((t['number'] for t in tests.values() if t['run'] == run_id), default=0) + 1
            test = {'id': uuid4().hex, 'run': run_id, 'number': number,
                    'created_at': datetime.now().isoformat(timespec='seconds'), 'model': model}
            if data.get('automated'):
                device = data.get('device_details', {})
                if not isinstance(device, dict) or len(json.dumps(device, allow_nan=False)) > 8000:
                    raise ValueError('Camera and browser details must be a small JSON object.')
                test.update(automated=True, trials=[], device_details=deepcopy(device))
        name = data.get('name', test.get('name', f"Test {test['number']}"))
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError('Enter a test name of at most 200 characters.')
        # Inherit model descriptions, but keep each test's measurements independent.
        record = {**test['model'].get('record', {}), **test.get('record', {})}
        if test_id is None:
            record.update(decision='Testing', live_trials=None, live_correct=None, response_ms=None,
                          tested_on=datetime.now().date().isoformat())
        record.update({key: value for key, value in data.items()
                       if key not in ('id', 'run', 'name', 'prediction')})
        test['record'] = validate_record(record)
        test['name'] = name.strip()
        test['prediction'] = _prediction(data.get('prediction', test.get('prediction')), test['model'])
        if test.get('automated'):
            _measure(test)
        tests[test['id']] = test
        _write_tests(base, tests)
        return {'prediction_tests': tests, 'test': test}


def _write_tests(base, tests):
    path = base / 'runs' / 'prediction_tests.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.prediction-tests-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump({'version': 1, 'tests': tests}, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _measure(test):
    report = summarize_trials(test['trials'], test['model']['class_labels'])
    test['measurements'] = report
    test['record'].update(live_trials=report['trials'] or None,
                          live_correct=report['correct'] if report['trials'] else None,
                          response_ms=report['timing']['median_ms'])
    predictions = [trial for trial in test['trials'] if trial['outcome'] == 'prediction']
    latest = predictions[-1] if predictions else None
    test['prediction'] = {'source': 'live', 'sample': '', 'captured_at': latest['captured_at'],
                          'probabilities': latest['probabilities']} if latest else None


def append_trial(base, data):
    """Append once, including on retried requests; derive counts from saved evidence."""
    with _LOCK:
        tests = read_tests(base)
        test_id = data.get('id')
        if not isinstance(test_id, str) or test_id not in tests:
            raise ValueError('Prediction test not found.')
        test = deepcopy(tests[test_id])
        if not test.get('automated'):
            raise ValueError('Start an automated test before recording trials.')
        trial = validate_trial(data.get('trial'), test['model']['class_labels'])
        previous = next((t for t in test['trials'] if t['id'] == trial['id']), None)
        if previous is not None:
            if previous != trial:
                raise ValueError('A recorded trial cannot be replaced by a different result.')
            return {'prediction_tests': tests, 'test': test}
        if len(test['trials']) >= 1000:
            raise ValueError('This test has 1,000 trials. Start another test session.')
        test['trials'].append(trial)
        _measure(test)
        test['record']['updated_at'] = datetime.now().isoformat(timespec='seconds')
        tests[test_id] = test
        _write_tests(base, tests)
        return {'prediction_tests': tests, 'test': test}


def export_model(base, run_id, known_runs):
    return {'version': 1, 'exported_at': datetime.now().isoformat(timespec='seconds'),
            'model': model_snapshot(base, run_id, known_runs),
            'prediction_tests': [t for t in read_tests(base).values() if t['run'] == run_id]}
