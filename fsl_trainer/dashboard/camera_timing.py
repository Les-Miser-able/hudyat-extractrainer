"""Validated browser measurements for the unmodified HUDYAT recognition flow."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import threading

from ..reporting.benchmark import read_benchmark, save_benchmark
from .model_records import _read

_LOCK = threading.RLock()


def camera_session(base, session):
    if not isinstance(session, str) or len(session) != 16 or any(c not in '0123456789abcdef' for c in session):
        raise ValueError('Choose a prepared HUDYAT camera session.')
    folder = (base / 'exports/hudyat-timing' / session).resolve()
    if base.resolve() not in folder.parents:
        raise ValueError('Camera session files must stay inside the trainer.')
    config = json.loads((folder / 'session.json').read_text(encoding='utf-8'))
    if config.get('session') != session:
        raise ValueError('Camera session metadata does not match.')
    return folder, config


def save_camera_timing(base, data, known_runs):
    with _LOCK:
        folder, config = camera_session(base, data.get('session'))
        if not (folder / 'index.html').is_file() or not config.get('runtime_sources'):
            raise ValueError('Complete the camera test build before saving measurements.')
        run_id = config.get('run')
        row = next((r for r in known_runs if r['id'] == run_id), None)
        if row is None:
            raise ValueError('This camera session has no saved model.')
        run = (base / run_id).resolve()
        if (base / 'runs').resolve() not in run.parents:
            raise ValueError('Camera results must belong to a saved trainer run.')
        classes = json.loads((run / 'classes.json').read_text(encoding='utf-8'))
        if (classes != config['classes'] or hashlib.sha256((run / 'best.keras').read_bytes()).hexdigest()
                != config.get('checkpoint_sha256')):
            raise ValueError('The model changed since this camera session was prepared; create a new session.')
        trials = data.get('trials')
        device = data.get('device')
        if not isinstance(trials, list) or not 1 <= len(trials) <= 1000:
            raise ValueError('Save between 1 and 1,000 camera trials.')
        if not isinstance(device, dict) or len(json.dumps(device, allow_nan=False)) > 8000:
            raise ValueError('Include the browser and camera device details.')
        validated = []
        for trial in trials:
            if not isinstance(trial, dict) or trial.get('target') not in classes:
                raise ValueError('Each trial needs an expected sign from this model.')
            timeout, correct = trial.get('timeout'), trial.get('correct')
            response, elapsed = trial.get('response_ms'), trial.get('elapsed_ms')
            predicted = trial.get('predicted')
            if not isinstance(timeout, bool) or not isinstance(correct, bool):
                raise ValueError('Each trial needs a completion outcome.')
            if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                    or not math.isfinite(elapsed) or not 0 <= elapsed <= 600000):
                raise ValueError('Camera trial elapsed time must be finite milliseconds.')
            if timeout:
                if response is not None or correct or predicted is not None or elapsed < 15000:
                    raise ValueError('Timeouts must retain the 15-second missing response.')
            elif (isinstance(response, bool) or not isinstance(response, (int, float))
                  or not math.isfinite(response) or not 0 <= response <= 600000
                  or response != elapsed or predicted not in classes
                  or correct != (predicted.strip().lower() == trial['target'].strip().lower())):
                raise ValueError('Camera result timing and recognized sign must match the trial outcome.')
            validated.append({key: trial[key] for key in ('target', 'predicted', 'correct', 'response_ms', 'elapsed_ms', 'timeout')})
        values = sorted(t['response_ms'] for t in validated if not t['timeout'])
        p95_index = max(0, math.ceil(.95 * len(values)) - 1)
        timing = {'scope': 'hudyat_camera_to_result', 'statistic': 'median',
                  'median_ms': statistics.median(values) if values else None,
                  'p95_ms': values[p95_index] if values else None,
                  'max_ms': max(values) if values else None, 'samples': len(values),
                  'timeouts': sum(t['timeout'] for t in validated), 'total_trials': len(validated),
                  'correct_trials': sum(t['correct'] for t in validated),
                  'tested_signs': sorted({t['target'] for t in validated if not t['timeout']}),
                  'device': device, 'session': config['session'], 'runtime_sources': config['runtime_sources'],
                  'checkpoint_sha256': config['checkpoint_sha256'], 'dependencies': config.get('dependencies', {}),
                  'measured_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  'start': 'Before hand extraction on the first submitted camera frame with a detected hand.',
                  'no_hand_timeout_start': 'First submitted camera frame if no hand is ever detected.',
                  'end': 'After rendering the first HUDYAT-confirmed result and allowing a browser paint.',
                  'includes': ['hand extraction', 'landmark preprocessing', 'gesture recording', 'model inference',
                               'HUDYAT confidence and stability confirmation', 'result rendering'],
                  'excludes': ['model download and loading', 'time before the first submitted hand frame']}
        path = run / 'camera_timing.json'
        previous = _read(path, {}) if path.exists() else {'version': 1, 'sessions': {}}
        if previous.get('version') != 1 or not isinstance(previous.get('sessions'), dict):
            raise ValueError('Saved camera timing has an unsupported format.')
        previous['sessions'][config['session']] = {'timing': timing, 'trials': validated}
        previous.update(latest=config['session'], timing=timing)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=run, prefix='.camera-timing-', suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(previous, handle, indent=2, allow_nan=False)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        metrics = _read(run / 'test_metrics.json', {})
        local = read_benchmark(run, metrics, classes, run.parent.name).get('local_inference')
        report = save_benchmark(run, metrics, classes, local)
        return {'benchmark': report}
