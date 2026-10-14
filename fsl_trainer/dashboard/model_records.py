"""Saved run provenance and manual live-review records; standard library only."""
import csv
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import tempfile
import threading

_LOCK = threading.RLock()
DECISIONS = ('Not tested', 'Testing', 'Preferred', 'Rejected')


def _read(path, fallback):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return fallback
    return value if isinstance(value, type(fallback)) else fallback


def run_metadata(run):
    config = _read(run / 'training_config.json', {})
    pre = _read(run / 'preprocessing.json', {})
    history = _read(run / 'history.json', {})
    manifest = _read(run / 'split_manifest.json', [])
    epochs = len(history['accuracy']) if isinstance(history.get('accuracy'), list) else None
    if epochs is None and (run / 'history.csv').is_file():
        with (run / 'history.csv').open(encoding='utf-8', newline='') as handle:
            epochs = max(0, sum(bool(row) for row in csv.reader(handle)) - 1)
    counts = {}
    for split, values in config.get('split_counts', {}).items():
        if isinstance(values, dict):
            counts[split] = sum(v for v in values.values() if isinstance(v, (int, float)))
    originals = {row['source'] for row in manifest if isinstance(row, dict) and isinstance(row.get('source'), str)}
    kinds = set()
    for source in originals:
        suffix = Path(source).suffix.lower()
        kinds.add('Video' if suffix in {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
                  else 'Image' if suffix in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
                  else 'Landmarks')
    selection = pre.get('source_selection', {})
    augmentation = pre.get('video_augmentation', {})
    return {'dataset': config.get('dataset'), 'source_folders': selection.get('folders'),
            'selected_sources': selection.get('selected_files'),
            'original_sources': len(originals) if originals else None,
            'input_kind': ', '.join(sorted(kinds)) or None, 'split_counts': counts,
            'epochs_recorded': epochs,
            'training': {key: config.get(key) for key in
                         ('epochs', 'batch_size', 'learning_rate', 'patience', 'class_weights', 'seed', 'resume_from')},
            'preparation': {'static_variants': pre.get('static_variants'), 'jitter_std': pre.get('jitter_std'),
                            'video_extra_samples': augmentation.get('extra_samples', 0),
                            'video_time_warp': augmentation.get('max_displacement'),
                            'detection_confidence': pre.get('extraction', {}).get('detection_confidence')}}


def read_records(base):
    with _LOCK:
        path = base / 'runs' / 'model_records.json'
        if not path.exists():
            return {}
        try:
            stored = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise ValueError('Could not read saved model records. Check runs/model_records.json.') from error
        if (not isinstance(stored, dict) or stored.get('version') != 1
                or not isinstance(stored.get('records'), dict)
                or any(not isinstance(v, dict) for v in stored['records'].values())):
            raise ValueError('Saved model records have an unsupported format.')
        return stored['records']


def _number(data, key, minimum, maximum, integer=True):
    value = data.get(key)
    if value is None or value == '':
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum
            or integer and not float(value).is_integer()):
        raise ValueError(f'{key} must be between {minimum} and {maximum}' + (' as a whole number.' if integer else '.'))
    return int(value) if integer else float(value)


def validate_record(data):
    decision = data.get('decision', 'Not tested')
    if decision not in DECISIONS:
        raise ValueError('Choose Not tested, Testing, Preferred, or Rejected.')
    mode = data.get('mode') or None
    if mode not in (None, 'gesture', 'motion', 'pose'):
        raise ValueError('Choose a valid camera mode.')
    record = {'decision': decision, 'mode': mode}
    for key, maximum in (('notes', 4000), ('device', 200)):
        value = data.get(key, '')
        if not isinstance(value, str) or len(value) > maximum:
            raise ValueError(f'{key} must be text of at most {maximum} characters.')
        record[key] = value
    for key, minimum, maximum in (('settle_ms', 100, 2000), ('missing_ms', 100, 2000),
                                  ('frame_interval_ms', 20, 500), ('live_trials', 1, 1000000),
                                  ('live_correct', 0, 1000000)):
        record[key] = _number(data, key, minimum, maximum)
    record['response_ms'] = _number(data, 'response_ms', 0, 600000, integer=False)
    if record['live_correct'] is not None and (record['live_trials'] is None
            or record['live_correct'] > record['live_trials']):
        raise ValueError('Correct trials cannot exceed total trials. Enter both counts.')
    tested_on = data.get('tested_on') or None
    if tested_on is not None:
        if not isinstance(tested_on, str):
            raise ValueError('Enter the test date as YYYY-MM-DD.')
        try:
            if date.fromisoformat(tested_on).isoformat() != tested_on:
                raise ValueError()
        except ValueError:
            raise ValueError('Enter the test date as YYYY-MM-DD.') from None
    record['tested_on'] = tested_on
    record['updated_at'] = datetime.now().isoformat(timespec='seconds')
    return record


def save_record(base, data, known_runs):
    run_id = data.get('run')
    known = {row['id']: row['category'] for row in known_runs}
    if not isinstance(run_id, str) or run_id not in known:
        raise ValueError('Select an existing saved run.')
    record = validate_record(data)
    with _LOCK:
        records = read_records(base)
        if record['decision'] == 'Preferred':
            for other, previous in records.items():
                if other != run_id and known.get(other) == known[run_id] and previous.get('decision') == 'Preferred':
                    previous['decision'] = 'Testing'
        records[run_id] = record
        path = base / 'runs' / 'model_records.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                             prefix='.model-records-', suffix='.tmp', delete=False) as handle:
                temp_path = Path(handle.name)
                json.dump({'version': 1, 'records': records}, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write('\n')
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
        return records
