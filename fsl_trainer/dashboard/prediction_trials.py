"""Validate individual camera trials and derive exportable live-test measurements."""
import csv
import io
import math
import statistics

from .timing import capture_timing

TIMING_SCOPE = 'trainer_camera_first_hand_to_prediction_received'
OUTCOMES = ('prediction', 'timeout', 'incomplete', 'stopped', 'error')


def _milliseconds(value, name, optional=False):
    if optional and value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 600000):
        raise ValueError(f'{name} must be finite milliseconds between 0 and 600000.')
    return float(value)


def validate_trial(value, classes):
    if not isinstance(value, dict):
        raise ValueError('Send a camera trial.')
    identity = value.get('id')
    if (not isinstance(identity, str) or len(identity) != 32
            or any(c not in '0123456789abcdef' for c in identity)):
        raise ValueError('Each trial needs a unique recording ID.')
    expected, outcome = value.get('expected'), value.get('outcome')
    if expected not in classes or outcome not in OUTCOMES:
        raise ValueError('Each trial needs an expected model gesture and a valid outcome.')
    captured_at = value.get('captured_at')
    if not isinstance(captured_at, str) or not captured_at or len(captured_at) > 100:
        raise ValueError('Each trial needs its recording date.')
    elapsed = _milliseconds(value.get('elapsed_ms'), 'elapsed_ms')
    response = _milliseconds(value.get('response_ms'), 'response_ms', optional=True)
    probabilities = value.get('probabilities', {})
    if not isinstance(probabilities, dict):
        raise ValueError('Trial probabilities must be an object.')
    if outcome == 'prediction':
        if (set(probabilities) != set(classes)
                or any(isinstance(p, bool) or not isinstance(p, (int, float))
                       or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
                or not math.isclose(math.fsum(probabilities.values()), 1., abs_tol=.001)):
            raise ValueError('A completed trial needs probabilities for every model gesture summing to 1.')
        if response is None or response > elapsed:
            raise ValueError('A prediction needs response time no longer than the trial elapsed time.')
        predicted = max(classes, key=lambda label: probabilities[label])
    else:
        if probabilities or response is not None:
            raise ValueError('Unrecognized trials cannot have predictions or response time.')
        if outcome == 'timeout' and elapsed < 15000:
            raise ValueError('A timed-out trial must retain the 15-second wait.')
        predicted = None
    settings = value.get('settings')
    if not isinstance(settings, dict) or settings.get('mode') not in ('gesture', 'motion', 'pose'):
        raise ValueError('Include the camera settings used for this trial.')
    timing = capture_timing({key: settings.get(key) for key in ('settle_ms', 'missing_ms')})
    interval = settings.get('frame_interval_ms')
    if (isinstance(interval, bool) or not isinstance(interval, int) or not 20 <= interval <= 500):
        raise ValueError('Frame interval must be a whole number between 20 and 500 ms.')
    frames = value.get('frames')
    if frames is not None and (isinstance(frames, bool) or not isinstance(frames, int) or not 1 <= frames <= 10000):
        raise ValueError('Captured frame count must be a positive whole number.')
    duration = _milliseconds(value.get('duration_ms'), 'duration_ms', optional=True)
    return {'id': identity, 'expected': expected, 'outcome': outcome, 'predicted': predicted,
            'correct': predicted == expected, 'confidence': probabilities.get(predicted),
            'probabilities': dict(probabilities), 'captured_at': captured_at,
            'elapsed_ms': elapsed, 'response_ms': response, 'timing_scope': TIMING_SCOPE,
            'frames': frames, 'duration_ms': duration,
            'settings': {'mode': settings['mode'], **timing, 'frame_interval_ms': interval}}


def summarize_trials(trials, classes):
    matrix = [[0 for _ in classes] for _ in classes]
    missed = {label: 0 for label in classes}
    indices = {label: i for i, label in enumerate(classes)}
    for trial in trials:
        if trial['predicted'] is None:
            missed[trial['expected']] += 1
        else:
            matrix[indices[trial['expected']]][indices[trial['predicted']]] += 1
    per_class = {}
    for label, index in indices.items():
        tp = matrix[index][index]
        support = sum(matrix[index]) + missed[label]
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.
        recall = tp / support if support else 0.
        per_class[label] = {'precision': precision, 'recall': recall,
                            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
                            'support': support, 'correct': tp, 'unrecognized': missed[label]}
    missing = [label for label in classes if not per_class[label]['support']]
    complete = bool(classes) and not missing
    values = sorted(t['response_ms'] for t in trials if t['response_ms'] is not None)
    total, correct = len(trials), sum(t['correct'] for t in trials)
    lowest = min((row['recall'] for row in per_class.values()), default=None) if complete else None
    return {'trials': total, 'correct': correct, 'accuracy': correct / total if total else None,
            **{'macro_' + key: math.fsum(row[key] for row in per_class.values()) / len(classes)
               if complete else None for key in ('precision', 'recall', 'f1')},
            'lowest_sign_recall': lowest,
            'lowest_recall_signs': [label for label in classes if per_class[label]['recall'] == lowest],
            'coverage_complete': complete, 'missing_signs': missing, 'per_class': per_class,
            'classes': list(classes), 'confusion_matrix': matrix, 'unrecognized': missed,
            'outcomes': {outcome: sum(t['outcome'] == outcome for t in trials) for outcome in OUTCOMES},
            'timing': {'scope': TIMING_SCOPE, 'samples': len(values),
                       'median_ms': statistics.median(values) if values else None,
                       'p95_ms': values[max(0, math.ceil(.95 * len(values)) - 1)] if values else None,
                       'max_ms': max(values) if values else None}}


def trials_csv(test):
    handle = io.StringIO(newline='')
    writer = csv.writer(handle)
    writer.writerow(['Test', 'Model', 'Run', 'Trial ID', 'Recorded at', 'Expected gesture',
                     'Predicted gesture', 'Outcome', 'Correct', 'Confidence', 'Response (ms)',
                     'Elapsed (ms)', 'Timing scope', 'Captured frames', 'Gesture duration (ms)',
                     'Camera mode', 'Stillness delay (ms)', 'Hands missing delay (ms)',
                     'Frame interval (ms)', 'Device / session'])
    def cell(value):
        # Spreadsheet applications must treat user-supplied labels and names as text.
        return "'" + value if isinstance(value, str) and value.startswith(('=', '+', '-', '@', '\t', '\r', '\n')) else value
    for trial in test.get('trials', []):
        settings = trial['settings']
        writer.writerow([cell(v) for v in [test['name'], test['model']['category'], test['run'],
            trial['id'], trial['captured_at'], trial['expected'], trial['predicted'], trial['outcome'],
            trial['correct'], trial['confidence'], trial['response_ms'], trial['elapsed_ms'],
            trial['timing_scope'], trial['frames'], trial['duration_ms'], settings['mode'],
            settings['settle_ms'], settings['missing_ms'], settings['frame_interval_ms'],
            test['record'].get('device', '')]])
    return handle.getvalue()


def metrics_csv(test):
    handle = io.StringIO(newline='')
    writer = csv.writer(handle)
    writer.writerow(['Scope', 'Gesture', 'Trials', 'Correct', 'Accuracy', 'Precision', 'Recall',
                     'F1', 'Unrecognized', 'Lowest gesture recall', 'Median response (ms)',
                     'P95 response (ms)', 'Maximum response (ms)', 'Timed predictions',
                     'Timeouts', 'Incomplete', 'Stopped', 'Errors', 'Coverage complete', 'Timing scope'])
    report = test['measurements']
    timing, outcomes = report['timing'], report['outcomes']
    writer.writerow(['Session', '', report['trials'], report['correct'], report['accuracy'],
                     report['macro_precision'], report['macro_recall'], report['macro_f1'],
                     sum(report['unrecognized'].values()), report['lowest_sign_recall'],
                     timing['median_ms'], timing['p95_ms'], timing['max_ms'], timing['samples'],
                     outcomes['timeout'], outcomes['incomplete'], outcomes['stopped'], outcomes['error'],
                     report['coverage_complete'], timing['scope']])
    for label, row in report['per_class'].items():
        safe_label = "'" + label if label.startswith(('=', '+', '-', '@', '\t', '\r', '\n')) else label
        writer.writerow(['Gesture', safe_label, row['support'], row['correct'], '',
                         row['precision'], row['recall'], row['f1'], row['unrecognized']])
    return handle.getvalue()
