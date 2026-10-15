"""HUDYAT benchmark summaries from saved results; safe to import without ML packages."""
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path

EXPECTED_SIGNS = {'alphabet': 26, 'numbers': 10, 'colors': 6, 'greetings': 10}
THRESHOLDS = {'accuracy': .85, 'macro_precision': .85, 'macro_recall': .85,
              'macro_f1': .85, 'lowest_sign_recall': .70, 'response_time_ms': 1000}
CSV_FIELDS = ['Model', 'Run', 'Signs', 'Required signs', 'Accuracy (>=85%)',
              'Macro precision (>=85%)', 'Macro recall (>=85%)', 'Macro F1 (>=85%)',
              'Lowest sign recall (>=70%)', 'Lowest recall signs',
              'Response time (<1000 ms)', 'Timing scope', 'Met benchmark?', 'Missing / failed checks']


def _score(value):
    return (float(value) if not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0 <= value <= 1 else None)


def classification_summary(metrics, classes):
    """Derive missing aggregates for legacy runs; absent evidence stays unknown."""
    result = {key: _score(metrics.get(key)) for key in THRESHOLDS if key != 'response_time_ms'}
    per_class = metrics.get('per_class', {})
    if not isinstance(per_class, dict):
        per_class = {}
    rows = [per_class.get(label, {}) for label in classes]
    rows = [row if isinstance(row, dict) else {} for row in rows]
    for field in ('precision', 'recall', 'f1'):
        values = [_score(row.get(field)) for row in rows]
        if values and all(v is not None for v in values):
            if result['macro_' + field] is None:
                result['macro_' + field] = math.fsum(values) / len(values)
            if field == 'recall':
                result['lowest_sign_recall'] = min(values)
                result['lowest_recall_signs'] = [label for label, value in zip(classes, values)
                                                  if value == min(values)]
    result.setdefault('lowest_recall_signs', [])
    supports = [row.get('support') for row in rows]
    known_supports = bool(supports) and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
                                          for v in supports)
    result['test_coverage_complete'] = all(v > 0 for v in supports) if known_supports else None
    result['test_samples'] = sum(supports) if known_supports else None
    return result


def benchmark_summary(metrics, classes, category, timing=None, local_inference=None):
    summary = classification_summary(metrics, classes)
    timing = timing if isinstance(timing, dict) else {}
    if timing.get('scope') != 'hudyat_camera_to_result':
        if timing.get('scope') == 'local_model_inference' and local_inference is None:
            local_inference = timing
        timing = {}
    response = timing.get('median_ms')
    if (isinstance(response, bool) or not isinstance(response, (int, float))
            or not math.isfinite(response) or response < 0
            or timing.get('scope') != 'hudyat_camera_to_result' or not timing.get('samples')):
        response = None
    values = {key: summary[key] for key in THRESHOLDS if key != 'response_time_ms'}
    values['response_time_ms'] = response
    checks = {key: (None if value is None else value < THRESHOLDS[key]
                    if key == 'response_time_ms' else value >= THRESHOLDS[key])
              for key, value in values.items()}
    expected = EXPECTED_SIGNS.get(category)
    checks['sign_count'] = len(classes) == expected if expected is not None else None
    checks['test_coverage'] = summary['test_coverage_complete']
    tested_signs = set(timing.get('tested_signs', []))
    checks['camera_sign_coverage'] = True if timing and all(label in tested_signs for label in classes) else None
    checks['camera_response_completion'] = timing.get('timeouts', 0) == 0 if timing else None
    status = ('Not met' if any(v is False for v in checks.values()) else
              'Incomplete' if any(v is None for v in checks.values()) else 'Met')
    return {'version': 1, 'category': category, 'signs': len(classes), 'required_signs': expected,
            **summary, 'response_time_ms': response, 'timing': timing or None, 'local_inference': local_inference,
            'timing_scope': timing.get('scope') if response is not None else None,
            'thresholds': dict(THRESHOLDS), 'checks': checks, 'status': status,
            'met_benchmark': True if status == 'Met' else False if status == 'Not met' else None,
            'benchmark_scope': 'Held-out classification and full HUDYAT camera-to-confirmed-result timing on the test browser/device.'}


def read_benchmark(run, metrics, classes, category):
    try:
        saved = json.loads((run / 'benchmark.json').read_text(encoding='utf-8'))
        timing = saved.get('timing') if isinstance(saved, dict) else None
        local_inference = saved.get('local_inference') if isinstance(saved, dict) else None
        if isinstance(timing, dict) and timing.get('scope') == 'local_model_inference':
            local_inference, timing = timing, None
    except (OSError, ValueError):
        timing, local_inference = None, None
    try:
        camera = json.loads((run / 'camera_timing.json').read_text(encoding='utf-8'))
        timing = camera.get('timing')
    except (OSError, ValueError, AttributeError):
        pass
    return benchmark_summary(metrics, classes, category, timing, local_inference)


def benchmark_csv(rows):
    handle = io.StringIO(newline='')
    writer = csv.writer(handle)
    writer.writerow(CSV_FIELDS)
    def percent(value):
        return '' if value is None else f'{value * 100:.6f}%'
    for row in rows:
        report = row['benchmark']
        missing = [key + (': missing' if value is None else ': failed')
                   for key, value in report['checks'].items() if value is not True]
        writer.writerow([row['category'].capitalize(), row['name'], report['signs'], report['required_signs'],
                         *[percent(report[key]) for key in ('accuracy', 'macro_precision', 'macro_recall',
                                                           'macro_f1', 'lowest_sign_recall')],
                         ', '.join(report['lowest_recall_signs']),
                         '' if report['response_time_ms'] is None else f"{report['response_time_ms']:.6f}",
                         report['timing_scope'] or '', report['status'], '; '.join(missing)])
    return handle.getvalue()


def save_benchmark(run, metrics, classes, timing):
    run = Path(run)
    previous = read_benchmark(run, metrics, classes, run.parent.name)
    report = benchmark_summary(metrics, classes, run.parent.name, previous['timing'], timing)
    report['measured_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    (run / 'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (run / 'benchmark.csv').write_text(benchmark_csv([{'category': run.parent.name, 'name': run.name,
                                                     'benchmark': report}]), encoding='utf-8')
    return report
