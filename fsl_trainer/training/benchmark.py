"""Single-sequence inference timing on the trainer device."""
import json
from pathlib import Path
import platform
import time
import numpy as np

from ..reporting.benchmark import save_benchmark


def measure_latency(model, samples, classes, repetitions=30, warmup=3):
    """Time a synchronized batch of one, including output transfer and label selection."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 3 or samples.shape[1:] != (32, 128) or not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Timing needs finite prepared test samples with shape (N, 32, 128).')
    if not classes or repetitions < 1 or warmup < 0:
        raise ValueError('Timing needs class labels and positive repetitions.')
    def infer(sample):
        # np.asarray synchronizes device output before stopping the clock.
        scores = np.asarray(model(sample[None], training=False))
        if (scores.shape != (1, len(classes)) or not np.isfinite(scores).all()
                or (scores < 0).any() or (scores > 1).any()):
            raise ValueError('The model returned invalid probabilities during timing.')
        return classes[int(scores[0].argmax())]
    for index in range(warmup):
        infer(samples[index % len(samples)])
    durations = []
    for index in range(repetitions):
        started = time.perf_counter()
        infer(samples[index % len(samples)])
        durations.append((time.perf_counter() - started) * 1000)
    return {'scope': 'local_model_inference', 'statistic': 'median', 'median_ms': float(np.median(durations)),
            'p95_ms': float(np.percentile(durations, 95)), 'mean_ms': float(np.mean(durations)),
            'min_ms': min(durations), 'max_ms': max(durations), 'samples': repetitions,
            'unique_test_samples': min(len(samples), repetitions), 'warmup': warmup, 'batch_size': 1,
            'device': {'system': platform.system(), 'machine': platform.machine(),
                       'processor': platform.processor(), 'python': platform.python_version()},
            'includes': ['model execution', 'output transfer', 'probability validation', 'top label selection'],
            'excludes': ['model loading', 'camera capture', 'hand extraction', 'landmark preprocessing',
                         'gesture recording', 'HUDYAT stability and confidence gates', 'UI rendering']}


def benchmark_run(args):
    from .trainer import tensorflow, load_sequence
    from ..paths import PROJECT_ROOT
    run = Path(args.run).resolve()
    classes = json.loads((run / 'classes.json').read_text(encoding='utf-8'))
    metrics = json.loads((run / 'test_metrics.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'training_config.json').read_text(encoding='utf-8'))
    manifest = json.loads((run / 'split_manifest.json').read_text(encoding='utf-8'))
    if not config.get('dataset'):
        raise ValueError('This run has no saved dataset path for response-time measurement.')
    dataset = Path(config['dataset'])
    dataset = (PROJECT_ROOT / dataset).resolve()
    rows = [row for row in manifest if row['split'] == 'test']
    if not rows:
        raise ValueError('This run has no saved held-out test samples for timing.')
    selected = np.linspace(0, len(rows) - 1, min(len(rows), 30), dtype=int)
    samples = []
    for index in selected:
        path = (dataset / rows[index]['path']).resolve()
        if dataset not in path.parents:
            raise ValueError('Test sample paths must stay inside the saved dataset.')
        samples.append(load_sequence(path, fixed=True))
    tf = tensorflow()
    model = tf.keras.models.load_model(run / 'best.keras', compile=False)
    if tuple(model.input_shape[1:]) != (32, 128) or model.output_shape[-1] != len(classes):
        raise ValueError('Model shape does not match its class labels or landmark layout.')
    timing = measure_latency(model, samples, classes)
    timing['tensorflow_version'] = tf.__version__
    report = save_benchmark(run, metrics, classes, timing)
    from ..reporting.paper import save_model_report
    save_model_report(run)
    print(json.dumps(report, indent=2))
