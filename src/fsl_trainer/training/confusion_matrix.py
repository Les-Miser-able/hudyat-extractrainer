"""Render saved test results: python confusion_matrix.py --run runs/alphabet/optimized."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def save_confusion_matrix(report, output):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    output = Path(output)
    classes = report['classes']
    counts = np.asarray(report['confusion_matrix_rows_true_columns_predicted'], dtype=int)
    n = len(classes)
    if counts.shape != (n, n) or not n or (counts < 0).any():
        raise ValueError('Confusion matrix must be square, nonnegative, and match the classes.')
    totals = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(counts * 100.0, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
    output.mkdir(parents=True, exist_ok=True)
    for suffix, values, normalized in [('', counts, False), ('_normalized', percentages, True)]:
        stem = output / ('confusion_matrix' + suffix)
        with stem.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['Actual / Predicted', *classes])
            for label, row in zip(classes, values):
                writer.writerow([label, *[round(float(v), 4) if normalized else int(v) for v in row]])
        size = max(8, n * 0.43 + 3)
        figure = Figure(figsize=(size, size), layout='constrained')
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        maximum = 100 if normalized else max(1, int(counts.max()))
        heatmap = ax.imshow(values, cmap='Blues', vmin=0, vmax=maximum)
        figure.colorbar(heatmap, ax=ax, shrink=0.8, label='Percent of actual class' if normalized else 'Test samples')
        ax.set(xticks=range(n), yticks=range(n), xticklabels=classes, yticklabels=classes,
               xlabel='Predicted class', ylabel='Actual class',
               title=f'Test confusion matrix — {output.name}\n' +
               ('Row percentages (zero-support rows remain zero)' if normalized else f'Counts · {int(counts.sum())} test samples'))
        ax.tick_params(axis='x', labelrotation=90)
        for i in range(n):
            for j in range(n):
                value = values[i, j]
                # Empty cells remain blank so errors are visible in larger matrices.
                if counts[i, j]:
                    ax.text(j, i, f'{value:.1f}%' if normalized else str(int(value)),
                            ha='center', va='center', fontsize=7 if n > 15 else 9,
                            color='white' if value > maximum * 0.5 else '#172b4d')
        figure.savefig(stem.with_suffix('.png'), dpi=160)
        figure.clear()
    print(f'Confusion matrices saved to {output}: counts and row percentages (PNG + CSV).', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', type=Path)
    source.add_argument('--report', type=Path, help='Downloaded webcam test JSON')
    parser.add_argument('--output', type=Path, help='Destination for plots and CSV files')
    args = parser.parse_args()
    report_path = args.report or args.run / 'test_metrics.json'
    destination = args.output or args.run or args.report.with_suffix('')
    save_confusion_matrix(json.loads(report_path.read_text(encoding='utf-8')), destination)


if __name__ == "__main__":
    main()
