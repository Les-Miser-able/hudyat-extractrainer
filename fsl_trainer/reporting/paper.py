"""Paper tables and figures derived from saved evidence, without fitting models."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import textwrap
from urllib.parse import quote

from .benchmark import classification_summary, read_benchmark
from .formatting import percentage

SPLITS = ('train', 'validation', 'test')
SCORES = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'lowest_sign_recall')
SCORE_NAMES = ('Accuracy', 'Macro precision', 'Macro recall', 'Macro F1', 'Lowest sign recall')
COLORS = ('#2166ac', '#d95f02', '#1b9e77', '#7570b3', '#b2182b')


def read_json(path, default=None):
    """Missing legacy evidence is allowed; corrupt evidence must be repaired."""
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def csv_cell(value):
    if isinstance(value, str) and value.startswith(('=', '+', '-', '@', '\t', '\r', '\n')):
        return "'" + value
    return value


def display(value):
    if value is None:
        return 'Not recorded'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


class Report:
    """Write the same observations as CSV, HTML tables, and exportable figures."""

    def __init__(self, output, title, description, *, tables_only=False):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.tables_only = tables_only
        self.parts = [f'<h1>{html.escape(title)}</h1><p>{html.escape(description)}</p>',
                      '<p>Missing evidence is shown as Not recorded and left blank in CSV. '
                      'Percentage measurements include a % sign and use two decimal places. '
                      'Charts use only recorded values. Each chart has PNG and SVG downloads.</p>']
        self.artifacts = []

    def note(self, message):
        self.parts.append(f'<p class="note">{html.escape(message)}</p>')

    def table(self, stem, title, columns, rows, *, percent_columns=(), percent_cells=(), percent_scale=100):
        rows = [[percentage(value, scale=percent_scale)
                 if column in percent_columns or (index, column) in percent_cells else value
                 for column, value in enumerate(row)] for index, row in enumerate(rows)]
        path = self.output / f'{stem}.csv'
        with path.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows([[csv_cell(v) for v in row] for row in rows])
        self.artifacts.append(path.name)
        self.parts.append(f'<section><h2>{html.escape(title)}</h2>'
                          f'<p><a href="{path.name}" download>Download table CSV</a></p>'
                          '<div class="scroll"><table><thead><tr>' +
                          ''.join(f'<th scope="col">{html.escape(c)}</th>' for c in columns) +
                          '</tr></thead><tbody>' + ''.join(
                              '<tr>' + ''.join(f'<td>{html.escape(display(v))}</td>' for v in row) + '</tr>'
                              for row in rows) + '</tbody></table></div></section>')

    def chart(self, stem, title, labels, series, unit, *, line=False, ratio=False, axis_label='Epoch'):
        """None remains missing, including for older incomplete runs."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.ticker import MaxNLocator, PercentFormatter

        if not labels or not any(numeric(v) for values in series.values() for v in values):
            self.note(f'{title}: no recorded numeric values to plot.')
            return
        if self.reuse_figure(stem, title):
            return
        count = len(labels)
        figure = Figure(figsize=(10, 4.8 if line else max(3.5, count * .34 + 1.6)), layout='constrained')
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        for i, (name, values) in enumerate(series.items()):
            values = [v if numeric(v) else float('nan') for v in values]
            if line:
                ax.plot(labels, values, label=name, color=COLORS[i % len(COLORS)], linewidth=1.6)
            else:
                width = .8 / len(series)
                positions = [j - .4 + width * (i + .5) for j in range(count)]
                ax.barh(positions, values, height=width, label=name, color=COLORS[i % len(COLORS)])
        if line:
            ax.set(xlabel=axis_label, ylabel=unit)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if ratio:
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.set(yticks=range(count), yticklabels=[textwrap.fill(str(label), 55) for label in labels], xlabel=unit)
            ax.invert_yaxis()
            ax.set_xlim(left=0)
            if ratio:
                ax.set_xlim(0, 1)
                ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(textwrap.fill(title, 85), loc='left', pad=16)
        ax.grid(axis='y' if line else 'x', alpha=.2)
        ax.set_axisbelow(True)
        if len(series) > 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9, frameon=False)
        self.figure(stem, title, figure)

    def matrix(self, stem, title, classes, matrix, extra_label=None):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        labels = [*classes, *([extra_label] if extra_label else [])]
        if not classes or len(matrix) != len(classes) or any(len(row) != len(labels) for row in matrix):
            self.note(f'{title}: no complete saved matrix.')
            return
        self.table(stem, title, ['Actual / predicted', *labels],
                   [[label, *row] for label, row in zip(classes, matrix)])
        for normalized in (False, True):
            values = [[100 * v / sum(row) if sum(row) else 0 for v in row] for row in matrix] if normalized else matrix
            suffix = '_normalized' if normalized else ''
            figure_title = title + (' — row percentages' if normalized else ' — counts')
            if normalized:
                self.table(stem + suffix, title + ' — row percentages', ['Actual / predicted (%)', *labels],
                           [[label, *row] for label, row in zip(classes, values)],
                           percent_columns=range(1, len(labels) + 1), percent_scale=1)
            if self.reuse_figure(stem + suffix, figure_title):
                continue
            figure = Figure(figsize=(max(7, len(labels) * .38 + 3), max(6, len(classes) * .38 + 2)),
                            layout='constrained')
            FigureCanvasAgg(figure)
            ax = figure.subplots()
            heatmap = ax.imshow(values, cmap='Blues', vmin=0, vmax=100 if normalized else None, aspect='auto')
            figure.colorbar(heatmap, ax=ax, label='Percent of actual sign' if normalized else 'Samples / attempts')
            ax.set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(classes)),
                   yticklabels=classes, xlabel='Predicted sign', ylabel='Actual sign',
                   title=title + (' — row percentages' if normalized else ' — counts'))
            ax.tick_params(axis='x', labelrotation=90)
            if len(classes) <= 12:
                maximum = max((max(row) for row in values), default=0)
                for y, row in enumerate(values):
                    for x, value in enumerate(row):
                        if value:
                            ax.text(x, y, f'{value:.1f}%' if normalized else str(value), ha='center', va='center',
                                    color='white' if value > maximum * .5 else '#172b4d', fontsize=8)
            self.figure(stem + suffix, figure_title, figure)

    def reuse_figure(self, stem, title):
        if self.tables_only and all((self.output / f'{stem}.{ext}').is_file() for ext in ('png', 'svg')):
            self.figure_links(stem, title)
            return True
        return False

    def figure(self, stem, title, figure):
        for extension in ('png', 'svg'):
            figure.savefig(self.output / f'{stem}.{extension}', dpi=200)
        figure.clear()
        self.figure_links(stem, title)

    def figure_links(self, stem, title):
        self.artifacts.extend(f'{stem}.{ext}' for ext in ('png', 'svg'))
        self.parts.append(f'<figure><img src="{stem}.png" alt="{html.escape(title, quote=True)}">'
                          f'<figcaption>{html.escape(title)} · <a href="{stem}.png" download>PNG</a> · '
                          f'<a href="{stem}.svg" download>SVG</a></figcaption></figure>')

    def settings(self, value, title='Recorded settings', stem='settings'):
        def flatten(mapping, prefix=''):
            for key, item in mapping.items():
                name = prefix + str(key)
                if isinstance(item, dict):
                    yield from flatten(item, name + ' / ')
                else:
                    is_percent = key in {'validation_fraction', 'test_fraction', 'mirror_probability',
                                         'detection_confidence', 'presence_confidence', 'tracking_confidence',
                                         'confidence_threshold'} or (key == 'max_displacement' and 'video_augmentation /' in prefix)
                    yield [name + (' (%)' if is_percent else ''), percentage(item) if is_percent else
                           json.dumps(item, ensure_ascii=False) if isinstance(item, list) else item]
        self.table(stem, title, ['Setting', 'Value'], flatten(value))

    def finish(self):
        generated = datetime.now(timezone.utc).isoformat(timespec='seconds')
        style = ('body{font:16px system-ui,sans-serif;color:#172b4d;max-width:1200px;margin:40px auto;padding:0 24px}'
                 'h1,h2{color:#16324f}section,figure{margin:32px 0}.scroll{overflow:auto;max-height:650px}'
                 'table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd5df;'
                 'padding:9px;text-align:left;vertical-align:top;overflow-wrap:anywhere}'
                 'th{background:#eef3f8;position:sticky;top:0}img{max-width:100%;height:auto}'
                 'a{color:#175fa3}.note{border-left:4px solid #d95f02;padding:12px;background:#fff6eb}'
                 '@media print{.scroll{max-height:none;overflow:visible}figure{break-inside:avoid}body{margin:0}}')
        content = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width, initial-scale=1">'
                   f'<title>{html.escape(self.title)}</title><style>{style}</style><body>' +
                   ''.join(self.parts) + f'<footer>Generated {generated} from saved evidence.</footer></body></html>')
        (self.output / 'index.html').write_text(content, encoding='utf-8')
        (self.output / 'report_manifest.json').write_text(json.dumps(
            {'version': 1, 'generated_at': generated, 'title': self.title,
             'artifacts': ['index.html', *self.artifacts]}, indent=2), encoding='utf-8')
        print(f'Paper tables and charts: {self.output / "index.html"}', flush=True)
        return self.output / 'index.html'


def dataset_counts(manifest):
    """Count unique accepted sources separately from output sequences, in one pass."""
    groups = defaultdict(lambda: {'sources': {}, 'groups': set(), 'outputs': 0,
                                  'synthetic': 0, 'synthetic_known': True})
    for row in manifest:
        for key in (('all', 'all'), ('split', row['split']), ('sign', row['label']),
                    ('sign_split', (row['label'], row['split']))):
            bucket = groups[key]
            bucket['sources'][row['source']] = row.get('static')
            bucket['groups'].add(row['group'])
            bucket['outputs'] += 1
            bucket['synthetic'] += bool(row.get('synthetic'))
            bucket['synthetic_known'] &= isinstance(row.get('synthetic'), bool)

    def counts(bucket):
        values = bucket['sources'].values()
        known_types = all(isinstance(v, bool) for v in values)
        return {'originals': len(bucket['sources']), 'groups': len(bucket['groups']),
                'static': sum(v is True for v in values) if known_types else None,
                'sequence': sum(v is False for v in values) if known_types else None,
                'outputs': bucket['outputs'],
                'synthetic': bucket['synthetic'] if bucket['synthetic_known'] else None,
                'clean': bucket['outputs'] - bucket['synthetic'] if bucket['synthetic_known'] else None}
    return {key: counts(value) for key, value in groups.items()}


def save_dataset_report(root, output=None, manifest=None, *, tables_only=False):
    root = Path(root)
    manifest = read_json(root / 'manifest.json') if manifest is None else manifest
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f'{root}: a nonempty prepared dataset manifest is required.')
    counts = dataset_counts(manifest)
    report = Report(output or root / 'paper_report', f'Dataset preparation — {root.parent.name} / {root.name}',
                    'Counts are derived from the saved manifest. Originals are accepted source files; '
                    'source groups can contain multiple related files. Output sequences include augmentation.', tables_only=tables_only)
    fields = ('originals', 'groups', 'static', 'sequence', 'clean', 'synthetic', 'outputs')
    headers = ['Original sources', 'Source groups', 'Static sources', 'Video / sequence sources',
               'Clean output sequences', 'Synthetic output sequences', 'Total output sequences']
    total = counts['all', 'all']
    report.table('dataset_overview', 'Preparation totals', ['Measurement', 'Count'],
                 [[name, total[field]] for field, name in zip(fields, headers)])
    report.chart('dataset_overview', 'Accepted originals and prepared output sequences',
                 ['Original sources', 'Clean outputs', 'Synthetic outputs', 'All outputs'],
                 {'Count': [total[k] for k in ('originals', 'clean', 'synthetic', 'outputs')]}, 'Count')
    for kind, labels, stem, title in (
        ('split', list(SPLITS), 'split_counts', 'Train / validation / test allocation'),
        ('sign', sorted({row['label'] for row in manifest}), 'sign_counts', 'Preparation counts by sign')):
        values = [counts[kind, label] for label in labels]
        report.table(stem, title, [kind.capitalize(), *headers],
                     [[label, *[v[f] for f in fields]] for label, v in zip(labels, values)])
        report.chart(stem, title, labels, {'Original sources': [v['originals'] for v in values],
                                          'Output sequences': [v['outputs'] for v in values]}, 'Count')
        report.chart(stem + '_augmentation', title + ' — clean and synthetic outputs', labels,
                     {'Clean outputs': [v['clean'] for v in values],
                      'Synthetic outputs': [v['synthetic'] for v in values]}, 'Output sequences')
    labels = sorted({row['label'] for row in manifest})
    for field, title in (('originals', 'Original source split by sign'), ('outputs', 'Output sequence split by sign')):
        values = [[counts['sign_split', (label, split)][field] if ('sign_split', (label, split)) in counts else 0
                   for split in SPLITS] for label in labels]
        report.table('sign_split_' + field, title, ['Sign', *SPLITS],
                     [[label, *row] for label, row in zip(labels, values)])
        report.chart('sign_split_' + field, title, labels,
                     {split: [row[i] for row in values] for i, split in enumerate(SPLITS)}, 'Count')
    report.table('source_types', 'Accepted source types', ['Type', 'Count'],
                 [['Static', total['static']], ['Video / sequence', total['sequence']]])
    report.chart('source_types', 'Accepted source types', ['Static', 'Video / sequence'],
                 {'Sources': [total['static'], total['sequence']]}, 'Original sources')
    extraction = read_json(root / 'extraction_report.json')
    if extraction:
        outcomes = {key: len(extraction.get(key, [])) for key in ('accepted', 'rejected', 'ignored')}
        report.table('extraction_outcomes', 'Extraction outcomes', ['Outcome', 'Files'], outcomes.items())
        report.chart('extraction_outcomes', 'Extraction outcomes', list(outcomes),
                     {'Files': list(outcomes.values())}, 'Files')
        reasons = Counter(row.get('reason', 'Unspecified') for row in extraction.get('rejected', []))
        report.table('extraction_rejections', 'Rejected files by reason', ['Reason', 'Files'], reasons.items())
        if reasons:
            report.chart('extraction_rejections', 'Rejected files by reason', list(reasons), {'Files': list(reasons.values())}, 'Files')
        source_labels = {row['source']: row['label'] for row in manifest}
        media = defaultdict(list)
        for item in extraction.get('accepted', []):
            label = source_labels.get(item['source'])
            if label is not None:
                media[label].append(item)
        media_rows = []
        for label, items in sorted(media.items()):
            frames = [v['frames'] for v in items if numeric(v.get('frames'))]
            detection = [v for v in items if numeric(v.get('frames')) and v['frames'] > 0
                         and numeric(v.get('detected_frames'))]
            durations = [v['frames'] / v['fps'] for v in items if v.get('kind') == 'video'
                         and numeric(v.get('frames')) and numeric(v.get('fps')) and v['fps'] > 0]
            media_rows.append([label, len(items), sum(frames) / len(frames) if frames else None,
                               sum(durations) / len(durations) if durations else None,
                               sum(v['detected_frames'] for v in detection) / sum(v['frames'] for v in detection)
                               if detection else None])
        report.table('recording_quality', 'Accepted recording length and hand detection',
                     ['Sign', 'Accepted files', 'Mean decoded frames', 'Mean video duration (s)',
                      'Detected / decoded frames (%)'], media_rows, percent_columns=(4,))
        for column, stem, title, unit, ratio in (
            (2, 'recording_frames', 'Mean decoded frames by sign', 'Frames', False),
            (3, 'recording_duration', 'Mean accepted video duration by sign', 'Seconds', False),
            (4, 'hand_detection', 'Hand detection coverage in accepted files', 'Detected / decoded frames', True)):
            report.chart(stem, title, [row[0] for row in media_rows],
                         {unit: [row[column] for row in media_rows]}, unit, ratio=ratio)
    else:
        report.note('No extraction report is saved for this dataset; rejected and ignored file counts are unknown.')
    leaking = {}
    for field in ('source', 'group', 'variant_of'):
        assignments = defaultdict(set)
        for row in manifest:
            assignments[row.get(field, row['source'])].add(row['split'])
        leaking[field] = sum(len(splits) > 1 for splits in assignments.values())
    report.table('split_isolation', 'Saved split isolation audit', ['Identity', 'Identities spanning multiple splits'], leaking.items())
    report.chart('split_isolation', 'Saved split isolation audit', list(leaking),
                 {'Violations': list(leaking.values())}, 'Identities spanning multiple splits')
    if any(leaking.values()):
        report.note('Saved source identities cross splits. This report does not certify this dataset as isolated.')
    if any(counts.get(('split', split), {}).get('synthetic', 0) for split in ('validation', 'test')):
        report.note('This saved dataset includes synthetic validation/test sequences. '
                    'Those output counts must not be described as independent original test sources.')
    augmentation = read_json(root / 'video_augmentation_report.json', {})
    if augmentation:
        fields = ('requested_extra_samples', 'generated_extra_samples', 'unfilled_extra_samples',
                  'duplicate_candidates', 'invalid_candidates')
        if augmentation.get('settings', {}).get('mirror_hands'):
            fields += ('requested_timing_samples', 'generated_timing_samples', 'unfilled_timing_samples',
                       'requested_mirrored_samples', 'generated_mirrored_samples', 'unfilled_mirrored_samples')
        sources = augmentation.get('sources', [])
        values = {key: sum(row.get(key, 0) for row in sources) for key in fields}
        report.table('video_augmentation', 'Video augmentation generation', ['Measurement', 'Count'], values.items())
        report.chart('video_augmentation', 'Video augmentation generation', [k.replace('_', ' ') for k in fields],
                     {'Count': list(values.values())}, 'Count')
        if augmentation.get('settings', {}).get('mirror_hands'):
            mirrored = [row for row in manifest if row.get('augmentation', {}).get('kind') == 'hand_mirror']
            labels = sorted({row['label'] for row in manifest if not row.get('static')})
            rows = [[label,
                     sum(row['label'] == label and row['augmentation']['mirror_of_variant'] == 0 for row in mirrored),
                     sum(row['label'] == label and row['augmentation']['mirror_of_variant'] != 0 for row in mirrored)]
                    for label in labels]
            report.table('hand_mirroring', 'Opposite-hand video copies by sign',
                         ['Sign', 'Mirrored clean sequences', 'Mirrored timing sequences'], rows)
            report.chart('hand_mirroring', 'Opposite-hand video copies by sign', labels,
                         {'Mirrored clean': [r[1] for r in rows], 'Mirrored timing': [r[2] for r in rows]}, 'Sequences')
            by_split = Counter(row['split'] for row in mirrored)
            report.table('hand_mirroring_splits', 'Opposite-hand copies by split', ['Split', 'Mirrored sequences'],
                         [[split, by_split[split]] for split in SPLITS])
            report.chart('hand_mirroring_splits', 'Opposite-hand copies by split', list(SPLITS),
                         {'Mirrored sequences': [by_split[split] for split in SPLITS]}, 'Sequences')
    report.settings(read_json(root / 'preprocessing.json', {}), 'Preparation and extraction settings')
    return report.finish()


def add_classification(report, metrics, classes, stem='', title='Held-out test'):
    summary = classification_summary(metrics, classes)
    if metrics.get('coverage_complete') is False:
        for key in SCORES[1:]:
            summary[key] = None
    rows = [[label, metrics.get('per_class', {}).get(label, {}).get('precision'),
             metrics.get('per_class', {}).get(label, {}).get('recall'),
             metrics.get('per_class', {}).get(label, {}).get('f1'),
             metrics.get('per_class', {}).get(label, {}).get('support')] for label in classes]
    report.table(stem + 'metrics', title + ' summary', ['Metric', 'Value'],
                 [[name, summary.get(key)] for key, name in zip(SCORES, SCORE_NAMES)] +
                 [['Loss', metrics.get('loss')], ['Evaluated samples / attempts', summary.get('test_samples')]] +
                 ([['Correct attempts', metrics['correct']]] if 'correct' in metrics else []),
                 percent_cells={(i, 1) for i in range(len(SCORES))})
    report.chart(stem + 'metrics', title + ' summary', list(SCORE_NAMES),
                 {'Score': [summary.get(k) for k in SCORES]}, 'Score', ratio=True)
    report.table(stem + 'per_sign', title + ' scores and support by sign',
                 ['Sign', 'Precision (%)', 'Recall (%)', 'F1 (%)', 'Support'], rows, percent_columns=(1, 2, 3))
    report.chart(stem + 'per_sign', title + ' scores by sign', classes,
                 {name: [row[i] for row in rows] for i, name in enumerate(('Precision', 'Recall', 'F1'), 1)}, 'Score', ratio=True)
    report.chart(stem + 'support', title + ' support by sign', classes,
                 {'Support': [row[4] for row in rows]}, 'Samples / attempts')


def add_timing(report, timing, stem, title, trials=None):
    if not isinstance(timing, dict):
        report.note(title + ': no saved measurements.')
        return
    report.note(title + ' timing scope: ' + str(timing.get('scope', 'Not recorded')))
    keys = ('median_ms', 'p95_ms', 'max_ms', 'mean_ms', 'min_ms')
    rows = [[key.replace('_ms', '').replace('_', ' '), timing.get(key)] for key in keys]
    report.table(stem, title + ' response time', ['Statistic', 'Milliseconds'], rows)
    report.chart(stem, title + ' response time', [row[0] for row in rows],
                 {'Milliseconds': [row[1] for row in rows]}, 'Milliseconds')
    report.settings(timing, title + ' measurement details', stem + '_details')
    if trials is not None:
        rows = [[i, t.get('target', t.get('expected')), t.get('predicted'),
                 t.get('outcome', 'timeout' if t.get('timeout') else 'prediction'),
                 t.get('correct'), t.get('response_ms'), t.get('elapsed_ms')] for i, t in enumerate(trials, 1)]
        report.table(stem + '_trials', title + ' individual trials',
                     ['Trial', 'Expected', 'Predicted', 'Outcome', 'Correct', 'Response (ms)', 'Elapsed (ms)'], rows)
        report.chart(stem + '_trials', title + ' response per trial', [row[0] for row in rows],
                     {'Response (ms)': [row[5] for row in rows]}, 'Milliseconds', line=True, axis_label='Trial')
        outcomes = Counter(row[3] for row in rows)
        report.table(stem + '_outcomes', title + ' outcomes (all attempts)', ['Outcome', 'Attempts'], outcomes.items())
        report.chart(stem + '_outcomes', title + ' outcomes (all attempts)', list(outcomes),
                     {'Attempts': list(outcomes.values())}, 'Attempts')


def save_model_report(root, output=None, tests=None, *, tables_only=False):
    root = Path(root)
    if tests is None:
        project = root.resolve().parents[2] if root.parent.parent.name == 'runs' else None
        saved = read_json(project / 'runs' / 'prediction_tests.json', {}).get('tests', {}) if project else {}
        tests = [t for t in saved.values() if (project / t['run']).resolve() == root.resolve()]
    classes = read_json(root / 'classes.json', [])
    metrics = read_json(root / 'test_metrics.json', {})
    history = read_json(root / 'history.json', {})
    if not classes:
        raise ValueError(f'{root}: saved class labels are required.')
    report = Report(output or root / 'paper_report', f'Model run — {root.parent.name} / {root.name}',
                    'Training, held-out evaluation, and timing evidence for this saved run. '
                    'Local inference, trainer camera measurements, and HUDYAT camera measurements have separate scopes.', tables_only=tables_only)
    if not metrics:
        report.note('Held-out test metrics have not been saved for this run.')
    add_classification(report, metrics, classes)
    report.matrix('confusion_matrix', 'Held-out test confusion matrix', classes,
                  metrics.get('confusion_matrix_rows_true_columns_predicted', []))
    epochs = max((len(v) for v in history.values() if isinstance(v, list)), default=0)
    keys = sorted(k for k, v in history.items() if isinstance(v, list))
    percent_columns = [i + 1 for i, key in enumerate(keys) if key in ('accuracy', 'val_accuracy')]
    report.table('learning_history', 'Learning history by epoch',
                 ['Epoch', *[key + ' (%)' if key in ('accuracy', 'val_accuracy') else key for key in keys]],
                 [[i + 1, *[history[k][i] if i < len(history[k]) else None for k in keys]] for i in range(epochs)],
                 percent_columns=percent_columns)
    for metric, title in (('accuracy', 'Training and validation accuracy'), ('loss', 'Training and validation loss')):
        series = {name: [history.get(k, [])[i] if i < len(history.get(k, [])) else None for i in range(epochs)]
                  for name, k in (('Training', metric), ('Validation', 'val_' + metric))}
        report.chart('learning_' + metric, title, list(range(1, epochs + 1)), series,
                     metric.capitalize(), line=True, ratio=metric == 'accuracy')
    times = read_json(root / 'epoch_times.json', [])
    report.table('epoch_times', 'Training duration by epoch', ['Epoch', 'Seconds'], enumerate(times, 1))
    report.chart('epoch_times', 'Training duration by epoch (including validation)',
                 list(range(1, len(times) + 1)), {'Seconds': times}, 'Seconds', line=True)
    validation = [(i + 1, v) for i, v in enumerate(history.get('val_loss', [])) if numeric(v)]
    best = min(validation, key=lambda pair: pair[1]) if validation else (None, None)
    report.table('training_summary', 'Training summary', ['Measurement', 'Value'],
                 [['Recorded epochs', epochs], ['Lowest recorded validation loss epoch', best[0]],
                  ['Lowest recorded validation loss', best[1]],
                  ['Total recorded epoch time (s)', sum(times) if times else None]])
    benchmark = read_benchmark(root, metrics, classes, root.parent.name)
    report.table('benchmark_checks', 'Existing benchmark checks', ['Check', 'Passed', 'Threshold'],
                 [[key, value, benchmark['thresholds'].get(key)] for key, value in benchmark['checks'].items()],
                 percent_cells={(i, 2) for i, key in enumerate(benchmark['checks']) if key in SCORES})
    report.note('Existing benchmark status: ' + benchmark['status'] + '. ' + benchmark['benchmark_scope'])
    add_timing(report, benchmark.get('local_inference'), 'local_inference', 'Local model inference')
    camera = read_json(root / 'camera_timing.json', {})
    for i, (identity, session) in enumerate(camera.get('sessions', {}).items(), 1):
        add_timing(report, session.get('timing'), f'hudyat_session_{i}', f'HUDYAT camera session {identity}', session.get('trials', []))
    if not camera.get('sessions'):
        add_timing(report, benchmark.get('timing'), 'hudyat_timing', 'HUDYAT camera-to-result')
    for i, test in enumerate(tests, 1):
        if not test.get('automated'):
            continue
        measured = test['measurements']
        stem = f'live_test_{i}_'
        title = 'Trainer live test: ' + test['name']
        adapted = {**measured, 'per_class': measured.get('per_class', {})}
        # Keep incomplete vocabulary coverage unknown, as in the recorded session.
        if not measured.get('coverage_complete'):
            report.note(title + ': vocabulary coverage is incomplete; macro scores remain unknown.')
        add_classification(report, adapted, classes, stem, title)
        matrix = measured.get('confusion_matrix', [])
        if matrix:
            missed = measured.get('unrecognized', {})
            report.matrix(stem + 'confusion_matrix', title + ' confusion matrix', classes,
                          [[*row, missed.get(label, 0)] for label, row in zip(classes, matrix)], 'Unrecognized')
        add_timing(report, measured.get('timing'), stem + 'timing', title, test.get('trials', []))
    report.settings({'training': read_json(root / 'training_config.json', {}),
                     'preprocessing': read_json(root / 'preprocessing.json', {})})
    return report.finish()


def saved_datasets(root):
    """Use the supported legacy/named layout; never walk sample or cache trees."""
    from ..data.dataset_paths import prepared_datasets
    root = Path(root)
    return [dataset for category in sorted(root.iterdir()) if category.is_dir()
            for dataset in prepared_datasets(category)] if root.is_dir() else []


def save_study_report(project, output=None, datasets='datasets_augmented', *, tables_only=False):
    project = Path(project)
    output = Path(output or project / 'reports' / 'paper')
    report = Report(output, 'Paper tables and charts',
                    'Inventory of saved preparation and model results. Run comparisons are descriptive; '
                    'they do not select models or establish comparable experimental conditions.', tables_only=tables_only)
    dataset_rows, model_rows, links, failures = [], [], [], []
    for root in saved_datasets(project / datasets):
        identity = root.relative_to(project).as_posix()
        target = output / 'datasets' / root.relative_to(project / datasets)
        try:
            manifest = read_json(root / 'manifest.json')
            counts = dataset_counts(manifest)['all', 'all']
            save_dataset_report(root, target, manifest, tables_only=tables_only)
            dataset_rows.append([identity, *[counts[k] for k in ('originals', 'groups', 'clean', 'synthetic', 'outputs')]])
            links.append((identity, (target / 'index.html').relative_to(output).as_posix()))
        except (ValueError, KeyError, OSError, TypeError) as error:
            failures.append([identity, str(error)])
    tests = read_json(project / 'runs' / 'prediction_tests.json', {}).get('tests', {})
    run_root = project / 'runs'
    roots = sorted(run_root.glob('*/*/classes.json')) if run_root.is_dir() else []
    for classes_path in roots:
        root = classes_path.parent
        identity = root.relative_to(project).as_posix()
        target = output / 'models' / root.relative_to(run_root)
        try:
            classes = read_json(classes_path)
            metrics = read_json(root / 'test_metrics.json', {})
            summary = classification_summary(metrics, classes)
            config = read_json(root / 'training_config.json', {})
            save_model_report(root, target, [t for t in tests.values() if t['run'] == identity], tables_only=tables_only)
            model_rows.append([identity, config.get('dataset'), config.get('seed'),
                               *[summary.get(k) for k in SCORES], summary.get('test_samples')])
            links.append((identity, (target / 'index.html').relative_to(output).as_posix()))
        except (ValueError, KeyError, OSError, TypeError) as error:
            failures.append([identity, str(error)])
    report.table('datasets', 'Prepared dataset comparison',
                 ['Dataset', 'Original sources', 'Source groups', 'Clean outputs', 'Synthetic outputs', 'Total outputs'], dataset_rows)
    report.chart('datasets', 'Prepared dataset counts', [r[0] for r in dataset_rows],
                 {'Original sources': [r[1] for r in dataset_rows], 'Output sequences': [r[5] for r in dataset_rows]}, 'Count')
    report.table('models', 'Saved model results', ['Run', 'Dataset', 'Training seed',
                 *[name + ' (%)' for name in SCORE_NAMES], 'Test support'], model_rows, percent_columns=range(3, 8))
    # Separate category figures keep different vocabularies visibly distinct.
    for category in sorted({Path(row[0]).parts[1] for row in model_rows}):
        rows = [r for r in model_rows if Path(r[0]).parts[1] == category]
        report.chart('models_' + category, category.capitalize() + ' saved model scores', [Path(r[0]).name for r in rows],
                     {name: [r[i + 3] for r in rows] for i, name in enumerate(SCORE_NAMES[:4])}, 'Score', ratio=True)
    report.parts.append('<h2>Detailed tables and charts</h2><ul>' + ''.join(
        f'<li><a href="{quote(path)}">{html.escape(name)}</a></li>' for name, path in links) + '</ul>')
    report.table('unavailable', 'Reports that could not be generated', ['Artifact', 'Reason'], failures)
    if failures:
        report.note(f'{len(failures)} report(s) could not be generated; see the reasons above.')
    return report.finish()


def report_command(args):
    from ..paths import PROJECT_ROOT
    if args.dataset:
        return save_dataset_report(args.dataset, args.output, tables_only=args.tables_only)
    if args.run:
        run = Path(args.run).resolve()
        tests = read_json(PROJECT_ROOT / 'runs' / 'prediction_tests.json', {}).get('tests', {})
        selected = [t for t in tests.values() if (PROJECT_ROOT / t['run']).resolve() == run]
        return save_model_report(run, args.output, selected, tables_only=args.tables_only)
    return save_study_report(PROJECT_ROOT, args.output, args.datasets, tables_only=args.tables_only)


def add_arguments(subparsers):
    parser = subparsers.add_parser('paper-report', help='Generate saved-evidence tables (CSV/HTML) and figures (PNG/SVG)')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', help='Prepared dataset directory')
    source.add_argument('--run', help='Saved model run directory')
    source.add_argument('--all', action='store_true', help='Create a study report for all saved datasets and runs')
    parser.add_argument('--datasets', default='datasets_augmented', help='Prepared dataset root for --all')
    parser.add_argument('--output', help='Report folder; defaults to paper_report beside the dataset/run, or reports/paper')
    parser.add_argument('--tables-only', action='store_true', help='Refresh tables while keeping existing figures; generate missing figures')
    parser.set_defaults(func=report_command)
