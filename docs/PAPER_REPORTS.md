# Paper tables and charts

New dataset preparations automatically save `paper_report/index.html` inside the
prepared dataset directory. Completed training runs save the same report folder
inside their run directory. Every plotted section has a CSV table, a 200-DPI PNG,
and an SVG figure. The HTML report displays the tables and figures together and
provides download links. Settings are recorded in tables.

Percentage measurements in HTML and CSV tables include a `%` sign and two decimal
places: accuracy, precision, recall, F1, detection coverage, accuracy learning
history, normalized confusion matrices, and probability/fraction settings. For
example, a saved accuracy of `0.987` is displayed as `98.70%`. Counts, durations,
losses, learning rates, and coordinate jitter retain their original units.
Raw JSON measurements and training logs retain their original numeric values.

In the native app, **Build paper report** in the toolbar generates a study report
for all saved datasets and models; **Open paper report** opens it. The selected
dataset's Run configuration and the Model runs tab also have build/open controls
for individual reports. Generation runs as a normal background job in Output /
Jobs. Reopen the app after updating to load the new controls.

In the browser dashboard, use **Model runs → Paper tables and charts** for the
study report, or the build/open controls beside individual datasets and models.
Follow the build in Activity, then open the report.

To build or refresh reports for existing saved data without extraction or training:

```powershell
.\.venv\Scripts\python.exe trainer.py paper-report --all
.\.venv\Scripts\python.exe trainer.py paper-report --all --tables-only
.\.venv\Scripts\python.exe trainer.py paper-report --dataset datasets_augmented/colors/colors-prototype1
.\.venv\Scripts\python.exe trainer.py paper-report --run runs/colors/colors-prototype1-MODEL
```

The study report is `reports/paper/index.html`; its subdirectories contain the
detailed reports. Use `--datasets <root>` for a different prepared dataset root
or `--output <folder>` to choose a destination. Individual reports otherwise go
in `paper_report/` beside the saved manifest or checkpoint. Rebuilding replaces
derived report files; saved models, manifests, predictions, and metrics are read
without modification. Model ZIP downloads include the run's generated report,
its split manifest, and individual test predictions when available.

Use `--tables-only` for presentation changes when the saved evidence has not
changed. It refreshes HTML/CSV tables, reuses existing PNG/SVG figures, and draws
any missing figures. Use the normal command after changing the underlying data.

| Stage | Tables and charts |
|---|---|
| Preparation | Original sources and source groups, static/video source counts, clean/synthetic output counts, class balance, split totals and per-sign allocations, extraction acceptance/rejection/ignored counts and rejection reasons, video augmentation requests/shortfalls, and split identity overlap. |
| Recording quality | Mean decoded frames, mean decoded video duration (frames/FPS), and detected/decoded frame fraction by sign, when extraction metadata exists. |
| Opposite-hand augmentation | When enabled, requested/generated/unfilled timing and mirror counts, mirrored clean/timing sequences by sign, and mirror counts by split. |
| Signer evaluation | LOSO protocol, source/output counts by signer and split, original-source coverage by signer/gesture, and isolation audit. All-fold training adds per-signer scores and equal-weight mean/sample-SD tables and figures. |
| Training | Training/validation accuracy and loss curves with epoch tables, epoch durations, and a summary of the lowest recorded validation loss. |
| Offline evaluation | Aggregate and per-sign classification scores, support, and count/row-percentage confusion matrices. |
| Timing | Saved local inference statistics, each saved HUDYAT camera session, and each automated trainer live session; individual trial response times and all outcome counts remain separate. |
| Study comparison | Prepared dataset sizes and per-category model score comparisons, with dataset identity, training seed, and test support in the tables. |

Live trial reports are included when rebuilding model or study reports after
collecting the sessions. Manual observations are retained in the original records;
they do not become automatic trial measurements. Rebuilding an individual model
report also includes its saved automated test sessions.

Reports preserve missing evidence: unknown values are blank in CSV and displayed
as **Not recorded** in HTML. Older macro scores may be derived from complete saved
per-class scores. Automated live macro scores remain unknown until every sign is
attempted. Wrong results and unsuccessful attempts remain in trial outcome tables;
trials without a response have no invented response time. Count and normalized
live confusion matrices include an Unrecognized column.

Original-source counts are distinct from generated sequence counts. Source groups
are saved grouping identities; they must not be described as signer counts without
verified participant metadata. Legacy synthetic validation/test outputs and saved
identities spanning splits are identified explicitly. The split audit reads saved
identities; it does not inspect recordings for undiscovered duplicates or replace
dataset validation. Rejected/ignored counts are unknown if no extraction report is
saved. The study report lists artifacts it could not report and their reasons.

Each timing section retains its original scope. Local model inference and trainer
camera response do not substitute for HUDYAT's painted confirmed result. Reports
do not introduce the revised protocol's post-capture clock, deferred test evaluation,
model selection across seeds, participant tracking, or negative-trial instrumentation.
They present the measurements the system has actually saved.
