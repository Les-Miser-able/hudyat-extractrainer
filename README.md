# FSL Studio — Conv1D–BiLSTM Trainer

A local workspace for preparing Filipino Sign Language (FSL) datasets, training
gesture classifiers, and testing saved models. FSL Studio combines a native Qt
desktop workspace with Python command-line tools, live camera testing, and
browser model exports for FSL Pocket and HUDYAT.

Static poses and moving gestures share the same **32-frame × 128-feature** input
and Conv1D–BiLSTM network. Each category has its own model and gesture labels;
select the category/model when predicting.

## Features

| Area | What the project supports |
|---|---|
| Training dashboard | Source counts, named datasets, configurable training, learning curves, saved-run comparisons, and streamed job logs with cancellation. |
| Landmark extraction | Local two-hand MediaPipe extraction from images/videos, plus compatible pre-extracted `.npy` inputs. |
| Dataset preparation | Source-folder and videos-only filters, group-aware splits, duplicate checks, provenance, and validation. |
| Preprocessing and augmentation | Hand-relative alphabet features that preserve movement, static-pose variants, and optional training-only sequence timing variations. |
| Model training | Separate category models, optional class weights, early stopping, best checkpoints, packed disk caches, and checkpoint continuation. |
| Evaluation | Per-class and macro metrics, confusion matrices, benchmark CSVs, local inference diagnostics, and HUDYAT camera timing. |
| Live prediction | Held poses, one-gesture capture, continuous movement capture, and adjustable recognition delays. |
| Experiment records | Preferred models, saved camera settings, notes, and independent prediction test histories. |
| Model delivery | Keras ZIP downloads, HUDYAT bundles with an importer, and FSL Pocket browser exports with parity fixtures. |

For a guided walkthrough, see the [user guide](USER_GUIDE.md).
[Repository setup](docs/REPOSITORY.md) covers fresh clones and artifact conventions.
Trained checkpoints, browser exports, and the shared hand detector are included.
Original recordings, prepared datasets, and caches stay local and must be
transferred separately when cloning the repository.

## Contents

- [Quick start](#quick-start)
- [Source data](#source-data)
- [Dataset preparation](#dataset-preparation)
- [Training and evaluation](#training-and-evaluation)
- [Live camera prediction](#live-camera-prediction)
- [Model comparison and test records](#model-comparison-and-test-records)
- [Automated benchmarks](#automated-benchmarks)
- [Downloads and browser exports](#downloads-and-browser-exports)
- [Project layout](#project-layout)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Quick start

Run these commands from the repository root in Windows PowerShell.
**Python 3.12 on Windows CPU** is the verified environment; `requirements.txt`
pins its dependencies. Node.js 22 or newer is used for the optional PWA workflow.
HUDYAT camera tests additionally need an existing HUDYAT checkout with its
JavaScript dependencies installed.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Open FSL Studio

Install the native desktop dependency and launch FSL Studio:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\.venv\Scripts\python.exe dashboard.py
```

For a double-click Windows launcher, build it once:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_windows_launcher.ps1
```

Then double-click **FSL Studio.exe** in this folder. It opens the native app
without a terminal, using the existing `.venv`. Keep the executable beside
`dashboard.py`; you can create a desktop shortcut to it. This launcher requires
the project and its installed dependencies on the computer. It does not bundle
Python or the ML libraries. Startup failures show a message with the log path;
logs are saved under `outputs/launcher/`.

The default app uses **Python / PySide6 Qt Widgets** with movable, dockable panels.
It does not embed a website, use WebView2, or start an HTTP server. The Extractor,
Trainer, Model runs, and Prediction workspaces call the existing Python services
directly. Existing datasets, checkpoints, and reports are reused.

Use **Import category** to copy a folder of gesture classes into a new category;
select a source or dataset in the explorer to preview media or configure a job.
See the [native desktop guide](docs/NATIVE_DESKTOP.md) for controls and shortcuts.

The previous browser dashboard remains an explicit optional mode:

```powershell
.\.venv\Scripts\python.exe dashboard.py --browser
```

Open **http://localhost:8090**. Pass `--port 8091` if the default port is occupied.
Browser mode does not need the desktop dependency. Browsing saved results does
not require TensorFlow; preparation, training, and prediction use the dependencies
in the **same Python environment that starts the dashboard**.

In native mode use **Extractor** to prepare a dataset, **Trainer** to validate and
train it, **Model runs** to review results, and **Prediction** to test a file or
camera. You can also add original files through your file manager and refresh.

### Prepare, train, and predict from the CLI

Add your recordings using the layout below, then run:

```powershell
.\.venv\Scripts\python.exe categories.py list
.\.venv\Scripts\python.exe categories.py prepare --category alphabet --dataset-name my_dataset_v1
.\.venv\Scripts\python.exe trainer.py inspect --dataset datasets_augmented/alphabet/my_dataset_v1
.\.venv\Scripts\python.exe categories.py train --category alphabet --dataset-name my_dataset_v1 --run my_run_v1
.\.venv\Scripts\python.exe trainer.py predict --run runs/alphabet/my_run_v1 --sample "data/alphabet/A/photo001.jpg"
```

Replace the sample path with an actual file. Choose unused dataset and run names:
existing outputs, including partial outputs from cancelled jobs, are protected
from overwrite. You can also select an included checkpoint without retraining.

To prepare and train every populated category sequentially:

```powershell
.\.venv\Scripts\python.exe categories.py prepare --all --dataset-name all_v1
.\.venv\Scripts\python.exe categories.py train --all --dataset-name all_v1 --run all_v1
```

Repeat `--category` to select several categories. With `--all`, empty categories
are skipped and reported; explicitly selecting an empty category fails.
Failed category jobs are reported while other selected jobs continue, and
successful outputs are kept.

## Source data

Organize originals by category, then gesture label. Nested folders beneath each
gesture are supported:

```text
data/
  alphabet/
    A/
      photo001.jpg
      S1/photo002.jpg
    B/photo001.png
    J/clip001.mp4
  colors/
    blue/S1/clip001.mp4
    red/S2/clip001.mp4
  numbers/
    1/sample001.npy
    2/sample001.npy
  family/
  greetings/
  shapes/
```

The named categories are starting folders; add others as needed. Each category
needs at least **two classes** and **three independent source groups per class**
after unusable files are rejected. These are processing minimums, not evidence
of recognition quality. Use recordings of real movement for dynamic signs.

Supported inputs:

- **Images:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`.
- **Videos:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`, subject to the decoder.
- **Landmarks:** `.npy` arrays following the contract below.

Extensions are case-insensitive. Animated/multipage images are rejected; convert
them to video. Unsupported files, including `.npz`, are reported as ignored.

### Raw image and video extraction

MediaPipe HandLandmarker extracts up to two hands locally. The shared detector is
`models/hand_landmarker.task`; if absent, it downloads automatically on first use.
Use `--hand-model PATH` for a custom compatible detector. Detection/presence
confidence defaults to 0.5 and is configurable with `--detection-confidence`.

Images honor EXIF orientation and produce one landmark frame. Videos decode
every frame in a fresh tracking session before full-sequence resampling.
Use constant-frame-rate recordings: timing uses frame index and reported FPS.
Unreadable/truncated clips, invalid FPS, and videos with fewer than two decodable
frames are rejected.

Left/right slots follow detected handedness; if both detections claim one slot,
the highest-confidence detection wins. `--swap-hands` reverses slots without
flipping pixels. Keep mirroring consistent between training and inference.
Missing slots have zero coordinates and zero presence flags. Sources with no
detected hand are rejected; partly detected clips retain missing-hand frames
and report detection coverage.

Identical extracted arrays within a class are deduplicated; conflicting labels
for identical arrays fail preparation. Roboflow filename variants are grouped
by a filename heuristic. Use explicit signer/session groups for relationships
that filenames cannot establish.

### Pre-extracted landmark contract

Each `.npy` file represents one original image or recording and contains a real,
finite numeric array:

| Field | Contract |
|---|---|
| Static source | Shape `(128,)` or `(1,128)`. |
| Recording | Shape `(T,128)` with `T >= 2`. |
| Columns 0–62 | Left-hand landmarks 0–20, flattened as `x0,y0,z0,…,x20,y20,z20`. |
| Columns 63–125 | Right-hand landmarks in the same order. |
| Columns 126–127 | Left/right presence flags, each exactly 0 or 1. |
| Absent hand | All 63 coordinates for its slot must be zero. |

The upstream extractor must use consistent normalization and hand identity.
Entirely empty arrays, nonfinite values, and incompatible shapes are rejected;
presence is never guessed from coordinates. Z may be negative.

For legacy mixed raw-media/`.npy` datasets, supplied arrays must use MediaPipe
image-normalized coordinates: X relative to width, Y relative to height, and Z
at approximately the X scale. Hand-relative preparation needs original frame
dimensions and **cannot accept or mix supplied `.npy` sources**. Normalized
`.npy` prediction inputs must already follow the run's saved feature convention.

## Dataset preparation

Prepared versions are saved under:

```text
datasets_augmented/<category>/<dataset name>/
  train/<gesture>/*.npy
  validation/<gesture>/*.npy
  test/<gesture>/*.npy
  manifest.json
  preprocessing.json
  augmentation_summary.json
```

Category preparation defaults to `prepared_v1`. Select `--dataset-name` when
training; it can be inferred only when exactly one dataset exists. Legacy
category-level datasets remain selectable as **Original dataset**, or with
`--dataset-name .`.

CLI roots can be changed with `--data`, `--datasets`, and `--runs` where applicable.
Category defaults resolve relative to the repository. Dashboard source/dataset
roots are configurable under **Datasets → Project folders → Scan folders**;
dashboard paths must remain inside the project.

### Splits, grouping, and validation

Default targets are **70% training / 15% validation / 15% test**, assigned to
independent source groups before generating variants. Rounding and group sizes
can change the exact proportions. All siblings retain their source identity
and stay in one split. Every class must appear in all three splits.

Give related recordings a shared signer/session/recording group ID so those
sources stay together across classes:

```csv
path,group
A/photo001.jpg,signer01
B/photo001.jpg,signer01
A/photo002.jpg,signer02
B/photo002.jpg,signer02
A/photo003.jpg,signer03
B/photo003.jpg,signer03
```

Save a complete CSV at `groups/alphabet.csv`, with paths relative to
`data/alphabet`, and prepare with `--groups-dir groups`. Each selected supported
source must appear exactly once with a nonempty group ID. Splits are independent
per category; cross-category signer separation is not imposed. A constrained
group assignment fails instead of permitting leakage.

The lower-level command accepts one category's source directory:

```powershell
.\.venv\Scripts\python.exe trainer.py prepare --source data/alphabet --output datasets_augmented/alphabet/grouped_v1 --groups groups/alphabet.csv --normalization hand-relative-sequence-v1
```

`trainer.py inspect --dataset PATH` validates shapes, values, manifest/class
coverage, and disjoint group/source/variant identities. Review
`extraction_report.json` for raw-media rejections. Successful preparation also
retains original-length arrays in `extracted/`, mappings in
`extracted_sources.json`, and detector settings/checksums in `preprocessing.json`.
Failed extraction can write a timestamped
`<dataset name>.extraction-failed-….json` beside the output folder.

Preparation reports stage names and elapsed time every 10 seconds, plus counts
where available. Completion is printed after validation and cleanup.

### Source-folder filtering

Under **Source selection**, choose folders such as S1/S2/S3 and optionally enable
**Videos only**. Preview counts update per gesture; gestures with no matching
files are omitted when filtering is active.

```powershell
.\.venv\Scripts\python.exe categories.py prepare --category colors --dataset-name selected_v1 --source-folder S1 --source-folder S2 --videos-only
```

Selected immediate gesture subfolders are searched recursively. Files directly
inside a gesture are included only with All source folders. Selection is recorded
in metadata, leaves originals untouched, and does not assign signer groups.

### Landmark preprocessing and static augmentation

New raw-media alphabet datasets prepared through FSL Studio or `categories.py`
default to **`hand-relative-sequence-v1`**. Other categories and supplied-landmark
datasets retain image coordinates. The lower-level `trainer.py prepare` defaults
to image coordinates; select hand-relative features explicitly as shown above.

| Profile | Static training samples | Static validation/test samples |
|---|---|---|
| Hand-relative | One clean original plus the remaining requested variants: ±15° rotation, 50% mirroring with hand-slot swaps, and jitter of 0.03 palm lengths by default. | One clean sequence per accepted source. |
| Original image coordinates | Eight repeat-and-jitter variants by default, with jitter standard deviation 0.0075 image units. | The same requested repeat-and-jitter variants in the source's assigned split. |

`--static-variants` defaults to 8. Leave dashboard Jitter empty for the profile's
default, or set `--jitter` in that profile's units. Static poses repeat for
32 timesteps; noise affects only present-hand coordinates. Legacy image-coordinate
augmentation uses independent Gaussian noise without rotation or mirroring.
Static hand-relative variants use random mirroring. Videos can separately gain
paired opposite-hand copies with **Add opposite-hand video copies** or
`--video-mirror-hands`, in either coordinate convention.

Hand-relative features convert XYZ to pixel units using width/height, center
landmarks 1–20 on the wrist, and scale them by each frame's pixel palm length
(wrist to middle-finger knuckle). Landmark 0 carries displacement from the first
valid wrist position, divided by the sequence's median pixel palm length.
This preserves J/Z trajectories while reducing sensitivity to size, placement,
and aspect ratio; it cannot correct perspective or missed detections.

Use `--normalization image` or **Original image coordinates** to reproduce older
alphabet preparation. Existing datasets/checkpoints are not converted: prepare
and train fresh versions. Saved metadata selects preprocessing for raw-file,
desktop-camera, and exported browser inference.

### Video and sequence augmentation

Optional timing augmentation adds sequences for **training videos and multi-frame
`.npy` sources**. It is disabled by default:

```powershell
.\.venv\Scripts\python.exe categories.py prepare --category alphabet --dataset-name timing_v1 --video-extra-samples 3 --video-time-warp 0.05
```

`--video-extra-samples` accepts 0–1000 extras per eligible source; zero disables
it. `--video-time-warp` accepts values greater than zero through 0.10 and defaults
to 0.05, allowing shifts up to 5% of the normalized clip timeline. It preserves
endpoints, movement order, labels, and groups. Validation/test videos keep their
original samples, and no modified video files are generated.

Motionless or sparsely detected sources can yield fewer distinct extras.
`video_augmentation_report.json` records requests, actual counts, skips, and
shortfalls. The manifest stores time maps; training copies the report into the
run. See the [augmentation design](docs/design/video-augmentation.md) and
[augmentation workspace guide](augmentation/README.md).

Enable **Add opposite-hand video copies** in dataset preparation to learn from
mirrored versions of right- or left-hand recordings. It is off by default and
also works with zero timing extras. With three valid timing extras, a training
video produces up to eight sequences: four in its original orientation and four
mirrored. Hand slots and presence flags are swapped across the entire gesture;
the label and source group remain the same. Validation/test inputs are not mirrored.

```powershell
.\.venv\Scripts\python.exe categories.py prepare --category colors --dataset-name both_hands_v1 --video-mirror-hands --video-extra-samples 3
```

Use this option only when the mirrored gesture retains its meaning. Prepare a
new dataset and train a new model; confirm recognition with actual left- and
right-hand test attempts. [Mirroring details](docs/design/hand-mirroring.md)
include provenance, duplicate handling, and automatically generated paper tables/charts.

All prepared samples are `float32` with shape `(32,128)`. Video coordinates use
uniform linear interpolation over the full recording, preserving endpoints;
presence flags use nearest-neighbor sampling and absent slots are zeroed.
Interpolation cannot recover missed hands. Inference applies ordinary
preprocessing without stochastic augmentation.

## Training and evaluation

Each category builds a fresh model in a separate process. Training defaults are
100 epochs, batch size 32, Adam learning rate 0.001, early-stopping patience 12,
and seed 42. Optional `--class-weights` uses training counts only.

Early stopping, learning-rate reduction, and best-checkpoint selection use
validation loss. Held-out test evaluation loads the best saved model. Labels
are saved in stable alphabetical order.

### Architecture

| Layer | Configuration |
|---|---|
| Input | `32 × 128` |
| Conv1D | 64 filters, kernel 3, same padding, ReLU |
| Batch normalization | Momentum 0.99, epsilon 0.0001 |
| MaxPooling1D | Pool size 2 |
| Dropout | 0.3 |
| Conv1D | 128 filters, kernel 3, same padding, ReLU |
| Batch normalization | Momentum 0.99, epsilon 0.0001 |
| Dropout | 0.3 |
| Bidirectional LSTM | 128 units per direction, returns sequences |
| Dropout | 0.3 |
| Bidirectional LSTM | 64 units per direction |
| Dropout | 0.3 |
| Dense | 64 units, ReLU |
| Output | One softmax probability per gesture class |

The model uses sparse categorical cross-entropy and has
`485,888 + 65 × number_of_classes` parameters, including batch-normalization state.

### Packed loading and checkpoint continuation

The default `--loader packed` validates and consolidates prepared samples into
`<dataset>/_training_cache/<fingerprint>/`, then reads pre-batched memory-mapped
arrays with one prefetched batch. It preserves features, labels, and splits
without repeating extraction or augmentation. Completed caches are reused when
manifest/preprocessing and source filename/size/modification metadata match.
Keep datasets immutable during training; old caches remain on disk.
Use `--loader files` to read individual samples instead.

Continue a checkpoint using the same dataset and a **new output run**:

```powershell
.\.venv\Scripts\python.exe categories.py train --category alphabet --dataset-name my_dataset_v1 --run my_run_resumed --resume-from runs/alphabet/my_run_v1/best.keras --epochs 20
```

Continuation restores weights and optimizer state, checks class order,
architecture, and exact preprocessing metadata, and restarts early-stopping
counters. `--epochs` counts additional epochs; it does not resume an interrupted
step exactly. Select one category when resuming. Changing preprocessing or
augmentation settings requires a fresh model.

Epoch durations, including validation, are saved in `epoch_times.json` and shown
with an approximate remaining time. The loaded optimizer's learning rate is
recorded in `training_config.json`.

### Saved run artifacts

Dataset preparation and completed model runs now save paper tables and charts in
`paper_report/index.html` beside their artifacts (CSV tables, PNG/SVG figures).
Use **Build paper report** in the desktop toolbar for all saved data, then **Open
paper report**. See [paper reporting](docs/PAPER_REPORTS.md) for dataset, model,
live-test, and comparison reports and the command for rebuilding existing results.

Runs live in `runs/<category>/<run name>/`:

| Artifact | Contents |
|---|---|
| `best.keras` | Best validation checkpoint, loadable with Keras. |
| `classes.json`, `preprocessing.json` | Label order and input/extraction convention. |
| `training_config.json`, `split_manifest.json` | Training settings, framework version, and sample provenance. |
| `model_summary.txt` | Layer shapes and parameter counts. |
| `history.csv`, `history.json`, `epoch_times.json` | Learning curves and durations. |
| `test_metrics.json` | Loss, accuracy, macro precision/recall/F1, lowest sign recall, per-class scores/support, and confusion matrix. |
| `test_predictions.csv` | Per-sample true/predicted labels and confidence. |
| `confusion_matrix*.png`, `confusion_matrix*.csv` | Counts and percentages within each actual class. |
| `benchmark.json`, `benchmark.csv` | Benchmark checks and available timing evidence. |
| `camera_timing.json` | HUDYAT timing sessions, once camera measurements are saved. |

Keep **`best.keras`, `classes.json`, and `preprocessing.json` together**.
Raw-file prediction verifies saved detector settings/checksums; use `--hand-model`
for the same custom asset used in training. Runs trained only from supplied
landmarks require compatible `.npy` predictions.

Confusion-matrix rows are actual labels and columns are predictions.
Regenerate plots without retraining:

```powershell
.\.venv\Scripts\python.exe confusion_matrix.py --run runs/alphabet/my_run_v1
```

Prepared test scores describe that split. Synthetic siblings are correlated;
independent recordings and live trials are needed to assess webcam performance.

## Live camera prediction

Select a run under **Try a prediction**, then start the camera. The local Python
worker loads the checkpoint once and follows its saved extraction/preprocessing
settings. Camera inference requires raw-media metadata; runs without it can still
predict compatible landmark files.

| Mode | Behavior |
|---|---|
| **Perform once** | Captures one complete movement, then holds the result until **Next gesture**. |
| **Moving gesture** | Captures movements continuously; holds each result until the next movement starts. |
| **Held pose** | Classifies the current pose, repeated to 32 frames; clears when no hand is detected. |

Use Held pose for static letters and a movement mode to evaluate J/Z with a model
trained on appropriate movement examples. No inference augmentation is applied.
Scores below 60% display uncertainty.

Recognition timing stays editable during capture:

| Control | Default | Range |
|---|---|---|
| Stillness delay | 600 ms | 100–2,000 ms |
| Hands missing delay | 400 ms | 100–2,000 ms |
| Frame interval | 50 ms | 20–500 ms |

Changes apply on the next frame without reloading the worker/model. Finish delays
apply to movement modes; frame interval applies to every mode. Actual frequency
depends on processing speed. Settings survive camera stops/starts within the
page and can be saved with model records.

Movement capture retains up to 600 ms before onset and resamples the full interval
to 32 frames. Brief missing-hand gaps remain zero-filled; the selected delay
finishes capture and holds the result. Initial boundary heuristics require two
moving frames, 0.04 image units/second, and at least 0.025 accumulated movement.
Pauses can finish capture early; camera shake can start it. Captures exceeding
six seconds or 512 frames are rejected. Frame delivery pauses over one second
reset unfinished captures.

In browser mode the preview is mirrored; model input is unmirrored. Native mode
offers a live **Mirror preview** toggle and **Fullscreen camera** view, with a
resizable camera area. Preview mirroring leaves model input unmirrored.
Frames go only to the local
worker and are not saved. Stop live prediction before other jobs. Leaving the
prediction view or hiding the tab ends capture; abandoned workers expire after
60 seconds. See the saved [greetings normalization diagnosis](docs/diagnostics/greetings-vs-colors.md)
when assessing older checkpoints.

The optional browser dashboard binds to loopback. Both modes start one background job at a time.
Activity history lasts until restart; generated artifacts persist. **Stop job**
keeps partial outputs. Closing the desktop window stops its job/camera workers
and asks before stopping an active job. In browser mode, use Ctrl+C in its terminal.

## Model comparison and test records

**Model runs → Model comparison** shows source/dataset metadata, original/prepared
counts, training/augmentation settings, offline scores, and recorded live settings.
Filter by category or **Preferred only**, inspect learning curves, and select
**Try model** to restore a run's recorded mode and sliders.

**Edit record** saves notes, camera settings, trial counts, recognition time,
date, and device/session, with statuses **Not tested**, **Testing**, **Preferred**,
or **Rejected**. Each category has one preferred run; selecting another returns
the previous choice to Testing while keeping notes. **Record current settings**
saves camera controls without interrupting capture.

After a sample or live prediction, choose **Save as test run** for independent
tests such as Test 1/Test 2, or enter a custom name. Tests snapshot model details,
camera settings, observations, and the latest prediction. Trial counts, correct
counts, and median recognition time are manual observations and stay blank until
measured. Saving an existing test edits it; choose Save as test run again for
the next test.

For automatic measurements, open **Try a prediction → Automated test recording**
below the live camera. Start the camera, select the **Expected gesture**, and
choose **Start test session**. This saves a model/settings snapshot and arms the
first attempt. After each attempt, select the next expected gesture and choose
**Next trial**. Each attempt automatically saves its actual/predicted label,
probabilities, confidence, outcome, timestamps, response time, capture details,
and camera settings. A held pose counts once per trial. Model loading is outside
the trial; a 15-second timeout starts with its first submitted camera frame.

Live accuracy, per-gesture precision/recall/F1, the confusion matrix, and timing
summaries are derived from the saved trials. Wrong predictions, timeouts,
incomplete captures, stopped attempts, and errors count as unsuccessful attempts.
Macro precision/recall/F1 and lowest gesture recall remain unknown until every
model gesture has been attempted. Response median/p95/max include all returned
predictions, including incorrect ones; attempts without a prediction retain an
outcome and elapsed time instead of a response time.

Camera response starts before submitting the first frame whose extraction
detects a hand and ends when the prediction response arrives in the browser.
It includes encoding, local HTTP/worker processing, hand extraction, gesture
capture and its finish delay, preprocessing, and inference. It excludes waiting
before that hand frame and browser result rendering. Its scope is
`trainer_camera_first_hand_to_prediction_received`; HUDYAT's independently
measured confirmation/rendering benchmark remains separate.

Choose **Finish session** when done. **Download metrics CSV** includes session
and per-gesture measurements; **Download trials CSV** includes individual
attempts. Both are also available under **Model runs → Prediction test runs**.
CSV classification scores and confidence values include a `%` sign, such as `85.00%`;
missing measurements are empty cells. The full
JSON additionally retains probabilities, the confusion matrix, browser/camera
details, and the frozen model snapshot. Camera images/videos are not stored.
Each completed attempt is written atomically. A failed save keeps the attempt
on the page for **Retry saving attempt** and blocks further trials; keep the
page open until it succeeds. Retrying the same attempt never duplicates it.

Records persist in `runs/model_records.json`; tests persist in
`runs/prediction_tests.json`. They reference models without copying checkpoints
or retraining. **Prediction test runs** supports filtering, editing, JSON downloads,
and **Try these settings**. **Download record** includes saved configurations,
dataset metadata, scores, labels, benchmark details, and prediction test history.

## Automated benchmarks

**Model runs → Benchmark metrics** reports accuracy, macro precision/recall/F1,
lowest sign recall (including ties), response time, and **Met / Not met /
Incomplete**. Category and Preferred filters also apply to **Download metrics CSV**.

| Check | Requirement |
|---|---|
| Accuracy, macro precision, macro recall, macro F1 | Each at least 85%. |
| Lowest sign recall | At least 70%. |
| Class count | Alphabet 26, Numbers 10, Colors 6, Greetings 10. |
| Held-out coverage | Every saved class has test support. |
| Camera coverage | Every model sign has a measured response; no timeouts. |
| Response time | Median full HUDYAT camera-to-confirmed-result time strictly below 1,000 ms. |

All saved classes contribute equally to macro averages; zero denominators give
zero scores. Checks use unrounded values. Any failed check gives Not met;
otherwise missing evidence gives Incomplete. All checks must pass for Met.
Categories without a specified required class count remain incomplete.
Legacy aggregates can be derived from per-class results without rewriting scores.

### HUDYAT camera timing

Choose **HUDYAT camera test** beside a model and supply the checkout path
(default `../HUDYAT`). Preparation exports the model into
`exports/hudyat-timing/<session>/` and builds a standalone page from HUDYAT's actual
recognition, preprocessing, and confirmation sources using its installed
Vite/React/TensorFlow.js/MediaPipe dependencies. The checkout and database remain
untouched during this test.

Open the test from **Activity**, start the camera in the target browser, choose
the expected sign, and begin a trial. Perform every sign, then **Save timing results**.
Timing starts before extraction on the first submitted frame with a detected
hand and stops after a confirmed result is rendered and allowed to paint.
Extraction, preprocessing, capture, inference, confidence/stability gates, and
rendering are included; loading/downloads and time before that hand frame are excluded.

`camera_timing.json` retains expected/confirmed signs, wrong confirmations,
15-second timeouts (including trials with no hand detected), response times,
browser/camera/backend details, and checkpoint/source hashes. Sessions stay
separate; the latest saved session supplies median/p95 timing. Downloads include
these camera reports.

The current HUDYAT flow requires at least **1.2 seconds of stable alphabet estimates**
or **two-second clips for other categories**, so that flow cannot meet the
sub-second requirement from the first hand frame. The trainer records that
outcome using HUDYAT's existing rules. Evaluate deployment integration separately
if it changes the measured recognition flow.

### Local inference diagnostic

New training runs also time local inference. Use **Measure local inference**, or
run this while the saved dataset/test files remain available:

```powershell
.\.venv\Scripts\python.exe trainer.py benchmark --run runs/alphabet/my_run_v1
```

This measures 30 single-sequence predictions after three warmups, using up to
30 evenly spaced held-out samples. Output transfer and label selection are
included; camera processing and confirmation are excluded. It is a diagnostic
and does not substitute for the camera response benchmark.

## Downloads and browser exports

| Destination | Workflow |
|---|---|
| Python/Keras | **Download model ZIP** includes the checkpoint, labels, preprocessing, record, test history, and available reports. |
| HUDYAT | **Export for HUDYAT → Create HUDYAT bundle** runs conversion in Activity; download when ready. |
| FSL Pocket | Use `pwa-sample/tools/export_model.py` to update the browser category catalog. |

HUDYAT bundles contain versioned browser weights, metadata, model/preprocessing
parity fixtures, the matching detector, reports, and an importer that preserves
other category releases. Importing prepares files; activate the release separately
with HUDYAT's `npm run models:register`. See the
[HUDYAT export guide](integrations/hudyat/README.md) for import and verification.

### FSL Pocket PWA

FSL Pocket performs recognition in the browser, supports category model downloads
and removal, cached offline use, held poses, two-second gestures, and live alphabet
prediction. Camera frames stay in the browser. Existing exports are included;
additional Keras runs must be exported.

```powershell
cd pwa-sample
npm ci
npm run setup
npm start
```

Open **http://localhost:8080**. Use `npm.cmd` if PowerShell blocks `npm`.
Later starts can reuse the installed dependencies/assets. For phone use, serve
the complete prepared `pwa-sample/dist/` directory over HTTPS.

From the repository root, export a compatible raw-media-trained model:

```powershell
.\.venv\Scripts\python.exe pwa-sample/tools/export_model.py --category alphabet --run runs/alphabet/my_run_v1
```

The exporter verifies architecture/detector compatibility and provides model-output
and camera-preprocessing parity fixtures. Exporting retains saved preprocessing
and does not retrain models or establish live accuracy.

The PWA can download labeled webcam trials as JSON containing labels, scores,
and landmarks without camera images. Turn a downloaded report into plots:

```powershell
.\.venv\Scripts\python.exe confusion_matrix.py --report "reports/webcam-test-alphabet.json" --output reports/webcam/alphabet_session1
```

Use an actual report path. See the [FSL Pocket guide](pwa-sample/README.md) for
setup, storage, camera evaluation, and browser verification.

## Project layout

```text
fsl_trainer/
  data/                  Source discovery, extraction, preprocessing, augmentation
  training/              Preparation/validation, training, prediction, disk cache
  reporting/             Benchmark summaries and confusion matrices
  dashboard/             Local server, desktop window, workers, UI, model records
  desktop/               Native Qt workspaces, dock panels, monitors, media workers
src/fsl_trainer/          Packaged CLI modules
tests/                   Python and dashboard JavaScript checks
tools/                   Export, timing, and portability utilities
integrations/hudyat/      Browser preprocessing and HUDYAT bundle importer
pwa-sample/               FSL Pocket app, export tools, and browser tests
data/                    Original local recordings
datasets_augmented/      Named prepared datasets and their training caches
runs/                    Checkpoints, evaluation reports, and testing records
models/                  Shared MediaPipe hand detector
exports/                 Generated export ZIPs and timing test pages
docs/                    Setup, design notes, diagnostics, validation records
learning/                Lessons, references, and learning notes
augmentation/            Augmentation development workspace and conventions
```

Root entry points `dashboard.py`, `categories.py`, `trainer.py`, and
`confusion_matrix.py` remain supported. New Python code can import the
`fsl_trainer` package directly. Default artifact locations follow the repository.

## Verification

From the repository root, with the Python requirements installed:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

The suite covers extraction, preprocessing, grouping/leakage, augmentation,
training caches, category commands, architecture, reporting, dashboard jobs,
records, exports, and camera capture/timing. TensorFlow-dependent checks are
skipped when TensorFlow is unavailable.

Dashboard-only checks do not need ML packages:

```powershell
python -m unittest discover -s tests/dashboard -v
node --test tests/dashboard/ui.test.cjs tests/dashboard/camera-timing.test.mjs
```

After PWA dependency/asset setup, run from `pwa-sample/`:

```powershell
npm test
npm run parity
```

Parity checks require exported models. Model-output parity does not measure live
accuracy, phone installation, or physical camera behavior.
See [validation records](docs/VALIDATION.md) for recorded verification.

## Troubleshooting

| Problem | Next step |
|---|---|
| Output already contains data | Choose a fresh dataset/run name; partial outputs are retained. |
| Not enough independent groups | Add independent usable sources and inspect duplicate/detection rejections. |
| Camera needs raw-media metadata | Use compatible `.npy` input, or prepare raw media and train a new run. |
| Hand model differs from training | Supply the original detector with `--hand-model`. |
| Desktop window cannot open | Install `requirements-desktop.txt` using the Python that launches the app. |
| Port 8090 is occupied in browser mode | Stop the other server or pass `--browser --port 8091`. Native mode uses no port. |
| `/api/live/start` returns 404 after an update | Restart `dashboard.py`; refreshing does not reload Python routes. |
| Browser category unavailable | Export a compatible run, reload online, and download its model. |

Run any CLI subcommand with `--help` for its complete option list.
Further material: [user guide](USER_GUIDE.md),
[repository setup](docs/REPOSITORY.md), [HUDYAT integration](integrations/hudyat/README.md),
[FSL Pocket](pwa-sample/README.md), [gesture glossary](GLOSSARY.md), and
[learning index](learning/README.md).

### Signer holdouts and deleting saved artifacts

Choose **Leave one signer out** during preparation to hold out a signer from
`data/<category>/<gesture>/<signer>/` subfolders. Training uses the saved signer
split automatically. Use `categories.py prepare/train --loso-all` to run every
signer fold and save an aggregate paper report. See the
[LOSO guide](docs/LEAVE_ONE_SIGNER_OUT.md) for requirements and commands.

Both interfaces include delete controls for saved models and prepared datasets.
They move the selected files to recoverable local trash and preserve raw recordings.
See [deleting saved artifacts](docs/DELETING_SAVED_ARTIFACTS.md).
