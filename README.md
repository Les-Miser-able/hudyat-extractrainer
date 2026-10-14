# FSL Conv1D–BiLSTM trainer

For a fresh clone, PWA asset setup, and first-push instructions, see
[REPOSITORY.md](docs/REPOSITORY.md). Trained models are included; training data stays local.
The [project layout](docs/REPOSITORY.md#project-layout) groups Python implementations
under `fsl_trainer/` and checks under `tests/`; existing root Python commands
and imports remain supported. Project setup and verification records live in
`docs/`; lessons, references, and learning notes live in [learning/](learning/README.md).

Augmentation development folders and output conventions are documented in
[augmentation/README.md](augmentation/README.md).

A TensorFlow trainer with automatic hand-landmark extraction from raw images and videos. Existing normalized .npy landmarks are also accepted. Static poses and dynamic gestures use the same (32, 128) input, classifier and saved Keras model. No per-letter special cases.

## Local training dashboard

On Windows, install the optional desktop window dependency into the Python
environment you use for training, then start FSL Studio from the repository root:

```powershell
python -m pip install -r requirements-desktop.txt
python dashboard.py
```

FSL Studio opens in its own resizable desktop window, with no browser tabs or
address bar. It displays the existing dashboard using Microsoft Edge WebView2;
no interface rewrite, executable packaging, or installer is needed. The terminal
stays available for diagnostics. If the window cannot open, install or repair the
[Microsoft Edge WebView2 Runtime](https://developer.microsoft.com/microsoft-edge/webview2/).

To use a browser instead (also supported outside Windows), run
`python dashboard.py --browser` and open **http://localhost:8090**. Browser mode
does not need the desktop dependencies.

FSL Studio provides an overview of real source
counts and model scores, dataset preparation and validation, configurable training,
saved-run comparisons with learning curves, and prediction from a project-local
image, video, or `.npy` file. Activity logs stream while jobs run; use **Stop job**
to cancel. Partial outputs are kept, so choose a new output folder/run name to retry.

The dashboard's browser mode uses only Python's standard library. Preparation, training,
and prediction use the **same Python environment that starts the dashboard** and
need the dependencies in `requirements.txt`. If using the quick-start environment,
launch it with `.\.venv\Scripts\python.exe dashboard.py`. Browsing saved results
does not require TensorFlow. Install desktop dependencies with
`.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt` when using
that environment. Use `--port 8091` if the default port is occupied.

Closing the desktop window stops the local server and camera workers. If a job
is running, FSL Studio asks before stopping it; Cancel keeps the window and job
running. Confirming exit keeps partial outputs. In browser mode, stop the server
with Ctrl+C in its terminal.

The default source root is `data`, dataset root is `datasets_augmented`, and model
root is `runs`. Change the first two under **Datasets → Project folders → Scan
folders** to use another source or dataset root. Paths must stay inside
the project. Existing datasets and runs are protected from overwrite. Add source
files through your file manager, then refresh the dashboard. Enter an editable
**Dataset name** when preparing: output is saved under
`datasets_augmented/<category>/<dataset name>/`. The output preview updates while
you type. Keep the existing split, augmentation, source filtering, and extraction
controls to configure each version independently. Use a new name for each version;
existing and partial outputs are protected from overwrite.

Each category lists its saved datasets with a **Validate dataset** button for each
one. Under **Training**, choose the category and **Prepared dataset**; the displayed
path is passed to training and saved in that run's training configuration. Prediction
continues using the run's checkpoint and copied preprocessing settings. Existing
datasets directly under `<dataset root>/<category>/` remain selectable as
**Original dataset**; no data is moved automatically. A manifest indicates a saved
dataset; **Validate dataset** performs the full trainer check.
The dashboard starts one job at a time and binds only to loopback. Activity history
is held in memory until restart; generated artifacts remain in the usual folders.

The existing [FSL Pocket webcam app](pwa-sample/README.md) remains available
separately. Its setup/export workflow is unchanged.

**Try a prediction → Live camera prediction** uses the trained run selected above
the camera panel. The default **Perform once** mode automatically captures one
gesture: show your hands, perform the complete gesture, then hold the ending
still for about a second. The result stays displayed even if you keep holding
the ending or lower your hands. Select **Next gesture** to reset capture without
reloading the camera or model. **Moving gesture** retains the rolling two-second
mode, and **Held pose** classifies the current pose. Press **Start camera** and
allow camera access. Frames are processed serially at up to five times per
second, depending on CPU speed. **Stop camera**, leaving
the prediction view, or hiding the tab ends capture; abandoned workers expire
after 60 seconds without frames. Stop live prediction before running other jobs.

The dashboard keeps one Python inference worker and checkpoint loaded per live
session, using the run's extraction confidence, handedness swap, and verified hand
model asset. The optional sample form's **Custom hand model** also applies to live
prediction. Runs without extraction metadata support normalized `.npy` file
prediction but cannot use the camera. No model export or browser ML dependencies
are needed for this dashboard feature. Launch the dashboard with the Python
environment containing `requirements.txt`.

If `/api/live/start` returns 404 after updating the project, the running dashboard
still has the previous Python code loaded. Finish any active job, stop the server,
restart `dashboard.py` in the same Python environment (with `--browser` if using
browser mode), and reopen the camera.
Refreshing the page alone updates browser files but does not reload Python routes.

Camera frames travel only to the local dashboard worker and are not written to
disk. The preview is mirrored visually; input pixels remain unmirrored. Held
poses repeat the current landmarks to 32 frames; rolling mode interpolates the
last two seconds using capture timestamps. Perform once interpolates the entire
captured interval to 32 frames, retaining up to 600 ms before movement onset.
No inference augmentation is applied. Scores below 60% show uncertainty.

Automatic capture starts after two consecutive moving frames and finishes after
800 ms of stillness or 600 ms without detected hands. Brief missing-hand frames
stay zero-filled within the sequence. Motion uses mean XY landmark displacement
per second (threshold 0.04 in image-normalized units); movement must accumulate
at least 0.025 image units. These are initial heuristics, not learned gesture
boundaries. A pause within a gesture can end capture early; camera shake can
trigger capture. Keep the camera steady and use **Next gesture** to retry.
Captures exceeding six seconds or the 256-frame limit are rejected without
classifying a truncated sequence. A frame-delivery pause longer than one second
resets an unfinished capture. Completed results remain held until reset or stop.

Missing hands clear the rolling window and held-pose predictions as before.
Webcam accuracy and automatic boundaries must be evaluated separately from the
prepared test split; physical camera interaction has not been manually verified
by the agent. Capture changes do not repair the saved greetings checkpoint's
[normalization issue](docs/diagnostics/greetings-vs-colors.md).

Camera API and timestamp conventions follow
[browser camera requirements](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
and [MediaPipe's Python hand-landmark guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python).

Under **Datasets → Prepare a dataset → Source selection**, select **Choose source
folders** and check any folders to include beneath each gesture (S1, S2, S3, S4,
PUB, or other names found in that category). Enable **Videos only** to exclude
images and landmark arrays. The preview updates counts per gesture; gestures
with no matching files are omitted when a filter is active. All source folders
and supported file types remain the default. Selected subfolders are searched
recursively; files directly inside a gesture are included only in All source
folders mode. Selection leaves original files untouched and is recorded in
preprocessing settings, plus the extraction report for raw media. A Group CSV must cover exactly the
selected supported sources; folder selection itself does not assign signer or
session groups. Choose a fresh dataset name for this preparation.

The equivalent CLI options can be repeated for any folder names:

```powershell
python categories.py prepare --category colors --dataset-name selected_sources --source-folder S1 --source-folder S2 --source-folder S3 --videos-only
```

Dashboard checks (no ML dependencies needed):
`python -m unittest discover -s tests/dashboard -v` and
`node --test tests/dashboard/ui.test.cjs`.

## Categories: one model for each

Put raw images, videos or normalized .npy landmarks under data/<category>/<gesture>/. Nested folders within a gesture are supported for raw-media preparation.

```text
data/
  alphabet/
    A/image001.jpg
    B/image001.png
    J/video001.mp4
  numbers/
    1/sample001.npy
    2/sample001.npy
  shapes/
  colors/
  greetings/
  family/
```

The six named category folders are created. Add the remaining categories as new folders when their names are known; the code supports any number, including eight. Inside each category create folders for the actual gesture labels. For example, alphabet is the model category, and A/B/J are its classes.

Preparation automatically extracts landmarks from JPG/PNG images and MP4 videos (other supported formats below). Each category needs at least two gesture classes and at least three independent source groups per class.

## Quick start

Open a terminal in this folder. Verified with Python 3.12 on Windows CPU; requirements.txt pins the tested versions.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe categories.py list
.\.venv\Scripts\python.exe categories.py prepare --category alphabet --dataset-name prepared_v1
.\.venv\Scripts\python.exe categories.py train --category alphabet --dataset-name prepared_v1 --run first
.\.venv\Scripts\python.exe trainer.py predict --run runs\alphabet\first --sample data\alphabet\A\image001.jpg
```

To prepare and train every populated category:

```powershell
.\.venv\Scripts\python.exe categories.py prepare --all --dataset-name prepared_v1
.\.venv\Scripts\python.exe categories.py train --all --dataset-name prepared_v1 --run first
```

Repeat --category to select several, for example --category alphabet --category numbers. --all skips empty categories and reports them. Explicitly selecting an empty category returns an error. A failed category is reported while other selected jobs continue; retry failed categories explicitly after fixing their data.

Each category starts a fresh model in a separate process, using the same Conv1D–BiLSTM architecture with an output size determined by that category's gesture classes. Models run sequentially.

Outputs are separate:

```text
datasets_augmented/alphabet/prepared_v1/{train,validation,test}/<gesture>/*.npy
datasets_augmented/numbers/prepared_v1/{train,validation,test}/<gesture>/*.npy
runs/alphabet/first/best.keras
runs/alphabet/first/classes.json
runs/alphabet/first/test_metrics.json
runs/numbers/first/best.keras
runs/numbers/first/classes.json
runs/numbers/first/test_metrics.json
```

Other training artifacts are stored alongside each model. Prediction uses the chosen category's run directory; there is no automatic category classifier. Category defaults resolve relative to the repository root, so commands work from another working directory when you pass the script's full path.

Existing outputs are protected. Use a new --run name for retraining (omitting it generates a timestamp). To prepare another dataset version, use a new `--dataset-name` on category prepare and select that name on category train. Preparation defaults to `prepared_v1` beneath `datasets_augmented/<category>/`. Training can infer the dataset only when exactly one exists; otherwise specify `--dataset-name`. Use `--dataset-name .` to train an original category-level dataset. The `--datasets` option still changes the dataset root, while `--data` and `--runs` change source and model roots.

Category preparation supports --static-variants 8, --jitter 0.0075, --validation 0.15, --test 0.15 and --seed 42. For signer/session groups, pass --groups-dir groups with groups/alphabet.csv, groups/numbers.csv, etc. CSV paths are relative to each category, for example A/image001.npy. Each model's split is independent; cross-category signer separation is not imposed.

Training supports --epochs 100, --batch-size 32, --learning-rate 0.001, --patience 12, --seed 42 and --class-weights. Run each command with --help for options.

The original trainer.py prepare/train/inspect commands remain available for one dataset. In the lower-level examples below, raw means one category's source folder, such as data/alphabet.

## Raw image and video preparation

The existing prepare commands now detect and process raw media automatically. Install the updated requirements in the Python environment you use for preparation:

```powershell
python -m pip install -r requirements.txt
python categories.py prepare --category alphabet
python categories.py train --category alphabet --run first
```

Supported images: .jpg, .jpeg, .png, .bmp, .webp, .tif, .tiff. Supported videos: .mp4, .avi, .mov, .mkv, .webm, .m4v, subject to the installed decoder. Extension matching is case-insensitive. Animated/multipage images are rejected; convert them to video. Other files (including .npz) are listed as ignored in extraction_report.json.

MediaPipe HandLandmarker extracts up to two hands. The official model downloads automatically once into models/hand_landmarker.task; subsequent use is local. You may supply --hand-model PATH for a custom compatible .task asset. Raw media is processed on your machine and is never uploaded.

Images are decoded with EXIF orientation applied and produce one (1,128) vector. Videos use a fresh tracking session per clip, decode every frame, and produce (T,128) before the existing 32-frame interpolation. Supply constant-frame-rate videos: the timeline uses frame index and reported FPS. Unreadable/truncated videos and invalid FPS are rejected. A video with fewer than two decodable frames is rejected rather than treated as a static image.

Coordinates use MediaPipe's image-normalized landmarks: X relative to image width, Y relative to height, and wrist-relative Z at approximately the X scale. There is no extra centering or hand-size scaling, preserving XY trajectory and relative placement. Returned Left/Right labels determine the slots, independent of detection order. If the detector reports the same handedness twice, only the highest-confidence detection fills that slot. Handedness is estimated and can be wrong; keep image mirroring consistent across training and inference. --swap-hands reverses reported slots consistently without flipping pixels.

Detection and presence confidence default to 0.5; change with --detection-confidence (0 to 1). Missing hands have zero coordinates and binary zero flags. Images with no detected hand, and videos with no hand in any frame, are rejected and reported. Partly detected videos retain all decoded frames and expose detected_frames in the report.

After extraction, static images still generate 8 Gaussian-jitter variants by default, videos still interpolate to 32 frames, and every sample remains (32,128). All synthetic siblings share the original raw path as variant_of. Roboflow filenames sharing a stem before .rf. are grouped; common names such as name_jpg.rf.HASH.jpg are also grouped with name.jpg. This is a filename heuristic: use --groups-dir CSVs for other known source/signer relationships. CSV paths now refer to original raw filenames, not temporary arrays.

Identical extracted arrays within the same class are deduplicated and reported as rejected duplicates. Identical arrays with conflicting class labels fail preparation. After rejections, all classes still need at least three independent groups. If this fails, a timestamped extraction-failed report is saved beside the requested output directory.

Successful datasets additionally contain:

- extraction_report.json: accepted/rejected/ignored sources and video detection coverage.
- extracted/<class>/<id>.npy: original-length extracted arrays, before interpolation/jitter.
- extracted_sources.json: mapping from those IDs back to original raw paths.
- preprocessing.json: extraction settings, model checksum and MediaPipe version for reproducibility.

Prediction accepts an image, video or .npy sample. For raw media it uses saved extraction settings, verifies the landmark model checksum, and repeats a single image without random jitter at inference. If a custom .task asset was used, pass the same asset with --hand-model. Models trained only from pre-extracted .npy inputs still require .npy prediction input because their upstream normalization cannot be inferred.

Implementation reference: [MediaPipe Hand Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python).

## Pre-extracted .npy input contract

Place one original extracted recording/image in each .npy file:

```text
raw/
  A/
    image001.npy
    image002.npy
    image003.npy
  J/
    video001.npy
    video002.npy
    video003.npy
```

Class names are arbitrary folder names. Each file contains a real, finite numeric array:

- Video/recording: (T, 128), with T >= 2.
- Static image: (128,) or (1, 128). A single frame identifies a static source.
- Columns 0–62: left-hand landmarks 0–20, flattened as x0,y0,z0,x1,y1,z1,...,x20,y20,z20.
- Columns 63–125: right-hand landmarks in the same order.
- Column 126: left-present flag. Column 127: right-present flag.
- Flags must be exactly 0 or 1. All 63 coordinates of an absent slot must be zero.

For supplied .npy files, the upstream extractor must normalize coordinates consistently and maintain left/right slot identity across time. The .npy loader does not guess normalization, handedness, or presence from zero coordinates. If mixing raw media and .npy in one category, supplied arrays must use the MediaPipe coordinate convention above. Use the same extraction convention at inference. Z may be negative; valid coordinates are not clipped to [0,1]. Entirely empty samples, nonfinite values and incompatible shapes are rejected.

The feature calculation is 2 × 21 × 3 + 2 = 128. The per-hand sequence is (32,21,3), flattened into the combined (32,128) array.

## Uniform video resampling

Every recording maps to 32 equally spaced points between its first and last frame. Each coordinate uses linear interpolation, preserving endpoints. There is no frame-count-based dropping/repetition for videos. Binary presence flags use nearest-neighbor selection rather than fractional interpolation; output coordinates of absent hand slots are zeroed.

When detection changes between present and absent, coordinate interpolation spans the zero-filled absent frame. This convention is explicit and consistent; it cannot recover a hand missed by the upstream detector. The input format assumes uniformly sampled frames because it contains no timestamps.

## Static-image repeat + jitter

Each original static image generates **8 variants** by default, in whichever split its source belongs to. Each variant repeats the 128-value source vector for 32 timesteps and adds independent Gaussian noise to every present-hand coordinate at each timestep:

- Mean: 0.
- Default standard deviation: **0.0075**, or 0.75% of a unit normalized coordinate scale.
- Suggested requested range: 0.005–0.01 (0.5–1%).
- Presence flags and all absent-hand coordinates remain unchanged.
- No rotation, scaling, translation transform, temporal smoothing or coordinate clipping.
- Random results are reproducible with the same source files, options and seed.

Example: prepare --source raw --output dataset --static-variants 8 --jitter 0.005

Each manifest row includes a shared **variant_of** source ID. Splitting occurs before synthesis, so every sibling remains in the same train/validation/test split. Static images are never interpreted as a particular class name; A, J, Z or any other label follows the same source-shape rules.

Noise prevents literally identical timesteps, but does not create real motion or guarantee that a model cannot distinguish synthetic and real sequences.

## Splits and provenance

Default split targets are 70% train / 15% validation / 15% test, based on independent source groups before synthesis. Integer rounding and group sizes can change the exact proportions. At least three independent groups per class and at least two classes are required. Eight variants from one image still count as one independent source.

By default each original file is a group. If several recordings share a signer or session, supply a CSV so related recordings stay together across all classes:

```csv
path,group
A/image001.npy,signer01
J/video001.npy,signer01
A/image002.npy,signer02
J/video002.npy,signer02
A/image003.npy,signer03
J/video003.npy,signer03
```

Use: prepare --source raw --output dataset --groups groups.csv

The CSV must cover every source file exactly once with a nonempty group ID. Group splitting requires every class in every split and fails rather than allowing leakage. For shared groups spanning classes, a seeded search chooses a balanced feasible assignment; unusually constrained datasets may require more groups or different fractions.

In .npy-only preparation, byte-identical numeric source arrays are rejected as duplicates. File-level grouping cannot identify visually similar images or clips from the same recording; use group IDs for known relationships. Feed original source arrays to prepare, not pre-generated synthetic variants.

Prepared output:

```text
dataset/
  train/<class>/<sample>.npy
  validation/<class>/<sample>.npy
  test/<class>/<sample>.npy
  manifest.json
  preprocessing.json
```

All prepared samples are float32 (32,128). Inspection validates shape, values, manifest coverage, class coverage and disjoint group/source/variant_of IDs.

## Architecture

Matches the supplied image, in order:

| Layer | Configuration |
|---|---|
| Input | 32 × 128 |
| Conv1D | 64 filters, kernel 3 |
| Batch normalization | momentum 0.99, epsilon 0.0001 |
| MaxPooling1D | pool size 2 |
| Dropout | 0.3 |
| Conv1D | 128 filters, kernel 3 |
| Batch normalization | momentum 0.99, epsilon 0.0001 |
| Dropout | 0.3 |
| Bidirectional LSTM | 128 units per direction, return_sequences=True |
| Dropout | 0.3 |
| Bidirectional LSTM | 64 units per direction, return_sequences=False |
| Dropout | 0.3 |
| Dense | 64 units |
| Output | one probability per gesture class |

Settings not specified in the image: ReLU for Conv1D and hidden Dense, same convolution padding, default LSTM tanh/sigmoid activations, softmax output, Adam optimizer, sparse categorical cross-entropy. The model has 485,888 + 65 × number_of_classes parameters, including batch-normalization state.

TensorFlow references: [Conv1D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D), [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization), [recurrent layers](https://www.tensorflow.org/guide/keras/working_with_rnns).

## Training, evaluation and export

By default the tf.data pipeline reads pre-batched data from consolidated memory-mapped arrays, shuffles training sample indices only, and prefetches one batch. The manifest determines a stable alphabetical class-to-index mapping saved as classes.json.

Early stopping, learning-rate reduction and best-model selection use validation loss only. The held-out test split is evaluated after loading the best saved checkpoint. Optional --class-weights computes inverse-frequency weights using training counts only.

Each run contains:

- best.keras: the selected model, usable with tf.keras.models.load_model.
- classes.json: class names in output-index order.
- preprocessing.json: input layout, resampling and synthesis settings.
- training_config.json and split_manifest.json: settings, framework version and exact sample provenance.
- model_summary.txt: layer shapes and parameter counts.
- history.csv and history.json: per-epoch metrics.
- test_metrics.json: test loss, accuracy, macro F1, per-class precision/recall/F1/support and confusion matrix.
- test_predictions.csv: sample path, true label, predicted label and confidence.

Keep best.keras, classes.json and preprocessing.json together. The predict command accepts one normalized source array, resamples it to (32,128), and prints all class probabilities. A single-frame prediction is repeated without stochastic augmentation for deterministic inference.

Test metrics are per sequence; synthetic siblings are correlated, not independent new observations. Classes with more static variants contribute more sequences to aggregate accuracy. Use macro F1 and collect independent real recordings to judge real-world performance. No gesture-recognition accuracy is claimed from generated test fixtures.

## Tests

Run `python -m unittest discover -v` to include both trainer and category tests. Category tests check discovery, separate output paths, empty/unknown categories, overwrite protection, and failure reporting.

The suite checks interpolation on shorter/longer recordings, binary presence and missing-hand invariants, independent Gaussian noise scale, reproducibility, group splitting, repeat+jitter preparation with 8 siblings, leakage detection, invalid input and confusion-matrix calculations.

With TensorFlow installed it also checks architecture parameters, runs a training batch, and verifies a saved/reloaded model gives matching predictions. Without TensorFlow that test is explicitly skipped; preprocessing remains usable with NumPy alone.

## Preparation progress messages

Long preparation stages report their name and elapsed time every 10 seconds, with counts and percentages when available. Messages cover source checking, splitting, sequence generation, validation, copying extracted landmarks, and cleanup. Preparation complete is printed only after final validation and cleanup finish. Progress goes to stderr and is flushed immediately.


## Optimized training loader

The default --loader packed validates and consolidates existing sequences once into dataset/_training_cache/<fingerprint>/, with one X and y array per split. Packing happens without rerunning raw extraction or generating new jitter variants. Every feature value, label, class order and split is preserved.

Subsequent epochs open two consolidated arrays per split and read whole batches through memory mapping. This avoids per-sample np.load calls and validation during every epoch. A small index permutation shuffles training examples, and only one batch is prefetched, avoiding the previous 10,000-sequence shuffle buffer. The training architecture, batch-size default, number of samples and optimization objective are unchanged; exact random ordering can differ.

The first run prints packing and validation progress. Later runs scan source filenames, sizes and modification times and reuse the completed cache. Manifest or preprocessing changes also invalidate it. Keep datasets immutable during training. This is metadata-based invalidation, not a cryptographic rehash of every source array on every run. A changed dataset gets a new cache directory; old caches are retained. The 80,768-sequence dataset needs roughly 1.32 GB of extra disk space per complete cache.

To continue your cancelled alphabet checkpoint with the optimized loader:

```powershell
python categories.py train --category alphabet --run optimized --resume-from runs\alphabet\first\best.keras
```

Choose a new --run name if optimized already exists. Without --resume-from, training starts a new model. A continuation restores saved weights and optimizer state, checks model architecture/class order/preprocessing, and restarts early-stopping counters. --epochs counts additional epochs in the new run; it is not an exact interrupted-step resume. The existing first run is retained.

Each epoch prints measured duration including validation, plus an approximate remaining time if all requested epochs run. Durations are saved in epoch_times.json; this projection can change with laptop load, caching and early stopping. actual_optimizer_learning_rate in training_config.json records the loaded optimizer's current rate when continuing.

Use --loader files only to return to the original per-file path. Packed loading removes avoidable loading overhead, but CPU computation still depends on the full specified Conv1D–BiLSTM workload. No full-dataset speedup or GPU acceleration is claimed from the fixture tests.

## FSL Pocket PWA sample

See [pwa-sample/README.md](pwa-sample/README.md) for the client-side category downloader and camera recognition demo. After dependency setup, start it with `npm start` inside `pwa-sample`. Trained checkpoints and existing browser exports are included. Export additional runs with `pwa-sample/tools/export_model.py`.


## Confusion matrices
After test evaluation, every training run saves `confusion_matrix.png` (counts), `confusion_matrix_normalized.png` (percentages within each actual class), and matching CSV files. Rows are actual classes; columns are predictions. The diagonal contains correct predictions. Blank cells contain zero samples. These results describe the prepared test split, not live webcam accuracy.

Install the updated plotting dependency with `python -m pip install -r requirements.txt` in your training environment. Training commands are unchanged.

For a completed run, generate the plots without retraining:

```powershell
python confusion_matrix.py --run runs/alphabet/optimized
```

## Video and sequence augmentation

Preparation can create extra timing variations from videos and pre-extracted
multi-frame `.npy` landmarks. It is disabled by default. Enable it with:

```powershell
python categories.py prepare --category alphabet --dataset-name video_v1 --video-extra-samples 3 --video-time-warp 0.05
python trainer.py inspect --dataset datasets_augmented/alphabet/video_v1
python categories.py train --category alphabet --dataset-name video_v1 --run video_augmented_v1
```

`--video-extra-samples` accepts integers 0–1000; zero disables the feature.
Three requests up to three extras per eligible training source, alongside the
original sample. Validation/test videos and existing static-image augmentation
keep their current behavior. All descendants remain in their source group's
split and retain their gesture label.

`--video-time-warp` defaults to 0.05 and accepts finite values greater than
zero through 0.10. It bounds the timing shift to 5% of the normalized clip
timeline by default, while retaining endpoints and movement order. The transform
changes timing within the gesture, with no rotation, mirroring, noise, or new
video files. Prepared samples remain `(32,128)`.

The dashboard exposes Extra samples and Timing strength under dataset
preparation. Set Extra samples to 3 to start; the existing Random seed controls
repeatability. Activity shows actual generated totals and shortfalls. Motionless
or sparsely detected sources may yield fewer valid, distinct samples than
requested; originals are retained and duplicates are skipped.

Enabled datasets save `video_augmentation_report.json` with source-level
requests, actual counts, skipped candidates, and shortfall reasons. The manifest
records each generated sample's time map; `preprocessing.json` records the
version and settings. `augmentation_summary.json` separates generated
sequence samples from static variants. Training copies the augmentation report
into its run folder. Prediction continues to use ordinary preprocessing.

Choose a fresh dataset output. The existing resume command requires exactly
matching preprocessing metadata; changing augmentation settings therefore calls
for a fresh training run. The initial timing limits are settings to evaluate,
not a recognition-accuracy claim. See the [agreed design](docs/design/video-augmentation.md).
