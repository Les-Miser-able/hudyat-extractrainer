# Verification

## Larger native camera and preview mirroring — 2026-10-20

- Enlarged the camera area, made Recognition compact with a draggable divider,
  and added a native fullscreen preview with Start/Stop/Next controls and Esc to
  return. Mirror preview updates live in both views and persists locally.
- Mirroring is a display transform: pixel checks verify the rendered left/right
  reversal while the original frame remains unchanged. Fullscreen checks verify
  incoming frame updates, synchronized mirror controls, and Escape behavior.
- All 100 dashboard checks and 15 existing hand-relative preprocessing checks
  passed. Visually verified the normal and fullscreen layouts using a local video
  sample with the camera off. Physical webcam hardware was not exercised.
- Augmentation behavior remains unchanged: synthetic static hand-relative training
  variants can mirror with hand-slot swaps; video variants use timing changes only.

## Native camera timing request fix — 2026-10-20

- Reproduced the reported `Timing must contain only settle_ms and missing_ms.`
  error through the native camera worker and the real `LiveInference.frame`
  validator. The camera incorrectly included its local frame interval in the
  gesture timing request.
- Kept frame cadence in the native camera loop and forwarded only gesture delays
  to inference. Strengthened the existing camera regression to exercise the real
  timing validation instead of mocking that boundary. Verified the test fails
  before the fix and passes afterward; all 98 dashboard tests pass.
- The regression simulates camera frames; physical webcam recognition was not
  exercised. Restart the running desktop app to load the corrected worker code.

## Native Qt desktop workspaces — 2026-10-20

- Replaced the default WebView2 host with native PySide6 Qt Widgets. Launch tests
  verify that desktop mode creates no HTTP server and browser mode does not load Qt.
- All 98 dashboard tests passed on Python 3.13.13, including native widget rendering,
  dataset selection, existing command validation, live CSV chart updates, cancellation,
  model review persistence, saved prediction tests, refresh preservation, camera
  transport with a simulated device, import protections, and worker shutdown.
- Rendered and visually inspected all four actual workspaces offscreen using the
  project's 9 category folders and 22 saved runs. Decoded an existing 90-frame video,
  sought to frame 21, and verified the actual MediaPipe hand overlay. Preview PNGs
  are in `outputs/native-desktop/`; `tools/preview_desktop.py` recreates them.
- `pip check` and `git diff --check` passed. Real datasets, checkpoints, and review
  records were read without starting training or modifying saved model artifacts.
  Physical webcam behavior was not exercised. The advanced browser-only trial
  recorder and HUDYAT timing page remain separate optional tools.

## In-dashboard model comparison and manual records — 2026-10-14

- Expanded Model runs into a comparison table with saved dataset/source details,
  distinct originals and prepared split counts, training/preparation parameters,
  class coverage, offline scores, recorded live timings, and editable notes.
- Manual decisions and live observations persist atomically in
  runs/model_records.json. Selecting Preferred replaces the previous preferred
  run in the same category without discarding its notes. Unknown runs and invalid
  fields are rejected. An unreadable record file is not silently overwritten.
- Record current settings saves the camera mode/sliders without interrupting a
  live session. Try model restores recorded settings. A stale project refresh
  cannot overwrite a newer saved record in the UI.
- All 54 dashboard Python and 26 JavaScript tests passed. Checks cover persistence,
  preferred selection, zero-correct trials, invalid input, failed atomic writes,
  unchanged checkpoints/metrics, HTTP token validation, escaped notes, filtering,
  record editing, settings capture/restoration, and refresh ordering.
- Read-only inspection loaded all 14 real saved runs, including legacy CSV epoch
  history and distinct original source counts for recent runs. No manual choices
  were made for the user's real runs. Browser/native surfaces were unavailable
  for visual verification; physical camera recognition was not re-evaluated.

## Live recognition timing sliders — 2026-10-14

- Added editable sliders for Stillness delay (100–2,000 ms), Hands missing
  delay (100–2,000 ms), and Frame interval (20–500 ms), defaulting to
  600/400/50 ms. Values and labels update immediately; capture settings travel
  with the next serial camera request and modify the existing capture without
  resetting buffered frames or restarting its worker/model.
- Shared validation rejects unsupported keys and invalid capture timings before
  worker exchange. Resetting or automatically rearming capture preserves its
  timing settings. Increased the bounded buffer to 512 frames so the fastest
  selectable interval can retain a full gesture within the six-second limit.
- All 47 dashboard Python tests and 20 JavaScript checks passed. New checks
  exercise live changes during stillness and missing hands, retention of early
  movement, worker reuse, HTTP validation, slider label/request/cadence updates,
  and a five-second gesture at the minimum 20 ms interval.
- Browser control was unavailable for visual inspection in this session.
  Physical camera behavior and recognition accuracy were not measured.

## Faster movement recognition — 2026-10-14

- Reduced completion thresholds from 800 to 600 ms for stillness and from 600
  to 400 ms for missing hands. The buffered sequence and held result behavior
  remain intact. Pauses or detection gaps reaching these thresholds can finish
  capture; physical-camera accuracy needs separate evaluation.
- Camera requests remain serial, with a 50 ms target interval that includes
  processing time. Replaced the unconditional 200 ms wait after each response;
  slower processing schedules the next frame without an extra fixed pause.
- Before/after replay through LivePredictor with deterministic 50 ms landmark
  frames measured completion delays of 800 -> 600 ms (stillness) and
  600 -> 400 ms (missing hands). These measure capture logic, not end-to-end
  physical webcam or trained-model inference latency.
- Timing regression checks failed before the change and passed afterward.
  All 42 dashboard Python tests and 19 JavaScript UI tests passed, including
  movement retention, brief tracking gaps, consecutive captures, serial frame
  requests, and scheduling with fast and slow processing. Reproduce with
  the dashboard commands recorded below.

## Movement retention after hands leave — 2026-10-14

- Moving gesture now buffers complete movement using the existing capture
  boundaries instead of repeatedly classifying a sliding two-second tail.
  Brief missing-hand gaps remain in the sequence; 600 ms without hands
  completes capture. The result remains visible while waiting for another
  movement. Perform once still requires explicit reset; held pose is unchanged.
- The real LivePredictor path was tested using deterministic landmark frames
  and a sequence-sensitive fixture classifier. Before the change, holding the
  ending replaced movement with a pose-only prediction, and removing hands
  immediately discarded the buffer. Both regressions now pass, along with
  brief-gap recovery and consecutive-movement capture.
- All 40 dashboard Python tests and 18 JavaScript UI tests passed. Reproduce
  with `.\.venv\Scripts\python.exe -m unittest discover -s tests/dashboard -v`
  and `node --test tests/dashboard/ui.test.cjs`.
- Physical camera behavior and real greetings accuracy were not measured in
  this check. Saved training data, checkpoints, and evaluation metrics were
  not modified.

## Named prepared datasets — 2026-10-14

- Dashboard preparation now targets `<dataset root>/<category>/<dataset name>`.
  The dashboard and category CLI share named/legacy dataset discovery. Training
  and validation pass the selected dataset directory to the existing trainer;
  prediction continues to use the selected run's preprocessing metadata.
- Passed 34 dashboard Python checks, 24 category/preparation/source-filtering
  checks, and 17 JavaScript UI checks. The new cases cover editable names, exact
  training/validation selection, multiple versions, invalid names/paths,
  overwrite protection, original datasets, and independent training caches.
- Real preparation from normalized landmark fixtures succeeded through both the
  category CLI and native desktop UI. Per-version validation succeeded through
  the desktop UI. An original category-level dataset still validated after a
  named child dataset was prepared; its original files and cache were unchanged.
- A native WebView2 check exercised the editable name and output preview,
  preparation, validation buttons, training selection, and selection persistence
  after refresh. Dataset and training screens were captured and inspected.
  Training used a lightweight worker after verifying its real command's dataset
  and run paths; full TensorFlow training and physical camera capture were not
  repeated for this path-layout change. No existing user datasets were moved.

## Desktop dashboard verification — 2026-10-14

- All 28 dashboard Python tests passed on Windows with Python 3.13.13, including
  desktop launch/exit, browser mode without desktop dependencies, startup failures,
  occupied ports, close cancellation/confirmation, bounded worker termination,
  partial-output preservation, and shutdown during a pending camera response.
- All 12 dashboard JavaScript tests passed. Reproduce with
  `python -m unittest discover -s tests/dashboard -v` and
  `node --test tests/dashboard/ui.test.cjs`.
- A native pywebview 6.2.1 / WebView2 smoke check opened the real desktop window
  against disposable project data, navigated all five views, resized the window,
  started a lightweight subprocess through the local job API, and verified its
  logs appeared in the UI. The captured WebView2 rendering was visually inspected.
  The window then closed and its server thread exited.
- The embedded page exposes the camera API in a secure local context. Physical
  camera capture, camera permissions, and full preparation/training/prediction
  were not exercised in this desktop check; the isolated verification environment
  contains desktop dependencies only. Existing ML verification below predates
  this desktop change.

## Portable layout verification — 2026-10-06

- All 24 Python tests passed with no skips. The original 22 tests also passed before restructuring.
- A fixed preparation fixture matches all 48 original file hashes: 45 arrays,
  the split manifest, preprocessing metadata, and augmentation summary.
- All 15 directory moves passed before/after inventory comparisons covering paths,
  sizes, and file modification times. Existing dataset metadata was not rewritten.
  The relocated detector asset also matches its original Git blob hash.
- A temporary copy in a path containing spaces was installed in editable mode
  without network access. Commands ran from outside that copy, using its own
  default raw, prepared, cache, run, and detector locations.
- The copied fixture completed preparation, inspection, one epoch of packed
  training, one epoch of checkpoint resume, and prediction. The existing
  alphabet/first checkpoint also loaded, predicted, and exported successfully.
- All 10 PWA tests passed. Existing alphabet and colors exports passed Python /
  TensorFlow.js parity with maximum probability differences of approximately
  4.17e-7 and 5.96e-8, respectively.
- Real datasets were moved, not regenerated or retrained. Existing runs, browser
  exports, and pre-existing catalog changes were preserved. Temporary verification
  data and models were removed.

Reproduce with `python -m unittest discover -v` and
`python tools/verify_portability.py` after editable installation. The integration
check requires the included alphabet/first checkpoint and detector. For browser
checks, run `npm test` and `npm run parity` inside `pwa-sample` (use `npm.cmd` on
Windows if PowerShell blocks the npm script).

## Earlier verification

Verified on Windows CPU with Python 3.12, NumPy 2.3.5, TensorFlow 2.21.0, Keras 3.15.1, MediaPipe 1.0.1, OpenCV contrib 5.0.0.93 and Pillow 12.3.0.

- All 19 unittest tests passed; no skips. Extraction tests were rerun after final provenance/report changes and passed.
- Category integration verified independent datasets, labels, saved models and evaluation reports.
- Architecture, finite training batch, and matching save/reload predictions verified.
- Raw-media integration used six sample images from two classes, generated 48 static sequences, completed a one-epoch functional training check, saved the model and predicted directly from a raw image.
- Video decoding and MediaPipe VIDEO tracking verified with an 8-frame local test clip.
- Original raw paths, shared variant_of IDs, Roboflow source grouping, missing slots, rejection reports and ignored-extension reports checked.

These are code-functionality checks only. The full user dataset has not been prepared or trained. Test data and test weights remain outside this project; no recognition-accuracy claim is made.

## Training optimization verification

- All 22 unit tests passed, including exact packed-array values and label alignment, shuffled sample coverage, cache reuse without revalidation, source-change invalidation, and leakage rejection.
- Small-fixture integration completed two epochs with packed loading, evaluation and export; a second run resumed the checkpoint for two more epochs and completed successfully.
- Epoch timings and optimizer learning-rate metadata were saved.
- The user's cancelled training run, checkpoint and prepared dataset were preserved. The full dataset cache has not been built and real-data training has not been restarted.
- Fixture loader timings are not an estimate of full-model training speed; measure the new epoch_times.json on the user's next run.
