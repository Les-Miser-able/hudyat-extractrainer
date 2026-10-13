# Verification

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

## Project reorganization — 2026-10-13

Verified after moving Python implementations into `fsl_trainer/` and grouping
checks under `tests/`. Existing root commands and imports remain supported.

- `py -3.11 -B -m unittest discover -v`: all 35 tests passed, with no skips.
  This includes preprocessing, extraction, split isolation, packed-cache reuse,
  dashboard HTTP/jobs, legacy imports, commands from another working directory,
  and TensorFlow training plus model save/reload.
- `py -3.14 -B -m unittest discover -s tests/dashboard -v`: all 9 tests passed
  in an environment without NumPy or TensorFlow. HTTP tests use a temporary
  project and copied dashboard assets, avoiding scans of developer datasets.
- `node --test tests/dashboard/ui.test.cjs`: all 3 tests passed.
- A temporary synthetic-data CLI check passed category preparation, dataset
  inspection, one-epoch category training with the packed cache, checkpoint
  prediction, and confusion-matrix rendering from another working directory.
  All generated data/models/reports were confined to a temporary folder.
- `py -3.11 -B pwa-sample/tools/export_model.py --help` passed after updating
  the exporter to import the package implementation.
- `git diff --check` passed.

PWA checks retain the failures observed before the reorganization: 8 of 10
JavaScript tests pass; the offline-shell check lacks `dist/shell-assets.json`,
and the exported-model integrity check reports a catalog/model checksum
mismatch. Parity cannot start because `@tensorflow/tfjs` is not installed.
The PWA setup instructions in `REPOSITORY.md` describe its local prerequisites.
The reorganization did not regenerate browser exports or change their catalog.

## Documentation and learning layout

Project setup and verification records now live under `docs/`. The learning
mission, teaching notes, resources, HTML lessons, reference pages, and shared
styles/quiz scripts live together under `learning/`. `learning/README.md`
provides an index. README, PWA setup, lesson, and resource links were updated.

All 34 local documentation and lesson links were checked, including stylesheet
and quiz-script paths. The mission and teaching notes were preserved byte for
byte. `git diff --check` passed. This follow-up moved documentation and learning
materials only; the Python implementation and its artifact paths were unchanged.

## Video augmentation implementation

- Final `py -3.11 -B -m unittest discover -q`: all 47 tests passed, no skips.
- `py -3.14 -B -m unittest discover -s tests/dashboard -v`: all 10 checks passed
  without NumPy or TensorFlow installed in that environment.
- `node --test tests/dashboard/ui.test.cjs`: all 4 checks passed, including
  submission of the new preparation controls.
- Added preparation-interface checks for distinct training-only variants,
  original retention, unchanged mixed static outputs, disabled byte compatibility,
  source-group isolation, deterministic source streams, changed seeds, sparse
  detection, two-frame sources, signed-zero duplicate equivalence, retry
  shortfalls, option validation, report counts, and packed-cache behavior.
- Raw-media tests use a controlled extraction adapter to verify original filename
  restoration and unchanged original-length extracted arrays. Original video
  decoding and MediaPipe inference code were not changed by this feature.
- A temporary synthetic-data CLI workflow passed augmentation, dataset inspection,
  one-epoch cache-backed training, exact report copying into the run directory,
  and ordinary prediction. Generated artifacts remained in a temporary folder;
  the temporary smoke script was removed after verification.
- Python syntax checks parsed all 27 package/test files successfully. This
  repository has no configured static typechecker.
- All 31 local links in the updated guides and lessons were checked before the
  review/verification records were added. Staged whitespace checks passed.
- Independent [Standards and Spec reviews](design/video-augmentation-review.md)
  each reported zero actionable findings against the reorganization baseline.

The separate PWA checks still pass 8 of 10 tests. The existing missing
shell-assets.json and exported-model checksum mismatch remain; parity still
cannot start without the uninstalled @tensorflow/tfjs dependency. No original
media, existing datasets, saved runs, browser exports, or catalog were regenerated.

## Dashboard source-folder selection

- `py -3.11 -B -m unittest discover -q`: all 52 tests passed.
- `py -3.14 -B -m unittest discover -s tests/dashboard -q`: all 11 dashboard
  checks passed without ML dependencies.
- `node --test tests/dashboard/ui.test.cjs`: all 5 UI checks passed.
- New checks cover arbitrary immediate gesture subfolder choices (including S4),
  nested videos, exclusion of unchecked folders and non-video files, empty
  selections, omitted empty gestures, preview counts, CLI forwarding, raw-source
  provenance, training-only augmentation, and CSV group isolation across classes.
  Selected nested `.npy` inputs also prepare without invoking a media extractor.
- A read-only check of `data/colors` selected 180 videos from S1/S2/S3: 45 each
  for blue, green, orange, and red; violet and yellow had zero selected videos.
  Dashboard command construction used those same choices. No original media,
  existing datasets, or trained runs were changed, and extraction/training on
  this real dataset was not started.
- Working-tree whitespace checks passed. Changes remain uncommitted.

## Dashboard live camera prediction

- `py -3.11 -B -m unittest discover -q`: all 59 tests passed.
- `py -3.14 -B -m unittest discover -s tests/dashboard -q`: all 14 dashboard
  checks passed without ML imports in the server.
- `node --test tests/dashboard/ui.test.cjs`: all 9 UI checks passed, including
  permission denial, cancellation while permission is pending, serial frame
  submission, camera-track cleanup, job exclusion, and ignoring late predictions
  after stopping.
- Python checks cover worker reuse, stale sessions, startup/inference failure
  cleanup, authenticated endpoints, path validation, JPEG limits, timestamp
  validation, two-second window buffering, interpolation at capture timestamps,
  unchanged held-pose coordinates, missing-hand reset, and detector-asset mismatch.
- A read-only smoke check loaded `runs/colors/colors-run1-selfdata` in the actual
  persistent worker. Repeated no-hand video frames returned `no_hand`. A middle
  frame of a blue recording produced buffering followed by predictions on two
  successive rolling windows. This checks execution, not webcam accuracy or
  temporal gesture recognition quality. The worker was stopped after both checks.
- No camera was accessed, media or model files changed, or inference samples
  written to disk. Physical camera interaction and browser permission handling
  have not been manually verified. The independent PWA was unchanged.
- Changes remain saved and uncommitted; whitespace checks passed.

### Stale server live-route recovery

The user's running dashboard returned 404 for an authenticated `/api/live/start`
probe. Its process started before the live routes were written. All jobs had
finished; the activity history was saved to the OS temporary directory, then only
that dashboard process was restarted with its existing Python 3.11 environment.
The same probe reached the route afterward and returned the expected validation
error rather than 404. The UI now gives explicit restart instructions for missing
live routes, covered by a regression test that failed with the former generic
error and passed with the new message. All 10 dashboard UI tests passed.

## Automatic single-gesture camera capture

- `py -3.11 -B -m unittest discover -q`: all 65 tests passed in the final run.
  An earlier run encountered a Windows connection abort in an HTTP authorization
  check; that check passed in isolation. The HTTP test helper now closes error
  responses, and the final full suite passed.
- `py -3.14 -B -m unittest discover -s tests/dashboard -q`: all 15 dashboard
  checks passed without importing ML dependencies into the server.
- `node --test tests/dashboard/ui.test.cjs`: all 12 UI checks passed.
- Capture checks verify one prediction from a complete gesture longer than two
  seconds, retention of opening motion and the short missing-hand lead-in,
  unchanged 32x128 inference layout, held results, explicit reset, small-jitter
  rejection, brief detection gaps, completion after lowering hands, and rejection
  of long or interrupted captures. UI checks cover preserving a held prediction
  and ignoring a previous capture's late response after reset.
- Read-only replay through the actual persistent worker captured one existing S1
  good-afternoon clip over 2.8 seconds and one good-evening clip over 3 seconds.
  Each produced exactly one prediction followed by held responses. Both used the
  saved greetings checkpoint; this verifies capture and inference execution, not
  class accuracy. Camera hardware was not accessed, and no media, datasets, or
  saved model weights were modified.
- The dashboard was consolidated to one updated Python 3.11 instance after
  confirming the older instances had no active training or camera workers.
  Activity history was preserved in OS temporary files. Startup, reset, and stop
  for the new gesture mode were checked through the running server.
- Automatic boundaries use motion and stillness heuristics. Physical webcam
  behavior, sensitivity to movement speed/camera shake, and pauses within real
  gestures still require user trials. The greetings checkpoint's separately
  documented normalization issue remains unchanged.
- Changes remain saved and uncommitted; whitespace checks passed.
