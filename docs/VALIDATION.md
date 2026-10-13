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
