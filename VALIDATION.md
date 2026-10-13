# Verification

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
