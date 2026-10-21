# Leave-one-signer-out preparation and training

Use the same signer ID beneath every gesture folder:

```text
data/colors/blue/S1/clip01.mp4
data/colors/red/S1/clip01.mp4
data/colors/blue/S2/clip01.mp4
data/colors/red/S2/clip01.mp4
data/colors/blue/S3/clip01.mp4
data/colors/red/S3/clip01.mp4
```

Images, videos, and image-coordinate `.npy` sequences support this layout, including
recording/session subfolders beneath the signer. Hand-relative preparation still
requires raw media. Folder IDs represent people: use the same ID for all sessions
and both hands of one person. A public-source folder is not a signer ID unless all
its recordings are known to come from that person. Use source-folder selection to
exclude sources with unknown signers. Names must use consistent capitalization.

In the native preparation inspector or browser preparation form, choose **Leave
one signer out**, select the **held-out test signer**, and optionally select a
**validation signer**. By default, validation uses the next signer in sorted order,
wrapping around after the final signer. All remaining signers train the model.
The validation/test fractions are ignored in this mode.

At least three signers are required. Training, validation, and test must each cover
every included gesture; missing coverage produces an error. A fold holds out all
recordings of its test signer, across all gestures. Its validation signer is also
separate from training. Source groups, source files, and synthetic siblings cannot
cross splits. Splits are assigned before augmentation. Validation and test each
contain one clean resampled sequence per accepted source, including static poses
with original image coordinates. Timing, opposite-hand, rotation, and jitter
augmentation are restricted to training in LOSO mode.

Prepare a **new dataset name for each fold**, such as `colors-loso-S1`. Select that
prepared dataset in the Trainer and start a new run. Training automatically reads
and audits the saved signer protocol, including when reusing packed data. The run
copies the protocol and signer-tagged manifest, so its saved results identify the
held-out signer. Checkpoint continuation requires exactly the same preprocessing
and fold; weights must not be transferred from a different signer fold.

One fold estimates performance on its selected unseen signer. Repeat once per
signer for full LOSO cross-validation. Group holdouts are the approach described
in the [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html#leave-one-group-out).

## Command line

One fold:

```powershell
.\.venv\Scripts\python.exe categories.py prepare --category colors --dataset-name colors-loso-S1 --split-strategy loso --test-signer S1 --validation-signer S2
.\.venv\Scripts\python.exe categories.py train --category colors --dataset-name colors-loso-S1 --run colors-loso-S1-model
```

The lower-level `trainer.py prepare` accepts the same three signer options with
`--source` and `--output`. Existing source-selection and augmentation options work
with both entry points. The default **Source groups** strategy retains its previous
behavior; existing datasets are not converted automatically.

To prepare and train **all folds automatically**:

```powershell
.\.venv\Scripts\python.exe categories.py prepare --category colors --dataset-name study1 --loso-all --source-folder S1 --source-folder S2 --source-folder S3
.\.venv\Scripts\python.exe categories.py train --category colors --dataset-name study1 --run experiment1 --loso-all
```

This creates `study1-loso-S1`, `study1-loso-S2`, and `study1-loso-S3` datasets, then
fresh `experiment1-loso-S1`, `experiment1-loso-S2`, and `experiment1-loso-S3` runs.
Use `--all` instead of `--category` for multiple categories. Preparation extracts
media independently for each fold. A failed category/fold leaves successful outputs
available; retry a failed fold using the single-fold commands and its exact names.
The training batch rejects incomplete fold sets and folds with different source
fingerprints or preparation settings. `--resume-from` is disabled for the all-fold
batch to prevent sharing weights between folds.

After retrying a failed fold, refresh the overall report without training again:

```powershell
.\.venv\Scripts\python.exe categories.py train --category colors --dataset-name study1 --run experiment1 --loso-all --report-only
```

Each dataset and model paper report includes signer allocation, original-source
coverage by signer/gesture, and a signer-isolation audit. All-fold training also
saves `reports/paper/loso/<category>/<run>/index.html`, with per-signer score tables
and figures, equal-weight fold means, and sample standard deviations. Score tables
use percentages; standard deviations are expressed on the percentage-point scale.
Incomplete rotations show available fold evidence without an overall mean or SD.
Individual model runs retain the numeric metrics and full confusion matrices.

These controls prepare the evaluation protocol; they do not guarantee a target
recognition accuracy. Collect all gestures from multiple real signers and evaluate
the trained models using the saved holdouts.
