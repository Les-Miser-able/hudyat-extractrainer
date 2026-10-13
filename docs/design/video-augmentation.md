# Video augmentation design

Status: implemented and verified. The feature follows the confirmed scope and
contract below. Verification results are recorded in [VALIDATION.md](../VALIDATION.md).

## Goal

Generate additional training samples from source videos while preserving
source-group isolation and existing functionality.

## Agreed scope

- Augment extracted hand landmark sequences; do not save modified video files.
- Generate new video variants only from the training split. Validation/test video
  samples and all existing static-image behavior remain unchanged.
- The feature is optional and disabled by default. Existing commands preserve
  their existing output behavior when augmentation is disabled.
- Provide configuration in both the command-line tools and the dashboard.
- Eligible inputs include raw videos and pre-extracted multi-frame .npy landmark
  sequences. Static inputs retain their existing behavior.
- Keep the original prepared multi-frame sample. When enabled, the initial
  selection requests three additional samples per eligible training recording;
  the extra count is configurable, with zero disabling augmentation.
- Timing strength defaults to a maximum normalized timing displacement of 0.05
  (5% of the clip timeline), adjustable above zero up to 0.10. Both endpoints and
  movement order are preserved. These are starting limits, not empirically
  optimized settings for the project gestures.
- Save only valid variants that differ from the original and from already saved
  variants of that source. If fewer than requested can be produced, retain the
  original, save the available variants, and report the shortfall.
- Controls are Extra samples and Timing strength, reusing the existing Seed.
  Report original, generated, and skipped counts in logs and saved reports.
  A visual preview is outside this first version.
- Version one uses timing variation only: preserve the full movement, its order,
  and its path while varying progression within the gesture. Spatial changes,
  coordinate noise, frame reversal, and hand swapping are outside this version.

## Verified current behavior

- Media extraction produces original-length landmark sequences before source
  groups are assigned to splits.
- Preparation generates output sequences after source-group assignment. A static
  source produces the requested jitter variants; each multi-frame source
  currently produces one resampled output.
- Every prepared sample has shape (32,128), with binary hand-presence flags and
  zero coordinates for absent hands.
- Static variants currently occur in all three splits; changing that behavior is
  outside the agreed video feature.
- The preparation seam currently distinguishes static and multi-frame inputs by
  sequence length; it does not retain raw-video versus multi-frame .npy kind.
  The agreed scope includes both kinds of multi-frame input.

Sources: [media extraction](../../fsl_trainer/data/extraction.py),
[dataset preparation](../../fsl_trainer/training/trainer.py), and
[existing augmentation conventions](../../augmentation/README.md).

## Final review

The user confirmed shared understanding of the complete design. The scope,
implementation contract, compatibility details, and acceptance checks below
are agreed. Implementation was subsequently authorized through the implement
skill. The implementation passed both review axes; see the
[review record](video-augmentation-review.md).

Terminology: [project glossary](../../GLOSSARY.md).

## Research and design notes

[Isolated Sign Language Recognition based on Tree Structure Skeleton Images](https://arxiv.org/pdf/2304.05403)
(section 4.5) evaluates spatial and temporal skeleton augmentation in a different
model representation. It provides primary-source context, not evidence for an
optimal transformation or parameter setting in this project.

In this project, resampling maps the full source sequence to 32 index-based
timesteps. Changing FPS metadata alone does not change those outputs. The timing
variation proposed in Q6 would change progression within the gesture while
retaining its endpoints and movement order; this transformation family is now
agreed, with a default displacement bound of 5% and a maximum of 10%.

Source inspection also confirmed that summary accounting currently equates
synthetic samples with static augmentation. Implementation must distinguish new
video variants so static counts and original-source totals stay meaningful.

## Implementation contract

### Preparation controls

| Control | CLI option | Default | Valid values |
| --- | --- | --- | --- |
| Extra samples per eligible training source | `--video-extra-samples N` | 0, disabled | Integers 0 through 1000 |
| Maximum normalized timing displacement | `--video-time-warp S` | 0.05 | Finite values above 0 through 0.10 |
| Reproducibility | Existing `--seed` | Existing default, 42 | Existing seed rules |

Use the same names, meaning, and limits in `trainer.py prepare`,
`categories.py prepare`, and dashboard command validation. Boolean, negative,
fractional sample counts and nonfinite/out-of-range strengths are rejected before
generation. The sample-count ceiling matches the existing dashboard static-variant
ceiling. When N is zero, no augmentation-specific metadata or report is added;
the preparation output contract remains the existing one.

The dashboard displays Extra samples and Timing strength in dataset preparation.
Zero extra samples means off; three is the suggested starting selection. Help text
states that extras apply only to training videos and multi-frame landmark inputs,
that originals are retained, and that the requested count is an upper bound.
No new training or prediction settings are required.

Example, from the repository root:

```powershell
python categories.py prepare --category alphabet --datasets datasets_video_v1 --video-extra-samples 3 --video-time-warp 0.05 --seed 42
python trainer.py inspect --dataset datasets_video_v1/alphabet
python categories.py train --category alphabet --datasets datasets_video_v1 --run video_augmented_v1
```

### Timing variation

Generate every candidate from the original-length source sequence, after the
source group has its split. Do not extract video landmarks again or warp a
previously generated variant.

Use a monotonic piecewise-linear time map with fixed endpoints at 0 and 1 and
interior reference anchors at 1/3 and 2/3. Independently displace each interior
anchor by a seeded uniform offset in [-S, S]. With S at most 0.10, the mapped
anchors remain strictly increasing. Interpolating between these anchors bounds
the timing displacement by S everywhere. This is a concrete initial algorithm;
its effectiveness still needs evaluation on the project's gesture recordings.

Sample this mapped timeline at 32 positions in the original sequence. Use linear
interpolation for XYZ coordinates, nearest-frame selection for binary presence,
and zero coordinates for absent hands, matching the current resampling rules.
The original first and last frames are retained. Do not reverse, crop, drop,
mirror, rotate, scale, translate, swap hands, or add coordinate noise.

### Original retention and distinctness

- Preserve the existing uniformly resampled original at variant index 0.
- For eligible training sources, request up to N additional samples. A source in
  validation/test still produces its existing original only.
- Every new sample must satisfy the existing finite float32 (32,128) contract,
  binary presence flags, absent-hand zeros, and some hand presence.
- A saved variant must differ from its own prepared original and from all saved
  variants of that source. Compare the final float32 arrays, not just time maps.
- Allow at most 10 candidate attempts per requested extra sample. Duplicate or
  invalid candidates are skipped. Exhaustion produces a reported shortfall,
  never duplicate padding and never a failure solely for missing extras.
- Existing failures for invalid sources, impossible group splits, or invalid
  original prepared samples retain their current behavior.

Ten eligible training sources produce 10 originals and up to 30 extra samples
with N=3. With sufficient distinct candidates the total is 40; a motionless
source may produce only its original. Synthetic samples never count toward the
minimum independent-source-group requirement.

### Reproducibility and isolation

Derive independent deterministic random streams from the existing seed, stable
relative source identity, and candidate identity. Use a stable digest, never
Python's process-randomized hash. New randomness must not consume the static
augmentation stream or change source-group split assignment. With identical
inputs and settings, variants, identities, and counts are repeatable.

All descendants retain their original label, source, source group, and
variant_of identity. Raw-media provenance restoration must update every new
source reference in manifests/reports to the original raw filename. Existing
original-length extracted arrays and extracted_sources.json remain source
artifacts, not destinations for generated variants.

### Metadata and reports

For enabled augmentation, record a versioned video_augmentation section in
preprocessing.json: eligible input kind, training-only scope, requested extras,
maximum timing displacement, seed, timing-map algorithm, and retry bound.
Record each generated sample's transformation version and actual anchor mapping
in its manifest entry so the sample can be explained and reproduced.

Use the existing static field to identify static sources and synthetic=true for
new generated multi-frame samples. Update summary classification so synthetic
video/sequence variants never inflate augmented_static_sequences. Keep original
source totals and original resampled-sequence totals meaningful. Add separate
counts for generated multi-frame samples, requested extras, and unfilled extras.
Include these counts by split and gesture as appropriate.

Save video_augmentation_report.json for enabled preparation, listing original
source paths, requested/generated/unfilled extras, duplicate/invalid candidate
counts, and reasons for shortfalls. The console and dashboard Activity stream
show original and generated totals and shortfalls; requests for N extras must
not be reported as N successfully generated extras when candidates were skipped.
The extraction report remains a report of original source extraction.

Training retains the new manifest/preprocessing metadata in run artifacts using
its existing copy behavior. Include the augmentation report in the run artifacts
when present. Inference remains extraction plus ordinary resampling; prediction
does not apply augmentation. Existing cache fingerprints already include the
manifest and preprocessing metadata and will distinguish the new dataset.

### Compatibility and placement

Use a fresh prepared-dataset output, following existing overwrite protection.
Do not modify original media, previously prepared datasets, or saved runs.
Retain existing root commands/imports, class ordering, model architecture,
static augmentation behavior, and prediction interfaces.

Keep the existing exact-preprocessing-equality resume rule. A dataset with
changed augmentation metadata is therefore incompatible with a checkpoint whose
preprocessing metadata differs; use a fresh training run. Relaxing that rule for
fine-tuning is separate work.

Place implementation with the package's data-preparation code and checks with
the existing data/training/dashboard tests. Update augmentation/README.md and
project/learning documentation to describe the implemented feature accurately.
No new dependencies, raw-video export, on-the-fly epoch augmentation, visual
preview, automatic dataset migration, or model-accuracy claim is part of v1.

## Acceptance evidence

- Disabled preparation preserves baseline arrays, manifests, metadata, counts,
  split assignment, and static random results. Existing tests continue to pass.
- Enabling video augmentation leaves static output arrays and source assignments
  unchanged, including in mixed image/video datasets.
- A moving source produces distinct valid variants with the agreed displacement
  bound, preserved endpoints, movement order, hand slots, and presence rules.
- Raw videos and pre-extracted multi-frame .npy sources follow the same policy.
- No augmented sample appears in validation/test or crosses source groups.
- Identical seeds reproduce results; changing the seed changes suitable moving
  samples. Source processing order does not change a source's augmentation stream.
- Motionless sources, sparse detection, two-frame sources, candidate duplicates,
  and retry exhaustion exercise the reported-shortfall path correctly.
- CLI/category/dashboard controls share validated values and forwarded settings.
- Original/generated/unfilled counts agree with files and manifest rows; provenance
  names original sources and static totals remain correct.
- Packed-cache coverage and invalidation, ordinary prediction, model save/reload,
  and current resume compatibility continue to work under their existing rules.
- Run the full Python suite and dashboard JavaScript checks, plus a temporary
  preparation/training/prediction smoke check. Existing unrelated PWA setup and
  checksum failures remain separately identified.

This evidence establishes feature correctness. Compare a baseline model with an
augmented-data model on the same original held-out sources in a separate training
experiment before claiming better recognition accuracy.
