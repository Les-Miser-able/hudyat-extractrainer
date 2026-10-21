# Opposite-hand video augmentation

Dataset preparation can generate opposite-hand training examples from recordings
using one hand. Enable **Add opposite-hand video copies** in the native app or
browser dashboard, or pass `--video-mirror-hands` to `trainer.py prepare` or
`categories.py prepare`. It defaults off and is independent of timing extras.
Only enable it for a vocabulary whose mirrored gestures preserve their labels.

Each eligible multi-frame training source retains its original clean sequence.
Timing augmentation runs as before. Mirroring then adds a paired copy of the clean
sequence and each accepted timing variant. With three accepted timing variants,
there are at most eight sequences: four original orientation, four mirrored.
With zero timing extras there are at most two. Single-frame/static augmentation
keeps its existing behavior; the new control applies to multi-frame sequences.

The pure `mirror_sequence` transform reflects x coordinates consistently across
all frames and exchanges the left/right 63-coordinate slots and presence flags.
Image coordinates use `1 - x`; hand-relative finger coordinates and wrist
trajectories use `-x`. The transform keeps y/z, frame order, capture progression,
and missing-hand zeros. It does not modify the original array or reverse time.
No modified videos are written; the generated artifacts are landmark sequences.

Mirrored candidates are deduplicated against the clean sequence, accepted timing
variants, and other accepted mirrors for that source. Symmetric two-hand sequences
may not need a separate copy. Skipped duplicates and shortfalls are reported;
duplicates are never added merely to meet a requested count. Timing randomness is
unchanged by enabling mirroring, and the transform consumes no additional random
numbers from static augmentation.

Source groups are assigned before augmentation. Every derived sequence keeps its
original source, label, group, hash, and split. Only training sequences gain these
new variants. Validation/test sources are evaluated as captured; the setting
cannot establish unseen-hand or unseen-signer performance by itself.

Enabled datasets save version 2 video-augmentation settings with `mirror_hands`
and `mirror_normalization`. Disabled datasets keep the version 1 settings and
existing preparation outputs. The manifest identifies each mirror as `hand_mirror`
and records `mirror_of_variant` (zero for the clean original), so the complete
transformation can be reconstructed from its parent. Raw-media preparation restores
original filenames in both the manifest and augmentation report.

`video_augmentation_report.json` records total requested/generated/unfilled extras,
plus separate timing and mirror counts and duplicate counts. The requested mirror
budget is one plus the requested timing budget; missing timing parents also leave
their requested mirrors unfilled. `augmentation_summary.json` exposes these counts
overall, per sign, and per split. Dataset paper reports include CSV tables and
PNG/SVG charts for mirrored clean/timing sequences by sign and mirrored counts by
split. Model records and copied preprocessing settings retain the enabled flag.

Prepare a new named dataset and fit a new model to use these examples. Existing
datasets/checkpoints are not converted, and inference applies no augmentation.
Actual left- and right-hand test attempts are required to assess recognition
accuracy; synthetic examples alone do not establish that result.

Validation covers both coordinate conventions, left/right and two-hand inputs,
missing frames, inverse reflection, movement direction, unchanged originals and
timing streams, symmetric duplicates, split isolation, raw-media provenance, both
CLIs, desktop/browser controls, paper report counts, and disabled-mode regressions.
