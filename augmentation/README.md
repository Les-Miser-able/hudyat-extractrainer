# Data augmentation workspace

Executable video/sequence timing augmentation lives in
`fsl_trainer/data/augmentation.py`. This folder retains scaffolding for future
presets, previews, reports, and additional experiments; a preset loader is not
implemented.

```text
data/<category>/<gesture>/       Original images, videos, or landmark arrays
augmentation/
  configs/                      Future augmentation presets by category
  transforms/                   Future image, video, and landmark transforms
  tests/                        Future transform and integration tests
  previews/                     Generated before/after previews (ignored by Git)
  reports/                      Generated augmentation reports (ignored by Git)
datasets_augmented/             Separate prepared dataset output root
  <category>/                   Created by preparation, not pre-populated
    train/<gesture>/
    validation/<gesture>/
    test/<gesture>/
    manifest.json
    preprocessing.json
```

Keep original inputs in the existing `data/` folders. Do not put generated
variants beneath `data/`: preparation scans those folders for source samples.
Keep presets and implementation files here, outside the source-data tree.

## Existing augmentation

`trainer.py` already repeats static poses to 32 frames and applies Gaussian
landmark jitter. It assigns source groups to splits before creating variants;
the current implementation generates static variants in all three splits.
These existing options can write into the new output root now:

```powershell
python categories.py prepare --category alphabet --datasets datasets_augmented --static-variants 8 --jitter 0.0075
python categories.py train --category alphabet --datasets datasets_augmented --run augmented_v1
```

Preparation creates category folders itself and requires each category output
to be empty. For another experiment use a fresh root such as
`--datasets datasets_augmented/v2` for both commands. Training writes to the
existing `runs/<category>/<run>/` layout.

Successful preparation prints accepted original-source counts, augmented static
sequence counts, resampled video/sequence counts, and total output sequences,
plus totals per split and gesture. It also saves `augmentation_summary.json`
inside the prepared category folder. Rejected or ignored sources are excluded.
Eight static variants means eight outputs per static source, not nine. Videos
and multi-frame landmark inputs produce one resampled original by default.
Optional timing augmentation adds distinct training samples as described below.
Existing prepared datasets are not changed automatically.

## Video / sequence timing augmentation

```powershell
python categories.py prepare --category alphabet --datasets datasets_video_v1 --video-extra-samples 3 --video-time-warp 0.05
```

Extra samples range from 0 to 1000 (default 0, off). Timing strength is greater
than 0 through 0.10, default 0.05. Each eligible training source keeps its original
and gains up to the requested number of distinct, valid timing variations.
Validation/test videos are unchanged. Static augmentation retains its existing
behavior in all splits. No new video files are written.

The seeded transform changes progression within the full gesture, preserving
endpoints, movement order, hand slots, presence flags, and `(32,128)` arrays.
Ten attempts per requested extra bound duplicate/invalid candidate retries. A
shortfall is reported instead of padding with duplicates or failing solely for
missing extras. Reports use original raw filenames when extraction is involved.

Enabled preparation records settings in `preprocessing.json`, actual timing maps
in the manifest, separate generated-sequence counts in the summary, and detailed
counts/reasons in `video_augmentation_report.json`. Training retains the report
in its run directory. Inference applies no augmentation. See the
[feature design](../docs/design/video-augmentation.md) for the complete contract.

## Future implementation

- Keep executable transforms with `fsl_trainer/data/` and integration checks
  with the project tests. Put documented presets under `configs/` once a loader
  is implemented.
- Integrate new augmentation after source-group split assignment. Apply new
  random transforms to training samples only; keep evaluation preprocessing
  deterministic. Changing existing validation/test jitter is a separate code change.
- Preserve source/group identity in the dataset manifest so variants from one
  source or signer cannot cross splits. Record the seed and transform settings
  in preprocessing metadata.
- Preserve gesture labels, left/right slot conventions, binary presence flags,
  and zero coordinates for absent hands. Prepared arrays must remain `(32, 128)`.
- Write previews and reports under their respective folders using an experiment
  name, for example `previews/alphabet_v1/` and `reports/alphabet_v1/`.

The `.gitkeep` files retain empty directories in Git. Generated previews,
reports, and prepared datasets are ignored.
