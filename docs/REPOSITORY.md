# Repository setup

This repository contains the Python trainer, tests, documentation, augmentation
scaffolding, PWA source, trained Keras checkpoints, their supporting metadata and
training results, existing browser model exports, and the MediaPipe hand detector.
Original recordings, prepared datasets, large split manifests, per-sample prediction
CSVs, browser vendor libraries, virtual environments, and editor settings stay local.
The PWA source lives in `pwa-sample/dist/`; do not ignore that entire folder.

## Fresh clone on Windows

Install Python 3.12 and Node.js 22 or newer. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Add your own media to `data/<category>/<gesture>/`, then prepare and train:

```powershell
.\.venv\Scripts\python.exe categories.py prepare --category family --dataset-name prepared_v1
.\.venv\Scripts\python.exe categories.py train --category family --dataset-name prepared_v1 --run first
```

The MediaPipe hand model and existing trained checkpoints are included. Source
data must be supplied separately to prepare datasets or retrain on another machine.
Prepared datasets live under `datasets_augmented/<category>/<dataset name>/`.
Use a fresh dataset name/run name when existing outputs are present.

Included runs: `alphabet/{first,optimized,webcam_v2,augmented_v1}`,
`colors/{first,augmented_v1}`, and `family/augmented_v1`. Each checkpoint ships
with `classes.json` and `preprocessing.json` for prediction and browser export.
The PWA catalog currently selects the existing alphabet and colors exports;
including additional Keras checkpoints does not automatically select/export them.

## PWA setup

```powershell
cd pwa-sample
npm ci
npm run setup
npm start
```

Setup preserves the included catalog and exports. If no catalog exists, it creates
one with categories unavailable. To add the included family checkpoint, run from
the repository root:

```powershell
.\.venv\Scripts\python.exe pwa-sample/tools/export_model.py --category family --run runs/family/augmented_v1
```

Open http://localhost:8080. For website deployment, publish the complete local
`pwa-sample/dist/` after setup and export, including its Git-ignored assets.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
cd pwa-sample
npm test
npm run parity
```

Run the PWA tests after setup. Model integrity checks and parity require exported
models; an empty catalog cannot validate a trained model's predictions.

## First push

Create an empty repository with your Git hosting provider, then from this root:

```powershell
git status --short
git diff --cached --stat
git commit -m "Initial FSL trainer and PWA"
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_REPOSITORY_URL` with your repository's clone URL. Before committing,
check the staged file list; original media and generated datasets should not
appear, but trained checkpoints and browser exports should. Root-level npm manifests are legacy local files; use npm only inside
`pwa-sample/`.

## Project layout

Python implementation lives in one package, organized by responsibility:

```text
fsl_trainer/
  data/              Source extraction and per-category preparation/training commands
  training/          Dataset preparation/validation, model training/prediction, disk cache
  reporting/         Confusion-matrix rendering from saved results
  dashboard/
    server.py        Local HTTP interface, artifact browsing, and worker jobs
    ui/              Dashboard HTML, CSS, and JavaScript
  paths.py           Repository root for default artifact locations
  progress.py        Shared terminal progress reporting
tests/
  data/              Category and extraction checks
  training/          Preprocessing, model, and training-cache checks
  dashboard/         HTTP, job, and JavaScript control checks
  test_compatibility.py
pwa-sample/          Independent webcam application, export tools, and its tests
docs/                Project setup and verification records
learning/            Lessons, learning notes, resources, and their browser assets
```

The root `trainer.py`, `categories.py`, `dashboard.py`, and
`confusion_matrix.py` remain executable entry points. Existing imports of those
names, `extraction`, `training_cache`, and `progress` resolve to their package
implementations, so callers and patches share the same module globals. New
Python code should import from `fsl_trainer` directly. The PWA exporter follows
this convention; workers retain the established root CLI entry points.

Artifact folders retain their existing locations: `data/`,
`datasets_augmented/` (or a chosen dataset root), `runs/`, and `models/`.
Default project paths do not depend on the current working directory. Dashboard
assets are resolved beside its server implementation and retain the same URLs.
Learning materials live together in `learning/`: `lessons/`, `assets/`,
`reference/`, and the mission, notes, and resources documents. Start with
[the learning index](../learning/README.md). `augmentation/` retains its
development scaffolding and output conventions.

The root retains `README.md`, dependency/Git configuration, and compatibility
Python entry points. Existing `data/`, `datasets_augmented/`, `runs/`, and
`models/` paths remain the defaults used by commands and saved artifacts.

Run checks from the repository root:

```powershell
python -m unittest discover -v
python -m unittest discover -s tests/dashboard -v
node --test tests/dashboard/ui.test.cjs
cd pwa-sample
npm test
npm run parity
```

The full Python suite uses the environment with the ML requirements installed.
The dashboard-only Python and JavaScript checks need no ML dependencies.
PWA tests and parity require the setup assets and browser exports described above.
