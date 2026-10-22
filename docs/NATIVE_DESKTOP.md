# FSL Studio native desktop

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\.venv\Scripts\python.exe dashboard.py
```

To open the app by double-clicking on Windows, build the launcher once:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_windows_launcher.ps1
```

Open **FSL Studio.exe** beside `dashboard.py`, or create a shortcut to it on your
desktop. The launcher finds the project relative to its own location, so a
shortcut can use any **Start in** folder. It uses `.venv\Scripts\python.exe`
without showing a console; training and camera workers use that same interpreter.
The project, virtual environment, and installed dependencies are required. When
moving to another computer, follow the repository setup to recreate `.venv`.

Each launch writes a UTF-8 log under `outputs/launcher/`. A failed launch shows
an error dialog with the log location and a dependency installation command.
The executable is generated locally and ignored by Git; rebuild it after a fresh
clone or changes to `tools/windows/StudioLauncher.cs`. Building uses the .NET
Framework 4.x compiler included with Windows and does not download packages.

The interface uses PySide6 / Qt Widgets, without HTML, JavaScript, WebView2,
Electron, or an HTTP server in the desktop launch path. Python workers still
handle training, preparation, export, and inference using the same environment
and artifact formats as the CLI. The ML dependencies remain in `requirements.txt`.

Qt provides [native widgets](https://doc.qt.io/qtforpython-6/gettingstarted.html)
and [dockable panels](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QDockWidget.html).
Only Qt Essentials is required; QtWebEngine is not installed by the desktop requirements.

## Workspaces

| Workspace | Controls |
|---|---|
| Extractor | Browse category/class folders, preview images/videos/`.npy` sequences, play and seek frames, optionally preview MediaPipe hand landmarks, configure extraction/splits/augmentation, and create a named prepared dataset. |
| Trainer | Select a category and prepared dataset, validate it, configure training, watch epoch progress and training/validation accuracy and loss, and stop a worker. |
| Model runs | Compare saved accuracy/F1, inspect provenance and class labels, view saved learning curves, record decisions/notes, save Keras ZIPs, build HUDYAT bundles, and export records/benchmarks. |
| Prediction | Recognize a project sample, use a local camera in held-pose or movement modes, configure capture timing, and save recognition results as independent prediction tests. |

The explorer, inspector, output panel, and toolbar are native widgets. Drag panel
titles to rearrange or float them. Use **View** to show hidden panels or reset the
layout. Window geometry, dock positions, selected workspace, and source/dataset
roots are stored locally in `outputs/desktop.ini`.

**Import category** copies a folder containing gesture class subfolders into a
new category under the configured source root. For example, select a folder with
`A/`, `B/`, and other class folders inside it. Import does not merge into an
existing category. For an existing category, add files in File Explorer and
refresh. Source and dataset roots must be paths inside the project, matching the
existing job validation. Opening an external file for preview does not import it.

Use a new dataset name and run name when outputs already exist. The underlying
job service protects previous outputs. Stopping a job keeps partial artifacts;
closing the app while a worker runs asks whether to stop it and exit. Camera
and preparation/training jobs share the existing exclusive worker rules.

In **Model runs**, choose a model and click **Build HUDYAT bundle…**. Enter an
**Export folder name**, then confirm to start conversion. For example,
`alphabet-v2` saves `exports/hudyat/alphabet-v2/alphabet-v2.zip`. Use **Show exported
bundle** in **Output / Jobs** to open that folder. Existing ZIPs are protected; choose another
name to export that model again. Cancel closes the prompt without starting a job.

## Monitors

Source playback shows actual decoded frames. The hand-overlay checkbox uses the
default two-hand MediaPipe detector for preview; extraction jobs use their own
configured confidence, hand model, and coordinate convention. The current model
input is **32 frames × 128 features** (two hands and presence flags). Body-pose
extraction is not part of this model.

Training curves read the flushed `history.csv` after every epoch, with
`history.json` as a fallback for saved runs. Preparation progress comes from
worker stage messages. There are no simulated metrics, GPU meters, or job results.
The frame slider navigates samples; it does not edit annotation intervals.

The camera opens only when **Start camera** is selected. The larger preview has a
draggable divider above Recognition so you can adjust how much space it uses.
**Fullscreen camera** opens a full-screen native preview with camera controls;
press **Esc** to return to the workspace. The camera keeps running when you return.

**Mirror preview** flips the display immediately, including in fullscreen, and
remembers your choice locally. Model input stays unmirrored. Capture mode, camera
index, and timing values are applied at start; restart the camera after changing
them. Physical camera availability depends on device selection and Windows camera
permissions. A live camera cannot run while a training/preparation/export job runs.

Preview mirroring is separate from data augmentation. Synthetic static training
variants under **Hand-relative sequence** preprocessing have a 50% mirror chance,
including left/right hand-slot and presence-flag swaps. Automatic preprocessing
selects that convention for raw alphabet media. Clean originals and validation/test
samples remain unmirrored. **Add opposite-hand video copies** separately enables
paired mirrored copies of complete training video sequences and timing variants,
in either feature convention. It works with zero timing extras and defaults off.
Only enable it for signs whose mirrored form has the same meaning, then prepare
a new dataset and retrain. Legacy image-coordinate static augmentation uses jitter
without mirroring. Existing
prepared datasets and trained models are not changed by the preview toggle.

Model record export includes existing saved prediction tests. The advanced
automated trial recorder and HUDYAT browser timing screen remain in the optional
browser dashboard. Checkpoint continuation remains available through the trainer
CLI. These are separate tools; the native app does not open or embed them.

## Shortcuts

| Shortcut | Action |
|---|---|
| Ctrl+1 / 2 / 3 / 4 | Extractor / Trainer / Model runs / Prediction |
| Ctrl+O | Open a sample for preview |
| Ctrl+Enter | Main action for the selected workspace |
| F5 | Rescan source folders, datasets, and runs |

## Verification and previews

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/dashboard -v
.\.venv\Scripts\python.exe tools/preview_desktop.py
```

The preview tool renders all four real workspaces offscreen into
`outputs/native-desktop/`. It reads project artifacts without starting training,
preparation, or a camera. Pass `--sample data/category/gesture/clip.mp4` to include
a media preview. Native interaction checks use temporary project fixtures. Qt
tests are skipped in environments that have not installed the desktop requirement.

For the previous browser dashboard only, run `dashboard.py --browser --port 8090`.
The browser mode remains independent of the Qt dependency.

## Signer holdouts and deletion

The extraction inspector includes **Signer evaluation**. Choose **Leave one signer
out**, a test signer, and an optional validation signer from the signer subfolders.
The Trainer displays the selected dataset's saved fold and audits it before use.
See [LOSO preparation and training](LEAVE_ONE_SIGNER_OUT.md).

Use **Delete selected model…** in the model inspector or **Delete selected
dataset…** in Run configuration. The confirmation shows the exact path; files move
to recoverable `.trash/`. See [deletion and recovery](DELETING_SAVED_ARTIFACTS.md).
