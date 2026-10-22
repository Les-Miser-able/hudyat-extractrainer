# Download and export models for HUDYAT

In **Model runs**, each trained model has these actions:

- **Download model ZIP** saves the Keras checkpoint, class labels, preprocessing,
  recorded model details and test history, and available training reports.
- **Export for HUDYAT** opens an export form. Choose an **Export folder name**.
  The hand detector is matched to
  training automatically, or you can supply a project-relative `.task` path.
  **Create HUDYAT bundle** runs conversion in Activity. When complete, choose
  **Download HUDYAT bundle** there.

In the native desktop app, **Build HUDYAT bundle…** prompts for the folder name
before starting the job. For example, `alphabet-v2` saves the ZIP at
`exports/hudyat/alphabet-v2/alphabet-v2.zip`; browser downloads use
`alphabet-v2.zip` too. Names can contain letters, numbers, spaces, hyphens,
underscores, and dots, up to 100 characters. Choose a new name for each export;
existing ZIPs are never overwritten. The model version folders inside the bundle
retain their content hashes for HUDYAT's importer.

The HUDYAT ZIP contains versioned browser weights, metadata, parity fixtures,
the matching MediaPipe detector, and a catalog with just the selected category.
It also includes the model's recorded description and testing history. When
available, `benchmark.json` and `benchmark.csv` carry the held-out metrics
and the full camera timing from the trainer's HUDYAT camera test. Local inference
is recorded separately. `camera_timing.json` contains the measured trials and
source/device details. These reports do not change the model/runtime contract.

Extract the ZIP. In its extracted folder, run:

```powershell
python install_hudyat.py --project "D:/path/to/your/HUDYAT"
```

Replace `D:/path/to/your/HUDYAT` with the actual project location on the recipient's
computer. Select the root containing both `client/` and `server/`, not the
extracted bundle or the `server/` folder alone. The trainer, bundle, and HUDYAT
can be in different locations. The installer uses the path passed to `--project`;
it does not require the exporter's username, drive, or desktop layout.

The importer verifies hashes before writing, keeps other catalog categories,
and installs the camera preprocessing adapter when needed. The adapter supports
the current HUDYAT runtime with lab settings as well as the earlier category
runtime. It preserves HUDYAT's confidence and margin thresholds, hold times,
camera options, baseline/enhanced profiles, and capture settings. Both legacy image
coordinates and `hand-relative-sequence-v1` are supported. Existing HUDYAT models
using `hand-relative-v1` keep their original per-frame normalization. It leaves the
database untouched. To activate the imported release, run from HUDYAT/server:

```powershell
npm run models:register
```

Rebuild/restart HUDYAT and re-enter the category online. The registration command
uses HUDYAT's existing category matching and registers all available entries in
its merged catalog. No database migrations are required by this export flow.

Before registering a release, run this from the extracted bundle with HUDYAT's
client dependencies installed:

```powershell
node verify-bundle.mjs . "D:/path/to/your/HUDYAT"
```

Verification runs the selected checkout's actual camera runtime in baseline and
enhanced configurations. It checks landmark packing, preprocessing, model inputs,
and prediction parity against the Python export. Camera frames and asset transport
are simulated; it does not open the camera or access the database. It also checks
the sign-capture input path and preserves the app's existing acceptance gates.
The runtime test requires the TypeScript dependency from HUDYAT's client project.

Sequence normalization preserves wrist trajectories and anatomical left/right
slots. The adapter uses HUDYAT's sign-window trimming and supplies the sequence
features directly, rather than applying the pose model's automatic hand mirroring.
Alphabet still follows HUDYAT's held-pose recognition mode. This integration does
not turn that mode into movement capture for J/Z or establish real-camera accuracy.

HUDYAT still captures held poses for Alphabet and two-second clips for other
categories. A successful export or parity check does not measure camera accuracy
or establish motion recognition for J/Z.

For command-line export from the trainer folder:

```powershell
.\.venv\Scripts\python.exe -B tools/export_hudyat_model.py --run "runs/alphabet/alphabet-prototype1-MODEL" --output "exports/hudyat/alphabet-v2/alphabet-v2.zip"
```

To apply only the camera preprocessing update to an existing HUDYAT checkout:

```powershell
.\.venv\Scripts\python.exe -B integrations/hudyat/install_hudyat.py --project "D:/path/to/your/HUDYAT" --runtime-only
```

To verify an extracted export with HUDYAT's installed TensorFlow.js and real model
loader, without starting its server or accessing its database:

```powershell
node tools/verify_hudyat_export.mjs "path/to/extracted-bundle" "D:/path/to/your/HUDYAT"
```

If installation reports changed camera code, a different preprocessing integration,
or a different test command, the recipient's HUDYAT code differs from the
integration this bundle supports. Those checks are separate from the installation
location. Record the exact error to identify the necessary compatibility update.

Generated ZIPs are saved under `exports/hudyat/`. Activity download links last
for the current dashboard server session; the ZIP files remain after restart.
In the Python FSL Studio window, Download opens a Windows Save As dialog so you
can choose the destination. It normally starts in your Windows Downloads folder.
In browser mode, downloads follow your browser's download settings.
