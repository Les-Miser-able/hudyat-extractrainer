# FSL Trainer: step-by-step user guide

Use this guide to add sign images or videos, train a model, and try it in the browser.
Commands are written for **Windows PowerShell**. Run each command separately and wait
for it to finish before continuing.

**Already have a trained model and just want to try the camera? Go to [Step 8](#8-open-the-browser-app).**

## The workflow

```text
Copy original files → Prepare data → Train → Check results → Predict or export → Use camera app
```

Three words used in this guide:

| Word | Meaning | Example |
|---|---|---|
| Category | A collection of signs with its own model | `alphabet` |
| Gesture / class | A sign the model learns to identify | `A`, `B`, `J` |
| Run | One training attempt and its saved results | `guide_v1` |

Each category gets a separate model. You select the category when predicting;
the trainer does not automatically decide whether a sign is a letter, color, or number.

## 1. Open the project

Open the `fsl_trainer` folder in File Explorer. Click the address bar, type
`powershell`, and press Enter. This opens PowerShell in the correct folder.

Check that you can see the project files:

```powershell
Get-ChildItem
```

You should see `trainer.py`, `categories.py`, `src`, `data`, and `pwa-sample`.

**All commands below start from this folder**, except where a step explicitly
says to enter `pwa-sample`. Paths containing spaces need quotation marks.

## 2. Set up Python once per computer

This project has been tested with **Python 3.12 on Windows CPU**.
Check that Python 3.12 is installed:

```powershell
py -3.12 --version
```

If the command is unavailable, install Python 3.12 with the Windows Python
launcher, then reopen PowerShell.

For a fresh copy of the project, create its Python environment:

```powershell
py -3.12 -m venv .venv
```

Install the project and its dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Installation may take several minutes and needs internet access for missing
dependencies. Wait until it finishes without an error.

**On the computer where the trainer is already installed, skip environment creation.**
There is no need to reinstall before every training session. These commands use
the environment directly, so you do not need to activate it.

Check that the trainer starts:

```powershell
.\.venv\Scripts\python.exe categories.py list
```

Expected result: category names and source-file counts. Empty categories are allowed here.

## 3. Add your original images or videos

Copy your files into this layout using File Explorer:

```text
data/
  raw/
    alphabet/
      A/
        photo001.jpg
        photo002.jpg
        photo003.jpg
      B/
        photo001.jpg
        photo002.jpg
        photo003.jpg
      J/
        clip001.mp4
        clip002.mp4
        clip003.mp4
```

The folder immediately under `raw` is the category. The next folder is the correct
label for every file inside it. To add `colors`, use folders such as
`data/raw/colors/blue` and `data/raw/colors/red`.

- Use original JPG/PNG images or MP4 videos for the simplest workflow.
- A category needs at least **two gesture folders**, each with at least **three independent source groups** remaining after preparation rejects unusable files. This is a processing minimum, not enough to establish recognition quality.
- Renaming or copying one image several times does not create independent examples.
- Keep hands visible. Use video to capture signs whose motion matters.
- Keep all new originals under `data/raw`; do not put prepared or generated files there.
- If several files come from the same signer, session, or recording, read [Grouping related recordings](#grouping-related-recordings) before preparing them.

There is no upload page. Adding data means copying it into these folders.
Original files are preserved during preparation.

Check the counts again:

```powershell
.\.venv\Scripts\python.exe categories.py list
```

## 4. Prepare a dataset

The examples use **`alphabet`**, dataset version **`guide_v1`**, and run name
**`guide_v1`**. Replace `alphabet` consistently if you are working on another category.
If you have already used `guide_v1`, choose a fresh name such as `guide_v2`.

Run:

```powershell
.\.venv\Scripts\python.exe categories.py prepare --category alphabet --datasets data/processed/guide_v1
```

This command:

1. Reads originals from `data/raw/alphabet`.
2. Extracts hand landmarks and applies the existing preprocessing.
3. Separates source groups into training, validation, and test sets.
4. Converts each example into the model's 32-frame input format.
5. Saves everything under `data/processed/guide_v1/alphabet`.

Images generate eight slightly varied landmark sequences by default. Videos are
resampled to 32 frames. You do not need to run a separate normalization script.
The default split targets are 70% training, 15% validation, and 15% testing;
actual proportions depend on the source groups.

**Wait for preparation to finish successfully.** Progress messages appear during
long stages. A successful run ends with a completion message and an augmentation summary.

Check the prepared dataset:

```powershell
.\.venv\Scripts\python.exe trainer.py inspect --dataset data/processed/guide_v1/alphabet
```

Expected result: sample counts for each class in each split, with no validation error.

For raw-media inputs, review
`data/processed/guide_v1/alphabet/extraction_report.json` if files were rejected.
It records accepted, rejected, and ignored inputs. A failed extraction may instead
write a timestamped `alphabet.extraction-failed-....json` beside the output folder.

## 5. Train the model

Use the **same dataset path** you used during preparation:

```powershell
.\.venv\Scripts\python.exe categories.py train --category alphabet --datasets data/processed/guide_v1 --run guide_v1
```

The trainer saves results under `runs/alphabet/guide_v1`.
Training runs for up to 100 epochs by default, and may stop earlier when validation
loss stops improving. An epoch is one pass through the training examples.

The first training run may spend time building a cache before epochs begin.
This is normal. Later runs can reuse it.

For a **short pipeline check**, use a separate run name and fewer epochs:

```powershell
.\.venv\Scripts\python.exe categories.py train --category alphabet --datasets data/processed/guide_v1 --run guide_check --epochs 2 --batch-size 16
```

This checks that training works; it is not a substitute for training and evaluating
a useful model. You can then run the full command above to create `guide_v1`.

Wait until training, test evaluation, and saving results have all finished.
If you stop with **Ctrl+C**, the run may be incomplete. A checkpoint exists only
if the trainer has already saved one; see [Resume a checkpoint](#resume-a-checkpoint).

## 6. Read the results

Open `runs/alphabet/guide_v1` in File Explorer.

| File | What to use it for |
|---|---|
| `best.keras` | The saved model selected using validation loss |
| `classes.json` | The labels the model can recognize |
| `preprocessing.json` | The preprocessing settings needed with this model |
| `test_metrics.json` | Test accuracy, macro F1, and per-class results |
| `confusion_matrix.png` | See which actual signs were mistaken for other signs |
| `confusion_matrix_normalized.png` | The same comparison as percentages per actual class |
| `history.csv` | Training and validation results for each epoch |
| `epoch_times.json` | How long each epoch took |

In the confusion matrix, **rows are actual labels and columns are predictions**.
Correct predictions lie along the diagonal.

Accuracy is the fraction of test examples predicted correctly. Macro F1 summarizes
performance with equal weight for each class. Also inspect individual classes:
a good overall score can hide a poorly recognized sign.

Test scores are based on your prepared dataset. Check separately recorded examples
and webcam trials before drawing conclusions about real-world performance.
Synthetic variants of the same image are not independent real-world observations.

**Keep `best.keras`, `classes.json`, and `preprocessing.json` together.**
Keeping the entire run folder is the simplest way to preserve the results.

## 7. Predict one image or video

Replace the sample path with a file that actually exists:

```powershell
.\.venv\Scripts\python.exe trainer.py predict --run runs/alphabet/guide_v1 --sample "data/raw/alphabet/A/photo001.jpg"
```

For a video, use its filename instead:

```powershell
.\.venv\Scripts\python.exe trainer.py predict --run runs/alphabet/guide_v1 --sample "data/raw/alphabet/J/clip001.mp4"
```

The command prints labels and their scores, highest first. A score of `0.85` is
an 85% model score, not proof that the prediction is correct.
For a meaningful new-example check, use a recording that was not used for training.

You can also select an existing run instead of `guide_v1`.
Models trained only from supplied `.npy` landmarks require `.npy` prediction input;
they do not contain the raw-media extraction settings needed for images or video.

## 8. Open the browser app

You can do this without preparing data or training a new model.
This checkout already includes browser exports for **Alphabet** and **Colors**.

The browser app needs Node.js; the project setup specifies **Node.js 22 or newer**.
Check your installation:

```powershell
node --version
```

From the main project folder, enter the app folder:

```powershell
cd pwa-sample
```

On a fresh copy, install dependencies and prepare browser assets:

```powershell
npm.cmd ci
npm.cmd run setup
```

Start the app:

```powershell
npm.cmd start
```

Keep this terminal open and visit **http://localhost:8080** on the same computer.
On later visits, you only need to enter `pwa-sample` and run `npm.cmd start`.

In the app:

1. Select a category.
2. Click **Download model** and wait until it is ready. This saves the model in this browser.
3. Click **Turn on camera** and allow camera access.
4. Hold a static sign and click **Recognize held pose**, or click **Record 2-second gesture** and perform the complete movement.
5. For repeated alphabet predictions, click **Start live alphabet**. Use **Pause live** to pause it.
6. Click **Stop camera** when finished.

The current alphabet app notes that its checkpoint was trained on still images,
including J and Z. Recording movement does not itself make that checkpoint trained
to recognize moving signs.

To stop the server, press **Ctrl+C** in its terminal. To return to the main folder:

```powershell
cd ..
```

## 9. Put your newly trained model into the app

Run this from the **main project folder**, after your training run has completed:

```powershell
.\.venv\Scripts\python.exe pwa-sample/tools/export_model.py --category alphabet --run runs/alphabet/guide_v1
```

Exporting converts the saved model to the app's browser format and updates the
catalog to use that export for the selected category. It does not retrain the model.
Reload the app while the server is running, then download the updated category model.

To add another category, substitute its category and run, for example:

```powershell
.\.venv\Scripts\python.exe pwa-sample/tools/export_model.py --category family --run runs/family/augmented_v1
```

Category IDs for browser export must use lowercase letters, digits, hyphens, or
underscores. The exporter needs the compatible model architecture and raw-media
preprocessing metadata. A run made only from externally supplied `.npy` arrays
cannot automatically provide that metadata.

If training used a custom hand detector, supply the same file with
`--hand-model "path/to/detector.task"` when exporting or predicting raw media.

## 10. Common follow-up tasks

### Add more data and train again

Copy the new originals into the correct `data/raw` folders. Then prepare a new
dataset version and train a new run:

```powershell
.\.venv\Scripts\python.exe categories.py prepare --category alphabet --datasets data/processed/guide_v2
.\.venv\Scripts\python.exe categories.py train --category alphabet --datasets data/processed/guide_v2 --run guide_v2
```

Adding raw files does not update an existing prepared dataset, trained model, or
browser export. Repeat preparation, training, and export to include them.
Old datasets and runs remain available for comparison.

### Train using an already prepared dataset

Skip preparation. For the migrated default dataset:

```powershell
.\.venv\Scripts\python.exe categories.py train --category alphabet --datasets data/processed/default --run another_attempt
```

For an augmented dataset, use `--datasets data/processed/augmented` instead.
The folder name `augmented` does not turn on additional transforms; preparation
options determine what was generated.

### Resume a checkpoint

Use the same prepared dataset and a **new output run name**:

```powershell
.\.venv\Scripts\python.exe categories.py train --category alphabet --datasets data/processed/guide_v1 --run guide_resumed --resume-from runs/alphabet/guide_v1/best.keras --epochs 20
```

This continues from the saved best checkpoint, including its optimizer state,
for up to 20 additional epochs. Early-stopping counters restart. It is not an exact
restart at the instant training was interrupted. Class order, preprocessing, and
architecture must match. Only one category can be selected when resuming.

### Prepare and train all populated categories

Use an unused dataset version and run name:

```powershell
.\.venv\Scripts\python.exe categories.py prepare --all --datasets data/processed/all_v1
.\.venv\Scripts\python.exe categories.py train --all --datasets data/processed/all_v1 --run all_v1
```

Categories run one after another. Preparation skips empty source categories;
training discovers categories in the selected prepared-data folder. Review all
failure messages before training. After correcting a failed category, retry it
explicitly with `--category`, using a fresh output location if partial files remain.

### Grouping related recordings

If the same person, session, or original recording contributes related samples,
give them a shared group ID so they stay in one split. Do this before preparation.

Create `groups/alphabet.csv` in the project folder. For example:

```csv
path,group
A/photo001.jpg,signer01
B/photo001.jpg,signer01
A/photo002.jpg,signer02
B/photo002.jpg,signer02
A/photo003.jpg,signer03
B/photo003.jpg,signer03
```

Replace this example with **every supported source file in your category**, exactly
once. Paths are relative to `data/raw/alphabet`. Include enough independent groups
for every class to appear in all three splits.

```powershell
.\.venv\Scripts\python.exe categories.py prepare --category alphabet --groups-dir groups --datasets data/processed/grouped_v1
```

Train with `--datasets data/processed/grouped_v1`. Grouping is independent per category.

### Measure webcam recognition

In the app, select Alphabet, download its model, and enable the camera. Expand
**Measure webcam accuracy**, choose the **Actual letter**, recognize a sign, then
click **Save labeled trial**. Include incorrect and uncertain predictions.
Reposition between trials rather than repeatedly saving the same pose.

Click **Download test results** before switching categories or closing the page.
From the main project folder, convert the downloaded JSON into plots:

```powershell
.\.venv\Scripts\python.exe confusion_matrix.py --report "C:\path\to\webcam-test-alphabet.json" --output reports/webcam/alphabet_session1
```

Replace the example report path with the actual downloaded file. The plots appear
in `reports/webcam/alphabet_session1`.

## 11. Move or back up the project

1. Stop training and the local browser server.
2. Copy the project, including `src`, root launchers, `pyproject.toml`, `requirements.txt`, `assets`, and any `data`, `runs`, and `pwa-sample` files you need.
3. Leave out `.venv`, `node_modules`, `pwa-sample/node_modules`, and `__pycache__` folders. `data/cache` is also optional because the trainer can rebuild it.
4. On the new computer, open PowerShell in the copied project and repeat Step 2.
5. If using the browser app, repeat its dependency setup in Step 8.

Default folders follow the copied project. Explicit custom paths must point to
locations that exist on the new computer.

**Copying the project folder and cloning its Git repository are different.** Raw
data, prepared datasets, and caches are ignored by Git. Transfer those separately
if you use a Git clone. Keep your original raw recordings backed up.

The app can work offline after its assets and selected models are cached, but
browser storage belongs to that browser/device and can be cleared. Download models
again on a different device. For phone use, host the complete prepared
`pwa-sample/dist` folder on HTTPS; the local server is for this computer only.
See [the browser guide](pwa-sample/README.md#install-or-use-on-a-phone).

## 12. Troubleshooting

| Message or problem | What to do |
|---|---|
| `Output already contains data` / `Output must be empty` | Pick a new dataset version or run name. Do not delete successful results just to rerun a command. |
| Unknown category or no populated categories | Check `categories.py list`. Raw files belong inside gesture folders, not directly inside the category. For training, check the `--datasets` path contains the prepared category. |
| Need at least 3 independent source groups | Add independent usable recordings for that class. Check whether duplicates or failed detection reduced the usable count. |
| No hand detected / unreadable media | Read the extraction report. Check the file opens, hands are visible, and videos are complete. |
| Class has no samples | Remove an accidental empty gesture folder or add its examples. |
| `.venv\Scripts\python.exe` not found | Return to the main project folder. If this is a fresh copy, create the environment in Step 2. |
| Missing Python module | Run `.\.venv\Scripts\python.exe -m pip install -e .` from the main folder. |
| PowerShell blocks `npm` | Use `npm.cmd` as shown in this guide. |
| Training runs out of memory | Start a new run with a smaller `--batch-size`, such as `8`. |
| Training cache changed or cache cannot be used | Retry into a new run with `--loader files` to read prepared samples directly. |
| No raw-media extraction settings | The run was trained from supplied landmark arrays. Use compatible `.npy` input, or prepare raw media and train a new model for camera use. |
| Hand model differs from training | Use the original detector file with `--hand-model`. |
| Category unavailable in the app | Export a compatible trained run, reload the app, and download that category. |
| Camera does not open | Allow camera access and use `http://localhost:8080` on this computer, or an HTTPS-hosted app on another device. |

For the complete option lists:

```powershell
.\.venv\Scripts\python.exe categories.py prepare --help
.\.venv\Scripts\python.exe categories.py train --help
.\.venv\Scripts\python.exe trainer.py predict --help
```

For deeper technical details, read [README.md](README.md). Preparation and
normalization code lives in `src/fsl_trainer/preparation`; training code lives in
`src/fsl_trainer/training`. You do not need to edit either folder for normal use.
