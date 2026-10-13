# FSL Pocket — client-side PWA sample

**Fresh clone:** exported models and the shared hand detector are included.
Run `npm ci` and `npm run setup` in this folder to recreate browser libraries.
Export additional trained models as described below.
See [repository setup](../docs/REPOSITORY.md).

Choose a category, download its model, turn on the camera, and recognize a held pose or a two-second gesture. Camera frames and predictions stay in the browser. The server only serves static files.

Alphabet (26 letters) and colors (8 classes) are included from the existing alphabet/optimized and colors/first checkpoints. Family, greetings, numbers and shapes appear as not exported until you add their trained models. Categories come from the trainer's data folders.

## Open locally

From PowerShell:

```powershell
cd pwa-sample
npm start
```

Open http://localhost:8080. Node.js 22 or newer is recommended. Complete the fresh-clone setup first; later starts can reuse those local assets. Alphabet and colors are already exported.

1. Select Alphabet or Colors.
2. Press Download model. Progress messages show app setup, model download and hand-detector download.
3. Press Turn on camera and allow camera access.
4. Use the held-pose button for a static sign, or record the complete moving sign within the two-second recording window.
5. Download another category whenever you want to switch. Already downloaded categories load from browser storage.

Only the selected category's weights are downloaded. Each included model is about 1.9 MiB including metadata. The 7.5 MiB hand detector is shared across categories; TensorFlow.js and MediaPipe runtime files are cached with the app on first use. Removing a category removes its model, while shared tools remain available.

After a successful download, the app and saved categories can work offline. Browser storage can be cleared or evicted, so the UI checks whether the model is still present. Downloads belong to the browser/device where they were made.

## Add another trained category

Run from the main fsl_trainer folder, using the Python environment with the trainer dependencies:

```powershell
python pwa-sample/tools/export_model.py --category family --run runs/family/first
```

Replace family and first with your category and run name. The run must contain best.keras, classes.json and preprocessing.json. Exporting reads the existing checkpoint; it does not train or extract your dataset. The exporter verifies the architecture and the MediaPipe model checksum, writes browser weights, and updates the category catalog. Reload the app online to see the new entry.

This sample uses its own documented weight format, fsl-conv1d-bilstm-v1. The browser reconstructs the exact Conv1D + bidirectional LSTM network and loads its weights; it does not load the .keras file directly.

## Install or use on a phone

Serve the contents of dist on an HTTPS static host. No prediction API or backend is needed. Open the HTTPS address on your phone, then use the browser's Install app / Add to Home Screen option when available. Installation controls differ by browser.

Localhost works for development on the same computer. A phone visiting your computer through an ordinary LAN HTTP address will not have the secure context needed for camera and offline features.

## Preprocessing

- Two fixed hand slots: 63 left coordinates, 63 right coordinates, two presence flags.
- Input shape: 32 frames × 128 features.
- Image-normalized XYZ and handedness swap settings follow the exported training metadata.
- Absent slots are zero. Presence flags stay binary.
- Held poses repeat one extracted frame for 32 timesteps. Jitter is a training augmentation and is not added during prediction.
- Video coordinates use uniform linear interpolation to 32 frames, with nearest-frame presence flags.
- The camera preview is mirrored visually; pixels passed to MediaPipe are unmirrored.
- No hands produces a no-hand message. Scores below 60% display uncertainty. Scores are model outputs, not a measured guarantee of accuracy.

## Development and verification

```powershell
npm install
npm run setup
npm test
npm run parity
```

setup copies pinned TensorFlow.js 4.22.0 and MediaPipe Tasks Vision 0.10.32 libraries into dist and rebuilds the offline asset list. Keep package-lock.json for reproducible dependencies. When changing app files, increment the shell-cache version in dist/sw.js so existing installations refresh their cached app. Model versions change automatically during export.

Tests cover feature packing, interpolation, missing hands, ranking, asset availability, model integrity, and service-worker caching. The parity command compares the actual exported models against Python predictions on three synthetic landmark sequences per category. Both included models passed with maximum probability differences below 0.0000001 using TensorFlow.js on the Node CPU backend.

Live camera recognition, browser GPU behavior, installation and offline reload have not been manually tested on a physical phone/browser. Cross-platform MediaPipe extraction can differ even with the same landmark model asset. The included checkpoints' real-world accuracy is unchanged by this sample.

Optional WebMCP category listing/selection tools are feature-detected; ordinary use does not depend on them.

References: [TensorFlow.js](https://www.tensorflow.org/js/guide/models_and_layers) and [MediaPipe Hand Landmarker for web](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js).

## Live alphabet diagnostics and webcam evaluation
Select Alphabet, enable the camera, and press Start live alphabet. This runs serial held-pose predictions at up to five updates per second (slower devices may update less often), reusing the image detector to match the still-image training pipeline. Pause live or Stop camera ends the loop. The overlay is mirrored with the preview, while model input remains unmirrored. No extra coordinate normalization or confidence-threshold reduction is applied.

All sources in the included alphabet checkpoint are still images, including J and Z. Moving-letter accuracy is not established. In a deterministic sample of 20 training sources per class (520 total), hand-width percentiles were 29%, 45%, and 74% at the 10th, 50th, and 90th percentiles. The displayed webcam hand width helps compare framing; these are diagnostics, not mandatory bounds or proof of the cause of mistakes.

Expand Measure webcam accuracy, choose the actual letter, and Save labeled trial after a prediction. Reposition between trials and include incorrect/uncertain predictions. The expected label never affects inference. Download test results before switching category or closing the page. The JSON contains labels, scores, and landmark coordinates, without camera images. Accuracy counts raw top-1 predictions, including those shown as uncertain. Repeated trials from one signer/session do not establish general accuracy.

From the trainer folder, turn the downloaded JSON into confusion matrices:

```powershell
python confusion_matrix.py --report "C:\path\to\webcam-test-alphabet.json" --output runs/alphabet/webcam-evaluation
```

Start with five separately posed trials per static letter in consistent lighting. Repeat in another session to check whether errors persist. Use this baseline to decide which webcam examples to collect for a new training dataset; keep evaluation recordings separate from training. Neither the model weights nor their measured real-world accuracy have been changed by these diagnostics.
