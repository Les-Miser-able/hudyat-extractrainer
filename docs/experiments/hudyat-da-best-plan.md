# HUDYAT experiment protocol for Da Best

Version 2 — defined on October 20, 2026. Submission deadline: October 30, 2026.

This is the recommended protocol for the shortened conference study. It defines
the technical choices rather than leaving them for the authors to design. The
numeric acceptance targets below are **researcher-defined engineering targets**,
not claimed Da Best rules or universal sign-recognition standards. Freeze them
before the new final experiments and report failures without lowering the targets.

This document defines work; no new datasets, training runs, HUDYAT deployments,
participant sessions, or Google Doc edits have been performed by writing it.

## 1. Fixed scope

Train four separate Conv1D–BiLSTM classifiers:

| Category | Outputs | Vocabulary |
|---|---:|---|
| Alphabet | 26 | A–Z, including movement signs J and Z. |
| Numbers | 10 | One through ten; no zero. |
| Colors | 6 | Blue, green, orange, red, violet, yellow. |
| Greetings | 10 | Good afternoon, good evening, good morning, hello, how are you, I'm fine, nice to meet you, see you tomorrow, thank you, you're welcome. |

Keep the confirmed **52-sign vocabulary**, with no learned None class. No-hand
detection and rejection are system behavior, not extra trained output labels.
HUDYAT is the application; Da Best is the conference.

Use this correction in the unfinished manuscript:

> Each category was recognized by a separate Conv1D–BiLSTM model whose output
> classes corresponded to that category's vocabulary signs. No-hand detection
> was handled separately from gesture classification; the models did not include
> a trained None class.

The [shared manuscript](https://docs.google.com/document/d/1u--b9hhhOQJOMbkiOqXNcgxfy2-oV_HKHGicXZwfkNU/edit?tab=t.0)
names accuracy, precision, recall, F1-score, and response time. Its accessible
export stops near the opening Methods paragraphs, and the user confirmed that
the manuscript is unfinished. The definitions below supply the missing protocol.

## 2. Acceptance benchmarks

Apply classification targets **to each category separately**, both for the
selected model's offline test and for its scored HUDYAT live trials.

| Measurement | Target | Calculation / evidence |
|---|---|---|
| Accuracy | At least 85% | Correct predictions divided by all evaluated sources or started scored attempts. |
| Macro precision | At least 85% | Mean of per-sign precision over every vocabulary sign in the category. |
| Macro recall | At least 85% | Mean of per-sign recall over every vocabulary sign in the category. |
| Macro F1 | At least 85% | Mean of per-sign F1, not F1 calculated from the macro precision/recall pair. |
| Lowest sign recall | At least 70% for every sign | Minimum per-sign recall; list the weakest signs and their support. |
| Post-capture feedback delay | Median below 1,000 ms and p95 below 2,000 ms | Timed in HUDYAT from the final required input frame's capture timestamp through the first painted confirmed result; see Section 3. |
| Live completion | At least 95% | Fraction of scored vocabulary attempts producing a confirmed result, right or wrong, within 15 seconds. Classification targets still penalize wrong results. |
| Functional behavior | All 40 critical checks pass | Ten checks in each category, defined in Section 8. |

These targets retain the repository's existing 85% classification and 70%
minimum-recall floors. The post-capture timing target is a **new, explicitly
different scope** from the existing first-hand-frame benchmark. It needs new
instrumentation; no old measurement is relabeled as meeting it.

For live scoring, wrong predictions, timeouts, incomplete captures, stopped
attempts, and errors remain in the denominator. Unrecognized attempts are false
negatives for the expected sign. Keep an unrecognized column in the live confusion
matrix; it is not a trained None class. Zero precision/recall denominators give
zero. Compare unrounded values to the thresholds.

Use per-sign TP/(TP+FP), TP/(TP+FN), and their harmonic-mean F1; use an unweighted
mean of sign-level values for macro metrics. These definitions follow the
[standard metric definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).

Report one table row per category. An optional overall summary is the unweighted
mean of the four category values, clearly labeled as such. A strong category
cannot compensate for another category failing its benchmark. Three signers and
15 live attempts/sign are a small convenience study, not proof of population-wide
performance; report individual signer results and counts.

## 3. Response-time definition: two clocks, both reported

For every scored attempt record these browser-monotonic timestamps:

- **t0:** Capture timestamp of the first submitted frame containing a usable hand,
  retained from before that frame's extraction.
- **t1:** Capture timestamp of the last frame needed for the classified input,
  retained from before processing that frame. A single-frame held pose uses its
  first usable frame. A movement uses the last frame in its captured window.
- **t2:** The first confirmed result is rendered and allowed to paint.

Calculate:

| Clock | Formula | Purpose |
|---|---|---|
| Post-capture feedback delay | t2 − t1 | Primary responsiveness benchmark: median <1 s and p95 <2 s. Counts remaining extraction/preprocessing, inference, confirmation, and rendering. |
| Full camera-to-result time | t2 − t0 | Mandatory end-to-end report, including the capture window. Report median, p95, maximum, completed count, and failures for each category. |

Do **not** move t1 to after inference, a confidence threshold, or a stability
confirmation. For a static pose, all subsequent stability waiting belongs in
feedback delay. t1 is an input-frame event, not an observer's estimate of when a
signer finished. Call this metric *post-capture feedback delay* consistently.

Compute median and nearest-rank p95 on all completed trials, including wrong
confirmations; also report correctly recognized trials separately. Never report
timing without completion/failure counts. A 15-second timeout has no confirmed
response timestamp and remains a failed recognition attempt. If no hand appears,
the timeout starts when the trial begins.

Exclude initial model download/loading from steady-state clocks, but record load
time separately. Make three unscored model warm-up calls of the actual input shape
before scored testing; [TensorFlow.js documents why warm-up matters](https://www.tensorflow.org/js/guide/platform_environment).

The current code's `hudyat_camera_to_result` scope is the full t2−t0 clock and
currently checks a median below 1,000 ms. Preserve that legacy result and its
status unchanged. A two-second capture cannot satisfy that original sub-second
end-to-end gate. The new post-capture criterion does not make that old gate pass.
Use separately versioned reports for the new study protocol.

## 4. Data preparation: one fixed protocol

### Source population and split

- Alphabet: the existing public static alphabet image collection and researcher
  videos for J/Z. Preserve source provenance and identify which letters use images.
- Numbers, Colors, Greetings: researcher recordings in the existing S1/S2/S3
  collections. Exclude PUB for this study to keep the source population controlled.
- Reject unreadable/corrupt and all-no-hand recordings, but retain an exclusion
  report. Deduplicate identical recordings/extracted arrays and group recognizable
  variants of the same original together. Review ambiguous sign labels before splitting.
- Split **original source groups** into 70% train, 15% validation, and 15% test,
  approximately stratified per sign, using preparation/split seed **20261020**.
  An original photograph/recording and all its derivatives stay in one split.
- This is **recording-level evaluation**, not signer-independent evaluation.
  Signers may occur in more than one split. State that limitation in the paper;
  do not invent signer identity from folder names or claim unseen-signer results.
- Keep the frozen test partition unavailable for model selection. Historical
  experiments used this source collection; describe the new holdout as held out
  from the new fits, not as a completely new, never-studied corpus.

Validation is used for model selection; final test performance is evaluated after
selection, consistent with [held-out evaluation guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

### Features and augmentation

Use these values in **all four new datasets**:

| Setting | Value |
|---|---|
| Detector | Existing MediaPipe two-hand detector; record its SHA-256. |
| Detection/presence confidence | 0.5. |
| Features | `hand-relative-sequence-v1`, explicitly selected for every category. |
| Input | 32 frames × 128 features, float32; two hand slots plus presence flags. |
| Hand swapping | Off for extraction; retain detector handedness consistently. |
| Static training variants | 8 total/original: one clean plus seven synthetic variants. |
| Static transforms | Existing ±15° rotation, 50% mirror probability with slot/flag swaps, and jitter 0.03 palm lengths. |
| Video training extras | 3 timing variants/original, time-warp strength 0.05, in addition to the clean sequence. |
| Validation and test | One clean, unaugmented sequence per accepted original. |

Video augmentation is temporal only; do not add video mirroring or new augmentation
methods for this deadline. The camera mirror toggle only affects the preview.

This standardizes preprocessing across the four categories. It changes the
preferred Numbers model's current image-coordinate convention, so all models are
fresh fits; do not resume incompatible checkpoints or mix new features with old
weights. Keep old models as historical baselines, not new-protocol results.

Save named datasets as `hudyat_dabest_v2`, with manifests, split assignments,
original-source counts, generated counts, preprocessing/augmentation metadata,
and extraction/rejection reports. Validate finite values, shapes, labels, group
isolation, hand-presence flags, and absent-hand zeros before training.

## 5. Training and selection: 12 controlled fits

Use the existing Conv1D–BiLSTM architecture without a new architecture search.

| Setting | Fixed value |
|---|---|
| Training seeds | 17, 42, 73 on the same prepared split: 3 × 4 categories = 12 fits. |
| Optimizer / initial rate | Adam / 0.001. |
| Batch size / maximum epochs | 32 / 100. |
| Early stopping | Validation loss, patience 12, restore best weights. |
| Rate reduction | Existing factor 0.5, validation-loss patience 6. |
| Class weights | On; inverse-frequency weights calculated from training counts only. |
| Loader | Packed. |
| Run names | `dabest_v2_s17`, `dabest_v2_s42`, `dabest_v2_s73` under each category. |

First run a one-epoch smoke check per category in separately named disposable
experiment outputs. Keep smoke checks out of the paper's model-result table.

For each seed, keep its minimum-validation-loss checkpoint. Select the deployment
seed by the **lowest best validation loss** among the three; ties go to the lower
numeric seed. Freeze the four selected checkpoint hashes before final evaluation.
Do not select a seed by its test or live accuracy.

Evaluate each frozen seed once on the offline test partition after selection.
Report the selected model plus the mean and sample standard deviation across the
three seeds. This measures training variation, not population or signer uncertainty.
Generate per-source predictions, per-sign scores/support, count/row-normalized
confusion matrices, and learning curves. Report J/Z separately.

There is no mandatory augmentation ablation or large hyperparameter sweep in this
conference scope. Any development repair must be validated and logged before the
October 24 model freeze; the predeclared test is not a tuning source.

**Tooling prerequisite:** active `train()` currently evaluates test automatically.
Add a training mode that defers final test evaluation and an explicit saved-model
evaluation step before these fits. The current `benchmark` command is model-only
timing, not this evaluator. This prerequisite is planned work, not an existing flag.

## 6. HUDYAT live-test protocol

Use the current Windows laptop with its webcam, Chrome stable, and the HUDYAT
TensorFlow.js WebGL backend. Record the actual CPU/GPU, browser/OS versions,
backend, camera resolution/FPS, and checkpoint/export hashes on test day. Request
640×480 at 30 FPS and record what the camera actually supplies. Do not claim phone
performance from this one-device evaluation.

Test indoors with steady front lighting, a fixed background, camera around face
height, and the signer approximately 0.8 m away with both hands in frame. Keep
conditions fixed rather than adding an uncontrolled robustness study. Preview may
be mirrored for comfort; inference frames remain unmirrored.

Use **three FSL-capable signers**, identified as P1/P2/P3. They can be the existing
recorded signers; record any training-participant overlap and make no unseen-signer
claim. Signers receive the reference demonstration and three unscored practice
attempts per category before scoring. Do not assume three people constitute a
statistically representative population.

| Category | Signs × 3 signers × 5 attempts/sign | Scored attempts |
|---|---|---:|
| Alphabet | 26 × 3 × 5 | 390 |
| Numbers | 10 × 3 × 5 | 150 |
| Colors | 6 × 3 × 5 | 90 |
| Greetings | 10 × 3 × 5 | 150 |
| Total | 52 × 3 × 5 | **780** |

Use five randomized rounds per category, with every sign once per round per signer;
save the order using seed 20261020 plus participant index. Take breaks between
category blocks. Remove hands/reset between trials so a held result is not scored
as a new prediction. Once a scored trial begins, keep it: a retry is a new record,
not a replacement for an unfavorable outcome. Fix model versions, confidence
thresholds, capture settings, and confirmation rules before scored testing.

J/Z require complete movement capture in HUDYAT. The existing Alphabet held-pose
flow is not sufficient evidence for those signs. Before October 25, implement and
validate full-gesture capture for J/Z while preserving the other letters' intended
capture behavior. Do not claim all 26 work if only the static letters were tested.

Record an additional **20 diagnostic negative attempts per category**: ten with no
hands and ten reviewed non-sign hand movements. Keep these 80 attempts outside
the 780 vocabulary-trial denominator. Report false confirmations and rejection
counts; they are not a learned None class and do not establish general open-set
recognition. The existing class-based trial schema needs a separate negative-test
record format rather than an invented model label.

The live report must show all classification metrics and per-sign support,
participant-level results, both timing distributions, and all outcome counts.
Timing cannot pass on a small successful subset when completion is below 95%.
If a signer/device is unavailable, record a protocol deviation and the actual
sample size; do not silently present a smaller experiment as this completed design.

## 7. Instrumentation that must exist before final tests

1. Validation-only training and explicit final evaluation, as specified above.
2. Per-attempt t0/t1/t2 logging with input timestamps and painted-result events;
   new scope name `hudyat_post_capture_feedback_v2` for t2−t1. Preserve legacy
   `hudyat_camera_to_result` sessions and their original thresholds/status.
3. Reliable J/Z movement capture in the actual HUDYAT practice flow.
4. Exported-model parity, exact class-order mapping, and detector/preprocessing
   identity for all four selected models.
5. Raw trial export including participant, expected/predicted sign, outcome,
   timing events, settings, and model/build identity; separate diagnostic negatives.

Native FSL Studio manual prediction records and local model-only latency are
development diagnostics. They do not substitute for this deployed HUDYAT test.
Do not derive the new feedback delay by subtracting an assumed two seconds from
old timings. Existing Alphabet stability waiting of about 1.2 seconds may itself
fail the new static feedback target; any change must be developed and validated
before the scored sessions.

## 8. Functional testing: 40 critical checks

For each of four categories, verify these ten cases with expected/actual results,
build/device identity, evidence, and pass/fail:

1. Category and lessons open with the correct vocabulary.
2. Reference demonstration and meaning match the selected sign.
3. Practice loads the intended category model and correct label mapping.
4. Camera starts, stops, and releases the device correctly.
5. A correct recognized result credits the expected practice sign.
6. A wrong recognized sign does not credit that expected sign.
7. No-hand input does not automatically credit a sign.
8. A correct quiz answer receives the expected score/feedback.
9. A wrong quiz answer receives the expected score/feedback.
10. Saved learning/quiz progress is preserved on reload as designed.

Functional decision logic is checked separately from the live model's accuracy.
All 40 cases must pass before declaring the four-category learning/practice/quiz
flow functionally verified. This does not measure improved learning outcomes.

## 9. Calendar and stopping rules

| Dates | Work and deliverable |
|---|---|
| Oct 20–21 | Freeze this protocol, source inventory/group mapping, input settings, model-selection rules, and test forms. Prepare validation-only and timestamp logging changes. |
| Oct 22 | Prepare four datasets, validate isolation/counts, run four smoke checks, and verify the HUDYAT J/Z capture and both clocks. |
| Oct 23–24 | Complete 12 fits, log validation evidence, select four models by the fixed rule, and freeze checkpoints/runtime settings. |
| Oct 25 | Evaluate final offline data, export frozen models, verify parity, and freeze the evaluation HUDYAT build. |
| Oct 26–27 | Collect 780 scored live attempts plus 80 diagnostic negatives under the fixed protocol. |
| Oct 28 | Complete 40 functional checks and verify evidence completeness. Preserve failed attempts; document any required build changes and affected-test reruns. |
| Oct 29 | Produce tables/figures and reconcile Methods, abstract, results, and limitations with measured outcomes. |
| Oct 30 | Final consistency review, archive, and submission buffer; no new model search. |

If a target is missed, report the miss and its likely cause; do not change a
threshold, clock, sign list, trial exclusion rule, or selected seed after examining
final results. New model/runtime tuning after final tests turns those tests into
development feedback; a fresh-final claim then needs new independent evidence.
If instrumentation is incomplete, report the measurement as missing rather than
replace it with manual response notes or model-only timing.

## 10. Final evidence and manuscript wording

Archive source inventory/provenance, exclusion logs, hashes and frozen splits,
prepared-data settings, all run configurations/curves, selected checkpoints,
class order, exports, offline predictions/metrics/matrices, raw live trials,
per-participant summaries, timing scopes, and the functional-test matrix.
Keep identifying source/participant material in the private study archive.

The abstract's old 98.25% figures are historical results, not required outcomes.
Replace them with this protocol's measured results if these become the reported
experiments. Retain the limitations: mixed public images/researcher videos,
recording-level split, three signers, one device/condition, repeated sources from
prior development, and limited live support per sign.

Suggested benchmark wording for Methods:

> Researcher-defined acceptance targets were set before final evaluation:
> accuracy, macro precision, macro recall, and macro F1-score of at least 85%
> for each category, with recall of at least 70% for every sign. HUDYAT live
> evaluation used three signers performing each of the 52 signs five times.
> Post-capture feedback delay was measured from the capture of the final required
> input frame until the confirmed result was visibly rendered, targeting a median
> below 1,000 ms and a 95th percentile below 2,000 ms. Full camera-to-result time,
> trial completion, and failed attempts were reported separately.

This is proposed wording for the new experiment, not a claim those benchmarks
have already been met. Neither the shared Google Doc nor the existing benchmark
code/status has been changed by this document.

## 11. Historical evidence and code references

Current preferred offline results are Alphabet 98.70% accuracy / 98.74% macro F1,
Numbers 97.14% / 97.13%, Colors 100% / 100%, and Greetings 97.14% / 97.13%.
The preferred Numbers model uses image coordinates while the other three use
hand-relative features; new-protocol values require fresh training.

Existing HUDYAT full camera-to-result medians are 2,117.4 ms for Colors (8 trials,
7 correct) and 2,127.9 ms for Greetings (11 trials, 10 correct). These are small
development samples, not the 780-trial study. Preferred Alphabet/Numbers lack
equivalent saved HUDYAT timings. Old full-clock results still fail the existing
sub-second full-clock benchmark and cannot establish the new feedback target.

- [Preparation, model training, and evaluation](../../fsl_trainer/training/trainer.py)
- [Extraction and source provenance](../../fsl_trainer/data/extraction.py)
- [Hand-relative features and static mirroring](../../fsl_trainer/data/preprocessing.py)
- [Video timing augmentation](../../fsl_trainer/data/augmentation.py)
- [Existing benchmark definitions](../../fsl_trainer/reporting/benchmark.py)
- [Model-only latency](../../fsl_trainer/training/benchmark.py)
- [Trainer live-trial metrics](../../fsl_trainer/dashboard/prediction_trials.py)
- [HUDYAT legacy camera timing](../../fsl_trainer/dashboard/camera_timing.py)
- [HUDYAT export/integration behavior](../../integrations/hudyat/README.md)
