# Greetings versus colors: preprocessing and checkpoint audit

Compared `runs/greetings/greetings-run2-20261013-122531` with
`runs/colors/Colors-run1-20261013-113518`. This investigation read existing data
and ran diagnostics in memory. It did not alter datasets, training code, learned
weights, saved checkpoints, or recorded metrics.

## Finding

Both prepared datasets pass the existing shape, finite-value, presence-flag,
coverage, class, and source-group isolation checks. Each dataset's manifest and
preprocessing metadata exactly match the copies in its run directory. A major
cause of the poor greetings predictions is the saved model's BatchNormalization
statistics. The evidence does not support a general landmark-extraction failure.

| Saved setting or result | Greetings run 2 | Colors run 1 |
| --- | ---: | ---: |
| Classes | 10 | 4 |
| Selected source videos | 451 | 180 |
| Accepted originals | 450 | 180 |
| Training originals | 310 | 124 |
| Timing extras per training video | 0 | 8 |
| Total training samples | 310 | 1,116 |
| Validation / test samples | 70 / 70 | 28 / 28 |
| Early-stopping patience | 1 | 12 |
| Completed epochs | 4 | 35 |
| Best validation-loss epoch | 3 | 23 |
| Recorded test accuracy | 25.7% | 100% |
| Median source-frame hand detection coverage | 55.6% | 47.5% |

Both runs used the same hand-model checksum, MediaPipe version, detection
confidence 0.5, handedness configuration, 32x128 layout, interpolation method,
seed 42, S1/S2/S3 selection, and 15% validation / 15% test targets. Although the
static-variants field is 8 in both runs, all accepted sources are videos; that
field does not enable video augmentation. Greetings has no video-augmentation
settings or generated timing variants.

## Extraction details

Greetings rejected exactly one selected video:
`nice_to_meet_you/S2/nice_to_meet_you_S2_01.mp4`, with the reason
`No hand detected in any video frame`. All remaining 450 videos were prepared.
`nice_to_meet_you` therefore has 44 accepted sources, while `youre_welcome` has
46 (its S2 folder contains 16 files). Other greetings have 45 each. This small
count difference does not explain the broad prediction collapse.

Both datasets contain approximately three-second clips and substantial periods
without detected hands. Greetings has 17.5 detected timesteps per original
32-frame prepared sample at the median, compared with 15 for colors. One
greetings training source, `im_fine/S1/im_fine_S1_03.mp4`, has only 7 detected
prepared timesteps. Detection coverage alone is not a visual judgment of
landmark correctness, but the overall greetings coverage is not worse than
colors. No source videos were manually relabeled or visually audited here.

## Training-versus-inference diagnostic

At greetings' best checkpoint, epoch 3 reported 81.9% training accuracy. Loading
that checkpoint and using ordinary inference on the same training samples gives
only 26.5% accuracy. Its validation accuracy is 28.6% (20 of 70).

The network uses two BatchNormalization layers with momentum 0.99. These layers
use current batch statistics while training and stored moving statistics during
inference; see the [Keras BatchNormalization documentation](https://keras.io/api/layers/normalization_layers/batch_normalization/).
With 310 samples and batch size 32, the epoch-3 greetings checkpoint has seen
approximately 30 training batches. Colors has 35 batches per epoch and its best
checkpoint is much later. This makes slow-moving normalization statistics a
plausible explanation for the larger early training/inference discrepancy.

Two diagnostic recalibration methods were evaluated using training samples only:

| In-memory checkpoint state | Training inference accuracy | Validation accuracy | Validation loss |
| --- | ---: | ---: | ---: |
| Original saved checkpoint | 26.5% | 28.6% | 1.8666 |
| Ten shuffled training-data passes updating moving averages, dropout disabled | 36.1% | 32.9% | 2.2662 |
| Exact training-data activation means/variances, recalculated in layer order | 91.0% | 85.7% | 0.4716 |

The exact method computed each BatchNormalization layer's input activations on
training data with inference behavior, then assigned that layer's moving mean
and variance before proceeding to the next layer. Validation samples were only
scored, never used to compute those statistics. Trainable weights were checked
for exact equality before and after; the saved checkpoint SHA-256 was also
unchanged. Dropout rates were zeroed only in the diagnostic model in memory;
ordinary inference disables dropout regardless. No optimizer updates occurred.

The 85.7% figure is a diagnostic validation result, not a replacement test score
or a repaired saved model. Test data was not evaluated after recalibration.
The original recorded 25.7% test result remains unchanged.

## Why increasing patience alone is insufficient

`greetings-run1-20261013-122531` used the same unaugmented dataset with patience
12, trained for 15 epochs, and also recorded 25.7% test accuracy. Its minimum
validation loss was still at epoch 3. The trainer saves `best.keras` by minimum
validation loss, so later higher validation accuracy did not replace that early
checkpoint. Lowering patience to 1 shortened run 2 but did not create the entire
problem.

## Next implementation and experiment

Address how BatchNormalization statistics are established and validated before
checkpoint selection and final inference. A fix should use training data only,
preserve model interfaces, and be checked on held-out validation data before
final test evaluation. The diagnostic intervention identifies a cause; it is not
yet an implemented production fix.

For a controlled comparison with colors, prepare a fresh greetings dataset with
8 timing extras and strength 0.05, retain patience 12, and use a fresh run name.
Those controls match colors but do not guarantee equal accuracy across different
classes. Fixing or controlling normalization behavior is more directly supported
by this investigation than changing extraction confidence or resampling blindly.

Both manifests isolate individual source recordings, but S1, S2, and S3 each
appear across train/validation/test. If those folders identify signers or
sessions, use explicit shared group IDs for an evaluation of unseen signers or
sessions. The colors 100% result describes 28 prepared test recordings and does
not establish webcam or unseen-signer accuracy.

Evidence comes from each run's `training_config.json`, `preprocessing.json`,
`history.json`, `split_manifest.json`, and `test_metrics.json`, plus each source
dataset's `extraction_report.json`, `augmentation_summary.json`, original-length
extracted landmarks, and prepared arrays.
