# Gesture Training Resources

## Knowledge

- [Project training guide](../README.md)
  The contract for this exact trainer: category/class folders, independent source
  groups, raw-media preparation, 32-by-128 inputs, training and prediction commands.
  Use this before applying generic ML advice to the repository.
- [Trainer implementation](../fsl_trainer/training/trainer.py)
  Source of truth for the Conv1D–BiLSTM architecture, group-disjoint splits,
  augmentation, checkpoint selection by validation loss, and saved test metrics.
- [MediaPipe Hand Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
  Google's primary documentation for image/video hand detection, handedness and
  landmark outputs. Use to understand what the extraction stage supplies.
- [Keras: Training and evaluation with built-in methods](https://keras.io/guides/training_with_built_in_methods/)
  Primary guide to fitting a model, validation, evaluation and prediction.
  Use to distinguish training feedback from evaluation on unseen examples.
- [TensorFlow: Overfit and underfit](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)
  Primary tutorial on interpreting training and validation behavior.
  Use after completing the first working training run.

- [Project media extractor](../fsl_trainer/data/extraction.py)
  Exact video decoding, timestamps, hand-slot packing, rejection rules, and
  extraction-report fields used by this workspace.
- [FFmpeg: fps filter](https://ffmpeg.org/ffmpeg-filters.html#fps)
  Official constant-frame-rate conversion behavior. Use when source videos
  have variable timing or need a consistent frame rate before extraction.

## Wisdom (Communities)

No community has been selected. The current goal is a small local learning
exercise; revisit practitioner feedback when evaluating broader performance.

## Gaps

- A few practice recordings do not establish performance on unfamiliar signers,
  cameras, lighting conditions, or real FSL use. That needs a separate evaluation.
