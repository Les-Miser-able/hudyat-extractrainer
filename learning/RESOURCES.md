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

- [Current preparation pipeline reference](reference/dataset-pipeline.html)
  Code-grounded map of named dataset routing, extraction, grouping, sampling,
  metadata, validation, and the handoff to training. Use with Lesson 0003.
- [Shared sequence sampler](../fsl_trainer/data/sequences.py)
  Exact full-timeline linear interpolation, nearest-frame presence selection,
  and absent-slot zeroing used for fixed 32-by-128 model inputs.
- [Video timing augmentation](../fsl_trainer/data/augmentation.py)
  Exact training-only timing variations, deterministic candidate generation,
  duplicate rejection, and shortfall reporting. Use to distinguish requested
  extras from generated samples.
- [Named dataset paths](../fsl_trainer/data/dataset_paths.py)
  The current version discovery and legacy-dataset selection rules. Use when
  connecting preparation outputs to the selected training input.
- [scikit-learn: Cross-validation iterators for grouped data](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
  Primary explanation of why related observations must remain in the same split.
  Use for the rationale; this project's splitter is its own implementation.

## Wisdom (Communities)

No community has been selected. The current goal is a small local learning
exercise; revisit practitioner feedback when evaluating broader performance.

## Gaps

- A few practice recordings do not establish performance on unfamiliar signers,
  cameras, lighting conditions, or real FSL use. That needs a separate evaluation.
