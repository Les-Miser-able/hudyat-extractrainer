"""Training, checkpoint resume, and evaluation."""
import csv
import json
import time
from pathlib import Path
import numpy as np
from fsl_trainer.common import FRAMES, FEATURES, Stage, save_json
from fsl_trainer.preparation.sequences import load_sequence
from fsl_trainer.preparation.dataset import inspect_dataset
from fsl_trainer.models.network import tensorflow, build_model
from .evaluation import evaluation

def train(args):
    root, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Dataset and run output must be separate, non-nested directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Run output must be empty; choose a new directory")
    cache = None
    if getattr(args, "loader", "packed") == "packed":
        from .cache import open_cache
        cache, manifest, classes, stats = open_cache(root)
    else:
        manifest, classes, stats = inspect_dataset(root)
    tf = tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    label_ids = {label: i for i, label in enumerate(classes)}
    splits = {split: [r for r in manifest if r["split"] == split]
              for split in ("train", "validation", "test")}

    shuffle_rng = np.random.default_rng(args.seed)
    def dataset(split):
        rows = splits[split]
        if cache is not None:
            from .cache import batches
            ds = tf.data.Dataset.from_generator(
                lambda: batches(cache, split, args.batch_size, shuffle_rng if split == "train" else None),
                output_signature=(tf.TensorSpec((None, FRAMES, FEATURES), tf.float32),
                                  tf.TensorSpec((None,), tf.int32)))
            ds = ds.apply(tf.data.experimental.assert_cardinality(
                (len(rows) + args.batch_size - 1) // args.batch_size))
            return ds.prefetch(1)
        def generate():
            for row in rows:
                yield load_sequence(root / row["path"], fixed=True), np.int32(label_ids[row["label"]])
        ds = tf.data.Dataset.from_generator(generate, output_signature=(
            tf.TensorSpec((FRAMES, FEATURES), tf.float32), tf.TensorSpec((), tf.int32)))
        ds = ds.apply(tf.data.experimental.assert_cardinality(len(rows)))
        if split == "train":
            ds = ds.shuffle(min(len(rows), 10000), seed=args.seed, reshuffle_each_iteration=True)
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "classes.json", classes)
    save_json(output / "split_manifest.json", manifest)
    config = {k: v for k, v in vars(args).items() if k != "func"}
    config["tensorflow_version"] = tf.__version__
    config["split_counts"] = stats
    save_json(output / "training_config.json", config)
    (output / "preprocessing.json").write_text(
        (root / "preprocessing.json").read_text(encoding="utf-8"), encoding="utf-8")
    resume = getattr(args, "resume_from", None)
    if resume:
        saved_classes = json.loads((Path(resume).parent / "classes.json").read_text(encoding="utf-8"))
        if saved_classes != classes:
            raise ValueError("Checkpoint class order differs from the dataset")
        saved_preprocessing = json.loads((Path(resume).parent / "preprocessing.json").read_text(encoding="utf-8"))
        if saved_preprocessing != json.loads((root / "preprocessing.json").read_text(encoding="utf-8")):
            raise ValueError("Checkpoint preprocessing differs from the dataset")
        model = tf.keras.models.load_model(resume)
        expected = build_model(len(classes), args.learning_rate)
        def architecture(model):
            config = model.get_config()
            for layer in config["layers"]:
                layer["config"].pop("name", None)
                if layer["class_name"] == "Bidirectional":
                    for key in ("layer", "backward_layer"):
                        layer["config"][key]["config"].pop("name", None)
            config.pop("name", None)
            return json.loads(json.dumps(config))
        if architecture(model) != architecture(expected):
            raise ValueError("Checkpoint architecture differs from the required model")
        del expected
        print("Continuing saved weights and optimizer state; early-stopping counters restart.", flush=True)
    else:
        model = build_model(len(classes), args.learning_rate)
    config["actual_optimizer_learning_rate"] = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    config["data_loader"] = "packed" if cache is not None else "files"
    save_json(output / "training_config.json", config)
    class EpochTiming(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.times = []
        def on_epoch_begin(self, epoch, logs=None):
            self.started = time.perf_counter()
        def on_epoch_end(self, epoch, logs=None):
            seconds = time.perf_counter() - self.started
            self.times.append(seconds)
            save_json(output / "epoch_times.json", self.times)
            average = float(np.mean(self.times[-3:]))
            remaining = max(0, args.epochs - epoch - 1) * average / 60
            print(f"Epoch {epoch+1}: {seconds/60:.2f} min including validation; "
                  f"about {remaining:.0f} min remaining if all {args.epochs} epochs run.", flush=True)
    with (output / "model_summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))
    model.summary()
    counts = np.array([stats["train"][label] for label in classes])
    weights = {i: float(counts.sum() / (len(classes)*count)) for i, count in enumerate(counts)}
    callbacks = [
        EpochTiming(),
        tf.keras.callbacks.ModelCheckpoint(str(output / "best.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, args.patience//2)),
        tf.keras.callbacks.CSVLogger(str(output / "history.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(dataset("train"), validation_data=dataset("validation"),
                        epochs=args.epochs, callbacks=callbacks, shuffle=False,
                        class_weight=weights if args.class_weights else None)
    if not all(np.isfinite(v).all() for v in history.history.values()):
        raise RuntimeError("Training produced nonfinite metrics; run is incomplete")
    save_json(output / "history.json", {k: [float(v) for v in values]
                                      for k, values in history.history.items()})
    best = tf.keras.models.load_model(output / "best.keras")
    test = dataset("test")
    loss, accuracy = best.evaluate(test, verbose=0)
    probabilities = best.predict(test, verbose=0)
    truth = np.array([label_ids[r["label"]] for r in splits["test"]])
    report = evaluation(truth, probabilities, classes)
    report["loss"] = float(loss)
    save_json(output / "test_metrics.json", report)
    from .confusion_matrix import save_confusion_matrix
    with Stage("Saving test confusion matrices"):
        save_confusion_matrix(report, output)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "true_class", "predicted_class", "confidence"])
        for row, probs in zip(splits["test"], probabilities):
            writer.writerow([row["path"], row["label"], classes[int(probs.argmax())], float(probs.max())])
    print(json.dumps(report, indent=2))

