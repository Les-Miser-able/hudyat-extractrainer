"""Conv1D + BiLSTM gesture trainer. Run python trainer.py --help."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from ..progress import Stage, progress
from ..data.sequences import resample
from ..data.augmentation import augment_sequence, configuration, build_report
from ..data.augmentation_options import add_arguments
from ..data.source_selection import add_arguments as add_source_arguments, folder_selection, source_files
from ..data.preprocessing import (IMAGE, HAND_RELATIVE, COORDINATES, augment_pose,
                                  preparation_normalization, model_normalization,
                                  add_arguments as add_preprocessing_arguments)

FRAMES, FEATURES = 32, 128
LAYOUT = ["left_xyz_landmarks_0_to_20", "right_xyz_landmarks_0_to_20",
          "left_present", "right_present"]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def save_augmentation_summary(root, manifest):
    """Count accepted originals separately from generated output sequences."""
    report_path = Path(root) / "video_augmentation_report.json"
    video_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None

    def counts(rows):
        sources = {row["source"]: row["static"] for row in rows}
        static = sum(sources.values())
        result = {
            "original_sources": len(sources),
            "static_sources": static,
            "sequence_sources": len(sources) - static,
            "augmented_static_sequences": sum(bool(row["synthetic"]) and row["static"] for row in rows),
            "resampled_sequences": sum(not row["synthetic"] and not row["static"] for row in rows),
            "total_output_sequences": len(rows),
        }
        clean_static = sum(not row["synthetic"] and row["static"] for row in rows)
        if clean_static:
            result["unaugmented_static_sequences"] = clean_static
        if video_report is not None:
            details = [item for item in video_report["sources"] if item["source"] in sources]
            result["augmented_sequence_samples"] = sum(
                row["synthetic"] and not row["static"] for row in rows)
            for field in ("requested_extra_samples", "unfilled_extra_samples",
                          "duplicate_candidates", "invalid_candidates"):
                result[field] = sum(item[field] for item in details)
        return result

    summary = counts(manifest)
    summary["by_split"] = {
        split: counts([row for row in manifest if row["split"] == split])
        for split in ("train", "validation", "test")}
    summary["by_gesture"] = {
        label: counts([row for row in manifest if row["label"] == label])
        for label in sorted({row["label"] for row in manifest})}
    path = Path(root) / "augmentation_summary.json"
    save_json(path, summary)
    print(f"\nAugmentation summary: {Path(root).name}", flush=True)
    print(f"Original sources accepted: {summary['original_sources']:,} "
          f"({summary['static_sources']:,} static, {summary['sequence_sources']:,} video/sequence)")
    print(f"Augmented static sequences: {summary['augmented_static_sequences']:,}")
    if "unaugmented_static_sequences" in summary:
        print(f"Unaugmented static sequences: {summary['unaugmented_static_sequences']:,}")
    print(f"Resampled video/sequence outputs: {summary['resampled_sequences']:,}")
    if video_report is not None:
        print(f"Video/sequence extras generated: {summary['augmented_sequence_samples']:,} "
              f"of {summary['requested_extra_samples']:,} requested; "
              f"{summary['unfilled_extra_samples']:,} unfilled.", flush=True)
        print(f"Skipped candidates: {summary['duplicate_candidates']:,} duplicate, "
              f"{summary['invalid_candidates']:,} invalid.", flush=True)
        for item in video_report["sources"]:
            if item["unfilled_extra_samples"]:
                print(f"  {item['source']}: {item['generated_extra_samples']} of "
                      f"{item['requested_extra_samples']} extras generated "
                      f"({', '.join(item['shortfall_reasons'])}).", flush=True)
    print(f"Total output sequences: {summary['total_output_sequences']:,}")
    print("By split (output sequences):")
    for split, values in summary["by_split"].items():
        print(f"  {split}: {values['total_output_sequences']:,}")
    print("By gesture (accepted originals / augmented static / resampled / total):")
    for label, values in summary["by_gesture"].items():
        print(f"  {label}: {values['original_sources']:,} / "
              f"{values['augmented_static_sequences']:,} / "
              f"{values['resampled_sequences']:,} / {values['total_output_sequences']:,}")
    print(f"Summary saved: {path}", flush=True)


def load_sequence(path, fixed=False):
    """Input is already normalized. Never infer hand presence from coordinates."""
    x = np.load(path, allow_pickle=False)
    if x.ndim == 1 and x.shape == (FEATURES,):
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != FEATURES or len(x) < 1:
        raise ValueError(f"{path}: expected (T,128), T>=1; got {x.shape}")
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError(f"{path}: expected real numeric coordinates")
    x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: contains NaN or infinity")
    if not np.isin(x[:, 126:], [0, 1]).all():
        raise ValueError(f"{path}: presence columns 126:128 must be 0 or 1")
    if fixed and x.shape != (FRAMES, FEATURES):
        raise ValueError(f"{path}: prepared samples must be (32,128)")
    for hand in range(2):
        absent = x[:, 126 + hand] == 0
        if np.any(x[absent, hand * 63:(hand + 1) * 63] != 0):
            raise ValueError(f"{path}: absent hand {hand} must have zero coordinates")
    if not x[:, 126:].any():
        raise ValueError(f"{path}: no hand present in the entire sample")
    return x


def synthetic_sequence(frame, rng, jitter=0.0075):
    """Repeat one pose and add independent per-timestep Gaussian coordinate noise."""
    out = np.repeat(frame[:1], FRAMES, axis=0).astype(np.float32)
    for hand in range(2):
        if frame[0, 126 + hand] == 1:
            out[:, hand*63:(hand+1)*63] += rng.normal(
                0, jitter, (FRAMES, 63)).astype(np.float32)
    return out


def split_sources(records, val_fraction, test_fraction, seed):
    """Group-disjoint splits. Groups can represent source images, sessions or signers."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1
            and val_fraction + test_fraction < 1):
        raise ValueError("validation and test fractions must be positive and sum to <1")
    labels = sorted({r["label"] for r in records})
    groups = sorted({r["group"] for r in records})
    for label in labels:
        if len({r["group"] for r in records if r["label"] == label}) < 3:
            raise ValueError(f"{label}: need at least 3 independent source groups "
                             "for train/validation/test; synthetic variants do not count")
    rng = np.random.default_rng(seed)
    # Class-local groups admit a direct stratified assignment.
    memberships = {g: {r["label"] for r in records if r["group"] == g} for g in groups}
    assignment = {}
    if all(len(v) == 1 for v in memberships.values()):
        for label in labels:
            local = np.array([g for g in groups if label in memberships[g]], dtype=object)
            rng.shuffle(local)
            nv = max(1, round(len(local) * val_fraction))
            nt = max(1, round(len(local) * test_fraction))
            while nv + nt >= len(local):
                if nv >= nt and nv > 1:
                    nv -= 1
                elif nt > 1:
                    nt -= 1
                else:
                    raise ValueError("Cannot allocate three nonempty splits")
            for i, group in enumerate(local):
                assignment[group] = "validation" if i < nv else "test" if i < nv + nt else "train"
        return assignment
    # Shared signer/session IDs must remain together across every class.
    matrix = np.array([[sum(r["group"] == g and r["label"] == label for r in records)
                        for label in labels] for g in groups])
    fractions = np.array([1 - val_fraction - test_fraction, val_fraction, test_fraction])
    best, best_score = None, float("inf")
    for _ in range(3000):
        choices = rng.choice(3, size=len(groups), p=fractions)
        counts = np.array([matrix[choices == k].sum(axis=0) for k in range(3)])
        if (counts == 0).any():
            continue
        score = np.mean((counts / matrix.sum(axis=0) - fractions[:, None]) ** 2)
        if score < best_score:
            best, best_score = choices.copy(), score
    if best is None:
        raise ValueError("Could not find group-disjoint splits containing every class. "
                         "Add independent groups or revise group IDs/fractions.")
    return {g: ["train", "validation", "test"][int(k)] for g, k in zip(groups, best)}


def prepare(args):
    args.normalization = preparation_normalization(getattr(args, "normalization", IMAGE))
    if getattr(args, "jitter", None) is None:
        args.jitter = .03 if args.normalization == HAND_RELATIVE else .0075
    video_settings = configuration(getattr(args, "video_extra_samples", 0),
                                   getattr(args, "video_time_warp", .05), args.seed)
    source, dest = Path(args.source).resolve(), Path(args.output).resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory not found: {source}")
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Output must be empty; existing datasets are never overwritten")
    selected = folder_selection(getattr(args, "source_folders", None))
    videos_only = getattr(args, "videos_only", False)
    filtered = selected is not None or videos_only
    if not getattr(args, "_landmarks_only", False):
        from ..data.extraction import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, prepare_media
        media_files = [p for folder in source.iterdir() if folder.is_dir() and not folder.name.startswith(".")
                       for p in source_files(folder, selected, videos_only)]
        if any(p.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS for p in media_files):
            return prepare_media(args, prepare)
        if args.normalization == HAND_RELATIVE:
            raise ValueError("Hand-relative preparation requires raw images/videos, not supplied .npy files")
    mapping = None
    if args.groups:
        with open(args.groups, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        mapping = {}
        for row in rows:
            key, group = row["path"].replace("\\", "/"), row["group"].strip()
            if key in mapping or not group:
                raise ValueError("Group CSV has duplicate paths or empty group IDs")
            mapping[key] = group
    records, fingerprints, excluded_classes = [], {}, []
    for folder in progress(sorted(source.iterdir()), "Checking source classes"):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        paths = ([p for p in source_files(folder, selected, videos_only) if p.suffix.lower() == ".npy"]
                 if filtered else sorted(folder.glob("*.npy")))
        if not paths:
            if filtered:
                excluded_classes.append(folder.name)
                continue
            raise ValueError(f"Class folder has no .npy samples: {folder}")
        for path in progress(paths, f"Checking {folder.name} landmark files"):
            relative = path.relative_to(source).as_posix()
            x = load_sequence(path)
            fingerprint = hashlib.sha256(x.tobytes()).hexdigest()
            if fingerprint in fingerprints:
                raise ValueError(f"Duplicate source data: {relative} and {fingerprints[fingerprint]}. "
                                 "Remove duplicates before splitting.")
            fingerprints[fingerprint] = relative
            if mapping is not None and relative not in mapping:
                raise ValueError(f"Missing group mapping: {relative}")
            records.append({"source": relative, "label": folder.name,
                            "group": mapping[relative] if mapping is not None else relative,
                            "static": len(x) == 1, "sha256": fingerprint})
    if len({r["label"] for r in records}) < 2:
        raise ValueError("Need at least two gesture class folders")
    if mapping is not None and set(mapping) != {r["source"] for r in records}:
        raise ValueError("Group CSV contains paths not present in the source dataset")
    with Stage("Assigning train / validation / test groups"):
        assignment = split_sources(records, args.validation, args.test, args.seed)
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest = []
    video_sources = []
    kind = "base sequences" if video_settings else "sequences"
    print(f"Generating {sum(args.static_variants if r['static'] else 1 for r in records):,} {kind} from {len(records):,} source files.", flush=True)
    for record in progress(records, "Generating and saving sequences (source files)"):
        split = assignment[record["group"]]
        x = load_sequence(source / record["source"])
        # Assign the source first; all synthetic siblings inherit the same split.
        relative_features = args.normalization == HAND_RELATIVE
        count = args.static_variants if record["static"] and (not relative_features or split == "train") else 1
        for variant in range(count):
            synthetic = record["static"] and (not relative_features or (split == "train" and variant > 0))
            if synthetic:
                frame = augment_pose(x, rng) if relative_features else x
                sequence = synthetic_sequence(frame, rng, args.jitter)
            else:
                sequence = resample(x)
            token = hashlib.sha256(record["source"].encode()).hexdigest()[:16]
            relative = Path(split) / record["label"] / f"{token}_{variant:03d}.npy"
            (dest / relative).parent.mkdir(parents=True, exist_ok=True)
            np.save(dest / relative, sequence.astype(np.float32), allow_pickle=False)
            manifest.append({**record, "split": split, "path": relative.as_posix(),
                             "variant": variant, "variant_of": record["source"], "synthetic": synthetic})
        if video_settings and not record["static"] and split == "train":
            extras, detail = augment_sequence(x, record["source"], args.seed,
                                      video_settings["extra_samples"],
                                      video_settings["max_displacement"])
            video_sources.append({**detail, "label": record["label"], "group": record["group"],
                                  "split": split})
            for variant, (sequence, transformation) in enumerate(extras, 1):
                relative = Path(split) / record["label"] / f"{token}_{variant:03d}.npy"
                np.save(dest / relative, sequence, allow_pickle=False)
                manifest.append({**record, "split": split, "path": relative.as_posix(),
                                 "variant": variant, "variant_of": record["source"],
                                 "synthetic": True, "augmentation": transformation})
    print("Saving dataset manifest and preprocessing settings...", flush=True)
    save_json(dest / "manifest.json", manifest)
    save_json(dest / "preprocessing.json", {
        "frames": FRAMES, "features": FEATURES, "layout": LAYOUT,
        "coordinates": "already normalized by upstream extractor; no additional normalization",
        "interpolation": "linear coordinates, nearest binary flags, absent coordinates zeroed",
        "seed": args.seed, "validation_fraction": args.validation, "test_fraction": args.test,
        "static_variants": args.static_variants, "jitter_std": args.jitter,
        "static_method": "repeat + independent per-timestep Gaussian coordinate noise",
        "grouping": "provided CSV" if mapping is not None else "original source file"})
    if args.normalization == HAND_RELATIVE:
        metadata = json.loads((dest / "preprocessing.json").read_text(encoding="utf-8"))
        metadata.update(normalization=HAND_RELATIVE, coordinates=COORDINATES,
                        static_method="one clean original; training-only repeat + Gaussian noise in palm lengths",
                        pose_augmentation={"split": "train", "rotation_degrees": 15., "mirror_probability": .5},
                        evaluation="one unaugmented sequence per accepted source")
        save_json(dest / "preprocessing.json", metadata)
    if filtered:
        metadata = json.loads((dest / "preprocessing.json").read_text(encoding="utf-8"))
        metadata["source_selection"] = {"folders": selected, "videos_only": videos_only,
                                        "selected_files": len(records), "excluded_classes": excluded_classes}
        save_json(dest / "preprocessing.json", metadata)
    if video_settings:
        metadata = json.loads((dest / "preprocessing.json").read_text(encoding="utf-8"))
        metadata["video_augmentation"] = video_settings
        save_json(dest / "preprocessing.json", metadata)
        save_json(dest / "video_augmentation_report.json", build_report(
            video_settings, video_sources, sum(not record["static"] for record in records)))
    print(json.dumps(inspect_dataset(dest)[2], indent=2), flush=True)
    print("Sequence generation and dataset validation complete.", flush=True)
    if not getattr(args, "_landmarks_only", False):
        save_augmentation_summary(dest, manifest)


def inspect_dataset(root, on_sample=None):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError("Dataset manifest is empty")
    classes = sorted({r["label"] for r in manifest})
    if len(classes) < 2:
        raise ValueError("Need at least two gesture classes")
    group_splits, source_splits, variant_splits, paths, stats = {}, {}, {}, set(), {}
    for row in progress(manifest, "Validating sequence files and split isolation"):
        split = row["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {split}")
        for key, lookup in (("group", group_splits), ("source", source_splits), ("variant_of", variant_splits)):
            identity = row[key]
            if identity in lookup and lookup[identity] != split:
                raise ValueError(f"Data leakage: {key} {identity} spans splits")
            lookup[identity] = split
        relative = Path(row["path"])
        resolved = (root / relative).resolve()
        if root not in resolved.parents or relative.parts[:2] != (split, row["label"]):
            raise ValueError(f"Invalid sample path: {relative}")
        if resolved in paths:
            raise ValueError(f"Duplicate manifest path: {relative}")
        paths.add(resolved)
        sequence = load_sequence(resolved, fixed=True)
        if on_sample is not None:
            on_sample(row, sequence)
        stats.setdefault(split, {}).setdefault(row["label"], 0)
        stats[split][row["label"]] += 1
    with Stage("Checking dataset file coverage"):
        actual = {p.resolve() for s in ("train", "validation", "test")
                  for p in (root / s).glob("*/*.npy")}
    if actual != paths:
        raise ValueError("Dataset files and manifest disagree")
    for split in ("train", "validation", "test"):
        if set(stats.get(split, {})) != set(classes):
            raise ValueError(f"{split} must contain all classes")
    return manifest, classes, stats


def tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required: python -m pip install -r requirements.txt") from exc


def build_model(num_classes, learning_rate=0.001):
    tf = tensorflow()
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input(shape=(32, 128)),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(momentum=0.99, epsilon=0.0001),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(momentum=0.99, epsilon=0.0001),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="fsl_conv1d_bilstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def evaluation(y_true, probabilities, classes):
    predicted = probabilities.argmax(axis=1)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(cm, (y_true, predicted), 1)
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(len(classes)), where=cm.sum(axis=0)>0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(len(classes)), where=cm.sum(axis=1)>0)
    f1 = np.divide(2*precision*recall, precision+recall,
                   out=np.zeros(len(classes)), where=(precision+recall)>0)
    return {"accuracy": float(np.mean(predicted == y_true)), "macro_f1": float(f1.mean()),
            "confusion_matrix_rows_true_columns_predicted": cm.tolist(), "classes": classes,
            "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(cm[i].sum())}
                          for i, label in enumerate(classes)}}


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
    augmentation_report = root / "video_augmentation_report.json"
    if augmentation_report.is_file():
        (output / augmentation_report.name).write_bytes(augmentation_report.read_bytes())
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
    from ..reporting.confusion_matrix import save_confusion_matrix
    with Stage("Saving test confusion matrices"):
        save_confusion_matrix(report, output)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "true_class", "predicted_class", "confidence"])
        for row, probs in zip(splits["test"], probabilities):
            writer.writerow([row["path"], row["label"], classes[int(probs.argmax())], float(probs.max())])
    print(json.dumps(report, indent=2))


def predict(args):
    tf = tensorflow()
    root = Path(args.run)
    classes = json.loads((root / "classes.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(root / "best.keras")
    if Path(args.sample).suffix.lower() == ".npy":
        x = resample(load_sequence(args.sample))
    else:
        from ..data.extraction import Extractor, DEFAULT_MODEL
        metadata = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
        settings = metadata.get("extraction")
        if not settings:
            raise ValueError("This model has no raw-media extraction settings; supply normalized .npy input")
        normalization = model_normalization(metadata)
        feature_options = {"normalization": normalization} if normalization != IMAGE else {}
        with Extractor(model=getattr(args, "hand_model", None) or DEFAULT_MODEL,
                       confidence=settings["detection_confidence"],
                       swap_hands=settings["swap_hands"], **feature_options) as extractor:
            if hashlib.sha256(extractor.model.read_bytes()).hexdigest() != settings["model_sha256"]:
                raise ValueError("Hand-landmark model differs from training; use --hand-model with the original asset")
            sequence, _ = extractor.extract(args.sample)
            x = resample(sequence)
    probabilities = model.predict(x[None], verbose=0)[0]
    print(json.dumps({label: float(probabilities[i]) for i, label in
                      sorted(enumerate(classes), key=lambda pair: -probabilities[pair[0]])}, indent=2))


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return result


def nonnegative(value):
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and >= 0")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Split original sources, resample videos, synthesize grouped image variants")
    add_arguments(prep)
    add_source_arguments(prep)
    add_preprocessing_arguments(prep)
    prep.add_argument("--source", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--groups", help="Optional CSV: path,group; use signer/session IDs")
    prep.add_argument("--validation", type=float, default=0.15)
    prep.add_argument("--test", type=float, default=0.15)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--static-variants", type=positive_int, default=8)
    prep.add_argument("--jitter", type=nonnegative, default=None,
                      help="Default: 0.0075 image units, or 0.03 palm lengths for hand-relative preprocessing")
    prep.add_argument("--hand-model", help="Custom MediaPipe .task file; default model downloads automatically")
    prep.add_argument("--detection-confidence", type=nonnegative, default=0.5)
    prep.add_argument("--swap-hands", action="store_true", help="Swap reported Left/Right slots consistently")
    prep.set_defaults(func=prepare)
    check = sub.add_parser("inspect", help="Validate data, shape, classes and split isolation")
    check.add_argument("--dataset", required=True)
    check.set_defaults(func=lambda a: print(json.dumps(inspect_dataset(a.dataset)[2], indent=2)))
    fit = sub.add_parser("train")
    fit.add_argument("--dataset", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--epochs", type=positive_int, default=100)
    fit.add_argument("--batch-size", type=positive_int, default=32)
    fit.add_argument("--learning-rate", type=nonnegative, default=0.001)
    fit.add_argument("--patience", type=positive_int, default=12)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--class-weights", action="store_true")
    fit.add_argument("--loader", choices=["packed", "files"], default="packed",
                     help="Packed disk-backed arrays (default), or the original per-file loader")
    fit.add_argument("--resume-from", help="Continue an existing best.keras checkpoint into a new run")
    fit.set_defaults(func=train)
    infer = sub.add_parser("predict")
    infer.add_argument("--run", required=True)
    infer.add_argument("--sample", required=True)
    infer.add_argument("--hand-model", help="Custom MediaPipe .task file used during training")
    infer.set_defaults(func=predict)
    args = parser.parse_args()
    try:
        if args.command == "prepare" and args.jitter == 0:
            raise ValueError("--jitter must be > 0 to avoid flat synthetic sequences")
        if args.command == "train" and args.learning_rate == 0:
            raise ValueError("--learning-rate must be > 0")
        args.func(args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
