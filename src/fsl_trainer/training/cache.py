<<<<<<< HEAD:training_cache.py
"""Compatibility entry point for fsl_trainer.training.cache."""
import sys
from fsl_trainer.training import cache as _implementation

# Keep legacy imports and patches attached to the implementation module.
sys.modules[__name__] = _implementation
=======
"""Validated disk-backed training arrays; no per-sample file opens during epochs."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from fsl_trainer.common import Stage, save_json
from fsl_trainer.paths import CACHE
from fsl_trainer.preparation.dataset import inspect_dataset

SPLITS = ("train", "validation", "test")
VERSION = 1


def snapshot(root):
    """Fast directory metadata scan. Reject links; changes invalidate the cache."""
    files = {}
    with Stage("Checking source file sizes and modification times"):
        for split in SPLITS:
            folder = root / split
            if folder.is_symlink() or (hasattr(folder, "is_junction") and folder.is_junction()):
                raise ValueError(f"Linked split directories are unsupported: {folder}")
            with os.scandir(folder) as classes:
                for cls in classes:
                    if not cls.is_dir(follow_symlinks=False):
                        if cls.is_symlink():
                            raise ValueError("Linked class directories are unsupported")
                        continue
                    class_path = Path(cls.path)
                    if hasattr(class_path, "is_junction") and class_path.is_junction():
                        raise ValueError("Linked class directories are unsupported")
                    with os.scandir(cls.path) as entries:
                        for entry in entries:
                            if not entry.name.endswith(".npy"):
                                continue
                            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                                raise ValueError(f"Linked/non-file sample: {entry.path}")
                            stat = entry.stat(follow_symlinks=False)
                            files[f"{split}/{cls.name}/{entry.name}"] = [stat.st_size, stat.st_mtime_ns]
    return files


def close_arrays(arrays):
    for x, y in arrays.values():
        for array in (x, y):
            array.flush()
            array._mmap.close()


def open_cache(root, cache_root=None):
    root = Path(root).resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    preprocessing_bytes = (root / "preprocessing.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if not manifest:
        raise ValueError("Empty dataset manifest")
    classes = sorted({r["label"] for r in manifest})
    rows = {s: [r for r in manifest if r["split"] == s] for s in SPLITS}
    if any(not items for items in rows.values()):
        raise ValueError("All three splits must be nonempty")
    before = snapshot(root)
    if set(before) != {r["path"] for r in manifest}:
        raise ValueError("Dataset files and manifest disagree")
    fingerprint = hashlib.sha256(manifest_bytes + preprocessing_bytes +
                                 json.dumps(before, sort_keys=True).encode() +
                                 str(VERSION).encode()).hexdigest()
    cache_root = Path(cache_root) if cache_root is not None else CACHE
    cache = cache_root / fingerprint
    marker = cache / "cache.json"
    if marker.is_file():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata["fingerprint"] != fingerprint or metadata["classes"] != classes:
            raise ValueError("Invalid training-cache metadata")
        for split in SPLITS:
            for name, shape, dtype in (
                (f"{split}_x.npy", (len(rows[split]), 32, 128), np.dtype("float32")),
                (f"{split}_y.npy", (len(rows[split]),), np.dtype("int32"))):
                path = cache / name
                stat = path.stat()
                if metadata["files"][name] != [stat.st_size, stat.st_mtime_ns]:
                    raise ValueError(f"Training cache changed: {path}. Choose --loader files or rebuild the cache.")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                valid = array.shape == shape and array.dtype == dtype
                array._mmap.close()
                if not valid:
                    raise ValueError(f"Invalid cached array: {path}")
        print("Reusing validated consolidated training cache.", flush=True)
        return cache, manifest, classes, metadata["stats"]
    cache_root.mkdir(parents=True, exist_ok=True)
    print("Building training cache once: validating and packing each sequence. No extraction is needed.", flush=True)
    with tempfile.TemporaryDirectory(prefix="building-", dir=cache_root) as temporary:
        stage = Path(temporary)
        arrays = {}
        try:
            for split in SPLITS:
                arrays[split] = (
                    np.lib.format.open_memmap(stage / f"{split}_x.npy", mode="w+",
                                             dtype=np.float32, shape=(len(rows[split]), 32, 128)),
                    np.lib.format.open_memmap(stage / f"{split}_y.npy", mode="w+",
                                             dtype=np.int32, shape=(len(rows[split]),)))
            positions = {s: 0 for s in SPLITS}
            labels = {label: i for i, label in enumerate(classes)}
            def collect(row, sequence):
                split = row["split"]
                index = positions[split]
                arrays[split][0][index] = sequence
                arrays[split][1][index] = labels[row["label"]]
                positions[split] += 1
            checked, checked_classes, stats = inspect_dataset(root, on_sample=collect)
            if checked != manifest or checked_classes != classes:
                raise ValueError("Manifest changed during cache construction")
        finally:
            close_arrays(arrays)
        if (root / "manifest.json").read_bytes() != manifest_bytes or \
           (root / "preprocessing.json").read_bytes() != preprocessing_bytes or snapshot(root) != before:
            raise ValueError("Dataset changed while building cache; try again when it is idle")
        file_stats = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in stage.glob("*.npy")}
        save_json(stage / "cache.json", {"fingerprint": fingerprint, "classes": classes,
                                       "stats": stats, "files": file_stats, "version": VERSION})
        stage.rename(cache)
    print(f"Consolidated training cache ready: {cache}", flush=True)
    return cache, manifest, classes, stats


def batches(cache, split, batch_size, rng=None):
    """Read one batch at a time, with exact sample coverage and aligned labels."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    x = np.load(Path(cache) / f"{split}_x.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(Path(cache) / f"{split}_y.npy", mmap_mode="r", allow_pickle=False)
    try:
        indices = np.arange(len(y))
        if rng is not None:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start+batch_size]
            yield np.array(x[ids], copy=True), np.array(y[ids], copy=True)
    finally:
        x._mmap.close()
        y._mmap.close()
>>>>>>> 46d0e1656159055e1137771beb6aeb8d123f107d:src/fsl_trainer/training/cache.py
