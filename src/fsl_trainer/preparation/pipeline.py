"""Prepare raw media or supplied landmark arrays without changing originals."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from fsl_trainer.common import FRAMES, FEATURES, LAYOUT, Stage, progress, save_json
from .sequences import load_sequence, resample, synthetic_sequence
from .split import split_sources
from .dataset import inspect_dataset, save_augmentation_summary

def prepare(args):
    source, dest = Path(args.source).resolve(), Path(args.output).resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory not found: {source}")
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Output must be empty; existing datasets are never overwritten")
    if not getattr(args, "_landmarks_only", False):
        from .extraction import source_files, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, prepare_media
        if any(p.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS for p in source_files(source)):
            return prepare_media(args, prepare)
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
    records, fingerprints = [], {}
    for folder in progress(sorted(source.iterdir()), "Checking source classes"):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        paths = sorted(folder.glob("*.npy"))
        if not paths:
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
    print(f"Generating {sum(args.static_variants if r['static'] else 1 for r in records):,} sequences from {len(records):,} source files.", flush=True)
    for record in progress(records, "Generating and saving sequences (source files)"):
        split = assignment[record["group"]]
        x = load_sequence(source / record["source"])
        # Assign the source first; all synthetic siblings inherit the same split.
        count = args.static_variants if record["static"] else 1
        for variant in range(count):
            if record["static"]:
                sequence = synthetic_sequence(x, rng, args.jitter)
            else:
                sequence = resample(x)
            token = hashlib.sha256(record["source"].encode()).hexdigest()[:16]
            relative = Path(split) / record["label"] / f"{token}_{variant:03d}.npy"
            (dest / relative).parent.mkdir(parents=True, exist_ok=True)
            np.save(dest / relative, sequence.astype(np.float32), allow_pickle=False)
            manifest.append({**record, "split": split, "path": relative.as_posix(),
                             "variant": variant, "variant_of": record["source"], "synthetic": record["static"]})
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
    print(json.dumps(inspect_dataset(dest)[2], indent=2), flush=True)
    print("Sequence generation and dataset validation complete.", flush=True)
    if not getattr(args, "_landmarks_only", False):
        save_augmentation_summary(dest, manifest)

