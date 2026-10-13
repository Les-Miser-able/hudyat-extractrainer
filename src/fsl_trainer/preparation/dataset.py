"""Dataset validation and preparation reports shared by pipeline stages."""
import json
from pathlib import Path
from fsl_trainer.common import Stage, progress, save_json
from .sequences import load_sequence

def save_augmentation_summary(root, manifest):
    """Count accepted originals separately from generated output sequences."""
    def counts(rows):
        sources = {row["source"]: row["static"] for row in rows}
        static = sum(sources.values())
        return {
            "original_sources": len(sources),
            "static_sources": static,
            "sequence_sources": len(sources) - static,
            "augmented_static_sequences": sum(bool(row["synthetic"]) for row in rows),
            "resampled_sequences": sum(not row["synthetic"] for row in rows),
            "total_output_sequences": len(rows),
        }

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
    print(f"Resampled video/sequence outputs: {summary['resampled_sequences']:,}")
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

