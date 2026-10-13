"""Prepare and train one independent gesture model per category."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from fsl_trainer.preparation.extraction import source_files

from fsl_trainer.paths import ROOT as BASE, RAW, PROCESSED, RUNS


def category_name(value):
    if not value or value in (".", "..") or any(c in value for c in '/\\:*?"<>|'):
        raise argparse.ArgumentTypeError("Use a single folder name, not a path")
    return value


def discover(root):
    if not root.is_dir():
        raise ValueError(f"Category root does not exist: {root}")
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and not p.name.startswith("."))


def inventory(root):
    result = {}
    for name in discover(root):
        folder = root / name
        classes = {p.name: len(source_files(p)) for p in sorted(folder.iterdir())
                   if p.is_dir() and not p.name.startswith(".")}
        result[name] = {"classes": classes, "source_files": sum(classes.values()),
                        "status": "has samples" if sum(classes.values()) else "empty"}
    return result


def build_jobs(args):
    preparing = args.command == "prepare"
    source_root = Path(args.data if preparing else args.datasets).resolve()
    target_root = Path(args.datasets if preparing else args.runs).resolve()
    if (source_root == target_root or source_root in target_root.parents
            or target_root in source_root.parents):
        raise ValueError("Input and output roots must be separate, non-nested directories")
    available = discover(source_root)
    selected = list(dict.fromkeys(args.category)) if args.category else available
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown categories: {', '.join(sorted(unknown))}")
    jobs, skipped = [], []
    for name in selected:
        source = source_root / name
        has_data = (any(source_files(p) for p in source.iterdir() if p.is_dir()) if preparing
                    else (source / "manifest.json").is_file())
        if not has_data:
            if args.category:
                raise ValueError(f"{name}: no {'images, videos or landmark samples' if preparing else 'prepared dataset'} found")
            skipped.append(name)
            continue
        target = target_root / name if preparing else target_root / name / args.run
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise ValueError(f"Output already contains data: {target}. "
                             "Choose another --datasets root or --run name.")
        command = [sys.executable, "-B", str(BASE / "trainer.py")]
        if preparing:
            command += ["prepare", "--source", str(source), "--output", str(target),
                        "--validation", str(args.validation), "--test", str(args.test),
                        "--seed", str(args.seed), "--static-variants", str(args.static_variants),
                        "--jitter", str(args.jitter)]
            command += ["--detection-confidence", str(getattr(args, "detection_confidence", 0.5))]
            if getattr(args, "hand_model", None):
                command += ["--hand-model", args.hand_model]
            if getattr(args, "swap_hands", False):
                command.append("--swap-hands")
            if args.groups_dir:
                groups = Path(args.groups_dir).resolve() / f"{name}.csv"
                if not groups.is_file():
                    raise ValueError(f"Missing group CSV: {groups}")
                command += ["--groups", str(groups)]
        else:
            command += ["train", "--dataset", str(source), "--output", str(target),
                        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                        "--learning-rate", str(args.learning_rate),
                        "--patience", str(args.patience), "--seed", str(args.seed)]
            command += ["--loader", getattr(args, "loader", "packed")]
            if getattr(args, "resume_from", None):
                if not args.category or len(args.category) != 1:
                    raise ValueError("--resume-from requires exactly one --category")
                command += ["--resume-from", args.resume_from]
            if args.class_weights:
                command.append("--class-weights")
        jobs.append((name, target, command))
    if not jobs:
        raise ValueError("No populated categories found. Add images, videos or .npy files under data/raw/<category>/<gesture>/ "
                         "and prepare before training.")
    return jobs, skipped


def execute(args):
    jobs, skipped = build_jobs(args)
    if skipped:
        print("Skipping empty categories: " + ", ".join(skipped), flush=True)
    failures = []
    for name, target, command in jobs:
        print(f"\n{args.command}: {name} -> {target}", flush=True)
        # A fresh process builds a fresh model and releases its memory on exit.
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append(name)
            print(f"FAILED: {name}; exit code {result.returncode}", file=sys.stderr, flush=True)
        else:
            print(f"Completed: {name}", flush=True)
    if failures:
        raise RuntimeError("Failed categories: " + ", ".join(failures) +
                           ". Other successful outputs have been kept; retry failed categories explicitly.")
    print(f"\nCompleted {len(jobs)} independent category jobs.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="Show categories, gesture folders and source counts")
    listing.add_argument("--data", default=str(RAW))
    for action in ("prepare", "train"):
        cmd = commands.add_parser(action)
        choice = cmd.add_mutually_exclusive_group(required=True)
        choice.add_argument("--category", action="append", type=category_name,
                            help="One category; repeat to select several")
        choice.add_argument("--all", action="store_true", help="Process every populated category")
        cmd.add_argument("--datasets", default=str(PROCESSED))
        cmd.add_argument("--seed", type=int, default=42)
        if action == "prepare":
            cmd.add_argument("--data", default=str(RAW))
            cmd.add_argument("--groups-dir", help="Directory with one <category>.csv per selected category")
            cmd.add_argument("--validation", type=float, default=0.15)
            cmd.add_argument("--test", type=float, default=0.15)
            cmd.add_argument("--static-variants", type=int, default=8)
            cmd.add_argument("--jitter", type=float, default=0.0075)
            cmd.add_argument("--hand-model", help="Custom MediaPipe .task file")
            cmd.add_argument("--detection-confidence", type=float, default=0.5)
            cmd.add_argument("--swap-hands", action="store_true")
        else:
            cmd.add_argument("--runs", default=str(RUNS))
            cmd.add_argument("--run", type=category_name,
                             default=datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
            cmd.add_argument("--epochs", type=int, default=100)
            cmd.add_argument("--batch-size", type=int, default=32)
            cmd.add_argument("--learning-rate", type=float, default=0.001)
            cmd.add_argument("--patience", type=int, default=12)
            cmd.add_argument("--class-weights", action="store_true")
            cmd.add_argument("--loader", choices=["packed", "files"], default="packed")
            cmd.add_argument("--resume-from", help="Existing best.keras; select exactly one category")
    args = parser.parse_args()
    try:
        if args.command == "list":
            print(json.dumps(inventory(Path(args.data).resolve()), indent=2))
        else:
            execute(args)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
