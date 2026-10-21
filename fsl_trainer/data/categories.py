"""Prepare and train one independent gesture model per category."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from .augmentation_options import add_arguments, extra_samples, time_warp, mirror_hands
from .source_selection import add_arguments as add_source_arguments, folder_selection, source_files
from ..paths import PROJECT_ROOT
from .dataset_paths import dataset_name, select_dataset
from .signer_splits import (add_arguments as add_split_arguments, command_options as split_command_options,
                            signer_from_source, split_signers, fold_datasets)
from .preprocessing import IMAGE, HAND_RELATIVE, preparation_normalization, add_arguments as add_preprocessing_arguments
from .source_selection import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

BASE = PROJECT_ROOT


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
    if getattr(args, 'report_only', False) and not getattr(args, 'loso_all', False):
        raise ValueError('--report-only requires --loso-all.')
    if not getattr(args, 'loso_all', False):
        return _build_jobs(args)
    if getattr(args, 'test_signer', None) or getattr(args, 'validation_signer', None):
        raise ValueError('--loso-all rotates test/validation signers automatically; omit explicit signer holdouts.')
    if getattr(args, 'resume_from', None):
        raise ValueError('--loso-all trains a fresh model per fold; checkpoints cannot be shared across folds.')
    preparing = args.command == 'prepare'
    root = Path(args.data if preparing else args.datasets).resolve()
    available = discover(root)
    selected = list(dict.fromkeys(args.category)) if args.category else available
    if set(selected) - set(available):
        raise ValueError('Unknown categories: ' + ', '.join(sorted(set(selected) - set(available))))
    name = dataset_name(getattr(args, 'dataset_name', None) or 'prepared_v1')
    jobs, skipped = [], []
    for category in selected:
        if preparing:
            records = []
            source = root / category
            for folder in sorted(source.iterdir()):
                if not folder.is_dir() or folder.name.startswith('.'):
                    continue
                for path in source_files(folder, getattr(args, 'source_folders', None), getattr(args, 'videos_only', False)):
                    relative = path.relative_to(source).as_posix()
                    records.append({'source': relative, 'label': folder.name, 'group': relative,
                                    'signer': signer_from_source(relative)})
            if not records and not args.category:
                skipped.append(category)
                continue
            signers = sorted({r['signer'] for r in records})
            if not signers:
                raise ValueError(f'{category}: no signer recordings found.')
            for signer in signers:
                split_signers(records, signer)
            versions = [(f'{name}-loso-{signer}', signer) for signer in signers]
        else:
            versions = [(p.name, p.name[len(name + '-loso-'):]) for p in fold_datasets(root / category, name)]
        for version, signer in versions:
            fold = argparse.Namespace(**vars(args))
            fold.category, fold.dataset_name, fold.loso_all = [category], version, False
            if preparing:
                fold.split_strategy, fold.test_signer, fold.validation_signer = 'loso', signer, None
            else:
                fold.run = category_name(args.run + '-loso-' + signer)
            new_jobs, _ = _build_jobs(fold)
            jobs.extend((f'{category} / test {signer}', target, command) for _, target, command in new_jobs)
    if not jobs:
        raise ValueError('No populated signer categories found.')
    return jobs, skipped


def _build_jobs(args):
    preparing = args.command == "prepare"
    version = getattr(args, "dataset_name", "prepared_v1" if preparing else None)
    if preparing or (version is not None and version != "."):
        version = dataset_name(version)
    if preparing:
        selected_folders = folder_selection(getattr(args, "source_folders", None))
        videos_only = getattr(args, "videos_only", False)
        video_count = extra_samples(getattr(args, "video_extra_samples", 0))
        video_strength = time_warp(getattr(args, "video_time_warp", .05))
        video_mirroring = mirror_hands(getattr(args, 'video_mirror_hands', False))
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
        if not preparing:
            source = select_dataset(source, version)
        has_data = (any(source_files(p, selected_folders, videos_only) for p in source.iterdir() if p.is_dir()) if preparing
                    else source is not None)
        if not has_data:
            if args.category:
                raise ValueError(f"{name}: no {'images, videos or landmark samples' if preparing else 'prepared dataset'} found")
            skipped.append(name)
            continue
        target = target_root / name / version if preparing else target_root / name / args.run
        if not getattr(args, 'report_only', False) and target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise ValueError(f"Output already contains data: {target}. "
                             "Choose another --dataset-name or --run name.")
        command = [sys.executable, "-B", str(BASE / "trainer.py")]
        if preparing:
            has_media = any(p.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
                            for folder in source.iterdir() if folder.is_dir()
                            for p in source_files(folder, selected_folders, videos_only))
            normalization = preparation_normalization(getattr(args, "normalization", "auto"), name, has_media)
            jitter = getattr(args, "jitter", None)
            if jitter is None:
                jitter = .03 if normalization == HAND_RELATIVE else .0075
            command += ["prepare", "--source", str(source), "--output", str(target),
                        "--validation", str(args.validation), "--test", str(args.test),
                        "--seed", str(args.seed), "--static-variants", str(args.static_variants),
                        "--jitter", str(jitter)]
            command += split_command_options(args)
            if normalization != IMAGE:
                command += ["--normalization", normalization]
            for folder in selected_folders or []:
                command += ["--source-folder", folder]
            if videos_only:
                command += ["--videos-only"]
            if video_count:
                command += ["--video-extra-samples", str(video_count),
                            "--video-time-warp", str(video_strength)]
            if video_mirroring:
                command.append('--video-mirror-hands')
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
        raise ValueError("No populated categories found. Add images, videos or .npy files under data/<category>/<gesture>/ "
                         "and prepare before training.")
    return jobs, skipped


def execute(args):
    jobs, skipped = build_jobs(args)
    if skipped:
        print("Skipping empty categories: " + ", ".join(skipped), flush=True)
    failures = []
    for name, target, command in ([] if getattr(args, 'report_only', False) else jobs):
        print(f"\n{args.command}: {name} -> {target}", flush=True)
        # A fresh process builds a fresh model and releases its memory on exit.
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append(name)
            print(f"FAILED: {name}; exit code {result.returncode}", file=sys.stderr, flush=True)
        else:
            print(f"Completed: {name}", flush=True)
    if args.command == 'train' and getattr(args, 'loso_all', False):
        from ..reporting.paper import save_loso_report
        for category in sorted({target.parent.name for _, target, _ in jobs}):
            folds = [(Path(command[command.index('--dataset') + 1]), target)
                     for _, target, command in jobs if target.parent.name == category]
            save_loso_report(folds, BASE / 'reports' / 'paper' / 'loso' / category / args.run)
    if failures:
        raise RuntimeError("Failed jobs: " + ", ".join(failures) +
                           ". Other successful outputs have been kept; retry failed categories explicitly.")
    description = ('LOSO reports refreshed' if getattr(args, 'report_only', False) else
                   f"Completed {len(jobs)} " + ('signer folds' if getattr(args, 'loso_all', False) else 'independent category jobs'))
    print('\n' + description + '.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="Show categories, gesture folders and source counts")
    listing.add_argument("--data", default=str(BASE / "data"))
    for action in ("prepare", "train"):
        cmd = commands.add_parser(action)
        choice = cmd.add_mutually_exclusive_group(required=True)
        choice.add_argument("--category", action="append", type=category_name,
                            help="One category; repeat to select several")
        choice.add_argument("--all", action="store_true", help="Process every populated category")
        cmd.add_argument("--datasets", default=str(BASE / "datasets_augmented"))
        cmd.add_argument("--dataset-name", default="prepared_v1" if action == "prepare" else None,
                         help="Prepared folder under <datasets>/<category>; use . to train an original category-level dataset")
        cmd.add_argument("--seed", type=int, default=42)
        cmd.add_argument('--loso-all', action='store_true',
                         help='Prepare/train one independent fold per signer; dataset/run names get -loso-<signer> suffixes')
        if action == "prepare":
            add_arguments(cmd)
            add_source_arguments(cmd)
            add_split_arguments(cmd)
            add_preprocessing_arguments(cmd, default="auto")
            cmd.add_argument("--data", default=str(BASE / "data"))
            cmd.add_argument("--groups-dir", help="Directory with one <category>.csv per selected category")
            cmd.add_argument("--validation", type=float, default=0.15)
            cmd.add_argument("--test", type=float, default=0.15)
            cmd.add_argument("--static-variants", type=int, default=8)
            cmd.add_argument("--jitter", type=float, default=None,
                             help="Default: 0.0075 image units, or 0.03 palm lengths for raw alphabet media")
            cmd.add_argument("--hand-model", help="Custom MediaPipe .task file")
            cmd.add_argument("--detection-confidence", type=float, default=0.5)
            cmd.add_argument("--swap-hands", action="store_true")
        else:
            cmd.add_argument("--runs", default=str(BASE / "runs"))
            cmd.add_argument("--run", type=category_name,
                             default=datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
            cmd.add_argument("--epochs", type=int, default=100)
            cmd.add_argument("--batch-size", type=int, default=32)
            cmd.add_argument("--learning-rate", type=float, default=0.001)
            cmd.add_argument("--patience", type=int, default=12)
            cmd.add_argument("--class-weights", action="store_true")
            cmd.add_argument("--loader", choices=["packed", "files"], default="packed")
            cmd.add_argument("--resume-from", help="Existing best.keras; select exactly one category")
            cmd.add_argument('--report-only', action='store_true',
                             help='With --loso-all, refresh aggregate reports from saved results without training')
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
