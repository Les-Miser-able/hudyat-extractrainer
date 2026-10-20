"""Small readers shared by the native monitors; no GUI or ML imports."""
import csv
import json
import math
from pathlib import Path
import re
import shutil


def read_history(run):
    """CSVLogger flushes each epoch, before history.json exists at training end."""
    run = Path(run)
    keys = ("accuracy", "val_accuracy", "loss", "val_loss")
    try:
        with (run / "history.csv").open(encoding="utf-8", newline="") as handle:
            result = {key: [] for key in keys}
            for row in csv.DictReader(handle):
                try:
                    values = {key: float(row[key]) for key in keys}
                except (KeyError, TypeError, ValueError):
                    continue
                if all(math.isfinite(value) for value in values.values()):
                    for key, value in values.items():
                        result[key].append(value)
            if result["loss"]:
                return result
    except OSError:
        pass
    try:
        saved = json.loads((run / "history.json").read_text(encoding="utf-8"))
        return {key: [float(v) for v in saved.get(key, []) if isinstance(v, (int, float))
                      and math.isfinite(v)] for key in keys}
    except (OSError, ValueError, TypeError, AttributeError):
        return {key: [] for key in keys}


def stage_progress(logs):
    for line in reversed(logs):
        match = re.search(r"\[([^\]]+)\].*?([\d,]+)/([\d,]+) \(([\d.]+)%\)", line)
        if match:
            return match[1], int(match[2].replace(",", "")), int(match[3].replace(",", ""))
    return None


def import_category(source, destination):
    """Copy into a NEW category; never merge or replace existing recordings."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise ValueError("Choose a folder containing gesture class folders.")
    if destination == source or destination in source.parents or source in destination.parents:
        raise ValueError("Source and destination must be separate, non-nested folders.")
    if destination.exists():
        raise ValueError("That category already exists. Choose a new category name.")
    if not any(p.is_dir() for p in source.iterdir()):
        raise ValueError("The selected folder must contain a folder for each gesture class.")
    for path in source.rglob("*"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("Import a source folder without symbolic links or junctions.")
    shutil.copytree(source, destination)
    return str(destination)
