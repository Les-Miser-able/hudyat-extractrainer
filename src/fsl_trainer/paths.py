"""Default locations follow this source checkout, never the working directory.

Install with ``pip install -e .`` after copying the project to a new machine.
Explicit CLI paths retain normal working-directory-relative semantics.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "default"
CACHE = ROOT / "data" / "cache"
RUNS = ROOT / "runs"
HAND_MODEL = ROOT / "assets" / "hand_landmarker.task"
