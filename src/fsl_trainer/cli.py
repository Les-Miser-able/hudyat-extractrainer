"""Prepare, inspect, train, or predict with the FSL Conv1D-BiLSTM trainer."""
import argparse
import json
import numpy as np
from fsl_trainer.preparation.pipeline import prepare
from fsl_trainer.preparation.dataset import inspect_dataset
from fsl_trainer.training.trainer import train
from fsl_trainer.training.predict import predict

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
    prep.add_argument("--source", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--groups", help="Optional CSV: path,group; use signer/session IDs")
    prep.add_argument("--validation", type=float, default=0.15)
    prep.add_argument("--test", type=float, default=0.15)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--static-variants", type=positive_int, default=8)
    prep.add_argument("--jitter", type=nonnegative, default=0.0075)
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

