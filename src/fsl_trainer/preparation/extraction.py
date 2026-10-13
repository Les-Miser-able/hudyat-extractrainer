<<<<<<< HEAD:extraction.py
"""Compatibility entry point for fsl_trainer.data.extraction."""
import sys
from fsl_trainer.data import extraction as _implementation

# Keep legacy imports and patches attached to the implementation module.
sys.modules[__name__] = _implementation
=======
"""Raw image/video -> normalized two-slot hand landmarks."""
import argparse
import csv
from contextlib import contextmanager
from fsl_trainer.common import Stage, progress
from datetime import datetime
import hashlib
import json
import re
from pathlib import Path
import shutil
import tempfile
import urllib.request
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SUPPORTED = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | {".npy"}
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
from fsl_trainer.paths import HAND_MODEL as DEFAULT_MODEL


def source_files(folder):
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() in SUPPORTED
                  and not any(part.startswith(".") for part in p.relative_to(folder).parts))


def default_group(relative):
    # Roboflow exports encode the common original name before .rf.<hash>.
    path = Path(relative)
    stem = path.name.split(".rf.")[0] if ".rf." in path.name else path.name
    if ".rf." in path.name:
        stem = re.sub(r"_(jpg|jpeg|png|webp|bmp)$", r".\1", stem, flags=re.IGNORECASE)
    return (path.parent / stem).as_posix()


def ensure_model(path):
    path = Path(path)
    if not path.is_file():
        if path.resolve() != DEFAULT_MODEL.resolve():
            raise ValueError(f"Hand model not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading MediaPipe hand-landmark model (one-time setup)...", flush=True)
        temporary = path.with_suffix(".download")
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=90) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def pack_result(result, swap_hands=False):
    """Use reported handedness, not detector list order; preserve binary flags."""
    out = np.zeros(128, dtype=np.float32)
    scores = [-1.0, -1.0]
    for landmarks, categories in zip(result.hand_landmarks, result.handedness):
        if not categories or len(landmarks) != 21:
            continue
        label = categories[0].category_name.lower()
        if label not in ("left", "right"):
            continue
        hand = (0 if label == "left" else 1) ^ int(swap_hands)
        score = float(categories[0].score)
        if score <= scores[hand]:
            continue
        xyz = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
        if not np.isfinite(xyz).all():
            continue
        out[hand*63:(hand+1)*63] = xyz.ravel()
        out[126+hand] = 1
        scores[hand] = score
    return out


class Extractor:
    def __init__(self, model=DEFAULT_MODEL, confidence=0.5, swap_hands=False):
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Detection confidence must be between 0 and 1")
        try:
            import mediapipe as mp
            import cv2
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise RuntimeError("Install raw-media dependencies: python -m pip install -r requirements.txt") from exc
        self.mp, self.cv2, self.Image, self.ImageOps = mp, cv2, Image, ImageOps
        self.model = ensure_model(model)
        self.confidence, self.swap_hands = confidence, swap_hands
        self.image_detector = self.detector(video=False)

    def detector(self, video):
        vision = self.mp.tasks.vision
        return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=self.mp.tasks.BaseOptions(model_asset_path=str(self.model)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=self.confidence,
            min_hand_presence_confidence=self.confidence, min_tracking_confidence=0.5))

    def image(self, rgb):
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(rgb))

    def extract(self, path):
        path = Path(path)
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            with self.Image.open(path) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Animated/multipage images are unsupported; use a video")
                rgb = np.array(self.ImageOps.exif_transpose(image).convert("RGB"))
            x = pack_result(self.image_detector.detect(self.image(rgb)), self.swap_hands)[None]
            if not x[:, 126:].any():
                raise ValueError("No hand detected")
            return x, {"kind": "image", "frames": 1, "detected_frames": 1}
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported media type: {path.suffix}")
        capture = self.cv2.VideoCapture(str(path))
        frames, previous, index = [], -1, 0
        try:
            if not capture.isOpened():
                raise ValueError("Video could not be opened")
            fps = float(capture.get(self.cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("Video has invalid FPS; convert to a constant-frame-rate video")
            expected = int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT))
            # A new VIDEO detector per clip resets tracking and timestamp state.
            with self.detector(video=True) as detector:
                while True:
                    ok, bgr = capture.read()
                    if not ok:
                        break
                    timestamp = max(previous+1, round(index*1000/fps))
                    rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
                    result = detector.detect_for_video(self.image(rgb), timestamp)
                    frames.append(pack_result(result, self.swap_hands))
                    previous, index = timestamp, index+1
                    if index % 500 == 0:
                        print(f"  {path.name}: extracted {index} frames", flush=True)
            if len(frames) < 2:
                raise ValueError("Video must contain at least two decodable frames")
            if expected > 0 and len(frames) < expected:
                raise ValueError(f"Video ended early: decoded {len(frames)} of {expected} declared frames")
            x = np.stack(frames)
            detected = int(np.any(x[:, 126:] > 0, axis=1).sum())
            if not detected:
                raise ValueError("No hand detected in any video frame")
            return x, {"kind": "video", "frames": len(x), "detected_frames": detected, "fps": fps}
        finally:
            capture.release()

    def close(self):
        self.image_detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()



@contextmanager
def temporary_landmarks():
    temporary = tempfile.TemporaryDirectory(prefix="fsl-extraction-")
    try:
        yield temporary.name
    finally:
        with Stage("Cleaning up temporary landmark files"):
            temporary.cleanup()


def prepare_media(args, prepare_landmarks):
    """Stage extraction, then reuse the tested splitter and repeat+jitter pipeline."""
    from .sequences import load_sequence
    from .dataset import inspect_dataset, save_augmentation_summary
    from fsl_trainer.common import save_json
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be empty; choose a new dataset folder")
    folders = [p for p in sorted(source.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    all_files = [p for folder in folders for p in source_files(folder)]
    mapping = None
    if args.groups:
        with open(args.groups, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        mapping = {}
        for row in rows:
            key, group = row["path"].replace("\\", "/"), row["group"].strip()
            if key in mapping or not group:
                raise ValueError("Group CSV contains duplicate paths or empty group IDs")
            mapping[key] = group
        if set(mapping) != {p.relative_to(source).as_posix() for p in all_files}:
            raise ValueError("Group CSV must cover every supported raw source exactly once")
    ignored = [{"source": p.relative_to(source).as_posix(), "reason": "Unsupported extension"}
               for folder in folders for p in sorted(folder.rglob("*"))
               if p.is_file() and p.suffix.lower() not in SUPPORTED
               and not any(part.startswith(".") for part in p.relative_to(source).parts)]
    report = {"source": str(source), "accepted": [], "rejected": [], "ignored": ignored, "status": "extracting"}
    if ignored:
        print(f"Ignoring {len(ignored)} unsupported files; details will be in the extraction report.", flush=True)
    extractor = Extractor(model=getattr(args, "hand_model", None) or DEFAULT_MODEL,
                          confidence=getattr(args, "detection_confidence", 0.5),
                          swap_hands=getattr(args, "swap_hands", False))
    try:
        with temporary_landmarks() as temp:
            stage = Path(temp) / "landmarks"
            for folder in folders:
                (stage / folder.name).mkdir(parents=True)
            provenance, groups, fingerprints = {}, [], {}
            for i, path in enumerate(all_files):
                original = path.relative_to(source).as_posix()
                try:
                    if path.suffix.lower() == ".npy":
                        x = load_sequence(path)
                        details = {"kind": "landmarks", "frames": len(x),
                                   "detected_frames": int(np.any(x[:, 126:] > 0, axis=1).sum())}
                    else:
                        x, details = extractor.extract(path)
                    fingerprint = hashlib.sha256(x.tobytes()).hexdigest()
                    if fingerprint in fingerprints:
                        previous = fingerprints[fingerprint]
                        if Path(previous).parts[0] != Path(original).parts[0]:
                            raise RuntimeError(f"Identical landmark data has conflicting labels: {previous}, {original}")
                        raise ValueError(f"Duplicate landmark sequence of {previous}")
                    fingerprints[fingerprint] = original
                    staged = Path(original).parts[0] + "/" + hashlib.sha256(original.encode()).hexdigest() + ".npy"
                    np.save(stage / staged, x, allow_pickle=False)
                    provenance[staged] = original
                    groups.append([staged, mapping[original] if mapping else default_group(original)])
                    report["accepted"].append({"source": original, **details})
                except (ValueError, OSError) as error:
                    report["rejected"].append({"source": original, "reason": str(error)})
                if (i+1) % 50 == 0 or i+1 == len(all_files):
                    print(f"Extraction {i+1}/{len(all_files)}: {len(report['accepted'])} accepted, "
                          f"{len(report['rejected'])} rejected", flush=True)
            print("Extraction finished. Next: check landmarks, split source groups, generate variants, and validate output.", flush=True)
            group_path = Path(temp) / "groups.csv"
            with group_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["path", "group"])
                writer.writerows(groups)
            staged_args = argparse.Namespace(**vars(args))
            staged_args.source, staged_args.groups = str(stage), str(group_path)
            staged_args._landmarks_only = True
            prepare_landmarks(staged_args)
            print("Restoring original source filenames and extraction settings...", flush=True)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            for row in manifest:
                row["source"] = provenance[row["source"]]
                row["variant_of"] = row["source"]
            save_json(output / "manifest.json", manifest)
            metadata = json.loads((output / "preprocessing.json").read_text(encoding="utf-8"))
            metadata["extraction"] = {
                "backend": "MediaPipe HandLandmarker", "mediapipe_version": extractor.mp.__version__,
                "model_sha256": hashlib.sha256(extractor.model.read_bytes()).hexdigest(),
                "model_url": MODEL_URL, "num_hands": 2,
                "detection_confidence": extractor.confidence, "swap_hands": extractor.swap_hands,
                "normalization": "MediaPipe image landmarks: x/width, y/height, wrist-relative z at x scale",
                "image_orientation": "EXIF transpose; no horizontal flip",
                "video_timeline": "all decoded frames, constant FPS timeline; missing frames zero-filled"}
            metadata["grouping"] = "raw source CSV" if mapping else "source path with Roboflow .rf. siblings grouped"
            metadata["coordinates"] = metadata["extraction"]["normalization"]
            save_json(output / "preprocessing.json", metadata)
            with Stage("Copying original-length landmark files", len(provenance)) as copying:
                def copy_landmark(source_path, target_path):
                    result = shutil.copy2(source_path, target_path)
                    copying.advance()
                    return result
                shutil.copytree(stage, output / "extracted", copy_function=copy_landmark)
            save_json(output / "extracted_sources.json", provenance)
            report["status"] = "validating"
            save_json(output / "extraction_report.json", report)
            print("Final validation with original source IDs...", flush=True)
            inspect_dataset(output)
            save_augmentation_summary(output, manifest)
            report["status"] = "complete"
            save_json(output / "extraction_report.json", report)
    except Exception as error:
        report["status"], report["error"] = "failed", str(error)
        output.parent.mkdir(parents=True, exist_ok=True)
        report_path = output.parent / f"{output.name}.extraction-failed-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
        save_json(report_path, report)
        raise RuntimeError(f"{error}\nExtraction report: {report_path}") from error
    finally:
        with Stage("Closing landmark extractor"):
            extractor.close()
    print(f"Preparation complete. Dataset: {output} | Report: {output / 'extraction_report.json'}", flush=True)
>>>>>>> 46d0e1656159055e1137771beb6aeb8d123f107d:src/fsl_trainer/preparation/extraction.py
