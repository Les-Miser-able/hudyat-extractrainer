"""Camera frame -> landmarks -> inference; JSON lines on stdin/stdout, nothing saved."""
import argparse
import base64
from collections import deque
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

from ..data.extraction import Extractor, DEFAULT_MODEL, pack_result
from ..data.sequences import resample
from ..training.trainer import tensorflow
from .gesture_capture import GestureCapture


class LivePredictor:
    def __init__(self, model, classes, extractor, mode):
        self.model, self.classes, self.extractor, self.mode = model, classes, extractor, mode
        self.detector = extractor.detector(video=True) if mode in ("motion", "gesture") else None
        self.capture = GestureCapture() if mode == 'gesture' else None
        self.frames = deque(maxlen=256)
        self.previous = -1

    def frame(self, rgb, timestamp):
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < 0 or timestamp > 2**53):
            raise ValueError("Invalid camera timestamp.")
        timestamp = round(timestamp)
        if timestamp <= self.previous:
            raise ValueError("Camera timestamps must increase.")
        if self.capture and self.capture.state in ('held', 'incomplete'):
            self.previous = timestamp
            return self.capture.push(None, timestamp)
        # Long pauses start a new gesture window instead of interpolating a stale one.
        if timestamp - self.previous > 1000:
            self.frames.clear()
            if self.capture:
                self.capture.reset()
        self.previous = timestamp
        image = self.extractor.image(rgb)
        result = (self.detector.detect_for_video(image, timestamp) if self.detector
                  else self.extractor.image_detector.detect(image))
        features = pack_result(result, self.extractor.swap_hands)
        captured = None
        if self.capture:
            captured = self.capture.push(features, timestamp)
            if captured['status'] != 'complete':
                return captured
            frames = captured['frames']
            times = np.array([item[0] for item in frames])
            positions = np.interp(np.linspace(times[0], times[-1], 32), times, np.arange(len(times)))
            sequence = resample(np.stack([item[1] for item in frames]), positions)
        elif not features[126:].any():
            self.frames.clear()
            return {"status": "no_hand", "probabilities": {}, "message": "No hand detected."}
        elif self.mode == "pose":
            sequence = resample(features[None])
        else:
            self.frames.append((timestamp, features))
            # Keep one boundary frame to cover the full two-second interval.
            while len(self.frames) > 2 and self.frames[1][0] <= timestamp - 2000:
                self.frames.popleft()
            duration = timestamp - self.frames[0][0]
            if duration < 2000:
                return {"status": "buffering", "probabilities": {},
                        "progress": duration / 2000, "message": "Collecting two seconds of movement…"}
            times = np.array([item[0] for item in self.frames])
            positions = np.interp(np.linspace(timestamp - 2000, timestamp, 32), times, np.arange(len(times)))
            sequence = resample(np.stack([item[1] for item in self.frames]), positions)
        probabilities = np.asarray(self.model(sequence[None], training=False))[0]
        if (probabilities.shape != (len(self.classes),) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()):
            raise ValueError("The selected model returned invalid class probabilities.")
        response = {"status": "prediction", "probabilities": dict(zip(self.classes, map(float, probabilities))),
                    "frames": len(captured['frames']) if captured else 1 if self.mode == "pose" else len(self.frames)}
        if captured:
            response.update(complete=True, duration_ms=captured['duration_ms'],
                            message='Gesture captured. Result held until you select Next gesture.')
        return response

    def reset(self):
        if not self.capture:
            raise ValueError('Next gesture is available only in Perform once mode.')
        self.capture.reset()
        return {'status': 'waiting', 'message': 'Ready. Perform the gesture once, then hold the ending still.'}

    def close(self):
        if self.detector:
            self.detector.close()
        self.extractor.close()


def decode_frame(encoded):
    prefix = "data:image/jpeg;base64,"
    if not isinstance(encoded, str) or not encoded.startswith(prefix) or len(encoded) > 900_000:
        raise ValueError("Send a JPEG camera frame smaller than 900 KB.")
    from PIL import Image
    raw = base64.b64decode(encoded[len(prefix):], validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "JPEG" or max(image.size) > 1024 or image.width * image.height > 1_048_576:
            raise ValueError("Camera frames must be JPEG images no larger than 1024 pixels per side.")
        return np.array(image.convert("RGB"))


def load_predictor(run, hand_model, mode):
    run = Path(run)
    metadata = json.loads((run / "preprocessing.json").read_text(encoding="utf-8"))
    settings = metadata.get("extraction")
    if not settings:
        raise ValueError("This run has no camera extraction settings. Use normalized .npy sample prediction.")
    extractor = Extractor(model=hand_model or DEFAULT_MODEL, confidence=settings["detection_confidence"],
                          swap_hands=settings["swap_hands"])
    try:
        if hashlib.sha256(extractor.model.read_bytes()).hexdigest() != settings["model_sha256"]:
            raise ValueError("Hand model differs from training. Select the original .task asset.")
        classes = json.loads((run / "classes.json").read_text(encoding="utf-8"))
        model = tensorflow().keras.models.load_model(run / "best.keras", compile=False)
        if tuple(model.input_shape[1:]) != (32, 128) or model.output_shape[-1] != len(classes):
            raise ValueError("Model shape does not match its landmark layout or class labels.")
        return LivePredictor(model, classes, extractor, mode)
    except Exception:
        extractor.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--hand-model")
    parser.add_argument("--mode", choices=["pose", "motion", "gesture"], required=True)
    args = parser.parse_args()
    protocol = sys.stdout

    def reply(value):
        protocol.write(json.dumps(value, allow_nan=False) + "\n")
        protocol.flush()

    predictor = None
    try:
        with redirect_stdout(sys.stderr):
            predictor = load_predictor(args.run, args.hand_model, args.mode)
        reply({"ready": True})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                with redirect_stdout(sys.stderr):
                    result = (predictor.reset() if request.get('action') == 'reset'
                              else predictor.frame(decode_frame(request["image"]), request["timestamp"]))
                reply(result)
            except Exception as error:
                reply({"error": str(error)})
    except Exception as error:
        reply({"error": str(error)})
    finally:
        if predictor:
            predictor.close()


if __name__ == "__main__":
    main()
