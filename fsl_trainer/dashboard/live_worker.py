"""Camera frame -> landmarks -> inference; JSON lines on stdin/stdout, nothing saved."""
import argparse
import base64
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
from .timing import capture_timing


class LivePredictor:
    def __init__(self, model, classes, extractor, mode):
        self.model, self.classes, self.extractor, self.mode = model, classes, extractor, mode
        self.detector = extractor.detector(video=True) if mode in ("motion", "gesture") else None
        self.capture = GestureCapture() if mode in ('gesture', 'motion') else None
        self.has_result = False
        self.previous = -1

    def frame(self, rgb, timestamp, timing=None):
        settings = capture_timing(timing)
        if self.capture:
            self.capture.configure(settings)
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
                if self.mode == 'motion' and captured['status'] == 'waiting' and self.has_result:
                    captured.update(keep_result=True,
                                    message='Result held. Perform another gesture when ready.')
                elif captured['status'] == 'capturing':
                    self.has_result = False
                return captured
            frames = captured['frames']
            times = np.array([item[0] for item in frames])
            positions = np.interp(np.linspace(times[0], times[-1], 32), times, np.arange(len(times)))
            sequence = resample(np.stack([item[1] for item in frames]), positions)
        elif not features[126:].any():
            return {"status": "no_hand", "probabilities": {}, "message": "No hand detected."}
        else:
            sequence = resample(features[None])
        probabilities = np.asarray(self.model(sequence[None], training=False))[0]
        if (probabilities.shape != (len(self.classes),) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()):
            raise ValueError("The selected model returned invalid class probabilities.")
        response = {"status": "prediction", "probabilities": dict(zip(self.classes, map(float, probabilities))),
                    "frames": len(captured['frames']) if captured else 1}
        if captured:
            response.update(complete=True, duration_ms=captured['duration_ms'],
                            message='Gesture captured. Result held until you select Next gesture.')
            if self.mode == 'motion':
                # Keep the completed result while buffering the next movement. Missing
                # hands and a held ending must never replace it with a pose-only tail.
                self.capture.reset()
                self.has_result = True
                response['message'] = 'Gesture captured. Result held; perform another gesture when ready.'
        return response

    def reset(self):
        if not self.capture:
            raise ValueError('Next gesture is available only in movement capture modes.')
        self.capture.reset()
        self.has_result = False
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
                              else predictor.frame(decode_frame(request["image"]), request["timestamp"], request.get("timing")))
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
