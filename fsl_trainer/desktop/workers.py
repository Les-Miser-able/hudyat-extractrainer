"""Background filesystem, media, and camera work; Qt signals return owned images."""
import base64
from pathlib import Path
import threading
import time

from PySide6.QtCore import QObject, QRunnable, QThread, Signal
from PySide6.QtGui import QImage


class TaskSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Task(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function, self.signals = function, TaskSignals()

    def run(self):
        try:
            self.signals.done.emit(self.function())
        except Exception as error:
            self.signals.failed.emit(str(error))


def image_from_rgb(rgb):
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()


class MediaThread(QThread):
    frame = Signal(QImage, object)
    failed = Signal(str)

    def __init__(self, path, overlay=False, parent=None):
        super().__init__(parent)
        self.path, self.overlay = Path(path), overlay
        self.stopped = threading.Event()
        self.condition = threading.Condition()
        self.pending, self.playing = 0, False

    def seek(self, index):
        with self.condition:
            self.pending = index
            self.condition.notify()

    def play(self, enabled):
        with self.condition:
            self.playing = enabled
            self.condition.notify()

    def stop(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify()

    def run(self):
        import numpy as np
        capture = extractor = None
        try:
            import cv2
            from ..data.source_selection import IMAGE_EXTENSIONS
            from ..data.extraction import DEFAULT_MODEL, Extractor, pack_result
            kind, fps, total, index = "landmarks", 30.0, 1, 0
            sequence = still = None
            if self.path.suffix.lower() == ".npy":
                sequence = np.load(self.path, allow_pickle=False)
                if sequence.ndim == 1:
                    sequence = sequence[None]
                if sequence.ndim != 2 or sequence.shape[1] != 128 or not len(sequence) or not np.isfinite(sequence).all():
                    raise ValueError("Landmark preview requires a finite (T, 128) array.")
                total = len(sequence)
                still = np.full((480, 640, 3), (17, 28, 35), dtype=np.uint8)
            elif self.path.suffix.lower() in IMAGE_EXTENSIONS:
                from PIL import Image, ImageOps
                kind = "image"
                with Image.open(self.path) as image:
                    still = np.array(ImageOps.exif_transpose(image).convert("RGB"))
            else:
                kind = "video"
                capture = cv2.VideoCapture(str(self.path))
                if not capture.isOpened():
                    raise ValueError("Video could not be opened.")
                fps, total = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if not np.isfinite(fps) or fps <= 0 or total < 1:
                    raise ValueError("Video has invalid frame count or FPS.")
            if self.overlay and sequence is None:
                if not DEFAULT_MODEL.is_file():
                    raise ValueError("The default hand detector is missing. Prepare a dataset to download it before using the preview overlay.")
                extractor = Extractor()
            while not self.stopped.is_set():
                with self.condition:
                    if self.pending is None and not self.playing:
                        self.condition.wait(.1)
                        continue
                    seek, self.pending = self.pending, None
                started = time.monotonic()
                if seek is not None:
                    index = min(max(0, seek), total - 1)
                    if capture:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                if capture:
                    ok, bgr = capture.read()
                    if not ok:
                        self.playing = False
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                else:
                    rgb = still
                features = sequence[index].tolist() if sequence is not None else None
                if extractor:
                    features = pack_result(extractor.image_detector.detect(extractor.image(rgb))).tolist()
                self.frame.emit(image_from_rgb(rgb), {"index": index, "total": total, "fps": fps,
                                                       "kind": kind, "landmarks": features})
                if index >= total - 1:
                    self.playing = False
                else:
                    index += 1
                self.stopped.wait(max(0, 1 / fps - (time.monotonic() - started)))
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            if capture:
                capture.release()
            if extractor:
                extractor.close()


class CameraThread(QThread):
    frame = Signal(QImage, object)
    prediction = Signal(object)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, live, run, mode, device, timing, hand_model=None, parent=None):
        super().__init__(parent)
        self.live, self.run_path, self.mode = live, run, mode
        self.device, self.timing, self.hand_model = device, dict(timing), hand_model
        # Frame cadence belongs to this capture loop; inference accepts gesture delays only.
        self.frame_interval_ms = self.timing.pop("frame_interval_ms", 50)
        self.stopped, self.reset_requested = threading.Event(), threading.Event()

    def stop(self):
        self.stopped.set()

    def run(self):
        capture = None
        try:
            import cv2
            capture = cv2.VideoCapture(self.device)
            if not capture.isOpened():
                raise ValueError("Camera could not be opened. Check the camera index and Windows camera permissions.")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if self.stopped.is_set():
                return
            self.status.emit("Loading model and hand detector…")
            session = self.live.start(self.run_path, self.hand_model, self.mode)["session"]
            while not self.stopped.is_set():
                started = time.monotonic()
                if self.reset_requested.is_set():
                    self.live.reset(session)
                    self.reset_requested.clear()
                ok, bgr = capture.read()
                if not ok:
                    raise ValueError("The camera stopped delivering frames.")
                height, width = bgr.shape[:2]
                scale = min(1, 640 / max(height, width))
                if scale < 1:
                    bgr = cv2.resize(bgr, (round(width * scale), round(height * scale)))
                self.frame.emit(image_from_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), {})
                ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise ValueError("The camera frame could not be encoded.")
                result = self.live.frame(session, "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
                                         time.monotonic() * 1000, self.timing)
                self.prediction.emit(result)
                self.stopped.wait(max(0, self.frame_interval_ms / 1000 - (time.monotonic() - started)))
        except Exception as error:
            if not self.stopped.is_set():
                self.failed.emit(str(error))
        finally:
            if capture:
                capture.release()
            self.live.stop()
