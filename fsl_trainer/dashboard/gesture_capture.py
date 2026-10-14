"""Capture one movement and keep its boundaries until explicitly reset."""
from collections import deque
import numpy as np
from .timing import DEFAULT_SETTLE_MS, DEFAULT_MISSING_MS, capture_timing


class GestureCapture:
    PRE_ROLL_MS = 600
    SETTLE_MS = DEFAULT_SETTLE_MS
    MISSING_MS = DEFAULT_MISSING_MS
    MAX_DURATION_MS = 6000
    SPEED_THRESHOLD = .04  # Mean landmark XY movement in image units per second.

    def __init__(self):
        self.settle_ms = self.SETTLE_MS
        self.missing_ms = self.MISSING_MS
        self.reset()

    def configure(self, timing):
        settings = capture_timing(timing)
        self.settle_ms = settings.get('settle_ms', self.settle_ms)
        self.missing_ms = settings.get('missing_ms', self.missing_ms)

    def reset(self):
        self.state = 'waiting'
        self.frames = deque(maxlen=512)
        self.previous = None
        self.onset_hits = 0
        self.onset_distance = 0.
        self.distance = 0.
        self.last_motion = None
        self.missing_since = None
        self.started_at = None

    def push(self, features, timestamp):
        if self.state in ('held', 'incomplete'):
            return {'status': self.state, 'keep_result': self.state == 'held',
                    'message': 'Result held. Select Next gesture when ready.' if self.state == 'held'
                    else 'Capture took too long. Select Next gesture and try again.'}
        present = bool(features[126:].any())
        distance = 0.
        if present and self.previous is not None:
            previous_time, previous = self.previous
            common = (features[126:] > 0) & (previous[126:] > 0)
            distances = []
            for hand in np.flatnonzero(common):
                current_xyz = features[hand * 63:(hand + 1) * 63].reshape(21, 3)
                previous_xyz = previous[hand * 63:(hand + 1) * 63].reshape(21, 3)
                distances.append(float(np.linalg.norm(current_xyz[:, :2] - previous_xyz[:, :2], axis=1).mean()))
            distance = max(distances, default=0.)
            elapsed = max(1, timestamp - previous_time) / 1000
            moving = distance / elapsed >= self.SPEED_THRESHOLD
        else:
            moving = False
        self.previous = (timestamp, features.copy()) if present else None

        if self.state == 'waiting':
            self.frames.append((timestamp, features.copy()))
            while len(self.frames) > 1 and self.frames[1][0] < timestamp - self.PRE_ROLL_MS:
                self.frames.popleft()
            if not present:
                self.onset_hits = 0
                self.onset_distance = 0.
                return {'status': 'waiting', 'message': 'Show your hands, then perform the gesture once.'}
            self.onset_hits = self.onset_hits + 1 if moving else 0
            self.onset_distance = self.onset_distance + distance if moving else 0.
            if self.onset_hits < 2:
                return {'status': 'waiting', 'message': 'Ready. Perform the gesture once, then hold the ending still.'}
            self.state = 'capturing'
            self.distance = self.onset_distance
            self.last_motion = timestamp
            self.started_at = self.frames[0][0]
        else:
            if len(self.frames) == self.frames.maxlen:
                self.state = 'incomplete'
                self.frames.clear()
                return {'status': 'incomplete', 'message': 'Capture exceeded the frame limit. Select Next gesture and try again.'}
            self.frames.append((timestamp, features.copy()))
            self.distance += distance
            if moving:
                self.last_motion = timestamp

        if present:
            if self.missing_since is not None:
                self.last_motion = timestamp
            self.missing_since = None
        elif self.missing_since is None:
            self.missing_since = timestamp

        duration = timestamp - self.started_at
        if duration > self.MAX_DURATION_MS:
            self.state = 'incomplete'
            self.frames.clear()
            return {'status': 'incomplete', 'message': 'Capture took too long. Select Next gesture and try again.'}
        settled = present and timestamp - self.last_motion >= self.settle_ms
        lowered = self.missing_since is not None and timestamp - self.missing_since >= self.missing_ms
        if settled or lowered:
            if duration < 400 or self.distance < .025:
                self.reset()
                return {'status': 'waiting', 'message': 'Movement was too small. Perform the complete gesture again.'}
            frames = list(self.frames)
            self.frames.clear()
            self.state = 'held'
            return {'status': 'complete', 'frames': frames, 'duration_ms': duration}
        return {'status': 'capturing', 'message': 'Capturing the full gesture… Hold the ending still to finish.'}
