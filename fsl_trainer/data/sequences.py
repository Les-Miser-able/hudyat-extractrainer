"""Shared fixed-length landmark sampling for preparation and inference."""
import numpy as np


def resample(x, timeline=None):
    """Linearly interpolate coordinates on a uniform timeline; flags use nearest."""
    if timeline is None:
        timeline = np.linspace(0, len(x) - 1, 32)
    out = np.empty((32, 128), dtype=np.float32)
    for col in range(126):
        out[:, col] = np.interp(timeline, np.arange(len(x)), x[:, col])
    nearest = np.floor(timeline + 0.5).astype(int)
    out[:, 126:] = x[nearest, 126:]
    for hand in range(2):
        out[out[:, 126 + hand] == 0, hand * 63:(hand + 1) * 63] = 0
    return out
