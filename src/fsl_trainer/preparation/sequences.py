"""Validation and preprocessing of already normalized landmarks."""
import numpy as np
from fsl_trainer.common import FRAMES, FEATURES

def load_sequence(path, fixed=False):
    """Input is already normalized. Never infer hand presence from coordinates."""
    x = np.load(path, allow_pickle=False)
    if x.ndim == 1 and x.shape == (FEATURES,):
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != FEATURES or len(x) < 1:
        raise ValueError(f"{path}: expected (T,128), T>=1; got {x.shape}")
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError(f"{path}: expected real numeric coordinates")
    x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: contains NaN or infinity")
    if not np.isin(x[:, 126:], [0, 1]).all():
        raise ValueError(f"{path}: presence columns 126:128 must be 0 or 1")
    if fixed and x.shape != (FRAMES, FEATURES):
        raise ValueError(f"{path}: prepared samples must be (32,128)")
    for hand in range(2):
        absent = x[:, 126 + hand] == 0
        if np.any(x[absent, hand * 63:(hand + 1) * 63] != 0):
            raise ValueError(f"{path}: absent hand {hand} must have zero coordinates")
    if not x[:, 126:].any():
        raise ValueError(f"{path}: no hand present in the entire sample")
    return x



def resample(x):
    """Linearly interpolate coordinates on a uniform timeline; flags use nearest."""
    timeline = np.linspace(0, len(x) - 1, FRAMES)
    out = np.empty((FRAMES, FEATURES), dtype=np.float32)
    for col in range(126):
        out[:, col] = np.interp(timeline, np.arange(len(x)), x[:, col])
    nearest = np.floor(timeline + 0.5).astype(int)
    out[:, 126:] = x[nearest, 126:]
    for hand in range(2):
        out[out[:, 126 + hand] == 0, hand * 63:(hand + 1) * 63] = 0
    return out



def synthetic_sequence(frame, rng, jitter=0.0075):
    """Repeat one pose and add independent per-timestep Gaussian coordinate noise."""
    out = np.repeat(frame[:1], FRAMES, axis=0).astype(np.float32)
    for hand in range(2):
        if frame[0, 126 + hand] == 1:
            out[:, hand*63:(hand+1)*63] += rng.normal(
                0, jitter, (FRAMES, 63)).astype(np.float32)
    return out

