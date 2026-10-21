"""Versioned camera features; legacy image coordinates remain the fallback."""
IMAGE = "image"
HAND_RELATIVE = "hand-relative-sequence-v1"
NORMALIZATIONS = (IMAGE, HAND_RELATIVE)
COORDINATES = ("landmarks 1-20: per-frame wrist-relative pixels / pixel palm length; "
               "landmark 0: wrist displacement from first valid frame / median pixel palm length")


def preparation_normalization(value, category=None, has_media=False):
    if value == "auto":
        return HAND_RELATIVE if category == "alphabet" and has_media else IMAGE
    if value not in NORMALIZATIONS:
        raise ValueError(f"Unsupported normalization: {value}")
    return value


def model_normalization(metadata):
    """Old runs have no top-level normalization field and use image coordinates."""
    return preparation_normalization(metadata.get("normalization", IMAGE))


def add_arguments(parser, default=IMAGE):
    parser.add_argument("--normalization", choices=("auto", *NORMALIZATIONS), default=default,
                        help="Camera feature convention; auto uses hand-relative features only for raw alphabet media")


def normalize_sequence(sequence, width, height):
    """Center and scale once per hand, preserving wrist trajectories for J/Z.

    A still frame matches wrist-relative/palm-scaled pose normalization. A video
    keeps per-frame hand shape, and stores the wrist's trajectory in landmark 0
    using its first valid position and median palm length for the whole clip.
    """
    import numpy as np
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Hand-relative features require positive, finite image width and height")
    source = np.asarray(sequence, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 128 or len(source) < 1:
        raise ValueError("Hand-relative input must have shape (T,128), T >= 1")
    if not np.isfinite(source).all() or not np.isin(source[:, 126:], [0, 1]).all():
        raise ValueError("Hand-relative input needs finite coordinates and binary presence flags")
    out = np.zeros(source.shape, dtype=np.float32)
    for hand in range(2):
        points = source[:, hand * 63:(hand + 1) * 63].reshape(-1, 21, 3) * [width, height, width]
        palm = np.linalg.norm(points[:, 9, :2] - points[:, 0, :2], axis=1)
        valid = (source[:, 126 + hand] == 1) & (palm >= 1e-6)
        if not valid.any():
            continue
        origin = points[np.flatnonzero(valid)[0], 0].copy()
        scale = float(np.median(palm[valid]))
        normalized = (points[valid] - points[valid, :1]) / palm[valid, None, None]
        normalized[:, 0] = (points[valid, 0] - origin) / scale
        if not np.isfinite(normalized).all() or np.max(np.abs(normalized)) > np.finfo(np.float32).max:
            raise ValueError("Hand-relative coordinates exceed float32 range")
        out[valid, hand * 63:(hand + 1) * 63] = normalized.reshape(-1, 63)
        out[valid, 126 + hand] = 1
    return out


def preprocess_sequence(sequence, width, height, normalization=IMAGE):
    normalization = preparation_normalization(normalization)
    return normalize_sequence(sequence, width, height) if normalization == HAND_RELATIVE else sequence


def augment_pose(frame, rng):
    """Training-only in-plane rotation and mirror in hand-relative units."""
    import numpy as np
    out = frame.copy()
    if rng.random() < .5:
        out[:, :63], out[:, 63:126] = frame[:, 63:126], frame[:, :63]
        out[:, 126:] = frame[:, [127, 126]]
        out[:, :126:3] *= -1
    angle = np.radians(rng.uniform(-15, 15))
    c, s = np.cos(angle), np.sin(angle)
    for hand in range(2):
        points = out[:, hand * 63:(hand + 1) * 63].reshape(-1, 21, 3)
        x, y = points[:, :, 0].copy(), points[:, :, 1].copy()
        points[:, :, 0], points[:, :, 1] = c * x - s * y, s * x + c * y
    return out


def mirror_sequence(sequence, normalization=IMAGE):
    """Reflect a whole gesture horizontally and exchange anatomical hand slots.

    Image x coordinates reflect around 0.5; wrist-relative poses and trajectories
    reflect around zero. Timeline, depth, hand presence, and missing frames remain
    consistent throughout the sequence. The input is never modified.
    """
    import numpy as np
    normalization = preparation_normalization(normalization)
    source = np.asarray(sequence)
    if (source.ndim != 2 or source.shape[1] != 128 or len(source) < 1
            or not np.issubdtype(source.dtype, np.number) or np.iscomplexobj(source)):
        raise ValueError('Hand mirroring requires real coordinates with shape (T,128), T >= 1')
    source = source.astype(np.float32)
    if not np.isfinite(source).all() or not np.isin(source[:, 126:], [0, 1]).all():
        raise ValueError('Hand mirroring requires finite coordinates and binary presence flags')
    out = np.empty_like(source)
    out[:, :63], out[:, 63:126] = source[:, 63:126], source[:, :63]
    out[:, 126:] = source[:, [127, 126]]
    for hand in range(2):
        start, stop = hand * 63, (hand + 1) * 63
        present = out[:, 126 + hand] == 1
        if np.any(out[~present, start:stop] != 0):
            raise ValueError('Hand mirroring requires zero coordinates for absent hands')
        x = out[present, start:stop:3]
        out[present, start:stop:3] = 1 - x if normalization == IMAGE else -x
        out[~present, start:stop] = 0
    return out
