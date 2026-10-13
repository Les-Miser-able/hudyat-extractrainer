"""Preparation control validation shared by the CLI and dependency-free dashboard."""
import math


def extra_samples(value):
    try:
        result = int(value)
        valid = not isinstance(value, bool) and result == float(value) and 0 <= result <= 1000
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("Video extra samples must be an integer from 0 to 1000")
    return result


def time_warp(value):
    try:
        result = float(value)
        valid = not isinstance(value, bool) and math.isfinite(result) and 0 < result <= .10
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("Video timing strength must be greater than 0 and at most 0.10")
    return result


def add_arguments(parser):
    parser.add_argument("--video-extra-samples", type=extra_samples, default=0,
                        help="Up to N extra timing samples per training video/multi-frame input (0 disables)")
    parser.add_argument("--video-time-warp", type=time_warp, default=.05,
                        help="Maximum normalized timing shift, greater than 0 through 0.10 (default: 0.05)")
