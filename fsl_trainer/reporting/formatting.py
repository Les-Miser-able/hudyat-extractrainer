"""Presentation formats; saved measurements retain their original numeric values."""
import math


def percentage(value, *, scale=100):
    """Format a known proportion, or an already scaled percentage with scale=1."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        return None
    return f'{value * scale:.2f}%'
