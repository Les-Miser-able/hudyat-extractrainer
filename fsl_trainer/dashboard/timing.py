"""Validate live capture timing without importing ML dependencies."""

DEFAULT_SETTLE_MS = 600
DEFAULT_MISSING_MS = 400


def capture_timing(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) - {'settle_ms', 'missing_ms'}:
        raise ValueError('Timing must contain only settle_ms and missing_ms.')
    for name, milliseconds in value.items():
        if (isinstance(milliseconds, bool) or not isinstance(milliseconds, int)
                or not 100 <= milliseconds <= 2000):
            raise ValueError(f'{name} must be a whole number between 100 and 2000 ms.')
    return dict(value)
