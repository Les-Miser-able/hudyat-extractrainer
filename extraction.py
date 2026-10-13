"""Compatibility entry point for fsl_trainer.data.extraction."""
import sys
from fsl_trainer.data import extraction as _implementation

# Keep legacy imports and patches attached to the implementation module.
sys.modules[__name__] = _implementation
