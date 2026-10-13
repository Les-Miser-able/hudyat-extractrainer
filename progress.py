"""Compatibility entry point for fsl_trainer.progress."""
import sys
from fsl_trainer import progress as _implementation

# Keep legacy imports and patches attached to the implementation module.
sys.modules[__name__] = _implementation
