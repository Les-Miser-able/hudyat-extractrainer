"""Compatibility entry point for fsl_trainer.training.cache."""
import sys
from fsl_trainer.training import cache as _implementation

# Keep legacy imports and patches attached to the implementation module.
sys.modules[__name__] = _implementation
