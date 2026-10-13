"""Compatibility entry point for fsl_trainer.reporting.confusion_matrix."""
import sys
from fsl_trainer.reporting import confusion_matrix as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    # Keep legacy imports and patches attached to the implementation module.
    sys.modules[__name__] = _implementation
