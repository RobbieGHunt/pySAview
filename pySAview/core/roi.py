"""Backward-compatible re-export for the shared whole-pixel ROI convention.

Implementation lives in :mod:`pySAview.core.reduction.roi_extraction`.
"""

from pySAview.core.reduction.roi_extraction import pixel_bounds

__all__ = ["pixel_bounds"]
