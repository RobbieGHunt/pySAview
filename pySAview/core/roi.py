"""Shared whole-pixel detector ROI convention in displayed coordinates."""
import math


def pixel_bounds(roi, nx, ny):
    """Return clipped (x0, x1, y0, y1) for half-open whole-pixel slices.

    Displayed frame is raw_frame.T. Bounds round to nearest integer (ties to
    even, matching Python round), then clip to [0, axis length]. Reversed
    endpoints are accepted. Empty/outside regions select no pixels.
    """
    if len(roi) != 4 or not all(math.isfinite(v) for v in roi):
        raise ValueError("ROI requires four finite bounds")
    x0, x1, y0, y1 = roi
    def clip(v, length):
        return max(0, min(length, round(v)))
    return (clip(min(x0, x1), nx), clip(max(x0, x1), nx),
            clip(min(y0, y1), ny), clip(max(y0, y1), ny))
