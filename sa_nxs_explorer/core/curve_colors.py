"""Matplotlib palette sampling for SuperADAM d/u scan families."""

from matplotlib import colormaps
from matplotlib.colors import to_hex

DATASET_SYMBOLS = ("o", "s", "t", "d", "+", "x", "star", "p", "h", "t1", "t2", "t3")


def palette_range(dark=False):
    """Use broad shades while avoiding the least legible endpoints."""
    return (0.20, 0.85) if dark else (0.30, 0.95)


def spin_palette(index=0, count=1, *, down="Blues", up="Reds", dark=False):
    """Sample one shade per loaded scan, avoiding nearly-white/dark endpoints.

    Visibility does not change a scan's index. Both spin channels use the same
    position in their independently selected colormaps.
    """
    count = max(1, count)
    fraction = index / max(1, count - 1) if count > 1 else 0.5
    low, high = palette_range(dark)
    position = low + (high - low) * fraction
    return {"down": to_hex(colormaps[down](position)),
            "up": to_hex(colormaps[up](position)),
            "all": to_hex(colormaps["viridis"](0.25 + 0.6 * fraction))}
