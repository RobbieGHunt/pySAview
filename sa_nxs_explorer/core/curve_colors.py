"""Matplotlib palette sampling for SuperADAM d/u scan families."""

from matplotlib import colormaps
from matplotlib.colors import to_hex

DATASET_SYMBOLS = ("o", "s", "t", "d", "+", "x", "star", "p", "h", "t1", "t2", "t3")


def palette_range(dark=False):
    """Default to the full colormap in either theme."""
    return (0.0, 1.0)


def spin_palette(index=0, count=1, *, down="Blues", up="Reds", dark=False, shade_range=None):
    """Sample evenly within the user-selected shade range.

    Visibility does not change a scan's index. Both spin channels use the same
    position in their independently selected colormaps.
    """
    count = max(1, count)
    fraction = index / max(1, count - 1) if count > 1 else 0.5
    low, high = shade_range if shade_range is not None else palette_range(dark)
    position = low + (high - low) * fraction
    return {"down": to_hex(colormaps[down](position)),
            "up": to_hex(colormaps[up](position)),
            "all": to_hex(colormaps["viridis"](0.25 + 0.6 * fraction))}
