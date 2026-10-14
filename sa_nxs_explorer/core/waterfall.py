"""Display-only waterfall ordering and affine transforms."""
from pathlib import Path
import re
import numpy as np


def waterfall_order(paths):
    """Numbered scans first, ascending; other files retain load order."""
    def key(item):
        index, path = item
        match = re.fullmatch(r"([0-9]+)\.nxs", Path(path).name, re.IGNORECASE)
        return (0, int(match[1]), index) if match else (1, index, index)
    return [path for _, path in sorted(enumerate(paths), key=key)]


def display_values(values, rank, spacing, mode):
    """Transform central values or interval endpoints without mutating data."""
    if values is None:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        if mode == "log":
            return np.asarray(values) * np.power(10.0, rank * spacing)
        return np.asarray(values) + rank * spacing
