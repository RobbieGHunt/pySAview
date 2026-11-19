"""Calculations for reflectometry scans: Q conversion, normalization, errors, spin splitting.

Pure reduction physics (Q kinematics, ROI extraction, peak tracking, count
normalization, and spin-state index splitting) live in
:mod:`pySAview.core.reduction` and are re-exported here for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pySAview.core.curve_colors import DATASET_SYMBOLS, spin_palette
from pySAview.core.reduction.kinematics import compute_q
from pySAview.core.reduction.normalization import normalize_signal
from pySAview.core.reduction.polarization import split_indices_by_flipper
from pySAview.core.reduction.roi_extraction import (
    auto_fit_from_profiles,
    auto_fit_scan_rois,
    find_reflection_peak_center,
    integrate_roi_2d,
    pixel_bounds,
)


@dataclass
class PlottedCurve:
    """A prepared curve ready for pyqtgraph rendering."""

    label: str
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray | None
    color: str
    indices: np.ndarray
    dataset_name: str = ""
    y_lower: np.ndarray | None = None
    y_upper: np.ndarray | None = None
    uncertainty_method: str = ""
    source_path: str = ""
    normalization: str = "raw"
    reference_monitor_rate: float | None = None
    x_units: str = ""
    y_units: str = ""
    wavelength: float | None = None
    symbol: str = "o"
    spin: str | None = None
    resolution: np.ndarray | None = None
    resolution_units: str = ""
    overill_coeff: np.ndarray | None = None
    theta_deg: np.ndarray | None = None
    theta_offset_deg: float = 0.0
    reduction_summary: str = ""
    show_overillumination_curve: bool = False


@dataclass
class CurveStats:
    """Summary statistics for a plotted curve."""

    n_points: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    y_sum: float
    peak_x: float
    peak_y: float
    centroid_x: float


def compute_normalized_y(
    y_data: np.ndarray,
    norm_mode: Literal["raw", "monitor", "time", "monitor_time", "monitor_cps"] = "raw",
    monitor_data: np.ndarray | None = None,
    time_data: np.ndarray | None = None,
    scale_monitor: bool = False,
    reference_monitor_rate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility API returning values and propagated standard uncertainty.

    Monitor normalization now means N/M, without an implicit scan-specific
    scale. Use normalize_signal for exact confidence limits and invalid reasons.
    Standard uncertainty is an estimate, not an exact low-count interval.
    """
    if scale_monitor:
        raise ValueError("Implicit mean-monitor scaling is unsupported; use a fixed reference rate.")
    result = normalize_signal(
        y_data, norm_mode, monitor=monitor_data, time=time_data,
        uncertainty="standard", reference_monitor_rate=reference_monitor_rate,
    )
    return result.values, result.sigma


def split_by_flipper(
    x_data: np.ndarray,
    y_data: np.ndarray,
    y_err: np.ndarray | None,
    flipper_data: np.ndarray,
    spin_mode: str = "split",
    palette_index: int = 0,
    dataset_name: str = "",
    dataset_count: int = 1,
    down_colormap: str = "Blues",
    up_colormap: str = "Reds",
    shade_range: tuple[float, float] | None = None,
    dark: bool = False,
) -> list[PlottedCurve]:
    """Split X and Y curves by spin state (Flipper 0 or 1).

    spin_mode:
      - 'all': Single curve with all points
      - 'split': Two curves (d / Flipper ON=0 and u / Flipper OFF=1)
      - 'flipper0': Only Flipper=0
      - 'flipper1': Only Flipper=1
    """
    pal = spin_palette(
        palette_index,
        dataset_count,
        down=down_colormap,
        up=up_colormap,
        dark=dark,
        shade_range=shade_range,
    )
    symbol = DATASET_SYMBOLS[palette_index % len(DATASET_SYMBOLS)]
    color_map = {"d": pal["down"], "u": pal["up"], None: pal["all"]}
    prefix = f"[{dataset_name}] " if dataset_name else ""

    curves: list[PlottedCurve] = []
    for spin_key, label, idx in split_indices_by_flipper(x_data, flipper_data, spin_mode=spin_mode):
        is_all = spin_key is None
        curves.append(
            PlottedCurve(
                label=f"{prefix}{label}",
                spin=spin_key,
                x=x_data if is_all else x_data[idx],
                y=y_data if is_all else y_data[idx],
                y_err=(y_err if is_all else y_err[idx]) if y_err is not None else None,
                color=color_map[spin_key],
                indices=idx,
                dataset_name=dataset_name,
                symbol=symbol,
            )
        )
    return curves


def calculate_curve_stats(x: np.ndarray, y: np.ndarray) -> CurveStats:
    """Compute summary statistics for curve: peak, min, max, centroid."""
    if len(x) == 0 or len(y) == 0:
        return CurveStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    x_clean = np.asarray(x, dtype=float)
    y_clean = np.asarray(y, dtype=float)

    valid = ~(np.isnan(x_clean) | np.isnan(y_clean) | np.isinf(x_clean) | np.isinf(y_clean))
    if not np.any(valid):
        return CurveStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    x_v = x_clean[valid]
    y_v = y_clean[valid]

    peak_idx = int(np.argmax(y_v))
    peak_x = float(x_v[peak_idx])
    peak_y = float(y_v[peak_idx])

    y_sum = float(np.sum(y_v))
    centroid_x = float(np.sum(x_v * y_v) / y_sum) if y_sum != 0 else peak_x

    return CurveStats(
        n_points=len(x_v),
        x_min=float(np.min(x_v)),
        x_max=float(np.max(x_v)),
        y_min=float(np.min(y_v)),
        y_max=float(np.max(y_v)),
        y_sum=y_sum,
        peak_x=peak_x,
        peak_y=peak_y,
        centroid_x=centroid_x,
    )


__all__ = [
    "CurveStats",
    "PlottedCurve",
    "auto_fit_from_profiles",
    "auto_fit_scan_rois",
    "calculate_curve_stats",
    "compute_normalized_y",
    "compute_q",
    "find_reflection_peak_center",
    "integrate_roi_2d",
    "pixel_bounds",
    "split_by_flipper",
]
