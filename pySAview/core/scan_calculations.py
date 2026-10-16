"""Calculations for reflectometry scans: Q conversion, normalization, errors, spin splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
from pySAview.core.roi import pixel_bounds
from pySAview.core.normalization import normalize_signal
from pySAview.core.curve_colors import spin_palette, DATASET_SYMBOLS


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


def compute_q(theta_deg: np.ndarray, wavelength_angstrom: float = 5.21) -> np.ndarray:
    """Compute momentum transfer Q (1/Angstrom) from incident angle theta (deg).

    Q = (4 * pi / wavelength) * sin(theta_rad)
    """
    if not np.isfinite(wavelength_angstrom) or wavelength_angstrom <= 0:
        raise ValueError("Wavelength must be strictly positive.")
    theta_rad = np.radians(np.asarray(theta_deg, dtype=float))
    return (4.0 * np.pi / wavelength_angstrom) * np.sin(theta_rad)


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
      - 'split': Two curves (d / Flipper OFF=0 and u / Flipper ON=1)
      - 'flipper0': Only Flipper=0
      - 'flipper1': Only Flipper=1
    """
    indices = np.arange(len(x_data))
    f = np.asarray(flipper_data)

    pal = spin_palette(palette_index, dataset_count, down=down_colormap, up=up_colormap, dark=dark, shade_range=shade_range)
    symbol = DATASET_SYMBOLS[palette_index % len(DATASET_SYMBOLS)]
    color_all = pal["all"]
    color_up = pal["up"]
    color_down = pal["down"]

    prefix = f"[{dataset_name}] " if dataset_name else ""

    if spin_mode == "all":
        return [
            PlottedCurve(
                label=f"{prefix}All states",
                x=x_data,
                y=y_data,
                y_err=y_err,
                color=color_all,
                indices=indices,
                dataset_name=dataset_name,
                symbol=symbol,
            )
        ]

    # Split into 0 and 1
    mask0 = (f == 0)
    mask1 = (f == 1)

    curves: list[PlottedCurve] = []

    if spin_mode in ("split", "flipper0") and np.any(mask0):
        # Sort by x for clean line plotting
        idx0 = indices[mask0]
        sort_order0 = np.argsort(x_data[mask0])
        sorted_idx0 = idx0[sort_order0]
        curves.append(
            PlottedCurve(
                label=f"{prefix}d (OFF)",
                spin="d",
                x=x_data[sorted_idx0],
                y=y_data[sorted_idx0],
                y_err=y_err[sorted_idx0] if y_err is not None else None,
                color=color_down,
                indices=sorted_idx0,
                dataset_name=dataset_name,
                symbol=symbol,
            )
        )

    if spin_mode in ("split", "flipper1") and np.any(mask1):
        idx1 = indices[mask1]
        sort_order1 = np.argsort(x_data[mask1])
        sorted_idx1 = idx1[sort_order1]
        curves.append(
            PlottedCurve(
                label=f"{prefix}u (ON)",
                spin="u",
                x=x_data[sorted_idx1],
                y=y_data[sorted_idx1],
                y_err=y_err[sorted_idx1] if y_err is not None else None,
                color=color_up,
                indices=sorted_idx1,
                dataset_name=dataset_name,
                symbol=symbol,
            )
        )

    if not curves:
        if spin_mode in ("flipper0", "flipper1"):
            return []
        # Fallback if no flipper states found
        return [
            PlottedCurve(
                label=f"{prefix}Scan",
                x=x_data,
                y=y_data,
                y_err=y_err,
                color=color_all,
                indices=indices,
                dataset_name=dataset_name,
                symbol=symbol,
            )
        ]

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


def integrate_roi_2d(
    frame: np.ndarray,
    roi: tuple[float, float, float, float],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Integrate a 2D detector frame within ROI (x0, x1, y0, y1).

    Returns:
      (total_counts, x_profile, y_profile)
    """
    ny, nx = frame.shape
    ix0, ix1, iy0, iy1 = pixel_bounds(roi, nx, ny)

    sub = frame[iy0:iy1, ix0:ix1]
    total = float(np.sum(sub))

    # Profiles summed along axes
    x_profile = np.sum(sub, axis=0)  # profile along horizontal within ROI
    y_profile = np.sum(sub, axis=1)  # profile along vertical within ROI

    return total, x_profile, y_profile


def find_reflection_peak_center(
    profile_x: np.ndarray,
    prev_center: float,
    search_half_width: int = 16,
    min_peak_signal: float = 12.0,
    max_step_change: float = 4.0,
) -> float:
    """Find the reflection beam peak center for a 1D horizontal profile.

    If intensity is extremely low or no distinct peak is resolved within the local
    search window, returns prev_center unchanged.
    """
    nx = len(profile_x)
    if nx == 0:
        return prev_center

    x_min = max(0, int(round(prev_center - search_half_width)))
    x_max = min(nx, int(round(prev_center + search_half_width + 1)))

    if x_max <= x_min:
        return prev_center

    sub = profile_x[x_min:x_max]
    if len(sub) == 0:
        return prev_center

    base = float(np.median(sub))
    peak_val = float(np.max(sub))
    net_signal = peak_val - base

    # Low intensity or flat profile condition:
    # Keep previous ROI position if net signal is negligible or indistinguishable from Poisson noise
    if net_signal < min_peak_signal or peak_val <= 1.0 or net_signal <= 2.0 * np.sqrt(max(0.0, base) + 1.0):
        return prev_center

    loc_peak = int(np.argmax(sub))
    peak_x = x_min + loc_peak

    # Sub-pixel centroid refinement around peak +- 3 pixels
    sub_min = max(0, peak_x - 3)
    sub_max = min(nx, peak_x + 4)
    sub_prof = np.maximum(profile_x[sub_min:sub_max] - base, 0.0)
    if np.sum(sub_prof) > 0:
        xs = np.arange(sub_min, sub_max, dtype=float)
        new_center = float(np.sum(xs * sub_prof) / np.sum(sub_prof))
    else:
        new_center = float(peak_x)

    # Clamping maximum frame-to-frame change to preserve smooth tracking
    if max_step_change > 0 and abs(new_center - prev_center) > max_step_change:
        new_center = prev_center + np.sign(new_center - prev_center) * max_step_change

    return float(new_center)


def auto_fit_from_profiles(
    profiles_x: np.ndarray,
    initial_roi: tuple[float, float, float, float],
    skip_first_n: int = 6,
    search_half_width: int = 16,
    min_peak_signal: float = 12.0,
    max_step_change: float = 4.0,
) -> list[tuple[float, float, float, float]]:
    """Compute per-frame ROIs from an array of horizontal profiles shape (N, nx).

    Excludes the first skip_first_n frames (preserves initial_roi center).
    For subsequent frames, centers ROI on the reflection peak, or retains previous
    position if intensity is low.
    """
    n_frames = len(profiles_x)
    x0, x1, y0, y1 = initial_roi
    width = abs(x1 - x0)
    default_center = (x0 + x1) / 2.0

    rois: list[tuple[float, float, float, float]] = []
    prev_center = default_center
    skip_n = min(max(0, skip_first_n), n_frames)

    for i in range(n_frames):
        if i < skip_n:
            center = default_center
        else:
            center = find_reflection_peak_center(
                profiles_x[i],
                prev_center=prev_center,
                search_half_width=search_half_width,
                min_peak_signal=min_peak_signal,
                max_step_change=max_step_change,
            )
        rois.append((center - width / 2.0, center + width / 2.0, y0, y1))
        prev_center = center

    return rois


def auto_fit_scan_rois(
    frame_provider: Any,
    total_frames: int,
    initial_roi: tuple[float, float, float, float],
    skip_first_n: int = 6,
    search_half_width: int = 16,
    min_peak_signal: float = 12.0,
    max_step_change: float = 4.0,
) -> tuple[list[tuple[float, float, float, float]], np.ndarray]:
    """Automatically track and fit the reflection beam ROI across all scan frames.

    Returns:
      (rois, counts) where rois is a list of (x0, x1, y0, y1) per frame,
      and counts is a 1D ndarray of integrated counts per frame.
    """
    from typing import Any  # local fallback if needed

    x0, x1, y0, y1 = initial_roi
    width = abs(x1 - x0)
    default_center = (x0 + x1) / 2.0
    iy0 = int(round(min(y0, y1)))
    iy1 = int(round(max(y0, y1)))

    if total_frames <= 0:
        return [], np.zeros(0, dtype=float)

    # Detect 3D dataset access (NexusReader or raw 3D array)
    ds = None
    if hasattr(frame_provider, "file") and "entry0/data_scan/detector_data/data" in getattr(frame_provider, "file", {}):
        ds = frame_provider.file["entry0/data_scan/detector_data/data"]
    elif hasattr(frame_provider, "ndim") and frame_provider.ndim == 3:
        ds = frame_provider

    rois: list[tuple[float, float, float, float]] = []
    counts = np.zeros(total_frames, dtype=float)
    skip_n = min(max(0, skip_first_n), total_frames)

    if ds is not None:
        nx = ds.shape[1]
        ny = ds.shape[2]
        c_iy0 = max(0, min(ny, iy0))
        c_iy1 = max(0, min(ny, iy1))

        # Fast path: load bounding volume across X around default center
        x_min_bound = max(0, int(round(default_center - 70)))
        x_max_bound = min(nx, int(round(default_center + 70)))

        use_vol = False
        sub_profs = None
        vol = None
        try:
            vol = ds[:, x_min_bound:x_max_bound, c_iy0:c_iy1]
            sub_profs = np.sum(vol, axis=2)  # shape (total_frames, x_sub_width)
            use_vol = True
        except Exception:
            use_vol = False

        prev_center = default_center
        for i in range(total_frames):
            if i < skip_n:
                center = default_center
            elif use_vol and sub_profs is not None:
                p_rel_center = prev_center - x_min_bound
                center_rel = find_reflection_peak_center(
                    sub_profs[i],
                    prev_center=p_rel_center,
                    search_half_width=search_half_width,
                    min_peak_signal=min_peak_signal,
                    max_step_change=max_step_change,
                )
                center = center_rel + x_min_bound
            else:
                raw_prof = np.sum(ds[i, :, c_iy0:c_iy1], axis=1)
                center = find_reflection_peak_center(
                    raw_prof,
                    prev_center=prev_center,
                    search_half_width=search_half_width,
                    min_peak_signal=min_peak_signal,
                    max_step_change=max_step_change,
                )

            rois.append((center - width / 2.0, center + width / 2.0, y0, y1))
            prev_center = center

            ix0, ix1, c_iy0, c_iy1 = pixel_bounds(rois[-1], nx, ny)
            if use_vol and vol is not None and ix0 >= x_min_bound and ix1 <= x_max_bound:
                counts[i] = float(np.sum(vol[i, ix0 - x_min_bound:ix1 - x_min_bound, :]))
            else:
                counts[i] = float(np.sum(ds[i, ix0:ix1, c_iy0:c_iy1]))

        return rois, counts

    # Fallback for callable or frame provider
    prev_center = default_center
    for i in range(total_frames):
        if callable(frame_provider):
            frame = frame_provider(i)
        elif hasattr(frame_provider, "read_detector_frame"):
            raw_f = frame_provider.read_detector_frame(i)
            frame = np.fliplr(np.rot90(raw_f, -1)) if raw_f is not None else None
        else:
            frame = None

        if frame is None:
            rois.append((prev_center - width / 2.0, prev_center + width / 2.0, y0, y1))
            continue

        ny, nx = frame.shape
        c_iy0 = max(0, min(ny, iy0))
        c_iy1 = max(0, min(ny, iy1))
        prof = np.sum(frame[c_iy0:c_iy1, :], axis=0)

        if i < skip_n:
            center = default_center
        else:
            center = find_reflection_peak_center(
                prof,
                prev_center=prev_center,
                search_half_width=search_half_width,
                min_peak_signal=min_peak_signal,
                max_step_change=max_step_change,
            )

        rois.append((center - width / 2.0, center + width / 2.0, y0, y1))
        prev_center = center

        ix0, ix1, c_iy0, c_iy1 = pixel_bounds(rois[-1], nx, ny)
        counts[i] = float(np.sum(frame[c_iy0:c_iy1, ix0:ix1]))

    return rois, counts
