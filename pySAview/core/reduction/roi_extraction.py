"""Detector ROI pixel bounds, 2D/Y integration, and reflection peak tracking.

All ROI bounds follow pySAview's shared whole-pixel convention in displayed
coordinates (``displayed = raw_frame.T``, where displayed X is raw axis 1 and
displayed Y is raw axis 2): bounds round to nearest integer (ties to even),
clip to ``[0, axis_length]``, accept reversed endpoints, and slice half-open.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


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

    return (
        clip(min(x0, x1), nx),
        clip(max(x0, x1), nx),
        clip(min(y0, y1), ny),
        clip(max(y0, y1), ny),
    )


def y_pixel_bounds(y_range: tuple[float, float] | None, nx: int, ny: int) -> tuple[int, int]:
    """Clip a displayed Y window with the shared ROI convention; ``None`` means full height."""
    if y_range is None:
        return 0, ny
    _, _, iy0, iy1 = pixel_bounds((0.0, float(nx), float(y_range[0]), float(y_range[1])), nx, ny)
    return iy0, iy1


def integrate_frames_along_y(raw_frames: np.ndarray, y_range: tuple[float, float] | None = None) -> np.ndarray:
    """Integrate raw frames (n, raw_axis1, raw_axis2) over displayed Y.

    Displayed X is raw axis 1 and displayed Y is raw axis 2, so the result
    has shape (n, nx) with ``result[i, x] = sum(displayed_i[iy0:iy1, x])``.
    """
    frames = np.asarray(raw_frames)
    if frames.ndim != 3:
        raise ValueError("Detector data must be a 3D (frame, x, y) volume")
    nx, ny = frames.shape[1], frames.shape[2]
    iy0, iy1 = y_pixel_bounds(y_range, nx, ny)
    if iy1 <= iy0:
        return np.zeros((frames.shape[0], nx), dtype=float)
    return np.sum(frames[:, :, iy0:iy1], axis=2, dtype=np.float64)


def frame_y_bounds(y_range, n_frames: int, nx: int, ny: int) -> np.ndarray:
    """Clipped (n_frames, 2) Y windows from None, one (y0, y1) pair, or one pair per frame."""
    if y_range is None or (len(y_range) == 2 and np.isscalar(y_range[0])):
        return np.tile(np.array(y_pixel_bounds(y_range, nx, ny), dtype=int), (n_frames, 1))
    if len(y_range) != n_frames:
        raise ValueError("Per-frame Y windows must supply exactly one window per detector frame")
    return np.array([y_pixel_bounds(tuple(r), nx, ny) for r in y_range], dtype=int).reshape(n_frames, 2)


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
    x0, x1, y0, y1 = initial_roi
    width = abs(x1 - x0)
    default_center = (x0 + x1) / 2.0
    iy0 = int(round(min(y0, y1)))
    iy1 = int(round(max(y0, y1)))

    if total_frames <= 0:
        return [], np.zeros(0, dtype=float)

    # Detect 3D dataset access (duck-typed reader with .file or raw 3D array)
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


__all__ = [
    "auto_fit_from_profiles",
    "auto_fit_scan_rois",
    "find_reflection_peak_center",
    "frame_y_bounds",
    "integrate_frames_along_y",
    "integrate_roi_2d",
    "pixel_bounds",
    "y_pixel_bounds",
]
