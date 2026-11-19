"""Sample curvature correction following pySAred 1.5.3.

Implements the ROI shear-and-regrid correction from Alexey Klechikov's
``pySAred_V1.5.3.py`` (``f_SFM_reflectivityPreview_load``):

A bent sample tilts Kiessig fringes / specular intensity linearly across the
detector X pixels of the specular ROI as a function of scan point.  For an
integrated 2D ``(point, x)`` ROI slice of shape ``(N, W)`` and curvature shear
angle ``curvature_rad`` (radians):

* Column ``x`` (``0 <= x < W``) is shifted along the scan-point axis by
  ``displacement = x * tan(curvature_rad)``.
* The origin is anchored at the midpoint of the ROI:
  ``zero_level = round((W / 2) * tan(curvature_rad) - min_y)``, where
  ``min_y = min(0, (W - 1) * tan(curvature_rad))``.
* Intensities are linearly interpolated onto the integer target grid
  ``y_target = min_y + arange(zero_level, zero_level + N)`` with ``0`` outside
  the shifted support.

Equivalence & safety notes:
* Because the sheared rectangle is a convex parallelogram on integer ``x``
  columns, per-column 1D linear interpolation onto ``y_target`` matches
  ``scipy.interpolate.griddata(..., method="linear", fill_value=0.0)`` to
  floating-point precision (<= 1.4e-14) everywhere ``griddata`` succeeds, while
  avoiding the off-by-one ``np.mgrid`` shape truncation bug in pySAred 1.5.3
  and avoiding SciPy startup cost.
* Every function returns a **new** array and never mutates input counts, so
  setting ``curvature_rad = 0`` always restores the exact uncorrected baseline
  (reversible).
* For polarized Nomad scans with interleaved frames, ``apply_curvature_by_spin``
  splits frames by ``Flipper1`` state (and excludes unwritten frames) before
  shearing so spin-up and spin-down frames are never interpolated together.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "apply_curvature_by_spin",
    "apply_curvature_to_map",
    "apply_curvature_to_slice",
]


def apply_curvature_to_slice(roi_slice: np.ndarray, curvature_rad: float) -> np.ndarray:
    """Shear and linearly regrid one single-spin ROI slice of shape ``(n_points, width)``.

    Returns a new ``float64`` array with the same shape as ``roi_slice``.
    """
    arr = np.asarray(roi_slice, dtype=float)
    if arr.ndim != 2:
        raise ValueError("roi_slice must be a 2D array of shape (n_points, width).")
    curv = float(curvature_rad)
    if not np.isfinite(curv):
        raise ValueError("Sample curvature must be a finite number (radians).")
    if abs(curv) >= (np.pi / 2.0 - 1e-6):
        raise ValueError("Sample curvature magnitude must be strictly less than pi/2 rad.")

    n_points, width = arr.shape
    if curv == 0.0 or n_points == 0 or width == 0:
        return arr.copy()

    tan_c = float(np.tan(curv))
    displacements = np.arange(width, dtype=float) * tan_c
    min_y = float(min(0.0, float(displacements.min())))
    roi_mid = width / 2.0
    zero_level = int(round(roi_mid * tan_c - min_y))

    y_target = min_y + np.arange(zero_level, zero_level + n_points, dtype=float)
    y_base = np.arange(n_points, dtype=float)

    out = np.empty((n_points, width), dtype=float)
    for x in range(width):
        out[:, x] = np.interp(y_target, y_base + displacements[x], arr[:, x], left=0.0, right=0.0)
    return out


def _normalize_x_window(window: tuple[int, int], nx: int) -> tuple[int, int]:
    a, b = int(round(window[0])), int(round(window[1]))
    x0 = max(0, min(nx, min(a, b)))
    x1 = max(0, min(nx, max(a, b)))
    return x0, x1


def apply_curvature_to_map(
    map_2d: np.ndarray,
    curvature_rad: float,
    roi_x_bounds: tuple[int, int],
    bkg_x_bounds: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply sample curvature correction to a single-spin Y-integrated map ``(n_points, nx)``.

    Only columns inside ``roi_x_bounds`` (and independently ``bkg_x_bounds`` if
    provided) are corrected, matching pySAred's ROI-scoped curvature convention.
    The input array is never modified.
    """
    arr = np.asarray(map_2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError("map_2d must be a 2D array of shape (n_points, nx).")

    out = arr.copy()
    curv = float(curvature_rad)
    if not np.isfinite(curv):
        raise ValueError("Sample curvature must be a finite number (radians).")
    if curv == 0.0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return out

    nx = arr.shape[1]
    sx0, sx1 = _normalize_x_window(roi_x_bounds, nx)
    windows = [(sx0, sx1)]

    if bkg_x_bounds is not None:
        bx0, bx1 = _normalize_x_window(bkg_x_bounds, nx)
        if bx1 > bx0:
            if sx1 > sx0 and not (bx1 <= sx0 or sx1 <= bx0):
                raise ValueError(
                    "Signal ROI and background ROI overlap in X; keep them disjoint when curvature correction is active."
                )
            windows.append((bx0, bx1))

    for x0, x1 in windows:
        if x1 > x0:
            out[:, x0:x1] = apply_curvature_to_slice(arr[:, x0:x1], curv)

    return out


def apply_curvature_by_spin(
    counts_2d: np.ndarray,
    flipper: np.ndarray | None,
    written: np.ndarray | None,
    curvature_rad: float,
    roi_x_bounds: tuple[int, int],
    bkg_x_bounds: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply curvature correction independently to each spin state on written frames only.

    Nomad NeXus files interleave spin-up (``Flipper1 == 1``) and spin-down
    (``Flipper1 == 0``) frames in one detector stack.  This helper extracts each
    spin state's written frames in scan order, applies ``apply_curvature_to_map``,
    and returns a new ``(n_frames, nx)`` array without mutating ``counts_2d``.
    """
    arr = np.asarray(counts_2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError("counts_2d must be a 2D array of shape (n_frames, nx).")

    curv = float(curvature_rad)
    if not np.isfinite(curv):
        raise ValueError("Sample curvature must be a finite number (radians).")
    out = arr.copy()
    if curv == 0.0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return out

    n_frames = arr.shape[0]
    usable = np.ones(n_frames, dtype=bool)
    if written is not None:
        usable &= np.asarray(written, dtype=bool)[:n_frames]

    if flipper is None:
        groups = [np.flatnonzero(usable)]
    else:
        f = np.asarray(flipper, dtype=float)[:n_frames]
        down = np.flatnonzero(usable & (f == 0))
        up = np.flatnonzero(usable & (f == 1))
        other = np.flatnonzero(usable & (f != 0) & (f != 1))
        groups = [g for g in (down, up, other) if g.size > 0]

    for idx in groups:
        if idx.size == 0:
            continue
        out[idx] = apply_curvature_to_map(
            arr[idx], curv, roi_x_bounds, bkg_x_bounds=bkg_x_bounds
        )

    return out
