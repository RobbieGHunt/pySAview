"""Background ROI geometry and subtraction following pySAred 1.5.3.

Implements the single background ROI convention from ``pySAred_V1.5.3.py``:

* The background ROI shares the signal ROI's vertical ``[y0, y1)`` bounds and
  horizontal pixel width ``w = x1 - x0``.
* By default it sits one ROI-width to the left of the signal ROI:
  ``[x0 - 2*w, x0 - w]`` (leaving a gap of width ``w`` between signal and
  background so specular tails do not leak into the background estimate).
* For every point with ``Qz > bkg_skip_qz`` and ``bkg_counts > 0``:

      IntensErr = sqrt(Intens + Intens_bkg)
      Intens    = Intens - Intens_bkg

  Points with ``Qz <= bkg_skip_qz`` or ``bkg_counts <= 0`` are left unchanged.
"""

from __future__ import annotations

import numpy as np

from pySAview.core.reduction.roi_extraction import pixel_bounds

__all__ = [
    "bkg_roi_from_signal",
    "default_bkg_x_bounds",
    "rois_overlap_x",
    "subtract_background_pysared",
]


def default_bkg_x_bounds(roi_x0: float, roi_x1: float, nx: int = 1024) -> tuple[float, float]:
    """Return pySAred 1.5.3's default background X bounds ``(x_left - 2*w, x_left - w)``.

    If placing the background ROI to the left would extend below pixel 0, it is
    placed symmetrically to the right ``(x_right + w, x_right + 2*w)`` (or
    clamped inside ``[0, nx]``) so the default never starts out-of-bounds.
    """
    x_left = min(float(roi_x0), float(roi_x1))
    x_right = max(float(roi_x0), float(roi_x1))
    width = max(1.0, x_right - x_left)

    bkg_x0 = x_left - 2.0 * width
    bkg_x1 = x_left - width
    if bkg_x0 >= 0.0:
        return (bkg_x0, bkg_x1)

    # Fallback to the right side if signal ROI is near the left detector edge
    alt_x0 = x_right + width
    alt_x1 = x_right + 2.0 * width
    if alt_x1 <= float(nx):
        return (alt_x0, alt_x1)

    # Final safe clamp inside [0, nx]
    clamped_x1 = min(float(nx), max(width, x_left))
    clamped_x0 = max(0.0, clamped_x1 - width)
    return (clamped_x0, clamped_x1)


def bkg_roi_from_signal(
    signal_roi: tuple[float, float, float, float],
    bkg_x_bounds: tuple[float, float] | None,
    nx: int,
    ny: int,
) -> tuple[int, int, int, int]:
    """Resolve integer half-open ``(bx0, bx1, by0, by1)`` using shared ROI pixel bounds.

    Background Y bounds always match the signal ROI's Y bounds (as in pySAred 1.5.3).
    """
    sx0, sx1, sy0, sy1 = pixel_bounds(signal_roi, nx, ny)
    if bkg_x_bounds is None:
        bx_raw = default_bkg_x_bounds(float(sx0), float(sx1), nx=nx)
    else:
        bx_raw = (float(bkg_x_bounds[0]), float(bkg_x_bounds[1]))
    bx0, bx1, _, _ = pixel_bounds((bx_raw[0], bx_raw[1], float(sy0), float(sy1)), nx, ny)
    return bx0, bx1, sy0, sy1


def rois_overlap_x(
    signal_x: tuple[float, float],
    bkg_x: tuple[float, float],
    nx: int = 1024,
) -> bool:
    """True when the pixel-rounded signal and background X intervals intersect."""
    sx0, sx1, _, _ = pixel_bounds((signal_x[0], signal_x[1], 0.0, 1.0), nx, 1)
    bx0, bx1, _, _ = pixel_bounds((bkg_x[0], bkg_x[1], 0.0, 1.0), nx, 1)
    return sx1 > sx0 and bx1 > bx0 and not (bx1 <= sx0 or sx1 <= bx0)


def subtract_background_pysared(
    intensity: np.ndarray,
    intensity_err: np.ndarray,
    bkg_intensity: np.ndarray,
    qz: np.ndarray,
    bkg_skip_qz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract background ROI counts using pySAred 1.5.3's exact rule.

    Only points with finite ``qz > bkg_skip_qz`` and finite ``bkg_intensity > 0``
    are subtracted, with ``err = sqrt(max(intensity, 0) + bkg_intensity)`` and
    ``net = intensity - bkg_intensity``.
    """
    intens = np.asarray(intensity, dtype=float)
    err = np.asarray(intensity_err, dtype=float)
    bkg = np.asarray(bkg_intensity, dtype=float)
    q = np.asarray(qz, dtype=float)

    if not (intens.ndim == err.ndim == bkg.ndim == q.ndim == 1):
        raise ValueError("Background subtraction inputs must be 1D arrays.")
    if not (intens.size == err.size == bkg.size == q.size):
        raise ValueError("Background subtraction arrays must have matching lengths.")
    if not np.isfinite(bkg_skip_qz):
        raise ValueError("Background Qz skip threshold must be finite.")

    out_i = intens.copy()
    out_err = err.copy()

    apply = np.isfinite(intens) & np.isfinite(bkg) & np.isfinite(q) & (q > float(bkg_skip_qz)) & (bkg > 0.0)
    out_err[apply] = np.sqrt(np.maximum(intens[apply], 0.0) + bkg[apply])
    out_i[apply] = intens[apply] - bkg[apply]

    return out_i, out_err
