"""Detector X pixel -> exit angle calibration for off-specular α_i / α_f maps.

Geometry (displayed detector coordinates, X = raw axis 1)::

    α_i         = θ
    α_f(x)      = (2θ − θ) + sign · (x − x₀) · deg_per_pixel

* ``deg_per_pixel`` comes from the direct beam. With the sample in, part of the
  beam passes the sample unreflected and stays at lab angle 0, i.e. at an
  angular offset −2θ from the detector reference direction. Its pixel position
  therefore moves linearly with 2θ; the fitted slope (px per degree of 2θ)
  gives degrees per pixel and the direction of X (``sign``).
* ``x₀`` is the pixel of exit angle 2θ − θ. It is taken from the specular peak
  (α_f = α_i in a specular scan), so small θ/2θ misalignments do not shift the
  specular ridge off the α_i = α_f diagonal.
* The linear pixel→angle mapping is the one the calibration fits; across the
  ~8° the detector covers it differs from atan geometry by < 0.002°.

Each scan holds its own calibration (in memory, like its ROI). Nothing is
written to measurement files.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


class CalibrationError(ValueError):
    """The data do not support an automatic calibration; a manual value is needed."""


@dataclass(frozen=True)
class DirectBeamFit:
    """Linear fit of direct-beam pixel position against 2θ."""

    slope_px_per_deg: float
    intercept_px: float
    rms_px: float
    frames: np.ndarray  # frame indices used in the fit
    positions_px: np.ndarray  # fitted-to centroid positions for those frames
    tth: np.ndarray

    @property
    def deg_per_pixel(self) -> float:
        return 1.0 / abs(self.slope_px_per_deg)

    @property
    def sign(self) -> int:
        # The direct beam sits at lower α_f than the reflected beam; if it
        # moves to larger X as 2θ grows, α_f decreases with X.
        return -1 if self.slope_px_per_deg > 0 else 1

    def position_at(self, tth) -> np.ndarray:
        return self.intercept_px + self.slope_px_per_deg * np.asarray(tth, dtype=float)


@dataclass(frozen=True)
class AngleCalibration:
    deg_per_pixel: float
    x0: float
    sign: int  # +1: α_f increases with displayed X; −1: decreases
    fit: DirectBeamFit | None = None
    deg_per_pixel_source: str = "direct-beam fit"
    x0_source: str = "specular peak"
    fit_y_rows: tuple[int, int] | None = None  # detector Y rows integrated for the fit

    def __post_init__(self):
        if not (np.isfinite(self.deg_per_pixel) and self.deg_per_pixel > 0):
            raise ValueError("Degrees per pixel must be a positive finite number")
        if not np.isfinite(self.x0):
            raise ValueError("Reference pixel x₀ must be finite")
        if self.sign not in (-1, 1):
            raise ValueError("Direction sign must be +1 or −1")

    def with_manual(self, *, deg_per_pixel=None, x0=None, sign=None) -> AngleCalibration:
        """Independent copy with user overrides; the fit is kept for reference."""
        changes = {}
        if deg_per_pixel is not None and deg_per_pixel != self.deg_per_pixel:
            changes.update(deg_per_pixel=float(deg_per_pixel), deg_per_pixel_source="manual")
        if x0 is not None and x0 != self.x0:
            changes.update(x0=float(x0), x0_source="manual")
        if sign is not None and sign != self.sign:
            changes.update(sign=int(sign))
        return replace(self, **changes) if changes else self

    def describe(self) -> str:
        text = (f"{self.deg_per_pixel:.5f}°/px ({self.deg_per_pixel_source}), "
                f"x₀ = {self.x0:.1f} px ({self.x0_source}), "
                f"α_f {'increases' if self.sign > 0 else 'decreases'} with X")
        if self.fit is not None:
            text += (f" | Direct-beam fit: {abs(self.fit.slope_px_per_deg):.2f} px/° over "
                     f"{self.fit.frames.size} frames, rms {self.fit.rms_px:.2f} px")
        if self.fit_y_rows is not None:
            text += f" | fitted on Y rows [{self.fit_y_rows[0]}, {self.fit_y_rows[1]})"

        return text


def find_direct_beam(counts: np.ndarray, tth: np.ndarray, exclude: tuple[float, float], *,
                     min_tth: float = 0.1, min_counts: float = 50.0, contrast: float = 8.0,
                     half_width: int = 6):
    """Per-frame centroid (pixel-centre coordinates) of the strongest peak outside the specular band.

    A frame contributes only if that peak is well above the frame's background
    (``contrast`` × median + 1 counts) and has at least ``min_counts``.
    Returns (frame indices, centroid pixels).
    """
    counts = np.asarray(counts, dtype=float)
    tth = np.asarray(tth, dtype=float)
    lo, hi = int(np.floor(min(exclude))), int(np.ceil(max(exclude)))
    x = np.arange(counts.shape[1], dtype=float) + 0.5  # pixel i spans [i, i + 1)
    frames, positions = [], []
    for i in range(min(counts.shape[0], tth.size)):
        if not np.isfinite(tth[i]) or abs(tth[i]) < min_tth:
            continue
        profile = np.nan_to_num(counts[i], nan=0.0)
        outside = profile.copy()
        outside[max(lo, 0):max(hi, 0)] = 0.0
        j = int(np.argmax(outside))
        peak = outside[j]
        if peak < min_counts or peak < contrast * (np.median(profile) + 1.0):
            continue
        a, b = max(j - half_width, 0), min(j + half_width + 1, profile.size)
        if a < hi and b > lo:  # peak touching the specular band: not separable
            continue
        w = outside[a:b]
        frames.append(i)
        positions.append(float(np.sum(x[a:b] * w) / np.sum(w)))
    return np.asarray(frames, dtype=int), np.asarray(positions, dtype=float)


def fit_direct_beam(counts: np.ndarray, tth: np.ndarray, exclude: tuple[float, float], *,
                    min_frames: int = 5, reject_px: float = 3.0) -> DirectBeamFit:
    """Robust straight-line fit of direct-beam position against 2θ.

    Outliers (other scattering features winning the per-frame peak search) are
    removed iteratively at max(reject_px, 3 × robust σ) from the line.
    """
    frames, pos = find_direct_beam(counts, tth, exclude)
    tth = np.asarray(tth, dtype=float)
    slope = intercept = np.nan
    keep = np.zeros(frames.size, dtype=bool)
    t_all = tth[frames]
    if frames.size >= min_frames and np.ptp(t_all) > 0:
        # Theil–Sen start: median pairwise slope is not dragged by a few strong outliers.
        i, j = np.triu_indices(frames.size, k=1)
        dt = t_all[j] - t_all[i]
        ok = dt != 0
        slope = float(np.median((pos[j] - pos[i])[ok] / dt[ok]))
        intercept = float(np.median(pos - slope * t_all))
        keep = np.ones(frames.size, dtype=bool)
    for _ in range(10):
        if keep.sum() < min_frames:
            break
        resid = pos - (intercept + slope * t_all)
        sigma = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        new_keep = np.abs(resid) <= max(reject_px, 3.0 * sigma)
        if new_keep.sum() < min_frames or np.ptp(t_all[new_keep]) <= 0:
            keep = new_keep
            break
        slope, intercept = np.polyfit(t_all[new_keep], pos[new_keep], 1)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if keep.sum() < min_frames or not np.isfinite(slope) or slope == 0:
        raise CalibrationError(
            f"No direct-beam streak found ({int(keep.sum())} usable frames, {min_frames} needed). "
            "Enter °/pixel manually.")
    used = frames[keep]
    resid = pos[keep] - (intercept + slope * tth[used])
    return DirectBeamFit(float(slope), float(intercept), float(np.sqrt(np.mean(resid**2))),
                         used, pos[keep], tth[used])


def specular_center(counts: np.ndarray, x_ranges: np.ndarray, *, min_counts: float = 200.0) -> float:
    """Median over frames of the intensity centroid inside each frame's ROI X range."""
    counts = np.asarray(counts, dtype=float)
    centres = []
    for i, (a, b) in enumerate(np.asarray(x_ranges, dtype=float)[: counts.shape[0]]):
        lo, hi = int(round(min(a, b))), int(round(max(a, b)))
        lo, hi = max(lo, 0), min(hi, counts.shape[1])
        if hi <= lo:
            continue
        w = np.nan_to_num(counts[i, lo:hi], nan=0.0)
        if w.sum() >= min_counts:
            # Pixel i spans [i, i + 1): centroid in the same edge coordinates as α_f.
            centres.append(float(np.sum((np.arange(lo, hi) + 0.5) * w) / w.sum()))
    if not centres:
        raise CalibrationError("No frame has enough specular counts in the ROI to locate x₀.")
    return float(np.median(centres))


def calibrate(counts: np.ndarray, tth: np.ndarray, roi_x_ranges: np.ndarray, *,
              guard_px: float = 15.0, y_rows: tuple[int, int] | None = None) -> AngleCalibration:
    """Direct-beam °/pixel and direction, with x₀ from the specular peak."""
    x0 = specular_center(counts, roi_x_ranges)
    fit = fit_direct_beam(counts, tth, (x0 - guard_px, x0 + guard_px))
    return AngleCalibration(fit.deg_per_pixel, x0, fit.sign, fit,
                            fit_y_rows=None if y_rows is None else (int(y_rows[0]), int(y_rows[1])))


def alpha_f_offsets(nx: int, calibration: AngleCalibration, *, edges: bool = False) -> np.ndarray:
    """α_f − (2θ − θ) for pixel centres (default) or the nx + 1 pixel edges."""
    x = np.arange(nx + 1, dtype=float) if edges else np.arange(nx, dtype=float) + 0.5
    return calibration.sign * (x - calibration.x0) * calibration.deg_per_pixel


def alpha_angles(th, tth, nx: int, calibration: AngleCalibration):
    """(α_i per frame, α_f per frame and pixel centre) for the given frames."""
    th = np.asarray(th, dtype=float)
    tth = np.asarray(tth, dtype=float)
    alpha_f = (tth - th)[:, None] + alpha_f_offsets(nx, calibration)[None, :]
    return th.copy(), alpha_f


def alpha_i_strip_edges(alpha_i: np.ndarray) -> np.ndarray:
    """(n, 2) display extent of each frame's α_i strip.

    Each frame is measured at one θ; its strip spans halfway to the nearest
    distinct neighbouring θ on each side (display width only, no interpolation
    of counts). A single frame, or repeated θ, gets the typical step.
    """
    a = np.asarray(alpha_i, dtype=float)
    if a.size == 0:
        return np.zeros((0, 2))
    distinct = np.unique(a[np.isfinite(a)])
    steps = np.diff(distinct)
    default = float(np.median(steps)) if steps.size else 0.01
    out = np.empty((a.size, 2))
    for k, v in enumerate(a):
        below = distinct[distinct < v]
        above = distinct[distinct > v]
        lo = (v - below[-1]) / 2 if below.size else default / 2
        hi = (above[0] - v) / 2 if above.size else default / 2
        out[k] = (v - lo, v + hi)
    return out


def q_from_angles(alpha_i, alpha_f, wavelength: float):
    """Momentum transfer (Å⁻¹) for incident/exit angles in degrees (as in pySAred).

        Qx = (2π/λ)(cos α_f − cos α_i)      Qz = (2π/λ)(sin α_f + sin α_i)

    Qx is reported as defined here (pySAred plots −Qx). The specular condition
    α_f = α_i gives Qx = 0; the direct beam (α_f = −α_i) gives Qz = 0.
    """
    if not (np.isfinite(wavelength) and wavelength > 0):
        raise ValueError("Wavelength must be a positive finite number")
    k0 = 2.0 * np.pi / wavelength
    ai = np.radians(np.asarray(alpha_i, dtype=float))
    af = np.radians(np.asarray(alpha_f, dtype=float))
    return k0 * (np.cos(af) - np.cos(ai)), k0 * (np.sin(af) + np.sin(ai))


def angles_from_q(qx, qz, wavelength: float):
    """Inverse of :func:`q_from_angles` (degrees), for |angles| < 90°.

    With A = Qz/k0 = 2 sin(s/2) cos(d/2) and B = Qx/k0 = −2 sin(s/2) sin(d/2),
    where s = α_i + α_f and d = α_f − α_i: d = 2 atan(−B / A), s = 2 asin(A / (2 cos(d/2))).
    |d/2| < 90°, so the plain arctan (not atan2) is correct also below the horizon
    (Qz < 0). The direct-beam line α_f = −α_i all maps to Q = 0 and cannot be
    inverted; it is returned as α_i = α_f = 0.
    """
    k0 = 2.0 * np.pi / wavelength
    a = np.asarray(qz, dtype=float) / k0
    b = np.asarray(qx, dtype=float) / k0
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(a != 0, 2.0 * np.arctan(-b / np.where(a != 0, a, 1.0)), 0.0)
        s = 2.0 * np.arcsin(np.clip(a / (2.0 * np.cos(d / 2.0)), -1.0, 1.0))
    return np.degrees((s - d) / 2.0), np.degrees((s + d) / 2.0)
