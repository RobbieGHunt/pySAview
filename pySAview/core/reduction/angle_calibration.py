"""Detector X pixel -> exit angle calibration for off-specular α_i / α_f maps.

Geometry (displayed detector coordinates, X = raw axis 1)::

    α_i         = θ
    α_f(x)      = (2θ − θ) + sign · (x − x₀) · deg_per_pixel

* ``deg_per_pixel`` comes from the direct beam. With the sample in, part of the
  beam passes the sample unreflected and stays at lab angle 0, i.e. at an
  angular offset −2θ from the detector reference direction. Its pixel position
  therefore moves linearly with 2θ; the fitted slope (px per degree of 2θ)
  gives degrees per pixel and the direction of X (``sign``).
* Low-2θ exclusion (default ``min_tth = 0.40°``): below ~0.40°, grazing-angle
  sample masking shifts the transmitted direct-beam centroid toward the
  specular ridge (Gibaud, Vignaud & Sinha 1993, *Acta Cryst.* A49, Eq. 13 &
  Fig. 4) and the direct beam overlaps and contaminates the specular ROI.
* Active-detector-edge exclusion (default ``edge_margin = 20`` px): peaks
  within ``edge_margin`` of the array borders, or within ``edge_margin`` of a
  data-detected internal active-cathode cutoff (such as SuperADAM's TDC edge
  near x ≈ 912 px inside a 1024 px readout), are excluded so clipped beam
  tails cannot pull the linear fit.
* ``x₀`` is the pixel of exit angle 2θ − θ. It is taken from the specular peak
  (α_f = α_i in a specular scan), preferring frames with |2θ| ≥ ``min_tth`` so
  low-angle direct-beam overlap does not bias ``x₀``.
* The linear pixel→angle mapping is the one the calibration fits. Across the
  detector (about −3° to +5.4° from x₀ in the example scans) it differs from
  flat-detector atan geometry by at most ~0.006°, under one pixel.
* No sample-to-detector distance is needed. With the known detector width
  (``DETECTOR_WIDTH_MM`` over the X pixels) the distance the calibration
  implies is L = p / tan(Δ) (:func:`implied_distance_mm`), shown for checking.

Each scan holds its own calibration (in memory, like its ROI). Nothing is
written to measurement files.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


DETECTOR_WIDTH_MM = 300.0  # SuperADAM 2D detector active area is 300 mm × 300 mm


def pixel_pitch_mm(n_pixels: int, width_mm: float = DETECTOR_WIDTH_MM) -> float:
    """Pixel spacing (mm) of a detector ``width_mm`` wide read out in ``n_pixels``."""
    if n_pixels <= 0:
        raise ValueError("Number of pixels must be positive")
    return width_mm / n_pixels


def implied_distance_mm(deg_per_pixel: float, pitch_mm: float) -> float:
    """Sample-to-detector distance at which one pixel at the detector normal subtends Δ.

        L = p / tan(Δ)

    Δ comes from the data (direct-beam fit or manual), so this is a derived
    check value: it is not used in any angle or Q conversion.
    """
    return pitch_mm / np.tan(np.radians(deg_per_pixel))


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
    min_tth_used: float | None = None
    max_tth_used: float | None = None
    excluded_low_tth: int = 0
    excluded_edge: int = 0
    detected_edge_px: float | None = None

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

    def distance_mm(self, pitch_mm: float) -> float:
        """Sample-to-detector distance implied by this Δ (see :func:`implied_distance_mm`)."""
        return implied_distance_mm(self.deg_per_pixel, pitch_mm)

    def describe(self, pitch_mm: float | None = None) -> str:
        text = (f"{self.deg_per_pixel:.5f}°/px ({self.deg_per_pixel_source}), "
                f"x₀ = {self.x0:.1f} px ({self.x0_source}), "
                f"α_f {'increases' if self.sign > 0 else 'decreases'} with X")
        if pitch_mm is not None:
            text += (f" | Implied sample–detector distance {self.distance_mm(pitch_mm):.0f} mm "
                     f"(pixel {pitch_mm:.4f} mm)")
        if self.fit is not None:
            tth_lo = self.fit.min_tth_used
            tth_hi = self.fit.max_tth_used
            if (tth_lo is None or tth_hi is None) and self.fit.tth.size:
                tth_lo = float(np.min(self.fit.tth))
                tth_hi = float(np.max(self.fit.tth))
            tth_span = (f" (2θ {tth_lo:.2f}°–{tth_hi:.2f}°)"
                        if tth_lo is not None and tth_hi is not None else "")
            text += (f" | Direct-beam fit: {abs(self.fit.slope_px_per_deg):.2f} px/° over "
                     f"{self.fit.frames.size} frames{tth_span}, rms {self.fit.rms_px:.2f} px")
        if self.fit_y_rows is not None:
            text += f" | fitted on Y rows [{self.fit_y_rows[0]}, {self.fit_y_rows[1]})"

        return text


def _validate_tth_bounds(min_tth: float, max_tth: float | None = None) -> tuple[float, float | None]:
    if isinstance(min_tth, bool) or not (np.isfinite(min_tth) and min_tth >= 0):
        raise ValueError("min_tth must be a finite non-negative angle in degrees")
    lo = float(min_tth)
    if max_tth is not None:
        if isinstance(max_tth, bool) or not (np.isfinite(max_tth) and max_tth > lo):
            raise ValueError("max_tth must be finite and strictly greater than min_tth")
        return lo, float(max_tth)
    return lo, None


def _validate_edge_margin(edge_margin: int) -> int:
    if isinstance(edge_margin, bool) or not isinstance(edge_margin, (int, np.integer)):
        raise ValueError("edge_margin must be a non-negative integer")
    margin = int(edge_margin)
    if margin < 0:
        raise ValueError("edge_margin must be non-negative")
    return margin


def _validate_written(written: np.ndarray | None) -> np.ndarray | None:
    if written is None:
        return None
    arr = np.asarray(written, dtype=bool)
    if arr.ndim != 1:
        raise ValueError("written mask must be a 1D boolean array")
    return arr


def _find_direct_beam_candidates(
    counts: np.ndarray,
    tth: np.ndarray,
    exclude: tuple[float, float],
    *,
    min_tth: float = 0.40,
    max_tth: float | None = None,
    min_counts: float = 50.0,
    contrast: float = 8.0,
    half_width: int = 6,
    edge_margin: int = 20,
    written: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, dict[int, int]]:
    min_tth, max_tth = _validate_tth_bounds(min_tth, max_tth)
    edge_margin = _validate_edge_margin(edge_margin)
    written_arr = _validate_written(written)
    counts = np.asarray(counts, dtype=float)
    tth = np.asarray(tth, dtype=float)
    nx = counts.shape[1]
    if nx == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float), 0, {}

    lo, hi = int(np.floor(min(exclude))), int(np.ceil(max(exclude)))
    x = np.arange(nx, dtype=float) + 0.5  # pixel i spans [i, i + 1)
    n = min(counts.shape[0], tth.size)
    if written_arr is not None:
        n = min(n, written_arr.size)

    frames: list[int] = []
    positions: list[float] = []
    excluded_low_tth = 0
    border_peaks: dict[int, int] = {}

    for i in range(n):
        if written_arr is not None and not written_arr[i]:
            continue
        if not np.isfinite(tth[i]):
            continue
        angle = abs(tth[i])
        if angle < min_tth:
            excluded_low_tth += 1
            continue
        if max_tth is not None and angle > max_tth:
            continue
        profile = np.nan_to_num(counts[i], nan=0.0, posinf=0.0, neginf=0.0)
        outside = profile.copy()
        outside[max(lo, 0):max(hi, 0)] = 0.0
        j = int(np.argmax(outside))
        peak = outside[j]
        if peak < min_counts or peak < contrast * (np.median(profile) + 1.0):
            continue
        if j < edge_margin or j >= nx - edge_margin:
            border_peaks[i] = j
            continue
        a, b = max(j - half_width, 0), min(j + half_width + 1, nx)
        if a < hi and b > lo:  # peak touching the specular band: not separable
            continue
        # Subtract local sideband background outside [a, b) and outside the specular band [lo, hi).
        flanks = []
        left_start = max(0, a - 12)
        if a > left_start:
            left_idx = np.arange(left_start, a)
            left_idx = left_idx[(left_idx < lo) | (left_idx >= hi)]
            if left_idx.size:
                flanks.append(profile[left_idx])
        right_stop = min(nx, b + 12)
        if right_stop > b:
            right_idx = np.arange(b, right_stop)
            right_idx = right_idx[(right_idx < lo) | (right_idx >= hi)]
            if right_idx.size:
                flanks.append(profile[right_idx])
        local_bg = float(np.median(np.concatenate(flanks))) if flanks else float(np.median(profile))
        w = np.clip(outside[a:b] - local_bg, 0.0, None)
        if w.sum() <= 0:
            w = np.clip(outside[a:b], 0.0, None)
        w_sum = float(w.sum())
        if w_sum <= 0:
            continue
        frames.append(i)
        positions.append(float(np.sum(x[a:b] * w) / w_sum))

    return (
        np.asarray(frames, dtype=int),
        np.asarray(positions, dtype=float),
        excluded_low_tth,
        border_peaks,
    )


def find_direct_beam(
    counts: np.ndarray,
    tth: np.ndarray,
    exclude: tuple[float, float],
    *,
    min_tth: float = 0.40,
    max_tth: float | None = None,
    min_counts: float = 50.0,
    contrast: float = 8.0,
    half_width: int = 6,
    edge_margin: int = 20,
    written: np.ndarray | None = None,
):
    """Per-frame centroid (pixel-centre coordinates) of the strongest peak outside the specular band.

    Below ``min_tth`` (default 0.40°), grazing-angle sample masking shifts the
    transmitted direct-beam centre toward the specular ridge (Gibaud, Vignaud &
    Sinha 1993, *Acta Cryst.* A49, Eq. 13 & Fig. 4) and the direct beam
    overlaps/contaminates the specular ROI, so those frames are excluded.
    Candidate peaks within ``edge_margin`` pixels of the detector array borders
    and unwritten frames (``written[i] == False``) are also skipped.

    A frame contributes only if the peak is well above the frame's background
    (``contrast`` × median + 1 counts) and has at least ``min_counts``. The
    centroid is weighted after subtracting a local sideband median background.
    Returns ``(frame indices, centroid pixels)``.
    """
    frames, positions, _, _ = _find_direct_beam_candidates(
        counts,
        tth,
        exclude,
        min_tth=min_tth,
        max_tth=max_tth,
        min_counts=min_counts,
        contrast=contrast,
        half_width=half_width,
        edge_margin=edge_margin,
        written=written,
    )
    return frames, positions


def fit_direct_beam(
    counts: np.ndarray,
    tth: np.ndarray,
    exclude: tuple[float, float],
    *,
    min_frames: int = 5,
    reject_px: float = 3.0,
    min_tth: float = 0.40,
    max_tth: float | None = None,
    edge_margin: int = 20,
    written: np.ndarray | None = None,
) -> DirectBeamFit:
    """Robust straight-line fit of direct-beam position against 2θ.

    Below ``min_tth`` (default 0.40°), grazing-angle sample masking shifts the
    direct-beam centre toward the specular ridge (Gibaud, Vignaud & Sinha 1993,
    *Acta Cryst.* A49, Eq. 13 & Fig. 4) and contaminates the specular ROI;
    those frames are excluded. Outliers (other scattering features winning the
    per-frame peak search) are removed iteratively at ``max(reject_px, 3 ×
    robust σ)`` from the line.

    After the initial robust fit, valid measured frames whose predicted direct-
    beam position lies past the inlier streak yet inside the detector array are
    used to detect an internal active-detector cutoff (such as SuperADAM's
    cathode border around x ≈ 912 px inside a 1024 px readout). When detected,
    inliers within ``edge_margin`` pixels of that cutoff are excluded and the
    line is refitted on the clean interior.
    """
    min_tth, max_tth = _validate_tth_bounds(min_tth, max_tth)
    edge_margin = _validate_edge_margin(edge_margin)
    written_arr = _validate_written(written)
    half_width = 6

    frames, pos, excluded_low_tth, border_peaks = _find_direct_beam_candidates(
        counts,
        tth,
        exclude,
        min_tth=min_tth,
        max_tth=max_tth,
        half_width=half_width,
        edge_margin=edge_margin,
        written=written_arr,
    )
    counts = np.asarray(counts, dtype=float)
    tth = np.asarray(tth, dtype=float)
    nx = counts.shape[1]
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
    if (keep.sum() < min_frames or not np.isfinite(slope) or slope == 0
            or np.ptp(pos[keep]) == 0 or not np.isfinite(1.0 / abs(slope))):
        raise CalibrationError(
            f"No direct-beam streak found ({int(keep.sum())} usable frames, {min_frames} needed). "
            "Enter °/pixel manually.")

    # Data-driven terminal active-detector cutoff detection.
    used_set = set(frames[keep].tolist())
    lo, hi = int(np.floor(min(exclude))), int(np.ceil(max(exclude)))
    n = min(counts.shape[0], tth.size)
    if written_arr is not None:
        n = min(n, written_arr.size)

    border_tol = max(reject_px, float(edge_margin))
    on_line_border = {
        k for k, j_pk in border_peaks.items()
        if abs((j_pk + 0.5) - (intercept + slope * tth[k])) <= border_tol
    }

    probe_x_pred: list[float] = []
    for k in range(n):
        if written_arr is not None and not written_arr[k]:
            continue
        if not np.isfinite(tth[k]):
            continue
        angle = abs(tth[k])
        if angle < min_tth or (max_tth is not None and angle > max_tth):
            continue
        if k in used_set or k in on_line_border:
            continue
        xp = float(intercept + slope * tth[k])
        if (lo - half_width - 1) <= xp <= (hi + half_width + 1):
            continue
        probe_x_pred.append(xp)

    probe_x = np.asarray(probe_x_pred, dtype=float)
    x_hi = float(np.max(pos[keep]))
    x_lo = float(np.min(pos[keep]))
    upper_border = [border_peaks[k] + 0.5 for k in on_line_border if border_peaks[k] >= nx - edge_margin]
    lower_border = [border_peaks[k] + 0.5 for k in on_line_border if border_peaks[k] < edge_margin]
    x_hi_edge = max(x_hi, max(upper_border)) if upper_border else x_hi
    x_lo_edge = min(x_lo, min(lower_border)) if lower_border else x_lo

    has_upper_edge = int(np.sum((probe_x > x_hi_edge + 2.0) & (probe_x < nx - 1.0))) >= 2
    has_lower_edge = int(np.sum((probe_x > 1.0) & (probe_x < x_lo_edge - 2.0))) >= 2

    detected_edge_px: float | None = None
    trimmed_mask = np.zeros(frames.size, dtype=bool)
    if has_upper_edge:
        detected_edge_px = x_hi_edge
        if edge_margin > 0:
            cand_trim = keep & (pos > x_hi_edge - edge_margin)
            remaining = keep & ~cand_trim
            if remaining.sum() >= min_frames and np.ptp(t_all[remaining]) > 0:
                trimmed_mask |= cand_trim
    if has_lower_edge:
        if detected_edge_px is None or slope < 0:
            detected_edge_px = x_lo_edge
        if edge_margin > 0:
            cand_trim = keep & ~trimmed_mask & (pos < x_lo_edge + edge_margin)
            remaining = keep & ~(trimmed_mask | cand_trim)
            if remaining.sum() >= min_frames and np.ptp(t_all[remaining]) > 0:
                trimmed_mask |= cand_trim

    trimmed_edge = int(trimmed_mask.sum())
    if trimmed_edge > 0:
        keep = keep & ~trimmed_mask
        if np.ptp(pos[keep]) == 0:
            raise CalibrationError(
                f"No direct-beam streak found ({int(keep.sum())} usable frames after edge trim, "
                f"{min_frames} needed). Enter °/pixel manually.")
        new_slope, new_intercept = np.polyfit(t_all[keep], pos[keep], 1)
        if not np.isfinite(new_slope) or new_slope == 0 or not np.isfinite(1.0 / abs(new_slope)):
            raise CalibrationError(
                f"No direct-beam streak found ({int(keep.sum())} usable frames after edge trim, "
                f"{min_frames} needed). Enter °/pixel manually.")
        slope, intercept = float(new_slope), float(new_intercept)

    used = frames[keep]
    tth_used = tth[used]
    resid = pos[keep] - (intercept + slope * tth_used)
    return DirectBeamFit(
        float(slope),
        float(intercept),
        float(np.sqrt(np.mean(resid**2))),
        used,
        pos[keep],
        tth_used,
        min_tth_used=float(np.min(tth_used)),
        max_tth_used=float(np.max(tth_used)),
        excluded_low_tth=int(excluded_low_tth),
        excluded_edge=int(len(on_line_border) + trimmed_edge),
        detected_edge_px=detected_edge_px,
    )


def specular_center(
    counts: np.ndarray,
    x_ranges: np.ndarray,
    *,
    min_counts: float = 200.0,
    tth: np.ndarray | None = None,
    min_tth: float = 0.40,
    written: np.ndarray | None = None,
) -> float:
    """Median over frames of the intensity centroid inside each frame's ROI X range.

    When ``tth`` is provided, prefers frames with ``|2θ| >= min_tth`` (default
    0.40°) so low-angle direct-beam overlap inside the specular ROI cannot bias
    ``x₀`` (Gibaud, Vignaud & Sinha 1993, *Acta Cryst.* A49, Eq. 13 & Fig. 4).
    Falls back to all valid measured frames only if no frame with ``|2θ| >=
    min_tth`` meets ``min_counts``. Unwritten frames (``written[i] == False``)
    are always skipped.
    """
    min_tth, _ = _validate_tth_bounds(min_tth)
    written_arr = _validate_written(written)
    counts = np.asarray(counts, dtype=float)
    ranges = np.asarray(x_ranges, dtype=float)
    tth_arr = None if tth is None else np.asarray(tth, dtype=float)
    n = min(counts.shape[0], ranges.shape[0])
    if written_arr is not None:
        n = min(n, written_arr.size)

    high_tth_centres: list[float] = []
    all_centres: list[float] = []
    for i in range(n):
        if written_arr is not None and not written_arr[i]:
            continue
        a, b = ranges[i]
        lo, hi = int(round(min(a, b))), int(round(max(a, b)))
        lo, hi = max(lo, 0), min(hi, counts.shape[1])
        if hi <= lo:
            continue
        w = np.clip(np.nan_to_num(counts[i, lo:hi], nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        total = float(w.sum())
        if total > 0 and total >= min_counts:
            # Pixel i spans [i, i + 1): centroid in the same edge coordinates as α_f.
            centre = float(np.sum((np.arange(lo, hi) + 0.5) * w) / total)
            all_centres.append(centre)
            if tth_arr is not None and i < tth_arr.size and np.isfinite(tth_arr[i]) and abs(tth_arr[i]) >= min_tth:
                high_tth_centres.append(centre)
    centres = high_tth_centres if (tth_arr is not None and high_tth_centres) else all_centres
    if not centres:
        raise CalibrationError("No frame has enough specular counts in the ROI to locate x₀.")
    return float(np.median(centres))


def calibrate(
    counts: np.ndarray,
    tth: np.ndarray,
    roi_x_ranges: np.ndarray,
    *,
    guard_px: float = 15.0,
    y_rows: tuple[int, int] | None = None,
    min_tth: float = 0.40,
    max_tth: float | None = None,
    edge_margin: int = 20,
    written: np.ndarray | None = None,
) -> AngleCalibration:
    """Direct-beam °/pixel and direction, with x₀ from the specular peak.

    Applies low-2θ exclusion (``min_tth``, default 0.40°; Gibaud, Vignaud &
    Sinha 1993, *Acta Cryst.* A49, Eq. 13 & Fig. 4) to both the specular centre
    and direct-beam fit, optional upper ``max_tth`` cutoff, active-detector-
    edge exclusion (``edge_margin`` px), and unwritten-frame skipping
    (``written``).
    """
    _validate_tth_bounds(min_tth, max_tth)
    _validate_edge_margin(edge_margin)
    written_arr = _validate_written(written)
    x0 = specular_center(counts, roi_x_ranges, tth=tth, min_tth=min_tth, written=written_arr)
    fit = fit_direct_beam(
        counts,
        tth,
        (x0 - guard_px, x0 + guard_px),
        min_tth=min_tth,
        max_tth=max_tth,
        edge_margin=edge_margin,
        written=written_arr,
    )
    return AngleCalibration(
        fit.deg_per_pixel,
        x0,
        fit.sign,
        fit,
        fit_y_rows=None if y_rows is None else (int(y_rows[0]), int(y_rows[1])),
    )


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


__all__ = [
    "AngleCalibration",
    "CalibrationError",
    "DETECTOR_WIDTH_MM",
    "DirectBeamFit",
    "alpha_angles",
    "alpha_f_offsets",
    "alpha_i_strip_edges",
    "calibrate",
    "find_direct_beam",
    "fit_direct_beam",
    "implied_distance_mm",
    "pixel_pitch_mm",
    "specular_center",
]
