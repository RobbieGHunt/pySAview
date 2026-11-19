"""Reflectometry data reduction pipeline following pySAred 1.5.3.

Chains the standalone physical calculation modules in the exact pySAred 1.5.3
execution order while keeping all transformations in memory and per-scan:

    1. Offset-corrected incident angle:  th_corr = th - theta_offset_deg
       and Qz = (4 * pi / lambda) * sin(radians(th_corr))
    2. Optional per-spin sample curvature shear inside the signal (and BKG) ROI
       (``pySAview.core.reduction.curvature``) -> integrated ROI counts ``I`` (and ``I_bkg``)
    3. Counting error initialization:  dI = 1 if I == 0 else sqrt(I)
       (or sqrt(variance) when explicit variance is supplied)
    4. Optional background subtraction for Qz > bkg_skip_qz and I_bkg > 0
       (``pySAview.core.reduction.background``)
    5. Monitor / dwell-time normalization (pySAred 1.5.3 error propagation)
    6. Optional overillumination (footprint) correction: I /= c0, dI /= c0
       (``pySAview.core.reduction.overillumination``)
    7. Optional manual scale factor: I /= scale_factor, dI /= scale_factor
    8. Instrumental resolution sigma_Qz / sigma_theta
       (``pySAview.core.reduction.resolution``)
    9. Optional left/right point trimming and zero-intensity filtering

Original measurement files remain read-only; every scan owns its own
``ReductionParams`` instance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from pySAview.core.reduction.background import subtract_background_pysared
from pySAview.core.reduction.overillumination import (
    DEFAULT_DISTANCE_S1_TO_SAMPLE_MM,
    DEFAULT_DISTANCE_S2_TO_SAMPLE_MM,
    DEFAULT_SAMPLE_LENGTH_MM,
    compute_overillumination_series,
)
from pySAview.core.reduction.resolution import (
    DEFAULT_WAVELENGTH_RESOLUTION,
    compute_resolution_pysared,
)

__all__ = [
    "ReductionParams",
    "ReductionResult",
    "reduce_counts_pysared",
    "trim_curve_mask",
]


@dataclass
class ReductionParams:
    """Per-scan pySAred 1.5.3 reduction parameters (kept in memory only)."""

    # Angle offset & instrument geometry (SuperADAM / pySAred 1.5.3 defaults)
    theta_offset_deg: float = 0.0
    wavelength_resolution: float = DEFAULT_WAVELENGTH_RESOLUTION
    distance_s1_mm: float = DEFAULT_DISTANCE_S1_TO_SAMPLE_MM
    distance_s2_mm: float = DEFAULT_DISTANCE_S2_TO_SAMPLE_MM

    # Background subtraction from 2nd ROI (freely resizable; area-scaled if A_bkg != A_sig)
    subtract_background: bool = False
    bkg_x_bounds: tuple[float, float] | None = None
    bkg_roi_bounds: tuple[float, float, float, float] | None = None
    bkg_skip_qz: float = 0.0

    # Sample curvature correction (shear angle in radians inside ROI X window)
    curvature_rad: float = 0.0

    # Overillumination (footprint) correction
    overillumination: bool = False
    sample_length_mm: float = DEFAULT_SAMPLE_LENGTH_MM
    show_overillumination_curve: bool = False

    # Manual scale factor (divide I and dI by scale_factor; Direct Beam deferred)
    use_scale_factor: bool = False
    scale_factor: float = 1.0

    # Resolution calculation & column
    resolution_mode: Literal["pysared", "sared"] = "pysared"
    include_resolution: bool = True

    # Per-curve point trimming & zero filter
    skip_left: int = 0
    skip_right: int = 0
    remove_zeros: bool = False

    def clone(self) -> ReductionParams:
        """Create an independent copy for explicit reuse on another scan."""
        return replace(
            self,
            bkg_x_bounds=None if self.bkg_x_bounds is None else (float(self.bkg_x_bounds[0]), float(self.bkg_x_bounds[1])),
            bkg_roi_bounds=(
                None
                if self.bkg_roi_bounds is None
                else tuple(float(v) for v in self.bkg_roi_bounds)
            ),
        )

    def has_active_reductions(self) -> bool:
        """True if any reduction correction beyond plain normalization is active."""
        return bool(
            self.overillumination
            or self.subtract_background
            or abs(self.curvature_rad) > 0.0
            or abs(self.theta_offset_deg) > 0.0
            or (self.use_scale_factor and self.scale_factor != 1.0)
            or self.skip_left > 0
            or self.skip_right > 0
            or self.remove_zeros
        )

    def summary(self) -> str:
        """Compact human-readable description of active pySAred 1.5.3 reductions."""
        parts: list[str] = []
        if abs(self.theta_offset_deg) > 0.0:
            parts.append(f"th_offset={self.theta_offset_deg:g} deg")
        if abs(self.curvature_rad) > 0.0:
            parts.append(f"curvature={self.curvature_rad:g} rad")
        if self.subtract_background:
            if self.bkg_roi_bounds is not None:
                bx = (
                    f" X=[{self.bkg_roi_bounds[0]:.0f}..{self.bkg_roi_bounds[1]:.0f}],"
                    f" Y=[{self.bkg_roi_bounds[2]:.0f}..{self.bkg_roi_bounds[3]:.0f}]"
                )
            elif self.bkg_x_bounds is not None:
                bx = f" X=[{self.bkg_x_bounds[0]:.0f}..{self.bkg_x_bounds[1]:.0f}]"
            else:
                bx = ""
            skip = f", skip Qz<{self.bkg_skip_qz:g}" if self.bkg_skip_qz > 0 else ""
            parts.append(f"BKG sub{bx}{skip}")
        if self.overillumination:
            parts.append(
                f"overillumination(L={self.sample_length_mm:g} mm, D1={self.distance_s1_mm:g}, D2={self.distance_s2_mm:g})"
            )
        if self.use_scale_factor:
            parts.append(f"scale/{self.scale_factor:g}")
        if self.skip_left > 0 or self.skip_right > 0:
            parts.append(f"skip(L={self.skip_left}, R={self.skip_right})")
        if self.remove_zeros:
            parts.append("remove zeros")
        if self.include_resolution:
            parts.append(f"res={self.resolution_mode}(dlam/lam={self.wavelength_resolution:g})")
        return "; ".join(parts) if parts else "none"


@dataclass
class ReductionResult:
    """Output of ``reduce_counts_pysared`` for one scan channel."""

    values: np.ndarray
    sigma: np.ndarray | None
    lower: np.ndarray | None
    upper: np.ndarray | None
    valid: np.ndarray
    reasons: np.ndarray
    method: str
    units: str
    overill_coeff: np.ndarray | None = None
    fwhm_proj: np.ndarray | None = None
    resolution: np.ndarray | None = None
    warnings: list[str] | None = None


def _as_1d(data, name: str, length: int | None = None) -> np.ndarray:
    if data is None:
        raise ValueError(f"{name} is required for the selected normalization.")
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1 or (length is not None and arr.size != length):
        raise ValueError(f"{name} must be a 1D array matching the scan length.")
    return arr


def reduce_counts_pysared(
    data,
    mode: str = "raw",
    *,
    monitor=None,
    time=None,
    count_data: bool = True,
    variance=None,
    monitor_variance=None,
    reference_monitor_rate: float = 1.0,
    numerator_is_monitor: bool = False,
    input_units: str = "counts",
    sigma_level: float = 1.0,
    theta_deg=None,
    wavelength_angstrom: float = 5.21,
    s1_mm=None,
    s2_mm=None,
    bkg_counts=None,
    bkg_area_ratio: float = 1.0,
    params: ReductionParams | None = None,
    resolution_angle_unit: Literal["Qz", "Degrees", "Radians"] = "Qz",
) -> ReductionResult:
    """Execute the pySAred 1.5.3 normalization, error propagation, and reduction pipeline.

    Matches ``pySAred_V1.5.3.py`` conventions:
    * Zero raw counts receive ``dI = 1`` before normalization (avoiding zero
      error bars at ``I = 0``).
    * Background subtraction runs before monitor/time division with
      ``dI = sqrt(max(I, 0) + I_bkg)`` where ``Qz > bkg_skip_qz`` and ``I_bkg > 0``.
    * Monitor normalization propagates ``sqrt((dI/M)^2 + I^2/M^3)`` (algebraically
      identical to pySAred's ``(I/M)*sqrt((dI/I)^2 + 1/M)`` for ``I > 0``,
      ``dI/M`` at ``I == 0``, and strictly nonnegative if ``I < 0`` after BKG
      subtraction).
    * Overillumination divides both ``I`` and ``dI`` by ``overill_corr > 0``.
    * Manual scale factor divides both ``I`` and ``dI`` by ``scale_factor > 0``.
    * Non-count channels (motors, temperature) can only be viewed raw and never
      receive invented counting uncertainties.
    """
    if not np.isfinite(sigma_level) or not 1.0 <= float(sigma_level) <= 6.0:
        raise ValueError("Sigma must be between 1 and 6.")
    if mode not in {"raw", "time", "monitor", "monitor_time", "monitor_cps"}:
        raise ValueError(f"Unknown normalization: {mode}")

    p = params if params is not None else ReductionParams()
    n = _as_1d(data, "Signal")
    if mode != "raw" and not count_data:
        raise ValueError("Count normalization is available only for count channels.")

    reasons = np.full(n.shape, "", dtype=object)
    warnings: list[str] = []

    def reject(mask: np.ndarray, reason: str) -> None:
        hit = mask & (reasons == "")
        reasons[hit] = reason

    reject(~np.isfinite(n), "nonfinite signal")

    # Non-count channels without supplied variance: raw display only, no fake error bars.
    if not count_data and variance is None:
        valid = reasons == ""
        values = np.where(valid, n, np.nan)
        return ReductionResult(
            values=values,
            sigma=None,
            lower=None,
            upper=None,
            valid=valid,
            reasons=reasons,
            method="unavailable (no supplied uncertainty)",
            units=input_units,
            warnings=warnings,
        )

    # Validate explicit signal variance if provided; otherwise apply pySAred 1.5.3 count rule.
    if variance is not None:
        var = _as_1d(variance, "Signal variance", n.size)
        reject(~np.isfinite(var) | (var < 0.0), "invalid signal variance")
        base_err = np.where(np.isfinite(var) & (var >= 0.0), np.sqrt(var), np.nan)
    else:
        reject(n < 0.0, "negative raw count")
        # Exact pySAred 1.5.3 rule: IntensErr = 1 if Intens == 0 else sqrt(Intens)
        base_err = np.where(n == 0.0, 1.0, np.sqrt(np.maximum(n, 0.0)))

    # Validate denominator inputs upfront so missing/bad channels fail explicitly.
    m = None
    vm = None
    t = None
    scale = 1.0
    units = input_units
    if mode == "time":
        t = _as_1d(time, "Dwell time", n.size)
        reject(~np.isfinite(t) | (t <= 0.0), "invalid dwell time")
        units = "counts/s"
    elif mode in {"monitor", "monitor_time", "monitor_cps"}:
        m = _as_1d(monitor, "Monitor counts", n.size)
        reject(~np.isfinite(m) | (m <= 0.0), "invalid monitor counts")
        if monitor_variance is not None:
            vm = _as_1d(monitor_variance, "Monitor variance", n.size)
            reject(~np.isfinite(vm) | (vm < 0.0), "invalid monitor variance")
        units = "counts/monitor count"
        if mode != "monitor":
            if not np.isfinite(reference_monitor_rate) or reference_monitor_rate <= 0:
                raise ValueError("Reference monitor rate must be finite and strictly positive.")
            scale = float(reference_monitor_rate)
            units = "counts/s at reference monitor rate"
        if numerator_is_monitor and not np.array_equal(n, m, equal_nan=True):
            raise ValueError("A shared numerator/monitor must reference the same observations.")

    # Offset-corrected incident angle & Qz (used for BKG skip, overillumination, resolution).
    th_corr = None
    qz = None
    if theta_deg is not None:
        th_arr = _as_1d(theta_deg, "Incident angle th", n.size)
        if not np.isfinite(p.theta_offset_deg):
            raise ValueError("Sample angle offset must be finite.")
        th_corr = th_arr - float(p.theta_offset_deg)
        if np.isfinite(wavelength_angstrom) and wavelength_angstrom > 0:
            qz = (4.0 * np.pi / float(wavelength_angstrom)) * np.sin(np.radians(th_corr))

    # Step 1: Work on finite valid raw intensities
    intens = n.copy()
    err = base_err.copy()

    # Step 2: Background subtraction (before monitor/time division, as in pySAred 1.5.3)
    if p.subtract_background and not numerator_is_monitor:
        if bkg_counts is None:
            warnings.append("Background subtraction skipped: no background ROI counts available.")
        elif qz is None:
            warnings.append("Background subtraction skipped: incident angle th is required for Qz threshold.")
        else:
            bkg_arr = _as_1d(bkg_counts, "Background counts", n.size)
            reject(~np.isfinite(bkg_arr) | (bkg_arr < 0.0), "invalid background counts")
            safe_i = np.where(np.isfinite(intens), intens, 0.0)
            safe_e = np.where(np.isfinite(err), err, 0.0)
            safe_b = np.where(np.isfinite(bkg_arr) & (bkg_arr >= 0.0), bkg_arr, 0.0)
            safe_q = np.where(np.isfinite(qz), qz, -np.inf)
            intens, err = subtract_background_pysared(
                safe_i, safe_e, safe_b, safe_q,
                bkg_skip_qz=p.bkg_skip_qz,
                area_ratio=bkg_area_ratio,
            )

    # Step 3: Monitor / Dwell-time normalization (pySAred 1.5.3 convention)
    valid = reasons == ""
    values = np.full_like(n, np.nan)
    sigma = np.full_like(n, np.nan)
    if mode == "raw":
        values[valid] = intens[valid]
        sigma[valid] = err[valid]
    elif mode == "time":
        values[valid] = intens[valid] / t[valid]
        sigma[valid] = err[valid] / t[valid]
    else:
        if numerator_is_monitor:
            values[valid] = scale
            sigma[valid] = 0.0
        else:
            mv = m[valid]
            iv = intens[valid]
            ev = err[valid]
            values[valid] = scale * iv / mv
            if vm is None:
                # Algebraically identical to pySAred 1.5.3:
                #   (I / M) * sqrt((dI / I)^2 + 1 / M)  for I > 0
                #   dI / M                             for I == 0
                # and stays positive if I < 0 after background subtraction.
                sigma[valid] = scale * np.sqrt((ev / mv) ** 2 + (iv ** 2) / (mv ** 3))
            else:
                sigma[valid] = scale * np.sqrt((ev / mv) ** 2 + (iv ** 2) * vm[valid] / (mv ** 4))

    # Step 4: Overillumination (footprint) calculation & optional division
    overill_coeff: np.ndarray | None = None
    fwhm_proj: np.ndarray | None = None
    need_overill_eval = (
        p.overillumination or p.show_overillumination_curve or p.include_resolution
    ) and not numerator_is_monitor
    if need_overill_eval and th_corr is not None and s1_mm is not None and s2_mm is not None:
        try:
            c0, fp, ovl_ok = compute_overillumination_series(
                s1_mm, s2_mm, th_corr,
                sample_length_mm=p.sample_length_mm,
                distance_s1_mm=p.distance_s1_mm,
                distance_s2_mm=p.distance_s2_mm,
            )
            overill_coeff = c0
            fwhm_proj = fp
        except ValueError as exc:
            if p.overillumination:
                raise ValueError(f"Overillumination parameters invalid: {exc}") from exc
            warnings.append(f"Overillumination/resolution geometry note: {exc}")
            ovl_ok = np.zeros(n.size, dtype=bool)

        if p.overillumination:
            reject(~ovl_ok | ~np.isfinite(overill_coeff) | (overill_coeff <= 0.0), "invalid slits or overillumination factor")
            valid = reasons == ""
            values[valid] = values[valid] / overill_coeff[valid]
            sigma[valid] = sigma[valid] / overill_coeff[valid]
    elif p.overillumination and not numerator_is_monitor:
        raise ValueError("Overillumination correction requires incident angle th and slit widths s1h/s2h.")

    # Step 5: Manual scale factor (divide I and dI, as in pySAred 1.5.3)
    if p.use_scale_factor and not numerator_is_monitor:
        if not np.isfinite(p.scale_factor) or p.scale_factor <= 0.0:
            raise ValueError("Scale factor must be finite and strictly positive.")
        valid = reasons == ""
        values[valid] = values[valid] / float(p.scale_factor)
        sigma[valid] = sigma[valid] / float(p.scale_factor)
        if mode != "raw":
            units = "scaled " + units

    # Step 6: Resolution calculation (Gaussian 1-sigma, pySAred 1.5.3)
    resolution: np.ndarray | None = None
    if p.include_resolution and th_corr is not None and s1_mm is not None and s2_mm is not None:
        try:
            s1_arr = np.asarray(s1_mm, dtype=float)
            s2_arr = np.asarray(s2_mm, dtype=float)
            # Match pySAred 1.5.3 SFM: use projected FWHM when overillumination is active, else s2h
            fp_for_res = fwhm_proj if (p.overillumination and fwhm_proj is not None) else s2_arr
            res = compute_resolution_pysared(
                th_corr, s1_arr, s2_arr, fp_for_res,
                wavelength_angstrom=wavelength_angstrom,
                wavelength_resolution=p.wavelength_resolution,
                distance_s1_mm=p.distance_s1_mm,
                distance_s2_mm=p.distance_s2_mm,
                mode=p.resolution_mode,
                angle_unit=resolution_angle_unit,
            )
            bad_slits = ~np.isfinite(s1_arr) | ~np.isfinite(s2_arr) | (s1_arr <= 0) | (s2_arr <= 0)
            res = np.where(bad_slits, np.nan, res)
            resolution = res
        except ValueError as exc:
            warnings.append(f"Resolution unavailable: {exc}")

    # Step 7: Final finiteness guard & symmetric display bounds (1-sigma reported in sigma)
    reject(~np.isfinite(values), "nonfinite reduced value")
    reject(~np.isfinite(sigma), "nonfinite uncertainty")
    valid = reasons == ""
    values[~valid] = np.nan
    sigma[~valid] = np.nan

    lower = np.full_like(n, np.nan)
    upper = np.full_like(n, np.nan)
    if numerator_is_monitor and m is not None:
        lower[valid] = scale
        upper[valid] = scale
        method = "shared monitor cancels exactly"
    else:
        lower[valid] = values[valid] - float(sigma_level) * sigma[valid]
        upper[valid] = values[valid] + float(sigma_level) * sigma[valid]
        method = (
            f"pySAred 1.5.3 standard uncertainty (1σ; displayed at {sigma_level:g}σ)"
            if variance is None
            else f"propagated standard uncertainty (first order); {sigma_level:g}σ bounds; standard uncertainty reported at 1σ"
        )

    return ReductionResult(
        values=values,
        sigma=sigma,
        lower=lower,
        upper=upper,
        valid=valid,
        reasons=reasons,
        method=method,
        units=units,
        overill_coeff=overill_coeff,
        fwhm_proj=fwhm_proj,
        resolution=resolution,
        warnings=warnings,
    )


def trim_curve_mask(
    n_points: int,
    y_values: np.ndarray,
    *,
    skip_left: int = 0,
    skip_right: int = 0,
    remove_zeros: bool = False,
) -> np.ndarray:
    """Boolean keep mask for one sorted spin curve (matches pySAred 1.5.3 SFM trimming)."""
    keep = np.ones(n_points, dtype=bool)
    left = max(0, int(skip_left))
    right = max(0, int(skip_right))
    if left > 0:
        keep[: min(left, n_points)] = False
    if right > 0:
        keep[max(0, n_points - right) :] = False
    if remove_zeros:
        y = np.asarray(y_values, dtype=float)
        keep &= y != 0.0
    return keep
