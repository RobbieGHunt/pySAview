"""1D and 2D count normalization with explicit statistical models and invalid-point masks.

Exact intervals assume unweighted, independent Poisson counts. Propagated
standard uncertainties use first-order derivatives, not confidence intervals.
See docs/normalization.md for formulas and the limits of these assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt

import numpy as np

# scipy.special is imported inside the exact-interval functions so SciPy stays
# off the launch path and unused SciPy subpackages can be excluded from bundles.

NORMALIZATION_UNITS = {
    "raw": "counts",
    "time": "counts/s",
    "monitor": "detector counts per monitor count",
}


def confidence_from_sigma(sigma_level):
    """Central Gaussian-equivalent coverage for an exact count interval."""
    if not np.isfinite(sigma_level) or not 1 <= sigma_level <= 6:
        raise ValueError("Sigma must be between 1 and 6")
    return erf(float(sigma_level) / sqrt(2.0))


CONFIDENCE = confidence_from_sigma(2)


@dataclass
class NormalizationResult:
    values: np.ndarray
    sigma: np.ndarray | None
    lower: np.ndarray | None
    upper: np.ndarray | None
    valid: np.ndarray
    reasons: np.ndarray
    method: str
    units: str


@dataclass
class NormalizedMap:
    """Display-ready values for the selected frames, with first-order 1σ uncertainty."""

    frame_indices: np.ndarray
    values: np.ndarray
    standard_uncertainty: np.ndarray
    mode: str
    units: str
    warnings: list[str] = field(default_factory=list)


def _vector(data, name, length=None):
    if data is None:
        raise ValueError(f"{name} is required for the selected normalization.")
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1 or (length is not None and arr.size != length):
        raise ValueError(f"{name} must be a 1D array matching the scan length.")
    return arr


def is_count_channel(name: str, unit: str) -> bool:
    """Conservative identification; integer-valued motors are not counters."""
    key = name.lower()
    if any(word in key for word in ("error", "sigma", "variance", "rate")):
        return False
    units = unit.strip().lower()
    return (units in {"count", "counts", "cts"}
            or (not units and name in {"ROI_Counts", "Detector1", "Monitor1", "Monitor2"}))


def poisson_interval(counts, confidence=CONFIDENCE):
    """Equal-tailed Garwood confidence limits on a Poisson mean."""
    n = _vector(counts, "Counts")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one.")
    if np.any(~np.isfinite(n) | (n < 0) | (n != np.rint(n))):
        raise ValueError("Poisson intervals require finite, nonnegative integer counts.")
    from scipy.special import gammainccinv, gammaincinv

    alpha = 1 - confidence
    lower = np.zeros_like(n)
    positive = n > 0
    lower[positive] = gammaincinv(n[positive], alpha / 2)
    upper = gammainccinv(n + 1, alpha / 2)
    return lower, upper


def poisson_ratio_interval(counts, monitor, confidence=CONFIDENCE):
    """Exact conditional limits on the ratio of two independent Poisson means.

    Conditional on N+M, N is binomial with p=lambda_N/(lambda_N+lambda_M).
    Transform equal-tailed Clopper-Pearson limits using p/(1-p).
    """
    n = _vector(counts, "Counts")
    m = _vector(monitor, "Monitor", len(n))
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one.")
    if np.any(~np.isfinite(n) | ~np.isfinite(m) | (n < 0) | (m <= 0)
              | (n != np.rint(n)) | (m != np.rint(m))):
        raise ValueError("Ratio intervals require integer counts and a positive monitor.")
    from scipy.special import betaincinv

    alpha = 1 - confidence
    lower = np.zeros_like(n)
    positive = n > 0
    p_low = betaincinv(n[positive], m[positive] + 1, alpha / 2)
    lower[positive] = p_low / (1 - p_low)
    # Compute the upper odds from the lower complementary probability. This
    # avoids subtracting an upper quantile extremely close to one.
    q_low = betaincinv(m, n + 1, alpha / 2)
    upper = (1 - q_low) / q_low
    return lower, upper


def normalize_signal(data, mode="raw", *, monitor=None, time=None,
                     count_data=True, variance=None, monitor_variance=None,
                     uncertainty="exact", reference_monitor_rate=1.0,
                     numerator_is_monitor=False, input_units="counts", sigma_level=2):
    """Prepare values, uncertainty bounds, and reasons for excluded points.

    raw: N; time: N/t; monitor: N/M; monitor_time: fixed_rate*N/M.
    The fixed rate is an exact display convention, shared across all scans.
    No mode falls back and no denominator is replaced by a fabricated value.
    """
    confidence = confidence_from_sigma(sigma_level)
    n = _vector(data, "Signal")
    if mode not in {"raw", "time", "monitor", "monitor_time", "monitor_cps"}:
        raise ValueError(f"Unknown normalization: {mode}")
    if uncertainty not in {"exact", "standard"}:
        raise ValueError(f"Unknown uncertainty method: {uncertainty}")
    if mode != "raw" and not count_data:
        raise ValueError("Count normalization is available only for count channels.")

    reasons = np.full(n.shape, "", dtype=object)

    def reject(mask, reason):
        mask = mask & (reasons == "")
        reasons[mask] = reason

    reject(~np.isfinite(n), "nonfinite signal")
    # Explicit variances describe processed/weighted observations; never
    # pretend those observations are independent integer Poisson events.
    poisson = count_data and variance is None
    if poisson:
        reject((n < 0) | (n != np.rint(n)), "not an unweighted Poisson count")
    var = _vector(variance, "Signal variance", len(n)) if variance is not None else (n if poisson else None)
    if var is not None:
        reject(~np.isfinite(var) | (var < 0), "invalid signal variance")

    denominator = np.ones_like(n)
    m = None
    vm = None
    scale = 1.0
    units = input_units
    if mode == "time":
        denominator = _vector(time, "Dwell time", len(n))
        reject(~np.isfinite(denominator) | (denominator <= 0), "invalid dwell time")
        units = "counts/s"
    elif mode in {"monitor", "monitor_time", "monitor_cps"}:
        m = _vector(monitor, "Monitor counts", len(n))
        denominator = m
        reject(~np.isfinite(m) | (m <= 0), "invalid monitor counts")
        if monitor_variance is None:
            reject(m != np.rint(m), "not an unweighted Poisson monitor count")
            vm = m
        else:
            vm = _vector(monitor_variance, "Monitor variance", len(n))
            reject(~np.isfinite(vm) | (vm < 0), "invalid monitor variance")
        units = "counts/monitor count"
        if mode != "monitor":
            if not np.isfinite(reference_monitor_rate) or reference_monitor_rate <= 0:
                raise ValueError("Reference monitor rate must be finite and strictly positive.")
            scale = float(reference_monitor_rate)
            units = "counts/s at reference monitor rate"
        if numerator_is_monitor and not np.array_equal(n, m, equal_nan=True):
            raise ValueError("A shared numerator/monitor must reference the same observations.")

    valid = reasons == ""
    values = np.full_like(n, np.nan)
    values[valid] = scale * n[valid] / denominator[valid]
    sigma = None
    lower = upper = None
    method = "unavailable (no supplied uncertainty)"
    if var is not None:
        sigma = np.full_like(n, np.nan)
        v = scale**2 * var[valid] / denominator[valid]**2
        if m is not None:
            if numerator_is_monitor:
                v = np.zeros_like(v)
            else:
                v += scale**2 * n[valid]**2 * vm[valid] / m[valid]**4
        sigma[valid] = np.sqrt(v)
        lower = np.full_like(n, np.nan)
        upper = np.full_like(n, np.nan)
        if numerator_is_monitor and m is not None:
            lower[valid] = upper[valid] = scale
            method = "shared monitor cancels exactly"
        elif uncertainty == "exact" and poisson and monitor_variance is None:
            if m is None:
                lo, hi = poisson_interval(n[valid], confidence)
                lower[valid] = lo / denominator[valid]
                upper[valid] = hi / denominator[valid]
                method = f"Garwood exact {100*confidence:.10g}% Poisson interval ({sigma_level:g} sigma equivalent)"
            else:
                lo, hi = poisson_ratio_interval(n[valid], m[valid], confidence)
                lower[valid] = scale * lo
                upper[valid] = scale * hi
                method = f"conditional exact {100*confidence:.10g}% Poisson ratio interval ({sigma_level:g} sigma equivalent)"
        else:
            lower[valid] = values[valid] - sigma_level * sigma[valid]
            upper[valid] = values[valid] + sigma_level * sigma[valid]
            method = f"propagated standard uncertainty (first order); {sigma_level:g} sigma bounds; standard uncertainty reported at 1 sigma"
    # Overflow cannot become an apparently valid plotted observation.
    reject(~np.isfinite(values), "nonfinite normalized value")
    if sigma is not None:
        reject(~np.isfinite(sigma), "nonfinite propagated uncertainty")
    if lower is not None:
        reject(~np.isfinite(lower) | ~np.isfinite(upper), "nonfinite interval")
    valid = reasons == ""
    values[~valid] = np.nan
    for arr in (sigma, lower, upper):
        if arr is not None:
            arr[~valid] = np.nan
    return NormalizationResult(values, sigma, lower, upper, valid, reasons, method, units)


def log_error_bounds(values, lower, upper):
    """Map positive finite interval endpoints to log10 distances.

    A zero/negative endpoint has no finite log coordinate. Return NaN for
    that side so it is omitted, never clipped to an invented positive value.
    """
    y = np.asarray(values, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    top = np.full_like(y, np.nan)
    bottom = np.full_like(y, np.nan)
    positive = np.isfinite(y) & (y > 0)
    top_ok = positive & np.isfinite(hi) & (hi > 0)
    bottom_ok = positive & np.isfinite(lo) & (lo > 0)
    top[top_ok] = np.log10(hi[top_ok]) - np.log10(y[top_ok])
    bottom[bottom_ok] = np.log10(y[bottom_ok]) - np.log10(lo[bottom_ok])
    return top, bottom


def normalize_map(
    counts: np.ndarray,
    frame_indices: np.ndarray,
    mode: str = "raw",
    *,
    exposure: np.ndarray | None = None,
    monitor: np.ndarray | None = None,
    monitor_variance: np.ndarray | None = None,
) -> NormalizedMap:
    """Normalize selected rows of an integrated map.

    raw:     N,      var = N
    time:    N / t,  var = N / t**2           (t known exposure in seconds)
    monitor: N / M,  var = N / M**2 + N**2 * v_M / M**4  (v_M = M unless supplied)

    Rows with missing, zero, negative or nonfinite denominators become NaN
    and are reported in ``warnings``; nothing is clipped or substituted.
    """
    if mode not in NORMALIZATION_UNITS:
        raise ValueError(f"Unknown normalization mode: {mode}")
    idx = np.asarray(frame_indices, dtype=int)
    n = np.asarray(counts, dtype=float)[idx]
    warnings: list[str] = []
    variance = n.copy()

    if mode == "raw":
        values = n.copy()
    else:
        source = exposure if mode == "time" else monitor
        name = "exposure time" if mode == "time" else "Monitor1"
        if source is None:
            raise ValueError(f"{name} is required for {mode} normalization")
        source = np.asarray(source, dtype=float)
        if source.shape[0] < (idx.max() + 1 if idx.size else 0):
            raise ValueError(f"{name} has fewer points than the detector has frames")
        denom = source[idx][:, None]
        valid = np.isfinite(denom) & (denom > 0)
        safe = np.where(valid, denom, np.nan)
        if mode == "time":
            values = n / safe
            variance = n / safe**2
        else:
            v_m = safe
            if monitor_variance is not None:
                v_m = np.asarray(monitor_variance, dtype=float)[idx][:, None]
                bad_var = ~(np.isfinite(v_m) & (v_m >= 0))
                valid = valid & ~bad_var
                v_m = np.where(bad_var, np.nan, v_m)
            values = np.where(valid, n / safe, np.nan)
            variance = n / safe**2 + n**2 * v_m / safe**4
        bad_rows = int(np.count_nonzero(~valid[:, 0]))
        if bad_rows:
            warnings.append(f"{bad_rows} point(s) excluded: invalid {name} or its variance.")

    return NormalizedMap(
        frame_indices=idx,
        values=values,
        standard_uncertainty=np.sqrt(variance),
        mode=mode,
        units=NORMALIZATION_UNITS[mode],
        warnings=warnings,
    )


__all__ = [
    "CONFIDENCE",
    "NORMALIZATION_UNITS",
    "NormalizationResult",
    "NormalizedMap",
    "confidence_from_sigma",
    "is_count_channel",
    "log_error_bounds",
    "normalize_map",
    "normalize_signal",
    "poisson_interval",
    "poisson_ratio_interval",
]
