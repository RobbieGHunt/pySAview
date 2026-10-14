"""Count normalization with explicit statistical models and invalid-point masks.

Exact intervals assume unweighted, independent Poisson counts. Propagated
standard uncertainties use first-order derivatives, not confidence intervals.
See docs/normalization.md for formulas and the limits of these assumptions.
"""

from dataclasses import dataclass
import numpy as np
from scipy.stats import beta, chi2

CONFIDENCE = 0.6826894921370859


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
    return (name in {"ROI_Counts", "Detector1", "Monitor1", "Monitor2"}
            or unit.strip().lower() in {"count", "counts", "cts"})


def poisson_interval(counts, confidence=CONFIDENCE):
    """Equal-tailed Garwood confidence limits on a Poisson mean."""
    n = _vector(counts, "Counts")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one.")
    if np.any(~np.isfinite(n) | (n < 0) | (n != np.rint(n))):
        raise ValueError("Poisson intervals require finite, nonnegative integer counts.")
    alpha = 1 - confidence
    lower = np.zeros_like(n)
    positive = n > 0
    lower[positive] = chi2.ppf(alpha / 2, 2 * n[positive]) / 2
    upper = chi2.isf(alpha / 2, 2 * (n + 1)) / 2
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
    alpha = 1 - confidence
    lower = np.zeros_like(n)
    positive = n > 0
    p_low = beta.ppf(alpha / 2, n[positive], m[positive] + 1)
    lower[positive] = p_low / (1 - p_low)
    # Compute the upper odds from the lower complementary probability. This
    # avoids subtracting an upper quantile extremely close to one.
    q_low = beta.ppf(alpha / 2, m, n + 1)
    upper = (1 - q_low) / q_low
    return lower, upper


def normalize_signal(data, mode="raw", *, monitor=None, time=None,
                     count_data=True, variance=None, monitor_variance=None,
                     uncertainty="exact", reference_monitor_rate=1.0,
                     numerator_is_monitor=False, input_units="counts"):
    """Prepare values, uncertainty bounds, and reasons for excluded points.

    raw: N; time: N/t; monitor: N/M; monitor_time: fixed_rate*N/M.
    The fixed rate is an exact display convention, shared across all scans.
    No mode falls back and no denominator is replaced by a fabricated value.
    """
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
                lo, hi = poisson_interval(n[valid])
                lower[valid] = lo / denominator[valid]
                upper[valid] = hi / denominator[valid]
                method = "Garwood exact 68.27% Poisson interval"
            else:
                lo, hi = poisson_ratio_interval(n[valid], m[valid])
                lower[valid] = scale * lo
                upper[valid] = scale * hi
                method = "conditional exact 68.27% Poisson ratio interval"
        else:
            lower[valid] = values[valid] - sigma[valid]
            upper[valid] = values[valid] + sigma[valid]
            method = "propagated standard uncertainty (first order)"
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
