"""Automated pre-critical-edge (total external reflection plateau -> 1) scale estimator.

Below the critical momentum transfer ``Q_c``, an ideal neutron reflectivity
curve forms a total-reflection plateau at ``R(Q_z < Q_c) = 1``.  When no
direct-beam reference scan is available, dividing the overillumination-corrected
intensity by the pre-critical-edge plateau level normalizes the reflectivity
onto an absolute ``R = 1`` scale.

This standalone, pure-NumPy module estimates that plateau scale factor robustly
even when:

* Very-low-angle points contain a direct-beam / ``1/theta`` blow-up artifact
  after overillumination division.
* Overillumination correction was imperfect (wrong sample length or slight
  misalignment), leaving a gentle positive or negative slope across the
  pre-critical-edge plateau.
* A polarized scan has two spin channels (``u`` and ``d``) with different
  critical edges ``Q_c+ != Q_c-``: :func:`estimate_multi_curve_plateau_scale`
  uses the shared region below the lower critical edge so both spin channels
  receive the same physical beam scale factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "PlateauEstimate",
    "estimate_multi_curve_plateau_scale",
    "estimate_plateau_scale",
]


@dataclass(frozen=True)
class PlateauEstimate:
    """Result of an automated pre-critical-edge plateau normalization estimate."""

    scale_factor: float
    q_low: float
    q_high: float
    n_points: int
    relative_drift: float  # fractional change across [q_low, q_high], e.g. +0.25 = +25%
    scatter_rms: float  # robust fractional scatter around the linear log-trend
    is_sloped: bool
    note: str


def _rolling_median_3(y: np.ndarray) -> np.ndarray:
    """3-point median filter with edge clamping (pure NumPy)."""
    n = y.size
    if n < 3:
        return y.copy()
    padded = np.empty(n + 2, dtype=float)
    padded[0] = y[0]
    padded[1:-1] = y
    padded[-1] = y[-1]
    windows = np.stack((padded[:-2], padded[1:-1], padded[2:]), axis=1)
    return np.median(windows, axis=1)


def _theil_sen_log(q: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust Theil-Sen slope & intercept of ``ln(y) = b + s * q``."""
    log_y = np.log(y)
    n = q.size
    if n < 2:
        return 0.0, float(log_y[0])
    dq = q[None, :] - q[:, None]
    dy = log_y[None, :] - log_y[:, None]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1) & (np.abs(dq) > 1e-12)
    if not np.any(upper):
        return 0.0, float(np.median(log_y))
    slopes = dy[upper] / dq[upper]
    slope = float(np.median(slopes))
    intercept = float(np.median(log_y - slope * q))
    return slope, intercept


def _locate_pre_edge_window(
    q: np.ndarray,
    y: np.ndarray,
    *,
    q_max_search: float = 0.045,
) -> tuple[int, int, np.ndarray]:
    """Return ``(win_lo, win_hi, y_med)`` spanning the pre-critical-edge plateau."""
    n = q.size
    search_mask = q <= float(q_max_search)
    if int(np.count_nonzero(search_mask)) < 4:
        n_search = min(n, max(4, int(np.ceil(0.30 * n))))
        n_end = n_search
    else:
        n_end = int(np.flatnonzero(search_mask)[-1]) + 1

    qs = q[:n_end]
    ys = y[:n_end]
    y_med = _rolling_median_3(ys)

    # 1. Skip any near-zero direct-beam / 1/theta blow-up artifact (typically Q < 0.0045 A^-1).
    start_idx = 0
    low_q_limit = 0.0048
    low_idx = np.flatnonzero(qs <= low_q_limit)
    if low_idx.size >= 2 and qs[-1] > 0.006:
        # Look for a steep drop from the left edge followed by a trough and recovery onto the plateau
        last_low = int(low_idx[-1])
        trough_rel = int(np.argmin(y_med[: last_low + 1]))
        after_end = min(n_end, trough_rel + 12)
        if trough_rel < n_end - 2:
            post_peak = float(np.max(y_med[trough_rel + 1 : after_end]))
            if y_med[0] > 1.30 * y_med[trough_rel] and post_peak > 1.12 * y_med[trough_rel]:
                # Enter plateau once intensity climbs past the trough onto >= 85% of the post-trough level
                recovered = np.flatnonzero(y_med[trough_rel + 1 : after_end] >= 0.85 * post_peak)
                start_idx = trough_rel + 1 + (int(recovered[0]) if recovered.size else 0)
            else:
                # Monotonic 1/theta decay directly into a flat plateau (no dip): trim leading spike
                ref_mask = (qs >= 0.004) & (qs <= 0.012)
                if np.count_nonzero(ref_mask) >= 3:
                    ref_level = float(np.median(y_med[ref_mask]))
                    while start_idx < last_low and y_med[start_idx] > 1.30 * ref_level:
                        start_idx += 1

    if start_idx >= n_end - 1:
        start_idx = 0

    # 2. Find the plateau top and Fresnel critical-edge drop-off (y drops below 50% of peak).
    post = y_med[start_idx:]
    peak_idx = start_idx + int(np.argmax(post))
    y_peak = float(y_med[peak_idx])

    below_half = np.flatnonzero(y_med[peak_idx:] < 0.50 * y_peak)
    drop_idx = peak_idx + int(below_half[0]) if below_half.size else n_end

    # Trim the rounding shoulder at the critical edge (keep points >= 85% of local plateau peak
    # walking backward from drop_idx toward peak_idx).
    win_hi = drop_idx
    while win_hi > peak_idx + 1 and y_med[win_hi - 1] < 0.85 * y_peak:
        win_hi -= 1
    if win_hi <= start_idx:
        win_hi = min(n_end, start_idx + 1)

    # Trim any low-angle rise-in on the left if overillumination was missing/severely undercorrected
    # while keeping at least 4 points (when available) so we can detect the slope cleanly.
    win_lo = start_idx
    min_pts = min(4, win_hi - start_idx)
    while (win_hi - win_lo) > min_pts and y_med[win_lo] < 0.45 * y_peak:
        win_lo += 1

    # Also ensure we don't accidentally include a residual single-point left spike above 1.25x plateau median
    if (win_hi - win_lo) > min_pts:
        core_med = float(np.median(y_med[win_lo:win_hi]))
        while (win_hi - win_lo) > min_pts and y_med[win_lo] > 1.25 * core_med:
            win_lo += 1

    return win_lo, win_hi, y_med


def estimate_plateau_scale(
    q_z: np.ndarray,
    intensity: np.ndarray,
    *,
    q_max_search: float = 0.045,
    slope_tolerance: float = 0.12,
    q_upper_override: float | None = None,
) -> PlateauEstimate:
    """Estimate the pre-critical-edge plateau intensity to normalize ``R(Q < Q_c) -> 1``.

    Parameters
    ----------
    q_z : array-like
        Momentum transfer ``Qz`` in inverse Angstroms.
    intensity : array-like
        Reduced intensity BEFORE scale-factor division (after background
        subtraction, monitor/time normalization, and overillumination).
    q_max_search : float
        Upper ``Qz`` bound (Å⁻¹) searched for the critical edge (default ``0.045``).
    slope_tolerance : float
        Maximum fractional drift across the detected pre-edge plateau still
        treated as flat (default ``0.12``, i.e. 12%).  When drift exceeds this
        threshold (e.g. poor or missing overillumination correction), the scale
        is evaluated from a robust Theil-Sen log-linear trend anchored near the
        upper end (80th percentile of ``Q``) of the pre-edge plateau where
        footprint sensitivity is smallest.
    q_upper_override : float | None
        Optional hard upper ``Qz`` cap (used by :func:`estimate_multi_curve_plateau_scale`
        to stay below the lower critical edge ``Q_c-`` of a polarized scan).
    """
    q_arr = np.asarray(q_z, dtype=float)
    y_arr = np.asarray(intensity, dtype=float)
    if q_arr.ndim != 1 or y_arr.ndim != 1 or q_arr.size != y_arr.size:
        raise ValueError("q_z and intensity must be 1D arrays of equal length.")

    valid = np.isfinite(q_arr) & np.isfinite(y_arr) & (q_arr > 0.0) & (y_arr > 0.0)
    if q_upper_override is not None:
        valid &= q_arr <= float(q_upper_override)
    if not np.any(valid):
        raise ValueError("No positive finite (Qz, Intensity) points available to locate the critical-edge plateau.")

    order = np.argsort(q_arr[valid], kind="stable")
    q = q_arr[valid][order]
    y = y_arr[valid][order]

    if q.size == 1:
        val = float(y[0])
        return PlateauEstimate(
            scale_factor=val,
            q_low=float(q[0]),
            q_high=float(q[0]),
            n_points=1,
            relative_drift=0.0,
            scatter_rms=0.0,
            is_sloped=False,
            note=f"Plateau: {val:.4g} (1 point at Q={q[0]:.4f} Å⁻¹)",
        )

    win_lo, win_hi, _ = _locate_pre_edge_window(q, y, q_max_search=q_max_search)
    qw = q[win_lo:win_hi]
    yw = y[win_lo:win_hi]

    slope, intercept = _theil_sen_log(qw, yw)
    span_q = float(qw[-1] - qw[0])
    rel_drift = float(np.expm1(slope * span_q)) if span_q > 0 else 0.0
    fitted = np.exp(intercept + slope * qw)
    scatter = float(np.median(np.abs(yw / fitted - 1.0)) * 1.4826) if yw.size >= 2 else 0.0

    is_sloped = abs(rel_drift) > float(slope_tolerance)
    if is_sloped and qw.size >= 3:
        # Anchor near the upper-Q side of the pre-edge window (80% across [q_lo, q_hi]),
        # where c0 is largest and overillumination error is smallest.
        q_anchor = float(qw[0] + 0.80 * span_q)
        raw_est = float(np.exp(intercept + slope * q_anchor))
        scale = float(np.clip(raw_est, float(np.min(yw)), float(np.max(yw))))
        sign_str = "+" if rel_drift >= 0 else ""
        note = (
            f"Plateau: {scale:.4g} (Q {qw[0]:.4f}–{qw[-1]:.4f} Å⁻¹, {yw.size} pts; "
            f"sloped {sign_str}{rel_drift * 100:.0f}% — check overillumination)"
        )
    else:
        scale = float(np.median(yw))
        note = (
            f"Plateau: {scale:.4g} (Q {qw[0]:.4f}–{qw[-1]:.4f} Å⁻¹, {yw.size} pts, "
            f"flat ±{scatter * 100:.1f}%)"
        )

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Could not determine a positive pre-critical-edge scale factor.")

    return PlateauEstimate(
        scale_factor=scale,
        q_low=float(qw[0]),
        q_high=float(qw[-1]),
        n_points=int(yw.size),
        relative_drift=rel_drift,
        scatter_rms=scatter,
        is_sloped=is_sloped,
        note=note,
    )


def estimate_multi_curve_plateau_scale(
    curves: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    q_max_search: float = 0.045,
    slope_tolerance: float = 0.12,
) -> PlateauEstimate:
    """Estimate one shared physical scale factor across spin states of a scan.

    In polarized reflectometry, ``u`` and ``d`` spin channels can have different
    critical edges (``Q_c+ != Q_c-``), but both equal ``R = 1`` below the lower
    critical edge ``min(Q_c)``.  When multiple curves are supplied, each curve's
    pre-edge region is located first and the final scale is estimated below the
    shared upper cutoff ``min(q_high)`` so spin splitting is preserved.
    """
    valid_curves = [
        (np.asarray(q, dtype=float), np.asarray(y, dtype=float))
        for q, y in curves
        if len(q) > 0 and len(y) == len(q)
    ]
    if not valid_curves:
        raise ValueError("No valid curves available for plateau normalization.")
    if len(valid_curves) == 1:
        return estimate_plateau_scale(
            valid_curves[0][0],
            valid_curves[0][1],
            q_max_search=q_max_search,
            slope_tolerance=slope_tolerance,
        )

    per_curve: list[PlateauEstimate] = []
    for q, y in valid_curves:
        try:
            per_curve.append(
                estimate_plateau_scale(q, y, q_max_search=q_max_search, slope_tolerance=slope_tolerance)
            )
        except ValueError:
            continue
    if not per_curve:
        raise ValueError("Could not locate a pre-critical-edge plateau on any spin channel.")

    # Use the lower critical edge so both spin channels sit on R = 1
    shared_q_hi = min(est.q_high for est in per_curve)
    shared_q_lo = max(est.q_low for est in per_curve)
    if shared_q_hi > shared_q_lo:
        q_all = np.concatenate([q for q, _ in valid_curves])
        y_all = np.concatenate([y for _, y in valid_curves])
        try:
            return estimate_plateau_scale(
                q_all,
                y_all,
                q_max_search=q_max_search,
                slope_tolerance=slope_tolerance,
                q_upper_override=shared_q_hi,
            )
        except ValueError:
            pass

    # Fallback: median of per-curve plateau scales
    scales = np.array([e.scale_factor for e in per_curve], dtype=float)
    scale = float(np.median(scales))
    q_lo = float(min(e.q_low for e in per_curve))
    q_hi = float(min(e.q_high for e in per_curve))
    drift = float(np.median([e.relative_drift for e in per_curve]))
    scatter = float(np.median([e.scatter_rms for e in per_curve]))
    sloped = any(e.is_sloped for e in per_curve)
    note = (
        f"Plateau: {scale:.4g} (Q {q_lo:.4f}–{q_hi:.4f} Å⁻¹, {sum(e.n_points for e in per_curve)} pts"
        + (f"; sloped {drift * 100:+.0f}% — check overillumination)" if sloped else f", flat ±{scatter * 100:.1f}%)")
    )
    return PlateauEstimate(
        scale_factor=scale,
        q_low=q_lo,
        q_high=q_hi,
        n_points=sum(e.n_points for e in per_curve),
        relative_drift=drift,
        scatter_rms=scatter,
        is_sloped=sloped,
        note=note,
    )
