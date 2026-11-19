"""Unit tests for automated pre-critical-edge plateau normalization (R -> 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pySAview.core.nexus_reader import NexusReader
from pySAview.core.reduction.overillumination import extract_slit_widths
from pySAview.core.reduction.plateau import (
    PlateauEstimate,
    estimate_multi_curve_plateau_scale,
    estimate_plateau_scale,
)
from pySAview.core.reduction.pysared import ReductionParams, reduce_counts_pysared
from pySAview.core.scan_calculations import compute_q

EXAMPLE_5572 = Path(__file__).resolve().parent.parent / "example_data" / "005572.nxs"
EXAMPLE_5594 = Path(__file__).resolve().parent.parent / "example_data" / "005594.nxs"


def _synthetic_reflectivity(
    q: np.ndarray,
    qc: float,
    scale: float = 5.0,
    slope_frac: float = 0.0,
) -> np.ndarray:
    """Fresnel-like reflectivity curve with optional pre-edge slope (imperfect overillumination)."""
    # Exact Fresnel: R = 1 for Q <= Qc, drops steeply for Q > Qc
    r = np.ones_like(q, dtype=float)
    above = q > qc
    u = q[above] / qc
    r[above] = ((u - np.sqrt(np.maximum(u * u - 1.0, 0.0))) / (u + np.sqrt(np.maximum(u * u - 1.0, 0.0)))) ** 2
    if slope_frac != 0.0:
        # Ramp linearly in log-intensity across the pre-edge region up to Qc
        ramp = np.exp(np.log(1.0 + slope_frac) * np.minimum(q / qc, 1.0))
        # Normalize ramp so value right at Qc is `scale`
        ramp = ramp / np.exp(np.log(1.0 + slope_frac))
        r = r * ramp
    return scale * r


def test_flat_plateau_recovers_exact_scale():
    q = np.linspace(0.002, 0.08, 160)
    y = _synthetic_reflectivity(q, qc=0.015, scale=7.25)
    est = estimate_plateau_scale(q, y)
    assert isinstance(est, PlateauEstimate)
    assert est.scale_factor == pytest.approx(7.25, rel=1e-2)
    assert est.is_sloped is False
    assert est.q_high <= 0.0155
    assert "flat" in est.note


def test_direct_beam_blowup_with_trough_is_skipped():
    q = np.linspace(0.001, 0.08, 160)
    y = _synthetic_reflectivity(q, qc=0.014, scale=4.8)
    # Inject strong low-Q 1/theta blow-up + small trough before the true plateau
    y[q < 0.0025] = 80.0
    y[(q >= 0.0025) & (q < 0.0033)] = 3.1  # trough
    est = estimate_plateau_scale(q, y)
    assert est.scale_factor == pytest.approx(4.8, rel=0.01)
    assert est.q_low >= 0.0025
    assert est.is_sloped is False


def test_monotonic_low_q_spike_without_trough_is_trimmed():
    q = np.linspace(0.001, 0.08, 160)
    y = _synthetic_reflectivity(q, qc=0.014, scale=3.5)
    # Monotonic decay from 30 down into flat 3.5 plateau
    spike = q < 0.0035
    y[spike] = 3.5 * (0.0035 / q[spike]) ** 2
    est = estimate_plateau_scale(q, y)
    assert est.scale_factor == pytest.approx(3.5, rel=0.03)
    assert est.q_low >= 0.0030


def test_imperfect_overillumination_positive_slope_anchors_near_edge():
    q = np.linspace(0.002, 0.08, 160)
    # Rises by +60% across the pre-edge region (under-corrected overillumination), peaking at 6.0 at Qc
    y = _synthetic_reflectivity(q, qc=0.015, scale=6.0, slope_frac=0.60)
    est = estimate_plateau_scale(q, y)
    assert est.is_sloped is True
    assert est.relative_drift > 0.12
    assert "sloped" in est.note and "overillumination" in est.note
    # Must anchor near the upper end (~6.0), NOT underestimate at the low-Q midpoint (~4.6)
    assert 5.4 <= est.scale_factor <= 6.2


def test_imperfect_overillumination_negative_slope_is_robust():
    q = np.linspace(0.002, 0.08, 160)
    # Falls by -30% across the pre-edge region (over-corrected sample length too small)
    y = _synthetic_reflectivity(q, qc=0.015, scale=5.0, slope_frac=-0.30)
    est = estimate_plateau_scale(q, y)
    assert est.is_sloped is True
    assert est.relative_drift < -0.12
    assert np.isfinite(est.scale_factor) and est.scale_factor > 0.0


def test_multi_curve_pnr_uses_shared_window_below_lower_qc():
    q = np.linspace(0.002, 0.08, 160)
    y_down = _synthetic_reflectivity(q, qc=0.010, scale=4.2)
    y_up = _synthetic_reflectivity(q, qc=0.017, scale=4.2)
    est = estimate_multi_curve_plateau_scale([(q, y_down), (q, y_up)])
    assert est.scale_factor == pytest.approx(4.2, rel=0.02)
    assert est.q_high <= 0.011
    assert est.is_sloped is False


def test_insufficient_points_raises():
    with pytest.raises(ValueError, match="No positive finite"):
        estimate_plateau_scale(np.array([0.005, 0.01]), np.array([0.0, -1.0]))
    with pytest.raises(ValueError, match="No valid curves"):
        estimate_multi_curve_plateau_scale([])


@pytest.mark.parametrize("example_path", [EXAMPLE_5572, EXAMPLE_5594])
def test_real_superadam_scans_normalize_plateau_to_unity(example_path: Path):
    with NexusReader(example_path) as reader:
        scan = reader.parse_scan()
    th = np.asarray(scan.variables["th"].data, dtype=float)
    q = compute_q(th, 5.21)
    y = np.asarray(scan.variables["ROI_Counts"].data, dtype=float)
    mon = np.asarray(scan.variables["Monitor1"].data, dtype=float)
    flip = np.asarray(scan.variables["Flipper1"].data, dtype=float)
    s1, s2, _ = extract_slit_widths(scan, len(y))

    # With overillumination (L = 50 mm): plateau should be flat and normalize to ~1.0
    p_ovl = ReductionParams(overillumination=True, sample_length_mm=50.0)
    res_ovl = reduce_counts_pysared(
        y, "monitor", monitor=mon, theta_deg=th, wavelength_angstrom=5.21,
        s1_mm=s1, s2_mm=s2, params=p_ovl,
    )
    curves = []
    for st in (0, 1):
        m = res_ovl.valid & (flip == st) & (res_ovl.values > 0)
        curves.append((q[m], res_ovl.values[m]))
    est = estimate_multi_curve_plateau_scale(curves)
    assert est.is_sloped is False
    for cq, cy in curves:
        in_plat = (cq >= 0.004) & (cq <= 0.010)
        med_r = float(np.median(cy[in_plat] / est.scale_factor))
        assert med_r == pytest.approx(1.0, abs=0.03)

    # Without overillumination: estimator should detect strong positive slope
    p_raw = ReductionParams(overillumination=False)
    res_raw = reduce_counts_pysared(
        y, "monitor", monitor=mon, theta_deg=th, wavelength_angstrom=5.21,
        s1_mm=s1, s2_mm=s2, params=p_raw,
    )
    raw_curves = [
        (q[res_raw.valid & (flip == st) & (res_raw.values > 0)],
         res_raw.values[res_raw.valid & (flip == st) & (res_raw.values > 0)])
        for st in (0, 1)
    ]
    est_raw = estimate_multi_curve_plateau_scale(raw_curves)
    assert est_raw.is_sloped is True
    assert est_raw.relative_drift > 0.25
