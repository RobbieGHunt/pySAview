"""Unit tests for pySAview.core.curvature cross-checked against pySAred 1.5.3."""

import numpy as np
import pytest
from scipy.interpolate import griddata

from pySAview.core.reduction.curvature import (
    apply_curvature_by_spin,
    apply_curvature_to_map,
    apply_curvature_to_slice,
)


def _verbatim_pysared_153_curvature(roi_slice: np.ndarray, curv: float) -> np.ndarray:
    """Exact copy of pySAred_V1.5.3.py lines 1241-1257."""
    det = [[], [], []]
    for x, col in enumerate(np.flipud(np.rot90(roi_slice))):
        displacement = x * np.tan(curv)
        for y, value in enumerate(col):
            det[0].append(x)
            det[1].append(y + displacement)
            det[2].append(value)
    roi_mid = roi_slice.shape[1] / 2
    zero_level = int(round(roi_mid * np.tan(curv) - min(det[1])))
    grid_x, grid_y = np.mgrid[0:roi_slice.shape[1]:1, min(det[1]):max(det[1]):1]
    return np.flipud(
        np.rot90(
            griddata((det[0], det[1]), det[2], (grid_x, grid_y), method="linear", fill_value=0.0)
        )
    )[zero_level : zero_level + roi_slice.shape[0], :]


@pytest.mark.parametrize("curv", [-0.25, -0.08, -0.05, 0.05, 0.12, 0.25, 0.6])
def test_matches_verbatim_pysared_griddata(curv):
    rng = np.random.default_rng(123)
    roi_slice = rng.poisson(30, size=(60, 28)).astype(float)
    got = apply_curvature_to_slice(roi_slice, curv)
    ref = _verbatim_pysared_153_curvature(roi_slice, curv)
    assert got.shape == roi_slice.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-12, rtol=1e-12)


def test_handles_mgrid_boundary_curvature_without_shape_crash():
    # At curv=0.0365 and width=28, pySAred's np.mgrid truncates to 79 rows instead of 80.
    roi_slice = np.ones((80, 28), dtype=float)
    got = apply_curvature_to_slice(roi_slice, 0.0365)
    assert got.shape == (80, 28)
    assert np.all(np.isfinite(got))


def test_zero_curvature_is_exact_reversible_copy():
    raw = np.arange(200, dtype=float).reshape(20, 10)
    snapshot = raw.copy()
    out = apply_curvature_to_map(raw, 0.0, (2, 7))
    np.testing.assert_array_equal(out, raw)
    out[0, 0] = -999.0
    np.testing.assert_array_equal(raw, snapshot)

    warped = apply_curvature_to_map(raw, 0.15, (2, 7))
    np.testing.assert_array_equal(raw, snapshot)  # raw never mutated
    np.testing.assert_array_equal(warped[:, :2], raw[:, :2])  # outside ROI untouched
    np.testing.assert_array_equal(warped[:, 7:], raw[:, 7:])


def test_straightens_tilted_fringe_inside_roi():
    # Build a synthetic tilted ridge: peak at y = 30 - (x - 14) * tan(0.15)
    n, w = 80, 28
    gamma = 0.15
    y = np.arange(n, dtype=float)[:, None]
    x = np.arange(w, dtype=float)[None, :]
    ridge = np.exp(-0.5 * ((y - (30.0 - (x - 14.0) * np.tan(gamma))) / 1.5) ** 2)
    uncorrected_sum = ridge.sum(axis=1)
    corrected = apply_curvature_to_slice(ridge, gamma)
    corrected_sum = corrected.sum(axis=1)
    # Straightening the tilted ridge recovers almost the full 28-column peak height (~27 vs ~21)
    assert corrected_sum.max() > 26.0
    assert corrected_sum.max() > uncorrected_sum.max() * 1.25


def test_curvature_by_spin_isolates_interleaved_spin_states_and_unwritten_frames():
    # Interleaved frames: even = spin down (constant 100), odd = spin up (constant 1), tail unwritten
    counts = np.zeros((24, 16), dtype=float)
    flipper = np.tile([0.0, 1.0], 12)
    written = np.ones(24, dtype=bool)
    written[20:] = False
    counts[0:20:2, 4:10] = 100.0
    counts[1:20:2, 4:10] = 1.0
    counts[20:, 4:10] = 9999.0  # unwritten garbage must not leak into written frames

    out = apply_curvature_by_spin(counts, flipper, written, 0.12, (4, 10))
    # Spin-up interior rows must stay near 1.0, never polluted by spin-down 100.0 or unwritten 9999.0
    assert np.all(out[3:17:2, 4:10] <= 1.0 + 1e-12)
    assert np.all(out[3:17:2, 4:10] >= 0.0)
    # Spin-down interior rows stay at 100.0
    np.testing.assert_allclose(out[4:16:2, 5:9], 100.0)


def test_overlapping_signal_and_bkg_roi_rejected_when_curvature_nonzero():
    data = np.ones((20, 30), dtype=float)
    with pytest.raises(ValueError, match="overlap"):
        apply_curvature_to_map(data, 0.1, (10, 20), bkg_x_bounds=(15, 25))
    # Disjoint BKG ROI succeeds and is corrected independently
    out = apply_curvature_to_map(data, 0.1, (12, 20), bkg_x_bounds=(2, 10))
    assert out.shape == data.shape


@pytest.mark.parametrize("bad", [np.nan, np.inf, np.pi / 2, -np.pi / 2])
def test_invalid_curvature_rejected(bad):
    with pytest.raises(ValueError):
        apply_curvature_to_slice(np.ones((10, 5)), bad)
