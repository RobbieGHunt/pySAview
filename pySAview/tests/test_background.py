"""Unit tests for pySAview.core.background following pySAred 1.5.3."""

import numpy as np
import pytest

from pySAview.core.reduction.background import (
    bkg_roi_from_signal,
    default_bkg_x_bounds,
    resolve_bkg_roi,
    roi_area_pixels,
    roi_area_ratio,
    rois_overlap_2d,
    rois_overlap_x,
    subtract_background_pysared,
)


def test_default_bkg_x_bounds_matches_pysared_153():
    # Signal ROI [354, 382] (width 28) -> [354 - 56, 354 - 28] = [298, 326]
    assert default_bkg_x_bounds(354, 382, nx=1024) == (298.0, 326.0)
    assert not rois_overlap_x((354, 382), (298, 326), nx=1024)


def test_default_bkg_falls_back_to_right_near_left_edge():
    bx0, bx1 = default_bkg_x_bounds(10, 30, nx=1024)
    assert (bx0, bx1) == (50.0, 70.0)
    assert not rois_overlap_x((10, 30), (bx0, bx1), nx=1024)


def test_bkg_roi_shares_signal_y_bounds():
    assert bkg_roi_from_signal((354.0, 382.0, 285.4, 742.9), None, 1024, 1024) == (298, 326, 285, 743)
    assert bkg_roi_from_signal((354.0, 382.0, 285.4, 742.9), (300.0, 328.0), 1024, 1024) == (300, 328, 285, 743)


def test_subtract_background_matches_pysared_153():
    intens = np.array([100.0, 50.0, 10.0, 5.0])
    err = np.sqrt(intens)
    bkg = np.array([10.0, 20.0, 0.0, 12.0])
    qz = np.array([0.01, 0.03, 0.05, 0.08])

    net, net_err = subtract_background_pysared(intens, err, bkg, qz, bkg_skip_qz=0.02)
    # Point 0: Qz <= 0.02 -> skipped
    assert net[0] == 100.0 and net_err[0] == 10.0
    # Point 1: Qz > 0.02 and bkg > 0 -> subtracted
    assert net[1] == 30.0 and net_err[1] == pytest.approx(np.sqrt(50.0 + 20.0))
    # Point 2: bkg == 0 -> unchanged (matches pySAred `if Intens_bkg > 0:`)
    assert net[2] == 10.0 and net_err[2] == pytest.approx(np.sqrt(10.0))
    # Point 3: net can be negative, err = sqrt(5 + 12)
    assert net[3] == -7.0 and net_err[3] == pytest.approx(np.sqrt(17.0))


def test_resolve_bkg_roi_and_area_ratio():
    sig = (354.0, 382.0, 285.0, 743.0)  # 28 x 458 px
    # Explicit 4-tuple overrides legacy bkg_x_bounds
    bkg_full = resolve_bkg_roi(sig, (250.0, 306.0, 285.0, 743.0), (298.0, 326.0), 1024, 1024)
    assert bkg_full == (250, 306, 285, 743)
    assert roi_area_pixels(sig, 1024, 1024) == 28 * 458
    assert roi_area_pixels(bkg_full, 1024, 1024) == 56 * 458
    # BKG is twice as wide -> k = A_sig / A_bkg = 0.5
    assert roi_area_ratio(sig, bkg_full, 1024, 1024) == pytest.approx(0.5)

    # Half the height -> k = 2.0
    bkg_short = resolve_bkg_roi(sig, (298.0, 326.0, 285.0, 514.0), None, 1024, 1024)
    assert roi_area_ratio(sig, bkg_short, 1024, 1024) == pytest.approx(2.0)


def test_rois_overlap_2d_respects_both_axes():
    sig = (354.0, 382.0, 285.0, 743.0)
    # Same X, disjoint Y -> no 2D overlap
    assert not rois_overlap_2d(sig, (354.0, 382.0, 50.0, 200.0), 1024, 1024)
    # Disjoint X, same Y -> no 2D overlap
    assert not rois_overlap_2d(sig, (298.0, 326.0, 285.0, 743.0), 1024, 1024)
    # Overlapping in both X and Y -> overlap
    assert rois_overlap_2d(sig, (360.0, 400.0, 300.0, 500.0), 1024, 1024)


def test_subtract_background_area_ratio_scales_mean_and_variance():
    intens = np.array([100.0, 50.0, 10.0])
    err = np.sqrt(intens)
    bkg = np.array([40.0, 20.0, 0.0])
    qz = np.array([0.01, 0.05, 0.08])

    # Double-area BKG ROI -> k = 0.5 -> subtract 0.5 * bkg, variance = I + 0.25 * bkg
    net, net_err = subtract_background_pysared(intens, err, bkg, qz, bkg_skip_qz=0.0, area_ratio=0.5)
    np.testing.assert_allclose(net, [80.0, 40.0, 10.0])
    np.testing.assert_allclose(net_err, [np.sqrt(100.0 + 0.25 * 40.0), np.sqrt(50.0 + 0.25 * 20.0), np.sqrt(10.0)])

    # Half-area BKG ROI -> k = 2.0 -> subtract 2.0 * bkg, variance = I + 4.0 * bkg
    net2, net_err2 = subtract_background_pysared(intens, err, bkg, qz, bkg_skip_qz=0.0, area_ratio=2.0)
    np.testing.assert_allclose(net2, [20.0, 10.0, 10.0])
    np.testing.assert_allclose(net_err2, [np.sqrt(100.0 + 4.0 * 40.0), np.sqrt(50.0 + 4.0 * 20.0), np.sqrt(10.0)])
