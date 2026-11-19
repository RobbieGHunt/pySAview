"""Unit tests for pySAview.core.background following pySAred 1.5.3."""

import numpy as np
import pytest

from pySAview.core.reduction.background import (
    bkg_roi_from_signal,
    default_bkg_x_bounds,
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
