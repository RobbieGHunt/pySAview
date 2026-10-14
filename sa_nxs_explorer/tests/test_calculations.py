"""Unit tests for scan calculations, Q conversion, normalization, and ROI integration."""

import numpy as np
import pytest

from sa_nxs_explorer.core.scan_calculations import (
    calculate_curve_stats,
    compute_normalized_y,
    compute_q,
    integrate_roi_2d,
    split_by_flipper,
)


def test_compute_q():
    # At theta = 0, Q = 0
    q_zero = compute_q(np.array([0.0]), wavelength_angstrom=5.21)
    assert q_zero[0] == pytest.approx(0.0)

    # At theta = 30 deg, sin(30) = 0.5 -> Q = 4*pi / 5.21 * 0.5 = 2*pi / 5.21
    q_30 = compute_q(np.array([30.0]), wavelength_angstrom=5.21)
    expected = (4.0 * np.pi / 5.21) * 0.5
    assert q_30[0] == pytest.approx(expected, rel=1e-4)

    # Invalid wavelength
    with pytest.raises(ValueError):
        compute_q(np.array([1.0]), wavelength_angstrom=-1.0)


def test_compute_normalized_y_raw():
    y = np.array([100.0, 400.0, 900.0])
    y_norm, y_err = compute_normalized_y(y, norm_mode="raw")
    np.testing.assert_array_equal(y_norm, y)
    # Poisson error: sqrt(100)=10, sqrt(400)=20, sqrt(900)=30
    np.testing.assert_allclose(y_err, [10.0, 20.0, 30.0])


def test_compute_normalized_y_monitor():
    y = np.array([100.0, 200.0])
    mon = np.array([1000.0, 1000.0])
    y_norm, y_err = compute_normalized_y(y, norm_mode="monitor", monitor_data=mon)
    # Monitor normalization is the unscaled ratio, shared across scans.
    np.testing.assert_allclose(y_norm, y / mon)
    assert len(y_err) == 2
    assert np.all(y_err > 0)


def test_compute_normalized_y_time():
    y = np.array([100.0, 200.0])
    t = np.array([10.0, 20.0])
    y_norm, y_err = compute_normalized_y(y, norm_mode="time", time_data=t)
    np.testing.assert_allclose(y_norm, [10.0, 10.0])
    # dy/t = 10 / 10 = 1.0; sqrt(200)/20 = 14.14 / 20 = 0.707
    assert y_err[0] == pytest.approx(1.0)
    assert y_err[1] == pytest.approx(np.sqrt(200.0) / 20.0)


def test_compute_normalized_y_monitor_time():
    y = np.array([100.0, 200.0])
    mon = np.array([1000.0, 2000.0])
    t = np.array([10.0, 20.0])
    y_norm, y_err = compute_normalized_y(y, norm_mode="monitor_time", monitor_data=mon, time_data=t, reference_monitor_rate=100)
    # The fixed reference monitor rate is 100 monitor counts/s.
    # (100 / 1000) * 100 = 10.0; (200 / 2000) * 100 = 10.0
    np.testing.assert_allclose(y_norm, [10.0, 10.0])
    assert y_err[0] == pytest.approx(10.0 * np.sqrt((10.0 / 100.0)**2 + (np.sqrt(1000.0) / 1000.0)**2), rel=1e-3)
    assert y_err[1] > 0

    # Missing sources are never silently replaced with another normalization.
    with pytest.raises(ValueError, match="Monitor counts is required"):
        compute_normalized_y(y, norm_mode="monitor_time", monitor_data=None, time_data=t)


def test_split_by_flipper():
    x = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    y_err = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flipper = np.array([0, 1, 0, 1, 0, 1])

    # Split mode
    curves = split_by_flipper(x, y, y_err, flipper, spin_mode="split")
    assert len(curves) == 2
    c_up = curves[0]
    c_down = curves[1]
    assert "Flipper OFF / 0" in c_up.label
    assert "Flipper ON / 1" in c_down.label
    np.testing.assert_array_equal(c_up.x, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(c_up.y, [10.0, 30.0, 50.0])
    np.testing.assert_array_equal(c_down.y, [20.0, 40.0, 60.0])

    # Combined mode
    curves_all = split_by_flipper(x, y, y_err, flipper, spin_mode="all")
    assert len(curves_all) == 1
    assert len(curves_all[0].x) == 6


def test_calculate_curve_stats():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 100.0, 40.0, 5.0])
    st = calculate_curve_stats(x, y)
    assert st.n_points == 5
    assert st.peak_x == 3.0
    assert st.peak_y == 100.0
    assert st.y_min == 5.0
    assert st.y_max == 100.0
    assert st.y_sum == pytest.approx(175.0)

    # Empty stats
    st_empty = calculate_curve_stats(np.array([]), np.array([]))
    assert st_empty.n_points == 0


def test_integrate_roi_2d():
    frame = np.ones((100, 100))
    roi = (10.0, 20.0, 30.0, 50.0)  # width 10, height 20
    total, x_prof, y_prof = integrate_roi_2d(frame, roi)
    assert total == pytest.approx(200.0)
    assert len(x_prof) == 10
    assert len(y_prof) == 20
    np.testing.assert_allclose(x_prof, 20.0)  # each x summed across 20 y-pixels
    np.testing.assert_allclose(y_prof, 10.0)  # each y summed across 10 x-pixels


def test_find_reflection_peak_center():
    from sa_nxs_explorer.core.scan_calculations import find_reflection_peak_center

    # 1. Distinct peak at x = 370
    prof = np.zeros(500)
    prof[368:373] = [20, 80, 150, 80, 20]
    center = find_reflection_peak_center(prof, prev_center=368.0, search_half_width=15)
    assert center == pytest.approx(370.0, abs=0.2)

    # 2. Flat / low intensity profile -> keeps prev_center
    prof_flat = np.ones(500) * 2.0
    center_flat = find_reflection_peak_center(prof_flat, prev_center=368.0, search_half_width=15)
    assert center_flat == 368.0

    # 3. Clamped jump -> cannot jump more than max_step_change (4.0)
    prof_far = np.zeros(500)
    prof_far[385] = 200.0  # peak at 385, prev is 368 -> jump is +17
    center_clamped = find_reflection_peak_center(prof_far, prev_center=368.0, search_half_width=20, max_step_change=4.0)
    assert center_clamped == pytest.approx(372.0)


def test_auto_fit_from_profiles():
    from sa_nxs_explorer.core.scan_calculations import auto_fit_from_profiles

    # 10 frames: frames 0..2 are skipped (stay at default 100), frames 3..9 shift to 103
    profiles = np.zeros((10, 200))
    for i in range(10):
        peak = 100 if i < 3 else 103
        profiles[i, peak] = 100.0

    initial_roi = (90.0, 110.0, 20.0, 80.0)  # center = 100, width = 20
    rois = auto_fit_from_profiles(profiles, initial_roi=initial_roi, skip_first_n=3)

    assert len(rois) == 10
    # First 3 frames stayed at 100
    for r in rois[:3]:
        assert (r[0] + r[1]) / 2.0 == pytest.approx(100.0)
        assert r[1] - r[0] == pytest.approx(20.0)

    # Subsequent frames moved towards 103
    last_center = (rois[-1][0] + rois[-1][1]) / 2.0
    assert last_center == pytest.approx(103.0, abs=0.2)


def test_auto_fit_scan_rois_3d():
    from sa_nxs_explorer.core.scan_calculations import auto_fit_scan_rois

    # 3D dataset: (N=8, nx=100, ny=100)
    vol = np.zeros((8, 100, 100))
    for i in range(8):
        # Beam at x=50, y from 30..70
        vol[i, 50, 30:70] = 50.0

    initial_roi = (45.0, 55.0, 20.0, 80.0)
    rois, counts = auto_fit_scan_rois(vol, total_frames=8, initial_roi=initial_roi, skip_first_n=2)

    assert len(rois) == 8
    assert len(counts) == 8
    assert np.all(counts > 0)
    for r in rois:
        assert (r[0] + r[1]) / 2.0 == pytest.approx(50.0, abs=0.5)

