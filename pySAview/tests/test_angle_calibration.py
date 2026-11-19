"""Direct-beam angle calibration and the α_i / α_f view of the Off-Specular tab."""

from pathlib import Path

import numpy as np
import pytest

from pySAview.core.angle_calibration import (
    DETECTOR_WIDTH_MM,
    PYSARED_DEFAULT_DISTANCE_MM,
    AngleCalibration,
    CalibrationError,
    alpha_angles,
    alpha_f_offset_for_pixel,
    alpha_f_offsets,
    alpha_i_strip_edges,
    angles_from_q,
    calibrate,
    effective_deg_per_pixel,
    fit_direct_beam,
    implied_distance_mm,
    pixel_for_alpha_f_offset,
    pixel_pitch_mm,
    q_from_angles,
)
from pySAview.ui.main_window import MainWindow

DATA = Path(__file__).resolve().parent.parent / "example_data"


def _synthetic(slope=120.0, x0=400.0, intercept=395.0, n=60, nx=1024, outliers=()):
    rng = np.random.default_rng(7)
    tth = np.linspace(0.0, 5.0, n)
    counts = rng.poisson(1.0, size=(n, nx)).astype(float)
    x = np.arange(nx)
    counts += 3000 * np.exp(-0.5 * ((x - x0) / 3.0) ** 2)  # specular
    for i, t in enumerate(tth):
        centre = intercept + slope * t
        counts[i] += 800 * np.exp(-0.5 * ((x - centre) / 2.0) ** 2)  # direct beam
    for i, pos in outliers:  # another feature beats the direct beam in a few frames
        counts[i] += 5000 * np.exp(-0.5 * ((x - pos) / 2.0) ** 2)
    return counts, tth


def test_direct_beam_fit_recovers_scale_direction_and_specular_x0():
    counts, tth = _synthetic(outliers=[(30, 100), (40, 900)])
    cal = calibrate(counts, tth, np.tile([385, 415], (tth.size, 1)))
    assert cal.deg_per_pixel == pytest.approx(1 / 120, rel=2e-3)
    assert cal.sign == -1  # beam moves to larger X as 2θ grows -> α_f decreases with X
    assert cal.x0 == pytest.approx(400.5, abs=0.3)  # peak in pixel 400, whose centre is 400.5
    assert cal.fit.rms_px < 0.5
    assert 30 not in cal.fit.frames and 40 not in cal.fit.frames  # outliers rejected


def test_direction_follows_fit_slope():
    counts, tth = _synthetic(slope=-110.0, intercept=600.0, x0=605.0)
    cal = calibrate(counts, tth, np.tile([590, 620], (tth.size, 1)))
    assert cal.sign == 1 and cal.deg_per_pixel == pytest.approx(1 / 110, rel=2e-3)


def test_centroids_use_pixel_centre_coordinates():
    from pySAview.core.angle_calibration import specular_center
    counts = np.zeros((3, 20))
    counts[:, 7] = 500  # all counts in pixel 7, which spans [7, 8)
    assert specular_center(counts, np.tile([5, 10], (3, 1))) == 7.5
    cal = AngleCalibration(0.01, 7.5, -1)
    assert alpha_f_offsets(20, cal)[7] == 0.0  # that pixel's centre is exactly at 2θ − θ


def test_no_streak_requires_manual_value():
    rng = np.random.default_rng(1)
    counts = rng.poisson(1.0, size=(20, 256)).astype(float)
    with pytest.raises(CalibrationError):
        fit_direct_beam(counts, np.linspace(0, 3, 20), (100, 130))


def test_geometry_and_manual_override():
    cal = AngleCalibration(0.01, 10.0, -1)
    np.testing.assert_allclose(alpha_f_offsets(4, cal, edges=True), [0.1, 0.09, 0.08, 0.07, 0.06])
    alpha_i, alpha_f = alpha_angles([1.0], [2.5], 4, cal)
    assert alpha_i[0] == 1.0
    np.testing.assert_allclose(alpha_f[0], 1.5 + np.array([0.095, 0.085, 0.075, 0.065]))
    manual = cal.with_manual(deg_per_pixel=0.02)
    assert manual.deg_per_pixel_source == "manual" and manual.x0_source == "specular peak"
    assert cal.deg_per_pixel == 0.01  # original untouched
    with pytest.raises(ValueError):
        AngleCalibration(0.0, 1.0, -1)


def test_implied_distance_from_detector_width():
    pitch = pixel_pitch_mm(1024)
    assert pitch == pytest.approx(300.0 / 1024)
    # A pixel p at distance L subtends atan(p / L): the inverse must give L back.
    for distance in (1000.0, 2050.0, 3500.0):
        dpp = np.degrees(np.arctan(pitch / distance))
        assert implied_distance_mm(dpp, pitch) == pytest.approx(distance, rel=1e-12)
    cal = AngleCalibration(0.00819, 370.0, -1)
    assert cal.distance_mm(pitch) == pytest.approx(2049.6, abs=0.5)
    assert cal.with_manual(deg_per_pixel=0.01).distance_mm(pitch) < cal.distance_mm(pitch)
    assert "Implied sample–detector distance 2050 mm" in cal.describe(pitch)
    assert "distance" not in cal.describe()
    with pytest.raises(ValueError):
        pixel_pitch_mm(0)


def test_strip_edges_do_not_interpolate_counts():
    edges = alpha_i_strip_edges(np.array([0.0, 0.1, 0.3, 0.3]))
    np.testing.assert_allclose(edges[0], [-0.075, 0.05])  # end strip: half the median θ step
    np.testing.assert_allclose(edges[1], [0.05, 0.2])
    np.testing.assert_allclose(edges[2], edges[3])  # repeated θ shares a strip


@pytest.fixture
def window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_file(DATA / "005572.nxs")
    yield win
    win.view_offspec.shutdown()


def _ready(qtbot, view):
    view._request_compute()
    qtbot.waitUntil(lambda: view._norm is not None and not view._jobs, timeout=60000)


def test_alpha_view_on_example_scan(qtbot, window, tmp_path):
    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    _ready(qtbot, view)
    scan = window.loaded_scans[str((DATA / "005572.nxs").resolve())]
    cal = scan.angle_calibration
    assert cal.deg_per_pixel == pytest.approx(0.00819, abs=2e-5)
    assert cal.sign == -1 and cal.x0 == pytest.approx(369.9, abs=1.0)
    assert cal.fit_y_rows == (0, 1024)
    # Distance implied by the fitted scale and the 300 mm / 1024 px detector.
    assert view.lbl_distance.text().startswith("2051 mm")
    assert "Implied sample–detector distance" in view.cal_status
    assert scan.detector_roi is not None  # file ROI untouched

    view.combo_y_axis.setCurrentIndex(view.combo_y_axis.findData("alpha"))
    g = view._alpha_geom
    assert g is not None and len(view._alpha_items) == g["rows"].size == view._norm.frame_indices.size
    assert view.img_item.image is None
    # Specular pixel x0 lands on α_f = α_i for every point.
    specular_af = g["base"] + cal.sign * (cal.x0 - cal.x0) * cal.deg_per_pixel
    np.testing.assert_allclose(specular_af - g["alpha_i"], g["base"] - g["alpha_i"])
    assert np.median(np.abs(g["base"] - g["alpha_i"])) < 0.005  # 2θ − θ ≈ θ in a specular scan

    # Crosshair maps view (α_i, α_f) back to the measured cell.
    view.btn_crosshair.setChecked(True)
    view._place_crosshair(1.2, 0.6)
    row, x = view._sel_row, view._sel_x
    k = int(np.argmin(np.abs(g["alpha_i"] - 1.2)))
    assert row == g["rows"][k]
    expected_x = int(np.floor(cal.x0 + cal.sign * (0.6 - g["base"][k]) / cal.deg_per_pixel))
    assert x == expected_x and "α_f = 0.6000" in view.lbl_col_cut.text()



def test_alpha_view_needs_single_spin(qtbot, window):
    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    _ready(qtbot, view)
    view.combo_y_axis.setCurrentIndex(view.combo_y_axis.findData("alpha"))
    view.combo_spin.setCurrentIndex(view.combo_spin.findData("all"))
    assert view._alpha_geom is None and "one spin state" in view.lbl_notice.text()
    view.combo_spin.setCurrentIndex(view.combo_spin.findData("down"))
    assert view._alpha_geom is not None


def test_manual_override_is_per_scan_and_refit_restores(qtbot, window):
    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    _ready(qtbot, view)
    first = window.loaded_scans[str((DATA / "005572.nxs").resolve())]
    fitted_x0 = first.angle_calibration.x0
    view.spin_dpp.setValue(0.009)
    assert first.angle_calibration.deg_per_pixel == pytest.approx(0.009)
    assert first.angle_calibration.deg_per_pixel_source == "manual"
    assert first.angle_calibration.x0 == fitted_x0  # only the edited quantity changes

    window.load_file(DATA / "005594.nxs")  # becomes active
    _ready(qtbot, view)
    second = window.loaded_scans[str((DATA / "005594.nxs").resolve())]
    assert second.angle_calibration.deg_per_pixel_source == "direct-beam fit"  # not inherited
    assert second.angle_calibration.deg_per_pixel == pytest.approx(0.00819, abs=2e-5)
    view.combo_y_axis.setCurrentIndex(view.combo_y_axis.findData("alpha"))
    view.combo_spin.setCurrentIndex(view.combo_spin.findData("down"))  # 005594's unwritten frames read F=0
    assert "78 unwritten/invalid frame(s) omitted" in view.lbl_notice.text()

    window.set_active_file(DATA / "005572.nxs")
    _ready(qtbot, view)
    assert view.spin_dpp.value() == pytest.approx(0.009)
    view.btn_fit_db.click()
    assert first.angle_calibration.deg_per_pixel_source == "direct-beam fit"


def test_q_conversion_and_inverse():
    from pySAview.core.angle_calibration import angles_from_q, q_from_angles
    ai = np.array([0.3, 1.2, 2.5, 0.8])
    af = np.array([0.3, -1.2, 3.7, -2.0])
    qx, qz = q_from_angles(ai, af, 5.21)
    k0 = 2 * np.pi / 5.21
    assert qx[0] == pytest.approx(0.0, abs=1e-15)  # specular: Qx = 0
    assert qz[1] == pytest.approx(0.0, abs=1e-15)  # direct beam: Qz = 0
    assert qz[2] == pytest.approx(k0 * (np.sin(np.radians(3.7)) + np.sin(np.radians(2.5))))
    invertible = [0, 2, 3]  # the direct beam (index 1) collapses to Q = 0 and has no unique inverse
    back_i, back_f = angles_from_q(qx[invertible], qz[invertible], 5.21)
    np.testing.assert_allclose(back_i, ai[invertible], atol=1e-10)  # includes Qz < 0 (index 3)
    np.testing.assert_allclose(back_f, af[invertible], atol=1e-10)
    with pytest.raises(ValueError):
        q_from_angles(1.0, 1.0, 0.0)


def test_q_view_on_example_scan(qtbot, window):
    from pySAview.ui.off_specular_view import QX_SCALE
    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    _ready(qtbot, view)
    view.combo_y_axis.setCurrentIndex(view.combo_y_axis.findData("q"))
    assert len(view._alpha_items) == 1  # one exact-cell mesh
    x0, x1, y0, y1 = view._map_extent()
    assert x0 < 0 < x1 and y1 > 0.2  # Qx in 10⁻³ Å⁻¹ spans the specular rod; Qz up to ~0.23 Å⁻¹

    # View coordinates round-trip to the same measured cell.
    for row, x in ((40, 370), (100, 600), (10, 120), (10, 1000)):  # (10, 1000): below the horizon, Qz < 0
        assert view._cell_at(*view._cell_center(row, x)) == (row, x)

    # The specular pixel x0 maps onto Qx ≈ 0 for every point.
    g = view._alpha_geom
    cal = g["cal"]
    qx, _ = view._to_view(g["alpha_i"], g["base"])
    assert np.median(np.abs(qx)) < 0.01  # 10⁻³ Å⁻¹ units

    view.btn_crosshair.setChecked(True)
    view._place_crosshair(-0.5, 0.04)
    assert "constant Qz = 0.0400" in view.lbl_row_cut.text()
    assert view._row_curve.xData.size > 50 and view._col_curve.xData.size > 50
    np.testing.assert_allclose(np.diff(view._row_curve.xData) >= 0, True)  # sorted along Qx

    # Q scales as 1/λ.
    view.spin_wavelength.setValue(5.21 * 2)
    assert view._map_extent()[3] == pytest.approx(y1 / 2, rel=1e-6)

    view.combo_norm.setCurrentIndex(view.combo_norm.findData("monitor"))
    view.combo_spin.setCurrentIndex(view.combo_spin.findData("asym"))
    assert view._alpha_geom is not None


def test_cached_mesh_blits_and_invalidates(qtbot):
    """The Qx–Qz mesh is drawn from cached images while the view moves, sharpened to the
    exact cells once still, and re-rendered when data, levels or colours change."""
    import pyqtgraph as pg
    from PyQt6.QtGui import QImage, QPainter
    from pySAview.ui.off_specular_view import CachedMeshItem
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    x, y = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 6))
    z = np.arange(50, dtype=float).reshape(5, 10)
    z[2] = np.nan  # gaps between strips stay empty
    mesh = CachedMeshItem(x, y + 0.1 * x, z, enableAutoLevels=False)
    mesh.setLevels((0, 50))
    widget.addItem(mesh)
    widget.show()
    qtbot.waitUntil(lambda: mesh._detail is not None, timeout=5000)
    first_picture, first_detail = mesh._picture_used, mesh._detail
    assert first_picture is mesh.qpicture and mesh._overview is not None

    widget.getViewBox().scaleBy((0.5, 0.5))  # the view moved: same cells, sharpened again once still
    qtbot.waitUntil(lambda: mesh._detail is not first_detail and mesh._detail is not None, timeout=5000)
    assert mesh._picture_used is first_picture

    mesh.setLevels((10, 20))  # new levels: cached images are dropped and redrawn
    qtbot.waitUntil(lambda: mesh._picture_used is not first_picture and mesh._detail is not None,
                    timeout=5000)

    mesh.setExportMode(True, {})  # exports draw the exact cells, never a cached image
    image = QImage(50, 50, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    detail = mesh._detail
    mesh.paint(painter, None, None)
    painter.end()
    assert mesh._detail is detail and not mesh._detail_timer.isActive()
    mesh.setExportMode(False)


def test_low_tth_exclusion_blocks_gibaud_shift_and_validates_bounds():
    from pySAview.core.angle_calibration import find_direct_beam

    rng = np.random.default_rng(11)
    nx = 1024
    x = np.arange(nx, dtype=float) + 0.5
    tth = np.linspace(0.0, 4.0, 81)  # 0.05° steps
    slope_true, intercept_true, x0_true = 122.0, 364.0, 370.0
    counts = rng.poisson(1.0, size=(tth.size, nx)).astype(float)
    counts += 3000.0 * np.exp(-0.5 * ((x - x0_true) / 2.5) ** 2)
    for i, t in enumerate(tth):
        # Below 0.40°, inject a Gibaud-style grazing-angle masking shift toward the specular ridge.
        shift = -2.2 if t < 0.40 else 0.0
        centre = intercept_true + slope_true * t + shift
        counts[i] += 1000.0 * np.exp(-0.5 * ((x - centre) / 2.0) ** 2)

    roi_x = np.tile([355.0, 385.0], (tth.size, 1))
    cal = calibrate(counts, tth, roi_x)
    assert cal.fit is not None
    assert cal.fit.min_tth_used >= 0.40
    assert np.all(np.abs(cal.fit.tth) >= 0.40)
    assert cal.fit.excluded_low_tth == int(np.sum(tth < 0.40))
    assert cal.deg_per_pixel == pytest.approx(1.0 / slope_true, rel=5e-4)
    assert cal.fit.rms_px < 0.15

    # Including the shifted low-2θ region (min_tth=0.10) degrades RMS markedly.
    cal_unfiltered = calibrate(counts, tth, roi_x, min_tth=0.10)
    assert cal_unfiltered.fit.min_tth_used < 0.40
    assert cal_unfiltered.fit.rms_px > 0.35

    # max_tth caps the fitted 2θ range cleanly.
    cal_capped = calibrate(counts, tth, roi_x, min_tth=0.50, max_tth=2.50)
    assert 0.50 <= cal_capped.fit.min_tth_used and cal_capped.fit.max_tth_used <= 2.50
    assert "(2θ 0.50°–2.50°)" in cal_capped.describe()

    # Input validation for min_tth, max_tth, and edge_margin.
    with pytest.raises(ValueError):
        find_direct_beam(counts, tth, (355, 385), min_tth=-0.1)
    with pytest.raises(ValueError):
        find_direct_beam(counts, tth, (355, 385), min_tth=0.40, max_tth=0.40)
    with pytest.raises(ValueError):
        find_direct_beam(counts, tth, (355, 385), min_tth=0.40, max_tth=0.20)
    with pytest.raises(ValueError):
        fit_direct_beam(counts, tth, (355, 385), edge_margin=-1)


def test_active_edge_cutoff_detection_upper_and_lower():
    nx = 1024
    x = np.arange(nx, dtype=float) + 0.5
    slope_true, intercept_true, x0_true = 122.0, 364.0, 370.0
    tth = np.linspace(0.0, 5.1, 103)  # x_pred spans 364..986 px
    counts = np.full((tth.size, nx), 1.0, dtype=float)
    counts += 3000.0 * np.exp(-0.5 * ((x - x0_true) / 2.5) ** 2)
    for i, t in enumerate(tth):
        centre = intercept_true + slope_true * t
        counts[i] += 1200.0 * np.exp(-0.5 * ((x - centre) / 2.2) ** 2)
    # Dead cathode border (x >= 900) with realistic 15-px rolloff clipping the beam over 885..900.
    counts *= np.clip((900.0 - x) / 15.0, 0.0, 1.0)[None, :]

    roi_x = np.tile([355.0, 385.0], (tth.size, 1))
    cal = calibrate(counts, tth, roi_x)
    assert cal.fit is not None
    assert cal.fit.detected_edge_px is not None
    assert 890.0 <= cal.fit.detected_edge_px <= 901.0
    assert cal.fit.excluded_edge > 0
    assert cal.fit.positions_px.max() <= cal.fit.detected_edge_px - 20.0
    assert cal.deg_per_pixel == pytest.approx(1.0 / slope_true, rel=5e-4)
    assert cal.fit.rms_px < 0.05

    # Without edge_margin protection, the clipped edge frames bias slope and inflate RMS.
    cal_no_margin = calibrate(counts, tth, roi_x, edge_margin=0)
    assert cal_no_margin.fit.rms_px > 5.0 * cal.fit.rms_px

    # Symmetric check with negative slope (beam moves toward lower X and hits a dead border at x < 120).
    slope_neg, intercept_neg, x0_neg = -120.0, 660.0, 655.0
    counts_neg = np.full((tth.size, nx), 1.0, dtype=float)
    counts_neg += 3000.0 * np.exp(-0.5 * ((x - x0_neg) / 2.5) ** 2)
    for i, t in enumerate(tth):
        centre = intercept_neg + slope_neg * t
        counts_neg[i] += 1200.0 * np.exp(-0.5 * ((x - centre) / 2.2) ** 2)
    counts_neg *= np.clip((x - 120.0) / 15.0, 0.0, 1.0)[None, :]

    cal_neg = calibrate(counts_neg, tth, np.tile([640.0, 670.0], (tth.size, 1)))
    assert cal_neg.sign == 1
    assert cal_neg.fit.detected_edge_px is not None
    assert 119.0 <= cal_neg.fit.detected_edge_px <= 130.0
    assert cal_neg.fit.excluded_edge > 0
    assert cal_neg.fit.positions_px.min() >= cal_neg.fit.detected_edge_px + 20.0
    assert cal_neg.deg_per_pixel == pytest.approx(1.0 / abs(slope_neg), rel=5e-4)
    assert cal_neg.fit.rms_px < 0.05


def test_short_scan_before_edge_is_not_trimmed():
    nx = 1024
    x = np.arange(nx, dtype=float) + 0.5
    tth = np.linspace(0.0, 2.0, 41)  # direct beam stays in [364, 608], well before any edge
    counts = np.full((tth.size, nx), 1.0, dtype=float)
    counts += 3000.0 * np.exp(-0.5 * ((x - 370.0) / 2.5) ** 2)
    for i, t in enumerate(tth):
        counts[i] += 1000.0 * np.exp(-0.5 * ((x - (364.0 + 122.0 * t)) / 2.0) ** 2)

    cal = calibrate(counts, tth, np.tile([355.0, 385.0], (tth.size, 1)))
    assert cal.fit is not None
    assert cal.fit.detected_edge_px is None
    assert cal.fit.excluded_edge == 0
    assert cal.fit.max_tth_used == pytest.approx(2.0)
    assert cal.deg_per_pixel == pytest.approx(1.0 / 122.0, rel=5e-4)


def test_local_background_subtraction_removes_centroid_bias():
    from pySAview.core.angle_calibration import find_direct_beam

    nx = 1024
    x = np.arange(nx, dtype=float) + 0.5
    true_centres = np.array([500.85, 560.20, 620.75, 680.15])
    tth = np.array([1.0, 1.5, 2.0, 2.5])
    # Sloped & elevated background (~115–145 counts around the peak) that would pull an unsubtracted centroid by ~0.09 px.
    bg = 40.0 + 0.15 * x
    counts = np.tile(bg, (tth.size, 1))
    counts += 4000.0 * np.exp(-0.5 * ((x - 370.0) / 2.5) ** 2)
    for i, c in enumerate(true_centres):
        counts[i] += 1400.0 * np.exp(-0.5 * ((x - c) / 2.0) ** 2)

    frames, pos = find_direct_beam(counts, tth, (355.0, 385.0))
    np.testing.assert_array_equal(frames, [0, 1, 2, 3])
    np.testing.assert_allclose(pos, true_centres, atol=0.015)


def test_specular_center_guards_low_tth_and_unwritten_frames():
    from pySAview.core.angle_calibration import specular_center

    nx = 200
    x = np.arange(nx, dtype=float) + 0.5
    tth = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 1.50, 2.00, 2.50])
    written = np.array([True, True, True, True, True, True, True, False, False])
    counts = np.zeros((tth.size, nx), dtype=float)
    # Low-2θ frames: contaminated inside ROI (shifted to 104.5).
    for i in range(4):
        counts[i] = 2000.0 * np.exp(-0.5 * ((x - 104.5) / 1.5) ** 2)
    # Clean high-2θ frames: true specular peak at 100.5.
    for i in range(4, 7):
        counts[i] = 2000.0 * np.exp(-0.5 * ((x - 100.5) / 1.5) ** 2)
    # Unwritten trailing frames: bogus counts at 92.5 that must be ignored.
    for i in range(7, 9):
        counts[i] = 5000.0 * np.exp(-0.5 * ((x - 92.5) / 1.5) ** 2)

    ranges = np.tile([85.0, 115.0], (tth.size, 1))
    assert specular_center(counts, ranges, tth=tth, written=written) == pytest.approx(100.5, abs=0.05)

    # Fallback when no frame reaches min_tth: uses written low-2θ frames, still ignoring unwritten frames.
    low_only_tth = np.array([0.05, 0.10, 0.20, 0.30])
    low_written = np.array([True, True, False, False])
    low_counts = np.zeros((4, nx), dtype=float)
    low_counts[:2] = 2000.0 * np.exp(-0.5 * ((x - 101.5) / 1.5) ** 2)
    low_counts[2:] = 5000.0 * np.exp(-0.5 * ((x - 90.5) / 1.5) ** 2)
    assert specular_center(low_counts, ranges[:4], tth=low_only_tth, written=low_written) == pytest.approx(101.5, abs=0.05)

    # Written mask is also respected by fit_direct_beam (unwritten bogus peaks are skipped).
    counts_db, tth_db = _synthetic(n=60)
    written_db = np.ones(tth_db.size, dtype=bool)
    written_db[-10:] = False
    counts_db[-10:] = 0.0
    counts_db[-10:, 700] = 9000.0  # bogus unwritten peaks
    cal_db = calibrate(counts_db, tth_db, np.tile([385.0, 415.0], (tth_db.size, 1)), written=written_db)
    assert np.all(cal_db.fit.frames < tth_db.size - 10)
    assert cal_db.deg_per_pixel == pytest.approx(1.0 / 120.0, rel=2e-3)


def test_real_scans_exclude_low_tth_and_edge_and_agree():
    from pySAview.core.nexus_reader import NexusReader
    from pySAview.core.off_specular import DETECTOR_PATH, compute_off_specular_map

    results = {}
    for fname in ("005572.nxs", "005594.nxs"):
        path = DATA / fname
        with NexusReader(path) as reader:
            scan = reader.parse_scan()
            omap = compute_off_specular_map(reader.file[DETECTOR_PATH], None, source_path=path)
        tth = np.asarray(scan.variables["tth"].data, dtype=float)
        roi = scan.detector_roi
        roi_x = np.tile([roi[0], roi[1]], (omap.n_frames, 1)).astype(float)
        cal = calibrate(omap.counts, tth, roi_x, written=omap.written)
        assert cal.fit is not None
        assert cal.fit.min_tth_used >= 0.40
        assert cal.fit.excluded_low_tth > 0
        assert cal.fit.detected_edge_px is not None
        assert cal.fit.excluded_edge > 0
        assert cal.fit.positions_px.max() < 895.0
        assert cal.fit.rms_px < 0.45
        results[fname] = cal

    dpp_5572 = results["005572.nxs"].deg_per_pixel
    dpp_5594 = results["005594.nxs"].deg_per_pixel
    assert abs(dpp_5572 - dpp_5594) / dpp_5572 < 1e-3  # agree within 0.1%


def test_post_trim_constant_position_raises_calibration_error():
    """Post-trim refit with zero slope (constant positions) raises CalibrationError, not ZeroDivisionError."""
    nx = 1024
    x = np.arange(nx, dtype=float) + 0.5
    # 10 interior frames pinned at x = 500.5, 4 edge-adjacent frames stepped slightly higher
    # so the initial fit has positive slope, followed by 6 dead frames triggering upper edge trim.
    tth = np.concatenate([
        np.linspace(0.5, 2.3, 10),
        np.array([2.5, 2.7, 2.9, 3.1]),
        np.linspace(6.0, 12.0, 6),
    ])
    counts = np.full((tth.size, nx), 1.0, dtype=float)
    for i in range(10):
        counts[i] += 1200.0 * np.exp(-0.5 * ((x - 500.5) / 1.5) ** 2)
    for k, centre in enumerate((502.0, 502.2, 502.4, 502.6), start=10):
        counts[k] += 1200.0 * np.exp(-0.5 * ((x - centre) / 1.5) ** 2)

    with pytest.raises(CalibrationError):
        fit_direct_beam(counts, tth, (355.0, 385.0), edge_margin=2)


def test_narrow_dead_border_does_not_double_trim():
    """A narrow dead border (< edge_margin wide, e.g. x >= 1015 on nx=1024) trims from ~1015, not ~1003."""
    nx = 1024
    x = np.arange(nx, dtype=float) + 0.5
    slope_true, intercept_true, x0_true = 122.0, 364.0, 370.0
    tth = np.linspace(0.0, 5.4, 361)  # ~1.83 px steps; x_pred runs 364..1022.8
    counts = np.full((tth.size, nx), 1.0, dtype=float)
    counts += 3000.0 * np.exp(-0.5 * ((x - x0_true) / 2.5) ** 2)
    for i, t in enumerate(tth):
        counts[i] += 1200.0 * np.exp(-0.5 * ((x - (intercept_true + slope_true * t)) / 1.2) ** 2)
    counts[:, 1015:] = 0.0  # narrow 9-px dead border (< edge_margin=20)

    cal = calibrate(counts, tth, np.tile([355.0, 385.0], (tth.size, 1)), edge_margin=20)
    assert cal.fit is not None
    assert cal.fit.detected_edge_px is not None
    assert 1012.0 <= cal.fit.detected_edge_px <= 1015.5
    # Trimming from ~1014.5 - 20 = 994.5 keeps interior points above 990 (would drop to ~983 if double-trimmed).
    assert cal.fit.positions_px.max() > 990.0
    assert cal.fit.positions_px.max() <= cal.fit.detected_edge_px - 20.0
    assert cal.deg_per_pixel == pytest.approx(1.0 / slope_true, rel=5e-4)


def test_negative_tth_scan_filters_by_abs_tth():
    nx = 1024
    x = np.arange(nx, dtype=float) + 0.5
    slope_true, intercept_true, x0_true = -120.0, 364.0, 370.0
    tth = np.round(np.linspace(-4.0, 0.0, 81), 6)  # exact 0.05° steps from -4.00° to 0.00°
    counts = np.full((tth.size, nx), 1.0, dtype=float)
    counts += 3000.0 * np.exp(-0.5 * ((x - x0_true) / 2.5) ** 2)
    for i, t in enumerate(tth):
        shift = -2.0 if abs(t) < 0.40 else 0.0
        counts[i] += 1000.0 * np.exp(-0.5 * ((x - (intercept_true + slope_true * t + shift)) / 2.0) ** 2)

    cal = calibrate(counts, tth, np.tile([355.0, 385.0], (tth.size, 1)))
    assert cal.fit is not None
    assert np.all(np.abs(cal.fit.tth) >= 0.40)
    assert cal.fit.excluded_low_tth == int(np.sum(np.abs(tth) < 0.40))
    assert cal.fit.min_tth_used == pytest.approx(-4.0)
    assert cal.fit.max_tth_used == pytest.approx(-0.40)
    assert cal.deg_per_pixel == pytest.approx(1.0 / abs(slope_true), rel=5e-4)
    assert cal.sign == 1


def test_small_detector_width_within_edge_margin_raises_calibration_error():
    from pySAview.core.angle_calibration import find_direct_beam

    nx = 32  # 2 * edge_margin (40) >= nx
    tth = np.linspace(0.5, 3.0, 20)
    counts = np.full((tth.size, nx), 1.0, dtype=float)
    counts[:, 16] = 1000.0
    frames, pos = find_direct_beam(counts, tth, (2.0, 6.0), edge_margin=20)
    assert frames.size == 0 and pos.size == 0
    with pytest.raises(CalibrationError):
        fit_direct_beam(counts, tth, (2.0, 6.0), edge_margin=20)


def test_nonfinite_and_negative_count_safety():
    from pySAview.core.angle_calibration import find_direct_beam, specular_center

    nx = 256
    x = np.arange(nx, dtype=float) + 0.5
    tth = np.array([1.0, 1.5, 2.0])
    counts = np.full((3, nx), 1.0, dtype=float)
    for i, c in enumerate((140.5, 160.5, 180.5)):
        counts[i] += 1000.0 * np.exp(-0.5 * ((x - c) / 2.0) ** 2)
        counts[i] += 2500.0 * np.exp(-0.5 * ((x - 60.5) / 2.0) ** 2)
    # Inject ±inf, NaN, and negative values into both the ROI and outside regions.
    counts[:, 100] = np.inf
    counts[:, 101] = -np.inf
    counts[:, 102] = np.nan
    counts[:, 120:130] = -50.0
    counts[:, 52] = np.inf
    counts[:, 53] = -np.inf
    counts[:, 54] = -200.0

    frames, pos = find_direct_beam(counts, tth, (45.0, 75.0))
    np.testing.assert_array_equal(frames, [0, 1, 2])
    np.testing.assert_allclose(pos, [140.5, 160.5, 180.5], atol=0.05)

    assert specular_center(counts, np.tile([45.0, 75.0], (3, 1)), tth=tth) == pytest.approx(60.5, abs=0.05)

    # All-negative / zero ROI with min_counts=0.0 must raise CalibrationError rather than divide by zero.
    bad_roi_counts = np.full((3, 40), -5.0, dtype=float)
    bad_roi_counts[:, 10] = np.inf
    bad_roi_counts[:, 11] = -np.inf
    with pytest.raises(CalibrationError):
        specular_center(bad_roi_counts, np.tile([5.0, 20.0], (3, 1)), min_counts=0.0)


def test_pysared_mode_exact_arctan_offsets_and_edges():
    nx = 1024
    x0 = 370.5
    cal = AngleCalibration.from_pysared(x0=x0, sign=-1, nx=nx)
    assert cal.mode == "pysared"
    assert cal.sample_detector_distance_mm == PYSARED_DEFAULT_DISTANCE_MM == 2500.0
    assert cal.detector_width_mm == DETECTOR_WIDTH_MM == 300.0
    assert cal.deg_per_pixel_source == "pySAred geometry"
    assert cal.x0_source == "manual"
    expected_dpp = np.degrees(np.arctan((300.0 / nx) / 2500.0))
    assert cal.deg_per_pixel == pytest.approx(expected_dpp, rel=1e-14)
    assert cal.distance_mm(pixel_pitch_mm(nx)) == pytest.approx(2500.0, rel=1e-14)
    assert cal.distance_mm() == pytest.approx(2500.0, rel=1e-14)

    # Exact pySAred arctan formula at pixel centres and pixel edges.
    mm_per_pix = 300.0 / nx
    x_centres = np.arange(nx, dtype=float) + 0.5
    expected_centres = -np.degrees(np.arctan((x_centres - x0) * mm_per_pix / 2500.0))
    offsets = alpha_f_offsets(nx, cal)
    np.testing.assert_allclose(offsets, expected_centres, atol=1e-14)
    assert offsets[370] == 0.0

    x_edges = np.arange(nx + 1, dtype=float)
    expected_edges = -np.degrees(np.arctan((x_edges - x0) * mm_per_pix / 2500.0))
    np.testing.assert_allclose(alpha_f_offsets(nx, cal, edges=True), expected_edges, atol=1e-14)

    # Must use true arctan geometry, distinct from the linear small-angle approximation at far pixels.
    linear_far = -(x_centres[-1] - x0) * cal.deg_per_pixel
    assert abs(offsets[-1] - linear_far) > 5e-3

    # Custom sign (+1), detector width, distance, and nx.
    cal_custom = AngleCalibration.from_pysared(
        distance_mm=1500.0, x0=200.0, sign=1, detector_width_mm=200.0, nx=512
    )
    x_512 = np.arange(512, dtype=float) + 0.5
    expected_custom = np.degrees(np.arctan((x_512 - 200.0) * (200.0 / 512) / 1500.0))
    np.testing.assert_allclose(alpha_f_offsets(512, cal_custom), expected_custom, atol=1e-14)

    # Scalar and arbitrary float pixel helper agrees identically.
    xb = 354.25
    scalar_off = alpha_f_offset_for_pixel(xb, nx, cal)
    assert isinstance(scalar_off, float)
    assert scalar_off == pytest.approx(
        -np.degrees(np.arctan((xb - x0) * mm_per_pix / 2500.0)), abs=1e-14
    )


def test_pixel_for_alpha_f_offset_round_trip_both_modes():
    nx = 1024
    for sign in (-1, 1):
        # Direct-beam linear mode round-trip (centres, edges, scalars).
        cal_lin = AngleCalibration(0.008185, 369.9, sign)
        off_c = alpha_f_offsets(nx, cal_lin)
        x_back = pixel_for_alpha_f_offset(off_c, nx, cal_lin)
        assert isinstance(x_back, np.ndarray)
        np.testing.assert_allclose(x_back, np.arange(nx, dtype=float) + 0.5, atol=1e-12)
        off_e = alpha_f_offsets(nx, cal_lin, edges=True)
        np.testing.assert_allclose(
            pixel_for_alpha_f_offset(off_e, nx, cal_lin), np.arange(nx + 1, dtype=float), atol=1e-12
        )
        px_scalar = pixel_for_alpha_f_offset(float(off_c[700]), nx, cal_lin)
        assert isinstance(px_scalar, float)
        assert px_scalar == pytest.approx(700.5, abs=1e-12)
        assert np.isnan(pixel_for_alpha_f_offset(float("nan"), nx, cal_lin))

        # pySAred arctan mode round-trip, including short distance where curvature is strong (~26 deg).
        for dist, width in ((2500.0, 300.0), (300.0, 300.0), (1800.0, 240.0)):
            cal_ps = AngleCalibration.from_pysared(
                distance_mm=dist, x0=370.25, sign=sign, detector_width_mm=width, nx=nx
            )
            ps_off_c = alpha_f_offsets(nx, cal_ps)
            ps_x_back = pixel_for_alpha_f_offset(ps_off_c, nx, cal_ps)
            assert isinstance(ps_x_back, np.ndarray)
            np.testing.assert_allclose(ps_x_back, np.arange(nx, dtype=float) + 0.5, atol=1e-11)

            ps_off_e = alpha_f_offsets(nx, cal_ps, edges=True)
            np.testing.assert_allclose(
                pixel_for_alpha_f_offset(ps_off_e, nx, cal_ps),
                np.arange(nx + 1, dtype=float),
                atol=1e-11,
            )

            # Arbitrary wide offsets round-trip through inverse -> forward.
            sample_offsets = np.array([-75.0, -20.0, -0.5, 0.0, 3.25, 45.0, 75.0])
            x_inv = pixel_for_alpha_f_offset(sample_offsets, nx, cal_ps)
            np.testing.assert_allclose(
                alpha_f_offset_for_pixel(x_inv, nx, cal_ps), sample_offsets, atol=1e-11
            )

            # Scalar input returns Python float.
            x_s = pixel_for_alpha_f_offset(float(ps_off_c[900]), nx, cal_ps)
            assert isinstance(x_s, float)
            assert x_s == pytest.approx(900.5, abs=1e-11)

            # |offset_deg| >= 90 deg and non-finite offsets have no flat-detector preimage -> NaN.
            oob = np.array([-120.0, -90.0, 90.0, 95.0, np.inf, -np.inf, np.nan])
            res_oob = pixel_for_alpha_f_offset(oob, nx, cal_ps)
            assert np.all(np.isnan(res_oob))
            assert np.isnan(pixel_for_alpha_f_offset(90.0, nx, cal_ps))
            assert np.isnan(pixel_for_alpha_f_offset(-95.0, nx, cal_ps))


def test_pysared_mode_qx_qz_consistency():
    nx = 1024
    lam = 5.21
    dist = 2500.0
    width = 300.0
    x0 = 370.5
    sign = -1
    th = np.array([0.5, 1.2, 2.4])
    tth = 2.0 * th

    cal = AngleCalibration.from_pysared(
        distance_mm=dist, x0=x0, sign=sign, detector_width_mm=width, nx=nx
    )
    ai, af = alpha_angles(th, tth, nx, cal)
    qx, qz = q_from_angles(ai[:, None], af, lam)

    # Compare directly against pySAred's explicit equations.
    k0 = 2.0 * np.pi / lam
    mm_per_pix = width / nx
    x = np.arange(nx, dtype=float) + 0.5
    af_ref = (tth - th)[:, None] + sign * np.degrees(np.arctan((x - x0) * mm_per_pix / dist))[None, :]
    qx_ref = k0 * (np.cos(np.radians(af_ref)) - np.cos(np.radians(th[:, None])))
    qz_ref = k0 * (np.sin(np.radians(af_ref)) + np.sin(np.radians(th[:, None])))
    np.testing.assert_allclose(qx, qx_ref, atol=1e-15)
    np.testing.assert_allclose(qz, qz_ref, atol=1e-15)

    # Specular pixel (x = x0 = 370.5 -> index 370) sits on Qx = 0, Qz = (4pi/lam) sin(th).
    np.testing.assert_allclose(qx[:, 370], 0.0, atol=1e-15)
    np.testing.assert_allclose(qz[:, 370], 2.0 * k0 * np.sin(np.radians(th)), atol=1e-12)

    # Full inverse round-trip: (Qx, Qz) -> (alpha_i, alpha_f) -> pixel coordinate x.
    for row_idx in range(th.size):
        for px in (25, 200, 370, 600, 950):
            ai_b, af_b = angles_from_q(qx[row_idx, px], qz[row_idx, px], lam)
            assert ai_b == pytest.approx(th[row_idx], abs=1e-10)
            x_b = pixel_for_alpha_f_offset(af_b - (tth[row_idx] - th[row_idx]), nx, cal)
            assert x_b == pytest.approx(px + 0.5, abs=1e-10)


def test_mode_switching_preserves_fit_and_describe():
    counts, tth = _synthetic(n=60)
    cal_db = calibrate(counts, tth, np.tile([385.0, 415.0], (tth.size, 1)), y_rows=(12, 900))
    assert cal_db.mode == "direct_beam"
    assert cal_db.fit is not None
    assert cal_db.sample_detector_distance_mm == PYSARED_DEFAULT_DISTANCE_MM
    assert cal_db.detector_width_mm == DETECTOR_WIDTH_MM

    # Switch to pySAred mode: fit, fit_y_rows, x0, x0_source, and sign are preserved; original untouched.
    cal_ps = cal_db.with_mode("pysared", nx=1024)
    assert cal_db.mode == "direct_beam"
    assert cal_ps.mode == "pysared"
    assert cal_ps.fit is cal_db.fit
    assert cal_ps.fit_y_rows == (12, 900)
    assert cal_ps.x0 == cal_db.x0 and cal_ps.x0_source == "specular peak"
    assert cal_ps.sign == cal_db.sign
    assert cal_ps.deg_per_pixel == pytest.approx(effective_deg_per_pixel(2500.0, 1024, 300.0), rel=1e-14)
    assert cal_ps.deg_per_pixel_source == "pySAred geometry"

    # Update pySAred geometry parameters while keeping fit.
    cal_ps2 = cal_ps.with_pysared(distance_mm=2100.0, detector_width_mm=280.0, x0=405.0, nx=1024)
    assert cal_ps2.mode == "pysared"
    assert cal_ps2.sample_detector_distance_mm == 2100.0
    assert cal_ps2.detector_width_mm == 280.0
    assert cal_ps2.x0 == 405.0 and cal_ps2.x0_source == "manual"
    assert cal_ps2.fit is cal_db.fit
    assert cal_ps2.deg_per_pixel == pytest.approx(effective_deg_per_pixel(2100.0, 1024, 280.0), rel=1e-14)

    # Switch back to direct_beam: restores fit.deg_per_pixel and remembers L/W for next toggle.
    cal_back = cal_ps2.with_mode("direct_beam")
    assert cal_back.mode == "direct_beam"
    assert cal_back.fit is cal_db.fit
    assert cal_back.fit_y_rows == (12, 900)
    assert cal_back.deg_per_pixel == pytest.approx(cal_db.fit.deg_per_pixel, rel=1e-14)
    assert cal_back.deg_per_pixel_source == "direct-beam fit"
    assert cal_back.sample_detector_distance_mm == 2100.0
    assert cal_back.detector_width_mm == 280.0
    assert cal_back.x0 == 405.0 and cal_back.x0_source == "manual"

    # Switching back to pysared remembers L=2100 and W=280.
    cal_ps3 = cal_back.with_mode("pysared", nx=1024)
    assert cal_ps3.sample_detector_distance_mm == 2100.0
    assert cal_ps3.detector_width_mm == 280.0
    assert cal_ps3.deg_per_pixel == pytest.approx(effective_deg_per_pixel(2100.0, 1024, 280.0), rel=1e-14)

    # No-op direct_beam switch keeps manual deg_per_pixel intact.
    cal_manual = cal_db.with_manual(deg_per_pixel=0.009)
    assert cal_manual.with_mode("direct_beam") is cal_manual

    # Switching a fit-less pySAred calibration to direct_beam keeps valid positive deg_per_pixel as manual.
    cal_no_fit = AngleCalibration.from_pysared(distance_mm=2400.0, x0=370.0, sign=-1, nx=1024)
    cal_no_fit_db = cal_no_fit.with_mode("direct_beam")
    assert cal_no_fit_db.mode == "direct_beam"
    assert cal_no_fit_db.fit is None
    assert cal_no_fit_db.deg_per_pixel == pytest.approx(cal_no_fit.deg_per_pixel, rel=1e-14)
    assert cal_no_fit_db.deg_per_pixel_source == "manual"

    # Describe outputs clear mode-specific summaries.
    pitch = pixel_pitch_mm(1024)
    desc_db = cal_db.describe(pitch)
    assert "Implied sample–detector distance" in desc_db
    assert "Direct-beam fit:" in desc_db
    desc_ps = cal_ps2.describe(pixel_pitch_mm(1024, 280.0))
    assert "pySAred geometry: L = 2100 mm, detector width = 280 mm" in desc_ps
    assert "Effective scale" in desc_ps
    assert "Direct-beam fit:" in desc_ps
    desc_ps_no_pitch = cal_no_fit.describe()
    assert "pySAred geometry: L = 2400 mm, detector width = 300 mm" in desc_ps_no_pitch
    assert "Effective scale" in desc_ps_no_pitch


def test_pysared_validation_errors():
    valid = AngleCalibration.from_pysared(distance_mm=2500.0, x0=370.0, sign=-1, nx=1024)

    for bad_pos in (0.0, -100.0, float("nan"), float("inf"), -float("inf"), True, False, "2500"):
        with pytest.raises(ValueError):
            AngleCalibration(0.01, 370.0, -1, sample_detector_distance_mm=bad_pos)
        with pytest.raises(ValueError):
            AngleCalibration(0.01, 370.0, -1, detector_width_mm=bad_pos)
        with pytest.raises(ValueError):
            AngleCalibration.from_pysared(distance_mm=bad_pos, x0=370.0)
        with pytest.raises(ValueError):
            AngleCalibration.from_pysared(detector_width_mm=bad_pos, x0=370.0)
        with pytest.raises(ValueError):
            valid.with_pysared(distance_mm=bad_pos)
        with pytest.raises(ValueError):
            valid.with_pysared(detector_width_mm=bad_pos)
        with pytest.raises(ValueError):
            effective_deg_per_pixel(bad_pos, 1024)
        with pytest.raises(ValueError):
            effective_deg_per_pixel(2500.0, 1024, detector_width_mm=bad_pos)
        with pytest.raises(ValueError):
            AngleCalibration(bad_pos, 370.0, -1)

    with pytest.raises(ValueError):
        AngleCalibration(0.01, 370.0, -1, sample_detector_distance_mm=None)
    with pytest.raises(ValueError):
        AngleCalibration(0.01, 370.0, -1, detector_width_mm=None)
    with pytest.raises(ValueError):
        AngleCalibration.from_pysared(distance_mm=None, x0=370.0)

    for bad_mode in ("invalid", "", None, 1):
        with pytest.raises(ValueError):
            AngleCalibration(0.01, 370.0, -1, mode=bad_mode)
        with pytest.raises(ValueError):
            valid.with_mode(bad_mode)

    for bad_sign in (0, 2, -2, True, False):
        with pytest.raises(ValueError):
            AngleCalibration(0.01, 370.0, bad_sign)
        with pytest.raises(ValueError):
            AngleCalibration.from_pysared(x0=370.0, sign=bad_sign)
        with pytest.raises(ValueError):
            valid.with_pysared(sign=bad_sign)

    for bad_x0 in (float("nan"), float("inf"), True, False):
        with pytest.raises(ValueError):
            AngleCalibration(0.01, bad_x0, -1)
        with pytest.raises(ValueError):
            AngleCalibration.from_pysared(x0=bad_x0)

    for bad_nx in (0, -10, True, False, 1024.5):
        with pytest.raises(ValueError):
            AngleCalibration.from_pysared(nx=bad_nx)
        with pytest.raises(ValueError):
            alpha_f_offsets(bad_nx, valid)
        with pytest.raises(ValueError):
            alpha_f_offset_for_pixel(370.5, bad_nx, valid)
        with pytest.raises(ValueError):
            pixel_for_alpha_f_offset(0.0, bad_nx, valid)
        with pytest.raises(ValueError):
            pixel_pitch_mm(bad_nx)

    with pytest.raises(ValueError):
        valid.with_mode("pysared", deg_per_pixel=0.01)

    cal_db = AngleCalibration(0.01, 370.0, -1)
    with pytest.raises(ValueError):
        cal_db.distance_mm()



