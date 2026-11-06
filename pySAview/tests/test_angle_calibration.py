"""Direct-beam angle calibration and the α_i / α_f view of the Off-Specular tab."""

from pathlib import Path

import numpy as np
import pytest

from pySAview.core.angle_calibration import (
    AngleCalibration,
    CalibrationError,
    alpha_angles,
    alpha_f_offsets,
    alpha_i_strip_edges,
    calibrate,
    fit_direct_beam,
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
    assert view._alpha_geom is None and "one spin state" in view.lbl_info.text()
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
    assert "78 unwritten/invalid frame(s) omitted" in view.lbl_info.text()

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
    assert view._alpha_geom is not None and "Qx–Qz" in view.lbl_info.text()


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
