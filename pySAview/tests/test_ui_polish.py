"""Regression tests for UI polish, scientific axes, wheel guard, resizable BKG, stable curvature levels, auto plateau, and map smoothing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from pySAview.ui.main_window import MainWindow
from pySAview.ui.plot_tools import ScientificAxisItem, format_scientific_tick
from pySAview.ui.theme import DARK_THEME, LIGHT_THEME
from pySAview.ui.wheel_guard import install_wheel_focus_guard

EXAMPLE_5572 = Path(__file__).resolve().parent.parent / "example_data" / "005572.nxs"
RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"


def _wait_offspec(qtbot, view):
    view._request_compute()
    qtbot.waitUntil(lambda: view._norm is not None and not view._jobs, timeout=60000)


# ----------------------------------------------------------------- 1. Scientific Y-axis
def test_format_scientific_tick_consistent_across_decades():
    log_labels = [format_scientific_tick(e, is_log_exponent=True) for e in range(-6, 4)]
    assert log_labels == ["10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "10⁰", "10¹", "10²", "10³"]
    # Never prints decimal "0.001" or "0.0001"
    assert all("0." not in lbl for lbl in log_labels)
    # Fractional exponent (zoomed-in log) uses m×10ⁿ
    assert format_scientific_tick(np.log10(2e-3), is_log_exponent=True) == "2×10⁻³"
    # Linear values also use consistent scientific notation
    assert format_scientific_tick(1e-3, is_log_exponent=False) == "10⁻³"
    assert format_scientific_tick(2.5e-2, is_log_exponent=False) == "2.5×10⁻²"
    assert format_scientific_tick(0.0, is_log_exponent=False) == "0"


def test_scientific_axis_handles_none_spacing_and_is_installed_on_plots(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    try:
        left_axis = win.view_charts.plot_widget.getAxis("left")
        assert isinstance(left_axis, ScientificAxisItem)
        # PyQtGraph passes spacing=None for minor log ticks
        minor = left_axis.logTickStrings([-4.0, -3.0, -2.0, -1.0, 0.0], 1.0, None)
        assert minor == ["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "10⁰"]
        major = left_axis.logTickStrings([-4.0, -3.0, -2.0, -1.0, 0.0], 1.0, 1.0)
        assert major == ["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "10⁰"]

        # Off-specular cut plots also use ScientificAxisItem on Y
        assert isinstance(win.view_offspec.plot_row.getAxis("left"), ScientificAxisItem)
        assert isinstance(win.view_offspec.plot_col.getAxis("left"), ScientificAxisItem)
    finally:
        win.view_offspec.shutdown()


# ----------------------------------------------------------------- 2. Tab bar scroll arrows
def test_tab_bar_scroll_buttons_styled_in_both_themes():
    assert (RESOURCES_DIR / "chevron_left.svg").is_file()
    assert (RESOURCES_DIR / "chevron_left_dark.svg").is_file()
    for qss in (LIGHT_THEME, DARK_THEME):
        assert "QTabBar::scroller" in qss
        assert "QTabBar QToolButton" in qss
        assert "QTabBar QToolButton::left-arrow" in qss
        assert "QTabBar QToolButton::right-arrow" in qss
        assert "chevron_left" in qss and "chevron_right" in qss


# ----------------------------------------------------------------- 3. Resizable BKG ROI + area scaling
def test_freely_resizable_bkg_roi_updates_area_ratio_and_subtraction(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    try:
        win.load_file(EXAMPLE_5572)
        det = win.view_detector
        charts = win.view_charts
        scan = win.loaded_scans[str(EXAMPLE_5572)]

        charts.combo_y.setCurrentIndex(charts.combo_y.findData("ROI_Counts"))
        win.tabs.setCurrentWidget(det)
        det.chk_enable_bkg.setChecked(True)
        assert det._bkg_roi_rect_item is not None
        assert det._bkg_roi_rect_item.resizable is True

        # Default BKG matches Signal area (k = 1)
        x0, x1 = det.spin_bkg_x0.value(), det.spin_bkg_x1.value()
        y0, y1 = det.spin_bkg_y0.value(), det.spin_bkg_y1.value()
        orig_w = x1 - x0
        assert orig_w > 0

        # Double BKG width and shrink Y -> area ratio changes and is reflected in UI & reduction
        det.spin_bkg_x0.setValue(max(0, x0 - orig_w))
        det.spin_bkg_y0.setValue(y0 + 50)
        det.spin_bkg_y1.setValue(y1 - 50)
        assert scan.reduction_params.bkg_roi_bounds is not None
        bx0, bx1, by0, by1 = scan.reduction_params.bkg_roi_bounds
        assert (bx1 - bx0) == pytest.approx(2 * orig_w, abs=1.0)
        assert by0 == pytest.approx(y0 + 50, abs=1.0)
        assert by1 == pytest.approx(y1 - 50, abs=1.0)
        assert "k=" in det.lbl_bkg_metrics.text()
        assert "k=" in charts.lbl_red_bkg_bounds.text()
        assert len(charts._current_curves) == 2

        # Reset BKG restores matching Signal width and Y (k = 1)
        det.btn_bkg_auto.click()
        assert det.spin_bkg_y0.value() == det.spin_y0.value()
        assert det.spin_bkg_y1.value() == det.spin_y1.value()
        assert (det.spin_bkg_x1.value() - det.spin_bkg_x0.value()) == (det.spin_x1.value() - det.spin_x0.value())
    finally:
        win.view_offspec.shutdown()


# ----------------------------------------------------------------- 4. Stable curvature colorbar levels
def test_curvature_preserves_colorbar_levels_and_defaults_to_zero(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    try:
        win.load_file(EXAMPLE_5572)
        offspec = win.view_offspec
        scan = win.loaded_scans[str(EXAMPLE_5572)]

        assert scan.reduction_params.curvature_rad == 0.0
        assert offspec.spin_curvature.value() == 0.0

        win.tabs.setCurrentWidget(offspec)
        _wait_offspec(qtbot, offspec)
        levels_zero = tuple(float(v) for v in offspec.colorbar.levels())
        assert levels_zero[0] >= 0.0  # raw log10 counts never dip below 0

        # Adjusting curvature across multiple values must not fluctuate the colour scale
        for gamma in (0.035, -0.025, 0.015, 0.0):
            offspec.spin_curvature.setValue(gamma)
            levels_now = tuple(float(v) for v in offspec.colorbar.levels())
            assert levels_now == pytest.approx(levels_zero, abs=1e-9)

        # User-adjusted custom colorbar levels are also preserved during curvature tuning
        offspec.colorbar.setLevels((0.75, 3.25))
        offspec.spin_curvature.setValue(0.02)
        assert tuple(float(v) for v in offspec.colorbar.levels()) == pytest.approx((0.75, 3.25), abs=1e-9)
    finally:
        win.view_offspec.shutdown()


# ----------------------------------------------------------------- 5. Auto plateau button in ScanPlotView
def test_auto_plateau_button_normalizes_pre_critical_edge_to_one(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    try:
        win.load_file(EXAMPLE_5572)
        charts = win.view_charts
        charts.combo_x.setCurrentIndex(charts.combo_x.findData("Q"))
        charts.combo_y.setCurrentIndex(charts.combo_y.findData("ROI_Counts"))
        charts.combo_norm.setCurrentIndex(charts.combo_norm.findData("monitor"))
        charts.spin_sample_length.setValue(50.0)
        charts.chk_overillumination.setChecked(True)

        charts.btn_auto_scale.click()

        assert charts.chk_scale_factor.isChecked() is True
        assert charts.spin_scale_factor.value() > 1.0
        assert not charts.lbl_plateau_info.isHidden()
        assert "Plateau" in charts.lbl_plateau_info.text()
        assert len(charts._current_curves) == 2
        for curve in charts._current_curves:
            mask = (curve.x >= 0.004) & (curve.x <= 0.010)
            assert np.any(mask)
            assert float(np.median(curve.y[mask])) == pytest.approx(1.0, abs=0.03)
    finally:
        win.view_offspec.shutdown()


# ----------------------------------------------------------------- 6. Wheel focus guard
def test_wheel_focus_guard_blocks_unfocused_and_clears_on_click_away(qtbot):
    container = QWidget()
    container.resize(300, 160)
    outer = QVBoxLayout(container)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    inner = QWidget()
    inner_layout = QVBoxLayout(inner)

    lbl = QLabel("Click away target")
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 100.0)
    spin.setValue(10.0)
    spin.setSingleStep(1.0)
    combo = QComboBox()
    combo.addItems(["A", "B", "C"])
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(50)

    inner_layout.addWidget(lbl)
    inner_layout.addWidget(spin)
    inner_layout.addWidget(combo)
    inner_layout.addWidget(slider)
    spacer = QLabel("Tall spacer")
    spacer.setMinimumHeight(400)
    inner_layout.addWidget(spacer)
    scroll.setWidget(inner)
    outer.addWidget(scroll)

    install_wheel_focus_guard(container)
    qtbot.addWidget(container)
    container.show()
    qtbot.waitExposed(container)

    def send_wheel(widget: QWidget, delta_y: int = -120) -> None:
        center = QPointF(widget.width() / 2.0, widget.height() / 2.0)
        global_pos = QPointF(widget.mapToGlobal(QPoint(int(center.x()), int(center.y()))))
        ev = QWheelEvent(
            center,
            global_pos,
            QPoint(0, 0),
            QPoint(0, delta_y),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(widget, ev)

    # 1. Unfocused spinbox/combo/slider must ignore wheel events
    spin.clearFocus()
    combo.clearFocus()
    slider.clearFocus()
    send_wheel(spin, 120)
    assert spin.value() == pytest.approx(10.0)
    send_wheel(combo, -120)
    assert combo.currentIndex() == 0
    send_wheel(slider, 120)
    assert slider.value() == 50

    # 2. Explicitly focused spinbox accepts wheel events
    spin.setFocus(Qt.FocusReason.MouseFocusReason)
    QApplication.processEvents()
    assert spin.hasFocus()
    send_wheel(spin, 120)
    assert spin.value() == pytest.approx(11.0)

    # 3. Clicking away on a non-focusable QLabel clears focus from spinbox
    click_pos = QPointF(lbl.width() / 2.0, lbl.height() / 2.0)
    click_global = QPointF(lbl.mapToGlobal(QPoint(int(click_pos.x()), int(click_pos.y()))))
    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        click_pos,
        click_global,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(lbl, press)
    QApplication.processEvents()
    assert not spin.hasFocus()

    # 4. Subsequent wheel over spinbox no longer changes value
    send_wheel(spin, 120)
    assert spin.value() == pytest.approx(11.0)


# ----------------------------------------------------------------- 7. Display-only Q & alpha map smoothing
def test_q_and_alpha_map_smoothing_is_display_only_and_preserves_levels(qtbot):
    from pySAview.ui.off_specular_view import (
        CachedMeshItem,
        SmoothResampledMapItem,
        _bilateral_2d_display,
    )

    # Edge-preserving log-space bilateral must NOT bloom a 10^5 specular ridge onto nearby 10^1 background
    ridge_log = np.ones((21, 21), dtype=float)
    ridge_log[:, 10] = 5.0
    filtered = _bilateral_2d_display(ridge_log, sigma_s=1.5, sigma_r=0.45, fill_isolated_nans=False)
    assert filtered[10, 10] == pytest.approx(5.0, abs=0.05)
    assert filtered[10, 11] == pytest.approx(1.0, abs=0.05)  # zero blooming 1 px away!
    assert filtered[10, 12] == pytest.approx(1.0, abs=0.05)

    win = MainWindow()
    qtbot.addWidget(win)
    try:
        win.load_file(EXAMPLE_5572)
        offspec = win.view_offspec
        win.tabs.setCurrentWidget(offspec)
        _wait_offspec(qtbot, offspec)

        # Hidden in pixel vs theta mode
        assert offspec.combo_y_axis.currentData() == "th"
        assert offspec._smooth_group.isHidden()

        # Visible in Qx vs Qz mode, placed on the display bar level with Colormap
        offspec.combo_y_axis.setCurrentIndex(offspec.combo_y_axis.findData("q"))
        assert not offspec._smooth_group.isHidden()
        assert offspec._smooth_group.parentWidget() is offspec._cmap_group.parentWidget()
        assert offspec.chk_smooth_map.isChecked() is False
        assert isinstance(offspec._alpha_items[0], CachedMeshItem)

        norm_before = offspec._norm.values.copy()
        cuts_before = offspec._cut_values.copy()
        _, _, rows_before_gen = offspec.export_table()
        export_before = [next(rows_before_gen) for _ in range(1000)]
        levels_before = tuple(float(v) for v in offspec.colorbar.levels())

        # Enable smooth re-binning -> switches to SmoothResampledMapItem (zero flat-quad boxing)
        offspec.chk_smooth_map.setChecked(True)
        assert len(offspec._alpha_items) == 1
        smooth_item = offspec._alpha_items[0]
        assert isinstance(smooth_item, SmoothResampledMapItem)

        # Verify orientation & accuracy: Bragg peak around Qx=0, Qz≈0.116 Å⁻¹ is at the right Qz (no Y-flip)
        grid = smooth_item.sample_grid(-0.2, 0.2, 0.05, 0.16, 80, 110)
        assert np.any(np.isfinite(grid))
        peak_row, _ = np.unravel_index(int(np.nanargmax(grid)), grid.shape)
        peak_qz = 0.05 + (peak_row + 0.5) * (0.16 - 0.05) / 110.0
        assert 0.105 <= peak_qz <= 0.128

        # Pure bilinear mode (sigma = 0.0) and stronger denoising (sigma = 2.0) both work cleanly
        offspec.spin_smooth_sigma.setValue(0.0)
        grid_zero = smooth_item.sample_grid(-0.2, 0.2, 0.05, 0.16, 40, 55)
        assert np.any(np.isfinite(grid_zero))
        offspec.spin_smooth_sigma.setValue(2.0)

        # Numerical arrays, cuts, exports, and colorbar levels stayed strictly untouched
        np.testing.assert_array_equal(offspec._norm.values, norm_before)
        np.testing.assert_array_equal(offspec._cut_values, cuts_before)
        header_after, _, rows_after_gen = offspec.export_table()
        export_after = [next(rows_after_gen) for _ in range(1000)]
        assert export_after == export_before
        assert any("display-only" in line and "bilinear" in line.lower() for line in header_after)
        assert tuple(float(v) for v in offspec.colorbar.levels()) == pytest.approx(levels_before, abs=1e-9)

        # Alpha_i vs alpha_f mode also uses SmoothResampledMapItem cleanly
        offspec.combo_y_axis.setCurrentIndex(offspec.combo_y_axis.findData("alpha"))
        assert not offspec._smooth_group.isHidden()
        assert len(offspec._alpha_items) == 1
        assert isinstance(offspec._alpha_items[0], SmoothResampledMapItem)
        alpha_grid = offspec._alpha_items[0].sample_grid(0.5, 3.5, 0.5, 3.5, 60, 60)
        assert np.any(np.isfinite(alpha_grid))
        np.testing.assert_array_equal(offspec._norm.values, norm_before)

        # Unchecking Smooth restores exact unsmoothed CachedMeshItem in Q mode
        offspec.combo_y_axis.setCurrentIndex(offspec.combo_y_axis.findData("q"))
        offspec.chk_smooth_map.setChecked(False)
        assert len(offspec._alpha_items) == 1
        assert isinstance(offspec._alpha_items[0], CachedMeshItem)
    finally:
        win.view_offspec.shutdown()


# ----------------------------------------------------------------- 8. Window title, Transmitted-beam label & pySAred mode switch crash guard
def test_window_title_transmitted_beam_label_and_pysared_smooth_switch(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    try:
        assert win.windowTitle() == "pySAview"
        offspec = win.view_offspec
        assert offspec.btn_mode_direct_beam.text() == "Transmitted-beam fit"
        assert offspec.btn_fit_db.text() == "Fit Transmitted Beam"

        win.load_file(EXAMPLE_5572)
        assert win.windowTitle().startswith("pySAview — ")

        win.tabs.setCurrentWidget(offspec)
        win.show()
        _wait_offspec(qtbot, offspec)

        # Switching Transmitted-beam fit <-> pySAred geometry in Qx vs Qz and alpha views
        # with Smooth enabled must render cleanly without NaN-cast IndexError crashes
        for axes in ("q", "alpha", "th"):
            offspec.combo_y_axis.setCurrentIndex(offspec.combo_y_axis.findData(axes))
            if axes in ("q", "alpha"):
                offspec.chk_smooth_map.setChecked(True)
            offspec.btn_mode_pysared.click()
            QApplication.processEvents()
            offspec.grab()  # forces full SmoothResampledMapItem.paint() + _ensure_overview()
            assert win.loaded_scans[str(EXAMPLE_5572)].angle_calibration.mode == "pysared"

            offspec.btn_mode_direct_beam.click()
            QApplication.processEvents()
            offspec.grab()
            assert win.loaded_scans[str(EXAMPLE_5572)].angle_calibration.mode == "direct_beam"
    finally:
        win.view_offspec.shutdown()
