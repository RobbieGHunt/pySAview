"""UI tests with pytest-qt for SuperADAM NeXus Explorer."""

from pathlib import Path
import pyqtgraph as pg
import pytest
from PyQt6.QtCore import Qt

from pySAview.ui.main_window import MainWindow


@pytest.fixture
def example_nxs_path() -> Path:
    p = Path("pySAview/example_data/005572.nxs")
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / "example_data" / "005572.nxs"
    assert p.exists()
    return p


@pytest.fixture
def main_window(qtbot, example_nxs_path: Path) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    window.load_file(example_nxs_path)
    return window


def test_window_tabs_and_loading(main_window: MainWindow):
    assert main_window.tabs.count() == 5
    tab_titles = [main_window.tabs.tabText(i) for i in range(5)]
    assert "File Hierarchy" in tab_titles[0]
    assert "Scan Charts" in tab_titles[1]
    assert "Detector Image" in tab_titles[2]
    assert "2D Map" in tab_titles[3]
    assert "Metadata & Motors" in tab_titles[4]

    assert main_window.current_file_path is not None
    assert "005572.nxs" in main_window.current_file_path.name
    assert "5572" in main_window.windowTitle()


def test_dark_mode_toggle(main_window: MainWindow):
    assert main_window.is_dark_mode is False
    main_window.act_dark_mode.setChecked(True)
    assert main_window.is_dark_mode is True
    assert main_window.view_charts.is_dark_mode is True

    main_window.act_dark_mode.setChecked(False)
    assert main_window.is_dark_mode is False
    assert main_window.view_charts.is_dark_mode is False


def test_scan_charts_view(main_window: MainWindow):
    view = main_window.view_charts
    assert view.scan_data is not None

    # Check X combo options
    assert view.combo_x.count() > 0
    assert view.combo_y.count() > 0

    # Switch X axis to Q
    idx_q = view.combo_x.findData("Q")
    assert idx_q >= 0
    view.combo_x.setCurrentIndex(idx_q)
    assert len(view._current_curves) == 2  # Spin up and down

    # Change normalization to monitor
    idx_norm = view.combo_norm.findData("monitor")
    view.combo_norm.setCurrentIndex(idx_norm)

    # Verify error bars are plotted on default Log Y scale
    assert view.chk_log_y.isChecked() is True
    assert view.chk_errors.isChecked() is True
    err_items_log = [it for it in view.plot_widget.items() if isinstance(it, pg.ErrorBarItem)]
    assert len(err_items_log) == 2  # One error bar item per spin state curve
    for err in err_items_log:
        assert err.opts['beam'] == 0  # No caps, just lines
        assert err.opts['pen'].width() == 2  # Same width as regular lines

    # Verify error bars also render on linear scale
    view.chk_log_y.setChecked(False)
    err_items_lin = [it for it in view.plot_widget.items() if isinstance(it, pg.ErrorBarItem)]
    assert len(err_items_lin) == 2
    for err in err_items_lin:
        assert err.opts['beam'] == 0
        assert err.opts['pen'].width() == 2
    view.chk_log_y.setChecked(True)

    # Test chart interactive tools
    assert view.btn_pan.isChecked() is True
    assert view.plot_widget.getViewBox().state['mouseMode'] == pg.ViewBox.PanMode

    view.btn_box_zoom.click()
    assert view.btn_box_zoom.isChecked() is True
    assert view.btn_pan.isChecked() is False
    assert view.plot_widget.getViewBox().state['mouseMode'] == pg.ViewBox.RectMode

    view.btn_pan.click()
    assert view.btn_pan.isChecked() is True

    view.btn_zoom_in.click()
    view.btn_zoom_out.click()
    view.btn_reset_view.click()

    # Test toolbar log toggles
    assert view.btn_tool_log_y.isChecked() is True
    view.btn_tool_log_y.click()
    assert view.chk_log_y.isChecked() is False
    view.btn_tool_log_y.click()
    assert view.chk_log_y.isChecked() is True

    # Check stats table
    assert view.bottom_tabs.count() == 1
    assert view.bottom_tabs.tabText(0) == "Plotted Data Points"
    assert view.curve_data_table.rowCount() == 302


def test_detector_view(main_window: MainWindow):
    view = main_window.view_detector
    assert view._total_frames == 302
    assert view.slider_step.maximum() == 301
    assert view.spin_step.maximum() == 301

    # Check default colormap is turbo
    assert view.combo_cmap.currentText() == "turbo"

    # Test changing to gray colormap does NOT error
    idx_gray = view.combo_cmap.findText("gray")
    assert idx_gray >= 0
    view.combo_cmap.setCurrentIndex(idx_gray)

    # Change step to 15
    view.spin_step.setValue(15)
    assert view._current_frame_idx == 15
    assert view.lbl_frame_info.text().startswith("Frame 15 ") and "θ" in view.lbl_frame_info.text()
    assert not hasattr(view, "lbl_step_info")  # the blue step banner was removed

    # Test interactive user-defined ROI
    assert view._roi_rect_item is not None
    view.spin_x0.setValue(350)
    view.spin_x1.setValue(385)
    view.spin_y0.setValue(280)
    view.spin_y1.setValue(750)

    # Click Apply ROI to Scan Chart
    view.btn_apply_roi.click()

    # Check scan chart view now selected ROI_Counts
    assert main_window.view_charts.combo_y.currentData() == "ROI_Counts"
    assert len(main_window.view_charts._current_curves) > 0

    # Test Auto-Fit ROI button and tracking
    assert hasattr(view, "btn_auto_fit")
    assert view.btn_auto_fit.isEnabled() is True
    view.btn_auto_fit.click()

    assert view._auto_fit_active is True
    assert view._per_frame_rois is not None
    assert len(view._per_frame_rois) == 302
    assert view._per_frame_counts is not None
    assert len(view._per_frame_counts) == 302
    assert "Auto-Fit active" in view.lbl_roi_status.text()
    assert main_window.view_charts.combo_y.currentData() == "ROI_Counts"

    # Scrub to step 20 and verify ROI box dynamically follows frame 20 auto-fit ROI
    view.spin_step.setValue(20)
    expected_roi_20 = view._per_frame_rois[20]
    cur_roi_bounds = view._get_current_roi_bounds()
    assert cur_roi_bounds[0] == pytest.approx(expected_roi_20[0], abs=1.0)
    assert cur_roi_bounds[1] == pytest.approx(expected_roi_20[1], abs=1.0)

    # Test that manual spinbox edit reverts to manual mode
    view.spin_x0.setValue(int(round(expected_roi_20[0])) - 10)
    assert view._auto_fit_active is False
    assert "Manual ROI" in view.lbl_roi_status.text()

    # Test Reset to File ROI
    view.btn_reset_roi.click()
    assert view._auto_fit_active is False
    assert view._per_frame_rois is None
    assert "Reset to default" in view.lbl_roi_status.text()

    # Test 2D detector interactive tools
    assert view.btn_img_pan.isChecked() is True
    assert view.plot_2d.getViewBox().state['mouseMode'] == pg.ViewBox.PanMode

    view.btn_img_box_zoom.click()
    assert view.btn_img_box_zoom.isChecked() is True
    assert view.btn_img_pan.isChecked() is False
    assert view.plot_2d.getViewBox().state['mouseMode'] == pg.ViewBox.RectMode

    view.btn_img_pan.click()
    assert view.btn_img_pan.isChecked() is True

    view.btn_img_zoom_in.click()
    view.btn_img_zoom_out.click()
    view.btn_img_reset.click()

    # Test log/lin toggle on detector image
    assert view.chk_log_img.isChecked() is True
    view.chk_log_img.click()
    assert view.chk_log_img.isChecked() is False
    view.chk_log_img.click()
    assert view.chk_log_img.isChecked() is True
    for btn in (view.btn_play, view.btn_prev, view.btn_next, view.btn_img_pan, view.btn_img_box_zoom,
                view.btn_img_zoom_in, view.btn_img_zoom_out, view.btn_img_reset, view.btn_img_export):
        assert btn.text() == "" and btn.toolTip()  # icon-only, as in the Off-Specular toolbar

    # Toggle Play
    view._toggle_play()
    assert view._play_timer.isActive()
    view._toggle_play()
    assert not view._play_timer.isActive()


def test_file_hierarchy_view(main_window: MainWindow):
    view = main_window.view_hierarchy
    assert view.tree.topLevelItemCount() >= 1

    top_item = view.tree.topLevelItem(0)
    assert top_item is not None
    # Ensure no emoji prefix like 📁 or 📊
    assert "📁" not in top_item.text(0)
    assert "📊" not in top_item.text(0)
    assert not top_item.icon(0).isNull()

    # Search filter test
    view.search_input.setText("detector_data")
    # Clear filter
    view.search_input.setText("")


def test_metadata_view(main_window: MainWindow):
    view = main_window.view_metadata
    assert "5572" in view.lbl_exp_details.text()
    assert "Hydrogen-induced" in view.lbl_exp_details.text()
    assert "Tb/Co" in view.lbl_sample_details.text()

    # Motors table
    assert view.table_motors.rowCount() >= 30

    # Motor filter
    view.filter_motor.setText("th")
    visible_count = sum(
        1 for r in range(view.table_motors.rowCount()) if not view.table_motors.isRowHidden(r)
    )
    assert visible_count < view.table_motors.rowCount()
    view.filter_motor.setText("")


def test_multi_file_loading_and_overlay(main_window: MainWindow):
    # Second scan path: 005594.nxs
    p2 = Path("pySAview/example_data/005594.nxs")
    if not p2.exists():
        p2 = Path(__file__).resolve().parent.parent / "example_data" / "005594.nxs"
    assert p2.exists()

    # Load second file into session
    main_window.load_file(p2, make_active=False)
    assert len(main_window.loaded_scans) == 2
    assert main_window.combo_active_file.count() == 2

    # Verify global session files sidebar
    sidebar = main_window.session_files_view
    assert sidebar.list_files.count() == 2
    assert "Loaded Scans (2)" in sidebar.lbl_header.text()

    # Check multi-dataset overlay in Scan Charts tab
    view_charts = main_window.view_charts
    assert view_charts.list_datasets.count() == 2

    # Both scans checked by default -> 4 curves plotted (2 spin states x 2 scans)
    assert len(view_charts._current_curves) == 4
    labels = [c.label for c in view_charts._current_curves]
    assert any("[#5572]" in lbl for lbl in labels)
    assert any("[#5594]" in lbl for lbl in labels)

    # Test toggling via global sidebar: uncheck first scan
    sidebar.list_files.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert len(view_charts._current_curves) == 2
    for c in view_charts._current_curves:
        assert "[#5594]" in c.label or "5594" in c.label

    # Test Select All / Clear buttons on global sidebar
    sidebar.btn_select_none.click()
    assert len(view_charts._current_curves) == 0

    sidebar.btn_select_all.click()
    assert len(view_charts._current_curves) == 4

    # Test switching active file to scan 2 via sidebar selection
    k2 = sidebar.list_files.item(1).data(Qt.ItemDataRole.UserRole)
    sidebar.set_active_key(k2)
    assert "005594" in main_window.active_file_path.name
    assert main_window.view_detector._total_frames == 488
    assert "5594" in main_window.view_metadata.lbl_exp_details.text()
    assert "5594" in sidebar.lbl_info_title.text()

    # Test closing active file
    main_window.close_active_file()
    assert len(main_window.loaded_scans) == 1
    assert sidebar.list_files.count() == 1
    assert main_window.combo_active_file.count() == 1
    assert "005572" in main_window.active_file_path.name


def test_spin_filter_in_detector_view(main_window: MainWindow):
    det = main_window.view_detector
    assert det.combo_spin_filter.count() == 4
    assert det.combo_spin_filter.isEnabled() is True

    total = det._total_frames
    assert total > 0

    # Test Spin Up (Flipper OFF / 1)
    idx_up = det.combo_spin_filter.findData("up")
    det.combo_spin_filter.setCurrentIndex(idx_up)
    assert len(det._filtered_indices) == total // 2
    assert all(det.scan_data.variables["Flipper1"].data[i] == 1 for i in det._filtered_indices)
    assert "u (OFF)" in det.lbl_frame_info.text()
    det.btn_next.click()
    assert det._current_filtered_pos == 1
    assert "u (OFF)" in det.lbl_frame_info.text()

    # Test Spin Down (Flipper ON / 0)
    idx_down = det.combo_spin_filter.findData("down")
    det.combo_spin_filter.setCurrentIndex(idx_down)
    assert len(det._filtered_indices) == total // 2
    assert all(det.scan_data.variables["Flipper1"].data[i] == 0 for i in det._filtered_indices)
    assert "d (ON)" in det.lbl_frame_info.text()

    # Test All Frames
    idx_all = det.combo_spin_filter.findData("all")
    det.combo_spin_filter.setCurrentIndex(idx_all)
    assert len(det._filtered_indices) == total


def test_detector_view_coordinates_fixed_width_and_cross_cursor(main_window: MainWindow):
    det = main_window.view_detector
    assert det.lbl_img_coords.maximumWidth() == 290
    assert det.lbl_img_coords.minimumWidth() == 290
    assert "Pixel: (X: ----, Y: ----)" in det.lbl_img_coords.text()
    assert det.plot_2d.getViewBox().cursor().shape() == Qt.CursorShape.CrossCursor


def test_q_toggle_and_si_prefix_disabled(main_window: MainWindow):
    charts = main_window.view_charts
    assert charts.btn_toggle_q.accessibleName() == "X axis: θ"

    # Toggle to Q
    charts.btn_toggle_q.click()
    assert charts.combo_x.currentData() == "Q"
    assert charts.btn_toggle_q.accessibleName() == "X axis: Q"
    assert charts.btn_toggle_q.isChecked() is True

    # Verify SI prefix is disabled on bottom axis
    bot_ax = charts.plot_widget.getPlotItem().getAxis("bottom")
    assert bot_ax.autoSIPrefix is False
    assert bot_ax.labelUnitPrefix == ""
    assert "mÅ" not in bot_ax.labelString()

    # Toggle back to theta
    charts.btn_toggle_q.click()
    assert charts.combo_x.currentData() == "th"
    assert charts.btn_toggle_q.accessibleName() == "X axis: θ"
    assert charts.btn_toggle_q.isChecked() is False


def test_subtitle_in_plot_title_and_closed_border_axes(main_window: MainWindow):
    charts = main_window.view_charts
    sub = main_window.loaded_scans[str(main_window.active_file_path)].metadata.subtitle
    assert len(sub) > 0
    # Chart title takes from subtitle
    assert sub in charts.lbl_title.text()

    # Closed border frame (top and right axes enabled without values)
    pi = charts.plot_widget.getPlotItem()
    top_ax = pi.getAxis("top")
    right_ax = pi.getAxis("right")
    assert top_ax.isVisible() is True
    assert right_ax.isVisible() is True
    assert top_ax.style['showValues'] is False
    assert right_ax.style['showValues'] is False


def test_session_files_sidebar_minimization(main_window: MainWindow):
    main_window.show()
    sidebar = main_window.session_files_view
    assert sidebar.is_minimized is False
    assert not sidebar.expanded_container.isHidden()
    assert sidebar.minimized_container.isHidden()

    # Minimize via button
    sidebar.btn_minimize.click()
    assert sidebar.is_minimized is True
    assert sidebar.expanded_container.isHidden()
    assert not sidebar.minimized_container.isHidden()
    assert sidebar.maximumWidth() == 34

    # Expand via mini strip button
    sidebar.btn_expand.click()
    assert sidebar.is_minimized is False
    assert not sidebar.expanded_container.isHidden()
    assert sidebar.minimized_container.isHidden()

    # Toggle via Ctrl+B / action
    main_window.act_toggle_sidebar.trigger()
    assert sidebar.is_minimized is True
    main_window.act_toggle_sidebar.trigger()
    assert sidebar.is_minimized is False


def test_file_hierarchy_tree_minimization(main_window: MainWindow):
    main_window.show()
    hier = main_window.view_hierarchy
    assert hier.is_tree_minimized is False
    assert not hier.tree_expanded_container.isHidden()
    assert hier.tree_minimized_container.isHidden()

    # Minimize tree to left
    hier.btn_minimize_tree.click()
    assert hier.is_tree_minimized is True
    assert hier.tree_expanded_container.isHidden()
    assert not hier.tree_minimized_container.isHidden()
    assert hier.left_widget.maximumWidth() == 34

    # Expand tree
    hier.btn_expand_tree.click()
    assert hier.is_tree_minimized is False
    assert not hier.tree_expanded_container.isHidden()
    assert hier.tree_minimized_container.isHidden()


def test_flipper_labels_and_detector_sum_mode(main_window: MainWindow):
    import numpy as np

    # 1. Flipper ON = spin d (0), Flipper OFF = spin u (1) across all three spin selectors
    charts = main_window.view_charts
    det = main_window.view_detector
    offspec = main_window.view_offspec

    assert "Flipper ON" in charts.combo_spin.itemText(charts.combo_spin.findData("flipper0"))
    assert "Flipper OFF" in charts.combo_spin.itemText(charts.combo_spin.findData("flipper1"))
    assert "Flipper OFF" in det.combo_spin_filter.itemText(det.combo_spin_filter.findData("up"))
    assert "Flipper ON" in det.combo_spin_filter.itemText(det.combo_spin_filter.findData("down"))
    assert "Flipper OFF" in offspec.combo_spin.itemText(offspec.combo_spin.findData("up"))
    assert "Flipper ON" in offspec.combo_spin.itemText(offspec.combo_spin.findData("down"))

    # Frame 0 in 005572.nxs has Flipper1 == 1 -> u (OFF); Frame 1 has Flipper1 == 0 -> d (ON)
    det.combo_spin_filter.setCurrentIndex(det.combo_spin_filter.findData("all"))
    det.spin_step.setValue(0)
    frame_0 = det._current_frame_data.copy()
    assert "u (OFF)" in det.lbl_frame_info.text()
    det.spin_step.setValue(1)
    frame_1 = det._current_frame_data.copy()
    assert "d (ON)" in det.lbl_frame_info.text()

    # 2. Sum of spin channels (u + d) in Detector Image
    idx_sum = det.combo_spin_filter.findData("sum")
    assert idx_sum >= 0
    det.combo_spin_filter.setCurrentIndex(idx_sum)
    assert det.slider_step.maximum() == 150  # 151 paired (θ, 2θ) steps
    det.spin_step.setValue(0)
    assert "Frames 0+1" in det.lbl_frame_info.text()
    assert "u+d (sum)" in det.lbl_frame_info.text()
    np.testing.assert_array_equal(det._current_frame_data, frame_0 + frame_1)


def test_scan_charts_log_toolbar_buttons(qtbot, main_window: MainWindow):
    main_window.tabs.setCurrentWidget(main_window.view_charts)
    main_window.show()
    qtbot.wait(10)

    charts = main_window.view_charts
    plot_item = charts.plot_widget.getPlotItem()

    # Clear text labels, no icons, checkable, preserved tooltips & accessible names
    for btn, label in ((charts.btn_tool_log_y, "Log Y"), (charts.btn_tool_log_x, "Log X")):
        assert btn.text() == label
        assert btn.icon().isNull()
        assert btn.isCheckable() is True
        assert label in btn.toolTip()
        assert btn.accessibleName() == label
        # Flush with adjacent toolbar buttons and wide enough to never clip text
        assert btn.height() == charts.btn_pan.height()
        assert btn.width() >= btn.sizeHint().width()
        assert btn.width() >= btn.fontMetrics().horizontalAdvance(label) + 16
        assert btn.height() >= btn.fontMetrics().height() + 4

    # Initial states: Log Y on, Log X off
    assert charts.btn_tool_log_y.isChecked() is True
    assert charts.btn_tool_log_x.isChecked() is False
    assert plot_item.getAxis("left").logMode is True
    assert plot_item.getAxis("bottom").logMode is False

    # Toggling Log Y button updates checkbox, plot log mode, and keeps button width stable
    w_y_checked = charts.btn_tool_log_y.width()
    charts.btn_tool_log_y.click()
    assert charts.btn_tool_log_y.isChecked() is False
    assert charts.chk_log_y.isChecked() is False
    assert plot_item.getAxis("left").logMode is False
    assert charts.btn_tool_log_y.width() == w_y_checked

    charts.chk_log_y.setChecked(True)
    assert charts.btn_tool_log_y.isChecked() is True
    assert plot_item.getAxis("left").logMode is True

    # Toggling Log X button updates checkbox, plot log mode, and keeps button width stable
    w_x_unchecked = charts.btn_tool_log_x.width()
    charts.btn_tool_log_x.click()
    assert charts.btn_tool_log_x.isChecked() is True
    assert charts.chk_log_x.isChecked() is True
    assert plot_item.getAxis("bottom").logMode is True
    assert charts.btn_tool_log_x.width() == w_x_unchecked

    charts.btn_tool_log_x.click()
    assert charts.btn_tool_log_x.isChecked() is False
    assert charts.chk_log_x.isChecked() is False
    assert plot_item.getAxis("bottom").logMode is False

    # Light & dark themes keep text labels, keep icon null, and preserve obvious checked/unchecked contrast
    for dark in (False, True):
        main_window.act_dark_mode.setChecked(dark)
        qtbot.wait(10)
        for btn, label in ((charts.btn_tool_log_y, "Log Y"), (charts.btn_tool_log_x, "Log X")):
            assert btn.text() == label
            assert btn.icon().isNull()
        # Log Y is checked and Log X is unchecked: rendered center fill colors must clearly differ
        img_checked = charts.btn_tool_log_y.grab().toImage()
        img_unchecked = charts.btn_tool_log_x.grab().toImage()
        bg_checked = img_checked.pixelColor(4, img_checked.height() // 2)
        bg_unchecked = img_unchecked.pixelColor(4, img_unchecked.height() // 2)
        color_dist = (
            abs(bg_checked.red() - bg_unchecked.red())
            + abs(bg_checked.green() - bg_unchecked.green())
            + abs(bg_checked.blue() - bg_unchecked.blue())
        )
        assert color_dist > 150
    main_window.act_dark_mode.setChecked(False)


def test_window_unrestricted_compression_empty_and_loaded(qtbot, qapp, example_nxs_path: Path):
    """MainWindow launches at 1300x860 and compresses freely horizontally & vertically across all tabs."""
    win = MainWindow()
    qtbot.addWidget(win)
    win.show()
    qapp.processEvents()

    try:
        # 1. Default launch size is 1300x860 with tiny minimum size floor
        assert (win.width(), win.height()) == (1300, 860)
        assert win.minimumSize().width() <= 250
        assert win.minimumSize().height() <= 200
        assert win.minimumSizeHint().width() <= 250
        assert win.minimumSizeHint().height() <= 200

        # 2. Empty window compresses cleanly across all 5 tabs with sidebar expanded, minimized, and re-expanded
        for minimized in (False, True, False):
            win.session_files_view.set_minimized(minimized)
            qapp.processEvents()
            for tab_idx in range(win.tabs.count()):
                win.tabs.setCurrentIndex(tab_idx)
                for target_w, target_h in ((320, 240), (500, 350)):
                    win.resize(target_w, target_h)
                    qapp.processEvents()
                    assert (win.width(), win.height()) == (target_w, target_h)
        assert win.session_files_view.minimumWidth() == 0

        # 3. Loaded scan (005572.nxs) + long status text + tree minimize/expand cycle + dark mode
        win.resize(1300, 860)
        assert win.load_file(example_nxs_path)
        win.status_bar.showMessage("LongStatusTextWithoutSpaces_" * 20)
        win.tabs.setCurrentIndex(0)
        qapp.processEvents()
        win.view_hierarchy.set_tree_minimized(True)
        qapp.processEvents()
        assert win.view_hierarchy.left_widget.maximumWidth() == 34
        win.view_hierarchy.set_tree_minimized(False)
        qapp.processEvents()
        assert win.view_hierarchy.left_widget.minimumWidth() == 0

        for dark in (False, True):
            win.act_dark_mode.setChecked(dark)
            qapp.processEvents()
            assert win.minimumSize().width() <= 250
            assert win.minimumSize().height() <= 200
            assert win.minimumSizeHint().width() <= 250
            assert win.minimumSizeHint().height() <= 200
            for tab_idx in range(win.tabs.count()):
                win.tabs.setCurrentIndex(tab_idx)
                for target_w, target_h in ((320, 240), (500, 350)):
                    win.resize(target_w, target_h)
                    qapp.processEvents()
                    assert (win.width(), win.height()) == (target_w, target_h)
        win.act_dark_mode.setChecked(False)

        # 4. Restoring to 1300x860 keeps primary plots and cut plots comfortably visible
        win.resize(1300, 860)
        win.tabs.setCurrentIndex(2)
        qapp.processEvents()
        det = win.view_detector
        assert det.plot_2d.width() > 300 and det.plot_2d.height() > 200
        assert det.plot_x_cut.height() >= 100 and det.plot_y_cut.height() >= 100
        assert det.lbl_img_coords.minimumWidth() == 290 and det.lbl_img_coords.maximumWidth() == 290

        win.tabs.setCurrentIndex(3)
        qapp.processEvents()
        off = win.view_offspec
        assert off.plot.width() > 300 and off.plot.height() > 200
        assert off.plot_row.height() >= 110 and off.plot_col.height() >= 110
    finally:
        win.view_offspec.shutdown()
        win.close()


def test_flow_layout_wraps_control_bars_without_overlap(qtbot, qapp, main_window: MainWindow):
    """Wide top control bars in DetectorView and OffSpecularView wrap smoothly without overlap."""
    from PyQt6.QtWidgets import QComboBox
    from pySAview.ui.flow_layout import FlowLayout

    main_window.show()
    qapp.processEvents()

    try:
        # Compact combo boxes must not bloat minimumSizeHint horizontally
        for combo in (
            main_window.combo_active_file,
            main_window.view_detector.combo_spin_filter,
            main_window.view_detector.combo_cmap,
            main_window.view_offspec.combo_spin,
            main_window.view_offspec.combo_y_source,
            main_window.view_offspec.combo_norm,
            main_window.view_offspec.combo_y_axis,
            main_window.view_offspec.combo_cmap,
        ):
            assert combo.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            assert combo.minimumSizeHint().width() < 300

        # Toggle optional asymmetry spinbox visible so FlowLayout wrapping is exercised with it too
        main_window.view_offspec.spin_min_counts.setVisible(True)

        for tab_idx in (2, 3):
            main_window.tabs.setCurrentIndex(tab_idx)
            view = main_window.tabs.widget(tab_idx)
            top_flow = view.layout().itemAt(0).layout()
            assert isinstance(top_flow, FlowLayout)

            heights = {}
            for width, height in ((1600, 860), (1300, 860), (500, 350)):
                main_window.resize(width, height)
                qapp.processEvents()
                heights[width] = top_flow.geometry().height()

                # No overlapping visible sibling widgets in any FlowLayout inside the tab
                for flow in view.findChildren(FlowLayout):
                    visible_rects = []
                    for i in range(flow.count()):
                        w = flow.itemAt(i).widget()
                        if w is not None and not w.isHidden() and w.isVisibleTo(view):
                            visible_rects.append(w.geometry())
                    for a in range(len(visible_rects)):
                        for b in range(a + 1, len(visible_rects)):
                            inter = visible_rects[a].intersected(visible_rects[b])
                            assert not (inter.width() > 0 and inter.height() > 0)

            assert heights[500] > heights[1600]
    finally:
        main_window.view_offspec.shutdown()
        main_window.close()






