"""UI tests with pytest-qt for SuperADAM NeXus Explorer."""

from pathlib import Path
import pyqtgraph as pg
import pytest
from PyQt6.QtCore import Qt

from sa_nxs_explorer.ui.main_window import MainWindow


@pytest.fixture
def example_nxs_path() -> Path:
    p = Path("sa_nxs_explorer/example_data/005572.nxs")
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
    assert main_window.tabs.count() == 4
    tab_titles = [main_window.tabs.tabText(i) for i in range(4)]
    assert "File Hierarchy" in tab_titles[0]
    assert "Scan Charts" in tab_titles[1]
    assert "2D Detector" in tab_titles[2]
    assert "Metadata & Motors" in tab_titles[3]

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
    assert view.stats_table.rowCount() == 2
    peak_y = float(view.stats_table.item(0, 3).text())
    assert peak_y > 0

    # Check curve data table
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
    assert "Step 16 of 302" in view.lbl_step_info.text()
    assert "Mirrored" in view.lbl_step_info.text()

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
    assert view.btn_img_log.isChecked() is True
    view.btn_img_log.click()
    assert view.chk_log_img.isChecked() is False
    view.btn_img_log.click()
    assert view.chk_log_img.isChecked() is True

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
    p2 = Path("sa_nxs_explorer/example_data/005594.nxs")
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
    assert det.combo_spin_filter.count() == 3
    assert det.combo_spin_filter.isEnabled() is True

    total = det._total_frames
    assert total > 0

    # Test Spin Up (F=1)
    idx_up = det.combo_spin_filter.findData("up")
    det.combo_spin_filter.setCurrentIndex(idx_up)
    assert len(det._filtered_indices) == total // 2
    assert all(det.scan_data.variables["Flipper1"].data[i] == 1 for i in det._filtered_indices)
    assert "u — Up (F=1 / ON)" in det.lbl_step_info.text()
    det.btn_next.click()
    assert det._current_filtered_pos == 1
    assert "u — Up (F=1 / ON)" in det.lbl_step_info.text()

    # Test Spin Down (F=0)
    idx_down = det.combo_spin_filter.findData("down")
    det.combo_spin_filter.setCurrentIndex(idx_down)
    assert len(det._filtered_indices) == total // 2
    assert all(det.scan_data.variables["Flipper1"].data[i] == 0 for i in det._filtered_indices)
    assert "d — Down (F=0 / OFF)" in det.lbl_step_info.text()

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
    assert "θ" in charts.btn_toggle_q.text()

    # Toggle to Q
    charts.btn_toggle_q.click()
    assert charts.combo_x.currentData() == "Q"
    assert "Q" in charts.btn_toggle_q.text()
    assert charts.btn_toggle_q.isChecked() is True

    # Verify SI prefix is disabled on bottom axis
    bot_ax = charts.plot_widget.getPlotItem().getAxis("bottom")
    assert bot_ax.autoSIPrefix is False
    assert bot_ax.labelUnitPrefix == ""
    assert "mÅ" not in bot_ax.labelString()

    # Toggle back to theta
    charts.btn_toggle_q.click()
    assert charts.combo_x.currentData() == "th"
    assert "θ" in charts.btn_toggle_q.text()
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



