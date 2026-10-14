"""Interactive X-Y scan plotter with spin state separation, log scaling, and normalization."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sa_nxs_explorer.core.models import ScanData
from sa_nxs_explorer.core.scan_calculations import (
    CurveStats,
    PlottedCurve,
    calculate_curve_stats,
    compute_normalized_y,
    compute_q,
    split_by_flipper,
)
from sa_nxs_explorer.ui import icons
from sa_nxs_explorer.ui.theme import apply_plot_theme


class ScanPlotView(QWidget):
    """Interactive Charting and Analysis View for scanned scan variables."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.loaded_scans: dict[str, ScanData] = {}
        self.scan_data: ScanData | None = None
        self.is_dark_mode: bool = False
        self._current_curves: list[PlottedCurve] = []
        self._overlay_keys: list[str] = []
        self._has_explicit_overlay_keys: bool = False

        # Compatibility list widget and buttons kept for test backwards-compatibility
        self.list_datasets = QListWidget()
        self.list_datasets.itemChanged.connect(self._on_dataset_item_changed)
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self._select_all_datasets)
        self.btn_select_none = QPushButton("Clear")
        self.btn_select_none.clicked.connect(self._select_none_datasets)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # ---------------- Left Panel: Controls ----------------
        ctrl_box = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_box)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(10)

        # 1. Axis Selection Group
        grp_axis = QGroupBox("Scan Variables (Axes)")
        axis_layout = QVBoxLayout(grp_axis)
        axis_layout.setSpacing(6)

        axis_layout.addWidget(QLabel("X Axis:"))
        self.combo_x = QComboBox()
        self.combo_x.currentIndexChanged.connect(self._on_axis_or_setting_changed)
        axis_layout.addWidget(self.combo_x)

        # Wavelength configuration for Q
        self.row_wavelength = QHBoxLayout()
        self.lbl_wavelength = QLabel("λ (Å):")
        self.spin_wavelength = QDoubleSpinBox()
        self.spin_wavelength.setRange(0.1, 50.0)
        self.spin_wavelength.setValue(5.21)  # Default for SuperADAM monochromatic beam
        self.spin_wavelength.setSingleStep(0.01)
        self.spin_wavelength.setDecimals(3)
        self.spin_wavelength.valueChanged.connect(self._on_axis_or_setting_changed)
        self.row_wavelength.addWidget(self.lbl_wavelength)
        self.row_wavelength.addWidget(self.spin_wavelength)
        axis_layout.addLayout(self.row_wavelength)

        axis_layout.addWidget(QLabel("Y Axis:"))
        self.combo_y = QComboBox()
        self.combo_y.currentIndexChanged.connect(self._on_axis_or_setting_changed)
        axis_layout.addWidget(self.combo_y)

        ctrl_layout.addWidget(grp_axis)

        # 2. Normalization & Spin Splitting
        grp_processing = QGroupBox("Processing & Polarization")
        proc_layout = QVBoxLayout(grp_processing)
        proc_layout.setSpacing(6)

        proc_layout.addWidget(QLabel("Normalization:"))
        self.combo_norm = QComboBox()
        self.combo_norm.addItem("Raw Counts (counts)", "raw")
        self.combo_norm.addItem("Normalized to Monitor1 (relative)", "monitor")
        self.combo_norm.addItem("Normalized to Time (cps)", "time")
        self.combo_norm.addItem("Normalized to Monitor & Time (cps)", "monitor_time")
        self.combo_norm.currentIndexChanged.connect(self._on_axis_or_setting_changed)
        proc_layout.addWidget(self.combo_norm)

        proc_layout.addWidget(QLabel("Polarization (Spin State):"))
        self.combo_spin = QComboBox()
        self.combo_spin.addItem("Split by Flipper1 (Up vs Down)", "split")
        self.combo_spin.addItem("Combined (All points)", "all")
        self.combo_spin.addItem("Spin Up only (Flipper 0)", "flipper0")
        self.combo_spin.addItem("Spin Down only (Flipper 1)", "flipper1")
        self.combo_spin.currentIndexChanged.connect(self._on_axis_or_setting_changed)
        proc_layout.addWidget(self.combo_spin)

        ctrl_layout.addWidget(grp_processing)

        # 3. Plot Display Options
        grp_display = QGroupBox("Chart Display Options")
        disp_layout = QVBoxLayout(grp_display)
        disp_layout.setSpacing(6)

        scales_row = QHBoxLayout()
        self.chk_log_y = QCheckBox("Log Y")
        self.chk_log_y.setChecked(True)
        self.chk_log_y.toggled.connect(self._on_axis_or_setting_changed)
        scales_row.addWidget(self.chk_log_y)

        self.chk_log_x = QCheckBox("Log X")
        self.chk_log_x.toggled.connect(self._on_axis_or_setting_changed)
        scales_row.addWidget(self.chk_log_x)
        disp_layout.addLayout(scales_row)

        disp_layout.addWidget(QLabel("Plot Style:"))
        self.combo_style = QComboBox()
        self.combo_style.addItem("Line + Points", "both")
        self.combo_style.addItem("Line Only", "line")
        self.combo_style.addItem("Points Only", "points")
        self.combo_style.currentIndexChanged.connect(self._on_axis_or_setting_changed)
        disp_layout.addWidget(self.combo_style)

        self.chk_errors = QCheckBox("Show Error Bars (±1σ)")
        self.chk_errors.setChecked(True)
        self.chk_errors.toggled.connect(self._on_axis_or_setting_changed)
        disp_layout.addWidget(self.chk_errors)

        ctrl_layout.addWidget(grp_display)

        # 4. Action Buttons
        self.btn_reset = QPushButton("Reset View / Auto-Scale")
        self.btn_reset.setIcon(icons.get_icon("maximize", is_dark=self.is_dark_mode))
        self.btn_reset.clicked.connect(self._reset_zoom)
        ctrl_layout.addWidget(self.btn_reset)

        self.btn_export_csv = QPushButton("Export Plotted Data (CSV)")
        self.btn_export_csv.setIcon(icons.get_icon("download", is_dark=self.is_dark_mode))
        self.btn_export_csv.clicked.connect(self._export_csv)
        ctrl_layout.addWidget(self.btn_export_csv)

        ctrl_layout.addStretch()
        splitter.addWidget(ctrl_box)

        # ---------------- Right Panel: Plot & Data Tabs ----------------
        right_box = QWidget()
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Coordinate readout header
        header_row = QHBoxLayout()
        self.lbl_title = QLabel("Load a scan file to plot data.")
        self.lbl_title.setStyleSheet("font-weight: 700; font-size: 14px; color: #0284c7;")
        header_row.addWidget(self.lbl_title, 1)

        self.lbl_coords = QLabel("Cursor: (X: -- , Y: --)")
        self.lbl_coords.setStyleSheet("color: #64748b; font-size: 12px;")
        self.lbl_coords.setFixedWidth(240)
        self.lbl_coords.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_row.addWidget(self.lbl_coords)
        right_layout.addLayout(header_row)

        # Interactive Plot Toolbar (Pan, Box Zoom, Zoom In, Zoom Out, Auto-scale, Log Toggles)
        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(6)

        self.btn_pan = QPushButton("Pan")
        self.btn_pan.setCheckable(True)
        self.btn_pan.setChecked(True)
        self.btn_pan.setToolTip("Pan Mode: Drag with left mouse button to pan chart")
        self.btn_pan.clicked.connect(self._set_mouse_mode_pan)
        toolbar_row.addWidget(self.btn_pan)

        self.btn_box_zoom = QPushButton("Box Zoom")
        self.btn_box_zoom.setCheckable(True)
        self.btn_box_zoom.setToolTip("Box Zoom Mode: Drag with left mouse button to draw a zoom rectangle")
        self.btn_box_zoom.clicked.connect(self._set_mouse_mode_box_zoom)
        toolbar_row.addWidget(self.btn_box_zoom)

        self.mouse_mode_group = QButtonGroup(self)
        self.mouse_mode_group.addButton(self.btn_pan)
        self.mouse_mode_group.addButton(self.btn_box_zoom)

        self.btn_zoom_in = QPushButton("Zoom In")
        self.btn_zoom_in.setToolTip("Zoom into chart center (+30%)")
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        toolbar_row.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("Zoom Out")
        self.btn_zoom_out.setToolTip("Zoom out from chart center (-30%)")
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        toolbar_row.addWidget(self.btn_zoom_out)

        self.btn_reset_view = QPushButton("Fit All")
        self.btn_reset_view.setToolTip("Auto-scale / Reset view to fit all data curves")
        self.btn_reset_view.clicked.connect(self._reset_zoom)
        toolbar_row.addWidget(self.btn_reset_view)

        toolbar_row.addSpacing(10)

        self.btn_tool_log_y = QPushButton("Log Y")
        self.btn_tool_log_y.setCheckable(True)
        self.btn_tool_log_y.setChecked(self.chk_log_y.isChecked())
        self.btn_tool_log_y.setToolTip("Toggle Logarithmic / Linear scale on Y axis")
        self.btn_tool_log_y.toggled.connect(self._on_tool_log_y_toggled)
        toolbar_row.addWidget(self.btn_tool_log_y)

        self.btn_tool_log_x = QPushButton("Log X")
        self.btn_tool_log_x.setCheckable(True)
        self.btn_tool_log_x.setChecked(self.chk_log_x.isChecked())
        self.btn_tool_log_x.setToolTip("Toggle Logarithmic / Linear scale on X axis")
        self.btn_tool_log_x.toggled.connect(self._on_tool_log_x_toggled)
        toolbar_row.addWidget(self.btn_tool_log_x)

        toolbar_row.addSpacing(10)

        self.btn_toggle_q = QPushButton("Axis: θ (deg)")
        self.btn_toggle_q.setCheckable(True)
        self.btn_toggle_q.setToolTip("Toggle horizontal axis between Incident Angle θ (deg) and Momentum Transfer Q (Å⁻¹)")
        self.btn_toggle_q.clicked.connect(self._toggle_theta_q)
        toolbar_row.addWidget(self.btn_toggle_q)

        toolbar_row.addStretch(1)
        right_layout.addLayout(toolbar_row)

        # pyqtgraph Plot
        self.plot_widget = pg.PlotWidget()
        self.legend = self.plot_widget.addLegend()
        self.legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-15, 15))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        apply_plot_theme(self.plot_widget, self.is_dark_mode)
        right_layout.addWidget(self.plot_widget, 3)

        # Bottom Panel: Statistics and Spreadsheet
        bottom_tabs = QTabWidget()

        # Tab 1: Curve Statistics
        tab_stats = QWidget()
        stats_layout = QVBoxLayout(tab_stats)
        stats_layout.setContentsMargins(4, 4, 4, 4)
        self.stats_table = QTableWidget(0, 7)
        self.stats_table.setHorizontalHeaderLabels([
            "Curve", "Points", "Peak X", "Peak Y", "Centroid X", "Y Sum (Integrated)", "Y Min / Max"
        ])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        stats_layout.addWidget(self.stats_table)
        bottom_tabs.addTab(tab_stats, "Curve Statistics")

        # Tab 2: Spreadsheet Data Table
        tab_data = QWidget()
        data_tbl_layout = QVBoxLayout(tab_data)
        data_tbl_layout.setContentsMargins(4, 4, 4, 4)
        self.curve_data_table = QTableWidget()
        data_tbl_layout.addWidget(self.curve_data_table)
        bottom_tabs.addTab(tab_data, "Plotted Data Points")

        right_layout.addWidget(bottom_tabs, 1)

        splitter.addWidget(right_box)
        splitter.setSizes([320, 760])

    def _format_scan_label(self, scan: ScanData) -> str:
        name = Path(scan.file_path).name if scan.file_path else "Scan"
        run_str = f"#{scan.metadata.run_number}" if scan.metadata.run_number else ""
        sub_str = f" - {scan.metadata.subtitle}" if scan.metadata.subtitle else ""
        if run_str:
            return f"{name} ({run_str}{sub_str})"
        return f"{name}{sub_str}"

    def set_overlay_keys(self, keys: list[str]) -> None:
        """Set list of file path keys to overlay in the plot."""
        self._overlay_keys = list(keys)
        self._has_explicit_overlay_keys = True
        self.list_datasets.blockSignals(True)
        for i in range(self.list_datasets.count()):
            item = self.list_datasets.item(i)
            k = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(Qt.CheckState.Checked if k in self._overlay_keys else Qt.CheckState.Unchecked)
        self.list_datasets.blockSignals(False)
        self._update_plot()

    def _select_all_datasets(self) -> None:
        self.list_datasets.blockSignals(True)
        for i in range(self.list_datasets.count()):
            self.list_datasets.item(i).setCheckState(Qt.CheckState.Checked)
        self.list_datasets.blockSignals(False)
        self._overlay_keys = list(self.loaded_scans.keys())
        self._has_explicit_overlay_keys = True
        self._update_plot()

    def _select_none_datasets(self) -> None:
        self.list_datasets.blockSignals(True)
        for i in range(self.list_datasets.count()):
            self.list_datasets.item(i).setCheckState(Qt.CheckState.Unchecked)
        self.list_datasets.blockSignals(False)
        self._overlay_keys = []
        self._has_explicit_overlay_keys = True
        self._update_plot()

    def _on_dataset_item_changed(self, item: QListWidgetItem) -> None:
        checked: list[str] = []
        for i in range(self.list_datasets.count()):
            it = self.list_datasets.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                k = it.data(Qt.ItemDataRole.UserRole)
                if k:
                    checked.append(k)
        self._overlay_keys = checked
        self._has_explicit_overlay_keys = True
        self._update_plot()

    def add_scan(self, scan_data: ScanData, make_checked: bool = True, make_active: bool = True) -> None:
        """Add or update a scan dataset in the multi-dataset overlay manager."""
        key = str(scan_data.file_path) if scan_data.file_path else f"scan_{len(self.loaded_scans)+1}"
        self.loaded_scans[key] = scan_data
        if make_active or self.scan_data is None:
            self.scan_data = scan_data

        if make_checked and key not in self._overlay_keys:
            self._overlay_keys.append(key)

        # Update or insert list item for backward compatibility
        self.list_datasets.blockSignals(True)
        found_item: QListWidgetItem | None = None
        for i in range(self.list_datasets.count()):
            it = self.list_datasets.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == key:
                found_item = it
                break

        if found_item is None:
            item = QListWidgetItem(self._format_scan_label(scan_data))
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if make_checked else Qt.CheckState.Unchecked)
            self.list_datasets.addItem(item)
        else:
            found_item.setText(self._format_scan_label(scan_data))
            if make_checked:
                found_item.setCheckState(Qt.CheckState.Checked)

        self.list_datasets.blockSignals(False)

        self._refresh_axis_combos()
        self._update_plot()

    def remove_scan(self, file_path: str) -> None:
        """Remove a scan dataset from the multi-dataset overlay manager."""
        key = str(file_path)
        if key in self.loaded_scans:
            del self.loaded_scans[key]

        if key in self._overlay_keys:
            self._overlay_keys.remove(key)

        self.list_datasets.blockSignals(True)
        for i in range(self.list_datasets.count()):
            it = self.list_datasets.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == key:
                self.list_datasets.takeItem(i)
                break
        self.list_datasets.blockSignals(False)

        if self.scan_data and (str(self.scan_data.file_path) == key):
            self.scan_data = next(iter(self.loaded_scans.values())) if self.loaded_scans else None

        self._refresh_axis_combos()
        self._update_plot()

    def clear_scans(self) -> None:
        """Clear all loaded scans and reset chart."""
        self.loaded_scans.clear()
        self.scan_data = None
        self._overlay_keys.clear()
        self._has_explicit_overlay_keys = False
        self.list_datasets.blockSignals(True)
        self.list_datasets.clear()
        self.list_datasets.blockSignals(False)
        self._refresh_axis_combos()
        self._update_plot()

    def set_data(self, scan_data: ScanData, select_y: str | None = None) -> None:
        """Load parsed scan data and configure axis choices (backward-compatible)."""
        self.add_scan(scan_data, make_checked=True, make_active=True)
        if select_y and self.combo_y.findData(select_y) >= 0:
            self.combo_y.setCurrentIndex(self.combo_y.findData(select_y))
            self._update_plot()

    def _refresh_axis_combos(self) -> None:
        prev_x = self.combo_x.currentData()
        prev_y = self.combo_y.currentData()

        self.combo_x.blockSignals(True)
        self.combo_y.blockSignals(True)
        self.combo_x.clear()
        self.combo_y.clear()

        if not self.loaded_scans:
            self.combo_x.blockSignals(False)
            self.combo_y.blockSignals(False)
            return

        all_vars: dict[str, Any] = {}
        has_th = False
        has_tth = False
        has_flipper = False

        for scan in self.loaded_scans.values():
            if "th" in scan.variables:
                has_th = True
            if "tth" in scan.variables:
                has_tth = True
            if "Flipper1" in scan.variables:
                has_flipper = True
            for k, v in scan.variables.items():
                if k not in all_vars:
                    all_vars[k] = v

        if has_th:
            self.combo_x.addItem("th (Incident Angle, deg)", "th")
            self.combo_x.addItem("Q (Momentum Transfer, Å⁻¹)", "Q")
        if has_tth:
            self.combo_x.addItem("tth (Two-Theta, deg)", "tth")
        self.combo_x.addItem("Point Index", "index")

        for name, var in all_vars.items():
            if name not in ("th", "tth"):
                lbl = f"{name} ({var.label})" if var.label and var.label != name else name
                if var.unit:
                    lbl += f" [{var.unit}]"
                self.combo_x.addItem(lbl, name)

        y_keys = list(all_vars.keys())
        if "ROI_Counts" in y_keys:
            y_keys.remove("ROI_Counts")
            y_keys.insert(0, "ROI_Counts")

        for name in y_keys:
            var = all_vars[name]
            lbl = f"{name} ({var.label})" if var.label and var.label != name else name
            if var.unit:
                lbl += f" [{var.unit}]"
            self.combo_y.addItem(lbl, name)

        if prev_x and self.combo_x.findData(prev_x) >= 0:
            self.combo_x.setCurrentIndex(self.combo_x.findData(prev_x))
        elif has_th:
            self.combo_x.setCurrentIndex(self.combo_x.findData("th"))

        if prev_y and self.combo_y.findData(prev_y) >= 0:
            self.combo_y.setCurrentIndex(self.combo_y.findData(prev_y))
        elif "ROI_Counts" in all_vars:
            self.combo_y.setCurrentIndex(self.combo_y.findData("ROI_Counts"))
        elif "Detector1" in all_vars:
            self.combo_y.setCurrentIndex(self.combo_y.findData("Detector1"))

        self.combo_spin.setEnabled(has_flipper)

        self.combo_x.blockSignals(False)
        self.combo_y.blockSignals(False)

    def update_roi_counts(
        self,
        roi_counts: np.ndarray,
        roi_bounds: tuple[float, float, float, float],
        file_path: str | None = None,
    ) -> None:
        """Update or insert ROI_Counts variable on the specified (or active) scan and re-plot."""
        target_scan: ScanData | None = None
        if file_path and file_path in self.loaded_scans:
            target_scan = self.loaded_scans[file_path]
        elif self.scan_data:
            target_scan = self.scan_data
        elif self.loaded_scans:
            target_scan = next(iter(self.loaded_scans.values()))

        if target_scan is None:
            return

        from sa_nxs_explorer.core.models import ScanVariable
        target_scan.detector_roi = roi_bounds
        target_scan.variables["ROI_Counts"] = ScanVariable(
            index=len(target_scan.variables),
            name="ROI_Counts",
            label="User ROI",
            unit="counts",
            property="Integrated ROI",
            axis=0,
            is_scanned=False,
            data=roi_counts,
        )
        self._refresh_axis_combos()
        if self.combo_y.findData("ROI_Counts") >= 0:
            self.combo_y.setCurrentIndex(self.combo_y.findData("ROI_Counts"))
        self._update_plot()

    def set_theme(self, is_dark: bool) -> None:
        """Update chart styling for theme."""
        self.is_dark_mode = is_dark
        self.btn_reset.setIcon(icons.get_icon("maximize", is_dark=is_dark))
        self.btn_export_csv.setIcon(icons.get_icon("download", is_dark=is_dark))
        self.btn_pan.setIcon(icons.get_icon("hand", is_dark=is_dark))
        self.btn_box_zoom.setIcon(icons.get_icon("box_select", is_dark=is_dark))
        self.btn_zoom_in.setIcon(icons.get_icon("zoom_in", is_dark=is_dark))
        self.btn_zoom_out.setIcon(icons.get_icon("zoom_out", is_dark=is_dark))
        self.btn_reset_view.setIcon(icons.get_icon("maximize", is_dark=is_dark))
        self.btn_tool_log_y.setIcon(icons.get_icon("activity", is_dark=is_dark))
        self.btn_tool_log_x.setIcon(icons.get_icon("activity", is_dark=is_dark))
        self.btn_toggle_q.setIcon(icons.get_icon("sliders", is_dark=is_dark))
        apply_plot_theme(self.plot_widget, is_dark)
        self._update_plot()

    def _toggle_theta_q(self) -> None:
        """Quickly toggle horizontal axis between Incident Angle θ and Momentum Transfer Q."""
        cur_x = self.combo_x.currentData()
        target = "th" if cur_x == "Q" else "Q"
        idx = self.combo_x.findData(target)
        if idx >= 0:
            self.combo_x.setCurrentIndex(idx)
        else:
            fallback = self.combo_x.findData("th" if target == "Q" else "Q")
            if fallback >= 0:
                self.combo_x.setCurrentIndex(fallback)

    def _set_mouse_mode_pan(self) -> None:
        self.btn_pan.setChecked(True)
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.PanMode)

    def _set_mouse_mode_box_zoom(self) -> None:
        self.btn_box_zoom.setChecked(True)
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.RectMode)

    def _zoom_in(self) -> None:
        self.plot_widget.getViewBox().scaleBy((0.7, 0.7))

    def _zoom_out(self) -> None:
        self.plot_widget.getViewBox().scaleBy((1.4, 1.4))

    def _on_tool_log_y_toggled(self, checked: bool) -> None:
        self.chk_log_y.setChecked(checked)

    def _on_tool_log_x_toggled(self, checked: bool) -> None:
        self.chk_log_x.setChecked(checked)

    def _reset_zoom(self) -> None:
        self.plot_widget.autoRange()

    def _on_mouse_moved(self, pos: Any) -> None:
        plot_item = self.plot_widget.getPlotItem()
        vb = plot_item.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x_val = mouse_point.x()
            y_val = mouse_point.y()
            if self.chk_log_y.isChecked():
                y_val = 10**y_val
            if self.chk_log_x.isChecked():
                x_val = 10**x_val
            self.lbl_coords.setText(f"Cursor: (X: {x_val:.4g}, Y: {y_val:.4g})")

    def _on_axis_or_setting_changed(self) -> None:
        # Toggle wavelength input visibility
        x_key = self.combo_x.currentData()
        is_q = x_key == "Q"
        self.lbl_wavelength.setVisible(is_q)
        self.spin_wavelength.setVisible(is_q)

        # Synchronize plot toolbar buttons
        self.btn_toggle_q.blockSignals(True)
        self.btn_toggle_q.setChecked(is_q)
        self.btn_toggle_q.setText("Axis: Q (Å⁻¹)" if is_q else "Axis: θ (deg)")
        self.btn_toggle_q.blockSignals(False)

        self.btn_tool_log_y.blockSignals(True)
        self.btn_tool_log_y.setChecked(self.chk_log_y.isChecked())
        self.btn_tool_log_y.blockSignals(False)

        self.btn_tool_log_x.blockSignals(True)
        self.btn_tool_log_x.setChecked(self.chk_log_x.isChecked())
        self.btn_tool_log_x.blockSignals(False)

        self._update_plot()

    def _update_plot(self) -> None:
        # Collect checked scans
        checked_scans: list[tuple[str, ScanData]] = []
        if self._has_explicit_overlay_keys:
            for k in self._overlay_keys:
                if k in self.loaded_scans:
                    checked_scans.append((k, self.loaded_scans[k]))
        else:
            for i in range(self.list_datasets.count()):
                item = self.list_datasets.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    k = item.data(Qt.ItemDataRole.UserRole)
                    if k in self.loaded_scans:
                        checked_scans.append((k, self.loaded_scans[k]))

        # Fallback only if no explicit keys/items yet, but self.scan_data is set
        if not checked_scans and not self._has_explicit_overlay_keys and self.list_datasets.count() == 0 and self.scan_data is not None:
            k = str(self.scan_data.file_path) if self.scan_data.file_path else "default"
            if k in self.loaded_scans:
                checked_scans.append((k, self.loaded_scans[k]))
            elif self.loaded_scans:
                checked_scans.append(next(iter(self.loaded_scans.items())))

        if not checked_scans:
            self.lbl_title.setText("No Datasets Selected")
            self.plot_widget.clear()
            if self.plot_widget.plotItem.legend is not None:
                self.plot_widget.plotItem.legend.clear()
                self.plot_widget.plotItem.legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-15, 15))
            self._current_curves = []
            self.stats_table.setRowCount(0)
            self.curve_data_table.setRowCount(0)
            return

        # Title header
        if len(checked_scans) == 1:
            meta = checked_scans[0][1].metadata
            scan_desc = meta.subtitle or meta.title
            title_text = f"Scan #{meta.run_number}: {scan_desc}" if meta.run_number else (scan_desc or "Scan Data")
        else:
            runs = []
            for _, s in checked_scans:
                rn = f"#{s.metadata.run_number}" if s.metadata.run_number else Path(s.file_path).name
                if s.metadata.subtitle:
                    rn += f" ({s.metadata.subtitle})"
                runs.append(rn)
            title_text = f"Multi-Scan Overlay: {', '.join(runs)} ({len(checked_scans)} datasets)"
        self.lbl_title.setText(title_text)

        x_key = self.combo_x.currentData()
        y_key = self.combo_y.currentData()
        if not x_key or not y_key:
            return

        norm_mode = self.combo_norm.currentData()
        spin_mode = self.combo_spin.currentData()
        is_multi = len(self.loaded_scans) > 1 or len(checked_scans) > 1

        all_curves: list[PlottedCurve] = []
        x_label = x_key
        x_units = ""
        y_label = y_key

        for idx, (k, scan) in enumerate(checked_scans):
            ds_name = ""
            if is_multi:
                ds_name = f"#{scan.metadata.run_number}" if scan.metadata.run_number else Path(scan.file_path).name

            # X Data
            if x_key == "index":
                n_pts = len(next(iter(scan.variables.values())).data) if scan.variables else 0
                x_data = np.arange(n_pts, dtype=float)
                x_label = "Point Index"
                x_units = ""
            elif x_key == "Q":
                if "th" not in scan.variables:
                    continue
                th_data = scan.variables["th"].data
                x_data = compute_q(th_data, self.spin_wavelength.value())
                x_label = "Q (Momentum Transfer)"
                x_units = "Å⁻¹"
            else:
                if x_key not in scan.variables:
                    continue
                x_var = scan.variables[x_key]
                x_data = np.asarray(x_var.data, dtype=float)
                x_label = x_var.name
                x_units = x_var.unit

            # Y Data
            if y_key in scan.variables:
                y_var = scan.variables[y_key]
                raw_y = np.asarray(y_var.data, dtype=float)
                y_label = y_var.name
            elif "Detector1" in scan.variables:
                y_var = scan.variables["Detector1"]
                raw_y = np.asarray(y_var.data, dtype=float)
                y_label = "Detector1"
            else:
                continue

            # Normalization
            mon_data = scan.variables.get("Monitor1").data if "Monitor1" in scan.variables else None
            time_data = scan.variables.get("Count").data if "Count" in scan.variables else None

            y_data, y_err = compute_normalized_y(
                raw_y,
                norm_mode=norm_mode,
                monitor_data=mon_data,
                time_data=time_data,
            )

            # Spin splitting
            flipper_data = scan.variables.get("Flipper1").data if "Flipper1" in scan.variables else np.zeros_like(x_data)
            curves = split_by_flipper(
                x_data,
                y_data,
                y_err,
                flipper_data,
                spin_mode=spin_mode,
                palette_index=idx,
                dataset_name=ds_name,
            )
            all_curves.extend(curves)

        self._current_curves = all_curves

        # Format Y axis label
        if norm_mode == "monitor":
            y_label += " / Monitor1"
        elif norm_mode == "time":
            y_label += " / Time (cps)"
        elif norm_mode in ("monitor_time", "monitor_cps"):
            y_label += " / Monitor1 (cps)"

        # Clear existing items
        self.plot_widget.clear()
        if self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.clear()
            self.plot_widget.plotItem.legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-15, 15))

        # Labels
        text_color = "#f1f5f9" if self.is_dark_mode else "#1e293b"
        self.plot_widget.setLabel("bottom", x_label, units=x_units, color=text_color)
        self.plot_widget.setLabel("left", y_label, color=text_color)

        bot_ax = self.plot_widget.getPlotItem().getAxis("bottom")
        if bot_ax is not None:
            bot_ax.enableAutoSIPrefix(False)
            bot_ax.setSIPrefixEnableRanges(())
            bot_ax.autoSIPrefixScale = 1.0
            bot_ax.labelUnitPrefix = ""
            bot_ax._updateLabel()

        log_x = self.chk_log_x.isChecked()
        log_y = self.chk_log_y.isChecked()
        self.plot_widget.setLogMode(x=log_x, y=log_y)

        style = self.combo_style.currentData()
        show_errors = self.chk_errors.isChecked()

        # Render curves
        for curve in all_curves:
            px = curve.x
            py = curve.y
            perr = curve.y_err

            valid = np.ones(len(px), dtype=bool)
            if log_x:
                valid &= (px > 0)
            if log_y:
                valid &= (py > 0)

            if not np.any(valid):
                continue

            cx = px[valid]
            cy = py[valid]
            cerr = perr[valid] if perr is not None else None

            pen = pg.mkPen(curve.color, width=2) if style in ("both", "line") else None
            symbol = "o" if style in ("both", "points") else None
            brush = curve.color if style in ("both", "points") else None

            self.plot_widget.plot(
                cx,
                cy,
                pen=pen,
                symbol=symbol,
                symbolSize=4,
                symbolBrush=brush,
                name=curve.label,
            )

            # Error bars (±1σ) rendered accurately in both linear and log scale
            if show_errors and cerr is not None and len(cx) > 0:
                if log_x:
                    x_pts = np.log10(cx)
                else:
                    x_pts = cx

                if log_y:
                    # Map to ViewBox log10 space with asymmetric Poisson bounds
                    y_pts = np.log10(cy)
                    cy_safe = np.maximum(cy, 1e-12)
                    rel_err = cerr / cy_safe
                    top_err = np.log10(1.0 + rel_err)
                    # Safe relative lower bound (capped at 1.5 decades down for zero-count drops)
                    lower_ratio = np.maximum(1.0 - rel_err, 0.01)
                    bot_err = np.minimum(-np.log10(lower_ratio), 1.5)
                else:
                    y_pts = cy
                    top_err = cerr
                    bot_err = cerr

                err_item = pg.ErrorBarItem(
                    x=x_pts,
                    y=y_pts,
                    top=top_err,
                    bottom=bot_err,
                    beam=0,  # No caps on the error bars, just vertical lines
                    pen=pg.mkPen(curve.color, width=2),  # Keep error bar line width same as regular line width
                )
                self.plot_widget.addItem(err_item)

        apply_plot_theme(self.plot_widget, self.is_dark_mode)

        # Update curve stats and spreadsheet
        self._update_stats_table(all_curves)
        self._update_data_table(all_curves, x_label, y_label)

    def _update_stats_table(self, curves: list[PlottedCurve]) -> None:
        self.stats_table.setRowCount(len(curves))
        for row, c in enumerate(curves):
            st = calculate_curve_stats(c.x, c.y)
            self.stats_table.setItem(row, 0, QTableWidgetItem(c.label))
            self.stats_table.setItem(row, 1, QTableWidgetItem(str(st.n_points)))
            self.stats_table.setItem(row, 2, QTableWidgetItem(f"{st.peak_x:.4g}"))
            self.stats_table.setItem(row, 3, QTableWidgetItem(f"{st.peak_y:.4g}"))
            self.stats_table.setItem(row, 4, QTableWidgetItem(f"{st.centroid_x:.4g}"))
            self.stats_table.setItem(row, 5, QTableWidgetItem(f"{st.y_sum:.4g}"))
            self.stats_table.setItem(row, 6, QTableWidgetItem(f"{st.y_min:.4g} / {st.y_max:.4g}"))

    def _update_data_table(self, curves: list[PlottedCurve], x_label: str, y_label: str) -> None:
        total_rows = sum(len(c.x) for c in curves)
        self.curve_data_table.setRowCount(total_rows)
        self.curve_data_table.setColumnCount(6)
        self.curve_data_table.setHorizontalHeaderLabels([
            "Dataset", "State / Curve", "Step Index", x_label, y_label, "Error (±1σ)"
        ])
        self.curve_data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        cur_row = 0
        for c in curves:
            ds_name = c.dataset_name or "Default"
            for i in range(len(c.x)):
                orig_idx = c.indices[i] if i < len(c.indices) else i
                err_val = f"{c.y_err[i]:.4g}" if c.y_err is not None and i < len(c.y_err) else "--"
                self.curve_data_table.setItem(cur_row, 0, QTableWidgetItem(ds_name))
                self.curve_data_table.setItem(cur_row, 1, QTableWidgetItem(c.label))
                self.curve_data_table.setItem(cur_row, 2, QTableWidgetItem(str(orig_idx)))
                self.curve_data_table.setItem(cur_row, 3, QTableWidgetItem(f"{c.x[i]:.6g}"))
                self.curve_data_table.setItem(cur_row, 4, QTableWidgetItem(f"{c.y[i]:.6g}"))
                self.curve_data_table.setItem(cur_row, 5, QTableWidgetItem(err_val))
                cur_row += 1

    def _export_csv(self) -> None:
        if not self._current_curves:
            QMessageBox.information(self, "No Data", "No curves currently plotted to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Plotted Data as CSV", "scan_curve_data.csv", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Dataset", "Curve", "OriginalStepIndex", "X", "Y", "Y_Error"])
                for c in self._current_curves:
                    ds_name = c.dataset_name or "Default"
                    for i in range(len(c.x)):
                        idx = c.indices[i] if i < len(c.indices) else i
                        err = c.y_err[i] if c.y_err is not None and i < len(c.y_err) else 0.0
                        writer.writerow([ds_name, c.label, idx, c.x[i], c.y[i], err])

            QMessageBox.information(self, "Export Succeeded", f"Exported {sum(len(c.x) for c in self._current_curves)} data points to {Path(file_path).name}.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to save CSV:\n{e}")
