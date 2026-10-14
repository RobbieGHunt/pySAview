"""Interactive X-Y scan plotter with spin state separation, log scaling, and normalization."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from matplotlib import colormaps
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sa_nxs_explorer.core.curve_colors import palette_range
from sa_nxs_explorer.core.waterfall import waterfall_order, display_values
from sa_nxs_explorer.core.models import ScanData
from sa_nxs_explorer.core.scan_calculations import (
    CurveStats,
    PlottedCurve,
    calculate_curve_stats,
    compute_q,
    split_by_flipper,
)
from sa_nxs_explorer.core.normalization import (
    is_count_channel, normalize_signal, log_error_bounds,
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

        self.control_tabs = QTabWidget()
        ctrl_layout.addWidget(self.control_tabs)
        processing_panel = QWidget()
        processing_layout = QVBoxLayout(processing_panel)
        processing_layout.setContentsMargins(4, 4, 4, 4)
        self.control_tabs.addTab(processing_panel, "Data & Processing")

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

        processing_layout.addWidget(grp_axis)

        # 2. Normalization & Spin Splitting
        grp_processing = QGroupBox("Processing & Polarization")
        proc_layout = QVBoxLayout(grp_processing)
        proc_layout.setSpacing(6)

        proc_layout.addWidget(QLabel("Normalization:"))
        self.combo_norm = QComboBox()
        self.combo_norm.addItem("Raw Counts (counts)", "raw")
        self.combo_norm.addItem("Counts / Monitor1 (unscaled ratio)", "monitor")
        self.combo_norm.addItem("Normalized to Time (cps)", "time")
        self.combo_norm.addItem("Monitor ratio × fixed reference rate", "monitor_time")
        self.combo_norm.currentIndexChanged.connect(self._on_axis_or_setting_changed)
        proc_layout.addWidget(self.combo_norm)

        self.lbl_reference_rate = QLabel("Reference monitor rate (monitor counts/s):")
        self.lbl_reference_rate.setWordWrap(True)
        self.spin_reference_rate = QDoubleSpinBox()
        self.spin_reference_rate.setRange(0.000001, 1e12)
        self.spin_reference_rate.setDecimals(6)
        self.spin_reference_rate.setValue(1.0)
        self.spin_reference_rate.valueChanged.connect(self._on_axis_or_setting_changed)
        self.spin_reference_rate.setToolTip("Fixed exact display scale shared by every overlaid scan; not estimated from each scan.")
        proc_layout.addWidget(self.lbl_reference_rate)
        proc_layout.addWidget(self.spin_reference_rate)
        self.lbl_reference_rate.hide()
        self.spin_reference_rate.hide()

        proc_layout.addWidget(QLabel("Counting uncertainty:"))
        self.combo_uncertainty = QComboBox()
        self.combo_uncertainty.addItem("Exact 68.27% count intervals", "exact")
        self.combo_uncertainty.addItem("Propagated standard uncertainty", "standard")
        self.combo_uncertainty.setToolTip("Exact intervals assume independent unweighted Poisson counts. Supplied variances use first-order propagation.")
        self.combo_uncertainty.currentIndexChanged.connect(self._on_axis_or_setting_changed)
        proc_layout.addWidget(self.combo_uncertainty)

        proc_layout.addWidget(QLabel("Polarization (Spin State):"))
        self.combo_spin = QComboBox()
        self.combo_spin.addItem("Split by Flipper1 (d / u)", "split")
        self.combo_spin.addItem("Combined (All points)", "all")
        self.combo_spin.addItem("d — Down only (Flipper OFF / 0)", "flipper0")
        self.combo_spin.addItem("u — Up only (Flipper ON / 1)", "flipper1")
        self.combo_spin.currentIndexChanged.connect(self._on_axis_or_setting_changed)
        proc_layout.addWidget(self.combo_spin)

        processing_layout.addWidget(grp_processing)
        processing_layout.addStretch()

        # 3. Plot Display Options
        grp_display = QGroupBox("Chart Display Options")
        disp_layout = QVBoxLayout(grp_display)
        disp_layout.setSpacing(6)

        self.spin_style_controls = {}
        for spin, title, default_map in (("d", "d / Down — Flipper OFF (0)", "Blues"),
                                          ("u", "u / Up — Flipper ON (1)", "Reds")):
            group = QGroupBox(title)
            form = QFormLayout(group)
            palette = QComboBox()
            palette.addItems(sorted(colormaps, key=str.casefold))
            palette.setCurrentText(default_map)
            palette.setToolTip("Matplotlib colormap for this spin family; loaded files use different shades.")
            preview = QLabel()
            preview.setFixedHeight(12)
            form.addRow("Colormap:", palette)
            form.addRow("Shades:", preview)
            line_style = QComboBox()
            line_style.addItem("Solid", "solid")
            line_style.addItem("Dashed", "dashed")
            form.addRow("Line style:", line_style)
            line_width = QDoubleSpinBox()
            line_width.setRange(0.5, 12)
            line_width.setDecimals(1)
            line_width.setSingleStep(0.5)
            line_width.setValue(2)
            line_width.setSuffix(" px")
            form.addRow("Line width:", line_width)
            marker_size = QDoubleSpinBox()
            marker_size.setRange(0, 24)
            marker_size.setDecimals(1)
            marker_size.setValue(5)
            marker_size.setSuffix(" px")
            marker_size.setToolTip("Zero hides markers for this spin channel.")
            form.addRow("Marker size:", marker_size)
            self.spin_style_controls[spin] = {"colormap": palette, "preview": preview,
                                               "line_style": line_style,
                                               "line_width": line_width,
                                               "marker_size": marker_size}
            disp_layout.addWidget(group)
        self.combo_down_colormap = self.spin_style_controls["d"]["colormap"]
        self.combo_up_colormap = self.spin_style_controls["u"]["colormap"]
        for controls in self.spin_style_controls.values():
            controls["colormap"].currentTextChanged.connect(self._on_axis_or_setting_changed)
            controls["line_style"].currentIndexChanged.connect(self._on_axis_or_setting_changed)
            controls["line_width"].valueChanged.connect(self._on_axis_or_setting_changed)
            controls["marker_size"].valueChanged.connect(self._on_axis_or_setting_changed)

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

        self.chk_errors = QCheckBox("Show Intervals / Uncertainties")
        self.chk_errors.setChecked(True)
        self.chk_errors.toggled.connect(self._on_axis_or_setting_changed)
        disp_layout.addWidget(self.chk_errors)

        self.chk_waterfall = QCheckBox("Waterfall: offset scans")
        self.chk_waterfall.setToolTip("Numbered scans ascend from the base; other files follow in load order. Both spins share an offset. Tables and exports are unchanged.")
        disp_layout.addWidget(self.chk_waterfall)
        self.combo_waterfall_mode = QComboBox()
        self.combo_waterfall_mode.addItem("Logarithmic (decades)", "log")
        self.combo_waterfall_mode.addItem("Linear (Y units)", "linear")
        self.spin_waterfall_spacing = QDoubleSpinBox()
        self.spin_waterfall_spacing.setDecimals(3)
        self.spin_waterfall_spacing.setRange(0.001, 1000000)
        self.spin_waterfall_spacing.setValue(1.0)
        waterfall_form = QFormLayout()
        waterfall_form.addRow("Offset mode", self.combo_waterfall_mode)
        waterfall_form.addRow("Spacing per scan", self.spin_waterfall_spacing)
        disp_layout.addLayout(waterfall_form)
        self.combo_waterfall_mode.setEnabled(False)
        self.spin_waterfall_spacing.setEnabled(False)
        self.chk_waterfall.toggled.connect(self._on_waterfall_changed)
        self.combo_waterfall_mode.currentIndexChanged.connect(self._on_waterfall_changed)
        self.spin_waterfall_spacing.valueChanged.connect(self._on_axis_or_setting_changed)

        disp_layout.addStretch()
        self.control_tabs.addTab(grp_display, "Chart Display Options")

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
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(ctrl_box)
        controls_scroll.setMinimumWidth(240)
        splitter.addWidget(controls_scroll)

        # ---------------- Right Panel: Plot & Data Tabs ----------------
        right_box = QWidget()
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Coordinate readout header
        header_row = QHBoxLayout()
        self.lbl_title = QLabel("Load a scan file to plot data.")
        self.lbl_title.setWordWrap(True)
        self.lbl_title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.lbl_title.setMinimumWidth(0)
        self.lbl_title.setTextFormat(Qt.TextFormat.PlainText)
        self.lbl_title.setStyleSheet("font-weight: 700; font-size: 14px; color: #0284c7;")
        header_row.addWidget(self.lbl_title, 1)

        self.lbl_coords = QLabel("Cursor: (X: -- , Y: --)")
        self.lbl_coords.setStyleSheet("color: #64748b; font-size: 12px;")
        self.lbl_coords.setFixedWidth(240)
        self.lbl_coords.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_row.addWidget(self.lbl_coords)
        right_layout.addLayout(header_row)
        self.lbl_processing = QLabel("Open a scan to inspect processing.")
        self.lbl_processing.setWordWrap(True)
        self.lbl_processing.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.lbl_processing.setMinimumWidth(0)
        self.lbl_processing.setTextFormat(Qt.TextFormat.PlainText)
        self.lbl_processing.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_processing.setStyleSheet("font-size: 12px; padding: 4px;")
        right_layout.addWidget(self.lbl_processing)

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
        self._update_palette_previews()
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
            self.lbl_coords.setText(f"Cursor: (X: {x_val:.4g}, {'Display Y' if self.chk_waterfall.isChecked() else 'Y'}: {y_val:.4g})")

    def _on_axis_or_setting_changed(self) -> None:
        # Toggle wavelength input visibility
        x_key = self.combo_x.currentData()
        is_q = x_key == "Q"
        self.lbl_wavelength.setVisible(is_q)
        self.spin_wavelength.setVisible(is_q)
        is_reference = self.combo_norm.currentData() == "monitor_time"
        self.lbl_reference_rate.setVisible(is_reference)
        self.spin_reference_rate.setVisible(is_reference)

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

        self._update_palette_previews()
        self._update_plot()

    def _on_waterfall_changed(self, *_args) -> None:
        enabled = self.chk_waterfall.isChecked()
        self.combo_waterfall_mode.setEnabled(enabled)
        self.spin_waterfall_spacing.setEnabled(enabled)
        if enabled:
            self.chk_log_y.setChecked(self.combo_waterfall_mode.currentData() == "log")
        self._on_axis_or_setting_changed()

    def _update_palette_previews(self) -> None:
        from matplotlib.colors import to_hex
        for controls in self.spin_style_controls.values():
            cmap = colormaps[controls["colormap"].currentText()]
            low, high = palette_range(self.is_dark_mode)
            stops = ", ".join(f"stop:{i/4:g} {to_hex(cmap(low + (high-low)*i/4))}" for i in range(5))
            controls["preview"].setStyleSheet(
                f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,{stops}); border-radius: 3px;")

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
            self.lbl_processing.setText("No datasets selected for plotting.")
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
            descriptions = []
            for _, s in checked_scans:
                rn = f"#{s.metadata.run_number}" if s.metadata.run_number else Path(s.file_path).name
                runs.append(rn if len(rn) <= 24 else rn[:21] + "…")
                descriptions.append(f"{rn}: {s.metadata.subtitle or s.metadata.title}")
            suffix = f" (+{len(runs)-5} more)" if len(runs) > 5 else ""
            title_text = f"Overlay: {len(runs)} scans — {', '.join(runs[:5])}{suffix}"
        self.lbl_title.setToolTip(title_text if len(checked_scans) == 1 else "\n".join(descriptions))
        self.lbl_title.setText(title_text if len(title_text) <= 160 else title_text[:157] + "…")

        x_key = self.combo_x.currentData()
        y_key = self.combo_y.currentData()
        if not x_key or not y_key:
            return

        norm_mode = self.combo_norm.currentData()
        spin_mode = self.combo_spin.currentData()
        is_multi = len(self.loaded_scans) > 1 or len(checked_scans) > 1

        all_curves: list[PlottedCurve] = []
        warnings = []
        methods = set()
        assumptions = set()
        excluded = 0
        x_label = x_key
        x_units = ""
        y_label = y_key
        y_units = ""
        expected_units = None

        def canonical_unit(unit):
            key = unit.strip()
            return {"degree": "deg", "degrees": "deg", "°": "deg",
                    "count": "counts", "cts": "counts", "sec": "s",
                    "seconds": "s", "second": "s"}.get(key.lower(), key)

        for idx, (k, scan) in enumerate(checked_scans):
            name = Path(scan.file_path).name
            try:
                if y_key not in scan.variables:
                    raise ValueError(f"missing selected Y channel {y_key}")
                y_var = scan.variables[y_key]
                raw_y = np.asarray(y_var.data, dtype=float)
                if x_key == "index":
                    x_data = np.arange(len(raw_y), dtype=float)
                    current_x_units = ""
                    current_x_label = "Point Index"
                elif x_key == "Q":
                    if "th" not in scan.variables:
                        raise ValueError("missing incident-angle channel th")
                    if canonical_unit(scan.variables["th"].unit) != "deg":
                        raise ValueError("Q conversion requires th in degrees")
                    x_data = compute_q(scan.variables["th"].data, self.spin_wavelength.value())
                    current_x_units = "Å⁻¹"
                    current_x_label = "Q (Momentum Transfer)"
                else:
                    if x_key not in scan.variables:
                        raise ValueError(f"missing selected X channel {x_key}")
                    x_var = scan.variables[x_key]
                    x_data = np.asarray(x_var.data, dtype=float)
                    current_x_units = x_var.unit
                    current_x_label = x_var.name
                current_units = (canonical_unit(current_x_units), canonical_unit(y_var.unit))
                if expected_units is not None and current_units != expected_units:
                    raise ValueError("selected channel units differ from the other plotted scans")
                if x_data.ndim != 1 or raw_y.ndim != 1 or len(x_data) != len(raw_y):
                    raise ValueError("X and Y must be matching 1D scan channels")

                mon_var = scan.variables.get("Monitor1")
                time_var = scan.variables.get("Count", scan.variables.get("Time"))
                if norm_mode in ("monitor", "monitor_time") and mon_var is not None and not is_count_channel(mon_var.name, mon_var.unit):
                    raise ValueError("Monitor1 is not a count channel")
                if norm_mode == "time" and time_var is not None and canonical_unit(time_var.unit) != "s":
                    raise ValueError("dwell-time channel must have units of seconds")
                if norm_mode == "time" and time_var is not None and time_var.unit_source not in ("file", "unspecified"):
                    assumptions.add(f"{time_var.name} in seconds: {time_var.unit_source}")
                result = normalize_signal(
                    raw_y, norm_mode,
                    monitor=mon_var.data if mon_var is not None else None,
                    time=time_var.data if time_var is not None else None,
                    count_data=is_count_channel(y_var.name, y_var.unit),
                    variance=y_var.variance,
                    monitor_variance=mon_var.variance if mon_var is not None else None,
                    uncertainty=self.combo_uncertainty.currentData(),
                    reference_monitor_rate=self.spin_reference_rate.value(),
                    numerator_is_monitor=y_key == "Monitor1",
                    input_units=y_var.unit,
                )
                valid = result.valid & np.isfinite(x_data)
                f_var = scan.variables.get("Flipper1")
                effective_spin = spin_mode
                if f_var is None:
                    if spin_mode in ("flipper0", "flipper1"):
                        raise ValueError("selected spin state requires Flipper1")
                    effective_spin = "all"
                    f_data = np.zeros(len(raw_y))
                else:
                    f_data = np.asarray(f_var.data, dtype=float)
                    if f_data.shape != raw_y.shape:
                        raise ValueError("Flipper1 length differs from the scan")
                    if spin_mode != "all":
                        valid &= np.isin(f_data, [0, 1])
                dropped = len(raw_y) - int(np.count_nonzero(valid))
                excluded += dropped
                if dropped:
                    why = sorted(set(result.reasons[~result.valid]))
                    if np.any(~np.isfinite(x_data)):
                        why.append("nonfinite X")
                    if effective_spin != "all" and np.any(~np.isin(f_data, [0, 1])):
                        why.append("invalid Flipper1 state")
                    warnings.append(f"{name}: {dropped} excluded ({', '.join(why)})")
                if not np.any(valid):
                    continue
                expected_units = current_units
                x_label, x_units, y_units = current_x_label, current_x_units, result.units
                methods.add(result.method)
                original_indices = np.flatnonzero(valid)
                curves = split_by_flipper(
                    x_data[valid], result.values[valid],
                    result.sigma[valid] if result.sigma is not None else None,
                    f_data[valid], spin_mode=effective_spin,
                    palette_index=waterfall_order([s.file_path for s in self.loaded_scans.values()]).index(scan.file_path),
                    dataset_count=len(self.loaded_scans),
                    down_colormap=self.combo_down_colormap.currentText(),
                    up_colormap=self.combo_up_colormap.currentText(),
                    dark=self.is_dark_mode,
                    dataset_name=f"#{scan.metadata.run_number}" if is_multi and scan.metadata.run_number else (name if is_multi else ""),
                )
                for curve in curves:
                    source_indices = original_indices[curve.indices]
                    curve.y_lower = result.lower[source_indices] if result.lower is not None else None
                    curve.y_upper = result.upper[source_indices] if result.upper is not None else None
                    curve.indices = source_indices
                    curve.uncertainty_method = result.method
                    curve.source_path = scan.file_path
                    curve.normalization = norm_mode
                    curve.reference_monitor_rate = self.spin_reference_rate.value() if norm_mode == "monitor_time" else None
                    curve.x_units, curve.y_units = x_units, y_units
                    curve.wavelength = self.spin_wavelength.value() if x_key == "Q" else None
                all_curves.extend(curves)
            except ValueError as error:
                warnings.append(f"{name}: skipped — {error}")

        self._current_curves = all_curves
        formulas = {"raw": "Y", "monitor": "N / Monitor1", "time": "N / dwell time",
                    "monitor_time": f"(N / Monitor1) × {self.spin_reference_rate.value():g} monitor counts/s (fixed reference)"}
        status = f"Processing: {formulas.get(norm_mode, norm_mode)}; {'; '.join(sorted(methods)) or 'no compatible data'}; {excluded} invalid points excluded."
        if x_key == "Q":
            status += f" Wavelength: {self.spin_wavelength.value():g} Å."
        hidden = sum(np.count_nonzero((c.x <= 0) if self.chk_log_x.isChecked() else np.zeros(len(c.x), dtype=bool))
                     + np.count_nonzero((c.y <= 0) if self.chk_log_y.isChecked() else np.zeros(len(c.y), dtype=bool)) for c in all_curves)
        if hidden:
            status += f" Log axes hide nonpositive coordinates ({hidden} coordinate values); retained in table/export."
        if self.chk_log_y.isChecked() and any(c.y_lower is not None and np.any(c.y_lower <= 0) for c in all_curves):
            status += " Nonpositive lower bounds cannot be drawn on a log axis; use linear view/table."
        self._processing_warnings = warnings
        if assumptions:
            status += " Unit assumptions: " + "; ".join(sorted(assumptions)) + "."
        if warnings:
            status += " " + " | ".join(warnings)
        waterfall = self.chk_waterfall.isChecked()
        offset_mode = self.combo_waterfall_mode.currentData()
        spacing = self.spin_waterfall_spacing.value()
        visible_paths = {c.source_path for c in all_curves}
        ordered_paths = waterfall_order([scan.file_path for scan in self.loaded_scans.values()
                                         if scan.file_path in visible_paths])
        ranks = {path: rank for rank, path in enumerate(ordered_paths)}
        if waterfall:
            units = "decades" if offset_mode == "log" else "Y units"
            status += f" Waterfall display: {spacing:g} {units} per scan, ascending scan number then unnumbered load order; spins share offsets. Table/export values are unshifted."
        self.lbl_processing.setText(status)

        # Clear existing items
        self.plot_widget.clear()
        if self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.clear()
            self.plot_widget.plotItem.legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-15, 15))

        # Labels
        text_color = "#f1f5f9" if self.is_dark_mode else "#1e293b"
        self.plot_widget.setLabel("bottom", x_label, units=x_units, color=text_color)
        self.plot_widget.setLabel("left", y_label + (" (waterfall display)" if waterfall else ""), units=y_units, color=text_color)

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
            rank = ranks.get(curve.source_path, 0) if waterfall else 0
            py = display_values(curve.y, rank, spacing, offset_mode) if waterfall else curve.y
            lower_display = display_values(curve.y_lower, rank, spacing, offset_mode) if waterfall else curve.y_lower
            upper_display = display_values(curve.y_upper, rank, spacing, offset_mode) if waterfall else curve.y_upper
            perr = curve.y_err

            if waterfall and np.any(np.isfinite(curve.y) & ~np.isfinite(py)):
                self.lbl_processing.setText(self.lbl_processing.text() + " Offset exceeds display range; reduce spacing. Overflowing points are hidden; original data are retained.")
            valid = np.isfinite(px) & np.isfinite(py)
            if log_x:
                valid &= (px > 0)
            if log_y:
                valid &= (py > 0)

            if not np.any(valid):
                continue

            cx = px[valid]
            cy = py[valid]
            cerr = perr[valid] if perr is not None else None

            controls = self.spin_style_controls.get(curve.spin)
            line_width = controls["line_width"].value() if controls else 2
            marker_size = controls["marker_size"].value() if controls else 5
            dashed = controls is not None and controls["line_style"].currentData() == "dashed"
            pen = pg.mkPen(curve.color, width=line_width,
                           style=Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine) if style in ("both", "line") else None
            symbol = curve.symbol if style in ("both", "points") and marker_size > 0 else None
            brush = curve.color if symbol is not None else None

            self.plot_widget.plot(
                cx,
                cy,
                pen=pen,
                symbol=symbol,
                symbolSize=marker_size,
                symbolPen=pg.mkPen(curve.color, width=1),
                symbolBrush=brush,
                name=curve.label + (f" [offset {rank * spacing:g} {'dec' if offset_mode == 'log' else 'Y'}]" if waterfall else ""),
            )

            # Exact confidence limits remain asymmetric. Standard uncertainty
            # is separately identified in the processing banner and export.
            if show_errors and curve.y_lower is not None and len(cx) > 0:
                lower = lower_display[valid]
                upper = upper_display[valid]
                x_pts = np.log10(cx) if log_x else cx
                if log_y:
                    y_pts = np.log10(cy)
                    top_err, bot_err = log_error_bounds(cy, lower, upper)
                else:
                    y_pts = cy
                    top_err, bot_err = upper - cy, cy - lower
                self.plot_widget.addItem(pg.ErrorBarItem(
                    x=x_pts, y=y_pts, top=top_err, bottom=bot_err,
                    beam=0, pen=pg.mkPen(curve.color, width=line_width),
                ))

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
        self.curve_data_table.setColumnCount(9)
        self.curve_data_table.setHorizontalHeaderLabels([
            "Dataset", "State / Curve", "Step Index", x_label, y_label, "Std uncertainty", "Lower bound", "Upper bound", "Uncertainty method"
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
                for col, value in ((6, c.y_lower), (7, c.y_upper)):
                    self.curve_data_table.setItem(cur_row, col, QTableWidgetItem(f"{value[i]:.6g}" if value is not None else "--"))
                self.curve_data_table.setItem(cur_row, 8, QTableWidgetItem(c.uncertainty_method))
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
                writer.writerow(["Dataset", "Curve", "OriginalStepIndex", "X", "Y", "StandardUncertainty", "LowerBound", "UpperBound", "UncertaintyMethod", "Normalization", "ReferenceMonitorRate", "XUnits", "YUnits", "WavelengthAngstrom", "SourceFile", "ProcessingWarnings", "ProcessingStatus"])
                for c in self._current_curves:
                    ds_name = c.dataset_name or "Default"
                    for i in range(len(c.x)):
                        idx = c.indices[i] if i < len(c.indices) else i
                        err = c.y_err[i] if c.y_err is not None and i < len(c.y_err) else ""
                        writer.writerow([ds_name, c.label, idx, c.x[i], c.y[i], err,
                                         c.y_lower[i] if c.y_lower is not None else "",
                                         c.y_upper[i] if c.y_upper is not None else "",
                                         c.uncertainty_method, c.normalization,
                                         c.reference_monitor_rate, c.x_units, c.y_units,
                                         c.wavelength, c.source_path,
                                         " | ".join(getattr(self, "_processing_warnings", [])), self.lbl_processing.text()])

            QMessageBox.information(self, "Export Succeeded", f"Exported {sum(len(c.x) for c in self._current_curves)} data points to {Path(file_path).name}.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to save CSV:\n{e}")
