"""2D Detector Frame Viewer with horizontal mirroring, interactive user ROI, and scan chart integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pySAview.core.models import ScanData
from pySAview.core.off_specular import _written_frames, pair_spin_frames
from pySAview.core.reduction.background import (
    default_bkg_x_bounds,
    resolve_bkg_roi,
    roi_area_ratio,
    rois_overlap_2d,
)
from pySAview.core.roi_record import ROIRecord, save_roi, load_roi
from pySAview.core.nexus_reader import NexusReader
from pySAview.core.scan_calculations import integrate_roi_2d
from pySAview.core.roi import pixel_bounds
from pySAview.ui import icons
from pySAview.ui.flow_layout import (
    FlowContainer,
    FlowLayout,
    configure_compact_combo,
    fit_combo_popup,
    make_flow_group,
)
from pySAview.ui.plot_tools import export_plot_image, install_context_export, tool_button
from pySAview.ui.theme import apply_plot_theme, get_safe_colormap
from pySAview.ui.wheel_guard import install_wheel_focus_guard


class _CompactPlotWidget(pg.PlotWidget):
    """PlotWidget with a compact preferred sizeHint so side-panel scroll areas do not inflate."""

    def sizeHint(self) -> QSize:
        return QSize(180, 120)


class DetectorView(QWidget):
    """Visualizes 2D detector frames oriented and mirrored with interactive user-defined ROI."""

    # Signal emitted when user applies a new ROI: passes (roi_counts_array, (x0, x1, y0, y1))
    roi_applied = pyqtSignal(object, tuple)
    # Signal emitted when background ROI state or X bounds change for the active scan
    bkg_roi_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.reader: NexusReader | None = None
        self.scan_data: ScanData | None = None
        self.session_source_paths = lambda: []  # every loaded file; set by the main window
        self.is_dark_mode: bool = False
        self._current_frame_idx: int = 0
        self._total_frames: int = 0
        self._frame_data: np.ndarray | None = None
        self._frame_dirty: bool = False
        self._frame_total_counts: float | None = None
        self._loaded_cmap_name: str = "turbo"
        self._filtered_indices: list[int] = []
        self._paired_frames: dict[int, int] = {}
        self._current_filtered_pos: int = 0

        # Animation timer for scan play
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(120)  # ~8 fps
        self._play_timer.timeout.connect(self._step_forward)

        self._roi_rect_item: pg.RectROI | None = None
        self._bkg_roi_rect_item: pg.RectROI | None = None
        self._updating_roi_internally: bool = False
        self._updating_bkg_internally: bool = False

        # Per-frame auto-fit tracking state
        self._per_frame_rois: list[tuple[float, float, float, float]] | None = None
        self._per_frame_counts: np.ndarray | None = None
        self._auto_fit_active: bool = False

        self._init_ui()

    def _init_ui(self) -> None:
        self.setMinimumSize(0, 0)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # ---------------- Top Controls / Scrubber Bar ----------------
        ctrl_bar = FlowContainer(margin=0, h_spacing=8, v_spacing=6)

        self.btn_play = tool_button("Play / pause through the frames")
        self.btn_play.setIcon(icons.get_icon("play", is_dark=self.is_dark_mode))
        self.btn_play.clicked.connect(self._toggle_play)

        self.btn_prev = tool_button("Previous frame")
        self.btn_prev.setIcon(icons.get_icon("chevron_left", is_dark=self.is_dark_mode))
        self.btn_prev.clicked.connect(self._step_backward)

        self.btn_next = tool_button("Next frame")
        self.btn_next.setIcon(icons.get_icon("chevron_right", is_dark=self.is_dark_mode))
        self.btn_next.clicked.connect(self._step_forward)
        for btn in (self.btn_play, self.btn_prev, self.btn_next):
            ctrl_bar.addWidget(btn)

        self.combo_spin_filter = configure_compact_combo(QComboBox(), min_chars=14)
        self.combo_spin_filter.addItem("All Frames", "all")
        self.combo_spin_filter.addItem("u — Spin Up (Flipper OFF / 1)", "up")
        self.combo_spin_filter.addItem("d — Spin Down (Flipper ON / 0)", "down")
        self.combo_spin_filter.addItem("u + d — Sum of channels", "sum")
        fit_combo_popup(self.combo_spin_filter)
        self.combo_spin_filter.currentIndexChanged.connect(self._on_spin_filter_changed)
        ctrl_bar.addWidget(make_flow_group(QLabel("Spin:"), self.combo_spin_filter, spacing=6))

        self.spin_step = QSpinBox()
        self.spin_step.setRange(0, 0)
        self.spin_step.setFixedWidth(70)
        self.spin_step.valueChanged.connect(self._on_step_spin_changed)

        self.slider_step = QSlider(Qt.Orientation.Horizontal)
        self.slider_step.setRange(0, 0)
        self.slider_step.setMinimumWidth(40)
        self.slider_step.valueChanged.connect(self._on_slider_changed)
        ctrl_bar.addWidget(
            make_flow_group(QLabel("Step:"), self.spin_step, self.slider_step, spacing=6, expanding=True)
        )

        self.chk_log_img = QCheckBox("Log Scale")
        self.chk_log_img.setChecked(True)
        self.chk_log_img.toggled.connect(self._refresh_current_frame)
        ctrl_bar.addWidget(self.chk_log_img)

        self.chk_show_roi = QCheckBox("Show ROI")
        self.chk_show_roi.setChecked(True)
        self.chk_show_roi.toggled.connect(self._toggle_roi_visibility)
        ctrl_bar.addWidget(self.chk_show_roi)

        self.combo_cmap = configure_compact_combo(QComboBox(), min_chars=8)
        self.combo_cmap.addItems(["turbo", "viridis", "inferno", "plasma", "cividis", "gray", "magma"])
        self.combo_cmap.setCurrentText("turbo")
        fit_combo_popup(self.combo_cmap)
        self.combo_cmap.currentTextChanged.connect(self._on_cmap_changed)
        ctrl_bar.addWidget(make_flow_group(QLabel("Colormap:"), self.combo_cmap, spacing=6))

        main_layout.addWidget(ctrl_bar)

        # ---------------- Central Splitter: 2D Image + ROI Controls & 1D Profiles ----------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(True)
        splitter.setMinimumSize(0, 0)
        splitter.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        # Left / Center: 2D Detector Image
        img_container = QWidget()
        img_container.setMinimumSize(0, 0)
        img_container.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        img_layout = QVBoxLayout(img_container)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.setSpacing(6)

        # Interactive 2D Image Toolbar
        img_toolbar = FlowContainer(margin=0, h_spacing=6, v_spacing=4)

        # Icon-only tools, matching the Scan Charts and Off-Specular toolbars.
        self.btn_img_pan = tool_button("Pan: drag to move the detector image", checkable=True)
        self.btn_img_pan.setChecked(True)
        self.btn_img_pan.clicked.connect(self._set_img_mouse_mode_pan)
        self.btn_img_box_zoom = tool_button("Box zoom: drag a rectangle to zoom", checkable=True)
        self.btn_img_box_zoom.clicked.connect(self._set_img_mouse_mode_box_zoom)
        self.img_mouse_group = QButtonGroup(self)
        self.img_mouse_group.addButton(self.btn_img_pan)
        self.img_mouse_group.addButton(self.btn_img_box_zoom)
        self.btn_img_zoom_in = tool_button("Zoom in")
        self.btn_img_zoom_in.clicked.connect(self._img_zoom_in)
        self.btn_img_zoom_out = tool_button("Zoom out")
        self.btn_img_zoom_out.clicked.connect(self._img_zoom_out)
        self.btn_img_reset = tool_button("Fit the whole frame")
        self.btn_img_reset.clicked.connect(self._img_reset_view)
        self.btn_img_export = tool_button("Export the frame as a PNG or SVG image")
        self.btn_img_export.clicked.connect(self.export_image)
        for btn in (
            self.btn_img_pan,
            self.btn_img_box_zoom,
            self.btn_img_zoom_in,
            self.btn_img_zoom_out,
            self.btn_img_reset,
            self.btn_img_export,
        ):
            img_toolbar.addWidget(btn)

        # Per-frame angles, spin and counts (the step itself is in the controls above).
        self.lbl_frame_info = QLabel("")
        self.lbl_frame_info.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_frame_info.setMinimumWidth(0)
        self.lbl_frame_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        img_toolbar.addWidget(self.lbl_frame_info)

        self.lbl_img_coords = QLabel("Pixel: (X: ----, Y: ----)  |  Counts: --------")
        self.lbl_img_coords.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_img_coords.setFixedWidth(290)
        self.lbl_img_coords.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        img_toolbar.addWidget(self.lbl_img_coords)

        img_layout.addWidget(img_toolbar)

        self.plot_2d = pg.PlotWidget()
        self.plot_2d.setMinimumSize(0, 0)
        self.plot_2d.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.plot_2d.setAspectLocked(True)
        self.plot_2d.showGrid(x=False, y=False)
        self.plot_2d.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.plot_2d.getViewBox().setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.plot_2d.scene().sigMouseMoved.connect(self._on_mouse_moved_2d)
        self.img_item = pg.ImageItem()
        self.img_item.setColorMap(get_safe_colormap("turbo"))
        self.plot_2d.addItem(self.img_item)

        apply_plot_theme(self.plot_2d, self.is_dark_mode)
        img_layout.addWidget(self.plot_2d, 1)
        splitter.addWidget(img_container)

        # Right: Interactive ROI Definition & 1D Cuts (scrollable when compressed)
        sidebar_container = QWidget()
        sidebar_container.setMinimumSize(0, 0)
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        # 1. Interactive ROI Definition Card
        roi_ctrl_group = QGroupBox("Define Region of Interest (ROI)")
        roi_ctrl_group.setMinimumSize(0, 0)
        roi_ctrl_layout = QVBoxLayout(roi_ctrl_group)
        roi_ctrl_layout.setSpacing(8)

        inputs_grid = QHBoxLayout()
        inputs_grid.setSpacing(6)

        # X bounds
        vbox_x = QVBoxLayout()
        vbox_x.addWidget(QLabel("X Min:"))
        self.spin_x0 = QSpinBox()
        self.spin_x0.setMinimumWidth(55)
        self.spin_x0.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_x0.setRange(0, 1023)
        self.spin_x0.valueChanged.connect(self._on_spin_roi_changed)
        vbox_x.addWidget(self.spin_x0)

        vbox_x.addWidget(QLabel("X Max:"))
        self.spin_x1 = QSpinBox()
        self.spin_x1.setMinimumWidth(55)
        self.spin_x1.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_x1.setRange(0, 1024)
        self.spin_x1.valueChanged.connect(self._on_spin_roi_changed)
        vbox_x.addWidget(self.spin_x1)
        inputs_grid.addLayout(vbox_x)

        # Y bounds
        vbox_y = QVBoxLayout()
        vbox_y.addWidget(QLabel("Y Min:"))
        self.spin_y0 = QSpinBox()
        self.spin_y0.setMinimumWidth(55)
        self.spin_y0.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_y0.setRange(0, 1023)
        self.spin_y0.valueChanged.connect(self._on_spin_roi_changed)
        vbox_y.addWidget(self.spin_y0)

        vbox_y.addWidget(QLabel("Y Max:"))
        self.spin_y1 = QSpinBox()
        self.spin_y1.setMinimumWidth(55)
        self.spin_y1.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_y1.setRange(0, 1024)
        self.spin_y1.valueChanged.connect(self._on_spin_roi_changed)
        vbox_y.addWidget(self.spin_y1)
        inputs_grid.addLayout(vbox_y)

        roi_ctrl_layout.addLayout(inputs_grid)

        # Action buttons for ROI
        btns_row = FlowLayout(margin=0, h_spacing=6, v_spacing=6)
        self.btn_apply_roi = QPushButton("Apply ROI to Scan Chart")
        self.btn_apply_roi.setIcon(icons.get_icon("chart", is_dark=self.is_dark_mode))
        self.btn_apply_roi.setStyleSheet("font-weight: 700; color: #0284c7;")
        self.btn_apply_roi.clicked.connect(self._apply_roi_to_scan_chart)
        btns_row.addWidget(self.btn_apply_roi)

        self.btn_auto_fit = QPushButton("Auto-Fit ROI")
        self.btn_auto_fit.setIcon(icons.get_icon("sparkles", is_dark=self.is_dark_mode))
        self.btn_auto_fit.setStyleSheet("font-weight: 600;")
        self.btn_auto_fit.setToolTip("Automatically adjust ROI to center on reflection peak across frames")
        self.btn_auto_fit.clicked.connect(self._on_auto_fit_clicked)
        btns_row.addWidget(self.btn_auto_fit)

        self.btn_reset_roi = QPushButton("Reset to File ROI")
        self.btn_reset_roi.setIcon(icons.get_icon("refresh_cw", is_dark=self.is_dark_mode))
        self.btn_reset_roi.clicked.connect(self._reset_to_file_roi)
        btns_row.addWidget(self.btn_reset_roi)
        roi_ctrl_layout.addLayout(btns_row)

        sidecar_row = FlowLayout(margin=0, h_spacing=6, v_spacing=6)
        self.btn_save_roi = QPushButton("Save ROI…")
        self.btn_load_roi = QPushButton("Load ROI…")
        self.btn_save_roi.clicked.connect(self._save_roi_definition)
        self.btn_load_roi.clicked.connect(self._load_roi_definition)
        self.btn_load_roi.setToolTip("Explicitly load a fixed ROI into this scan only; review it for drift before applying.")
        sidecar_row.addWidget(self.btn_save_roi)
        sidecar_row.addWidget(self.btn_load_roi)
        roi_ctrl_layout.addLayout(sidecar_row)

        self.lbl_roi_status = QLabel("Drag yellow box on detector to adjust ROI.")
        self.lbl_roi_status.setWordWrap(True)
        self.lbl_roi_status.setMinimumWidth(0)
        self.lbl_roi_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.lbl_roi_status.setStyleSheet("color: #64748b; font-size: 11px;")
        roi_ctrl_layout.addWidget(self.lbl_roi_status)

        self.lbl_roi_metrics = QLabel("ROI Counts: --")
        self.lbl_roi_metrics.setWordWrap(True)
        self.lbl_roi_metrics.setMinimumWidth(0)
        self.lbl_roi_metrics.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        roi_ctrl_layout.addWidget(self.lbl_roi_metrics)

        sidebar_layout.addWidget(roi_ctrl_group)

        # 1b. Background ROI Card (freely resizable; area-scaled k = A_sig / A_bkg when size differs)
        bkg_ctrl_group = QGroupBox("Background ROI (pySAred 1.5.3)")
        bkg_ctrl_group.setMinimumSize(0, 0)
        bkg_layout = QVBoxLayout(bkg_ctrl_group)
        bkg_layout.setSpacing(6)

        self.chk_enable_bkg = QCheckBox("Enable background ROI subtraction")
        self.chk_enable_bkg.setToolTip(
            "Use a second freely resizable ROI to subtract background counts. "
            "Counts are scaled by exact pixel area ratio k = A_sig / A_bkg (k = 1 matches pySAred 1.5.3 identically)."
        )
        self.chk_enable_bkg.toggled.connect(self._on_bkg_toggled)
        bkg_layout.addWidget(self.chk_enable_bkg)

        bkg_grid = QHBoxLayout()
        bkg_grid.setSpacing(6)

        bkg_vx = QVBoxLayout()
        bkg_vx.addWidget(QLabel("BKG X Min:"))
        self.spin_bkg_x0 = QSpinBox()
        self.spin_bkg_x0.setMinimumWidth(55)
        self.spin_bkg_x0.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_bkg_x0.setRange(0, 1023)
        self.spin_bkg_x0.valueChanged.connect(self._on_bkg_spin_changed)
        bkg_vx.addWidget(self.spin_bkg_x0)

        bkg_vx.addWidget(QLabel("BKG X Max:"))
        self.spin_bkg_x1 = QSpinBox()
        self.spin_bkg_x1.setMinimumWidth(55)
        self.spin_bkg_x1.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_bkg_x1.setRange(1, 1024)
        self.spin_bkg_x1.valueChanged.connect(self._on_bkg_spin_changed)
        bkg_vx.addWidget(self.spin_bkg_x1)
        bkg_grid.addLayout(bkg_vx)

        bkg_vy = QVBoxLayout()
        bkg_vy.addWidget(QLabel("BKG Y Min:"))
        self.spin_bkg_y0 = QSpinBox()
        self.spin_bkg_y0.setMinimumWidth(55)
        self.spin_bkg_y0.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_bkg_y0.setRange(0, 1023)
        self.spin_bkg_y0.valueChanged.connect(self._on_bkg_spin_changed)
        bkg_vy.addWidget(self.spin_bkg_y0)

        bkg_vy.addWidget(QLabel("BKG Y Max:"))
        self.spin_bkg_y1 = QSpinBox()
        self.spin_bkg_y1.setMinimumWidth(55)
        self.spin_bkg_y1.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_bkg_y1.setRange(1, 1024)
        self.spin_bkg_y1.valueChanged.connect(self._on_bkg_spin_changed)
        bkg_vy.addWidget(self.spin_bkg_y1)
        bkg_grid.addLayout(bkg_vy)
        bkg_layout.addLayout(bkg_grid)

        self.btn_bkg_auto = QPushButton("Reset BKG to Default (Match Signal Size)")
        self.btn_bkg_auto.setToolTip("Place BKG ROI at pySAred default [x0 - 2w, x0 - w] matching Signal ROI width & Y (k = 1)")
        self.btn_bkg_auto.clicked.connect(self._reset_bkg_to_default)
        bkg_layout.addWidget(self.btn_bkg_auto)

        self.lbl_bkg_metrics = QLabel("BKG ROI inactive (enable to subtract background in Scan Charts).")
        self.lbl_bkg_metrics.setWordWrap(True)
        self.lbl_bkg_metrics.setMinimumWidth(0)
        self.lbl_bkg_metrics.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.lbl_bkg_metrics.setStyleSheet("color: #64748b; font-size: 11px;")
        bkg_layout.addWidget(self.lbl_bkg_metrics)
        sidebar_layout.addWidget(bkg_ctrl_group)

        # 2. 1D Horizontal Cut
        lbl_x_cut = QLabel("Horizontal Profile (X-Cut through ROI):")
        lbl_x_cut.setMinimumWidth(0)
        lbl_x_cut.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        sidebar_layout.addWidget(lbl_x_cut)
        self.plot_x_cut = _CompactPlotWidget()
        self.plot_x_cut.showGrid(x=True, y=True, alpha=0.3)
        self.plot_x_cut.setMinimumHeight(100)
        self.plot_x_cut.setMaximumHeight(160)
        apply_plot_theme(self.plot_x_cut, self.is_dark_mode)
        sidebar_layout.addWidget(self.plot_x_cut)

        # 3. 1D Vertical Cut
        lbl_y_cut = QLabel("Vertical Profile (Y-Cut through ROI):")
        lbl_y_cut.setMinimumWidth(0)
        lbl_y_cut.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        sidebar_layout.addWidget(lbl_y_cut)
        self.plot_y_cut = _CompactPlotWidget()
        self.plot_y_cut.showGrid(x=True, y=True, alpha=0.3)
        self.plot_y_cut.setMinimumHeight(100)
        self.plot_y_cut.setMaximumHeight(160)
        apply_plot_theme(self.plot_y_cut, self.is_dark_mode)
        sidebar_layout.addWidget(self.plot_y_cut)

        for widget, kind in ((self.plot_2d, "frame"), (self.plot_x_cut, "xcut"), (self.plot_y_cut, "ycut")):
            install_context_export(widget, (lambda k=kind: self._default_export_name(k)), self._source_paths)

        sidebar_layout.addStretch()

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setMinimumSize(0, 0)
        sidebar_scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        sidebar_scroll.setWidget(sidebar_container)
        splitter.addWidget(sidebar_scroll)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([680, 360])

        main_layout.addWidget(splitter, 1)
        install_wheel_focus_guard(self)

    @property
    def _current_frame_data(self) -> np.ndarray | None:
        if self._frame_dirty:
            self._ensure_frame_loaded()
        return self._frame_data

    @_current_frame_data.setter
    def _current_frame_data(self, value: np.ndarray | None) -> None:
        self._frame_data = value
        self._frame_dirty = False
        self._frame_total_counts = None

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._frame_dirty:
            self._ensure_frame_loaded()

    def _ensure_frame_loaded(self) -> None:
        if not self._frame_dirty:
            return
        self._frame_dirty = False
        if 0 <= self._current_filtered_pos < len(self._filtered_indices):
            self._load_frame(self._filtered_indices[self._current_filtered_pos])

    def set_data(self, reader: NexusReader, scan_data: ScanData) -> None:
        """Assign reader and scan data to detector view."""
        self.reader = reader
        self.scan_data = scan_data

        if not scan_data.has_detector_2d or scan_data.detector_shape is None:
            self._frame_data = None
            self._frame_dirty = False
            self._frame_total_counts = None
            self.lbl_frame_info.setText("No 2D detector data found in this NeXus file.")
            self.slider_step.setEnabled(False)
            self.spin_step.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.btn_auto_fit.setEnabled(False)
            self.btn_apply_roi.setEnabled(False)
            self.combo_spin_filter.setEnabled(False)
            self.chk_enable_bkg.setEnabled(False)
            for sp in (self.spin_bkg_x0, self.spin_bkg_x1, self.spin_bkg_y0, self.spin_bkg_y1):
                sp.setEnabled(False)
            self.btn_bkg_auto.setEnabled(False)
            if self._bkg_roi_rect_item is not None:
                self.plot_2d.removeItem(self._bkg_roi_rect_item)
                self._bkg_roi_rect_item = None
            return

        self._per_frame_rois = None
        self._per_frame_counts = None
        self._auto_fit_active = False
        self.lbl_roi_status.setText("Drag yellow box on detector to adjust ROI, or click 'Auto-Fit ROI'.")
        self.lbl_roi_status.setStyleSheet("color: #64748b; font-size: 11px;")

        record = scan_data.working_roi_record
        if record is not None:
            record.validate_scan(scan_data.detector_shape, scan_data.file_path)
            if record.mode == "tracked":
                self._per_frame_rois = list(record.bounds)
                existing = scan_data.variables.get("ROI_Counts")
                self._per_frame_counts = existing.data.copy() if existing is not None else reader.compute_per_frame_scan_roi(self._per_frame_rois)
                self._auto_fit_active = True
                self.lbl_roi_status.setText("Restored this scan's tracked ROI (counting intervals exclude tracking uncertainty).")
            else:
                self.lbl_roi_status.setText("Restored this scan's fixed ROI.")
        self._total_frames = scan_data.detector_shape[0]
        self.slider_step.setEnabled(True)
        self.spin_step.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_auto_fit.setEnabled(True)
        self.btn_apply_roi.setEnabled(True)

        # Setup interactive RectROI with scale and move handles
        if self._roi_rect_item is not None:
            self.plot_2d.removeItem(self._roi_rect_item)
            self._roi_rect_item = None

        initial_roi = (record.bounds[0] if record is not None else None) or scan_data.working_detector_roi or scan_data.detector_roi or (354.0, 382.0, 285.0, 743.0)
        x0, x1, y0, y1 = initial_roi
        pos_x = min(x0, x1)
        pos_y = min(y0, y1)
        size_x = max(1.0, abs(x1 - x0))
        size_y = max(1.0, abs(y1 - y0))

        self._roi_rect_item = pg.RectROI(
            pos=[pos_x, pos_y],
            size=[size_x, size_y],
            pen=pg.mkPen("#facc15", width=2),
            movable=True,
            resizable=True,
        )
        self._roi_rect_item.addScaleHandle([1, 1], [0, 0])
        self._roi_rect_item.addScaleHandle([0, 0], [1, 1])
        self._roi_rect_item.addScaleHandle([1, 0], [0, 1])
        self._roi_rect_item.addScaleHandle([0, 1], [1, 0])
        self._roi_rect_item.sigRegionChanged.connect(self._on_roi_region_changed)

        self.plot_2d.addItem(self._roi_rect_item)
        self._roi_rect_item.setVisible(self.chk_show_roi.isChecked())

        # Setup 2nd Background RectROI (cyan dashed, freely resizable like Signal ROI)
        if self._bkg_roi_rect_item is not None:
            self.plot_2d.removeItem(self._bkg_roi_rect_item)
            self._bkg_roi_rect_item = None

        nx = int(scan_data.detector_shape[1])
        ny = int(scan_data.detector_shape[2])
        params = scan_data.reduction_params
        if params.bkg_roi_bounds is None:
            if params.bkg_x_bounds is None:
                bx0, bx1 = default_bkg_x_bounds(pos_x, pos_x + size_x, nx=nx)
            else:
                bx0, bx1 = params.bkg_x_bounds
            params.bkg_roi_bounds = (float(bx0), float(bx1), float(pos_y), float(pos_y + size_y))
        bx0, bx1, by0, by1 = params.bkg_roi_bounds
        bx_min, bx_max = min(float(bx0), float(bx1)), max(float(bx0), float(bx1))
        by_min, by_max = min(float(by0), float(by1)), max(float(by0), float(by1))
        b_sx = max(1.0, bx_max - bx_min)
        b_sy = max(1.0, by_max - by_min)
        params.bkg_x_bounds = (bx_min, bx_min + b_sx)
        params.bkg_roi_bounds = (bx_min, bx_min + b_sx, by_min, by_min + b_sy)

        self._bkg_roi_rect_item = pg.RectROI(
            pos=[bx_min, by_min],
            size=[b_sx, b_sy],
            pen=pg.mkPen("#06b6d4", width=2, style=Qt.PenStyle.DashLine),
            movable=True,
            resizable=True,
        )
        self._bkg_roi_rect_item.addScaleHandle([1, 1], [0, 0])
        self._bkg_roi_rect_item.addScaleHandle([0, 0], [1, 1])
        self._bkg_roi_rect_item.addScaleHandle([1, 0], [0, 1])
        self._bkg_roi_rect_item.addScaleHandle([0, 1], [1, 0])
        self._bkg_roi_rect_item.sigRegionChanged.connect(self._on_bkg_roi_dragged)
        self.plot_2d.addItem(self._bkg_roi_rect_item)
        self._bkg_roi_rect_item.setVisible(self.chk_show_roi.isChecked() and params.subtract_background)

        self.chk_enable_bkg.setEnabled(True)
        for sp in (self.spin_bkg_x0, self.spin_bkg_x1, self.spin_bkg_y0, self.spin_bkg_y1):
            sp.setEnabled(True)
        self.btn_bkg_auto.setEnabled(True)
        self._updating_bkg_internally = True
        self.chk_enable_bkg.setChecked(bool(params.subtract_background))
        self.spin_bkg_x0.setRange(0, max(0, nx - 1))
        self.spin_bkg_x1.setRange(1, max(1, nx))
        self.spin_bkg_y0.setRange(0, max(0, ny - 1))
        self.spin_bkg_y1.setRange(1, max(1, ny))
        self.spin_bkg_x0.setValue(max(0, min(nx - 1, int(round(bx_min)))))
        self.spin_bkg_x1.setValue(max(1, min(nx, int(round(bx_min + b_sx)))))
        self.spin_bkg_y0.setValue(max(0, min(ny - 1, int(round(by_min)))))
        self.spin_bkg_y1.setValue(max(1, min(ny, int(round(by_min + b_sy)))))
        self._updating_bkg_internally = False

        self._sync_spinboxes_from_bounds(pos_x, pos_x + size_x, pos_y, pos_y + size_y)

        self._update_spin_filter_options()
        self._update_filtered_indices(reset_pos=True, defer_if_hidden=True)

    def _update_spin_filter_options(self) -> None:
        has_flipper = (
            self.scan_data is not None
            and "Flipper1" in self.scan_data.variables
            and len(self.scan_data.variables["Flipper1"].data) > 0
        )
        self.combo_spin_filter.setEnabled(has_flipper)

    def _on_spin_filter_changed(self) -> None:
        self._update_filtered_indices(reset_pos=False)

    def _update_filtered_indices(self, reset_pos: bool = False, *, defer_if_hidden: bool = False) -> None:
        self._paired_frames = {}
        if self.scan_data is None or self._total_frames <= 0:
            self._filtered_indices = []
            self.slider_step.setRange(0, 0)
            self.spin_step.setRange(0, 0)
            return

        mode = self.combo_spin_filter.currentData() or "all"
        flipper = self.scan_data.variables.get("Flipper1")
        if mode == "all" or flipper is None:
            self._filtered_indices = list(range(self._total_frames))
        elif mode == "up":
            f_data = flipper.data
            self._filtered_indices = [i for i in range(min(self._total_frames, len(f_data))) if float(f_data[i]) == 1]
        elif mode == "down":
            f_data = flipper.data
            self._filtered_indices = [i for i in range(min(self._total_frames, len(f_data))) if float(f_data[i]) == 0]
        elif mode == "sum":
            f_data = np.asarray(flipper.data, dtype=float)[:self._total_frames]
            th_var = self.scan_data.variables.get("th")
            tth_var = self.scan_data.variables.get("tth")
            written = None
            if (self.reader is not None and self.scan_data.detector_path
                    and self.scan_data.detector_path in self.reader.file):
                written = _written_frames(self.reader.file[self.scan_data.detector_path])
            if th_var is not None and tth_var is not None:
                ups, downs = pair_spin_frames(
                    th_var.data[:self._total_frames], tth_var.data[:self._total_frames], f_data, written,
                )
            else:
                ups_raw = np.flatnonzero(f_data == 1)
                downs_raw = np.flatnonzero(f_data == 0)
                n_p = min(ups_raw.size, downs_raw.size)
                ups, downs = ups_raw[:n_p], downs_raw[:n_p]
            self._paired_frames = {int(u): int(d) for u, d in zip(ups, downs)}
            self._filtered_indices = [int(u) for u in ups]

        if not self._filtered_indices:
            if mode == "sum":
                self._current_frame_data = None
                self.img_item.clear()
                self.plot_x_cut.clear()
                self.plot_y_cut.clear()
                self.slider_step.blockSignals(True)
                self.spin_step.blockSignals(True)
                self.slider_step.setRange(0, 0)
                self.spin_step.setRange(0, 0)
                self.slider_step.setValue(0)
                self.spin_step.setValue(0)
                self.slider_step.blockSignals(False)
                self.spin_step.blockSignals(False)
                self.lbl_frame_info.setText("No matching u/d frame pairs found to sum.")
                return
            self._filtered_indices = list(range(self._total_frames))

        n_filt = len(self._filtered_indices)
        self.slider_step.blockSignals(True)
        self.spin_step.blockSignals(True)
        self.slider_step.setRange(0, max(0, n_filt - 1))
        self.spin_step.setRange(0, max(0, n_filt - 1))

        if reset_pos or self._current_filtered_pos >= n_filt:
            self._current_filtered_pos = 0
        else:
            cur_raw = self._current_frame_idx
            best_pos = 0
            best_dist = float("inf")
            for pos, raw_idx in enumerate(self._filtered_indices):
                dist = abs(raw_idx - cur_raw)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = pos
            self._current_filtered_pos = best_pos

        self.slider_step.setValue(self._current_filtered_pos)
        self.spin_step.setValue(self._current_filtered_pos)
        self.slider_step.blockSignals(False)
        self.spin_step.blockSignals(False)

        raw_idx = self._filtered_indices[self._current_filtered_pos]
        if defer_if_hidden and not self.isVisible():
            self._current_frame_idx = raw_idx
            self._frame_data = None
            self._frame_total_counts = None
            self._frame_dirty = True
            return
        self._load_frame(raw_idx)

    def _source_paths(self) -> list[str]:
        own = [self.scan_data.file_path] if self.scan_data is not None and self.scan_data.file_path else []
        return own + list(self.session_source_paths())

    def _default_export_name(self, kind: str) -> str:
        stem = Path(self.scan_data.file_path).stem if self.scan_data is not None and self.scan_data.file_path else "scan"
        if kind == "frame":
            partner = self._paired_frames.get(self._current_frame_idx) if self.combo_spin_filter.currentData() == "sum" else None
            if partner is not None:
                return f"{stem}_frame{self._current_frame_idx:04d}_{partner:04d}_sum.png"
            return f"{stem}_{kind}{self._current_frame_idx:04d}.png"
        return f"{stem}_{kind}.png"

    def export_image(self) -> Path | None:
        """Save the displayed frame (with ROI box) as PNG or SVG."""
        if self._current_frame_data is None:
            QMessageBox.information(self, "No Data", "No detector frame to export.")
            return None
        return export_plot_image(self, self.plot_2d.getPlotItem(), self._default_export_name("frame"),
                                 self._source_paths(), "Export Detector Frame Image")

    def set_theme(self, is_dark: bool) -> None:
        """Update styling for dark / light mode."""
        self.is_dark_mode = is_dark
        play_icon_name = "pause" if self._play_timer.isActive() else "play"
        self.btn_play.setIcon(icons.get_icon(play_icon_name, is_dark=is_dark))
        self.btn_prev.setIcon(icons.get_icon("chevron_left", is_dark=is_dark))
        self.btn_next.setIcon(icons.get_icon("chevron_right", is_dark=is_dark))
        self.btn_apply_roi.setIcon(icons.get_icon("chart", is_dark=is_dark))
        self.btn_auto_fit.setIcon(icons.get_icon("sparkles", is_dark=is_dark))
        self.btn_reset_roi.setIcon(icons.get_icon("refresh_cw", is_dark=is_dark))
        self.btn_img_pan.setIcon(icons.get_icon("hand", is_dark=is_dark))
        self.btn_img_box_zoom.setIcon(icons.get_icon("box_select", is_dark=is_dark))
        self.btn_img_zoom_in.setIcon(icons.get_icon("zoom_in", is_dark=is_dark))
        self.btn_img_zoom_out.setIcon(icons.get_icon("zoom_out", is_dark=is_dark))
        self.btn_img_reset.setIcon(icons.get_icon("maximize", is_dark=is_dark))
        self.btn_img_export.setIcon(icons.get_icon("download", is_dark=is_dark))

        apply_plot_theme(self.plot_2d, is_dark)
        apply_plot_theme(self.plot_x_cut, is_dark)
        apply_plot_theme(self.plot_y_cut, is_dark)
        self._refresh_current_frame()

    def _set_img_mouse_mode_pan(self) -> None:
        self.btn_img_pan.setChecked(True)
        self.plot_2d.getViewBox().setMouseMode(pg.ViewBox.PanMode)

    def _set_img_mouse_mode_box_zoom(self) -> None:
        self.btn_img_box_zoom.setChecked(True)
        self.plot_2d.getViewBox().setMouseMode(pg.ViewBox.RectMode)

    def _img_zoom_in(self) -> None:
        self.plot_2d.getViewBox().scaleBy((0.7, 0.7))

    def _img_zoom_out(self) -> None:
        self.plot_2d.getViewBox().scaleBy((1.4, 1.4))

    def _img_reset_view(self) -> None:
        self.plot_2d.autoRange()

    def _on_mouse_moved_2d(self, pos: Any) -> None:
        if self._current_frame_data is None:
            return
        vb = self.plot_2d.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            pt = vb.mapSceneToView(pos)
            x = int(round(pt.x()))
            y = int(round(pt.y()))
            ny, nx = self._current_frame_data.shape
            if 0 <= x < nx and 0 <= y < ny:
                val = self._current_frame_data[y, x]
                self.lbl_img_coords.setText(f"Pixel: (X: {x:4d}, Y: {y:4d})  |  Counts: {int(val):8,d}")
            else:
                self.lbl_img_coords.setText("Pixel: (X: ----, Y: ----)  |  Counts: --------")
        else:
            self.lbl_img_coords.setText("Pixel: (X: ----, Y: ----)  |  Counts: --------")

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setIcon(icons.get_icon("play", is_dark=self.is_dark_mode))
        else:
            self._play_timer.start()
            self.btn_play.setIcon(icons.get_icon("pause", is_dark=self.is_dark_mode))

    def _step_forward(self) -> None:
        if not self._filtered_indices:
            return
        nxt = (self._current_filtered_pos + 1) % len(self._filtered_indices)
        self.spin_step.setValue(nxt)

    def _step_backward(self) -> None:
        if not self._filtered_indices:
            return
        prev = (self._current_filtered_pos - 1) % len(self._filtered_indices)
        self.spin_step.setValue(prev)

    def _on_step_spin_changed(self, val: int) -> None:
        self.slider_step.blockSignals(True)
        self.slider_step.setValue(val)
        self.slider_step.blockSignals(False)
        self._current_filtered_pos = val
        if 0 <= val < len(self._filtered_indices):
            self._load_frame(self._filtered_indices[val])

    def _on_slider_changed(self, val: int) -> None:
        self.spin_step.blockSignals(True)
        self.spin_step.setValue(val)
        self.spin_step.blockSignals(False)
        self._current_filtered_pos = val
        if 0 <= val < len(self._filtered_indices):
            self._load_frame(self._filtered_indices[val])

    def _on_cmap_changed(self, cmap_name: str) -> None:
        self._loaded_cmap_name = cmap_name
        cm = get_safe_colormap(cmap_name)
        self.img_item.setColorMap(cm)

    def _toggle_roi_visibility(self, visible: bool) -> None:
        if self._roi_rect_item is not None:
            self._roi_rect_item.setVisible(visible)
        if self._bkg_roi_rect_item is not None:
            bkg_on = bool(self.scan_data is not None and self.scan_data.reduction_params.subtract_background)
            self._bkg_roi_rect_item.setVisible(visible and bkg_on)

    def _refresh_current_frame(self) -> None:
        if self._frame_dirty or self._frame_data is None:
            return
        self._render_current_frame(total_counts=self._frame_total_counts)

    def _get_current_roi_bounds(self) -> tuple[float, float, float, float]:
        if self._roi_rect_item is None:
            return (0.0, 1024.0, 0.0, 1024.0)
        pos = self._roi_rect_item.pos()
        size = self._roi_rect_item.size()
        x0 = float(pos.x())
        x1 = float(pos.x() + size.x())
        y0 = float(pos.y())
        y1 = float(pos.y() + size.y())
        return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    def _sync_spinboxes_from_bounds(self, x0: float, x1: float, y0: float, y1: float) -> None:
        self._updating_roi_internally = True
        self.spin_x0.setValue(max(0, min(1023, int(round(x0)))))
        self.spin_x1.setValue(max(0, min(1024, int(round(x1)))))
        self.spin_y0.setValue(max(0, min(1023, int(round(y0)))))
        self.spin_y1.setValue(max(0, min(1024, int(round(y1)))))
        self._updating_roi_internally = False

    def _on_roi_region_changed(self) -> None:
        """Fired continuously while dragging or resizing RectROI."""
        if self._updating_roi_internally:
            return

        if self._auto_fit_active:
            self._auto_fit_active = False
            self._per_frame_rois = None
            self._per_frame_counts = None
            self.lbl_roi_status.setText("Manual ROI active. Drag box or click 'Auto-Fit ROI'.")
            self.lbl_roi_status.setStyleSheet("color: #64748b; font-size: 11px;")

        x0, x1, y0, y1 = self._get_current_roi_bounds()
        self._sync_spinboxes_from_bounds(x0, x1, y0, y1)
        self._sync_bkg_geometry_to_signal(x0, x1, y0, y1)
        self._update_profiles_and_counts(x0, x1, y0, y1)

    def _on_spin_roi_changed(self) -> None:
        """Fired when user manually edits numeric ROI spinboxes."""
        if self._updating_roi_internally or self._roi_rect_item is None:
            return

        if self._auto_fit_active:
            self._auto_fit_active = False
            self._per_frame_rois = None
            self._per_frame_counts = None
            self.lbl_roi_status.setText("Manual ROI active. Drag box or click 'Auto-Fit ROI'.")
            self.lbl_roi_status.setStyleSheet("color: #64748b; font-size: 11px;")

        x0 = float(self.spin_x0.value())
        x1 = float(self.spin_x1.value())
        y0 = float(self.spin_y0.value())
        y1 = float(self.spin_y1.value())

        min_x = min(x0, x1)
        size_x = max(1.0, abs(x1 - x0))
        min_y = min(y0, y1)
        size_y = max(1.0, abs(y1 - y0))

        self._updating_roi_internally = True
        self._roi_rect_item.setPos([min_x, min_y])
        self._roi_rect_item.setSize([size_x, size_y])
        self._updating_roi_internally = False

        self._sync_bkg_geometry_to_signal(min_x, min_x + size_x, min_y, min_y + size_y)
        self._update_profiles_and_counts(min_x, min_x + size_x, min_y, min_y + size_y)

    def _reset_to_file_roi(self) -> None:
        """Reset ROI to the default file detector ROI."""
        if self.scan_data is None:
            return
        self._auto_fit_active = False
        self._per_frame_rois = None
        self._per_frame_counts = None

        initial_roi = self.scan_data.detector_roi or (354.0, 382.0, 285.0, 743.0)
        x0, x1, y0, y1 = initial_roi
        min_x = min(x0, x1)
        size_x = max(1.0, abs(x1 - x0))
        min_y = min(y0, y1)
        size_y = max(1.0, abs(y1 - y0))

        if self._roi_rect_item is not None:
            self._updating_roi_internally = True
            self._roi_rect_item.setPos([min_x, min_y])
            self._roi_rect_item.setSize([size_x, size_y])
            self._updating_roi_internally = False

        self._sync_spinboxes_from_bounds(min_x, min_x + size_x, min_y, min_y + size_y)
        self._sync_bkg_geometry_to_signal(min_x, min_x + size_x, min_y, min_y + size_y)
        self._apply_roi_to_scan_chart()
        self.lbl_roi_status.setText("Reset to default file ROI.")
        self.lbl_roi_status.setStyleSheet("color: #64748b; font-size: 11px;")

    # ------------------------------------------------------------------ Background ROI (freely resizable)
    def _detector_nx(self) -> int:
        if self.scan_data is not None and self.scan_data.detector_shape is not None:
            return int(self.scan_data.detector_shape[1])
        return 1024

    def _detector_ny(self) -> int:
        if self.scan_data is not None and self.scan_data.detector_shape is not None:
            return int(self.scan_data.detector_shape[2])
        return 1024

    def _get_current_bkg_bounds(self) -> tuple[float, float, float, float]:
        if self._bkg_roi_rect_item is None:
            sx0, sx1, sy0, sy1 = self._get_current_roi_bounds()
            bx0, bx1 = default_bkg_x_bounds(sx0, sx1, nx=self._detector_nx())
            return (bx0, bx1, sy0, sy1)
        pos = self._bkg_roi_rect_item.pos()
        size = self._bkg_roi_rect_item.size()
        x0, x1 = float(pos.x()), float(pos.x() + size.x())
        y0, y1 = float(pos.y()), float(pos.y() + size.y())
        return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    def _sync_bkg_spinboxes(self, bx0: float, bx1: float, by0: float, by1: float) -> None:
        nx, ny = self._detector_nx(), self._detector_ny()
        self._updating_bkg_internally = True
        self.spin_bkg_x0.setValue(max(0, min(nx - 1, int(round(min(bx0, bx1))))))
        self.spin_bkg_x1.setValue(max(1, min(nx, int(round(max(bx0, bx1))))))
        self.spin_bkg_y0.setValue(max(0, min(ny - 1, int(round(min(by0, by1))))))
        self.spin_bkg_y1.setValue(max(1, min(ny, int(round(max(by0, by1))))))
        self._updating_bkg_internally = False

    def _sync_bkg_geometry_to_signal(self, sx0: float, sx1: float, sy0: float, sy1: float) -> None:
        """Initialize BKG ROI from Signal ROI only if unset; preserve user-resized BKG ROI."""
        if self.scan_data is None or self._bkg_roi_rect_item is None:
            return
        nx, ny = self._detector_nx(), self._detector_ny()
        params = self.scan_data.reduction_params
        if params.bkg_roi_bounds is None:
            if params.bkg_x_bounds is None:
                bx0, bx1 = default_bkg_x_bounds(sx0, sx1, nx=nx)
            else:
                bx0, bx1 = params.bkg_x_bounds
            by0, by1 = min(sy0, sy1), max(sy0, sy1)
            params.bkg_roi_bounds = (float(bx0), float(bx1), float(by0), float(by1))
            params.bkg_x_bounds = (float(bx0), float(bx1))
        else:
            bx0, bx1, by0, by1 = params.bkg_roi_bounds
            bx0, bx1 = max(0.0, min(float(nx), float(bx0))), max(0.0, min(float(nx), float(bx1)))
            by0, by1 = max(0.0, min(float(ny), float(by0))), max(0.0, min(float(ny), float(by1)))
            if bx1 <= bx0:
                bx1 = min(float(nx), bx0 + 1.0)
            if by1 <= by0:
                by1 = min(float(ny), by0 + 1.0)
            params.bkg_roi_bounds = (bx0, bx1, by0, by1)
            params.bkg_x_bounds = (bx0, bx1)

        self._updating_bkg_internally = True
        self._bkg_roi_rect_item.setPos([bx0, by0])
        self._bkg_roi_rect_item.setSize([max(1.0, bx1 - bx0), max(1.0, by1 - by0)])
        self._updating_bkg_internally = False
        self._sync_bkg_spinboxes(bx0, bx1, by0, by1)

    def _on_bkg_toggled(self, checked: bool) -> None:
        if self._updating_bkg_internally or self.scan_data is None:
            return
        self.scan_data.reduction_params.subtract_background = bool(checked)
        self.scan_data.working_bkg_counts = None
        if self._bkg_roi_rect_item is not None:
            self._bkg_roi_rect_item.setVisible(bool(checked) and self.chk_show_roi.isChecked())
        x0, x1, y0, y1 = self._get_current_roi_bounds()
        if self._frame_data is not None:
            self._update_profiles_and_counts(x0, x1, y0, y1)
        else:
            self._refresh_bkg_metrics(x0, x1, y0, y1)
        self.bkg_roi_changed.emit(self.scan_data.file_path)

    def _on_bkg_roi_dragged(self) -> None:
        """Free drag or resize of the cyan BKG ROI (same behavior as the yellow Signal ROI)."""
        if self._updating_bkg_internally or self.scan_data is None or self._bkg_roi_rect_item is None:
            return
        sx0, sx1, sy0, sy1 = self._get_current_roi_bounds()
        nx, ny = float(self._detector_nx()), float(self._detector_ny())
        bx0, bx1, by0, by1 = self._get_current_bkg_bounds()
        bx0 = max(0.0, min(nx - 1.0, bx0))
        bx1 = max(bx0 + 1.0, min(nx, bx1))
        by0 = max(0.0, min(ny - 1.0, by0))
        by1 = max(by0 + 1.0, min(ny, by1))

        self._sync_bkg_spinboxes(bx0, bx1, by0, by1)
        self.scan_data.reduction_params.bkg_roi_bounds = (bx0, bx1, by0, by1)
        self.scan_data.reduction_params.bkg_x_bounds = (bx0, bx1)
        self.scan_data.working_bkg_counts = None
        if self._frame_data is not None:
            self._update_profiles_and_counts(sx0, sx1, sy0, sy1)
        else:
            self._refresh_bkg_metrics(sx0, sx1, sy0, sy1)
        self.bkg_roi_changed.emit(self.scan_data.file_path)

    def _on_bkg_spin_changed(self, _value: int = 0) -> None:
        if self._updating_bkg_internally or self.scan_data is None or self._bkg_roi_rect_item is None:
            return
        sx0, sx1, sy0, sy1 = self._get_current_roi_bounds()
        x0 = float(self.spin_bkg_x0.value())
        x1 = float(self.spin_bkg_x1.value())
        y0 = float(self.spin_bkg_y0.value())
        y1 = float(self.spin_bkg_y1.value())
        bx0, bx1 = min(x0, x1), max(x0, x1)
        by0, by1 = min(y0, y1), max(y0, y1)
        size_x = max(1.0, bx1 - bx0)
        size_y = max(1.0, by1 - by0)
        bx1 = bx0 + size_x
        by1 = by0 + size_y

        self._updating_bkg_internally = True
        self._bkg_roi_rect_item.setPos([bx0, by0])
        self._bkg_roi_rect_item.setSize([size_x, size_y])
        self._updating_bkg_internally = False

        self.scan_data.reduction_params.bkg_roi_bounds = (bx0, bx1, by0, by1)
        self.scan_data.reduction_params.bkg_x_bounds = (bx0, bx1)
        self.scan_data.working_bkg_counts = None
        if self._frame_data is not None:
            self._update_profiles_and_counts(sx0, sx1, sy0, sy1)
        else:
            self._refresh_bkg_metrics(sx0, sx1, sy0, sy1)
        self.bkg_roi_changed.emit(self.scan_data.file_path)

    def _reset_bkg_to_default(self) -> None:
        if self.scan_data is None or self._bkg_roi_rect_item is None:
            return
        sx0, sx1, sy0, sy1 = self._get_current_roi_bounds()
        nx = self._detector_nx()
        bx0, bx1 = default_bkg_x_bounds(sx0, sx1, nx=nx)
        by0, by1 = min(sy0, sy1), max(sy0, sy1)
        self.scan_data.reduction_params.bkg_x_bounds = (bx0, bx1)
        self.scan_data.reduction_params.bkg_roi_bounds = (bx0, bx1, by0, by1)
        self.scan_data.working_bkg_counts = None
        self._sync_bkg_geometry_to_signal(sx0, sx1, sy0, sy1)
        if self._frame_data is not None:
            self._update_profiles_and_counts(sx0, sx1, sy0, sy1)
        else:
            self._refresh_bkg_metrics(sx0, sx1, sy0, sy1)
        self.bkg_roi_changed.emit(self.scan_data.file_path)

    def sync_from_scan(self) -> None:
        """Refresh BKG ROI controls & overlay from ``scan_data.reduction_params`` (e.g. toggled in Scan Charts)."""
        if self.scan_data is None or self._bkg_roi_rect_item is None:
            return
        params = self.scan_data.reduction_params
        sx0, sx1, sy0, sy1 = self._get_current_roi_bounds()
        self._sync_bkg_geometry_to_signal(sx0, sx1, sy0, sy1)
        self._updating_bkg_internally = True
        self.chk_enable_bkg.setChecked(bool(params.subtract_background))
        self._updating_bkg_internally = False
        self._bkg_roi_rect_item.setVisible(bool(params.subtract_background) and self.chk_show_roi.isChecked())
        if self._frame_data is not None:
            self._update_profiles_and_counts(sx0, sx1, sy0, sy1)
        else:
            self._refresh_bkg_metrics(sx0, sx1, sy0, sy1)

    def _refresh_bkg_metrics(self, sx0: float, sx1: float, sy0: float, sy1: float) -> None:
        if self.scan_data is None:
            return
        params = self.scan_data.reduction_params
        if not params.subtract_background:
            self.lbl_bkg_metrics.setText("BKG ROI inactive (enable to subtract background in Scan Charts).")
            self.lbl_bkg_metrics.setStyleSheet("color: #64748b; font-size: 11px;")
            return
        nx, ny = self._detector_nx(), self._detector_ny()
        sig_roi = (sx0, sx1, sy0, sy1)
        bx0, bx1, by0, by1 = resolve_bkg_roi(sig_roi, params.bkg_roi_bounds, params.bkg_x_bounds, nx, ny)
        bkg_roi = (float(bx0), float(bx1), float(by0), float(by1))
        overlap = rois_overlap_2d(sig_roi, bkg_roi, nx=nx, ny=ny)
        try:
            area_k = roi_area_ratio(sig_roi, bkg_roi, nx, ny)
            k_txt = f" | area scale k={area_k:.3g}" if abs(area_k - 1.0) > 1e-4 else " | k=1"
        except ValueError:
            k_txt = " | empty BKG"
        _, _, isy0, isy1 = pixel_bounds(sig_roi, nx, ny)
        y_note = " (Y differs from signal)" if (by0, by1) != (isy0, isy1) else ""
        if self._frame_data is not None:
            bkg_cnt, _, _ = integrate_roi_2d(self._frame_data, bkg_roi)
            count_txt = f"BKG Frame Counts: <b>{bkg_cnt:,.0f}</b> (X [{bx0}..{bx1}], Y [{by0}..{by1}]{k_txt}){y_note}"
        else:
            count_txt = f"BKG X [{bx0}..{bx1}], Y [{by0}..{by1}]{k_txt}{y_note}"
        if overlap:
            self.lbl_bkg_metrics.setText(f"⚠ BKG ROI overlaps Signal ROI! {count_txt}")
            self.lbl_bkg_metrics.setStyleSheet("color: #d97706; font-weight: 600; font-size: 11px;")
        else:
            self.lbl_bkg_metrics.setText(count_txt)
            self.lbl_bkg_metrics.setStyleSheet("color: #0891b2; font-size: 11px;")

    def _current_roi_record(self):
        shape = self.scan_data.detector_shape
        tracked = self._auto_fit_active and self._per_frame_rois is not None
        bounds = tuple(tuple(r) for r in self._per_frame_rois) if tracked else (tuple(self._get_current_roi_bounds()),)
        return ROIRecord("tracked" if tracked else "fixed", (shape[2], shape[1]), shape[0], bounds, self.scan_data.file_path)

    def _emit_roi(self, counts, bounds):
        self.scan_data.working_roi_record = self._current_roi_record()
        self.scan_data.working_detector_roi = tuple(bounds)
        self.roi_applied.emit(counts, bounds)

    def _save_roi_definition(self):
        if self.scan_data is None or not self.scan_data.has_detector_2d:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save ROI definition (new file)", "scan.roi.json", "ROI definitions (*.json)")
        if not path:
            return
        try:
            save_roi(path, self._current_roi_record())
        except (ValueError, OSError, TypeError) as error:
            QMessageBox.warning(self, "ROI not saved", str(error))
            return
        self.lbl_roi_status.setText("ROI definition saved separately; measurement file unchanged.")

    def _load_roi_definition(self):
        if self.scan_data is None or not self.scan_data.has_detector_2d:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load ROI into this scan", "", "ROI definitions (*.json)")
        if not path:
            return
        try:
            record = load_roi(path)
            record.validate_scan(self.scan_data.detector_shape, self.scan_data.file_path)
        except (ValueError, OSError, TypeError) as error:
            QMessageBox.warning(self, "ROI not loaded", str(error))
            return
        # Stage only: users review the overlay before applying it to the chart.
        self._auto_fit_active = record.mode == "tracked"
        self._per_frame_rois = list(record.bounds) if self._auto_fit_active else None
        self._per_frame_counts = None
        if self._auto_fit_active:
            self._per_frame_counts = self.reader.compute_per_frame_scan_roi(self._per_frame_rois)
        bounds = record.bounds[min(self._current_frame_idx, len(record.bounds)-1)]
        x0, x1, y0, y1 = bounds
        self._updating_roi_internally = True
        self._roi_rect_item.setPos([min(x0, x1), min(y0, y1)])
        self._roi_rect_item.setSize([abs(x1-x0), abs(y1-y0)])
        self._updating_roi_internally = False
        self._sync_spinboxes_from_bounds(*bounds)
        self._sync_bkg_geometry_to_signal(*bounds)
        self._update_profiles_and_counts(*bounds)
        self.lbl_roi_status.setText("Loaded into this scan only. Review for drift, then Apply ROI to Scan Chart.")

    def _apply_roi_to_scan_chart(self) -> None:
        """Compute full scan integration for the current ROI and emit roi_applied."""
        if self.reader is None or self.scan_data is None:
            return

        try:
            self._current_roi_record()
        except (ValueError, TypeError) as error:
            QMessageBox.warning(self, "Invalid ROI", str(error))
            return
        if self._auto_fit_active and self._per_frame_counts is not None and self._per_frame_rois is not None:
            roi_counts = self._per_frame_counts
            cur_roi = self._per_frame_rois[self._current_frame_idx] if 0 <= self._current_frame_idx < len(self._per_frame_rois) else self._get_current_roi_bounds()
            self._emit_roi(roi_counts, cur_roi)
            self.lbl_roi_status.setText(
                f"Auto-Fit ROI ({len(self._per_frame_rois)} frames) applied to Scan Chart!"
            )
            self.lbl_roi_status.setStyleSheet("color: #10b981; font-weight: 600; font-size: 11px;")
            return

        x0, x1, y0, y1 = self._get_current_roi_bounds()
        self.lbl_roi_status.setText("Recalculating scan points for ROI...")

        roi_bounds = (x0, x1, y0, y1)
        roi_counts = self.reader.compute_full_scan_roi(roi_bounds)

        if roi_counts is not None:
            # Emit signal for MainWindow / ScanPlotView
            self._emit_roi(roi_counts, roi_bounds)
            self.lbl_roi_status.setText(
                f"ROI applied to Scan Chart! (X: [{x0:.0f}..{x1:.0f}], Y: [{y0:.0f}..{y1:.0f}])"
            )
            self.lbl_roi_status.setStyleSheet("color: #10b981; font-weight: 600; font-size: 11px;")
        else:
            self.lbl_roi_status.setText("Failed to calculate ROI curve.")
            self.lbl_roi_status.setStyleSheet("color: #ef4444; font-weight: 600; font-size: 11px;")

    def _on_auto_fit_clicked(self) -> None:
        """Automatically adjust ROI to center on reflection peak across all frames."""
        if self.reader is None or self.scan_data is None:
            return

        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        self.lbl_roi_status.setText("Auto-fitting ROI across all scan frames...")
        self.lbl_roi_status.setStyleSheet("color: #0284c7; font-weight: 600; font-size: 11px;")
        QApplication.processEvents()

        try:
            initial_roi = self._get_current_roi_bounds()
            if initial_roi[1] <= initial_roi[0] or initial_roi[3] <= initial_roi[2]:
                initial_roi = self.scan_data.detector_roi or (354.0, 382.0, 285.0, 743.0)

            result = self.reader.auto_fit_detector_rois(initial_roi=initial_roi)
        finally:
            QApplication.restoreOverrideCursor()

        if result is None:
            self.lbl_roi_status.setText("Auto-fit failed: detector data unavailable.")
            self.lbl_roi_status.setStyleSheet("color: #ef4444; font-weight: 600; font-size: 11px;")
            return

        rois, counts = result
        self._per_frame_rois = rois
        self._per_frame_counts = counts
        self._auto_fit_active = True

        # Update current frame displayed ROI box
        if 0 <= self._current_frame_idx < len(rois):
            cur_roi = rois[self._current_frame_idx]
            x0, x1, y0, y1 = cur_roi
            min_x = min(x0, x1)
            size_x = max(1.0, abs(x1 - x0))
            min_y = min(y0, y1)
            size_y = max(1.0, abs(y1 - y0))

            if self._roi_rect_item is not None:
                self._updating_roi_internally = True
                self._roi_rect_item.setPos([min_x, min_y])
                self._roi_rect_item.setSize([size_x, size_y])
                self._updating_roi_internally = False

            self._sync_spinboxes_from_bounds(min_x, min_x + size_x, min_y, min_y + size_y)
            self._sync_bkg_geometry_to_signal(min_x, min_x + size_x, min_y, min_y + size_y)
            self._update_profiles_and_counts(min_x, min_x + size_x, min_y, min_y + size_y)

        centers = [(r[0] + r[1]) / 2.0 for r in rois]
        mean_c = float(np.mean(centers))
        std_c = float(np.std(centers))
        self.lbl_roi_status.setText(
            f"Auto-Fit active: {len(rois)} frames tracked (Peak X: {mean_c:.1f} ± {std_c:.1f} px). Applied to Scan Chart! Counting intervals exclude tracking uncertainty."
        )
        self.lbl_roi_status.setStyleSheet("color: #10b981; font-weight: 600; font-size: 11px;")

        # Automatically apply to scan chart
        cur_roi = rois[self._current_frame_idx] if 0 <= self._current_frame_idx < len(rois) else initial_roi
        self._emit_roi(counts, cur_roi)

    def _update_profiles_and_counts(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        *,
        total_counts: float | None = None,
    ) -> None:
        frame = self._current_frame_data
        if frame is None:
            return

        if total_counts is None:
            total_counts = self._frame_total_counts
        if total_counts is None:
            total_counts = float(np.sum(frame))
        roi_counts, x_profile, y_profile = integrate_roi_2d(frame, (x0, x1, y0, y1))

        pct = (roi_counts / total_counts * 100.0) if total_counts > 0 else 0.0
        roi_text = (
            f"<b>ROI Box:</b> X [{x0:.0f} .. {x1:.0f}], Y [{y0:.0f} .. {y1:.0f}]<br>"
            f"<b>ROI Counts:</b> {roi_counts:,.0f} ({pct:.1f}% of frame)<br>"
            f"<b>Peak in ROI:</b> {np.max(x_profile) if len(x_profile) else 0:.0f} cts"
        )
        self.lbl_roi_metrics.setText(roi_text)
        self._refresh_bkg_metrics(x0, x1, y0, y1)

        # Plot horizontal cut (expand to show both Signal and BKG windows when BKG ROI is active)
        self.plot_x_cut.clear()
        pen_x = "#38bdf8" if self.is_dark_mode else "#0284c7"
        ix0, ix1, iy0, iy1 = pixel_bounds((x0, x1, y0, y1), frame.shape[1], frame.shape[0])
        bkg_on = bool(self.scan_data is not None and self.scan_data.reduction_params.subtract_background)
        if bkg_on and iy1 > iy0:
            bx0, bx1, _, _ = resolve_bkg_roi(
                (x0, x1, y0, y1),
                self.scan_data.reduction_params.bkg_roi_bounds,
                self.scan_data.reduction_params.bkg_x_bounds,
                frame.shape[1],
                frame.shape[0],
            )
            pad = max(10, (ix1 - ix0) // 2)
            vx0 = max(0, min(ix0, bx0) - pad)
            vx1 = min(frame.shape[1], max(ix1, bx1) + pad)
            wide_prof = np.sum(frame[iy0:iy1, vx0:vx1], axis=0, dtype=float)
            self.plot_x_cut.plot(np.arange(vx0, vx1), wide_prof, pen=pg.mkPen(pen_x, width=2))
            sig_band = pg.LinearRegionItem([ix0, ix1], movable=False, brush=pg.mkBrush(250, 204, 21, 55), pen=pg.mkPen("#facc15", width=1))
            bkg_band = pg.LinearRegionItem([bx0, bx1], movable=False, brush=pg.mkBrush(6, 182, 212, 55), pen=pg.mkPen("#06b6d4", width=1, style=Qt.PenStyle.DashLine))
            self.plot_x_cut.addItem(sig_band)
            self.plot_x_cut.addItem(bkg_band)
        else:
            x_coords = np.arange(ix0, ix1)
            self.plot_x_cut.plot(x_coords, x_profile, pen=pg.mkPen(pen_x, width=2), symbol="o", symbolSize=3)
        self.plot_x_cut.setLabel("bottom", "X Pixel", color="#f1f5f9" if self.is_dark_mode else "#0f172a")
        self.plot_x_cut.setLabel("left", "Counts", color="#f1f5f9" if self.is_dark_mode else "#0f172a")

        # Plot vertical cut
        self.plot_y_cut.clear()
        pen_y = "#f43f5e" if self.is_dark_mode else "#e11d48"
        y_coords = np.arange(iy0, iy1)
        self.plot_y_cut.plot(y_coords, y_profile, pen=pg.mkPen(pen_y, width=2), symbol="o", symbolSize=3)
        self.plot_y_cut.setLabel("bottom", "Y Pixel", color="#f1f5f9" if self.is_dark_mode else "#0f172a")
        self.plot_y_cut.setLabel("left", "Counts", color="#f1f5f9" if self.is_dark_mode else "#0f172a")

    def _render_current_frame(
        self,
        *,
        raw_combined: np.ndarray | None = None,
        total_counts: float | None = None,
    ) -> None:
        frame = self._frame_data
        if frame is None:
            return

        # In pyqtgraph ImageItem displays array with transposed axis mapping, so raw_combined == frame.T.
        img_src = raw_combined if raw_combined is not None else frame.T
        if self.chk_log_img.isChecked():
            img_display = np.log10(np.maximum(img_src, 1.0))
        else:
            img_display = img_src

        self.img_item.setImage(img_display)
        cmap_name = self.combo_cmap.currentText()
        if cmap_name != self._loaded_cmap_name:
            self._loaded_cmap_name = cmap_name
            self.img_item.setColorMap(get_safe_colormap(cmap_name))

        x0, x1, y0, y1 = self._get_current_roi_bounds()
        self._update_profiles_and_counts(x0, x1, y0, y1, total_counts=total_counts)

    def _load_frame(self, frame_idx: int) -> None:
        if self.reader is None or self.scan_data is None:
            return

        self._frame_dirty = False
        self._current_frame_idx = frame_idx
        partner_idx = self._paired_frames.get(frame_idx) if self.combo_spin_filter.currentData() == "sum" else None
        try:
            raw_frame = self.reader.read_detector_frame(frame_idx)
            partner_raw = self.reader.read_detector_frame(partner_idx) if partner_idx is not None else None
        except Exception:
            raw_frame = None
            partner_raw = None

        if raw_frame is None or (partner_idx is not None and partner_raw is None):
            return

        if partner_raw is not None:
            raw_combined = np.asarray(raw_frame, dtype=float) + np.asarray(partner_raw, dtype=float)
        else:
            raw_combined = np.asarray(raw_frame, dtype=float)

        # Rotate 90 degrees clockwise and mirror horizontally:
        # np.fliplr(np.rot90(raw_combined, -1)) is identical to raw_combined.T (zero-copy view).
        frame = raw_combined.T
        total_counts = float(np.sum(raw_combined))
        self._frame_data = frame
        self._frame_total_counts = total_counts

        # If auto-fit is active, update displayed ROI box to current frame's fitted position
        if self._auto_fit_active and self._per_frame_rois is not None and frame_idx < len(self._per_frame_rois):
            cur_roi = self._per_frame_rois[frame_idx]
            rx0, rx1, ry0, ry1 = cur_roi
            pos_x = min(rx0, rx1)
            pos_y = min(ry0, ry1)
            size_x = max(1.0, abs(rx1 - rx0))
            size_y = max(1.0, abs(ry1 - ry0))
            if self._roi_rect_item is not None:
                self._updating_roi_internally = True
                self._roi_rect_item.setPos([pos_x, pos_y])
                self._roi_rect_item.setSize([size_x, size_y])
                self._updating_roi_internally = False
            self._sync_spinboxes_from_bounds(pos_x, pos_x + size_x, pos_y, pos_y + size_y)

        # Update metadata banner
        th_val = self.scan_data.variables["th"].data[frame_idx] if "th" in self.scan_data.variables else 0.0
        tth_val = self.scan_data.variables["tth"].data[frame_idx] if "tth" in self.scan_data.variables else 0.0
        flip_val = self.scan_data.variables["Flipper1"].data[frame_idx] if "Flipper1" in self.scan_data.variables else np.nan

        if partner_idx is not None:
            self.lbl_frame_info.setText(
                f"Frames {frame_idx}+{partner_idx}   θ {th_val:.4f}°   2θ {tth_val:.4f}°   u+d (sum)   {total_counts:,.0f} counts")
        else:
            spin_str = {0: "d (ON)", 1: "u (OFF)"}.get(flip_val, "spin ?")
            self.lbl_frame_info.setText(
                f"Frame {frame_idx}   θ {th_val:.4f}°   2θ {tth_val:.4f}°   {spin_str}   {total_counts:,.0f} counts")

        self._render_current_frame(raw_combined=raw_combined, total_counts=total_counts)
