"""2D Detector Frame Viewer with horizontal mirroring, interactive user ROI, and scan chart integration."""

from __future__ import annotations

from typing import Any
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sa_nxs_explorer.core.models import ScanData
from sa_nxs_explorer.core.nexus_reader import NexusReader
from sa_nxs_explorer.core.scan_calculations import integrate_roi_2d
from sa_nxs_explorer.ui import icons
from sa_nxs_explorer.ui.theme import apply_plot_theme, get_safe_colormap


class DetectorView(QWidget):
    """Visualizes 2D detector frames oriented and mirrored with interactive user-defined ROI."""

    # Signal emitted when user applies a new ROI: passes (roi_counts_array, (x0, x1, y0, y1))
    roi_applied = pyqtSignal(object, tuple)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.reader: NexusReader | None = None
        self.scan_data: ScanData | None = None
        self.is_dark_mode: bool = False
        self._current_frame_idx: int = 0
        self._total_frames: int = 0
        self._current_frame_data: np.ndarray | None = None
        self._filtered_indices: list[int] = []
        self._current_filtered_pos: int = 0

        # Animation timer for scan play
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(120)  # ~8 fps
        self._play_timer.timeout.connect(self._step_forward)

        self._roi_rect_item: pg.RectROI | None = None
        self._updating_roi_internally: bool = False

        # Per-frame auto-fit tracking state
        self._per_frame_rois: list[tuple[float, float, float, float]] | None = None
        self._per_frame_counts: np.ndarray | None = None
        self._auto_fit_active: bool = False

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # ---------------- Top Controls / Scrubber Bar ----------------
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(8)

        self.btn_play = QPushButton("Play")
        self.btn_play.setIcon(icons.get_icon("play", is_dark=self.is_dark_mode))
        self.btn_play.setFixedWidth(85)
        self.btn_play.clicked.connect(self._toggle_play)
        ctrl_bar.addWidget(self.btn_play)

        self.btn_prev = QPushButton("Prev")
        self.btn_prev.setIcon(icons.get_icon("chevron_left", is_dark=self.is_dark_mode))
        self.btn_prev.setFixedWidth(75)
        self.btn_prev.clicked.connect(self._step_backward)
        ctrl_bar.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next")
        self.btn_next.setIcon(icons.get_icon("chevron_right", is_dark=self.is_dark_mode))
        self.btn_next.setFixedWidth(75)
        self.btn_next.clicked.connect(self._step_forward)
        ctrl_bar.addWidget(self.btn_next)

        ctrl_bar.addWidget(QLabel("Spin:"))
        self.combo_spin_filter = QComboBox()
        self.combo_spin_filter.addItem("All Frames", "all")
        self.combo_spin_filter.addItem("u — Spin Up (F=1)", "up")
        self.combo_spin_filter.addItem("d — Spin Down (F=0)", "down")
        self.combo_spin_filter.currentIndexChanged.connect(self._on_spin_filter_changed)
        ctrl_bar.addWidget(self.combo_spin_filter)

        ctrl_bar.addWidget(QLabel("Step:"))
        self.spin_step = QSpinBox()
        self.spin_step.setRange(0, 0)
        self.spin_step.setFixedWidth(70)
        self.spin_step.valueChanged.connect(self._on_step_spin_changed)
        ctrl_bar.addWidget(self.spin_step)

        self.slider_step = QSlider(Qt.Orientation.Horizontal)
        self.slider_step.setRange(0, 0)
        self.slider_step.valueChanged.connect(self._on_slider_changed)
        ctrl_bar.addWidget(self.slider_step, 1)

        self.chk_log_img = QCheckBox("Log Scale")
        self.chk_log_img.setChecked(True)
        self.chk_log_img.toggled.connect(self._refresh_current_frame)
        ctrl_bar.addWidget(self.chk_log_img)

        self.chk_show_roi = QCheckBox("Show ROI")
        self.chk_show_roi.setChecked(True)
        self.chk_show_roi.toggled.connect(self._toggle_roi_visibility)
        ctrl_bar.addWidget(self.chk_show_roi)

        ctrl_bar.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["turbo", "viridis", "inferno", "plasma", "cividis", "gray", "magma"])
        self.combo_cmap.setCurrentText("turbo")
        self.combo_cmap.currentTextChanged.connect(self._on_cmap_changed)
        ctrl_bar.addWidget(self.combo_cmap)

        main_layout.addLayout(ctrl_bar)

        # Step details banner
        self.lbl_step_info = QLabel("No detector data loaded.")
        self.lbl_step_info.setStyleSheet(
            "background-color: #0284c7; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: 600;"
        )
        main_layout.addWidget(self.lbl_step_info)

        # ---------------- Central Splitter: 2D Image + ROI Controls & 1D Profiles ----------------
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left / Center: 2D Detector Image
        img_container = QWidget()
        img_layout = QVBoxLayout(img_container)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.setSpacing(6)

        # Interactive 2D Image Toolbar
        img_toolbar = QHBoxLayout()
        img_toolbar.setSpacing(6)

        self.btn_img_pan = QPushButton("Pan")
        self.btn_img_pan.setCheckable(True)
        self.btn_img_pan.setChecked(True)
        self.btn_img_pan.setToolTip("Pan Mode: Drag with left mouse button to pan detector image")
        self.btn_img_pan.clicked.connect(self._set_img_mouse_mode_pan)
        img_toolbar.addWidget(self.btn_img_pan)

        self.btn_img_box_zoom = QPushButton("Box Zoom")
        self.btn_img_box_zoom.setCheckable(True)
        self.btn_img_box_zoom.setToolTip("Box Zoom Mode: Drag with left mouse button to draw a zoom rectangle")
        self.btn_img_box_zoom.clicked.connect(self._set_img_mouse_mode_box_zoom)
        img_toolbar.addWidget(self.btn_img_box_zoom)

        self.img_mouse_group = QButtonGroup(self)
        self.img_mouse_group.addButton(self.btn_img_pan)
        self.img_mouse_group.addButton(self.btn_img_box_zoom)

        self.btn_img_zoom_in = QPushButton("Zoom In")
        self.btn_img_zoom_in.setToolTip("Zoom in (+30%)")
        self.btn_img_zoom_in.clicked.connect(self._img_zoom_in)
        img_toolbar.addWidget(self.btn_img_zoom_in)

        self.btn_img_zoom_out = QPushButton("Zoom Out")
        self.btn_img_zoom_out.setToolTip("Zoom out (-30%)")
        self.btn_img_zoom_out.clicked.connect(self._img_zoom_out)
        img_toolbar.addWidget(self.btn_img_zoom_out)

        self.btn_img_reset = QPushButton("Fit Frame")
        self.btn_img_reset.setToolTip("Reset view to fit full 2D detector frame")
        self.btn_img_reset.clicked.connect(self._img_reset_view)
        img_toolbar.addWidget(self.btn_img_reset)

        img_toolbar.addSpacing(10)

        self.btn_img_log = QPushButton("Log Scale")
        self.btn_img_log.setCheckable(True)
        self.btn_img_log.setChecked(self.chk_log_img.isChecked())
        self.btn_img_log.setToolTip("Toggle Logarithmic / Linear 2D intensity scaling")
        self.btn_img_log.toggled.connect(self._on_img_log_toggled)
        img_toolbar.addWidget(self.btn_img_log)

        img_toolbar.addStretch(1)

        self.lbl_img_coords = QLabel("Pixel: (X: ----, Y: ----)  |  Counts: --------")
        self.lbl_img_coords.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_img_coords.setFixedWidth(290)
        self.lbl_img_coords.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        img_toolbar.addWidget(self.lbl_img_coords)

        img_layout.addLayout(img_toolbar)

        self.plot_2d = pg.PlotWidget()
        self.plot_2d.setAspectLocked(True)
        self.plot_2d.showGrid(x=False, y=False)
        self.plot_2d.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.plot_2d.getViewBox().setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.plot_2d.scene().sigMouseMoved.connect(self._on_mouse_moved_2d)
        self.img_item = pg.ImageItem()
        self.img_item.setColorMap(get_safe_colormap("turbo"))
        self.plot_2d.addItem(self.img_item)

        apply_plot_theme(self.plot_2d, self.is_dark_mode)
        img_layout.addWidget(self.plot_2d)
        splitter.addWidget(img_container)

        # Right: Interactive ROI Definition & 1D Cuts
        sidebar_container = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        # 1. Interactive ROI Definition Card
        roi_ctrl_group = QGroupBox("Define Region of Interest (ROI)")
        roi_ctrl_layout = QVBoxLayout(roi_ctrl_group)
        roi_ctrl_layout.setSpacing(8)

        inputs_grid = QHBoxLayout()
        inputs_grid.setSpacing(6)

        # X bounds
        vbox_x = QVBoxLayout()
        vbox_x.addWidget(QLabel("X Min:"))
        self.spin_x0 = QSpinBox()
        self.spin_x0.setRange(0, 1023)
        self.spin_x0.valueChanged.connect(self._on_spin_roi_changed)
        vbox_x.addWidget(self.spin_x0)

        vbox_x.addWidget(QLabel("X Max:"))
        self.spin_x1 = QSpinBox()
        self.spin_x1.setRange(0, 1024)
        self.spin_x1.valueChanged.connect(self._on_spin_roi_changed)
        vbox_x.addWidget(self.spin_x1)
        inputs_grid.addLayout(vbox_x)

        # Y bounds
        vbox_y = QVBoxLayout()
        vbox_y.addWidget(QLabel("Y Min:"))
        self.spin_y0 = QSpinBox()
        self.spin_y0.setRange(0, 1023)
        self.spin_y0.valueChanged.connect(self._on_spin_roi_changed)
        vbox_y.addWidget(self.spin_y0)

        vbox_y.addWidget(QLabel("Y Max:"))
        self.spin_y1 = QSpinBox()
        self.spin_y1.setRange(0, 1024)
        self.spin_y1.valueChanged.connect(self._on_spin_roi_changed)
        vbox_y.addWidget(self.spin_y1)
        inputs_grid.addLayout(vbox_y)

        roi_ctrl_layout.addLayout(inputs_grid)

        # Action buttons for ROI
        btns_row = QHBoxLayout()
        self.btn_apply_roi = QPushButton("Apply ROI to Scan Chart")
        self.btn_apply_roi.setIcon(icons.get_icon("chart", is_dark=self.is_dark_mode))
        self.btn_apply_roi.setStyleSheet("font-weight: 700; color: #0284c7;")
        self.btn_apply_roi.clicked.connect(self._apply_roi_to_scan_chart)
        btns_row.addWidget(self.btn_apply_roi, 1)

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

        self.lbl_roi_status = QLabel("Drag yellow box on detector to adjust ROI.")
        self.lbl_roi_status.setStyleSheet("color: #64748b; font-size: 11px;")
        roi_ctrl_layout.addWidget(self.lbl_roi_status)

        self.lbl_roi_metrics = QLabel("ROI Counts: --")
        self.lbl_roi_metrics.setWordWrap(True)
        roi_ctrl_layout.addWidget(self.lbl_roi_metrics)

        sidebar_layout.addWidget(roi_ctrl_group)

        # 2. 1D Horizontal Cut
        sidebar_layout.addWidget(QLabel("Horizontal Profile (X-Cut through ROI):"))
        self.plot_x_cut = pg.PlotWidget()
        self.plot_x_cut.showGrid(x=True, y=True, alpha=0.3)
        self.plot_x_cut.setMaximumHeight(160)
        apply_plot_theme(self.plot_x_cut, self.is_dark_mode)
        sidebar_layout.addWidget(self.plot_x_cut)

        # 3. 1D Vertical Cut
        sidebar_layout.addWidget(QLabel("Vertical Profile (Y-Cut through ROI):"))
        self.plot_y_cut = pg.PlotWidget()
        self.plot_y_cut.showGrid(x=True, y=True, alpha=0.3)
        self.plot_y_cut.setMaximumHeight(160)
        apply_plot_theme(self.plot_y_cut, self.is_dark_mode)
        sidebar_layout.addWidget(self.plot_y_cut)

        sidebar_layout.addStretch()
        splitter.addWidget(sidebar_container)
        splitter.setSizes([680, 360])

        main_layout.addWidget(splitter, 1)

    def set_data(self, reader: NexusReader, scan_data: ScanData) -> None:
        """Assign reader and scan data to detector view."""
        self.reader = reader
        self.scan_data = scan_data

        if not scan_data.has_detector_2d or scan_data.detector_shape is None:
            self.lbl_step_info.setText("No 2D detector data found in this NeXus file.")
            self.slider_step.setEnabled(False)
            self.spin_step.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.btn_auto_fit.setEnabled(False)
            self.btn_apply_roi.setEnabled(False)
            self.combo_spin_filter.setEnabled(False)
            return

        self._per_frame_rois = None
        self._per_frame_counts = None
        self._auto_fit_active = False
        self.lbl_roi_status.setText("Drag yellow box on detector to adjust ROI, or click 'Auto-Fit ROI'.")
        self.lbl_roi_status.setStyleSheet("color: #64748b; font-size: 11px;")

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

        initial_roi = scan_data.working_detector_roi or scan_data.detector_roi or (354.0, 382.0, 285.0, 743.0)
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

        self._sync_spinboxes_from_bounds(pos_x, pos_x + size_x, pos_y, pos_y + size_y)

        self._update_spin_filter_options()
        self._update_filtered_indices(reset_pos=True)

    def _update_spin_filter_options(self) -> None:
        has_flipper = (
            self.scan_data is not None
            and "Flipper1" in self.scan_data.variables
            and len(self.scan_data.variables["Flipper1"].data) > 0
        )
        self.combo_spin_filter.setEnabled(has_flipper)

    def _on_spin_filter_changed(self) -> None:
        self._update_filtered_indices(reset_pos=False)

    def _update_filtered_indices(self, reset_pos: bool = False) -> None:
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

        if not self._filtered_indices:
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
        self._load_frame(raw_idx)

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
        self.btn_img_log.setIcon(icons.get_icon("activity", is_dark=is_dark))

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

    def _on_img_log_toggled(self, checked: bool) -> None:
        self.chk_log_img.setChecked(checked)

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
            self.btn_play.setText("Play")
            self.btn_play.setIcon(icons.get_icon("play", is_dark=self.is_dark_mode))
        else:
            self._play_timer.start()
            self.btn_play.setText("Pause")
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
        cm = get_safe_colormap(cmap_name)
        self.img_item.setColorMap(cm)

    def _toggle_roi_visibility(self, visible: bool) -> None:
        if self._roi_rect_item is not None:
            self._roi_rect_item.setVisible(visible)

    def _refresh_current_frame(self) -> None:
        self.btn_img_log.blockSignals(True)
        self.btn_img_log.setChecked(self.chk_log_img.isChecked())
        self.btn_img_log.blockSignals(False)
        self._load_frame(self._current_frame_idx)

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
        self._apply_roi_to_scan_chart()
        self.lbl_roi_status.setText("Reset to default file ROI.")
        self.lbl_roi_status.setStyleSheet("color: #64748b; font-size: 11px;")

    def _apply_roi_to_scan_chart(self) -> None:
        """Compute full scan integration for the current ROI and emit roi_applied."""
        if self.reader is None or self.scan_data is None:
            return

        if self._auto_fit_active and self._per_frame_counts is not None and self._per_frame_rois is not None:
            roi_counts = self._per_frame_counts
            cur_roi = self._per_frame_rois[self._current_frame_idx] if 0 <= self._current_frame_idx < len(self._per_frame_rois) else self._get_current_roi_bounds()
            self.roi_applied.emit(roi_counts, cur_roi)
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
            self.roi_applied.emit(roi_counts, roi_bounds)
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
            self._update_profiles_and_counts(min_x, min_x + size_x, min_y, min_y + size_y)

        centers = [(r[0] + r[1]) / 2.0 for r in rois]
        mean_c = float(np.mean(centers))
        std_c = float(np.std(centers))
        self.lbl_roi_status.setText(
            f"Auto-Fit active: {len(rois)} frames tracked (Peak X: {mean_c:.1f} ± {std_c:.1f} px). Applied to Scan Chart!"
        )
        self.lbl_roi_status.setStyleSheet("color: #10b981; font-weight: 600; font-size: 11px;")

        # Automatically apply to scan chart
        cur_roi = rois[self._current_frame_idx] if 0 <= self._current_frame_idx < len(rois) else initial_roi
        self.roi_applied.emit(counts, cur_roi)

    def _update_profiles_and_counts(self, x0: float, x1: float, y0: float, y1: float) -> None:
        if self._current_frame_data is None:
            return

        frame = self._current_frame_data
        total_counts = float(np.sum(frame))
        roi_counts, x_profile, y_profile = integrate_roi_2d(frame, (x0, x1, y0, y1))

        pct = (roi_counts / total_counts * 100.0) if total_counts > 0 else 0.0
        roi_text = (
            f"<b>ROI Box:</b> X [{x0:.0f} .. {x1:.0f}], Y [{y0:.0f} .. {y1:.0f}]<br>"
            f"<b>ROI Counts:</b> {roi_counts:,.0f} ({pct:.1f}% of frame)<br>"
            f"<b>Peak in ROI:</b> {np.max(x_profile) if len(x_profile) else 0:.0f} cts"
        )
        self.lbl_roi_metrics.setText(roi_text)

        # Plot horizontal cut
        self.plot_x_cut.clear()
        pen_x = "#38bdf8" if self.is_dark_mode else "#0284c7"
        x_coords = np.arange(int(round(min(x0, x1))), int(round(min(x0, x1))) + len(x_profile))
        self.plot_x_cut.plot(x_coords, x_profile, pen=pg.mkPen(pen_x, width=2), symbol="o", symbolSize=3)
        self.plot_x_cut.setLabel("bottom", "X Pixel", color="#f1f5f9" if self.is_dark_mode else "#0f172a")
        self.plot_x_cut.setLabel("left", "Counts", color="#f1f5f9" if self.is_dark_mode else "#0f172a")

        # Plot vertical cut
        self.plot_y_cut.clear()
        pen_y = "#f43f5e" if self.is_dark_mode else "#e11d48"
        y_coords = np.arange(int(round(min(y0, y1))), int(round(min(y0, y1))) + len(y_profile))
        self.plot_y_cut.plot(y_coords, y_profile, pen=pg.mkPen(pen_y, width=2), symbol="o", symbolSize=3)
        self.plot_y_cut.setLabel("bottom", "Y Pixel", color="#f1f5f9" if self.is_dark_mode else "#0f172a")
        self.plot_y_cut.setLabel("left", "Counts", color="#f1f5f9" if self.is_dark_mode else "#0f172a")

    def _load_frame(self, frame_idx: int) -> None:
        if self.reader is None or self.scan_data is None:
            return

        self._current_frame_idx = frame_idx
        try:
            raw_frame = self.reader.read_detector_frame(frame_idx)
        except Exception:
            raw_frame = None

        if raw_frame is None:
            return

        # Rotate 90 degrees clockwise and mirror horizontally as requested:
        # np.fliplr(np.rot90(raw_frame, -1)) is equivalent to raw_frame.T
        frame_rot = np.rot90(raw_frame, -1)
        frame_mir = np.fliplr(frame_rot)
        frame = np.asarray(frame_mir, dtype=float)
        self._current_frame_data = frame

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
        total_counts = float(np.sum(frame))

        spin_str = {0: "d — Down (F=0 / OFF)", 1: "u — Up (F=1 / ON)"}.get(flip_val, "Unknown flipper state")
        filt_count = len(self._filtered_indices) if self._filtered_indices else self._total_frames
        step_pos = (self._current_filtered_pos + 1) if self._filtered_indices else (frame_idx + 1)
        info_str = (
            f"Step {step_pos} of {filt_count} (Frame #{frame_idx + 1})  |  "
            f"θ = {th_val:.4f}°  |  2θ = {tth_val:.4f}°  |  "
            f"Spin: {spin_str}  |  Frame Counts: {total_counts:,.0f}  |  "
            f"90° CW + Mirrored"
        )
        self.lbl_step_info.setText(info_str)

        # Prepare 2D Image
        img_display = frame
        if self.chk_log_img.isChecked():
            img_display = np.log10(np.maximum(img_display, 1.0))

        # In pyqtgraph ImageItem displays array with transposed axis mapping
        self.img_item.setImage(img_display.T)
        cm = get_safe_colormap(self.combo_cmap.currentText())
        self.img_item.setColorMap(cm)

        # Compute ROI integration & 1D profiles on the frame
        x0, x1, y0, y1 = self._get_current_roi_bounds()
        self._update_profiles_and_counts(x0, x1, y0, y1)

        apply_plot_theme(self.plot_2d, self.is_dark_mode)
        apply_plot_theme(self.plot_x_cut, self.is_dark_mode)
        apply_plot_theme(self.plot_y_cut, self.is_dark_mode)
