"""Off-specular map tab: every detector frame integrated along Y, stacked by scan point.

Rows are scan points (detector frames), columns are displayed detector X pixels,
matching the 2D Detector tab orientation (displayed = raw.T). The heatmap is the
pySAred "Pixel vs. Point" map for SuperADAM NeXus files.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Event

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor, QImage, QPainter, QTransform
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pySAview.core.angle_calibration import (
    AngleCalibration,
    CalibrationError,
    alpha_f_offsets,
    alpha_i_strip_edges,
    angles_from_q,
    calibrate,
    q_from_angles,
    specular_center,
)
from pySAview.core.models import ScanData
from pySAview.core.nexus_reader import NexusReader
from pySAview.core.off_specular import (
    DETECTOR_PATH,
    AsymmetryMap,
    pair_spin_frames,
    spin_asymmetry,
    OffSpecularMap,
    compute_off_specular_map,
    frame_y_bounds,
    log_display,
    normalize_map,
    select_frames,
)
from pySAview.core.export_tables import map_table, write_table
from pySAview.ui import icons
from pySAview.ui.plot_tools import (
    TABLE_FILTERS, ask_output_path, export_plot_image, install_context_export, tool_button,
)
from pySAview.ui.theme import apply_plot_theme, get_safe_colormap

DEFAULT_ROI = (354.0, 382.0, 285.0, 743.0)


QX_SCALE = 1e3  # Qx is shown in 10⁻³ Å⁻¹ (it is ~100× smaller than Qz)


def _decimals_for_step(step: float) -> int:
    """Fewest decimals that still resolve ``step`` (e.g. 0.0097 -> 2, 0.04 -> 2, 0.0082 -> 3)."""
    if not np.isfinite(step) or step <= 0:
        return 4
    return int(min(6, max(0, np.ceil(-np.log10(step) - 0.05))))


_tool_button = tool_button


CUT_HINT = "Turn on the crosshair tool (⌖ in the map toolbar) and click the map to drop the cuts."


def _fixed_width_text(label: QLabel) -> None:
    """Changing text must never resize the layout (the map would jump while hovering)."""
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setMinimumWidth(0)


def _same_transform(a: QTransform, b: QTransform, tol: float = 1e-6) -> bool:
    va = (a.m11(), a.m12(), a.m21(), a.m22(), a.dx(), a.dy())
    vb = (b.m11(), b.m12(), b.m21(), b.m22(), b.dx(), b.dy())
    return all(abs(x - y) <= tol * max(1.0, abs(x), abs(y)) for x, y in zip(va, vb))


class CachedMeshItem(pg.PColorMeshItem):
    """Exact-cell mesh that blits rendered images while the view moves.

    ``PColorMeshItem`` replays one polygon per cell on every repaint (~150k for a
    Qx–Qz map, tens of ms), which makes panning, zooming and crosshair dragging
    stutter. Here the same cells are rendered once into images: a whole-map
    overview and a screen-resolution image of the current view (plus a margin).
    While the view moves those images are only transformed; once it has been
    still for ``DETAIL_DELAY_MS`` the view image is re-rendered pixel-exact.
    Display only: the cell geometry and values are unchanged, and exports always
    draw the cells directly.
    """

    DETAIL_DELAY_MS = 120
    MARGIN = 0.25  # fraction of the view rendered beyond each edge, so short pans stay sharp
    OVERVIEW_MAX_PX = 4096

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._picture_used = None
        self._overview = None  # (QImage, item-coordinate rect)
        self._detail = None  # (QImage, item-coordinate rect, device transform, device offset)
        self._last_view = None  # (device transform, viewport size, device pixel ratio)
        self._detail_timer = QTimer()
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(self.DETAIL_DELAY_MS)
        self._detail_timer.timeout.connect(self._render_detail)

    def _render(self, size: QSize, dpr: float, transform: QTransform) -> QImage:
        image = QImage(size * dpr, QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(dpr)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setTransform(transform)
        painter.drawPicture(0, 0, self.qpicture)
        painter.end()
        return image

    def _render_overview(self, viewport: QSize, dpr: float) -> None:
        rect = self.boundingRect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        limit = self.OVERVIEW_MAX_PX / dpr
        w = int(min(limit, max(64, 2 * viewport.width())))
        h = int(min(limit, max(64, 2 * viewport.height())))
        transform = QTransform.fromScale(w / rect.width(), h / rect.height())
        transform.translate(-rect.left(), -rect.top())
        self._overview = (self._render(QSize(w, h), dpr, transform), QRectF(rect))

    def _render_detail(self) -> None:
        if self._last_view is None or self.qpicture is None or self.scene() is None:
            return
        transform, viewport, dpr = self._last_view
        mx, my = int(viewport.width() * self.MARGIN), int(viewport.height() * self.MARGIN)
        shifted = transform * QTransform.fromTranslate(mx, my)
        image = self._render(QSize(viewport.width() + 2 * mx, viewport.height() + 2 * my), dpr, shifted)
        inverse, ok = transform.inverted()
        if not ok:
            return
        device = QRectF(-mx, -my, viewport.width() + 2 * mx, viewport.height() + 2 * my)
        self._detail = (image, inverse.mapRect(device), QTransform(transform), QPointF(-mx, -my))
        self.update()

    def paint(self, painter, opt, widget):
        if self.z is None:
            return
        if self.qpicture is None:
            self.qpicture = self._drawPicture()
        if self._exportOpts is not False or widget is None:
            painter.drawPicture(0, 0, self.qpicture)  # exports: draw the exact cells
            return
        if self._picture_used is not self.qpicture:  # new data, levels or colours
            self._picture_used = self.qpicture
            self._overview = self._detail = None
        transform = painter.transform()
        viewport = widget.size()
        dpr = widget.devicePixelRatioF()
        self._last_view = (QTransform(transform), viewport, dpr)
        detail = self._detail
        if detail is not None and _same_transform(detail[2], transform):
            painter.save()
            painter.resetTransform()
            painter.drawImage(detail[3], detail[0])  # pixel-exact for this view
            painter.restore()
            return
        if self._overview is None:
            self._render_overview(viewport, dpr)
        if self._overview is not None:
            painter.drawImage(self._overview[1], self._overview[0])
        if detail is not None:
            painter.drawImage(detail[1], detail[0])
        self._detail_timer.start()  # sharpen once the view has stopped moving


class OffSpecularThread(QThread):
    """Integrates the detector volume with its own read-only handle."""

    progress = pyqtSignal(int, int, int)
    computed = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, token: int, path: str, y_range, parent=None) -> None:
        super().__init__(parent)
        self.token = token
        self.path = path
        self.y_range = y_range
        self.cancelled = Event()

    def run(self) -> None:
        try:
            with NexusReader(self.path) as reader:
                if DETECTOR_PATH not in reader.file:
                    raise ValueError("No 2D detector data in this file")
                result = compute_off_specular_map(
                    reader.file[DETECTOR_PATH],
                    self.y_range,
                    cancelled=self.cancelled.is_set,
                    progress=lambda done, total: self.progress.emit(self.token, done, total),
                    source_path=self.path,
                )
            if not self.cancelled.is_set():
                self.computed.emit(self.token, result)
        except InterruptedError:
            pass
        except Exception as error:  # reported in the banner
            if not self.cancelled.is_set():
                self.failed.emit(self.token, str(error))


class OffSpecularView(QWidget):
    """Heatmap of Y-integrated detector intensity versus X pixel and scan point."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scan_data: ScanData | None = None
        self.is_dark_mode = False
        self._map: OffSpecularMap | None = None
        self._norm = None
        self._cache: dict[tuple, OffSpecularMap] = {}
        self._jobs: dict[int, OffSpecularThread] = {}
        self._token = 0
        self._pending_key: tuple | None = None
        self._dirty = False
        self._sel_row = 0
        self._sel_x = 0
        self._crosshair_placed = False
        self._roi_items: list[pg.PlotDataItem] = []
        self._alpha_items: list[pg.ImageItem] = []
        self._alpha_geom: dict | None = None
        self._cal_message = ""
        self._colorbar_on_strips = False
        self.session_source_paths = lambda: []  # every loaded file; set by the main window
        self._cut_values: np.ndarray | None = None  # what the map shows (asymmetry: low counts blanked)
        self._init_ui()
        self._set_enabled(False)

    # ------------------------------------------------------------------ layout
    def _init_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(8)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(QLabel("Spin:"))
        self.combo_spin = QComboBox()
        self.combo_spin.addItem("All Frames", "all")
        self.combo_spin.addItem("u — Spin Up (F=1)", "up")
        self.combo_spin.addItem("d — Spin Down (F=0)", "down")
        self.combo_spin.addItem("Spin asymmetry (u−d)/(u+d)", "asym")
        self.combo_spin.currentIndexChanged.connect(self._on_spin_changed)
        bar.addWidget(self.combo_spin)
        self.spin_min_counts = QSpinBox()
        self.spin_min_counts.setRange(0, 1_000_000)
        self.spin_min_counts.setValue(10)
        self.spin_min_counts.setPrefix("u+d ≥ ")
        self.spin_min_counts.setSuffix(" cts")
        self.spin_min_counts.setKeyboardTracking(False)
        self.spin_min_counts.setToolTip("Asymmetry display: pixels with fewer raw counts in u + d are left blank "
                                        "(values and CSV are unchanged)")
        self.spin_min_counts.setVisible(False)
        self.spin_min_counts.valueChanged.connect(self._redraw)
        bar.addWidget(self.spin_min_counts)

        bar.addWidget(QLabel("Integrate Y:"))
        self.combo_y_source = QComboBox()
        self.combo_y_source.addItem("Full detector height", "full")
        self.combo_y_source.addItem("Scan ROI Y range", "roi")
        self.combo_y_source.addItem("Custom Y range", "custom")
        self.combo_y_source.setToolTip("Rows of each detector image summed for every X pixel")
        self.combo_y_source.currentIndexChanged.connect(self._on_y_source_changed)
        bar.addWidget(self.combo_y_source)
        # The row spinboxes always show the Y window being integrated. Editing
        # them switches to "Custom Y range" (e.g. to exclude noisy detector edges).
        bar.addWidget(QLabel("Rows:"))
        self.spin_y0 = QSpinBox()
        self.spin_y1 = QSpinBox()
        for spin, tip in ((self.spin_y0, "First detector Y row integrated (inclusive)"),
                          (self.spin_y1, "Last detector Y row integrated (exclusive)")):
            spin.setRange(0, 1024)
            spin.setFixedWidth(88)
            spin.setKeyboardTracking(False)
            spin.setToolTip(tip + ". Editing switches to Custom Y range.")
            spin.valueChanged.connect(self._on_row_spin_changed)
            bar.addWidget(spin)
        self._row_timer = QTimer(self)
        self._row_timer.setSingleShot(True)
        self._row_timer.setInterval(350)  # wait for arrow-key/click bursts before recomputing
        self._row_timer.timeout.connect(self._request_compute)

        bar.addWidget(QLabel("Normalize:"))
        self.combo_norm = QComboBox()
        self.combo_norm.addItem("Raw counts", "raw")
        self.combo_norm.addItem("Counts / s", "time")
        self.combo_norm.addItem("Counts / Monitor1", "monitor")
        self.combo_norm.currentIndexChanged.connect(self._redraw)
        bar.addWidget(self.combo_norm)

        bar.addWidget(QLabel("Axes:"))
        self.combo_y_axis = QComboBox()
        self.combo_y_axis.addItem("X pixel vs Point", "point")
        self.combo_y_axis.addItem("X pixel vs θ", "th")
        self.combo_y_axis.addItem("α_i vs α_f", "alpha")
        self.combo_y_axis.addItem("Qx vs Qz", "q")
        self.combo_y_axis.setToolTip("α_i vs α_f and Qx vs Qz use this scan's angle calibration "
                                     "(right-hand panel); Qx–Qz also uses λ")
        self.combo_y_axis.currentIndexChanged.connect(self._on_axes_changed)
        bar.addWidget(self.combo_y_axis)

        self.chk_log = QCheckBox("Log Scale")
        self.chk_log.setChecked(True)
        self.chk_log.toggled.connect(self._redraw)
        bar.addWidget(self.chk_log)
        self.chk_show_roi = QCheckBox("Show ROI")
        self.chk_show_roi.setChecked(True)
        self.chk_show_roi.toggled.connect(self._draw_roi_overlay)
        bar.addWidget(self.chk_show_roi)

        bar.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["turbo", "viridis", "inferno", "plasma", "cividis", "gray", "magma"])
        self.combo_cmap.currentTextChanged.connect(self._on_cmap_changed)
        bar.addWidget(self.combo_cmap)
        bar.addStretch(1)
        main.addLayout(bar)

        # Only problems and caveats are shown here (hidden otherwise); the settings
        # above already describe what the map shows.
        self.lbl_notice = QLabel("")
        self.lbl_notice.setStyleSheet("color: #b45309; font-size: 11px; padding: 0 2px;")
        self.lbl_notice.setWordWrap(True)
        _fixed_width_text(self.lbl_notice)
        self.lbl_notice.setVisible(False)
        main.addWidget(self.lbl_notice)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        tools = QHBoxLayout()
        tools.setSpacing(6)
        # Icon-only map tools (names in tooltips) to leave room for the readout.
        self.btn_pan = _tool_button("Pan: drag to move the map")
        self.btn_box_zoom = _tool_button("Box zoom: drag a rectangle to zoom")
        for btn in (self.btn_pan, self.btn_box_zoom):
            btn.setCheckable(True)
            tools.addWidget(btn)
        self.btn_pan.setChecked(True)
        self.mouse_group = QButtonGroup(self)
        self.mouse_group.addButton(self.btn_pan)
        self.mouse_group.addButton(self.btn_box_zoom)
        self.btn_pan.clicked.connect(lambda: self.plot.getViewBox().setMouseMode(pg.ViewBox.PanMode))
        self.btn_box_zoom.clicked.connect(lambda: self.plot.getViewBox().setMouseMode(pg.ViewBox.RectMode))
        self.btn_crosshair = _tool_button("Crosshair: click the map to drop it, drag its centre to move "
                                          "the cuts, click elsewhere to jump; toggle off to remove")
        self.btn_crosshair.setCheckable(True)
        self.btn_crosshair.toggled.connect(self._on_crosshair_toggled)
        tools.addWidget(self.btn_crosshair)
        self.btn_zoom_in = _tool_button("Zoom in")
        self.btn_zoom_in.clicked.connect(lambda: self.plot.getViewBox().scaleBy((0.7, 0.7)))
        self.btn_zoom_out = _tool_button("Zoom out")
        self.btn_zoom_out.clicked.connect(lambda: self.plot.getViewBox().scaleBy((1.4, 1.4)))
        self.btn_fit = _tool_button("Fit the whole map")
        self.btn_fit.clicked.connect(lambda: self.plot.autoRange())
        self.btn_export_png = _tool_button("Export the map as a PNG or SVG image")
        self.btn_export_png.clicked.connect(self.export_image)
        for btn in (self.btn_zoom_in, self.btn_zoom_out, self.btn_fit, self.btn_export_png):
            tools.addWidget(btn)
        tools.addSpacing(8)
        self.lbl_coords = QLabel("")
        self.lbl_coords.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_coords.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _fixed_width_text(self.lbl_coords)
        tools.addWidget(self.lbl_coords, 1)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setTextVisible(False)
        self.progress.setToolTip("Integrating detector frames")
        self.progress.setVisible(False)
        tools.addWidget(self.progress)
        left_layout.addLayout(tools)

        self.plot = pg.PlotWidget()
        self.plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.plot.getViewBox().setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.plot.setLabel("bottom", "Detector X pixel")
        self.plot.setLabel("left", "Point")
        self.img_item = pg.ImageItem()
        self.img_item.setColorMap(get_safe_colormap("turbo"))
        self.plot.addItem(self.img_item)
        self.colorbar = pg.ColorBarItem(values=(0, 1), colorMap=get_safe_colormap("turbo"), interactive=True, width=14)
        self.colorbar.setImageItem(self.img_item, insert_in=self.plot.getPlotItem())
        # Crosshair: guide lines follow one draggable centre marker; hidden until dropped.
        guide = pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DashLine)
        self.line_row = pg.InfiniteLine(angle=0, movable=False, pen=guide)
        self.line_x = pg.InfiniteLine(angle=90, movable=False, pen=guide)
        self.crosshair = pg.TargetItem(pos=(0, 0), size=16, symbol="crosshair", movable=True,
                                       pen=pg.mkPen("#ffffff", width=2), hoverPen=pg.mkPen("#facc15", width=2),
                                       brush=pg.mkBrush(0, 0, 0, 0))
        for item, z in ((self.line_row, 20), (self.line_x, 20), (self.crosshair, 21)):
            item.setZValue(z)
            item.setVisible(False)
            self.plot.addItem(item)
        self.crosshair.sigPositionChanged.connect(self._on_crosshair_moved)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._on_map_clicked)
        apply_plot_theme(self.plot, self.is_dark_mode)
        self.plot.showGrid(x=False, y=False)  # see set_theme
        self._apply_no_data_background()
        left_layout.addWidget(self.plot, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(self._build_calibration_panel())
        group = QGroupBox("Cuts through the map")
        group_layout = QVBoxLayout(group)
        self.lbl_row_cut = QLabel("Horizontal cut (selected point):")
        _fixed_width_text(self.lbl_row_cut)
        group_layout.addWidget(self.lbl_row_cut)
        self.plot_row = pg.PlotWidget()
        self.plot_row.setLabel("bottom", "Detector X pixel")
        group_layout.addWidget(self.plot_row)
        self.lbl_col_cut = QLabel("Vertical cut (selected X pixel):")
        _fixed_width_text(self.lbl_col_cut)
        group_layout.addWidget(self.lbl_col_cut)
        self.plot_col = pg.PlotWidget()
        group_layout.addWidget(self.plot_col)
        self.lbl_cut_hint = QLabel(CUT_HINT)
        self.lbl_cut_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_cut_hint.setWordWrap(True)
        _fixed_width_text(self.lbl_cut_hint)
        group_layout.addWidget(self.lbl_cut_hint)
        for p in (self.plot_row, self.plot_col):
            apply_plot_theme(p, self.is_dark_mode)
        for widget, kind in ((self.plot, "offspecular"), (self.plot_row, "offspecular_hcut"),
                             (self.plot_col, "offspecular_vcut")):
            install_context_export(widget, (lambda k=kind: self._default_export_name(k, ".png")),
                                   self._source_paths)
        self._row_curve = self.plot_row.plot([], [], connect="finite")
        self._col_curve = self.plot_col.plot([], [], connect="finite")
        right_layout.addWidget(group, 1)
        splitter.addWidget(right)
        splitter.setSizes([760, 340])
        main.addWidget(splitter, 1)

    def _build_calibration_panel(self) -> QGroupBox:
        group = self.cal_group = QGroupBox("Angle calibration (α_i, α_f) — this scan")
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(6)
        self.spin_dpp = QDoubleSpinBox()
        self.spin_dpp.setDecimals(6)
        self.spin_dpp.setRange(0.0, 1.0)
        self.spin_dpp.setSingleStep(0.00001)
        self.spin_dpp.setSuffix(" °/px")
        self.spin_dpp.setSpecialValueText("not set")
        self.spin_dpp.setToolTip("Exit-angle step per detector X pixel (from the direct-beam fit; editable)")
        self.spin_x0 = QDoubleSpinBox()
        self.spin_x0.setDecimals(1)
        self.spin_x0.setRange(0.0, 4096.0)
        self.spin_x0.setSuffix(" px")
        self.spin_x0.setToolTip("Pixel of exit angle 2θ − θ (from the specular peak; editable)")
        for spin in (self.spin_dpp, self.spin_x0):
            spin.setKeyboardTracking(False)
        self.chk_af_increases = QCheckBox("α_f increases with X")
        self.chk_af_increases.setToolTip("Direction of the exit angle along displayed X (set by the fit)")
        self.spin_dpp.valueChanged.connect(lambda v: self._on_calibration_edited(deg_per_pixel=v))
        self.spin_x0.valueChanged.connect(lambda v: self._on_calibration_edited(x0=v))
        self.chk_af_increases.toggled.connect(lambda on: self._on_calibration_edited(sign=1 if on else -1))
        self.btn_fit_db = QPushButton("Fit Direct Beam")
        self.btn_fit_db.setToolTip("Refit °/pixel from the direct-beam streak and x₀ from the specular peak "
                                   "(discards manual values for this scan)")
        self.btn_fit_db.clicked.connect(self._refit_calibration)
        self.chk_show_db = QCheckBox("Show fit / reference lines")
        self.chk_show_db.setToolTip("Pixel axes: fitted direct-beam line. α axes: α_f = α_i (specular), "
                                    "α_f = −α_i (direct beam) and the ROI; untick for a clean α map")
        self.chk_show_db.setChecked(True)
        self.chk_show_db.toggled.connect(self._draw_roi_overlay)
        grid.addWidget(QLabel("Scale:"), 0, 0)
        grid.addWidget(self.spin_dpp, 0, 1)
        grid.addWidget(QLabel("x₀:"), 0, 2)
        grid.addWidget(self.spin_x0, 0, 3)
        grid.addWidget(self.chk_af_increases, 1, 0, 1, 2)
        grid.addWidget(self.btn_fit_db, 1, 2, 1, 2)
        grid.addWidget(self.chk_show_db, 2, 0, 1, 2)
        self.spin_wavelength = QDoubleSpinBox()
        self.spin_wavelength.setRange(0.1, 50.0)
        self.spin_wavelength.setDecimals(3)
        self.spin_wavelength.setSingleStep(0.01)
        self.spin_wavelength.setValue(5.21)  # SuperADAM monochromatic beam, as in Scan Charts
        self.spin_wavelength.setSuffix(" Å")
        self.spin_wavelength.setKeyboardTracking(False)
        self.spin_wavelength.setToolTip("Neutron wavelength used for Qx and Qz")
        self.spin_wavelength.valueChanged.connect(self._on_wavelength_changed)
        grid.addWidget(QLabel("λ:"), 2, 2)
        grid.addWidget(self.spin_wavelength, 2, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        self._set_cal_status("Calibrated automatically from the direct beam when the map is computed.")
        return group

    def _set_cal_status(self, text: str) -> None:
        """Calibration details stay available on hover without cluttering the panel."""
        self.cal_status = text
        self.cal_group.setToolTip(text)

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self.combo_spin, self.combo_y_source, self.combo_norm, self.combo_y_axis,
                       self.btn_export_png):
            widget.setEnabled(enabled)
        for widget in (self.spin_y0, self.spin_y1, self.btn_crosshair, self.spin_dpp, self.spin_x0,
                       self.chk_af_increases, self.btn_fit_db, self.chk_show_db, self.spin_wavelength):
            widget.setEnabled(enabled)

    # -------------------------------------------------------------- public API
    def set_data(self, scan_data: ScanData | None) -> None:
        """Show the active scan; the map is computed lazily when the tab is visible."""
        self._cancel_jobs()
        self.scan_data = scan_data
        self._map = None
        self._norm = None
        self._pending_key = None
        self.img_item.clear()
        self._clear_alpha_items()
        self._clear_roi_overlay()
        self._clear_cuts()
        self._cal_message = ""
        self._sync_cal_controls()
        has_det = bool(scan_data is not None and scan_data.has_detector_2d and scan_data.detector_shape
                       and len(scan_data.detector_shape) == 3)
        self._set_enabled(has_det)
        if not has_det:
            self._set_notice("No 2D detector data found in this NeXus file." if scan_data and scan_data.file_path
                             else "")
            self._dirty = False
            return
        self._sync_row_spins()
        self._update_spin_options()
        self._set_notice("")
        self._dirty = True
        if self.isVisible():
            self._request_compute()

    def refresh_roi(self) -> None:
        """Called after the scan's working ROI changes in the 2D Detector tab."""
        if self.scan_data is None or not self.scan_data.has_detector_2d:
            return
        if self.combo_y_source.currentData() == "roi":
            self._sync_row_spins()
            self._dirty = True
            if self.isVisible():
                self._request_compute()
        self._draw_roi_overlay()

    def set_theme(self, is_dark: bool) -> None:
        self.is_dark_mode = is_dark
        self.btn_pan.setIcon(icons.get_icon("hand", is_dark=is_dark))
        self.btn_box_zoom.setIcon(icons.get_icon("box_select", is_dark=is_dark))
        self.btn_zoom_in.setIcon(icons.get_icon("zoom_in", is_dark=is_dark))
        self.btn_zoom_out.setIcon(icons.get_icon("zoom_out", is_dark=is_dark))
        self.btn_fit.setIcon(icons.get_icon("maximize", is_dark=is_dark))
        self.btn_crosshair.setIcon(icons.get_icon("crosshair", is_dark=is_dark))
        self.btn_export_png.setIcon(icons.get_icon("download", is_dark=is_dark))
        for p in (self.plot, self.plot_row, self.plot_col):
            apply_plot_theme(p, is_dark)
        # No grid over the heatmap: grid lines across the map cost ~30 ms per repaint
        # (laggy pan/zoom/crosshair) and hide the data. The cut plots keep theirs.
        self.plot.showGrid(x=False, y=False)
        self._apply_no_data_background()
        self._update_y_ticks()
        self._update_cuts()

    def shutdown(self) -> None:
        """Cancel and wait for background integration (window closing)."""
        jobs = list(self._jobs.values())
        self._cancel_jobs()
        for job in jobs:
            job.wait()  # cancellation is checked between small frame chunks

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self._request_compute()

    # ------------------------------------------------------------- helpers
    def _scan_roi_bounds(self) -> list[tuple[float, float, float, float]]:
        """Per-frame ROI bounds for this scan (one entry for a fixed ROI)."""
        scan = self.scan_data
        record = scan.working_roi_record if scan is not None else None
        if record is not None:
            return [tuple(b) for b in record.bounds]
        roi = (scan.working_detector_roi or scan.detector_roi) if scan is not None else None
        return [tuple(roi or DEFAULT_ROI)]

    def _y_range(self):
        source = self.combo_y_source.currentData()
        if source == "full":
            return None
        if source == "custom":
            return (float(self.spin_y0.value()), float(self.spin_y1.value()))
        bounds = self._scan_roi_bounds()
        windows = [(float(b[2]), float(b[3])) for b in bounds]
        if len(set(windows)) > 1:
            return windows  # tracked ROI whose Y window changes per frame
        return windows[0]

    def _variable(self, name):
        if self.scan_data is None:
            return None
        var = self.scan_data.variables.get(name)
        return None if var is None else np.asarray(var.data, dtype=float)

    def _update_spin_options(self) -> None:
        flipper = self._variable("Flipper1")
        has_both = flipper is not None and np.any(flipper == 1) and np.any(flipper == 0)
        self.combo_spin.blockSignals(True)
        self.combo_spin.setEnabled(flipper is not None)
        if has_both and self.combo_spin.currentData() == "all":
            # Interleaved spin states make stripes; show one polarisation by default.
            self.combo_spin.setCurrentIndex(self.combo_spin.findData("up"))
        elif flipper is None:
            self.combo_spin.setCurrentIndex(0)
        self.combo_spin.blockSignals(False)
        asym = self._asym_mode()
        self.spin_min_counts.setVisible(asym)
        self.chk_log.setEnabled(not asym)
        self.combo_cmap.setEnabled(not asym)

    def _on_y_source_changed(self) -> None:
        self._sync_row_spins()
        self._row_timer.stop()
        self._request_compute()

    def _sync_row_spins(self) -> None:
        """Show the integrated Y window in the row spinboxes (custom values are kept, clipped)."""
        if self.scan_data is None or not self.scan_data.detector_shape:
            return
        ny = self.scan_data.detector_shape[2]
        source = self.combo_y_source.currentData()
        if source == "full":
            y0, y1 = 0, ny
        elif source == "roi":
            _, _, a, b = self._scan_roi_bounds()[0]
            y0, y1 = round(min(a, b)), round(max(a, b))
        else:
            y0, y1 = self.spin_y0.value(), self.spin_y1.value()
        for spin, value in ((self.spin_y0, y0), (self.spin_y1, y1)):
            spin.blockSignals(True)
            spin.setRange(0, ny)
            spin.setValue(int(max(0, min(ny, value))))
            spin.blockSignals(False)

    def _on_row_spin_changed(self) -> None:
        if self.combo_y_source.currentData() != "custom":
            self.combo_y_source.blockSignals(True)
            self.combo_y_source.setCurrentIndex(self.combo_y_source.findData("custom"))
            self.combo_y_source.blockSignals(False)
        self._row_timer.start()

    def _cache_key(self, y_bounds):
        path = self.scan_data.file_path
        try:
            mtime = os.stat(path).st_mtime_ns
        except OSError:
            mtime = None
        return (str(Path(path).resolve()), mtime, y_bounds)

    def _cancel_jobs(self) -> None:
        for job in self._jobs.values():
            job.cancelled.set()
        # Invalidate results already queued by cancelled workers.
        self._token += 1
        self._pending_key = None
        self.progress.setVisible(False)

    def _request_compute(self) -> None:
        if self.scan_data is None or not self.scan_data.has_detector_2d or not self.scan_data.detector_shape:
            return
        self._dirty = False
        n_frames, nx, ny = self.scan_data.detector_shape
        y_range = self._y_range()
        try:
            rows = frame_y_bounds(y_range, n_frames, nx, ny)
        except ValueError as error:
            self._set_notice(f"Off-specular map unavailable: {error}")
            return
        key = self._cache_key(tuple(map(tuple, rows.tolist())))
        if key == self._pending_key:
            return
        if key in self._cache:
            self._cancel_jobs()
            self._pending_key = None
            self._map = self._cache[key]
            self._redraw()
            return
        self._cancel_jobs()
        self._token += 1
        self._pending_key = key
        job = OffSpecularThread(self._token, self.scan_data.file_path, y_range, self)
        job.cache_key = key
        self._jobs[self._token] = job
        job.progress.connect(self._on_progress)
        job.computed.connect(self._on_computed)
        job.failed.connect(self._on_failed)
        job.finished.connect(self._on_job_finished)
        self.progress.setRange(0, self.scan_data.detector_shape[0])
        self.progress.setValue(0)
        self.progress.setVisible(True)
        job.start()

    def _on_progress(self, token, done, total) -> None:
        if token == self._token:
            self.progress.setRange(0, total)
            self.progress.setValue(done)

    def _on_computed(self, token, result) -> None:
        job = self._jobs.get(token)
        if job is not None:
            if len(self._cache) >= 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[job.cache_key] = result
        if token != self._token:
            return
        self._pending_key = None
        self.progress.setVisible(False)
        self._map = result
        self._redraw()
        self.plot.autoRange()

    def _on_failed(self, token, message) -> None:
        if token != self._token:
            return
        self._pending_key = None
        self.progress.setVisible(False)
        self._set_notice(f"Off-specular map failed: {message}")

    def _on_job_finished(self) -> None:
        job = self.sender()
        self._jobs.pop(getattr(job, "token", None), None)
        if job is not None:
            job.deleteLater()

    # ------------------------------------------------------------- drawing
    def _redraw(self) -> None:
        if self._map is None or self.scan_data is None:
            return
        mode = self.combo_norm.currentData()
        spin = self.combo_spin.currentData()
        flipper = self._variable("Flipper1")
        exposure = None
        count = self.scan_data.variables.get("Count")
        monitor = self.scan_data.variables.get("Monitor1")
        try:
            if mode == "time":
                # Same exposure rule as the Scan Charts: seconds only, no raw fallback.
                if count is None or count.unit != "s":
                    raise ValueError("Counts / s needs a Count exposure channel in seconds")
                exposure = np.asarray(count.data, dtype=float)
            kw = dict(exposure=exposure,
                      monitor=None if monitor is None else np.asarray(monitor.data, dtype=float),
                      monitor_variance=None if monitor is None else monitor.variance)
            if spin == "asym":
                th, tth = self._variable("th"), self._variable("tth")
                if flipper is None or th is None or tth is None:
                    self._show_unavailable("Spin asymmetry needs Flipper1, θ and 2θ channels.")
                    return
                up, down = pair_spin_frames(th, tth, flipper, self._map.written)
                if up.size == 0:
                    self._show_unavailable("Spin asymmetry: no u/d frame pairs at matching θ and 2θ.")
                    return
                norm = spin_asymmetry(self._map.counts, up, down, mode,
                                      min_counts=self.spin_min_counts.value(), **kw)
            else:
                frames = select_frames(self._map.n_frames, flipper, spin)
                norm = normalize_map(self._map.counts, frames, mode, **kw)
        except ValueError as error:
            self._show_unavailable(f"{error}. Choose another normalization.")
            return
        frames = norm.frame_indices
        self._norm = norm
        if norm.values.size == 0:
            self._show_unavailable("No frames match the selected spin state.")
            return

        self._ensure_calibration()
        self._sync_cal_controls()
        asym = isinstance(norm, AsymmetryMap)
        if asym:
            # Display only: low-count pixels blanked; colour range symmetric about zero.
            display = np.where(norm.low_count_mask, np.nan, norm.values)
            finite = np.abs(display[np.isfinite(display)])
            half = float(np.percentile(finite, 99)) if finite.size else 1.0
            lo, hi = -(half or 1.0), (half or 1.0)
        else:
            display = log_display(norm.values) if self._log_active() else norm.values
            finite = display[np.isfinite(display)]
            lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            if hi <= lo:
                hi = lo + 1.0
        self._cut_values = display if asym else norm.values
        alpha_note = ""
        if self._alpha_mode():
            alpha_note = self._draw_alpha(display)
            if alpha_note is None:
                return
        else:
            self._clear_alpha_items()
            self.img_item.setImage(display.T, autoLevels=False)
            self.img_item.setRect(0, 0, self._map.nx, len(frames))
            if self._colorbar_on_strips:  # re-link only when coming back from the α view
                self.colorbar.setImageItem(self.img_item)
                self._colorbar_on_strips = False
        self.colorbar.setLevels((lo, hi))
        self.colorbar.setColorMap(self._current_colormap())
        label = f"log₁₀({norm.units})" if self._log_active() else norm.units
        axis = getattr(self.colorbar, "axis", None)
        if axis is not None:
            axis.setLabel(label)

        notes = []
        if asym and mode == "raw":
            notes.append("Raw-count asymmetry assumes equal u/d exposure; Counts / Monitor1 corrects for this.")
        notes.extend(norm.warnings)
        if alpha_note:
            notes.append(alpha_note)
        self._set_notice("  ".join(notes))

        self._update_y_ticks()
        self._draw_roi_overlay()
        self._fix_cut_ranges()
        if self._crosshair_placed:
            # Keep the crosshair on the same map cell (also across axis modes).
            self._place_crosshair(*self._cell_center(self._sel_row, self._sel_x))

    def _apply_no_data_background(self) -> None:
        # Neutral slate behind the map: blank (log-omitted / excluded) pixels stay
        # visibly distinct from any colormap value instead of being recoloured.
        self.plot.getViewBox().setBackgroundColor("#334155")

    def _show_unavailable(self, message: str) -> None:
        self._norm = None
        self.img_item.clear()
        self._clear_alpha_items()
        self._clear_roi_overlay()
        self._clear_cuts()
        self.lbl_coords.setText("")
        self._set_notice(message)

    def _asym_mode(self) -> bool:
        return self.combo_spin.currentData() == "asym"

    def _log_active(self) -> bool:
        """Log display applies to intensities, never to the signed spin asymmetry."""
        return self.chk_log.isChecked() and not self._asym_mode()

    def _current_colormap(self) -> pg.ColorMap:
        if self._asym_mode():
            try:  # diverging map centred on zero asymmetry
                return pg.colormap.getFromMatplotlib("RdBu_r")
            except Exception:
                return get_safe_colormap("cividis")
        return get_safe_colormap(self.combo_cmap.currentText())

    def _on_spin_changed(self) -> None:
        asym = self._asym_mode()
        self.spin_min_counts.setVisible(asym)
        self.chk_log.setEnabled(not asym)
        self.combo_cmap.setEnabled(not asym)
        self._redraw()

    def _on_cmap_changed(self, name: str) -> None:
        self.colorbar.setColorMap(self._current_colormap())

    def _row_label(self, row: int) -> str:
        frame = int(self._norm.frame_indices[row])
        th = self._variable("th")
        if self.combo_y_axis.currentData() == "th" and th is not None and frame < len(th):
            return f"{th[frame]:.3f}"
        return str(frame)

    def _update_y_ticks(self) -> None:
        axis = self.plot.getPlotItem().getAxis("left")
        col_axis = self.plot_col.getPlotItem().getAxis("bottom")
        if self._norm is None or self._norm.frame_indices.size == 0:
            axis.setTicks(None)
            col_axis.setTicks(None)
            return
        if self._alpha_mode():
            axis.setTicks(None)
            col_axis.setTicks(None)
            if self._q_mode():
                self.plot.setLabel("bottom", "Qx (10⁻³ Å⁻¹)")
                self.plot.setLabel("left", "Qz (Å⁻¹)")
                self.plot_row.setLabel("bottom", "Qx (10⁻³ Å⁻¹)")
                self.plot_col.setLabel("bottom", "Qz (Å⁻¹)")
            else:
                self.plot.setLabel("bottom", "α_i (°)")
                self.plot.setLabel("left", "α_f (°)")
                self.plot_row.setLabel("bottom", "α_f (°)")
                self.plot_col.setLabel("bottom", "α_i (°)")
            return
        self.plot.setLabel("bottom", "Detector X pixel")
        self.plot_row.setLabel("bottom", "Detector X pixel")
        n = self._norm.frame_indices.size
        use_th = self.combo_y_axis.currentData() == "th" and self._variable("th") is not None
        self.plot.setLabel("left", "θ (°)" if use_th else "Point (frame index)")
        step = max(1, int(np.ceil(n / 12)))
        major = [(row + 0.5, self._row_label(row)) for row in range(0, n, step)]
        minor = [(row + 0.5, "") for row in range(n) if row % step]
        axis.setTicks([major, minor])
        col_axis.setTicks([major, minor])
        self.plot_col.setLabel("bottom", "θ (°)" if use_th else "Point (frame index)")

    def _clear_roi_overlay(self) -> None:
        for item in self._roi_items:
            self.plot.removeItem(item)
        self._roi_items = []

    def _add_overlay(self, xs, ys, color: str, connect="all", width: int = 2) -> None:
        item = pg.PlotDataItem(np.asarray(xs, float), np.asarray(ys, float), connect=connect,
                               pen=pg.mkPen(color, width=width, style=Qt.PenStyle.DashLine))
        item.setZValue(10)
        self.plot.addItem(item)
        self._roi_items.append(item)

    def _draw_roi_overlay(self) -> None:
        self._clear_roi_overlay()
        if self._norm is None or self.scan_data is None or self._norm.frame_indices.size == 0:
            return
        if self._alpha_mode():
            self._draw_alpha_overlay()
            return
        cal = getattr(self.scan_data, "angle_calibration", None)
        tth = self._variable("tth")
        if self.chk_show_db.isChecked() and cal is not None and cal.fit is not None and tth is not None:
            frames = self._norm.frame_indices
            ok = frames < tth.size
            xs = cal.fit.position_at(tth[frames[ok]])
            ys = np.flatnonzero(ok) + 0.5
            inside = (xs >= 0) & (xs <= self._map.nx)
            self._add_overlay(xs[inside], ys[inside], "#e879f9", connect="finite")
        if not self.chk_show_roi.isChecked():
            return
        bounds = self._scan_roi_bounds()
        frames = self._norm.frame_indices
        if len(bounds) == 1:
            per_row = np.array([bounds[0]] * frames.size, dtype=float)
        else:
            per_row = np.array([bounds[min(int(f), len(bounds) - 1)] for f in frames], dtype=float)
        rows = np.arange(frames.size, dtype=float)
        # Step outline: each row spans [row, row + 1).
        ys = np.repeat(rows, 2) + np.tile([0.0, 1.0], frames.size)
        pen = pg.mkPen("#facc15", width=2, style=Qt.PenStyle.DashLine)
        for col in (0, 1):
            xs = np.repeat(np.minimum(per_row[:, 0], per_row[:, 1]) if col == 0
                           else np.maximum(per_row[:, 0], per_row[:, 1]), 2)
            item = pg.PlotDataItem(xs, ys, pen=pen)
            item.setZValue(10)
            self.plot.addItem(item)
            self._roi_items.append(item)

    # ------------------------------------------------------------- crosshair
    def _map_ready(self) -> bool:
        ready = self._norm is not None and self._map is not None and self._norm.frame_indices.size > 0
        return ready and (not self._alpha_mode() or self._alpha_geom is not None)

    def _on_crosshair_toggled(self, checked: bool) -> None:
        if not checked:
            self._crosshair_placed = False
            for item in (self.crosshair, self.line_row, self.line_x):
                item.setVisible(False)
            self._clear_cuts()
        else:
            self.lbl_cut_hint.setText("Click the map to drop the crosshair.")

    def _on_map_clicked(self, event) -> None:
        if not self.btn_crosshair.isChecked() or not self._map_ready():
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        pt = vb.mapSceneToView(event.scenePos())
        x0, x1, y0, y1 = self._map_extent()
        if x0 <= pt.x() < x1 and y0 <= pt.y() < y1:
            self._place_crosshair(pt.x(), pt.y())
            event.accept()

    def _place_crosshair(self, x: float, y: float) -> None:
        self._crosshair_placed = True
        for item in (self.crosshair, self.line_row, self.line_x):
            item.setVisible(True)
        self.crosshair.setPos((x, y))
        self._on_crosshair_moved(force=True)

    def _on_crosshair_moved(self, *_, force: bool = False) -> None:
        if not self._crosshair_placed or not self._map_ready():
            return
        pos = self.crosshair.pos()
        x0, x1, y0, y1 = self._map_extent()
        x = min(max(pos.x(), x0), x1 - 1e-9 * max(1.0, abs(x1)))
        y = min(max(pos.y(), y0), y1 - 1e-9 * max(1.0, abs(y1)))
        if (x, y) != (pos.x(), pos.y()):  # keep the marker on the map
            self.crosshair.blockSignals(True)
            self.crosshair.setPos((x, y))
            self.crosshair.blockSignals(False)
        self.line_x.setValue(x)
        self.line_row.setValue(y)
        if self._alpha_mode():
            self._cross_alpha = (x, y)
            row, col = self._alpha_cut_cell(x, y)
            self._sel_row, self._sel_x = row, col
            self._update_cuts()
            return
        row, col = int(y), int(x)
        if force or (row, col) != (self._sel_row, self._sel_x):
            self._sel_row, self._sel_x = row, col
            self._update_cuts()

    def _clear_cuts(self) -> None:
        for curve in (self._row_curve, self._col_curve):
            curve.setData([], [])
            curve.setVisible(False)
        self.lbl_row_cut.setText("Horizontal cut (selected point):")
        self.lbl_col_cut.setText("Vertical cut (selected X pixel):")
        self.lbl_cut_hint.setText(CUT_HINT)

    def _fix_cut_ranges(self) -> None:
        """Axis ranges follow the whole map, so dragging the crosshair never rescales the cuts."""
        if not self._map_ready():
            return
        log = self._log_active()
        values = self._cut_values
        shown = values[np.isfinite(values) & (values > 0)] if log else values[np.isfinite(values)]
        lo, hi = (float(shown.min()), float(shown.max())) if shown.size else (0.0, 1.0)
        if log:
            lo, hi = float(np.log10(lo)), float(np.log10(hi))
        pad = 0.05 * (hi - lo) if hi > lo else 0.5
        if self._alpha_mode():
            x0, x1, y0, y1 = self._map_extent()
            # α view: row cut runs along α_f, column cut along α_i. Q view: constant-Qz cut
            # runs along Qx, constant-Qx cut along Qz.
            spans = (((self.plot_row, x0, x1), (self.plot_col, y0, y1)) if self._q_mode()
                     else ((self.plot_row, y0, y1), (self.plot_col, x0, x1)))
        else:
            spans = ((self.plot_row, 0, self._map.nx), (self.plot_col, 0, self._norm.frame_indices.size))
        for plot, start, stop in spans:
            plot.setLogMode(y=log)
            plot.setLabel("left", self._norm.units)
            plot.setXRange(start, stop, padding=0)
            plot.setYRange(lo - pad, hi + pad, padding=0)
            plot.getViewBox().disableAutoRange()

    def _update_cuts(self) -> None:
        if not self._crosshair_placed or not self._map_ready():
            self._clear_cuts()
            return
        row = min(self._sel_row, self._norm.frame_indices.size - 1)
        x = min(self._sel_x, self._map.nx - 1)
        log = self._log_active()

        def positive(values):
            # Log axes cannot show nonpositive values; omit them, never clip.
            return np.where(values > 0, values, np.nan) if log else values

        self._row_curve.setPen(pg.mkPen("#38bdf8" if self.is_dark_mode else "#0284c7", width=2))
        self._col_curve.setPen(pg.mkPen("#f43f5e" if self.is_dark_mode else "#e11d48", width=2))
        if self._alpha_mode():
            self._update_alpha_cuts(positive)
            return
        self._row_curve.setData(np.arange(self._map.nx) + 0.5, positive(self._cut_values[row]))
        col_vals = self._cut_values[:, x]
        self._col_curve.setData(np.arange(col_vals.size) + 0.5, positive(col_vals))
        for curve in (self._row_curve, self._col_curve):
            curve.setVisible(True)
        frame = int(self._norm.frame_indices[row])
        self.lbl_row_cut.setText(f"Horizontal cut — point {frame} ({self._describe_frame(frame)}):")
        self.lbl_col_cut.setText(f"Vertical cut — X pixel {x}:")
        self.lbl_cut_hint.setText("Drag the crosshair centre to move the cuts; click elsewhere to jump.")

    def _angle_decimals(self, name: str) -> int:
        """Decimals matching the scan's resolution in an angle channel (its finest step)."""
        cache = self.__dict__.setdefault("_decimals_cache", {})
        key = (id(self.scan_data), name)
        if key not in cache:
            values = self._variable(name)
            steps = np.diff(np.unique(np.round(values[np.isfinite(values)], 6))) if values is not None else []
            steps = np.asarray(steps)[np.asarray(steps) > 1e-5]
            cache[key] = _decimals_for_step(float(steps.min())) if steps.size else 4
        return cache[key]

    def _describe_frame(self, frame: int) -> str:
        parts = []
        for name, label in (("th", "θ"), ("tth", "2θ")):
            values = self._variable(name)
            if values is not None and frame < len(values):
                parts.append(f"{label} = {values[frame]:.{self._angle_decimals(name)}f}°")
        flipper = self._variable("Flipper1")
        if flipper is not None and frame < len(flipper):
            parts.append({0.0: "d", 1.0: "u"}.get(float(flipper[frame]), "?"))
        return ", ".join(parts)

    def _format_value(self, row: int, x: int) -> str:
        """Map value at a cell, without uncertainty (shown as measured or normalized)."""
        value = self._norm.values[row, x]
        if not np.isfinite(value):
            return "—"
        if isinstance(self._norm, AsymmetryMap):
            down = int(self._norm.partner_indices[row])
            raw = self._map.counts
            frame = int(self._norm.frame_indices[row])
            return f"SA {value:+.2f} (u {raw[frame, x]:,.0f} / d {raw[down, x]:,.0f})"
        if self._norm.mode == "raw":
            return f"{value:,.0f} counts"
        return f"{value:.4g} {self._norm.units}"

    def _on_mouse_moved(self, pos) -> None:
        if self._norm is None or self._map is None or self._norm.frame_indices.size == 0:
            return
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            self.lbl_coords.setText("")
            return
        pt = vb.mapSceneToView(pos)
        cell = self._cell_at(pt.x(), pt.y())
        if cell is None:
            self.lbl_coords.setText("")
            return
        row, x = cell
        frame = int(self._norm.frame_indices[row])
        th, tth = self._variable("th"), self._variable("tth")
        parts = []
        if th is not None and frame < th.size:
            parts.append(f"θ {th[frame]:.{self._angle_decimals('th')}f}°")
        if tth is not None and frame < tth.size:
            parts.append(f"2θ {tth[frame]:.{self._angle_decimals('tth')}f}°")
        if self._alpha_mode() and self._alpha_geom is not None:
            cal = self._alpha_geom["cal"]
            af = (tth[frame] - th[frame]) + alpha_f_offsets(self._map.nx, cal)[x]
            parts.append(f"α_f {af:.{_decimals_for_step(cal.deg_per_pixel)}f}°")
            if self._q_mode():
                lam = self.spin_wavelength.value()
                qx, qz = q_from_angles(th[frame], af, lam)
                dq = 2 * np.pi / lam * np.radians(cal.deg_per_pixel)  # Qz step per pixel
                parts.append(f"Qx {qx * QX_SCALE:.3g}×10⁻³")
                parts.append(f"Qz {qz:.{_decimals_for_step(dq)}f} Å⁻¹")
        parts.append(f"X {x}")
        self.lbl_coords.setText("   ".join(parts) + "   |   " + self._format_value(row, x))

    # ------------------------------------------------------------- export
    def _set_notice(self, text: str) -> None:
        self.lbl_notice.setText(text)
        self.lbl_notice.setVisible(bool(text))

    def _source_paths(self) -> list[str]:
        own = [self.scan_data.file_path] if self.scan_data is not None and self.scan_data.file_path else []
        return own + list(self.session_source_paths())

    def _default_export_name(self, kind: str, suffix: str) -> str:
        stem = Path(self.scan_data.file_path).stem if self.scan_data is not None and self.scan_data.file_path else "scan"
        view = {"point": "", "th": "_theta", "alpha": "_alpha", "q": "_q"}.get(self.combo_y_axis.currentData(), "")
        return f"{stem}_{kind}{view}{suffix}"

    def has_export_data(self) -> bool:
        return self._norm is not None and self._map is not None and self.scan_data is not None

    def export_image(self) -> Path | None:
        """Save the map as shown (axes, colour bar, overlays) as PNG or SVG."""
        if not self.has_export_data():
            QMessageBox.information(self, "No Data", "No off-specular map to export.")
            return None
        return export_plot_image(self, self.plot.getPlotItem(), self._default_export_name("offspecular", ".png"),
                                 self._source_paths(), "Export Off-Specular Map Image")

    def export_table(self):
        """(header, columns, rows) of the map in the current axes, spin and normalization.

        Values are the normalized map values with 1σ uncertainty; the log colour
        scale and the asymmetry low-count blanking are display-only and not applied.
        """
        norm, omap = self._norm, self._map
        axes = self.combo_y_axis.currentData()
        frames = norm.frame_indices
        rows = np.arange(frames.size)
        coords: dict[str, np.ndarray] = {}
        th = self._variable("th")
        if axes == "th" and th is not None:
            theta = np.full(frames.size, np.nan)
            ok = frames < th.size
            theta[ok] = th[frames[ok]]
            coords["Theta (deg)"] = theta
        if self._alpha_mode():
            geom = self._alpha_geom
            if geom is None:
                raise ValueError(self.lbl_notice.text() or "The angle view is not available for these settings")
            rows = geom["rows"]
            alpha_f = geom["base"][:, None] + alpha_f_offsets(omap.nx, geom["cal"])[None, :]
            alpha_i = np.broadcast_to(geom["alpha_i"][:, None], alpha_f.shape)
            if self._q_mode():
                qx, qz = q_from_angles(alpha_i, alpha_f, self.spin_wavelength.value())
                coords["Qx (1/Angstrom)"], coords["Qz (1/Angstrom)"] = qx, qz
            else:
                coords["AlphaI (deg)"], coords["AlphaF (deg)"] = geom["alpha_i"], alpha_f
        asym = isinstance(norm, AsymmetryMap)
        columns, data = map_table(frames[rows], norm.values[rows], norm.standard_uncertainty[rows], coords,
                                  partner_indices=norm.partner_indices[rows] if asym else None)
        y0, y1 = omap.y_bounds
        header = [
            f"Off-specular map: {self.combo_y_axis.currentText()}; detector frames integrated along Y, "
            "displayed X pixel (90 deg CW + mirrored, as in the 2D Detector tab).",
            f"Source: {self.scan_data.file_path}",
            f"Spin: {self.combo_spin.currentText()}"
            + (" (Frame = u frame, PartnerFrame = paired d frame)" if asym else ""),
            f"Integrated Y rows: {'per-frame tracked ROI within ' if omap.y_varies else ''}[{y0}, {y1})",
            f"Normalization: {self.combo_norm.currentText()}; Value units: {norm.units}",
            "StdUncertainty is first-order 1 sigma (docs/normalization.md, docs/off_specular.md).",
            "Frame is the 0-based detector frame (scan point) index.",
        ]
        if self._alpha_mode():
            cal = self._alpha_geom["cal"]
            header.append(f"Angle calibration: {cal.deg_per_pixel!r} deg/pixel ({cal.deg_per_pixel_source}), "
                          f"x0 = {cal.x0!r} px, alpha_f {'increases' if cal.sign > 0 else 'decreases'} with X; "
                          "coordinates are pixel centres")
            if self._q_mode():
                header.append(f"Wavelength: {self.spin_wavelength.value()!r} Angstrom; "
                              "Qx = (2pi/lambda)(cos af - cos ai), Qz = (2pi/lambda)(sin af + sin ai)")
        if asym:
            header.append(f"Display blanks pixels with u+d < {norm.min_counts:g} raw counts; "
                          "exported values are not blanked.")
        if self._log_active():
            header.append("Log colour scale is display-only; values are linear.")
        if norm.warnings:
            header.append("Warnings: " + " ".join(norm.warnings))
        return header, columns, data

    def export_data(self) -> Path | None:
        """Save the map in the current display format as tab-delimited text."""
        if not self.has_export_data():
            QMessageBox.information(self, "No Data", "No off-specular map to export.")
            return None
        try:
            target = ask_output_path(self, "Export Off-Specular Map Data (tab-delimited)",
                                     self._default_export_name("offspecular", ".txt"), TABLE_FILTERS,
                                     self._source_paths())
            if target is None:
                return None
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                count = write_table(target, *self.export_table(), self._source_paths())
            finally:
                QApplication.restoreOverrideCursor()
        except Exception as error:  # noqa: BLE001 - shown to the user
            QMessageBox.critical(self, "Export Failed", f"Failed to save map data:\n{error}")
            return None
        QMessageBox.information(self, "Export Succeeded", f"Exported {count:,} map cells to {target.name}.")
        return target

    # ------------------------------------------------------------- α_i / α_f
    def _alpha_mode(self) -> bool:
        """Angle-space views (α_i–α_f and Qx–Qz) share the calibrated strip geometry."""
        return self.combo_y_axis.currentData() in ("alpha", "q")

    def _q_mode(self) -> bool:
        return self.combo_y_axis.currentData() == "q"

    def _to_view(self, alpha_i, alpha_f):
        """(α_i, α_f) in degrees → plot coordinates of the current angle-space view."""
        if self._q_mode():
            qx, qz = q_from_angles(alpha_i, alpha_f, self.spin_wavelength.value())
            return qx * QX_SCALE, qz
        return np.asarray(alpha_i, dtype=float), np.asarray(alpha_f, dtype=float)

    def _from_view(self, x, y):
        """Plot coordinates of the current angle-space view → (α_i, α_f) in degrees."""
        if self._q_mode():
            return angles_from_q(np.asarray(x, dtype=float) / QX_SCALE, y, self.spin_wavelength.value())
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def _on_wavelength_changed(self) -> None:
        if self._q_mode():
            self._redraw()
            if self._map_ready():
                self.plot.autoRange()

    def _update_q_cuts(self, positive, qx_view: float, qz: float) -> None:
        """Constant-Qz cut versus Qx and constant-Qx cut versus Qz.

        Each measured point contributes the one pixel whose centre lies on the
        cut line (no interpolation); points whose detector does not reach the
        line are left out.
        """
        g = self._alpha_geom
        cal = g["cal"]
        lam = self.spin_wavelength.value()
        k0 = 2 * np.pi / lam
        qx = qx_view / QX_SCALE
        ai = np.radians(g["alpha_i"])
        _, af_cross = self._from_view(qx_view, qz)
        side = 1.0 if af_cross >= 0 else -1.0

        def sample(af_deg):
            xs_q, vals = [], []
            for j, af in enumerate(af_deg):
                if not np.isfinite(af):
                    continue
                px = self._pixel_for_alpha(j, af)
                if px is None:
                    continue
                af_c = g["base"][j] + alpha_f_offsets(self._map.nx, cal)[px]
                xs_q.append(q_from_angles(g["alpha_i"][j], af_c, lam))
                vals.append(self._cut_values[g["rows"][j], px])
            return np.array(xs_q, dtype=float).reshape(-1, 2), np.array(vals, dtype=float)

        with np.errstate(invalid="ignore"):
            af_const_qz = np.degrees(np.arcsin(qz / k0 - np.sin(ai)))
            af_const_qx = side * np.degrees(np.arccos(qx / k0 + np.cos(ai)))
        q_rows, v_rows = sample(af_const_qz)
        order = np.argsort(q_rows[:, 0])
        self._row_curve.setData(q_rows[order, 0] * QX_SCALE, positive(v_rows[order]))
        q_cols, v_cols = sample(af_const_qx)
        order = np.argsort(q_cols[:, 1])
        self._col_curve.setData(q_cols[order, 1], positive(v_cols[order]))
        for curve in (self._row_curve, self._col_curve):
            curve.setVisible(True)
        self.lbl_row_cut.setText(f"Cut at constant Qz = {qz:.4f} Å⁻¹ ({v_rows.size} points):")
        self.lbl_col_cut.setText(f"Cut at constant Qx = {qx_view:.3g}×10⁻³ Å⁻¹ ({v_cols.size} points):")
        self.lbl_cut_hint.setText("Drag the crosshair centre to move the cuts; click elsewhere to jump.")

    def _on_axes_changed(self) -> None:
        self._redraw()
        if self._map_ready():
            self.plot.autoRange()

    def _ensure_calibration(self) -> AngleCalibration | None:
        """Fit this scan's calibration once, from its own data (never inherited from another scan)."""
        scan = self.scan_data
        if scan is None or self._map is None:
            return None
        if scan.angle_calibration is not None:
            return scan.angle_calibration
        if self._cal_message:  # already failed for this scan; wait for manual input or refit
            return None
        tth = self._variable("tth")
        if tth is None:
            self._cal_message = "No 2θ channel: angle calibration unavailable."
            return None
        try:
            scan.angle_calibration = calibrate(self._map.counts, tth, self._roi_x_ranges(),
                                               y_rows=self._map.y_bounds)
        except CalibrationError as error:
            self._cal_message = str(error)
        return scan.angle_calibration

    def _roi_x_ranges(self) -> np.ndarray:
        bounds = self._scan_roi_bounds()
        n = self._map.n_frames
        if len(bounds) == 1:
            return np.tile([bounds[0][0], bounds[0][1]], (n, 1)).astype(float)
        return np.array([[b[0], b[1]] for b in bounds[:n]], dtype=float)

    def _refit_calibration(self) -> None:
        if self.scan_data is None:
            return
        self.scan_data.angle_calibration = None
        self._cal_message = ""
        self._ensure_calibration()
        self._sync_cal_controls()
        self._redraw()

    def _sync_cal_controls(self) -> None:
        cal = getattr(self.scan_data, "angle_calibration", None) if self.scan_data is not None else None
        widgets = (self.spin_dpp, self.spin_x0, self.chk_af_increases)
        for widget in widgets:
            widget.blockSignals(True)
        if self.scan_data is not None and self.scan_data.detector_shape:
            self.spin_x0.setMaximum(float(self.scan_data.detector_shape[1]))
        if cal is not None:
            self.spin_dpp.setValue(cal.deg_per_pixel)
            self.spin_x0.setValue(cal.x0)
            self.chk_af_increases.setChecked(cal.sign > 0)
            self._set_cal_status(cal.describe())
        else:
            self.spin_dpp.setValue(0.0)
            if self._map is not None:
                try:
                    self.spin_x0.setValue(specular_center(self._map.counts, self._roi_x_ranges()))
                except CalibrationError:
                    self.spin_x0.setValue(float(np.mean(self._roi_x_ranges()[0])))
            self.chk_af_increases.setChecked(False)
            self._set_cal_status(self._cal_message or
                                 "Calibrated automatically from the direct beam when the map is computed.")
        for widget in widgets:
            widget.blockSignals(False)

    def _on_calibration_edited(self, *, deg_per_pixel=None, x0=None, sign=None) -> None:
        """Manual override for this scan only; only the edited quantity changes."""
        scan = self.scan_data
        if scan is None:
            return
        cal = scan.angle_calibration
        try:
            if cal is None:
                dpp = self.spin_dpp.value()
                if dpp <= 0:
                    return
                cal = AngleCalibration(dpp, self.spin_x0.value(), 1 if self.chk_af_increases.isChecked() else -1,
                                       None, "manual", "manual")
            elif deg_per_pixel is not None and deg_per_pixel <= 0:
                self._sync_cal_controls()
                return
            else:
                cal = cal.with_manual(deg_per_pixel=deg_per_pixel, x0=x0, sign=sign)
        except ValueError as error:
            self._set_notice(f"Calibration not changed: {error}")
            return
        scan.angle_calibration = cal
        self._set_cal_status(cal.describe())
        self._redraw()

    def _clear_alpha_items(self) -> None:
        for item in self._alpha_items:
            self.plot.removeItem(item)
        self._alpha_items = []
        self._alpha_geom = None

    def _draw_alpha(self, display: np.ndarray) -> str | None:
        """Draw each frame as its own α_i strip of exact pixel cells. Returns a notice,
        or None (after showing why) when the α view cannot be drawn."""
        cal = getattr(self.scan_data, "angle_calibration", None)
        name = "Qx vs Qz" if self._q_mode() else "α_i vs α_f"
        if cal is None:
            self._show_unavailable(f"{name} needs an angle calibration: "
                                   + (self._cal_message or "enter °/pixel in the calibration panel."))
            return None
        flipper = self._variable("Flipper1")
        if (self.combo_spin.currentData() == "all" and flipper is not None
                and np.any(flipper == 0) and np.any(flipper == 1)):
            self._show_unavailable(f"{name}: choose one spin state (u or d) or the spin asymmetry; "
                                   "both states share the same angles and would overlap.")
            return None
        th, tth = self._variable("th"), self._variable("tth")
        if th is None or tth is None:
            self._show_unavailable(f"{name} needs θ (th) and 2θ (tth) channels.")
            return None
        frames = self._norm.frame_indices
        ok = (frames < th.size) & (frames < tth.size)
        ok[ok] &= np.isfinite(th[frames[ok]]) & np.isfinite(tth[frames[ok]])
        written = self._map.written
        if written is not None:
            ok &= written[frames]
        rows = np.flatnonzero(ok)
        if rows.size == 0:
            self._show_unavailable(f"{name}: no measured frames with valid angles.")
            return None
        self.img_item.clear()
        self._clear_alpha_items()
        sel = frames[rows]
        alpha_i = th[sel]
        base = tth[sel] - th[sel]
        strips = alpha_i_strip_edges(alpha_i)
        edges = alpha_f_offsets(self._map.nx, cal, edges=True)
        span = abs(edges[-1] - edges[0])
        geom = dict(rows=rows, alpha_i=alpha_i, base=base, strips=strips, cal=cal,
                    af_lo=float(base.min() + edges.min()), af_hi=float(base.max() + edges.max()))
        items = []
        if self._q_mode():
            # Exact cells: every pixel's four (α_i, α_f) corners mapped to (Qx, Qz). One mesh;
            # the NaN rows between strips keep points separate (no interpolation).
            n = rows.size
            ai_corners = np.broadcast_to(strips.reshape(-1, 1), (2 * n, edges.size))  # strip lo/hi edges
            af_corners = np.repeat(base, 2)[:, None] + edges[None, :]  # pixel edges at each strip's 2θ − θ
            X, Y = self._to_view(ai_corners, af_corners)
            Z = np.full((2 * n - 1, self._map.nx), np.nan)
            Z[0::2] = display[rows]
            mesh = CachedMeshItem(X, Y, Z, enableAutoLevels=False)
            self.plot.addItem(mesh)
            items.append(mesh)
            geom["extent"] = (float(np.nanmin(X)), float(np.nanmax(X)), float(np.nanmin(Y)), float(np.nanmax(Y)))
        else:
            for k, r in enumerate(rows):
                values = display[r] if cal.sign > 0 else display[r][::-1]  # image runs to increasing α_f
                item = pg.ImageItem(values[None, :], autoLevels=False)
                item.setRect(strips[k, 0], base[k] + edges.min(), strips[k, 1] - strips[k, 0], span)
                self.plot.addItem(item)
                items.append(item)
        self._alpha_items = items
        self._alpha_geom = geom
        self.colorbar.setImageItem(items)
        self._colorbar_on_strips = True
        skipped = frames.size - rows.size
        return f"{name}: {skipped} unwritten/invalid frame(s) omitted." if skipped else ""

    def _map_extent(self) -> tuple[float, float, float, float]:
        if self._alpha_mode() and self._alpha_geom is not None:
            g = self._alpha_geom
            if "extent" in g:
                return g["extent"]
            return float(g["strips"][:, 0].min()), float(g["strips"][:, 1].max()), g["af_lo"], g["af_hi"]
        n_rows = self._norm.frame_indices.size if self._norm is not None else 1
        nx = self._map.nx if self._map is not None else 1
        return 0.0, float(nx), 0.0, float(n_rows)

    def _pixel_for_alpha(self, k: int, alpha_f: float) -> int | None:
        g = self._alpha_geom
        cal = g["cal"]
        px = int(np.floor(cal.x0 + cal.sign * (alpha_f - g["base"][k]) / cal.deg_per_pixel))
        return px if 0 <= px < self._map.nx else None

    def _cell_at(self, xv: float, yv: float):
        """(map row, X pixel) under a view position, or None."""
        if not self._map_ready():
            return None
        if self._alpha_mode():
            g = self._alpha_geom
            xv, yv = (float(v) for v in self._from_view(xv, yv))  # → (α_i, α_f)
            inside = np.flatnonzero((g["strips"][:, 0] <= xv) & (xv < g["strips"][:, 1]))
            if inside.size == 0:
                return None
            k = int(inside[-1])  # strips at repeated θ overlap; the last drawn is visible
            px = self._pixel_for_alpha(k, yv)
            return None if px is None else (int(g["rows"][k]), px)
        row, x = int(np.floor(yv)), int(np.floor(xv))
        if 0 <= x < self._map.nx and 0 <= row < self._norm.frame_indices.size:
            return row, x
        return None

    def _nearest_strip(self, xv: float) -> int:
        """Strip nearest in α_i; among repeated θ the last drawn (visible) one."""
        dist = np.abs(self._alpha_geom["alpha_i"] - xv)
        return int(dist.size - 1 - np.argmin(dist[::-1]))

    def _alpha_cut_cell(self, xv: float, yv: float) -> tuple[int, int]:
        g = self._alpha_geom
        xv, yv = (float(v) for v in self._from_view(xv, yv))  # → (α_i, α_f)
        k = self._nearest_strip(xv)
        px = self._pixel_for_alpha(k, yv)
        if px is None:
            cal = g["cal"]
            px = int(np.clip(np.floor(cal.x0 + cal.sign * (yv - g["base"][k]) / cal.deg_per_pixel),
                             0, self._map.nx - 1))
        return int(g["rows"][k]), px

    def _cell_center(self, row: int, x: int) -> tuple[float, float]:
        """View coordinates of a map cell's centre in the current axis mode."""
        n_rows = self._norm.frame_indices.size if self._norm is not None else 1
        row = int(min(max(row, 0), n_rows - 1))
        x = int(min(max(x, 0), (self._map.nx if self._map is not None else 1) - 1))
        if self._alpha_mode() and self._alpha_geom is not None:
            g = self._alpha_geom
            k = int(np.argmin(np.abs(g["rows"] - row)))
            ai, af = g["alpha_i"][k], g["base"][k] + alpha_f_offsets(self._map.nx, g["cal"])[x]
            vx, vy = self._to_view(ai, af)
            return float(vx), float(vy)
        return x + 0.5, row + 0.5

    def _update_alpha_cuts(self, positive) -> None:
        """Row cut: the nearest measured point versus α_f. Column cut: constant α_f across points."""
        g = self._alpha_geom
        view = getattr(self, "_cross_alpha", self._cell_center(self._sel_row, self._sel_x))
        if self._q_mode():
            self._update_q_cuts(positive, *view)
            return
        xv, yv = (float(v) for v in self._from_view(*view))
        k = self._nearest_strip(xv)
        row = int(g["rows"][k])
        cal = g["cal"]
        self._row_curve.setData(g["base"][k] + alpha_f_offsets(self._map.nx, cal),
                                positive(self._cut_values[row]))
        col = np.full(g["rows"].size, np.nan)
        for j in range(g["rows"].size):
            px = self._pixel_for_alpha(j, yv)
            if px is not None:
                col[j] = self._cut_values[g["rows"][j], px]
        order = np.argsort(g["alpha_i"], kind="stable")
        self._col_curve.setData(g["alpha_i"][order], positive(col[order]))
        for curve in (self._row_curve, self._col_curve):
            curve.setVisible(True)
        frame = int(self._norm.frame_indices[row])
        self.lbl_row_cut.setText(f"Cut at α_i = {g['alpha_i'][k]:.4f}° — point {frame} ({self._describe_frame(frame)}):")
        self.lbl_col_cut.setText(f"Cut at constant α_f = {yv:.4f}°:")
        self.lbl_cut_hint.setText("Drag the crosshair centre to move the cuts; click elsewhere to jump.")

    def _draw_alpha_overlay(self) -> None:
        g = self._alpha_geom
        if g is None:
            return
        cal = g["cal"]
        a0, a1 = float(g["strips"][:, 0].min()), float(g["strips"][:, 1].max())
        if not self.chk_show_db.isChecked():
            return  # unticked: the map is shown without any overlay, ROI included
        ai = np.linspace(a0, a1, 200)
        self._add_overlay(*self._to_view(ai, ai), "#ffffff", width=1)  # specular: α_f = α_i (Qx = 0)
        if not self._q_mode():  # in Q space the direct beam (α_f = −α_i) collapses to the origin
            self._add_overlay(*self._to_view(ai, -ai), "#e879f9", width=1)  # direct beam: α_f = −α_i
        if not self.chk_show_roi.isChecked():
            return
        bounds = self._scan_roi_bounds()
        frames = self._norm.frame_indices[g["rows"]]
        for col in (0, 1):
            xs, ys = [], []
            for k, f in enumerate(frames):
                b = bounds[0] if len(bounds) == 1 else bounds[min(int(f), len(bounds) - 1)]
                xb = min(b[0], b[1]) if col == 0 else max(b[0], b[1])
                af = g["base"][k] + cal.sign * (xb - cal.x0) * cal.deg_per_pixel
                vx, vy = self._to_view([g["strips"][k, 0], g["strips"][k, 1]], [af, af])
                xs += list(vx)
                ys += list(vy)
            self._add_overlay(xs, ys, "#facc15", connect="pairs")
