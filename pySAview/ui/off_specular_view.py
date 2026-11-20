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
from PyQt6.QtGui import QColor, QCursor, QImage, QPainter, QPainterPath, QPicture, QTransform, qRgba
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pySAview.core.angle_calibration import (
    AngleCalibration,
    DETECTOR_WIDTH_MM,
    PYSARED_DEFAULT_DISTANCE_MM,
    CalibrationError,
    alpha_f_offset_for_pixel,
    alpha_f_offsets,
    alpha_i_strip_edges,
    angles_from_q,
    calibrate,
    pixel_for_alpha_f_offset,
    pixel_pitch_mm,
    q_from_angles,
    specular_center,
)
from pySAview.core.reduction.background import resolve_bkg_roi
from pySAview.core.reduction.curvature import apply_curvature_by_spin
from pySAview.core.reduction.kinematics import (
    constant_qx_alpha_f,
    constant_qz_alpha_f,
    qz_step_per_pixel,
)
from pySAview.core.reduction.roi_extraction import pixel_bounds
from pySAview.core.models import ScanData
from pySAview.core.nexus_reader import NexusReader
from pySAview.core.off_specular import (
    DETECTOR_PATH,
    AsymmetryMap,
    SummedSpinMap,
    pair_spin_frames,
    spin_asymmetry,
    sum_spin_frames,
    OffSpecularMap,
    compute_off_specular_map,
    frame_y_bounds,
    log_display,
    normalize_map,
    select_frames,
)
from pySAview.core.export_tables import map_table, write_table
from pySAview.ui import icons
from pySAview.ui.flow_layout import (
    FlowContainer,
    FlowLayout,
    configure_compact_combo,
    fit_combo_popup,
    make_flow_group,
)
from pySAview.ui.plot_tools import (
    ScientificAxisItem, TABLE_FILTERS, ask_output_path, export_plot_image, install_context_export, tool_button,
)
from pySAview.ui.theme import apply_plot_theme, get_safe_colormap
from pySAview.ui.wheel_guard import install_wheel_focus_guard

DEFAULT_ROI = (354.0, 382.0, 285.0, 743.0)


QX_SCALE = 1e3  # Qx is shown in 10⁻³ Å⁻¹ (it is ~100× smaller than Qz)


def _decimals_for_step(step: float) -> int:
    """Fewest decimals that still resolve ``step`` (e.g. 0.0097 -> 2, 0.04 -> 2, 0.0082 -> 3)."""
    if not np.isfinite(step) or step <= 0:
        return 4
    return int(min(6, max(0, np.ceil(-np.log10(step) - 0.05))))


_tool_button = tool_button


CUT_HINT = "Click Drop Cut Lines, Horizontal, or Vertical above (or ⌖ in the toolbar) to drop draggable cuts."


def _fixed_width_text(label: QLabel) -> None:
    """Changing text must never resize the layout (the map would jump while hovering)."""
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setMinimumWidth(0)


def _bilateral_1d_display(
    arr: np.ndarray,
    sigma_s: float,
    sigma_r: float,
    axis: int,
    *,
    fill_isolated_nans: bool = False,
) -> np.ndarray:
    """1D NaN-safe edge-preserving bilateral filter along ``axis`` (pure NumPy, display-only)."""
    r = max(1, min(8, int(np.ceil(2.5 * float(sigma_s)))))
    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0)
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    f_pad = np.pad(filled, pad, mode="constant", constant_values=0.0)
    m_pad = np.pad(finite.astype(float), pad, mode="constant", constant_values=0.0)
    num = np.zeros_like(filled)
    den = np.zeros_like(filled)
    w_tot = 0.0
    inv_2s2 = 0.5 / (float(sigma_s) ** 2)
    inv_2r2 = 0.5 / (max(float(sigma_r), 1e-9) ** 2)
    n_ax = arr.shape[axis]
    for d in range(-r, r + 1):
        ws = float(np.exp(-(d * d) * inv_2s2))
        w_tot += ws
        if axis == 0:
            nb = f_pad[r + d : r + d + n_ax, :]
            nm = m_pad[r + d : r + d + n_ax, :]
        else:
            nb = f_pad[:, r + d : r + d + n_ax]
            nm = m_pad[:, r + d : r + d + n_ax]
        diff = nb - filled
        wr = np.where(finite, np.exp(-(diff * diff) * inv_2r2), 1.0)
        w = ws * wr * nm
        num += w * nb
        den += w
    out = np.full_like(filled, np.nan)
    if fill_isolated_nans:
        ok = (finite & (den > 1e-12)) | (~finite & (den >= 0.35 * w_tot))
    else:
        ok = finite & (den > 1e-12)
    out[ok] = num[ok] / den[ok]
    return out


def _bilateral_2d_display(
    values: np.ndarray,
    sigma_s: float,
    sigma_r: float,
    *,
    fill_isolated_nans: bool = False,
) -> np.ndarray:
    """Separable edge-preserving 2D bilateral filter in display space (never blooms bright ridges)."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.size == 0 or sigma_s <= 0.0:
        return arr.copy()
    tmp = _bilateral_1d_display(arr, sigma_s, sigma_r, axis=1, fill_isolated_nans=fill_isolated_nans)
    return _bilateral_1d_display(tmp, sigma_s, sigma_r, axis=0, fill_isolated_nans=fill_isolated_nans)


def _sample_bilinear_ai_af(
    src: np.ndarray,
    ai_sorted: np.ndarray,
    base_sorted: np.ndarray,
    strips_sorted: np.ndarray,
    nx: int,
    cal: AngleCalibration,
    ai_query: np.ndarray,
    af_query: np.ndarray,
    *,
    min_weight: float = 0.45,
) -> np.ndarray:
    """Bilinearly sample ``src[frame, pixel]`` at physical ``(alpha_i, alpha_f)`` coordinates."""
    n = ai_sorted.size
    if n == 0 or nx <= 0:
        return np.full(ai_query.shape, np.nan, dtype=float)
    valid_in = (
        np.isfinite(ai_query)
        & np.isfinite(af_query)
        & (ai_query >= float(strips_sorted[0, 0]))
        & (ai_query <= float(strips_sorted[-1, 1]))
    )
    k1 = np.searchsorted(ai_sorted, ai_query)
    k0 = np.clip(k1 - 1, 0, n - 1)
    k1c = np.clip(k1, 0, n - 1)
    in_strip = (
        ((ai_query >= strips_sorted[k0, 0]) & (ai_query <= strips_sorted[k0, 1]))
        | ((ai_query >= strips_sorted[k1c, 0]) & (ai_query <= strips_sorted[k1c, 1]))
    )
    valid_in &= in_strip

    denom = ai_sorted[k1c] - ai_sorted[k0]
    safe_denom = np.where(denom > 1e-12, denom, 1.0)
    t_ai = np.where(denom > 1e-12, np.clip((ai_query - ai_sorted[k0]) / safe_denom, 0.0, 1.0), 0.0)

    def _sample_row(k_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xf = pixel_for_alpha_f_offset(af_query - base_sorted[k_idx], nx, cal) - 0.5
        in_x = np.isfinite(xf) & (xf >= -0.5) & (xf <= nx - 0.5)
        xf_c = np.clip(np.where(in_x, xf, 0.0), 0.0, float(nx - 1))
        x0 = np.floor(xf_c).astype(np.int32)
        x1 = np.minimum(x0 + 1, nx - 1)
        tx = xf_c - x0
        v0 = src[k_idx, x0]
        v1 = src[k_idx, x1]
        m0 = np.isfinite(v0) & in_x
        m1 = np.isfinite(v1) & in_x
        w0 = np.where(m0, 1.0 - tx, 0.0)
        w1 = np.where(m1, tx, 0.0)
        val = np.where(m0, v0, 0.0) * w0 + np.where(m1, v1, 0.0) * w1
        return val, w0 + w1

    s0, w0 = _sample_row(k0)
    s1, w1 = _sample_row(k1c)
    num = (1.0 - t_ai) * s0 + t_ai * s1
    den = (1.0 - t_ai) * w0 + t_ai * w1
    out = np.full(ai_query.shape, np.nan, dtype=float)
    ok = valid_in & (den >= min_weight)
    out[ok] = num[ok] / den[ok]
    return out


def _quantize_indexed_image(
    vals: np.ndarray, levels: tuple[float, float] | None, color_table: list[int],
) -> tuple[np.ndarray, QImage]:
    """Quantize a 2D float grid ``(H, W)`` into a ``Format_Indexed8`` QImage (0 = transparent NaN)."""
    h, w = vals.shape
    bins = np.zeros((h, w), dtype=np.uint8)
    valid = np.isfinite(vals)
    if np.any(valid):
        if levels is None:
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        else:
            lo, hi = float(levels[0]), float(levels[1])
        rng = hi - lo if hi > lo else 1.0
        scaled = np.clip(np.rint((vals[valid] - lo) * (254.0 / rng)), 0.0, 254.0).astype(np.uint8)
        bins[valid] = scaled + 1
    bins = np.ascontiguousarray(bins)
    img = QImage(bins.data, w, h, w, QImage.Format.Format_Indexed8)
    img.setColorTable(color_table)
    return bins, img


_RDBU_R_POS = np.linspace(0.0, 1.0, 11)
_RDBU_R_RGBA = np.array(
    [
        [5, 48, 97, 255],
        [33, 102, 172, 255],
        [67, 147, 195, 255],
        [146, 197, 222, 255],
        [209, 229, 240, 255],
        [247, 247, 247, 255],
        [253, 219, 199, 255],
        [244, 165, 130, 255],
        [214, 96, 77, 255],
        [178, 24, 43, 255],
        [103, 0, 31, 255],
    ],
    dtype=np.ubyte,
)
_CMAP_CACHE: dict[str, pg.ColorMap] = {}
_IDENTITY_GRAY_ALPHA_TABLE = [qRgba(0, 0, 0, 0)] + [qRgba(i, i, i, 255) for i in range(1, 256)]


def _diverging_rdbu_r() -> pg.ColorMap:
    """ColorBrewer RdBu_r diverging map without importing matplotlib."""
    cmap = _CMAP_CACHE.get("__rdbu_r__")
    if cmap is None:
        cmap = pg.ColorMap(pos=_RDBU_R_POS, color=_RDBU_R_RGBA)
        _CMAP_CACHE["__rdbu_r__"] = cmap
    return cmap


def _color_table_from_colormap(cmap: pg.ColorMap) -> list[int]:
    """256-entry ARGB color table for Format_Indexed8: index 0 transparent, 1..255 LUT."""
    lut = cmap.getLookupTable(nPts=255, alpha=True)
    return [qRgba(0, 0, 0, 0)] + [qRgba(int(r), int(g), int(b), int(a)) for r, g, b, a in lut]


def _color_table_from_qcolors(lut) -> list[int]:
    n = len(lut)
    if n == 0:
        return [qRgba(0, 0, 0, 0)] + [qRgba(0, 0, 0, 255)] * 255
    idx = np.linspace(0, n - 1, 255).round().astype(int)
    return [qRgba(0, 0, 0, 0)] + [QColor(lut[int(i)]).rgba() for i in idx]


def _same_transform(a: QTransform, b: QTransform, tol: float = 1e-6) -> bool:
    va = (a.m11(), a.m12(), a.m21(), a.m22(), a.dx(), a.dy())
    vb = (b.m11(), b.m12(), b.m21(), b.m22(), b.dx(), b.dy())
    return all(abs(x - y) <= tol * max(1.0, abs(x), abs(y)) for x, y in zip(va, vb))


class CachedMeshItem(pg.PColorMeshItem):
    """Exact-cell mesh that blits cached indexed-color images while the view moves.

    ``PColorMeshItem`` replays one polygon per cell on every repaint (~150k for a
    Qx–Qz or θ map, ~1–2 s), which makes panning, zooming, and adjusting plot
    settings sluggish. Here the exact cells are rasterized into ``Format_Indexed8``
    images (a whole-map overview and a screen-resolution image of the current view
    plus a margin):
    - Rectilinear uniform-X meshes (such as ``X pixel vs θ``) blit per-row strips
      directly without allocating 300k polygons (~100× faster).
    - Curved meshes (such as ``Qx vs Qz``) bin finite quads into chunked
      ``QPainterPath``s without ``QuadInstances`` Python overhead.
    - Colormap switches swap the 256-entry color table in-place (<1 ms, zero
      re-rasterization).
    Display only: cell geometry and values are unchanged, and exports always draw
    the exact cells directly.
    """

    DETAIL_DELAY_MS = 120
    MARGIN = 0.25  # fraction of the view rendered beyond each edge, so short pans stay sharp
    OVERVIEW_MAX_PX = 4096
    PATH_CHUNK_QUADS = 4096

    def __init__(self, *args, **kwargs) -> None:
        self._color_table: list[int] | None = None
        self._geom_cache = None  # ("rect", x0, x1, y_lo, y_hi) or ("quads",)
        self._render_plan = None  # ("empty",) | ("rect", ...) | ("paths", [...]) | ("picture", QPicture)
        self._conn_buf = np.ones((self.PATH_CHUNK_QUADS, 5), dtype=np.int32)
        self._conn_buf[:, 4] = 0
        self._conn_flat = self._conn_buf.ravel()
        super().__init__(*args, **kwargs)
        self._color_table = (
            _color_table_from_colormap(self.cmap)
            if self.cmap is not None
            else _color_table_from_qcolors(self.lut_qcolor)
        )
        self._picture_used = None
        self._overview = None  # (QImage, item-coordinate rect)
        self._detail = None  # (QImage, inverse device transform, device transform, device offset)
        self._last_view = None  # (device transform, viewport size, device pixel ratio)
        self._detail_timer = QTimer()
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(self.DETAIL_DELAY_MS)
        self._detail_timer.timeout.connect(self._render_detail)

    def _prepareData(self, args):
        dirty = super()._prepareData(args)
        # Invalidate cached XY geometry only when XY coordinates or dimensions actually changed.
        if len(args) != 3 or args[0] is not None or args[1] is not None:
            self._geom_cache = None
        self._render_plan = None
        return dirty

    def _rerender(self, *, autoLevels):
        super()._rerender(autoLevels=autoLevels)
        self._render_plan = None

    def setLevels(self, levels, update=True):
        if (
            self.levels is not None
            and levels is not None
            and len(levels) == 2
            and float(levels[0]) == float(self.levels[0])
            and float(levels[1]) == float(self.levels[1])
        ):
            return
        super().setLevels(levels, update=update)

    def _apply_color_table(self, table: list[int], *, update: bool) -> None:
        self._color_table = table
        if self.edgecolors is not None:
            if update:
                self._rerender(autoLevels=False)
                self.update()
            return
        recolored = False
        if self._overview is not None and self._overview[0].format() == QImage.Format.Format_Indexed8:
            self._overview[0].setColorTable(table)
            recolored = True
        if self._detail is not None and self._detail[0].format() == QImage.Format.Format_Indexed8:
            self._detail[0].setColorTable(table)
            recolored = True
        if update:
            if not recolored:
                self.qpicture = None
                self._render_plan = None
            self.update()

    def setColorMap(self, cmap):
        self.cmap = cmap
        self.lut_qcolor = cmap.getLookupTable(nPts=256, mode=cmap.QCOLOR)
        if self.glstate is not None:
            self.setLookupTable(self.lut_qcolor, update=True)
            self.cmap = cmap
            return
        self._apply_color_table(_color_table_from_colormap(cmap), update=True)

    def setLookupTable(self, lut, update=True):
        self.cmap = None
        self.lut_qcolor = lut[:]
        if self.glstate is not None:
            super().setLookupTable(lut, update=update)
            return
        self._apply_color_table(_color_table_from_qcolors(lut), update=update)

    def _ensure_geom(self):
        if self._geom_cache is not None:
            return self._geom_cache
        x, y = self.x, self.y
        if x is None or y is None or self.z is None or self.z.size == 0:
            self._geom_cache = ("empty",)
            return self._geom_cache
        # O(1) quick check before scanning arrays: rectilinear uniform-X strip mesh?
        r_count, c_count = self.z.shape
        is_rect = False
        if (
            c_count > 0
            and r_count > 0
            and x[0, 0] == x[-1, 0]
            and x[0, -1] == x[-1, -1]
            and x[0, -1] > x[0, 0]
            and y[0, 0] == y[0, -1]
            and y[-1, 0] == y[-1, -1]
            and np.all(np.isfinite(x))
            and np.all(np.isfinite(y))
        ):
            expected_x = np.linspace(float(x[0, 0]), float(x[0, -1]), c_count + 1)
            if np.allclose(x, expected_x[None, :], rtol=1e-7, atol=1e-9) and np.allclose(
                y, y[:, :1], rtol=1e-7, atol=1e-9
            ):
                is_rect = True
        if is_rect:
            y_col = np.asarray(y[:, 0], dtype=float)
            self._geom_cache = ("rect", float(x[0, 0]), float(x[0, -1]), y_col[:-1].copy(), y_col[1:].copy())
        else:
            self._geom_cache = ("quads",)
        return self._geom_cache

    def _build_render_plan(self):
        if self.edgecolors is not None:
            return ("picture", super()._drawPicture())
        if self.z is None or self.z.size == 0:
            return ("empty",)
        geom = self._ensure_geom()
        if geom[0] == "empty":
            return ("empty",)
        if self.levels is None:
            finite = self.z[np.isfinite(self.z)]
            if finite.size == 0:
                return ("empty",)
            lo, hi = float(finite.min()), float(finite.max())
        else:
            lo, hi = float(self.levels[0]), float(self.levels[1])
        rng = hi - lo
        if rng == 0.0:
            rng = 1.0
        scale = 254.0 / rng

        if geom[0] == "rect":
            _, x0, x1, y_lo, y_hi = geom
            valid = np.isfinite(self.z)
            active_rows = np.flatnonzero(np.any(valid, axis=1))
            if active_rows.size == 0:
                return ("empty",)
            z_act = self.z[active_rows]
            v_act = valid[active_rows]
            bins = np.zeros(z_act.shape, dtype=np.uint8)
            scaled = np.clip(np.rint((z_act[v_act] - lo) * scale), 0.0, 254.0).astype(np.uint8)
            bins[v_act] = scaled + 1
            bins = np.ascontiguousarray(bins)
            rects = [
                (
                    QRectF(x0, float(min(y_lo[r], y_hi[r])), x1 - x0, float(abs(y_hi[r] - y_lo[r]))),
                    QRectF(0.0, float(k), float(bins.shape[1]), 1.0),
                )
                for k, r in enumerate(active_rows)
                if y_hi[r] != y_lo[r]
            ]
            if not rects:
                return ("empty",)
            src_img = QImage(bins.data, bins.shape[1], bins.shape[0], bins.shape[1], QImage.Format.Format_Indexed8)
            src_img.setColorTable(_IDENTITY_GRAY_ALPHA_TABLE)
            return ("rect", bins, src_img, rects)

        # Curved / general quadrilateral mesh: group finite cells by quantized color bin into chunked paths.
        valid = np.isfinite(self.z)
        rows, cols = np.nonzero(valid)
        if rows.size == 0:
            return ("empty",)
        zv = self.z[rows, cols]
        bins = np.clip(np.rint((zv - lo) * scale), 0.0, 254.0).astype(np.int32) + 1
        order = np.argsort(bins, kind="mergesort")
        sbins = bins[order]
        srows = rows[order]
        scols = cols[order]
        u_bins, starts, counts = np.unique(sbins, return_index=True, return_counts=True)

        x, y = self.x, self.y
        chunk = self.PATH_CHUNK_QUADS
        qx = np.empty((chunk, 5), dtype=np.float64)
        qy = np.empty((chunk, 5), dtype=np.float64)
        paths: list[tuple[int, QPainterPath]] = []
        for b_val, st, cnt in zip(u_bins, starts, counts):
            b_int = int(b_val)
            st_i = int(st)
            cnt_i = int(cnt)
            for off in range(0, cnt_i, chunk):
                end = min(cnt_i, off + chunk)
                k = end - off
                rr = srows[st_i + off : st_i + end]
                cc = scols[st_i + off : st_i + end]
                qx[:k, 0] = x[rr, cc]
                qy[:k, 0] = y[rr, cc]
                qx[:k, 1] = x[rr + 1, cc]
                qy[:k, 1] = y[rr + 1, cc]
                qx[:k, 2] = x[rr + 1, cc + 1]
                qy[:k, 2] = y[rr + 1, cc + 1]
                qx[:k, 3] = x[rr, cc + 1]
                qy[:k, 3] = y[rr, cc + 1]
                qx[:k, 4] = qx[:k, 0]
                qy[:k, 4] = qy[:k, 0]
                path = pg.arrayToQPath(qx[:k].ravel(), qy[:k].ravel(), connect=self._conn_flat[: 5 * k])
                paths.append((b_int, path))
        return ("paths", paths)

    def _ensure_plan(self):
        if self._render_plan is None:
            self._render_plan = self._build_render_plan()
        return self._render_plan

    def _paint_exact(self, painter: QPainter) -> None:
        """Draw exact cells directly to ``painter`` (used for PNG/SVG exports)."""
        plan = self._ensure_plan()
        kind = plan[0]
        if kind == "empty":
            return
        if kind == "picture":
            painter.drawPicture(0, 0, plan[1])
            return
        table = self._color_table or _IDENTITY_GRAY_ALPHA_TABLE
        if kind == "rect":
            _, bins, _, rects = plan
            export_img = QImage(bins.data, bins.shape[1], bins.shape[0], bins.shape[1], QImage.Format.Format_Indexed8)
            export_img.setColorTable(table)
            for dst_rect, src_rect in rects:
                painter.drawImage(dst_rect, export_img, src_rect)
            return
        # kind == "paths"
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        for b_int, path in plan[1]:
            painter.fillPath(path, QColor.fromRgba(table[b_int]))
        painter.restore()

    def _render(self, size: QSize, dpr: float, transform: QTransform) -> QImage:
        plan = self._ensure_plan()
        kind = plan[0]
        if kind == "picture":
            image = QImage(size * dpr, QImage.Format.Format_ARGB32_Premultiplied)
            image.setDevicePixelRatio(dpr)
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            painter.setTransform(transform)
            painter.drawPicture(0, 0, plan[1])
            painter.end()
            return image
        image = QImage(size * dpr, QImage.Format.Format_Grayscale8)
        image.setDevicePixelRatio(dpr)
        image.fill(0)
        if kind != "empty":
            painter = QPainter(image)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setTransform(transform)
            if kind == "rect":
                _, _, src_img, rects = plan
                for dst_rect, src_rect in rects:
                    painter.drawImage(dst_rect, src_img, src_rect)
            else:  # kind == "paths"
                for b_int, path in plan[1]:
                    painter.fillPath(path, QColor(b_int, b_int, b_int))
            painter.end()
        image.reinterpretAsFormat(QImage.Format.Format_Indexed8)
        image.setColorTable(self._color_table or _IDENTITY_GRAY_ALPHA_TABLE)
        return image

    def _render_overview(self, viewport: QSize, dpr: float) -> None:
        rect = self.boundingRect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        limit = self.OVERVIEW_MAX_PX / dpr
        w = int(min(limit, max(64, viewport.width())))
        h = int(min(limit, max(64, viewport.height())))
        transform = QTransform.fromScale(w / rect.width(), h / rect.height())
        transform.translate(-rect.left(), -rect.top())
        self._overview = (self._render(QSize(w, h), dpr, transform), QRectF(rect))

    def _render_detail(self) -> None:
        if self._last_view is None or self.qpicture is None or self.scene() is None:
            return
        transform, viewport, dpr = self._last_view
        inverse, ok = transform.inverted()
        if not ok:
            return
        mx, my = int(viewport.width() * self.MARGIN), int(viewport.height() * self.MARGIN)
        shifted = transform * QTransform.fromTranslate(mx, my)
        image = self._render(QSize(viewport.width() + 2 * mx, viewport.height() + 2 * my), dpr, shifted)
        self._detail = (image, inverse, QTransform(transform), QPointF(-mx, -my))
        self.update()

    def paint(self, painter, opt, widget):
        if self.z is None:
            return
        if self._exportOpts is not False or widget is None:
            self._paint_exact(painter)  # exports: draw the exact cells
            return
        if self.qpicture is None:
            self._ensure_plan()
            self.qpicture = self._render_plan[1] if self._render_plan[0] == "picture" else QPicture()
        if self._picture_used is not self.qpicture:  # new data or levels
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
        if detail is not None:
            detail_img, detail_inv, _, offset = detail
            # Map detail_img logical pixels -> old viewport -> item -> current viewport.
            # Drawing through this relative device transform preserves orientation under
            # Y-up plot coordinates and inverted X/Y viewboxes alike (no vertical/horizontal flip).
            rel = QTransform.fromTranslate(offset.x(), offset.y()) * detail_inv * transform
            detail_dpr = detail_img.devicePixelRatio() or 1.0
            detail_logical = QRectF(
                0.0,
                0.0,
                detail_img.width() / detail_dpr,
                detail_img.height() / detail_dpr,
            )
            detail_view_rect = rel.mapRect(detail_logical)
            viewport_rect = QRectF(0.0, 0.0, float(viewport.width()), float(viewport.height()))
            if not detail_view_rect.contains(viewport_rect):
                if self._overview is None:
                    self._render_overview(viewport, dpr)
                if self._overview is not None:
                    painter.save()
                    if detail_view_rect.width() > 2.0 and detail_view_rect.height() > 2.0:
                        clip_outside = QPainterPath()
                        clip_outside.addRect(viewport_rect)
                        inner = QPainterPath()
                        inner.addRect(detail_view_rect.adjusted(0.5, 0.5, -0.5, -0.5))
                        clip_outside = clip_outside.subtracted(inner)
                        painter.resetTransform()
                        painter.setClipPath(clip_outside, Qt.ClipOperation.IntersectClip)
                        painter.setTransform(transform)
                    painter.drawImage(self._overview[1], self._overview[0])
                    painter.restore()
            painter.save()
            painter.setTransform(rel)
            painter.drawImage(QPointF(0.0, 0.0), detail_img)
            painter.restore()
        else:
            if self._overview is None:
                self._render_overview(viewport, dpr)
            if self._overview is not None:
                painter.drawImage(self._overview[1], self._overview[0])
        self._detail_timer.start()  # sharpen once the view has stopped moving


class SmoothResampledMapItem(pg.GraphicsObject):
    """Smoothly re-binned ``(α_i, α_f)`` or ``(Qx, Qz)`` map rendered at viewport resolution.

    Eliminates rectangular flat-cell / quad boxing artifacts by evaluating the exact
    analytic inverse mapping ``(Qx, Qz) -> (α_i, α_f) -> (frame, pixel)`` with bilinear
    interpolation directly on a regular display grid matching the active viewport (plus
    an overview image for smooth panning/zooming). Display-only: never alters data, cuts,
    cursor readouts, or table exports.
    """

    DETAIL_DELAY_MS = 120
    MARGIN = 0.20
    OVERVIEW_MAX_PX = 800
    DETAIL_MAX_PX = 1000
    EXPORT_MAX_PX = 1200

    def __init__(self, *, colorMap: pg.ColorMap | None = None) -> None:
        super().__init__()
        self.cmap: pg.ColorMap | None = colorMap
        self.levels: tuple[float, float] | None = None
        self._color_table: list[int] = (
            _color_table_from_colormap(colorMap) if colorMap is not None else _IDENTITY_GRAY_ALPHA_TABLE
        )
        self._source: dict | None = None
        self._rect = QRectF()
        self._overview = None  # (vals, bins, QImage, QRectF)
        self._detail = None  # (vals, bins, QImage, QRectF, scale_xy)
        self._last_view = None  # (QTransform, QSize, dpr)
        self._detail_timer = QTimer()
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(self.DETAIL_DELAY_MS)
        self._detail_timer.timeout.connect(self._render_detail)

    def set_source(
        self,
        src: np.ndarray,
        ai_sorted: np.ndarray,
        base_sorted: np.ndarray,
        strips_sorted: np.ndarray,
        nx: int,
        cal: AngleCalibration,
        extent: tuple[float, float, float, float],
        *,
        q_mode: bool,
        wavelength: float | None = None,
    ) -> None:
        self._detail_timer.stop()
        x0, x1, y0, y1 = (float(v) for v in extent)
        new_rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        if new_rect != self._rect:
            self.prepareGeometryChange()
            self._rect = new_rect
        self._source = dict(
            src=np.asarray(src, dtype=float),
            ai=np.asarray(ai_sorted, dtype=float),
            base=np.asarray(base_sorted, dtype=float),
            strips=np.asarray(strips_sorted, dtype=float),
            nx=int(nx),
            cal=cal,
            q_mode=bool(q_mode),
            wavelength=None if wavelength is None else float(wavelength),
        )
        self._overview = None
        self._detail = None
        self.update()

    def getColorMap(self) -> pg.ColorMap | None:
        return self.cmap

    def setColorMap(self, cmap: pg.ColorMap) -> None:
        self.cmap = cmap
        self._apply_color_table(_color_table_from_colormap(cmap))

    def setLookupTable(self, lut, update: bool = True) -> None:
        self.cmap = None
        self._apply_color_table(_color_table_from_qcolors(lut), update=update)

    def _apply_color_table(self, table: list[int], *, update: bool = True) -> None:
        self._color_table = table
        if self._overview is not None:
            self._overview[2].setColorTable(table)
        if self._detail is not None:
            self._detail[2].setColorTable(table)
        if update:
            self.update()

    def getLevels(self) -> tuple[float, float] | None:
        return self.levels

    def setLevels(self, levels, update: bool = True) -> None:
        if levels is None or len(levels) != 2:
            return
        new_lv = (float(levels[0]), float(levels[1]))
        if self.levels == new_lv:
            return
        self.levels = new_lv
        # Re-bin cached float arrays into uint8 LUT indices in-place (<2 ms, no resampling)
        if self._overview is not None:
            vals, _, _, rect = self._overview
            bins, img = _quantize_indexed_image(vals, self.levels, self._color_table)
            self._overview = (vals, bins, img, rect)
        if self._detail is not None:
            vals, _, _, rect, scale_xy = self._detail
            bins, img = _quantize_indexed_image(vals, self.levels, self._color_table)
            self._detail = (vals, bins, img, rect, scale_xy)
        if update:
            self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(self._rect)

    def sample_grid(self, x0: float, x1: float, y0: float, y1: float, w_px: int, h_px: int) -> np.ndarray:
        """Evaluate the smoothly re-binned display field on a regular ``(h_px, w_px)`` grid."""
        s = self._source
        if s is None or w_px <= 0 or h_px <= 0 or x1 <= x0 or y1 <= y0:
            return np.full((max(1, h_px), max(1, w_px)), np.nan, dtype=float)
        dx = (x1 - x0) / float(w_px)
        dy = (y1 - y0) / float(h_px)
        xs = np.linspace(x0 + 0.5 * dx, x1 - 0.5 * dx, w_px)
        ys = np.linspace(y0 + 0.5 * dy, y1 - 0.5 * dy, h_px)
        X, Y = np.meshgrid(xs, ys)
        if s["q_mode"]:
            ai_q, af_q = angles_from_q(X / QX_SCALE, Y, s["wavelength"])
        else:
            ai_q, af_q = X, Y
        return _sample_bilinear_ai_af(s["src"], s["ai"], s["base"], s["strips"], s["nx"], s["cal"], ai_q, af_q)

    def _ensure_overview(self) -> None:
        if self._overview is not None or self._source is None or self._rect.isEmpty():
            return
        r = self._rect
        w_px = self.OVERVIEW_MAX_PX
        h_px = int(round(self.OVERVIEW_MAX_PX * 0.8))
        vals = self.sample_grid(r.left(), r.right(), r.top(), r.bottom(), w_px, h_px)
        bins, img = _quantize_indexed_image(vals, self.levels, self._color_table)
        self._overview = (vals, bins, img, QRectF(r))

    def _view_item_rect(self, transform: QTransform, viewport: QSize) -> QRectF | None:
        inv, ok = transform.inverted()
        if not ok or viewport.width() <= 0 or viewport.height() <= 0:
            return None
        vp = QRectF(0.0, 0.0, float(viewport.width()), float(viewport.height()))
        return inv.mapRect(vp).normalized()

    def _render_detail(self) -> None:
        if self._source is None or self._last_view is None or self._rect.isEmpty():
            return
        transform, viewport, dpr = self._last_view
        vis = self._view_item_rect(transform, viewport)
        if vis is None or vis.isEmpty():
            return
        mx = vis.width() * self.MARGIN
        my = vis.height() * self.MARGIN
        padded = vis.adjusted(-mx, -my, mx, my)
        target = padded.intersected(self._rect)
        if target.isEmpty() or target.width() <= 0.0 or target.height() <= 0.0:
            return
        dev_rect = transform.mapRect(target).normalized()
        w_px = int(min(self.DETAIL_MAX_PX, max(64, round(dev_rect.width() * dpr))))
        h_px = int(min(self.DETAIL_MAX_PX, max(64, round(dev_rect.height() * dpr))))
        vals = self.sample_grid(target.left(), target.right(), target.top(), target.bottom(), w_px, h_px)
        bins, img = _quantize_indexed_image(vals, self.levels, self._color_table)
        scale_xy = (abs(transform.m11()), abs(transform.m22()))
        self._detail = (vals, bins, img, QRectF(target), scale_xy)
        self.update()

    def paint(self, painter: QPainter, opt, widget) -> None:
        if self._source is None or self._rect.isEmpty():
            return
        if self._exportOpts is not False or widget is None:
            r = self._rect
            vals = self.sample_grid(r.left(), r.right(), r.top(), r.bottom(), self.EXPORT_MAX_PX, int(round(self.EXPORT_MAX_PX * 0.8)))
            _, img = _quantize_indexed_image(vals, self.levels, self._color_table)
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(r, img)
            painter.restore()
            return

        transform = painter.transform()
        viewport = widget.size()
        dpr = widget.devicePixelRatioF()
        self._last_view = (QTransform(transform), viewport, dpr)
        vis = self._view_item_rect(transform, viewport)
        cur_scale = (abs(transform.m11()), abs(transform.m22()))

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        detail = self._detail
        detail_fresh = False
        if detail is not None and vis is not None:
            _, _, detail_img, detail_rect, detail_scale = detail
            same_scale = (
                abs(detail_scale[0] - cur_scale[0]) <= 0.08 * max(1e-12, cur_scale[0])
                and abs(detail_scale[1] - cur_scale[1]) <= 0.08 * max(1e-12, cur_scale[1])
            )
            needed = vis.intersected(self._rect)
            covers = detail_rect.adjusted(-1e-9, -1e-9, 1e-9, 1e-9).contains(needed)
            if same_scale and covers:
                painter.drawImage(detail_rect, detail_img)
                painter.restore()
                return
            detail_fresh = False
            self._ensure_overview()
            if self._overview is not None:
                painter.drawImage(self._overview[3], self._overview[2])
            painter.drawImage(detail_rect, detail_img)
        else:
            self._ensure_overview()
            if self._overview is not None:
                painter.drawImage(self._overview[3], self._overview[2])
        painter.restore()
        if not detail_fresh:
            self._detail_timer.start()


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


class _CompactPlotWidget(pg.PlotWidget):
    """PlotWidget with a compact preferred sizeHint so side-panel scroll areas do not inflate."""

    def sizeHint(self) -> QSize:
        return QSize(180, 120)


class OffSpecularView(QWidget):
    """Heatmap of Y-integrated detector intensity versus X pixel and scan point."""

    curvature_changed = pyqtSignal(str)  # emits active scan file_path when curvature changes

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
        self._alpha_items: list[pg.GraphicsObject] = []
        self._alpha_geom: dict | None = None
        self._th_geom: dict | None = None
        self._cal_message = ""
        self._colorbar_on_strips = False
        self._applied_cmap_key: tuple[bool, str] | None = None
        self._colorbar_items_rebuilt = True
        self._last_tick_key = None
        self._baseline_levels: dict[tuple, tuple[float, float]] = {}
        self._last_level_key: tuple | None = None
        self._preserve_levels = False
        self.session_source_paths = lambda: []  # every loaded file; set by the main window
        self._cut_values: np.ndarray | None = None  # what the map shows (asymmetry: low counts blanked)
        self._init_ui()
        self._set_enabled(False)

    # ------------------------------------------------------------------ layout
    def _init_ui(self) -> None:
        self.setMinimumSize(0, 0)
        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(8)

        bar = FlowContainer(margin=0, h_spacing=8, v_spacing=6)
        self.combo_spin = configure_compact_combo(QComboBox(), min_chars=14)
        self.combo_spin.addItem("All Frames", "all")
        self.combo_spin.addItem("u — Spin Up (Flipper OFF / 1)", "up")
        self.combo_spin.addItem("d — Spin Down (Flipper ON / 0)", "down")
        self.combo_spin.addItem("u + d — Sum of channels", "sum")
        self.combo_spin.addItem("Spin asymmetry (u−d)/(u+d)", "asym")
        fit_combo_popup(self.combo_spin)
        self.combo_spin.currentIndexChanged.connect(self._on_spin_changed)
        bar.addWidget(make_flow_group(QLabel("Spin:"), self.combo_spin, spacing=6))

        self.spin_min_counts = QSpinBox()
        self.spin_min_counts.setMinimumWidth(0)
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

        self.combo_y_source = configure_compact_combo(QComboBox(), min_chars=12)
        self.combo_y_source.addItem("Full detector height", "full")
        self.combo_y_source.addItem("Scan ROI Y range", "roi")
        self.combo_y_source.addItem("Custom Y range", "custom")
        fit_combo_popup(self.combo_y_source)
        self.combo_y_source.setToolTip("Rows of each detector image summed for every X pixel")
        self.combo_y_source.currentIndexChanged.connect(self._on_y_source_changed)
        bar.addWidget(make_flow_group(QLabel("Integrate Y:"), self.combo_y_source, spacing=6))

        # The row spinboxes always show the Y window being integrated. Editing
        # them switches to "Custom Y range" (e.g. to exclude noisy detector edges).
        self.spin_y0 = QSpinBox()
        self.spin_y1 = QSpinBox()
        for spin, tip in ((self.spin_y0, "First detector Y row integrated (inclusive)"),
                          (self.spin_y1, "Last detector Y row integrated (exclusive)")):
            spin.setRange(0, 1024)
            spin.setFixedWidth(88)
            spin.setKeyboardTracking(False)
            spin.setToolTip(tip + ". Editing switches to Custom Y range.")
            spin.valueChanged.connect(self._on_row_spin_changed)
        bar.addWidget(make_flow_group(QLabel("Rows:"), self.spin_y0, spacing=6))
        bar.addWidget(self.spin_y1)
        self._row_timer = QTimer(self)
        self._row_timer.setSingleShot(True)
        self._row_timer.setInterval(350)  # wait for arrow-key/click bursts before recomputing
        self._row_timer.timeout.connect(self._request_compute)

        self.combo_norm = configure_compact_combo(QComboBox(), min_chars=11)
        self.combo_norm.addItem("Raw counts", "raw")
        self.combo_norm.addItem("Counts / s", "time")
        self.combo_norm.addItem("Counts / Monitor1", "monitor")
        fit_combo_popup(self.combo_norm)
        self.combo_norm.currentIndexChanged.connect(self._redraw)
        bar.addWidget(make_flow_group(QLabel("Normalize:"), self.combo_norm, spacing=6))

        self.combo_y_axis = configure_compact_combo(QComboBox(), min_chars=10)
        self.combo_y_axis.addItem("X pixel vs θ", "th")
        self.combo_y_axis.addItem("X pixel vs Point", "point")
        self.combo_y_axis.addItem("α_i vs α_f", "alpha")
        self.combo_y_axis.addItem("Qx vs Qz", "q")
        fit_combo_popup(self.combo_y_axis)
        self.combo_y_axis.setToolTip("α_i vs α_f and Qx vs Qz use this scan's angle calibration "
                                     "(right-hand panel); Qx–Qz also uses λ")
        self.combo_y_axis.currentIndexChanged.connect(self._on_axes_changed)
        bar.addWidget(make_flow_group(QLabel("Axes:"), self.combo_y_axis, spacing=6))
        main.addWidget(bar)

        # Second control row: display options (Colormap + Smooth sit below Normalize / Axes)
        self._display_bar = display_bar = FlowContainer(margin=0, h_spacing=8, v_spacing=6)
        self.chk_log = QCheckBox("Log Scale")
        self.chk_log.setChecked(True)
        self.chk_log.toggled.connect(self._redraw)
        display_bar.addWidget(self.chk_log)
        self.chk_show_roi = QCheckBox("Show ROI")
        self.chk_show_roi.setChecked(True)
        self.chk_show_roi.toggled.connect(self._draw_roi_overlay)
        display_bar.addWidget(self.chk_show_roi)

        self.combo_cmap = configure_compact_combo(QComboBox(), min_chars=7)
        self.combo_cmap.addItems(["turbo", "viridis", "inferno", "plasma", "cividis", "gray", "magma"])
        fit_combo_popup(self.combo_cmap)
        self.combo_cmap.currentTextChanged.connect(self._on_cmap_changed)
        self._cmap_group = make_flow_group(QLabel("Colormap:"), self.combo_cmap, spacing=6)
        display_bar.addWidget(self._cmap_group)

        self.chk_smooth_map = QCheckBox("Smooth")
        self.chk_smooth_map.setChecked(False)
        self.chk_smooth_map.setToolTip(
            "Smoothly re-bin α_i vs α_f and Qx vs Qz onto a regular display grid (bilinear interpolation + "
            "optional edge-preserving bilateral denoising in display space). Display-only; cuts, cursor readout, "
            "and exported tables remain exact and unsmoothed."
        )
        self.chk_smooth_map.toggled.connect(self._on_smooth_changed)
        self.spin_smooth_sigma = QDoubleSpinBox()
        self.spin_smooth_sigma.setRange(0.0, 4.0)
        self.spin_smooth_sigma.setDecimals(1)
        self.spin_smooth_sigma.setSingleStep(0.5)
        self.spin_smooth_sigma.setValue(1.0)
        self.spin_smooth_sigma.setSuffix(" px")
        self.spin_smooth_sigma.setFixedWidth(78)
        self.spin_smooth_sigma.setKeyboardTracking(False)
        self.spin_smooth_sigma.setEnabled(False)
        self.spin_smooth_sigma.setToolTip(
            "Edge-preserving bilateral smoothing radius σ (0.0 px = smooth bilinear re-binning only; "
            "> 0 adds log-space edge-preserving denoising without blooming bright ridges)."
        )
        self.spin_smooth_sigma.valueChanged.connect(self._on_smooth_changed)
        self._smooth_group = make_flow_group(self.chk_smooth_map, self.spin_smooth_sigma, spacing=4)
        self._smooth_group.setVisible(self._alpha_mode())
        display_bar.addWidget(self._smooth_group)
        main.addWidget(display_bar)

        # Only problems and caveats are shown here (hidden otherwise); the settings
        # above already describe what the map shows.
        self.lbl_notice = QLabel("")
        self.lbl_notice.setStyleSheet("color: #b45309; font-size: 11px; padding: 0 2px;")
        self.lbl_notice.setWordWrap(True)
        _fixed_width_text(self.lbl_notice)
        self.lbl_notice.setVisible(False)
        main.addWidget(self.lbl_notice)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(True)
        splitter.setMinimumSize(0, 0)
        splitter.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        left = QWidget()
        left.setMinimumSize(0, 0)
        left.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        tools = FlowContainer(margin=0, h_spacing=6, v_spacing=4)
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
        self.btn_crosshair = _tool_button("Cut lines / crosshair: drop draggable horizontal & vertical cut "
                                          "lines on the map; click to jump or toggle off to remove")
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
        self.lbl_coords = QLabel("")
        self.lbl_coords.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_coords.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _fixed_width_text(self.lbl_coords)
        tools.addWidget(self.lbl_coords)
        self.progress = QProgressBar()
        self.progress.setMinimumWidth(0)
        self.progress.setMaximumWidth(160)
        self.progress.setTextVisible(False)
        self.progress.setToolTip("Integrating detector frames")
        self.progress.setVisible(False)
        tools.addWidget(self.progress)
        left_layout.addWidget(tools)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumSize(0, 0)
        self.plot.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.plot.getViewBox().setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.plot.setLabel("bottom", "Detector X pixel")
        self.plot.setLabel("left", "θ (°)")
        self.img_item = pg.ImageItem()
        self.img_item.setColorMap(get_safe_colormap("turbo"))
        self.plot.addItem(self.img_item)
        self.colorbar = pg.ColorBarItem(values=(0, 1), colorMap=get_safe_colormap("turbo"), interactive=True, width=14)
        self.colorbar.setImageItem(self.img_item, insert_in=self.plot.getPlotItem())
        # Draggable cut lines linked to the 1D cut plots, plus a shared crosshair intersection marker.
        hover_pen = pg.mkPen("#facc15", width=2.5)
        self.line_row = pg.InfiniteLine(
            angle=0,
            movable=True,
            pen=pg.mkPen("#38bdf8", width=1.5, style=Qt.PenStyle.DashLine),
            hoverPen=hover_pen,
        )
        self.line_x = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen("#f43f5e", width=1.5, style=Qt.PenStyle.DashLine),
            hoverPen=hover_pen,
        )
        self.crosshair = pg.TargetItem(pos=(0, 0), size=16, symbol="crosshair", movable=True,
                                       pen=pg.mkPen("#ffffff", width=2), hoverPen=pg.mkPen("#facc15", width=2),
                                       brush=pg.mkBrush(0, 0, 0, 0))
        for item, z in ((self.line_row, 20), (self.line_x, 20), (self.crosshair, 21)):
            item.setZValue(z)
            item.setVisible(False)
            self.plot.addItem(item)
        self.line_row.sigPositionChanged.connect(self._on_line_row_moved)
        self.line_x.sigPositionChanged.connect(self._on_line_x_moved)
        self.crosshair.sigPositionChanged.connect(self._on_crosshair_moved)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._on_map_clicked)
        apply_plot_theme(self.plot, self.is_dark_mode)
        self.plot.showGrid(x=False, y=False)  # see set_theme
        self._apply_no_data_background()
        left_layout.addWidget(self.plot, 1)
        splitter.addWidget(left)

        right = QWidget()
        right.setMinimumSize(0, 0)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(self._build_calibration_panel())
        right_layout.addWidget(self._build_curvature_panel())
        group = QGroupBox("Cuts through the map")
        group.setMinimumSize(0, 0)
        group_layout = QVBoxLayout(group)
        cut_bar = FlowLayout(margin=0, h_spacing=6, v_spacing=4)
        self.btn_drop_cuts = QPushButton("Drop Cut Lines")
        self.btn_drop_cuts.setCheckable(True)
        self.btn_drop_cuts.setToolTip(
            "Drop or remove linked draggable horizontal and vertical cut lines on the 2D map"
        )
        self.btn_drop_cuts.toggled.connect(self._on_drop_cuts_toggled)
        self.btn_cut_h = QPushButton("Horizontal")
        self.btn_cut_h.setCheckable(True)
        self.btn_cut_h.setToolTip("Show or hide the draggable horizontal cut line on the 2D map")
        self.btn_cut_h.toggled.connect(self._on_single_cut_toggled)
        self.btn_cut_v = QPushButton("Vertical")
        self.btn_cut_v.setCheckable(True)
        self.btn_cut_v.setToolTip("Show or hide the draggable vertical cut line on the 2D map")
        self.btn_cut_v.toggled.connect(self._on_single_cut_toggled)
        for btn in (self.btn_drop_cuts, self.btn_cut_h, self.btn_cut_v):
            btn.setStyleSheet("QPushButton { padding: 4px 8px; }")
            cut_bar.addWidget(btn)
        self.chk_roi_sum_cut = QCheckBox("ROI-summed profile")
        self.chk_roi_sum_cut.setToolTip(
            "Plot the X-integrated intensity across the signal ROI (updates live as sample curvature changes)"
        )
        self.chk_roi_sum_cut.toggled.connect(self._on_roi_sum_cut_toggled)
        cut_bar.addWidget(self.chk_roi_sum_cut)
        group_layout.addLayout(cut_bar)
        self.lbl_row_cut = QLabel("Horizontal cut (selected point):")
        _fixed_width_text(self.lbl_row_cut)
        group_layout.addWidget(self.lbl_row_cut)
        self.plot_row = _CompactPlotWidget(
            axisItems={"left": ScientificAxisItem(orientation="left", linear_scientific=False)}
        )
        self.plot_row.setMinimumHeight(110)
        self.plot_row.setLabel("bottom", "Detector X pixel")
        group_layout.addWidget(self.plot_row, 1)
        self.lbl_col_cut = QLabel("Vertical cut (selected X pixel):")
        _fixed_width_text(self.lbl_col_cut)
        group_layout.addWidget(self.lbl_col_cut)
        self.plot_col = _CompactPlotWidget(
            axisItems={"left": ScientificAxisItem(orientation="left", linear_scientific=False)}
        )
        self.plot_col.setMinimumHeight(110)
        group_layout.addWidget(self.plot_col, 1)
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

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setMinimumSize(0, 0)
        right_scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([760, 340])
        main.addWidget(splitter, 1)
        install_wheel_focus_guard(self)

    def _build_calibration_panel(self) -> QGroupBox:
        group = self.cal_group = QGroupBox("Angle calibration (α_i, α_f) — this scan")
        group.setMinimumSize(0, 0)
        outer = QVBoxLayout(group)
        outer.setSpacing(6)

        # Mode toggle at the top of the calibration card: Transmitted-beam fit vs pySAred geometry.
        mode_row = FlowLayout(margin=0, h_spacing=6, v_spacing=4)
        self.btn_mode_direct_beam = QPushButton("Transmitted-beam fit")
        self.btn_mode_direct_beam.setCheckable(True)
        self.btn_mode_direct_beam.setChecked(True)
        self.btn_mode_direct_beam.setToolTip(
            "Linear pixel-to-angle calibration fitted from the transmitted-beam streak (°/px)"
        )
        self.btn_mode_pysared = QPushButton("pySAred geometry")
        self.btn_mode_pysared.setCheckable(True)
        self.btn_mode_pysared.setToolTip(
            "Flat-detector arctan geometry using manual sample–detector distance L and detector width (pySAred convention)"
        )
        self.cal_mode_group = QButtonGroup(self)
        self.cal_mode_group.setExclusive(True)
        self.cal_mode_group.addButton(self.btn_mode_direct_beam, 0)
        self.cal_mode_group.addButton(self.btn_mode_pysared, 1)
        for btn in (self.btn_mode_direct_beam, self.btn_mode_pysared):
            btn.setStyleSheet("QPushButton { padding: 4px 8px; }")
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            mode_row.addWidget(btn)
        self.btn_mode_direct_beam.clicked.connect(lambda: self._on_cal_mode_changed("direct_beam"))
        self.btn_mode_pysared.clicked.connect(lambda: self._on_cal_mode_changed("pysared"))
        outer.addLayout(mode_row)

        # Mode-specific controls in a QStackedWidget (2 rows per page so toggling never jumps layout).
        self.cal_mode_stack = QStackedWidget()
        self.cal_mode_stack.setMinimumSize(0, 0)

        page_db = QWidget()
        db_grid = QGridLayout(page_db)
        db_grid.setContentsMargins(0, 0, 0, 0)
        db_grid.setHorizontalSpacing(6)
        self.spin_dpp = QDoubleSpinBox()
        self.spin_dpp.setMinimumWidth(65)
        self.spin_dpp.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_dpp.setDecimals(6)
        self.spin_dpp.setRange(0.0, 1.0)
        self.spin_dpp.setSingleStep(0.00001)
        self.spin_dpp.setSuffix(" °/px")
        self.spin_dpp.setSpecialValueText("not set")
        self.spin_dpp.setKeyboardTracking(False)
        self.spin_dpp.setToolTip("Exit-angle step per detector X pixel (from the transmitted-beam fit; editable)")
        self.spin_dpp.valueChanged.connect(lambda v: self._on_calibration_edited(deg_per_pixel=v))
        self.btn_fit_db = QPushButton("Fit Transmitted Beam")
        self.btn_fit_db.setToolTip("Refit °/pixel from the transmitted-beam streak and x₀ from the specular peak "
                                   "(discards manual values for this scan)")
        self.btn_fit_db.clicked.connect(self._refit_calibration)
        self.lbl_distance = QLabel("—")
        _fixed_width_text(self.lbl_distance)
        self.lbl_distance.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_distance.setToolTip(
            "Check value only: sample-to-detector distance L = p / tan(Δ) implied by the "
            "calibrated °/pixel scale Δ assuming nominal pixel pitch p = 300 mm / nx "
            "(full readout width). Not used in any angle (α_f) or Q conversion.")
        db_grid.addWidget(QLabel("Scale:"), 0, 0)
        db_grid.addWidget(self.spin_dpp, 0, 1)
        db_grid.addWidget(self.btn_fit_db, 0, 2, 1, 2)
        db_grid.addWidget(QLabel("Distance:"), 1, 0)
        db_grid.addWidget(self.lbl_distance, 1, 1, 1, 3)
        db_grid.setColumnStretch(1, 1)
        db_grid.setColumnStretch(3, 1)
        self.cal_mode_stack.addWidget(page_db)

        page_ps = QWidget()
        ps_grid = QGridLayout(page_ps)
        ps_grid.setContentsMargins(0, 0, 0, 0)
        ps_grid.setHorizontalSpacing(6)
        self.spin_distance_mm = QDoubleSpinBox()
        self.spin_distance_mm.setMinimumWidth(65)
        self.spin_distance_mm.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_distance_mm.setDecimals(1)
        self.spin_distance_mm.setRange(10.0, 50000.0)
        self.spin_distance_mm.setSingleStep(10.0)
        self.spin_distance_mm.setValue(PYSARED_DEFAULT_DISTANCE_MM)
        self.spin_distance_mm.setSuffix(" mm")
        self.spin_distance_mm.setKeyboardTracking(False)
        self.spin_distance_mm.setToolTip("Sample-to-detector distance L used in pySAred flat-detector arctan geometry")
        self.spin_distance_mm.valueChanged.connect(lambda v: self._on_calibration_edited(distance_mm=v))
        self.spin_det_width_mm = QDoubleSpinBox()
        self.spin_det_width_mm.setMinimumWidth(65)
        self.spin_det_width_mm.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_det_width_mm.setDecimals(1)
        self.spin_det_width_mm.setRange(1.0, 5000.0)
        self.spin_det_width_mm.setSingleStep(1.0)
        self.spin_det_width_mm.setValue(DETECTOR_WIDTH_MM)
        self.spin_det_width_mm.setSuffix(" mm")
        self.spin_det_width_mm.setKeyboardTracking(False)
        self.spin_det_width_mm.setToolTip("Active detector readout width along X (pixel pitch p = width / nx)")
        self.spin_det_width_mm.valueChanged.connect(lambda v: self._on_calibration_edited(detector_width_mm=v))
        self.lbl_effective_scale = QLabel("—")
        _fixed_width_text(self.lbl_effective_scale)
        self.lbl_effective_scale.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_effective_scale.setToolTip(
            "Effective small-angle scale degrees(arctan((width / nx) / L)) at the detector normal (x₀); "
            "actual exit angles use the full flat-detector arctan mapping."
        )
        ps_grid.addWidget(QLabel("Distance L:"), 0, 0)
        ps_grid.addWidget(self.spin_distance_mm, 0, 1)
        ps_grid.addWidget(QLabel("Width:"), 0, 2)
        ps_grid.addWidget(self.spin_det_width_mm, 0, 3)
        ps_grid.addWidget(QLabel("Eff. scale:"), 1, 0)
        ps_grid.addWidget(self.lbl_effective_scale, 1, 1, 1, 3)
        ps_grid.setColumnStretch(1, 1)
        ps_grid.setColumnStretch(3, 1)
        self.cal_mode_stack.addWidget(page_ps)

        outer.addWidget(self.cal_mode_stack)

        # Shared parameters across both calibration modes: x₀, λ, direction sign, and reference lines.
        shared = QGridLayout()
        shared.setHorizontalSpacing(6)
        self.spin_x0 = QDoubleSpinBox()
        self.spin_x0.setMinimumWidth(65)
        self.spin_x0.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_x0.setDecimals(1)
        self.spin_x0.setRange(0.0, 4096.0)
        self.spin_x0.setSuffix(" px")
        self.spin_x0.setKeyboardTracking(False)
        self.spin_x0.setToolTip("Pixel of exit angle 2θ − θ (from the specular peak; editable)")
        self.spin_x0.valueChanged.connect(lambda v: self._on_calibration_edited(x0=v))
        self.spin_wavelength = QDoubleSpinBox()
        self.spin_wavelength.setMinimumWidth(65)
        self.spin_wavelength.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_wavelength.setRange(0.1, 50.0)
        self.spin_wavelength.setDecimals(3)
        self.spin_wavelength.setSingleStep(0.01)
        self.spin_wavelength.setValue(5.21)  # SuperADAM monochromatic beam, as in Scan Charts
        self.spin_wavelength.setSuffix(" Å")
        self.spin_wavelength.setKeyboardTracking(False)
        self.spin_wavelength.setToolTip("Neutron wavelength used for Qx and Qz")
        self.spin_wavelength.valueChanged.connect(self._on_wavelength_changed)
        shared.addWidget(QLabel("x₀:"), 0, 0)
        shared.addWidget(self.spin_x0, 0, 1)
        shared.addWidget(QLabel("λ:"), 0, 2)
        shared.addWidget(self.spin_wavelength, 0, 3)
        shared.setColumnStretch(1, 1)
        shared.setColumnStretch(3, 1)
        outer.addLayout(shared)

        checks_flow = FlowLayout(margin=0, h_spacing=8, v_spacing=4)
        self.chk_af_increases = QCheckBox("α_f increases with X")
        self.chk_af_increases.setToolTip("Direction of the exit angle along displayed X")
        self.chk_af_increases.toggled.connect(lambda on: self._on_calibration_edited(sign=1 if on else -1))
        self.chk_show_db = QCheckBox("Show fit / reference lines")
        self.chk_show_db.setToolTip("Pixel axes: fitted transmitted-beam line. α axes: α_f = α_i (specular), "
                                    "α_f = −α_i (transmitted beam) and the ROI; untick for a clean α map")
        self.chk_show_db.setChecked(True)
        self.chk_show_db.toggled.connect(self._draw_roi_overlay)
        checks_flow.addWidget(self.chk_af_increases)
        checks_flow.addWidget(self.chk_show_db)
        outer.addLayout(checks_flow)

        self._set_cal_status("Calibrated automatically from the transmitted beam when the map is computed.")
        return group

    def _build_curvature_panel(self) -> QGroupBox:
        group = self.curv_group = QGroupBox("Sample curvature (pySAred 1.5.3) — this scan")
        group.setMinimumSize(0, 0)
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        self.spin_curvature = QDoubleSpinBox()
        self.spin_curvature.setMinimumWidth(65)
        self.spin_curvature.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.spin_curvature.setRange(-1.5, 1.5)
        self.spin_curvature.setDecimals(4)
        self.spin_curvature.setSingleStep(0.005)
        self.spin_curvature.setSuffix(" rad")
        self.spin_curvature.setKeyboardTracking(False)
        self.spin_curvature.setToolTip(
            "Sample curvature shear angle γ (rad) applied inside the signal ROI (and BKG ROI) per spin state. "
            "Synced with Scan Charts ▸ Reduction."
        )
        self.spin_curvature.valueChanged.connect(self._on_curvature_spin_changed)

        self.btn_reset_curvature = QPushButton("Reset 0")
        self.btn_reset_curvature.setToolTip("Reset sample curvature to 0 rad (restores uncorrected baseline)")
        self.btn_reset_curvature.clicked.connect(self._reset_curvature)

        self.slider_curvature = QSlider(Qt.Orientation.Horizontal)
        self.slider_curvature.setMinimumWidth(0)
        self.slider_curvature.setRange(-300, 300)
        self.slider_curvature.setValue(0)
        self.slider_curvature.setToolTip("Fine curvature slider (−0.300 to +0.300 rad)")
        self.slider_curvature.valueChanged.connect(self._on_curvature_slider_changed)

        grid.addWidget(QLabel("γ:"), 0, 0)
        grid.addWidget(self.spin_curvature, 0, 1)
        grid.addWidget(self.btn_reset_curvature, 0, 2)
        grid.addWidget(self.slider_curvature, 1, 0, 1, 3)
        grid.setColumnStretch(1, 1)
        return group

    def _sync_curvature_controls(self) -> None:
        curv = float(self.scan_data.reduction_params.curvature_rad) if self.scan_data is not None else 0.0
        self.spin_curvature.blockSignals(True)
        self.slider_curvature.blockSignals(True)
        self.spin_curvature.setValue(curv)
        self.slider_curvature.setValue(int(round(max(-300.0, min(300.0, curv * 1000.0)))))
        self.spin_curvature.blockSignals(False)
        self.slider_curvature.blockSignals(False)

    def _apply_curvature_value(self, value: float) -> None:
        if self.scan_data is None:
            return
        curv = float(value)
        if self.scan_data.reduction_params.curvature_rad == curv:
            return
        self.scan_data.reduction_params.curvature_rad = curv
        self._sync_curvature_controls()
        self._preserve_levels = True
        try:
            self._redraw()
        finally:
            self._preserve_levels = False
        self.curvature_changed.emit(self.scan_data.file_path)

    def _on_curvature_spin_changed(self, value: float) -> None:
        self._apply_curvature_value(value)

    def _on_curvature_slider_changed(self, tick: int) -> None:
        self._apply_curvature_value(tick / 1000.0)

    def _reset_curvature(self) -> None:
        self._apply_curvature_value(0.0)

    def refresh_reduction(self) -> None:
        """Sync curvature & BKG ROI display after changes in Scan Charts or Detector Image."""
        if self.scan_data is None:
            return
        self._sync_curvature_controls()
        if self._map is not None:
            self._redraw()
        else:
            self._draw_roi_overlay()

    def _detector_nx(self) -> int:
        if self._map is not None:
            return int(self._map.counts.shape[1])
        if self.scan_data is not None and self.scan_data.detector_shape:
            return int(self.scan_data.detector_shape[1])
        return 1024

    def _pixel_pitch(self, width_mm: float = DETECTOR_WIDTH_MM) -> float | None:
        """Displayed-X pixel spacing (mm) from the detector width, or None without detector data."""
        if self._map is not None:
            nx = self._map.counts.shape[1]
        elif self.scan_data is not None and self.scan_data.detector_shape:
            nx = self.scan_data.detector_shape[1]
        else:
            return None
        return pixel_pitch_mm(nx, width_mm)

    def _show_calibration(self, cal: AngleCalibration | None) -> None:
        """Status tooltip and distance / effective-scale readouts for this scan's calibration."""
        if cal is None:
            self.lbl_distance.setText("—")
            self.lbl_effective_scale.setText("—")
            return
        width = cal.detector_width_mm if cal.mode == "pysared" else DETECTOR_WIDTH_MM
        pitch = self._pixel_pitch(width)
        if pitch is None:
            self.lbl_distance.setText("—")
            self.lbl_effective_scale.setText("—")
            return
        nx = self._detector_nx()
        if cal.mode == "pysared":
            self.lbl_effective_scale.setText(
                f"{cal.deg_per_pixel:.5f} °/px  ({cal.detector_width_mm:g} mm / {nx} px)"
            )
            self.lbl_distance.setText(f"{cal.sample_detector_distance_mm:.0f} mm  (pySAred geometry)")
        else:
            self.lbl_distance.setText(f"{cal.distance_mm(pitch):.0f} mm  (from {DETECTOR_WIDTH_MM:g} mm / {nx} px)")
            self.lbl_effective_scale.setText(f"{cal.deg_per_pixel:.5f} °/px")
        self._set_cal_status(cal.describe(pitch))

    def _set_cal_status(self, text: str) -> None:
        """Calibration details stay available on hover without cluttering the panel."""
        self.cal_status = text
        self.cal_group.setToolTip(text)

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self.combo_spin, self.combo_y_source, self.combo_norm, self.combo_y_axis,
                       self.btn_export_png):
            widget.setEnabled(enabled)
        for widget in (self.spin_y0, self.spin_y1, self.btn_crosshair, self.btn_drop_cuts,
                       self.btn_cut_h, self.btn_cut_v, self.chk_roi_sum_cut,
                       self.btn_mode_direct_beam, self.btn_mode_pysared,
                       self.spin_dpp, self.spin_distance_mm, self.spin_det_width_mm, self.spin_x0,
                       self.chk_af_increases, self.btn_fit_db, self.chk_show_db, self.spin_wavelength,
                       self.spin_curvature, self.slider_curvature, self.btn_reset_curvature,
                       self.chk_smooth_map):
            widget.setEnabled(enabled)
        self.spin_smooth_sigma.setEnabled(enabled and self.chk_smooth_map.isChecked())

    # -------------------------------------------------------------- public API
    def set_data(self, scan_data: ScanData | None) -> None:
        """Show the active scan; the map is computed lazily when the tab is visible."""
        self._cancel_jobs()
        self.scan_data = scan_data
        self._map = None
        self._norm = None
        self._last_level_key = None
        self._pending_key = None
        self._crosshair_placed = False
        self._set_cut_buttons(False, False)
        for item in (self.crosshair, self.line_row, self.line_x):
            item.setVisible(False)
        self.img_item.clear()
        self._clear_alpha_items()
        self._clear_roi_overlay()
        self._clear_cuts()
        self._cal_message = ""
        self._sync_cal_controls()
        self._sync_curvature_controls()
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
        self._last_tick_key = None
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
    def _detector_ny_full(self) -> int:
        if self.scan_data is not None and self.scan_data.detector_shape:
            return int(self.scan_data.detector_shape[2])
        if self._map is not None:
            return max(1, int(self._map.y_bounds[1]))
        return 1024

    def _compute_norm_map(
        self, counts: np.ndarray, mode: str, spin: str, flipper, exposure, monitor,
    ):
        kw = dict(
            exposure=exposure,
            monitor=None if monitor is None else np.asarray(monitor.data, dtype=float),
            monitor_variance=None if monitor is None else monitor.variance,
        )
        if spin in ("asym", "sum"):
            th, tth = self._variable("th"), self._variable("tth")
            label_name = "Spin asymmetry" if spin == "asym" else "Sum of channels"
            if flipper is None or th is None or tth is None:
                return None, f"{label_name} needs Flipper1, θ and 2θ channels."
            up, down = pair_spin_frames(th, tth, flipper, self._map.written)
            if up.size == 0:
                return None, f"{label_name}: no u/d frame pairs at matching θ and 2θ."
            if spin == "asym":
                return spin_asymmetry(counts, up, down, mode,
                                      min_counts=self.spin_min_counts.value(), **kw), ""
            return sum_spin_frames(counts, up, down, mode, **kw), ""
        frames = select_frames(self._map.n_frames, flipper, spin)
        return normalize_map(counts, frames, mode, **kw), ""

    def _levels_from_norm(self, norm) -> tuple[float, float]:
        if isinstance(norm, AsymmetryMap):
            disp = np.where(norm.low_count_mask, np.nan, norm.values)
            finite = np.abs(disp[np.isfinite(disp)])
            half = float(np.percentile(finite, 99)) if finite.size else 1.0
            return -(half or 1.0), (half or 1.0)
        disp = log_display(norm.values) if self._log_active() else norm.values
        finite = disp[np.isfinite(disp)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _redraw(self) -> None:
        if self._map is None or self.scan_data is None:
            return
        mode = self.combo_norm.currentData()
        spin = self.combo_spin.currentData()
        flipper = self._variable("Flipper1")
        exposure = None
        count = self.scan_data.variables.get("Count")
        monitor = self.scan_data.variables.get("Monitor1")

        # Non-destructive per-spin sample curvature shear inside Signal (and disjoint BKG) ROI
        effective_counts = self._map.counts
        curv_warning = ""
        params = self.scan_data.reduction_params
        curv_active = abs(params.curvature_rad) > 0.0
        if curv_active:
            bounds = self._scan_roi_bounds()
            if len(bounds) > 1:
                curv_warning = "Sample curvature skipped: requires a fixed ROI (tracked ROI active)."
            else:
                ny_full = self._detector_ny_full()
                sx0, sx1, _, _ = pixel_bounds(bounds[0], self._map.nx, ny_full)
                bkg_win = None
                if params.subtract_background:
                    bx0, bx1, _, _ = resolve_bkg_roi(
                        bounds[0], params.bkg_roi_bounds, params.bkg_x_bounds, self._map.nx, ny_full
                    )
                    if bx1 <= sx0 or sx1 <= bx0:
                        bkg_win = (bx0, bx1)
                try:
                    effective_counts = apply_curvature_by_spin(
                        self._map.counts,
                        flipper,
                        self._map.written,
                        params.curvature_rad,
                        (sx0, sx1),
                        bkg_x_bounds=bkg_win,
                    )
                except ValueError as err:
                    curv_warning = f"Sample curvature skipped: {err}"

        try:
            if mode == "time":
                # Same exposure rule as the Scan Charts: seconds only, no raw fallback.
                if count is None or count.unit != "s":
                    raise ValueError("Counts / s needs a Count exposure channel in seconds")
                exposure = np.asarray(count.data, dtype=float)
            norm, unavailable = self._compute_norm_map(effective_counts, mode, spin, flipper, exposure, monitor)
            if unavailable:
                self._show_unavailable(unavailable)
                return
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
        level_key = (
            self.scan_data.file_path,
            self._map.y_bounds,
            spin,
            mode,
            self._log_active(),
            asym,
            self.spin_min_counts.value() if asym else 0,
        )

        # Anchor automatic colour-bar levels to the uncurved (gamma = 0), unsmoothed baseline so
        # fractional counts from curvature shearing or display smoothing never destabilize the palette.
        if not curv_active:
            base_lo, base_hi = self._levels_from_norm(norm)
            self._baseline_levels[level_key] = (base_lo, base_hi)
        elif level_key in self._baseline_levels:
            base_lo, base_hi = self._baseline_levels[level_key]
        else:
            try:
                base_norm, _ = self._compute_norm_map(self._map.counts, mode, spin, flipper, exposure, monitor)
                base_lo, base_hi = self._levels_from_norm(base_norm) if base_norm is not None else self._levels_from_norm(norm)
            except ValueError:
                base_lo, base_hi = self._levels_from_norm(norm)
            self._baseline_levels[level_key] = (base_lo, base_hi)

        if self._preserve_levels and self._last_level_key == level_key:
            levels_attr = getattr(self.colorbar, "levels", None)
            cur_levels = levels_attr() if callable(levels_attr) else levels_attr
            if (cur_levels is not None and len(cur_levels) == 2
                    and np.isfinite(cur_levels[0]) and np.isfinite(cur_levels[1])
                    and float(cur_levels[1]) > float(cur_levels[0])):
                lo, hi = float(cur_levels[0]), float(cur_levels[1])
            else:
                lo, hi = base_lo, base_hi
        else:
            lo, hi = base_lo, base_hi
        self._last_level_key = level_key

        if asym:
            # Display only: low-count pixels blanked; colour range symmetric about zero.
            unsmoothed_linear = np.where(norm.low_count_mask, np.nan, norm.values)
            self._cut_values = unsmoothed_linear
        else:
            unsmoothed_linear = norm.values
            self._cut_values = norm.values

        display = unsmoothed_linear if asym else (log_display(unsmoothed_linear) if self._log_active() else unsmoothed_linear)
        alpha_note = ""
        if self._alpha_mode():
            alpha_note = self._draw_alpha(display, levels=(lo, hi), asym=asym)
            if alpha_note is None:
                return
        elif self._theta_mode():
            alpha_note = self._draw_theta(display)
            if alpha_note is None:
                return
        else:
            self._clear_alpha_items()
            self.img_item.setImage(display.T, autoLevels=False)
            self.img_item.setRect(0, 0, self._map.nx, len(frames))
            if self._colorbar_on_strips:  # re-link only when coming back from mesh/strip views
                self.colorbar.setImageItem(self.img_item)
                self._colorbar_on_strips = False
                self._colorbar_items_rebuilt = True
        cmap = self._current_colormap()
        cmap_key = (asym, "__rdbu_r__" if asym else self.combo_cmap.currentText())
        self.colorbar.setLevels((lo, hi))
        if self._colorbar_items_rebuilt or cmap_key != self._applied_cmap_key:
            self.colorbar.setColorMap(cmap)
            self._applied_cmap_key = cmap_key
            self._colorbar_items_rebuilt = False
        label = f"log₁₀({norm.units})" if self._log_active() else norm.units
        axis = getattr(self.colorbar, "axis", None)
        if axis is not None:
            axis.setLabel(label)

        notes = []
        if asym and mode == "raw":
            notes.append("Raw-count asymmetry assumes equal u/d exposure; Counts / Monitor1 corrects for this.")
        if curv_warning:
            notes.append(curv_warning)
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
        elif self.btn_cut_h.isChecked() or self.btn_cut_v.isChecked():
            self._place_crosshair(*self._default_cut_pos())
        elif self.chk_roi_sum_cut.isChecked():
            self._update_roi_sum_cut()

    def _apply_no_data_background(self) -> None:
        # Neutral slate behind the map: blank (log-omitted / excluded) pixels stay
        # visibly distinct from any colormap value instead of being recoloured.
        self.plot.getViewBox().setBackgroundColor("#334155")

    def _show_unavailable(self, message: str) -> None:
        self._norm = None
        self.img_item.clear()
        self._clear_alpha_items()
        self._clear_roi_overlay()
        for item in (self.crosshair, self.line_row, self.line_x):
            item.setVisible(False)
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
            return _diverging_rdbu_r()
        name = self.combo_cmap.currentText()
        cmap = _CMAP_CACHE.get(name)
        if cmap is None:
            cmap = get_safe_colormap(name)
            _CMAP_CACHE[name] = cmap
        return cmap

    def _on_spin_changed(self) -> None:
        asym = self._asym_mode()
        self.spin_min_counts.setVisible(asym)
        self.chk_log.setEnabled(not asym)
        self.combo_cmap.setEnabled(not asym)
        self._redraw()

    def _on_cmap_changed(self, name: str) -> None:
        self.colorbar.setColorMap(self._current_colormap())
        asym = self._asym_mode()
        self._applied_cmap_key = (asym, "__rdbu_r__" if asym else self.combo_cmap.currentText())

    def _row_label(self, row: int) -> str:
        frame = int(self._norm.frame_indices[row])
        return str(frame)

    def _update_y_ticks(self) -> None:
        axis = self.plot.getPlotItem().getAxis("left")
        col_axis = self.plot_col.getPlotItem().getAxis("bottom")
        if self._norm is None or self._norm.frame_indices.size == 0:
            if self._last_tick_key is not None:
                axis.setTicks(None)
                col_axis.setTicks(None)
                self._last_tick_key = None
            return
        mode = self.combo_y_axis.currentData()
        frames = self._norm.frame_indices
        if mode == "point":
            step = max(1, int(np.ceil(frames.size / 12)))
            tick_sig = tuple(int(frames[r]) for r in range(0, frames.size, step))
            tick_key = (mode, frames.size, tick_sig, self.is_dark_mode)
        else:
            tick_key = (mode, self.is_dark_mode)
        if tick_key == self._last_tick_key:
            return
        self._last_tick_key = tick_key
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
        if self._theta_mode():
            axis.setTicks(None)
            col_axis.setTicks(None)
            self.plot.setLabel("bottom", "Detector X pixel")
            self.plot.setLabel("left", "θ (°)")
            self.plot_row.setLabel("bottom", "Detector X pixel")
            self.plot_col.setLabel("bottom", "θ (°)")
            return
        self.plot.setLabel("bottom", "Detector X pixel")
        self.plot_row.setLabel("bottom", "Detector X pixel")
        n = frames.size
        self.plot.setLabel("left", "Point (frame index)")
        major = [(row + 0.5, self._row_label(row)) for row in range(0, n, step)]
        minor = [(row + 0.5, "") for row in range(n) if row % step]
        axis.setTicks([major, minor])
        col_axis.setTicks([major, minor])
        self.plot_col.setLabel("bottom", "Point (frame index)")

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
        if self._theta_mode():
            self._draw_theta_overlay()
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
        # Dashed cyan Background ROI boundaries when BKG subtraction is active (pySAred 1.5.3)
        params = self.scan_data.reduction_params
        if params.subtract_background:
            bx0, bx1, _, _ = resolve_bkg_roi(
                bounds[0], params.bkg_roi_bounds, params.bkg_x_bounds, self._map.nx, self._detector_ny_full()
            )
            for xb in (float(bx0), float(bx1)):
                self._add_overlay([xb, xb], [0.0, float(frames.size)], "#06b6d4")

    def _draw_theta_overlay(self) -> None:
        g = self._th_geom
        if g is None:
            return
        rows, theta, strips = g["rows"], g["theta"], g["strips"]
        sel_frames = self._norm.frame_indices[rows]
        cal = getattr(self.scan_data, "angle_calibration", None)
        tth = self._variable("tth")
        if self.chk_show_db.isChecked() and cal is not None and cal.fit is not None and tth is not None:
            ok = sel_frames < tth.size
            xs = cal.fit.position_at(tth[sel_frames[ok]])
            ys = theta[ok]
            order = np.argsort(ys, kind="stable")
            xs, ys = xs[order], ys[order]
            inside = (xs >= 0) & (xs <= self._map.nx)
            self._add_overlay(xs[inside], ys[inside], "#e879f9", connect="finite")
        if not self.chk_show_roi.isChecked():
            return
        bounds = self._scan_roi_bounds()
        y0, y1 = float(strips[:, 0].min()), float(strips[:, 1].max())
        if len(bounds) == 1:
            x_lo, x_hi = min(bounds[0][0], bounds[0][1]), max(bounds[0][0], bounds[0][1])
            for xb in (x_lo, x_hi):
                self._add_overlay([xb, xb], [y0, y1], "#facc15")
        else:
            per_row = np.array([bounds[min(int(f), len(bounds) - 1)] for f in sel_frames], dtype=float)
            ys = strips.reshape(-1)
            for col in (0, 1):
                xb = np.minimum(per_row[:, 0], per_row[:, 1]) if col == 0 else np.maximum(per_row[:, 0], per_row[:, 1])
                self._add_overlay(np.repeat(xb, 2), ys, "#facc15")
        params = self.scan_data.reduction_params
        if params.subtract_background:
            bx0, bx1, _, _ = resolve_bkg_roi(
                bounds[0], params.bkg_roi_bounds, params.bkg_x_bounds, self._map.nx, self._detector_ny_full()
            )
            for xb in (float(bx0), float(bx1)):
                self._add_overlay([xb, xb], [y0, y1], "#06b6d4")

    # ------------------------------------------------------------- crosshair
    def _map_ready(self) -> bool:
        ready = self._norm is not None and self._map is not None and self._norm.frame_indices.size > 0
        if not ready:
            return False
        if self._alpha_mode():
            return self._alpha_geom is not None
        if self._theta_mode():
            return self._th_geom is not None
        return True

    def _cut_line_flags(self) -> tuple[bool, bool]:
        """(show_horizontal_line, show_vertical_line) on the 2D map."""
        return self.btn_cut_h.isChecked(), self.btn_cut_v.isChecked()

    def _cut_curve_flags(self) -> tuple[bool, bool]:
        """(show_row_plot_curve, show_col_plot_curve) matching the active cut lines."""
        show_h, show_v = self._cut_line_flags()
        if self._alpha_mode() and not self._q_mode():
            # In α_i vs α_f: vertical line (x = α_i) drives plot_row (vs α_f);
            # horizontal line (y = α_f) drives plot_col (vs α_i).
            return show_v, show_h
        return show_h, show_v

    def _set_cut_buttons(self, h_on: bool, v_on: bool) -> None:
        for btn, state in (
            (self.btn_cut_h, h_on),
            (self.btn_cut_v, v_on),
            (self.btn_drop_cuts, h_on and v_on),
            (self.btn_crosshair, h_on or v_on),
        ):
            btn.blockSignals(True)
            btn.setChecked(state)
            btn.blockSignals(False)

    def _sync_cut_visibility(self) -> None:
        show_h, show_v = self._cut_line_flags()
        placed = self._crosshair_placed and self._map_ready()
        self.line_row.setVisible(placed and show_h)
        self.line_x.setVisible(placed and show_v)
        self.crosshair.setVisible(placed and show_h and show_v)

    def _update_cut_line_styles(self) -> None:
        """Colour-match each cut line on the map to the 1D cut plot it drives."""
        row_color = "#38bdf8"
        col_color = "#f43f5e"
        if self._alpha_mode() and not self._q_mode():
            h_color, v_color = col_color, row_color
        else:
            h_color, v_color = row_color, col_color
        self.line_row.setPen(pg.mkPen(h_color, width=1.5, style=Qt.PenStyle.DashLine))
        self.line_x.setPen(pg.mkPen(v_color, width=1.5, style=Qt.PenStyle.DashLine))

    def _default_cut_pos(self) -> tuple[float, float]:
        """Useful default drop position: specular x0 (or ROI centre) at mid-scan row."""
        if self._alpha_mode() and self._alpha_geom is not None:
            rows = self._alpha_geom["rows"]
            row = int(rows[rows.size // 2])
        elif self._theta_mode() and self._th_geom is not None:
            rows = self._th_geom["rows"]
            row = int(rows[rows.size // 2])
        else:
            n = self._norm.frame_indices.size if self._norm is not None else 1
            row = max(0, n // 2)
        cal = getattr(self.scan_data, "angle_calibration", None) if self.scan_data is not None else None
        nx = self._map.nx if self._map is not None else 1
        if cal is not None and 0.0 <= cal.x0 < nx:
            x = int(min(max(round(cal.x0), 0), nx - 1))
        else:
            bounds = self._scan_roi_bounds()
            x_mid = 0.5 * (bounds[0][0] + bounds[0][1]) if bounds else nx / 2.0
            x = int(min(max(round(x_mid), 0), nx - 1))
        return self._cell_center(row, x)

    def _apply_cut_selection(self, h_on: bool, v_on: bool) -> None:
        self._set_cut_buttons(h_on, v_on)
        if not (h_on or v_on):
            self._crosshair_placed = False
            for item in (self.crosshair, self.line_row, self.line_x):
                item.setVisible(False)
            self._clear_cuts()
            return
        if not self._map_ready():
            self.lbl_cut_hint.setText("Click the map to drop the cut lines once loaded.")
            return
        if not self._crosshair_placed:
            self._place_crosshair(*self._default_cut_pos())
        else:
            self._sync_cut_visibility()
            self._update_cut_line_styles()
            self._update_cuts()

    def _on_crosshair_toggled(self, checked: bool) -> None:
        self._apply_cut_selection(checked, checked)

    def _on_drop_cuts_toggled(self, checked: bool) -> None:
        self._apply_cut_selection(checked, checked)

    def _on_single_cut_toggled(self, _checked: bool) -> None:
        self._apply_cut_selection(self.btn_cut_h.isChecked(), self.btn_cut_v.isChecked())

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
        if not (self.btn_cut_h.isChecked() or self.btn_cut_v.isChecked()):
            self._set_cut_buttons(True, True)
        self._crosshair_placed = True
        self._sync_cut_visibility()
        self._update_cut_line_styles()
        self.crosshair.blockSignals(True)
        self.crosshair.setPos((x, y))
        self.crosshair.blockSignals(False)
        self._on_crosshair_moved(force=True)

    def _on_line_row_moved(self) -> None:
        if not self._crosshair_placed or not self._map_ready():
            return
        x = float(self.crosshair.pos().x())
        y = float(self.line_row.value())
        self.crosshair.blockSignals(True)
        self.crosshair.setPos((x, y))
        self.crosshair.blockSignals(False)
        self._on_crosshair_moved()

    def _on_line_x_moved(self) -> None:
        if not self._crosshair_placed or not self._map_ready():
            return
        x = float(self.line_x.value())
        y = float(self.crosshair.pos().y())
        self.crosshair.blockSignals(True)
        self.crosshair.setPos((x, y))
        self.crosshair.blockSignals(False)
        self._on_crosshair_moved()

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
        self.line_x.blockSignals(True)
        self.line_x.setValue(x)
        self.line_x.blockSignals(False)
        self.line_row.blockSignals(True)
        self.line_row.setValue(y)
        self.line_row.blockSignals(False)
        if self._alpha_mode():
            self._cross_alpha = (x, y)
            row, col = self._alpha_cut_cell(x, y)
            self._sel_row, self._sel_x = row, col
            self._update_cuts()
            return
        if self._theta_mode():
            g = self._th_geom
            col = int(np.clip(np.floor(x), 0, self._map.nx - 1))
            dist = np.abs(g["theta"] - y)
            k = int(dist.size - 1 - np.argmin(dist[::-1]))
            row = int(g["rows"][k])
            if force or (row, col) != (self._sel_row, self._sel_x):
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
            if self._last_level_key in self._baseline_levels:
                base_lo, _ = self._baseline_levels[self._last_level_key]
                lo = max(lo, float(base_lo))
                if hi <= lo:
                    hi = lo + 1.0
        pad = 0.05 * (hi - lo) if hi > lo else 0.5
        if self._alpha_mode():
            x0, x1, y0, y1 = self._map_extent()
            # α view: row cut runs along α_f, column cut along α_i. Q view: constant-Qz cut
            # runs along Qx, constant-Qx cut along Qz.
            spans = (((self.plot_row, x0, x1), (self.plot_col, y0, y1)) if self._q_mode()
                     else ((self.plot_row, y0, y1), (self.plot_col, x0, x1)))
        elif self._theta_mode():
            x0, x1, y0, y1 = self._map_extent()
            spans = ((self.plot_row, x0, x1), (self.plot_col, y0, y1))
        else:
            spans = ((self.plot_row, 0, self._map.nx), (self.plot_col, 0, self._norm.frame_indices.size))
        for plot, start, stop in spans:
            plot.setLogMode(y=log)
            plot.setLabel("left", self._norm.units)
            plot.setXRange(start, stop, padding=0)
            plot.setYRange(lo - pad, hi + pad, padding=0)
            plot.getViewBox().disableAutoRange()

    def _on_roi_sum_cut_toggled(self, checked: bool) -> None:
        if checked:
            self._update_roi_sum_cut()
        else:
            self._update_cuts()

    def _update_roi_sum_cut(self) -> None:
        """Plot X-integrated intensity across the Signal ROI in plot_col (live curvature feedback)."""
        if not self._map_ready() or self._cut_values is None:
            self._col_curve.setData([], [])
            self._col_curve.setVisible(False)
            return
        bounds = self._scan_roi_bounds()
        sx0, sx1, _, _ = pixel_bounds(bounds[0], self._map.nx, max(1, self._map.y_bounds[1]))
        if sx1 <= sx0:
            self._col_curve.setData([], [])
            self._col_curve.setVisible(False)
            return
        roi_sum = np.nansum(self._cut_values[:, sx0:sx1], axis=1)
        log = self._log_active()
        vals = np.where(roi_sum > 0, roi_sum, np.nan) if log else roi_sum
        if self._theta_mode() and self._th_geom is not None:
            g = self._th_geom
            order = np.argsort(g["theta"], kind="stable")
            xs = g["theta"][order]
            ys = vals[g["rows"][order]]
        elif self._alpha_mode() and self._alpha_geom is not None:
            g = self._alpha_geom
            order = np.argsort(g["alpha_i"], kind="stable")
            xs = g["alpha_i"][order]
            ys = vals[g["rows"][order]]
        else:
            xs = np.arange(vals.size, dtype=float) + 0.5
            ys = vals
        self._col_curve.setPen(pg.mkPen("#f59e0b", width=2))
        self._col_curve.setData(xs, ys)
        self._col_curve.setVisible(True)
        self.plot_col.setLogMode(y=log)
        self.plot_col.getViewBox().enableAutoRange()
        curv = self.scan_data.reduction_params.curvature_rad if self.scan_data is not None else 0.0
        self.lbl_col_cut.setText(f"ROI-summed profile (X [{sx0}..{sx1}], γ = {curv:+.4f} rad):")

    def _update_cuts(self) -> None:
        if not self._crosshair_placed or not self._map_ready():
            self._clear_cuts()
            if self.chk_roi_sum_cut.isChecked() and self._map_ready():
                self._update_roi_sum_cut()
            return
        row_active, col_active = self._cut_curve_flags()
        if not (row_active or col_active):
            self._clear_cuts()
            if self.chk_roi_sum_cut.isChecked():
                self._update_roi_sum_cut()
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
            if self.chk_roi_sum_cut.isChecked():
                self._update_roi_sum_cut()
            return
        frame = int(self._norm.frame_indices[row])
        if row_active:
            self._row_curve.setData(np.arange(self._map.nx) + 0.5, positive(self._cut_values[row]))
            self._row_curve.setVisible(True)
            self.lbl_row_cut.setText(f"Horizontal cut — point {frame} ({self._describe_frame(frame)}):")
        else:
            self._row_curve.setData([], [])
            self._row_curve.setVisible(False)
            self.lbl_row_cut.setText("Horizontal cut (toggle Horizontal above):")
        if self.chk_roi_sum_cut.isChecked():
            self._update_roi_sum_cut()
        elif col_active:
            if self._theta_mode():
                g = self._th_geom
                order = np.argsort(g["theta"], kind="stable")
                self._col_curve.setData(g["theta"][order], positive(self._cut_values[g["rows"][order], x]))
            else:
                col_vals = self._cut_values[:, x]
                self._col_curve.setData(np.arange(col_vals.size) + 0.5, positive(col_vals))
            self._col_curve.setVisible(True)
            self.lbl_col_cut.setText(f"Vertical cut — X pixel {x}:")
        else:
            self._col_curve.setData([], [])
            self._col_curve.setVisible(False)
            self.lbl_col_cut.setText("Vertical cut (toggle Vertical above):")
        self.lbl_cut_hint.setText("Drag the cut lines or crosshair on the map to move cuts; click to jump.")

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
        if isinstance(self._norm, SummedSpinMap):
            parts.append("u+d")
        elif isinstance(self._norm, AsymmetryMap):
            parts.append("u/d")
        else:
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
                dq = qz_step_per_pixel(cal.deg_per_pixel, lam)
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
        if axes == "th":
            if self._th_geom is None:
                raise ValueError(self.lbl_notice.text() or "The θ view is not available for these settings")
            rows = self._th_geom["rows"]
            coords["Theta (deg)"] = self._th_geom["theta"]
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
        summed = isinstance(norm, SummedSpinMap)
        paired = asym or summed
        columns, data = map_table(frames[rows], norm.values[rows], norm.standard_uncertainty[rows], coords,
                                  partner_indices=norm.partner_indices[rows] if paired else None)
        y0, y1 = omap.y_bounds
        spin_note = ""
        if asym:
            spin_note = " (Frame = u frame, PartnerFrame = paired d frame)"
        elif summed:
            spin_note = " (Frame = u frame, PartnerFrame = paired d frame; pooled u + d exposure)"
        header = [
            f"Off-specular map: {self.combo_y_axis.currentText()}; detector frames integrated along Y, "
            "displayed X pixel (90 deg CW + mirrored, as in the Detector Image tab).",
            f"Source: {self.scan_data.file_path}",
            f"Spin: {self.combo_spin.currentText()}" + spin_note,
            f"Integrated Y rows: {'per-frame tracked ROI within ' if omap.y_varies else ''}[{y0}, {y1})",
            f"Normalization: {self.combo_norm.currentText()}; Value units: {norm.units}",
            "StdUncertainty is first-order 1 sigma (docs/normalization.md, docs/off_specular.md).",
            "Frame is the 0-based detector frame (scan point) index.",
        ]
        if self._alpha_mode():
            cal = self._alpha_geom["cal"]
            if cal.mode == "pysared":
                pitch_ps = pixel_pitch_mm(omap.nx, cal.detector_width_mm)
                header.append(
                    f"Angle calibration: pySAred geometry (L = {cal.sample_detector_distance_mm!r} mm, "
                    f"detector width = {cal.detector_width_mm!r} mm, pixel = {pitch_ps!r} mm, "
                    f"effective {cal.deg_per_pixel!r} deg/pixel at x0), "
                    f"x0 = {cal.x0!r} px, alpha_f {'increases' if cal.sign > 0 else 'decreases'} with X; "
                    "coordinates are pixel centres"
                )
            else:
                header.append(f"Angle calibration: {cal.deg_per_pixel!r} deg/pixel ({cal.deg_per_pixel_source}), "
                              f"x0 = {cal.x0!r} px, alpha_f {'increases' if cal.sign > 0 else 'decreases'} with X; "
                              "coordinates are pixel centres")
                pitch = self._pixel_pitch()
                if pitch is not None:
                    header.append(f"Implied sample-detector distance: {cal.distance_mm(pitch):.1f} mm "
                                  f"(pixel {pitch!r} mm = {DETECTOR_WIDTH_MM:g} mm detector width; "
                                  "derived from deg/pixel, not used in the conversion)")
            if self._q_mode():
                header.append(f"Wavelength: {self.spin_wavelength.value()!r} Angstrom; "
                              "Qx = (2pi/lambda)(cos af - cos ai), Qz = (2pi/lambda)(sin af + sin ai)")
        if asym:
            header.append(f"Display blanks pixels with u+d < {norm.min_counts:g} raw counts; "
                          "exported values are not blanked.")
        curv = self.scan_data.reduction_params.curvature_rad
        if abs(curv) > 0.0:
            header.append(f"Sample curvature correction (pySAred 1.5.3): gamma = {curv!r} rad (applied inside ROI per spin state).")
        if self._log_active():
            header.append("Log colour scale is display-only; values are linear.")
        if self._alpha_mode() and self.chk_smooth_map.isChecked():
            sigma = self.spin_smooth_sigma.value()
            desc = (
                f"bilinear re-binning + edge-preserving bilateral (sigma = {sigma:g} px)"
                if sigma > 0.0
                else "bilinear re-binning (sigma = 0 px)"
            )
            header.append(f"Smooth map display ({desc}) is display-only; exported values are unsmoothed.")
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

    # ------------------------------------------------------------- α_i / α_f & θ
    def _theta_mode(self) -> bool:
        return self.combo_y_axis.currentData() == "th"

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
        qx = qx_view / QX_SCALE
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

        row_active, col_active = self._cut_curve_flags()
        if row_active:
            af_const_qz = constant_qz_alpha_f(g["alpha_i"], qz, lam)
            q_rows, v_rows = sample(af_const_qz)
            order = np.argsort(q_rows[:, 0])
            self._row_curve.setData(q_rows[order, 0] * QX_SCALE, positive(v_rows[order]))
            self._row_curve.setVisible(True)
            self.lbl_row_cut.setText(f"Cut at constant Qz = {qz:.4f} Å⁻¹ ({v_rows.size} points):")
        else:
            self._row_curve.setData([], [])
            self._row_curve.setVisible(False)
            self.lbl_row_cut.setText("Horizontal cut at constant Qz (toggle Horizontal above):")
        if col_active:
            af_const_qx = constant_qx_alpha_f(g["alpha_i"], qx, lam, side=side)
            q_cols, v_cols = sample(af_const_qx)
            order = np.argsort(q_cols[:, 1])
            self._col_curve.setData(q_cols[order, 1], positive(v_cols[order]))
            self._col_curve.setVisible(True)
            self.lbl_col_cut.setText(f"Cut at constant Qx = {qx_view:.3g}×10⁻³ Å⁻¹ ({v_cols.size} points):")
        else:
            self._col_curve.setData([], [])
            self._col_curve.setVisible(False)
            self.lbl_col_cut.setText("Vertical cut at constant Qx (toggle Vertical above):")
        self.lbl_cut_hint.setText("Drag the cut lines or crosshair on the map to move cuts; click to jump.")

    def _on_smooth_changed(self, *_args) -> None:
        self.spin_smooth_sigma.setEnabled(self.chk_smooth_map.isEnabled() and self.chk_smooth_map.isChecked())
        if self._alpha_mode():
            self._preserve_levels = True
            try:
                self._redraw()
            finally:
                self._preserve_levels = False

    def _on_axes_changed(self) -> None:
        self._smooth_group.setVisible(self._alpha_mode())
        if self._display_bar.layout() is not None:
            self._display_bar.layout().invalidate()
            self._display_bar.layout().activate()
        self._display_bar.updateGeometry()
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
            scan.angle_calibration = calibrate(
                self._map.counts,
                tth,
                self._roi_x_ranges(),
                y_rows=self._map.y_bounds,
                written=self._map.written,
            )
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
        widgets = (
            self.btn_mode_direct_beam,
            self.btn_mode_pysared,
            self.spin_dpp,
            self.spin_distance_mm,
            self.spin_det_width_mm,
            self.spin_x0,
            self.chk_af_increases,
        )
        for widget in widgets:
            widget.blockSignals(True)
        if self.scan_data is not None and self.scan_data.detector_shape:
            self.spin_x0.setMaximum(float(self.scan_data.detector_shape[1]))
        if cal is not None:
            is_pysared = cal.mode == "pysared"
            self.btn_mode_pysared.setChecked(is_pysared)
            self.btn_mode_direct_beam.setChecked(not is_pysared)
            self.cal_mode_stack.setCurrentIndex(1 if is_pysared else 0)
            self.spin_dpp.setValue(cal.deg_per_pixel)
            self.spin_distance_mm.setValue(cal.sample_detector_distance_mm)
            self.spin_det_width_mm.setValue(cal.detector_width_mm)
            self.spin_x0.setValue(cal.x0)
            self.chk_af_increases.setChecked(cal.sign > 0)
            self._show_calibration(cal)
        else:
            self.btn_mode_direct_beam.setChecked(True)
            self.btn_mode_pysared.setChecked(False)
            self.cal_mode_stack.setCurrentIndex(0)
            self._show_calibration(None)
            self.spin_dpp.setValue(0.0)
            self.spin_distance_mm.setValue(PYSARED_DEFAULT_DISTANCE_MM)
            self.spin_det_width_mm.setValue(DETECTOR_WIDTH_MM)
            if self._map is not None:
                try:
                    self.spin_x0.setValue(
                        specular_center(
                            self._map.counts,
                            self._roi_x_ranges(),
                            tth=self._variable("tth"),
                            written=self._map.written,
                        )
                    )
                except CalibrationError:
                    self.spin_x0.setValue(float(np.mean(self._roi_x_ranges()[0])))
            self.chk_af_increases.setChecked(False)
            self._set_cal_status(self._cal_message or
                                 "Calibrated automatically from the transmitted beam when the map is computed.")
        for widget in widgets:
            widget.blockSignals(False)

    def _on_cal_mode_changed(self, mode: str) -> None:
        """Switch this scan's calibration between direct-beam fit and pySAred manual geometry."""
        is_pysared = mode == "pysared"
        self.cal_mode_stack.setCurrentIndex(1 if is_pysared else 0)
        scan = self.scan_data
        if scan is None:
            return
        nx = self._detector_nx()
        cal = scan.angle_calibration
        try:
            if is_pysared:
                if cal is None:
                    cal = AngleCalibration.from_pysared(
                        distance_mm=self.spin_distance_mm.value(),
                        x0=self.spin_x0.value(),
                        sign=1 if self.chk_af_increases.isChecked() else -1,
                        detector_width_mm=self.spin_det_width_mm.value(),
                        nx=nx,
                    )
                else:
                    cal = cal.with_mode("pysared", nx=nx)
            else:
                if cal is not None:
                    cal = cal.with_mode("direct_beam", nx=nx)
                else:
                    cal = self._ensure_calibration()
        except ValueError as error:
            self._set_notice(f"Calibration mode not changed: {error}")
            self._sync_cal_controls()
            return
        scan.angle_calibration = cal
        self._sync_cal_controls()
        self._redraw()

    def _on_calibration_edited(
        self,
        *,
        deg_per_pixel=None,
        x0=None,
        sign=None,
        distance_mm=None,
        detector_width_mm=None,
    ) -> None:
        """Manual override for this scan only; only the edited quantity changes."""
        scan = self.scan_data
        if scan is None:
            return
        cal = scan.angle_calibration
        nx = self._detector_nx()
        in_pysared = (
            self.btn_mode_pysared.isChecked()
            or distance_mm is not None
            or detector_width_mm is not None
        )
        try:
            if in_pysared:
                if cal is None:
                    cal = AngleCalibration.from_pysared(
                        distance_mm=self.spin_distance_mm.value() if distance_mm is None else distance_mm,
                        detector_width_mm=(
                            self.spin_det_width_mm.value() if detector_width_mm is None else detector_width_mm
                        ),
                        x0=self.spin_x0.value() if x0 is None else x0,
                        sign=(1 if self.chk_af_increases.isChecked() else -1) if sign is None else sign,
                        nx=nx,
                    )
                else:
                    cal = cal.with_pysared(
                        distance_mm=distance_mm,
                        detector_width_mm=detector_width_mm,
                        x0=x0,
                        sign=sign,
                        nx=nx,
                    )
            elif cal is None:
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
        self._sync_cal_controls()
        self._redraw()

    def _clear_alpha_items(self) -> None:
        if len(self._alpha_items) > 1:
            self.plot.setUpdatesEnabled(False)
            try:
                for item in self._alpha_items:
                    self.plot.removeItem(item)
            finally:
                self.plot.setUpdatesEnabled(True)
        else:
            for item in self._alpha_items:
                self.plot.removeItem(item)
        self._alpha_items = []
        self._alpha_geom = None
        self._th_geom = None

    def _draw_theta(self, display: np.ndarray) -> str | None:
        """Draw each frame at its measured θ as an exact PColorMesh strip (no uniform-row kink)."""
        flipper = self._variable("Flipper1")
        if (self.combo_spin.currentData() == "all" and flipper is not None
                and np.any(flipper == 0) and np.any(flipper == 1)):
            self._show_unavailable("X pixel vs θ: choose one spin state (u or d), their sum (u + d), "
                                   "or the spin asymmetry; both states share the same angles and would overlap.")
            return None
        th = self._variable("th")
        if th is None:
            self._show_unavailable("X pixel vs θ needs a θ (th) channel.")
            return None
        frames = self._norm.frame_indices
        ok = frames < th.size
        ok[ok] &= np.isfinite(th[frames[ok]])
        written = self._map.written
        if written is not None:
            ok &= written[frames]
        rows = np.flatnonzero(ok)
        if rows.size == 0:
            self._show_unavailable("X pixel vs θ: no measured frames with valid θ angles.")
            return None
        if self.img_item.image is not None:
            self.img_item.clear()
        theta = th[frames[rows]]
        n = rows.size
        nx = self._map.nx
        prev_th = self._th_geom
        can_reuse_mesh = (
            len(self._alpha_items) == 1
            and isinstance(self._alpha_items[0], CachedMeshItem)
        )
        same_th_geom = (
            can_reuse_mesh
            and prev_th is not None
            and prev_th["theta"].shape == theta.shape
            and np.array_equal(prev_th["theta"], theta)
            and self._alpha_items[0].z is not None
            and self._alpha_items[0].z.shape == (2 * n - 1, nx)
        )
        strips = prev_th["strips"] if same_th_geom else alpha_i_strip_edges(theta)
        Z = np.full((2 * n - 1, nx), np.nan)
        Z[0::2] = display[rows]
        if same_th_geom:
            mesh = self._alpha_items[0]
            mesh.setData(None, None, Z, autoLevels=False)
            self._colorbar_items_rebuilt = False
        else:
            x_edges = np.arange(nx + 1, dtype=float)
            X = np.broadcast_to(x_edges[None, :], (2 * n, nx + 1)).copy()
            Y = np.broadcast_to(strips.reshape(-1, 1), (2 * n, nx + 1)).copy()
            if can_reuse_mesh:
                mesh = self._alpha_items[0]
                mesh.setData(X, Y, Z, autoLevels=False)
                self._colorbar_items_rebuilt = False
            else:
                self._clear_alpha_items()
                mesh = CachedMeshItem(X, Y, Z, colorMap=self._current_colormap(), enableAutoLevels=False)
                self.plot.addItem(mesh)
                self._alpha_items = [mesh]
                self.colorbar.setImageItem(self._alpha_items)
                self._colorbar_on_strips = True
                self._colorbar_items_rebuilt = True
        self._alpha_geom = None
        self._th_geom = dict(
            rows=rows,
            theta=theta,
            strips=strips,
            extent=(0.0, float(nx), float(strips[:, 0].min()), float(strips[:, 1].max())),
        )
        skipped = frames.size - rows.size
        return f"X pixel vs θ: {skipped} unwritten/invalid frame(s) omitted." if skipped else ""

    def _draw_alpha(
        self,
        display: np.ndarray,
        *,
        levels: tuple[float, float] = (0.0, 1.0),
        asym: bool = False,
    ) -> str | None:
        """Draw each frame as its own α_i strip of exact pixel cells (or smoothly re-binned
        regular display grid when Smooth is enabled). Returns a notice, or None when unavailable."""
        cal = getattr(self.scan_data, "angle_calibration", None)
        name = "Qx vs Qz" if self._q_mode() else "α_i vs α_f"
        if cal is None:
            self._show_unavailable(f"{name} needs an angle calibration: "
                                   + (self._cal_message or "enter °/pixel in the calibration panel."))
            return None
        flipper = self._variable("Flipper1")
        if (self.combo_spin.currentData() == "all" and flipper is not None
                and np.any(flipper == 0) and np.any(flipper == 1)):
            self._show_unavailable(f"{name}: choose one spin state (u or d), their sum (u + d), or the spin asymmetry; "
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
        if self.img_item.image is not None:
            self.img_item.clear()
        sel = frames[rows]
        alpha_i = th[sel]
        base = tth[sel] - th[sel]
        prev_geom = self._alpha_geom
        same_angles = (
            prev_geom is not None
            and prev_geom["alpha_i"].shape == alpha_i.shape
            and np.array_equal(prev_geom["alpha_i"], alpha_i)
            and np.array_equal(prev_geom["base"], base)
            and prev_geom["cal"] == cal
            and prev_geom.get("nx") == self._map.nx
        )
        strips = prev_geom["strips"] if same_angles else alpha_i_strip_edges(alpha_i)
        edges = prev_geom["edges"] if same_angles else alpha_f_offsets(self._map.nx, cal, edges=True)
        span = abs(edges[-1] - edges[0])
        edges_min = float(edges.min())
        geom = dict(
            rows=rows,
            alpha_i=alpha_i,
            base=base,
            strips=strips,
            edges=edges,
            cal=cal,
            nx=self._map.nx,
            af_lo=float(base.min() + edges_min),
            af_hi=float(base.max() + float(edges.max())),
        )
        if self._q_mode():
            lam = self.spin_wavelength.value()
            geom["wavelength"] = lam
            if same_angles and prev_geom.get("wavelength") == lam and "extent" in prev_geom:
                q_extent = prev_geom["extent"]
                X_q = Y_q = None
            else:
                n = rows.size
                ai_corners = np.broadcast_to(strips.reshape(-1, 1), (2 * n, edges.size))
                af_corners = np.repeat(base, 2)[:, None] + edges[None, :]
                X_q, Y_q = self._to_view(ai_corners, af_corners)
                q_extent = (float(np.nanmin(X_q)), float(np.nanmax(X_q)), float(np.nanmin(Y_q)), float(np.nanmax(Y_q)))
            geom["extent"] = q_extent

        # Display-only smooth re-binning path (bilinear interpolation + optional edge-preserving
        # bilateral denoising in display space -> eliminates rectangular boxing without log blooming).
        if self.chk_smooth_map.isChecked():
            order = np.argsort(alpha_i, kind="stable")
            ai_s = alpha_i[order]
            if ai_s.size > 1:
                keep_u = np.concatenate((ai_s[:-1] < ai_s[1:], [True]))
            else:
                keep_u = np.ones(ai_s.size, dtype=bool)
            idx_u = order[keep_u]
            ai_u = alpha_i[idx_u]
            base_u = base[idx_u]
            strips_u = alpha_i_strip_edges(ai_u)
            src = display[rows[idx_u]]
            sigma = self.spin_smooth_sigma.value()
            if sigma > 0.0:
                sigma_r = 0.45 if self._log_active() else max(1e-6, 0.20 * max(1e-6, levels[1] - levels[0]))
                src = _bilateral_2d_display(src, sigma, sigma_r, fill_isolated_nans=not asym)
            view_extent = (
                geom["extent"]
                if self._q_mode()
                else (float(strips[:, 0].min()), float(strips[:, 1].max()), geom["af_lo"], geom["af_hi"])
            )
            if len(self._alpha_items) == 1 and isinstance(self._alpha_items[0], SmoothResampledMapItem):
                smooth_item = self._alpha_items[0]
                self._colorbar_items_rebuilt = False
            else:
                self._clear_alpha_items()
                smooth_item = SmoothResampledMapItem(colorMap=self._current_colormap())
                self.plot.addItem(smooth_item)
                self._alpha_items = [smooth_item]
                self.colorbar.setImageItem(self._alpha_items)
                self._colorbar_on_strips = True
                self._colorbar_items_rebuilt = True
            smooth_item.set_source(
                src,
                ai_u,
                base_u,
                strips_u,
                self._map.nx,
                cal,
                view_extent,
                q_mode=self._q_mode(),
                wavelength=geom.get("wavelength") if self._q_mode() else None,
            )
            self._th_geom = None
            self._alpha_geom = geom
            skipped = frames.size - rows.size
            return f"{name}: {skipped} unwritten/invalid frame(s) omitted." if skipped else ""

        if self._q_mode():
            # Exact cells: every pixel's four (α_i, α_f) corners mapped to (Qx, Qz). One mesh;
            # the NaN rows between strips keep points separate (no interpolation).
            lam = geom["wavelength"]
            n = rows.size
            Z = np.full((2 * n - 1, self._map.nx), np.nan)
            Z[0::2] = display[rows]
            can_reuse_mesh = (
                len(self._alpha_items) == 1
                and isinstance(self._alpha_items[0], CachedMeshItem)
            )
            same_q_geom = (
                can_reuse_mesh
                and same_angles
                and prev_geom.get("wavelength") == lam
                and "extent" in prev_geom
                and self._alpha_items[0].z is not None
                and self._alpha_items[0].z.shape == Z.shape
            )
            if same_q_geom:
                mesh = self._alpha_items[0]
                mesh.setData(None, None, Z, autoLevels=False)
                self._colorbar_items_rebuilt = False
            else:
                if X_q is None or Y_q is None:
                    ai_corners = np.broadcast_to(strips.reshape(-1, 1), (2 * n, edges.size))
                    af_corners = np.repeat(base, 2)[:, None] + edges[None, :]
                    X_q, Y_q = self._to_view(ai_corners, af_corners)
                if can_reuse_mesh:
                    mesh = self._alpha_items[0]
                    mesh.setData(X_q, Y_q, Z, autoLevels=False)
                    self._colorbar_items_rebuilt = False
                else:
                    self._clear_alpha_items()
                    mesh = CachedMeshItem(X_q, Y_q, Z, colorMap=self._current_colormap(), enableAutoLevels=False)
                    self.plot.addItem(mesh)
                    self._alpha_items = [mesh]
                    self.colorbar.setImageItem(self._alpha_items)
                    self._colorbar_on_strips = True
                    self._colorbar_items_rebuilt = True
        else:
            can_reuse_strips = (
                len(self._alpha_items) == rows.size
                and rows.size > 0
                and isinstance(self._alpha_items[0], pg.ImageItem)
            )
            if can_reuse_strips:
                self.plot.setUpdatesEnabled(False)
                try:
                    for k, r in enumerate(rows):
                        values = display[r] if cal.sign > 0 else display[r][::-1]
                        item = self._alpha_items[k]
                        item.setImage(values[None, :], autoLevels=False)
                        if not same_angles:
                            item.setRect(strips[k, 0], base[k] + edges_min, strips[k, 1] - strips[k, 0], span)
                finally:
                    self.plot.setUpdatesEnabled(True)
                self._colorbar_items_rebuilt = False
            else:
                self._clear_alpha_items()
                items = []
                cmap = self._current_colormap()
                self.plot.setUpdatesEnabled(False)
                try:
                    for k, r in enumerate(rows):
                        values = display[r] if cal.sign > 0 else display[r][::-1]  # image runs to increasing α_f
                        item = pg.ImageItem(values[None, :], autoLevels=False)
                        item.setColorMap(cmap)
                        item.setRect(strips[k, 0], base[k] + edges_min, strips[k, 1] - strips[k, 0], span)
                        self.plot.addItem(item)
                        items.append(item)
                finally:
                    self.plot.setUpdatesEnabled(True)
                self._alpha_items = items
                self.colorbar.setImageItem(items)
                self._colorbar_on_strips = True
                self._colorbar_items_rebuilt = True
        self._th_geom = None
        self._alpha_geom = geom
        skipped = frames.size - rows.size
        return f"{name}: {skipped} unwritten/invalid frame(s) omitted." if skipped else ""

    def _map_extent(self) -> tuple[float, float, float, float]:
        if self._alpha_mode() and self._alpha_geom is not None:
            g = self._alpha_geom
            if "extent" in g:
                return g["extent"]
            return float(g["strips"][:, 0].min()), float(g["strips"][:, 1].max()), g["af_lo"], g["af_hi"]
        if self._theta_mode() and self._th_geom is not None:
            return self._th_geom["extent"]
        n_rows = self._norm.frame_indices.size if self._norm is not None else 1
        nx = self._map.nx if self._map is not None else 1
        return 0.0, float(nx), 0.0, float(n_rows)

    def _pixel_for_alpha(self, k: int, alpha_f: float) -> int | None:
        g = self._alpha_geom
        cal = g["cal"]
        x_coord = pixel_for_alpha_f_offset(alpha_f - g["base"][k], self._map.nx, cal)
        if not np.isfinite(x_coord):
            return None
        px = int(np.floor(x_coord))
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
        if self._theta_mode():
            g = self._th_geom
            x = int(np.floor(xv))
            if not (0 <= x < self._map.nx):
                return None
            inside = np.flatnonzero((g["strips"][:, 0] <= yv) & (yv < g["strips"][:, 1]))
            if inside.size == 0:
                return None
            k = int(inside[-1])
            return int(g["rows"][k]), x
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
            offset = yv - g["base"][k]
            x_coord = pixel_for_alpha_f_offset(offset, self._map.nx, cal)
            if np.isfinite(x_coord):
                px = int(np.clip(np.floor(x_coord), 0, self._map.nx - 1))
            else:
                px = (self._map.nx - 1) if (cal.sign * offset > 0) else 0
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
        if self._theta_mode() and self._th_geom is not None:
            g = self._th_geom
            k = int(np.argmin(np.abs(g["rows"] - row)))
            return x + 0.5, float(g["theta"][k])
        return x + 0.5, row + 0.5

    def _update_alpha_cuts(self, positive) -> None:
        """Row cut: the nearest measured point versus α_f. Column cut: constant α_f across points."""
        g = self._alpha_geom
        view = getattr(self, "_cross_alpha", self._cell_center(self._sel_row, self._sel_x))
        if self._q_mode():
            self._update_q_cuts(positive, *view)
            return
        row_active, col_active = self._cut_curve_flags()
        xv, yv = (float(v) for v in self._from_view(*view))
        k = self._nearest_strip(xv)
        row = int(g["rows"][k])
        cal = g["cal"]
        if row_active:
            self._row_curve.setData(g["base"][k] + alpha_f_offsets(self._map.nx, cal),
                                    positive(self._cut_values[row]))
            self._row_curve.setVisible(True)
            frame = int(self._norm.frame_indices[row])
            self.lbl_row_cut.setText(f"Cut at α_i = {g['alpha_i'][k]:.4f}° — point {frame} ({self._describe_frame(frame)}):")
        else:
            self._row_curve.setData([], [])
            self._row_curve.setVisible(False)
            self.lbl_row_cut.setText("Vertical cut at α_i (toggle Vertical above):")
        if col_active:
            col = np.full(g["rows"].size, np.nan)
            for j in range(g["rows"].size):
                px = self._pixel_for_alpha(j, yv)
                if px is not None:
                    col[j] = self._cut_values[g["rows"][j], px]
            order = np.argsort(g["alpha_i"], kind="stable")
            self._col_curve.setData(g["alpha_i"][order], positive(col[order]))
            self._col_curve.setVisible(True)
            self.lbl_col_cut.setText(f"Cut at constant α_f = {yv:.4f}°:")
        else:
            self._col_curve.setData([], [])
            self._col_curve.setVisible(False)
            self.lbl_col_cut.setText("Horizontal cut at constant α_f (toggle Horizontal above):")
        self.lbl_cut_hint.setText("Drag the cut lines or crosshair on the map to move cuts; click to jump.")

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
        if len(bounds) == 1:
            x_lo = np.full(frames.size, min(bounds[0][0], bounds[0][1]), dtype=float)
            x_hi = np.full(frames.size, max(bounds[0][0], bounds[0][1]), dtype=float)
        else:
            per_row = np.array([bounds[min(int(f), len(bounds) - 1)] for f in frames], dtype=float)
            x_lo = np.minimum(per_row[:, 0], per_row[:, 1])
            x_hi = np.maximum(per_row[:, 0], per_row[:, 1])
        ai_pairs = g["strips"].reshape(-1)
        for xb in (x_lo, x_hi):
            af = g["base"] + alpha_f_offset_for_pixel(xb, self._map.nx, cal)
            vx, vy = self._to_view(ai_pairs, np.repeat(af, 2))
            self._add_overlay(vx, vy, "#facc15", connect="pairs")
        params = self.scan_data.reduction_params
        if params.subtract_background:
            bx0, bx1, _, _ = resolve_bkg_roi(
                bounds[0], params.bkg_roi_bounds, params.bkg_x_bounds, self._map.nx, self._detector_ny_full()
            )
            for xb_val in (float(bx0), float(bx1)):
                xb_arr = np.full(frames.size, xb_val, dtype=float)
                af = g["base"] + alpha_f_offset_for_pixel(xb_arr, self._map.nx, cal)
                vx, vy = self._to_view(ai_pairs, np.repeat(af, 2))
                self._add_overlay(vx, vy, "#06b6d4", connect="pairs")
