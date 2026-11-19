"""Shared plot toolbar buttons and guarded plot-image export.

Every plot in the application uses the same icon-only tool buttons and the same
image export path (toolbar button, File ▸ Export and the right-click menu). All
writes go through :func:`validate_output_path`, so a measurement file can never
be selected as an export destination.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import QRectF, QSize
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QPushButton, QWidget

from pySAview.core.source_safety import validate_output_path

_SUPERSCRIPT_MAP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def format_scientific_tick(value: float, *, is_log_exponent: bool = False) -> str:
    """Format a Y-axis tick consistently in scientific ``10ⁿ`` / ``m×10ⁿ`` notation.

    Unlike default ``%g`` formatting (which prints ``0.001`` for ``10⁻³`` and
    ``1e-05`` for ``10⁻⁵``), every decade from ``10⁻¹²`` through ``10¹²`` uses
    consistent ``10ⁿ`` superscripts (e.g. ``10⁻⁴``, ``10⁻³``, ``10⁻²``, ``10⁻¹``,
    ``10⁰``, ``10¹``).
    """
    if not math.isfinite(value):
        return ""
    if is_log_exponent:
        nearest = round(value)
        if abs(value - nearest) < 1e-6:
            return f"10{str(int(nearest)).translate(_SUPERSCRIPT_MAP)}"
        actual = 10.0 ** value
        if not math.isfinite(actual) or actual <= 0.0:
            return ""
        exp = int(math.floor(value))
        mant = actual / (10.0 ** exp)
        if round(mant, 2) >= 10.0:
            exp += 1
            mant = 1.0
        exp_str = str(exp).translate(_SUPERSCRIPT_MAP)
        if abs(mant - 1.0) < 1e-3:
            return f"10{exp_str}"
        mant_str = f"{mant:.2g}"
        return f"{mant_str}×10{exp_str}"

    if value == 0.0:
        return "0"
    sign = "−" if value < 0.0 else ""
    mag = abs(value)
    exp = int(math.floor(math.log10(mag)))
    mant = mag / (10.0 ** exp)
    if round(mant, 3) >= 10.0:
        exp += 1
        mant = 1.0
    exp_str = str(exp).translate(_SUPERSCRIPT_MAP)
    if abs(mant - 1.0) < 1e-6:
        return f"{sign}10{exp_str}"
    mant_str = f"{mant:.3g}"
    return f"{sign}{mant_str}×10{exp_str}"


class ScientificAxisItem(pg.AxisItem):
    """PyQtGraph axis that consistently labels ticks in scientific ``10ⁿ`` notation."""

    def __init__(self, *args, linear_scientific: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._linear_scientific = bool(linear_scientific)

    def logTickStrings(self, values, scale, spacing):
        scale_val = float(scale) if scale is not None else 1.0
        log_scale = math.log10(scale_val) if scale_val > 0.0 else 0.0
        labels: list[str] = []
        for v in values:
            exp_val = float(v) + log_scale
            if spacing is not None and spacing >= 1.0 and abs(exp_val - round(exp_val)) >= 1e-6:
                labels.append("")
            else:
                labels.append(format_scientific_tick(exp_val, is_log_exponent=True))
        return labels

    def tickStrings(self, values, scale, spacing):
        if self.logMode:
            return self.logTickStrings(values, scale, spacing)
        if not self._linear_scientific:
            return super().tickStrings(values, scale, spacing)
        scale_val = float(scale) if scale is not None else 1.0
        return [format_scientific_tick(float(v) * scale_val, is_log_exponent=False) for v in values]

IMAGE_FILTERS = "PNG image (*.png);;SVG vector image (*.svg)"
TABLE_FILTERS = "Tab-delimited text (*.txt);;Tab-delimited data (*.dat);;Tab-separated values (*.tsv)"


def tool_button(tooltip: str, *, checkable: bool = False) -> QPushButton:
    """Compact icon-only plot tool button (the name lives in the tooltip)."""
    button = QPushButton()
    button.setToolTip(tooltip)
    button.setAccessibleName(tooltip.split(":")[0])
    button.setFixedSize(34, 30)
    button.setIconSize(QSize(18, 18))
    button.setCheckable(checkable)
    return button


def ask_output_path(parent: QWidget | None, title: str, default_name: str, filters: str,
                    source_paths: Iterable[str]) -> Path | None:
    """Ask for a destination; adds the selected filter's extension and applies the source guard.

    Returns None when cancelled. Raises ValueError for a forbidden destination.
    """
    path, selected = QFileDialog.getSaveFileName(parent, title, default_name, filters)
    if not path:
        return None
    target = Path(path)
    if not target.suffix:
        start = (selected or "").find("*.")
        suffix = selected[start + 1:].split(")")[0].split()[0] if start >= 0 else Path(default_name).suffix
        target = target.with_suffix(suffix or Path(default_name).suffix)
    validate_output_path(target, [p for p in source_paths if p])
    return target


def save_plot_image(plot_item, target: Path) -> None:
    """Render a PlotItem (axes, legend, colour bar) to PNG or SVG."""
    suffix = Path(target).suffix.lower()
    if suffix == ".png":
        from pyqtgraph.exporters import ImageExporter
        ImageExporter(plot_item).export(str(target))
        return
    if suffix != ".svg":
        raise ValueError("Choose a .png or .svg output file")
    # Render through Qt directly: pyqtgraph's SVG post-processor cannot parse some
    # paths generated by current Qt versions.
    from PyQt6.QtSvg import QSvgGenerator
    scene = plot_item.scene()
    rect = plot_item.sceneBoundingRect()
    bounds = QRectF(0, 0, rect.width(), rect.height())
    generator = QSvgGenerator()
    generator.setFileName(str(target))
    generator.setSize(QSize(math.ceil(rect.width()), math.ceil(rect.height())))
    generator.setViewBox(bounds)
    painter = QPainter(generator)
    if not painter.isActive():
        raise OSError("Could not open SVG output")
    try:
        plot_item.setExportMode(True, {"antialias": True})
        views = scene.views()
        if views:
            painter.fillRect(bounds, views[0].backgroundBrush())
        scene.render(painter, bounds, rect)
    finally:
        painter.end()
        plot_item.setExportMode(False)


def export_plot_image(parent: QWidget | None, plot_item, default_name: str,
                      source_paths: Iterable[str], title: str = "Export Plot Image") -> Path | None:
    """Ask for a PNG/SVG destination and save the plot; errors are reported in a dialog."""
    try:
        target = ask_output_path(parent, title, default_name, IMAGE_FILTERS, source_paths)
        if target is None:
            return None
        save_plot_image(plot_item, target)
        return target
    except Exception as error:  # noqa: BLE001 - shown to the user
        QMessageBox.critical(parent, "Export Failed", f"Failed to save image:\n{error}")
        return None


def install_context_export(plot_widget, default_name: Callable[[], str] | str,
                           source_paths: Callable[[], Iterable[str]]) -> None:
    """Route the plot's right-click "Export…" entry through the guarded image export.

    pyqtgraph's own export dialog writes without our source-file guard and can
    export display-shifted data as numbers; numerical exports live in File ▸ Export.
    """
    scene = plot_widget.scene()
    if not scene.contextMenu:
        return
    action = scene.contextMenu[0]
    try:
        action.triggered.disconnect()
    except TypeError:
        pass
    action.setText("Export Image…")

    def _export() -> None:
        name = default_name() if callable(default_name) else default_name
        export_plot_image(plot_widget, plot_widget.getPlotItem(), name, source_paths())

    action.triggered.connect(_export)
