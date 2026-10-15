"""Local UI preferences only: no measurement files or ROI state."""
import json
import math
from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox


def controls(window):
    chart = window.view_charts
    result = {name: getattr(chart, name) for name in (
        "combo_norm", "combo_uncertainty", "combo_spin", "combo_style",
        "spin_wavelength", "spin_reference_rate", "chk_log_x", "chk_log_y",
        "chk_errors", "chk_waterfall", "combo_waterfall_mode", "spin_waterfall_spacing")}
    for spin, items in chart.spin_style_controls.items():
        for name in ("colormap", "line_style", "line_width", "marker_size"):
            result[f"{spin}/{name}"] = items[name]
    for name in ("combo_cmap", "chk_log_img", "chk_show_roi"):
        result[f"detector/{name}"] = getattr(window.view_detector, name)
    return result


def save_preferences(window):
    values = {}
    for name, widget in controls(window).items():
        if isinstance(widget, QCheckBox):
            values[name] = widget.isChecked()
        elif isinstance(widget, QComboBox):
            values[name] = widget.currentData() if widget.currentData() is not None else widget.currentText()
        else:
            values[name] = widget.value()
    data = dict(version=1, controls=values, dark=window.is_dark_mode,
                directory=window.last_load_directory, shades=window.view_charts.shade_range)
    window.settings.setValue("preferences", json.dumps(data, allow_nan=False))
    window.settings.sync()


def restore_preferences(window):
    try:
        data = json.loads(window.settings.value("preferences", "{}"))
    except (TypeError, ValueError):
        return
    if not isinstance(data, dict) or data.get("version") != 1:
        return
    directory = data.get("directory")
    if isinstance(directory, str):
        window.last_load_directory = directory
    widgets = controls(window)
    blockers = [QSignalBlocker(w) for w in widgets.values()]
    saved = data.get("controls", {})
    if isinstance(saved, dict):
        for name, widget in widgets.items():
            if name not in saved:
                continue
            value = saved.get(name)
            if isinstance(widget, QCheckBox) and type(value) is bool:
                widget.setChecked(value)
            elif isinstance(widget, QComboBox):
                index = widget.findData(value)
                if index < 0 and isinstance(value, str):
                    index = widget.findText(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif isinstance(widget, QDoubleSpinBox) and type(value) in (float, int) and math.isfinite(value) and widget.minimum() <= value <= widget.maximum():
                widget.setValue(value)
    del blockers
    shades = data.get("shades")
    if isinstance(shades, list) and len(shades) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in shades) and 0 <= shades[0] <= shades[1] <= 1:
        window.view_charts._on_shade_range_changed(*shades)
    chart = window.view_charts
    chart.combo_waterfall_mode.setEnabled(chart.chk_waterfall.isChecked())
    chart.spin_waterfall_spacing.setEnabled(chart.chk_waterfall.isChecked())
    chart._on_axis_or_setting_changed()
    window.view_detector._on_cmap_changed(window.view_detector.combo_cmap.currentText())
    if type(data.get("dark")) is bool:
        window.act_dark_mode.setChecked(data["dark"])
