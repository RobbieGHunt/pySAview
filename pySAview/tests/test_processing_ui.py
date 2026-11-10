import csv
from pathlib import Path
import numpy as np
import pytest
import pyqtgraph as pg
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from pySAview.core.models import ScanData, ScanMetadata, ScanVariable
from pySAview.ui.scan_plot_view import ScanPlotView


def scan(name="first.nxs", detector=(0, 100, 200), monitor=(1000, 1000, 2000),
         detector_unit="counts", with_detector=True, extra=None):
    def variable(key, values, unit):
        return ScanVariable(0, key, key, unit, "", 0, False, np.asarray(values, dtype=float))
    variables = {"th": variable("th", [0, 1, 2], "deg"),
                 "Count": variable("Count", [1, 10, 20], "s")}
    if with_detector:
        variables["Detector1"] = variable("Detector1", detector, detector_unit)
    if monitor is not None:
        variables["Monitor1"] = variable("Monitor1", monitor, "counts")
    if extra:
        variables.update(extra)
    return ScanData(name, ScanMetadata(), variables)


@pytest.fixture
def view(qtbot):
    result = ScanPlotView()
    qtbot.addWidget(result)
    return result


def set_mode(view, mode):
    view.combo_norm.setCurrentIndex(view.combo_norm.findData(mode))


def test_exact_intervals_and_log_hidden_zero_remain_in_table(view):
    view.add_scan(scan())
    assert len(view._current_curves) == 1
    curve = view._current_curves[0]
    assert curve.y[0] == 0
    assert curve.y_upper[0] > 1.8
    assert "Garwood" in view.processing_status
    assert "Log axes hide" in view.processing_status
    assert view.curve_data_table.rowCount() == 3
    assert float(view.curve_data_table.item(0, 7).text()) > 1.8


def test_missing_monitor_skips_scan_without_raw_fallback(view):
    view.add_scan(scan())
    view.add_scan(scan("missing.nxs", monitor=None))
    set_mode(view, "monitor")
    assert len(view._current_curves) == 1
    assert "missing.nxs: skipped" in view.processing_status
    assert "Monitor counts is required" in view.processing_status
    np.testing.assert_allclose(view._current_curves[0].y, [0, 0.1, 0.1])


def test_selected_roi_does_not_become_detector_total_in_other_scan(view):
    first = scan()
    first.variables["ROI_Counts"] = ScanVariable(1, "ROI_Counts", "ROI", "counts", "", 0, False, np.array([0, 1, 2]))
    view.add_scan(first)
    view.add_scan(scan("no_roi.nxs"))
    view.combo_y.setCurrentIndex(view.combo_y.findData("ROI_Counts"))
    assert len(view._current_curves) == 1
    np.testing.assert_array_equal(view._current_curves[0].y, [0, 1, 2])
    assert "missing selected Y channel ROI_Counts" in view.processing_status


def test_overlay_unit_mismatch_is_reported(view):
    view.add_scan(scan())
    view.add_scan(scan("wrong_units.nxs", detector_unit="counts/s"))
    assert len(view._current_curves) == 1
    assert "units differ" in view.processing_status


def test_unit_symbols_preserve_case_and_prefix_scale(view):
    view.add_scan(scan(detector_unit="mA"))
    view.add_scan(scan("megamp.nxs", detector_unit="MA"))
    assert len(view._current_curves) == 1
    assert "units differ" in view.processing_status


def test_invalid_monitor_points_keep_original_step_indices(view):
    view.add_scan(scan(monitor=(1000, 0, 2000)))
    set_mode(view, "monitor")
    curve = view._current_curves[0]
    np.testing.assert_array_equal(curve.indices, [0, 2])
    assert "1 invalid points excluded" in view.processing_status
    assert "invalid monitor counts" in view.processing_status


def test_raw_motor_channel_has_no_poisson_error_bars(view):
    view.add_scan(scan())
    view.combo_y.setCurrentIndex(view.combo_y.findData("th"))
    assert view._current_curves[0].y_err is None
    assert view._current_curves[0].y_lower is None
    assert not any(isinstance(item, pg.ErrorBarItem) for item in view.plot_widget.items())
    assert "no supplied uncertainty" in view.processing_status
    set_mode(view, "monitor")
    assert not view._current_curves
    assert "only for count channels" in view.processing_status


def test_fixed_reference_rate_applies_identically_to_overlays(view):
    view.add_scan(scan())
    view.add_scan(scan("second.nxs", detector=(0, 100, 200), monitor=(2000, 1000, 1000)))
    set_mode(view, "monitor_time")
    view.spin_reference_rate.setValue(250)
    assert len(view._current_curves) == 2
    assert view._current_curves[0].y[1] == view._current_curves[1].y[1] == 25
    assert "250 monitor counts/s" in view.processing_status
    assert all(c.reference_monitor_rate == 250 for c in view._current_curves)


def test_monitor_on_itself_cancels_correlated_uncertainty(view):
    view.add_scan(scan())
    view.combo_y.setCurrentIndex(view.combo_y.findData("Monitor1"))
    set_mode(view, "monitor")
    np.testing.assert_array_equal(view._current_curves[0].y, [1]*3)
    np.testing.assert_array_equal(view._current_curves[0].y_err, [0]*3)
    assert "cancels exactly" in view.processing_status


TAB = chr(9)


def read_table(path):
    """Header comments and rows of a tab-delimited export."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = [line[2:] for line in lines if line.startswith("#")]
    rows = list(csv.DictReader([line for line in lines if not line.startswith("#")], delimiter=TAB))
    return header, rows


def test_export_retains_interval_precision_method_and_processing(view, tmp_path, monkeypatch):
    target = tmp_path / "export.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target), "Tab-delimited text (*.txt)"))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    view.add_scan(scan())
    set_mode(view, "monitor_time")
    view.spin_reference_rate.setValue(123.456789)
    view.export_data()
    header, rows = read_table(target)
    assert len(rows) == 3
    assert TAB in target.read_text(encoding="utf-8").splitlines()[len(header)]
    assert float(rows[0]["UpperBound"]) == view._current_curves[0].y_upper[0]
    assert float(rows[0]["StdUncertainty"]) == view._current_curves[0].y_err[0]
    text = " ".join(header)
    assert "Normalization: monitor_time" in text
    assert "Reference monitor rate: 123.456789" in text
    assert "conditional exact" in text and "source: first.nxs" in text
    assert "Log axes hide" in view.processing_status
    np.testing.assert_array_equal([float(r["UpperBound"]) for r in rows], view._current_curves[0].y_upper)


def test_processing_label_shows_only_warnings(view):
    view.add_scan(scan())
    assert "Garwood" in view.processing_status or "exact" in view.processing_status
    assert "Processing:" not in view.lbl_processing.text()
    assert view.lbl_processing.isHidden() == (view.lbl_processing.text() == "")


def test_requesting_absent_spin_does_not_plot_other_state(view):
    data = scan()
    data.variables["Flipper1"] = ScanVariable(3, "Flipper1", "F1", "", "", 0, False, np.array([0, 0, 0]))
    view.add_scan(data)
    view.combo_spin.setCurrentIndex(view.combo_spin.findData("flipper1"))
    assert not view._current_curves


def test_theme_resets_stale_tick_scaling_on_both_axes(view):
    view.add_scan(scan())
    for side in ("left", "bottom"):
        axis = view.plot_widget.getAxis(side)
        axis.autoSIPrefixScale = 1000
        axis.labelUnitPrefix = "m"
    view.set_theme(True)
    for side in ("left", "bottom"):
        axis = view.plot_widget.getAxis(side)
        assert axis.autoSIPrefixScale == 1
        assert axis.labelUnitPrefix == ""


def test_unknown_time_units_are_not_assumed_to_be_seconds(view):
    data = scan()
    data.variables["Count"].unit = ""
    view.add_scan(data)
    set_mode(view, "time")
    assert not view._current_curves
    assert "must have units of seconds" in view.processing_status


def test_explicit_variance_fallback_is_visible(view):
    data = scan()
    data.variables["Detector1"].variance = np.array([4, 100, 200])
    view.add_scan(data)
    assert "propagated standard uncertainty" in view.processing_status
    assert "Exact intervals unavailable" in view.lbl_processing.text()  # requested method not used: visible
    assert "exact" not in view.processing_status
    assert view._current_curves[0].y_err[0] == 2
