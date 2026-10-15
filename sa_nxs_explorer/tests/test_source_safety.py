import hashlib
import pytest
from sa_nxs_explorer.core.source_safety import validate_output_path
from sa_nxs_explorer.tests.test_processing_ui import scan
from sa_nxs_explorer.ui.scan_plot_view import ScanPlotView
from sa_nxs_explorer.ui.scan_plot_view import QFileDialog, QMessageBox
import numpy as np


def test_export_cannot_overwrite_measurement(qtbot, tmp_path, monkeypatch):
    source = tmp_path / "scan.nxs"
    source.write_bytes(b"original measurement")
    digest = hashlib.sha256(source.read_bytes()).digest()
    view = ScanPlotView()
    qtbot.addWidget(view)
    view.add_scan(scan(str(source)))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(source), "CSV"))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    view._export_csv()
    assert hashlib.sha256(source.read_bytes()).digest() == digest


def test_output_guard_blocks_sources_and_hardlinks(tmp_path):
    source = tmp_path / "measurement.dat"
    source.write_bytes(b"counts")
    alias = tmp_path / "export.csv"
    alias.hardlink_to(source)
    for target in (source, alias, tmp_path / "other.NXS"):
        with pytest.raises(ValueError):
            validate_output_path(target, [source])
    assert validate_output_path(tmp_path / "safe.csv", [source]).name == "safe.csv"


def test_working_roi_is_independent_for_each_scan(qtbot):
    view = ScanPlotView()
    qtbot.addWidget(view)
    first, second = scan("first.nxs"), scan("second.nxs")
    original = (1., 3., 2., 4.)
    first.detector_roi = original
    second.detector_roi = (4., 6., 5., 7.)
    view.add_scan(first)
    view.add_scan(second)
    bounds = (10., 12., 20., 22.)
    view.update_roi_counts(np.array([1., 2., 3.]), bounds, file_path=first.file_path)
    assert first.detector_roi == original
    assert first.working_detector_roi == bounds
    assert second.working_detector_roi is None
    assert "ROI_Counts" not in second.variables
