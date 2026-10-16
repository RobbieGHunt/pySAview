from PyQt6.QtWidgets import QFileDialog, QMessageBox
from pySAview.ui.scan_plot_view import ScanPlotView
from pySAview.tests.test_processing_ui import scan
import pytest


@pytest.mark.parametrize("extension", ["png", "svg"])
def test_export_plot_button_writes_image(qtbot, tmp_path, monkeypatch, extension):
    view = ScanPlotView()
    qtbot.addWidget(view)
    view.add_scan(scan())
    view.resize(1000, 700)
    view.show()
    qtbot.wait(20)
    target = tmp_path / f"plot.{extension}"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), extension))
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: errors.append(a))
    view.btn_export_plot.click()
    assert not errors
    assert target.stat().st_size > 100
    if extension == "png":
        assert target.read_bytes().startswith(b"\x89PNG")
    else:
        assert "<svg" in target.read_text(encoding="utf-8")
    assert not view.plot_widget.scene().contextMenu


def test_plot_export_cannot_overwrite_source(qtbot, tmp_path, monkeypatch):
    view = ScanPlotView()
    qtbot.addWidget(view)
    source = tmp_path / "measurement.nxs"
    source.write_bytes(b"measurement")
    view.add_scan(scan(str(source)))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(source), "PNG"))
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: errors.append(a))
    view._export_plot()
    assert errors
    assert source.read_bytes() == b"measurement"
