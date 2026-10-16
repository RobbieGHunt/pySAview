import json
from PyQt6.QtCore import QSettings
from pySAview.ui.main_window import MainWindow, QFileDialog
from pySAview.ui.preferences import save_preferences


def test_preferences_roundtrip_empty_workspace(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    first = MainWindow(settings=settings)
    qtbot.addWidget(first)
    first.last_load_directory = str(tmp_path)
    first.act_dark_mode.setChecked(True)
    chart = first.view_charts
    chart.combo_up_colormap.setCurrentText("viridis")
    chart.chk_show_legend.setChecked(False)
    chart.chk_show_data_panel.setChecked(False)
    chart.spin_axis_font_size.setValue(17)
    chart.spin_style_controls["d"]["marker_size"].setValue(9)
    chart._on_shade_range_changed(.2, .8)
    chart.combo_norm.setCurrentIndex(chart.combo_norm.findData("monitor"))
    chart.chk_waterfall.setChecked(True)
    chart.spin_waterfall_spacing.setValue(.5)
    chart.chk_log_y.setChecked(False)
    save_preferences(first)
    second = MainWindow(settings=settings)
    qtbot.addWidget(second)
    restored = second.view_charts
    assert not second.loaded_scans
    assert second.last_load_directory == str(tmp_path)
    assert second.is_dark_mode
    assert restored.combo_up_colormap.currentText() == "viridis"
    assert not restored.chk_show_legend.isChecked()
    assert not restored.legend.isVisible()
    assert restored.bottom_tabs.isHidden()
    assert restored.spin_axis_font_size.value() == 17
    assert restored.spin_style_controls["d"]["marker_size"].value() == 9
    assert restored.shade_range == (.2, .8)
    assert restored.combo_norm.currentData() == "monitor"
    assert restored.chk_waterfall.isChecked()
    assert restored.combo_waterfall_mode.isEnabled()
    assert restored.spin_waterfall_spacing.value() == .5
    assert not restored.chk_log_y.isChecked()


def test_open_dialog_remembers_directory(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.last_load_directory = str(tmp_path)
    captured = []
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a: (captured.append(a[2]) or [], ""))
    window._on_action_open()
    window._on_action_add()
    assert captured == [str(tmp_path)] * 2


def test_invalid_preferences_use_defaults(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    settings.setValue("preferences", json.dumps(dict(version=1, controls={"spin_wavelength": -9, "combo_norm": "missing"}, shades=[-1, 5])))
    window = MainWindow(settings=settings)
    qtbot.addWidget(window)
    assert window.view_charts.spin_wavelength.value() == 5.21
    assert window.view_charts.combo_norm.currentData() == "raw"
    assert window.view_charts.combo_up_colormap.currentText() == "Reds"
    assert window.view_charts.shade_range == (0., 1.)


def test_startup_only_opens_explicit_file(monkeypatch):
    from pySAview import app
    opened = []
    class Window:
        def load_file(self, p): opened.append(p)
        def show(self): pass
    class Application:
        def exec(self): return 0
    monkeypatch.setattr(app, "MainWindow", Window)
    monkeypatch.setattr(app, "create_app", Application)
    monkeypatch.setattr("sys.argv", ["pySAview"])
    assert app.main() == 0
    assert opened == []
