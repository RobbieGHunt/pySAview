"""File ▸ Export, tab-delimited data exports and the guarded right-click image export."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from pySAview.core.angle_calibration import alpha_f_offsets, q_from_angles
from pySAview.core.export_tables import write_table
from pySAview.tests.test_processing_ui import read_table
from pySAview.ui.main_window import MainWindow

EXAMPLE = Path(__file__).resolve().parent.parent / "example_data" / "005572.nxs"


@pytest.fixture
def window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_file(EXAMPLE)
    yield win
    win.view_offspec.shutdown()


@pytest.fixture
def dialogs(monkeypatch):
    """Answer save dialogs with ``answer['path']`` and collect message boxes."""
    answer = {"path": None, "errors": [], "info": []}
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(answer["path"]), ""))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: answer["errors"].append(a))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: answer["info"].append(a))
    return answer


def _ready(qtbot, view):
    view._request_compute()
    qtbot.waitUntil(lambda: view._norm is not None and not view._jobs, timeout=60000)


def test_write_table_round_trips_full_precision_and_guards_sources(tmp_path):
    target = tmp_path / "t.txt"
    value = 0.1 + 0.2
    write_table(target, ["note\twith tab"], ["A", "B"], [(1, value), (2, float("nan"))])
    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "# note with tab"
    assert lines[2] == "A\tB"
    assert float(lines[3].split("\t")[1]) == value
    assert lines[4].split("\t")[1] == "nan"
    source = tmp_path / "scan.nxs"
    source.write_bytes(b"measurement")
    for bad in (source, tmp_path / "out.h5"):
        with pytest.raises(ValueError):
            write_table(bad, [], ["A"], [], [source])
    assert source.read_bytes() == b"measurement"


def test_file_menu_has_export_entries(window):
    titles = [a.text() for a in window.menu_export.actions() if not a.isSeparator()]
    assert any("Reflectivity Data" in t for t in titles)
    assert any("Reflectivity" in t and "Image" in t for t in titles)
    assert any("Off-Specular Map Data" in t for t in titles)
    assert any("Off-Specular" in t and "Image" in t for t in titles)
    assert window.menu_export.menuAction() in window.menuBar().actions()[0].menu().actions()


def test_reflectivity_export_from_file_menu(window, dialogs, tmp_path):
    dialogs["path"] = tmp_path / "refl"  # no extension: the default .txt is added
    window.act_export_refl_data.trigger()
    target = tmp_path / "refl.txt"
    header, rows = read_table(target)
    curves = window.view_charts._current_curves
    assert len(rows) == sum(len(c.x) for c in curves)
    assert float(rows[0]["StdUncertainty"]) == curves[0].y_err[0]
    assert any("own ROI" in line for line in header)
    dialogs["path"] = tmp_path / "refl.png"
    window.act_export_refl_image.trigger()
    assert (tmp_path / "refl.png").read_bytes().startswith(b"\x89PNG")
    assert not dialogs["errors"]


@pytest.mark.parametrize("axes", ["point", "th", "q"])
def test_offspecular_export_matches_map_in_current_axes(qtbot, window, dialogs, tmp_path, axes):
    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    _ready(qtbot, view)
    view.combo_spin.setCurrentIndex(view.combo_spin.findData("down"))
    view.combo_y_axis.setCurrentIndex(view.combo_y_axis.findData(axes))
    dialogs["path"] = tmp_path / f"map_{axes}.txt"
    window.act_export_offspec_data.trigger()
    assert not dialogs["errors"]
    header, rows = read_table(dialogs["path"])
    norm = view._norm
    rows_shown = view._alpha_geom["rows"] if axes == "q" else np.arange(norm.frame_indices.size)
    nx = view._map.nx
    assert len(rows) == rows_shown.size * nx
    k = 5 * nx + 123
    r, x = rows_shown[k // nx], k % nx
    row = rows[k]
    assert int(row["Frame"]) == norm.frame_indices[r] and int(row["XPixel"]) == x
    assert float(row["Value"]) == norm.values[r, x]  # linear values, not the log display
    assert float(row["StdUncertainty"]) == norm.standard_uncertainty[r, x]
    if axes == "th":
        assert float(row["Theta (deg)"]) == view._variable("th")[norm.frame_indices[r]]
    if axes == "q":
        g = view._alpha_geom
        af = g["base"][k // nx] + alpha_f_offsets(nx, g["cal"])[x]
        qx, qz = q_from_angles(g["alpha_i"][k // nx], af, view.spin_wavelength.value())
        assert float(row["Qx (1/Angstrom)"]) == pytest.approx(qx, rel=1e-12)
        assert float(row["Qz (1/Angstrom)"]) == pytest.approx(qz, rel=1e-12)
        assert any("Wavelength" in line for line in header)


def test_offspecular_asymmetry_export_has_partner_frames(qtbot, window, dialogs, tmp_path):
    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    _ready(qtbot, view)
    view.combo_spin.setCurrentIndex(view.combo_spin.findData("asym"))
    header, columns, rows = view.export_table()
    assert columns[:3] == ["Frame", "PartnerFrame", "XPixel"]
    first = next(rows)
    assert first[1] == view._norm.partner_indices[0]
    assert any("not blanked" in line for line in header)


def test_offspecular_menu_export_before_map_opens_tab(window, dialogs):
    window.tabs.setCurrentWidget(window.view_charts)
    window.view_offspec._norm = None
    window.act_export_offspec_data.trigger()
    assert window.tabs.currentWidget() is window.view_offspec
    assert dialogs["info"]


def _plots(window):
    return [window.view_charts.plot_widget, window.view_detector.plot_2d, window.view_detector.plot_x_cut,
            window.view_offspec.plot, window.view_offspec.plot_row, window.view_hierarchy.plot_widget]


def test_right_click_export_is_guarded_on_every_plot(window, dialogs, tmp_path):
    digest = hashlib.sha256(EXAMPLE.read_bytes()).digest()
    for n, plot in enumerate(_plots(window)):
        actions = plot.scene().contextMenu
        assert [a.text() for a in actions] == ["Export Image…"]
        dialogs["path"] = EXAMPLE
        actions[0].trigger()
        assert len(dialogs["errors"]) == n + 1  # refused: source file
        dialogs["path"] = tmp_path / f"plot{n}.png"
        actions[0].trigger()
        assert (tmp_path / f"plot{n}.png").read_bytes().startswith(b"\x89PNG")
    assert hashlib.sha256(EXAMPLE.read_bytes()).digest() == digest


def test_toolbars_are_icon_only(window):
    charts = window.view_charts
    for btn in (charts.btn_pan, charts.btn_box_zoom, charts.btn_zoom_in, charts.btn_zoom_out,
                charts.btn_reset_view, charts.btn_tool_log_y, charts.btn_tool_log_x, charts.btn_export_plot):
        assert btn.text() == "" and btn.toolTip() and btn.size() == window.view_offspec.btn_fit.size()
    assert charts.btn_toggle_q.text() == "" and not charts.btn_toggle_q.icon().isNull()
    assert charts.btn_toggle_q.accessibleName() in ("X axis: θ", "X axis: Q")
    assert not hasattr(charts, "btn_export_csv")


def test_exports_refuse_inactive_loaded_files(window, dialogs):
    other = EXAMPLE.with_name("005594.nxs")
    window.load_file(other)  # becomes active; 005572 stays loaded
    digest = hashlib.sha256(EXAMPLE.read_bytes()).digest()
    dialogs["path"] = EXAMPLE.with_suffix(".nxs")
    for view in (window.view_detector, window.view_offspec, window.view_hierarchy):
        assert str(EXAMPLE.resolve()) in {str(Path(p).resolve()) for p in view.session_source_paths()}
    window.view_detector.export_image()
    window.view_hierarchy.plot_widget.scene().contextMenu[0].trigger()
    assert len(dialogs["errors"]) == 2
    assert hashlib.sha256(EXAMPLE.read_bytes()).digest() == digest
