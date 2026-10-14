import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex, to_rgb
from sa_nxs_explorer.core.curve_colors import spin_palette
from sa_nxs_explorer.tests.test_processing_ui import scan
from sa_nxs_explorer.core.models import ScanVariable
from sa_nxs_explorer.ui.scan_plot_view import ScanPlotView


def test_default_spin_colors_follow_superadam_convention():
    palette = spin_palette()
    assert palette["up"] == to_hex(colormaps["Reds"](0.65))
    assert palette["down"] == to_hex(colormaps["Blues"](0.65))
    for dark in [False, True]:
        shades = [spin_palette(i, 12, dark=dark) for i in range(12)]
        assert len({p["up"] for p in shades}) == 12
        assert len({p["down"] for p in shades}) == 12


def test_overlay_palette_changes_preserve_data_and_visibility_identity(qtbot):
    view = ScanPlotView()
    qtbot.addWidget(view)
    for name in ("first.nxs", "second.nxs", "third.nxs"):
        data = scan(name)
        data.variables["Flipper1"] = ScanVariable(3, "Flipper1", "F1", "", "", 0, False, np.array([0, 1, 0]))
        view.add_scan(data)
    assert view.combo_down_colormap.currentText() == "Blues"
    assert view.combo_up_colormap.currentText() == "Reds"
    before = {c.source_path + c.label.split("]")[-1]: (c.color, c.y.copy(), c.y_lower.copy(), c.symbol) for c in view._current_curves}
    for c in view._current_curves:
        rgb = to_rgb(c.color)
        if "d (OFF)" in c.label:
            assert rgb[2] > rgb[0]
            assert np.all(c.indices % 2 == 0)
        else:
            assert rgb[0] > rgb[2]
            assert np.all(c.indices % 2 == 1)
    view.set_overlay_keys(["second.nxs", "third.nxs"])
    for c in view._current_curves:
        original = before[c.source_path + c.label.split("]")[-1]]
        assert c.color == original[0]
        assert c.symbol == original[3]
    down_colors = [c.color for c in view._current_curves if "d (OFF)" in c.label]
    view.combo_up_colormap.setCurrentText("Greens")
    assert [c.color for c in view._current_curves if "d (OFF)" in c.label] == down_colors
    for c in view._current_curves:
        original = before[c.source_path + c.label.split("]")[-1]]
        np.testing.assert_array_equal(c.y, original[1])
        np.testing.assert_array_equal(c.y_lower, original[2])
        if "u (ON)" in c.label:
            assert to_rgb(c.color)[1] > to_rgb(c.color)[0]
    view.combo_down_colormap.setCurrentText("Purples_r")
    assert "Purples_r" == view.combo_down_colormap.currentText()
