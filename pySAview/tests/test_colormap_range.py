import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex
from PyQt6.QtCore import QPoint, Qt
from pySAview.core.curve_colors import spin_palette
from pySAview.tests.test_processing_ui import scan
from pySAview.ui.scan_plot_view import ScanPlotView


def test_full_range_and_even_sampling():
    for i in range(5):
        p = spin_palette(i, 5)
        assert p["up"] == to_hex(colormaps["Reds"](i/4))
        p = spin_palette(i, 5, shade_range=(.2, .8))
        assert p["down"] == to_hex(colormaps["Blues"](.2 + .6*i/4))


def test_drag_handles_shared_and_data_unchanged(qtbot):
    view = ScanPlotView()
    qtbot.addWidget(view)
    view.add_scan(scan("1.nxs"))
    before = [c.y.copy() for c in view._current_curves]
    down = view.spin_style_controls["d"]["preview"]
    up = view.spin_style_controls["u"]["preview"]
    qtbot.addWidget(down)
    down.resize(215, 34)
    down.show()
    qtbot.mousePress(down, Qt.MouseButton.LeftButton, pos=QPoint(7, 22))
    qtbot.mouseMove(down, QPoint(57, 22))
    qtbot.mouseRelease(down, Qt.MouseButton.LeftButton, pos=QPoint(57, 22))
    assert 0 < down.low < down.high == 1
    assert (up.low, up.high) == (down.low, down.high) == view.shade_range
    view.combo_up_colormap.setCurrentText("viridis")
    view.set_theme(True)
    assert (up.low, up.high) == view.shade_range
    assert up.colormap == "viridis"
    for c, original in zip(view._current_curves, before):
        np.testing.assert_array_equal(c.y, original)
