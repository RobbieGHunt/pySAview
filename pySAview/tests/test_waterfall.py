import numpy as np
import pyqtgraph as pg
from pySAview.core.waterfall import waterfall_order, display_values
from pySAview.tests.test_processing_ui import scan
from pySAview.core.models import ScanVariable
from pySAview.ui.scan_plot_view import ScanPlotView


def test_scan_number_order_and_unnumbered_load_order():
    assert waterfall_order(["z.nxs", "0010.nxs", "2.nxs", "a.nxs", "3.NXS"]) == ["2.nxs", "3.NXS", "0010.nxs", "z.nxs", "a.nxs"]


def test_display_transform_preserves_inputs_and_bounds():
    values = np.array([0., 1., 3.])
    np.testing.assert_array_equal(display_values(values, 2, 1, "log"), [0, 100, 300])
    np.testing.assert_array_equal(display_values(values, 2, 5, "linear"), [10, 11, 13])
    np.testing.assert_array_equal(values, [0, 1, 3])


def test_waterfall_pairs_intervals_and_original_data(qtbot):
    view = ScanPlotView()
    qtbot.addWidget(view)
    for name in ("10.nxs", "2.nxs"):
        data = scan(name, detector=(10, 100, 200))
        data.variables["Flipper1"] = ScanVariable(3, "Flipper1", "F1", "", "", 0, False, np.array([0, 1, 0]))
        view.add_scan(data)
    original = [(c.y.copy(), c.y_lower.copy(), c.y_upper.copy()) for c in view._current_curves]
    view.chk_waterfall.setChecked(True)
    assert view.chk_log_y.isChecked()
    plots = view.plot_widget.listDataItems()
    bars = [i for i in view.plot_widget.items() if isinstance(i, pg.ErrorBarItem)]
    assert len(bars) == len(plots) == 4
    for curve, item, bar, saved in zip(view._current_curves, plots, bars, original):
        rank = 1 if curve.source_path == "10.nxs" else 0
        np.testing.assert_allclose(item.yData, curve.y * 10**rank)
        bar = next(b for b in bars if np.array_equal(b.opts["x"], curve.x) and np.allclose(b.opts["y"], np.log10(curve.y) + rank))
        np.testing.assert_allclose(bar.opts["top"], np.log10(curve.y_upper / curve.y))
        np.testing.assert_array_equal(curve.y, saved[0])
        np.testing.assert_array_equal(curve.y_lower, saved[1])
    view.combo_waterfall_mode.setCurrentIndex(1)
    view.spin_waterfall_spacing.setValue(7)
    assert not view.chk_log_y.isChecked()
    plots = view.plot_widget.listDataItems()
    bars = [i for i in view.plot_widget.items() if isinstance(i, pg.ErrorBarItem)]
    for curve, item, bar in zip(view._current_curves, plots, bars):
        rank = 1 if curve.source_path == "10.nxs" else 0
        np.testing.assert_allclose(item.yData, curve.y + 7 * rank)
        bar = next(b for b in bars if np.array_equal(b.opts["x"], curve.x) and np.allclose(b.opts["y"], curve.y + 7 * rank))
        np.testing.assert_allclose(bar.opts["top"], curve.y_upper - curve.y)
        np.testing.assert_allclose(bar.opts["y"], curve.y + 7 * rank)
    view.chk_waterfall.setChecked(False)
    for curve, item in zip(view._current_curves, view.plot_widget.listDataItems()):
        np.testing.assert_array_equal(item.yData, curve.y)
