from sa_nxs_explorer.ui.scan_plot_view import ScanPlotView
from sa_nxs_explorer.tests.test_processing_ui import scan


def test_data_panel_collapse_keeps_data_and_expands_chart(qtbot):
    view = ScanPlotView()
    qtbot.addWidget(view)
    view.resize(1200, 900)
    view.add_scan(scan())
    view.show()
    qtbot.wait(20)
    height = view.plot_widget.height()
    rows = view.curve_data_table.rowCount()
    view.chk_show_data_panel.setChecked(False)
    qtbot.wait(20)
    assert view.bottom_tabs.isHidden()
    assert view.plot_widget.height() > height
    assert view.curve_data_table.rowCount() == rows
    view.chk_show_data_panel.setChecked(True)
    assert not view.bottom_tabs.isHidden()


def test_axis_fonts_and_hidden_legend_survive_replot_and_theme(qtbot):
    view = ScanPlotView()
    qtbot.addWidget(view)
    view.add_scan(scan())
    view.spin_axis_font_size.setValue(18)
    view.chk_show_legend.setChecked(False)
    view.chk_log_y.setChecked(False)
    view.set_theme(True)
    assert not view.legend.isVisible()
    for name in ("left", "bottom"):
        axis = view.plot_widget.getAxis(name)
        assert axis.style["tickFont"].pointSizeF() == 18
        assert axis.labelStyle["font-size"] == "18pt"
    view.chk_show_legend.setChecked(True)
    assert view.legend.isVisible()


def test_axis_titles_keep_text_units_and_contrast(qtbot):
    view = ScanPlotView()
    qtbot.addWidget(view)
    view.add_scan(scan())
    original = {name: (view.plot_widget.getAxis(name).labelText,
                       view.plot_widget.getAxis(name).labelUnits)
                for name in ("left", "bottom")}
    for dark in (True, False, True):
        view.set_theme(dark)
        view.spin_axis_font_size.setValue(20 if dark else 12)
        view.chk_show_legend.setChecked(not dark)
        view.chk_show_data_panel.setChecked(not dark)
        view._update_plot()
        for name in ("left", "bottom"):
            axis = view.plot_widget.getAxis(name)
            assert axis.labelText
            assert (axis.labelText, axis.labelUnits) == original[name]
            assert axis.label.isVisible()
            assert axis.labelStyle["color"] == ("#f1f5f9" if dark else "#1e293b")
            assert axis.labelText in axis.label.toPlainText()
