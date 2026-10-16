import csv
import numpy as np
import pytest
from scipy.stats import norm, poisson
from pySAview.core.normalization import confidence_from_sigma, normalize_signal, poisson_interval
from pySAview.tests.test_processing_ui import scan
from pySAview.ui.scan_plot_view import ScanPlotView, QFileDialog, QMessageBox


@pytest.mark.parametrize("level", range(1, 7))
def test_exact_coverage_zero_limit_and_normalized_scaling(level):
    confidence = norm.cdf(level) - norm.cdf(-level)
    assert confidence_from_sigma(level) == pytest.approx(confidence, abs=2e-16)
    raw = normalize_signal([0, 10, 100], sigma_level=level)
    assert raw.lower[0] == 0
    assert raw.upper[0] == pytest.approx(-np.log((1-confidence_from_sigma(level))/2))
    rate = normalize_signal([0, 10, 100], "time", time=[2, 5, 10], sigma_level=level)
    np.testing.assert_allclose(rate.lower, raw.lower/[2, 5, 10])
    np.testing.assert_allclose(rate.upper, raw.upper/[2, 5, 10])
    ratio = normalize_signal([0], "monitor", monitor=[10], sigma_level=level)
    expected = ((1-confidence_from_sigma(level))/2)**(-.1)-1
    assert ratio.upper[0] == pytest.approx(expected)
    assert f"{level} sigma equivalent" in ratio.method
    observations = np.arange(200)
    lo, hi = poisson_interval(observations, confidence_from_sigma(level))
    for mean in [.1, 1, 10]:
        covered = (lo <= mean) & (mean <= hi)
        assert poisson.pmf(observations[covered], mean).sum() >= confidence_from_sigma(level)-1e-14


@pytest.mark.parametrize("level", [0, -1, 7, np.inf, np.nan])
def test_invalid_sigma_rejected(level):
    with pytest.raises(ValueError):
        normalize_signal([1], sigma_level=level)


def test_sigma_changes_exact_bounds_not_values_or_standard_uncertainties(qtbot, monkeypatch, tmp_path):
    view = ScanPlotView()
    qtbot.addWidget(view)
    view.add_scan(scan())
    assert view.combo_sigma.currentData() == 2
    assert view.bottom_tabs.count() == 1
    assert view.bottom_tabs.tabText(0) == "Plotted Data Points"
    assert not hasattr(view, "stats_table")
    original = view._current_curves[0]
    view.combo_sigma.setCurrentIndex(view.combo_sigma.findData(3))
    changed = view._current_curves[0]
    np.testing.assert_array_equal(changed.y, original.y)
    np.testing.assert_array_equal(changed.y_err, original.y_err)
    assert np.all(changed.y_upper > original.y_upper)
    assert "3 sigma equivalent" in view.lbl_processing.text()
    target = tmp_path / "plot.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(target), "CSV"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a: None)
    view._export_csv()
    with target.open(newline="",encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert "3 sigma equivalent" in rows[0]["UncertaintyMethod"]
    assert float(rows[0]["UpperBound"]) == changed.y_upper[0]
    view.combo_uncertainty.setCurrentIndex(view.combo_uncertainty.findData("standard"))
    assert view.combo_sigma.isEnabled()
    standard = view._current_curves[0]
    np.testing.assert_allclose(standard.y_upper, standard.y+3*standard.y_err)


@pytest.mark.parametrize("level", range(1,7))
def test_propagated_bounds_scale_without_changing_standard_uncertainty(level):
    result = normalize_signal([10.,20.], variance=[4.,9.], uncertainty="standard", sigma_level=level)
    np.testing.assert_allclose(result.sigma, [2.,3.])
    np.testing.assert_allclose(result.lower, result.values-level*result.sigma)
    np.testing.assert_allclose(result.upper, result.values+level*result.sigma)
    assert f"{level} sigma bounds" in result.method
    default = normalize_signal([10.], variance=[4.], uncertainty="standard")
    assert default.lower[0] == 6.
    assert default.upper[0] == 14.
