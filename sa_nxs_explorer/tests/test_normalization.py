import numpy as np
import pytest
from scipy.stats import binom, poisson
from sa_nxs_explorer.core.normalization import (
    CONFIDENCE, normalize_signal, poisson_interval, poisson_ratio_interval,
    log_error_bounds, is_count_channel,
)
from sa_nxs_explorer.core.scan_calculations import compute_normalized_y


def test_zero_count_has_exact_upper_limit_and_separate_zero_variance_estimate():
    result = normalize_signal([0, 1, 100])
    assert result.values[0] == 0
    assert result.sigma[0] == 0  # plug-in variance estimate, not a confidence limit
    assert result.lower[0] == 0
    assert result.upper[0] == pytest.approx(-np.log((1 - CONFIDENCE) / 2))
    assert result.upper[0] == pytest.approx(1.841021645, rel=1e-8)
    assert "Garwood" in result.method


def test_time_rate_interval_scales_by_known_exposure():
    raw = normalize_signal([0, 100])
    rate = normalize_signal([0, 100], "time", time=[10, 20])
    np.testing.assert_allclose(rate.values, [0, 5])
    np.testing.assert_allclose(rate.sigma, [0, 0.5])
    np.testing.assert_allclose(rate.lower, raw.lower / [10, 20])
    np.testing.assert_allclose(rate.upper, raw.upper / [10, 20])


def test_monitor_ratio_propagation_includes_monitor_and_no_hidden_scale():
    result = normalize_signal([100, 200], "monitor", monitor=[1000, 2000], uncertainty="standard")
    np.testing.assert_allclose(result.values, [0.1, 0.1])
    expected_variance = np.array([100, 200]) / np.array([1000, 2000])**2 + np.array([100, 200])**2 / np.array([1000, 2000])**3
    np.testing.assert_allclose(result.sigma**2, expected_variance)


def test_zero_numerator_ratio_exact_upper_limit():
    result = normalize_signal([0], "monitor", monitor=[10])
    assert result.lower[0] == 0
    expected = ((1 - CONFIDENCE) / 2)**(-1 / 10) - 1
    assert result.upper[0] == pytest.approx(expected)
    assert result.upper[0] > 0


def test_conditional_ratio_coverage_at_fixed_total():
    # Independent reference check: exact limits must contain the true odds
    # for >= nominal binomial probability at each chosen parameter value.
    total = 25
    n = np.arange(total)  # M=0 has no finite measured ratio and is excluded
    lo, hi = poisson_ratio_interval(n, total - n)
    for p in [0.05, 0.2, 0.5, 0.8]:
        odds = p / (1 - p)
        covered = (lo <= odds) & (odds <= hi)
        probability = np.sum(binom.pmf(n[covered], total, p))
        # Include M=0 in the confidence procedure as [lower, infinity].
        lower_all_n = (1 - CONFIDENCE) / 2
        if p**total >= lower_all_n:
            probability += p**total
        assert probability >= CONFIDENCE - 1e-12


def test_garwood_coverage_for_low_count_means():
    observations = np.arange(100)
    lo, hi = poisson_interval(observations)
    for mean in [0.1, 0.5, 1, 3, 10]:
        contained = (lo <= mean) & (mean <= hi)
        assert np.sum(poisson.pmf(observations[contained], mean)) >= CONFIDENCE


def test_fixed_rate_is_shared_and_does_not_depend_on_scan_composition():
    first = normalize_signal([10], "monitor_time", monitor=[100], reference_monitor_rate=250)
    second = normalize_signal([10, 999], "monitor_time", monitor=[100, 1e6], reference_monitor_rate=250)
    assert first.values[0] == second.values[0] == 25
    assert first.lower[0] == second.lower[0]
    assert first.upper[0] == second.upper[0]


@pytest.mark.parametrize("method", ["exact", "standard"])
def test_shared_monitor_covariance_cancels(method):
    result = normalize_signal([10, 50], "monitor", monitor=[10, 50], numerator_is_monitor=True, uncertainty=method)
    np.testing.assert_array_equal(result.values, [1, 1])
    np.testing.assert_array_equal(result.sigma, [0, 0])
    np.testing.assert_array_equal(result.lower, [1, 1])
    np.testing.assert_array_equal(result.upper, [1, 1])


@pytest.mark.parametrize("mode,kwargs", [
    ("monitor", {}), ("time", {}), ("monitor_time", {}),
    ("monitor", {"monitor": [10, 20]}),
    ("time", {"time": [[10]]}), ("unknown", {}),
])
def test_missing_or_mismatched_inputs_do_not_fall_back(mode, kwargs):
    with pytest.raises(ValueError):
        normalize_signal([10], mode, **kwargs)


@pytest.mark.parametrize("mode,key", [("monitor", "monitor"), ("time", "time")])
def test_bad_denominators_mask_values_and_uncertainties(mode, key):
    result = normalize_signal([10]*5, mode, **{key: [10, 0, -1, np.nan, np.inf]})
    np.testing.assert_array_equal(result.valid, [True, False, False, False, False])
    assert np.all(np.isnan(result.values[1:]))
    assert np.all(np.isnan(result.lower[1:]))
    assert all(result.reasons[1:])


def test_bad_poisson_counts_are_not_clamped_or_rounded():
    result = normalize_signal([10, -1, 0.5, np.nan, np.inf])
    np.testing.assert_array_equal(result.valid, [True, False, False, False, False])
    assert result.values[0] == 10


def test_noncount_variables_have_no_invented_uncertainty():
    result = normalize_signal([-1.2, 0, 300], count_data=False, input_units="K")
    assert result.sigma is None
    assert result.lower is None
    np.testing.assert_array_equal(result.valid, [True]*3)
    assert result.units == "K"
    with pytest.raises(ValueError, match="only for count channels"):
        normalize_signal([300], "monitor", count_data=False, monitor=[100])


def test_explicit_variance_uses_derivative_propagation_even_at_zero():
    result = normalize_signal([0, -2.5], "monitor", monitor=[10, 20], variance=[4, 9], monitor_variance=[1, 4])
    assert result.sigma[0] == pytest.approx(0.2)
    assert result.sigma[1] == pytest.approx(np.sqrt(9/400 + 2.5**2*4/20**4))
    assert "first order" in result.method
    assert "exact" not in result.method


@pytest.mark.parametrize("variance", [[-1], [np.nan], [np.inf]])
def test_invalid_variance_is_flagged(variance):
    result = normalize_signal([1], variance=variance)
    assert not result.valid[0]
    assert result.reasons[0] == "invalid signal variance"


def test_log_bounds_are_transformed_without_statistical_clipping():
    top, bottom = log_error_bounds([10, 10], [0, 0.001], [20, 100])
    np.testing.assert_allclose(top, [np.log10(2), 1])
    assert np.isnan(bottom[0])
    assert bottom[1] == pytest.approx(4)  # previously arbitrarily capped at 1.5


@pytest.mark.parametrize("rate", [0, -1, np.inf, np.nan])
def test_invalid_reference_rate(rate):
    with pytest.raises(ValueError):
        normalize_signal([10], "monitor_time", monitor=[100], reference_monitor_rate=rate)


def test_legacy_mean_monitor_scaling_requires_explicit_migration():
    with pytest.raises(ValueError, match="Implicit mean-monitor"):
        compute_normalized_y(np.array([10]), "monitor", np.array([100]), scale_monitor=True)


def test_count_channel_classification():
    assert is_count_channel("ROI_Counts", "counts")
    assert is_count_channel("Monitor1", "")
    assert not is_count_channel("th", "deg")
    assert not is_count_channel("LakeshoreMain", "K")
    assert not is_count_channel("Detector1_variance", "counts")
