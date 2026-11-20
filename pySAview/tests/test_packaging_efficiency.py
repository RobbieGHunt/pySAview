"""Regression tests for Matplotlib-free colormaps, scipy.special parity, bundle paths, and smoke-test safety."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from matplotlib import colormaps as mpl_colormaps
from matplotlib.colors import to_hex as mpl_to_hex
from scipy.stats import beta as stats_beta
from scipy.stats import chi2 as stats_chi2

from pySAview.core.colormaps import colormaps, to_hex
from pySAview.core.normalization import poisson_interval, poisson_ratio_interval
from pySAview.core.paths import example_data_dir, package_root, resources_dir


def test_prebaked_colormaps_match_matplotlib_exactly():
    assert sorted(colormaps) == sorted(mpl_colormaps)
    sample_points = (0.0, 0.2, 0.25, 0.5, 0.75, 0.8, 1.0)
    for name in ("Blues", "Reds", "Greens", "Purples_r", "RdBu_r", "viridis", "tab10", "jet_r"):
        lut = colormaps[name]
        mpl = mpl_colormaps[name]
        for x in sample_points:
            assert to_hex(lut(x)) == mpl_to_hex(mpl(x))
            assert lut.to_hex(x) == mpl_to_hex(mpl(x))


def test_scipy_special_intervals_match_scipy_stats_reference():
    counts = np.array([0, 1, 5, 42, 1000], dtype=float)
    monitors = np.array([10, 25, 100, 500, 10000], dtype=float)
    for conf in (0.6826894921370859, 0.9544997361036416, 0.9973002039367398):
        alpha = 1.0 - conf
        lo, hi = poisson_interval(counts, confidence=conf)
        exp_lo = np.zeros_like(counts)
        pos = counts > 0
        exp_lo[pos] = stats_chi2.ppf(alpha / 2, 2 * counts[pos]) / 2
        exp_hi = stats_chi2.isf(alpha / 2, 2 * (counts + 1)) / 2
        np.testing.assert_allclose(lo, exp_lo, rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(hi, exp_hi, rtol=1e-14, atol=0.0)

        r_lo, r_hi = poisson_ratio_interval(counts, monitors, confidence=conf)
        exp_r_lo = np.zeros_like(counts)
        p_low = stats_beta.ppf(alpha / 2, counts[pos], monitors[pos] + 1)
        exp_r_lo[pos] = p_low / (1 - p_low)
        q_low = stats_beta.ppf(alpha / 2, monitors, counts + 1)
        exp_r_hi = (1 - q_low) / q_low
        np.testing.assert_allclose(r_lo, exp_r_lo, rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(r_hi, exp_r_hi, rtol=1e-14, atol=0.0)


def test_bundle_paths_resolve_source_and_meipass(monkeypatch, tmp_path):
    assert (resources_dir() / "colormaps.npz").is_file()
    assert (resources_dir() / "chevron_right.svg").is_file()
    assert (example_data_dir() / "005572.nxs").is_file()
    assert (example_data_dir() / "005594.nxs").is_file()

    fake_pkg = tmp_path / "pySAview"
    fake_pkg.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert package_root() == fake_pkg
    assert resources_dir() == fake_pkg / "resources"
    assert example_data_dir() == fake_pkg / "example_data"


def test_smoke_test_cli_succeeds_and_guards_source_paths(tmp_path):
    report = tmp_path / "smoke.json"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    proc = subprocess.run(
        [sys.executable, "-m", "pySAview", "--smoke-test", str(report)],
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["ok"] is True
    assert data["svg_ok"] is True
    assert data["example_data_ok"] is True
    assert data["pre_import_clean"] is True
    assert data["exact_zero_upper"] == pytest.approx(3.783184, rel=1e-5)

    # Source-path safety guard must refuse writing to .nxs paths
    bad_target = tmp_path / "forbidden.nxs"
    bad = subprocess.run(
        [sys.executable, "-m", "pySAview", "--smoke-test", str(bad_target)],
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
        check=False,
    )
    assert bad.returncode != 0
    assert not bad_target.exists()
