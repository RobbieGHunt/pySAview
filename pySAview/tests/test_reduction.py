"""End-to-end unit tests for pySAview.core.reduction and pySAred .dat export."""

import numpy as np
import pytest

from pySAview.core.export_tables import write_pysared_dat
from pySAview.core.reduction.overillumination import overillumination_coeff
from pySAview.core.reduction.pysared import ReductionParams, reduce_counts_pysared, trim_curve_mask
from pySAview.core.scan_calculations import PlottedCurve


def test_pysared_zero_count_and_monitor_error_convention():
    res = reduce_counts_pysared(
        [0.0, 100.0, 400.0],
        mode="monitor",
        monitor=[1000.0, 1000.0, 2000.0],
        sigma_level=1.0,
    )
    # Zero counts get IntensErr = 1 -> 1 / 1000 after monitor division
    assert res.values[0] == 0.0
    assert res.sigma[0] == pytest.approx(1.0 / 1000.0)
    # Non-zero matches (I/M)*sqrt((sqrt(I)/I)^2 + 1/M)
    assert res.values[1] == pytest.approx(0.1)
    assert res.sigma[1] == pytest.approx(0.1 * np.sqrt((10.0 / 100.0) ** 2 + 1.0 / 1000.0))
    assert res.values[2] == pytest.approx(0.2)
    assert res.sigma[2] == pytest.approx(0.2 * np.sqrt((20.0 / 400.0) ** 2 + 1.0 / 2000.0))


def test_full_pysared_pipeline_order_matches_step_by_step():
    # Raw counts, BKG, monitor, overillumination, scale factor, theta offset, resolution
    params = ReductionParams(
        theta_offset_deg=0.05,
        subtract_background=True,
        bkg_skip_qz=0.01,
        overillumination=True,
        sample_length_mm=15.0,
        use_scale_factor=True,
        scale_factor=2.5,
        resolution_mode="pysared",
        include_resolution=True,
    )
    th_raw = np.array([0.15, 0.55, 2.05])
    th_corr = th_raw - 0.05  # [0.10, 0.50, 2.00]
    lam = 5.183
    qz = (4.0 * np.pi / lam) * np.sin(np.radians(th_corr))
    counts = np.array([1000.0, 500.0, 100.0])
    bkg = np.array([20.0, 25.0, 10.0])
    mon = np.array([2000.0, 2000.0, 2000.0])
    s1 = np.array([1.0, 1.0, 1.0])
    s2 = np.array([1.0, 1.0, 1.0])

    res = reduce_counts_pysared(
        counts, mode="monitor", monitor=mon,
        theta_deg=th_raw, wavelength_angstrom=lam,
        s1_mm=s1, s2_mm=s2, bkg_counts=bkg, params=params,
    )

    # Verify each point directly against verbatim pySAred 1.5.3 order
    for i in range(3):
        c0, _ = overillumination_coeff(1.0, 1.0, round(th_corr[i], 4), 15.0, 2300.0, 290.0)
        I = counts[i]
        dI = np.sqrt(I)
        if qz[i] > 0.01 and bkg[i] > 0:
            dI = np.sqrt(I + bkg[i])
            I = I - bkg[i]
        dI = (I / mon[i]) * np.sqrt((dI / I) ** 2 + 1.0 / mon[i])
        I = I / mon[i]
        I /= c0
        dI /= c0
        I /= 2.5
        dI /= 2.5
        assert res.values[i] == pytest.approx(I, rel=1e-12)
        assert res.sigma[i] == pytest.approx(dI, rel=1e-12)
    assert res.resolution is not None and np.all(res.resolution > 0)


def test_negative_net_intensity_after_bkg_keeps_positive_uncertainty():
    params = ReductionParams(subtract_background=True, bkg_skip_qz=0.0)
    res = reduce_counts_pysared(
        [5.0], mode="monitor", monitor=[100.0],
        theta_deg=[1.0], wavelength_angstrom=5.183, bkg_counts=[20.0], params=params,
    )
    assert res.values[0] == pytest.approx(-15.0 / 100.0)
    assert res.sigma[0] > 0.0
    assert res.sigma[0] == pytest.approx(np.sqrt((25.0 / 100.0**2) + (15.0**2 / 100.0**3)))


def test_reduction_params_clone_is_independent():
    a = ReductionParams(overillumination=True, sample_length_mm=12.0, curvature_rad=0.08, bkg_x_bounds=(298.0, 326.0))
    b = a.clone()
    b.sample_length_mm = 40.0
    b.curvature_rad = 0.0
    assert a.sample_length_mm == 12.0 and a.curvature_rad == 0.08


def test_trim_curve_mask_skips_ends_and_zeros():
    y = np.array([1.0, 0.0, 5.0, 3.0, 0.0, 2.0])
    keep = trim_curve_mask(len(y), y, skip_left=1, skip_right=1, remove_zeros=True)
    np.testing.assert_array_equal(keep, [False, False, True, True, False, False])


def test_write_pysared_dat_formats_and_guards_source(tmp_path):
    curve = PlottedCurve(
        label="u (ON)",
        x=np.array([0.01, 0.02, 0.03]),
        y=np.array([0.5, 0.0, -0.1]),
        y_err=np.array([0.05, 0.01, 0.02]),
        color="#dc2626",
        indices=np.array([0, 1, 2]),
        wavelength=5.183,
        x_units="Å⁻¹",
        resolution=np.array([0.0004, 0.0005, 0.0006]),
        theta_deg=np.array([0.236, 0.472, 0.708]),
    )
    source = tmp_path / "scan.nxs"
    source.write_bytes(b"raw")

    with pytest.raises(ValueError):
        write_pysared_dat(source, [curve], source_paths=[str(source)])

    written = write_pysared_dat(
        tmp_path / "out.dat", [curve], angle_mode="Qz",
        include_resolution=True, remove_zeros=True, exclude_negative=True,
        wavelength=5.183, source_paths=[str(source)],
    )
    assert len(written) == 1
    out_path, count = written[0]
    assert count == 1  # zero and negative excluded
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    cols = lines[0].split()
    assert len(cols) == 4
    assert float(cols[1]) == pytest.approx(0.5)
    assert float(cols[2]) == pytest.approx(0.05)
    assert float(cols[3]) == pytest.approx(0.0004)
