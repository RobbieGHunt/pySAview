"""Unit tests for pySAview.core.overillumination cross-checked against pySAred 1.5.3."""

from pathlib import Path
import numpy as np
import pytest

from pySAview.core.nexus_reader import NexusReader
from pySAview.core.reduction.overillumination import (
    compute_overillumination_series,
    extract_slit_widths,
    overillumination_coeff,
)


def _verbatim_pysared_153(s1hg, s2hg, th, sample_len, d1, d2):
    """Exact copy of pySAred_V1.5.3.py lines 881-905 for numerical verification."""
    if s1hg < s2hg:
        OB = abs(((d1 * (s2hg - s1hg)) / (2 * (d1 - d2))) + s1hg / 2)
        OC = ((d1 * (s2hg + s1hg)) / (2 * (d1 - d2))) - s1hg / 2
    elif s1hg > s2hg:
        OB = abs(((s2hg * d1) - (s1hg * d2)) / (2 * (d1 - d2)))
        OC = (d1 / (d1 - d2)) * (s2hg + s1hg) / 2 - (s1hg / 2)
    else:
        OB = s1hg / 2
        OC = s1hg * (d1 / (d1 - d2) - 1 / 2)

    BC = OC - OB
    AO = 1 / (BC / 2 + OB)
    FWHM_beam = BC / 2 + OB
    sampleLen_relative = sample_len * np.sin(np.radians(np.fabs(th if not th == 0 else 0.00001)))

    if sampleLen_relative / 2 >= OC:
        c0 = 1.0
    elif sampleLen_relative / 2 <= OB:
        c0 = AO * sampleLen_relative / 2
    else:
        c0 = AO * (OB + BC / 2 - ((OC - sampleLen_relative / 2) ** 2) / (2 * BC))

    fwhm_proj = s2hg if sampleLen_relative / 2 >= FWHM_beam else sampleLen_relative
    return float(c0), float(fwhm_proj)


@pytest.mark.parametrize("s1,s2", [(1.0, 1.0), (0.5, 1.2), (1.5, 0.4), (0.2, 0.2), (2.0, 0.5)])
@pytest.mark.parametrize("th", [0.0, 0.01, 0.1, 0.35, 0.8, 1.5, 3.0, 6.0, -0.25])
@pytest.mark.parametrize("sample_len", [10.0, 20.0, 50.0])
def test_matches_verbatim_pysared_153(s1, s2, th, sample_len):
    d1, d2 = 2300.0, 290.0
    got_c, got_f = overillumination_coeff(s1, s2, th, sample_len, d1, d2)
    ref_c, ref_f = _verbatim_pysared_153(s1, s2, th, sample_len, d1, d2)
    assert got_c == pytest.approx(ref_c, rel=1e-14, abs=1e-15)
    assert got_f == pytest.approx(ref_f, rel=1e-14, abs=1e-15)
    assert 0.0 < got_c <= 1.0


def test_overillumination_physical_limits():
    # High angle -> full beam captured (coeff == 1, fwhm_proj == s2)
    c_hi, f_hi = overillumination_coeff(1.0, 1.0, 10.0, sample_length_mm=50.0)
    assert c_hi == 1.0
    assert f_hi == 1.0

    # Small angle -> coeff increases monotonically with angle
    angles = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    coeffs = [overillumination_coeff(1.0, 1.0, a, sample_length_mm=20.0)[0] for a in angles]
    assert all(a <= b for a, b in zip(coeffs, coeffs[1:]))
    assert coeffs[0] < 0.2 and coeffs[-1] == 1.0


@pytest.mark.parametrize("kwargs", [
    dict(s1hg=0.0),
    dict(s2hg=-0.5),
    dict(sample_length_mm=0.0),
    dict(distance_s1_mm=290.0, distance_s2_mm=290.0),
    dict(distance_s1_mm=200.0, distance_s2_mm=290.0),
    dict(theta_deg=np.nan),
])
def test_invalid_inputs_rejected(kwargs):
    params = dict(s1hg=1.0, s2hg=1.0, theta_deg=1.0, sample_length_mm=50.0, distance_s1_mm=2300.0, distance_s2_mm=290.0)
    params.update(kwargs)
    with pytest.raises(ValueError):
        overillumination_coeff(**params)


def test_series_flags_zero_slits_from_unwritten_frames():
    s1 = np.array([1.0, 1.0, 0.0, np.nan])
    s2 = np.array([1.0, 1.0, 0.0, 1.0])
    th = np.array([0.2, 2.0, 0.0, 1.0])
    c, f, valid = compute_overillumination_series(s1, s2, th, sample_length_mm=20.0)
    np.testing.assert_array_equal(valid, [True, True, False, False])
    assert np.all(np.isfinite(c[:2]))
    assert np.all(np.isnan(c[2:]))
    assert np.all(np.isnan(f[2:]))


def test_extract_slits_from_example_nomad_scans():
    example_dir = Path(__file__).resolve().parent.parent / "example_data"
    for name in ("005572.nxs", "005594.nxs"):
        with NexusReader(example_dir / name) as reader:
            scan = reader.parse_scan()
        n = len(scan.variables["th"].data)
        s1, s2, src = extract_slit_widths(scan, n)
        assert s1 is not None and s2 is not None
        assert "s1h" in src
        assert scan.slit_defaults == (1.0, 1.0)
        assert s1[0] == 1.0 and s2[0] == 1.0
