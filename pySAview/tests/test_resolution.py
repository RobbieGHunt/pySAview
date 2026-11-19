"""Unit tests for pySAview.core.resolution cross-checked against pySAred 1.5.3."""

import numpy as np
import pytest

from pySAview.core.reduction.overillumination import overillumination_coeff
from pySAview.core.reduction.resolution import compute_resolution_pysared


def _verbatim_pysared_resolution(th, s1, s2, fwhm_proj, lam, dlam, d1, d2, like_sared, angle_unit):
    qz = (4.0 * np.pi / lam) * np.sin(np.radians(th))
    if like_sared:
        res = np.sqrt(
            ((2 * np.pi / lam) ** 2) * ((np.cos(np.radians(th))) ** 2) * (0.68 ** 2) * ((s1 ** 2) + (s2 ** 2)) / ((d1 - d2) ** 2)
            + ((dlam ** 2) * (qz ** 2))
        )
    else:
        d_alpha = np.arctan((s1 + fwhm_proj) / ((d1 - d2) * 2))
        if angle_unit == "Qz":
            k0 = 2 * np.pi / lam
            res = np.sqrt((k0 ** 2) * ((((np.cos(np.radians(th))) ** 2) * d_alpha ** 2) + ((dlam ** 2) * ((np.sin(np.radians(th))) ** 2))))
        else:
            res = d_alpha if angle_unit == "Radians" else np.degrees(d_alpha)
    return res / (2 * np.sqrt(2 * np.log(2)))


@pytest.mark.parametrize("mode", ["pysared", "sared"])
@pytest.mark.parametrize("angle_unit", ["Qz", "Degrees", "Radians"])
def test_matches_verbatim_pysared_153(mode, angle_unit):
    lam, dlam, d1, d2 = 5.183, 0.004, 2300.0, 290.0
    th = np.array([0.1, 0.5, 1.5, 4.0])
    s1 = np.array([0.5, 1.0, 1.0, 2.0])
    s2 = np.array([0.5, 1.0, 0.8, 1.5])
    fp = np.array([overillumination_coeff(a, b, t, 20.0, d1, d2)[1] for a, b, t in zip(s1, s2, th)])

    got = compute_resolution_pysared(th, s1, s2, fp, lam, dlam, d1, d2, mode=mode, angle_unit=angle_unit)
    ref = np.array([
        _verbatim_pysared_resolution(t, a, b, f, lam, dlam, d1, d2, mode == "sared", angle_unit)
        for t, a, b, f in zip(th, s1, s2, fp)
    ])
    np.testing.assert_allclose(got, ref, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("kwargs", [
    dict(mode="invalid"),
    dict(angle_unit="km"),
    dict(wavelength_angstrom=0.0),
    dict(wavelength_resolution=-0.01),
    dict(distance_s1_mm=200.0, distance_s2_mm=290.0),
])
def test_invalid_resolution_parameters_rejected(kwargs):
    base = dict(
        theta_deg=[1.0], s1_mm=[1.0], s2_mm=[1.0], fwhm_proj_mm=[1.0],
        wavelength_angstrom=5.183, wavelength_resolution=0.004,
        distance_s1_mm=2300.0, distance_s2_mm=290.0,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        compute_resolution_pysared(**base)
