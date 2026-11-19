"""Angular and momentum-transfer resolution calculations following pySAred 1.5.3.

Implements both resolution models from Alexey Klechikov's ``pySAred_V1.5.3.py``
(lines 754-770 and 1294-1306):

1. **pySAred default** (``mode="pysared"``): uses the projected beam width
   ``fwhm_proj`` returned by ``pySAview.core.reduction.overillumination``::

       d_alpha = arctan((s1hg + fwhm_proj) / (2 * (D1 - D2)))
       Qz mode:  Res_FWHM = sqrt(k0^2 * (cos^2(th) * d_alpha^2 + (dlam/lam)^2 * sin^2(th)))
       Radians:  Res_FWHM = d_alpha
       Degrees:  Res_FWHM = degrees(d_alpha)

2. **Original Gunnar 'Sared' formula** (``mode="sared"``)::

       Res_FWHM = sqrt(
           k0^2 * cos^2(th) * 0.68^2 * (s1hg^2 + s2hg^2) / (D1 - D2)^2
           + (dlam/lam)^2 * Qz^2
       )

Both formulas convert FWHM to Gaussian standard deviation ``sigma``::

    Resolution = Res_FWHM / (2 * sqrt(2 * ln(2)))
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from pySAview.core.reduction.overillumination import (
    DEFAULT_DISTANCE_S1_TO_SAMPLE_MM,
    DEFAULT_DISTANCE_S2_TO_SAMPLE_MM,
)

__all__ = [
    "DEFAULT_WAVELENGTH_RESOLUTION",
    "FWHM_TO_SIGMA",
    "compute_resolution_pysared",
]

DEFAULT_WAVELENGTH_RESOLUTION: float = 0.004
FWHM_TO_SIGMA: float = 1.0 / (2.0 * float(np.sqrt(2.0 * np.log(2.0))))


def compute_resolution_pysared(
    theta_deg: np.ndarray,
    s1_mm: np.ndarray,
    s2_mm: np.ndarray,
    fwhm_proj_mm: np.ndarray,
    wavelength_angstrom: float,
    wavelength_resolution: float = DEFAULT_WAVELENGTH_RESOLUTION,
    distance_s1_mm: float = DEFAULT_DISTANCE_S1_TO_SAMPLE_MM,
    distance_s2_mm: float = DEFAULT_DISTANCE_S2_TO_SAMPLE_MM,
    *,
    mode: Literal["pysared", "sared"] = "pysared",
    angle_unit: Literal["Qz", "Degrees", "Radians"] = "Qz",
) -> np.ndarray:
    """Compute Gaussian 1-sigma resolution array matching pySAred 1.5.3.

    Parameters
    ----------
    theta_deg : array-like
        Offset-corrected incident angle ``theta - theta_offset`` in degrees.
    s1_mm, s2_mm : array-like
        Horizontal slit widths ``s1h``, ``s2h`` in mm.
    fwhm_proj_mm : array-like
        Projected FWHM width in mm from ``overillumination_coeff`` (or ``s2_mm``
        when not overilluminated).
    wavelength_angstrom : float
        Neutron wavelength ``lambda`` in Angstroms (must be > 0).
    wavelength_resolution : float
        Fractional wavelength spread ``d_lambda / lambda`` (default ``0.004``).
    distance_s1_mm, distance_s2_mm : float
        Distances from S1 and S2 slits to the sample in mm (defaults ``2300``, ``290``).
    mode : {"pysared", "sared"}
        ``"pysared"`` uses ``fwhm_proj_mm``; ``"sared"`` uses Gunnar's ``0.68`` formula.
    angle_unit : {"Qz", "Degrees", "Radians"}
        Output coordinate unit (``"sared"`` mode always returns ``Qz`` resolution in
        inverse Angstroms, matching pySAred 1.5.3).
    """
    if mode not in ("pysared", "sared"):
        raise ValueError(f"Unknown resolution mode: {mode!r}")
    if angle_unit not in ("Qz", "Degrees", "Radians"):
        raise ValueError(f"Unknown angle unit: {angle_unit!r}")
    if not np.isfinite(wavelength_angstrom) or wavelength_angstrom <= 0:
        raise ValueError("Wavelength must be a positive finite number.")
    if not np.isfinite(wavelength_resolution) or wavelength_resolution < 0:
        raise ValueError("Wavelength resolution must be finite and nonnegative.")
    if not (np.isfinite(distance_s1_mm) and np.isfinite(distance_s2_mm)) or distance_s1_mm <= distance_s2_mm:
        raise ValueError("Monochromator-slit distance (D1) must exceed sample-slit distance (D2).")

    th = np.asarray(theta_deg, dtype=float)
    s1 = np.asarray(s1_mm, dtype=float)
    s2 = np.asarray(s2_mm, dtype=float)
    fp = np.asarray(fwhm_proj_mm, dtype=float)
    if not (th.ndim == s1.ndim == s2.ndim == fp.ndim == 1):
        raise ValueError("Resolution inputs must be 1D arrays.")
    if not (th.size == s1.size == s2.size == fp.size):
        raise ValueError("Resolution input arrays must have matching lengths.")

    th_rad = np.radians(th)
    k0 = 2.0 * np.pi / float(wavelength_angstrom)
    d_slit = float(distance_s1_mm) - float(distance_s2_mm)
    d_lam = float(wavelength_resolution)

    if mode == "sared":
        qz = 2.0 * k0 * np.sin(th_rad)
        res_fwhm = np.sqrt(
            (k0 ** 2) * (np.cos(th_rad) ** 2) * (0.68 ** 2) * ((s1 ** 2) + (s2 ** 2)) / (d_slit ** 2)
            + (d_lam ** 2) * (qz ** 2)
        )
    else:
        d_alpha = np.arctan((s1 + fp) / (d_slit * 2.0))
        if angle_unit == "Qz":
            res_fwhm = np.sqrt(
                (k0 ** 2) * (((np.cos(th_rad) ** 2) * (d_alpha ** 2)) + ((d_lam ** 2) * (np.sin(th_rad) ** 2)))
            )
        elif angle_unit == "Radians":
            res_fwhm = d_alpha
        else:
            res_fwhm = np.degrees(d_alpha)

    return res_fwhm * FWHM_TO_SIGMA
