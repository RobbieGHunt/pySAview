"""Overillumination (beam footprint) correction following pySAred 1.5.3.

Implements the trapezoidal incident-beam profile model from Alexey Klechikov's
``pySAred_V1.5.3.py`` (``f_overilluminationCorrCoeff``).

Geometry (all lengths in mm, incident angle ``theta_deg`` in degrees)::

    S1 (monochromator slit, width s1hg) ---- D1 - D2 ----> S2 (sample slit, width s2hg) ---- D2 ----> Sample (length L)

At the sample position the horizontal beam profile is a symmetric trapezoid
with flat-top half-width ``OB`` and full-foot half-width ``OC`` (shoulder width
``BC = OC - OB``), normalized to unit total area with height ``AO = 1 / (BC/2 + OB)``
and half-FWHM ``FWHM_beam = BC/2 + OB``.

For a sample of length ``L`` at incident angle ``theta``, its projection onto
the beam cross-section is ``sampleLen_relative = L * sin(|theta|)``.  The
returned tuple ``(overill_corr, fwhm_proj)`` gives:

* ``overill_corr`` in ``(0, 1]``: fraction of the total incident beam
  intercepted by the sample (intensity and its uncertainty are divided by
  this factor when overillumination correction is enabled).
* ``fwhm_proj`` (mm): effective projected slit width at the sample, equal to
  ``s2hg`` when the sample spans the beam FWHM and ``sampleLen_relative``
  otherwise; consumed by ``pySAview.core.reduction.resolution``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_DISTANCE_S1_TO_SAMPLE_MM",
    "DEFAULT_DISTANCE_S2_TO_SAMPLE_MM",
    "DEFAULT_SAMPLE_LENGTH_MM",
    "compute_overillumination_series",
    "extract_slit_widths",
    "overillumination_coeff",
]

DEFAULT_DISTANCE_S1_TO_SAMPLE_MM: float = 2300.0
DEFAULT_DISTANCE_S2_TO_SAMPLE_MM: float = 290.0
DEFAULT_SAMPLE_LENGTH_MM: float = 50.0


def overillumination_coeff(
    s1hg: float,
    s2hg: float,
    theta_deg: float,
    sample_length_mm: float = DEFAULT_SAMPLE_LENGTH_MM,
    distance_s1_mm: float = DEFAULT_DISTANCE_S1_TO_SAMPLE_MM,
    distance_s2_mm: float = DEFAULT_DISTANCE_S2_TO_SAMPLE_MM,
) -> tuple[float, float]:
    """Return ``(overill_corr, fwhm_proj)`` for one point, matching pySAred 1.5.3.

    Raises ``ValueError`` if geometric parameters are nonfinite or physically
    invalid (nonpositive slits, sample length, or ``distance_s1_mm <= distance_s2_mm``).
    """
    s1 = float(s1hg)
    s2 = float(s2hg)
    th = float(theta_deg)
    sample_len = float(sample_length_mm)
    d1 = float(distance_s1_mm)
    d2 = float(distance_s2_mm)

    if not all(np.isfinite((s1, s2, th, sample_len, d1, d2))):
        raise ValueError("Overillumination parameters must be finite numbers.")
    if sample_len <= 0:
        raise ValueError("Sample length must be strictly positive.")
    if d1 <= d2:
        raise ValueError("Monochromator-slit distance (D1) must exceed sample-slit distance (D2).")
    if s1 <= 0 or s2 <= 0:
        raise ValueError("Slit widths s1h and s2h must be strictly positive.")

    # Trapezoidal beam half-widths at sample: flat region (OB) and widest beam (OC),
    # kept in the exact three-branch form of pySAred_V1.5.3.py for direct inspection.
    if s1 < s2:
        OB = abs(((d1 * (s2 - s1)) / (2.0 * (d1 - d2))) + s1 / 2.0)
        OC = ((d1 * (s2 + s1)) / (2.0 * (d1 - d2))) - s1 / 2.0
    elif s1 > s2:
        OB = abs(((s2 * d1) - (s1 * d2)) / (2.0 * (d1 - d2)))
        OC = (d1 / (d1 - d2)) * (s2 + s1) / 2.0 - (s1 / 2.0)
    else:
        OB = s1 / 2.0
        OC = s1 * (d1 / (d1 - d2) - 0.5)

    BC = OC - OB
    FWHM_beam = BC / 2.0 + OB  # half of the beam FWHM
    AO = 1.0 / FWHM_beam  # normalized height of trapezoid

    # Projection of sample surface on the beam (matches pySAred 1.5.3 zero guard)
    th_eff = abs(th if th != 0.0 else 0.00001)
    sampleLen_relative = sample_len * float(np.sin(np.radians(th_eff)))

    # Fraction of total beam intensity illuminating the sample
    half_proj = sampleLen_relative / 2.0
    if half_proj >= OC:
        overill_corr = 1.0
    elif half_proj <= OB:
        overill_corr = AO * half_proj  # flat top part only
    else:
        # Flat top + trapezoid shoulder minus unilluminated outer triangle
        overill_corr = AO * (OB + BC / 2.0 - ((OC - half_proj) ** 2) / (2.0 * BC))

    # Effective beam width covered by the sample for angular resolution
    fwhm_proj = s2 if half_proj >= FWHM_beam else sampleLen_relative

    return float(overill_corr), float(fwhm_proj)


def compute_overillumination_series(
    s1_mm: np.ndarray,
    s2_mm: np.ndarray,
    theta_deg: np.ndarray,
    sample_length_mm: float = DEFAULT_SAMPLE_LENGTH_MM,
    distance_s1_mm: float = DEFAULT_DISTANCE_S1_TO_SAMPLE_MM,
    distance_s2_mm: float = DEFAULT_DISTANCE_S2_TO_SAMPLE_MM,
    *,
    round_theta_decimals: int | None = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vector-shaped evaluation of ``overillumination_coeff`` across a scan.

    ``round_theta_decimals=4`` matches pySAred 1.5.3's ``round(th, 4)`` call.
    Points with nonfinite or nonpositive slits (e.g. unwritten tail frames in an
    aborted scan) are marked ``False`` in ``valid`` and receive ``NaN`` outputs
    rather than dividing by zero.

    Returns ``(overill_corr, fwhm_proj, valid)``.
    """
    s1 = np.asarray(s1_mm, dtype=float)
    s2 = np.asarray(s2_mm, dtype=float)
    th = np.asarray(theta_deg, dtype=float)
    if not (s1.ndim == s2.ndim == th.ndim == 1 and s1.size == s2.size == th.size):
        raise ValueError("s1_mm, s2_mm, and theta_deg must be 1D arrays of equal length.")

    if not np.isfinite(sample_length_mm) or sample_length_mm <= 0:
        raise ValueError("Sample length must be a positive finite number.")
    if not (np.isfinite(distance_s1_mm) and np.isfinite(distance_s2_mm)) or distance_s1_mm <= distance_s2_mm:
        raise ValueError("Monochromator-slit distance (D1) must exceed sample-slit distance (D2).")

    n = th.size
    coeff = np.full(n, np.nan, dtype=float)
    fwhm = np.full(n, np.nan, dtype=float)
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        if not (np.isfinite(s1[i]) and np.isfinite(s2[i]) and np.isfinite(th[i])):
            continue
        if s1[i] <= 0 or s2[i] <= 0:
            continue
        th_val = round(float(th[i]), round_theta_decimals) if round_theta_decimals is not None else float(th[i])
        c0, fp = overillumination_coeff(
            float(s1[i]), float(s2[i]), th_val,
            sample_length_mm=sample_length_mm,
            distance_s1_mm=distance_s1_mm,
            distance_s2_mm=distance_s2_mm,
        )
        coeff[i] = c0
        fwhm[i] = fp
        valid[i] = True

    return coeff, fwhm, valid


def extract_slit_widths(scan: Any, n_points: int) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Read per-point horizontal slit openings ``(s1h, s2h)`` in mm from a ``ScanData``.

    Prefers per-step scanned variables ``s1h`` / ``s2h`` (so variable-slit scans
    are handled accurately). Falls back to scalar ``SUPERADAM/VirtualSlit`` widths
    recorded on ``scan.slit_defaults`` when scanned slit channels are absent.
    Float32 display noise is cleaned to 8 decimal places.
    """
    variables = getattr(scan, "variables", {}) or {}
    s1_var = variables.get("s1h") or variables.get("s1hg")
    s2_var = variables.get("s2h") or variables.get("s2hg")

    if s1_var is not None and s2_var is not None:
        s1 = np.round(np.asarray(s1_var.data, dtype=float), 8)
        s2 = np.round(np.asarray(s2_var.data, dtype=float), 8)
        if s1.ndim == 1 and s2.ndim == 1 and s1.size == n_points and s2.size == n_points:
            return s1, s2, f"scanned channels ({s1_var.name}, {s2_var.name})"

    slit_defaults = getattr(scan, "slit_defaults", None)
    if slit_defaults is not None and len(slit_defaults) == 2:
        w1, w2 = float(slit_defaults[0]), float(slit_defaults[1])
        if np.isfinite(w1) and np.isfinite(w2) and w1 > 0 and w2 > 0:
            s1 = np.full(n_points, round(w1, 8), dtype=float)
            s2 = np.full(n_points, round(w2, 8), dtype=float)
            return s1, s2, "VirtualSlit defaults"

    return None, None, "unavailable"
