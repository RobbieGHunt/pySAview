"""Momentum-transfer and angle-to-Q kinematics for specular and off-specular reflectometry.

All angles are in degrees, wavelengths in Angstroms (Å), and wavevector transfers
(Q, Qx, Qz) in inverse Angstroms (Å⁻¹), with wavevector magnitude k0 = 2π / λ.

Sign convention
---------------
``q_from_angles`` and its inverses follow ``pySAred``'s sign convention::

    Qx = k0 * (cos(alpha_f) - cos(alpha_i))
    Qz = k0 * (sin(alpha_f) + sin(alpha_i))

Note explicitly that this Qx convention is **opposite in sign** to Gibaud et al.
(1993), Eq. 15a (which writes ``Qx = k0 * (cos(alpha_i) - cos(alpha_f))``).
``pySAred`` plots ``-Qx``; pySAview reports ``Qx`` directly as defined above.
The specular ridge (``alpha_f = alpha_i``) lies at ``Qx = 0`` with
``Qz = (4π / λ) sin(alpha_i)``, and the direct beam (``alpha_f = -alpha_i``)
lies at ``Qz = 0, Qx = 0``.
"""

from __future__ import annotations

import numpy as np


def _validate_wavelength(wavelength: float) -> float:
    if not (np.isfinite(wavelength) and wavelength > 0):
        raise ValueError("Wavelength must be a positive finite number")
    return float(wavelength)


def compute_q(theta_deg: np.ndarray, wavelength_angstrom: float = 5.21) -> np.ndarray:
    """Compute specular momentum transfer Q (Å⁻¹) from incident angle theta (deg).

    Uses the standard specular reflectometry relation::

        Q = (4 * pi / wavelength) * sin(theta_rad)
    """
    if not np.isfinite(wavelength_angstrom) or wavelength_angstrom <= 0:
        raise ValueError("Wavelength must be strictly positive.")
    theta_rad = np.radians(np.asarray(theta_deg, dtype=float))
    return (4.0 * np.pi / wavelength_angstrom) * np.sin(theta_rad)


def q_from_angles(alpha_i, alpha_f, wavelength: float):
    """Momentum transfer (Qx, Qz) in Å⁻¹ from incident/exit angles in degrees.

    Follows ``pySAred``'s sign convention::

        Qx = (2π / λ) * (cos(alpha_f) - cos(alpha_i))
        Qz = (2π / λ) * (sin(alpha_f) + sin(alpha_i))

    Note: ``Qx`` here has the opposite sign to Gibaud et al. (1993) Eq. 15a
    (``Qx = k0 * (cos(alpha_i) - cos(alpha_f))``); ``pySAred`` plots ``-Qx``.
    The specular condition ``alpha_f = alpha_i`` gives ``Qx = 0``; the direct
    beam (``alpha_f = -alpha_i``) gives ``Qz = 0``.
    """
    lam = _validate_wavelength(wavelength)
    k0 = 2.0 * np.pi / lam
    ai = np.radians(np.asarray(alpha_i, dtype=float))
    af = np.radians(np.asarray(alpha_f, dtype=float))
    return k0 * (np.cos(af) - np.cos(ai)), k0 * (np.sin(af) + np.sin(ai))


def angles_from_q(qx, qz, wavelength: float):
    """Inverse of :func:`q_from_angles` returning ``(alpha_i, alpha_f)`` in degrees for |angles| < 90°.

    Uses ``pySAred``'s sign convention (opposite in ``Qx`` sign to Gibaud et al.
    1993 Eq. 15a). With ``A = Qz / k0 = 2 sin(s / 2) cos(d / 2)`` and
    ``B = Qx / k0 = -2 sin(s / 2) sin(d / 2)``, where ``s = alpha_i + alpha_f``
    and ``d = alpha_f - alpha_i``::

        d = 2 * arctan(-B / A)
        s = 2 * arcsin(A / (2 * cos(d / 2)))

    Because ``|d / 2| < 90°``, the plain ``arctan`` (not ``atan2``) remains valid
    below the horizon (``Qz < 0``). The direct-beam line ``alpha_f = -alpha_i``
    maps to ``Q = 0`` and has no unique inverse; it is returned as
    ``alpha_i = alpha_f = 0``.
    """
    lam = _validate_wavelength(wavelength)
    k0 = 2.0 * np.pi / lam
    a = np.asarray(qz, dtype=float) / k0
    b = np.asarray(qx, dtype=float) / k0
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(a != 0, 2.0 * np.arctan(-b / np.where(a != 0, a, 1.0)), 0.0)
        s = 2.0 * np.arcsin(np.clip(a / (2.0 * np.cos(d / 2.0)), -1.0, 1.0))
    return np.degrees((s - d) / 2.0), np.degrees((s + d) / 2.0)


def qz_step_per_pixel(deg_per_pixel: float, wavelength: float) -> float:
    """Small-angle Qz resolution step (Å⁻¹) corresponding to one detector pixel.

    Computes ``(2π / λ) * radians(deg_per_pixel)``, i.e. ``k0 * Δα_f`` at grazing
    exit angles where ``cos(alpha_f) ≈ 1``.
    """
    if isinstance(deg_per_pixel, bool) or not (np.isfinite(deg_per_pixel) and deg_per_pixel > 0):
        raise ValueError("deg_per_pixel must be a positive finite number")
    lam = _validate_wavelength(wavelength)
    return 2.0 * np.pi / lam * np.radians(deg_per_pixel)


def constant_qz_alpha_f(alpha_i_deg, qz, wavelength: float):
    """Exit angle ``alpha_f`` (deg) along a constant-``Qz`` contour at ``alpha_i_deg``.

    Inverts ``Qz = k0 * (sin(alpha_f) + sin(alpha_i))``::

        alpha_f = degrees(arcsin(Qz / k0 - sin(alpha_i)))

    Points where ``|Qz / k0 - sin(alpha_i)| > 1`` are outside the kinematically
    accessible range and return ``NaN`` without raising runtime warnings.
    """
    lam = _validate_wavelength(wavelength)
    k0 = 2.0 * np.pi / lam
    ai = np.radians(np.asarray(alpha_i_deg, dtype=float))
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arcsin(np.asarray(qz, dtype=float) / k0 - np.sin(ai)))


def constant_qx_alpha_f(alpha_i_deg, qx, wavelength: float, side: float = 1.0):
    """Exit angle ``alpha_f`` (deg) along a constant-``Qx`` contour at ``alpha_i_deg``.

    Inverts ``Qx = k0 * (cos(alpha_f) - cos(alpha_i))`` (``pySAred`` sign
    convention, opposite to Gibaud et al. 1993 Eq. 15a)::

        alpha_f = side * degrees(arccos(Qx / k0 + cos(alpha_i)))

    ``side`` (``+1.0`` or ``-1.0``) selects the branch above or below the sample
    horizon (``alpha_f >= 0`` vs ``alpha_f < 0``). Kinematically inaccessible
    points return ``NaN`` without raising runtime warnings.
    """
    if isinstance(side, bool) or side not in (-1.0, 1.0):
        raise ValueError("side must be +1.0 or -1.0")
    lam = _validate_wavelength(wavelength)
    k0 = 2.0 * np.pi / lam
    ai = np.radians(np.asarray(alpha_i_deg, dtype=float))
    with np.errstate(invalid="ignore"):
        return side * np.degrees(np.arccos(np.asarray(qx, dtype=float) / k0 + np.cos(ai)))


__all__ = [
    "angles_from_q",
    "compute_q",
    "constant_qx_alpha_f",
    "constant_qz_alpha_f",
    "q_from_angles",
    "qz_step_per_pixel",
]
