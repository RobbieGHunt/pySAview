"""Backward-compatible re-exports for detector angle calibration and Q kinematics.

Implementation lives in :mod:`pySAview.core.reduction.angle_calibration` and
:mod:`pySAview.core.reduction.kinematics`.
"""

from pySAview.core.reduction.angle_calibration import (
    DETECTOR_WIDTH_MM,
    AngleCalibration,
    CalibrationError,
    DirectBeamFit,
    alpha_angles,
    alpha_f_offsets,
    alpha_i_strip_edges,
    calibrate,
    find_direct_beam,
    fit_direct_beam,
    implied_distance_mm,
    pixel_pitch_mm,
    specular_center,
)
from pySAview.core.reduction.kinematics import (
    angles_from_q,
    q_from_angles,
)

__all__ = [
    "AngleCalibration",
    "CalibrationError",
    "DETECTOR_WIDTH_MM",
    "DirectBeamFit",
    "alpha_angles",
    "alpha_f_offsets",
    "alpha_i_strip_edges",
    "angles_from_q",
    "calibrate",
    "find_direct_beam",
    "fit_direct_beam",
    "implied_distance_mm",
    "pixel_pitch_mm",
    "q_from_angles",
    "specular_center",
]
