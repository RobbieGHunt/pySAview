"""Pure, UI-free, I/O-free neutron reflectometry reduction physics package.

This package organizes pySAview's numerical reduction steps cleanly alongside
the SuperADAM ``pySAred`` / ORSO reflectometry reduction pipeline, using only
NumPy/SciPy (with zero ``PyQt6``, ``pyqtgraph``, or ``h5py`` dependencies):

1. :mod:`~pySAview.core.reduction.roi_extraction`
   Detector ROI whole-pixel bounds, 2D and Y-axis integration, and reflection
   peak tracking across scan frames.
2. :mod:`~pySAview.core.reduction.angle_calibration`
   Direct-beam and specular pixel-to-angle calibration (``deg_per_pixel``,
   specular reference ``x0``, detector orientation ``sign``, and
   ``(alpha_i, alpha_f)`` coordinates).
3. :mod:`~pySAview.core.reduction.kinematics`
   Specular and off-specular angle-to-Q conversions (``Q``, ``Qx``, ``Qz``),
   inverse ``Q -> (alpha_i, alpha_f)`` mapping, pixel ``Qz`` resolution step,
   and constant-``Qz`` / constant-``Qx`` iso-contours.
4. :mod:`~pySAview.core.reduction.normalization`
   1D and 2D count normalization (raw, exposure time, monitor, reference
   monitor rate) with exact Garwood/Clopper-Pearson Poisson intervals and
   first-order uncertainty propagation.
5. :mod:`~pySAview.core.reduction.polarization`
   Spin-state frame selection, 1D flipper splitting, angle-matched ``u``/``d``
   pairing, pooled ``u + d`` intensity maps, and spin asymmetry
   ``(I_u - I_d) / (I_u + I_d)``.
"""

from pySAview.core.reduction.angle_calibration import (
    DETECTOR_WIDTH_MM,
    PYSARED_DEFAULT_DISTANCE_MM,
    AngleCalibration,
    CalibrationError,
    DirectBeamFit,
    alpha_angles,
    alpha_f_offset_for_pixel,
    alpha_f_offsets,
    alpha_i_strip_edges,
    calibrate,
    effective_deg_per_pixel,
    find_direct_beam,
    fit_direct_beam,
    implied_distance_mm,
    pixel_for_alpha_f_offset,
    pixel_pitch_mm,
    specular_center,
)
from pySAview.core.reduction.kinematics import (
    angles_from_q,
    compute_q,
    constant_qx_alpha_f,
    constant_qz_alpha_f,
    q_from_angles,
    qz_step_per_pixel,
)
from pySAview.core.reduction.normalization import (
    CONFIDENCE,
    NORMALIZATION_UNITS,
    NormalizationResult,
    NormalizedMap,
    confidence_from_sigma,
    is_count_channel,
    log_error_bounds,
    normalize_map,
    normalize_signal,
    poisson_interval,
    poisson_ratio_interval,
)
from pySAview.core.reduction.polarization import (
    AsymmetryMap,
    SummedSpinMap,
    pair_spin_frames,
    select_frames,
    spin_asymmetry,
    split_indices_by_flipper,
    sum_spin_frames,
)
from pySAview.core.reduction.roi_extraction import (
    auto_fit_from_profiles,
    auto_fit_scan_rois,
    find_reflection_peak_center,
    frame_y_bounds,
    integrate_frames_along_y,
    integrate_roi_2d,
    pixel_bounds,
    y_pixel_bounds,
)

__all__ = [
    # kinematics
    "angles_from_q",
    "compute_q",
    "constant_qx_alpha_f",
    "constant_qz_alpha_f",
    "q_from_angles",
    "qz_step_per_pixel",
    # angle_calibration
    "AngleCalibration",
    "CalibrationError",
    "DETECTOR_WIDTH_MM",
    "DirectBeamFit",
    "PYSARED_DEFAULT_DISTANCE_MM",
    "alpha_angles",
    "alpha_f_offset_for_pixel",
    "alpha_f_offsets",
    "alpha_i_strip_edges",
    "calibrate",
    "effective_deg_per_pixel",
    "find_direct_beam",
    "fit_direct_beam",
    "implied_distance_mm",
    "pixel_for_alpha_f_offset",
    "pixel_pitch_mm",
    "specular_center",
    # roi_extraction
    "auto_fit_from_profiles",
    "auto_fit_scan_rois",
    "find_reflection_peak_center",
    "frame_y_bounds",
    "integrate_frames_along_y",
    "integrate_roi_2d",
    "pixel_bounds",
    "y_pixel_bounds",
    # normalization
    "CONFIDENCE",
    "NORMALIZATION_UNITS",
    "NormalizationResult",
    "NormalizedMap",
    "confidence_from_sigma",
    "is_count_channel",
    "log_error_bounds",
    "normalize_map",
    "normalize_signal",
    "poisson_interval",
    "poisson_ratio_interval",
    # polarization
    "AsymmetryMap",
    "SummedSpinMap",
    "pair_spin_frames",
    "select_frames",
    "spin_asymmetry",
    "split_indices_by_flipper",
    "sum_spin_frames",
]
