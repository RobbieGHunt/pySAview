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
6. :mod:`~pySAview.core.reduction.overillumination`
   Trapezoidal beam footprint correction (``c0``, ``fwhm_proj``) and Nomad
   slit-width extraction following pySAred 1.5.3.
7. :mod:`~pySAview.core.reduction.curvature`
   Reversible per-spin ROI shear-and-regrid sample curvature correction
   following pySAred 1.5.3.
8. :mod:`~pySAview.core.reduction.background`
   Width- and Y-locked background ROI geometry, overlap check, and pySAred
   1.5.3 background subtraction with ``Qz`` skip cutoff.
9. :mod:`~pySAview.core.reduction.resolution`
   Gaussian 1-sigma angular and ``Qz`` resolution in both ``pySAred``
   (``FWHM_proj``) and Gunnar ``Sared`` (``0.68``) modes.
10. :mod:`~pySAview.core.reduction.pysared`
    Per-scan ``ReductionParams``, the pySAred 1.5.3 reduction/error pipeline,
    and per-curve point trimming.
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
from pySAview.core.reduction.background import (
    bkg_roi_from_signal,
    default_bkg_x_bounds,
    rois_overlap_x,
    subtract_background_pysared,
)
from pySAview.core.reduction.curvature import (
    apply_curvature_by_spin,
    apply_curvature_to_map,
    apply_curvature_to_slice,
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
from pySAview.core.reduction.overillumination import (
    DEFAULT_DISTANCE_S1_TO_SAMPLE_MM,
    DEFAULT_DISTANCE_S2_TO_SAMPLE_MM,
    DEFAULT_SAMPLE_LENGTH_MM,
    compute_overillumination_series,
    extract_slit_widths,
    overillumination_coeff,
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
from pySAview.core.reduction.pysared import (
    ReductionParams,
    ReductionResult,
    reduce_counts_pysared,
    trim_curve_mask,
)
from pySAview.core.reduction.resolution import (
    DEFAULT_WAVELENGTH_RESOLUTION,
    FWHM_TO_SIGMA,
    compute_resolution_pysared,
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
    # overillumination
    "DEFAULT_DISTANCE_S1_TO_SAMPLE_MM",
    "DEFAULT_DISTANCE_S2_TO_SAMPLE_MM",
    "DEFAULT_SAMPLE_LENGTH_MM",
    "compute_overillumination_series",
    "extract_slit_widths",
    "overillumination_coeff",
    # curvature
    "apply_curvature_by_spin",
    "apply_curvature_to_map",
    "apply_curvature_to_slice",
    # background
    "bkg_roi_from_signal",
    "default_bkg_x_bounds",
    "rois_overlap_x",
    "subtract_background_pysared",
    # resolution
    "DEFAULT_WAVELENGTH_RESOLUTION",
    "FWHM_TO_SIGMA",
    "compute_resolution_pysared",
    # pysared pipeline
    "ReductionParams",
    "ReductionResult",
    "reduce_counts_pysared",
    "trim_curve_mask",
]
