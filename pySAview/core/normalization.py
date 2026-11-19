"""Backward-compatible re-exports for count normalization.

Implementation lives in :mod:`pySAview.core.reduction.normalization`.
"""

from pySAview.core.reduction.normalization import (
    CONFIDENCE,
    NORMALIZATION_UNITS,
    NormalizationResult,
    NormalizedMap,
    _vector,
    confidence_from_sigma,
    is_count_channel,
    log_error_bounds,
    normalize_map,
    normalize_signal,
    poisson_interval,
    poisson_ratio_interval,
)

__all__ = [
    "CONFIDENCE",
    "NORMALIZATION_UNITS",
    "NormalizationResult",
    "NormalizedMap",
    "_vector",
    "confidence_from_sigma",
    "is_count_channel",
    "log_error_bounds",
    "normalize_map",
    "normalize_signal",
    "poisson_interval",
    "poisson_ratio_interval",
]
