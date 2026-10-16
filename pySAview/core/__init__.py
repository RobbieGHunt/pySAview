"""Core models, NeXus reader, and scan calculation modules."""

from pySAview.core.models import (
    DatasetPreview,
    MotorInfo,
    NodeInfo,
    ScanData,
    ScanMetadata,
    ScanVariable,
)
from pySAview.core.nexus_reader import NexusReader
from pySAview.core.scan_calculations import (
    CurveStats,
    PlottedCurve,
    calculate_curve_stats,
    compute_normalized_y,
    compute_q,
    integrate_roi_2d,
    split_by_flipper,
)

__all__ = [
    "CurveStats",
    "DatasetPreview",
    "MotorInfo",
    "NexusReader",
    "NodeInfo",
    "PlottedCurve",
    "ScanData",
    "ScanMetadata",
    "ScanVariable",
    "calculate_curve_stats",
    "compute_normalized_y",
    "compute_q",
    "integrate_roi_2d",
    "split_by_flipper",
]
