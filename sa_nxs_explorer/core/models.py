"""Data models for NeXus / SuperADAM data exploration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class NodeInfo:
    """Represents a node (Group or Dataset) in an HDF5 / NeXus file hierarchy."""

    path: str
    name: str
    kind: str  # "group" or "dataset"
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    n_children: int = 0
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetPreview:
    """Preview metadata and sampled/full data payload for a dataset."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    data: np.ndarray | None = None
    text: str | None = None
    truncated: bool = False
    stats: dict[str, float] = field(default_factory=dict)
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class ScanVariable:
    """One measured channel/variable in entry0/data_scan/scanned_variables."""

    index: int
    name: str
    label: str
    unit: str
    property: str
    axis: int
    is_scanned: bool
    data: np.ndarray
    variance: np.ndarray | None = None


@dataclass
class MotorInfo:
    """Motor / positioner state from instrument metadata."""

    name: str
    value: float | str | None = None
    target_value: float | str | None = None
    offset_value: float | str | None = None
    units: str = ""


@dataclass
class ScanMetadata:
    """Experiment and scan metadata from NeXus entry0."""

    run_number: int | None = None
    title: str = ""
    subtitle: str = ""
    proposal: str = ""
    user: str = ""
    local_contact: str = ""
    start_time: str = ""
    end_time: str = ""
    duration: float | None = None
    mode: str = ""
    preset: float | None = None
    reactor_power: float | None = None
    sample_id: str = ""
    chemical_formula: str = ""
    sample_consistance: str = ""
    sample_mass: str = ""
    sample_size: str = ""
    surface_area: str = ""
    environment_variables: dict[str, float] = field(default_factory=dict)


@dataclass
class ScanData:
    """Full parsed scan container for quick charting and analysis."""

    file_path: str
    metadata: ScanMetadata
    variables: dict[str, ScanVariable]
    motors: dict[str, MotorInfo] = field(default_factory=dict)
    has_detector_2d: bool = False
    detector_shape: tuple[int, ...] | None = None
    detector_roi: tuple[float, float, float, float] | None = None  # (x0, x1, y0, y1)
    detector_path: str = ""
