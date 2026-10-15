"""Immutable, versioned ROI definitions; measurement files stay read-only."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from sa_nxs_explorer.core.roi import pixel_bounds
from sa_nxs_explorer.core.source_safety import validate_output_path

CONVENTION = "display=raw.T; nearest-ties-even; half-open; whole-pixels"


@dataclass(frozen=True)
class ROIRecord:
    mode: str
    detector_shape: tuple[int, int]  # displayed (ny, nx)
    frame_count: int
    bounds: tuple[tuple[float, float, float, float], ...]
    source_path: str
    version: int = 1
    convention: str = CONVENTION

    def __post_init__(self):
        if self.version != 1 or self.convention != CONVENTION:
            raise ValueError("Unsupported ROI format or coordinate convention")
        if self.mode not in ("fixed", "tracked"):
            raise ValueError("ROI mode must be fixed or tracked")
        if len(self.detector_shape) != 2 or any(type(n) is not int or n <= 0 for n in self.detector_shape):
            raise ValueError("Invalid detector dimensions")
        if type(self.frame_count) is not int or self.frame_count <= 0:
            raise ValueError("Invalid frame count")
        if len(self.bounds) != (1 if self.mode == "fixed" else self.frame_count):
            raise ValueError("ROI requires complete bounds for its mode")
        for roi in self.bounds:
            x0, x1, y0, y1 = pixel_bounds(roi, self.detector_shape[1], self.detector_shape[0])
            if x1 <= x0 or y1 <= y0:
                raise ValueError("ROI must select at least one detector pixel")
        if not isinstance(self.source_path, str) or not self.source_path:
            raise ValueError("ROI source path is required")

    def validate_scan(self, shape, source_path):
        if tuple(shape[1:][::-1]) != self.detector_shape:
            raise ValueError("ROI detector dimensions do not match this scan")
        if self.mode == "tracked" and (shape[0] != self.frame_count or Path(source_path).resolve() != Path(self.source_path).resolve()):
            raise ValueError("Tracked ROIs belong to their original scan and frame count. Use a fixed ROI for reuse.")


def save_roi(path, record, source_paths=()):
    target = validate_output_path(path, [record.source_path, *source_paths])
    if target.suffix.lower() != ".json":
        raise ValueError("Save ROI definitions to a separate .json file")
    text = json.dumps(asdict(record), indent=2, allow_nan=False)
    # New sidecars only: never truncate an existing file or alias.
    with target.open("x", encoding="utf-8") as output:
        output.write(text + "\n")


def load_roi(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("ROI file must contain an object")
    try:
        data["detector_shape"] = tuple(data["detector_shape"])
        data["bounds"] = tuple(tuple(roi) for roi in data["bounds"])
        if "version" not in data or "convention" not in data:
            raise ValueError("Missing ROI version or coordinate convention")
        return ROIRecord(**data)
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed ROI definition") from error
