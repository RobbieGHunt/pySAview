"""NeXus HDF5 Reader for SuperADAM and neutron reflectometry scan files."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import h5py
import numpy as np

from sa_nxs_explorer.core.models import (
    DatasetPreview,
    MotorInfo,
    NodeInfo,
    ScanData,
    ScanMetadata,
    ScanVariable,
)


def decode_val(val: Any) -> Any:
    """Decode raw HDF5 dataset or attribute value to clean Python string / numeric."""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray):
        if val.dtype.kind in ("S", "U"):
            try:
                decoded = [x.decode("utf-8", errors="replace") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in val.ravel()]
                if len(decoded) == 1:
                    return decoded[0]
                return decoded
            except Exception:
                return str(val)
        if val.size == 1:
            item = val.item()
            if isinstance(item, (bytes, np.bytes_)):
                return item.decode("utf-8", errors="replace")
            return item
    return val


def decode_attr_str(val: Any) -> str:
    """Decode attribute value to human-readable string."""
    decoded = decode_val(val)
    if isinstance(decoded, list):
        return ", ".join(str(x) for x in decoded)
    return str(decoded)


class NexusReader:
    """Read-only lazy accessor for NeXus (.nxs) files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        self._file: h5py.File | None = h5py.File(self.path, "r")

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __enter__(self) -> NexusReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError(f"NeXus file '{self.path}' is closed")
        return self._file

    def list_children(self, h5path: str = "/") -> list[NodeInfo]:
        """List immediate child nodes under group h5path, sorted groups first, then datasets."""
        obj = self.file[h5path] if h5path != "/" else self.file
        if not isinstance(obj, h5py.Group):
            return []

        groups: list[NodeInfo] = []
        datasets: list[NodeInfo] = []

        for name, sub in obj.items():
            sub_attrs = {k: decode_attr_str(v) for k, v in sub.attrs.items()}
            if isinstance(sub, h5py.Group):
                groups.append(
                    NodeInfo(
                        path=sub.name,
                        name=name,
                        kind="group",
                        n_children=len(sub),
                        attrs=sub_attrs,
                    )
                )
            elif isinstance(sub, h5py.Dataset):
                datasets.append(
                    NodeInfo(
                        path=sub.name,
                        name=name,
                        kind="dataset",
                        shape=sub.shape,
                        dtype=str(sub.dtype),
                        attrs=sub_attrs,
                    )
                )

        groups.sort(key=lambda x: x.name.lower())
        datasets.sort(key=lambda x: x.name.lower())
        return groups + datasets

    def get_node_info(self, h5path: str) -> NodeInfo:
        """Get NodeInfo for an arbitrary path."""
        obj = self.file[h5path]
        name = h5path.rstrip("/").split("/")[-1] or "/"
        attrs = {k: decode_attr_str(v) for k, v in obj.attrs.items()}
        if isinstance(obj, h5py.Group):
            return NodeInfo(
                path=obj.name,
                name=name,
                kind="group",
                n_children=len(obj),
                attrs=attrs,
            )
        return NodeInfo(
            path=obj.name,
            name=name,
            kind="dataset",
            shape=obj.shape,
            dtype=str(obj.dtype),
            attrs=attrs,
        )

    def get_attributes(self, h5path: str) -> dict[str, str]:
        """Return decoded attributes dictionary for the given path."""
        obj = self.file[h5path]
        return {k: decode_attr_str(v) for k, v in obj.attrs.items()}

    def read_preview(self, h5path: str, max_elements: int = 150_000) -> DatasetPreview:
        """Safely fetch a preview of the dataset at h5path."""
        ds = self.file[h5path]
        if not isinstance(ds, h5py.Dataset):
            raise TypeError(f"Path '{h5path}' is a Group, not a Dataset.")

        shape = ds.shape
        dtype_str = str(ds.dtype)
        attrs = {k: decode_attr_str(v) for k, v in ds.attrs.items()}
        stats: dict[str, float] = {}

        # String / Object scalar or array
        if not np.issubdtype(ds.dtype, np.number):
            raw = ds[()]
            decoded = decode_val(raw)
            if isinstance(decoded, list):
                text_preview = f"[{', '.join(str(x) for x in decoded[:20])}]"
                if len(decoded) > 20:
                    text_preview += f" ... ({len(decoded)} items)"
            else:
                text_preview = str(decoded)
            return DatasetPreview(
                path=ds.name,
                shape=shape,
                dtype=dtype_str,
                data=None,
                text=text_preview,
                truncated=False,
                stats={},
                attrs=attrs,
            )

        if ds.size == 0:
            return DatasetPreview(
                path=ds.name,
                shape=shape,
                dtype=dtype_str,
                data=np.array([]),
                text="(empty dataset)",
                truncated=False,
                stats={},
                attrs=attrs,
            )

        # Scalar or 1-element
        if ds.size == 1:
            val = ds[()].item()
            return DatasetPreview(
                path=ds.name,
                shape=shape,
                dtype=dtype_str,
                data=np.array([val]),
                text=str(val),
                truncated=False,
                stats={"min": float(val), "max": float(val), "mean": float(val)},
                attrs=attrs,
            )

        # 3D or higher volume: e.g. detector_data/data (302, 1024, 1024)
        if ds.ndim >= 3:
            # Load only the first frame (slice [0]) for preview
            first_frame = ds[0]
            if first_frame.ndim > 2:
                first_frame = first_frame[0]
            arr = np.asarray(first_frame, dtype=float)
            try:
                stats["min"] = float(np.nanmin(arr))
                stats["max"] = float(np.nanmax(arr))
                stats["mean"] = float(np.nanmean(arr))
            except Exception:
                pass
            return DatasetPreview(
                path=ds.name,
                shape=shape,
                dtype=dtype_str,
                data=arr,
                text=f"3D volume {shape}. Showing frame 0 slice ({arr.shape[0]}x{arr.shape[1]}).",
                truncated=True,
                stats=stats,
                attrs=attrs,
            )

        # 1D or 2D dataset
        truncated = ds.size > max_elements
        if truncated:
            if ds.ndim == 1:
                data = ds[:max_elements]
            else:
                # 2D downsample or crop
                data = ds[: int(np.sqrt(max_elements)), : int(np.sqrt(max_elements))]
            arr = np.asarray(data, dtype=float)
        else:
            arr = np.asarray(ds[()], dtype=float)

        if arr.size > 0:
            try:
                stats["min"] = float(np.nanmin(arr))
                stats["max"] = float(np.nanmax(arr))
                stats["mean"] = float(np.nanmean(arr))
            except Exception:
                pass

        return DatasetPreview(
            path=ds.name,
            shape=shape,
            dtype=dtype_str,
            data=arr,
            text=None,
            truncated=truncated,
            stats=stats,
            attrs=attrs,
        )

    def read_detector_frame(self, frame_index: int = 0) -> np.ndarray | None:
        """Read a single 2D detector frame by index from entry0/data_scan/detector_data/data."""
        path = "entry0/data_scan/detector_data/data"
        if path not in self.file:
            return None
        ds = self.file[path]
        if ds.ndim != 3:
            return None
        total_frames = ds.shape[0]
        if frame_index < 0 or frame_index >= total_frames:
            raise IndexError(f"Frame index {frame_index} out of bounds (total frames: {total_frames})")
        return ds[frame_index]

    def get_detector_roi(self) -> tuple[float, float, float, float] | None:
        """Get (roi_x0, roi_x1, roi_y0, roi_y1) if present in SUPERADAM/Detector1."""
        det_path = "entry0/SUPERADAM/Detector1"
        if det_path not in self.file:
            return None
        det = self.file[det_path]
        try:
            x0 = float(det["roi_x0"][()][0])
            x1 = float(det["roi_x1"][()][0])
            y0 = float(det["roi_y0"][()][0])
            y1 = float(det["roi_y1"][()][0])
            return (x0, x1, y0, y1)
        except Exception:
            return None

    def compute_full_scan_roi(self, roi: tuple[float, float, float, float]) -> np.ndarray | None:
        """Compute integrated counts for each scan frame within ROI (x0, x1, y0, y1).

        Coordinates (x, y) are in the displayed frame orientation (where displayed_frame = raw_frame.T),
        so the raw 3D volume is sliced along raw_data[:, ix0:ix1, iy0:iy1].
        """
        path = "entry0/data_scan/detector_data/data"
        if path not in self.file:
            return None
        ds = self.file[path]
        if ds.ndim != 3:
            return None

        x0, x1, y0, y1 = roi
        nx = ds.shape[1]
        ny = ds.shape[2]

        ix0 = max(0, min(nx, int(round(min(x0, x1)))))
        ix1 = max(0, min(nx, int(round(max(x0, x1)))))
        iy0 = max(0, min(ny, int(round(min(y0, y1)))))
        iy1 = max(0, min(ny, int(round(max(y0, y1)))))

        if ix1 <= ix0 or iy1 <= iy0:
            return np.zeros(ds.shape[0], dtype=float)

        sub = ds[:, ix0:ix1, iy0:iy1]
        return np.asarray(np.sum(sub, axis=(1, 2)), dtype=float)

    def compute_per_frame_scan_roi(
        self, rois: list[tuple[float, float, float, float]]
    ) -> np.ndarray | None:
        """Compute integrated counts for each scan frame given per-frame ROIs [(x0, x1, y0, y1), ...]."""
        path = "entry0/data_scan/detector_data/data"
        if path not in self.file:
            return None
        ds = self.file[path]
        if ds.ndim != 3:
            return None

        n_frames = ds.shape[0]
        nx = ds.shape[1]
        ny = ds.shape[2]
        counts = np.zeros(n_frames, dtype=float)

        for i in range(min(n_frames, len(rois))):
            x0, x1, y0, y1 = rois[i]
            ix0 = max(0, min(nx, int(round(min(x0, x1)))))
            ix1 = max(0, min(nx, int(round(max(x0, x1)))))
            iy0 = max(0, min(ny, int(round(min(y0, y1)))))
            iy1 = max(0, min(ny, int(round(max(y0, y1)))))

            if ix1 > ix0 and iy1 > iy0:
                counts[i] = float(np.sum(ds[i, ix0:ix1, iy0:iy1]))

        return counts

    def auto_fit_detector_rois(
        self,
        initial_roi: tuple[float, float, float, float] | None = None,
        skip_first_n: int = 6,
    ) -> tuple[list[tuple[float, float, float, float]], np.ndarray] | None:
        """Automatically find reflection ROI for each frame and calculate integrated counts."""
        path = "entry0/data_scan/detector_data/data"
        if path not in self.file:
            return None
        ds = self.file[path]
        if ds.ndim != 3:
            return None

        if initial_roi is None:
            initial_roi = self.get_detector_roi() or (354.0, 382.0, 285.0, 743.0)

        from sa_nxs_explorer.core.scan_calculations import auto_fit_scan_rois

        return auto_fit_scan_rois(
            frame_provider=self,
            total_frames=ds.shape[0],
            initial_roi=initial_roi,
            skip_first_n=skip_first_n,
        )

    def parse_scan(self) -> ScanData:
        """Parse complete scan data: metadata, scanned variables, motors."""
        f = self.file
        entry = f["entry0"] if "entry0" in f else f[list(f.keys())[0]]

        # 1. Metadata
        def _get_scalar(group: h5py.Group, key: str, default: Any = None) -> Any:
            if key in group:
                try:
                    val = group[key][()]
                    return decode_val(val)
                except Exception:
                    return default
            return default

        run_num = _get_scalar(entry, "run_number")
        if isinstance(run_num, (list, np.ndarray)) and len(run_num) > 0:
            run_num = run_num[0]
        try:
            run_num_int = int(run_num) if run_num is not None else None
        except Exception:
            run_num_int = None

        subtitle_val = str(
            _get_scalar(entry, "subtitle", "")
            or _get_scalar(entry, "experiment_identifier", "")
            or _get_scalar(entry, "sub_title", "")
        )

        meta = ScanMetadata(
            run_number=run_num_int,
            title=str(_get_scalar(entry, "title", "")),
            subtitle=subtitle_val,
            duration=float(_get_scalar(entry, "duration", 0.0) or 0.0),
            mode=str(_get_scalar(entry, "modestring", "")),
            preset=float(_get_scalar(entry, "preset", 0.0) or 0.0),
            reactor_power=float(_get_scalar(entry, "reactor_power", 0.0) or 0.0),
            start_time=str(_get_scalar(entry, "start_time", "")),
            end_time=str(_get_scalar(entry, "end_time", "")),
        )

        # User metadata
        if "user" in entry and isinstance(entry["user"], h5py.Group):
            u = entry["user"]
            meta.user = str(_get_scalar(u, "name", ""))
            meta.local_contact = str(_get_scalar(u, "namelocalcontact", ""))
            meta.proposal = str(_get_scalar(u, "proposal", ""))

        # Sample metadata
        if "sample" in entry and isinstance(entry["sample"], h5py.Group):
            s = entry["sample"]
            meta.chemical_formula = str(_get_scalar(s, "chemicalFormula", ""))
            meta.sample_consistance = str(_get_scalar(s, "consistance", ""))
            meta.sample_id = str(_get_scalar(s, "sampleId", ""))
            meta.sample_mass = str(_get_scalar(s, "mass", ""))
            meta.sample_size = str(_get_scalar(s, "size", ""))
            meta.surface_area = str(_get_scalar(s, "surfaceArea", ""))

            # Environment variables
            if "additional_environment" in s and isinstance(s["additional_environment"], h5py.Group):
                env = s["additional_environment"]
                for env_name in env.keys():
                    sub = env[env_name]
                    if isinstance(sub, h5py.Group):
                        for k in sub.keys():
                            if isinstance(sub[k], h5py.Dataset):
                                try:
                                    meta.environment_variables[f"{env_name}/{k}"] = float(sub[k][()][0])
                                except Exception:
                                    pass

        # 2. Scanned Variables
        variables: dict[str, ScanVariable] = {}
        if "data_scan" in entry and "scanned_variables" in entry["data_scan"]:
            sv = entry["data_scan/scanned_variables"]
            if "data" in sv and "variables_names" in sv:
                raw_data = sv["data"][()]  # shape (N_vars, N_steps)
                vn = sv["variables_names"]
                names = [decode_attr_str(x) for x in vn["name"][()]]
                labels = [decode_attr_str(x) for x in vn["label"][()]] if "label" in vn else names
                units = [decode_attr_str(x) for x in vn["unit"][()]] if "unit" in vn else [""] * len(names)
                props = [decode_attr_str(x) for x in vn["property"][()]] if "property" in vn else [""] * len(names)
                axes = vn["axis"][()].tolist() if "axis" in vn else [0] * len(names)
                scanneds = vn["scanned"][()].tolist() if "scanned" in vn else [0] * len(names)

                # Explicit sibling arrays only: do not guess uncertainty channels
                # from their names. NeXus errors are standard deviations;
                # variance already contains squared uncertainties.
                variances = None
                for error_key in ("variance", "errors"):
                    if error_key in sv:
                        error_data = np.asarray(sv[error_key][()], dtype=float)
                        if error_data.shape != raw_data.shape:
                            raise ValueError(f"{sv.name}/{error_key} does not match scan data shape")
                        if error_key == "errors":
                            error_data = np.where(error_data >= 0, error_data**2, np.nan)
                        variances = error_data
                        break

                for i, name in enumerate(names):
                    var_data = raw_data[i]
                    unit = units[i] if i < len(units) else ""
                    unit_source = "file" if unit else "unspecified"
                    # SuperADAM writes its measured exposure as Count/Time
                    # with an empty unit field. Keep this interpretation explicit.
                    if (not unit and name == "Count" and "SUPERADAM" in entry
                            and i < len(labels) and labels[i] == "Time"
                            and i < len(props) and props[i] == "Time"):
                        unit = "s"
                        unit_source = "SuperADAM Count/Time convention (file unit absent)"
                    variables[name] = ScanVariable(
                        index=i,
                        name=name,
                        label=labels[i] if i < len(labels) else name,
                        unit=unit,
                        property=props[i] if i < len(props) else "",
                        axis=int(axes[i]) if i < len(axes) else 0,
                        is_scanned=bool(scanneds[i]) if i < len(scanneds) else False,
                        data=var_data,
                        variance=variances[i] if variances is not None else None,
                        unit_source=unit_source,
                    )

        # 3. Motors from entry0/SUPERADAM
        motors: dict[str, MotorInfo] = {}
        if "SUPERADAM" in entry and isinstance(entry["SUPERADAM"], h5py.Group):
            sa = entry["SUPERADAM"]
            for m_name in sa.keys():
                m_obj = sa[m_name]
                if isinstance(m_obj, h5py.Group):
                    val = _get_scalar(m_obj, "value")
                    tgt = _get_scalar(m_obj, "target_value")
                    off = _get_scalar(m_obj, "offset_value")
                    # Try units attribute
                    units_val = ""
                    if "value" in m_obj and "units" in m_obj["value"].attrs:
                        units_val = decode_attr_str(m_obj["value"].attrs["units"])
                    motors[m_name] = MotorInfo(
                        name=m_name,
                        value=float(val[0]) if isinstance(val, (list, np.ndarray)) and len(val) > 0 else (float(val) if isinstance(val, (int, float)) else val),
                        target_value=float(tgt[0]) if isinstance(tgt, (list, np.ndarray)) and len(tgt) > 0 else (float(tgt) if isinstance(tgt, (int, float)) else tgt),
                        offset_value=float(off[0]) if isinstance(off, (list, np.ndarray)) and len(off) > 0 else (float(off) if isinstance(off, (int, float)) else off),
                        units=units_val,
                    )

        # 4. Detector 2D status
        det_path = "entry0/data_scan/detector_data/data"
        has_det = det_path in f
        det_shape = f[det_path].shape if has_det else None
        roi = self.get_detector_roi()

        if has_det and roi:
            roi_counts = self.compute_full_scan_roi(roi)
            if roi_counts is not None:
                variables["ROI_Counts"] = ScanVariable(
                    index=len(variables),
                    name="ROI_Counts",
                    label="User ROI",
                    unit="counts",
                    property="Integrated ROI",
                    axis=0,
                    is_scanned=False,
                    data=roi_counts,
                )

        return ScanData(
            file_path=str(self.path),
            metadata=meta,
            variables=variables,
            motors=motors,
            has_detector_2d=has_det,
            detector_shape=det_shape,
            detector_roi=roi,
            detector_path=det_path if has_det else "",
        )
