"""NeXus HDF5 Reader for SuperADAM and neutron reflectometry scan files."""

from __future__ import annotations

import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
import h5py
import numpy as np
from pySAview.core.roi import pixel_bounds

from pySAview.core.models import (
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


_INFLATE_STEP_BYTES = 131_072  # 128 KiB: stays in L2 cache and below Windows VirtualAlloc threshold
_MAX_BATCH_FRAMES = 512
_MAX_BATCH_BYTES = 16 * 1024 * 1024


def _roi_fast_path_supported(dataset: h5py.Dataset) -> bool:
    """Integer frame-per-chunk deflate volumes, where per-frame sums are exact."""
    from pySAview.core.off_specular import _supports_parallel_gzip

    return dataset.dtype.kind in "iu" and _supports_parallel_gzip(dataset)


def _can_use_raw_deflate(raw: bytes, needed: int, frame_bytes: int) -> bool:
    """True when a partial read can safely use raw deflate without unverified Adler-32."""
    if needed >= frame_bytes or len(raw) < 2:
        return False
    cmf = raw[0]
    flg = raw[1]
    return (
        (cmf & 0x0F) == 8
        and (cmf >> 4) <= 7
        and ((cmf << 8) | flg) % 31 == 0
        and (flg & 0x20) == 0
    )


def _integrate_roi_gzip_parallel(dataset: h5py.Dataset, bounds, cancelled) -> np.ndarray:
    """Integrate one raw-pixel box per frame by inflating stored chunks on worker threads.

    ``bounds[i]`` is ``(ix0, ix1, iy0, iy1)`` in raw (axis 1, axis 2) pixels.
    Frames are row-major, so inflation stops after raw row ``ix1``: the rest of
    the frame cannot contribute. Chunks are read serially in memory-bounded
    macro-batches (avoiding h5py global-lock contention across workers) and
    inflated in 128 KiB row-aligned strips on worker threads where zlib
    releases the GIL. For partial frames with standard RFC-1950 zlib headers,
    raw deflate (``wbits=-15``) skips computing an Adler-32 checksum whose
    trailer is never reached. Results equal ``np.sum(dataset[i, ix0:ix1, iy0:iy1])``
    exactly: integer sums, unwritten chunks read as the fill value, and a chunk
    whose deflate filter was skipped is used as stored. Source data is only read.
    """
    n_frames, nx, ny = dataset.shape
    dtype = dataset.dtype
    row_bytes = ny * dtype.itemsize
    frame_bytes = nx * row_bytes
    fill = int(dataset.fillvalue)
    dsid = dataset.id
    step = _INFLATE_STEP_BYTES
    strip_rows = max(1, step // row_bytes)

    def _integrate_batch(
        batch: list[tuple[int, int, int, int, int, int, bytes]],
    ) -> list[tuple[int, float]]:
        results: list[tuple[int, float]] = []
        for i, ix0, ix1, iy0, iy1, mask, raw in batch:
            if cancelled():
                raise InterruptedError("File loading cancelled")
            needed = ix1 * row_bytes
            if mask & 1:
                if len(raw) < needed:
                    raise ValueError(f"Detector frame {i} has an unexpected stored size")
                rows = np.frombuffer(
                    raw,
                    dtype=dtype,
                    count=(ix1 - ix0) * ny,
                    offset=ix0 * row_bytes,
                ).reshape(ix1 - ix0, ny)
                results.append((i, float(int(np.sum(rows[:, iy0:iy1])))))
                continue

            if _can_use_raw_deflate(raw, needed, frame_bytes):
                decompressor = zlib.decompressobj(-15)
                buf: bytes | memoryview = memoryview(raw)[2:]
            else:
                decompressor = zlib.decompressobj()
                buf = raw

            # Discard leading rows 0..ix0 in bounded chunks without allocating the whole prefix.
            rem = ix0 * row_bytes
            while rem > 0:
                out = decompressor.decompress(buf, step if rem > step else rem)
                if not out:
                    raise ValueError(f"Detector frame {i} has an unexpected stored size")
                buf = decompressor.unconsumed_tail
                rem -= len(out)

            # Inflate ROI rows ix0..ix1 in bounded row-aligned strips and sum exactly.
            rows_left = ix1 - ix0
            total = 0
            while rows_left > 0:
                take_rows = strip_rows if rows_left > strip_rows else rows_left
                take_bytes = take_rows * row_bytes
                chunk = decompressor.decompress(buf, take_bytes)
                buf = decompressor.unconsumed_tail
                if len(chunk) < take_bytes:
                    parts = [chunk]
                    got = len(chunk)
                    while got < take_bytes:
                        more = decompressor.decompress(buf, take_bytes - got)
                        if not more:
                            raise ValueError(f"Detector frame {i} has an unexpected stored size")
                        parts.append(more)
                        got += len(more)
                        buf = decompressor.unconsumed_tail
                    chunk = b"".join(parts)
                rows = np.frombuffer(chunk, dtype=dtype).reshape(take_rows, ny)
                total += int(np.sum(rows[:, iy0:iy1]))
                rows_left -= take_rows
            results.append((i, float(total)))
        return results

    counts = np.empty(n_frames, dtype=float)
    workers = max(1, min(8, os.cpu_count() or 2))
    pool: ThreadPoolExecutor | None = None

    def _flush_work(work: list[tuple[int, int, int, int, int, int, bytes]]) -> None:
        nonlocal pool
        if not work:
            return
        if len(work) <= 8 or workers == 1:
            for idx, val in _integrate_batch(work):
                counts[idx] = val
            return
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=workers)
        # Strided round-robin balances dense low-angle and sparse high-angle frames evenly.
        batches = [work[t::workers] for t in range(workers) if work[t::workers]]
        futures = [pool.submit(_integrate_batch, batch) for batch in batches]
        try:
            for future in futures:
                for idx, val in future.result():
                    counts[idx] = val
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    work: list[tuple[int, int, int, int, int, int, bytes]] = []
    batch_bytes = 0
    try:
        for i in range(n_frames):
            if cancelled():
                raise InterruptedError("File loading cancelled")
            ix0, ix1, iy0, iy1 = bounds[i]
            if ix1 <= ix0 or iy1 <= iy0:
                counts[i] = 0.0
                continue
            if dsid.get_chunk_info_by_coord((i, 0, 0)).byte_offset is None:
                counts[i] = float(fill * (ix1 - ix0) * (iy1 - iy0))
                continue
            mask, raw = dsid.read_direct_chunk((i, 0, 0))
            work.append((i, ix0, ix1, iy0, iy1, mask, raw))
            batch_bytes += len(raw)
            if len(work) >= _MAX_BATCH_FRAMES or batch_bytes >= _MAX_BATCH_BYTES:
                _flush_work(work)
                work.clear()
                batch_bytes = 0
        _flush_work(work)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return counts


class NexusReader:
    """Read-only lazy accessor for NeXus (.nxs) files."""

    _DETECTOR_PATH = "entry0/data_scan/detector_data/data"

    def __init__(self, path: str | Path, *, cancelled=None) -> None:
        self._cancelled = cancelled or (lambda: False)
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        self._file: h5py.File | None = h5py.File(self.path, "r")
        self._detector_ds: h5py.Dataset | None = None
        self._detector_ds_checked: bool = False

    def close(self) -> None:
        self._detector_ds = None
        self._detector_ds_checked = False
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def _get_detector_dataset(self) -> h5py.Dataset | None:
        """Return cached read-only detector 3D dataset handle (or None)."""
        f = self.file
        if not self._detector_ds_checked:
            ds = f.get(self._DETECTOR_PATH)
            self._detector_ds = ds if isinstance(ds, h5py.Dataset) else None
            self._detector_ds_checked = True
        return self._detector_ds

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
        ds = self._get_detector_dataset()
        if ds is None or ds.ndim != 3:
            return None
        total_frames = ds.shape[0]
        if frame_index < 0 or frame_index >= total_frames:
            raise IndexError(f"Frame index {frame_index} out of bounds (total frames: {total_frames})")
        return ds[frame_index]

    def get_detector_roi(self) -> tuple[float, float, float, float] | None:
        """Get (roi_x0, roi_x1, roi_y0, roi_y1) if present in SUPERADAM/Detector1."""
        det = self.file.get("entry0/SUPERADAM/Detector1")
        if det is None:
            return None
        try:
            x0 = float(det["roi_x0"][()][0])
            x1 = float(det["roi_x1"][()][0])
            y0 = float(det["roi_y0"][()][0])
            y1 = float(det["roi_y1"][()][0])
            return (x0, x1, y0, y1)
        except Exception:
            return None

    def get_virtual_slit_widths(self) -> tuple[float, float] | None:
        """Read scalar horizontal slit openings ``(s1h, s2h)`` in mm from ``SUPERADAM/VirtualSlit``."""
        vs = self.file.get("entry0/SUPERADAM/VirtualSlit")
        if not isinstance(vs, h5py.Group):
            return None
        try:
            s1_raw = vs["s1h_actual_width"][()]
            s2_raw = vs["s2h_actual_width"][()]
            s1 = float(np.asarray(s1_raw).flat[0])
            s2 = float(np.asarray(s2_raw).flat[0])
            if np.isfinite(s1) and np.isfinite(s2) and s1 > 0 and s2 > 0:
                return (round(s1, 8), round(s2, 8))
        except Exception:
            return None
        return None

    def compute_y_integrated_map(self, y_range: tuple[float, float] | None = None):
        """Return an ``OffSpecularMap`` integrated along ``y_range`` for the 2D detector."""
        ds = self._get_detector_dataset()
        if ds is None or ds.ndim != 3:
            return None
        from pySAview.core.off_specular import compute_off_specular_map

        return compute_off_specular_map(
            ds,
            y_range=y_range,
            cancelled=self._cancelled,
            source_path=str(self.path),
        )

    def compute_full_scan_roi(
        self, roi: tuple[float, float, float, float], *, parallel: bool = True
    ) -> np.ndarray | None:
        """Compute integrated counts for each scan frame within ROI (x0, x1, y0, y1).

        Coordinates (x, y) are in the displayed frame orientation (where displayed_frame = raw_frame.T),
        so the raw 3D volume is sliced along raw_data[:, ix0:ix1, iy0:iy1].
        """
        ds = self._get_detector_dataset()
        if ds is None or ds.ndim != 3:
            return None

        ix0, ix1, iy0, iy1 = pixel_bounds(roi, ds.shape[1], ds.shape[2])

        if ix1 <= ix0 or iy1 <= iy0:
            return np.zeros(ds.shape[0], dtype=float)

        if parallel and _roi_fast_path_supported(ds):
            return _integrate_roi_gzip_parallel(ds, [(ix0, ix1, iy0, iy1)] * ds.shape[0], self._cancelled)

        counts = np.empty(ds.shape[0], dtype=float)
        bytes_per_frame = max(1, (ix1-ix0) * (iy1-iy0) * ds.dtype.itemsize)
        chunk_frames = max(1, min(16, (8 * 1024 * 1024) // bytes_per_frame))
        for start in range(0, ds.shape[0], chunk_frames):
            if self._cancelled():
                raise InterruptedError("File loading cancelled")
            stop = min(start + chunk_frames, ds.shape[0])
            sub = ds[start:stop, ix0:ix1, iy0:iy1]
            counts[start:stop] = np.sum(sub, axis=(1, 2))
        return counts

    def compute_per_frame_scan_roi(
        self, rois: list[tuple[float, float, float, float]], *, parallel: bool = True
    ) -> np.ndarray | None:
        """Compute integrated counts for each scan frame given per-frame ROIs [(x0, x1, y0, y1), ...]."""
        ds = self._get_detector_dataset()
        if ds is None or ds.ndim != 3:
            return None

        n_frames = ds.shape[0]
        nx = ds.shape[1]
        ny = ds.shape[2]
        counts = np.zeros(n_frames, dtype=float)

        if len(rois) != n_frames:
            raise ValueError("Tracked ROI must supply exactly one region per detector frame")
        bounds = [pixel_bounds(roi, nx, ny) for roi in rois]
        if parallel and _roi_fast_path_supported(ds):
            return _integrate_roi_gzip_parallel(ds, bounds, self._cancelled)
        for i, (ix0, ix1, iy0, iy1) in enumerate(bounds):
            if ix1 > ix0 and iy1 > iy0:
                counts[i] = float(np.sum(ds[i, ix0:ix1, iy0:iy1]))

        return counts

    def auto_fit_detector_rois(
        self,
        initial_roi: tuple[float, float, float, float] | None = None,
        skip_first_n: int = 6,
    ) -> tuple[list[tuple[float, float, float, float]], np.ndarray] | None:
        """Automatically find reflection ROI for each frame and calculate integrated counts."""
        ds = self._get_detector_dataset()
        if ds is None or ds.ndim != 3:
            return None

        if initial_roi is None:
            initial_roi = self.get_detector_roi() or (354.0, 382.0, 285.0, 743.0)

        from pySAview.core.scan_calculations import auto_fit_scan_rois

        return auto_fit_scan_rois(
            frame_provider=self,
            total_frames=ds.shape[0],
            initial_roi=initial_roi,
            skip_first_n=skip_first_n,
        )

    def parse_scan(self) -> ScanData:
        """Parse complete scan data: metadata, scanned variables, motors."""
        if self._cancelled():
            raise InterruptedError("File loading cancelled")
        f = self.file
        entry = f.get("entry0")
        if entry is None:
            entry = f[next(iter(f.keys()))]

        # 1. Metadata
        def _read_ds_scalar(ds: Any, default: Any = None) -> Any:
            if ds is None:
                return default
            try:
                return decode_val(ds[()])
            except Exception:
                return default

        def _get_scalar(group: h5py.Group, key: str, default: Any = None) -> Any:
            return _read_ds_scalar(group.get(key), default)

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
        u = entry.get("user")
        if isinstance(u, h5py.Group):
            meta.user = str(_get_scalar(u, "name", ""))
            meta.local_contact = str(_get_scalar(u, "namelocalcontact", ""))
            meta.proposal = str(_get_scalar(u, "proposal", ""))

        # Sample metadata
        s = entry.get("sample")
        if isinstance(s, h5py.Group):
            meta.chemical_formula = str(_get_scalar(s, "chemicalFormula", ""))
            meta.sample_consistance = str(_get_scalar(s, "consistance", ""))
            meta.sample_id = str(_get_scalar(s, "sampleId", ""))
            meta.sample_mass = str(_get_scalar(s, "mass", ""))
            meta.sample_size = str(_get_scalar(s, "size", ""))
            meta.surface_area = str(_get_scalar(s, "surfaceArea", ""))

            # Environment variables
            env = s.get("additional_environment")
            if isinstance(env, h5py.Group):
                for env_name, sub in env.items():
                    if isinstance(sub, h5py.Group):
                        for k, sub_item in sub.items():
                            if isinstance(sub_item, h5py.Dataset):
                                try:
                                    meta.environment_variables[f"{env_name}/{k}"] = float(sub_item[()][0])
                                except Exception:
                                    pass

        sa = entry.get("SUPERADAM")
        has_superadam = isinstance(sa, h5py.Group)

        # 2. Scanned Variables
        variables: dict[str, ScanVariable] = {}
        sv = entry.get("data_scan/scanned_variables")
        if isinstance(sv, h5py.Group):
            sv_data = sv.get("data")
            vn = sv.get("variables_names")
            if sv_data is not None and isinstance(vn, h5py.Group):
                raw_data = sv_data[()]  # shape (N_vars, N_steps)
                names = [decode_attr_str(x) for x in vn["name"][()]]
                lbl_ds = vn.get("label")
                labels = [decode_attr_str(x) for x in lbl_ds[()]] if lbl_ds is not None else names
                unit_ds = vn.get("unit")
                units = [decode_attr_str(x) for x in unit_ds[()]] if unit_ds is not None else [""] * len(names)
                prop_ds = vn.get("property")
                props = [decode_attr_str(x) for x in prop_ds[()]] if prop_ds is not None else [""] * len(names)
                axis_ds = vn.get("axis")
                axes = axis_ds[()].tolist() if axis_ds is not None else [0] * len(names)
                scanned_ds = vn.get("scanned")
                scanneds = scanned_ds[()].tolist() if scanned_ds is not None else [0] * len(names)

                # Explicit sibling arrays only: do not guess uncertainty channels
                # from their names. NeXus errors are standard deviations;
                # variance already contains squared uncertainties.
                variances = None
                for error_key in ("variance", "errors"):
                    err_ds = sv.get(error_key)
                    if err_ds is not None:
                        error_data = np.asarray(err_ds[()], dtype=float)
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
                    if (not unit and name == "Count" and has_superadam
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
        if has_superadam:
            for m_name, m_obj in sa.items():
                if isinstance(m_obj, h5py.Group):
                    val_ds = m_obj.get("value")
                    val = _read_ds_scalar(val_ds)
                    tgt = _read_ds_scalar(m_obj.get("target_value"))
                    off = _read_ds_scalar(m_obj.get("offset_value"))
                    # Try units attribute without re-opening the value dataset
                    units_val = ""
                    if val_ds is not None:
                        units_attr = val_ds.attrs.get("units")
                        if units_attr is not None:
                            units_val = decode_attr_str(units_attr)
                    motors[m_name] = MotorInfo(
                        name=m_name,
                        value=float(val[0]) if isinstance(val, (list, np.ndarray)) and len(val) > 0 else (float(val) if isinstance(val, (int, float)) else val),
                        target_value=float(tgt[0]) if isinstance(tgt, (list, np.ndarray)) and len(tgt) > 0 else (float(tgt) if isinstance(tgt, (int, float)) else tgt),
                        offset_value=float(off[0]) if isinstance(off, (list, np.ndarray)) and len(off) > 0 else (float(off) if isinstance(off, (int, float)) else off),
                        units=units_val,
                    )

        # 4. Detector 2D status
        det_path = self._DETECTOR_PATH
        det_ds = self._get_detector_dataset()
        has_det = det_ds is not None
        det_shape = det_ds.shape if has_det else None
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

        slit_defaults = self.get_virtual_slit_widths()
        from pySAview.core.reduction.background import default_bkg_x_bounds
        from pySAview.core.reduction.pysared import ReductionParams

        red_params = ReductionParams()
        if has_det and roi and det_shape is not None:
            red_params.bkg_x_bounds = default_bkg_x_bounds(roi[0], roi[1], nx=int(det_shape[1]))

        return ScanData(
            file_path=str(self.path),
            metadata=meta,
            variables=variables,
            motors=motors,
            has_detector_2d=has_det,
            detector_shape=det_shape,
            detector_roi=roi,
            detector_path=det_path if has_det else "",
            reduction_params=red_params,
            slit_defaults=slit_defaults,
        )
