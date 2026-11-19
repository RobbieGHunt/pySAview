"""Off-specular maps: HDF5 detector-volume integration along Y and display helpers.

Each detector frame is shown as ``displayed = raw.T`` (rows = Y, columns = X).
The specular reflection sits in a narrow X band, so off-specular scattering is
spread along X. Integrating every displayed column over a Y window gives one
X profile per scan point; stacking those profiles gives a (point, X pixel) map,
equivalent to the pySAred "Pixel vs. Point" 2D map.

Pure reduction calculations (Y-window bounds, 2D normalization, spin selection,
pairing, pooling, and asymmetry) live in :mod:`pySAview.core.reduction` and are
re-exported here for compatibility. Source data is only read; all results are
new in-memory arrays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from pySAview.core.reduction.normalization import (
    NORMALIZATION_UNITS,
    NormalizedMap,
    normalize_map,
)
from pySAview.core.reduction.polarization import (
    AsymmetryMap,
    SummedSpinMap,
    pair_spin_frames,
    select_frames,
    spin_asymmetry,
    sum_spin_frames,
)
from pySAview.core.reduction.roi_extraction import (
    frame_y_bounds,
    integrate_frames_along_y,
    pixel_bounds,
    y_pixel_bounds,
)

DETECTOR_PATH = "entry0/data_scan/detector_data/data"


@dataclass(frozen=True)
class OffSpecularMap:
    """Raw integrated counts for every frame, shape (n_frames, nx)."""

    counts: np.ndarray
    row_bounds: np.ndarray  # (n_frames, 2) clipped half-open displayed rows [iy0, iy1) per frame
    source_path: str = ""
    written: np.ndarray | None = None  # per-frame: chunk stored in the file (None = all)

    @property
    def y_bounds(self) -> tuple[int, int]:
        """Common Y window, or the (min start, max stop) envelope when it varies per frame."""
        return int(self.row_bounds[:, 0].min()), int(self.row_bounds[:, 1].max())

    @property
    def y_varies(self) -> bool:
        return bool(np.any(self.row_bounds != self.row_bounds[0]))

    @property
    def n_frames(self) -> int:
        return int(self.counts.shape[0])

    @property
    def nx(self) -> int:
        return int(self.counts.shape[1])


def _written_frames(dataset) -> np.ndarray | None:
    """Which frames were actually stored (one chunk per frame); None if unknown/all."""
    try:
        import h5py
        if not isinstance(dataset, h5py.Dataset) or dataset.chunks != (1, *dataset.shape[1:]):
            return None
        dsid = dataset.id
        return np.array([dsid.get_chunk_info_by_coord((i, 0, 0)).byte_offset is not None
                         for i in range(dataset.shape[0])], dtype=bool)
    except Exception:
        return None


def _supports_parallel_gzip(dataset) -> bool:
    """True for an h5py dataset stored as one gzip-only chunk per frame.

    That is how SuperADAM writes detector volumes. Anything else (other
    filters, shuffle, checksums, partial-frame chunks, non-numeric or
    byte-swapped types) uses the plain h5py read path.
    """
    try:
        import h5py
    except ImportError:  # pragma: no cover - h5py is a hard dependency
        return False
    if not isinstance(dataset, h5py.Dataset) or dataset.chunks != (1, *dataset.shape[1:]):
        return False
    if dataset.dtype.kind not in "iuf" or not dataset.dtype.isnative:
        return False
    if dataset.compression != "gzip" or dataset.shuffle or dataset.fletcher32 or dataset.scaleoffset is not None:
        return False
    plist = dataset.id.get_create_plist()
    return plist.get_nfilters() == 1 and plist.get_filter(0)[0] == h5py.h5z.FILTER_DEFLATE


def _integrate_gzip_frames_parallel(dataset, row_bounds, cancelled, progress) -> np.ndarray:
    """Read compressed frame chunks directly and inflate/sum them on worker threads.

    h5py serializes decompression behind its global lock; zlib releases the
    GIL, so inflating raw chunks in a thread pool scales with cores. Values are
    identical to ``dataset[...]`` reads: unwritten chunks take the dataset fill
    value and a chunk whose deflate filter was skipped is used as stored.
    """
    import zlib
    from concurrent.futures import ThreadPoolExecutor

    n_frames, nx, ny = dataset.shape
    frame_bytes = nx * ny * dataset.dtype.itemsize
    fill = dataset.fillvalue
    dsid = dataset.id

    def integrate(i: int) -> np.ndarray:
        if cancelled():
            raise InterruptedError("Off-specular integration cancelled")
        a, b = (int(v) for v in row_bounds[i])
        if b <= a:
            return np.zeros(nx)
        if dsid.get_chunk_info_by_coord((i, 0, 0)).byte_offset is None:
            # Chunk never written (e.g. an aborted scan): the frame reads as the fill value.
            return np.full(nx, float(fill) * (b - a))
        mask, raw = dsid.read_direct_chunk((i, 0, 0))
        data = raw if mask & 1 else zlib.decompress(raw)
        if len(data) != frame_bytes:
            raise ValueError(f"Detector frame {i} has an unexpected stored size")
        frame = np.frombuffer(data, dtype=dataset.dtype).reshape(nx, ny)
        return np.sum(frame[:, a:b], axis=1, dtype=np.float64)

    counts = np.zeros((n_frames, nx), dtype=float)
    workers = max(1, min(8, (os.cpu_count() or 2)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(integrate, i) for i in range(n_frames)]
        try:
            for i, future in enumerate(futures):
                counts[i] = future.result()
                if progress is not None:
                    progress(i + 1, n_frames)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return counts


def compute_off_specular_map(dataset, y_range=None, *, cancelled=None, progress=None, source_path="",
                             parallel=True) -> OffSpecularMap:
    """Chunked integration of an HDF5 (or array) detector volume over displayed Y.

    ``y_range`` is None (full height), one (y0, y1) window for every frame, or
    a sequence of per-frame windows (e.g. a tracked ROI whose Y bounds vary).
    Gzip frame-chunked HDF5 data is inflated on parallel threads
    (``parallel=False`` forces the plain h5py path, used to cross-check).
    """
    cancelled = cancelled or (lambda: False)
    if dataset.ndim != 3:
        raise ValueError("Detector data must be a 3D (frame, x, y) volume")
    n_frames, nx, ny = dataset.shape
    row_bounds = frame_y_bounds(y_range, n_frames, nx, ny)
    if parallel and _supports_parallel_gzip(dataset):
        counts = _integrate_gzip_frames_parallel(dataset, row_bounds, cancelled, progress)
        return OffSpecularMap(counts=counts, row_bounds=row_bounds, source_path=str(source_path),
                              written=_written_frames(dataset))
    counts = np.zeros((n_frames, nx), dtype=float)
    if np.any(row_bounds != row_bounds[0]):
        for i, (a, b) in enumerate(row_bounds):
            if cancelled():
                raise InterruptedError("Off-specular integration cancelled")
            if b > a:
                counts[i] = np.sum(np.asarray(dataset[i, :, a:b]), axis=1, dtype=np.float64)
            if progress is not None:
                progress(i + 1, n_frames)
        return OffSpecularMap(counts=counts, row_bounds=row_bounds, source_path=str(source_path),
                              written=_written_frames(dataset))
    iy0, iy1 = (int(v) for v in row_bounds[0])
    if iy1 > iy0:
        itemsize = getattr(getattr(dataset, "dtype", None), "itemsize", 8)
        bytes_per_frame = max(1, nx * (iy1 - iy0) * itemsize)
        chunk = max(1, min(16, (32 * 1024 * 1024) // bytes_per_frame))
        for start in range(0, n_frames, chunk):
            if cancelled():
                raise InterruptedError("Off-specular integration cancelled")
            stop = min(start + chunk, n_frames)
            block = np.asarray(dataset[start:stop, :, iy0:iy1])
            counts[start:stop] = np.sum(block, axis=2, dtype=np.float64)
            if progress is not None:
                progress(stop, n_frames)
    return OffSpecularMap(counts=counts, row_bounds=row_bounds, source_path=str(source_path),
                          written=_written_frames(dataset))


def log_display(values: np.ndarray) -> np.ndarray:
    """Display-only log10 transform.

    As for log chart axes (docs/normalization.md), zero/negative values cannot
    be shown and are omitted (NaN, drawn transparent), never replaced or
    clipped. The numerical map is not modified.
    """
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan)
    positive = np.isfinite(arr) & (arr > 0)
    out[positive] = np.log10(arr[positive])
    return out


__all__ = [
    "AsymmetryMap",
    "DETECTOR_PATH",
    "NORMALIZATION_UNITS",
    "NormalizedMap",
    "OffSpecularMap",
    "SummedSpinMap",
    "_integrate_gzip_frames_parallel",
    "_supports_parallel_gzip",
    "_written_frames",
    "compute_off_specular_map",
    "frame_y_bounds",
    "integrate_frames_along_y",
    "log_display",
    "normalize_map",
    "pair_spin_frames",
    "pixel_bounds",
    "select_frames",
    "spin_asymmetry",
    "sum_spin_frames",
    "y_pixel_bounds",
]
