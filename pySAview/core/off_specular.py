"""Off-specular maps: detector frames integrated along displayed Y, stacked by scan point.

Each detector frame is shown as ``displayed = raw.T`` (rows = Y, columns = X).
The specular reflection sits in a narrow X band, so off-specular scattering is
spread along X. Integrating every displayed column over a Y window gives one
X profile per scan point; stacking those profiles gives a (point, X pixel) map,
equivalent to the pySAred "Pixel vs. Point" 2D map.

Pixel selection follows the shared ROI convention in ``core.roi``: displayed
coordinates, nearest-integer bounds (ties to even), clipped, half-open slices,
whole pixels. Normalization follows docs/normalization.md. Source data is only
read; all results are new in-memory arrays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from pySAview.core.roi import pixel_bounds

DETECTOR_PATH = "entry0/data_scan/detector_data/data"

NORMALIZATION_UNITS = {
    "raw": "counts",
    "time": "counts/s",
    "monitor": "detector counts per monitor count",
}


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


@dataclass
class NormalizedMap:
    """Display-ready values for the selected frames, with first-order 1σ uncertainty."""

    frame_indices: np.ndarray
    values: np.ndarray
    standard_uncertainty: np.ndarray
    mode: str
    units: str
    warnings: list[str] = field(default_factory=list)


def y_pixel_bounds(y_range: tuple[float, float] | None, nx: int, ny: int) -> tuple[int, int]:
    """Clip a displayed Y window with the shared ROI convention; ``None`` means full height."""
    if y_range is None:
        return 0, ny
    _, _, iy0, iy1 = pixel_bounds((0.0, float(nx), float(y_range[0]), float(y_range[1])), nx, ny)
    return iy0, iy1


def integrate_frames_along_y(raw_frames: np.ndarray, y_range: tuple[float, float] | None = None) -> np.ndarray:
    """Integrate raw frames (n, raw_axis1, raw_axis2) over displayed Y.

    Displayed X is raw axis 1 and displayed Y is raw axis 2, so the result
    has shape (n, nx) with ``result[i, x] = sum(displayed_i[iy0:iy1, x])``.
    """
    frames = np.asarray(raw_frames)
    if frames.ndim != 3:
        raise ValueError("Detector data must be a 3D (frame, x, y) volume")
    nx, ny = frames.shape[1], frames.shape[2]
    iy0, iy1 = y_pixel_bounds(y_range, nx, ny)
    if iy1 <= iy0:
        return np.zeros((frames.shape[0], nx), dtype=float)
    return np.sum(frames[:, :, iy0:iy1], axis=2, dtype=np.float64)


def frame_y_bounds(y_range, n_frames: int, nx: int, ny: int) -> np.ndarray:
    """Clipped (n_frames, 2) Y windows from None, one (y0, y1) pair, or one pair per frame."""
    if y_range is None or (len(y_range) == 2 and np.isscalar(y_range[0])):
        return np.tile(np.array(y_pixel_bounds(y_range, nx, ny), dtype=int), (n_frames, 1))
    if len(y_range) != n_frames:
        raise ValueError("Per-frame Y windows must supply exactly one window per detector frame")
    return np.array([y_pixel_bounds(tuple(r), nx, ny) for r in y_range], dtype=int).reshape(n_frames, 2)


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


def select_frames(n_frames: int, flipper: np.ndarray | None, spin: str) -> np.ndarray:
    """Frame indices for 'all', 'up' (Flipper1 == 1) or 'down' (Flipper1 == 0)."""
    every = np.arange(n_frames)
    if spin == "all" or flipper is None:
        return every
    state = np.asarray(flipper, dtype=float)[:n_frames]
    wanted = 1.0 if spin == "up" else 0.0
    return np.flatnonzero(state == wanted)


def normalize_map(
    counts: np.ndarray,
    frame_indices: np.ndarray,
    mode: str = "raw",
    *,
    exposure: np.ndarray | None = None,
    monitor: np.ndarray | None = None,
    monitor_variance: np.ndarray | None = None,
) -> NormalizedMap:
    """Normalize selected rows of an integrated map.

    raw:     N,      var = N
    time:    N / t,  var = N / t**2           (t known exposure in seconds)
    monitor: N / M,  var = N / M**2 + N**2 * v_M / M**4  (v_M = M unless supplied)

    Rows with missing, zero, negative or nonfinite denominators become NaN
    and are reported in ``warnings``; nothing is clipped or substituted.
    """
    if mode not in NORMALIZATION_UNITS:
        raise ValueError(f"Unknown normalization mode: {mode}")
    idx = np.asarray(frame_indices, dtype=int)
    n = np.asarray(counts, dtype=float)[idx]
    warnings: list[str] = []
    variance = n.copy()

    if mode == "raw":
        values = n.copy()
    else:
        source = exposure if mode == "time" else monitor
        name = "exposure time" if mode == "time" else "Monitor1"
        if source is None:
            raise ValueError(f"{name} is required for {mode} normalization")
        source = np.asarray(source, dtype=float)
        if source.shape[0] < (idx.max() + 1 if idx.size else 0):
            raise ValueError(f"{name} has fewer points than the detector has frames")
        denom = source[idx][:, None]
        valid = np.isfinite(denom) & (denom > 0)
        safe = np.where(valid, denom, np.nan)
        if mode == "time":
            values = n / safe
            variance = n / safe**2
        else:
            v_m = safe
            if monitor_variance is not None:
                v_m = np.asarray(monitor_variance, dtype=float)[idx][:, None]
                bad_var = ~(np.isfinite(v_m) & (v_m >= 0))
                valid = valid & ~bad_var
                v_m = np.where(bad_var, np.nan, v_m)
            values = np.where(valid, n / safe, np.nan)
            variance = n / safe**2 + n**2 * v_m / safe**4
        bad_rows = int(np.count_nonzero(~valid[:, 0]))
        if bad_rows:
            warnings.append(f"{bad_rows} point(s) excluded: invalid {name} or its variance.")

    return NormalizedMap(
        frame_indices=idx,
        values=values,
        standard_uncertainty=np.sqrt(variance),
        mode=mode,
        units=NORMALIZATION_UNITS[mode],
        warnings=warnings,
    )


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


def pair_spin_frames(th, tth, flipper, written=None, tol_deg: float = 2e-3):
    """Pair each spin-up frame (Flipper1 = 1) with an unused spin-down frame (0) at the same θ and 2θ.

    The nearest-in-sequence down frame within ``tol_deg`` in both angles is
    used; unwritten frames are never paired. Returns (up, down) index arrays.
    """
    th = np.asarray(th, dtype=float)
    tth = np.asarray(tth, dtype=float)
    flipper = np.asarray(flipper, dtype=float)
    n = min(th.size, tth.size, flipper.size)
    usable = np.isfinite(th[:n]) & np.isfinite(tth[:n])
    if written is not None:
        usable &= np.asarray(written, dtype=bool)[:n]
    ups = np.flatnonzero(usable & (flipper[:n] == 1))
    downs = np.flatnonzero(usable & (flipper[:n] == 0))
    taken = np.zeros(downs.size, dtype=bool)
    pairs = []
    for u in ups:
        match = (~taken & (np.abs(th[downs] - th[u]) <= tol_deg) & (np.abs(tth[downs] - tth[u]) <= tol_deg))
        if not np.any(match):
            continue
        candidates = np.flatnonzero(match)
        k = candidates[np.argmin(np.abs(downs[candidates] - u))]
        taken[k] = True
        pairs.append((u, downs[k]))
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


@dataclass
class AsymmetryMap(NormalizedMap):
    """Spin asymmetry rows: ``frame_indices`` are the up frames, ``partner_indices`` the down frames."""

    partner_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    low_count_mask: np.ndarray | None = None  # True where raw N_u + N_d is below the display threshold
    min_counts: float = 0.0


def spin_asymmetry(counts, up, down, mode: str = "raw", *, exposure=None, monitor=None,
                   monitor_variance=None, min_counts: float = 0.0) -> AsymmetryMap:
    """Per-pixel spin asymmetry SA = (I_u − I_d) / (I_u + I_d) of paired frames.

    I_u and I_d are the selected normalization of each frame (raw, per second or
    per monitor, as in ``normalize_map``), so unequal exposures cancel when a
    normalization is chosen. First-order 1σ uncertainty for independent u/d:

        var(SA) = (2 I_d / S²)² var(I_u) + (2 I_u / S²)² var(I_d),  S = I_u + I_d

    Pixels with S ≤ 0 or invalid normalization are NaN. ``low_count_mask``
    flags pixels with fewer than ``min_counts`` raw counts in u + d; it is for
    display only and does not change the values.
    """
    up = np.asarray(up, dtype=int)
    down = np.asarray(down, dtype=int)
    kw = dict(exposure=exposure, monitor=monitor, monitor_variance=monitor_variance)
    iu = normalize_map(counts, up, mode, **kw)
    idn = normalize_map(counts, down, mode, **kw)
    s = iu.values + idn.values
    valid = np.isfinite(s) & (s > 0)
    safe = np.where(valid, s, np.nan)
    sa = (iu.values - idn.values) / safe
    var = ((2 * idn.values / safe**2) ** 2 * iu.standard_uncertainty**2
           + (2 * iu.values / safe**2) ** 2 * idn.standard_uncertainty**2)
    raw = np.asarray(counts, dtype=float)
    low = (raw[up] + raw[down]) < min_counts
    warnings = list(dict.fromkeys(iu.warnings + idn.warnings))
    return AsymmetryMap(frame_indices=up, values=sa, standard_uncertainty=np.sqrt(var), mode=mode,
                        units="spin asymmetry (u−d)/(u+d)", warnings=warnings,
                        partner_indices=down, low_count_mask=low, min_counts=float(min_counts))
