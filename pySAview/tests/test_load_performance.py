"""Fast ROI integration must equal plain h5py reads; heavy imports stay off the launch path."""

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from pySAview.core.nexus_reader import NexusReader, _roi_fast_path_supported

DETECTOR_PATH = "entry0/data_scan/detector_data/data"
EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "example_data"


def _write_detector(path, frames, written, fillvalue=0, **storage):
    with h5py.File(path, "w") as handle:
        ds = handle.create_dataset(DETECTOR_PATH, shape=frames.shape, dtype=frames.dtype,
                                   chunks=(1, *frames.shape[1:]), fillvalue=fillvalue, **storage)
        for i in written:  # the rest stay unallocated, like a scan still being written
            ds[i] = frames[i]


ROIS = [
    (2, 7, 3, 11),       # interior box
    (0, 12, 0, 14),      # full frame: inflation cannot stop early
    (9.5, 12, 0, 14),    # bottom rows only, half-pixel bound
    (-5, 3, 10, 99),     # clipped at edges
    (4, 4, 0, 14),       # empty
]


@pytest.mark.parametrize("roi", ROIS)
@pytest.mark.parametrize("fillvalue", [0, 7])
def test_parallel_roi_matches_h5py_reads(tmp_path, roi, fillvalue):
    frames = np.random.default_rng(5).poisson(40, size=(9, 12, 14)).astype(np.uint32)
    path = tmp_path / "gzip.nxs"
    _write_detector(path, frames, written=[0, 1, 2, 4, 5, 8], fillvalue=fillvalue,
                    compression="gzip", compression_opts=6)
    with NexusReader(path) as reader:
        assert _roi_fast_path_supported(reader.file[DETECTOR_PATH])
        fast = reader.compute_full_scan_roi(roi)
        slow = reader.compute_full_scan_roi(roi, parallel=False)
        np.testing.assert_array_equal(fast, slow)
        per_frame = [roi, (0, 3, 0, 5), (5, 12, 2, 9), roi, (11, 12, 13, 14), (0, 0, 0, 0), roi, (1, 6, 1, 6), roi]
        np.testing.assert_array_equal(reader.compute_per_frame_scan_roi(per_frame),
                                      reader.compute_per_frame_scan_roi(per_frame, parallel=False))
    # The h5py path itself: unwritten frames read as the fill value.
    expected = frames.copy()
    expected[[3, 6, 7]] = fillvalue
    from pySAview.core.roi import pixel_bounds
    ix0, ix1, iy0, iy1 = pixel_bounds(roi, 12, 14)
    np.testing.assert_array_equal(slow, expected[:, ix0:ix1, iy0:iy1].sum(axis=(1, 2)))


@pytest.mark.parametrize("dtype,storage", [
    (np.uint16, dict(compression="gzip", shuffle=True)),
    (np.uint16, dict(compression="lzf")),
    (np.uint32, dict()),
    (np.float32, dict(compression="gzip")),  # float sums are order-dependent: keep h5py order
])
def test_other_layouts_use_h5py_path(tmp_path, dtype, storage):
    frames = np.random.default_rng(6).poisson(5, size=(4, 8, 8)).astype(dtype)
    path = tmp_path / "other.nxs"
    _write_detector(path, frames, written=range(4), **storage)
    with NexusReader(path) as reader:
        assert not _roi_fast_path_supported(reader.file[DETECTOR_PATH])
        np.testing.assert_array_equal(reader.compute_full_scan_roi((1, 6, 2, 7)),
                                      frames[:, 1:6, 2:7].sum(axis=(1, 2), dtype=np.float64))


def test_parallel_roi_cancellation(tmp_path):
    frames = np.ones((5, 6, 6), dtype=np.uint32)
    path = tmp_path / "cancel.nxs"
    _write_detector(path, frames, written=range(5), compression="gzip")
    with NexusReader(path, cancelled=lambda: True) as reader:
        with pytest.raises(InterruptedError):
            reader.compute_full_scan_roi((0, 6, 0, 6))


@pytest.mark.parametrize("name", ["005572.nxs", "005594.nxs"])
def test_example_files_parallel_roi_is_exact(name):
    with NexusReader(EXAMPLE_DIR / name) as reader:
        roi = reader.get_detector_roi()
        assert _roi_fast_path_supported(reader.file[DETECTOR_PATH])
        np.testing.assert_array_equal(reader.compute_full_scan_roi(roi),
                                      reader.compute_full_scan_roi(roi, parallel=False))


def test_launch_does_not_import_scipy_stats():
    code = "import sys, pySAview.ui.main_window; print('scipy.stats' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
