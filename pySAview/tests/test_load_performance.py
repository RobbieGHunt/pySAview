"""Fast ROI integration must equal plain h5py reads; heavy imports stay off the launch path."""

import os
import subprocess
import sys
import time
import zlib
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
    code = (
        "import sys, pySAview.ui.main_window; "
        "print('scipy.stats' in sys.modules, 'scipy.special' in sys.modules, 'matplotlib' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False False"


def test_app_help_does_not_import_qt():
    code = (
        "import sys, pySAview.app; "
        "sys.argv = ['pySAview', '--help'];\n"
        "try:\n"
        "    pySAview.app.main()\n"
        "except SystemExit:\n"
        "    pass\n"
        "print('PyQt6' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().endswith("False")


def test_parallel_roi_multi_strip_and_uncompressed_chunks(tmp_path):
    """Exercise >128 KiB multi-strip inflation, strided workers, unwritten chunks, and mask&1 chunks."""
    rng = np.random.default_rng(19)
    # Each frame is 128 x 512 x 4 = 256 KiB (> 128 KiB inflate strip size); 24 frames triggers thread pool.
    frames = rng.poisson(12, size=(24, 128, 512)).astype(np.uint32)
    path = tmp_path / "multistrip.nxs"
    written = [i for i in range(24) if i not in (3, 11, 20)]
    with h5py.File(path, "w") as handle:
        ds = handle.create_dataset(
            DETECTOR_PATH,
            shape=frames.shape,
            dtype=frames.dtype,
            chunks=(1, *frames.shape[1:]),
            fillvalue=3,
            compression="gzip",
            compression_opts=4,
        )
        for i in written:
            if i in (5, 17):
                # Store uncompressed chunk with filter_mask=1 (deflate filter skipped).
                ds.id.write_direct_chunk((i, 0, 0), frames[i].tobytes(), filter_mask=1)
            else:
                ds[i] = frames[i]

    rois = [
        (10, 100, 50, 450),     # 90 rows = 180 KiB (multi-strip partial frame)
        (0, 128, 0, 512),       # full frame (Adler-32 verified path, multi-strip)
        (95, 128, 120, 380),    # tail rows after multi-chunk skip
    ]
    with NexusReader(path) as reader:
        assert _roi_fast_path_supported(reader.file[DETECTOR_PATH])
        for roi in rois:
            fast = reader.compute_full_scan_roi(roi, parallel=True)
            slow = reader.compute_full_scan_roi(roi, parallel=False)
            np.testing.assert_array_equal(fast, slow)
        tracked = [rois[i % len(rois)] for i in range(24)]
        np.testing.assert_array_equal(
            reader.compute_per_frame_scan_roi(tracked, parallel=True),
            reader.compute_per_frame_scan_roi(tracked, parallel=False),
        )


def test_parallel_roi_rejects_truncated_chunk(tmp_path):
    frames = np.ones((2, 16, 16), dtype=np.uint32)
    path = tmp_path / "truncated.nxs"
    with h5py.File(path, "w") as handle:
        ds = handle.create_dataset(
            DETECTOR_PATH,
            shape=frames.shape,
            dtype=frames.dtype,
            chunks=(1, 16, 16),
            compression="gzip",
        )
        ds[0] = frames[0]
        # Valid zlib stream, but only 4 rows instead of 16.
        ds.id.write_direct_chunk((1, 0, 0), zlib.compress(frames[1, :4, :].tobytes()), filter_mask=0)
    with NexusReader(path) as reader:
        with pytest.raises(ValueError, match="unexpected stored size"):
            reader.compute_full_scan_roi((2, 12, 0, 16))


@pytest.mark.parametrize("name", ["005572.nxs", "005594.nxs"])
def test_example_files_parallel_roi_faster_than_h5py(name):
    with NexusReader(EXAMPLE_DIR / name) as reader:
        roi = reader.get_detector_roi()
        assert roi is not None
        t0 = time.perf_counter()
        slow = reader.compute_full_scan_roi(roi, parallel=False)
        t_slow = time.perf_counter() - t0
        t1 = time.perf_counter()
        fast = reader.compute_full_scan_roi(roi, parallel=True)
        t_fast = time.perf_counter() - t1
        np.testing.assert_array_equal(fast, slow)
        assert t_fast < t_slow


@pytest.mark.parametrize("name,run_num,n_frames", [("005572.nxs", 5572, 302), ("005594.nxs", 5594, 488)])
def test_parse_scan_metadata_motors_and_roi_exact(name, run_num, n_frames):
    with NexusReader(EXAMPLE_DIR / name) as reader:
        assert reader.file.mode == "r"
        scan = reader.parse_scan()
        expected_roi = reader.compute_full_scan_roi(scan.detector_roi, parallel=False)
        assert scan.metadata.run_number == run_num
        assert scan.detector_shape == (n_frames, 1024, 1024)
        assert "th" in scan.motors and scan.motors["th"].units
        assert "ROI_Counts" in scan.variables
        np.testing.assert_array_equal(scan.variables["ROI_Counts"].data, expected_roi)


def test_sync_load_avoids_duplicate_activation_and_redraws(qtbot):
    from pySAview.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    p1 = (EXAMPLE_DIR / "005572.nxs").resolve()
    p2 = (EXAMPLE_DIR / "005594.nxs").resolve()

    counts = {"set_active": 0, "plot_actual": 0}
    orig_sa = window.set_active_file
    orig_pl = window.view_charts._update_plot

    def wrap_sa(*args, **kwargs):
        counts["set_active"] += 1
        return orig_sa(*args, **kwargs)

    def wrap_pl(*args, **kwargs):
        if not getattr(window.view_charts, "_defer_plot", False):
            counts["plot_actual"] += 1
        return orig_pl(*args, **kwargs)

    window.set_active_file = wrap_sa
    window.view_charts._update_plot = wrap_pl

    assert window.load_file(p1) is True
    assert counts == {"set_active": 1, "plot_actual": 1}
    assert len(window.view_charts._current_curves) == 2

    counts["set_active"] = 0
    counts["plot_actual"] = 0
    window.load_files([p1, p2], clear_existing=True)
    assert counts == {"set_active": 1, "plot_actual": 1}
    assert len(window.loaded_scans) == 2
    assert len(window.view_charts._current_curves) == 4


def test_hidden_tabs_defer_heavy_reads_until_shown(qtbot):
    from pySAview.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    p1 = (EXAMPLE_DIR / "005572.nxs").resolve()
    assert window.load_file(p1) is True

    # Scan Charts is active; Detector Image and File Hierarchy are hidden.
    assert window.tabs.currentWidget() is window.view_charts
    shared_reader = window.loaded_readers[str(p1)]
    assert shared_reader.file.mode == "r"

    # File Hierarchy reuses the read-only reader and defers /entry0 child expansion while hidden.
    hier = window.view_hierarchy
    assert hier.reader is shared_reader
    assert hier.tree.topLevelItemCount() >= 1
    assert hier._pending_root_expand is True

    # Detector Image defers frame 0 HDF5 read while hidden, even across theme toggles.
    det = window.view_detector
    assert det._total_frames == 302
    assert det._frame_dirty is True
    assert det._frame_data is None

    read_calls = []
    orig_read = shared_reader.read_detector_frame

    def spy_read(idx=0):
        read_calls.append(idx)
        return orig_read(idx)

    shared_reader.read_detector_frame = spy_read

    window.act_dark_mode.setChecked(True)
    window.act_dark_mode.setChecked(False)
    assert read_calls == []
    assert det._frame_dirty is True

    # Showing Detector Image tab loads frame 0 on demand; log toggle reuses cached frame without HDF5 I/O.
    window.show()
    window.tabs.setCurrentWidget(det)
    qtbot.waitUntil(lambda: not det._frame_dirty)
    assert read_calls == [0]
    expected_frame0 = np.asarray(orig_read(0), dtype=float).T
    np.testing.assert_array_equal(det._current_frame_data, expected_frame0)

    read_calls.clear()
    det.chk_log_img.click()
    det.chk_log_img.click()
    assert read_calls == []

    # Showing File Hierarchy tab expands /entry0 children transparently.
    window.tabs.setCurrentWidget(hier)
    qtbot.waitUntil(lambda: not hier._pending_root_expand)
    top = hier.tree.topLevelItem(0)
    assert top is not None and top.isExpanded()
    child_names = {top.child(i).text(0) for i in range(top.childCount())}
    assert {"data_scan", "SUPERADAM"}.issubset(child_names)

    # Closing the active file detaches the shared reader cleanly.
    window.close_active_file()
    assert hier.reader is None


def test_async_load_warms_scipy_stats_off_gui_thread():
    """First async load warms scipy.special in FileLoadThread without importing scipy.stats."""
    example_path = str((EXAMPLE_DIR / "005572.nxs").resolve())
    code = f"""
import sys
from pathlib import Path
from PyQt6.QtCore import QEventLoop
from PyQt6.QtWidgets import QApplication
from pySAview.ui.main_window import MainWindow

assert "scipy.stats" not in sys.modules
assert "scipy.special" not in sys.modules
app = QApplication([])
win = MainWindow()
warm_before_gui_finish = []
stats_loaded = []
orig_finish = win._on_load_finished
def spy_finish():
    warm_before_gui_finish.append("scipy.special" in sys.modules)
    stats_loaded.append("scipy.stats" in sys.modules)
    orig_finish()
win._on_load_finished = spy_finish

p = Path({example_path!r}).resolve()
win.start_load_files([p], clear_existing=True)
loop = QEventLoop()
next(iter(win._load_jobs.values())).finished.connect(loop.quit)
loop.exec()
win.close()
print("WARM=" + str(warm_before_gui_finish) + " STATS=" + str(stats_loaded))
"""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert "WARM=[True] STATS=[False]" in out.stdout
