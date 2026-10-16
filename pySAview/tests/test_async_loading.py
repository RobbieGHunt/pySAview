from pathlib import Path
from threading import Event, get_ident
import pytest
from PyQt6.QtCore import QTimer
from pySAview.tests.test_processing_ui import scan
from pySAview.ui.main_window import MainWindow


def setup_fake(qtbot, monkeypatch, gates=None, errors=()):
    gates = gates or {}
    parsed_threads, closed = [], []
    class Reader:
        def __init__(self, path, **kwargs): self.path = Path(path)
        def parse_scan(self):
            parsed_threads.append(get_ident())
            if self.path.name in gates:
                entered, release = gates[self.path.name]
                entered.set()
                release.wait(5)
            if self.path.name in errors:
                raise ValueError("broken file")
            return scan(str(self.path))
        def close(self): closed.append((str(self.path), get_ident()))
    monkeypatch.setattr("pySAview.ui.file_loading.NexusReader", Reader)
    monkeypatch.setattr("pySAview.ui.main_window.NexusReader", Reader)
    w = MainWindow()
    qtbot.addWidget(w)
    monkeypatch.setattr(w, "set_active_file", lambda path: setattr(w, "active_file_path", Path(path)))
    return w, parsed_threads, closed


def wait_done(qtbot, window):
    qtbot.waitUntil(lambda: not window._load_jobs, timeout=10000)


def test_background_parse_keeps_event_loop_alive_and_renders_once(qtbot, monkeypatch, tmp_path):
    entered, release = Event(), Event()
    w, threads, closed = setup_fake(qtbot, monkeypatch, {"1.nxs": (entered, release)})
    ticks = []
    timer = QTimer(w)
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    renders = []
    original = w.view_charts._update_plot
    def update():
        if not w.view_charts._defer_plot:
            renders.append(1)
        original()
    monkeypatch.setattr(w.view_charts, "_update_plot", update)
    paths = [tmp_path / "1.nxs", tmp_path / "2.nxs"]
    w.start_load_files(paths + paths[:1])
    qtbot.waitUntil(entered.is_set)
    qtbot.waitUntil(lambda: len(ticks) >= 3)
    assert not w.loaded_scans
    release.set()
    wait_done(qtbot, w)
    assert len(w.loaded_scans) == 2
    assert list(w.loaded_scans) == [str(p) for p in paths]
    assert all(thread != get_ident() for thread in threads)
    assert len([thread for _, thread in closed if thread != get_ident()]) == 2
    assert len(renders) == 1
    assert w.active_file_path == paths[0]


def test_cancel_retains_completed_files(qtbot, monkeypatch, tmp_path):
    entered, release = Event(), Event()
    w, _, _ = setup_fake(qtbot, monkeypatch, {"2.nxs": (entered, release)})
    w.start_load_files([tmp_path / "1.nxs", tmp_path / "2.nxs"])
    qtbot.waitUntil(lambda: entered.is_set() and len(w.loaded_scans) == 1)
    w.cancel_loading()
    release.set()
    wait_done(qtbot, w)
    assert list(w.loaded_scans) == [str(tmp_path / "1.nxs")]
    assert "Cancelled" in w.status_bar.currentMessage()
    assert not w.view_charts._defer_plot


def test_new_batch_discards_late_old_results(qtbot, monkeypatch, tmp_path):
    entered, release = Event(), Event()
    w, _, _ = setup_fake(qtbot, monkeypatch, {"old.nxs": (entered, release)})
    w.start_load_files([tmp_path / "old.nxs"])
    qtbot.waitUntil(entered.is_set)
    w.start_load_files([tmp_path / "new.nxs"])
    release.set()
    wait_done(qtbot, w)
    assert list(w.loaded_scans) == [str(tmp_path / "new.nxs")]


def test_close_cancels_without_destroying_live_worker(qtbot, monkeypatch, tmp_path):
    entered, release = Event(), Event()
    w, _, _ = setup_fake(qtbot, monkeypatch, {"slow.nxs": (entered, release)})
    w.start_load_files([tmp_path / "slow.nxs"])
    qtbot.waitUntil(entered.is_set)
    w.close()
    assert w._closing
    assert w._load_jobs
    release.set()
    wait_done(qtbot, w)
    assert not w.loaded_scans


def test_bad_file_does_not_block_good_file(qtbot, monkeypatch, tmp_path):
    w, _, _ = setup_fake(qtbot, monkeypatch, errors=("bad.nxs",))
    w.start_load_files([tmp_path / "bad.nxs", tmp_path / "good.nxs"])
    wait_done(qtbot, w)
    assert len(w.loaded_scans) == 1
    assert "1 failed" in w.status_bar.currentMessage()


def test_real_files_async_match_read_only_reader(qtbot):
    from pySAview.core.nexus_reader import NexusReader
    import numpy as np
    w = MainWindow()
    qtbot.addWidget(w)
    example = Path(__file__).resolve().parent.parent / "example_data/005572.nxs"
    w.start_load_files([example], clear_existing=True)
    qtbot.waitUntil(lambda: not w._load_jobs, timeout=60000)
    assert w.active_file_path == example
    with_reader = NexusReader(example)
    try:
        expected = with_reader.parse_scan()
        actual = w.loaded_scans[str(example)]
        np.testing.assert_array_equal(actual.variables["ROI_Counts"].data, expected.variables["ROI_Counts"].data)
        assert w.loaded_readers[str(example)].file.mode == "r"
    finally:
        with_reader.close()


def test_failed_replace_preserves_existing_workspace(qtbot, monkeypatch, tmp_path):
    w, _, _ = setup_fake(qtbot, monkeypatch, errors=("bad.nxs",))
    original = scan(str(tmp_path / "original.nxs"))
    w.loaded_scans[original.file_path] = original
    w.start_load_files([tmp_path / "bad.nxs"], clear_existing=True)
    wait_done(qtbot, w)
    assert w.loaded_scans[original.file_path] is original


def test_roi_read_cancellation_between_chunks(tmp_path):
    import h5py
    import numpy as np
    from pySAview.core.nexus_reader import NexusReader
    target = tmp_path / "chunked.nxs"
    with h5py.File(target, "w") as f:
        f.create_dataset("entry0/data_scan/detector_data/data", data=np.ones((40, 4, 3)))
    checks = []
    def cancel():
        checks.append(1)
        return len(checks) == 2
    reader = NexusReader(target, cancelled=cancel)
    try:
        with pytest.raises(InterruptedError):
            reader.compute_full_scan_roi((0, 4, 0, 3))
    finally:
        reader.close()


def test_many_files_registered_with_one_final_plot(qtbot, monkeypatch, tmp_path):
    w, _, _ = setup_fake(qtbot, monkeypatch)
    renders = []
    original = w.view_charts._update_plot
    def update():
        if not w.view_charts._defer_plot:
            renders.append(1)
        original()
    monkeypatch.setattr(w.view_charts, "_update_plot", update)
    paths = [tmp_path / f"{i}.nxs" for i in range(25)]
    w.start_load_files(paths)
    wait_done(qtbot, w)
    assert list(w.loaded_scans) == [str(p) for p in paths]
    assert w.session_files_view.list_files.count() == 25
    assert len(w.view_charts._current_curves) == 25
    assert len(renders) == 1
