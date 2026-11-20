"""Background scan parsing and async scipy.special warm-up; HDF5 handles never leave their owning worker."""
from __future__ import annotations

import atexit
import sys
from pathlib import Path
from threading import Event, Lock, Thread

from PyQt6.QtCore import QThread, pyqtSignal

from pySAview.core.nexus_reader import NexusReader

_SCIPY_WARM_LOCK = Lock()
_SCIPY_WARM_THREAD: Thread | None = None
_SCIPY_ATEXIT_REGISTERED = False


def _import_scipy_special() -> None:
    try:
        import scipy.special  # noqa: F401
    except Exception:
        pass


def _join_scipy_warm_at_exit() -> None:
    thread = _SCIPY_WARM_THREAD
    if thread is not None and thread.is_alive():
        thread.join(timeout=5.0)


def warm_scipy_special_async() -> Thread | None:
    """Import scipy.special once on a background thread so the first exact-interval plot never stalls the GUI."""
    global _SCIPY_WARM_THREAD, _SCIPY_ATEXIT_REGISTERED
    if "scipy.special" in sys.modules:
        return None
    with _SCIPY_WARM_LOCK:
        if "scipy.special" in sys.modules:
            return None
        if _SCIPY_WARM_THREAD is not None and _SCIPY_WARM_THREAD.is_alive():
            return _SCIPY_WARM_THREAD
        if not _SCIPY_ATEXIT_REGISTERED:
            atexit.register(_join_scipy_warm_at_exit)
            _SCIPY_ATEXIT_REGISTERED = True
        thread = Thread(target=_import_scipy_special, name="scipy-special-warmup", daemon=True)
        _SCIPY_WARM_THREAD = thread
        thread.start()
        return thread


# Backward-compatible alias for callers written before the scipy.special switch.
warm_scipy_stats_async = warm_scipy_special_async


class FileLoadThread(QThread):
    parsed = pyqtSignal(int, object)
    progress = pyqtSignal(int, int, str)

    def __init__(self, token, paths, parent=None):
        super().__init__(parent)
        self.token = token
        self.paths = paths
        self.cancelled = Event()
        self.errors = []

    def run(self):
        any_parsed = False
        for index, path in enumerate(self.paths):
            if self.cancelled.is_set():
                break
            self.progress.emit(self.token, index, Path(path).name)
            reader = None
            try:
                reader = NexusReader(path, cancelled=self.cancelled.is_set)
                scan = reader.parse_scan()
                if not self.cancelled.is_set():
                    any_parsed = True
                    self.parsed.emit(self.token, scan)
            except InterruptedError:
                if not self.cancelled.is_set():
                    self.errors.append((str(path), "Read interrupted"))
            except Exception as error:
                self.errors.append((str(path), str(error)))
            finally:
                if reader is not None:
                    reader.close()
            self.progress.emit(self.token, index + 1, "")
        # Warm scipy.special off the GUI thread *after* parallel ROI inflation finishes
        # to avoid GIL contention with zlib workers while keeping the UI responsive.
        if any_parsed and not self.cancelled.is_set():
            warm_thread = warm_scipy_special_async()
            if warm_thread is not None:
                while warm_thread.is_alive() and not self.cancelled.is_set():
                    warm_thread.join(0.02)
