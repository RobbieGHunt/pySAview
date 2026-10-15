"""Background scan parsing; HDF5 handles never leave their owning worker."""
from pathlib import Path
from threading import Event
from PyQt6.QtCore import QThread, pyqtSignal
from sa_nxs_explorer.core.nexus_reader import NexusReader


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
        for index, path in enumerate(self.paths):
            if self.cancelled.is_set():
                break
            self.progress.emit(self.token, index, Path(path).name)
            reader = None
            try:
                reader = NexusReader(path, cancelled=self.cancelled.is_set)
                scan = reader.parse_scan()
                if not self.cancelled.is_set():
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
