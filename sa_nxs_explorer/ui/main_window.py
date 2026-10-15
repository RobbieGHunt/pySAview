"""Main Window for SuperADAM NeXus Explorer with Lucide icons and sleek modern tabs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSlot
from PyQt6.QtGui import QAction, QDragEnterEvent, QDropEvent, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QProgressDialog,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QWidget,
)

from sa_nxs_explorer.core.nexus_reader import NexusReader
from sa_nxs_explorer.core.models import ScanData, ScanMetadata
from sa_nxs_explorer.ui.file_loading import FileLoadThread
from sa_nxs_explorer.ui import icons
from sa_nxs_explorer.ui.preferences import restore_preferences, save_preferences
from sa_nxs_explorer.ui.detector_view import DetectorView
from sa_nxs_explorer.ui.file_structure_view import FileStructureView
from sa_nxs_explorer.ui.metadata_view import MetadataView
from sa_nxs_explorer.ui.scan_plot_view import ScanPlotView
from sa_nxs_explorer.ui.session_files_view import SessionFilesView
from sa_nxs_explorer.ui.theme import DARK_THEME, LIGHT_THEME


class MainWindow(QMainWindow):
    """Primary application window housing all NeXus explorer tabs."""

    def __init__(self, parent: QWidget | None = None, *, settings=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SuperADAM NeXus Explorer")
        self.resize(1300, 860)
        self.setAcceptDrops(True)

        self._load_jobs = {}
        self._load_token = 0
        self._load_dialog = None
        self._closing = False
        self.loaded_readers: dict[str, NexusReader] = {}
        self.loaded_scans: dict[str, Any] = {}
        self.active_file_path: Path | None = None
        self.is_dark_mode: bool = False

        self.settings = settings if settings is not None else QSettings("ILL SuperADAM", "pySAview")
        self.last_load_directory = str(Path.home())
        self._init_ui()
        self._apply_app_theme()
        restore_preferences(self)

    @property
    def current_file_path(self) -> Path | None:
        return self.active_file_path

    @current_file_path.setter
    def current_file_path(self, val: Path | None) -> None:
        self.active_file_path = val

    def _init_ui(self) -> None:
        # Central Horizontal Splitter: Session Files Sidebar on Left, Tabbed Views on Right
        self.session_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.session_splitter)

        self.session_files_view = SessionFilesView()
        self.session_splitter.addWidget(self.session_files_view)

        self.tabs = QTabWidget()
        self.session_splitter.addWidget(self.tabs)
        self.session_splitter.setSizes([280, 1020])

        self.view_hierarchy = FileStructureView()
        self.view_charts = ScanPlotView()
        self.view_detector = DetectorView()
        self.view_metadata = MetadataView()

        self.tabs.addTab(self.view_hierarchy, "File Hierarchy")
        self.tabs.addTab(self.view_charts, "Scan Charts")
        self.tabs.addTab(self.view_detector, "2D Detector")
        self.tabs.addTab(self.view_metadata, "Metadata & Motors")

        # Connect session files view signals
        self.session_files_view.active_file_changed.connect(lambda k: self.set_active_file(Path(k)))
        self.session_files_view.overlay_selection_changed.connect(self.view_charts.set_overlay_keys)
        self.session_files_view.open_requested.connect(self._on_action_open)
        self.session_files_view.add_requested.connect(self._on_action_add)
        self.session_files_view.close_requested.connect(self.close_active_file)

        # Connect interactive ROI signal
        self.view_detector.roi_applied.connect(self._on_detector_roi_applied)

        # Menu Bar & Actions
        self._create_actions_and_menus()

        # Tool Bar
        self._create_toolbar()

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Open one or more NeXus (.nxs) files to begin.")

    def _create_actions_and_menus(self) -> None:
        menubar = self.menuBar()

        # File Menu
        menu_file = menubar.addMenu("&File")

        self.act_open = QAction("&Open NeXus File(s)...", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self._on_action_open)
        menu_file.addAction(self.act_open)

        self.act_add = QAction("&Add File(s) to Session...", self)
        self.act_add.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.act_add.triggered.connect(self._on_action_add)
        menu_file.addAction(self.act_add)

        self.act_open_example = QAction("Open &Example Scans (5572 & 5594)", self)
        self.act_open_example.setShortcut(QKeySequence("Ctrl+E"))
        self.act_open_example.triggered.connect(self._on_action_open_example)
        menu_file.addAction(self.act_open_example)

        self.act_close_file = QAction("&Close Active File", self)
        self.act_close_file.setShortcut(QKeySequence.StandardKey.Close)
        self.act_close_file.triggered.connect(self.close_active_file)
        menu_file.addAction(self.act_close_file)

        self.act_reload = QAction("&Reload Active File", self)
        self.act_reload.setShortcut(QKeySequence.StandardKey.Refresh)
        self.act_reload.triggered.connect(self._on_action_reload)
        menu_file.addAction(self.act_reload)

        menu_file.addSeparator()

        self.act_exit = QAction("E&xit", self)
        self.act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        self.act_exit.triggered.connect(self.close)
        menu_file.addAction(self.act_exit)

        # View Menu
        menu_view = menubar.addMenu("&View")

        self.act_toggle_sidebar = QAction("&Toggle Sidebar", self)
        self.act_toggle_sidebar.setShortcut(QKeySequence("Ctrl+B"))
        self.act_toggle_sidebar.setToolTip("Minimize or expand the session files sidebar to free up horizontal space (Ctrl+B)")
        self.act_toggle_sidebar.triggered.connect(self.session_files_view.toggle_minimized)
        menu_view.addAction(self.act_toggle_sidebar)

        menu_view.addSeparator()

        self.act_dark_mode = QAction("&Dark Mode", self)
        self.act_dark_mode.setCheckable(True)
        self.act_dark_mode.setChecked(self.is_dark_mode)
        self.act_dark_mode.setShortcut(QKeySequence("Ctrl+D"))
        self.act_dark_mode.toggled.connect(self._toggle_dark_mode)
        menu_view.addAction(self.act_dark_mode)

        menu_view.addSeparator()

        act_tab0 = QAction("File Hierarchy Tab", self)
        act_tab0.setShortcut(QKeySequence("Ctrl+1"))
        act_tab0.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        menu_view.addAction(act_tab0)

        act_tab1 = QAction("Scan Charts Tab", self)
        act_tab1.setShortcut(QKeySequence("Ctrl+2"))
        act_tab1.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        menu_view.addAction(act_tab1)

        act_tab2 = QAction("2D Detector Tab", self)
        act_tab2.setShortcut(QKeySequence("Ctrl+3"))
        act_tab2.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        menu_view.addAction(act_tab2)

        act_tab3 = QAction("Metadata Tab", self)
        act_tab3.setShortcut(QKeySequence("Ctrl+4"))
        act_tab3.triggered.connect(lambda: self.tabs.setCurrentIndex(3))
        menu_view.addAction(act_tab3)

        # Help Menu
        menu_help = menubar.addMenu("&Help")
        act_about = QAction("&About SuperADAM NeXus Explorer", self)
        act_about.setIcon(icons.get_icon("info", is_dark=self.is_dark_mode))
        act_about.triggered.connect(self._show_about)
        menu_help.addAction(act_about)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction(self.act_toggle_sidebar)
        toolbar.addSeparator()

        toolbar.addAction(self.act_open)
        toolbar.addAction(self.act_add)
        toolbar.addAction(self.act_open_example)
        toolbar.addAction(self.act_close_file)
        toolbar.addAction(self.act_reload)
        toolbar.addSeparator()

        lbl_active = QLabel(" Active File: ")
        toolbar.addWidget(lbl_active)

        self.combo_active_file = QComboBox()
        self.combo_active_file.setMinimumWidth(260)
        self.combo_active_file.setMinimumContentsLength(24)
        self.combo_active_file.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_active_file.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_active_file.currentIndexChanged.connect(self._on_active_file_combo_changed)
        toolbar.addWidget(self.combo_active_file)

        toolbar.addSeparator()
        toolbar.addAction(self.act_dark_mode)

    def _apply_app_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            theme_qss = DARK_THEME if self.is_dark_mode else LIGHT_THEME
            app.setStyleSheet(theme_qss)

        # Tab vector icons
        is_d = self.is_dark_mode
        self.tabs.setTabIcon(0, icons.get_icon("folder_tree", is_dark=is_d))
        self.tabs.setTabIcon(1, icons.get_icon("chart", is_dark=is_d))
        self.tabs.setTabIcon(2, icons.get_icon("scan", is_dark=is_d))
        self.tabs.setTabIcon(3, icons.get_icon("sliders", is_dark=is_d))

        # Action vector icons
        self.act_toggle_sidebar.setIcon(icons.get_icon("panel_left", is_dark=is_d))
        self.act_open.setIcon(icons.get_icon("folder_open", is_dark=is_d))
        self.act_add.setIcon(icons.get_icon("file_plus", is_dark=is_d))
        self.act_open_example.setIcon(icons.get_icon("sparkles", is_dark=is_d))
        self.act_close_file.setIcon(icons.get_icon("x_circle", is_dark=is_d))
        self.act_reload.setIcon(icons.get_icon("refresh_cw", is_dark=is_d))
        dark_icon_name = "sun" if is_d else "moon"
        self.act_dark_mode.setIcon(icons.get_icon(dark_icon_name, is_dark=is_d))

        self.session_files_view.set_theme(is_d)
        self.view_hierarchy.set_theme(is_d)
        self.view_charts.set_theme(is_d)
        self.view_detector.set_theme(is_d)

    def _toggle_dark_mode(self, checked: bool) -> None:
        self.is_dark_mode = checked
        self._apply_app_theme()

    def _on_action_open(self) -> None:
        start_dir = self.last_load_directory if Path(self.last_load_directory).is_dir() else str(Path.home())
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open NeXus Scan File(s)",
            start_dir,
            "NeXus / HDF5 Files (*.nxs *.h5 *.hdf5);;All Files (*.*)",
        )
        if file_paths:
            self.last_load_directory = str(Path(file_paths[0]).resolve().parent)
            self.start_load_files([Path(p) for p in file_paths], clear_existing=True)

    def _on_action_add(self) -> None:
        start_dir = self.last_load_directory if Path(self.last_load_directory).is_dir() else str(Path.home())
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add NeXus Scan File(s) to Session",
            start_dir,
            "NeXus / HDF5 Files (*.nxs *.h5 *.hdf5);;All Files (*.*)",
        )
        if file_paths:
            self.last_load_directory = str(Path(file_paths[0]).resolve().parent)
            self.start_load_files([Path(p) for p in file_paths], clear_existing=False)

    def _on_action_open_example(self) -> None:
        # Locate example_data/005572.nxs and 005594.nxs
        base_dir = Path(__file__).resolve().parent.parent
        example_dir = base_dir / "example_data"
        if not example_dir.exists():
            example_dir = Path("sa_nxs_explorer/example_data")

        files_to_load = []
        for name in ("005572.nxs", "005594.nxs"):
            p = example_dir / name
            if p.exists():
                files_to_load.append(p)

        if files_to_load:
            self.start_load_files(files_to_load, clear_existing=False)
        else:
            QMessageBox.warning(self, "Example Not Found", f"Could not find example scans in {example_dir}")

    def _on_action_reload(self) -> None:
        if self.active_file_path and self.active_file_path.exists():
            self.start_load_files([self.active_file_path])

    def cancel_loading(self, token=None) -> None:
        job = self._load_jobs.get(self._load_token if token is None else token)
        if job is not None:
            job.cancelled.set()
            self.status_bar.showMessage("Cancelling load; completed files will be retained.")

    def start_load_files(self, paths, clear_existing=False) -> None:
        """Start a cancellable batch; all view updates remain on the GUI thread."""
        if self._closing or not paths:
            return
        self.cancel_loading()
        if self._load_dialog is not None:
            self._load_dialog.close()
            self._load_dialog.deleteLater()
        self._load_token += 1
        token = self._load_token
        # Deduplicate while preserving the user's load order.
        paths = list(dict.fromkeys(Path(p).resolve() for p in paths))
        job = FileLoadThread(token, paths, self)
        job.clear_existing = clear_existing
        job.accepted = []
        job.initial_active = self.active_file_path
        self._load_jobs[token] = job
        self.view_charts._defer_plot = True
        dialog = self._load_dialog = QProgressDialog("Preparing files…", "Cancel", 0, len(paths), self)
        dialog.setWindowTitle("Loading NeXus scans")
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.canceled.connect(lambda: self.cancel_loading(token))
        dialog.show()
        job.parsed.connect(self._on_scan_parsed)
        job.progress.connect(self._on_load_progress)
        job.finished.connect(self._on_load_finished)
        job.start()

    @pyqtSlot(int, int, str)
    def _on_load_progress(self, token, completed, filename):
        if token != self._load_token or self._closing or self._load_dialog is None:
            return
        job = self._load_jobs.get(token)
        if job is None or job.cancelled.is_set():
            return
        self._load_dialog.setValue(completed)
        self._load_dialog.setLabelText(f"{completed}/{self._load_dialog.maximum()} files processed" + (f" — {filename}" if filename else ""))

    def _clear_workspace(self):
        self.view_detector._play_timer.stop()
        self.view_hierarchy.close()
        for reader in self.loaded_readers.values():
            reader.close()
        self.loaded_readers.clear()
        self.loaded_scans.clear()
        self.active_file_path = None
        self.session_files_view.clear_all()
        self.view_charts.clear_scans()
        self.view_detector.set_data(None, ScanData("", ScanMetadata(), {}))
        self.view_detector._current_frame_data = None
        self.view_detector.img_item.clear()
        self.view_hierarchy.tree.clear()
        self.view_metadata.set_data(ScanData("", ScanMetadata(), {}))

    @pyqtSlot(int, object)
    def _on_scan_parsed(self, token, scan):
        job = self._load_jobs.get(token)
        if self._closing or token != self._load_token or job is None or job.cancelled.is_set():
            return
        reader = None
        try:
            # A separate read-only handle is owned exclusively by the GUI.
            reader = NexusReader(scan.file_path)
            if job.clear_existing and not job.accepted:
                self._clear_workspace()
            key = scan.file_path
            previous = self.loaded_readers.get(key)
            if previous is not None:
                previous.close()
            self.loaded_readers[key] = reader
            self.loaded_scans[key] = scan
            self.view_charts.add_scan(scan, make_active=not job.accepted)
            self.session_files_view.blockSignals(True)
            try:
                self.session_files_view.add_file(scan, make_active=False)
            finally:
                self.session_files_view.blockSignals(False)
            job.accepted.append(Path(key))
            self.last_load_directory = str(Path(key).parent)
            if previous is not None and self.active_file_path == Path(key):
                self.set_active_file(Path(key))
        except Exception as error:
            if reader is not None and reader not in self.loaded_readers.values():
                reader.close()
            job.errors.append((scan.file_path, str(error)))

    @pyqtSlot()
    def _on_load_finished(self):
        job = self.sender()
        self._load_jobs.pop(job.token, None)
        if self._closing:
            job.deleteLater()
            if not self._load_jobs:
                QTimer.singleShot(0, self.close)
            return
        if job.token == self._load_token:
            if self._load_dialog is not None:
                self._load_dialog.close()
                self._load_dialog.deleteLater()
                self._load_dialog = None
            self.view_charts._defer_plot = False
            self._update_file_combo()
            job.accepted = [path for path in job.accepted if str(path) in self.loaded_scans]
            if job.accepted:
                if self.active_file_path is None or self.active_file_path == job.initial_active:
                    self.set_active_file(job.accepted[0])
                elif self.active_file_path in job.accepted:
                    self.set_active_file(self.active_file_path)
                self.tabs.setCurrentIndex(1)
            self.view_charts._refresh_axis_combos()
            self.view_charts._update_plot()
            state = "Cancelled" if job.cancelled.is_set() else "Finished"
            self.status_bar.showMessage(f"{state}: {len(job.accepted)} files loaded, {len(job.errors)} failed.")
            if job.errors:
                message = QMessageBox(self)
                message.setWindowTitle("Some scans could not be loaded")
                message.setIcon(QMessageBox.Icon.Warning)
                message.setText(f"{len(job.errors)} file(s) failed. Successfully loaded scans have been retained.")
                message.setDetailedText("\n\n".join(f"{path}: {error}" for path, error in job.errors))
                message.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
                message.show()
        job.deleteLater()

    def load_file(self, path: Path | str, make_active: bool = True) -> bool:
        """Load and parse a single NeXus file into the session."""
        target = Path(path).resolve()
        if not target.exists():
            QMessageBox.critical(self, "File Error", f"File does not exist:\n{target}")
            return False

        key = str(target)
        try:
            if key in self.loaded_readers:
                try:
                    self.loaded_readers[key].close()
                except Exception:
                    pass
            reader = NexusReader(target)
            self.loaded_readers[key] = reader
            scan_data = reader.parse_scan()
            self.loaded_scans[key] = scan_data

            # Add to multi-dataset scan plot view
            self.view_charts.add_scan(scan_data, make_checked=True, make_active=make_active)

            # Add to global session files sidebar
            self.session_files_view.add_file(scan_data, is_checked=True, make_active=make_active)

            # Update toolbar dropdown
            self._update_file_combo()

            if make_active or self.active_file_path is None:
                self.set_active_file(target)

            # Focus charts tab on initial open
            if self.tabs.currentIndex() == 0 and scan_data.variables:
                self.tabs.setCurrentIndex(1)

            self.last_load_directory = str(target.parent)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Failed to load NeXus file:\n{target.name}\n\n{e}")
            return False

    def load_files(self, paths: list[Path | str], clear_existing: bool = False) -> None:
        """Load multiple NeXus files into the session."""
        if not paths:
            return

        if clear_existing:
            for r in self.loaded_readers.values():
                try:
                    r.close()
                except Exception:
                    pass
            self.loaded_readers.clear()
            self.loaded_scans.clear()
            self.active_file_path = None
            self.session_files_view.clear_all()
            self.view_charts.clear_scans()

        first_loaded: Path | None = None
        for p in paths:
            target = Path(p).resolve()
            if self.load_file(target, make_active=(first_loaded is None)):
                if first_loaded is None:
                    first_loaded = target

        if first_loaded is not None:
            self.set_active_file(first_loaded)

    def set_active_file(self, path: Path | str) -> None:
        """Set the active file for hierarchy, 2D detector, and metadata inspection."""
        target = Path(path).resolve()
        key = str(target)
        if key not in self.loaded_scans or key not in self.loaded_readers:
            return

        self.active_file_path = target
        reader = self.loaded_readers[key]
        scan_data = self.loaded_scans[key]

        # Update views
        self.view_hierarchy.set_file(target)
        self.view_detector.set_data(reader, scan_data)
        self.view_metadata.set_data(scan_data)

        # Update session files sidebar active selection
        self.session_files_view.set_active_key(key, emit_signal=False)

        # Update combo selection without triggering recursive signal
        self.combo_active_file.blockSignals(True)
        idx = self.combo_active_file.findData(key)
        if idx >= 0:
            self.combo_active_file.setCurrentIndex(idx)
        self.combo_active_file.blockSignals(False)

        # Update window title and status bar
        size_mb = (target.stat().st_size / (1024 * 1024)) if target.exists() else 0.0
        n_steps = len(next(iter(scan_data.variables.values())).data) if scan_data.variables else 0
        run_str = f"#{scan_data.metadata.run_number}" if scan_data.metadata.run_number else ""
        n_total = len(self.loaded_scans)
        self.setWindowTitle(
            f"SuperADAM NeXus Explorer — {target.name} ({run_str}) [{n_total} file{'s' if n_total != 1 else ''} loaded]"
        )
        self.status_bar.showMessage(
            f"Active: {target.name} ({size_mb:.2f} MB)  |  Run: {run_str}  |  Steps: {n_steps}  |  Total Loaded: {n_total}"
        )

    def close_active_file(self) -> None:
        """Close the currently active file from the explorer session."""
        if self.active_file_path is None:
            return

        self.cancel_loading()
        key = str(self.active_file_path)
        if key in self.loaded_readers:
            try:
                self.loaded_readers[key].close()
            except Exception:
                pass
            del self.loaded_readers[key]

        if key in self.loaded_scans:
            del self.loaded_scans[key]

        self.view_charts.remove_scan(key)
        self.session_files_view.remove_file(key)
        self._update_file_combo()

        if self.loaded_scans:
            next_key = next(iter(self.loaded_scans.keys()))
            self.set_active_file(Path(next_key))
        else:
            self.active_file_path = None
            self.view_hierarchy.tree.clear()
            self.setWindowTitle("SuperADAM NeXus Explorer")
            self.status_bar.showMessage("Ready. Open one or more NeXus (.nxs) files to begin.")

    def _update_file_combo(self) -> None:
        self.combo_active_file.blockSignals(True)
        self.combo_active_file.clear()
        for k, scan in self.loaded_scans.items():
            name = Path(k).name
            run_str = f"#{scan.metadata.run_number}" if scan.metadata.run_number else ""
            sub_str = f" - {scan.metadata.subtitle}" if scan.metadata.subtitle else ""
            lbl = f"{name} ({run_str}{sub_str})" if run_str else f"{name}{sub_str}"
            self.combo_active_file.addItem(lbl, k)
            self.combo_active_file.setItemData(self.combo_active_file.count()-1, lbl, Qt.ItemDataRole.ToolTipRole)

        if self.active_file_path:
            idx = self.combo_active_file.findData(str(self.active_file_path))
            if idx >= 0:
                self.combo_active_file.setCurrentIndex(idx)
        self.combo_active_file.blockSignals(False)

    def _on_active_file_combo_changed(self, index: int) -> None:
        key = self.combo_active_file.itemData(index)
        if key:
            self.set_active_file(Path(key))

    def _on_detector_roi_applied(self, roi_counts: Any, roi_bounds: tuple[float, float, float, float]) -> None:
        """Update scan plot view when a new ROI is applied from detector view."""
        cur_path = str(self.active_file_path) if self.active_file_path else None
        self.view_charts.update_roi_counts(roi_counts, roi_bounds, file_path=cur_path)
        x0, x1, y0, y1 = roi_bounds
        self.status_bar.showMessage(
            f"User ROI Applied to Scan Charts: X [{x0:.0f}..{x1:.0f}], Y [{y0:.0f}..{y1:.0f}]  |  Total ROI Counts: {np.sum(roi_counts):,.0f}"
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        valid_paths = []
        for u in urls:
            path = Path(u.toLocalFile())
            if path.suffix.lower() in (".nxs", ".h5", ".hdf5"):
                valid_paths.append(path)
        if valid_paths:
            self.start_load_files(valid_paths, clear_existing=False)

    def closeEvent(self, event) -> None:
        if self._load_jobs:
            self._closing = True
            for job in self._load_jobs.values():
                job.cancelled.set()
            if self._load_dialog is not None:
                self._load_dialog.close()
            self.status_bar.showMessage("Finishing the current file read before closing…")
            event.ignore()
            return
        save_preferences(self)
        for reader in self.loaded_readers.values():
            try:
                reader.close()
            except Exception:
                pass
        self.loaded_readers.clear()
        self.view_hierarchy.close()
        super().closeEvent(event)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About SuperADAM NeXus Explorer",
            "<h3>SuperADAM NeXus Explorer</h3>"
            "<p>A specialized GUI explorer for ILL SuperADAM polarized neutron reflectometry NeXus (.nxs) scan files.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Multi-file loading and comparison overlay</li>"
            "<li>Aesthetic HDF5 hierarchy tree browser and dataset inspector with Lucide vector icons</li>"
            "<li>Scan charting with spin-state splitting (Flipper 0/1) and log scaling</li>"
            "<li>Poisson count error propagation and monitor rate normalization</li>"
            "<li>Q momentum transfer conversion with wavelength selection</li>"
            "<li>2D detector frame movie player (90° CW oriented) with interactive ROI overlay</li>"
            "<li>Motor positions and sample environment dashboard</li>"
            "</ul>",
        )
