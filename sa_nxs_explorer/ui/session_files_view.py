"""Global session files panel for managing loaded NeXus scans and chart overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from sa_nxs_explorer.core.models import ScanData
from sa_nxs_explorer.ui import icons


class SessionFilesView(QWidget):
    """Global sidebar listing all loaded NeXus scans across the explorer."""

    active_file_changed = pyqtSignal(str)          # Emits file_path when clicked/selected
    overlay_selection_changed = pyqtSignal(list)   # Emits list[str] of checked file_paths for plot overlay
    open_requested = pyqtSignal()
    add_requested = pyqtSignal()
    close_requested = pyqtSignal()
    minimized_changed = pyqtSignal(bool)           # Emits True when minimized, False when expanded

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.is_dark_mode: bool = False
        self.is_minimized: bool = False
        self._last_expanded_width: int = 280
        self._scans: dict[str, ScanData] = {}
        self._active_key: str | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------------- 1. Expanded State Container ----------------
        self.expanded_container = QWidget()
        exp_layout = QVBoxLayout(self.expanded_container)
        exp_layout.setContentsMargins(8, 8, 8, 8)
        exp_layout.setSpacing(8)

        # 1. Header with file count and minimize button
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self.lbl_header = QLabel("Loaded Scans (0)")
        self.lbl_header.setStyleSheet("font-weight: 700; font-size: 13px; color: #0284c7;")
        header_row.addWidget(self.lbl_header, 1)

        self.btn_minimize = QPushButton()
        self.btn_minimize.setIcon(icons.get_icon("panel_left_close", is_dark=self.is_dark_mode))
        self.btn_minimize.setToolTip("Minimize Sidebar to left (Ctrl+B)")
        self.btn_minimize.setFixedSize(28, 26)
        self.btn_minimize.clicked.connect(self.toggle_minimized)
        header_row.addWidget(self.btn_minimize)

        exp_layout.addLayout(header_row)

        # 2. File actions row: Open, Add, Close
        act_row = QHBoxLayout()
        act_row.setSpacing(4)

        self.btn_open = QPushButton("Open")
        self.btn_open.setToolTip("Open new scan file(s) (Ctrl+O)")
        self.btn_open.clicked.connect(self.open_requested.emit)
        act_row.addWidget(self.btn_open)

        self.btn_add = QPushButton("Add")
        self.btn_add.setToolTip("Add scan file(s) to current session (Ctrl+Shift+O)")
        self.btn_add.clicked.connect(self.add_requested.emit)
        act_row.addWidget(self.btn_add)

        self.btn_close = QPushButton("Close")
        self.btn_close.setToolTip("Close currently active file (Ctrl+W)")
        self.btn_close.clicked.connect(self.close_requested.emit)
        act_row.addWidget(self.btn_close)

        exp_layout.addLayout(act_row)

        # 3. List of loaded scans with checkboxes for chart overlay
        self.list_files = QListWidget()
        self.list_files.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list_files.currentItemChanged.connect(self._on_current_item_changed)
        self.list_files.itemChanged.connect(self._on_item_check_changed)
        exp_layout.addWidget(self.list_files, 1)

        # 4. Chart Overlay Selection Controls
        grp_overlay = QGroupBox("Chart Overlay Controls")
        grp_layout = QVBoxLayout(grp_overlay)
        grp_layout.setContentsMargins(6, 10, 6, 6)
        grp_layout.setSpacing(6)

        lbl_hint = QLabel("☑ Tick box to include in Scan Chart overlay")
        lbl_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        lbl_hint.setWordWrap(True)
        grp_layout.addWidget(lbl_hint)

        btn_row = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self._select_all_overlay)
        self.btn_select_none = QPushButton("Clear All")
        self.btn_select_none.clicked.connect(self._select_none_overlay)
        btn_row.addWidget(self.btn_select_all)
        btn_row.addWidget(self.btn_select_none)
        grp_layout.addLayout(btn_row)

        exp_layout.addWidget(grp_overlay)

        # 5. Active File Info Card
        self.info_frame = QFrame()
        self.info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.info_frame.setStyleSheet("""
            QFrame {
                background-color: #f1f5f9;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        info_layout = QVBoxLayout(self.info_frame)
        info_layout.setContentsMargins(6, 6, 6, 6)
        info_layout.setSpacing(3)

        self.lbl_info_title = QLabel("Active Sample:")
        self.lbl_info_title.setWordWrap(True)
        self.lbl_info_title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.lbl_info_title.setStyleSheet("font-weight: 700; font-size: 11px; color: #0369a1;")
        info_layout.addWidget(self.lbl_info_title)

        self.lbl_info_desc = QLabel("No file active")
        self.lbl_info_desc.setStyleSheet("color: #334155; font-size: 11px;")
        self.lbl_info_desc.setWordWrap(True)
        self.lbl_info_desc.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        info_layout.addWidget(self.lbl_info_desc)

        exp_layout.addWidget(self.info_frame)
        main_layout.addWidget(self.expanded_container, 1)

        # ---------------- 2. Minimized State Strip Container ----------------
        self.minimized_container = QWidget()
        self.minimized_container.setVisible(False)
        self.minimized_container.setCursor(Qt.CursorShape.PointingHandCursor)
        self.minimized_container.setToolTip("Click to expand sidebar (Ctrl+B)")

        mini_layout = QVBoxLayout(self.minimized_container)
        mini_layout.setContentsMargins(3, 8, 3, 8)
        mini_layout.setSpacing(10)
        mini_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        self.btn_expand = QPushButton()
        self.btn_expand.setIcon(icons.get_icon("panel_left_open", is_dark=self.is_dark_mode))
        self.btn_expand.setToolTip("Expand Sidebar (Ctrl+B)")
        self.btn_expand.setFixedSize(28, 26)
        self.btn_expand.clicked.connect(self.toggle_minimized)
        mini_layout.addWidget(self.btn_expand)

        self.lbl_mini_count = QLabel("0")
        self.lbl_mini_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_mini_count.setToolTip("Loaded Scans Count — Click to Expand")
        self.lbl_mini_count.setStyleSheet(
            "background-color: #0284c7; color: #ffffff; border-radius: 9px; "
            "font-size: 10px; font-weight: 700; min-width: 18px; max-width: 18px; min-height: 18px;"
        )
        mini_layout.addWidget(self.lbl_mini_count, 0, Qt.AlignmentFlag.AlignHCenter)

        mini_layout.addStretch(1)
        main_layout.addWidget(self.minimized_container)

    def mousePressEvent(self, event) -> None:
        if self.is_minimized and event.button() == Qt.MouseButton.LeftButton:
            self.toggle_minimized()
            return
        super().mousePressEvent(event)

    def toggle_minimized(self) -> None:
        self.set_minimized(not self.is_minimized)

    def set_minimized(self, minimized: bool) -> None:
        if self.is_minimized == minimized:
            return
        self.is_minimized = minimized
        splitter = self.parent() if (self.parent() and hasattr(self.parent(), "sizes")) else None

        if self.is_minimized:
            if splitter is not None:
                cur_w = self.width()
                if cur_w > 80:
                    self._last_expanded_width = cur_w
            self.expanded_container.setVisible(False)
            self.minimized_container.setVisible(True)
            self.setFixedWidth(34)
            if splitter is not None:
                tot = sum(splitter.sizes())
                splitter.setSizes([34, max(50, tot - 34)])
        else:
            self.setMinimumWidth(180)
            self.setMaximumWidth(16777215)
            self.minimized_container.setVisible(False)
            self.expanded_container.setVisible(True)
            if splitter is not None:
                tot = sum(splitter.sizes())
                target_w = self._last_expanded_width if self._last_expanded_width >= 180 else 280
                splitter.setSizes([target_w, max(50, tot - target_w)])

        self.minimized_changed.emit(self.is_minimized)

    def _format_scan_text(self, scan: ScanData) -> str:
        name = Path(scan.file_path).name if scan.file_path else "Scan"
        run_str = f"#{scan.metadata.run_number}" if scan.metadata.run_number else ""
        sub_str = f" - {scan.metadata.subtitle}" if scan.metadata.subtitle else ""
        if run_str:
            return f"{name} ({run_str}{sub_str})"
        return f"{name}{sub_str}"

    def add_file(self, scan_data: ScanData, is_checked: bool = True, make_active: bool = True) -> None:
        """Add or update a scan in the global session list."""
        key = str(scan_data.file_path) if scan_data.file_path else f"scan_{len(self._scans)+1}"
        self._scans[key] = scan_data

        self.list_files.blockSignals(True)
        found_item: QListWidgetItem | None = None
        for i in range(self.list_files.count()):
            it = self.list_files.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == key:
                found_item = it
                break

        if found_item is None:
            item = QListWidgetItem(self._format_scan_text(scan_data))
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(f"{scan_data.metadata.title}\n{scan_data.file_path}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Checked if is_checked else Qt.CheckState.Unchecked)
            self.list_files.addItem(item)
            target_item = item
        else:
            found_item.setText(self._format_scan_text(scan_data))
            if is_checked:
                found_item.setCheckState(Qt.CheckState.Checked)
            target_item = found_item

        self.lbl_header.setText(f"Loaded Scans ({len(self._scans)})")
        self.lbl_mini_count.setText(str(len(self._scans)))
        self.list_files.blockSignals(False)

        if make_active or self._active_key is None:
            self.set_active_key(key)

        self._emit_checked_list()

    def remove_file(self, file_path: str) -> None:
        """Remove a file from the session list."""
        key = str(file_path)
        if key in self._scans:
            del self._scans[key]

        self.list_files.blockSignals(True)
        for i in range(self.list_files.count()):
            it = self.list_files.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == key:
                self.list_files.takeItem(i)
                break
        self.list_files.blockSignals(False)

        self.lbl_header.setText(f"Loaded Scans ({len(self._scans)})")
        self.lbl_mini_count.setText(str(len(self._scans)))

        if self._active_key == key:
            if self.list_files.count() > 0:
                next_key = self.list_files.item(0).data(Qt.ItemDataRole.UserRole)
                self.set_active_key(next_key)
            else:
                self._active_key = None
                self._update_info_card(None)

        self._emit_checked_list()

    def clear_all(self) -> None:
        """Clear all session files."""
        self._scans.clear()
        self._active_key = None
        self.list_files.blockSignals(True)
        self.list_files.clear()
        self.list_files.blockSignals(False)
        self.lbl_header.setText("Loaded Scans (0)")
        self.lbl_mini_count.setText("0")
        self._update_info_card(None)
        self._emit_checked_list()

    def set_active_key(self, key: str, emit_signal: bool = True) -> None:
        """Set the active scan by file path key."""
        self._active_key = key
        self.list_files.blockSignals(True)
        for i in range(self.list_files.count()):
            it = self.list_files.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == key:
                self.list_files.setCurrentItem(it)
                break
        self.list_files.blockSignals(False)

        scan = self._scans.get(key)
        self._update_info_card(scan)
        if emit_signal:
            self.active_file_changed.emit(key)

    def get_checked_keys(self) -> list[str]:
        """Return list of file path keys currently checked for chart overlay."""
        checked = []
        for i in range(self.list_files.count()):
            it = self.list_files.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                k = it.data(Qt.ItemDataRole.UserRole)
                if k:
                    checked.append(k)
        return checked

    def _on_current_item_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if current is not None:
            key = current.data(Qt.ItemDataRole.UserRole)
            if key and key != self._active_key:
                self.set_active_key(key)

    def _on_item_check_changed(self, item: QListWidgetItem) -> None:
        self._emit_checked_list()

    def _select_all_overlay(self) -> None:
        self.list_files.blockSignals(True)
        for i in range(self.list_files.count()):
            self.list_files.item(i).setCheckState(Qt.CheckState.Checked)
        self.list_files.blockSignals(False)
        self._emit_checked_list()

    def _select_none_overlay(self) -> None:
        self.list_files.blockSignals(True)
        for i in range(self.list_files.count()):
            self.list_files.item(i).setCheckState(Qt.CheckState.Unchecked)
        self.list_files.blockSignals(False)
        self._emit_checked_list()

    def _emit_checked_list(self) -> None:
        keys = self.get_checked_keys()
        self.overlay_selection_changed.emit(keys)

    def _update_info_card(self, scan: ScanData | None) -> None:
        if scan is None:
            self.lbl_info_title.setText("Active Sample:")
            self.lbl_info_desc.setText("No file active")
            return

        name = Path(scan.file_path).name if scan.file_path else "Scan"
        run_str = f"Run #{scan.metadata.run_number}" if scan.metadata.run_number else ""
        n_steps = len(next(iter(scan.variables.values())).data) if scan.variables else 0
        title = scan.metadata.title or "(untitled)"
        sub = scan.metadata.subtitle

        self.lbl_info_title.setText(f"Active: {name} {f'({run_str})' if run_str else ''}")
        sub_html = f"<b>Subtitle:</b> {sub[:60]}<br>" if sub else ""
        self.lbl_info_desc.setText(
            f"<b>Title:</b> {title[:50]}<br>"
            f"{sub_html}"
            f"<b>Scan steps:</b> {n_steps} &bull; <b>Motors:</b> {len(scan.motors)}"
        )

    def set_theme(self, is_dark: bool) -> None:
        """Update vector icons and theme colors."""
        self.is_dark_mode = is_dark
        self.btn_open.setIcon(icons.get_icon("folder_open", is_dark=is_dark))
        self.btn_add.setIcon(icons.get_icon("file_plus", is_dark=is_dark))
        self.btn_close.setIcon(icons.get_icon("x_circle", is_dark=is_dark))
        self.btn_minimize.setIcon(icons.get_icon("panel_left_close", is_dark=is_dark))
        self.btn_expand.setIcon(icons.get_icon("panel_left_open", is_dark=is_dark))

        if is_dark:
            self.lbl_header.setStyleSheet("font-weight: 700; font-size: 13px; color: #38bdf8;")
            self.minimized_container.setStyleSheet("background-color: #0f172a; border-right: 1px solid #334155;")
            self.lbl_mini_count.setStyleSheet(
                "background-color: #38bdf8; color: #0f172a; border-radius: 9px; "
                "font-size: 10px; font-weight: 700; min-width: 18px; max-width: 18px; min-height: 18px;"
            )
            self.info_frame.setStyleSheet("""
                QFrame {
                    background-color: #0f172a;
                    border: 1px solid #334155;
                    border-radius: 6px;
                    padding: 6px;
                }
            """)
            self.lbl_info_title.setStyleSheet("font-weight: 700; font-size: 11px; color: #38bdf8;")
            self.lbl_info_desc.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        else:
            self.lbl_header.setStyleSheet("font-weight: 700; font-size: 13px; color: #0284c7;")
            self.minimized_container.setStyleSheet("background-color: #f1f5f9; border-right: 1px solid #cbd5e1;")
            self.lbl_mini_count.setStyleSheet(
                "background-color: #0284c7; color: #ffffff; border-radius: 9px; "
                "font-size: 10px; font-weight: 700; min-width: 18px; max-width: 18px; min-height: 18px;"
            )
            self.info_frame.setStyleSheet("""
                QFrame {
                    background-color: #f1f5f9;
                    border: 1px solid #cbd5e1;
                    border-radius: 6px;
                    padding: 6px;
                }
            """)
            self.lbl_info_title.setStyleSheet("font-weight: 700; font-size: 11px; color: #0369a1;")
            self.lbl_info_desc.setStyleSheet("color: #334155; font-size: 11px;")
