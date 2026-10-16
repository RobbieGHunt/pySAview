"""Aesthetic File structure explorer tab for NeXus / HDF5 hierarchies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pySAview.core.models import DatasetPreview, NodeInfo
from pySAview.core.nexus_reader import NexusReader
from pySAview.ui import icons
from pySAview.ui.theme import apply_plot_theme, get_safe_colormap


class FileStructureView(QWidget):
    """Interactive hierarchy browser and dataset inspector with modern Lucide icons."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.reader: NexusReader | None = None
        self.is_dark_mode: bool = False
        self._current_node: NodeInfo | None = None
        self.is_tree_minimized: bool = False
        self._last_tree_width: int = 390
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        # ---------------- Left Panel: Tree & Search ----------------
        self.left_widget = QWidget()
        left_layout = QVBoxLayout(self.left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # 1. Expanded tree container
        self.tree_expanded_container = QWidget()
        tree_exp_layout = QVBoxLayout(self.tree_expanded_container)
        tree_exp_layout.setContentsMargins(0, 0, 0, 0)
        tree_exp_layout.setSpacing(8)

        # Search / filter header with minimize button
        search_box = QHBoxLayout()
        search_box.setSpacing(6)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter hierarchy by name or path...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._on_search_text_changed)
        search_box.addWidget(self.search_input, 1)

        self.btn_collapse = QPushButton("Collapse")
        self.btn_collapse.setIcon(icons.get_icon("chevron_right", is_dark=self.is_dark_mode))
        self.btn_collapse.setToolTip("Collapse all tree groups")
        self.btn_collapse.clicked.connect(self._collapse_all)
        search_box.addWidget(self.btn_collapse)

        self.btn_minimize_tree = QPushButton()
        self.btn_minimize_tree.setIcon(icons.get_icon("panel_left_close", is_dark=self.is_dark_mode))
        self.btn_minimize_tree.setToolTip("Minimize File Hierarchy Tree to left (free up horizontal space)")
        self.btn_minimize_tree.setFixedSize(28, 26)
        self.btn_minimize_tree.clicked.connect(self.toggle_tree_minimized)
        search_box.addWidget(self.btn_minimize_tree)

        tree_exp_layout.addLayout(search_box)

        # Hierarchy Tree with modern styling
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Node", "Type & Shape", "NeXus Class"])
        self.tree.setIndentation(18)
        self.tree.setAnimated(True)
        self.tree.header().resizeSection(0, 260)
        self.tree.header().resizeSection(1, 150)
        self.tree.itemExpanded.connect(self._on_item_expanded)
        self.tree.itemClicked.connect(self._on_item_clicked)
        tree_exp_layout.addWidget(self.tree, 1)

        left_layout.addWidget(self.tree_expanded_container, 1)

        # 2. Minimized tree strip container
        self.tree_minimized_container = QWidget()
        self.tree_minimized_container.setVisible(False)
        self.tree_minimized_container.setCursor(Qt.CursorShape.PointingHandCursor)
        self.tree_minimized_container.setToolTip("Click to expand File Hierarchy Tree")

        tree_mini_layout = QVBoxLayout(self.tree_minimized_container)
        tree_mini_layout.setContentsMargins(3, 8, 3, 8)
        tree_mini_layout.setSpacing(10)
        tree_mini_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        self.btn_expand_tree = QPushButton()
        self.btn_expand_tree.setIcon(icons.get_icon("panel_left_open", is_dark=self.is_dark_mode))
        self.btn_expand_tree.setToolTip("Expand File Hierarchy Tree")
        self.btn_expand_tree.setFixedSize(28, 26)
        self.btn_expand_tree.clicked.connect(self.toggle_tree_minimized)
        tree_mini_layout.addWidget(self.btn_expand_tree)

        tree_mini_layout.addStretch(1)
        left_layout.addWidget(self.tree_minimized_container)

        self.splitter.addWidget(self.left_widget)

        # ---------------- Right Panel: Node Details & Previews ----------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 1. Node metadata card
        meta_group = QGroupBox("Node Information & Attributes")
        meta_layout = QVBoxLayout(meta_group)
        meta_layout.setContentsMargins(10, 14, 10, 10)
        meta_layout.setSpacing(8)

        path_row = QHBoxLayout()
        self.lbl_path = QLabel("Select a node from the hierarchy tree to view details.")
        self.lbl_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_path.setStyleSheet("font-weight: 600; font-family: monospace;")
        path_row.addWidget(self.lbl_path, 1)

        self.btn_copy = QPushButton("Copy Path")
        self.btn_copy.setIcon(icons.get_icon("copy", is_dark=self.is_dark_mode))
        self.btn_copy.setMaximumWidth(110)
        self.btn_copy.clicked.connect(self._copy_current_path)
        path_row.addWidget(self.btn_copy)
        meta_layout.addLayout(path_row)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        meta_layout.addWidget(self.lbl_summary)

        # Attributes table
        self.attr_table = QTableWidget(0, 2)
        self.attr_table.setHorizontalHeaderLabels(["Attribute Name", "Value"])
        self.attr_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.attr_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.attr_table.setMaximumHeight(135)
        meta_layout.addWidget(self.attr_table)

        right_layout.addWidget(meta_group)

        # 2. Dataset Visual & Tabular Preview
        preview_group = QGroupBox("Data Preview & Inspection")
        prev_layout = QVBoxLayout(preview_group)
        prev_layout.setContentsMargins(10, 14, 10, 10)

        self.prev_tabs = QTabWidget()

        # Tab 1: Visual Plot
        tab_visual = QWidget()
        vis_layout = QVBoxLayout(tab_visual)
        vis_layout.setContentsMargins(6, 6, 6, 6)

        vis_controls = QHBoxLayout()
        self.chk_log_y = QCheckBox("Log Scale")
        self.chk_log_y.toggled.connect(self._on_log_toggled)
        vis_controls.addWidget(self.chk_log_y)
        vis_controls.addStretch()

        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet("color: #0284c7; font-weight: 600;")
        vis_controls.addWidget(self.lbl_stats)
        vis_layout.addLayout(vis_controls)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        apply_plot_theme(self.plot_widget, self.is_dark_mode)
        vis_layout.addWidget(self.plot_widget, 1)

        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setVisible(False)
        vis_layout.addWidget(self.text_preview, 1)

        self.prev_tabs.addTab(tab_visual, "Visual Chart / Image")

        # Tab 2: Tabular Grid
        tab_table = QWidget()
        tbl_layout = QVBoxLayout(tab_table)
        tbl_layout.setContentsMargins(6, 6, 6, 6)
        self.data_table = QTableWidget()
        tbl_layout.addWidget(self.data_table)
        self.prev_tabs.addTab(tab_table, "Spreadsheet Table")

        prev_layout.addWidget(self.prev_tabs, 1)
        right_layout.addWidget(preview_group, 1)

        self.splitter.addWidget(right_widget)
        self.splitter.setSizes([390, 660])

    def mousePressEvent(self, event) -> None:
        if self.is_tree_minimized and event.button() == Qt.MouseButton.LeftButton:
            if hasattr(self, "left_widget") and self.left_widget.geometry().contains(event.pos()):
                self.toggle_tree_minimized()
                return
        super().mousePressEvent(event)

    def toggle_tree_minimized(self) -> None:
        self.set_tree_minimized(not self.is_tree_minimized)

    def set_tree_minimized(self, minimized: bool) -> None:
        if self.is_tree_minimized == minimized:
            return
        self.is_tree_minimized = minimized
        if self.is_tree_minimized:
            cur_w = self.left_widget.width()
            if cur_w > 80:
                self._last_tree_width = cur_w
            self.tree_expanded_container.setVisible(False)
            self.tree_minimized_container.setVisible(True)
            self.left_widget.setFixedWidth(34)
            tot = sum(self.splitter.sizes())
            self.splitter.setSizes([34, max(50, tot - 34)])
        else:
            self.left_widget.setMinimumWidth(200)
            self.left_widget.setMaximumWidth(16777215)
            self.tree_minimized_container.setVisible(False)
            self.tree_expanded_container.setVisible(True)
            tot = sum(self.splitter.sizes())
            target_w = self._last_tree_width if self._last_tree_width >= 200 else 390
            self.splitter.setSizes([target_w, max(50, tot - target_w)])

    def set_file(self, path: Path | str) -> None:
        """Load a new NeXus file into the hierarchy viewer."""
        if self.reader is not None:
            self.reader.close()
        self.reader = NexusReader(path)
        self._populate_root()

    def set_theme(self, is_dark: bool) -> None:
        """Update styling for dark or light mode."""
        self.is_dark_mode = is_dark
        self.btn_collapse.setIcon(icons.get_icon("chevron_right", is_dark=is_dark))
        self.btn_copy.setIcon(icons.get_icon("copy", is_dark=is_dark))
        self.btn_minimize_tree.setIcon(icons.get_icon("panel_left_close", is_dark=is_dark))
        self.btn_expand_tree.setIcon(icons.get_icon("panel_left_open", is_dark=is_dark))
        if is_dark:
            self.tree_minimized_container.setStyleSheet("background-color: #0f172a; border-right: 1px solid #334155;")
        else:
            self.tree_minimized_container.setStyleSheet("background-color: #f1f5f9; border-right: 1px solid #cbd5e1;")
        apply_plot_theme(self.plot_widget, is_dark)
        self._update_tree_icons()
        if self._current_node is not None:
            self._display_node(self._current_node)

    def close(self) -> None:
        """Close underlying HDF5 file handle."""
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def _collapse_all(self) -> None:
        self.tree.collapseAll()

    def _populate_root(self) -> None:
        self.tree.clear()
        if self.reader is None:
            return

        roots = self.reader.list_children("/")
        for r in roots:
            item = self._create_tree_item(r)
            if r.kind == "group":
                item.addChild(QTreeWidgetItem(["Loading...", "", ""]))
            self.tree.addTopLevelItem(item)

        # Expand root top-level items by default
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if top:
                top.setExpanded(True)

    def _create_tree_item(self, node: NodeInfo) -> QTreeWidgetItem:
        nx_class = node.attrs.get("NX_class", "")
        if node.kind == "group":
            col0 = node.name
            col1 = f"Group ({node.n_children})"
            icon_name = "database" if node.path in ("/", "/entry0") else "folder"
        else:
            col0 = node.name
            if node.shape:
                shape_str = " × ".join(str(d) for d in node.shape)
                col1 = f"{node.dtype} [{shape_str}]"
            else:
                col1 = f"{node.dtype} (scalar)"
            icon_name = "binary"

        item = QTreeWidgetItem([col0, col1, nx_class])
        item.setIcon(0, icons.get_icon(icon_name, is_dark=self.is_dark_mode))
        item.setData(0, Qt.ItemDataRole.UserRole, node)
        return item

    def _update_tree_icons(self) -> None:
        """Refresh vector icon stroke colors upon theme change."""
        def _recurse(item: QTreeWidgetItem):
            node: NodeInfo = item.data(0, Qt.ItemDataRole.UserRole)
            if node is not None:
                if node.kind == "group":
                    ic_name = "database" if node.path in ("/", "/entry0") else "folder"
                else:
                    ic_name = "binary"
                item.setIcon(0, icons.get_icon(ic_name, is_dark=self.is_dark_mode))
            for i in range(item.childCount()):
                _recurse(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            _recurse(self.tree.topLevelItem(i))

    def _on_item_expanded(self, item: QTreeWidgetItem) -> None:
        node: NodeInfo = item.data(0, Qt.ItemDataRole.UserRole)
        if node is None or node.kind != "group" or self.reader is None:
            return

        if item.childCount() == 1 and "Loading..." in item.child(0).text(0):
            item.takeChildren()
            children = self.reader.list_children(node.path)
            for c in children:
                sub_item = self._create_tree_item(c)
                if c.kind == "group" and c.n_children > 0:
                    sub_item.addChild(QTreeWidgetItem(["Loading...", "", ""]))
                item.addChild(sub_item)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        node: NodeInfo = item.data(0, Qt.ItemDataRole.UserRole)
        if node is None or self.reader is None:
            return
        self._current_node = node
        self._display_node(node)

    def _copy_current_path(self) -> None:
        if self._current_node is not None:
            QApplication.clipboard().setText(self._current_node.path)

    def _on_search_text_changed(self, text: str) -> None:
        query = text.strip().lower()
        if not query:
            for i in range(self.tree.topLevelItemCount()):
                self._filter_tree_item(self.tree.topLevelItem(i), "")
            return

        for i in range(self.tree.topLevelItemCount()):
            self._filter_tree_item(self.tree.topLevelItem(i), query)

    def _filter_tree_item(self, item: QTreeWidgetItem | None, query: str) -> bool:
        if item is None:
            return False
        if not query:
            item.setHidden(False)
            for i in range(item.childCount()):
                self._filter_tree_item(item.child(i), "")
            return True

        node: NodeInfo = item.data(0, Qt.ItemDataRole.UserRole)
        match_self = False
        if node is not None:
            match_self = query in node.name.lower() or query in node.path.lower()
        else:
            match_self = query in item.text(0).lower()

        match_child = False
        for i in range(item.childCount()):
            if self._filter_tree_item(item.child(i), query):
                match_child = True

        visible = match_self or match_child
        item.setHidden(not visible)
        if visible and match_child:
            item.setExpanded(True)
        return visible

    def _display_node(self, node: NodeInfo) -> None:
        if self.reader is None:
            return

        self.lbl_path.setText(f"Path: {node.path}")
        summary = f"Kind: {node.kind.capitalize()}"
        if node.kind == "dataset":
            shape_str = f"[{' × '.join(str(d) for d in node.shape)}]" if node.shape else "Scalar"
            summary += f"  |  Shape: {shape_str}  |  Dtype: {node.dtype}"
        else:
            summary += f"  |  Children: {node.n_children}"
        self.lbl_summary.setText(summary)

        # Populate Attributes Table
        attrs = self.reader.get_attributes(node.path)
        self.attr_table.setRowCount(len(attrs))
        for row, (k, v) in enumerate(attrs.items()):
            self.attr_table.setItem(row, 0, QTableWidgetItem(k))
            self.attr_table.setItem(row, 1, QTableWidgetItem(str(v)))

        # Clear visual and table previews
        self.plot_widget.clear()
        self.lbl_stats.setText("")
        self.data_table.clear()
        self.data_table.setRowCount(0)
        self.data_table.setColumnCount(0)

        if node.kind == "group":
            self.plot_widget.setVisible(False)
            self.text_preview.setVisible(True)
            self.text_preview.setText(
                f"Group: {node.path}\nNumber of children: {node.n_children}\n\n"
                f"Expand this group in the tree view to inspect its child datasets and sub-groups."
            )
            return

        # It's a Dataset
        prev: DatasetPreview = self.reader.read_preview(node.path)

        if prev.stats:
            stats_str = f"Min: {prev.stats.get('min', 0):.4g}  |  Max: {prev.stats.get('max', 0):.4g}  |  Mean: {prev.stats.get('mean', 0):.4g}"
            self.lbl_stats.setText(stats_str)

        # Visual preview
        if prev.text and prev.data is None:
            self.plot_widget.setVisible(False)
            self.text_preview.setVisible(True)
            self.text_preview.setText(prev.text)
        elif prev.data is not None and prev.data.size > 0:
            self.text_preview.setVisible(False)
            self.plot_widget.setVisible(True)

            text_color = "#f1f5f9" if self.is_dark_mode else "#0f172a"
            pen_color = "#38bdf8" if self.is_dark_mode else "#0284c7"

            if prev.data.ndim == 1:
                self.plot_widget.setLabel("bottom", "Point Index", color=text_color)
                self.plot_widget.setLabel("left", node.name, color=text_color)
                y_plot = prev.data
                if self.chk_log_y.isChecked():
                    y_plot = np.where(y_plot > 0, y_plot, np.nan)
                    self.plot_widget.setLogMode(y=True)
                else:
                    self.plot_widget.setLogMode(y=False)

                self.plot_widget.plot(
                    y_plot,
                    pen=pg.mkPen(pen_color, width=2),
                    symbol="o",
                    symbolSize=4,
                    symbolBrush=pen_color,
                )
            elif prev.data.ndim == 2:
                self.plot_widget.setLogMode(y=False)
                img = pg.ImageItem()
                img_data = prev.data
                if self.chk_log_y.isChecked():
                    img_data = np.log10(np.maximum(img_data, 1.0))
                img.setImage(img_data)
                # Default colormap: turbo (safe)
                img.setColorMap(get_safe_colormap("turbo"))
                self.plot_widget.addItem(img)
                self.plot_widget.setLabel("bottom", "X Pixel", color=text_color)
                self.plot_widget.setLabel("left", "Y Pixel", color=text_color)

            apply_plot_theme(self.plot_widget, self.is_dark_mode)

        # Tabular data preview
        if prev.data is not None and prev.data.size > 0:
            self._populate_table_preview(prev.data)

    def _on_log_toggled(self) -> None:
        if self._current_node is not None:
            self._display_node(self._current_node)

    def _populate_table_preview(self, data: np.ndarray, max_rows: int = 500, max_cols: int = 30) -> None:
        """Populate spreadsheet table preview with data array."""
        if data.ndim == 1:
            n_rows = min(len(data), max_rows)
            self.data_table.setRowCount(n_rows)
            self.data_table.setColumnCount(2)
            self.data_table.setHorizontalHeaderLabels(["Index", "Value"])
            for r in range(n_rows):
                self.data_table.setItem(r, 0, QTableWidgetItem(str(r)))
                val = data[r]
                val_str = f"{val:.6g}" if isinstance(val, (int, float, np.floating)) else str(val)
                self.data_table.setItem(r, 1, QTableWidgetItem(val_str))
        elif data.ndim == 2:
            n_rows = min(data.shape[0], max_rows)
            n_cols = min(data.shape[1], max_cols)
            self.data_table.setRowCount(n_rows)
            self.data_table.setColumnCount(n_cols)
            self.data_table.setHorizontalHeaderLabels([f"Col {c}" for c in range(n_cols)])
            for r in range(n_rows):
                for c in range(n_cols):
                    val = data[r, c]
                    val_str = f"{val:.5g}" if isinstance(val, (int, float, np.floating)) else str(val)
                    self.data_table.setItem(r, c, QTableWidgetItem(val_str))
