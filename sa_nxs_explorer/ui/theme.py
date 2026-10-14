"""UI Themes, safe colormaps, and pyqtgraph styling for SuperADAM NeXus Explorer."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QColor

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
CHEVRON_RIGHT_LIGHT = (RESOURCES_DIR / "chevron_right.svg").as_posix()
CHEVRON_DOWN_LIGHT = (RESOURCES_DIR / "chevron_down.svg").as_posix()
CHEVRON_UP_LIGHT = (RESOURCES_DIR / "chevron_up.svg").as_posix()
CHEVRON_RIGHT_DARK = (RESOURCES_DIR / "chevron_right_dark.svg").as_posix()
CHEVRON_DOWN_DARK = (RESOURCES_DIR / "chevron_down_dark.svg").as_posix()
CHEVRON_UP_DARK = (RESOURCES_DIR / "chevron_up_dark.svg").as_posix()
CHECK_SVG = (RESOURCES_DIR / "check.svg").as_posix()

LIGHT_THEME = f"""
QWidget {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
    color: #0f172a;
    background-color: #f8fafc;
}}

QMainWindow, QDialog {{
    background-color: #f1f5f9;
}}

QTabWidget::pane {{
    border: 1px solid #e2e8f0;
    background: #ffffff;
    border-radius: 8px;
}}

QTabBar::tab {{
    background: #e2e8f0;
    color: #64748b;
    padding: 9px 18px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
}}

QTabBar::tab:selected {{
    background: #ffffff;
    color: #0284c7;
    border: 1px solid #e2e8f0;
    border-bottom: 2px solid #0284c7;
}}

QTabBar::tab:hover:!selected {{
    background: #f1f5f9;
    color: #0f172a;
}}

QHeaderView::section {{
    background-color: #f8fafc;
    color: #64748b;
    padding: 7px 10px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: none;
    border-bottom: 1px solid #e2e8f0;
}}

/* Sleek Modern File Hierarchy Tree (IDE-like aesthetic) */
QTreeWidget, QTreeView {{
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    color: #0f172a;
    show-decoration-selected: 1;
    outline: none;
    padding: 4px;
}}

QTreeWidget::item, QTreeView::item {{
    height: 28px;
    padding: 2px 6px;
    border-radius: 4px;
    color: #1e293b;
}}

QTreeWidget::item:hover, QTreeView::item:hover {{
    background-color: #f1f5f9;
}}

QTreeWidget::item:selected, QTreeView::item:selected {{
    background-color: #e0f2fe;
    color: #0369a1;
    font-weight: 600;
}}

/* Sleek chevron branch expanders instead of Windows dropdown arrows */
QTreeView::branch:has-children:!has-siblings:closed,
QTreeView::branch:closed:has-children:has-siblings {{
    border-image: none;
    image: url("{CHEVRON_RIGHT_LIGHT}");
}}

QTreeView::branch:open:has-children:!has-siblings,
QTreeView::branch:open:has-children:has-siblings {{
    border-image: none;
    image: url("{CHEVRON_DOWN_LIGHT}");
}}

QTableWidget, QTableView {{
    background-color: #ffffff;
    alternate-background-color: #f8fafc;
    color: #0f172a;
    gridline-color: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    selection-background-color: #e0f2fe;
    selection-color: #0369a1;
    outline: none;
}}

QTableWidget::item, QTableView::item {{
    padding: 6px 10px;
}}

QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 10px;
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1.5px solid #0284c7;
    background-color: #ffffff;
}}

/* Sleek SpinBox with custom SVG chevron arrows matching ComboBox */
QSpinBox, QDoubleSpinBox {{
    padding-right: 22px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    height: 14px;
    border-left: 1px solid #cbd5e1;
    border-bottom: 0.5px solid #cbd5e1;
    border-top-right-radius: 6px;
    background: #f8fafc;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
    background: #e2e8f0;
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{CHEVRON_UP_LIGHT}");
    width: 9px;
    height: 9px;
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    height: 14px;
    border-left: 1px solid #cbd5e1;
    border-top: 0.5px solid #cbd5e1;
    border-bottom-right-radius: 6px;
    background: #f8fafc;
}}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: #e2e8f0;
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{CHEVRON_DOWN_LIGHT}");
    width: 9px;
    height: 9px;
}}

/* Sleek ComboBox with custom SVG chevron drop-down arrow */
QComboBox {{
    background-color: #ffffff;
    color: #0f172a;
    border: 1.5px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 28px 5px 10px;
    min-height: 20px;
    font-weight: 500;
}}

QComboBox:hover {{
    border-color: #0284c7;
}}

QComboBox:focus {{
    border: 1.5px solid #0284c7;
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: none;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}}

QComboBox::down-arrow {{
    image: url("{CHEVRON_DOWN_LIGHT}");
    width: 12px;
    height: 12px;
}}

QComboBox QAbstractItemView {{
    background-color: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    selection-background-color: #e0f2fe;
    selection-color: #0369a1;
    padding: 4px;
    outline: none;
}}

/* High-contrast Tick Boxes */
QCheckBox {{
    spacing: 8px;
    color: #0f172a;
    font-weight: 500;
}}

QCheckBox::indicator, QListWidget::indicator, QTreeWidget::indicator {{
    width: 17px;
    height: 17px;
    border: 2px solid #475569;
    border-radius: 4px;
    background-color: #ffffff;
}}

QCheckBox::indicator:hover, QListWidget::indicator:hover, QTreeWidget::indicator:hover {{
    border: 2px solid #0284c7;
    background-color: #f0f9ff;
}}

QCheckBox::indicator:checked, QListWidget::indicator:checked, QTreeWidget::indicator:checked {{
    border: 2px solid #0284c7;
    background-color: #0284c7;
    image: url("{CHECK_SVG}");
}}

QCheckBox::indicator:checked:hover, QListWidget::indicator:checked:hover, QTreeWidget::indicator:checked:hover {{
    border: 2px solid #0369a1;
    background-color: #0369a1;
}}

QPushButton {{
    background-color: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    padding: 7px 14px;
    border-radius: 6px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: #f1f5f9;
    border-color: #94a3b8;
}}

QPushButton:pressed {{
    background-color: #e2e8f0;
}}

QPushButton:checked {{
    background-color: #e0f2fe;
    color: #0369a1;
    border: 1.5px solid #0284c7;
}}

QPushButton:checked:hover {{
    background-color: #bae6fd;
}}

QGroupBox {{
    font-weight: 700;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 14px;
    background-color: #ffffff;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #0369a1;
}}

QSlider::groove:horizontal {{
    border: 1px solid #e2e8f0;
    height: 6px;
    background: #e2e8f0;
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: #0284c7;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: #ffffff;
    border: 2px solid #0284c7;
    width: 16px;
    margin-top: -6px;
    margin-bottom: -6px;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: #e0f2fe;
}}

QToolBar {{
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 6px 8px;
    spacing: 8px;
}}

QStatusBar {{
    background: #f8fafc;
    border-top: 1px solid #e2e8f0;
    color: #64748b;
    font-size: 12px;
    padding: 4px 8px;
}}

/* Custom ScrollBars for TreeViews, Tables, and panels */
QScrollBar:vertical {{
    background: #f1f5f9;
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background: #cbd5e1;
    min-height: 24px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical:hover {{
    background: #94a3b8;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
    background: transparent;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}

QScrollBar:horizontal {{
    background: #f1f5f9;
    height: 10px;
    margin: 0px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background: #cbd5e1;
    min-width: 24px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal:hover {{
    background: #94a3b8;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
    background: transparent;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}
"""

DARK_THEME = f"""
QWidget {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
    color: #f8fafc;
    background-color: #0b1120;
}}

QMainWindow, QDialog {{
    background-color: #0b1120;
}}

QTabWidget::pane {{
    border: 1px solid #1e293b;
    background: #0f172a;
    border-radius: 8px;
}}

QTabBar::tab {{
    background: #0f172a;
    color: #94a3b8;
    padding: 9px 18px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
}}

QTabBar::tab:selected {{
    background: #1e293b;
    color: #38bdf8;
    border: 1px solid #334155;
    border-bottom: 2px solid #38bdf8;
}}

QTabBar::tab:hover:!selected {{
    background: #1e293b;
    color: #f1f5f9;
}}

QHeaderView::section {{
    background-color: #0f172a;
    color: #94a3b8;
    padding: 7px 10px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: none;
    border-bottom: 1px solid #1e293b;
}}

/* Sleek Modern File Hierarchy Tree (Dark Mode) */
QTreeWidget, QTreeView {{
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    color: #f8fafc;
    show-decoration-selected: 1;
    outline: none;
    padding: 4px;
}}

QTreeWidget::item, QTreeView::item {{
    height: 28px;
    padding: 2px 6px;
    border-radius: 4px;
    color: #f1f5f9;
}}

QTreeWidget::item:hover, QTreeView::item:hover {{
    background-color: #1e293b;
}}

QTreeWidget::item:selected, QTreeView::item:selected {{
    background-color: #0c4a6e;
    color: #7dd3fc;
    font-weight: 600;
}}

/* Sleek chevron branch expanders in Dark Mode */
QTreeView::branch:has-children:!has-siblings:closed,
QTreeView::branch:closed:has-children:has-siblings {{
    border-image: none;
    image: url("{CHEVRON_RIGHT_DARK}");
}}

QTreeView::branch:open:has-children:!has-siblings,
QTreeView::branch:open:has-children:has-siblings {{
    border-image: none;
    image: url("{CHEVRON_DOWN_DARK}");
}}

QTableWidget, QTableView {{
    background-color: #0f172a;
    alternate-background-color: #0b1120;
    color: #f8fafc;
    gridline-color: #1e293b;
    border: 1px solid #1e293b;
    border-radius: 8px;
    selection-background-color: #0c4a6e;
    selection-color: #7dd3fc;
    outline: none;
}}

QTableWidget::item, QTableView::item {{
    padding: 6px 10px;
}}

QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: #0f172a;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 10px;
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1.5px solid #38bdf8;
}}

/* Sleek SpinBox with custom SVG chevron arrows matching ComboBox in Dark Theme */
QSpinBox, QDoubleSpinBox {{
    padding-right: 22px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    height: 14px;
    border-left: 1px solid #475569;
    border-bottom: 0.5px solid #475569;
    border-top-right-radius: 6px;
    background: #1e293b;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
    background: #334155;
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{CHEVRON_UP_DARK}");
    width: 9px;
    height: 9px;
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    height: 14px;
    border-left: 1px solid #475569;
    border-top: 0.5px solid #475569;
    border-bottom-right-radius: 6px;
    background: #1e293b;
}}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: #334155;
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{CHEVRON_DOWN_DARK}");
    width: 9px;
    height: 9px;
}}

/* Sleek ComboBox with custom SVG chevron drop-down arrow in Dark Theme */
QComboBox {{
    background-color: #0f172a;
    color: #f8fafc;
    border: 1.5px solid #475569;
    border-radius: 6px;
    padding: 5px 28px 5px 10px;
    min-height: 20px;
    font-weight: 500;
}}

QComboBox:hover {{
    border-color: #38bdf8;
}}

QComboBox:focus {{
    border: 1.5px solid #38bdf8;
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: none;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}}

QComboBox::down-arrow {{
    image: url("{CHEVRON_DOWN_DARK}");
    width: 12px;
    height: 12px;
}}

QComboBox QAbstractItemView {{
    background-color: #0f172a;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 6px;
    selection-background-color: #0c4a6e;
    selection-color: #7dd3fc;
    padding: 4px;
    outline: none;
}}

/* High-contrast Tick Boxes in Dark Theme */
QCheckBox {{
    spacing: 8px;
    color: #f1f5f9;
    font-weight: 500;
}}

QCheckBox::indicator, QListWidget::indicator, QTreeWidget::indicator {{
    width: 17px;
    height: 17px;
    border: 2px solid #94a3b8;
    border-radius: 4px;
    background-color: #0f172a;
}}

QCheckBox::indicator:hover, QListWidget::indicator:hover, QTreeWidget::indicator:hover {{
    border: 2px solid #38bdf8;
    background-color: #1e293b;
}}

QCheckBox::indicator:checked, QListWidget::indicator:checked, QTreeWidget::indicator:checked {{
    border: 2px solid #38bdf8;
    background-color: #0284c7;
    image: url("{CHECK_SVG}");
}}

QCheckBox::indicator:checked:hover, QListWidget::indicator:checked:hover, QTreeWidget::indicator:checked:hover {{
    border: 2px solid #0369a1;
    background-color: #0369a1;
}}

QPushButton {{
    background-color: #1e293b;
    color: #f8fafc;
    border: 1px solid #334155;
    padding: 7px 14px;
    border-radius: 6px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: #334155;
    border-color: #475569;
}}

QPushButton:pressed {{
    background-color: #475569;
}}

QPushButton:checked {{
    background-color: #0c4a6e;
    color: #38bdf8;
    border: 1.5px solid #38bdf8;
}}

QPushButton:checked:hover {{
    background-color: #075985;
}}

QGroupBox {{
    font-weight: 700;
    border: 1px solid #1e293b;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 14px;
    background-color: #0f172a;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #38bdf8;
}}

QSlider::groove:horizontal {{
    border: 1px solid #1e293b;
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: #0284c7;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: #f8fafc;
    border: 2px solid #38bdf8;
    width: 16px;
    margin-top: -6px;
    margin-bottom: -6px;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: #e0f2fe;
}}

QToolBar {{
    background: #0f172a;
    border-bottom: 1px solid #1e293b;
    padding: 6px 8px;
    spacing: 8px;
}}

QStatusBar {{
    background: #0b1120;
    border-top: 1px solid #1e293b;
    color: #94a3b8;
    font-size: 12px;
    padding: 4px 8px;
}}

/* Custom ScrollBars for TreeViews, Tables, and panels in Dark Mode */
QScrollBar:vertical {{
    background: #0f172a;
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background: #334155;
    min-height: 24px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical:hover {{
    background: #475569;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
    background: transparent;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}

QScrollBar:horizontal {{
    background: #0f172a;
    height: 10px;
    margin: 0px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background: #334155;
    min-width: 24px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal:hover {{
    background: #475569;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
    background: transparent;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}
"""


def get_safe_colormap(name: str) -> pg.ColorMap:
    """Return a valid pyqtgraph ColorMap, supporting 'turbo', 'viridis', 'inferno', 'plasma', and custom 'gray'.

    Guarantees no FileNotFoundError or crashing when 'gray' is chosen.
    """
    name_l = name.strip().lower()
    if name_l in ("gray", "grey", "grayscale"):
        pos = np.array([0.0, 1.0])
        color = np.array([[0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte)
        return pg.ColorMap(pos=pos, color=color)

    try:
        return pg.colormap.get(name_l)
    except Exception:
        return pg.colormap.get("turbo")


def apply_plot_theme(plot_widget: pg.PlotWidget, is_dark: bool) -> None:
    """Style pyqtgraph PlotWidget canvas, axes, grid, and legend for light or dark mode.

    Includes a closed border box on top and right side for consistency without tick labels,
    and ensures scientific units (e.g. Q in 1/Å) do not get auto-prefixed with milli (m1/Å).
    """
    bg_color = "#0b1120" if is_dark else "#ffffff"
    text_color = "#f1f5f9" if is_dark else "#0f172a"
    axis_color = "#94a3b8" if is_dark else "#64748b"
    grid_alpha = 0.22 if is_dark else 0.3

    plot_widget.setBackground(bg_color)
    plot_item = plot_widget.getPlotItem()

    # Enable top and right border axes for a consistent closed frame without tick labels
    plot_item.showAxis("top", show=True)
    plot_item.showAxis("right", show=True)
    top_ax = plot_item.getAxis("top")
    if top_ax is not None:
        top_ax.setStyle(showValues=False)
    right_ax = plot_item.getAxis("right")
    if right_ax is not None:
        right_ax.setStyle(showValues=False)

    for ax_name in ("left", "bottom", "right", "top"):
        ax = plot_item.getAxis(ax_name)
        if ax is not None:
            ax.setPen(pg.mkPen(axis_color, width=1))
            ax.setTextPen(pg.mkPen(axis_color))
            # Never prepend milli or SI prefixes to Q or other custom units
            ax.enableAutoSIPrefix(False)
            if hasattr(ax, "labelText") and ax.labelText:
                units = getattr(ax, "labelUnits", None) or ""
                ax.setLabel(ax.labelText, units=units, color=text_color)
            # Disabling prefixes does not clear a previously computed scale
            # in every pyqtgraph version, especially after a theme/log change.
            if hasattr(ax, "setSIPrefixEnableRanges"):
                ax.setSIPrefixEnableRanges(())
            ax.autoSIPrefixScale = 1.0
            ax.labelUnitPrefix = ""
            ax._updateLabel()
            ax.picture = None
            ax.update()
    plot_item.showGrid(x=True, y=True, alpha=grid_alpha)

    if plot_item.legend is not None:
        plot_item.legend.setLabelTextColor(text_color)
        legend_bg = QColor(11, 17, 32, 220) if is_dark else QColor(255, 255, 255, 220)
        plot_item.legend.setBrush(pg.mkBrush(legend_bg))
        plot_item.legend.setPen(pg.mkPen(axis_color, width=1))
