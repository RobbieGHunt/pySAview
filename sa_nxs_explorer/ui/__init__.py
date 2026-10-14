"""UI modules for SuperADAM NeXus Explorer."""

from sa_nxs_explorer.ui.detector_view import DetectorView
from sa_nxs_explorer.ui.file_structure_view import FileStructureView
from sa_nxs_explorer.ui.main_window import MainWindow
from sa_nxs_explorer.ui.metadata_view import MetadataView
from sa_nxs_explorer.ui.scan_plot_view import ScanPlotView
from sa_nxs_explorer.ui import icons
from sa_nxs_explorer.ui.theme import DARK_THEME, LIGHT_THEME, apply_plot_theme, get_safe_colormap

__all__ = [
    "DARK_THEME",
    "DetectorView",
    "FileStructureView",
    "LIGHT_THEME",
    "MainWindow",
    "MetadataView",
    "ScanPlotView",
    "apply_plot_theme",
    "get_safe_colormap",
    "icons",
]
