"""Bundle-safe package data path helpers for source and frozen builds."""

from __future__ import annotations

import sys
from pathlib import Path


def package_root() -> Path:
    """Return the root directory of the ``pySAview`` package.

    In PyInstaller frozen builds, data files are unpacked under ``sys._MEIPASS``.
    In normal Python installations, they live alongside the source package.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        frozen_pkg = Path(sys._MEIPASS) / "pySAview"
        if frozen_pkg.exists():
            return frozen_pkg
    return Path(__file__).resolve().parent.parent


def resources_dir() -> Path:
    """Return the ``pySAview/resources`` directory."""
    return package_root() / "resources"


def example_data_dir() -> Path:
    """Return the ``pySAview/example_data`` directory."""
    return package_root() / "example_data"
