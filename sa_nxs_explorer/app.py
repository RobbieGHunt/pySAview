"""Application entry point and lifecycle manager for SuperADAM NeXus Explorer."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from sa_nxs_explorer.ui.main_window import MainWindow


def exception_hook(exc_type, exc_value, exc_traceback) -> None:
    """Global exception hook to report unexpected crashes cleanly."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"Unhandled Exception:\n{err_msg}", file=sys.stderr)

    app = QApplication.instance()
    if app is not None:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Application Error")
        msg_box.setText("An unexpected error occurred:")
        msg_box.setDetailedText(err_msg)
        msg_box.exec()


def create_app() -> QApplication:
    """Create and configure QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("SuperADAM NeXus Explorer")
    app.setOrganizationName("ILL SuperADAM")
    sys.excepthook = exception_hook
    return app


def main() -> int:
    """Main function parsing arguments and starting event loop."""
    parser = argparse.ArgumentParser(description="SuperADAM NeXus Scan File Explorer")
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Optional path to a .nxs NeXus scan file to open upon launch.",
    )
    args = parser.parse_args()

    app = create_app()
    window = MainWindow()

    # Determine file to open
    target_file: Path | None = None
    if args.file:
        target_file = Path(args.file).resolve()
    else:
        # Default to example scan if present
        default_example = Path(__file__).resolve().parent / "example_data" / "005572.nxs"
        if default_example.exists():
            target_file = default_example

    if target_file and target_file.exists():
        window.load_file(target_file)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
