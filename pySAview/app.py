"""Application entry point and lifecycle manager for SuperADAM NeXus Explorer."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

MainWindow = None


def exception_hook(exc_type, exc_value, exc_traceback) -> None:
    """Global exception hook to report unexpected crashes cleanly."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"Unhandled Exception:\n{err_msg}", file=sys.stderr)

    from PyQt6.QtWidgets import QApplication, QMessageBox

    app = QApplication.instance()
    if app is not None:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Application Error")
        msg_box.setText("An unexpected error occurred:")
        msg_box.setDetailedText(err_msg)
        msg_box.exec()


def create_app():
    """Create and configure QApplication."""
    from PyQt6.QtWidgets import QApplication

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ILLSuperADAM.pySAview")
        except Exception:
            pass
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("pySAview")
    app.setApplicationDisplayName("pySAview")
    app.setOrganizationName("ILL SuperADAM")
    sys.excepthook = exception_hook
    return app


def _run_smoke_test(report_path: str) -> int:
    """Verify frozen/installed startup, resources, QThread HDF5 load, colormaps, and scipy.special."""
    t0 = time.perf_counter()
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QColor, QPainter, QPixmap
    from PyQt6.QtSvg import QSvgRenderer

    from pySAview.core.colormaps import colormaps
    from pySAview.core.curve_colors import spin_palette
    from pySAview.core.normalization import normalize_signal, poisson_ratio_interval
    from pySAview.core.paths import example_data_dir, resources_dir
    from pySAview.core.source_safety import validate_output_path
    from pySAview.ui.file_loading import FileLoadThread
    from pySAview.ui.main_window import MainWindow

    app = create_app()
    sys.excepthook = sys.__excepthook__
    window = MainWindow()
    # Snapshot import cleanliness before showEvent schedules async scipy.special warmup.
    pre_import_scipy_stats = "scipy.stats" in sys.modules
    pre_import_scipy_special = "scipy.special" in sys.modules
    pre_import_matplotlib = "matplotlib" in sys.modules

    window.show()
    t_show = time.perf_counter()

    outcome: dict[str, object] = {"ok": False}

    def execute_checks() -> None:
        try:
            t_tick = time.perf_counter()

            res_dir = resources_dir()
            svg_path = res_dir / "chevron_right.svg"
            if not svg_path.exists():
                raise FileNotFoundError(f"Missing SVG resource: {svg_path}")
            renderer = QSvgRenderer(str(svg_path))
            pm = QPixmap(20, 20)
            pm.fill(QColor(0, 0, 0, 0))
            painter = QPainter(pm)
            renderer.render(painter)
            painter.end()
            if pm.isNull() or not renderer.isValid():
                raise RuntimeError("QtSvg rendering failed")

            ex_dir = example_data_dir()
            sample = ex_dir / "005572.nxs"
            if not sample.exists():
                raise FileNotFoundError(f"Missing bundled example scan: {sample}")

            # Verify output path guard before writing any report
            safe_report = validate_output_path(report_path, source_paths=[sample])

            # Load example scan read-only on a QThread (exercises PyQt6 threading + h5py + parallel ROI)
            t_load0 = time.perf_counter()
            loaded: list[object] = []
            worker = FileLoadThread(1, [sample])
            worker.parsed.connect(lambda _tok, scan: loaded.append(scan))
            worker.start()
            if not worker.wait(60_000):
                raise TimeoutError("Timed out loading bundled example scan on QThread")
            app.processEvents()
            t_load1 = time.perf_counter()
            if worker.errors:
                raise RuntimeError(f"FileLoadThread error: {worker.errors}")
            if not loaded:
                raise RuntimeError("FileLoadThread produced no scan")
            scan = loaded[0]

            # Exact Poisson & ratio intervals via scipy.special
            t_norm0 = time.perf_counter()
            norm_res = normalize_signal([0, 10, 100])
            r_lo, r_hi = poisson_ratio_interval([0, 10], [10, 100])
            t_norm1 = time.perf_counter()
            if abs(float(norm_res.upper[0]) - 3.783184333682032) > 1e-6:
                raise AssertionError(f"Unexpected exact zero upper limit: {norm_res.upper[0]}")
            if not (r_lo[0] == 0.0 and r_hi[0] > 0.0 and r_lo[1] < 0.1 < r_hi[1]):
                raise AssertionError("Unexpected poisson_ratio_interval result")

            # Colormaps (no matplotlib)
            pal = spin_palette(0, 3, down="Purples_r", up="Reds")
            rdbu = colormaps["RdBu_r"].rgb_uint8
            if rdbu.shape != (256, 3):
                raise AssertionError(f"Unexpected RdBu_r LUT shape: {rdbu.shape}")
            if "matplotlib" in sys.modules or "scipy.stats" in sys.modules:
                raise AssertionError("Unexpected heavy module imported during runtime")

            outcome.update({
                "ok": True,
                "frozen": bool(getattr(sys, "frozen", False)),
                "platform": sys.platform,
                "launch_to_tick_s": round(t_tick - t0, 4),
                "launch_to_show_s": round(t_show - t0, 4),
                "qthread_h5py_load_s": round(t_load1 - t_load0, 4),
                "scipy_special_norm_s": round(t_norm1 - t_norm0, 4),
                "scan_variables": len(scan.variables),
                "has_detector_2d": bool(scan.has_detector_2d),
                "exact_zero_upper": round(float(norm_res.upper[0]), 6),
                "palette_sample": pal,
                "colormap_count": len(colormaps),
                "pre_import_clean": not (pre_import_scipy_stats or pre_import_scipy_special or pre_import_matplotlib),
                "svg_ok": True,
                "example_data_ok": True,
            })
            safe_report.parent.mkdir(parents=True, exist_ok=True)
            safe_report.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
            window.close()
            app.exit(0)
        except Exception as exc:
            outcome["ok"] = False
            outcome["error"] = traceback.format_exc()
            try:
                safe_err = validate_output_path(report_path)
                safe_err.parent.mkdir(parents=True, exist_ok=True)
                safe_err.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
            except Exception:
                pass
            print(f"Smoke test failed: {exc}", file=sys.stderr)
            window.close()
            app.exit(1)

    QTimer.singleShot(0, execute_checks)
    return int(app.exec())


def main() -> int:
    """Main function parsing arguments and starting event loop."""
    parser = argparse.ArgumentParser(description="SuperADAM NeXus Scan File Explorer")
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Optional path to a .nxs NeXus scan file to open upon launch.",
    )
    parser.add_argument(
        "--smoke-test",
        dest="smoke_test",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.smoke_test:
        return _run_smoke_test(args.smoke_test)

    app = create_app()
    window_cls = MainWindow
    if window_cls is None:
        from pySAview.ui.main_window import MainWindow as window_cls
    window = window_cls()

    # Determine file to open
    target_file: Path | None = None
    if args.file:
        target_file = Path(args.file).resolve()

    if target_file and target_file.exists():
        window.start_load_files([target_file])

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
