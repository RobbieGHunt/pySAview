"""End-to-end UI and integration tests for Option A pySAred 1.5.3 data reduction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from pySAview.core.reduction.overillumination import compute_overillumination_series, extract_slit_widths
from pySAview.tests.test_processing_ui import read_table
from pySAview.ui.main_window import MainWindow

EXAMPLE_5572 = Path(__file__).resolve().parent.parent / "example_data" / "005572.nxs"
EXAMPLE_5594 = Path(__file__).resolve().parent.parent / "example_data" / "005594.nxs"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_file(EXAMPLE_5572)
    yield win
    win.view_offspec.shutdown()


@pytest.fixture
def dialogs(monkeypatch):
    answer = {"path": None, "errors": [], "info": []}
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(answer["path"]), ""))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: answer["errors"].append(a))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: answer["info"].append(a))
    return answer


def _wait_offspec(qtbot, view):
    view._request_compute()
    qtbot.waitUntil(lambda: view._norm is not None and not view._jobs, timeout=60000)


def test_per_scan_reduction_isolation_and_explicit_copy(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    try:
        win.load_file(EXAMPLE_5572)
        win.load_file(EXAMPLE_5594)
        charts = win.view_charts

        # Both scans checked in Loaded Scans so Copy Settings can target #5594
        flist = win.session_files_view.list_files
        for row in range(flist.count()):
            flist.item(row).setCheckState(Qt.CheckState.Checked)

        # Activate 5572 and modify reduction controls
        win.set_active_file(EXAMPLE_5572)
        scan_5572 = win.loaded_scans[str(EXAMPLE_5572)]
        scan_5594 = win.loaded_scans[str(EXAMPLE_5594)]

        charts.chk_overillumination.setChecked(True)
        charts.spin_sample_length.setValue(18.5)
        charts.spin_red_curvature.setValue(0.025)
        charts.spin_theta_offset.setValue(0.015)
        charts.chk_scale_factor.setChecked(True)
        charts.spin_scale_factor.setValue(4.0)

        # 5572 updated; 5594 untouched (never auto-inherited)
        assert scan_5572.reduction_params.overillumination is True
        assert scan_5572.reduction_params.sample_length_mm == pytest.approx(18.5)
        assert scan_5572.reduction_params.curvature_rad == pytest.approx(0.025)
        assert scan_5572.reduction_params.theta_offset_deg == pytest.approx(0.015)
        assert scan_5594.reduction_params.overillumination is False
        assert scan_5594.reduction_params.sample_length_mm == pytest.approx(50.0)
        assert scan_5594.reduction_params.curvature_rad == pytest.approx(0.0)
        assert scan_5594.reduction_params.theta_offset_deg == pytest.approx(0.0)

        # Switching active scan to 5594 updates Reduction tab widgets to 5594's defaults
        win.set_active_file(EXAMPLE_5594)
        assert charts.chk_overillumination.isChecked() is False
        assert charts.spin_sample_length.value() == pytest.approx(50.0)
        assert charts.spin_red_curvature.value() == pytest.approx(0.0)

        # Switch back to 5572 and explicitly copy settings to checked scans
        win.set_active_file(EXAMPLE_5572)
        assert charts.chk_overillumination.isChecked() is True
        assert charts.spin_sample_length.value() == pytest.approx(18.5)
        charts.btn_copy_reduction.click()

        assert scan_5594.reduction_params.overillumination is True
        assert scan_5594.reduction_params.sample_length_mm == pytest.approx(18.5)
        assert scan_5594.reduction_params.curvature_rad == pytest.approx(0.025)
        assert scan_5594.reduction_params.theta_offset_deg == pytest.approx(0.015)
        assert scan_5594.reduction_params is not scan_5572.reduction_params

        # Subsequent edits on 5572 do not mutate the independent copy on 5594
        charts.spin_sample_length.setValue(33.0)
        assert scan_5572.reduction_params.sample_length_mm == pytest.approx(33.0)
        assert scan_5594.reduction_params.sample_length_mm == pytest.approx(18.5)
    finally:
        win.view_offspec.shutdown()


def test_detector_background_roi_links_to_charts_and_offspecular(qtbot, window):
    charts = window.view_charts
    det = window.view_detector
    offspec = window.view_offspec
    scan = window.loaded_scans[str(EXAMPLE_5572)]

    # Switch Scan Charts to ROI_Counts so background subtraction operates on ROI counts
    charts.combo_y.setCurrentIndex(charts.combo_y.findData("ROI_Counts"))
    baseline_curves = [c.y.copy() for c in charts._current_curves]
    assert len(baseline_curves) >= 1

    # Enable Background ROI in 2D Detector
    window.tabs.setCurrentWidget(det)
    det.chk_enable_bkg.setChecked(True)
    assert det._bkg_roi_rect_item is not None and det._bkg_roi_rect_item.isVisible()
    assert scan.reduction_params.subtract_background is True
    assert charts.chk_subtract_bkg.isChecked() is True

    # Width and Y bounds of BKG ROI are locked to Signal ROI
    sig_w = det._roi_rect_item.size().x()
    bkg_x0, bkg_x1 = scan.reduction_params.bkg_x_bounds
    assert (bkg_x1 - bkg_x0) == pytest.approx(sig_w, abs=1.0)
    assert det._bkg_roi_rect_item.pos().y() == pytest.approx(det._roi_rect_item.pos().y())
    assert det._bkg_roi_rect_item.size().y() == pytest.approx(det._roi_rect_item.size().y())

    # Subtracted curves in Scan Charts are <= baseline where background > 0
    subtracted_curves = [c.y.copy() for c in charts._current_curves]
    assert any(np.any(sub < base) for sub, base in zip(subtracted_curves, baseline_curves))

    # Raising skip Qz threshold leaves low-Qz points unsubtracted
    charts.spin_bkg_skip_qz.setValue(0.5)  # higher than entire scan Qz range
    skipped_curves = [c.y.copy() for c in charts._current_curves]
    for sk, base in zip(skipped_curves, baseline_curves):
        np.testing.assert_allclose(sk, base)
    charts.spin_bkg_skip_qz.setValue(0.0)

    # Off-Specular shows dashed cyan BKG ROI overlay when show_roi is enabled
    window.tabs.setCurrentWidget(offspec)
    _wait_offspec(qtbot, offspec)
    offspec.chk_show_roi.setChecked(True)
    # 2 yellow signal lines + 2 cyan BKG lines (+ optional direct-beam fit line)
    assert len(offspec._roi_items) >= 4

    # Overlap warning when moving BKG X1 onto Signal ROI
    window.tabs.setCurrentWidget(det)
    det.spin_bkg_x1.setValue(det.spin_x1.value())
    assert "overlap" in det.lbl_bkg_metrics.text().lower()

    # Disabling BKG ROI restores exact baseline curves
    det.chk_enable_bkg.setChecked(False)
    assert scan.reduction_params.subtract_background is False
    assert charts.chk_subtract_bkg.isChecked() is False
    restored_curves = [c.y.copy() for c in charts._current_curves]
    for rest, base in zip(restored_curves, baseline_curves):
        np.testing.assert_allclose(rest, base)


def test_curvature_sync_and_non_destructive_shearing(qtbot, window):
    charts = window.view_charts
    offspec = window.view_offspec
    scan = window.loaded_scans[str(EXAMPLE_5572)]

    charts.combo_y.setCurrentIndex(charts.combo_y.findData("ROI_Counts"))
    baseline_chart_y = [c.y.copy() for c in charts._current_curves]

    # Compute Off-Specular map in single-spin pixel view
    window.tabs.setCurrentWidget(offspec)
    _wait_offspec(qtbot, offspec)
    offspec.combo_spin.setCurrentIndex(offspec.combo_spin.findData("down"))
    baseline_map = offspec._cut_values.copy()

    # Edit curvature in Off-Specular -> syncs to Scan Charts & shears map
    offspec.spin_curvature.setValue(0.035)
    assert scan.reduction_params.curvature_rad == pytest.approx(0.035)
    assert charts.spin_red_curvature.value() == pytest.approx(0.035)
    sheared_map = offspec._cut_values.copy()
    assert not np.allclose(sheared_map, baseline_map, equal_nan=True)

    # ROI-summed profile cut toggles on cleanly and contains finite positive values
    offspec.chk_roi_sum_cut.setChecked(True)
    assert "ROI-summed" in offspec.lbl_col_cut.text()
    xs, ys = offspec._col_curve.getData()
    assert xs is not None and len(xs) == sheared_map.shape[0]
    assert np.any(np.isfinite(ys) & (ys > 0))

    # Chart ROI counts reflect the sheared ROI integration
    sheared_chart_y = [c.y.copy() for c in charts._current_curves]
    assert any(not np.allclose(s, b) for s, b in zip(sheared_chart_y, baseline_chart_y))

    # Resetting curvature from Scan Charts restores exact uncorrected map & chart curves
    charts.spin_red_curvature.setValue(0.0)
    assert offspec.spin_curvature.value() == pytest.approx(0.0)
    np.testing.assert_array_equal(offspec._cut_values, baseline_map)
    for rest, base in zip([c.y for c in charts._current_curves], baseline_chart_y):
        np.testing.assert_allclose(rest, base)


def test_overillumination_offset_resolution_trimming_and_exports(window, dialogs, tmp_path):
    source_hash_before = _sha256(EXAMPLE_5572)
    charts = window.view_charts
    scan = window.loaded_scans[str(EXAMPLE_5572)]

    # Check File ▸ Export menu includes pySAred .dat export
    titles = [a.text() for a in window.menu_export.actions() if not a.isSeparator()]
    assert any("pySAred" in t and ".dat" in t for t in titles)

    # Set X to Q, Y to ROI_Counts, Monitor normalization
    charts.combo_x.setCurrentIndex(charts.combo_x.findData("Q"))
    charts.combo_y.setCurrentIndex(charts.combo_y.findData("ROI_Counts"))
    charts.combo_norm.setCurrentIndex(charts.combo_norm.findData("monitor"))

    base_curves = [(c.x.copy(), c.y.copy(), c.y_err.copy(), c.indices.copy()) for c in charts._current_curves]
    assert len(base_curves) == 2  # Spin Down + Spin Up

    # Enable overillumination (L = 10 mm so beam overilluminates at small angles) + c0 overlay
    charts.spin_sample_length.setValue(10.0)
    charts.chk_overillumination.setChecked(True)
    charts.chk_show_overill_curve.setChecked(True)

    # Verify c0 division matches core/overillumination directly
    s1_mm, s2_mm, _ = extract_slit_widths(scan, len(scan.variables["th"].data))
    c0_full, _, _ = compute_overillumination_series(
        s1_mm, s2_mm, scan.variables["th"].data, sample_length_mm=10.0
    )
    for (bx, by, berr, bidx), c in zip(base_curves, charts._current_curves):
        np.testing.assert_allclose(c.y, by / c0_full[bidx], rtol=1e-12)
        np.testing.assert_allclose(c.y_err, berr / c0_full[bidx], rtol=1e-12)

    # Legend contains Overillumination c₀ overlay
    legend_labels = [lbl.text for _, lbl in charts.legend.items]
    assert any("Overillumination" in txt for txt in legend_labels)

    # Apply theta offset -> Q shifts downward
    charts.spin_theta_offset.setValue(0.05)
    for (bx, _, _, _), c in zip(base_curves, charts._current_curves):
        assert np.all(c.x < bx)

    # Resolution column present in Plotted Data Points table (column 9)
    assert charts.curve_data_table.columnCount() == 10
    assert "Resolution" in charts.curve_data_table.horizontalHeaderItem(9).text()
    first_res = float(charts.curve_data_table.item(0, 9).text())
    assert np.isfinite(first_res) and first_res > 0

    # Trimming: skip_left=2, skip_right=3
    n_before = [len(c.x) for c in charts._current_curves]
    charts.spin_skip_left.setValue(2)
    charts.spin_skip_right.setValue(3)
    n_after = [len(c.x) for c in charts._current_curves]
    for before, after in zip(n_before, n_after):
        assert after == before - 5

    # Export tab-delimited table -> includes ResolutionSigma and reduction summary
    txt_target = tmp_path / "reduced_table.txt"
    dialogs["path"] = txt_target
    window.act_export_refl_data.trigger()
    header, rows = read_table(txt_target)
    assert any(k.startswith("ResolutionSigma") for k in rows[0])
    assert any("overillumination(L=10 mm" in line for line in header)
    assert any("th_offset=0.05 deg" in line for line in header)

    # Export pySAred .dat (4 columns: Qz  I  dI  dQz) via File ▸ Export menu
    dat_target = tmp_path / "reduced_qz.dat"
    dialogs["path"] = dat_target
    charts.combo_dat_angle.setCurrentIndex(charts.combo_dat_angle.findData("Qz"))
    window.act_export_pysared_dat.trigger()
    assert not dialogs["errors"]
    dat_files = sorted(tmp_path.glob("reduced_qz_*.dat"))
    assert len(dat_files) == 2  # one file per spin state
    for df in dat_files:
        lines = [ln for ln in df.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) > 0
        for ln in lines:
            assert not ln.startswith("#")  # pySAred .dat has no comment header
            cols = ln.split()
            assert len(cols) == 4
            vals = [float(v) for v in cols]
            assert all(np.isfinite(vals))
            assert vals[1] >= 0.0  # pySAred excludes negatives

    # Export 3-column .dat when include_resolution is unchecked
    charts.chk_include_resolution.setChecked(False)
    dat_target_3 = tmp_path / "reduced_deg.dat"
    dialogs["path"] = dat_target_3
    charts.combo_dat_angle.setCurrentIndex(charts.combo_dat_angle.findData("Degrees"))
    charts.export_pysared_dat()
    dat_files_3 = sorted(tmp_path.glob("reduced_deg_*.dat"))
    assert len(dat_files_3) == 2
    for df in dat_files_3:
        first_line = df.read_text(encoding="utf-8").splitlines()[0]
        assert len(first_line.split()) == 3

    # Source-safety guard: refusing to overwrite loaded .nxs file
    dialogs["errors"].clear()
    dialogs["path"] = EXAMPLE_5572
    charts.export_pysared_dat()
    assert len(dialogs["errors"]) >= 1

    # Source NeXus file is strictly untouched byte-for-byte
    assert _sha256(EXAMPLE_5572) == source_hash_before
