"""Off-specular map: Y integration, frame selection, normalization and the UI tab."""

import csv
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from pySAview.core.off_specular import (
    DETECTOR_PATH,
    compute_off_specular_map,
    integrate_frames_along_y,
    log_display,
    normalize_map,
    select_frames,
    y_pixel_bounds,
)
from pySAview.ui.main_window import MainWindow

EXAMPLE = Path(__file__).resolve().parent.parent / "example_data" / "005572.nxs"


def _raw(n=3, nx=5, ny=4):
    return np.arange(n * nx * ny, dtype=np.uint32).reshape(n, nx, ny)


def test_integration_matches_displayed_frame_columns():
    raw = _raw()
    result = integrate_frames_along_y(raw, (1, 3))
    for i in range(raw.shape[0]):
        displayed = raw[i].T  # rows = Y, columns = X
        np.testing.assert_array_equal(result[i], displayed[1:3, :].sum(axis=0))
    assert result.shape == (3, 5)


def test_full_height_and_shared_pixel_convention():
    raw = _raw()
    np.testing.assert_array_equal(integrate_frames_along_y(raw, None), raw.sum(axis=2))
    # Nearest-integer bounds with ties to even, reversed endpoints, clipping.
    assert y_pixel_bounds((2.5, 0.5), 5, 4) == (0, 2)
    assert y_pixel_bounds((-10, 99), 5, 4) == (0, 4)
    assert np.all(integrate_frames_along_y(raw, (10, 20)) == 0)


def test_chunked_dataset_matches_in_memory():
    raw = np.random.default_rng(1).poisson(3, size=(37, 8, 6)).astype(np.uint32)
    calls = []
    result = compute_off_specular_map(raw, (1, 5), progress=lambda d, t: calls.append((d, t)))
    np.testing.assert_array_equal(result.counts, integrate_frames_along_y(raw, (1, 5)))
    assert result.y_bounds == (1, 5)
    assert calls[-1] == (37, 37)
    with pytest.raises(InterruptedError):
        compute_off_specular_map(raw, None, cancelled=lambda: True)


def _write_detector(path, frames, written, **storage):
    with h5py.File(path, "w") as handle:
        ds = handle.create_dataset(DETECTOR_PATH, shape=frames.shape, dtype=frames.dtype,
                                   chunks=(1, *frames.shape[1:]), fillvalue=0, **storage)
        for i in written:  # leave the rest unallocated, like an aborted scan
            ds[i] = frames[i]


@pytest.mark.parametrize("windows", [None, (2, 9), [(0, 3), (1, 12), (5, 5), (4, 10), (0, 1), (7, 12)]])
def test_parallel_gzip_path_matches_h5py_reads(tmp_path, windows):
    frames = np.random.default_rng(3).poisson(4, size=(6, 10, 12)).astype(np.uint32)
    path = tmp_path / "gzip.h5"
    _write_detector(path, frames, written=[0, 1, 3, 5], compression="gzip", compression_opts=6)
    with h5py.File(path, "r") as handle:
        ds = handle[DETECTOR_PATH]
        from pySAview.core.off_specular import _supports_parallel_gzip
        assert _supports_parallel_gzip(ds)
        calls = []
        fast = compute_off_specular_map(ds, windows, progress=lambda d, t: calls.append(d))
        slow = compute_off_specular_map(ds, windows, parallel=False)
        np.testing.assert_array_equal(fast.counts, slow.counts)
        expected = frames.copy()
        expected[[2, 4]] = 0  # unwritten chunks read as the fill value
        np.testing.assert_array_equal(fast.counts, compute_off_specular_map(expected, windows).counts)
        assert calls == list(range(1, 7))
        with pytest.raises(InterruptedError):
            compute_off_specular_map(ds, windows, cancelled=lambda: True)


@pytest.mark.parametrize("storage", [dict(compression="gzip", shuffle=True), dict(compression="lzf"), dict()])
def test_other_storage_layouts_use_h5py_path(tmp_path, storage):
    frames = np.random.default_rng(4).poisson(4, size=(3, 8, 8)).astype(np.uint16)
    path = tmp_path / "other.h5"
    _write_detector(path, frames, written=range(3), **storage)
    from pySAview.core.off_specular import _supports_parallel_gzip
    with h5py.File(path, "r") as handle:
        assert not _supports_parallel_gzip(handle[DETECTOR_PATH])
        np.testing.assert_array_equal(compute_off_specular_map(handle[DETECTOR_PATH], (1, 6)).counts,
                                      integrate_frames_along_y(frames, (1, 6)))


def test_example_file_parallel_matches_plain_reads():
    with h5py.File(EXAMPLE, "r") as handle:
        ds = handle[DETECTOR_PATH]
        np.testing.assert_array_equal(compute_off_specular_map(ds, (300, 700)).counts,
                                      compute_off_specular_map(ds, (300, 700), parallel=False).counts)


def test_select_frames_by_spin():
    flipper = np.array([0, 1, 0, 1, 1])
    np.testing.assert_array_equal(select_frames(5, flipper, "up"), [1, 3, 4])
    np.testing.assert_array_equal(select_frames(5, flipper, "down"), [0, 2])
    np.testing.assert_array_equal(select_frames(5, None, "up"), np.arange(5))


def test_normalization_follows_documented_formulas():
    counts = np.array([[4.0, 0.0], [9.0, 16.0], [1.0, 1.0]])
    rows = np.array([0, 1, 2])
    raw = normalize_map(counts, rows, "raw")
    np.testing.assert_allclose(raw.standard_uncertainty, np.sqrt(counts))

    t = np.array([2.0, 4.0, 0.0])
    timed = normalize_map(counts, rows, "time", exposure=t)
    np.testing.assert_allclose(timed.values[:2], counts[:2] / t[:2, None])
    np.testing.assert_allclose(timed.standard_uncertainty[:2] ** 2, counts[:2] / t[:2, None] ** 2)
    assert np.all(np.isnan(timed.values[2])) and timed.warnings

    m = np.array([100.0, 50.0, 10.0])
    mon = normalize_map(counts, rows, "monitor", monitor=m)
    expected_var = counts / m[:, None] ** 2 + counts**2 * m[:, None] / m[:, None] ** 4
    np.testing.assert_allclose(mon.values, counts / m[:, None])
    np.testing.assert_allclose(mon.standard_uncertainty ** 2, expected_var)
    assert mon.standard_uncertainty[0, 1] == 0.0  # N = 0 plug-in estimate, not invented


def test_monitor_variance_is_used_when_supplied():
    counts = np.array([[4.0], [9.0]])
    m = np.array([100.0, 50.0])
    v_m = np.array([400.0, -1.0])  # second variance invalid -> excluded, not guessed
    mon = normalize_map(counts, np.array([0, 1]), "monitor", monitor=m, monitor_variance=v_m)
    assert mon.standard_uncertainty[0, 0] ** 2 == pytest.approx(4 / 100**2 + 16 * 400 / 100**4)
    assert np.isnan(mon.values[1, 0]) and mon.warnings


def test_per_frame_y_windows_for_tracked_roi():
    raw = _raw(n=3, nx=5, ny=6)
    windows = [(0, 2), (1, 4), (3, 6)]
    result = compute_off_specular_map(raw, windows)
    for i, (a, b) in enumerate(windows):
        np.testing.assert_array_equal(result.counts[i], raw[i].T[a:b, :].sum(axis=0))
    assert result.y_varies and result.y_bounds == (0, 6)
    with pytest.raises(ValueError):
        compute_off_specular_map(raw, windows[:2])


def test_log_display_is_display_only():
    values = np.array([[0.0, 10.0], [100.0, np.nan]])
    before = values.copy()
    shown = log_display(values)
    np.testing.assert_array_equal(values, before)
    assert np.isnan(shown[0, 0])  # zero omitted from log rendering, never replaced
    assert shown[0, 1] == 1.0 and shown[1, 0] == 2.0 and np.isnan(shown[1, 1])


def test_example_file_map_matches_direct_sum_and_source_unchanged():
    digest = hashlib.sha256(EXAMPLE.read_bytes()).hexdigest()
    with h5py.File(EXAMPLE, "r") as handle:
        ds = handle[DETECTOR_PATH]
        result = compute_off_specular_map(ds, (285, 743))
        for frame in (0, 150, ds.shape[0] - 1):
            displayed = np.asarray(ds[frame], dtype=float).T
            np.testing.assert_array_equal(result.counts[frame], displayed[285:743, :].sum(axis=0))
    assert result.counts.shape == (302, 1024)
    assert hashlib.sha256(EXAMPLE.read_bytes()).hexdigest() == digest


@pytest.fixture
def window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_file(EXAMPLE)
    yield win
    win.view_offspec.shutdown()


def _wait_for_map(qtbot, view):
    qtbot.waitUntil(lambda: view._norm is not None and not view._jobs, timeout=60000)


def test_tab_computes_map_and_cuts(qtbot, window, tmp_path):
    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    view._request_compute()  # window is not shown in tests; showEvent normally triggers this
    _wait_for_map(qtbot, view)
    scan = window.loaded_scans[str(EXAMPLE.resolve())]
    # Full detector height is the default; the row spinboxes show it.
    assert view.combo_y_source.currentData() == "full"
    assert view._map.y_bounds == (0, 1024)
    assert (view.spin_y0.value(), view.spin_y1.value()) == (0, 1024)
    assert view.spin_y0.isEnabled() and view.spin_y1.isEnabled()
    flipper = scan.variables["Flipper1"].data
    assert view.combo_spin.currentData() == "up"
    assert view.combo_y_axis.currentData() == "th"
    assert view._norm.values.shape == (int(np.sum(flipper == 1)), 1024)
    assert view._th_geom is not None and len(view._alpha_items) == 1  # exact-cell PColorMesh in θ
    assert view.img_item.image is None
    assert len(view._roi_items) == 3  # two ROI edges + the fitted direct-beam line

    # Switching to Point mode uses the uniform-row ImageItem.
    view.combo_y_axis.setCurrentIndex(view.combo_y_axis.findData("point"))
    assert view._th_geom is None and view.img_item.image is not None
    view.combo_y_axis.setCurrentIndex(view.combo_y_axis.findData("th"))

    view.combo_y_source.setCurrentIndex(view.combo_y_source.findData("roi"))
    _, _, y0, y1 = scan.detector_roi
    expected = y_pixel_bounds((y0, y1), 1024, 1024)
    assert (view.spin_y0.value(), view.spin_y1.value()) == expected
    qtbot.waitUntil(lambda: not view._jobs and view._map.y_bounds == expected, timeout=60000)

    assert not hasattr(view, "write_csv") and not hasattr(view, "btn_export_csv")
    # Compact readout: θ and 2θ at the scan's resolution, value without uncertainty.
    row, x = 40, 370
    text = view._format_value(row, x)
    assert "±" not in text and text.endswith("counts")
    assert view._angle_decimals("th") == 2 and view._angle_decimals("tth") == 2
    assert view._describe_frame(int(view._norm.frame_indices[row])).count("°") == 2
    for btn in (view.btn_pan, view.btn_box_zoom, view.btn_crosshair, view.btn_zoom_in,
                view.btn_zoom_out, view.btn_fit, view.btn_export_png):
        assert btn.text() == "" and btn.toolTip()


def test_missing_normalization_channel_has_no_raw_fallback(qtbot, window):
    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    view._request_compute()  # window is not shown in tests; showEvent normally triggers this
    _wait_for_map(qtbot, view)
    scan = window.loaded_scans[str(EXAMPLE.resolve())]
    monitor = scan.variables.pop("Monitor1")
    try:
        view.combo_norm.setCurrentIndex(view.combo_norm.findData("monitor"))
        assert view._norm is None
        assert view.img_item.image is None
        assert "Monitor1" in view.lbl_notice.text()
    finally:
        scan.variables["Monitor1"] = monitor
    view.combo_norm.setCurrentIndex(view.combo_norm.findData("raw"))
    assert view._norm is not None


def test_stale_result_after_cancel_is_ignored(qtbot, window):
    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    view._request_compute()  # window is not shown in tests; showEvent normally triggers this
    _wait_for_map(qtbot, view)
    old_token = view._token
    view._cancel_jobs()  # e.g. a new scan became active
    current = view._map
    stale = compute_off_specular_map(np.zeros((302, 1024, 2), dtype=np.uint32), None)
    view._on_computed(old_token, stale)
    assert view._map is current


def test_row_spinboxes_switch_to_custom(qtbot, window):
    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    view._request_compute()
    _wait_for_map(qtbot, view)
    view.spin_y0.setValue(100)
    assert view.combo_y_source.currentData() == "custom"
    view.spin_y1.setValue(900)
    qtbot.waitUntil(lambda: not view._jobs and view._map.y_bounds == (100, 900), timeout=60000)
    view.combo_y_source.setCurrentIndex(view.combo_y_source.findData("full"))
    assert (view.spin_y0.value(), view.spin_y1.value()) == (0, 1024)


def test_crosshair_drop_and_drag_updates_cuts_without_rescaling(qtbot, window):
    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    view._request_compute()
    _wait_for_map(qtbot, view)
    assert not view.crosshair.isVisible() and not view._row_curve.isVisible()

    # Crosshair in the default X pixel vs θ view maps real θ coordinates to measured cells.
    view.btn_crosshair.setChecked(True)
    th_40 = float(view._th_geom["theta"][40])
    view._place_crosshair(370.5, th_40)
    assert view.crosshair.isVisible() and view.line_x.isVisible()
    assert (view._sel_row, view._sel_x) == (40, 370)
    np.testing.assert_allclose(view._col_curve.xData, view._th_geom["theta"])

    # Switching to Point mode preserves the selected cell and drags in row coordinates.
    view.combo_y_axis.setCurrentIndex(view.combo_y_axis.findData("point"))
    assert (view._sel_row, view._sel_x) == (40, 370)
    view._place_crosshair(370.5, 40.2)  # what a click on the Point map does
    assert view.crosshair.isVisible() and view.line_x.isVisible()
    assert (view._sel_row, view._sel_x) == (40, 370)
    expected = np.where(view._norm.values[40] > 0, view._norm.values[40], np.nan)  # log: zeros omitted
    np.testing.assert_array_equal(view._row_curve.yData, expected)
    y_range = view.plot_row.getViewBox().viewRange()[1]

    view.crosshair.setPos((600.7, 100.1))  # dragging the centre marker
    assert (view._sel_row, view._sel_x) == (100, 600)
    assert view.line_x.value() == pytest.approx(600.7) and view.line_row.value() == pytest.approx(100.1)
    assert view.plot_row.getViewBox().viewRange()[1] == y_range  # cut axes do not jump
    assert "point" in view.lbl_row_cut.text() and "600" in view.lbl_col_cut.text()

    view.crosshair.setPos((5000, -20))  # clamped onto the map
    assert view._sel_x == 1023 and view._sel_row == 0

    view.btn_crosshair.setChecked(False)
    assert not view.crosshair.isVisible() and not view._row_curve.isVisible()


def test_status_text_does_not_resize_map(qtbot, window):
    view = window.view_offspec
    window.resize(1500, 900)
    window.tabs.setCurrentWidget(view)
    view._request_compute()
    _wait_for_map(qtbot, view)
    window.show()
    qtbot.waitExposed(window)
    widths = []
    for text in ("", "X: 512  |  Point 150 (θ = 1.2488°, 2θ = 2.5002°, u)  |  1234.56 ± 35.1 counts (1σ)" * 2, ""):
        view.lbl_coords.setText(text)
        view.lbl_row_cut.setText(text or "short")
        qtbot.wait(20)
        widths.append((view.plot.width(), view.plot_row.width()))
    window.hide()
    assert len(set(widths)) == 1


def test_tab_follows_working_roi(qtbot, window):
    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    view.combo_y_source.setCurrentIndex(view.combo_y_source.findData("roi"))
    view._request_compute()  # window is not shown in tests; showEvent normally triggers this
    _wait_for_map(qtbot, view)
    det = window.view_detector
    det.spin_y0.setValue(400)
    det.spin_y1.setValue(600)
    det._apply_roi_to_scan_chart()
    assert view._dirty  # recomputed on next show
    view._request_compute()
    qtbot.waitUntil(lambda: view._map is not None and view._map.y_bounds == (400, 600), timeout=60000)
    scan = window.loaded_scans[str(EXAMPLE.resolve())]
    assert scan.detector_roi != scan.working_detector_roi  # file ROI preserved


def test_pair_spin_frames_matches_angles_and_skips_unwritten():
    from pySAview.core.off_specular import pair_spin_frames
    th = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.35, 0.4, 0.4])
    tth = 2 * th
    flipper = np.array([1, 0, 0, 1, 1, 0, 1, 0])
    written = np.array([True] * 7 + [False])
    up, down = pair_spin_frames(th, tth, flipper, written)
    np.testing.assert_array_equal(up, [0, 3])  # 0.3/0.35 differ; frame 7 unwritten
    np.testing.assert_array_equal(down, [1, 2])


def test_spin_asymmetry_values_uncertainty_and_display_mask():
    from pySAview.core.off_specular import spin_asymmetry
    counts = np.array([[30.0, 4.0, 0.0], [10.0, 4.0, 0.0]])  # row 0 = u, row 1 = d
    sa = spin_asymmetry(counts, [0], [1], "raw", min_counts=10)
    np.testing.assert_allclose(sa.values[0, :2], [0.5, 0.0])
    assert np.isnan(sa.values[0, 2])  # u + d = 0: undefined, not zero
    # var = (2 I_d / S²)² I_u + (2 I_u / S²)² I_d for Poisson counts
    expected = np.sqrt((2 * 10 / 40**2) ** 2 * 30 + (2 * 30 / 40**2) ** 2 * 10)
    assert sa.standard_uncertainty[0, 0] == pytest.approx(expected)
    np.testing.assert_array_equal(sa.low_count_mask[0], [False, True, True])
    assert sa.values[0, 1] == 0.0  # masking is display-only

    # Monitor normalization removes an exposure imbalance between u and d.
    counts = np.array([[20.0], [10.0]])
    monitor = np.array([200.0, 100.0])
    assert spin_asymmetry(counts, [0], [1], "monitor", monitor=monitor).values[0, 0] == pytest.approx(0.0)
    assert spin_asymmetry(counts, [0], [1], "raw").values[0, 0] == pytest.approx(1 / 3)


def test_asymmetry_mode_in_tab(qtbot, window, tmp_path):
    from pySAview.core.off_specular import AsymmetryMap
    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    view._request_compute()
    _wait_for_map(qtbot, view)
    view.combo_norm.setCurrentIndex(view.combo_norm.findData("monitor"))
    view.combo_spin.setCurrentIndex(view.combo_spin.findData("asym"))
    norm = view._norm
    assert isinstance(norm, AsymmetryMap) and norm.values.shape == (151, 1024)
    assert not view.chk_log.isEnabled() and not view.spin_min_counts.isHidden()
    assert np.nanmax(np.abs(norm.values)) <= 1.0 + 1e-12
    shown = view._cut_values
    assert np.all(np.isnan(shown[norm.low_count_mask]))
    lo, hi = view.colorbar.levels()
    assert lo == pytest.approx(-hi)  # symmetric about zero asymmetry

    np.testing.assert_array_equal(norm.partner_indices, norm.frame_indices + 1)  # interleaved u/d
    assert view._format_value(0, 370).startswith("SA ")

    view.combo_y_axis.setCurrentIndex(view.combo_y_axis.findData("alpha"))
    assert view._alpha_geom is not None and len(view._alpha_items) == 151
    view.chk_show_db.setChecked(False)
    assert not view._roi_items  # unticked: no ROI or reference lines on the α map
    view.chk_show_db.setChecked(True)
    assert len(view._roi_items) == 4  # two reference lines + two ROI edges

    view.combo_spin.setCurrentIndex(view.combo_spin.findData("up"))
    assert view.chk_log.isEnabled() and view.spin_min_counts.isHidden()


def test_theta_view_uses_true_angle_mesh_without_step_size_kink(qtbot, window):
    """Example scan 005572 changes Δθ from 0.01° to 0.03° to 0.04°; the θ mesh must place
    each frame at its actual θ so linear features in angle have no artificial slope kink."""
    from pySAview.ui.off_specular_view import CachedMeshItem

    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    view._request_compute()
    _wait_for_map(qtbot, view)
    assert view.combo_y_axis.currentData() == "th"
    assert len(view._alpha_items) == 1 and isinstance(view._alpha_items[0], CachedMeshItem)

    g = view._th_geom
    assert g is not None
    # Step widths in the mesh follow actual Δθ (~0.01° early, ~0.03° mid, ~0.04° late).
    widths = g["strips"][:, 1] - g["strips"][:, 0]
    assert widths[10] == pytest.approx(0.01, abs=1e-3)
    assert widths[70] == pytest.approx(0.03, abs=1e-3)
    assert widths[130] == pytest.approx(0.04, abs=1e-3)

    # Every cell centre round-trips back to its exact (row, x) across all step regimes.
    for row, x in ((10, 370), (50, 500), (70, 620), (130, 800)):
        assert view._cell_at(*view._cell_center(row, x)) == (row, x)

    # Direct-beam overlay (first _roi_items curve) is linear in θ across step transitions.
    db_line = view._roi_items[0]
    slopes = np.diff(db_line.xData) / np.diff(db_line.yData)
    cal = window.loaded_scans[str(EXAMPLE.resolve())].angle_calibration
    # Since tth ≈ 2 * th in a specular scan, dx / dθ ≈ 2 * slope_px_per_deg everywhere.
    np.testing.assert_allclose(slopes, 2.0 * cal.fit.slope_px_per_deg, rtol=0.05)


def test_sum_spin_frames_pooled_exposure_and_uncertainty():
    from pySAview.core.off_specular import SummedSpinMap, sum_spin_frames

    counts = np.array([[30.0, 0.0], [10.0, 0.0], [5.0, 2.0], [7.0, 3.0]])  # (0,1) and (2,3) pairs
    raw = sum_spin_frames(counts, [0, 2], [1, 3], "raw")
    assert isinstance(raw, SummedSpinMap)
    np.testing.assert_allclose(raw.values, [[40.0, 0.0], [12.0, 5.0]])
    np.testing.assert_allclose(raw.standard_uncertainty, np.sqrt([[40.0, 0.0], [12.0, 5.0]]))

    # Pooled time normalization: (N_u + N_d) / (t_u + t_d); invalid partner exposure -> NaN + warning.
    timed = sum_spin_frames(counts, [0, 2], [1, 3], "time", exposure=[2.0, 6.0, 3.0, 0.0])
    np.testing.assert_allclose(timed.values[0], [40.0 / 8.0, 0.0])
    np.testing.assert_allclose(timed.standard_uncertainty[0], [np.sqrt(40.0) / 8.0, 0.0])
    assert np.all(np.isnan(timed.values[1])) and timed.warnings

    # Pooled monitor normalization: (N_u + N_d) / (M_u + M_d) with Poisson monitor variance M_u + M_d.
    mon = sum_spin_frames(counts[:2], [0], [1], "monitor", monitor=[200.0, 100.0])
    assert mon.values[0, 0] == pytest.approx(40.0 / 300.0)
    expected_var = 40.0 / 300.0**2 + 40.0**2 * 300.0 / 300.0**4
    assert mon.standard_uncertainty[0, 0] == pytest.approx(np.sqrt(expected_var))
    assert mon.standard_uncertainty[0, 1] == 0.0

    # Supplied monitor variances add in quadrature: v_M = v_Mu + v_Md.
    mon_v = sum_spin_frames(counts[:2], [0], [1], "monitor", monitor=[200.0, 100.0], monitor_variance=[250.0, 150.0])
    expected_var_custom = 40.0 / 300.0**2 + 40.0**2 * 400.0 / 300.0**4
    assert mon_v.standard_uncertainty[0, 0] == pytest.approx(np.sqrt(expected_var_custom))


def test_sum_spin_mode_in_tab_and_export(qtbot, window):
    from pySAview.core.off_specular import SummedSpinMap

    view = window.view_offspec
    window.tabs.setCurrentWidget(view)
    view._request_compute()
    _wait_for_map(qtbot, view)

    view.combo_norm.setCurrentIndex(view.combo_norm.findData("monitor"))
    view.combo_spin.setCurrentIndex(view.combo_spin.findData("sum"))
    norm = view._norm
    assert isinstance(norm, SummedSpinMap) and norm.values.shape == (151, 1024)
    assert view.chk_log.isEnabled() and view.combo_cmap.isEnabled() and view.spin_min_counts.isHidden()

    # Check pooled monitor value against raw map counts and Monitor1.
    scan = window.loaded_scans[str(EXAMPLE.resolve())]
    mon = scan.variables["Monitor1"].data
    u0, d0 = int(norm.frame_indices[0]), int(norm.partner_indices[0])
    expected = (view._map.counts[u0] + view._map.counts[d0]) / (mon[u0] + mon[d0])
    np.testing.assert_allclose(norm.values[0], expected)

    # Works in α_i/α_f and Qx/Qz views and exports PartnerFrame.
    for axes in ("alpha", "q", "th"):
        view.combo_y_axis.setCurrentIndex(view.combo_y_axis.findData(axes))
        assert view._map_ready()
    header, columns, rows = view.export_table()
    assert columns[:3] == ["Frame", "PartnerFrame", "XPixel"]
    first = next(rows)
    assert first[0] == u0 and first[1] == d0
    assert any("pooled u + d" in line for line in header)


def _grab_rgba(widget) -> np.ndarray:
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication

    QApplication.processEvents()
    img = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    return np.frombuffer(ptr, dtype=np.uint8).reshape(img.height(), img.width(), 4).copy()


def _data_box_slice(plot_widget, frame_shape):
    """Pixel slice strictly inside the plot's ViewBox (excluding axis ticks/borders)."""
    vb_rect = plot_widget.mapFromScene(plot_widget.getViewBox().sceneBoundingRect()).boundingRect()
    h, w = frame_shape[:2]
    pad_x = max(8, int(vb_rect.width() * 0.1))
    pad_y = max(8, int(vb_rect.height() * 0.1))
    y0 = max(0, vb_rect.top() + pad_y)
    y1 = min(h, vb_rect.bottom() - pad_y)
    x0 = max(0, vb_rect.left() + pad_x)
    x1 = min(w, vb_rect.right() - pad_x)
    return slice(y0, y1), slice(x0, x1)


def test_cached_mesh_no_flip_during_pan_zoom_or_inverted_axes(qtbot):
    """Panning or zooming CachedMeshItem must never vertically or horizontally flip the
    cached image before the detail timer fires (including with inverted X/Y viewboxes)."""
    import pyqtgraph as pg
    from pySAview.ui.off_specular_view import CachedMeshItem
    from pySAview.ui.theme import get_safe_colormap

    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.resize(420, 320)
    # Strongly asymmetric 2D field: bright (high) at top (+y) and right (+x), dark (low) at
    # bottom (-y) and left (-x), with a NaN gap band in the middle to catch overview bleed.
    x, y = np.meshgrid(np.linspace(0.0, 10.0, 41), np.linspace(0.0, 10.0, 41))
    yy = np.arange(40, dtype=float)[:, None]
    xx = np.arange(40, dtype=float)[None, :]
    z = yy * 2.0 + xx * 0.8
    z[19:21, :] = np.nan
    mesh = CachedMeshItem(x, y, z, enableAutoLevels=False)
    mesh.setColorMap(get_safe_colormap("gray"))
    mesh.setLevels((0.0, 110.0))
    widget.addItem(mesh)
    widget.getViewBox().setRange(xRange=(0.0, 10.0), yRange=(0.0, 10.0), padding=0.0)
    widget.show()
    qtbot.waitExposed(widget)
    qtbot.waitUntil(lambda: mesh._detail is not None, timeout=5000)

    vb = widget.getViewBox()

    def assert_no_flip(move_fn, *, expect_top_brighter: bool, expect_right_brighter: bool):
        prev_detail = mesh._detail
        move_fn()
        # Capture immediately BEFORE the 120 ms detail timer sharpens the view.
        during = _grab_rgba(widget)
        assert mesh._detail is prev_detail
        qtbot.waitUntil(lambda: mesh._detail is not prev_detail and mesh._detail is not None, timeout=5000)
        after = _grab_rgba(widget)

        ys, xs = _data_box_slice(widget, after.shape)
        d_box = during[ys, xs, :3].astype(float)
        a_box = after[ys, xs, :3].astype(float)
        h, w = d_box.shape[:2]
        # Orientation check DURING pan/zoom (before detail timer):
        top_luma = d_box[: h // 3, :, :].mean()
        bot_luma = d_box[2 * h // 3 :, :, :].mean()
        left_luma = d_box[:, : w // 3, :].mean()
        right_luma = d_box[:, 2 * w // 3 :, :].mean()
        if expect_top_brighter:
            assert top_luma > bot_luma + 15.0
        else:
            assert bot_luma > top_luma + 15.0
        if expect_right_brighter:
            assert right_luma > left_luma + 5.0
        else:
            assert left_luma > right_luma + 5.0
        # Rendered image during motion must closely match the settled detail image (no flip!).
        diff = np.mean(np.abs(d_box - a_box))
        diff_vflip = np.mean(np.abs(d_box[::-1, :, :] - a_box))
        assert diff < 8.0 and diff < diff_vflip * 0.25

    # 1. Small pan (inside the 25% detail margin)
    assert_no_flip(
        lambda: vb.setRange(xRange=(0.4, 10.4), yRange=(0.4, 10.4), padding=0.0),
        expect_top_brighter=True,
        expect_right_brighter=True,
    )
    # 2. Zoom in
    assert_no_flip(
        lambda: vb.setRange(xRange=(2.0, 8.0), yRange=(2.0, 8.0), padding=0.0),
        expect_top_brighter=True,
        expect_right_brighter=True,
    )
    # 3. Zoom out + large pan (exercises _overview outside detail_view_rect alongside _detail)
    assert_no_flip(
        lambda: vb.setRange(xRange=(-1.0, 11.0), yRange=(-1.0, 11.0), padding=0.0),
        expect_top_brighter=True,
        expect_right_brighter=True,
    )
    # 4. Inverted X axis + pan
    vb.invertX(True)
    qtbot.waitUntil(lambda: not mesh._detail_timer.isActive(), timeout=5000)
    _grab_rgba(widget)
    qtbot.waitUntil(lambda: not mesh._detail_timer.isActive(), timeout=5000)
    assert_no_flip(
        lambda: vb.setRange(xRange=(0.5, 9.5), yRange=(0.5, 9.5), padding=0.0),
        expect_top_brighter=True,
        expect_right_brighter=False,
    )
    # 5. Inverted Y axis + pan
    vb.invertX(False)
    vb.invertY(True)
    _grab_rgba(widget)
    qtbot.waitUntil(lambda: not mesh._detail_timer.isActive(), timeout=5000)
    assert_no_flip(
        lambda: vb.setRange(xRange=(0.2, 9.8), yRange=(0.2, 9.8), padding=0.0),
        expect_top_brighter=False,
        expect_right_brighter=True,
    )


@pytest.mark.parametrize("axis_mode", ["th", "point", "alpha", "q"])
def test_offspec_map_all_modes_stay_upright_during_pan_and_zoom(qtbot, window, axis_mode):
    """Every 2D map representation keeps exact orientation while panning and zooming."""
    from pySAview.ui.off_specular_view import CachedMeshItem

    view = window.view_offspec
    window.resize(1100, 750)
    window.tabs.setCurrentWidget(view)
    view._request_compute()
    _wait_for_map(qtbot, view)
    window.show()
    qtbot.waitExposed(window)

    view.combo_y_axis.setCurrentIndex(view.combo_y_axis.findData(axis_mode))
    _grab_rgba(view.plot)
    mesh = view._alpha_items[0] if (view._alpha_items and isinstance(view._alpha_items[0], CachedMeshItem)) else None
    if mesh is not None:
        qtbot.waitUntil(lambda: mesh._detail is not None and not mesh._detail_timer.isActive(), timeout=5000)

    vb = view.plot.getViewBox()
    xr, yr = vb.viewRange()
    dx = 0.04 * (xr[1] - xr[0])
    dy = 0.04 * (yr[1] - yr[0])
    prev_detail = mesh._detail if mesh is not None else None
    vb.setRange(xRange=(xr[0] + dx, xr[1] + dx), yRange=(yr[0] + dy, yr[1] + dy), padding=0.0)
    during = _grab_rgba(view.plot)
    if mesh is not None:
        qtbot.waitUntil(lambda: mesh._detail is not prev_detail and mesh._detail is not None, timeout=5000)
    after = _grab_rgba(view.plot)

    ys, xs = _data_box_slice(view.plot, after.shape)
    d_box = during[ys, xs, :3].astype(float)
    a_box = after[ys, xs, :3].astype(float)
    # 8x8 block averaging removes sub-pixel rasterization jitter on sparse 1-px Poisson speckles
    # while staying sharply sensitive to any vertical/horizontal flip or displacement.
    bh, bw = (d_box.shape[0] // 8) * 8, (d_box.shape[1] // 8) * 8
    d_blk = d_box[:bh, :bw].reshape(bh // 8, 8, bw // 8, 8, 3).mean(axis=(1, 3))
    a_blk = a_box[:bh, :bw].reshape(bh // 8, 8, bw // 8, 8, 3).mean(axis=(1, 3))
    assert np.mean(np.abs(d_blk - a_blk)) < 6.0
    corr = float(np.corrcoef(d_blk.ravel(), a_blk.ravel())[0, 1])
    corr_vflip = float(np.corrcoef(d_blk[::-1].ravel(), a_blk.ravel())[0, 1])
    corr_hflip = float(np.corrcoef(d_blk[:, ::-1].ravel(), a_blk.ravel())[0, 1])
    assert corr > 0.95 and corr > corr_vflip + 0.1 and corr > corr_hflip + 0.1
    window.hide()
