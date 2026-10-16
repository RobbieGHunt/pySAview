import json
import numpy as np
import pytest
from pySAview.core.roi_record import ROIRecord, save_roi, load_roi
from pySAview.ui.detector_view import DetectorView, QFileDialog
from pySAview.tests.test_processing_ui import scan


def fixed(path="first.nxs"):
    return ROIRecord("fixed", (6, 8), 3, ((1., 4., 2., 5.),), path)


def test_roundtrip_reuse_and_no_overwrite(tmp_path):
    record = fixed()
    target = tmp_path / "roi.json"
    save_roi(target, record)
    assert load_roi(target) == record
    record.validate_scan((12, 8, 6), "different.nxs")
    with pytest.raises(ValueError):
        record.validate_scan((3, 6, 8), "different.nxs")
    with pytest.raises(FileExistsError):
        save_roi(target, record)
    with pytest.raises(ValueError):
        save_roi(tmp_path / "source.nxs", record)


def test_tracked_records_require_same_scan_and_all_frames(tmp_path):
    r = ROIRecord("tracked", (6, 8), 3, ((1., 3., 2., 5.), (2., 4., 2., 5.), (3., 5., 2., 5.)), "first.nxs")
    target = tmp_path / "tracked.json"
    save_roi(target, r)
    assert load_roi(target) == r
    r.validate_scan((3, 8, 6), "first.nxs")
    for shape, name in [((3, 8, 6), "second.nxs"), ((2, 8, 6), "first.nxs")]:
        with pytest.raises(ValueError):
            r.validate_scan(shape, name)
    with pytest.raises(ValueError):
        ROIRecord("tracked", (6, 8), 3, r.bounds[:2], "first.nxs")


@pytest.mark.parametrize("field,value", [("version", 2), ("convention", "unknown"), ("bounds", [[0, 0, 0, 0]]), ("frame_count", -1)])
def test_bad_sidecars_rejected(tmp_path, field, value):
    target = tmp_path / "roi.json"
    save_roi(target, fixed())
    data = json.loads(target.read_text())
    data[field] = value
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_roi(target)


class Reader:
    def read_detector_frame(self, i):
        return np.ones((8, 6))
    def compute_full_scan_roi(self, roi):
        return np.array([9., 9., 9.])
    def compute_per_frame_scan_roi(self, rois):
        return np.array([6., 6., 6.])


def detector_scan(name):
    data = scan(name)
    data.has_detector_2d = True
    data.detector_shape = (3, 8, 6)
    data.detector_roi = (0., 2., 0., 2.)
    return data


def test_explicit_load_stages_fixed_roi_for_one_scan_only(qtbot, monkeypatch, tmp_path):
    view = DetectorView()
    qtbot.addWidget(view)
    first, second = detector_scan("first.nxs"), detector_scan("second.nxs")
    view.set_data(Reader(), second)
    target = tmp_path / "fixed.json"
    save_roi(target, fixed())
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(target), "JSON"))
    received = []
    view.roi_applied.connect(lambda *args: received.append(args))
    view._load_roi_definition()
    assert not received
    assert second.working_roi_record is None
    assert view._get_current_roi_bounds() == fixed().bounds[0]
    view._apply_roi_to_scan_chart()
    assert second.working_roi_record.mode == "fixed"
    assert second.working_roi_record.source_path == "second.nxs"
    assert first.working_roi_record is None
    assert second.detector_roi == (0., 2., 0., 2.)


def test_tracked_state_survives_switch_and_reset(qtbot):
    view = DetectorView()
    qtbot.addWidget(view)
    first, second = detector_scan("first.nxs"), detector_scan("second.nxs")
    first.working_roi_record = ROIRecord("tracked", (6, 8), 3, ((1.,3.,1.,4.), (2.,4.,1.,4.), (3.,5.,1.,4.)), first.file_path)
    view.set_data(Reader(), first)
    assert view._auto_fit_active
    assert len(view._per_frame_rois) == 3
    view.set_data(Reader(), second)
    assert not view._auto_fit_active
    assert view._per_frame_rois is None
    view.set_data(Reader(), first)
    assert view._auto_fit_active
    view._reset_to_file_roi()
    assert first.working_roi_record.mode == "fixed"
    assert first.working_roi_record.bounds == (first.detector_roi,)
