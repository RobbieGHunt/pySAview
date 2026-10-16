import hashlib
import h5py
import numpy as np
import pytest
from pySAview.core.roi import pixel_bounds
from pySAview.core.nexus_reader import NexusReader
from pySAview.core.scan_calculations import integrate_roi_2d, auto_fit_scan_rois


@pytest.mark.parametrize("roi, expected", [
    ((1, 3, 0, 2), (1, 3, 0, 2)),
    ((3, 1, 2, 0), (1, 3, 0, 2)),
    ((-5, 2, -1, 9), (0, 2, 0, 3)),
    ((8, 10, 4, 6), (4, 4, 3, 3)),
    ((-8, -2, -4, -1), (0, 0, 0, 0)),
    ((.5, 2.5, 1.5, 2.5), (0, 2, 2, 2)),
    ((4, 4, 0, 3), (4, 4, 0, 3)),
])
def test_preview_scan_and_profiles_select_identical_pixels(tmp_path, roi, expected):
    # Non-square detector and asymmetric pixel values catch transpose mistakes.
    raw = np.arange(1, 25).reshape(2, 4, 3)
    target = tmp_path / "scan.nxs"
    with h5py.File(target, "w") as f:
        f.create_dataset("entry0/data_scan/detector_data/data", data=raw)
    digest = hashlib.sha256(target.read_bytes()).digest()
    assert pixel_bounds(roi, 4, 3) == expected
    x0, x1, y0, y1 = expected
    reader = NexusReader(target)
    try:
        assert reader.file.mode == "r"
        counts = reader.compute_full_scan_roi(roi)
        tracked = reader.compute_per_frame_scan_roi([roi, roi])
        for i in range(2):
            total, xp, yp = integrate_roi_2d(raw[i].T, roi)
            selected = raw[i, x0:x1, y0:y1]
            assert total == counts[i] == tracked[i] == selected.sum()
            np.testing.assert_array_equal(xp, selected.sum(axis=1))
            np.testing.assert_array_equal(yp, selected.sum(axis=0))
    finally:
        reader.close()
    assert hashlib.sha256(target.read_bytes()).digest() == digest


def test_invalid_roi_and_incomplete_tracking_rejected(tmp_path):
    with pytest.raises(ValueError):
        pixel_bounds((0, float("nan"), 0, 3), 4, 3)
    target = tmp_path / "scan.nxs"
    with h5py.File(target, "w") as f:
        f.create_dataset("entry0/data_scan/detector_data/data", data=np.ones((2, 4, 3)))
    reader = NexusReader(target)
    try:
        with pytest.raises(ValueError, match="one region per detector frame"):
            reader.compute_per_frame_scan_roi([(0, 2, 0, 2)])
    finally:
        reader.close()


def test_tracking_counts_match_returned_bounds():
    raw = np.arange(3*12*7).reshape(3, 12, 7)
    rois, counts = auto_fit_scan_rois(raw, 3, (2.5, 6.5, -2, 9), skip_first_n=3)
    for i, roi in enumerate(rois):
        assert counts[i] == integrate_roi_2d(raw[i].T, roi)[0]
