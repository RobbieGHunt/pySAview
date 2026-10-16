"""Unit tests for NexusReader and data models."""

from pathlib import Path
import pytest
import numpy as np

from pySAview.core.nexus_reader import NexusReader, decode_val, decode_attr_str


@pytest.fixture
def example_nxs_path() -> Path:
    p = Path("pySAview/example_data/005572.nxs")
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / "example_data" / "005572.nxs"
    assert p.exists(), f"Example file not found: {p}"
    return p


def test_decode_val():
    assert decode_val(b"test") == "test"
    assert decode_val(np.bytes_(b"hello")) == "hello"
    assert decode_val(42) == 42
    assert decode_val(np.array([123])) == 123
    assert decode_val(np.array([b"a", b"b"])) == ["a", "b"]


def test_reader_open_and_children(example_nxs_path: Path):
    with NexusReader(example_nxs_path) as reader:
        roots = reader.list_children("/")
        assert len(roots) >= 1
        root_names = [r.name for r in roots]
        assert "entry0" in root_names

        entry_children = reader.list_children("/entry0")
        child_names = [c.name for c in entry_children]
        assert "data_scan" in child_names
        assert "SUPERADAM" in child_names
        assert "sample" in child_names
        assert "user" in child_names


def test_reader_attributes(example_nxs_path: Path):
    with NexusReader(example_nxs_path) as reader:
        attrs = reader.get_attributes("/entry0")
        assert "NX_class" in attrs
        assert attrs["NX_class"] == "NXentry"


def test_reader_preview(example_nxs_path: Path):
    with NexusReader(example_nxs_path) as reader:
        # String scalar
        prev_title = reader.read_preview("/entry0/title")
        assert prev_title.text is not None
        assert "Hydrogen-induced" in prev_title.text

        # 1D numeric array
        prev_vars = reader.read_preview("/entry0/data_scan/scanned_variables/data")
        assert prev_vars.data is not None
        assert prev_vars.data.ndim == 2
        assert "min" in prev_vars.stats

        # 3D volume (should slice frame 0 without loading 300M elements)
        prev_det = reader.read_preview("/entry0/data_scan/detector_data/data")
        assert prev_det.data is not None
        assert prev_det.data.ndim == 2  # first frame slice
        assert prev_det.truncated is True


def test_read_detector_frame_and_roi(example_nxs_path: Path):
    with NexusReader(example_nxs_path) as reader:
        frame0 = reader.read_detector_frame(0)
        assert frame0 is not None
        assert frame0.shape == (1024, 1024)

        roi = reader.get_detector_roi()
        assert roi is not None
        x0, x1, y0, y1 = roi
        assert x0 == pytest.approx(354.0)
        assert x1 == pytest.approx(382.0)
        assert y0 == pytest.approx(285.467, rel=1e-2)
        assert y1 == pytest.approx(742.935, rel=1e-2)

        # Out of bounds frame check
        with pytest.raises(IndexError):
            reader.read_detector_frame(9999)


def test_parse_scan(example_nxs_path: Path):
    with NexusReader(example_nxs_path) as reader:
        scan = reader.parse_scan()

        # Metadata
        assert scan.metadata.run_number == 5572
        assert "Hydrogen-induced" in scan.metadata.title
        assert scan.metadata.user == "kabanova"
        assert scan.metadata.local_contact == "VOROBIEV"
        assert scan.metadata.chemical_formula == "Tb/Co"
        assert scan.metadata.duration == pytest.approx(200.0)

        # Variables
        assert len(scan.variables) >= 13
        assert "th" in scan.variables
        assert "tth" in scan.variables
        assert "Detector1" in scan.variables
        assert "Monitor1" in scan.variables
        assert "Flipper1" in scan.variables
        assert "ROI_Counts" in scan.variables

        th = scan.variables["th"]
        assert len(th.data) == 302
        assert th.unit == "deg"

        # Motors
        assert len(scan.motors) >= 30
        assert "th" in scan.motors
        assert "tth" in scan.motors
        assert "ts" in scan.motors

        # Detector 2D
        assert scan.has_detector_2d is True
        assert scan.detector_shape == (302, 1024, 1024)


def test_compute_full_scan_roi(example_nxs_path: Path):
    with NexusReader(example_nxs_path) as reader:
        roi = reader.get_detector_roi()
        assert roi is not None
        counts = reader.compute_full_scan_roi(roi)
        assert counts is not None
        assert len(counts) == 302
        assert np.all(counts >= 0)
        assert np.sum(counts) > 0


def test_auto_fit_detector_rois(example_nxs_path: Path):
    with NexusReader(example_nxs_path) as reader:
        result = reader.auto_fit_detector_rois(skip_first_n=6)
        assert result is not None
        rois, counts = result
        assert len(rois) == 302
        assert len(counts) == 302
        assert np.all(counts >= 0)
        assert np.sum(counts) > 0

        # Centers should track reflection beam around ~368-380
        centers = [(r[0] + r[1]) / 2.0 for r in rois]
        assert np.all(np.array(centers) >= 365.0)
        assert np.all(np.array(centers) <= 385.0)

        # Test compute_per_frame_scan_roi
        pf_counts = reader.compute_per_frame_scan_roi(rois)
        assert pf_counts is not None
        assert len(pf_counts) == 302
        np.testing.assert_allclose(counts, pf_counts)
