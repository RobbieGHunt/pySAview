import h5py
import numpy as np
import pytest
from sa_nxs_explorer.core.nexus_reader import NexusReader


def write_scan(path, key, values):
    with h5py.File(path, "w") as file:
        entry = file.create_group("entry0")
        variables = entry.create_group("data_scan/scanned_variables")
        variables.create_dataset("data", data=[[10, 20], [1, 2]])
        names = variables.create_group("variables_names")
        names.create_dataset("name", data=[b"Detector1", b"th"])
        names.create_dataset("unit", data=[b"counts", b"deg"])
        variables.create_dataset(key, data=values)


@pytest.mark.parametrize("key,values,expected", [
    ("errors", [[2, 3], [0, 0]], [4, 9]),
    ("variance", [[2, 3], [0, 0]], [2, 3]),
])
def test_explicit_errors_and_variances_have_distinct_meaning(tmp_path, key, values, expected):
    path = tmp_path / "variance.nxs"
    write_scan(path, key, values)
    with NexusReader(path) as reader:
        np.testing.assert_array_equal(reader.parse_scan().variables["Detector1"].variance, expected)


def test_mismatched_uncertainty_shape_is_rejected(tmp_path):
    path = tmp_path / "bad.nxs"
    write_scan(path, "errors", [2, 3])
    with NexusReader(path) as reader:
        with pytest.raises(ValueError, match="does not match"):
            reader.parse_scan()


def test_negative_standard_error_is_not_squared_into_valid_variance(tmp_path):
    path = tmp_path / "negative.nxs"
    write_scan(path, "errors", [[-2, 3], [0, 0]])
    with NexusReader(path) as reader:
        assert np.isnan(reader.parse_scan().variables["Detector1"].variance[0])


def test_superadam_count_time_unit_assumption_is_recorded(tmp_path):
    path = tmp_path / "time.nxs"
    with h5py.File(path, "w") as file:
        entry = file.create_group("entry0")
        entry.create_group("SUPERADAM")
        variables = entry.create_group("data_scan/scanned_variables")
        variables.create_dataset("data", data=[[10, 20]])
        names = variables.create_group("variables_names")
        names.create_dataset("name", data=[b"Count"])
        names.create_dataset("label", data=[b"Time"])
        names.create_dataset("property", data=[b"Time"])
    with NexusReader(path) as reader:
        exposure = reader.parse_scan().variables["Count"]
        assert exposure.unit == "s"
        assert "file unit absent" in exposure.unit_source
