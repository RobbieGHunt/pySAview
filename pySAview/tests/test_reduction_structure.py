"""Structural, purity, re-export identity, and unit tests for pySAview.core.reduction."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import pySAview.core.angle_calibration as legacy_angle_cal
import pySAview.core.normalization as legacy_norm
import pySAview.core.off_specular as legacy_offspec
import pySAview.core.reduction as reduction
import pySAview.core.reduction.angle_calibration as red_angle_cal
import pySAview.core.reduction.background as red_background
import pySAview.core.reduction.curvature as red_curvature
import pySAview.core.reduction.kinematics as red_kinematics
import pySAview.core.reduction.normalization as red_norm
import pySAview.core.reduction.overillumination as red_overillumination
import pySAview.core.reduction.plateau as red_plateau
import pySAview.core.reduction.polarization as red_pol
import pySAview.core.reduction.pysared as red_pysared
import pySAview.core.reduction.resolution as red_resolution
import pySAview.core.reduction.roi_extraction as red_roi
import pySAview.core.roi as legacy_roi
import pySAview.core.scan_calculations as legacy_scan_calc

REDUCTION_DIR = Path(reduction.__file__).resolve().parent


def _imported_roots(node: ast.AST) -> list[str]:
    roots: list[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            roots.append(alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom) and node.module:
        roots.append(node.module.split(".")[0])
    return roots


def _iter_module_level_nodes(tree: ast.Module):
    """Yield AST nodes outside function/class definitions (module top-level scope)."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def test_reduction_package_ast_purity():
    """pySAview/core/reduction/ must have zero PyQt6, pyqtgraph, h5py, and zero top-level scipy imports."""
    py_files = sorted(REDUCTION_DIR.glob("*.py"))
    assert py_files, "Expected Python modules in pySAview/core/reduction/"

    forbidden_anywhere = {"PyQt6", "pyqtgraph", "h5py"}
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            roots = _imported_roots(node)
            bad = forbidden_anywhere.intersection(roots)
            assert not bad, f"{path.name} imports forbidden module(s): {sorted(bad)}"
        for node in _iter_module_level_nodes(tree):
            roots = _imported_roots(node)
            assert "scipy" not in roots, f"{path.name} has a top-level scipy import (must stay lazy)"


def test_reduction_public_exports_complete():
    """pySAview.core.reduction.__all__ exposes every submodule's public API."""
    submodules = (
        red_angle_cal,
        red_background,
        red_curvature,
        red_kinematics,
        red_norm,
        red_overillumination,
        red_plateau,
        red_pol,
        red_pysared,
        red_resolution,
        red_roi,
    )
    expected = set()
    for mod in submodules:
        expected.update(mod.__all__)
    exported = set(reduction.__all__)
    assert exported == expected
    for name in reduction.__all__:
        assert hasattr(reduction, name), f"Missing export {name} on pySAview.core.reduction"


def test_backward_compatible_reexport_identity():
    """Legacy core/* modules re-export identical objects from core.reduction."""
    for name in red_angle_cal.__all__:
        assert getattr(legacy_angle_cal, name) is getattr(red_angle_cal, name)
    for name in ("q_from_angles", "angles_from_q"):
        assert getattr(legacy_angle_cal, name) is getattr(red_kinematics, name)

    for name in red_norm.__all__:
        assert getattr(legacy_norm, name) is getattr(red_norm, name)

    assert legacy_roi.pixel_bounds is red_roi.pixel_bounds

    assert legacy_scan_calc.compute_q is red_kinematics.compute_q
    assert legacy_scan_calc.normalize_signal is red_norm.normalize_signal
    for name in (
        "auto_fit_from_profiles",
        "auto_fit_scan_rois",
        "find_reflection_peak_center",
        "integrate_roi_2d",
        "pixel_bounds",
    ):
        assert getattr(legacy_scan_calc, name) is getattr(red_roi, name)

    for name in ("NORMALIZATION_UNITS", "NormalizedMap", "normalize_map"):
        assert getattr(legacy_offspec, name) is getattr(red_norm, name)
    for name in (
        "AsymmetryMap",
        "SummedSpinMap",
        "pair_spin_frames",
        "select_frames",
        "spin_asymmetry",
        "sum_spin_frames",
    ):
        assert getattr(legacy_offspec, name) is getattr(red_pol, name)
    for name in ("frame_y_bounds", "integrate_frames_along_y", "pixel_bounds", "y_pixel_bounds"):
        assert getattr(legacy_offspec, name) is getattr(red_roi, name)


def test_qz_step_per_pixel():
    lam = 5.21
    dpp = 0.008185
    step = red_kinematics.qz_step_per_pixel(dpp, lam)
    expected = (2.0 * np.pi / lam) * np.radians(dpp)
    assert step == pytest.approx(expected, rel=1e-12)
    # Matches small-angle finite difference of Qz near alpha_f = 0.
    _, qz0 = red_kinematics.q_from_angles(1.0, 0.0, lam)
    _, qz1 = red_kinematics.q_from_angles(1.0, dpp, lam)
    assert step == pytest.approx(qz1 - qz0, rel=1e-7)
    with pytest.raises(ValueError):
        red_kinematics.qz_step_per_pixel(dpp, 0.0)
    with pytest.raises(ValueError):
        red_kinematics.qz_step_per_pixel(dpp, float("nan"))
    for bad_dpp in (0.0, -0.01, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            red_kinematics.qz_step_per_pixel(bad_dpp, lam)


def test_constant_qz_alpha_f():
    lam = 5.21
    ai = np.array([0.2, 1.0, 2.5, 4.0])
    qz_target = 0.06
    af = red_kinematics.constant_qz_alpha_f(ai, qz_target, lam)
    _, qz_back = red_kinematics.q_from_angles(ai, af, lam)
    np.testing.assert_allclose(qz_back, qz_target, atol=1e-12)

    # Below-horizon branch (Qz < 0) works cleanly.
    af_neg = red_kinematics.constant_qz_alpha_f(np.array([0.5, 1.0]), -0.02, lam)
    assert np.all(af_neg < 0)
    _, qz_neg_back = red_kinematics.q_from_angles(np.array([0.5, 1.0]), af_neg, lam)
    np.testing.assert_allclose(qz_neg_back, -0.02, atol=1e-12)

    # Kinematically inaccessible Qz (> 2*k0) yields NaN without warnings.
    k0 = 2.0 * np.pi / lam
    af_oob = red_kinematics.constant_qz_alpha_f(np.array([1.0, 2.0]), 3.0 * k0, lam)
    assert np.all(np.isnan(af_oob))

    with pytest.raises(ValueError):
        red_kinematics.constant_qz_alpha_f(ai, qz_target, -5.0)


def test_constant_qx_alpha_f():
    lam = 5.21
    ai = np.array([0.5, 1.5, 2.5, 3.5])
    # pySAred convention: Qx = k0 * (cos(alpha_f) - cos(alpha_i)).
    # For Qx < 0, cos(alpha_f) < cos(alpha_i), so |alpha_f| > |alpha_i| for all ai.
    qx_target = -5e-4
    af_pos = red_kinematics.constant_qx_alpha_f(ai, qx_target, lam, side=1.0)
    assert np.all(af_pos > 0)
    qx_back, _ = red_kinematics.q_from_angles(ai, af_pos, lam)
    np.testing.assert_allclose(qx_back, qx_target, atol=1e-12)

    # Lower branch (below horizon, side = -1.0) reproduces the same Qx with negative alpha_f.
    af_neg = red_kinematics.constant_qx_alpha_f(ai, qx_target, lam, side=-1.0)
    np.testing.assert_allclose(af_neg, -af_pos, atol=1e-12)
    qx_back_neg, _ = red_kinematics.q_from_angles(ai, af_neg, lam)
    np.testing.assert_allclose(qx_back_neg, qx_target, atol=1e-12)

    # Positive Qx becomes inaccessible when cos(alpha_i) + Qx/k0 > 1 (small alpha_i).
    af_mixed = red_kinematics.constant_qx_alpha_f(np.array([0.1, 3.0]), 5e-4, lam, side=1.0)
    assert np.isnan(af_mixed[0])
    assert np.isfinite(af_mixed[1])
    qx_ok, _ = red_kinematics.q_from_angles(3.0, af_mixed[1], lam)
    assert qx_ok == pytest.approx(5e-4, abs=1e-12)

    with pytest.raises(ValueError):
        red_kinematics.constant_qx_alpha_f(ai, qx_target, 0.0)
    for bad_side in (0.0, 2.0, -2.0, float("nan"), True, False):
        with pytest.raises(ValueError):
            red_kinematics.constant_qx_alpha_f(ai, qx_target, lam, side=bad_side)


def test_split_indices_by_flipper():
    x = np.array([3.0, 1.0, 2.0, 4.0, 0.5])
    flipper = np.array([1, 0, 1, 0, 1])

    all_groups = red_pol.split_indices_by_flipper(x, flipper, spin_mode="all")
    assert len(all_groups) == 1
    assert all_groups[0][0] is None and all_groups[0][1] == "All states"
    np.testing.assert_array_equal(all_groups[0][2], [0, 1, 2, 3, 4])

    split_groups = red_pol.split_indices_by_flipper(x, flipper, spin_mode="split")
    assert [(k, lbl) for k, lbl, _ in split_groups] == [("d", "d (ON)"), ("u", "u (OFF)")]
    np.testing.assert_array_equal(split_groups[0][2], [1, 3])  # x = 1.0, 4.0
    np.testing.assert_array_equal(split_groups[1][2], [4, 2, 0])  # x = 0.5, 2.0, 3.0

    f0 = red_pol.split_indices_by_flipper(x, flipper, spin_mode="flipper0")
    assert len(f0) == 1 and f0[0][0] == "d"
    np.testing.assert_array_equal(f0[0][2], [1, 3])

    f1 = red_pol.split_indices_by_flipper(x, flipper, spin_mode="flipper1")
    assert len(f1) == 1 and f1[0][0] == "u"
    np.testing.assert_array_equal(f1[0][2], [4, 2, 0])

    # Unrecognized flipper values fall back to single "Scan" group in split mode, empty in single-state modes.
    unknown = np.full(5, -1)
    fallback = red_pol.split_indices_by_flipper(x, unknown, spin_mode="split")
    assert len(fallback) == 1 and fallback[0][0] is None and fallback[0][1] == "Scan"
    np.testing.assert_array_equal(fallback[0][2], [0, 1, 2, 3, 4])
    assert red_pol.split_indices_by_flipper(x, unknown, spin_mode="flipper0") == []
    assert red_pol.split_indices_by_flipper(x, unknown, spin_mode="flipper1") == []
