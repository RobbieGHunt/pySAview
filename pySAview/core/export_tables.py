"""Tab-delimited text exports of derived data (never written into source files).

Format: ``#``-prefixed header lines describing the processing, one tab-separated
column-name line, then one tab-separated row per value. Floats are written with
Python's shortest round-trip representation (full precision); missing values
are empty fields and undefined values are ``nan``. Read back with e.g.
``pandas.read_csv(path, sep="\\t", comment="#")``, or ``csv.DictReader`` with
``delimiter="\\t"`` over the lines that do not start with ``#``. Spreadsheet
and plotting programs import it as tab-separated text.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from pySAview.core.source_safety import validate_output_path


def _field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return repr(value) if np.isfinite(value) else ("nan" if np.isnan(value) else repr(value))
    return " ".join(str(value).split())  # no tabs/newlines inside a field


def _header_line(text: str) -> str:
    return "# " + " ".join(str(text).split())


def write_table(path, header: Sequence[str], columns: Sequence[str], rows: Iterable[Sequence],
                source_paths: Iterable[str] = ()) -> int:
    """Write a tab-delimited table after the source-file guard; returns the number of rows."""
    target = validate_output_path(path, [p for p in source_paths if p])
    count = 0
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(_header_line(f"pySAview export, {datetime.now().isoformat(timespec='seconds')}") + "\n")
        for line in header:
            handle.write(_header_line(line) + "\n")
        handle.write("\t".join(_field(c) for c in columns) + "\n")
        for row in rows:
            handle.write("\t".join(_field(v) for v in row) + "\n")
            count += 1
    return count


def _units(label: str, units: str | None) -> str:
    return f"{label} ({units})" if units else label


def reflectivity_table(curves, x_label: str, y_label: str, status: str, warnings: Sequence[str]):
    """(header, columns, rows) for the plotted Scan Charts curves (unshifted values)."""
    first = curves[0]
    header = [
        "Reflectivity / scan-curve data as plotted in Scan Charts (each scan uses its own ROI).",
        f"Normalization: {first.normalization}",
    ]
    if first.reference_monitor_rate is not None:
        header.append(f"Reference monitor rate: {first.reference_monitor_rate!r} monitor counts/s")
    if first.wavelength is not None:
        header.append(f"Wavelength: {first.wavelength!r} Å")
    header.append(f"Processing: {status}")
    if warnings:
        header.append("Warnings: " + " | ".join(warnings))
    header.append("StdUncertainty is 1σ; LowerBound/UpperBound are the plotted interval at the selected sigma level.")
    header.append("Display offsets (waterfall) and log axes are not applied to these values.")
    for n, c in enumerate(curves, 1):
        red_info = f" | reduction: {c.reduction_summary}" if getattr(c, "reduction_summary", "") else ""
        header.append(f"Curve {n}: {c.dataset_name or 'Default'} | {c.label} | source: {c.source_path} "
                      f"| uncertainty: {c.uncertainty_method}{red_info}")
    has_res = any(getattr(c, "resolution", None) is not None for c in curves)
    res_units = next((getattr(c, "resolution_units", "") for c in curves if getattr(c, "resolution", None) is not None), "")
    columns = ["Dataset", "Curve", "StepIndex", _units(x_label, first.x_units), _units(y_label, first.y_units),
               "StdUncertainty", "LowerBound", "UpperBound"]
    if has_res:
        columns.append(_units("ResolutionSigma", res_units))

    def rows():
        for c in curves:
            name = c.dataset_name or "Default"
            res = getattr(c, "resolution", None)
            for i in range(len(c.x)):
                base = (name, c.label, int(c.indices[i]) if i < len(c.indices) else i,
                        c.x[i], c.y[i],
                        c.y_err[i] if c.y_err is not None and i < len(c.y_err) else None,
                        c.y_lower[i] if c.y_lower is not None else None,
                        c.y_upper[i] if c.y_upper is not None else None)
                if has_res:
                    yield base + (res[i] if res is not None and i < len(res) else None,)
                else:
                    yield base

    return header, columns, rows()


def _safe_dat_stem(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "curve"


def _curve_export_angle(curve, angle_mode: str, wavelength: float) -> np.ndarray:
    """Return angle column for pySAred .dat export in Qz, Degrees, or Radians."""
    theta = getattr(curve, "theta_deg", None)
    if theta is not None and len(theta) == len(curve.x):
        th_deg = np.asarray(theta, dtype=float)
        if angle_mode == "Qz":
            return (4.0 * np.pi / float(wavelength)) * np.sin(np.radians(th_deg))
        if angle_mode == "Degrees":
            return th_deg
        if angle_mode == "Radians":
            return np.radians(th_deg)
    # Fallback: derive from curve.x assuming Qz if x_units has Å⁻¹ / Angstrom, else degrees
    x = np.asarray(curve.x, dtype=float)
    x_is_q = "Å" in (getattr(curve, "x_units", "") or "") or getattr(curve, "wavelength", None) is not None
    if angle_mode == "Qz":
        return x if x_is_q else (4.0 * np.pi / float(wavelength)) * np.sin(np.radians(x))
    if angle_mode == "Degrees":
        if not x_is_q:
            return x
        return np.degrees(np.arcsin(np.clip(x * float(wavelength) / (4.0 * np.pi), -1.0, 1.0)))
    if angle_mode == "Radians":
        if not x_is_q:
            return np.radians(x)
        return np.arcsin(np.clip(x * float(wavelength) / (4.0 * np.pi), -1.0, 1.0))
    raise ValueError(f"Unsupported angle mode: {angle_mode!r}")


def write_pysared_dat(
    path,
    curves,
    *,
    angle_mode: str = "Qz",
    include_resolution: bool = True,
    remove_zeros: bool = False,
    exclude_negative: bool = True,
    wavelength: float = 5.21,
    source_paths: Iterable[str] = (),
) -> list[tuple[Path, int]]:
    """Write pySAred 1.5.3 space-delimited 3- or 4-column ``.dat`` file(s).

    Columns (no header, matching pySAred 1.5.3 ``f_button_reduceSFM``)::

        angle  Intensity  dIntensity  [Resolution_sigma]

    When ``curves`` has a single curve, writes directly to ``path``; when
    multiple curves (e.g. spin states ``u`` and ``d`` or multiple overlaid
    scans) are exported, writes one ``.dat`` file per curve suffixed with a
    sanitized curve tag. Every output file is validated by
    ``validate_output_path`` before opening.
    """
    if angle_mode not in ("Qz", "Degrees", "Radians"):
        raise ValueError(f"Unsupported angle mode: {angle_mode!r}")
    if not curves:
        raise ValueError("No curves available to export.")

    base = Path(path)
    if not base.suffix:
        base = base.with_suffix(".dat")
    sources = [p for p in source_paths if p]
    written: list[tuple[Path, int]] = []
    used_targets: set[Path] = set()

    for idx, curve in enumerate(curves):
        if len(curves) == 1:
            candidate = base
        else:
            tag = _safe_dat_stem(curve.label)
            candidate = base.with_name(f"{base.stem}_{tag}{base.suffix}")
            if candidate in used_targets:
                candidate = base.with_name(f"{base.stem}_{tag}_{idx + 1}{base.suffix}")
        used_targets.add(candidate)
        target = validate_output_path(candidate, sources)

        lam = float(curve.wavelength) if getattr(curve, "wavelength", None) else float(wavelength)
        angles = _curve_export_angle(curve, angle_mode, lam)
        ys = np.asarray(curve.y, dtype=float)
        errs = np.asarray(curve.y_err, dtype=float) if curve.y_err is not None else np.zeros_like(ys)
        res = getattr(curve, "resolution", None)
        write_res = bool(include_resolution and res is not None and len(res) == len(ys))

        lines: list[str] = []
        for i in range(len(ys)):
            ang = float(angles[i])
            val = float(ys[i])
            err = float(errs[i])
            r_val = float(res[i]) if write_res else 0.0
            if not (np.isfinite(ang) and np.isfinite(val) and np.isfinite(err)):
                continue
            if write_res and not np.isfinite(r_val):
                continue
            if exclude_negative and val < 0.0:
                continue
            if remove_zeros and val == 0.0:
                continue
            ang_str = repr(round(ang, 10))
            if write_res:
                lines.append(f"{ang_str} {repr(val)} {repr(err)} {repr(r_val)}\n")
            else:
                lines.append(f"{ang_str} {repr(val)} {repr(err)}\n")

        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            if lines:
                handle.writelines(lines)
            else:
                handle.write("All points are either zeros or negatives.\n")
        written.append((target, len(lines)))

    return written


def map_table(frame_indices: np.ndarray, values: np.ndarray, uncertainty: np.ndarray,
              coordinates: dict[str, np.ndarray], *, partner_indices: np.ndarray | None = None):
    """(columns, rows) for a frame × pixel map in long format.

    ``coordinates`` maps column names to arrays of shape (n_rows,) [per frame] or
    (n_rows, nx) [per cell]; they are written between the index and value columns.
    """
    n_rows, nx = values.shape
    columns = ["Frame"] + (["PartnerFrame"] if partner_indices is not None else []) + ["XPixel"]
    columns += list(coordinates) + ["Value", "StdUncertainty"]
    pixels = np.arange(nx)
    per_cell = {k: (np.broadcast_to(np.asarray(v, float)[:, None], (n_rows, nx)) if np.ndim(v) == 1
                    else np.asarray(v, float)) for k, v in coordinates.items()}

    def rows():
        for r in range(n_rows):
            frame = int(frame_indices[r])
            lead = (frame,) if partner_indices is None else (frame, int(partner_indices[r]))
            cols = [per_cell[k][r].tolist() for k in per_cell]
            for x, rest in zip(pixels.tolist(), zip(*cols, values[r].tolist(), uncertainty[r].tolist())):
                yield lead + (x,) + rest

    return columns, rows()
