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
        header.append(f"Curve {n}: {c.dataset_name or 'Default'} | {c.label} | source: {c.source_path} "
                      f"| uncertainty: {c.uncertainty_method}")
    columns = ["Dataset", "Curve", "StepIndex", _units(x_label, first.x_units), _units(y_label, first.y_units),
               "StdUncertainty", "LowerBound", "UpperBound"]

    def rows():
        for c in curves:
            name = c.dataset_name or "Default"
            for i in range(len(c.x)):
                yield (name, c.label, int(c.indices[i]) if i < len(c.indices) else i,
                       c.x[i], c.y[i],
                       c.y_err[i] if c.y_err is not None and i < len(c.y_err) else None,
                       c.y_lower[i] if c.y_lower is not None else None,
                       c.y_upper[i] if c.y_upper is not None else None)

    return header, columns, rows()


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
