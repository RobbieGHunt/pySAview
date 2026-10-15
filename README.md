# pySAview

SuperADAM NeXus file inspection, detector viewing, and quick scan comparison.

The initial import preserves the existing `sa_nxs_explorer` application,
tests, resources, launchers, and supplied example scans without changing
application behavior. Machine-specific shortcuts and Python caches are excluded.
The other PNR tools are maintained separately.

See [the application guide](sa_nxs_explorer/README.md) for the feature inventory.

## Installation and launch

Requires Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m sa_nxs_explorer
```

## Tests

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.venv\Scripts\python.exe -m pytest
```

The supplied scans are used by the reader and UI integration tests.

## Scientific processing

See [normalization and uncertainty](docs/normalization.md) for the formulas,
statistical assumptions, exact low-count intervals, and invalid-point policy.

Monitor mode is the unscaled count ratio `N/M`. The reference-rate mode uses
a visible fixed scale shared across files; it does not silently rescale each
scan to its own mean monitor. Standard uncertainties and exact confidence
limits are exported separately, with their methods identified.

## Incremental development

The current scope is the viewer's P0 work: validated normalization and
uncertainty, explicit channel/unit compatibility, and visible processing status.
Persistent ROI/session state, background processing, and further layout changes
remain later milestones. The magnetometry application is maintained separately.

## Supported launchers

On Windows, use `sa_nxs_explorer/run.bat`, which runs the repository's own
`.venv` and reports a missing environment rather than choosing another Python
installation. The module entry point `python -m sa_nxs_explorer` and installed
`sa-nxs-explorer` command remain supported.

Legacy `run.vbs`, `sa_nxs_launcher.py`, `create_shortcut.ps1`, and
`SuperADAM_Explorer.lnk` are retired and should not be used.

### Waterfall display

In **Chart Display Options**, enable **Waterfall: offset scans**. Logarithmic mode defaults to one decade per scan (multiply by `10 ** (rank * spacing)`); linear mode adds `rank * spacing` in plotted Y units. The mode selects the corresponding Y axis scale; the axis controls remain available. Both spin channels of a file share its offset. Numbered `number.nxs` files ascend from the base by scan number; other files follow in load order. Only visible scans occupy ranks. Offsets apply equally to interval endpoints and never change normalization, statistical results, tables, or exported numerical data. Legend labels identify offsets.

### Shared shade range

The u/d shade bars always display the full selected colormaps. Drag either triangular endpoint on either bar to choose a shared sampling range. Loaded scans are spaced evenly between the arrows; a single scan uses the midpoint. The initial range is 0–100%, and changing theme or colormap preserves the selected range.

### Source data and scan ownership

Measurement files are opened read-only and must never be overwritten by the app. Processing changes only in-memory data. Export destinations reject NeXus/HDF5 extensions and aliases of loaded source files. Each scan owns its working ROI independently; applying a user ROI preserves the original file ROI. Reusing an ROI across scans must be an explicit operation, never automatic inheritance. Future ROI/session saves belong in separate sidecar files.

### ROI definitions

In the 2D Detector ROI panel, **Save ROI…** saves the currently displayed fixed region or all tracked frame regions into a new `.json` file. Existing files are never overwritten by this operation; choose a new name. Definitions include a format version, pixel convention, detector dimensions, source path and frame count.

**Load ROI…** explicitly stages the definition for the active scan only. Check its overlay for drift, adjust it if needed, then choose **Apply ROI to Scan Chart**. Fixed definitions may be reused with matching detector dimensions; tracked definitions require the original source path and frame count. Moving a tracked ROI manually changes it into a fixed ROI. Applied working definitions survive switching scans in the running app; reopen them from JSON after restarting (full session persistence remains future work). Reset restores the file ROI without modifying it. Counting intervals do not quantify uncertainty introduced by peak tracking.

### Startup and preferences

Startup opens an empty workspace unless a file is explicitly passed on the command line. Example scans remain available from File → Open Example Scans. The last successful load folder and local preferences (theme, chart processing and display options, spin styles, shared shade range, and detector colormap/contrast) are remembered on normal exit using Qt user settings. File lists, ROI definitions and plot zoom are not automatically restored. No session files are created and measurement files remain read-only.
