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
