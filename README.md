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
