# pySAview

A desktop viewer for ILL SuperADAM neutron reflectometry NeXus (`.nxs`) files.
Inspect file structure, view detector images, and plot or compare scans.

## Installation and launch

Requires **Python 3.12**. On Windows:

```powershell
git clone https://github.com/RobbieGHunt/pySAview.git
cd pySAview
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pySAview
```

After installation, launch with **`pySAview\run.bat`** or
`.venv\Scripts\pysaview.exe`. To open a file directly:

```powershell
.venv\Scripts\python.exe -m pySAview "C:\path\to\scan.nxs"
```

## Using pySAview

Use **Open**, **Add**, or drag-and-drop to load files. Select an active file for
inspection and check files in **Loaded Scans** to compare them on the chart.

| Tab | Purpose |
| --- | --- |
| **File Hierarchy** | Browse datasets and attributes; preview data as plots, images, tables or text. |
| **Scan Charts** | Plot selected channels, compare scans, separate spin states, normalize counts and display uncertainties. |
| **2D Detector** | Browse detector frames, define or track an ROI, inspect profiles and apply ROI counts to the chart. |
| **Metadata & Motors** | View experiment details, sample environment and instrument positions. |

Chart processing includes angle-to-Q conversion, time or monitor normalization,
and selectable **1σ–6σ** uncertainty bounds, defaulting to **2σ**. Spin separation
uses Flipper OFF/0 for **d/down** and ON/1 for **u/up**.

**Chart Display Options** controls spin colormaps, a shared shade range, lines,
markers, axis fonts, legend visibility and waterfall offsets. **Export CSV**
saves numerical results; **Export Plot** saves PNG or SVG figures.

ROI definitions can be saved and loaded as JSON files. File loading supports
progress and cancellation. The last load folder and display preferences are
remembered between launches.

See [Normalization and uncertainty](docs/normalization.md) for calculation
methods and statistical conventions.

## Development and tests

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:QT_QPA_PLATFORM = "offscreen"
.venv\Scripts\python.exe -m pytest
```

Windows executable builds use [pySAview/packaging/build.ps1](pySAview/packaging/build.ps1).

## License

See [LICENSE](LICENSE).
