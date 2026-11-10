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

After installation, launch with **`pySAview\run.bat`** (no console window) or
`.venv\Scripts\pysaview.exe`. If the application fails to start, run
`pySAview\run_debug.bat`, which keeps a console open to show the error. To open
a file directly:

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
| **Off-Specular** | Heatmap of every detector frame integrated along Y (full height, scan ROI or custom rows), shown as X pixel versus scan point, α_i versus α_f or Qx versus Qz, with spin selection or spin asymmetry, normalization and crosshair cuts. |
| **Metadata & Motors** | View experiment details, sample environment and instrument positions. |

Chart processing includes angle-to-Q conversion, time or monitor normalization,
and selectable **1σ–6σ** uncertainty bounds, defaulting to **2σ**. Spin separation
uses Flipper OFF/0 for **d/down** and ON/1 for **u/up**.

**Chart Display Options** controls spin colormaps, a shared shade range, lines,
markers, axis fonts, legend visibility and waterfall offsets.

Every plot toolbar uses the same icon buttons (pan, box zoom, zoom in/out, fit,
export image); hover a button for its name.

### Exporting

**File ▸ Export** collects the exports:

| Entry | Contents |
| --- | --- |
| **Reflectivity Data (tab-delimited)** (Ctrl+Shift+E) | The curves plotted in Scan Charts, each scan with its own ROI: step index, X, Y, 1σ standard uncertainty and the plotted interval bounds. `#` header lines record normalization, wavelength, the processing description, warnings and each curve's source file and uncertainty method. |
| **Reflectivity Plot Image** | The chart as PNG or SVG. |
| **Off-Specular Map Data (tab-delimited)** | The active scan's map in the current axes (X pixel vs point or θ, α_i/α_f, or Qx/Qz), spin selection or asymmetry and normalization: one row per frame and pixel with its coordinates, value and 1σ uncertainty. |
| **Off-Specular Map Image** | The map as PNG or SVG. |

Data files are tab-separated text with `#` comment lines, readable by
spreadsheets and e.g. `pandas.read_csv(path, sep="\t", comment="#")`. Display
settings (log colour/axis scales, waterfall offsets, asymmetry low-count
blanking) are never applied to exported values. Right-click any plot and choose
**Export Image…** to save just that plot. Exports never overwrite a loaded
measurement file.

ROI definitions can be saved and loaded as JSON files. File loading supports
progress and cancellation. The last load folder and display preferences are
remembered between launches.

See [Normalization and uncertainty](docs/normalization.md) for calculation
methods and statistical conventions.

The α_i–α_f view is calibrated per scan from the direct beam visible in the
detector images; see [Off-specular angle calibration](docs/off_specular.md).

## Development and tests

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:QT_QPA_PLATFORM = "offscreen"
.venv\Scripts\python.exe -m pytest
```

Windows executable builds use [pySAview/packaging/build.ps1](pySAview/packaging/build.ps1).

## License

See [LICENSE](LICENSE).
