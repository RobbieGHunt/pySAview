# SuperADAM NeXus Explorer (`pySAview`)

A specialized graphical explorer for ILL SuperADAM neutron reflectometry NeXus (`.nxs`) scan files.
Intended to act as a way to rapidly view data file trees, detector images, and 1D scans, including comparison of several scans.

Processing conventions are described in:
[normalization and uncertainty](../docs/normalization.md).

---

## Features

Use is divided into top-level tabs for specific functions.

### File Tree (HDF5 / NeXus Browser)

- **Lazy loading tree view** for smooth browsing of complex NeXus hierarchies (`/entry0/...`).
- **Live filtering** by node name and path.
- **Node details**: Type, shape, dtype, element count, and decoded NeXus attributes (`NX_class`, units, etc.).
- **Dataset inspection**:
  - Interactive 1D pyqtgraph plots with index / value axes and log scale.
  - 2D heatmap rendering with colormap support.
  - Formatted text/scalar preview.
  - Spreadsheet table view of raw dataset values.

### Scan Charts (Reflectometry X-Y Plotter)

- **Channel selection**: Plot any scanned variable (`th`, `tth`, `Detector1`, `Monitor1`, `Time`, etc.).
- **$Q$ Momentum Transfer conversion**: Converts incident angle $\theta$ (deg) to $Q = \frac{4\pi}{\lambda}\sin(\theta)$ with configurable neutron wavelength $\lambda$ (default: 5.21 Å).
- **Polarization / Spin state separation**: Splits curves automatically by `Flipper1` into **d / Spin Down** (Flipper OFF / 0) and **u / Spin Up** (Flipper ON / 1) states. Full support for 2-channel measurements (u, d) but not necessarily 4-channel (uu, ud, dd, du) at present.
- **Multi-file spin colors**: `d` uses Matplotlib `Blues` and `u` uses `Reds` by default. Independent palette selectors include all Matplotlib defaults and reversed maps. Loaded scans use different shades and marker shapes; hiding a scan preserves the other scans' colors.
- **Chart Display Options tab**: Separate `d` and `u` controls for colormap (with a shade preview), solid/dashed line style, line width, and marker size. Set marker size to zero to hide one spin family's markers. Full colormap shade bars have draggable endpoint arrows; both spin families share the selected sampling range, initially 0–100%. Axis font size and legend visibility are also configurable.
- **Count Normalization**:
  - Raw counts
  - Unscaled beam-monitor ratio (`Detector1 / Monitor1`)
  - Normalized to dwell time (`Detector1 / Time`, cps)
  - Monitor ratio multiplied by a visible fixed reference monitor rate, shared across scans
- **Counting Uncertainty**: The **Sigma** selector offers 1σ–6σ and defaults to 2σ for both exact count intervals and propagated uncertainty bounds. Exact intervals use Gaussian-equivalent coverage; propagated bounds use ± the selected number of standard uncertainties. The numerical standard uncertainty is separately reported at 1σ.
- **Processing Status**: Formula, uncertainty method, wavelength, invalid-point exclusions, and unit assumptions are shown beside the plot. Missing/incompatible channels never silently change the selected processing.
- **Interactive cursor tracker**: Live mouse coordinate readout ($X$, $Y$).
- **Data Summary & Export**: Collapsible Plotted Data Points table with standard uncertainties and exact bounds, plus CSV and PNG/SVG plot export. The bottom data panel can be collapsed with its panel icon.

### 2D Detector Viewer & User-Defined ROI Integration

- **2D Frame Visualization**: Renders 2D position-sensitive detector (PSD) frames, oriented 90° CW and mirrored horizontally to match beamline physical geometry.
- **Scan Step Scrubber**: Slider, spinbox, prev/next stepper, and automated play/pause animation across the loaded scan’s frames.
- **Interactive User-Defined ROI**: Freely move and resize the ROI box directly on the detector image to correct for ROI mismatch, or use the automatic ROI calculator to track drift within a scan.
- **Apply to Scan Charts**: 1-click **"Apply ROI to Scan Chart"** integrates the loaded scan’s frames using the user-defined ROI and updates the Scan Charts view (`ROI_Counts`).
- **Per-scan ROI definitions**: Each scan owns its fixed or tracked ROI independently. Save/load definitions using separate JSON files; loading stages an overlay for review before applying it. The original file ROI remains available for reset.
- **1D Profile Cuts**: Real-time horizontal and vertical intensity cuts through the user ROI.

### Metadata & Motor Positions

- **Experiment Overview**: Proposal, run number, title, user, local contact, timestamps, duration, and reactor power.
- **Sample Environment**: Chemical formula, sample consistency, mass, temperature, and pressures.
- **Instrument Motors**: Searchable table of all 33+ SuperADAM motors and positioners (`entry0/SUPERADAM/*`) with current values, targets, offsets, and units. Shows most of the interesting and relevant motor values as compared to the file tree.

---

## How to Launch

### Option 1: Windows Batch File
Double-click:
```bat
pySAview\run.bat
```

### Option 2: Command Line (Module)
```powershell
.venv\Scripts\python.exe -m pySAview
```
Or open a specific file directly:
```powershell
.venv\Scripts\python.exe -m pySAview pySAview/example_data/005572.nxs
```

Run these commands from the repository root after following the
[installation instructions](../README.md#installation-and-launch).
The supported Windows launcher is `pySAview/run.bat`; the installed `pysaview`
command and `python -m pySAview` are also supported. Existing local
`Launch_pySAview.bat` wrappers point to this maintained checkout.
Obsolete VBScript, direct Python, shortcut-generator and `.lnk` launchers have
been removed. The old `sa_nxs_explorer/run.bat` forwarding launcher is retired.

## Startup and preferences

Startup opens an empty workspace unless a file is explicitly requested.
Example scans remain available from the File menu. The last load folder,
theme, Sigma selection, plotting styles, axis fonts, legend visibility and
data-panel visibility are remembered on normal exit. Files and ROIs are not
automatically reopened; session saving is deferred.

File loading runs in the background with progress and cancellation. Completed
files are retained when a batch is cancelled. Interactive ROI integration and
tracking still run on the GUI thread.

## Source data protection

Original `.nxs` and HDF5 files are read-only. Normalization, ROI changes and
display transformations happen in memory. CSV, plot exports and ROI definitions
are saved separately; the app must never overwrite a measurement file.

---

## Running Tests

Run the test suite with pytest:
```powershell
.venv\Scripts\pytest.exe pySAview/tests
```
Tests verify the reader, Q calculations, normalizations, statistical coverage and zero-count limits, flipper spin splitting, ROI integration, exports, and PyQt6 GUI components.
