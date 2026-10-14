# SuperADAM NeXus Explorer (`sa_nxs_explorer`)

A specialized PyQt6 and pyqtgraph graphical explorer for ILL SuperADAM polarized neutron reflectometry NeXus (`.nxs`) scan files.

Processing conventions and P0 changes are described in
[normalization and uncertainty](../docs/normalization.md).

---

## Features

### 1. 🌳 File Hierarchy (HDF5 / NeXus Browser)
- **Lazy loading tree view** for smooth browsing of complex NeXus hierarchies (`/entry0/...`).
- **Live filtering** by node name and path.
- **Node details**: Type, shape, dtype, element count, and decoded NeXus attributes (`NX_class`, units, etc.).
- **Dataset inspection**:
  - Interactive 1D pyqtgraph plots with index / value axes and log scale.
  - 2D heatmap rendering with colormap support.
  - Formatted text/scalar preview.
  - Spreadsheet table view of raw dataset values.

### 2. 📈 Scan Charts (Reflectometry X-Y Plotter)
- **Channel selection**: Plot any scanned variable (`th`, `tth`, `Detector1`, `Monitor1`, `Time`, etc.).
- **$Q$ Momentum Transfer conversion**: Converts incident angle $\theta$ (deg) to $Q = \frac{4\pi}{\lambda}\sin(\theta)$ with configurable neutron wavelength $\lambda$ (default: 5.21 Å).
- **Polarization / Spin state separation**: Splits curves automatically by `Flipper1` into **d / Spin Down** (Flipper OFF / 0) and **u / Spin Up** (Flipper ON / 1) states.
- **Multi-file spin colors**: `d` uses Matplotlib `Blues` and `u` uses `Reds` by default. Independent palette selectors include all Matplotlib defaults and reversed maps. Loaded scans use different shades and marker shapes; hiding a scan preserves the other scans' colors.
- **Chart Display Options tab**: Separate `d` and `u` controls for colormap (with a shade preview), solid/dashed line style, line width, and marker size. Set marker size to zero to hide one spin family's markers. Palette sampling avoids pale endpoints for clearer red/blue contrast.
- **Count Normalization**:
  - Raw counts
  - Unscaled beam-monitor ratio (`Detector1 / Monitor1`)
  - Normalized to dwell time (`Detector1 / Time`, cps)
  - Monitor ratio multiplied by a visible fixed reference monitor rate, shared across scans
- **Counting Uncertainty**: Exact 68.27% Poisson confidence intervals by default; independently identified first-order standard uncertainties as an alternative or for supplied variances.
- **Processing Status**: Formula, uncertainty method, wavelength, invalid-point exclusions, and unit assumptions are shown beside the plot. Missing/incompatible channels never silently change the selected processing.
- **Interactive cursor tracker**: Live mouse coordinate readout ($X$, $Y$).
- **Data Summary & Export**: Curve statistics (peak position, centroid, integrated counts) and 1-click CSV data export.

### 3. 2D Detector Viewer & User-Defined ROI Integration
- **2D Frame Visualization**: Renders 1024×1024 2D position-sensitive detector (PSD) frames, oriented 90° CW and mirrored horizontally to match beamline physical geometry.
- **Scan Step Scrubber**: Slider, spinbox, prev/next stepper, and automated play/pause animation across all 302 scan steps.
- **Interactive User-Defined ROI**: Freely move and resize the ROI box directly on the detector image, or type exact numeric bounds ($X_{\min}, X_{\max}, Y_{\min}, Y_{\max}$).
- **Apply to Scan Charts**: 1-click **"Apply ROI to Scan Chart"** integrates all 302 scan frames using the user-defined ROI and instantly updates the Scan Charts view (`ROI_Counts`).
- **1D Profile Cuts**: Real-time horizontal and vertical intensity cuts through the user ROI.
- **Intensity Colormaps**: Turbo (default), Viridis, Plasma, Inferno, Cividis, and Gray, with Log-intensity contrast toggle.

### 4. 📋 Metadata & Motor Positions
- **Experiment Overview**: Proposal, run number, title, user, local contact, timestamps, duration, and reactor power.
- **Sample Environment**: Chemical formula, sample consistency, mass, temperature, and pressures.
- **Instrument Motors**: Searchable table of all 33+ SuperADAM motors and positioners (`entry0/SUPERADAM/*`) with current values, targets, offsets, and units.

---

## How to Launch

### Option 1: Windows Batch File
Double-click:
```bat
sa_nxs_explorer\run.bat
```

### Option 2: Command Line (Module)
```powershell
.venv\Scripts\python.exe -m sa_nxs_explorer
```
Or open a specific file directly:
```powershell
.venv\Scripts\python.exe -m sa_nxs_explorer sa_nxs_explorer/example_data/005572.nxs
```

The supported Windows launcher is `run.bat`. The obsolete VBScript launcher,
direct Python launcher, shortcut generator, and old `.lnk` shortcut are no
longer supported. Use the batch file above or the Python module entry point.

---

## Running Tests

Run the test suite with pytest:
```powershell
.venv\Scripts\pytest.exe sa_nxs_explorer/tests
```
Tests verify the reader, Q calculations, normalizations, statistical coverage and zero-count limits, flipper spin splitting, ROI integration, exports, and PyQt6 GUI components.
