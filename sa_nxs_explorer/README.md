# SuperADAM NeXus Explorer (`sa_nxs_explorer`)

A specialized PyQt6 and pyqtgraph graphical explorer for ILL SuperADAM polarized neutron reflectometry NeXus (`.nxs`) scan files.

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
- **Polarization / Spin state separation**: Splits curves automatically by `Flipper1` into **Spin Up** (Flipper OFF / 0) and **Spin Down** (Flipper ON / 1) states.
- **Count Normalization**:
  - Raw counts
  - Normalized to beam monitor (`Detector1 / Monitor1`)
  - Normalized to dwell time (`Detector1 / Time`, cps)
- **Poisson Error Bars**: Automatic $\pm 1\sigma$ error bars with propagated uncertainties.
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

### Option 3: Direct Launcher Script
```powershell
.venv\Scripts\python.exe sa_nxs_explorer/sa_nxs_launcher.py
```

---

## Running Tests

Run the test suite with pytest:
```powershell
.venv\Scripts\pytest.exe sa_nxs_explorer/tests
```
All 19 dedicated tests verify the reader, Q calculations, normalizations, flipper spin splitting, ROI integration, and PyQt6 GUI components.
