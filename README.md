# pySAview

A graphical explorer for ILL SuperADAM neutron reflectometry NeXus (`.nxs`)
files. Quickly inspect file trees, view detector images, and plot or compare
one-dimensional scans.

**Original measurement files are read-only.** Processing happens in memory;
data, plots and ROI definitions are exported to separate files.

## Features

The interface is organized into four main tabs:

| Tab | What you can do |
| --- | --- |
| **File Hierarchy** | Browse the HDF5/NeXus file tree, inspect shapes, data types, units and attributes, and preview datasets as text, tables, 1D plots or 2D images. |
| **Scan Charts** | Select scan channels, compare files, separate spin states, switch between angle and Q, normalize counts, and inspect uncertainty bounds and individual data points. |
| **2D Detector** | Browse or play detector frames, filter by spin, define a fixed ROI or track drift, and inspect horizontal/vertical profiles. Apply an ROI to the scan chart or save its definition. |
| **Metadata & Motors** | Inspect experiment details, sample environment and searchable instrument motor values, targets, offsets and units. |

File loading runs in the background with progress and cancellation. Light/dark
themes, customizable chart styles and collapsible panels help when comparing
many scans.

## Installation and launch

Requires **Python 3.12**. Run these commands from a terminal:

```powershell
git clone https://github.com/RobbieGHunt/pySAview.git
cd pySAview
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pySAview
```

Ensure `python` refers to Python 3.12 before creating the environment.
The commands above use Windows paths.

After installation, Windows users can double-click **[pySAview/run.bat](pySAview/run.bat)**.
It uses this checkout's `.venv` and reports a missing environment rather than
selecting a different Python installation. Existing local `Launch_pySAview.bat`
wrappers remain compatible.

Alternatively, use `.venv\Scripts\pysaview.exe` or open a particular file:

```powershell
.venv\Scripts\python.exe -m pySAview "C:\path\to\scan.nxs"
```

When updating an existing installation, rerun the `pip install -e .` command
after pulling the latest changes.

## Using pySAview

### Open and compare scans

Startup opens an empty workspace. Use **Open** to replace the current file set,
**Add** to extend it, or drag files into the window. Example scans are available
explicitly from **File → Open Example Scans**.

The active file determines the file-tree, detector and metadata views.
Checkboxes in **Loaded Scans** select the curves included in the chart overlay.
These are separate selections, so you can inspect one file while comparing
several others.

During batch loading, **Cancel** retains files already loaded. Failed files are
reported together while the remaining files continue loading. An Open operation
keeps the previous workspace until the first replacement scan loads successfully.

### Select channels and processing

In **Scan Charts → Data & Processing**, select the X/Y channels, normalization,
uncertainty method and spin display. Incident angle can be converted to
`Q = 4π sin(θ) / λ`, with a configurable wavelength (initially 5.21 Å).

| Normalization | Plotted value |
| --- | --- |
| Raw counts | `N` |
| Time | `N / t` |
| Monitor | `N / M` |
| Fixed reference monitor rate | `r_ref × N / M` |

Here `N` is detector/ROI counts, `M` is monitor counts and `t` is dwell time.
The reference rate is an explicit scale shared across scans, rather than a
separate scale estimated for each scan.

**Sigma** offers **1σ–6σ**, defaulting to **2σ**. Exact count intervals use the
corresponding Gaussian-equivalent confidence coverage; propagated bounds use
± the selected number of standard uncertainties. Numerical standard uncertainty
values remain separately identified at 1σ. The processing banner identifies the
method, exclusions and relevant assumptions.

The collapsible **Plotted Data Points** panel shows source step indices, X/Y
values, standard uncertainties, interval bounds and their method.

### Spin states and chart appearance

Spin separation follows `Flipper1`:

| Flipper | Incident spin | Default colormap |
| --- | --- | --- |
| OFF / 0 | `d` — down | Blues |
| ON / 1 | `u` — up | Reds |

This supports the two-channel `u`/`d` workflow. Full four-channel separation
(`uu`, `ud`, `dd`, `du`) is not established by this mapping.

In **Chart Display Options**, choose each spin's Matplotlib colormap, solid or
dashed lines, line width and marker size. Set marker size to zero to hide markers.
Both full-colormap previews have draggable arrows: their shared positions select
the range over which scan colors are evenly spaced. Hiding a scan preserves the
other scans' colors.

The same tab controls **Axis font size**, **Show legend**, log axes and the
optional **Waterfall** display. Waterfall spacing is logarithmic by default,
with a linear option; both spins from a scan share an offset. Numbered
`number.nxs` files ascend by scan number, followed by unnumbered files in load
order. Tables and CSV exports retain the unshifted values.

### Define and reuse an ROI

In **2D Detector**, drag or resize the ROI, enter numeric bounds, or use
**Auto-Fit ROI** to track drift within the scan. The detector orientation matches
the beamline display convention: rotate 90° clockwise and mirror horizontally.
Use the frame controls and profile plots to inspect the selection, then choose
**Apply ROI to Scan Chart** to calculate `ROI_Counts`.

Each scan owns its ROI independently. **Reset to File ROI** restores the original
file definition. Applying a working ROI never changes that original definition
or writes to the measurement file.

**Save ROI…** creates a separate JSON definition containing the coordinate
convention, detector dimensions, source path and fixed bounds or all tracked
frame bounds. It requires a new output filename.

**Load ROI…** stages an overlay for the current scan. Review it for drift before
applying it. Fixed definitions can be explicitly reused with matching detector
dimensions; tracked definitions require the original scan path and frame count.
Moving a tracked ROI manually changes it into a fixed ROI.

### Export results

**Export CSV** and **Export Plot** sit side by side below the chart controls:

- **CSV** saves full-precision calculated values, standard uncertainties, bounds
  and processing provenance, without waterfall offsets.
- **PNG / SVG** saves the current chart appearance, including axes, display
  offsets and the legend when visible.

Export destinations are checked to prevent overwriting measurement files.

### Remembered preferences

On normal exit, the app remembers the last load folder, theme, processing/display
choices, Sigma, spin styling, shade range, axis fonts, legend and data-panel
visibility, and detector display preferences.

File lists, ROIs and plot zoom are not automatically restored after restarting.
Save ROI definitions separately when needed; full session saving is deferred.

## Scientific scope and limitations

pySAview is an inspection and quick-comparison tool. Monitor-normalized intensity
is not automatically calibrated reflectivity: direct-beam, footprint, detector
efficiency, dead-time and polarization corrections are not implied.

Read **[Normalization and uncertainty](docs/normalization.md)** for formulas,
exact count-interval assumptions, supplied variance handling, invalid-point
policy and ROI pixel conventions. Counting intervals do not include uncertainty
introduced by automatic peak tracking.

Background loading keeps file parsing off the GUI thread. Interactive ROI
integration/tracking and final chart/table rendering still run on the GUI thread.

## Development and tests

Install development dependencies and run the suite from the repository root:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:QT_QPA_PLATFORM = "offscreen"
.venv\Scripts\python.exe -m pytest
```

Tests cover normalization, coverage, zero-count limits, ROI integration and
ownership, source-file protection, exports, preferences, background loading
and UI behavior. Supplied example scans support the real-file integration tests.

The Windows standalone build configuration is
[pySAview/packaging/build.ps1](pySAview/packaging/build.ps1), using
`pySAview/packaging/pySAview.spec` to produce `pySAview.exe`.

## License

See [LICENSE](LICENSE).
