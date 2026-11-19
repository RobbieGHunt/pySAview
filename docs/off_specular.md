# Off-specular maps and angle calibration

## Map

Each detector frame is displayed as `raw.T` (rows = Y, columns = X), the same
orientation as the Detector Image tab. For every X pixel the frame is summed over
a Y window: the full detector height by default, the scan ROI's Y rows, or
custom rows. The summed profiles of all frames form the map, shown as
**X pixel versus θ** (default, drawn on exact θ cell strips so step-size changes
never kink the map), **X pixel versus scan point** (as in pySAred's "Pixel vs.
Point" map), **α_i versus α_f**, or **Qx versus Qz**. Pixel selection follows the
ROI convention in
[normalization.md](normalization.md#detector-roi-pixel-selection), and
normalization and uncertainties follow the same document. Log display leaves
zero and invalid pixels blank; it never substitutes a value.

## α_i / α_f geometry

Two per-scan calibration modes are available in `AngleCalibration` and in the
Off-Specular calibration panel:

1. **Direct-beam fit (`mode="direct_beam"`, default)** — linear pixel-to-angle
   mapping calibrated directly from the transmitted beam streak:

   ```
   α_i     = θ
   α_f(x)  = (2θ − θ) + s · (x − x₀) · Δ
   ```

2. **pySAred geometry (`mode="pysared"`)** — flat-detector `arctan` mapping
   matching classic `pySAred`, parameterized by manual sample-to-detector
   distance *L* (default `2500.0` mm, `PYSARED_DEFAULT_DISTANCE_MM`) and
   detector width *W* (default `300.0` mm, `DETECTOR_WIDTH_MM`):

   ```
   mm_per_pix = W / nx
   α_f(x)     = (2θ − θ) + s · degrees(arctan((x − x₀) · mm_per_pix / L))
   ```

   Its exact inverse `pixel_for_alpha_f_offset(offset_deg, nx, calibration)`
   uses `x₀ + s · (L / mm_per_pix) · tan(radians(offset_deg))` for
   `|offset_deg| < 90°` (returning `NaN` outside that range) and
   `x₀ + s · offset_deg / Δ` in `direct_beam` mode, keeping crosshair selection
   and constant-`Q` cuts exact in both modes. Switching modes preserves any
   stored `DirectBeamFit`, `x₀`, `s`, and manual `L`/`W` values for that scan.

| Symbol | Meaning | Source |
| --- | --- | --- |
| θ, 2θ | sample and detector angles of the frame | scan channels `th`, `tth` |
| Δ | exit-angle step per displayed X pixel (°/px) | direct-beam fit (`direct_beam` mode) or effective `atan((W/nx)/L)` (`pysared` mode) |
| L, W | sample-to-detector distance and detector width (mm) | manual geometry (`pysared` mode; defaults `2500 mm`, `300 mm`) |
| s | +1 if α_f increases with X, −1 if it decreases | sign of the direct-beam fit (or manual) |
| x₀ | pixel whose exit angle is 2θ − θ | specular peak (or manual) |

The SuperADAM NeXus files contain `Detector1/sample_distance` and
`Detector1/angular_width`, but in the example files both are written as 0, so
the calibration is measured from the data.

**Direct beam → Δ and s.** Part of the beam passes the sample without
reflecting and stays at lab angle 0, i.e. at an angular offset −2θ from the
detector reference direction. Its X position therefore moves linearly with 2θ.
Frames that were not written to disk (e.g. an aborted scan) are skipped, and
frames with |2θ| < 0.40° (`min_tth`) are excluded because grazing-angle sample
masking shifts the transmitted direct-beam centroid toward the specular ridge
(Gibaud, Vignaud & Sinha 1993, *Acta Cryst.* A49, Eq. 13 & Fig. 4) and the
direct beam overlaps the specular ROI. For each remaining frame, the strongest
peak outside a ±15 px band around the specular peak and at least `edge_margin`
(20 px) from the detector array borders is located (it must exceed 50 counts
and 8× the frame's median + 1). Its centroid is computed after subtracting a
local sideband median background around the peak window. A Theil–Sen line is
fitted to these centroids, then refined by least squares with iterative
rejection of points further than max(3 px, 3 robust σ) from the line.

To guard against internal active-detector cutoffs (such as SuperADAM's 1024 px
TDC readout whose active cathode ends near x ≈ 912 px), valid measured frames
not in the inlier set are checked against the fitted line: if at least two
valid frames have predicted positions past the end of the inlier streak yet
inside the detector array, that terminal pixel is treated as an active-edge
cutoff, inliers within `edge_margin` (20 px) of it are removed, and the line
is refitted on the clean interior. Scans that end before reaching an edge are
not trimmed. Δ = 1 / |slope|. The direct beam lies at lower α_f than the
reflection, so s = −sign(slope). At least five frames are required; otherwise
Δ must be entered manually.

On the example scans 5572 and 5594 the fit gives 122.17 px/° and 122.16 px/°
(Δ = 0.008185°/px and 0.008186°/px, agreeing to <0.01%, with 1024 px ≈ 8.4°),
residuals of 0.35–0.44 px RMS over ~168–169 interior frames (2θ ≈ 0.40°–4.30°),
and α_f decreasing with X.

**Specular peak → x₀.** x₀ is the median, over measured frames with at least
200 counts in the scan's ROI X range, of the intensity centroid in that range.
When 2θ is available, frames with |2θ| ≥ `min_tth` (0.40°) are preferred so
low-angle direct-beam overlap inside the specular ROI cannot bias x₀ (falling
back to all measured frames only if none at |2θ| ≥ `min_tth` meet the count
threshold). Centroids use pixel-centre coordinates (pixel *i* spans [*i*, *i* + 1)),
the same coordinates as the map and the α_f formula.
Using the specular peak puts the specular ridge on α_i = α_f even when the
recorded θ/2θ have a small misalignment. In the example scans the direct-beam
line extrapolates to about 6 px (≈0.05°) below the specular peak at 2θ = 0,
consistent with such an offset.

The mapping from pixel to angle is linear, which is the form the calibration
fits. In the example scans the detector covers about −3° to +5.4° around x₀;
over that range the linear mapping differs from flat-detector atan geometry by
at most ~0.006°, less than one pixel (0.0082°). Fitting the direct beam with atan geometry instead does not reduce the residuals,
so the data cannot tell the two apart.

## Sample curvature and background ROI overlay

Per-scan **Sample curvature** (`γ`, in radians, default `0.0`, following pySAred 1.5.3)
can be adjusted directly in **2D Map** (synced bidirectionally with
**Scan Charts ▸ Reduction**). Shearing is applied non-destructively per spin state
inside the signal ROI columns (and independently inside the background ROI columns
when enabled); resetting `γ = 0` restores the exact uncorrected map. Automatic colour-bar
levels are anchored to the uncurved (`γ = 0`) baseline (and any manual colour-bar range is
preserved during curvature adjustment) so fractional interpolated edge counts never cause
the palette scale to jump. Enable **ROI-summed profile** in the cuts panel to watch Kiessig
fringes sharpen live as `γ` is tuned. When background ROI subtraction is active, dashed
cyan lines mark the background ROI boundaries on the map. See [Data reduction](reduction.md).

## Display smoothing (`α_i vs α_f` and `Qx vs Qz`)

In **α_i vs α_f** and **Qx vs Qz** views, the optional **Smooth** control applies a
separable, NaN-safe 2D Gaussian filter (configurable radius `σ` in pixels) to the rendered
map image only. Colour-bar levels stay anchored to the unsmoothed baseline, and all
numerical data (`_norm`, cursor readouts, 1D cuts, and exported tables) remain strictly
unsmoothed.

**Sample-to-detector distance.** In **Direct-beam fit** mode no distance is
needed or entered: Δ is measured directly in degrees per pixel, and α_f, Qx and
Qz follow from Δ, x₀, θ, 2θ and λ. The panel still shows the distance that Δ
implies, as a check assuming nominal pixel pitch `p = 300 mm / nx`:

```
p = 300 mm / 1024 px = 0.2930 mm      L = p / tan(Δ)
```

(the SuperADAM detector is 300 mm × 300 mm; `DETECTOR_WIDTH_MM` in
`core/reduction/angle_calibration.py`). For the example scans L ≈ 2051 mm
(5572: 2050.8 mm, 5594: 2050.6 mm); an atan fit to the direct beam gives
2047 mm. In **Direct-beam fit** mode this implied distance follows any manual
change of Δ and is written to the export header, but is never used in an angle
or Q conversion. In **pySAred geometry** mode, by contrast, the entered *L*
(`sample_detector_distance_mm`) and detector width *W* (`detector_width_mm`)
directly drive the `arctan` pixel-to-angle conversion above.

## Reduction package layout

The pure, UI-free, I/O-free numerical reduction steps live under
`pySAview/core/reduction/` (re-exported by the legacy `pySAview/core/*` modules
for backward compatibility):

* `roi_extraction.py` — whole-pixel ROI bounds, 2D and Y-axis integration, and
  reflection peak tracking.
* `angle_calibration.py` — direct-beam fit and classic `pySAred` flat-detector
  pixel-to-angle calibration (`AngleCalibration`, `DirectBeamFit`,
  `find_direct_beam`, `fit_direct_beam`, `specular_center`, `calibrate`,
  `alpha_f_offsets`, `alpha_f_offset_for_pixel`, `pixel_for_alpha_f_offset`,
  `effective_deg_per_pixel`, `(α_i, α_f)` strip coordinates).
* `kinematics.py` — specular `Q`, off-specular `(Qx, Qz)` and inverse
  `(α_i, α_f)` mapping, pixel `Qz` step, and constant-`Qz` / constant-`Qx`
  iso-contours.
* `normalization.py` — 1D and 2D count normalization with exact Garwood /
  Clopper–Pearson Poisson intervals and first-order uncertainty propagation.
* `polarization.py` — spin-state selection, 1D flipper splitting, angle-matched
  `u`/`d` pairing, pooled `u + d` sum maps, and spin asymmetry.

## Display and checks

* Each frame is drawn as its own strip of pixel cells at its measured θ. The
  α_i width of a strip extends halfway to the neighbouring θ values and only
  affects the display. Counts are never interpolated or regridded. Frames at a
  repeated θ overlap; the last one is drawn on top, and hover and cuts report
  that visible frame.
* The view requires a single spin state, because u and d frames share angles
  and would overlap. Frames whose detector chunk was never written (for
  example an aborted scan) are omitted and counted in a notice above the map.
* **Show fit / reference lines** draws the fitted direct-beam line on the
  pixel map, and the lines α_f = α_i (specular) and α_f = −α_i (direct beam)
  on the α map. Data that follow these lines confirm the calibration.

## Scope and storage

The calibration belongs to each scan and is kept in memory only. It is fitted
automatically the first time a scan's map is computed, from that map's Y
window (full detector height by default; the window used is recorded and
shown), and is never inherited from another scan. Editing Δ, x₀ or the direction changes only that quantity
for that scan and marks it *manual*; **Fit Direct Beam** discards the manual
values and refits. Hovering over the calibration panel shows the fit details.
Measurement files are never modified.

## Qx–Qz

The **Qx vs Qz** view uses the same calibration and the wavelength λ set in
the calibration panel (default 5.21 Å, as in Scan Charts):

```
Qx = (2π/λ)(cos α_f − cos α_i)      Qz = (2π/λ)(sin α_f + sin α_i)
```

These are the formulas pySAred uses. Qx is shown as defined here (pySAred
plots −Qx), in units of 10⁻³ Å⁻¹ because it is about 100× smaller than Qz;
the axes are not aspect-locked. Each pixel cell is drawn exactly by mapping
its four (α_i, α_f) corners to Q, so cells are slightly curved quadrilaterals.
Nothing is interpolated or regridded. The specular rod lies on Qx = 0 (white
reference line). The direct beam (α_f = −α_i) collapses to the origin and
has no reference line in this view.

Drawing ~150 000 cell polygons on every repaint makes panning slow, so the
cells are rendered once into images. While the view moves, those images are
only shifted and scaled. About 0.1 s after it stops, the visible region is
re-rendered at screen resolution from the same cells. Image exports always
draw the cells directly. This changes the display only, never the values.

Cuts follow lines in Q: the horizontal cut is at constant Qz versus Qx, and
the vertical cut is at constant Qx versus Qz. Each measured point contributes
the one pixel whose centre lies on the line; points whose detector does not
reach the line are left out. The readout adds Qx and Qz for the pixel under
the cursor.

## Pooled spin sum and spin asymmetry

Choosing **u + d — Sum of channels** or **Spin asymmetry** pairs each spin-up
frame (Flipper1 = 1, OFF) with the spin-down frame (0, ON) measured at the same
θ and 2θ (within 0.002°); unwritten frames are never paired.

For **u + d — Sum of channels**, counts and exposures are pooled across each
pair (`N_u + N_d`, `(N_u + N_d) / (t_u + t_d)`, or `(N_u + N_d) / (M_u + M_d)`)
with first-order Poisson uncertainty from `N_tot = N_u + N_d` and
`M_tot = M_u + M_d` (or supplied monitor variances `v_Mu + v_Md`). Pairs where
either frame has an invalid exposure or monitor are excluded as `NaN`.

For **Spin asymmetry**, at every pixel

```
SA = (I_u − I_d) / (I_u + I_d)
var(SA) = (2 I_d / S²)² var(I_u) + (2 I_u / S²)² var(I_d),   S = I_u + I_d
```

where I_u and I_d use the selected normalization. With raw counts, SA assumes
equal u and d exposure (in the example files the monitor ratio is ≈ 0.99), so
**Counts / Monitor1** is recommended. Pixels with S ≤ 0 are undefined. Pixels
with fewer raw counts in u + d than the threshold (default 10) are blanked on
the map and cuts only; the values themselves are unchanged. The colour scale
is diverging and symmetric about zero. Log scaling does not apply. The direct
beam, which does not interact with the sample, shows SA ≈ 0 and is a useful
check.

## Export

**File ▸ Export ▸ Off-Specular Map Data** writes the map exactly as currently
selected (axes, spin or asymmetry, Y rows, normalization) as tab-delimited text,
one row per frame and displayed X pixel: `Frame` (0-based; asymmetry adds the
paired d frame as `PartnerFrame`), `XPixel`, the view's coordinates (θ; α_i and
α_f; or Qx and Qz in Å⁻¹, at pixel centres), `Value` and its first-order 1σ
`StdUncertainty`. In the angle views only the drawn frames are written. The
`#` header records the source file, Y rows, normalization, calibration and
wavelength. Values are linear and unblanked: the log colour scale and the
asymmetry low-count threshold are display only. **Off-Specular Map Image**
(or the toolbar's export button, or right-click ▸ **Export Image…**) saves the
map as shown as PNG or SVG.

Only problems and caveats (a missing channel, omitted frames, raw-count
asymmetry) appear in the notice line above the map; the controls already show
what is plotted.

## Readout

Hovering shows θ, 2θ (and α_f in the α view), the X pixel and the map value
without uncertainty. Angles are given to the resolution of the scan: the
fewest decimals that resolve the scan's finest angular step (α_f: the
calibrated step per pixel).
