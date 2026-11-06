# Off-specular maps and angle calibration

## Map

Each detector frame is displayed as `raw.T` (rows = Y, columns = X), the same
orientation as the 2D Detector tab. For every X pixel the frame is summed over
a Y window: the full detector height by default, the scan ROI's Y rows, or
custom rows. The summed profiles of all frames form the map, shown either as
**X pixel versus scan point** (as in pySAred's "Pixel vs. Point" map) or as
**α_i versus α_f**. Pixel selection follows the ROI convention in
[normalization.md](normalization.md#detector-roi-pixel-selection), and
normalization and uncertainties follow the same document. Log display leaves
zero and invalid pixels blank; it never substitutes a value.

## α_i / α_f geometry

```
α_i     = θ
α_f(x)  = (2θ − θ) + s · (x − x₀) · Δ
```

| Symbol | Meaning | Source |
| --- | --- | --- |
| θ, 2θ | sample and detector angles of the frame | scan channels `th`, `tth` |
| Δ | exit-angle step per displayed X pixel (°/px) | direct-beam fit |
| s | +1 if α_f increases with X, −1 if it decreases | sign of the direct-beam fit |
| x₀ | pixel whose exit angle is 2θ − θ | specular peak |

The SuperADAM NeXus files contain `Detector1/sample_distance` and
`Detector1/angular_width`, but in the example files both are written as 0, so
the calibration is measured from the data.

**Direct beam → Δ and s.** Part of the beam passes the sample without
reflecting and stays at lab angle 0, i.e. at an angular offset −2θ from the
detector reference direction. Its X position therefore moves linearly with 2θ.
For every frame with 2θ > 0.1°, the strongest peak outside a ±15 px band
around the specular peak is located (it must exceed 50 counts and 8× the
frame's median + 1). A Theil–Sen line is fitted to its centroids, then refined
by least squares with iterative rejection of points further than
max(3 px, 3 robust σ) from the line. Δ = 1 / |slope|. The direct beam lies at
lower α_f than the reflection, so s = −sign(slope). At least five frames are
required; otherwise Δ must be entered manually.

On the example scans 5572 and 5594 the fit gives 122.2 px per degree of 2θ
(Δ = 0.00819°/px, so 1024 px ≈ 8.4°), with residuals of 0.4–0.6 px RMS over
about 190 frames, and α_f decreasing with X.

**Specular peak → x₀.** x₀ is the median, over frames with at least 200
counts in the scan's ROI X range, of the intensity centroid in that range.
Centroids use pixel-centre coordinates (pixel *i* spans [*i*, *i* + 1)), the
same coordinates as the map and the α_f formula.
Using the specular peak puts the specular ridge on α_i = α_f even when the
recorded θ/2θ have a small misalignment. In the example scans the direct-beam
line extrapolates to about 6 px (≈0.05°) below the specular peak at 2θ = 0,
consistent with such an offset.

The mapping from pixel to angle is linear, which is the form the calibration
fits. Over the ±4° the detector covers, it differs from flat-detector atan
geometry by less than 0.002°. No sample-curvature correction is applied.

## Display and checks

* Each frame is drawn as its own strip of pixel cells at its measured θ. The
  α_i width of a strip extends halfway to the neighbouring θ values and only
  affects the display. Counts are never interpolated or regridded. Frames at a
  repeated θ overlap; the last one is drawn on top, and hover and cuts report
  that visible frame.
* The view requires a single spin state, because u and d frames share angles
  and would overlap. Frames whose detector chunk was never written (for
  example an aborted scan) are omitted and counted in the banner.
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

## Spin asymmetry

Choosing **Spin asymmetry** pairs each spin-up frame (Flipper1 = 1) with the
spin-down frame (0) measured at the same θ and 2θ (within 0.002°); unwritten
frames are never paired. For every pixel

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

## Readout

Hovering shows θ, 2θ (and α_f in the α view), the X pixel and the map value
without uncertainty. Angles are given to the resolution of the scan: the
fewest decimals that resolve the scan's finest angular step (α_f: the
calibrated step per pixel).
