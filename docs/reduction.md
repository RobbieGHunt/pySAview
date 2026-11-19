# Data reduction (pySAred 1.5.3)

`pySAview` implements the reflectometry data reduction routines of Alexey Klechikov's
**pySAred v1.5.3** (`pySAred_V1.5.3.py`). Each physical correction lives in its own
small, standalone, pure-NumPy module under `pySAview/core/reduction/` so every formula can be
audited independently without stepping through GUI code.

All transformations are computed **in memory** from read-only NeXus/HDF5 measurement
files and stored per scan in `ScanData.reduction_params` (`ReductionParams`). Reduction
settings are **never automatically inherited** when loading or switching to another
scan; explicit cloning via **Copy Settings to Checked Scans** creates independent copies.

## Module layout

| Module | Physical step | Reference in `pySAred_V1.5.3.py` |
| --- | --- | --- |
| `pySAview/core/reduction/overillumination.py` | Trapezoidal beam footprint factor `c₀` and projected width `fwhm_proj`; Nomad `s1h`/`s2h` extraction | `f_overilluminationCorrCoeff` |
| `pySAview/core/reduction/curvature.py` | ROI shear-and-regrid sample curvature correction (per spin state, reversible) | `f_SFM_reflectivityPreview_load` |
| `pySAview/core/reduction/background.py` | Second Background ROI geometry (Y- and width-locked) and BKG subtraction with `Qz` cutoff | `f_button_reduceSFM` / BKG ROI |
| `pySAview/core/reduction/resolution.py` | Gaussian 1σ resolution (`σ_Qz`, `σ_θ`) in both `pySAred` (`FWHM_proj`) and Gunnar `Sared` (`0.68`) modes | `f_button_reduceSFM` |
| `pySAview/core/reduction/pysared.py` | Per-scan `ReductionParams` dataclass, pySAred 1.5.3 normalization/error pipeline, and curve trimming | `f_button_reduceSFM` |
| `pySAview/core/export_tables.py` | Tab-delimited reflectivity export (`ResolutionSigma` + metadata) and pySAred 3/4-column `.dat` writer | `f_button_reduceSFM` `.dat` writer |

## Pipeline order

For each checked scan in **Scan Charts**, reduction runs in the exact order used by
pySAred 1.5.3:

1. **ROI & curvature integration** (`pySAview/core/reduction/curvature.py`)
   - When `curvature_rad == 0`, signal counts come from the selected channel (e.g.
     `ROI_Counts`) and, when enabled, background counts come from integrating the
     2nd Background ROI over the same Y rows.
   - When `curvature_rad != 0` and plotting `ROI_Counts`, the Y-integrated detector
     map is sheared **per `Flipper1` spin state** (excluding unwritten frames) inside
     the signal ROI columns — and independently inside the BKG ROI columns when enabled
     — before summing across X.
2. **Sample angle offset**
   - `θ_corr = θ − θ₀` (degrees). Both `Q = (4π / λ) sin(θ_corr)` and displayed `th`
     use `θ_corr`.
3. **Background subtraction** (`pySAview/core/reduction/background.py`)
   - Applied **before** monitor or dwell-time normalization, matching pySAred 1.5.3.
   - For points with `Qz > bkg_skip_qz` and `I_bkg > 0`:
     ```
     dI = sqrt(I + I_bkg)
     I  = I - I_bkg
     ```
   - Points with `Qz <= bkg_skip_qz` or `I_bkg <= 0` are left unchanged.
4. **Monitor / dwell-time normalization** (`pySAview/core/reduction/pysared.py`)
   - Uses the pySAred 1.5.3 counting error convention (`dI = 1` when `I == 0`,
     `sqrt(I)` when `I > 0`), propagated through monitor counts or dwell time.
     See [Normalization and uncertainty](normalization.md).
5. **Overillumination (footprint) correction** (`pySAview/core/reduction/overillumination.py`)
   - Computes the trapezoidal beam footprint factor `c₀ ∈ (0, 1]` and projected slit
     width `fwhm_proj` from per-step `s1h`/`s2h` (falling back to Nomad
     `entry0/SUPERADAM/VirtualSlit/{s1h,s2h}_actual_width`), sample length `L`,
     and slit-to-sample distances `D1 = 2300 mm` and `D2 = 290 mm`.
   - When enabled, both intensity and uncertainty are divided by `c₀`:
     `I ← I / c₀`, `dI ← dI / c₀`.
   - Optionally overlays `c₀` on the chart as a dashed amber curve.
6. **Manual scale factor**
   - When enabled, divides `I` and `dI` by `scale_factor > 0`. Direct-beam
     normalization is intentionally deferred until verified direct-beam reference
     datasets are available.
7. **Instrumental resolution** (`pySAview/core/reduction/resolution.py`)
   - **pySAred (default)**:
     ```
     d_alpha = arctan((s1h + fwhm_proj) / (2 * (D1 - D2)))
     Res_FWHM(Qz)  = sqrt(k0² * (cos²(θ) * d_alpha² + (Δλ/λ)² * sin²(θ)))
     Res_FWHM(rad) = d_alpha
     Res_FWHM(deg) = degrees(d_alpha)
     σ = Res_FWHM / (2 * sqrt(2 * ln 2))
     ```
     where `k0 = 4π / λ` and `fwhm_proj` is the footprint-clipped slit width when
     overillumination is active (otherwise `s2h`).
   - **Original Gunnar `Sared` (`0.68`)**:
     ```
     Res_FWHM(Qz) = sqrt(k0² * cos²(θ) * 0.68² * (s1h² + s2h²) / (D1 - D2)² + (Δλ/λ)² * Qz²)
     σ = Res_FWHM / (2 * sqrt(2 * ln 2))
     ```
8. **Per-curve point trimming** (`trim_curve_mask`)
   - Optionally skips `skip_left` leading points, `skip_right` trailing points,
     and/or `I == 0` points independently on each sorted spin curve.

## Sample curvature equivalence & safety

In `pySAred 1.5.3`, shearing an `(N, W)` ROI slice by angle `γ` (radians) shifts column
`x` along the scan-point axis by `x * tan(γ)`, anchors the origin at
`zero_level = round((W / 2) * tan(γ) - min_y)`, and linearly interpolates onto
`y_target = min_y + arange(zero_level, zero_level + N)` with `0` outside support.

Because the sheared rectangle is a convex parallelogram on integer `x` columns,
`pySAview/core/reduction/curvature.py` evaluates exact per-column 1D linear interpolation
(`np.interp`) onto `y_target`. This agrees with `scipy.interpolate.griddata(..., method="linear")`
to `<= 1.4e-14` everywhere `griddata` succeeds, avoids `pySAred 1.5.3`'s off-by-one
`np.mgrid` shape bug at specific `(W, γ)` combinations, and never mutates the underlying
cached detector map — setting `γ = 0` always restores the exact original data.

On polarized scans with interleaved `Flipper1` frames, `apply_curvature_by_spin` separates
`Flipper1 == 0` and `Flipper1 == 1` (ignoring unwritten frames) before shearing so
opposite spin states are never interpolated together.

## Background ROI geometry

In **Detector Image**, enabling **Background ROI (pySAred 1.5.3)** displays a dashed cyan
box locked to the Signal ROI's vertical `[y0, y1)` range and pixel width `w = x1 - x0`.
Its default X placement is `[x0 - 2w, x0 - w]` (one ROI-width gap to the left of the
signal ROI, falling back to `[x1 + w, x1 + 2w]` near the left edge of the detector).
Adjusting or toggling the Background ROI updates **Scan Charts** and draws matching
dashed cyan bounds on **2D Map**. Overlapping the Signal and Background ROIs
displays an explicit warning.

## Exporting pySAred `.dat` files

**File ▸ Export ▸ pySAred Reflectivity (.dat)…** (or **Export pySAred .dat…** in the
**Scan Charts ▸ Reduction** tab) writes space-delimited, headerless `.dat` files matching
pySAred 1.5.3:

- 4 columns when **Include resolution σ column** is enabled: `angle  I  dI  Resolution_sigma`
- 3 columns when disabled: `angle  I  dI`
- Selectable angle coordinate: **Qz (Å⁻¹)**, **Degrees (°)**, or **Radians (rad)**
- Negative intensities are omitted (matching pySAred 1.5.3); when multiple spin states or
  scans are plotted, one `.dat` file is written per curve with a sanitized suffix.
- Every output path is checked by `validate_output_path` so no loaded `.nxs`/`.h5` source
  file can ever be overwritten.
