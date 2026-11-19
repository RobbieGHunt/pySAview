# Normalization and uncertainty

pySAview is a viewer and data-reduction interface for ILL SuperADAM scan channels,
aligned with **pySAred v1.5.3** (`pySAred_V1.5.3.py`). Monitor-normalized intensity
alone is not absolute reflectivity: direct-beam normalization is not implied unless
applied explicitly (see [Data reduction](reduction.md) for overillumination,
sample curvature, background ROI subtraction, angle offset, resolution, and `.dat`
export).

## Definitions

For unweighted detector or ROI counts `N`, simultaneous monitor counts `M`,
and dwell time `t` in seconds:

| Mode | Value | Units |
| --- | --- | --- |
| Raw | `N` | counts |
| Time | `N / t` | counts/s |
| Monitor | `N / M` | detector counts per monitor count |
| Monitor norm. cps (monitor ratio × fixed reference rate) | `r_ref * N / M` | counts/s at the reference monitor rate |

Monitor counts already integrate incident flux over the exposure. Dividing
`N / M` by `t` again would double-normalize exposure and is not done.
The last mode uses a user-specified **fixed, exact display scale** `r_ref`,
in monitor counts/s. The same value applies to every overlaid file.
It does not estimate `mean(M)` or `sum(M) / sum(t)` separately for each file.
No uncertainty is assigned to this chosen display convention. If a physical
reference-rate calibration has uncertainty, that needs a separate model.

`monitor_time` and `monitor_cps` are internal names retained for the last
mode; it requires monitor counts but not a second division by dwell time.

## pySAred 1.5.3 uncertainty convention (Scan Charts)

**Scan Charts** uses the counting-uncertainty and propagation rules from
`pySAred 1.5.3` (`pySAview/core/reduction.py`):

1. **Raw count uncertainty**:
   ```
   dI = 1           when I = 0
   dI = sqrt(I)     when I > 0
   ```
2. **Background subtraction** (applied **before** monitor or dwell-time division,
   for points with `Qz > bkg_skip_qz` and `I_bkg > 0`):
   ```
   dI = sqrt(I + I_bkg)
   I  = I - I_bkg
   ```
3. **Exposure propagation**:
   ```
   Raw:      val = I,            sigma = dI
   Time:     val = I / t,        sigma = dI / t
   Monitor:  val = I / M,        sigma = sqrt((dI / M)**2 + I**2 / M**3)
   Ref rate: val = r_ref * I/M,  sigma = r_ref * sqrt((dI / M)**2 + I**2 / M**3)
   ```
   For `I > 0`, `sqrt((dI / M)**2 + I**2 / M**3)` is algebraically identical to
   pySAred's `(I / M) * sqrt((dI / I)**2 + 1 / M)` while avoiding division by zero
   at `I = 0` and staying nonnegative if background subtraction yields `I < 0`.
4. **Overillumination and manual scale factor**:
   When enabled, both `val` and `sigma` are divided by `c₀` and/or `scale_factor`.
5. **Display Sigma selector**:
   The **Sigma** selector in Scan Charts defaults to **1σ** (matching pySAred 1.5.3)
   and scales the plotted symmetric bounds to `value ± selected_sigma * sigma` for
   `1σ`–`6σ`. The numerical `StdUncertainty` table/export column and pySAred `.dat`
   `dI` column always retain the **1σ** standard uncertainty.

Independent supplied variance arrays (`ScanVariable.variance`) override the raw
count rule using first-order propagation:

```
var(N)       = v_N
var(N / t)   = v_N / t**2
var(N / M)   = v_N / M**2 + N**2 * v_M / M**4
var(r*N / M) = r**2 * var(N / M)
```

Supplied errors are squared once; supplied variances are not squared again.
If the selected numerator is `Monitor1` itself, the shared measurement cancels:
`M/M = 1` (or `r_ref` in reference-rate mode), with zero statistical uncertainty
for positive valid `M`. It is never treated as two independent observations.

## Exact counting confidence intervals (`core/normalization.py`)

For standalone statistical analysis and **Off-Specular** normalization,
`pySAview/core/normalization.py` retains exact Garwood Poisson limits and
Clopper-Pearson Poisson-ratio limits at Gaussian-equivalent coverage
`confidence = erf(sigma / sqrt(2))` (`1σ`–`6σ`):

```
lower = 0                               when N = 0
lower = 0.5 * chi2.ppf(alpha/2, 2*N)     otherwise
upper = 0.5 * chi2.isf(alpha/2, 2*(N+1))
```

For independent detector and monitor Poisson counts measured simultaneously,
conditioning on `N+M` yields a binomial proportion with Clopper-Pearson limits
transformed via `p/(1-p)`. Exact intervals apply only to finite, nonnegative
integer, unweighted counts without background subtraction.

## Validation and display

- Missing channels, mismatched shapes, and incompatible overlay units skip the
  affected scan and explain why. No Detector1 or raw-count fallback is made.
- Zero/negative/nonfinite denominators and invalid Poisson observations are
  excluded, with reasons in the warning line under the chart title and in
  the export header.
- Non-count channels can be plotted raw. They receive no invented Poisson
  uncertainty; supplied variances remain available.
- SuperADAM files can omit the unit for `Count`, with both label and property
  set to `Time`. For that specific instrument/schema combination the reader
  interprets exposure as seconds, and the processing status records the unit
  assumption. Other unknown exposure units are rejected.
- Logarithmic axes cannot show zero/negative values or endpoints. Such values
  remain in the table/export when scientifically valid. Nonpositive lower
  bounds are omitted from the log rendering, never replaced or clipped. Use
  a linear axis or the table to inspect the complete interval.
- The chart shows only warnings under its title. The full processing
  description (formula, interval method and sigma level, excluded points,
  waterfall settings) is written to the export header; the table's
  *Uncertainty method* column shows each curve's method.
- **File ▸ Export ▸ Reflectivity Data** writes tab-delimited values, standard
  uncertainties and lower/upper bounds per point with the original step index,
  and `#` header lines with method, units, wavelength (when applicable),
  reference rate, source paths and processing warnings, at full numerical
  precision (shortest round-trip representation).

## References

- [Mantid ILL reflectometry preprocessing](https://docs.mantidproject.org/nightly/algorithms/ReflectometryILLPreprocess-v1.html): normalization by time or monitor counts.
- [Mantid error propagation](https://docs.mantidproject.org/nightly/concepts/ErrorPropagation.html): independent-observation propagation; the formulas above avoid the removable singularity at a zero numerator.
- [Thulin and Zwanzig, Exact confidence intervals and hypothesis tests for parameters of discrete distributions](https://arxiv.org/abs/1412.0442): Garwood and Clopper-Pearson intervals.
- [SciPy exact binomial proportion intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).

Regression tests include analytic zero-count limits, explicit variance formulas,
count-rate scaling, shared-monitor cancellation, and coverage checks using the
Poisson/binomial sampling distributions.

## Detector ROI pixel selection

ROI bounds are expressed in displayed detector coordinates (`displayed_frame = raw_frame.T`). Bounds round to the nearest integer, with ties to even, and clip to the actual detector dimensions. Slices include the lower bound and exclude the upper bound. No fractional pixel weighting is used. Reversed endpoints are accepted; empty or wholly outside regions contain zero counts. Preview, profiles, fixed scan integration and tracked integration share this convention. Tracked integration requires exactly one ROI per frame; missing frame definitions must not be silently filled with zero counts. Original measurement files remain read-only.

Gaussian-equivalent coverage follows the [normal distribution CDF](https://docs.scipy.org/doc/scipy/tutorial/stats/continuous_norm.html); Poisson limits still use the exact count construction above.

The Sigma selector applies to both exact and propagated display modes. The numerical StandardUncertainty column always retains the 1σ standard uncertainty, while LowerBound/UpperBound and their method describe the selected display width.
