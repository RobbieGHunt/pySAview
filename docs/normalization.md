# Normalization and uncertainty

pySAview is a viewer for SuperADAM scan channels. Monitor-normalized intensity
is not absolute reflectivity: no direct-beam, footprint, transmission, detector
efficiency, dead-time, or polarization correction is implied by these controls.

## Definitions

For unweighted detector or ROI counts `N`, simultaneous monitor counts `M`,
and dwell time `t` in seconds:

| Mode | Value | Units |
| --- | --- | --- |
| Raw | `N` | counts |
| Time | `N / t` | counts/s |
| Monitor | `N / M` | detector counts per monitor count |
| Monitor ratio × fixed reference rate | `r_ref * N / M` | counts/s at the reference monitor rate |

Monitor counts already integrate incident flux over the exposure. Dividing
`N / M` by `t` again would double-normalize exposure and is not done.
The last mode uses a user-specified **fixed, exact display scale** `r_ref`,
in monitor counts/s. The same value applies to every overlaid file.
It does not estimate `mean(M)` or `sum(M) / sum(t)` separately for each file.
No uncertainty is assigned to this chosen display convention. If a physical
reference-rate calibration has uncertainty, that needs a separate model.

`monitor_time` and `monitor_cps` are internal names retained for the last
mode; it requires monitor counts but not a second division by dwell time.

## Standard uncertainties

Unweighted Poisson observations have plug-in variance estimates `v_N = N`
and `v_M = M`. Independent supplied variance arrays can replace these
estimates for weighted/processed observations. Supplied errors are squared
once; supplied variances are not squared again.

Using first-order derivatives for independent observations:

```
var(N)       = v_N
var(N / t)   = v_N / t**2
var(N / M)   = v_N / M**2 + N**2 * v_M / M**4
var(r*N / M) = r**2 * var(N / M)
```

Dwell time is treated as known exposure. The ratio formula is evaluated
directly, without relative-error divisions by `N` or an arbitrary count floor.
If the selected numerator is Monitor1 itself, the shared measurement cancels:
`M/M = 1` (or `r_ref` in reference-rate mode), with zero statistical uncertainty
for positive valid `M`. It must not be treated as two independent observations.

These standard uncertainties are estimates and first-order propagation is
an approximation; it is **not** an exact low-count confidence interval.
At `N=0`, the plug-in Poisson standard uncertainty is zero, while the exact
confidence interval has a nonzero upper bound. Both are identified separately
in the table and CSV, so zero standard uncertainty cannot be mistaken for
proof that the underlying intensity is exactly zero.

## Exact counting confidence intervals

The default display uses equal-tailed **68.26894921370859% confidence limits**.
These are confidence intervals, not symmetric `±1σ` error bars. Because counts
are discrete, coverage is at least nominal and can exceed it.

For raw counts, Garwood limits on the Poisson mean are:

```
lower = 0                               when N = 0
lower = 0.5 * chi2.ppf(alpha/2, 2*N)     otherwise
upper = 0.5 * chi2.isf(alpha/2, 2*(N+1))
```

Divide these bounds by the known exposure for time-normalized rates.
For zero counts, the raw interval is `[0, 1.841021645...]`.

For independent detector and monitor Poisson counts measured simultaneously,
conditioning on `N+M` gives a binomial observation with
`p = lambda_N / (lambda_N + lambda_M)`. Compute exact Clopper-Pearson
limits for `p`, then transform using `p/(1-p)` to obtain limits on the ratio
of Poisson means. Scale these limits by the fixed reference rate when selected.
Thus uncertainty in the monitor is included; it is not treated as exact.

Exact intervals apply only to finite, nonnegative integer, unweighted counts
with an independent Poisson monitor. Explicit variance arrays use propagated
standard uncertainty, and the banner says so even if the exact option is selected.
Background-subtracted data, detector/monitor covariance other than the identical
Monitor1 channel, weighted pixels, dead time, and systematic calibration errors
need additional models; no exact Poisson claim is made for them.

## Validation and display

- Missing channels, mismatched shapes, and incompatible overlay units skip the
  affected scan and explain why. No Detector1 or raw-count fallback is made.
- Zero/negative/nonfinite denominators and invalid Poisson observations are
  excluded, with reasons in the processing banner and CSV warning column.
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
- CSV exports values, standard uncertainties, lower/upper bounds, method,
  units, wavelength (when applicable), reference rate, source path, original
  step index, and processing warnings at full numerical precision.

## References

- [Mantid ILL reflectometry preprocessing](https://docs.mantidproject.org/nightly/algorithms/ReflectometryILLPreprocess-v1.html): normalization by time or monitor counts.
- [Mantid error propagation](https://docs.mantidproject.org/nightly/concepts/ErrorPropagation.html): independent-observation propagation; the formulas above avoid the removable singularity at a zero numerator.
- [Thulin and Zwanzig, Exact confidence intervals and hypothesis tests for parameters of discrete distributions](https://arxiv.org/abs/1412.0442): Garwood and Clopper-Pearson intervals.
- [SciPy exact binomial proportion intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).

Regression tests include analytic zero-count limits, explicit variance formulas,
count-rate scaling, shared-monitor cancellation, and coverage checks using the
Poisson/binomial sampling distributions.
