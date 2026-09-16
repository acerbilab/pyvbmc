# Uniform box sampling and noisy Rosenbrock GP conditioning

The box proposal in PyVBMC's acquisition sieve used standard-normal draws
where MATLAB VBMC uses uniform draws. The correction restores uniform
sampling inside the proposal box. It affects searches that generate box
candidates (`box_search_frac` defaults to 0.25), including noiseless
targets, and changes the random stream and subsequent optimization
trajectories.

## Source comparison

MATLAB's `getSearchPoints` draws `rand(Nbox,D)` and scales and translates
the result to `[box_lb, box_ub]` ([source at `396d649c`, line 624](https://github.com/acerbilab/vbmc/blob/396d649c3490f1459828ac85f552482869edf41c/private/activesample_vbmc.m#L624)).
PyVBMC's `_get_search_points` used
`standard_normal((N_box, D)) * (box_ub - box_lb) + box_lb`. The normal
proposal is centered on the lower edge, with standard deviation equal to
the box width. Before clipping to the global search bounds, about 88% of
its two-dimensional draws lie outside the proposal box. The bounds of
that box depend on the live training inputs and the current search limits.

The error entered with the original Python implementation in `2551469`
(2021-08-03). The [MATLAB version predating that port](https://github.com/acerbilab/vbmc/blob/20445308b2957fc4756e51a3e418bd4cb57bf4e9/private/activesample_vbmc.m)
also uses `rand(Nbox,D)`. The generator migration in `d02c517` preserved the normal
distribution. The fix in `pyvbmc/vbmc/active_sample.py` replaces that draw
with `rng.random((N_box, D))`.

The former box test mocked the normal draws and checked their affine
transformation. Its replacement draws from an explicitly seeded generator
and checks support, mean and quartiles against the uniform distribution.
It covers wide finite search bounds, bounds that restrict the training
box, the unbounded-search fallback, and exclusion of inactive logger rows.
All three parameterizations fail against the original sampler and pass
with the correction.

The unused outer search limits are also unused in MATLAB: its assignments
of `LB_searchmin` and `UB_searchmin` to the active search bounds are
commented out. The existing expanding search bounds are preserved.
Noise shaping remains an unported, default-off option, as recorded in the
[modernization discussion](../2026-09-02-modernization-discussion.md).

## Paired probe

`dev/scripts/gp_box_probe.py` runs the pool's
`rosenbrock_D2_noise3_svbmc` configuration at seeds 1000, 1001 and 1002,
once with each sampler. The normal arm uses the package archived from
`69a7a184`; the uniform arm uses the corrected checkout. Their numerical
sources differ only in `vbmc/active_sample.py`. Both use the same frozen
gpyreg 1.2.1 source and single-threaded BLAS. Each run starts in a fresh
process, with the pool's options and default evaluation budget, and all
six runs are retained without filtering or replacement. The seeds are the
three local pilot seeds, selected before either arm ran.

The probe wraps the target to record every call, including evaluations
later trimmed from the logger. After optimization, it measures each
recorded iteration's GP and the GP belonging to `results["best_iter"]`,
which supplies the returned posterior's final optimization. Conditioning
uses the campaign's `gp_condition_number`: the largest condition number
over hyperparameter samples, computed from the posterior factor. A tail
observation is a log density more than 1000 below the maximum in the
specified set. Full-call diagnostics use original coordinates and noisy
target values; GP diagnostics use the transformed training targets.

The baseline's seed 1000 reproduces the historical pilot's minimum target
value exactly. The paired experiment uses fresh runs of both arms so
that the comparison isolates the sampler within the current source and
dependency environment. Three seeds provide a diagnostic comparison;
they do not establish population-level changes in accuracy or robustness.
All three pairs have identical initial designs: the first ten target-call
coordinates and observed values match exactly.

The raw calls, complete VBMC saves, diagnostic JSONs, frozen baseline and
execution script are listed in `dev/scripts/runs/LOCAL.md`. The tracked
diagnostics and provenance are under `dev/experiments/gp_box_20260916/`.

## Results

All six runs report convergence. Total optimization time was 898 seconds,
excluding post-run diagnostics. Minimum log density and tail counts below
refer to every target call; conditioning refers to the returned posterior's
GP. Evidence error is `abs(ELBO - ln Z)`; gsKL and MMTV use the benchmark
suite's definitions and independent diagnostic random streams.

| Seed | Sampler | Calls | Minimum log density | Tail calls | GP condition | Evidence error | gsKL | MMTV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | Normal | 170 | -68064 | 17 | 7.82e12 | 0.022 | 0.036 | 0.083 |
| 1000 | Uniform | 150 | -257529 | 22 | 9.67e13 | 0.261 | 0.312 | 0.119 |
| 1001 | Normal | 180 | -143043 | 22 | 2.53e13 | 0.034 | 0.144 | 0.100 |
| 1001 | Uniform | 195 | -64983 | 19 | 7.16e12 | 0.099 | 0.003 | 0.028 |
| 1002 | Normal | 205 | -227002 | 32 | 1.06e14 | 0.272 | 0.026 | 0.030 |
| 1002 | Uniform | 170 | -102374 | 26 | 1.79e13 | 0.382 | 0.028 | 0.069 |

The corrected sampler retains the GP failure mode on all three seeds.
Its final condition numbers span 7.16e12 to 9.67e13, with 19–26 tail calls
per run and minimum log densities between -257529 and -64983. Conditioning
improves on two seeds and worsens on one; posterior metrics move in both
directions. Evidence error increases on all three pairs, from a median
0.034 to 0.261 nats. This small probe cannot distinguish a systematic
accuracy effect from the change in random trajectories.

The histories locate the onset rather than just showing a poor final
matrix. For normal seed 1000, the first tail evaluation is call 149;
the GP condition rises from 4.39e5 at iteration 25 to 7.82e12 at iteration
34. With uniform proposals, the first tail call is 80, and conditioning
reaches 9.67e13. Uniform seeds 1001 and 1002 first enter the same tail
category at calls 134 and 98. The corresponding normal runs do so at
calls 107 and 131. Thus the sampler correction does not consistently
delay entry into the tails.

## Implication for the investigation

Uniform box sampling is the intended proposal and a confirmed porting-bug
fix. It does not resolve the noisy unbounded-target GP issue. The next
diagnostic should examine the acquisition immediately before the first
tail evaluations: whether a distant point offers a credible reduction in
posterior uncertainty, whether its ranking is sensitive to numerical
cancellation, and how the ensuing hyperparameter fit changes. The saved
histories and full call records support that investigation. Noise shaping
and additional limits on exploration remain separate algorithmic choices;
the 1.5 scope decision keeps those deferred unless another bug is found.

## Validation

- The three box-distribution regression cases fail with the normal
  sampler and pass with uniform draws.
- Fixed-input oracle tests: 136 passed, 14 skipped, with the eight
  active-sampling cases deselected. All acquisition oracles pass.
- Seven fixtures' `active_sample_step` references were updated from their
  stored states with the targeted generator mode and a recorded reason.
  A comparison with the archived originals confirms that every other
  array is bit-identical, including the GP and variational states.
- `make_oracle_fixtures.py --check --exact`: all 11 fixture groups pass,
  including the GP-fit history fixtures. The original references remain
  available at `69a7a184` and in the frozen baseline archive.
- Full package suite, `pytest --reruns=5 -x -q`: 1264 passed, 47 skipped,
  no reruns, 324 seconds with single-threaded BLAS and gpyreg 1.2.1.
  Pytest used a fresh workspace temporary directory because its default
  temporary directory was not writable in the sandbox.
- Standard golden replay, seed 0 on `normal_D5`, `banana_D2`,
  `halfnormal_D2`, `cigar_D4` and `rosenbrock_D2_noise1`: zero flags in
  five runs. All initial designs match exactly; trajectories diverge at
  iteration 1, and all final evidence-error, gsKL and MMTV values lie
  within the reference population's `Q3 + 3 IQR` fences. The
  [replay report](../experiments/gp_box_20260916/replay/replay.md) retains
  the paired metrics and the harness's limits. The golden references
  were preserved; this five-run check is separate from the final 1.5
  release gate on fresh cluster pools.
