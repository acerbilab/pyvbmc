# Golden runs: the PyVBMC regression reference

The golden runs are saved, seeded, end-to-end PyVBMC runs on a collection
of benchmark targets. They let us check whether a code change alters the
solver's path and whether it changes inference quality across many seeds.
Each benchmark has reference evidence and posterior information, obtained
analytically or by independent numerical calculation, against which we
measure the fitted result.

There are two complementary checks:

- **Population comparison:** compare final results across seeds to detect
  changes in accuracy or the number of model evaluations.
- **Trajectory replay:** rerun selected seeds and compare their initial
  designs, sampled points and per-iteration ELBOs with the saved traces.

The reference records the behavior of a particular implementation. It is
not a guarantee that every fit is accurate; known bugs may be present, and
an intentional correctness fix may change its results.

## Current reference

**`reference_990_20260912`: 990 runs across 23 configurations.** Sixteen
configurations use seeds 0–49, six use seeds 0–29 (two noisy additions of
2026-09-07 and four real-data configurations of 2026-09-12), and the
15-dimensional budget-exhaustion case uses seeds 0–9. The population includes
250 noisy runs across seven configurations and 92 KS tests. The
[results summary](baseline/summary.md) gives the measured outcomes for every
configuration.

The four real-data configurations are the Bayesian timing model and the
multisensory causal-inference model on two subjects from the 2020
noisy-VBMC paper, on exported data with spline-trapezoidal priors and the
paper's plausible boxes, defined in
[benchmark-realistic-targets.md](../plans/benchmark-realistic-targets.md):
subject 1 noiseless (`multisensory_s1_D6`) and all three at the paper's
noise levels (`timing_D5_noise2.2`, `multisensory_s1_D6_noise1.3`,
`multisensory_s2_D6_noise1.3`) with its budget of 50 (D + 2) evaluations.
Their ground truths are importance-weighted populations under
`dev/scripts/data/truths/`. The 120 runs were generated from the frozen
checkout `fc50ee1` with gpyreg `a2f8ddc`, whose default-path numerics equal
the frozen treatment `68a43db` of the population campaigns; all produced
complete result pairs, 119 converged and one timing run reached its budget.
Usable counts are 28/30 for noiseless subject 1, 5/30 and 22/30 for the
noisy subjects 1 and 2, and 28/30 for timing; the returned posteriors are
under-dispersed on every real-data target, timing's `w_s` and `sigma_p` by
about 30 %. The record is
[`realdata_extension_20260912`](realdata_extension_20260912/README.md).

The 60 noisy additions of 2026-09-07 are `rosenbrock_D2_noise3` and
`student_D8_noise3`, each at seeds 0–29, with noise SD 3 and evaluation
budgets 200 and 500, generated from pinned source `623f5cd` with gpyreg
`a2f8ddc`; 53 converged and seven noisy Rosenbrock runs reached their
budget, with usable counts 23/30 and 14/30 and a median absolute ELBO error
of 1.08 for Student. The reference records these outcomes rather than
treating every completed run as an accurate fit. `lumpy_D10_noise3` remains
registered but has no reference traces.

The historical 810 pairs remain byte-identical. Their execution record is
[`extension_20260907`](extension_20260907/README.md); the noisy additions
and the 870-run reference are recorded in
[`noisy_extension_20260907`](noisy_extension_20260907/README.md). All 990
archives passed integrity checks and the 92-test even/odd comparison had no
flags. The real-data configurations replay bit-exactly with the current
code; the older configurations' traces predate the 1.5 numerics, so their
replays part early and only the accuracy envelopes apply. No trace stores
the returned posterior's transformer, so replay reports that field as
uncertifiable.

| Target | Dimensions | What it exercises |
|---|---|---|
| `normal` | 5 | Independent Gaussian parameters with different scales |
| `corr` | 5 | Correlated Gaussian parameters |
| `halfnormal` | 2 | A posterior constrained to positive parameters |
| `rosenbrock` | 2 | A curved, narrow posterior |
| `banana` | 2, 6, 10 | Nonlinear dependence across increasing dimensions |
| `cigar` | 4, 8 | A strongly elongated, rotated posterior |
| `lumpy` | 4, 10 | A Gaussian-mixture target with multiple components |
| `student` | 4, 8 | Student-t likelihoods with broad Gaussian priors |
| `logreg` | 5 | Logistic regression with bounded parameters |
| `rosenbrock_D2_noise1` | 2 | Noisy log-density evaluations, noise SD 1 |
| `logreg_D5_noise3` | 5 | Bounded parameters and noisy evaluations, noise SD 3 |
| `rosenbrock_D2_noise3` | 2 | Harder noisy curved posterior, noise SD 3 |
| `student_D8_noise3` | 8 | Heavy-tailed posterior and noisy evaluations, noise SD 3 |
| `cigar_D15_exhaust` | 15 | The late GP regime under a fixed 750-evaluation budget |
| `multisensory_s1` | 6 | Real-data causal-inference likelihood, subject 1: noiseless and with noise SD 1.3 |
| `multisensory_s2` | 6 | The same model on subject 2, with noise SD 1.3 |
| `timing` | 5 | Real-data Bayesian timing likelihood with noise SD 2.2 |

A run's seed controls the solver and separate streams for the starting point
and, where applicable, evaluation noise. Starting points are drawn uniformly
inside the plausible bounds. Target definitions, bounds, reference values
and configuration options are in
[`benchmark_targets.py`](../scripts/benchmark_targets.py).

The D15 case disables early convergence termination so it always uses its
750 evaluations. Its ten seeds provide coverage of this expensive regime;
they support less precise statistical comparisons than the 50-seed cases.

## Reading the results

Most entries in the summary are **median [25th percentile, 75th percentile]**.
Smaller values are better for the three accuracy metrics:

| Metric | Meaning |
|---|---|
| `elbo_err` | Absolute difference between the estimated and reference log evidence |
| `gskl` | Gaussianized symmetrized KL divergence: agreement of posterior means and covariances |
| `mmtv` | Mean marginal total variation distance: agreement of the one-dimensional posterior marginals |
| `evals` | Number of target evaluations used |

`gskl` and `mmtv` capture different aspects of a posterior; neither alone
certifies its full joint shape. The summary also reports posterior-mean
RMSE, iterations, mixture size, warps and wall time. `usable` is the fraction
meeting all three accuracy thresholds: `elbo_err < 1`, `gskl < 1`, and
`mmtv < 0.2`. `failed` counts execution failures, not unconverged or inaccurate
fits. All 990 reference runs completed. Of these, 972 converged and 18 reached
their budgets: the ten intentional D15 exhaust cases, seven noisy Rosenbrock
runs and one noisy timing run.

The population check applies two-sample Kolmogorov–Smirnov tests to
`elbo_err`, `gskl`, `mmtv` and evaluation count, with a Holm correction at
alpha 0.05 across the family (92 tests for all 23 configurations). A flag
means a distribution changed and needs investigation; it does not establish
that the change is a regression. No flag is not proof of equivalence.
Wall time is recorded but is not a pass/fail metric. These runs assess
inference behavior; controlled speed comparisons need dedicated benchmarks.

## Files and checks

- [`baseline/`](baseline/) contains the 990 JSON sidecars and summary, tracked
  in Git. Each sidecar records the target, seed, requested/effective options,
  code and dependency versions, and final metrics.
- Full `.npz` traces contain per-iteration data and final arrays. They remain
  local under `dev/scripts/runs/golden/reference_990_20260912/` (gitignored). Copy that
  directory from the generating laptop for exact replay elsewhere. Publishing
  a downloadable trace archive is planned for the 1.5 release.

From the repository root, compare a new population with the reference:

```console
python dev/scripts/golden_trace.py compare dev/golden/baseline <new_run_directory>
```

This needs only the JSON sidecars. Check matching configuration/seed coverage,
complete JSON/NPZ pairs and absence of error files separately: the statistical
comparison alone does not certify completeness.

With the reference traces available, run the default five-case replay:

```console
python dev/scripts/golden_replay.py
```

The default cases cover a Gaussian, banana, bounded half-normal, cigar and
noisy Rosenbrock at seed 0. The report shows where trajectories diverge and
checks changed runs against the reference population's accuracy ranges.
Both commands return nonzero when their checks flag a problem.

## Code and reproducibility

Runs use one worker, single-threaded BLAS, and `vectorized_target=False`.
Reproducing a stored trajectory requires its recorded code, seed, options,
dependency versions and numerical platform. These are part of the reference;
matching the seed alone is insufficient.

Exact environment information, reproduction commands, file hashes and
validation reports are in the [810-run record](extension_20260907/README.md),
the [60-run integration record](noisy_extension_20260907/README.md) and the
[120-run real-data record](realdata_extension_20260912/README.md).
The legacy `regenerate_baseline.sh` uses a different seed allocation and
masks comparison failures; use the recorded procedure when reproducing this
snapshot.
