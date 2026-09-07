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

**`item7_20260906`: 810 runs across 17 configurations.** Sixteen configurations
use seeds 0–49; the 15-dimensional budget-exhaustion case uses seeds 0–9.
The [results summary](baseline/summary.md) gives the measured outcomes for
every configuration.

**Approved next extension (2026-09-07; not yet run):** add
`rosenbrock_D2_noise3`, `student_D8_noise3`, and `lumpy_D10_noise3` to the
standard benchmark set, each at seeds 0–49. These 150 runs will extend this
reference to 960 runs across 20 configurations, including 250 noisy runs
across five configurations. Existing reference pairs remain unchanged.
Preparation and validation are recorded in
[`latent-bug-fixes.md`](../plans/latent-bug-fixes.md).

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
| `cigar_D15_exhaust` | 15 | The late GP regime under a fixed 750-evaluation budget |

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
fits. All 810 reference runs completed; the ten D15 cases reached their
intended budget limit.

The population check applies two-sample Kolmogorov–Smirnov tests to
`elbo_err`, `gskl`, `mmtv` and evaluation count, with a Holm correction at
alpha 0.05 across the family (68 tests for all 17 configurations). A flag
means a distribution changed and needs investigation; it does not establish
that the change is a regression. No flag is not proof of equivalence.
Wall time is recorded but is not a pass/fail metric. These runs assess
inference behavior; controlled speed comparisons need dedicated benchmarks.

## Files and checks

- [`baseline/`](baseline/) contains the 810 JSON sidecars and summary, tracked
  in Git. Each sidecar records the target, seed, requested/effective options,
  code and dependency versions, and final metrics.
- Full `.npz` traces contain per-iteration data and final arrays. They remain
  local under `dev/scripts/runs/golden/item7_20260906/` (gitignored). Copy that
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
validation reports are in the [execution record](extension_20260907/README.md).
The legacy `regenerate_baseline.sh` uses a different seed allocation and
masks comparison failures; use the recorded procedure when reproducing this
snapshot.
