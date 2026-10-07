# Golden runs: the PyVBMC regression reference

The golden runs are saved, seeded, end-to-end PyVBMC runs on a collection of
benchmark targets. They let us check whether a code change alters the solver's
path and whether it changes inference quality across many seeds. Each benchmark
has reference evidence and posterior information, obtained analytically or by
independent numerical calculation, against which we measure the fitted result.

There are two complementary checks:

- **Population comparison:** compare final results across seeds to detect
  changes in accuracy or the number of model evaluations.
- **Trajectory replay:** rerun selected seeds and compare their initial
  designs, sampled points and per-iteration ELBOs with the saved traces.

The reference records the behavior of a particular implementation. It is not a
guarantee that every fit is accurate; known bugs may be present, and an
intentional correctness fix may change its results.

## Current reference

**`reference_2400_20261007`: 2400 runs of the 24 configurations of the
`production` suite, at seeds 0–99.** 800 of them are noisy, across 8
configurations. The `production` suite is the `golden` suite at PyVBMC's
production defaults: its noiseless configurations are the golden ones, and its
noisy ones, under `production`-tagged labels, run at the package's defaults for
a specified-noise target (75 (D + 2) evaluations, VIQR) where the golden ones
pin the 2020 paper's budget of 50 (D + 2). The [results
summary](baseline/summary.md) gives the measured outcomes for every
configuration.

The runs were made on a Slurm cluster by the array mode of `population_run.py`,
one case per task, from PyVBMC `ff3ed014` with gpyreg `682585f7` (Python
3.12.14, NumPy 2.5.2, SciPy 1.18.1, cma 4.4.4). The campaign's redacted
records, with every case's completion record, sidecar and boost report, are
under
[`dev/experiments/release_gate_20261002/population_after`](../experiments/release_gate_20261002/population_after).
Its traces are in its archive, a draft release that only the repository's
collaborators see, which the [README of the release gate's
records](../experiments/release_gate_20261002/README.md) names. 2299 of the
2400 runs converged and 101 reached their budgets (the 100 runs of
`cigar_D15_exhaust`, by design, and 1 others: 1 of
`rosenbrock_D2_noise1_production`); 2157 meet the usability thresholds below.

The [promotion record](promotion_20261007/README.md) gives the assessment,
which compares the population seed by seed with the same runs of the code at
the start of the port correctness review (`f91fdf00`) and which the PI
accepted, and the provenance, hashes and validation. The 96-test even/odd
comparison of the population had no flags.

Exact replay depends on the machine and its BLAS, so the traces that
`golden_replay.py` compares with are not the cluster's: they are the
reference's replay fingerprints, one run at seed 0 of each of the 24
configurations, made with the same code on the machine that
`dev/scripts/runs/LOCAL.md` lists and judged against the population's accuracy
envelopes: 1 of them lies outside its configuration's accuracy envelope, which
the population's own rate of runs outside theirs (0.037) makes plausible. 0 of
the 24 equal the cluster's run of seed 0 in every semantic final field. The
port review's six seeded gate runs, two of them with a prior object
(`dev/scripts/seeded_gate_runs.py`), were recorded twice on that machine and
reproduced bit for bit. The five default cases replayed with identical
non-timer NPZ arrays, semantic final results and initial designs under the code
at promotion (2026-10-07). No trace stores the returned posterior's
transformer, so replay reports that field as uncertifiable.

The previous reference, `reference_990_20260913`, 990 runs of the `golden`
suite, is preserved with its [promotion record](promotion_20260913/README.md):
its sidecars are `dev/golden/baseline/` at commit `948bac0d`, its traces are
under `dev/scripts/runs/golden/reference_990_20260913/` on the machine that
`dev/scripts/runs/LOCAL.md` lists, and this file as it stood for it, whose
links resolve from `dev/golden/`, is
[`previous_reference_README.md`](promotion_20261007/previous_reference_README.md).

| Target | Dimensions | What it exercises |
|---|---|---|
| `normal` | 5 | Independent Gaussian parameters with different scales |
| `corr` | 5 | Correlated Gaussian parameters |
| `halfnormal` | 2 | A posterior constrained to positive parameters |
| `rosenbrock` | 2 | A curved, narrow posterior: noiseless, and with noise SD 1 and 3 |
| `banana` | 2, 6, 10 | Nonlinear dependence across increasing dimensions |
| `cigar` | 4, 8 | A strongly elongated, rotated posterior |
| `lumpy` | 4, 10 | A Gaussian-mixture target with multiple components; D = 10 also with noise SD 3 |
| `student` | 4, 8 | Student-t likelihoods with broad Gaussian priors; D = 8 also with noise SD 3 |
| `logreg` | 5 | Logistic regression with bounded parameters: noiseless, and with noise SD 3 |
| `cigar_D15_exhaust` | 15 | The late GP regime under a fixed 750-evaluation budget |
| `multisensory_s1` | 6 | Real-data causal-inference likelihood, subject 1: noiseless, and with noise SD 1.3 |
| `multisensory_s2` | 6 | The same model on subject 2, with noise SD 1.3 |
| `timing` | 5 | Real-data Bayesian timing likelihood with noise SD 2.2 |

The real-data configurations are the Bayesian timing model and the multisensory
causal-inference model on two subjects from the 2020 noisy-VBMC paper, on
exported data with spline-trapezoidal priors and the paper's plausible boxes,
defined in
[benchmark-realistic-targets.md](../plans/benchmark-realistic-targets.md), at
the paper's noise levels. Their ground truths are importance-weighted
populations under `dev/scripts/data/truths/`.

A run's seed controls the solver and separate streams for the starting point
and, where applicable, evaluation noise. Starting points are drawn uniformly
inside the plausible bounds. Target definitions, bounds, reference values and
configuration options are in
[`benchmark_targets.py`](../scripts/benchmark_targets.py).

The D15 case disables early convergence termination so it always uses its 750
evaluations, the one configuration that spends long in the regime where the GP
keeps a single hyperparameter sample.

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
certifies its full joint shape. The summary also reports posterior-mean RMSE,
iterations, mixture size, warps and wall time. `usable` is the fraction meeting
all three accuracy thresholds: `elbo_err < 1`, `gskl < 1`, and `mmtv < 0.2`.
`failed` counts execution failures, not unconverged or inaccurate fits.

The population check applies two-sample Kolmogorov–Smirnov tests to `elbo_err`,
`gskl`, `mmtv` and evaluation count, with a Holm correction at alpha 0.05
across the family (96 tests for all 24 configurations). A flag means a
distribution changed and needs investigation; it does not establish that the
change is a regression. No flag is not proof of equivalence. Wall time is
recorded but is not a pass/fail metric. These runs assess inference behavior;
controlled speed comparisons need dedicated benchmarks.

## Files and checks

- [`baseline/`](baseline/) contains the 2400 JSON sidecars and the summary,
  tracked in Git: the sidecars of the campaign's redacted records, copied byte
  for byte. Each sidecar records the target, seed, requested and effective
  options, the provenance of its run (the commits and import paths of its code,
  the imported versions) and final metrics.
- The population's `.npz` traces and boost captures are in the campaign's
  archive. The replay fingerprints' traces and the record of the gate runs are
  local under `dev/scripts/runs/golden/reference_2400_20261007_fingerprints/`
  (gitignored) on the machine that made them.

From the repository root, compare a new population with the reference:

```console
python dev/scripts/golden_trace.py compare dev/golden/baseline <new_run_directory>
```

This needs only the JSON sidecars. Check matching configuration/seed coverage,
complete JSON/NPZ pairs and absence of error files separately: the statistical
comparison alone does not certify completeness.

On the machine that holds the replay fingerprints, run the default five-case
replay:

```console
python dev/scripts/golden_replay.py
```

The default cases cover a Gaussian, banana, bounded half-normal, cigar and
noisy Rosenbrock at the production budget, at seed 0. The report shows where
trajectories diverge and checks changed runs against the reference population's
accuracy ranges. Elsewhere, replay at the parent commit first and pass that
replay's `--out` directory as `--baseline`. Both commands return nonzero when
their checks flag a problem.

A machine gets replay fingerprints of its own as the reference's were made,
with the reference's code and gpyreg: a clean checkout at a commit that holds
this reference and whose code is that of PyVBMC `ff3ed014`, which the first
command below confirms by printing nothing (the commit that published the
reference is one, the last to change `dev/golden/baseline/summary.md` while
this reference is current), with gpyreg `682585f7` installed from a checkout of
its own. From the repository root, one at a time:

```console
git diff --stat ff3ed014 HEAD -- pyvbmc ':(exclude)pyvbmc/testing' ':(exclude)pyvbmc/svbmc' dev/scripts/golden_trace.py dev/scripts/benchmark_targets.py dev/scripts/profile_run.py dev/scripts/data
python -u dev/scripts/golden_replay.py --seeds 0 \
    --sidecars dev/experiments/release_gate_20261002/population_after \
    --baseline dev/scripts/runs/no-traces \
    --out dev/scripts/runs/<fingerprints> \
    --configs normal_D5,corr_D5,halfnormal_D2,rosenbrock_D2,banana_D2,banana_D6,banana_D10,cigar_D4,lumpy_D4,lumpy_D10,student_D4,logreg_D5,cigar_D8,student_D8,multisensory_s1_D6,cigar_D15_exhaust,rosenbrock_D2_noise1_production,rosenbrock_D2_noise3_production,logreg_D5_noise3_production,student_D8_noise3_production,lumpy_D10_noise3_production,timing_D5_noise2.2_production,multisensory_s1_D6_noise1.3_production,multisensory_s2_D6_noise1.3_production
python -u dev/scripts/seeded_gate_runs.py run --out dev/scripts/runs/<gate_runs>
```

The `--baseline` names a directory that does not exist, so that each run is
judged against the population's accuracy envelopes alone. The replay exits 1
when a run lies outside its envelope, as some correct runs do, so the
fingerprints are judged as a set, as the reference's were: no more of them may
lie outside than the population's own rate of runs outside theirs makes
plausible, a binomial tail of at least 0.01 (the promotion record's
`validation.json` gives that rate). That machine's `dev/scripts/runs/LOCAL.md`
then lists them, and its replays pass the fingerprints' directory as
`--baseline`.

## Code and reproducibility

The runs use single-threaded BLAS, `vectorized_target=False`,
`performance_calibration="off"` and `tol_elcbo_boost=0.1`. Reproducing a stored
trajectory requires its recorded code, seed, options, dependency versions and
numerical platform. These are part of the reference; matching the seed alone is
insufficient.

Exact environment information, file hashes and validation reports are in the
[promotion record](promotion_20261007/README.md). The legacy
`regenerate_baseline.sh` uses a different seed allocation and masks comparison
failures; use the recorded procedure when reproducing this snapshot.
