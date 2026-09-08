# Noisy-target acquisition functions: analysis and prototypes

Status: prototypes, unit tests and same-machine experiments on the two
noisy Rosenbrock configurations complete; the combination arm and the
harder configurations running (2026-09-08). No default changed.
Branch: `claude/pyvbmc-noisy-acq-funcs-9kwdny`.

## Question

The PI asked whether VIQR, which is slow, could be replaced or joined by
acquisition functions that work with noisy targets, given that every
pointwise acquisition (`AcqFcn`, `AcqFcnLog`, `AcqFcnNoisy`,
`AcqFcnVanilla`) is unusable with noise (the 2020 paper's NPRO result).
This note records the analysis, the measurements that shaped it, what was
built, and what the experiments say.

## Where a noisy run spends its time

`dev/scripts/profile_run.py --config rosenbrock_D2_noise1 --cprofile`
(seed 0, one BLAS thread, this container; 518 s profiled, 39 iterations,
200 evaluations), share of `optimize`:

| bucket | share | calls |
|---|---|---|
| `train_gp` (`gpyreg.GP.fit`, slice sampler 315 s) | 65 % | 115 fits, 76 of them the in-loop refits |
| acquisition `__call__` | 28 % | 3352 |
| of which the VIQR core | 17 % | |
| of which `GP.predict` on the candidates | 10 % | |
| `optimize_vp` | 5 % | 116 |
| `cma.fmin` | 2.7 % | 190 |

The laptop profile of 2026-09-02 (`plans/benchmark-suite-and-golden-traces.md`,
`banana_D2_noise1`) had the acquisition at 50 % and the fits at 25 %; the
machines differ in the relative cost of a Cholesky and of elementwise
work, but the two items are the same. So "VIQR is slow" is two things:

1. **Frequent retrain.** `Options.update_defaults` turns on
   `active_sample_gp_update` and `active_sample_vp_update` for noisy runs
   (the 2020 paper's "frequent retrain", Section B.1): a full hyperparameter
   fit with slice sampling and a VP optimization after every evaluation,
   during warmup and whenever the reliability index exceeds 3.
2. **The sieve.** One VIQR call per new point on `ns_search = 8192`
   candidates: an (8192 × N) by (N × 100) product per hyperparameter sample
   plus a sqrt, exp, log1p, exp, log sweep over 8192 × 100 values per
   sample. The CMA-ES refinement is nearly free (about 26 acquisition calls
   per point on this path).

Noisy runs also get 1.5× the budget and stability count, so N is larger.

## The family view, and why pointwise acquisitions fail

Every noise-capable acquisition here has three parts: a weight saying
where the posterior mass is (exp of the GP mean through importance sampling
for IMIQR, the VP density for VIQR); a look-ahead of the GP variance after a
hypothetical observation at the candidate (Eq. 5 of the 2020 paper), which
is exact, closed form and independent of the value observed there; and a
loss transform of that look-ahead integrated over the weight (sinh for the
interquantile range). The pointwise acquisitions use the candidate's own
variance reduction and its own exp(f̄)·q weight. With noise the variance at
an observed point never collapses, so the local reduction is tiny and says
nothing about the neighbourhood the observation informs, and the exp(f̄)
weight follows every bump the noise leaves in the GP mean. Tweaks to the
weight do not fix the missing global term; the work below changes the loss,
the criterion, or what one acquisition step is.

## Measurements on the noisy oracle snapshot

`rosenbrock_D2_noise1_viqr` (N = 25, Ns = 8, K = 2), 8192 VP draws as
candidates, one BLAS thread (`scratch: check_viqr_identity.py`,
`check_eig.py`, `cmaes_diag.py`, `mc_ranking.py`):

| acquisition | ms per 8192 candidates | range over the candidates |
|---|---|---|
| `GP.predict` alone | 69–74 | |
| `AcqFcnVIQR()` (loss `"iqr"`) | 521–649 | 0.057 nats |
| `AcqFcnVIQR(loss="iqr_reduction")` | 785 | 4.2 |
| `AcqFcnVIQR(loss="var_reduction")` | 209 | 4.0 |
| `AcqFcnVIQR(loss="sd_reduction")` | 602 | 4.4 |
| `AcqFcnEIG()` | 104 | 0.11 |
| `AcqFcnEIG(components=True)` | 107 | 0.20 |

- **The VIQR surface is flat.** The log post-observation interquantile
  range spans 0.057 nats across all 8192 candidates and about 1e-4 among
  the best ones, against the search optimizer's `tol_fun = 1e-2` for
  log-valued acquisitions. CMA-ES on VIQR stops on `tolfun` after 17–183
  evaluations (three seeds) and in two of three seeds finds nothing better
  than the sieve's point; on the reduction losses it runs 180–250
  evaluations and still improves the VIQR value of the chosen point by
  under 1e-4. The sieve is the search on this path; the local refinement
  is a no-op either way, which is why `ns_search` matters and CMA-ES does
  not.
- **Monte Carlo jitter of the importance set.** Across eight draws of the
  100 VP samples, the argmin over a fixed candidate set moves by a median
  0.13 length scales (max 0.4) for every loss. Second order; the
  deterministic quadrature idea (sigma points or scrambled Sobol through
  the mixture) was not pursued.
- **EIG chooses where VIQR does**, 0.04 length scales apart on this
  snapshot, at a fifth of the cost; the per-component variant chooses 0.3
  length scales away.

## Implemented

- `AcqFcnVIQR(quantile, loss)` (`acquisition_functions/acq_fcn_viqr.py`):
  the default `loss="iqr"` is bit-identical to the previous code (checked
  on the noisy oracle state against `HEAD`'s module). `"iqr_reduction"`
  (same minimizer over any candidate set, log of the reduction instead of
  the log of the residual range), `"var_reduction"` (integrated variance
  reduction: one matrix-vector product per sample, no transcendental
  sweep) and `"sd_reduction"`. The reductions use the normalized
  importance weights and average over hyperparameter samples. Tests in
  `testing/acquisition_functions/test_acq_fcn_viqr_losses.py` check them
  against `gp.predict_full` and against the algebraic identity with VIQR.
- `AcqFcnEIG(components)` (`acq_fcn_eig.py`): the port of
  `acqeig_vbmc.m`. `integrated_posterior_covariance` is the closed form of
  Eq. S18 (checked against grid quadrature in D = 1). `components=True` is
  new: the mutual information between the observation and the vector of
  per-component expected log joints, `c^T J^{-1} c / v`, with `J` the
  `J_sjk` of `_gp_log_joint`. The `compute_var_log_joint` hook in
  `active_sample.py` now stores that covariance too (the hook was dead for
  built-ins before; `_gp_log_joint` is unchanged).
- **Repeated observations** (`active_sample.py`, behind
  `max_repeated_observations > 0` on noisy runs): the training inputs join
  the sieve, a chosen repeat skips the local optimizer so that it stays an
  exact repeat (the logger pools it by precision, no new GP row), and a
  streak cap resets. The MATLAB block that did this after the search with
  a discount stays commented out. Tests in `test_vbmc_active_sample.py`.
- **`ns_gp_max_active`** (`advanced_vbmc_options.ini`, default `np.inf`):
  a cap on the hyperparameter samples of the GP refits inside active
  sampling; `0` makes them MAP fits (warm-started L-BFGS-B, no slice
  sampling) while the main-loop fit keeps sampling. Applied through the
  `options_update` copy in `active_sample.py`; saved runs without the key
  behave as before.

## End-to-end arms

`golden_trace.py run --only rosenbrock_D2_noise1,rosenbrock_D2_noise3
--seeds 0-9` on this container (four cores, one BLAS thread per worker,
two to four workers at a time, so wall times are indicative), every arm
against a same-machine baseline with the defaults. The reference
population (`dev/golden/baseline`, 50 and 30 seeds) gives the accuracy
context. Paired columns compare the same seeds.

**rosenbrock_D2_noise1** (seeds 0–9; paired columns against the defaults on the same seeds)

| arm | n | gsKL | MMTV | ELBO err | usable | evals | wall min | gsKL ratio | wins | wall ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| defaults (VIQR, both refits) | 10 | 0.023 | 0.038 | 0.087 | 0.90 | 138 | 3.4 | | | |
| both refits off | 10 | 0.021 | 0.042 | 0.075 | 1.00 | 138 | 1.6 | 1.16 | 4/10 | 0.47 |
| GP refit off, VP refit on | 10 | 0.019 | 0.030 | 0.048 | 1.00 | 135 | 1.8 | 0.26 | 6/10 | 0.56 |
| in-loop GP refit MAP-only (`ns_gp_max_active = 0`) | 10 | 0.031 | 0.039 | 0.071 | 1.00 | 125 | 1.3 | 1.04 | 5/10 | 0.37 |
| `AcqFcnVIQR(loss="var_reduction")` | 10 | 0.017 | 0.028 | 0.061 | 1.00 | 140 | 1.8 | 0.44 | 6/10 | 0.55 |
| `ns_search = 2048` | 10 | 0.019 | 0.038 | 0.088 | 1.00 | 135 | 1.0 | 0.71 | 5/10 | 0.30 |
| `AcqFcnEIG()` | 10 | 0.710 | 0.168 | 0.205 | 0.60 | 128 | 1.7 | 23.74 | 0/10 | 0.51 |
| `AcqFcnEIG(components=True)` | 10 | 0.014 | 0.033 | 0.086 | 1.00 | 148 | 2.4 | 0.54 | 7/10 | 0.86 |
| `max_repeated_observations = 3` | 10 | 0.021 | 0.038 | 0.134 | 1.00 | 145 | 2.1 | 1.04 | 4/10 | 0.57 |

**rosenbrock_D2_noise3** (seeds 0–9; paired columns against the defaults on the same seeds)

| arm | n | gsKL | MMTV | ELBO err | usable | evals | wall min | gsKL ratio | wins | wall ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| defaults (VIQR, both refits) | 10 | 0.076 | 0.066 | 0.196 | 0.90 | 198 | 3.2 | | | |
| both refits off | 10 | 0.231 | 0.130 | 0.131 | 0.60 | 162 | 2.1 | 5.14 | 2/10 | 0.67 |
| GP refit off, VP refit on | 10 | 0.064 | 0.079 | 0.147 | 0.70 | 182 | 3.3 | 1.00 | 5/10 | 1.02 |
| in-loop GP refit MAP-only (`ns_gp_max_active = 0`) | 10 | 0.262 | 0.106 | 0.152 | 0.80 | 178 | 1.9 | 1.84 | 4/10 | 0.64 |
| `AcqFcnVIQR(loss="var_reduction")` | 10 | 0.035 | 0.073 | 0.150 | 0.90 | 190 | 2.8 | 0.93 | 5/10 | 0.88 |
| `ns_search = 2048` | 10 | 0.326 | 0.119 | 0.146 | 0.80 | 190 | 1.5 | 2.10 | 2/10 | 0.47 |
| `AcqFcnEIG()` | 10 | 1.682 | 0.272 | 0.369 | 0.00 | 168 | 3.3 | 19.74 | 1/10 | 1.06 |
| `AcqFcnEIG(components=True)` | 10 | 0.126 | 0.081 | 0.204 | 0.80 | 178 | 4.5 | 2.58 | 3/10 | 1.35 |
| `max_repeated_observations = 3` | 10 | 0.031 | 0.039 | 0.253 | 0.90 | 200 | 2.9 | 0.48 | 7/10 | 0.93 |

Repeated evaluations in the `max_repeated_observations = 3` arm: a median
of 3 (0–5) of 145 evaluations at σ = 1 and 7 (3–11) of 200 at σ = 3, so the
GP had 3 and 10 fewer rows than evaluations; every other arm made none.


## Conclusions

1. **The frequent retrain earns its cost at σ = 3, not at σ = 1.** Without
   the per-evaluation refits the σ = 3 runs are 5× worse in gsKL and the
   usable fraction falls from 0.90 to 0.60, while at σ = 1 nothing is
   lost and the run takes half the time. Refitting only the VP keeps the
   σ = 3 median but not the usable fraction (0.70), and a MAP-only in-loop
   GP refit loses (1.84×). The batch-selection-with-fantasies idea of the
   proposal is off the table: a fantasized update is strictly less
   informed than the real posterior update the refits-off arm already had.
2. **The sieve is the search.** Cutting `ns_search` to 2048 costs nothing at
   σ = 1 (wall 0.30×) and 2.1× in gsKL at σ = 3. CMA-ES never improves on
   the sieve's point on this path (flat log surface, `tolfun` at 1e-2), so
   the reduction losses' better-conditioned surface changes nothing there.
3. **`loss="var_reduction"` ties VIQR at σ = 3 (0.93, 5/10) and beats it at
   σ = 1 (0.44, 6/10), at 0.88× and 0.55× the wall time.** The cheapest
   member of the family is at least as good as the interquantile range on
   these targets. Candidate noisy default, pending the harder
   configurations.
4. **Repeated observations halve gsKL at σ = 3 (0.48, 7/10; MMTV 0.039
   against 0.066) and are neutral at σ = 1**, from a handful of repeats
   per run. Candidate noisy default (`max_repeated_observations = 3`),
   pending the harder configurations; the ELBO error moved the other way
   (0.253 against 0.196), to be watched.
5. **The scalar EIG is unusable here** (0/10 usable at σ = 3, 24× the gsKL
   at σ = 1), worse than the 2020 paper's "reasonable" verdict. The
   per-component variant is competitive at σ = 1 (0.54, 7/10) and 2.6×
   worse at σ = 3, and slower per evaluation because its non-log values
   give CMA-ES a relative tolerance that keeps it running. Not a candidate
   now; the σ = 1 result says the criterion is not wrong, only weaker than
   the look-ahead losses under noise.

The combination `var_reduction` + repeats on the Rosenbrock configs and
the four arms (defaults, `var_reduction`, repeats, both) on
`logreg_D5_noise3` and `student_D8_noise3` (seeds 0–5) are running; their
results are appended below when done.


## Not done

- Deterministic quadrature for the VP integral (second order, see above).
- Oracle entries for the new acquisitions (`--add-oracle`), API docs
  beyond the automodule page, and the harder configurations
  (`logreg_D5_noise3`, `student_D8_noise3`), pending a decision on which
  of these to keep.
