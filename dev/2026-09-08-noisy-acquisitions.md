# Noisy-target acquisition functions: analysis and prototypes

Status: prototypes, unit tests and same-machine experiments complete
(2026-09-08/09): nine arms on the two noisy Rosenbrock configurations, then
the two candidates and their combination on `logreg_D5_noise3` and
`student_D8_noise3`. No default changed; recommendations at the end.
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

### Operation-level timers on four targets (2026-09-09)

The cProfile above covers one D = 2 target and carries profiler overhead
on the elementwise work. `scratch: profile_ops.py` wraps the default run
with `perf_counter` timers instead (GP fits by call site, the slice
sampler and the space-filling design inside them, both VP optimizations,
the sieve call against the CMA-ES calls, `GP.predict` inside each, and
the VIQR core split into its operations; the instrumented core performs
the original operations in the original order, so the trajectory is the
default one). Seed 0 of five configurations; four processes shared the
container's four cores, the 10-D run then ran alone.

**Run summary** (seed 0, defaults, one BLAS thread, four processes sharing four cores)

| config | wall min | evals | final N | iterations | gsKL | GP fits main / in-loop | sieve calls |
|---|---|---|---|---|---|---|---|
| rosenbrock_D2_noise1 | 4.1 | 200 | 191 | 40 | 0.257 | 43 / 72 | 190 |
| rosenbrock_D2_noise3 | 1.7 | 200 | 192 | 39 | 0.136 | 41 / 44 | 190 |
| logreg_D5_noise3 | 2.8 | 230 | 222 | 47 | 0.524 | 50 / 64 | 220 |
| student_D8_noise3 | 5.2 | 350 | 350 | 69 | 0.447 | 72 / 104 | 340 |
| lumpy_D10_noise3 | 8.9 | 450 | 448 | 89 | 0.392 | 92 / 120 | 440 |

**Share of wall time by top-level bucket** (nesting removed: the GP fits include their slice sampler, the sieve and CMA-ES buckets include their GP predictions and the VIQR core)

| bucket | rosenbrock_D2_noise1 | rosenbrock_D2_noise3 | logreg_D5_noise3 | student_D8_noise3 | lumpy_D10_noise3 |
|---|---|---|---|---|---|
| GP fit, main loop | 19.7 % | 8.8 % | 14.4 % | 17.4 % | 23.2 % |
| GP fit, in-loop refits | 40.3 % | 8.8 % | 14.0 % | 20.6 % | 17.3 % |
| VP optimization, main loop | 3.1 % | 5.7 % | 9.6 % | 7.1 % | 4.7 % |
| VP optimization, in-loop refits | 1.7 % | 1.1 % | 1.1 % | 1.7 % | 1.0 % |
| acquisition: sieve call | 31.8 % | 70.2 % | 53.6 % | 46.7 % | 49.2 % |
| acquisition: CMA-ES (calls + overhead) | 2.3 % | 2.8 % | 3.8 % | 3.4 % | 1.3 % |
| importance-sample set-up | 0.3 % | 0.8 % | 0.6 % | 0.7 % | 0.8 % |
| GP posterior re-update between points | 0.1 % | 0.2 % | 0.2 % | 0.2 % | 0.3 % |
| final boost | 0.6 % | 1.3 % | 1.1 % | 0.5 % | 0.4 % |
| everything else (warping, bookkeeping, target) | 0.1 % | 0.3 % | 1.5 % | 1.9 % | 1.7 % |

**Inside the GP fits** (share of the fit time)

| part | rosenbrock_D2_noise1 | rosenbrock_D2_noise3 | logreg_D5_noise3 | student_D8_noise3 | lumpy_D10_noise3 |
|---|---|---|---|---|---|
| slice sampler | 94 % | 71 % | 75 % | 73 % | 50 % |
| space-filling initial design | 4 % | 26 % | 23 % | 23 % | 26 % |
| optimizer, posterior, rest | 1 % | 3 % | 2 % | 4 % | 23 % |

**Inside one sieve call** (share of the sieve time; ms per call in the last column group)

| operation | rosenbrock_D2_noise1 | rosenbrock_D2_noise3 | logreg_D5_noise3 | student_D8_noise3 | lumpy_D10_noise3 |
|---|---|---|---|---|---|
| GP prediction at the candidates | 31 % | 33 % | 36 % | 41 % | 47 % |
| nearest-neighbour noise | 1 % | 2 % | 2 % | 3 % | 3 % |
| kernel matrices (cdist, exp) | 21 % | 21 % | 22 % | 23 % | 25 % |
| matrix product (Nx x N)(N x Na) | 11 % | 12 % | 11 % | 10 % | 12 % |
| tau2 and sqrt | 10 % | 11 % | 9 % | 7 % | 4 % |
| sinh in log form (exp, log1p) | 15 % | 13 % | 12 % | 10 % | 6 % |
| log-sum-exp (max, exp, sum, log) | 9 % | 8 % | 7 % | 5 % | 3 % |
| wrapper rest (bounds, transform, mask) | 1 % | 1 % | 1 % | 1 % | 1 % |
| **ms per sieve call** | **407** | **387** | **415** | **430** | **595** |

**Inside the CMA-ES stage** (share of the CMA-ES time)

| part | rosenbrock_D2_noise1 | rosenbrock_D2_noise3 | logreg_D5_noise3 | student_D8_noise3 | lumpy_D10_noise3 |
|---|---|---|---|---|---|
| acquisition calls | 60 % | 55 % | 59 % | 61 % | 53 % |
| of which GP prediction | 30 % | 35 % | 30 % | 33 % | 53 % |
| of which VIQR core | 31 % | 28 % | 30 % | 33 % | 26 % |
| acquisition calls per new point | 17 | 9 | 18 | 16 | 9 |

**Scaling with the training-set size** (sieve call and GP fit, by quartile of N within each run)

| config | N range | ms per sieve call (prediction / VIQR core) | s per GP fit (in-loop) | s per GP fit (main) |
|---|---|---|---|---|
| rosenbrock_D2_noise1 | 10–48 | 275 (44 / 229) | 0.2 | 0.2 |
| rosenbrock_D2_noise1 | 48–95 | 375 (95 / 277) | 0.2 | 0.2 |
| rosenbrock_D2_noise1 | 95–142 | 450 (163 / 285) | - | 0.6 |
| rosenbrock_D2_noise1 | 142–190 | 528 (210 / 316) | 2.8 | 2.5 |
| rosenbrock_D2_noise3 | 10–49 | 279 (45 / 232) | 0.2 | 0.2 |
| rosenbrock_D2_noise3 | 49–96 | 403 (104 / 296) | 0.3 | 0.2 |
| rosenbrock_D2_noise3 | 96–143 | 418 (160 / 255) | 0.2 | 0.2 |
| rosenbrock_D2_noise3 | 143–191 | 450 (205 / 242) | - | 0.2 |
| logreg_D5_noise3 | 10–58 | 301 (52 / 246) | 0.3 | 0.3 |
| logreg_D5_noise3 | 58–111 | 408 (119 / 284) | 0.4 | 0.4 |
| logreg_D5_noise3 | 111–166 | 425 (178 / 243) | - | 0.5 |
| logreg_D5_noise3 | 166–221 | 528 (252 / 272) | - | 0.7 |
| student_D8_noise3 | 10–94 | 404 (86 / 315) | 0.5 | 0.5 |
| student_D8_noise3 | 94–179 | 465 (189 / 272) | 0.8 | 0.9 |
| student_D8_noise3 | 179–264 | 586 (292 / 290) | - | 1.2 |
| student_D8_noise3 | 264–349 | 267 (143 / 121) | - | 0.5 |
| lumpy_D10_noise3 | 10–117 | 487 (138 / 346) | 0.6 | 0.6 |
| lumpy_D10_noise3 | 117–227 | 884 (408 / 472) | 1.1 | 1.4 |
| lumpy_D10_noise3 | 227–337 | 750 (418 / 329) | 1.0 | 1.9 |
| lumpy_D10_noise3 | 337–447 | 259 (157 / 100) | 1.1 | 1.4 |

Reading:

- **The sieve call is the largest item on four of the five targets**
  (47–70 % of wall, 49 % at D = 10) and second on `rosenbrock_D2_noise1` (32 %), where
  the run never stabilizes and the in-loop GP refits run at N up to 190
  (40 %). A sieve call costs about 0.4 s and is nearly flat in N: the
  VIQR core is 26–54 ms per hyperparameter sample per call, dominated by
  elementwise work on (8192 × 100) arrays (the log-form sinh, the
  log-sum-exp, the square roots: about a third of the call) and the kernel
  matrices (a fifth; `K(Xs, X)` is recomputed here although `predict`
  just formed it), with the matrix product itself only a tenth. The GP
  prediction at the candidates is the other third and is the part that
  grows with N (45 ms at N < 50, 210–250 ms at N > 150). The call halves
  once hyperparameter sampling stops (`student_D8_noise3` at N ≥ 280):
  the cost is proportional to Ns (the same halving at N ≥ 300 on
  `lumpy_D10_noise3`).
- **The GP fits are 17–60 %**, the slice sampler 71–94 % of them and the
  space-filling initial design (`f_min_fill`, 1024 random hyperparameter
  vectors before every fit) 23–26 % on four targets: an avoidable
  in-loop cost, since the refits could start from the previous samples.
  Per fit the cost spans 0.2–5 s depending on Ns and N.
- **The in-loop VP optimizations are 1–2 %**: the "frequent retrain" is
  expensive only through the GP hyperparameter refit.
- CMA-ES is 2–4 % (9–18 batched acquisition calls per new point), the
  importance-sample set-up under 1 %, the final boost about 1 %.

What this says about speeding VIQR up without touching the algorithm:
skip the log-form sinh and the log-sum-exp in favour of a direct `sinh`,
sum and one `log` (two transcendental sweeps instead of four, about 15 %
of the sieve), reuse the cross-kernel from the prediction (about 20 %),
and start the in-loop refits from the previous hyperparameters instead
of the space-filling design (about a quarter of the fit time). Together
that is roughly a third of the wall time of these runs.

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


## Acquisition optimization: the sieve and CMA-ES (2026-09-09)

The PI recalled that CMA-ES did poorly on the acquisition and that a larger
sieve kept helping, which is why `ns_search` is 8192. Measured on saved
mid-run states (seed 0, budget-limited, no final boost) at D = 2 (N = 86),
5 (N = 139), 8 (N = 200) and 10 (N = 246), one fixed VIQR surface per state
(same GP, VP and importance set), `scratch: acq_opt_bench.py`. The score is
the integrated-IQR *reduction* at the returned point as a fraction of the
best any method found (a 65536-point sieve, then L-BFGS-B from eight
mutually distant tops and from every strategy's result); the cost is
acquisition evaluations (points) and batched calls.

| strategy | D = 2 | D = 5 | D = 8 | points D = 2 / 5 / 8 |
|---|---|---|---|---|
| sieve of 8192, no refinement (production without CMA-ES) | 100 % | 77 % | 60 % | 8192 |
| CMA-ES as in production (log post-IQR, `tolfun` 1e-2, noise handler, sigma0 = VP sd) | 100 % | 77 % | 60 % | 10 / 12 / 15 |
| CMA-ES on the log-reduction, `tolfun` 1e-3, no noise handler, sigma0 = VP sd | 100 % | 100 % | 100 % | 158 / 442 / 1012 |
| same, sigma0 = 0.3 length scales | 100 % | 77 % | 73 % | 236 / 626 / 1172 |
| L-BFGS-B from the sieve best, batched finite-difference gradient | 100 % | 100 % | 100 % | 18 / 72 / 279 |
| L-BFGS-B from 8 mutually distant sieve tops | 100 % | 100 % | 100 % | 270 / 786 / 2142 |
| coarse-to-fine resampling around the top 32, 4 rounds of 1024 | 100 % | 99 % | 97 % | 4096 |

The sieve alone, median best reduction over five draws as a fraction of
the reference:

| `ns_search` | D = 2 | D = 5 | D = 8 |
|---|---|---|---|
| 512 | 99.9 % | 60 % | 59 % |
| 2048 | 100 % | 61 % | 60 % |
| 8192 | 100 % | 72 % | 63 % |
| 32768 | 100 % | 76 % | 68 % |

Reading:

- **At D = 2 the sieve is the optimizer and it is enough**: 512 points
  already sit at the optimum, so nothing downstream can matter, which is
  why the CMA-ES diagnostics on the D = 2 snapshot showed no gain.
- **At D = 5 and 8 the sieve leaves 25–40 % of the achievable reduction
  on the table and grows only logarithmically with its size** (D = 8:
  59 % at 512, 68 % at 32768). That is the "bigger sieve keeps helping"
  memory: the sieve has been doing the optimizer's job.
- **Production CMA-ES never improves on the sieve point**: it stops on
  `tolfun` after two generations (10–15 evaluations) on every state. Two
  compounding causes. The objective is the log of the *residual*
  interquantile range, which varies by 5e-3 across the whole candidate set
  (the reduction is 1e-3 of the range) against an absolute `tolfun` of
  1e-2; and `sigma0` is the VP's largest standard deviation, so the first
  generations sample the whole posterior width around the sieve point and
  find nothing better. (The noise handler is pointless on a deterministic
  objective but was not isolated as a cause.)
- **A local gradient step from the sieve point recovers the whole gap at
  1–3 % of a sieve's cost**: L-BFGS-B with a forward-difference gradient
  from one batched call of D + 1 rows reaches the reference in 72 (D = 5)
  and 279 (D = 8) evaluations, 12 and 31 calls, 0.02 and 0.07 s. Eight
  restarts found nothing better on these states. CMA-ES also gets there
  once it optimizes the log-reduction without the noise handler, at 3–4×
  the evaluations, and only with the wide `sigma0` (the length scales here
  are far larger than the posterior width, so 0.3 ℓ is not a local step).
- **`lumpy_D10_noise3` at N = 246 is a different failure and is excluded**:
  the GP's length scales collapse in two dimensions (per-sample minima of
  1e-4 and 0 against a posterior width of order 1), the surface is flat to
  8e-13 across the sieve and the best reduction is 1.5e-13 of the range,
  ten orders of magnitude below D = 5 and 8. No optimizer can act on that;
  the GP hyperparameter fit under heavy noise is the problem there.

What to do, in order:

1. **Optimize the log-reduction, not the log residual.** Same minimizer,
   O(1) dynamic range, so any tolerance means what it says. Available as
   `AcqFcnVIQR(loss="iqr_reduction")`; the search stage should use it as
   the optimizer's objective while the sieve can keep either.
2. **Replace CMA-ES by L-BFGS-B from the sieve's best point (optionally a
   few mutually distant tops), with the gradient from one batched
   acquisition call of D + 1 rows.** The batched objective already exists
   for CMA-ES. This is the standard recipe of modern BO libraries (raw
   samples, top-k restarts, L-BFGS-B). Analytic gradients (every operation
   in the VIQR core is closed form) or autodiff in the torch port would
   halve the cost and remove the finite-difference step.
3. **Then shrink the sieve.** With refinement in place, 2048 candidates
   plus L-BFGS-B should match 8192 plus nothing on these states; the
   remaining role of the sieve is coverage of separate basins, which eight
   restarts did not find here. To be confirmed end to end, since the
   `ns_search = 2048` arm lost accuracy at σ = 3 without refinement.
4. If CMA-ES stays: log-reduction objective, `tolfun` 1e-3 or relative, no
   noise handler, `sigma0` from the VP width, a minimum number of
   generations before `tolfun` can fire.
5. The coarse-to-fine batched resampling (97–99 %) is the gradient-free
   fallback where the surface is not smooth (integer variables, the
   piecewise-constant nearest-neighbour noise estimate).

### Sieve size before L-BFGS-B

Same states and scoring (`scratch: sieve_size_lbfgs.py`): a sieve of
`n` candidates, then L-BFGS-B from its best point or from 2, 4 or 8
mutually distant tops; five draws each; median [min] of the reduction as
a fraction of the reference, and the median total evaluations (sieve plus
refinement).

| state | sieve n | sieve alone | 1 start, evals | 4 starts, evals | 8 starts, evals |
|---|---|---|---|---|---|
| D = 2 | 128 | 99.6 % [99.0] | 100 % [100], 156 | 100 % [100], 258 | 100 % [100], 447 |
| D = 2 | 1024 | 100 % [99.8] | 100 % [100], 1043 | 100 % [100], 1154 | 100 % [100], 1307 |
| D = 5 | 128 | 59 % [58] | 74.5 % [70], 231 | 100 % [74.5], 681 | 100 % [100], 1209 |
| D = 5 | 256 | 60 % [59] | 74.5 % [60], 383 | 100 % [100], 701 | 100 % [100], 1295 |
| D = 5 | 512 | 60 % [59] | 74.5 % [60], 597 | 100 % [100], 1005 | 100 % [100], 1551 |
| D = 5 | 1024 | 61 % [60] | 70 % [70], 1115 | 100 % [100], 1487 | 100 % [100], 2009 |
| D = 5 | 8192 | 72 % [66] | 100 % [70], 8277 | 100 % [100], 8595 | 100 % [100], 9009 |
| D = 8 | 128 | 52 % [51] | 100 % [100], 363 | 100 % [100], 1056 | 100 % [100], 2073 |
| D = 8 | 256 | 54 % [52] | 100 % [100], 500 | 100 % [100], 1148 | 100 % [100], 2183 |
| D = 8 | 1024 | 57 % [53] | 100 % [100], 1241 | 100 % [100], 1934 | 100 % [100], 2933 |
| D = 8 | 8192 | 63 % [61] | 100 % [100], 8409 | 100 % [100], 9084 | 100 % [100], 10002 |

- The sieve size barely matters once a local optimizer follows: at D = 8
  a 128-point sieve and one start reach the optimum every time (363
  evaluations against 8192 for a sieve that reaches 63 %).
- The number of restarts matters where the surface has several basins:
  the D = 5 state has a second basin at 74.5 % that catches a single start
  in most draws whatever the sieve size (even the 8192 sieve's own best
  point sits in it in some draws); four mutually distant starts reach the
  optimum from every sieve of 256 points or more.
- A defensible default is therefore a sieve of 512–1024 candidates and
  four restarts from mutually distant tops, about 1000–2000 evaluations at
  D = 5–8, a fifth of the present sieve, reaching the optimum where the
  present search stops at 60–75 %. The distance criterion for "mutually
  distant" should be scaled by the VP width rather than the GP length
  scales, which can be far larger than the posterior.
- Caveat: one state per dimension, seed 0; the end-to-end arms decide.

### Bumpiness and Monte Carlo overfitting of the refinement

The PI's caution about generalizing from smooth synthetic states: checked
on the three states above plus three built to be harder (`scratch:
landscape.py`, `refine_na.py`): a noisy multimodal `lumpy_D4` (σ = 1), a
correlated `banana_D6` (σ = 2), and the Student target late in its run
(N = 325, one hyperparameter sample, K = 29).

**Basins**, from L-BFGS-B started at 32 VP draws on one importance set:

| state | GP length scale / VP width, min and median | distinct optima | starts within 5 % of the best |
|---|---|---|---|
| rosenbrock_D2_noise3 | 2.1, 8.3 | 2 | 59 % |
| logreg_D5_noise3 | 5.0, 6.6 | 3 | 94 % |
| student_D8_noise3, N = 200 | 4.1, 7.4 | 2 | 91 % |
| student_D8_noise3, N = 325 | 3.6, 4.0 | 5 | 3 % |
| banana_D6_noise2 | 1.8, 274 | 7 | 9 % |
| lumpy_D4_noise1 | 0.05, 0.95 | 31 | 6 % |

Once the length scales approach the posterior width, or N grows, the
surface fragments and random starts land in poor basins. The earlier
D = 5 and 8 states were the smooth end.

**Monte Carlo overfitting.** Refine on the 100-sample importance set,
judge the point on an independent 4000-sample set (median of five draws;
the refined point's reduction over the sieve point's):

| state | gain on its own 100-sample set | gain under 4000 samples | refined point worse than the sieve's |
|---|---|---|---|
| logreg_D5_noise3 | ×1.18 | ×1.06 | 2 / 5 |
| student_D8_noise3, N = 200 | ×1.36 | ×0.96 | 3 / 5 |
| student_D8_noise3, N = 325 | ×1.40 | ×0.82 | 3 / 5 |
| banana_D6_noise2 | ×1.18 | ×1.16 | 1 / 5 |
| lumpy_D4_noise1 | ×1.00 | ×1.00 | 0 / 5 |

Per draw the judged gain ranges from ×0.57 to ×1.28 on the Student state.
The "100 % of the reference" numbers of the two previous subsections were
measured on the surface the optimizer climbed, so they measured how well
L-BFGS-B fits the sampling noise of 100 importance points, not how much
better the chosen point is. The coarse-to-fine resampling overfits the
same way; the sieve is robust because it does not resolve the fine
structure.

**Refining on a larger set fixes it.** Sieve and one-start L-BFGS-B on a
set of `Na` samples, judged on an independent 6400-sample set (median
[min, max] over five draws):

| state | Na = 100 | Na = 400 | Na = 1600 |
|---|---|---|---|
| logreg_D5_noise3 | ×1.02 [0.96, 1.26], worse 2/5 | ×1.22 [1.00, 1.35], worse 1/5 | ×1.21 [0.99, 1.24], worse 1/5 |
| student_D8_noise3, N = 200 | ×0.90 [0.54, 1.27], worse 3/5 | ×1.22 [1.17, 1.38], worse 0/5 | ×1.27 [1.18, 1.31], worse 0/5 |
| banana_D6_noise2 | ×1.21 [0.96, 1.25], worse 1/5 | ×1.20 [0.97, 1.25], worse 2/5 | ×1.24 [1.05, 1.34], worse 0/5 |
| student_D8_noise3, N = 325 | ×0.87 [0.67, 1.16], worse 3/5 | ×1.07 [1.02, 1.11], worse 0/5 | ×1.08 [1.03, 1.09], worse 0/5 |

Revised recommendation, replacing items 2–3 of the list above:

1. The sieve stays the coarse selector, on the 100-sample set as now (its
   choice was the best of the compared points under the judge in most
   draws); its size can come down once a refinement follows.
2. Refinement only on an importance set of at least 400, preferably 1600
   samples (or a deterministic quadrature of that accuracy), which costs
   little because the refinement makes tens of batched calls of D + 1
   rows: re-evaluate the sieve's top few on the large set, then L-BFGS-B
   on the log-reduction from the best of them, and keep the sieve point
   if the large-set value does not improve. Judged independently this
   gains ×1.07–1.27 over the sieve point on four states with no draw
   worse at 1600 samples.
3. The gain is modest and state-dependent; whether it is worth anything
   end to end is still the open question, and the synthetic targets
   remain the caveat.

## The combination and the harder configurations

Same protocol; the harder configurations at seeds 0–5 only (4.5–12
minutes per run here). The reference population's medians for the
defaults are gsKL 0.42 and 0.55 on these two configs (50 and 30 seeds), so
the six-seed baselines here sit on the accurate side of their
distributions.
**rosenbrock_D2_noise1** (seeds 0–9; paired columns against the defaults on the same seeds)

| arm | n | gsKL | MMTV | ELBO err | usable | evals | GP rows | wall min | gsKL ratio | wins | wall ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| defaults | 10 | 0.023 | 0.038 | 0.087 | 0.90 | 138 | 138 | 3.4 | | | |
| `var_reduction` | 10 | 0.017 | 0.028 | 0.061 | 1.00 | 140 | 140 | 1.8 | 0.44 | 6/10 | 0.55 |
| repeats, cap 3 | 10 | 0.021 | 0.038 | 0.134 | 1.00 | 145 | 142 | 2.1 | 1.04 | 4/10 | 0.57 |
| `var_reduction` + repeats | 10 | 0.032 | 0.040 | 0.104 | 1.00 | 152 | 146 | 2.1 | 0.57 | 5/10 | 0.69 |

**rosenbrock_D2_noise3** (seeds 0–9; paired columns against the defaults on the same seeds)

| arm | n | gsKL | MMTV | ELBO err | usable | evals | GP rows | wall min | gsKL ratio | wins | wall ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| defaults | 10 | 0.076 | 0.066 | 0.196 | 0.90 | 198 | 198 | 3.2 | | | |
| `var_reduction` | 10 | 0.035 | 0.073 | 0.150 | 0.90 | 190 | 190 | 2.8 | 0.93 | 5/10 | 0.88 |
| repeats, cap 3 | 10 | 0.031 | 0.039 | 0.253 | 0.90 | 200 | 190 | 2.9 | 0.48 | 7/10 | 0.93 |
| `var_reduction` + repeats | 10 | 0.073 | 0.081 | 0.368 | 0.90 | 188 | 176 | 2.8 | 1.48 | 4/10 | 0.84 |

**logreg_D5_noise3** (seeds 0–5; paired columns against the defaults on the same seeds)

| arm | n | gsKL | MMTV | ELBO err | usable | evals | GP rows | wall min | gsKL ratio | wins | wall ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| defaults | 6 | 0.312 | 0.114 | 0.086 | 0.83 | 262 | 262 | 5.7 | | | |
| `var_reduction` | 6 | 0.554 | 0.184 | 0.131 | 0.67 | 220 | 220 | 5.3 | 1.88 | 0/6 | 0.99 |
| repeats, cap 3 | 6 | 0.355 | 0.129 | 0.169 | 0.83 | 270 | 268 | 6.0 | 0.94 | 3/6 | 1.03 |
| `var_reduction` + repeats | 6 | 0.486 | 0.174 | 0.142 | 0.83 | 228 | 226 | 5.2 | 1.50 | 0/6 | 1.02 |

**student_D8_noise3** (seeds 0–5; paired columns against the defaults on the same seeds)

| arm | n | gsKL | MMTV | ELBO err | usable | evals | GP rows | wall min | gsKL ratio | wins | wall ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| defaults | 6 | 0.487 | 0.134 | 0.541 | 0.67 | 342 | 342 | 8.8 | | | |
| `var_reduction` | 6 | 0.864 | 0.140 | 1.029 | 0.50 | 360 | 360 | 11.5 | 2.26 | 0/6 | 1.29 |
| repeats, cap 3 | 6 | 0.421 | 0.130 | 0.763 | 0.67 | 348 | 346 | 9.3 | 0.96 | 4/6 | 1.02 |
| `var_reduction` + repeats | 6 | 0.928 | 0.145 | 1.130 | 0.17 | 362 | 362 | 11.1 | 2.47 | 0/6 | 1.25 |

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
3. **`loss="var_reduction"` ties VIQR at D = 2 and loses at D = 5 and 8.**
   On the Rosenbrock configs it is at least as accurate (0.44 at σ = 1,
   0.93 at σ = 3) at 0.55× and 0.88× the wall time; on `logreg_D5_noise3`
   and `student_D8_noise3` it is 1.9× and 2.3× worse in gsKL with no win
   in twelve paired runs, and on the Student target it runs longer (1.29×,
   more evaluations before stability). The interquantile range's
   exponential weighting of high-variance regions is doing real work on
   the harder targets; the cheapest member of the family is not a
   replacement. Not a candidate default.
4. **Repeated observations never hurt and help once.** gsKL 0.48 (7/10) at
   σ = 3 on the Rosenbrock, 1.04 at σ = 1, 0.94 (3/6) and 0.96 (4/6) on the
   two harder configs, at 0.93–1.03× the wall time, from a handful of
   repeats per run (2–10 fewer GP rows than evaluations). The ELBO error
   moves the other way in three of four configs (0.253 against 0.196,
   0.169 against 0.086, 0.763 against 0.541 medians), which the small
   samples cannot separate from noise but is consistent in sign. Safe to
   enable on noisy runs; the evidence for a gain is confined to the
   low-dimensional high-noise case. The combination with `var_reduction`
   inherits the loss's failure.
5. **The scalar EIG is unusable here** (0/10 usable at σ = 3, 24× the gsKL
   at σ = 1), worse than the 2020 paper's "reasonable" verdict. The
   per-component variant is competitive at σ = 1 (0.54, 7/10) and 2.6×
   worse at σ = 3, and slower per evaluation because its non-log values
   give CMA-ES a relative tolerance that keeps it running. Not a candidate
   now; the σ = 1 result says the criterion is not wrong, only weaker than
   the look-ahead losses under noise.



## Not done

- Deterministic quadrature for the VP integral (second order, see above).
- Oracle entries for the new acquisitions (`--add-oracle`) and API docs
  beyond the automodule page, pending a decision on which of these to
  keep.
- The L-BFGS-B search stage (item 2 of the acquisition-optimization
  section) as a `search_optimizer` option, and the end-to-end arms with a
  smaller sieve.
- A noise-adaptive policy for the frequent retrain and the sieve size:
  at σ = 1 both can be cut for half the wall time at no cost, at σ = 3
  neither can. Keying `active_sample_gp_update`, `active_sample_vp_update`
  and `ns_search` to the estimated noise at the high-posterior-density
  region (`sn2_hpd`, already computed each iteration) is the one speed
  lever these experiments leave open that does not trade accuracy; the
  threshold needs a sweep over noise levels.
- `sd_reduction` and `iqr_reduction` end to end (dropped as redundant
  once CMA-ES proved a no-op on this path; `sd_reduction` is the
  untested middle ground between the variance and the interquantile
  range).
