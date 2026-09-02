# Benchmark target suite, profile campaign, golden-trace baseline

Created: 2026-09-02 21:20. Status: APPROVED 22:03, IN PROGRESS (tracker at
the end of this file).
Budget: about 10 hours of wall clock from 21:07, i.e. done by ~07:00 on
2026-09-03, everything on the laptop (22 logical CPUs: 6 P-cores, 8 E-cores,
2 LP-E; ~6.5 GB free RAM), no PRs. Roadmap: `plans/modernization-roadmap.md`
pickup point 1 and the harness half of point 2 (fixture generator and the
remaining finite-difference checks stay open); rationale in
`dev/2026-09-02-modernization-discussion.md` §10 ("Benchmark target suite",
"Stage 0 — golden-trace harness"). This file is the plan now and the worklog
afterwards: measurements and deviations are appended in place. Reviewed once
before approval (read-only doublecheck, two Opus agents: facts vs code, and
mathematics + feasibility); every finding is folded in below.

## Summary

Build one shared module of benchmark targets under `dev/scripts/`, make the
existing profiler use it, profile the harder targets the devlog asked for and
apply its decision rule to fix the Stage 2 priority order. Then build the
golden-trace harness on the same targets and record a first reference
population (10 seeds per configuration tonight, resumable to 20 and 50) of
the current NumPy implementation, with a statistical `compare` command that
later stages run against it.

## Scope

- **In scope**: `dev/scripts/benchmark_targets.py` (targets, truth,
  suites, `--check`, `--smoke`), `profile_run.py` refactor onto it, a
  sequential campaign driver with aggregation, `dev/scripts/golden_trace.py`
  with `run` / `summary` / `compare`, the profile campaign, the golden
  sweep, the Stage 2 order decision, worklog + roadmap + devlog annotations,
  `dev/README.md` and a one-line `AGENTS.md` pointer.
- **Out of scope**: any change to `pyvbmc/` package code or the notebooks
  (bugs found along the way are recorded in devlog §9, not fixed); Stage 2
  vectorization; the fixture generator and remaining finite-difference
  checks; gpyreg changes; HPC runs; the 50-seed population; the full
  D ∈ {2,…,10} sweep of devlog §10 (the suite goes to D=6 tonight).
- **No CI run**: nothing under `pyvbmc/` changes, so the roadmap's "run the
  `tests` workflow after each substantive commit" does not apply here.

## Findings the plan rests on (verified by the doublecheck unless marked)

- `dev/scripts/profile_run.py` defines `normal` and `corr` inline as an
  8-tuple `(f, x0, lb, ub, plb, pub, ln_Z, mu_bar)`, seeds with the legacy
  `np.random.seed`, has no channel for per-problem options, hardcodes
  `choices=["normal","corr"]`, and does not account for post-loop time
  (`determine_best_vp`, `final_boost` are untimed).
- `pyvbmc/testing/vbmc/test_vbmc_optimize.py` holds Rosenbrock (+N(0,3²)
  prior, D=2, plausible box ±9; notebook 1 uses ±3), the half-normal
  (bounded, probit path, box [−6, −0.05]), `cigar` (rotation literals for
  D ≤ 6 only; `cov = R.T @ diag(ell²) @ R`), an unused `noisy_cigar`, and
  the metric expressions `rmse = sqrt(mean((vp.moments() − mu_bar)²))`,
  `|elbo − lnZ|`.
- The 2018 paper (`papers/acerbi2018variational_main.md:254`) specifies
  lumpy (12 Gaussians, means U[0,1]^D, diagonal SDs in [0.2, 0.6],
  Dirichlet(1) weights), Student-t (likelihood with diagonal covariance, ν
  equally spaced in [2.5, 2 + D/2]) and cigar (one axis 100× longer, random
  rotation), **each multiplied by a broad normal prior with SD 3–4× the
  target SD**. The prior matters for Student-t: a bare t has
  logarithmically decaying log-density, which the GP's hard-wired
  negative-quadratic mean cannot follow. No real-data likelihood exists
  anywhere in the repo.
- **Notebook 1's `lml_true = −2.272` is wrong**: two independent
  quadratures give −2.2598 for Rosenbrock + N(0, 3²) at D=2 (the x2 integral
  is analytic, leaving a 1-D integral in x1). Finding for §9 and a one-line
  notebook fix later; `--check` reports the correct value.
- Noisy path: `specify_target_noise=True` makes the target return
  `(y, sd)` (finite scalar, `sd > 0`), switches the default acquisition to
  VIQR, turns on per-sample GP+VP updates (during warmup **and whenever
  `r_index > 3`**, i.e. most of a hard run), and multiplies `max_fun_evals`
  and `tol_stable_count` by 1.5, **but only for keys the caller did not set**
  (`options.py:79`).
- Budget exhaustion: `tol_stable_count = 10**6` also freezes the K-growth
  window (`variational_optimization.py:49-54`) and the ELCBO-improvement
  fit; `tol_improvement = −inf` also feeds the pruning threshold and the
  warmup-end check. **`tol_stable_excpt_frac = −10**6` is used in exactly one
  place** (`vbmc.py:1860`, the stability vote) and makes the required stable
  count unreachable with no other effect; `stable` then never becomes True,
  so `determine_best_vp` takes its no-stable branch. The Ns = 0 regime starts
  at `N ≥ stable_gp_sampling = 200 + 10D`; `normal` at D=5 (budget 350,
  `fun_eval_start = 10`, 69 iterations) spends ~20 iterations there; the D=4
  hard targets (budget 300, threshold 240) will also enter it if they ride
  out their budget.
- **`max_iter` cannot stop a run before `min_iter = D` iterations or
  `min_fun_evals = 5D` evaluations** (`vbmc.py:1888-1892`): smoke runs set
  both to 0.
- `iteration_history` records `elbo, elbo_sd, sKL, sKL_true, r_index,
  stable, warmup, Ns_gp, func_count, n_eff, pruned, gp_hyp_full, timer,
  logging_action, vp, gp, function_logger, optim_state, random_state` per
  iteration; `K` is `vp.K`, `N` is `optim_state["N"]`; `sKL_true` is `None`
  unless truth options are set. Timer keys vary per iteration.
  `results["elbo"]` is post-`final_boost`.
- **Do not pass `true_mean`/`true_cov` options.** The guard at
  `vbmc.py:1274-1278` evaluates the raw value's truthiness (arrays raise),
  and when it runs it draws 10⁶ samples from the run's own `vp.rng` every
  iteration inside the `finalize` timer. The harness computes gsKL itself,
  once, at the end, and can compute per-iteration sKL-to-truth post hoc from
  the stored VP parameters and transformer state.
- `VariationalPosterior.rng` has a setter; `__deepcopy__` shares the
  generator. `get_parameters()` **mutates** the VP and omits blocks whose
  `optimize_*` flag is off: record `vp.w, vp.mu, vp.sigma, vp.lambd`.
- Transform for unbounded coordinates: `u = (x − mu)/delta`, then, once
  rotoscale warping has fired (`warp_rotoscaling = True` by default; after
  warmup once `K ≥ 5`, `r_index < 3`, `D > 1`), `u = (u @ R_mat)/scale`.
  Inverse: `x = mu + A u`, `A = diag(delta) R_mat diag(scale)` (`R_mat = I`,
  `scale = 1` when `None`). Original-space moments: `mean = pt.inverse(
  mean_u)`, `cov = A cov_u Aᵀ`, asserted against `A@mean_u + pt.inverse(0)`.
  A warp re-bases the VP's coordinates mid-run, so per-iteration VP
  parameters are only interpretable together with that iteration's
  transformer state. Bounded coordinates need Monte Carlo.
- `kl_div_mvn` returns both KL directions; it is wrapped by
  `handle_0D_1D_input`, which swallows the first argument as `self`, so call
  it positionally with `mu1` already a `(1,D)` ndarray.
- `display="off"` still emits the termination message and ELBO line as
  warnings (`vbmc.py:1622-1630`).
- `main_timer` is a process-wide singleton and gpyreg/cma still draw from
  the global legacy state: one VBMC per **process**, never threads. Worker
  reuse is safe (`VBMC.__init__` reseeds, `optimize()` reinstalls its
  snapshot). Bit-for-bit checks need the same BLAS thread setting. Pinning
  BLAS to one thread costs ~1 % here (matrices ≤ 400×400; the Cholesky is
  ~3 % of GP training); the parallel penalty is core sharing and E-cores,
  ~1.4× on average, up to 1.8× for the unlucky worker.
- `spawn` re-imports the main script in every child and there is no
  copy-on-write on Windows: each worker pays the full `import pyvbmc`
  baseline (~250 MB: matplotlib, corner, cma, imageio). `__main__` guards,
  no import-time work, lazy caches; `MPLBACKEND=Agg` and the thread
  variables set in `os.environ` before the pool is created.
- Bugs and stale text found by the explorers and reviewers, to record in
  devlog §9 (not fixed): `vbmc.py:1341` checks `stop_gp_sampling` but only
  `stop_sampling` is ever set, so `_is_gp_sampling_finished` and
  `tol_gp_var_mcmc` are dead, and that method reads undeclared history keys
  (`N`, `gp_sample_var`), so it is broken code behind a dead guard. The
  `display` and `log_file_level` comments advertise values the code does
  not handle. `FunctionLogger.finalize()` is never called by VBMC and does
  not trim `n_evals`. `results["rng_state"]` is the literal string `"rng"`.
  `search_cmaes_best` is read nowhere. The MCMC branch of
  `active_importance_sampling` is unreachable. `gaussian_process_train.py:610`
  cubic uses the closure `x` instead of `x_`. Notebook 6's heteroskedastic
  noise uses one norm for the whole batch and broadcasts to `(n,n)`.
  Notebook 1's `lml_true` is −2.2598, not −2.272. `kl_div_mvn`'s `mu1`
  escapes the decorator's 2-D promotion. `noisy_cigar` is dead test code.

## Phases and timeline

Reviewer cost model: per-evaluation cost is flat in N and ≈ 1.2 s (D=2),
1.5 s (D=4), 1.7 s (D=5), 2.2 s (D=6) on this machine, and the hard targets
ride out `max_fun_evals = 50(2+D)`; noisy runs cost ~2.7× per evaluation;
cProfile inflates 1.5–1.8×.

| Phase | Wall clock (approx.) | Machine |
|---|---|---|
| 0. Commit the three pending doc edits | 22:05 | |
| 1. Target module + truth check + smoke (~10 min) | 22:05 – 23:30 | light |
| 2a. `profile_run` refactor + campaign driver | 23:30 – 00:00 | light |
| 2b-plain. Profile campaign, plain runs (7 configs, ~85 min) | 00:00 – ~01:30 | one heavy process, machine otherwise quiet |
| 3a. Golden-trace harness code + tiny smoke; memory probe | 00:00 – 01:30 | light |
| 3b. Golden sweep, 10 seeds, 6 workers (~3 h) | ~01:30 – ~04:30 | heavy |
| 2b-cprof. cProfile runs alongside the sweep (~2 h, one process) | ~01:30 – ~03:30 | heavy (7th process) |
| 3c. Write up plain profile results; then attribution | 01:30 – 03:45 | light |
| 4. Summary, null check, worklog, roadmap, devlog, commits, push | 04:30 – 06:30 | light |
| slack | 06:30 – 07:00 | |

cProfile percentages are self-normalizing and far less load-sensitive than
wall times, so they may share the machine with the sweep; the plain runs may
not.

Cut list, in order, if behind schedule: (1) drop the cProfile run of the
noisy config, then of the exhaust config; (2) golden seeds 10 → 5 (append
later; the harness skips existing outputs); (3) cap `max_fun_evals` at 200
for the D=4 profile configs; (4) drop `banana_D6` from the golden set (add
later); (5) only then `rosenbrock`/`halfnormal`, the sole quadrature-truth
and sole bounded-domain entries.

### Phase 0 — clean tree

Commit the uncommitted edits to devlog §11, the roadmap and the Stage 1
worklog as `docs(dev): defer the dev-next PR to the end of the work`, so the
deliverable commits below start from a clean tree.

### Phase 1 — `dev/scripts/benchmark_targets.py`

**Goal**: one place that defines every target, its ground truth, its VBMC
setup, and the named suites, used by both the profiler and the harness.

**Work**:
- `Problem` dataclass: `name, D, fun, log_density_vec, x0, lb, ub, plb,
  pub, ln_Z | None, true_mean (1,D) | None, true_cov (D,D) | None,
  sampler | None, options: dict, noisy: bool, notes: str`, with
  `vbmc_args()` (positional args + `options`; **truth is never put into
  options**) and `all_unbounded`. `fun` is a thin scalar wrapper over
  `log_density_vec(X)` (rows → values), which `--check` and the harness use
  directly. `sampler(n, rng)` draws exact samples where the generative
  process is known (banana, lumpy, cigar, student, halfnormal, normal, corr).
- `Config` (what a suite lists): `name, D, noise_sd=None, options={}`,
  `label` like `banana_D4` / `banana_D2_noise1.0_mfe150` /
  `normal_D5_exhaust`.
- `make_problem(name, D, noise_sd=None, seed=None, options=None)`. Targets
  (hard bounds ±inf unless stated; `x0` at the box centre unless stated).
  Plausible-box rule for the new targets: the per-coordinate 0.5 % and
  99.5 % quantiles of 2·10⁶ exact `sampler` draws (fixed seed), rounded to
  one decimal (≈ ±2.6 SD for Gaussian marginals; follows skew, which
  `mean ± 3 SD` does not: the banana prototype showed the symmetric box
  wasting its lower half). Legacy targets keep their legacy boxes.

  | name | definition | truth | plb / pub |
  |---|---|---|---|
  | `normal` | as today: independent N, SDs 1..D | lnZ 0, mean 0, cov diag(1..D²) | ∓2D, x0 = −1 (legacy) |
  | `corr` | as today: seeded rotation, SDs linspace(0.2, 1) | lnZ 0, analytic | ∓2.5, x0 = 0 (legacy) |
  | `halfnormal` | test target: N with SDs 1..D restricted to the negative orthant; lb = −10D, ub = 0 | lnZ −D ln 2 (to 1e-23), mean −s√(2/π), cov diag(s²(1 − 2/π)) | −6 / −0.05, x0 = −1 (test) |
  | `rosenbrock` | test/notebook target + N(0, 3²) prior, D=2 only | x2 analytic, 1-D quadrature in x1 → lnZ −2.2598, mean, cov (lazy cache) | ∓3, x0 = 0 (notebook 1) |
  | `banana` | D ≥ 2: `z ~ N(0, diag σ²)`, σ1 = 2, other σ = 1; `x1 = z1`, `x2 = z2 + b(z1² − σ1²)`, b = 0.5, rest identity; log-density `log N(z(x))`, no Jacobian term. **Prototyped 22:00** (`scratchpad/proto_targets.py`): sampled cov diag(3.99, 9.00), off-diagonal 0.01; a clear banana whose curvature in unit-box coordinates matches notebook 1's Rosenbrock | lnZ 0, mean 0, cov diag(4, 9, 1, …). **Diagonal, so RMSE and gsKL cannot see the ridge; `elbo_err` is the shape-sensitive metric here** | quantile rule: x1 ±5.2, x2 [−4.1, 14.0], rest ±2.6 |
  | `cigar` | mean linspace(−0.5, 0.5, D), `ell` = 0.01 on D−1 axes and 1.0 on the last, `Q` from `qr(default_rng(20260900 + D).standard_normal((D,D)))` (orthogonal, not necessarily a rotation); **one expression** `cov = Q diag(ell²) Qᵀ` shared by the density and `true_cov` | lnZ 0, analytic | rule (per-coordinate SDs differ: 0.29/0.92/0.18/0.19 at D=4) |
  | `lumpy` | 12 components; means U[0,1]^D, per-dim SDs U[0.2, 0.6], weights Dirichlet(1), frozen `default_rng(20260900 + D)`; log-sum-exp density | lnZ 0; mean Σ wμ; cov Σ w(diag(sd²) + μμᵀ) − μ̄μ̄ᵀ | rule (≈ [−1, 2] in expectation) |
  | `student` | product of 1-D t(ν_d), `ν = linspace(2.5, 2 + D/2, D)`, unit scale, **times the paper's prior N(0, (3·sd_d)²)** with `sd_d = sqrt(ν/(ν−2))` | factorizes: lnZ, mean, cov by per-coordinate 1-D quadrature (exact to ~1e-10) | rule, from the quadrature moments |
  | `logreg` | Bayesian logistic regression, D=5 (intercept + 4 weights), `default_rng(20260905)`: **50 trials**, predictors 1–2 correlated at ρ = 0.95, predictor 4 a rare binary (6 ones) whose outcomes are all 1; generating weights (0.3, 1, −1, 0.8, 1.5); **N(0, 5²) prior**. **Prototyped 22:10** (`scratchpad/proto_targets.py`): a tight ridge (corr(w1, w2) = −0.94), a long right tail on the prior-identified w4 (skew 0.8, SD 3.0 vs Laplace 2.6), skew 0.2–0.3 elsewhere. Trial count barely matters for skew (0.25–0.43 for n = 10…100 at prior SD 2, because the symmetric prior takes over as n shrinks); the prior width is the lever (0.7–0.9 at SD 5) | Laplace + importance sampling (t(4) proposal, 2× Laplace covariance, 2·10⁶ draws in chunks; prototype: lnZ −33.341 ± 0.002, ESS 44 % of draws), constants hard-coded with the generating command, SE and ESS in the docstring | ±5 (one prior SD, paper convention) |

  The cost half of devlog §10 item 3 ("per-evaluation cost non-trivial") is
  deliberately dropped: target time is additive and reported separately as
  `target_eval_s`, so a slow target would only dilute the algorithm profile.
- `noise_sd` wraps any target: `fun` returns `(y + sd·ε, sd)`, `ε` from
  `default_rng(SeedSequence(seed).spawn(1)[0])` (not byte-identical to
  `vbmc.rng`); rejects `sd ≤ 0`; sets `options["specify_target_noise"] =
  True`. `seed=None` → fresh entropy (profiling).
- Acquisition-function overrides live only in `Problem.options` (objects,
  not JSON); the CLI `--options` JSON covers scalars.
- `SUITES`:
  - `smoke`: `normal_D2`, `normal_D5`, `corr_D5`.
  - `profile`: `banana_D4`, `cigar_D4`, `lumpy_D4`, `student_D4`,
    `logreg_D5`, `banana_D2_noise1.0`, `normal_D5_exhaust`
    (`tol_stable_excpt_frac = −10**6`).
  - `golden`: `normal_D5`, `corr_D5`, `halfnormal_D2`, `rosenbrock_D2`,
    `banana_D2`, `banana_D6`, `cigar_D4`, `lumpy_D4`, `student_D4`,
    `logreg_D5`, `banana_D2_noise1.0_mfe150` (`max_fun_evals = 150`, set
    explicitly, so the ×1.5 does not apply; the VIQR path is what Stage 2
    is most likely to disturb, and the full-budget noisy run costs 24
    min/seed). Deviation from §10 "the same suite": exhaust is profile-only
    for cost; the golden set adds the cheap smoke/bounded/notebook targets.
- CLI: `--list`. `--check [--suite]`: for every target, (a) `log_density_vec`
  against an independent reference (`scipy.stats` composed with the target's
  own transform) at 200 random points, (b) mean and covariance against
  4·10⁶ exact `sampler` draws (report z-scores), (c) lnZ numerically only
  where it is not analytic by construction: `rosenbrock` and `student`
  (1-D quadratures, tolerance 1e-8) and `logreg` (IS, prints the constants,
  SE and ESS). `--smoke [--suite]`: each config through `VBMC.optimize()`
  with `max_iter=2, min_iter=0, min_fun_evals=0, do_final_boost=False,
  display="off"`, asserting a finite elbo and, for noisy configs, the
  `(y, sd)` contract; prints per-config wall and peak RSS.

**Steps**:
1. Write the module; `--list`.
2. `--check --suite all`; paste the logreg constants, SE and ESS into the
   module.
3. `--smoke --suite all` (~8–12 min: two iterations each, D ≤ 6).
4. `pre-commit run --files dev/scripts/*.py`.

**Verification**:
- [ ] `--check`: density matches the reference to 1e-10; sampled moments
      within |z| < 4; quadrature lnZ stable to 1e-8 under a 2× finer grid;
      logreg constants stored with SE and ESS.
- [ ] `--smoke` runs every config in all three suites without error.
- [ ] Hooks clean.

### Phase 2 — profiler on the shared module, then the campaign

**Goal**: `profile_run.py` drives any suite config; a driver runs the
`profile` suite and aggregates one table; the devlog's decision rule is
applied to the result.

**Work**:
- `profile_run.py`: import `make_problem`/`SUITES`; `--problem` accepts any
  registry name, plus `--config <label>`, `--noise-sd`, `--options` JSON;
  construct with `VBMC(..., seed=args.seed)` instead of `np.random.seed`
  (a different stream from the earlier baseline: the new `normal_D5` is a
  new trajectory, comparable in distribution, not bit-for-bit); merge
  per-problem options; report `elbo_err`, `rmse`, `gskl` only when truth is
  known, using the harness's moment routine (shared helper in
  `benchmark_targets.py`); add `untimed_s = wall − Σ stage totals`
  (approximate: stages nest); add a `determine_best_vp` attribution entry
  labelled "incl. in-loop warping calls"; record `noise_sd`, the requested
  and the **effective** options, thread settings and the git SHA in `meta`.
  `normal`/`corr` keep working unchanged.
- `dev/scripts/profile_suite.py`: run each config of a suite as a
  subprocess (`sys.executable`, absolute path to `profile_run.py`, `-u`),
  `--mode plain|cprof|both`, into `runs/profile_<timestamp>/<label>_{plain,
  cprof}/`, each streaming to its own log; `--aggregate <dir>` builds
  `aggregate.md`: per config wall, stage totals and %, untimed, iterations,
  N, final K, Ns path, elbo_err, rmse, gskl, and the cProfile attribution %
  for the standard bucket list (active_sample / cma.fmin / GP.predict /
  vp.pdf / train_gp / SliceSampler / optimize_vp / _gp_log_joint /
  _eval_full_elcbo / entmc / final_boost / active_importance_sampling /
  scipy minimize / determine_best_vp).
- Launch `--mode plain` first, alone on the machine (BLAS threads at
  default, as the existing D=5/D=10 numbers); `--mode cprof` later,
  alongside the sweep. Estimated plain: banana/cigar/lumpy/student D=4
  8–9 min each, logreg 10, noisy 24, exhaust 13, total ≈ 85 min; cProfile
  ≈ 135 min.
- Decision rule (devlog §10): if banana/lumpy/noisy shift the balance to
  the variational stage at large K, `_gp_log_joint` einsum moves up; if the
  exhaust run (and any D=4 run that reaches Ns = 0) is dominated by GP
  refits, the L-BFGS-B path joins the gpyreg sampler item. Write the
  resulting order into devlog §10 (dated, in place) and the roadmap.

**Steps**:
1. Refactor `profile_run.py`; quick regression: `--problem normal --D 5
   --max-fun-evals 40 --quiet`.
2. Write `profile_suite.py`; start `--mode plain`; note the start time here.
3. While it runs: Phase 3a. When it ends: start the sweep and `--mode
   cprof`; `--aggregate` as results land; apply the decision rule on the
   plain numbers first, confirm with the attribution.

**Verification**:
- [ ] `summary.json` of each run has `untimed_s`, `gskl`, git SHA,
      effective options.
- [ ] `aggregate.md` covers every config in the `profile` suite (or the
      cut list says which were dropped and why).
- [ ] Exhaust run reached `Ns_gp = 0` and ended on "maximum number of
      function evaluations"; noisy run used VIQR and entered
      `active_importance_sampling` (attribution > 0).

### Phase 3 — `dev/scripts/golden_trace.py` and the first population

**Goal**: a resumable, parallel sweep that stores one trace per (config,
seed); a summary; a statistical comparison usable as a gate by later stages.

**Work**:
- `run --suite golden --seeds 0-9 --workers 6 --out runs/golden/<label>`:
  `ProcessPoolExecutor` with the `spawn` context; the parent sets
  `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` and
  `MPLBACKEND=Agg` before creating the pool; tasks submitted
  **longest-first** (noisy, `banana_D6`, `logreg`, then the rest) to
  shorten the tail; one VBMC per task; skips (config, seed) whose `.npz`
  exists; a failing run writes `<tag>.error.txt` with the traceback and the
  sweep continues; one flushed progress line per finished task with wall
  and peak RSS.
- Per task: `make_problem(..., seed=seed)`, `VBMC(*args, options={...,
  "display": "off", "plot": False, "print_iteration_header": False},
  seed=seed)`, `optimize()`, extract, write `<label>_seed<seed>.npz` and a
  JSON sidecar, drop the instance.
- Trace content. Per iteration (float/int vectors, `None → nan`): `iter,
  elbo, elbo_sd, sKL, r_index, stable, warmup, K, Ns_gp, func_count, N,
  n_eff, pruned, warped` (from `logging_action`), one vector per
  stage-timer key over the **union** of keys, zero-filled; the transformer
  state `pt_mu, pt_delta (n_iter × D), pt_scale (n_iter × D), pt_R (n_iter
  × D × D)` read from `hist["vp"][i].parameter_transformer`; ragged blocks
  concatenated with index vectors: `gp_hyp` (Σ Ns_i × hyp_N) +
  `gp_hyp_iter`; `vp_w, vp_mu, vp_sigma` over Σ K_i + `vp_iter`, `vp_lambd`
  (n_iter × D), read from the attributes. Final: `elbo, elbo_sd`
  (post-boost), `best_iter, iterations, func_count, final_K, success_flag,
  message, wall_s, target_eval_s, post_mean (1,D), post_cov (D,D),
  elbo_err, rmse, gskl, X_orig, y_orig` (live rows), the final VP's `w, mu,
  sigma, lambd`. Sidecar JSON: config, seed, requested and effective
  options, git SHA and dirty flag, package versions, platform, thread
  settings, peak RSS, timestamps.
- Posterior moments (shared helper): analytic through `A = diag(delta)
  R_mat diag(scale)` when `all_unbounded`, with the assert above; otherwise
  `vp.moments(N=2·10⁵, orig_flag=True, cov_flag=True)` on a deep copy with
  `vp.rng = default_rng(2026)` (SE on the mean ≈ 0.2 % of an SD; the 10⁶
  version was the largest memory transient). Smoke test asserts the two
  routes agree on an unbounded target. `gskl = 0.5 Σ kl_div_mvn(m1, c1,
  m2, c2)` (positional, `m1` a `(1,D)` array) when `true_cov` is known.
- `summary <dir>`: per config, n runs, n failed, median and IQR of
  `elbo_err, rmse, gskl, func_count, iterations, final_K, wall_s`, fraction
  warped, fraction meeting the paper's "usable" criterion (`|Δelbo| < 1` and
  `gskl < 1`). Markdown output.
- `compare <ref_dir> <new_dir>`: per config and metric (`elbo_err`,
  `rmse`, `gskl` where available), two-sample KS statistic, raw p-value and
  median shift; median `func_count` ratio. Family-wise decision: Holm
  adjustment over all (config, metric) tests at α = 0.05, plus a flag when
  the func_count ratio leaves ±5 %; nonzero exit if anything is flagged.
  Configs present in only one population are listed, not compared; timer
  keys missing on one side are treated as zero. `compare --split <dir>`
  compares even against odd seeds of one population as a null check
  (expected: nothing flagged after Holm; a raw p < 0.01 in ~1 of 33 tests is
  normal). With 5 seeds per side the null check has little power and is
  reported as informational.
- Memory: expected 450–600 MB per worker peak (≈250 MB import baseline,
  ≈130 MB of retained GP factors on a 400-evaluation run, the MC
  transient). Probe before the sweep: `banana_D6` seed 0 alone with
  `--workers 1`, record peak RSS, set workers to `min(6, floor((free_GB −
  1.5)/peak_GB))`.
- Sweep cost (reviewer's solo estimates × 1.4 load factor, minutes per
  seed): `normal_D5` 2, `corr_D5` 2.5, `halfnormal_D2` 2, `rosenbrock_D2`
  4, `banana_D2` 4, `banana_D6` 15, `cigar_D4` 7.5, `lumpy_D4` 7.5,
  `student_D4` 7.5, `logreg_D5` 10, noisy (150 evals) 12; ≈ 74 solo → ≈ 104
  loaded worker-minutes per seed; 10 seeds ≈ 1040 / 6 ≈ 2.9 h plus tail.
  20 seeds would be ≈ 5.8 h: not tonight.

**Steps**:
1. Write the harness; smoke it with `--suite smoke --seeds 0-1 --workers 2
   --options '{"max_iter": 3, "min_iter": 0, "min_fun_evals": 0,
   "do_final_boost": false}'` while the plain campaign runs (single-thread
   workers, a few minutes, noted in the write-up).
2. `summary` and `compare --split` on the smoke output; rerun `run` on the
   same directory (must do nothing); run one config twice into two
   directories and diff `elbo` and `X_orig`; run a config with a NaN target
   to exercise the error path.
3. Memory probe (above). Start the golden sweep as soon as the plain
   campaign has finished; note the start time here. Start `--mode cprof`
   alongside.
4. When the sweep finishes: `summary`, `compare --split`; record both.

**Verification**:
- [ ] Rerunning `run` on a finished directory does nothing.
- [ ] A failing config produces an `.error.txt` and does not stop the sweep.
- [ ] Same (config, seed) twice gives identical `elbo` and evaluated
      points under the same thread setting.
- [ ] Analytic and MC moment routes agree on an unbounded target; the
      affine assert holds on a warped run.
- [ ] `compare --split` on the golden population flags nothing after Holm
      (or the write-up explains the exception).
- [ ] `summary` shows every golden config with ≥ 9 successful seeds (or
      the cut list applies).

### Phase 4 — records

**Goal**: everything a reader needs is in the repo's conventional places.

**Work**:
- This file: append §Results (campaign table, stage balance per config,
  decision-rule outcome, golden summary, null check, run times, failures,
  memory probe, deviations from plan) and §Follow-ups (seeds 10 → 20 → 50,
  higher D and the exhaust config in the golden set, HPC option, notebook 1
  `lml_true` fix, remaining Stage 0 items).
- `dev/2026-09-02-modernization-discussion.md`: extend the "Measured
  2026-09-02" paragraph in §2 with one paragraph on the hard targets;
  revise the Stage 2 priority order in §10 in place (dated); add the §9
  items listed above.
- `dev/plans/modernization-roadmap.md`: tick the benchmark suite ("at
  D ≤ 6") and the golden-trace harness ("baseline at 10 seeds, expand");
  update the Stage 2 order; new pickup point (Stage 2 item 1 in the
  measured order, seeds to 20/50, remaining Stage 0 items).
- `dev/README.md`: Scripts entries for `benchmark_targets.py`,
  `profile_suite.py`, `golden_trace.py`; this file in the "Plans, worklogs
  and task files" list; rewrite the stale sentence "results that matter get
  summarized in a dated devlog entry" to point at the `plans/` worklogs.
- `AGENTS.md`: one sentence pointing at `dev/scripts/` for the benchmark
  suite, profiler and golden-trace harness.
- Commits on `dev-next`, conventional style, one per deliverable: (1)
  targets module + `profile_run` refactor, (2) campaign driver, (3) golden
  harness, (4) docs/plan/roadmap/devlog. Pre-commit hooks run on each.
  Push `dev-next` at the end of the session.

**Verification**:
- [ ] `git status` clean; `git log` shows Phase 0 plus the four commits.
- [ ] Roadmap pickup point names the next concrete task.

## Decisions

- **Banana is a volume-preserving transform of a Gaussian, Rosenbrock kept
  only at D=2 with quadrature truth** — analytic lnZ, mean and covariance at
  every D; curvature comparable to notebook 1's Rosenbrock. Its covariance
  is diagonal, so `elbo_err` carries the shape information there. Rejected:
  Rosenbrock-only (no truth above D=2).
- **Student-t gets the paper's broad normal prior** — a bare t's log-tails
  decay logarithmically and fight the GP's negative-quadratic mean; the
  paper's version has Gaussian log-joint tails. Truth stays exact by
  per-coordinate quadrature. Rejected: bare t (fails for reasons unrelated
  to what the benchmark measures).
- **Cigar rotation from a seeded QR at every D, one covariance expression
  shared by density and truth** — one code path, reaches D=8 and 10, no
  convention mismatch. Rejected: the test file's literals (D ≤ 6 only).
- **Plausible boxes from the 0.5 %–99.5 % quantiles of exact samples for
  the new targets; legacy boxes kept for `normal`, `corr`, `halfnormal`,
  `rosenbrock` (notebook 1's ±3)** — one rule that follows skew,
  self-consistent at every seed and D; the legacy targets stay comparable
  with tests and notebooks. Rejected: `mean ± 3 SD` (the banana prototype
  showed it wasting half the x2 box).
- **Real-likelihood entry is a small, correlated, partly prior-identified
  logistic regression on fixed synthetic data (50 trials, prior SD 5),
  truth by Laplace + importance sampling stored as constants; box ±1 prior
  SD** — closed form, cheap, a −0.94 ridge and a skew-0.8 tail (prototyped;
  the prior width, not the trial count, sets the skew); 400 trials with a
  SD-2 prior would be a near-Gaussian blob of SD 0.13 in a ±24-SD box. The "non-trivial
  per-evaluation cost" half of §10 item 3 is dropped on purpose (target
  time is additive and measured separately). Rejected: rodent 2AFC (bounds
  not recoverable), g-and-k (needs a simulator), notebook models (all
  Rosenbrock).
- **Noise is a generic wrapper (`noise_sd`) on any target; the golden
  noisy config is capped at 150 evaluations** — one mechanism, seedable;
  the full-budget noisy run is 24 min/seed and the least comparable across
  versions. Rejected: notebook 6's heteroskedastic form (shape bug), the
  full-budget noisy run in the golden set.
- **Exhaust run via `tol_stable_excpt_frac = −10**6`** — the only stability
  knob used in exactly one place. Rejected: `tol_stable_count = 10**6`
  (freezes K growth and the ELCBO fit), `tol_improvement = −inf` (pruning
  threshold, warmup-end check).
- **No `true_mean`/`true_cov` options; gsKL computed once by the harness;
  per-iteration transformer state stored** — the VBMC path crashes on
  arrays and, when it runs, consumes the run's generator and the `finalize`
  timer every iteration; per-iteration sKL-to-truth can be recomputed post
  hoc from the stored VP parameters and transformer. Rejected: VBMC's
  `sKL_true`.
- **Original-space moments computed exactly through the full affine map
  (delta, R_mat, scale) when all coordinates are unbounded, otherwise MC
  with 2·10⁵ draws on a copy with a dedicated generator** — deterministic
  where possible, cheap where not, never touching the run stream.
  Rejected: `vp.moments()` on the returned VP; the `mu/delta`-only map
  (wrong after any rotoscale warp); 10⁶ draws (100–150 MB transient).
- **`--check` verifies implementations against references and sampled
  moments, and integrates numerically only where lnZ is not analytic by
  construction** — grid integration of normalized densities only measures
  truncation (a ±15 box gives −3.7e-3 on the banana), and importance
  sampling with a light proposal against t(2.5) has infinite variance.
  Rejected: "verify lnZ = 0 by quadrature/IS for every target".
- **One VBMC per spawned process, BLAS threads pinned to 1, 6 workers set
  by a memory probe, longest tasks first** — the timer singleton and the
  gpyreg/cma global-state seam forbid threads; 7 workers leaves too little
  headroom at 450–600 MB each; longest-first shortens the E-core tail.
  Rejected: threads, sequential, 7 workers, seed-major order, HPC (user
  chose the laptop).
- **Trace format is flat `.npz` plus a JSON sidecar; ragged arrays are
  concatenated with index vectors; VP parameters read from attributes** —
  small, version-independent, loadable without pyvbmc, no VP mutation.
  Rejected: pickling `VBMC` objects, object arrays with `allow_pickle`,
  `get_parameters()`.
- **Gate = two-sample KS on final metrics plus median `func_count` ratio,
  Holm-adjusted over the family at α = 0.05, no per-iteration comparison** —
  distribution-free, and the family adjustment keeps the null check honest
  across 11 configs × 3 metrics. Rejected: t-tests, unadjusted per-config
  α = 0.01, per-iteration gating.
- **10 seeds tonight, resumable to 20 and 50** — 10 seeds is ≈ 2.9 h on 6
  workers; 20 is ≈ 5.8 h and does not fit with the campaign. Rejected: 20
  or 50 tonight.
- **Plain profile runs alone on a quiet machine first; cProfile runs share
  the machine with the sweep** — wall times are the stage-balance numbers;
  cProfile percentages are self-normalizing. Rejected: all profiling before
  the sweep (serializes ~3.6 h of profiling ahead of a 3 h sweep).
- **D=4 for the main profile set, D=2 noisy, D=5 logreg and exhaust;
  golden set to D=6** — fits the budget. Rejected: the paper's full D sweep.
- **Plan lives at `dev/plans/benchmark-suite-and-golden-traces.md` and
  becomes the worklog** — repo convention for agent plans.

## Open questions (defaults in bold; the plan proceeds on them)

1. Noise SD for the noisy configuration: **1.0** (the test uses 0.5, the
   2020 paper's noisy problems sit around 1–3).
2. The golden noisy config capped at 150 evaluations (instead of the
   default 300): **yes**; the full-budget noisy run stays in the profile
   suite.
3. Push `dev-next` to origin at the end of the session: **yes** (no PR).
4. If the sweep runs long: **cut seeds 10 → 5 and append later** rather
   than dropping configurations.

## Risks

- Runtime overrun: the cut list above, applied in order; each phase notes
  its actual start time in §Results. The schedule has ~30 min of slack.
- Memory during the sweep: 450–600 MB per worker; worker count set by the
  probe; drop workers if free memory falls under 1.5 GB.
- Hard targets may fail ("Cannot optimize variational parameters", NaN
  ELBO) on some seeds: captured per run, counted in `summary`, and a
  finding in itself.
- Single-seed profile numbers: the campaign uses seed 0 only; the golden
  sweep's wall times (single-threaded, 6-way loaded) give the spread.
- Importance-sampling truth for logreg: SE and ESS reported; a 10⁻² error
  in lnZ is irrelevant at the 0.5 test tolerance but must be stated.

---

## Execution tracker

Legend: `[ ]` not started, `[~]` in progress, `[x]` done, `[!]` blocked or
needs attention. Times are wall clock on 2026-09-02/03.

Phase 0 — clean tree
- [x] Commit the pending doc edits (PR deferral) — 22:05, `48dad4a`
- [x] Commit this plan — 22:06

Phase 1 — `dev/scripts/benchmark_targets.py`
- [ ] Module: `Problem`, `Config`, `make_problem`, `SUITES`, CLI
- [ ] `--list`
- [ ] `--check --suite all`; logreg constants pasted
- [ ] `--smoke --suite all`
- [ ] pre-commit clean; commit (1) with the `profile_run` refactor

Phase 2 — profiler and campaign
- [ ] `profile_run.py` refactor; quick regression run
- [ ] `profile_suite.py` (`--mode`, `--aggregate`)
- [ ] Plain campaign started (time: ) / finished (time: )
- [ ] cProfile campaign started (time: ) / finished (time: )
- [ ] `aggregate.md`; decision rule applied
- [ ] commit (2)

Phase 3 — golden traces
- [ ] `golden_trace.py` (`run`, `summary`, `compare`)
- [ ] Smoke: resumable, error path, bit-for-bit, moment routes agree
- [ ] Memory probe (`banana_D6`, peak RSS: ) → workers:
- [ ] Sweep started (time: ) / finished (time: )
- [ ] `summary`, `compare --split`
- [ ] commit (3)

Phase 4 — records
- [ ] §Results and §Follow-ups appended to this file
- [ ] Devlog §2, §9, §10 annotations
- [ ] Roadmap ticks, Stage 2 order, pickup point
- [ ] `dev/README.md`, `AGENTS.md`
- [ ] commit (4); push `dev-next`
- [ ] Final doublecheck
