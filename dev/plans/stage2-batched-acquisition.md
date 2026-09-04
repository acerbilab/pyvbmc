# Stage 2 item 3: batched acquisition evaluation

Created: 2026-09-04 14:50. Status: **IN PROGRESS**. Roadmap pickup point 2
(`plans/modernization-roadmap.md`); rationale in
`dev/2026-09-02-modernization-discussion.md` §10 (Stage 2, item 3, first in
the measured order) and the profile in
`plans/benchmark-suite-and-golden-traces.md` §Results (regenerated): active
sampling is 54–69 % of wall on every noiseless target, and the single-point
acquisition calls made by CMA-ES (`GP.predict` 40–50 % of profiled time,
41k–249k calls per run; `vp.pdf` 4–6 %, 15 % at D = 15) are most of it.
This file is the plan now and the worklog afterwards. Reviewed before
execution by two read-only Opus agents (facts vs the `cma` source and the
PyVBMC call sites; design and gates); findings folded in below.

## Summary

Same algorithm, fewer Python round trips. (1) A small replay script turns
the stored golden traces into a per-change trajectory check that runs in
minutes. (2) `active_sample` hands CMA-ES the whole population per
generation through `cma.fmin`'s `parallel_objective`, and the noise
handler's re-evaluations are batched the same way, so the acquisition is
evaluated twice per generation instead of `popsize + 2` times. (3)
`vp.pdf` broadcasts over the mixture components instead of looping. Gates:
the oracles on every commit, the replay per step, the profile suite once
for the speedup. The gpyreg-side half of item 3 (`predict`'s loop over the
hyperparameter samples and the `sW` tiling) moves to item 8.

## Scope

- **In**: `dev/scripts/golden_replay.py` (new); `pyvbmc/vbmc/active_sample.py`
  (the CMA-ES call and its objective); `pyvbmc/variational_posterior/
  variational_posterior.py` (`pdf`, Gaussian branch);
  `dev/scripts/make_oracle_fixtures.py` (a targeted re-baseline of one
  oracle from the stored state, if the step oracle flips); records
  (`dev/README.md`, roadmap, this file, devlog §9 if anything new turns up).
- **Out**: gpyreg changes (item 8); `_gp_log_joint` (item 1);
  `_eval_full_elcbo` (item 2); the Nelder–Mead path (D = 1 only);
  removing or changing the noise handler (algorithmic); the heavy-tailed
  branches of `vp.pdf` (not on any hot path, no oracle); the 50-seed
  population; the end-of-stage 20-seed population run.

## Findings the plan rests on

Verified against the installed `cma` 4.4.4 and the code on 2026-09-04.

- `active_sample.py:357` wraps the acquisition as `acq_fun = lambda X:
  acq_eval(X, gp, vp, function_logger, optim_state).item()` and calls
  `cma.fmin(acq_fun, x0, max(insigma), options=cma_options,
  noise_handler=cma.NoiseHandler(D))` with `maxfevals = 500 (D + 2)`,
  `tolfun` 1e-2 (log acquisitions) and bounds `(lb_search, ub_search)`.
  `AbstractAcqFcn.__call__` already accepts an `(n, D)` batch (it is the
  sieve path with `Nc = 2^13`, and the oracle path with `Nc = 512`),
  promotes a 1-D input to one row, and returns `(n,)`.
- `cma.fmin` takes `parallel_objective` as a keyword argument (its 11th
  positional; `evolution_strategy.py:4900`). With `objective_function=None`
  it defines a scalar wrapper `objective_function(x) =
  parallel_objective([x])[0]` (`:4978-4981`) and calls
  `es.ask_and_eval(parallel_objective, ..., parallel_mode=True)` (`:5077`).
  The batched objective always receives a **list** of 1-D arrays and must
  return a **list** (`ask_and_eval` pops from it, `:2222`). `eval_initial_x`
  defaults to `False`; `eval_final_mean` defaults to `True`, so the
  strategy's final mean is evaluated once more after the loop through the
  scalar path (`:5131-5137`): `2 n_generations + 1` acquisition calls per
  search, against `(popsize + 2) n_generations + 1` today.
- `CMAEvolutionStrategy.ask_and_eval` draws the population with **one**
  `self.ask(popsize)` call in both modes, then either `func(X_first)` (a
  list of 1-D arrays, parallel) or `func(x)` per solution (serial). Bound
  repair happens inside `ask` (the phenotype map), so both modes see the
  same repaired points. Rejection resampling (`is_feasible`, default
  "not None and not NaN"; `inf` is feasible) calls `func(x)` on a single
  array even in parallel mode, so the batched objective must distinguish
  a list of arrays from one array. The random stream is therefore
  identical in both modes as long as the fitness values are.
- The noise handler (`NoiseHandler(D)`: `maxevals=[1, 1, 1]`, so
  `evaluations` is always 1; `reevals=None`, so `2 + popsize/20`
  solutions per generation, the fractional part decided by
  `np.random.rand()` from the **legacy global state**, the seam already
  recorded in `AGENTS.md`) re-evaluates each chosen solution at
  `ask(1, X_i, epsilon=1e-7)` through the *scalar* `objective_function`
  (fmin l. 356). Its `reeval` loops over the chosen indices calling `ask`
  then `func` for each; batching it means calling `ask` for every index
  first (same calls, same order, same draws) and `func` once on the list.
  `evaluations_just_done` and `es.countevals` are unaffected. `parallel=True`
  on the stock handler does not batch across solutions (it batches the
  `evaluations` repeats, which are 1 here), so a subclass is needed.
- Batched versus single-point acquisition values are not bit-identical,
  by a few ulp (measured by the fact-check): in `gp.predict` the
  triangular solve (TRSM vs TRSV, 6e-16 absolute), `Ks.T @ alpha` (GEMM vs
  GEMV, 2e-15 relative) and NumPy's own reduction order in `np.sum(V*V,
  0)` (pairwise on `(N, 1)` vs sequential into `(N, k)`, 8e-16, BLAS
  independent); the kernel matrix itself (`cdist`) is row-independent and
  bit-identical. A second, structural mechanism on the noisy acquisitions:
  `AbstractAcqFcn._sq_dist` (`abstract_acq_fcn.py:214-216`) centres both
  point sets on a mean that depends on the batch size and the batch mean
  (2e-15 relative), and `_estimate_observation_noise` takes an `argmin`
  over it, so a near-tie in the nearest training point can flip and change
  `sn2` by a finite amount (`AcqFcnNoisy`, VIQR, IMIQR). Either flips a
  CMA-ES ranking and changes the trajectory. The oracle plan foresaw this
  for the `active_sample_step` oracle: the `acq_*` oracles, which evaluate
  the batched path on 512 candidates, are the arbiter.
- **`_real2int` snaps its input in place** (`abstract_acq_fcn.py:186-192`)
  and the pointwise call reached CMA-ES's own solution arrays through the
  view `Xs[None, :]`, so with `integer_vars` set the told solutions were
  snapped to the integer grid. A batched call copies the rows, so it has
  to write the snapped rows back to keep that behaviour (no benchmark
  config, oracle or test sets `integer_vars`; found by the fact-check).
- Rejection resampling (NaN fitness) in parallel mode warns through
  `warnings.warn` at `global_verbosity`, which the strategy's `verbose:
  -9` does set (`utils.py:357-373`); the acquisition never returns NaN
  today (`np.maximum(acq, -realmax)` would propagate one), so the path is
  unreachable in practice.
- Dormant batch-only shape bug (dead code): the variance-regularization
  block `acq[mask] += …` (`abstract_acq_fcn.py:121-127`) would fail on a
  batch when `acq` is still 2-D (VIQR/IMIQR with `Ns = 1`); the block is
  never reached because the option key is misspelt (devlog §9).
- **Re-baselining the step oracle must not rerun the source run.** The
  generator's `--only <recipe>` reruns the recipe's VBMC run, whose
  trajectory changes with the batched search, so every reference of that
  snapshot would move and the fixture would stop pinning the pre-change
  numerics. A targeted mode is needed: load the stored state, recompute one
  named oracle, rewrite only its `ref/<oracle>/*` entries, note it in
  `meta`.
- `vp.pdf` (`variational_posterior.py:491`, `:525`, `:553`) loops over `k`
  accumulating `y += nn_k`; the acquisition functions call it with
  `orig_flag=False` (plain and `log_flag=True`). In a broadcast form the
  per-component terms are bit-identical (the same scalar products; the
  sum over `D` stays on the contiguous axis), and only the sum over `K`
  changes order (NumPy pairwise summation for `K > 8`): about 1e-16
  relative, against the `vp_pdf` oracle's 1e-10.
- The golden harness's `run_task(label, seed, extra_options, out_dir)`
  (`golden_trace.py:103`) builds the run exactly as the baseline did and
  writes the trace; it does not create the directory, swallows every
  exception into `<tag>.error.txt`, and expects `benchmark_targets` and
  `profile_run` importable by module name (run scripts from `dev/scripts`
  as scripts). `cmd_run` sets the three BLAS thread variables *and*
  `MPLBACKEND=Agg` before spawning; in-process they must be set before
  NumPy is imported. The baseline traces are on this machine under
  `dev/scripts/runs/golden/baseline_20260903/` (gitignored) and the
  sidecars in git under `dev/golden/baseline/` (20 seeds × 14 configs).
  `X_orig`/`y_orig` in the trace are the **live** rows (`X_flag`): warm-up
  trimming removes low-`y` rows anywhere, including in the initial
  design, so a positional comparison of live rows is only a lower bound on
  the evaluation horizon; the per-iteration `elbo` path is the clean
  measure (iteration 0 identical certifies the initial design). Timers,
  wall, `target_eval_s`, `peak_rss_mb` (cumulative and process-wide
  in-process) and the `meta` block are not comparable.
- The baseline's "determinism" had been shown only on smoke runs in
  spawned workers (2026-09-02); the first full-length in-process replay
  (Step 1) is what establishes it for the golden configs. Supporting
  fact: `git diff 5020879..HEAD -- pyvbmc/ ':!pyvbmc/testing'` is empty
  and `golden_trace.py` changed only in `EST_MINUTES` since the baseline.
- Popsize is `int(4 + 3 ln D)`: 6 at D = 2, 8 at D = 4 and 5, 10 at
  D = 10, 12 at D = 15 and 20. Per generation the acquisition is called
  `popsize + 2` (or 3) times today and will be called twice.
- The one unit test on the CMA-ES path, `test_active_uncertainty_sampling`
  (`testing/vbmc/test_vbmc_active_sample.py:32-92`), mocks
  `AbstractAcqFcn.__call__` with a Rosenbrock on `np.atleast_2d(x)` and
  asserts the chosen points to `atol = 1e-3`: it must keep passing with a
  batch (it does, see the tracker).
- `profile_suite.py`: `--aggregate <dir>` is a path argument and a normal
  run aggregates at the end anyway; `run_one` inherits the environment
  (pin the BLAS threads by hand for comparability with 2026-09-03) and
  skips configs whose `summary.json` exists (use a fresh stamp); the full
  `profile` suite plain is ≈ 59 min because of `cigar_D15_exhaust`.
- `test_vbmc_resume_optimization` asserts `elbo_1 == elbo_1`
  (`test_vbmc_optimize.py:630`), a self-comparison: it pins nothing
  (devlog §9 candidate, not fixed here).

## Design

### Step 1: `dev/scripts/golden_replay.py`

`python dev/scripts/golden_replay.py [--configs a,b,c] [--seeds 0]
[--baseline dev/scripts/runs/golden/baseline_20260903] [--sidecars
dev/golden/baseline] [--out <dir>]`. For each (config, seed): set the
BLAS thread variables to 1 (before importing NumPy, as the baseline was
run), call `golden_trace.run_task` into a scratch directory, load both
traces and report:

- **agreement horizon**: the first iteration at which the `elbo` path
  differs (exactly; also beyond 1e-6), and the number of leading live
  points identical (a lower bound, see Findings); iteration 0 identical
  is the initial-design check;
- **final metrics** side by side (ΔLML, gsKL, MMTV, evaluations,
  iterations, wall) and whether ΔLML, gsKL and MMTV lie inside the
  baseline population's envelope for that config, the Tukey far-out fence
  `Q3 + 3 IQR` over the seeds' sidecars (the plain maximum is vacuous
  where a seed is a known failure: `student_D4` seed 19, gsKL 54);
- verdict per config: `identical` / `parted at iteration i` / flags for a
  failed run, a different iteration 0, or a final outside the envelope.
  Nonzero exit if anything is flagged. `--report-only --out <dir>`
  re-renders a finished run.

Without the `.npz` traces (fresh checkout) it degrades to the final-metric
comparison against the sidecars. Default set, seed 0, about 7 minutes:
`normal_D5`, `banana_D2`, `halfnormal_D2` (probit), `cigar_D4` (warp,
large K), `rosenbrock_D2_noise1` (VIQR path). Verification of the script
itself: with the unchanged code every default config must replay
`identical`; this run is also what establishes the baseline's
reproducibility at full length in-process.

### Step 2: batched CMA-ES objective

In `active_sample.py`, replace the scalar lambda by a batched objective
that accepts either a list of 1-D arrays (one generation) or one 1-D array
(the rejection path, the final-mean evaluation and the Nelder–Mead path)
and returns a list of floats or one float accordingly, writing the rows
back into the caller's arrays when `integer_vars` is set (the in-place
snapping of the pointwise call); call `cma.fmin(acq_fun, x0, sigma0,
options=cma_options, parallel_objective=acq_fun, noise_handler=...)` and
keep everything else (`x0`, `insigma`, options, the `res[:2]` unpacking,
the `f_val_optim < f_val_old` acceptance) unchanged. Add a
`_BatchedNoiseHandler(cma.NoiseHandler)` that receives the batched
objective (fmin hands `reeval` only the scalar wrapper) and whose `reeval`
performs the stock method's `ask` calls in the stock order and then one
batched evaluation, wrapping each value as `f_aggregate([v])` as the stock
code does, falling back to the parent when `evaluations != 1` (never the
case here). Nelder–Mead keeps the scalar form. `res[:2]` is the best-ever
told solution (or the final mean), never a re-evaluation, in both modes.

Gates: `pytest pyvbmc/testing/oracles` (the `acq_*` oracles are the
arbiter); the active-sampling unit tests; the replay; the full suite once
before the commit. If `active_sample_step` fails: confirm every `acq_*`
oracle passed on that snapshot, then re-baseline that oracle alone with
the targeted generator mode (Step 2b) and record the distance between old
and new chosen points here.

### Step 2b: targeted re-baseline in the generator

`make_oracle_fixtures.py --rebaseline <oracle> --reason "..." [--only
<snapshots>]`: read the raw `.npz` arrays and the JSON tree, rebuild the
state *with the target* (the step oracle's `applies` needs it), recompute
the named oracle under the stored oracle seed, replace its `ref/<oracle>/*`
arrays (same key set, asserted), append a `meta["rebaselined"]` entry
(oracle, date, git SHA, reason, per-output max change) and rewrite both
files (`np.savez_compressed` has no partial update: git sees the whole
binary change); then reload and assert every other array bit-identical and
`check_one(path, fun, exact=True)` clean. Refuses the step oracle off the
generating platform (the JSON `@@npz:` markers are unchanged since the
keys are). Note `test_fixture_complete` cannot police the step oracle (it
builds the state without a target), which is why the key-set assertion
lives in this mode.

### Step 3: `vp.pdf` over K

Gaussian branch only (`df` infinite or 0). With `sig = sigma.reshape(1,
K, 1)`, `lam = lambd.reshape(1, 1, D)`, `diff = x[:, None, :] −
mu.T[None]` (`(N, K, D)`): `z = diff / (sig · lam)`; `d2 = (z²).sum(2)`
(`(N, K)`); `nn = nf · w / sigma^D · exp(−d2/2)` (`(N, K)`); `y = nn.sum(1,
keepdims=True)`; `dy = −(nn[:, :, None] · diff / (lam² · sig²)).sum(1)`.
Per-component terms stay bit-identical to the loop's (the same scalar
products, the `D`-sum on the contiguous axis); only the `K`-sum order
changes: measured (tracker), `y` is bit-identical for K < 8 and within
1e-15 relative from K = 8 on (NumPy's 8-way unrolled pairwise reduction
over the last axis), and `dy` is bit-identical at every K because the
reduction over the middle axis of `(n, K, D)` is sequential, the loop's
order (so the signed-cancellation concern raised in review does not
materialize). Two or three `(N, K, D)` temporaries are live at once, so the
sieve's worst case (`N = 8192`, `K = 50`, `D = 20`) is 200–260 MB, not 66;
chunk over `N` so that `n · K · D ≤ 2^22` elements (32 MB per temporary;
at the sieve this means two chunks at `K = 50, D = 20`, no chunking
below `K · D = 512`). Keep the computation on all rows and the `mask`
handling exactly where it is (`dy = dy / y` runs before masking, `mask`
is never applied to `dy`); `handle_0D_1D_input` only promotes a 1-D `x`
and ravels the return, so `(N, 1)` / `(N, D)` outputs are unaffected.
Heavy-tailed branches unchanged. Gates: `vp_pdf` oracle (1e-10), the VP
unit tests (`test_variational_posterior.py:108-235` pins `pdf` against
MATLAB arrays at `rtol 1e-12` with `K = 2`, where any summation order is
identical; `test_variational_posterior_grad_fd.py` finite differences),
the `acq_*` oracles, the replay.

### Step 4: measure

`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u
dev/scripts/profile_suite.py --suite profile --mode plain --out
runs/profile_<stamp>` alone on a quiet machine (≈ 59 min before the
change; the run aggregates at the end); compare wall and the
active-sampling share per config with `runs/profile_20260903/aggregate.md`
(the pre-change numbers, seed 0, same thread setting). Then
cProfile on the four D = 4 configs to see where the acquisition time went
(expected: the per-call `GP.predict` overhead now over ~1/4 of the calls;
the `Ns` loop and `sW` tiling inside gpyreg become the visible remainder,
which is item 8). Record here, in the roadmap and in devlog §2/§10 as a
dated addendum.

## Steps

- [x] Plan reviewed (two read-only Opus agents), findings folded in
- [x] Step 1 `golden_replay.py`; verification: unchanged code replays
      `identical` on the default set
- [x] Step 2 batched CMA-ES objective + batched noise handler; oracles,
      unit tests, replay, full suite; commit `7a07c0b`
- [x] Step 2b targeted re-baseline (the step oracle flipped on four
      snapshots; mode written and run, `50c1e50` / `7a07c0b`)
- [x] Step 3 `vp.pdf` broadcast; oracles, unit tests, replay, full suite;
      commit
- [~] Step 4 profile campaign (plain; cProfile on the D = 4 set); write-up
- [ ] Records: `dev/README.md` (replay script), roadmap ticks and pickup
      point, devlog §2/§10 addendum, this file's results; push; CI smoke
      green
- [ ] `/doublecheck`

## Verification

- [ ] Replay of the unchanged code: every default config `identical`.
- [ ] After Step 2: `pytest pyvbmc/testing/oracles` green (or the step
      oracle re-baselined with the `acq_*` oracles green and the distance
      recorded); `pytest --reruns=5 -x` green; replay: initial design
      identical on every config, finals inside the population.
- [ ] After Step 3: same gates; `vp_pdf` oracle green without re-baseline.
- [ ] Step 4: wall time per config and the active-sampling share
      recorded against the 2026-09-03 numbers; no config's ΔLML/gsKL/MMTV
      outside the population range at seed 0.
- [ ] CI smoke (Ubuntu / 3.12) green on the push.

## Decisions

- **`parallel_objective` plus a batching `NoiseHandler` subclass, not
  ask/tell.** Rewriting the loop with `es.ask()`/`es.tell()` would have to
  reproduce fmin's noise-handling block (sigma and `cmean` updates,
  `countevals`) to stay the same algorithm; `parallel_objective` is the
  supported way and the subclass only reorders `ask`/`func` calls.
- **Noise handler kept.** It changes CMA-ES's step-size adaptation on a
  deterministic acquisition (it measures the 1e-7-scale local variation
  as "noise"); removing it would be an algorithmic change, out of Stage 2.
- **The replay gate is soft by design**: ranking flips from one-ulp
  differences are expected, so it reports an agreement horizon and checks
  the finals against the population range rather than demanding identity.
  Identity is demanded only of the initial design (drawn from the
  generator before any numerics).
- **Step oracle re-baselined in place, never by rerunning the recipe.**
- **gpyreg half of item 3 deferred to item 8** so that all gpyreg changes
  land in one PR against `acerbilab/gpyreg` and one pin bump.
- **Heavy-tailed `pdf` branches untouched**: no hot path, no oracle, and
  the gradient there raises `NotImplementedError` anyway.

## Open questions (defaults in bold)

1. Should the replay's default set include `logreg_D5` (bounded, 2 min)?
   **No**: `halfnormal_D2` covers the probit path in 30 s.
2. Chunking threshold for the broadcast `pdf`: **`N · K · D > 2^24`**.

## Risks

- A `cma` upgrade could change `fmin`'s treatment of `parallel_objective`
  or `NoiseHandler.reeval`; the subclass pins itself to the stock
  method's structure and falls back to the parent when the assumptions
  fail. `cma` is unpinned in `pyproject.toml`.
- The batched path can change trajectories enough that a golden config
  ends outside the population range at seed 0 by chance; the replay then
  reports it and a second seed decides.
- Profiling must run alone (one heavy process; the 2026-09-02 crash).

## Execution tracker

Legend: `[ ]` not started, `[~]` in progress, `[x]` done, `[!]` needs
attention. Times are wall clock on 2026-09-04.

- [x] Plan written — 14:50
- [x] Two read-only Opus reviews of the plan — 14:55–15:12 (facts vs the
  `cma` source and PyVBMC call sites, 7 questions all confirmed with 7
  additional risks; design and gates, 32 findings). Folded in: the
  `_real2int` in-place snapping (write-back added), the `_sq_dist`
  batch-mean centring as a second flip mechanism, `eval_final_mean`, the
  list return, popsize 8 at D = 5, the masked `X_orig` (ELBO horizon is
  primary, iteration 0 is the initial-design check), the Tukey fence
  instead of the max, the corrected broadcast formulas and memory
  figures for Step 3, the whole-file rewrite and key-set/platform checks
  for Step 2b, the profile-suite thread pin and runtime, the one CMA-ES
  unit test and its mock, the `elbo_1 == elbo_1` test typo
- [x] Step 1 `dev/scripts/golden_replay.py` written — 15:00; replay of the
  unchanged code on the default set (one process, BLAS 1 thread, 7.0 min)
  → `runs/golden/replay_step1_unchanged/`: **all five configs
  `identical`** (same live points, same ELBO path, same finals). Two
  script fixes after the run: the report's arrows are not cp1252 (stdout
  reconfigured to UTF-8 when redirected on Windows), and the "first
  differing iteration" column reads `none` for identical runs
- [~] Step 2 code in place (batched `acq_fun`, `parallel_objective`,
  `_BatchedNoiseHandler`, integer-variable write-back) — 15:08
- [x] Oracle gate — 15:10: `pytest pyvbmc/testing/oracles` **96 passed, 4
  failed, 15 skipped**; every `acq_*` oracle green on every snapshot; the
  four failures are all `active_sample_step`, on `cigar_D4_boosted`,
  `cigar_D4_largeK`, `corr_D5_warped`, `halfnormal_D2_bounded`; the three
  `normal_D2_*` step oracles pass unchanged
- [x] **Plumbing check** — 15:12 (scratch script; the step oracle on all 7
  snapshots in three modes): the new plumbing with per-row evaluation
  inside the batch call reproduces the stored points **bit-for-bit on all
  seven** (max |ΔX| = 0), as does the old serial call; the batched
  evaluation moves the chosen points by 1.07 (`cigar_D4_boosted`: a
  different local optimum of the same search, as on Ubuntu in the oracle
  plan), 0.16 (`corr_D5_warped`), 7.5e-4 (`halfnormal_D2_bounded`),
  1.0e-4 (`cigar_D4_largeK`) and 0 on the three `normal_D2_*` snapshots.
  So `parallel_objective` + `_BatchedNoiseHandler` preserve draws and
  values exactly, and the divergence is the batched arithmetic alone
- [x] `test_vbmc_active_sample.py` — 15:14: 18 passed (the CMA-ES mock
  included)
- [x] Step 2 replay — 15:14–15:21 → `runs/golden/replay_step2_batched_cmaes/`
  (seed 0; wall in minutes, baseline → batched):

  | config | ELBO path identical through iteration | finals (ΔLML, gsKL, MMTV) baseline → batched | evals | wall |
  |---|---|---|---|---|
  | normal_D5 | all 13 (`identical`) | unchanged | 70 → 70 | 1.0 → 0.81 |
  | banana_D2 | 4 of 16 | 0.042, 0.202, 0.048 → 0.055, 0.207, 0.051 | 85 → 85 | 0.59 → 0.52 |
  | halfnormal_D2 | 7 of 13 | 0.0062, 0.00026, 0.0167 → 0.0016, **0.00075**, 0.0188 | 70 → 70 | 0.54 → 0.49 |
  | cigar_D4 | 2 of 27 | 0.00065, 0.00032, 0.0076 → 0.00002, 0.00096, 0.0071 | 135 → 125 | 2.4 → 1.4 |
  | rosenbrock_D2_noise1 | 20 of 27 | 0.059, 0.050, 0.040 → 0.059, 0.025, 0.026 | 140 → 140 | 2.0 → 2.0 |

  Iteration 0 identical everywhere (initial designs preserved); every
  final inside the population's `Q3 + 3 IQR` envelope except
  `halfnormal_D2` gsKL (0.00075 vs fence 0.00063; population median
  0.00012, max 0.00040; an absolute gsKL of 7.5e-4 is far below the
  papers' usability threshold of 1). The noisy config barely moves in
  wall time, as expected (CMA-ES is 5–8 % of that path). Follow-up:
  halfnormal on seeds 1–4 to tell chance from a shift
- [x] `halfnormal_D2` seeds 1–4 — 15:23–15:25 →
  `runs/golden/replay_step2_halfnormal_seeds/`: nothing flagged; gsKL
  baseline → batched by seed: 9.0e-5 → 7.7e-5, 4.0e-4 → 3.3e-4,
  1.4e-4 → 8.1e-5, 6.3e-5 → 6.0e-4 (seed 0: 2.6e-4 → 7.5e-4). Three
  seeds down, two up: noise, not a shift; the 20-seed population
  comparison at the end of the stage is the arbiter
- [x] **Step 2b re-baseline run** — 15:21: `--rebaseline
  active_sample_step --only cigar_D4_boosted,cigar_D4_largeK,
  corr_D5_warped,halfnormal_D2_bounded`; max |ΔX_new| 1.07, 1.0e-4,
  0.16, 7.5e-4 (as in the plumbing check); every other array asserted
  bit-identical, exact round trip clean; `pytest pyvbmc/testing/oracles`
  **100 passed, 15 skipped**; git shows the four `.json` (+15 lines of
  `meta.rebaselined` each) and the four `.npz`
- [x] `dev/README.md` (replay script, `--rebaseline`, this plan) and
  `AGENTS.md` (replay gate) — 15:25
- [x] Full suite (`pytest --reruns=5 -x`, one BLAS thread) — 15:27–15:37:
  **517 passed, 15 skipped, 0 reruns, 9:30**
- [x] Devlog §9: four findings of the reviews recorded (the `elbo_1 ==
  elbo_1` test, the batch-only shape bug in the dead regularization
  block, `_sq_dist`'s batch dependence, `_real2int`'s in-place snapping)
- [x] Commits — 15:40: `50c1e50 test(dev)` (replay gate, `--rebaseline`
  mode, records) then `7a07c0b perf(vbmc)` (the batched objective, the
  noise handler, the four re-baselined step-oracle references)
- [x] Step 3 `vp.pdf` broadcast written — 15:45 (Gaussian branch; rows
  chunked at 2^22 elements per `(n, K, D)` temporary). First run tripped
  on the MATLAB fixture VP, whose `K` is a `np.uint8` (NumPy 2 refuses
  `2**22 // K` in `uint8`): cast to `int`. Then VP, acquisition, oracle
  and active-sampling tests: **201 passed, 15 skipped**; the step oracle
  did not flip
- [x] Bit check old loop vs broadcast (scratch, random VPs): `dy`
  bit-identical at every K tested (2, 8, 9, 50; the reduction over the
  middle axis of `(n, K, D)` is sequential, the loop's order); `y`
  bit-identical at K = 2 and within 9e-16 relative from K = 8 upward
  (NumPy's 8-way unrolled pairwise reduction over the last axis), which
  is the `K`-sum reordering the plan predicted, one power of ten below
  its "~1e-16" wording's spirit but not its letter
- [!] Step 3 replay started 15:38 → `runs/golden/replay_step3_pdf_broadcast/`;
  **killed at 15:43 on the PI's request (laptop needed)** after 4 of 5
  configs, all with the same verdicts and finals as the Step 2 replay
  (`normal_D5` identical; the others parted at the same iterations).
  **Pickup point**: Step 3 code is in the working tree, uncommitted (the
  `pdf` broadcast in `variational_posterior.py`; 201 targeted tests
  green). Resume with: (1) `python -u dev/scripts/golden_replay.py --out
  dev/scripts/runs/golden/replay_step3_pdf_broadcast` (rerun in full,
  7 min), (2) the full suite `pytest --reruns=5 -x` with one BLAS thread
  (10 min), (3) commit `perf(vp): broadcast pdf over the mixture
  components`, (4) Step 4 profiling (§Design), (5) records and
  `/doublecheck`. The two Step 2 commits (`50c1e50`, `7a07c0b`) are local,
  not pushed.
- [x] Resumed 16:37 (PI); Step 3 replay rerun in full, 16:37–16:44 →
  `runs/golden/replay_step3_pdf_broadcast/`: **the same verdicts and the
  same finals as the Step 2 replay on every config** (the `K`-sum
  reordering did not flip a single ranking on these runs); the only flag
  is halfnormal's seed-0 gsKL, already shown to be chance. Wall, baseline
  → Step 2 → Step 3 (min): normal_D5 1.0 → 0.81 → 0.72; banana_D2 0.59 →
  0.52 → 0.46; halfnormal_D2 0.54 → 0.49 → 0.49; cigar_D4 2.4 → 1.4 →
  1.3; rosenbrock_D2_noise1 2.0 → 2.0 → 1.9
- [x] Full suite for Step 3 — 16:45–16:52: **517 passed, 15 skipped, 0
  reruns, 7:09** (9:30 before Step 2: the six end-to-end runs got faster)
- [x] Commit — 16:53: `perf(vp): broadcast pdf over the mixture components`
- [~] Step 4 profile campaign (plain) started 16:55 →
  `runs/profile_20260904/`
- [x] Step 2b `--rebaseline ORACLE --reason` mode written in
  `make_oracle_fixtures.py` — 15:18; to run on the four snapshots after
  the replay (one process at a time)
