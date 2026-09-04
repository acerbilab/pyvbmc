# Stage 2 item 3: batched acquisition evaluation

Created: 2026-09-04 14:50. Status: **DONE 2026-09-04 18:00** (PyVBMC
half of item 3; follow-ups at the end). Roadmap pickup point 2
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
- The strategy's final mean (`eval_final_mean`) and the noise handler's
  re-evaluations are evaluated at different batch sizes from the
  population (1 and `2 + popsize/20` against `popsize`), so `es.best` and
  the noise measure compare values of two provenances; on a near-tie this
  can differ from the pointwise run by a few ulp. Pre-existing in kind
  (`f_val_old` comes from the 8192-point sieve and is compared with the
  CMA-ES optimum).
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
- verdict per config: `identical` / `parted at iteration i` (0-based:
  iterations `0..i−1` bit-identical) / flags for a failed run, a
  different iteration 0, a non-finite final, or a parted run's final
  outside the envelope (an identical run is exempt: its own seed may be
  the population's far outlier). Exit 1 if anything is flagged or nothing
  was compared. `--report-only --out <dir>` re-renders a finished run,
  keeping its provenance.

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
objective explicitly (the stock `reeval` calls its `func` argument once
per solution whatever fmin hands it; here `acq_fun` is passed as both
`objective_function` and `parallel_objective`, so `func` happens to be the
same dual-mode callable) and whose `reeval` performs the stock method's
`ask` calls in the stock order and then one batched evaluation, wrapping
each value as `f_aggregate([v])` as the stock code does, falling back to
the parent when `evaluations != 1` (never the case here). Nelder–Mead keeps the scalar form. `res[:2]` is the best-ever
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
materialize). Three or four `(N, K, D)` temporaries are live at once, so
the sieve's worst case (`N = 8192`, `K = 50`, `D = 20`) is 200–260 MB, not
66; chunk over `N` so that `n · K · D ≤ 2^16` elements (0.5 MB per
temporary, in cache; the first commit used 2^22, which §Results shows
was memory-bound and slower than the loop on large inputs). Keep the
computation on all rows and the `mask`
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
      commit `f441172`; chunk size corrected in `e923163` (VP and oracle
      tests only: no result changes)
- [x] Step 4 profile campaign (plain; cProfile on the D = 4 set); write-up
- [x] Records: `dev/README.md` (replay script), roadmap ticks and pickup
      point, devlog §2/§10 addendum, this file's results; pushed
- [x] CI smoke of the push green (see Verification)
- [x] `/doublecheck` (three read-only Opus verifiers; findings folded in,
      `da0aa91` and the commit after it)

## Verification

- [x] Replay of the unchanged code: every default config `identical`.
- [x] After Step 2: `pytest pyvbmc/testing/oracles` green (or the step
      oracle re-baselined with the `acq_*` oracles green and the distance
      recorded); `pytest --reruns=5 -x` green; replay: initial design
      identical on every config, finals inside the population (one
      chance excursion on `halfnormal_D2` seed 0, cleared on seeds 1–4).
- [x] After Step 3: same gates; `vp_pdf` oracle green without re-baseline.
- [x] Step 4: wall time per config and the active-sampling share
      recorded against the 2026-09-03 numbers; no config's ΔLML/gsKL/MMTV
      outside the population range at seed 0 (the exhaust row flagged as
      measured on a throttling machine).
- [x] CI smoke (Ubuntu / 3.12) green on the push of `3033526` (run
      33886022057, 15:10). The doublecheck follow-up push re-runs it.

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
2. Chunking threshold for the broadcast `pdf`: **`n · K · D ≤ 2^16` per
   chunk**, measured (§Results). The first draft said 2^24, which the
   review showed could never fire at a supported dimension; the first
   commit used 2^22, which timing showed to be memory-bound.

## Risks

- A `cma` upgrade could change `fmin`'s treatment of `parallel_objective`
  or `NoiseHandler.reeval`; the subclass pins itself to the stock
  method's structure and falls back to the parent when the assumptions
  fail. `cma` is unpinned in `pyproject.toml`.
- The batched path can change trajectories enough that a golden config
  ends outside the population range at seed 0 by chance; the replay then
  reports it and a second seed decides.
- Profiling must run alone (one heavy process; the 2026-09-02 crash).

## Results (2026-09-04)

### Speedup: plain profile campaign, seed 0, one process, BLAS single-threaded

`runs/profile_20260904/` (code `f441172`, clean tree, i.e. the batched
search and the broadcast `pdf` with its first, 2^22-element chunking; the
cache-sized chunking of `e923163` came after the campaign and was not
re-profiled: it changes no result and, in a run, only the once-per-point
sieve call and the final `mode`) against `runs/profile_20260903/` (code
`5020879`, the pre-change numbers of
`plans/benchmark-suite-and-golden-traces.md` §Results (regenerated)).
Both campaigns on the same laptop, otherwise idle; started 16:51. Wall and
stage totals in seconds; the arrow reads old → new.

| config | wall (×) | active sampling (× ; % of wall) | GP train | var. fit | iters | evals | ΔLML | gsKL | MMTV |
|---|---|---|---|---|---|---|---|---|---|
| banana_D4 | 67 → 43 (1.56) | 39 → 18 (2.25; 59 → 41 %) | 11 → 10 | 8 → 8 | 17 → 17 | 95 → 95 | 0.053 → 0.041 | 0.144 → 0.118 | 0.029 → 0.021 |
| cigar_D4 | 141 → 79 (1.79) | 84 → 34 (2.44; 60 → 44 %) | 19 → 17 | 23 → 18 | 26 → 24 | 135 → 125 | 0.001 → 0.000 | 0.000 → 0.001 | 0.008 → 0.007 |
| lumpy_D4 | 61 → 41 (1.48) | 33 → 15 (2.23; 54 → 36 %) | 12 → 11 | 7 → 7 | 15 → 15 | 85 → 85 | 0.040 → 0.040 | 0.013 → 0.013 | 0.026 → 0.026 |
| student_D4 | 72 → 47 (1.52) | 45 → 21 (2.13; 63 → 45 %) | 14 → 13 | 7 → 7 | 19 → 19 | 105 → 105 | 0.019 → 0.019 | 0.008 → 0.008 | 0.039 → 0.039 |
| logreg_D5 | 110 → 81 (1.36) | 61 → 29 (2.09; 55 → 36 %) | 17 → 18 | 21 → 22 | 23 → 23 | 120 → 120 | 0.066 → 0.066 | 0.012 → 0.012 | 0.029 → 0.029 |
| rosenbrock_D2_noise1 | 126 → 116 (1.09) | 95 → 87 (1.09; 75 → 75 %) | 24 → 23 | 18 → 15 | 26 → 26 | 140 → 140 | 0.059 → 0.059 | 0.050 → 0.025 | 0.040 → 0.026 |
| logreg_D5_noise3 | 267 → 251 (1.06) | 178 → 164 (1.09; 67 → 65 %) | 73 → 72 | 44 → 44 | 44 → 44 | 220 → 220 | 0.132 → 0.132 | 0.771 → 0.771 | 0.247 → 0.247 |
| lumpy_D10 | 283 → 182 (1.55) | 170 → 72 (2.35; 60 → 40 %) | 79 → 76 | 19 → 19 | 37 → 37 | 185 → 185 | 0.784 → 0.784 | 0.808 → 0.808 | 0.125 → 0.125 |
| banana_D10 | 175 → 103 (1.70) | 121 → 48 (2.52; 69 → 47 %) | 38 → 39 | 7 → 7 | 25 → 25 | 135 → 135 | 0.127 → 0.127 | 0.493 → 0.493 | 0.024 → 0.024 |
| cigar_D15_exhaust (retry 21:51, cool machine) | 2123 → 1288 (1.65) | 1204 → 363 (3.32; 57 → 28 %) | 367 → 308 | 503 → 585 | 149 → 150 | 750 → 750 | 0.021 → 0.015 | 0.004 → 0.001 | 0.009 → 0.007 |
| cigar_D15_exhaust † (first run 17:07, throttled) | 2123 → 1838 (1.15) | 1204 → 511 (2.36; 57 → 28 %) | 367 → 379 | 503 → 900 | 149 → 150 | 750 → 750 | 0.021 → 0.015 | 0.004 → 0.001 | 0.009 → 0.007 |

**Retry (2026-09-04 21:51–22:12, `runs/profile_20260904_retry/`, code
`93dc29d`, machine idle for four hours; speed probe just before: `banana_D4`
45.5 s against 42.8 s at 16:51 and 56.5 s hot).** The trajectory is
bit-identical to the throttled run (same K path, same ELBO at every
iteration, same finals), so the two rows measure the same computation on
two machine states: wall 1838 → 1288 s, active sampling 511 → 363 s, GP
training 379 → 308 s, variational fit 900 → 585 s. Against the 2026-09-03
run the exhaust configuration is **1.65× faster end to end**, its active
sampling **3.3× faster** (per optimize-only iteration 7.35 → 1.89 s: one
hyperparameter sample, so gpyreg's per-sample overhead is small there and
the call-count reduction shows in full), GP training 16 % faster (the same
three refits, faster hyperparameter reuse) and the variational fit 16 %
*slower* (per tail iteration with no refit and no K change 2.77 → 3.21 s,
n = 46/50; a different trajectory with a different Adam path, and the
machine still ~6 % warm), not doubled. The throttling factor on the first
run's tail was therefore about 1.5, matching the 17:47 probe.

† **The first exhaust run's tail was not a valid measurement: the laptop was
throttling.** Its variational fit doubled (503 → 900 s) on code that was
not touched, at equal K and N (median 2.8 → 6.1 s per optimize-only
iteration with no refit and no K change; mean K 25.9 vs 26.3 over the
tail, three GP refits in each, 33 vs 30 K changes). GP training, also
untouched, tells when: per iteration it is equal or faster in the new run
up to N = 325 (1.39 → 1.24 s over the first 20 iterations, 8.0 → 6.7 s in
the sampling regime at N ≥ 205), then 2.1–2.5× slower at N = 330–345
(iterations 66–69: 24.3/23.8/23.1/19.1 s against 9.6/10.7/11.0/12.7 s),
the three tail refits are 1.8× slower by median (4.1 → 7.4 s), and the
cheap hyperparameter-reusing iterations are unchanged (0.29 → 0.28 s). So
the slowdown set in around iteration 66, about 27 minutes of sustained
single-core load after the campaign started, when this run, the
campaign's last, was entering its optimize-only tail. Confirmed by a
probe: `banana_D4` plain rerun at 17:47 on the same package code (tree
`eca45ec`, which differs from `f441172` only under `dev/`) took 56.5 s
against 42.8 s at 16:51, every stage 1.31–1.39× slower (GP training
10.1 → 13.4 s, variational fit 7.9 → 11.0 s), same 17 iterations, 95
evaluations and metrics. The laptop had been closed and resumed an hour
before the campaign. The nine converging configs ran in the campaign's
first 16 minutes, and their untouched stages are within ±3 % (a few
seconds either way, mixed signs) of the old run, so their numbers stand.
The retry above replaces this row; the residual risk that part of the
variational-fit doubling was real is closed by it (585 s on the identical
trajectory).

Reading:

1. **Noiseless targets run 1.4–1.8× faster end to end** (1.36–1.79);
   active sampling itself is 2.1–2.5× faster and drops from 54–69 % of
   wall to 36–47 %. GP training and the variational fit move by at most
   3 % (a few seconds, mixed signs), as they must (nothing in them was
   touched).
2. **Six of the nine trajectories are bit-identical to the 2026-09-03
   runs** (same iterations, evaluations and metrics to every stored
   digit: lumpy_D4, student_D4, logreg_D5, logreg_D5_noise3, lumpy_D10,
   banana_D10): the batched arithmetic flipped no CMA-ES ranking on those
   seeds. banana_D4, cigar_D4 and rosenbrock_D2_noise1 parted; their
   finals stay well inside the population spread (banana and rosenbrock
   improved on all three metrics; cigar_D4's gsKL went 0.0003 → 0.001
   against a population fence of 0.008).
3. **The noisy VIQR path gains 6–9 %**, as the profile predicted: its
   active-sampling bucket is the per-sample GP refits and VP
   re-optimizations (items 8 and 1), not the acquisition search.
4. The acquisition is called `2 n_gen + 1` times per search instead of
   `(popsize + 2) n_gen + 1`, i.e. 5× fewer calls at D = 4–5, but the
   bucket shrinks only 2.1–2.5× at D ≤ 10 (3.3× on the 15-D exhaust run,
   where a single hyperparameter sample leaves little per-call overhead
   to pay): a batched `GP.predict` on `popsize` rows
   costs more than a single-row one (the triangular solve scales with the
   number of columns, and gpyreg's per-sample Python loop and `sW` tiling
   are paid once per call either way), and the sieve (`2^13` points, one
   call per acquired point) is untouched. Where the remaining
   active-sampling time goes is the cProfile question below and the input
   to item 8.

### Where the time goes now: cProfile of the four D = 4 configs

`runs/profile_20260904_retry/<config>_cprof/` (22:12–22:19, cool machine:
speed probes 45.5 s before and 42.6 s after, against 42.8 s at the
campaign's start) against the 2026-09-03 pass. Percentages of profiled
`VBMC.optimize`; call counts in parentheses (old → new). A first pass at
17:38 (`runs/profile_20260904/`) ran on the throttling machine; its
percentages and call counts agree with this one within a few points, its
absolute times were inflated 1.1–1.65×, and it is superseded. Absolute
times are now consistent: banana_D4's 3.09k → 3.14k `_neg_elcbo` calls
took 18.4 s before and 17.5 s now; the profiled wall itself falls 95 → 58,
213 → 105, 84 → 58 and 102 → 65 s (1.6–2.0×, more than the plain 1.5–1.8×,
because cProfile's own per-call overhead shrinks with the call count).

| bucket | banana_D4 | cigar_D4 | lumpy_D4 | student_D4 |
|---|---|---|---|---|
| active_sample | 60.9 → 40.0 | 63.8 → 43.8 | 54.6 → 34.5 | 63.8 → 43.5 |
| ├ cma.fmin | 51.7 → 25.6 | 57.4 → 33.0 | 45.9 → 21.6 | 53.2 → 27.0 |
| ├ acquisition `__call__` (calls) | 53.3 → 29.2 (50.7k → 10.0k) | 55.4 → 29.9 (130.7k → 24.1k) | 48.0 → 25.4 (40.5k → 7.9k) | 55.9 → 32.2 (58.4k → 11.4k) |
| ├ `GP.predict` (calls) | 44.2 → 26.5 (50.8k → 10.1k) | 44.6 → 26.4 (130.8k → 24.2k) | 39.7 → 23.2 (40.6k → 8.0k) | 46.2 → 29.3 (58.5k → 11.5k) |
| └ `vp.pdf` (calls) | 4.7 → 1.1 (50.7k → 10.0k) | 5.4 → 1.4 (130.7k → 24.1k) | 4.4 → 0.9 (40.5k → 7.9k) | 5.1 → 1.2 (58.4k → 11.4k) |
| train_gp | 18.3 → 27.4 | 14.1 → 24.0 | 21.5 → 31.8 | 20.1 → 31.3 |
| ├ `SliceSampler.sample` | 15.8 → 23.5 | 12.1 → 20.8 | 19.2 → 28.4 | 17.6 → 27.4 |
| ├ `GP.__core_computation` (calls) | 12.8 → 19.0 (58k → 57k) | 10.0 → 16.8 (97k → 89k) | 14.8 → 22.0 (65k → 65k) | 14.1 → 22.1 (69k → 69k) |
| └ `solve_triangular` (calls) | 9.6 → 8.4 (633k → 280k) | 9.1 → 7.7 (1.41M → 477k) | 9.3 → 8.2 (585k → 285k) | 10.2 → 9.2 (712k → 304k) |
| optimize_vp | 19.8 → 31.1 | 21.3 → 30.9 | 22.9 → 32.3 | 15.2 → 23.6 |
| ├ `_neg_elcbo` (calls) | 19.3 → 30.2 (3.1k → 3.1k) | 20.8 → 30.2 (7.1k → 6.7k) | 22.4 → 31.5 (2.4k → 2.4k) | 14.7 → 22.8 (2.7k → 2.7k) |
| ├ `_gp_log_joint` | 14.6 → 23.5 | 14.9 → 22.6 | 17.2 → 24.2 | 11.6 → 18.2 |
| ├ `minimize_adam` | 11.0 → 15.9 | 13.7 → 19.2 | 11.6 → 16.5 | 7.0 → 11.1 |
| ├ `_eval_full_elcbo` | 3.6 → 6.3 | 3.3 → 4.6 | 4.8 → 6.6 | 3.3 → 4.8 |
| └ `entmc_vbmc` (calls) | 4.3 → 6.1 (2.0k → 1.9k) | 5.6 → 6.9 (4.6k → 4.4k) | 4.8 → 6.7 (1.3k → 1.3k) | 2.7 → 4.2 (1.5k → 1.5k) |
| final_boost | 10.0 → 13.7 | 7.5 → 10.0 | 12.4 → 16.9 | 7.0 → 10.7 |
| `copy.deepcopy` | 0.4 → 0.6 | 0.4 → 0.6 | 0.4 → 0.5 | 0.4 → 0.6 |
| profiled wall s | 95 → 58 | 213 → 105 | 84 → 58 | 102 → 65 |

Reading:

- **The acquisition is called 5× less often** (popsize + 2 → 2 per
  generation, as designed: e.g. 50.7k → 10.0k on banana_D4, 130.7k → 24.1k
  on cigar_D4) and `GP.predict` with it. The solve-triangular call count
  in GP *training* also halves (633k → 280k): those are the per-point
  `predict` calls' solves, which gpyreg's `predict` makes once per
  hyperparameter sample per call.
- **`GP.predict` is still 23–29 % of the run at 8k–24k calls**, i.e.
  1.2–1.7 ms per batched call against 0.7–0.8 ms per single-row call in
  the 2026-09-03 pass: the cost of a `predict` call is dominated by gpyreg's Python
  loop over the `Ns` posteriors (kernel evaluation, `sW` tiling to
  `(N, N_star)`, two triangular solves, per sample) and grows only mildly
  with the number of rows. That per-call overhead, over `Ns`, is exactly
  the gpyreg half of item 3 that moved to item 8, and it is now the
  largest single remaining piece of active sampling. `vp.pdf` is 0.9–1.4 %.
- **The other stages' shares grew because the denominator shrank**: GP
  training 24–32 % (slice sampler 21–28 %, `__core_computation` 17–22 %),
  the variational stage 24–32 % (`_gp_log_joint` 18–24 %), `final_boost`
  10–17 %, with call counts within ±6 % of before (`_neg_elcbo` 3.09k →
  3.14k on banana, 7.1k → 6.7k on cigar, equal on lumpy and student). On
  these D = 4 targets the three stages are now of comparable size, which
  is what the roadmap's revised weighting of items 8 and 1 says.

### `vp.pdf` chunk size (found while investigating the exhaust run)

Timing the old loop against the broadcast at fixed shapes (one thread,
`scratchpad`, median of 3):

| shape (D, K, N) | loop | 2^14 | **2^16** | 2^18 | 2^22 |
|---|---|---|---|---|---|
| 15, 26, 1e5 | 303 ms | 230 | **159** | 379 | 436 |
| 15, 50, 1e5 | 715 | 388 | **372** | 788 | 906 |
| 10, 15, 1e5 | 155 | 103 | **98** | 162 | 230 |
| 4, 12, 1e5 | 64 | 60 | **51** | 68 | 103 |
| 15, 26, 8192 (the sieve) | 13.1 | 16.3 | **16.8** | 34 | 38 |
| 4, 50, 8192 | 11.4 | 18.2 | **12.5** | 22.8 | 27.3 |
| 15, 26, 8 (a CMA-ES batch) | 0.33 | 0.03 | **0.04** | 0.06 | 0.05 |
| 20, 50, 1e5 | 1100 | 484 | **434** | 900 | 842 |

The 2^22-element chunks of the first commit (33 MB per `(n, K, D)`
temporary) were memory-bound: slower than the loop by 1.3–1.7× on the
1e5-row calls (`vp.mode`, `vp.mtv`, `kl_div` with sampling) and by 30 %
at the sieve. With 2^16 elements (0.5 MB, in cache) the broadcast is
1.2–1.9× faster than the loop on 1e5 rows, 5–10× on a CMA-ES batch, and
within 10–30 % of the loop at the sieve (one call per acquired point:
a few seconds per run). Rows are independent, so the chunk size changes
no result; committed as `e923163`. In VBMC itself the
large-N calls are rare (`kl_div` in the main loop uses the Gaussian
approximation, `kl_gauss`), so this matters for the user-facing `mode`
and `mtv` more than for a run.

## Follow-ups

- ~~Re-profile `cigar_D15_exhaust` plain on a cool machine~~ done 21:51
  (§Results). Keep the lesson: check the laptop's thermal behaviour before
  any long campaign, and interleave a short reference config (e.g.
  `banana_D4`) at the start and the end of a campaign as a built-in speed
  probe (a `profile_suite.py` option is the natural home).
- Item 8 (gpyreg PR) now also owns the per-call `predict` overhead: the
  Python loop over the posteriors, the `sW` tiling to `(N, N_star)`, the
  two triangular solves per sample; at 5× fewer calls it is still 23–29 %
  of a D = 4 run. `_sq_dist`'s batch-mean centring in the noisy
  acquisitions is worth making batch-invariant at the same time.
- The 20-seed golden population against `dev/golden/baseline` once at the
  end of Stage 2 (about 7 h now).
- The noisy path's remaining acquisition cost is the pointwise
  `is_log_full` calls inside `active_importance_sampling`'s MCMC (found by
  the fact-check); out of item 3's scope, a candidate for item 8 or a
  small item of its own.
- `test_active_uncertainty_sampling` is the only unit test on the CMA-ES
  path and asserts the found points to `atol = 1e-3`; a unit test of the
  batched objective's contract (list in → list out, array in → float,
  integer write-back) would pin the plumbing directly.

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
  reconfigured to UTF-8 when redirected on Windows), and the iteration
  column was made unambiguous (later replaced by the count of identical
  leading iterations, after the design review)
- [x] Step 2 code in place (batched `acq_fun`, `parallel_objective`,
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
  seeds down, two up (sign test p = 1; paired t on log ratios p ≈ 0.4),
  so no shift is established, but n = 5 has no power and the two upward
  moves are large (2.9× and 9.7× against 1.2–1.8× downward, geometric
  mean 1.6× upward; seed 4 sits at 95 % of the gsKL fence): the 20-seed
  population comparison at the end of the stage is the arbiter
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
- [x] Commits — 15:37: `50c1e50 test(dev)` (replay gate, `--rebaseline`
  mode, records) then `7a07c0b perf(vbmc)` (the batched objective, the
  noise handler, the four re-baselined step-oracle references)
- [x] Step 3 `vp.pdf` broadcast written — 15:38 (Gaussian branch; rows
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
- [x] Step 3 replay, first attempt, 15:39 → **killed at 15:43 on the PI's
  request (laptop needed)** after 4 of 5 configs, all with the same
  verdicts and finals as the Step 2 replay. The pickup note written then
  (Step 3 uncommitted; rerun the replay, the full suite, commit, profile,
  records, doublecheck; Step 2 commits local) was followed from 16:37 and
  is superseded by the entries below.
- [x] Resumed 16:37 (PI); Step 3 replay rerun in full, 16:37–16:44 →
  `runs/golden/replay_step3_pdf_broadcast/`: **the same verdicts and the
  same finals as the Step 2 replay on every config** (the `K`-sum
  reordering did not flip a single ranking on these runs); the only flag
  is halfnormal's seed-0 gsKL, already shown to be chance. Wall, baseline
  → Step 2 → Step 3 (min): normal_D5 1.0 → 0.81 → 0.72; banana_D2 0.59 →
  0.52 → 0.46; halfnormal_D2 0.54 → 0.49 → 0.49; cigar_D4 2.4 → 1.4 →
  1.3; rosenbrock_D2_noise1 2.0 → 2.0 → 1.9
- [x] Full suite for Step 3 — 16:45–16:52: **517 passed, 15 skipped, 0
  reruns, 7:09** (the Step 2 gate run at 15:27 took 9:30; both runs had
  the batched search, so the difference is not attributed)
- [x] Commit — 16:53: `perf(vp): broadcast pdf over the mixture components`
- [x] Step 4 profile campaign (plain) 16:55–17:37 →
  `runs/profile_20260904/` (§Results); cProfile pass on the four D = 4
  configs 17:38–17:46. **Throttling found**: the exhaust run's tail and
  the cProfile pass ran on a throttled CPU (probe 17:47: `banana_D4`
  56.5 s vs 42.8 s at 16:55 on the same code and trajectory); the D ≤ 10
  plain numbers, taken in the campaign's first 12 minutes with untouched
  stages matching the old run, stand; the exhaust row is a lower bound
  and its variational-fit doubling is a machine artefact
- [x] `vp.pdf` chunk size 2^22 → 2^16 after timing the broadcast against
  the loop (§Results): the first version was memory-bound and slower than
  the loop on 1e5-row calls; results unchanged (rows independent), VP and
  oracle tests green, bit check vs the loop as before
- [x] Read-only Opus review of the three commits — 16:55–17:12: no
  blockers; should-fixes applied in `eca45ec` (replay: identical runs
  exempt from the envelope, NaN finals flagged, empty report exits 1,
  provenance kept on re-render, 0-based iterations; re-baseline: two-phase
  with temp files, shape check, finite JSON) and the records aligned
  (AGENTS.md exception paragraph, open question 2, this tracker)
- [x] Commits `e923163 perf(vp)` (2^16 chunks) and `3033526 docs(dev)`
  (results, roadmap, devlog) — 17:58; pushed `348dd0e..3033526`
- [x] `/doublecheck` — 18:00–18:30: three read-only Opus verifiers (final
  code state; records consistency; every number against the run files).
  Every table cell of both profile tables and all four replay reports
  reproduced; corrections folded in: the exhaust paragraph's GP-refit
  figure was the variational fit at the refit iterations (the GP-training
  evidence is the 2.1–2.5× slower iterations 66–69 and the 1.8× slower
  refits), "first 90 iterations" → about 65, "first 12 minutes" → 16,
  campaign start 16:51, cProfile inflation 1.1–1.65×, "equal or better"
  → within the population spread (cigar_D4 gsKL 3× worse), "to the
  second" → within 3 %; stale 2^22 mentions, the chronology of the killed
  replay, the mislabelled 9:30 comparison, the missing `e923163`, the
  noise-handler rationale and the Step 1 design text updated; devlog
  next-step sentence removed (roadmap owns it); AGENTS.md and README
  wording tightened (`da0aa91`). Code verifier: no blocker; the batched
  objective, the `fmin` wiring and the noise handler confirmed against
  cma 4.4.4, `pdf`'s gradient bit-identical up to K = 1000; two
  should-fixes applied: the replay's `identical` now also requires
  identical `y_orig` (a noisy target whose noise stream moved would
  otherwise have been exempt from the envelope), and the re-baseline
  mode's post-write check is scoped to the re-baselined oracle (exact)
  with the other oracles at their own tolerances (the broadcast `pdf` no
  longer reproduces the stored `vp_pdf` references bit-exactly at K ≥ 8,
  by ~1e-15, so the old exact check would have aborted the next
  re-baseline after its first write); temp files now live outside the
  fixtures directory and are removed on failure; `--rebaseline` refuses
  `--list`/`--check`; `--report-only` restores `threads`; a missing new
  trace degrades to finals only. CI smoke of `3033526` green
  (33886022057)
- [~] **Retry of the throttled measurements (PI, 21:49)**: speed probe
  `banana_D4` plain 45.5 s (42.8 s at 16:51, 56.5 s hot at 17:47; GP
  training 11.1 s, variational fit 8.7 s), so the machine is back within
  ~6 % of its cool speed; `cigar_D15_exhaust` plain 21:51–22:12 →
  `runs/profile_20260904_retry/`: **1288 s (2123 before the change,
  1838 throttled), bit-identical trajectory to the throttled run,
  variational fit 585 s (503 old, 900 throttled)**: the doubling was the
  machine (§Results). cProfile pass on the D = 4 set 22:12–22:19,
  percentages as before and absolute times now consistent (§Results
  table replaced). Closing probe `banana_D4` 42.6 s = the campaign's
  start (42.8): no throttling during the retry. The §Results and the
  roadmap/devlog sentences on the exhaust run updated; the follow-up
  ticked
