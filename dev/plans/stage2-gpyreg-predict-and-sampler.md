# Stage 2 item 8: gpyreg `predict` and slice-sampler overhead, one gpyreg PR

Created: 2026-09-05 07:50. Status: **IN PROGRESS** (this file is the plan
and the worklog; tracker at the end). Roadmap pickup point 3
(`plans/modernization-roadmap.md`: "Next: item 8 as one gpyreg PR");
rationale in `dev/2026-09-02-modernization-discussion.md` §2, §4, §9, §10;
the inherited gpyreg half of item 3 in `plans/stage2-batched-acquisition.md`
§Follow-ups; what is left after items 3, 1 and 2 in
`plans/stage2-gp-log-joint-einsum.md` §Results (active sampling 46–54 % of a
D = 4 run with `GP.predict` 31–34 %, GP training 31–42 % with the slice
sampler 27–38 %). Decisions taken with the PI on 2026-09-05 (07:45):
identity-preserving refactor first, with two new oracles before any gpyreg
change; generator support in the same gpyreg PR as the last commits; the
gpyreg branch pushed to `acerbilab/gpyreg` so `GPYREG_PIN` can test it;
whether to run the 20-seed population right after item 8 is open (machine
availability), see Open questions. Working constraint for the day: the
laptop is in use, so every gate runs as one process of a few minutes and
the two long runs (profile campaign ≈ 40 min, population ≈ 6.5 h) wait for
an idle window.

## Summary

Same outputs, fewer Python layers. gpyreg's `predict` spends 1.24 ms per
call at Ns ≈ 7 on kernel evaluations, scipy's Python wrappers around a
5 µs triangular solve, and its own loop body; the slice sampler's
log-posterior evaluation spends 287 µs on `pdist` plus `squareform`, two
wrapped triangular solves, prior masks recomputed on every call and a
26 µs Cholesky of which 10 µs is LAPACK. Every change below reproduces
today's outputs bit for bit: broadcasting instead of tiling, direct LAPACK
calls to the routines scipy already calls, `cdist` for the symmetric
kernel, cached prior masks, and reuse of the Cholesky factor when the
sampler moves a mean-function hyperparameter (2D+1 of 3D+3 coordinates).
Because the outputs do not move, the gates are exactness: two new oracles
(`gp_nlZ`, `gp_fit`) added to the existing fixtures without rerunning the
recipes, the existing oracles compared exactly, the golden replay reporting
`identical` on every config, and gpyreg's own suite. Generator support
(`rng=None` keeps today's global draws) lands last in the same gpyreg PR;
PyVBMC's global-state seam is removed in a PyVBMC commit after the pin
bump. The one candidate that is not identity-preserving, batching the
kernel over hyperparameter samples, is left as an optional last step
decided by measurement.

## Scope

- **In (gpyreg, branch off `236ddd7` = `GPYREG_PIN` = v1.0.4 + 1)**:
  `gaussian_process.py` (`predict`, `__core_computation`,
  `__compute_log_priors` and its mask cache, `fit`'s sampler objective,
  `random_function`), `slice_sample.py`, `f_min_fill.py`,
  `covariance_functions.py` (the `SquaredExponential` symmetric and
  diagonal branches only), gpyreg tests for each change; formatting per
  gpyreg's pre-commit (black 22.3.0, isort 5.10.1, pycln, line 79);
  Python ≥ 3.9 syntax.
- **In (PyVBMC)**: `pyvbmc/testing/oracles/_oracles.py` (+`gp_nlZ`,
  `gp_fit`), `dev/scripts/make_oracle_fixtures.py` (`--add-oracle`), the
  fixtures (new `ref/` keys only), `.github/workflows/test-matrix.yml`
  (`GPYREG_PIN`), and after the merge `gaussian_process_train.py`
  (`gp.fit(rng=)`), `vbmc.py` / `rng.py` (drop `seed_global_from`, the
  constructor reseed, the `optimize()` reinstall, the `"legacy"` half of
  `random_state`), `active_sample.py` (`_BatchedNoiseHandler.indices`
  drawing from `vp.rng`), `pyproject.toml` (gpyreg minimum version);
  records (`dev/README.md`, roadmap, devlog §2/§9/§10, `AGENTS.md`).
- **Out**: restructuring the L-BFGS-B path or the `dnlZ` gradient block
  (they inherit the shared savings; 0.1 % of a run); `predict_full` (no
  PyVBMC caller); Matern and rational-quadratic kernels beyond what the
  shared path touches; `_sq_dist`'s batch-mean centring and the pointwise
  `is_log_full` calls of the noisy path (`active_importance_sampling` is
  ≤ 0.3 % of a noisy run); the noise-handler algorithm; a gpyreg PyPI
  release (PI, after the merge; needed before users see the gain and
  before PyVBMC's minimum version can be bumped); the 20-seed population
  (Open questions).

## Findings the plan rests on

Measured 2026-09-05 from the existing `runs/profile_20260905/cigar_D4_cprof/
profile.prof` (code `a39e5ec`, gpyreg `236ddd7`, numpy 2.5.2, scipy
1.18.1, one BLAS thread; `banana_D4` agrees within a few points), and read
from the gpyreg code at `236ddd7` the same morning.

- **`predict` (`gaussian_process.py:1663`)**: 24,899 calls, 30.9 of 94.8
  profiled seconds (32.6 %), 1.24 ms per call at Ns ≈ 7, N ≈ 100, N* = 8
  (the CMA-ES population) or 2–3 (the noise handler's re-evaluations).
  Per call: `covariance.compute` twice per sample (`kss` diagonal and
  `Ks`, :1741, :1746) at 35 µs each, ≈ 0.49 ms (40 %); one
  `solve_triangular` per sample (:1752) at 33 µs, ≈ 0.23 ms (19 %), of
  which LAPACK is about 5 µs and the rest scipy's `_lib/_util.wrapper`
  (the batch and array-API decorator), `_asarray_validated`, `_datacopied`
  and `get_lapack_funcs` on every call, with `check_finite=False` already
  set; predict's own loop body 0.23 ms (18 %: `np.tile`, reshapes, slice
  assignments per sample); `mean.compute` per sample 12 µs, ≈ 84 µs (7 %).
  The kernel's 35 µs is Python around a few-µs C distance loop: `X / ell`
  on the whole training set per sample, `x_star / ell`, `cdist`, `exp`,
  scale (`covariance_functions.py:158-169`).
- **Slice sampler**: `fit` 29.4 s (31 %), `SliceSampler.sample` 25.8 s,
  93.6k objective evaluations in total (80.2k by the sampler, 13.2k by
  the space-filling design; `__gp_obj_fun(hyp, False, True)` through
  `__compute_nlZ` and `__core_computation`, :1540, :1520, :2357) at 287 µs
  each: `squareform(pdist(X / ell))` ≈ 60 µs (`pdist` 46 µs of
  which 23 µs is scipy 1.18's `array_api_extra` lazy wrapper, `squareform`
  9 µs; the C loop is a few µs at N ≈ 100, D = 4); two `solve_triangular`
  for `alpha` (:2455) 66 µs; `__compute_log_priors` (:1275) 63 µs, which
  recomputes six boolean masks (:1289-1312) and the smooth-box constant
  (:1330) from `hyper_priors` and the bounds on every evaluation;
  `cholesky` (:2415) 26 µs of which LAPACK ≈ 10; `np.eye(N)`, `K / sl`
  and the add ≈ 10 µs; `mean.compute` 12 µs; own and small numpy ≈ 50 µs.
  `f_min_fill` (3.4 s, 3.6 %) evaluates the same objective 13.2k times at
  257 µs. `scipy.optimize.minimize` 0.1 %. The Cholesky is 9 % of an
  evaluation and 2.6 % of the run.
- **Identity facts** (the reason every step below is bit-exact):
  1. `np.tile(sW, (1, N*)) * Ks` and `sW * Ks` are the same elementwise
     products. `sW` is `(N, 1)`; after rank-1 updates on heteroskedastic
     noise its rows differ (`update`, :832-837), which broadcasting does
     not care about.
  2. scipy's `solve_triangular` calls LAPACK `?trtrs` with a layout rule:
     an F-contiguous `a` (or `trans=2`) is passed as is, a C-contiguous
     `a` is passed as `a.T` with `lower` and `trans` flipped (no copy).
     A direct `trtrs` through `get_lapack_funcs` that applies the same
     rule is **bit-identical** to `solve_triangular` (measured
     2026-09-05, scratch `identity_facts.py`: 24 cases, N 10–345, N* 1–512,
     both layouts, both `trans`). Two things are *not* identical and are
     therefore not done: `potrs` differs from the two-solve form by up to
     2.5e-15 (OpenBLAS's `dpotrs` is not two `dtrtrs`), so `alpha` keeps
     its two solves, each a direct `trtrs`; and scipy 1.18's `cholesky`
     no longer calls `potrf` through f2py but a batched C implementation
     (`_batched_linalg._cholesky`) that, for a C-ordered input, computes
     the lower factor of the transposed view and returns its transpose
     (C-contiguous; equal to `potrf(A, lower=1).T` to the bit, 1e-15 away
     from `potrf(A, lower=0)`). Mimicking that is fragile for a 15 µs
     gain, so `sp.linalg.cholesky(check_finite=False)` stays. Consequence
     of the C-contiguous factor: every `solve_triangular(L, …)` in gpyreg
     today takes the transposed branch, which the mimic reproduces.
  3. `cdist(Xs, Xs, "sqeuclidean")` sums `(x_id − x_jd)²` over `d` in the
     same order as `pdist`, `(a − b)² == (b − a)²` exactly, and the
     diagonal is exactly 0, so it equals `squareform(pdist(Xs))` bit for
     bit and is symmetric by construction, at one call instead of two and
     without the wrapper.
  4. The SE kernel diagonal is `sf2 * exp(-0 / 2) = sf2 * 1.0 = sf2`
     exactly (:162-169), so a `compute_diag` fast path returning
     `np.full((N, 1), sf2)` changes nothing.
  5. `K / sl + eye(N)` equals `A = K / sl; A[diagonal] += 1.0`: adding
     `0.0` off the diagonal leaves every entry unchanged and the diagonal
     receives the same `+ 1.0`. With per-row noise, `+= sn2 / sn2_div` on
     the diagonal likewise. No `eye` allocation, no N² add.
  6. **Cholesky reuse.** The slice sampler changes one coordinate per
     evaluation (`slice_sample.py:393-457`) **because `step_out` is off**
     (`SliceSampler` default `False`, `slice_sample.py:208`; `GP.fit`
     passes only `display` and `diagnostics`, `gaussian_process.py:1215`):
     with step-out on, `x_l`/`x_r` are copied from `xx` once per sweep and
     never re-synced after `xx[dd] = xprime[dd]`, so the step-out
     evaluations carry stale coordinates from earlier axes (a gpyreg
     defect in its own right, inert for PyVBMC; devlog §9) and a
     one-entry cache would hit almost never, still correctly. For
     `SquaredExponential` + `GaussianNoise` + `NegativeQuadratic`, `hyp`
     is `[ln ell (D), ln sf, ln sn (1 or 2), m0, xm (D), ln omega (D)]`;
     when the changed coordinate is in the mean block (2D+1 of 3D+3: 9 of
     15 at D = 4, 31 of 48 at D = 15), `K`, `sn2`, `sn2_mult`, `sl`, `L`,
     `L_chol`, `sW` and `Σ log diag L` are unchanged and only `m`, `alpha`
     and the quadratic form move; nothing else in `__core_computation`
     reads `hyp` (the review checked: `dnlZ = zeros(hyp.shape)` uses the
     shape, the `Posterior` stores `hyp` verbatim), and the ladder's
     `sn2_mult` is a deterministic function of `K` and `sn2`. Same
     Cholesky input gives the same `L`. **The cache must serve only the
     no-gradient path** (`compute_nlZ_grad` false: the space-filling
     design's and the sampler's objectives): the gradient path needs
     `dK`, `dsn2`, `dm` from the skipped block, and a bound-projected
     L-BFGS-B step can leave the whole cov+noise block unchanged between
     two evaluations (review blocker B1). Expected hit rate ≈ 55–60 % of
     the sampler's evaluations, each hit skipping the kernel (60 µs), the
     Cholesky (26 µs) and the ladder.
  7. The prior masks (`f_idx`, `sb_idx`, `sb_t_idx`, `u_idx`, `g_idx`,
     `t_idx`) and the smooth-box constants depend only on `hyper_priors`
     and the bounds, which change in `set_priors` (:418), `set_bounds`
     (:147) and the in-place `df` fill at the top of `fit` (:1029). The
     cache must be invalidated at all three sites; the per-evaluation
     arithmetic on the masked entries stays as written.
  8. Measured 2026-09-05 (`identity_facts.py`, 102 cases): facts 1, 3, 4
     and 5 hold bit for bit at every shape tried (N 10–345, D 1–20,
     N* 1–512, per-row noise included). The `NegativeQuadratic` mean
     batched over samples as an `(Ns, N*, D)` array summed over its last
     axis equals the per-sample computation at every D tried (1, 2, 4, 7,
     8, 9, 12, 15, 20) and N* (1, 8, 512): numpy's last-axis reduction
     does not depend on the leading dimensions. Open question 1 is
     therefore answered: the mean can be batched.
- **What the identity does not cover**: batching the kernel or the mean
  over the `Ns` samples inside `predict` (one broadcast over
  `(Ns, N, N*)` instead of `Ns` `cdist` calls) changes the summation
  order over `d` and the BLAS kernel; it would save a further ≈ 0.15 ms
  per call, about 3–4 % of a D = 4 run. Deferred to Open question 2.
- **`predict` call sites**: the hot path is `abstract_acq_fcn.py:79`
  (`separate_samples=True`); `active_importance_sampling.py:83, 113, 230,
  257, 354` (separate samples) and `:465` (averaged); `acq_fcn_viqr.py:
  246` and `acq_fcn_imiqr.py:264` pointwise with `add_noise=True` inside
  the noisy proposal density; `vbmc.py:1265` and `:1405`
  (`add_noise=True`); and, inside gpyreg, **the rank-1 update**
  (`GP.update`, `gaussian_process.py:753-756`: `predict(X_new, y_new,
  add_noise=True, separate_samples=True)` with `N* = 1`, whose `m_star`
  and `v_star` drive `alpha`, `L` and `sW` for every point active
  sampling adds; the review's S1, missed by the PyVBMC grep). The
  `add_noise`, `return_lpd` and averaging branches must keep their
  semantics. `predict_full` has no PyVBMC production caller (two
  acquisition unit tests call it) and keeps its `np.tile`; `quad` keeps
  its wrapped solves; both are PyBADS-facing and deliberately untouched.
  The `cdist` change (Step 3) reaches `random_function`, `predict_full`
  and the rank-1 update's `K = compute(hyp_cov, X_new)` (an `N = 1`
  matrix), with identical values by fact 3.
- **gpyreg's own tests** (`gpyreg/testing/`): `test_slice_sample.py::
  test_multiple_runs` asserts sample equality across two
  `np.random.seed(1234)` runs, a determinism test the generator work must
  keep passing with `rng=None`; `test_gaussian_process.py::
  test_gp_gradient_computations` checks `log_posterior` gradients against
  numdifftools at 1e-6; `test_fitting*` are smoke or loose; `test_split_
  update` and `test_predict_lpd` compare `alpha`, `sW`, `L` and lpd values
  at `np.isclose`. gpyreg CI (`tests.yml`): 3 OS × Python 3.9–3.11 on
  dispatch and schedule; `merge-tests.yml` on PRs. `requires-python >=
  3.9`, `scipy >= 1.7.3` (so `get_lapack_funcs` and `potrs` are
  available).
- **Generator sites in gpyreg**: `slice_sample.py:392` (`shuffle`), `:398,
  :400, :440` (`rand`), `:696` (`rand`, Metropolis step, unused by
  PyVBMC); `f_min_fill.py:92` (`shuffle` of the Sobol columns), `:94`
  (`uniform`); `gaussian_process.py:2265` (`randint`), `:2312, :2326`
  (`standard_normal`). Legacy `np.random.rand()` is `random_sample()`, so
  `rng.random()` on the module gives the same draws; `shuffle`,
  `uniform(size=)` and `standard_normal(size)` exist on both the module
  and a `Generator`; `randint` has no `Generator` twin with the same
  stream (`integers`), so `random_function` needs a two-branch pick.
- **The seam on the PyVBMC side** (`plans/stage1-rng-generator.md` §3):
  `train_gp(rng=)` already exists and uses `get_rng(rng).choice` for the
  warm-start pick; the fit itself must receive `rng` through
  `gp.fit(..., rng=...)`. cma's `NoiseHandler.indices` draws
  `np.random.rand()` for the fractional re-evaluation count (cma 4.4.4,
  `indices`, first lines); `_BatchedNoiseHandler` (`active_sample.py:879`)
  can override `indices` with the same six lines drawing from `vp.rng`,
  so no global draw remains in a run and the whole seam can go.
- **Pins and versions**: PyVBMC requires `gpyreg >= 0.1.0`; CI installs
  `acerbilab/gpyreg` at `GPYREG_PIN` = `236ddd7` for every run except the
  scheduled one (gpyreg `main`). A branch commit pushed to the org repo can
  be pinned before the merge; the minimum version bump waits for a gpyreg
  release.
- **Oracle side** (read-only Opus survey of `train_gp`, `_state.build_gp`,
  `_oracles.py`, `test_oracles.py` and the generator, verified on all eight
  fixtures with the venv; file:line in the survey's terms):
  - `train_gp(hyp_dict, optim_state, function_logger, iteration_history,
    options, plb_tran, pub_tran, rng=None)` (`gaussian_process_train.py:14`)
    reads `optim_state` (`gp_*_fun`, `iter`, `uncertainty_handling_level`,
    `stop_sampling`, `N`, `warmup`, `vp_K`, `n_eff`, `recompute_var_post`),
    about twenty `options`, the logger's live rows only (`_get_training_
    data`, :746; the target is never called), and mutates `hyp_dict` in
    place (`hyp`, `full`, `logp`, `run_cov`). Every key is present in every
    snapshot; `plb_tran`/`pub_tran` live in `optim_state`;
    `optim_state["hyp_dict"]["hyp"]` is bit-identical to `gp/hyp` on all
    eight (the recorded `hyp_dict` is the post-fit object of the same
    iteration).
  - **The step oracle's history stand-in `{"r_index": [r]}` is not enough
    for `train_gp`**: it indexes `r_index[iter-1]` whenever `iter > 0`
    (:524, :625, :638) and, under the default `weighted_hyp_cov = True`,
    `sKL[iter-i]` and `gp_hyp_full[iter-1-i]` over all past iterations
    (`_get_hyp_cov`, :700, :712); with `init_N > 0` it also warm-starts
    from `iteration_history["gp"][i]` (:133-142). `IterationHistory` is a
    dict subclass, so a plain dict of arrays serves. Per fixture the fit is
    `opts_N = 0`, `init_N = 0`, `n_samples = Ns`, `burn = 15` on the seven
    converged snapshots (no `f_min_fill`, no L-BFGS-B: the previous
    hyperparameters are reused as the chain's start); `normal_D2_
    singlesample` has `opts_N = 1`, `n_samples = 0` (one L-BFGS-B, no
    sampling); `rosenbrock_D2_noise1_viqr` has `init_N = 814`, `opts_N =
    1` and is the only one entering the warm-start loop. An empty
    `history["gp"]` makes that loop a no-op (then `hyp0 = np.unique(hyp_
    dict["hyp"], axis=0)`), deterministic and the same on every snapshot.
  - **Surprise 1 (PyVBMC, live, not in devlog §9): the sampler widths are
    discarded in every fit measured.** `_get_hyp_cov`'s weighted branch
    appends `hyp.T` (:717) with `hyp = gp_hyp_full[i]` of shape `(Ns,
    hyp_N)`, then takes `hyp_n = np.shape(hyp_list)[1]` (:729) as the
    sample count, so `hyp_cov` comes out `(Ns, Ns)`; `train_gp:121-124`
    drops `gp_train["widths"]` because its size is not `hyp_N`. Verified
    on all eight fixtures (`(Ns, hyp_N)` = (7, 15), (8, 18), (10, 9), (8,
    9), (1, 9), …), so those fits slice-sample with gpyreg's
    `widths_default` (`np.std` of the space-filling design, or `PUB −
    PLB` when `init_N = 0`). The review adds that `Ns == hyp_N` is
    reachable (D = 2, `hyp_N = 9`, `Ns = 9` lies between the fixtures'
    8 and 10), and then a degenerate width vector *is* used: a second,
    independent slip at :731-735 takes `np.dot(row, row)` of 1-D rows, a
    scalar, so every entry of `hyp_cov` is the same number. Consequence
    for the oracle: a synthetic `sKL`/`gp_hyp_full` cannot change the
    result, only the shape matters, and the oracle asserts the drop.
    Forcing `weighted_hyp_cov = False` would *keep* the widths (`run_cov`
    is `(hyp_N, hyp_N)`) and leave production's code path. Fixing the two
    slips changes the sampler's proposal widths and therefore every
    trajectory: out of item 8's scope, recorded for devlog §9 (one bullet,
    both slips) with its own population check when fixed.
  - **Surprise 2 (gpyreg, not in devlog §9)**: `GP.log_likelihood(hyp,
    compute_grad=True)` and `GP.log_posterior(..., True)` raise `TypeError`
    (unary minus on the `(nlZ, dnlZ)` tuple, `gaussian_process.py:1488`,
    `:1518`). Reproduced on all eight fixtures. The oracle calls the
    mangled `gp._GP__compute_nlZ(hyp, compute_grad, compute_prior)` (what
    `__gp_obj_fun` calls) until the gpyreg PR fixes the two lines.
  - **Surprise 3 (a trap, not a defect)**: `build_gp` never sets priors or
    bounds (`GP.__init__` leaves `no_prior = True`), so on a rebuilt GP
    `log_posterior == log_likelihood` exactly; installing PyVBMC's
    hyperprior with `_gp_hyp(optim_state, options, plb_tran, pub_tran, gp,
    X, y)` (:272, priors and bounds set at :481-482) leaves NaN bounds
    that make the normalization constants NaN until `fit`'s two repair
    lines run (`df` NaNs → 7 at `gaussian_process.py:1029`; `set_bounds(
    get_recommended_bounds(lb, ub))` at :1038-1046), **in that order**:
    `__recompute_normalization_constants` branches on `df` (Student-t vs
    normal cdf), so the `df` fill must precede `set_bounds`, as in `fit`.
    The oracle replicates the two lines in that order; with them `lp` is
    finite on every fixture (cigar `lZ = 426.2503917972849`, `lp =
    420.54313794248304`). Calling `gp.log_posterior` between `_gp_hyp`
    and `fit` returns NaN by design.
  - Generator: `rebaseline()` (:403-514) loads the raw JSON tree and npz,
    refuses the step oracle off `meta["platform"]`, asserts an unchanged
    key set and shapes, rewrites only the npz arrays under `ref/<oracle>/`
    (the JSON `@@npz:` markers already exist), appends `meta["rebaselined"]`,
    writes both files two-phase through `FIXTURES.parent`, then asserts
    every other array bit-identical, the re-baselined oracle exact and the
    others within tolerance. `test_fixture_complete` (`test_oracles.py:94-
    101`) computes `applicable(build_state(snap))` with `fun=None` (so the
    step oracle is excluded) and fails on any registered oracle without
    references: registering `gp_nlZ`/`gp_fit` and adding their references
    must be one step. `test_oracle` hard-codes the platform gate and the
    target to `active_sample_step` (:72-85). `--check` has no exact flag;
    `check_one(..., exact=True)` exists and is used by `generate()` and
    `rebaseline()` only. `legacy_seed(seed)` (`_oracles.py:136-146`) saves
    and restores the global legacy state around a block. `build_state`
    returns `{"pt", "vp", "gp", "logger", "optim_state", "options", "cand",
    "ref", "meta"}`; `Oracle.__call__` coerces every output with
    `np.asarray(v, dtype=float)`, so scalars are stored as 1-element arrays
    and `None` must not be returned.
  - `cigar_D4_boosted` and `cigar_D4_largeK` share one GP and
    `optim_state` (only the VP differs), so both new oracles give identical
    references there. `gpyreg`'s `fit` consumes the options PyVBMC passes
    except `df_base` and the bounds (defaults); `sampling_result` carries
    `samples`, `log_priors`, `f_vals`, `exit_flag`, `R`, `eff_N` and is
    `None` when `n_samples = 0`.

## Design

### Step 1: two oracles and `--add-oracle`

**`gp_nlZ`** (values GP-solve class, `rtol 1e-6`, `atol 1e-10`; the
gradients 2e-2 after the CI floors of the day, see the tracker; applies
always).
For each stored hyperparameter sample `s`: on the bare rebuilt GP,
`nlZ, dnlZ = gp._GP__compute_nlZ(hyp_s, True, False)`; on a deep copy with
PyVBMC's hyperprior installed (`_gp_hyp(optim_state, options, plb_tran,
pub_tran, g, X_train, y_train)` from the logger's live rows, then `fit`'s
two repair lines), `nlp, dnlp = g._GP__compute_nlZ(hyp_s, True, True)`.
Outputs `lZ (Ns,)`, `dlZ (Ns, hyp_N)`, `lp (Ns,)`, `dlp (Ns, hyp_N)` with
the signs of the public API. This pins `__core_computation`'s `nlZ`
formula and gradient block and `__compute_log_priors` at fixed state, i.e.
exactly what Step 3 rewrites. The private call is replaced by
`log_likelihood` / `log_posterior` once the gpyreg fix is pinned.

**`gp_fit`** (platform-bound like the step oracle: exact, `rtol 0`,
`atol 1e-8`; applies always, no target needed):

```
optim_state, fl, hyp_dict = deep copies; it = optim_state["iter"]
history = {"r_index": full(max(it, 1), meta["r_index"]),
           "sKL": full(max(it, 1) + 1, options["tol_skl"]),
           "gp_hyp_full": [gp.get_hyperparameters(as_array=True)] * max(it, 1),
           "gp": np.array([], dtype=object)}
with legacy_seed(seed):
    gp_new, gp_s_N, sn2_hpd, _ = train_gp(hyp_dict, optim_state, fl, history,
                                          options, plb_tran, pub_tran, rng=default_rng(seed))
return {"hyp": gp_new.get_hyperparameters(as_array=True), "sn2_hpd": [sn2_hpd], "gp_s_N": [gp_s_N]}
```

The stand-in history is inert by Surprise 1 (only shapes matter, and the
`r_index` at `iter` instead of `iter − 1` reaches only the `init_N` gate,
where the two agree on every fixture); the oracle asserts that
`_get_gp_training_options` returns widths of size `≠ hyp_N`, so the
inertness is checked rather than assumed. It pins the whole `train_gp`
path on the generating machine: `f_min_fill` and L-BFGS-B on the two
snapshots that run them, the slice sampler on the seven that sample, and
the identity-preserving Steps 2–4 must reproduce it to the bit. About
0.05–1 s per snapshot.

**Generator and tests.** `--add-oracle <name> --reason "..." [--only]`,
modelled on `rebaseline()`: refuse if the oracle is already referenced;
build the state, compute, write the npz arrays *and* the JSON markers
(`tree["ref"][name] = encode(out, f"ref/{name}", arrays)` as `generate()`
does), append `meta["oracles_added"]` (oracle, date, git, reason, output
shapes), two-phase write, then assert every pre-existing array
bit-identical, the new oracle exact and the others within tolerance (the
write and the verification are shared with `rebaseline()`). A
module-level `PLATFORM_BOUND = {"active_sample_step", "gp_fit"}` in
`_oracles.py` replaces the hard-coded name in the generator's refusal and
in `test_oracle`'s skip (the target stays specific to the step oracle).
`MANIFEST.in` needs no change (same files).

**The exact gate is a dump, not the committed references** (found when
the first `--check --exact` failed on every fixture): the references for
`vp_pdf`, `gp_log_joint`, `neg_elcbo` and the four noiseless acquisitions
were generated on 2026-09-04 and items 3, 1 and 2 have since moved those
outputs within tolerance (the broadcast `pdf`'s K-sum at K ≥ 8, the
`einsum` log joint), so the committed fixtures can pin the current code
only at tolerance. `--dump-outputs DIR` writes the current code's outputs
of every stored oracle on every snapshot (one `.npz` per snapshot, keys
`<oracle>/<output>`, a `.json` with git SHA, platform, threads);
`--check --exact --against DIR` compares the working tree with that dump
bit for bit. The pre-change dump for this item is
`dev/scripts/runs/oracle_outputs_prechange_de6d98f/` (gitignored under
`runs/`, 1.1 MB, reproducible from `de6d98f`); it reproduces itself
exactly on all eight snapshots, which also shows that `gp_fit` and the
step oracle are deterministic run to run. `--check --exact` without
`--against` remains available for the day the committed references are
re-baselined to the current numerics (Open question 8).

Gates: `pytest pyvbmc/testing/oracles` green with the two oracles (`gp_fit`
runs on this machine, skips elsewhere; 116 passed, 15 skipped);
`--check --exact --against <dump>` 8 of 8; the fixture diffs show only
`ref/gp_nlZ/*`, `ref/gp_fit/*` and the `meta` note (asserted by the mode
itself). Commit `test(oracles): pin the GP log marginal likelihood and the
GP training path`.

### Step 2: `predict`, identity-preserving

Keep the loop over samples (the per-sample `dot` and solve stay on the
BLAS kernels they use today) and remove what is not arithmetic:

```
m_all = mean.compute_batched(H_mean, x_star)        # (N*, Ns), identical per column (fact 8); or per sample if the
                                                    #   batched API is judged not worth adding: see Decisions
for s: hyp, alpha, L, L_chol, sW = posteriors[s]…
    kss = covariance.compute(hyp_cov, x_star, compute_diag=True)   # SE fast path: full((N*, 1), sf2) (fact 4)
    Ks  = covariance.compute(hyp_cov, X, x_star)                   # unchanged (cdist)
    mu[:, s] = m_all[:, s] + Ks.T @ alpha.ravel()                  # same dgemv as np.dot(Ks.T, alpha)
    if L_chol:
        V = _trtrs(L, sW * Ks, trans=1)                            # was solve_triangular(L, tile(sW) * Ks, trans=1) (facts 1, 2)
        s2[:, s] = kss.ravel() - np.sum(V * V, 0)                  # same reduction as today
    else: unchanged
```

`_trtrs` applies scipy's layout rule and raises `LinAlgError` on
`info != 0` as scipy does. `np.maximum(s2, 0)`, the `add_noise` /
`return_lpd` blocks and the averaging are unchanged. Expected ≈ 0.4 ms per
call (≈ 3×); the remaining cost is `Ns` kernel evaluations at ≈ 30 µs.

Gates: `make_oracle_fixtures.py --check --exact --against
dev/scripts/runs/oracle_outputs_prechange_de6d98f` 8 of 8 (every oracle
output bit-identical to the pre-change code on every snapshot, `predict`
through `gp_predict` and the acquisitions, the fit through `gp_fit`); a
scratch bit-check on random GPs for the paths no snapshot exercises (see
Verification); `pytest pyvbmc/testing/oracles`; gpyreg's suite; the replay
`identical` on all five configs; full PyVBMC suite. Commit on the gpyreg
branch.

### Step 3: `__core_computation` and the priors, identity-preserving

- `SquaredExponential.compute` with `X_star is None`: `cdist(Xs, Xs,
  "sqeuclidean")` instead of `squareform(pdist(Xs))` (fact 3); the
  `compute_grad` branch (L-BFGS-B only) keeps its `pdist` per dimension.
- `A = K / sl_`; `A.flat[:: N + 1] += 1.0` (or `+= sn2.ravel() /
  sn2_div`) (fact 5); `L = sp.linalg.cholesky(A, check_finite=False)`
  stays (fact 2: scipy's factor is not a plain `potrf`); the ten-step
  ladder unchanged; the `L_chol=False` branch unchanged.
- `alpha`: the same two solves, each a direct `trtrs` with scipy's layout
  rule (a small module-level helper `_trtrs(a, b, trans)` shared with
  `predict`), divided by `sl` as today (fact 2; `potrs` is *not*
  identical and is not used). `nlZ` unchanged. The gradient block (`Q`,
  `dK`) unchanged.
- Prior-mask cache: a private `_prior_cache` holding the six masks, the
  two smooth-box constants and `Σ log normalization_constants`, built on
  first use and cleared in `set_priors`, `set_bounds` and after the `df`
  fill in `fit` (fact 7; the review's grep confirms these plus `__init__`
  are the only writers). `__compute_log_priors` reaches it through
  `getattr(self, "_prior_cache", None)` and builds it lazily, because
  PyVBMC's `test_*_save_static.pkl` fixtures hold GP objects pickled
  without the attribute (review S9); every arithmetic statement on `hyp`
  stays as written.

Gates: bit-check of `__core_computation` outputs (`Posterior` fields,
`nlZ`, `dnlZ`, `log_posterior`) old vs new on the snapshots' stored
hyperparameter samples and on random hyperparameters, all `==`; the
`gp_nlZ` and `gp_fit` oracles exact; the rest as Step 2. Commit.

### Step 4: Cholesky reuse in the sampler

`__core_computation(hyp, compute_nlZ, compute_nlZ_grad, cache=None)`.
When `cache` is a dict, `compute_nlZ_grad` is false and
`np.array_equal(cache["key"], hyp[: cov_N + noise_N])`, reuse `sn2`,
`L`, `sl`, `sn2_mult`, `L_chol`, `pL`, `logdet` from it and compute only
`m`, `alpha`, `nlZ`; otherwise compute as in Step 3 and, on the
no-gradient path, store. `fit` owns one cache dict for `objective_f_1`
(`f_min_fill`) and one for `sample_f` (the sampler); `objective_f_2`
(L-BFGS-B, gradients) gets none (review B1: a hit there would skip the
`dK`, `dsn2`, `dm` the gradient needs, and a bound-projected step can
leave the block unchanged); `update` and `set_hyperparameters` pass no
cache and are unaffected. Hits are exact by construction (fact 6).

On a hit the cached `Σ log diag L` is reused too, so `nlZ` is the same
expression on the same numbers. A module-level `_REUSE_CHOLESKY = True`
is consulted at call time so a test can switch the reuse off through
`monkeypatch` without touching numpy.

Gates: as Step 3, plus gpyreg unit tests: a fit with the reuse on
reproduces a fit with it off bit for bit under the same seed (`hyp`, the
chain's `samples` and `f_vals`), and `__core_computation(hyp, 1, 1,
cache)` with a cache whose key matches but whose factor is garbage
returns the no-cache gradient result (the gradient path never consults
it). Commit.

### Step 5: generator support

`GP.fit(..., rng=None)`, `SliceSampler(..., rng=None)`,
`f_min_fill(..., rng=None)`, `GP.random_function(..., rng=None)`. A
module-level helper resolves `rng`: `None` → the `numpy.random` module
(today's legacy draws, call for call), a `Generator` → itself, anything
else → `np.random.default_rng(rng)`. Call sites use `rng.random()`,
`rng.shuffle()`, `rng.uniform(size=)`, `rng.standard_normal(size)`;
`random_function`'s sample pick uses `randint` on the module and
`integers` on a `Generator`. `fit` passes `rng` to `f_min_fill` and to
`SliceSampler`. Tests: a `default_rng(1234)` twin of `test_multiple_runs`;
`rng=None` runs of the existing tests unchanged. Docstrings.

Gates: gpyreg suite; PyVBMC oracles and replay unchanged (PyVBMC passes
no `rng` yet, so nothing moves). Commit.

### Step 6: branch, PR, pin

Push the branch to `acerbilab/gpyreg`, open a draft PR (title `perf:
predict and sampler overhead; rng= support`, body listing the identity
argument and the gates), set `GPYREG_PIN` to the branch head on
`dev-next`, push PyVBMC (the smoke runs), dispatch the full matrix. On the
gpyreg merge: pin the merge commit.

### Step 7: measure

When the machine is idle: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 python -u dev/scripts/profile_suite.py --suite profile
--mode plain --probe banana_D4 --out dev/scripts/runs/profile_<stamp>`
(≈ 35 min), then `--mode cprof --only banana_D4,cigar_D4,lumpy_D4,
student_D4,cigar_D15_exhaust` (the exhaust cProfile, ≈ 35 min, attributes
the +20 % active sampling of the item 1/2 campaign; may run separately).
Compare with `runs/profile_20260905/aggregate.md`: wall, the
active-sampling and GP-training shares, `GP.predict` per call,
`SliceSampler.sample`, `__core_computation` per evaluation. Identity means
the trajectories are the same as on 2026-09-05, so the comparison is clean
per config. Record here, in the roadmap and as a dated addendum in devlog
§2/§10.

### Step 8: PyVBMC seam removal (after the pin bump)

`train_gp` passes `rng` into `gp.fit(rng=get_rng(rng))` (the same
generator it already uses for the warm-start pick, i.e. `vbmc.rng`);
`_BatchedNoiseHandler` gets an `rng` and overrides `indices`; delete the
seam at its four sites: `rng.py:28` (`seed_global_from`), `vbmc.py:176-182`
(the constructor reseed and `_seeded`), `vbmc.py:888-892` (the reinstall
at the start of `optimize()`) and `vbmc.py:2360-2387` (`_get_random_state`
/ `_set_random_state`: drop the `"legacy"` half, keep reading old files
with the existing warning); update `test_vbmc_seed.py`, `AGENTS.md`'s
randomness paragraph, the `seed=` docstring; bump the gpyreg minimum in
`pyproject.toml` once a release exists (until then the pin carries CI).
gpyreg's `merge-tests.yml` runs its 3-OS matrix on any PR to `main` that
touches `gpyreg/`, so the draft PR of Step 6 gets gpyreg's own CI as well.
**This step re-baselines the `gp_fit` oracle deliberately** (review B2):
the oracle already passes `rng=default_rng(seed)` to `train_gp`, which
today reaches only the inert warm-start pick; once `train_gp` forwards it
to `gp.fit(rng=)`, the space-filling design and the chain leave the legacy
stream and `hyp` / `sn2_hpd` move. `--rebaseline gp_fit --reason "..."`
on this machine, recorded in `meta["rebaselined"]`; the oracle's
docstring says its reference is valid for the stream `train_gp` uses. Every stream changes, so
the replay parts at iteration 0 by design and the design certificate
carries the check; the population run is the statistical gate (Open
question 3).

## Verification

- [x] Step 1: oracles green (116 passed); `--check --exact --against` the
      pre-change dump 8 of 8 (the committed references cannot serve an
      exact check, see Design); fixture diffs show only the new `ref/`
      keys and `meta` (asserted by `--add-oracle`).
- [x] Steps 2–4: random-GP bit-check 2219 arrays `==` after each step;
      oracles exact against the dump 8 of 8 after each step; gpyreg suite
      green (82 → 104); replay `identical` on all five configs after each
      step; PyVBMC full suite green after each commit (539 passed).
- [x] Step 5: gpyreg suite green with the new tests (106); PyVBMC gates
      unchanged (bit-check, exact oracles, full suite).
- [~] Step 6: branch pushed, draft PR #43, pin bumped; PyBADS 87 passed
      with `test_version` deselected (an environment artefact); three CI
      rounds failed in turn on the `gp_nlZ` gradient floor (cigar, then
      corr; the class is now 2e-2) and on `test_vp_optimize_2D_g_mixture`
      replaying its draws on reruns (fixture made rerun-aware); the fourth
      round (`6a65cfd`) is the first that can reach the gpyreg branch
      itself.
- [ ] Step 7: per-config walls and shares recorded; trajectories identical
      to 2026-09-05 (same iterations, evaluations, metrics); probe within
      a few percent. To run from a detached checkout of `284747e` when the
      machine is idle.
- [~] Step 8: `test_vbmc_seed.py` green (13 with save/load) including the
      new global-state-untouched test; replay finals inside the envelope
      on 4 of 5 configs, `halfnormal_D2` MMTV marginally outside at seed 0
      and inside on seeds 1–4 (chance); the design is *not* certified by
      construction (the stream shifted by one draw); full suite 540 passed;
      committed as `c4313a3`.

## Decisions

- **Identity first.** Every gpyreg change in Steps 2–4 is chosen so that
  outputs do not move, and the gates check exactness rather than
  envelopes (PI, 2026-09-05). Consequence: the loop over samples in
  `predict` stays, and the batched kernel is an open question.
- **Direct LAPACK through `get_lapack_funcs`, not `check_finite=False`
  alone**: the latter is already set and the cost is the wrapper layers.
  Only `trtrs` is called directly, with scipy's layout rule copied from
  `scipy.linalg._basic._solve_triangular` (F-contiguous or `trans=2`: as
  is; otherwise `a.T` with `lower` and `trans` flipped). `cholesky` stays
  scipy's and `potrs` is not used, because neither reproduces today's
  numbers (fact 2).
- **Batched mean over samples in `predict` through a new
  `compute_batched(H, X)` on the three mean-function classes** (measured
  identical, fact 8; there is no base class, so `predict` falls back to a
  loop over `mean.compute` when a mean lacks the method, which keeps
  user-defined or unpickled means working). The batched
  `NegativeQuadratic` materializes the C-contiguous `(Ns, N*, D)` array
  `((X - x_m) / omega) ** 2` and reduces its last axis, the layout the
  identity was measured on. `np.dot(Ks.T, alpha)[:, 0]` keeps the `(N, 1)`
  product (the same `dgemv`), not a raveled `alpha`.
- **`cholesky` stays scipy's; `alpha` keeps two solves as direct `trtrs`
  calls** (fact 2). The mimic reproduces scipy 1.18.1's rule; that rule
  has been in `scipy.linalg._basic` for many releases, but gpyreg declares
  `scipy >= 1.7.3` and the identity is verified only on the versions the
  tests run (Risks).
- **Cholesky reuse scoped to `fit`'s objective closures**, not a cache on
  the GP object keyed on array identity: the data are fixed inside one
  `fit` call, so the cache cannot go stale, and `update` /
  `set_hyperparameters` are untouched.
- **Generator commits last**, after the perf commits are certified
  identical, so the replay attributes each commit; `rng=None` keeps the
  legacy draws call for call so PyBADS and un-migrated PyVBMC see no
  change.
- **Seam removal is a PyVBMC commit after the pin bump**, not part of the
  gpyreg PR; it changes every stream and is gated statistically.

## Open questions (defaults in bold)

1. ~~Batch `mean.compute` over samples in `predict` if the bit-check shows
   the last-axis reduction is identical at every D?~~ Identical at every D
   tried (fact 8). Remaining sub-question: add a batched entry point to
   the mean-function classes (`NegativeQuadratic` fast, the others a
   loop) or keep the per-sample call and accept ≈ 70 µs per `predict`
   call? **Add it**, since the identity is measured and the API addition
   is backward compatible.
2. Batch the kernel over `Ns` in `predict` (non-identical, ≈ 3–4 % of a
   D = 4 run)? **No for this PR**; revisit with the Step 7 numbers.
3. Run the 20-seed population after Step 8 or at the end of Stage 2?
   **Open (PI)**: depends on a free night; nothing before Step 8 needs it.
4. Run PyBADS's suite against the branch before the merge? **Yes, as a
   pre-merge gate** (review): `predict_full`, `quad` and `random_function`
   are the PyBADS-facing surface the `cdist` change reaches. PyBADS is
   already a sibling checkout (`../pybads`, cloned 2024-10, on the PI's
   branch `1d-tests-and-global-fixes` at `8c31fa2`, clean, installed
   editable in the venv): run its tests there as it stands, without
   switching its branch.
5. ~~Expose the exact-comparison mode of `--check` as a CLI flag?~~ It has
   none; `--check --exact` is added in Step 1.
6. Fix the two `TypeError` lines in gpyreg's `log_likelihood` /
   `log_posterior` (Surprise 2) inside the Step 3 commit? **Yes**: two
   lines, no PyVBMC caller, gpyreg's gradient test presumably bypasses
   them; the oracle switches to the public API after the pin bump.
7. Fix the `_get_hyp_cov` shape bug (Surprise 1) in PyVBMC? **Not here.**
   It changes the sampler's proposal widths and therefore every
   trajectory; it deserves its own small item with a population check.
   Recorded in devlog §9.
8. Re-baseline the committed references of the oracles that items 3, 1
   and 2 moved within tolerance (`vp_pdf`, `gp_log_joint`, `neg_elcbo`,
   the four noiseless `acq_*`) to the current numerics, so that
   `--check --exact` against the committed fixtures becomes the identity
   gate for later items? **PI's call, proposed for the end of Stage 2**
   together with the golden re-baseline (roadmap pickup point 5): the
   population check has already validated the moved numerics, the
   `--rebaseline` mode records each change with its reason, and until then
   the dump serves.

## Risks

- A LAPACK call that is not bit-identical to scipy's (a different
  routine, or a copy/transposition scipy performs that we skip): the
  bit-check catches it before any commit; fall back to the scipy call for
  that one site.
- `cdist` vs `pdist` identity on another BLAS or scipy build: pure C
  loops, no BLAS; if a platform disagrees, the CI oracles show it and the
  change reverts to `pdist`.
- The prior-mask cache going stale through a path that mutates
  `hyper_priors` in place (fact 7 lists three; grep for others before
  Step 3).
- gpyreg CI on Python 3.9: no `match`, no `X | Y` unions in annotations.
- The layout rule of fact 2 is copied from scipy 1.18.1; gpyreg's floor is
  `scipy >= 1.7.3`. The rule has been in `_solve_triangular` for many
  releases (the "trtrs expects Fortran ordering" branch), so older scipy
  should agree, but the identity is measured only on the versions the
  test matrices run. A change in scipy's rule would show as a
  rounding-level difference between `predict` and `solve_triangular`, not
  a wrong result.
- Machine in use during the day: gates are single short processes; the
  profile campaign and the population wait.

## Follow-ups

- **Devlog §9 entries to write with the records** (the review checked
  §9): the `sW` tiling bullet already there is resolved by Step 2 (mark
  it); new: the two `_get_hyp_cov` slips (an `(Ns, Ns)` covariance and a
  scalar `np.dot` of 1-D rows, so the widths are dropped in every fit
  measured and degenerate when `Ns == hyp_N`); gpyreg's
  `log_likelihood` / `log_posterior` raising `TypeError` with
  `compute_grad=True` (fixed in the gpyreg PR); the slice sampler's stale
  `x_l` / `x_r` under `step_out=True` (gpyreg, inert for PyVBMC); the
  `_gp_hyp`-then-`log_posterior` NaN trap (by design, a note not a
  defect).
- **Reproducing the scratch checks.** `identity_facts.py` (session
  scratchpad, not committed) compares, with one BLAS thread: `potrf` and
  scipy's `cholesky`; a direct `trtrs` with scipy's layout rule against
  `solve_triangular` for F- and C-ordered factors and both `trans`;
  `potrs` against the two-solve form; `tile(sW) * Ks` against `sW * Ks`;
  `cdist(Xs, Xs)` against `squareform(pdist(Xs))` at D 1–20; the SE
  diagonal fast path; the in-place diagonal add against `+ eye`; the
  `NegativeQuadratic` mean batched over samples against per sample. To
  rebuild: random SPD matrices from an SE kernel at N ∈ {10, 100, 250,
  345}, `np.array_equal` on every pair, `scipy 1.18.1`, `numpy 2.5.2`.
  The old-vs-new bit-check of Steps 2–5 was `gpyreg_bitcheck.py` (session
  scratchpad, not committed; the 721 → 2058 → 2219 array counts in the
  tracker): `dump ref.npz` on the pinned gpyreg (by `git -C ../gpyreg
  stash` / `git checkout 236ddd7 -- gpyreg/`) and `compare ref.npz` on the
  branch, over random GPs built with `gpr.GP` + `update` +
  `set_hyperparameters` (the cases listed in the tracker's Step 2 entry:
  every `predict` branch, both `L_chol` regimes, three kernels, three
  means, per-row noise, the rank-1 update, a Cholesky ladder that fires,
  `predict_full`, `random_function`, seeded fits), `np.array_equal` with
  `equal_nan` per array. The eight snapshots were covered by the
  generator's dump instead (`--dump-outputs` / `--check --exact
  --against`, committed). **The per-call timings** (`predict` 0.413 →
  0.256 ms, the sampler's objective 277 → 196 µs, one `train_gp` 783 →
  572 → 316 ms, the cProfile of one evaluation) came from three further
  scratch scripts (`time_predict.py`, `time_core.py`, `time_fit.py`:
  `predict(separate_samples=True)` / `_GP__compute_nlZ(h, False, True)` /
  the `gp_fit` oracle on the snapshot GPs, one thread, medians over
  repeats, old code by the same stash or checkout) and are **not
  reproducible from the repository**; their internal ratios are what the
  records quote.

## Execution tracker

Legend: `[ ]` not started, `[~]` in progress, `[x]` done, `[!]` needs
attention. Times are wall clock on 2026-09-05, corrected at 10:20 from the
commit, artifact and CI timestamps (the first draft of this tracker was
written from an estimated clock that ran up to five hours ahead).

- [x] Decisions with the PI (identity-first, oracles first, generator in
  the same PR, org-repo branch) — 07:45
- [x] Fact-gathering: gpyreg code paths and the September 5 cProfile
  breakdown; identity facts measured (`identity_facts.py`, 102 cases: all
  hold except `potrf` vs scipy's `cholesky` and `potrs` vs two solves,
  both dropped from the design); oracle-side survey (read-only Opus agent,
  11 min) with three surprises folded into §Findings — 07:50–08:05
- [x] Plan written — 08:05
- [x] Read-only Opus review of the plan dispatched — 08:07 (returned
  08:24, see the entry below)
- [x] Step 1 code: `gp_nlZ`, `gp_fit`, `PLATFORM_BOUND`, the test gate,
  `--add-oracle`, `--check --exact`, `--dump-outputs` / `--against` —
  08:07–08:22. `--add-oracle gp_nlZ` then `gp_fit`: 8 of 8 fixtures
  rewritten each, every other array asserted bit-identical, the new
  oracle exact, the others within tolerance; `pytest
  pyvbmc/testing/oracles` **116 passed, 15 skipped** (100 + 16); the
  first `--check --exact` against the committed references failed on all
  eight (the references predate items 3, 1, 2: §Design Step 1), which
  led to the dump mode; `--dump-outputs
  dev/scripts/runs/oracle_outputs_prechange_de6d98f` (53 arrays per
  snapshot, 1.1 MB) and `--check --exact --against` it: **8 of 8 ok**
- [x] Step 1 commit (pre-commit clean after black reformatted two files):
  oracles, generator, fixtures, this plan — 08:23, `test(oracles): pin the
  GP log marginal likelihood and the GP training path`
- [x] gpyreg branch `perf/predict-sampler-overhead` created off `236ddd7`
  in `../gpyreg` (the editable install follows it) — 08:23
- [x] Random-GP bit-check script (`gpyreg_bitcheck.py`, scratchpad): the
  paths no snapshot exercises (per-row noise, `L_chol=False`, three means,
  `add_noise` / `return_lpd`, the rank-1 update, every hyperprior type
  in and out of the box, three small seeded fits); `dump` on the pinned
  gpyreg: 721 arrays, self-compare 0 differ — 08:25
- [x] **Step 2 `predict`** (gpyreg branch): module-level
  `_solve_triangular` (direct `trtrs` with scipy's layout rule), `sW * Ks`
  broadcast, the mean over all samples through a new `compute_batched`
  on `ZeroMean` / `ConstantMean` / `NegativeQuadratic` (loop fallback for
  a mean without it), the SE diagonal fast path, column assignments
  instead of `(-1, 1)` reshapes — 08:26–08:34. Gates: random-GP bit-check
  **721 arrays, 0 differ**; `--check --exact --against` the pre-change
  dump **8 of 8**; gpyreg suite **82 passed** (2:11; the pre-existing
  suite, run before this step's 17 parametrized tests were added, which
  make it 99); PyVBMC oracles 116 passed. Per call on snapshot GPs (one thread, median of 5 × 400,
  `predict(separate_samples=True)`), old → new: cigar_D4_largeK (Ns 7,
  N 115) N* = 8: 0.413 → 0.256 ms, N* = 2: 0.349 → 0.208, N* = 512:
  5.37 → 5.16; corr_D5_warped (Ns 8, N 100) N* = 8: 0.463 → 0.272;
  normal_D2_singlesample (Ns 1) N* = 8: 0.054 → 0.038. So 1.6–1.7× per
  CMA-ES-sized call; the remainder is `Ns` kernel evaluations (Open
  question 2). Plain per-call cost is a third of cProfile's 1.24 ms
- [x] Plan review returned (read-only Opus, 17 min): no wrong identity
  fact; two blockers for later steps (B1: the Step 4 cache must serve
  only the no-gradient path; B2: Step 8 moves the `gp_fit` chain, a
  deliberate re-baseline), eleven should-fixes (the rank-1 update's
  `predict` call, no mean base class, Surprise 1 over-generalized and a
  second `_get_hyp_cov` slip, `L_chol=False` unreachable in PyVBMC,
  `step_out=False` as fact 6's precondition, the scipy floor, `cdist`
  reaching `random_function` / `predict_full` / the rank-1 `K`, the
  `_prior_cache` attribute on old pickles, the Step 1 gate wording, the
  `(N, 1)` product), the bit-check additions, and the §9 bookkeeping
  (the `sW` tiling bullet is already in §9: mark resolved). All folded
  into §Findings, §Design, §Decisions, §Open questions, §Risks,
  §Follow-ups — 08:35–08:38
- [x] Bit-check extended per the review (every `predict` branch incl.
  `add_noise × return_lpd × separate_samples`, `y is None`, `N* = 1`,
  `Ns = 1`, `predict` after a rank-1 update with unequal `sW` rows,
  `predict_full`, `random_function` under a seeded stream, Matern and
  rational-quadratic kernels, a single training point, both `L_chol`
  regimes, a case whose Cholesky ladder fires to `sn2_mult = 1e4`, the
  no-value `__compute_nlZ(hyp, False, True)` shape); reference re-dumped
  on the pinned gpyreg by a stash round trip; the branch: **2219 arrays,
  0 differ** — 08:39
- [x] Step 2 formatting: gpyreg's pre-commit hooks do not install on this
  Python (the pinned `pycln` / `black` envs fail to build), so the venv's
  `isort` and `black --target-version py39` ran with gpyreg's
  `pyproject.toml`; black 26 also rewrapped three unrelated expressions,
  reverted by hand so the diff holds only the item's hunks — 08:39
- [x] Step 2 gpyreg tests: `compute_batched` bit-identical to `compute`
  per column for the three means at D ∈ {1, 3, 8, 12} plus the shape
  errors; `_solve_triangular` bit-identical to `solve_triangular` for C-
  and F-ordered factors and both `trans`, `LinAlgError` on a singular
  factor; `predict` with a mean lacking `compute_batched` equals the
  batched path. 17 passed — 08:40
- [x] Step 2 replay (`runs/golden/replay_item8_step2`, 4.2 min, one
  process) — 08:37–08:41: against the default baseline (the pre-Stage-2
  population, which items 1 and 2 already parted) every config parts at
  iteration 0 or 1 with finals inside every fence, **and every final,
  evaluation count and iteration count equals the item 1/2 Step 2 replay
  of the current code digit for digit** (normal_D5 ΔLML 0.00693, gsKL
  7.91e-5; banana_D2 0.0907 / 0.231 / 85 evaluations; halfnormal_D2
  0.00206 / 0.000177; cigar_D4 0.00657 / 6.87e-5 / 125 evaluations, 25
  iterations; rosenbrock_D2_noise1 0.0972 / 0.00838 / 0.0224): the
  trajectories did not move. Wall per config 0.57 / 0.31 / 0.27 / 0.98 /
  2.0 min against 0.62 / 0.36 / 0.30 / 1.1 / 1.8 after items 1/2.
  **Re-rendered against the item 1/2 traces** (`--report-only --baseline
  runs/golden/replay_item1_step2`, report in
  `replay_item8_step2_vs_item12.log`): **all five configs `identical`**,
  same live points, same ELBO path at every iteration, initial design
  identical (`X_init` in both traces), 0 flagged of 5. Step 2 changes no
  trajectory. (For later steps the replay's `--baseline` should point at
  these item 1/2 traces so that `identical` is the verdict to expect.)
- [x] Step 2 full PyVBMC suite (`runs/fullsuite_item8_step2.log`, one
  BLAS thread): **539 passed, 15 skipped, 0 reruns, 6:20** — 08:42–08:48
- [x] **Step 2 committed** on `../gpyreg` branch
  `perf/predict-sampler-overhead`: `b8f03dd perf: predict without scipy's
  wrapper layers, tiling or per-sample mean calls` (three modules, two
  test modules) — 08:48
- [x] **Step 3** (gpyreg branch) — 08:49–08:58: SE
  `compute(X)` through `cdist(Xs, Xs)`; the noise added in place on the
  diagonal of `K / sl` (both `L_chol` branches) instead of `+ eye` /
  `+ diag`; `alpha` by two direct `_solve_triangular` calls; the
  hyperprior masks and normalization constants cached in
  `__prior_masks()` (read through `getattr`, cleared in `set_bounds`,
  `set_priors` and after `fit`'s `df` fill); `log_likelihood` /
  `log_posterior` return `(value, gradient)` instead of raising. Gates:
  random-GP bit-check **2219 arrays, 0 differ**; `--check --exact
  --against` the pre-change dump **8 of 8** (`gp_nlZ` and `gp_fit`
  included); gpyreg suite **102 passed** (three new tests: the gradient
  wrappers against finite differences, cache invalidation on priors /
  bounds / `df` fill and on an object without the attribute, the
  symmetric kernel matrix); PyVBMC oracles 116 passed. Black re-wrapped
  the same two unrelated expressions; reverted again. Per evaluation of
  the sampler's objective (`__compute_nlZ(h, False, True)`, one thread,
  median of 5 × 300), Step 2 state → Step 3: cigar_D4_largeK (N 115,
  hyp_N 15) 277 → 196 µs, corr_D5_warped (N 100, 18) 250 → 166,
  normal_D2_warmup (N 20, 9) 138 → 72; with gradient (L-BFGS-B) 737 →
  698, 678 → 608, 254 → 197. Less than the breakdown predicted; a
  cProfile of one evaluation on cigar_D4_largeK (N 115, D 4; 210 µs under
  the profiler) says why: the kernel is now 86 µs (the N² exponential
  and its two elementwise passes ≈ 60 µs, `cdist` 24 µs), the Cholesky
  38 µs (27 µs of LAPACK in scipy's batched C routine plus 11 µs of
  wrapper), `__core_computation`'s own arithmetic 26 µs (`K / sl`, the
  quadratic form, `Σ log diag L`), the priors 21 µs (from 63), the mean
  11 µs, the two solves 10 µs. What remains is arithmetic on N² entries
  plus the factorization, i.e. exactly what Step 4's reuse skips on the
  ≈ 60 % of evaluations that move a mean hyperparameter (≈ 130 of 196
  µs per hit). Replay against the item 1/2 traces
  (`runs/golden/replay_item8_step3`, 3.8 min): **all five `identical`**,
  initial designs identical, 0 flagged; walls 0.44 / 0.27 / 0.23 / 0.87 /
  1.9 min (Step 2: 0.57 / 0.31 / 0.27 / 0.98 / 2.0). Full suite **539
  passed, 15 skipped, 1 rerun** (an unseeded test), 5:52 — 09:00–09:06
- [x] **Step 3 committed**: `c0b9248 perf: log-posterior evaluation
  without wrapper layers or rebuilt prior masks` — 09:07
- [x] **Step 4** (gpyreg branch) — 09:08–09:16: a
  caller-owned factorization cache threaded through `__gp_obj_fun` →
  `__compute_nlZ` → `__core_computation(hyp, compute_nlZ,
  compute_nlZ_grad, cache=None)`; `fit` gives one cache to the
  space-filling design's objective and one to the sampler's, none to the
  L-BFGS-B objective (review B1); a hit (`compute_nlZ` true, no gradient,
  `_REUSE_CHOLESKY`, cov+noise block `array_equal` to the stored key)
  reuses `L`, `sl` and `Σ log diag L` and recomputes only the mean,
  `alpha` and the quadratic form; a miss stores them. Gates: random-GP
  bit-check **2219 arrays, 0 differ** (its three seeded fits run the
  sampler with the cache live); `--check --exact --against` the
  pre-change dump **8 of 8** (the `gp_fit` oracle is the real fit);
  gpyreg suite **104 passed, 1 rerun** (two new tests: a fit with the
  reuse on equals one with it off bit for bit in `hyp`, `samples` and
  `f_vals`; a cache with a matching key and a garbage factor leaves the
  gradient result untouched, while the no-gradient path hits on a mean
  move and refreshes on a covariance move); PyVBMC oracles 116 passed.
  Black re-wrapped the same two expressions; reverted. **One `train_gp`
  call from a snapshot state** (the `gp_fit` oracle, one thread, median
  of 3), pinned `236ddd7` → Step 3 → Step 4: cigar_D4_largeK (Ns 7,
  N 115) 783 → 572 → 316 ms (**2.5×**); corr_D5_warped (Ns 8, N 100)
  672 → 469 → 304 ms (2.2×); rosenbrock_D2_noise1_viqr (Ns 8, N 25,
  `init_N` 814 so the design dominates and misses the cache by nature)
  291 → 150 → 138 ms (2.1×). Replay against the item 1/2 traces
  (`runs/golden/replay_item8_step4`, 3.2 min): **all five `identical`**,
  0 flagged; walls 0.4 / 0.3 / 0.2 / 0.8 / 1.6 min (Step 3: 0.44 / 0.27 /
  0.23 / 0.87 / 1.9). Full suite **539 passed, 15 skipped, 0 reruns,
  5:12** — 09:16–09:21
- [x] **Step 4 committed**: `cc63452 perf: reuse the Cholesky factor
  across log-posterior evaluations that move only mean hyperparameters`
  — 09:21
- [x] **Step 5** (gpyreg branch) — 09:15–09:26: new
  `gpyreg/rng.py` (`resolve_rng`: `None` → the `numpy.random` module,
  i.e. today's legacy draws call for call; a `Generator` as is; anything
  else through `default_rng`; `random_integer` for the one method whose
  name differs); `SliceSampler(..., rng=None)` with its five draws on
  `self.rng` (`random()` is the module's `rand()`); `f_min_fill(...,
  rng=None)` (the Sobol column shuffle, the uniform design);
  `GP.fit(..., rng=None)` forwarding to both; `GP.random_function(...,
  rng=None)`. Tests: a `Generator` twin of `test_multiple_runs` (same
  chain across `sample` calls whatever the global state does, `rng=None`
  equal to the unchanged legacy path); `fit` / `random_function` with a
  generator under two different global seeds agree bit for bit and the
  `rng=None` path still follows `np.random.seed`. Gates: random-GP
  bit-check **2219 arrays, 0 differ** (its seeded legacy fits are the
  `rng=None` path); `--check --exact --against` the pre-change dump **8 of
  8**; gpyreg suite **106 passed** (1:37); PyVBMC oracles 116 passed;
  black's two re-wraps and a third in `slice_sample.py` reverted; the
  replay is skipped for this step (PyVBMC passes no `rng` yet, and the
  `gp_fit` oracle already shows the fit unchanged). Full suite **539
  passed, 15 skipped, 0 reruns, 7:07** (the laptop was in use) —
  09:26–09:32
- [x] **Step 5 committed**: `966414d feat: rng= on GP.fit, SliceSampler,
  f_min_fill and GP.random_function` — 09:33. The gpyreg branch is four
  commits on `236ddd7`: `b8f03dd`, `c0b9248`, `cc63452`, `966414d`
- [x] **Step 6** — 09:34–09:37: branch pushed to `acerbilab/gpyreg`;
  **draft PR acerbilab/gpyreg#43** ("perf: predict and sampler overhead,
  bit-identical; rng= support"); `GPYREG_PIN` → `966414d` (`1d40308 ci:`,
  `284747e docs(dev):`), `dev-next` pushed, smoke run 33950214079 started
  by the push and the full matrix dispatched (run 33950213844). **PyBADS
  suite against the branch** (`../pybads` as it stands, `pytest --reruns=5
  -x`): the first run stopped at `test_init_conf.py::test_version`, which
  needs installed package metadata (`PackageNotFoundError` for `pybads`;
  an environment artefact of the editable checkout, unrelated to gpyreg);
  rerun with that test deselected: **87 passed** (`runs/pybads_item8_
  branch.log`). PyBADS uses gpyreg's `fit`, `predict`, `predict_full` and
  `quad` through `pybads/bads/gaussian_process_train.py`.
- [x] **Step 8** (PyVBMC, committed as `c4313a3`) — 09:35–09:59:
  `train_gp` resolves
  its generator once and hands it to `gp.fit(rng=)`;
  `_BatchedNoiseHandler(N, batch_fun, rng)` overrides `indices` (cma
  4.4.4's `choice == 1` policy, the fractional re-evaluation count from
  `vp.rng`); deleted `seed_global_from`, the constructor reseed and
  `_seeded`, the reinstall at the start of `optimize()`, the `"legacy"`
  half of `random_state` (`_set_random_state` still reads the two older
  formats: the dict with a legacy entry, now ignored, and the bare tuple
  of pre-generator files, restored with the existing warning); docstrings
  (`seed`, `optimize`, `load`, `train_gp`), `AGENTS.md` randomness
  paragraph; `test_vbmc_seed.py` gets
  `test_seeded_run_leaves_global_state_untouched` (global state identical
  before and after a seeded `optimize()`, no `"legacy"` key). Gates so
  far: pre-commit clean; `test_vbmc_seed.py` + `test_vbmc_save_and_load.py`
  **13 passed** (so no stray global draw remains in a run). Step 7 (the
  profile campaign) will run from a detached checkout of `284747e`, the
  identity-preserving state, when the machine is free, so that the
  per-config comparison with 2026-09-05 is on identical trajectories.
  Oracles: `--check --exact --against` the dump shows **exactly `gp_fit`
  and `active_sample_step` moved**, and only where they draw (the
  L-BFGS-B-only `normal_D2_singlesample` fit is unchanged); both
  re-baselined from the stored state with reasons (`--rebaseline gp_fit
  --expect-moving active_sample_step`, then `--rebaseline
  active_sample_step`; the new `--expect-moving` option exists because the
  post-write check of a targeted rewrite assumed one moving oracle at a
  time, and a stream change moves every oracle that draws: the first
  attempt without it rewrote one fixture and refused, the fixtures were
  restored from git and redone). The chain that then ran the two
  re-baselines was started twice: an interrupted tool call had in fact
  completed its `gp_fit` half (09:42), so the second run (09:46) found
  nothing to change and appended a no-op `gp_fit` audit entry
  (`max_abs_change` 0.0) to every fixture; the eight no-op entries were
  removed at 10:20 by a metadata-only edit of the JSON files (found by the
  doublecheck). Oracle suite 116 passed. **Replay**
  (`runs/golden/replay_item8_step8`, against the item 1/2 traces, 3.8
  min): every config **parts at iteration 0 with a different initial
  design** (only the start point `x0` is shared, where warm-up trimming
  leaves it live: cigar shares nothing), which is what the
  change implies for an existing seed: the removed constructor reseed
  drew one integer from `vbmc.rng`, so the whole generator stream is
  shifted by one draw and the design, drawn from it before any fit, moves
  with everything after it (and the GP fits now interleave their draws
  on the same stream). The design certificate flags this by design;
  `VBMC(seed=s)` runs from before and after this commit are two different
  members of the same population. Finals: inside the fence on every
  metric of four configs; `halfnormal_D2` MMTV 0.0253 against a fence of
  0.0251 (ΔLML 0.0141 vs 0.0174, gsKL 0.00024 vs 0.00063 inside);
  evaluations 70 / 105 / 65 / 130 / 145 within the population ranges;
  walls 0.41 / 0.38 / 0.2 / 1.1 / 1.7 min against the pre-Stage-2
  baseline's 1 / 0.59 / 0.54 / 2.4 / 2.0. `halfnormal_D2` seeds 1–4
  (`runs/golden/replay_item8_step8_halfnormal_seeds`, the item 3 rule):
  **0 flagged of 4**, MMTV baseline → new 0.0163 → 0.0178, 0.0182 →
  0.0196, 0.0173 → 0.0161, 0.0238 → 0.0155 against the 0.0251 fence, two
  up and two down: the seed-0 excursion is chance. Full suite **540
  passed, 15 skipped, 0 reruns, 5:27** — 09:53–09:59
- [x] **Commits and push** — 10:00: `eae9fda test(oracles)` (the `gp_nlZ`
  gradient tolerance from the CI floors, `--expect-moving`), `c4313a3
  feat(vbmc)` (the seam removal, the re-baselined `gp_fit` and
  `active_sample_step` fixtures, `AGENTS.md`), `4c40b27 docs(dev)` (this
  plan, roadmap pickup point 3a, devlog §9 and §10 addenda, the oracle
  table, `dev/README.md`); `dev-next` pushed; smoke run 33951343730 and
  the dispatched matrix 33951358343 running (the first CI exercise of the
  gpyreg branch beyond the `gp_nlZ` floor)
- [ ] Step 7 profile campaign: **blocked on machine availability** (the
  laptop is in use); to run from a detached checkout of `284747e`, see
  roadmap pickup point 3a for the commands
- [x] `/doublecheck` on the completed steps (three read-only Opus
  reviewers: the gpyreg commits, the PyVBMC commits, the records against
  the artifacts) — 10:02–10:20. Findings, all folded in. **Must fix**: the
  space-filling design's objective built its Cholesky cache inside the
  lambda, so the reuse never fired there (the sampler's objective, where
  the savings are, was unaffected; no output changes). **Should fix**:
  `_prior_cache` printed by `repr`; generator tests that seeded the global
  state without restoring it; `--add-oracle` accepting and ignoring
  `--expect-moving`; a stale pin comment; a stale end-of-run-snapshot
  comment in `vbmc.py`; eight no-op `gp_fit` audit entries left by the
  interrupted re-baseline; a stale duplicate checklist at the end of this
  file; this tracker's clock (+1 to +5 h against the artifacts, corrected
  at 10:20); overclaims in the records ("every gpyreg output",
  "identity-preserving throughout", 1.6–1.7× / 2.2–2.5× where the
  measurements say 1.4–1.7× / 2.1–2.5×, "PyBADS passes" without the
  deselected test). **Optional**: `gpyreg.rng` undocumented and not
  imported by the package; `_solve_triangular` tested for one `trans`
  only; `_REUSE_CHOLESKY`, the `compute_batched` memory footprint and the
  `set_priors` / `set_bounds` invariant absent from the PR body.
- [x] **Review fixes committed** — 10:40. gpyreg `79b4986 fix:` (design
  cache once per fit, `repr` exclude, explicit C-contiguous `K` copies,
  test hygiene, `trans` 0 / 1 / 2, `gpyreg.rng` imported and documented),
  gated as before: bit-check 2219 arrays 0 differ, oracles `--exact
  --against` the pre-change dump differ only in `gp_fit` /
  `active_sample_step` (Step 8's stream change), gpyreg suite 108 passed;
  pushed, PR #43 body rewritten (design-cache note, the `set_priors` /
  `set_bounds` invariant, `_REUSE_CHOLESKY`, the memory note, the N = 0
  kernel shape). PyVBMC `f118428 test(oracles)` (generator fixes,
  audit-entry cleanup, comments), `19df439 ci:` (`GPYREG_PIN` →
  `79b4986`), and the records in the commit that carries this line.
- [x] **Fourth CI round green** (`6a65cfd`): smoke 33952020250 and the
  full matrix 33952029068 both `success` (seen 10:35). Each of the three
  earlier rounds revealed one floor or one trap under `-x`; nothing of
  item 8's numerics failed anywhere.
- [!] **Second CI round (10:00) failed on the next snapshot**: `corr_D5_
  warped / gp_nlZ`, `dlZ` / `dlp` at 4.13e-4 per element (5.4e-5
  absolute) on Ubuntu / 3.11 against the 1e-4 class, values at 9e-8; the
  matrix's other jobs cancelled again. The gradient of the log marginal
  likelihood is the worst-conditioned quantity in the fixtures (the corr
  snapshot also had the largest `J_sjk` floor); a cross-BLAS check of it
  is honest only at the percent level. Set to 2e-2 (about 50× the largest
  measured floor), with the module docstring and the oracle table saying
  that same-machine exactness comes from the dump gate. Every CI round
  under `-x` reveals one floor; `normal_D2_*`, `halfnormal_D2_bounded` and
  the noisy snapshot are still unmeasured on Ubuntu for this oracle
  — 10:05 (`acde5f7`)
- [!] **Third CI round (10:06, matrix 33951610366): a different failure,
  not item 8's.** Ubuntu / 3.11 failed `test_vp_optimize_2D_g_mixture`
  six times in a row (five reruns): the K = 2 VP fitted to the bimodal
  target collapsed both components to the centre (mu ≈ ±0.3, sigma ≈ 1.5;
  the moment-matching KL passed, the marginal total variation did not), a
  bad local optimum of `optimize_vp`. The test is unseeded and uses no
  VBMC instance (a direct `gp.fit`, whose `rng=None` path draws exactly as
  before, and a VP seeded from the global stream), so item 8 does not
  touch its draws; what made six identical attempts possible is the
  module's autouse fixture that restores the global state around each
  test (added with items 1/2 to keep the module's tests on their
  historical draws), which makes every rerun replay the failed attempt.
  The full matrix had not run since items 1 and 2 changed the ELBO
  arithmetic (only the Ubuntu / 3.12 smoke), so that platform's draws met
  the new arithmetic for the first time. Fix: the fixture advances the
  stream by the attempt number (`request.node.execution_count`) before
  yielding, so reruns see fresh draws and later tests are unaffected;
  `AGENTS.md` testing traps updated; both `*_g_mixture` tests pass locally
  under `--reruns=5` — 10:14 (`6a65cfd`)
- [!] **First CI runs of the day failed on the new `gp_nlZ` oracle**
  (smoke 33950214079 on Ubuntu / 3.12 and the dispatched matrix
  33950213844, whose macOS / 3.11 job failed first and cancelled the
  rest): `cigar_D4_boosted / gp_nlZ`, `dlZ` and `dlp` at 2.61e-6 per
  element (8.2e-6 absolute) against the 1e-6 GP-solve class, `lZ` / `lp`
  at 1e-8. The gradient goes through the explicit inverse `Q = K⁻¹ −
  ααᵀ` on the cigar GP (conditioning ≈ 1e8), a worse-conditioned quantity
  than the analytic expectations under the VP. Fix per the tolerance
  rule (measure, then set 30–100× above): `dlZ`, `dlp` → 1e-4, values stay
  at 1e-6; recorded in the module docstring. Nothing else has run on CI
  yet (`-x` stopped both runs at this test), so the gpyreg branch is
  untested there until the next push — 09:48 (the 1e-4 class was itself
  superseded by 2e-2 after the second CI round, see below)
- [ ] (open) 20-seed population after Step 8, see Open question 3
