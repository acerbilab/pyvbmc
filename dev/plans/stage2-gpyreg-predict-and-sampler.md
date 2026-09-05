# Stage 2 item 8: gpyreg `predict` and slice-sampler overhead, one gpyreg PR

Created: 2026-09-05 08:20. Status: **IN PROGRESS** (this file is the plan
and the worklog; tracker at the end). Roadmap pickup point 3
(`plans/modernization-roadmap.md`: "Next: item 8 as one gpyreg PR");
rationale in `dev/2026-09-02-modernization-discussion.md` §2, §4, §9, §10;
the inherited gpyreg half of item 3 in `plans/stage2-batched-acquisition.md`
§Follow-ups; what is left after items 3, 1 and 2 in
`plans/stage2-gp-log-joint-einsum.md` §Results (active sampling 46–54 % of a
D = 4 run with `GP.predict` 31–34 %, GP training 31–42 % with the slice
sampler 27–38 %). Decisions taken with the PI on 2026-09-05 (08:00):
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
  93.6k log-posterior evaluations (`__gp_obj_fun(hyp, False, True)`
  through `__compute_nlZ` and `__core_computation`, :1540, :1520, :2357)
  at 287 µs each: `squareform(pdist(X / ell))` ≈ 60 µs (`pdist` 46 µs of
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
     evaluation (`slice_sample.py:393-457`). For `SquaredExponential` +
     `GaussianNoise` + `NegativeQuadratic`, `hyp` is `[ln ell (D), ln sf,
     ln sn (1 or 2), m0, xm (D), ln omega (D)]`; when the changed
     coordinate is in the mean block (2D+1 of 3D+3: 9 of 15 at D = 4, 31
     of 48 at D = 15), `K`, `sn2`, `sn2_mult`, `sl`, `L`, `L_chol` and
     `Σ log diag L` are unchanged and only `m`, `alpha` and the quadratic
     form move. Same `potrf` input gives the same `L`. Expected hit rate
     ≈ 55–60 % of the sampler's evaluations, each hit skipping the kernel
     (60 µs), the Cholesky (26 µs) and the ladder.
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
- **PyVBMC's `predict` call sites** (grep, non-test): the hot path is
  `abstract_acq_fcn.py:79` (`separate_samples=True`);
  `active_importance_sampling.py:83, 113, 230, 257, 354` (separate
  samples) and `:465` (averaged); `acq_fcn_viqr.py:246` and
  `acq_fcn_imiqr.py:264` pointwise with `add_noise=True` inside the noisy
  proposal density; `vbmc.py:1265` and `:1405` (`add_noise=True`). The
  `add_noise`, `return_lpd` and averaging branches must keep their
  semantics; `predict_full` has no PyVBMC caller.
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
    always discarded.** `_get_hyp_cov`'s weighted branch (:712-718) takes
    `gp_hyp_full[i]`, shape `(Ns, hyp_N)`, uses `shape[1]` as the sample
    count and appends `hyp.T`, so `hyp_cov` is `(Ns, Ns)`; `train_gp:121-
    124` then drops `gp_train["widths"]` because its size is not `hyp_N`.
    Verified on all eight fixtures (`Ns ≠ hyp_N` everywhere), so every
    production fit slice-samples with gpyreg's `widths_default`
    (`np.std` of the space-filling design, or `PUB − PLB` when `init_N =
    0`). Consequence for the oracle: a synthetic `sKL`/`gp_hyp_full` cannot
    change the result, only the shape matters. Forcing `weighted_hyp_cov
    = False` would *keep* the widths (`run_cov` is `(hyp_N, hyp_N)`) and
    leave production's code path. Fixing the bug changes the sampler's
    proposal widths and therefore every trajectory: out of item 8's scope,
    recorded for devlog §9 with its own population check when fixed.
  - **Surprise 2 (gpyreg, not in devlog §9)**: `GP.log_likelihood(hyp,
    compute_grad=True)` and `GP.log_posterior(..., True)` raise `TypeError`
    (unary minus on the `(nlZ, dnlZ)` tuple, `gaussian_process.py:1488`,
    `:1518`). Reproduced on all eight fixtures. The oracle calls the
    mangled `gp._GP__compute_nlZ(hyp, compute_grad, compute_prior)` (what
    `__gp_obj_fun` calls) until the gpyreg PR fixes the two lines.
  - **Surprise 3**: `build_gp` never sets priors or bounds (`GP.__init__`
    leaves `no_prior = True`), so on a rebuilt GP `log_posterior ==
    log_likelihood` exactly; installing PyVBMC's hyperprior with `_gp_hyp(
    optim_state, options, plb_tran, pub_tran, gp, X, y)` (:272, priors and
    bounds set at :481-482) leaves NaN bounds that make the normalization
    constants NaN until `fit`'s two repair lines run (`df` NaNs → 7 at
    `gaussian_process.py:1029`; `set_bounds(get_recommended_bounds(lb,
    ub))` at :1038-1046). The oracle replicates the two lines; with them
    `lp` is finite on every fixture (cigar `lZ = 426.2503917972849`, `lp =
    420.54313794248304`).
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

**`gp_nlZ`** (GP-solve class, `rtol 1e-6`, `atol 1e-10`; applies always).
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
  fill in `fit` (fact 7). `__compute_log_priors` reads the cache; every
  arithmetic statement on `hyp` stays as written.

Gates: bit-check of `__core_computation` outputs (`Posterior` fields,
`nlZ`, `dnlZ`, `log_posterior`) old vs new on the snapshots' stored
hyperparameter samples and on random hyperparameters, all `==`; the
`gp_nlZ` and `gp_fit` oracles exact; the rest as Step 2. Commit.

### Step 4: Cholesky reuse in the sampler

`__core_computation(hyp, compute_nlZ, compute_nlZ_grad, cache=None)`.
When `cache` is a dict and `np.array_equal(cache["key"],
hyp[: cov_N + noise_N])`, reuse `sn2`, `L`, `sl`, `sn2_mult`, `L_chol`,
`pL`, `logdet` from it and compute only `m`, `alpha`, `nlZ`; otherwise
compute as in Step 3 and store. `fit` owns one cache dict per objective
closure (`objective_f_1` for `f_min_fill`, `objective_f_2` for L-BFGS-B,
`sample_f` for the sampler); `update` and `set_hyperparameters` pass no
cache and are unaffected. Hits are exact by construction (fact 6).

Gates: as Step 3, plus a gpyreg unit test that a fit with the cache
reproduces a fit without it bit for bit under the same seed (run the
sampler twice from one seed with the cache monkey-patched off). Commit.

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
touches `gpyreg/`, so the draft PR of Step 6 gets gpyreg's own CI as well. Every stream changes, so
the replay parts at iteration 0 by design and the design certificate
carries the check; the population run is the statistical gate (Open
question 3).

## Verification

- [ ] Step 1: oracles green; `--check` exact; fixture diffs show only new
      `ref/` keys and `meta`.
- [ ] Steps 2–4: bit-check `==` on every output; oracles exact; gpyreg
      suite green; replay `identical` on all five configs after each
      step; PyVBMC full suite green after each commit.
- [ ] Step 5: gpyreg suite green with the new tests; PyVBMC gates
      unchanged.
- [ ] Step 6: PyVBMC smoke and full matrix green against the branch pin.
- [ ] Step 7: per-config walls and shares recorded; trajectories identical
      to 2026-09-05 (same iterations, evaluations, metrics); probe within
      a few percent.
- [ ] Step 8: `test_vbmc_seed.py` green; seeded runs reproducible without
      any global-state write; replay finals inside the envelope; design
      certified.

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
  `compute_batched(H, X)` on the mean-function classes** (measured
  identical, fact 8): `NegativeQuadratic` and `ConstantMean` / `ZeroMean`
  get one-line implementations, the base class a loop over rows of `H`.
  Backward compatible; PyBADS unaffected.
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
4. Clone `acerbilab/pybads` and run its suite against the branch before
   the merge? **Yes, once**, if its tests run without extra setup.
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
- Machine in use during the day: gates are single short processes; the
  profile campaign and the population wait.

## Follow-ups

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
  The old-vs-new bit-check of `predict` and `__core_computation` (Steps
  2–4) will be a second scratch script over the eight oracle snapshots
  (`pyvbmc.testing.oracles._state.build_state`) and random GPs, comparing
  the pinned gpyreg (`git -C ../gpyreg stash` / branch switch) with the
  working tree; its numbers go in the tracker.

## Execution tracker

Legend: `[ ]` not started, `[~]` in progress, `[x]` done, `[!]` needs
attention. Times are wall clock on 2026-09-05.

- [x] Decisions with the PI (identity-first, oracles first, generator in
  the same PR, org-repo branch) — 08:00
- [x] Fact-gathering: gpyreg code paths and the September 5 cProfile
  breakdown; identity facts measured (`identity_facts.py`, 102 cases: all
  hold except `potrf` vs scipy's `cholesky` and `potrs` vs two solves,
  both dropped from the design); oracle-side survey (read-only Opus agent,
  11 min) with three surprises folded into §Findings — 08:20–09:15
- [x] Plan written — 09:20
- [~] Read-only Opus review of the plan dispatched — 09:25
- [x] Step 1 code: `gp_nlZ`, `gp_fit`, `PLATFORM_BOUND`, the test gate,
  `--add-oracle`, `--check --exact`, `--dump-outputs` / `--against` —
  09:30–10:05. `--add-oracle gp_nlZ` then `gp_fit`: 8 of 8 fixtures
  rewritten each, every other array asserted bit-identical, the new
  oracle exact, the others within tolerance; `pytest
  pyvbmc/testing/oracles` **116 passed, 15 skipped** (100 + 16); the
  first `--check --exact` against the committed references failed on all
  eight (the references predate items 3, 1, 2: §Design Step 1), which
  led to the dump mode; `--dump-outputs
  dev/scripts/runs/oracle_outputs_prechange_de6d98f` (53 arrays per
  snapshot, 1.1 MB) and `--check --exact --against` it: **8 of 8 ok**
- [ ] Step 1 commit (after pre-commit): oracles, generator, fixtures, plan
- [ ] Step 2: gpyreg branch; `predict`; bit-check; gates; commit
- [ ] Step 3: `__core_computation`, SE `cdist`, prior-mask cache;
  bit-check; gates; commit
- [ ] Step 4: Cholesky reuse; gates; commit
- [ ] Step 5: generator support; gpyreg tests; commit
- [ ] Step 6: push, draft PR, `GPYREG_PIN` → branch head, smoke + matrix
- [ ] Step 7: profile campaign (idle machine) + exhaust cProfile; write-up
- [ ] Step 8: PyVBMC seam removal after the pin bump; tests; records
- [ ] Records: roadmap, devlog §2/§9/§10, `dev/README.md`, `AGENTS.md`;
  commits; push
- [ ] `/doublecheck`
- [ ] (open) 20-seed population after Step 8
