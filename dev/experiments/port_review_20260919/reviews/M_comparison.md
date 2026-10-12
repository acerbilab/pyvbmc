# Slice M — MATLAB changes since the port began (2021-01-19 to `396d649`), comparison track

Raw reviewer report, wave 1 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-19. One Opus agent,
fresh context, read-only on the repositories; its check scripts ran in the
session scratchpad and are not retained. The text below is the agent's
final message, unedited.

---

## 1. Coverage

68 commits in `../vbmc` since 2021-01-19 (`git log --format='%h %ad %s' --date=short --since=2021-01-19`). I ran `git show --stat` on all 68 and `git show` on every one that touches a `.m` file outside `README`/`LICENSE`/`test/`/`vbmc_examples.m`.

| # | Commit | Date | Subject (abbrev.) | Disposition |
|---|---|---|---|---|
| 1 | `43fdcf7` | 2021-01-21 | updated ibslike | algorithmic, but in `utils/ibslike.m` — the IBS estimator, an excluded utility with no Python counterpart (counterpart map, "Excluded files"). N/A |
| 2 | `9cdc71d` | 2021-01-21 | merge (README) | not algorithmic |
| 3 | `d30fc9c` | 2021-01-23 | README | not |
| 4 | **`a5240d2`** | 2021-02-02 | warp robustness, version command | **algorithmic** |
| 5 | `a606ba3` | 2021-02-02 | merge (README) | not |
| 6 | `d1a18fe` | 2021-02-02 | README | not |
| 7 | **`f9c04bc`** | 2021-02-02 | rotoscaling fix | **algorithmic** |
| 8 | `7471d84` | 2021-02-02 | error description in `funlogger_vbmc.m` | algorithmic (diagnostics only) |
| 9 | `9a091e9` | 2021-02-02 | correlated-normal tests | test harness + `vbmc('test',opts)` dispatch; not algorithmic |
| 10 | **`f3f8b80`** | 2021-03-13 | options; D=1 fix | **algorithmic** |
| 11–13 | `af96373`,`794bfac`,`15e2715` | 2021-03 | README | not |
| 14 | **`48c82e0`** | 2021-03-27 | SAMPLES struct; `vbmc_mode` start points | **algorithmic** |
| 15 | `11737f5` | 2021-03-27 | same subject, but the diff is only `vbmc_version = '1.0.4'` | not |
| 16 | `0c3071e` | 2021-03-27 | README | not |
| 17 | `1d57b49` | 2021-05-26 | removed `pause` in a catch | algorithmic (debug-only control flow) |
| 18 | **`46b6f5e`** | 2021-05-26 | `Xs = []` for `N <= N0` | **algorithmic** |
| 19 | **`fcf2674`** | 2021-06-07 | slice-sampler fix | **algorithmic** |
| 20 | `0917171` | 2021-06-07 | version string | not |
| 21 | `7b06510` | 2021-06-07 | README | not |
| 22 | **`2044530`** | 2021-06-18 | cleanup: removed unused/experimental features | **algorithmic** |
| 23–24 | `0d59e93`,`a9b7bb4` | 2021-06-18 | README | not |
| 25 | **`1b72896`** | 2021-06-23 | MC-entropy weight-gradient fix | **algorithmic** |
| 26–27 | `12a50f8`,`b7308aa` | 2021-06-23 | README | not |
| 28–32 | `09bf182`,`52bbba4`,`5a736ee`,`eccf99a`,`843e141` | 2021-07-12 | README | not |
| 33 | **`2cb857d`** | 2021-07-13 | `sKL_true` per iteration | **algorithmic** |
| 34–37 | `71a54e4`,`ce33fa9`,`1a4ec88`,`ec551f8` | 2021-08…10 | README | not |
| 38 | **`1d1f20d`** | 2022-01-30 | `uuinv` fix in `gplite/private/fminfill.m` | **algorithmic** |
| 39 | `bd38b48` | 2022-01-30 | `utils/fminfill.m` updated; `acq/acqbald_vbmc.m` re-added | algorithmic, but in a dead MATLAB copy (see §2) |
| 40 | `ee7ad00` | 2022-01-30 | deleted `acqbald_vbmc.m` again | not (net zero with `bd38b48`; the file was absent at 2021-01-23) |
| 41 | `c01ba45` | 2022-01-30 | version doc | not |
| 42 | `fe67ab8` | 2022-01-30 | README | not |
| 43 | **`68a197b`** | 2022-06-25 | GP log-density fix; slice sampling for active sampling | **algorithmic** |
| 44 | **`74b046e`** | 2022-06-25 | fixed alternative bounded transforms | **algorithmic** |
| 45 | `08203af` | 2022-06-25 | README | not |
| 46 | **`f3e5d76`** | 2022-07-06 | `probit` name for `norminv` | **algorithmic** |
| 47 | **`a9615ba`** | 2022-07-23 | VIQR speed | **algorithmic** |
| 48 | **`c387612`** | 2022-10-26 | new priors (v1.0.11) | **algorithmic** |
| 49 | `84b29cc` | 2022-10-26 | examples table of contents | not |
| 50 | **`f3c05d6`** | 2022-10-26 | log pdfs (v1.0.12) | **algorithmic** |
| 51–52 | `c892b3b`,`418ce22` | 2022-10-27 | README | not |
| 53 | `6bd4f2f` | 2022-10-30 | `vbmc.m`: repo URLs + one mojibake character | not |
| 54–60 | `5436ce2`,`96ca388`,`3aa280c`,`0ec31c3`,`a52a722`,`a903ec2`,`a0e38be` | 2022-10-30 | README / examples / test runner | not |
| 61 | `b51c005` | 2022-10-30 | `ibslike.m` doc/URL/encoding | not |
| 62–63 | `50eec70`,`f446b63` | 2022-10-30 | LICENSE | not |
| 64–66 | `1aaea63`,`5d96eb9`,`54ba2cd` | 2022-10-31 | README | not |
| 67 | `396d649` | 2023-05-03 | README | not |
| 68 | `9a091e9` counted above | | | |

**Read fully:** every algorithmic diff above; `misc/warp_input_vbmc.m`, `shared/warpvars_vbmc.m` (create-transform block), `private/activesample_vbmc.m` (search-optimizer switch, rank-1 update block), `acq/acqviqr_vbmc.m`, `private/activeimportancesampling_vbmc.m` (only-VP branch and the `Ctmp_mat` precompute), `ent/entmc_vbmc.m`, the eight MATLAB prior pdf/logpdf/rnd files; on the Python side `pyvbmc/whitening/whitening.py`, `pyvbmc/parameter_transformer/parameter_transformer.py`, `pyvbmc/entropy/entmc_vbmc.py`, `pyvbmc/priors/{trapezoidal,spline_trapezoidal,uniform_box,smooth_box}.py`, `pyvbmc/acquisition_functions/acq_fcn_viqr.py`, `gpyreg/f_min_fill.py`, and the relevant regions of `pyvbmc/vbmc/vbmc.py`, `active_sample.py`, `active_importance_sampling.py`, `variational_optimization.py`, `gaussian_process_train.py`, `gpyreg/gaussian_process.py`, `gpyreg/slice_sample.py`.

**Skimmed:** the README-only and example-only diffs (stat + subject only, plus a spot check of `6bd4f2f` and `b51c005` in full).

**Not reached:** nothing in the commit range.

**Checks run** (all with `.venv/Scripts/python.exe`, scripts in my scratchpad):
1. `chk_transform.py` — `ParameterTransformer` for `logit`/`probit`/`norminv`/`student4` on a 2-D bounded box: every type maps `plb_orig → −0.5`, `pub_orig → +0.5`, with type-specific `mu`/`delta`. Settles `74b046e`/`f3e5d76`.
2. `chk_entmc_w.py` — central finite differences of `entmc_vbmc`'s entropy w.r.t. `eta` at `D=2, K=3, Ns=4e5`, common random numbers: analytic `[0.0947149, 0.07205114, −0.16676605]`, numeric identical to `2.5e-11`. Settles `1b72896`.
3. `chk_uuinv.py` / `chk_uuinv2.py` — `gpyreg.f_min_fill.uuinv` inverted against the mixture CDF on 9999 quantiles for five bound/`w` configurations including `w=0`, `w=1` and the `L=0` degenerate case: `max|F(uuinv(p)) − p| ≤ 1.1e-16`, monotone, range exactly `[LB, UB]`. Settles `1d1f20d`.
4. `chk_minimize_x0.py` — `scipy.optimize.minimize` with `x0` outside `bounds` clips silently (scipy 1.18.1); used only to bound the consequence of the `vbmc_mode` note below.

No test file or suite was run; no `optimize()` run.

---

## 2. Per algorithmic commit

### `a5240d2` (2021-02-02) — warp robustness — **reflected**
Three changes: (a) `warp_input_vbmc.m` recomputes the plausible range from `Nrnd = 1e5` uniform draws in the original plausible box, takes the 5%/95% quantiles and widens by `delta/9`; (b) `warpvars_vbmc.m` moves the centering to "at the end of the transform" — `plb`/`pub` are mapped through the transform *first*, then `mu`/`delta` are set from the transformed values; (c) `vbmc.m` adds `WarpTolSDMultiplier = 2`, `WarpTolSDBase = 1` and an `elbo_sd` test for undoing a warp.

Python: (a) `pyvbmc/whitening/whitening.py:170-187` (`Nrnd = 100000`, `np.quantile(yy, [0.05, 0.95])`, `± delta_temp/9`); (b) `pyvbmc/parameter_transformer/parameter_transformer.py:144-160`, verified numerically by check 1; (c) `pyvbmc/vbmc/vbmc.py:1353-1363` with `warp_tol_sd_multiplier`/`warp_tol_sd_base` at `advanced_vbmc_options.ini:333,335`, values 2 and 1.

### `f9c04bc` (2021-02-02) — rotoscaling fix — **reflected**
MATLAB adds `vp_Sigma = diag(trinfo.delta)*vp_Sigma*diag(trinfo.delta)` before the whitening SVD, and replaces the `PLB/PUB = ∓0.5` normalization with `optimState.PLB = plb; optimState.PUB = pub` (the `mu`/`delta` rescaling is commented out).

Python: `whitening.py:140` (`vp_cov = np.diag(delta) @ vp_cov @ np.diag(delta)`, using the pre-reset `delta` as MATLAB does), and `whitening.py:186-187` assigns the quantile bounds directly to `plb_tran`/`pub_tran`, with `mu`/`delta` left at 0/1 (`:170-171`). The `warp_roto_corr_thresh` masking, `warp_cov_reg` diagonal regularization, sign-fixed SVD and `sqrt(s + eps)` scaling all line up with `warp_input_vbmc.m:52-72`.

### `7471d84` (2021-02-02) — error description for cluster runs — **not applicable (diagnostics)**
MATLAB prints the caught error's identifier, message, file and line before rethrowing. `pyvbmc/function_logger/function_logger.py:302-309` appends a `FunctionLogger:FuncError` note to `err.args` and bare-`raise`s, so Python's traceback already carries file/line/message. No behavioral gap.

### `f3f8b80` (2021-03-13) — options and D=1 — **reflected**
The algorithmic part is `vbmc.m:544`: `&& vp.D > 1` added to the warping condition. Python: `pyvbmc/vbmc/vbmc.py:1255` (`and (self.vp.D > 1)`). The `setupoptions_vbmc.m` part (a `skipextra_flag` argument and `isfield` guards so the options struct can be built partially) is MATLAB options plumbing with no Python analogue (sheet, P1b "Options are layered `.ini` files evaluated in Python").

### `48c82e0` (2021-03-27) — SAMPLES struct; `vbmc_mode` start points — **partly reflected**
- `vbmc_mode.m:41` `x0 = min(max(x0,LB),UB)` **is** reflected at `pyvbmc/variational_posterior/variational_posterior.py:1282-1286`. One cosmetic deviation: MATLAB clamps to the same `LB = lb_orig + sqrt(eps)` / `UB = ub_orig − sqrt(eps)` it hands `fmincon`, while Python clamps to the raw `lb_orig`/`ub_orig` and passes the `± sqrt(eps)` box as `bounds`. Check 4 shows SciPy clips such an `x0` silently, so the consequence is nil; I am not raising it as a finding.
- The `samples` output struct (`X`, `y`, `y_sd`, `active_flag`, `nevals`, temperature-scaled) has no counterpart in `_create_result_dict` (`pyvbmc/vbmc/vbmc.py:3097-3158`). Finding **F4**.

### `1d57b49` (2021-05-26) — removed `pause` after an evaluation error — **not applicable**
MATLAB commented out a debugging `pause` inside a `catch`. PyVBMC has no such block; `active_sample.py` lets the `FunctionLogger` exception propagate.

### `46b6f5e` (2021-05-26) — `Xs = []` for `N <= N0` — **reflected**
MATLAB adds an `else Xs = []` so `X = [x0; Xs]` does not use a stale `Xs`. `gpyreg/f_min_fill.py:90` initializes `sX = None` before the `if N > N0` block and `:179-182` concatenates only when `sX is not None`. Same effect.
(The counterpart map attributes this commit to `misc/initdesign_vbmc.m`; the commit touches only the two `fminfill.m` copies — see Sheet notes.)

### `fcf2674` (2021-06-07) — slice-sampler fix — **reflected**
MATLAB deleted the code that *shifted* the interval `[x_l, x_r]` back inside `[LB_out, UB_out]` and kept only the clamp. `gpyreg/slice_sample.py:451-453` does only `np.fmax(x_l[dd], LB_out[dd])` / `np.fmin(x_r[dd], UB_out[dd])`; there is no shifting code. The `isfinite(LB)||isfinite(UB)` guard is omitted, which is a no-op because `LB_out` is `−inf` for unbounded coordinates.

### `2044530` (2021-06-18) — cleanup — **reflected in behavior; leftover options are a finding**
What the commit removed and where Python stands:
- `intmeanfun` blocks in `gplogjoint.m` and `acqimiqr_vbmc.m` — N/A, unported (sheet, G1/G2 "Integrated mean function (`intmeanfun`) never ported").
- `gplogjoint_multi`, `entmcub_vbmc`, `entub_vbmc`, `entropy_alpha` blending in `negelcbo_vbmc.m` — Python has the post-cleanup form: `pyvbmc/vbmc/variational_optimization.py:1259-1264` uses `entmc_vbmc` when `Ns > 0`, else `entlb_vbmc`. `entub_vbmc` unported (sheet, P7).
- `DoubleGP` recursion in `gptrain_vbmc.m` — absent from Python.
- `EmpiricalGPPrior` branch — Python has exactly the post-cleanup path, including the replacement comment, at `pyvbmc/vbmc/gaussian_process_train.py:445-457` (`log(gp_length_prior_mean * (pub_tran − plb_tran))`, `gp_length_prior_std`), and not the removed `numel(GPLengthPriorMean)==2` random-draw branch.
- `IntegrateGPMean` / `check_quadcoefficients_vbmc` in `gptrain_vbmc.m`, and `&& ~options.IntegrateGPMean` in the rank-1 update condition — Python's condition at `active_sample.py:730-733` carries no `integrate_gp_mean` term, matching post-cleanup MATLAB. (Its `and`/`||` difference from MATLAB is Finding **F3**.)
- `GPStochasticStepsize` stepsize block in `vpoptimize_vbmc.m` — Python has the fixed master stepsize only: `variational_optimization.py:261-273`.
- `output.version` added to `vbmc_output.m` — Python: `_create_result_dict` sets `output["version"]`.

**The leftover:** `double_gp`, `empirical_gp_prior`, `integrate_gp_mean`, `gp_stochastic_step_size` — and, from the same commit's `evalfields` edit, `annealed_gp_mean` and `constrained_gp_mean` — survive in `advanced_vbmc_options.ini` and are read nowhere in `pyvbmc/`. Finding **F2** (this answers the checklist question: they are **dead**, not half-ported — no branch anywhere consumes them).

Also worth recording as a MATLAB-side defect with no Python consequence: this commit introduced a missing comma in `misc/setupoptions_vbmc.m:47` (`'ConstrainedGPMean''FeatureTest'`), which concatenates the two names. All three of `AnnealedGPMean`, `ConstrainedGPMean`, `FeatureTest` are now referenced nowhere else in MATLAB.

### `1b72896` (2021-06-23) — MC-entropy weight gradient — **reflected (verified numerically)**
Fixed MATLAB: `w_grad(:) -= w(j)*squeeze(sum(norm_jl(1,1,:,:)./q_j,3))/Ns` — the full `K`-vector of unweighted component densities, not just column `j` broadcast to every entry. Python `pyvbmc/entropy/entmc_vbmc.py:154-163`: `w_acc += einsum("j,jnk,jn->k", w[jj], E, 1/q)` then `w_grad = −(sum_logq + nf * w_acc)/Ns`, with `nf[k] = nconst/sigma_k**D`, which expands term-for-term to the fixed expression. Check 2 confirms the gradient matches finite differences to 2.5e-11.

### `2cb857d` (2021-07-13) — `sKL_true` per iteration — **reflected**
MATLAB changed `stats.sKL_true = sKL_true` (overwrite) to `stats.sKL_true(iter) = sKL_true`. Python records it per iteration: declared in the `IterationHistory` key list at `pyvbmc/vbmc/vbmc.py:479`, computed at `:1571` (`_compute_true_diagnostic`, returning `None` when `true_mean`/`true_cov` are absent) and stored at `:1605`.

### `1d1f20d` (2022-01-30) — `uuinv` fix — **reflected**
`gpyreg/f_min_fill.py:192-255` is the fixed MATLAB function line for line, including `L = B[3] − B[0] + B[1] − B[2]`, the `w == 1` shortcut, the `L == 0` degenerate branch and the three-step inversion. Check 3 verifies it is an exact, monotone inverse CDF over `[LB, UB]`. gpyreg's reference copy `matlab/gplite/private/fminfill.m` also carries the fix (diff against MATLAB master is trailing whitespace only).

### `bd38b48` (2022-01-30) — `utils/fminfill.m` — **not applicable, confirmed**
`utils/fminfill.m` is a dead copy: `grep -rn fminfill --include=*.m` over the MATLAB repo shows the only two call sites are in `gplite/gplite_train.m`, which resolve to `gplite/private/fminfill.m`. The commit brought the dead copy up to the private one's state and added a `Method` option (`'sobol'` default) — `gpyreg.f_min_fill` has the same `design` parameter defaulting to `"sobol"` (`f_min_fill.py:66-67`). The `acq/acqbald_vbmc.m` file this commit added was absent before 2021-01-23 and was deleted again by `ee7ad00`; net zero.

### `68a197b` (2022-06-25) — GP log-density; slice sampling — **partly reflected**
- **GP log-density: reflected in gpyreg's Python.** MATLAB changed `lp` from the noise variance alone to the total predictive variance `ys2`. `gpyreg/gaussian_process.py` computes a log predictive density in `GP.predict(..., return_lpd=True)`: `:2035-2040` for `separate_samples=True` and `:2056-2064` for the averaged case, in both cases dividing by `y_s2 = s2 + sn2_star*sn2_mult` — i.e. the fixed formula. gpyreg's own reference copy `matlab/gplite/gplite_pred.m:108` still has the *pre-fix* line, so the counterpart map's remark that "gpyreg's reference copy predates it" is right about the copy but the Python code is current. The averaged-sample branch (which adds the between-sample variance `v` to `y_s2` before forming `lpd`) is a Python-side extension; MATLAB only produces per-sample `lp`.
- **Nothing in PyVBMC consumes it.** `return_lpd` appears nowhere under `pyvbmc/`. The lpd is gpyreg-API-only.
- **Slice sampling for the acquisition search: not ported.** MATLAB added `case 'slicesample'` to `activesample_vbmc.m:265`'s switch (and left an unsuppressed `[x0;xsearch_optim]` display in it, marking it as experimental). PyVBMC supports `"cmaes"`, `"Nelder-Mead"` and `"none"`, and `active_sample.py:552-553` raises `NotImplementedError` for anything else. Default is `"cmaes"` on both sides, so no default behavior differs. The sheet does not list this unported branch (Sheet notes).

### `74b046e` (2022-06-25) — alternative bounded transforms — **reflected**
MATLAB's defect: the transform was built with the default logit type and the type was flipped to 12/13 *afterwards*, so `mu`/`delta` were computed in the logit space for `norminv`/`student4`. The fix passes `bounded_type` into the constructor. PyVBMC's `ParameterTransformer.__init__` sets `self.type[i] = bounded_type` (`:135-142`) *before* computing `mu`/`delta` from the transformed plausible bounds (`:144-160`). Check 1 confirms each bounded type produces its own `mu`/`delta` mapping `plb/pub → ∓0.5`.

### `f3e5d76` (2022-07-06) — `probit` name — **reflected**
`parameter_transformer.py:110-115` maps both `"norminv"` and `"probit"` to type 12. (A related default difference is in Sheet notes: MATLAB's `BoundedTransform` default is `logit`, PyVBMC's `bounded_transform` is `"probit"`.)

### `a9615ba` (2022-07-23) — VIQR speed — **reflected; no default or formula changes**
Answering the question directly: **pure speed, plus one option-handling change, and no formula or default moves.**
- `Ctmp_mat` caching moves `L\(L'\Kax')/sn2_eff` (or `L*Kax'`) out of the acquisition into the importance-sampling precompute. Python: `active_importance_sampling.py:284-324` builds `active_is["C_tmp"]` exactly this way, and `acq_fcn_viqr.py:345-358` forms `C = K_Xs_Xa ∓ K_Xs_X @ C_tmp`.
- Dropping `lnw` from `zz` is a no-op: VIQR's `islogf1` case returns `zeros(size(fs2))` (`acqviqr_vbmc.m:16-19`), and the only-VP branch sets `ActiveImportanceSampling.lnw = lny'`. Python drops it with the same reasoning (`acq_fcn_viqr.py:381-383`) and `_log_viqr_sum(u*s_pred)` equals `logsumexp(u*s + log1p(−exp(−2u*s)))`.
- `ActiveImportanceSamplingMCMCSamples` was removed from `evalfields` so it can be a string evaluated at runtime with `K` and `nvars` bound, plus a positivity check. Python: `active_importance_sampling.py:68-79` calls `options.eval("active_importance_sampling_mcmc_samples", {"K": vp.K, "n_vars": D, "D": D})`, `ceil`s it and raises on a non-positive value. Default is `100` on both sides.

### `c387612` (2022-10-26) — new priors (v1.0.11) — **partly reflected**
- **`vbinit_vbmc.m`: reflected.** `add_jitter = true` moved inside the `for iOpt` loop (it previously leaked `false` from one initialization to the next). Python sets `add_jitter = True` inside the loop at `pyvbmc/vbmc/variational_optimization.py:924`, with `add_jitter = False` only for `i == 0` in types 1 and 2 (`:935-936`, `:960-961`).
- **`defopts.FunEvalStart`: not reflected.** Changed from `max(D,10)` to `10*ceil((D+1)/10)`. PyVBMC still has `fun_eval_start = np.maximum(D, 10)`. Finding **F1**.
- **The prior pdf/rnd families: reflected.** `mtrapezpdf.m` and `msplinetrapezpdf.m` were only renamed/row-vectorized; the formulas are unchanged, and PyVBMC's `Trapezoidal._log_pdf` (`:95-122`) and `SplineTrapezoidal._log_pdf` (`:97-126`) reproduce them (I checked the normalization algebra term by term; PyVBMC folds `log(u−a)` into the tail terms instead of the norm factor, which is the same expression). `munifboxrnd` ↔ `UniformBox.sample`, `mtrapezrnd`/`msplinetrapezrnd` (rejection sampling against the plateau maximum) ↔ the identical rejection loops in `Trapezoidal.sample:141-173` and `SplineTrapezoidal.sample:145-177`, `msmoothboxrnd` ↔ `SmoothBox.sample:118-146` (same `nf = 1 + (b−a)/(√(2π)σ)` component draw, same `u<0.5` / `0.5≤u<1` / `u≥1` split).
  MATLAB-side defect not replicated: `shared/msmoothboxrnd.m` writes `r(idx,d) = a(idx) − z1` and `b(idx) + z1` (linear indexing into column 1) where it means `a(idx,d)` / `b(idx,d)`; for `D>1` MATLAB uses dimension 1's bounds for every tail. `SmoothBox.sample` uses `self.a[d]`/`self.b[d]` and is correct. No PyVBMC finding.
- `add2path` and `runtest_vbmc` renaming are MATLAB infrastructure; N/A.

### `f3c05d6` (2022-10-26) — log pdfs (v1.0.12) — **reflected**
MATLAB added four `m*logpdf.m` files and reduced each `m*pdf.m` to `exp(m*logpdf(...))`. PyVBMC already has exactly this structure: each prior defines `_log_pdf` and `Prior.pdf` exponentiates it (counterpart map rows for `shared/*pdf.m`). I compared all four log densities against the MATLAB files:
`mtrapezlogpdf` vs `Trapezoidal._log_pdf` ✓; `msplinetrapezlogpdf` (`lnf = log(0.5*(c−b+d−a))`, `log(−2z³+3z²)`) vs `SplineTrapezoidal._log_pdf` ✓; `munifboxlogpdf` (`−sum(log(b−a))`, `−inf` outside) vs `UniformBox._log_pdf` ✓; `msmoothboxlogpdf` (`lnf = log(1/(√(2π)σ)) − log1p((b−a)/(√(2π)σ))`, Gaussian tails at `a`/`b`, plateau on `a ≤ x ≤ b`) vs `SmoothBox._log_pdf` ✓, including the closed/open interval conventions.

---

## 3. Findings

### F1. MATLAB's 2022 change to the initial-design size (`FunEvalStart`) is not reflected
- Location: `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:11`; MATLAB: `vbmc.m:198` and commit `c387612` (2022-10-26)
- Category: defaults
- Proposed classification: port discrepancy
- Confidence: high
- History: MATLAB changed `defopts.FunEvalStart` from `max(D,10)` to `10*ceil((D+1)/10)` in `c387612`, whose own message says "modified FunEvalStart option (only for D > 9)". The PyVBMC line has read `np.maximum(D, 10)` since at least `d9bfe16` (2022-08-28, the snake_case rename), i.e. it was ported from the pre-`c387612` MATLAB and never updated. `git log -S "fun_eval_start = "` on the `.ini` returns only that one commit.
- What the code does / should do: PyVBMC starts the run with `max(D, 10)` initial target evaluations; current MATLAB starts with `10*ceil((D+1)/10)`. For `D ≤ 9` the two agree (both 10). For `D = 10…19` MATLAB uses 20 and PyVBMC uses `D`; for `D = 20…29` MATLAB uses 30 and PyVBMC uses `D`.
- Consequence if real: on every problem with `D ≥ 10`, PyVBMC's initial space-filling design is 1.0–2.0× smaller than current MATLAB's (e.g. `D = 10`: 10 vs 20 points; `D = 20`: 20 vs 30). The initial design feeds the first GP fit, so the whole trajectory diverges from MATLAB from iteration 0, and the budget accounting (`max_fun_evals = 50*(D+2)`) is spent differently. This is exactly the regime (10–20 parameters) the algorithm is documented to target. It never triggers for `D ≤ 9`, which is why no fixture caught it.
- Suggested reproduction: `Options(...)` for `D = 12` and print `fun_eval_start` (10 vs MATLAB's `10*ceil(13/10) = 20`); or compare `VBMC(...).options["fun_eval_start"]` against `10*ceil((D+1)/10)` across `D = 1…25`.
- Test adequacy: no. No test asserts the default; `test_vbmc_precomputed.py:16` and `test_vbmc_binding.py:48` override it, and the three `gp_fit_history` oracle fixtures were generated at `D = 2`, where both formulas give 10.

### F2. Six options that MATLAB deleted in 2021 survive in PyVBMC as dead `.ini` entries
- Location: `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:135` (`gp_stochastic_step_size`), `:231` (`annealed_gp_mean`), `:233` (`constrained_gp_mean`), `:235` (`empirical_gp_prior`), `:305` (`integrate_gp_mean`), `:313` (`double_gp`); MATLAB: `vbmc.m` `defopts` block and `misc/setupoptions_vbmc.m`, commit `2044530` (2021-06-18)
- Category: defaults (user-facing option surface)
- Proposed classification: port discrepancy (harmless numerically, misleading to users)
- Confidence: high
- History: `2044530` deleted `defopts.DoubleGP`, `defopts.EmpiricalGPPrior`, `defopts.IntegrateGPMean` and `defopts.GPStochasticStepsize` together with every branch that read them, and (via the missing-comma edit at `setupoptions_vbmc.m:47`) left `AnnealedGPMean` and `ConstrainedGPMean` referenced nowhere. PyVBMC was ported across that boundary and kept all six names.
- What the code does / should do: `grep` over `pyvbmc/` finds each of the six names **only** on its own `.ini` line — no module reads any of them. Because `Options` validation raises on unknown keys and freezes after init, a user can legitimately pass `options={"double_gp": True}` or `{"empirical_gp_prior": True}`, it will be accepted without error, and it will do nothing. The behavior PyVBMC implements is the correct post-cleanup one (e.g. `gaussian_process_train.py:445-457` always uses the fixed plausible-bounds length-scale prior, with the post-cleanup MATLAB comment copied verbatim); only the knobs are vestigial.
- Consequence if real: no numerical effect. The risk is a silently ignored user setting and six documented-but-inert options in the comment-derived user documentation (the `# description` line above each is what users read).
- Suggested reproduction: `VBMC(fun, x0, lb, ub, plb, pub, options={"double_gp": True})` constructs without error; `grep -rn double_gp pyvbmc/` returns only the `.ini` line.
- Test adequacy: no. `test_options.py` exercises the layering machinery, not whether each declared key has a consumer. A test that asserts every `.ini` key is read somewhere would catch this class.

### F3. The rank-1 GP update in active sampling is gated with `and` where MATLAB uses `||`
- Location: `pyvbmc/vbmc/active_sample.py:730-733`; MATLAB: `private/activesample_vbmc.m:468` (touched by commit `2044530`, 2021-06-18)
- Category: control flow
- Proposed classification: possibly intentional
- Confidence: medium (on the deviation: high; on whether it was deliberate: low)
- History: MATLAB reads `update1 = (isempty(s2new) || optimState.nevals(idx_new) == 1) && ~options.NoiseShaping;`. `2044530` edited this very line (it dropped `&& ~options.IntegrateGPMean`), so the `||` is current. The Python line reads `update1 = ((s2new is None) and function_logger.n_evals[idx_new] == 1) and not options["noise_shaping"]`. The `and` predates `2044530`'s edit; only the `integrate_gp_mean` term was at issue in that commit.
- What the code does / should do: `s2new` is `None` exactly when the logger has no `S` array, i.e. for noiseless targets (`function_logger.py:214-215`, `active_sample.py:640-643`). So for noiseless targets both sides take the rank-1 path; for noisy targets MATLAB takes it whenever the new location is fresh (`nevals == 1`), while PyVBMC never does and always falls through to `reupdate_gp(function_logger, gp)`. Note that `gpyreg.GP.update` does accept `s2_new` (`gaussian_process.py:763-769`) and performs a genuine rank-1 Cholesky extension (`:826-900`), so the MATLAB path is available; PyVBMC's call at `:736` omits `s2_new`, which is probably why the condition was tightened.
- Consequence if real: for noisy targets, every intra-iteration GP refresh during active sampling is a full posterior recomputation instead of a rank-1 extension. Mathematically the same posterior, but the Cholesky factors differ at rounding level, so seeded noisy trajectories drift from MATLAB's, and the cost is O(N³) instead of O(N²) per acquired point (with `fun_evals_per_iter = 5` and `N` up to a few hundred, that is the dominant per-iteration GP cost late in a noisy run).
- Suggested reproduction: in a noisy configuration, instrument `active_sample.py:730` and confirm `update1` is `False` on every pass; then compare `gp.posteriors[0].L` after `gp.update(xnew, ynew, s2_new=s2new)` against `reupdate_gp(...)` on the same state — expect agreement to ~1e-12, not bitwise.
- Test adequacy: no existing test distinguishes the two paths; `test_vbmc_active_sample.py` mocks the search rather than exercising the noisy rank-1 branch. This belongs jointly to slice P2.

### F4. The `samples` output struct added in 2021 has no counterpart in `results`
- Location: `pyvbmc/vbmc/vbmc.py:3097-3158` (`_create_result_dict`); MATLAB: `vbmc.m:946-962` and commit `48c82e0` (2021-03-27)
- Category: cross-module (public output surface)
- Proposed classification: possibly intentional
- Confidence: high (on the absence), low (on whether it matters)
- History: `48c82e0` added a sixth return value `samples` with fields `X` (original coordinates), `y` (temperature-scaled), `y_sd`, `active_flag`, `nevals`, and threaded it through the `RetryMaxFunEvals` retry. PyVBMC's `optimize()` returns `(vp, results)` and `results` has no such entry.
- What the code does / should do: the same information is reachable through the public `vbmc.function_logger` (`X_orig`, `y_orig`, `S`, `X_flag`, `n_evals`), so nothing is lost, but a user following MATLAB's documented output will not find it, and the temperature scaling MATLAB applies to `y`/`y_sd` is not applied anywhere on the Python side.
- Consequence if real: documentation/ergonomics only; no numerical effect.
- Suggested reproduction: inspect the keys of the `results` dict returned by `optimize()`.
- Test adequacy: `test_vbmc_optimize.py` checks individual result keys it knows about; nothing would flag a missing one.

### F5. `warp_cov_reg`'s non-numeric branch cannot execute
- Location: `pyvbmc/whitening/whitening.py:151-158`; MATLAB: `misc/warp_input_vbmc.m:59-63`
- Category: control flow
- Proposed classification: suspected defect (Python side only)
- Confidence: medium
- History: not caused by any commit in this range; I found it while checking `f9c04bc`'s neighborhood. MATLAB's `isnumeric(options.WarpCovReg)` / `options.WarpCovReg(optimState.N)` pair is the model.
- What the code does / should do: the `else` branch evaluates `options.warp_cov_reg[optim_state["N"]]`. `Options` is a `MutableMapping`/`dict` subclass (`pyvbmc/vbmc/options.py:17`) with no such attribute, so the expression raises `AttributeError`; and even with attribute access it indexes rather than calls, whereas MATLAB calls a function handle of `N`. It should be `options["warp_cov_reg"](optim_state["N"])`.
- Consequence if real: none at the default (`warp_cov_reg = 0` on both sides, so the numeric branch is always taken). A user supplying a schedule for `warp_cov_reg` gets an exception inside `warp_input` rather than the schedule.
- Suggested reproduction: construct with `options={"warp_cov_reg": lambda N: 0.5}` and reach a warping iteration; or call `warp_input` directly with such an options object.
- Test adequacy: no; `pyvbmc/testing/whitening/` does not vary `warp_cov_reg`.

---

## 4. Sheet notes

Errors and gaps I found in the two reference documents. I did not read anything else under `dev/`.

**Entries I found wrong**
- None of the sheet's entries contradicted the code in the places my slice touched. The two G1/G2 entries my slice depends on (`gplite/private/fminfill.m` "gpyreg's reference copy already carries the fix"; "gpyreg's `SliceSampler` replaces MATLAB's two samplers") both hold, verified by diffing gpyreg's reference copy against MATLAB master and by reading `slice_sample.py:440-453`.
- `counterpart_map.md`, row `misc/initdesign_vbmc.m`, misattributes commit `46b6f5e`: "MATLAB commit `46b6f5e` (2021-05-26) set `Xs = []` for `N <= N0`; slice M checks it". That commit touches only `gplite/private/fminfill.m` and `utils/fminfill.m`; `misc/initdesign_vbmc.m` is untouched. The correct counterpart is `gpyreg/f_min_fill.py` (the map's own `gplite/private/fminfill.m` row), and it reflects the change.
- `counterpart_map.md`, row `gplite/gplite_pred.m`: "MATLAB commit `68a197b` (2022-06-25) fixed the GP log density here and gpyreg's reference copy predates it". True of the reference copy, but potentially misleading: gpyreg's *Python* `GP.predict(return_lpd=True)` already computes the fixed (total-predictive-variance) formula at `gaussian_process.py:2035-2064`.

**Deliberate differences the sheet lacks**
1. **Default bounded transform.** MATLAB `defopts.BoundedTransform = 'logit'` (`vbmc.m:344`); PyVBMC `bounded_transform = "probit"` (`advanced_vbmc_options.ini:311`, passed at `vbmc.py:404`). `AGENTS.md` records probit as intended, but the sheet has no entry, and the word "probit" does not appear in it. Every bounded-variable run therefore uses a different transform from MATLAB's default. Worth an entry (slice P8).
2. **`SearchOptimizer` values.** MATLAB supports `cmaes`, `fmincon`, `bads`, `slicesample`, `none`; PyVBMC supports `cmaes`, `Nelder-Mead`, `none` and raises `NotImplementedError` otherwise (`active_sample.py:552`). `Nelder-Mead` stands in for `fmincon`; `bads` and the `slicesample` branch added by `68a197b` are unported. The sheet's P2 section covers the `cmaes` substitution but not the other three values.
3. **`gp.D == 1` overrides `search_optimizer` permanently.** `active_sample.py:483-488` force-sets `search_optimizer = "Nelder-Mead"` for 1-D problems (a consequence of the `cma` package needing `D ≥ 2`) and never restores it; MATLAB uses CMA-ES at `D = 1`. Python-only, plausibly deliberate, unrecorded.
4. **Six inert options** (Finding F2) — if they are kept on purpose, they belong in the "Python-only user-interface and infrastructure options" entry with a note that they are inert; that entry currently lists only live Python-only options.
5. **The `samples` output struct** (Finding F4) — the `private/vbmc_output.m` row notes `overhead` and `rng_state` but not this.
6. **MATLAB-side defects PyVBMC does not replicate** (useful for a later reviewer, both harmless here): `shared/msmoothboxrnd.m` indexes `a(idx)`/`b(idx)` where it means `a(idx,d)`/`b(idx,d)`, so for `D > 1` MATLAB draws the Gaussian tails around dimension 1's bounds — PyVBMC is correct; and `misc/setupoptions_vbmc.m:47` has a missing comma (`'ConstrainedGPMean''FeatureTest'`) introduced by `2044530`, leaving those two option names unevaluated — PyVBMC's layered `.ini` has no analogue.
