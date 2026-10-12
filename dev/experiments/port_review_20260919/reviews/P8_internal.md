# P8 — parameter transformer, warping, function logger, internal-correctness track

Raw reviewer report, wave 3 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, fresh
context, read-only on the repositories; it did not read the MATLAB source.
Its check scripts ran in the session scratchpad and are kept only on the
machine that ran them (`dev/scripts/runs/LOCAL.md`). The text below is the
agent's final message, unedited.

---

# P8 — parameter transformer, whitening, function logger, input-shape decorator (internal-correctness track)

## 1. Coverage

**Read completely (every line):**
- `pyvbmc/parameter_transformer/parameter_transformer.py` (609 lines)
- `pyvbmc/whitening/whitening.py` (398 lines)
- `pyvbmc/function_logger/function_logger.py` (808 lines)
- `pyvbmc/decorators/handle_0D_1D_input.py` (65 lines)
- The four subpackage `README.md` files; `pyvbmc/vbmc/README.md` (nothing P8-specific in it beyond the MATLAB pointers).
- `dev/experiments/port_review_20260919/known_differences.md` §"Slice P8" and §"Settled non-differences" (plus the P1a/P1b/P2 entries that touch the logger and the warp). No other file under `dev/` was opened.
- Tests: `pyvbmc/testing/parameter_transformer/test_parameter_transformer.py` (index + the init/Jacobian/tail tests in full), `test_parameter_transformer_jacobian_fd.py` (full), `pyvbmc/testing/whitening/test_rotoscaling.py` (full), `pyvbmc/testing/function_logger/test_function_logger.py` (lines 100–319 in full, rest skimmed), `pyvbmc/testing/decorators/test_handle_0D_1D_input_decorator.py` (full).
- `papers/acerbi2020variational_main.md` §2.2 and `_appendix.md` §B.2 (the variational-whitening specification).

**Read as far as needed to judge the slice (callers/consumers):** `pyvbmc/vbmc/vbmc.py` (the warp block 1255–1420, `_init_optim_state` bounds/search-bounds setup 875–1090, `_setup_vbmc_after_warmup` trimming 2060–2115, `_recompute_lcb_max`, `_initialize_precomputed_evaluations`, the `FunctionLogger` construction), `pyvbmc/vbmc/_bounds.py` (full), `pyvbmc/vbmc/active_sample.py` (initial-design and acquisition record paths), `pyvbmc/vbmc/gaussian_process_train.py` (`train_gp` head, `_get_hyp_cov`), `pyvbmc/variational_posterior/variational_posterior.py` (`pdf`/`_pdf`, `log_pdf`, `get/set_parameters` lambda normalization).

**Skimmed / not reached:** the rest of `variational_optimization.py`, the acquisition functions (only their `y_max` read), gpyreg (not needed: nothing in this slice calls into it except `gpr.mean_functions` isinstance checks).

**Checks run** (all in the scratchpad, read-only on the repos, using `.venv/Scripts/python.exe -u`; no `optimize()` run, no test suite):
- `chk_transforms.py` — `_logit/_probit/_student4` and their inverses against `scipy.stats.norm`/`t(4)`, the closed-form log densities, an edge sweep for NaN, `scale` validation, bound aliasing. **All three transform pairs and all three log-density formulas are exactly right** (max abs err ≤ 9e-16; student4 ppf and log-pdf bit-identical to SciPy).
- `chk_logit_tail.py` — the logit Jacobian's tail branch against 60-digit `decimal`, across the `y = -709.7827…` switch. **Exact, continuous, no relative error above 3e-17.**
- `chk_ctor_rotoscale.py` — the constructor with `scale`/`rotation_matrix` (F1).
- `chk_warp.py` — `unscent_warp` exactness for identity and affine maps; `warp_input`'s unit-diagonal property on a *bounded* problem; `warp_gp_and_vp`'s pushforward, weight and lambda normalization. Confirms the composite warp is affine, the length-scale/omega/sigma-lambda refactoring is the exact projected-scale formula, `Σλ² = D` matches `VariationalPosterior.set_parameters`, and the weight update `w·exp((dy−dy_old)/T)` is an exact no-op after renormalization (as it must be for a linear warp).
- `chk_unscent.py` — dtype and broadcast behaviour of `unscent_warp` (F2, F3).
- `chk_logger.py`, `chk_logger2.py` — `fun_eval_time` averaging (F4), repeats into deactivated rows (F5), return types (F12), inverse-variance pooling against exact `fractions` arithmetic (**correct**, including the overflow fallback), `add`'s SD default (F6), the `noise_flag`/level-0 mismatch.
- `chk_psd.py` — the correlation threshold vs. positive-definiteness (F7).
- `chk_oob.py` — out-of-bound inputs to the transformer, half-bounded variables (F10).
- `chk_fvals.py` — `VBMC(..., options={"specify_target_noise": True, "f_vals": [...]})` construction, for F6's reachability.

**What I found to be correct and did not report:** all bounded-transform formulas, inverses and log-Jacobians; the rotation/scale composition in `__call__`/`inverse`/`log_abs_det_jacobian` (the `|det R| = 1` assumption is sound given the construction-time orthogonality check); the mapping of `vp_cov` back through `scale`, `R_mat` and `delta` before the SVD (lines 150–151 — I re-derived it and verified the warped posterior attains unit diagonal covariance on a bounded problem); the unscented `sqrt(D)`/`ddof=1` pairing (self-consistent: the identity map reproduces `sigma` exactly); the GP mean/length-scale/quadratic warp (the log-Jacobian shift is exactly constant for a rotoscaling, so `m0w` and the unchanged output/noise scales are exact, not approximate, in the only warp PyVBMC supports); inverse-variance pooling of repeated noisy observations and its extreme-SD fallback.

---

## 2. Findings

### F1. The constructor derives `mu`/`delta` in the rotated-and-scaled coordinates, then applies them before the rotation
- Location: `pyvbmc/parameter_transformer/parameter_transformer.py:79-80`, `:148-160` (used by `:162-202`); MATLAB: "not read (internal track)"
- Category: control flow (order of operations)
- Proposed classification: suspected defect
- Confidence: high (that the behaviour is inconsistent); the impact is latent
- What the code does: `__init__` stores `scale` and `R_mat` at `:79-80`, *before* computing the centering constants. At `:153-154` it calls `self.__call__(plb_orig)` / `(pub_orig)` with `mu = 0`, `delta = 1`, and `__call__` at `:195-200` applies the rotation and the rescaling at the end. The resulting `plb_tran`/`pub_tran` are therefore coordinates of the **post-rotation, post-scale** space, and `mu[i]`, `delta[i]` are read off them. But in every later call those constants are applied by `_center` at the **pre-rotation** stage (`:188-193` runs before `:197-200`). The two are different coordinate systems whenever `R_mat` mixes coordinates or `scale != 1`.
- What it should do: the comment at `:144` ("Centering (at the end of the transform)") and the code at `:159-160` define `mu` as the midpoint and `delta` as the width of the transformed plausible box, so that the plausible box maps to `[-0.5, 0.5]` per coordinate — which is what happens with no warp installed.
- Consequence if real: a transformer built through the public constructor with `rotation_matrix=`/`scale=` **and** plausible bounds tighter than the hard bounds has a meaningless centering. Reproduction (`chk_ctor_rotoscale.py`, D=2, 40° rotation, `scale=[2, 0.5]`, `plb=[-1,-3]`, `pub=[1,5]`): `pt(plb) = [-0.302, -0.207]`, `pt(pub) = [0.193, 0.288]` instead of `[-0.5,-0.5]` and `[0.5,0.5]`; without the warp arguments the same bounds give exactly `∓0.5`. The map remains a bijection and its log-Jacobian remains self-consistent (I checked against finite differences: agreement to 1.5e-11), so nothing produces NaN — the transformed space is just scaled/offset differently from what the plausible bounds ask for, which propagates to the GP hyperparameter priors and the variational sieve of any caller that builds such a transformer. **`VBMC` never takes this path** (`vbmc.py:414-421` passes no `scale`/`rotation_matrix`, and `warp_input` sets the two attributes after construction and then resets `mu`/`delta` itself), so this is latent in the public API only.
- Suggested reproduction: ran it — `chk_ctor_rotoscale.py`, output above.
- Test adequacy: no. `test_parameter_transformer.py::test_init_type3_mu_all_params` / `test_init_type3_delta_all_params` check `mu`/`delta` by rebuilding them from a *second transformer built the same way* (mirroring the implementation), and neither transformer carries a warp. `test_rotoscaling.py::test_parameter_transformer_log_abs_det` does pass `rotation_matrix` and `scale` to the constructor but leaves the plausible bounds equal to the hard bounds, so the `:149-160` block is skipped. `test_parameter_transformer_jacobian_fd.py` covers rotation+scale thoroughly but installs them *after* construction (`_apply_warp`), exactly the case that works.

### F2. `unscent_warp` silently truncates when `x` has an integer dtype
- Location: `pyvbmc/whitening/whitening.py:59-65`; MATLAB: "not read (internal track)"
- Category: indexing/shape (dtype)
- Proposed classification: suspected defect
- Confidence: high
- What the code does: `xx = np.tile(x, [U, 1, 1])` inherits `x`'s dtype, and `xx[2*d+1, :, d] = xx[2*d+1, :, d] + sigma_slice` writes a float sum back into an integer array, truncating it. Neither `x` nor `sigma` is ever cast to float.
- What it should do: the sigma points are `x ± sqrt(D)·sigma·e_d`; they must be computed in floating point whatever the caller passes.
- Consequence if real: wrong mean *and* wrong warped scale, with no warning. Reproduction (`chk_unscent.py`): `unscent_warp(identity, np.array([[1,2],[3,4]]), np.array([0.25,0.25]))` returns mean `[0.8, 1.8, 2.8, 3.8]` (should be `[1,2,3,4]`) and sigma `0.447` (should be `0.25`). Not reachable from inside PyVBMC — `gp_old.X`, the hyperparameter slices and `vp.mu` are all float64 — but `unscent_warp` is exported from `pyvbmc.whitening` with a full numpydoc contract.
- Suggested reproduction: ran it (above).
- Test adequacy: no. `test_rotoscaling.py::test_unscent_warp` uses float inputs only.

### F3. `unscent_warp`'s single-row-`x` broadcast branch cannot work
- Location: `pyvbmc/whitening/whitening.py:50-53` together with `:74-75`; MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high
- What the code does: `:35` captures `x_shape_orig = x.shape` *before* `np.atleast_2d`; `:50-51` tiles a single-row `x` up to `N` rows to match `sigma`; `:74-75` then reshape the `(N, D)` results back to `x_shape_orig`, which is still `(1, D)`.
- What it should do: the docstring declares `x : (n,D) or (D,)` and `sigma : (n,D) or (D,)` and the explicit tiling shows the author intended `x` with one row and `sigma` with many to be supported. The output shape should follow the broadcast result, not the pre-broadcast `x`.
- Consequence if real: `unscent_warp(f, x_1xD, sigma_NxD)` raises `ValueError: cannot reshape array of size 6 into shape (1,2)` instead of returning `N` warped points. The mirror branch (`sigma` with one row, the only one in-tree use needs, at `:324`) works. Public-API only.
- Suggested reproduction: ran it — `chk_unscent.py` case A.
- Test adequacy: no test exercises `N1 == 1 < N2`.

### F4. A repeated observation with an unknown evaluation time overwrites the row's recorded time with NaN
- Location: `pyvbmc/function_logger/function_logger.py:719-723`, against the guard at `:734-736`; MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- What the code does: the duplicate branch unconditionally computes `self.fun_eval_time[idx] = (N * self.fun_eval_time[idx] + fun_eval_time) / (N + 1)` and only *afterwards* tests `if not np.isnan(fun_eval_time)` before adding to `total_fun_eval_time`. The new-row branch at `:734-736` puts the same NaN test *around* the assignment. `add()` defaults `fun_eval_time=np.nan` (`:472`), and `active_sample.py:218` and `:698` call it that way.
- What it should do: the running average is of *measured* times; an unknown time should leave the stored average alone, as the new-row branch already does. The asymmetry between the two branches is the tell.
- Consequence if real: `function_logger.fun_eval_time[idx]` becomes NaN permanently for that row, and stays NaN through every later repeat (`(N·NaN + t)/(N+1)`). `total_fun_eval_time` stays correct. Reproduction: record `(x, 0.5 s)` then `add(x, y)` → the row's time goes `0.5 → nan` (`chk_logger.py` section A). Small consequence — `fun_eval_time` is per-row bookkeeping, and the timing reported to the user comes from `total_fun_eval_time` — but it is a silent, irreversible corruption of a recorded quantity.
- Suggested reproduction: ran it (above).
- Test adequacy: no. `test_function_logger.py::test_record_duplicate` asserts `fun_eval_time[1] == 5` for two *known* times (9 and 1), mirroring `(N·t + t_new)/(N+1)`; no test records a repeat with an unknown time.

### F5. A repeat at an input whose row was deactivated is pooled into the dead row and never re-enters the training set
- Location: `pyvbmc/function_logger/function_logger.py:657-661` and `:724-727` (no `X_flag` write in the duplicate branch; contrast `:749`); with `pyvbmc/vbmc/vbmc.py:2104` and `pyvbmc/whitening/whitening.py:213-221`; MATLAB: "not read (internal track)"
- Category: state/caching
- Proposed classification: suspected defect
- Confidence: medium (the behaviour is certain; how reachable it is in a default run, low)
- What the code does: `duplicate_flag = (self.X == x).all(axis=1)` scans **every** row of `self.X`, including rows whose `X_flag` the warm-up trimming (`vbmc.py:2104`, `X_flag = idx_keep & X_flag`) has set to `False`. When a match lands on such a row, the value is pooled into `y_orig[idx]`/`S[idx]`, `n_evals[idx]` is incremented, `func_count` is charged — and `X_flag[idx]` stays `False`, so the observation is invisible to `get_traindata`, the GP, `_recompute_lcb_max` and `y_max` (`:726` maxes over `X_flag` only).
- What it should do: either the match should be restricted to live rows, or pooling into a dead row should revive it. Silently paying for a target evaluation that no consumer can see is neither.
- Consequence if real: one or more wasted target evaluations, and an `n_evals` count that no longer matches the training set. Reproduction (`chk_logger.py` section B): two points logged, `X_flag[0] = False`, then re-call at point 0 → `idx = 0`, `X_flag[0] == False`, `n_evals[0] == 2`, `func_count == 3`, `Xn` unchanged. Reachability in a real run is low: the repeat-candidate set (`active_sample.py:403`) is drawn from `function_logger.X[X_flag]`, so it never offers a dead row, and a CMA-ES point hitting a dead row by exact float equality is essentially impossible. A related, milder consequence of the same gap: `warp_input` rewrites only `function_logger.X[X_flag]` (`:219`) and `y[X_flag]` (`:220`), so after a warp the dead rows' `X` and `y` are stale coordinates of the *previous* inference space while their `X_orig`/`y_orig` are not — the two arrays disagree, and the duplicate scan compares new-space points against old-space rows.
- Suggested reproduction: ran it (above). To settle whether it is worth fixing, the question is whether any path can propose a previously trimmed input exactly.
- Test adequacy: no. `test_function_logger.py::test_finalize_preserves_noisy_duplicate_and_inactive_rows_then_appends` sets `X_flag[0] = False` *after* the duplicate and only checks that `finalize` preserves it; `test_vbmc_active_sample.py:1333` sets a flag false for an unrelated purpose. Nothing records a repeat *at* a deactivated row.

### F6. `FunctionLogger.add` fabricates `f_sd = 1` on a user-noise (level 2) target
- Location: `pyvbmc/function_logger/function_logger.py:528-532`, reached from `:461` (`batch_call`'s cached rows) and from `pyvbmc/vbmc/active_sample.py:218`, `:698`; contrast `__call__` at `:290-291`; MATLAB: "not read (internal track)"
- Category: defaults / cross-module
- Proposed classification: suspected defect
- Confidence: medium-high
- What the code does: `add` does `if self.noise_flag: if f_sd is None: f_sd = 1`, with no reference to `self.uncertainty_handling_level`. `__call__` distinguishes the levels: at level 2 it unpacks `f_val, f_sd = self.fun(x_orig)` and validates the SD; SD 1 is the level-1 convention only.
- What it should do: at `uncertainty_handling_level == 2` the whole point is that the SD is supplied per observation; a missing SD should be refused, not replaced by an arbitrary value with the same units as the log density. `_initialize_precomputed_evaluations` (`vbmc.py:721-730`, `:801-803`) shows the intended contract — it *requires* `y_sd` when the level is 2 and passes it to `add`.
- Consequence if real: with `options={"specify_target_noise": True, "f_vals": [...]}`, every supplied value enters the GP with `S = 1`, whatever the target's real noise is; nothing warns. I confirmed construction accepts this combination (`chk_fvals.py`: `uncertainty_handling_level = 2`, `cache["y_orig"] = [-0.025, -0.05]`, `add(...)` → `S = 1.0` while the target returns SD 0.01). The `f_vals` interface has no SD channel at all, so on a level-2 run it can only produce wrong noise. Both the sequential (`active_sample.py:218`) and the vectorized (`batch_call`, `:461`) initial-design paths go through it, as does an acquired cached starting point (`active_sample.py:698`). A GP trained with noise 1 where the truth is 0.01 will over-smooth exactly at the design points.
- Suggested reproduction: ran the recording half (`chk_logger2.py` "batch_call with a cached value on a level-2 noisy target" → `sds = [1.0, 0.3]`, row 0 supplied, row 1 measured) and the construction half (`chk_fvals.py`). A full settle would be a short seeded run comparing the fitted noise hyperparameter with and without `f_vals`.
- Test adequacy: no — worse, `test_function_logger.py::test_add_no_f_sd` *pins* the current behaviour (`add(x, y, None)` → `S == 1`), but only on a **level-1** logger, where SD 1 is the correct convention. No test calls `add` without an SD on a level-2 logger.

### F7. The correlation threshold can make the covariance indefinite, and the whitening then silently misses unit variance
- Location: `pyvbmc/whitening/whitening.py:154-159` and `:179-184`; MATLAB: "not read (internal track)"
- Category: formula
- Proposed classification: possibly intentional
- Confidence: medium
- What the code does: `vp_cov[|corr| <= warp_roto_corr_thresh] = 0` (default threshold 0.05), then `U, s, __ = np.linalg.svd(vp_cov)` and `scale = sqrt(s + eps)`. The SVD is used as an eigendecomposition, which is only valid for a symmetric **positive semi-definite** matrix. Zeroing off-diagonal entries of a valid covariance can leave it indefinite; the SVD then returns `s = |λ|` and flips the sign of the corresponding `U` column, and nothing notices.
- What it should do: the paper (`papers/acerbi2020variational_appendix.md` §B.2 and `_main.md` §2.2) defines the transform as "a rotation and rescaling of the inference space such that the variational posterior obtains unit diagonal covariance", and the module docstring (`whitening.py:85-87`) repeats it. When an eigenvalue of the corrected matrix is negative, `sqrt(|λ|)` is not the whitening scale for that direction and the goal is missed. Note the paper prescribes the same SVD and says nothing about the corrected matrix staying PSD, so the recipe itself may be the source; that is why I classify it as possibly intentional rather than as a port error.
- Consequence if real: reproduction (`chk_psd.py`) with the valid correlation matrix `[[1,.72,.72],[.72,1,.04],[.72,.04,1]]` (eigenvalues `1.6e-3, 0.96, 2.04`): thresholding zeroes the `.04` pair and gives eigenvalues `-0.018, 1, 2.018`. The attained covariance in the warped space is `diag(1.010, 0.960, 0.0969)` — the narrowest direction, the one the whitening exists to fix, ends up with variance 0.097 instead of 1, a scale error of a factor ~3.2. The sign of the third `U` column is also flipped relative to the eigenvector. In the worst case `|λ|` can approach `eps`, and `scale = sqrt(|λ| + eps)` then stretches that coordinate by up to ~1e8. This is a plausible posterior shape (two parameters each correlated with a third, weakly correlated with each other), not a contrived one.
- Suggested reproduction: ran it (above). A cheap guard check would be `np.linalg.eigvalsh(vp_cov).min() > 0` after the threshold, on the stored warp states.
- Test adequacy: no. `test_warp_input` and `test_warp_gp_and_vp` use a two-dimensional posterior with a single strong correlation (the threshold never bites); `test_warp_input_cov_reg` only checks that three spellings of the same regularization agree and that a different value gives a different transform.

### F8. `warp_input`'s "Reset GP Hyperparameters" resets the variational-moment average, not the GP hyperparameter statistics
- Location: `pyvbmc/whitening/whitening.py:254-257`; consumers `pyvbmc/vbmc/vbmc.py:1578-1595`, `pyvbmc/vbmc/gaussian_process_train.py:136-150`, `:194-203`, `:691-724`; MATLAB: "not read (internal track)"
- Category: state/caching
- Proposed classification: unsure
- Confidence: low-medium
- What the code does: under the comment `# Reset GP Hyperparameters`, `warp_input` clears `optim_state["run_mean"]`, `["run_cov"]` and `["last_run_avg"]`. Those three keys are the running average of the **variational moments** (`vbmc.py:1578-1595`), not GP hyperparameters. The GP hyperparameter statistics that survive the warp in the *old* coordinate system are `hyp_dict["run_cov"]` and `hyp_dict["full"]` (`gaussian_process_train.py:194-203`, read by `_get_hyp_cov` at `:724` to set the slice sampler's widths) and the recorded GPs in `iteration_history["gp"]`, which `:136-150` concatenates into the optimizer's starting points. `warp_gp_and_vp` warps only `hyp_dict["hyp"]`.
- What it should do: after a rotoscaling the length-scale, quadratic-mean-location and quadratic-mean-width hyperparameters all change meaning (`whitening.py:322-363` recomputes exactly those), so a covariance and a sample set accumulated before the warp describe a different parameterization. Either the comment names a reset the code does not perform, or the reset is in the wrong place.
- Consequence if real: after a warp the hyperparameter slice sampler's proposal widths, and the L-BFGS-B starting points, are drawn from pre-warp coordinates. This does not bias the fitted posterior (the sampler is still targeting the correct density) but it changes the trajectory and can cost efficiency in the iterations right after a warp. I did not measure it.
- Suggested reproduction: instrument `_get_hyp_cov`'s return across a warp in one short seeded run and compare with the post-warp `np.cov(hyp_dict["full"].T)`; or read what MATLAB's `warp_input_vbmc.m` resets (the comparison reviewer's job).
- Test adequacy: no test inspects `hyp_dict` across a warp.

### F9. `scale` is unvalidated, and `log_abs_det_jacobian` takes `log(scale)` where `log|scale|` is the correct term
- Location: `pyvbmc/parameter_transformer/parameter_transformer.py:79` (no validation) and `:292-294`; contrast the rotation validation at `:58-77`; MATLAB: "not read (internal track)"
- Category: formula / defaults
- Proposed classification: suspected defect
- Confidence: medium
- What the code does: `rotation_matrix` is checked for shape, realness, finiteness and orthogonality; `scale` is stored with no check at all. The Jacobian adds `np.log(self.scale)` per coordinate. For a negative entry that is NaN, even though the map is a perfectly good bijection whose |det| is well defined; for a zero entry the map is singular and `__call__` divides by zero.
- What it should do: the method is named `log_abs_det_jacobian` and the sheet's own entry ("Rotation matrices are validated as orthogonal … reflections are accepted") shows that sign-reversing transforms are meant to be supported. A reflection expressed in `R_mat` is handled correctly (I verified: `log|det J| = 0`); the same reflection expressed as a negative `scale` gives NaN. Either `scale` should be required positive at construction, as `rotation_matrix` is required orthogonal, or the term should be `np.log(np.abs(scale))`.
- Consequence if real: a silently NaN log-Jacobian propagating into `function_logger.y` and `vp.pdf`, or a silently singular transform. `warp_input` always produces `scale = sqrt(s + eps) > 0`, so this is latent in the public API only.
- Suggested reproduction: ran it (`chk_transforms.py`): `ParameterTransformer(2, scale=[1., -2.])` → `log_abs_det_jacobian = [nan]` with a `RuntimeWarning`, round trip still exact; `scale=[1., 0.]` → `-inf` and a broken round trip.
- Test adequacy: no. `test_init_rotation_matrix_validation` exists; there is no `scale` counterpart.

### F10. Transformer state is aliased to caller arrays, and a half-bounded variable is silently treated as unbounded
- Location: `pyvbmc/parameter_transformer/parameter_transformer.py:106-107`, `:79-80` (aliasing) and `:135-142` (bound typing); MATLAB: "not read (internal track)"
- Category: state/caching; defaults
- Proposed classification: suspected defect (both parts), low impact
- Confidence: high on the behaviour, low on the impact
- What the code does: (a) `self.lb_orig = lb_orig`, `self.ub_orig = ub_orig`, `self.scale = scale`, `self.R_mat = rotation_matrix` store the caller's arrays by reference (contrast `_normalize_bounds`, which explicitly detaches its inputs so "the permissive bound repairs must not modify arrays owned by the caller"). (b) `self.type[i]` is set to the bounded type only when *both* bounds are finite; a variable with one finite bound gets type 0 and is carried through the identity, so the finite bound is not enforced in either direction.
- What it should do: `AGENTS.md` records the invariant "a transformer is never mutated after construction; a warp installs a fresh copy" — an alias makes that invariant depend on the caller. And the class docstring says `lb_orig`/`ub_orig` "define a set of **strict** lower and upper bounds for each parameter"; a half-bounded specification is accepted and the bound silently ignored, with no error and no warning.
- Consequence if real: (a) reproduction — build a transformer from an array, mutate one entry of that array, and the transform changes (`chk_transforms.py`: `pt([[0.5,0.5]])` goes from `[1.0986, 1.0986]` to `[5.303, 1.0986]`). Not reachable from `VBMC`, whose bounds are freshly detached, but `ParameterTransformer` is a documented public class. (b) `VBMC` rejects half-bounds outright (`_bounds.py:291-299`, `vbmc:HalfBounds`), so this too is public-API only; a user driving the transformer directly (e.g. for a Torch export) gets no support and no diagnostic. Related documentation defect, same file: the constructor docstring documents the parameter as `bounded_transform_type` (`:42-44`) while the signature calls it `transform_type` (`:56`), so the documented keyword raises `TypeError`.
- Suggested reproduction: ran both (`chk_transforms.py`, `chk_oob.py`).
- Test adequacy: no test mutates an input array after construction; `test_init_mixed_bounds` covers fully-bounded-mixed-with-fully-unbounded, never a half-bounded variable.

### F11. `warp_input` scales the log-Jacobian by the temperature; `_record` does not
- Location: `pyvbmc/whitening/whitening.py:217-218` against `pyvbmc/function_logger/function_logger.py:716-718`, `:742-746`; MATLAB: "not read (internal track)"
- Category: formula
- Proposed classification: unsure
- Confidence: low
- What the code does: `warp_input` sets `function_logger.y[X_flag] = y_orig + dy / T`; `FunctionLogger._record` sets `y = y_orig + log_abs_det_jacobian(x)` with no `T`. The same array is written by two formulas that disagree whenever `T != 1`.
- What it should do: `function_logger.y` has one definition; whichever one is right, both writers should use it.
- Consequence if real: none today — posterior tempering is unported and `temperature` is in `INERT_OPTIONS` (sheet, P7/P1b), so `T` is always 1 and `optim_state.get("temperature")` even falls back to 1 when the key is absent. It would bite whoever revives tempering.
- Suggested reproduction: set `optim_state["temperature"] = 2` and compare `function_logger.y` after `warp_input` with what a fresh `_record` of the same point would write.
- Test adequacy: no test sets a temperature.

### F12. `handle_0D_1D_input` does not unwrap a 0-D input, and `input_dims` can be unbound
- Location: `pyvbmc/decorators/handle_0D_1D_input.py:26-46`; MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: possibly intentional
- Confidence: medium
- What the code does: three separate gaps. (a) The docstring says the decorator "handles 0D, 1D inputs", but only `input_dims == 1` triggers the unwrapping at `:46`; a 0-D input gets the raw 2-D result back even with `return_scalar=True`. Reproduction (`chk_logger2.py`, D=1): `pt.log_abs_det_jacobian(np.float64(0.3))` → `array([0.693])` while `pt.log_abs_det_jacobian(np.array([0.3]))` → `np.float64(0.693)` — the *more* scalar input gets the *less* scalar output. (b) `input_dims` is assigned only inside the loop body; if neither the kwarg nor the positional slot is present, line 46 raises `UnboundLocalError`. All four in-tree decorated methods require their patched argument, so the wrapped call raises `TypeError` first (I confirmed: `vp.log_pdf()` → `TypeError: pdf() missing 1 required positional argument: 'x'`) — the gap only opens if a decorated method ever gives that argument a default. (c) `input_dims` is a single variable reused across the `patched_kwargs` loop, so with two or more patched arguments only the last one's dimensionality decides the unwrapping; no in-tree call site passes more than one.
- What it should do: (a) either document that 0-D is out of scope or fold it into the `return_scalar` path; (b) initialise `input_dims` (or raise a clear error); (c) track the dimensionality per argument.
- Consequence if real: (a) an inconsistent public return type on three `ParameterTransformer` methods and two `VariationalPosterior` methods; (b) and (c) are latent traps for the next caller. `pyvbmc/decorators/README.md` already flags the decorator as needing a rethink.
- Suggested reproduction: ran (a) and (b) (`chk_logger2.py`).
- Test adequacy: no. `test_handle_0D_1D_input_decorator.py` has eight tests covering 1-D and 2-D inputs, keyword and positional, tuple and scalar returns — and not one 0-D input, despite the decorator's name, nor a missing patched argument, nor two patched arguments.

### F13. `FunctionLogger` returns a `(1,)` array for a repeat and a scalar for a fresh point
- Location: `pyvbmc/function_logger/function_logger.py:715-727` against `:741-752`, and the vectorized path at `:255-268`; MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: suspected defect (cosmetic today)
- Confidence: medium
- What the code does: the duplicate branch builds `f_val = self.y_orig[idx].copy()`, a `(1,)` array, and returns it; the new-row branch builds a NumPy scalar. The docstrings of `__call__` and `add` both declare `f_val : float`. The vectorized `__call__` path (`:266-268`) always returns `values[0]`, a plain float, so the two `__call__` routes also disagree with each other. Reproduction (`chk_logger.py` section C): fresh point → `float64 ()`, repeat → `ndarray (1,)`.
- What it should do: one declared return type.
- Consequence if real: none observed today. The one consumer that would break, `active_sample.py:806` (`ynew = np.array([[ynew]])` before the rank-one `gp.update`), is guarded by `update1 = n_evals[idx_new] == 1`, which is true exactly in the branch that returns a scalar. That guard is the only thing standing between this and a `(1,1,1)` array reaching gpyreg, and it was set for a different reason (see the sheet's P2 entry on the rank-one update).
- Suggested reproduction: ran it (above).
- Test adequacy: no. `test_record_duplicate` unpacks `_, idx = f_logger._record(...)` and never looks at the value's type.

### F14. `batch_call` and `__call__` disagree when `noise_flag` is true at uncertainty level 0
- Location: `pyvbmc/function_logger/function_logger.py:144-152`, `:416-419`, `:440-465` against `:294-296`; MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: unsure
- Confidence: low
- What the code does: `_validate_vectorized_target_output` returns `sds = None` unless the level is 1 or 2, while `batch_call` still allocates `sds = np.empty(n_rows)` whenever `noise_flag` is set; `sds[row] = None` then writes NaN, and `_record` leaves `self.S[Xn]` at NaN. The sequential `__call__` returns SD 1 for the same configuration. Reproduction (`chk_logger2.py`): vectorized → `sds = array([nan])`; sequential → `f_sd = 1`.
- What it should do: `noise_flag` and `uncertainty_handling_level` should not be independently settable into an inconsistent pair, or the two routes should agree.
- Consequence if real: none from `VBMC`, which sets `noise_flag = (level > 0)` at `vbmc.py:465`. A NaN in `S` would reach the GP if it ever occurred.
- Suggested reproduction: ran it (above).
- Test adequacy: no test constructs that pair.

---

## 3. Test adequacy notes

Tests in this slice that pin the implementation rather than the specification:

1. **`test_parameter_transformer.py::test_init_type3_mu_all_params` and `::test_init_type3_delta_all_params`** — the expected `mu`/`delta` are produced by calling *the same constructor* on a second transformer without plausible bounds and then transforming the plausible bounds through it. That is a restatement of lines `:153-160`, not of the specification ("the plausible box maps to `[-0.5, 0.5]`"). Asserting `pt(plb) == -0.5` and `pt(pub) == +0.5` would be the spec, and would catch F1.
2. **`test_parameter_transformer.py::test_log_abs_det_jacobian_logit_extreme_tails`** — asserts `np.array_equal` against `-y + 2*(-log1p(exp(-y)))`, the code's own ordinary-branch expression, and against `-|y| - 2*log1p(exp(-|y|))` for the tails, the code's own tail expression. It does check the property that matters (finiteness beyond `y = -709.78`), but the values are the implementation's. I compared them against 60-digit `decimal` arithmetic instead and they are exact, so nothing is hiding here — but the test would not have detected a wrong constant.
3. **`test_function_logger.py::test_add_no_f_sd`** — asserts `S == 1` after `add(x, y, None)`. On the level-1 logger it uses, SD 1 is the specification. The test encodes the *mechanism* (`f_sd is None → 1`) rather than the rule (`level 1 → 1; level 2 → the caller must supply one`), which is why F6 is unguarded.
4. **`test_function_logger.py::test_record_duplicate`** — `fun_eval_time[1] == 5` restates `(N·t + t_new)/(N+1)` for two known times; the specification question, what a repeat with an unknown time should do to a recorded average, is not asked (F4).
5. **`test_rotoscaling.py::test_warp_input_cov_reg`** — the assertions are that three spellings of `0.75` give *the same* transform and that `0.0` gives *a different* one. Neither pins `(1-w)·C + w·diag(C)`; any monotone use of `w` would pass.
6. **Coverage gaps worth naming**, beyond the per-finding notes: every warp test in `test_rotoscaling.py` runs on an *unbounded* transformer (they all assert `type == [0, 0]`), so the warp's interaction with the bounded transforms and with `mu`/`delta` ≠ 0/1 is untested — I exercised it by hand (`chk_warp.py`) and it is correct, including the exact unit diagonal. And `plb_tran`/`pub_tran`, the only outputs of `warp_input` that depend on its 100 000 random draws (`whitening.py:189-204`), are asserted nowhere: `test_warp_input` checks `R_mat`, `scale`, `mu`, `delta`, `type` and the bounds against MATLAB values, and stops there. `test_rotoscaling_rotation_2d` draws its rotation angle from an unseeded `np.random.uniform`.

Tests that are genuinely spec-grounded and worth keeping in mind as the gate for this slice: `test_parameter_transformer_jacobian_fd.py` (builds the full `D×D` Jacobian of `inverse` by central differences and compares `slogdet` with the closed form, across 11 transform cases × 3 dimensions × 3 warp configurations — a real independent check, which is why I found nothing wrong with the Jacobian); `test_function_logger.py::test_record_duplicate_extreme_finite_sds` and `::test_record_duplicate_transform_preserves_original_space_pool` (both recompute the pooled value from the inverse-variance formula); and the MATLAB-derived reference values in `test_rotoscaling.py::test_unscent_warp` and `::test_warp_gp_and_vp`.
