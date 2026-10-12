# Wave 3 fix pass, agent C: parameter transformer, warping, function logger

Raw report of a fix agent, wave 3 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, in
its own git worktree cut from `dev-port-review` at `fc8e561`, fixed the
wave-3 findings of slice P8 (W3-19, W3-6, W3-34, W3-27, W3-21, W3-22,
W3-32, W3-28, W3-23, W3-26, W3-24, W3-25, W3-31), one finding per commit
with a test written against the contract, running only the focused test
files it touched. The commit hashes in the report are those of the agent's
worktree branch; the orchestrator reviewed each diff and cherry-picked the
commits onto `dev-port-review`, where they carry other hashes
(`../verification/wave3.md` lists them). The test the report stopped on was
corrected by the orchestrator. The text below the rule is the agent's final
message, unedited.

---

# Wave-3 fix pass, agent C

Worktree: `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-ad76352a6deebb612`
Branch: `worktree-agent-ad76352a6deebb612` (fast-forwarded from `dev-port-review`; nothing pushed)

## Setup outputs

- `git merge --ff-only dev-port-review` → fast-forward `b2c16ebf..fc8e5617`; `git log -1 --format=%h` printed `fc8e5617`.
- `python -c "import pyvbmc; print(pyvbmc.__file__)"` from the worktree root printed
  `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-ad76352a6deebb612\pyvbmc\__init__.py`.
- Every Python command ran with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Working tree clean at the end (`git status --porcelain --ignored`, ignoring `!!`, is empty). No standalone script was run against the package (the one `PYTHONPATH=<worktree>` run, reading the oracle fixture JSONs, printed the worktree's `pyvbmc.__file__`).
- **Process note:** early on I ran `git stash push -u` once by mistake while checking a pre-fix failure. I restored immediately with `git stash apply <sha>` and dropped the entry by its unique tag; the stash stack is empty of my entries and nothing was lost. Every later pre-fix check was done by editing the source file in place and restoring it.

## `git log --oneline fc8e561..HEAD`

```
f53e8eb3 docs(decorators): state what a 0-D input returns
c5fcb50a fix(parameter_transformer): derive the centering before the rotation
ace07a9a docs(parameter_transformer): name the bounded-transform keyword correctly
29691404 fix(parameter_transformer): keep copies of the arrays given to it
e7db7dae fix(parameter_transformer): validate the rescaling vector
e32b8044 fix(parameter_transformer): refuse a variable bounded on one side only
dfa49bb6 fix(vbmc): refuse cached starting values for a target with given noise
fdd888b9 fix(function_logger): require the SD of a value added at level 2
c7aa724b fix(function_logger): refuse a noise flag that contradicts the level
9a368ba8 fix(function_logger): return the recorded value as the documented float
f10115e4 fix(function_logger): keep a recorded evaluation time over a repeat
3a766567 fix(whitening): honour the documented shapes and dtypes of unscent_warp
0c131f69 fix(whitening): drop a covariance entry whose correlation is undefined
06922a1b fix(whitening): keep the covariance a threshold would make indefinite
92abfa31 fix(whitening): re-express every filled row of the logger at a warp
```

Diffstat over the range: 9 files, +685 / −43. Source files touched: `pyvbmc/whitening/whitening.py`, `pyvbmc/function_logger/function_logger.py`, `pyvbmc/parameter_transformer/parameter_transformer.py`, `pyvbmc/decorators/handle_0D_1D_input.py`, `pyvbmc/vbmc/vbmc.py` (seven lines, the one construction check).

## Commits in order

| # | hash | finding | what it does | files | tests added or changed |
|---|---|---|---|---|---|
| 1 | `92abfa31` | W3-19 | the warp re-transforms rows `0..Xn` of the logger instead of the rows `X_flag` selects; the mask is untouched | `whitening.py`, `test_rotoscaling.py` | +`test_warp_input_rewrites_every_filled_row_of_the_logger` |
| 2 | `06922a1b` | W3-6 | the thresholded covariance is used only when it is positive definite (`_is_positive_definite`, a Cholesky test); otherwise the covariance before the threshold is used | `whitening.py`, `test_rotoscaling.py` | +`test_warp_input_keeps_a_covariance_the_threshold_makes_indefinite`, +`test_warp_input_whitens_the_thresholded_covariance_when_it_is_definite`, + helpers `_unbounded_vbmc`, `_posterior_with_covariance`, `_whitening_map` |
| 3 | `0c131f69` | W3-34 | the mask follows MATLAB's `abs(corr) > thresh` keep-rule, in a new private `_drop_low_correlations` | `whitening.py`, `test_rotoscaling.py` | +`test_drop_low_correlations_keeps_only_strong_correlations`, +`test_drop_low_correlations_drops_an_undefined_correlation` |
| 4 | `3a766567` | W3-27 | `unscent_warp` converts both inputs to float and keeps one result row per row of the broadcast, reshaping to the given mean's shape only when the broadcast is one row; docstring updated | `whitening.py`, `test_rotoscaling.py` | +`test_unscent_warp_does_not_truncate_an_integer_mean`, +`test_unscent_warp_broadcasts_one_mean_over_several_scales` |
| 5 | `f10115e4` | W3-21 | the guard for an unknown time moves around the running average in the duplicate branch of `_record`; the charge to `total_fun_eval_time` is unchanged | `function_logger.py`, `test_function_logger.py` | +`test_record_duplicate_unknown_time_keeps_the_stored_average` |
| 6 | `9a368ba8` | W3-22 | both branches of `_record` return `float(...)`; the pooled value is read out of its row (`y_orig[idx, 0]`) instead of copied | `function_logger.py`, `test_function_logger.py` | +`test_recorded_value_is_a_float_for_a_new_point_and_for_a_repeat` (parametrized, with and without a transformer) |
| 7 | `c7aa724b` | W3-32 | `FunctionLogger.__init__` requires `noise_flag == (uncertainty_handling_level > 0)`; class docstring says so | `function_logger.py`, `test_function_logger.py` | +`test_init_rejects_a_noise_flag_that_contradicts_the_level`, +`test_init_accepts_a_noise_flag_that_matches_the_level` |
| 8 | `fdd888b9` | W3-28 | `add` without an SD raises at level 2; level 1 keeps SD 1; `add` docstring updated | `function_logger.py`, `test_function_logger.py` | +`test_add_without_an_sd_is_refused_at_level_two` |
| 9 | `dfa49bb6` | W3-28 | `VBMC` refuses `f_vals` with `specify_target_noise` at construction, pointing to `precomputed_evaluations` | `vbmc.py`, new `test_vbmc_cached_values_noise.py` | +3 tests in the new file |
| 10 | `e32b8044` | W3-23 | the transformer constructor refuses a variable with one finite and one infinite bound, naming the dimensions | `parameter_transformer.py`, `test_parameter_transformer.py` | +`test_init_rejects_half_bounded_variables`; changed `test_init_lower_bounds` and `test_init_upper_bounds` (each gave one bound only) |
| 11 | `e7db7dae` | W3-26 | `scale` is checked for shape `(D,)`, realness, finiteness and positivity, in the style of the rotation checks | `parameter_transformer.py`, `test_parameter_transformer.py` | +`test_init_scale_validation` |
| 12 | `29691404` | W3-24 | the constructor stores copies of `lb_orig`, `ub_orig`, `scale`, `rotation_matrix` | `parameter_transformer.py`, `test_parameter_transformer.py` | +`test_init_copies_the_arrays_it_is_given` |
| 13 | `ace07a9a` | W3-24 | the class docstring documents `transform_type`, the keyword the signature has | `parameter_transformer.py`, `test_parameter_transformer.py` | +`test_class_docstring_documents_the_constructor_arguments` |
| 14 | `c5fcb50a` | W3-25 | `mu` and `delta` are measured with the rotation and the rescaling held back, so they act on the coordinates they are applied to | `parameter_transformer.py`, `test_parameter_transformer.py` | +`test_centering_takes_the_plausible_box_to_the_unit_interval`, +`test_centering_is_derived_before_the_rotation_and_the_rescaling` |
| 15 | `f53e8eb3` | W3-31 | the decorator's docstring states that a 0-D input's result comes back as the wrapped function produced it | `handle_0D_1D_input.py` | none (the eight `test_0D_*` tests already pin it) |

Every fix was seen to fail first: the new test was run against the unfixed source (the hunk edited out in place through a scratch script, then restored) before each commit.

## Contract and test, per commit

1. **W3-19.** `misc/warp_input_vbmc.m:112-119` re-transforms `idx_n = 1:optimState.Xn`. Test: a `VBMC` whose logger holds four added evaluations with row 1 deactivated; after the warp, every filled row must satisfy `X == new(X_orig)` and `y == y_orig + new.log_abs_det_jacobian(X)`, the mask must be unchanged, and the rewrite must be non-vacuous. Before: row 1 stayed at `[1.1, 0.2]` where the new space wants `[0.0257, −0.987]`.
2. **W3-6.** The PI's ruling. Test 1: a posterior with correlations 0.72, 0.72, 0.04 (`vp.moments` stood in for, as sanctioned); the linear map the installed transform applies must give the covariance unit diagonal. Before: `1.0099, 0.96, 0.0969` — the ledger's `1.01, 0.96, 0.097`. Test 2: a covariance whose thresholded form is positive definite must give `R_mat` and `scale` bit-equal (`np.array_equal`) to the thresholded recipe, and different from the untouched covariance's.
3. **W3-34.** The masking could not be observed through `warp_input` after commit 2: a correlation that is not a number needs a non-positive or non-finite variance, and the guard of commit 2 then keeps the covariance whichever rule the mask follows. Rather than write a test against an unreachable path, I moved the two lines into `_drop_low_correlations(vp_cov, corr_thresh)`, which states the rule, and tested it directly on P8-16's example (a negative variance, whose off-diagonals MATLAB zeroes and the negated comparison kept) plus an ordinary covariance and the exclusive boundary. **This is the one place where I went a little beyond "one comparison"; flagging it for review.**
4. **W3-27.** `utils/unscent_warp.m:18-35` works in double precision and reshapes on `N > 1`. Tests: under the identity warp the transform returns the mean and the scales it was given, so an integer mean must come back exactly (before: `[[0.8, 1.8], [2.8, 3.8]]` with scales 0.4472); and one row of `x` against three rows of `sigma` must give `(3,D)`, `(3,D)`, `(5,3,D)` (before: `ValueError: cannot reshape array of size 6 into shape (1,2)`). The three in-package call sites keep their shapes: `(N,D)`-against-`(D,)` and `(K,D)`-against-`(K,D)` give `N > 1` and are untouched; `(D,)`-against-`(D,)` gives `N == 1` and still returns `(D,)`.
5. **W3-21.** The branch for a new row puts the test around the assignment; `misc/funlogger_vbmc.m:187` passes a time of 0 from `'add'`. Test: a row recorded with a time of 4.0, then a repeat with `np.nan`, must leave `fun_eval_time[0] == 4.0` and the total at 4.0. Before: `array([nan])`.
6. **W3-22.** Both docstrings say `float`; `misc/funlogger_vbmc.m:243`, `:269` return a scalar from both branches. Test: `add` on a new point and on a repeat must both return exactly `float`. Before: `numpy.float64` for a new point and a `(1,)` array for a repeat. Every consumer listed in P8-4 was re-read: `active_sample.py:696`/`:808` (`np.array([[ynew]])`, same float64 `(1,1)` array), `active_sample.py:216`/`:218` and `vbmc.py:801` (discarded), `batch_call:462` (`np.asarray(f_val).item()`).
7. **W3-32.** Test: the three disagreeing pairs raise, the three agreeing pairs build. I scanned every `FunctionLogger(...)` call in the repository with an AST walk: all literal ones already agree; the three non-literal ones are `vbmc.py:462` (`noise_flag = level > 0`), `pyvbmc/testing/oracles/_state.py:358` (reads the fixture's stored pair — I read all eight fixture JSONs: seven are `False/0`, `rosenbrock_D2_noise1_viqr` is `True/2`), and one script under `dev/` (below). **No test line in the suite needed correcting.**
8. **W3-28 (add).** Test: at level 2, `add` without an SD raises and records nothing; with one it records it. `test_add_no_f_sd` covers level 1 and is untouched.
9. **W3-28 (construction).** Written inline in `_init_optim_state`, where `f_vals` is first read, seven lines above the `MismatchedStartingInputs` check and far from `_validate_noise_shaping_option`. Every `add` call site was checked at level 2: `vbmc.py:801` always passes `supplied_sd` (a level-2 run without `y_sd` is already refused at `vbmc.py:722`); `active_sample.py:218`, `:698` and `function_logger.py:461` take their values from `optim_state["cache"]["y_orig"]`, which is finite only from `f_vals` — now impossible at level 2 — and the `precomputed_evaluations` rows are marked `skip_logger` rather than replayed. **No change to `active_sample.py` is needed.**
10. **W3-23.** `misc/boundscheck_vbmc.m:138-143` rejects the same bounds; the log transforms (types 1 and 2 of `shared/warpvars_vbmc.m`) were never ported. I confirmed no in-package construction is affected: `vbmc.py:414` (bounds validated by `_normalize_bounds`, `vbmc:HalfBounds`), `variational_posterior.py:156` (unbounded), `calibration/_campaign.py:149` (fixed ±2 box); `pyvbmc/pymc/` and `pyvbmc/svbmc/` construct none, and the PyMC adapter's `lb`/`ub` reach `VBMC`, which validates them. An AST scan of the package found no other construction mixing a finite with an infinite bound.
11. **W3-26.** Test: a valid scale is stored; seven invalid ones (wrong length, `(1,D)`, `inf`, `nan`, `0`, negative, complex) each raise naming `` `scale` ``. The oracle rebuild assigns `scale` after construction, so it is unaffected.
12. **W3-24 (copies).** Test: after construction, changing the caller's `lb_orig`, `ub_orig`, `scale` and `rotation_matrix` must leave both the stored values and the transform of a fixed point unchanged. Before: the first coordinate moved from 0.2554 to −0.0010.
13. **W3-24 (docstring).** Test: the names the class docstring documents equal the constructor's parameters. Before: `bounded_transform_type` documented, `transform_type` in the signature.
14. **W3-25.** The PI's ruling. Test 1 (the contract, control case): without a rotation or a rescaling the plausible box maps to ∓0.5 in every coordinate. Test 2: with a rotation and a rescaling the transform must equal `(plain(x) @ R) / scale`. Before: the two differed by up to a factor of four. `test_init_type3_mu_all_params` and `test_init_type3_delta_all_params` build without a rotation or a scale, so they are unaffected and were left as they are.
15. **W3-31.** Documentation only, as ruled.

## Test commands run, and results

All from the worktree root with `-q -p no:cacheprovider` and the three thread variables:

- `pyvbmc/testing/whitening/test_rotoscaling.py` — **15 passed**
- `pyvbmc/testing/function_logger/test_function_logger.py` — **59 passed**
- `pyvbmc/testing/parameter_transformer` — **249 passed, 18 skipped**
- `pyvbmc/testing/decorators` — **14 passed**
- `pyvbmc/testing/vbmc/test_vbmc_cached_values_noise.py` — **3 passed**

Two single test ids outside my area were run to establish facts for this report: `test_vbmc_active_sample.py::test_vectorized_initial_design_matches_scalar` (see below) and `test_vbmc_init.py::test_vbmc_setupvars_f_vals` together with `test_vbmc_active_sample.py::test_vectorized_initial_design_all_values_cached` (both **pass**, the other two readers of the `f_vals` line I edited). No whole test directory beyond the four of my area, no `optimize()` test, no oracle or golden command, no install.

## Changelog sentences (for a user of release 1.0.4), and whether a script can stop

Can stop a script written for 1.0.4 (each deserves an "Upgrading from" line):

- `c7aa724b` — Building a `FunctionLogger` with a noise flag that contradicts its uncertainty handling level now raises; the flag is true exactly above level 0. **Stops a script** that built such a logger.
- `fdd888b9` — `FunctionLogger.add` now requires the SD of the value when the target provides its own noise, instead of recording an SD of 1. **Stops a script** that added values to a level-2 logger without one.
- `dfa49bb6` — The `f_vals` option is now refused together with `specify_target_noise`, since it carries no noise for the values it supplies; pass such observations through the `precomputed_evaluations` argument. **Stops a script** that combined them (which was recording an invented noise of 1).
- `e32b8044` — `ParameterTransformer` now refuses a variable with one finite and one infinite bound instead of treating it as unbounded. **Stops a script** that built one directly; `VBMC` rejected such bounds already.
- `e7db7dae` — `ParameterTransformer` now requires `scale` to have one finite positive entry per dimension. **Stops a script** that passed a zero, negative or wrongly shaped scale.
- `9a368ba8` — The function logger now returns the recorded value as a float for a repeated point as well as for a new one, where a repeat gave a one-element array. **Changes what it returns.**
- `3a766567` — `unscent_warp` now works in floating point whatever the dtype of the mean it is given, and returns one row per row of the broadcast of its two arguments, so a single mean against several scales works instead of raising. **Changes what it returns** for an array of integer means (which was truncated).
- `c5fcb50a` — A `ParameterTransformer` built with a rotation or a rescaling and with plausible bounds inside its hard bounds now centres the plausible box on `[-0.5, 0.5]` as one built without them does. **Changes what it returns** for such a transformer.

User-noticeable, cannot stop a script:

- `92abfa31` — After an input warp, the function logger's transformed-space points and values are re-expressed for every recorded evaluation, including rows dropped at the end of warm-up, which kept the coordinates of the space the run had left.
- `06922a1b` — The whitening transform keeps the posterior covariance as it is when dropping its weakly correlated entries would leave a matrix that is not positive definite, which could otherwise leave the posterior far from unit variance along one direction.
- `f10115e4` — Recording a repeated evaluation whose duration is unknown no longer replaces the stored average duration of that point with NaN.
- `29691404` — `ParameterTransformer` keeps copies of the bound, scale and rotation arrays it is given, so changing one of them afterwards no longer changes the transform.
- `ace07a9a` — The `ParameterTransformer` documentation names the bounded-transform keyword `transform_type`, which is what the constructor takes.

No changelog line needed: `0c131f69` (unreachable in practice) and `f53e8eb3` (documentation of behaviour that does not change).

## What I stopped on

**One test outside my area now fails, and I did not change it.** `pyvbmc/testing/vbmc/test_vbmc_active_sample.py::test_vectorized_initial_design_matches_scalar` builds a `VBMC` with `{"f_vals": [2 * D, np.nan], "specify_target_noise": noisy}`, so its two `noisy=True` parametrizations (`[True-1]`, `[True-2]`) now raise at construction from commit `dfa49bb6`. I confirmed it by running that one test id: **2 failed (`[True-1]`, `[True-2]`), 2 passed (`[False-1]`, `[False-2]`)**. The combination is exactly what the PI ruled to refuse, and the test uses it incidentally — to give the noisy arm one cached value so that `cache_count == 1` and `func_count == 4` in both arms — rather than to assert the old behaviour, so correcting it is a small redesign of that test and belongs to its owner. The candidates I see: drop `f_vals` for the noisy arm and let it assert `cache_count == 0` / `func_count == 5`, or split the cached-value comparison off into its own noiseless test and give the noisy arm its own counts. Note that commit `fdd888b9` alone would already have made that arm fail, one step later, inside `add`.

Nothing else was left undone; all thirteen findings are committed.

## What else I noticed

- **`handle_0D_1D_input`'s docstring names its parameters `kwarg` and `argpos`** where the signature has `patched_kwargs` and `patched_argpos`, and misspells "handled" twice. Not in the ledger and not covered by the W3-31 ruling, so I left them, following the wave-2 precedent of flagging a neighbouring docstring defect rather than fixing it. It is the same kind of defect as W3-24's `bounded_transform_type`, one line away from my edit.
- **The `f_vals` option's description** in `advanced_vbmc_options.ini` ("Evaluated fcn values at X0") says nothing about the absence of a noise channel, which is what commit `dfa49bb6` now refuses over. A line there would tell the user before the error does; `option_configs/` was not in my area and another agent works next door in `options.py`.
- **`dev/experiments/port_review_20260919/verification/scripts/wave3_P8_logger.py:32`** builds `FunctionLogger(fun, 2, noise, level)` with deliberately disagreeing pairs to demonstrate P8-14; it will now raise on those calls. It is under `dev/`, so I did not touch it.
- **The `corr_D5_warped` oracle fixture** stores `logger/X` and `logger/y` for its five inactive rows in the space the run had left, as P8-1 records. Nothing recomputes them — `_state.build_logger` assigns the stored arrays — so commit `92abfa31` moves no oracle reference, but a future regeneration of the fixtures will change those five rows. I ran no oracle command.
- **`pyvbmc/testing/whitening/` still has no `__init__.py`**, which `AGENTS.md` already records; I added tests to it without adding one.
- The three tests in `test_rotoscaling.py` and the two in `test_parameter_transformer.py` that draw from the unseeded global generator (`test_rotoscaling_rotation_2d`, `test_bounded_log_abs_det_jacobian_numerically`, `test_transform_bounded_and_unbounded`), which `wave3.md` lists under "test notes worth acting on", were left as they are: no wave-3 row asks for them. All the tests I added are deterministic and make no draws.
