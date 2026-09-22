Raw report of fix agent B of the independent check of the wave-1 pass (2026-09-22), on the branch w1check-agent-B cut at 326c7676. Saved verbatim from the agent's transcript; its commits were cherry-picked onto dev-port-review-w1check.

# Wave-1 check, fix agent B

I made 12 commits on `w1check-agent-B` in `C:\Users\luigi\Documents\GitHub\pyvbmc-w1check-B`, on top of `326c7676`. Nothing is pushed. No commit has a `Claude-Session:` trailer; every commit ends with the `Co-Authored-By: Claude Opus 5.5 (1M context)` line. I did not touch `CHANGELOG.md` or anything under `dev/`. Every new test failed on the code before its fix, except where noted.

Two places depart from the letter of a ruling, and the PI should look at them: B1 (how the cached point's identity is carried) and B4 (the case the count is below `fun_eval_start`). Details are under "Decisions to review".

## Commits

**`a56d6f17` fix(active_sample): recognize a cached starting point after a warp (B1)**
- **What changed:**
  - `_get_search_points` takes an optional `cache_rows` dict. It fills it with the rows it made of the cached points, keyed by cache index, before the clip into the search box.
  - `active_sample` compares the acquired candidate exactly with that row, and no longer with a new single-row transform of the point.
  - The comment at the site says why.
  - Without a starting cache nothing changes. Wherever the transform is batch-invariant (every run without a warp) the decision is exactly the one made before.
- **Why the optional argument:** it keeps the two return values of `_get_search_points`. That leaves working the test mocks, `dev/scripts/make_oracle_fixtures.py:372` and the wrapper in `dev/scripts/noisy_acq_search.py`, which unpacks two values.
- **Test:** `test_a_cached_point_nothing_moved_keeps_its_value_after_a_warp`, with a helper `_warped_state_with_gp` (a real `warp_input` / `warp_gp_and_vp` from a correlated posterior, then a GP refit).
  - Before the fix: `assert 11 == 10` (the target was called). On this machine 5 of 7 cached rows differ from their single-row transform.
  - The second half checks that a clipped cached row is evaluated at the clipped point.

**`39742840` fix(active_sample): refresh the training-set counts after each evaluation (B2)**
- **What changed:**
  - A helper `_refresh_training_counts` (same expressions as before) now runs at the top of each acquisition, after each evaluation in the loop, after the initial design, and in `VBMC.optimize` after active sampling. That last call covers an iteration that acquires no point after the warm-up trim.
  - The commented refresh in `FunctionLogger.__call__` is replaced by a note saying where the refresh happens.
  - The commit message states the readers correctly and says that `optimize_vp` reads no `n_eff`, contrary to `118626e8`. One test docstring that repeated that claim is corrected.
- **Test:** `test_in_loop_gp_refit_reads_the_counts_after_the_evaluation`. On a noisy target with the in-loop GP update on, the first acquisition repeats a training input and the second is a fresh point.
  - Before the fix: `(10, 10) != (10, 11)`.

**`fea56ec1` fix(options): register gp_int_mean_fun and proposal_fcn as having no effect (B3)**
- **What changed:**
  - Both options are in `INERT_OPTIONS`.
  - Their descriptions read: "(not used, the integrated mean function of the GP is not ported)" and "(not used, MATLAB VBMC stores the option and never reads it)".
  - The two dead copies into `optim_state` are removed.
  - Both names leave `_CONSTRUCTION_ONLY_OPTIONS`.
  - The two `test_vbmc_init.py` tests that asserted the copies are removed.
- **Grep:** no reader of either key remains in `pyvbmc/`, `load` or the oracle harness. The oracle fixtures' JSON still hold the keys, and `--check` compares outputs only, so this does not matter.
- **Saved runs:** `test_vbmc_save_static.pkl` holds both keys and still loads (`test_vbmc_load_static`).
- **Tests:** `test_the_integrated_mean_and_the_proposal_function_are_inert` (no warning before) and `test_load_takes_an_option_that_has_no_effect` (before: `ValueError: VBMC.load cannot change the option ...`).

**`18887832` fix(gaussian_process_train): a budget equal to the initial design fits (B4)**
- **What changed:** a zero span sets `x = NaN`, and a NaN schedule gives `init_N = 0`, as MATLAB's `max` ignoring NaN makes it (`misc/get_GPTrainOptions.m:98-100`). The budget-accounted branch is unchanged.
- **Test:** `test_a_schedule_without_span_gives_no_space_filling_points`, with D = 10 and `max_fun_evals=20`, at four counts. All raised before:
  - count equal to `fun_eval_start`, NumPy integer: `ValueError: cannot convert float NaN`;
  - count equal to `fun_eval_start`, Python integer: `ZeroDivisionError`;
  - count past it: `ValueError`;
  - count short of it: `OverflowError`.

**`620c728c` fix(active_sample): read ns_elbo and ns_ent_fine_active through Options.eval (B5)**
- **What changed:** both options are read through `options.eval(..., {"K": ...})`. No direct option call remains in the package.
- **Test:** `test_a_scalar_count_option_reaches_the_in_loop_update`. It gives each option the number that the shipped function gives at K = 2.
  - Before the fix: `TypeError: 'int' object is not callable`.

**`2d64fc6a` fix(vbmc): check warp_cov_reg at construction and what its function returns (B6)**
- **Construction and `load`:** `_validate_warp_cov_reg_option`, called from `_validate_option_values`, accepts a callable or a finite real number. That means a Python or NumPy int or float, or a 0-d array of one; booleans are refused. Values outside [0, 1] are still accepted. The message names the option and the load remedy.
- **At the warp:** a function's result must be a finite real number, and a one-element array counts as the number it holds. Otherwise a `ValueError` names the option, N and the returned value. A value written into built options is checked there too.
- **Shared helpers:** `_is_finite_real_number` and `_WARP_COV_REG_REFUSAL` in `whitening.py`.
- **Description:** now says the value is clamped to [0, 1] and may be a function of N, the number of distinct points logged so far (`optim_state["N"]`).
- **Tests** (22 new cases failed before the fix):
  - In `test_vbmc_option_names.py`: refusal of None, str, complex, bool, `np.True_`, nan, inf and a list; acceptance of the finite forms, a 0-d array, values outside [0, 1] and a function; the NaN case added to the load-equals-construction test; `True` added to the remedy test.
  - In `test_rotoscaling.py`: a function's bad results, and bad values written into built options.
  - `test_warp_input_cov_reg` now also covers a function that returns a one-element array.

**`338d61bf` fix(active_sample): score a one-component posterior by its exact entropy (B7)**
- **What changed:** when `vp0.K == 1`, `NSentFineK = 0`. That sends `_neg_elcbo` to the deterministic entropy, which is exact for one Gaussian, as `_eval_full_elcbo` does. K > 1 is unchanged. No helper was needed in `variational_optimization.py`.
- **Test:** `test_in_loop_comparison_scores_one_component_with_its_exact_entropy`. It spies on `_neg_elcbo` and checks both the sample count and the closed-form entropy.
  - Before the fix: `assert 200 == 0`.

**`6ed3748e` fix(vbmc): load warns of an option without effect, as construction does (B8)**
- **What changed:**
  - `Options._warn_inert_options(options_paths, names=None)`: `names` defaults to `useroptions`.
  - `load` calls it on the names of `new_options`, after the value checks and refusals pass. The warning and the rule for callable defaults are those of construction, and only the names given to load are weighed.
  - The names also join `useroptions`; the reason is under "Decisions to review".
  - The `load` docstring mentions the warning.
- **Tests:**
  - `test_load_warns_of_an_option_that_has_no_effect` (no warning before);
  - the B3 load test, extended to require the warning (failed before);
  - `test_load_warns_only_of_the_options_it_is_given`, which passes either way and guards against warning again for construction values.

**`83182bd6` fix(vbmc): the noise_shaping refusal names the remedy for a saved run (B9)**
- **Test:** `test_a_saved_run_that_carries_noise_shaping_is_loaded_as_the_refusal_says`. The message lacked the remedy before; the test also loads the run with the remedy.

**`3862fe05` test(active_sample): check the rank-one noise variance and seed the GP fit (B10.1, B10.2)**
- **Rank-one test:** `_noisy_run` takes `noise_sd` and `uncertainty_level`; its defaults are unchanged. The rank-one test runs at levels 1 and 2 with SD 0.3 and asserts `s2_new == S**2` (0.09 at level 2; 1 at level 1, where the logger records SD 1).
  - A mutation check (passing the SD instead of the variance) fails the level-2 case.
- **Seeding:** `test_active_uncertainty_sampling` passes `rng=vbmc.vp.rng` to `train_gp`, and its comment is corrected. It passed in 3 fresh processes.

**`b56eb0d7` test(active_sample): cover the branches of the local acquisition search (B10.3)**
- **New tests:**
  - `test_one_dimensional_search_failure_keeps_the_sieve_point_and_index`: a real starting cache, the stored value used, the row gone, the warning logged.
  - `test_cmaes_search_with_a_zero_scale_starts_isotropic`: the real cma runs, no `CMA_stds` is passed, sigma0 is the largest scale.
- **Strengthened tests:** the fallback-bounds test asserts MATLAB's formula exactly (`private/activesample_vbmc.m:253-254`), and the 1-D test asserts `[min(x0, lb_search), max(x0, ub_search)]`.
- These cover code that is already fixed. Each test fails under a matching mutation: the try removed, the zero guard dropped, 0.2 in place of 0.1 times the range, the lower end shifted.

**`54ce66e9` docs(active_sample): say what the search and the surplus starting points do (B11)**
- The four comments or docstrings are corrected: cma's `eval_final_mean` call; SciPy's bounded method as a local search bracketed by the interval; the zero-entry versus non-finite-entry behaviour of the `CMA_stds` guard; and the stored value only when `f_vals` supplied one.

## Tests run (individual functions, one pytest process at a time)

- **Each new test, before and after its fix.**
- **Around the changed functions:**
  - the cached-point tests (reuse, integer grid, without value, box-moved, grid-moved);
  - the `_get_search_points` tests (all-cache, cache share, search bounds, sieve counts);
  - two steps with a search cache; more initial points provided; vectorized initial design; initial-sample tests;
  - rollback preserves the transformer; the refresh-`n_eff` test; `ns_gp_max_active`; the timer and repository tests;
  - the CMA-ES per-coordinate and no-noise-handling tests; the local-search failure; `test_active_uncertainty_sampling`;
  - the option registry tests (inert set, construction-only set, defaults silent, descriptions);
  - the option-name and load tests; `test_vbmc_load_static`; the larger-budget load test; `test_noise_shaping_on_is_rejected`;
  - `test_warp_input`, `test_rotoscaling_rotation_2d`;
  - the GP-training option tests (samplers, `opts_N`, weighted covariance).
- **Final pass at HEAD:** 31 active-sampling tests passed, then 80 option, load, warp and GP-training tests passed.
- **One slip:** I ran the whole of `pyvbmc/testing/vbmc/test_gp_training_policy.py` once by mistake (34 tests, 1.5 s, all passed).
- **Not run (yours to run):** the module test files, the seeded gate runs and the oracle check.
  - The oracles should not move. Every fixture has an empty starting cache, so B1 cannot change them. `active_sample_step` excludes full updates, so B2, B5 and B7 cannot.
  - Noisy seeded runs will move through B2, and through B7 when K = 1.

## B2: the readers, and why noiseless default runs are unchanged

- **Readers of `N` / `n_eff`:**
  - `train_gp` reads `N` in `_gp_hyp` (the number of hyperparameter samples and the stop-sampling test) and `n_eff` in `_get_gp_training_options` (the space-filling schedule).
  - `update_K` reads `n_eff` for `k_fun_max`.
  - `warp_input` passes `N` to a callable `warp_cov_reg`.
  - `VBMC.optimize` reads `N` for `stop_sampling`, the running averages, the history record and the warm-up trim check.
  - No acquisition function and not `optimize_vp` read either count.
- **The one in-loop reader:** the GP refit, reached only with `active_sample_gp_update` (the noisy default).
- **Why a noiseless default run is unchanged:**
  - `active_sample_gp_update` and `active_sample_vp_update` are both False, so there is no full update. Between an evaluation and the next refresh the loop runs only `gp.update` or `reupdate_gp`, the acquisition, the search and the bound expansion, and none of these reads the counts.
  - After the loop, `optimize` refreshes with the same expressions before `train_gp`.
  - The new refresh after the initial design writes exactly what `optimize` writes right after it.

## Decisions to review

- **B1, the identity carried.** I carry the sieve's own pre-clip row and compare the final candidate with it. I did not implement the ruling's two flags (a clip-moved mask plus a snap-changed flag).
  - The one comparison covers both the clip and the snap.
  - It is the only form that meets the ruling's other requirement, that the decision be the same as before for every batch-invariant candidate, in all cases. The two-flag form would differ in one corner: an integer coordinate of an on-grid cached point lies outside the search box, the clip moves it, and the snap moves it back exactly. The point is then the cached point, the old code used the stored value, and the two-flag form would call the target.
- **B4, a count short of `fun_eval_start` with a zero span.**
  - MATLAB's value there is +Inf with the shipped options, which then fails in `fminfill` (`rand(Inf, nvars)`). PyVBMC gives 0 in every zero-span case.
  - A zero span occurs at construction with `max_fun_evals == fun_eval_start` (the only fit then has count equal to `fun_eval_start`).
  - Counts past it, and short of it after the warm-up trim, need `fun_eval_start = 1000` with a larger budget, or a budget lowered through `load`.
  - The PI should confirm 0 for the short case.
- **B8, joining `useroptions`.** After construction, `useroptions` drives only the summary that `str(vbmc.options)` and `str(vbmc)` print. `update_defaults` and `load_options_file` read it only at construction, and `init_from_existing_options` is unused in the package. So joining changes only that summary, which then lists the values a continued run was given, and a later save keeps the record. I judged that matches the set's documented meaning, "options set by the user".

## Found outside these findings

- **Shared `useroptions` set.** `Options(..., user_options=<an Options instance>)` copies the source's `useroptions` set object and then mutates it (`options.py:248-251`). Passing an earlier run's options to a new `VBMC` therefore changes the earlier object; R4's `routes2.py` shows it. Not fixed.
- **`_noisy_run` is unseeded.** The test helper builds an unseeded `VBMC` and trains without `rng=`, so its tests run on a fresh stream in each process. They are written to hold on any stream. Not changed.
- **Non-finite CMA-ES scale.** A non-finite entry of `insigma` still leads to a cma start that does not return. The comment now documents it; no guard was added, per the ruling.
- **Warp with integer variables.** The snap's round trip through a rotated transform generally changes an integer coordinate in its last bits, so an on-grid cached point is evaluated rather than given its stored value. This was so before B1 as well.
- **Candidate MATLAB-side defects:**
  - `WarpCovReg = true`: `isnumeric(true)` is false, so MATLAB calls `true(optimState.N)`, builds an N-by-N logical matrix, and fails later. A NaN weight is clamped to 1.
  - The B4 short-count case gives Inf and then an error in `fminfill`.

## Proposed text for the records

**Changelog (Unreleased).** Entries that change an unreleased entry are marked "revise".

- **Changed → results-differ list, new sub-bullet (B2):** "On a noisy target, the refit of the GP between the new points of an iteration counts the point just evaluated, as MATLAB VBMC does; 1.0.4 counted the training set as it stood before that point."
- **Revise "Smaller changes" (B7):** "…the entropy of a posterior with a single component, which is computed exactly, also where the variational update between the new points of an iteration on a noisy target is compared with the posterior from before it;…"
- **Revise the Fixed → Active sampling entry on surplus `x0` (B1):** "When `x0` holds more points than the initial design uses, the others stay available as candidates for later evaluations. One that is acquired where it lies is recorded with its value from `f_vals`, without a call to the target, also after an input warp, and a starting point that has been evaluated is not proposed again. 1.0.4 discarded the surplus points."
- **Fixed, new (B4):** "A run whose `max_fun_evals` equals the size of its initial design raised an error in its first GP fit; with the initial design of `10 * ceil((D + 1) / 10)` points, `max_fun_evals=20` does this for `D` from 10 to 19. The fit now starts without the space-filling design of the GP hyperparameters, as in MATLAB VBMC."
- **Fixed, new (B5):** "A number given for `ns_elbo` or `ns_ent_fine_active`, in place of a function of the number of components, raised `TypeError` in the variational update between the new points of an iteration, which a noisy target runs by default."
- **Revise the `warp_cov_reg` entry (B6):** "`warp_cov_reg` takes a finite number of any numeric type, NumPy's included, or a function of the number of points logged so far; the function may return its number in an array of one element. Any other value (a boolean, NaN, a string) is refused at construction and by `VBMC.load`, and a function's result that is not a finite number is refused at the warp, with a message that names the option. 1.0.4 took only a Python `int` or `float`, read `True` as 1, and failed at the first warp on other values."
- **Revise the "Setting an option that has no effect…" entry (B3, B8):** "…gives a warning that names the option, whether it is given at construction or to `VBMC.load(new_options=...)`. `gp_int_mean_fun` and `proposal_fcn` are among them. Options given to `load` are listed with the user's options (`print(vbmc.options)`)."
- **Revise the "errors that refuse a value … say how a saved run that carries it is loaded" entry (B9):** add `noise_shaping` and `warp_cov_reg` to its list.
- **Upgrading from 1.0.4:** add to the "Option values are checked" line: "…a `warp_cov_reg` that is neither a finite number nor a function (`True`, which 1.0.4 read as 1, among them) raises an error." B2 and B7 fall under the existing "Results differ" line; the other changes cannot stop a 1.0.4 script.

**Known-differences sheet**

- **B1, revise "The stored value of a cached starting point is reused at that point alone":**
  - Python: `pyvbmc/vbmc/active_sample.py:720-737` compares the acquired candidate exactly with the row the sieve made of the cached point before the clip. The row is handed back through `_get_search_points(..., cache_rows=...)` (`:1011`), and a new transform of the point is not used: after a warp a one-row product can round differently from the sieve's many-row one.
  - A candidate that the clip or the snap moved is evaluated; one that the snap returns exactly to the cached point takes the stored value. MATLAB is unchanged; fixed in `a56d6f17`.
- **B2, new Settled non-difference:** "The training-set counts follow every logged evaluation. `N` and `n_eff` are refreshed after each evaluation of active sampling and after the initial design (`pyvbmc/vbmc/active_sample.py: _refresh_training_counts`), as `misc/funlogger_vbmc.m:278-279` refreshes them in MATLAB. The in-loop GP refit reads the counts after the evaluation. Fixed in `39742840`; until then the refit read them one evaluation stale."
- **B4, new entry, deliberate change in a degenerate case:**
  - Python: `pyvbmc/vbmc/gaussian_process_train.py:592-610` gives a zero span of the space-filling schedule (`min(max_fun_evals, 1e3) == fun_eval_start`) `init_N = 0`.
  - MATLAB (`misc/get_GPTrainOptions.m:98-100`) gives 0 at a count equal to or past `fun_eval_start` (NaN, which `max` ignores). At a count short of it, reachable after the warm-up trim with `FunEvalStart = 1000`, it gives `Ninit = Inf` and fails in `fminfill`.
  - PyVBMC gives 0 there too (`18887832`). A candidate `matlab_side_defects.md` entry.
- **B6, new entry, deliberate change:**
  - PyVBMC checks `warp_cov_reg` at construction and in `load` (`VBMC._validate_warp_cov_reg_option`) and a function's result at the warp (`pyvbmc/whitening/whitening.py:262-279`). It refuses booleans and non-finite values, and accepts a one-element array returned by a function.
  - MATLAB (`misc/warp_input_vbmc.m:59-63`) checks nothing: `true` is not numeric and is called as `true(optimState.N)`, and NaN is clamped to 1 by `max(0, min(1, NaN))`.
  - Both clamp a finite value into [0, 1] and pass `optimState.N`. Commit `2d64fc6a`.
- **B7, revise "The full ELCBO evaluation uses the exact entropy of a one-component posterior":** add that `pyvbmc/vbmc/active_sample.py:914-925` scores a one-component `vp0` in the in-loop comparison with the exact entropy as well, where MATLAB samples on both sides (`private/activesample_vbmc.m:525-526`). Commit `338d61bf`.
