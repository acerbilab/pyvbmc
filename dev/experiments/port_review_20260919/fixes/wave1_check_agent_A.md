Raw report of fix agent A of the independent check of the wave-1 pass (2026-09-22), on the branch w1check-agent-A cut at 326c7676. Saved verbatim from the agent's transcript; its commits were cherry-picked onto dev-port-review-w1check.

# Wave-1 check, fix agent A

All the findings are implemented as 7 commits on `w1check-agent-A`, in `C:\Users\luigi\Documents\GitHub\pyvbmc-w1check-A`, on top of `326c7676`. The worktree is clean and nothing was pushed. Every new test was run against the code before its fix, and every test that could fail there did. Three test cases cannot fail on the old code, for the reasons given under A1. There is one point where I read the ruling beyond its words: the fallback for "every ELCBO is NaN" in the ranking branch (A2).

## Commits

**fa6a4be9 `fix(vbmc): optimize_vp selects only among the evaluated full ELCBOs`** (A1)
- `optimize_vp` now tracks which slots were evaluated in a local `evaluated` mask, set after each `_eval_full_elcbo` call in the loop.
- The final selection runs over `np.flatnonzero(evaluated & ~np.isnan(nelcbo))` and takes the first smallest value. If no slot qualifies it raises `ValueError`, with the existing message.
- The midpoint takes `np.nanargmin(f_val_lst)`, or index 0 when every value is NaN.
- The docstring gains a Raises section: all evaluations NaN, or `stochastic_optimizer` not `"adam"`.
- I first tried `nanargmin(np.where(evaluated, nelcbo, nan))`. It still picks an empty slot when every evaluated slot is `+inf`, because `nanargmin` treats NaN as `+inf`. That is why the fix builds an explicit list of candidate indices.
- **Unchanged without NaN:** 200,000 random NaN-free cases went through the old and new rules.
  - The midpoint index never differs.
  - The final index differs in only 4,164 cases. In every one, every evaluated slot is `+inf` and the old rule picked an empty slot with NaN parameters.
  - With finite values the index never differs.

**b6bce940 `fix(vbmc): determine_best_vp ranks a NaN score last and skips it`** (A2)
- The ranking branch now reads `elbo`, `elbo_sd` and `r_index` as `dtype=float` and keeps the stable argsorts. NumPy sorts NaN last in both rankings.
- The look-back branch converts its scores to float and uses `nanargmax`.
- When every candidate ELCBO is NaN, the selection returns `max_idx`, and the new Notes section of the docstring says so.
- The misleading `asarray` comment is corrected.

**7d37148d `fix(vbmc): determine_best_vp defaults to the run's selection options`** (A3)
- `safe_sd`, `frac_back` and `rank_criterion_flag` now default to `None`, which resolves to `self.options["best_safe_sd"]`, `["best_frac_back"]` and `["rank_criterion"]`. An explicit argument still wins.
- Docstring updated. Both call sites in `optimize` still pass the options explicitly, and `test_determine_best_vp_receives_the_option_values` still asserts that.

**c87dc121 `test(vbmc): the global-stream fixture describes the seeded mixture tests`** (A4.1)
- The fixture docstring now says the following:
  - The two `*_g_mixture` tests seed the global stream in their bodies, and the restore keeps those seeds from reaching later tests.
  - Advancing the stream by the attempt number only affects tests that draw from the stream unseeded.
- It also no longer claims the `_grad_fd` fixture is the same: that one restores the state but does not advance it.

**9a7e23a6 `docs(vbmc): the sieve's single candidate and when var_ss is zero`** (A4.2 and A4.3, grouped because both are docstrings in one file)
- `_sieve` returns arrays of shape `(max(init_N, 1),)`. With `init_N = 0` the given VP is the single type-1 candidate, so `fast_opts_N = 0` requires `slow_opts_N = 1`, as in MATLAB (`misc/vpsieve_vbmc.m:84-86`). The `fast_opts_N` entry of the `optimize_vp` docstring now says the same.
- `var_ss` and `varG_ss` are also zero when `compute_var` is False, and in `_gp_log_joint` when `avg_flag` is False.

**0eec914e `fix(variational_posterior): a zero weight sets eta without a warning`** (A4.4)
- The `eta = log(w)` line now runs under `np.errstate(divide="ignore")`. No value changes.

**b8ce6749 `docs(vbmc): when best_frac_back acts and how adaptive_k is called`** (A4.5)
- The two option descriptions in `advanced_vbmc_options.ini` are updated as asked.

## Tests that fail before the fix

- **A1** (`pyvbmc/testing/vbmc/test_variational_optimization.py`):
  - `test_optimize_vp_raises_if_every_deterministic_evaluation_is_nan[1]` and `[2]`: `Failed: DID NOT RAISE ValueError`.
  - `test_optimize_vp_raises_if_every_evaluation_is_nan_without_midpoints`: `DID NOT RAISE ValueError`.
  - `test_optimize_vp_selects_the_best_evaluation_that_is_not_nan[{1: nan, 3: inf} -> 3]`: the returned `mu` was `[[nan],[nan]]`, an empty slot.
  - `test_optimize_vp_midpoint_skips_nan[[3, nan, 1, 2] -> 2]`: the midpoint was taken at the NaN iterate, column 1.
  - Three cases pass on the old code by construction:
    - the two mixed NaN/finite parametrizations of the selection test, because a finite value always beats an empty slot's `+inf`;
    - the all-NaN midpoint case, because `np.argmin` also gives 0 there.

    They pin the contract.
- **A2** (`pyvbmc/testing/vbmc/test_vbmc_determine_best_vp.py`), all four fail:
  - `test_nan_elcbo_ranks_last`: `assert 0 == 2` for elbo `[1, nan, 2]`.
  - `test_nan_reliability_index_ranks_last`: `assert 2 == 4`.
  - `test_look_back_skips_nan_elcbo`: `assert 1 == 3`, the window opening on a NaN.
  - `test_every_elcbo_nan_selects_the_last_iteration`: `assert 1 == 3`.
- **A3:** `test_determine_best_vp_defaults_to_the_run_options` fails with `assert 3 == 0` at the default options.
- **A4.4:** `test_set_parameters_zero_weight[True]` and `[False]` fail with `RuntimeWarning: divide by zero encountered in log`.

## Tests run after the fixes, one process at a time, all pass

- **A1:**
  - the new tests (8 cases);
  - `test_optimize_vp_passes_over_a_nan_evaluation`, `test_vp_optimize_one_component_exact_entropy`, `test_optimize_vp_returns_eta_matching_weights`, `test_optimize_vp_without_shotgun_evaluation`, `test_optimize_vp_takes_an_unconverged_iterate`, `test_optimize_vp_preserves_transformer_through_pruning` (6 cases);
  - `test_vp_optimize_1D_g_mixture`, `test_vp_optimize_2D_g_mixture`, `test_vp_optimize_deterministic_entropy_approximation`.
  - The reviewer's `chk_nan.py` now raises `ValueError` in all four cases. `chk_midpoint_nan.py` returns finite parameters with ELBO -9.6917, the value of the reviewer's simulation of MATLAB's `min`.
- **A2 and A3:**
  - every test function of `test_vbmc_determine_best_vp.py`, named one by one;
  - `test_vbmc_loop_state.py::test_a_closing_line_reports_a_posterior_from_an_earlier_iteration` and `::test_the_closing_line_leaves_the_random_stream_of_an_unboosted_run`.
  - The reviewer's `best_compare.py` gives 0 of 19,520 NaN-free mismatches with `best_vbmc.m`. With NaN it gives 516 of 6,400 (it was 1,048 before the fix), all of them in the ranking branch with a NaN ELCBO.
  - My check (`scratchpad/w1check_A/a2_ruled.py`) compares 24,000 cases against a transcription with the ruled NaN handling and finds 0 mismatches. All 2,166 mismatches with `best_vbmc.m` are cases where MATLAB selects an iteration whose ELCBO is NaN.
- **A4.2 and A4.3:** `test_optimize_vp_without_shotgun_evaluation`, `test_sieve_orders_tied_candidates_stably`.
- **A4.4:** the new test, `test_set_parameters_eta_matches_weights` (4 cases), `_raw`, `_not_raw`, `_not_raw_negative_error`, `_not_raw_checks_the_constrained_entries`, `_reference_regression`, and the four `test_get_set_parameters_roundtrip*` / `_delete_mode` tests.
- **A4.5:** `test_options.py::test_shipped_descriptions_are_stored_in_full`, `::test_description_keeps_the_whole_comment_line`, `test_variational_optimization.py::test_update_K_callable_adaptive_k`.

I did not run the oracle gates, the golden replay or whole test files; those are yours. No oracle calls `optimize_vp` or `determine_best_vp`.

## Things to check or decide

- **My reading of the A2 fallback.** The ruling asks to select the last iteration when every candidate in the window is NaN. I applied that to the ranking branch as well, where the candidates are all iterations up to `max_idx`. Without it, NaN-last with stable ties makes the ELCBO rank exactly cancel the position rank when every ELCBO is NaN, which favours the earliest iteration. It is one `if np.all(np.isnan(elcbo))` in the ranking branch, plus the ranking half of `test_every_elcbo_nan_selects_the_last_iteration`; revert both if you want the ruling read narrowly.
- **A test now runs a different branch.** After A3, `test_best_safe_sd_changes_the_selection` calls without `rank_criterion_flag`, so it runs the ranking branch. It gives the same answers, 5 and 4. Pin `rank_criterion_flag=False` there if its intent was the look-back.
- **Same empty-slot defect in MATLAB.** `misc/vpoptimize_vbmc.m:176` has the defect A1 fixed; a note is proposed below.
- **Warning left in `get_parameters`.** `vp.get_parameters(raw_flag=True)` with a zero weight still warns (`log(0)`, `variational_posterior.py` around line 1155). This was outside A4.4, so I left it.
- **Stale source hashes.** `dev/scripts/torch_vi_step.py` `_EXPECTED_SOURCE_HASHES` were already stale for all six functions at `326c7676`. My commits change the source of `optimize_vp` and `_sieve` further.

## Proposed text for the records

**Changelog, A1 (Fixed):**
> The variational optimization passes over NaN. The best iterate of a stochastic optimization is taken among the objective values that are not NaN, as in MATLAB VBMC, and a candidate solution whose ELBO is NaN is not selected. `optimize_vp` raises `ValueError` if the ELBO of every candidate is NaN. 1.0.4 took the first NaN it met in both places and could go on with a posterior whose ELBO was NaN.

Upgrading line:
> - A run in which every candidate of a variational optimization has a NaN ELBO raises `ValueError`, where 1.0.4 went on with a posterior whose ELBO was NaN.

**Changelog, A2** (Changed, after "So are iterations with equal scores when the best posterior of a run is selected. 1.0.4 ordered them arbitrarily."):
> An iteration whose ELCBO or reliability index is NaN ranks last in that selection, and without the ranking criterion an iteration whose ELCBO is NaN is passed over. If the ELCBO of every candidate iteration is NaN, the posterior of the last iteration considered is returned. 1.0.4 could return a posterior whose ELBO was NaN.

**Changelog, A3** (extends "That choice follows the options `rank_criterion`, `best_safe_sd` and `best_frac_back`, which 1.0.4 ignored."):
> `VBMC.determine_best_vp()`, called without `safe_sd`, `frac_back` or `rank_criterion_flag`, follows these options too, so on a finished run it returns the iteration the run selected. In 1.0.4 these arguments defaulted to 5, 0.25 and no ranking criterion. Called directly on a run whose last iteration is not stable, it can therefore return another iteration than in 1.0.4. The ranking criterion is on by default. Without it, when no iteration was stable, the look-back window covers `ceil(n * best_frac_back)` iterations before the last of `n`, up to two more than 1.0.4 counted (four instead of three with ten iterations and the default quarter).

Upgrading line:
> - `vbmc.determine_best_vp()` without arguments follows the options `rank_criterion`, `best_safe_sd` and `best_frac_back`, with the ranking criterion on by default, and can return another iteration than in 1.0.4 when the last iteration is not stable.

A4.4 needs no entry, because 1.0.4 never set `eta` in `set_parameters`. A4.1, A4.2, A4.3 and A4.5 are documentation only.

**Known-differences sheet, A2** (slice P1a):
> ### A NaN score ranks last in the selection of the returned posterior
> - Python: `pyvbmc/vbmc/vbmc.py`, `determine_best_vp`: the scores are read as floats, the ELCBO and reliability rankings are stable `argsort`s, the look-back uses `np.nanargmax`, and the fallback is `max_idx`.
> - MATLAB: `misc/best_vbmc.m:36` (`sort(elcbo,'descend')`), `:40` (`sort(stats.rindex(1:idx),'ascend')`), `:66` (`max(elcbo)`).
> - What differs: MATLAB's `sort` places NaN first in a descending sort and last in an ascending one. `max` skips NaN and returns the first index when every value is NaN. So MATLAB ranks an iteration with a NaN ELCBO first on ELCBO and can return a posterior whose ELBO or ELBO SD is NaN. Without the ranking criterion it returns the first iteration of a look-back window whose ELCBOs are all NaN. PyVBMC ranks a NaN ELCBO last; the reliability ranking agrees with MATLAB. When every candidate ELCBO is NaN, PyVBMC returns `max_idx`, the posterior the run ended on; the candidates are every iteration up to `max_idx` with the ranking criterion and the look-back window without it. Without NaN the two select the same iteration: 0 of 19,520 constructed histories differ from a transcription of `best_vbmc.m`. With NaN, every difference found (2,166 of 24,000 cases) is one where `best_vbmc.m` selects an iteration whose ELCBO is NaN.
> - Why: PI ruling on the wave-1 check, 2026-09-22, finding A2.
> - Kind: deliberate change.

**MATLAB-side defects** (next numbers after 53; read, not run):
- `misc/best_vbmc.m:36`, with `:66`
  - **What the code does:** `[~,ord] = sort(elcbo,'descend')` uses MATLAB's default NaN placement, which puts NaN first in a descending sort. This rests on MATLAB's documented semantics and was not run.
  - **Consequence:** with `RankCriterion` on (`vbmc.m:201`, the default), an iteration whose ELBO or ELBO SD is NaN takes the best ELCBO rank and can be returned as the best posterior. `max(elcbo)` at `:66` returns the first iteration of an all-NaN look-back window.
  - **PyVBMC:** not shared. A NaN ELCBO ranks last, with the fallback to the last iteration (sheet entry above).
  - **Record:** wave-1 check, A2.
- `misc/vpoptimize_vbmc.m:176`, with `:264`, `:271-272`, `:189`
  - **What the code does:** `[~,idx] = min(elbostats.nelcbo)` runs over every slot. That includes the midpoint slots no evaluation writes when `NSentK == 0` or `ELCBOmidpoint` is off, which keep `nelcbo = Inf` and `theta = NaN`. `min` skips NaN.
  - **Consequence:** when every evaluated slot is NaN, `min` returns an empty slot. `rescale_params` at `:189` then gets a theta of NaN, with `elbo = -Inf` and `elbo_sd = NaN`. The NaN ELCBO this records is exactly what the previous row ranks first.
  - **PyVBMC:** not shared since the wave-1 check, A1. The selection runs over evaluated slots only and raises when all of them are NaN.
  - **Record:** wave-1 check, A1.
