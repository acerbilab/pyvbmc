# Wave 2 fix pass, agent A: the loop

Raw report of a fix agent, wave 2 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, in
its own git worktree cut from `dev-port-review` at `2f2bc94`, fixed the
wave-2 findings on the main loop, warm-up, termination, the final boost,
the best-posterior selection, the result fields and `whitening.warp_input`
(W2-10, W2-11, W2-16, W2-6, W2-12, W2-13, B-M2, B-M3, B-M5, B-M9; then
W2-14, W2-1, W2-2, W2-3, W2-5), one finding per commit with a test written
against the contract, running only the focused test files it touched. The
commit hashes in the report are those of the agent's worktree branch; the
orchestrator reviewed each diff and cherry-picked the commits onto
`dev-port-review`, where they carry other hashes (the plan's worklog lists
them, and says what became of the commit for B-M9, whose effect on the
run's random stream the report describes). The text below the rule is the
agent's final message, unedited; `../verification/wave2.md` holds the
ledger and the PI's dispositions.

---

# Wave-2 fix pass, agent A

Worktree: `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-a7adf9bedf8397252`
Branch: `worktree-agent-a7adf9bedf8397252` (cut from `2f2bc94`; nothing pushed)

Setup as instructed: `git merge --ff-only dev-port-review` → `git log -1 --format=%h` printed `2f2bc94`; `python -c "import pyvbmc; print(pyvbmc.__file__)"` from the worktree root printed
`C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-a7adf9bedf8397252\pyvbmc\__init__.py`.
Every Python command ran with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Working tree clean; nothing untracked left in the worktree (`git status --porcelain --ignored`, ignoring `!!`, is empty). The one standalone script I ran (a probe, below) carried `PYTHONPATH=<worktree>` and printed `pyvbmc.__file__`.

## `git log --oneline 2f2bc94..HEAD`

```
8226833 fix(vbmc): hold termination on the count of iterations performed
36f2cc9 fix(vbmc): start the warping clocks when warm-up ends
d2907f5 fix(vbmc): recompute the running maximum of the lower confidence bound
8b948f2 fix(vbmc): measure the warm-up improvement over the intended window
6d34bb1 fix(vbmc): report no stability count when the recent window is empty
0972a14 fix(vbmc): close the display for every posterior that is not the last one
3697685 fix(vbmc): return a copy from the best-posterior selection
9533cdd fix(vbmc): run the final boost on a copy of the optimization state
abc4ab1 fix(vbmc): copy the training inputs into the warp refit's components
aa083b9 fix(vbmc): report the number of iterations performed
70a0bc4 fix(vbmc): report the problem type from the original bounds
1fdd515 fix(whitening): invert the search state with the current inference space
a4d089b fix(vbmc): size the warp refit's sieve at the components it optimizes
db996b2 fix(vbmc): average the recorded variational moments over iterations
cb1fa41 fix(vbmc): read the forced entropy switch from the options
```

Diffstat over the range: 11 files, +920 / −72. Source files touched: `pyvbmc/vbmc/vbmc.py`, `pyvbmc/whitening/whitening.py`, `dev/scripts/benchmark_targets.py`.

## Commits in order

Phase 1 (1–10), phase 2 (11–15).

| # | hash | finding | what it does | files | tests |
|---|---|---|---|---|---|
| 1 | `cb1fa41` | W2-10 + B-M12 | forced entropy switch reads `options["entropy_force_switch"]`; the stability branch that turns the switch off requires that fraction to be finite | `vbmc.py`, `test_vbmc_loop_termination.py` | +`test_entropy_force_switch_is_read_from_the_options`, +`test_stability_termination_with_an_infinite_forced_switch` |
| 2 | `db996b2` | W2-11 | `len(run_cov == 0)` → `len(run_cov) == 0`; the exponentially weighted update runs | `vbmc.py`, new `test_vbmc_loop_state.py` | +`test_running_moments_are_an_exponentially_weighted_average` |
| 3 | `a4d089b` | W2-16 | warp-undo refit sieves `ceil(ns_elbo(Knew))` | `vbmc.py`, new `test_vbmc_warp_branch.py` | +`test_the_run_warps`, +`test_warp_refit_sieve_is_sized_at_the_new_component_count` |
| 4 | `1fdd515` | W2-6 | `warp_input` inverts search bounds and search cache with the function logger's transformer (the current space); the handed posterior's transformer stays on its covariance | `whitening.py`, `test_rotoscaling.py` | +`test_warp_input_inverts_the_search_state_with_the_current_transform`; changed `test_warp_input_search_cache` |
| 5 | `70a0bc4` | W2-12 | `results["problem_type"]` tests `lb_orig`/`ub_orig`; string stays `"bounded"` | `vbmc.py`, new `test_vbmc_result_fields.py` | +`test_problem_type_reports_the_bounds_the_run_was_given` |
| 6 | `aa083b9` | W2-13 | `results["iterations"]` is the count; `optimize` and `_create_result_dict` docstrings state that and that `best_iter` is a 0-based index; benchmark runner drops its `+ 1` | `vbmc.py`, `dev/scripts/benchmark_targets.py`, `test_vbmc_result_fields.py` | +`test_iterations_counts_them_and_best_iter_indexes_them` |
| 7 | `abc4ab1` | B-M2 | warp branch copies `gp.X.T` | `vbmc.py`, `test_vbmc_warp_branch.py` | +`test_warp_refit_gets_its_own_copy_of_the_training_inputs` |
| 8 | `9533cdd` | B-M3 | `final_boost` writes `warmup=False`/`entropy_alpha=0` on a deep copy of `optim_state` and passes that copy to `optimize_vp` | `vbmc.py`, `test_vbmc_finalboost.py` | +`test_final_boost_leaves_the_optimization_state_alone`; changed one assertion in `test_final_boost_guard_selects_posterior` (`captured["optim_state"] is not vbmc.optim_state`) |
| 9 | `3697685` | B-M5 | `determine_best_vp` deep-copies the recorded posterior and writes the stability flag into the copy; the redundant `copy.deepcopy` in the warp branch removed | `vbmc.py`, `test_vbmc_determine_best_vp.py` | +`test_determine_best_vp_returns_a_copy_and_leaves_the_history_alone`; five identity assertions replaced by a shared `assert_copy_of_recorded` helper |
| 10 | `0972a14` | B-M9 | closing "finalize" line printed when `idx_best != iteration` or the boost changed the posterior; its `sKL` compares with the posterior captured after the loop; `Nkl` hoisted above the loop | `vbmc.py`, `test_vbmc_loop_state.py` | +`test_a_closing_line_reports_a_posterior_from_an_earlier_iteration` |
| 11 | `6d34bb1` | W2-14 | an empty recent window leaves `stable_count_flag` False instead of raising | `vbmc.py`, `test_vbmc_loop_termination.py` | +`test_check_warmup_end_conditions_with_a_window_of_one_iteration` (+ helper `_warmup_history`) |
| 12 | `8b948f2` | W2-1 | `recent_past = iteration + 1 - tol_stable_warmup_iters` | `vbmc.py`, `test_vbmc_loop_termination.py` | +`test_warmup_recent_improvement_window_is_the_stability_length` |
| 13 | `d2907f5` | W2-2 | `_recompute_lcb_max` implemented; `_check_warmup_end_conditions` prefers the recomputed vector; the `.T` at the call site dropped | `vbmc.py`, new `test_vbmc_recompute_lcb_max.py` | +`test_recomputed_maximum_is_the_running_maximum_of_the_current_process`, +`test_recomputed_maximum_reaches_the_warmup_check` |
| 14 | `36f2cc9` | W2-3 | ending warm-up sets `last_warping` and `last_successful_warping` to the current iteration (`LastNonlinearWarping` not ported) | `vbmc.py`, `test_vbmc_loop_termination.py` | +`test_setup_vbmc_after_warmup_starts_the_warping_clocks` |
| 15 | `8226833` | W2-5 | `iteration + 1 < min_iter` | `vbmc.py`, `test_vbmc_loop_termination.py` | changed `test_vbmc_check_termination_conditions_prevent_early_termination` |

## Contract and test, per commit

1. **W2-10 / B-M12.** `vbmc.m:524-525` reads `options.EntropyForceSwitch*options.MaxFunEvals`; `private/vbmc_termination.m:80` is `optimState.EntropySwitch && isfinite(options.EntropyForceSwitch)`. The budget factor stays `optim_state["max_fun_evals"]` as ruled. Test 1: a two-iteration `D=2` seeded run with `entropy_force_switch = 0.0` and the switch forced on after construction (it is off below `det_entropy_min_d`); the run must turn the switch off in iteration 0 and record the action. Before: `TypeError` at the loop expression. Test 2: `entropy_force_switch = np.inf` with a stable iteration and the switch on must terminate and leave the switch on. Before: `assert terminated` failed.
2. **W2-11.** `vbmc.m:781-792`: the instantaneous moments are stored only while `RunMean` or `RunCov` is empty; otherwise `RunMean = wRun*RunMean + (1-wRun)*mubar` with `wRun = MomentsRunWeight^(N-LastRunAvg)`. Test: a two-iteration run; the recorded `optim_state` of iteration 1 must equal that formula, computed independently from the recorded posterior's analytic transformed-space moments and iteration 0's stored values, and must differ from iteration 1's moments alone. Before: `assert False` on the average. Shape remark of B-2: PyVBMC stores `run_mean` as a `(1, D)` row where MATLAB stores a column; both branches use the row consistently, so the update broadcasts correctly. I left the row shape (it is reported state only); flagging it rather than changing it.
3. **W2-16.** `vbmc.m:584`: `Nfastopts = ceil(evaloption_vbmc(options.NSelbo,Knew))`. Test: one short seeded run (`D=2`, 4 iterations, warm-up off, `variable_means=False`, cheap `ns_elbo`, `tol_weight=0.2` so pruning keeps `vp.K` below the training-set size, `do_final_boost=False`) with `optimize_vp` wrapped to record its arguments; the warp refit is identified by `optim_state["skip_active_sampling"]` still being set. The test asserts `K != vp.K` (non-vacuity) and `n_fast_opts == ceil(ns_elbo(K))`. Before: `assert 30 == 40`. A companion test asserts a warp actually happened. Confirmed against `vbmc.m` that the main-loop site is a recorded deliberate difference (MATLAB's `K = options.Kwarmup` at `:459`, issue #98 / PR #101), so only the warp branch changed.
4. **W2-6.** `misc/warp_input_vbmc.m:8`, `:133` take the old transform from the posterior handed in; the ruling is that PyVBMC inverts the search state with the transform of the current space. Implemented without a signature change by capturing `function_logger.parameter_transformer` before the function replaces it. Test (hand-built, in the spirit of `wave2_stale_transformer_mechanism.py`): the same distribution expressed in a rescaled inference space is handed in; both calls give the same whitening transform (asserted), both draw from generators seeded identically, and `lb_search`, `ub_search` and `search_cache` must agree. Before: `[-0.238, -3.001]` vs `[-1.176, -7.971]`. `test_warp_input_search_cache` previously expected the inversion with `vp.parameter_transformer`; it now expects the logger's, and it also failed before the fix.
5. **W2-12.** The reported type must distinguish a bound-constrained problem; the transform sends a bounded variable's bounds to ±inf. Test builds results for bounded, mixed and unbounded problems through `_create_result_dict` on a hand-filled history (no run). Before: `'unconstrained' == 'bounded'` failed.
6. **W2-13.** `private/vbmc_output.m:10` copies MATLAB's 1-based `optimState.iter`, the count. Test: seven recorded iterations with `idx_best = 5` must give `iterations == 7`, `best_iter == 5`; and a single iteration gives `1` / `0`. Before: `assert 6 == 7`.
7. **B-M2.** `vbmc.m:576` (`vp.mu = gp.X'`) assigns by value, and the main loop's twin already copies. Test: the recorded `optimize_vp` calls must have `np.shares_memory(vp.mu, gp.X)` false. Before: `assert not True`.
8. **B-M3.** `misc/finalboost_vbmc.m:40`, `:48` write the two fields on an `optimState` passed by value. Test: with `warmup=True` and `entropy_alpha=0.5` on the instance, the optimization the boost runs must see `False` and `0`, and the instance must keep `True` and `0.5`. Before: nine failures in that file (the new test plus the `optim_state is` assertion in the parametrized guard test).
9. **B-M5.** `misc/best_vbmc.m:79` writes `vp.stats.stable` into a by-value copy. Test: the returned posterior is a different object with the same variational parameters and the same generator, carries the flag, and the recorded posterior's `stats` still has no `"stable"` key. Before: six failures in that file.
10. **B-M9.** `vbmc.m:884-913`: `vp_old` captured after the loop; `new_final_vp_flag = idx_best ~= iter`, set again by `changedflag`; `sKL` recomputed and the line printed under that flag. Test: a short seeded run with `do_final_boost=False` and the selection forced to the first iteration must emit exactly one "finalize" line (caplog on the `VBMC` logger at INFO, `display: "iter"`). Before: no such line.
11. **W2-14.** `private/vbmc_warmup.m:39` reaches the same empty window; the ruling is that an empty window means no stable count. Test: with `tol_stable_warmup == fun_evals_per_iter` the check at three recorded iterations returns False, and at four returns True on a flat history; the helper sets the other two criteria so the return value is the stability count alone. Before: `ValueError: zero-size array to reduction operation maximum`.
12. **W2-1.** `private/vbmc_warmup.m:60-63`, 1-based `RecentPast = iter-ceil(...)+1` with `idx_last(max(2,RecentPast):end)`. Test: seven recorded iterations, window three long; a jump in `lcb_max` at index 3 (before the window) must let warm-up end, a jump at index 4 (inside it) must not. Before: the first assertion failed (`np.False_`) because the defective window reached back to index 2.
13. **W2-2.** `private/recompute_lcbmax.m` and the plan's specification: NaN at inactive rows, latent prediction with the current GP at the active ones, `lcb = fmu - ELCBOImproWeight*sqrt(fs2)`, trailing cumulative maximum skipping NaN (`np.fmax.accumulate`), sampled at `iteration_history["N"] - 1`; `private/vbmc_warmup.m:46-50` prefers the recomputed vector. Test 1 rebuilds the expectation independently, one single-point `gp.predict` per active row and a Python maximum over the rows below each count, and checks the sequence is non-decreasing and that a dropped row carrying by far the largest value never appears. Test 2 drives `_check_warmup_end_conditions` in a state where only the lcb vector decides the long-term-improvement criterion: flat recorded maxima end warm-up, the rising recomputed sequence holds it back. Before: shape `(0,)` and an `IndexError`. Entries whose `N` is unrecorded come back NaN (documented); `_ensure_gp_sampling_history` normalizes `N` to a finite value or `None` at the start of every `optimize`, so `int(N)` is safe.
14. **W2-3.** `private/vbmc_warmup.m:97-102`. Test: on a real end of warm-up both clocks move from `-inf` to the current iteration; on a false alarm (which prunes and continues) they stay at `-inf`. `LastNonlinearWarping` is dead in MATLAB and not ported. Before: `assert -inf == 100`.
15. **W2-5.** `private/vbmc_termination.m:98-99` compares the 1-based counter. The existing test asserted the defect; rewritten to state the contract: with `min_iter = 101`, a run that has performed 100 iterations does not terminate and one that has performed 101 does. Before the fix the second half failed.

## Test commands run, and results

All from the worktree root, `PY = C:/Users/luigi/Documents/GitHub/pyvbmc/.venv/Scripts/python.exe`, with the three thread variables set and `-q -p no:cacheprovider --basetemp=<scratch>/pytest_tmp`:

- `PY -m pytest pyvbmc/testing/vbmc/test_vbmc_loop_termination.py` — 32 passed (per-commit runs before/after each fix as described above)
- `PY -m pytest pyvbmc/testing/vbmc/test_vbmc_loop_state.py` — 2 passed
- `PY -m pytest pyvbmc/testing/vbmc/test_vbmc_recompute_lcb_max.py` — 2 passed
- `PY -m pytest pyvbmc/testing/vbmc/test_vbmc_result_fields.py` — 2 passed
- `PY -m pytest pyvbmc/testing/vbmc/test_vbmc_warp_branch.py` — 3 passed (~6.6 s, one module-scoped run shared by the three tests)
- `PY -m pytest pyvbmc/testing/vbmc/test_vbmc_finalboost.py` — 31 passed
- `PY -m pytest pyvbmc/testing/vbmc/test_vbmc_determine_best_vp.py` — 11 passed
- `PY -m pytest pyvbmc/testing/whitening/test_rotoscaling.py` — 8 passed
- Final combined run of all eight files at HEAD: **91 passed in 13.9 s**

Every fix was seen to fail first: either by running the new test on the unfixed code, or by temporarily reverting the source hunk (through a scratch copy of `vbmc.py`), running, and restoring. No whole test directory, no `test_vbmc_optimize.py`, no oracle or golden command, no benchmark, no install was run.

## Numbers a default run may move — read this before the bit-identity gate

Phase-1 commits that are provably inert at the defaults: W2-10 (the switch is off, the expression short-circuits), W2-16 (`Knew == vp.K` whenever the means are free, which is the default), B-M2 (aliasing only), B-M5 (a deep copy makes no draws; the extra copy that the warp branch used to take was removed, so the count of copies is unchanged), B-M3 (`optimize_vp` only reads `optim_state`; the deep copy carries equal values).

Three phase-1 commits change something a run reports, none of them a computed quantity:

- **W2-11** changes the recorded `optim_state["run_mean"]` and `["run_cov"]` from the instantaneous moments to the running average. Nothing on either side reads them, and no golden trace or oracle records them. `last_run_avg` is unchanged.
- **W2-12** and **W2-13** change two reported fields (`problem_type` for every bounded run; `iterations` by one for every run).
- **B-M3** leaves `vbmc.optim_state["warmup"]` and `["entropy_alpha"]` at their pre-boost values after `optimize()` returns, instead of `False` and `0`. The recorded `optim_state` of the last iteration is written before the boost, and a continued run restores `optim_state` from that record, so no trajectory depends on it.

**One phase-1 commit can change the random stream: B-M9 (`0972a14`).** The recomputation of `sKL` after the loop draws from `vbmc.rng` (`kl_div` with `kl_gauss=True` calls `vp.moments(N=1e5, orig_flag=True)` twice). It used to run only when the boost reported a change; it now also runs when the returned posterior comes from an earlier iteration. When the boost changed the posterior the number of draws is the same as before (only the printed value differs, because the divergence is measured against the posterior the loop ended on rather than the one the last iteration started from), so the generator state is untouched. When `changed_flag` is False and `idx_best != iteration`, 2 × 10^5 draws now happen where none did, which moves `results["rng_state"]`, the pickled generator, and hence any continued run.

That case is reachable. A probe (`D=2`, four iterations, seeds 1–3, `display: "off"`):

```
seed=1 boost=False iterations=4 best_iter=3 last=3 earlier=False
seed=1 boost=True  iterations=4 best_iter=3 last=3 earlier=False
seed=2 boost=False iterations=4 best_iter=2 last=3 earlier=True
seed=2 boost=True  iterations=4 best_iter=2 last=3 earlier=True
seed=3 boost=False iterations=4 best_iter=3 last=3 earlier=False
seed=3 boost=True  iterations=4 best_iter=3 last=3 earlier=False
```

The test I would look at first when the orchestrator runs the directory is the 4 + 4 save/load comparison in `pyvbmc/testing/vbmc/test_vbmc_optimize.py` (around line 602): its first leg runs with `do_final_boost=False`, so if that leg's selection lands on an earlier iteration the pickled generator advances and `elbo_1 == elbo_2` can fail. I could not run it (forbidden) and did not edit it (not my file). I considered gating the recomputation on the display level, as MATLAB's `prnt > 2` does, and rejected it: it would remove draws that happen today for `display: "off"` runs whenever the boost changes the posterior, which is the common case and a much larger divergence. If the PI prefers the run to be insulated, the alternative is to snapshot and restore the generator state around this display-only divergence — that is a deliberate departure from both the current behavior and MATLAB's, so I left it alone.

## Skipped, uncertain, or out of scope

- Nothing was left undone; all fifteen findings are committed.
- **B-M4** (the `final_boost` docstring typing `elbo`/`elbo_sd` as `VariationalPosterior`) sits two lines from an edit of mine but is not among the seven minor observations ruled to be fixed, so I left it.
- **W2-11 shape**: `run_mean` stays a `(1, D)` row where `vbmc.m:782` writes a column. Reported state only; not changed.
- **W2-6 handle**: I used `function_logger.parameter_transformer` as "the transform of the current space". It is set from `self.vp.parameter_transformer` at the top of every active-sampling block, so at a warp it holds the value (not necessarily the object) of the current space's transform. If the PI would rather the caller pass it explicitly, that is a signature change to `warp_input` and its call site.
- The `test_vbmc_warp_branch.py` fixture needs `do_final_boost: False`: with `variable_means=False` the components are fixed to the training inputs and the boost to `min_final_components` raises a broadcast error inside `_gp_log_joint`. That is a pre-existing incompatibility of `variable_means=False` with the final boost, unrelated to this pass, and might be worth a finding of its own.

## Edits needed outside my area

None made, none blocked. Two observations for whoever owns them:

- `examples/pyvbmc_example_3_diagnostics_and_saving.ipynb` prints `'problem_type': 'unconstrained'` and `'iterations': 6` beside `'best_iter': 5` in its stored output. Both fields change with commits 5 and 6; the notebooks are not executed in CI or docs, so the stored output is now one behind.
- `dev/scripts/*.py` other than `benchmark_targets.py` store `results["iterations"]` as a count (`make_oracle_fixtures.py`, `profile_run.py`, `noisy_acq_experiment.py`, `pymc_feasibility.py`, `reference_noisy_extension.py`); they become correct with no edit. `golden_trace.py` writes its own `n_it` and `golden_replay.py` compares against `len(trace["iter"])`, so neither is affected.

## Record updates for the orchestrator

- `AGENTS.md`: nothing it says about the loop becomes wrong. If a line is wanted, the two that changed meaning are the results dictionary (`iterations` is a count) and `determine_best_vp` (returns a copy; the iteration history is never written to by the selection, so recorded posteriors never carry the stability flag — which is what MATLAB does, B-M1).
- Known-differences sheet: the entry on the sieve's candidate count (`known_differences.md:1050-1070`) and the statement in `verification/wave1_P6.md` that the warp branch agrees with MATLAB both need the correction W2-16 asked for — after this commit the warp branch does agree, and the sheet's description of the sieve should say so. Add the MATLAB-side `problem_type` defect (`private/vbmc_output.m:5-9` tests the transformed bounds) to `matlab_side_defects.md`. A line is also due for the two result-field strings: PyVBMC keeps `"bounded"` where MATLAB has `'boundconstraints'`, and `best_iter` stays 0-based where `output.bestiter` is 1-based (B-M13).
- Counterpart map: `private/recompute_lcbmax.m` is now genuinely ported (`pyvbmc/vbmc/vbmc.py`, `_recompute_lcb_max`); the row that already claimed "ported" is finally true. `private/vbmc_warmup.m:46-50` (the preference for the recomputed vector) is now ported as well.
- Porting log (`pyvbmc/vbmc/README.md`): its "Porting status" section did not list `recompute_lcbmax` among the unported functions, so nothing to remove there; worth a line that the input-warping clocks are set at the end of warm-up.
- Documentation: `optimize`'s docstring and `_create_result_dict`'s docstring now define `iterations` and `best_iter`. No new `.rst` is needed (no new public class or function). No new fixture files or directories, so `MANIFEST.in` is untouched.
