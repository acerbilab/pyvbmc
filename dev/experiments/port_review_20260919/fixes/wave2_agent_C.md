# Wave 2 fix pass, agent C: construction state and load

Raw report of a fix agent, wave 2 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, in
its own git worktree cut from `dev-port-review` at `2f2bc94`, fixed the
wave-2 findings on the construction state and on `load` (W2-9, W2-21,
C-M1, C-M2, then W2-4), one finding per commit with a test written against
the contract, running only the focused test files it touched. The commit
hashes in the report are those of the agent's worktree branch; the
orchestrator reviewed each diff and cherry-picked the commits onto
`dev-port-review`, where they carry other hashes (the plan's worklog lists
them). The text below the rule is the agent's final message, unedited;
`../verification/wave2.md` holds the ledger and the PI's dispositions.

---

# Wave-2 fix pass, agent C

Setup: `git merge --ff-only dev-port-review` succeeded, HEAD was `2f2bc94` before the first commit; `pyvbmc.__file__` resolves inside the worktree. All Python ran with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. No push, no touch of the main checkout.

## `git log --oneline 2f2bc94..HEAD`

```
90d4897 fix(vbmc): build the initial posterior from the transformed starting point
e325bad fix(vbmc): clean up the log-file handlers without touching the others
9c16942 fix(vbmc): report the prior and the Gaussian process in the summary
a8d489b fix(vbmc): restore the transformer of the loaded iteration and report x0
a4eb068 fix(vbmc): follow the current evaluation budget when a run is continued
```

## Commits in order

| hash | finding | what it does | files | tests |
|---|---|---|---|---|
| `a4eb068` | W2-9 | `optim_state["max_fun_evals"]` is written from the current options in `load` and in the continuation block of `optimize`, on the ordinary path as it already was on the budget path; the reader in `gaussian_process_train.py` is untouched | `pyvbmc/vbmc/vbmc.py` | `test_vbmc_save_and_load.py`: 2 tests added + 2 helpers |
| `a8d489b` | W2-21 | `load` hands the instance and the function logger the restored posterior's transformer; `__init__` keeps `self.x0_orig` (the caller's coordinates) and `__str__` prints it; `load` backfills `x0_orig` for instances saved without it | `pyvbmc/vbmc/vbmc.py` | `test_vbmc_save_and_load.py`: 2 tests added; new file `test_vbmc_summary.py`: 3 tests |
| `9c16942` | C-M1 | `__str__` reads the GP from `self.gp` and the log-prior/sampler from `self.prior` | `pyvbmc/vbmc/vbmc.py` | `test_vbmc_summary.py`: 2 tests added |
| `e325bad` | C-M2 | `_init_logger` walks a copy of `logger.handlers`, considers `logging.FileHandler` only, and the level message has its missing space | `pyvbmc/vbmc/vbmc.py` | new file `test_vbmc_logger.py`: 3 tests + a fixture that restores the package loggers' handlers |
| `90d4897` | W2-4 | `x0` is transformed before the variational posterior is built from it; the initial-design cache keeps the caller's coordinates | `pyvbmc/vbmc/vbmc.py` | new file `test_vbmc_initial_posterior.py`: 4 tests |

Each commit carries the two required trailer lines. The pre-commit hooks reformatted the save/load test file twice; the rewrite was staged and the commit repeated (never `--no-verify`).

## Contract and check, per finding

- **W2-9.** Contract: a run loaded with a larger `max_fun_evals` schedules the hyperparameter fit as a fresh run with that budget would at the same evaluation count. Both tests call `_get_gp_training_options` at a series of `n_eff` values and compare `init_N` between the loaded/continued run and a freshly constructed `VBMC` of the same `D` carrying that budget (`fun_eval_start` is asserted equal first so a mismatch is legible). `test_load_with_a_larger_budget_...` covers `load`; `test_resumed_run_...` calls `optimize()` on the loaded finished instance and leaves it at the last step before the first iteration by monkeypatching `VBMC._log_column_headers` to raise, so the continuation block is exercised without paying for an iteration. Before the fix both gave 64/9 against 858/759/…; after, they agree. (The second test fails even with only the `load` half fixed, since the continuation block restores the recorded copy.)
- **W2-21.** Contract: the three transformers are one object, the chosen iteration's; and the `x0` line of `__str__` is in original coordinates whatever warps the run has made. `test_load_shares_the_parameter_transformer_of_the_chosen_iteration` asserts `is` identity for `iteration` in `(None, 0, 3, 6)` on the committed `test_vbmc_save_static.pkl` and that the shared map agrees with the recorded posterior's on a probe point. `test_load_prefers_the_iterations_map_over_the_saved_live_one` saves a doctored copy of that run whose live map differs from the recorded ones (what a warped run looks like) and checks the restored instance takes the iteration's map, not the later one. `test_summary_keeps_the_starting_point_across_a_warp` installs a changed map on instance, posterior and logger (as `whitening.warp_input` does), asserts the map really moved the point, and requires the printed `x0` to be unchanged. `test_summary_of_a_run_loaded_from_a_file` pins that an instance saved without `x0_orig` still loads and prints. The static pickle keeps working; no attribute was renamed or removed.
- **C-M1.** Contract: the summary reports what the instance holds. `test_summary_reports_the_prior_the_caller_gave` compares the two fields with `str()` of the very functions passed as `log_prior=`/`sample_prior=`; `test_summary_reports_the_gaussian_process_of_the_run` requires `None` before a run and the loaded run's own GP text afterwards.
- **C-M2.** Contract: every stale file handler removed, other handler kinds tolerated, the level message spelt correctly. One test attaches two duplicate `FileHandler`s for the configured file to the process-wide `"VBMC"` logger and requires that neither survives and exactly one writer of that file remains; one attaches a `StreamHandler` and requires it to survive (it raised `AttributeError` before); one requires `ValueError` matching `not a recognized string or logging level`. An autouse-free fixture restores the handlers of `VBMC`, `VBMC_init` and `VBMC.stream_only` and closes the ones the test added.
- **W2-4.** Contract: `misc/setupvars_vbmc.m` records `optimState.Cache.X_orig = x0` at `:55`, transforms `x0` at `:65`, and tiles the transformed `x0` into `vp.mu` at `:80-82`. The tests map `vp.mu` back with the instance's transformer and require the starting point: the bounded box 0/10 with plausible 4/6 and `x0 = 5`; the same box with `x0 = 5.5` (off centre); the same under `bounded_transform="logit"`; and the 2-D unbounded case with `x0 = (3.5, 0)`. All four failed before (5 → 9.94, etc.) and pass after. The cache line in `_init_optim_state` was left alone and still receives original coordinates, since the transformed value is only bound to `self.x0` where it was before.

## Commands run and results

All from the worktree root, with `-x -q -p no:cacheprovider --basetemp=<scratch>/pytest_tmp`:

- `test_vbmc_save_and_load.py` — before W2-9: 2 failed (the new tests), 9 deselected; after: 11 passed; before W2-21: 2 more failed; after: 16 passed with `test_vbmc_summary.py`.
- `test_vbmc_summary.py` — before W2-21: 1 failed, 2 passed; before C-M1: 2 failed, 3 passed; after both: 5 passed.
- `test_vbmc_logger.py` — before C-M2: 3 failed; after: 3 passed.
- `test_vbmc_initial_posterior.py` — before W2-4: 4 failed; after: 4 passed.
- `test_vbmc_init.py` (permitted): 68 passed both before and after W2-4, so nothing there asserted the defect and the one exception in my brief was not needed.
- Final combined run of the five files: **93 passed in 3.3 s**.
- Reproduction scripts (with `PYTHONPATH=<worktree>`, `pyvbmc.__file__` printed from the worktree each time): `wave2_resume_budget.py` before the fix (stale horizon 40 against the option's 500, 9 starting points against 759 at 60 evaluations); `wave2_initial_vp_mu.py` before and after (means now map back to the starting point in every case).

No test suite, no `optimize()` run, no oracle or golden command, no install, no benchmark.

## Phase-1 invariance

I believe no number of a default run moves in `a4eb068`, `a8d489b`, `9c16942`, `e325bad`: W2-9 touches only `load` and the `if self.is_finished:` block, neither on a default run's path; W2-21 adds a read-only attribute (a copy of `x0`, no RNG draw, no reordering) plus `load` and `__str__`; C-M1 is `__str__` only; C-M2 is handler bookkeeping and a message string. The oracle fixtures rebuild their variational posteriors directly from stored arrays (`pyvbmc/testing/oracles/_state.py`), not through `VBMC.__init__`, so W2-4 should leave the oracle check exact as well — worth confirming with the gate.

## Skipped, uncertain, outside my area

- **Left deliberately, not in the ruling:** the `log-density` line of `__str__` reads `getattr(self, "log_likelihood", self.log_joint)`. `self.log_likelihood` is always assigned and is `None` whenever no separate prior was given, so the fallback never fires and the field prints `None` for an ordinary `VBMC(f, x0, lb, ub)`. It is the same shape of defect as C-M1 and one line from it, but the PI's ruling named three fields; I left it and flag it for triage.
- **Nothing required an edit outside my area**, and I made none: the areas of agents A and B (`optimize`'s loop and the search-GP branch, the option loading, `_init_optim_state`, the `new_options` lines of `load`) are untouched. In `load` my hunks sit above the `options.update(new_options)` block (the transformer lines, right after the restore loop) and below it (the `x0_orig` backfill and the budget lines), as instructed.
- **A dependency worth knowing:** `test_resumed_run_schedules_the_gp_fit_as_a_fresh_run` monkeypatches `VBMC._log_column_headers` as the stopping point between the continuation block and the first iteration. If agent A's work renames or moves that call, the test needs the new seam.
- `load`'s `x0_orig` backfill gives the exact starting point for a stored run that never warped, and the old (wrong for a warped run) inverse otherwise; nothing better is recoverable from such a file, since `optim_state["cache"]["x_orig"]` is consumed by the initial design (`active_sample.py:195`).
- The two gitignored `test_vbmc_save_dynamic*.pkl` files the suite writes were removed; `git status` is clean and nothing untracked remains.

## Record updates for the orchestrator

- **`AGENTS.md`**, "Two coordinate spaces": the sharing invariant now also holds after `load`, which hands the instance and the function logger the restored posterior's transformer; the sentence "tests assert identity" can name the loaded case too. A line that `VBMC.x0` holds the starting points in the transformed space while the new `VBMC.x0_orig` holds the caller's would help, since the static pickle paragraph is about attribute names.
- **Known-differences sheet / counterpart map:** the row for `misc/setupvars_vbmc.m` (already listed for `integer_vars`) should record that the initial variational means are now built from the transformed starting points, matching `:65`/`:80-82`; the P1b comparison report's claim that the initial posterior is equivalent was about the tiling only.
- **Trajectory-moving fix:** `90d4897` (W2-4) changes the initial component means of every run whose starting point differs from its transform, so the golden references and the production-reference pools move with it, as the plan's wave-2 note anticipates.
- **Documentation:** none of the changed behavior is documented in `docsrc/`; the `max_fun_evals` route of `load(new_options=)` is described in the `load` docstring and now behaves as that docstring promises, so no text change is needed.

Worktree: `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-a2c2481b9dbf2d776`, branch `worktree-agent-a2c2481b9dbf2d776`, HEAD `90d48971a0fcea99d15596d58d2319f72e2cbb0e`.
