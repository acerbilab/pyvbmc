# Wave 5 fix pass: fix agent C

Raw report of fix agent C, wave 5 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-21. One Opus agent on its
own git worktree, cut at `a2104e6`, made the fixes of the rows of
`../verification/wave5.md` on the active sampling, the option checks and the rounding, one commit per
row as the PI ruled them; the orchestrator reviewed the diffs and
cherry-picked the commits onto `dev-port-review`, where they have other
hashes (`../verification/wave5.md`, "Fix commits"). The text below the rule
is the agent's final message, unedited.

---

# Wave 5 fix pass: report of fix agent C

Worktree: `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-a462f3e878bb2dc26`
Branch: `worktree-agent-a462f3e878bb2dc26` (cut from `b2c16ebf`, fast-forwarded to `a2104e61`; nothing pushed)

## Setup outputs

- `git merge --ff-only dev-port-review` → fast-forward `b2c16ebf..a2104e61`; `git log -1 --format=%h` printed `a2104e61`.
- From the worktree root, `python -c "import os, sys; sys.path.insert(0, os.getcwd()); import pyvbmc; print(pyvbmc.__file__)"` printed
  `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-a462f3e878bb2dc26\pyvbmc\__init__.py`.
- Every Python command ran with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. One standalone probe script, in the scratchpad (`.../scratchpad/fix_wave5_C/check_rounding.py`) with `PYTHONPATH=<worktree>`, printed the same `pyvbmc.__file__`.
- `git status --porcelain` is empty at the end; HEAD is `46a20143`.

## `git log --oneline a2104e61..HEAD`

```
46a20143 docs: say what integer variables do and do not do
f4a2eb58 fix(stats): round a half away from zero where MATLAB VBMC does
cb45630d fix(active_sample): evaluate a cached starting point that was moved
5c396688 fix(vbmc): check the fractions that divide the acquisition search
9fb11028 fix(vbmc): refuse the acquisition-function portfolio that is not ported
5536e744 fix(vbmc): refuse the acquisition search that is not available
```

Diffstat over the range: 16 files, +586 / −72.

## Commits in order

| # | hash | finding | what it does | files | tests added or changed |
|---|---|---|---|---|---|
| 1 | `5536e744` | W5-25 | the `"Nelder-Mead"` branch of the acquisition search is removed; `search_optimizer` takes `"cmaes"` or `"none"`, anything else refused by a new `_validate_search_optimizer_option` (`ValueError`) at construction and in `load`; `load` replaces a stored `"Nelder-Mead"` by `"cmaes"` for `D = 1`, beside the other back-fills and before `new_options` is applied, so `new_options` still wins and an explicitly supplied `"Nelder-Mead"` is refused; the `.ini` description and the `acq_fun` docstring drop the removed search | `active_sample.py`, `vbmc.py`, `advanced_vbmc_options.ini`, `test_options.py`, `test_vbmc_option_names.py`, `test_vbmc_optimize.py` | **changed** the pinned `search_optimizer` description in `test_options.py`; **changed** `test_vbmc_uniform` (`options = {}`) and the comment at `_test_optimize_reproducibility`; +two rows in the parametrized `test_new_options_are_checked_as_at_construction`; +`test_a_search_optimizer_outside_the_two_values_is_refused` (4 values), +`test_the_two_search_optimizers_are_accepted` (2), +`test_a_one_dimensional_run_that_stored_nelder_mead_loads`, +`test_a_wider_run_that_stored_nelder_mead_is_refused`, +`test_a_stored_nelder_mead_that_new_options_replaces_loads`, with a helper `_vbmc_of_dimension` |
| 2 | `9fb11028` | W5-23 | `acq_hedge` refused at construction and in `load` by a new `_validate_acq_hedge_option` (`NotImplementedError` naming `private/acqhedge_vbmc.m`), as `noise_shaping` is; the option stays declared and read, so it is not inert; the `.ini` description says it must be False and why; `load`'s `Raises` section names it | `vbmc.py`, `active_sample.py` (one comment), `advanced_vbmc_options.ini`, `test_vbmc_init.py`, `test_vbmc_option_names.py` | **changed** `test_vbmc_optimstate_acq_hedge`, which asserted the empty `optim_state["hedge"]`; it asserts the refusal, and its docstring says so; +one row in the parametrized test |
| 3 | `5c396688` | W5-24 (e), (c), (a) | (e) new `_validate_search_fraction_options` checks the five fractions of the sieve, each in `[0, 1]` and together `<= 1`, naming them, their values and their sum; the guard of `_get_search_points` raises one string instead of three arguments. (c) the training rows at the head of the search set are kept out of `optim_state["search_cache"]` (`X_search[inds[inds >= n_train_cand]]`; identical to the old expression when no repeats are offered). (a) the two `np.delete` calls stay, with a comment saying nothing reads either array again and that they stand where `private/activesample_vbmc.m:242` has them | `vbmc.py`, `active_sample.py`, `test_vbmc_option_names.py`, `test_vbmc_active_sample.py` | **changed** `test_get_search_points_more_points_randomly_than_requested`, which built its instance with fractions construction now refuses; it writes them into a built instance's options; +one row in the parametrized test; +`test_search_fractions_that_claim_more_than_the_whole_are_refused`, +`test_a_search_fraction_outside_the_unit_interval_is_refused`, +`test_the_shipped_search_fractions_leave_room_for_a_search_cache`, +`test_two_steps_with_a_search_cache`, +`test_the_search_cache_holds_no_training_input` |
| 4 | `cb45630d` | W5-7 | the stored value of a cached starting point is reused only where the candidate equals `parameter_transformer(cache["x_orig"][idx])` exactly, in the inference space; where the clip into the search box or the snap to the integer grid moved it, the target is called at the candidate. The row leaves the starting cache either way (the existing deletion is keyed on the cache index, untouched) | `active_sample.py`, `test_vbmc_active_sample.py` | +`_acquire_one_cached_point` helper, +`test_a_cached_point_the_search_box_moved_is_evaluated`, +`test_a_cached_point_the_integer_grid_moved_is_evaluated`; the two existing cached-point tests pass unchanged |
| 5 | `f4a2eb58` | W5-6 | new private `pyvbmc/stats/_rounding.py` with `round_half_away_from_zero`, reading the decision from the exact fractional part; called at the twelve sites (`get_hpd.py:38`; `active_sample.py` ×6; `gaussian_process_train.py` ×4; `variational_optimization.py:46`) and in `AbstractAcqFcn._real2int`, whose expression it replaces. Scalars give a Python `int`, arrays a float array | `_rounding.py` (new), `get_hpd.py`, `active_sample.py`, `gaussian_process_train.py`, `variational_optimization.py`, `abstract_acq_fcn.py`, `test_rounding.py` (new), `test_get_hpd.py` | +`pyvbmc/testing/stats/test_rounding.py` (7 tests: the six halves, the largest double below a half and its negative, the largest odd integer, a scalar giving an `int`, an array rounded elementwise with the argument left alone, a list); +`test_get_hpd_sizes_a_tie_away_from_zero` (`N = 5 → 1`, `N = 25 → 3` at `hpd_frac = 0.1`); the `_real2int` tests pass unchanged |
| 6 | `46a20143` | W5-26, W5-27, W5-25's neighbour | the FAQ answer and the `integer_vars` description say the same thing: experimental; the active-sampling search is snapped; the initial design and a provided `x0` are not, as in MATLAB VBMC, so the first `fun_eval_start` evaluations are off the grid unless the starting points cover the whole design; on a grid the search often returns a point already evaluated, which on a noiseless target spends an evaluation and adds nothing; the hard bounds sit half an integer outside the range. The FAQ keeps its smoothness warning; the section title is unchanged | `docsrc/source/faq.md`, `advanced_vbmc_options.ini` | none (the `test_options.py` test that pins descriptions passes; `integer_vars` is not among the pinned three) |

## Contract and check, per commit

1. **W5-25.** MATLAB's counterpart is a bounded `fmincon` with `MaxFunEvals` and `TolFun` (`private/activesample_vbmc.m:291-296`); the branch had none of the three. `v1.0.4:pyvbmc/vbmc/active_sample.py:356-360` was read and confirms that release forced every `D = 1` run onto the value with `force=True`. The refusal message names the option, both values, and for `"Nelder-Mead"` says the search is not available, that one dimension is searched by a bounded scalar method whatever the option holds, and how `new_options` continues a saved run. The unreachable final `else` of the branch chain stays as a guard.
2. **W5-23.** `_validate_acq_hedge_option` is modelled on `_validate_noise_shaping_option`. `test_inert_options_are_the_declared_options_nothing_reads`, which recomputes the registry from the package, passes: `acq_hedge` is still read at `active_sample.py:320` and `vbmc.py:1048`, and now by the validator. I left `vbmc.py:1048` (the empty `optim_state["hedge"]`) alone, being outside my area; it is unreachable.
3. **W5-24.** (a) I **kept** the two `np.delete` calls, with the comment, rather than removing them: they are a faithful port of `private/activesample_vbmc.m:242`. (c) `inds[inds >= n_train_cand]` keeps the sorted order and is bit-identical to `X_search[inds]` when `n_train_cand == 0`, which is every default run. (e) The type check is `numbers.Real`, so a string gives a message of its own rather than a `TypeError`, and NaN is refused. Two consecutive steps at `search_cache_frac = 0.25` complete; the acquired point of the second is the one the cache offers first (the shared behaviour of W5-24 (b)), pooled by the logger, so the test counts target calls rather than rows.
4. **W5-7.** `private/activesample_vbmc.m:555`, `:637`, `:219`, `:388` clip, snap and add the stored value in that order, so this departs from MATLAB deliberately, as ruled. The comparison is `np.array_equal` on the transformed row; the transform is elementwise, so transforming the single cache row reproduces bit for bit what `_get_search_points` computed for the whole block. Both new tests fail on the unfixed code (verified by reverting the hunk: `func_count` unchanged, the stored value recorded at the moved point).
5. **W5-6.** `check_rounding.py` in the scratchpad reports: the former `_real2int` expression and the helper agree **bit for bit** on 2000 random rows, on a grid of exact halves, on the six ties, on the largest double below a half and on ±(2^53 − 1); `round(v)` and the helper disagree on none of 200 000 random doubles; `hpd_frac = 0.8` has no tie for `N` up to 2000; the shipped sieve fractions at `ns_search = 8192` give `[2048, 2048, 0, 2048, 0]`, no tie; `ns_gp_max / sqrt(N)` has its single tie at `N = 1024`, which `stable_gp_sampling = 200 + 10 D` puts out of reach (sampling has stopped and `gp_s_N` is `stable_gp_samples = 0` long before).
6. **W5-26/27.** The FAQ anchor `(faq-does-vbmc-support-inference-with-integer-parameters)=` and the section title are untouched, so `skills/pyvbmc/SKILL.md` and the FAQ's own table of contents still resolve.

## Test runs, and results

All from the worktree root, `-q -p no:cacheprovider`, with the three thread variables set. `test_vbmc_optimize.py` was never run.

- `test_vbmc_option_names.py`, `test_options.py` — after commit 1: **87 passed**
- `test_vbmc_active_sample.py` — after commit 4: **51 passed**
- `test_vbmc_option_names.py`, `test_options.py`, `test_vbmc_init.py` — after commit 2: **214 passed**
- `test_gaussian_process_train.py`, `test_variational_optimization.py`, `test_gp_training_policy.py`, `pyvbmc/testing/acquisition_functions/`, `pyvbmc/testing/stats/` — after commit 5: **258 passed**
- Final combined run at HEAD of `test_vbmc_active_sample.py`, `test_vbmc_option_names.py`, `test_options.py`, `test_vbmc_init.py`, `test_vbmc_save_and_load.py`, `pyvbmc/testing/stats/`, `pyvbmc/testing/acquisition_functions/`, `test_gaussian_process_train.py`, `test_variational_optimization.py`, `test_gp_training_policy.py` — **540 passed in 42.7 s**

Each fix was seen to fail first where the defect was observable: commits 3 (c) and 4 by reverting the hunk with an Edit and putting it back; commits 1, 2 and 3 (e) introduce refusals where nothing was refused before. No full suite, no oracle command, no `golden_replay.py`, no `optimize()` run, no install.

## Changelog sentences (for a user of release 1.0.4)

- **W5-25** — "`options['search_optimizer']` takes `'cmaes'` or `'none'`; the `'Nelder-Mead'` value is gone, together with the search it named, which ignored the search bounds and `options['search_max_fun_evals']` and could drive the acquired point far outside the search box on an unbounded variable. A run of one variable is searched by a bounded scalar method whatever the option holds, as before. A file saved by 1.0.4 that holds `'Nelder-Mead'` loads with `'cmaes'` where the problem has one variable, since 1.0.4 wrote the value into every such run; with more variables `VBMC.load` refuses it and says to pass `new_options={'search_optimizer': 'cmaes'}`." Can stop a script that set the option, or that loads a saved multi-variable run made with it, so it belongs in the "Upgrading from 1.0.4" list.
- **W5-23** — "`options['acq_hedge']` must be `False` and is refused when the `VBMC` object is created and by `VBMC.load`: the portfolio of acquisition functions it asks for (MATLAB VBMC's `acqhedge_vbmc.m`) is not ported, and a run that set the option used to end in an `UnboundLocalError` at its first active-sampling step, after paying for the initial design." Can stop a script that set it — such a run used to fail anyway — so it belongs in the "Upgrading from 1.0.4" list.
- **W5-24** — "The five fractions that divide the candidates of the acquisition search among their sources (`search_cache_frac`, `heavy_tail_search_frac`, `mvn_search_frac`, `hpd_search_frac`, `box_search_frac`) are checked when the `VBMC` object is created and by `VBMC.load`: each must lie in `[0, 1]` and together they may claim at most the whole search set. With the shipped fractions any `search_cache_frac` above 0.25 used to raise a `ValueError` at the second active-sampling step of the run, with a message that printed as a tuple. With `options['max_repeated_observations']` above zero on a noisy target, the training inputs offered as candidates for a repeat no longer enter the search cache, where they used to come back at a later step past the cap on consecutive repeats." Can stop a script that set fractions summing to more than one — such a run used to fail one iteration in — so it belongs in the "Upgrading from 1.0.4" list.
- **W5-7** — "A starting point taken from the cache whose stored value was recorded at a point the search box or the integer grid had moved it to is now evaluated at that point instead: the target is called, and the value the run records belongs to the point it records." Changes what a run returns only where `options['integer_vars']` moved a cached starting point, or where a starting point lay outside the search box; no line in the upgrading list is needed beyond the entry.
- **W5-6** — "A quantity that falls exactly halfway between two integers is rounded away from zero, as MATLAB VBMC rounds it, wherever PyVBMC sizes something from a fraction: the high-posterior-density subset, the shares of the acquisition search among its sources, the number of GP hyperparameter samples with its burn-in and initial training points, and the bonus number of variational components. `pyvbmc.stats.get_hpd(X, y, 0.1)` on 5 points returns 1 point, where it used to return none." Can change what a script that calls `get_hpd` returns, and what a run with a non-default `hpd_frac` or `hpd_search_frac` returns, so it belongs in the "Upgrading from 1.0.4" list.
- **W5-26/27** — "The FAQ and the description of `options['integer_vars']` say what integer variables do: the feature is experimental, the points of the active-sampling search are snapped to the integer grid, the initial design and a provided `x0` are not, as in MATLAB VBMC, and on a grid the search often returns a point that has been evaluated already." Documentation only; cannot stop a script.

Commits 1 to 4 and 6 change no number of a run at the shipped options (`acq_hedge` and `search_cache_frac` are off, `search_optimizer` is `"cmaes"`, and a cached starting point is moved only by a clip that `active_search_bound = 2` puts out of reach or by `integer_vars`). Commit 5 is bit-identical at the shipped options on every check I could make without the heavy gates; see the caveat below.

## What I stopped on

Nothing. All six items are done as ruled.

## What else I noticed

- **One tie of commit 5 that the shipped options can still reach.** The ledger lists "a starting cache that leaves a number of random points not divisible by four" among the ways to a tie, and that case does not need a non-default option: with a user-supplied `x0` of more rows than the initial design consumes, the starting cache is non-empty at an active-sampling step, `N_random_points = ns_search − N_cache` can be `≡ 2 (mod 4)`, and the quarter shares of the sieve then differ by one candidate from the old rounding. Everything the golden traces and the oracles exercise should be checked for it; I ran none of those gates.
- **`_validate_search_optimizer_option`, `_validate_acq_hedge_option` and `_validate_search_fraction_options` all run through `_validate_option_values`**, so unlike `_validate_search_acq_fcn_option` (wave 4's note) they fire on `load` as well as at construction, which is what the ruling asks for.
- **The `load` back-fill of `search_optimizer` sits before the `new_options` update**, beside the `performance_calibration` and `show_tips` back-fills. That is what makes `new_options` win over the stored value and makes an explicit `new_options={"search_optimizer": "Nelder-Mead"}` refused even at `D = 1`; putting it after the update would have swallowed the explicit value.
- **The verifier's scripts `wave5_P2_2_search_cache.py` and `wave5_P2_11_search_cache_frac_second_step.py` no longer run against the fixed package**: they build `VBMC` with `search_cache_frac` of 0.5 or 1.0 beside the shipped fractions, which construction refuses from commit 3 on. They are records of a state of the code, not gates, so I left them; the orchestrator may want a line in the wave-5 record saying so.
- **`pyvbmc/stats/_rounding.py` is not exported by `pyvbmc/stats/__init__.py`**, so it needs no API page. Importing it from `pyvbmc/acquisition_functions/abstract_acq_fcn.py` pulls in `pyvbmc.stats`, which imports nothing from `pyvbmc`, so no cycle.
- **`test_vbmc_optimize.py:230` had been passing `"Nelder-Mead"` in a `D = 1` test since wave 1 took the one-dimensional search off that branch**, so the value had no effect there; the file was not run, as instructed.
