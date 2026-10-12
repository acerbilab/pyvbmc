# Wave 5, fixes after the independent check: fix agent E

Raw report of fix agent E, wave 5 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-21. One Opus agent on its
own git worktree, cut at `b1bab4d`, made the fixes that the independent check
of the wave-5 pass asked for on the active sampling, the option checks and
the variational posterior (`../verification/wave5.md`, "The independent check
of the pass"); the orchestrator reviewed the diffs and cherry-picked the
commits onto `dev-port-review`, where they have other hashes. The text below
the rule is the agent's final message, unedited.

---

# Wave 5, fixes after the independent check: report of fix agent E

Worktree: `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-ae7473ad5def2bdc3`
Branch: `worktree-agent-ae7473ad5def2bdc3` (fast-forwarded from `b2c16ebf` to `b1bab4d2`; nothing pushed)

## Setup outputs

- `git merge --ff-only dev-port-review` → fast-forward `b2c16ebf..b1bab4d2`; `git log -1 --format=%h` printed `b1bab4d2`.
- From the worktree root, `python -c "import os, sys; sys.path.insert(0, os.getcwd()); import pyvbmc; print(pyvbmc.__file__)"` printed
  `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-ae7473ad5def2bdc3\pyvbmc\__init__.py`.
- Every Python command ran with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. The two standalone scripts (scratchpad `…\fix_wave5_E\`, `PYTHONPATH=<worktree>`) printed the same `pyvbmc.__file__`.
- `git status --porcelain` empty at the end; `--ignored` shows nothing but `!!` entries. HEAD is `500e264f`.

## `git log --oneline b1bab4d2..HEAD`

```
500e264f docs(options): give the search fractions and integer_vars what a user needs
2784acd5 fix(vbmc): take every scalar count, and say what the code does
138f2f24 fix(variational_posterior): keep the stored mode for the default search
32688c3e test(vbmc): pin the value a cached point on the integer grid keeps
d80fb311 test(vbmc): cover the rounding tie the shipped fractions can reach
4b7b0cfa fix(active_sample): cap each source of the search set at what is left
65a2fd99 fix(vbmc): say what the local search does with one variable
```

Diffstat over the range: 9 files, +573 / −111.

## Commits in order

| # | hash | item | what it does | files | tests added or changed, and the failure seen before the fix |
|---|---|---|---|---|---|
| 1 | `65a2fd99` | 1 | The `.ini` description of `search_optimizer`, the Nelder-Mead refusal message and the test that pins the description say that `"cmaes"` runs CMA-ES, replaced by a bounded scalar search where the problem has one variable, and that `"none"` runs no local search. The last `else` of the search chain names the option, its two values and the value given, in place of `NotImplementedError("Not implemented yet")` | `advanced_vbmc_options.ini`, `vbmc.py`, `active_sample.py`, `test_options.py`, `test_vbmc_active_sample.py` | **changed** the `search_optimizer` row of `test_shipped_descriptions_are_stored_in_full` (fails against the old `.ini`: the diff of the two strings was printed); +`test_a_search_optimizer_forced_into_a_built_instance_is_named` (fails on the old message: `assert 'search_optimizer' in 'Not implemented yet'`); +`test_one_dimensional_search_is_skipped_without_a_local_optimizer`, which pins the fact the three texts denied and passes before and after |
| 2 | `4b7b0cfa` | 2 | Each source of the sieve takes the rounded share its fraction gives it or what the sources before it left, in the drawing order (search cache, heavy-tailed, multivariate normal, high-posterior-density, box); the variational posterior draws the rest, and the high-posterior-density split follows its capped total. The guard that raised is gone, with its `Raises` section; the `Notes` state what each source takes. The check at construction is untouched | `active_sample.py`, `test_vbmc_active_sample.py`, `test_vbmc_option_names.py` | **replaced** `test_get_search_points_more_points_randomly_than_requested` by `test_fractions_that_claim_more_than_the_whole_leave_later_sources_out`; +`test_the_sieve_returns_the_number_of_points_it_is_asked_for` (1–9), +`test_the_sieve_returns_the_points_asked_for_with_a_search_cache` (0–3 rows left in the starting cache), +`test_two_halves_of_the_search_set_stay_within_it` (7–10), +`test_each_source_draws_its_rounded_share`; **renamed** `test_the_shipped_search_fractions_leave_room_for_a_search_cache` → `…_leave_a_quarter_unclaimed`. Six of the new cases fail on the old sieve (`ns_search = 2`; a starting cache leaving 1 or 2 rows; the two odd counts of the halves; the forced fractions), each with the old `ValueError: A maximum of N points…` |
| 3 | `d80fb311` | 3 | tests only | `test_vbmc_active_sample.py`, `test_gp_training_policy.py` | +`test_a_share_of_half_a_point_goes_away_from_zero` (a starting cache leaving ten points: 3 heavy-tailed, 3 multivariate-normal, 3 box, 1 from the variational posterior); `_matlab_n_init` calls `round_half_away_from_zero` in place of the builtin `round`. Neither can fail on the code before it: the first covers a tie the shipped options reach that no gate entered, the second has no tie over the schedule it checks (test-only commit, as the item states) |
| 4 | `32688c3e` | 4 | tests only; no package change was needed (no case failed) | `test_vbmc_active_sample.py` | +`test_acquiring_a_cached_point_on_the_integer_grid_reuses_its_value` (one and two integer coordinates, bounded problem, cached point on the grid: no target call, the stored value recorded, the cache emptied); the one unseeded `_state_with_gp` call in `test_a_cached_point_the_search_box_moved_is_evaluated` is seeded. Both new cases pass as the code stands; with the exact comparison perturbed by 1e-15 they fail, as does the existing case beside them |
| 5 | `138f2f24` | 5 | (a) `mode(n_opts=…)` neither reads nor writes the stored mode, so the store always holds the default search's result; (b) the `Returns` section parses (blank line, `mode : np.ndarray`); (c) the Notes qualify "two calls give the same answer" by nothing having drawn from the generator in between; (d) the `get_parameters` docstring separates the rescaling that leaves the density alone from the weight normalization that does not, and cites `misc/rescale_params.m:39-40` | `variational_posterior.py`, `test_variational_posterior.py` | +`test_a_mode_search_with_n_opts_keeps_the_stored_mode`, +`test_a_mode_search_with_n_opts_stores_nothing` (counts the runs of `minimize`). Both fail on the old unconditional write |
| 6 | `2784acd5` | 6 | `sample` accepts any scalar holding a whole number (`np.ndim(N) == 0`, a 0-D array and a NumPy boolean included) and refuses a fractional value or a non-scalar with the same message; the dead `(jacobian == 0)` clause of the `pdf` recovery mask is gone; the `x0` docstring covers the single element and the rows past the `K`-th; the dead `optim_state["hedge"]` line is removed; the `acq_hedge` and search-fraction refusals say how a saved run is loaded; the comment above the two `np.delete` calls is true | `variational_posterior.py`, `vbmc.py`, `active_sample.py`, `test_variational_posterior.py`, `test_vbmc_option_names.py` | +`test_sample_takes_any_scalar_holding_a_whole_number`, +`test_sample_takes_a_boolean_as_the_count_it_stands_for`, +`test_sample_refuses_what_is_not_a_scalar_count`, +`test_constructor_broadcasts_one_element_over_every_coordinate`, +`test_constructor_drops_the_starting_points_past_the_components`, +`test_a_refused_value_says_how_a_saved_run_carrying_it_is_loaded`. Three of the count cases and three of the message cases fail on the code before; the two dead-code removals and the two constructor cases cannot fail (see "What else I noticed") |
| 7 | `500e264f` | 7 | The five search fractions each name their group, their range `[0, 1]`, the sum of at most 1 and the variational posterior that draws the rest; `integer_vars` says that a prior given with `prior=` must cover the half-integer hard bounds, with `UniformBox(-0.5, 10.5)` for the values 0 to 10 | `advanced_vbmc_options.ini`, `test_options.py` | +`test_the_search_fractions_are_described_with_their_range_and_sum` (five cases), +`test_integer_variables_are_described_with_the_prior_they_need`; all six fail against the old `.ini` |

## Bit-for-bit check of item 2

`_get_search_points` was dumped before and after the change over five option sets (the shipped fractions at 8192 points, a starting cache, a high-posterior-density share, a quarter for the search cache, and five uneven fractions) at one and three variables, with the construction seeded through the global state and `vp.rng` seeded: **24 of 24 entries identical byte for byte**, the generator's state after the draws included (so the number and the order of the draws are unchanged), and the only entries the old code did not produce are the `search_cache_frac = 0.25` case, which used to raise. Script and dumps in the scratchpad (`sieve_bitwise.py`, `compare_npz.py`, `old.npz`, `new.npz`).

## Focused test runs

All from the worktree root, `-x -q -p no:cacheprovider` (the `-x` dropped for the counting runs), with the three thread variables set:

- `pyvbmc/testing/vbmc/test_vbmc_active_sample.py` — **74 passed**
- `pyvbmc/testing/variational_posterior/test_variational_posterior.py` — **130 passed**
- `pyvbmc/testing/vbmc/test_options.py` — **76 passed**
- `pyvbmc/testing/vbmc/test_vbmc_option_names.py` — **28 passed**
- `pyvbmc/testing/vbmc/test_gp_training_policy.py` — **27 passed**
- The five together at HEAD — **335 passed in 17.1 s**
- Two single tests outside my files, for the `hedge` removal and the inert registry: `test_vbmc_init.py::test_vbmc_optimstate_acq_hedge` and `test_options.py::test_inert_options_are_the_declared_options_nothing_reads` — both pass.

No full suite, no oracle command, no `golden_replay.py`, no `optimize()` run, no install, no static pickle written.

## Changelog sentences (for a user of release 1.0.4)

- **Commit 2** — "The candidate search of an iteration no longer ends the run with `ValueError: A maximum of N points should be randomly sampled…`: each source of the search set takes at most what the sources before it left, and the variational posterior draws the rest. It happened with the shipped options whenever two candidates were left to draw, and with `options['search_cache_frac'] = 0.25` whenever the starting cache left 2 or 3 candidates modulo 4." A run at the default options computes what it computed before, so no line in the "Upgrading from 1.0.4" list.
- **Commit 1** — "The description of `options['search_optimizer']` and the error that refuses another value say what the two values do: `\"cmaes\"` searches the acquisition with CMA-ES, replaced by a bounded scalar search where the problem has one variable, and `\"none\"` runs no local search, a problem of one variable included." Documentation only.
- **Commit 5** — belongs to the unreleased entry for the mode cache (a 1.0.4 `vp.mode(n_opts=k)` returned the stored mode, which the wave-5 pass already changed): "`vp.mode(n_opts=…)` runs its own search and leaves the mode stored by `vp.mode()` alone, so a later `vp.mode()` still answers with the default search's result."
- **Commit 6** — belongs to the unreleased entries for the count check and for the two option refusals: "`vp.sample` takes a count given as a NumPy scalar or a zero-dimensional array, as 1.0.4 did, and refuses a fractional count or anything that is not a single number"; "the errors that refuse `options['acq_hedge']` and a search fraction say how a saved run carrying the value is loaded, as the one for `options['search_optimizer']` does".
- **Commit 7** — belongs to the unreleased entries for the fraction check and for `integer_vars`: "the descriptions of the five search fractions state that each lies in [0, 1] and that together they claim at most the whole search set, and the description of `integer_vars` says that a prior given with `prior=` must cover the half-integer hard bounds (`UniformBox(-0.5, 10.5)` for the values 0 to 10)".
- **Commits 3 and 4** — tests only, no changelog line.

None of the seven commits changes a number of a run at the default options.

## What I stopped on

Nothing. All seven items are done as ruled.

## What else I noticed

- **`cache_frac` above one still overfills the search set.** `_get_search_points` takes `ceil(number_of_points * cache_frac)` rows from the starting cache, capped only by the cache's size; with `cache_frac > 1` and a large enough starting cache, `search_X.shape[0] > number_of_points` and the random branch is skipped, so the caller gets more candidates than it asked for. `cache_frac` is not one of the five fractions and no check refuses a value above one. Outside the ruling, so I left it; the guard that used to raise never covered this path either (it watched the random rows alone).
- **Two of item 6's repairs cannot have a failing test.** Removing `optim_state["hedge"]` and the `(jacobian == 0)` clause changes nothing observable: the option is refused before `_init_optim_state` runs (and `test_vbmc_init.py::test_vbmc_optimstate_acq_hedge` already asserts `"hedge" not in optim_state`, which still holds), and every row with a zero Jacobian already gives a non-finite quotient, so the finiteness test covers it alone. The two `x0` docstring cases of the same commit likewise pin behavior the code already had.
- **`test_each_source_draws_its_rounded_share` passes before and after by design** — it is the half of item 2 that states the counts must not move where no cap binds.
- **`np.ndim` decides the scalar test in `sample`,** as the ruling names it; a nested sequence of uneven lengths makes NumPy raise its own `ValueError` from inside `np.ndim` before the method's message is reached. Such a value is refused either way, with a `ValueError` either way.
- **numpydoc still reports RT02 for `mode`,** as it does for `get_parameters` and the other methods: the house style names the returned value (`mode : np.ndarray`) where numpydoc wants the type alone. PR02 and RT01, the two the item named, are gone.
- R3's finding 5 asked for the tie case to be closed "before the golden references are regenerated"; the new test builds the state directly in `_get_search_points` and runs no `optimize()`, so the golden traces and the oracles are untouched by this pass (I ran neither).
