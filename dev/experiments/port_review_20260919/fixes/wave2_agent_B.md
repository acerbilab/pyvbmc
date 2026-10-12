# Wave 2 fix pass, agent B: options and inputs

Raw report of a fix agent, wave 2 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, in
its own git worktree cut from `dev-port-review` at `2f2bc94`, fixed the
wave-2 findings on the options and the input handling (W2-18, W2-7, W2-8,
W2-17, W2-19, W2-20, W2-26, W2-27, W2-22, W2-15, W2-23, W2-24; it left
C-M5 to the orchestrator), one finding per commit with a test written
against the contract, running only the focused test files it touched. The
commit hashes in the report are those of the agent's worktree branch; the
orchestrator reviewed each diff and cherry-picked the commits onto
`dev-port-review`, where they carry other hashes (the plan's worklog lists
them). The text below the rule is the agent's final message, unedited;
`../verification/wave2.md` holds the ledger and the PI's dispositions.

---

# Wave-2 fix pass, agent B

Worktree: `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-a82f05cb088c2ea8b`, branch `worktree-agent-a82f05cb088c2ea8b`, cut at `2f2bc94` (`git merge --ff-only dev-port-review` gave `2f2bc94` as required). Working tree clean, nothing untracked, not pushed. `pyvbmc.__file__` from the worktree root resolves inside the worktree; every command ran with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

## `git log --oneline 2f2bc94..HEAD`

```
a4099aa docs(vbmc): say what an unseeded construction does to the global random state
1ef8b59 fix(options): keep the whole description of an option
f9ba442 fix(vbmc): remove the separate search GP
9b13afe fix(options): register the declared options the algorithm does not read
4b29201 fix(vbmc): switch the budget accounting on the values, not the arguments
e7d4033 fix(options): check the limits on iterations and evaluations
db8e5ff fix(options): refuse an option name PyVBMC does not declare, on every route
e67a3a8 fix(vbmc): replicate a bound given as a single value
da3ffd8 fix(options): read integer_vars as a mask or as indices, strictly
b705c29 fix(vbmc): settle option defaults after the options file is read
a12ba7c fix(options): apply the noisy defaults at either noise level
77e8255 fix(options): read uncertainty_handling as a boolean
```

## Commits, in order

| # | hash | finding | what it does | files | tests |
|---|---|---|---|---|---|
| 1 | `77e8255` | W2-18 | `uncertainty_handling` read as a boolean; MATLAB's error for the option explicitly off with `specify_target_noise` on; description rewritten | `options.py`, `vbmc.py`, `advanced_vbmc_options.ini`, `test_options.ini` | `test_vbmc_init.py` (+5 new, 3 changed), `test_vbmc_precomputed.py` (2 changed) |
| 2 | `a12ba7c` | W2-7 | the five noisy defaults apply whenever uncertainty handling is on | `options.py` | `test_options.py` (+3) |
| 3 | `b705c29` | W2-8 | `update_defaults` runs after the `options_path=` file is read; a file's options count as the user's | `options.py`, `vbmc.py` | `test_vbmc_init.py` (+3) |
| 4 | `da3ffd8` | W2-17 | `integer_vars` type-strict: boolean mask of length `D`, or distinct 0-based indices in range; `D` integers all 0/1 refused as ambiguous | `options.py`, `vbmc.py`, `advanced_vbmc_options.ini` | `test_vbmc_init.py` (+18 params, 1 changed), `test_vbmc_active_sample.py` (integer mask changed) |
| 5 | `e67a3a8` | W2-19 (+C-M6) | each of the four bounds given as a single value is replicated across the variables, before the plausible box is derived; the remaining error formats its dimension | `_bounds.py`, `vbmc.py` (class docstring) | `test_vbmc_init.py` (+2, `test_vbmc_bounds_check_not_vectors` changed) |
| 6 | `db8e5ff` | W2-20 | unknown option names raise on the file route and on `load(new_options=)`; `VBMC` validates against the two shipped files only | `options.py`, `vbmc.py` | new `test_vbmc_option_names.py` (5), `test_init_options_path` rewritten (+1 new) |
| 7 | `e7d4033` | W2-26 | `max_fun_evals`/`max_iter` must be positive integers; `max_iter` below `min_iter` raised to it with a warning | `options.py`, `vbmc.py` | `test_options.py` (+26 params) |
| 8 | `4b29201` | W2-27 | the budget path follows the values of `precomputed_evaluations`/`initialization_cost`, not their presence | `vbmc.py` | `test_vbmc_precomputed.py` (+6, 1 changed) |
| 9 | `9b13afe` | W2-22 | `temperature` and `diagnostics` registered inert; the guard test counts only reads through an options mapping, comments dropped | `options.py`, `advanced_vbmc_options.ini`, `test_options.py` | guard test rewritten |
| 10 | `f9ba442` | W2-15 | the search-GP branch of `optimize` removed; `separate_search_gp` registered inert | `vbmc.py`, `options.py`, `advanced_vbmc_options.ini`, `test_options.py` | `test_options.py` (+1) |
| 11 | `1ef8b59` | W2-23 | descriptions read from the raw ini lines, so a `=` or `:` inside one is kept; the wrong description texts corrected | `options.py`, `advanced_vbmc_options.ini`, `test_options.py` | `test_options.py` (+4) |
| 12 | `a4099aa` | W2-24 | the `seed` docstring says the unseeded construction draws four integers from NumPy's global state and advances it | `vbmc.py`, `dev/plans/stage1-rng-generator.md` | `test_vbmc_init.py` (+2) |

## Contract and test, per commit

1. **W2-18.** Contract: `misc/setupvars_vbmc.m:230-236` truth-tests `options.UncertaintyHandling`; `misc/setupoptions_vbmc.m:135-137` errors when it is off while `SpecifyTargetNoise` is on. Tests: `True/1/np.True_/np.int64(1)` give level 1; `False/0/np.False_/np.int64(0)/[]/()/np.array([])/None` give level 0; `"yes"/"no"/"off"/[0]/[1]/[2]/[3]/np.array([1,0])/2` raise with a message naming `uncertainty_handling` and "True or False"; the off-with-`specify_target_noise` combination raises; both on gives level 2. The `.ini` description of `uncertainty_handling` states the convention (the docs `:literal:`-include the file).
2. **W2-7.** Contract: `misc/setupoptions_vbmc.m:143-163` changes the five defaults whenever the noise handling is on. Test: both routes (`specify_target_noise=True`, `uncertainty_handling=True`) give `ceil(1.5x)` for the budget and the stability count, both in-sampling updates `True` and a `AcqFcnVIQR` search acquisition, measured against a noiseless build; a value the user gave survives; the noiseless defaults are unchanged.
3. **W2-8.** Contract: the same, with the sources of options read before the dependent defaults are settled, and a file's options treated as the user's (MATLAB's `updated` list). Test: a run configured by file gets the same `max_fun_evals` as one configured by dictionary, plus the other three defaults; `max_fun_evals = 33` written in the file survives.
4. **W2-17.** Contract: `misc/setupvars_vbmc.m:15-17` reads indices, the ledger row rules the two accepted forms. Tests: `[0,2]`, `(0,2)`, `np.array([0,2])`, `np.array([2,0])`, `np.array([True,False,True])`, `[True,False,True]` all mark variables 0 and 2 at `D=3`; `[]/()/np.array([])/None` mark none; `np.array([1,0,1])` raises asking for a boolean mask; a 1-based index vector, a negative or out-of-range index, a repeated index, a short boolean mask, a float array, a string and a 2-D array all raise naming `integer_vars`. `test_vbmc_optimstate_integer_vars` keeps its four bound checks with a boolean mask.
5. **W2-19.** Contract: `misc/boundscheck_vbmc.m:6-10` expands all four bounds; the class docstring says so (I added the same sentence to the plausible-bounds paragraph, which lacked it). Tests: `VBMC(fun, x0(2,3), -2, 2, -1, 1)` gives four `(1,3)` bounds at the right values; a `D=1` starting set without width gives a usable plausible box (the `IndexError` of C-3 is gone); a genuinely non-conforming `(2,D)` bound still raises, and the test now matches the formatted message `Bounds must match problem dimension D=3.`
6. **W2-20.** Contract: the ruling — an unknown name raises on both routes, `VBMC` validates against the two shipped files. Tests: `max_fun_eval` raises "The option max_fun_eval does not exist." from the dictionary, from a file, and from `load(new_options=)`; `max_fun_evals` is accepted through a file and through `load`.
7. **W2-26.** Contract: `misc/setupoptions_vbmc.m:109-119`. Tests: `0, -1, 7.5, -inf, nan, None` raise for either limit; `1, 40, 40.0, np.int64(40), np.inf` pass (MATLAB compares with `round`, so an integer-valued float and `Inf` pass); `max_iter=2` with `min_iter=7` becomes 7 with a warning naming both; equal values are left alone and silent.
8. **W2-27.** Contract: `dev/plans/pymc-target-adapter.md` (the sentinel lets an explicit `None`/`0` override the adapter's values; it says nothing about the budget switch). Tests: the four ways of writing the defaults (omitted, `precomputed_evaluations=None`, `initialization_cost=0`, both) all give `_budget_active is False` and the ordinary allowance; `initialization_cost=3` or actual evaluations turn it on.
9. **W2-22.** Contract: the ruling — the two names are registered, and the guard counts reads through an options object. The rewritten guard matches `options["name"]`, `options.get("name")`, `options.eval("name", …)` and the same through `self` inside `Options`, over comment-stripped, whitespace-flattened sources of the whole package except `testing/`. **This test fails in my worktree, by design, and its only complaint is `entropy_force_switch`** (`Extra items in the left set: 'entropy_force_switch'`); it passes once agent A's W2-10 makes `optimize` read that option from `self.options`. I did not register it. The `optim_state["temperature"]` reads of `whitening.py` are untouched.
10. **W2-15.** Contract: `vbmc.m:471`, `:638-648` keep a second hyperparameter struct and discard the returned state; the ruling removes the branch and keeps the option declared and inert. Test: setting `separate_search_gp=True` is reported as having no effect (the inert-registry contract). The branch, its timer and its `sn2_hpd` write are gone; `active_sample` receives `self.gp`.
11. **W2-23.** Contract: the description of an option is the comment line above it, in full. Tests: a temporary ini whose two descriptions contain a `:` and a `=` stores both whole; the shipped descriptions of `search_optimizer`, `stable_gp_samples` and `upper_gp_length_factor` are stored as written. Corrections in the file: `upper_gp_length_factor` and `temperature` as ruled, plus the other seventeen places where a substitution had turned "on" into "True" (`tol_gp_var`, `tol_gp_var_mcmc`, `elcbo_impro_weight`, `tol_sd`, `tol_skl`, `box_search_frac`, `gp_retrain_threshold`, `gp_quadratic_mean_bound`, `acq_hedge`, `scale_lower_bound`, `optimistic_variational_bound`, `active_importance_sampling_box_samples`, `tol_bound_x`, `recompute_lcb_max`, `warp_tol_reliability`, `warp_roto_corr_thresh`, `warp_tol_sd_base`), each checked against the corresponding `defopts` line of `vbmc.m`. "True mean"/"True covariance" of `true_mean`/`true_cov` are genuine and were left.
12. **W2-24.** Contract: `pyvbmc/rng.py: get_rng` draws four `uint32` from the global legacy state when `seed is None`. Tests: a construction with `seed=42` leaves the global state where it was; an unseeded one leaves it exactly where four such draws leave it. **These two tests pass before the fix as well** — the defect was in the text, and they are the gate against the docstring drifting from the behavior again. I also corrected the phrase "only read once" in `dev/plans/stage1-rng-generator.md`, the one `dev/` edit permitted.

## Commands run and results

Focused files only, always `-x`-free `-q -p no:cacheprovider --basetemp=<scratch>/pytest_tmp`, from the worktree root:

- `pytest pyvbmc/testing/vbmc/test_options.py pyvbmc/testing/vbmc/test_vbmc_init.py pyvbmc/testing/vbmc/test_vbmc_precomputed.py pyvbmc/testing/vbmc/test_vbmc_option_names.py pyvbmc/testing/vbmc/test_vbmc_active_sample.py::test_repeated_observation_is_exact_with_integer_vars` → **1 failed, 225 passed**; the failure is the W2-22 guard test with `entropy_force_switch` as its only complaint.
- Every commit's tests were run against the unfixed source first (source files set aside with a patch, restored afterwards) and seen to fail, except W2-24 as noted above. Counts seen failing beforehand: W2-18 20, W2-7 2, W2-8 3, W2-17 15, W2-19 3, W2-20 2, W2-26 13, W2-27 3, W2-22 1 (the guard, then 1 remaining after), W2-15 1, W2-23 4.
- No `optimize()` run, no whole directory, no oracle or golden script, no install.

## No number of a default or noisy run moves

I checked this directly rather than by reasoning alone. A script builds a `D=3` default run and a `D=3` run with `options={"specify_target_noise": True}`, seed 17, and prints every option value, every `optim_state` entry, the four bound arrays, `x0`, the initial `vp.mu/sigma/lambd/w`, the generator's bit-generator state and the three budget fields at full precision. Run against the worktree and against the base checkout at `2f2bc94`, the two dumps differ **only** in the `pyvbmc.__file__` line. I also checked that the new ini parser returns the identical list of `(name, value expression)` pairs for all four shipped and test ini files (169, 14, 5, 3 entries, equal). The base checkout was only imported, never written.

Two behavior notes that are not numbers:
- With a `D=1` bound given as a length-1 vector, the log line "Reshaping lower bounds to (1, 1)" no longer appears, because the expansion has already given it the right shape.
- An option set through `options_path=` now appears under "User Options" in `str(options)` and, if it is an inert option, is reported as having no effect. Both follow from treating a file's options as the user's.

## Not done

- **C-M5 (frozen `Options` refuses removal)** is **not implemented**. The fix itself is four lines — guard `Options.__delitem__` with the same `is_initialized` check and `force=` parameter that `__setitem__` has; I verified that `pop`, `popitem` and `clear` resolve to `MutableMapping` in the MRO and therefore go through `__delitem__`, and that `update`/`setdefault` go through `__setitem__`, so `load`'s path (which clears `is_initialized` around `options.update`) is unaffected. What stops me is the collateral: four tests build a legacy state by deleting a key from an **initialized** options object and would start raising:
  - `pyvbmc/testing/vbmc/test_vbmc_finalboost.py:328` `del vbmc.options["tol_elcbo_boost"]` (agent A's file)
  - `pyvbmc/testing/vbmc/test_vbmc_save_and_load.py:229` `del vbmc.options["vectorized_target"]` (agent C's file)
  - `pyvbmc/testing/calibration/test_lifecycle.py:303` `del vbmc.options["performance_calibration"]`
  - `pyvbmc/testing/vbmc/test_runtime_tips.py:377` and `:386` `del legacy.options["show_tips"]`

  Each needs `vbmc.options.__delitem__("<name>", force=True)` instead. Two of those files are explicitly not mine and the other two are outside my allowlist, so per the working rule I left the finding. Suggested commit message subject: `fix(options): refuse removal of an option after initialization`, with a test in `test_options.py` asserting that `pop` and `del` raise `AttributeError` on an initialized object while `test_del` (which pops before validation) keeps working.

## Edits needed outside my area

Only the five C-M5 call sites above. Nothing else.

## Observations for the PI

- `Options.update_defaults` builds its five replacement values eagerly, before checking which of them the user already set. With `specify_target_noise=True` **and** `max_fun_evals=np.inf` this raises `OverflowError: cannot convert float infinity to integer` from `ceil(inf)`, even though the user set the budget and the default would not have been used. This predates the pass and is outside every ruled finding, so I left it; the cure is to compute each replacement only for the keys not in `useroptions`. A finite budget, noisy or not, and an infinite `max_iter` are all fine (checked).
- MATLAB's `misc/setupoptions_vbmc.m:120-124` also raises `MaxFunEvals` to `MinFunEvals`, but its assignment is a no-op (`options.MinFunEvals = options.MinFunEvals;`), so the warning fires and nothing changes. The ruling named only the `max_iter` clause and I ported only that; the MATLAB-side defect may be worth a line in `matlab_side_defects.md`.
- `Options` gained a small public surface that `AGENTS.md` may want to mention: `integer_vars_mask(D)`, `uncertainty_handling_on()`, `validate_run_limits()`, `validate_supplied_option_names(names)`, `load_options_file(..., as_user_options=)`, and module-level `SHIPPED_OPTIONS_PATHS` / `declared_option_names()`.

## Record updates the orchestrator should make

- **`AGENTS.md`, the "Options are layered" bullet.** It currently says "Unknown keys raise at validation"; that is now true of all three routes (dictionary, `options_path=` file, `load(new_options=)`), validated against the two shipped files. Worth adding: the file of `options_path=` is read before the dependent defaults are settled and its options count as the user's; the noisy defaults follow uncertainty handling at either level, not only `specify_target_noise`; `uncertainty_handling`, `integer_vars`, `max_fun_evals` and `max_iter` are now type- and value-checked.
- **Known-differences sheet.** New or changed entries: the stricter option handling (`uncertainty_handling` boolean, `integer_vars` mask-or-indices, the run-limit checks, scalar bound replication — all four now match MATLAB, so some of these may be sheet *removals*); `separate_search_gp` removed from the loop and registered inert; the "Declared options that nothing reads" entry gains `temperature`, `diagnostics` and `separate_search_gp` and its closing claim about the guard's matching rule needs rewriting; the posterior-tempering entry should say that `temperature` is read nowhere (the `whitening.py` reads are of an `optim_state` key nothing writes).
- **Counterpart map.** `misc/setupvars_vbmc.m` (the `integer_vars` row: indices are now accepted), `misc/boundscheck_vbmc.m` (scalar expansion is now ported, so the row needs no "difference" note), `misc/setupoptions_vbmc.m:109-163` (the run-limit checks and the noisy defaults are now ported).
- **Documentation.** `docsrc/source/api/options/vbmc_options.rst` `:literal:`-includes both `.ini` files, so the rewritten descriptions ship with no `.rst` change. I searched `docsrc/`, `examples/` and `README.md` for the options whose accepted values changed: nothing uses `uncertainty_handling`, `integer_vars` or `separate_search_gp`; `docsrc/source/faq.md:686` and `examples/scripts/pyvbmc_example_3_full_code.py:94-99` use `load(new_options=...)` with declared names only, so W2-20 leaves them working. No notebook needs editing.
