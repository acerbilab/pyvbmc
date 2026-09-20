# Wave 2 verification, group C: setup, options, inputs

Raw report of a verifier, wave 2 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent,
read-only on the repositories, verified the wave-2 findings on the setup,
the options and the input handling (reports `../reviews/P1b_internal.md`
and `../reviews/P1b_comparison.md`), without `optimize()` runs. Its
scripts are kept as `scripts/wave2_C_*.py`, with the repository root
derived from the script's location and the one script that writes files
pointed at a temporary directory. The text below the rule is the agent's
final message, unedited; `wave2.md` holds the consolidated ledger and the
PI's dispositions. Two findings of those reports were verified by the
orchestrator and are not in this file: the noisy-target defaults and the
budget copy after a load.

---

# Wave-2 verification, slice C: the P1b findings (setup, options, inputs)

Verified 2026-09-20 on `dev-port-review` at `d6c3827` against MATLAB VBMC at
`396d649`. All reproductions are in
`C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\b8a4c56e-a31c-47b4-87d0-66a2c9861ded\scratchpad\port_review\verify_wave2_C`
(`f1_integer_vars.py`, `f2_uncertainty_handling.py`, `f3_scalar_bounds.py`,
`f4_f5_options_path_and_load.py`, `f5b_warp_x0.py`, `f6_inert_options.py`,
`f7_descriptions.py`, `f8_f11_rng_and_minors.py`, `f9_acq_forms.py`,
`f10_budget_and_maxiter.py`, `f11b_logger_display.py`,
`f11c_dead_optim_state.py`). Nothing in `pyvbmc`, `../gpyreg` or `../vbmc` was
written; no `optimize()` run, no test suite. `pyvbmc.__file__` printed as
`C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\__init__.py` in every script.
`gh` is authenticated and was used read-only for pull request 102.

## Ledger

| id | reports | statement | verdict (class) | fires at defaults? | how verified | dating and rationale |
|---|---|---|---|---|---|---|
| C-1 | P1b int F7, P1b cmp F3 | `_init_optim_state` reads `integer_vars` as a length-`D` mask (`integeridx = options.get("integer_vars") != 0`), where `misc/setupvars_vbmc.m:14-24` reads 1-based **indices**; a plain Python list makes `!= 0` the scalar `True`, so every variable is marked integer, silently | confirmed port discrepancy | no (`integer_vars = []`); on every run that sets the option | `f1_integer_vars.py`, `f2_uncertainty_handling.py` (ten value forms at `D=3`); both sources read | Python: `489598a` 2021-06-03 Marlon "feat: add init to VBMC" introduced the mask reading verbatim; only renamed in `d9bfe16` (2022-08-28). Never matched MATLAB, whose block last changed `28f4fee` (2019-07-21), before the port. Rationale: none found |
| C-2 | P1b cmp F4 | `uncertainty_handling` is tested with `len(...) > 0`, where `misc/setupvars_vbmc.m:232` truth-tests a boolean produced by `utils/evalbool.m`: `True`/`1`/`0`/`False`/`None` raise `TypeError`, and `[0]`, `[False]`, `'no'`, `'off'` all select level 1 | confirmed port discrepancy | no (`[]` → level 0 correctly); on any run that sets the option | `f2_uncertainty_handling.py` (17 value forms) | Python: length test since `489598a` (2021-06-03) — then with **inverted** polarity (`== 0` → level 1), corrected to `> 0` in `aa45473` 2021-07-26 Mikko Aarnos "feat: added GP training for VBMC (#19)". Never a truth test. MATLAB's since `d9822e3` (2019-09-17). Rationale: none found |
| C-3 | P1b int F6, P1b cmp F7 | `_normalize_bounds` never replicates a scalar bound, so `VBMC(f, x0, -10, 10, ...)` raises for every `D > 1` although the class docstring (`vbmc.py:98-104`) and `misc/boundscheck_vbmc.m:6-10` say it is replicated; the message is the unformatted `('Bounds must match problem dimension D=%d.', 5)`; at `D == 1` the scalar survives as a 0-d array and raises `IndexError` in the degenerate plausible-bound repair | confirmed port discrepancy | yes, whenever a caller writes a scalar bound | `f3_scalar_bounds.py` (`D = 1,2,3,5`; the `D=1` multi-row cases) | Docstring promise since `489598a` (2021-06-03); the code never expanded a scalar. Reshape-and-`ValueError` path since `d34fc15` 2022-08-30 Bobby-Huggins "Spring cleaning (#93)" (which formatted the message with an f-string); the unformatted two-argument form since `dd0e17e` 2022-11-23 "Pip/Conda (#118)". `_bounds.py` extracted in `975e45a` (2026-09-16). MATLAB expansion present since the first commit `a1680e3` (2018-05-04). Rationale: none found |
| C-4 | P1b int F12 | Option names are validated only for the `options=` dict: a name written in an `options_path=` file is declared by construction (the path is passed to `validate_option_names`), and `load(new_options=)` never validates | confirmed Python-only inconsistency (not a port discrepancy: MATLAB validates no option name at all) | no; on the two routes when the user misspells | `f4_f5_options_path_and_load.py`: `options={"max_fun_eval":123}` → `ValueError`; the same name plus `completely_made_up` through `options_path` → accepted and stored; `load(new_options={"max_fun_eval":999})` → accepted, `max_fun_evals` unchanged | `validate_option_names` since `4b94e1e` 2021-07-17 (#18), with `options_path` among the allowed paths from the start; `load(new_options=)` since `ad96d50` 2023-03-08 (#131). MATLAB `misc/setupoptions_vbmc.m:9-26` copies defaults over the user struct and leaves unknown fields alone. Rationale: none found |
| C-5 | P1b int F10 | After `load`, `vbmc.parameter_transformer`, `vp.parameter_transformer` and `function_logger.parameter_transformer` are three distinct objects, and `vbmc.parameter_transformer` is the pickled live one, not the selected iteration's | confirmed Python-only defect (state restoration; no MATLAB counterpart — MATLAB has no save/load) | yes for any loaded finished run; the wrong-map consequence only after a warp | `f4_f5_options_path_and_load.py` (fresh round trip keeps identity because a fresh instance has no history; the stored `test_vbmc_save_static.pkl` breaks it three ways, equal by value); `f5b_warp_x0.py` | `load` restoring `vp`/`function_logger`/`optim_state` from the history since `ad96d50` 2023-03-08 (#131). Rationale: none found. `AGENTS.md` states the sharing invariant; `test_vbmc_optimize.py:20-25` asserts it with `is` at the end of every live iteration, nothing after a load |
| C-6 | P1a int F9, P1b int F8, P1b cmp F6/F10 | 25 declared options are read through no options mapping; `INERT_OPTIONS` lists 22. `diagnostics`, `temperature` and `entropy_force_switch` are missing, and the guard test cannot see them because it matches the quoted name anywhere in the package | confirmed (bookkeeping); `temperature` additionally a port discrepancy (MATLAB validates `T` and stores it, `misc/setupvars_vbmc.m:249-255`, and reads `optimState.temperature` in six files) | no numerical effect at the defaults; a user setting one of the three gets silence instead of the "no effect" warning | `f6_inert_options.py` recomputes both sets independently; the test rule read at `test_options.py:183-205` | `INERT_OPTIONS` added `a63b17a`/`0d9a422` (2026-09-19). The sheet's P7 entry and the closing claim of its "Declared options that nothing reads" entry are both inaccurate, as the reviewers say |
| C-7 | P1b int F11 | `_read_config_file` parses the `.ini` files with configparser's default delimiters, so a `# description` containing `=` or `:` keeps only the part before it. Exactly 8 of the 183 declared options are affected, all in the advanced file | confirmed Python-only defect (documentation surface); no MATLAB counterpart (MATLAB's comments are stripped, not stored) | yes, but no numerical effect | `f7_descriptions.py` lists all 8 with stored and full text | `comment_prefixes=""` parser since `d72c7df` 2021-05-29 "feat: generalize options with config files". Rationale: none found |
| C-8 | P1b int F9 | `VBMC(seed=None)` draws four `uint32` from NumPy's global legacy state, advancing it, against the `seed` docstring's "NumPy's global random state is never written" | confirmed documentation defect; the behaviour is deliberate and pinned by a test | yes, on every unseeded construction | `f8_f11_rng_and_minors.py`: MT19937 position 624 → 4, key rewritten; `seed=42` and `seed=Generator` leave it untouched | `pyvbmc/rng.py` `d02c517` (2026-09-02), unchanged by `c4313a3` (2026-09-05). **Recorded decision**, `dev/plans/stage1-rng-generator.md` §3: "**`seed=None` derives the generator from the legacy global stream** (four `uint32` draws from `np.random`)", and §"the one seam": "With `seed=None` the global state is never written, only read once". `test_vbmc_seed.py::test_seed_none_does_not_reseed_global_state` pins the four draws |
| C-9 | P1b cmp F5 | The default search acquisition is `[AcqFcnLog()]` where `vbmc.m:213` has `@acqf_vbmc`. The two are monotone transformations with the same exact-arithmetic minimizer and consistently applied regularization; the plain form loses resolution by underflow | intentional difference, missing from the sheet | yes, every run | `f9_acq_forms.py` on all 8 oracle fixtures: identical ordering on the non-saturated finite entries everywhere; same argmin in 7 of 8; on `normal_D2_singlesample` the plain form underflows on **all 512** candidates | Python matched MATLAB (`[AcqFcn()]`) until `e20d081` 2022-09-21 Bobby-Huggins "Acq fcn log (#102)". **Recorded reason**, PR #102 body: "Changes default acquisition function for non-noisy targets from `AcqFcn` to log-parametrized version `AcqFcnLog`, in order to avoid overflow warnings in `exp()`." MATLAB's default unchanged since `0b51bd0` (2020-02-29) |
| C-10a | P1b cmp F8 | MATLAB errors on a non-positive or non-integer `MaxFunEvals`/`MaxIter` and raises `MaxIter` to `MinIter` with a warning (`misc/setupoptions_vbmc.m:109-124`); PyVBMC has neither check | confirmed port discrepancy (validation only) | no | `f10_budget_and_maxiter.py`: `max_iter` 0, 2.5, −3 and `max_fun_evals` 0, −1, 7.5 all accepted silently; `max_iter=1, min_iter=5` left as given | MATLAB checks present since `a1680e3` (2018-05-04); `f3f8b80` (2021-03-13) moved them behind `skipextra_flag` without changing them. PyVBMC never had them. Rationale: none found |
| C-10b | P1b int F13 | `_budget_active = bool(precomputed_was_provided or initialization_cost_was_provided)` is set from an argument's **presence**, so `initialization_cost=0` or `precomputed_evaluations=None` — the documented defaults — take the opted-in budget path | confirmed Python-only defect (surprising API), partly anticipated by a recorded design note | no | `f10_budget_and_maxiter.py`: `_budget_active` False when omitted, True for either explicit default | `6769a9a` 2026-09-16 "feat(vbmc): reuse precomputed evaluations and charge initialization". `dev/plans/pymc-target-adapter.md:394` records the sentinel: "Use an internal omitted-argument sentinel for these two keywords so omission is distinct from an explicit `None` or zero; **for ordinary callables the effective defaults remain `None` and zero**" — the *values* are equated, the budget switch is not. No note addresses the explicit-default case |

### Minor observations, P1b internal

| id | statement | verdict | fires at defaults? | how verified | dating / note |
|---|---|---|---|---|---|
| C-M1 | `VBMC.__str__` reads the GP from `vp.gp`, which does not exist, so it always prints `Gaussian process = None`; `log-prior` and `prior sampler` read attributes `__init__` never assigns, so both always print `None` | confirmed Python-only defect; **a real defect**, the only one of the internal minors I would call so besides C-M5 | yes | `f8_f11_rng_and_minors.py`: all three `None` even with `log_prior=`/`sample_prior=` supplied | `4f149c5` 2022-09-30 "Printing (#103)". `test_vbmc_init.py:1192-1195` calls `__str__` and asserts nothing |
| C-M2 | `_init_logger` removes handlers while iterating `logger.handlers` (skipping one when two match) and reads `handler.baseFilename` on every handler of the process-wide `"VBMC"` logger; `log_file_level` rejects any level outside `{0,10,20,30,40,50}` | confirmed Python-only defect; leftover-grade except the `AttributeError`, which is reachable from ordinary user logging configuration | no (needs `log_file_name`) | `f11b_logger_display.py`: two duplicate `FileHandler`s → one stale survives; a `StreamHandler` on `"VBMC"` → `AttributeError: 'StreamHandler' object has no attribute 'baseFilename'`; `logging.DEBUG+5` → `ValueError` | `97d19a4` 2022-03-10 "Improved Logging (#71)" |
| C-M3 | Fourteen `optim_state` entries are written and never read: `H`, `hedge`, `int_mean_fun`, `iter_list`, `lb_orig`, `ub_orig`, `lcb_max_vec`, `proposal_fcn`, `pruned`, `redo_roto_scaling`, `vp_repo`, `warmup_stable_count`, `initialization_cost`, `max_fun_evals_total` | confirmed; leftovers | no | `f11c_dead_optim_state.py` recomputes the set: exactly those 14 (my scanner's fifteenth, `acq_info`, is a commented-out line) | `plb_orig`/`pub_orig` *are* read by the warp, as the reviewer says |
| C-M4 | `iteration_history["data_trim_list"]` is declared and never recorded | confirmed; leftover | yes (stays `None`) | `f8_f11_rng_and_minors.py` | same as comparison M9 |
| C-M5 | `Options` freezes `__setitem__` but not `__delitem__`, so `vbmc.options.pop("max_iter")` succeeds on an initialized object and removes a key the algorithm reads | confirmed Python-only defect; **a real defect** (the freeze has a hole that leaves the object unusable) | no | `f8_f11_rng_and_minors.py`: `__setitem__` → `AttributeError`, `pop("max_iter")` → returns 200 and the key is gone | freeze added `4b94e1e` (2021-07-17) |
| C-M6 | `_bounds.py:155-157` raises `ValueError("Bounds must match problem dimension D=%d.", D)` with the template and the argument as two arguments | confirmed Python-only defect (message never formatted) | only on the failing path | `f3_scalar_bounds.py` prints `ValueError('Bounds must match problem dimension D=%d.', 5)` | `dd0e17e` 2022-11-23 "Pip/Conda (#118)" replaced the correct f-string of `d34fc15` |
| C-M7 | `_bounds.py:252-261` warns "…not inside the provided plausible bounds… Expanding the plausible bounds" when `x0` is exactly **on** a plausible bound, where the following `minimum`/`maximum` expands nothing | confirmed; leftover (misleading message only) | on that input | `f8_f11_rng_and_minors.py`: `x0=[-1, 0]` with `plb=[-1,-1]` warns, `plb`/`pub` unchanged | the `<=`/`>=` test is MATLAB's own (`boundscheck_vbmc.m:120-126` uses the same); shared wording issue, no numerical difference |
| C-M8 | `VBMC.__init__` calls `x0.ndim` before converting, so a list or a float raises `AttributeError` | confirmed Python-only defect; leftover-grade (a hard, if unhelpful, failure) | on that input | `f8_f11_rng_and_minors.py`: `'list' object has no attribute 'ndim'`, `'float' object has no attribute 'ndim'` | `489598a` (2021-06-03) |
| C-M9 | One non-finite entry anywhere in `x0` replaces the whole starting set with one row at the plausible centre | confirmed; **shared with MATLAB** (`misc/setupvars_vbmc.m:7-12`: `if any(~isfinite(x0)); x0 = 0.5*(PLB+PUB)`), as the comparison report says | on that input | `f8_f11_rng_and_minors.py`: three rows, one `NaN` → `x0` shape `(1,2)`, cache `[[0. 0.]]` | `489598a` (2021-06-03); faithful port |
| C-M10 | `IterationHistory.__setitem__` deep-copies, so `load`'s truncation loop copies the whole history a second time | confirmed; cost only | on every load | `f11c_dead_optim_state.py` | no correctness effect |
| C-M11 | `vbmc.py:1409`/`:1438` read `options.get("varactivesample")`, a name no `.ini` declares | confirmed; the branch is dead twice over (`separate_search_gp` is `False` by default, and `:1438` would `sys.exit`) | no | `f6_inert_options.py` lists the undeclared reads; the sites read | see C-6's second list |

### Minor observations, P1b comparison (M1–M12)

| id | statement | verdict | note |
|---|---|---|---|
| C-C1 (M1) | MATLAB errors on a non-positive `NoiseSize`; PyVBMC has no check but clamps | confirmed; harmless | `gaussian_process_train.py:342-350`, `max(options["noise_size"], min_noise)` read; a negative value becomes `tol_gp_noise` |
| C-C2 (M2) | MATLAB errors on `SpecifyTargetNoise` with `UncertaintyHandling` explicitly off and warns when `NoiseSize` is also given; PyVBMC has neither | confirmed; the contradictory configuration cannot arise in PyVBMC because level 2 is decided from `specify_target_noise` first (`vbmc.py:1016-1017`) | |
| C-C3 (M3) | `misc/setupoptions_vbmc.m:123` is the self-assignment `options.MinFunEvals = options.MinFunEvals;` inside the `MaxFunEvals < MinFunEvals` branch, so the announced change is never applied | confirmed MATLAB-side defect, not inherited | read at `misc/setupoptions_vbmc.m:120-124` |
| C-C4 (M4) | `misc/setupoptions_vbmc.m:47` has `'ConstrainedGPMean''FeatureTest'`, one string (MATLAB's doubled quote is an escape), so neither name is in `evalfields` | confirmed MATLAB-side defect, not inherited | read at `:47`; both names unread in MATLAB |
| C-C5 (M5) | `misc/boundscheck_vbmc.m:27` computes `idx = any(PLB == PUB)`, a scalar, so `PLB(idx) = LB(idx)` repairs only coordinate 1; `_bounds.py:114-117` repairs every degenerate coordinate | confirmed MATLAB-side defect, not inherited | `f3_scalar_bounds.py`, `D=3` with two rows differing in coordinates 0 and 1: Python repairs coordinate 2 alone |
| C-C6 (M6) | Two bound-repair warnings differ at the exact-equality boundary, where MATLAB's subsequent repair is a no-op | confirmed; warning text only | `boundscheck_vbmc.m:99`/`:113` against `_bounds.py:213`/`:235` read |
| C-C7 (M7) | `Options.eval` calls a callable option by keyword (`options.py:326`), MATLAB's `evaloption_vbmc` positionally | confirmed Python-only interface difference; leftover-grade | `f8_f11_rng_and_minors.py`: `options={"ns_ent": lambda n: ...}` is accepted at construction and raises `TypeError: <lambda>() got an unexpected keyword argument 'K'` at `options.eval`. Since `3bfb485` (2021-07-06) |
| C-C8 (M8) | Accepted `display` values are `off`/`iter`/`full` where MATLAB's are `iter`/`notify`/`final`/`off`; both fall back to the iteration level | confirmed Python-only interface choice, not on the sheet | `f11b_logger_display.py`: `off`→WARN, `iter`→INFO, `full`→DEBUG, anything else→INFO. Since `c477a7b` (2021-11-01) |
| C-C9 (M9) | `data_trim_list` declared as a history key, never recorded; MATLAB keeps it in `optimState` only | confirmed | same as C-M4 |
| C-C10 (M10) | `optim_state["vp_repo"]` is written at `vbmc.py:1663`, never initialized, never read; MATLAB initializes it at `setupvars_vbmc.m:261` and reads it in the unported `misc/vpsieve_vbmc.m` | confirmed | both sites read |
| C-C11 (M11) | `trinfo.x0_orig` (`misc/setupvars_vbmc.m:45`) is written and never read in MATLAB; PyVBMC's transformer correctly has no counterpart | confirmed MATLAB-side leftover, not inherited | `grep -rn x0_orig --include=*.m` returns that one line |
| C-C12 (M12) | MATLAB's integer-variable bound check reduces to "the bound is not finite": `(~isfinite(LB(d)) && floor(LB(d)) ~= 0.5)` — `floor` of any real is an integer (and `floor(±Inf)`/`floor(NaN)` are not 0.5 either), so the half-integer requirement its own message states is never tested. PyVBMC implements the documented intent and is strictly stricter | confirmed MATLAB-side defect, not inherited | `f1_integer_vars.py`: PyVBMC rejects plain integer bounds 0/10 and rejects infinite bounds; `misc/setupvars_vbmc.m:19-22` read |

---

## Supporting detail

### C-1 `integer_vars`

`pyvbmc/vbmc/vbmc.py:877-892`:

```python
optim_state["integer_vars"] = np.full(self.D, False)
if len(self.options.get("integer_vars")) > 0:
    integeridx = self.options.get("integer_vars") != 0
    optim_state["integer_vars"][integeridx] = True
```

MATLAB `misc/setupvars_vbmc.m:15-17`:

```matlab
optimState.integervars = false(1, nvars);
if ~isempty(options.IntegerVars)
    optimState.integervars(options.IntegerVars) = true;
```

Reproduction at `D = 3`, hard bounds `-0.5`/`10.5` in every coordinate:

```
default []                                    -> [False False False]
list [0, 2]  (0-based indices)                -> [ True  True  True]
list [1, 3]  (MATLAB 1-based indices)         -> [ True  True  True]
list [0, 1, 0]  (mask written as a list)      -> [ True  True  True]
list [True, False, True]                      -> [ True  True  True]
np.array([0, 2])                              -> IndexError: boolean index did not match ...
np.array([1, 3])                              -> IndexError: ...
np.array([1, 0, 1])  (mask)                   -> [ True False  True]
np.array([True, False, True])  (mask)         -> [ True False  True]
np.array([0, 1, 2])  (all three, 0-based)     -> [False  True  True]
```

`[0, 2] != 0` is the scalar `True`; `np.full(3, False)[True] = True` sets every
entry and `self.lower_bounds[:, True]` has shape `(1, 1, 3)`, so the bound check
covers every coordinate too. The only working spelling is a length-`D` NumPy
array, and even then a genuine index array of full length is silently wrong
(`np.array([0, 1, 2])`, meaning "all three" under either index convention, marks
only variables 1 and 2 because index 0 is falsy). The consequence is that
`AbstractAcqFcn._real2int` (`abstract_acq_fcn.py:279-283`, reached from `:86` and
from `active_sample.py:384`, `:489`, `:627`) rounds every coordinate of every
acquisition candidate to the integer grid.

**Convention.** The `.ini` description is MATLAB's verbatim, "Array with indices
of integer variables", and MATLAB's indices are 1-based (`IntegerVars = [1 3]`
marks variables 1 and 3). A Python user reading that description would write
0-based indices; neither convention works, and neither is documented anywhere in
PyVBMC. The only statement of the mask convention is the test.

**MATLAB's half-integer check never fires.** `misc/setupvars_vbmc.m:19-22`:

```matlab
if (~isfinite(LB(d)) && floor(LB(d)) ~= 0.5) || ...
        (~isfinite(UB(d)) && floor(UB(d)) ~= 0.5)
```

`floor(x) ~= 0.5` is true for every real `x` and for `±Inf` and `NaN`, so each
conjunct reduces to its `~isfinite` half: MATLAB errors exactly when a bound of a
declared integer variable is infinite, and never checks the half-integer
placement its own message describes. PyVBMC's `vbmc.py:882-892` tests both,
which is the documented intent (verified: PyVBMC rejects hard bounds `0`/`10`).

**Error in P1b internal F7.** The measured line reads "`[0,1,0]` and
`[True,False,True]` work as masks; `[1]`, `[0,1]` and `[0,2]` raise
`IndexError`". As plain Python lists all five mark every variable (see
`f2_uncertainty_handling.py`'s addendum); the `IndexError` arises only for NumPy
arrays of the wrong length. The internal reviewer's script must have wrapped them
in `np.array`. The P1b comparison report F3 has this right.

**Scope note.** A fix touches `vbmc.py:878-892`, the `.ini` description (a
convention has to be chosen and stated), `test_vbmc_init.py:397-425`
(`test_vbmc_optimstate_integer_vars`, which builds a mask and asserts the mask),
and the counterpart-map row for `misc/setupvars_vbmc.m`, which currently reads
"ported".

### C-2 `uncertainty_handling`

`pyvbmc/vbmc/vbmc.py:1016-1021` against `misc/setupvars_vbmc.m:230-236`
(`elseif options.UncertaintyHandling`, a truth test on the result of
`utils/evalbool.m`, which maps `'yes'/'on'/'true'/'1 '` to 1 and
`'no'/'off'/'false'/'0 '` to 0). Reproduction:

```
default (absent)   -> level 0    []                 -> level 0
np.array([])       -> level 0    True  -> TypeError: object of type 'bool' has no len()
False -> TypeError               1 / 0 / 2 -> TypeError: object of type 'int' has no len()
None  -> TypeError               np.True_ -> TypeError
[0]   -> level 1  gp_noise_fun=[1, 2, 0]
[False] -> level 1               [1] -> level 1        [2] -> level 1
'yes' -> level 1                 'no' -> level 1       'off' -> level 1
```

Two points sharper than the report's: `'no'` and `'off'` select level 1, the
opposite of what `evalbool` gives them; and `[2]` selects level 1, so the
description's "2: user-provided noise" is unreachable through this option at all
(level 2 comes only from `specify_target_noise`).

**What the user is told.** The only documentation is the `.ini` description
"Explicit noise handling (0: none; 1: unknown noise level; 2: user-provided
noise)", which `docsrc/source/api/options/vbmc_options.rst` includes literally
(so the docs show the full sentence) but which `Options.descriptions` stores as
"Explicit noise handling (0" — see C-7. No example notebook sets it; example 6
uses `specify_target_noise` only. The tests pass `[3]` and `[]`
(`test_vbmc_init.py:468-484`, `:604-613`) and `[1]`
(`test_vbmc_precomputed.py:123`, `:152`).

**Scope note.** A fix touches `vbmc.py:1016-1021` and those four test sites, and
interacts with the orchestrator's finding on `update_defaults` (P1b comparison
F1): both concern what "uncertainty handling is on" means.

### C-3 Scalar bounds

`_normalize_bounds` (`pyvbmc/vbmc/_bounds.py:136-157`) goes straight from
`np.atleast_1d` to `reshape((1, D))`. Reproduction:

```
D=1, scalar LB/UB, array PLB/PUB     OK  lb=[[-10.]] ub=[[10.]]
D=2, scalar LB/UB, array PLB/PUB     ValueError: ('Bounds must match problem dimension D=%d.', 2)
D=3 / D=5                            ValueError: ('Bounds must match problem dimension D=%d.', 3 / 5)
D=1, all four bounds scalar          OK
D=3, all four bounds scalar          ValueError: ('Bounds must match problem dimension D=%d.', 3)
```

MATLAB `misc/boundscheck_vbmc.m:6-10` expands all four.

**The `D == 1` edge.** The scalar reaches the plausible-bound derivation as a 0-d
array, and `plausible_lower_bounds[idx] = lower_bounds[idx]` (`_bounds.py:116-117`)
then raises. It needs a **degenerate** coordinate, not merely several rows:

```
D=1, x0 = [[1.],[3.]], scalar LB/UB, no PLB/PUB     OK   plb=[[0.]] pub=[[4.]]
D=1, x0 = [[2.],[2.]], scalar LB/UB, no PLB/PUB     IndexError: too many indices for
                                                    array: array is 0-dimensional
D=1, x0 = [[2.],[2.]], array  LB/UB, no PLB/PUB     OK
```

P1b internal F6 states the condition as "with more than one starting row and no
plausible bounds supplied", omitting the zero-width requirement; that is an
overstatement, minor.

**Scope note.** A fix touches the expansion in `_bounds.py:136-157` and the
message at `:155-157`; the `D == 1` `IndexError` disappears with the expansion,
so `:114-117` needs no separate change. No test pins the rejection, so none has
to be changed. The counterpart-map row for `misc/boundscheck_vbmc.m` says
"ported" and would need a note if the difference is kept.

### C-4 Option-name validation

`Options.validate_option_names` checks each key of the object against the union
of the paths it is handed, and `VBMC.__init__:312-324` hands it `options_path`,
so anything the user writes in their own file is declared by construction.
`load` (`vbmc.py:2956-2959`) clears `is_initialized`, calls `options.update`, and
sets it back without validating. Reproduction:

```
options={'max_fun_eval': 123}                         ValueError: The option max_fun_eval does not exist.
options_path with 'max_fun_eval'/'completely_made_up' accepted; both stored;
                                                      max_fun_evals still 200
load(new_options={'max_fun_eval': 999})               accepted; max_fun_evals still 200
```

MATLAB's `misc/setupoptions_vbmc.m:9-26` copies every `defopts` field into the
user's struct and never objects to an extra field, so **all three** PyVBMC routes
are at least as strict as MATLAB. This is an internal inconsistency, not a port
discrepancy — worth flagging because the `load` case silently produces a run that
terminates at the old budget.

### C-5 The transformer after `load`

On the committed `pyvbmc/testing/vbmc/test_vbmc_save_static.pkl` (7 recorded
iterations, no warp):

```
vbmc.pt is vp.pt -> False | vbmc.pt is fl.pt -> False | vp.pt is fl.pt -> False
(all three equal by value)
load(iteration=0): vbmc.pt is vp.pt -> False
load(iteration=6): vbmc.pt is vp.pt -> False
```

A fresh round trip in the scratch directory keeps the identity, because a fresh
instance has `iteration = -1` and `iteration_history["vp"] is None`, so `load`'s
restore loop is skipped. The break therefore needs a run with history.

**No numerical consequence in a continued run.** `self.parameter_transformer` is
read nowhere in `optimize()` before `vbmc.py:1385`, which sets it (and the
logger's) from `self.vp.parameter_transformer`. The full list of reads of
`self.parameter_transformer` is `vbmc.py:399, 413, 450, 460, 774, 910-919,
1385-1387, 3496`; all but `:1385-1387` and `:3496` are in `__init__` /
`_init_optim_state`.

**The `__str__` consequence is not confined to `load`.** `self.x0` is written
once, at `:460`, in the *initial* transformed space, and is never re-transformed
by a warp; `__str__` inverts it with the current transformer (`:3496`). Warping
the stored run's state once with `whitening.warp_input`:

```
x0 (stored, initial transformed space): [[ 3.967e-07 -1.071e-07  5.172e-08 -2.086e-07]]
inverse with the initial transformer:   [[ 1.374e-06 -3.710e-07  1.792e-07 -7.227e-07]]
warp action: rotoscale
inverse of the SAME x0 with the warped transformer:
                                        [[ 4.105e-08  1.403e-07  4.687e-08  3.064e-07]]
```

So a warped run prints a wrong starting point with or without a load; what `load`
adds is that for `iteration=` before a warp, `vbmc.parameter_transformer` is the
*later* map while `vp`, `function_logger` and `optim_state` are the earlier one,
so a user converting coordinates with `vbmc.parameter_transformer` gets the wrong
map for the state they asked for.

**Scope note.** A fix touches `load` (`vbmc.py:2915-2935`), and if the `__str__`
line is to be right it also needs `self.x0` re-transformed at each warp or
`__str__` reading through `vp.parameter_transformer`. Nothing pins the invariant
after a load: `test_vbmc_optimize.py:20-25` asserts it only during a live run.

### C-6 Inert options

Independent recomputation (`f6_inert_options.py`, scanning the 78 non-test
modules of `pyvbmc/` for `…options[...]`, `.get(...)`, `.eval(...)` with
whitespace flattened):

- **183** declared options; **25** read through no options mapping; `INERT_OPTIONS`
  holds **22**, and no registered name is read. The three extra are `diagnostics`,
  `entropy_force_switch` and `temperature`.
- The guard test (`test_options.py:183-205`) excludes `testing/` and
  `pyvbmc/vbmc/options.py` and then matches the quoted name anywhere. The 22
  registered names occur only in `options.py`, hence "unread"; the three others
  are matched by `sampler_opts["diagnostics"]`
  (`active_importance_sampling.py:437`), `optim_state["entropy_force_switch"]`
  (`vbmc.py`) and `optim_state["temperature"]` (`whitening.py`). Confirmed.
- **Read through an options mapping but declared in no `.ini`**: `varactivesample`
  (`vbmc.py:1409`, `:1438`), `warp_nonlinear` (`vbmc.py:1243`,
  `whitening.py:123`), `skip_elbo_variance` (`variational_optimization.py:498`,
  guarded by `"skip_elbo_variance" in options`, so never true). The reviewer's
  "plus two in comments" are `elcboweight` (`variational_optimization.py:785-786`)
  and `repeated_acq_discount` (`active_sample.py:670`) — both commented out, so
  the live list is exactly the three named.
- **`optim_state` keys named after an option**: `temperature` read
  (`whitening.py`) and never written; `entropy_force_switch` read (`vbmc.py`) and
  never written; `proposal_fcn`, `lcb_max_vec` and `int_mean_fun` written and
  never read; `diagnostics` and `recompute_lcb_max` neither.

**`temperature` against MATLAB.** `misc/setupvars_vbmc.m:249-255` validates
(`round(T) ~= T || T > 4 || T < 1` → `error('vbmc:PosterioTemperature')`) and
stores `optimState.temperature`, which MATLAB then reads in
`misc/funlogger_vbmc.m:132-134`, `:180-182`, `:223-224`,
`misc/vpoptimize_vbmc.m:19`, `misc/vpoptimizeweights_vbmc.m:10`, `:133` and
`misc/finalboost_vbmc.m:50-51`. PyVBMC copies neither the validation nor the
value. Both reviewers' correction of the sheet's P7 entry ("is read only by
`whitening.py:196`, `:291`") stands: those lines read an `optim_state` key nothing
writes, so `whitening` always takes `T = 1` and the option is read nowhere.

**Error in P1b comparison F10.** Its history line says `defopts.Diagnostics` "is
read nowhere in the MATLAB repository either (the `sampleopts.Diagnostics` hits
are the slice/ensemble samplers' own option)". MATLAB reads it twice, at
`vbmc.m:805` (passed to `savestats`, deciding whether the GP is kept in `stats`)
and `vbmc.m:961` (`if ~options.Diagnostics`, removing `gp` and `gpHypFull` from
the returned `stats`). So `Diagnostics` is a live MATLAB option whose function
PyVBMC covers differently — the history always stores a lean GP, with
`record_full_history_details` as the nearest knob. That makes the Python
`diagnostics` option an unported-feature leftover rather than a name dead on both
sides, which is a slightly different registry entry.

**Scope note.** Registering the three names in `INERT_OPTIONS` also requires
tightening the guard test's matching rule (otherwise the assertion
`unread == set(INERT_OPTIONS)` fails in the other direction), correcting the
sheet's P7 entry and the closing claim of its "Declared options that nothing
reads" entry, and deciding separately about MATLAB's `Temperature` validation and
about the dead `optim_state["temperature"]` reads in `whitening.py`.
`recompute_lcb_max` and `entropy_force_switch` belong to the findings being fixed
elsewhere; only their absence from the registry is at issue here.

### C-7 Truncated descriptions

`configparser.ConfigParser(comment_prefixes="", allow_no_value=True)` keeps `#`
lines as entries, but still splits each line at its first `=` or `:`, and
`_read_config_file` takes the description from the *key* half. Exactly 8 of the
183 options are affected, all in `advanced_vbmc_options.ini`; `basic_…ini` has
none:

| option | stored | in the file |
|---|---|---|
| `uncertainty_handling` | `Explicit noise handling (0` | `Explicit noise handling (0: none; 1: unknown noise level; 2: user-provided noise)` |
| `ns_gp_max_active` | `Cap on the number of GP hyperparameter samples of the GP refits within active sampling (0` | `… (0: MAP fit only; inf: as the main fit)` |
| `stable_gp_samples` | `Number of GP samples when GP is stable (0` | `Number of GP samples when GP is stable (0 = optimize)` |
| `search_optimizer` | `Local optimizer of the acquisition search` | `Local optimizer of the acquisition search: "cmaes", "Nelder-Mead" or "none" (no local search); with one variable a bounded scalar search is used instead of either` |
| `gp_sample_widths` | `Multiplier to widths from previous posterior for GP sampling (Inf` | `… (Inf = do not use previous widths)` |
| `upper_gp_length_factor` | `Upper bound True GP input lengths based True plausible box (0` | `… (0 = ignore)` |
| `temperature` | `Temperature for posterior tempering (allowed values T` | `Temperature for posterior tempering (allowed values T = 1234)` |
| `performance_calibration` | `Fixed numerical chunk profile` | `Fixed numerical chunk profile: "cached" reuses a compatible machine-local calibration, "off" pins historical defaults, or pass a CalibrationProfile` |

**Where the stored text surfaces**: `Options.__str__` (`options.py:330-345`, the
user options, which `VBMC.__str__` prints as "user options") and
`Options.__repr__` (`:352-380`, every option). The published documentation is
unaffected: `docsrc/source/api/options/vbmc_options.rst` `:literal:`-includes both
`.ini` files.

**Two errors in P1b internal F11.** (i) It attributes the `adaptive_k` text to
this defect; `adaptive_k`'s description ("Added variational components for stable
solution (a number, or a function of the current number of components K)") has no
delimiter and is stored in full. Only the `search_optimizer` half of `e6a21a4` is
lost. (ii) The "True" in the stored `upper_gp_length_factor` description is not a
parser artefact: the `.ini` file itself reads "Upper bound True GP input lengths
based True plausible box", and `temperature`'s reads "allowed values T = 1234" —
two independent `.ini` text defects (an "on" → "True" substitution and lost
commas) that a fix to the parser would expose rather than repair.

### C-8 `seed=None` and the global state

```
bit generator: MT19937
position before / after: 624 / 4        key changed: True
the four draws consumed: [2357136044 2546248239 3071714933 3626093760]
seed=42:        global state untouched: True
seed=Generator: global state untouched: True
```

The claims it conflicts with:

- `vbmc.py:152-154` (`seed` parameter): "Every random draw of a run comes from
  ``vbmc.rng``; NumPy's global random state is never written."
- `vbmc.py:1132-1134` (`optimize()` Notes): "Every inference draw of the run comes
  from ``vbmc.rng`` (see the ``seed`` parameter); NumPy's global random state is
  neither read nor written." — true as written, since it scopes to the run.
- `AGENTS.md`: "`seed=None` derives the generator from the global `np.random`
  state … so a run never reads or writes NumPy's global state (since 2026-09-05…)"
  — also scoped to the run.
- Sheet, "Randomness is threaded through `numpy.random.Generator`": "a PyVBMC run
  never reads or writes NumPy's global state … `seed=None` derives the generator
  from the global `np.random` state so that `np.random.seed()` beforehand still
  fixes a run."

So only the `seed` parameter docstring makes the unqualified claim. The behaviour
is a recorded decision (`dev/plans/stage1-rng-generator.md` §3, quoted in the
ledger) and is pinned by `test_vbmc_seed.py::test_seed_none_does_not_reseed_global_state`,
whose docstring says "Construction only consumes the draws that derive the
generator … it does not reseed the global state". The plan's phrase "only read
once" is the inaccuracy that propagated into the docstring: four `randint` draws
advance MT19937's position and rewrite its key.

**Scope note.** A fix is one docstring sentence; the alternative (deriving the
seed without advancing the global state, e.g. from `np.random.get_state()`) would
change every `seed=None` stream and move the golden traces, so it is not a
documentation-only change.

### C-9 The default search acquisition

Forms, with `u = vtot · exp(fbar − z) · p > 0`:

- `acq/acqf_vbmc.m:11` / `acq_fcn.py`: `acq = −u`, range `(−inf, 0]`.
- `acq/acqflog_vbmc.m:18` / `acq_fcn_log.py`: `acq = −log u`.

Both are strictly decreasing in `u`, so both are minimized at the same candidate
in exact arithmetic. The regularization is applied consistently:
`acqwrapper_vbmc.m:40-44` adds `TolVar/vtot − 1 = c > 0` in the log form and
multiplies by `exp(−c)` in the plain form, i.e. both compute the same
`u' = u·exp(−c)`; `abstract_acq_fcn.py:166-175` mirrors this and adds a
zero-variance branch (`acq = +inf` for the log form, `0.0` for the plain form)
that in both cases is the worst representable value of that form and matches what
MATLAB's arithmetic produces. The hard-bound penalty is `+inf` in both forms on
both sides, worse than any valid value of either. `max(acq, −realmax)` is inert in
both.

Measured on all eight oracle fixtures (512 candidates each):

| fixture | AcqFcn argmin | AcqFcnLog argmin | exact zeros in AcqFcn | same order on non-saturated entries |
|---|---|---|---|---|
| cigar_D4_boosted | 172 | 172 | 39 | yes (473) |
| cigar_D4_largeK | 201 | 201 | 39 | yes (473) |
| corr_D5_warped | 20 | 20 | 104 | yes (408) |
| halfnormal_D2_bounded | 393 | 393 | 22 (+54 `inf` in both) | yes (436) |
| normal_D2_K1 | 235 | 235 | 0 | yes (512) |
| **normal_D2_singlesample** | **0** | **335** | **512** | — (no live entries) |
| normal_D2_warmup | 85 | 85 | 0 | yes (512) |
| rosenbrock_D2_noise1_viqr | 120 | 120 | 25 | yes (487) |

So the reviewer's claim is right, and the loss of resolution is concrete: where
`vtot·exp(fbar−z)·p` underflows, the plain form returns exactly `−0.0`, its worst
value, and ties those candidates; the log form spreads the same candidates over,
e.g., `[7.47e2, 5.35e3]` on `corr_D5_warped`. On `normal_D2_singlesample` **every**
candidate underflows, so the plain form's argmin is index 0 — an arbitrary
tie-break — while the log form ranks them. A second, subtler consequence: once the
plain form has underflowed to 0 the variance regularization `acq *= exp(−c)` is a
no-op, while the log form's `+= c` still shifts.

The recorded reason names overflow ("to avoid overflow warnings in `exp()`", PR
#102); the dominant measured effect is underflow. The change is pinned by
`test_acq_fcn_log.py::test_acq_fcn_vs_acq_fcn_log`, added in the same commit,
which asserts `np.allclose(-np.log(-acq), log_acq)` — i.e. the monotone relation
itself — and by `test_options.py:158-159`, which asserts the default is an
`AcqFcnLog`.

**Scope note.** If the PI keeps the difference, the sheet needs an entry; nothing
in the code changes. Note the `search_acq_fcn` default is also what
`Options.update_defaults` overrides to `[AcqFcnVIQR()]` for noisy targets, which is
the orchestrator's P1b comparison F1.

### C-10 The two smaller findings

(a) `misc/setupoptions_vbmc.m:109-124` errors for a non-positive or non-integer
`MaxFunEvals`/`MaxIter` and raises `MaxIter` to `MinIter` with a warning; PyVBMC
accepts `max_iter` ∈ {0, 2.5, −3} and `max_fun_evals` ∈ {0, −1, 7.5} and leaves
`max_iter = 1 < min_iter = 5` as given. MATLAB's checks sit after
`if skipextra_flag; return; end` (`:95`), so the main path runs them; they date
from `a1680e3` (2018-05-04) and were moved behind the flag by `f3f8b80`
(2021-03-13) unchanged. PyVBMC never had them. No recorded rationale, no test.

(b) `_budget_active` from presence:

```
VBMC(f, ...)                         _budget_active=False  effective_max_fun_evals=200
initialization_cost=0                _budget_active=True   effective_max_fun_evals=200
precomputed_evaluations=None         _budget_active=True   effective_max_fun_evals=200
both at their documented defaults    _budget_active=True   effective_max_fun_evals=200
```

`_budget_active` gates six sites: `_validate_initial_fresh_budget`
(`vbmc.py:798`), `_fresh_evaluations_for_batch` (`:823`, which caps each batch),
the three extra `optim_state` keys (`:1002-1005`), `optimize()`'s refusal to start
with no allowance (`:1138`), the `load` resume path (`:1171`, `:2985`), the
termination budget (`:2104`), and the extra clause of "prevent early termination"
(`:2198`). With `initialization_cost=0` the *values* are identical, so the only
live differences are the batch cap in the last iteration (the ordinary path may
overshoot `max_fun_evals` by up to `fun_evals_per_iter - 1`) and the termination
clause. Recorded intent: `dev/plans/pymc-target-adapter.md:389-397` explains the
sentinel as a way to let an explicit `None`/`0` override the **adapter's**
defaults, and says "for ordinary callables the effective defaults remain `None`
and zero"; nothing addresses the budget switch, and no test pins the equivalence
(`test_vbmc_precomputed.py:490-501` only checks the `load` backfill).

### C-11 Minor observations

`__str__` on a fresh instance, with `log_prior=` and `sample_prior=` supplied:

```
log-density = None,            (reads `log_likelihood` with `log_joint` as fallback)
log-prior = None,              (no `self.log_prior` attribute exists)
prior sampler = None,          (no `self.sample_prior` attribute exists)
Gaussian process = None,       (VariationalPosterior has no `gp` attribute)
```

`_init_logger`: with two duplicate `FileHandler`s on the process-wide `"VBMC"`
logger, one stale handler survives the removal loop; with a `StreamHandler`
attached and a log file configured, `_init_logger` raises
`AttributeError: 'StreamHandler' object has no attribute 'baseFilename'`;
`log_file_level = logging.DEBUG + 5` raises
`ValueError: Log file logging level is not a recognizedstring or logging level.`
(note the missing space, a third small defect in the same block). `display`
accepts `off` → WARN, `iter` → INFO, `full` → DEBUG, anything else → INFO.

`Options.pop("max_iter")` on an initialized object returns 200 and removes the
key, while `options["max_iter"] = 5` raises `AttributeError`.

`x0` as a list or a float raises `AttributeError: 'list'/'float' object has no
attribute 'ndim'`. Three starting rows with one `NaN` give `x0` of shape `(1, 2)`
and cache `[[0. 0.]]`, the plausible centre — the same rule as
`misc/setupvars_vbmc.m:7-12`.

`x0 = [-1, 0]` exactly on `plb = [-1, -1]` warns "The starting points X0 are not
inside the provided plausible bounds PLB and PUB. Expanding the plausible
bounds…" and leaves `plb`/`pub` unchanged. MATLAB's `boundscheck_vbmc.m:120-126`
uses the same `<=`/`>=` test and the same no-op repair, so this wording problem is
shared, not a port discrepancy.

`Options.eval` with a user callable whose parameter is `n` rather than `K`:
construction succeeds, `options.eval("ns_ent", {"K": 2})` raises
`TypeError: <lambda>() got an unexpected keyword argument 'K'`.

`optim_state` keys written and never read, recomputed independently: exactly the
fourteen the reviewer lists. `IterationHistory.__setitem__` deep-copies, so
`load`'s truncation loop copies the whole history a second time — confirmed, cost
only.

## Summary of errors found in the reviewers' reports

1. **P1b internal F7** — the measured list is wrong as written: plain Python lists
   `[1]`, `[0,1]`, `[0,2]`, `[0,1,0]`, `[True,False,True]` all mark every variable;
   none raises `IndexError`. Only NumPy arrays of the wrong length raise. (P1b
   comparison F3 states this correctly.)
2. **P1b internal F6** — the `D == 1` `IndexError` also needs a zero-width
   coordinate in `x0`, not merely several starting rows.
3. **P1b internal F11** — `adaptive_k`'s description is not truncated (it contains
   no delimiter); and the "True" in `upper_gp_length_factor`'s stored text, and
   the missing commas in `temperature`'s, are defects of the `.ini` text itself,
   not of the parser.
4. **P1b comparison F10** — MATLAB's `options.Diagnostics` is *not* dead: it is
   read at `vbmc.m:805` and `vbmc.m:961`, where it decides whether the GP stays in
   the returned `stats`. The Python option is therefore a knob of a feature the
   port handles differently (lean GP history plus `record_full_history_details`),
   not a name dead on both sides.
5. Minor: **P1b comparison F7**'s history ("`misc/boundscheck_vbmc.m` last changed
   in `969c4ff`") is about the file; the scalar expansion itself dates from the
   first MATLAB commit `a1680e3` (2018-05-04). No consequence for the finding.

Not errors but worth recording: P1b comparison F5's claim that the two
acquisition forms are order-equivalent is correct including the regularization
and the bound penalty, and the underflow it predicts is measurable on the shipped
oracle fixtures, including one fixture where the plain form has no usable ranking
at all.
