# Wave 2 verification

The findings of wave 2 (slices P1a and P1b; reports under `../reviews/`),
verified on 2026-09-20 on `dev-port-review` against MATLAB VBMC at
`396d649`. Part 1 holds the six findings that fire at the default options
and can change a run; part 2 holds the others. Three sources feed the
ledger: the orchestrator's own checks, with the scripts named in the rows;
a read-only Opus agent that traced three warm-up findings through both
histories (`wave2_warmup_history.md`); and two Opus verifiers, one on the
loop, termination and result findings (`wave2_B_loop.md`, rows B-n) and
one on the setup, options and input findings (`wave2_C_setup.md`, rows
C-n), whose files hold the supporting detail, the dating of each row and
the errors they found in the reviewers' reports. Every verifier searched
the code comments, the commit and pull-request messages, the porting logs,
the documentation, `dev/` and the tests for a recorded reason for each
difference; a row says so where one was found. The MATLAB lines involved
all predate the Python lines, so no row is a faithful port of an older
MATLAB.

## Part 1: findings that fire at the default options and can change a run

| id | reports | statement | verdict (class) | fires at defaults? | how verified | PI disposition |
|---|---|---|---|---|---|---|
| W2-1 | P1a comparison F1 | The "recent improvement" window of `_check_warmup_end_conditions` starts at `iteration - (T + 1)`, with `T = ceil(tol_stable_warmup / fun_evals_per_iter)`, where `private/vbmc_warmup.m:60-65` gives `iteration + 1 - T` in 0-based terms: five iterations at the defaults where MATLAB's window holds three, so the no-recent-improvement test is met less often and warm-up tends to end later | confirmed port discrepancy; never matched MATLAB (one iteration too long as written in `35be58b`, 2021-07-07; two since `9267816`, 2021-08-25, moved the loop to a 0-based counter, corrected the neighbouring floor and left this offset) | yes, from the fifth iteration of every run | both sources read; the window's index sets tabulated for both sides; history traced | fix |
| W2-2 | P1a internal F6, P1a comparison F2, P1b internal F8 | `_recompute_lcb_max` returns an empty array that nothing reads, and `_check_warmup_end_conditions` has no counterpart of MATLAB's preference for the recomputed vector (`private/vbmc_warmup.m:46-50`), while `recompute_lcb_max` defaults to `True` as `RecomputeLCBmax` does: both warm-up criteria rest on the per-iteration maxima, each from that iteration's GP, where MATLAB's rest on a cumulative maximum recomputed with the current GP (`private/recompute_lcbmax.m`) | confirmed port discrepancy; an unfinished placeholder (`pass` in the first commit, an empty array since `ec94eba`, 2021-06-08; the ToDo comment came with a pylint clean-up, `39b29e6`); the porting log and the counterpart map list the function as ported | yes, every run | both sources read; reads of `lcb_max_vec` searched; history traced | fix, the function and the consuming branch |
| W2-3 | P1a comparison F3 | `_setup_vbmc_after_warmup` sets `last_warmup` only, where `private/vbmc_warmup.m:97-102` also sets `LastWarping` and `LastSuccessfulWarping` to the current iteration. Both stay at minus infinity, so the first warp can come as soon as its other conditions hold, where MATLAB waits more than `WarpEveryIters` iterations after warm-up, and the stability termination is not held back for `TolStableIters/3` iterations after warm-up | confirmed port discrepancy; omitted in `35be58b`, eight months before warping existed on the Python side; the two readers came with `4aa4ca9` (2022-03-03) and `0a27ee3` (2023-02-15), neither of which touched the end of warm-up. The third MATLAB line, `LastNonlinearWarping`, is read nowhere in MATLAB | yes | both sources read; readers and writers of the two keys searched; history traced | fix |
| W2-4 | P1b internal F1 | `VBMC.__init__` builds the variational posterior from the untransformed `x0` and transforms `x0` afterwards, so the initial component means, a transformed-space quantity, hold original coordinates. `misc/setupvars_vbmc.m:65` transforms `x0` before `:82` sets `vp.mu`. With bounds 0 and 10, plausible bounds 4 and 6 and `x0 = 5`, the means map back to 9.94 | confirmed port discrepancy; in this order since the first commit of the constructor (`489598a`, 2021-06-03), against a MATLAB that already transformed first. The P1b comparison report lists the initial posterior as equivalent: it compared the tiling of the means, not their coordinates | yes, whenever `x0` and its transform differ; invisible when `x0` is the centre of a plausible box symmetric about it | `scripts/wave2_initial_vp_mu.py` (five cases); both sources read; `git log -S` | fix |
| W2-5 | P1a internal F4, P1a comparison F4, P1b comparison F9 | `_check_termination_conditions` holds termination while `iteration < min_iter`, with the 0-based index, where `private/vbmc_termination.m:98-99` compares the 1-based counter, and where the same function compares `iteration + 1` with `max_iter`: a run on which the minimum binds performs `min_iter + 1` iterations | confirmed port discrepancy; `test_vbmc_check_termination_conditions_prevent_early_termination` asserts the off-by-one | yes, when the minimum binds | `scripts/wave2_min_iter_guard.py` (iterations performed for six pairs of limits against `max(MinIter, MaxIter)`); both sources read | fix |
| W2-6 | P1a internal F8 | `warp_input` inverts the search bounds and the search cache with the transformer of the posterior it is handed, the best recorded one, while both live in the current inference space; a posterior recorded before an earlier warp carries another transformer. Handed such a posterior on a stored four-dimensional state, the second warp returns a search box 8.5 to 12.5 times wider per coordinate than the one the current transformer gives; handed the current posterior it returns that one | confirmed shared defect: `misc/warp_input_vbmc.m:8` and `:133` take the old transform from the posterior handed in, as PyVBMC does | latent: on 21 complete stored runs (42 warps, 13 of them after an earlier kept warp) the selection never returned a posterior from another space, under the rank criterion or under the look-back rule | `scripts/wave2_stale_transformer_mechanism.py`, `scripts/wave2_stale_transformer_frequency.py` | fix |

## Part 2: the other findings

| id | reports | statement | verdict (class) | fires at defaults? | how verified | PI disposition |
|---|---|---|---|---|---|---|
| W2-7 | P1b comparison F1 | `Options.update_defaults` applies the five noisy-target defaults (1.5 times `max_fun_evals` and `tol_stable_count`, the GP and VP updates within active sampling, VIQR as the search acquisition) when `specify_target_noise` is set; `misc/setupoptions_vbmc.m:127-163` applies them whenever `UncertaintyHandling` is on, which covers inferred noise (level 1) as well. A level-1 run keeps the noiseless defaults | confirmed port discrepancy; since `70325a3` (2022-06-02), whose pull request says the defaults update "when `specifytargetnoise=True`, as in MATLAB"; no reason for leaving level 1 out is recorded. No documentation page or example mentions `uncertainty_handling`: the documented noisy route is `specify_target_noise` through the options dict, which is right | no; every run with `uncertainty_handling` set | `scripts/wave2_noisy_defaults.py` (five routes at `D = 3`); both sources read | fix: the noisy defaults apply whenever uncertainty handling is on |
| W2-8 | P1b internal F4 | `VBMC.__init__` calls `update_defaults` before it reads the file of `options_path=`, so `specify_target_noise = True` written there gives a level-2 run with the noiseless defaults | confirmed Python-only defect | no; a noisy run configured through an options file | same script | fix |
| W2-9 | P1a internal F7, P1b internal F3 | `load(new_options={"max_fun_evals": n})` updates the option and leaves `optim_state["max_fun_evals"]` at the saved value unless the budget path is active; the GP training options read that copy as the horizon of the schedule for the number of starting points of the hyperparameter fit, which falls to its floor of 9 past the stale horizon (at 60 evaluations: 9 against 759) | confirmed Python-only defect, a regression: until `6769a9a` (2026-09-16) the schedule read `options["max_fun_evals"]`, as `misc/get_GPTrainOptions.m:98` reads `options.MaxFunEvals`. In no release | no; the documented way to continue a run with a larger budget | `scripts/wave2_resume_budget.py` | fix |
| W2-10 | P1a internal F1, P1a comparison F5, P1b internal F2, P1b comparison F2 | `optimize` reads `optim_state["entropy_force_switch"]`, which nothing writes, where `vbmc.m:524-525` reads the option; the stability branch also lacks the `isfinite(options.EntropyForceSwitch)` test of `private/vbmc_termination.m:80` | confirmed port discrepancy (B-1, B-M12); since the loop skeleton `ec94eba` (2021-06-08) | no; `entropy_switch=True` with `D >= 5` raises `TypeError` at `vbmc.py:1226` before the first iteration is recorded | B-1; `scripts/wave2_two_option_runs.py` inside the loop | fix, both sites |
| W2-11 | P1a internal F2, P1a comparison F7, P1b internal F5 | The guard of the running average of the variational moments reads `len(run_cov == 0)` for `len(run_cov) == 0`, so the average is never taken and `moments_run_weight` has no effect | confirmed port discrepancy (B-2). Nothing reads `run_mean`, `run_cov` or `last_run_avg` for a computation in either toolbox: only the recorded state differs | yes, every iteration; no computed number changes | B-2 | fix |
| W2-12 | P1a internal F3, P1a comparison F8 | `results["problem_type"]` tests the transformed bounds, which are infinite for a bounded variable too, so it always says "unconstrained" | confirmed shared defect (B-3a): `private/vbmc_output.m:5-9` tests `optimState.LB`, the transformed bound | yes; a reported field only | B-3a | fix in PyVBMC (test the original bounds); MATLAB's defect goes on the sheet |
| W2-13 | P1a internal F5, P1a comparison F9 | `results["iterations"]` holds the 0-based index of the last iteration where MATLAB's `output.iterations` is the count | confirmed port discrepancy (B-3b). No documentation defines either result field | yes; a reported field only | B-3b | fix: the field reports the number of iterations and its documentation says so; `best_iter` stays an index |
| W2-14 | P1a internal F10 | `_check_warmup_end_conditions` takes the maximum of an empty slice when `tol_stable_warmup <= fun_evals_per_iter` | confirmed shared defect (B-4): `private/vbmc_warmup.m:39` reaches the same empty window; that MATLAB then raises at `:87` is inferred from its operator rules, not executed | no | B-4 | fix |
| W2-15 | P1a comparison F6 | With `separate_search_gp=True` the constant-mean search GP is trained from, and overwrites, the main `hyp_dict` and `optim_state["sn2_hpd"]`, where `vbmc.m:471`, `:638-648` keep a separate struct and discard the returned state | confirmed port discrepancy (B-5); the run fails in the search GP's own training call, iteration 1 (the orchestrator's note in `wave2_B_loop.md`). A fix carries a second hyperparameter struct through save, load and the recorded state | no; the option is unusable | B-5; `scripts/wave2_two_option_runs.py` | remove the branch (a development option of MATLAB VBMC, not needed in PyVBMC); the option stays declared and is registered as inert; sheet entry |
| W2-16 | P1a comparison F10 | The warp-undo refit sizes its sieve at `vp.K` where `vbmc.m:584` uses `Knew`; they differ only with `variable_means=False` | confirmed port discrepancy (B-6). Issue 98 and `2df0d7e` (2022-09-20) replaced the stale `k_warmup`; the issue names `Knew` as the intended value. The sheet's entry on the sieve's candidate count and `wave1_P6.md` say the warp branch agrees with MATLAB | no | B-6 | fix; correct the sheet entry and `wave1_P6.md` |
| W2-17 | P1b internal F7, P1b comparison F3 | `integer_vars` is read as a length-`D` mask where the option's description and `misc/setupvars_vbmc.m:14-24` use indices; a plain list marks every variable integer, silently; only a length-`D` array works | confirmed port discrepancy (C-1); since `489598a` (2021-06-03). MATLAB's own half-integer bound check never fires, and PyVBMC's does | no; every run that sets the option | C-1 | fix, type-strict: a boolean array of length `D` is a mask; an integer array holds 0-based indices, unique and in range; an integer array of length `D` holding only zeros and ones is rejected as ambiguous, with a message asking for a boolean mask |
| W2-18 | P1b comparison F4 | `uncertainty_handling` is tested by length: `True`, `1`, `0`, `False`, `None` raise, and `[0]`, `'no'`, `'off'` select level 1; MATLAB truth-tests a boolean | confirmed port discrepancy (C-2) | no; every run that sets the option | C-2 | fix, type-strict: a boolean as in MATLAB (`True`/`False`, 1/0, empty for unset); any other value raises with a message naming the accepted forms; the description rewritten; MATLAB's error for the option explicitly off with `specify_target_noise` on |
| W2-19 | P1b internal F6, P1b comparison F7 | Scalar bounds raise for `D > 1`, with an unformatted message, although the class docstring and `misc/boundscheck_vbmc.m:6-10` replicate them | confirmed port discrepancy (C-3, C-M6) | yes, for a caller who writes a scalar bound | C-3 | fix |
| W2-20 | P1b internal F12 | Option names are validated for the `options=` dict and not for an `options_path=` file or for `load(new_options=)`, where a misspelt budget is accepted and ignored | confirmed Python-only inconsistency (C-4); MATLAB validates no option name | no | C-4 | fix: an unknown name raises on both routes; `VBMC` validates against the two shipped option files |
| W2-21 | P1b internal F10 | After `load` of a run with history, `vbmc`, `vp` and `function_logger` hold three transformer objects, equal by value, and `vbmc.parameter_transformer` is not the chosen iteration's | confirmed Python-only defect (C-5). No number changes in a continued run, which restores the identity in its first iteration. Separately, `__str__` inverts the stored `x0` with the current transformer, which is wrong after any warp, with or without a load | yes for a loaded run; the wrong map only after a warp | C-5 | fix, the `x0` line of `__str__` included |
| W2-22 | P1a internal F9, P1b internal F8, P1b comparison F6 and F10 | `temperature`, `diagnostics` and `entropy_force_switch` are read through no options object and are missing from `INERT_OPTIONS`; the guard test matches the quoted name anywhere in the package, so it cannot see them; the sheet's entry on posterior tempering says `temperature` is read by `whitening.py`, which reads an `optim_state` key nothing writes | confirmed (C-6). MATLAB validates `Temperature` and reads it in six files, and reads `Diagnostics` at `vbmc.m:805`, `:961`: both are knobs of unported features, not names dead on both sides. The omitted `vptrain2real` calls are harmless at every reachable setting | no numerical effect | C-6 | register `temperature` and `diagnostics` as inert; the guard test matches reads through an options object; correct the two sheet entries |
| W2-23 | P1b internal F11 | `_read_config_file` keeps only the part of an option description before its first `=` or `:`; 8 of 183 options are affected, among them `search_optimizer`, `uncertainty_handling` and `performance_calibration` | confirmed Python-only defect (C-7); the stored text surfaces in `str` and `repr` of the options, not in the published documentation. Two descriptions are also wrong in the file itself ("True" for "on" in `upper_gp_length_factor`; "T = 1234" in `temperature`) | yes; no numerical effect | C-7 | fix the parser and the two description texts |
| W2-24 | P1b internal F9 | `VBMC(seed=None)` draws four integers from NumPy's global state, against the `seed` docstring's "never written" | documentation defect (C-8); the behavior is a recorded decision (`dev/plans/stage1-rng-generator.md`, section 3) pinned by `test_seed_none_does_not_reseed_global_state`; the plan's phrase "only read once" is what reached the docstring | yes, every unseeded construction | C-8 | docstring only (and the phrase in the generator plan) |
| W2-25 | P1b comparison F5 | The default search acquisition is `AcqFcnLog` where `vbmc.m:213` has `acqf_vbmc` | intentional difference, missing from the sheet (C-9); `e20d081` (2022-09-21), pull request 102: "in order to avoid overflow warnings in `exp()`". The two forms rank candidates identically in exact arithmetic, regularization and bound penalty included; on the eight oracle states the plain form underflows on up to all 512 candidates | yes, every run | C-9 | keep; sheet entry |
| W2-26 | P1b comparison F8 | MATLAB rejects a non-positive or non-integer `MaxFunEvals` or `MaxIter` and raises `MaxIter` to `MinIter` with a warning; PyVBMC has neither check and overrides a low `max_iter` silently at run time | confirmed port discrepancy, validation only (C-10a) | no | C-10a | fix, together with W2-5 |
| W2-27 | P1b internal F13 | Passing `initialization_cost=0` or `precomputed_evaluations=None` explicitly activates the budget path, which caps the last batch at the remaining allowance | confirmed Python-only defect (C-10b); `dev/plans/pymc-target-adapter.md` records the omitted-argument sentinel and equates the values, not the budget switch | no | C-10b | fix: the budget path follows the values of the two arguments, not their presence |

The PI ruled on part 2 on 2026-09-20, with one general rule for the
input-handling rows: making an interface stricter, so that a value that
used to be accepted now fails with a clear message, is acceptable, since
the worst case is a user's script that fails once and is corrected in one
place; what is not acceptable is a change of behavior that happens
silently.

### Minor observations

The two verifier files rule on each minor observation of the four reports.
Those they call defects rather than leftovers or faithful ports, all seven
to be fixed (PI, 2026-09-20):

- `final_boost` leaves `warmup = False` and `entropy_alpha = 0` on the live
  `optim_state`, where MATLAB's function works on a copy (B-M3); harmless
  at the end of `optimize`, wrong for a call of the public method.
- The warp branch binds `vp.mu` to a view of `gp.X` where the main loop
  copies (B-M2); only with `variable_means=False`.
- With `do_final_boost=False`, `vbmc.vp` is the iteration-history entry
  itself (B-M5).
- The final "finalize" line is printed only when the boost changed the
  posterior, where MATLAB also prints it when the returned posterior comes
  from an earlier iteration, and its `sKL` compares with the posterior of
  the start of the last iteration (B-M9); display only.
- `VBMC.__str__` always prints `None` for the GP, the log-prior and the
  prior sampler (C-M1).
- `_init_logger` removes handlers while iterating over them and raises
  `AttributeError` when the `"VBMC"` logger carries a non-file handler
  (C-M2).
- `Options.pop` succeeds on a frozen options object (C-M5).

Left to slice P8 (PI, 2026-09-20): the warp re-transforms only the active rows of the function
logger where `misc/warp_input_vbmc.m:112-119` re-transforms every row
(B-M11). Both P8 reviewers found it without being told, and it is row W3-19
of `wave3.md` (P8-1 of `wave3_P8.md`), fixed in `9ebaa48`. Faithful ports, not defects: the recorded posteriors lacking the
stability flag (B-M1) and a non-finite entry of `x0` replacing the whole
starting set (C-M9).

## Notes on part 1

- W2-1 to W2-5 move default trajectories when fixed: the end of warm-up
  (W2-1, W2-2), the first warp and the earliest stable termination
  (W2-3), the first variational optimization of a run whose start point
  changes under the transform (W2-4), and the length of a run on which
  `min_iter` binds (W2-5).
- W2-4 costs the first iteration one of its two slow optimizations: with
  two of them, `optimize_vp` starts the first from the candidate built on
  the incoming means, and the second from the best training points.
- The frequency script of W2-6 reads complete VBMC saves kept under the
  gitignored `dev/scripts/runs/` (the box-sampler runs and the S-VBMC
  pilot pool; `dev/scripts/runs/LOCAL.md` lists them). Of the 42 warps in
  those runs, 25 were undone by the undo check.

## Fix commits

The fixes of 2026-09-20 on `dev-port-review`, one commit per finding, each
with a test written against the contract (the MATLAB lines, the docstring or
the ruling); `d5b2139`, which changes two types in a docstring, has none. Three Opus agents made them in worktrees; their reports are
`../fixes/wave2_agent_A.md`, `_B.md` and `_C.md`, whose hashes are those of
the worktree branches. The hashes below are the cherry-picked commits.

| finding | commit | note |
|---|---|---|
| W2-1 | `c918d12` | moves default trajectories |
| W2-2 | `8e591ff` | moves default trajectories. The orchestrator added the handling of `NaN` entries to the agent's commit: a warm-up trim can drop every point an early iteration logged, `movmax` then leaves that iteration's entry `NaN`, and the maxima of the warm-up check pass over it as MATLAB's `max` does; with `np.amax` the check raised |
| W2-3 | `fb8a12e` | moves default trajectories |
| W2-4 | `8cd4bbc` | moves default trajectories |
| W2-5 | `567444f` | moves default trajectories |
| W2-6 | `7d93f60` | |
| W2-7 | `0ba7680` | |
| W2-8 | `640ac92` | an option set in the file of `options_path=` counts as set by the user |
| W2-9 | `6fa6073` | |
| W2-10, B-M12 | `e71f97b` | |
| W2-11 | `ab5c603` | changes the recorded `run_mean` and `run_cov`, which nothing reads |
| W2-12 | `41ea8b1` | the string stays `"bounded"` |
| W2-13 | `4822ae1` | `dev/scripts/benchmark_targets.py` drops its `+ 1` |
| W2-14 | `54f0f19` | |
| W2-15 | `c7a01f3` | |
| W2-16 | `d7c7887` | |
| W2-17 | `5b3b093` | |
| W2-18 | `1a9f37a` | |
| W2-19, C-M6 | `477226b` | |
| W2-20 | `972bbf4` | |
| W2-21 | `6bb129c` | adds `VBMC.x0_orig`, the starting points in the caller's coordinates; `load` fills it in for instances saved without it |
| W2-22 | `82c624c` | |
| W2-23 | `3722210` | also corrects the seventeen other descriptions in which "on" had become "True" |
| W2-24 | `77575b1` | |
| W2-26 | `17badf9` | |
| W2-27 | `f9689dc` | |
| B-M2 | `fe5f9dc` | |
| B-M3 | `261b9ae` | |
| B-M5 | `b3ad32b` | |
| C-M1 | `9a068c4` | and `1c1f2e4` for the `log-density` line of the same summary, which always printed `None` without a separate prior (found by the fix agent; PI: fix) |
| C-M2 | `658ecb8` | |
| C-M5 | `5d2e543` | made by the orchestrator: four tests outside the agent's files delete an option from an initialized object to build the state of an older save, and use the `force` override |
| B-M9 | `e9e7803`, `9ff44c5` | the agent's fix with one change (PI, 2026-09-20): the divergence of the closing line is estimated on a copy of the run's generator, so that the display moves no random stream and a run stopped without a boost still continues as an uninterrupted one would. The first commit kept the run's generator for the case in which the boost changed the posterior, to leave existing streams alone; the corrected reference of the divergence moved them all the same, because the balanced sampler draws a number of values that depends on the mixture weights, and the second commit takes the copy in every case |
| B-M4 | `d5b2139` | not among the observations ruled on at first; PI, 2026-09-20: fix |

## Found during the fix pass

| id | found by | statement | verdict (class) | fires at defaults? | how verified | PI disposition |
|---|---|---|---|---|---|---|
| W2-28 | fix agent B | `Options.update_defaults` computes all five noisy-target defaults before it checks which of them the user set, so a noisy run with `max_fun_evals = np.inf` raises `OverflowError` from `ceil(inf * 1.5)` although the user's budget would have been kept | confirmed Python-only defect; older than the pass | no | reproduced by the test of the fix, which fails without it | fix (`1421759`) |
| W2-29 | fix agent A | With `variable_means=False`, `final_boost` asks for `max(vp.K, min_final_components)` components while the candidates of the sieve keep the posterior's `vp.K` fixed means, and `_gp_log_joint` raises a broadcast error. A fixed-means run past warm-up holds one component per training input, so it is affected only while it has fewer than `min_final_components` (50) of them; a run that ends during warm-up always is, with its two components | confirmed shared defect, by a reading of MATLAB: `misc/finalboost_vbmc.m:6` takes `Knew = max(MinFinalComponents, vp.K)` and `misc/vbinit_vbmc.m:132-136` keeps `mu0` beside `Knew` weights and scales; older than the pass | no; only with `variable_means=False` | `scripts/wave2_variable_means_boost.py` (the boost raises, the same run without a boost completes); both sources read | fix (`73d2a81`): the boost of a posterior with fixed means places its components at the training inputs of the GP it is handed, one each, as the main loop does after warm-up |
| W2-30 | the CI matrix, on a test of the fix pass | `ParameterTransformer` keeps its bounded transforms as functions defined inside `_set_bounded_transforms`, in an instance attribute, so dill pickles them by value, as bytecode of the Python version that writes the file. Every saved variational posterior, every saved `VBMC` instance and every pickled `SVBMC` object carries them. Under another minor version of Python the file loads, and for a bounded problem the first call of the transformer (`vp.sample`, `vp.pdf`, the construction of an `SVBMC`) ends the interpreter, with a segmentation fault or "Illegal instruction"; an unbounded problem never calls them, which is why the two static fixtures never showed it. Pickling a loaded function again makes dill disassemble it, which corrupted memory on the Python 3.11 cells of the CI (the symptom that led here) | confirmed Python-only defect; older than the pass | yes, for a posterior of a bounded problem used under another Python version than the one that saved it | `scripts/wave2_xver_*.py` and `scripts/wave2_pickled_functions*.py`, with two interpreters on the orchestrator's machine, 3.11.9 and 3.12.6, one process per step: a short bounded run saved under each and sampled, evaluated and saved again under the other, in both directions (every use crashed; with the fix every step works, on the files written before it); the whole S-VBMC path (three converged bounded runs saved under 3.12 by the code before the fix, stacked, optimized, sampled and saved under 3.11, loaded back under 3.12 with the same ELBO); dill's trace, which shows these three functions as the only ones a posterior file or an `SVBMC` object stores by value | fix (PI, 2026-09-20): the transformer pickles and copies without the functions and rebuilds them from `bounded_types` when restored, which also rescues files written before. A saved `VBMC` instance still holds the target and the log-joint wrapper by value: under another Python version it can be inspected, not continued or saved again, and the `save` and `load` docstrings say so |

## The independent check of the pass

Six fresh Opus reviewers, read-only, checked the pass on 2026-09-21 without
the context of the session that made it, on the head of the branch 144
commits after the pass (`a093f2e`, and `0bf7963` once the records of wave 5
were committed; `pyvbmc/` is the same in both): the fixes of the warm-up, the
termination and the initial posterior; the rest of the loop, the warp branch,
the results and the final boost; the options and the inputs; `load`, the
state and the pickling; this ledger and the developer records; and the
user-facing records with the consequences of the pass outside the lines it
changed. Each judged whether a fix was right when made and whether it still
holds. All 47 commits hold, and nothing in the passes of waves 3 to 5 undid
one. `_check_warmup_end_conditions` agrees with a 1-based transcription of
`private/vbmc_warmup.m` on 300 random histories (48 of them with the empty
window, none a mismatch), and `_recompute_lcb_max` with
`private/recompute_lcbmax.m` line by line. The transform that `warp_input`
inverts the search state with (W2-6) survives the rewrite of `whitening.py`
by the wave-3 pass. A transformer restored by `dill`, by `pickle` or by a
copy gives `__call__`, `inverse` and `log_abs_det_jacobian` bit for bit, for
the four bounded types with and without a rotation and a scale, and no
attribute has joined the class since W2-30. No reader of
`results["iterations"]` still adds or subtracts one, and no notebook,
generated script, script of `dev/scripts/` or test helper gives an option in
a form the pass refuses, by a scan of every `VBMC(` call against the declared
names. The ledger accounts for all 43 findings of the four reports; the 21
MATLAB line ranges opened, the datings and the 35 pairs of the table of fix
commits are right, and every disposition traced was carried out with its
secondary clauses.

In the code they found what these commits follow up, each fix with a test
against the contract:

| commit | |
|---|---|
| `b7a2cd0` | `load` checks the limits on iterations and evaluations as construction does. `validate_run_limits` (W2-26) had its one call in the constructor, and the check of the wave-3 pass that gave `load` the value checks (`9b5213d`) left it out, so `load(new_options=)` took `max_iter = 0`, a fractional or negative `max_fun_evals`, and a `max_iter` below `min_iter`, against the docstring of `load` and the changelog. Found by three reviewers independently |
| `ce59fbe` | the `x0_orig` that `load` fills in for a file saved without it (W2-21) comes back through the map of the first recorded iteration, the map of construction, since `x0` is kept in that space. It was inverted with the map of the restored iteration, which gave another point for a run that had warped (`[6.78, 0.66]` for `[4.0, 0.3]` on the reviewer's doctored run), and the fix agent's report says that nothing better could be recovered. The test of the pass asserted the expression `load` used |
| `6aab88a` | `final_boost` with `variable_means` off refuses a GP with fewer training inputs than the posterior has components. Since W2-29 the number of components comes from the GP handed in, and a smaller one ended in the broadcast error of the sieve; `optimize` cannot reach the case |
| `3f85501` | an option the user set keeps its description. `load_options_file` stored a description only with a value it loaded, so `print(options)`, which lists the user's options, showed `None` for every option of the advanced file, the surface that W2-23 had repaired for the cut-off texts. Older than the pass |
| `37c32ec` | `specify_target_noise` is read as a boolean, as its partner is since W2-18: `"no"`, `"off"` and `[0]` turned the noise handling on without a word, and an empty array raised NumPy's own error. MATLAB reads the option through `evalbool`. Made under the PI's rule for the input rows of part 2 and not ruled on by itself; two tests gave the option the value 2 |
| `98a8a91` | the final plot of `optimize` and the last frame of `create_vbmc_animation` count the iterations in their title; both printed the index of the last one, which W2-13 left one below `results["iterations"]` |
| `57c5631` | without `x0`, two scalar plausible bounds raise an error that names the problem, where the shape lookup raised `AttributeError` or `IndexError` under a docstring that promises the replication of W2-19; one plausible bound with an entry per variable gives the number of variables, and a list is accepted |
| `2d3f876` | the first case of `test_vbmc_check_termination_conditions_prevent_early_termination` set `max_iter` below `min_iter`, which W2-26 raises, so nothing asked the run to stop and its assertion held whatever the guard of W2-5 did; the budget is now spent |
| `8afeebf` | `load` states `tol_elcbo_boost = None` for a file saved before the option existed; the unguarded boost of such a file had rested on the default of a lookup |
| `c1e1634` | `build_options` of the oracle harness, which says it mirrors the constructor, runs the check of the run limits; no fixture changes |

In the records they found, and this check corrected:

- The note of `VBMC.save` and `VBMC.load` (`1c0d0d3`) tied the bytecode in a
  saved instance to a target that cannot be pickled by name. The nine options
  whose value is a function (`ns_ent`, `k_fun_max` and the like) are
  evaluated from the ini files, so dill stores them by value in every saved
  instance, the stored functions of `test_vbmc_save_static.pkl` hold bytecode
  of another Python version, and a continued run calls them in every
  iteration. The note, the trap in `AGENTS.md`, the fixtures sheet and the
  FAQ on continuing a run say so (`1f9d872`, `82dbfa5`). Whether `load`
  should rebuild them from the ini files is a question for the PI, below.
- `known_differences.md`: three citations of MATLAB and ini files that
  `refresh_citations.py` had carried through Python files on the day of the
  check (`9f6f0c7`: `private/vbmc_warmup.m:97-102` written as `:140-145`, in
  a file of 134 lines, and two lines of `advanced_vbmc_options.ini`). The
  script took a bare `:N` for a continuation of the last Python path unless
  a file of another kind had been cited with a line number of its own; a
  named file now ends the run either way, a bare citation beyond the end of
  its file is reported, and so is a Python file cited without its path
  (`895ced8`). An audit of the 112 citations that commit rewrote found those
  three, one range that overran its block and, passed over by the script,
  three stale Python citations. Also corrected: the count of the inert
  options (twenty-seven, with `cov_sample_thresh` named), the range of the
  `display` switch, what the entry on `results["rng_state"]` cites, and the
  NaN condition of the recomputed maxima, which is a condition on the points
  logged up to an iteration, in the sheet, the docstring and the comment
  (`224c7a1`). The sheet gains the line on MATLAB's total run time that row
  B-M13 of `wave2_B_loop.md` had asked for.
- `counterpart_map.md`: the 17 Python citations of the eleven rows the pass
  changed pointed up to 120 lines away, and the header dated them to the
  wave-1 refresh. The other rows keep the lines of that refresh.
- `matlab_side_defects.md`: the header listed the entries that mark an
  inferred step and left out entries 4 and 14 of this wave; it no longer
  lists them.
- This ledger: where observation B-M11 was settled, and the commit without a
  test. `wave2_B_loop.md`: a flag under its header on the agent's inference
  that the corrected running average (W2-11) would change nothing.
- `CHANGELOG.md`: its "Upgrading" list had no line for
  `results["problem_type"]`, for the run limits or for what a list of
  `integer_vars` indices means; `vbmc.x0_orig` was in no record a user reads
  and is now an attribute of the class docstring (`4f9e3d6`).
- The stored `print(vbmc)` of example 2 showed the starting point in
  transformed coordinates and the initial means in original ones, the two
  defects of W2-21 and W2-4 (`f40da0a`; three lines, not run again).
- The porting log named `_is_finished()` for
  `_check_termination_conditions()`; the docstring of
  `scripts/wave2_initial_vp_mu.py` cited `:66`, `:84` for
  `misc/setupvars_vbmc.m:65`, `:82`; the `n_iterations` of the eight oracle
  fixtures that record it holds the index that `results["iterations"]` was
  before W2-13, which a comment at the generator's line now says.

Left for the PI:

- The option functions of a saved run. Rebuilding them from the ini files in
  `load`, for the options the user did not set, would let a run with an
  importable target continue under another minor version of Python. It needs
  the two-interpreter experiment of W2-30, and a rule for a default that
  changes between releases.
- `vbmc.x0` stays in the inference space of construction when the run warps
  that space; `AGENTS.md` and the class docstring now say so. Nothing reads
  it after construction but the back-fill of `load`.
- `optim_state["last_warmup"] = 0` for a run without warm-up is MATLAB's
  1-based value in a 0-based field, masked at every default
  (`active_sample_full_update_threshold`); the fallback of
  `.get("last_successful_warping", 0)` is an iteration index where the key
  starts at minus infinity.
- The main loop sizes its sieve at `self.vp.K` where the warp branch and the
  boost use the count they optimize; with `variable_means` off that is the
  count of the iteration before. The sheet's entry covers free means alone.
- `validate_run_limits` checks neither the sign nor the integrality of
  `min_iter`, as MATLAB does not. The guard of the inert options runs one
  way: `warp_nonlinear`, `varactivesample` and `skip_elbo_variance` are read
  through the options and declared nowhere. The warnings of `Options` go to
  the root logger and miss `log_file_name`.
- `_normalized_hard_bound` of the `PyMCTarget` route does not replicate a
  scalar bound. The tests of `_recompute_lcb_max` reach neither its wiring
  into the loop, nor the noisy branch, nor `add_noise=False`.
- The changelog has no entry for W2-16 and W2-11, no "Upgrading" line for
  the copies that `determine_best_vp` and `final_boost` now work on or for
  the frozen `pop` and `del`, and says of the budget keys of the results
  that each is present "only when its argument was used", where W2-27 made
  it the values.

Gates, on the branch of the check (`dev-port-review-w2check`, cut at
`0bf7963`), with the package of that checkout: the tests of every module
touched, 413 passed; the oracle tests, 143 passed; and
`make_oracle_fixtures.py --check --exact`, 11 of 11 with nothing
re-baselined. Not run: the whole suite, the four seeded runs, the Torch and
PyMC environments and the CI matrix.
