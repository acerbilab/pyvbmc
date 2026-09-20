# Wave 3 fix pass, agent B: the GP training policy and its option values

Raw report of a fix agent, wave 3 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, in
its own git worktree cut from `dev-port-review` at `fc8e561`, fixed the
wave-3 findings on the policy of the GP hyperparameter fit, the order of
the high-posterior-density subset and the values that `gp_mean_fun` and
`gp_hyp_sampler` accept (W3-15, W3-14, W3-16, W3-12, W3-13), one finding
per commit with a test written against the contract, running only the
focused test files it touched. The commit hashes in the report are those
of the agent's worktree branch; the orchestrator reviewed each diff and
cherry-picked the commits onto `dev-port-review`, where they carry other
hashes (`../verification/wave3.md` lists them). The text below the rule is
the agent's final message, unedited.

---

# Wave-3 fix pass, agent B

Worktree: `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-ab2d8b038d815264f`
Branch: `worktree-agent-ab2d8b038d815264f` (cut from `fc8e561`; nothing pushed)

## Setup outputs

- `git merge --ff-only dev-port-review` → fast-forward `b2c16ebf..fc8e5617`; `git log -1 --format=%h` printed `fc8e5617`.
- `python -c "import pyvbmc; print(pyvbmc.__file__)"` from the worktree root printed
  `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-ab2d8b038d815264f\pyvbmc\__init__.py`.
- Every Python command ran with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. No standalone script was needed (one short `python -c` probe of NumPy's tie order, run from the worktree root).
- `git status --porcelain --ignored` (ignoring `!!`) is empty at the end.

## `git log --oneline fc8e561..HEAD`

```
bc4bcb6d fix(vbmc): accept slice sampling alone for the GP hyperparameters
0c05c59b fix(vbmc): accept only the GP mean functions the package can build
f0ccb4ed fix(stats): order the high-posterior-density subset with a stable sort
d135705c fix(vbmc): floor the GP design schedule at no points
599522a4 fix(vbmc): start the GP retrain branch at the first recorded reliability index
```

Diffstat over the range: 8 files, +361 / −153.

## Commits in order

| # | hash | finding | what it does | files | tests |
|---|---|---|---|---|---|
| 1 | `599522a4` | W3-15 | the retrain-threshold branch of `_get_gp_training_options` tests `iteration > 0`, the same translation of `iter > 1` that the function already uses when it picks the reliability index | `gaussian_process_train.py`, new `test_gp_training_policy.py` | +`test_retrain_branch_starts_at_the_first_recorded_reliability_index`, +`test_retrain_branch_is_not_taken_above_the_threshold` |
| 2 | `d135705c` | W3-14 | the design-size schedule is floored at 0, as `misc/get_GPTrainOptions.m:100` floors it, instead of 9 | `gaussian_process_train.py`, `test_gp_training_policy.py` | +`test_design_size_follows_the_schedule_past_its_horizon`, +`test_budget_path_clips_the_schedule_at_its_final_size` |
| 3 | `f0ccb4ed` | W3-16 | `get_hpd` and `_estimate_noise` order with `np.argsort(-y, axis=None, kind="stable")`, MATLAB's stable descending order | `stats/get_hpd.py`, `gaussian_process_train.py`, `testing/stats/test_get_hpd.py`, `test_gp_training_policy.py` | +`test_get_hpd_orders_distinct_values_by_value_alone`, +`test_get_hpd_breaks_a_tie_at_the_cut_by_index`, +`test_estimate_noise_breaks_a_tie_at_the_cut_by_index`, +`test_estimate_noise_takes_the_largest_of_distinct_values` |
| 4 | `0c05c59b` | W3-12 | `_init_optim_state` validates `gp_mean_fun` against the three names the package builds; the message quotes the value and names `'zero'`, `'const'`, `'negquad'` (the `egquad` misspelling and the other nine names are gone) | `vbmc.py`, `test_gp_training_policy.py` | +`test_construction_accepts_a_mean_function_the_package_builds` (3 cases), +`test_construction_refuses_a_mean_function_the_package_cannot_build` (10 cases) |
| 5 | `bc4bcb6d` | W3-13 | new `_validate_gp_hyp_sampler_option` raises `NotImplementedError` at construction for any `gp_hyp_sampler` but `"slicesample"`; the six other sampler branches and the `slicelite` burn-in are removed from `_get_gp_training_options`; `cov_sample_thresh` registered in `INERT_OPTIONS`; both option descriptions updated | `vbmc.py`, `gaussian_process_train.py`, `options.py`, `advanced_vbmc_options.ini`, `test_gaussian_process_train.py`, `test_gp_training_policy.py` | +`test_construction_accepts_slice_sampling`, +`test_construction_refuses_an_unported_hyperparameter_sampler` (7 cases); **rewrote** `test_get_gp_training_options_samplers`; **changed** `test_get_gp_training_options_opts_N` |

## Contract and test, per commit

1. **W3-15.** `misc/get_GPTrainOptions.m:109` is `iter > 1 && stats.rindex(iter-1) < options.GPRetrainThreshold`; `:5` is the same `iter > 1`, which `_get_gp_training_options` already translates as `iteration > 0` at its top. Test: `optim_state["iter"] = 1`, `recompute_var_post = False`, `gp_retrain_threshold = 10`, `r_index[0] = 5.0` must give `init_N == 0` and `opts_N == 0`; a companion at the same iteration with `r_index[0] = 50.0` must give `init_N > 0` and `opts_N == 1`. Before the fix the first asserted `1024 == 0`.
2. **W3-14.** `misc/get_GPTrainOptions.m:100` is `Ninit = max(round(f(x)),0)`. What `init_N = 0` means is in the commit body: gpyreg's `GP.fit` skips `f_min_fill` entirely, evaluates the objective at the starting vectors it was handed, orders them by value and takes `widths_default = PUB - PLB` (`gaussian_process.py:1229`, `:1264-1270`), which is `gplite_train.m:202`'s behavior; and `train_gp:136` stops collecting hyperparameter vectors from earlier iterations, the guard being `gp_train["init_N"] > 0`, which is `misc/gptrain_vbmc.m:38`. Test: `max_fun_evals = 5000`, `D = 2`, and the schedule evaluated at `n_eff ∈ {10, 500, 1400, 1500}` against the transcribed MATLAB line — `1024`, `> 9`, `1` and `0`. A second test sets `budget_active` and `n_eff = 1500` and asserts `init_N == gp_train_n_init_final == 64` (the clipped argument), unchanged by the fix. Before the fix the first asserted `9 == 1`.
3. **W3-16.** `misc/gethpd_vbmc.m:9` and `misc/gptrain_vbmc.m:355` are `sort(y,'descend')`, which is stable. Reference order in the tests is `sorted(range(N), key=..., reverse=True)` — Python's `sorted` is stable and `reverse=True` leaves equal elements in place, so it is MATLAB's order. Tests: 50 seeded random tie-free draws must give that order (this is the permutation the code already produced, so it pins that nothing moves for continuous targets); `y = [2,2,2,1,0]` with `hpd_frac = 0.4` must give indices `[0, 1]` (NumPy's quicksort gave `[2, 0]`). For `_estimate_noise` the same vector with distinct `s2` makes `hpd_N = ceil(0.2·5) = 1`, so the estimate is the noise at one point; it must be row 0's, and not row 1's or 2's; a seeded tie-free companion asserts the maximum. Both tie tests failed before the fix.
   The other `get_hpd` callers were read: `_gp_hyp:325` hands the subset to gpyreg's `get_bounds_info`, which uses only `max`/`min`/`std`/`median`/`quantile`; `active_sample.py:536` takes a covariance, `:1004` a mean and a covariance; `variational_optimization.py:798` passes it to `_vb_init`, which re-sorts with `np.argsort(-y_star, kind="stable")` and so inherits this order — exactly as `misc/vbinit_vbmc.m:26` inherits `gethpd_vbmc`'s. None of them reads the order among equal values in a way the fix breaks.
4. **W3-12.** `misc/setupvars_vbmc.m:287` lists twelve names because `gplite_meanfun.m` implements twelve; `_meanfun_name_to_mean_function` implements three. Test: each of `zero`, `const`, `negquad` constructs and reaches `optim_state`; each of the nine MATLAB-only names plus `notvalid` raises `ValueError` at construction, with the offending value and the three supported names in the message, and the helper still raises for the same name when called directly. The ten rejection cases failed before the fix. `test_vbmc_init.py::test_vbmc_optimstate_gp_functions` (not in my area) still passes: the `vbmc:UnknownGPmean:Unknown/unsupported GP mean` prefix it matches is kept.
5. **W3-13, the PI's ruling.** `_validate_gp_hyp_sampler_option` follows `_validate_noise_shaping_option`: a `NotImplementedError` naming the option, the value and the fact that the other samplers of MATLAB VBMC are not ported, called from `__init__` directly after it. What remains of the sampler block in `_get_gp_training_options` is the `'slicesample'` case of `misc/get_GPTrainOptions.m:19-27` plus the `otherwise` error, compared line by line; the rest of the function (the cubic, the `RecomputeVarPost` branch, the retrain branch) is unchanged apart from the `slicelite` burn-in, which only that sampler reached. `cov_sample_thresh` was the one option the removed branches alone read (checked by grep across the package), and `test_inert_options_are_the_declared_options_nothing_reads`, which recomputes the registry from the package, passes with it added. Tests: seven rejected values at construction (each message must carry the option name, the value and `slicesample`), and `"slicesample"` accepted. `test_get_gp_training_options_samplers` asserted `res8["sampler"] == "covsample"` where MATLAB answers `'slicesample'` — that whole test is replaced by one that states the supported sampler's policy: the name, the widths as `max(sqrt(diag(hyp_cov)), 1e-3) · max(gp_sample_widths, r_index)`, `widths is None` with no covariance and with `gp_sample_widths = 0`, and the `ValueError` for an unknown name. `test_get_gp_training_options_opts_N` no longer sets `gp_hyp_sampler = "slicelite"` (its assertions are unchanged and still hold). Both changes are in the same commit and named in its body.

## Test commands run, and results

All from the worktree root with the three thread variables set, `-q -p no:cacheprovider`:

- `pyvbmc/testing/vbmc/test_gp_training_policy.py` — **27 passed**
- `pyvbmc/testing/vbmc/test_gaussian_process_train.py` — **12 passed**
- `pyvbmc/testing/stats/test_get_hpd.py` — **4 passed**
- `pyvbmc/testing/vbmc/test_options.py` — **68 passed**
- `pyvbmc/testing/vbmc/test_vbmc_init.py::test_vbmc_optimstate_gp_functions` — **1 passed** (run because commit 4 changes the code it covers; the rest of that file was not run)
- Final combined run of all of the above at HEAD: **112 passed in 2.9 s**

Every fix was seen to fail first, by stashing the source files (never a bare `git stash`; a tagged `push`, `apply <sha>`, `drop`) or by reverting the hunk and restoring it. No whole test directory, no full suite, no `optimize()` run, no oracle or golden command, no install.

## Changelog sentences (for a user of release 1.0.4)

- **W3-15** — internal only, nothing a user can notice at any option setting; no changelog line needed. (The branch fires at the same iteration in a real run either way, because the reliability index is infinite for the first two iterations.)
- **W3-14** — "The schedule that shrinks the initial design of the GP hyperparameter fit now runs out as MATLAB VBMC's does, instead of stopping at nine points: a run with `max_fun_evals` above 1000 spends less time on that design past about 1400 evaluations." Cannot stop a script; it changes nothing a script reads, and nothing at the default budget.
- **W3-16** — "The high-posterior-density subset (`pyvbmc.stats.get_hpd` and the GP noise estimate) now breaks ties between equal log-density values the way MATLAB VBMC does, taking the earlier point first. Targets that can return exactly equal values, such as a quantized log likelihood, may give slightly different results." Can change what a script returns for such a target, so it belongs in the "Upgrading from 1.0.4" list; for a target with distinct values nothing moves.
- **W3-12** — "`options['gp_mean_fun']` now accepts only `'zero'`, `'const'` and `'negquad'`, the mean functions PyVBMC builds, and refuses the others when the `VBMC` object is created rather than part-way through the first iteration." Can stop a script that passed one of the nine MATLAB-only names — such a script used to fail anyway, but later; it belongs in the "Upgrading from 1.0.4" list.
- **W3-13** — "`options['gp_hyp_sampler']` now accepts only `'slicesample'`. The other samplers of MATLAB VBMC are not ported and used to fail somewhere inside a run; `options['cov_sample_thresh']`, which only covariance sampling read, has no effect and is reported as having none." Can stop a script that set either option, so it belongs in the "Upgrading from 1.0.4" list.

None of the five commits changes a number of a run at the default options, so I expect no oracle and no golden trace to move. The one fixture-adjacent touch is `pyvbmc/testing/oracles/_gp_fit_history.py`'s `OPTION_KEYS`, which snapshots `cov_sample_thresh` at its default; I left it alone and the option is still declared, so the capture and the rebuild are unaffected (a repeated default of an inert option is silent by design).

## What I stopped on

Nothing. All five findings are fixed as ruled.

## For the orchestrator

- **`train_gp`'s `npv` block is now unreachable.** `gaussian_process_train.py:166-171` (`if "hyp_vp" in hyp_dict and hyp_dict["hyp_vp"] is not None and gp_train["sampler"] == "npv"`) can no longer fire, since `gp_train["sampler"]` is `"slicesample"` or the call raised. It is in `train_gp`, outside my area, so I left it; nothing writes `hyp_dict["hyp_vp"]` anywhere in the package either (the MATLAB line that would is the commented-out block at `:185-187`).
- The `Raises` section of `_get_gp_training_options`' docstring still says `ValueError` for an unknown sampler, which remains true (the `otherwise` branch is kept, and the oracles' rebuilt states reach the function without going through `VBMC.__init__`).
- `_validate_gp_hyp_sampler_option` runs in `__init__` only, like `_validate_noise_shaping_option`. A run restored with `load(new_options={"gp_hyp_sampler": ...})` is not re-validated, and would reach the `ValueError` inside `_get_gp_training_options` instead. I left that as it is, matching the existing pattern; say the word if the check should also run on `load`.
- Two small things noticed while reading, both outside my area and neither touched: `gaussian_process_train.py:184` carries the comment "currently not used since we do not support samplers other than slice sampling", which the new construction-time check now makes literally true; and the `t_train` returned by `_get_training_data` is still dropped by `train_gp` (row W3-18 / P5-14, already on the sheet list).
- No `MANIFEST.in` change is needed for the new test file: `pyvbmc/testing/vbmc/` has an `__init__.py`, so its `.py` modules ship as package sources.
