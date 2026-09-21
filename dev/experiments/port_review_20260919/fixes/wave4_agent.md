# Wave 4 fix pass: the fix agent

Raw report of the fix agent, wave 4 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent on its
own git worktree, cut at `f873556`, made the fixes of the rows of
`../verification/wave4.md` that the ledger marks "fix" or "stricter
interface", one commit per row; the orchestrator reviewed the diffs and
cherry-picked the commits onto `dev-port-review`, where they have other
hashes (`../verification/wave4.md`, "Fix commits"). The text below the rule
is the agent's final message, unedited.

---

# Wave 4 fix pass: report of the fix agent

Worktree: `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-a74236c55279f1b25`
Branch: `worktree-agent-a74236c55279f1b25` (cut from `b2c16ebf`, fast-forwarded to `f8735567`; nothing pushed)

## Setup outputs

- `git merge --ff-only dev-port-review` → fast-forward `b2c16ebf..f8735567`; `git log -1 --format=%h` printed `f8735567`.
- From the worktree root, `python -c "import os, sys; sys.path.insert(0, os.getcwd()); import pyvbmc; print(pyvbmc.__file__)"` printed
  `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-a74236c55279f1b25\pyvbmc\__init__.py`.
- Every Python command ran with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. One standalone probe script (in the scratchpad, `PYTHONPATH=<worktree>`) checked the identity transformer, `np.around(0.5)` and the probit round trip of a box midpoint; it printed the same `pyvbmc.__file__`.
- `git status --porcelain --ignored` (ignoring `!!`) is empty at the end; HEAD is `0128550a`.

## `git log --oneline f8735567..HEAD`

```
0128550a test(vbmc): cover the candidate noise at uncertainty level 1
09813f57 fix(vbmc): weigh the resampling of the MCMC starting point by the samples
18807f33 fix(vbmc): refuse the MCMC step of the importance sampler that is not ported
ae8769d8 fix(vbmc): give no weight to a point the proposal cannot reach
c7ed0937 fix(vbmc): draw the samples that fess is asked for
cd50cfdb fix(acq): refuse a quantile outside the range the acquisition is defined on
21c13075 fix(acq): give VIQR and IMIQR the information keys of the base class
9d42ed74 fix(acq): read an acquisition function string as Python reads a call
e07d6fb1 fix(acq): round an integer coordinate away from zero at a half
36015c4a fix(acq): snap a single point to the integer grid
```

Diffstat over the range: 19 files, +869 / −89.

## Commits in order

| # | hash | finding | what it does | files | tests added or changed |
|---|---|---|---|---|---|
| 1 | `36015c4a` | W4-2 | `_real2int` snaps a one-dimensional point through a two-dimensional view of it, so it comes back in the shape given and is still snapped in place; the docstring states both shapes and the return | `abstract_acq_fcn.py`, `test_abstract_acquisition_function.py`, `test_vbmc_active_sample.py` | +`test_real2int_single_point`; +`test_search_result_is_snapped_with_integer_vars` with a new helper `_integer_var_state` |
| 2 | `e07d6fb1` | W4-4 | the integer snap is `sign(v)*floor(abs(v)+0.5)`, MATLAB's `round` (`misc/real2int_vbmc.m:7`), in place of `np.around` | `abstract_acq_fcn.py`, `test_abstract_acquisition_function.py` | **changed** `test_real2int` (its four bare comparisons now assert, with MATLAB's values); +`test_real2int_rounds_a_half_away_from_zero`, +`test_real2int_rounds_a_box_midpoint_away_from_zero` |
| 3 | `9d42ed74` | W4-3 | `string_to_acq` parses the string with `ast` as Python parses a call, literal arguments only, and raises a `ValueError` quoting the string for anything it cannot read, an unknown class name included | `utilities.py`, new `test_string_to_acq.py` | +`test_a_call_is_read_as_python_reads_it` (15 forms), +`test_a_literal_holding_a_comma_or_an_equals_is_read_whole`, +`test_what_cannot_be_read_raises_and_quotes_the_string` (12 cases) |
| 4 | `21c13075` | W4-18 | `AcqFcnVIQR.__init__` and `AcqFcnIMIQR.__init__` call the base constructor before setting their own keys | `acq_fcn_viqr.py`, `acq_fcn_imiqr.py`, the six shipped-acquisition test modules and `test_abstract_acquisition_function.py` | **changed** the `compute_var_log_joint` assertion in five modules from `not acq_info.get(...)` to `acq_info["…"] is False`; **added** that assertion to the VIQR and IMIQR `test_acq_info` |
| 5 | `cd50cfdb` | W4-19 | both constructors refuse a `quantile` that is not a number strictly between 0.5 and 1, with a `ValueError` naming the argument and the range; the range is in the documentation of the argument (a `Parameters` section is new on `AcqFcnIMIQR`) | `acq_fcn_viqr.py`, `acq_fcn_imiqr.py`, `test_acq_fcn_viqr.py`, `test_acq_fcn_imiqr.py` | in each module: +`test_a_quantile_outside_its_range_is_refused` (0, 0.25, 0.5, 1, direct and through `string_to_acq`), +`test_a_quantile_that_is_not_a_number_is_refused`, +`test_a_quantile_inside_its_range_is_accepted` |
| 6 | `c7ed0937` | W4-5 | `fess` takes the points from the pair `vp.sample` returns | `active_importance_sampling.py`, `test_active_importance_sampling.py` | +`test_fess_draws_the_points_it_is_asked_for` (default and explicit count, value against the definition), +`test_fess_checks_the_number_of_given_gp_means` |
| 7 | `ae8769d8` | W4-7 | a point at which every component of the proposal has density zero gets log weight `-inf` instead of raising; the other points keep their values bit for bit | `active_importance_sampling.py`, `test_active_importance_sampling.py` | +`test_proposal_pdf_gives_no_weight_where_the_proposal_is_zero` |
| 8 | `18807f33` | W4-6 | the `mcmc_importance_sampling` branch is removed; an acquisition that sets the flag is refused with a `NotImplementedError` at construction (new `_validate_search_acq_fcn_option`, called and defined next to `_validate_gp_hyp_sampler_option`) and at the top of `active_importance_sampling`; `active_importance_sampling_fess_thresh` registered in `INERT_OPTIONS` and its `.ini` description says it is not used | `active_importance_sampling.py`, `vbmc.py`, `options.py`, `advanced_vbmc_options.ini`, `test_active_importance_sampling.py`, `test_vbmc_init.py` | +`test_an_acquisition_asking_for_mcmc_importance_sampling_is_refused` (VIQR and IMIQR), +`test_an_acquisition_asking_for_mcmc_importance_sampling_is_rejected`, +`test_the_acquisitions_that_do_not_ask_for_it_are_accepted`; **changed** `_scenario()`, which no longer sets the now-inert option |
| 9 | `09813f57` | W4-1 | the maximum of the resampling log weights is taken over the samples, and the guard raises, saying what happened, only when no sample carries any weight; the `rng.choice` call is untouched | `active_importance_sampling.py`, `test_active_importance_sampling.py` | +`test_the_mcmc_chain_starts_at_a_sample_drawn_by_its_weight`, with a `_RecordingGenerator` subclass that keeps the `p` of the weighted draw |
| 10 | `0128550a` | W4-10 | tests only | `test_vbmc_active_sample.py`, `test_active_importance_sampling.py` | +`test_candidate_noise_is_one_observation_at_uncertainty_level_one`; **changed** `test_acq_log_f` (the three dead lines removed, a comment says what the stored MATLAB array is compared with) |

## Contract and check, per commit

1. **W4-2.** `misc/real2int_vbmc.m` indexes `xtemp(:,integervars)` on the row vector `private/activesample_vbmc.m:325` passes it. The regression test goes through `active_sample` with `integer_vars = [True, False]`, `search_optimizer = "cmaes"` and a mocked `cma.fmin` that improves on the sieve and returns a one-dimensional point off the grid; it asserts the step completes, that the returned point was off the grid, and that the acquired point is on it in original coordinates. Both new tests raise `IndexError` on the old code.
2. **W4-4.** `test_real2int`'s input, 0.5 under an identity transform, is the tie; with MATLAB's rule the first column is 1, not 0. The added tests are the ledger's tie set (`-2.5 … 2.5` → `-3 -2 -1 1 2 3`) and the reachable case: bounds `-0.5, 9.5`, midpoint 4.5, default probit transform. On this machine `pt(4.5)` is `7.54e-17` rather than the exact `0.0` the sheet records, but `pt.inverse(pt(4.5)) == 4.5` holds exactly, so the tie is reached; the test uses `pt(midpoint)` rather than a literal. All three fail with `np.around`.
3. **W4-3.** The ten forms of `verification/scripts/wave4_A3_string_to_acq.py` are in the accepted list, with `'AcqFcnVIQR(loss="iqr_reduction")'`, `"AcqFcnVIQR(0.9, 'iqr_reduction')"`, surrounding whitespace and two IMIQR forms. A literal holding `","` or `"="` is checked through a throwaway acquisition class installed on the package with `monkeypatch`, which is the only way to observe those two defects without a constructor that rejects the value. Nineteen of the 29 cases fail on the old parser.
4. **W4-18.** `AbstractAcqFcn.__init__` sets `compute_var_log_joint` and `log_flag`, and the class docstring describes the first as a hook every acquisition carries. The six shipped acquisitions assert the key and its value; a missing key now fails.
5. **W4-19.** The paper takes the upper quantile in (0.5, 1); outside it VIQR was all NaN or all `-realmax` and the run took the first sieve candidate every step. `numbers.Real` is the test for "a number", so NaN, a string and a bool are refused.
6. **W4-5.** `misc/fess_vbmc.m:4-12` supports the scalar form and defaults to 100. The test seeds `vp.rng`, calls `fess(vp, gp)` and `fess(vp, gp, 25)`, then redraws from the same seed and computes the fractional effective sample size from its definition (weights proportional to `exp(f_bar)/q`, `ESS/N`); it also covers the mismatch check on the drawing path with a matrix of means.
7. **W4-7.** MATLAB's arithmetic on an all `-Inf` row gives NaN and `private/activeimportancesampling_vbmc.m:148` turns it into `-Inf`; the caller's clean-up at `active_importance_sampling.py:201-203` is the port of that line, and the two now agree. The rows in support are computed by the same operations in the same order, so `test_active_sample_proposal_pdf` (the stored MATLAB array) is untouched and the test asserts the other rows are bit-equal to a call without the far point.
8. **W4-6.** Refusal in two places, as ruled. `get_mcmc_opts` keeps its caller (the MCMC step of IMIQR, `:209`); `fess` keeps none inside the package and stays with its tests; `widths`, `lb_tran` and `ub_tran` are still what step 2 hands the sampler; no import fell out of use (pycln is clean). `test_inert_options_are_the_declared_options_nothing_reads`, which recomputes the registry from the package, passes with the option added — nothing reads it any more.
9. **W4-1.** `private/activeimportancesampling_vbmc.m:206-208` builds `lnw` as a row and takes `max(lnw,[],2)`. The test records the `p` handed to `choice` (filtering on `replace is False`, since `vp.sample` also draws components with a `p`), and compares it, per hyperparameter sample, with the normalized `exp(lnw - max(lnw))` computed from the step-1 state (the same call with `active_importance_sampling_mcmc_samples = 0` and the same seed) plus `is_log_added`. The step-1 weights the test reads are renormalized by a constant, which cancels; the comparison is at `rtol=1e-10`. It also asserts `p.max()/p.min() > 10`, which the uniform weights of the old rule fail; on the old code the recorded `p` is exactly `1/23` everywhere.
10. **W4-10.** (a) `uncertainty_handling=True` without `specify_target_noise`, `D = 2`, `search_optimizer = "none"`, an initial design of eight points whose two identical starting rows the logger pools into one row with two evaluations. One `active_sample` step with an acquisition that records `gp.temporary_data["sn2_new"]`, the hyperparameter samples and `gp.s2`; the assertions are `sn2_new == mean_s(exp(2 h0) + exp(h1))` at every training point, `gp.s2` at the repeated row equal to 0.5, the GP's own noise there equal to `exp(2 h0) + exp(h1)/2` per sample, and `sn2_new` at that row not equal to the mean of those — the last is what a port that handed the acquisitions the pooled variance would fail. (b) The three dead lines of `test_acq_log_f` are gone and the comment says that the stored array holds `log_isbasefun`, which adds the variational log density that `AcqFcnVIQR.is_log_full` does not.

## Test runs, and results

All from the worktree root, `-q -p no:cacheprovider`, with the three thread variables set:

- `pyvbmc/testing/acquisition_functions/` — the ten files I touched or that build a VIQR (`test_abstract_acquisition_function.py`, `test_string_to_acq.py`, `test_acq_fcn.py`, `test_acq_fcn_log.py`, `test_acq_fcn_noisy.py`, `test_acq_fcn_vanilla.py`, `test_acq_fcn_viqr.py`, `test_acq_fcn_imiqr.py`, `test_acq_fcn_viqr_losses.py`, `test_viqr_kernel_reuse.py`) — **126 passed**
- `pyvbmc/testing/vbmc/test_active_importance_sampling.py`, `test_vbmc_active_sample.py`, `test_vbmc_init.py`, `test_options.py` — **248 passed**
- Final combined run of all fourteen files at HEAD: **374 passed in 19.1 s**

Every fix was seen to fail first, by reverting the hunk (Edit in and out, never a bare `git stash`; for commit 3 the new module was copied to the scratchpad and `git checkout --` used on the one file). No whole test directory in the final rounds, no full suite, no `optimize()` run, no oracle or golden command, no install. (One early exploratory run did cover the whole `pyvbmc/testing/acquisition_functions/` directory, 123 passed; after that I named files.)

## Changelog sentences (for a user of release 1.0.4)

- **W4-2** — "A run that sets `options['integer_vars']` together with a search optimizer no longer ends in an `IndexError` after its initial design: the point the local search returns is snapped to the integer grid whatever its shape." Cannot stop a script; such a run used to fail outright.
- **W4-4** — "A coordinate of an integer variable that falls exactly halfway between two integers is rounded away from zero, as MATLAB VBMC rounds it, instead of to the nearer even integer." Can change what a run with `integer_vars` returns (only on an exact half, which a starting point at the midpoint of a box with an even number of levels produces), so it belongs in the "Upgrading from 1.0.4" list.
- **W4-3** — "An entry of `options['search_acq_fcn']` given as a string is read as Python reads a call: a keyword name with spaces around it, and a literal holding a comma or an equals sign, are read correctly, and a string that cannot be read raises instead of being mis-parsed or silently losing an argument." Can stop, or change the results of, a script whose string held a value with an `=` (which used to be dropped, the default taking its place), so it belongs in the "Upgrading from 1.0.4" list.
- **W4-18** — "`AcqFcnVIQR().get_info()` and `AcqFcnIMIQR().get_info()` carry `compute_var_log_joint`, the key the information dictionary of every acquisition function documents; reading it no longer raises a `KeyError`." Cannot stop a script.
- **W4-19** — "`AcqFcnVIQR` and `AcqFcnIMIQR` refuse a `quantile` that is not a number strictly between 0.5 and 1, where such a value used to give an acquisition that was all NaN and a search that took the first candidate it was offered." Can stop a script that passed one, so it belongs in the "Upgrading from 1.0.4" list.
- **W4-5** — "`pyvbmc.vbmc.active_importance_sampling.fess` works when its third argument is a number of samples to draw from the variational posterior, its documented default." Cannot stop a script.
- **W4-7** — "A point at which the importance-sampling proposal of a noisy run has zero density is given weight zero instead of ending the run with `ValueError('Invalid value.')`." Cannot stop a script.
- **W4-6** — "An acquisition function that sets `acq_info['mcmc_importance_sampling']` is refused when the `VBMC` object is created, and by `active_importance_sampling` for any other caller: the refinement of the importance samples by an ensemble sampler that the flag asks for is not ported and never ran. `options['active_importance_sampling_fess_thresh']`, which only that branch read, has no effect and is reported as having none." Can stop a script that supplied such an acquisition — it used to fail part-way through a run — so it belongs in the "Upgrading from 1.0.4" list.
- **W4-1** — "A run with `AcqFcnIMIQR` draws the starting point of each MCMC chain of the importance sampler in proportion to the importance weights, as MATLAB VBMC does, instead of uniformly. Such runs give different results." Changes what a script returns, so it belongs in the "Upgrading from 1.0.4" list.
- **W4-10** — tests only; no changelog line.

Commits 1 to 8 and 10 change no number of a run at the default options (`integer_vars` is empty by default, the string parser is reached only for a string entry, the two refusals fire on values that used to fail anyway, and the zero-density point is unreachable through the module's own draws). Commit 9 changes the numbers of runs with `AcqFcnIMIQR`, and so the stored `acq_AcqFcnIMIQR` oracle reference, as predicted; I ran no oracle command and no golden replay.

## What I stopped on

Nothing. All ten items are done as ruled.

## For the orchestrator

- **`AcqFcnVIQR.is_log_full` has no caller inside the package after commit 8.** VIQR takes the step-0 branch (`variational_importance_sampling`), whose only use of `is_log_full` was the removed hook; step 2 is reached by IMIQR alone. It is still pinned against the stored MATLAB array by `test_acq_log_f`, and it is the counterpart of MATLAB's own `acqviqr_vbmc.m` `'islogf'` branch (`matlab_side_defects.md` candidate 2 of the P3 slice), so I left it. The same holds for `fess`, which the ruling says to keep.
- **`_validate_search_acq_fcn_option` runs in `__init__` only**, as `_validate_noise_shaping_option` and `_validate_gp_hyp_sampler_option` do. A run restored with `load(new_options={"search_acq_fcn": …})` is not re-validated and would meet the `NotImplementedError` inside `active_importance_sampling` instead. I matched the existing pattern; say the word if the check should also run on `load`.
- **`_scenario()` in `test_active_importance_sampling.py` no longer sets `active_importance_sampling_fess_thresh`** (it set it to 0, away from the 0.9 default). Left in place it would have been an inert option set by the user, which `Options._warn_inert_options` reports whenever such an object is validated.
- **The `.ini` description of `active_importance_sampling_fess_thresh`** keeps its original wording, including the leading `%` and the `threhsold` spelling it shares with the line above, and gains the parenthetical that says it is not used. Both oddities are pre-existing; say the word if they should be cleaned up.
- **No `MANIFEST.in` change** is needed for the new `pyvbmc/testing/acquisition_functions/test_string_to_acq.py`: that directory has an `__init__.py`, so its modules ship as package sources.
- The stored oracle fixtures and their option snapshots name neither `active_importance_sampling_fess_thresh` nor the `mcmc_importance_sampling` flag (checked by grep over `pyvbmc/`), so commit 8 touches no oracle input.
