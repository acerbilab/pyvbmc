# P2 — initial design and active-sampling search, internal-correctness track

Raw reviewer report, wave 5 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-21. One Opus agent, fresh
context, read-only on the repositories, reading the code at `831acef0`, the
head of `dev-port-review` after the fix passes of waves 1 to 4; it did not
read the MATLAB source, and it did not see the comparison report of this
slice, which wave 1 produced on the code of `f91fdf0`. Its brief carried a
first question, which the report answers before its findings. Its check
scripts ran in the session scratchpad and are kept only on the machine that
ran them (`dev/scripts/runs/LOCAL.md`). The text below is the agent's final
message, unedited.

---

# P2 — initial design and active-sampling search (internal-correctness track)

## 1. Coverage

**Read completely**
- `pyvbmc/vbmc/active_sample.py` (whole file: `_selection_policy_callback`, `_log_search_failure`, `active_sample`, `_get_search_points`).
- `pyvbmc/acquisition_functions/abstract_acq_fcn.py` (`__call__`, `_real2int`, `_sq_dist`, `_estimate_observation_noise`, `_check_quantile`).
- `pyvbmc/function_logger/function_logger.py` (whole file, `_record` in particular).
- `pyvbmc/vbmc/iteration_history.py`, `pyvbmc/timer/timer.py`, `pyvbmc/stats/get_hpd.py`.
- `pyvbmc/testing/vbmc/test_vbmc_active_sample.py` (all 2246 lines).
- `dev/experiments/port_review_20260919/known_differences.md`, sections "Slice P2", "Slice P3", "Slice P4", "Settled non-differences" and the two P1b entries named in my brief (the only file under `dev/` I opened).

**Read in the parts that bear on the slice**
- `pyvbmc/vbmc/vbmc.py`: `_setup_vbmc` (optim_state construction, `integer_vars`, search bounds, uncertainty levels), the `active_sample` call site and what it reads back, `_initialize_precomputed_evaluations`, `_optim_state_record`, `_trim` (the `X_flag` write at `:2096`), `_normalize_bounds` (`pyvbmc/vbmc/_bounds.py:263-275`).
- `pyvbmc/vbmc/gaussian_process_train.py`: `train_gp`, `_get_training_data`, `reupdate_gp`, `_get_gp_training_options` (its read of `recompute_var_post`), `_lean_gp`.
- `pyvbmc/vbmc/variational_optimization.py`: signatures and contracts of `_gp_log_joint`, `_neg_elcbo`, `optimize_vp`, `update_K`.
- `pyvbmc/variational_posterior/variational_posterior.py`: `sample`, `moments`, `__deepcopy__`, the `rng` property.
- `pyvbmc/parameter_transformer/parameter_transformer.py` (construction and the bound handling), `pyvbmc/vbmc/options.py` (`INERT_OPTIONS`, `integer_vars_mask`, `_warn_inert_options`).
- gpyreg: `GP.update` (the whole rank-one branch and the `s2` bookkeeping), `GaussianNoise.compute` / `hyperparameter_count`, `GP.clean`.
- `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini` (whole file), `docsrc/source/faq.md` §"Does VBMC support inference with integer parameters?", `pyvbmc/vbmc/README.md` §`active_sample.py`.
- `pyvbmc/testing/oracles/_oracles.py`: `prepare_gp_for_acq`, `active_sample_step`.

**Skimmed**: `pyvbmc/whitening/whitening.py` (only the lines that write `plb_tran`, `pub_tran`, `lb_search`, `ub_search`, `search_cache`, `recompute_var_post`); `pyvbmc/vbmc/active_importance_sampling.py` (only its entry point and that it draws through `vp.rng`).

**Not reached**: the bodies of the individual acquisition functions (VIQR/IMIQR/Noisy) beyond their use of `sn2_new` and `gp_length_scale`; the interior of `optimize_vp`/`_sieve`; the MATLAB sources (out of track).

**Checks run** (all in the scratch directory, `OMP/OPENBLAS/MKL_NUM_THREADS=1`, `pyvbmc.__file__ = C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\__init__.py`, gpyreg from the sibling checkout, numpy 2.5.2 / scipy 1.18.1 / cma 4.4.4):
- `chk_integer.py` — initial design (plausible and narrow) and a provided `x0` under `integer_vars`; `_get_search_points` candidates before and after `_real2int`; idempotence of the snap; snapped candidates against the search and hard bounds.
- `chk_integer2.py` — one `active_sample` step per local-search branch (`none`, `cmaes`, `Nelder-Mead`, the `D=1` bounded search) with an integer variable; a repeated observation; `lb_search`/`ub_search` in original coordinates.
- `chk_cache.py` — a cached starting point outside the search bounds; a cached starting point off the integer grid; what `optim_state["search_cache"]` holds after a step; a candidate at `ub_search` under the snap.
- `chk_misc.py` — `acq_hedge=True`; the kwargs handed to `scipy.optimize.minimize`; `cma.fmin` honouring its bounds; a Monte Carlo check of the `df=3` draws against the exact Student-t.
- `chk_nm.py`, `chk_nm2.py` — Nelder-Mead vs CMA-ES against the search box; the accounting of `_get_search_points` sources for several requested sizes; the rounding of the fractions; the two `np.cov` normalizations of the HPD branch.
- `chk_rng.py` — the global NumPy stream before/after a seeded step; reproducibility across the three branches.
- `chk_trim.py` — an acquired point coinciding with an inactive (trimmed) logger row.

No test suite was run; nothing was written inside either repository.

---

## 2. First question — every path by which a point reaches the target or the GP training set with `integer_vars` set

Setup used throughout: `D = 2`, variable 0 an integer with hard bounds `-10.5 / 10.5`, variable 1 unbounded, `plb/pub = -3 / 3`. The snap is `AbstractAcqFcn._real2int`, which inverts to the original space, rounds the integer coordinates half-away-from-zero, transforms back, and writes the integer coordinates **in place** into the caller's array.

**Which transformer.** Two different objects are named. `active_sample.py:383` and `:624` pass `function_logger.parameter_transformer`; `AbstractAcqFcn.__call__:86` passes `vp.parameter_transformer`. `VBMC.optimize` assigns `self.function_logger.parameter_transformer = self.vp.parameter_transformer` immediately before the call (`vbmc.py:1405-1408`), so on entry they are the same object; if `active_sample_vp_update` is on, `optimize_vp` can return a posterior carrying a deep copy, numerically identical. No path snaps with a stale transformer.

Path by path:

| path | on the grid when the target is evaluated? | inside the hard bounds? | snapped where, with which transformer | acquisition value at the evaluated point? |
|---|---|---|---|---|
| provided `x0` (initial design) | **no** — passed through verbatim (`active_sample.py:206, 216-218`). Measured: `x0 = 3.4` reaches the target as `3.4`. | yes (construction checks) | never snapped | n/a (no acquisition) |
| `plausible` design | **no**. Measured first coordinates `-1.2215, -2.9698, 1.8081, …` | yes (drawn inside the plausible box) | never snapped | n/a |
| `narrow` design | **no**. Measured `-0.1226, -0.3034, 0.1822, …` | yes (clipped to the plausible box) | never snapped | n/a |
| starting cache consumed by the design | **no** (same code path as provided `x0`) | yes | never snapped | n/a |
| starting cache drawn by the sieve | yes | yes | `_get_search_points` transforms the cached row, then `active_sample.py:383` snaps it, in transformed space, with the logger's transformer | yes for the acquisition — **but the stored `y_orig` was computed at the unsnapped point** (finding F1) |
| search-cache / heavy-tail / mvn / HPD / box / VP draws | yes | yes | `active_sample.py:383`, then again inside `AbstractAcqFcn.__call__` (idempotent — verified) | yes |
| `search_optimizer = "none"` | yes (measured `x_orig = [0.0, 0.000286]`) | yes | as above | yes |
| `cmaes` | yes (measured `[0.0, 0.000286]`) | yes | the objective snaps each population member in place, then `:624` snaps `res[0]` | yes — the snap is idempotent on the point the objective evaluated, so the compared value and the evaluated point agree |
| `bounded` (`D = 1`) | yes (measured: the target saw `0.0`) | yes | `acq_fun_1d` builds a fresh `np.atleast_1d(x)`, so `res.x` comes back unsnapped and `:624` snaps it to exactly the point whose value is `res.fun` | yes |
| `Nelder-Mead` | yes (measured `[0.0, 0.000286]`) | **not guaranteed** — the simplex is unbounded (finding F4); I drove it to `x_orig ≈ 2.8e44` with a monotone acquisition | no | yes |
| repeated observation | **no, deliberately** — `X_acq = X_train_cand[[idx]].copy()` (`:460`) restores the *unsnapped* stored row so the logger pools it | yes | not snapped on purpose | **no** — `acq_fast[idx]` was computed on the snapped copy in `X_search`; the code comment at `:454-459` states this |
| expansion of the search bounds | n/a (no evaluation) | `lb_tran`/`ub_tran` are `∓inf` for every variable, so the expansion is effectively unbounded | tested in transformed units, which is what the file's own `# ADD DIFFERENT CHECKS FOR INTEGER VARIABLES!` comment at `:822` asks about | n/a |

**Can the snap move a point after its acquisition value was compared?** Only on the repeat path, and there it is deliberate and documented in the code. Everywhere else the value the comparison used was computed at the snapped point, because the acquisition wrapper snaps its own input and the snap is idempotent (verified: `_real2int(_real2int(X)) == _real2int(X)`).

**Which paths deliver an off-grid point.** By omission: the whole initial design (provided `x0`, the `plausible` and `narrow` designs, the starting cache that the design consumes). By design: the repeated observation. Additionally, the sieve's clip to `[lb_search, ub_search]` happens *before* the snap in `_get_search_points`, so a snapped candidate can end up outside the search box: `ub_search` is at `9.7947` in original coordinates in my setup, and a candidate sitting exactly on it snaps to `10.0 > ub_search` (verified). It stays inside the hard bounds, because the transform maps the open interval onto the reals and the hard bounds sit half an integer outside the range.

**Does the documentation tell a user the initial design is not snapped?** No, and the documentation is in fact self-contradictory. The option comment (`advanced_vbmc_options.ini:4`) describes only how `integer_vars` is written and the half-integer bound requirement; it says nothing about the initial design, nor that `x0` must itself be on the grid. `docsrc/source/faq.md:431` says the opposite of both: "No, VBMC does not support integer parameters (that is, variables forced to be integers …)". Nothing anywhere warns that the first `fun_eval_start` evaluations are made off the grid. See F5.

**One consequence worth stating even though it is not a defect of the snap itself.** Because the grid is discrete, the acquisition search collides with points already in the training set. In my `D = 1` integer run the bounded search's snapped result was exactly `x_orig = 0.0`, an existing (noiseless) training input: `FunctionLogger._record` pooled it into that row, `Xn` did not advance, one target call was spent, no training point was added, and `update1` became false so the whole posterior was recomputed. With integer variables this is a routine outcome, not an edge case, and there is no guard against it. When the coinciding row has been trimmed (`X_flag` false), the evaluation is lost entirely — see F6.

---

## 3. Findings

### F1. A cached starting point's stored value is recorded at a point the sieve moved
- Location: `pyvbmc/vbmc/active_sample.py:1072` (the clip), `:383` (the snap), `:679-707` (the reuse); MATLAB: "not read (internal track)"
- Category: state/caching
- Proposed classification: suspected defect
- Confidence: medium
- `_get_search_points` builds the cache candidates as `parameter_transformer(x0[idx_cache])` and then clips every row with `search_X = np.minimum(np.maximum(search_X, lb_search), ub_search)`. Back in `active_sample`, `_real2int` snaps the same array. `idx_cache` still points at the original cache row, so when such a candidate wins, `y_orig = optim_state["cache"]["y_orig"][idx]` is handed to `function_logger.add(xnew, y_orig)` with `xnew` the **clipped and/or snapped** point. The stored value `f(x)` is then recorded, and trained on, at a different input `x'`. The cache is a record of evaluations already made; moving the point invalidates the pair. Either the moved rows should lose their cache index (as the local-search branch correctly does at `:629`), or the cache rows should be excluded from the clip/snap.
- Consequence if real: one training pair whose target value does not belong to its input, which biases the GP wherever the two disagree. Reproduced: a cached point at `x_orig = [50, 0]` with stored value `123.5` was recorded as `X_orig = [15, 0], y_orig = 123.5`. Reachability at defaults is narrow: `_normalize_bounds` widens PLB/PUB to cover every `x0` (`_bounds.py:273-274`), so on a fresh run the cache lies inside the search box; the clip bites after a warp, which resets `lb_search`/`ub_search` from the warped training box (`whitening.py:309-310`) while the cache is still transformed afresh from original coordinates. The snap variant needs only `integer_vars` plus an off-grid `x0` with a supplied value, and off-grid `x0` is nowhere forbidden: reproduced, cached `[2.4, 0.1]` with `y = -77.0` recorded at `[2.0, 0.1]`. Both require more starting points than `fun_eval_start` so that valued rows survive in the cache.
- Suggested reproduction: ran it — `chk_cache.py` sections (A) and (B), outputs quoted above.
- Test adequacy: no. `test_acquiring_a_cached_point_reuses_its_value` uses a cache point already inside the search box and on no grid, so the clip and the snap are both inert there; it asserts `np.allclose(X_orig[last], x_cached[0])`, which the defect would break, but only in a state the test never builds.

### F2. `acq_hedge = True` raises `UnboundLocalError` at the first acquisition
- Location: `pyvbmc/vbmc/active_sample.py:320-323` and `:408`; MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- `idx_acq` is assigned only inside `if not options["acq_hedge"]:`. With the option on, nothing assigns it and `SearchAcqFcn[idx_acq]` at `:408` raises `UnboundLocalError: cannot access local variable 'idx_acq'`. `acq_hedge` is a declared, documented option ("Hedge on multiple acquisition functions"), it is *not* in `INERT_OPTIONS` (only `acq_hedge_decay` and `acq_hedge_iter_window` are, `options.py:25-26`), it is read in two places (`vbmc.py:1048`, `active_sample.py:320`) so the inert-option scan passes, and `_validate_option_values` does not refuse it the way it refuses `noise_shaping` and a non-`slicesample` `gp_hyp_sampler`. A user who sets it gets neither the hedge, nor a fall-back, nor a refusal — it crashes mid-run. The P2 sheet entry on the hedge describes the code ("`idx_acq` is never chosen") but not that the consequence is a crash.
- Consequence if real: any run with `acq_hedge=True` dies at the first active-sampling step of iteration 1, after the initial design has been paid for. Deterministic.
- Suggested reproduction: ran it — `chk_misc.py`, first block; `UnboundLocalError: cannot access local variable 'idx_acq' where it is not associated with a value`.
- Test adequacy: no test sets `acq_hedge`. The inert-option scan cannot catch it because the option *is* read.

### F3. "Remove selected points from search set" has no effect; the search cache keeps the acquired point at rank 0
- Location: `pyvbmc/vbmc/active_sample.py:430-436` and `:466-468`; MATLAB: "not read (internal track)"
- Category: control flow / state-caching
- Proposed classification: suspected defect
- Confidence: medium
- `optim_state["search_cache"] = X_search[inds]` is assigned at `:432`, *before* the two `np.delete` calls at `:467-468`; `np.delete` returns new arrays bound to the locals `X_search` and `idx_cache`, which are never read again in the loop body (I checked every use from `:467` to `:832`). So (a) the deletion the comment announces is dead code, and (b) the search cache retains the just-acquired point as its first row — and `_get_search_points` takes `search_cache[:N_search_cache]`, i.e. exactly the head of that sorted array. The point just evaluated is therefore the first candidate offered at the next step. Two further consequences of the same assignment: the cached rows include the snapped *training inputs* appended for the repeat mechanism, so with `search_cache_frac > 0` exact repeats can re-enter the sieve at the next step regardless of `max_repeated_observations` and the streak cap; and rows that came from the starting cache lose their `idx_cache` provenance on the way through the search cache, so if one wins later, the target is called instead of its stored value being reused and the row is not removed from the starting cache.
- Consequence if real: dormant at the shipped default (`search_cache_frac = 0`). With the option on, the sieve re-offers the point just evaluated; on a noiseless target picking it again costs one evaluation, pools into the existing row and adds no training point.
- Suggested reproduction: ran it — `chk_cache.py` section (C): with `search_cache_frac = 0.5` and `ns_search = 32`, `optim_state["search_cache"]` came back with shape `(32, 2)` and `search_cache[0]` equal to the acquired point.
- Test adequacy: no. `test_get_search_points_all_search_cache` writes `optim_state["search_cache"]` by hand and never exercises what `active_sample` stores there; no test asserts that the acquired point leaves the search set or the search cache.

### F4. The Nelder-Mead branch is unbounded, ignores `search_max_fun_evals`, and uses a function tolerance as a step tolerance
- Location: `pyvbmc/vbmc/active_sample.py:609-619`; MATLAB: "not read (internal track)"
- Category: defaults / control flow
- Proposed classification: suspected defect
- Confidence: high (the facts); the substitution itself is recorded, the omissions are not
- The call is `minimize(acq_fun, x0, method="Nelder-Mead", tol=tol_fun)`. Three things follow. (i) No `bounds` are passed, so `lb_search`/`ub_search` — which the `cmaes` branch computes two lines earlier at `:509-521` and hands to cma, and which the `bounded` branch also honours — constrain nothing. The acquisition's own hard-bound mask (`abstract_acq_fcn.py:181-186`) cannot substitute: for an unbounded variable `lb_eps_orig = -inf + inf = NaN` (`vbmc.py:927-929`), every comparison against NaN is false, and the mask never fires. (ii) `search_max_fun_evals` is not passed at all, so the documented "Max number of acquisition fcn evaluations during search" is silently replaced by SciPy's default `maxiter = maxfev = 200*D`. Captured kwargs: `{'method': 'Nelder-Mead', 'tol': 1e-12}` against `search_max_fun_evals = 2000`. (iii) SciPy's `tol` for Nelder-Mead sets *both* `xatol` and `fatol`; `tol_fun` is a tolerance on the acquisition value (`1e-2` for a log acquisition, `|f_val_old|*1e-3` otherwise), so it is also applied as an absolute step tolerance in transformed coordinates, where it has no meaning. The known-differences sheet records that "Nelder-Mead (SciPy's unbounded simplex) stands in for MATLAB's bounded `fmincon`", which covers the substitution but not the dropped budget or the conflated tolerance.
- Consequence if real: with `search_optimizer = "Nelder-Mead"` an acquisition that keeps improving in one direction drives the acquired point arbitrarily far outside the search box, and that point is evaluated by the target and added to the GP. Reproduced: with a linear acquisition the acquired point came back at `x_tran ≈ [4.7e43, 6.2e42]`, i.e. `x_orig ≈ [2.8e44, 3.7e43]`, against a search box of `±2.5`; the CMA-ES branch on the identical state stayed inside at `[1.857, 2.368]`. The option is not the default.
- Suggested reproduction: ran it — `chk_nm2.py`, first block (and `chk_misc.py` for the captured kwargs).
- Test adequacy: no. No test exercises `search_optimizer = "Nelder-Mead"` at all, and no test in the file asserts that an acquired point lies inside `[lb_search, ub_search]` in any branch except the one-dimensional one.

### F5. The initial design is off the integer grid, and the documentation contradicts itself about `integer_vars`
- Location: `pyvbmc/vbmc/active_sample.py:127-218`; `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:4`; `docsrc/source/faq.md:428-432`; MATLAB: "not read (internal track)"
- Category: control flow / defaults
- Proposed classification: suspected defect
- Confidence: medium
- `_real2int` is applied to search candidates (`:383`) and to the local search's result (`:624`), never to the initial design. The provided `x0`, the `plausible` draws and the `narrow` draws all reach the target and the GP training set at non-integer coordinates — measured first coordinates `-1.2215, -2.9698, 1.8081` (plausible) and `-0.1226, -0.3034, 0.1822` (narrow), and a provided `x0 = 3.4` evaluated as `3.4`. That is `fun_eval_start = 10*ceil((D+1)/10)` observations, the entire basis of the first GP, taken at inputs the user declared impossible. It also leaves the training set permanently off-grid, which is what forces the repeat path to re-evaluate an unsnapped row (`:454-460`) and what makes F6 reachable. Whatever the intended behaviour, the documentation does not state it: the option comment says only how the mask is written and where the hard bounds go, and the FAQ entry says integer parameters are not supported at all. A user reading either one cannot predict what happens.
- Consequence if real: for a target that is only defined at integers, the initial design is a set of invalid calls; for one that merely rounds internally, the GP is fitted to a step function with training inputs interior to the steps.
- Suggested reproduction: ran it — `chk_integer.py`, first two blocks.
- Test adequacy: no. `test_search_result_is_snapped_with_integer_vars` is the only `integer_vars` test of the local search and it mocks `cma.fmin`; no test looks at the initial design under `integer_vars`, and no test compares the option comment with the FAQ.

### F6. An acquired point that coincides with an inactive logger row is evaluated, pooled into that row, and never reaches the GP
- Location: `pyvbmc/vbmc/active_sample.py:694-720` with `pyvbmc/function_logger/function_logger.py:685-689`; MATLAB: "not read (internal track)"
- Category: cross-module
- Proposed classification: suspected defect
- Confidence: medium
- `FunctionLogger._record` looks for a duplicate with `(self.X == x).all(axis=1)` over the **whole** array, not over `X_flag`. Trimming at the end of warm-up only clears `X_flag` (`vbmc.py:2096`) and leaves `X` in place, so an acquired point equal to a trimmed row is pooled into it: `y_orig` is averaged, `n_evals` incremented, `X_flag` stays false, `Xn` does not advance. `active_sample` then computes `update1 = n_evals[idx_new] == 1`, which is false, so `reupdate_gp` rebuilds from `X_flag` — unchanged. One target evaluation is spent, `optim_state["N"]` and `n_eff` do not move, `y_max` ignores the new value, and the observation reaches neither the GP nor the training set. Either the duplicate scan should be restricted to live rows, or a pooled hit on a dead row should reactivate it.
- Consequence if real: a silently lost function evaluation. Exact coincidence is essentially unreachable in continuous coordinates, but with `integer_vars` the snap quantizes the candidate space and collisions with stored inputs are routine (see the last paragraph of the first-question answer); the trimmed-row variant is then a real possibility after warm-up ends. Only a handful of evaluations at worst, but each is silently wasted.
- Suggested reproduction: ran it — `chk_trim.py`: after marking one row inactive and steering the sieve onto exactly that input, `func_count 12 -> 13`, `Xn 11 -> 11`, live rows `11 -> 11`, that row's `n_evals = 2` with `X_flag = False`, GP training rows `11 -> 11`.
- Test adequacy: no. `test_repeated_observation_candidates` only ever repeats *live* rows (`X_train_cand` is built from `X_flag`), so the inactive-row case is untested.

### F7. The initial design's target time is not charged to `fun_time`
- Location: `pyvbmc/vbmc/active_sample.py:208-218` against `:693-708`; MATLAB: "not read (internal track)"
- Category: state/caching
- Proposed classification: suspected defect
- Confidence: low
- The active-sampling branch wraps every evaluation in `timer.start_timer("fun_time") … stop_timer("fun_time")`; the initial-design branch has no timer at all, and `main_timer` is reset at the top of each iteration (`vbmc.py:1233`) and recorded into the iteration history at `:1603`. Iteration 0's record therefore has no `fun_time` key, and `Timer.get_duration` on it logs a warning and returns `None`. Iteration 0 is where the bulk of a run's target time is spent, so the recorded timings are not comparable across iterations. Related, same lines: `start_timer` silently ignores a start on an already-running key and `stop_timer` is not in a `finally`, so a target that raises leaves `fun_time` running and charges the whole interval to the next stop; the same holds for `gp_train` at `:740/756` and `:802/815`.
- Consequence if real: diagnostics only — nothing in the package reads `main_timer`'s `fun_time` (the `FunctionLogger`'s per-row `fun_eval_time` comes from its own local `Timer`), and the commented-out time-cost block at `:228-247` is the only would-be consumer.
- Suggested reproduction: construct a `VBMC`, call `active_sample(gp=None, …)`, and inspect `pyvbmc.timer.main_timer._durations` — the key is absent. I did not run this one; it follows from reading, and `test_in_loop_gp_posterior_update_leaves_no_timer_running` already inspects `main_timer._durations` the same way.
- Test adequacy: no. That test checks only `gp_train` and `variational_fit`.

### F8. `recompute_var_post` is saved and restored around the in-loop update but is never changed
- Location: `pyvbmc/vbmc/active_sample.py:268` and `:849`; MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: unsure
- Confidence: low
- `recompute_var_post_old = optim_state["recompute_var_post"]` and the matching restore bracket a block that never writes the key: the only writers in the package are `vbmc.py:973`, `:1507`, `:2106` and `whitening.py:318`, all outside `active_sample`. The in-loop `train_gp` therefore reads whatever the main loop left, and the save/restore pair is dead. Either the pair is vestigial, or an assignment that the in-loop refits were meant to make (so that they take the cheap schedule of `_get_gp_training_options`, `gaussian_process_train.py:605-628`) is missing. The neighbouring `entropy_alpha` save/restore *is* live (`:772`). `optim_state["hyp_dict"] = hyp_dict` at `:851` is likewise a no-op, but harmlessly so: `train_gp` mutates the dict it is given and returns the same object.
- Consequence if real: if an assignment is missing, the in-loop refits use the full burn-in and initial-design schedule rather than the reduced one, which costs time and changes the fitted hyperparameters. Only with `active_sample_gp_update=True` and `sample_count > 1`, both non-default.
- Suggested reproduction: `grep -rn recompute_var_post pyvbmc/ --include=*.py`; the writer list above is the whole of it. Ran the grep; no run needed.
- Test adequacy: no test observes `recompute_var_post` during active sampling.

### F9. Two small formula inconsistencies inside `_get_search_points`
- Location: `pyvbmc/vbmc/active_sample.py:1010` against `:1015`, and `:1039-1040` against `vbmc.py:1071-1081`; MATLAB: "not read (internal track)"
- Category: formula
- Proposed classification: unsure
- Confidence: low
- (a) The Gaussian fitted to the high-posterior-density subset uses `np.cov(X_hpd, bias=True, rowvar=False)` when the subset is non-empty and `np.cov(X, rowvar=False)` — the unbiased normalization — in the empty-subset fallback. Same purpose, two normalizations differing by `N/(N-1)`; on a four-point example `1.5` against `2.0` for the same coordinate. (b) The box sampler's fallback when the search bounds are not finite uses `plb_tran ± 3*(pub_tran - plb_tran)`, a hard-coded 3, where the run's own search bounds are built with the `active_search_bound` option (default 2). Both are in code paths that are effectively unreachable at the shipped settings: `hpd_search_frac = 0`, and `lb_tran`/`ub_tran` are `∓inf` for every variable so `lb_search`/`ub_search` are always finite.
- Consequence if real: with `hpd_search_frac > 0` and an `hpd_frac` small enough that `round(hpd_frac*N) == 0`, the fallback Gaussian is slightly wider than the branch it stands in for. Nothing at defaults.
- Suggested reproduction: ran the normalization comparison in `chk_nm2.py`, last block.
- Test adequacy: `test_get_search_points_all_hpd_search_empty_get_hpd` exercises the empty branch but asserts only the output shape, so it would not notice either normalization.

### F10. The integer snap is applied after the search-bound clip, so a candidate can end up outside the search box
- Location: `pyvbmc/vbmc/active_sample.py:1072` then `:383`; and `:818-832` with its own `# ADD DIFFERENT CHECKS FOR INTEGER VARIABLES!`; MATLAB: "not read (internal track)"
- Category: indexing/shape (ordering of two coordinate-space operations)
- Proposed classification: possibly intentional
- Confidence: low
- `_get_search_points` ends by clipping every candidate into `[lb_search, ub_search]`; `active_sample` then snaps, which can move a coordinate back out. With the setup above, `ub_search` sits at `9.7947` in original coordinates while the nearest grid point above it is `10`, and a candidate placed exactly at `ub_search` snaps to `10 > ub_search` (verified). The hard bounds are never violated, because the transform maps the open interval onto the reals and the bounds sit half an integer outside the range. The same asymmetry is behind the file's own request for different bound-expansion checks with integer variables: the expansion at `:823` and `:828` compares transformed-space distances against `0.05*(ub_search - lb_search)`, a test that has no relation to the integer step.
- Consequence if real: a handful of sieve candidates just outside the nominal search box near a bound; harmless for the acquisition, but it means "the search bounds are respected" is not true of the candidate set that the acquisition actually ranks.
- Suggested reproduction: ran it — `chk_cache.py` section (D).
- Test adequacy: no test checks the candidate set against the search bounds after the snap. `test_get_search_points_search_bounds` checks the clip only, before any snapping.

---

## 4. Test adequacy notes

Things I verified and found correct, so that they are not re-derived: the noise variance handed to the acquisitions (`sn2_new = mean_s noise.compute(hyp[cov_N:cov_N+noise_N], gp.X, gp.y, S²·n_evals)`) is the variance of a *single new* observation and is indexed through `hyperparameter_count()` at all three uncertainty levels, and `S²·n_evals` is the right de-pooling of the logger's precision-weighted `S`; the length scale is the geometric mean over hyperparameter samples of `hyp[:D]` and `X_rescaled` is consistent with it; the heavy-tailed draws are an exact multivariate `t` (MC variance `2.875` against `3`, `P(|x|>2) = 0.1387` against `0.1393`); the source counts of `_get_search_points` always sum exactly to the number asked for (checked at 1, 2, 3, 5, 7, 100, 8191, 8192 and with a partly filled cache) and `idx_cache` stays aligned; `iteration_history["r_index"][-1]` really is the last recorded iteration (arrays grow to exactly `iteration+1`) and is short-circuited away at defaults; the `vp0` rollback returns a posterior that still shares the run's generator and gets the logger's transformer back at `:873`; and every draw goes through `vp.rng` — a seeded step is bit-reproducible in all three branches and leaves NumPy's global state untouched.

Tests in this slice that mirror the implementation rather than a specification:

- `test_one_dimensional_search_is_bounded` asserts `captured["options"]["maxiter"] == options["search_max_fun_evals"]`. That restates the line of code; the option's documented meaning is a *function-evaluation* budget, and `minimize_scalar`'s `maxiter` is an iteration count. The same test would pass whatever unit the implementation chose. It is also the only test that checks a budget reaches a search at all, which is why F4 (Nelder-Mead getting none) is invisible.
- `test_cmaes_search_starts_from_per_coordinate_step_sizes` asserts `CMA_stds == insigma/insigma.max()` and `sigma0 == insigma.max()` — a transcription of `:539-561`. Its second half, which re-draws a population and checks its per-coordinate spread, is a genuine property check; the first half is not.
- `test_repeated_observation_candidates` and `test_repeated_observation_skips_search_optimizer` assert exact streak values (`== 2`, `== 0`) that restate the bookkeeping at `:715-720`, and use an acquisition written to return `0.0` on exactly the rows `active_sample` appends. They confirm the mechanism is wired as written, not that the right points are repeated.
- `test_candidate_noise_is_one_observation_at_uncertainty_level_one` recomputes `exp(2*h0) + exp(h1)` in the test body and compares with `sn2_new`. The arithmetic is the implementation's; what makes this test worth more than a mirror is the second half, which shows the candidate noise differs from the training noise at a repeated row — a statement about the algorithm.
- `pyvbmc/testing/oracles/_oracles.py: prepare_gp_for_acq` is a line-for-line copy of `active_sample.py:328-363`, including the `S**2 * n_evals` expression and the `hasattr(function_logger, "S")` test. Every oracle that goes through it therefore pins that block against itself; a change of the formula in both places would pass. The `active_sample_step` oracle calls the real function and so is a real trajectory gate, but it is platform-bound and runs only where the fixtures were generated.
- `test_acquiring_a_cached_point_reuses_its_value` and `..._without_a_value_evaluates_it` both place the cache point inside the search box and away from any grid, so the two transformations that F1 rests on are inert; the tests assert `np.allclose` on the recorded coordinates, which is exactly the assertion that would fail, in a state they never construct.
- Nothing in the file exercises `search_optimizer = "Nelder-Mead"`, `acq_hedge = True`, `search_cache_frac > 0` through `active_sample`, `init_design` under `integer_vars`, or an acquired point coinciding with a trimmed row.
