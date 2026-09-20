# P1a — main loop, warmup, termination, final boost, MATLAB-comparison track

Raw reviewer report, wave 2 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent,
fresh context, read-only on the repositories; it compared the Python code
with MATLAB VBMC at `396d649`. Its check scripts ran in the session
scratchpad and are not retained. The text below is the agent's final
message, unedited.

---

# Slice P1a, MATLAB comparison track — main loop, warmup, termination, final boost

PyVBMC `dev-port-review` at `51451dc`; MATLAB VBMC `master` at `396d649`.

(Note: the harness blocked writing `REPORT.md` into the scratch directory — "Subagents should return findings as text, not write report files" — so the complete report is below. The check scripts `chk1.py`–`chk5.py` are in `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\b8a4c56e-a31c-47b4-87d0-66a2c9861ded\scratchpad\port_review\P1a_comparison`.)

## 1. Coverage

### Files read

| MATLAB file (region) | Python counterpart | Both read completely |
| --- | --- | --- |
| `vbmc.m:430-1000` (end of setup, the whole `while ~isFinished_flag` loop, the post-loop outputs) and the local function `savestats` (`vbmc.m:1022-1051`) | `pyvbmc/vbmc/vbmc.py:1114-1941` (`optimize`) | yes |
| `private/vbmc_warmup.m` (all 135 lines) | `pyvbmc/vbmc/vbmc.py:1943-2023` (`_check_warmup_end_conditions`), `:2025-2089` (`_setup_vbmc_after_warmup`) | yes |
| `private/vbmc_termination.m` (all 134 lines, including the local `getStableIter`) | `pyvbmc/vbmc/vbmc.py:2091-2206`, `:2208-2264`, `:2266-2274`, `:2276-2329`, `:2331-2401` | yes |
| `private/recompute_lcbmax.m` | `pyvbmc/vbmc/vbmc.py:2403-2408` (`_recompute_lcb_max`) | yes |
| `misc/finalboost_vbmc.m` | `pyvbmc/vbmc/vbmc.py:2412-2568` (`final_boost`), `:2583-2624` (the three helpers) | yes |
| `misc/best_vbmc.m` | `pyvbmc/vbmc/vbmc.py:2626-2745` (`determine_best_vp`) | yes |
| `private/vbmc_output.m` | `pyvbmc/vbmc/vbmc.py:3119-3180` (`_create_result_dict`) | yes |
| `misc/vptrain2real.m` | the two commented-out calls at `pyvbmc/vbmc/vbmc.py:1349`, `:1541` | yes |
| `private/updateK.m` | call site `pyvbmc/vbmc/vbmc.py:1504-1506`; body `pyvbmc/vbmc/variational_optimization.py:20-88` read only far enough to check the call site and the use of the result | call site only (body is P6's) |
| `misc/warp_input_vbmc.m`, `misc/warp_gpandvp_vbmc.m` | call sites `pyvbmc/vbmc/vbmc.py:1276-1290`; `pyvbmc/whitening/whitening.py:180-252` read to check which `optim_state` fields the warp leaves behind | call sites and state fields only (bodies are P8's) |
| `misc/gptrain_vbmc.m` (signature and `optimState.sn2hpd` only), `private/activesample_vbmc.m` (signature only), `misc/vpoptimize_vbmc.m` (signature only) | `train_gp`, `active_sample`, `optimize_vp` call sites in `optimize` | call sites, arguments and surrounding state only |
| `gpyreg` | `GP.predict` at `pyvbmc/vbmc/vbmc.py:1568-1570`, `_restore_gp_posteriors` in `get_gp` | read as an interface only |

### MATLAB code found unreachable at the defaults, and not reported

- The three `if 0 ... elseif/else` blocks in `private/vbmc_warmup.m:15-34`, `:55-59`, `:71-73`: dead branches kept beside the live ones.
- Every `options.BOWarmup` branch (`vbmc.m:496`, `:711-717`, `:825-828`, `:867-868`; `vbmc_warmup.m:16`, `:54`); default `'no'` (sheet, P1a).
- `options.AdaptiveEntropyAlpha` (`vbmc.m:764-770`), default `'no'`, and it needs `ent/entub_vbmc.m` (sheet, P7).
- The commented-out `vpoptimizeweights_vbmc` branch (`vbmc.m:712-714`) and the commented-out WSABI block (`vbmc.m:676-686`).
- `nargout > 5` (the `samples` struct) and `nargout > 7` (`stats` cleanup) at `vbmc.m:940-966`, and `RetryMaxFunEvals` at `vbmc.m:968-1000` (sheet, P1a).
- `options.VarActiveSample` (`vbmc.m:651-655`), marked `% Unused` in MATLAB itself (sheet, P2).
- `optimState.redoRotoscaling` (`vbmc.m:519`): written once per iteration and read nowhere, on both sides (`pyvbmc/vbmc/vbmc.py:1214`).

### MATLAB history checked

- `private/vbmc_warmup.m:60-68` (the `RecentPast` window): last changed `d6f1188` (2019-09-20).
- `private/vbmc_warmup.m:96-104` (`LastWarping` / `LastSuccessfulWarping` / `LastNonlinearWarping` at the end of warm-up): `8b266a3` (2020-02-07) and `d2989b1` (2021-01-12).
- `private/vbmc_warmup.m:45-50` (the `lcbmax_vec` source) and `private/recompute_lcbmax.m`: both `97ce2ec` (2020-04-11), together with `defopts.RecomputeLCBmax`.
- `private/vbmc_termination.m`: whole file last changed `d2989b1` (2021-01-12); lines 96-102 (the `MinIter` guard) last changed `d9822e3` (2019-09-17).
- `vbmc.m:636-650` (the separate search GP): last changed `d9822e3` (2019-09-17).
- `vbmc.m:779-793` (the running moment average): last changed `c6d723f` (2018-10-03).
- Every commit that touched `vbmc.m:505-960` after 2021-01-19: `48c82e0` (2021-03-27, the `samples` struct), `f3f8b80` (2021-03-13, added `vp.D > 1` to `DoWarping` — present in Python at `vbmc.py:1256`), `f9c04bc` and `a5240d2` (2021-02-02, the warp-undo `elbo_sd` test — present at `vbmc.py:1362-1367`), `2044530` (2021-06-18, comments and the `vbmc_version` argument), `2cb857d` (2021-07-13, `stats.sKL_true(iter)` indexing in `savestats`). Nothing else in the loop moved after the port began.
- Python side: `git log -L` on `vbmc.py` for the warm-up window (`35be58b`, 2021-07-07, first written; `9267816`, 2021-08-25, `max(2,·)` → `max(1,·)`), for the termination guards (`f1e6a0b`, 2021-06-30, first written; `d9bfe16`, 2022-08-28, the snake-case rename that already carried `iteration + 1 >= max_iter` while leaving `iteration < min_iter`).

### Checks run

All under `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/Scripts/python.exe`, which reported
`pyvbmc file: C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\__init__.py`.

1. `chk1.py` — constructed a `VBMC` with `entropy_switch=True`, `D=5`. Output: `optim_state entropy_switch: True`, `has entropy_force_switch key: False`, `get -> None`, `options entropy_force_switch: 0.8`, and the loop's own expression raises `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`. Settles F5.
2. `chk2.py` — pure arithmetic. (a) `len(run_cov == 0) = 3` for a `(3,3)` covariance, so the branch condition at `vbmc.py:1580-1582` is `True` after the first iteration; settles F7. (b) printed the MATLAB and PyVBMC "recent" index sets of the warm-up window for `T = 3`: they agree up to 0-based iteration 3 and part from iteration 4 on (MATLAB `[2,3,4]` against PyVBMC `[1,2,3,4]`, then `[3,4,5]` against `[1,2,3,4,5]`, …); settles F1. (c) `min_iter = 4` blocks 0-based iterations `[0,1,2]` in MATLAB and `[0,1,2,3]` in PyVBMC; settles F4.
3. `chk3.py` — loaded `pyvbmc/testing/vbmc/test_vbmc_save_static.pkl` read-only with `VBMC.load` (7 iterations, `D = 4`, warm-up never ended). Recomputed `impro_fcn` both ways on the recorded `lcb_max`: equal for iterations 1-3, then `0.0066` (PyVBMC) against `0.0047` and `0.0000` (MATLAB) for iterations 4-6. Also printed the recorded `run_mean` per iteration: it swings (index 3 → 4: `+0.0200` → `−0.2535` in the first coordinate) instead of relaxing with `wRun = 0.9^5 = 0.59`, confirming F7 on real recorded state.
4. `chk4.py` — constructed bounded, unbounded and half-bounded `VBMC` instances: `lb_tran`/`ub_tran` are `∓inf` in all three, so `_create_result_dict`'s test yields `"unconstrained"` every time. Settles F8.
5. `chk5.py` — `gpyreg` GPs at `D = 2`: constant mean has 5 hyperparameters, negative-quadratic mean has 9, and the `np.concatenate` of `gaussian_process_train.py:135` / `:147` raises `ValueError ... size 5 and ... size 9`. Settles the consequence of F6.

No test suite, no `optimize()` run, no writes inside any of the three repositories.

### Compared and found equivalent

Stated explicitly, so that the absence of a finding is informative.

- **Loop order.** Iteration counter, timer reset, `optim_state["iter"]`, `vp_old`, the `logging_action` reset, `"start warm-up"`, the entropy-switch test, the warping block, active sampling, `N`/`n_eff`, GP training, the `stop_sampling` latch, `update_K`, the sieve sizes, `optimize_vp`, `vp_K`/`H`, the ELBO read, `sKL`, `lcb_max`, `sKL_true`, the running moments, the per-iteration record, the GP-sampling stop, the termination check, `vp.stats["stable"]`, the warm-up check, `stats.warmup`, the output-warping threshold, the "stopped GP sampling" action, the display line, the plot: same sequence as `vbmc.m:507-881`, with the two documented reorderings (PyVBMC plots after the display line, and runs the GP-sampling stop as a separate call before the termination check rather than inside it).
- **1-based against 0-based arithmetic, verified case by case.** `iter == 1` ↔ `iteration == 0` (`vbmc.m:522`/`:630` ↔ `vbmc.py:1219`/`:1390`); `iter > 1` ↔ `iteration > 0` (`vbmc.m:537`, `:817` ↔ `vbmc.py:1245`, `:1641`); `optimState.Xn > 0` ↔ `function_logger.Xn >= 0` (`vbmc.m:631` ↔ `vbmc.py:1398`); `optimState.N = optimState.Xn` ↔ `Xn + 1` (`vbmc.m:663` ↔ `vbmc.py:1463`); `mod(iter,2) == 0` ↔ `iteration % 2 == 1` (`vbmc.m:640` ↔ `vbmc.py:1415`); `stats.rindex(iter-1)` ↔ `iteration_history["r_index"][iteration - 1]` (`vbmc.m:540` ↔ `vbmc.py:1253`); `stats.gpNsamples(max(1,iter-1))` ↔ `["Ns_gp"][max(0, iteration - 1)]` (`vbmc.m:857` ↔ `vbmc.py:1704`); `iter >= options.MaxIter` ↔ `iteration + 1 >= max_iter` (`vbmc_termination.m:16` ↔ `vbmc.py:2116`); `iter >= TolStableIters` ↔ `iteration + 1 >= tol_stable_iters` (`:70` ↔ `:2148`); `stats.rindex(iter-TolStableIters+1:iter-1)` ↔ `[iteration - tol_stable_iters + 1 : iteration]` (`:75` ↔ `:2154-2156`); `idx0 = max(1,iter-ceil(0.5*TolStableIters)+1)` ↔ `max(0, iter - ceil(0.5*T) + 1)` (`:51` ↔ `:2244-2251`); `optimState.iter < 3` ↔ `iter < 2` (`:113` ↔ `:2214`, `:2285`); `iter > TolStableWarmupIters + 1` ↔ `iteration > tol_stable_warmup_iters` (`vbmc_warmup.m:35` ↔ `vbmc.py:1960`); `max(4,end-T+1):end` and `3:max(3,end-T)` ↔ `[max(3, L-T):]` and `[2 : max(3, L-T)]` (`vbmc_warmup.m:39-40` ↔ `vbmc.py:1966-1971`); `min(NkeepMin,optimState.Xn)` ↔ `min(n_keep_min, Xn + 1)` (`vbmc_warmup.m:124` ↔ `vbmc.py:2075`); `fliplr(1:idx)` and `rank(:,4) = idx` ↔ `arange(1, max_idx+2)[::-1]` and `rank[:,3] = max_idx + 1` (`best_vbmc.m:30`, `:44` ↔ `vbmc.py:2679`, `:2710`); `BackIter = ceil(idx*FracBack); idx_start = max(1,idx-BackIter)` ↔ `max(0, max_idx - ceil((max_idx+1)*frac_back))` (`best_vbmc.m:58-59` ↔ `vbmc.py:2723-2727`); `idx_best = idx_start + idx_best - 1` ↔ `idx_start + argmax` (`:67` ↔ `:2738`).
- **Comparison operators.** `>=` for `funccount >= MaxFunEvals`, `Ns_gp == StableGPSamples`, `stablecount >= TolStableIters - floor(...) - 1`, `rindex < 1`, `ELCBOimpro < TolImprovement`, strict `<` for the weighted `var_ss` against `TolGPVarMCMC`, `>` for `(currentpos - pos) > WarmupNoImproThreshold` and `(iter - LastWarping) > WarpDelay`, `>=` for `(N - lastDataTrim) >= 10` and `(iter - LastSuccessfulWarping) >= TolStableIters/3`: all identical on the two sides.
- **Options evaluated as functions of the state.** `evaloption_vbmc(options.KfunMax, optimState.Neff)` ↔ `options.eval("k_fun_max", {"N": optim_state["n_eff"]})`; `evaloption_vbmc(options.AdaptiveK, Knew)` ↔ `options.eval("adaptive_k", {"K": K_new})`; `NSent`/`NSentFast`/`NSentFine` and their `*Boost` variants at `Knew` in `finalboost_vbmc.m:9-25` ↔ `vbmc.py:2453-2475`; `NSelbo` at `Knew` in `finalboost_vbmc.m:33` ↔ `vbmc.py:2493`. The one that differs in the *main* loop (`NSelbo` at the stale `K`) is the sheet's P6 entry.
- **Warm-up.** The three stopping conditions and their conjunction (`vbmc_warmup.m:87-89` ↔ `vbmc.py:2018-2021`); the ELCBO stability window (`:38-42` ↔ `:1962-1972`); the alternative long-term criterion (`:75-79` ↔ `:2000-2007`); the trim guard (`:82-89` ↔ `:2009-2016`); the false-alarm test and the two thresholds (`:93-113` ↔ `:2032-2064`); the trimming rule including the `NkeepMin` rescue with `-Inf` in place of non-finite values, the `X_flag` conjunction and the "trim data"/"end warm-up" actions (`:115-127` ↔ `:2066-2081`); `SkipActiveSampling` and `RecomputeVarPost` (`:130-133` ↔ `:2084-2089`); the caller's re-update of the GP after trimming (`vbmc.m:819-821` ↔ `vbmc.py:1646-1650`); the post-warm-up resets of `optimize_mu`, `optimize_weights`, `hypstruct.runcov` and `vp_repo` (`vbmc.m:826-833` ↔ `vbmc.py:1651-1663`). PyVBMC's `VariationalPosterior.__init__` leaves `optimize_weights = True` during warm-up where `setupvars_vbmc.m:90-91` sets it `false`, but `optimize_vp` forces it to `False` while `optim_state["warmup"]` holds (`variational_optimization.py:151-152`), and nothing reads the flag before that, so the behavior agrees.
- **Termination.** The reliability index and its three components including the noise-dependent `TolSD` ladder (`vbmc_termination.m:34-40` ↔ `vbmc.py:2219-2241`); the `polyfit` ELCBO improvement (`:51-55` ↔ `:2254-2263`); the inf/NaN values before the third iteration (`:57-59` ↔ `:2214-2217`); the stability count, the exception fraction and the distance-from-warping guard (`:70-92` ↔ `:2147-2188`); the recording of `rindex`, `elcbo_impro`, `stable` and `optimState.R` (`:62-66`, `:95` ↔ `:2139-2143`, `:2191`); the entropy-switch `TolStableIters` (`:22-26` ↔ `:2124-2132`); the GP-sampling stop, its `idx_stable = 1` weighting `0.5·onehot + 0.5·normalize(exp(-(N_end - N)/10))` and its strict threshold (`:43-48`, `:123-129` ↔ `:2317-2329`), which the sheet already records as settled.
- **Final boost.** `Knew = max(MinFinalComponents, vp.K)`, the six entropy-sample counts, the `do_boost` test, `Nfastopts = ceil(ceil(NSelbo(Knew)) * NSelboIncr)`, `Nslowopts = 1`, `TolWeight = 0`, `optimState.Warmup = false`, `vp.optimize_mu/optimize_weights` from the options, the three `NSent*` overrides, `MaxIterStochastic = Inf`, `entropy_alpha = 0`, the preservation of `vp.stats.stable` across the optimization, and the GP: `finalboost_vbmc.m:6-57` ↔ `vbmc.py:2450-2528`, with `gplite_post(stats.gp(idx_best))` ↔ `self.get_gp(idx_best)` (`vbmc.py:1835`, `:2747-2793`), which recomputes exactly the factors the lean record drops. Only the acceptance guard and `weight_penalty = 0` differ, both on the sheet.
- **Best posterior.** `best_vbmc.m` against `determine_best_vp` in both branches, including the four rank columns, the descending/ascending sort directions, the last-stable search and the final `vp.stats.stable` assignment; and both call sites, `vbmc.m:547` (with `iter-1`) ↔ `vbmc.py:1261-1265` (with `max_idx = iteration_history["iter"][-1]`, which is the previous iteration because the current one is not yet recorded) and `vbmc.m:888` ↔ `vbmc.py:1826-1830`.
- **`vptrain2real`.** `misc/vptrain2real.m` is the identity unless `vp.temperature ∈ {2,3,4,5}`. PyVBMC's `VariationalPosterior` has no `temperature` attribute at all, and `optim_state["temperature"]` is never written (the only reads are `whitening.py:196`, `:291`), so no PyVBMC setting can reach the non-identity branch. Leaving the two calls out (`vbmc.py:1349`, `:1541`) is harmless at every setting PyVBMC can reach; the same holds for the third, post-loop call `vbmc.m:904` (`entflag = 1`), which has no Python counterpart at all.
- **`update_K` call site.** `Knew = updateK(optimState,stats,options)` ↔ `update_K(self.optim_state, self.iteration_history, self.options)`, in the same branch, with the result passed to the variational optimization in the same position (`vbmc.m:692`, `:714` ↔ `vbmc.py:1504-1506`, `:1526-1534`). Both sides call it before the current iteration is recorded, so `stats.warmup(end)`/`stats.pruned(end)` and their Python equivalents are the previous iteration's on both sides.
- **Per-iteration recording.** `savestats` is called before `vbmc_termination` and before the warm-up check, and `stats.warmup(iter)` is written again after the warm-up check; PyVBMC's `record_iteration` (`vbmc.py:1622`) and the separate `record("warmup", ...)` (`:1671`) sit at the same two points. Of `savestats`'s 22 fields, PyVBMC omits `cachecount`, `vpK`, `gpNoise_hpd`, `outwarp_threshold` and `t`; a grep of the MATLAB tree shows none of those has a reader, so the omission changes nothing. Every field that *is* read somewhere (`N`, `Neff`, `funccount`, `elbo`, `elbo_sd`, `sKL`, `pruned`, `rindex`, `stable`, `warmup`, `gpSampleVar`, `lcbmax`, `vp`, `gp`) is recorded.
- **Result dict.** `function`, `func_count`, `best_iter`, `train_set_size`, `components`, `r_index`, `convergence_status`, `algorithm`, `version`, `message`, `elbo`, `elbo_sd` all agree with `vbmc_output.m:4-28`; `overhead`, `rng_state` and the missing `samples` struct are on the sheet. The `exitflag` logic of `vbmc.m:916-923` reduces to `exitflag = vp.stats.stable`, which is what `vbmc.py:1915-1918` computes as `success_flag`. The three termination messages match word for word up to the option spelling.
- **`_compute_true_diagnostic`.** `vbmc_moments(vp_real,1,1e6)` then `mvnkl` then `0.5*sum(kl)` ↔ `diagnostic_vp.moments(1e6, True, True)` then `kl_div_mvn` then `0.5*np.sum(kl)` (`vbmc.m:772-777` ↔ `vbmc.py:3087-3091`). PyVBMC additionally validates the shapes and works on a copy so the run's generator is not advanced; MATLAB's million draws come from the global stream and do move it.
- **Warping branch.** The `DoWarping` conjunction including `IncrementalWarpDelay`, `WarpMinK`, `WarpTolReliability` and `vp.D > 1`; the saved-and-restored state; the undo test on both ELBO and ELBO SD; `WarpingCount += 2` and `LastWarping = iter` on the undo; the `warp_action` strings (`vbmc.m:528-623` ↔ `vbmc.py:1233-1381`). The `gp` argument that `warp_input_vbmc.m:1` takes and PyVBMC's `warp_input` does not is used only inside the dead `GPsample_flag = false` branch (`warp_input_vbmc.m:20-21`), so the different signature is not a difference in behavior.

---

## 2. Findings

### F1. The warm-up "recent improvement" window starts two iterations too early
- Location: `pyvbmc/vbmc/vbmc.py:1979-1993`; MATLAB: `private/vbmc_warmup.m:60-65`
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): no. `private/vbmc_warmup.m:62-63` was written in `d6f1188` (2019-09-20) and has not changed since, so it predates the port. On the Python side the expression was written as it stands in `35be58b` (2021-07-07, "feat: handle end of vbmc warmup"); `9267816` (2021-08-25) only changed the floor from `max(2, ·)` to `max(1, ·)`. The two have never agreed.
- What the code does, what it should do, and why. MATLAB marks the last `T = ceil(TolStableWarmup/FunEvalsPerIter)` entries of `lcbmax_vec` as "recent": `RecentPast = iter - T + 1` (1-based), floored at index 2. Translating to PyVBMC's 0-based `iteration = iter - 1`, the window must start at `max(1, iteration + 1 - T)`. `vbmc.py:1981-1987` instead computes `recent_past = iteration - (ceil(tol_stable_warmup/fun_evals_per_iter) + 1)`, that is `iteration - T - 1`: the `+ 1` of MATLAB's expression was moved inside the parentheses (so its sign flipped) and the 1-based-to-0-based shift was not applied. The window therefore begins two entries earlier than it should. At the shipped defaults (`tol_stable_warmup = 15`, `fun_evals_per_iter = 5`, so `T = 3`) the "recent" set holds `T + 2 = 5` iterations instead of 3, and the complementary "before" set is two iterations smaller. Because `impro_fcn = max(0, max(recent) − max(before))` and the Python "recent" set is a superset of MATLAB's while its "before" set is a subset, PyVBMC's `impro_fcn` is always greater than or equal to MATLAB's, so `no_recent_improvement_flag = impro_fcn < stop_warmup_thresh` is satisfied strictly less often.
- Consequence if real: fires at the default options, in every run, from the fifth iteration on (check 2b: the two windows agree through 0-based iteration 3 and part from iteration 4). PyVBMC's first warm-up stopping criterion (`stable_count_flag and no_recent_improvement_flag`) is harder to satisfy than MATLAB's, so warm-up tends to persist longer; the second, long-term criterion is unaffected, so the effect is bounded by `WarmupNoImproThreshold`. Warm-up end governs the training-set trim, the switch to variable means and weights, the growth of `K` and the reset of the GP hyperparameter covariance, so a late end shifts the whole remainder of the run. On the stored 7-iteration run, `impro_fcn` is `0.0066` where MATLAB would compute `0.0047` and `0.0000` (check 3); both are far below the threshold `1.0` there, so that particular short run would not have changed, but the quantity genuinely differs.
- Suggested reproduction: already run (check 2b and check 3, outputs above). The smallest self-contained check is the index arithmetic in `chk2.py`: for `T = 3` and 0-based iteration 5, MATLAB's recent set is `[3,4,5]` and PyVBMC's is `[1,2,3,4,5]`.
- Test adequacy: no. `pyvbmc/testing/vbmc/test_vbmc_loop_termination.py:396-468` sets `iteration_history["lcb_max"] = np.ones(101)`, a constant vector, so `impro_fcn` is `0` under any window and the index arithmetic is unobservable. The tests exercise the flags, not the window.

### F2. `_recompute_lcb_max` is an unimplemented stub, and MATLAB's `RecomputeLCBmax` is on by default
- Location: `pyvbmc/vbmc/vbmc.py:2403-2408`, used at `:1642-1645` and (not) at `:1975`; MATLAB: `private/recompute_lcbmax.m:1-23`, `vbmc.m:815-817`, `private/vbmc_warmup.m:46-50`
- Category: control flow
- Proposed classification: port discrepancy (an unported feature that is active at the MATLAB defaults)
- Confidence: high
- History (comparison track): no. `private/recompute_lcbmax.m` and `defopts.RecomputeLCBmax = 'yes'` both entered in `97ce2ec` (2020-04-11) and neither has changed since, so both predate the port.
- What the code does, what it should do, and why. During warm-up MATLAB recomputes the whole history of LCB maxima with the *current* GP: `recompute_lcbmax.m:16-21` predicts on all active training inputs, forms `lcb = fmu - ELCBOImproWeight*sqrt(fs2)`, takes the trailing cumulative maximum (`movmax(lcb,[numel(lcb),0])`) and samples it at each iteration's training-set size `stats.N`. `vbmc.m:815-817` stores the result in `optimState.lcbmax_vec`, and `vbmc_warmup.m:46-50` uses it *in preference to* `stats.lcbmax` whenever it is non-empty. PyVBMC's `_recompute_lcb_max` returns `np.array([])` with the comment "ToDo: Recompute_lcb_max needs to be implemented", `vbmc.py:1643-1645` stores that empty array in `optim_state["lcb_max_vec"]`, and `_check_warmup_end_conditions` never consults it: `vbmc.py:1975` always reads `iteration_history["lcb_max"]`, the per-iteration values each computed with that iteration's own GP. So PyVBMC always takes MATLAB's `RecomputeLCBmax = 'no'` path while MATLAB's default is `'yes'`.
- Consequence if real: fires at the default options on every run that reaches the warm-up end check (`iteration > 0` and still warming up). The recomputed vector feeds both warm-up criteria: `impro_fcn` (`vbmc_warmup.m:61-64`) and the long-term `max_thresh` / `idx_1st` / `pos` criterion (`:75-79`). The two vectors differ whenever the GP has changed since an early iteration — which is the normal case, because the early LCB maxima were computed from a GP fitted to a handful of points. The historical `lcbmax` values are also not monotone (see the stored run in check 3, where `lcb_max` dips at iterations 3 and 5), while MATLAB's recomputed vector is a cumulative maximum and therefore non-decreasing; a non-monotone vector can make `impro_fcn` positive where MATLAB's would be exactly zero. The size of the effect on the warm-up end point is run-dependent and cannot be bounded from here.
- Suggested reproduction: comparing the two criteria on a real run needs the MATLAB toolbox (there is no Python implementation to run against). What can be settled here, and was, is that the Python function is a stub returning an empty array and that nothing reads its result (`grep lcb_max_vec pyvbmc/`: written at `vbmc.py:1644`, read nowhere).
- Test adequacy: no. No test under `pyvbmc/testing/` names `_recompute_lcb_max` or `lcb_max_vec`; the warm-up tests supply `iteration_history["lcb_max"]` directly.

### F3. Warm-up end does not reset the warping clocks, so PyVBMC can warp and can terminate earlier than MATLAB
- Location: `pyvbmc/vbmc/vbmc.py:2032-2042` (the end-warm-up branch, which sets only `last_warmup`), affecting `:1248-1250` and `:2176-2180`; MATLAB: `private/vbmc_warmup.m:97-102`
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): no. `optimState.LastWarping` and `optimState.LastNonlinearWarping` were set there from `8b266a3` (2020-02-07) and `optimState.LastSuccessfulWarping` was added in `d2989b1` (2021-01-12), a week before PyVBMC's first commit and six months before `35be58b` (2021-07-07) ported the warm-up module. The Python has never had the lines.
- What the code does, what it should do, and why. When warm-up really ends (not on a false-alarm trim), MATLAB starts the warping clocks at the current iteration:
  ```
  optimState.LastWarmup = optimState.iter;
  % Start warping
  optimState.LastWarping = optimState.iter;
  optimState.LastSuccessfulWarping = optimState.iter;
  optimState.LastNonlinearWarping = optimState.iter;
  ```
  PyVBMC's `_setup_vbmc_after_warmup` sets `last_warmup` only. Both implementations initialize `LastWarping` and `LastSuccessfulWarping` to `-Inf` (`misc/setupvars_vbmc.m:156`, `:159`; `vbmc.py:933`, `:937`) and both update them inside the warp itself (`misc/warp_input_vbmc.m:161-162`; `pyvbmc/whitening/whitening.py:240-241`), so the only missing update is this one. Two live reads see the stale `-inf`:
  - `vbmc.py:1248-1250`, `iteration - last_warping > WarpDelay`, is `inf > WarpDelay`, hence always true. MATLAB requires `iter - LastWarping > WarpEveryIters * max(1, WarpingCount) = 5` after warm-up, so it cannot warp for five more iterations.
  - `vbmc.py:2176-2180`, `iteration - last_successful_warping >= tol_stable_iters / 3`, is `inf >= 4`, hence always true. MATLAB blocks termination on stability for `TolStableIters/3 = 4` iterations after warm-up ends.
- Consequence if real: fires at the default options (`warp_rotoscaling = True`, `D > 1`). PyVBMC's first rotoscaling warp can happen as soon as `vp.K` reaches `warp_min_k = 5` and the reliability index is below 3 — in practice about two iterations after warm-up, where MATLAB needs at least six. A warp rewrites the training coordinates, the search bounds and the GP hyperparameters, so an earlier first warp changes the rest of the run. Independently, a run whose reliability history was already good during warm-up can be declared converged in PyVBMC in the first post-warm-up iteration, four iterations before MATLAB would allow it.
- Suggested reproduction: construct a `VBMC`, set `optim_state["warmup"] = True`, `optim_state["iter"] = k`, populate `iteration_history["r_index"][k]` below `stop_warmup_reliability`, call `_setup_vbmc_after_warmup()` and print `optim_state["last_warping"]` and `optim_state["last_successful_warping"]`: both remain `-inf` where MATLAB would hold `k`. Not run (the code path is a three-line read of the source and the greps above are conclusive: `last_successful_warping` is written only at `vbmc.py:937` and `whitening.py:241`).
- Test adequacy: no. `test_vbmc_loop_termination.py:470-575` checks the point-keeping and the `data_trim_list` bookkeeping of `_setup_vbmc_after_warmup`, never the warping clocks, and `test_vbmc_check_termination_conditions_stability` leaves `last_successful_warping` at its initial value so the guard is vacuously true there too.

### F4. The minimum-iteration guard compares a 0-based index with a 1-based count
- Location: `pyvbmc/vbmc/vbmc.py:2194-2201`; MATLAB: `private/vbmc_termination.m:98-101`
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): no. `vbmc_termination.m:98-101` last changed in `d9822e3` (2019-09-17). The Python line was written as `iteration < self.options.get("miniter")` in `f1e6a0b` (2021-06-30) and has only been renamed since. The neighbouring maximum-iteration test in the same function *was* corrected to `iteration + 1 >= max_iter` (present already before the snake-case rename `d9bfe16`, 2022-08-28), which shows the two were not meant to be treated differently.
- What the code does, what it should do, and why. MATLAB blocks termination while `optimState.iter < options.MinIter`, with `iter` 1-based, i.e. while fewer than `MinIter` iterations have been completed. PyVBMC writes `iteration < self.options.get("min_iter")` with `iteration` 0-based; the faithful translation is `iteration + 1 < min_iter`. PyVBMC therefore also blocks the iteration whose 0-based index equals `min_iter - 1`, that is MATLAB's `iter == MinIter`.
- Consequence if real: fires at the default options (`min_iter = D` on both sides, `vbmc.m:271` against `advanced_vbmc_options.ini:155`). A run that satisfies a termination condition exactly at the `MinIter`-th iteration performs one extra iteration, costing `fun_evals_per_iter = 5` further target evaluations and one more variational optimization. Check 2c: with `min_iter = 4`, MATLAB blocks 0-based iterations 0-2 and PyVBMC blocks 0-3. The effect is confined to short runs and to runs with an unusually large `min_iter`; at `D >= 5` the stability criterion normally cannot be met before `ceil(60/5) = 12` iterations anyway, so `min_iter = D` rarely binds at the default. It always binds when a user raises `min_iter`.
- Suggested reproduction: already run (check 2c).
- Test adequacy: no — worse, the existing test encodes the discrepancy. `pyvbmc/testing/vbmc/test_vbmc_loop_termination.py:78-92` sets `min_iter = 101` and `optim_state["iter"] = 100` and asserts `not terminated`. Under MATLAB's rule that state is `iter == 101 >= MinIter == 101`, which does *not* block. The test passes only because of the off-by-one.

### F5. `optim_state["entropy_force_switch"]` is never set, so enabling `entropy_switch` raises `TypeError`
- Location: `pyvbmc/vbmc/vbmc.py:1224-1230` (the read) against `:988-992` (where `optim_state` is populated); MATLAB: `vbmc.m:524-528`
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): no. `vbmc.m:524-528` reads `options.EntropyForceSwitch * options.MaxFunEvals` and has not changed since well before the port (the block predates `2044530`, 2021-06-18, which touched only comments here).
- What the code does, what it should do, and why. MATLAB takes both factors from `options`. PyVBMC takes `entropy_force_switch` from `optim_state`, which `_init_optim_state` never writes — it copies `entropy_switch` (`:988`), `max_fun_evals` (`:1001`) and `entropy_alpha` (`:1035`) but not `entropy_force_switch`. `optim_state.get("entropy_force_switch")` returns `None` and `None * max_fun_evals` raises. The `and` short-circuits on `entropy_switch`, which is `False` by default, so the expression is never evaluated in a default run; the moment a user sets `entropy_switch=True` with `D >= det_entropy_min_d = 5`, the first iteration of `optimize()` aborts. The reliable read is `self.options.get("entropy_force_switch")`, which returns `0.8`.
- Consequence if real: does not fire at the default options (`entropy_switch = False` on both sides, `vbmc.m:303`). With the option enabled the run cannot start at all, so the whole deterministic-to-stochastic entropy switch — one of MATLAB's documented switches, and the only user of `tol_stable_entropy_iters` and of the entropy-switch branches at `vbmc.py:2124` and `:2169` — is unreachable from Python.
- Suggested reproduction: already run (check 1, `chk1.py`): `has entropy_force_switch key: False`, `get -> None`, and the loop's own expression raises `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`.
- Test adequacy: no. `test_vbmc_is_finished_stability_entropy_switch` and `test_vbmc_check_termination_conditions_prevent_early_termination` set `optim_state["entropy_switch"] = True` by hand and call `_check_termination_conditions` directly, which never touches the missing key. No test constructs a `VBMC` with `entropy_switch=True` and enters the loop.

### F6. The separate search GP trains from, and overwrites, the main hyperparameter struct and `sn2_hpd`
- Location: `pyvbmc/vbmc/vbmc.py:1415-1431`; MATLAB: `vbmc.m:638-648` with `vbmc.m:471`
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high (that the code differs); medium (that the consequence is the crash described, which is inferred from the shapes rather than from a run)
- History (comparison track): no. `vbmc.m:636-650` last changed in `d9822e3` (2019-09-17).
- What the code does, what it should do, and why. MATLAB keeps a second hyperparameter struct for the search GP, declared beside the main one at `vbmc.m:471` (`gp = []; hypstruct = []; hypstruct_search = [];`) and used exclusively at `:643`:
  `[gp_search,hypstruct_search] = gptrain_vbmc(hypstruct_search,optimState,stats,options);`
  Capturing only two outputs also discards the `optimState` that `gptrain_vbmc` returns, so the `optimState.sn2hpd` written at `misc/gptrain_vbmc.m:103` by the constant-mean fit never reaches the run. PyVBMC passes and reassigns the *main* `self.hyp_dict` (`vbmc.py:1419`) and copies the constant-mean fit's noise estimate into the run (`:1430`). After that line, `hyp_dict["hyp"]`, `hyp_dict["full"]`, `hyp_dict["logp"]` and `hyp_dict["run_cov"]` all describe a GP with a constant mean, which has `D+3` hyperparameters instead of the negative-quadratic model's `3D+3` (check 5: 5 against 9 at `D = 2`).
  Two things follow. (a) `optim_state["sn2_hpd"]` is the constant-mean estimate while `active_sample` runs, and `pyvbmc/vbmc/active_sample.py:744` reads it. (b) The next main `train_gp` builds its starting points as `hyp0 = np.empty((0, hyp_dict["hyp"].T.shape[0]))` (`gaussian_process_train.py:135`) and then concatenates the negative-quadratic hyperparameters of the recorded GPs onto it (`:143-149`); the widths disagree and `np.concatenate` raises `ValueError` (check 5). The guard that exists for exactly this situation, `if hyp0.shape[1] != np.size(gp.hyper_priors["mu"]): hyp0 = None` (`:159-160`), sits after the failing concatenation.
- Consequence if real: does not fire at the default options (`separate_search_gp = False` on both sides, `vbmc.m:319`). With the option enabled, the run reaches its second iteration, trains the constant-mean search GP, and then fails in `train_gp` with a shape `ValueError`; if the history were empty or `init_N` were zero it would instead silently discard the hyperparameter warm start and carry a covariance of the wrong model into the sampler widths. Either way the option, which MATLAB supports, is unusable.
- Suggested reproduction: already run for the shape mismatch (check 5). A full reproduction would need one `optimize()` run with `separate_search_gp=True`, which is outside the rules here.
- Test adequacy: no. `grep separate_search_gp pyvbmc/testing/` returns nothing.

### F7. The running average of the variational moments never runs (misplaced parenthesis)
- Location: `pyvbmc/vbmc/vbmc.py:1580-1597`; MATLAB: `vbmc.m:780-793`
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): no. `vbmc.m:779-793` last changed in `c6d723f` (2018-10-03).
- What the code does, what it should do, and why. MATLAB initializes on the first iteration and then keeps an exponentially weighted running average:
  `if isempty(optimState.RunMean) || isempty(optimState.RunCov)`.
  PyVBMC writes
  ```
  if len(self.optim_state.get("run_mean")) == 0 or len(
      self.optim_state.get("run_cov") == 0
  ):
  ```
  The `== 0` of the second test is inside the `len(...)`, so instead of `len(run_cov) == 0` it evaluates `len(run_cov == 0)`, the number of rows of the elementwise comparison array, which is `D` and therefore truthy for every `D >= 1`. From the second iteration on, `len(run_mean) == 0` is `False` and the second operand is `D`, so the `or` is truthy and the initialization branch is taken every time. `run_mean` and `run_cov` are overwritten with the current iteration's moments and the `else` branch at `:1586-1597` — the only reader of `moments_run_weight` — never executes.
- Consequence if real: fires at the default options, on every iteration after the first. Nothing in PyVBMC or in MATLAB reads `RunMean`/`RunCov`/`LastRunAvg` (verified by grep over both trees), so no algorithmic quantity changes; what changes is the `optim_state` recorded in the iteration history and exposed to users and to any future reader of that state, which holds the raw per-iteration moments instead of the smoothed ones, and `moments_run_weight` (`advanced_vbmc_options.ini`, MATLAB `defopts.MomentsRunWeight = 0.9`) is effectively dead without being registered in `INERT_OPTIONS`.
- Suggested reproduction: already run. Check 2a: `len(np.eye(3) == 0)` is `3`, so the condition is `True`. Check 3 on the stored run: the recorded `run_mean` swings between iterations (first coordinate `+0.0200` at iteration 3, `−0.2535` at iteration 4) instead of moving by `1 − 0.9^5 = 0.41` of the gap, as the running average would.
- Test adequacy: no. No test reads `optim_state["run_mean"]` or `["run_cov"]`.

### F8. `results["problem_type"]` can never report a bounded problem
- Location: `pyvbmc/vbmc/vbmc.py:3127-3132`; MATLAB: `private/vbmc_output.m:5-9`
- Category: state/caching
- Proposed classification: suspected defect in both
- Confidence: high
- History (comparison track): no. `private/vbmc_output.m` at `396d649` has held these lines since the file was created; `misc/setupvars_vbmc.m:49-50` (which makes `optimState.LB` the *transformed* bound) predates the port.
- What the code does, what it should do, and why. Both implementations test whether all lower and upper bounds are infinite, and both test the *transformed* bounds: MATLAB reads `optimState.LB`/`optimState.UB`, which `misc/setupvars_vbmc.m:49-50` sets to `warpvars_vbmc(LB,'dir',trinfo)`; PyVBMC reads `optim_state["lb_tran"]`/`["ub_tran"]`, set at `vbmc.py:910-915`. For a bounded variable the transform maps the bound to `±inf` on both sides (MATLAB's logit branch `shared/warpvars_vbmc.m:120`, `y = log(z./(1-z))` with `z = 0` at the lower bound; PyVBMC's probit/logit likewise), and for an unbounded variable the identity leaves `±inf`. The test is therefore always true and the branch always yields "unconstrained". The field is meant to distinguish the two problem types and should read the original bounds (`optimState.LB_orig`, `self.lower_bounds`).
- Consequence if real: fires at the default options, but only on a reported field: `results["problem_type"]` is a diagnostic that no algorithmic decision reads on either side. A user with a bounded problem is told "unconstrained".
- Suggested reproduction: already run (check 4, `chk4.py`): a fully bounded, a half-bounded and an unbounded `D = 2` problem all give `lb_tran = [[-inf -inf]]`, `ub_tran = [[inf inf]]` and hence `"unconstrained"`.
- Test adequacy: no. `test_vbmc_optimize.py:506` checks only that the key `"iterations"` is present in `results`; nothing asserts `problem_type`.

### F9. `results["iterations"]` is the index of the last iteration, not the number of iterations
- Location: `pyvbmc/vbmc/vbmc.py:3134`; MATLAB: `private/vbmc_output.m:10`
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): no. `vbmc_output.m:10` is unchanged at `396d649` from before the port.
- What the code does, what it should do, and why. MATLAB's `output.iterations = optimState.iter` is the 1-based iteration counter, which is also the number of iterations performed. PyVBMC's `optim_state["iter"]` is the 0-based index of the last iteration (`vbmc.py:926`, `:1213`), so `results["iterations"]` is one less than the number of iterations. The neighbouring `best_iter` is deliberately an index on the Python side (it indexes `iteration_history`), but `iterations` is a count in both toolboxes' vocabulary, and the same quantity appears as a count in the log's first column (`vbmc.py:1732` prints `self.iteration`, also 0-based, against MATLAB's 1-based `iter`, which is a separate cosmetic offset in the trace).
- Consequence if real: fires at the default options, on a reported field only; no algorithmic decision reads it.
- Suggested reproduction: read `results["iterations"]` after any run and compare with `len(vbmc.iteration_history["iter"])`. Not run (no `optimize()` runs allowed); the two lines are conclusive on their own.
- Test adequacy: no. `test_vbmc_optimize.py:506` asserts only that the key exists.

### F10. The warp branch sizes the sieve at `vp.K` rather than at `Knew`, contradicting the sheet's P6 entry
- Location: `pyvbmc/vbmc/vbmc.py:1317-1328`; MATLAB: `vbmc.m:572-584`
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): no. `vbmc.m:575-584` predates the port (`d9822e3`, 2019-09-17, for the `optimize_mu` branch; the warp-undo block around it is `d2989b1`, 2021-01-12).
- What the code does, what it should do, and why. In the warp-undo check MATLAB sets `Knew` in both arms of the `~vp.optimize_mu` test and then sizes the fast sieve at `Knew`: `Nfastopts = ceil(evaloption_vbmc(options.NSelbo,Knew))` (`vbmc.m:584`). PyVBMC computes the same `Knew` at `vbmc.py:1320`/`:1323` but then sizes the sieve at `self.vp.K` (`:1326-1328`). When `vp.optimize_mu` is false, `Knew = self.vp.mu.shape[1]` is the number of training inputs while `self.vp.K` is still the previous component count, so the two disagree. The known-differences sheet's entry "The sieve asks for candidates in proportion to the current `K`" states that "`vbmc.m:584` (the warp branch) and `misc/finalboost_vbmc.m:33` use `Knew`" and that "PyVBMC's warp branch and final boost agree with MATLAB's". The final boost does agree (`vbmc.py:2493` evaluates `ns_elbo` at `K_new`); the warp branch does not, in the `optimize_mu = False` case.
- Consequence if real: does not fire at the default options (`variable_means = True` on both sides, so `vp.optimize_mu` is true after warm-up and `Knew == vp.K`). With `variable_means=False` the warp-undo check runs its sieve with a candidate count derived from the stale component count, so the undo decision rests on a differently sized search than MATLAB's. The same `vp.K`-instead-of-`Knew` pattern is in the main loop at `vbmc.py:1500-1511`, but there MATLAB itself uses a third value (the stale `K = Kwarmup`), which is the sheet's existing entry.
- Suggested reproduction: construct a `VBMC` with `variable_means=False`, set `vp.optimize_mu = False`, and compare `math.ceil(options.eval("ns_elbo", {"K": vp.K}))` with `math.ceil(options.eval("ns_elbo", {"K": gp.X.shape[0]}))`. Not run; the two lines are conclusive.
- Test adequacy: no. No test exercises the warp-undo branch with `variable_means=False`.

### Minor observations

- **The final "finalize" line is printed under a narrower condition than MATLAB's, and its `sKL` compares against a different posterior.** MATLAB sets `new_final_vp_flag = idx_best ~= iter` (`vbmc.m:889`), raises it further if the boost changed anything (`:895`), and prints the line whenever it is set; PyVBMC prints only when `changed_flag` is true (`vbmc.py:1839-1902`), so a run whose returned posterior comes from an earlier iteration but whose boost was a no-op prints nothing. Separately, MATLAB recomputes `sKL` against `vp_old = vp` captured *after* the loop (`vbmc.m:884`), i.e. the last iteration's final posterior; PyVBMC compares against the `vp_old` captured at the *start* of the last iteration (`vbmc.py:1215`, `:1846`), so the printed number mixes the boost's effect with the last iteration's update. Display only; nothing records `sKL` after the loop.
- **`vp.gp` is not attached to the returned posterior.** `vbmc.m:890-891` sets `gp = stats.gp(idx_best); vp.gp = gp;`. PyVBMC has no equivalent; its only MATLAB consumer is `vbmc_rnd`'s `balanceflag == 'gp'` branch, which the sheet records as unported.
- **`self.vp` aliases the iteration history when `do_final_boost` is false.** `determine_best_vp` returns `self.iteration_history.get("vp")[idx_best]` itself, not a copy (`vbmc.py:2741`), and assigns `vp.stats["stable"]` into it (`:2744`). With the default `do_final_boost = True` the subsequent `final_boost` deep-copies (`:2446`), so the alias survives only when the boost is disabled. MATLAB's struct assignment copies by value.
- **`final_boost` permanently mutates the instance's `optim_state`.** `vbmc.py:2509` and `:2517` set `warmup = False` and `entropy_alpha = 0` on `self.optim_state`; MATLAB's `finalboost_vbmc` receives `optimState` by value and its changes die with the call (`finalboost_vbmc.m:40`, `:48`). Inside `optimize()` this is harmless (a resume reloads `optim_state` from the history), but `final_boost` is a public method.
- **`self.vp.mu = self.gp.X.T` in the warp branch has no `.copy()`.** `vbmc.py:1319` binds a transposed view of `gp.X`, where the main loop writes `self.gp.X.T.copy()` (`:1500`). Only reachable with `variable_means=False`.
- **The warp re-warps only the active rows.** `pyvbmc/whitening/whitening.py:202-207` selects `function_logger.X_flag`, where `misc/warp_input_vbmc.m:112-119` re-warps rows `1:optimState.Xn` regardless of `X_flag`, so rows deactivated by a warm-up trim keep stale transformed coordinates in PyVBMC. Nothing in my slice reads them; flagged for the P8 reviewer.
- **The entropy-switch stability branch drops MATLAB's finiteness test.** `private/vbmc_termination.m:80` reads `if optimState.EntropySwitch && isfinite(options.EntropyForceSwitch)`; `vbmc.py:2169` tests only `entropy_switch`. With `EntropyForceSwitch = Inf` MATLAB would allow termination while PyVBMC would instead turn the switch off and continue. Unreachable today because of F5.
- **String and formatting divergences in the result dict and the trace.** `problem_type` uses `"bounded"` where MATLAB uses `'boundconstraints'`; `best_iter` is 0-based where MATLAB's `bestiter` is 1-based (consistent with PyVBMC's own indexing of `iteration_history`); PyVBMC adds a `cache_active` branch to the final "finalize" line (`vbmc.py:1859-1872`) that MATLAB does not have (`vbmc.m:908-913`); and MATLAB's per-iteration `stats.timer(iter).totalruntime = toc(t0)` (`vbmc.m:880`) has no Python counterpart.

---

## 3. Test adequacy notes

Tests that mirror the implementation rather than the specification:

- `pyvbmc/testing/vbmc/test_vbmc_loop_termination.py:78-113`, `test_vbmc_check_termination_conditions_prevent_early_termination`. The first half sets `min_iter = 101` with `optim_state["iter"] = 100` and asserts the run does not terminate. Under `private/vbmc_termination.m:98-101` that state is `iter (= 101) >= MinIter (= 101)`, which does not block; the assertion holds only because of F4. The test freezes the off-by-one rather than testing the rule.
- `test_check_warmup_end_conditions_*` (`:396-468`). All four set `iteration_history["lcb_max"] = np.ones(101)`, `elbo = np.ones(101)` and `elbo_sd = np.ones(101)*1e-4`, i.e. constant vectors. Every window-dependent quantity (`impro_fcn`, `max_now - max_before`, `max_thresh`, `idx_1st`) is then independent of the index arithmetic, which is where the port errors are (F1). The tests confirm the boolean wiring of the four flags and nothing else.
- `test_vbmc_is_finished_stability_entropy_switch` (`:166-189`) sets `optim_state["entropy_switch"] = True` directly and calls `_check_termination_conditions`, so it never reaches the loop line that F5 breaks, and it cannot notice the missing `isfinite(EntropyForceSwitch)` test either.

Tests that are good models:

- `test_is_gp_sampling_finished` and `test_gp_sampling_history_compatibility_and_stable_weights` (`:245-356`) parameterize `N` and `var_ss` over several histories and assert the *weights* `0.5·onehot + 0.5·normalize(exp(-(N_end-N)/10))` that `private/vbmc_termination.m:123-129` specifies, not the code that computes them. That is the shape a warm-up window test should have: vary `lcb_max` so that the window membership is observable, and assert the MATLAB index set.
- `test_vbmc_compute_reliability_index` (`:198-244`) builds a history with distinct values per iteration and checks the three components of the index and the `polyfit` slope against independently computed numbers.
- `pyvbmc/testing/oracles/` pins the numerics stage by stage from stored states, which is the right instrument for anything a unit test would have to mirror; nothing in it covers the loop-level bookkeeping of this slice.

---

## 4. Sheet notes

- **The counterpart map is wrong about `private/recompute_lcbmax.m`.** `counterpart_map.md:91` lists it as `ported` with `pyvbmc/vbmc/vbmc.py:2403: _recompute_lcb_max`. That function returns `np.array([])` and carries the comment "ToDo: Recompute_lcb_max needs to be implemented"; nothing reads its result. The status should be `unported`, and the difference belongs on the sheet, because `defopts.RecomputeLCBmax = 'yes'` (`vbmc.m:343`) makes the MATLAB path the default one (F2).
- **The P6 entry "The sieve asks for candidates in proportion to the current `K`" overstates the agreement.** It asserts that "PyVBMC's warp branch and final boost agree with MATLAB's". The final boost does; the warp branch evaluates `ns_elbo` at `self.vp.K` where `vbmc.m:584` uses `Knew`, which differ when `vp.optimize_mu` is false (F10).
- **The P7 entry on posterior tempering is incomplete about what `temperature` reaches.** It says the option "is read only by `pyvbmc/whitening/whitening.py:196`, `:291`" and that "a temperature other than 1 would be silently wrong outside the whitening code". Both whitening reads are of `optim_state["temperature"]`, and `_init_optim_state` never writes that key (grep over `pyvbmc/`: the only occurrences are the two reads and the `.ini` declaration), so `T` is always 1 even inside the whitening code. The option is completely inert, which strengthens the entry's conclusion about the two commented-out `vptrain2real` calls but means the option belongs with the "declared options that nothing reads" rather than being described as reaching the warp.
- **Deliberate and apparently settled, but absent from the sheet:** `_compute_true_diagnostic` (`vbmc.py:3061-3091`) draws its 10^6 diagnostic samples from a deep copy of the posterior's generator so that the run's stream is untouched, where `vbmc.m:772-776` draws them from MATLAB's global stream and thereby moves every later draw. This is a deliberate Python change of the same kind as the entries already on the sheet under "Randomness is threaded through `numpy.random.Generator`", but that entry does not mention the diagnostic.
- **Also absent, and worth one line so it is not re-reported:** MATLAB's `options.WarmupOptions` (a whole special option set for the warm-up stage, `vbmc.m:431-445`, restored at `:830`) is unported. It is already registered in `INERT_OPTIONS` (`pyvbmc/vbmc/options.py:43`, `warmup_options`), so the sheet's "Declared options that nothing reads" entry covers it implicitly; the restore line `options = options_main` that PyVBMC leaves as a comment (`vbmc.py:1657-1658`) is a no-op at the MATLAB default `WarmupOptions = []`.
