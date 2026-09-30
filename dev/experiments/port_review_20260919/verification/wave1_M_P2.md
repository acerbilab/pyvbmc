# Wave 1 verification: slices M and P2 (MATLAB changes since the port; initial design and active-sampling search)

Verification of the findings of `../reviews/M_comparison.md` and
`../reviews/P2_comparison.md` by a fresh Opus agent, 2026-09-19, on
`dev-port-review` and MATLAB VBMC at `396d649`. The agent wrote its own
checks, dated each finding from both histories, and searched `dev/`, the
porting log and the commit messages for recorded decisions. Its check
scripts ran in the session scratchpad and are not retained. The text below
is its final message, unedited.

Corrections by the orchestrator, from the independent check of the wave-1
pass (2026-09-22; `wave1.md`, which also records the disposition of every
finding below, since this ledger has no column for it):

- The revision read is `cb2d513`: `pyvbmc/` is the same there as at
  `6c151cb`, and every Python citation below resolves at it.
- **P2 F8 = M F3.** The verdict and "fires on noisy targets" do not hold for
  noisy targets, and the evidence's "MATLAB's rank-1 path and its full
  re-update agree" rests on a path MATLAB never takes there:
  `gplite/gplite_post.m:76-79` sets `update1 = false` whenever `s2` is
  given, so on a noisy target MATLAB recomputes the whole posterior, as
  PyVBMC's `and` did. The two differed only on a noiseless repeat. Item 3 of
  "Not mentioned by either report" recommends the rank-one update for a
  first noisy observation with its variance; `510a493` made that change,
  which takes PyVBMC away from MATLAB, and the PI ruled in wave 4 that it
  stays (`wave4.md`, row W4-11; `../matlab_side_defects.md`, entry 23).
- **P2 F9 and "M Sheet notes".** The correction of M's cause for the
  one-dimensional override does not hold. The message of `4949c82` includes
  "fix: D=1 is not supported by cmaes.", and `cma` 4.4.4 warns at
  construction that optimization in one dimension is not supported. With the
  bounds that active sampling passes, `cma.fmin` raises `ValueError` at
  `D = 1` once the step size outgrows a third of the bound range (10 of 50
  seeded runs on a flat objective, 2 of 60 searches on stored
  one-dimensional acquisition states); the reproduction below passed no
  bounds.
- **P2 F10.** "No effect at defaults" holds at the noiseless defaults only:
  the noisy defaults turn on the GP refit inside the active-sampling loop,
  which reads the count. P2's statement that `optimize_vp` reads `n_eff` is
  wrong; in `variational_optimization.py` only `update_K` does.
- **P2 minor observations.** The ledger lists eight, not seven.
- **M Sheet notes.** `shared/msmoothboxrnd.m:57,63` and `:69` are `:59`,
  `:66` and `:72`.

---

## Verdict table

| id | statement | verdict (class) | fires at defaults? | how verified | record |
|---|---|---|---|---|---|
| P2 F1 | CMA-ES gets scalar `sigma0=max(insigma)`, losing MATLAB's per-coordinate `diagD=insigma/max(insigma)` | confirmed port discrepancy | yes (D>1) | read `cmaes_modded.m:561-567`; `cma` run: sd `[1.006,0.992]` vs `[1.006,0.0099]` with `CMA_stds` | none found |
| P2 F2 = M F1 | `fun_eval_start` default | confirmed port discrepancy, **already fixed** by `cb2d513`; fix matches MATLAB for D=1..25 | yes (D≥10, before fix) | evaluated both formulas for D=1..25, identical | `cb2d513` commit message |
| P2 F3 | surplus starting points deleted from the cache instead of kept | confirmed port discrepancy | no (needs `len(x0) > fun_eval_start`) | `initdesign_vbmc.m:26,34` (`false(N0,1)`) vs `active_sample.py:160,170`; reproduced: cache `(0,2)` where MATLAB leaves 3 rows | sheet entry covers only the selection rule |
| P2 F4 | `N_random_points` subtracts the whole cache, not the points taken | confirmed port discrepancy (masked by F3) | no | reproduced: `(5,3)` returned where MATLAB returns 10 rows | none found |
| P2 F5 | CMA-ES noise handling enabled; MATLAB's `Noise.on = 0` | confirmed port discrepancy, knowingly retained | yes (D>1) | `cmaes_modded.m:215`; `evolution_strategy.py:5054-5067,5117-5124`; run: 546 vs 451 evals, 66 vs 75 iters, `tolfacupx=inf` | `dev/plans/stage2-batched-acquisition.md` §Decisions ("Noise handler kept") — retention only, no MATLAB-parity claim |
| P2 F6 | `search_cmaes_best` inert; PyVBMC always returns the best-ever solution | confirmed port discrepancy | yes (D>1) | `_result0` → `self.best.get()` (`evolution_strategy.py:3255-3267`); `cmaes_modded.m:1708`; 4/12 seeds had last-gen best worse than best-ever | none found |
| P2 F7 | `search_cache_frac > 0` raises on the first active-sampling step | confirmed Python-only defect | no | reproduced `ValueError` (dimension mismatch) and the `whitening.py:226` ambiguity `ValueError` | none found |
| P2 F8 = M F3 | rank-1 GP update gated with `and` where MATLAB has `\|\|` | confirmed port discrepancy (cost + rounding only) | yes, on noisy targets | `activesample_vbmc.m:481` vs `active_sample.py:730-733`; `get_traindata_vbmc.m:9` ≡ `_get_training_data` confirm `s2 = S²` on both sides | none found |
| P2 F9 | `D == 1` switches to an unbounded scipy Nelder-Mead | confirmed port discrepancy (Python-only substitution); M's stated cause does not hold | yes (D=1) | scipy `minimize` called with no `bounds`, no `maxfev`; measured scipy default budget 200 vs MATLAB 1500; **`cma.fmin` runs fine at D=1** | commit `4949c82` (2021-11-04) message records the change, not a rationale |
| P2 F10 | `active_sample` writes `optim_state["N_eff"]`; readers use `n_eff` | confirmed Python-only defect | no effect at defaults | grep: no source module reads `N_eff`; MATLAB readers `get_GPTrainOptions.m:77,98`, `gptrain_vbmc.m:118`, `vpsieve_vbmc.m:9,36`, `updateK.m:7` all confirmed | none found |
| P2 F11 | `noise_shaping` live switch, shaping never applied | confirmed port discrepancy (unported feature, partially live option) | no | grep: `noise_shaping` read at `vbmc.py:1073` and `active_sample.py:733` only; `noise_shaping_threshold/_factor` read nowhere | none found |
| P2 F12 | non-finite fallback builds `(N,D)` bounds | confirmed Python-only defect, latent (see below — weaker reachability than stated) | no | `np.minimum(gp.X, x0).shape == (3,2)`; could not construct a non-finite search bound | none found |
| P2 F13 | no `try/catch` fallback around the local search | confirmed port discrepancy | only on an optimizer exception | `activesample_vbmc.m:264,317-320` vs `active_sample.py:509-553` | none found |
| P2 F14 | `active_sample_fess_thresh` declared, never read | confirmed port discrepancy (inert option) | no (default behavior matches) | grep: option only on `.ini:299`; `active_sample.py:667` hardcodes `fESS_thresh = 1` | none found |
| P2 F15 | `vp_repo` append is a discarded expression | confirmed Python-only defect, inert; **reviewer's reasoning wrong** | no | `vbmc.py:1037` init is commented out; `:1658` is a warmup-end reset | `pyvbmc/vbmc/README.md` records `VariationalInitRepo` as unported |
| P2 F16 | one unbalanced `gp_train` timer; untimed initial design; no subtraction | confirmed Python-only defect (diagnostics) | (a) only with `vp_update=True, gp_update=False`; (b),(c) always | `Timer.start_timer` ignores a duplicate key; `vbmc.py:1467/1481`; `activesample_vbmc.m:533-535` | none found |
| P2 minor obs. (7) | see below | all confirmed | — | read + arithmetic | — |
| M F2 | six options MATLAB deleted in 2021 survive as dead `.ini` entries | confirmed port discrepancy (surface only) | no numerical effect | grep: each name only on its `.ini` line; `VBMC(..., options={"double_gp": True, ...})` constructs without error | none found |
| M F4 | MATLAB's `samples` output struct has no counterpart in `results` | confirmed port discrepancy (public surface); line cite off | no numerical effect | `vbmc.m:942-956` (not 946-962); `_create_result_dict` key list has no equivalent | none found |
| M F5 | `warp_cov_reg` non-numeric branch cannot execute | confirmed Python-only defect | no (`warp_cov_reg = 0`) | `options.warp_cov_reg` → `AttributeError: 'Options' object has no attribute 'warp_cov_reg'` | none found |
| M Sheet notes | 46b6f5e misattribution; `gplite_pred` note; probit/logit; `SearchOptimizer` values; D==1 override; MATLAB-side defects | all confirmed except the stated cause of the D==1 override | — | see below | — |

## Evidence and corrections

**P2 F1.** `utils/cmaes_modded.m:558-567`: `sigma = max(insigma)`, `diagD = insigma/max(insigma)`, `diagC = diagD.^2`, `C = diag(diagC)` — initial sampling covariance `diag(insigma.^2)`. `active_sample.py:516-542` passes only `np.max(insigma)` and sets neither `CMA_stds` nor `scaling_of_variables`. Measured: `CMAEvolutionStrategy([0,0], 1.0).ask(4000)` gives per-coordinate sd `[1.006, 0.992]` and `es.stds = [1, 1.000025]`; adding `CMA_stds = insigma/max(insigma)` gives `[1.006, 0.0099]` and `es.stds = [1, 0.01]`, i.e. MATLAB's shape. Unchanged in both histories.

**P2 F2 / M F1.** `cb2d513` sets `fun_eval_start = 10 * ceil((D + 1) / 10)`; `ceil` is `math.ceil` (imported in `options.py`), the value is an `int`, and it equals `10*ceil((D+1)/10)` for every D I checked (1,2,5,9,10,11,15,19,20,25). Correction to P2's history line ("MATLAB's comment text copied verbatim and no recorded rationale"): the port was faithful at port time — MATLAB read `max(D,10)` until `c387612` (2022-10-26) changed it, and the Python line has read `np.maximum(D,10)` since `d72c7df` (2021-05-29, as `funevalstart`). M's framing (a missed upstream change) is the correct one; P2's `d72c7df` date is also correct, M's "since at least `d9bfe16`" is weaker but not wrong. The option also anchors the GP-sample schedule (`gaussian_process_train.py:627,635`, mirroring `get_GPTrainOptions.m:98`), which the commit message records.

**P2 F3.** MATLAB `initdesign_vbmc.m:26` sets `idx_remove = true(N0,1)` in the `N0 <= Ns` branch and `:34` sets `false(N0,1)` in the `N0 > Ns` branch, marking only the `Ns` chosen rows. `active_sample.py:160` and `:170` set all-True in **both**. Reproduced with an 8-row cache and `sample_count=5`: `cache["x_orig"].shape == (0, 2)`, MATLAB would leave 3 rows. **Correction:** the consequence "the cache is permanently empty, the cache share of the sieve is dead code" is not a difference at the defaults — MATLAB empties the cache too in the `N0 <= Ns` branch, so `getSearchPoints`'s cache share is equally dead there. The difference is confined to `len(x0) > fun_eval_start`, where MATLAB keeps `N0 - Ns` rows with their `y_orig` for evaluation-free reuse (`activesample_vbmc.m:376-392`).

**P2 F5.** All three mechanisms confirmed in `cma` 4.4.4. Corrections: (c) `es.sp.cmean *= exp(-kappa·tanh(noiseS))` is clipped at 1, so it only shrinks when `noiseS > 0`; on my deterministic run `cmean` stayed `1.0`. (b) is real and measurable — the same seeded search used 546 evaluations against 451 without the handler, converged in 66 generations instead of 75, and reached a different point, and `es.countevals += noisehandler.evaluations_just_done` charges the extra evaluations against `maxfevals`. The record (`dev/plans/stage2-batched-acquisition.md` §Decisions) keeps the handler explicitly and says removing it "would be an algorithmic change, out of Stage 2"; it does not claim MATLAB parity, and the handler dates from the original port (`4949c82`, 2021-11-04, `noise_handler=cma.NoiseHandler(np.size(x0))`). So: a real discrepancy, deliberately retained, with the MATLAB comparison never made.

**P2 F6.** `_result0` (`evolution_strategy.py:3255-3267`) returns `self.best.get()`, and `BestSolution.update` (`optimization_tools.py:340-384`) keeps the minimum over all updates — best-ever. `cmaes_modded.m:1708`: `xmin = arxvalid(:, fitness.idx(1)); % Return best point of last generation.` Both sides fold in the final mean if it is better (`fmin` calls `es.best.update([mean_pheno], ...)`; MATLAB `:1709-1717`), so the difference is exactly best-ever vs best-of-last-generation. Demonstrated: on Rastrigin with 12 seeds, the last generation's best was worse than the best-ever in 4 of 12 (seed 3: 0.895 vs 0.995). `search_cmaes_best` (`advanced_vbmc_options.ini:185`) is read nowhere.

**P2 F7.** Reproduced verbatim: `optim_state["search_cache"]` is `[]` at `vbmc.py:1011`, `_get_search_points` raises `ValueError: all the input arrays must have same number of dimensions...`, and `_get_search_points` runs (`active_sample.py:355`) before the cache is first written (`:416`). The follow-on `whitening.py:226` guard also reproduces: `if optim_state.get("search_cache"):` on a `(4,3)` array raises the ambiguity error.

**P2 F8 / M F3.** Line correction: M cites `activesample_vbmc.m:468` — that line is `optimState.vp_repo{end+1} = ...`; the `update1` line is **`:481`** (P2 is right). Both `get_traindata_vbmc.m:9` and `_get_training_data` use `S²` without the `nevals` factor, and `s2new = S[idx_new]²`, so MATLAB's rank-1 path and its full re-update agree. **Extra neither report draws:** the noiseless-repeat cell where MATLAB takes the rank-1 path is a MATLAB-side defect. `funlogger_vbmc.m:232-247` pools a repeat into the existing row without incrementing `Xn`, so `gplite_post(gp,xnew,ynew,...)` appends a GP training row that is not in `optimState.X`. A literal port of the `||` would import that. The safe condition is `n_evals[idx_new] == 1 and not noise_shaping`, with `s2_new=s2new` passed to `gp.update` (`gpyreg` supports it) — which is what P2 means by "the two are coupled".

**P2 F9.** All three sub-claims confirmed by reading `active_sample.py:481-551`: no `bounds=`, no `maxfev`, and `options.__setitem__(..., force=True)` mutates the run's frozen options permanently. scipy 1.18.1 Nelder-Mead at D=1 has default `maxfev = 200*N = 200` against MATLAB's `SearchMaxFunEvals = 500*(D+2) = 1500`. **Correction to M's sheet note 3:** the stated cause ("a consequence of the `cma` package needing `D ≥ 2`") does not hold — `cma.fmin(lambda x: (x[0]-0.4)**2, [0.0], 0.5)` runs and converges at D=1. The only record is the PR commit text in `4949c82` ("feat: add Nelder-Mead method for 1D acquisition function optimization"), which states the change without a reason.

**P2 F10.** Confirmed; two additions. (i) MATLAB refreshes `optimState.Neff` in **two** places — `activesample_vbmc.m:84` and `funlogger_vbmc.m:279` (every logged evaluation); PyVBMC has the second one commented out at `function_logger.py:347` (`# optimstate.N_eff = np.sum(...)`), so `n_eff` is stale through the whole active-sampling loop, not only past the per-step write. (ii) `N_eff` is not literally absent from the repository: it appears as a captured key in the oracle fixture manifests (`pyvbmc/testing/oracles/fixtures/*.json`), because those snapshot `optim_state`. No source module reads it.

**P2 F12.** Shape confirmed: `np.minimum(gp.X, x0)` on `(3,2)` returns `(3,2)`; `.squeeze()` keeps it, so `cma` would get an `(N,D)` bound. **Strengthening the reachability caveat:** I could not produce a non-finite `lb_search`/`ub_search`. `ParameterTransformer.inverse` clamps to the bound and the forward transform of the bound is finite (logit, `[0,1]` box: `1e6 → 1.0 → 8.36`), so `whitening.py:222-223` does not appear able to write one either. MATLAB's only ±Inf writer for these bounds (`warp_input_vbmc.m:134-141`) is inside an `if 0` block and is correctly not ported. Classification stands as latent/Python-only, but the branch looks practically unreachable rather than merely "normally unreachable".

**P2 F15.** **Correction to the reasoning.** `vbmc.py:1037` (`# optim_state["vp_repo"] = []`) is commented out, so `optim_state.get("vp_repo")` is `None` at the start and the `else` branch *does* run once, storing one element; `vbmc.py:1658` is the warmup-end reset to `[]`, after which the `if` branch always runs and discards. The whole block is inside `if options["active_sample_vp_update"]` (off by default). The conclusion is unchanged: the repository never accumulates, `np.append`'s result is thrown away, and nothing reads it.

**P2 F16.** (a) confirmed: `Timer.start_timer` (`pyvbmc/timer/timer.py:28`) ignores a start for an existing key, so the unstopped `gp_train` start at `active_sample.py:669` makes `vbmc.py:1467` a no-op and `:1481` attribute the whole span. `timer.reset()` at `vbmc.py:1211` bounds the leak to one iteration. (b) confirmed: the `gp is None` branch has no global `fun_time` timer. (c) confirmed: `activesample_vbmc.m:533-535` subtracts `t_func`, `gpTrain`, `variationalFit`; PyVBMC records raw overlapping accumulators (`vbmc.py:1610`). P2's coverage note that `FunctionLogger` uses a *local* `Timer()` is right (`function_logger.py:270,422`), so the `fun_time` pair in `active_sample` is balanced.

**P2 minor observations.** All seven confirmed. `f_val_old = acq_fast[idx]` (`:490`) vs MATLAB's pointwise re-evaluation (`:247`); `round` half-to-even reaching `get_hpd` (`pyvbmc/stats/get_hpd.py:35`, builtin `round`); `np.argsort` default quicksort (not stable) vs MATLAB's stable `sort`; direct option calls at `:699` and `:787` bypassing `Options.eval` (which does handle non-callables); `!= "none"` case-sensitive vs `strcmpi`; draw ordering (MATLAB runs importance sampling *before* `getSearchPoints`, PyVBMC after; `optimState.acqrand = rand()` at `:215` is written and read nowhere in the MATLAB tree — grep confirms one hit); `gp_length_scale` shape `(D,)`; `cache["skip_logger"]` not deleted alongside the other two arrays at `:629-637` (unreachable while F3 keeps the cache empty).

**M F2.** All six names occur only on their `.ini` lines (`:135, 231, 233, 235, 305, 313`). Constructing `VBMC(..., options={"double_gp": True, "empirical_gp_prior": True, "integrate_gp_mean": True, "gp_stochastic_step_size": True, "constrained_gp_mean": True})` succeeds and the values are stored and ignored.

**M F5.** `Options(MutableMapping, dict)` (`options.py:17`) defines no `__getattr__`; `options.warp_cov_reg` raises `AttributeError`. **Extra hazard:** the guard is `type(options["warp_cov_reg"]) == float or type(...) == int`, so a `np.float64` (or `np.int64`) value also falls into the broken branch — `type(np.float64(0.5)) == float` is `False`, where MATLAB's `isnumeric` is true. The fix is `options["warp_cov_reg"](optim_state["N"])` plus a `np.isscalar`/`callable` test.

**M Sheet notes.** `46b6f5e` touches only `gplite/private/fminfill.m` and `utils/fminfill.m` (`git show --stat`); `misc/initdesign_vbmc.m` last changed in `1e17fb8` (2019-09-13) — the `counterpart_map.md` attribution is wrong, as both reviewers say. gpyreg's Python `GP.predict(return_lpd=True)` divides by `y_s2` (`gaussian_process.py:2035-2064`) while its reference copy `matlab/gplite/gplite_pred.m:108` still divides by `sn2_star*sn2_mult`; `return_lpd` appears nowhere under `pyvbmc/`. `defopts.BoundedTransform = 'logit'` (`vbmc.m:344`) vs `bounded_transform = "probit"` (`.ini:311`), and "probit" does not occur anywhere in `known_differences.md`. `NotImplementedError` is at `active_sample.py:553`. `shared/msmoothboxrnd.m:57,63` use `a(idx)`/`b(idx)` (column-1 linear indexing) where the plateau branch at `:69` correctly uses `a(idx,d)`. `misc/setupoptions_vbmc.m:47` has `'ConstrainedGPMean''FeatureTest'`. Also verified: only three commits touched `private/activesample_vbmc.m` after 2021-01-19 (`1d57b49`, `2044530`, `68a197b`), exactly as P2 states, so no P2 finding rests on a post-port MATLAB change.

## Not mentioned by either report

1. **`eval_initial_x`.** MATLAB's `cmaes_modded` has `defopts.EvalInitialX = 'yes'` (`:207`) and `options.CMAESopts` does not override it, so MATLAB evaluates `x0` and charges it to `counteval`. `cma.fmin`'s `eval_initial_x` defaults to `False` (`evolution_strategy.py`, `_get_value(6, 'eval_initial_x', False)`) and PyVBMC does not pass it. Same class as F1/F5/F6 and inside the "compare the settings passed" scope; low consequence, because PyVBMC compares against `f_val_old` afterwards anyway.
2. **`tolfunhist`.** `setupoptions_vbmc.m:170` sets `cmaes_opts.TolHistFun = 1e-13`; PyVBMC sets only `tolfun`, leaving `cma`'s default `tolfunhist = 1e-12`. P2's sheet note caught the `TolX` difference (MATLAB `1e-11*max(insigma)`, `cma` absolute `1e-11` — confirmed) but not this one.
3. **MATLAB's `||` is wrong in one of its two extra cells** (see P2 F8 above): the noiseless-repeat path appends a duplicate GP row. Any fix to `active_sample.py:730-733` should be written as `n_evals[idx_new] == 1 and not noise_shaping` with `s2_new=s2new`, not as a literal transcription of the MATLAB condition.
4. **`funlogger_vbmc.m:279`'s `Neff` refresh is commented out in Python** (`function_logger.py:347`) — the other half of P2 F10.
5. **Pre-computation ordering.** Besides the importance-sampling/`getSearchPoints` swap P2 records, MATLAB computes the `compute_varlogjoint` quantity first in the pre-computations (`activesample_vbmc.m:152-157`) while PyVBMC computes it last (`active_sample.py:397-404`). `_gp_log_joint` draws nothing, so there is no stream effect; noting it only so a later reviewer does not re-derive it.
6. **Test-adequacy claims all hold.** `test_get_search_points_all_cache` and `::_search_bounds` both use `cache_frac = 1` with a cache of exactly `number_of_points` rows (so F4 is invisible); `::_all_search_cache` installs a populated `search_cache` (so F7 is invisible); `::test_active_sample_initial_sample_more_provided` asserts over an array the implementation leaves empty.
