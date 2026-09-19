# P2 — initial design and active-sampling search, MATLAB-comparison track

Raw reviewer report, wave 1 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-19. One Opus agent,
fresh context, read-only on the repositories; its check scripts ran in the
session scratchpad and are not retained. The text below is the agent's
final message, unedited.

---

## 1. Coverage

**Read completely on both sides:**

| MATLAB | Python |
|---|---|
| `private/activesample_vbmc.m:1-542` (main body) | `pyvbmc/vbmc/active_sample.py:31-799` (`active_sample`) |
| `private/activesample_vbmc.m:545-639` (`getSearchPoints`) | `pyvbmc/vbmc/active_sample.py:802-991` (`_get_search_points`) |
| `misc/initdesign_vbmc.m` (whole file) | `pyvbmc/vbmc/active_sample.py:92-202` (the `gp is None` branch) |

**Substituted optimizer.** Per instructions I compared settings, not internals: `utils/cmaes_modded.m` option defaults (`:189-224`), its `insigma`/`diagD` handling (`:514-567`), and its return convention (`:1685-1721`), plus `misc/setupoptions_vbmc.m:165-178` (`options.CMAESopts`), against the installed `cma` package (`.venv/.../cma/evolution_strategy.py:4475-5193` for `fmin`, `cma/optimization_tools.py:528-830` for `NoiseHandler`) and `_BatchedNoiseHandler` (`active_sample.py:994-1053`).

**Unported, confirmed and left:** `misc/proposal_vbmc.m` (its only consumer is `misc/setupvars_vbmc.m:186-189`, which stores `optimState.ProposalFcn`; nothing in the MATLAB tree reads it — dead there too); `private/acqhedge_vbmc.m` (gated on `AcqHedge`, default off); `utils/fastkmeans.m` (only caller `initdesign_vbmc.m:31`); `utils/covcma.m` (only call site `activesample_vbmc.m:601` is commented out); `misc/check_quadcoefficients_vbmc.m` (integrated mean function).

**Called, not reviewed (arguments and signatures checked only):** the acquisition `__call__` and `AbstractAcqFcn._real2int` (verified `X_search` is snapped before the training-repeat rows are prepended, and that the acquisition's in-place snap is why `X_train_cand[[idx]].copy()` is used); `active_importance_sampling(vp, gp, acq_eval, options)` ≡ `activeimportancesampling_vbmc(vp,gp,SearchAcqFcn{idxAcq},acqInfo,options)`; `_gp_log_joint(vp, gp, 0, 0, 0, 1, separate_K=True)` ≡ `gplogjoint(vp,gp,0,0,0,1)` with `out[2] == varF` (`out[6]`, the component covariance, is a Python-only extra); `_neg_elcbo(theta0, gp, vp0, 0.0, NSentFineK, False, True)` ≡ `negelcbo_vbmc(theta0,0,vp0,gp,NSentFineK,0,1)`; `reupdate_gp`, `train_gp`, `optimize_vp`, `get_hpd`, `vp.sample`, `vp.moments`.

**MATLAB code noted as dead/unreachable at the defaults, not reported:** `LB_searchmin`/`UB_searchmin` (`:503-508`, assignments commented out — Python computes the same dead locals at `:759-770`); `ActiveVariationalSamples` (`:86-95`); `Nextra`/`SampleExtraVPMeans` (`:97-142`); the `acqtable` debug block (`:403-409`); `optimState.acqrand = rand()` (`:215`, written and never read anywhere in the MATLAB tree — the dropped draw only shifts the stream); the `y_orig`/`yacq` Jacobian block (`:367-372`) whose result `yacq` is never used (this is what the "Missing port: line 356-361" marker at `active_sample.py:609` refers to; it does not matter); `t_algoperfuneval` (`:204`) — grep confirms its only readers are `:347-351` inside the unported repeat block, so the "Missing port: line 185-205" marker at `:349` is inconsequential, as the sheet says.

**The six "Missing port" markers, judged at the defaults:**
- `:327` noise shaping — matters only with `noise_shaping=True`; see F11.
- `:349` cost model (`activesample_vbmc.m:185-205`) — inconsequential (see above).
- `:564-607` the repeat block — covered by the sheet's P2 entry; Python's replacement is `:378-390`/`:645-651`.
- `:609` (`activesample_vbmc.m:356-361` at port time = current `:367-372`) — the dead `yacq` block; inconsequential.
- `:652` (`:392-402` at port time = current `:403-413`) — debug `acqtable` and the unused `Nextra` restore, plus `tnew = optimState.funevaltime(idx_new)` which feeds only the also-commented `gp.t(end+1)`; inconsequential.
- `:665` (`:425-432` at port time = current `:436-447`) — the fESS branch; at the MATLAB default `ActiveSamplefESSThresh = 1` the branch is skipped and `fESS = 0`, exactly what Python hardcodes. See F14 for the dangling option.

**Checks run** (scripts kept in the scratchpad `port_review/P2_comparison/`, run with the repo venv; no `optimize()` runs, no test suites):
`check_cache_count.py` (reproduces F4), `check_search_cache_empty.py` (reproduces F7), `check_fun_timer.py`/`check_fun_timer2.py` (ruled out a suspected double-timing interaction — `FunctionLogger` uses a *local* `Timer()`, so the global `fun_time` pair in `active_sample` is balanced; no finding), and two one-liners confirming the CMA-ES initial population is isotropic and that `np.minimum(gp.X, x0)` returns an `(N,D)` array.

**Compared and found equivalent** (so the absence of a finding is informative):
- Initial design, `N0 <= Ns` branch: the `plausible` uniform draw in the transformed plausible box, the `narrow` draw `(rand-0.5)*0.1*(PUB-PLB) + warp(x0(1,:))` and its clipping, the inverse transform back to original space, the NaN padding of `ys`, the deletion from the cache, the forward transform of `Xs`, and the `isnan(y) → 'iter' else 'add'` dispatch — all line-for-line the same.
- `getSearchPoints`: `Ncache = ceil(NSsearch*cacheFrac)`; `randperm(n,k)` ≡ `rng.permutation(n)[:k]`; the draw source, location and scale of every share — heavy-tailed VP (`vbmc_rnd(vp,N,0,1,3)` ≡ `vp.sample(N, orig_flag=False, balance_flag=True, df=3)`), MVN at the VP moments, the six-`hpd_frac` ladder (`rand(1,4)` ≡ `rng.uniform(size=4)`, the `sort`, the `diff(round(linspace(...)))` split, `cov(X_hpd,1)` ≡ `bias=True`, the empty-HPD fallback `max(y)`/`cov(X)` with `N-1`, the `isscalar → ones(D,D)` promotion), the uniform box (now `rng.random`, matching `rand(Nbox,D)`, with the same `box_lb/box_ub` construction in both the finite and infinite `LB_search` branches), and the remainder `Nvp = max(0, Nrnd - ...)` drawn from the balanced VP; the final clipping to `[LB_search, UB_search]`; the `0`-vs-`NaN` "not from cache" sentinel, consistently used at `idx_cache_acq`.
- Acquisition sieve: `NSsearch = 2^13`; `randi(numel(SearchAcqFcn))` ≡ `rng.integers(len(...))` (1-based vs 0-based handled correctly); the `SearchCacheFrac > 0` sort-and-store versus plain `min`; `optimState.SearchCache` holds the whole sorted candidate set; selection, removal from the search set, and `x0 = real2int(Xacq(1,:))` (idempotent in Python because the whole set was snapped at `:367`).
- Pre-computations: `sn2new` per hyperparameter sample (`gplite_noisefun(hyp_noise,gp.X,gp.noisefun,gp.y,s2)` ≡ `gp.noise.compute(hyp_noise, gp.X, gp.y, s2)`, with `s2 = S²·nevals` over live rows and `isfield(optimState,'S')` ≡ `hasattr(function_logger,"S")` — `S` exists only when `noise_flag`); `gp.sn2new = mean(sn2new,2)`; the geometric-mean length scale; `gp.X_rescaled`. Shapes verified `(N,1)` throughout.
- Search-bound expansion: `delta_search = 0.05*(UB_search-LB_search)`, both masks, and the clamp to `optimState.LB/UB` ≡ `lb_tran/ub_tran` (MATLAB's `optimState.LB` is the transformed bound).
- `optimState.LB_search/UB_search` initialization (`setupvars_vbmc.m:271-272` ≡ `vbmc.py:1046-1056`), including `ActiveSearchBound = 2` and the `2*prange` factor in the dead `LB_searchmin`.
- Bounds for the local search in the finite branch: `min([x0;LB_search])` ≡ `np.minimum(x0, lb_search)`.
- `TolFun`: `1e-2` under `log_flag`, else `max(1e-12, |fval_old|*1e-3)`.
- Full-update gating: `(ActiveSampleVPUpdate || ActiveSampleGPUpdate) && ((iter - ActiveSampleFullUpdatePastWarmup) <= LastWarmup || rindex(end) > ActiveSampleFullUpdateThreshold)` — the 1-based/0-based shift cancels on both sides of the `<=`, and `iteration_history["r_index"][-1]` is the last *recorded* iteration (the history array is grown to exactly `iteration+1`), matching `stats.rindex(end)`.
- `options_update`: `GPTolOpt/GPTolOptMCMC/TolWeight=0/NSent/NSentFast/NSentFine` all mapped with matching defaults.
- Hyperparameter-struct threading in the full-update loop: I checked this specifically because Python's post-loop `optim_state["hyp_dict"] = hyp_dict` looks like it discards the in-loop updates. It does not — `train_gp` mutates `hyp_dict` in place and returns the same object (`gaussian_process_train.py:66-211`), so Python's local, `optim_state["hyp_dict"]` and `train_gp`'s return are one object, equivalent to MATLAB's explicit `hypstruct` threading. **No finding.**
- `Nfastopts = ceil(NSelboIncr * NSelbo(K))`, `UpdateRandomAlpha → 1 - sqrt(rand())`, `vpoptimize_vbmc(Nfastopts,1,vp,gp,[],...)` ≡ `optimize_vp(..., N_fastopts, slow_opts_N=1)` with `K` defaulted.
- Post-loop VP accept/reject: `NSentFineK = ceil(NSentFineActive(K0)/K0)`, the `numel(theta0) ~= numel(theta) || any(theta0 ~= theta)` guard, and `elbo0 > vp.stats.elbo → vp = vp0`.
- `optim_state["N"] = function_logger.Xn + 1` ≡ `optimState.N = optimState.Xn` (1-based vs 0-based index of the last filled row).
- Loop-termination index `if i + 1 < sample_count` ≡ `if is < Ns`.
- Option defaults that agree: `NSsearch 2^13`, `CacheFrac 0.5`, `SearchCacheFrac 0`, `HeavyTailSearchFrac 0.25`, `MVNSearchFrac 0.25`, `HPDSearchFrac 0`, `BoxSearchFrac 0.25`, `HPDFrac 0.8`, `SearchOptimizer cmaes`, `SearchCMAESVPInit yes`, `SearchMaxFunEvals 500*(D+2)`, `ActiveSearchBound 2`, `MaxRepeatedObservations 0`, `NoiseShaping no`, `InitDesign plausible`, `AcqHedge no`, `ActiveSampleVPUpdate/GPUpdate no`, `ActiveSampleFullUpdatePastWarmup 2`, `ActiveSampleFullUpdateThreshold 3`, `ActiveSamplefESSThresh 1`, `GPTolOptActive 1e-4`, `GPTolOptMCMCActive 1e-2`, `NSentActive/FastActive/FineActive`, `NSelbo/NSelboIncr`, `TolWeight 1e-2`, `UpdateRandomAlpha no`, `FunEvalsPerIter 5`.

**MATLAB history.** Only three commits touched these files after 2021-01-19: `1d57b49` (2021-05-26, comments out a `pause`), `2044530` (2021-06-18, cleanup — removed dead blocks, added comments, no semantic change to any line below), `68a197b` (2022-06-25, added the `'slicesample'` optimizer case, unported and not the default). `misc/initdesign_vbmc.m` has not changed since 2019. So **no finding below rests on a post-port MATLAB change**; I note this once rather than repeating it.

**Not reached:** `utils/cmaes_modded.m`'s internals beyond the sections named above (explicitly out of scope); the bodies of the acquisition functions, the importance sampler, `reupdate_gp`, `train_gp`, `optimize_vp`.

---

## 2. Findings

### F1. The CMA-ES search loses MATLAB's per-coordinate initial step sizes
- Location: `pyvbmc/vbmc/active_sample.py:516-542` (`insigma = np.sqrt(np.diag(Sigma))` then `cma.fmin(acq_fun, x0, np.max(insigma), options=cma_options, ...)`); MATLAB: `private/activesample_vbmc.m:275-283`, `utils/cmaes_modded.m:561-567`
- Category: defaults (optimizer settings)
- Proposed classification: port discrepancy
- Confidence: high (that the code differs and what it does); medium on the size of the effect
- History: unchanged since before 2021-01-19.
- MATLAB passes the whole vector `insigma = sqrt(diag(Sigma))` to `cmaes_modded`, which sets `sigma = max(insigma)` as the overall step size **and** `diagD = insigma/max(insigma)` as the coordinate scaling, so the initial sampling covariance is `diag(insigma.^2)`. PyVBMC passes only the scalar `np.max(insigma)` as `sigma0` and sets neither `CMA_stds` nor `scaling_of_variables`, so `cma` starts from `C = I` and the initial covariance is `max(insigma)² · I` — isotropic at the *largest* coordinate scale. Verified empirically: with `insigma = [1.0, 0.01]`, `cma.CMAEvolutionStrategy([0,0], 1.0).ask(4000)` has per-coordinate SD `[1.006, 0.992]`, where MATLAB would give `[1, 0.01]`. MATLAB is right: the whole point of `SearchCMAESVPInit` (default on) is to shape the search by the variational covariance.
- Consequence if real: on every non-isotropic target with `D > 1` — the normal case, since `Sigma` is the VP covariance in transformed space — the local acquisition search begins by sampling far outside the relevant region in the narrow directions and must spend generations adapting `C`. With `maxfevals = 500*(D+2)` it often will not fully recover, so a different point is acquired at every active-sampling step of every iteration. This fires at the defaults.
- Suggested reproduction: no MATLAB needed. Build one stored oracle state with a strongly anisotropic VP, run the `cmaes` branch as written, then again with `cma_options["CMA_stds"] = insigma / np.max(insigma)`, and compare the returned `f_val_optim` and the acquired point.
- Test adequacy: no. `test_vbmc_active_sample.py::test_active_uncertainty_sampling` does exercise the CMA-ES branch end to end (it swaps in Rosenbrock as the acquisition and asserts the minimum is found to 1e-3), but Rosenbrock's `insigma` from the HPD/VP covariance is near-isotropic there, so the missing scaling does not show. The `active_sample_step` oracle pins PyVBMC's own output, so it locks the current settings in rather than checking them.

### F2. `fun_eval_start` default differs from MATLAB for `D >= 10`
- Location: `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:11` (`fun_eval_start = np.maximum(D, 10)`); MATLAB: `vbmc.m:198` (`defopts.FunEvalStart = '10*ceil((D+1)/10)'`)
- Category: defaults
- Proposed classification: port discrepancy
- Confidence: high
- History: MATLAB unchanged. The Python value has been `np.maximum(D, 10)` since the option files were introduced (`d72c7df`, 2021-05-29), with MATLAB's comment text copied verbatim and no recorded rationale.
- This option is `sample_count` for the initial-design branch under review (`vbmc.py:1385`). The two agree for `D <= 9` (both 10) and diverge above: `D=10` → MATLAB 20, Python 10; `D=15` → 20 vs 15; `D=20` → 30 vs 20.
- Consequence if real: for `D >= 10` PyVBMC starts from a systematically smaller initial design — half of MATLAB's at `D = 10`. That shifts the first GP fit, warmup, and the whole trajectory, and it is exactly the regime (10-20 parameters) the method is advertised for. Fires at the defaults with no user action.
- Suggested reproduction: none needed beyond reading the two expressions; a one-line evaluation of both for `D = 1..20` settles it.
- Test adequacy: no test asserts the default value. `test_options.py` checks the machinery, not the numbers. (This straddles P1b's option table; I report it because it sets the initial-design size in this slice.)

### F3. Surplus starting points are deleted from the cache instead of being kept for later use
- Location: `pyvbmc/vbmc/active_sample.py:167-188` (`idx_remove = np.full(provided_sample_count, True)` in the `provided_sample_count > sample_count` branch, then all three cache arrays are deleted); MATLAB: `misc/initdesign_vbmc.m:29-46` (`idx_remove = false(N0,1)`, set true only for the `Ns` chosen rows)
- Category: control flow / state
- Proposed classification: port discrepancy (the known-differences sheet covers only part of it)
- Confidence: high
- History: MATLAB unchanged since 2019.
- When more than `fun_eval_start` starting points are supplied, MATLAB removes only the `Ns` points it consumed and leaves the other `N0 - Ns` in `optimState.Cache` **together with their `y_orig` values**. Those survivors then feed `getSearchPoints` (`CacheFrac = 0.5`, so up to half the sieve can come from them) and, when one is selected, `funlogger_vbmc(...,'add',y_orig)` reuses the cached value instead of calling the target (`activesample_vbmc.m:379-392`). PyVBMC deletes all `N0` rows, so the cache is permanently empty, the cache share of the sieve is dead code, and any pre-computed `f_vals` beyond the first `sample_count` are thrown away. The sheet's entry ("PyVBMC takes the first `sample_count` rows") describes only the selection rule, not the discard.
- Consequence if real: a user who supplies many starting points with known function values loses all but the first `sample_count` of them — both as free evaluations and as search candidates. Triggers whenever `len(x0) > fun_eval_start`. Does not fire at the defaults (single `x0`).
- Suggested reproduction: no MATLAB needed. Call `active_sample(gp=None, ...)` with a 100-row cache and `sample_count=90` and inspect `optim_state["cache"]["x_orig"]` — it is empty; MATLAB would leave 10 rows.
- Test adequacy: no. `test_active_sample_initial_sample_more_provided` asserts `np.all(np.isnan(optim_state["cache"]["x_orig"][:sample_count]))`, which is vacuously true on the empty array the implementation leaves behind; it mirrors the implementation rather than the specification.

### F4. `_get_search_points` derives the random-point count from the whole cache, not from the points it took
- Location: `pyvbmc/vbmc/active_sample.py:867-868` (`if x0.shape[0] < number_of_points: N_random_points = number_of_points - x0.shape[0]`); MATLAB: `private/activesample_vbmc.m:561-562` (`if size(Xsearch,1) < NSsearch; Nrnd = NSsearch - size(Xsearch,1);`)
- Category: indexing/shape (a fraction applied to the wrong count)
- Proposed classification: port discrepancy
- Confidence: high
- History: MATLAB unchanged.
- MATLAB subtracts the number of cache points actually taken, `min(ceil(NSsearch*CacheFrac), size(x0,1))`. Python subtracts the *available* cache size `x0.shape[0]`, which is generally larger. Whenever the cache holds more than `ceil(ns_search*cache_frac)` points the sieve is short, and when it holds at least `ns_search` points no random candidates are generated at all. Reproduced: with `cache_frac=0.5`, a 20-row cache and `number_of_points=10`, `_get_search_points` returns `(5, 3)`; MATLAB returns 10 (5 cache + 5 random).
- Consequence if real: an acquisition sieve smaller than `ns_search`, in the limit consisting only of untransformed starting points with no VP/MVN/box candidates. Currently masked: F3 empties the cache, so this branch never runs at present. It becomes live the moment F3 is fixed, which is why I report it separately.
- Suggested reproduction: `scratchpad/port_review/P2_comparison/check_cache_count.py` (already run; output above).
- Test adequacy: no. `test_get_search_points_all_cache` and `test_get_search_points_search_bounds` both use `cache_frac = 1` with a cache of exactly `number_of_points` rows, so `N_cache == x0.shape[0]` and the two expressions coincide; they then assert `idx_cache.shape == (number_of_points,)`, which is what the implementation happens to produce. A specification test would set `cache_frac = 0.5` with a cache larger than `cache_frac*ns_search` and assert the result still has `ns_search` rows.

### F5. PyVBMC turns on CMA-ES noise handling; MATLAB runs the search with it off
- Location: `pyvbmc/vbmc/active_sample.py:539-542` (`noise_handler=_BatchedNoiseHandler(np.size(x0), acq_fun, vp.rng)` passed to `cma.fmin`); MATLAB: `utils/cmaes_modded.m:215` (`defopts.Noise.on = '0 % uncertainty handling is off by default'`), `misc/setupoptions_vbmc.m:165-178` (`options.CMAESopts` never sets `Noise.on`)
- Category: defaults (optimizer settings)
- Proposed classification: possibly intentional (the sheet records the subclass, not the decision to enable the mechanism)
- Confidence: high on the code, medium on the size of the effect
- History: MATLAB unchanged.
- MATLAB's acquisition search runs with `Noise.on = 0`, i.e. `noiseReevals = 0` and no noise-driven sigma adaptation. In `cma.fmin`, passing any `noise_handler` sets `noise_handling = True` (`evolution_strategy.py:5054-5067`), which (a) sets `es.opts['tolfacupx'] = inf`, disabling that stopping criterion, (b) re-evaluates `2 + popsize/20` solutions per generation at a perturbation of `epsilon = 1e-7` and multiplies `es.sigma` by the handler's return each generation, and (c) shrinks `es.sp.cmean` by `exp(-kappa·tanh(noiseS))` (`:5117-5124`). The acquisition is deterministic, so the "noise" measured is its local variation, which is nonetheless nonzero and therefore does move sigma.
- Consequence if real: a different step-size trajectory and a different stopping behavior from MATLAB's, plus roughly `2 + popsize/20` extra acquisition evaluations per generation charged to `es.countevals` against `maxfevals = 500*(D+2)`. Fires at the defaults on every `D > 1` run.
- Suggested reproduction: no MATLAB needed. Run one CMA-ES search from a stored oracle state with and without the `noise_handler=` argument and compare `res[1]`, the returned point, and `res[3]` (evaluations).
- Test adequacy: no test covers the CMA-ES option set. `_BatchedNoiseHandler.indices` is a faithful copy of `cma`'s stock `choice == 1` policy with `np.random.rand()` replaced by the run's generator, and its `reeval` performs the same `ask` calls in the same order before the one batched objective call — I checked both against `cma/optimization_tools.py:740-825` and found them correct; the finding is about enabling the handler at all.

### F6. `search_cmaes_best` is never read, and PyVBMC always returns CMA-ES's best-ever solution
- Location: `pyvbmc/vbmc/active_sample.py:544` (`xsearch_optim, f_val_optim = res[:2]`), option declared at `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:185` and read nowhere; MATLAB: `private/activesample_vbmc.m:282-290`, `vbmc.m:282` (`defopts.SearchCMAESbest = 'no'`), `utils/cmaes_modded.m:1708-1721`
- Category: defaults / control flow
- Proposed classification: port discrepancy
- Confidence: high
- History: MATLAB unchanged.
- `cma.fmin` returns `es._result0`, whose `xbest/fbest` come from `self.best.get()` — the best solution ever evaluated (`evolution_strategy.py:3255-3267`, `:5193`). That is MATLAB's `bestever`, which `activesample_vbmc.m:285-288` uses only when `SearchCMAESbest` is on; at the MATLAB default it uses `xsearch_optim = xmin`, "the best point of the last generation" (or the final mean if better, `cmaes_modded.m:1708-1721`). PyVBMC therefore behaves as if `SearchCMAESbest = yes` while shipping an option that says otherwise and that nothing reads.
- Consequence if real: a different acquired point whenever the last generation's best is worse than the run's best — routine once the population has drifted. Because the acquisition is deterministic, PyVBMC's choice is never *worse* by the acquisition's own measure, but it is not MATLAB's and it makes a shipped option inert. Fires at the defaults for `D > 1`.
- Suggested reproduction: no MATLAB needed. From one stored state, compare `res[0]` against `res[7].best.last.x`/the final `xfavorite` for the same run.
- Test adequacy: no.

### F7. `search_cache_frac > 0` raises on the first active-sampling step
- Location: `pyvbmc/vbmc/active_sample.py:871-880`; initial value set at `pyvbmc/vbmc/vbmc.py:1011` (`optim_state["search_cache"] = []`); MATLAB: `private/activesample_vbmc.m:565-568`, `misc/setupvars_vbmc.m:226`
- Category: control flow
- Proposed classification: suspected defect (Python side only)
- Confidence: high (reproduced)
- History: MATLAB unchanged.
- MATLAB's `[Xrnd; optimState.SearchCache(1:min(end,Nsearchcache),:)]` is a no-op on an empty cache. Python evaluates `np.append(random_Xs, search_cache[:0], axis=0)` with `random_Xs` of shape `(0, D)` and a 1-D empty list, which raises `ValueError: all the input arrays must have same number of dimensions`. Reproduced in situ with `options={"search_cache_frac": 0.25}` (`check_search_cache_empty.py`). A related Python-only break follows once the cache is populated: `pyvbmc/whitening/whitening.py:226` guards with `if optim_state.get("search_cache"):`, which raises "truth value of an array with more than one element is ambiguous" where MATLAB's `warp_input_vbmc.m:152` uses `~isempty`.
- Consequence if real: the option is unusable — the run aborts on the first iteration that has a GP. Does not fire at the defaults (`search_cache_frac = 0` in both).
- Suggested reproduction: `scratchpad/port_review/P2_comparison/check_search_cache_empty.py` (already run).
- Test adequacy: no. `test_get_search_points_all_search_cache` installs a non-empty `search_cache` before calling, so it never exercises the initial state the solver actually starts from.

### F8. The rank-1 GP update condition uses `and` where MATLAB uses `or`
- Location: `pyvbmc/vbmc/active_sample.py:730-733` (`update1 = ((s2new is None) and function_logger.n_evals[idx_new] == 1) and not options["noise_shaping"]`); MATLAB: `private/activesample_vbmc.m:481` (`update1 = (isempty(s2new) || optimState.nevals(idx_new) == 1) && ~options.NoiseShaping;`)
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high
- History: MATLAB unchanged. The Python `and` has been there since the original port (traced back through `343b9c7`, `d34fc15`, `d9bfe16` — only renames).
- MATLAB takes the cheap rank-1 path when there is no user-supplied noise **or** the point is being evaluated for the first time; Python requires both. The two disagree in exactly two cells: a noisy target's first observation at a new point (MATLAB rank-1 with `s2new` passed to `gplite_post`, Python full `reupdate_gp`), and a deterministic target's repeated location (MATLAB rank-1, Python full re-update). Note the Python call `gp.update(xnew, ynew, compute_posterior=True)` passes no `s2_new`, which is only safe *because* of the `and`; the two are coupled.
- Consequence if real: on noisy targets — the common case for the acquisitions in P3/P4 — the GP is fully re-decomposed after every intermediate acquisition instead of rank-1 updated, for every one of `fun_evals_per_iter - 1` inner steps. The posterior should agree to rounding, so the cost is runtime, growing as O(N³) per step instead of O(N²). Fires at the defaults for any run with `uncertainty_handling`.
- Suggested reproduction: no MATLAB needed. On a noisy stored state, evaluate both expressions for a fresh point and time `gp.update(..., s2_new=s2new)` against `reupdate_gp`.
- Test adequacy: no. The tests that reach this line are the end-to-end noisy runs, which cannot distinguish the two paths by value.

### F9. `D == 1` silently switches the run to an unbounded scipy Nelder-Mead
- Location: `pyvbmc/vbmc/active_sample.py:484-488` (`options.__setitem__("search_optimizer", "Nelder-Mead", force=True)`) and `:545-551` (`minimize(acq_fun, x0, method="Nelder-Mead", tol=tol_fun)`); MATLAB: `private/activesample_vbmc.m:265-316` (`cmaes` for every `D`, with `LBounds/UBounds`)
- Category: control flow / defaults
- Proposed classification: possibly intentional, undocumented
- Confidence: high
- History: MATLAB unchanged; it has no Nelder-Mead branch at all.
- Three differences follow. (a) MATLAB uses bounded CMA-ES for `D = 1`; PyVBMC substitutes Nelder-Mead. (b) `scipy.optimize.minimize` is called **without `bounds=`**, so the local search is unconstrained, while every MATLAB optimizer branch (`cmaes`, `fmincon`, `bads`, `slicesample`) receives `LB`/`UB`; the improved point is then accepted at `:555-561` with no clipping, so the acquired point can lie outside `lb_search/ub_search` (it remains inside the user's hard bounds only because the parameter transform maps back into them). (c) `search_max_fun_evals` is not passed, so scipy's default `maxfev = 200*N = 200` applies instead of MATLAB's `500*(D+2) = 1500`. Additionally the `force=True` write mutates the run's option object permanently rather than using a local variable.
- Consequence if real: for one-dimensional problems the whole local acquisition search is a different algorithm with a different (and unbounded) feasible set and a smaller budget.
- Suggested reproduction: no MATLAB needed. On a `D = 1` stored state, check whether `xsearch_optim` can fall outside `optim_state["lb_search"]/["ub_search"]`, and read back `options["search_optimizer"]` after one call.
- Test adequacy: partly. `create_vbmc(1, ...)` fixtures exercise `D = 1`, but nothing asserts that the returned point respects the search bounds.

### F10. `active_sample` writes `optim_state["N_eff"]`; every reader uses `optim_state["n_eff"]`
- Location: `pyvbmc/vbmc/active_sample.py:293-295`; readers at `pyvbmc/vbmc/gaussian_process_train.py:607,635` and `pyvbmc/vbmc/variational_optimization.py:42`; MATLAB: `private/activesample_vbmc.m:84` (`optimState.Neff = sum(optimState.nevals(optimState.X_flag))`), read by `misc/get_GPTrainOptions.m:77,98`, `misc/gptrain_vbmc.m:118`, `misc/vpsieve_vbmc.m:36`, `private/updateK.m:7`
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high
- History: MATLAB unchanged.
- MATLAB refreshes `optimState.Neff` at the top of every active-sampling step (and again inside `funlogger_vbmc.m:279`), so the in-loop GP refit and VP re-optimization see the current count. Python's per-step write goes to a differently spelled key that nothing reads, while `optim_state["n_eff"]` is refreshed only after the loop (`vbmc.py:1459`). Two further details: the value is computed with the builtin `sum` over an `(N,1)` array, so it is a shape-`(1,)` array rather than the scalar `np.sum` produces elsewhere; and `test_active_uncertainty_sampling` has to set `optim_state["n_eff"]` by hand between its two `active_sample` calls, which is a symptom of the same gap.
- Consequence if real: with `active_sample_gp_update` or `active_sample_vp_update` enabled, the in-loop `train_gp` (its `n_eff < 30` branch and its `ns_gp` schedule) and `optimize_vp` (`K_max` from `k_fun_max(n_eff)`) run on the previous iteration's count, lagging by up to `fun_evals_per_iter`. At the defaults both update options are off, so nothing reads a stale value and the impact is nil.
- Suggested reproduction: no MATLAB needed. Enable `active_sample_gp_update`, log `optim_state["n_eff"]` inside the loop, and compare with `sum(n_evals[X_flag])`.
- Test adequacy: no.

### F11. `noise_shaping` is a live option whose shaping is never applied
- Location: `pyvbmc/vbmc/active_sample.py:327` ("Missing port: noise_shaping") and `:733`; `pyvbmc/vbmc/gaussian_process_train.py:823`; the option is read at `pyvbmc/vbmc/vbmc.py:1073`; MATLAB: `private/activesample_vbmc.m:169-171`, `misc/noiseshaping_vbmc.m`, also used by `misc/get_traindata_vbmc.m` and `misc/gpsample_vbmc.m`
- Category: cross-module (unported feature with a live switch)
- Proposed classification: port discrepancy
- Confidence: high
- History: MATLAB unchanged.
- Setting `noise_shaping=True` in PyVBMC does three things: it changes the GP noise function (`vbmc.py:1073`, matching `setupvars_vbmc.m`), it disables the rank-1 update (`:733`, matching MATLAB), and it is *not* applied where MATLAB applies it — neither to `s2` before `gp.noise.compute` here nor in `_get_training_data`. So the option half-works: the GP gains an input-dependent noise term but is never told to discount low-density observations.
- Consequence if real: a user who enables `noise_shaping` gets a configuration that exists in neither toolbox. Off at the defaults in both.
- Suggested reproduction: read the two "Missing port" markers; grep confirms `noiseshaping` has no Python implementation.
- Test adequacy: no test sets `noise_shaping`.

### F12. The non-finite search-bound fallback builds `(N, D)` bounds instead of one per coordinate
- Location: `pyvbmc/vbmc/active_sample.py:500-502`; MATLAB: `private/activesample_vbmc.m:253-254` (`LB = min([gp.X;x0]) - 0.1*xrange`)
- Category: indexing/shape
- Proposed classification: suspected defect (Python side only), latent
- Confidence: high on the shape, high that it is currently hard to reach
- History: MATLAB unchanged.
- MATLAB's `min([gp.X;x0])` is a column-wise minimum over the stacked matrix, giving a `1×D` row. Python's `np.minimum(gp.X, x0)` broadcasts `x0` across rows and returns an `(N, D)` array, so `lb_search.squeeze()` hands `cma` an `(N, D)` bound. Verified: for `gp.X` of shape `(3,2)`, Python yields shape `(3,2)` where MATLAB yields `(1,2)`.
- Consequence if real: the branch runs only when `optim_state["lb_search"]` or `["ub_search"]` contains a non-finite entry. At initialization they are clamped to the finite transformed bounds, and the expansion at `:748-757` keeps them there, so the branch is normally unreachable; `pyvbmc/whitening/whitening.py:222-223` is the one place that could write a non-finite bound (a re-transformed point landing exactly on a bound). If reached, `cma.fmin` receives malformed bounds.
- Suggested reproduction: no MATLAB needed. Set `optim_state["ub_search"][0,0] = np.inf` on a stored state and call the branch.
- Test adequacy: no.

### F13. The local search has no failure fallback
- Location: `pyvbmc/vbmc/active_sample.py:509-553`; MATLAB: `private/activesample_vbmc.m:264, 317-320` (`try ... catch; fprintf('Active search failed.\n'); fval_optim = Inf; end`)
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high
- History: MATLAB unchanged.
- MATLAB wraps the whole optimizer switch in `try/catch`; on any failure it sets `fval_optim = Inf`, so the comparison at `:324` rejects the search result and the sieve's best candidate is used. PyVBMC lets any exception from `cma.fmin` or `scipy.optimize.minimize` propagate out of `active_sample` and abort the run.
- Consequence if real: a recoverable optimizer failure (e.g. a singular `Sigma` giving `insigma` with a zero, or `cma` rejecting degenerate bounds) ends the run instead of costing one acquisition. Rare, but the mitigation MATLAB has is absent.
- Suggested reproduction: none needed; the absence of the handler is the finding.
- Test adequacy: no.

### F14. `active_sample_fess_thresh` is declared but never read
- Location: `pyvbmc/vbmc/active_sample.py:665-668` (`gptmp = None; fESS, fESS_thresh = 0, 1`), option at `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:299`; MATLAB: `private/activesample_vbmc.m:436-443`, `vbmc.m:339` (`ActiveSamplefESSThresh = 1`)
- Category: defaults
- Proposed classification: port discrepancy (inert option)
- Confidence: high
- History: MATLAB unchanged.
- At MATLAB's default (`1`), `fESS_thresh < 1` is false, `fESS = 0`, and the `fESS <= fESS_thresh` branch is taken — exactly what Python hardcodes, so the default behavior matches. But Python ships the option and ignores it: setting `active_sample_fess_thresh = 0.5` has no effect, where MATLAB would pre-update the GP, compute `fess_vbmc(vp, gptmp, 100)` and possibly skip the full update. (`fess` itself is ported, at `pyvbmc/vbmc/active_importance_sampling.py:442`.)
- Consequence if real: none at the defaults; a silently inert option otherwise.
- Suggested reproduction: none needed.
- Test adequacy: no.

### F15. The `vp_repo` append is a discarded expression
- Location: `pyvbmc/vbmc/active_sample.py:715-720` (`np.append(optim_state["vp_repo"], vp.get_parameters())` — the result is not assigned); MATLAB: `private/activesample_vbmc.m:468` (`optimState.vp_repo{end+1} = get_vptheta(vp);`)
- Category: state/caching
- Proposed classification: suspected defect (Python side only), inside an already-unported feature
- Confidence: high
- `optim_state["vp_repo"]` is initialized to `[]` at `vbmc.py:1658`, so `optim_state.get("vp_repo") is not None` is true from the start and the `else` branch that would actually store something never runs; the `if` branch computes a new array and throws it away, so the repository stays empty forever. Its only MATLAB consumer, `misc/vpsieve_vbmc.m:60-66` under `VariationalInitRepo` (default off), is unported by design (`pyvbmc/vbmc/README.md`, and the comment at `variational_optimization.py:803`), so nothing reads it in Python either.
- Consequence if real: none today; it would become one the moment `variational_init_repo` were ported.
- Suggested reproduction: none needed.
- Test adequacy: no.

### F16. Timer accounting: one unbalanced `gp_train` timer, and an untimed initial design
- Location: `pyvbmc/vbmc/active_sample.py:669-691` and `:192-202`; MATLAB: `private/activesample_vbmc.m:446-457`, `:533-535`, `misc/initdesign_vbmc.m:50-57`
- Category: control flow (diagnostics only)
- Proposed classification: suspected defect (Python side only)
- Confidence: high
- Three points. (a) `timer.start_timer("gp_train")` at `:669` is stopped only inside the `if options["active_sample_gp_update"]` branch (`:686`); the `else` branch at `:687-691` leaves it running, and because `Timer.start_timer` ignores a start for a key already present, the next `stop_timer("gp_train")` — in `vbmc.py:1481` — attributes everything since to GP training. This fires with `active_sample_vp_update=True, active_sample_gp_update=False`. (b) MATLAB's `initdesign_vbmc` accumulates `t_func` around each target call; PyVBMC's initial-design branch has no `fun_time` timer, so iteration 0's evaluation time is not attributed. (c) MATLAB finishes with `timer.activeSampling += toc - t_func - timer.gpTrain - timer.variationalFit`; PyVBMC's accumulators simply overlap, so `active_sampling` includes the function, GP and VP time that MATLAB subtracts out.
- Consequence if real: reported timings only. `t_algoperfuneval`, the one place MATLAB feeds timings back into the algorithm, is unported and dead (see Coverage), so nothing numerical depends on this.
- Suggested reproduction: none needed.
- Test adequacy: no.

### Minor observations (pointable, low consequence, not worth separate entries)
- `f_val_old = acq_fast[idx]` (`:490`) reuses the 8192-point batched value where MATLAB re-evaluates the acquisition pointwise (`activesample_vbmc.m:247`). Given `_sq_dist`'s batch-dependent centering (sheet, P2 and Settled non-differences), the two differ by ~2e-15 relative and can flip the `f_val_optim < f_val_old` comparison on a near-tie.
- Rounding: MATLAB's `round` is half-away-from-zero, Python's builtin `round` and `np.round` are half-to-even. This reaches `round(frac * N_random_points)` (`:871,882,891,899,944`), `np.round(np.linspace(...))` (`:912`), and `round(hpd_frac * N)` inside `get_hpd` (`pyvbmc/stats/get_hpd.py:35`) — e.g. `hpd_min = 0.1` with 25 live points gives 2 points in Python, 3 in MATLAB. Never triggers with the default fractions and `ns_search = 2^13`.
- `np.argsort(acq_fast)` (`:415`) is not stable by default, whereas MATLAB's `sort` is; ties are broken differently. Only in the `search_cache_frac > 0` path.
- `options["ns_ent_fine_active"](vp0.K)` (`:787`) and `options_update["ns_elbo"](vp.K)` (`:699`) call the option directly instead of going through `Options.eval`/`evaloption_vbmc`, so a scalar value for either option raises `TypeError` where MATLAB accepts it.
- `options["search_optimizer"] != "none"` (`:481`) is case-sensitive; MATLAB uses `strcmpi`.
- Draw ordering: MATLAB draws `optimState.acqrand` and runs `activeimportancesampling_vbmc` *before* `getSearchPoints`; PyVBMC drops the `acqrand` draw (dead in MATLAB) and runs importance sampling *after* `_get_search_points`. Per the review rules this is a stream difference, not a finding; the distributions and the set of draws are otherwise identical.
- `optim_state["gp_length_scale"]` is stored as shape `(D,)` where MATLAB stores a `1×D` row; every consumer (`abstract_acq_fcn.py:343`, `active_sample.py:346`) broadcasts correctly.
- The active-loop cache deletion (`:632-637`) does not delete the matching row of the Python-only `cache["skip_logger"]`, which would desynchronize the three arrays; unreachable while F3 keeps the cache empty.

---

## 3. Test adequacy notes

Tests that pin the implementation rather than the specification:

- `pyvbmc/testing/vbmc/test_vbmc_active_sample.py::test_get_search_points_all_cache` and `::test_get_search_points_search_bounds` use `cache_frac = 1` with a cache of exactly `number_of_points` rows. In that configuration `ceil(n*1) == x0.shape[0]`, so MATLAB's count and Python's coincide and F4 is invisible; the assertions then restate the implementation's own output shapes.
- `::test_active_sample_initial_sample_more_provided` asserts `np.all(np.isnan(optim_state["cache"]["x_orig"][:sample_count]))`. The implementation leaves that array empty, so the assertion is vacuously true. It encodes "the cache was emptied" as a non-check rather than testing what should remain (F3).
- `::test_get_search_points_all_search_cache` installs a non-empty `search_cache` before calling, bypassing the initial `[]` that the solver actually starts from (F7).
- `::test_active_uncertainty_sampling` substitutes Rosenbrock for the acquisition and asserts the CMA-ES search finds `(1,1)`. It is a real optimizer check, but on a near-isotropic objective, so it cannot detect the missing per-coordinate `insigma` scaling (F1); and nothing anywhere asserts the CMA-ES option set against MATLAB's `CMAESopts` (F1/F5/F6). Note also that this test has to set `optim_state["n_eff"]` by hand between the two `active_sample` calls — the symptom of F10.
- The `active_sample_step` oracle (`pyvbmc/testing/oracles/`) pins a full seeded CMA-ES search against PyVBMC's own recorded output. That is the right tool for detecting drift, but it necessarily ratifies whatever settings are currently passed to `cma.fmin`, so it cannot surface F1, F5 or F6.
- `::test_get_search_points_all_box_search` is a good counter-example and worth citing as the model: it checks the *distribution* of the box candidates (mean and quartiles of the unit-scaled draws) rather than re-deriving them from the implementation. That is the shape of test that would have caught the historical standard-normal-for-uniform error, and the shape the cache-count and search-cache tests lack.

---

## 4. Sheet notes

Entries I found incomplete or that the sheet lacks:

- **"The initial design does not cluster surplus starting points" is incomplete.** It describes only the selection rule (k-means vs "first `sample_count` rows"). It omits that PyVBMC also *deletes* the remaining `N0 - Ns` rows from the cache, while MATLAB keeps them with their cached `y_orig` for the search sieve and for evaluation-free reuse (F3). A reviewer reading the entry would conclude the cache difference is settled when the discard is not described anywhere.
- **"The `cma` package replaces `cmaes_modded.m`" does not cover the settings.** It records the substitution (`seed=np.nan`, the `randn` callback, the subclassed noise handler) but not three concrete setting differences that the instruction to "compare the settings passed" puts squarely in scope: the collapse of `insigma` to a scalar `sigma0` (F1), the enabling of noise handling where MATLAB's `Noise.on = 0` (F5), and the best-ever return convention with the inert `search_cmaes_best` option (F6). Also uncovered: `TolX` is relative to `max(insigma)` in MATLAB and absolute (`1e-11`) in `cma`.
- **Missing deliberate difference: `fun_eval_start`.** `np.maximum(D, 10)` vs `10*ceil((D+1)/10)` (F2) is nowhere in the sheet, in P1b or P2.
- **Missing: the `D == 1` Nelder-Mead branch** (F9) — a Python-only substitution of the search optimizer, unbounded and with a different evaluation budget.
- **Missing: `noise_shaping` is unported while the option is live** (F11). The sheet documents several unported features with dead options; this one has a partially live option.
- **Missing: `active_sample_fess_thresh` is declared and unread** (F14).
- **Missing: the `_selection_policy_callback` seam** (`active_sample.py:28`, `:354-362`, `:421-448`, `:619-624`) — a Python-only dormant hook with no MATLAB counterpart, comparable to the "MCMC branch of importance sampling is a retained but dormant hook" entry in P4, but absent from the sheet.
- **Missing: `VariationalInitRepo`/`vp_repo` is unported** (`misc/vpsieve_vbmc.m:60-66`). `pyvbmc/vbmc/README.md` records the decision, but the sheet has no entry, and the P2-side write is dead code (F15).
- **`counterpart_map.md` note is misattributed.** The `misc/initdesign_vbmc.m` row says "MATLAB commit `46b6f5e` (2021-05-26) set `Xs = []` for `N <= N0`". That commit touches `gplite/private/fminfill.m` and `utils/fminfill.m`, not `initdesign_vbmc.m`, which has not changed since 2019 (`1e17fb8`).
- **Confirmed correct:** the "uniform box sampler now matches MATLAB" and "the unused outer search limits are unused in MATLAB too" entries both hold as written — `:963` draws `rng.random`, and `activesample_vbmc.m:507-508` are commented out. The `_real2int`-snaps-in-place entry also holds, and I verified the consequence it describes is what `active_sample.py:438-444` compensates for.
