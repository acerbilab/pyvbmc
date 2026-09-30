# P4 — noisy importance sampling, MATLAB-comparison track

Raw reviewer report, wave 4 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, fresh
context, read-only on the repositories, reading the code at `f873556`; it
compared the Python code with MATLAB VBMC at `396d649`. Its brief carried a
first question on uncertainty level 1, which the report answers before its
findings. Its check scripts ran in the session scratchpad and are kept only
on the machine that ran them (`dev/scripts/runs/LOCAL.md`). The text below
is the agent's final message, unedited.

---

# P4 — noisy importance sampling (MATLAB comparison track)

## 1. Coverage

Read completely, both sides, line by line:

- `pyvbmc/vbmc/active_importance_sampling.py` (all 499 lines: `active_importance_sampling`, `active_sample_proposal_pdf`, `get_mcmc_opts`, `fess`, `renormalize_weights`).
- `../vbmc/private/activeimportancesampling_vbmc.m` (all 420 lines, including the local `activesample_proposalpdf`, `log_isbasefun`, `sq_dist`, `get_mcmcopts`, `catrnd`).
- `../vbmc/misc/fess_vbmc.m`.
- `pyvbmc/acquisition_functions/acq_fcn_viqr.py`, `acq_fcn_imiqr.py`, `abstract_acq_fcn.py` against `../vbmc/acq/acqviqr_vbmc.m` and `acq/acqimiqr_vbmc.m` (the `islogf1`/`islogf2`/`islogf` branches and the consumption of `Xa`, `lnw`, `Kax_mat`, `Ctmp_mat`, `fs2a`).
- The call sites: `pyvbmc/vbmc/active_sample.py` (the whole file) against `../vbmc/private/activesample_vbmc.m` (whole file), with attention to lines 326–351 / 159–174 and 724–815 / 415–490.
- `../vbmc/gplite/gplite_pred.m`, `gplite_noisefun.m`, `gplite_post.m`, `gplite/private/gplite_core.m`.
- gpyreg: `noise_functions.py` (all), `gaussian_process.py` `update` (763–1000), `predict` (1856–2072), `__core_computation` (2680–2845), `_convert_shapes`; `slice_sample.py` (all 944 lines).
- `pyvbmc/function_logger/function_logger.py` (all), `pyvbmc/vbmc/gaussian_process_train.py` (`train_gp` noise setup, `_get_training_data`, `reupdate_gp`), `pyvbmc/vbmc/vbmc.py:1090–1110`; `../vbmc/misc/setupvars_vbmc.m:278–283`, `misc/funlogger_vbmc.m`, `misc/get_traindata_vbmc.m`, `misc/gpreupdate.m`.
- Tests: `pyvbmc/testing/vbmc/test_active_importance_sampling.py` (all).
- Sheet: `dev/experiments/port_review_20260919/known_differences.md`, sections "Slice P4" (both entries), "Slice P3" headings, "Settled non-differences", the P2 entries on repeated observations and on the rank-one GP update, the GP-layer `SliceSampler` entry, the P1b entry on the omitted importance samples. No other file under `dev/` was opened.

Skimmed: `gplite/private/eissample_lite.m` — only what the call site depends on (the `(logPfuns, x0, N, W, widths, LB, UB, sampleopts)` signature, the `[Xa, logp]` outputs, the `Thin`/`Burnin`/`VarTransform`/`InversionSample`/`FitGMM` options, the walker semantics). I did not audit its transition operators. Not reached: `gpyreg`'s covariance gradient code, MATLAB's `vbmc_rnd`/`warpvars_vbmc` internals (only their signatures and flag order).

Checks run (all in the scratchpad, interpreter `…/pyvbmc/.venv/Scripts/python.exe -u`, BLAS single-threaded; `pyvbmc.__file__ = C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\__init__.py`, `gpyreg.__file__ = C:\Users\luigi\Documents\GitHub\gpyreg\gpyreg\__init__.py`):

1. `check1_isr_weights.py` — step 1 + the step-2 starting-point weights on the test-module scenario, PyVBMC's rule vs MATLAB's. **Result: PyVBMC's weights are exactly uniform (1/23 for every sample, both hyperparameter samples); MATLAB's are concentrated (max weight 0.72 / 0.80, ESS 1.76 / 1.47 out of 23).**
2. `check2_level1_noise.py` — the level-1 logger record and the noise function; rank-one update vs full recompute at level 1. **Result: `S^2 * n_evals == 1` identically (including after 2 and 3 repeats), `sn2 = exp(2h₀)+exp(h₁)·1` as MATLAB's `[1 2]`; rank-one and recompute agree to 2e-16.**
3. `check3_level2_rank1.py` — level 2 with heteroskedastic `s2` and a new point whose noise is below/inside/above the existing range. **Result: the rank-one update keeps the old factorization scale `sl` where a recompute picks a new one, yet α, the predictions and the slice's `C_tmp = (L\(Lᵀ\K_Xa_Xᵀ))/sn2_eff` agree with the recompute and with the exact `(K+Σ)⁻¹K_Xa_Xᵀ` to ~2e-15.**
4. `check4_misc.py` — `fess(vp, gp, 100)` and the only-VP `mcmc_importance_sampling` hook. **Result: both raise (`AttributeError: 'tuple' object has no attribute 'ndim'`; `ValueError: The initial point x0 needs to be a scalar or a 1D array`).**
5. `check5_chain_effect.py` — the IMIQR MCMC step at production sizes (200 proposal samples, 100 MCMC samples, thin 1, burn-in 50) under both selection rules, comparing the log-target at `x0` and over the chain. **Result: the starting points differ (log-target −3.013/−5.335 vs −3.038/−5.749) but the chains land in the same region; with a 50-iteration burn-in in D = 3 the effect on the recorded samples is small.**

I did not run any test suite, `VBMC.optimize()`, or the oracle generators.

## 2. First question — observation noise at uncertainty-handling level 1

**Setup, established from the sources, not from the description.** `misc/setupvars_vbmc.m:278–280` sets `gpNoisefun` to `[1 0]`, `[1 2]`, `[1 1]` for levels 0, 1, 2; `pyvbmc/vbmc/vbmc.py:1094–1102` sets `gp_noise_fun` to `[1,0,0]`, `[1,2,0]`, `[1,1,0]`, and `gaussian_process_train.py:97–106` maps those to `GaussianNoise(constant_add, user_provided_add, scale_user_provided)` — identical to `gplite_noisefun.m:177–194` (`[1 2]` → `exp(2h₀) + exp(h₁)·s2`; `[1 1]` → `exp(2h₀) + s2`). `misc/funlogger_vbmc.m:51,101–107` allocates `optimState.S` whenever the level is above 0 and records `fsd = 1` at level 1; `FunctionLogger.__init__:222` and `__call__:301–302` do the same. Repeats pool by precision on both sides (`funlogger_vbmc.m:235–238`, `function_logger.py:688–707`), so after k evaluations `S = 1/√k` and `n_evals = k` at level 1. `get_traindata_vbmc.m:8–12` and `_get_training_data:761–764` both pass the *pooled* `S²` as the GP's `s2`.

**(a) The noise at every training point (`active_sample.py:326–351` vs `activesample_vbmc.m:159–174`): matches at all three levels.**

| quantity | MATLAB | PyVBMC | verdict |
|---|---|---|---|
| hyperparameter slice | `hyp(gp.Ncov+1 : gp.Ncov+gp.Nnoise)` | `hyp[cov_N : cov_N+noise_N]`, `noise_N = gp.noise.hyperparameter_count()` (2 at level 1, 1 at levels 0 and 2) | same |
| `s2` | `(S(X_flag).^2).*nevals(X_flag)` when `optimState.S` exists, else `[]`→0 | `(S[X_flag]**2) * n_evals[X_flag]` when `hasattr(function_logger,"S")`, else `None`→0 | same; `(N,1)` on both sides |
| value at level 1 | `s2 ≡ 1`; `sn2 = exp(2h₀)+exp(h₁)` | `s2 ≡ 1` (check 2, including pooled rows); `sn2 = exp(2h₀)+exp(h₁)` | same |
| value at level 2 | `s2 = S²·nevals` (single-observation variance); `sn2 = exp(2h₀)+s2` | same | same |
| value at level 0 | `s2 = []` → 0; `sn2 = exp(2h₀)` scalar, broadcast into the column | `s2 = None` → 0; scalar `.reshape(-1,)` broadcast into `sn2new[:,s]` | same |
| average over samples | `gp.sn2new = mean(sn2new,2)` → `(N,1)` | `gp.temporary_data["sn2_new"] = sn2new.mean(1)` → `(N,)`, later `reshape(-1,1)` in `_estimate_observation_noise`/`y_s2` | same |

The multiplier `exp(h₁)` is therefore in exactly the place MATLAB puts it, and because `S² · n_evals == 1` at level 1 the quantity MATLAB intends — the variance of one *new* observation, not of the pooled row — is what both sides hand to the acquisitions. Noise shaping is the only unported piece of this block (sheet).

**(b) The GP update after an acquisition (`active_sample.py:724–815` vs `activesample_vbmc.m:415–490`): `s2new` is MATLAB's at every level, but the *path* differs at levels 1 and 2, and the sheet describes MATLAB wrongly here (F6).**

- `s2new`: MATLAB `s2new = optimState.S(idx_new)^2` when `optimState.S` exists, else `[]`; PyVBMC `s2new = function_logger.S[idx_new]**2` when the logger has `S`, else `None`. Identical at all three levels (level 1: `1`; level 2: the caller's SD squared; level 0: absent). `_convert_shapes` reshapes `(1,)` to `(1,1)`.
- The path. PyVBMC calls `gp.update(xnew, ynew, s2_new=s2new, compute_posterior=True)`; gpyreg takes the rank-one branch whenever one row is added and `hyp is None` (`gaussian_process.py:815–824`), *including* with `s2_new`. MATLAB's `gplite_post.m:76–79` sets `update1 = false` as soon as `s2` is non-empty ("Rank-1 update is not supported with heteroskedastic noise") and falls through to appending `X`, `y`, `s2` and recomputing every posterior with `gplite_core`. So at levels 1 and 2 MATLAB never rank-one updates; PyVBMC always does.
- Does gpyreg apply the noise function to `s2new` as `gplite_post` would? Yes, and correctly: `gaussian_process.py:849–857` evaluates `self.noise.compute(hyp_noise, X_new, y_new, s2_new)` — i.e. `exp(2h₀)+exp(h₁)·s2new` at level 1, `exp(2h₀)+s2new` at level 2 — then `sn2_eff = sn2 · sn2_mult`. Its Cholesky extension is the *generalization* of MATLAB's: MATLAB writes the new column as `(Lᵀ\k*)/sn2_eff` and the new diagonal as `√(1 + K/sn2_eff − uᵀu)` , which is correct only because it assumes the factor's scale equals the new point's `sn2_eff` (true only for homoskedastic noise — hence its refusal); gpyreg keeps the factor's own scale `sl` and writes `u/sl` and `√(sl·sn2_eff + sl·K − uᵀu)/sl`, which reduces to MATLAB's when `sl == sn2_eff`. It appends `1/√sl` to `sW` (MATLAB appends `1/√sn2_eff`), which keeps `sW` the constant vector `gplite_core.m:281` produces and keeps `1/sW[0]²` equal to the factorization scale that `active_importance_sampling.py:299` and `acq_fcn_imiqr.py:105` read as `sn2_eff`. Check 3 confirms rank-one and recompute agree to ~1e-15 in α, `L`, the predictions and the slice's `C_tmp`, even when the recompute would have chosen a different `sl`. **So (b) has no numerical defect; it is a real path difference, mis-stated on the sheet (F6).** `gp.s2` is kept aligned on both sides (`gaussian_process.py:963–972`; `gplite_post.m:89`).

**(c) Inside the slice: exactly one quantity depends on observation noise, and it is MATLAB's at every level.**

- `gp.predict(..., separate_samples=True)` at `active_importance_sampling.py:83, 120, 237, 270` and inside `active_sample_proposal_pdf:367` is `add_noise=False` — the *latent* `f_mu, f_s2`, matching MATLAB's `[~,~,fmu,fs2] = gplite_pred(gp,Xa,[],[],1,0)` (the 5th/6th arguments of `gplite_pred` are `ssflag`/`nowarpflag`, not an add-noise flag; the 3rd and 4th outputs are latent). So the importance weights, `is_log_base`, `f_s2`, `K_Xa_X` and `C_tmp` carry no observation noise on either side.
- The one noise-dependent call is the MCMC target: `is_log_full` falls back to `gp.predict(x, add_noise=True)` (`acq_fcn_imiqr.py:264`, `acq_fcn_viqr.py:526`), and MATLAB's `log_isbasefun` calls `[fmu,fs2] = gplite_pred(gp,x)`, which takes the *first two* outputs `ymu, ys2` — also noise-inclusive. gpyreg's `predict` (`:2024–2032`) computes `sn2_star = noise.compute(hyp_noise, x_star, y_star, s2_star)` with `s2_star = None` and adds `sn2_star·sn2_mult`, byte-for-byte the structure of `gplite_pred.m:62–63,121`. With no `s2_star` at a test point the noise function's second term vanishes on both sides, so at level 1 the MCMC target uses only `exp(2h₀)` — the inferred per-observation variance `exp(h₁)` is *not* included — and at level 2 likewise. This is a shared modelling quirk (MATLAB has it too), not a port discrepancy, but it is worth knowing: the IMIQR proposal's `sinh(u·s)` term is evaluated at a variance far below that of an actual new observation at a level-1/2 point.
- `gp.temporary_data["sn2_new"]` from part (a) is consumed by the acquisitions' `_estimate_observation_noise`, not inside the slice.

**Answer in one line: at level 1 the importance sampling and the GP updates treat the observation noise exactly as MATLAB's do — the one behavioural difference (rank-one instead of recompute) is numerically equivalent — and nothing in the slice is wrong because of the noise model. The defect I found in the slice (F1) is independent of the noise level.**

## 3. Findings

### F1. The MCMC step's importance-resampling weights collapse to uniform: the max is taken over a length-1 axis
- Location: `pyvbmc/vbmc/active_importance_sampling.py:240-251`; MATLAB: `private/activeimportancesampling_vbmc.m:206-214`
- Category: indexing/shape (with consequences for random draws)
- Proposed classification: port discrepancy
- Confidence: high
- History: the MATLAB lines are unchanged since before 2021-01-19 (`git -C ../vbmc log -L206,214:private/activeimportancesampling_vbmc.m` shows no post-2021 commit; `68a197b` and `a9615ba` touch other regions). The Python lines have never matched: the shape and the `axis=1` date from the first port of this file, `70325a34` "Noisy likelihoods, small fixes (#80)"; `c4035ff8` only changed `replace=True`→`False`, `d34fc154` the `assert`→`raise`, `d02c517e` `np.random.choice`→`rng.choice`.
- What the code does, what it should do, and why. MATLAB builds the per-sample importance weights as a **row**: `lnw = ActiveImportanceSampling_old.lnw(s,:) + acqfun('islogf2',...)'` is `(1, Na)`, so `max(lnw,[],2)` is the maximum **over the Na importance samples**, a scalar, and `w = exp(lnw - max)` is a genuine categorical distribution over the samples from which `catrnd` draws the walkers' starting points. PyVBMC builds the same quantity as a **column**: `active_is_old["ln_weights"][s, :].reshape(-1, 1)` is `(Na, 1)` and `acq_fcn.is_log_added(...)` is `(Na, 1)` because `gp1` holds one posterior, so `np.amax(ln_weights, axis=1)` maximizes each row against itself. `ln_weights - ln_weights_max` is then identically zero, `weights` is all-ones, and after `weights / np.sum(weights)` the `rng.choice` at `:250` is **uniform over the proposal samples**. The importance-sampling-resampling step of §"Step 2" is therefore inert: the MCMC chain starts at an arbitrary one of the 200 proposal points instead of one drawn in proportion to `exp(lnw_old + log 2sinh(u·s))`. A secondary consequence: MATLAB has no `-inf` guard here at all, while `:244-245` now raises `ValueError("Invalid value.")` if **any single** sample has an infinite weight, where MATLAB's scalar maximum would only be `-inf` if *every* sample were.
- Consequence if real: only the IMIQR path reaches this code (VIQR takes the `only_vp_flag` branch), and only when `active_importance_sampling_mcmc_samples > 0` — the default, 100. It fires at every active-sampling step of every noisy IMIQR run, once per GP hyperparameter sample. The effect is on the chain's starting point, not on a formula, so with the default burn-in `ceil(thin·Nmcmc/2) = 50` a well-mixing chain recovers: check 5 in D = 3 shows the recorded samples land in the same region (mean log-target −4.03/−6.10 vs −4.55/−5.71). It matters where the chain mixes slowly — higher D, a multimodal GP surrogate, or a user lowering `active_importance_sampling_mcmc_samples` (the burn-in scales with it) — because a cold start in a region the base IS density gives weight ~1e-30 (check 1 shows weights down to 1e-107) then dominates the 100 recorded importance points that the IMIQR acquisition integrates against.
- Suggested reproduction: `scratchpad/wave4_P4_comparison/check1_isr_weights.py` (run; output above): PyVBMC's weights are exactly `1/23` while MATLAB's semantics give max weight 0.72/0.80 and ESS 1.76/1.47. `check5_chain_effect.py` measures the downstream effect. The smallest possible check is a one-line assertion that `ln_weights.shape[1] == 1` at `:243`.
- Test adequacy: no. `test_active_importance_sampling` exercises this branch but asserts only shapes `(2, 10)` and `(2, 10, D)`. `test_draws_come_from_vp_rng_only` asserts reproducibility, not values. The MATLAB-derived fixtures cover `active_sample_proposal_pdf`, `fess` and `log_isbasefun` — step 1 and the MCMC *target* — but nothing covers step 2's selection. The `acq_AcqFcnIMIQR` oracle pins the current behaviour (re-baselined 2026-09-11) and would flag a fix, by design.

### F2. `active_is["ln_weights"]` is renormalized at the end; MATLAB leaves it unnormalized
- Location: `pyvbmc/vbmc/active_importance_sampling.py:329` and `:497-499` (`renormalize_weights`); MATLAB: no counterpart (`grep lnw` over `../vbmc/**.m` finds the weights only in `activeimportancesampling_vbmc.m`, `acqimiqr_vbmc.m` and `acqviqr_vbmc.m`, none of which normalizes)
- Category: formula
- Proposed classification: possibly intentional
- Confidence: high that it differs; medium that it is deliberate
- History: MATLAB never had it. Present in Python since the first port (`70325a34`).
- What the code does: `renormalize_weights` subtracts the global log-sum-exp over the **whole** `(Ns_gp, Na)` array, so every entry of `ln_weights` is shifted by one scalar. MATLAB uses `lny' - logp'` (MCMC branch) or `lny' - lpdf` (ISR branch) as they come.
- Consequence if real: the shift is a constant per call and `AcqFcnIMIQR._compute_acquisition_function` is log-linear in it (`acq[:,s] = logsumexp(ln_weights + …)` then a log-mean-exp over `s`), so the **ranking of candidates is unchanged**, the local optimizer's `tol_fun` is the fixed `1e-2` of the `log_flag` branch on both sides, and the variance-regularization term is additive on both sides. What differs is the *value* of `optim_state["active_importance_sampling"]["ln_weights"]` and of the IMIQR acquisition (by `log Σ exp(lnw_MATLAB)`), which anyone comparing against MATLAB numbers, or interpreting the acquisition as the log integrated IQR, will see. It is also the reason the `iqr_reduction` VIQR loss carries the documented constant `log N_s`.
- Suggested reproduction: not run; it is a one-scalar shift, readable from the code. A check would compute IMIQR's acquisition on a fixed candidate set with and without line 329 and confirm the difference is constant.
- Test adequacy: no. The MATLAB fixture `activesample_proposalpdf.npz` pins `active_sample_proposal_pdf`'s output *before* renormalization; nothing pins `active_importance_sampling`'s returned weights against MATLAB. Not on the known-differences sheet.

### F3. `fess(vp, gp, N)` with a scalar third argument raises
- Location: `pyvbmc/vbmc/active_importance_sampling.py:470-474` (`X = vp.sample(N, orig_flag=False)` discards the tuple unpacking); MATLAB: `misc/fess_vbmc.m:4-12`
- Category: cross-module
- Proposed classification: port discrepancy (latent, currently unreachable)
- Confidence: high
- History: MATLAB's lines are unchanged since before 2021. The Python has never worked: `vp.sample` has returned `(X, I)` for as long as the snake_case API has existed.
- What the code does: MATLAB's `fess_vbmc` accepts either a matrix of points or a sample count, defaulting to 100; PyVBMC's docstring advertises the same (`X : np.ndarray(N, D) or int`) and `X=100` is the default. `vp.sample` returns a 2-tuple, so `X` becomes a tuple and the next line (`gp.predict(X)` or `X.shape[0]`) raises `AttributeError: 'tuple' object has no attribute 'ndim'`. Only `X = vp.sample(N, orig_flag=False)[0]` would do.
- Consequence if real: no production path hits it today. The slice's own call is `fess(vp, f_mu, Xa)` (arrays). MATLAB's scalar caller is `activesample_vbmc.m:440` (`fESS = fess_vbmc(vp,gptmp,100)`, the `ActiveSamplefESSThresh` gate), and PyVBMC stubs that out at `active_sample.py:736-737` (`gptmp = None; fESS, fESS_thresh = 0, 1`). So this is a trap waiting for whoever ports that gate, plus a documented public behaviour that does not work.
- Suggested reproduction: `scratchpad/wave4_P4_comparison/check4_misc.py` (run): `fess(vp, gp, 100) raised: AttributeError 'tuple' object has no attribute 'ndim'`.
- Test adequacy: no. `test_fess` calls `fess(vp, gp_means, X)` and `fess(vp, gp, Xa)` — both array paths — and checks them against `compare_MATLAB/fess.npz`. The scalar path is untested.

### F4. The retained `mcmc_importance_sampling` hook cannot be used: `SliceSampler` rejects the ensemble of starting points
- Location: `pyvbmc/vbmc/active_importance_sampling.py:86-120` (`sampler = gpr.slice_sample.SliceSampler(log_p_fun, Xa, …)` with `Xa` of shape `(Na, D)`); MATLAB: `private/activeimportancesampling_vbmc.m:81` (`eissample_lite(logPfuns, Xa, Nmcmc_samples, W, …)` with `W = Na` walkers)
- Category: control flow / cross-module
- Proposed classification: port discrepancy — the sheet's justification does not hold
- Confidence: high
- History: MATLAB unchanged since before 2021 (`68a197b`, 2022-06-25, touches `activesample_vbmc.m`, not this). Python has passed a 2-D `Xa` since the first port.
- What the code does: MATLAB's ensemble sampler takes a `(W, D)` matrix of walker positions and advances all of them one step; `gpyreg.SliceSampler.__init__:145,194-197` sets `D = x0.size` and raises `ValueError("The initial point x0 needs to be a scalar or a 1D array")` for any 2-D `x0`. The branch aborts before sampling. The sheet's entry "The MCMC branch of importance sampling is a retained but dormant hook" says it is "kept so that a user-supplied acquisition object can still request it" — requesting it raises.
- Consequence if real: none for shipped acquisitions (neither `AcqFcnVIQR` nor `AcqFcnIMIQR` sets `mcmc_importance_sampling`), but the stated purpose of the retained hook is not met, and a user-supplied acquisition that sets the flag fails immediately rather than degrading.
- Suggested reproduction: `check4_misc.py` (run): setting `acq.acq_info["mcmc_importance_sampling"] = True` on a VIQR instance gives `ValueError: The initial point x0 needs to be a scalar or a 1D array`.
- Test adequacy: no test sets the flag.

### F5. `active_sample_proposal_pdf` raises where MATLAB degrades to `-inf`
- Location: `pyvbmc/vbmc/active_importance_sampling.py:396-398`; MATLAB: `private/activeimportancesampling_vbmc.m:335-337` (no check) together with `:148`
- Category: control flow
- Proposed classification: possibly intentional
- Confidence: medium that it differs (high — the MATLAB check simply does not exist); low that it can trigger in practice
- History: MATLAB unchanged; Python has had the guard (as an `assert` until `d34fc154`, as a `ValueError` since) from the first port.
- What the code does: when every mixture component of the proposal gives density zero at a point, MATLAB computes `mmax = -Inf`, `templpdf - mmax` yields `NaN`, `lnw` becomes `NaN`, and the caller's `ActiveImportanceSampling.lnw(~isfinite(...)) = -Inf` (`:148`, ported at `active_importance_sampling.py:201-203`) turns it into `-Inf`: the point survives with zero weight. PyVBMC instead raises `ValueError("Invalid value.")` and the whole `optimize()` call fails. Both sides compute `vp.pdf(..., log_flag=True)` as `log(Σ …)` (`variational_posterior.py:905-911`; `vbmc_pdf.m:107-110`), so both can underflow to `-inf` in principle.
- Consequence if real: an aborted run instead of one dropped importance point. It needs a variational-posterior sample whose smoothed density underflows float64 *and* that lies in no training-point box; with the `[0.05, 0.2, 1.0]` smoothing this requires very large `sigma·lambd`, so it is remote. Reported for completeness because the same author wrote the analogous guard at `:244-245`, where F1 makes it far more reachable.
- Suggested reproduction: not run. Call `active_sample_proposal_pdf` with a `vp_is` whose `sigma` is ~1e16 and `rect_delta` tiny, so the VP log density underflows and no box covers the point.
- Test adequacy: `test_active_sample_proposal_pdf` uses well-conditioned inputs and would not hit it.

### F6. The sheet's rank-one entry misdescribes MATLAB on noisy targets
- Location: sheet §"Slice P2" → "The rank-one GP update is taken for a fresh observation, noisy or not"; code: `pyvbmc/vbmc/active_sample.py:803-811`; MATLAB: `private/activesample_vbmc.m:481-487` **and** `gplite/gplite_post.m:76-79`
- Category: state/caching
- Proposed classification: possibly intentional (the code), documentation defect (the entry)
- Confidence: high
- History: `gplite_post.m:76-79` is unchanged since before 2021. The Python condition is from `510a493` (2026-09-19).
- What the code does: the entry states "On a noisy target the observation's noise variance goes into the rank-one update on both sides." It does not on MATLAB's side: `gplite_post.m:76-79` sets `update1 = false` whenever `s2` is non-empty, so at uncertainty levels 1 and 2 — every noisy target — MATLAB appends the row and recomputes all posteriors with `gplite_core`, exactly as `gpreupdate` would. `activesample_vbmc.m:481` passing `update1 = 1` is overridden inside `gplite_post`. PyVBMC always takes gpyreg's rank-one branch. The entry's two-cell comparison ("The two conditions part in one cell, a repeated input on a noiseless target") therefore misses the larger cell: every first observation on a noisy target.
- Consequence if real: for the algorithm, none I could measure — check 3 shows the rank-one result matches the recompute to ~1e-15 in α, `L`, predictions and the slice's `C_tmp`, at level 1 (where the effective noise is homoskedastic anyway, `s2 ≡ 1`) and at level 2 including the case where the new point's noise is below every existing one and the recompute picks a different factorization scale `sl`. gpyreg's formula is the correct heteroskedastic generalization of MATLAB's, which is why MATLAB refuses it. The consequences are a per-step cost difference (O(N²) vs O(N³)) and a difference in `sn2_mult`/`sl` after a Cholesky retry, and the entry needs its MATLAB half corrected.
- Suggested reproduction: `scratchpad/wave4_P4_comparison/check3_level2_rank1.py` (run; output above) and `check2_level1_noise.py`.
- Test adequacy: no test compares the two paths; the oracles pin whichever path the code takes.

### F7. The importance samples and the search set are drawn in the opposite order
- Location: `pyvbmc/vbmc/active_sample.py:379-381` (`_get_search_points`) then `:414-417` (`active_importance_sampling`); MATLAB: `private/activesample_vbmc.m:208-211` (importance sampling) then `:215-218` (`acqrand`, `getSearchPoints`)
- Category: random draws
- Proposed classification: possibly intentional
- Confidence: high that the order differs; high that nothing but the stream depends on it
- History: MATLAB unchanged. PyVBMC has had the call after the search set since `c4035ff8` "Active sample refactor (#81)".
- What the code does: MATLAB prepares `optimState.ActiveImportanceSampling` before it generates the `ns_search` candidates; PyVBMC generates the candidates (and appends the repeat candidates) first. Neither routine mutates the GP or the VP — `active_importance_sampling` deep-copies both the VP it smooths and the GP it splits per hyperparameter sample — so the only effect is that the two routines consume `vp.rng` in the opposite order.
- Consequence if real: no formula changes; a seeded PyVBMC run and a seeded MATLAB run would diverge here even if every other draw matched. Since the generators differ anyway (sheet, P1b), this is bookkeeping, but it is a real ordering difference in a file the review compares step by step, and it also moves `compute_var_log_joint` (`active_sample.py:422-425`, MATLAB `:153-157`) from before the noise evaluation to after the importance sampling — harmless, `_gp_log_joint` is deterministic.
- Suggested reproduction: read only.
- Test adequacy: not applicable; no test could distinguish orderings of independent draws.

### F8. The sheet says the experimental VIQR losses were removed; `iqr_reduction` is still in the code at HEAD
- Location: `pyvbmc/acquisition_functions/acq_fcn_viqr.py:120` (`LOSSES = ("iqr", "iqr_reduction")`), `:364-370`, `:396-440`; sheet §"Slice P3" → "The EIG acquisition and the experimental VIQR losses were removed"
- Category: cross-module (documentation)
- Proposed classification: unsure — either the removal is incomplete or the entry is stale
- Confidence: high on the fact, low on the disposition
- History: not applicable (Python-only feature; MATLAB has no counterpart).
- What the code does: the entry states that "`AcqFcnVIQR` offers only the standard `loss='iqr'`" and that the `iqr_reduction` variant "is retained on the branch `retain/experimental-acquisitions`", with the removal taking effect 2026-09-14. `git show HEAD:pyvbmc/acquisition_functions/acq_fcn_viqr.py` on `dev-port-review` at `f8735567` still declares, documents and implements `loss="iqr_reduction"`. I flag it because this loss is the only consumer of `active_is["ln_weights"]` on the VIQR side and therefore the only VIQR path affected by F2's renormalization; it belongs to the P3 slice otherwise.
- Consequence if real: none numerically (the default is `"iqr"`); the sheet cannot be trusted as the inventory of what `AcqFcnVIQR` offers.
- Suggested reproduction: `git show HEAD:pyvbmc/acquisition_functions/acq_fcn_viqr.py | grep LOSSES` (run).
- Test adequacy: not applicable.

## 4. Test adequacy notes

- `test_active_importance_sampling` is the only test of `active_importance_sampling` itself and asserts nothing but array shapes. It exercises the IMIQR MCMC branch with `Nmcmc = 10`, `thin = 2`, so it runs straight through the degenerate weighting of F1 and passes. This is the "unit test mirrors the implementation" pattern the review exists to find: had it asserted anything about the *distribution* the starting point is drawn from, or compared one value with MATLAB, F1 would have shown in 2021.
- `test_draws_come_from_vp_rng_only` pins reproducibility and the isolation of the global stream, not correctness. It is a good test of what it tests, and it is the only guard over the MCMC branch's values.
- `test_active_sample_proposal_pdf`, `test_fess` and `test_acq_log_f` are the three real specifications in this slice: each checks PyVBMC's output against stored MATLAB arrays (`compare_MATLAB/activesample_proposalpdf.npz`, `fess.npz`, `log_isbasefun.npz`). They cover step 1's proposal density and weights, `fess`'s array path, and the MCMC target density — and they are why I found nothing wrong in those three places. Their gaps are (i) `fess` with a scalar count (F3), (ii) `active_importance_sampling`'s returned `ln_weights`, which the renormalization of F2 shifts away from MATLAB's, and (iii) everything in step 2.
- `test_acq_log_f` has a quirk worth noting: it constructs a VIQR instance, sets `acq_info["importance_sampling_vp"] = True`, computes `y_viqr`, then *discards* it (lines 352-358 recompute `viqr = AcqFcnVIQR()` and `y_viqr = viqr.is_log_full(...)`), and adds the VP log density by hand at `:360-362`. The comparison against MATLAB is therefore of `is_log_added + vlnpdf` assembled in the test, not of anything `AcqFcnVIQR` would produce on that flag — PyVBMC's `AcqFcnVIQR.is_log_full` never adds the VP density, unlike `acqviqr_vbmc.m:28`. The flag is off for both shipped acquisitions, so this is a latent gap rather than a defect.
- The `acq_AcqFcnIMIQR` and `acq_AcqFcnVIQR` oracles pin the current numerics of the consumers stage by stage. They pin F1's behaviour; a fix moves them, which is the sanctioned re-baseline path, not evidence against the fix.

## MATLAB-side defects noticed (no PyVBMC counterpart)

1. `acq/acqviqr_vbmc.m:26-28`: the `'islogf'` branch returns `vp + u*fs + log1p(...)`, where `vp` is the `vlnpdf` argument. VIQR sets `importance_sampling_vp = false`, so `log_isbasefun` calls it with `[]` and MATLAB returns `[]` (empty), not a log density. The branch is unreachable for VIQR only because `variational_importance_sampling = true` sends it down the `onlyvp_flag` path and no built-in sets `mcmc_importance_sampling`. PyVBMC's `AcqFcnVIQR.is_log_full` is correct on this point.
2. `gplite_post.m:76-79` silently downgrades a requested rank-one update to a full recompute and returns no indication; `activesample_vbmc.m:481-484` still executes `gp.t(end+1) = tnew` as though the cheap path had been taken. Harmless, but it is why the MATLAB code reads as if it rank-one updates noisy observations when it never does (see F6).
3. `private/activeimportancesampling_vbmc.m:208`: `w = exp(bsxfun(@minus,lnw,max(lnw,[],2)))` is never renormalized before `catrnd`, which is fine (`catrnd` scales by `cdf(end)`), but the subsequent `w(idx) = 0` inside the walker loop can zero *every* surviving weight when `W` approaches the number of non-negligible samples, at which point `catrnd` divides by zero and returns index 1 for the remaining walkers. With `W = 2(D+1)` and 200 samples this needs a very peaked weight vector; check 1 shows ESS as low as 1.47, so it is not far-fetched in higher dimensions.
4. `misc/fess_vbmc.m:21-22` compares `size(fbar,1)` with `size(X,1)` *after* possibly generating `X` from a scalar, so the mismatch error can never fire on the scalar path. Cosmetic; PyVBMC copies the check faithfully.
