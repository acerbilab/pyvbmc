# P6 — variational optimization, the ELBO and `update_K`, MATLAB-comparison track

Raw reviewer report, wave 1 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-19. One Opus agent,
fresh context, read-only on the repositories; its check scripts ran in the
session scratchpad and are not retained. The text below is the agent's
final message, unedited.

---

## 1. Coverage

Everything in the slice was read in full on both sides.

| MATLAB file | Python counterpart(s) | Both read completely |
|---|---|---|
| `private/updateK.m` | `variational_optimization.py: update_K` | yes |
| `misc/vpoptimize_vbmc.m` (incl. its nested `eval_fullelcbo`) | `optimize_vp`, `_initialize_full_elcbo`, `_eval_full_elcbo` | yes |
| `misc/vpsieve_vbmc.m` | `_sieve` | yes |
| `misc/negelcbo_vbmc.m` | `_neg_elcbo` | yes |
| `misc/gplogjoint.m` | `_gp_log_joint` | yes |
| `misc/gplogjoint_weights.m` | none (unported; confirmed) | yes |
| `misc/vpbndloss.m` | `_vp_bound_loss` | yes |
| `utils/softbndloss.m` | `_soft_bound_loss` | yes |
| `misc/vbinit_vbmc.m` | `_vb_init`, `_candidate_vp` | yes |
| `utils/fminadam.m` | `minimize_adam.py` | yes |
| `misc/vpoptimizeweights_vbmc.m` | none (unported; confirmed) | yes |
| `misc/vpbounds.m`, `misc/get_vptheta.m`, `misc/rescale_params.m` | `VariationalPosterior.get_bounds` / `get_parameters` / `set_parameters` (read as far as this slice calls them) | yes |
| `misc/vpsample_vbmc.m`, `utils/slicesample_vbmc.m`, `utils/malasample_vbmc.m` | confirmed absent from PyVBMC | read enough to confirm |
| call sites: `vbmc.m:683-725`, `vbmc.m:575-590`, `misc/finalboost_vbmc.m`, `private/activesample_vbmc.m:110-145,455-470` | `vbmc.py:1312-1340`, `:1492-1530`, `:2470-2520`, `active_sample.py:692-722` | yes |
| option defaults: `vbmc.m:160-370` (every option this slice reads) | `option_configs/*.ini` | yes |

**Unreachable MATLAB noted, not reported:** `vpoptimizeweights_vbmc.m` (its only call site, `vbmc.m:717`, is commented out) and therefore `gplogjoint_weights.m` and `negelcbo_vbmc.m`'s `onlyweights_flag` branches; `compute_var == 2` (the diagonal variance) and the whole `compute_vargrad` machinery in `gplogjoint.m` (`compute_var` is always the logical `ELCBOWeight ~= 0`); the CMA-ES stochastic-optimizer branch (`StochasticOptimizer = 'adam'`); the `vp_repo` block of `vpsieve_vbmc.m` (`VariationalInitRepo = 'no'`); all mean functions other than `negquad`/`zero`; the `altent_flag`/`entropy_alpha` arguments (MATLAB itself comments them "unused, kept for retrocompatibility").

**MATLAB history checked.** Since 2021-01-19 the slice's MATLAB files changed only in `2044530` (2021-06-18, deletion of the integrated-mean-function and `GPStochasticStepsize` code — nothing PyVBMC ports) and `c387612` (2022-10-26, which moved `add_jitter = true` *inside* the `vbinit_vbmc.m` loop). PyVBMC already has `add_jitter` inside the loop, i.e. it matches the *post-port* MATLAB fix — not a finding. `vpsieve_vbmc.m`, `updateK.m`, `vpbndloss.m`, `softbndloss.m`, `vpbounds.m`, `rescale_params.m`, `get_vptheta.m` are unchanged since long before the port.

**Checks run** (scripts in the session scratchpad `port_review/P6_comparison/`; no test suites, no `optimize()` runs):

1. **`_gp_log_joint` vs a line-by-line loop transcription of `gplogjoint.m`** (`ref_gplogjoint.py`, `cmp2.py`), on all 8 oracle fixture states (D=2..5, K=1..50, Ns=1..10, Cholesky and non-Cholesky branches). Max absolute differences over `G`, `dG`, `varG`, `var_ss`, `I_sk`, `J_sjk`, with and without the log/softmax Jacobians: all at rounding level (e.g. `G` ≤ 4.3e-10 on values ~10; `varG` ≤ 3.4e-12 on values ~1e-6; `J_sjk` ≤ 1.5e-10 on values ~3.6e-5). The larger relative figures on `varG`/`J_sjk` are cancellation in `J_jk = nf_jk − z_k'K⁻¹z_j`, inherent to the formula and present in MATLAB too.
2. **All 15 non-empty `grad_flags` combinations** (`cmp3.py`): gradient block order, packing (`mu` with *d* fastest inside *k*, i.e. MATLAB's column-major `reshape([D*K,Ns])`), block lengths and values all agree with the transcription.
3. **`_vp_bound_loss`/`_soft_bound_loss` vs a transcription of `vpbndloss.m` + `softbndloss.m`** (`cmp3.py`): **bit-identical** (`dL` max diff 0.0) when `optimize_weights=False`; with `optimize_weights=True` only the eta block differs, exactly as the known-differences sheet describes. The `order="F"` packing/unpacking of `lnscale` matches MATLAB's `reshape(...,[D,K])`.
4. **`np.polyfit(..., cov=True)` vs MATLAB `polyfit`'s `(R⁻¹R⁻ᵀ)·normr²/df`** (`check1.py`): identical to 1e-16 with the installed NumPy (2.5.2). NumPy's old `len(x)-order-2` "Bayesian" scaling was removed in 1.15; `minimize_adam`'s `slope_err` therefore matches `fminadam`'s.
5. **`sn2_eff = 1/sW[0]²`**: checked against gpyreg's and gplite's posterior parametrization. Both set `sW = ones(N,1)/sqrt(min(sn2)*sn2_mult)` (gpyreg `gaussian_process.py:2835-2841`, `gplite/private/gplite_core.m:281`), so `sW` is constant by construction and `(L\(L'\z))/sn2_eff` is exact even for heteroskedastic noise. **Not** a defect in either — recorded so the absence of a finding is informative.
6. **`_vb_init` type 2** on the real fixtures (`cmp4.py`, quantified in a follow-up run) — see F1.
7. **`vp.bounds` lifetime through `optimize_vp`** (`cmp4.py`, `cmp6.py`, `cmp7.py`) — see F2.

**Verified equivalent, no finding** (stated explicitly so the absence is informative): `update_K`'s window arithmetic (`recent_iters`, `lower_end`, the `-inf` masking of the first two post-warmup entries, the `iter > 0` ↔ `iter > 1` 0/1-based guard, the `pruned`/`r_index` last-entry reads, the half-window bonus test, `max(vp_K, min(K_new, K_max))`); all option defaults this slice reads (`ns_elbo`, `ns_elbo_incr`, `elbo_starts`, `ns_ent`, `ns_ent_fast`, `ns_ent_fine`, `k_fun_max`, `k_warmup`, `adaptive_k`, `hpd_frac`, `elcbo_impro_weight`, `tol_length`, `tol_con_loss`, `tol_weight`, `weight_penalty`, `pruning_threshold_multiplier`, `tol_improvement`, `sgd_step_size`, `tol_fun_stochastic`, `max_iter_stochastic`, `det_entropy_tol_opt`, `entropy_switch`, `det_entropy_min_d`, `elcbo_midpoint`, `always_refit_vp`, `stochastic_optimizer`, `tol_stable_count`, `fun_evals_per_iter` — all identical to `vbmc.m`); the Adam schedule, bias correction, minibatch-end test (`math.remainder` ≡ `mod`), `dx` random-walk windows, both termination clauses, the returned batch-mean iterate and `iter` count, and the `ftab`/`xtab` off-by-one that both sides share; `optimize_vp`'s candidate-type selection (`1`/`2`/`(i mod 3)+1`), the `i_mid`/`i_end` slot arithmetic, the Adam master step-size rules including the `warmup or not optimize_weights` branch, the `elcbo_midpoint` recomputation and `argmin(f_val_lst)` pairing; the pruning loop (uniform choice among sub-threshold unchecked components, `already_checked` deletion-vs-marking, the `|Δ ELCBO| < TolImprovement·(1/√K)` acceptance test evaluated at the *original* `K`, `I_sk` pruning); `_neg_elcbo`'s assembly (`F = −G − H`, `varF = varG + 0`, the `beta` guard, the capped small-weight penalty `Σ(w·1[w<θ] + θ·1[w≥θ])·WeightPenalty` and its softmax-Jacobian gradient, the `compute_grad ∧ β≠0 ∧ compute_var≠2` error, the `separate_K ∧ compute_grad` error); `_sieve`'s candidate split `⌈N/3⌉, ⌈N/3⌉, N−2⌈N/3⌉`, the entropy-switch/`K==1` deterministic override, the `ns_ent_K = ⌈NSent(K)/K⌉` rounding, and the ordering by `nelbo + β√varF`; `_vb_init`'s random draws — *what* is drawn and *in which order* is identical in every branch (`randi` → `integers`, `randn(D,1)`, `exp(0.2·randn())`, `0.25+0.25·rand()`, and the four jitter draws), only the fill order of the 2-D `randn` differs (stream, not distribution); the fact that `set_parameters` renormalizes `lambda`/`sigma` inside the objective while MATLAB's `negelcbo_vbmc` does not — the log joint, the entropies and all four gradient blocks are invariant under `σ→σn, λ→λ/n`, and the bound loss reads raw `theta` on both sides, so this is neutral (verified analytically and consistent with check 1, whose states came through `set_parameters`).

**Not reached:** the entropies themselves (`entlb_vbmc`, `entmc_vbmc`) — P7's slice; I checked only that `_neg_elcbo` calls them with MATLAB's arguments (`(vp, Ns, grad_flags, jacobian_flag)` / `(vp, grad_flags, jacobian_flag)`) and the correct `Ns > 0` switch. `get_hpd`, `ParameterTransformer` and the GP fit are other slices.

---

## 2. Findings

### F1. `_vb_init` type 2 divides the component variances by `lambda²` with the wrong broadcast, inflating the starting widths of every HPD-seeded sieve candidate
- Location: `pyvbmc/vbmc/variational_optimization.py:908-910`; MATLAB: `misc/vbinit_vbmc.m:32`
- Category: formula/gradient (broadcasting/shape)
- Proposed classification: port discrepancy
- Confidence: high
- History: no. `vbinit_vbmc.m:32` is unchanged since `1eb4030` (2019-04-17); the Python line dates from the original port (`9267816`, 2021-08-25).
- MATLAB: `sigma0 = sqrt(mean(V./lambda0.^2)/Knew).*exp(0.2*randn(1,Knew))` with `V` and `lambda0` both `(D,1)`, so `V./lambda0.^2` is the elementwise ratio and the scalar is `mean_d(V_d/λ_d²)`. Python: `V = np.var(mu0, axis=1, ddof=1)` has shape `(D,)` while `lambd0 = vp.lambd.copy()` has shape `(D,1)`, so `V / lambd0**2` broadcasts to a `(D,D)` matrix whose entry `(i,j)` is `V_j/λ_i²`, and `np.mean` of it is `mean(V)·mean(1/λ²)`. MATLAB is right: the intended quantity is the per-dimension variance measured in units of that dimension's length scale. By Chebyshev's sum inequality `mean(V)·mean(1/λ²) ≥ mean(V/λ²)` whenever `V` and `1/λ²` are oppositely ordered, so PyVBMC's starting width is systematically too large. Inert only for `D == 1`.
- Consequence if real: every type-2 sieve candidate (the "start from highest-posterior-density training points" family) starts with mixture widths scaled by `sqrt(mean(V)·mean(1/λ²) / mean(V/λ²))`. Measured on the eight oracle fixture states, that factor is **1.01×** (`halfnormal_D2`), **1.03×** (`rosenbrock_D2`), **1.04×** (`corr_D5`), **1.47×** and **1.51×** (`normal_D2`), **2.15×** and **3.02×** (`cigar_D4`) — i.e. it grows with the anisotropy of `lambda`, exactly the regime the `lambda` division exists to correct. It triggers on every `optimize_vp` call with `slow_opts_N ≥ 2`, which is the first iteration of every run, the iteration after warmup ends, and every iteration after an input warp (`recompute_var_post`), plus the warp-undo refit at `vbmc.py:1329`. The effect is on the *starting points* of the slow optimizations, so it degrades (does not corrupt) the fit: worse type-2 starts mean the sieve ranks them lower and `optimize_vp` more often spends its type-2 slot on a poor initialization. Note the shape stays `(1, K_new)`, so nothing downstream errors.
- Suggested reproduction (no MATLAB needed): with `vp.lambd` of shape `(D,1)` and non-constant, `np.shape(np.var(mu0, axis=1, ddof=1) / vp.lambd**2) == (D, D)`; compare `np.mean(V / vp.lambd**2)` with `np.mean(np.ravel(V) / np.ravel(vp.lambd)**2)`. `cmp4.py` in the scratchpad does this against the `halfnormal_D2_bounded` fixture and then through `_vb_init` itself.
- Test adequacy: no. `test_vb_init_candidates` (`pyvbmc/testing/vbmc/test_variational_optimization.py:576`) asserts only `cand.sigma.shape == (1, K_new)` and generator/transformer identity; `test_vb_init_type3_preserves_fixed_sigma_without_extra_draws` covers type 3 only. No test compares a type-2 `sigma0` against a formula, and the oracle fixtures do not cover `_vb_init`.

### F2. The soft optimization bounds are rebuilt from the current training set every iteration; MATLAB accumulates them over the whole run
- Location: `pyvbmc/vbmc/variational_optimization.py:1027` (`new_vp.bounds = None` in `_vb_init`), with `optimize_vp:163` and `:325`; `pyvbmc/variational_posterior/variational_posterior.py:334-366` (`get_bounds`). MATLAB: `misc/vbinit_vbmc.m:39` (`vp0_vec(iOpt) = vp;` copies `bounds`), `misc/vpsieve_vbmc.m:40`, `misc/vpoptimize_vbmc.m:27` and `:188`, `misc/vpbounds.m:9-24`, `misc/setupvars_vbmc.m:98`
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high (mechanism), medium (size of the effect)
- History: no. `vpbounds.m`, `vpsieve_vbmc.m` and `vbinit_vbmc.m:39` are unchanged since before 2021-01-19.
- `vpbounds.m` is written to *accumulate*: it initializes `mu_lb = +Inf`, `mu_ub = −Inf`, `lnscale_lb = +Inf`, `lnscale_ub = −Inf` only when `vp.bounds` is empty, then takes `min`/`max` against the current `gp.X`. In MATLAB the struct survives: `vpsieve_vbmc` updates the incoming `vp`, `vbinit_vbmc` copies the whole struct (bounds included) into every candidate, and `vpoptimize_vbmc:188` returns `vp0_fine(idx)` — a candidate — so the accumulated bounds ride back into `vbmc.m`'s `vp` and into the next iteration. `setupvars_vbmc.m:98` is the only place that clears them. PyVBMC's `_vb_init` explicitly sets `new_vp.bounds = None` on every candidate, and `optimize_vp` returns a candidate, so the accumulation is wiped on every call: `theta_bnd` is always the current `gp.X`'s min/max. Verified: `optimize_vp` on a fixture state returns `vp.bounds is None` while the *incoming* posterior's bounds were computed and then discarded (`cmp4.py`).
- Consequence if real: identical while the training set only grows (min/max are then monotone). It bites whenever the training range shrinks — i.e. after the post-warmup trim (`_setup_vbmc_after_warmup` drops low-density rows and `reupdate_gp` rebuilds the GP on the survivors) and after an input warp, which changes the coordinates entirely. There PyVBMC's soft box on `mu` and its soft upper bound on `ln(sigma·lambda)` are strictly tighter than MATLAB's, so configurations MATLAB leaves unpenalized acquire a quadratic penalty `0.5·((x−ub)/((ub−lb)·0.01))²` in the objective and its gradient. Demonstrated in `cmp7.py`: on the `normal_D2_singlesample` state with an artificial narrowing of the training set, `lnscale_ub` goes from `[−0.07, −0.18]` (MATLAB, accumulated) to `[−5.64, −4.79]` (PyVBMC, reset) and `mu_ub` from `[0.435, 0.363]` to `[0.001, 0.007]`. Note that MATLAB's own behaviour across a warp (retaining bounds expressed in the *pre-warp* coordinates) is questionable; this finding is about the divergence, not about which is better.
- Suggested reproduction: call `optimize_vp` on any state and assert `returned_vp.bounds is None`; then call `get_bounds` twice on the same posterior with a wide `X` followed by a narrower `X` and compare with a single call on the narrow `X`. `cmp4.py`/`cmp7.py` do both. Settling *which* behaviour MATLAB actually produces end-to-end would need MATLAB, but the struct-copy chain above is unambiguous in the source.
- Test adequacy: no. `test_vb_init_candidates:604` asserts `cand.bounds is None` — it pins the current Python behaviour, i.e. it mirrors the implementation rather than the MATLAB specification. Nothing tests bound persistence across iterations.

### F3. The number of sieve candidates is computed from a different `K` than MATLAB's
- Location: `pyvbmc/vbmc/vbmc.py:1505-1507` (`math.ceil(self.options.eval("ns_elbo", {"K": self.vp.K}))`); MATLAB: `vbmc.m:699` (`Nfastopts = ceil(evaloption_vbmc(options.NSelbo,K))`) with `K` assigned once at `vbmc.m:459` (`K = options.Kwarmup;`) and never reassigned
- Category: control flow (and defaults, via `NSelbo = @(K) 50*K`)
- Proposed classification: port discrepancy (PyVBMC's value is the more sensible one, but it differs from MATLAB and from MATLAB's two other call sites)
- Confidence: high
- History: no. The line has read `options.NSelbo(K)` since `c476ab9` (2019-01-10); `fc19a5a` (2019-11-24) only replaced the `isa(...,'function_handle')` test with `evaloption_vbmc`. `K = options.Kwarmup` at `vbmc.m:459` is the only bare-`K` assignment in the file (verified by grepping every `K` occurrence).
- MATLAB's main loop therefore always uses `Nfastopts = ceil(50·Kwarmup) = 100` (and `ceil(100·0.1) = 10` on incremental iterations), regardless of how many components the posterior has grown to. PyVBMC uses `50·vp.K` (and `⌈5·vp.K⌉` incrementally). MATLAB's other two call sites both use the *current* count — `vbmc.m:584` uses `Knew`, `finalboost_vbmc.m:32` uses `Knew` — which is strong evidence that `vbmc.m:699` is a stale-variable slip in MATLAB. Note PyVBMC uses `vp.K` (the pre-`update_K` count) where those MATLAB sites use `Knew`; `Knew` is already available two lines above (`vbmc.py:1499`).
- Consequence if real: the sieve evaluates a different number of candidate posteriors. With `K = 10` PyVBMC generates 500 candidates on a full refit and 50 on an incremental one, against MATLAB's 100 and 10 — 5× the candidates and 5× the `_neg_elcbo` evaluations in the sieve, growing linearly with `K`. It changes both the quality of the selected starting points and the per-iteration cost, and (because the candidates are drawn from `vp.rng`) the random stream. It does not change the number of slow optimizations. PyVBMC's `final_boost` (`vbmc.py:2484`) correctly mirrors `finalboost_vbmc.m` and the warp-undo branch (`vbmc.py:1322`) agrees with `vbmc.m:584` because `Knew == vp.K` there.
- Suggested reproduction: read-only. `grep -n '\bK\s*=' vbmc.m` returns only line 459; compare with `vbmc.m:584` and `finalboost_vbmc.m:32`. Confirming the runtime value would need MATLAB but the static argument is conclusive.
- Test adequacy: no. No test asserts the number of sieve candidates; `test_vbmc_optimize.py`'s end-to-end runs would not notice.

### F4. The deterministic-entropy branch: no function-evaluation cap, a different stopping tolerance, and a hard error where MATLAB falls back to CMA-ES
- Location: `pyvbmc/vbmc/variational_optimization.py:227-241`; MATLAB: `misc/vpoptimize_vbmc.m:76-101`
- Category: control flow
- Proposed classification: port discrepancy (the missing CMA-ES fallback is a known substituted-library consequence; the missing cap and the tolerance semantics are not recorded anywhere)
- Confidence: high on the code difference, medium on the consequence
- History: no (`vpoptimize_vbmc.m`'s only post-port change, `2044530`, touched the Adam branch only)
- MATLAB sets `vbtrain_options.TolFun = options.DetEntTolOpt` (1e-3) **and** `vbtrain_options.MaxFunEvals = 50*(vp.D+2)`, calls `fminunc`, and on an exception retries with CMA-ES using explicit `insigma` per parameter block. PyVBMC calls `sp.optimize.minimize(vb_train_fun, theta0, jac=True, tol=1e-3)` — method BFGS by default — with no iteration or function-evaluation cap (BFGS's own default is `maxiter = 200·len(theta)`), and raises `RuntimeError` when `res.success` is false. Three differences: (a) `tol` maps to BFGS's `gtol` (a *gradient-norm* tolerance) whereas MATLAB's `TolFun` is a function-value tolerance, so the same 1e-3 stops at different places; (b) MATLAB's `50*(D+2)` budget (200 evaluations at D=2) is absent, so PyVBMC can run far longer on a hard problem; (c) BFGS reports `success=False` on `maxiter` and on "Desired error not necessarily achieved due to precision loss" — ordinary outcomes on a near-flat ELBO — and PyVBMC then aborts the whole run, while MATLAB's `fminunc` returns its best iterate with `exitflag = 0` and never enters the `catch`.
- Consequence if real: the branch runs only when `ns_ent_K == 0`, i.e. when `entropy_switch` is on (default off, and forced off for `D < 5`) or `K == 1` (reachable after pruning to a single component). When it runs, PyVBMC can terminate the optimization at a different point, or abort with `RuntimeError("Cannot optimize variational parameters with", "scipy.optimize.minimize.")` where MATLAB would continue. Low frequency, but a hard failure when it fires.
- Suggested reproduction: call `optimize_vp` with `optim_state["entropy_switch"] = True` on a state whose ELBO surface is flat enough that BFGS reports precision loss, and check `res.success`. A direct count check: instrument `vb_train_fun` and compare the number of calls with `50*(D+2)`.
- Test adequacy: partially. `test_vp_optimize_deterministic_entropy_approximation` (`:521`) exercises the branch on a well-conditioned D=1 mixture that converges; it asserts the ELBO and a KL divergence, not the evaluation budget, and would not reach a `success=False` outcome.

### F5. The GP-smoothing bandwidth term (`vp.delta`) is missing from `tau` and `nu`
- Location: `pyvbmc/vbmc/variational_optimization.py:1476-1479` and `:1499-1503`, `:1588-1592`; MATLAB: `misc/gplogjoint.m:86-90, 164, 173, 274, 309, 314`, `misc/vpsieve_vbmc.m:16`, `misc/setupvars_vbmc.m:247`
- Category: formula/gradient
- Proposed classification: possibly intentional (unported feature, inactive at MATLAB defaults)
- Confidence: high
- History: no.
- MATLAB widens every kernel integral by an extra bandwidth: `tau_k = sqrt(sigma_k²·lambda² + ell² + delta²)`, `tau_kk = sqrt(2·sigma_k²·lambda² + ell² + 2·delta²)`, `tau_jk = sqrt((sigma_j²+sigma_k²)·lambda² + ell² + 2·delta²)`, and `nu_k` gains `+ delta²` inside the sum. `delta` comes from `vp.delta`, set by `vpsieve_vbmc.m:16` from `optimState.delta = options.Bandwidth*(optimState.PUB-optimState.PLB)` (`setupvars_vbmc.m:247`). PyVBMC omits it everywhere. `defopts.Bandwidth = 0` (`vbmc.m:313`), so `delta` is identically zero at MATLAB defaults and the two sides agree exactly; PyVBMC has no `bandwidth` option at all, so a user cannot reach the difference.
- Consequence if real: none at defaults. A MATLAB user who sets `Bandwidth > 0` gets a smoothed log joint (and a matching change in `acqwrapper_vbmc.m` and `intkernel.m`); PyVBMC cannot express that.
- Suggested reproduction: none needed — `defopts.Bandwidth = 0` settles it statically.
- Test adequacy: not applicable (the feature does not exist in Python).

### F6. `vp.eta` survives on the returned posterior and can disagree with `vp.w`; MATLAB deletes the field
- Location: `pyvbmc/vbmc/variational_optimization.py:308-309`, `:326`, `:1179-1181`, `:1026`; `VariationalPosterior.set_parameters` (`variational_posterior.py:1122-1140`, which never writes `self.eta`); MATLAB: `misc/rescale_params.m:33-37` (`vp = rmfield(vp,'eta')`)
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high on the mechanism, low on any consequence
- History: no.
- MATLAB's `rescale_params` — called by `get_vptheta` and by `vpoptimize_vbmc:189` on the winning parameters — normalizes `w` and then **removes** `eta`, so no consumer can read a value left over from the optimizer. PyVBMC's `set_parameters` derives `w` from `theta[-K:]` but leaves `self.eta` untouched, and `_neg_elcbo:1179` is the only writer. `optimize_vp` deep-copies `vp0` into `vp0_fine[i_mid]` and `vp0_fine[i_end]` *after* the last objective evaluation of that slow optimization, then calls `set_parameters(theta_best)`. When the winning index is a midpoint (`elcbo_midpoint = True` by default), the returned posterior's `eta` is the one from the *endpoint* evaluation while its `w` comes from the midpoint `theta`: `softmax(vp.eta) != vp.w`. `_vb_init` also assigns candidates a uniform `eta = ones/K_new` unrelated to their `w`.
- Consequence if real: currently cosmetic. `vp.eta` is read only by `entlb_vbmc`, `entmc_vbmc` and `_gp_log_joint`'s softmax Jacobian, all of which run inside `_neg_elcbo` after `eta` has been rewritten from `theta`. It becomes real if a stored or loaded posterior is handed to `pyvbmc.entropy.entlb_vbmc`/`entmc_vbmc` with `grad_flags[3]` set, or if any future consumer reads `vp.eta` — the softmax Jacobian would then be built from the wrong weights.
- Suggested reproduction: run `optimize_vp` with `slow_opts_N = 2` and `elcbo_midpoint = True`, then compare `np.exp(vp.eta)/np.sum(np.exp(vp.eta))` with `vp.w` on the returned posterior (`cmp4.py` prints both; with `slow_opts_N = 1` they agree, which is the case it exercises).
- Test adequacy: no. `test_vb_init_candidates` asserts `cand.eta.shape == (1, K_new)` but nothing asserts consistency with `w`.

### F7. `adaptive_k` is evaluated with the keyword `unkn`, so a user-supplied callable cannot be called; `round` semantics also differ
- Location: `pyvbmc/vbmc/variational_optimization.py:45` (`options.eval("adaptive_k", {"unkn": K_new})`) and `pyvbmc/vbmc/options.py:248-251`; MATLAB: `private/updateK.m:10` with `misc/evaloption_vbmc.m`
- Category: cross-module (and defaults)
- Proposed classification: port discrepancy
- Confidence: high
- History: no.
- `Options.eval` invokes a callable as `f(**evaluation_parameters)`, so the option's parameter *name* is part of its contract. Every other option in this slice is called with the name its default lambda uses (`{"K": ...}` for `ns_elbo`/`ns_ent`/`ns_ent_fine`/`pruning_threshold_multiplier`, `{"N": ...}` for `k_fun_max`), but `adaptive_k`'s default is the constant `2`, so nothing forced the key to be meaningful and it is `"unkn"`. A user who passes `adaptive_k=lambda K: ...` — exactly what MATLAB's `evaloption_vbmc` supports by calling the handle positionally — gets `TypeError: <lambda>() got an unexpected keyword argument 'unkn'`. Secondly, Python's `round` is banker's rounding while MATLAB's `round` is half-away-from-zero, so a callable returning e.g. `2.5` yields 2 in Python and 3 in MATLAB.
- Consequence if real: only for user-supplied `adaptive_k`. At the default (`2`) both sides give 2.
- Suggested reproduction: `VBMC(..., options={"adaptive_k": lambda K: K/2})` and call `update_K`; or directly `options.eval("adaptive_k", {"unkn": 5})` with a lambda-valued option.
- Test adequacy: no. `test_update_K` uses only the default constant.

### F8. `_sieve` with `init_N == 0` returns values `optimize_vp` cannot consume
- Location: `pyvbmc/vbmc/variational_optimization.py:834-841` and `:168`, `:203-204`; MATLAB: `misc/vpsieve_vbmc.m:84-87`, `misc/vpoptimize_vbmc.m:32, 62-63`
- Category: control flow
- Proposed classification: port discrepancy (unreachable in PyVBMC)
- Confidence: high
- History: no.
- MATLAB returns `vp0_vec = vp` (a 1×1 struct array) and `vp0_type = 1`, which `vpoptimize_vbmc` consumes normally (`vp0_vec(1)`, `vp0_vec(idx) = []`). PyVBMC returns a bare `VariationalPosterior` and the scalar `1`; `optimize_vp:168` then does `vp0_vec[0]`, and `VariationalPosterior` defines no `__getitem__`, so the call raises `TypeError`. `np.delete(vp0_vec, idx)` would also fail.
- Consequence if real: none today — no PyVBMC call site passes `fast_opts_N = 0`. MATLAB reaches it from `activesample_vbmc.m:138` (`vpoptimize_vbmc(0,1,...)`), inside the `SampleExtraVPMeans` block that PyVBMC does not port. It is a trap for any future caller.
- Suggested reproduction: `_sieve(options, optim_state, vp, gp, init_N=0, best_N=1)` and inspect the first return value's type; or `optimize_vp(..., fast_opts_N=0, slow_opts_N=1)`.
- Test adequacy: no test passes `init_N = 0`.

### F9. `minimize_adam` mutates its starting point in place
- Location: `pyvbmc/vbmc/minimize_adam.py:81` (`x = x0`) and `:97` (`x -= ...`); MATLAB: `utils/fminadam.m:39` (`x = x0(:)`, which copies)
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high on the mechanism, low on the consequence
- History: no.
- MATLAB's `x0(:)` always produces a fresh column. Python binds `x` to the caller's array and the first update is an in-place `-=`; only from the second iteration on is `x` rebound to fresh arrays by `np.minimum`/`np.maximum`. The caller's `x0` therefore carries the first Adam step.
- Consequence if real: harmless today — `optimize_vp:258` passes `theta_opt = theta0` where `theta0 = vp0.get_parameters()` is a fresh `np.concatenate` result that is not read again. It is a latent aliasing trap for any caller that reuses its starting point (and `test_minimize_adam_sphere` silently has its `x0` overwritten).
- Suggested reproduction: `x0 = np.array([-3.0, -4.0]); minimize_adam(f, x0); print(x0)` — it is no longer `[-3, -4]`.
- Test adequacy: no. None of the five `test_minimize_adam_*` tests checks `x0`.

### F10. Two ordering/selection edge cases: unstable sort in the sieve, NaN-propagating `argmin` in the finalize
- Location: `pyvbmc/vbmc/variational_optimization.py:821` (`np.argsort(nelcbo_fill)`) and `:313` (`np.argmin(elbo_stats["nelcbo"])`); MATLAB: `misc/vpsieve_vbmc.m:81` (`sort(...,'ascend')`) and `misc/vpoptimize_vbmc.m:176` (`min(...)`)
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: medium
- History: no.
- (a) MATLAB's `sort` is stable; NumPy's default `argsort` (quicksort/introsort) is not. Equal `nelcbo` values — possible when two candidates coincide — are ordered differently, and `optimize_vp:198-201` picks the *first* candidate of each type from that order, so a tie can change which starting point is optimized. (b) MATLAB's `min` skips `NaN` and returns the smallest non-NaN entry; `np.argmin` returns the index of the first `NaN`. `elbo_stats["nelcbo"]` is initialized to `+inf` and filled by `_eval_full_elcbo`; a `NaN` from a failed evaluation would be selected by PyVBMC and skipped by MATLAB, and `vp.set_parameters` would then be handed a `NaN` row.
- Consequence if real: (a) measure-zero in practice; (b) turns a single bad evaluation into a `NaN` posterior instead of falling back to the other slow optimization's result.
- Suggested reproduction: (b) set one `elbo_stats["nelcbo"]` entry to `np.nan` and compare `np.argmin` with `np.nanargmin`.
- Test adequacy: no.

---

## 3. Test adequacy notes

- `test_vb_init_candidates` (`pyvbmc/testing/vbmc/test_variational_optimization.py:576`) checks shapes, object identity and non-mutation of the base posterior, but never a value against a formula — which is why F1 (shape-correct, value-wrong) and F2 (`assert cand.bounds is None`, which *pins* the divergent behaviour) both pass. The `bounds is None` assertion is the clearest case in this slice of a test that mirrors the implementation rather than MATLAB.
- `test_minimize_adam_*` (`test_minimize_adam.py`) tests only that Adam converges on toy objectives. Nothing pins the stopping rule (`slope_err`, `dx`), the returned batch-mean iterate, the `x_tab`/`y_tab` off-by-one pairing that `elcbo_midpoint` depends on, or `x0` immutability. Three of the five tests disable early stopping, so the termination clauses — the part that must match `fminadam.m` — are exercised by only two.
- `test_update_K` supplies its own `iteration_history` dict and `optim_state` and checks the resulting `K`. It is a good behavioural test of the rules, but it never varies `adaptive_k` or `k_fun_max` away from their defaults, so F7 is invisible, and it does not exercise the "all recent iterations were warmup" edge (`elcbos_after` empty) where both implementations would raise.
- `test_gp_log_joint` (`:148`) compares against stored MATLAB arrays (`dG_gp_log_joint.txt`) and is the real gate on `_gp_log_joint`; my independent transcription agrees with it. `test_variational_optimization_grad_fd.py` plus the oracle fixtures cover the gradients and the stage-by-stage numerics well. Between them, the formula-level part of this slice is well tested; the *control-flow* part (candidate counts, bounds lifetime, branch selection, error paths) is not tested at all.
- `test_vp_optimize_1D_g_mixture` / `test_vp_optimize_2D_g_mixture` / `test_vp_optimize_deterministic_entropy_approximation` assert statistical closeness to a known posterior. They would not detect F1, F2 or F3: all three change starting points or penalties, not the optimum the optimizer converges to on an easy target.

## 4. Sheet notes

**Entries I found inaccurate:**

- **"`vp.stats["J_sjk"]` is pruned on both component axes"** (P7 section, line 931). The entry's MATLAB claim — "`misc/vpoptimize_vbmc.m` prunes the corresponding quantity consistently with `I_sk`" — is wrong. `vpoptimize_vbmc.m:236-237` reads `I_sk(:,idx) = []; J_sjk(:,:,idx) = [];`: `I_sk` loses a column (dim 2) but `J_sjk` loses a slice on **dim 3 only**, leaving a non-square `(Ns, K, K−1)`. PyVBMC's two-axis deletion is therefore a deliberate *improvement on* MATLAB, not agreement with it. The entry's disposition (deliberate change, restores the intended shape) stands; only its characterization of the MATLAB side needs correcting.
- **"Sampling of the variational parameters is not ported"** (P6 section, line 821). Accurate as to the code, but it cites only `defopts.VarParamsBack`. PyVBMC additionally carries two dead options that belong to this unported feature and are not mentioned: `variational_sampler = "malasample"` and `active_variational_samples = 0` (`advanced_vbmc_options.ini:143`, `:273`), neither of which is read anywhere in `pyvbmc/` outside the options files.

**Deliberate differences the sheet lacks** (all reported above as findings, listed here because they look settled rather than accidental and a future reviewer should not have to rediscover them):

- The sieve candidate count is evaluated at `vp.K` rather than MATLAB's stale `K = Kwarmup` (F3). If this was a deliberate correction of the MATLAB slip it deserves an entry; if not, it is a discrepancy.
- The GP-smoothing bandwidth (`options.Bandwidth` → `optimState.delta` → the `delta²` terms of `gplogjoint.m`) is unported and PyVBMC exposes no corresponding option (F5). It is inactive at MATLAB's default of 0, exactly like the `ELCBOWeight` entry that *is* on the sheet.
- The soft optimization bounds do not accumulate across iterations (F2).
- `misc/rescale_params.m` removes `eta` (and `mode`) from the posterior; PyVBMC's `set_parameters` clears `_mode` but retains `eta` (F6).
- `scipy.optimize.minimize` (BFGS) replaces `fminunc` in the deterministic-entropy branch, dropping MATLAB's `MaxFunEvals = 50*(D+2)` cap and the CMA-ES fallback, and reinterpreting the 1e-3 `DetEntTolOpt` as a gradient-norm tolerance (F4). The sheet has a comparable "substituted library" entry for CMA-ES in slice P2 but nothing for this one.

**One observation adjacent to the `vp_repo` entry** (P2's file, noted because the sheet entry is in P6): `pyvbmc/vbmc/active_sample.py:716-718` calls `np.append(optim_state["vp_repo"], vp.get_parameters())` and discards the result, so after the first element the repository never grows. Consistent with "not ported" (nothing reads it), but the write is a no-op rather than a working producer.
