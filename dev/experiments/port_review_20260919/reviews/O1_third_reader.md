# O1 — third reader: the expected log joint, the ELCBO and their gradients

Raw reviewer report, wave 7 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-23. One Opus agent, fresh
context, read-only on the repositories, combining both tracks: it
re-derived each formula of the slice, checked it by finite differences and
against gpyreg's quadrature, and then compared the code with MATLAB VBMC at
`396d649`. It read PyVBMC at `a65b96f4`, in a worktree frozen at that
commit while the merged waves were gated in the main checkout, with gpyreg
at `v1.3.0` (`0186d89`). Its brief carried one first question: which
configurations the finite-difference tests of `_gp_log_joint`,
`_neg_elcbo`, `_vp_bound_loss` and `_soft_bound_loss` leave out, and
whether each gradient holds there (several GP hyperparameter samples, each
mean function, `K = 1`, `D = 1`, a weight near zero, the `separate_K`
output, the variance branch, the stochastic path of the Adam optimizer);
the report answers it after its table of derivations. Its check scripts ran
in the session scratchpad and are kept only on the machine that ran them
(`dev/scripts/runs/LOCAL.md`). The text below is the agent's final message,
unedited.

---

# O1 — third reader: the expected log joint, the ELCBO and their gradients

## 1. Coverage

**Read completely (worktree at `a65b96f4`):**
- `pyvbmc/vbmc/variational_optimization.py`, the whole file. `_gp_log_joint`, `_neg_elcbo`, `_vp_bound_loss`, `_soft_bound_loss`, `optimize_vp`, `_sieve`, `_eval_full_elcbo` and `_initialize_full_elcbo` were reviewed. `update_K`, `_vb_init` and `_candidate_vp` were read for context only, not reviewed.
- `pyvbmc/vbmc/minimize_adam.py`.
- In `variational_posterior.py`: `get_parameters`, `set_parameters`, `get_bounds` and `__init__`.
- The two call sites in `active_sample.py` (lines 443 and 940).
- gpyreg `v1.3.0`: `Posterior`, `__training_cholesky`, the rank-one `update`, `NegativeQuadratic`, `SquaredExponential`, and the signature of `quad`.
- MATLAB `396d649`: `misc/gplogjoint.m`, `misc/negelcbo_vbmc.m`, `misc/vpbndloss.m`, `utils/softbndloss.m`, `misc/get_vptheta.m`, `misc/rescale_params.m`, `misc/vpoptimize_vbmc.m`, `misc/vpsieve_vbmc.m` (lines 1–90), `misc/vpbounds.m`, `utils/fminadam.m`.
- MATLAB history since 2021-01-19: the only commit touching these files is `2044530` (2021-06-18). It deletes the integrated-mean-function code, which was never ported. The kept formulas are unchanged since before the port.
- Paper: `papers/acerbi2018variational_appendix.md` §A.2–A.3.1.
- Known-differences sheet: slice P6, the P7 entries on `J_sjk`, and all of "Settled non-differences".
- Tests: `test_variational_optimization_grad_fd.py` completely; the `_gp_log_joint`, `_neg_elcbo` and bound-loss tests in `test_variational_optimization.py` (lines 180–440); the headers of the `optimize_vp` tests.

**Skimmed:** `test_variational_optimization_single_sample.py`, `test_minimize_adam.py`, the oracle fixture metadata.

**Not reached:** the entropy internals (`entlb_vbmc`, `entmc_vbmc`, which belong to another reader; only how they are called and combined was checked), and the rest of `active_sample.py`.

## 2. Derivations

Every check below was run through the frozen worktree, with `pyvbmc.__file__` and `gpyreg.__file__` printed. All scripts are in the scratch directory `wave7/O1`.

| Quantity | Derived independently (key identity) | Finite differences / independent check | MATLAB | Verdict |
|---|---|---|---|---|
| Kernel integrated against a component, `z_kn` | ∫N(x;μ_k,diag s_k²)k(x,x_n)dx = sf²∏_d(ℓ_d/τ_kd)exp(−½Σ(μ_kd−x_nd)²/τ_kd²), with τ² = σ_k²λ_d²+ℓ_d² | Enters every value check below | Same | agrees |
| `I_sk` and `G` (zero, const, negquad means) | I_k = z_kᵀα + m0 + ν_k, with ν_k = −½Σ[(μ−x_m)²+σ²λ²]/ω² for negquad | Against `GP.quad(separate_samples=True)`: D 1–3, K 1–3, Ns 1/3, Cholesky and inverse posteriors, heteroskedastic user noise, a rank-one-updated GP. Error ≤ 5e-15 (inverse form ≤ 8e-11) (`check_values.py`, `check_hetero.py`, `check_rank1.py`) | Same | agrees |
| dG/dμ | −w_k Σ_n α_n z_kn δ_kdn/τ_kd − w_k(μ_kd−x_m,d)/ω_d² | FD, all 3 means × D 1–3 × K 1–3 × Ns 1/3 × both noise levels, with and without a weight near 1e-4. Relative error ~1e-10 (`check_grads.py`). The only flags are D=1 at ln sn=−8, which is round-off from \|α\|≈1.5e6: the error falls as h grows (`check_grads_d1.py`). That branch is unreachable in VBMC. | Same | agrees |
| dG/d ln σ_k | σ_k·[w_k σ_k Σ_d(λ_d/τ_kd)² Σ_n(δ²−1)zα − w_k σ_k Σ_d λ_d²/ω_d²] | Same FD sweep | Same | agrees |
| dG/d ln λ_d | λ_d·[λ_d Σ_k w_k(σ_k/τ_kd)² B_kd − λ_d/ω_d² Σ w_kσ_k²] | Same FD sweep | Same | agrees |
| dG/dη (softmax Jacobian, line 1619) | J = diag(w) − wwᵀ, so dG/dη_k = w_k(I_k − G); invariant to a common shift of η | Same FD sweep, including w ≈ 1e-4 | Same form (MATLAB does not subtract max η; the result is the same) | agrees |
| Pair integrals `J_sjk`, `varG` | J_jk = sf²∏ℓ/τ_jk·exp(−½Σ(μ_j−μ_k)²/τ_jk²) − z_jᵀ(K+Σn)⁻¹z_k, with τ_jk² = (σ_j²+σ_k²)λ²+ℓ² (paper A.2.2). For gpyreg, (K+Σn)⁻¹ = (LᵀL)⁻¹/sl with sl = 1/sW₀², also correct for heteroskedastic noise | Diagonal against the `GP.quad` variance (≤1.3e-13, Cholesky). Full matrix against Gauss–Hermite quadrature of gpyreg's `predict_full`, converging to 9e-10 as the nodes increase (`check_values_gh.py`). Heteroskedastic 3e-14; rank-one update 6e-14 | Same, including eps-clamping of the diagonal only and an unclamped `J_sjk` | agrees |
| Average over GP samples, `var_ss` | varG = mean(varG_s) + sample var(G_s); var_ss = sample var + std(varG_s, ddof=1); dG averaged | FD with Ns=3; outputs rebuilt from the returned `I_sk`/`J_sjk` exactly (`check_separate_K.py`) | Same (settled note) | agrees |
| `_neg_elcbo`: F = −G−H(+β√varF) + L_bnd + weight penalty p·Σ min(w_k, thr), with gradient p·1(w<thr) through J | Derived | FD with entlb, soft bounds and the penalty active (w down to 0.024 < thr): all 3 means × D 1–3 × K 1–3 × Ns 1/3, worst 1.2e-11 (`check_grads.py`). Six flag combinations (`check_flags.py`) | Same, apart from the known eta-bound removal | agrees, except F1 |
| Adam path (MC entropy) | dF_mc − dF_lb = −(dH_mc − dH_lb) exactly (identity to 3e-14) | Mean dF over 60 seeds against FD of the mean F: differences within Monte Carlo error (z ≈ 1 per entry over 200 seeds; `check_mc_path.py`, `check_mc_se.py`) | Same call and arguments; `fminadam` matches, including the polyfit slope covariance | agrees |
| `_soft_bound_loss` | ½((slb−x)/ℓ)², gradient (x−slb)/ℓ², with ℓ = (ub−lb)·tol | FD (`check_bound_flags.py`) | Identical | agrees |
| `_vp_bound_loss` fold | d/d ln σ_k = Σ_d ∂L/∂lnscale_dk; d/d ln λ_d = Σ_k | FD under six flag combinations with mean and scale bounds active, D≠K: ~1e-10 | Same | agrees, except F1 |
| `get_parameters`/`set_parameters` gauge | F depends only on σ_kλ_d and w, so rescaling λ to unit RMS leaves F, and gradients taken in raw coordinates, unchanged | FD in raw coordinates agrees | Same as `get_vptheta`/`rescale_params`; `negelcbo_vbmc` does not rescale | agrees, except F1 |

**Answers to the first question:**

The shipped FD tests (`test_variational_optimization_grad_fd.py`) use K=2 throughout, only the NegativeQuadratic mean, and D ∈ {2,3}. They leave out the configurations below. Each gradient holds in all of them, by my checks.

- **Several GP hyperparameter samples:** covered (8 and 4 samples, plus a single sample). Holds.
- **Zero and constant means:** never FD-checked and never value-checked. Hold.
- **K=1:** not tested. Holds. This is the configuration of the deterministic BFGS path in production.
- **D=1:** not tested. Holds.
- **A weight near zero:** not FD-checked; the tests only check invariance to a common shift of η. The weight-penalty gradient is never FD-checked while active: the FD starting points have w ≈ 0.5/0.6, above the 0.125 threshold. Holds.
- **`separate_K`:** it cannot return a gradient (it raises by design). Its `I_sk`/`J_sjk` outputs match independent quadrature and rebuild G, varG and var_ss exactly.
- **The variance branch:**
  - *Value:* correct, as above.
  - *Gradients:* none exist. `dvarG` is always `None`, and requesting the variance together with gradients raises (MATLAB offers a gradient only for `compute_var == 2`; known difference).
  - *What the optimizer does without them:* `gradient_available` is always true, because `elcbo_beta` is hard-coded to 0. If it were false, the deterministic path would run BFGS with SciPy's finite-difference gradient, and the stochastic path raises `ValueError`, where MATLAB switches to CMA-ES. This is unreachable and in effect covered by the known difference "`ELCBOWeight` is not ported".
- **The Adam path:** it is the default whenever K ≥ 2 (`entropy_switch` defaults to False). It holds, within Monte Carlo error.

## 3. Findings

### F1. With `optimize_sigma=False` and `optimize_lambd=True`, `set_parameters` rescales the frozen σ on every call, so the objective depends on call history and the bound loss uses a wrong log scale and a wrong gradient
- Location: `pyvbmc/variational_posterior/variational_posterior.py:1237-1240` (together with `pyvbmc/vbmc/variational_optimization.py:1228`, `:615`); MATLAB: `misc/negelcbo_vbmc.m:33-48`, `misc/vpbndloss.m:16-21`.
- Category: state/caching.
- Proposed classification: port discrepancy.
- Confidence: high that the behavior is as described; low consequence.
- History:
  - The MATLAB lines are unchanged since 2019 (`b7ade85` added `optimize_sigma`).
  - The Python once matched: until `abf5c3ec` (2021-10-31), `_negelcbo` assigned `exp(theta)` directly, as MATLAB does.
  - That commit replaced the assignment with `vp.set_parameters(theta)`, which normalizes λ to unit RMS and multiplies σ by the factor nl (`0089b35a`, 2021-03-16).
- What the code does, and why it is wrong:
  - When σ is not taken from θ, each `_neg_elcbo` call multiplies the stored `vp.sigma` by nl(θ_λ). The optimizers reuse one `vp0` object, so σ drifts from call to call.
  - `_vp_bound_loss` then reads ln σ from the rescaled `vp.sigma` and ln λ from raw θ. The log scale it penalizes is therefore off by ln nl, and its gradient ignores nl's dependence on θ_λ.
  - MATLAB passes `vp` by value and never rescales inside `negelcbo_vbmc`, so there the objective is a function of θ alone. It should be one here too.
- Consequence if real:
  - A VBMC run does not reach it on either side. Both sides always set `optimize_sigma = optimize_lambd = True` (`VariationalPosterior.__init__`; `setupvars_vbmc.m:87-88`), on noiseless and noisy targets alike.
  - It is reachable only by calling `optimize_vp`/`_neg_elcbo` directly with a posterior whose `optimize_sigma` was set to False. `_vb_init` carries a deliberate frozen-σ branch, a known dormant path.
  - There it corrupts the optimization: the widths drift, and the penalty and its gradient are wrong.
- Suggested reproduction: `check_frozen_sigma.py`.
  - Same θ, fresh posterior: F = −0.9717. After five calls at another θ: F = 9.0962, with σ grown from [0.6, 0.4] to [4.66, 3.11].
  - A single call with a log-scale bound active: the bound loss reads a log scale 2.18 above the true one, and the ln λ gradient is analytic [318.1, 125.7] against FD [707.1, 145.1].
- Test adequacy: none would catch it. `test_vp_bound_loss_weight_flags` freezes σ only together with λ, where nl = 1 and nothing drifts. The FD tests always optimize both σ and λ.

### F2. `_neg_elcbo(compute_var=None)` does not follow MATLAB's default for computing the variance
- Location: `pyvbmc/vbmc/variational_optimization.py:1211-1212`; MATLAB: `misc/negelcbo_vbmc.m:16`.
- Category: defaults.
- Proposed classification: possibly intentional.
- Confidence: high on the facts; the consequence is negligible.
- History: MATLAB unchanged since before the port. The Python has been this way since its first version (`9267816c`, 2021-08-25) and never matched.
- What the code does: Python computes the variance only if β ≠ 0. MATLAB also computes it whenever the caller asks for `varF` (`nargout > 4`). Python always returns `varF`, so a caller that omits `compute_var` silently gets `varF = 0.0`. The docstring says only "determined automatically".
- Consequence if real: none in a run, noiseless or noisy. Every production caller passes `compute_var` explicitly (`optimize_vp`, `_sieve`, `_eval_full_elcbo`, `active_sample`).
- Suggested reproduction: `_neg_elcbo(theta, gp, vp)` returns `varF == 0.0`, where the MATLAB call `[F,dF,G,H,varF] = negelcbo_vbmc(...)` computes it.
- Test adequacy: `test_neg_elcbo_without_variance_returns_float_placeholders` pins the Python behavior.

### F3. Docstrings misdescribe arguments in this slice
- Location: `pyvbmc/vbmc/variational_optimization.py:452-457` (`_initialize_full_elcbo`), `:1162-1163` and `:1171` (`_neg_elcbo`); MATLAB: `misc/vpoptimize_vbmc.m:32-34`.
- Category: indexing/shape.
- Proposed classification: port discrepancy (documentation only).
- Confidence: high.
- History: present since the first version (`9267816c`).
- What the docstrings say, against what the code does:
  - `_initialize_full_elcbo` calls `D` "The dimension"; it is the length of θ. It calls `Ns` "Number of samples for entropy approximation"; it is the number of GP hyperparameter samples, and sizes `I_sk`/`J_sjk`. These are what MATLAB's call passes (`Ntheta`, `numel(gp.post)`).
  - `_neg_elcbo`'s `Ns` is a sample count per component.
  - The documented parameter `entropy_alpha` is named `_entropy_alpha` in the signature.
- Consequence if real: none numerically.
- Suggested reproduction: read the call at line 186.
- Test adequacy: not applicable.

No other discrepancy from MATLAB was found in the slice beyond the known differences: the eta-bound removal, `compute_var == 2`, the weight-only fast path, `ELCBOWeight`, `Bandwidth`/`vp.delta`, the exact entropy for K=1, BFGS, and the Adam table offset. The code matches each of those entries as written.

## 4. Test adequacy notes
- **Configurations left out of the FD tests:** zero and constant means, K=1, D=1, an active weight penalty, and a weight near zero; details under "Answers to the first question" in §2.
- **The Monte Carlo FD test** (`test_neg_elcbo_grad_fd_mc_entropy`) uses rtol = atol = 1e-2, against gradients dominated by a bound term of about 200. The η block (~0.08) could be wrong by more than 10% and still pass.
- **The small-weight penalty test** (`test_neg_elcbo_retains_capped_small_weight_penalty`) asserts only that the penalty gradient differs from the gradient without it, not that it is correct.
- **Implementation mirrors:**
  - `test_gp_log_joint_softmax_jacobian_1d_eta` rebuilds J_w exactly as the implementation does. The FD tests at non-uniform η cover the spec.
  - `test_gp_log_joint_variance_non_cholesky_branch` compares two implementations of one formula. That branch is unreachable in VBMC (`tol_gp_noise`, so sn² ≥ 1e-5 ≥ 1e-6).
- **The variance value** is pinned against the specification only by the MATLAB fixture: D=2, K=2, Ns=8, negquad, both components near 0 with σ = 1e-3 and uniform weights. It is a real gate for that configuration: the variance is a near-cancellation, and dropping the width terms of the pair integrals moves it by a factor of about 3700 (`check_fixture_sensitivity.py`). Nothing checks it independently at separated or wide components, K≠2, other means, or the value at Ns=1 (the single-sample test checks shape only).
- **Untested paths:** the no-gradient branch of `optimize_vp` (unreachable), and `_neg_elcbo`/`_vp_bound_loss` with σ frozen and λ free (F1).
- **Oracles** pin the current numerics; they are regression evidence, not specification.
