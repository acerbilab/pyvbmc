# P6 — variational optimization, the ELBO and the update of K, internal-correctness track

Raw reviewer report, wave 1 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-19. One Opus agent,
fresh context, read-only on the repositories; it did not read the MATLAB
source. Its check scripts ran in the session scratchpad and are not
retained. The text below is the agent's final message, unedited.

---

## 1. Coverage

**Read completely**
- `pyvbmc/vbmc/variational_optimization.py` (all 1665 lines: `update_K`, `optimize_vp`, `_initialize_full_elcbo`, `_eval_full_elcbo`, `_vp_bound_loss`, `_soft_bound_loss`, `_sieve`, `_vb_init`, `_candidate_vp`, `_neg_elcbo`, `_gp_log_joint`).
- `pyvbmc/vbmc/minimize_adam.py` (all 146 lines).
- `papers/acerbi2018variational_appendix.md` §A.1–A.3 (entropy, expected log joint, its variance, negative-quadratic mean, reparameterization, starting points, Adam schedule) and the relevant paragraphs of `acerbi2018variational_main.md` (ELCBO, warm-up, adding/removing components, K_max, reliability).

**Read as interfaces**
- `pyvbmc/variational_posterior/variational_posterior.py`: `__init__`, `__deepcopy__`, `rng`, `get_bounds`, `get_parameters`, `set_parameters`.
- `pyvbmc/entropy/entlb_vbmc.py` (whole), `entmc_vbmc.py` (header/docstring only).
- `gpyreg/gaussian_process.py`: `__core_computation` (the `L`/`sl`/`sW`/`alpha` parametrization), `update`'s rank-1 branch, `Posterior`.
- `pyvbmc/vbmc/option_configs/*.ini` (every option this module reads), `pyvbmc/vbmc/options.py` (`eval`), `pyvbmc/vbmc/iteration_history.py` (`record`, growth), and the four call sites in `pyvbmc/vbmc/vbmc.py` (warp branch ~1329, main loop ~1499–1533, `final_boost` ~2425–2520, `_check_gp_sampling_stop` ~2270–2320).
- `pyvbmc/testing/vbmc/test_variational_optimization.py`, `_grad_fd.py`, `_single_sample.py`, `test_minimize_adam.py` (read for adequacy, not as specification).

**Not reached**: the body of `entmc_vbmc` (out of scope), `whitening/`, acquisition functions, anything under `dev/`.

**Derivations re-done from the paper and first principles**, then checked numerically against an independent loop implementation I wrote from Eqs. S9–S20 (SE-ARD kernel `κ = σ_f²Λ N(x;x',Σ_ℓ)`, negative-quadratic mean, `z_k`, `ν_k`, `I_k`, `J_jk`) and against central finite differences:

| quantity | check | result |
|---|---|---|
| `I_sk` (per sample, per component) | vs. independent loop implementation, D=3, K=4, N=18, Ns=2 | max abs diff 4.4e-16 |
| `J_sjk` (variance kernel) | same | max abs diff 1.1e-16 |
| `G`, `varG` (incl. the `varG_ss` term) | same | 2.2e-16 / 1.1e-16 |
| `dG` w.r.t. `(mu, ln sigma, ln lambd, eta)` | FD through `set_parameters` | max rel err 2.0e-10, all four blocks |
| `_neg_elcbo` gradient, deterministic entropy, with soft bounds | FD, **all 15 combinations** of `optimize_{mu,sigma,lambd,weights}` | max rel err ≤ 2.3e-9 |
| `_neg_elcbo` gradient far outside the soft bounds (penalty active, L≈8.6e4) | FD | 1.0e-9 |
| `_vp_bound_loss` gradient | FD, all 15 flag combinations | ≤ 1.6e-9 |
| pruning bookkeeping | `vp.stats["I_sk"]/["J_sjk"]` after 3 prunes vs. recomputation on the pruned posterior | 0.0 / 2.2e-16 |
| global NumPy random state around `optimize_vp` | byte comparison of `np.random.get_state()` | unchanged |
| reproducibility from `vp.rng` | two runs, same seed | identical ELBO |

**Verified correct and worth stating explicitly** (absence of a finding is informative here):
- Every einsum subscript and normalization constant in `_gp_log_joint`. The `(Ns,K,D,N)` `delta`, `dsq_sum`, `z`, `zalpha`, `M`, `B`, the `(Ns,K,K,D)` `tau_jk`/`delta_jk`, the `mu_grad/sigma_grad/lambd_grad/w_grad` assembly and its `reshape(Ns, K*D).T` (d fastest, matching `mu.ravel(order="F")`), the log-reparameterization factors, and the softmax Jacobian `diag(w) − w wᵀ` all reproduce my derivation exactly. `lnnf` correctly carries the `Λ = (2π)^{D/2}∏ℓ` factor that the appendix's displayed `z_k^{(p)}` drops (the appendix is internally inconsistent there; the code is right).
- The variance solve. `Y = L⁻¹L⁻ᵀ z_sᵀ / sn2_eff` with `sn2_eff = 1/sW[0]²` equals `(K+diag(sn²))⁻¹ z_sᵀ` **because gpyreg sets `sW = ones(N,1)/sqrt(sl)` with `sl = min(sn²)·sn2_mult` and `LᵀL = (K + sn2_mult·diag(sn²))/sl`** — including after rank-1 updates, which append `1/sqrt(sl)`. The non-Cholesky branch (`post.L = −A⁻¹`) is also right. Heteroskedastic noise does *not* break this, contrary to what the scalar `sW[0]` suggests at first reading.
- The parameter renormalization inside `set_parameters` (`lambd /= nl`, `sigma *= nl`) does **not** corrupt the gradients: `G`, `H` and the bound loss depend on `sigma_k·lambd_d` only, so the projection is along a direction on which the objective is constant, and the gradient field is invariant along it. The FD checks go through `set_parameters` and confirm this.
- Pruning: `already_checked` deletion, `I_sk`/`J_sjk` column/row deletion, `K -= 1`, the re-normalization in `get_parameters`, and the canonical shapes `(D,K')`, `(1,K')`, `(D,1)`, `(1,K')` of the returned posterior. Deleting the pruned column from the pre-pruning `I_sk` is exactly equivalent to recomputing (verified to 0.0/2e-16), because `I_sk` does not depend on `w` and `lambd` is unchanged.
- `update_K`: `elcbo = elbo − 3·elbo_sd` matches the paper's ELCBO with `β_LCB`; `K_max = ceil(n_eff^{2/3})` matches `K_max = n^{2/3}`; `K_new = max(vp_K, min(K_new, K_max))` never decreases K; `iteration_history[...][-1]` is genuinely the last recorded iteration (the arrays grow to `iteration+1`, they are not preallocated); `elcbos_after` cannot be empty when the branch runs, because `record("warmup", ...)` happens *after* the warm-up end check.
- Option names and `eval` keyword names: `k_fun_max(N=)`, `adaptive_k(unkn=)`, `ns_elbo(K=)`, `ns_ent(K=)`, `ns_ent_fast(K=)`, `ns_ent_fine(K=)`, `pruning_threshold_multiplier(K=)`, plus the 12 scalar options — all exist under those names with the meaning the code assumes. `ns_ent_K = ceil(ns_ent(K)/K)` is per-component, matching `entmc_vbmc`'s `Ns` contract.
- Boundary cases: K=1 works end to end (deterministic branch, `I_sk (1,1)`, `J_sjk (1,1,1)`); D=1 is fine; a single hyperparameter sample returns scalar `G`/`varG` and `var_ss = 0.0`; `Ns = 0` is unreachable (gpyreg always leaves ≥1 posterior); a negligible-weight component is handled by the pruning loop without shape damage.
- Random draws: every draw in this module comes from `vp.rng` (`_vb_init` explicitly, `entmc_vbmc` by default, the pruning choice via `vp.rng.integers`); `_candidate_vp` shares `_rng` so candidates stay on one stream. No path touches NumPy's global state (verified byte-for-byte).
- The weight penalty `Σ_k min(w_k, thresh)·weight_penalty` is a capped-L1 sparsity term (minimizing it pushes small weights down), its gradient and softmax chain rule are correct, and it is applied only when `theta_bnd is not None` — i.e. during optimization, never in the reported ELBO. Correct.
- For K=1 the deterministic entropy `entlb_vbmc` uses the *exact* Gaussian entropy, so `_sieve`'s `K == 1 → ns_ent_K = 0` is not an approximation at all. The `entropy_switch` polarity (`True` = use deterministic) is consistent between `_sieve` and `vbmc.py`.

---

## 2. Findings

### F1. `_vb_init` type 2: `V / lambd0**2` broadcasts to (D, D), so the starting scale averages over D² cross terms
- Location: `pyvbmc/vbmc/variational_optimization.py:908` (with `V` from :905/:907, `lambd0` from :889)
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high
- `V` has shape `(D,)`; `lambd0 = vp.lambd.copy()` has shape `(D, 1)`. `V / lambd0**2` therefore broadcasts to `(D, D)` with entry `[i,j] = V_j / λ_i²`, and `np.mean` of that is `(mean_j V_j)·(mean_i 1/λ_i²)` instead of the intended `mean_i (V_i / λ_i²)`. The only reading under which the expression makes sense is the element-wise one (it is the per-dimension variance measured in units of the shared length scale `λ`), which needs `lambd0.ravel()`. The sibling branch (type 3, :981–986) computes the same quantity without the `λ` division at all, so no other line disambiguates the intent.
- Consequence if real: only the *starting points* of the sieve are affected, so the final answer is not biased, but type-2 candidates (start from the highest-posterior-density training points) get systematically wrong component widths whenever `lambd` is non-uniform, which is the normal case once `optimize_lambd` has run. In my reproduction with `lambd ∝ (0.3, 1.0, 1.6)` the base scale came out 0.785× the element-wise value; the skew grows with the spread of `lambd` (Chebyshev's sum inequality makes the two agree only for uniform `λ`). Type-2 candidates are one of the three groups the sieve ranks, and with `elbo_starts = 2` one slow optimization is always started from the type-2/3 group, so worse type-2 candidates can cost an optimization restart.
- Suggested reproduction (ran it): with `D=3`, `lambd = [0.3,1.0,1.6]` normalized, `K=2`, `K_new=4`: `np.shape(V / vp.lambd**2) == (3,3)`; `sqrt(mean(V/lambd**2)/K_new) = 0.936` as coded vs `1.193` element-wise.
- Test adequacy: no. `test_vb_init_candidates` checks only shapes, sharing and non-mutation of the base; `test_vb_init_type3_preserves_fixed_sigma_without_extra_draws` covers type 3. No test looks at the *value* of a type-2 `sigma0`, and `lambd` is `exp(randn)` there so the bug is silently absorbed.

### F2. The returned posterior's `eta` can disagree with its `w` (the Adam midpoint is deep-copied after the endpoint evaluation)
- Location: `pyvbmc/vbmc/variational_optimization.py:308–309` and `:326` (with `variational_posterior.py:1121–1127`, which sets `w` but never `eta`)
- Category: state/caching
- Proposed classification: suspected defect
- Confidence: high
- Inside the slow-optimization loop, `_eval_full_elcbo(i_mid, ...)` runs first (setting `vp0.eta` from the midpoint theta via `_neg_elcbo:1178–1181`), then `_eval_full_elcbo(i_end, ...)` overwrites `vp0.eta` with the endpoint's, and only then are *both* `vp0_fine[i_mid]` and `vp0_fine[i_end]` deep-copied — both carrying the endpoint's `eta`. If `argmin(nelcbo)` selects a midpoint slot, the final `vp.set_parameters(elbo_stats["theta"][idx, :])` restores `mu`, `sigma`, `lambd` and `w` from the midpoint theta but leaves `eta` at the endpoint value, because `set_parameters` assigns `self.w` and never `self.eta`. `softmax(vp.eta) ≠ vp.w` from then on. The class docstring defines `eta` as "the unbounded (softmax) parametrization of the VP mixture components", so the invariant is stated.
- Consequence if real: no numerical error *today* — the only consumers of `eta` (`entlb_vbmc:52`, `entmc_vbmc:88`, `_gp_log_joint:1565`) are reached exclusively through `_neg_elcbo`, which resets `eta` first. But the inconsistency is carried into `vbmc.vp`, into every `IterationHistory` VP record, into saved `.pkl` posteriors and into anything a user reads off `vp.eta`; and any future call of an entropy function or of `_gp_log_joint(..., grad_flags)` on a returned posterior would silently use the wrong softmax Jacobian for the weight block. It triggers whenever `elcbo_midpoint` (default `True`) is on, the stochastic-entropy branch is used, and the midpoint wins.
- Suggested reproduction (ran it): 8 `optimize_vp` runs, `D=2, K=3`, `max_iter_stochastic=60`, no pruning; 4 of 8 returned posteriors had `softmax(eta) ≠ w` (e.g. `w = [0.1399, 0.4787, 0.3814]` vs `softmax(eta) = [0.1397, 0.4907, 0.3696]`).
- Test adequacy: no test asserts `softmax(vp.eta) == vp.w` on the object `optimize_vp` returns. `test_optimize_vp_preserves_transformer_through_pruning` mocks `_eval_full_elcbo` entirely, so it cannot see this.

### F3. `_eval_full_elcbo` ignores `K == 1` and `entropy_switch`, so the reported ELBO replaces an exact entropy with a Monte Carlo one
- Location: `pyvbmc/vbmc/variational_optimization.py:482` (and the `Ns=ns_ent_fine_K` argument at :493)
- Category: control flow
- Proposed classification: suspected defect
- Confidence: medium
- `_sieve:766–768` sets `ns_ent_K = ns_ent_K_fast = 0` when `optim_state["entropy_switch"]` or `K == 1`, i.e. it recognizes both as cases where the deterministic entropy should be used. `_eval_full_elcbo` computes `ns_ent_fine_K = ceil(ns_ent_fine(K)/K) = 4096` unconditionally and never consults `K` or `entropy_switch`, so the final, "more precise" ELBO — the one that ranks the slow optimizations, decides pruning, and is stored in `vp.stats["elbo"]` and reported to the user — always uses the MC entropy. For `K == 1` this is strictly worse: `entlb_vbmc:60–70` returns the **exact** single-Gaussian entropy, while the MC estimate adds noise that `elbo_sd` does not report (`varH = 0.0`, :1275).
- Consequence if real: for `K == 1` the reported ELBO carries an entropy error with SD ≈ 0.015 (D=1) to 0.055 (D=10) — at or above `tol_improvement = 0.01`, the threshold on ELCBO improvement used for termination and for the pruning acceptance test — where an exact value was available. When `entropy_switch` is on (non-default), the objective that was optimized (Jensen lower bound) is not the objective used to rank the endpoints, a second, milder inconsistency.
- Suggested reproduction (ran it): `D∈{1,5,10}`, `K=1`, unit `sigma`/`lambd`: exact `H` = 1.41894 / 7.09469 / 14.18939; 20 repeats of `entmc_vbmc(vp, 4096)` give SD 0.0150 / 0.0323 / 0.0546.
- Test adequacy: no. `test_vp_optimize_deterministic_entropy_approximation` exercises the deterministic *optimization* branch but not the entropy used by `_eval_full_elcbo`.

### F4. `minimize_adam`'s `x_tab` and `y_tab` are offset by one, so `optimize_vp` picks the iterate *after* the best-valued one
- Location: `pyvbmc/vbmc/minimize_adam.py:87` / `:101` (producer) and `pyvbmc/vbmc/variational_optimization.py:290–293` (consumer)
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high (that the offset exists), medium (that it is unintended)
- In the loop body `y_tab[i], grad = f(x)` is evaluated *before* the update, and `x_tab[:, i] = x` is stored *after* it. Hence `y_tab[i] == f(x_tab[:, i-1])`, with `x_tab[:, -1]` conceptually `x0`. `optimize_vp` then does `idx_mid = np.argmin(f_val_lst)` and evaluates the full ELCBO at `theta_lst[:, idx_mid]` — one Adam step past the point whose value was minimal. The final `x = mean(x_tab[:, i-19:i+1])` / `y = mean(y_tab[i-19:i+1])` are averages over windows shifted by one against each other for the same reason.
- Consequence if real: the "best midpoint" candidate is systematically one stochastic step off the best sampled iterate. `_eval_full_elcbo` recomputes the ELCBO at whatever theta it is handed, so no stored value is wrong; the effect is a slightly worse midpoint candidate. Small, and it triggers on every stochastic optimization (the default path).
- Suggested reproduction (ran it): minimize `f(x)=‖x‖²` from `x0=[1,-2]`, 60 iterations, early stopping off; `max|y_tab[1:] − ‖x_tab[:, :-1]‖²| = 0.0` while `max|y_tab − ‖x_tab‖²| = 0.577`. The sequence of points at which `f` was called equals `[x0] + list(x_tab[:, :-1].T)`.
- Test adequacy: no. `test_minimize_adam.py` checks only that the returned `x`/`y` are near the optimum on five test functions; it never inspects `x_tab`/`y_tab` alignment.

### F5. `_sieve` with `init_N == 0` returns objects of the wrong type, and `optimize_vp` then raises
- Location: `pyvbmc/vbmc/variational_optimization.py:834–841`, consumed at `:168` and `:197–201`
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- The `init_N > 0` branch returns `(np.ndarray of VPs, np.ndarray of types, ...)`; the fall-through returns `(copy.deepcopy(vp), 1, ...)` — a bare `VariationalPosterior` and a Python `int`. `optimize_vp:168` immediately does `vp0_vec[0].get_parameters()`, which raises `TypeError: 'VariationalPosterior' object is not subscriptable`; `np.where(vp0_type == 1)[0][0]` would also fail on a scalar. The docstring advertises `vp0_vec : np.ndarray, shape (init_N,)` for both returns.
- Consequence if real: an immediate crash rather than the intended "no sieve, start from the current posterior". Unreachable with shipped options (`fast_opts_N` is `ceil(50·K)` or `ceil(5·K)`, always ≥ 5), so it is a latent defect on a documented code path.
- Suggested reproduction (ran it): `_sieve(options, optim_state, vp, gp, init_N=0, best_N=1, K=2)` → `type(out[0]) is VariationalPosterior`, `out[1] == 1`, and `out[0][0]` raises `TypeError`.
- Test adequacy: no test calls `_sieve` with `init_N = 0`.

### F6. `var_ss` adds a standard deviation of variances to a variance of means
- Location: `pyvbmc/vbmc/variational_optimization.py:1645`
- Category: formula/gradient
- Proposed classification: unsure
- Confidence: low
- `varG_ss = Σ_s (G_s − Ḡ)²/(Ns−1)` is the sample variance of the expected log joint across hyperparameter samples; `np.std(varG, ddof=1)` is the sample standard deviation of the per-sample *variances*. Their sum is what `var_ss` returns. `optimize_vp`'s docstring describes `var_ss` as "Estimated variance of the ELBO, due to variance of the expected log-joint, for each GP hyperparameter sample", which is `varG_ss` alone; the inline comment says only "Variability due to sampling". Nothing in the appendix (§A.2.2 gives `Var[G] = Σ_jk w_j w_k J_jk`, and the multi-sample combination `varG = mean_s varG_s + varG_ss` on the next line is the standard law of total variance) defines the extra term. The two summands are at least dimensionally compatible (both in units of `G²`), so this is not obviously wrong — only unexplained.
- Consequence if real: `var_ss` is consumed only by `VBMC._check_gp_sampling_stop` (`vbmc.py:2317`), compared against `tol_gp_var_mcmc = 1e-4`. An inflated `var_ss` delays the switch to stable GP sampling, i.e. keeps slice-sampling hyperparameters longer — conservative, costs time, does not corrupt results. In my D=3/Ns=2 example `var_ss = 0.02843` against `varG_ss` alone of essentially the same magnitude (the two terms were comparable).
- Suggested reproduction: compute `_gp_log_joint(..., compute_var=True)` with `Ns ≥ 2` and compare `var_ss` with `varG_ss` and with `np.std(varG, ddof=1)` separately; deciding it needs the MATLAB source or the author's intent, which is outside this track.
- Test adequacy: `test_gp_log_joint` pins `var_ss == 1.0317e-04` against a stored number, which mirrors the implementation; it would not detect a wrong formula.

### F7. The deterministic-entropy branch aborts the whole run when SciPy reports failure, discarding the iterate it found
- Location: `pyvbmc/vbmc/variational_optimization.py:227–241`
- Category: control flow
- Proposed classification: possibly intentional
- Confidence: medium
- `sp.optimize.minimize(vb_train_fun, theta0, jac=True, tol=1e-3)` selects BFGS (no bounds, no constraints). `if not res.success: raise RuntimeError(...)` discards `res.x`, which BFGS always returns, including for its most common non-success status ("Desired error not necessarily achieved due to precision loss"). Every other optimization path in this module degrades gracefully. Note also that the `RuntimeError` is constructed with two positional arguments, so the message renders as a tuple with a missing space.
- Consequence if real: a single BFGS precision-loss report — realistic on a near-singular ELBO surface with the stiff soft-bound penalty (`ell = (ub−lb)·0.01`, so violations are amplified by ~1e4) — terminates `VBMC.optimize()` with an exception rather than accepting a usable iterate. The branch is taken whenever `K == 1` or `entropy_switch` is active.
- Suggested reproduction: monkeypatch `sp.optimize.minimize` to return `Mock(success=False, x=theta0)` and call `optimize_vp` with `K=1`; or run the branch on an objective with a strongly violated soft bound.
- Test adequacy: no. `test_optimize_vp_preserves_transformer_through_pruning` mocks `minimize` with `success=True`; no test covers the failure branch.

### F8. `eta` is excluded from the soft-bound loss while `get_bounds` still computes `eta_lb`/`eta_ub`, which are then discarded
- Location: `pyvbmc/vbmc/variational_optimization.py:551–561` (and `variational_posterior.py:357–366`, which produces the bounds)
- Category: state/caching
- Proposed classification: possibly intentional
- Confidence: high (that it happens), medium (that it is a problem)
- `_vp_bound_loss` copies `theta_bnd["lb"]/["ub"]` and sets the last `K` entries to `∓inf` whenever `optimize_weights` is on, so the `[log(0.5·tol_weight), 0]` bounds that `get_bounds` builds for `eta` never have any effect. The in-code rationale is sound: `theta[-K:]` enters only through `softmax`, so it is defined up to an additive constant, and bounding the raw values penalizes a pure gauge shift (with `tol_con = 0.01` and a range of 5.3, a uniform +1 shift of three etas would add ≈ 534 to the objective while leaving `w` untouched). Two things are left inconsistent, though: (a) `get_bounds` still computes and ships those entries, so `theta_bnd` carries values nothing reads — including the `tol_weight == 0` special case whose only stated purpose was "prevent warning to be printed when doing final boost"; (b) with them gone, the only thing that bounds the weights from below is the pruning threshold, while the weight penalty at `:1307–1325` actively *rewards* driving weights below `thresh = max(1/(4K), tol_weight)`.
- Consequence if real: the weight parameters drift freely during Adam; in practice the drift per call is bounded by `max_iter · master_max` (≈6 with the defaults) and small weights are caught by the pruning loop, so I found no run-time failure. The residual risk is a very small `w_k` underflowing to 0 in `get_parameters` (`log(0) = −inf` propagating into `elbo_stats["theta"]`).
- Suggested reproduction (ran it): with the default options, `L(theta0) == L(theta0 + 20·e_eta) = 1754.85` — a 20-unit shift of every `eta` changes the bound loss not at all, while `theta_bnd["lb"][-K:] = −5.298`, `["ub"][-K:] = 0`.
- Test adequacy: the behavior is deliberately pinned (`test_neg_elcbo_grad_fd_deterministic_entropy` asserts `np.all(dL[-K:] == 0.0)`, `test_vp_bound_loss_grad_fd` asserts `shifted_L == L`), which is exactly the "test mirrors the implementation" pattern — it documents the choice but cannot flag the dead `eta_lb`/`eta_ub`.

### F9. `elcbo_beta` is hard-coded to 0, so the "ELCBO" used throughout `optimize_vp` is the plain ELBO
- Location: `pyvbmc/vbmc/variational_optimization.py:771–775`
- Category: defaults
- Proposed classification: possibly intentional
- Confidence: high
- The comment marks it "Missing port: elcboweight does not exist". With `elcbo_beta = 0`, `compute_var = False` everywhere in the sieve and the optimization, `nelcbo = nelbo` in `_eval_full_elcbo` (`:501`), and consequently: the selection among slow optimizations (`:313`), the sieve ranking (`:818`) and the pruning acceptance test all use the ELBO, not `ELCBO = G + H − β√Var` (paper Eq. 8). Three code paths become unreachable as a result: `_neg_elcbo`'s `beta != 0` confidence-bound term and its `NotImplementedError` (`:1162–1166`), `_gp_log_joint`'s variance-gradient `NotImplementedError` (`:1414–1418`), and `compute_var == 2` (`:1419–1423`). Note that the pruning test at `:361–364` *does* use `elcbo_impro_weight = 3`, so β is only absent from the optimization objective, which is consistent with the paper's use of β_LCB for the improvement/fallback checks rather than for the fit.
- Consequence if real: no variance-aware optimization is possible even by option; a user cannot re-enable it. Behaviourally this is likely what the original defaults do, so I classify it as intentional, but it should be read as "β is not an option in PyVBMC", not as "β defaults to 0".
- Suggested reproduction: none needed — set `elcbo_beta` to any nonzero value in a scratch copy and `_neg_elcbo` raises `NotImplementedError` on the first gradient call.
- Test adequacy: `test_neg_elcbo_grad_fd_*` all use `beta = 0`, consistent with the unreachable branch.

### F10. `minimize_adam` mutates its `x0` argument, and its trailing-average window wraps when `max_iter < 2·batch_size`
- Location: `pyvbmc/vbmc/minimize_adam.py:81`, `:97`, `:139–140`
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high
- `x = x0` aliases the caller's array, and `x -= ...` on the first iteration writes through the alias (from the second iteration on, `np.minimum(ub, np.maximum(lb, x))` has rebound `x` to a fresh array, so exactly one update leaks). The docstring gives no hint that `x0` is consumed. Separately, `x = np.mean(x_tab[:, i-batch_size+1 : i+1], axis=1)` uses a negative start index when `i + 1 < batch_size`, which Python interprets as a wrap-around, so a short run silently averages the wrong (and possibly much shorter) window instead of all available iterates.
- Consequence if real: the aliasing is harmless at the only call site (`theta0 = vp0.get_parameters()` is a fresh array and is not reused afterwards), but it is a trap for any future caller. The window wrap requires `max_iter_stochastic < 40`; the shipped value is `100·(2+D) ≥ 300` and `final_boost` sets it to `inf`, so it is latent.
- Suggested reproduction (ran it): `x0 = np.array([1.0, -2.0]); minimize_adam(f, x0, max_iter=5)` leaves `x0 == [0.90049, -1.90049]`. With `max_iter=15` the returned `x` is the mean of columns 11–14 only.
- Test adequacy: no. `test_minimize_adam.py` passes freshly constructed `x0` arrays and always uses large `max_iter`.

### F11. Adam constants and schedule differ from the paper's stated values
- Location: `pyvbmc/vbmc/minimize_adam.py:63` and `pyvbmc/vbmc/variational_optimization.py:261–268`
- Category: defaults
- Proposed classification: possibly intentional
- Confidence: medium
- Appendix §A.3.3 states `β₁ = 0.9, β₂ = 0.99, ε ≈ 1.49e-8, α_min = 0.001, τ = 200, n_batch = 20`, with `α_max = 0.1` during warm-up and `0.01` thereafter. The code matches `β₁`, `ε` (`sqrt(spacing(1))`), `α_min`, `τ` and `n_batch`, but uses `β₂ = 0.999` (the generic Adam default, not the paper's `0.99`), and derives `α_max` from `sgd_step_size = 0.005` as `min(0.1, 10·sgd_step_size) = 0.05` during warm-up and `min(0.1, sgd_step_size) = 0.005` after — exactly half the paper's values in both regimes (`sgd_step_size = 0.01` would reproduce the paper exactly). The warm-up branch additionally fires whenever `not vp.optimize_weights`, a condition the paper does not mention.
- Consequence if real: a different, generally slower/steadier stochastic optimization than the paper describes. No correctness impact; the ELBO is re-evaluated exactly afterwards. Worth recording because the paper is the only written specification of these constants and `β₂` is not exposed as an option.
- Suggested reproduction: read `minimize_adam.py:63` and evaluate `min(0.1, 0.005*10)` / `min(0.1, 0.005)`.
- Test adequacy: not applicable — no test pins the schedule.

### F12. `update_K`: the "two iterations right after warm-up" exclusion is really "the two oldest iterations in the window", and the recency window is 6, not the paper's 4
- Location: `pyvbmc/vbmc/variational_optimization.py:49–65`
- Category: defaults / control flow
- Proposed classification: possibly intentional
- Confidence: medium
- `elcbos_after` is the non-warm-up subset of the *recent* window `[iter − recent_iters :]`. `elcbos_after[0 : min(2, iter+1)] = −inf` therefore excludes the two oldest entries of that window, which coincide with "right after warm-up" only for the first few post-warm-up iterations; once warm-up has scrolled out of the window (after ~7 post-warm-up iterations with the defaults) the code permanently ignores the two oldest recent iterations, making `improving_flag` easier to satisfy and K grow more readily than the comment implies. Separately, `recent_iters = ceil(0.5 · tol_stable_count / fun_evals_per_iter) = 6` against the paper's `n_recent = 4` (the paper's `n_stable = 8` corresponds to `tol_stable_count = 40`, not the shipped 60), and `pruning_threshold = tol_improvement · K^{-1/2}` (`:366–368`) against the paper's flat `ε = 0.01`. The bonus branch (`:73–77`) also requires `improving_flag`, which the paper's "add two extra components if the solution is stable and no component was recently pruned" does not.
- Consequence if real: K grows slightly more eagerly and pruning is slightly stricter at large K than the paper describes; all three are parameterized by shipped options, so this reads as post-paper tuning rather than a defect. The comment/behaviour mismatch on the `-inf` line is the part I would call wrong as written.
- Suggested reproduction: call `update_K` with a 10-entry history all marked `warmup=False` and a monotonically increasing `elbo`; `elcbos_after[0:2]` is set to `−inf` even though warm-up ended nine iterations earlier.
- Test adequacy: `test_update_K` exercises the branches but constructs histories in which warm-up is still inside the window, so it never reaches the regime where the comment stops describing the code; and it pins the implementation's `recent_iters`, not the paper's.

### F13. Two small dead/ineffective pieces of state
- Location: `pyvbmc/vbmc/variational_optimization.py:484`; and `:1027` together with `variational_posterior.py:331–352`
- Category: state/caching
- Proposed classification: possibly intentional
- Confidence: high
- (a) `if "skip_elbo_variance" in options and options["skip_elbo_variance"]` — `skip_elbo_variance` appears in neither `.ini`, and unknown keys are rejected at validation, so the check is always `False` and the `compute_var = False` path is unreachable. (Were it reachable it would break: `_neg_elcbo` would return `J_sjk = None` and `_eval_full_elcbo:513` would try to assign `None` into a float array.)
- (b) `_vb_init` sets every candidate's `bounds = None`, and `optimize_vp` returns one of those candidates, so `VBMC.vp.bounds` is `None` at the start of every iteration and `get_bounds`'s accumulate-across-calls logic (`np.minimum(..., self.bounds["mu_lb"])` etc.) never accumulates across iterations — it only runs twice within one `optimize_vp` call, idempotently. Because `X` normally only grows, the fresh bounds usually equal the accumulated ones; they differ after warm-up trimming removes training points and after a warping changes the coordinates, which is precisely when the accumulation was presumably meant to matter.
- Consequence if real: (a) none. (b) the soft bounds track the current training set rather than the union over the run; arguably the better behavior after a warp, but not what `get_bounds` is written to do.
- Suggested reproduction (ran (b)): after `optimize_vp`, `v.bounds is None`.
- Test adequacy: no test covers either.

---

## 3. Test adequacy notes

- **`test_gp_log_joint` (`:148`) and `test_variational_optimization_single_sample.py` pin stored numbers, not the specification.** `assert np.isclose(var_ss, 1.031705745662353e-04)` would pass unchanged if the `varG_ss + std(varG)` formula (F6) were wrong, since the reference was produced by the same code path. The MATLAB-derived `dG_gp_log_joint.txt` is a genuine external check of the gradient, but only at one point with `K=2, D=2, Ns=3`.
- **The FD suite (`test_variational_optimization_grad_fd.py`) is the strongest part of the coverage** and my independent FD runs agree with it: `_gp_log_joint`, `_neg_elcbo`, `_vp_bound_loss` and `_soft_bound_loss` gradients are checked including the `D ≠ K` and no-Jacobian variants. Its `test_neg_elcbo_grad_fd_mc_entropy` correctly uses a loose `rtol=1e-2, atol=1e-2`, because the paper's MC entropy gradient (Eqs. following S6) deliberately drops the score term using `E_q[∂ log q/∂φ] = 0`; I confirmed the resulting mismatch is O(0.1) at `Ns = 40` and is *not* a defect. Worth stating so the loose tolerance is not mistaken for sloppiness.
- **`test_neg_elcbo_grad_fd_deterministic_entropy:291` (`np.all(dL[-K:] == 0.0)`) and `test_vp_bound_loss_grad_fd:376` (`shifted_L == L`) encode the F8 deviation as the expected behavior.** They are implementation mirrors: they would keep passing if the exclusion were wrong, and they are the only place the decision is recorded outside the source comment.
- **`test_optimize_vp_preserves_transformer_through_pruning` mocks `_sieve`, `sp.optimize.minimize` and `_eval_full_elcbo`.** It verifies the transformer identity and the `I_sk`/`J_sjk` deletion bookkeeping well, but because `_eval_full_elcbo` is a stub it cannot see F2 (stale `eta`), F3 or F7.
- **`test_update_K` reproduces the implementation's arithmetic** (`recent_iters = 6`, the bonus of 2, the `K_max` cap) with histories in which warm-up is still inside the recent window, so it does not distinguish the code from the paper (F12) and does not reach the regime where the `-inf` comment stops holding.
- **`test_minimize_adam.py` only checks end-point accuracy on five analytic functions.** Nothing tests the `x_tab`/`y_tab` contract (F4), the in-place mutation of `x0` or the trailing-window slice (F10), which are exactly the parts `optimize_vp` depends on.
- **No test covers `_sieve(init_N=0)` (F5), the value of a type-2 `sigma0` (F1), or `entropy_switch = True` end to end.**
