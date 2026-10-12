# O4 — third reader: the GP's marginal likelihood and its gradient

Raw reviewer report, wave 7 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-23. One Opus agent, fresh
context, read-only on the repositories, combining both tracks: it re-derived
each formula of the slice, checked it by finite differences, and then
compared the code with `gplite/` of MATLAB VBMC at `396d649`. It read gpyreg
at `v1.3.0` (`0186d89`), with PyVBMC at `a65b96f4` as its caller, in a
worktree whose package code stayed at that commit while the merged waves
were gated in the main checkout. Its brief carried one first question:
whether the gradient holds with respect to every hyperparameter in every
configuration PyVBMC builds (each mean function, the three noise
configurations with the multiplier of the recorded noise at uncertainty
level 1, the low-noise representation of the posterior, the hyperpriors, `D
= 1`, repeated training inputs); the report answers it after its table of
derivations. Its check scripts ran in the session scratchpad and are kept
only on the machine that ran them (`dev/scripts/runs/LOCAL.md`). The text
below is the agent's final message, unedited.

---

# O4 — third reader: the GP's marginal likelihood and its gradient

## 1. Coverage

`git -C gpyreg describe --tags` returned `v1.3.0` at the start and again at the end. The gpyreg tree and the `pyvbmc-wave7` worktree were clean both times.

**Read completely:**
- gpyreg `gaussian_process.py`:
  - `__core_computation` (3071–3234), including the `Posterior` it returns and the Cholesky cache.
  - `__training_cholesky` (2987–3069).
  - `__compute_log_priors` (1690–1856), `__prior_masks` (1609–1688) and `__recompute_normalization_constants` (1568–1607).
  - `log_likelihood`, `log_posterior`, `__compute_nlZ` and `__gp_obj_fun` (1858–1962).
  - `set_priors`, `get_priors`, `set_bounds` and `get_recommended_bounds`.
  - The prior, `df` and bounds handling of `fit` (1171–1566).
- Other gpyreg modules, read whole: `covariance_functions.py`, `isotropic_covariance_functions.py`, `mean_functions.py`, `noise_functions.py`, and `f_min_fill.py: smoothbox_cdf` / `smoothbox_student_t_cdf`.
- MATLAB: `gplite/private/gplite_core.m`, `gplite_nlZ.m`, `gplite_hypprior.m`, `gplite_noisefun.m` and `private/derivcheck.m` in full. The compute and gradient branches of `gplite_covfun.m` and `gplite_meanfun.m` (cases 0, 1, 4), and `gplite_train.m:95–300` and `:490–545`.
- PyVBMC (worktree): `gaussian_process_train.py` (`train_gp`, `_gp_hyp`, `_get_training_data`), the `gp_noise_fun` and uncertainty-level setup in `vbmc.py:1189–1248`, and the noise recording in `FunctionLogger`.
- Known-differences sheet: the GP-layer entries and the GP items of the settled non-differences.

**Skimmed:** the gpyreg tests (`test_gaussian_process.py` gradient, prior and fixed-bounds tests; the kernel-gradient tests; `test_utils.check_grad`), PyVBMC's `test_gp_hyp` and the `gp_nlZ` oracle helper.

**Not reached:** the rest of `fit` beyond its prior and bounds handling, the slice sampler and `f_min_fill`'s design (all outside the slice).

## 2. Derivations

Notation: C = K + m·diag(sn2), where m is `sn2_mult`; r = y − mean; α = C⁻¹r; Q = C⁻¹ − αα′.

"FD" means Richardson central differences in double precision. The gradient configurations are:
- **Likelihood grid:** D ∈ {1, 3} × mean ∈ {zero, const, negquad} × noise ∈ {[1,0,0], [1,2,0] with s2, [1,1,0] with s2}, each normal, low-noise, and with repeated inputs; plus [0,0,0], [1,0,1], [1,2,1], [0,2,0], [0,1,0], and repeated inputs with low noise.
- **Scalar total noise:** s2 absent (`check_scalar_noise.py`).
- **GPs built by PyVBMC's own `_gp_hyp`** at uncertainty levels 0, 1 and 2, each mean function, D = 1 and 3, through the full fit objective with its priors (`check_pyvbmc_config.py`).

| Quantity | Independent derivation | FD check | MATLAB | Verdict |
|---|---|---|---|---|
| nlZ, Cholesky representation | L = chol(C/sl), so log\|C\| = N·log(sl) + 2Σ log L_ii, giving nlZ = ½r′α + Σ log L_ii + (N/2)·log(2π·sl) | Value vs dense `slogdet`: ≤ 4e-15 relative, whole grid | `gplite_core.m:79,102,193` | agrees |
| nlZ, low-noise representation (min sn2 < 1e-6) | L = chol(C), sl = 1; same formula | Value vs dense: ≤ 4e-10; vs a 60-digit decimal reference: ≤ 6e-10 (`check_lownoise_mp.py`) | `:93,96,193` | agrees |
| ∂nlZ/∂θ_cov | ½ tr(Q ∂K) | Grid, Cholesky representation: per-component relative error ≤ 2e-8. Low-noise: analytic vs 60-digit reference 4e-11 to 2e-9 of the max component. Double-precision FD is what fails there (conditioning), not the analytic gradient | `:226,229–238` | agrees |
| ∂nlZ/∂θ_noise, per-point noise | ½ m Σ_i ∂sn2_i Q_ii | Levels 1 and 2, rectified, both representations, repeated inputs | `:246–247` | agrees |
| ∂nlZ/∂θ_noise, scalar total noise | ½ m ∂sn2 tr(Q), taking row 0 of ∂sn2 | [1,2,0], [0,2,0] and [1,1,0] without s2: ≤ 5e-9 | `:244` indexes linearly (settled entry) | agrees; matches the sheet |
| ∂nlZ/∂θ_mean | −∂m′α | Grid | `:259` | agrees |
| SE ARD ∂K | ∂K/∂log ℓ_i = K·(Δx_i/ℓ_i)²; ∂K/∂log sf = 2K | Entrywise ≤ 2.4e-10; through nlZ (`check_kernels.py`) | `gplite_covfun.m:180–184`, `gplite_core.m:229–234` | agrees |
| Matérn ARD, d = 1, 3, 5 | ∂K/∂log ℓ_i = sf²·g(t)·e^(−t)·K_i, where g = (f − f′)/t = 1/t, 1, (1+t)/3 | Entrywise and through nlZ, with repeated inputs and a shared coordinate | `covfun.m:196–200,214–219`; the d = 1 zero repair is settled | agrees |
| Rational-quadratic ARD | ∂K/∂log ℓ_i = sf²·M^(−α−1)·K_i; ∂K/∂log α = K·(r²/(2M) − α·log M) | Entrywise and through nlZ | no counterpart | agrees |
| Isotropic SE and Matérn | as above with t² in place of K_i | Entrywise and through nlZ | no compute branch in MATLAB | agrees |
| Mean functions | const: ∂m = 1. negquad: ∂m/∂m₀ = 1; ∂m/∂x_m,j = (x_j − x_m,j)/ω_j²; ∂m/∂log ω_j = z_j² | Through nlZ, whole grid | `meanfun.m:403–406,425–436` | agrees |
| `GaussianNoise` | ∂/∂h₀ = 2e^(2h₀); ∂/∂h_mult = e^h·s2; ∂/∂y_t = 2w²(y_t − y)·1[zz > 0]; ∂/∂log w = 2w²zz² | Through nlZ, including the level-1 multiplier with s2 = 1/n | `noisefun.m:177–210` | agrees |
| Gaussian prior | −½log(2πσ²) − z²/2; gradient −(h − μ)/σ² | Value vs `scipy.stats` ≤ 4e-15; gradient FD (`check_prior.py`) | `hypprior.m:44–49` | agrees |
| Student-t prior | lgamma terms − log σ − ((ν+1)/2)·log1p(z²/ν); gradient −((ν+1)/ν)·(h − μ)/σ² / (1 + z²/ν) | ν ∈ {0.5, 1, 3, 30, ∞, NaN→Gaussian} | `:52–57` | agrees |
| Smooth box, Gaussian tails | flat on [a, b], normalizer C = 1 + (b − a)/(σ√(2π)); tail gradient −(h − a or b)/σ², zero inside | Independent density plus integration; blocks with below, inside and above coordinates (`check_prior_block.py`) | none | agrees |
| Smooth box, Student-t tails | C = 1 + (b − a)·Γ((ν+1)/2) / (Γ(ν/2)·σ·√(νπ)) | same, ν up to 400 | none | agrees |
| Renormalization over the bounds | subtract log of the prior mass in [lb, ub] | vs numerical integration ≤ 7e-14 for finite, infinite and half-infinite bounds | none (settled entry) | agrees, except **F2** |
| Fixed prior (lb == ub) | a coordinate without a prior contributes nothing, so its gradient term should be 0 | `check_fixed*.py` | no counterpart; MATLAB gives dlp = 0 | **F1** |
| `log_likelihood`, `log_posterior`, `__compute_nlZ`, `__gp_obj_fun` | composition and signs | FD of the fit objective on PyVBMC-built GPs: error ≤ 3e-9 of the max component | `gplite_nlZ.m:45–66`, `gplite_train.m:498–545` | agrees |
| Cholesky cache | a hit reuses the factor of an identical covariance and noise block | cache vs fresh evaluation: difference exactly 0 over 6 moves of the mean hyperparameters | none | agrees |
| Returned `Posterior` (α, L in both representations, sW) | definitions | vs dense ≤ 2e-14; `predict` vs dense ≤ 4e-15 (`check_posterior.py`) | `:82–99,279–284` | agrees |
| Jitter ladder (m > 1) | the gradient treats m as a constant, which is exact for the matrix actually factored | Only reachable at condition ~1e16, where double precision loses the gradient on any implementation (`check_ladder.py`) | same as MATLAB | not settled numerically; no finding |

**Answer to the brief's first question.** Yes: the gradient holds with respect to every hyperparameter in every configuration PyVBMC builds. That covers each mean function; the three noise configurations, including the level-1 multiplier; with and without the hyperprior; D = 1; and repeated inputs.

- The low-noise representation's gradient is also correct. At PyVBMC's defaults it is unreachable: the noise floor gives sn2 ≥ 1e-5 > 1e-6.
- The one exception is a hyperparameter fixed by equal bounds with no prior on it (F1). PyVBMC does not build that configuration.

## 3. Findings

### F1. A hyperparameter fixed by equal bounds and without a prior gets a NaN gradient, which stops the optimizer in `fit`
- Location: gpyreg/gaussian_process.py:1716–1720 (`if compute_grad: dlp[f_idx] = np.nan`), reached through `__compute_nlZ`:1933 and `__gp_obj_fun`; consumed by `fit`:1503. MATLAB: no counterpart (`gplite_hypprior.m:25` initializes dlp to zeros and has no fixed branch).
- Category: formula/gradient
- Proposed classification: port discrepancy (in a Python-only addition)
- Confidence: high
- History: the MATLAB files of this slice have no commits after 2021-01-19 (the last is 2020-05-08). `f_idx` and the NaN arrived together in gpyreg `6754f01` (2021-06-22) and never matched MATLAB.
- What the code does, and what it should do:
  - The code sets dlp = NaN on every coordinate with lb == ub, before the family branches run.
  - Where the fixed coordinate also has a Gaussian, Student-t or smooth-box prior, those branches overwrite the NaN, so there is no effect.
  - Where it has no prior, the NaN survives into the gradient that L-BFGS-B receives, and L-BFGS-B stops at its starting point.
  - A coordinate without a prior contributes no term to the log prior, so its gradient contribution is 0, as MATLAB returns.
  - This contradicts the known-differences entry "The hyperprior is renormalized to the bounds, and a fixed hyperparameter gets a prior", which says `f_idx` "changes only what a caller who evaluates off a fixed value is told". The NaN arrives even at the fixed value.
  - The same entry's "Why" cites `0ca35b3` and `27f8d66`. `git log -S` shows that `f_idx` came with `6754f01` and the renormalization with `64dc49d` (2021-06-29).
- Consequence if real:
  - With `n_samples = 0`, `fit` returns the best point of the space-filling design.
  - With sampling, the slice sampler starts from that design point instead of the optimum.
  - Triggers in gpyreg alone: a caller fixes a hyperparameter by equal bounds without a prior (exactly what `test_fitting_with_fixed_bounds` does), or `get_recommended_bounds` collapses the noise pair for targets whose range is below 1e-6.
  - PyVBMC does not reach it, on noiseless or noisy targets. The one bound pair `_gp_hyp` can collapse is `noise_log_scale` (floor log(tol_gp_noise) above the recommended log range of y), and it always carries a Student-t prior. The coordinates without a prior (output scale, mean) do not get equal bounds unless every training input shares a coordinate.
- Suggested reproduction:
  - `check_fixed_fit.py` (D = 2, negquad mean, noise fixed at log(1e-2), no prior on it): `fit` returns nit = 0, "ABNORMAL", log likelihood −61.88. A hand optimization of the free coordinates from that same point reaches +28.66 in 14 iterations. The same fit with a prior on the fixed coordinate runs 19 iterations normally.
  - `check_fixed_testcfg.py` (the configuration of `test_fitting_with_fixed_bounds`, without sampling): nit = 1, gradient at the result [−5.51, 4.61, 0.22, nan]. Widening the pair by 1e-9 gives nit = 11 and free gradients around 1e-4.
- Test adequacy: `test_fitting_with_fixed_bounds` runs this configuration but asserts only that the fixed value is kept.

### F2. The mass of a prior inside its bounds is computed as cdf(ub) − cdf(lb), which is 0 in the far upper tail, so `log_posterior` becomes +inf at every point
- Location: gpyreg/gaussian_process.py:1592–1607, with the log at :1670 and the subtraction at :1851; `f_min_fill.py:300,337` (the upper branches of the smooth-box CDFs). MATLAB: no counterpart.
- Category: formula/gradient (the value only)
- Proposed classification: port discrepancy (Python-only renormalization; MATLAB has none)
- Confidence: high that it happens; low severity
- History: introduced in `64dc49d` (2021-06-29); MATLAB has never had it.
- What the code does, and what it should do:
  - When both bounds lie far in the upper tail of a Gaussian prior (or of the Gaussian tail of a smooth box), the two CDF values round to 1 and the stored mass is exactly 0.
  - The log is then −inf, and `log_posterior` = +inf for every hyperparameter, so the fit's objective is −inf everywhere.
  - Short of that, the relative error of the mass is about 1e-16/mass. That is harmless, because the term is a constant.
  - Taking the upper tail through the survival function would keep it; the lower tail is already exact.
  - In the far-tail regime this contradicts the sheet's statement that the constant is invisible to the optimizer and the sampler.
- Consequence if real: it needs bounds about 8.3 standard deviations into the upper tail, that is, a prior grossly inconsistent with the bounds. PyVBMC does not reach it: its priors are Student-t with ν = 3, and their masses inside their bounds are of order 0.1 to 1.
- Suggested reproduction: `check_norm_tail.py`, N(0, 1) prior on `mean_const`.
  - Bounds [9, 10]: stored mass 0 against a true 1.129e-19, and `log_posterior` = inf.
  - Mirrored bounds [−10, −9]: stored mass 1.129e-19, exact.
- Test adequacy: no test places the bounds in a prior's upper tail.

### F3. `set_priors` accepts a smooth box or a Gaussian whose location is not finite, and the log posterior is then NaN or −inf everywhere
- Location: gpyreg/gaussian_process.py:659–684 (only `sigma` is validated), :1632–1655 (the masks) and :1592 (the normalization). MATLAB: no counterpart (`gplite_hypprior.m:35` treats a non-finite mu as uniform).
- Category: defaults
- Proposed classification: unsure (input validation or a missing feature)
- Confidence: medium
- History: Python-only.
- What the code does, and what it should do:
  - `("smoothbox", (0, inf, 1))` passes `set_priors`. It fails `sb_idx`, which requires finite a and b, falls into `g_idx` with mu = NaN, and gives `log_posterior` = NaN, even with finite bounds, where the truncated density (flat above 0) is proper.
  - `("gaussian", (inf, 1))` gives −inf everywhere.
  - Either refusing such priors, like the refusal of a bad sigma, or supporting a half-open box would avoid a silent NaN.
- Consequence if real: an unusable fit for a user who writes such a prior. PyVBMC is unaffected.
- Suggested reproduction: `check_prior_inputs.py` prints NaN, NaN, −inf.
- Test adequacy: `test_set_priors_refuses_a_scale_that_is_not_positive` covers sigma only.

There is no MATLAB-side defect in this slice beyond the two already settled: `gplite_core.m:244` and the degree-1 Matérn NaN.

## 4. Test adequacy notes

- **No likelihood-gradient test with per-point noise.** No test checks the `dsn2[:, i]·diag(Q)` branch, meaning user-provided noise, the scaled multiplier with s2 given (PyVBMC levels 1 and 2), or rectified noise. The only noise-gradient test, `test_noise_gradient_with_a_constant_total_noise`, uses a scalar total noise. `test_noise_functions.py` and `test_mean_functions.py` have no gradient checks at all.
- **No test of the low-noise representation's nlZ or gradient.** The low-noise tests cover rank-one updates only.
- **No gradient test with D = 1 or with repeated inputs.**
- **`test_gp_gradient_computations` is unseeded** and draws noise hyperparameters of order 1, so it never enters the low-noise branch. Its smooth-box coordinates lie inside the box, so the tail gradient is covered only by `test_log_likelihood_and_posterior_gradients`.
- **`test_fitting_with_fixed_bounds` misses F1**, as described above.
- **PyVBMC has no test of the GP objective's gradient in the configurations it builds.** `test_gp_hyp` checks the priors' parameters, and the `gp_nlZ` oracle pins stored values, which mirrors the implementation.

All scripts and logs are in the scratch directory `…/scratchpad/wave7/O4`: `check_lik`, `check_lownoise_mp`, `check_prior`, `check_prior_block`, `check_kernels`, `check_scalar_noise`, `check_pyvbmc_config`, `check_posterior`, `check_ladder`, `check_fixed`, `check_fixed_fit`, `check_fixed_testcfg`, `check_norm_tail` and `check_prior_inputs`, each `.py` with its `.log` (no log for `check_prior_inputs`), plus the helpers `fd_common.py` and `hp.py`.
