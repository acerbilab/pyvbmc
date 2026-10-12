# G1 — gpyreg: fit, objective, posterior factors, samplers, internal-correctness track

Raw reviewer report, wave 6 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-21. One Opus agent, fresh
context, read-only on the repositories, reading gpyreg at `fdbafdf`, whose
code under `gpyreg/` is that of the pinned revision `9e70e6b`, with PyVBMC at
`8af5daac` as its caller; it read no MATLAB source, neither `../vbmc` nor
gpyreg's copy under `matlab/`, and it did not see the comparison report of
this slice. Its brief carried two first questions, which the report answers
before its findings: whether `uuinv` of `f_min_fill.py` inverts the mixture
of uniforms in every configuration of the bounds, and whether the rank-one
extension of `update` equals a full recomputation in every state that
`update` accepts. Its check scripts ran in the session scratchpad and are
kept only on the machine that ran them (`dev/scripts/runs/LOCAL.md`). The
text below is the agent's final message, unedited.

---

# G1 — gpyreg: fit, objective, posterior factors, samplers (internal-correctness track)

## 1. Coverage

**Read completely, line by line, in `C:\Users\luigi\Documents\GitHub\gpyreg\gpyreg\`:**
- `gaussian_process.py`: `__core_computation` and `__robust_cholesky` (2612–2841), the `Posterior` class (2891–2943), `update` (763–995), `clean` (996–1020), `fit` (1021–1370), `__recompute_normalization_constants` (1373–1412), `__prior_masks` (1413–1492), `__compute_log_priors` (1493–1648), `log_likelihood` / `log_posterior` / `__compute_nlZ` / `__gp_obj_fun` (1649–1753), and the bounds/priors/hyperparameter accessors (217–762). Also `_solve_triangular` and the module-level cache flag (1–90), `_convert_shapes` (2840–2890).
- `f_min_fill.py` in full, including `uuinv`, `smoothbox_cdf/ppf` and the Student-t versions.
- `slice_sample.py` in full, including `__diagnose`, `__log_pdf_bound`, `__metropolis_step`, `__gelman_rubin`, `__effective_n`.
- `rng.py` in full.

**Read in part (as far as my slice needed):** `predict` (1755–2065, to confirm how `sW`, `L`, `sn2_mult` and `add_noise` are consumed, since `update`'s rank-one path calls it); `noise_functions.py` in full (the `s2 is None → 0` convention and the noise gradient are load-bearing for `__core_computation` and `update`); `covariance_functions.get_bounds_info` and `mean_functions.get_bounds_info` only where the `-inf` bound of F9 originates. On the PyVBMC side: `gaussian_process_train.py` (`train_gp`, `_gp_hyp`, `_get_gp_training_options`, `_get_hyp_cov`, `_lean_gp`, `_restore_gp_posteriors`), `active_importance_sampling.py`'s `SliceSampler` call and `get_mcmc_opts`, the GP options in `advanced_vbmc_options.ini`.

**Skimmed:** `quad`, `predict_full`, `random_function` (other slice; read only to locate the one caller of `__robust_cholesky`); gpyreg's `README.md`, `docsrc/source/*.rst` and release notes; the test files `test_gaussian_process.py`, `test_slice_sample.py`, `test_smoothbox.py`, `test_smoothbox_student_t.py` (read for adequacy, not run as suites).

**Read as instructed, not re-derived:** the gradient of the marginal likelihood in `__core_computation` (the `Q`, `dK`, `dsn2`, `dm` block). I checked its structure and the noise/mean terms against the standard identity ∂nlZ/∂θ = ½ tr(Q ∂C/∂θ) with Q = C⁻¹ − αα^T, and checked the noise function's own gradients, but did not re-derive the kernel derivatives.

**Did not reach:** `plot`, `formatting.py`, `isotropic_covariance_functions.py`, the MATLAB sources (per the track), anything under PyVBMC's `dev/` except the two named sections of `known_differences.md`.

**Known-differences sheet:** read "Slices G1, G2, P5 — the GP layer" and "Settled non-differences" in full, and searched the rest for `gpyreg` / `gplite`. Nothing below contradicts an entry. Two entries were confirmed rather than challenged: "The space-filling design and the optimizer are SciPy" (F7 and F9 are in that SciPy-based code, not in the substitution itself), and "The rank-one GP update is taken for a fresh observation, noisy or not" together with its note that "gpyreg's extension" is what PyVBMC relies on — F4 is about the branch of that extension a PyVBMC run does not enter.

**Interpreter check:** `gpyreg.__file__` printed `C:\Users\luigi\Documents\GitHub\gpyreg\gpyreg\__init__.py` at the head of every script. All scripts ran with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` from the scratchpad; nothing outside it was created or changed.

---

## 2. The first questions

### Q1 — `uuinv` and the placement of the design

**The inverse is exact for the mixture the body implements, in every configuration I tried, including every degenerate one — but that mixture is not the one the docstring names.**

The docstring says the mixture is `w·U(B1,B2) + ((1−w)/2)·(U(B0,B1) + U(B2,B3))`: equal weight to each tail. The body instead spreads the weight `1−w` over the *union* of the two tails **in proportion to their lengths**, through `L = (B1−B0) + (B3−B2)`. Deriving the CDF of each candidate mixture myself and testing `F(uuinv(p)) == p` over `p ∈ (0,1)` on 4001 points (`q1_uuinv.py`):

| bounds `[LB, PLB, PUB, UB]` | w | max err vs docstring | max err vs body |
|---|---|---|---|
| `[-3,-1,0.5,4]` (asymmetric) | 0, .25, .5, .9, 1 | 1.4e-1 … 1.4e-2 | ≤ 1.1e-16 |
| `[-3,-1,1,3]` (symmetric) | all | ≤ 1.1e-16 | ≤ 1.1e-16 |
| `[-1,-1,0.5,4]` (PLB = LB) | 0 … .9 | 5.0e-1 … 5.0e-2 | ≤ 2.2e-16 |
| `[-3,-1,4,4]` (PUB = UB) | 0 … .9 | 5.0e-1 … 5.0e-2 | ≤ 2.2e-16 |
| `[-3,0.5,0.5,4]` (zero-width plateau) | all | = w (an atom) | = w (an atom) |
| `[-1,-1,2,2]` (both tails empty, `L == 0`) | all | = (1−w)/2 (atoms) | = (1−w)/2 (atoms) |

The two conventions coincide exactly when the tails have equal length, and the residual in the two "atom" rows is the unavoidable artifact of inverting a CDF that has a jump, not an inversion error: the map is monotone and inside `[B0,B3]` in every row, and continuous at both breakpoints (I checked `x(p1) = B1`, `x(p2) = B2`, `x(1) = B3` symbolically and numerically). `w = 0` and `w = 1` are both handled: at `w = 0` the `i2` mask is provably empty so no gap opens, and at `w = 1` the early return is the plain uniform over the plausible box. The `L == 0` branch (both tails of zero width) switches to the equal-split convention, `(1−w)/2` as an atom on each hard bound — the only consistent choice there, since length-weighting is `0/0`.

**Where the points land.** The per-coordinate probability of the plausible box is exactly `w` under both conventions, so the intent behind `w = 0.5**(1/n_vars)` ("Half of all starting points from inside the plausible box") is met *per uniform coordinate*. It is not met overall. In a PyVBMC-shaped fit (D = 2, SE-ARD + negative quadratic + constant noise, Student-t priors on the length scales and the noise scale, 1024 Sobol points — `q1_design.py`):

- the six coordinates that take the `uuinv` path (output scale, mean constant, two mean locations, two mean log scales) land in the plausible box with frequency 0.9277 against `w = 0.92587` — as intended;
- the three coordinates with Student-t priors land there with 0.945–0.977 — a different probability, because they take a different branch;
- **the fraction of design points with *every* coordinate inside the plausible box is 0.559, not 0.5.**

And the length-weighting is live in that run, not academic: `covariance_log_outputscale` has `LB=-11.7, PLB=-4.79, PUB=2.12, UB=4.42`, so the lower tail is three times the longer of the two, and the design puts 5.47% below PLB against 1.76% above PUB. The docstring's mixture would put 3.7% on each side.

The distribution functions themselves are right: `smoothbox_ppf`/`smoothbox_cdf` and the Student-t pair round-trip to 2.2e-16 over `q ∈ (0,1)` for σ ∈ {1e-3, 0.3, 1, 2} and both `a < b` and `a == b`, the densities integrate to 1.0000000000, and the CDFs match quadrature of the density I wrote from the definition to 1e-10. I derived the plateau height `h = 1/(Cσ√(2π))` with `C = 1 + (b−a)/(σ√(2π))`, and the Student-t analogue `h = Γ((ν+1)/2)/(Γ(ν/2)·C_t·σ√(νπ))`, and both match the code's `C` exactly.

Reported as **F12** (documentation against implementation) with the proportion mismatch as minor observation M-A.

### Q2 — the rank-one extension against a full recomputation

**Yes in every state I could construct, to rounding — except in the low-noise branch, where the shortcut is unguarded and can be wrong by O(1) (F4).**

I derived the extension myself. For `L_new = [[L, c],[0, d]]` with `L_new^T L_new = C_new/sl`, `c = L^{-T}k_*/sl` and `d = √(sl(k_{**} + σ²_eff) − u^Tu)/sl` with `u = L^{-T}k_*`; the code's `new_L_column/sl` and `√(sqrt_arg)/sl` are exactly these. For `alpha`, block inversion gives top `α + b(m_* − y_*)/v` and bottom `(y_* − m_*)/v` with `b = C⁻¹k_*`, `v = k_{**} + σ²_eff − k_*^TC⁻¹k_*`; the code's single concatenated expression is exactly that, and `alpha_update` equals `b` in both representations. For the low-noise branch, `−C_new^{-1}` expands to the code's `[[L + v·b^T, −v],[−v^T, −1/v]]` with `v = −b/v_star`. `v_star` from `predict(..., add_noise=True)` is `k_{**} + σ²·sn2_mult − k_*^TC⁻¹k_*`, the quantity the derivation needs, so the two halves agree.

Numerically (`q2_rank_one.py`, `q2_analytic.py`, `q2_edge.py`, `q2_edge2.py`), comparing both against a fresh GP fitted on the enlarged set and against `C = K + sn2_mult·diag(σ²)` and `α = C⁻¹(y − m)` built from scratch:

| state | worst relative error |
|---|---|
| constant noise / +user `s2` / +scaled user `s2`, 1 and 3 hyperparameter samples, `L_chol` True | α ≤ 9.3e-14, represented C ≤ 6.6e-16, `predict` μ ≤ 3.7e-14, s² ≤ 4.4e-16 |
| same with near-duplicate training rows | unchanged |
| `s2_new = None` on a GP that carries `s2` (the zero back-fill) | α ≤ 1.2e-15 |
| `sn2_mult` = 10, 100, 1000 (Cholesky retry forced by a patched `sp.linalg.cholesky`), 1 and 3 samples, retried sample mixed with unretried ones | α ≤ 7.4e-14, C ≤ 4.0e-16 |
| five consecutive single-point updates, 3 samples | α ≤ 8.5e-14 |
| `update(X_new, y_new, hyp=new)` | full recomputation taken, α ≤ 8.2e-14 |
| a new point whose `s2` is 4 orders of magnitude below every existing one | α ≤ 1.7e-16, `predict` μ 5.6e-17, s² 2.2e-16 |
| `sqrt_arg ≤ 0` fallback (exact duplicate, output scale `e^28`, σ² at the `L_chol` boundary), mixed with a rank-one sample | warns and reverts; α 2.4e-15, μ 3.3e-15 against a fresh fit |
| **`L_chol` False, exact duplicate, σ² = eps** | **α relative error 1.84, predictive variance off by 10x, no warning** |
| `L_chol` False, σ² ≈ 1.5e-8, duplicate | α 1.7e-6 (conditioning of the explicit inverse) |

Two things that look like discrepancies and are not. First, the stored `sl` and `sW` keep the *old* noise scale: after appending a point with `s2 = 1e-4` to a set with `s2 = 1`, the rank-one path holds `sl = 1.0067` where a fresh fit computes `sl = 0.006838`. `L` and `sW` therefore differ from a fresh fit's, but `sl·L^TL` and every prediction agree to 1e-16, because `predict` consumes `sW` and `L` only through `sW·Ks` and `L^{-T}`. This is what the `Posterior` docstring states, and it holds. Second, a state `update` *enters* but cannot handle: posteriors emptied by `clean()` or built with `compute_posterior=False` still take the rank-one shortcut and raise `TypeError` (F8).

---

## 3. Findings

### F1. The prior type masks test `df == 0 | ~np.isfinite(df)`, which Python parses as `df == (0 | ~isfinite(df))`
- Location: `gpyreg/gaussian_process.py:1474` (`sb_idx`), `:1494` (`g_idx`); MATLAB: "not read (internal track)"
- Category: formula/gradient
- Proposed classification: suspected defect
- Confidence: high
- `|` binds tighter than `==` in Python. `0 | ~np.isfinite(df)` evaluates to the boolean array `~isfinite(df)`, so the expression becomes: where `df` is finite, `df == False`, i.e. `df == 0` (the intended test); where `df` is not finite, `df == True`, i.e. `df == 1`, which is never true. The intent — visible in `set_priors`' own comment "Implicit flag for gaussian, is set to inf later", in `get_priors:456` which decodes `df[i] == 0 or df[i] == np.inf` as `"gaussian"`/`"smoothbox"`, and in `__recompute_normalization_constants:1397` which correctly writes `if df == 0 or not np.isfinite(df)` with Python's `or` — is that an infinite `df` also means "Gaussian". A coordinate with `df = inf` therefore falls out of `sb_idx`, out of `g_idx` (same broken test), out of `sb_t_idx` and `t_idx` (both require `isfinite(df)`) and out of `u_idx` (σ is finite): **no mask claims it, so it contributes no prior density at all**, while `__recompute_normalization_constants` still computes and `__compute_log_priors:1646` still subtracts its truncation constant. The same holds for `df = NaN` when `__compute_log_priors` is reached without going through `fit` (which fills NaN with `df_base`).
- Consequence if real: a hyperparameter whose prior is encoded with `df = inf` is sampled and optimized as if it had no prior, offset by a spurious `−log Z`. The hyperparameter posterior of that coordinate is then the likelihood alone, truncated to the bounds. A default PyVBMC run does not reach it: `_gp_hyp` sets only `"student_t"` priors with `df = 3`, and a `get_priors`→`set_priors` round trip re-encodes a Gaussian as `df = 0`, which takes the correct path. The reachable states are a caller that writes `hyper_priors["df"]` directly, and any future code that acts on the `set_priors` comment.
- Suggested reproduction: ran it (`q3_priors.py`). With a `("gaussian", (0, 1))` prior on `mean_const`, `log_posterior − log_likelihood` is `−0.9189385332` with `df = 0` and exactly `0.0000000000` after setting that entry of `df` to `inf`; `−0.9189385332` is `−½log(2π)`, the whole Gaussian term. The masks alone: `(np.array([inf, nan]) == 0 | ~np.isfinite(...))` gives `[False False]` where `(df == 0) | ~isfinite(df)` gives `[True True]`.
- Test adequacy: no. `test_prior_mask_cache_follows_priors_and_bounds` and `_small_gp_with_priors` use `df ∈ {0, 3, 4}` only; no test sets an infinite `df`, and `get_priors`' acceptance of `df == np.inf` is untested.

### F2. The smoothbox normalization constant is not masked to the coordinates it multiplies
- Location: `gpyreg/gaussian_process.py:1541-1552` and `:1567-1580` (smoothbox), `:1594-1608` (smoothbox Student-t); the constants are built at `:1505` and `:1509`; MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high
- `masks["C_sb"]` has one entry per smoothbox coordinate (it is built from `a[sb_idx]`, `b[sb_idx]`, `sigma[sb_idx]`). It is then used as `np.log(C**2 * 2*np.pi * sigma[sb_idx_b | sb_idx_a]**2)` and `np.log(C * sigma[sb_idx_btw])`, where the right-hand factors hold one entry per smoothbox coordinate *currently on that side of the box*. The three masks `sb_idx_b`, `sb_idx_a`, `sb_idx_btw` partition `sb_idx`, so whenever two or more smoothbox coordinates are split between a tail and the plateau, `C` is longer than what it multiplies. NumPy then either broadcasts (when the shorter side has exactly one entry), summing `n_sb` terms instead of one, or raises. `C` should be subscripted with the same mask, e.g. `C[sb_idx_btw[sb_idx]]`. The same holds for `C_sb_t` in the Student-t branch. The gradient branches do not use `C` and are unaffected.
- Consequence if real: the log prior of a smoothbox or smoothbox-Student-t prior spanning two or more hyperparameters is silently wrong, or the evaluation raises mid-fit. A default PyVBMC run does not reach it (`_gp_hyp` sets only Student-t priors), but `"smoothbox"` and `"smoothbox_student_t"` are documented values of `set_priors`, `f_min_fill` has a dedicated branch for them, and a smoothbox prior on `covariance_log_lengthscale` is exactly a multi-entry block for D ≥ 2.
- Suggested reproduction: ran it (`q8_sb.py`). Smoothbox `(a,b,σ) = (−1,1,0.7)` on `covariance_log_lengthscale`, bounds ±10, evaluated against the log prior I wrote from the definition. All coordinates on the plateau or all in one tail: agreement to <1e-9 for D = 1…4. **D = 2 with one coordinate below `a` and one on the plateau: `−7.332786` against the correct `−3.666393`, exactly twice the right value.** D = 3 and D = 4 with the same split: `ValueError: operands could not be broadcast together with shapes (3,) (2,)`. Smoothbox-Student-t behaves the same way.
- Test adequacy: no, and the one test that touches these branches cannot catch it. `_small_gp_with_priors` (`test_gaussian_process.py:1617`) puts the smoothbox on `noise_log_scale` and the smoothbox-Student-t on `mean_const`, both single-entry blocks — the one configuration in which `C` always has length 1 and broadcasts correctly. `test_smoothbox.py` and `test_smoothbox_student_t.py` test the distribution functions in `f_min_fill`, never `__compute_log_priors`.

### F3. `__robust_cholesky`'s sign-fixing flips individual matrix entries, not whole eigenvectors, so the returned factor no longer reproduces the input
- Location: `gpyreg/gaussian_process.py:2620-2624`; MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high
- `maxidx = np.argmax(np.abs(U), axis=0)` is the row index of the largest-magnitude entry of each *column*, shape `(n,)`. `U[maxidx]` then indexes **rows** of `U` by those indices and has shape `(n, n)`, so `negidx` is a full boolean matrix and `U[negidx] *= -1` negates scattered individual entries. The intent is the usual sign convention — make each eigenvector's largest component positive — which needs a per-column mask, `U[maxidx, np.arange(n)] < 0` followed by `U[:, mask] *= -1`. Done per column the step is a mathematical no-op (`U D Uᵀ` is invariant under a column sign flip), which is why it is harmless when correct; done per entry it destroys `U D Uᵀ = Σ`, and the returned `T = diag(√D) · real(U[:,t])ᵀ` then satisfies `TᵀT ≠ Σ`.
- Consequence if real: when `sp.linalg.cholesky` fails and this fallback is taken, `GP.random_function` draws its sample functions from the wrong covariance. The only caller is `random_function:2593`, and the fallback fires only on a non-positive-definite input — a symmetrized posterior covariance at closely spaced test points is a realistic case. A default PyVBMC run does not reach it: the brief states `random_function` is not called, and nothing else in gpyreg or PyVBMC calls `__robust_cholesky`.
- Suggested reproduction: ran it (`q4_misc.py`). On a rank-deficient PSD matrix `A Aᵀ` with `A` of shape (5,3), `max|TᵀT − Σ| = 2.804` against `‖Σ‖ ≈ 3.6`. Step by step on the same matrix: before the flip `max|U diag(D) Uᵀ − Σ| = 4.4e-15`, after the flip as written `3.871`, after a per-column flip `4.4e-15`. (A diagonal input is unaffected, because its eigenvectors are unit vectors.)
- Test adequacy: no. `test_random_function` draws functions with `np.random` (unseeded) and asserts nothing about them beyond output shapes via `predict(..., return_lpd=True)`; no test constructs an input that makes the Cholesky fail, so the fallback has no coverage at all.

### F4. The rank-one update has no stability guard in the low-noise branch
- Location: `gpyreg/gaussian_process.py:915-925` (the `else:  # Low-noise parametrization` block), against the guard at `:887-908`; MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- The `L_chol` branch tests `sqrt_arg <= 0`, warns, and marks the sample for a full recomputation. The `L_chol == False` branch has no analogous test: it divides by `v_star[:, s]` unconditionally, where `v_star` is the predictive variance `predict` returned — and `predict:2016` clamps the latent variance at zero (`s2 = np.maximum(s2, 0)`) before adding the noise, so when the exact latent variance is at or below rounding level, `v_star` is not the quantity the derivation requires. With an explicitly stored `−C⁻¹` the loss is much worse than in the factored branch, and nothing reports it. (The clamp also introduces a small inconsistency in the `L_chol` branch, where `alpha` uses the clamped `v_star` while `L` uses the independently computed `sqrt_arg`; there the residual is of order `eps·k_**` — see minor observation M-I.)
- Consequence if real: appending a point at or very near an existing training input to a GP whose minimum noise variance is below `1e-6` corrupts `alpha`, and hence every posterior mean, silently. A default PyVBMC run does not reach it: `_gp_hyp` sets `bounds["noise_log_scale"] = (log(min_noise), …)` with `min_noise = tol_gp_noise = sqrt(1e-5)`, so the total noise variance is at least `1e-5 > 1e-6` and `L_chol` is always True. It is reachable from gpyreg standalone, where the recommended noise lower bound is `log(1e-6)` and the branch is the documented representation for low noise. Note that PyVBMC's active sampling does take single-point `gp.update` calls and does re-evaluate at existing inputs for noisy targets, so only the noise floor stands between a run and this branch.
- Suggested reproduction: ran it (`q2_edge2.py`). `GaussianNoise()` (σ² = `np.spacing(1)` = 2.22e-16, so `L_chol` False), 10 points, then `update` with an exact copy of an existing row: no warning, and against a fresh fit on the same 11 points `alpha` has relative error **1.84** and the predictive variance **10.4**. The same construction with `L_chol` True (σ² at the `1e-6` boundary, output scale `e^28`) warns "Rank-one update of Cholesky factor unstable for posterior 0. Reverting to full update." and then matches a fresh fit to 3e-15.
- Test adequacy: no. `test_update_one_point_with_new_hyperparameters`, `test_rank_one_update_with_heteroskedastic_noise`, `test_rank_one_update_without_stored_noise_scale` and `test_update_aligns_user_provided_noise` all use `log(0.1)` or `log(0.05)` noise and assert `post.L_chol and post_ref.L_chol`; every one of them uses a single hyperparameter sample and `sn2_mult == 1`. Neither the low-noise branch of the rank-one path nor the `sqrt_arg <= 0` fallback has any test.

### F5. `fit` reads the sampler from `options["sampler"]`, while its docstring documents `sampler_name`
- Location: `gpyreg/gaussian_process.py:1128` (`sampler_name = options.get("sampler", "slicesample")`) against the docstring at `:1073-1075`; MATLAB: "not read (internal track)"
- Category: defaults
- Proposed classification: suspected defect
- Confidence: high
- The documented option name is `sampler_name`; the code reads `sampler`. A caller using the documented spelling has the value silently ignored and gets slice sampling. The variable `sampler_name` then holds the value of the undocumented key, which is checked at `:1341` (`if sampler_name != "slicesample": raise ValueError("Unknown sampler!")`) and, at `:1206`, against the string `"laplace"` — a value that can never survive the later check, so that half of the condition is dead.
- Consequence if real: a documented option has no effect. A default PyVBMC run is unaffected, because `_get_gp_training_options` writes `gp_train["sampler"] = "slicesample"` — the key the code reads — and the sheet's entry "Slice sampling is the only sampler of the GP hyperparameters" records that PyVBMC refuses every other value at construction anyway. It affects gpyreg's own documented surface, and any caller that types the documented name to select or verify a sampler.
- Suggested reproduction: ran it (`q3_priors.py`). `gp.fit(..., options={..., "sampler_name": "does_not_exist"})` completes normally; `options={..., "sampler": "does_not_exist"}` raises `ValueError: Unknown sampler!`.
- Test adequacy: no. `test_fitting_options` exercises eight option dictionaries, none of which names a sampler under either spelling.

### F6. `log_likelihood` and `log_posterior` cannot take the dictionary their docstrings document
- Location: `gpyreg/gaussian_process.py:1666-1670` and `:1699-1703` (the `isinstance(hyp, dict)` branches), with `hyperparameters_from_dict:720`; MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high
- Both docstrings say `hyp` is "Either an 1D array or a dictionary of hyperparameters", and both convert a dict with `hyperparameters_from_dict`, which returns shape `(hyp_samples, hyp_N)` — `(1, hyp_N)` for one dict. `__core_computation` then slices that array along axis 0 (`hyp[0:cov_N]`, `hyp[cov_N:cov_N+noise_N]`, …), so the covariance block becomes the whole row-array and the noise and mean blocks become empty. The conversion needs a `[0]` (or `np.ravel`), as `_small_gp_with_priors` does for its own fixture.
- Consequence if real: the documented dict form of two public methods raises instead of computing. Not reached by a default PyVBMC run, which calls neither method.
- Suggested reproduction: ran it (`q4_misc.py`). `gp.log_likelihood(gp.hyperparameters_to_dict(hyp)[0])` raises `ValueError: Expected 5 mean function hyperparameters, 0 passed instead.`; with `compute_grad=True`, `ValueError: Expected 1 noise function hyperparameters, 0 passed instead.`. The same array via `gp.log_likelihood(hyp)` returns `-18.457432569129455`. `log_posterior` behaves identically.
- Test adequacy: no. `test_log_likelihood_and_posterior_gradients` and the two `test_fitting`-family call sites all pass a 1-D array; the dict branch is never executed.

### F7. `fit` overwrites a row of the space-filling design in place, and the sampler widths are then taken from the altered design
- Location: `gpyreg/gaussian_process.py:1234` (`hyp = X0[0 : np.maximum(opts_N, 1), :]`), `:1248` (`hyp[1, :] = xx[idx_best, :]`), `:1251` (`widths_default = np.std(X0, axis=0, ddof=1)`); MATLAB: "not read (internal track)"
- Category: state/caching
- Proposed classification: suspected defect
- Confidence: high
- `hyp` is a basic slice of `X0`, hence a view. Writing the low-noise starting point into `hyp[1, :]` therefore replaces row 1 of `X0` with a copy of a later design row — the design point that was there is lost — and `widths_default`, computed three lines later from `X0`, is the standard deviation of that altered design. The intended write is to the starting-point table, not to the design.
- Consequence if real: the slice sampler's default widths are perturbed, which changes the hyperparameter chain and so the whole trajectory of a fit. The size is small (one row of `init_N`) but the effect is not nothing. It is reached by **gpyreg's own defaults**: the write needs `noise_N > 0 and 1 < opts_N < init_N`, true at `opts_N = 3`, `init_N = 1024`, and `widths_default` is used whenever `n_samples > 0`, the default 10. It is **not** reached by a default PyVBMC run: `_get_gp_training_options` sets `opts_N = 1` whenever `gp_s_N > 0`, and sets `opts_N = 2` only when `gp_s_N == 0`, in which case `fit` returns at `:1334` before `widths_default` is read.
- Suggested reproduction: ran it (`q9_alias.py`). Wrapping `f_min_fill` to keep a copy of what it returned: after `gp.fit(..., {"init_N": 256, "opts_N": 3})`, row 1 of the returned design has been overwritten, and `np.std(X0, axis=0, ddof=1)` differs before and after by up to **6.1e-2** (on the noise coordinate, whose width is ~0.42 — about 15%).
- Test adequacy: no. No test inspects the design array after `fit`, nor the widths handed to the sampler.

### F8. `update` takes the rank-one shortcut on posteriors that hold no factors, contradicting `clean`'s docstring
- Location: `gpyreg/gaussian_process.py:800-810` (the `rank_one_update` decision) and `:996-1020` (`clean`); MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- `clean`'s docstring says the auxiliary structures "can be reconstructed with a call to `update` with `compute_posterior=True`". The `rank_one_update` test requires only that `X`, `y`, `X_new` and `y_new` are present, `X_new` has one row and `hyp` is None; it does not check that the existing posteriors carry their factors. After `clean()`, or after `update(..., compute_posterior=False)`, a single-point `update` therefore enters the shortcut with `alpha = L = sW = sn2_mult = None` and dies inside it. The same condition should require a computed posterior and fall through to the full recomputation otherwise.
- Consequence if real: the documented way to rebuild the factors fails for the one shape of call — appending a single observation — that the memory-saving workflow makes natural. A default PyVBMC run does not reach it: PyVBMC never calls `clean()`, and `_lean_gp` deliberately uses `update(hyp=..., compute_posterior=False)` rather than `clean()` with a comment explaining why, while `_restore_gp_posteriors` rebuilds through `update(hyp=...)`, which takes the full path.
- Suggested reproduction: ran it (`q2_edge2.py`). After `gp.clean()`, `gp.update(X_new=np.array([[0.2, 0.2]]), y_new=np.array([[0.1]]))` raises `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`. `gp.update(compute_posterior=True)` on the same cleaned GP restores the factors correctly.
- Test adequacy: no. `test_cleaning` calls `gp.update(compute_posterior=True)` with no data, the one call that works.

### F9. `fit` hands scipy a `(-inf, -inf)` bound pair, which raises `KeyError` from inside L-BFGS-B
- Location: `gpyreg/gaussian_process.py:1312-1317` (`bounds=list(zip(LB, UB))`), with `get_recommended_bounds:419` (`ub = np.maximum(lb, ub)`); MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- `get_recommended_bounds` guarantees `lb <= ub` with `np.maximum` but not that the pair is usable: when the components' recommended upper bound is `-inf`, the pair stays `(-inf, -inf)`. `scipy.optimize._lbfgsb_py` looks that pair up in a dictionary keyed by finiteness patterns and raises `KeyError` rather than a diagnosable error. A training set whose targets are all equal produces it: `height = max(y) - min(y) == 0`, so `covariance_functions.get_bounds_info` writes `np.log(height * 10) = -inf` into the output-scale upper bound (and the noise function does the same at `noise_functions.py:131`).
- Consequence if real: `fit` fails with an opaque `KeyError` deep inside scipy on a degenerate but legal training set, instead of either a clear error or a usable bound. A default PyVBMC run reaches it only for a target whose log joint is constant across the training inputs — a uniform prior over the box together with a flat log-likelihood, which is a plausible smoke test rather than a real inference problem. The origin of the `-inf` is in the component modules (another slice); what is actionable here is that neither `get_recommended_bounds` nor `fit` rejects or repairs a pair no optimizer accepts.
- Suggested reproduction: ran it (`q7_degen.py`). 12 random inputs in D = 2 with `y = np.full((12,1), 1.3)`: `gp.fit(...)` raises `KeyError: (np.float64(-inf), np.float64(-inf))` at `_lbfgsb_py.py:388`, with `gp.lower_bounds[2] = gp.upper_bounds[2] = -inf`. With two distinct target values the same fit completes and all bounds are finite.
- Test adequacy: no. `test_fitting_options` and `test_fitting` use `1 + np.sin(X)` and a random GP draw, both with a non-degenerate range; no test fits a constant target.

### F10. `SliceSampler` validates the widths after replacing infinities, but keeps the infinity in `base_widths`
- Location: `gpyreg/slice_sample.py:186` (`self.base_widths = self.widths.copy()`), `:188` (`self.widths[np.isinf(self.widths)] = 10`), `:213-220` (the check), `:590-592` (the geometric-mean recombination); MATLAB: "not read (internal track)"
- Category: defaults
- Proposed classification: suspected defect
- Confidence: high
- `base_widths` is copied from the user's widths *before* infinities are replaced by 10, and the validity check "The widths vector needs to be all positive real numbers" runs *after* the replacement, so an infinite user width passes validation. At the end of burn-in, `self.widths = np.maximum(new_widths, np.sqrt(new_widths * self.base_widths))` reintroduces the infinity, and the coordinate sweep then computes `xprime[dd] = rand * (inf - (-inf)) + (-inf)`, which is NaN. The check should run on the input, or `base_widths` should be copied after the replacement.
- Consequence if real: the chain fills with NaN from the first post-burn-in iteration, the target reports "Target density function returned NaN. Trying to continue." on every evaluation, and the recorded samples are unusable — silently, since `sample` returns normally. Not reached by a default PyVBMC run, nor by gpyreg's own `fit`: `fit` passes `np.minimum(widths, widths_default)` with `widths_default` a standard deviation or `PUB - PLB`, both finite, and PyVBMC's `active_importance_sampling` passes `np.std(gp.X, axis=0, ddof=1)`.
- Suggested reproduction: ran it (`q4_misc.py`). `SliceSampler(target, x0=[0,0], widths=[inf, 1.0], LB=[-inf,-5], UB=[inf,5])` constructs without complaint, `widths` is `[10, 1]` and `base_widths` is `[inf, 1]`; after `sample(20, burn=10)`, `widths` is `[inf, 3.813]` and the recorded samples contain non-finite values.
- Test adequacy: no. `test_init_sanity_checks` covers non-positive and complex widths, not infinite ones.

### F11. A free parameter whose chain never moved is not reliably flagged, contrary to the documented diagnostics
- Location: `gpyreg/slice_sample.py:683-685` and the comment at `:676-682`, against the `R` and `eff_N` docstrings at `:313-326`; the arithmetic is at `:834-853` and `:905-943`; MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: possibly intentional
- Confidence: high on the behavior, medium on the classification
- `__diagnose` detects a frozen free parameter by testing whether `R` and `eff_N` are non-finite, and the docstring states that `R` "is not finite for a parameter whose chain is constant within each half of the sampled sequence". That holds only when the constant is exactly preserved by averaging. For any other constant, `np.mean` carries a rounding error, so the within-chain variance `W` and the between-chain term are of order 1e-33 rather than zero, their ratio is an ordinary finite number, and `undefined` stays empty.
- Consequence if real: the "did not move" message and `exit_flag = -3` are replaced by whatever the rounding ratio happens to give. In the case I measured the parameter was still caught, by the low-effective-sample route with `exit_flag = -1`; but `R` came out **0.9487**, which passes both the `R > 1.5` and `R > 1.1` tests, so the classification of the failure rests on `eff_N` alone and on rounding. Neither gpyreg's `fit` nor PyVBMC reaches it: both pass `diagnostics: False` (`gaussian_process.py:1345`, `active_importance_sampling.py:430`), and PyVBMC reads only `samples` and `f_vals` from the result. It affects a direct `SliceSampler` user who turns the diagnostics on.
- Suggested reproduction: ran it (`q6_last.py`). Feeding `__diagnose` a 20-sample constant trace: for 0.0, 1.0, 0.5, 0.25, 2.0 and −7.0 → `R = nan`, `eff_N = nan`, `exit_flag = -3`; for **0.3 and 1e-3 → `R = 0.9487`, `eff_N = 1.0526`, `exit_flag = -1`**; for `log(10)` → `R = inf` with a `RuntimeWarning: divide by zero`, `exit_flag = -3`.
- Test adequacy: no, and both relevant tests pass for a reason that does not generalize. `test_constant_trace_in_a_free_parameter_fails_the_diagnostics` builds the trace as `np.zeros((20, 1))`, and `test_fixed_parameter_is_left_out_of_the_diagnostics` fixes the parameter at exactly `1.0`; 0 and 1 are the two constants whose mean is exact, so both get the `nan` path. Substituting 0.3 for either constant would fail the assertions.

### F12. `uuinv`'s docstring names a different mixture from the one it implements
- Location: `gpyreg/f_min_fill.py:193-214` (the docstring) against `:218-250` (the body); the caller's intent is at `:127-136`; MATLAB: "not read (internal track)"
- Category: formula/gradient
- Proposed classification: suspected defect (in the docstring)
- Confidence: high
- The docstring states the mixture is `w·U(B1,B2) + ((1−w)/2)·(U(B0,B1) + U(B2,B3))`. The body builds `L = (B1−B0) + (B3−B2)` and spreads `1−w` over the union of the two tails in proportion to their lengths, which is a different distribution whenever the tails differ in length. The two agree only for symmetric bounds. In the extreme case `PLB == LB`, the docstring's mixture puts an atom of mass `(1−w)/2` on the hard lower bound while the body puts nothing there and gives the whole `1−w` to the upper tail.
- Consequence if real: a reader specifying or reviewing the initial design has the wrong distribution in hand; anything derived from the docstring (a reimplementation, a MATLAB comparison, an estimate of how much of the design lies near a hard bound) is wrong by up to `(1−w)/2` per tail. **The difference is live in a default PyVBMC run**: the output scale, the mean constant and the mean length scales all take this path with markedly asymmetric tails. The implemented behavior is the defensible one — putting the outer weight where there is actually room — so the docstring is what should change; I have not read MATLAB's `uuinv.m` and take no position on which side matches it.
- Suggested reproduction: ran it (`q1_uuinv.py`, `q1_design.py`). `F_docstring(uuinv(p))` deviates from `p` by up to 0.50 for `[LB,PLB,PUB,UB] = [-1,-1,0.5,4]` while `F_body(uuinv(p))` matches to 2.2e-16; over 200001 quantiles with `w = 0.25` the tail masses are 0.0000 / 0.7500 where the docstring requires 0.3750 / 0.3750. In the realistic fit, `covariance_log_outputscale` receives 5.47% below PLB and 1.76% above PUB, in the 3:1 ratio of the tail lengths.
- Test adequacy: the test asserts the implementation, not the documented specification — see the test-adequacy notes below.

### Minor observations

- **M-A. The "half from inside the plausible box" comment holds only per uniform coordinate.** `f_min_fill.py:130-132` names `w = 0.5**(1/n_vars)` so that half the design points lie wholly inside the plausible box, and `n_vars` counts every hyperparameter. Coordinates with a prior take a different branch and a different probability, so the overall fraction is not 0.5: measured 0.559 in a PyVBMC-shaped fit with 3 of 9 coordinates under Student-t priors. Design intent, not arithmetic, but the comment overstates what the code achieves.
- **M-B. The Metropolis option key is misspelled, so the feature is dead.** `slice_sample.py:238` reads `options.get("metopolis_rnd", None)` (no `r`), so `metropolis_flag` is False for a caller passing `metropolis_rnd` and the `__metropolis_step` calls at `:423` and `:548` never run. Verified: the documented-looking spelling gives `metropolis_flag == False`, the misspelled one `True`. The option is not in the `options` docstring at all; nothing in gpyreg or PyVBMC sets it.
- **M-C. `uuinv`'s out-of-range guard is skipped by two of its three return paths.** `x[p < 0] = nan; x[p > 1] = nan` (`f_min_fill.py:252-253`) sits only in the general path. `uuinv(np.array([-0.1]), [-3,-1,0.5,4], w=1)` returns `-1.15`, outside `[B0, B3]`; the `L == 0` path returns `B[0]`.
- **M-D. `set_priors`' documented `ValueError` for an unknown hyperparameter does not exist.** The docstring promises "Raised when `priors` is given, but a specified hyperparameter is unknown", but the loop only looks up the names it knows, so `priors["not_a_hyperparameter"] = …` is accepted silently. `set_bounds` is the same (and does not document the case).
- **M-E. `get_recommended_bounds` rough edges.** A `tuple` argument raises `AttributeError: 'tuple' object has no attribute 'copy'` although the code tests `isinstance(..., (list, tuple, np.ndarray))` and the docstring says "array_like"; the `ValueError` message in the `upper_bounds` branch (`:376`) names `` `lower_bounds` ``; and `ub = np.maximum(lb, ub)` silently repairs an inverted user bound rather than reporting it.
- **M-F. σ is validated nowhere and treated inconsistently.** `set_priors` accepts a negative or zero `sigma`; `__prior_masks` takes `np.abs(sigma)` while `f_min_fill.py:169-177` uses the raw value, so a negative σ would give a prior and a design drawn from different distributions.
- **M-G. `f_min_fill` drops provided starting points without evaluating them when `N <= N0`.** With `x0` of 4 rows and `N = 2` it returns 2 rows after 2 objective calls; the other two provided points are never evaluated and never returned. In `fit` this can only happen if `hyp0` has more rows than `init_N`.
- **M-H. `fit` fills `hyper_priors["df"]` permanently.** `:1146` writes `df_base` into every NaN entry of the GP's own prior dict, so a second `fit` with a different `df_base` has no effect on coordinates already filled. Not documented as a side effect.
- **M-I. `predict`'s clamp of the latent variance feeds the rank-one update.** `predict:2016` applies `np.maximum(s2, 0)` before adding the noise, and `update:838` uses that `v_star` for `alpha` while the `L_chol` branch computes `sqrt_arg` independently. When the clamp bites, `alpha` and `L` rest on slightly different variances. The residual is of order `eps·k_**`; the serious case is the unguarded branch of F4.
- **M-J. Two diagnostics conventions differ from the references the docstrings name.** `__diagnose:667-672` splits an odd number of samples by dropping the *last* sample rather than the middle one; `__effective_n:923-932` pairs the autocorrelations as `(ρ₀,ρ₁), (ρ₂,ρ₃), …` where BDA3 and Stan pair `(ρ₁,ρ₂), (ρ₃,ρ₄), …`, and since `ρ₀ = 1` the first pair is effectively always accepted. Both are stated accurately in the `__effective_n` docstring, so this is a note on the convention, not a mismatch with the code's own documentation.
- **M-K. Dead and cosmetic details in `fit`.** `if s_N > 0 and sampler_name != "laplace"` (`:1206`) — the second clause can never be false at the point the value survives to `:1341`. `nll = np.full((N,), -np.inf)` in the `init_N == 0` branch (`:1226`) initializes to `-inf` where `+inf` is the sensible sentinel (every entry is overwritten). `hyperparameters_to_dict:713-720` rebinds its loop variable `i` inside the loop; it works because `for i in range(...)` reassigns at the top of each pass, but it is one edit away from breaking. I verified the multi-sample round trip is exact.
- **M-L. `sp.special.gamma` in the smoothbox-t normalizers overflows for large `df`.** `:1509` and `f_min_fill.py:307`, `:370` use `gamma` rather than `gammaln`, so `C_sb_t` becomes NaN for `df` above roughly 340. Unreachable through `f_min_fill`, which caps `df` at 3; reachable through `__prior_masks` and `__recompute_normalization_constants` with a user-set `df`.
- **M-M. `update` on a GP with no hyperparameters reports the wrong thing.** `GP(...); gp.update(X_new=X, y_new=y)` takes the full path with `hyp` full of NaN and surfaces as `LinAlgError("Singular matrix for L Cholesky decomposition")` after ten Cholesky retries, rather than saying no hyperparameters are set.

### What I checked and found correct

Worth recording, so a later reader does not repeat it. The **value of the negative log marginal likelihood** matches `½(y−m)ᵀC⁻¹(y−m) + ½log|C| + (N/2)log2π` computed independently, for all three noise parameterizations and for `L_chol` both True and False, to 1.2e-14 at moderate noise (the 1e-8 residuals at σ² = 1e-12 are the conditioning of my reference, not of the code): the `sl` bookkeeping, `logdet = Σlog diag(L)` and the `N log(2π·sl)/2` term are all consistent. The **posterior factors** satisfy their documented definitions exactly (`alpha = C⁻¹(y−m)`, `sW = 1/√sl`, `L` the factor of `C/sl` or `−C⁻¹`, `sl = min(sn2)·sn2_mult` at computation time), and the retry ladder keeps `A`, `sl` and `sn2_mult` mutually consistent at every rung (verified at `sn2_mult` = 10, 100, 1000). The **log-prior densities and gradients** are right for all four prior types: I derived the smoothbox and smoothbox-Student-t normalizations from scratch and they match `C_sb`, `C_sb_t` and the plateau/tail terms; finite differences agree to 6e-10 for all four types away from `a` and `b`, and the 5e-7 residual exactly at a breakpoint is `h/(4σ²)`, the expected second-order artifact of a discontinuous third derivative, not an error (the density is C¹ there and the code's zero gradient on the plateau is the correct one-sided limit). The **Cholesky reuse cache** is bit-identical to a fresh computation over a sweep of mean hyperparameters and to a run with `_REUSE_CHOLESKY = False`, and its key `hyp[:cov_N+noise_N]` is complete given that `X`, `y` and `s2` are fixed for the life of a fit. **Thinning and burn-in** are exact: `eff_s_N = s_N·thin` draws thinned as `[thin-1::thin]` give precisely `s_N` rows; the sampler's `total_N = N + (N−1)(thin−1)` with `record = i >= burn and (i−burn) % thin == 0` records precisely `N`; the burn-in summary window `burn/2 <= i < burn` holds exactly `floor(burn/2)` iterations, matching `burn_stored`. `rng.py` is correct and complete for the five methods its consumers use, `random_integer` covers the one name that differs, and the `_LegacyRNG` proxy is genuinely stateless. The **Gelman–Rubin and effective-sample-size recursions index parameters correctly** even when the number of chains equals the number of parameters (I checked the reverse-transpose explicitly against per-parameter calls and against a two-scale target). The sampler keeps every draw inside `LB`/`UB`, leaves an `LB == UB` coordinate untouched, and excludes it from the diagnostics as documented.

---

## 4. Test adequacy notes

Tests that encode the implementation rather than the specification, or that pass for a reason that does not generalize:

1. **`test_uuinv` (`test_smoothbox.py:95`) asserts the length-weighted tail masses**, `(1-w)*(B[1]-B[0])/((B[1]-B[0])+(B[3]-B[2]))`, i.e. exactly what the body computes and exactly what the docstring contradicts. It is a transcription of the implementation. Its bound sets are also chosen so that the docstring's convention would pass three of the four (`[0,0,30,66]`, `[0,10,66,66]`, `[0,10,10,66]` all have one empty tail or a zero-width plateau), leaving only `[0,10,30,66]` to distinguish the two conventions — and for that one the test asserts the body's answer. A test written from the docstring would fail today. See F12.

2. **`test_constant_trace_in_a_free_parameter_fails_the_diagnostics` (`test_slice_sample.py:118`) and `test_fixed_parameter_is_left_out_of_the_diagnostics` (`:133`)** both pass because their constants — exactly 0 and exactly 1 — are the values whose mean is computed without rounding error, so `W` is exactly zero and `R`/`eff_N` are NaN. Substituting 0.3 turns `R` into 0.9487 and `eff_N` into 1.0526, and the first test's `exit_flag == -3`, `isnan(R)`, `isnan(eff_N)` and "did not move" assertions all fail. The tests document the documented behavior; they do not establish it. See F11.

3. **The four rank-one update tests** (`test_gaussian_process.py:436`, `:483`, `:555`, `:586`) cover the high-noise branch thoroughly — heteroskedastic noise in three parameterizations, a missing `sl` from an old pickle, replacement hyperparameters, `s2` alignment — and two of them assert `post.L_chol and post_ref.L_chol`, pinning the branch under test. Nothing covers `L_chol == False`, nothing covers `sn2_mult > 1`, nothing covers the `sqrt_arg <= 0` fallback, and every one uses a single hyperparameter sample. The unguarded branch of F4 sits in the complement.

4. **`test_random_function` (`:91`) cannot catch F3**, or anything else about the draws: it uses `np.random` unseeded, never exercises `__robust_cholesky`'s eigendecomposition fallback (the covariance is always factorizable), and asserts only the shapes returned by `predict(..., return_lpd=True)`. `__robust_cholesky` has no test of its own.

5. **`_small_gp_with_priors` (`:1617`) places the smoothbox on `noise_log_scale` and the smoothbox-Student-t on `mean_const`**, both single-entry blocks. That is the one configuration in which the unmasked `C_sb`/`C_sb_t` of F2 broadcasts correctly, so `test_log_likelihood_and_posterior_gradients` and `test_prior_mask_cache_follows_priors_and_bounds` exercise those branches without being able to detect the defect. Moving the smoothbox onto `covariance_log_lengthscale` (D = 2) and evaluating with the two coordinates on opposite sides of the box would fail immediately.

6. **`test_fitting_options` (`:1034`) asserts nothing.** Its comment is candid — "Test that all these at least can be run in a row" — and it is a smoke test over eight option dictionaries. It never names a sampler under either spelling (F5), never passes `widths`, and never inspects the returned hyperparameters or the sampling result.

7. **`test_split_update` (`:626`) does not exercise the rank-one path** despite its name: both of its incremental calls append ten rows and pass `hyp`, so both take the full recomputation. It is unseeded (`np.random.standard_normal`, `np.random.randint`), so the number of hyperparameter samples varies between runs.

8. **The dict form of `log_likelihood`/`log_posterior` (F6) has no test at all**, although both docstrings document it and `test_log_likelihood_and_posterior_gradients` deliberately flattens its fixture's hyperparameters to a 1-D array (`return gp, hyp[0]`) — the shape the dict path fails to produce.
