# P7 — variational posterior, entropies, VP statistics, internal-correctness track

Raw reviewer report, wave 5 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, fresh
context, read-only on the repositories, reading the code at `f873556`; it
did not read the MATLAB source. Its brief carried two first questions,
which the report answers before its findings. Its check scripts ran in the
session scratchpad and are kept only on the machine that ran them
(`dev/scripts/runs/LOCAL.md`). The text below is the agent's final message,
unedited. Its answer on the underflowed log density, that no path reaches
it in practice, is corrected by wave 7 (`../verification/wave7.md`, W7-3):
`vp.mode()` reaches it at the first trial step of its optimizer on a narrow
posterior.

---

# P7 — variational posterior, entropies, VP statistics (internal-correctness track)

## 1. Coverage

**Read completely**
- `pyvbmc/variational_posterior/variational_posterior.py` (the whole file; `plot`, `to_torch`, `to_arviz` skimmed only for their use of `sample`/`pdf`).
- `pyvbmc/entropy/entlb_vbmc.py`, `pyvbmc/entropy/entmc_vbmc.py`.
- `pyvbmc/stats/get_hpd.py`, `kde_1d.py`, `kl_div_mvn.py`.
- Tests: `pyvbmc/testing/variational_posterior/test_variational_posterior.py`, `test_variational_posterior_grad_fd.py`, `FIXTURES.md`; `pyvbmc/testing/entropy/test_entlb_vbmc.py`, `test_entmc_vbmc.py`; `pyvbmc/testing/stats/*`.
- `pyvbmc/variational_posterior/README.md`, `pyvbmc/entropy/README.md`; the "Slice P7", "Slice P6", "Slice P1b (randomness)" and "Settled non-differences" sections of `known_differences.md` (the only `dev/` file opened).

**Read as far as needed to judge the slice**
- `pyvbmc/vbmc/variational_optimization.py` (`_neg_elcbo`, `_vp_bound_loss`, `_soft_bound_loss`, the entropy call sites, `jacobian_flag = 1`), `pyvbmc/vbmc/vbmc.py` (the `kl_div`/`moments`/`sample` call sites at `:1553`, `:1573`, `:1677`, `:1854`, `:3214`), `pyvbmc/vbmc/active_sample.py` (`:534`, `:968`, `:975`, `:1004`, `:1065`), `pyvbmc/vbmc/active_importance_sampling.py` (`:81`, `:152`, `:378`, `:472`, `:488`), the four acquisition functions' VP-density reads, `pyvbmc/svbmc/svbmc.py:932`, `pyvbmc/pymc/_target.py:1402`, `pyvbmc/decorators/handle_0D_1D_input.py`, `ParameterTransformer.log_abs_det_jacobian`.

**Not reached**: `_torch.py`/`_arviz.py` internals, `plot`, the calibration profile machinery beyond its effect on chunking, the oracle fixtures (I built states by hand instead).

**Checks run** (all in the scratch directory, single-threaded BLAS, `pyvbmc.__file__ = C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\__init__.py`, NumPy 2.5.2 / SciPy 1.18.1):
1. `c1_balance.py` — expected component counts of the balanced sampler, analytic and empirical; `I` dtype/shape.
2. `c2_pdf.py` — `pdf` vs `scipy.stats` (normal, multivariate t, product of univariate t); `d pdf/dx` and `d log pdf/dx` vs central differences; quadrature of the original-space density over bounded and unbounded spaces.
3. `c3_underflow_kl.py` — `kl_div` guards, the `samples` + `gauss_flag` path, the mode cache, `entmc` with a zero weight.
4. `c4_stats.py` — `kl_div_mvn` vs the closed form, `get_hpd` rounding and ties, `kde_1d` normalization/bandwidth/fallback.
5. `c5_entropy.py` — both entropies vs a 1-D quadrature entropy; `entlb` gradients vs finite differences under the log/softmax reparameterization; `entmc`'s w-gradient vs the exact separated-mixture value; `Ns` rounding.
6. `c6_params.py` — `get_parameters`/`set_parameters` round trip over all 16 flag combinations × `raw_flag`; deepcopy; save/load; `get_bounds` lengths.
7. `c7/c8_mode.py` — `mtv` vs quadrature total variation (4 unbounded + 1 bounded case); `mode` vs a grid maximum in both spaces; the D = 1 path.
8. `c9_misc.py` — `fess` with an integer argument; `kl_div_mvn` with underflowing determinants; `moments` closed form and MC; KS tests of the sampler (plain/balanced, both spaces, `df` = 3, 8, K = 1 and 2); zero and tiny weights; float `N`.
9. `c10/c11` — where the log density underflows; whether the original-space density overflows; the mode's bound offset on narrow bounds.
10. `c12` — the exact indices the `raw_flag=False` positivity check inspects; the log-pdf gradient at an underflowed point.

What is **correct** and I confirmed numerically: the mixture density and both heavy-tailed variants (rel. err ≤ 1.5e-15 against SciPy); `d pdf/dx` and `d log pdf/dx` (≤ 7e-10 vs FD); the change of variables (original-space density integrates to 1.000000000 on bounded and unbounded transforms; the Jacobian sign is right); the analytic moments in transformed space (exact vs the closed form, and vs 2e6 draws); `entlb` as a genuine lower bound and all four of its gradient blocks under the log/σ, log/λ and softmax reparameterizations (≤ 3.4e-7 vs FD); `entmc` as an unbiased entropy estimator with the correct reparameterization gradients — and I verified that keeping the score term for `w` while dropping it for `μ, σ, λ` is *right*, not an inconsistency (∫q = 1 is constant in μ, σ, λ but not in w, so ∂H/∂w_k genuinely carries the −1); `kl_div_mvn`'s two KL formulas; the sampler's law (KS p-values 0.14–0.60 across balanced/plain, K = 1 and K > 1, both spaces, t and normal); `mtv` against quadrature (≤ 1.5e-3 at N = 2e5); `mode` finding the grid maximum of the density it claims to maximize in both spaces for D ≥ 2; `__deepcopy__` sharing only the generator; `get_bounds`' apparent length mismatch with `theta` (it is the *extended* vector of `_vp_bound_loss`, and the `order="F"` tiling matches).

## 2. First questions

### Q1 — Where an underflowed or zero log density reaches, and whether each reader is right

**Where the underflow starts.** `_pdf` sums component densities in linear space and logs the sum (`variational_posterior.py:905-911`). Measured on a unit 1-D Gaussian: the log density is exact to ≈ −744 (the sum reaches subnormals) and then drops to `−inf` (at `x = 40`, exact −800.9, returned `−inf`). A log-sum-exp evaluation would be exact to ≈ −1e308.

**Readers inside the slice**
- `mode` (`:1230-1240`) — `neg_log_pdf` turns `−inf` into `+inf`. The starting points come from `sample`, so `np.argmin` always prefers a finite start; an underflowed start is selected only if *every* start underflows, which the sampler's own draws make effectively impossible. If it did happen, SciPy returns `x0` with `fun = nan` (`ABNORMAL`, reproduced) — a silently wrong mode, not a crash. For `orig_flag=False` the mode uses `grad_flag=True, log_flag=True`, and `dy = dy / y` at `:907` gives `0/0 = nan` where `y == 0` (reproduced at `x = 40`: log pdf `−inf`, gradient `nan`, exact −40). A log-sum-exp form would make both finite and correct.
- `kl_div(gauss_flag=False)` (`:1487-1501`) — uses the **linear** density with explicit zero/inf guards. The zero guard works, but only by luck (see F1), and it caps the answer: two disjoint 1-D posteriors (N(0,1) vs N(80,1), true KL 3200 each way) return **706.98 / 706.97**, because the underflowed `q2` is replaced by `sys.float_info.min`. A silently wrong, capped number. The inf guard never fires (F1), so an overflowing original-space density makes `kl_div` return `[nan, nan]` — reproduced. With log-sum-exp *and* a log-space formulation, the 707 cap would disappear.
- `mtv` does not touch the density (it works on a KDE of samples). The entropies compute their own mixture sums; `entmc_vbmc.py:218` takes `log(E @ wnf)` and has exactly the same underflow structure (see F9).

**Readers outside the slice**
- `acq_fcn.py:36`, `acq_fcn_vanilla.py:36`, `acq_fcn_noisy.py:36` — `np.maximum(vp.pdf(...), realmin)`, floored in linear space.
- `acq_fcn_log.py:42` — `np.maximum(vp.pdf(..., log_flag=True), np.log(realmin))`. Note the two floors coincide at −708.4, so the log form buys **no extra range for the VP factor** (it only avoids `exp()` overflow on the GP mean); it even discards the correct values between −708 and −744 that `pdf` does return. A log-sum-exp `pdf` alone would change nothing here unless the floor moved with it.
- `active_importance_sampling.py:488` (`fess`) — same floor.
- `active_importance_sampling.py:378` — the proposal log density is used raw; when a point's VP log density is `−inf` *and* it falls outside every box, `m_max == −inf` and `:398` raises `ValueError("Invalid value.")`. This is the one reader that turns an underflow into a hard failure; a log-sum-exp `pdf` would make it unreachable from this cause.
- `calibration/_campaign.py:121` (benchmark harness only).

**The `orig_flag=True` path and points on/outside the bounds.** The strict mask (`x > lb_orig`, `x < ub_orig`) sets the density to 0 / the log density to `−inf` *before* the Jacobian correction, and the correction is applied only to masked-in rows — correct. The sign is correct (`y -= log_abs_det_jacobian(u)`, i.e. `p(x) = q(u)·|du/dx|`), and quadrature confirms normalization. What is *not* right is the arithmetic form: `:924` divides by `exp(logJ)` instead of subtracting in log space, so `exp(logJ)` can underflow to 0 and yield `+inf` for a value that is still representable (see F10). And the original-space density genuinely diverges at a bound, so `pdf(orig_flag=True)` **can** return `+inf` — I reproduced 9475/20000 infinite densities on a probit-transformed unit box with σ = 60, with log densities up to 1471.

**Summary answer:** underflow reaches the mode search (nan gradient, unreachable in practice), `kl_div` (a silent cap at ~707, and `nan` on the overflow side), and the IMIQR proposal (a hard `ValueError`). A log-sum-exp evaluation would fix the mode gradient and the `ValueError`, would not change the acquisition functions while their explicit floors stand, and would only help `kl_div` if that method also moved to log space.

### Q2 — Callers of `vp.sample`, and the sampler against its specification

**Every call site in the package:**

| Site | Unpacking |
|---|---|
| `variational_posterior.py:552` (`to_arviz`), `:1167` (`moments`), `:1251` (`mode`), `:1346`, `:1348` (`mtv`), `:1489`, `:1496` (`kl_div`), `:1571` (`plot`) | `x, _ = …` ✓ |
| `vbmc/active_sample.py:968`, `:1065` | `heavy_Xs, _ = …`, `vp_Xs, _ = …` ✓ |
| `vbmc/active_importance_sampling.py:81`, `:152` | `Xa, __ = …` ✓ |
| `vbmc/vbmc.py:1677` | `Xrnd, _ = …` ✓ |
| `svbmc/svbmc.py:932` | `X_m, _ = …` ✓ |
| `pymc/_target.py:1402` | `samples, _ = …` ✓ |
| **`vbmc/active_importance_sampling.py:472` (`fess`)** | **`X = vp.sample(N, orig_flag=False)` — takes the pair for the array** |

`fess(vp, gp, X)` documents `X : np.ndarray(N, D) or int`; with an integer (or the default `X=100`) it raises `AttributeError: 'tuple' object has no attribute 'shape'` — reproduced for both `fess(vp, means)` and `fess(vp, means, 50)`. The one live call site (`:88`) passes an array, so the branch is dead in a run; it is a documented public entry point that cannot work. This is in slice P4's file, so I report it only as the answer to this question, not as a P7 finding.

**The sampler against its specification.** Statistically the draws are right: KS tests on the marginal give p = 0.29 (plain) and 0.59 (balanced) for a 3-component 1-D mixture at N = 2e5, p = 0.60 after mapping original-space draws back through a probit transform, and p = 0.14 / 0.53 / 0.60 for the `df = 3, 8` heavy-tailed variants at K = 1 and K = 2. I re-derived the t scaling: `rng.gamma(df/2, df/2)` gives `G = (df/4)·χ²_df`, so `t = (df/2)/√G = √(df/W)` — exactly the multivariate-t scale, and it matches `pdf(df>0)`. K = 1, zero weights and tiny weights all behave (`w = [0, .5, .5]` produces no draws from component 0 in both modes). Analytic moments and the 2e6-draw MC moments agree to 3.7e-3.

The one defect is in the **balanced** assignment's residual allocation (F6): `sum(w_extra)` is Python's builtin `sum` over a `(1,K)` array, which returns the row rather than the total, so the extra draws are allocated with the wrong probabilities. Reproduced: `w = [0.7, 0.2, 0.1]`, `N = 13` gives empirical frequencies `[0.7238, 0.1832, 0.0930]` where the exact stratified answer is `[0.7, 0.2, 0.1]`. The bias is bounded by ≈ `K` samples in expectation, so it is negligible at N = 1e5–1e6 and visible at small N.

## 3. Findings

### F1. `kl_div`'s zero/inf guards are parsed as `q == isinf(q)`, so the infinity branch never fires and the method returns NaN
- Location: `pyvbmc/variational_posterior/variational_posterior.py:1492`, `:1493`, `:1499`, `:1500`; MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- `q1[q1 == 0 | np.isinf(q1)] = 1.0` binds as `q1 == (0 | np.isinf(q1))`, i.e. `q1 == np.isinf(q1)`, because `|` has higher precedence than `==`. For a non-infinite entry the right-hand side is `False → 0.0`, so the intended `q == 0` test survives; for an infinite entry it is `True → 1.0`, and `inf == 1.0` is never true. The guard therefore silently drops the infinity case it was written for. Intended: `(q1 == 0) | np.isinf(q1)`.
- Consequence if real: `vp.pdf(..., orig_flag=True)` genuinely returns `+inf` when a draw lands close enough to a hard bound for the original-space density to exceed the float range (the density diverges there). `np.log(inf) - np.log(inf)` then makes `kl1`/`kl2` NaN, `np.maximum(0, nan)` is NaN, and in `optimize` that NaN becomes `sKL` (`vbmc.py:1549`) and flows into the reliability/stability diagnostics. Reachable only with `kl_gauss=False` (the shipped default is `True`) and a posterior much wider than the bounded transform's natural scale.
- Suggested reproduction: ran it. `ParameterTransformer(1, [[0]], [[1]], "probit")`, `sigma = 60`, two such posteriors: `vp1.kl_div(vp2=vp2, N=20000, gauss_flag=False)` → `[nan nan]`, with 5335/20000 infinite `q1` values at `vp1`'s own draws. Also ran the mask directly: for `q = [0, inf, 1, 0.5]` the coded mask is `[T, F, F, F]`, the intended one `[T, T, F, F]`.
- Test adequacy: no. `test_kl_div_two_vp_no_gauss_flag` and `..._identical_no_gauss_flag` use unbounded `D=1` posteriors where the density is never infinite; nothing exercises a bounded transform in `kl_div`.

### F2. `kl_div(samples=…, gauss_flag=True)` uses the grand mean of the sample matrix as the mean vector
- Location: `pyvbmc/variational_posterior/variational_posterior.py:1482`; MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high
- `q2mu = np.mean(samples)` averages *every element* of the `(N, D)` matrix into one scalar, where the next line correctly takes `np.cov(samples.T)` (a `D×D` matrix). `kl_div_mvn` then reshapes the scalar to `(1,1)` and broadcasts it against the `(D,1)` mean of `self`, so every coordinate is compared against the same number. It should be `np.mean(samples, axis=0)`.
- Consequence if real: the Gaussianized KL against a set of samples is wrong whenever the marginal means differ across dimensions — silently, with no error. Measured: `vp` = N(µ=[10,−5,2], I) against 20000 draws of the *same* distribution returns `[56.58, 56.32]` instead of ≈ 0. Public-API only; the main loop always passes `vp2`.
- Suggested reproduction: ran it (`c3_underflow_kl.py`); `np.mean(samples) = 2.339` against `np.mean(samples, axis=0) = [9.998, −4.997, 2.015]`.
- Test adequacy: no, and this is the clearest "test mirrors the implementation" case in the slice. `test_kl_div_two_vp_identical_samples_gauss_flag` builds the VP from `x0 = np.array([[5]])`, so all three coordinates have mean ≈ 5 and the grand mean coincides with the per-coordinate mean; `test_kl_div_two_vp_samples_gauss_flag` is `D = 1`, where the two are identical by definition. Any test with distinct per-dimension means would have caught it.

### F3. `vp.mode(orig_flag=True)` raises `AxisError` for every one-dimensional posterior
- Location: `pyvbmc/variational_posterior/variational_posterior.py:1269-1277`; MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high
- `self.parameter_transformer.lb_orig` has shape `(1, D)`; `.squeeze()` gives a 0-d array for `D = 1`, and `np.stack((0-d, 0-d), axis=1)` raises `AxisError: axis 1 is out of bounds for array of dimension 1`. `np.reshape(-1)`/`np.atleast_1d` would be the shape-safe form.
- Consequence if real: the documented default call `vp.mode()` is unusable for any `D = 1` problem, bounded or unbounded. `mode(orig_flag=False)` works. Nothing inside the package calls `mode`, so this is user-facing only (example notebook 2 uses it at `D = 2`).
- Suggested reproduction: ran it. `VariationalPosterior(1, 2)` with either `ParameterTransformer(1)` or `ParameterTransformer(1, [[-3]], [[3]])`: `mode(orig_flag=True)` raises, `mode(orig_flag=False)` returns `[1.299]`.
- Test adequacy: no. `test_mode_orig_flag` and `test_mode_no_orig_flag` both use the `D = 2` MATLAB fixture. (Those two tests also assert the *same* value for the original and the transformed mode, because the fixture carries an identity transformer — so no test exercises the Jacobian term inside the mode search either.)

### F4. The mode search offsets the bounds by an absolute `sqrt(eps)`, which is not a small fraction of a narrow range
- Location: `pyvbmc/variational_posterior/variational_posterior.py:1269-1277`; MATLAB: "not read (internal track)"
- Category: formula
- Proposed classification: suspected defect
- Confidence: high
- The optimizer's box is `[lb_orig + √eps, ub_orig − √eps]` with `√eps = 1.49e-8` regardless of the width of `[lb_orig, ub_orig]`. The offset is meant to keep the start strictly interior (the density is 0 on the bound), so it should scale with the range, e.g. `√eps · (ub − lb)` or `np.nextafter`.
- Consequence if real: for a range of 1e-6 the box loses 1.49% of the interval at each end; for a range below `2√eps ≈ 3e-8` the lower bound exceeds the upper one and SciPy raises `ValueError: An upper bound is less than the corresponding lower bound`. Separately, `x0` is clamped to `[lb_orig, ub_orig]` (`:1278-1281`), i.e. to a different box than `bounds`, and relies on L-BFGS-B silently re-clipping it.
- Suggested reproduction: ran it. `D = 2`, bounds `[0, 1e-6]`: the mode is found but the box is 1.49% narrower at each end; bounds `[0, 1e-8]` and `[0, 1e-9]`: `ValueError`.
- Test adequacy: no test uses narrow bounds with `mode`.

### F5. The `_mode` cache ignores `n_opts` and is not invalidated by direct changes to the posterior
- Location: `pyvbmc/variational_posterior/variational_posterior.py:1227-1228`, `:1295-1296`; MATLAB: "not read (internal track)"
- Category: state/caching
- Proposed classification: possibly intentional
- Confidence: high (the behavior), low (that it should change)
- `if orig_flag and self._mode is not None: return self._mode` returns the cached value whatever `n_opts` the caller passes, and `_mode` is cleared only by `set_parameters` (`:1138`). Assigning `vp.mu`, `vp.sigma`, `vp.lambd` or `vp.w` directly — which is how the shipped tests and the `_sieve`/`_vb_init` helpers build posteriors — leaves the stale mode in place. `get_parameters` renormalizes `w` (F13) without clearing it either.
- Consequence if real: a user who asks for a more thorough search after a first call gets the first result; a user who edits the posterior gets the mode of the old one. Reproduced: multiplying `vp.mu` by 100 leaves `vp.mode()` unchanged; `mode(n_opts=1)` and `mode(n_opts=50)` return bit-identical arrays.
- Suggested reproduction: ran it (`c3`, `c8`).
- Test adequacy: `test_mode_exists_already` only asserts that the cache *is* returned; `test_get_set_parameters_delete_mode` only covers the `set_parameters` invalidation. The module's own README records this as an open question ("double-check that it is being cleared correctly when the mode changes").

### F6. The balanced sampler's residual correction uses Python's `sum`, so the stratified component counts are biased
- Location: `pyvbmc/variational_posterior/variational_posterior.py:643` (and `:642` for the count); MATLAB: "not read (internal track)"
- Category: formula
- Proposed classification: suspected defect
- Confidence: high
- The module imports only `numpy as np`, so `sum(w_extra)` is the builtin applied to a `(1, K)` array: iterating yields the single row, and the result is the `(K,)` vector of residuals, not their total. `repeats_extra - sum(w_extra)` is therefore a vector `R − e` instead of the scalar `R − Σe` (which is 0 up to rounding), and the extra draws are allocated with probabilities proportional to `e_k + w_k(R − e_k)` rather than to the residuals `e_k`. Line `:642` correctly uses `np.sum` on the same array two lines earlier.
- Consequence if real: the "exactly proportionally (or as close as possible)" promise of the docstring is not met. For `w = [0.7, 0.2, 0.1]`, `N = 13`, the exact stratified expectation is `[9.1, 2.6, 1.3]` and the code gives `[9.41, 2.38, 1.21]`; empirical frequencies over 40000 repeats: `[0.7238, 0.1832, 0.0930]` against `[0.7, 0.2, 0.1]`. The absolute bias is bounded by roughly one sample per component, so it is negligible at the `N = 1e5`–`1e6` used by `moments`, `kl_div` and `mtv`, and matters for the small-`N` calls (`active_sample.py:968`, `:1065`, `svbmc.py:932`). *Related, independent*: `repeats_extra` is derived from the residual sum rather than from `N − i.shape[0]`, so with unnormalized weights (`sum(w) = 0.5`) too few indices are produced and `sample` dies with a broadcast error (`operands could not be broadcast together with shapes (11,2) (6,1)`) — reproduced.
- Suggested reproduction: ran it (`c1_balance.py`), both the analytic expectation and 40000 repetitions.
- Test adequacy: no. `test_sample_balance_no_extra` and `test_sample_balance_extra` both use `w = [0.5, 0.5]`, the one weight vector for which the bug cancels exactly (I verified: coded and intended probabilities both `[0.5, 0.5]`).

### F7. `set_parameters(raw_flag=False)`'s positivity check slices from the wrong end of `theta`
- Location: `pyvbmc/variational_posterior/variational_posterior.py:1079-1090` (the slice at `:1086`); MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high
- `check_idx` accumulates *negative* offsets (`-K`, `-D`, `-K`), so the tail it names is `theta[check_idx:]`. The code writes `theta[-check_idx:]`, which negates it back into a *positive start index*. The two coincide only when `D*K` happens to equal the tail length. Concretely, for `D=3, K=4` with all flags set (`len(theta) = 23`, `mu` at 0..11) the check inspects indices 11..22 instead of 12..22 — one `mu` coordinate too many; for `D=2, K=2` (`len(theta) = 10`, `mu` at 0..3) it inspects 6..9 instead of 4..9 — **skipping both `sigma` entries**. With no scale or weight optimized, `check_idx == 0` and `theta[-0:]` is the whole vector.
- Consequence if real: two opposite failures, both silent. (i) Spurious rejection: a perfectly valid parameter vector is refused because a *component mean* is negative — and in the transformed space means are routinely negative. Reproduced: `D=3, K=4` with random normal means raises `ValueError: sigma, lambda and weights must be positive` for 7 of the 8 flag combinations with `optimize_mu=True`. (ii) Missed rejection: reproduced at `D=2, K=2`, where `theta[4] = -0.5` (a negative `sigma`) is accepted and leaves `vp.sigma = [[-0.5, 0.9]]`, a posterior whose "density" is meaningless. Production always uses `raw_flag=True`, so this is public-API only.
- Suggested reproduction: ran it (`c6_params.py`, `c12_final.py`); the index lists above are printed directly.
- Test adequacy: no. `test_set_parameters_not_raw` and `test_get_set_parameters_roundtrip_non_raw` use `rng.random(...)` (all non-negative) or a VP built from `x0 = [[5]]`, so `mu` is never negative; `test_set_parameters_not_raw_negative_error` makes *every* entry negative, so it raises for the wrong reason.

### F8. `kl_div_mvn` tests and divides raw determinants, so it returns `(inf, inf)` for a well-conditioned but small-scaled covariance
- Location: `pyvbmc/stats/kl_div_mvn.py:34-39`; MATLAB: "not read (internal track)"
- Category: formula
- Proposed classification: suspected defect
- Confidence: high (the behavior), medium (that it is reachable in a run)
- `detq1 = np.linalg.det(sigma1)` underflows to 0 for a perfectly invertible matrix once `∏ λ_i < 1e-308`; the guard at `:36` then declares "KL divergence is infinite" and returns `(inf, inf)`. Above the underflow threshold the ratio `detq2/detq1` loses precision in the subnormal range. `np.linalg.slogdet` computes `lndet` in log space and separates true singularity from scale.
- Consequence if real: `kl_div(gauss_flag=True)` is the *default* route in the main loop (`vbmc.py:1553`, `:1854`, `kl_gauss = True`) and in `_compute_true_diagnostic` (`:3215`), where the covariance is the original-space MC covariance. A tightly concentrated posterior in higher `D` (e.g. `D = 40` with per-coordinate sd 1e-8, or `D = 30` with sd 1e-12) returns `inf` where the true symmetrized KL is ~22; `sKL = max(0, 0.5·sum(...))` then becomes `inf` and the stability/termination diagnostics see a non-finite reliability input. At `D = 20`, sd 1e-8 the determinant is subnormal and the answer is already only 6 digits correct (`6.36294918` vs `6.362944`).
- Suggested reproduction: ran it (`c9_misc.py`): `D=40`, `S = 1e-16·I` vs `4S` → `[inf inf]`, exact `[12.73, 32.27]`.
- Test adequacy: no. All five tests in `test_kldiv_mvn.py` use `np.eye` or `1×1` unit covariances.

### F9. `entmc_vbmc` returns NaN when a mixture weight is exactly zero and the components are separated
- Location: `pyvbmc/entropy/entmc_vbmc.py:218`, `:221`, `:232`, `:234`; MATLAB: "not read (internal track)"
- Category: formula
- Proposed classification: suspected defect
- Confidence: medium
- The estimator draws `Ns` samples from *every* component, including one with `w_j = 0`. At those samples the mixture density `q = E @ wnf` can be exactly 0 (the other components' densities underflow), so `np.log(q) = -inf`, and `H = -(w * sum_logq).sum() / Ns` evaluates `0 · (-inf) = nan`. The gradient accumulators `r = lsum / q` and `1.0 / q` go to `nan`/`inf` on the same rows. A component of zero weight contributes nothing to the entropy and could simply be skipped, or `q` floored.
- Consequence if real: `_neg_elcbo` (`variational_optimization.py:1279`) would hand Adam a NaN objective and gradient. Reaching it needs `w_j` to underflow to exactly 0, which `set_parameters` produces when `eta_j − max(eta) < −745`; the eta soft bound was removed (P6 known difference), so eta is unbounded during Adam, though the small-weight penalty and pruning push against it. The degradation is gradual: with `w_j = 1e-12` everything is finite; with `w_j = 1e-320` the gradient already overflows (`1.0/q`), while `H` is still finite.
- Suggested reproduction: ran it. `D=1`, `mu = [0, 60]`, `sigma = [1,1]`, `w = [1, 0]`, `entmc_vbmc(vp, 100)` → `H = nan`, `dH = [−2.2e-18, nan, 0.908, nan, nan, nan, nan]`.
- Test adequacy: no. Every entropy test uses uniform or softmax weights; `test_entmc_vbmc_nonoverlapping_mixture` has separated components but equal weights.

### F10. The original-space density is formed as `y / exp(logJ)` rather than `exp(log y − logJ)`
- Location: `pyvbmc/variational_posterior/variational_posterior.py:917-928`; MATLAB: "not read (internal track)"
- Category: formula
- Proposed classification: suspected defect
- Confidence: medium
- In the non-log branch the Jacobian correction is a division by `np.exp(log_abs_det_jacobian(u))`. When the log-Jacobian falls below about −745 the exponential underflows to 0 and the division returns `+inf` (with a `divide by zero` RuntimeWarning) for values that are still representable: the quotient only needs `log y − logJ < 709`, which leaves roughly 36 nats of range that the division form throws away. The log branch (`:920`) already does the right thing.
- Consequence if real: `vp.pdf(x, orig_flag=True)` returns `inf` instead of a finite density for points near a hard bound, and that `inf` then reaches `kl_div` (F1) and any user code. Note that the divergence itself is genuine — for a posterior much wider than the transform's scale the density really does exceed the float range near the bound (I measured log densities up to 1471) — so this finding is about the ~36 nats lost to the division, not about the physics.
- Suggested reproduction: ran the surrounding case (`c11_inf_density.py`): 9475/20000 infinite densities, `#(log pdf > 709) = 9477`, i.e. two rows where the log branch is finite and the linear branch is not. A sharper reproduction is `D = 2`, probit on `[0,1]`, `u1 = u2 = 27.3`, `σ = 4.5`: `−logJ ≈ 747` (so `exp(logJ)` is 0) while `log y − logJ ≈ 705` is representable.
- Test adequacy: no. `test_pdf_outside_bounds` only checks 0 / `−inf` on and outside the bounds, never the near-bound interior.

### F11. `get_hpd` rounds the subset size with Python's banker's rounding
- Location: `pyvbmc/stats/get_hpd.py:36`; MATLAB: "not read (internal track)"
- Category: defaults
- Proposed classification: unsure
- Confidence: medium (the behavior), low (that it matters)
- `hpd_N = round(hpd_frac * N)` uses Python 3's round-half-to-even. Measured at `N = 10`: `hpd_frac = 0.05` gives 0 points (half-away-from-zero would give 1), `0.25` gives 2 (not 3), `0.45` gives 4 (not 5). Only exact half-integers are affected.
- Consequence if real: the high-posterior-density subset can be one point smaller than the fraction asks for, which moves the GP length-scale and output-scale lower bounds (`gaussian_process_train.py:327`) and the ELCBO reference set (`variational_optimization.py:798`). The default `hpd_frac = 0.8` hits a half-integer only when `N` is an odd multiple of 5 (`N = 5, 15, 25, …`), i.e. a handful of early iterations; `active_sample.py:1004` passes random fractions, where exact halves have measure zero. `hpd_frac = 0` is handled correctly (empty subset, NaN range).
- Suggested reproduction: ran it (`c4_stats.py`).
- Test adequacy: no. `test_get_hpd` uses `N = 100` with fractions 0.8, 0.5 and 0.01, none of which lands on a half-integer.

### F12. Shape and type inconsistencies in `sample` and `moments` that the docstrings contradict
- Location: `pyvbmc/variational_posterior/variational_posterior.py:625-626`, `:690`, `:1167-1170`, `:655`, `:665`; MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: possibly intentional
- Confidence: high (the behavior), low (the severity)
- Four unrelated small ones, all verified: (a) `sample` documents `I` as "an `N`-by-1 array" but returns `(N,)` `int64` for `K > 1`, `(N,)` `float64` for `K == 1` (`:690`, `np.zeros(N)`), and `(0, 1)` for `N < 1`. (b) `moments(orig_flag=True, cov_flag=True)` returns a **0-d** covariance for `D = 1` (`np.cov` of a `(1, N)` array), while `orig_flag=False` returns `(1, 1)`; downstream `kl_div_mvn` survives only because it calls `np.atleast_2d`. (c) `pdf` accepts negative `df` (the product-of-univariate-t branch, `:876-903`) but `sample` raises `ValueError: shape < 0` for the same value — the two heavy-tailed families are not symmetric between density and draws. (d) `sample` requires an integer `N`: `vp.kl_div(vp2, N=1e5, gauss_flag=False)` and `vp.mtv(vp2, N=1e5)` raise `TypeError: slice indices must be integers`, while `moments` guards with `int(N)` (`:1167`).
- Consequence if real: surprising failures or wrong shapes in user code; none of the four is reachable from a default run (production passes `int(...)` everywhere and never uses `df < 0`).
- Suggested reproduction: ran all four (`c1`, `c9`).
- Test adequacy: partially. `test_sample_*` assert only `i.shape[0] == N`, never the dtype; no test calls `moments` with `D = 1`.

### F13. `get_parameters` mutates the posterior, undocumented, and does not clear the mode cache
- Location: `pyvbmc/variational_posterior/variational_posterior.py:1012-1019`; MATLAB: "not read (internal track)"
- Category: state/caching
- Proposed classification: possibly intentional
- Confidence: high (the behavior), low (that it should change)
- The getter rewrites `self.lambd`, `self.sigma` and (when weights are optimized) `self.w` in place. The `lambd`/`sigma` renormalization is density-preserving (`σ_k λ_d` is invariant — verified), but the weight normalization is not: with `w = [1, 2, 3]` the density at a fixed point changes from 0.0349 to 0.0058 (the factor 6). The docstring says only "Return all the active parameters flattened as a 1D array". `set_parameters` sets `self._mode = None` at the end; `get_parameters` does not, so a cached mode can survive a change of `w`.
- Consequence if real: a caller that inspects a posterior with `get_parameters` silently changes it. In production the weights are always normalized already, so nothing moves.
- Suggested reproduction: ran it (`c6_params.py`).
- Test adequacy: no test asserts that `get_parameters` leaves the object alone.

### F14. `kde_1d`'s density integrates to `n/(n−1)`, not 1
- Location: `pyvbmc/stats/kde_1d.py:233`, `:260`; MATLAB: "not read (internal track)"
- Category: formula
- Proposed classification: possibly intentional
- Confidence: high (the behavior), low (the severity)
- `xmesh = np.linspace(lower_bound, upper_bound, n)` has spacing `Δ/(n−1)` while the DCT/IDCT pair normalizes as if the `n` nodes tiled the interval, so the grid sum times the spacing is `n/(n−1)`. Measured: 1.000244200 at `n = 2^12` (`4096/4095 = 1.000244200`), 1.0000610 at `n = 2^14`. The docstring calls the return value "the values of the kernel density estimate at the grid points", i.e. a density.
- Consequence if real: any caller that uses the output as a normalized density is off by 2.4e-4 (at `n = 2^12`). The one in-package caller, `mtv` (`:1379`, `:1383`), renormalizes explicitly, so nothing in PyVBMC is affected.
- Suggested reproduction: ran it (`c4_stats.py`), for `n = 2^12` and `2^14`, with and without explicit bounds.
- Test adequacy: no. `test_kde1d.py` checks shapes and a loose (`< 0.03`) total-variation distance against the true density, which cannot see a 2.4e-4 normalization error.

### F15. `entlb_vbmc` and `entmc_vbmc` disagree on the weight gradient of a one-component posterior
- Location: `pyvbmc/entropy/entlb_vbmc.py:60-78` (`w_grad = np.zeros(1)` at `:78`) against `pyvbmc/entropy/entmc_vbmc.py:239`; MATLAB: "not read (internal track)"
- Category: formula/gradient
- Proposed classification: possibly intentional
- Confidence: low
- For `K == 1` `entlb` takes the exact single-Gaussian entropy, which it treats as independent of `w`, and returns `w_grad = 0`. `entmc` returns `H − 1` for the same posterior, which is the `K > 1` formula's limit and the true `∂H/∂w`. Verified: `entlb(D=3, K=1, jacobian_flag=False)` gives a trailing 0; `entmc` on a separated mixture reproduces `H_k − 1 − log w_k` to within Monte Carlo error.
- Consequence if real: none in production — `_neg_elcbo` fixes `jacobian_flag = 1` (`variational_optimization.py:1191`), and the softmax Jacobian of a one-component posterior is exactly 0, so both convert to 0. It matters only to a direct caller passing `jacobian_flag=False`, which is what both entropy test modules do.
- Suggested reproduction: ran it (`c5_entropy.py`).
- Test adequacy: the tests encode each function's own convention (`single_gaussian_entropy` in `test_entmc_vbmc.py` returns `H − 1`; `test_entlb_vbmc_single_gaussian` finite-differences a wrapper whose `H` really is `w`-independent), so neither can see the disagreement.

## 4. Test adequacy notes

- **`test_kl_div_two_vp_identical_samples_gauss_flag`** is the clearest instance of a test mirroring the implementation. It is the only `D > 1` test of the `samples` + `gauss_flag` path, and it chooses a posterior (`x0 = [[5]]`, so every coordinate has the same mean) for which the scalar grand mean of F2 coincides with the correct mean vector. Its sibling `test_kl_div_two_vp_samples_gauss_flag` is `D = 1`, where the distinction does not exist.
- **`test_sample_balance_no_extra` / `test_sample_balance_extra`** both use `w = [0.5, 0.5]`, the single weight vector for which F6's arithmetic error cancels exactly. Neither asserts anything about the component *frequencies* relative to `w` — only that the counts are `N/2` or `⌊N/2⌋ ± 1`, which the buggy code satisfies.
- **`test_set_parameters_not_raw`, `test_set_parameters_not_raw_negative_error`, `test_get_set_parameters_roundtrip_non_raw`** all feed `theta` values that are uniformly non-negative or uniformly negative, so the off-by-one slice of F7 is invisible. The round-trip tests cover only 2 of the 16 `optimize_*` combinations (all-true, and `optimize_mu=False`).
- **`test_mode_no_orig_flag` / `test_mode_orig_flag`** assert the *same* MATLAB value for both spaces, because `get_matlab_vp()` installs a default identity `ParameterTransformer`. They therefore pin the optimizer but not the change of variables that distinguishes the two modes, and the docstring's own warning ("the mode is not invariant to nonlinear reparameterizations") is untested. Neither covers `D = 1` (F3) or narrow bounds (F4).
- **`test_pdf_default_no_orig_flag` / `test_pdf_grad_*`** pin hard-coded density and gradient values at a single point 0.004 away from a mean with `sigma = 1e-3`; they are a fixture check, not a specification check. The independent-reference checks I ran (SciPy `multivariate_normal`, `multivariate_t`, a product of univariate `t`s, and central differences) all pass, so the formulas are right — but the suite would not have told anyone that.
- **`test_kldiv_mvn.py`** uses only identity covariances and `1×1` unit matrices, so neither the conditioning (F8) nor the general `Σ₂⁻¹Σ₁` trace term is exercised against an independent reference.
- **`test_get_hpd.py`** never puts `hpd_frac * N` on a half-integer, so the rounding convention (F11) is unpinned; the tie-breaking tests added later are good and do pin the stable sort.
- **`test_kde1d.py`** checks shapes plus a `< 0.03` total-variation distance; it pins neither the bandwidth nor the normalization (F14), and its local `mtv` helper duplicates `VariationalPosterior.mtv`'s integration code, so a change to that integration scheme would be checked against a copy of itself.
- The entropy modules' finite-difference tests are genuinely useful (they use common random numbers via a fixed `rng=42`), and the comment in `test_entmc_vbmc_single_gaussian` correctly documents that the reparameterization gradient differs from the exact derivative of the *estimator* by Monte Carlo noise. But `entmc_vbmc_wrapper` and `entlb_vbmc_wrapper` both leave `vp.eta` at the constructor's `np.ones((1,K))/K` and pass `jacobian_flag=False`, so no test checks that a posterior's `eta` and `w` are in step when the softmax Jacobian is applied — the one place where a stale `eta` would silently corrupt the weight gradient.
