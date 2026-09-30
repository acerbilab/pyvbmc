# G2 — gpyreg: prediction, quadrature, kernels, means, noise, MATLAB-comparison track

Raw reviewer report, wave 6 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-21. One Opus agent, fresh
context, read-only on the repositories, reading gpyreg at `fdbafdf`, whose
code under `gpyreg/` is that of the pinned revision `9e70e6b`, with PyVBMC at
`8af5daac` as its caller; it compared the Python code with `gplite/` of
MATLAB VBMC at `396d649`, and was told that gpyreg's copy under `matlab/` is
not the reference. It did not see the internal report of this slice. Its
brief carried a first question, which the report answers before its
findings: whether gpyreg's `predict` has the predictive log-density with the
total predictive variance that `gplite_pred.m` computes since `68a197b`, in
each noise parameterization and over the hyperparameter samples, and which
noise variance each noise-inclusive output of `predict`, `predict_full` and
`quad` adds at a test point. Its check scripts ran in the session scratchpad
and are kept only on the machine that ran them (`dev/scripts/runs/LOCAL.md`).
The text below is the agent's final message, unedited.

---

# G2 — gpyreg: prediction, quadrature, kernels, means, noise (MATLAB-comparison track)

## 1. Coverage

**Python, read completely** (all under `C:\Users\luigi\Documents\GitHub\gpyreg\gpyreg\`):
- `gaussian_process.py`: `predict` (1856–2072), `predict_full` (1754–1854), `quad` (2074–2255), `random_function` (2515–2611), `_convert_shapes` (2846–2888), `_can_retain_cross_covariance` (53–62), `_solve_triangular` (65–90), `Posterior` (2891–2943), `__robust_cholesky` (2613–2637, read in full because `random_function` depends on it).
- `noise_functions.py`, `mean_functions.py`, `covariance_functions.py`, `isotropic_covariance_functions.py`: whole, including every `hyperparameter_count`, `hyperparameter_info`, `get_bounds_info`/`_bounds_info_helper` and every `compute_grad` branch (the gradient branches were checked by derivation, not by finite differences — gpyreg's own tests cover them; I re-derived SE, Matern 1/3/5, rational-quadratic, both isotropic kernels, the three means and all four noise terms).

**Python, read in part** (for invariants my slice depends on, not reviewed): `__core_computation` (2639–2844, for `sl`, `sW`, `L`, `L_chol`, `sn2_mult`), `update`'s rank-one branch (763–900, for the `sl` invariant `quad` relies on), `fit`'s option defaults and bounds resolution (1021–1140), `get_recommended_bounds` (331–430), `gpyreg/rng.py` whole. **Skimmed:** `plot`, `__tight_subplot`. **Not reached** (out of slice): `slice_sample.py`, `f_min_fill.py`, `formatting.py`, the objective and the samplers inside `fit`.

**MATLAB, read completely** (`C:\Users\luigi\Documents\GitHub\vbmc\gplite\`): `gplite_pred.m`, `gplite_quad.m`, `gplite_covfun.m`, `gplite_noisefun.m`, `gplite_rnd.m`, `private/sq_dist.m`, `private/quantile1.m`, `private/gplite_core.m` (for the posterior factors). `gplite_meanfun.m`: the `info` branch in full, the compute branch for cases 0–9 (cases 0, 1, 4 — the ported families — line by line). `misc/gptrain_vbmc.m` + its local `vbmc_gphyp` (to establish which bound recommendations a default run consumes).

**Confirmed unported and left alone:** `gplite_qpred.m`, `gplite_sample.m`, `gplite_fmin.m`, `gplite_intmeanfun.m`, `outwarp_*.m`, and the `intmeanfun`/output-warping branches inside `gplite_pred.m`, `gplite_core.m`, `gplite_rnd.m` — no Python counterpart anywhere in gpyreg, as the sheet states.

**History:** of the six MATLAB files in my slice, only `gplite_pred.m` changed after gpyreg's history begins (2021-05-24), in `68a197b` (2022-06-25). `gplite_quad.m` (2019-06-30), `gplite_covfun.m` (2019-06-17), `gplite_meanfun.m` (2020-05-08), `gplite_noisefun.m` (2019-06-28), `gplite_rnd.m` (2019-11-29) are untouched.

**Reproductions** (scratchpad `…/wave6/G2_comparison/`): `matlab_gplite.py` is a transcription of `gplite_pred.m`, `gplite_quad.m`, `gplite_noisefun.m`, the SE-ARD branch of `gplite_covfun.m`, cases 0/1/4 of `gplite_meanfun.m`, all four `info` branches, `sq_dist.m` and `quantile1.m`; `c1`–`c9` are the checks. `gpyreg.__file__` printed as the sibling checkout in every run; all runs single-threaded BLAS.

**Bulk agreement.** `GP.predict` reproduces `gplite_pred.m` to ≤7.7e-14 (relative) across: all three PyVBMC noise parameterizations, `Ns` = 1 and 3, `s2_star` present and absent, `separate_samples` both ways, `add_noise` both ways, `return_lpd` on, and both `L` representations (the non-Cholesky representation agrees to 4e-8, the residual being my use of a solve where the code stores an explicit inverse). `GP.quad`'s mean matches `gplite_quad.m` bit for bit in every configuration tried. The noise bound recommendations match MATLAB's exactly in all four parameterizations. `NegativeQuadratic.compute_batched` is bit-identical to the per-sample loop (D = 1, 3, 8, 12 checked by gpyreg's test; D = 3 by mine).

---

## 2. First question: the predictive log density, and which noise variance each output adds

**Does gpyreg have the post-`68a197b` quantity?** Yes, and it never had the pre-fix one. `predict(..., return_lpd=True)` builds the log density on the **total** predictive variance `y_s2 = max(latent s2, 0) + sn2_star * sn2_mult` (`gaussian_process.py:2032`, used at `:2036-2040`, `:2057`, `:2062`), which is `gplite_pred.m:126` as amended (`-0.5*(ystar-ymu).^2./ys2 - 0.5*log(2*pi*ys2)`). The `add_noise` flag changes only which variance `predict` *returns*, not which one the lpd uses: `lpd(add_noise=True)` and `lpd(add_noise=False)` are equal in every case I ran. Chronology: `return_lpd` entered gpyreg in `9b3d46b` (2022-06-02), 23 days *before* MATLAB's fix, already with the total variance; MATLAB's pre-fix line used `sn2_star*sn2_mult` alone, and gpyreg never transcribed it. So the 2022 MATLAB change brought MATLAB into agreement with gpyreg rather than the other way round.

**Which noise variance, per parameterization, at a test point** (`noise_functions.py:248-278`, against `gplite_noisefun.m:176-210`):

| PyVBMC setting | `GaussianNoise` flags / MATLAB id | `sn2_star`, `s2_star` given | `sn2_star`, `s2_star=None` |
|---|---|---|---|
| uncertainty level 0 | `constant_add` / `[1 0 0]` | `exp(2h0)` (scalar; `s2_star` read by no branch) | `exp(2h0)` |
| level 2 (target returns noise) | `+user_provided_add` / `[1 1 0]` | `exp(2h0) + s2_star` (M,1) | `exp(2h0)` |
| level 1 (noise inferred) | `+scale_user_provided` / `[1 2 0]` | `exp(2h0) + exp(h1)*s2_star` (M,1) | `exp(2h0)` |
| not used by PyVBMC | `rectified_linear_…` / `[· · 1]` | adds `exp(2h)*max(0, h_thresh - y_star)^2`; contributes nothing when `y_star is None`, as MATLAB's `if ~isempty(y)` does | same |

In every case the whole `sn2_star`, the user-provided part included, is multiplied by the posterior's `sn2_mult`, exactly as `gplite_pred.m:121` does; `predict` additionally substitutes 1 for a `None` multiplier, a guard MATLAB has in `gplite_rnd.m:69` but not in `gplite_pred.m` (where `post.sn2_mult` is always set). With `parameters[0] == 0` (no constant term) both sides start from machine epsilon (`np.spacing(1.0)` / `eps`). Verified numerically: `ymu`/`ys2` agree with the transcription to ≤7.7e-14 in all three parameterizations, with and without `s2_star`.

**Is it combined over the hyperparameter samples as MATLAB combines it?** For `separate_samples=True`, yes — column for column, to 7.7e-14. For `separate_samples=False` with `Ns>1`, no, and the difference is one of kind: MATLAB never averages `lp` (only `fmu/ymu/fs2/ys2` are averaged at `gplite_pred.m:154-165`), so it returns the whole `(Nstar, Ns)` matrix; Python returns one column, the log density of the moment-matched Gaussian `N(mean of mu, mean of y_s2 + sample variance of mu)`. Measured gap against the mean of MATLAB's columns: 3.7e-2 to 6.0e-2 nats on a 25-point fit. This is documented in the docstring and is finding **F9**; no PyVBMC caller passes `return_lpd`.

**`predict_full`**: adds `sn2_star * sn2_mult`, and does so correctly only when `sn2_star` is a scalar (constant noise alone, or user-provided noise with `s2_star=None`). With a per-point noise it adds a broadcast column rather than a diagonal — finding **F1**.

**`quad`**: adds no noise at a test point (it integrates the latent function; so does `gplite_quad.m`). The only noise it uses is the normalization of the Cholesky factor in the variance: `sl = min(sn2 over the training set) * sn2_mult` (`:2196`), where MATLAB uses `exp(2*hyp(Ncov+1)) * sn2_mult` (`gplite_quad.m:66-67`). The two coincide exactly when the constant term is the whole training noise — PyVBMC's noiseless setting — and they part with user-provided noise, where MATLAB's is wrong and Python's right (finding **F8**), except after a rank-one update, where Python's recomputation is stale (finding **F2**).

---

## 3. Findings

### F1. `predict_full(add_noise=True)` adds a broadcast column instead of a diagonal for input-dependent noise
- Location: `gpyreg/gaussian_process.py:1852`; MATLAB: no counterpart (`gplite_pred.m` returns only the diagonal; `gplite_rnd.m:54` is the nearest relative)
- Category: indexing/shape
- Proposed classification: suspected defect (Python-only surface)
- Confidence: high
- History: the line dates from gpyreg's 2021-07-02 `3b39954`; its sample axis was corrected in `4ab4aee` ("Indexing"), the noise term itself never. No MATLAB line changed.
- The code is `cov[s, :, :] += np.dot(np.eye(N_star), sn2_star) * sn2_mult`. When `sn2_star` is a scalar, `np.dot(eye, scalar)` is `scalar*eye` and the result is the intended diagonal. When the noise function returns a per-point column `(M,1)` — user-provided noise with `s2_star`, or the output-dependent term with `y_star` — `eye @ sn2_star` is that same `(M,1)` column, which then broadcasts across all `M` columns of the covariance. Every entry of row `i` gains `sn2_star[i]`, so the matrix stops being symmetric and every off-diagonal entry is corrupted; only the diagonal comes out right. The intent is plainly `np.diag(np.ravel(sn2_star))`.
- Consequence if real: any caller who asks for the full predictive covariance *with* noise and a per-point noise variance gets a non-symmetric matrix whose off-diagonal entries are wrong by up to the full noise variance. Measured: with `s2_star = [0.5, 0.1, 0.3, 0.05]` the added matrix is the column `[0.56, 0.12, 0.34, 0.065]` replicated across all four columns, instead of that vector on the diagonal; the returned `cov` is not symmetric. The diagonal still agrees with `predict(add_noise=True)`. **Not reached by a default PyVBMC run**: no PyVBMC production code calls `predict_full` at all (only two test helpers, both with `add_noise=False`).
- Suggested reproduction: `c2_full_quad.py` (ran it; output above). Two lines: build a GP with `user_provided_add`, call `predict_full(Xs, s2_star=..., add_noise=True)` and subtract the `add_noise=False` covariance.
- Test adequacy: no. `test_gaussian_process.py:73-74` calls `predict_full(add_noise=True)` on a prior-only GP with constant noise and asserts nothing at all; `test_utils.gauss_hermite_quadrature_reference` uses `add_noise=False`.

### F2. `quad`'s variance recomputes the factorization scale instead of reading the posterior's stored `sl`
- Location: `gpyreg/gaussian_process.py:2188-2196` (used at `:2234`); MATLAB: `gplite_quad.m:66-67`, `:101`
- Category: state/caching
- Proposed classification: suspected defect (in the Python fix of a MATLAB defect)
- Confidence: high
- History: introduced by gpyreg `cec1f85` ("fix: quad variance with heteroskedastic noise", release 1.2.1), which replaced `sn2_eff = exp(2*hyp[cov_N])*sn2_mult` with `sl = np.min(sn2)*sn2_mult`. MATLAB unchanged since 2019.
- `L` is the Cholesky factor of `(K + sn2_mult*diag(sn2)) / sl` for the `sl` that held **when the factor was built**, which is exactly what `Posterior.sl` records — its docstring says so: "Rank-one updates extend the factorization with this scale, so it can differ from the minimum of the current training noise after observations have been appended". `quad` ignores that attribute and recomputes `min(sn2)*sn2_mult` from the current training set. After a rank-one update that appends a point whose total noise is below the previous minimum, the recomputed value is smaller than the true scale, `invKzk` is over-scaled, and `J_kk = nf_kk - sum(z*invKzk')` goes negative and is clamped to `eps`.
- Consequence if real: `F_var` collapses to `2.2e-16` where the true value is order 1e-2 — a silent, total loss of the integral's variance. Measured (`c2_full_quad.py`): after appending a point with `s2_new = 1e-5` to a GP whose stored `sl` is 0.02654, `quad(compute_var=True)` returns `[2.2e-16, 2.2e-16, 2.2e-16]` where the same posterior recomputed from scratch gives `[0.0284, 0.1046, 0.0260]`; the means agree to 3.8e-16. Triggers only with heteroskedastic training noise (`sn2` not constant) *and* a rank-one update *and* `compute_var=True`. **Not reached by a default PyVBMC run** (`GP.quad` has no PyVBMC caller — `pyvbmc/svbmc/_jacobian.py:81` is `scipy.integrate.quad`; PyVBMC reimplements the quadrature in `_gp_log_joint`).
- Suggested reproduction: as above; the one-line check is `abs(gp.posteriors[s].sl - np.min(sn2)*sn2_mult)` after a rank-one update with a low-noise point.
- Test adequacy: no. `test_quadrature_with_noise_matches_numerical_integration` covers heteroskedastic `quad` thoroughly (three parameterizations, Gauss–Hermite reference) but always on a freshly computed posterior, and `test_rank_one_update_with_heteroskedastic_noise` checks `predict`/`alpha`, never `quad`.

### F3. The mean-function bound recommendations use statistics of the whole of `X` where MATLAB uses per-dimension statistics
- Location: `gpyreg/mean_functions.py:488` (`w = np.max(X) - np.min(X)`), `:511-515` (`xm` bounds and `x0`), `:518-524` (`omega` bounds and `x0`); MATLAB: `gplite_meanfun.m:142`, `:220-230`
- Category: indexing/shape
- Proposed classification: port discrepancy (contradicts the characterization of the sheet entry "NumPy quantile and standard-deviation conventions in GP bound recommendations")
- Confidence: high
- History: present since the port (the same expressions appear in the pre-`3b39954` `MeanInfo` class). MATLAB's block unchanged since 2020-05-08, i.e. before gpyreg's history begins.
- MATLAB's `w = max(X) - min(X)`, `min(X)`, `max(X)`, `median(X)` and `std(X)` are all **per column** for a matrix `X`, so `dm.LB(2:D+1) = min(X) - 0.5*w` and the rest give each dimension its own box, and `dm.x0(D+2:2*D+1) = log(std(X))` gives each dimension its own starting scale. Python reduces `X` to scalars (`np.max(X)`, `np.min(X)`, `np.median(X)`, `np.std(X, ddof=1)` with no `axis`), so every dimension gets the same number. The sheet's entry describes the std difference as a convention that makes values "differ by a small amount that depends on `N`" and cites the code comment at `:522-523`; the `ddof=1` the code passes already matches MATLAB's normalization, and what actually differs is the axis, whose effect is not small and does not depend on `N`.
- Consequence if real: on a training set whose transformed coordinates differ in scale (D = 3, per-dimension σ of 0.4/1/4), measured against the transcription: `x0` for the quadratic mean's location `[-0.097, -0.097, -0.097]` against MATLAB's per-dimension medians `[-0.110, 2.159, -4.995]`; `x0` for `log omega` `[1.312]*3` against `[-0.635, 0.017, 1.311]` (a factor of 7 in the starting scale of dimension 0); `LB` for the location `-22.15` for all three against `[-2.36, -2.54, -21.82]`; `UB` `11.92` against `[2.56, 5.39, 10.93]`; `LB` for `log omega` `-10.98` against `[-12.92, -12.44, -11.02]` (Python's is the *tighter* one here); `PUB` `2.835` against `[0.90, 1.38, 2.80]`. The `mean_const` entry agrees apart from the listed quantile convention. **Reached by a default PyVBMC run**: `pyvbmc/vbmc/gaussian_process_train.py:335, 339, 365` puts `mean_bounds_info["x0"]` straight into `hyp0`, and PyVBMC leaves the location and scale bounds `NaN`, so `GP.fit` fills them from this same recommendation on the full training set (`get_recommended_bounds`, `fit` options default to `"current"`). The effect is on the fit's starting point, on the slice-sampler widths (`PUB-PLB`) and on hard bounds that can bind for the whole run.
- Suggested reproduction: `c3_bounds_batched.py` (ran it; numbers above). `NegativeQuadratic().get_bounds_info(X, y)` on an `X` with unequal per-dimension spread, against `gplite_meanfun('info', X, 4, y)`.
- Test adequacy: no. No test in gpyreg calls `get_bounds_info` on any mean function and checks a value; `test_mean_functions.py` tests error messages and `compute_batched` only.

### F4. The covariance's recommended starting length scales are the global standard deviation, not the per-dimension one
- Location: `gpyreg/covariance_functions.py:451` (`_bounds_info_helper`) and `:401` (`RationalQuadraticARD.get_bounds_info`); MATLAB: `gplite_covfun.m:126`
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- History: present since the port; `gplite_covfun.m` unchanged since 2019-06-17.
- The same function computes `width = np.max(X, axis=0) - np.min(X, axis=0)` per dimension (matching `LB`, `UB`, `PLB`, `PUB` to the last bit) but then `plausible_x0[0:D] = np.log(np.std(X, ddof=1))`, a scalar over all entries, where MATLAB has `dK.x0(1:D) = log(std(X))`, one per column. The inconsistency inside one function is itself the evidence that the axis was dropped by accident rather than chosen.
- Consequence if real: measured on the same D = 3 set, `x0` is `[1.312, 1.312, 1.312]` where MATLAB gives `[-0.635, 0.017, 1.311]` — dimension 0's starting length scale is `e^1.31 = 3.7` instead of `e^-0.64 = 0.53`, a factor of 7, and the bias is systematically upward (the pooled standard deviation is dominated by the widest dimension). `LB`/`UB`/`PLB`/`PUB` agree with MATLAB exactly. **Reached by a default PyVBMC run**: `gaussian_process_train.py:338, 365` puts `cov_bounds_info["x0"]` into `hyp0`, so the GP fit of every iteration starts from these length scales (PyVBMC separately takes `cov_bounds_info["LB"]`, which does match MATLAB).
- Suggested reproduction: `c3_bounds_batched.py`, section "covariance (SE-ARD) bound recommendations".
- Test adequacy: no. `test_covariance_functions.py` tests `compute` and the gradients only.

### F5. `__robust_cholesky`'s sign-convention step flips individual entries of the eigenvector matrix, destroying the factorization `random_function` draws from
- Location: `gpyreg/gaussian_process.py:2620-2622`; MATLAB: `gplite_rnd.m:97-99` (local `robustchol`)
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- History: entered with `009007b` ("added robust Cholesky decomposition"), early in gpyreg's history; `gplite_rnd.m` unchanged since 2019-11-29.
- MATLAB: `[~,maxidx] = max(abs(U),[],1); negidx = (U(maxidx + (0:n:(m-1)*n)) < 0); U(:,negidx) = -U(:,negidx);` — the linear index picks the one entry `(maxidx(j), j)` per column, giving a `1×m` mask, and whole *columns* are negated. Flipping an eigenvector's sign leaves `U D U'` unchanged, so MATLAB's step is cosmetic. Python writes `maxidx = np.argmax(np.abs(U), axis=0); negidx = U[maxidx] < 0; U[negidx] *= -1`. `U[maxidx]` is fancy *row* indexing: it returns an `(n, n)` matrix of re-ordered rows, so `negidx` is an `(n, n)` element mask unrelated to the column convention, and `U[negidx] *= -1` negates scattered individual entries of `U`. The columns are no longer orthonormal eigenvectors, so `T = diag(sqrt(D)) @ real(U[:, t]).T` no longer satisfies `T' T = C`.
- Consequence if real: on a rank-deficient `6×6` covariance (`c5_robustchol.py`), `max|T'T - C| = 4.33` against `max|C| = 4.36` — the factorization is unrelated to the matrix it should factor — where the MATLAB transcription gives `2.7e-15`. A `random_function` draw taken through this branch therefore has an arbitrary covariance rather than the GP's predictive covariance (the mean is unaffected). The branch is entered whenever `scipy.linalg.cholesky` of the predictive covariance fails, which is the normal case for a dense grid of test points — the very use `random_function` and `GP.plot` are for. **Not reached by a default PyVBMC run** (`random_function` has no PyVBMC caller). Note that F6 masks F5 in many grid cases: when negative eigenvalues survive the tolerance, `T` is zeroed before the corrupted `U` is used.
- Suggested reproduction: `c5_robustchol.py`, which prints both `max|T'T - C|` values and shows `U[maxidx]` has shape `(6,6)` with 15 of 36 entries flagged, where MATLAB's mask is the 6-element `[True False False False True True]`.
- Test adequacy: no. `test_random_function` draws functions and asserts only shapes of the subsequent `predict` calls; nothing checks the second moment of the draws. (It uses `Matern(1)`, whose small training set factorizes, so it does not even enter the branch.)

### F6. `random_function` silently returns the predictive mean when the covariance has numerical-noise negative eigenvalues
- Location: `gpyreg/gaussian_process.py:2632-2635` and `:2594`; MATLAB: `gplite_rnd.m:105-111`, `:61`
- Category: control flow
- Proposed classification: suspected defect in both (the trigger), with a Python-only silent failure
- Confidence: high
- History: same commit as F5; MATLAB unchanged.
- Both sides filter eigenvalues by `|D| > eps(max(D)) * length(D)` and then refuse to build a factor if any surviving eigenvalue is negative. On a GP predictive covariance the surviving set routinely contains eigenvalues of order `-5e-16` while the tolerance is `~7e-17`, so the refusal triggers on ordinary rounding noise. What the two do next differs: MATLAB sets `T = zeros(0,'like',Sigma)` and `Fstar = T'*randn(size(T,1),1) + fmu` then fails loudly on incompatible sizes; Python sets `T = np.zeros(sigma.shape)` (the full `(n,n)`), so `f_star = T.T @ rng.standard_normal((n,1)) + f_mu` returns the predictive mean exactly, with no error, no warning, and the random stream advanced by `n` normals.
- Consequence if real: `random_function` returns a deterministic function — the posterior mean — for any moderately dense set of test points. Measured (`c6_eig.py`): on a 40-point grid conditioned on 8 training points, 10 of the 37 surviving eigenvalues are `~-5e-16`, `T` is all zeros, and two draws made with different generators are bit-identical to `predict`'s mean; over 4000 draws the empirical covariance is zero against a predictive covariance of scale 2.4e-3. **Not reached by a default PyVBMC run.** The underlying tolerance test is MATLAB's own, so the trigger is shared; the silence is Python's.
- Suggested reproduction: `c6_eig.py` — `gp.random_function(Xs, rng=...)` twice on a 40-point grid and compare.
- Test adequacy: no; same gap as F5.

### F7. `RationalQuadraticARD.get_bounds_info` writes the shape parameter's plausible upper bound into the output scale's slot
- Location: `gpyreg/covariance_functions.py:414`; MATLAB: no counterpart (no rational-quadratic kernel in gplite)
- Category: indexing/shape
- Proposed classification: suspected defect (Python-only kernel)
- Confidence: high
- History: predates `2338bb3` (which only reformatted the line, turning `plausible_upper_bounds[D] = 5.` into `= 5.0`); no MATLAB line involved.
- The block that initializes the shape hyperparameter (`cov_N = D+2`, so the shape is index `D+1`) reads `lower_bounds[-1] = -5.0; upper_bounds[-1] = 5; plausible_lower_bounds[-1] = -5.0; plausible_upper_bounds[D] = 5.0; plausible_x0[-1] = 1.0`. The fourth line indexes `D` where the other four index `-1`. It therefore (a) overwrites the output scale's `PUB`, which the preceding block had set to `log(height)`, with the constant 5.0, and (b) leaves the shape's own `PUB` at `+inf`.
- Consequence if real: measured, `PUB = [0.900, 1.377, 2.796, 5.0, inf]` where the output scale's recommendation should be `log(height) = 2.383` and the shape's should be 5.0. A plausible box of infinite width feeds whatever consumes `PUB` (sampler widths, the plausible-box priors), and the output scale's plausible upper bound is decoupled from the data. The sheet's entry "Isotropic kernels and the rational-quadratic kernel are Python-only" covers the kernel's existence, not this line. **Not reached by a default PyVBMC run** (PyVBMC hard-wires SE-ARD).
- Suggested reproduction: `c3_bounds_batched.py`, section "RationalQuadraticARD bounds" (ran it; output above).
- Test adequacy: no. `test_covariance_functions.py` has four rational-quadratic tests, all on `compute` and the gradient; none calls `get_bounds_info`.

### F8. `quad`'s variance with heteroskedastic noise deliberately diverges from MATLAB, and the sheet has no entry
- Location: `gpyreg/gaussian_process.py:2193-2196`; MATLAB: `gplite_quad.m:66-67`
- Category: formula
- Proposed classification: possibly intentional (a fix of a MATLAB-side defect), missing from the known-differences sheet
- Confidence: high
- History: gpyreg `cec1f85` (release 1.2.1) changed the Python; MATLAB unchanged since 2019-06-30.
- MATLAB normalizes `invKzk` by `sn2_eff = exp(2*hyp(Ncov+1)) * sn2_mult`, the **constant** noise hyperparameter, whereas `gplite_core.m` normalizes `L` by `sl = min(sn2) * sn2_mult` with `sn2` the **total** training noise vector. The two coincide only when the constant term is the whole noise. Where they differ, MATLAB's `J_kk` is over-corrected and `max(eps, J_kk)` clamps the variance to `eps`. gpyreg now uses the total minimum, which is the quantity `L` actually carries.
- Consequence if real: gpyreg's `quad` variance is right and MATLAB's is not. Measured against a 2001-point grid integration of `predict_full`'s latent covariance (`c4_quad_truth.py`): with user-provided noise, truth `5.5991764157e-02`, gpyreg `5.5991764157e-02`, the MATLAB formula `2.2204460493e-16`; with constant noise alone all three agree to ten digits. The means agree in both cases. So this is a Python improvement, not a regression — but it is a documented difference in the numerics of a public method that the sheet's G-section does not list (the G entries cover `intmeanfun`, output warping, the mean families, `qpred`, `sample`/`fmin`, the sampler, `derivcheck`, the design, the quantile/std conventions, the isotropic kernels, `return_cross_covariance`, `_get_hyp_cov`, `gpsample`, noise shaping, the lean GP history, the HPD lower bounds, the hyperparameter sampler and `gp.t`). **Not reached by a default PyVBMC run** (no `GP.quad` caller), and inert for a noiseless target even if it were.
- Suggested reproduction: `c4_quad_truth.py` (ran it; numbers above).
- Test adequacy: yes — `test_quadrature_with_noise_matches_numerical_integration` pins exactly this, against a Gauss–Hermite reference, in all three parameterizations. It is the specification, not a mirror of the implementation.

### F9. The lpd of averaged hyperparameter samples is the moment-matched Gaussian's, where MATLAB returns the per-sample matrix
- Location: `gpyreg/gaussian_process.py:2046-2064`; MATLAB: `gplite_pred.m:124-127` with `:154-165`
- Category: formula
- Proposed classification: possibly intentional (documented in the docstring), missing from the known-differences sheet
- Confidence: high
- History: `gplite_pred.m:126` changed in `68a197b` (2022-06-25), after gpyreg's `return_lpd` (`9b3d46b`, 2022-06-02) — but the change was to the variance, not to the combination, which MATLAB has never averaged.
- MATLAB computes `lp` per sample and leaves it as an `(Nstar, Ns)` matrix even when `ssflag` is false and the moments are averaged; a caller who wants one number decides how to pool. Python, when `separate_samples=False`, returns a single column: the log density of `N(mean_s mu_s, mean_s y_s2_s + var_s mu_s)`. Neither the mean of MATLAB's columns nor the log of the mean of the densities. The docstring says "If separate_samples is `False`, returns the lpd of the corresponding mean approximation", so the choice looks deliberate; it is just not on the sheet.
- Consequence if real: a caller comparing gpyreg's averaged lpd with MATLAB's gets different numbers and a different shape. Measured: 3.7e-2 to 6.0e-2 nats against the mean of MATLAB's columns, over three hyperparameter samples on a 25-point fit; with `separate_samples=True` the two agree to 7.7e-14. **Not reached by a default PyVBMC run**: no PyVBMC production call site passes `return_lpd` (the callers are `abstract_acq_fcn.py:207`, `acq_fcn_imiqr.py:272`, `acq_fcn_viqr.py:146,529`, `active_importance_sampling.py:98,216,254,351`, `vbmc.py:1645,1760,2543`).
- Suggested reproduction: `c1_predict.py`, the `ss=False` rows with `Ns=3`.
- Test adequacy: partly, and in the wrong direction — see the test note on `test_predict_lpd` below.

### F10. The isotropic kernels' recommended bounds take the log of the mean width where MATLAB takes the mean of the log widths
- Location: `gpyreg/isotropic_covariance_functions.py:233-246`; MATLAB: `gplite_covfun.m:112-119` (the `isoflag` branch)
- Category: formula
- Proposed classification: port discrepancy of a low-reach path / possibly intentional
- Confidence: medium (the sheet declares the isotropic *kernels* Python-only, but this bound branch does have a MATLAB counterpart, which is why I report it)
- History: MATLAB's `isoflag` bound branch unchanged since 2019-06-17.
- MATLAB computes `mean(log(width))`, `mean(log(width*10))` and `mean(log(std(X)))` — logs first, then the mean, i.e. the geometric mean of the per-dimension widths. Python computes `width = np.mean(max-min)` and then `np.log(min_width)` / `np.log(max_width*10)` — the arithmetic mean, then the log — and `np.log(np.std(X, ddof=1))`, the pooled scalar standard deviation rather than the mean of the per-dimension logs. The `min_width = np.min(width)` / `max_width = np.max(width)` lines are no-ops, `width` already being a scalar; they read as the residue of an intent to use the per-dimension vector, as MATLAB's `meanfun == 10` branch does with `min(log(w))`/`max(log(w))`.
- Consequence if real: on the D = 3 set, `LB = [-11.788, …]` against MATLAB's `[-12.125, …]`, `UB = 4.330` against `3.993`, `PUB = 2.028` against `1.691`, `x0 = 1.312` against `0.231`. By Jensen the Python bound is always the looser one, and the gap grows with the spread of the per-dimension widths. **Not reached by a default PyVBMC run**; the sheet notes MATLAB has no isotropic *compute* branch either, so no MATLAB run reaches it.
- Suggested reproduction: `c3_bounds_batched.py`, section "isotropic SE bounds".
- Test adequacy: no; `test_isotropic_covariance_functions.py` covers `compute` and the gradients only.

### F11. The Matern degree-1 length-scale gradient is NaN on the diagonal, in both implementations
- Location: `gpyreg/covariance_functions.py:221` (`self.df = lambda t: 1 / t`) with `:289`, and `gpyreg/isotropic_covariance_functions.py:156`; MATLAB: `gplite_covfun.m:198`, `:218`
- Category: formula/gradient
- Proposed classification: suspected defect in both
- Confidence: high
- History: MATLAB unchanged since 2019-06-17; the Python is a faithful transcription, including the `df(t) = (f(t)-f'(t))/t` comment.
- For degree 1, `df(t) = 1/t`, and the gradient is `sf2 * (df(tmp)*exp(-tmp)) * Ki`. On the diagonal `tmp = 0` and `Ki = 0`, so the product is `inf * 0 = NaN` in NumPy and in MATLAB alike. The true derivative there is 0 (the kernel is constant `sf2` at coincident inputs). MATLAB carries the fix as a commented-out line immediately below (`% dK(:,:,i) = dK(:,:,i).*(Ki > 1e-12); % fix numerical errors`); gpyreg carries a comment asserting the NaN is acceptable ("This is OK, the kernel is just not differentiable there").
- Consequence if real: the NaN propagates through `dnlZ[i] = np.sum(np.sum(Q * dK[:, :, i]))/2`, so the whole marginal-likelihood gradient with respect to a length scale is NaN and `GP.fit` cannot optimize a Matern-1 kernel. Measured: `log_likelihood(hyp, compute_grad=True)` on a 15-point 1-D set returns `dnlZ = [nan, -11.609, -0.475, -0.051]` for `Matern(1)` and finite values for `Matern(3)`; the isotropic Matern-1 gradient has the same NaNs. `Matern(3)` and `Matern(5)` have an exact 0 on the diagonal and are unaffected. **Not reached by a default PyVBMC run** (SE-ARD is hard-wired); reachable by any gpyreg user who picks `Matern(1)`.
- Suggested reproduction: `c8_matern1.py` and `c9_matern1_fit.py` (ran both; output above).
- Test adequacy: no. `test_matern_kernel_gradient` and `test_matern_isotropic_kernel_gradient` both instantiate degree 3 only, so the one degree with the singularity is never differentiated.

### Minor observations
- **M1.** `quad` validates the covariance function (raises for anything but a `SquaredExponential` subclass — which correctly admits `SquaredExponentialIsotropic` and correctly rejects `Matern`, `MaternIsotropic` and `RationalQuadraticARD`) but not the mean function, where `gplite_quad.m:16-19` errors for any `meanfun` outside `[0 1 4 6 8]`. With gpyreg's three mean families the gap is empty; a user-supplied mean object would have `hyp[cov_N+noise_N]` silently taken as `m0` and the rest of its hyperparameters ignored.
- **M2.** `quad:2196` and `:2226-2237` dereference `self.posteriors[s].sn2_mult` without the `is None` guard that `predict:2026` and `predict_full:1846` carry; after `GP.clean()` or `update(compute_posterior=False)` this is a `TypeError` rather than a clear message (it would fail on `alpha` anyway).
- **M3.** `quad` accepts a scalar `mu` (tiled to `(1, D)`) and a `(1, D)` or `(N, D)` `sigma` (MATLAB's `repmat` is covered by NumPy broadcasting, verified), but a 1-D `mu` of shape `(D,)` with `D > 1` is read as `N_star = D` and then raises on `mu[:, i]`, where MATLAB's `size(mu,1)` treats a row vector as one measure.
- **M4.** `predict_full` never clamps the diagonal at zero, where `gplite_pred.m:120` applies `max(fs2, 0)` to the variances it returns (and `predict:2023` does the same). Defensible for a covariance matrix, but it makes `predict_full`'s diagonal and `predict`'s variance disagree in sign near machine precision — worth knowing, since gpyreg's own `quad` reference helper integrates `predict_full`.
- **M5.** `predict_full`'s docstring says `add_noise : bool, defaults to True` while the signature defaults to `False`. The `get_bounds_info` docstrings of all four component modules declare shapes `(n, 1)`; the returned arrays are `(n,)`.
- **M6.** `_convert_shapes` accepts `s2` only as `float`, `int` or `ndarray`; a 0-d array or a NumPy integer scalar (`np.int64` is not a Python `int`) falls into the `TypeError` branch. Where MATLAB raises named errors on a row-count mismatch of `ystar`/`s2star` (`gplite_pred.m:16-23`), Python relies on `reshape(N, 1)`, which also silently accepts a `(1, N)` input.
- **M7.** `update(hyp=...)` accepts a hyperparameter row of the wrong width when the GP has no training data: a 12-entry row on an 11-hyperparameter GP is stored as is, and `predict` then reads the blocks by offset, so a hyperparameter intended as the noise is consumed as the mean constant. (`update` belongs to another slice; it matters here because `test_predict_lpd` depends on it — see below.)
- **M8.** `predict` raises `ValueError` when `return_lpd=True` and `y_star is None`, where `gplite_pred.m:32-36` returns `lp = []`. A deliberate, harmless strictness.
- **M9.** `Matern.__init__` requires an explicit degree, where `gplite_covfun.m:195` defaults to 5 when no feature is given — no silent mismatch, since gpyreg has no default to disagree with.
- **M10.** The remaining component formulas and gradients I re-derived all match MATLAB exactly: the SE-ARD kernel and its two gradients, the Matern 3/5 kernels and gradients, the negative-quadratic mean (`m = m0 - 0.5*sum(z2)`, `dm = [1, (X-xm)/omega^2, z2]`, the `sgn = -1` specialization of `gplite_meanfun.m:425-436`), the constant and zero means, and all four noise terms with their gradients. `sq_dist`'s centering is replaced by `cdist`, which changes results only at rounding level (≤4e-14 in my comparisons).

---

## 4. Test adequacy notes

- **`test_predict_lpd` (`test_gaussian_process.py:1206`) mirrors the implementation, not the specification.** It sets `s2_star = np.arange(-3, 3).reshape((-1, 1))` and then immediately overwrites it with `np.zeros((6, 1))`, so the user-provided noise contributes nothing and the `np.pi * s2_star` term in its assertion is vacuous. Its noise function is `GaussianNoise(user_provided_add=True)`, which has **zero** hyperparameters, while its `hyp` rows carry 12 entries for the 11 the GP expects (D = 3: 4 covariance + 0 noise + 7 mean); `update(hyp=...)` does not check the width, so the `np.log(np.pi)` entry written as the noise is read as the mean constant and the whole mean block is shifted by one (verified in `c7_lpdtest.py`: the mean block `predict` reads begins `[1.1447, -2.7568, 0, …]`, and `sn2_star` at the test points is `2.2e-16`). The test still passes because the expected value is recomputed from the very `f_mu`/`f_s2` that `predict` returned. Consequently it pins neither which variance enters the lpd in a realistic parameterization, nor `sn2_mult`, nor the combination over samples — its two hyperparameter samples are identical, so `separate_samples` is untested in substance (F9). `test_random_function` adds four shape assertions on `return_lpd` and no values.
- **`predict_full`'s noise term is untested.** The two calls with `add_noise=True` (`test_gaussian_process.py:73`, `test_gaussian_process_isotropic.py:76`) are bare smoke calls on a prior-only GP with constant noise, and every value-checking use passes `add_noise=False` (F1).
- **No test anywhere calls `get_bounds_info` and checks a value**, for any covariance, mean, noise or isotropic component. `test_mean_functions.py` and `test_noise_functions.py` assert only error messages (plus the batched-mean equality); `test_covariance_functions.py` and `test_isotropic_covariance_functions.py` only `compute` and gradients. This leaves F3, F4, F7 and F10 entirely uncovered, including the one line (F7) that is a plain index slip.
- **`random_function` is tested for shapes alone.** Neither the eigendecomposition fallback nor the second moment of the draws is exercised, so F5 and F6 are invisible; the test's `Matern(1)` training set is small enough to factorize, so the fallback is not even entered.
- **The Matern gradient tests pin degree 3 only**, the one degree whose `df` has no pole, so F11's NaN never appears in a test.
- **What is well tested, and against the specification rather than the implementation:** `test_quadrature_with_noise_matches_numerical_integration` and `test_quadrature_with_noise_fitting` check `quad`'s mean *and* variance against independent numerical integration, per hyperparameter sample and averaged, in all three noise parameterizations — a genuine gate for F8. `test_compute_batched_matches_compute` asserts bit-identity across four dimensionalities. `test_solve_triangular_matches_scipy` pins `_solve_triangular` against scipy for both layouts and all `trans` values. `test_predict_cross_covariance.py` (351 lines) covers the `return_cross_covariance` contract, including the defensive-copy fallback for overridden `compute`. The gap in all of these is `quad` after a rank-one update (F2).
