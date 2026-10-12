# P7 — variational posterior, entropies, VP statistics, MATLAB-comparison track

Raw reviewer report, wave 5 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, fresh
context, read-only on the repositories, reading the code at `f873556`; it
compared the Python code with MATLAB VBMC at `396d649`. Its brief carried
two first questions, which the report answers before its findings. Its check scripts
ran in the session scratchpad and are kept only on the machine that ran them
(`dev/scripts/runs/LOCAL.md`). The text below is the agent's final message,
unedited.

---

# P7 — variational posterior, entropies, VP statistics (MATLAB comparison track)

## 1. Coverage

**Read completely (Python).** `pyvbmc/variational_posterior/variational_posterior.py` (whole file; `plot`, `to_torch`, `to_arviz` skimmed as out of slice), `pyvbmc/entropy/entlb_vbmc.py`, `pyvbmc/entropy/entmc_vbmc.py`, `pyvbmc/stats/get_hpd.py`, `pyvbmc/stats/kde_1d.py`, `pyvbmc/stats/kl_div_mvn.py`, `pyvbmc/decorators/handle_0D_1D_input.py`.

**Read completely (MATLAB).** `vbmc_pdf.m`, `vbmc_rnd.m`, `vbmc_moments.m`, `vbmc_kldiv.m`, `vbmc_mtv.m`, `vbmc_mode.m`, `ent/entlb_vbmc.m`, `ent/entmc_vbmc.m`, `shared/mvnkl.m`, `shared/kde1d.m`, `shared/qtrapz.m`, `misc/gethpd_vbmc.m`, `misc/vpbounds.m`, `misc/get_vptheta.m`, `misc/rescale_params.m`.

**Established file-to-file map** (all confirmed by reading both sides): `sample`↔`vbmc_rnd.m`, `pdf`/`log_pdf`↔`vbmc_pdf.m`, `moments`↔`vbmc_moments.m`, `mode`↔`vbmc_mode.m`, `kl_div`↔`vbmc_kldiv.m`, `mtv`↔`vbmc_mtv.m` (+`shared/qtrapz.m`), `get_bounds`↔`misc/vpbounds.m`, `get_parameters`↔`misc/get_vptheta.m`, `set_parameters`↔`misc/rescale_params.m` (the `theta` branch), `stats/kl_div_mvn.py`↔`shared/mvnkl.m`, `stats/kde_1d.py`↔`shared/kde1d.m`, `stats/get_hpd.py`↔`misc/gethpd_vbmc.m`. `__init__` has no single counterpart: `misc/setupvars_vbmc.m:78-85` builds the struct PyVBMC's constructor builds.

**Confirmed unported, no further time spent:** `vbmc_power.m`, `ent/entub_vbmc.m`, `misc/vptrain2real.m` — `pyvbmc/entropy/` holds only the two ported files, `VariationalPosterior` has no `temperature`, and the only occurrences of `vptrain2real` in the package are the two commented-out lines at `pyvbmc/vbmc/vbmc.py:1376` and `:1537`. Matches the sheet's P7 entries.

**Skimmed / followed out of the slice:** every caller of `vp.sample`, `vp.pdf`/`log_pdf`, `vp.moments`, `vp.mode`, `vp.kl_div`, `vp.mtv`, `get_hpd`, `kde_1d`, `kl_div_mvn`, `get_bounds`, `get_parameters`, `set_parameters` in `pyvbmc/` (grep, then read at each site); on the MATLAB side every caller of `vbmc_pdf`/`vbmc_rnd`/`vbmc_moments`/`vbmc_kldiv`/`vbmc_mtv`/`get_vptheta`/`rescale_params`; `shared/warpvars_vbmc.m` `case 'd'`; `private/activesample_vbmc.m:95-120`, `:565-630`; `acq/acqf_vbmc.m`, `acq/acqflog_vbmc.m`; `misc/fess_vbmc.m`. Tests: `testing/variational_posterior/test_variational_posterior.py` (whole), its `FIXTURES.md`, `testing/stats/*` (whole), `testing/entropy/FIXTURES.md` and the head of `test_entmc_vbmc.py`. Sheet: sections P7, P6, P1b, "Settled non-differences", plus the intro.

**Checks run** (scripts and output in `…/scratchpad/wave5_P7_comparison`, `OMP/OPENBLAS/MKL_NUM_THREADS=1`, `pyvbmc.__file__` = the checkout):
- `c1` mask semantics of `q1 == 0 | np.isinf(q1)`.
- `c2`, `c9` MATLAB vs PyVBMC remainder weights of the balanced sampler, on toy weights and on the K=50 MATLAB weights of `vp-test.npz`.
- `c3` the positivity check of `set_parameters(raw_flag=False)` over four (D,K).
- `c4` `kl_div(samples=…, gauss_flag=True)` against the MATLAB-equivalent computation.
- `c5` `kde_1d` against a line-by-line transcription of `shared/kde1d.m` (MATLAB's `histc` binning, `dct1d`/`idct1d`, `fixed_point`, `root`).
- `c6` `entlb_vbmc` and `entmc_vbmc` against literal transcriptions of the two MATLAB files, for `(D,K)` = (2,3), (3,1), (1,4), (4,5), both `jacobian_flag` values, with the Monte Carlo draws injected so the sample paths are identical.
- `c7` `get_hpd` rounding, `sample` index dtype, `get_parameters` in-place rescaling.
- `c8` `qtrapz` vs `scipy.trapezoid`, normalization of the original-space density (Jacobian direction), constructor with a column-vector `x0`.
- ad-hoc: integer input to `vp.pdf`; non-finite densities near a bound.
No test suite was run, no `optimize()` run, nothing written into the three repositories.

**Headline result on the entropies:** both entropy functions reproduce the MATLAB files to machine precision (max gradient difference ≤ 8.9e-16 over all eight configurations, value differences ≤ 2.2e-16), including the `K == 1` branch and the lambda-gradient convention, which the two sides express differently but equivalently (MATLAB accumulates dH/dlog λ and divides by λ when `jacobian_flag` is false; PyVBMC accumulates dH/dλ and multiplies when it is true). The post-2021 MATLAB fix `1b72896` (weights gradient of the Monte Carlo entropy, "Fix by Chengkun Li") is present in the Python. **I have no findings in `entlb_vbmc.py` or `entmc_vbmc.py`.**

---

## 2. First questions

### Q1. Where can an underflowed (or zero) log density reach?

`vp.pdf(..., log_flag=True)` takes `log` of the sum over components (`variational_posterior.py:905-911`), exactly as `vbmc_pdf.m:107-110` does, so both sides return `-inf` far from the posterior. PyVBMC adds two things MATLAB does not: `y[y == 0] = -inf` before the log (only to avoid the warning — the value is MATLAB's) and `y[~mask] = -inf` for points on or outside the original bounds.

**The original-space path.** For `orig_flag=True` PyVBMC masks with strict inequalities (`:791-795`), transforms only the rows inside, and then subtracts the log-Jacobian on those rows only (`:918-928`). MATLAB has no mask: `vbmc_pdf.m:36-39` warps every row, and for a bounded variable `shared/warpvars_vbmc.m:104-110` takes `log` of a negative number for an out-of-bounds point, which in MATLAB is **complex**; on the bound itself it gives `±Inf`, and the Jacobian correction then yields `NaN`. So PyVBMC returns `0` / `-inf` where MATLAB returns complex or `NaN` garbage. This is a Python-only guard and an improvement; it is not on the sheet (F10 below). The Jacobian direction is right: `∫ vp.pdf(x) dx = 0.9999999999999998` over a bounded 1-D transform (check `c8`), and `log(pdf)` agrees with `log_flag=True` to 2.8e-14 in the interior.

**Readers inside the slice.**
- `mode` (`:1230-1240`) minimizes `-log q`. At an underflowed point the objective is `+inf`; starting points are component means (MATLAB) or posterior samples (PyVBMC), where the density is never zero, so neither side reaches it in practice. Same class of behavior on both sides.
- `kl_div(gauss_flag=False)` (`:1487-1502`) is the one reader that neutralizes zero/non-finite densities. It is the site of **F2**: the Python mask does not catch non-finite densities, MATLAB's `~isfinite` does. Zero densities (the common case, e.g. a sample of `vp1` outside `vp2`'s bounds, which the mask sends to `0`) are caught identically on both sides.
- `mtv` never reads the density; it works from samples and the KDE.
- `entmc_vbmc` takes `np.log(q)` of its own mixture density at samples drawn from its own components, as `ent/entmc_vbmc.m:67` does; `entlb_vbmc` takes `log(gammasum)`, as `ent/entlb_vbmc.m:83` does. Neither floors, on either side; both would produce `-inf`/`inf` in the same circumstances. No difference.

**Readers outside the slice** — all four match their MATLAB counterparts:
- `acquisition_functions/acq_fcn.py:36`, `acq_fcn_vanilla.py:36`, `acq_fcn_noisy.py:36`: `np.maximum(vp.pdf(Xs, orig_flag=False), realmin)` = `acq/acqf_vbmc.m:7`, `acq/acqfsn2_vbmc.m:7`, `acq/acqus_vbmc.m:7`.
- `acq_fcn_log.py:42`: `np.maximum(vp.pdf(..., log_flag=True), np.log(realmin))` against `acq/acqflog_vbmc.m:14`'s `log(max(p,realmin))`. Equivalent, including the underflow case: PyVBMC's `-inf` and MATLAB's `log(realmin)` both floor to `log(realmin)`.
- `vbmc/active_importance_sampling.py:488` (`fess`) and `:378`: `max(log-pdf, log(realmin))` matching `misc/fess_vbmc.m:26` and `private/activeimportancesampling_vbmc.m:95`, `:222`, `:313`, `:320`, `:352`.
- `calibration/_campaign.py:121` is Python-only (no MATLAB counterpart).
- The main loop's diagnostics never read the log density: `sKL` goes through `kl_div` with `kl_gauss = True` by default (`advanced_vbmc_options.ini:147`, MATLAB `defopts.KLgauss='yes'`), i.e. the moment-based branch, and `vbmc.py:1677` only draws samples.

**Can a non-finite density actually arise?** I could not produce one. With a probit transform and `sigma` up to 6 the original-space density does rise steeply toward a bound (4.3e6 at `x = 1e-16` from the bound) but then underflows to `0` rather than overflowing, because the transformed density underflows first. So F2's practical exposure is small; the mechanism is nonetheless a regression from correct code.

### Q2. `vp.sample` returning a pair, and the sampler itself

**Every caller in the package**, checked one by one: `variational_posterior.py:552` (`to_arviz`), `:1167` (`moments`), `:1251` (`mode`), `:1346`, `:1348` (`mtv`), `:1489`, `:1496` (`kl_div`), `:1571` (`plot`); `vbmc/vbmc.py:1677`; `vbmc/active_sample.py:968`, `:1065`; `vbmc/active_importance_sampling.py:81`, `:152`; `svbmc/svbmc.py:932`; `pymc/_target.py:1402`. All unpack the pair.

**The one exception is `pyvbmc/vbmc/active_importance_sampling.py:472`**, `X = vp.sample(N, orig_flag=False)` inside `fess`, where MATLAB's `misc/fess_vbmc.m:9` writes `X = vbmc_rnd(vp,N,0)` and so takes the first output alone. `X` is then the tuple, and the next statements (`gp.predict(X)`, `X.shape[0]`) fail. The path is only reached when `fess` is called with a scalar third argument, which no live call site does (`active_importance_sampling.py:88` passes the array `Xa`), so it is dormant. It lies in P4's file, and the untracked script `verification/scripts/wave4_P4_2_fess_scalar.py` suggests wave 4 already probed it; I report it here only because the question asks for every caller (F9).

**`sample` against `vbmc_rnd.m`, line by line.** Agreeing: the `N < 1` early return (`(0,D)` and `(0,1)`); the `K > 1` unbalanced draw (`catrnd` vs `rng.choice`, same distribution); the floor-then-remainder structure of the balanced draw; "shuffle and take N" (`randperm(numel(I),N)` vs `shuffle` + `[:N]`); the Gaussian and heavy-tailed formulas, with `t = df/2/sqrt(gamrnd(df/2,df/2))` matching `rng.gamma(df/2, df/2)` (MATLAB's `gamrnd` and NumPy's `gamma` both take shape and *scale*); the `K == 1` branch; and the `orig_flag` inverse transform. **One disagreement: the remainder weights (F3)** — PyVBMC applies MATLAB's `w*delta_extra` correction elementwise rather than with the total, because a builtin `sum` is used on a `(1,K)` array. `df` and `orig_flag` are faithful. The `'gp'` sampling path (`vbmc_rnd.m:45-49` → `misc/gpsample_vbmc.m`) is absent behind the dead local `gp_sample = False` at `:622`; that is the sheet's `gpsample_vbmc.m` entry, and the code matches its description. Draw counts and order are not comparable with MATLAB's stream at all (sheet, P1b randomness).

Minor shape/dtype note: the returned index is `int64`, shape `(N,)` for `K>1`; `float64`, shape `(N,)` for `K==1` (`i = np.zeros(N)`, `:690`); `float64`, shape `(0,1)` for `N<1`. MATLAB returns `ones(N,1)` (1-based) in the `K==1` case. The docstring promises `(N,1)`. Nothing reads the second output except tests, so this is a consistency wrinkle, not a defect.

---

## 3. Findings

### F1. `kl_div(samples=…, gauss_flag=True)` uses the mean of all entries instead of the per-dimension mean
- Location: `pyvbmc/variational_posterior/variational_posterior.py:1482`; MATLAB: `vbmc_kldiv.m:63` (`q2mu = mean(vp2,1)`)
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- History: the MATLAB line is unchanged since 2018 (`c4bde11`/`cba3a98`). The Python line entered with the first port of the method, `fdb1e56d` (2021-04-22, "feat: add kldiv to VariationalPosterior"), already as `np.mean(samples)`; it never matched MATLAB.
- What it does: `q2mu = np.mean(samples)` collapses the `(N,D)` sample matrix to a single scalar. `kl_div_mvn` then reshapes it to `(1,1)` and `dmu = mu2 - mu1` broadcasts the same scalar into every coordinate, so the second distribution is treated as centered at `(m,…,m)` with `m` the grand mean. MATLAB takes the column-wise mean, a `1×D` row. The companion line `q2sigma = np.cov(samples.T)` is correct.
- Consequence: `vp.kl_div(samples=…, gauss_flag=True)` returns a meaningless number whenever `D > 1` and the coordinates do not share a mean. In my reproduction (`c4`), a `D=2` standard posterior at the origin against 20000 samples from a Gaussian at `(3,−3)`: PyVBMC returns `[2.8e-4, 2.9e-4]`, the MATLAB-equivalent computation returns `[9.05, 8.88]` — a KL of zero reported for two well-separated distributions. Not reached from `optimize()` (the loop always passes `vp2=`), so this is a user-facing API defect.
- Suggested reproduction: `c4_kldiv_gauss_samples.py` in the scratch directory; ran it, numbers above.
- Test adequacy: no. `test_kl_div_two_vp_identical_samples_gauss_flag` draws the samples from the *same* posterior, whose means are equal in every coordinate (all components at `x0 = 5`), and `test_kl_div_two_vp_samples_gauss_flag` is `D = 1`. Both are blind to the defect by construction.

### F2. The non-finite guard of `kl_div` is neutralized by Python operator precedence
- Location: `pyvbmc/variational_posterior/variational_posterior.py:1492-1493`, `:1499-1500`; MATLAB: `vbmc_kldiv.m:75-76`, `:82-83`
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high on the mechanism, low on the consequence
- History: MATLAB unchanged since 2018. The first Python port (`fdb1e56d`, 2021-04-22) wrote `q1[np.logical_or(q1 == 0, np.isinf(q1))] = 1`, which is correct for zeros and infinities. Commit `4a071d8c` (2022-09-13, "pdf reference fix (#96)") rewrote it to `q1[q1 == 0 | np.isinf(q1)] = 1.0`. So the Python matched MATLAB's Inf handling for 17 months and then regressed.
- What it does: `|` binds tighter than `==` in Python, so the expression is `q1 == (0 | np.isinf(q1))`, i.e. an int array of 0/1. The mask therefore selects `q1 == 0` where the density is finite and `q1 == 1` where it is infinite — never true. Infinities pass through to `np.log`, giving `+inf`, and the KL estimate becomes `±inf`. Separately, and already true of the 2021 code, `np.isinf` does not catch `NaN` where MATLAB's `~isfinite(q1)` does, so a `NaN` density makes the estimate `NaN` on the Python side only. Verified in `c1`: for `q = [0, 1, inf, nan, 0.5]` the Python mask selects only the first entry, MATLAB's selects the first, third and fourth.
- Consequence: only when `vp.pdf` returns `inf` or `NaN` in original coordinates. I tried to produce one (probit transform, `sigma` up to 6, 2e5 samples, and direct evaluation at `1e-16` from a bound) and got steep growth followed by underflow to `0`, never a non-finite value — so the exposure looks small. If it happens, `kl_div(gauss_flag=False)` returns `inf`/`NaN` instead of a finite number, and a run configured with `kl_gauss=False` would record a `NaN` `sKL` into the iteration history and the reliability index. The default `kl_gauss=True` avoids the branch entirely.
- Suggested reproduction: `c1_kldiv_masks.py`; ran it. A direct fix check would be `np.logical_or(q1 == 0, ~np.isfinite(q1))`.
- Test adequacy: no test constructs a zero, infinite or NaN density for this branch; `test_kl_div_two_vp_no_gauss_flag` uses two well-separated unit Gaussians in `D=1` with an identity transform, where every density is finite and positive.

### F3. The balanced sampler's remainder weights use an elementwise correction, not MATLAB's total
- Location: `pyvbmc/variational_posterior/variational_posterior.py:643`; MATLAB: `vbmc_rnd.m:68-72`
- Category: formula (and a Python `sum`/`np.sum` trap)
- Proposed classification: port discrepancy
- Confidence: high
- History: MATLAB's block dates from `910c264`/`4a68401` (2019-07-21, "patch for high K", "fixed indexing"), unchanged since. The Python line entered with `42a0deef` (2021-03-14, "feat: initial version of VariationalPosterior.sample()") in its present form; it never matched MATLAB.
- What it does: MATLAB computes `delta_extra = N_extra - sum(w_extra)`, one scalar, and adds `w*delta_extra` so that the remainder weights sum to exactly `N_extra`. The Python line `w_extra += self.w * (repeats_extra - sum(w_extra))` uses the builtin `sum` on the `(1,K)` array `w_extra`, which iterates over rows and returns the single row — a length-`K` vector, not a scalar (the line above, deliberately, uses `np.sum`). The correction becomes `w_k * (N_extra − w_extra_k)` per component. After the normalization on the next line the remainder is drawn from a different categorical distribution.
- Consequence: only the `N − Σ floor(w·N)` "remainder" draws of a balanced sample are affected; the floor part is identical. With `w = [1/3, 2/3]`, `N = 10`: MATLAB draws the one extra sample with probabilities `[0.333, 0.667]`, PyVBMC with `[0.385, 0.615]`. On the real K=50 weights of `vp-test.npz`: `N = 100` leaves 25 remainder draws (25% of the sample) whose distribution differs from MATLAB's by a total variation of 0.20 (max per-component gap 0.045); `N = 1000` leaves 28 draws (3%). The affected call sites are `active_sample.py:1065` and `:968` (variational and heavy-tailed search points, `N` in the hundreds — the largest exposure), and `moments`/`kl_div`/`mtv` at `N = 1e5`–`1e6`, where ~28 of a million samples are drawn with slightly wrong component proportions, i.e. numerically irrelevant there.
- Suggested reproduction: `c2_balanced_remainder.py` and `c9_balance_scale.py`; ran both, numbers above.
- Test adequacy: no. `test_sample_balance_extra` uses `K = 2` with equal weights, where the elementwise and total corrections are symmetric and give the same normalized probabilities; `test_sample_balance_no_extra` has no remainder at all.

### F4. The positivity check of `set_parameters(raw_flag=False)` inspects the wrong slice
- Location: `pyvbmc/variational_posterior/variational_posterior.py:1078-1090` (the test at `:1086`); MATLAB: no counterpart (`misc/rescale_params.m` has no such check)
- Category: indexing/shape
- Proposed classification: suspected defect (Python-only)
- Confidence: high
- History: entered with `8a4644bf`/`b30a771d` (2021-03-17) and unchanged in substance since; MATLAB never had the check, so there is nothing it once matched.
- What it does: `check_idx` accumulates to `−(2K+D)`, the negative offset of the constrained tail, and the intended slice is `theta[check_idx:]`. The code writes `theta[-check_idx:]`, i.e. `theta[2K+D:]`, which is the *last* `D*K` entries. Two failure modes, both confirmed in `c3`: with `D*K < 2K+D` (e.g. `D=3, K=2`, `D=1, K=4`) a negative `sigma` slips through unchecked; with `D*K > 2K+D` (e.g. `D=5, K=5`) the slice reaches into `mu`, and a perfectly legitimate posterior with negative component means is rejected with "sigma, lambda and weights must be positive".
- Consequence: `raw_flag=False` is not used anywhere inside the package (`variational_optimization.py:335` and `:1194` use the default `raw_flag=True`), so this is a user-facing API defect: either a silently accepted negative scale, or a spurious `ValueError` for any `D*K > 2K+D` problem with a negative mean — which is most problems of `D ≥ 3, K ≥ 3`.
- Suggested reproduction: `c3_set_parameters_check.py`; ran it, output as described.
- Test adequacy: no. `test_set_parameters_not_raw_negative_error` uses `D=3, K=2` and makes *every* entry negative, so the wrong slice still fires and the test passes for the wrong reason — a textbook case of a test mirroring the implementation.

### F5. `vp.pdf` on an integer array silently truncates the transformed coordinates
- Location: `pyvbmc/variational_posterior/variational_posterior.py:787` (`x = x.copy()`) with `:801` (`x[mask] = self.parameter_transformer(x[mask])`); MATLAB: no counterpart (`vbmc_pdf.m:38` assigns to a fresh `X`, and MATLAB arrays are double)
- Category: indexing/shape
- Proposed classification: suspected defect (Python-only)
- Confidence: high
- History: the `x.copy()` / in-place masked assignment pattern dates from the original `pdf` port; MATLAB has no dtype to preserve.
- What it does: `x.copy()` keeps the caller's dtype. When the input is an integer array and `orig_flag=True`, the float values written back by the transformer are cast to the integer dtype, so the density is evaluated at a truncated point. With hard bounds `[0,10]^2` and a point `[3,4]`: `vp.pdf(np.array([[3,4]]))` returns `0.0710647` and `vp.pdf(np.array([[3.,4.]]))` returns `0.00335623`, a factor of 21. `orig_flag=False` is unaffected (no assignment happens).
- Consequence: a wrong, silently returned density for any user who passes integer-valued parameters — plausible with `integer_vars` problems or plain Python lists of ints promoted by `np.atleast_2d`. No internal caller passes integers.
- Suggested reproduction: the two-line call above; ran it, numbers as given.
- Test adequacy: no. Every `pdf` test builds its input from float literals or `np.ones`.

### F6. `mode` uses a different search strategy from `vbmc_mode.m`, and is not on the sheet
- Location: `pyvbmc/variational_posterior/variational_posterior.py:1242-1290`; MATLAB: `vbmc_mode.m:21-47`
- Category: control flow (and random draws)
- Proposed classification: possibly intentional (an unrecorded deliberate change)
- Confidence: high that it differs, medium that it is intended
- History: MATLAB's file last changed on 2021-03-27 (`48c82e0`, "fixed starting points within bounds in vbmc_mode.m") — the `x0 = min(max(x0,LB),UB)` line, which PyVBMC has in a variant. The first Python port (`4b761589`, 2021-03-25) did follow MATLAB: `x0_mat = self.mu.T`, `n_max = 20`, `if n_max < self.K` keep the best. Commit `ba8116fa` (2022-11-03, "Some changes for better docs and fix vp.mode() function (#115)", co-authored by the PI) replaced it with the present scheme.
- What it does: MATLAB runs one optimization from **each** component mean, capped at `nmax = 20` best means, and returns the best result. PyVBMC runs `n_opts = ceil(sqrt(K))` optimizations; each one draws `1e5` fresh samples from the posterior and starts from the best of them, with the component means added to the candidate set of the first run only. For `K = 50` that is 8 starts against MATLAB's 20; for `K ≤ 4` it is 2 against `K`. PyVBMC also consumes `n_opts × 1e5` draws from `self.rng` (MATLAB draws none in `vbmc_mode`), and it caches the result in `self._mode` unconditionally where MATLAB stores `vp.mode` only when a second output is requested. Two smaller deviations: MATLAB clamps `x0` to the *shrunken* bounds `lb+sqrt(eps)`/`ub−sqrt(eps)` that it then hands to `fmincon`, while PyVBMC (`:1278-1281`) clamps to the raw `lb_orig`/`ub_orig` and passes the shrunken pair as `bounds`, so an `x0` exactly on a bound is infeasible for the optimizer; and `fmincon`/`fminunc` are replaced by `scipy.optimize.minimize` (L-BFGS-B with bounds, BFGS without).
- Consequence: a different mode estimate on a multimodal mixture. PyVBMC's first start is the highest-density point of `1e5` samples ∪ the means, which is a reasonable global heuristic, but only one basin is polished, where MATLAB polishes up to 20. `mode()` is not called anywhere in the package (user-facing only), and the stored MATLAB mode of the `K = 50` fixture is still reproduced to `1e-4`.
- Suggested reproduction: none run; `test_mode_no_orig_flag`/`test_mode_orig_flag` already exercise both paths against the MATLAB answer. The check that would separate the two is a mixture with two well-separated near-equal peaks and `K` large, comparing `mode()` with a from-every-mean search.
- Test adequacy: the two `test_mode_*` tests are genuine MATLAB references, but for one unimodal-ish `D=2, K=50` mixture; they would not catch a missed basin.

### F7. `get_hpd` rounds half-integers to even; MATLAB's `round` goes away from zero
- Location: `pyvbmc/stats/get_hpd.py:36`; MATLAB: `misc/gethpd_vbmc.m:10`
- Category: defaults (numeric convention)
- Proposed classification: port discrepancy
- Confidence: high on the mechanism, low on the consequence
- History: MATLAB unchanged since 2019 (`bad3fac`). Python has used the builtin `round` since the function was ported.
- What it does: Python's `round(x)` is banker's rounding, MATLAB's `round(x)` rounds halves away from zero. They differ exactly when `hpd_frac * N` is a half-integer whose floor is even: `0.1 * 5 = 0.5` → 0 against 1, `0.5 * 5 = 2.5` → 2 against 3, `0.1 * 25 = 2.5` → 2 against 3 (`c7`).
- Consequence: one training point more or less in the high-posterior-density subset, in a narrow set of cases. The main use, `hpd_frac = 0.8` (`gaussian_process_train.py:327`, `variational_optimization.py:798`), can never produce a half-integer (`0.8N = 4N/5` is a half-integer for no integer `N`). It is reachable from `active_sample.py:1004`, whose `hpd_fracs` vector always contains the exact endpoint `hpd_frac/8 = 0.1`: at `N = 5, 25, 45, …` live training points the subsets differ by one point, and at `N = 5` PyVBMC returns an empty subset where MATLAB returns one point (PyVBMC then takes its `X_hpd.size == 0` branch, `active_sample.py:1006`, which MATLAB does not have). The effect is one search-point Gaussian fitted slightly differently on a handful of early iterations.
- Suggested reproduction: `c7_misc.py`; ran it, table as described.
- Test adequacy: no. `test_get_hpd` uses `N = 100` with fractions 0.8, 0.5, 0.01 — no half-integer product.

### F8. `kde_1d` bins on grid centres where `kde1d.m` bins on grid edges, and falls back differently
- Location: `pyvbmc/stats/kde_1d.py:22-31` (`_linear_binning`, especially `:26`) and `:247-250`; MATLAB: `shared/kde1d.m:46-48` (`histc(data,xmesh)`) and `:124-140` (`root`)
- Category: formula
- Proposed classification: possibly intentional (an unrecorded improvement)
- Confidence: high that it differs, medium that it is intended
- History: MATLAB's `kde1d.m` is Botev's original, added `abbf946` (2020-12-05) and unchanged. The Python is an independent implementation (its docstring credits Botev, D. B. Smith and KDEpy), so it never was a transcription.
- What it does: `histc(data,xmesh)` treats the grid points as left bin **edges**, so a sample is credited to the grid point at or below it; the resulting density is biased half a bin to the left. `_linear_binning` credits the **nearest** grid point (`floor((x − (x0 − dx/2))/dx)`). Against a transcription of `kde1d.m` on 3000 samples with `n = 1024` (`c5`), the PyVBMC density is the MATLAB one shifted right by `dx/2`: the residual against the half-bin-shifted MATLAB curve is 5.3e-4 against 1.5e-3 unshifted and 3.1e-3 shifted the other way. The bandwidth differs by 7e-4 relative (the binning feeds the fixed-point solve). Two further differences: on a failed root solve MATLAB falls back to `fminbnd` on `|f|` while PyVBMC falls back to Scott's rule (`:247-250`), and negative round-off values are set to `0` rather than `eps`. The DCT scaling is consistent despite `scipy.fftpack.dct` returning `2·Σx` where `dct1d.m` returns `Σx` for the zeroth coefficient: both integrate to the same value (1.00098 in the check).
- Consequence: the only consumer in the package is `mtv` with `nkde = 2^13`, where a half-bin shift is of order `range/16384`; the induced error on a marginal total variation is ~3e-4 for a Gaussian marginal, i.e. far below the 1e-2 tolerances the MTV is used at. The centred binning is the more accurate of the two.
- Suggested reproduction: `c5_kde1d_transcription.py`; ran it, numbers above.
- Test adequacy: no MATLAB reference exists for `kde_1d`. `testing/stats/test_kde1d.py` checks shapes and a loose (`mtv < 0.03`) agreement with the analytic density, which a half-bin shift cannot trip.

### F9. `fess` takes the `(X, I)` pair of `vp.sample` for the sample array
- Location: `pyvbmc/vbmc/active_importance_sampling.py:472`; MATLAB: `misc/fess_vbmc.m:9` (`X = vbmc_rnd(vp,N,0)`)
- Category: cross-module
- Proposed classification: suspected defect (dormant)
- Confidence: high
- History: not traced (outside my slice's files).
- What it does: with a scalar third argument, `X = vp.sample(N, orig_flag=False)` binds the tuple, and `gp.predict(X)` / `X.shape[0]` then fail. MATLAB's assignment takes the first output only.
- Consequence: none today — the only live call, `active_importance_sampling.py:88`, passes an array. The documented `X : … or an integer number of samples` interface is unusable.
- Suggested reproduction: `fess(vp, gp, 100)`; not run (the untracked `verification/scripts/wave4_P4_2_fess_scalar.py` indicates P4 already ran exactly this).
- Test adequacy: `test_fess` (`testing/vbmc/test_active_importance_sampling.py:212-213`) passes arrays only.

### F10. `vp.pdf` masks points outside the original bounds; MATLAB does not, and the sheet does not record it
- Location: `pyvbmc/variational_posterior/variational_posterior.py:791-795`, `:913`, `:915`; MATLAB: `vbmc_pdf.m:36-39` (no counterpart to the mask)
- Category: control flow
- Proposed classification: possibly intentional (an unrecorded improvement; sheet gap)
- Confidence: high
- History: the mask is in the Python from the first port of `pdf`. MATLAB never had it.
- What it does: PyVBMC returns `0` (or `-inf`) for any row not strictly inside `lb_orig`/`ub_orig`, and skips the transform and the Jacobian on those rows. MATLAB warps every row: for a bounded variable `warpvars_vbmc.m:104-110` computes `log(z/(1-z))`, which is complex in MATLAB for an out-of-bounds point and `±Inf` on the bound, and the Jacobian correction then gives `NaN`. So MATLAB returns garbage where PyVBMC returns the mathematically right answer. The behavior is pinned by `test_pdf_outside_bounds`.
- Consequence: PyVBMC is correct and MATLAB is not; worth a sheet entry beside the existing P7 entry on the original-space gradient refusal, which records the neighbouring decision. As a side effect, a `NaN` coordinate also maps to density `0` in PyVBMC (`NaN > -inf` is false) instead of propagating.
- Suggested reproduction: none needed; `c8` and `test_pdf_outside_bounds` cover it.
- Test adequacy: covered (`test_pdf_outside_bounds`), but as current behavior, not against MATLAB.

### F11. The sheet's `qtrapz` entry misdescribes the difference: the two rules are identical
- Location: `known_differences.md` §P7, "qtrapz.m is replaced by SciPy's trapezoid rule"; code: `variational_posterior.py:1379`, `:1383`, `:1406`; MATLAB: `shared/qtrapz.m:34`
- Category: cross-module (sheet accuracy)
- Proposed classification: port discrepancy in the *record*, not in the code
- Confidence: high
- History: n/a.
- What it says and what is true: the entry states that "the quadrature helper and its endpoint handling differ; MATLAB's `qtrapz` omits the endpoint correction that `trapezoid` applies". `qtrapz(y)` is `sum(y) − 0.5*(y(1)+y(end))`, which is exactly `scipy.integrate.trapezoid(y)` with unit spacing (`Σ (y_i+y_{i+1})/2`). On 1000 random values the two agree to the last bit (difference 0.0, check `c8`). `mtv` calls `trapezoid(y)` and multiplies by the spacing afterwards, exactly as `vbmc_mtv.m:68`, `:71`, `:77` do — so the substitution is bit-for-bit equivalent up to summation order, not a difference in endpoint handling.
- Consequence: the entry's disposition (do not report the missing `qtrapz` module) stands; its "What differs" clause should say that the two formulas agree, so a future reviewer does not chase a quadrature difference that does not exist.
- Suggested reproduction: `c8_jacobian_and_trapz.py`; ran it.
- Test adequacy: n/a.

### Minor observations (below the bar for a finding)
- `get_parameters` rescales `lambd`, `sigma` and `w` **in place** (`:1012-1019`), as `misc/get_vptheta.m:17` does through `rescale_params`; but MATLAB structs are by value, and 4 of the 6 call sites (`vpoptimize_vbmc.m:32`, `vpsieve_vbmc.m:61`, `activesample_vbmc.m:522-523`) discard the rescaled struct, while 2 keep it. In PyVBMC the rescaling always sticks to the caller's object (`c7`: `lambd = [2,4] → [0.632, 1.265]`, `sigma` scaled by the same factor). The operation is idempotent and the rescaled posterior is the same distribution, so I found no numerical consequence; noted because it is a state-mutation asymmetry across the port.
- `VariationalPosterior(D, K, x0)` rejects a `(D,1)` column `x0` with a broadcast `ValueError` (`c8`), because `x0.reshape(-1, 1)` at `:136` discards its result — the dead line shows the intent was to accept it. MATLAB's `misc/setupvars_vbmc.m:82-83` always gets an `(n0,D)` matrix, so there is no counterpart; the general `(n0,D)` branch matches MATLAB's `repmat`/`(1:K,:)` tiling exactly.
- `mode()` clamps `x0` to the raw bounds while handing `lb+sqrt(eps)`/`ub−sqrt(eps)` to the optimizer (`:1268-1282`); MATLAB clamps to the same shrunken pair it passes to `fmincon` (`vbmc_mode.m:39-41`). An `x0` landing exactly on a bound would be infeasible for SciPy.
- `get_bounds` matches `misc/vpbounds.m` line for line except for the accumulation, which the sheet records and which the code does implement as the entry describes (`test_soft_bounds_follow_the_training_inputs` pins it).
- `moments`, `kl_div_mvn` and `mtv` match their MATLAB counterparts throughout: `np.cov` default `ddof=1` = MATLAB `cov`'s `N-1`; `interp1d(kind="cubic")` = `interp1(...,'spline',0)` (both not-a-knot with 0 outside); the `nkde = 2^13`, `N = 1e5`, `range/10` padding and three-subinterval quadrature of `vbmc_mtv.m` are reproduced exactly. `kl_div_mvn` reproduces `mvnkl.m` with an added `det == 0 → inf` guard MATLAB lacks (MATLAB would return `-Inf`/`NaN`).

---

## 4. Test adequacy notes

- `test_set_parameters_not_raw_negative_error` (`test_variational_posterior.py:388`) is the clearest case of a test written against the implementation: it makes every entry of `theta` negative, so the wrong slice of F4 still raises, and the test has passed since 2021 over a check that inspects the wrong entries.
- `test_kl_div_two_vp_identical_samples_gauss_flag` and `test_kl_div_two_vp_samples_gauss_flag` are the only tests of the `samples=` branch, and both are constructed so that the grand mean equals the per-coordinate mean (all components at the same `x0`, or `D = 1`). They cannot see F1 — a defect that turns a KL of 9 into 3e-4.
- `test_sample_balance_no_extra`/`test_sample_balance_extra` use `K = 2` with equal weights, where F3's elementwise correction is symmetric and normalizes back to MATLAB's answer. They check only that the counts are `N/2` or `N/2 ± 1`, which any remainder rule satisfies.
- `pyvbmc/testing/stats/test_kde1d.py` has no MATLAB reference of any kind: shapes, error paths, and `mtv < 0.03` against the analytic density. A half-bin shift, a different bandwidth fallback or a different DCT normalization would all pass. `test_get_hpd` likewise pins no rounding convention.
- The `pdf` tests pin values to 12 digits, but they were produced with the implementation, not with MATLAB (`FIXTURES.md` lists only `vp-test.npz` and the moments file as MATLAB-derived). The two genuinely MATLAB-derived gates in this slice are the mode of the `K=50` fixture and the transformed-space moments of the `D=6, K=3` mixture; `mtv`, `kl_div`, `sample` and `get_bounds` (beyond `bnd_lb.txt`/`bnd_ub.txt`) have no MATLAB reference.
- On the positive side, the entropy fixtures (`entropy-test.npz`, value and gradient, both estimators) are real MATLAB references, and my independent transcription agrees with the Python to machine precision — including the `jacobian_flag=False` branch, which the fixture does not cover but which the finite-difference tests do.

---

## Defects on the MATLAB side

1. `vbmc_pdf.m:113-123`: in original coordinates with `logflag = 0`, the gradient is divided by nothing — the transformed-space `dy` is returned uncorrected beside a Jacobian-corrected `y`. The log case errors out instead. (Already the basis of a sheet entry; recorded here as the MATLAB-side defect it is.)
2. `vbmc_kldiv.m:44-52`: the analytical (`Ns == 0`) branch is unreachable. `origflag` is hard-coded to `1` at `:34`, and the branch's first statement is `if origflag; error(...); end`, so `vbmc_moments(vp1,0)` at `:50-51` can never run.
3. `vbmc_rnd.m:84`, `:88`: `sigma(I(1:N))'` — `I` has exactly `N` elements at that point (it was truncated by `randperm(numel(I),N)` at `:77`), so the index is redundant; harmless.
4. `shared/kde1d.m:74`: `bandwidth_cdf` is computed and never returned or used.
5. `vbmc_mode.m:3-10`: the help text documents the function as `VBMC_PDF(VP,ORIGFLAG)` while the signature is `vbmc_mode(vp,nmax,origflag)`, so a user following the documentation passes `origflag` as `nmax`.
6. `shared/kde1d.m:46-48`: binning with `histc` on the grid points as left edges biases the estimate half a bin low (see F8). PyVBMC's centred binning is the correction.
