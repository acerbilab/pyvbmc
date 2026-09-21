# P9 — priors, MATLAB-comparison track

Raw reviewer report, wave 5 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, fresh
context, read-only on the repositories, reading the code at `f873556`; it
compared the Python code with MATLAB VBMC at `396d649`. Its brief carried
a first question, which the report answers before its findings. Its check scripts
ran in the session scratchpad and are kept only on the machine that ran them
(`dev/scripts/runs/LOCAL.md`). The text below is the agent's final message,
unedited.

---

# P9 — priors (MATLAB comparison track)

## 1. Coverage

**Read completely (Python):** `pyvbmc/priors/prior.py`, `uniform_box.py`, `trapezoidal.py`, `smooth_box.py`, `spline_trapezoidal.py`, `product.py`, `scipy.py`, `user_function.py`, `convert_to_prior.py`, `tile_inputs.py`, `__init__.py`. All of `pyvbmc/testing/priors/` (10 modules). `pyvbmc/rng.py`.

**Read completely (MATLAB, `../vbmc` @ 396d649):** all twelve `shared/m{unifbox,trapez,smoothbox,splinetrapez}{pdf,logpdf,rnd}.m`, `lpostfun.m`, `test/test_pdfs_vbmc.m`, and the prior section of `vbmc_examples.m` (lines 440–720).

**Read for the cross-module edges:** `pyvbmc/vbmc/vbmc.py` `_init_log_joint` (`:3471-3551`), the `prior`/`log_prior`/`sample_prior` docstrings (`:142-159`), the precomputed-evaluation prior path (`:777-797`), `_rebuild_log_joint`/`__str__` (`:3601-3660`); `pyvbmc/function_logger/function_logger.py: __call__` (`:232-356`) and MATLAB's `misc/funlogger_vbmc.m` value checks; `docsrc/source/api/classes/priors.rst`; the P9 and "Settled non-differences" sections of the known-differences sheet, and the P1b entry on randomness.

**Skimmed:** `pyvbmc/vbmc/README.md` and `CHANGELOG.md` (searched for prior-related entries); git history of `pyvbmc/priors/` (three commits after the original port: `2da98b5f` 2023-02-08, `d02c517e` 2026-09-02 rng threading, `3e67d9e1` 2026-09-07 SciPy type recognition) and of `shared/` (`c387612`, `f3c05d6`, both 2022-10-26).

**Not reached:** nothing in the slice. There is no MATLAB on this machine, so every MATLAB result below comes from a line-by-line Python transcription of the `.m` files (written to the scratchpad as `matlab_priors.py`), not from MATLAB itself.

**Checks run** (all in `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\e2e318de-17da-4589-80c5-c950bb45f270\scratchpad\wave5_P9_comparison`, `.venv` interpreter, BLAS single-threaded; `pyvbmc.__file__ = C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\__init__.py`):

1. `check1_density.py` — the four PyVBMC classes against the MATLAB transcriptions on the same arguments: five random 1-D parameter sets over a 509-point grid that includes the exact pivots and their two neighbouring floats, a random `D=3` set on 2004 points, and the exact arguments of `test_pdfs_vbmc.m`. **Max difference 7.1e-15 in the log-density, no `-inf` mismatch, no NaN on either side, in every case.**
2. `check2_norm_and_samplers.py` — 1-D quadrature of each density (Python and MATLAB transcription), 2-D `dblquad` of each product, and a Kolmogorov–Smirnov test of each sampler (Python and MATLAB) against the closed-form CDF, n = 200000.
3. `check3_edges_dtypes.py` — degenerate pivots, input dtypes, warnings, `np.seterr` restoration, `Prior.log_pdf` shape handling, `Product`/`SciPy`/`UserFunction` contracts, `tile_inputs`.
4. `check4_crossmodule.py` — `SciPy.support()` on shifted/scaled frozen distributions, `Product` with a `UserFunction` marginal, and a live `VBMC` object's `log_joint` (no `optimize()` run).
5. `check5_multivariate_ks.py` plus a follow-up — per-dimension KS in `D=3` with heterogeneous parameters, extreme regimes (`b-a = eps`, a plateau 10^5 times the tail scale, a 0.002-wide plateau), `n = 0`, and a 20-million-draw binomial check of the smooth-box mixture weights.

No test suite was run, no `optimize()` was run, nothing was written inside either repository.

## 2. First question

**Does each sampler draw from exactly the density its density function defines, on both sides? Yes, for all four families on both sides, with one exception on the MATLAB side (`msmoothboxrnd` for `D > 1`, defect M1 below).**

Family by family (PyVBMC first; MATLAB is identical unless stated):

- **Uniform box.** Density `C = 1/∏(b_i-a_i)` inside the box, 0 outside; `1/(b-a)` integrates to 1 per dimension (quadrature: 1.000000000000; 2-D product 0.99999999). Edges are **inclusive** on both sides (`x < a | x > b` marks the zero region), so `p(a) = p(b) = C`, never NaN. `a < b` is required (PyVBMC at construction, MATLAB at every call). The sampler is the exact inverse CDF: PyVBMC `rng.uniform(a, b, (n,D))`, MATLAB `a + rand(n,D).*(b-a)` — the same affine map of a uniform. KS vs the exact CDF: D = 0.00167, p = 0.63 (Python), D = 0.00214, p = 0.32 (MATLAB).
- **Trapezoid.** `C = 1/(½(b-a+v-u))`; the left ramp is `C(x-a)/(u-a)` on `[a,u)`, the plateau `C` on `[u,v)`, the right ramp `C(b-x)/(b-v)` on `[v,b)`. Integral `C(½(u-a) + (v-u) + ½(b-v)) = 1` — quadrature 1.000000000000, 2-D 1.0000000000. `p(a) = p(b) = 0` (log `-inf`, reached through `log(0)`, no NaN); `p(u) = p(v) = C` (the interval tests are `>=` on the left and `<` on the right at every pivot, on both sides). MATLAB's `lnf` carries an extra `+log(u-a)` that every branch cancels; PyVBMC cancels it algebraically instead. The sampler is rejection inside `[a,b] × [0, f(½(u+v))]`, and `½(u+v)` is on the plateau, i.e. the mode, so the envelope dominates the density everywhere: the accepted points are exactly `f`-distributed. KS: D = 0.00186, p = 0.50 (Python), D = 0.00198, p = 0.41 (MATLAB); a 0.002-wide plateau (rejection rate ≈ ½) gives D = 0.0028, p = 0.82 against the symmetric triangular CDF.
- **Spline trapezoid.** Same `C = 1/(½(b-a+v-u))`, with the ramps replaced by `C(3z²-2z³)`, `z` the normalized distance into the shoulder; `∫₀¹(3z²-2z³)dz = ½`, so the normalization is the trapezoid's and is correct — quadrature 1.000000000000, 2-D 0.9999999999. Edge values and pivot tests are the trapezoid's. Sampler identical in structure; the shoulder maximum is again on the plateau. KS: D = 0.00103, p = 0.98 (Python), D = 0.00247, p = 0.17 (MATLAB).
- **Smooth box.** `C = 1/((b-a) + √(2π)σ)` on `[a,b]`, `C·exp(-½((x-a)/σ)²)` below `a` and the mirror above `b`. Each tail carries mass `C σ√(2π)/2`, the plateau `C(b-a)`, summing to 1 — quadrature 1.000000000000, 2-D 1.0000000001. The density is continuous at both pivots (the tail equals `C` there), positive everywhere, never `-inf` or NaN; PyVBMC and MATLAB both put the pivots in the plateau branch (`a <= x <= b`). `σ > 0` and `a < b` are checked on both sides. The sampler is exact composition, not rejection: it draws `u ~ U(0, nf)` with `nf = 1 + (b-a)/(√(2π)σ)`, takes the left half-normal for `u < 0.5`, the right for `0.5 <= u < 1` and the plateau for `u >= 1`. Those weights are `0.5/nf = Cσ√(2π)/2` per tail and `(nf-1)/nf = C(b-a)` for the plateau — the exact masses. Verified: KS D = 0.00135, p = 0.86; with `b-a = eps` the draws are `N(0,σ)` (KS p = 0.20); with a plateau 10^5 times the tail scale the tail fraction over 20 million draws is 497 against an expected 501.3 (binomial p = 0.88).
- **Degenerate arguments.** PyVBMC refuses `u == a`, `u == v`, `v == b`, `a == b` at construction. MATLAB's `munifboxlogpdf`/`msmoothboxlogpdf` refuse `a >= b`; `mtrapezlogpdf`/`msplinetrapezlogpdf` check nothing and compute `u == v` (the triangular limit) and `v == b` correctly and normalized — only the trapezoid's `u == a` gives NaN there (finding F4, defect M2).

**Does each Python class reproduce its MATLAB functions on the same arguments? Yes, to floating point.** Across all the parameter sets and grids of check 1, the largest difference in the log-density is 7.1e-15 and the `-inf` sets coincide exactly. The only places where the two implementations are not bit-identical are the associativity of the normalization constant (MATLAB folds `log(u-a)` into `lnf`; MATLAB writes `log(1/√(2π)/σ)` where Python writes `-log(√(2π)·σ)`). Both samplers of every family draw the same number of variates in the same order, from the same distributions; the streams differ because PyVBMC threads a `numpy.random.Generator` (the sheet's P1b entry). The argument order matches throughout: `UniformBox(a,b)` ↔ `munifbox*(x,a,b)`, `Trapezoidal(a,u,v,b)` ↔ `mtrapez*(x,a,u,v,b)`, `SplineTrapezoidal(a,u,v,b)` ↔ `msplinetrapez*(x,a,b,c,d)` with `(a,b,c,d) ≡ (a,u,v,b)`, `SmoothBox(a,b,scale)` ↔ `msmoothbox*(x,a,b,sigma)` (the third parameter is named `scale` in Python and `sigma` in MATLAB; the position is the same, and `scale` was already the name at v1.0.4).

**Two things that have no MATLAB counterpart and were judged on their own contract:** `Product`, `SciPy`, `UserFunction`, `convert_to_prior`, `tile_inputs` and the `Prior` base class (findings F1, F3, F6). The combination of prior and likelihood inside `VBMC._init_log_joint` matches MATLAB's `lpostfun.m` exactly, noisy branch included (the prior is noiseless, the SD comes from the likelihood alone), and the prior is evaluated in the caller's original coordinates on both sides — verified live: `log_joint((0,0))` returns the likelihood plus `-2 log 8` for a `UniformBox([-4,-4],[4,4])`. `sample_prior`/`prior.sample` is never read by a run; that matches MATLAB, where `vbmc.m` never calls the `m*rnd` functions either (only `test_pdfs_vbmc.m` does), and the PyVBMC docstring says "Currently unused".

## 3. Findings

### F1. `SciPy` priors report the standardized support of a shifted or scaled frozen distribution
- Location: `C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\priors\scipy.py:69-70` (and `:122-135`, `C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\priors\product.py:66-67`); MATLAB: no counterpart
- Category: cross-module
- Proposed classification: suspected defect in the Python-only code
- Confidence: high
- History: entered with the original priors commit `2da98b5f` (2023-02-08), in the `rv_continuous_frozen` branch, as `self.a = np.atleast_1d(distribution.a)`. `3e67d9e1` (2026-09-07) rewrote the type recognition around these two lines but left them untouched. There is no MATLAB counterpart, so the Python never matched anything.
- What the code does: a frozen `scipy.stats` univariate carries `.a`/`.b` as the support of the *standardized* distribution, before `loc` and `scale` are applied. `SciPy(uniform(loc=2, scale=3)).support()` returns `(0.0, 1.0)` while its own `sample()` returns values in `[2, 5]`; `SciPy(beta(2,1.5,loc=1,scale=4)).support()` returns `(0.0, 1.0)` for a distribution on `[1, 5]`. `Prior.support`'s documented contract (`prior.py:62-73`) is "a box which bounds the support of the distribution", so the value should be `(loc, loc+scale·b)`, i.e. `distribution.ppf(0)` and `distribution.ppf(1)`. `Product.__init__` copies these into `Product.a`/`Product.b`, so a product built on such a marginal reports `[0,1]` for it.
- Consequence if real: `support()`, `Product.a`, `Product.b` and the `__str__`/`__repr__` of both classes are wrong whenever a user passes a `loc`/`scale`-shifted univariate SciPy distribution — a natural way to write `prior=[scipy.stats.uniform(loc=lb, scale=ub-lb), ...]`. Nothing inside a VBMC run reads `support()`, so results and the log-joint are unaffected; the damage is to reported bounds and to any caller (or future code path) that trusts the box.
- Suggested reproduction: ran it. `SciPy(uniform(loc=2, scale=3)).support()` → `(array([0.]), array([1.]))`, sample range `[2.0003, 5.0000]`; `Product([uniform(loc=2, scale=3), UniformBox(0,1)]).a, .b` → `[0,0], [1,1]`.
- Test adequacy: no. `test_priors.test_sample` does assert every sample lies inside `support()`, but only for `_generic` instances (`multivariate_normal`), and `test_scipy_prior.py`/`test_product_prior.py`/`test_convert_to_prior.py` use only unshifted `norm()`, `lognorm(1.0)`, `beta(2,1.5)`, `t(nu)`, whose `.a`/`.b` happen to be right. Adding `uniform(loc=2, scale=3)` to `test_priors.classes`'s `_generic` or to `test_sample` catches it immediately.

### F2. An integer or float32 input silently corrupts the log-density of the three `np.full_like` priors
- Location: `C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\priors\trapezoidal.py:96`, `C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\priors\smooth_box.py:79`, `C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\priors\spline_trapezoidal.py:98`; MATLAB: `shared/mtrapezlogpdf.m:49`, `shared/msmoothboxlogpdf.m:51`, `shared/msplinetrapezlogpdf.m:52`
- Category: indexing/shape (dtype)
- Proposed classification: port discrepancy
- Confidence: high
- History: entered with `2da98b5f` (2023-02-08) and unchanged since; the MATLAB lines entered with `c387612`/`f3c05d6` (2022-10-26) and never changed. The Python never matched.
- What the code does: `np.full_like(x, -np.inf)` inherits `x`'s dtype. MATLAB's `-inf(size(x))` uses only the *size* and always produces a double, so `mtrapezlogpdf` returns a double whatever `x` is. With an `int64` input the Python accumulator is integer: `Trapezoidal(0, .25, .75, 1, D=2).log_pdf(np.array([[0, 1]]))` returns `0` — a density of 1 — where both the true value and MATLAB's are `-inf` (both coordinates sit exactly on a bound); `SmoothBox(0, 1, 1, D=2).log_pdf(np.array([[0, 1]]))` returns `-2` instead of `-2.5093`. No exception and no warning. With a `float32` input the result is `float32`, against the codebase's float64 convention that the dtype canary pins. `UniformBox` is immune: it builds its output with `np.full((n,1), ...)`.
- Consequence if real: silently wrong prior densities for any caller who hands integer coordinates to `prior.pdf`/`prior.log_pdf` — a grid from `np.arange`, integer-valued parameters, a hand-written point. A VBMC run is not affected: `VBMC.__init__` widens to float64 and the transformer returns float64, so the prior only ever sees float64 there. The float32 leak is a dtype leak of the same line.
- Suggested reproduction: ran it. Table in `check3.log`: `Trapezoidal` int64 `-> int64  value [0]`, `SmoothBox` int64 `-> int64  value [-2]`, all three families float32 `-> float32`.
- Test adequacy: no. No test passes a non-float64 array to any prior; `test_priors.test_shape` uses `np.random.normal` throughout. One extra line there (assert the output dtype is float64 for a float32 and an int input) would catch it.

### F3. `Product.log_pdf` raises when one of the marginals is a `UserFunction`
- Location: `C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\priors\product.py:88` (against the `UserFunction` special case at `:111-116`); MATLAB: no counterpart
- Category: cross-module
- Proposed classification: suspected defect in the Python-only code
- Confidence: high
- History: `_log_pdf` has called `marginal.log_pdf(x[:, m], keepdims=False)` since `2da98b5f` (2023-02-08). The `isinstance(marginal, UserFunction)` special case in `sample` was added by `d02c517e` (2026-09-02) when `rng=` was threaded through the priors; the density path was not given the matching case.
- What the code does: a `UserFunction`'s `log_pdf` is the user's own callable, bound as an instance attribute (`user_function.py:42`), and takes only `x`. `Product` accepts such a marginal (it only requires `D == 1`) and `Product.sample` routes around it, but `Product.log_pdf` passes `keepdims=False` and gets `TypeError: <lambda>() got an unexpected keyword argument 'keepdims'`. Either the call should drop `keepdims` for that case and reshape the result itself, or `Product.__init__` should refuse the marginal.
- Consequence if real: `VBMC(log_likelihood, ..., prior=[my_user_function_prior, other_marginal])` constructs successfully and then fails with a `TypeError` on the first target evaluation, inside `FunctionLogger`'s exception wrapper. Narrow configuration, total failure when hit.
- Suggested reproduction: ran it. `Product([UserFunction(f, s, D=1), UniformBox(0,1)])` samples fine and `log_pdf(np.array([[0.0, 0.5]]))` raises the `TypeError` above.
- Test adequacy: no, and strikingly so — `test_product_prior.test_product_sample_user_function_marginal` builds exactly this object and calls only `sample`. One added `log_pdf` assertion in that test catches it.

### F4. PyVBMC refuses degenerate pivots that MATLAB computes correctly
- Location: `C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\priors\trapezoidal.py:72-77`, `C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\priors\spline_trapezoidal.py:74-79`; MATLAB: `shared/mtrapezlogpdf.m` and `shared/msplinetrapezlogpdf.m` (no order check at all; `munifboxlogpdf.m:44` and `msmoothboxlogpdf.m:46` check only `a < b`)
- Category: control flow (argument checks present on one side only)
- Proposed classification: possibly intentional
- Confidence: high on the facts, low on whether it should change
- History: the Python check entered with `2da98b5f` (2023-02-08) and is documented in the class docstrings. MATLAB never had one.
- What the code does: PyVBMC raises `ValueError` for any of `a >= u`, `u >= v`, `v >= b`. Two of those are legal limits of the family with well-defined, normalized densities, and MATLAB returns them correctly: `u == v` is the triangular ("tent") prior (`mtrapezlogpdf(0.5, 0, .5, .5, 1) = log 2`, the exact triangular peak on `[0,1]`), and `v == b` is a right-truncated trapezoid (`mtrapezlogpdf(0.5, 0, .25, 1, 1) = log(1/0.875)`, correctly normalized). The spline version handles `u == a` and `u == v` correctly too. Only the trapezoid's `u == a` is genuinely broken in MATLAB, where it returns NaN (defect M2); PyVBMC's refusal is the better behavior there.
- Consequence if real: a user who wants a tent prior — the natural choice when the plausible bounds coincide, and the shape the MATLAB docstring diagram draws — has to perturb the pivots. No effect on any existing run; nothing in PyVBMC constructs these priors internally.
- Suggested reproduction: ran it. All three refusals confirmed on the Python side, all three MATLAB values confirmed against the transcription.
- Test adequacy: `test_trapezoid_prior.test_trapezoidal_error_handling` and its spline twin assert the exact refusal message, so they pin the current behavior rather than a specification; they would not let the check be relaxed without being edited.

### F5. The trapezoid warns at its lower bound where the spline trapezoid does not, and the spline's guard mutates NumPy's global error state
- Location: `C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\priors\trapezoidal.py:101-120` (no guard) against `C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\priors\spline_trapezoidal.py:104-124` (`np.seterr(divide="ignore")` … `np.seterr(**old_settings)`); MATLAB: `shared/mtrapezlogpdf.m:54`, `shared/msplinetrapezlogpdf.m:60` (MATLAB's `log(0)` returns `-Inf` silently, with no warning on either)
- Category: control flow
- Proposed classification: unsure (cosmetic on one half, a real hazard on the other)
- Confidence: high on the facts, low on the importance
- History: both shapes entered with `2da98b5f` (2023-02-08) and are unchanged.
- What the code does: `Trapezoidal.log_pdf(a)` emits `RuntimeWarning: divide by zero encountered in log` while returning the correct `-inf`; `SplineTrapezoidal.log_pdf(a)` returns the same value silently. Neither MATLAB function warns, so the spline's behavior is the ported one and the trapezoid's is not. Separately, `np.seterr` is process-global state: `_log_pdf` installs `divide="ignore"` for the duration of its loop and restores it only on the normal path — there is no `try`/`finally`, so an exception raised inside the loop (a broadcast error, a `KeyboardInterrupt`) leaves `divide="ignore"` installed for the rest of the process. A local `with np.errstate(divide="ignore"):` would be both exception-safe and thread-local in intent.
- Consequence if real: a stray warning from one of the two twin classes; and a window in which another thread's `np.seterr(divide="raise")` is silently disabled. Neither affects a run's numbers.
- Suggested reproduction: ran it. The warning is emitted for `Trapezoidal` and not for `SplineTrapezoidal`; on the normal path the `divide="raise"` setting is correctly restored.
- Test adequacy: no test covers either behavior.

### F6. Documented shapes and parameter names do not match the code
- Location: `C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\priors\prior.py:17-19` and `:50-52`; `uniform_box.py:20-23`, `trapezoidal.py:33-41`, `smooth_box.py:22-27`, `spline_trapezoidal.py:34-42`; `convert_to_prior.py:28-35`. MATLAB: no counterpart
- Category: defaults/contract (documentation)
- Proposed classification: suspected defect in the Python-only documentation
- Confidence: high
- History: entered with `2da98b5f` (2023-02-08); the `Attributes` blocks were never updated after `tile_inputs(..., squeeze=True)` was adopted.
- What the code does: every class documents `a`, `u`, `v`, `b`, `scale` as "shape `(1, D)`", but `tile_inputs(..., squeeze=True)` stores them as `(D,)` — confirmed (`UniformBox(np.zeros((1,3)), np.ones((1,3))).a.shape == (3,)`). The `keepdims` parameter of `Prior.log_pdf`/`Prior.pdf` is described as returning "an array of shape `(1, D)`" or "`(D,)`", where the actual — and correctly documented, two paragraphs later — shapes are `(n, 1)` and `(n,)`. `convert_to_prior`'s docstring names its second parameter `sample_prior` twice (`:28` and `:32`); the first block describes `log_prior`. This matters a little more than usual because the rest of PyVBMC really does use `(1, D)` bounds, so a reader will assume these are the same.
- Consequence if real: documentation only.
- Suggested reproduction: read the attributes of any constructed prior; ran it in `check3`.
- Test adequacy: `test_priors.test_shape` pins the actual return shapes (which are correct); nothing checks the attribute shapes, and the `__str__`/`__repr__` test reads the attributes back from the object, so it cannot disagree with them.

### F7. The normalization comment in `SplineTrapezoidal` mistranscribes the MATLAB comment it came from
- Location: `C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\priors\spline_trapezoidal.py:101`; MATLAB: `shared/msplinetrapezlogpdf.m:54`
- Category: formula (comment only)
- Proposed classification: port discrepancy, cosmetic
- Confidence: high
- History: entered with `2da98b5f` (2023-02-08).
- What the code does: the comment reads `# norm_factor = u - v + 0.5 * (b - v + u - a)`, where MATLAB's `% nf = c - b + 0.5*(d - c + b - a)` is, in PyVBMC's names, `v - u + 0.5*(b - v + u - a)`. The sign of the plateau term is flipped. Line 102, the code, is correct (`log(0.5 * (v - u + b - a))`, which equals MATLAB's `log(0.5*(c - b + d - a))`).
- Consequence if real: none in behavior; a reader checking the normalization against the comment gets a contradiction.
- Suggested reproduction: algebra: `(v-u) + 0.5(b-v+u-a) = 0.5(v-u+b-a)`, which is what line 102 computes.
- Test adequacy: not applicable.

## 4. Test adequacy notes

- **No test compares any PyVBMC sampler with its own density.** `test_{trapezoid,spline_trapezoid,smooth_box,uniform_box}_prior.test_*_sample_rng` check only reproducibility under a fixed generator and the output shape; `test_priors.test_sample` checks only the shape and that the draws lie in the support box. The single distributional check anywhere is `test_product_prior.test_product_mixed_distribution_type`, which compares the mean and variance of five marginals — and of the PyVBMC families it touches only `UniformBox(0,1)` and a degenerate `SmoothBox(0, eps)`, i.e. the smooth box's plateau branch is never exercised and the two rejection samplers (`Trapezoidal.sample`, `SplineTrapezoidal.sample`) have no distributional test at all. MATLAB's `test/test_pdfs_vbmc.m` histograms all four samplers against their pdf at n = 10^6. This is precisely the shape of the failure this review exists for — a sampler with no test against the distribution it claims to draw from — and it is worth closing even though I found the samplers correct (KS at n = 200000 for all four, and per-dimension in `D=3`).
- `test_trapezoid_prior.test_trapezoidal_error_handling` and `test_spline_trapezoid_prior.test_spline_trapezoidal_error_handling` assert the exact text of an argument check that MATLAB does not have and that refuses two legal members of the family (F4): they mirror the implementation, not a specification.
- `test_user_function_prior.test_function_prior` asserts `log_pdf(x) ≈ log(pdf(x))` where `UserFunction.pdf` is *defined* as `exp(log_pdf(...))` — tautological. (`assert np.isclose(prior.pdf(x), np.prod(np.exp(x)))` on the next line is a real check.)
- `test_priors.test__str__and__repr__` asserts `f"lower bounds = {new_prior.a}"` is in the string, reading the value back off the object, so it cannot detect a wrong `a`. Harmless, but it is the string test and not a value test.
- `test_scipy_prior.py` compares the wrapper against the frozen distribution's own `logpdf` — implementation-mirroring by necessity for a thin wrapper — but it never touches `support()`, which is where the wrapper actually does something of its own and gets it wrong (F1).
- The genuinely specification-shaped tests of this slice are `test_priors.test_unit_integral_1d`/`test_unit_integral_2d` (normalization by `nquad` over the reported support, for all six classes) and the `*_basic_pdf` / `*_random_pdf` tests, which compute the plateau height from the closed form rather than from the implementation. Those are good, and they are why the densities are in the state they are in.

## MATLAB-side defects (no PyVBMC consequence)

- **M1. `shared/msmoothboxrnd.m:59` and `:66` index the pivots linearly instead of by column.** `r(idx,d) = a(idx) - z1;` and `r(idx,d) = b(idx) + z1;` use a logical `n×1` index into the `n×D` matrices `a` and `b`, which MATLAB resolves as column-major linear indexing and so always reads column 1. For `D > 1`, every dimension after the first places its Gaussian tails at dimension 1's pivots. (The plateau branch at `:72` uses `a(idx,d)` correctly, so the defect affects only the tails.) Demonstrated with the faithful transcription: `a = [-5,0,10]`, `b = [-4,1,12]`, `sigma = [0.2,0.3,0.4]`, 200000 draws give marginal means `[-4.50, -1.64, 5.83]` against the correct `[-4.50, 0.50, 11.00]`; the repaired `a(idx,d)` reproduces PyVBMC exactly. `test/test_pdfs_vbmc.m:101` histograms only `r(:,1)`, so it cannot see it. PyVBMC's `smooth_box.py:131-138` uses `self.a[d]`/`self.b[d]` and is correct.
- **M2. `shared/mtrapezlogpdf.m:50` produces NaN for `u == a`.** `lnf = log(0.5) + log(b-a+v-u) + log(u-a)` is `-Inf` when `u == a`, and the plateau branch then computes `log(u-a) - lnf = -Inf - (-Inf) = NaN`, for a density that is perfectly well defined (the right-hand half of a trapezoid). Confirmed: `mtrapezlogpdf(0.5, 0, 0, 0.75, 1)` → NaN. It is a consequence of folding a factor into the normalization that every branch then cancels; PyVBMC's formulation has no such cancellation, and it refuses the argument anyway.
- **M3. `shared/mtrapezrnd.m:22` and `shared/msplinetrapezrnd.m:22` test the wrong `nargin`.** Both declare five inputs (`a,u,v,b,n`) and test `if nargin < 3 || isempty(n)`. The documented four-argument call `mtrapezrnd(a,u,v,b)` therefore falls into the `else` branch and references the undefined `n`. It should be `nargin < 5`. (`munifboxrnd.m:19` uses `nargin < 3` for three inputs and `msmoothboxrnd.m:26` uses `nargin < 4` for four — both correct, so this is a slip in two files only.) No PyVBMC counterpart: `sample(n)` always takes `n`.
- **M4. Error identifiers name the wrong function.** `mtrapezlogpdf.m:40` raises `mtrapezpdf:SizeError`, `msmoothboxlogpdf.m:27,38,47` raise `msmoothboxpdf:*`, and `munifboxrnd.m:33` raises `munifboxpdf:OrderError`. Cosmetic; it only affects `catch` blocks that match on the identifier.
