# P9 — priors, internal-correctness track

Raw reviewer report, wave 5 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, fresh
context, read-only on the repositories, reading the code at `f873556`; it
did not read the MATLAB source. Its brief carried a first question,
which the report answers before its findings. Its check scripts ran in the
session scratchpad and are kept only on the machine that ran them
(`dev/scripts/runs/LOCAL.md`). The text below is the agent's final message,
unedited.

---

# P9 — priors (internal-correctness track)

## 1. Coverage

**Read completely** (every line): `pyvbmc/priors/prior.py`, `uniform_box.py`, `trapezoidal.py`, `smooth_box.py`, `spline_trapezoidal.py`, `product.py`, `scipy.py`, `user_function.py`, `convert_to_prior.py`, `tile_inputs.py`, `__init__.py`; all ten modules of `pyvbmc/testing/priors/`; `docsrc/source/api/classes/priors.rst`; `examples/pyvbmc_example_5_prior_distributions.ipynb` (all cells).

**Read the relevant parts of** (followed calls out of the slice): `pyvbmc/vbmc/vbmc.py` — the `prior`/`log_prior`/`sample_prior` constructor arguments and their docstrings (:142-158), `_init_log_joint` (:3471-3551), the PyMC-target guard (:544-603), the precomputed-evaluation prior loop (:695-805), `_rebuild_log_joint`/`__str__`/`__repr__` (:3601-3700); `pyvbmc/function_logger/function_logger.py: __call__` and `_validate_vectorized_output`; `pyvbmc/rng.py`; `pyvbmc/testing/vbmc/test_vbmc_init.py` (the prior tests, :790-1100 and the vectorized-target prior tests :1484-1620). Grepped the whole package for `prior`, `sample_prior`, `.support()`. `pyvbmc/vbmc/README.md` has no prior entry (grep for "prior" returns nothing). Read the known-differences sheet, sections "Slice P9", "Settled non-differences", and the P1b randomness entry.

**Skimmed**: `pyvbmc/pymc/_target.py` and `_plausible.py` (they use PyMC's own prior machinery, never `pyvbmc.priors`; `VBMC` refuses `prior=`/`log_prior=` with a `PyMCTarget`). **Not reached**: the MATLAB sources (other track), `pyvbmc/svbmc/`, `papers/`.

**Checks run** (scripts and logs in `…/scratchpad/wave5_P9_internal/`, `OMP/OPENBLAS/MKL_NUM_THREADS=1`, `.venv` interpreter, `pyvbmc C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\__init__.py`, numpy 2.5.2 / scipy 1.18.1 / Python 3.12.6):

- `c1_scipy_support.py` — `SciPy.support()` against `frozen.support()` for six shifted/scaled frozen distributions; the effect on `Product`; the package's own `integrate()` helper on a shifted uniform.
- `c2_dtype_nan_edges.py` — integer and NaN inputs; densities at and just outside `a`, `u`, `v`, `b`; captured warnings; finite-difference continuity of the density and its derivative at all four pivots of both trapezoids.
- `c3_dtype32.py` — float64 / float32 / int input dtype propagation for all six families, in and out of support.
- `c4_samplers.py` — numerical integration (`scipy.integrate.quad`) of each package density; hand-derived CDFs validated against those integrals (max |F_num − F_hand| ≤ 1.3e−10); Kolmogorov–Smirnov tests of every sampler against its own density's CDF, 200 000 draws each, 8 parameter settings including extreme aspect ratios; separability and normalization with parameters differing across three dimensions.
- `c5_product_order_rng.py` — per-column KS/moments and correlations for multi-D families and for a mixed-family `Product`; dimension ordering; seed reproducibility; NumPy global-state isolation; pickle/deepcopy round-trips.
- `c6_degenerate.py` — NaN, infinite and degenerate constructor arguments for every family; `tile_inputs` with a disagreeing `size`; `convert_to_prior` corner cases.
- `c7_vbmc_integration.py` — a constructed `VBMC` (no `optimize()`): which coordinate space the prior is evaluated in, the log-joint against a hand-computed value plus `log|J|`, and what happens when the prior's support is narrower or wider than the hard bounds.
- Ad-hoc one-liners: base-class shape rules, `Product` with a `UserFunction` marginal, `SciPy` with array-valued frozen parameters, `n=0` draws, `UserFunction(D=None).support()`, integer out-of-support wraparound for D = 1…4.

No test suite was run, no `optimize()` was run, nothing was written into either repository.

## 2. First question: does `sample` draw from the density that `pdf`/`log_pdf` define, and do `pdf` and `log_pdf` agree?

`pdf` is `np.exp(log_pdf)` in the base class (and in `UserFunction`), so `pdf` and `log_pdf` agree identically for every family; there is no independently coded `pdf`. For valid arguments, **every sampler draws from exactly its own density** — all KS tests pass comfortably (p between 0.12 and 0.97 at n = 200 000, including strongly skewed and extreme-aspect-ratio parameter settings, and per-column for 3-D priors with different parameters per dimension). Family by family:

- **`UniformBox`** — density `1/∏(b−a)`, re-derived and confirmed by integration (1.000000000000). Support is closed: the density is `h` at both `a` and `b`, zero strictly outside, never NaN, no warning. `sample` is `rng.uniform(a, b)` broadcast over `(n, D)`, so it matches (KS p = 0.55; per-column p = 0.47–0.64 with heterogeneous bounds). Two defects: a NaN coordinate is reported as *inside* the box with a finite density (F3), and NaN or infinite bounds are accepted at construction (F4).
- **`Trapezoidal`** — normalizer re-derived by hand: total area is `h·[½(u−a) + (v−u) + ½(b−v)] = h·½(b−a+v−u)`, so `h = 2/(b−a+v−u)`; the code's `log_norm_factor = log(0.5) + log(b−a+v−u)` is exactly `−log h`. Ramps are linear, the density is continuous at `a`, `u`, `v`, `b` (checked numerically), zero at both edges (`x = a` gives `−inf` correctly, but with a divide-by-zero RuntimeWarning — F8), and zero outside. `sample` is rejection sampling from `U(a,b)` against `y_max = pdf(½(u+v))`, which is the true maximum `h`; acceptance is never worse than ½. KS p = 0.53 (balanced) and 0.23 (`a=0, u=0.01, v=0.02, b=10`). Defects: integer/float32 inputs (F2), degenerate arguments (F4).
- **`SplineTrapezoidal`** — shoulders are the smoothstep `3z²−2z³`; `∫₀¹(3z²−2z³)dz = ½`, so the total mass is `h·[½(u−a)+(v−u)+½(b−v)] = h·½(v−u+b−a)`, matching `log_norm_factor = log(0.5·(v−u+b−a))`. Numerical integration gives 1.000000000000. The docstring's claim that the derivative is zero (hence continuous) at all four pivots holds: finite differences give f′ ≈ ±6.4e−5 at the pivots against a plateau height of 1.33, i.e. zero to the order of the step. `sample` is the same rejection scheme with the correct bounding height; KS p = 0.43 and 0.53. Same defects as the trapezoid (F2, F4); the in-code comment for the normalizer has a sign typo (F7).
- **`SmoothBox`** — density `h` on `[a,b]` with Gaussian tails of sd `scale` anchored at `a` and `b`; mass `h·[(b−a) + √(2π)·scale] = 1`, and `log_norm_factor = −log(√(2π)σ) − log1p((b−a)/(√(2π)σ)) = −log(√(2π)σ + b − a)` is exactly `log h`. Continuous at `a` and `b`; finite everywhere; `support()` correctly returns `(−inf, inf)`. The sampler's mixture weights are right: the code draws `u ~ U(0, 1 + (b−a)/(√(2π)σ))` and assigns the plateau when `u ≥ 1`, giving plateau : each-tail = `(b−a)/(√(2π)σ)` : ½, which equals the true mass ratio `(b−a) : ½√(2π)σ`. KS p = 0.57, 0.67 (b−a = 100σ… 200σ) and 0.26 (plateau width 1e−3 with σ = 5, i.e. essentially a Gaussian). Defects: F2, F4.
- **`Product`** — `log_pdf` is the sum of the marginal log-densities in dimension order and `sample` fills column *m* from marginal *m*: checked with disjoint per-dimension supports ([0,1], [10,11], [100,101]) — the columns land in the right ranges and the in-order `log_pdf` is finite while the reversed one is `−inf`. Draws are independent (max |off-diagonal correlation| 0.005–0.017 over 10⁴–2·10⁵ draws) and all marginals share one generator, as documented. Mixed families (scipy `norm`, `UniformBox`, `Trapezoidal`, `SciPy(expon)`) pass per-column KS and the sum identity. Two defects: the recorded support is wrong for shifted scipy marginals (F1) and a `UserFunction` marginal makes `log_pdf` raise (F5).
- **`SciPy`** — `logpdf` and `rvs` are delegated to the frozen distribution, so `sample` and `log_pdf` agree by construction for frozen univariate continuous, multivariate normal and multivariate *t* distributions. Discrete frozen distributions are rejected at construction (`isinstance(distribution.dist, rv_continuous)`), as are non-frozen classes and unsupported multivariates. The generator passed to `sample` takes precedence over any `random_state` on the frozen distribution, as its docstring says, and the constructor draws nothing. The one defect is `support()` (F1); a frozen univariate with array-valued parameters is wrongly accepted as `D = 1` (F11).
- **`UserFunction`** — has no density of its own: `log_pdf` *is* the user's callable and `sample` *is* the user's sampler (or `None`), so nothing can be checked and nothing is: `convert_to_prior` compares the two callables by identity only, although its docstring says they "should agree".

One fact that bounds the consequence of any sampler defect: **`prior.sample` is never called anywhere in the algorithm.** `VBMC` stores it (and `__str__` prints it) but draws no initial points from it; the `sample_prior` argument is documented as "Currently unused". Every sampler here is exercised only by users and by the tests.

## 3. Findings

### F1. `SciPy.support()` returns the *unshifted, unscaled* support of a frozen univariate distribution
- Location: `pyvbmc/priors/scipy.py:69-70` (and `:122-135`), consumed at `pyvbmc/priors/product.py:66-67`; MATLAB: "not read (internal track)"
- Category: cross-module
- Proposed classification: suspected defect
- Confidence: high
- The code reads `distribution.a` / `distribution.b` from the frozen object. In SciPy, `rv_frozen.__init__` computes `shapes, _, _ = self.dist._parse_args(*args, **kwds); self.a, self.b = self.dist._get_support(*shapes)` — it **discards `loc` and `scale`**, so `.a`/`.b` are the support of the standard distribution, not of the frozen one. The frozen object's own `support()` method applies `loc`/`scale` and is what should be read. Measured: `SciPy(uniform(loc=2, scale=3)).support()` returns `([0.], [1.])` while the distribution lives on `[2, 5]`; `beta(2,3,loc=-1,scale=4)` → `([0.],[1.])` instead of `[-1,3]`; `expon(loc=5)`, `gamma(2,loc=3,scale=2)`, `lognorm(1.0,loc=4)` all report a lower bound of 0. The base-class docstring says `support()` returns bounds "such that [lb[i], ub[i]] bounds the support of the i-th marginal" — a box that excludes the whole support does not bound it.
- Consequence if real: `Prior.support()` is wrong for any location- or scale-shifted univariate scipy prior, and `Product.__init__` copies those wrong numbers into `self.a`/`self.b`, so `Product.support()`, `Product.__str__` and `Product.__repr__` all report the wrong box (measured: a `Product` of `uniform(loc=2,scale=3)` and `uniform(loc=-5,scale=1)` prints "lower bounds = [0. 0.], upper bounds = [1. 1.]"). No numerical effect on a `VBMC` run, because nothing in `vbmc.py` reads `support()`; the damage is to reported/inspected values and to any user or test that trusts them. The package's own unit-integral helper integrates the density over `support()`, which for `SciPy(uniform(loc=2,scale=3))` returns **0.0**.
- Suggested reproduction: ran `c1_scipy_support.py`; `SciPy(uniform(loc=2, scale=3)).support()` → `([0.],[1.])` vs `dist.support()` → `(2.0, 5.0)`.
- Test adequacy: no. `test_priors.test_unit_integral_1d/_2d` would catch it, but `SciPy._generic` builds `multivariate_normal(np.zeros(D))`, whose support is `(−inf, inf)` and which never touches this branch; `test_convert_to_prior` and `test_product_*` use only `norm()`, `lognorm(1.0)`, `beta(2,1.5)`, `t(nu)` — all unshifted, so `.a`/`.b` happen to be right for each of them.

### F2. Integer or float32 input arrays silently produce wrong log-densities (`np.full_like` inherits the input dtype)
- Location: `pyvbmc/priors/trapezoidal.py:96`, `pyvbmc/priors/spline_trapezoidal.py:98`, `pyvbmc/priors/smooth_box.py:79`; MATLAB: "not read (internal track)"
- Category: indexing/shape (dtype)
- Proposed classification: suspected defect
- Confidence: high
- `log_pdf = np.full_like(x, -np.inf)` takes its dtype from the caller's `x`. The base class `Prior.log_pdf` does no dtype coercion (`np.atleast_2d` preserves dtype). With an integer `x` the accumulator is `int64`: every computed log-density is truncated toward zero, and the `-inf` fill is cast to `INT64_MIN`. Measured for `Trapezoidal(0,2,8,10,D=2)` at `[[3,5]]`: `-4` (int64) instead of `-4.158883`; `SmoothBox` at the same point: `-4` instead of `-5.052518`. Worse, because the out-of-support fill is `INT64_MIN` and the per-dimension sum wraps in two's complement, a point with an **even number** of out-of-support coordinates comes back with log-density `0`, i.e. **density 1**: `Trapezoidal(0,2,8,10,D=2).log_pdf(np.array([[20,20]]))` → `0` → `pdf = 1.0`; D=4 likewise; D=1 and D=3 give `INT64_MIN` → `pdf = 0`. With a float32 `x` the same line returns a float32 log-density, which contradicts the package's "float64 everywhere" convention (`AGENTS.md`; the dtype canary). `UniformBox` (`np.full((n,1), …)`), `SciPy` and `Product` (`np.zeros((n,D))`) are not affected — though a `Product` with a trapezoidal marginal inherits the truncated value (measured −4.302585 instead of −4.382027).
- Consequence if real: only outside the run loop. Both call sites inside `VBMC` pass float64 (`FunctionLogger.__call__` hands over `parameter_transformer.inverse(...)`, and `vbmc.py:779` uses the float64 copy made by `_precomputed_float64_array`), so a run is unaffected. A user evaluating or plotting a prior on integer coordinates — which the public API documents as `np.ndarray` without a dtype restriction — gets silently truncated values and, in even dimensions, density 1 arbitrarily far outside the support.
- Suggested reproduction: ran `c3_dtype32.py` and the D=1…4 sweep above; both outputs are in the scratchpad.
- Test adequacy: no. Every prior test builds `x` from `np.random.normal` / `np.random.uniform` / `np.linspace`, so the accumulator is always float64. The oracle dtype canary never walks a prior.

### F3. `UniformBox` reports a finite density at a NaN coordinate
- Location: `pyvbmc/priors/uniform_box.py:69-70`; MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- The out-of-support mask is `np.any((x < self.a) | (x > self.b), axis=1)`. Both comparisons are `False` for NaN, so a row containing NaN is classified as inside the box and receives `-log ∏(b−a)`. Measured: `UniformBox(0,1,D=2).log_pdf([[nan, 0.5]])` → `-0.0`, i.e. `pdf = 1.0`, with no warning. The other box families return `-inf` for the same input (their masks are positive membership tests, so NaN falls through to the `-inf` fill), and `SciPy` propagates `nan`. Three different answers to the same question.
- Consequence if real: a NaN parameter is reported as having full prior density rather than zero. Inside `VBMC` the log-joint would then be `likelihood(NaN) + finite`, which `FunctionLogger` rejects on the likelihood term, so a run fails loudly rather than silently; the exposure is to direct use of the prior and to any caller that screens points with `prior.pdf`.
- Suggested reproduction: ran `c2_dtype_nan_edges.py`, section "NaN input".
- Test adequacy: no test passes a non-finite coordinate to any prior.

### F4. NaN and infinite constructor arguments are accepted by every family; with a NaN pivot, `sample` and `pdf` describe different distributions
- Location: `pyvbmc/priors/uniform_box.py:44-47`, `trapezoidal.py:72-77`, `spline_trapezoidal.py:74-79`, `smooth_box.py:54-61`; MATLAB: "not read (internal track)"
- Category: control flow (argument checks)
- Proposed classification: suspected defect
- Confidence: medium
- The validators are `np.any(a >= b)` / `np.any((a >= u) | (u >= v) | (v >= b))` / `np.any(scale <= 0)`. Every comparison involving NaN is `False`, so NaN parameters pass, and `±inf` satisfies the strict orderings, so infinite bounds pass too. The documented refusals (`u == a`, `v == b`, `u == v`, `a == b`, `scale == 0`) are all correctly enforced — it is only NaN and infinity that slip through. Measured consequences: `UniformBox(nan, 1)` → density NaN everywhere, `sample` raises `OverflowError`; `UniformBox(-inf, inf)` and `UniformBox(0, inf)` → density identically 0, `sample` raises `OverflowError`; `SmoothBox(0,1,inf)` → density 0 everywhere and samples `±inf`; `SmoothBox(0,1,nan)` → density NaN. The sharpest case is a NaN pivot: `Trapezoidal(0, nan, 0.75, 1)` and `SplineTrapezoidal(0, nan, 0.75, 1)` give a density that is `-inf` (zero) **everywhere**, while `sample` returns ordinary `U(a,b)` draws — the rejection test `z1 > y1` is `False` for NaN, so every proposal is accepted. `sample` and `pdf` then describe completely different distributions, which is the exact failure mode the first question asks about, reachable from a silently accepted argument.
- Consequence if real: a user who computes bounds (e.g. from data, or by passing the problem's infinite hard bounds to `UniformBox`) and produces a NaN or infinity gets a silently degenerate prior object instead of a `ValueError`. Inside a run it surfaces later as `FunctionLogger:InvalidFuncValue` or as NaN, far from the cause.
- Suggested reproduction: ran `c6_degenerate.py`; full table in the scratchpad.
- Test adequacy: no. `test_*_error_handling` covers only `a >= b`, a mis-ordered pivot, `scale = 0` and a shape mismatch.

### F5. `Product` accepts a `UserFunction` marginal, but `Product.log_pdf` cannot evaluate one
- Location: `pyvbmc/priors/product.py:88` (against `:112-114`); MATLAB: "not read (internal track)"
- Category: cross-module
- Proposed classification: suspected defect
- Confidence: high
- `Product.sample` has an explicit special case for `UserFunction` marginals ("`sample` is the user's own callable and takes only `n`", `:112-114`) with a dedicated test, so such marginals are clearly meant to be supported, and `Product.__init__` accepts any `Prior` with `D == 1`. But `_log_pdf` calls `marginal.log_pdf(x[:, m], keepdims=False)`, and for a `UserFunction` `marginal.log_pdf` *is* the user's callable, which is documented (`user_function.py:28-31`) to "take a one-dimensional array as a single argument". Measured: `Product([UserFunction(lambda x: float(np.sum(x)), sampler, D=1), UniformBox(0,1)]).log_pdf(x)` raises `TypeError: <lambda>() got an unexpected keyword argument 'keepdims'`. If the user's callable does tolerate `keepdims`, the second problem appears: it is handed a **column of `n` points** rather than one point, and a scalar return is broadcast to all `n` rows (measured: three different rows all given the same log-density).
- Consequence if real: `VBMC(log_likelihood, …, prior=[user_function_prior, other_marginal])` — a combination the documentation describes as supported ("a list of one-dimensional PyVBMC `Prior` and/or … `scipy.stats` distributions") — fails at the first target evaluation with an opaque `TypeError`, or, for a `**kwargs`-tolerant callable, silently evaluates the wrong density for multi-row inputs. `UserFunction(…, D=None)` is rejected earlier by the `marginal.D != 1` check, so only `D=1` user functions reach this.
- Suggested reproduction: the two one-liners above (logged in the scratchpad transcript).
- Test adequacy: no. `test_product_sample_user_function_marginal` exercises `sample` for exactly this configuration and never calls `log_pdf` on it — a test that pins half of a feature whose other half cannot work.

### F6. Nothing relates the prior's support to the problem's hard bounds
- Location: `pyvbmc/vbmc/vbmc.py:3471-3551` (`_init_log_joint`), `pyvbmc/priors/prior.py:62-78`; MATLAB: "not read (internal track)"
- Category: cross-module
- Proposed classification: possibly intentional
- Confidence: medium
- `_init_log_joint` checks only `prior.D != self.D`. `Prior.support()` exists and every bounded family implements `_support()`, but `vbmc.py` never calls it (the only consumers in the package are `Product.__init__` and two tests). Measured on a constructed `VBMC` (no `optimize()`): with `prior=UniformBox(0,1,D=2)` and hard bounds `[0,10]²` the object is built without complaint, and evaluating the target at `[5,5]` — a point well inside the hard box, exactly the kind active sampling proposes — raises `ValueError: FunctionLogger:InvalidFuncValue`, because the log-joint is `-inf`. In the opposite direction (`prior=UniformBox(-100,100,D=2)` on the same box) the run proceeds while the prior is normalized over its own support rather than over the hard box, so the reported ELBO is the evidence under a prior that is truncated by the bounds but not renormalized. The notebook and the docs always set the prior's bounds equal to the hard bounds (`SplineTrapezoidal(lb, plb, pub, ub)`), i.e. the design assumes agreement without enforcing it.
- Consequence if real: a mismatch either kills the run mid-flight with a message that names neither the prior nor the bounds, or silently shifts the log model evidence by the log of the prior mass inside the box. Triggered whenever a user's prior support and hard bounds differ.
- Suggested reproduction: ran `c7_vbmc_integration.py` (both directions).
- Test adequacy: no. `test_vbmc_init_log_joint_prior` uses `lb=-inf, ub=+inf` throughout, so support and bounds never disagree.

### F7. Docstrings that contradict the code, in the public (autodoc'd) API
- Location: `pyvbmc/priors/prior.py:17-19` and `:50-52`; `uniform_box.py:20-23`, `trapezoidal.py:33-40`, `smooth_box.py:22-27`, `spline_trapezoidal.py:35-42`; `convert_to_prior.py:28-35`; `spline_trapezoidal.py:101`; MATLAB: "not read (internal track)"
- Category: defaults (documentation)
- Proposed classification: suspected defect
- Confidence: high
- Four separate mismatches, all rendered into `docsrc/source/api/classes/priors.rst`: (a) the `keepdims` parameter of `Prior.log_pdf` and `Prior.pdf` is described as choosing between shapes `(1, D)` and `(D,)`, contradicting the Returns section of the same docstring, which correctly says `(n, 1)` and `(n,)`; (b) every box family documents its attributes `a`, `u`, `v`, `b`, `scale` as shape `(1, D)`, but `tile_inputs(..., squeeze=True)` stores them as `(D,)` (measured for all three families); (c) `convert_to_prior` documents the parameter name `sample_prior` twice — the first block ("A function of a single argument `x` which returns the log-density") describes `log_prior`, which is therefore undocumented; (d) the in-code comment at `spline_trapezoidal.py:101` writes the normalizer as `u - v + 0.5 * (b - v + u - a)` where the correct (and implemented) expression is `v - u + 0.5*(b - v + u - a) = 0.5*(v - u + b - a)`.
- Consequence if real: readers of the API documentation are told the wrong shapes for the returned arrays and the stored parameters, and one public keyword is undocumented; a maintainer re-deriving the spline normalizer from the comment gets a sign error.
- Suggested reproduction: attribute shapes printed in the scratchpad (`{'a': (3,), 'b': (3,)}` etc.); the rest is by reading.
- Test adequacy: no test compares a docstring with behavior.

### F8. Divide-by-zero warning at `x == a`, suppressed in one trapezoid and not the other, and with a non-exception-safe global `seterr`
- Location: `pyvbmc/priors/trapezoidal.py:104-108` vs `pyvbmc/priors/spline_trapezoidal.py:105,124`; MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: suspected defect
- Confidence: medium
- `Trapezoidal._log_pdf` evaluates `np.log(x - a)`, which emits `RuntimeWarning: divide by zero encountered in log` whenever an input equals the lower bound exactly (measured). The value `-inf` is correct, so no wrong value is hidden, but `SplineTrapezoidal` suppresses the identical warning and `Trapezoidal` does not — one of the two is wrong. `SplineTrapezoidal`'s suppression uses `old_settings = np.seterr(divide="ignore")` … `np.seterr(**old_settings)` around the loop rather than a `with np.errstate(...)` block, so any exception raised inside the loop leaves "divide" ignored for the rest of the process (or thread), silencing later genuine warnings.
- Consequence if real: cosmetic in the first part (a spurious warning on a boundary evaluation, e.g. plotting a prior on a grid that includes `lb`); in the second part, a leaked global error state after an unrelated failure.
- Suggested reproduction: ran `c2_dtype_nan_edges.py` — `Trapezoidal` reports `['divide by zero encountered in log']` for the same point set on which `SplineTrapezoidal` reports none.
- Test adequacy: no test asserts on warnings from the priors.

### F9. `sample_prior` bypasses the PyMC guard and can build a prior with no density
- Location: `pyvbmc/vbmc/vbmc.py:573` and `:3474-3479`; `pyvbmc/priors/convert_to_prior.py:50-51`; MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: possibly intentional
- Confidence: low
- The `PyMCTarget` guard is `if prior is not None or log_prior is not None`, omitting `sample_prior`, while `_init_log_joint` enters the prior branch when *any* of the three is given. So `VBMC(pymc_target, …, sample_prior=f)` is accepted and yields a `PyMCTarget` run whose `self.prior` is a `UserFunction` with `log_pdf = None`. Separately, `convert_to_prior(sample_prior=f)` alone returns a `UserFunction` whose `log_pdf` is `None` (measured), which `_init_log_joint` then treats as "no prior", so `log_density` is silently interpreted as the log-joint. Both are harmless today only because `sample_prior` is never used.
- Consequence if real: an inconsistent object state and a guard that does not cover all three prior arguments; no numerical effect while `sample_prior` stays unused.
- Suggested reproduction: `convert_to_prior(sample_prior=lambda n: np.zeros((n,2)), D=2)` → `UserFunction, log_pdf is None, D = 2` (ran); the PyMC half is by reading, not run (it needs a PyMC model).
- Test adequacy: partially — `test_vbmc_init_log_joint` asserts `vbmc.prior.log_pdf is None` for the `sample_prior`-only case, i.e. it pins the behavior rather than questioning it.

### F10. `tile_inputs` silently reshapes an argument whose shape disagrees with `size`
- Location: `pyvbmc/priors/tile_inputs.py:37-42, 56`; MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: medium
- The docstring promises `ValueError` "if the non-scalar arguments do not have the same shape, **or if they do not agree with `size`**". Only the first is checked: non-scalar arguments are compared against each other, never against `size`; the second is left to `reshape`, which succeeds whenever the element count matches. Measured: `UniformBox(np.zeros((2,2)), np.ones((2,2)), D=4)` builds a `D = 4` prior from `(2,2)` inputs without complaint.
- Consequence if real: a mis-shaped bound array is silently flattened into a prior of the requested dimension, mapping parameters to the wrong coordinates. Needs the element counts to coincide, so it is rare.
- Suggested reproduction: ran `c6_degenerate.py`, section "tile_inputs".
- Test adequacy: no. `test_tile_inputs_wrong_size` uses shapes whose element counts differ, so `reshape` raises and the gap is invisible.

### F11. A frozen univariate with array-valued parameters is accepted as a one-dimensional `SciPy` prior
- Location: `pyvbmc/priors/scipy.py:22-30, 67-70`; MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: unsure
- Confidence: low
- `_scipy_distribution_kind` only checks the frozen type, so `norm(loc=np.array([0., 10., 100.]))` is classified univariate with `D = 1`. `sample(3)` then returns a `(3,1)` array whose entries come from three *different* normals (measured: `0.126, 9.868, 100.640`), while `log_pdf` raises `ValueError: cannot reshape array of size 9 into shape (3,1)`. `VBMC` catches the case indirectly through `prior.D != self.D`, so the exposure is to direct use of the wrapper and to `Product` marginals.
- Consequence if real: a natural way to write a per-parameter prior (`scipy.stats.norm(loc=mu_vector)`) produces an object whose sampler and density disagree about what it is. Rejecting it at construction, or reading the parameter shape, would be the alternative.
- Suggested reproduction: ran the one-liner above; output in the scratchpad transcript.
- Test adequacy: no test builds a vectorized frozen distribution.

## 4. Test adequacy notes

- **The trapezoidal and spline-trapezoidal samplers have no distributional test at all.** `test_priors.test_sample` checks only `samples.shape` and that the draws lie inside `support()` — a sampler returning uniform draws over `[a, b]` would pass it for both trapezoids. The only sampler test that looks at a distribution is `test_product_mixed_distribution_type` (means and variances of 200 000 draws), and its marginals are `norm`, `lognorm`, `UniformBox`, a degenerate `SmoothBox(0, eps)` and `beta` — neither trapezoid appears. Both samplers are in fact correct (KS p = 0.43–0.53 against the CDFs I derived), but nothing in the suite would have said so. This is the same shape as the defect that motivated this review.
- **The unit-integral tests only ever see `_generic` instances**: unit-scale, identical parameters in every dimension, and for `SciPy` a standard multivariate normal. Normalization with parameters differing across dimensions, with shifted scipy distributions, or with an extreme aspect ratio is never integrated. F1 is invisible for exactly this reason: the one test that would compute `∫ pdf` over `support()` never meets a distribution whose `support()` is wrong.
- **Several assertions in the VBMC-side prior tests are missing their `assert`.** `pyvbmc/testing/vbmc/test_vbmc_init.py` contains bare `np.isclose(vbmc.log_joint(x), log_likelihood(x) + new_prior.log_pdf(x))` statements at :836, :837, :915, :945-947, :971-976, :1004-1006 (and similar in `test_vbmc_init_log_joint_noisy_prior`). They compute a boolean and discard it. The claim that `VBMC`'s log-joint equals likelihood + prior is therefore asserted in exactly one place (:899-901, `assert vbmc.log_joint(x) == …`, for the `prior=`-only noiseless path); the noisy path, the `log_prior=` path and the scipy/list paths are unchecked.
- **`test_user_function_prior_sample_rng_ignored` and `test_product_sample_user_function_marginal` pin the implementation rather than a specification.** The first asserts that the unused class-level `UserFunction.sample` returns `None`; the second asserts that a `UserFunction` marginal can be sampled inside a `Product` whose density, as F5 shows, cannot be evaluated at all. Together they certify a half-working feature.
- **`test_priors.test__str__and__repr__` asserts that `__str__` contains `lower bounds = {new_prior.a}`,** i.e. it compares the printout with the attribute rather than with the distribution's actual support. For a `Product` of shifted scipy marginals both are wrong in the same way, so the test passes (F1).
- The error-handling tests are genuinely specification-shaped (they name the documented conditions), but they stop at the documented list and never probe NaN or infinity (F4).
