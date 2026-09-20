# N2 — PyMC target adapter and posterior exports, internal-correctness track (rerun with the PyMC environment)

Raw reviewer report, wave 0 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-19. A second, fresh Opus
agent on the same slice as `N2_internal.md`, this time told which
interpreters hold PyMC, ArviZ and Torch, so that it could exercise the code
(the first reviewer worked statically). It did not see the first report.
Its check scripts (`c1` to `c25`) ran in the session scratchpad and are not
retained. Interpreters: **[P]** is
`dev/scripts/runs/pymc_feasibility_20260914/venv/Scripts/python.exe`
(PyMC 6.3.2, PyTensor 3.3.1, ArviZ 1.3.0), **[T]** is
`../pyvbmc-stage3/.venv/Scripts/python.exe` (Torch 2.7 CPU). The text below
is the agent's final message, unedited.

---

## 1. Coverage

### Read completely
- `pyvbmc/pymc/_target.py` (1317 lines), `pyvbmc/pymc/_compat.py`, `pyvbmc/pymc/_snapshot.py`, `pyvbmc/pymc/_plausible.py`, `pyvbmc/pymc/__init__.py`
- `pyvbmc/variational_posterior/_torch.py`, `pyvbmc/variational_posterior/_arviz.py`

### Read as needed (interfaces, not under review)
- `pyvbmc/parameter_transformer/parameter_transformer.py` — whole file (the forward/inverse/Jacobian formulas and the `_to_unit_interval`/`_from_unit_interval` "safe" nudges are the reference for `_torch.py`).
- `pyvbmc/variational_posterior/variational_posterior.py` — `to_torch`, `to_arviz`, `sample`, `pdf`/`_pdf` headers.
- `pyvbmc/vbmc/vbmc.py` — `_resolve_pymc_target`, `target` property, `_validate_initialization_cost`, `_initialize_precomputed_evaluations`, `_validate_vectorized_target_option`; `pyvbmc/vbmc/_bounds.py` in full.
- `pyvbmc/function_logger/function_logger.py` — call-shape path only (target is called with `(1, D)`).
- Tests skimmed for adequacy: `pyvbmc/testing/pymc/{models,test_target,test_rejections,test_export,test_target_setup}.py`, `pyvbmc/testing/variational_posterior/test_vp_torch.py` (fixtures + first three tests in full).

### Not reached
`pyvbmc/testing/pymc/{test_guards,test_imports,test_model_snapshot,test_save_load,test_target_snapshot,test_vbmc_binding,test_plausible,test_optimize_short}.py` were not read; pickling/save-load of a `PyMCTarget` was not exercised. `pyvbmc/whitening/whitening.py` only grepped for how it installs `R_mat`/`scale`.

### Checks run

| # | Check | Int. | Outcome |
|---|---|---|---|
| c1 | `log_joint` / `log_joint_no_jacobian` for all 5 accepted models vs the hand-written densities plus the retained-transform log-Jacobian | P | agree to ~1e-7 (a per-model constant from the hand function's summation order); the Jacobian term equals the sum of the kept coordinates, as `lower+exp(v)` / `upper-exp(v)` / `exp(v)` require |
| c2 | gradient and Hessian ordering on `mixed_order_model` (partial transform removal reorders PyMC's `free_RVs`) vs central differences of `log_joint` | P | gradient max diff 4.7e-10, Hessian 4.1e-10; ordering follows `self.names`, not `_partial.free_RVs` |
| c3 | snapshot isolation: `pm.set_data` + `pytensor.shared.set_value` after construction; `__deepcopy__` | P | `log_joint` bit-identical before/after (live model moved by 421 nats); deepcopy clones `model`/`_snapshot`/`_partial`, shares compiled callables; `coords` deep-copied |
| c4 | `PyMCTarget.to_arviz`: layout, coords, attrs, rng advancement, rng restore on validation error | P | dims `(chain, draw, coef)`, values bit-equal to `to_model_variables(vp.sample(...))`, `vp.rng` advanced exactly as one `vp.sample`, unchanged on error |
| c5/c19 | torch `log_prob` (orig + internal) vs `vp.pdf` across logit/probit/student4, D=1/2/3, K=1..3, with and without whitening installed the way `pyvbmc/whitening` does it | T | ≤4e-10 away from the bounds; see **F3** for the near-bound divergence |
| c6/c7/c20/c21 | which side is accurate near a bound, against an mpmath reference | T | torch `_inverse` exact to 1 ulp; the NumPy `ParameterTransformer.__call__` loses up to 2e-2 within ~1000 ulps of a non-zero bound |
| c8 | torch: no draws during conversion (torch RNG / NumPy global / `vp.rng`), no torch state left on VP or transformer, zero-weight component never drawn, empirical mean/cov vs `vp.moments`, samples strictly inside bounds, dtype/device, `float16` rejected, out-of-support `log_prob`, autograd vs FD, `expand`, `has_rsample` | T | all as documented; grad max rel diff 5.5e-8 |
| c9 | edge cases: no free RVs, gradient-free model, `initval="prior"` seeding/reproducibility, explicit `start`/`plausible_bounds` combinations, `setup_budget` validation and the exact-minimum budget, boundary `start`, on/outside-bound `log_joint`, wrong shapes, all 8 `rejected_models` | P | all behave as documented; budget accounting never exceeds `setup_budget` |
| c10 | multi-dimensional variables `(2,3)`, `(2,)`, `(2,2)` with declared and generated dims; flatten/unflatten/`to_model_variables` C-order consistency; ArviZ export | P | consistent; `coordinate_names` follow `np.ndindex` C order and match the flat layout |
| c11 | 2-iteration `D=1` `VBMC(target)` run + `to_arviz` on the fitted VP + `deepcopy(VBMC)` | P | `VBMC.target is t`, `initialization_cost == setup_cost == 4`, transformer shared by vbmc/vp/logger, run completes, posterior mean 1.63 vs analytic 1.57 |
| c12 | distributions whose default transform may not match their support (Pareto, Wald, Triangular, Weibull, Gamma, Kumaraswamy, Moyal, ExGaussian, AsymmetricLaplace, LogNormal, HalfStudentT, Rice, SkewStudentT) | P | **F1** (Wald) and **F2** (Rice) found here |
| c13/c14 | `MethodNotDefined` class hierarchy and the failing frame for `pm.Rice` | P | `MethodNotDefined(Exception)` — not a `NotImplementedError`; escapes from `Model.dlogp` |
| c15/c24 | Wald(`alpha=2`) reproduction, including a full short `VBMC.optimize()` | P | run aborts with `ValueError: Non-finite log density for model variables x=0.589…` |
| c16 | one-sided **upper** bound (`TruncatedNormal(upper=-0.5)`): support, map monotonicity, the `min/max` swap in `_validate_plausible_bounds`, extreme coordinates | P | swap handled correctly; **F4** found here |
| c17 | `_student4_icdf` vs `scipy.stats.t(4).ppf` over p ∈ [1e-300, 1−1e-16]; the forward t4 CDF across the \|v\|=2 branch switch vs `t(4).cdf` | T | icdf max rel err 3.8e-13; cdf max abs err 1.1e-16 |
| c18 | `log_abs_det_jacobian` vs the FD log-determinant of `_call`, and vs `ParameterTransformer.log_abs_det_jacobian`, for all types ± whitening | T | FD ≤1.1e-8, NumPy ≤8.9e-16 |
| c22 | dim-metadata edge cases (partial `None` dims, dim name == variable name, dim without coords, `__str__`) | P | PyMC itself rejects the first two; the third and fourth are handled |
| c23 | `vp.to_arviz` structured layout: block slicing/reshape vs manual, default names, `orig_flag=False`, and 7 validation paths | P | all correct |
| c25 | `setup_evaluations` strictly inside the hard bounds and flowing into the logger, for all 5 accepted models | P | strict-inside True, no duplicate rows, `init_cost == setup_cost`, logger rows == setup rows |
| — | baseline: `test_target.py::test_accepted_layout_and_support` [P], `test_vp_torch.py::test_centers_and_tails` [T] | | both pass |

---

## 2. Findings

### F1. A variable's support is inferred from the transform kind, not from the distribution, so a `log`-transformed shifted-support variable gets a wrong `support` and a coordinate range on which the joint is `-inf`
- Location: `pyvbmc/pymc/_target.py:349` (and the consequence at `pyvbmc/pymc/_target.py:1133`)
- Category: formula/gradient (support derivation) + control flow
- Proposed classification: suspected defect
- Confidence: high (facts reproduced); medium that it is out of the adapter's intended scope

  The branch
  ```python
  if kind == "log":
      lower = np.zeros(shape, dtype=np.float64)
      upper = np.full(shape, np.inf, dtype=np.float64)
  ```
  equates "PyMC attached a `LogTransform`" with "the support is `(0, ∞)`". That holds for Gamma, Weibull, LogNormal, HalfStudentT, Exponential and so on, but not for `pm.Wald(mu, lam, alpha)`, whose support is `(alpha, ∞)` while PyMC 6.3.2 still registers `LogTransform` as its default. The adapter has no cross-check: `_supported_transform_classes` and `_validate_source_transform_calls` only verify the transform's *class* and call signature, and `is_known_real_line` is consulted only when there is no transform at all. The `interval` branch does read the real limits (`args_fn`), which is why `pm.Pareto` (default `Interval`, lower `m`) comes out right; only the `log` branch hardcodes.

  Two consequences follow. (a) The documented `support` attribute ("Model-variable names mapped to `(lower, upper)` support arrays") is factually wrong, so `_validate_model_point` and `_validate_plausible_bounds` accept a `start` or a plausible box that lies entirely in the dead region. (b) Because the kept transform makes the PyVBMC hard bounds `±inf`, the whole interval `v < log(alpha − 0)` is "strictly inside the box", where `_evaluate_density` raises `ValueError` rather than returning `-inf`.

- Consequence if real: an ordinary, otherwise fully supported model aborts the VBMC run. With `pm.Wald(mu=1, lam=2, alpha=2)` the mode is at x≈3.0 and the automatic plausible box is `[0.947, 1.259]` in log coordinates, i.e. x∈[2.58, 3.52] — so a short run survives, but any acquisition step that probes below `log 2 = 0.693` kills it. Supplying a plausible box the adapter accepts as valid (`{"x": (0.5, 1.5)}`) makes it fail on the first design point. Impact is confined to distributions with a `log` default transform and a nonzero shift; in PyMC 6.3.2 `Wald` with `alpha != 0` is the case I found.
- Reproduction (`c12`, `c15`, `c24`, interpreter **[P]**):
  ```python
  with pm.Model() as m:
      pm.Wald("x", mu=1.0, lam=2.0, alpha=2.0)      # true support (2, inf)
  t = PyMCTarget(m, seed=0)
  t.support            # {'x': (array(0.), array(inf))}   <- claims (0, inf)
  t.log_joint(np.array([np.log(1.9)]))
  # ValueError: Non-finite log density for model variables x=1.9.
  PyMCTarget(m, start={"x": 1.0}, seed=0)
  # accepted by _validate_model_point, then
  # ValueError: Non-finite log density for model variables x=1.0.
  t2 = PyMCTarget(m, start={"x": 3.0}, plausible_bounds={"x": (0.5, 1.5)}, seed=0)  # accepted
  VBMC(t2, options={"max_iter": 2, ...}).optimize()
  # ValueError: Non-finite log density for model variables x=0.5895145243614798.
  ```
- Test adequacy: no. `test_target.py::test_accepted_layout_and_support` asserts `positive.support["sigma"] == (0, inf)` for a `HalfNormal`, where the transform and the distribution happen to agree; none of the five accepted models, and none of the eight `rejected_models`, has a shifted support under a `log` transform. `test_rejections.py` covers unsupported *transform classes*, not supported classes with a mismatched support.

---

### F2. `_compile_gradient` and `_compile_hessian` catch only two of PyTensor's "no gradient" signals, so a partially non-differentiable model fails at construction instead of taking the documented gradient-free route
- Location: `pyvbmc/pymc/_target.py:836` (and the identical clause at `pyvbmc/pymc/_target.py:859`)
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high

  Both helpers are documented as "Compile the ordered joint value/gradient, or return None" / "…the ordered exact Hessian, or return None", and the whole `route == "prior"` branch (`_target.py:667`, the warning at `_target.py:1187`) exists to serve models without a gradient. The guard is
  ```python
  except (NotImplementedError, NullTypeGradError):
      return None
  ```
  PyTensor signals a missing gradient in more than those two ways. When a scalar `Op` implements no `pullback`/`grad` at all, `pytensor.scalar.basic` raises `MethodNotDefined`, whose MRO is `(MethodNotDefined, Exception, BaseException, object)` — it is neither a `NotImplementedError` nor the `TypeError`-derived `NullTypeGradError`. It therefore escapes `PyMCTarget.__init__`.

- Consequence if real: `PyMCTarget(model)` raises a raw PyTensor error (`MethodNotDefined: ('pullback', <class 'pymc.distributions.dist_math.I1e'>, 'I1e')`) for a model that is otherwise entirely within the supported scope (float64, `LogTransform`, fixed data). The user gets neither the adapter's actionable `ImportError`/`UnsupportedModel` message nor the prior-quantile fallback that the same model would get if the gradient had failed one line differently. Any PyMC distribution whose logp routes through an `Op` without a gradient implementation is affected; `pm.Rice` (Bessel `I1e`) is the concrete one I found.
- Reproduction (`c12`, `c13`, `c14`, interpreter **[P]**):
  ```python
  with pm.Model() as m:
      pm.Rice("x", nu=1.0, sigma=1.0)
  PyMCTarget(m, seed=0)
  # pytensor.graph.utils.MethodNotDefined: ('pullback', <class 'pymc.distributions.dist_math.I1e'>, 'I1e')
  # traceback passes through pymc/pytensorf.py:338 grad_i  ->  Model.dlogp  ->  _compile_gradient
  ```
- Test adequacy: no. The only gradient-free fixture is `models.no_gradient_model()`, built with `pytensor.compile.ops.as_op`, which yields a `NullType` gradient and therefore `NullTypeGradError` — exactly one of the two caught types. `test_rejections.py::test_recognized_prior_with_custom_likelihood_is_accepted` and `test_target_setup.py::test_setup_diagnostic_routes_and_warnings` both use it, so the fallback is exercised only on the path that already works.

---

### F3. The torch export's density in original coordinates diverges from `vp.pdf(..., orig_flag=True)` near a hard bound, by up to ~0.5 nat
- Location: `pyvbmc/variational_posterior/_torch.py:147` (`_InverseParameterTransform._inverse`)
- Category: formula/gradient
- Proposed classification: possibly intentional
- Confidence: high (behavior reproduced and attributed)

  `_inverse` computes both distances to the bounds, `lower_distance = v - lower[d]` and `upper_distance = upper[d] - v`, and then takes the *smaller* tail directly (`ndtri` of the small side for probit, `log(lower_distance) - log(upper_distance)` for logit, `min(p, 1-p)` in `_student4_icdf`). `ParameterTransformer.__call__` instead forms one unit-interval value `z = (x - lb)/(ub - lb)` (`parameter_transformer.py:515`), which loses the upper tail to cancellation whenever `lb != 0`. The torch route is the accurate one; against an mpmath reference with `lb=-2, ub=3`:

  | gap below `ub` | NumPy `pt(x)` error | torch `inv(x)` error |
  |---|---|---|
  | 1e-6 | 1.6e-11 | 4.4e-16 |
  | 1e-12 | 2.2e-05 | 0 |
  | 1e-14 | 3.0e-03 | 0 |
  | 1 ulp | 2.0e-02 | 0 |

  The code comments state the intent ("Compute the small tail directly; 2*p-1 loses tiny p"; "The width cancels; log distances avoid division loss"), so this is a deliberate improvement. What is not stated anywhere a reader of `to_torch` would see is that the exported density therefore *does not reproduce* `vp.pdf`: the `to_torch` docstring only warns that "Explicit float32 exports have less resolution near bounds", which implies the float64 export is faithful.

- Consequence if real: for a posterior with mass pushed against a bound, `to_torch().log_prob(x)` and `np.log(vp.pdf(x))` differ. In a realistic draw set (`c19`: D=3, bounds `[-2,0,-1]`/`[3,1,5]`, probit, whitened, 400 draws) the maximum disagreement was 0.46 nat overall and 0.21 nat restricted to draws more than one ulp from a bound; logit reached 1.5e-6 and student4 2.4e-14. Downstream torch inference (importance weights, stacking) uses the more accurate value, so this is a benign-direction inconsistency, but it makes the two APIs non-interchangeable.
- Reproduction: `c19` (side-by-side on a VP), `c21` (mpmath adjudication), interpreter **[T]**.
- Test adequacy: no test would catch it, by construction. `test_vp_torch.py::test_density_and_gradient` builds its points as `pt.inverse(u)` for `u ∈ [-0.6, 0.8]`, never entering the near-bound regime, and compares at `atol=2e-11`. `test_centers_and_tails` does probe the tails (`p = 1e-20 … 1−1e-12`) but first sets `pt.lb_orig[:] = 0; pt.ub_orig[:] = 1`, the one geometry where `(x-lb)/(ub-lb)` is exact and the two routes cannot disagree; it then checks torch against scipy rather than against `vp.pdf`. That is the right way to test the torch code, but it leaves unstated which of the two is authoritative near a bound.

---

### F4. `to_model_variables` rejects a whole batch when a single draw's backward map is absorbed onto its bound
- Location: `pyvbmc/pymc/_target.py:1093`
- Category: control flow (boundary condition)
- Proposed classification: possibly intentional
- Confidence: medium

  For a kept one-sided `Interval` transform the backward map is `lower + exp(v)`. When `exp(v) < ulp(lower)/2` the sum rounds to exactly `lower`, and the check `np.any(value <= lower)` then raises for the entire `(n, D)` batch. The threshold depends on the magnitude of `lower`, not on how extreme the draw is: for `lower = 1.0` absorption starts at `v ≈ −36.7`, for `lower = 1e6` at `v ≈ −16.7`. A `log` transform (`exp(v)`, no addition) only underflows at `v < −745`, so the one-sided interval case is far more reachable. `PyMCTarget.to_arviz` propagates this, so one saturated draw out of `n_samples` fails the whole export (with `vp.rng` correctly restored).

- Consequence if real: `to_arviz` raises `ValueError: Mapped model values for 't' must be finite and strictly inside its support.` instead of returning draws, for a posterior that sits close to a truncation bound whose value is large in magnitude. No partial result and no guidance on what to change. The comparable PyVBMC-side map (`_from_unit_interval`, `parameter_transformer.py:526`) nudges to `nextafter` instead of rejecting.
- Reproduction (`c16`, interpreter **[P]**), on `models.one_sided_model()` (`t` truncated at `lower=1.0`):
  ```python
  t.to_model_variables(np.array([[-50.0, -50.0]]))
  # ValueError: Mapped model values for 't' must be finite and strictly inside its support.
  # because 1.0 + exp(-50) == 1.0 in float64
  ```
- Test adequacy: the strictness itself is pinned by `test_target.py::test_model_mapping_rejects_nonfinite_or_boundary_transform_output`, but only with `value = ±1000` on the `positive` (pure `exp`) target, where the rejection is genuinely unavoidable (`exp(-1000) == 0`). No test covers the `lower + exp(v)` absorption, which begins two orders of magnitude earlier in `v` and scales with `lower`.

---

## 3. Test adequacy notes

- **`test_target.py::test_model_mapping_rejects_nonfinite_or_boundary_transform_output`** mirrors the implementation. It asserts that `to_model_variables` raises, without stating a requirement the export has to meet; the chosen inputs (`±1000` on a pure `exp` map) are the case where any implementation must fail, so the test would pass unchanged if the strict check were replaced by a `nextafter` nudge on one side and not the other. See F4.
- **`test_vp_torch.py::test_density_and_gradient`** is the only test that ties `to_torch` to `vp.pdf`, and its sample points are constructed by pushing moderate unconstrained values through the transform, so the comparison region is chosen (implicitly) to be where the two agree. The independent-reference tests (`test_centers_and_tails` against `scipy`) are the stronger ones, and they neutralize the geometry (`lb=0, ub=1`) under which the NumPy and torch inverses part company. Together they document neither that the two APIs differ near a bound nor which is authoritative. See F3.
- **The gradient-free route is pinned by a single fixture** (`models.no_gradient_model`, `as_op`) used by both `test_rejections.py::test_recognized_prior_with_custom_likelihood_is_accepted` and `test_target_setup.py::test_setup_diagnostic_routes_and_warnings`. It exercises one of the two exception types the code catches, so the tests describe the implementation's `except` clause rather than the contract "a model without a usable gradient falls back to the prior-quantile route". See F2.
- **`test_target.py::test_accepted_layout_and_support`** asserts the `support` dict only for models where the transform and the distribution agree, so it pins the transform-derived value, not the support. See F1.
- Positively: `test_export.py` and `test_target_setup.py` are specification-shaped — the export test checks values against an independently seeded reference posterior and then feeds the result back through `pm.compute_deterministics` / `pm.sample_posterior_predictive` on the original model, and the setup tests assert the exact budget arithmetic per route rather than the numbers the code happens to produce. My own accounting check (`c25`) found no discrepancy in `setup_cost` / `initialization_cost` / `setup_evaluations` for any accepted model.
