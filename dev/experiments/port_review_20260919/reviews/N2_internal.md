# N2 — PyMC target adapter and posterior exports, internal-correctness track

Raw reviewer report, wave 0 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-19. One Opus agent,
fresh context, read-only on the repositories; its check scripts ran in the
session scratchpad and are not retained. The text below is the agent's
final message, unedited. Note its coverage section: the agent used an
interpreter without PyMC, Torch, ArviZ or gpyreg and reviewed those parts
statically.

---

## 1. Coverage

### Read completely
- `pyvbmc/pymc/_target.py` (all 1316 lines)
- `pyvbmc/pymc/_compat.py`, `pyvbmc/pymc/_plausible.py`, `pyvbmc/pymc/_snapshot.py`, `pyvbmc/pymc/__init__.py`
- `pyvbmc/variational_posterior/_torch.py`, `pyvbmc/variational_posterior/_arviz.py`
- `pyvbmc/parameter_transformer/parameter_transformer.py` (the reference for the Torch export)
- `pyvbmc/vbmc/_bounds.py` (`_effective_bounds`, `_normalize_bounds`), `pyvbmc/rng.py`
- Tests read completely: `pyvbmc/testing/pymc/{models,test_target,test_target_setup,test_export,test_guards,test_plausible,test_rejections,test_vbmc_binding,test_save_load}.py`, `pyvbmc/testing/variational_posterior/test_vp_torch.py`

### Skimmed
- `pyvbmc/variational_posterior/variational_posterior.py` — `to_torch`, `to_arviz`, `sample`, `pdf` (lines 390–950) read in full; the rest skimmed.
- `pyvbmc/vbmc/vbmc.py` — `_resolve_pymc_target`, `_normalized_hard_bound`, `_validate_initialization_cost`, `_initialize_precomputed_evaluations`, the `target` property and the `vectorized_target` guard read in full; the rest grepped.
- `pyvbmc/function_logger/function_logger.py` — only the Jacobian-correction sites (`_record`, lines 690–752) and `__deepcopy__`.
- `pyvbmc/testing/pymc/{test_target_snapshot,test_imports}.py` (partially), `pyvbmc/testing/variational_posterior/test_vp_arviz.py` (test names and assertions only).

### Not reached / not verifiable here
- **None of the optional dependencies are installed in this environment** (`pymc`, `pytensor`, `torch`, `arviz_base`, `xarray` all missing), and `gpyreg` is missing too, so `import pyvbmc` itself fails. Everything touching PyMC, PyTensor, Torch or ArviZ was reviewed **statically**; no PyMC model was constructed and no `PyMCTarget`, `to_torch` or `to_arviz` call was executed.
- Consequently the deep PyTensor graph surgery in `_snapshot.snapshot_model` (the `ModelFreeRV` value-variable rebuild at `pyvbmc/pymc/_snapshot.py:135–147`, the `clone_replace`/`FunctionGraph` reconstruction) could not be exercised. I judged it only for internal consistency and for the presence of the capability guards; its agreement with PyMC's own semantics is untested here.
- `pyvbmc/testing/pymc/{test_model_snapshot,test_optimize_short}.py` not read.

### Small checks run
1. `_compat._expit` and `_compat.closed_form` loaded standalone via `importlib` (bypassing `pyvbmc/__init__.py`) and exercised on 0-d, scalar, 1-D and 2-D inputs, and on two-sided / lower-only / upper-only interval bounds. All return correct values and shapes; no 0-d boolean-indexing failure.
2. A NumPy transcription of the Torch formulas in `_torch.py` (`_student4_icdf`, the Student-t(4) CDF branch of `_call`, the probit branch of `_inverse`, the logit and student4 Jacobian terms) checked against `scipy.stats.t`, `scipy.special.ndtri` and the corresponding helpers copied verbatim from `parameter_transformer.py`, over ~220 points including `p = 1e-300 … 1-1e-12` and the branch switches (`p = 0.0999/0.1/0.1001`, `|v| = 1.9999/2/2.0001`). Results: the Student-t(4) CDF agrees with SciPy to 2.2e-16 absolute; the inverse CDF agrees to 4.1e-8 relative in the worst case at `p = 0.5 ± 1e-5`, and a round trip through the CDF shows the *Torch* formula is the accurate one there while SciPy's `t.ppf` is the one that loses precision. The logit Jacobian identity `-softplus(v) - softplus(-v) == -v + 2 log σ(v)` agrees to 3.6e-15 over `v ∈ [-800, 800]`; the student4 Jacobian term is bit-identical.
3. Hand-verification (algebraic, not executed) that `_InverseParameterTransform._call`, `._inverse` and `.log_abs_det_jacobian` reproduce `ParameterTransformer.inverse`, `.__call__` and `.log_abs_det_jacobian` for types 0, 3, 12, 13 including the `scale`/`R_mat` whitening; and that `torch.distributions.TransformedDistribution.log_prob`'s sign convention matches (`log p_x = log p_u − log|det dx/du|`).

No pytest run was possible (all the relevant modules `importorskip` a missing dependency, and `pyvbmc` cannot be imported without `gpyreg`).

Things I checked and found correct, so that the absence of findings there is informative: the log-density returned by `log_joint` is `self._partial.logp(jacobian=True)`, which adds a Jacobian term for exactly the one-sided transforms that were kept and none for the two-sided ones that were removed — the right target for VBMC, since `FunctionLogger._record` separately adds VBMC's own transform Jacobian; the log-Jacobian of every kept transform is `v` itself (PyMC's `IntervalTransform.log_jac_det` and `LogTransform.log_jac_det` both return `value` in the one-sided case), which is what `test_densities_match_hand_written_references` pins; `Model.dlogp(vars=self._ordered_rvs)` and `Model.d2logp(vars=…, negate_output=False)` concatenate per-variable gradients in the order given, matching `self._offsets`, and `negate_output=False` is the sign `marginal_sd`'s `precision = -(H + Hᵀ)/2` expects; source free-variable order is explicitly restored after `remove_value_transforms` reorders it; the plausible-bound mapping takes `np.minimum`/`np.maximum` after the forward map, which is required because the upper-only interval map `v = log(b − x)` is decreasing; `pm.draw(..., random_seed=self._rng)` consumes the target's own generator and never NumPy's global state; the C-order reshape in both ArviZ exports is correct; and `to_torch` makes no draws.

---

## 2. Findings

### F1. Every default `PyMCTarget(model)` construction takes 4000 prior draws whose only use is a log warning
- Location: `pyvbmc/pymc/_target.py:664` (`check_location=start is None`), with `pyvbmc/pymc/_plausible.py:289–292` and `pyvbmc/pymc/_target.py:968–984`
- Category: defaults
- Proposed classification: possibly intentional
- Confidence: high (that the code does this); medium (that it is unintended)
- `laplace_box` is called with `check_location=start is None`, i.e. **true on the default construction path**. Inside `laplace_box`, `check_location` forces `get_draws()`, which calls `PyMCTarget._prior_draws()` → `pm.draw(variables, draws=_PRIOR_DRAWS=4000, random_seed=self._rng)`. In PyMC, `pm.draw` with `draws > 1` compiles one function and then calls it in a Python loop once per draw, so this is 4000 separate forward evaluations. The resulting `location_mask` is used **only** at `_target.py:724` (`plausible_info["location_outside_prior"]`) and `_target.py:1175–1179` (a `logging.warning`). It does not influence `x0`, `plb`, `pub`, `setup_cost` or anything else numerical. By contrast the *other* consumer of `get_draws()` — the curvature fallback at `_plausible.py:280–287` — does feed `plb`/`pub`, and is correctly gated on `np.any(curvature_mask)`. `_plausible.laplace_box` already supports skipping the draws (`test_laplace_box_skips_prior_draws_when_not_needed` passes `check_location=False`), so the machinery exists; the default adapter path never uses it.
- Consequence if real: a fixed, uncharged setup cost on every `PyMCTarget(model)` where the user does not supply `start`. It is invisible to `setup_budget` (which counts only log-density and Hessian units, so the *accounting* is correct — these are prior draws, not target evaluations), but it is wall-clock work proportional to the model's forward-sampling cost, paid to produce a diagnostic string. For a model whose prior involves expensive forward simulation it dominates construction. It also consumes `self._rng` even when a well-conditioned Hessian makes the prior irrelevant.
- Suggested reproduction: monkeypatch `PyMCTarget._prior_draws` to raise, then construct `PyMCTarget(models.scalar_model(rng)["model"])` (a model whose Hessian is PD at the mode, so `curvature == []`). The construction fails, showing the draws are taken regardless. `PyMCTarget(model, start={"mu": 0.0})` does not raise.
- Test adequacy: no. `test_scalar_laplace_box_and_exact_accounting` asserts `curvature == []` *and* `location_outside_prior == []` for the scalar model — it silently pays for 4000 draws and asserts the answer is empty. `test_laplace_box_skips_prior_draws_when_not_needed` tests the unit in isolation with a flag the adapter never passes on the default path.

### F2. A one-sided model value that rounds onto its support bound aborts the whole ArviZ export
- Location: `pyvbmc/pymc/_target.py:1093–1102` (`to_model_variables`), reached from `pyvbmc/pymc/_target.py:1252` (`to_arviz`)
- Category: control flow
- Proposed classification: possibly intentional
- Confidence: high (behavior); medium (that it matters in practice)
- For a variable with a kept transform, `to_model_variables` applies the compiled `backward` map and then rejects the *whole call* if any entry is non-finite or not strictly inside `self.support[name]`. For a lower-only interval transform the backward map is `b + exp(v)`; in float64 this returns exactly `b` as soon as `exp(v) < eps·|b|`, i.e. `v < log(eps·|b|) ≈ −36.7 + log|b|`. An upper-only transform (`b − exp(v)`) has the same threshold. `to_arviz` then raises `ValueError("Mapped model values for … must be finite and strictly inside its support.")` and (correctly) rewinds `vp.rng`, so the user gets no export at all rather than 999 usable draws out of 1000.
- Consequence if real: with e.g. `pm.TruncatedNormal("t", lower=1e6, …)` and a variational posterior on `v = log(t − 1e6)` with mean 0 and sd ≈ 5, a draw below `v ≈ −22.9` saturates — a ~4.6σ event, so roughly one export in 500 at `n_samples=1000` fails outright. With `lower = 1.0` the threshold is `v < −36.7`, effectively unreachable; exposure scales with the magnitude of the finite support bound. The `log` transform needs `v < −745` and is not a practical concern.
- Suggested reproduction: build `PyMCTarget` for `pm.TruncatedNormal("t", 0.0, 2.0, lower=1.0)`; `target.to_model_variables(np.array([[-40.0]]))` raises while `-30.0` does not. The boundary is `np.log(np.finfo(float).eps * lower)`.
- Test adequacy: partially. `test_model_mapping_rejects_nonfinite_or_boundary_transform_output` pins exactly this rejection for the `log` kind at ±1000, and `test_export_mapping_failure_restores_posterior_rng` pins the rng rewind. What no test covers is that a single tail draw out of `n_samples` makes the whole export unavailable, or the much lower saturation threshold of an interval transform with a large finite bound.

### F3. `UnsupportedModel` raised inside the mode search is relabelled as a mode-search failure
- Location: `pyvbmc/pymc/_target.py:608–612`, with `_target.py:34–42` (`_as_float64`) and `pyvbmc/pymc/_compat.py:16` (`class UnsupportedModel(ValueError)`)
- Category: control flow
- Proposed classification: suspected defect (minor)
- Confidence: high
- `value_and_grad` (line 580) calls `_as_float64` on the compiled density and gradient outputs; `_as_float64` raises `UnsupportedModel`, which is a `ValueError` subclass. The `except ValueError as exc` around `_plausible.search_mode` therefore catches it and re-raises a plain `ValueError(f"Mode search failed for model variables …: {exc}")`. The caller loses the `UnsupportedModel` type — which the class docstring lists as the signal for "a free variable, transform, support, or dtype is unsupported" and which `test_rejections.py` relies on — and the attribution to a dtype problem rather than an optimizer failure. The contrast is deliberate elsewhere: `_compatibility_error` produces an `ImportError`, which is not a `ValueError` and so propagates intact, and `_snapshot.py:151–152` has an explicit `except UnsupportedModel: raise` placed before its broad `except (…, ValueError)` for exactly this reason.
- Consequence if real: diagnostic quality only. A float32 gradient from a model that passed the construction-time dtype checks is reported as "Mode search failed for model variables x=…: … is float32 …". It cannot change a numerical result.
- Suggested reproduction: monkeypatch `pyvbmc.pymc._target._as_float64` to raise `UnsupportedModel` on its second call and construct `PyMCTarget(scalar_model(...)["model"])`; the message is the mode-search wrapper rather than the dtype message.
- Test adequacy: no. No test drives an `UnsupportedModel` through `search_mode`; `test_nonfinite_mode_names_model_variable` exercises the wrapper only for the genuine "no finite log density" case.

### F4. The no-gradient warning claims the initial point was used even when the user supplied `start`
- Location: `pyvbmc/pymc/_target.py:1187–1191`, against `_target.py:619–621` and `_target.py:667–672`
- Category: control flow (diagnostics)
- Proposed classification: suspected defect (cosmetic)
- Confidence: high
- When `start` is supplied and `plausible_bounds` is not, `need_derivatives` is true (line 532) and the gradient is compiled; if the model has no usable gradient, `has_gradient` is false, so `route == "prior"` while `start_kind == "user"` (line 621). `_report_setup_warnings` then logs "The model gradient is unavailable; using the initial point and prior-quantile plausible bounds." The plausible-bounds half is accurate; `x0` is the user's `start` normalized by `_normalize_explicit_start`, not the initial point. The message keys on `info["route"]` although `info["start"]` is the field that distinguishes the two cases, and `plausible_info` already records it.
- Consequence if real: a misleading log line; no numerical effect.
- Suggested reproduction: `PyMCTarget(no_gradient_model(), start={"x": 0.5})` with `caplog`; compare the message with `plausible_info["start"] == "user"` and `x0 == 0.5`.
- Test adequacy: no. `test_setup_diagnostic_routes_and_warnings` triggers the warning only for `no_gradient` constructed without a `start`, where the wording happens to be right.

### F5. Dead branch and an inaccurate rejection message for an interval transform with no finite bound
- Location: `pyvbmc/pymc/_target.py:384` (`unbounded = ~finite_lower & ~finite_upper`, never read) and `_target.py:388–392`
- Category: control flow
- Proposed classification: possibly intentional
- Confidence: high (the variable is unused); low (that the message path is reachable)
- `unbounded` is computed and never used. If an interval transform ever produced `(-inf, +inf)` on all coordinates, `two_sided` and `one_sided` would both be all-false and the code would raise `UnsupportedModel("… its coordinates mix one-sided and other bounds.")`, describing a different condition than the one that occurred; `unbounded` looks like the remains of a branch meant to give that case its own message. In practice PyMC's `IntervalTransform.backward` itself raises `ValueError("Both edges of IntervalTransform cannot be None")`, and `_validate_source_transform_calls` (lines 90–107) calls `backward` earlier while catching only `TypeError` and `AttributeError`, so such a model surfaces as a bare PyMC `ValueError` first.
- Consequence if real: none numerically; at most a confusing message on an already-rejected model.
- Suggested reproduction: attach `pm.distributions.transforms.Interval(bounds_fn=lambda *a: (None, None))` to a `pm.Normal` and observe which of the three errors fires first.
- Test adequacy: not covered; `test_rejections.py::mixed` covers only the genuine mixed case.

---

## 3. Test adequacy notes

- **The one-point closed-form check is pinned as a one-point check.** `_target.py:480–496` validates each kept transform by comparing `backward(probe)` against `_compat.closed_form` at a *single* point (the initial value) and round-tripping it. `test_guards.py::test_transform_check_uses_partial_initial_point` asserts `len(probes) == 1` and the exact probe value — it pins the implementation's sampling of the transform rather than the property ("the compiled map equals the closed form on its domain"). Adequate as a version guard against a renamed or reshaped PyMC API; it would not detect a transform that agrees with the closed form only at the initial point. `test_flatten_and_model_coordinate_round_trips` partly compensates, but again only at `x0`.
- **`_assert_accounting` re-derives the invariant from the fields it checks.** `test_target_setup.py:40–52` asserts `setup_cost == n_evaluations`, `setup_cost == n_target_calls + hessian_cost` and `hessian_cost == D * n_hessian_calls`. Those three are exactly how `plausible_info` is assembled at `_target.py:717–720` and how `setup_cost` is defined at `_target.py:732–734`, so the block is self-consistent by construction. The only independent content is `setup_cost <= setup_budget` and the re-evaluation loop over `(X, y)`. The route-specific budget arithmetic (`minimum`, `max_search`) is checked only through three `pytest.raises(match="at least N")` cases.
- **The two-sided branches of `_compat.closed_form` are unreachable from the adapter.** Two-sided intervals and `logodds` always have their transform removed (`_target.py:386`), so `self._maps` only ever holds `log` and one-sided intervals. `test_closed_form_backward_maps` and `test_closed_form_handles_mixed_one_sided_interval_coordinates` exercise the two-sided formula directly, so the helper is tested, but nothing pins that `a + (b−a)·expit(v)` matches PyMC's `sigmoid(v)·b + (1−sigmoid(v))·a`; a divergence would be invisible because the path is dead. Fine today, a gap if a future change ever keeps a two-sided transform.
- **`test_export_layout_values_and_rng_stream` and `test_fixed_flat_coordinate_oracles_and_export` drive `to_arviz` with a `FixedPosterior` stub** whose `sample` returns a fixed array. Good for pinning the layout, but it means the interaction between real `vp.sample(orig_flag=True)` output ranges and `to_model_variables`' strict-interior rejection (F2) is exercised only by `test_export_layout_values_and_rng_stream`, whose `vp.sigma = 0.2` keeps every draw far from saturation.
- **The Torch export tests are specification-level and thorough.** `test_vp_torch.py` compares `dist.log_prob` against `vp.log_pdf` for every transform type, in both coordinate systems, with `torch.autograd.gradcheck`, and separately checks the inverse map and its derivative against `scipy.stats.t`/`ndtri` at the branch switches and in both tails. I found nothing in the Torch export that these do not already constrain, and my independent NumPy transcription agreed with the NumPy `ParameterTransformer` and with SciPy everywhere I probed.
