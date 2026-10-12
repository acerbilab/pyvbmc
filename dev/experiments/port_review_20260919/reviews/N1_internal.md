# N1 — S-VBMC, internal-correctness track

Raw reviewer report, wave 0 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-19. One Opus agent,
fresh context, read-only on the repositories; its check scripts ran in the
session scratchpad and are not retained. The text below is the agent's
final message, unedited.

---

## 1. Coverage

**Read completely**

- `pyvbmc\svbmc\svbmc.py` (951 lines)
- `pyvbmc\svbmc\_entropy.py`
- `pyvbmc\svbmc\_jacobian.py`
- `pyvbmc\svbmc\_elbo_shrinkage.py`
- `pyvbmc\svbmc\utils.py`
- `pyvbmc\svbmc\__init__.py`, `_runtime_tips.py`, `_tip_catalog.py` (the last two for context only)
- `papers/silvestrin2025stacking_main.md` §3 (Eq. 10–17, Algorithm 1), §4.1 (filtering criteria, S = 20 / S = 100, lr = 0.1), §5.2 (the capped ELBO, `E_median`, `I_median`)
- `papers/silvestrin2025stacking_appendix.md` §A.2 (Eq. A.13–A.20, the change-of-variables contract) and §A.3.1 (the "posterior-only" variant)
- `pyvbmc\parameter_transformer\parameter_transformer.py` (all three bounded transforms, the forward/inverse/`log_abs_det_jacobian` triple, the rotoscale path)
- All test modules under `pyvbmc\testing\svbmc\` plus `_fixtures.py`

**Read in part**

- `pyvbmc\variational_posterior\variational_posterior.py`: `sample`, `__deepcopy__`, the `rng` property, `stats` initialization
- `pyvbmc\vbmc\variational_optimization.py`: `_gp_log_joint` (the definition of `I_sk` and `J_sjk`, lines 1480–1663) and the block that writes `vp.stats` (lines 311–406)
- `pyvbmc\whitening\whitening.py` lines 120–220 (how `R_mat`/`scale`/`mu`/`delta` are installed; confirms the GP target in transformed space carries `+ log|det J|`)
- `pyvbmc\rng.py`

**Not reached**

- `pyvbmc\entropy\` (the VBMC entropy estimators — S-VBMC does not call them)
- `SVBMC.plot` / `utils.overlay_corner_plot` beyond a read (plotting only)
- Anything under `dev/` (excluded by the task)

**Small checks run** (`torch` is absent from the default interpreter, so the checks ran under the `compstats` conda environment, which has pyvbmc's dependencies and torch 2.5.0)

1. `check_jacobian2.py` — `expected_log_jacobian` against 2e6-draw Monte Carlo for probit, logit and student4, with and without a whitening rotation + scale, and for a mixed bounded/unbounded box. Agreement within 2 standard errors in every cell (max |diff| 2.7e-3, max z-score 2.2).
2. `check_entropy_matrix.py` — `component_log_densities` against `VariationalPosterior.pdf` with one-hot weights, on two runs (one warped, different `K`): max |diff| 1.4e-14. The stacked entropy estimator on a single run against a plain Monte Carlo entropy: 1.6996 vs 1.6974.
3. `check_elbo_endtoend.py` — a full pipeline check on a normalized 1-D target over a box, two runs with different transforms (probit / logit), different centerings and a `scale` on one. `stats["I_sk"]` was set to the exact transformed-space expected log joint by grid quadrature. `I_corrected` reproduced the original-space expected log joint to 1.4e-13, and `stacked_ELBO` reproduced the grid ELBO (= −KL) to 7e-4 at 40000 entropy draws per component. This pins the sign and magnitude of the Jacobian correction, the `I_sk` → `I_corrected` → `G` chain and the entropy estimator together.
4. `check_bounds_mismatch.py`, `check_misc.py`, `check_balance.py` — the reproductions for F1–F4 below.
5. `python -m pytest pyvbmc/testing/svbmc/test_elbo_shrinkage.py::test_unequal_original_weights_drive_between_run_levels -q` — passes (environment sanity).

**Checked and found correct** (no finding): the stacked ELBO (Eq. 11) and its softmax parameterization; the logit initialization (Eq. 16) including the max-subtraction and the `log w` term; the "posterior-only" reduction to per-run weights (Eq. A.21) — the logits `ω_m + log w_mk` do reproduce `w̃_mk = ω̃_m w_mk`; the entropy estimator (Eq. 12) and its weight-gradient path (the draws do not depend on `w̃`, so no reparameterization term is needed, and the extra `w / w.sum()` in `stacked_ELBO` leaves the softmax gradient unchanged because `(I − w1ᵀ)(diag(w) − wwᵀ) = diag(w) − wwᵀ`); the run/component/hyperparameter-sample axes and the `offsets` indexing in `__init__`, `optimize` and `sample`; the cap heuristic (`I_median` over all components, `E_median` over runs with each run's own weights, §5.2); `_expected_log_joint_variance` as the law of total variance summed over independent runs, mirroring VBMC's own `varG`; `_run_covariance` and `_moments` (the noise term `(tr Σ − 1ᵀΣ1/n)/(n−1)` is the exact inflation of the sample variance) and the empirical-Bayes solve; the defaults (`s_max = √5`, `n_samples = 20`, `n_samples_final = 100`, `lr = 0.1`, `version = "all-weights"`) against the paper; the generator plumbing (construction consumes no draws, `sample` deep-copies each VP and rebinds `rng` to the object's generator so the inputs' streams never advance, `_balanced_counts` cannot ask `rng.choice` for more distinct entries than there are nonzero fractional parts); `K = 1` runs, `M = 1` stacks, runs with different `K`, and zero-weight runs in `sample` and `_expected_log_joint_variance`.

---

## 2. Findings

### F1. Runs whose original-space boxes differ are accepted and produce a silent NaN ELBO
- Location: `pyvbmc\svbmc\svbmc.py:57` (`_validate_posteriors`), effect at `pyvbmc\svbmc\_entropy.py:91`
- Category: cross-module
- Proposed classification: suspected defect
- Confidence: high

`_validate_posteriors` checks the dimension, the component shapes, the presence of a `parameter_transformer` and the four statistics, but never compares the transformers of different runs. The stacked posterior is only defined in the common original space, and `component_log_densities` realizes that by evaluating *every* run's forward transform at *every* run's draws (`U = np.asarray(transform(X_orig), ...)`). If run A's box is wider than run B's, some of A's draws fall outside B's `(lb_orig, ub_orig)`; `_to_unit_interval` then returns `z < 0` or `z > 1`, and `_probit`/`_logit`/`_student4` return NaN there. The NaN propagates through `logsumexp` into the entropy, the ELBO, the Adam gradients and the returned weights, with no exception and only a NumPy `RuntimeWarning: invalid value encountered in log`.

The class docstring disclaims only that "the object structure cannot verify that the runs describe the same target"; the bounds and transform types *are* on the transformer and are verifiable.

Consequence if real: `optimize()` returns `elbo = nan`, `entropy = nan` and `w = [nan, …]` with no error. It triggers whenever the stacked runs were launched with different `LB`/`UB` (a natural mistake when re-running an analysis with a widened prior box, or when mixing runs from two scripts), and it is total — every reported number is NaN.

Suggested reproduction (`check_bounds_mismatch.py`, `check_misc.py` part 1): build two 1-D posteriors on `(-5, 5)` and `(-2, 2)` with otherwise valid `stats`, then `s = SVBMC([a, b], M_min=2, seed=0); s.optimize(n_samples=3, max_steps=20, n_samples_final=4)`. Observed: `elbo nan entropy nan w [nan nan nan nan]`. With identical boxes the same script gives finite values. `component_log_densities([a, b], 5, rng)` alone already returns 8 NaNs out of 80 entries.

Test adequacy: no. `test_svbmc_filters.py` covers dimension mismatch (`test_mismatched_dimension_raises`) and malformed statistics, and `test_svbmc.py` (`test_bounded_runs_have_different_transforms`, `test_warped_and_unwarped_runs_stack`) covers runs that differ in centering and rotation — but all fixtures in a group share one box, so no test stacks runs with different `lb_orig`/`ub_orig`.

---

### F2. A second `optimize()` initializes from the first solution, not from the runs' own weights
- Location: `pyvbmc\svbmc\svbmc.py:605` (`w_init = torch.as_tensor(self.w, …)`) together with `pyvbmc\svbmc\svbmc.py:747` (`self.w = w.detach()…`)
- Category: state/caching
- Proposed classification: suspected defect
- Confidence: high

`maximize_ELBO` builds its logits from `self.w` — `log_w = torch.log(w_init)`, then `w_logits_init = log_w + broadcasted_elbos` for `"all-weights"`, and `repeat_interleave(omega_logits) + log_w` for `"posterior-only"`. `optimize()` overwrites `self.w` with the optimized weights at line 747. `self.w` is therefore both the documented *input* (`maximize_ELBO`: "The weights are the softmax of unconstrained logits, initialized from the runs' own weights and ELBOs so that better runs start heavier"; Eq. 16 of the paper, `a_mk = log w_mk + ELBO(φ_m) − max(...)`) and the *output* state. After one `optimize()` the documented initialization no longer holds: the logits start at the previous optimum, shifted again by the per-run ELBOs.

`"posterior-only"` is affected in a second way: its derivation assumes `log_w` carries the runs' *internal* weights with equal mass per run, which is what makes `w̃_mk = ω̃_m w_mk`. After a first `optimize()` the per-run masses are no longer equal, so the second call optimizes per-run multipliers on top of the already reweighted stack. (`"ns"` is unaffected — it rebuilds from `self._naive_weights`.)

Consequence if real: `optimize()` is not idempotent, and each repeated call pushes the weights further in the direction of the first solution (the per-run ELBO term is re-applied every time). It triggers whenever a user re-runs `optimize()` on the same object, e.g. to try more steps or a larger `n_samples`.

Suggested reproduction (`check_misc.py` part 2): two 3-component runs, `version="posterior-only"`, 60 steps each.
```
initial per-run mass: [0.5, 0.5]
after 1st: [0.03879 0.03879 0.03879 0.29454 0.29454 0.29454]
after 2nd: [0.02211 0.02211 0.02211 0.31122 0.31122 0.31122]
```
and for `"all-weights"`, `[0.0089 0.00787 0.06822 0.88762 0.01299 0.0144]` becomes `[0.00385 0.0034 0.0295 0.94367 0.00927 0.01031]`.

Test adequacy: no. `test_repeated_optimization_uses_current_selected_weights` (`test_elbo_reporting.py:333`) is the only repeated-`optimize` test, and it monkeypatches `maximize_ELBO` to return a fixed weight vector, so the initialization path is never executed. `test_naive_mode_reuses_final_evaluation_and_original_weights` sets `stacked.w` by hand before `optimize(version="ns")` and confirms the "ns" reset — the one mode that does *not* read `self.w`.

---

### F3. The two-level shrinkage compounds the within-run and between-run corrections
- Location: `pyvbmc\svbmc\_elbo_shrinkage.py:159` (`within + (shifted[m] - levels[m])`)
- Category: formula/gradient
- Proposed classification: unsure (possibly intentional)
- Confidence: medium

The estimator first shrinks each run's component estimates toward the run's unweighted component mean (`within`, lines 100–115), then shrinks the run *levels* `levels[m] = own @ I` toward the between-run mean (`shifted`, lines 143–156), and finally forms `corrected = within + (shifted[m] − levels[m])`. The shift is the difference between the shrunk level and the **raw** level, but it is added to the **already within-shrunk** components, whose own-weighted level `own @ within` generally differs from `own @ I`. The resulting run level is `own @ within + shifted[m] − own @ I`, which equals `shifted[m]` only when the within-run stage happens to preserve the own-weighted level. The `elbo_details` docstring (`svbmc.py:263`) describes the estimate as "Shrinkage adjusts component estimates within each run and then shifts run levels", which reads as if the second stage sets the run level to `shifted[m]`.

The within-stage preserves the own-weighted level in all the configurations the tests use (symmetric two-component runs with uniform `own`), which is why the discrepancy is invisible there.

Consequence if real: `elbo_details["shrunk_two_level"]` is biased by the within-run level change of each run, weighted by that run's selected mass. In the reproduction below the two runs' levels miss their shrunk targets by 0.16 and 0.30 nats. `shrunk_two_level` is a diagnostic, not the headline `elbo`, so no reported posterior or headline evidence moves.

Suggested reproduction (`check_misc.py` part 3): two runs with asymmetric component estimates and non-uniform own weights, `I0 = [0, 1, 9]`, `C0 = diag(1, 4, 0.25)`, `own0 = [0.7, 0.2, 0.1]`; `I1 = [20, 21, 22]`, `C1 = I`, `own1 = [0.2, 0.3, 0.5]`:
```
target shifted levels : [ 1.13228  21.281176]
levels of corrected   : [ 1.295235 20.981176]
own@within vs own@I   : [1.262955 1.1] [21.0 21.3]
```
The smallest decisive check is the last line: `own @ within != own @ I`, so the two stages do not compose to `shifted[m]`.

Test adequacy: no. `test_elbo_shrinkage.py::test_unequal_original_weights_drive_between_run_levels` is the only test that exercises both stages, and (a) it recomputes the expectation with the very formula under review (`means[m] + shifted[m] - levels[m]`), and (b) in its data (`I = [0, 2]` with `C = ones`, `I = [10, 14]` with `C = 4·ones`) the within-run solve returns the inputs unchanged, so the composition is never exercised.

---

### F4. `sample(balance_flag=True)` does not hold its per-component accuracy claim
- Location: `pyvbmc\svbmc\svbmc.py:846` (docstring) and `pyvbmc\svbmc\svbmc.py:889` (the delegation)
- Category: control flow
- Proposed classification: suspected defect (documentation contract)
- Confidence: high

The docstring states: "With `balance_flag=True` the draws are split across runs, and across components within a run, in proportion to the weights, **every count within one draw of its exact share**." The run split (`_balanced_counts`) does guarantee that, but the within-run split is delegated to `VariationalPosterior.sample(balance_flag=True)`, which allocates its remainder with `rng.choice(..., p=w_extra)` **with replacement** (`variational_posterior.py:658`) — so a single component can receive two extra draws while another receives none. The two roundings also compose: run `m` already deviates from `n·ω_m` by up to one draw before its components are split.

Consequence if real: a user relying on the stated bound (for instance to build a deterministic stratified estimator) gets counts off by 2–3 draws. Small-`n` plots and expectations are slightly noisier than advertised; nothing else changes.

Suggested reproduction (`check_balance.py`): two runs (K = 3 and K = 4) with far-apart, tight components so each draw identifies its component; hand-set `w = [0.20, 0.13, 0.07, 0.25, 0.15, 0.12, 0.08]`; 200 balanced draws at each `n`:
```
n=   10  worst |count - n*w| = 1.800
n=   37  worst |count - n*w| = 2.750
n=  100  worst |count - n*w| = 0.000
n= 1000  worst |count - n*w| = 0.000
```

Test adequacy: no. `test_balanced_counts_are_exact_and_proportional` asserts the bound for `_balanced_counts` alone (the run split). `test_sample_returns_exactly_n_rows` only checks the row count and finiteness; no test looks at per-component counts of `SVBMC.sample`.

---

### F5. `stacked_entropy`'s Returns section attributes the Jacobian subtraction to `stacked_ELBO`
- Location: `pyvbmc\svbmc\svbmc.py:467`
- Category: cross-module (documented contract)
- Proposed classification: suspected defect (documentation)
- Confidence: high

The docstring says the returned `J_corrections` are "Per-component expected log-Jacobian, computed deterministically at construction, which :meth:`stacked_ELBO` subtracts from the stored expected log-joints to express them in original space." `stacked_ELBO` does no such subtraction: it reads `self.I_corrected`, which `__init__` already formed as `self.I - self._jacobian_corrections` (line 428). The returned array is informational only.

Consequence if real: no numerical effect. A caller building their own objective from `stacked_entropy` plus `self.I_corrected` and following the docstring would subtract the corrections a second time, shifting the expected log joint by the (typically O(1)–O(10) nats) Jacobian term.

Suggested reproduction: read `stacked_ELBO` (lines 552–558) — `I_corr_t` is built from `self.I_corrected`, and the value returned by `stacked_entropy` is discarded at line 552.

Test adequacy: partially. `test_constructor_caches_corrections_without_rng_draws` and `test_stacked_entropy_matches_the_literal_reference` check that the returned array equals `_jacobian_corrections` and is a copy, but no test states where the subtraction happens.

---

## 3. Test adequacy notes

- **`test_svbmc_references.py`** replays the whole pipeline and compares against stored numbers at `rtol=1e-8`. It is an exact mirror of the current implementation: it would catch numerical drift but cannot distinguish a correct estimator from an incorrect one that has not changed.
- **`test_entropy.py`** compares `component_log_densities` and `_stacked_entropy` against "a literal per-component transcription of the same estimator". That is a refactor guard for the vectorization, not a check of the estimator against the paper. (The estimator itself is correct — `check_elbo_endtoend.py` reproduces the grid ELBO — but nothing in the suite establishes that.)
- **`test_elbo_shrinkage.py::test_unequal_original_weights_drive_between_run_levels`** computes its expected value with the same `means[m] + shifted[m] - levels[m]` expression the code uses, and its data make the within-run solve an identity, so the two shrinkage stages are never tested in composition (F3).
- **`test_elbo_reporting.py`** leans heavily on `_fixed_final_evaluation`, which monkeypatches both `maximize_ELBO` and `_stacked_entropy`. That isolates the reporting algebra cleanly, but it means the tests named for optimization behaviour (`test_repeated_optimization_uses_current_selected_weights`, `test_optimize_maps_show_tips_and_info_logging_to_scheduler`) never execute the optimizer (F2).
- **`test_jacobian.py`** is the strongest module in the slice: it checks the analytic cases against an independent closed form, the scalar quadrature against `scipy.integrate.quad`, and the warped mixed case against seeded Monte Carlo, and it exercises the non-convergence branches with monkeypatched rules. My independent Monte Carlo check agrees.
- **Gaps**: no test stacks runs with mismatched `lb_orig`/`ub_orig` (F1); no test inspects per-component counts from `SVBMC.sample(balance_flag=True)` (F4); no test compares the stacked ELBO against an externally computed value on a problem where it is known.
