# P3 — acquisition functions, internal-correctness track

Raw reviewer report, wave 4 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, fresh
context, read-only on the repositories, reading the code at `f873556`; it
did not read the MATLAB source. Its brief carried a first question on
uncertainty level 1, which the report answers before its findings. Its check
scripts ran in the session scratchpad and are kept only on the machine that
ran them (`dev/scripts/runs/LOCAL.md`). The text below is the agent's final
message, unedited.

---

# P3 — acquisition functions (internal-correctness track)

## 1. Coverage

**Read completely.** Every file of `pyvbmc/acquisition_functions/`:
`abstract_acq_fcn.py`, `acq_fcn.py`, `acq_fcn_log.py`, `acq_fcn_vanilla.py`,
`acq_fcn_noisy.py`, `acq_fcn_viqr.py`, `acq_fcn_imiqr.py`, `utilities.py`,
`__init__.py`, `README.md`.

Read completely for the interface and the state the caller prepares:
`pyvbmc/vbmc/active_sample.py`, `pyvbmc/vbmc/active_importance_sampling.py`,
`pyvbmc/vbmc/gaussian_process_train.py`,
`pyvbmc/function_logger/function_logger.py`,
`gpyreg/gpyreg/noise_functions.py`, and in `gpyreg/gpyreg/gaussian_process.py`
the `predict` and `__core_computation` bodies (the `Posterior` construction,
`sW`, `sl`, `sn2_mult`, the `L_chol` branches). Read the relevant parts of
`pyvbmc/vbmc/vbmc.py` (`_setup_vbmc`'s noise/bounds/regularization state,
`_initialize_precomputed_evaluations`).

Read completely: `pyvbmc/testing/acquisition_functions/` (all 12 modules and
the three helpers), the oracle state/oracle helpers
(`pyvbmc/testing/oracles/_state.py`, `_oracles.py`), the acquisition sections
of `papers/acerbi2020variational_main.md` (Eqs. 5, 6, 8, 9) and the
uncertainty-sampling equations of the 2018 paper. Read the known-differences
sheet's P2, P3, P4, GP-layer and "Settled non-differences" sections.

Skimmed: the rest of `papers/`, `pyvbmc/vbmc/README.md`, the `.ini` option
files (searched for the options the slice reads).

**Not reached.** The MATLAB sources (other reviewer's track). I did not run
`VBMC.optimize()`, the test suites, or the oracle regeneration.

**Checks run** (scripts and logs under
`…/scratchpad/wave4_P3_internal`; `pyvbmc.__file__` and `gpyreg.__file__`
confirmed as the two checkouts; `OMP/OPENBLAS/MKL_NUM_THREADS=1`):

1. `t_utilities.py` — `string_to_acq` on seven argument forms.
2. `t_state_info.py` — the uncertainty-handling level, noise switches, `S`
   and `n_evals` of all 8 oracle fixtures.
3. `t_level1.py` — hand-built loggers and GPs at levels 0, 1 and 2, with
   repeated evaluations; `sn2_new` vs the GP's own training-row noise vs the
   noise of one fresh observation; `predict(add_noise=True)` minus latent.
4. `t_level1_fit.py` — a real level-1 `train_gp` fit (26 points, D=2, 8
   hyperparameter samples) to see what the noise multiplier fits to and how
   much `sn2_new` varies across samples.
5. `t_lookahead.py` — VIQR's `tau2`/`s_pred` on the noisy oracle fixture
   against a brute-force refit of the GP with the hypothetical observation
   appended (8 hyperparameter samples × 6 candidates × 100 importance
   points): max relative difference **4.4e-12**.
6. `t_imiqr_and_reuse.py` — IMIQR's `C` vs VIQR's `C` on the same state
   (1.1e-9 relative); `sW` constant across training points; the
   `return_cross_covariance` path vs `cdist` (bit-identical, and the two
   acquisition outputs bit-identical).
7. `t_logsum.py` — `_log_viqr_sum` against a 60-digit `decimal` reference on
   seven rows (ordinary, tiny, exact zeros, at the overflow guard, above it);
   `_log_iqr_reduction` against the naive `sinh(us_a) − sinh(us_pred)` form
   (4.5e-16 relative).
8. `t_real2int.py`, `t_quantile.py`, `t_misc.py`, `t_islogfull.py` — the
   small behavioural checks cited in the findings.

## 2. First question — where the observation noise of a candidate comes from, at each level

**Answer: yes at all three levels, with two caveats (F4, F5).** At level 1
the noise that `_estimate_observation_noise`, `gp.temporary_data["sn2_new"]`,
the VIQR/IMIQR denominators and `AcqFcnNoisy` use is exactly the variance the
GP's own noise model assigns to **one new single observation** at that point.
The level-1 model is homoskedastic by construction, so the value is the same
at every training point and the nearest-neighbour lookup is a constant.

**What each level records and fits** (established from the code, not the
brief):

| | level 0 | level 1 | level 2 |
|---|---|---|---|
| `optim_state["gp_noise_fun"]` (`vbmc.py:1094-1102`) | `[1,0,0]` | `[1,2,0]` | `[1,1,0]` |
| `GaussianNoise` switches | `constant_add` | `constant_add`, `user_provided_add`, `scale_user_provided` | `constant_add`, `user_provided_add` |
| noise hyperparameters | `noise_log_scale` | `noise_log_scale`, `noise_provided_log_multiplier` | `noise_log_scale` |
| `sn2(x_i)` (`noise_functions.py:248-266`) | `exp(2h₀)` | `exp(2h₀) + exp(h₁)·s2ᵢ` | `exp(2h₀) + s2ᵢ` |
| `FunctionLogger.S` | absent (`noise_flag` False) | **always 1** for a first observation | the target's SD |
| `s2_train` fed to `gp.fit` (`_get_training_data`) | `None` | `S²` (= `1/n` after `n` pooled repeats) | `S²` (= 1/Σ 1/sᵢ² after repeats) |

Every route into the logger gives `S = 1` at level 1: `__call__` sets
`f_sd = 1` (`function_logger.py:302`), `add` defaults it to 1
(`:547`), `batch_call` fills `np.ones(n_rows)`
(`_validate_vectorized_target_output`, `:148-152`), and
`_initialize_precomputed_evaluations` **refuses** a supplied `y_sd` unless
`specify_target_noise` is set (`vbmc.py:726-731`). So no level-1 row can
carry an `S` other than `1/√n`.

**The candidate's noise.** `active_sample.py:328-351`, once per
active-sampling step, for each hyperparameter sample `s`:

```python
hyp_noise = gp.posteriors[s].hyp[cov_N : cov_N + noise_N]
s2 = (function_logger.S[X_flag] ** 2) * function_logger.n_evals[X_flag]   # or None at level 0
sn2new[:, s] = gp.noise.compute(hyp_noise, gp.X, gp.y, s2)
gp.temporary_data["sn2_new"] = sn2new.mean(1)
```

The `* n_evals` **undoes the pooling**: the GP is fitted with `s2ᵢ = Sᵢ²`
(the variance of the *pooled* value at row `i`), while `sn2_new` is evaluated
at `Sᵢ²·nᵢ`, the variance of one single evaluation there. So:

- **level 0**: `s2 = None` → `sn2_new = exp(2h₀)`, the GP's only noise term.
  That is what the GP would give a new observation. Consistent.
- **level 1**: `Sᵢ²·nᵢ = 1` exactly (verified numerically for rows with 1, 2
  and 3 evaluations), so `sn2_new = exp(2h₀) + exp(h₁)`, which is the GP's
  noise for a fresh single observation anywhere. In the real level-1 fit the
  GP's row noise for a twice-evaluated point was 1.007 while `sn2_new` was
  1.976 — the correct factor of two. Consistent.
- **level 2**: `Sᵢ²·nᵢ` is the harmonic mean of the individual variances at
  that input, i.e. the target's own per-observation variance there;
  `sn2_new = exp(2h₀) + that`. Consistent.

`_estimate_observation_noise` (`abstract_acq_fcn.py:317-350`) then picks the
nearest training point in units of the geometric-mean length scale
(`_sq_dist` on `Xs/gp_length_scale` and `gp.temporary_data["X_rescaled"]`)
and returns its `sn2_new`. That is the right quantity for the formulas:
paper Eq. 5 puts `σ²_obs(θ⋆)` — the noise of the *hypothetical new
observation at the candidate* — in the denominator
`C_Ξ(θ⋆,θ⋆) + σ²_obs(θ⋆)`, and `y_s2 = f_s2 + sn2` in
`acq_fcn_viqr.py:301` / `acq_fcn_imiqr.py:82` is exactly that; Eq. 6's
`s²/(s²+σ²_obs)` is what `AcqFcnNoisy` writes as `1 − sn2/(var_tot+sn2)`.
The brute-force refit check (check 5) confirms the whole look-ahead variance
to 4e-12 relative.

Two caveats, reported as F4 and F5: the per-sample noises are **averaged
across hyperparameter samples** before being combined with per-sample `f_s2`
and per-sample GP covariances, and `sn2_mult` (gpyreg's Cholesky-retry
inflation, which `predict(add_noise=True)` does apply) is omitted. A third,
F3, concerns `is_log_full`, the one place in the slice that asks gpyreg for a
noise-inclusive prediction and does not get one.

## 3. Findings

### F1. `_real2int` cannot take the 1-D array `active_sample.py:624` gives it; any run with `integer_vars` raises
- Location: `pyvbmc/acquisition_functions/abstract_acq_fcn.py:279-285`
  (the failing expression is `X_temp[:, integer_vars]` at `:281`); call site
  `pyvbmc/vbmc/active_sample.py:624-628`; MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high
- `_real2int` is written for a 2-D `X`: it calls
  `parameter_transformer.inverse(X)` and then indexes the result and `X` with
  `[:, integer_vars]`. `ParameterTransformer.inverse` preserves the input's
  dimensionality (the `handle_0D_1D_input` decorator), so for a 1-D input of
  shape `(D,)` `X_temp` comes back with shape `(D,)` and
  `X_temp[:, integer_vars]` raises
  `IndexError: too many indices for array: array is 1-dimensional, but 2
  were indexed`. `AbstractAcqFcn.__call__` always promotes to 2-D before
  calling it (`:81-87`), and `active_sample.py:383` passes the 2-D
  `X_search`, so the only 1-D caller is
  `X_acq[0, :] = AbstractAcqFcn._real2int(xsearch_optim, …)` after a local
  search. `xsearch_optim` is 1-D in all three branches: `res[:2]` from
  `cma.fmin` (`:582`), `np.atleast_1d(res.x)` from `minimize_scalar`
  (`:607`), `res.x` from Nelder-Mead (`:619`). The guard
  `if np.any(integer_vars)` is why nothing happens by default: with the
  default all-False mask the 1-D array is returned untouched.
- Consequence if real: a run that sets `integer_vars` raises `IndexError`
  out of `optimize()` at the first active-sampling step in which the local
  search improves on the sieve's best candidate — i.e. essentially the first
  active-sampling step of the first post-initial-design iteration. The
  exception is not inside the `try` that `_log_search_failure` guards, so it
  is not downgraded to a warning. Integer variables are unusable.
- Suggested reproduction (run, result above):
  `AbstractAcqFcn._real2int(np.array([0.31, 0.44]), pt, np.array([True, False]))`
  raises `IndexError`; the 2-D form of the same call returns the snapped
  array. A full reproduction would be one `VBMC(..., options={"integer_vars":
  [0]})` run, which I did not do.
- Test adequacy: no. `test_abstract_acquisition_function.py::test_real2int`
  calls `_real2int` only with `X = np.ones((10, D)) * 0.5`, and its four
  "assertions" are bare `np.all(...)` expressions with no `assert`
  (`:471-473`, `:477`), so that test cannot fail at all. No test anywhere in
  `pyvbmc/testing/` runs `optimize()` or `active_sample` with a non-empty
  `integer_vars` (the only non-`None` value in the whole suite is
  `test_viqr_kernel_reuse.py:253`, which goes through `__call__`).

### F2. `string_to_acq` mangles the second and later keyword arguments
- Location: `pyvbmc/acquisition_functions/utilities.py:17-22`; MATLAB: "not
  read (internal track)"
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- The parser splits the argument string on `","` and then calls
  `arg.rstrip().split("=")`. `rstrip()` removes trailing whitespace only, so
  every argument after the first keeps the space that follows the comma and
  the keyword key becomes `" loss"` instead of `"loss"`. `acq_fcn(*args,
  **kwargs)` then raises `TypeError: AcqFcnVIQR.__init__() got an unexpected
  keyword argument ' loss'`. A leading space is harmless for positional
  arguments because `literal_eval` ignores surrounding whitespace, which is
  why `"AcqFcnVIQR(0.9, 'iqr')"` works and
  `"AcqFcnVIQR(quantile=0.9, loss='iqr')"` does not.
- Consequence if real: `options={"search_acq_fcn": ["AcqFcnVIQR(quantile=0.9,
  loss='iqr_reduction')"]}` — the string form documented by the
  `search_acq_fcn` option and exercised by the tests — raises inside the
  active-sampling loop, after the run has already spent its initial design.
  Any acquisition taking two or more keyword arguments is unreachable by
  name. A value containing `","` or `"="` would be mis-split as well.
- Suggested reproduction (run): `t_utilities.py` above; six forms pass,
  `"AcqFcnVIQR(quantile=0.9, loss='iqr')"` raises `TypeError`.
- Test adequacy: no. Every `string_to_acq` case in the suite passes at most
  one keyword argument (`test_acq_fcn_viqr.py:48`,
  `test_acq_fcn_imiqr.py:49`, `test_acq_fcn_viqr_losses.py:196`), or two
  positionals (`:199`).

### F3. `is_log_full` asks for a noise-inclusive prediction; every other consumer of `is_log_added` is given the latent variance, and at a noisy level `add_noise=True` does not add the observation noise
- Location: `pyvbmc/acquisition_functions/acq_fcn_imiqr.py:264`,
  `pyvbmc/acquisition_functions/acq_fcn_viqr.py:526`; consumers at
  `pyvbmc/vbmc/active_importance_sampling.py:226`, `:237-242`, `:270-277`,
  `:367`, `:385`; MATLAB: "not read (internal track)"
- Category: cross-module
- Proposed classification: unsure (possibly intentional)
- Confidence: medium
- `is_log_added` is documented as `log[sinh(u·f_s)]` with `f_s` "the GP
  predictive variance at the input points", and the quantity the acquisition
  integrates (paper Eqs. 5, 8, 9) is the **latent** posterior standard
  deviation `s_Ξ(θ′)`. Accordingly `active_importance_sampling` feeds
  `is_log_base`/`is_log_added` the output of `gp.predict(…,
  separate_samples=True)` with `add_noise=False` at every site. Only
  `is_log_full` — the density the IMIQR MCMC step actually samples from
  (`active_importance_sampling.py:226`, with 100 samples by default) — calls
  `gp.predict(x, add_noise=True)`. Two things follow. (a) The MCMC target is
  not the same density as the resampling target that chooses its starting
  point (`:237-242` uses the latent variance), so the chain is started from a
  draw of a different distribution than it targets. (b) gpyreg's
  `predict(add_noise=True)` computes `sn2_star = noise.compute(hyp, x_star,
  y_star, s2_star)` with `s2_star = None`
  (`gaussian_process.py:2028-2032`), and `GaussianNoise.compute` treats a
  missing `s2` as 0 (`noise_functions.py:258-259`). At levels 1 and 2 —
  precisely the levels IMIQR is used at — the user-provided/inferred term is
  therefore **not** added: only the constant `exp(2h₀)`. So whichever
  quantity was intended, the code delivers neither the latent variance nor a
  genuinely noise-inclusive one.
  The self-normalized importance-sampling estimator itself stays unbiased,
  because the weights are formed as `ln_y − log_p` with `log_p` the sampler's
  own returned log-density (`:277`); what is affected is the proposal, its
  agreement with the resampling step, and the weight spread.
- Consequence if real: a proposal mismatch in IMIQR's importance sampling,
  hence more weight variance and a noisier acquisition surface, not a bias.
  Measured magnitude: on the level-2 oracle state the log density differs by
  2e-6 to 1.4e-5 (the constant noise is at its `tol_gp_noise` floor, variance
  1.3e-5); on my level-1 fit the fitted constant term reached variance 0.13
  across samples, against a per-observation noise of about 2, so the
  difference stays small but is not always negligible. VIQR's `is_log_full`
  is reached only through the dormant `mcmc_importance_sampling` hook.
- Suggested reproduction (run): `t_islogfull.py` — compares
  `acq.is_log_full(X, gp=gp1)` with
  `is_log_base + is_log_added(latent f_s2)` on the noisy oracle fixture, and
  prints `predict(add_noise=True) − predict()` against `gp.s2` (1.0) and
  `sn2_new` (1.000037).
- Test adequacy: no. `test_acq_fcn_imiqr.py::test_complex__call__` pins the
  acquisition value against the samples and weights that were actually drawn
  (`weighted_samples_reference`), so it is invariant to which proposal
  produced them; nothing anywhere compares `is_log_full` with
  `is_log_base + is_log_added`.

### F4. The candidate's observation noise is averaged over hyperparameter samples, then used inside each per-sample formula
- Location: written at `pyvbmc/vbmc/active_sample.py:351`
  (`gp.temporary_data["sn2_new"] = sn2new.mean(1)`), consumed at
  `pyvbmc/acquisition_functions/abstract_acq_fcn.py:348` and used at
  `acq_fcn_viqr.py:301,362`, `acq_fcn_imiqr.py:82,143`,
  `acq_fcn_noisy.py:39,44`; MATLAB: "not read (internal track)"
- Category: formula
- Proposed classification: possibly intentional
- Confidence: low
- Paper Eq. 5 is a statement about one GP. Averaging over hyperparameter
  samples should therefore happen at the level of the acquisition — which it
  does, `acq_fcn_viqr.py:383-392` and `acq_fcn_imiqr.py:170-175` take the log
  of the mean over samples. But the `σ²_obs(θ⋆)` that enters each sample's
  `tau2 = C_s² / (f_s2[:, s] + sn2)` is the arithmetic mean over samples,
  while `C_s` and `f_s2[:, s]` are that sample's own. At levels 0 and 2 the
  quantity that varies across samples is only `exp(2h₀)`, which is tiny; at
  level 1 the inferred multiplier `exp(h₁)` *is* the noise, and it is the
  broadest-prior hyperparameter of the model (`student_t(log 1, log 10, 3)`,
  `gaussian_process_train.py:352-355`). In my level-1 fit the eight samples
  gave `sn2_new` from 1.244 to 2.906 (mean 1.976) — a factor of 2.3 — and
  every sample's look-ahead variance was computed with 1.976.
- Consequence if real: the look-ahead variance reduction is overstated for
  the high-noise samples and understated for the low-noise ones, i.e. the
  per-sample terms of the integrated IQR are biased in opposite directions
  and the averaged acquisition is smoothed. It only bites at level 1, and
  only when the hyperparameter chain has real spread; a MAP-only fit
  (`Ns = 1`, `N ≥ 200 + 10D`) is unaffected. I did not measure the effect on
  a selected point.
- Suggested reproduction: evaluate VIQR on one level-1 state twice — once as
  the code does and once with `sn2` taken per sample inside the loop — and
  compare the argmin over a sieve of 2^13 candidates. Not run.
- Test adequacy: no, and deliberately so: `_look_ahead.py`'s references take
  `sn2_new` from `acq_fcn._estimate_observation_noise` itself
  (`test_acq_fcn_viqr.py:173`, `test_acq_fcn_imiqr.py:179`), and the test
  docstrings say "the estimator itself is not what these checks pin". No
  fixture and no test is at uncertainty-handling level 1 (checked: all eight
  oracle snapshots are level 0 except `rosenbrock_D2_noise1_viqr`, which is
  level 2 with `S = 1`, `n_evals = 1`).

### F5. `sn2_new` omits `sn2_mult`, which gpyreg applies both to the training diagonal and to the test-point noise
- Location: `pyvbmc/vbmc/active_sample.py:345-351`, consumed through
  `pyvbmc/acquisition_functions/abstract_acq_fcn.py:348`; compare
  `gpyreg/gaussian_process.py:2024-2032` and `:2714-2758`; MATLAB: "not read
  (internal track)"
- Category: cross-module
- Proposed classification: unsure
- Confidence: low
- When a Cholesky factorization fails, `__core_computation` retries with
  `sn2_mult *= 10` and the posterior it returns is the posterior of a GP
  whose noise is `sn2_mult · sn2` (the `L_chol` branch folds it into
  `sl = sn2_div·sn2_mult`, the other into
  `A.flat[::N+1] += sn2_mult·sn2_diag`). `predict(add_noise=True)`
  consistently reports `s2 + sn2_star·sn2_mult`. `active_sample`'s
  `sn2new` calls `gp.noise.compute` directly and never multiplies, so when
  `sn2_mult > 1` the acquisitions' `y_s2 = f_s2 + sn2` understates the
  denominator of Eq. 5 by up to that factor while `f_s2` already reflects the
  inflated posterior.
- Consequence if real: `tau2` overstated, look-ahead variance understated, on
  ill-conditioned GPs only. On the noisy oracle fixture `sn2_mult` is 1 for
  every sample, so nothing is visible there. The Cholesky ladder is described
  in `AGENTS.md` as a real occurrence ("the matrices are borderline
  singular"), so the path is not hypothetical, but I could not measure how
  often `sn2_mult > 1` in practice.
- Suggested reproduction: construct a GP whose kernel matrix forces one retry
  (duplicate rows and a very small `noise_log_scale`), then compare
  `gp.noise.compute(...)` with `predict(add_noise=True) − predict()`. Not
  run.
- Test adequacy: no test constructs a state with `sn2_mult > 1`; every test
  and fixture I inspected has `sn2_mult == 1`.

### F6. The known-differences sheet says `AcqFcnVIQR` offers only `loss="iqr"`; the shipped class offers `"iqr_reduction"` as well
- Location: `pyvbmc/acquisition_functions/acq_fcn_viqr.py:87-112`
  (documented parameter), `:120` (`LOSSES = ("iqr", "iqr_reduction")`),
  `:364-370`, `:384-386`, `:396-440` (`_log_iqr_reduction`); the entry is
  "Slice P3 → The EIG acquisition and the experimental VIQR losses were
  removed", `known_differences.md:1084-1098`; MATLAB: "not read (internal
  track)"
- Category: defaults
- Proposed classification: suspected defect (in the sheet, not the code)
- Confidence: high
- The entry states that the `loss` variants `iqr_reduction`, `var_reduction`
  and `sd_reduction` were removed and that "`AcqFcnVIQR` offers only the
  standard `loss="iqr"`". `var_reduction` and `sd_reduction` are indeed gone
  and `acq_fcn_eig.py` does not exist, but `iqr_reduction` is present,
  documented in the class docstring, validated in `__init__`, implemented in
  `_log_iqr_reduction`, covered by `test_acq_fcn_viqr_losses.py` and
  `test_viqr_kernel_reuse.py:131`, and announced to users in
  `CHANGELOG.md:219` ("`AcqFcnVIQR(loss="iqr_reduction")` scores the
  reduction of the integrated interquantile range").
- Consequence if real: a reviewer using the sheet would skip a live, shipped
  code path — `_log_iqr_reduction` and the `logsumexp` averaging branch at
  `:386` are the least-covered arithmetic in the file, and
  `_can_reuse_prediction_kernel` deliberately disables the fast path for it
  (`:188-191`), so it is also the only VIQR path that never sees the reused
  kernels.
- Suggested reproduction (run): `AcqFcnVIQR(loss="iqr_reduction")`
  constructs and evaluates; `grep -rn iqr_reduction` over the package and the
  changelog.
- Test adequacy: the code path is tested (`test_iqr_reduction_matches_brute_force`,
  `test_iqr_reduction_identity_with_viqr`); it is the sheet that is stale.

### F7. `_log_iqr_reduction`'s `s_a − s_pred` is not the difference of the two values it then uses
- Location: `pyvbmc/acquisition_functions/acq_fcn_viqr.py:418-431`; MATLAB:
  "not read (internal track)"
- Category: formula
- Proposed classification: unsure
- Confidence: low
- `s_pred = sqrt(max(f_s2_a − tau2, 0))` is clamped, but
  `d = tau2 / (s_a + s_pred)` is used as `s_a − s_pred`. The identity
  `s_a − s_pred = tau2/(s_a+s_pred)` holds only while `tau2 ≤ f_s2_a`. That
  inequality is guaranteed in exact arithmetic
  (`C² ≤ s²(θ⋆)s²(a) ≤ (s²(θ⋆)+σ²)s²(a)`), but `f_s2_a` comes from
  `gp.predict` on the importance points while `tau2` is assembled from a
  separate kernel and triangular-solve path, so the two can disagree at
  rounding level and `tau2` can exceed `f_s2_a` by an ulp or two near a
  training input, where `f_s2_a` is itself near zero from cancellation. There
  `d` overstates the reduction while `s_pred` is clamped to 0.
- Consequence if real: a relative error in one term of a `logsumexp` over
  ~100 terms, of the order of the cancellation in `f_s2_a`; negligible unless
  a single importance point dominates the weighted sum. Affects the
  non-default `loss="iqr_reduction"` only.
- Suggested reproduction (run, partially): `t_logsum.py` perturbs one
  `tau2` entry to `1.0000001 × f_s2_a[0]`; with 40 equally weighted terms the
  result is unchanged to 8 digits, so the effect is invisible unless the
  perturbed term carries the weight. A targeted check would take `Na = 1`.
- Test adequacy: no. `test_iqr_reduction_matches_brute_force` builds `tau2`
  from `predict_full` and so reproduces the same consistent pair; it cannot
  see the mismatch.

### F8. `AcqFcnVIQR` and `AcqFcnIMIQR` replace `acq_info` instead of extending it, dropping `compute_var_log_joint`
- Location: `pyvbmc/acquisition_functions/acq_fcn_viqr.py:127`,
  `pyvbmc/acquisition_functions/acq_fcn_imiqr.py:21`; contract stated in
  `abstract_acq_fcn.py:13-34`; MATLAB: "not read (internal track)"
- Category: state/caching
- Proposed classification: possibly intentional
- Confidence: medium
- Both `__init__` do `self.acq_info = {}` without calling
  `super().__init__()`, so the two keys the base class guarantees
  (`compute_var_log_joint`, `log_flag`) are not both present:
  `log_flag` is set explicitly, `compute_var_log_joint` is absent. `AcqFcn`,
  `AcqFcnLog`, `AcqFcnNoisy` and the test subclasses all have it. Inside the
  package nothing breaks — `active_sample.py:422` reads it with `.get` — but
  the base class's docstring documents the flag as part of `acq_info`, and
  `AcqFcnVIQR().get_info()["compute_var_log_joint"]` raises `KeyError`
  (verified). Anything that subclasses `AcqFcnVIQR` or reads `get_info()` by
  key, including a future in-package reader that uses `[...]`, would see the
  difference.
- Consequence if real: a `KeyError` in third-party or future code; no effect
  on a current run.
- Suggested reproduction (run): `t_misc.py` prints the key's presence for
  all five shipped acquisitions and the `KeyError` from `get_info()`.
- Test adequacy: partially. `test_acq_fcn_viqr.py::test_acq_info` and
  `test_acq_fcn_imiqr.py::test_acq_info` list the keys they expect and do not
  include `compute_var_log_joint`, whereas `test_acq_fcn.py`,
  `test_acq_fcn_log.py`, `test_acq_fcn_noisy.py` and
  `test_abstract_acquisition_function.py::test_acq_info` assert
  `not acq_fcn.acq_info.get("compute_var_log_joint")` — which passes both for
  a `False` value and for a missing key, so it cannot distinguish them.

### F9. `quantile` is not validated; a value outside (0.5, 1) silently produces `nan` or `-inf` acquisitions
- Location: `pyvbmc/acquisition_functions/acq_fcn_viqr.py:122-136`,
  `pyvbmc/acquisition_functions/acq_fcn_imiqr.py:20-27`; MATLAB: "not read
  (internal track)"
- Category: defaults
- Proposed classification: possibly intentional
- Confidence: low
- `__init__` validates `loss` against `LOSSES` but takes any `quantile`.
  The 2020 paper states `p_u ∈ (0.5, 1)`. With `quantile = 0.5`, `u = 0`,
  every `sinh` term is 0 and the acquisition is `-inf` at every candidate;
  with `quantile < 0.5`, `u < 0` and both `log(sum(sinh(negative)))` (VIQR's
  direct path) and `log1p(-exp(-2u·s))` (VIQR's fallback and IMIQR) produce
  `nan`; with `quantile` 0 or 1, `u` is infinite and the result is `nan`.
  Verified for all five values. `np.argmin` over an all-`nan` array returns
  index 0, so `active_sample.py:435` would silently take the first sieve
  candidate for the rest of the run; over an all-`-inf` array it does the
  same.
- Consequence if real: a mistyped quantile turns active sampling into "always
  take the first candidate", with no error and no warning. Only reachable
  through an explicit user argument.
- Suggested reproduction (run): `t_quantile.py`.
- Test adequacy: no. `test_acq_info` in both modules checks only `0.666` and
  the default; no test passes an out-of-range quantile.

**Checked and found correct** (so that the absence of a finding is
deliberate): the paper equations of all four pointwise acquisitions
(`a_us = V·q²`, `a_pro = V·q·exp(f̄)`, its log form, and
`a_npro = V·(1 − sn2/(V+sn2))·q·exp(f̄) = V²/(V+sn2)·q·exp(f̄)`, paper Eqs. 3
and 6); the `-z` shift as a pure constant under both the multiplicative and
the additive regularization; `var_tot = mean(f_s2) + var(f_mu, ddof=1)`
matching gpyreg's own sample averaging; the variance-regularization limits at
zero variance (`+inf` for a log acquisition, `0` for a plain one) and their
mutual consistency; the bound mask; `_sq_dist` against GPML's centering and
shapes; the look-ahead variance and both `L_chol` branches of VIQR and IMIQR
against a brute-force refit (4.4e-12) and against each other (1.1e-9);
`sW` being constant across training points, so `sW[0]` is the right `sl`;
`K_Xa_X`, `C_tmp` and `f_s2` as prepared by `active_importance_sampling`;
the `return_cross_covariance` reuse path (bit-identical to `cdist`, and
bit-identical acquisition values); `_log_viqr_sum` against 60-digit
references including its overflow guard, exact zeros and the `-inf` row, and
the agreement of its two internal paths; `_log_iqr_reduction`'s
`2 cosh(A) sinh(B)` derivation (4.5e-16); the `log of a mean` averaging over
hyperparameter samples in both losses and in IMIQR; the global (rather than
per-sample) weight normalization, which is right for IMIQR because Eq. 8
integrates the *unnormalized* joint; VIQR's uniform weights, which match
`vp.sample(Na, orig_flag=False)` being plain Monte Carlo from `q_φ`; and all
array shapes and broadcasting axes in both loops.

## 4. Test adequacy notes

- **`test_real2int`** (`test_abstract_acquisition_function.py:445-477`)
  asserts nothing: its four checks are bare `np.all(...)` expressions
  without `assert`. It would pass whatever `_real2int` returned. This is the
  test that should have covered F1.
- **`_regularization.check_variance_regularization`** mirrors the
  implementation: it builds `expected` with the same expression the code uses
  (`expected[mask] += tol_var / var_tot[mask] - 1`, `:116`) and then asserts
  the code produces it. It pins batch-vs-pointwise agreement and the mask,
  which is valuable, but it cannot detect a wrong penalty. The log/non-log
  equivalence (that `-log` of the multiplicative penalty is the additive one)
  is never checked.
- **The `_look_ahead.py` references deliberately do not pin the noise.** Both
  `gauss_hermite_reference` and `weighted_samples_reference` are handed
  `sn2_new = acq_fcn._estimate_observation_noise(...)`, so the whole subject
  of the first question — which `s2`, which hyperparameters, whether the
  per-observation or the pooled variance — is outside every check. The test
  docstrings say so honestly. The references themselves are genuinely
  independent (they use `gp.predict_full`), and they are the strongest tests
  in this slice.
- **No test or fixture is at uncertainty-handling level 1.** All eight oracle
  snapshots are level 0 except `rosenbrock_D2_noise1_viqr`, which is level 2
  with `S = 1` and `n_evals = 1` — the one configuration in which levels 1
  and 2 give the same `s2`. Every hand-built GP in
  `pyvbmc/testing/acquisition_functions/` uses
  `GaussianNoise(constant_add=True, user_provided_add=True)`, i.e. the
  level-2 switches. Nothing exercises `scale_user_provided`, so F4's
  cross-sample averaging and the `S² · n_evals` un-pooling are untested.
- **`test_acq_fcn_noisy.py::test_complex__call__`** is the only test in the
  slice with stored MATLAB reference values, and it is a good one; but
  `gp.temporary_data["sn2_new"]` is set by hand to an arbitrary vector, so it
  pins `AcqFcnNoisy`'s arithmetic, not where its noise comes from.
- **`_setup`/`look_ahead_scenario` fix the hyperparameters by hand** and the
  acquisition tests never go through `train_gp`, so no test in this slice
  ever sees the hyperparameter values a real fit produces (which is what
  makes F4's spread visible).
- `test_acq_fcn_viqr.py::test_acq_info` and the IMIQR counterpart enumerate
  the `acq_info` keys they expect without asserting the base-class contract,
  which is how F8 survives; the sibling modules' `assert not
  acq_info.get("compute_var_log_joint")` passes equally for `False` and for
  a missing key.
