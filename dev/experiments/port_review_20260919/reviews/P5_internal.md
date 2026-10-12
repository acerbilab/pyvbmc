# P5 — GP training policy, hyperpriors, training data, GP re-update, lean GP records, internal-correctness track

Raw reviewer report, wave 3 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, fresh
context, read-only on the repositories; it did not read the MATLAB source.
Its check scripts ran in the session scratchpad and are kept only on the
machine that ran them (`dev/scripts/runs/LOCAL.md`). The text below is the
agent's final message, unedited.

---

# P5 — GP training policy, hyperpriors and bounds, training data, GP re-update, lean records (internal-correctness track)

## 1. Coverage

**Read completely**
- `pyvbmc/vbmc/gaussian_process_train.py` (all 963 lines): `train_gp`, `_meanfun_name_to_mean_function`, `_cov_identifier_to_covariance_function`, `_gp_hyp`, `_get_gp_training_options`, `_get_hyp_cov`, `_get_training_data`, `_estimate_noise`, `_lean_gp`, `_restore_gp_posteriors`, `reupdate_gp`.
- gpyreg, the parts this file hands work to: `gaussian_process.py` `GP.__init__`, `set_bounds`, `get_bounds`, `bounds_to_dict`, `get_recommended_bounds`, `get_priors`, `set_priors`, `get_hyperparameters`, `set_hyperparameters`, `update` (including the rank-one path), `clean`, `fit`, `__recompute_normalization_constants`, `__prior_masks`; `noise_functions.py` in full; `covariance_functions.py` in full; `mean_functions.py` bounds helper; `slice_sample.py` `sample`'s result contract and shapes.
- Callers and state producers: `pyvbmc/vbmc/vbmc.py` (`_init_optim_state` GP settings, the `optimize` loop around `train_gp`/`_lean_gp`/`reupdate_gp`, `get_gp`, `_compute_reliability_index`, `_check_gp_sampling_stop`), `pyvbmc/vbmc/active_sample.py` (the in-loop `train_gp`/`reupdate_gp`/`gp.update` paths and the `temporary_data` pre-computations), `pyvbmc/vbmc/iteration_history.py` `record`, `pyvbmc/function_logger/function_logger.py` (`X_flag`/`y`/`S`/`n_evals` shapes, the pooling of repeats, `noise_flag`), `pyvbmc/stats/get_hpd.py`, `pyvbmc/rng.py`.
- Specification sources: `papers/acerbi2020variational_appendix.md` Table S1 and §B.1, `papers/acerbi2018variational_appendix.md` §B.2, `pyvbmc/vbmc/README.md` (the five mapping lines for this file), the option `.ini` descriptions, `INERT_OPTIONS`.
- Tests: `pyvbmc/testing/vbmc/test_gaussian_process_train.py`, `pyvbmc/testing/vbmc/test_gp_records.py`, the oracle harness `pyvbmc/testing/oracles/_gp_fit_history.py` (option key list).
- Known-differences sheet: the "Slices G1, G2, P5" section and "Settled non-differences".

**Skimmed**: the rest of `vbmc.py`'s loop, `whitening.py` (only to confirm a warp rewrites `plb_tran`/`pub_tran`), `abstract_acq_fcn.py` (only the two `temporary_data` reads).

**Not reached**: MATLAB sources (other track); the acquisition functions, `_gp_log_joint`, the entropies.

**Checks run** (all small, under `…/scratchpad/wave3_P5_internal`, with the repo venv):
1. `_gp_hyp` bounds with `upper_gp_length_factor ∈ {0, 3}`, then gpyreg's fit-time NaN filling — confirms F1.
2. `GaussianNoise` flag combinations for uncertainty levels 0/1/2; hyperparameter names and counts; `noise.compute` with per-point `s2` — confirms F3.
3. `_lean_gp` / `_restore_gp_posteriors` round trip on a fitted GP (bit-exact) and, for contrast, rank-one-updated vs fully recomputed factors.
4. `_get_hyp_cov` against `np.cov(ddof=1)` on a single equal-weight block (agrees to 2.2e-16).
5. `np.min(a, b)` two-argument behavior; `np.cov` on a one-row sample matrix — confirms F2 and F5.
6. `_gp_hyp` end-to-end for all three uncertainty levels: resulting priors, bounds, `hyp0` size.

**Checked and found not to be defects** (so the orchestrator need not re-derive them): the length-scale / output-scale / noise / mean hyperpriors match Table S1 of `acerbi2020variational_appendix.md` (mean `log[sqrt(D/6)·L]`, scale `log sqrt(1e3)`, ν=3; noise mean `log sqrt(1e-5)`, scale 0.5; uniform for σ_f, m₀, x_m, ω — gpyreg's `u_idx` does treat NaN mu/sigma as uniform regardless of the `df_base=7` fill in `fit`); `hpd_frac = 0.8` matches f_hpd; hyperparameter block order (cov, noise, mean) is consistent between `hyp0`, `_estimate_noise`'s slice and gpyreg's `hyper_info`; the units of every bound and prior match gpyreg's parameterization (`ell = exp(h)`, `sf2 = exp(2h)`, `sn2 = exp(2h)`); `_get_hyp_cov`'s weighting (per-iteration weight split equally over that iteration's samples) and its estimator (reliability-weight unbiased, `/(1 - Σwᵢ²)`) are correct and its `None` guards are sound; the half-history window in `train_gp`'s starting-point loop (`ceil((n+1)/2)-1 … n`) is the correct 0-based translation of a 1-based `ceil((n+1)/2) … n`; the `init_N` cubic runs from `gp_train_n_init` at x=0 to `gp_train_n_init_final` at x=1 and is monotone; `_lean_gp` does not touch the source GP and the restored factors are bit-identical for the GPs that are actually recorded (the record is taken from the freshly fitted GP, before the post-record `reupdate_gp`); `gp.temporary_data` caches are recomputed inside the active-sampling loop before every use, so `reupdate_gp` not clearing them is harmless; the `iteration > 1` guard at line 651 differs from the `iteration > 0` guard at line 534 but is inert, because `_compute_reliability_index` returns `inf` for iterations 0 and 1; `t_train` (returned by `_get_training_data`) and `hyp_dict["logp"]` are written and never read, which is consistent with the "Missing port" comments.

## 2. Findings

### F1. `upper_gp_length_factor` is silently discarded: the length-scale upper bound is overwritten two statements later
- Location: `pyvbmc/vbmc/gaussian_process_train.py:367-373` and `:394-402`; MATLAB: "not read (internal track)"
- Category: control flow (bound construction)
- Proposed classification: suspected defect
- Confidence: high
- What the code does: `bounds = gp.get_bounds()` returns a dict of `(lower, upper)` **tuples**. Line 368-373 sets `bounds["covariance_log_lengthscale"] = (-inf, log(upper_gp_length_factor * (pub_tran - plb_tran)))`. Line 399-402 then assigns the same key again, `= (cov_bounds_info["LB"][:D], np.nan)`, replacing the whole tuple. `np.nan` means "use gpyreg's recommendation", and `GP.fit` fills it with `log(10 * (max(X) - min(X)))` computed on the full training set. So the user's upper bound never reaches the fit. The two statements are written as if bounds were two independent vectors (as they are on the MATLAB side), which is exactly the shape of a port artifact. The option is documented in `advanced_vbmc_options.ini:243` as "Upper bound on GP input lengths based on plausible box (0 = ignore)", is *not* in `INERT_OPTIONS`, and its description is asserted verbatim by `test_options.py:427`.
- Consequence if real: the option is inert. At the shipped default (`0`) nothing changes; a user who sets it to cap the GP input length scales — the documented way to stop the GP from smoothing across the whole plausible box — gets no effect and no warning. With the default SE kernel the clobbering branch always executes, so there is no configuration in which the option works.
- Suggested reproduction (ran it): build a GP and call `_gp_hyp` with `upper_gp_length_factor = 3`, `plb = -1`, `pub = +1`, D=2, then `gp.get_recommended_bounds(gp.lower_bounds, gp.upper_bounds)`. Intended UB would be `log(3·2) ≈ 1.79`; actual is `[3.720, 3.474]` (= `log(10·width)`), identical to the `factor = 0` run.
- Test adequacy: no. No test sets the option or inspects `_gp_hyp`'s bounds at all (`test_gp_hyp` asserts only two prior entries). The `INERT_OPTIONS` test would not catch it either: it scans for a *syntactic* read through an options mapping, and `options["upper_gp_length_factor"]` is read — its result is just thrown away. The `gp_fit` oracle captures the option but only at its default.

### F2. The running hyperparameter-covariance guard tests the number of hyperparameters, not the number of samples
- Location: `pyvbmc/vbmc/gaussian_process_train.py:195`; MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high (that the test is on the wrong axis); medium (on the size of the consequence)
- What the code does: `if hyp_dict["full"] is not None and hyp_dict["full"].shape[1] > 1:` then `hyp_cov = np.cov(hyp_dict["full"].T)`. `hyp_dict["full"]` is `sampling_result["samples"]` from `gpyreg.GP.fit`, whose shape is `(n_samples, hyp_N)` (`slice_sample.py`: `samples = np.zeros((N, D))`). The guard therefore tests `hyp_N > 1`, which is true for every model PyVBMC can build (the minimum is 6, at D=1). The quantity that must exceed 1 for `np.cov(..., ddof=1)` to be defined is the number of rows, `shape[0]`. This is the mirror image of a MATLAB-shaped test: the stored matrix is transposed relative to MATLAB's, so index 2 became `shape[1]` when it should have become `shape[0]`.
- Consequence if real: the guard never fires, so the `else` branch that sets `run_cov = None` is dead whenever any samples exist. With one pre-thinning sample (`gp_sample_thin = 1` and `gp_s_N` rounding to 1) `np.cov` returns an all-NaN matrix with a "Degrees of freedom <= 0" warning, and `hyp_dict["run_cov"]` becomes NaN; `_get_hyp_cov` would then hand NaN widths to the slice sampler. This is unreachable at the shipped defaults for two independent reasons: `gp_sample_thin = 5` makes the pre-thinning block at least 5 rows, and `weighted_hyp_cov = True` means `run_cov` is never read (`:723-724`). It bites only under `weighted_hyp_cov = False` together with `gp_sample_thin = 1`. Worth reporting because the guard reads as protection that is not there, and because the same axis confusion is the class of error this review is looking for.
- Suggested reproduction (ran it): a `(1, 6)` array gives `shape[1] > 1 → True`, `shape[0] > 1 → False`, and `np.cov(a.T)` all-NaN.
- Test adequacy: no. Nothing exercises the `run_cov` update block; `test_get_hyp_cov*` only feed `run_cov` in ready-made (`hyp_dict = {"run_cov": np.eye(3)}`).

### F3. Uncertainty level 1 ("infer noise") builds the same noise function as level 0: the `scale_user_provided` flag is a no-op without `user_provided_add`, so the per-point `s2` and its multiplier hyperparameter are lost
- Location: `pyvbmc/vbmc/gaussian_process_train.py:96-105`, with the dependent code at `:341-356` and `:410-419`; producer at `pyvbmc/vbmc/vbmc.py:1086-1094`; MATLAB: "not read (internal track)"
- Category: cross-module (defaults / model construction)
- Proposed classification: suspected defect
- Confidence: medium-high
- What the code does: `optim_state["gp_noise_fun"] = [1, 2, 0]` for `uncertainty_handling_level == 1`. `_gp_hyp` decodes it as `user_add = (…[1] == 1) → False`, `user_scale = (…[1] == 2) → True` and passes `GaussianNoise(constant_add=True, user_provided_add=False, scale_user_provided=True)`. gpyreg's constructor only honors `scale_user_provided` **inside** the `if user_provided_add:` block (`noise_functions.py:34-41`, and the docstring says so: "If `user_provided_add = False` then this does nothing"). The resulting `parameters` is `[1, 0, 0]` — byte-identical to level 0. Consequences inside the file: `noise.hyperparameter_count()` is 1, so there is no `noise_provided_log_multiplier`; the prior built for it at `:415-419` is silently dropped, because `set_priors` iterates the model's `hyper_info` and never looks at extra keys; and `noise.compute` takes neither the `sn2 += s2` nor the `sn2 += exp(h)·s2` branch, so the `s2` that `_get_training_data` supplies is ignored. That `s2` is real data: `FunctionLogger` is created with `noise_flag = uncertainty_handling_level > 0` (`vbmc.py:465`), records `f_sd = 1` per evaluation for level 1 (`function_logger.py:290-297`), and pools repeats into `S = 1/sqrt(n_evals)`, so `s2_train = 1/n_evals` is exactly the per-row relative weight the multiplier was meant to scale.
- Consequence if real: with `uncertainty_handling=True` and no `specify_target_noise` — a supported mode, per `Options.uncertainty_handling_on`'s own docstring ("when `uncertainty_handling` asks for the noise level to be inferred") — every training row gets the same GP noise variance no matter how many times it was evaluated, so pooled repeated observations gain no weight in the GP fit; and the multiplier's hyperprior (`student_t(log(noise_mult), noise_mult_std, 3)`, with `noise_size` folded in) has no effect. The inferred noise can still be absorbed by the constant term, whose prior scale is widened to `log(10)` for this level, so the run does not fail — it just fits a homoskedastic noise where the code and the option name promise an inferred, repeat-aware one. The in-file comment "This branch is not used and tested at the moment" (`:348`) is stale: the branch is reachable from a documented option.
- Suggested reproduction (ran it): `GaussianNoise(constant_add=True, user_provided_add=False, scale_user_provided=True)` has `parameters = [1,0,0]`, `hyperparameter_count() == 1`, and `compute(hyp, X, y, s2=[[1],[0.25],[0.1]])` returns the single value `0.09`, where the flag combination the code names (`user_provided_add=True, scale_user_provided=True`) returns `[1.09, 0.34, 0.19]`. `_gp_hyp` run at levels 0/1/2 gives `hyp0` of size 9 in all three cases and no multiplier prior in any.
- Test adequacy: no. `test_gp_hyp` runs only `specify_target_noise=True` (level 2) and asserts two scalars of the noise prior. No test constructs a level-1 run's GP or checks its hyperparameter count. The oracle fixtures are level 0 and level 2.

### F4. The end-of-warm-up reset of the running hyperparameter covariance writes a key nothing reads
- Location: `pyvbmc/vbmc/vbmc.py:1657` (`self.hyp_dict["runcov"] = None`), against `pyvbmc/vbmc/gaussian_process_train.py:73-74, 195-203, 723-724`; MATLAB: "not read (internal track)"
- Category: state/caching (cross-module)
- Proposed classification: suspected defect
- Confidence: high (the key is wrong); medium (on the consequence, which is default-off)
- What the code does: when warm-up ends, `optimize` intends to discard the running average of the GP hyperparameter covariance — the line above it is the MATLAB line it ports, `# hypstruct.runcov = []`. It writes `hyp_dict["runcov"]`, but every producer and consumer in `gaussian_process_train.py` uses `hyp_dict["run_cov"]`. A repository-wide grep finds `"runcov"` at this one site and nowhere else, so the statement only inserts an inert key into `hyp_dict` (which is then deep-copied into `optim_state["hyp_dict"]` and into saved runs) and leaves `run_cov` untouched.
- Consequence if real: the covariance accumulated during warm-up, when the GP and the posterior are furthest from converged, survives into the main loop and keeps steering the slice-sampler widths through the exponential average at `:197-201`. Because `weighted_hyp_cov` defaults to `True`, `run_cov` is not read at the shipped defaults, so the visible effect is confined to `weighted_hyp_cov = False` runs; at defaults the consequence is that the whole `run_cov` block is dead work.
- Suggested reproduction: a two-line assertion after a short run with `weighted_hyp_cov=False` — `vbmc.hyp_dict["run_cov"]` is not `None` after the warm-up-end iteration, and `"runcov"` is present in `vbmc.hyp_dict`. Not run (it needs an `optimize()` call, outside my budget); the grep and the read are conclusive about the key.
- Test adequacy: no. No test inspects `hyp_dict` after warm-up ends; `test_get_hyp_cov*` construct `run_cov` by hand.

### F5. `np.min` called with two positional arguments in the output-dependent-noise bound
- Location: `pyvbmc/vbmc/gaussian_process_train.py:427`; MATLAB: "not read (internal track)"
- Category: formula (API misuse)
- Proposed classification: suspected defect
- Confidence: high (the call is wrong); the branch is currently unreachable
- What the code does: `[np.min(np.min(y), np.max(y) - 20 * D), -np.inf]`. `np.min`'s second positional parameter is `axis`, not a second operand; the intended function is `np.minimum`. Passing a float there raises `TypeError: 'numpy.float64' object cannot be interpreted as an integer`.
- Consequence if real: the branch is guarded by `optim_state["gp_noise_fun"][2] == 1`, which nothing sets — `vbmc.py:1086-1100` only ever writes index 0 and 1, and `noise_shaping` (the one option that touches index 1) is rejected at construction. So the line cannot execute today; if the rectified-linear output-dependent noise is ever enabled it raises immediately rather than producing a wrong bound.
- Suggested reproduction (ran it): `np.min(np.float64(-3.0), np.float64(-39.0))` raises `TypeError`; `np.minimum` of the same gives `-39.0`.
- Test adequacy: no. Nothing reaches the branch.

### F6. The observation-noise upper bound is set to `+inf` while the surrounding code uses `nan` to mean "keep gpyreg's recommendation"
- Location: `pyvbmc/vbmc/gaussian_process_train.py:374-375`; MATLAB: "not read (internal track)"
- Category: defaults
- Proposed classification: unsure (possibly intentional)
- Confidence: low
- What the code does: the comment is `# Increase minimum noise.` and the statement is `bounds["noise_log_scale"] = (np.log(min_noise), np.inf)`. It does raise the lower bound as the comment says, but it also replaces gpyreg's recommended upper bound `log(max(y) - min(y))` with `+inf`, which the comment does not mention. Twenty lines later the same function writes `np.nan` in the upper slot precisely to keep gpyreg's recommendation (`:396-402`), so `+inf` here is inconsistent with the file's own idiom; and the structure — a single-sided change written as a whole tuple — is the same shape as F1. I could not tell from inside Python whether the widening is deliberate.
- Consequence if real: the fit may place the observation-noise standard deviation above the whole range of the observed log-joint values, and the Student-t prior's truncation constant changes. In practice the prior is centred at `log sqrt(1e-5)` with scale 0.5 (level 0 and 2), so `log(diam y)` sits tens of prior scales away and the optimizer will not go there; the truncation constant is a constant in the hyperparameters and does not move the MAP. I judge the numerical effect negligible at the shipped defaults, but the divergence from the recommended bound is real and undocumented.
- Suggested reproduction (ran it): `_gp_hyp` at all three uncertainty levels returns `bounds["noise_log_scale"] == (log sqrt(1e-5), inf)`.
- Test adequacy: no test looks at this bound.

### F7. Ten of the thirteen accepted `gp_mean_fun` names fail only once the first GP is trained
- Location: `pyvbmc/vbmc/gaussian_process_train.py:214-242` (`_meanfun_name_to_mean_function`), reached from `:88`; validation at `pyvbmc/vbmc/vbmc.py:1102-1122`; MATLAB: "not read (internal track)"
- Category: control flow (validation placement)
- Proposed classification: possibly intentional
- Confidence: medium
- What the code does: `VBMC.__init__` validates `gp_mean_fun` against a 13-name list taken from MATLAB (`zero, const, negquad, se, negquadse, negquadfixiso, negquadfix, negquadsefix, negquadonly, negquadfixonly, negquadlinonly, negquadmix`), but `_meanfun_name_to_mean_function` implements three of them and raises `ValueError("Unknown mean function!")` for the rest. The known-differences sheet records that only three mean functions exist in gpyreg; what it does not cover is that construction accepts the other ten and the run dies inside the first `train_gp`, after the initial design has already spent its function evaluations. The downstream `raise TypeError("The mean function is not supported by gpyreg.")` in `_gp_hyp` (`:390-391`) is consequently unreachable.
- Consequence if real: a user who sets, say, `gp_mean_fun="se"` loses the initial-design evaluations of an expensive target before being told the option is unsupported. No effect at the default `"negquad"`.
- Suggested reproduction: construct `VBMC(..., options={"gp_mean_fun": "se"})` — construction succeeds — then call `train_gp`, which raises. Not run (construction plus one `train_gp` is cheap, but the two code paths are unambiguous by reading).
- Test adequacy: partly. `test_meanfun_name_to_mean_function` asserts that the unsupported names raise *in that helper*, which documents the late failure rather than questioning it; no test asserts that `VBMC.__init__` rejects them.

## 3. Test adequacy notes

- `test_gaussian_process_train.py::test_get_gp_training_options_samplers` asserts, for each of six `gp_hyp_sampler` values, that `gp_train["sampler"]` equals the string that was requested. That restates the assignment it is testing. It would not notice that `"covsample"`'s whole point is unreachable: the matrix widths built at `:594-596` have `np.size == hyp_N²`, which `train_gp:128-131` then discards because it compares against `np.size(hyp0) == hyp_N`, and gpyreg's `fit` raises `ValueError("Unknown sampler!")` for any `sampler != "slicesample"` before the widths could be used. The same holds for `"npv"`, `"mala"`, `"slicelite"`, `"splitsample"` and `"laplace"`: five of the six branches the test covers cannot complete a fit.
- `test_gp_hyp` is named for `_gp_hyp` but asserts only two scalars of one prior (`noise_log_scale`'s mean and scale). None of the bounds `_gp_hyp` exists to build is checked, which is why F1 and F6 are invisible to the suite. The test also passes `vbmc.plausible_lower_bounds`/`plausible_upper_bounds` where `train_gp` documents transformed bounds (`plb_tran`/`pub_tran`); with the identity-ish transform of that fixture it does not matter, but it means the test would not detect a mix-up of the two spaces in the length-scale prior.
- `test_get_hyp_cov` pins the weighted covariance to two hard-coded 16-digit constants. I re-derived the estimator independently (equal-weight case reproduces `np.cov(ddof=1)` to 2.2e-16) and it is right, so the constants are not wrong — but as written they are an output snapshot, and any future change to the weighting would be "fixed" by updating them.
- `test_get_gp_training_options_opts_N` pins `opts_N` per branch but never checks `init_N`, `burn` or `thin`, which are the values gpyreg actually spends its budget on, and never checks the `init_N` schedule against its endpoints (`gp_train_n_init` at `n_eff = fun_eval_start`, `gp_train_n_init_final` at the horizon).
- `test_gp_records.py` is the one part of this slice tested against a stated property rather than an output: it asserts the lean/restore round trip is bit-exact and that the source GP is untouched, on three stored states including a single-sample and a heteroskedastic one. It does not cover a GP whose factors came from rank-one updates — where the round trip is *not* exact (I measured `max|ΔL| ≈ 1.6e-5` on a 25-point D=2 GP). That is currently fine, because the only `_lean_gp` call site (`vbmc.py:1609`) runs on the freshly fitted GP from `train_gp`; but the sheet's claim that "the rebuilt factors are identical to the ones dropped" holds because of that call-site ordering, not because restoration is exact in general, and nothing in the tests pins the ordering.
- Nothing in the suite exercises `train_gp`'s `run_cov` update block (F2) or the warm-up reset (F4), and nothing constructs a `uncertainty_handling_level == 1` GP (F3).
