# P5 — GP training policy, hyperpriors, training data, GP re-update, lean GP records, MATLAB-comparison track

Raw reviewer report, wave 3 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, fresh
context, read-only on the repositories; it compared the Python code with
MATLAB VBMC at `396d649`. Its check scripts ran in the session scratchpad
and are kept only on the machine that ran them
(`dev/scripts/runs/LOCAL.md`). The text below is the agent's final message,
unedited.

---

# P5 — GP training policy, hyperpriors and bounds, training data, GP re-update, lean GP records (MATLAB comparison track)

## 1. Coverage

**Read completely (Python):** `pyvbmc/vbmc/gaussian_process_train.py` (all of it: `train_gp`, `_meanfun_name_to_mean_function`, `_cov_identifier_to_covariance_function`, `_gp_hyp`, `_get_gp_training_options`, `_get_hyp_cov`, `_get_training_data`, `_estimate_noise`, `_lean_gp`, `_restore_gp_posteriors`, `reupdate_gp`); `pyvbmc/stats/get_hpd.py`; `pyvbmc/testing/vbmc/test_gaussian_process_train.py`.

**Read completely (MATLAB):** `misc/gptrain_vbmc.m` (including the local `vbmc_gphyp` and `estimate_GPnoise`), `misc/get_GPTrainOptions.m` (including `GetHypCov`), `misc/get_traindata_vbmc.m`, `misc/gpreupdate.m`, `misc/gpsample_vbmc.m`, `misc/gethpd_vbmc.m`, `gplite/gplite_train.m`.

**Read for the interfaces the slice depends on:** `gpyreg/gaussian_process.py` (`set_bounds`, `get_bounds`, `get_recommended_bounds`, `set_priors`, `get_priors`, `get_hyperparameters`, `update`, `clean`, `fit`, `Posterior`), `gpyreg/noise_functions.py`, `gpyreg/f_min_fill.py` (the design branches), the bound helpers of `gpyreg/covariance_functions.py` and `gpyreg/mean_functions.py`; `gplite/gplite_noisefun.m`; `misc/setupvars_vbmc.m` (GP settings and uncertainty levels), `private/vbmc_termination.m` (`getStableIter`, the reliability index), `vbmc.m:540-690`, `:795-845`, `:1035-1045` (`savestats`), `private/activesample_vbmc.m:425-520`; on the Python side `pyvbmc/vbmc/vbmc.py` (call sites of `train_gp`/`reupdate_gp`/`_lean_gp`, `optim_state` GP settings, `_compute_reliability_index`, the iteration record), `pyvbmc/vbmc/active_sample.py:276-300`, `:725-770`, `pyvbmc/function_logger/function_logger.py` (the `S` path), `pyvbmc/vbmc/iteration_history.py` (`record`), the option `.ini` defaults.

**Skimmed:** the remainder of `gpyreg/gaussian_process.py` (`__core_computation`, prior masks), `utils/slicelite.m` (confirmed below), the rest of the known-differences sheet.

**Confirmed as asked:** `utils/slicelite.m` is unported. gpyreg contains no slicelite (nor MALA, covsample, npv or laplace) sampler; `GP.fit` raises `ValueError("Unknown sampler!")` for any `sampler` other than `"slicesample"` once `n_samples > 0`. The only mention is the dead `log_P` option at `gpyreg/gaussian_process.py:1121`. I spent no further time on it.

**Checks run** (scripts in the scratchpad, `check_p5.py`, `check_noise_level1.py`, `check_level1_train.py`; interpreter `.venv`, BLAS single-threaded; `pyvbmc.__file__` and `gpyreg.__file__` printed and confirmed to be these checkouts):
1. Enumeration of the starting-point history window for `n = 1..8` against MATLAB's `ceil(n/2):n`.
2. `floor(Ninit/2)` vs `ceil(init_N/2)` for several `init_N`.
3. Printing of the bounds `_gp_hyp` installs on a 12-point synthetic state, beside `GP.get_recommended_bounds("recommended","recommended")` (which is what MATLAB's `NaN` entries become), with and without `upper_gp_length_factor > 0`.
4. `np.min(np.min(y), np.max(y) - 20*D)` on real arrays.
5. Construction of `GaussianNoise` from `optim_state["gp_noise_fun"]` at all three uncertainty levels, plus `compute` with a user-supplied `s2`.
6. A tiny `train_gp` at `uncertainty_handling_level == 1` (12 points, `gp_s_N` forced to 0) reporting the fitted hyperparameter set.

**Not reached:** the arithmetic inside `gplite_covfun`/`gplite_meanfun` against gpyreg's kernel and mean modules (slices G1/G2); `warp_gp_and_vp`, which supplies `hyp_dict["hyp"]` after a warp.

---

## 2. Findings

### F1. At `uncertainty_handling_level == 1` the GP noise function drops the provided-noise term and its multiplier hyperparameter
- Location: `pyvbmc/vbmc/gaussian_process_train.py:96-105` (and the consequences at `:347-356`, `:415-419`); MATLAB: `misc/gptrain_vbmc.m:60-63` handing `optimState.gpNoisefun` to `gplite_train`, with `misc/setupvars_vbmc.m:277-280` and `gplite/gplite_noisefun.m:60-78`, `:108-118`
- Category: cross-module (with a defaults component)
- Proposed classification: port discrepancy
- Confidence: high
- History: the MATLAB lines are unchanged since before 2021-01-19 (`misc/setupvars_vbmc.m`'s noise switch and `gplite_noisefun`'s `noisefun(2)==2` branch predate the port; `2044530` of 2021-06-18 did not touch them). The Python lines date from the original port of this module and have never matched: the in-code comment at `:348` ("This branch is not used and tested at the moment") records that the level-1 path was never exercised.
- What the code does: `optim_state["gp_noise_fun"]` is `[1, 2, 0]` at level 1 (`pyvbmc/vbmc/vbmc.py:1089-1091`), matching MATLAB's `gpNoisefun = [1 2]`. `train_gp` then translates it as

  ```python
  user_add   = optim_state["gp_noise_fun"][1] == 1      # False for 2
  user_scale = optim_state["gp_noise_fun"][1] == 2      # True
  noise_f = gpr.noise_functions.GaussianNoise(
      constant_add=..., user_provided_add=user_add,
      scale_user_provided=user_scale, ...)
  ```

  but `GaussianNoise.__init__` nests the scale flag inside the add flag:

  ```python
  if user_provided_add:
      self.parameters[1] = 1
      if scale_user_provided:
          self.parameters[1] += 1
  ```

  With `user_provided_add=False` the whole block is skipped, so `parameters` comes out `[1, 0, 0]` — level 0's noise function. My check prints exactly that: `level=1 gp_noise_fun=[1, 2, 0] -> GaussianNoise.parameters=[1. 0. 0.] n_hyp=1`. `GaussianNoise.compute` then ignores `s2` entirely (`parameters[1] == 0` reaches neither the `+= s2` nor the `+= exp(hyp)*s2` branch; verified: with `s2 = [4, 9, 16]` it returns the scalar `0.01`, where `parameters = [1,2,0]` returns `[4.01, 9.01, 16.01]`).

  What it should do: MATLAB's `[1 2]` gives `Nnoise = 2` — a constant log noise scale plus `noise_provided_log_multiplier`, and the total variance is `exp(2*hyp1) + exp(hyp2)*s2`. Two further lines of the slice then depend on that second hyperparameter and are silently lost: `misc/gptrain_vbmc.m:165` (`hyp0(Ncov+2) = log(noisemult)`) has no Python counterpart at all (`_gp_hyp` sets only `noise_x0[0]`), and `_gp_hyp:415-419` sets `priors["noise_provided_log_multiplier"]`, a key that is not among `gp.noise.hyperparameter_info()`'s names, so `GP.set_priors` never reads it and drops it without an error.
- Consequence if real: any run with `uncertainty_handling=True` and no `specify_target_noise` — MATLAB's "infer noise" mode, the one that goes with repeated observations, where `FunctionLogger` fills `S` with `1` per evaluation and pools it down as `1/sqrt(n_evals)` for repeated points — fits a GP whose observation noise is a single homoskedastic constant. The per-point standard errors that the logger accumulates are computed, stored, passed to `GP.fit` as `s2`, and then ignored; the inferred noise multiplier and its Student-t hyperprior do not exist. My level-1 `train_gp` run fits 9 hyperparameters where MATLAB fits 10. This is a whole supported operating mode of the toolbox behaving differently from the reference, not a corner case; it affects the GP fit, the acquisition, `sn2_hpd` and therefore the reliability index and termination. It does not affect `uncertainty_handling=False` (level 0) or `specify_target_noise=True` (level 2, where `[1,1,0]` maps correctly to `parameters = [1,1,0]`).
- Suggested reproduction: ran it. `check_noise_level1.py` prints the three levels' `GaussianNoise.parameters`; `check_level1_train.py` builds a 12-point level-1 state and reports `noise hyperparameter info: [('noise_log_scale', 1)]` and 9 hyperparameters. The smallest possible check is the two-line construction in `check_noise_level1.py`.
- Test adequacy: no. `test_gp_hyp` (the only test that reaches `_gp_hyp` through `train_gp`) uses `specify_target_noise=True`, i.e. level 2, and asserts only the noise-scale prior location. No test in `pyvbmc/testing` constructs a level-1 run; `test_vbmc_optimize.py`'s noisy runs use `specify_target_noise`. A test asserting `gp.noise.hyperparameter_count() == 2` at level 1, or that the fitted `sn2` responds to `s2`, would catch it.

### F2. The `upper_gp_length_factor` upper bound on the length scales is overwritten before it is used
- Location: `pyvbmc/vbmc/gaussian_process_train.py:368-373` overwritten at `:399-402`; MATLAB: `misc/gptrain_vbmc.m:177-179`
- Category: control flow (dict key assigned twice)
- Proposed classification: suspected defect (Python side)
- Confidence: high
- History: MATLAB's `UB_gp(1:D) = log(options.UpperGPLengthFactor*(PUB-PLB))` was added by `01565e6` ("added upper GP length scale option") and is unchanged in the comparison target. On the Python side the first assignment dates from the original port; the second was introduced on top of it by `5f097094` "Set more conservative lower bounds for GP hyps. (#116)", which did not notice it was replacing the whole tuple.
- What the code does: `bounds` is a plain dict. `_gp_hyp` writes `bounds["covariance_log_lengthscale"] = (-inf, log(factor*(pub-plb)))`, and 26 lines later, inside the `SquaredExponential` branch (which is always taken — PyVBMC hard-wires `gp_cov_fun = 1`), writes `bounds["covariance_log_lengthscale"] = (cov_bounds_info["LB"][:D], np.nan)`. The second assignment replaces the tuple, so the upper bound becomes `NaN`, which `GP.fit` fills with gpyreg's recommended `log(10*width)`. MATLAB sets `UB_gp(1:D)` and `LB_gp(1:D)` independently in a `NaN`-filled vector, so both survive.
- Consequence if real: the option `upper_gp_length_factor` has no effect. Its default is `0` (as MATLAB's `UpperGPLengthFactor`), so a default run is unaffected; a user who sets it gets silently nothing. My check with `upper_gp_length_factor = 10` on a `D = 2` state printed `covariance_log_lengthscale UB=[nan nan]` where the option asks for `[2.303, 2.303]` and the recommended fallback gives `[3.508, 3.619]`.
- Suggested reproduction: ran it (`check_p5.py`, section `[C]`). Set `upper_gp_length_factor > 0`, call `_gp_hyp`, print `gp.get_bounds()["covariance_log_lengthscale"]`.
- Test adequacy: no test sets `upper_gp_length_factor`; `test_gp_hyp` asserts one prior value and no bound.

### F3. `_gp_hyp` writes explicit infinite bounds where MATLAB leaves the bound unset
- Location: `pyvbmc/vbmc/gaussian_process_train.py:370` (`-np.inf` lower length-scale bound), `:375` (`np.inf` noise upper bound), `:382` and `:389` (`-np.inf` lower `mean_const` bound), `:427-429`; MATLAB: `misc/gptrain_vbmc.m:174-180`, `:182-203`, with `gplite/gplite_train.m:120-127`
- Category: defaults (bound propagation)
- Proposed classification: port discrepancy
- Confidence: high (that the bounds differ); medium (on the size of the effect)
- History: these lines are the original 2021 port of this module and have never matched MATLAB. The MATLAB lines are unchanged since before the port.
- What the code does: MATLAB builds `LB_gp` and `UB_gp` as `NaN(1,Nhyp)` and assigns only the entries it means to change; `gplite_train.m:120-127` replaces every remaining `NaN` with the value `gplite_covfun`/`gplite_noisefun`/`gplite_meanfun` recommend from the **full** training set. gpyreg has the same convention (`GP.set_bounds` leaves `NaN` for an unset block; `GP.fit` fills `NaN`s from `get_recommended_bounds`). PyVBMC instead writes a finite-or-infinite pair for each block it touches, so the infinity is *not* a "leave it alone" marker — it overrides the recommendation. On the `D = 2` state of my check:

  | hyperparameter | PyVBMC | what MATLAB would use |
  | --- | --- | --- |
  | `noise_log_scale` UB | `+inf` | `log(max(y)-min(y))` = 1.159 |
  | `mean_const` LB | `-inf` | `min(y)` = −3.556 |

  The `mean_const` bound is the one with a visible effect: `mean_const` carries no hyperprior on either side, and `gpyreg/f_min_fill.py:117-139` branches on `np.isfinite(LB[i]) and np.isfinite(UB[i])`. With both finite (MATLAB) the design draws from the four-point mixture of uniforms `uuinv([LB, PLB, PUB, UB], w)` with `w = 0.5**(1/n_vars)`, so roughly 7% of the design points for that coordinate fall outside the plausible box; with `LB = -inf` (PyVBMC) the branch falls through to `S*(PUB-PLB)+PLB`, confining every point to `[median(y), max(y)]`. For `noise_log_scale`, which does carry a Student-t prior, the design is the prior truncated to `[LB,UB]`, so `+inf` only replaces `t.cdf((log(height)-mu)/sigma, 3)` (≈ 0.9998 at typical scales) with 1 — a small rescaling of that column — but it also removes the hard cap the optimizer and the slice sampler see.
- Consequence if real: the space-filling design that starts every GP fit differs from MATLAB's in at least one coordinate on the default path, and two hyperparameters are unbounded where MATLAB bounds them. This is a systematic, every-iteration difference, though a modest one: it shifts starting points rather than the model.
- Suggested reproduction: ran it (`check_p5.py`, section `[C]`, which prints the installed bounds beside `get_recommended_bounds("recommended","recommended")`). A sharper check would run `f_min_fill` twice on the same state with `mean_const` LB at `-inf` and at `min(y)` and compare the design columns.
- Test adequacy: no. `test_gp_hyp` checks one prior entry. No test inspects `gp.lower_bounds`/`gp.upper_bounds` after `_gp_hyp`.

### F4. The length-scale and output-scale lower bounds are derived from the HPD subset, not the training set
- Location: `pyvbmc/vbmc/gaussian_process_train.py:393-405`; MATLAB: no counterpart (`misc/gptrain_vbmc.m:174-180` sets no covariance lower bound; `gplite/gplite_train.m:74`, `:120` derive it from the full `X`)
- Category: defaults
- Proposed classification: possibly intentional (Python-only change, but absent from the known-differences sheet)
- Confidence: high on the behavior, medium on the disposition
- History: added 2022 by `5f097094` "Set more conservative lower bounds for GP hyps. (#116)"; no MATLAB counterpart at any revision.
- What the code does: `cov_bounds_info` is computed at `:331` from `hpd_X, hpd_y` (the top `hpd_frac = 0.8` of the training set), and the block installs `cov_bounds_info["LB"][:D]` and `["LB"][D]` as the hard lower bounds of the log length scales and log output scale. MATLAB leaves both `NaN`, so `gplite_train` fills them from `gplite_covfun('info', X, ...)` on the **full** training set. The in-code comment states the intent ("wider ... since cov_bounds_info is based on the high-posterior-density region"). On my 12-point state: length-scale LB `[-13.003, -12.499]` against the full-data `[-12.610, -12.499]`, output-scale LB `-12.937` against `-12.656`.
- Consequence if real: the bound is looser than MATLAB's by `log(width_hpd/width_full)`, a fraction of a nat at these sizes. Both are far below where a fitted length scale sits, so the direct effect on the optimum is nil; the visible effect is again through `f_min_fill`, whose uniform-mixture branch uses `LB` for the length scales only if they had no prior (they do have one), and through the truncation `tcdf_lb = t.cdf((LB-mu)/sigma, 3)` for the length scales, which moves by an amount of order 1e-3. Small, but it is a deliberate departure from the reference that the sheet does not record; flagging it so the sheet can be completed or the change revisited.
- Suggested reproduction: ran it (`check_p5.py`, section `[C]`, the two bound tables).
- Test adequacy: no test covers it.

### F5. The window of past GPs used for starting points begins one iteration late when the history has an even length
- Location: `pyvbmc/vbmc/gaussian_process_train.py:136-150`; MATLAB: `misc/gptrain_vbmc.m:38-42`
- Category: indexing
- Proposed classification: port discrepancy
- Confidence: high
- History: MATLAB's `for ii = ceil(numel(stats.gp)/2):numel(stats.gp)` is unchanged since before the port. The Python line, with its comment "Be very careful with off-by-one errors compared to MATLAB in the range here", dates from the original port and has never matched for even `n`.
- What the code does: MATLAB iterates the 1-based range `ceil(n/2) : n`, which in 0-based terms is `range(ceil(n/2)-1, n)`. Python computes `range(ceil((n+1)/2)-1, n)`. For odd `n` the two agree; for even `n` Python starts one index later and collects one GP fewer. Enumerated:

  ```
  n=2: MATLAB [0, 1]        Python [1]
  n=4: MATLAB [1, 2, 3]     Python [2, 3]
  n=6: MATLAB [2, 3, 4, 5]  Python [3, 4, 5]
  n=8: MATLAB [3,4,5,6,7]   Python [4, 5, 6, 7]
  ```

  (`np.size(iteration_history["gp"])` is exactly the number of recorded iterations: `IterationHistory.record` expands the array to `iteration+1` and no further.)
- Consequence if real: on every even-length history the pool of candidate starting hyperparameter vectors handed to `GP.fit` is missing the oldest GP of MATLAB's half-window (one `Ns`-row block out of roughly `n/2` blocks). After `np.unique` and the subsampling at `:152-158` this changes which vectors reach `f_min_fill` and hence, through the sorted design, the optimization's starting points. It never changes the model, only the search; but it changes it on half of all iterations.
- Suggested reproduction: ran it (`check_p5.py`, section `[A]`). Pure arithmetic — no state needed.
- Test adequacy: no. No test exercises the `init_N > 0 and iter > 0` branch of `train_gp` with a populated `iteration_history["gp"]`.

### F6. The historical starting points are subsampled with `ceil` where MATLAB uses `floor`
- Location: `pyvbmc/vbmc/gaussian_process_train.py:152-158`; MATLAB: `misc/gptrain_vbmc.m:43-45`
- Category: indexing/rounding
- Proposed classification: port discrepancy
- Confidence: high
- History: MATLAB `hyp0 = hyp0(:,randperm(N0,floor(gptrain_options.Ninit/2)))`, unchanged since before the port; the Python `math.ceil(gp_train["init_N"] / 2)` dates from the original port.
- What the code does: the guard is the same on both sides (`N0 > Ninit/2`), but the number kept differs by one whenever `init_N` is odd: `init_N = 9` keeps 5 in Python and 4 in MATLAB; `init_N = 63` keeps 32 against 31; `init_N = 65` keeps 33 against 32. The default schedule `f(x) = a x³ + b x² + c x + d` produces odd values routinely (it runs continuously from 1024 down to 64).
- Consequence if real: one extra historical starting point in the pool, on about half the iterations. Same kind of effect as F5 — the search, not the model. `rng.choice(..., replace=False)` is a correct stand-in for `randperm(N0,k)` otherwise (the order differs, but `np.unique`/`unique(...,'rows')` sorts both).
- Suggested reproduction: ran it (`check_p5.py`, section `[B]`).
- Test adequacy: not covered.

### F7. `gp_hyp_full` records the thinned hyperparameter samples; MATLAB records the pre-thinning chain
- Location: `pyvbmc/vbmc/vbmc.py:1610` (producer) read by `pyvbmc/vbmc/gaussian_process_train.py:762` (`_get_hyp_cov`); MATLAB: `misc/gptrain_vbmc.m:65` → `vbmc.m:804` → `vbmc.m:1041`, consumed at `misc/get_GPTrainOptions.m:143`
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high
- History: MATLAB's `hypstruct.full = gpoutput.hyp_prethin` and `stats.gpHypFull{iter} = hyp_full` predate the port (`99bac4d` "save full hyperparameters - thin later - for covariance structure" states the intent explicitly). The Python producer has recorded `self.gp.get_hyperparameters(as_array=True)` since the iteration history was written.
- What the code does: `train_gp` already keeps the right thing — `hyp_dict["full"] = res["samples"]`, gpyreg's pre-thinning chain of `n_samples * thin` rows (`gaussian_process.py:1363`). But the iteration record stores `self.gp.get_hyperparameters(as_array=True)`, the `n_samples` rows that survive `hyp_pre_thin[thin-1::thin, :]`. MATLAB stores `hypstruct.full`, i.e. the pre-thinning chain, and `GetHypCov` builds the weighted hyperparameter covariance from it. Orientation is handled correctly on both sides (MATLAB transposes its `Nhyp x Ns` block; PyVBMC's blocks are already `Ns x Nhyp`) and the per-block weight is normalized by the block's own row count on both sides, so only the content differs.
- Consequence if real: the weighted covariance that sets the slice-sampling widths (`gp_train["widths"] = max(sqrt(diag(hyp_cov)), 1e-3) * width_mult`) is estimated from 5× fewer, less autocorrelated draws than MATLAB's. Both estimate the same posterior covariance, so the difference is sampling error in the widths, not a bias — but the widths differ from MATLAB's on every iteration after the first, and with `gp_sample_widths = 5` they scale the sampler's step. Default-path, small-to-moderate.
- Suggested reproduction: in a short run, compare `vbmc.iteration_history["gp_hyp_full"][i].shape[0]` with `vbmc.hyp_dict["full"].shape[0]`; they should differ by the factor `gp_sample_thin = 5`.
- Test adequacy: no. `test_get_hyp_cov` feeds `gp_hyp_full` directly as synthetic arrays, so it tests the formula (correctly, and against the MATLAB expression) but not what the producer puts there.

### F8. When the fit does not sample, `hyp_dict["full"]` keeps the previous iteration's chain instead of being replaced, and the running covariance is never cleared
- Location: `pyvbmc/vbmc/gaussian_process_train.py:178-203`; MATLAB: `misc/gptrain_vbmc.m:65`, `:83-94`
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high
- History: original port; MATLAB unchanged since before it.
- What the code does: MATLAB assigns `hypstruct.full = gpoutput.hyp_prethin` unconditionally. When `Ns = 0`, `gplite_train.m:465` sets `hyp_prethin = hyp`, a single column, so `size(hypstruct.full,2) > 1` is false and `hypstruct.runcov = []`. PyVBMC writes `hyp_dict["full"]` only inside `if res is not None:`, and `GP.fit` returns `sampling_result = None` exactly when `n_samples == 0` (`gaussian_process.py:1334-1337`). So once sampling stops — which is the normal end state, `stop_sampling` being set at `N >= stable_gp_sampling = 200+10D` with `stable_gp_samples = 0` — `hyp_dict["full"]` silently keeps the last chain that was drawn, and the block at `:195-201` goes on folding that frozen covariance into `run_cov` iteration after iteration, where MATLAB clears `runcov`. Two smaller points in the same block: the guard tests the wrong axis (`hyp_dict["full"].shape[1] > 1` is the hyperparameter count for an `(Ns_eff, Nhyp)` array, where MATLAB's `size(...,2) > 1` is the sample count — the covariance itself, `np.cov(full.T)`, is computed over the right axis), and `hyp_dict["logp"]` goes stale the same way (harmless: `LogP` is commented out at `misc/gptrain_vbmc.m:32` and unused in gpyreg).
- Consequence if real: with the shipped default `weighted_hyp_cov = True`, `_get_hyp_cov` never reads `run_cov`, so results are unaffected — the cost is a stale array carried in `hyp_dict`, deep-copied into `optim_state["hyp_dict"]` and into saved runs. With `weighted_hyp_cov = False` the sampler widths of any later iteration that does sample (e.g. after a warp, or with a raised `stable_gp_sampling`) would come from a covariance MATLAB would have discarded.
- Suggested reproduction: call `train_gp` twice on the same state, the first time with sampling on and the second with `optim_state["stop_sampling"] = 1` (so `gp_s_N = stable_gp_samples = 0`), and check that `hyp_dict["full"]` is unchanged and `hyp_dict["run_cov"]` is not `None`.
- Test adequacy: no test calls `train_gp` twice, and none inspects `hyp_dict["full"]` or `hyp_dict["run_cov"]`.

### F9. The end-of-warm-up reset of the running hyperparameter covariance writes an unread key
- Location: `pyvbmc/vbmc/vbmc.py:1657`; MATLAB: `vbmc.m:831`
- Category: state/caching
- Proposed classification: suspected defect (Python side)
- Confidence: high
- History: MATLAB's `hypstruct.runcov = [];    % Reset GP hyperparameter covariance` is unchanged since before the port. The Python line, with MATLAB's own line quoted as a comment two lines above it (`# hypstruct.runcov = []`), dates from the port of the main loop.
- What the code does: `self.hyp_dict["runcov"] = None` creates a new dictionary entry named `runcov`. Every reader and writer of the running covariance uses `run_cov` (`gaussian_process_train.py:73-74`, `:195-203`, `:724`). So the reset does nothing, and `hyp_dict["run_cov"]` carries the warm-up covariance into the main stage; the `runcov` key is written once and read nowhere.
- Consequence if real: as with F8, inert at the shipped default `weighted_hyp_cov = True`, because `_get_hyp_cov` then ignores `run_cov` entirely. With `weighted_hyp_cov = False` the first post-warm-up iterations would size their slice-sampling widths from the warm-up covariance that MATLAB deliberately discards. The key also ends up in saved `hyp_dict`s.
- Suggested reproduction: `grep -n "runcov" pyvbmc/` returns this single line plus the comment; `"run_cov" in vbmc.hyp_dict and "runcov" in vbmc.hyp_dict` after a run that ends warm-up shows both present.
- Test adequacy: no test asserts anything about `hyp_dict` after warm-up ends.

### F10. The floor of the N-dependent initial design size is 9, where MATLAB's is 0
- Location: `pyvbmc/vbmc/gaussian_process_train.py:639`; MATLAB: `misc/get_GPTrainOptions.m:100`
- Category: defaults
- Proposed classification: port discrepancy
- Confidence: high on the difference, medium on the consequence
- History: MATLAB's `Ninit = max(round(f(x)),0)` is unchanged since before the port (the file's last commit is `1182783`, 2020-05-05). The Python `init_N = max(round(f(x)), 9)` comes from the module's first commit, `aa454736` "feat: added GP training for VBMC (#19)", and never matched.
- What the code does: the cubic `f` runs from `gp_train_n_init = 1024` at `x = 0` to `gp_train_n_init_final = 64` at `x = 1` and keeps falling, crossing zero at about `x = 1.45`. For `x <= 1` the clamp never binds and the two agree. `x` exceeds 1 only once `n_eff` passes `min(max_fun_evals, 1e3)`, so the clamp is reachable only when `max_fun_evals > 1e3`; it binds around `n_eff > 1.45 * 1e3`. There MATLAB sets `Ninit = 0`, which in `gplite_train`/`GP.fit` means no space-filling design at all (the starting points are the supplied `hyp0`, ordered by objective) and, back in `train_gp:136`, also switches off the collection of historical starting points; PyVBMC instead runs a 9-point design and keeps the historical pool.
- Consequence if real: none at the default budget; a different late-run training policy for very long runs (`max_fun_evals` above about 1450).
- Suggested reproduction: `_get_gp_training_options` with `optim_state["n_eff"] = 1600`, `options["max_fun_evals"] = 5000`; `init_N` comes back 9 where MATLAB gives 0.
- Test adequacy: no. `test_get_gp_training_options_opts_N` uses `n_eff = 10`, deep in the region where the clamp is inactive, and asserts only `opts_N`.

### F11. The GP-retrain-threshold branch tests `iteration > 1` where the same function's other test of the same MATLAB predicate uses `iteration > 0`
- Location: `pyvbmc/vbmc/gaussian_process_train.py:651-655`, against `:533-537`; MATLAB: `misc/get_GPTrainOptions.m:109`, against `:5`
- Category: indexing (0-based vs 1-based)
- Proposed classification: port discrepancy, currently without effect
- Confidence: high on the discrepancy, high that it is presently masked
- History: original port; MATLAB unchanged since before it.
- What the code does: `optim_state["iter"]` is MATLAB's `optimState.iter` minus one. `:534` translates MATLAB's `if iter > 1` as `if iteration > 0`, correctly; `:651` translates the same `iter > 1` as `iteration > 1`, one iteration too late. The branch is therefore skipped at Python iteration 1 (MATLAB iteration 2) whatever the reliability index says.
- Consequence if real: none today. `r_index` is `+inf` for the first two iterations on both sides — `private/vbmc_termination.m:113` returns early for `optimState.iter < 3` and `pyvbmc/vbmc/vbmc.py:2242` for `iter < 2` — so `r_index[iteration-1] < gp_retrain_threshold` is false at the iteration in question on both sides, and the first iteration at which the branch can fire is the same (MATLAB 4, Python 3). The condition is nonetheless not MATLAB's, and it would diverge the moment the reliability index became finite earlier.
- Suggested reproduction: set `optim_state["iter"] = 1`, record a finite `r_index` at index 0 below `gp_retrain_threshold`, and compare `init_N`/`opts_N` with the MATLAB expression.
- Test adequacy: no. `test_get_gp_training_options_opts_N` sets `optim_state["iter"] = 2` throughout, so the boundary is never tested.

### F12. Five of the six accepted `gp_hyp_sampler` values cannot run, and the `covsample` fall-through picks a different sampler than MATLAB
- Location: `pyvbmc/vbmc/gaussian_process_train.py:556-617` (especially `:603-604`); MATLAB: `misc/get_GPTrainOptions.m:18-91` (especially `:71-74`) and `gplite/gplite_train.m:316-457`
- Category: control flow / cross-module
- Proposed classification: port discrepancy (partly covered in spirit by the sheet's "gpyreg's `SliceSampler` replaces MATLAB's two samplers", which is about `slicesamplebnd`/`eissample_lite` and does not mention these option values)
- Confidence: high
- History: MATLAB's `switch lower(options.GPHypSampler)` predates the port; the Python transcription is the original one.
- What the code does: `_get_gp_training_options` accepts `slicesample`, `npv`, `mala`, `slicelite`, `splitsample`, `covsample` and `laplace` and passes the chosen name to `GP.fit`, which raises `ValueError("Unknown sampler!")` for anything but `slicesample` whenever `n_samples > 0` (`gaussian_process.py:1345`). So six of the seven values raise rather than fall back. Within that, one branch differs textually from MATLAB: `misc/get_GPTrainOptions.m:71-74`, the `covsample` case with no usable covariance, sets `Widths = []` **and** `Sampler = 'slicesample'`; `:603-604` sets `gp_train["sampler"] = "covsample"`. That is precisely the path taken at iteration 0, when `_get_hyp_cov` returns `None`, so `gp_hyp_sampler="covsample"` raises on the first fit instead of degrading to slice sampling as MATLAB does.
- Consequence if real: a documented option value that MATLAB honors is a hard error; and the one place where MATLAB's own fallback would have rescued it was transcribed away. No effect on a default run.
- Suggested reproduction: construct a `VBMC` with `options={"gp_hyp_sampler": "covsample"}` and call `train_gp`; compare with MATLAB's fall-through. (I did not run this one — it needs a full fit.)
- Test adequacy: the opposite of adequate. `test_get_gp_training_options_samplers` asserts `res8["sampler"] == "covsample"` for exactly the no-covariance case whose MATLAB answer is `'slicesample'`. The test records the implementation.

### F13. The `slicelite` burn-in formula is mis-parenthesized
- Location: `pyvbmc/vbmc/gaussian_process_train.py:657-673`; MATLAB: `misc/get_GPTrainOptions.m:112`
- Category: formula
- Proposed classification: port discrepancy
- Confidence: high
- History: original port; MATLAB unchanged.
- What the code does: MATLAB computes `max(1, ceil(Thin*log(rindex)/log(GPRetrainThreshold))) * Ns_gp`. Python computes `max(1, ceil(thin * log(rindex / log(gp_retrain_threshold)))) * gp_s_N` — the division by `log(threshold)` moved inside the logarithm's argument. The `TODO` comment beside it notes that the default `gp_retrain_threshold = 1` makes `log(1) = 0`, which is a division by zero in MATLAB's form and a division by zero inside the log in Python's.
- Consequence if real: none reachable — the branch requires `gp_hyp_sampler == "slicelite"`, which F12 shows cannot complete a fit. Reporting it so that the formula is not carried forward if the sampler is ever ported.
- Suggested reproduction: evaluate both expressions for, say, `thin=5, rindex=3, threshold=2`: MATLAB gives `ceil(5*1.0986/0.6931) = 8`, Python `ceil(5*log(3/0.6931)) = ceil(7.33) = 8` — they happen to agree there but not in general (`threshold=10`: MATLAB `ceil(5*1.0986/2.3026)=3`, Python `ceil(5*log(3/2.3026))=ceil(1.32)=2`).
- Test adequacy: `test_get_gp_training_options_samplers` selects `slicelite` but never reaches this branch (it needs `recompute_var_post` false and `r_index < gp_retrain_threshold`).

### F14. The output-dependent-noise bound branch raises `TypeError`
- Location: `pyvbmc/vbmc/gaussian_process_train.py:425-429`; MATLAB: `misc/gptrain_vbmc.m:232-243`
- Category: formula / API misuse
- Proposed classification: suspected defect (Python side), unreachable
- Confidence: high
- History: original port; MATLAB unchanged.
- What the code does: `np.min(np.min(y), np.max(y) - 20 * D)` passes the second value as `np.min`'s `axis` argument. Running it on real arrays gives `TypeError: 'numpy.float64' object cannot be interpreted as an integer` (verified). The intended MATLAB line is `LB_gp(Ncov+2) = min(min(y_all), max(y_all) - 20*D)`, i.e. `np.minimum`. Two further details of the same block: MATLAB uses `y_all = optimState.y(optimState.X_flag)` where Python uses the `y` argument, which is the same array; and the `df` entry of the prior is `NaN` for the first of the two parameters where MATLAB leaves `3` (harmless, since the corresponding `mu`/`sigma` are `NaN` and gpyreg's `fit` fills `NaN` `df` with `df_base`).
- Consequence if real: the branch is entered only when `optim_state["gp_noise_fun"][2] == 1`, which `vbmc.py:1082-1100` never sets (`noise_shaping` is rejected at construction, per the known-differences sheet, and it would set index 1, not 2). Unreachable today; it would be an immediate crash if the rectified-linear noise were ever enabled.
- Suggested reproduction: ran it (`check_p5.py`, section `[D]`).
- Test adequacy: not covered, and not coverable while the branch is unreachable.

### F15. The starting-point construction has no equivalent of MATLAB's `try`/`catch` fallback
- Location: `pyvbmc/vbmc/gaussian_process_train.py:135-160`; MATLAB: `misc/gptrain_vbmc.m:37-50`
- Category: control flow
- Proposed classification: possibly intentional
- Confidence: medium
- History: original port; MATLAB unchanged.
- What the code does: MATLAB wraps the whole block that gathers historical hyperparameters and concatenates them with `hypstruct.hyp` in `try ... catch hyp0 = hypstruct.hyp; end`, so a width mismatch between a recorded GP and the current one degrades to "use the summary vector only". PyVBMC's `np.concatenate` would raise. The `:163-164` guard that follows (`if hyp0.shape[1] != np.size(gp.hyper_priors["mu"]): hyp0 = None`) is MATLAB's `:52` and catches only a mismatch that survives the concatenation.
- Consequence if real: PyVBMC's GP model does not change shape mid-run (the mean function is fixed at construction and a warp preserves the hyperparameter count), so I could not construct a reachable case. Reporting it as an unreproduced robustness difference.
- Suggested reproduction: record an `iteration_history["gp"]` entry built with a different mean function and call `train_gp`.
- Test adequacy: not covered.

### F16. `gp.t` (per-point evaluation times) is not carried onto the GP
- Location: `pyvbmc/vbmc/gaussian_process_train.py:958` ("Missing port: gp.t = t_train"), and the same omission in `train_gp`; MATLAB: `misc/gpreupdate.m:8`, `misc/gptrain_vbmc.m:76`
- Category: state/caching
- Proposed classification: possibly intentional (harmless)
- Confidence: high
- History: original port; the code comment records the omission.
- What the code does: both MATLAB entry points attach `t_train` to the GP; `private/activesample_vbmc.m:484` extends it. `_get_training_data` computes `t_train` and both Python callers discard it.
- Consequence if real: none. A repository-wide search for readers of `gp.t` in the MATLAB toolbox finds only the two writes and the one append — nothing consumes it.
- Suggested reproduction: `grep -rn "gp\.t\b" --include=*.m` in the MATLAB checkout (ran it: three hits, all writes).
- Test adequacy: n/a.

### F17. `_estimate_noise` breaks ties in the opposite order from MATLAB's descending sort
- Location: `pyvbmc/vbmc/gaussian_process_train.py:849`; MATLAB: `misc/gptrain_vbmc.m:355`
- Category: indexing
- Proposed classification: port discrepancy, negligible
- Confidence: medium
- History: original port.
- What the code does: `np.argsort(gp.y, axis=None)[::-1]` reverses a stable ascending sort, so among equal `y` values the later index comes first; MATLAB's `sort(y,'descend')` is stable and keeps the earlier index first. Everything else in the function matches line for line (`ceil(0.2*N)`, the per-sample `gplite_noisefun` call over the `Ncov+(1:Nnoise)` block, `median(mean(sn2,2))`). `pyvbmc/stats/get_hpd.py` has the same construction and the same tie behavior against `misc/gethpd_vbmc.m`, which is otherwise an exact match.
- Consequence if real: only when exactly-equal `y` values straddle the `hpd_N` cut, which needs duplicated log-densities. Then one training point in the HPD subset differs and `sn2_hpd` moves slightly.
- Suggested reproduction: `_estimate_noise` on a GP whose `y` has a repeated value at the cut.
- Test adequacy: `test_estimate_noise` uses distinct values.

---

## 3. Test adequacy notes

- **`test_get_gp_training_options_samplers`, `res8`** (`pyvbmc/testing/vbmc/test_gaussian_process_train.py:466-473`) asserts `sampler == "covsample"` for the `covsample`-with-no-covariance case. `misc/get_GPTrainOptions.m:71-74` answers `'slicesample'`. This assertion records the implementation and locks in F12.
- **`test_get_gp_training_options_opts_N`** pins `optim_state["iter"] = 2` for all four cases, so the `iteration > 1` boundary of F11 is never crossed, and `n_eff = 10` keeps the cubic schedule far from the clamp of F10. It asserts `opts_N` only; `init_N`, `burn` and `thin`, all of which the same function sets, are never checked against MATLAB.
- **`test_gp_hyp`** is the only test that runs `train_gp` end to end in this module, and it asserts exactly one number: `priors["noise_log_scale"][1][0] == log(tol_gp_noise)`. Nothing about the bounds `_gp_hyp` installs, the length-scale prior, `gp_s_N`, or the noise-function shape. It runs at `uncertainty_handling_level == 2`, so the level-1 path of F1 is untouched by the suite.
- **`test_get_hyp_cov` / `test_get_hyp_cov_cutoff_and_degenerate_history`** are the good ones: their expected covariance is derived from the MATLAB weighting expression rather than from the code, and the comments spell out which `sKL` entries the decay reads in which order. I checked the index arithmetic against `misc/get_GPTrainOptions.m:134-161` line by line and it is correct, including `skl_index = iter - i` and `history_index = iter - 1 - i`. Their blind spot is the producer: they feed `gp_hyp_full` as synthetic arrays, so F7 (the recorded array being the thinned chain) is invisible to them.
- **`test_estimate_noise`**, **`test_get_training_data_*`**, **`test_meanfun_name_to_mean_function`**, **`test_cov_identifier_to_covariance_function`** are faithful but shallow; none of them would notice a bound or a policy change.
- **No test at all** covers `_lean_gp`, `_restore_gp_posteriors` (beyond `test_gp_records.py`, which I read and which does check that a restored GP's factors equal the dropped ones), `reupdate_gp`, or the `run_cov` update block. I verified `_lean_gp` by reading: `copy.copy` followed by `update(hyp=..., compute_posterior=False)` assigns a fresh `posteriors` array on the copy and touches nothing the copy shares with the live GP, and `IterationHistory.record` deep-copies afterwards, so the sharing is resolved; `_restore_gp_posteriors` reruns the same `__core_computation` path that `fit` used, so the rebuilt factors are the dropped ones. I found no defect in either.

---

## 4. Defects on the MATLAB side alone

1. **`misc/get_GPTrainOptions.m:112`** divides by `log(options.GPRetrainThreshold)`, which is `log(1) = 0` at the shipped default, so the `slicelite` burn-in is `Inf` or `NaN` whenever that branch is reached with the default threshold. (PyVBMC's transcription is separately wrong — F13 — but the MATLAB expression is itself unsafe; the Python `TODO` comment at `:658-659` notices this.)
2. **`misc/gptrain_vbmc.m:19-25`**: the `if optimState.Warmup && options.BOWarmup` branch and its `else` call `vbmc_gphyp` with identical arguments, so the condition does nothing. Not ported, and nothing is lost by that.
3. **`misc/get_GPTrainOptions.m:151-161`**: when no historical block is accepted, `w_list` is empty, `w_list/sum(w_list)` is `0/0`, `mustar` is empty and `hypcov` comes back a `0x0` matrix rather than `[]`; the caller's `~isempty(hypcov)` test then passes and `sqrt(diag(hypcov)')` yields an empty `Widths`. PyVBMC returns `None` instead, which the known-differences sheet records as deliberate.
4. **`misc/get_GPTrainOptions.m:61`**: `all(rindex < options.CovSampleThresh)` is applied to a scalar `rindex` that can be `Inf`; combined with `all(isfinite(widthmult))` on the same scalar the two tests are redundant. Cosmetic.
