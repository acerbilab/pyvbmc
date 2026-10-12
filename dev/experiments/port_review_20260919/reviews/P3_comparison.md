# P3 — acquisition functions, MATLAB-comparison track

Raw reviewer report, wave 4 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, fresh
context, read-only on the repositories, reading the code at `f873556`; it
compared the Python code with MATLAB VBMC at `396d649`. Its brief carried a
first question on uncertainty level 1, which the report answers before its
findings. Its check scripts ran in the session scratchpad and are kept only
on the machine that ran them (`dev/scripts/runs/LOCAL.md`). The text below
is the agent's final message, unedited.

---

# P3 — acquisition functions (MATLAB comparison track)

## 1. Coverage

**Read completely (Python).** Every file of `pyvbmc/acquisition_functions/`:
`abstract_acq_fcn.py`, `acq_fcn.py`, `acq_fcn_log.py`, `acq_fcn_vanilla.py`,
`acq_fcn_noisy.py`, `acq_fcn_viqr.py`, `acq_fcn_imiqr.py`, `utilities.py`,
`__init__.py`, `README.md`. Out of slice but read completely for the
interfaces: `pyvbmc/vbmc/active_importance_sampling.py`,
`pyvbmc/function_logger/function_logger.py`, the sn2/length-scale/search
block of `pyvbmc/vbmc/active_sample.py` (lines 300–500 and 760–875), the GP
setup block of `pyvbmc/vbmc/vbmc.py` (900–950, 1020–1132, 1410–1430),
`_get_training_data` and `_gp_hyp`'s noise section in
`pyvbmc/vbmc/gaussian_process_train.py`, `gpyreg/noise_functions.py`,
`gpyreg/covariance_functions.py: SquaredExponential.compute`,
`gpyreg/gaussian_process.py: predict` and `__core_computation`'s Cholesky
branch.

**Read completely (MATLAB).** `acq/acqwrapper_vbmc.m`, `acq/acqf_vbmc.m`,
`acq/acqflog_vbmc.m`, `acq/acqus_vbmc.m`, `acq/acqfsn2_vbmc.m`,
`acq/acqviqr_vbmc.m`, `acq/acqimiqr_vbmc.m`, `acq/acqeig_vbmc.m`,
`misc/real2int_vbmc.m`, `utils/sq_dist.m`,
`private/activeimportancesampling_vbmc.m`, `misc/funlogger_vbmc.m`,
`misc/get_traindata_vbmc.m`, `gplite/gplite_noisefun.m`,
`gplite/gplite_pred.m`; `private/activesample_vbmc.m` lines 1–250 and the GP
update block; `misc/setupvars_vbmc.m` and `misc/gptrain_vbmc.m` noise
sections; `misc/setupoptions_vbmc.m:127–165`; `misc/boundscheck_vbmc.m`
bound checks.

**History.** `git log` on `acq/`, `misc/real2int_vbmc.m`, `utils/sq_dist.m`,
`private/activeimportancesampling_vbmc.m` since 2021-01-19 (only `a9615ba`,
2022-07-23, touches the slice; its full diff read), and `git log -L` on the
Python `self.u` and `renormalize_weights` lines.

**Skimmed.** `misc/noiseshaping_vbmc.m` and `misc/intkernel.m` (confirmed
unported, as the sheet says: no `noiseshaping` call reaches PyVBMC's sn2
block — `active_sample.py:343` marks the gap, construction rejects
`noise_shaping=True`; `intkernel` has no Python counterpart and its only
caller, `acqeig_vbmc.m`, has no port — `acq_fcn_eig.py` is absent and
`AcqFcnEIG` appears nowhere under `pyvbmc/`). Tests:
`pyvbmc/testing/acquisition_functions/` (all 15 files),
`pyvbmc/testing/oracles/_oracles.py`, `_state.py`.

**Not reached.** The interior of `vp.pdf`, the slice sampler, and the details
of `active_importance_sampling`'s proposal (P4's slice) beyond what the
acquisitions consume.

**Checks run** (all in the scratchpad, `OMP/OPENBLAS/MKL_NUM_THREADS=1`,
`pyvbmc.__file__` and `gpyreg.__file__` both the checkouts under review):

1. `matlab_ref.py` — a line-by-line Python transcription of
   `acqwrapper_vbmc.m`, `acqf`, `acqflog`, `acqus`, `acqfsn2`, `acqviqr`,
   `acqimiqr`, `real2int_vbmc.m` and the inlined `sq_dist`, written from the
   MATLAB source, with MATLAB shapes and MATLAB's `round` and NaN-dropping
   `max`.
2. `compare.py` — that transcription against the shipped classes on all 8
   oracle fixture states (512 candidates each, `Ns_gp` 1–10, `D` 2–5).
   Result: `AcqFcn`, `AcqFcnVanilla`, `AcqFcnNoisy` **bit-identical**;
   `AcqFcnLog` max 1.4e-14 absolute (1.2e-16 relative, the `log(max(p,·))`
   vs `max(log p, ·)` ordering); `AcqFcnVIQR` 1.7e-5, `AcqFcnIMIQR` 8.0e-5;
   same argmin everywhere; identical non-finite patterns.
3. `u_const.py` — re-ran 2 with `u = norm.ppf(0.75)` in the transcription:
   VIQR residual 1.9e-14, IMIQR 1.6e-14. The whole VIQR/IMIQR residual is
   the `u` constant (F2).
4. `level1.py` — converted the noisy fixture into an
   `uncertainty_handling_level = 1` state (noise function `[1 2 0]`,
   `S = 1/sqrt(n_evals)`, a fresh GP carrying the extra log-multiplier) and
   compared `sn2_new`, `gp_length_scale` and all six acquisitions against an
   independent transcription of `activesample_vbmc.m:159–183` with
   `gplite_noisefun.m` inlined. `sn2_new` and `gp_length_scale`
   bit-identical; the four non-IQR acquisitions bit-identical; VIQR/IMIQR
   the same `u`-only residual as at level 2.
5. `renorm.py` — the size and shape of the log-weight renormalization PyVBMC
   applies and MATLAB does not (F4).
6. `reuse.py` — `AcqFcnVIQR` with and without the
   `return_cross_covariance=True` kernel reuse on the noisy fixture:
   bit-identical.
7. A three-line `_real2int` demonstration of the rounding convention (F3).

No test suite or `optimize()` run; nothing written inside the three
repositories.

## 2. First question — the observation noise at each uncertainty level

**Answer: yes. At all three levels, including level 1, every quantity in the
slice that needs an observation noise is computed from the same inputs, with
the same hyperparameters, by the same formula as MATLAB's, and the results
are bit-identical on a realistic state.** The `095c82c` fix put the last
missing piece (the noise *function*) in place; nothing downstream of it
needed a second change, because the downstream code never special-cases the
level — it reads whatever `gp.noise` and `function_logger.S` hold, exactly as
MATLAB reads `gp.noisefun` and `optimState.S`.

### What each level records and fits (established from the source)

| | level 0 | level 1 | level 2 |
|---|---|---|---|
| `function_logger.S` exists (`function_logger.py:222`) | no | yes | yes |
| SD recorded per evaluation (`:297–304`, `:539–549`) | `None` | `f_sd = 1` | caller's |
| pooling of a repeat (`:680–724`) | `(N·y+y')/(N+1)`, S untouched | precision pooling, `S → 1/sqrt(n)` | precision pooling |
| `optim_state["gp_noise_fun"]` (`vbmc.py:1094–1102`) | `[1,0,0]` | `[1,2,0]` | `[1,1,0]` |
| `gp.s2` for the fit (`gaussian_process_train.py:761–764`) | `None` | `S²` (= 1/n) | `S²` |
| noise start/prior (`:343–363`, `:421–429`) | size `max(NoiseSize,TolGPNoise)`, sd 0.5 | size `TolGPNoise`, sd `log 10`; multiplier `1` (or `max(NoiseSize,TolGPNoise)`), sd `log 10` (or `log 10 / 2`) | size `TolGPNoise`, sd 0.5 |

MATLAB counterparts: `misc/funlogger_vbmc.m:40,51,102–110,234–241`,
`misc/setupvars_vbmc.m:277–281`, `misc/get_traindata_vbmc.m:8–12`,
`misc/gptrain_vbmc.m:142–165`. Every cell matches, including MATLAB's
`max([],MinNoise) → []` idiom, which `gaussian_process_train.py:350` renders
as `if options["noise_size"] != []`.

### Where a candidate's noise comes from, on both sides

Both toolboxes compute the noise **of a hypothetical new single observation
at each training point**, once per active-sampling step, and then give a
candidate the value of its nearest training point.

- **Python** (`active_sample.py:327–363`): for each posterior sample `s`,
  `hyp_noise = gp.posteriors[s].hyp[cov_N : cov_N+noise_N]`;
  `s2 = S[X_flag]² · n_evals[X_flag]` when the logger has `S`, else `None`;
  `sn2new[:,s] = gp.noise.compute(hyp_noise, gp.X, gp.y, s2)`;
  `gp.temporary_data["sn2_new"] = sn2new.mean(1)`;
  `gp_length_scale = exp(mean_s hyp[:D])`;
  `X_rescaled = gp.X / gp_length_scale`.
- **MATLAB** (`private/activesample_vbmc.m:159–183`): the same five lines,
  `s2 = (optimState.S(X_flag).^2).*optimState.nevals(X_flag)` guarded by
  `isfield(optimState,'S')` — which `funlogger_vbmc.m:51` makes true exactly
  when `UncertaintyHandlingLevel > 0`, the same condition as Python's
  `hasattr(function_logger, "S")`. `gp.sn2new = mean(sn2new,2)`.

The `· n_evals` factor is the point of the construction: at level 1, `S²` is
`1/n` after `n` repeats and `S²·n = 1`, so `s2` is the per-observation
variance rather than the pooled one — the right input for a look-ahead. Both
sides do this identically, and both feed the *pooled* `S²` (without the
factor) to the GP fit.

The noise function then evaluates, per level:

| | MATLAB `gplite_noisefun.m:177–194` | gpyreg `noise_functions.py:248–266` |
|---|---|---|
| 0, `[1 0]`/`[1,0,0]` | `exp(2h₀)` (s2 defaulted to 0) | `exp(2h₀)` (s2 `None` → 0) |
| 1, `[1 2]`/`[1,2,0]` | `exp(2h₀) + exp(h₁)·s2` | `exp(2h₀) + exp(h₁)·s2` |
| 2, `[1 1]`/`[1,1,0]` | `exp(2h₀) + s2` | `exp(2h₀) + s2` |

Identical, including the hyperparameter ordering (`h₀` the log constant SD,
`h₁` the log multiplier) and the fact that the posterior's `sn2_mult` is
**not** applied here on either side (it is applied only inside
`gplite_pred`/`GP.predict` when noise is requested).

Consumers, all in the slice:

- `AbstractAcqFcn._estimate_observation_noise` (`abstract_acq_fcn.py:341–348`)
  = `acqfsn2_vbmc.m:10–11` = `acqviqr_vbmc.m:39–40` = `acqimiqr_vbmc.m:40–41`:
  `argmin` of `sq_dist(Xs/gp_length_scale, X_rescaled)` over axis 1, then
  `sn2_new[pos]`. MATLAB's `min(...,[],2)` and NumPy's `argmin` both return
  the first minimum. `_sq_dist` reproduces the size-weighted centering of the
  inlined MATLAB `sq_dist` exactly.
- `AcqFcnNoisy` (`acq_fcn_noisy.py:44`): `-vtot·(1 - sn2/(vtot+sn2))·exp(fbar-z)·p`,
  character for character `acqfsn2_vbmc.m:16`.
- VIQR/IMIQR (`acq_fcn_viqr.py:300–301`, `acq_fcn_imiqr.py:81–82`):
  `y_s2 = f_s2 + sn2[:,None]`, per hyperparameter sample, the latent
  predictive variance plus the nearest-neighbour observation noise and
  nothing else — `acqviqr_vbmc.m:42`, `acqimiqr_vbmc.m:43`.
- The variance regularization (`abstract_acq_fcn.py:154–175`) uses `var_tot`,
  which carries **no** observation noise on either side
  (`acqwrapper_vbmc.m:21–29` vs `abstract_acq_fcn.py:94–109`), and
  `gp.predict(separate_samples=True)` returns the latent `f_s2` just as
  `gplite_pred(gp,Xs,[],[],1,0)` returns `fs2`, not `ys2`.

The one place a noise-inclusive variance is wanted is the MCMC proposal
density: `AcqFcnVIQR.is_log_full`/`AcqFcnIMIQR.is_log_full` call
`gp.predict(..., add_noise=True)`, and MATLAB's `log_isbasefun`
(`activeimportancesampling_vbmc.m:348`) writes `[fmu,fs2] = gplite_pred(gp,x)`,
which takes the *first two* outputs — `ymu` and `ys2`, the observed variance.
The port gets this right.

**Verification (check 4).** On the level-1 state, `sn2_new` and
`gp_length_scale` are bit-identical to the independent transcription
(`sn2_new = 1.0000366466` at every training point, i.e.
`exp(2h₀) + exp(h₁)·1` with `h₁ = 0`), and `AcqFcn`, `AcqFcnLog`,
`AcqFcnVanilla`, `AcqFcnNoisy` are bit-identical; VIQR/IMIQR differ only by
the `u` constant of F2. The level-2 results are the same. Level 0 is covered
by the six noiseless fixtures in check 2 (there `s2` is `None` on both sides
and `gp.noise.compute` returns a scalar that broadcasts, matching MATLAB's
scalar assignment into a column).

## 3. Findings

### F1. `AcqFcnVIQR` still offers the `iqr_reduction` loss, which the known-differences sheet says was removed
- Location: `pyvbmc/acquisition_functions/acq_fcn_viqr.py:87–112` (docstring),
  `:120` (`LOSSES = ("iqr", "iqr_reduction")`), `:122–134`, `:364–370`,
  `:384–386`, `:396–440` (`_log_iqr_reduction`); tests
  `pyvbmc/testing/acquisition_functions/test_acq_fcn_viqr_losses.py:133–200`;
  MATLAB: no counterpart.
- Category: cross-module (sheet contradiction)
- Proposed classification: possibly intentional (the sheet entry is wrong, not the code)
- Confidence: high
- History: MATLAB never had any of these losses. The four variants were added
  by `014b287`; `fd9c7e8e` (2026-09-14) removed `var_reduction` and
  `sd_reduction` and `AcqFcnEIG`, and its own commit message says
  "`AcqFcnVIQR` offers the losses `iqr` and `iqr_reduction`".
- What the code does: the sheet's P3 entry "The EIG acquisition and the
  experimental VIQR losses were removed" lists `iqr_reduction` among the
  variants "retained on the branch `retain/experimental-acquisitions`" and
  states "`AcqFcnVIQR` offers only the standard `loss=\"iqr\"`". It does not.
  `iqr_reduction` is a documented, validated, publicly reachable loss
  (`AcqFcnVIQR(loss="iqr_reduction")`, and through `string_to_acq` from an
  `.ini` or an options dict), with five of its own tests. `AcqFcnEIG` and
  `acq_fcn_eig.py` are indeed gone, and `var_reduction`/`sd_reduction` are
  gone; the entry over-states the removal by one member.
- Consequence if real: a reviewer told "only `iqr` exists" will not examine
  `_log_iqr_reduction`, a Python-only ~45-line log-space computation with no
  MATLAB original, which is reachable from user configuration. I did read it:
  the identity `sinh x − sinh y = 2 cosh((x+y)/2) sinh((x−y)/2)` and its
  log-space form at `:430–439` are correct, and `d = tau2/(s_a+s_pred)` equals
  `s_a − s_pred` exactly whenever the `max(·,0)` clip is inactive (it can only
  be active at rounding level, since `tau2 ≤ f_s2_a` mathematically). No
  defect found there.
- Suggested reproduction: `grep -n "iqr_reduction" pyvbmc/acquisition_functions/acq_fcn_viqr.py`.
  Done.
- Test adequacy: `test_acq_fcn_viqr_losses.py` covers the loss thoroughly
  against brute-force `predict_full` and an exact algebraic identity with the
  default VIQR. The gap is in the sheet, not the tests.

### F2. `u` is `norm.ppf(quantile)` where MATLAB hard-codes `0.6745`
- Location: `pyvbmc/acquisition_functions/acq_fcn_viqr.py:136`,
  `pyvbmc/acquisition_functions/acq_fcn_imiqr.py:27`
  (`self.u = norm.ppf(quantile)`); MATLAB: `acq/acqviqr_vbmc.m:4`,
  `acq/acqimiqr_vbmc.m:4` (`u = 0.6745; % norminv(0.75)`)
- Category: defaults
- Proposed classification: possibly intentional (not on the sheet)
- Confidence: high
- History: the MATLAB lines have not changed since before 2021-01-19. The
  Python lines *did* match: the original port (`e38351ab`, "Noisy
  likelihoods") wrote `self.u = 0.6745  # inverse normal cdf of 0.75`;
  `70325a34` ("Noisy likelihoods, small fixes", #80) replaced it with
  `norm.ppf(quantile)` when it added the Python-only `quantile` argument.
- What the code does: `norm.ppf(0.75) = 0.6744897501960817`, which differs
  from MATLAB's rounded literal by 1.52e-5 relative. Both VIQR and IMIQR use
  `u` in the acquisition (`u·s_pred`, `sinh(u·s)`) and in the
  importance-sampling proposal (`is_log_added`), so the difference is
  systematic. MATLAB's own comment names `norminv(0.75)` as the intent, so
  PyVBMC is arguably the more faithful of the two; it is nevertheless a
  numerical difference from the comparison target that the sheet does not
  record, and the `quantile` argument itself has no MATLAB counterpart.
- Consequence if real: on the noisy oracle state, the acquisition values move
  by up to 1.7e-5 (VIQR) and 8.0e-5 (IMIQR) absolute, 3.7e-6 / 5.0e-5
  relative. The sieve argmin was unchanged on all 512 candidates; a
  perturbation of this size can in principle send the CMA-ES refinement to a
  different point, as the `active_sample_step` oracle's own re-baselining
  rule acknowledges for few-ulp changes.
- Suggested reproduction: ran `u_const.py` — substituting `0.6745` into the
  MATLAB transcription reproduces the full residual; substituting
  `norm.ppf(0.75)` drops it to 1.6e-14 (float rounding).
- Test adequacy: no. `test_acq_fcn_viqr.py:40` and
  `test_acq_fcn_imiqr.py:41` assert `sps.norm.cdf(acqf.u) ≈ 0.75`, which is
  the implementation's own definition; they would pass either way and would
  fail against MATLAB's literal at their own `np.isclose` default tolerance
  only marginally (`norm.cdf(0.6745) = 0.7500029`, so in fact they would
  still pass). The definition tests (`test_complex__call__`) use
  `u = sps.norm.ppf(0.75)` for their reference, mirroring the implementation.

### F3. `_real2int` rounds half-to-even; MATLAB's `round` rounds half away from zero
- Location: `pyvbmc/acquisition_functions/abstract_acq_fcn.py:281`
  (`np.around`); MATLAB: `misc/real2int_vbmc.m:7` (`round`)
- Category: formula (rounding convention)
- Proposed classification: port discrepancy
- Confidence: high (that they differ); low (that it can be reached in a run)
- History: the MATLAB line has not changed since before 2021-01-19. The
  Python line has used `np.around` since the acquisition functions were first
  added (`9d1b6590`), so it never matched.
- What the code does: NumPy's `around` is banker's rounding
  (`0.5 → 0`, `1.5 → 2`, `2.5 → 2`, `-0.5 → -0`), MATLAB's `round` is
  half-away-from-zero (`0.5 → 1`, `1.5 → 2`, `2.5 → 3`, `-0.5 → -1`). The
  sheet's P2 entry on `_real2int` records the in-place mutation and nothing
  about the rounding rule.
- Consequence if real: only when an inverse-transformed candidate coordinate
  of an integer variable lands exactly on a half-integer. PyVBMC and MATLAB
  both require an integer variable's hard bounds to sit at half-integers and
  to be finite, so the coordinate passes through a probit (or logit)
  transform and an exact `.5` has probability zero from the sieve draws and
  from the CMA-ES/Nelder-Mead search. I found no path that produces one
  systematically. Where it does occur, the candidate snaps to a different
  integer and the whole acquisition value changes.
- Suggested reproduction: ran it — with an identity transformer and
  `X[:,0] = [0.5, 1.5, -0.5, 2.5]`, `_real2int` gives `[0, 2, -0, 2]` where
  MATLAB gives `[1, 2, -1, 3]`.
- Test adequacy: no — and worse. `test_abstract_acquisition_function.py:445–477`
  (`test_real2int`) uses exactly this input (`np.ones((10,3)) * 0.5`) and
  writes `np.all(X_after[:, 0] == 0)` as a **bare expression, not an
  assertion** (three times, lines 471–473, and twice more at 477). The test
  asserts nothing at all; if the missing `assert` were supplied, the
  expectation encoded (`== 0`) is NumPy's convention, not MATLAB's. This is
  the exact pattern the review exists to find.

### F4. PyVBMC renormalizes the importance log-weights; MATLAB does not
- Location: `pyvbmc/vbmc/active_importance_sampling.py:329` and `:497–499`
  (`renormalize_weights`), consumed by
  `pyvbmc/acquisition_functions/acq_fcn_imiqr.py:152–154` and
  `acq_fcn_viqr.py:368`; MATLAB:
  `private/activeimportancesampling_vbmc.m:247–278` (no counterpart; `lnw` is
  stored raw)
- Category: cross-module (state/caching)
- Proposed classification: possibly intentional
- Confidence: high (that it differs); high (that it does not move the
  minimizer)
- History: the MATLAB file's last change is `a9615ba` (2022-07-23), which did
  not add any normalization. `renormalize_weights` was added by `70325a34`,
  the same commit as F2. The Python line never matched MATLAB.
- What the code does: `ln_weights` is shifted by `−logsumexp` over *both*
  axes, so IMIQR's `acq[:,s] = logsumexp_a(lnw_a + log 2 sinh(u s_pred))`
  loses a constant. On the noisy oracle state the constant is 6.1575 nats:
  MATLAB's IMIQR values there would be about +4.33 where PyVBMC reports
  −1.83. VIQR is unaffected (its `iqr` loss does not read the weights, as
  MATLAB's commented-out `lnw` lines record); the Python-only
  `iqr_reduction` loss reads them and documents the constant at
  `acq_fcn_viqr.py:106–110`.
- Consequence if real: none for the search. The shift is the same for every
  candidate, and the only thing added to a subset of candidates afterwards
  (the variance-regularization penalty `TolVar/vtot − 1`) is additive, so
  candidate differences — and therefore the argmin, the CMA-ES trajectory and
  the `f_val_optim < f_val_old` acceptance — are invariant. It does change
  the number a user or a log sees, and it makes MATLAB's and PyVBMC's IMIQR
  values incomparable by value. The module is another reviewer's slice; I
  report it because the effect lands in mine and it is not on the sheet.
- Suggested reproduction: ran `renorm.py` — patching `renormalize_weights` to
  the identity shifts every finite IMIQR value by exactly 6.157535651862 and
  leaves the argmin at candidate 318; VIQR is bit-identical either way.
- Test adequacy: no — `test_acq_fcn_imiqr.py:185` asserts
  `np.isclose(np.sum(np.exp(ln_weights[0])), 1.0)`, which is an assertion
  *of* the Python-only normalization.

## Minor observations (not full findings)

- **NaN handling of `max(acq,-realmax)`.** `abstract_acq_fcn.py:178` uses
  `np.maximum`, which propagates NaN; MATLAB's two-argument `max`
  (`acqwrapper_vbmc.m:47`) drops it and returns `-realmax`. Downstream,
  `np.argmin` on an array containing NaN returns the first NaN's index and
  MATLAB's `min` returns the first `-realmax`, so the same degenerate
  candidate is normally selected either way; the indices can differ only if
  several such candidates exist. Confidence low that this is ever reachable.
- **`-inf` guards in the log sums.** `acq_fcn_viqr.py:68`, `:389` and
  `acq_fcn_imiqr.py:164`, `:172` reset an all-`-inf` row's `lnmax` to 0 to
  avoid `-inf + inf`. MATLAB has no guard and produces NaN there, which its
  `max(acq,-realmax)` then turns into `-realmax`. Python yields `-inf`, which
  the same line turns into `-realmax`. Equivalent outcome; Python-only,
  benign.
- **`acqwrapper`'s `vp.delta` quadrature branch** (`acqwrapper_vbmc.m:12–14`,
  `gplite_quad`) has no Python counterpart; PyVBMC always predicts. Already
  covered by the sheet's P6 entry "GP smoothing (`Bandwidth`, `vp.delta`) is
  not ported".
- **Ordering of the search draws.** `active_sample.py` calls
  `_get_search_points` (which draws) *before* `active_importance_sampling`
  (which also draws); `activesample_vbmc.m` does the reverse (lines 208–218).
  Covered in substance by the sheet's "no Python draw sequence can be
  compared point by point with a MATLAB draw sequence"; belongs to P2/P4.
- **`string_to_acq`** (`utilities.py`) is Python-only (MATLAB uses
  `str2func` on `'@acqf_vbmc'`); it splits the argument string on `,` and
  `=`, so a keyword whose literal value contains either would be mis-parsed.
  No shipped configuration does.

## 4. Test adequacy notes

- `test_abstract_acquisition_function.py:445–477` (`test_real2int`) contains
  **no assertions**: five bare `np.all(...)` expressions. It exercises
  exactly the input where NumPy and MATLAB rounding disagree (F3) and, read
  as an intended expectation, encodes NumPy's answer. This is the clearest
  case in the slice of a test written from the implementation.
- `test_acq_fcn_viqr.py:40`, `test_acq_fcn_imiqr.py:41`
  (`sps.norm.cdf(acqf.u) ≈ 0.75`) restate the implementation's own definition
  of `u` rather than MATLAB's constant; the definition tests supply
  `u = sps.norm.ppf(0.75)` to their own references, so the whole module is
  self-consistent under either constant (F2).
- `test_acq_fcn_imiqr.py:185` asserts that the importance weights sum to one,
  which is true only because of the Python-only renormalization (F4).
- `test_acq_fcn.py`, `test_acq_fcn_log.py`, `test_acq_fcn_vanilla.py` and
  `test_acq_fcn_noisy.py::test__call__` mock `GP.predict` and `vp.pdf` with
  arrays of ones and assert a hand-computed constant; they pin arithmetic the
  implementation could not get wrong in a way that matters. The one real
  gate among them is
  `test_acq_fcn_noisy.py::test_complex__call__:129–137`, whose expected
  vector is labelled MATLAB-derived.
- Good, specification-based tests, worth saying so: `_look_ahead.py` +
  `test_acq_fcn_viqr.py::test_complex__call__` and
  `test_acq_fcn_imiqr.py::test_complex__call__` check the look-ahead
  predictive variance against `gp.predict_full` on the joint covariance and
  against Gauss-Hermite quadrature of the defining integral — a real
  specification, independent of the code under test.
  `test_viqr_kernel_reuse.py` and `_regularization.py` (batch vs pointwise)
  are internal-consistency checks and cannot detect a shared error, but that
  is their stated purpose.
- No test in the slice compares `sn2_new` at `uncertainty_handling_level = 1`
  against anything: `_scenario.py` and `_regularization.py` build GPs with
  `constant_add`(+`user_provided_add`) noise, `prepare_optim_state` never
  goes through the level, the `rosenbrock_D2_noise1_viqr` oracle is level 2,
  and `test_acq_fcn_noisy.py` hand-plants `sn2_new`. The level-1 path
  remains untested downstream of `095c82c`, even though (per §2) it is
  correct.

## MATLAB-side defects noticed

1. **`misc/funlogger_vbmc.m:244`** — in the duplicate branch,
   `optimState.y(optimState.Xn) = fval;` writes the pooled value into the
   *last newly added* row instead of `optimState.y(idx)`. The repeated
   point's own `y` keeps its stale value and an unrelated row is corrupted.
   This reaches my slice through `z = optimState.ymax`
   (`acqf_vbmc.m:10`, `acqflog_vbmc.m:17`, `acqfsn2_vbmc.m:13`) and the GP
   training targets (`get_traindata_vbmc.m:7`). It triggers only when a point
   is evaluated twice, i.e. with `MaxRepeatedObservations > 0` (default 0) or
   a cached/duplicate design point. PyVBMC's
   `function_logger.py:738` writes `self.y[idx]` and is correct. This is P8's
   slice and may already be in `matlab_side_defects.md`.
2. **`acq/acqviqr_vbmc.m:26–28`** — `case 'islogf'` computes
   `acq = vp + u*fs + log1p(...)`, where `vp` is the function's second
   positional argument. `activeimportancesampling_vbmc.m:350` calls it as
   `acqfun('islogf',[],[],[],fmu,fs2)` for VIQR (`importance_sampling_vp` is
   false), so `vp` is `[]` and the whole expression evaluates to `[]`. The
   branch is unreachable today because VIQR does not set
   `mcmc_importance_sampling`; if it were enabled the MCMC target density
   would be empty. PyVBMC's `AcqFcnVIQR.is_log_full` returns the added term
   with a zero base, which is what the code intends.
3. **`acq/acqviqr_vbmc.m:107–108`, `acq/acqimiqr_vbmc.m:93–94`** — when every
   entry of a row of `zz` is `-Inf`, `lnmax` is `-Inf` and
   `exp(zz - lnmax)` is `exp(NaN)`, so `acq` becomes NaN; line 47 of
   `acqwrapper_vbmc.m` then turns it into `-realmax`, the most attractive
   value for a minimizer. PyVBMC's guards give `-inf`, which the same line
   also turns into `-realmax`, so the outcome agrees, but MATLAB reaches it
   through a NaN.
