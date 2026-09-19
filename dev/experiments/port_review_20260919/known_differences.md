# Known differences between PyVBMC/gpyreg and MATLAB VBMC

This sheet lists the *settled, deliberate* differences between the Python
code and the MATLAB original. Its purpose is narrow: a reviewer who finds
one of these should not report it as a new finding.

**Every entry is a claim a reviewer may challenge.** The sheet says what not
to report as new, not what is beyond question. If the code does not match an
entry's description, or if the cited reason does not hold, that is itself a
finding: report it and say which entry it contradicts. Entries about open
questions, suspected defects and "not yet fixed" items were deliberately
excluded, so the absence of something here does not mean it is correct.

**Path conventions.** Paths beginning `pyvbmc/`, `dev/`, `docsrc/` are
relative to the PyVBMC repository root. Paths beginning `gpyreg/` are
relative to the gpyreg repository root. Bare MATLAB paths (`vbmc.m`,
`misc/...`, `acq/...`, `gplite/...`, `private/...`, `shared/...`,
`utils/...`, `ent/...`) are relative to the MATLAB VBMC repository root at
the comparison revision `396d649`.

**Kinds.** `deliberate change` (both sides implement the thing, differently
and on purpose), `unported feature` (MATLAB has it, Python never did),
`removed feature` (Python had it and it was taken out), `substituted
library` (a third-party package does what a MATLAB file did), `Python-only
addition` (no MATLAB counterpart at all).

A closing section, **Settled non-differences**, lists behaviors that look
like differences but are not: places where Python was recently brought back
into agreement with MATLAB, and places where a documented Python quirk is in
fact inherited from MATLAB.

---

## Slices G1, G2, P5 — the GP layer (gpyreg and `gaussian_process_train.py`)

### Integrated mean function (`intmeanfun`) never ported
- Python: no counterpart. `pyvbmc/vbmc/gaussian_process_train.py:961`
  ("Missing port: intmean part") and `:421` ("Missing port: hyperprior over
  mixture of quadratics mean function"); `gpyreg/gaussian_process.py` has no
  `intmeanfun` concept.
- MATLAB: `gplite/gplite_intmeanfun.m`; branches in
  `gplite/private/gplite_core.m`, `gplite/gplite_post.m`,
  `gplite/gplite_pred.m`, `gplite/gplite_train.m`, and on the VBMC side
  `acq/acqviqr_vbmc.m`, `misc/intkernel.m`, `misc/gpreupdate.m`,
  `misc/check_quadcoefficients_vbmc.m`,
  `private/activeimportancesampling_vbmc.m`; option
  `defopts.gpIntMeanFun = 0` (`vbmc.m:246`).
- What differs: MATLAB can place a Gaussian prior on a set of basis-function
  coefficients and marginalize them analytically, adding terms to the
  posterior factors, the predictive mean and variance, the negative log
  marginal likelihood and the acquisition functions. Python has none of it.
  PyVBMC keeps the option (`gp_int_mean_fun = 0`,
  `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:111`) and copies it
  into `optim_state["int_mean_fun"]` (`pyvbmc/vbmc/vbmc.py:1100`), where
  nothing reads it.
- Why: gpyreg never ported the feature. `gplite_intmeanfun.m` is absent from
  gpyreg's reference copy under `gpyreg/matlab/gplite/`, and every one of the
  remaining `intmeanfun` branches accounts for the bulk of the difference
  between that copy and the comparison target (verified file by file:
  `gplite_post.m`, `gplite_train.m` and `private/gplite_core.m` differ *only*
  in `intmeanfun` code). Recorded in
  `dev/plans/port-correctness-review.md` §"Reference revisions".
- Kind: unported feature.

### GP output warping never ported
- Python: `pyvbmc/vbmc/gaussian_process_train.py:334`, `:363`, `:443`, `:461`
  ("Missing port: output warping hyperparameters not implemented",
  "Missing port: priors and bounds for output warping hyperparameters");
  `pyvbmc/whitening/whitening.py:293` keeps the MATLAB line as a comment.
  gpyreg has no output-warping component.
- MATLAB: `gplite/outwarp_negpow.m`, `gplite/outwarp_negpowc1.m`,
  `gplite/outwarp_negscaledpow.m`; option `defopts.gpOutwarpFun = []`
  (`vbmc.m:366`).
- What differs: MATLAB can fit a monotone warping of the GP outputs with its
  own hyperparameters, priors and bounds. PyVBMC keeps the main-loop
  bookkeeping options (`fitness_shaping`, `out_warp_thresh_base`,
  `out_warp_thresh_mult`, `out_warp_thresh_tol` in
  `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:248-255`, and
  `optim_state["out_warp_delta"]` in `pyvbmc/vbmc/vbmc.py:1105`, `:1672`)
  but there is no warping function behind them.
- Why: gpyreg's `AGENTS.md` §"Relation to the MATLAB reference" states the
  `outwarp_*.m` functions are not ported; the PyVBMC comments record the same
  on the VBMC side.
- Kind: unported feature.

### Only three of gplite's mean functions exist in gpyreg
- Python: `gpyreg/mean_functions.py` defines `ZeroMean`, `ConstantMean`,
  `NegativeQuadratic`. `pyvbmc/vbmc/gaussian_process_train.py:377`
  ("Missing port: we only implement the mean functions that gpyreg
  supports"), `:459` ("Missing port: meanfun == 14 hyperprior case").
- MATLAB: `gplite/gplite_meanfun.m` switches over 24 identifiers
  (`zero`, `const`, `linear`, `quad`, `negquad`, `posquad`, `se`, `negse`,
  `negquadse`, `posquadse`, `negquadfixiso`, ..., `posquadmix`).
- What differs: the linear, positive-quadratic, squared-exponential, fixed,
  isotropic-fixed, quadratic-only and quadratic-mixture mean families, and
  the hyperpriors PyVBMC would need for them, have no Python counterpart.
  `pyvbmc/vbmc/gaussian_process_train.py:_meanfun_name_to_mean_function`
  accepts only the three that exist. VBMC's default (`negquad`) is ported.
- Why: recorded in the two "Missing port" comments above and in gpyreg's
  `AGENTS.md`, which lists the three component families gpyreg implements.
- Kind: unported feature.

### `gplite_qpred.m` (quantile prediction) not ported
- Python: no counterpart; `gpyreg/gaussian_process.py:2258` records
  "quantile doesn't work, requires gplite_qpred implementation" in `GP.plot`.
- MATLAB: `gplite/gplite_qpred.m`.
- What differs: gpyreg cannot produce predictive quantiles, and the
  corresponding branch of the GP plot is disabled.
- Why: gpyreg's `AGENTS.md` lists `gplite_qpred.m` as not ported, and the
  code comment names the missing dependency.
- Kind: unported feature.

### Sampling from, and minimizing, a GP-represented density not ported
- Python: no counterpart in gpyreg (`GP` has `predict`, `predict_full`,
  `quad`, `random_function`, `plot`, but no `sample` or `fmin`).
- MATLAB: `gplite/gplite_sample.m`, `gplite/gplite_fmin.m`,
  `gplite/private/eissample_lite.m`.
- What differs: MATLAB can draw MCMC samples from the log density a GP
  represents (`gplite_sample`, using `slicesamplebnd` or `eissample_lite`)
  and can locate that density's global optimum from multiple starts
  (`gplite_fmin`). gpyreg exposes neither.
- Why: gpyreg's `AGENTS.md` describes the ported surface
  (`gplite_train/post/pred/clean/rnd/plot.m`, the three component modules,
  the slice sampler and the space-filling design); these two entry points are
  not among the ported functions and nothing in PyVBMC calls them.
  (Note: gpyreg's `AGENTS.md` maps `gplite_sample.m` to `slice_sample.py` and
  `gplite_fmin.m` to `f_min_fill.py`; the counterpart map records why those
  two mappings are wrong. The substance — no Python counterpart — stands.)
- Kind: unported feature.

### gpyreg's `SliceSampler` replaces MATLAB's two samplers
- Python: `gpyreg/slice_sample.py` (`SliceSampler`), used by
  `gpyreg/gaussian_process.py: fit` for hyperparameter sampling and by
  `pyvbmc/vbmc/active_importance_sampling.py:106,259` for the IMIQR MCMC
  step.
- MATLAB: `gplite/private/slicesamplebnd.m` (hyperparameter sampling in
  `gplite/gplite_train.m`) and `gplite/private/eissample_lite.m` (the
  elliptical-slice/ensemble sampler used by
  `private/activeimportancesampling_vbmc.m`).
- What differs: one Python slice sampler stands in for both MATLAB samplers.
  The ensemble sampler `eissample_lite` has no Python implementation at all,
  so the noisy-acquisition MCMC step draws from a different algorithm and a
  different random stream than MATLAB's.
- Why: recorded in `dev/plans/port-correctness-review.md` §Slices, P4 ("MATLAB's
  MCMC sampler; PyVBMC uses gpyreg's `SliceSampler`, a substitution the
  comparison reviewer examines"), and in gpyreg's `AGENTS.md`.
- Kind: substituted library.

### Gradient checking: `derivcheck.m` replaced by `numdifftools`
- Python: `gpyreg/testing/test_utils.py: check_grad` (numdifftools);
  `pyvbmc/testing/_check_grad.py` and the
  `pyvbmc/testing/**/test_*_grad_fd.py` modules.
- MATLAB: `gplite/private/derivcheck.m`.
- What differs: finite-difference gradient verification uses a third-party
  numerical-differentiation package rather than MATLAB's hand-written helper.
- Why: gpyreg's `AGENTS.md` §Tests; the finite-difference gate list in
  `dev/plans/port-correctness-review.md` §"Fixes and gates".
- Kind: substituted library.

### The space-filling design and the optimizer are SciPy
- Python: `gpyreg/f_min_fill.py` (SciPy `stats.qmc.Sobol`, unscrambled, with
  a `rand` fallback) and `gpyreg/gaussian_process.py: fit`
  (`scipy.optimize.minimize` with analytic gradients).
- MATLAB: `gplite/private/fminfill.m` (its own Sobol/`uuinv` fill plus
  `fmincon`/`fminunc`, guarded by `options.OptimToolbox`).
- What differs: the candidate design comes from SciPy's Sobol engine and the
  local optimization from SciPy, so the starting points and the optimizer
  trajectory are not MATLAB's. MATLAB's `defopts.OptimToolbox` (`vbmc.m:210`),
  which chooses between toolbox and fallback optimizers at runtime, has no
  Python counterpart.
- Why: gpyreg's `AGENTS.md` §Lifecycle describes the SciPy-based `fit`; the
  `OptimToolbox` switch is meaningless outside MATLAB.
- Kind: substituted library.

### NumPy quantile and standard-deviation conventions in GP bound recommendations
- Python: `gpyreg/mean_functions.py:498-500` and `:522-523` (comments:
  "For future reference note that quantile behaviour in MATLAB and NumPy is
  slightly different", "note that std behaviour in MATLAB and NumPy is
  slightly different").
- MATLAB: `gplite/gplite_meanfun.m` bound section, via
  `gplite/private/quantile1.m` and `std`.
- What differs: `np.quantile` interpolates between order statistics at
  `(i-1)/(N-1)`, while `quantile1.m` uses the `(i-0.5)/N` convention, so the
  recommended plausible bounds and starting values of the mean-function
  hyperparameters differ by a small amount that depends on `N`.
- Why: the two code comments record the choice explicitly.
- Kind: deliberate change.

### Isotropic kernels and the rational-quadratic kernel are Python-only
- Python: `gpyreg/isotropic_covariance_functions.py`
  (`SquaredExponentialIsotropic`, `MaternIsotropic`) and
  `gpyreg/covariance_functions.py: RationalQuadraticARD`.
- MATLAB: `gplite/gplite_covfun.m` implements the ARD squared exponential
  (`case 1`) and the ARD Matern (`case 3`); the isotropic identifier `seiso`
  is declared in the hyperparameter-count switch but has no compute branch,
  and there is no rational-quadratic kernel.
- What differs: gpyreg offers kernels the MATLAB reference does not. PyVBMC
  itself hard-wires the ARD squared exponential, so these are not reachable
  from a default run.
- Why: gpyreg's `AGENTS.md` ("Isotropic kernels are a Python-only addition").
- Kind: Python-only addition.

### `predict(..., return_cross_covariance=True)` and the VIQR kernel reuse
- Python: `gpyreg/gaussian_process.py: predict` (the
  `_ZERO_COPY_CROSS_COVARIANCE_COMPUTES` path) and
  `pyvbmc/acquisition_functions/acq_fcn_viqr.py:144`.
- MATLAB: no counterpart; `acq/acqviqr_vbmc.m` recomputes the cross-kernel.
- What differs: gpyreg 1.2.0 added an API that hands the latent cross-kernel
  matrices back to the caller so VIQR does not recompute them. The returned
  matrices must be treated as read-only.
- Why: `AGENTS.md` §"Setup and commands" (gpyreg 1.2.0, acerbilab/gpyreg#45);
  validated in `dev/results/2026-09-13-viqr-kernel-production.md`
  (1.205x late-sieve speedup, 18/18 exact stored replays).
- Kind: Python-only addition (performance; arithmetic unchanged).

### `_get_hyp_cov` anchors the hyperparameter dimension on the current model
- Python: `pyvbmc/vbmc/gaussian_process_train.py:691` (`_get_hyp_cov`,
  parameter `hyp_n`).
- MATLAB: `misc/get_GPTrainOptions.m`, local function `GetHypCov`
  (lines 126-160).
- What differs: MATLAB accumulates historical hyperparameter blocks and
  accepts a block only if its width matches the width of what has already
  been accumulated (`isempty(hyp_list) || size(hyp_list,2) == size(hyp,1)`),
  so the first block encountered fixes the dimension. PyVBMC takes the
  dimension from the *current* GP (or from `hyp_dict["hyp"]`) and skips any
  historical block of a different width. PyVBMC additionally returns `None`
  when no usable samples remain or when the covariance would rest on a single
  effective sample, so that gpyreg's default sampler widths apply; MATLAB has
  no such guard.
- Why: `dev/plans/latent-bug-fixes.md` Phase 3 item 1 ("take hyp_N from the
  current model's hyperparameter count ... not the newest historical block";
  "Return None for no usable samples or an undefined single-effective-sample
  covariance") and §"Review and rollback" ("Anchor covariance dimensions to
  the current model, not historical blocks");
  `dev/results/2026-09-08-main-loop-fixes.md` §"Weighted GP covariance".
- Kind: deliberate change.

### `gpsample_vbmc.m` (GP-surrogate sampling) not ported
- Python: `pyvbmc/vbmc/gaussian_process_train.py:205` ("Missing port: sample
  for GP for debug (not used)"); `VariationalPosterior.sample` takes a
  boolean `balance_flag` only.
- MATLAB: `misc/gpsample_vbmc.m`, called from `vbmc_rnd.m:49` when
  `balanceflag` is the string `'gp'`.
- What differs: MATLAB can draw posterior samples from the GP surrogate
  itself instead of from the variational mixture. PyVBMC cannot.
- Why: the "Missing port" comment, and the porting log
  `pyvbmc/vbmc/README.md` ("sample(): vbmc_rnd.m — gp_sample is missing").
- Kind: unported feature.

### Noise shaping is an unported stub
- Python: `pyvbmc/vbmc/gaussian_process_train.py:823` and
  `pyvbmc/vbmc/active_sample.py:327` ("Missing port: noise_shaping"). The
  options `noise_shaping`, `noise_shaping_threshold`, `noise_shaping_factor`
  exist (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:261-265`,
  default off) and only flip a noise-function flag.
- MATLAB: `misc/noiseshaping_vbmc.m`, called from
  `misc/get_traindata_vbmc.m`, `misc/gpsample_vbmc.m` and
  `private/activesample_vbmc.m`; `defopts.NoiseShaping = 'no'` (`vbmc.m:320`).
- What differs: MATLAB inflates the observation noise of training points far
  below the maximum observed density; PyVBMC does not.
- Why: `dev/2026-09-02-modernization-discussion.md` §9
  ("`gaussian_process_train.py:768` — `noise_shaping` is an unported stub;
  the option only flips a noise-function flag"), and
  `dev/results/2026-09-16-gp-box-sampler.md` ("Noise shaping remains an
  unported, default-off option"). Out of 1.5 scope per `dev/TODO.md`
  §"Outside 1.5 scope".
- Kind: unported feature.

### The iteration history stores GPs without their posterior factors
- Python: `pyvbmc/vbmc/gaussian_process_train.py:872` (`_lean_gp`),
  `:910` (`_restore_gp_posteriors`), `pyvbmc/vbmc/vbmc.py:2725`
  (`get_gp`).
- MATLAB: `vbmc.m`, `save_stats` (stores `stats.gp` in full).
- What differs: PyVBMC records training data, hyperparameter samples and the
  model only, and rebuilds the Cholesky factors on demand, so the history
  does not grow as `Ns` times the square of the training-set size. Records
  written by older PyVBMC versions that carry complete GPs are copied as they
  are.
- Why: `dev/2026-09-02-modernization-discussion.md` §9 ("Resolved for the
  history 2026-09-05", Stage 2 item 7, `dev/plans/stage2-memory.md`);
  `AGENTS.md` §Architecture.
- Kind: deliberate change (memory; the rebuilt factors are identical to the
  ones dropped).

---

## Slice P1a — main loop, warmup, termination, final boost

### Final boost is guarded; MATLAB accepts its result unconditionally
- Python: `pyvbmc/vbmc/vbmc.py:2403` (`final_boost`), `:2575`
  (`_validate_final_boost_tolerance`), `:2591`
  (`_is_valid_final_boost_score`), `:2604`
  (`_accept_final_boost_candidate`); option `tol_elcbo_boost = 0.1`
  (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:51`).
- MATLAB: `misc/finalboost_vbmc.m` (no acceptance test).
- What differs: PyVBMC saves the pre-boost variational posterior and its
  stored ELBO/SD, runs the boost once, and keeps the boosted posterior only
  when `min(dE, dE - 5*dS) > -tol_elcbo_boost`, where `dE` and `dS` are the
  changes in ELBO and ELBO SD. This is the exact reduction of the PI's
  continuum criterion "`dE - b*dS > -tol_elcbo_boost` for every `b` in
  `[0,5]`" to its endpoints. A failing or non-finite candidate is rejected
  with a warning and the untouched pre-boost posterior is returned; the
  `changed_flag` reports whether the returned posterior was boosted.
  `tol_elcbo_boost = None` (and old saves missing the key) restores the
  legacy unguarded behavior.
- Why: `dev/results/2026-09-04-final-boost-failure.md` documents the failure
  that motivated it (`student_D4` seed 19: a converged posterior turned into
  ELBO −9.03 ± 0.49 for ln Z = −10.36, gsKL 54); ruled a bug rather than an
  algorithmic decision by the PI on 2026-09-06.
  `dev/plans/latent-bug-fixes.md` Phase 2 and Q4 fix the rule;
  §"Completed pickup (2026-09-08): PI selected final-boost defaults" records
  the selected tolerance 0.1 (859/870 candidates accepted on the paired
  campaign).
- Kind: deliberate change.

### Final boost runs without weight penalty and without pruning
- Python: `pyvbmc/vbmc/vbmc.py: final_boost` (boost-only copied options set
  `weight_penalty = 0`, retaining `tol_weight = 0`).
- MATLAB: `misc/finalboost_vbmc.m` sets `TolWeight = 0` (disabling the eta
  bound and pruning) but leaves the small-weight shrinkage term active.
- What differs: PyVBMC's final refinement has neither a weight penalty nor
  pruning, so the ELBO alone determines the final weights.
- Why: PI-selected direction, 2026-09-07, recorded in
  `dev/plans/latent-bug-fixes.md` Phase 2 item 1 and Q1; confirmed in the
  2026-09-08 completed pickup ("Remove the small-weight penalty only during
  final boost").
- Kind: deliberate change.

### `results["overhead"]` is not implemented
- Python: `pyvbmc/vbmc/vbmc.py:3137` (`output["overhead"] = np.nan`).
- MATLAB: `private/vbmc_output.m`.
- What differs: PyVBMC returns `NaN` where MATLAB returns the fractional
  overhead (total running time / total function time − 1).
- Why: `docsrc/source/faq.md` §"How is `overhead` in `results` defined?"
  ("PyVBMC currently returns `np.nan` for `results["overhead"]`; this field is
  not implemented").
- Kind: unported feature.

### `results["rng_state"]` is a generator snapshot, not MATLAB's `rng` state
- Python: `pyvbmc/vbmc/vbmc.py:3034` (`_get_random_state`, returning
  `{"generator": <bit generator state>}`), `:3138`, `:3071`
  (`_set_random_state`).
- MATLAB: `private/vbmc_output.m` stores `rng` (the global generator state).
- What differs: PyVBMC snapshots the instance's own `numpy.random.Generator`
  at return time, as an independent deep copy, and never touches a global
  stream. `load(set_random_state=True)` ignores the per-iteration legacy
  tuple written by versions before the generator migration.
- Why: `dev/plans/latent-bug-fixes.md` §"Numerical and state changes"
  (`results["rng_state"]` row: "Return an independent snapshot with the
  existing `{"generator": ...}` format at return-time, after post-loop
  draws") and Phase 1 item 6; `AGENTS.md` §"Randomness goes through
  `numpy.random.Generator` objects".
- Kind: deliberate change.

### The automated retry on failure (`RetryMaxFunEvals`) is not ported
- Python: no counterpart.
- MATLAB: `vbmc.m:968-982`; `defopts.RetryMaxFunEvals = 0` (`vbmc.m:164`).
- What differs: MATLAB can restart a failed run with a fresh initial design
  and a new evaluation budget. PyVBMC never retries.
- Why: `docsrc/source/faq.md:635` ("PyVBMC does not implement MATLAB VBMC's
  `RetryMaxFunEvals` automated retry option").
- Kind: unported feature.

### `vbmc_diagnostics.m` is not ported
- Python: no counterpart (the `diagnostics` occurrences under `pyvbmc/` are
  unrelated).
- MATLAB: `vbmc_diagnostics.m`.
- What differs: MATLAB provides a cross-run diagnostic routine that compares
  several VBMC solutions. PyVBMC has none.
- Why: `dev/plans/port-correctness-review.md` §Slices, P1b
  ("`vbmc_diagnostics.m` (unported)").
- Kind: unported feature.

### Bayesian-optimization warmup (`BOWarmup`) is not ported
- Python: no counterpart.
- MATLAB: `vbmc.m:496`, `:711`, `:825`, `:867`; `defopts.BOWarmup = 'no'`
  (`vbmc.m:365`).
- What differs: MATLAB has an optional warmup stage that behaves like
  Bayesian optimization. PyVBMC's warmup has no such branch.
- Why: `pyvbmc/vbmc/README.md` §"Porting status" ("The experimental features
  (listed in `vbmc.m`) have not been ported yet"); the option is default-off
  and absent from PyVBMC's `.ini` files.
- Kind: unported feature.

---

## Slice P1b — setup, options, defaults, bounds, history, save/load, RNG

### Options are layered `.ini` files evaluated in Python
- Python: `pyvbmc/vbmc/options.py`, `pyvbmc/vbmc/option_configs/*.ini`.
- MATLAB: `misc/setupoptions_vbmc.m`, `misc/evaloption_vbmc.m`,
  `utils/evalbool.m`, and the `defopts` block of `vbmc.m`.
- What differs: defaults live in `basic_vbmc_options.ini` then
  `advanced_vbmc_options.ini`, overridden by `options_path=` and then by the
  `options=` dict. `.ini` values are `eval`'d with `D` bound and only the
  names imported in `options.py` available; the `options=` dict is used
  verbatim. Unknown keys raise at validation, and options are frozen after
  initialization (`options.__setitem__(k, v, force=True)` overrides). MATLAB
  silently accepts and string-evaluates fields listed in `evalfields`, and
  `evalbool` accepts `'yes'`/`'no'` strings that Python does not.
- Why: `AGENTS.md` §Architecture ("Options are layered ...");
  `pyvbmc/vbmc/README.md` §"Options Class" (the `is_initialized` flag and the
  `force=True` keyword).
- Kind: deliberate change.

### Randomness is threaded through `numpy.random.Generator`
- Python: `pyvbmc/rng.py: get_rng`, `VBMC(seed=)` → `vbmc.rng`, shared with
  `vbmc.vp`; `train_gp(rng=)` → `gpyreg.GP.fit(rng=)`;
  `pyvbmc/vbmc/active_sample.py` (`_BatchedNoiseHandler` drawing from
  `vp.rng`, and the `randn` callback passed to `cma.fmin`);
  `pyvbmc/vbmc/active_importance_sampling.py` (the slice sampler receives
  `vp.rng`). `gpyreg/rng.py: resolve_rng`.
- MATLAB: `rand`, `randn`, `mvnrnd` on the global stream throughout.
- What differs: a PyVBMC run never reads or writes NumPy's global state, and
  every copy of a variational posterior shares one stream
  (`VariationalPosterior.__deepcopy__`). `seed=None` derives the generator
  from the global `np.random` state so that `np.random.seed()` beforehand
  still fixes a run. Consequently no Python draw sequence can be compared
  point by point with a MATLAB draw sequence.
- Why: `AGENTS.md` §"Randomness goes through `numpy.random.Generator`
  objects" (with the dates: the global-state reseeding ended 2026-09-05, the
  IMIQR path 2026-09-10); `dev/plans/stage1-rng-generator.md`;
  gpyreg `AGENTS.md` §"Random number handling".
- Kind: deliberate change.

### Vectorized targets, precomputed evaluations and the initialization cost
- Python: `pyvbmc/vbmc/vbmc.py:595` (`_validate_initialization_cost`),
  `:647` (`_initialize_precomputed_evaluations`), `:820`
  (`_fresh_evaluations_for_batch`), `:3396`
  (`_validate_vectorized_target_option`);
  `pyvbmc/function_logger/function_logger.py: batch_call`; option
  `vectorized_target = False`
  (`pyvbmc/vbmc/option_configs/basic_vbmc_options.ini:21`).
- MATLAB: no counterpart; `misc/funlogger_vbmc.m` always evaluates one point
  at a time.
- What differs: with `vectorized_target=True` the missing rows of the initial
  design are evaluated in one original-coordinate `(M,D)` call, then recorded
  in design order; later target calls remain sequential with shape `(1,D)`.
  `precomputed_evaluations` lets a caller supply already-evaluated points,
  whose `setup_cost` is charged once as `initialization_cost`. Missing flags
  in old saves mean `False`; load-time overrides synchronize the logger and
  the prior wrapper.
- Why: `AGENTS.md` §"Vectorized targets are opt-in" and §"PyMC targets";
  `dev/plans/stage3-pipeline-features.md`, `dev/plans/pymc-target-adapter.md`.
- Kind: Python-only addition.

### The iteration history omits the noisy acquisitions' importance samples
- Python: `pyvbmc/vbmc/vbmc.py:3013` (`_optim_state_record`, setting
  `record["active_importance_sampling"] = None`); option
  `record_full_history_details = False`
  (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:337`).
- MATLAB: `vbmc.m`, `save_stats`.
- What differs: the recorded `optim_state` leaves out the importance samples
  drawn afresh at every active-sampling step, which are not rebuildable from
  the record. Setting the option keeps them.
- Why: `AGENTS.md` §"`optim_state` is a plain dict"; the option's own comment
  line in the `.ini`.
- Kind: deliberate change.

### Python-only user-interface and infrastructure options
- Python: `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini`:
  `show_tips`, `print_iteration_header`, `log_file_name`, `log_file_level`,
  `log_file_mode`, `performance_calibration = "cached"` (`:339`),
  `do_final_boost = True` (`:49`), `ns_gp_max_active = np.inf` (`:77`).
- MATLAB: no counterpart for any of them.
- What differs: runtime tips and the logging configuration are Python
  concerns; `performance_calibration` selects a machine-local numerical chunk
  profile; `do_final_boost` can skip the final boost; `ns_gp_max_active` caps
  the hyperparameter samples of the GP refits inside active sampling (0 means
  a MAP-only fit), and saved runs without the key behave as before.
- Why: `dev/plans/runtime-tips.md` (tips and header),
  `dev/plans/machine-local-calibration.md` and
  `dev/results/2026-09-09-machine-local-calibration.md`
  (`performance_calibration`); commit `782da99` "Add do_final_boost option
  and fix test. (#111)"; commit `014b287` (`ns_gp_max_active`).
- Kind: Python-only addition.

### `misc/vbmc_gphyp.m` is an empty file in the comparison target
- Python: `pyvbmc/vbmc/gaussian_process_train.py:279` (`_gp_hyp`).
- MATLAB: `misc/vbmc_gphyp.m` is zero bytes at `396d649`; the function
  `vbmc_gphyp` is a local subfunction of `misc/gptrain_vbmc.m:109`.
- What differs: nothing in behavior, but a reviewer looking for
  `misc/vbmc_gphyp.m` will find an empty file. The GP hyperprior setup is in
  `gptrain_vbmc.m`, and `_gp_hyp` is its counterpart.
- Why: verified directly (`git cat-file -s HEAD:misc/vbmc_gphyp.m` is 0).
- Kind: not a difference; recorded here so no reviewer reports the empty
  file as a missing port.

---

## Slice P2 — initial design and active-sampling search

### The `cma` package replaces `cmaes_modded.m`
- Python: `pyvbmc/vbmc/active_sample.py:5` (`import cma`), the `cma.fmin`
  call at `:533`, and the `_BatchedNoiseHandler` subclass.
- MATLAB: `utils/cmaes_modded.m`.
- What differs: the acquisition search uses the third-party `cma` package
  with `seed=np.nan` and a `randn` callback bound to the run's generator, so
  neither the population draws nor the restart logic are MATLAB's. The noise
  handler is a PyVBMC subclass that draws its re-evaluation count from
  `vp.rng`.
- Why: `dev/plans/port-correctness-review.md` §Slices, P2 ("`utils/cmaes_modded.m`
  (PyVBMC uses the `cma` package)"); `AGENTS.md` §Architecture;
  `dev/plans/stage2-batched-acquisition.md` §"Findings the plan rests on".
- Kind: substituted library.

### The CMA-ES search evaluates the whole population in one acquisition call
- Python: `pyvbmc/vbmc/active_sample.py` (`parallel_objective=acq_fun` in the
  `cma.fmin` call; the noise handler's re-evaluations batched the same way).
- MATLAB: `utils/cmaes_modded.m` calls the acquisition one point at a time.
- What differs: the acquisition is evaluated twice per generation instead of
  `popsize + 2` times. The algorithm is the same and `cma`'s `ask` draws the
  same population in either mode, but
  `AbstractAcqFcn._sq_dist` centers both point sets on a mean that depends on
  the batch, so acquisition values computed in a batch can differ from
  pointwise values by about 2e-15 relative, which can flip a near-tie.
- Why: `dev/plans/stage2-batched-acquisition.md` §Summary and its "Findings
  the plan rests on"; `dev/2026-09-02-modernization-discussion.md` §9
  ("`_sq_dist` ... Not a bug, but it is why the noisy acquisitions evaluated
  one point at a time and in a batch can differ by a finite amount on a
  near-tie"); retained explicitly in `dev/plans/latent-bug-fixes.md`
  §"Supporting fixes and closed entries" ("Preserve the Stage 2 arithmetic;
  no change in this plan").
- Kind: deliberate change.

### The initial design does not cluster surplus starting points
- Python: `pyvbmc/vbmc/active_sample.py:164-168` ("In the MATLAB
  implementation there is a cluster algorithm being used to pick the best
  points, but we decided not to implement that yet and just pick the first
  sample_count points").
- MATLAB: `misc/initdesign_vbmc.m:29-43`, using `utils/fastkmeans.m`.
- What differs: when more than `fun_eval_start` starting points are supplied,
  MATLAB clusters them with k-means and takes the highest-density point of
  each cluster; PyVBMC takes the first `sample_count` rows and logs an
  informational message. The `N0 <= Ns` branch (uniform draws in the
  plausible box, or the `narrow` design) is ported faithfully.
- Why: the code comment above is the record of the decision.
- Kind: unported feature.

### Repeated observations are selected differently
- Python: `pyvbmc/vbmc/active_sample.py:370-390` (training inputs are
  appended to the sieve's search set when
  `max_repeated_observations > 0`, the target is noisy, and the streak is
  below the cap; a chosen repeat skips the local optimizer so the logger
  pools it into its row) and `:645-651` (streak bookkeeping). The
  corresponding MATLAB block is kept, commented out, at `:564-609`
  ("Missing port"), and `:609` records the unported
  `private/activesample_vbmc.m:356-361`.
- MATLAB: `private/activesample_vbmc.m:330-364`: after the search, the
  acquisition is re-evaluated on the whole training set with
  variance regularization disabled and with `t_algoperfuneval` temporarily
  set to the current rather than the projected GP cost, and the best training
  input replaces the search result when
  `acq_train < options.RepeatedAcqDiscount * acq_now`.
  Options `MaxRepeatedObservations = 0` and `RepeatedAcqDiscount = 1`
  (`vbmc.m:196-197`).
- What differs: PyVBMC lets the training inputs compete inside the sieve
  instead of running MATLAB's separate post-search comparison, so
  `RepeatedAcqDiscount` has no Python counterpart. The cost model
  `optimState.t_algoperfuneval` (`private/activesample_vbmc.m:185-205`),
  which is used only inside MATLAB's repeat block, is therefore also
  unported. Both sides default to `max_repeated_observations = 0`, so the
  mechanism is off unless a user enables it.
- Why: commit `014b287` "feat(acq): VIQR loss variants, EIG port, repeat
  candidates, in-loop GP sample cap" ("with max_repeated_observations > 0 on
  noisy runs the training inputs join the search set, a chosen repeat skips
  the local optimizer so the logger pools it into its row, and a streak cap
  resets"); measured in `dev/2026-09-08-noisy-acquisitions.md`
  §"Repeated observations".
- Kind: deliberate change.

### The acquisition-portfolio hedge (`acqhedge_vbmc.m`) is not ported
- Python: `pyvbmc/vbmc/vbmc.py:1023` (only an empty `optim_state["hedge"]`
  is created) and `pyvbmc/vbmc/active_sample.py:304` (`if not
  options["acq_hedge"]: idx_acq = rng.integers(...)`, so with the option on,
  `idx_acq` is never chosen). Options `acq_hedge`, `acq_hedge_iter_window`,
  `acq_hedge_decay` exist and default to off
  (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:266-271`).
- MATLAB: `private/acqhedge_vbmc.m`, called from
  `private/activesample_vbmc.m` and `vbmc.m:848`.
- What differs: with several acquisition functions, MATLAB runs a hedging
  portfolio that tracks each one's recent improvement; PyVBMC picks one
  uniformly at random per active-sampling step and has no hedge at all.
- Why: `pyvbmc/vbmc/README.md` §"Porting status"
  ("`acqhedge_vbmc.m` has not been ported yet as it is considered to be
  experimental"); `dev/plans/port-correctness-review.md` §Slices, P2 marks it
  unported.
- Kind: unported feature.

### Variational active sampling (`VarActiveSample`) is not ported
- Python: no counterpart; the option `active_variational_samples = 0` exists
  (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:272`) but nothing
  reads it in the sampling loop.
- MATLAB: `misc/vpsample_vbmc.m`, called from
  `private/activesample_vbmc.m`; `defopts.VarActiveSample = 'no'`
  (`vbmc.m:363`). MATLAB itself marks the branch `% Unused` at `vbmc.m:652`.
- What differs: MATLAB can interleave MCMC steps on the variational
  parameters with active sampling. PyVBMC cannot.
- Why: `dev/plans/port-correctness-review.md` §Slices, P6 lists
  `misc/vpsample_vbmc.m`, `utils/slicesample_vbmc.m` and
  `utils/malasample_vbmc.m` as "unported: sampling of variational
  parameters"; `pyvbmc/vbmc/README.md` §"Porting status" for the experimental
  features generally.
- Kind: unported feature.

### `check_quadcoefficients_vbmc.m` is not ported
- Python: no counterpart in `pyvbmc/vbmc/gaussian_process_train.py:
  reupdate_gp`.
- MATLAB: `misc/check_quadcoefficients_vbmc.m`, called from
  `misc/gpreupdate.m`.
- What differs: MATLAB checks (and can repair) the quadratic mean
  coefficients after a GP re-update; the check exists because of the
  integrated mean function. PyVBMC's `reupdate_gp` omits it.
- Why: `dev/plans/port-correctness-review.md` §Slices, P2 ("unported,
  integrated mean function"); follows from the integrated-mean-function entry
  above.
- Kind: unported feature.

### `AbstractAcqFcn._real2int` snaps its input in place
- Python: `pyvbmc/acquisition_functions/abstract_acq_fcn.py:261`
  (`X[:, integer_vars] = X_temp[:, integer_vars]` writes into the caller's
  array); called from `pyvbmc/vbmc/active_sample.py:367`, `:556` and from
  `AbstractAcqFcn.__call__:85`.
- MATLAB: `misc/real2int_vbmc.m` returns a new array.
- What differs: through the `Xs[None, :]` view of a 1-D input, the pointwise
  CMA-ES objective snapped CMA-ES's own solution arrays to the integer grid.
  The batched objective reproduces that side effect deliberately.
- Why: `dev/2026-09-02-modernization-discussion.md` §9 ("an undocumented side
  effect that the batched objective now reproduces deliberately");
  `dev/plans/latent-bug-fixes.md` §"Supporting fixes and closed entries"
  ("Deliberately preserved by the Stage 2 batched CMA-ES objective. Retain
  its current behavior").
- Kind: deliberate change.

---

## Slice P3 — acquisition functions

### The EIG acquisition and the experimental VIQR losses were removed
- Python: absent from `pyvbmc/acquisition_functions/`. The implementation
  (`acq_fcn_eig.py`, and the `loss` variants `iqr_reduction`,
  `var_reduction`, `sd_reduction` of `AcqFcnVIQR`) is retained on the branch
  `retain/experimental-acquisitions` at `fa6922f`.
- MATLAB: `acq/acqeig_vbmc.m` and its helper `misc/intkernel.m`.
- What differs: the package has no expected-information-gain acquisition and
  `AcqFcnVIQR` offers only the standard `loss="iqr"`.
- Why: PI decision 2026-09-13, recorded in
  `dev/plans/modernization-roadmap.md:58-67` and `dev/TODO.md`
  §"Outside 1.5 scope"; the measurements are in
  `dev/results/2026-09-08-noisy-acquisition-experiments.md` (scalar EIG poor,
  per-component EIG mixed, neither robustly better than default VIQR). The
  removal took effect 2026-09-14.
- Kind: removed feature.

### `misc/intkernel.m` has no standalone Python counterpart
- Python: no counterpart. The removed EIG implementation on
  `retain/experimental-acquisitions` computed the integrated posterior
  covariance in closed form rather than porting `intkernel`.
- MATLAB: `misc/intkernel.m`, used only by `acq/acqeig_vbmc.m`.
- What differs: MATLAB's helper integrates the kernel against a Gaussian
  measure (and branches on the integrated mean function); PyVBMC never had
  it.
- Why: commit `014b287` ("AcqFcnEIG(components=...): port of acqeig_vbmc.m
  from the closed-form integrated posterior covariance"); the file name does
  not appear anywhere on `retain/experimental-acquisitions`.
- Kind: unported feature (and the only caller is now a removed feature).

### VIQR's integrated-mean-function branches are not ported
- Python: `pyvbmc/acquisition_functions/acq_fcn_viqr.py:293` ("Missing port,
  integrated mean function, lines 49 to 57") and `:360` ("Missing port,
  integrated meanfun").
- MATLAB: `acq/acqviqr_vbmc.m`, the `gp.intmeanfun` branches.
- What differs: PyVBMC's VIQR omits the extra mean and covariance terms
  MATLAB adds when an integrated mean function is in use.
- Why: the two "Missing port" comments; follows from the
  integrated-mean-function entry above.
- Kind: unported feature.

### Variance regularization: canonical key plus a legacy alias
- Python: `pyvbmc/vbmc/vbmc.py` writes
  `optim_state["variance_regularized_acq_fcn"]`;
  `pyvbmc/acquisition_functions/abstract_acq_fcn.py` reads the canonical key
  and falls back to the legacy `variance_regularized_acqfcn` only when the
  canonical key is absent. The acquisition result is normalized to one value
  per point before the regularization and bounds masks; zero variance uses
  the limiting penalty.
- MATLAB: `acq/acqwrapper_vbmc.m` reads
  `optimState.VarianceRegularizedAcqFcn`, always present.
- What differs: PyVBMC carries the compatibility alias so that saved runs and
  stored fixture states written before the key was corrected keep loading;
  the shape normalization has no MATLAB analogue because MATLAB's acquisition
  values are already a column.
- Why: `dev/plans/latent-bug-fixes.md` Phase 4 items 1-2;
  `dev/results/2026-09-08-main-loop-fixes.md` §"Acquisition regularization
  (Phase 4 complete)".
- Kind: deliberate change.

---

## Slice P4 — noisy importance sampling

### The IMIQR MCMC step uses gpyreg's `SliceSampler`
- Python: `pyvbmc/vbmc/active_importance_sampling.py:106`, `:259`
  (`gpr.slice_sample.SliceSampler`, seeded from `vp.rng`), `:409`
  (`get_mcmc_opts`).
- MATLAB: `private/activeimportancesampling_vbmc.m` calls
  `eissample_lite` (`gplite/private/eissample_lite.m`,
  `utils/eissample_lite.m`).
- What differs: a coordinate-wise bounded slice sampler replaces MATLAB's
  ensemble slice sampler. The samples, their correlation structure and the
  random stream all differ; only the target density is shared.
- Why: `dev/plans/port-correctness-review.md` §Slices, P4. Also
  `AGENTS.md` §"Randomness ..." (the IMIQR path moved onto `vp.rng` on
  2026-09-10, when the `acq_AcqFcnIMIQR` oracle was re-baselined).
- Kind: substituted library.

### The MCMC branch of importance sampling is a retained but dormant hook
- Python: `pyvbmc/vbmc/active_importance_sampling.py:86`
  (`if acq_fcn.acq_info.get("mcmc_importance_sampling")`), which no built-in
  acquisition sets.
- MATLAB: `private/activeimportancesampling_vbmc.m` selects the branch from
  the acquisition's own `acqInfo`.
- What differs: the branch is unreachable with the shipped acquisition
  functions, and is kept so that a user-supplied acquisition object can still
  request it.
- Why: `dev/2026-09-02-modernization-discussion.md` §9 ("the MCMC branch of
  `active_importance_sampling` is unreachable"); decision D5 in
  `dev/plans/latent-bug-fixes.md` §"Approved decisions" ("Retain dormant
  compatibility hooks ... Do not delete custom acquisition hooks").
- Kind: deliberate change.

---

## Slice P6 — variational optimization and the ELBO

### The eta soft bound was removed, and `_neg_elcbo` no longer mutates `theta`
- Python: `pyvbmc/vbmc/variational_optimization.py: _neg_elcbo` (copies
  `eta` before the stable max-shift, and passes private bound arrays with the
  eta entries set to infinities, so the eta-bound loss and its gradient are
  zero for every finite eta) and `_vp_bound_loss`.
- MATLAB: `misc/negelcbo_vbmc.m` passes the original `theta` to
  `misc/vpbndloss.m`, which extracts raw `eta` and passes it to
  `utils/softbndloss.m` with `a = log(0.5*TolWeight)`, `b = 0` from
  `misc/vpbounds.m`; each component contributes
  `0.5*((a-eta)_+/s)^2 + 0.5*((eta-b)_+/s)^2` with `s = (b-a)*TolConLoss`.
- What differs: PyVBMC applies no penalty to the mixture-weight parameters'
  soft bounds at all. Location and scale losses and gradients, and the
  separate capped small-weight shrinkage term (`weight_penalty`), are
  unchanged and still match MATLAB. Neither the caller's `theta` nor the
  supplied bounds are modified.
- Why: PI decision of 2026-09-08 (treatment "A") after the three-arm
  comparison, recorded in `dev/plans/latent-bug-fixes.md` Phase 6
  ("Production decision (PI, 2026-09-08): A is selected") and Q1, and in
  `dev/results/2026-09-08-eta-bound-fix.md` (64 focused tests, all 11
  fixtures exact, three paired replays). The rationale: MATLAB's absolute-eta
  bounds assign different penalties to eta vectors representing the same
  posterior, and MATLAB parity alone did not justify keeping them; the
  supporting measurements are in
  `dev/results/2026-09-08-eta-bound-comparison.md` and
  `dev/results/2026-09-08-eta-equal-budget.md`.
- Kind: deliberate change.

### The diagonal approximation of the log-joint variance is not ported
- Python: `pyvbmc/vbmc/variational_optimization.py:1420` (`compute_var == 2`
  raises `NotImplementedError`, "Missing port: compute_var == 2 skipped since
  it is not used"), and the gradient-of-variance path raises before it.
- MATLAB: `misc/gplogjoint.m`, the `compute_var == 2` branch and the
  `dvarG` accumulators.
- What differs: PyVBMC can compute the exact variance of the expected log
  joint but not MATLAB's cheaper diagonal approximation, nor the gradient of
  either. The accumulators that once half-implemented it were deleted, so
  `dvarG` is always `None`.
- Why: `dev/2026-09-02-modernization-discussion.md` §9
  ("Resolved 2026-09-04 (evening) by deletion ... the path was doubly
  unreachable ... Porting MATLAB's diagonal variance approximation would be a
  feature, not a fix"); `dev/plans/stage2-gp-log-joint-einsum.md`;
  held deferred in `dev/plans/latent-bug-fixes.md` §"Purpose and boundaries"
  ("Noise shaping, `compute_var == 2`, log-space mixture sums ... stay
  deferred").
- Kind: unported feature.

### The weight-only fast paths are not ported
- Python: `pyvbmc/vbmc/variational_optimization.py:1198-1208` keeps the
  `onlyweights_flag` test commented out ("Not currently used, since it is
  only a speed optimization") and `:1207` records "Missing port: block below
  does not have branches for only weight optimization".
- MATLAB: `misc/gplogjoint_weights.m`, called from
  `misc/negelcbo_vbmc.m:63-86`; `misc/vpoptimizeweights_vbmc.m`, whose only
  call site (`vbmc.m:717`) is itself commented out in MATLAB.
- What differs: when only the weights are being optimized, MATLAB switches to
  a specialized log-joint routine; PyVBMC always uses the general one. The
  results are meant to agree; the cost does not.
- Why: the code comment naming it a speed optimization, and the "Missing
  port" comment.
- Kind: unported feature.

### `ELCBOWeight` is not ported; `elcbo_beta` is fixed at zero
- Python: `pyvbmc/vbmc/variational_optimization.py:771` ("Missing port:
  elcboweight does not exist", followed by `elcbo_beta = 0` and
  `compute_var = elcbo_beta != 0`).
- MATLAB: `misc/vpsieve_vbmc.m` evaluates
  `evaloption_vbmc(options.ELCBOWeight, optimState.n_eff)`;
  `defopts.ELCBOWeight = 0` (`vbmc.m:360`).
- What differs: PyVBMC cannot weight the ELBO uncertainty during variational
  optimization. MATLAB's default is `0`, so at default options the two agree
  and the variance is not computed on either side.
- Why: the "Missing port" comment; the MATLAB default is `0`.
- Kind: unported feature (inactive at MATLAB defaults).

### The variational-posterior repository (`vp_repo`) is not ported
- Python: no counterpart in `_sieve`.
- MATLAB: `misc/vpsieve_vbmc.m` reads `optimState.vp_repo`, filled during
  active sampling.
- What differs: MATLAB can seed the sieve with variational posteriors stored
  from earlier iterations; PyVBMC's sieve builds its candidates afresh.
- Why: `pyvbmc/vbmc/README.md` §"Porting status" ("The vp_repo has not been
  ported for now ... Even in MATLAB this part seems to be implemented quickly
  and then it is not used").
- Kind: unported feature.

### Sampling of the variational parameters is not ported
- Python: no counterpart.
- MATLAB: `misc/vpsample_vbmc.m`, `utils/slicesample_vbmc.m`,
  `utils/malasample_vbmc.m`; option `defopts.VarParamsBack = 0`
  (`vbmc.m:361`).
- What differs: MATLAB can run slice-sampling or MALA steps on the
  variational parameters and can look back at variational posteriors from
  previous iterations. PyVBMC has neither.
- Why: `dev/plans/port-correctness-review.md` §Slices, P6 ("unported:
  sampling of variational parameters"); `pyvbmc/vbmc/README.md`
  §"Porting status" (experimental features).
- Kind: unported feature.

### `_vb_init` type 3 with frozen sigma copies existing widths
- Python: `pyvbmc/vbmc/variational_optimization.py:844` (`_vb_init`), the
  `vb_type == 3` branch when `vp.optimize_sigma` is `False` and
  `K_new > vp.K`.
- MATLAB: `misc/vbinit_vbmc.m` has no such case (MATLAB never combines a
  frozen sigma with component growth in this path either).
- What differs: PyVBMC preserves the supplied widths for the existing
  components and cyclically copies them for the new slots, without drawing
  new sigmas, instead of leaving a shape mismatch. The default path
  (`optimize_sigma = True`, which normal construction always sets) retains
  its draw order and values exactly.
- Why: question Q3, approved by the PI on 2026-09-07 and recorded in
  `dev/plans/latent-bug-fixes.md` §"PI questions and resolutions"
  and §"Supporting fixes and closed entries"; the failure it replaces is in
  `dev/2026-09-02-modernization-discussion.md` §9
  ("Found 2026-09-05 (evening) by the bit-check of Stage 2 item 6").
- Kind: deliberate change (a dormant path; unreachable in production).

---

## Slice P7 — variational posterior, entropies, VP statistics

### Posterior tempering (`vbmc_power`, `vptrain2real`) is not ported
- Python: no counterpart. `pyvbmc/vbmc/vbmc.py:1344` and `:1536` keep
  `# vp_real = vp.vptrain2real(0, self.options)` commented out with
  `vp_real = self.vp`. The option `temperature = 1` exists
  (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:256`) and is read
  only by `pyvbmc/whitening/whitening.py:190`, `:284`.
- MATLAB: `vbmc_power.m`, `misc/vptrain2real.m` (the branch
  `any(T == [2,3,4,5])`).
- What differs: MATLAB can run the algorithm on a tempered posterior and
  convert the training posterior back to the untempered one, recomputing the
  ELBO, its SD and the entropy. PyVBMC has no conversion, so a temperature
  other than 1 would be silently wrong outside the whitening code. At the
  default `temperature = 1` MATLAB's `vptrain2real` returns its input
  unchanged, so the commented-out line is behaviorally equivalent at
  defaults.
- Why: `pyvbmc/variational_posterior/README.md` §"Porting status"
  ("The function `vbmc_power` ... has not been ported, but it is very low
  priority ... it was part of an experimental feature");
  `dev/plans/port-correctness-review.md` §Slices, P7 marks `vbmc_power.m`
  unported.
- Kind: unported feature.

### The upper-bound entropy (`entub_vbmc.m`) is not ported
- Python: `pyvbmc/entropy/` contains only `entlb_vbmc.py` and
  `entmc_vbmc.py`.
- MATLAB: `ent/entub_vbmc.m`.
- What differs: MATLAB offers a deterministic upper bound on the mixture
  entropy alongside the Jensen lower bound and the Monte Carlo estimate;
  PyVBMC offers only the latter two.
- Why: `dev/plans/port-correctness-review.md` §Slices, P7
  ("`ent/entub_vbmc.m` (unported)"); `pyvbmc/entropy/README.md` lists only
  the two ported files.
- Kind: unported feature.

### `AltMCEntropy` is not ported
- Python: no counterpart.
- MATLAB: `defopts.AltMCEntropy = 'no'` (`vbmc.m:362`).
- What differs: MATLAB has an alternative Monte Carlo entropy estimator
  behind a default-off switch. PyVBMC has only `entmc_vbmc`.
- Why: `pyvbmc/vbmc/README.md` §"Porting status" (experimental features
  listed in `vbmc.m` not ported); the option is absent from PyVBMC's `.ini`
  files.
- Kind: unported feature.

### `vp.pdf` refuses original-space gradients instead of returning a wrong one
- Python: `pyvbmc/variational_posterior/variational_posterior.py:795-798`
  (`if orig_flag and grad_flag: raise NotImplementedError`).
- MATLAB: `vbmc_pdf.m` divides the density by the transform Jacobian but
  returns the transformed-space gradient uncorrected in the non-log case, and
  refuses it in the log case.
- What differs: PyVBMC rejects both original-space gradient modes before
  doing any work. MATLAB silently returns an inconsistent gradient in one of
  them.
- Why: question Q2, approved by the PI on 2026-09-07:
  `dev/plans/latent-bug-fixes.md` §"PI questions and resolutions"
  ("raise NotImplementedError for both original-space gradient modes ...
  This is inherited from MATLAB, so agreement is not a correctness
  argument") and §"MATLAB evidence and bounded probes"; implemented in
  Phase 1 item 7. No solver or S-VBMC caller needs the gradient.
- Kind: deliberate change.

### `kl_div_mvn` takes its inputs directly
- Python: `pyvbmc/stats/kl_div_mvn.py` (a module-level function with no
  decorator; all four inputs normalized locally).
- MATLAB: `shared/mvnkl.m`.
- What differs: the function previously carried the `handle_0D_1D_input`
  decorator, which is written for methods and swallowed `mu1` as `self`.
  Scalar, 1-D, 2-D, keyword and mixed calls now all agree with the previous
  2-D results. The KL formula is unchanged and still matches MATLAB.
- Why: `dev/plans/latent-bug-fixes.md` §"Supporting fixes and closed
  entries" ("Remove the method-only decorator and normalize all four inputs
  locally ... Keep the KL formula unchanged") and Phase 1 item 3;
  the original defect is in `dev/2026-09-02-modernization-discussion.md` §9.
- Kind: deliberate change (API; values unchanged).

### `vp.stats["J_sjk"]` is pruned on both component axes
- Python: `pyvbmc/vbmc/variational_optimization.py:382-383` (`np.delete` on
  `axis=1` and `axis=2`).
- MATLAB: `misc/vpoptimize_vbmc.m` prunes the corresponding quantity
  consistently with `I_sk`.
- What differs: PyVBMC previously deleted only one axis, leaving a
  non-square `J_sjk` after pruning. Nothing in PyVBMC reads it, but S-VBMC's
  constructor filters posteriors using its maximum.
- Why: `dev/plans/latent-bug-fixes.md` §"Supporting fixes and closed entries"
  ("prune both component axes, keep shape `(Ns,K,K)` and correspondence with
  `I_sk`") and Phase 1 item 3.
- Kind: deliberate change (restores the intended shape).

### The corner plot uses the `corner` package
- Python: `pyvbmc/variational_posterior/variational_posterior.py: plot`
  (imports `corner` lazily).
- MATLAB: `vbmc_plot.m`, `utils/cornerplot.m`, `utils/kde2d.m`.
- What differs: the two-dimensional marginal plots are produced by a
  third-party package with its own binning and smoothing, so the figures are
  not MATLAB's.
- Why: `pyvbmc/variational_posterior/README.md` §"Matlab references"
  (`plot()` ↔ `vbmc_plot.m`); `AGENTS.md` §"Testing conventions and traps"
  ("Corner is imported inside `vp.plot`").
- Kind: substituted library.

### `qtrapz.m` is replaced by SciPy's trapezoid rule
- Python: `pyvbmc/variational_posterior/variational_posterior.py: mtv` uses
  `scipy.integrate.trapezoid` over a 1e5-point grid on each of three
  sub-intervals.
- MATLAB: `vbmc_mtv.m` calls `shared/qtrapz.m`.
- What differs: the quadrature helper and its endpoint handling differ;
  MATLAB's `qtrapz` omits the endpoint correction that `trapezoid` applies.
- Why: mechanical: `qtrapz.m`'s only caller is `vbmc_mtv.m`, whose
  counterpart is `mtv`. Recorded here so the absence of a `qtrapz` module is
  not reported as a missing port.
- Kind: substituted library.

### `vbmc_isavp.m` has no counterpart
- Python: `isinstance(..., VariationalPosterior)` at the call sites.
- MATLAB: `vbmc_isavp.m`, used by `vbmc.m`, `vbmc_kldiv.m`, `vbmc_mtv.m`,
  `vbmc_plot.m`.
- What differs: MATLAB needs a helper to recognize a variational-posterior
  struct; Python has a class.
- Why: mechanical; recorded so the missing file is not reported.
- Kind: deliberate change.

---

## Slice P8 — parameter transformer, warping, function logger

### Nonlinear input warping is not ported
- Python: `pyvbmc/whitening/whitening.py` implements rotation and rescaling
  only.
- MATLAB: `misc/warp_input_vbmc.m` and `misc/warp_gpandvp_vbmc.m` under
  `defopts.WarpNonlinear = 'off'` (`vbmc.m:359`), with
  `vbmc.m:537` gating on `options.WarpRotoScaling || options.WarpNonlinear`.
- What differs: PyVBMC supports `warp_rotoscaling` only; there is no
  nonlinear warp of the input space and no corresponding option.
- Why: `pyvbmc/whitening/README.md` §"Porting status" ("Rotating and
  rescaling has been implemented, but nonlinear warping has not").
- Kind: unported feature.

### The gradient of the log Jacobian is not ported
- Python: `pyvbmc/parameter_transformer/parameter_transformer.py` provides
  `log_abs_det_jacobian` but not its derivative.
- MATLAB: `shared/warpvars_vbmc.m` also returns the gradient of the log
  Jacobian.
- What differs: PyVBMC has no access to the derivative of the log Jacobian
  with respect to the transformed variables.
- Why: `dev/plans/port-correctness-review.md` §Slices, O3 ("MATLAB's
  `warpvars_vbmc.m` also provides the gradient of the log Jacobian, which
  PyVBMC lacks"). Whether any Python path needs it is an open question for
  the O3 reader, not a settled matter.
- Kind: unported feature.

### Rotation matrices are validated as orthogonal, and their determinant is
    taken to be one
- Python: `pyvbmc/parameter_transformer/parameter_transformer.py:70-80`
  (orthogonality check at construction, tolerance
  `100 * eps * max(1, D)`), and `log_abs_det_jacobian`, which adds no term
  for `R_mat`.
- MATLAB: `shared/warpvars_vbmc.m` applies the rotation without such a
  check.
- What differs: PyVBMC rejects a non-orthogonal or singular rotation matrix
  at construction instead of silently producing a wrong log Jacobian. The
  forward and inverse transforms use the transpose as the inverse, so general
  affine rotations are not supported by design.
- Why: `dev/plans/latent-bug-fixes.md` §"Supporting fixes and closed entries"
  ("Rotation determinant assumption ... Validate finite `(D,D)` orthogonal
  rotations (reflections allowed) and document the contract; do not add
  general affine support") and Phase 1 item 2; decision D5. The original
  observation is in `dev/2026-09-02-modernization-discussion.md` §9.
- Kind: deliberate change.

### `ParameterTransformer` equality and the logit tail
- Python: `pyvbmc/parameter_transformer/parameter_transformer.py`
  (`__eq__` compares `self.scale` with the other object's scale; the `logit`
  bounded Jacobian evaluates its tail stably).
- MATLAB: `shared/warpvars_vbmc.m` has no equality notion; its logit branch
  has the same closed form.
- What differs: two transformers differing only in `scale` no longer compare
  equal, and the tests that assert a shared transformer across `vbmc`, `vp`
  and `function_logger` use `is`. The logit log-Jacobian is finite beyond
  `y = -709.78`, where the naive `-log1p(exp(-y))` overflowed to `-inf`.
  Ordinary-range arithmetic is preserved.
- Why: `dev/plans/latent-bug-fixes.md` §"Supporting fixes and closed entries"
  (transformer equality; logit Jacobian overflow) and Phase 1 item 2; the
  findings are in `dev/2026-09-02-modernization-discussion.md` §9
  ("Found 2026-09-06 by the finite-difference checks that closed Stage 0").
- Kind: deliberate change.

### `FunctionLogger.finalize()` is explicit and trims every row array
- Python: `pyvbmc/function_logger/function_logger.py:562` (trims `X_orig`,
  `y_orig`, `X`, `y`, `S`, `X_flag`, `fun_eval_time`, `n_evals`); `optimize`
  does not call it.
- MATLAB: `misc/funlogger_vbmc.m`, the `'finalize'` action, called from
  `vbmc.m` at the end of a run.
- What differs: PyVBMC's logger keeps its preallocated rows after a run, so a
  continued run needs no reallocation; the caller trims explicitly. MATLAB
  always trims. The Python version now also trims `n_evals`, which the
  original omitted.
- Why: decision D4 in `dev/plans/latent-bug-fixes.md` §"Approved decisions"
  ("Repair explicit finalize; do not invoke it automatically. The live
  logger's allocation is useful for continuation ... Rejected: auto-finalize
  now") and Phase 1 item 5.
- Kind: deliberate change.

### The 0-D/1-D input decorator is a Python-only device
- Python: `pyvbmc/decorators/handle_0D_1D_input.py`.
- MATLAB: no counterpart (MATLAB's arrays carry their own conventions).
- What differs: public methods promote 1-D inputs to the rigid 2-D shapes the
  algorithm uses. The decorator assumes a `self` first argument and misbehaves
  on module-level functions.
- Why: `AGENTS.md` §"Shapes are rigid".
- Kind: Python-only addition.

---

## Slice P9 — priors

### PyVBMC has a separate prior API; MATLAB does not
- Python: `pyvbmc/priors/` (`Prior`, `UniformBox`, `Trapezoidal`,
  `SplineTrapezoidal`, `SmoothBox`, `SciPy`, `UserFunction`, `Product`,
  `convert_to_prior`), reached through `VBMC(log_likelihood, ...,
  prior=...)`.
- MATLAB: `shared/munifboxpdf.m`, `munifboxlogpdf.m`, `munifboxrnd.m`,
  `mtrapezpdf.m`, `mtrapezlogpdf.m`, `mtrapezrnd.m`, `msplinetrapezpdf.m`,
  `msplinetrapezlogpdf.m`, `msplinetrapezrnd.m`, `msmoothboxpdf.m`,
  `msmoothboxlogpdf.m`, `msmoothboxrnd.m`; `lpostfun.m` shows the user how to
  assemble a log posterior by hand.
- What differs: MATLAB supplies bare density functions and expects the user's
  target to be the log joint. PyVBMC wraps them as classes with a shared
  interface, adds `SciPy` (frozen `scipy.stats` distributions), `UserFunction`
  and `Product` (which have no MATLAB counterpart), and combines the prior
  with a supplied log likelihood inside `VBMC`. Frozen-distribution
  recognition goes through public SciPy factories and `rv_continuous`, and a
  multivariate frozen distribution's dimension is read from `.dim` rather than
  inferred by drawing a sample.
- Why: `dev/plans/latent-bug-fixes.md` §"Supporting fixes and closed entries"
  (SciPy private imports; the `rvs(1)` dimension probe) and Phase 1 item 8;
  decision D5; `AGENTS.md` §"Vectorized targets are opt-in" ("Separate priors
  retain their scalar interface").
- Kind: Python-only addition (with the MATLAB density functions ported
  inside it).

### `pyvbmc/priors/__init__.py` has a fixed import order
- Python: `pyvbmc/priors/__init__.py` carries `# isort:skip` markers.
- MATLAB: no counterpart.
- What differs: the import order is load-bearing (circular-import safety) and
  must not be reordered.
- Why: `AGENTS.md` §"Testing conventions and traps".
- Kind: Python-only addition.

---

## Slices N1, N2, N3 — subsystems with no MATLAB counterpart

### S-VBMC, the PyMC adapter, the Torch and ArviZ exports, and the calibration module
- Python: `pyvbmc/svbmc/`, `pyvbmc/pymc/`,
  `pyvbmc/variational_posterior/_torch.py`,
  `pyvbmc/variational_posterior/_arviz.py`, `pyvbmc/calibration/`.
- MATLAB: no counterpart for any of them.
- What differs: these are additions of the Python line. S-VBMC stacks
  finished posteriors of several runs (moved in from the standalone `svbmc`
  package at 0.1.1, `torch` extra, imported lazily). `PyMCTarget` adapts a
  PyMC model to a VBMC target. `vp.to_torch()` and `vp.to_arviz()` export the
  posterior. `pyvbmc/calibration/` fits machine-local numerical chunk sizes.
- Why: `dev/plans/port-correctness-review.md` §Decisions ("Modules without one
  (`svbmc/`, `pymc/`, the Torch and ArviZ exports, `calibration/`) get the
  internal-correctness track alone"); `AGENTS.md` §Architecture for each;
  `dev/plans/svbmc-integration.md`, `dev/plans/pymc-target-adapter.md`,
  `dev/plans/machine-local-calibration.md`.
- Kind: Python-only addition.

---

## Settled non-differences

These are recorded so that a reviewer who notices them does not spend time on
them. They are *not* differences from MATLAB.

- **The uniform box sampler now matches MATLAB.** PyVBMC's
  `_get_search_points` drew `standard_normal((N_box, D)) * (box_ub - box_lb)
  + box_lb` where `private/activesample_vbmc.m:624` draws `rand(Nbox,D)` and
  scales it into the box. The error entered with the original Python
  implementation (`2551469`, 2021-08-03) and was corrected on 2026-09-16 to
  `rng.random((N_box, D))`. The two implementations now agree.
  (`dev/results/2026-09-16-gp-box-sampler.md`.)
- **The unused outer search limits are unused in MATLAB too.** MATLAB's
  assignments of `LB_searchmin` and `UB_searchmin` to the active search
  bounds are commented out (`private/activesample_vbmc.m:420-427`). PyVBMC's
  expanding search bounds are the ported behavior. (Same report.)
- **The `_get_hyp_cov` decay denominator matches MATLAB.** PyVBMC divides
  `sKL` by `tol_skl * fun_evals_per_iter`, as
  `misc/get_GPTrainOptions.m:137-139` does. The earlier parenthesization
  mismatch was corrected with the Phase 3 repair
  (`dev/results/2026-09-08-main-loop-fixes.md`).
- **The GP-sampling termination criterion matches MATLAB.** PyVBMC's
  `_is_gp_sampling_finished` uses weights
  `0.5 * one_hot_current + 0.5 * normalize(exp(-(N_last - N_i)/10))` over the
  populated history and stops when the weighted `var_ss` falls strictly below
  `tol_gp_var_mcmc`, which is `private/vbmc_termination.m:42-47` with
  `idx_stable = 1`. `optim_state["N"]` counts logged distinct locations
  (`Xn+1`), including rows made inactive by trimming, exactly as
  `vbmc.m` sets `N` from its 1-based `Xn`. What is Python-only is the
  compatibility layer: `optim_state["N"]` is registered in the iteration
  history and backfilled for old saves, exponentials are evaluated after
  subtracting their maximum, and missing or non-finite history never triggers
  a stop. (`dev/plans/latent-bug-fixes.md` Phase 5 and §"MATLAB evidence and
  bounded probes"; `dev/results/2026-09-08-main-loop-fixes.md` §"GP sampling
  termination".)
- **Variance regularization of the acquisition is now active.** It was dead
  in PyVBMC because the producer and the reader spelled the `optim_state` key
  differently; Phase 4 made it live, so the behavior now matches
  `acq/acqwrapper_vbmc.m`. Only the legacy-alias fallback is Python-only
  (see the P3 entry).
- **`AbstractAcqFcn._sq_dist`'s centering is MATLAB's.** `gplite/private/
  sq_dist.m` centers both point sets on a size-weighted mean of the two, and
  the Python copy does the same. Its batch dependence is a consequence of the
  batched CMA-ES objective (see the P2 entry), not of a different formula.
- **`_gp_log_joint` is vectorized but arithmetically the loop's.** The
  `(Ns, K, D, N)` array, the `einsum` contractions and the multi-RHS variance
  solves of the 2026-09-04 rewrite compute the formulas of
  `misc/gplogjoint.m`. (`AGENTS.md` §"Gradients are hand-derived";
  `dev/plans/stage2-gp-log-joint-einsum.md`.)
