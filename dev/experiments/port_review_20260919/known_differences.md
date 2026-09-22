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
the comparison revision `396d649`. Line citations into gpyreg are at
`9e70e6b` (release 1.2.1, the revision PyVBMC's CI installs). Entries that
cite gpyreg by function name describe its branch `port-review-wave6`
(`dev/experiments/port_review_20260919/verification/wave6.md`, "Fix
commits"), to be released as gpyreg 1.3.0; their line citations are brought
to that release when PyVBMC moves to it. Citations of PyVBMC's `AGENTS.md` by
section name or by quotation refer to that file at revision `feadb6fe`; the
file was rewritten afterwards and does not hold every section cited.

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
- Python: no counterpart. `pyvbmc/vbmc/gaussian_process_train.py:903`
  ("Missing port: intmean part") and `:429` ("Missing port: hyperprior over
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
  into `optim_state["int_mean_fun"]` (`pyvbmc/vbmc/vbmc.py:1246`), where
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
- Python: `pyvbmc/vbmc/gaussian_process_train.py:331`, `:361`, `:453`, `:471`
  ("Missing port: output warping hyperparameters not implemented",
  "Missing port: priors and bounds for output warping hyperparameters");
  `pyvbmc/whitening/whitening.py:381-383` keeps the MATLAB line as a comment.
  gpyreg has no output-warping component.
- MATLAB: `gplite/outwarp_negpow.m`, `gplite/outwarp_negpowc1.m`,
  `gplite/outwarp_negscaledpow.m`; option `defopts.gpOutwarpFun = []`
  (`vbmc.m:366`).
- What differs: MATLAB can fit a monotone warping of the GP outputs with its
  own hyperparameters, priors and bounds. PyVBMC keeps the main-loop
  bookkeeping options (`fitness_shaping`, `out_warp_thresh_base`,
  `out_warp_thresh_mult`, `out_warp_thresh_tol` in
  `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:248-255`, and
  `optim_state["out_warp_delta"]` in `pyvbmc/vbmc/vbmc.py:1251`, `:1814`)
  but there is no warping function behind them.
- Why: gpyreg's `AGENTS.md` §"Relation to the MATLAB reference" states the
  `outwarp_*.m` functions are not ported; the PyVBMC comments record the same
  on the VBMC side.
- Kind: unported feature.

### Only three of gplite's mean functions exist in gpyreg
- Python: `gpyreg/mean_functions.py` defines `ZeroMean`, `ConstantMean`,
  `NegativeQuadratic`. `pyvbmc/vbmc/gaussian_process_train.py:382`
  ("Missing port: we only implement the mean functions that gpyreg
  supports"), `:469` ("Missing port: meanfun == 14 hyperprior case").
- MATLAB: `gplite/gplite_meanfun.m` switches over 24 identifiers
  (`zero`, `const`, `linear`, `quad`, `negquad`, `posquad`, `se`, `negse`,
  `negquadse`, `posquadse`, `negquadfixiso`, ..., `posquadmix`).
- What differs: the linear, positive-quadratic, squared-exponential, fixed,
  isotropic-fixed, quadratic-only and quadratic-mixture mean families, and
  the hyperpriors PyVBMC would need for them, have no Python counterpart.
  `pyvbmc/vbmc/gaussian_process_train.py:_meanfun_name_to_mean_function`
  accepts only the three that exist, and `VBMC` refuses the other names of
  `misc/setupvars_vbmc.m:287` at construction and in `load`
  (`pyvbmc/vbmc/vbmc.py:3839`);
  until 2026-09-20 it accepted all twelve and failed in the first GP
  training. VBMC's default (`negquad`) is ported.
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
  `pyvbmc/vbmc/active_importance_sampling.py:243` for the IMIQR MCMC step.
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
  trajectory are not MATLAB's. MATLAB's `defopts.OptimToolbox` (`vbmc.m:210`)
  has no Python counterpart; VBMC itself computes it
  (`misc/setupoptions_vbmc.m:98-106`) and reads it nowhere afterwards.
- Why: gpyreg's `AGENTS.md` §Lifecycle describes the SciPy-based `fit`; the
  `OptimToolbox` switch is meaningless outside MATLAB
  (`dev/experiments/port_review_20260919/reviews/P1b_comparison.md`, sheet
  note S4, for its having no reader in MATLAB).
- Kind: substituted library.

### NumPy's quantile convention in the GP mean-function bound recommendations
- Python: `gpyreg/mean_functions.py:498-501` and `:508` (comment: "For future
  reference note that quantile behaviour in MATLAB and NumPy is slightly
  different").
- MATLAB: `gplite/gplite_meanfun.m:157-160` and `:186`, via
  `gplite/private/quantile1.m`.
- What differs: `np.quantile` interpolates between order statistics at
  `(i-1)/(N-1)`, while `quantile1.m` uses the `(i-0.5)/N` convention, so the
  plausible bounds of `mean_const` and the recommended starting value of the
  constant of a negative-quadratic mean differ by an amount that depends on
  `N` and on the spread of `y`: +0.046 and -0.119 on a 30-point set whose
  targets span 24 nats. The same holds for the plausible bounds that
  `warp_input` takes after a warp (`pyvbmc/whitening/whitening.py:266` against
  `misc/warp_input_vbmc.m:85-86`, MATLAB's own `quantile`), by about 1e-4 on
  its 1e5 draws. The statistics of the training inputs beside them, the
  standard deviation (with `ddof=1`, as MATLAB's `std`), the minimum, the
  maximum and the median, are taken per column, as gplite takes them; gpyreg
  1.2.1 took them over all the entries of `X` (`verification/wave6.md`,
  W6-1).
- Why: the code comment records the choice explicitly.
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
  from a default run. MATLAB's `seiso` identifier does have a
  bound-recommendation branch (`gplite/gplite_covfun.m:112-118`, the mean of
  the per-dimension log widths), which `gpyreg/isotropic_covariance_functions.py`
  reproduces; gpyreg 1.2.1 took the log of the arithmetic mean of the
  widths, an upward shift of the four length-scale bounds by
  `log(mean w) - mean(log w)` (`verification/wave6.md`, W6-29).
- Why: gpyreg's `AGENTS.md` ("Isotropic kernels are a Python-only addition").
- Kind: Python-only addition.

### `predict(..., return_cross_covariance=True)` and the VIQR kernel reuse
- Python: `gpyreg/gaussian_process.py: predict` (the
  `_ZERO_COPY_CROSS_COVARIANCE_COMPUTES` path) and
  `pyvbmc/acquisition_functions/acq_fcn_viqr.py:144-148`.
- MATLAB: no counterpart; `acq/acqviqr_vbmc.m` recomputes the cross-kernel.
- What differs: gpyreg 1.2.0 added an API that hands the latent cross-kernel
  matrices back to the caller so VIQR does not recompute them. The returned
  matrices must be treated as read-only.
- Why: `AGENTS.md` §"Setup and commands" (gpyreg 1.2.0, acerbilab/gpyreg#45);
  validated in `dev/results/2026-09-13-viqr-kernel-production.md`
  (1.205x late-sieve speedup, 18/18 exact stored replays).
- Kind: Python-only addition (performance; arithmetic unchanged).

### `_get_hyp_cov` anchors the hyperparameter dimension on the current model
- Python: `pyvbmc/vbmc/gaussian_process_train.py:632` (`_get_hyp_cov`,
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
- Python: `pyvbmc/vbmc/gaussian_process_train.py:202` ("Missing port: sample
  for GP for debug (not used)"); `VariationalPosterior.sample` takes a
  boolean `balance_flag` only.
- MATLAB: `misc/gpsample_vbmc.m`, called from `vbmc_rnd.m:49` when
  `balanceflag` is the string `'gp'`.
- What differs: MATLAB can draw posterior samples from the GP surrogate
  itself instead of from the variational mixture. PyVBMC cannot.
- Why: the "Missing port" comment, and the porting log
  `pyvbmc/vbmc/README.md` ("sample(): vbmc_rnd.m — gp_sample is missing").
- Kind: unported feature.

### Noise shaping is not ported, and `noise_shaping=True` is rejected
- Python: `pyvbmc/vbmc/gaussian_process_train.py:764` and
  `pyvbmc/vbmc/active_sample.py:346` ("Missing port: noise_shaping"). The
  options `noise_shaping`, `noise_shaping_threshold`, `noise_shaping_factor`
  exist (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:261-265`,
  default off). `VBMC.__init__` and `VBMC.load` raise `NotImplementedError`
  when `noise_shaping` is set (`pyvbmc/vbmc/vbmc.py:488` and `:3275`, both
  through `_validate_option_values`, `:3754`, which calls
  `_validate_noise_shaping_option`, `:3795`).
- MATLAB: `misc/noiseshaping_vbmc.m`, called from
  `misc/get_traindata_vbmc.m`, `misc/gpsample_vbmc.m` and
  `private/activesample_vbmc.m`; `defopts.NoiseShaping = 'no'` (`vbmc.m:320`).
- What differs: MATLAB inflates the observation noise of training points far
  below the maximum observed density; PyVBMC does not. What the option does
  reach in PyVBMC is the rest of MATLAB's setting: an input-dependent GP
  noise function (`pyvbmc/vbmc/vbmc.py:1238-1242`) and no rank-one GP update
  inside active sampling (`pyvbmc/vbmc/active_sample.py:826-829`). A run with
  those two and no shaping is a configuration of neither toolbox, so
  construction refuses it and names the missing MATLAB feature.
- Why: `dev/2026-09-02-modernization-discussion.md` §9
  ("`gaussian_process_train.py:768` — `noise_shaping` is an unported stub;
  the option only flips a noise-function flag"), and
  `dev/results/2026-09-16-gp-box-sampler.md` ("Noise shaping remains an
  unported, default-off option"). Out of 1.5 scope per `dev/TODO.md`
  §"Outside 1.5 scope". The construction-time rejection was added on
  2026-09-19 by commit `bb6ab65` "fix(vbmc): reject noise_shaping, whose
  shaping is not ported"; what the option reaches without the shaping is set
  out in
  `dev/experiments/port_review_20260919/verification/wave1_options.md`
  (row O-2).
- Kind: unported feature.

### The iteration history stores GPs without their posterior factors
- Python: `pyvbmc/vbmc/gaussian_process_train.py:814` (`_lean_gp`),
  `:852` (`_restore_gp_posteriors`), `pyvbmc/vbmc/vbmc.py:2984`
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

### The lower bounds of the length scales and of the output scale come from
    the high-posterior-density subset
- Python: `pyvbmc/vbmc/gaussian_process_train.py:401` (`_gp_hyp`): the lower
  bounds of `covariance_log_lengthscale` and `covariance_log_outputscale` are
  the recommendations of gpyreg's kernel computed on the high-posterior-density
  subset of the training set (`hpd_frac`, 0.8).
- MATLAB: `misc/gptrain_vbmc.m:174-180` leaves both entries of `LB_gp` unset,
  and `gplite/gplite_train.m:74`, `:120` fill them from the full training
  set.
- What differs: a subset has a smaller range, so PyVBMC's lower bounds are
  lower, by a fraction of a nat on an ordinary training set and by much more
  when the training set holds a value far below the rest.
- Why: issue 99 and pull request 116 of `acerbilab/pyvbmc` (2022): the
  recommended bounds "may be not appropriate in some cases when we have a
  large range of point values", and bounds from the high posterior region
  give "a broader bounds on the gp's outputscale and lengthscale".
  `verification/wave3.md`, row W3-11, finds that the reason holds.
- Kind: deliberate change.

### Slice sampling is the only sampler of the GP hyperparameters
- Python: `pyvbmc/vbmc/vbmc.py:3856` (`_validate_gp_hyp_sampler_option`)
  refuses at construction every `gp_hyp_sampler` but `"slicesample"`, and
  `_get_gp_training_options` holds the policy of that sampler alone;
  `cov_sample_thresh`, which only covariance sampling read, is an inert
  option.
- MATLAB: `misc/get_GPTrainOptions.m:18-91` and `gplite/gplite_train.m:316-457`
  offer `slicesample`, `npv`, `mala`, `slicelite`, `splitsample`, `covsample`
  and `laplace`; `utils/slicelite.m` is one of them.
- What differs: gpyreg samples hyperparameters by slice sampling and raises
  for any other name, so PyVBMC refuses the other values where they are
  given. Until 2026-09-20 it accepted them, transcribed their branches (two
  of them wrongly: `verification/wave3.md`, row W3-13) and failed part-way
  through the run.
- Why: the PI's ruling on W3-13, after the pattern of `noise_shaping` and of
  the separate search GP.
- Kind: unported feature.

### The evaluation times are not carried onto the GP
- Python: `pyvbmc/vbmc/gaussian_process_train.py:900` ("Missing port: gp.t =
  t_train"); `train_gp` drops `t_train` as well.
- MATLAB: `misc/gpreupdate.m:8`, `misc/gptrain_vbmc.m:76`,
  `private/activesample_vbmc.m:484`.
- What differs: MATLAB attaches the evaluation time of every training point
  to the GP struct. Nothing reads the field anywhere in the MATLAB toolbox.
- Why: `verification/wave3.md`, row W3-18.
- Kind: unported feature (without a reader).

---

### The window of the burn-in statistics of the slice sampler
- Python: `gpyreg/slice_sample.py:562`, `:568`: the sums behind the adapted
  widths are accumulated over `burn/2 <= i < burn` (`i` from 0), exactly
  `floor(burn/2)` iterations, and divided by `floor(burn/2)`.
- MATLAB: `gplite/private/slicesamplebnd.m:362`, `:367-371`: accumulated
  over `ii > burn/2` (`ii` from 1) and divided by `floor(burn/2)`.
- What differs: with an even burn-in the two windows are the same and the
  two samplers agree bit for bit on one stream. With an odd burn-in MATLAB
  sums one term more than it divides by, its estimate is no variance and is
  negative wherever a coordinate's standard deviation is small against its
  mean, and the complex square root makes `:371` keep the widths of the
  shrinking phase for every coordinate. PyVBMC's burn-in is odd in every fit
  that does not recompute the variational posterior (`thin * 3 = 15`,
  `pyvbmc/vbmc/gaussian_process_train.py:608`, as
  `misc/get_GPTrainOptions.m:108` has it) and whenever the number of
  hyperparameter samples is odd, so MATLAB's adaptation is discarded as a
  rule there, and its widths are 1 to 74 times gpyreg's on the stored oracle
  states. Both are valid slice samplers: the widths change the chain, not
  the law it draws from. Where its estimate is not positive, gpyreg keeps
  the width a coordinate has, and a window of fewer than two iterations
  adapts nothing (`verification/wave6.md`, W6-3).
- Why: the MATLAB defect is entry 41 of `matlab_side_defects.md`; gpyreg's
  window matches its own divisor (`verification/wave6.md`, W6-2 and W6-3,
  with `verification/scripts/wave6_A2_burn_in_window.py`).
- Kind: deliberate change (gpyreg's window is consistent with its divisor;
  MATLAB's is the defect).

### `GP.quad`'s integral variance is normalized by the total training noise
- Python: `gpyreg/gaussian_process.py: quad`, the scale of the Cholesky
  solves of the variance, the noise scale `Posterior.sl` that the stored
  factor carries.
- MATLAB: `gplite/gplite_quad.m:66-67`, `:101`.
- What differs: MATLAB normalizes the Cholesky solves of the integral
  variance by `exp(2*hyp(Ncov+1)) * sn2_mult`, the constant noise
  hyperparameter, while `gplite/private/gplite_core.m:82` scales the factor
  by `min(sn2) * sn2_mult`, the minimum of the total training noise. The two
  coincide only when the constant term is the whole noise, so with
  user-provided or output-dependent noise MATLAB's `J_kk` is over-corrected
  and `max(eps, J_kk)` clamps the reported variance to machine epsilon.
  gpyreg uses the scale the factor carries, which is the right one: on a
  14-point one-dimensional GP with one point at a tenfold noise variance,
  gpyreg and an independent grid quadrature of the latent predictive
  covariance agree to twelve digits (9.437668150943e-03) where MATLAB's
  formula gives 2.2e-16. The integral's mean agrees on both sides in every
  configuration.
- Why: gpyreg 1.2.1, `acerbilab/gpyreg#49` ("`quad` scaled the Cholesky
  solves by the constant-noise hyperparameter, while the factorization is
  scaled by the minimum total training noise including `s2`"), issue #8's
  example (2.2e-16 against 0.2007), and the 1.2.1 release notes.
  `test_quadrature_with_noise_matches_numerical_integration` is the gate.
  No PyVBMC code calls `GP.quad` (`verification/wave6.md`, W6-30).
- Kind: deliberate change (a fix of a MATLAB-side defect; entry 42 of
  `matlab_side_defects.md`).

### `predict(return_lpd=True)` pools the hyperparameter samples where `gplite_pred.m` does not
- Python: `gpyreg/gaussian_process.py: predict`, the averaged branch of the
  log predictive density.
- MATLAB: `gplite/gplite_pred.m:124-127` with `:154-165`.
- What differs: MATLAB computes the log predictive density per
  hyperparameter sample and returns the whole `(Nstar, Ns)` matrix even when
  the moments are averaged, leaving the pooling to the caller. gpyreg with
  `separate_samples=False` returns one column, the log density of the
  moment-matched Gaussian `N(mean_s mu_s, mean_s y_s2_s + var_s mu_s)`,
  neither the mean of MATLAB's columns (0.093 nats away on a three-sample
  20-point fit) nor the log of the mean of the densities (0.181 nats). With
  `separate_samples=True` the two agree to 1e-13. Both sides put the total
  predictive variance, latent plus noise times `sn2_mult`, in the density;
  gpyreg has done so since `acerbilab/gpyreg`'s `9b3d46b` (2022-06-02),
  three weeks before MATLAB's `68a197b` (2022-06-25) replaced a line that
  used the noise variance alone.
- Why: the docstring of `predict` (with `separate_samples` `False`, "the
  log density of a Gaussian whose mean is the mean of the per-sample means
  and whose variance is the mean of the per-sample predictive variances plus
  the sample variance (``ddof=1``) of the per-sample means"). No PyVBMC call
  site passes `return_lpd` (`verification/wave6.md`, W6-31).
- Kind: deliberate change.

### Two smooth-box hyperprior families are Python-only
- Python: `gpyreg/gaussian_process.py: set_priors` accepts `"smoothbox"` and
  `"smoothbox_student_t"`; `__prior_masks` and `__compute_log_priors`
  evaluate them, and `gpyreg/f_min_fill.py` draws the initial design from
  them through `smoothbox_cdf`/`smoothbox_ppf` and their Student-t
  counterparts.
- MATLAB: `gplite/gplite_hypprior.m` has three families only, uniform,
  Gaussian and Student's t (`:35-37`), and `gplite/private/fminfill.m:69-96`
  has no corresponding design branch. `hprior` has no `a` or `b` field.
- What differs: gpyreg offers two hyperprior families with no MATLAB
  counterpart, flat over `[a, b]` with Gaussian or Student-t tails, and
  identifies them by a non-finite `mu` with a finite `sigma`, which is why
  the mask for no prior is an `and` where `gplite_hypprior.m:35` is an
  element-wise `|` (the entry below).
- Why: the families and their distribution functions arrived together in
  `27f8d66` (2021-06-11) and `b0c5cde` (2021-06-14), before the option
  surface was settled; PyVBMC sets only `student_t` priors
  (`pyvbmc/vbmc/gaussian_process_train.py: _gp_hyp`), so no PyVBMC run
  evaluates them. A smooth box over a block of hyperparameters has one
  normalization constant per coordinate (`verification/wave6.md`, W6-6).
- Kind: Python-only addition.

### The hyperprior is renormalized to the bounds, and a fixed hyperparameter gets a prior
- Python: `gpyreg/gaussian_process.py: __recompute_normalization_constants`
  stores, per coordinate, the probability the prior puts inside the bounds,
  and `__compute_log_priors` subtracts their log (`lp -= masks["log_norm"]`).
  Separately, the mask `f_idx` returns `-inf` for a hyperparameter whose
  bounds are equal and whose value is not exactly that bound.
- MATLAB: `gplite/gplite_hypprior.m` has neither; its priors are
  unnormalized over the bounds, and a fixed hyperparameter contributes
  whatever its prior gives.
- What differs: gpyreg's `log_posterior` is a normalized log posterior with
  respect to the bounds, so its value differs from `gplite_nlZ`'s by the sum
  of the log truncation constants. The term is constant in `hyp`, so neither
  the optimizer nor the slice sampler sees it, and no sampled hyperparameter
  and no PyVBMC number moves; only the reported value of `log_posterior`
  differs (by 0.747 in the case measured, and by whatever the bounds make it
  in general). `f_idx` likewise changes only what a caller who evaluates off
  a fixed value is told.
- Why: introduced with the bound handling of `0ca35b3` (2021-06-28) and the
  prior families of `27f8d66` (2021-06-11); PyVBMC never compares
  `log_posterior` values across the two implementations
  (`verification/wave6.md`, W6-24).
- Kind: Python-only addition.

### "No prior" is expressed by `None`, not by an infinite scale
- Python: `gpyreg/gaussian_process.py: set_priors` takes the value `None`
  for a hyperparameter with no prior and leaves its `mu`, `sigma` and `df`
  NaN; the uniform mask is `u_idx = ~np.isfinite(mu) & ~np.isfinite(sigma)`,
  and `gpyreg/f_min_fill.py` has the matching `and`. Within a block, a
  coordinate whose location (`mu`, or `a` and `b` for the smooth-box
  families) and `sigma` are both NaN has no prior; `set_priors` refuses a
  `sigma` that is not finite and positive on every other coordinate, with a
  message that says how no prior is written.
- MATLAB: `gplite/gplite_hypprior.m:35` is
  `uidx = ~isfinite(mu) | ~isfinite(sigma)`, and `gplite/gplite_nlZ.m:13`
  documents "Set HPRIOR.sigma(i) = Inf to have a (non-normalized) flat prior
  over the i-th hyperparameter"; `gplite/private/fminfill.m:73` uses the
  same `||`.
- What differs: MATLAB marks a hyperparameter as having no prior when either
  the location or the scale is non-finite, so `sigma = Inf` is its
  documented way of asking for a flat prior. gpyreg requires both, because a
  non-finite `mu` with a finite `sigma` is how it identifies its own
  smooth-box families (the entry above). A prior written MATLAB's way is
  refused (gpyreg 1.2.1 took it for a Student's t, with a log posterior of
  `-inf` or NaN and a design column of NaN). PyVBMC never writes such a
  prior: `_gp_hyp` gives every coordinate that has a prior a finite `mu` and
  `sigma`, and leaves both NaN where there is none.
- Why: the `and` is required by the smooth-box families of `27f8d66`
  (2021-06-11); the refusal: `verification/wave6.md`, W6-17, with
  `wave6_G1.md`, row G1-12, and the coordinates without a prior, "The
  independent check of the pass".
- Kind: deliberate change.

### An exception of the hyperparameter objective ends the fit
- Python: `gpyreg/f_min_fill.py`, the evaluation loop of the design, and
  `gpyreg/gaussian_process.py: fit`, the optimizer loop: an exception of the
  objective propagates.
- MATLAB: `gplite/private/fminfill.m:104-110` wraps each design evaluation
  (`catch % Something went wrong, try to continue`, leaving the preset
  `Inf`), `gplite/gplite_train.m:276-296` wraps each `fmincon` call
  (`catch % Could not optimize, keep starting point`), and `gp_objfun`
  itself catches everything and returns `NaN`.
- What differs: one failing hyperparameter vector ends gpyreg's fit, and a
  PyVBMC run with it, where MATLAB loses a starting point in silence. A
  value of NaN is handled alike on both sides (sorted last, and `-inf` for
  the sampler); a raised exception is not. At the defaults nothing raises,
  the noise floor keeping the factorization from failing ten times over.
- Why: ruled on 2026-09-22 to stay as it is, a failed factorization that
  ends a run being seen where one that costs a start in silence is not
  (`verification/wave6.md`, W6-19).
- Kind: deliberate change.

### The default starting point of `fit`
- Python: `gpyreg/gaussian_process.py: fit`: with no `hyp0` given, the
  current hyperparameters when posteriors exist, and otherwise the middle of
  the plausible box, clamped into the bounds.
- MATLAB: `gplite/gplite_train.m:98`: `hyp0 = zeros(Nhyp,1)`.
- What differs: the start of a fit that is given none. PyVBMC always passes
  `hyp0`, except when the model's hyperparameter count changes between two
  fits, which nothing in PyVBMC does.
- Why: `dbc91d6` (2021-06-24) and later; a plausible midpoint rather than
  the origin (`verification/wave6.md`, W6-24).
- Kind: deliberate change.

### A training set of equal targets takes a range of one
- Python: `gpyreg/covariance_functions.py: _target_spread`, which the bound
  recommendations of the kernels, the noise and the mean take the scale of
  the targets from.
- MATLAB: `gplite/gplite_covfun.m:130-131`, `gplite/gplite_noisefun.m:104`,
  `gplite/gplite_train.m:142`.
- What differs: with targets that are all equal, the range
  `max(y) - min(y)` is zero on both sides, and the recommended bounds of the
  output scale and the upper bound of the noise are `-inf`; `UB = max(LB,
  UB)` leaves the pair `(-Inf, -Inf)`, which gplite hands to `fmincon`.
  gpyreg gives such a training set the range and the standard deviation of a
  unit range, with a warning, as both sides do for a single target, and the
  fit completes. A PyVBMC run without hard bounds reaches it when the log
  joint is constant over the initial design.
- Why: `verification/wave6.md`, W6-4 (the PI's ruling); entry 53 of
  `matlab_side_defects.md`.
- Kind: deliberate change (gpyreg's repair of a defect of both sides).

### `random_function` takes negative eigenvalues of rounding size for zeros
- Python: `gpyreg/gaussian_process.py: random_function` and
  `__robust_cholesky`.
- MATLAB: `gplite/gplite_rnd.m:61`, `:97-110`.
- What differs: where the Cholesky factorization of the predictive
  covariance fails, both sides factor it through its eigendecomposition,
  keep the eigenvalues above MATLAB's drop tolerance (`gplite_rnd.m:102`)
  and flip whole eigenvectors to one sign convention (`:97-99`). MATLAB
  refuses a factor when a kept eigenvalue is negative and returns an empty
  one, which the draw then multiplies against a vector of another size; on a
  dense one-dimensional grid the kept set holds eigenvalues of rounding size
  and of both signs, and the draw fails. gpyreg takes a negative eigenvalue
  within `10 * n * eps` times the prior variance at the test points, the
  scale of the terms whose difference the predictive covariance is, for a
  zero, and raises `LinAlgError` for one beyond it. No PyVBMC code calls
  `random_function`.
- Why: `verification/wave6.md`, W6-7 and W6-8, and "The independent check
  of the pass" for the scale of the band; entry 46 of
  `matlab_side_defects.md`.
- Kind: deliberate change (gpyreg's repair of a MATLAB-side defect).

### An infinite width of the slice sampler does not return after the burn-in
- Python: `gpyreg/slice_sample.py: SliceSampler`, which copies the base
  widths after an infinite width is replaced.
- MATLAB: `gplite/private/slicesamplebnd.m:168`, `:176`, `:377`.
- What differs: MATLAB copies `basewidths = widths` before `:176` replaces
  an infinite width by 10, and `:377` recombines the adapted widths with
  `sqrt(newwidths.*basewidths)`, so an infinite width comes back after the
  burn-in and the chain is NaN from then on. gpyreg copies after the
  replacement. `gplite_train` and PyVBMC pass finite widths.
- Why: `verification/wave6.md`, W6-14; entry 47 of
  `matlab_side_defects.md`.
- Kind: deliberate change (gpyreg's repair of a MATLAB-side defect).

### The degree-1 Matern gradient is zero where two inputs coincide
- Python: `gpyreg/covariance_functions.py: Matern.compute`, the gradient
  with respect to the length scales at degree 1, which
  `gpyreg/isotropic_covariance_functions.py: MaternIsotropic` shares.
- MATLAB: `gplite/gplite_covfun.m:198`, `:218-219`.
- What differs: at degree 1 MATLAB's derivative factor is `1./t`, so the
  length-scale gradient is `Inf * 0 = NaN` on the diagonal and wherever two
  inputs coincide, and the gradient of the marginal likelihood is NaN for
  every length scale; the repair stands commented out below the line.
  gpyreg sets the gradient to zero there, its value. Degrees 3 and 5 agree,
  and PyVBMC uses the squared exponential.
- Why: `verification/wave6.md`, W6-32; entry 45 of
  `matlab_side_defects.md`.
- Kind: deliberate change (gpyreg's repair of a MATLAB-side defect).

### The slice sampler has no Metropolis step
- Python: `gpyreg/slice_sample.py: SliceSampler` has no Metropolis step, no
  option and no attribute for one.
- MATLAB: `gplite/private/slicesamplebnd.m:189`, the options `MetropolisPdf`
  and `MetropolisRnd`, and the step that interleaves proposals from them
  with the slice moves.
- What differs: MATLAB's sampler can interleave Metropolis moves with a user
  density; gpyreg's cannot. gpyreg 1.2.1 read the option under a misspelt
  key, so its options never turned the step on, while setting its three
  attributes directly did. Neither `gplite_train` nor PyVBMC sets the
  options.
- Why: the step had no caller and no documentation, and one test set its
  attributes; the PI ruled its removal (`verification/wave6.md`, W6-21).
- Kind: removed feature.

## Slice P1a — main loop, warmup, termination, final boost

### Final boost is guarded; MATLAB accepts its result unconditionally
- Python: `pyvbmc/vbmc/vbmc.py:2605` (`final_boost`), `:2813`
  (`_validate_final_boost_tolerance`), `:2829`
  (`_is_valid_final_boost_score`), `:2842`
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
- Python: `pyvbmc/vbmc/vbmc.py:3488` (`output["overhead"] = np.nan`).
- MATLAB: `private/vbmc_output.m`.
- What differs: PyVBMC returns `NaN` where MATLAB returns the fractional
  overhead (total running time / total function time − 1).
- Why: `docsrc/source/faq.md` §"How is `overhead` in `results` defined?"
  ("PyVBMC currently returns `np.nan` for `results["overhead"]`; this field is
  not implemented").
- Kind: unported feature.

### `results["rng_state"]` is a generator snapshot, not MATLAB's `rng` state
- Python: `pyvbmc/vbmc/vbmc.py:3379` (`_get_random_state`, returning
  `{"generator": <bit generator state>}`), `:3489`, `:3416`
  (`_set_random_state`).
- MATLAB: `private/vbmc_output.m` stores `rng` (the global generator state).
- What differs: PyVBMC snapshots the instance's own `numpy.random.Generator`
  at return time, as an independent deep copy, and a run never touches a
  global stream. `load(set_random_state=True)` ignores the per-iteration
  legacy tuple written by versions before the generator migration. The one
  place that writes NumPy's global state is `_set_random_state` on the
  oldest save format, which restores the saved global state in order to
  derive a generator from it, as the comment at that branch says.
- Why: `dev/plans/latent-bug-fixes.md` §"Numerical and state changes"
  (`results["rng_state"]` row: "Return an independent snapshot with the
  existing `{"generator": ...}` format at return-time, after post-loop
  draws") and Phase 1 item 6; `AGENTS.md` §"Randomness goes through
  `numpy.random.Generator` objects".
- Kind: deliberate change.

### `results["problem_type"]` tests the original bounds; MATLAB's never reports bounds
- Python: `pyvbmc/vbmc/vbmc.py:3442` (`_create_result_dict`), `:3456-3461`:
  the field is `"unconstrained"` when every original bound
  (`optim_state["lb_orig"]`, `["ub_orig"]`) is infinite and `"bounded"`
  otherwise.
- MATLAB: `private/vbmc_output.m:5-9` tests `optimState.LB` and
  `optimState.UB`, which `misc/setupvars_vbmc.m:49-50` fills with the
  transformed bounds. The transform sends a finite bound to infinity, so
  `output.problemtype` is always `'unconstrained'`.
- What differs: PyVBMC reports a bound-constrained problem as such. Until
  commit `41ea8b1` (2026-09-20) it shared MATLAB's test and its constant
  answer. The strings differ as well (`"bounded"` for MATLAB's
  `'boundconstraints'`), and `results["best_iter"]` is a 0-based index into
  the iteration history where `output.bestiter` is 1-based.
  `results["iterations"]` is the number of iterations on both sides (commit
  `4822ae1`; it held the index of the last one before). MATLAB also records
  the run time up to the end of each iteration
  (`stats.timer(iter).totalruntime`, `vbmc.m:880`, `:964`); the timers that
  PyVBMC records per iteration hold no such total, of which only the
  commented-out line at `pyvbmc/vbmc/vbmc.py:1720` remains.
- Why: `dev/experiments/port_review_20260919/verification/wave2.md`, rows
  W2-12 and W2-13, with the PI's dispositions; the MATLAB side is entry 2 of
  `dev/experiments/port_review_20260919/matlab_side_defects.md`.
- Kind: deliberate change.

### An empty warm-up stability window means no stability count yet
- Python: `pyvbmc/vbmc/vbmc.py:2074` (`_check_warmup_end_conditions`): with
  `tol_stable_warmup <= fun_evals_per_iter` the window of recent ELCBO
  values is empty on the first check, and `stable_count_flag` stays `False`.
- MATLAB: `private/vbmc_warmup.m:39` reaches the same empty window, `max`
  returns `[]`, and `:87` hands the empty logical to `&&`; that MATLAB raises
  there is inferred from the rules of `&&`, not executed.
- What differs: PyVBMC continues where MATLAB (by that reading) fails. Until
  commit `54f0f19` (2026-09-20) PyVBMC raised in `np.amax`.
- Why: `verification/wave2.md`, row W2-14; entry 4 of
  `matlab_side_defects.md`.
- Kind: deliberate change.

### The true-posterior diagnostic draws from a copy of the generator
- Python: `pyvbmc/vbmc/vbmc.py:3384` (`_compute_true_diagnostic`): the
  10^6 samples behind `sKL_true` are drawn on a deep copy of the posterior
  whose generator is a deep copy too (`:3410-3411`), so the run's stream is
  where it would be without the diagnostic.
- MATLAB: `vbmc.m:772-777` (`vbmc_moments(vp_real,1,1e6)`), drawing from the
  global stream.
- What differs: supplying `true_mean` and `true_cov` does not change a
  PyVBMC run; in MATLAB it moves every later draw.
- Why: the P1b entry "Randomness is threaded through
  `numpy.random.Generator`"; both P1a reviewers read the lines
  (`reviews/P1a_internal.md`, `reviews/P1a_comparison.md`, sheet notes).
- Kind: deliberate change.

### The separate search GP (`SeparateSearchGP`) is not available
- Python: no counterpart. The option `separate_search_gp = False` stays
  declared (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:259`) and
  is registered in `INERT_OPTIONS`.
- MATLAB: `vbmc.m:471`, `:638-648`: on every second iteration a GP with a
  constant mean is trained for the active-sampling search, with its own
  hyperparameter struct (`hypstruct_search`); `defopts.SeparateSearchGP =
  'no'`.
- What differs: PyVBMC always searches with the main GP. Its port of the
  branch shared the main hyperparameter struct and failed in its first
  training call (`verification/wave2_B_loop.md`, row B-5 and the
  orchestrator's correction); the branch was removed in commit `c7a01f3`
  (2026-09-20).
- Why: PI, 2026-09-20 (`verification/wave2.md`, row W2-15): a development
  option of MATLAB VBMC, not needed in PyVBMC.
- Kind: removed feature.

### The closing display line is computed at every display level, on a copy of the generator
- Python: `pyvbmc/vbmc/vbmc.py:1945-1985`, after the loop of `optimize`:
  whenever the returned posterior is not the one the loop ended on (it
  comes from an earlier iteration, or the final boost changed it), the
  symmetrized KL divergence between the two is estimated from 2 x 10^5
  samples drawn on copies of the two posteriors that share a deep copy of
  the run's generator, and the "finalize" line is logged.
- MATLAB: `vbmc.m:884-913` computes the divergence and prints the line
  under the same flag, but only when the display level asks for it
  (`prnt > 2`), and draws from the global stream.
- What differs: PyVBMC computes the divergence at every display level, so
  that the display level does not change what a run does, and the run's
  random stream is where it would be without the line: a run stopped
  without a boost continues as an uninterrupted one would
  (`test_vbmc_resume_optimization`). In MATLAB a run with the display on
  moves the global stream by those draws.
- Why: `verification/wave2.md`, minor observation B-M9 and its row in the
  table of fix commits (PI, 2026-09-20; commits `e9e7803`, `9ff44c5`); the
  P1b entry "Randomness is threaded through `numpy.random.Generator`".
- Kind: deliberate change.

### The final boost of a posterior with fixed means keeps its components at the training inputs
- Python: `pyvbmc/vbmc/vbmc.py:2605` (`final_boost`): with
  `variable_means=False` the boost takes the number of training inputs of
  the GP it is handed as the number of components and sets the means to
  those inputs, as the main loop does for such a posterior in every
  iteration after warm-up; `min_final_components` does not apply.
- MATLAB: `misc/finalboost_vbmc.m:6` takes
  `Knew = max(MinFinalComponents, vp.K)` whatever `VariableMeans` holds,
  and `misc/vbinit_vbmc.m:132-136` then keeps the posterior's own means
  beside `Knew` weights and scales.
- What differs: a fixed-means run that ends with fewer components than
  `min_final_components`, as one stopped during warm-up always does, is
  boosted in PyVBMC and, by a reading of the source, fails in MATLAB.
  PyVBMC raised a broadcast error in `_gp_log_joint` until commit `73d2a81`
  (2026-09-20).
- Why: `verification/wave2.md`, row W2-29 (found during the fix pass;
  PI: fix); entry 14 of `matlab_side_defects.md`.
- Kind: deliberate change.

### MATLAB's `samples` output struct has no counterpart in `results`
- Python: `pyvbmc/vbmc/vbmc.py:3442` (`_create_result_dict`) builds a results
  dict with no equivalent key. The arrays it would hold live on the run's
  `FunctionLogger` (`X_orig`, `y_orig`, `S`, `X_flag`, `n_evals`), which
  `finalize()` trims to the filled rows.
- MATLAB: `vbmc.m:942-957` (the `nargout > 5` block) returns a sixth output
  with `samples.X`, `samples.y`, `samples.y_sd`, `samples.active_flag` and
  `samples.nevals`, taken from `optimState` over `1:optimState.Xn` and scaled
  by the posterior temperature.
- What differs: a PyVBMC caller reads the evaluated points and their values
  from the function logger rather than from the results dict, and no
  temperature scaling is applied (PyVBMC runs at `temperature = 1`; see the
  P7 entry on posterior tempering).
- Why: `dev/experiments/port_review_20260919/verification/wave1_M_P2.md`
  (row "M F4") reads the MATLAB lines and confirms that
  `_create_result_dict`'s key list has no counterpart.
- Kind: unported feature.

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
  verbatim. An option set in the `options_path=` file counts as set by the
  user, as one set in the dict does, and the defaults that depend on other
  options are settled after every source has been read. A name that neither
  shipped file declares raises on every route (the dict, the file,
  `load(new_options=)`; `pyvbmc/vbmc/options.py:485`,
  `pyvbmc/vbmc/vbmc.py:485`, `:3262`), and options are frozen after
  initialization against assignment and removal
  (`options.__setitem__(k, v, force=True)` and
  `options.__delitem__(k, force=True)` override). MATLAB silently accepts
  unknown fields, string-evaluates the fields listed in `evalfields`, and
  its `evalbool` accepts `'yes'`/`'no'` strings that Python does not:
  `uncertainty_handling` is `True`/`False` (or 1/0, or empty for unset) and
  any other value raises (`pyvbmc/vbmc/options.py:165`, `:349`).
- Why: `AGENTS.md` §Architecture ("Options are layered ...");
  `pyvbmc/vbmc/README.md` §"Options Class" (the `is_initialized` flag and the
  `force=True` keyword); the PI's strictness rule of 2026-09-20
  (`dev/plans/port-correctness-review.md`, Decisions) and
  `verification/wave2.md`, rows W2-8, W2-18, W2-20 and the minor
  observation C-M5.
- Kind: deliberate change.

### Randomness is threaded through `numpy.random.Generator`
- Python: `pyvbmc/rng.py: get_rng`, `VBMC(seed=)` → `vbmc.rng`, shared with
  `vbmc.vp`; `train_gp(rng=)` → `gpyreg.GP.fit(rng=)`;
  `pyvbmc/vbmc/active_sample.py` (the `randn` callback passed to
  `cma.fmin`);
  `pyvbmc/vbmc/active_importance_sampling.py` (the slice sampler receives
  `vp.rng`). `gpyreg/rng.py: resolve_rng`.
- MATLAB: `rand`, `randn`, `mvnrnd` on the global stream throughout.
- What differs: a PyVBMC run never reads or writes NumPy's global state, and
  every copy of a variational posterior shares one stream
  (`VariationalPosterior.__deepcopy__`). `seed=None` derives the generator
  from the global `np.random` state so that `np.random.seed()` beforehand
  still fixes a run; deriving it draws four integers from that state, which
  advances it, and that construction is the only contact with it.
  Consequently no Python draw sequence can be compared point by point with a
  MATLAB draw sequence. The order of the draws differs in one place as well:
  in an active-sampling step of a noisy run PyVBMC draws the search
  candidates (`pyvbmc/vbmc/active_sample.py:382-384`) before the importance
  samples (`:417-420`), and MATLAB after them
  (`private/activesample_vbmc.m:208-218`).
  Neither routine changes what the other reads, so the order changes the
  stream and nothing else.
- Why: `AGENTS.md` §"Randomness goes through `numpy.random.Generator`
  objects" (with the dates: the global-state reseeding ended 2026-09-05, the
  IMIQR path 2026-09-10); `dev/plans/stage1-rng-generator.md`;
  gpyreg `AGENTS.md` §"Random number handling".
- Kind: deliberate change.

### Vectorized targets, precomputed evaluations and the initialization cost
- Python: `pyvbmc/vbmc/vbmc.py:757` (`_validate_initialization_cost`),
  `:809` (`_initialize_precomputed_evaluations`), `:982`
  (`_fresh_evaluations_for_batch`), `:3778`
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
- Python: `pyvbmc/vbmc/vbmc.py:3358` (`_optim_state_record`, setting
  `record["active_importance_sampling"] = None` at `:3376`); option
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

### The default search acquisition is the log form
- Python: `search_acq_fcn = [AcqFcnLog()]`
  (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:39`);
  `pyvbmc/acquisition_functions/acq_fcn_log.py`.
- MATLAB: `defopts.SearchAcqFcn = '@acqf_vbmc'` (`vbmc.m:213`);
  `acq/acqf_vbmc.m`, with the log form in `acq/acqflog_vbmc.m`.
- What differs: with `u` the product of the variance, the exponentiated
  mean and the posterior density, the plain form is `-u` and the log form
  `-log u`. They rank candidates identically in exact arithmetic, the
  variance regularization and the bound penalty included, so the search has
  the same minimizer. In floating point the plain form underflows to zero
  and ties candidates that the log form still ranks: on the eight oracle
  states it returns exact zeros on up to all 512 candidates.
- Why: commit `e20d081` (2022-09-21), pull request 102 ("in order to avoid
  overflow warnings in `exp()`"); kept by the PI on 2026-09-20
  (`verification/wave2.md`, row W2-25;
  `verification/wave2_C_setup.md`, C-9, for the measurement).
- Kind: deliberate change.

### `integer_vars` takes a boolean mask or 0-based indices
- Python: `pyvbmc/vbmc/options.py:251` (`integer_vars_mask`), read at
  `pyvbmc/vbmc/vbmc.py:1048`. A boolean array of length `D` is a mask; an
  integer array holds 0-based indices, distinct and in range; an integer
  array of length `D` holding only zeros and ones reads as either and is
  refused with a message asking for a boolean mask.
- MATLAB: `misc/setupvars_vbmc.m:14-17` reads 1-based indices.
- What differs: the index base, and the mask form, which MATLAB does not
  have. PyVBMC also checks what MATLAB's message demands and its code does
  not: that the hard bounds of an integer variable sit half an integer
  outside its range (next entry).
- Why: `verification/wave2.md`, row W2-17, with the PI's type-strict ruling;
  until commit `5b3b093` (2026-09-20) the option was read as a mask only,
  and a plain list marked every variable.
- Kind: deliberate change.

### PyVBMC does not reproduce two defects of MATLAB's input checks
- Python: `pyvbmc/vbmc/_bounds.py` (`_normalize_bounds`) repairs every
  coordinate whose estimated plausible bounds coincide;
  `pyvbmc/vbmc/vbmc.py:1048-1061` checks that the hard bounds of an integer
  variable are finite and half an integer outside its range.
- MATLAB: `misc/boundscheck_vbmc.m:27-30` computes `idx = any(PLB == PUB)`,
  one logical, and repairs the first coordinate only;
  `misc/setupvars_vbmc.m:19-22` tests `floor(LB(d)) ~= 0.5`, which holds for
  every value, so only the finiteness of the bound is checked.
- What differs: PyVBMC implements what the two MATLAB checks are written to
  do.
- Why: entries 10 and 11 of `matlab_side_defects.md`
  (`verification/wave2_C_setup.md`, C-C5 and C-C12).
- Kind: deliberate change.

### `display` takes `"off"`, `"iter"` and `"full"`
- Python: `display = "iter"`
  (`pyvbmc/vbmc/option_configs/basic_vbmc_options.ini:3`);
  `pyvbmc/vbmc/vbmc.py:3584` (`_init_logger`, `:3603-3608`): `"off"` logs
  warnings only, `"iter"` one line per iteration, `"full"` debugging detail;
  any other value is taken as `"iter"`.
- MATLAB: `defopts.Display = 'iter'` with `'iter'`, `'notify'`, `'final'`
  and `'off'` (`vbmc.m:158`, `:388-399`).
- What differs: the accepted values. PyVBMC has no `'notify'` or `'final'`
  level and adds `"full"`; both fall back on the per-iteration level.
- Why: the display goes through Python's `logging`, whose levels the three
  values name (commit `c477a7b`, 2021-11-01);
  `verification/wave2_C_setup.md`, C-C8.
- Kind: deliberate change.

### Declared options that nothing reads
- Python: `pyvbmc/vbmc/options.py:23-53` (`INERT_OPTIONS`, twenty-seven
  names) and `:538` (`_warn_inert_options`, called from
  `validate_option_names`, `:533`). The `# description` line of each name in
  `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini` says that the option
  is not used and why; those comments are the user documentation.
- MATLAB: four of the names have no MATLAB counterpart any more
  (`GPStochasticStepsize`, `EmpiricalGPPrior`, `IntegrateGPMean`, `DoubleGP`,
  whose `defopts` lines commit `2044530` (2021-06-18) deleted); two are still
  declared in MATLAB with no reader (`defopts.AnnealedGPMean` and
  `defopts.ConstrainedGPMean`, `vbmc.m:305-306`, whose readers the same
  commit removed from `misc/gptrain_vbmc.m`); the rest name features PyVBMC
  never ported.
- What differs: PyVBMC keeps the keys and their declared defaults, so option
  dictionaries recorded by earlier runs still load, and warns at construction
  when one of them is supplied at a value other than its default, naming the
  option. An inert option whose declared default is a callable
  (`annealed_gp_mean`) is left uncompared, because two functions cannot be
  told apart by value and a recorded option set repeats every default.
  MATLAB accepts and ignores such fields silently. Besides the six options
  above, the set holds the knobs of features that were never ported (the
  acquisition-portfolio hedge, variational active sampling and the sampling
  of the variational parameters, the repository of previous variational
  posteriors, output warping's callback, per-stage option overrides, noise
  shaping, posterior tempering through `temperature`, which MATLAB
  validates and reads in six files, and the separate search GP, see the P1a
  entry), `diagnostics` (MATLAB reads `Diagnostics` at `vbmc.m:805` and
  `:961` to decide whether the GP stays in the returned `stats`; PyVBMC's
  history always keeps a lean GP, with `record_full_history_details` as the
  nearest knob), `active_sample_fess_thresh`
  (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:299`, where
  `pyvbmc/vbmc/active_sample.py:760` fixes `fESS_thresh = 1`),
  `active_importance_sampling_fess_thresh`
  (`advanced_vbmc_options.ini:301`, the threshold of the
  MCMC refinement of the variational importance samples, which is not
  ported; see the P4 entry), `cov_sample_thresh`
  (`advanced_vbmc_options.ini:205`, read by covariance sampling alone, which
  is not ported; see the P5 entry on the hyperparameter sampler) and
  `search_cmaes_best` (`advanced_vbmc_options.ini:185`, see the P2 entry on
  the `cma` package).
- Why: commits `a63b17a` "docs(options): mark the declared options that
  nothing reads" and `0d9a422` "fix(options): leave a callable inert default
  uncompared", both 2026-09-19, on the scan recorded in
  `dev/experiments/port_review_20260919/verification/wave1_options.md`
  (row O-1); commits `82c624c` and `c7a01f3` (2026-09-20) for `temperature`,
  `diagnostics` and `separate_search_gp` (`verification/wave2.md`, rows
  W2-22 and W2-15). A test recomputes the set by scanning the package,
  comments dropped, for reads through an options mapping
  (`options["name"]`, `options.get("name")`, `options.eval("name", ...)`,
  and the same through `self` inside `Options`), so a newly dead option, or
  a newly read registered one, fails it; a key of another mapping that
  carries an option's name, such as an entry of `optim_state`, does not
  count as a read.
- Kind: deliberate change.

### `misc/vbmc_gphyp.m` is an empty file in the comparison target
- Python: `pyvbmc/vbmc/gaussian_process_train.py:276` (`_gp_hyp`).
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
- Python: `pyvbmc/vbmc/active_sample.py:5` (`import cma`), the options built
  at `:558-578` and the `cma.fmin` call at `:589-595`.
- MATLAB: `utils/cmaes_modded.m`, called from
  `private/activesample_vbmc.m:282-283` with the settings that
  `misc/setupoptions_vbmc.m:166-178` assembles.
- What differs: the acquisition search uses the third-party `cma` package
  with `seed=np.nan` and a `randn` callback bound to the run's generator, so
  neither the population draws nor the restart logic are MATLAB's. Of the
  settings the two packages both express, three agree: the overall step size
  is `max(insigma)` and the coordinate scaling `insigma / max(insigma)`
  (`CMA_stds`, which cma holds in a non-adapting `sigma_vec`), matching
  `sigma` and `diagD` at `utils/cmaes_modded.m:561`, `:567`; the search runs
  without cma's noise handler, matching `defopts.Noise.on = 0`
  (`utils/cmaes_modded.m:215`); and `tolfun` and `maxfevals` carry the
  acquisition tolerance and `search_max_fun_evals`. Four do not.
  `TolX` is `1e-11*max(insigma)` in MATLAB (`misc/setupoptions_vbmc.m:168`)
  against cma's absolute `tolx = 1e-11`; `TolHistFun = 1e-13`
  (`misc/setupoptions_vbmc.m:170`) against cma's `tolfunhist = 1e-12`;
  `EvalInitialX = 'yes'` (`utils/cmaes_modded.m:207`) makes MATLAB evaluate
  `x0` and charge it to the budget, while `cma.fmin`'s `eval_initial_x`
  defaults to `False` and PyVBMC does not pass it; and `cma.fmin` returns the
  best-ever solution, where MATLAB returns the best point of the last
  generation (`utils/cmaes_modded.m:1708`) unless `SearchCMAESbest` is set
  (`vbmc.m:282`, default `no`). PyVBMC's `search_cmaes_best`
  (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:185`) is therefore
  read nowhere: the search always takes the best-ever point.
- Why: `dev/plans/port-correctness-review.md` §Slices, P2 ("`utils/cmaes_modded.m`
  (PyVBMC uses the `cma` package)"); `AGENTS.md` §Architecture;
  `dev/plans/stage2-batched-acquisition.md` §"Findings the plan rests on";
  the settings are compared one by one in
  `dev/experiments/port_review_20260919/verification/wave1_M_P2.md`. The
  coordinate scaling and the absence of the noise handler date from
  2026-09-19, commits `4d1d984` "fix(active_sample): start the CMA-ES search
  at the per-coordinate scales" and `d617d99` "fix(active_sample): run the
  CMA-ES search without noise handling"; the second also deleted the
  `_BatchedNoiseHandler` subclass, which had drawn its re-evaluation count
  from `vp.rng`. The four settings that do not agree stay as they are
  (PI, 2026-09-19): the two tolerances sit far below `tolfun`, which
  stops the search first; the starting point's acquisition value is
  the sieve's, the search result is kept only when it beats that value
  (`pyvbmc/vbmc/active_sample.py:633`), and MATLAB spends two more
  acquisition evaluations per search (a pointwise re-evaluation before
  the search and cma's initial evaluation) for the same outcome; and
  the best-ever point is the best while the acquisition is fixed.
- Kind: substituted library.

### The CMA-ES search evaluates the whole population in one acquisition call
- Python: `pyvbmc/vbmc/active_sample.py:594` (`parallel_objective=acq_fun` in
  the `cma.fmin` call); the same wrapper is passed as the pointwise objective
  as well, for cma's resampling path.
- MATLAB: `utils/cmaes_modded.m:880`, `:917` hand the whole population matrix
  to the objective when `EvalParallel` is set, which
  `misc/setupoptions_vbmc.m:171` does for the acquisition search;
  `acq/acqwrapper_vbmc.m:5` transposes the matrix it receives. A point
  resampled because its value came back `NaN` is evaluated on its own
  (`utils/cmaes_modded.m:925`).
- What differs: how many points reach the acquisition per generation is
  MATLAB's, one call carrying one population. What belongs to the Python line
  is that a single acquisition wrapper serves both the batched and the
  pointwise route, and that the batched route carries
  `AbstractAcqFcn._sq_dist`'s dependence on the batch: the centering uses a
  size-weighted mean of the two point sets, so a point's value computed in a
  batch can differ from its pointwise value by about 2e-15 relative, which
  can flip a near-tie.
- Why: `dev/plans/stage2-batched-acquisition.md` §Summary and its "Findings
  the plan rests on"; `dev/2026-09-02-modernization-discussion.md` §9
  ("`_sq_dist` ... Not a bug, but it is why the noisy acquisitions evaluated
  one point at a time and in a batch can differ by a finite amount on a
  near-tie"); retained explicitly in `dev/plans/latent-bug-fixes.md`
  §"Supporting fixes and closed entries" ("Preserve the Stage 2 arithmetic;
  no change in this plan"). A second batch per generation, cma's
  noise-handler re-evaluations, was removed on 2026-09-19 by commit
  `d617d99` "fix(active_sample): run the CMA-ES search without noise
  handling".
- Kind: deliberate change.

### `search_optimizer` takes different values, and `D == 1` is a scalar search
- Python: `pyvbmc/vbmc/vbmc.py: _validate_search_optimizer_option` and
  `pyvbmc/vbmc/active_sample.py:511-631`. The option `search_optimizer`
  (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini`, default
  `"cmaes"`) takes `"cmaes"` or `"none"`; any other value is refused at
  construction and in `load`. Under `"cmaes"`, when `gp.D == 1` a local
  variable selects a bounded scalar search in place of CMA-ES
  (`scipy.optimize.minimize_scalar(method="bounded")`, `:600-625`), and
  the run's options are left as the user set them. Under `"none"` no
  local search runs, `gp.D == 1` included: the override sits inside the
  test of `search_optimizer != "none"`.
- MATLAB: `private/activesample_vbmc.m:246`, `:265-315`;
  `defopts.SearchOptimizer = 'cmaes'` (`vbmc.m:280`). The accepted values are
  `cmaes`, `fmincon`, `bads`, `slicesample` and `none`, anything else raising
  at `:314-315`, and there is no special case for `D == 1`.
- What differs: MATLAB's bounded `fmincon`, `bads` and the `slicesample`
  branch have no Python counterpart. At `D == 1` PyVBMC brackets the
  acquisition on the whole search interval rather than descending from the
  best candidate of the search set, with `maxiter` set to
  `search_max_fun_evals` and `xatol = 1e-11`, the step size at which the
  CMA-ES branch stops; the comparison against that candidate's value still
  decides whether the result is acquired, as in the other branch.
- Why: the interval search and the local override date from 2026-09-19,
  commit `cb8a51d` "fix(active_sample): search a one-dimensional acquisition
  over its interval". `cma.fmin` runs at `D = 1`, so a separate
  one-dimensional branch is not forced by the package
  (`dev/experiments/port_review_20260919/verification/wave1_M_P2.md`, P2 F9).
  Until 2026-09-21 a third value, `"Nelder-Mead"`, stood in for `fmincon`
  with SciPy's simplex, without the search bounds, without
  `search_max_fun_evals` and with the tolerance of the value applied to the
  step; release 1.0.4 forced every one-dimensional run onto it, and no
  default path reached it after the interval search. The PI removed the
  value with its code
  (`dev/experiments/port_review_20260919/verification/wave5.md`, W5-25).
  `load` replaces a stored `"Nelder-Mead"` by `"cmaes"` for a problem of one
  dimension, because release 1.0.4 wrote the value into the options of every
  such run, and refuses it otherwise.
- Kind: substituted library.

### The initial design does not cluster surplus starting points
- Python: `pyvbmc/vbmc/active_sample.py:174-176` ("In the MATLAB
  implementation there is a cluster algorithm being used to pick the best
  points, but we decided not to implement that yet and just pick the first
  sample_count points"), with `:184-185` marking for removal only the rows
  the design consumes.
- MATLAB: `misc/initdesign_vbmc.m:28-43`, using `utils/fastkmeans.m`; `:34`
  starts from `idx_remove = false(N0,1)` and `:42` marks the chosen point of
  each cluster.
- What differs: when more than `fun_eval_start` starting points are supplied,
  MATLAB clusters them with k-means and takes the highest-density point of
  each cluster; PyVBMC takes the first `sample_count` rows and logs an
  informational message. Both leave the other `N0 - Ns` rows in the starting
  cache with their values, where the search sieve can take them as candidates
  that need no target call, so only the selection rule differs. The
  `N0 <= Ns` branch (uniform draws in the plausible box, or the `narrow`
  design) is ported faithfully.
- Why: the code comment above is the record of the decision. That the rows
  beyond the design stay in the cache was restored on 2026-09-19 by commit
  `181a63b` "fix(active_sample): keep the starting points the initial design
  did not use", which also made the sieve complete its search set to the
  requested size after taking its share of the cache.
- Kind: unported feature.

### An acquired starting point leaves the cache whether or not it had a value
- Python: `pyvbmc/vbmc/active_sample.py:722-730`: once a point drawn from the
  starting cache has been acquired, its row is deleted from every cache array
  (`x_orig`, `y_orig`, `skip_logger`), whether its value was stored there or
  has just been evaluated.
- MATLAB: `private/activesample_vbmc.m:380-392` deletes the row only in the
  branch that reuses a stored value (`:389-391`); a cached point whose
  `y_orig` is `NaN` is evaluated through the target and left in the cache.
- What differs: in PyVBMC a cached starting point can be acquired once. In
  MATLAB a cached row without a stored value stays in the cache after it has
  been evaluated, so the sieve can draw it again and spend a second
  evaluation on the same input. Such rows exist only when more starting
  points are supplied than the initial design consumes.
- Why: commit `a7f323e` (2026-09-19) "fix(active_sample): drop an acquired
  starting point from the cache in every case", whose message records the
  departure from MATLAB as deliberate.
- Kind: deliberate change.

### The stored value of a cached starting point is reused at that point alone
- Python: `pyvbmc/vbmc/active_sample.py:696-711`: the value stored with a
  cached starting point is recorded without a target call only when the
  acquired candidate equals `parameter_transformer(cache["x_orig"][idx])`
  exactly; a candidate that the clip into the search box
  (`_get_search_points`) or the snap to the integer grid (`_real2int`) moved
  is evaluated through the target. The row leaves the cache either way.
- MATLAB: `private/activesample_vbmc.m:555`, `:637`, `:219`, `:388`: the
  cached rows are warped, clipped and snapped with the rest of the search
  set and keep their cache index, and `funlogger_vbmc(..., 'add', y_orig)`
  records the stored value at the candidate as moved.
- What differs: MATLAB can record a value at a point it was not computed at;
  PyVBMC calls the target there. The clip is out of reach at the default
  `active_search_bound`, the starting points lying in the plausible box,
  which the search box contains with a margin through a warp; the snap needs
  `integer_vars`, more starting points than the initial design takes, their
  values provided, and one of them off the grid.
- Why: `dev/experiments/port_review_20260919/verification/wave5.md`, W5-7
  (a defect shared with MATLAB, `matlab_side_defects.md`, entry 40; fixed on
  the PI's ruling of 2026-09-21).
- Kind: deliberate change.

### The fractions of the acquisition search are checked, and the search cache holds no training input
- Python: `pyvbmc/vbmc/vbmc.py: _validate_search_fraction_options` checks at
  construction and in `load` that `search_cache_frac`,
  `heavy_tail_search_frac`, `mvn_search_frac`, `hpd_search_frac` and
  `box_search_frac` each lie in `[0, 1]` and sum to at most 1, and that
  `cache_frac`, the share of the whole search set that the starting cache
  gives, lies in `[0, 1]`;
  `pyvbmc/vbmc/active_sample.py: _get_search_points` (`capped_share`)
  gives each source its rounded share or what the sources before it
  left, whichever is smaller, in the order search cache, heavy-tailed,
  multivariate normal, high-posterior-density, box, the variational
  posterior drawing the rest;
  `pyvbmc/vbmc/active_sample.py:434-443` leaves the training inputs that
  head the search set of a noisy target (the candidates for a repeated
  observation) out of `optim_state["search_cache"]`.
- MATLAB: `private/activesample_vbmc.m:565-633` (`getSearchPoints`) checks
  nothing: fractions that add up to more than one give a search set longer
  than its vector of cache indices (`matlab_side_defects.md`, entry 35).
  `CacheFrac` is unchecked too (`:552-554`): above one, with a starting
  cache of more rows than `NSsearch`, the search set is larger than asked.
  The shares are rounded and not capped (`:627`,
  `Nvp = max(0, ...)`), so fractions that sum to one do the same where
  the rounded shares claim more than `Nrnd`: three quarter-shares of two
  points are three points. MATLAB never puts training inputs into the
  search set, so its search cache cannot hold one.
- What differs: PyVBMC refuses a configuration that MATLAB mishandles, and
  returns the number of candidates it is asked for where MATLAB's rounded
  shares can return more. Where no cap binds the counts are MATLAB's. With
  the shipped fractions a `search_cache_frac` above 0.25 is refused; until
  2026-09-21 such a run raised at its second active-sampling step, the
  search cache being empty at the first. What the two share stays: the
  search cache, where one is kept, is written before the acquired point is
  deleted from the search set and so keeps that point in first place
  (unless, in PyVBMC, the point is a training input, which the cache
  leaves out), the deletion binds
  arrays that nothing reads again, and a row of the starting cache loses its
  cache index on the way through the search cache
  (`private/activesample_vbmc.m:230-242`, `:567`, `:633`).
- Why: `dev/experiments/port_review_20260919/verification/wave5.md`, W5-24.
- Kind: deliberate change.

### Repeated observations are selected differently
- Python: `pyvbmc/vbmc/active_sample.py:399-409` (training inputs are
  appended to the sieve's search set when
  `max_repeated_observations > 0`, the target is noisy, and the streak is
  below the cap; a chosen repeat skips the local optimizer so the logger
  pools it into its row) and `:738-743` (streak bookkeeping). The
  corresponding MATLAB block is kept, commented out, at `:642-685`
  ("Missing port"), and `:687` records the unported
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

### The rank-one GP update is taken for a fresh observation, noisy or not
- Python: `pyvbmc/vbmc/active_sample.py:826-829`:
  `update1 = function_logger.n_evals[idx_new] == 1 and not
  options["noise_shaping"]`.
- MATLAB: `private/activesample_vbmc.m:481`:
  `update1 = (isempty(s2new) || optimState.nevals(idx_new) == 1) &&
  ~options.NoiseShaping`.
- What differs: both extend the GP posterior by rank one when the observation
  adds a training row, and recompute the whole posterior when it is pooled
  into an existing row or when noise shaping rescales every training point's
  noise. The two conditions part in one cell, a repeated input on a noiseless
  target, where `s2new` is empty: MATLAB takes the rank-one path and appends
  a GP training row for an input that `optimState.X` holds once, because
  `misc/funlogger_vbmc.m:229-248` pools a repeat into the existing row
  without incrementing `Xn`; PyVBMC recomputes the posterior from the
  logger's training set. On a noisy target the two paths differ in every
  cell: `gplite/gplite_post.m:76-79` refuses a rank-one update whenever a
  noise variance is supplied, so at uncertainty levels 1 and 2 MATLAB
  appends the row and recomputes every posterior with `gplite_core`, while
  PyVBMC extends the factors by rank one with `s2_new`. gpyreg's extension
  is the generalization of MATLAB's to unequal noise (it keeps the scale of
  the factorization, `sl`, where MATLAB takes the scale to be the new
  point's effective noise, which represents the extended matrix only to
  0.116 relative once the noise varies by point) and reproduces a rebuild at
  the same hyperparameters to about 1e-15 in `alpha`, in the predictions and
  in the `C_tmp` that the noisy acquisitions read, in the Cholesky
  representation, the only one a PyVBMC run enters (the noise variance of a
  run is at least `tol_gp_noise**2` = 1e-5, the option being a floor on the
  standard deviation, and the representation's threshold is 1e-6), also for
  a new point whose noise lies below every existing one.
  What is left is the cost of a step, O(N^2) against O(N^3), and two
  differences of state: after a retried Cholesky factorization the rank-one
  update keeps the stored `sn2_mult`, where a recomputation derives it again
  (with a multiplier of 100 forced through a patched factorization, 0.75
  relative in `alpha` against a rebuild that needs none); and when the new
  point's total noise falls below 1e-6 while the training minimum stays
  above it, the rank-one update keeps the Cholesky representation and the
  stored `sl` where a rebuild takes the other representation (the numbers
  agree, to 1e-13 in `alpha`). In that other representation the extension
  recomputes in full, with a warning, where the predictive variance of the
  new point is at or below its noise, that is, where rounding drove the
  latent variance to zero or below and `predict` clamped it, the analogue of
  the Cholesky branch's test; a tiny positive latent variance proceeds, and
  the extension is inaccurate there, as the Cholesky branch is for a tiny
  positive argument of its square root (`verification/wave6.md`, W6-9).
- Why: commit `510a493` (2026-09-19) "fix(active_sample): rank-one GP update
  for a fresh noisy observation". Transcribing MATLAB's condition literally
  would import the duplicate training row, which
  `dev/experiments/port_review_20260919/verification/wave1_M_P2.md` (P2 F8)
  identifies as a MATLAB-side defect. The equivalence of the two paths on a
  noisy target: `verification/wave4.md`, W4-11, with
  `verification/scripts/wave4_P4_5_rank_one_update.py` and
  `wave4_P4_5b_sn2_mult.py`; the two states and the branch of the figure:
  `verification/wave6.md`, W6-20, with `wave6_G1.md`, rows G1-15 and G1-38.
- Kind: deliberate change.

### The acquisition-portfolio hedge (`acqhedge_vbmc.m`) is not ported
- Python: `pyvbmc/vbmc/vbmc.py: _validate_acq_hedge_option` refuses
  `acq_hedge=True` at construction and in `load` with a
  `NotImplementedError` that names the unported hedge;
  `pyvbmc/vbmc/active_sample.py:321-327` picks `idx_acq` at random among the
  entries of `search_acq_fcn`, in the one branch a run can take. The options
  `acq_hedge_iter_window` and `acq_hedge_decay` are declared, registered as
  inert and read nowhere
  (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini`).
- MATLAB: `private/acqhedge_vbmc.m`, called from
  `private/activesample_vbmc.m` and `vbmc.m:848`.
- What differs: with several acquisition functions, MATLAB runs a hedging
  portfolio that tracks each one's recent improvement; PyVBMC picks one
  uniformly at random per active-sampling step, has no hedge at all and
  refuses the option that asks for one. MATLAB assigns `idxAcq` through the
  hedge only with more than one acquisition function
  (`private/activesample_vbmc.m:22-26`), so its own default, one acquisition
  with `AcqHedge` on, reads a variable that was never set
  (`matlab_side_defects.md`, entry 33).
- Why: `pyvbmc/vbmc/README.md` §"Porting status"
  ("`acqhedge_vbmc.m` has not been ported yet as it is considered to be
  experimental"); `dev/plans/port-correctness-review.md` §Slices, P2 marks it
  unported. The refusal dates from 2026-09-21: until then the option was
  accepted, `idx_acq` was never assigned with it on, and the first
  active-sampling step raised `UnboundLocalError` after the initial design
  (`dev/experiments/port_review_20260919/verification/wave5.md`, W5-23).
- Kind: unported feature.

### Variational active sampling (`VarActiveSample`) is not ported
- Python: no counterpart; the option `active_variational_samples = 0` exists
  (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:273`) but nothing
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
- Python: `pyvbmc/acquisition_functions/abstract_acq_fcn.py:294`, `:330`
  (`X_2d[:, integer_vars] = X_temp[:, integer_vars]` writes into the
  caller's array, a single point `(D,)` through a two-dimensional view of
  it); called from `pyvbmc/vbmc/active_sample.py:386`, `:634` and from
  `AbstractAcqFcn.__call__` (`abstract_acq_fcn.py:85`).
- MATLAB: `misc/real2int_vbmc.m` returns a new array.
- What differs: through the `Xs[None, :]` view of a 1-D input, the pointwise
  CMA-ES objective snapped CMA-ES's own solution arrays to the integer grid.
  The batched objective reproduces that side effect deliberately. The
  rounding itself is MATLAB's `round`, a half away from zero.
- Why: `dev/2026-09-02-modernization-discussion.md` §9 ("an undocumented side
  effect that the batched objective now reproduces deliberately");
  `dev/plans/latent-bug-fixes.md` §"Supporting fixes and closed entries"
  ("Deliberately preserved by the Stage 2 batched CMA-ES objective. Retain
  its current behavior").
- Kind: deliberate change.

---

## Slice P3 — acquisition functions

### The EIG acquisition and two experimental VIQR losses were removed
- Python: `AcqFcnEIG` and its module `acq_fcn_eig.py` are absent from
  `pyvbmc/acquisition_functions/`, and `AcqFcnVIQR` has no `loss` variants
  `var_reduction` and `sd_reduction`. `acq_fcn_viqr.py:122`:
  `LOSSES = ("iqr", "iqr_reduction")`. The branch
  `retain/experimental-acquisitions` at `fa6922f` holds the EIG acquisition
  and all four losses.
- MATLAB: `acq/acqeig_vbmc.m` and its helper `misc/intkernel.m`;
  `acq/acqviqr_vbmc.m` has the one loss.
- What differs: the package has no expected-information-gain acquisition.
  `AcqFcnVIQR` offers `loss="iqr"`, the default and MATLAB's acquisition,
  and `loss="iqr_reduction"`, which has no MATLAB counterpart and is
  documented, validated, tested
  (`pyvbmc/testing/acquisition_functions/test_acq_fcn_viqr_losses.py`) and
  listed in `CHANGELOG.md`. It is the one path of VIQR that reads the
  importance weights (`_log_iqr_reduction`); the `iqr` loss ignores them, as
  the commented-out `lnw` lines of `acqviqr_vbmc.m:103-105` do.
- Why: PI decision 2026-09-13, recorded in
  `dev/plans/modernization-roadmap.md:58-67` and `dev/TODO.md`
  §"Outside 1.5 scope"; the measurements are in
  `dev/results/2026-09-08-noisy-acquisition-experiments.md` (scalar EIG poor,
  per-component EIG mixed, neither robustly better than default VIQR). The
  removal took effect 2026-09-14 (`fd9c7e8`, whose message names the two
  losses that stay).
- Kind: removed feature.

### The quantile of VIQR and IMIQR is an argument, and `u` is computed from it
- Python: `pyvbmc/acquisition_functions/acq_fcn_viqr.py:139` and
  `acq_fcn_imiqr.py:35`: `self.u = norm.ppf(quantile)`, 0.6744897501960817
  at the default `quantile=0.75`; both constructors refuse a quantile that is
  not strictly between 0.5 and 1 (`abstract_acq_fcn.py:261`,
  `_check_quantile`).
- MATLAB: `acq/acqviqr_vbmc.m:4` and `acq/acqimiqr_vbmc.m:4`:
  `u = 0.6745; % norminv(0.75)`.
- What differs: MATLAB's literal is the value the port computes, rounded to
  four decimals, 1.5e-5 above it in relative terms. It is the whole
  difference between the port and MATLAB's formulas of the two acquisitions:
  on the stored noisy state of the oracles the values move by up to 1.7e-5
  (VIQR) and 8e-5 (IMIQR), with the same best candidate, and with the same
  constant on both sides the port agrees with a transcription of the MATLAB
  files to 2e-14.
- Why: pull request 80 (`70325a3`, 2022-06-02), "The quantile for VIQR/IMIQR
  acquisition functions can now be specified as an argument (default is
  0.75)"; the port had the literal until then. A literal cannot serve a
  quantile that the user chooses. `verification/wave4.md`, W4-8.
- Kind: deliberate change.

### A NaN acquisition value stays a NaN
- Python: `pyvbmc/acquisition_functions/abstract_acq_fcn.py:179`:
  `np.maximum(acq, -realmax)`, which lets a NaN through.
- MATLAB: `acq/acqwrapper_vbmc.m:47`: `max(acq,-realmax)`, which drops a NaN
  and returns `-realmax`, the most attractive value for the search.
- What differs: only what happens to a NaN. No shipped acquisition returns
  one: the quantiles of VIQR and IMIQR that did are refused at construction,
  and a row of a log sum that is all `-inf` gives `-inf` in the port, which
  becomes the same `-realmax` that MATLAB reaches through a NaN.
- Why: `verification/wave4.md`, W4-22: MATLAB's answer would make the point
  with the NaN the best candidate of the search.
- Kind: deliberate change.

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
- Python: `pyvbmc/acquisition_functions/acq_fcn_viqr.py:296` ("Missing port,
  integrated mean function, lines 49 to 57") and `:363` ("Missing port,
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
- Python: `pyvbmc/vbmc/active_importance_sampling.py:243`
  (`gpr.slice_sample.SliceSampler`, seeded from `vp.rng`), `:402`
  (`get_mcmc_opts`).
- MATLAB: `private/activeimportancesampling_vbmc.m` calls
  `eissample_lite` (`gplite/private/eissample_lite.m`,
  `utils/eissample_lite.m`).
- What differs: a coordinate-wise bounded slice sampler replaces MATLAB's
  ensemble slice sampler. The samples, their correlation structure and the
  random stream all differ; only the target density is shared. PyVBMC runs
  one chain per GP hyperparameter sample, from one starting point drawn
  among the proposal samples in proportion to their importance weights,
  where MATLAB draws `2(D+1)` walkers the same way, without replacement
  (`private/activeimportancesampling_vbmc.m:206-214`).
- Why: `dev/plans/port-correctness-review.md` §Slices, P4. Also
  `AGENTS.md` §"Randomness ..." (the IMIQR path moved onto `vp.rng` on
  2026-09-10, when the `acq_AcqFcnIMIQR` oracle was re-baselined).
- Kind: substituted library.

### The MCMC refinement of the variational importance samples is not ported
- Python: no counterpart. An acquisition that sets
  `acq_info["mcmc_importance_sampling"]` is refused with a
  `NotImplementedError`, at construction for the objects of
  `options["search_acq_fcn"]` (`pyvbmc/vbmc/vbmc.py:3868`,
  `_validate_search_acq_fcn_option`) and in
  `pyvbmc/vbmc/active_importance_sampling.py:50` for any other caller. The
  option `active_importance_sampling_fess_thresh`, which the branch alone
  read, is registered as inert.
- MATLAB: `private/activeimportancesampling_vbmc.m:57-92`: when the
  acquisition's `acqInfo` sets the flag and the fractional effective sample
  size of the samples drawn from the variational posterior falls below
  `ActiveImportanceSamplingfESSThresh`, one step of the ensemble sampler
  `eissample_lite` moves all `Na` samples, as `Na` walkers. No MATLAB
  acquisition sets the flag.
- What differs: PyVBMC has no such step. Until the port review's wave 4 the
  module held a transcription of the branch that handed gpyreg's
  `SliceSampler` the matrix of walkers; the sampler takes one starting point
  and raised whenever the branch was reached, in every revision since the
  branch was written (`e38351a`, 2022-05-23). Reaching it took an acquisition
  that set `variational_importance_sampling` as well and a fractional
  effective sample size below the threshold; with the flag alone the run
  went on as if the flag were not there. The step itself never ran.
- Why: PI, 2026-09-20 (`verification/wave4.md`, W4-6). Decision D5 of
  `dev/plans/latent-bug-fixes.md` ("Do not delete custom acquisition hooks")
  kept the branch on the premise that it worked.
- Kind: unported feature.

### The importance log weights are normalized
- Python: `pyvbmc/vbmc/active_importance_sampling.py:313`, `:490`
  (`renormalize_weights`): one log-sum-exp over the whole `(Ns_gp, Na)` array
  is subtracted from the log weights before they are returned.
- MATLAB: `private/activeimportancesampling_vbmc.m` stores `lnw` as computed.
- What differs: a constant. The value of IMIQR, and of VIQR with
  `loss="iqr_reduction"`, differs from MATLAB's by that constant (6.16 on the
  stored noisy state of the oracles; `log(Ns_gp * Na)` for VIQR). Everything
  that is added to or compared with an acquisition value afterwards is
  unchanged by it: the variance regularization, the masks, the ranking of the
  candidates, the acceptance of the local search's result and the tolerances
  of the search.
- Why: present since `70325a3` (2022-06-02, pull request 80), the file's
  second commit, without a recorded reason; kept because it moves no
  decision (`verification/wave4.md`, W4-9).
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
- Python: `pyvbmc/vbmc/variational_optimization.py:1443-1447`
  (`compute_var == 2` raises `NotImplementedError`, "Missing port:
  compute_var == 2 skipped since it is not used"), and the
  gradient-of-variance path raises before it at `:1439-1442`.
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
- Python: `pyvbmc/vbmc/variational_optimization.py:1219-1226` keeps the
  `onlyweights_flag` test commented out ("Not currently used, since it is
  only a speed optimization") and `:1228` records "Missing port: block below
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
- Python: `pyvbmc/vbmc/variational_optimization.py:788` ("Missing port:
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
- Python: no counterpart. Three option names of the feature survive in
  `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini` and are read
  nowhere: `variational_sampler = "malasample"` (`:143`),
  `active_variational_samples = 0` (`:273`) and `scale_lower_bound = True`
  (`:275`). All three are in the `INERT_OPTIONS` set of
  `pyvbmc/vbmc/options.py` (see the P1b entry on declared options that
  nothing reads).
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
- Python: `pyvbmc/vbmc/variational_optimization.py:863` (`_vb_init`), the
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

### The sieve asks for candidates in proportion to the current `K`
- Python: `pyvbmc/vbmc/vbmc.py:1630-1632`, in the main loop:
  `N_fastopts = math.ceil(self.options.eval("ns_elbo", {"K": self.vp.K}))`,
  with `ns_elbo = lambda K : 50 * K`
  (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:65`).
- MATLAB: `vbmc.m:699`,
  `Nfastopts = ceil(evaloption_vbmc(options.NSelbo,K))`, where `K` is the
  local variable set once at `vbmc.m:459` to `options.Kwarmup` (default 2,
  `vbmc.m:248`) and never reassigned, so the main loop always asks for
  `50*2 = 100` fast candidates, or 10 on an incremental iteration.
  `vbmc.m:584` (the warp branch) and `misc/finalboost_vbmc.m:33` use `Knew`.
- What differs: PyVBMC's sieve grows its candidate set with the number of
  mixture components, MATLAB's main loop does not. The two agree while
  `vp.K == 2`, that is throughout warm-up, and part once `update_K` grows
  `K`: at `vp.K = 10`, 500 candidates against 100.
- Why: the option is declared as a function of `K` on both sides and PyVBMC
  passes the current `K`, which is what the declaration describes;
  `dev/experiments/port_review_20260919/verification/wave1_P6.md` (row
  "cmp F3") establishes that `vbmc.m:459` is the only assignment to MATLAB's
  `K`. PyVBMC's warp branch (`pyvbmc/vbmc/vbmc.py:1478-1480`) and final
  boost (`:2715`) evaluate the option at the new number of components, as
  `vbmc.m:584` and `misc/finalboost_vbmc.m:33` do; until commit `d7c7887`
  (2026-09-20) the warp branch evaluated it at `vp.K`, which differs from
  the new number only with `variable_means=False`
  (`verification/wave2.md`, row W2-16).
- Kind: deliberate change.

### GP smoothing (`Bandwidth`, `vp.delta`) is not ported
- Python: no counterpart. There is no `bandwidth` option in
  `pyvbmc/vbmc/option_configs/`, and `VariationalPosterior` has no `delta`.
- MATLAB: `defopts.Bandwidth = 0` (`vbmc.m:313`);
  `misc/setupvars_vbmc.m:247` sets
  `optimState.delta = options.Bandwidth*(PUB-PLB)` and `:97` initializes
  `vp.delta = []`; `misc/vpsieve_vbmc.m:16` copies the vector onto the
  candidate posterior; `misc/gplogjoint.m:86-89` and `:164` widen the
  integration scale `tau_k` by `delta`, and `acq/acqwrapper_vbmc.m:12-14`
  switches the GP prediction to `gplite_quad` with `vp.delta` when any of its
  entries is positive.
- What differs: MATLAB can smooth the GP surrogate with a fixed kernel width,
  in units of the plausible box, before the expected log joint and the
  acquisition are computed. PyVBMC always uses the unsmoothed quantities.
- Why: at the MATLAB default `Bandwidth = 0` the vector is all zeros and
  every branch above reduces to the unsmoothed formula, so the two agree at
  default options;
  `dev/experiments/port_review_20260919/verification/wave1_P6.md` (row
  "cmp F5").
- Kind: unported feature (inactive at MATLAB defaults).

### The full ELCBO evaluation uses the exact entropy of a one-component posterior
- Python: `pyvbmc/vbmc/variational_optimization.py:496-499`, inside
  `_eval_full_elcbo` (`:455`): the number of entropy samples is 0 when
  `vp.K == 1`, which sends `_neg_elcbo` to the deterministic branch
  (`pyvbmc/entropy/entlb_vbmc.py`), exact for a single Gaussian; for more
  components it is `ceil(ns_ent_fine(K)/K)` per component.
- MATLAB: `misc/vpoptimize_vbmc.m:279` computes
  `NSentFineK = ceil(evaloption_vbmc(options.NSentFine,K)/K)` with no test on
  `K`, and `:288-289` passes it to `negelcbo_vbmc`, which takes the Monte
  Carlo estimator whenever the sample count is positive
  (`misc/negelcbo_vbmc.m:104-110`).
- What differs: for a one-component posterior, which pruning can produce,
  PyVBMC's reported ELBO, the ranking of the finished optimizations and the
  pruning decisions rest on the closed-form Gaussian entropy instead of a
  4096-sample estimate whose standard deviation is a few hundredths and which
  the reported ELBO SD does not account for. For more components the
  deterministic expression is only a lower bound and both sides estimate.
- Why: commit `7331841` (2026-09-19) "fix(variational_optimization): report
  the exact entropy of a one-component posterior"; the sampling error it
  removes is measured in
  `dev/experiments/port_review_20260919/verification/wave1_P6.md` §(b).
- Kind: deliberate change.

### The deterministic-entropy optimization is SciPy's BFGS
- Python: `pyvbmc/vbmc/variational_optimization.py:233-247`:
  `scipy.optimize.minimize` with analytic gradients and
  `tol=options["det_entropy_tol_opt"]`, no cap on the number of evaluations,
  and the returned iterate kept, with the optimizer's own message logged as a
  warning, when SciPy reports failure.
- MATLAB: `misc/vpoptimize_vbmc.m:73-101`: `fminunc` with
  `TolFun = options.DetEntTolOpt` (`:76`) and
  `MaxFunEvals = 50*(vp.D+2)` (`:77`), inside a `try`/`catch` (`:79-82`) that
  falls back to `cmaes_modded` on a thrown error.
- What differs: SciPy hands `tol` to BFGS as `gtol`, a gradient-norm
  tolerance, where MATLAB's `TolFun` is a function-value tolerance, so the
  same option number means different things; PyVBMC leaves the evaluation
  count to BFGS's own default instead of capping it; and PyVBMC has no
  CMA-ES fallback. An ordinary non-success outcome, such as a loss of
  precision on a near-flat surface or the iteration limit, keeps the best
  point reached and warns, which is what MATLAB does with `fminunc`'s
  iterate, since its `catch` fires only on a thrown error. The branch runs
  when the entropy is evaluated deterministically: `vp.K == 1`, or
  `entropy_switch` (non-default).
- Why: commit `04d15ff` (2026-09-19) "fix(variational_optimization): keep the
  iterate the deterministic optimizer returns", which replaced a
  `RuntimeError` that ended the whole run;
  `dev/experiments/port_review_20260919/verification/wave1_P6.md` (row
  "int F7 + cmp F4") establishes the `tol`-to-`gtol` mapping and the MATLAB
  cap.
- Kind: substituted library.

---

## Slice P7 — variational posterior, entropies, VP statistics

### Posterior tempering (`vbmc_power`, `vptrain2real`) is not ported
- Python: no counterpart. `pyvbmc/vbmc/vbmc.py:1501` and `:1662` keep
  `# vp_real = vp.vptrain2real(0, self.options)` commented out with
  `vp_real = self.vp`. The option `temperature = 1` is declared
  (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:257`), registered
  in `INERT_OPTIONS` and read nowhere: `pyvbmc/whitening/whitening.py:275`
  and `:372` read `optim_state["temperature"]`, a key nothing writes, and so
  always take `T = 1`.
- MATLAB: `vbmc_power.m`, `misc/vptrain2real.m` (the branch
  `any(T == [2,3,4,5])`).
- What differs: MATLAB can run the algorithm on a tempered posterior and
  convert the training posterior back to the untempered one, recomputing the
  ELBO, its SD and the entropy; it validates `Temperature`
  (`misc/setupvars_vbmc.m:249-255`) and reads it in six files. PyVBMC has
  neither the tempering nor the conversion: setting the option has no
  effect, and PyVBMC warns that it has none. MATLAB's `vptrain2real` returns
  its input unchanged at `T = 1`, the only temperature a PyVBMC run has, so
  the commented-out line is behaviorally equivalent at every reachable
  setting. The same holds for the function logger: `misc/funlogger_vbmc.m`
  divides the recorded value, its SD and the log-Jacobian by the temperature
  (`:132-135`, `:180-183`, `:243`, `:269`), and `FunctionLogger._record`
  divides nothing, while `warp_input` does divide the log-Jacobian
  (`pyvbmc/whitening/whitening.py`, as `misc/warp_input_vbmc.m:117`); a port
  of tempering has the logger to complete (`verification/wave3.md`, row
  W3-30).
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
- Python: `pyvbmc/variational_posterior/variational_posterior.py:869-872`
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

### The soft bounds are the box of the training inputs of the call
- Python: `pyvbmc/variational_posterior/variational_posterior.py:406-412`, in
  `get_bounds` (`:362`): `mu_lb` and `mu_ub` are the componentwise minimum
  and maximum of the training inputs passed in, and the log-scale bounds are
  the log of that box's width, down to a factor `tol_length`. The result is
  stored on the posterior, replacing any box of an earlier call.
- MATLAB: `misc/vpbounds.m:9-24` fills the box with reversed infinite
  sentinels only when the posterior carries none, then widens the stored box
  with the current training inputs. The box travels on the candidate
  posteriors that `misc/vbinit_vbmc.m:39` copies and on the one
  `misc/vpoptimize_vbmc.m:188` returns, so it accumulates from iteration to
  iteration until `misc/setupvars_vbmc.m:98` resets it.
- What differs: PyVBMC's box follows the training set of the call; MATLAB's
  never shrinks. The two agree while the training set only grows in fixed
  coordinates, and part after the post-warm-up trim of the training set or
  after a warp changes the coordinates, where MATLAB keeps the widest box any
  iteration produced.
- Why: commit `7dd2e3f` (2026-09-19) "fix(variational_posterior): compute the
  soft bounds from the training inputs of the call"; the box is what the soft
  bounds are meant to follow when the training set is trimmed or the input
  space is warped. Kept as a deliberate change after a trace of two short
  seeded runs
  (`dev/experiments/port_review_20260919/verification/scripts/soft_bounds_trace.py`):
  the accumulated box reaches 2.2 times the width of the rebuilt one on a
  two-dimensional Rosenbrock run and 12.1 times on a three-dimensional
  two-mode target, while the fitted posterior stays within a small margin
  of the rebuilt box (a component mean at most 0.067 box widths outside
  it, one log scale 0.008 above its upper bound), so the rebuilt bounds
  bind marginally at most and the wider box would have changed little.
- Kind: deliberate change.

### `kl_div_mvn` takes its inputs directly, and the log determinants
- Python: `pyvbmc/stats/kl_div_mvn.py` (a module-level function with no
  decorator; all four inputs normalized locally; the log determinants of
  the two covariances from `np.linalg.slogdet`, `:41-48`, and both
  divergences infinite for a matrix that is singular or has a negative
  determinant).
- MATLAB: `shared/mvnkl.m` (`lndet = log(det(Sigma2) / det(Sigma1))`).
- What differs: the function previously carried the `handle_0D_1D_input`
  decorator, which is written for methods and swallowed `mu1` as `self`.
  Scalar, 1-D, 2-D, keyword and mixed calls all agree with the previous 2-D
  results. The formula is MATLAB's but for the log-determinant term, which
  MATLAB takes from the raw determinants: those leave the range of a double
  for a well-conditioned covariance at moderate dimension (zero below an SD
  of about 8e-9 per coordinate at `D = 20`, infinite above about 5e7), where
  MATLAB returns `Inf` or NaN (`matlab_side_defects.md`, entry 37) and
  PyVBMC the divergence. At ordinary scales the two agree to rounding.
- Why: `dev/plans/latent-bug-fixes.md` §"Supporting fixes and closed
  entries" ("Remove the method-only decorator and normalize all four inputs
  locally ... Keep the KL formula unchanged") and Phase 1 item 3;
  the original defect is in `dev/2026-09-02-modernization-discussion.md` §9.
  The log determinants:
  `dev/experiments/port_review_20260919/verification/wave5.md`, W5-2.
- Kind: deliberate change.

### `vp.stats["J_sjk"]` is pruned on both component axes
- Python: `pyvbmc/vbmc/variational_optimization.py:394-395` (`np.delete` on
  `axis=1` and then `axis=2`, after `I_sk` on `axis=1` at `:393`), so the
  array keeps the shape `(Ns,K,K)` and its correspondence with `I_sk`.
- MATLAB: `misc/vpoptimize_vbmc.m:236-237` deletes the pruned component from
  one axis of each quantity: `I_sk(:,idx) = []` from the component axis of
  the `(Ns,K)` array, and `J_sjk(:,:,idx) = []` from the last axis alone of
  the `(Ns,K,K)` array declared at `:185`, which leaves `J_sjk` with shape
  `(Ns,K,K-1)`.
- What differs: MATLAB's pruned `J_sjk` is rectangular and indexed
  inconsistently with its own `I_sk`; PyVBMC's stays square. Nothing in
  PyVBMC reads it, but S-VBMC's constructor filters posteriors using its
  maximum.
- Why: `dev/plans/latent-bug-fixes.md` §"Supporting fixes and closed entries"
  ("prune both component axes, keep shape `(Ns,K,K)` and correspondence with
  `I_sk`") and Phase 1 item 3; the MATLAB lines are read in
  `dev/experiments/port_review_20260919/verification/wave1_P6.md` (row
  "cmp sheet note 1").
- Kind: deliberate change.

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
- What differs: nothing numerically. `qtrapz(y)` is
  `sum(y) - 0.5*(y(1) + y(end))`, the trapezoid rule at unit spacing, which
  is algebraically `scipy.integrate.trapezoid(y)`; both `mtv` and
  `vbmc_mtv.m` multiply by the grid spacing afterwards. The two agree bit
  for bit on short vectors and to about 1e-14 on vectors of thousands of
  points, by the order of the sum alone
  (`dev/experiments/port_review_20260919/verification/wave5_P7.md`, row
  P7-10).
- Why: mechanical: `qtrapz.m`'s only caller is `vbmc_mtv.m`, whose
  counterpart is `mtv`. Recorded here so the absence of a `qtrapz` module is
  not reported as a missing port.
- Kind: substituted library.

### The mode search starts from draws of the posterior
- Python: `pyvbmc/variational_posterior/variational_posterior.py: mode`
  (`:1312`). It runs `n_opts = ceil(sqrt(K))` optimizations, each started at
  the point of highest density among 1e5 draws of the posterior, the
  component means joining the candidates of the first; the draws come from a
  copy of the posterior that holds a copy of its generator (`:1383-1387`), so
  a call leaves the stream of `vp.rng`, which a run shares, where it found
  it. The result is stored on the posterior whenever the mode is asked for
  in the original space; `mode()` without `n_opts` returns a stored mode, a
  call that gives `n_opts` runs the search, and `set_parameters` and
  `get_parameters` clear the store.
- MATLAB: `vbmc_mode.m:21-47` starts one bounded optimization at each
  component mean, keeping the `nmax = 20` best by density when `K > 20`, and
  draws nothing; it stores `vp.mode` only when the second output is asked
  for, and `misc/rescale_params.m:39-40` removes it.
- What differs: the starting points and their number (8 optimizations for
  `K = 50` against MATLAB's 20; PyVBMC chooses its starts globally, MATLAB
  polishes more basins), the optimizer (`scipy.optimize.minimize`, L-BFGS-B
  within the bounds, for `fmincon` and `fminunc`), and the unconditional
  store. The box of the search is MATLAB's, the original bounds moved in by
  an absolute `sqrt(eps)`, and the starting point is clamped to it on both
  sides.
- Why: `ba8116fa` (2022-11-03, pull request 115, "fix vp.mode() function")
  replaced a first port of 2021 that looped over the wrong axis;
  `pyvbmc/variational_posterior/README.md` records the bug it answered. The
  mode search was repaired and not ported again, and the stored MATLAB mode
  of the `K = 50` fixture is reproduced to 1e-4. The copy of the generator
  is the PI's ruling of 2026-09-21
  (`dev/experiments/port_review_20260919/verification/wave5.md`, W5-14);
  until then a call of `vp.mode()` moved every later draw of a run.
- Kind: deliberate change.

### `vp.pdf` gives a point on or outside the original bounds a density of zero
- Python: `pyvbmc/variational_posterior/variational_posterior.py:879-886`,
  `:1002-1004`: a row that is not strictly inside `lb_orig` and `ub_orig` gets
  the density 0 (the log density `-inf`) and is neither transformed nor
  corrected by the Jacobian; a row with a NaN coordinate takes the same
  branch. Inside the bounds the density is the transformed-space density
  divided by the Jacobian, and where that quotient is not finite because
  the Jacobian underflows, the exponential of the difference of the logs
  (`:1015-1028`).
- MATLAB: `vbmc_pdf.m:36-39` warps every row, and `:113-123` divides by
  `warpvars_vbmc(X,'prob',...)`.
- What differs: on a bound MATLAB's transformed coordinate is `±Inf` and the
  Jacobian correction gives NaN, and outside a bound the logit of a negative
  number is complex; PyVBMC returns the density of a bounded variable
  outside its support, zero. PyVBMC also keeps a density near the top of
  the range of a double that the quotient would report as infinite.
- Why: the mask is in the Python from the first port of `pdf` (2021-03) and
  is pinned by `test_pdf_outside_bounds`; the log form where the quotient
  fails is `dev/experiments/port_review_20260919/verification/wave5.md`,
  W5-4. The same method makes its working copy of the points float64, so an
  integer input is evaluated where it was given (W5-18), as a MATLAB double
  is.
- Kind: deliberate change (an improvement on MATLAB).

### `kde_1d` is an independent implementation of Botev's estimator
- Python: `pyvbmc/stats/kde_1d.py`, whose one consumer in the package is
  `VariationalPosterior.mtv`.
- MATLAB: `shared/kde1d.m`, Botev's original.
- What differs: `kde_1d` is no transcription. It credits a sample to the
  nearest grid point (`_linear_binning`) where `histc` credits the grid
  point at or below it, which puts MATLAB's estimate exactly half a grid
  step low (`matlab_side_defects.md`, entry 31); when the fixed-point solve
  for the bandwidth fails it falls back on Scott's rule (`:248-249`) where
  MATLAB has `fminbnd` of `|f|` over `[0, 0.1]`; and it floors negative
  round-off at 0 (`:261`) where MATLAB has `eps`. Both return a grid sum
  that, times the spacing, is `n/(n-1)` and not 1. `mtv` normalizes the
  density before it integrates, as `vbmc_mtv.m:68` and `:71` do, so the
  normalization cancels and only the half step and the bandwidth (7e-4
  relative at `n = 2^10`) remain: at `nkde = 2^13` the two normalized
  estimates are 2e-4 apart in total variation, against 0.02 between either
  and the true density.
- Why: `522a901b` (2021-05-26, "add and integrate better kde1d function")
  replaced `scipy.stats.gaussian_kde` with an implementation of the same
  algorithm; its docstring credits Botev's MATLAB code, D. B. Smith's Python
  and KDEpy. Measured against a line-by-line transcription of `kde1d.m` in
  `dev/experiments/port_review_20260919/verification/wave5_P7.md`, rows
  P7-6a to P7-6d.
- Kind: substituted library.

### The interface of `vp.sample`
- Python: `pyvbmc/variational_posterior/variational_posterior.py: sample`
  (`:637`).
- MATLAB: `vbmc_rnd.m`.
- What differs: the law of the draws is MATLAB's, the balanced draw with
  its remainder included. The interface differs in four places. The second
  output is an `(N,)` array of 0-based integer indices, MATLAB's an `N`-by-1
  column of 1-based ones. A whole number of samples given as a float is
  taken and a fractional one refused. A finite negative `df`, which `pdf`
  reads as the product of univariate `t` densities on both sides, is refused
  with a message, where MATLAB's `gamrnd` of a negative shape returns NaN.
  And weights that do not sum to one raise (`rng.choice`), where MATLAB's
  `catrnd` draws from the normalized weights; `set_parameters` normalizes
  the weights, so no run meets the case.
- Why: `dev/experiments/port_review_20260919/verification/wave5.md`, W5-15
  and W5-16, and `verification/wave5_P7.md`, the check of the second
  first-question answer.
- Kind: deliberate change (interface; the draws are MATLAB's).

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

### `warp_input` maps the search state back with the current transform
- Python: `pyvbmc/whitening/whitening.py:143` (`warp_input`): the search
  bounds and the cached search points, which live in the current inference
  space, are mapped to the original space with the function logger's
  transformer (`:195`, `:296-297`) and from there into the warped space. The
  transformer of the posterior handed in serves only to express that
  posterior's own covariance.
- MATLAB: `misc/warp_input_vbmc.m:8`, `:133` take the old transform from the
  posterior handed in, which `vbmc.m:546-557` takes from `best_vbmc`.
- What differs: a posterior recorded before an earlier warp carries an older
  transform, and MATLAB then maps the search state through the wrong one (a
  search box 8.5 to 12.5 times wider per coordinate in the reproduction).
  Latent: on 21 stored runs the selection never returned such a posterior.
  PyVBMC shared the defect until commit `7d93f60` (2026-09-20).
- Why: `verification/wave2.md`, row W2-6; entry 3 of
  `matlab_side_defects.md`.
- Kind: deliberate change.

### The default bounded transform is probit, not logit
- Python: `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:311`
  (`bounded_transform = "probit"`), handed to `ParameterTransformer` at
  `pyvbmc/vbmc/vbmc.py:557`;
  `pyvbmc/parameter_transformer/parameter_transformer.py:141-146` maps
  `"probit"` and `"norminv"` to one transform type and `"logit"` and
  `"student4"` to their own.
- MATLAB: `defopts.BoundedTransform = 'logit'` (`vbmc.m:344`);
  `misc/setupvars_vbmc.m:34-42` accepts `logit`, `norminv`/`probit` and
  `student4`, and rejects anything else.
- What differs: at default options a variable with both bounds is carried to
  the unbounded space by the inverse normal CDF in PyVBMC and by the logit in
  MATLAB, so the transformed coordinates, the GP training inputs and the
  variational posterior's parameters of a bounded problem are not MATLAB's.
  Both transforms are implemented on both sides and either can be selected by
  the option.
- Why: `AGENTS.md` §"Two coordinate spaces" ("probit by default").
- Kind: deliberate change.

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
- Python: `pyvbmc/parameter_transformer/parameter_transformer.py:73-80`
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
- Python: `pyvbmc/function_logger/function_logger.py:590` (trims `X_orig`,
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

### The warp keeps a covariance that its correlation threshold would leave
    indefinite
- Python: `pyvbmc/whitening/whitening.py:226` (`warp_input`, with
  `_drop_low_correlations` and `_is_positive_definite`): the covariance with
  its weak correlations set to zero is used when it is positive definite, and
  the covariance as it was otherwise.
- MATLAB: `misc/warp_input_vbmc.m:52-71` takes the SVD of the thresholded
  matrix in every case, as `papers/acerbi2020variational_appendix.md` §B.2
  prescribes.
- What differs: from three dimensions on a covariance with entries zeroed
  need not be positive semi-definite; the SVD then returns the absolute value
  of a negative eigenvalue, and the transform misses unit variance along that
  direction (0.097 for 1 in the example of `verification/wave3.md`, row
  W3-6). None of the 990 final posteriors of the population campaigns is
  affected, so the guard changes no run of that population.
- Why: the PI's ruling on W3-6 (2026-09-20).
- Kind: deliberate change.

### A variable with one finite bound is refused
- Python: `pyvbmc/parameter_transformer/parameter_transformer.py:128`: the
  constructor raises for a variable bounded on one side only. `VBMC` rejects
  such bounds itself (`pyvbmc/vbmc/_bounds.py`, `vbmc:HalfBounds`).
- MATLAB: `shared/warpvars_vbmc.m:897-901` assigns types 1 and 2 to such
  variables, with the log transforms of `:92-101`, `:302-312` and `:491-494`;
  `misc/boundscheck_vbmc.m:138-143` keeps them from reaching VBMC.
- What differs: the log transforms were never ported (commit `6a247e5`,
  2021-02-25, removed their type labels, under which the code already applied
  the identity). Until 2026-09-20 the public class carried such a variable
  through the identity, with an inverse outside the support.
- Why: the PI's ruling on W3-23.
- Kind: unported feature.

### Points within rounding of a bound: the nudge, the clamp, and the grouping
    of one product
- Python: `pyvbmc/parameter_transformer/parameter_transformer.py:581`
  (`_to_unit_interval`), `:593` (`_from_unit_interval`), `:646`
  (`_inverse_student4`).
- MATLAB: `shared/warpvars_vbmc.m:106-107`, `:256-257`, `:265-266` (direct),
  `:457-459` (clamp of the inverse), `:451` (inverse of type 13).
- What differs: the direct transform moves a unit-interval image that rounds
  to 0 or 1 to the adjacent number, where MATLAB returns an infinity; a run
  reaches this when a warp re-transforms a stored point that the inverse had
  clamped beside a bound. The clamp of the inverse uses `nextafter` where
  MATLAB uses `eps(bound)`, one ulp apart at a bound that is a power of two.
  The inverse of `student4` computes `(3/8)*(u/sqrt(...))` where MATLAB
  computes `((3/8)*u)/sqrt(...)`, one ulp. Everything else in the three
  bounded transforms, their inverses and their log-Jacobians is bit-identical
  to a transcription of `warpvars_vbmc.m` on 20 000 points per configuration
  (`verification/wave3_P8.md`, rows P8-15a to P8-15c).
- Why: pull request 89 (2022-08-28): "points strictly within the bounds in
  the original space should never map to points at infinity in the
  transformed space". The message of its commit `3e9d1c2`, "identically to
  MATLAB", is wrong for both of its changes: before it the clamp was
  MATLAB's.
- Kind: deliberate change (the nudge); the other two are differences of one
  ulp, kept.

### The pooling of a repeated noisy observation has a fallback for extreme SDs
- Python: `pyvbmc/function_logger/function_logger.py:720` (`_record`).
- MATLAB: `misc/funlogger_vbmc.m:234-238`.
- What differs: when the precisions of the inverse-variance pooling overflow
  (an SD below about 1e-154), PyVBMC pools with relative weights, where
  MATLAB's three lines give Inf or NaN. The ordinary path is MATLAB's
  arithmetic, bit for bit.
- Why: commit `6769a9a` (2026-09-16), recorded in
  `dev/plans/pymc-target-adapter.md`.
- Kind: Python-only addition.

### A cached value of a target that provides its noise needs its SD
- Python: `pyvbmc/vbmc/vbmc.py:1021` refuses an `f_vals` that holds a value
  together with `specify_target_noise` at construction (NaN entries mark
  points still to be evaluated and supply nothing), and
  `pyvbmc/function_logger/function_logger.py:552` (`add`) raises without an
  SD at uncertainty level 2, as `:415` (`batch_call`) does for a value in
  its `f_vals`, before the target is called; level 1 records SD 1.
- MATLAB: `misc/funlogger_vbmc.m:159-162` intends SD 1 for a missing SD at
  every noisy level, and raises before reaching that line, because both of
  its callers pass the value alone (`matlab_side_defects.md`).
- What differs: `f_vals` has no channel for an SD; the
  `precomputed_evaluations` argument has one. Until 2026-09-20 PyVBMC
  recorded SD 1 for such values.
- Why: the PI's ruling on W3-28.
- Kind: deliberate change.

### The evaluation time of a repeated point leaves out the times that are unknown
- Python: `pyvbmc/function_logger/function_logger.py:752` (`_record`, the
  branch of a repeated point): an unknown time, NaN, which is the default of
  `add`, leaves the stored average alone, and a known time takes the place
  of an average that is unknown.
- MATLAB: `misc/funlogger_vbmc.m:187` (`'add'` hands `record` a time of 0),
  `:245` (the running average, taken in every case).
- What differs: MATLAB records 0 for a value it did not time, so the average
  of a point is always a number and an added value pulls it towards 0.
  PyVBMC records NaN for such a value and keeps it out of the average.
  Nothing reads the times of single points on either side (the entry on
  `gp.t`, slice P5).
- Why: the PI's ruling on W3-21, whose fix covered the unknown time of a
  repeat; the independent check of the wave-3 pass found the other
  direction, the unknown time of the first evaluation
  (`verification/wave3.md`).
- Kind: deliberate change.

### `scale`, and the logger's noise flag, are validated
- Python: `pyvbmc/parameter_transformer/parameter_transformer.py:82-91`
  (`scale` finite, positive, one entry per dimension);
  `pyvbmc/function_logger/function_logger.py:199-204` (`noise_flag` true
  exactly above uncertainty level 0).
- MATLAB: `shared/warpvars_vbmc.m:763-765` adds `log(scale)` unchecked, and
  gives a user no way to supply a scale; `misc/funlogger_vbmc.m` derives the
  noise flag from the state.
- What differs: two public constructors refuse inputs that gave a NaN
  log-Jacobian or a singular map, and a logger whose two routes disagreed.
- Why: the PI's rulings on W3-26 and W3-32.
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
  inferred by drawing a sample. The support of a frozen univariate
  distribution is read from its `support()`, which applies `loc` and `scale`
  (`pyvbmc/priors/scipy.py:178-183`, in `_support_box`); a prior reads its
  input as float64, as a
  MATLAB double is (`pyvbmc/priors/prior.py:58`); and a `Product` applies a
  `UserFunction` marginal to one point at a time, as the contract of that
  class states (`pyvbmc/priors/product.py:90-104`).
- Why: `dev/plans/latent-bug-fixes.md` §"Supporting fixes and closed entries"
  (SciPy private imports; the `rvs(1)` dimension probe) and Phase 1 item 8;
  decision D5; `AGENTS.md` §"Vectorized targets are opt-in" ("Separate priors
  retain their scalar interface").
- Kind: Python-only addition (with the MATLAB density functions ported
  inside it).

### The prior classes check their arguments, and `VBMC` checks a prior against the hard bounds
- Python: `pyvbmc/priors/prior.py: _check_finite`, called first by the
  constructors of `UniformBox`, `Trapezoidal`, `SplineTrapezoidal` and
  `SmoothBox`, refuses a bound, a pivot or a scale that is NaN or infinite;
  the two trapezoid classes refuse `a >= u`, `u >= v` and `v >= b`
  (`trapezoidal.py:76-81`, `spline_trapezoidal.py:78-83`); `tile_inputs`
  refuses an array that is not a row, a column or a flat array of the
  dimension given;
  `pyvbmc/priors/scipy.py: _check_univariate_parameters` refuses a frozen
  univariate distribution with array-valued parameters; and
  `pyvbmc/vbmc/vbmc.py: _check_prior_covers_bounds`, called from
  `_init_log_joint`, refuses a prior whose `support()` box does not contain
  the box of the hard bounds (a prior without finite support, a
  `UserFunction` or a `log_prior=` callable among them, contains every box).
  The comparison allows a gap of `1e-9` times the range of a coordinate
  whose hard bounds are both finite, and none otherwise: a support
  computed as `loc + scale` from the hard bounds themselves ends a few
  units in the last place inside them
  (`uniform(loc=-20, scale=20.2).support()` ends at 0.1999999999999993).
  `SciPy` and `Product` read their support from the distribution at every
  call (`_support_box`), `a` and `b` being read-only properties. The
  checks of the classes run once, at construction; the check against the
  hard bounds runs wherever the log joint is built, a `load` that
  rebuilds it included.
- MATLAB: `munifboxlogpdf.m:44` and `msmoothboxlogpdf.m:26`, `:46` check
  `a < b` and `sigma > 0` at every call, by comparisons that are false at
  NaN; `mtrapezlogpdf.m` and `msplinetrapezlogpdf.m` check the sizes and no
  order; MATLAB has no prior object to hold against the bounds, and a target
  that returns `-Inf` inside them stops in `misc/funlogger_vbmc.m:121`.
- What differs: PyVBMC is the stricter side throughout. MATLAB takes NaN and
  infinite arguments, whose densities are not densities and whose rejection
  sampler turns uniform at a NaN pivot (`matlab_side_defects.md`, entry 39).
  MATLAB computes two limits of the trapezoids that PyVBMC refuses, `u == v`
  (the tent prior) and `v == b`, correctly and normalized, and returns NaN
  for the trapezoid's `u == a` (entry 25), which PyVBMC refuses as well; the
  PI kept the refusal of all three (2026-09-21), the usual call
  `Trapezoidal(lb, plb, pub, ub)` being unable to produce `u == v` since
  `VBMC` requires `plb < pub`. A prior narrower than the hard bounds stops a
  MATLAB run, and stopped a PyVBMC run until 2026-09-21, at the first
  evaluation outside the support, with a message that names neither the
  prior nor the bounds.
- Why: the order check is documented in the class docstrings since
  `2da98b5f` (2023-02-08), with no reason recorded for the legal limits; the
  others are rulings of 2026-09-21
  (`dev/experiments/port_review_20260919/verification/wave5.md`, W5-5,
  W5-33, W5-38, W5-39, W5-40).
- Kind: deliberate change (stricter interface).

### A NaN coordinate has density zero in every box prior
- Python: `pyvbmc/priors/uniform_box.py:83-87`: a row is in the support
  when every coordinate lies between the bounds, the bounds included, so a
  row with a NaN coordinate is out of it, as in the three other box
  families, whose membership tests are positive.
- MATLAB: `shared/munifboxlogpdf.m:51` marks the rows out of the support by
  `x < a | x > b`, false at NaN, and gives such a row the full density; the
  other three log densities return `-Inf`.
- What differs: the density of `UniformBox` at a point with a NaN
  coordinate, zero for MATLAB's plateau (`matlab_side_defects.md`, entry
  38). Inside a run the function logger refuses such a point on the value of
  the likelihood. The support is closed on both sides: the density at `a`
  and at `b` is the plateau.
- Why: PI ruling of 2026-09-21, a departure from MATLAB
  (`dev/experiments/port_review_20260919/verification/wave5.md`, W5-32).
- Kind: deliberate change.

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

- **The remainder of the balanced draw matches MATLAB.**
  `VariationalPosterior.sample` corrected the weights of the remainder of a
  balanced draw with the builtin `sum` of a `(1, K)` array, which returns
  its row, so that the remainder was drawn from other probabilities than
  `vbmc_rnd.m:67-71` has; the error entered with the first version of the
  method (`42a0deef`, 2021-03-14) and was corrected on 2026-09-21
  (`dev/experiments/port_review_20260919/verification/wave5.md`, W5-1). The
  same pass brought four more lines to MATLAB's: the mean per coordinate in
  `kl_div(samples=...)` (W5-9), the guards of `kl_div(gauss_flag=False)`
  against densities that are zero or not finite (W5-4), the rounding of a
  half away from zero at the twelve sites where MATLAB has `round` (W5-6,
  through `pyvbmc/stats/_rounding.py`), and the float64 accumulator of the
  box priors (W5-31).
- **The two entropies differ in the weight gradient of one component, in
  MATLAB alike.** Without the Jacobian, `entlb_vbmc` returns 0 and
  `entmc_vbmc` returns `H - 1` for `K = 1` (`ent/entlb_vbmc.m:45-47`,
  `ent/entmc_vbmc.m:96-101`); every production caller sets the Jacobian
  flag, under which both are zero. The weight gradient of the Monte Carlo
  entropy keeps a term that the other gradients drop, rightly: it estimates
  the derivative of the mixture's total mass, one for a weight and zero for
  the rest. MATLAB's fix of that gradient (`1b72896`, 2021-06-23) is in the
  Python, which agrees with a transcription of the fixed block to 1.8e-15
  (`verification/wave5.md`, W5-22). Both entropies return NaN for a weight
  of exactly zero with separated components, on both sides (W5-3).
- **Lines of `active_sample.py` that read as defects are MATLAB's own.**
  `recompute_var_post` is saved and restored around a block that never
  writes it, as in MATLAB, whose own assignment is commented out
  (`private/activesample_vbmc.m:55-58`, `:516`); the Gaussian fitted to a
  high-posterior-density subset takes `cov(X_hpd,1)` and its empty-subset
  fallback `cov(X)` (`:593`, `:596`); the fallback box of the box draws has
  the literal 3 (`:618-619`) and is unreachable, the search bounds being
  always finite; the candidates are clipped into the search box before the
  integer coordinates are snapped (`:637`, `:219`), so a snapped candidate
  can lie outside the box; and the comment that asks for other checks of
  the search bounds for integer variables is MATLAB's (`:495`). The initial
  design is not snapped to the integer grid on either side
  (`misc/initdesign_vbmc.m`), and neither side keeps an input that is in
  the training set out of the candidates (`verification/wave5.md`, W5-26,
  W5-27, W5-29).
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
- **The across-sample spread of `predict` and `quad` uses the `N-1`
  divisor in MATLAB too.** `gpyreg/gaussian_process.py` divides the sum of
  squared deviations of the per-sample means by `s_N - 1` in both methods,
  as `gplite/gplite_pred.m:158`, `:160` and `gplite/gplite_quad.m:115` do;
  `pyvbmc/vbmc/vbmc.py` reads the resulting variance to form `lcb_max`, so
  the convention reaches a default run and matches MATLAB
  (`verification/wave6.md`, W6-36). gpyreg's
  `test_quadrature_with_noise_matches_numerical_integration` pins it.
- **The quiet omissions of the noise function are MATLAB's.** `s2_star` is
  ignored where the noise declares no user-provided feature
  (`gplite/gplite_noisefun.m:186-194`), the output-dependent term is dropped
  when no target is given (`:198`), `random_function` and `gplite_rnd.m:67`
  evaluate the noise at the test points with neither target nor variance,
  and `gpyreg/gaussian_process.py: get_recommended_bounds` and
  `gplite/gplite_train.m:142` repair an inverted pair of recommended bounds
  alike, with `UB = max(LB, UB)` (W6-36). A training set of one point is a
  difference, entry 49 of `matlab_side_defects.md`, and equal targets are
  the entry "A training set of equal targets takes a range of one". `f_min_fill` with no more evaluations than provided
  starting points evaluates and returns the first `N` of them, as
  `gplite/private/fminfill.m:98-114` does since `46b6f5e` (W6-25).
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
- **`minimize_adam`'s point and value tables are offset in MATLAB too.**
  `pyvbmc/vbmc/minimize_adam.py:87` records `y_tab[i]` at the point the
  iteration starts from and `:101` records `x_tab[:, i]` after the update, so
  `y_tab[i]` is the objective at `x_tab[:, i-1]`, and the trailing averages
  at `:141-142` are shifted against each other. `utils/fminadam.m` has the
  same order: `[ftab(iter),grad] = fun(x)` at `:48`, the update at `:59-60`,
  `xtab(:,iter) = x` at `:63`, and the two averages at `:96-97`. The pairing
  that selects the returned parameters
  (`pyvbmc/vbmc/variational_optimization.py:297-300`,
  `misc/vpoptimize_vbmc.m:133-134`) is the same on both sides.
- **`var_ss` adds a standard deviation to a variance in MATLAB too.**
  `pyvbmc/vbmc/variational_optimization.py:1669` computes
  `var_ss = varG_ss + np.std(varG, ddof=1)`, the sample variance of the
  expected log joint across hyperparameter samples plus the sample standard
  deviation of their per-sample variances. `misc/gplogjoint.m:404` reads
  `varss = varFss + std(varF)`, with MATLAB's `std` normalizing by `N-1`.
  The term has no counterpart in the appendix of the algorithm paper on
  either side.
- **Adam's constants differ from the paper on both sides.**
  `pyvbmc/vbmc/minimize_adam.py:62-63` sets `beta_1 = 0.9`,
  `beta_2 = 0.999`, as `utils/fminadam.m:22-23` does, against the
  `beta_2 = 0.99` of `papers/acerbi2018variational_appendix.md:274`. The
  maximum step size is `min(0.1, 10*sgd_step_size) = 0.05` during warm-up and
  `min(0.1, sgd_step_size) = 0.005` afterwards
  (`pyvbmc/vbmc/variational_optimization.py:268-275`,
  `misc/vpoptimize_vbmc.m:110-118`, `sgd_step_size = 0.005`), against the
  paper's 0.1 and 0.01. The minimum step size 0.001, the decay 200 and the
  batch size 20 are the paper's on both sides.
- **`update_K`'s recent window is MATLAB's, mask included.**
  `pyvbmc/vbmc/variational_optimization.py:53-54` computes
  `recent_iters = ceil(0.5 * tol_stable_count / fun_evals_per_iter)`, which
  is 6 at the shipped defaults (60 and 5), where the paper describes a
  four-iteration window; `private/updateK.m:16` is the same expression with
  the same defaults (`vbmc.m:162-163`). Setting the first two entries of the
  post-warm-up window to `-inf`
  (`pyvbmc/vbmc/variational_optimization.py:65`, `private/updateK.m:24`)
  really does exclude the two oldest entries of that window, on both sides.
- **The `skip_elbo_variance` guard is dead in MATLAB too.**
  `pyvbmc/vbmc/variational_optimization.py:501` tests
  `"skip_elbo_variance" in options`, and no `.ini` file declares the key;
  `misc/vpoptimize_vbmc.m:281` tests
  `isfield(options,'SkipELBOVariance')`, and `vbmc.m`'s `defopts` block does
  not declare it either. Neither branch can execute.
- **The returned posterior is selected with the options that govern it.**
  `rank_criterion`, `best_safe_sd` and `best_frac_back` name the three
  choices `determine_best_vp` (`pyvbmc/vbmc/vbmc.py:2855`) makes when the
  last iteration is not stable, and both call sites (`:1414`, `:1948`) pass
  them, as `vbmc.m:547` and `:888` pass `RankCriterion`, `BestSafeSD` and
  `BestFracBack` to `misc/best_vbmc.m`. The defaults agree
  (`True`/`yes`, 5, 0.25;
  `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:17`, `:223`, `:225`,
  `vbmc.m:201`, `:301-302`). Until 2026-09-19 the three options were declared
  and never passed, and the ranking branch had never executed, because it
  indexed the object-dtype arrays of the iteration history; commit `bd47856`
  "feat(vbmc): select the returned posterior with the options that govern it"
  passes them and reads the history through `asarray`
  (`dev/experiments/port_review_20260919/verification/wave1_options.md`,
  rows O-3 and O-4).
- **The best-posterior rules count iterations, as `best_vbmc` does.**
  `misc/best_vbmc.m:58-59` takes `BackIter = ceil(idx*FracBack)` and
  `idx_start = max(1,idx-BackIter)` from the 1-based iteration index, which
  is also the number of iterations, and `:44` penalizes a non-stable
  iteration by that same count. `pyvbmc/vbmc/vbmc.py:2958-2962` and `:2945`
  use the number of iterations, `max_idx + 1`, where `max_idx` is the
  0-based index of the last one. Commit `4892fe3` (2026-09-19) "fix(vbmc):
  count iterations, not the last index, in the best-posterior rules" made
  both expressions agree with MATLAB
  (`dev/experiments/port_review_20260919/verification/wave1_options.md`,
  row O-5); they affect only runs that end without a stable iteration.
- **The end of warm-up follows `private/vbmc_warmup.m`.** Three
  discrepancies that dated from the port were removed on 2026-09-20
  (`verification/wave2.md`, rows W2-1 to W2-3;
  `verification/wave2_warmup_history.md` for their history). The window of
  recent iterations of the no-recent-improvement test starts at
  `iteration + 1 - T` in 0-based terms, MATLAB's `iter - T + 1`
  (`private/vbmc_warmup.m:60-65`; commit `c918d12`). `_recompute_lcb_max`
  (`pyvbmc/vbmc/vbmc.py:2555`) ports `private/recompute_lcbmax.m`, and
  `_check_warmup_end_conditions` prefers the recomputed vector to the
  recorded maxima as `private/vbmc_warmup.m:46-50` does; an entry is `NaN`
  for an iteration by whose end none of the logged points is still in the
  training set (the maximum is cumulative, so such entries can only open
  the sequence), and the maxima of the check pass over it as MATLAB's `max`
  does (commit `8e591ff`). The end of warm-up sets `last_warping` and
  `last_successful_warping` to the current iteration
  (`private/vbmc_warmup.m:97-102`; commit `fb8a12e`); `LastNonlinearWarping`,
  written on the same lines, is read nowhere in MATLAB and has no
  counterpart.
- **The initial variational means are in transformed coordinates.**
  `VBMC.__init__` transforms `x0` before it builds the variational posterior
  from it (`pyvbmc/vbmc/vbmc.py:563`), as `misc/setupvars_vbmc.m:65` and
  `:82` do. Until commit `8cd4bbc` (2026-09-20) the posterior was built from
  the untransformed `x0` (`verification/wave2.md`, row W2-4).
- **The iteration limits are MATLAB's.** The minimum-iteration guard of
  `_check_termination_conditions` compares the number of iterations
  performed with `min_iter`, as `private/vbmc_termination.m:98-99` compares
  its 1-based counter (commit `567444f`); `max_fun_evals` and `max_iter`
  must be positive integers, and a `max_iter` below `min_iter` is raised to
  it with a warning, as `misc/setupoptions_vbmc.m:109-119` does (commit
  `17badf9`; `verification/wave2.md`, rows W2-5 and W2-26).
- **The noisy-target defaults follow uncertainty handling.** The five
  defaults that change for a noisy target apply whenever uncertainty
  handling is on, with a noise level to infer or one the target returns, as
  `misc/setupoptions_vbmc.m:127-163` applies them (commit `0ba7680`); until
  then they followed `specify_target_noise` alone
  (`verification/wave2.md`, row W2-7).
- **Scalar bounds are replicated.** `_normalize_bounds` replicates a bound
  given as a single value across the variables, as
  `misc/boundscheck_vbmc.m:6-10` does (commit `477226b`, row W2-19).
- **The forced entropy switch and the running average of the moments run.**
  `optimize` reads `entropy_force_switch` from the options at both sites, the
  second with MATLAB's `isfinite` test (`vbmc.m:524-525`,
  `private/vbmc_termination.m:80`; commit `e71f97b`), and the running
  average of the variational moments is taken from the second iteration on
  (`vbmc.m:781-792`; commit `ab5c603`). Before, the first raised with
  `entropy_switch=True` at `D >= 5`, and a misplaced parenthesis kept the
  second from ever running. Nothing reads `run_mean` or `run_cov` for a
  computation on either side (rows W2-10 and W2-11).
- **`final_boost` and `determine_best_vp` leave the run's state alone.**
  `final_boost` sets `warmup = False` and `entropy_alpha = 0` on a copy of
  `optim_state`, and `determine_best_vp` returns a copy of the recorded
  posterior with the stability flag written into the copy, as
  `misc/finalboost_vbmc.m` and `misc/best_vbmc.m` work on by-value structs
  (commits `261b9ae`, `b3ad32b`; minor observations B-M3 and B-M5 of
  `verification/wave2.md`). The recorded posteriors therefore never carry
  the stability flag, on either side.
- **The noise of a candidate point is one value for all GP hyperparameter
  samples.** `active_sample.py:351` stores the mean over the samples of the
  noise at each training point, and VIQR, IMIQR and `AcqFcnNoisy` put that
  one value into each sample's own formula, as
  `private/activesample_vbmc.m:174` (`gp.sn2new = mean(sn2new,2)`) and the
  three MATLAB acquisitions do. §C.1 of the appendix of the 2020 paper
  defines the noise of a candidate as one function of the input, that of its
  nearest training point. At uncertainty level 1, where the fitted multiplier
  carries the noise, the per-sample values spread (1.4 to 3.7 around 1.9 in a
  fit of 26 points); a noise per sample would be a change of the algorithm.
  The value also leaves out the `sn2_mult` of a retried Cholesky
  factorization on both sides (`private/activesample_vbmc.m:172`), which the
  prediction with noise and the GP update apply. (Rows W4-14 and W4-15 of
  `verification/wave4.md`.)
- **The target of IMIQR's MCMC step includes the constant noise term, as
  MATLAB's does.** `is_log_full` predicts with `add_noise=True` and no `s2`
  at the test point, so at uncertainty levels 1 and 2 the constant term alone
  is added, about 1e-5 at the shipped noise floor, while the resampling
  weights and every other density of the importance sampling use the latent
  variance. `log_isbasefun` takes the first two outputs of `gplite_pred`, the
  prediction with noise, and the stored MATLAB output of
  `compare_MATLAB/log_isbasefun.npz` matches the value with noise to 1.4e-4
  and the latent one to 4.8e-2. The estimator is unbiased under either
  choice, the weights using the sampler's own log-densities. (Row W4-13.)
- **Each row of IMIQR's MCMC weights estimates a relative reduction.** A row
  belongs to one GP hyperparameter sample and to a chain drawn from that
  sample's unnormalized density, so it carries the factor `1/Z_s`, `Z_s`
  being the sample's integrated interquantile range before the new point;
  what the row contributes to the acquisition is `N` times the ratio of that
  range after and before. MATLAB's raw rows carry the same factor. The total
  weights of the rows spread widely (17 nats on the stored noisy state of the
  oracles) while their contributions to the acquisition sit within 0.05 nats
  of one another: a chain that stays where the GP is uncertain has small
  weights `1/(2 sinh(u s))` and large terms `2 sinh(u s_pred)`. (Row W4-10,
  `verification/scripts/wave4_A1c_isr_row_totals.py`.)
