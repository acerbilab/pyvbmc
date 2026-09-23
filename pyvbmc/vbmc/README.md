# The MATLAB-to-Python porting log

PyVBMC is a port of the MATLAB VBMC toolbox
([acerbilab/vbmc](https://github.com/acerbilab/vbmc)), and its GP backend
gpyreg is a port of the toolbox's `gplite`. This log records where the Python
code deliberately differs from MATLAB, the behaviors that look like
differences and are MATLAB's own, the notes left open from the port, and the
MATLAB counterparts of the main functions of this subpackage. Several
other subpackages keep notes of their own in their `README.md` files, and
`dev/experiments/port_review_20260919/counterpart_map.md` pairs every
MATLAB file with its Python counterpart.

The MATLAB reference is the `master` branch of acerbilab/vbmc at `396d649`
(2023-05-03). A MATLAB citation `file:line` is at that revision, relative to
the root of that repository. Python code is cited by module and function;
gpyreg as of release 1.3.1.

The catalogue was consolidated from the known-differences sheet of the port
correctness review of September 2026,
[`dev/experiments/port_review_20260919/known_differences.md`](../../dev/experiments/port_review_20260919/known_differences.md).
For each entry, the sheet holds the history of the difference, the
evidence and line citations: those of MATLAB and gpyreg at the revisions
its header names, those of PyVBMC from the review's branch, which can be
off at later revisions. The review's findings and their dispositions are
in its closing ledger,
[`dev/results/2026-09-23-port-correctness-review.md`](../../dev/results/2026-09-23-port-correctness-review.md),
whose per-wave tables record the discrepancies that were fixed to MATLAB's
behavior; corrections of that kind made before the review (the uniform
draws of the search box, the decay of `_get_hyp_cov`, the key of the
variance regularization, the GP-sampling termination) are in the sheet's
"Settled non-differences". This log is the catalogue that is kept
current: a change that creates, removes or alters a deliberate difference
updates its entry here. The sheet stays as the review left it, with a
list of the entries that this log corrects.

Each entry names its kind: a *deliberate change* (both sides implement the
thing, differently and on purpose), an *unported feature* (MATLAB has it,
Python never did), a *removed feature* (Python had it and it was taken out),
a *substituted library* (a third-party package does what a MATLAB file did)
or a *Python-only addition* (no MATLAB counterpart). Where an entry points at
a record, `verification/` is short for
`dev/experiments/port_review_20260919/verification/`, and "MATLAB-side
defect *n*" is entry *n* of
`dev/experiments/port_review_20260919/matlab_side_defects.md`.

## Deliberate differences from MATLAB VBMC

### The GP layer (gpyreg and `gaussian_process_train.py`)

- **The integrated mean function is not ported** (unported feature). MATLAB
  can put a Gaussian prior on the coefficients of a set of basis functions
  and marginalize them analytically, which adds terms to the posterior
  factors, the prediction, the marginal likelihood and the acquisitions.
  gpyreg has no such concept, and PyVBMC keeps the option
  `gp_int_mean_fun = 0` as an inert option (a value other than 0 draws a
  warning). MATLAB: `gplite/gplite_intmeanfun.m` and the `intmeanfun`
  branches of `gplite/private/gplite_core.m`, `gplite_post.m`,
  `gplite_pred.m`, `gplite_train.m`, `acq/acqviqr_vbmc.m`,
  `misc/intkernel.m`, `misc/gpreupdate.m`,
  `misc/check_quadcoefficients_vbmc.m` and
  `private/activeimportancesampling_vbmc.m`. The same holds for VIQR's
  integrated-mean branches and for `check_quadcoefficients_vbmc.m`, which
  exists for this feature (both unported; "Missing port" comments in
  `acq_fcn_viqr.py` and `gaussian_process_train.py`).
- **GP output warping is not ported** (unported feature). MATLAB can fit a
  monotone warping of the GP outputs (`gplite/outwarp_*.m`, option
  `gpOutwarpFun`); gpyreg has no such component. PyVBMC keeps the main-loop
  bookkeeping options (`fitness_shaping`, `out_warp_thresh_*`) and
  `optim_state["out_warp_delta"]` with no warping function behind them.
- **Only three of gplite's mean functions exist** (unported feature).
  gpyreg has `ZeroMean`, `ConstantMean` and `NegativeQuadratic`, where
  `gplite/gplite_meanfun.m` switches over 24 identifiers. `VBMC` refuses the
  other names of `misc/setupvars_vbmc.m:287` at construction and in `load`.
  VBMC's default, `negquad`, is ported.
- **Quantile prediction (`gplite_qpred.m`) is not ported** (unported
  feature); the quantile branch of `GP.plot` is disabled.
- **Sampling from, and minimizing, the density a GP represents is not
  ported** (unported feature): `gplite/gplite_sample.m`,
  `gplite/gplite_fmin.m` and `gplite/private/eissample_lite.m` have no
  counterpart, and nothing in PyVBMC would call them.
- **gpyreg's `SliceSampler` stands in for MATLAB's two samplers**
  (substituted library): `gplite/private/slicesamplebnd.m` for the GP
  hyperparameters and the ensemble sampler `eissample_lite.m` for the MCMC
  step of IMIQR's importance sampling. The second has no Python
  implementation, so that step draws with another algorithm and another
  random stream (see "Noisy importance sampling").
- **Gradient checks use `numdifftools`** (substituted library) in place of
  `gplite/private/derivcheck.m` (`gpyreg/testing/test_utils.py: check_grad`,
  `pyvbmc/testing/_check_grad.py`).
- **The space-filling design and the optimizer of `GP.fit` are SciPy's**
  (substituted library): SciPy's unscrambled Sobol engine and
  `scipy.optimize.minimize` with analytic gradients, where
  `gplite/private/fminfill.m` has its own fill and `fmincon`/`fminunc`. The
  starting points and the optimizer's path are not MATLAB's. MATLAB's
  `OptimToolbox` switch has no counterpart and no reader in MATLAB after
  `misc/setupoptions_vbmc.m:98-106` computes it.
- **NumPy's quantile convention in the bound recommendations of the mean**
  (deliberate change). `np.quantile` interpolates at `(i-1)/(N-1)` where
  `gplite/private/quantile1.m` uses `(i-0.5)/N`, so the plausible bounds of
  the constant of the mean and the starting value of the negative quadratic's
  constant differ by an amount that depends on `N` and on the spread of the
  targets (`gpyreg/mean_functions.py`, against `gplite/gplite_meanfun.m:157-160`,
  `:186`). The same holds for the plausible bounds that `warp_input` takes
  after a warp, against `misc/warp_input_vbmc.m:85-86`. The other statistics
  of the training inputs (standard deviation with `ddof=1`, minimum, maximum,
  median) are taken per column, as gplite takes them. The choice is stated in
  a comment of the code.
- **Isotropic kernels and the rational-quadratic kernel are Python-only**
  (Python-only addition): `SquaredExponentialIsotropic`, `MaternIsotropic`
  and `RationalQuadraticARD`. PyVBMC hard-wires the ARD squared exponential,
  so no run reaches them. MATLAB declares `seiso` and gives it bound
  recommendations, but `gplite/gplite_covfun.m` has no branch that computes
  that kernel. The isotropic bound recommendations take the mean of the
  per-dimension log widths, as the `seiso` branch of
  `gplite/gplite_covfun.m:112-118` does.
- **`GP.predict(return_cross_covariance=True)` hands VIQR the latent
  cross-kernel** (Python-only addition), so VIQR does not recompute it as
  `acq/acqviqr_vbmc.m` does; the caller must treat the returned matrices as
  read-only. Arithmetic
  unchanged (`dev/results/2026-09-13-viqr-kernel-production.md`).
- **`_get_hyp_cov` anchors the hyperparameter dimension on the current
  model** (deliberate change). `misc/get_GPTrainOptions.m`, local function
  `GetHypCov`, lets the first historical block fix the
  width it accepts; `gaussian_process_train.py: _get_hyp_cov` takes the width
  from the current GP, skips historical blocks of another width and returns
  `None` when no usable samples remain or the covariance would rest on one
  effective sample, so that gpyreg's default sampler widths apply
  (`dev/plans/latent-bug-fixes.md`, Phase 3;
  `dev/results/2026-09-08-main-loop-fixes.md`).
- **Sampling from the GP surrogate (`misc/gpsample_vbmc.m`) is not ported**
  (unported feature): `vbmc_rnd.m:49` draws from the GP when `balanceflag`
  is `'gp'`; `VariationalPosterior.sample` takes a boolean `balance_flag`
  only.
- **Noise shaping is not ported, and `noise_shaping=True` is refused**
  (unported feature). MATLAB inflates the noise of training points far below
  the maximum observed density (`misc/noiseshaping_vbmc.m`, default
  `'no'`). What the option would reach in PyVBMC without the shaping, an
  input-dependent noise function and no rank-one update inside active
  sampling, is a configuration of neither toolbox, so `VBMC` refuses it at
  construction and in `load` with a `NotImplementedError` that names the
  missing feature (`verification/wave1_options.md`, row O-2).
- **The lower bounds of the length scales and of the output scale come from
  the high-posterior-density subset** (deliberate change). `_gp_hyp` takes
  gpyreg's recommendations on the `hpd_frac` subset of the training set,
  where `misc/gptrain_vbmc.m:174-180` leaves the two entries unset and
  `gplite/gplite_train.m` fills them from the whole set, so PyVBMC's lower
  bounds are lower, by much when the training set holds a value far below
  the rest (issue 99 and pull request 116, 2022; `verification/wave3.md`,
  W3-11).
- **`hpd_frac` is checked** (deliberate change). `VBMC` refuses, at
  construction and in `load`, a value that is not a fraction in `(0, 1]` or
  that leaves fewer than two points of the initial design, from which the
  first GP fit cannot set its bounds; `misc/gethpd_vbmc.m:10` takes any
  value.
- **`noise_size` is checked** (deliberate change). `VBMC` refuses, at
  construction and in `load` and whatever the uncertainty level, a value
  that is neither empty (`[]`, an empty array or `None`) nor a positive
  finite number, a 0-d array counting as the number it holds.
  `misc/setupoptions_vbmc.m:131-132` refuses a first entry that is not
  positive and takes NaN, infinity and a vector, which
  `misc/gptrain_vbmc.m:148`, `:152` and `:164-165` then pass through `max`
  with `TolGPNoise` into the starting noise.
- **Slice sampling is the only sampler of the GP hyperparameters**
  (unported feature). `VBMC` refuses every `gp_hyp_sampler` but
  `"slicesample"` at construction and in `load`; `misc/get_GPTrainOptions.m:18-91` and
  `gplite/gplite_train.m:316-457` also offer `npv`, `mala`, `slicelite`,
  `splitsample`, `covsample` and `laplace`. `cov_sample_thresh`, read only by
  covariance sampling, is inert (`verification/wave3.md`, W3-13).
- **The evaluation times are not carried onto the GP** (unported feature
  without a reader): MATLAB attaches them as `gp.t`
  (`misc/gpreupdate.m:8`, `misc/gptrain_vbmc.m:76`) and nothing in the
  toolbox reads the field.
- **The window of the burn-in statistics of the slice sampler matches its
  divisor** (deliberate change). `gpyreg/slice_sample.py` accumulates the
  sums behind the adapted widths over exactly `floor(burn/2)` iterations and
  divides by that number; `gplite/private/slicesamplebnd.m:362`, `:367-371`
  sums one term more when the burn-in is odd, so its estimate is no variance,
  is negative wherever a coordinate's spread is small against its mean, and
  the complex square root makes it keep the widths of the shrinking phase for
  every coordinate. PyVBMC's burn-in is odd in most fits (`thin * 3 = 15`, as
  `misc/get_GPTrainOptions.m:108` has it), so MATLAB discards its adaptation
  there as a rule. With an even burn-in the two samplers agree bit for bit on
  one stream. Where its estimate is not positive, gpyreg keeps the width a
  coordinate has, and a window of fewer than two iterations adapts nothing.
  Both are valid slice samplers (MATLAB-side defect 41;
  `verification/wave6.md`, W6-2, W6-3).
- **`GP.quad`'s integral variance is normalized by the scale the factor
  carries** (deliberate change, the fix of MATLAB-side defect 42).
  `gplite/gplite_quad.m:66-67` normalizes by the constant noise
  hyperparameter while `gplite/private/gplite_core.m:82` scales the factor by the minimum
  total noise, so with user-provided or output-dependent noise MATLAB's
  variance collapses to `eps`; gpyreg uses `Posterior.sl` (gpyreg 1.2.1,
  `acerbilab/gpyreg#49`). The mean agrees on both sides. No PyVBMC code calls
  `GP.quad`.
- **`predict(return_lpd=True)` pools the hyperparameter samples** (deliberate
  change). With `separate_samples=False` gpyreg returns the log density of
  the moment-matched Gaussian over the samples, where
  `gplite/gplite_pred.m:124-127`, `:154-165` returns the matrix per sample;
  with `separate_samples=True` the two agree to 1e-13. Both put the total
  predictive variance, latent plus noise times `sn2_mult`, in the density.
  No PyVBMC call passes `return_lpd`.
- **Two smooth-box hyperprior families are Python-only** (Python-only
  addition): `"smoothbox"` and `"smoothbox_student_t"`, flat over `[a, b]`
  with Gaussian or Student's t tails, which `gplite/gplite_hypprior.m` and
  `gplite/private/fminfill.m` do not have. They are identified by a non-finite `mu` beside a
  finite `sigma`, which is why gpyreg's mask for no prior is an `and` (next
  entries). PyVBMC sets only Student's t priors.
- **The hyperprior is renormalized to the bounds, and a hyperparameter with
  equal bounds is held at its value** (Python-only addition). gpyreg
  subtracts the log of the prior mass inside the bounds of each coordinate
  (`cdf(ub) - cdf(lb)` where the lower bound is at or below the prior's
  centre, `sf(lb) - sf(ub)` above it) and gives a coordinate with equal
  bounds a log prior of `-inf` away from its value, a gradient of its own
  prior (zero where it has none) and its value in the space-filling design.
  `gplite/gplite_hypprior.m` has neither. The sum is constant in the
  hyperparameters, so the optimizer and the sampler do not see it while each
  mass is a positive double (to about 37.7 scales of a Gaussian prior). No
  PyVBMC number depends on it (`verification/wave6.md`, W6-24;
  `verification/wave7.md`, W7-14, W7-15).
- **"No prior" is written `None`, not as an infinite scale** (deliberate
  change). `set_priors` takes `None` for a hyperparameter without a prior,
  and within a block a coordinate whose location and `sigma` are both NaN;
  it refuses a `sigma` that is not finite and positive, a location that is
  not finite beside a finite `sigma`, a smooth box with an end that is not
  finite and one with `a > b` (a box of zero width is taken).
  `gplite/gplite_hypprior.m:35` reads either a non-finite location or a
  non-finite `sigma` as no prior, and `gplite/gplite_nlZ.m:13` documents
  `sigma = Inf` as the flat prior; gpyreg cannot, since a non-finite `mu`
  beside a finite `sigma` names its smooth boxes (`verification/wave6.md`,
  W6-17; `verification/wave7.md`, W7-16).
- **An exception of the hyperparameter objective ends the fit** (deliberate
  change). gpyreg lets it propagate from the design and from the optimizer
  loop, where `gplite/private/fminfill.m:104-110`,
  `gplite/gplite_train.m:276-296` and its `gp_objfun` catch it and lose a starting point in silence. NaN values are handled
  alike on both sides. Kept so that a failed factorization that ends a run is
  seen (`verification/wave6.md`, W6-19).
- **The default starting point of `GP.fit`** (deliberate change): with no
  `hyp0`, the current hyperparameters when posteriors exist and otherwise the
  middle of the plausible box, where `gplite/gplite_train.m:98` starts at
  zeros. PyVBMC always passes `hyp0`.
- **A training set of equal targets takes a range of one** (deliberate
  change, a repair of a defect of both sides). With all targets equal the
  range is zero, and the recommended bounds of the output scale and the upper
  bound of the noise are `-inf` on both sides
  (`gplite/gplite_covfun.m:130-131`, `gplite/gplite_noisefun.m:104`,
  `gplite/gplite_train.m:142`); gpyreg gives such a set the spread of a unit range,
  with a warning, as both sides do for a single target (MATLAB-side defect
  53; `verification/wave6.md`, W6-4).
- **`random_function` takes negative eigenvalues of rounding size for
  zeros** (deliberate change, a repair of MATLAB-side defect 46). Where the
  Cholesky factorization of the predictive covariance fails, both sides fall
  back on its eigendecomposition; `gplite/gplite_rnd.m:97-110` refuses a
  factor with a negative eigenvalue and the draw then fails on the sizes,
  while gpyreg takes a negative eigenvalue above `-10 * n * eps` times the
  larger of the largest prior variance at the test points and the largest
  eigenvalue for a zero, and raises `LinAlgError` below it. In the low-noise
  representation gpyreg forms the covariance from a Cholesky factor, where
  `gplite/gplite_rnd.m:50` uses the stored inverse. No PyVBMC code calls it.
- **An infinite width of the slice sampler stays replaced** (deliberate
  change, a repair of MATLAB-side defect 47):
  `gplite/private/slicesamplebnd.m:168`, `:176`, `:377` copy the base widths before replacing an infinite one, so it comes
  back after the burn-in; gpyreg copies after.
- **The degree-1 Matern gradient is zero where two inputs coincide**
  (deliberate change, a repair of MATLAB-side defect 45), where
  `gplite/gplite_covfun.m:198`, `:218-219` gives NaN. PyVBMC uses the squared
  exponential.
- **The noise gradient where the total noise is one value** (deliberate
  change, a repair of MATLAB-side defect 43): with more than one noise
  hyperparameter, `gplite/private/gplite_core.m:244` reads `dsn2(i)` by
  linear index and returns a wrong entry for every hyperparameter after the
  first; gpyreg reads the row.
- **The slice sampler has no Metropolis step** (removed feature):
  `gplite/private/slicesamplebnd.m:189` can interleave Metropolis moves from
  `MetropolisPdf`/`MetropolisRnd`; neither `gplite_train` nor PyVBMC sets
  them (`verification/wave6.md`, W6-21).
- **A schedule of space-filling points without span asks for none**
  (deliberate change). When `min(max_fun_evals, 1e3)` equals
  `fun_eval_start` and no evaluation budget is active,
  `_get_gp_training_options` gives no space-filling starting points, which
  is what `misc/get_GPTrainOptions.m:98-100` computes wherever it gives a
  number; at an evaluation count below `fun_eval_start`, which the warm-up
  trim can leave, MATLAB's count is `Inf` and its design fails (inferred,
  not run), where PyVBMC gives 0. With an active budget the schedule takes
  its end point.

### The main loop, warm-up, termination and final boost (`vbmc.py`)

- **The final boost is guarded** (deliberate change). `final_boost` keeps the
  boosted posterior only when `min(dE, dE - 5 dS) > -tol_elcbo_boost`
  (`dE`, `dS` the changes of the ELBO and its SD; `tol_elcbo_boost = 0.1`),
  and returns the untouched pre-boost posterior with a warning otherwise;
  `misc/finalboost_vbmc.m` accepts its result unconditionally.
  `tol_elcbo_boost = None`, and saved runs without the key, restore the
  unguarded boost (`dev/results/2026-09-04-final-boost-failure.md`;
  `dev/plans/latent-bug-fixes.md`, Phase 2).
- **The final boost runs without the weight penalty and without pruning**
  (deliberate change): `misc/finalboost_vbmc.m` sets `TolWeight = 0` and
  leaves the small-weight shrinkage on; PyVBMC also sets
  `weight_penalty = 0` while the guard is on, and `tol_elcbo_boost = None`,
  like a saved run without the key, keeps the penalty, as MATLAB does
  (`dev/plans/latent-bug-fixes.md`, Phase 2 and Q1).
- **`results["overhead"]` is not implemented** (unported feature): NaN,
  where `vbmc.m:937-939` replaces the NaN that `private/vbmc_output.m:21`
  sets by the fractional overhead, the total running time over the total
  time of the function evaluations minus one (FAQ).
- **`results["rng_state"]` is a snapshot of the run's generator**
  (deliberate change), taken at return time as `{"generator": ...}`, where
  `private/vbmc_output.m` stores the global `rng`.
- **`results["problem_type"]` tests the original bounds** (deliberate
  change): `"bounded"` unless every original bound is infinite.
  `private/vbmc_output.m:5-9` tests the transformed bounds, which are always
  infinite, so MATLAB always reports `'unconstrained'` (MATLAB-side defect
  2). The string for a bounded problem differs (`'boundconstraints'` in
  MATLAB); `results["best_iter"]` is a 0-based index into the iteration
  history, `output.bestiter` 1-based; `results["iterations"]` is the number
  of iterations on both sides; MATLAB's per-iteration total run time
  (`stats.timer(iter).totalruntime`) has no counterpart.
- **An empty warm-up stability window means no stability count yet**
  (deliberate change): with `tol_stable_warmup <= fun_evals_per_iter`
  `_check_warmup_end_conditions` leaves the flag false, where
  `private/vbmc_warmup.m:39`, `:87` hands an empty value to `&&` (by reading,
  an error; MATLAB-side defect 4).
- **The true-posterior diagnostic draws from a copy of the generator**
  (deliberate change): the 10^6 samples behind `sKL_true`
  (`_compute_true_diagnostic`) leave the run's stream alone, where
  `vbmc.m:772-777` draws from the global stream.
- **The separate search GP is not available** (removed feature). MATLAB's
  `SeparateSearchGP` (`vbmc.m:638-648`, default `'no'`) trains a
  constant-mean GP for the search on every second iteration; PyVBMC always
  searches with the main GP, and `separate_search_gp` is inert
  (`verification/wave2.md`, W2-15).
- **The closing display line is computed at every display level, on a copy
  of the generator** (deliberate change). When the returned posterior is not
  the one the loop ended on, `optimize` estimates their symmetrized KL
  divergence from 2 x 10^5 samples drawn with a copy of the run's generator,
  so neither the display level nor the line changes what a run does;
  `vbmc.m:884-913` computes it only when the display asks for it, from the
  global stream (`verification/wave2.md`, B-M9).
- **The final boost of a posterior with fixed means keeps its components at
  the training inputs** (deliberate change). With `variable_means=False`
  `final_boost` takes as many components as the GP has training inputs and
  puts the means there, as the main loop does after warm-up;
  `misc/finalboost_vbmc.m:6` asks for `max(MinFinalComponents, vp.K)` and
  `misc/vbinit_vbmc.m:132-136` keeps the posterior's own means, which by
  reading fails for a run that ends with fewer components (MATLAB-side defect
  14; `verification/wave2.md`, W2-29).
- **MATLAB's `samples` output has no counterpart in `results`** (unported
  feature): the evaluated points and values are on the run's
  `FunctionLogger` (`X_orig`, `y_orig`, `S`, `X_flag`, `n_evals`), with no
  temperature scaling (`vbmc.m:942-957`).
- **The automated retry (`RetryMaxFunEvals`) is not ported** (unported
  feature; `vbmc.m:968-982`; FAQ).
- **`vbmc_diagnostics.m`, the cross-run diagnostic, is not ported**
  (unported feature).
- **Bayesian-optimization warm-up (`BOWarmup`) is not ported** (unported
  feature; default off in MATLAB, absent from PyVBMC's options).
- **A NaN score ranks last in the selection of the returned posterior**
  (deliberate change). `determine_best_vp` ranks with stable sorts that put
  NaN last, takes `np.nanargmax` in the look-back window and, when every
  candidate ELCBO is NaN, selects the last iteration considered.
  `misc/best_vbmc.m:36`, `:40`, `:66` ranks a NaN ELCBO first and can return
  a posterior whose ELBO is NaN; without NaN the two select the same
  iteration (MATLAB-side defect 54; `verification/wave1.md`).

### Setup, options, inputs, history and randomness

- **Options are layered `.ini` files evaluated in Python** (deliberate
  change). `basic_vbmc_options.ini`, then `advanced_vbmc_options.ini`, then
  the `options_path=` file, then the `options=` dict; `.ini` values are
  `eval`'d with `D` bound. An option set in the file counts as set by the
  user, and dependent defaults are settled after every source is read. An
  undeclared name raises on every route, and options are frozen after
  validation. MATLAB (`misc/setupoptions_vbmc.m`, `misc/evaloption_vbmc.m`,
  `utils/evalbool.m`) accepts unknown fields silently and `'yes'`/`'no'`
  strings; `uncertainty_handling` is a boolean (`verification/wave2.md`,
  W2-8, W2-18, W2-20, C-M5).
- **Randomness goes through `numpy.random.Generator` objects** (deliberate
  change). A run reads and writes no global state (`seed=None` derives the
  generator from it once), and every copy of a posterior shares one stream,
  so no draw sequence compares point by point with MATLAB's `rand`/`randn`.
  In an active-sampling step of a noisy run PyVBMC draws the search
  candidates before the importance samples, MATLAB after
  (`private/activesample_vbmc.m:208-218`); neither reads what the other
  draws (`dev/plans/stage1-rng-generator.md`).
- **Vectorized targets, precomputed evaluations and the initialization
  cost** (Python-only addition). `vectorized_target=True` evaluates the
  missing rows of the initial design in one call; `precomputed_evaluations`
  supplies evaluated points whose `setup_cost` is charged once; MATLAB's
  `misc/funlogger_vbmc.m` evaluates one point at a time
  (`dev/plans/stage3-pipeline-features.md`,
  `dev/plans/pymc-target-adapter.md`).
- **The iteration history records `optim_state`, without the noisy
  acquisitions' importance samples** (Python-only addition). MATLAB's
  `savestats` (`vbmc.m:1021-1052`) records no `optimState`; PyVBMC records
  it at every iteration and leaves out the importance samples, which are
  drawn afresh at every step and cannot be rebuilt;
  `record_full_history_details` keeps them.
- **Python-only options of the interface and the infrastructure**
  (Python-only addition): `show_tips`, `print_iteration_header`, the
  `log_file_*` options, `performance_calibration` (a machine-local chunk
  profile, `dev/plans/machine-local-calibration.md`), `do_final_boost` and
  `ns_gp_max_active` (a cap on the hyperparameter samples of the refits
  inside active sampling).
- **The default search acquisition is the log form** (deliberate change):
  `AcqFcnLog` for MATLAB's `acqf_vbmc` (`vbmc.m:213`). The two rank
  candidates identically in exact arithmetic; in floating point the plain
  form underflows and ties candidates the log form still ranks (commit
  `e20d081`, pull request 102; `verification/wave2.md`, W2-25).
- **`integer_vars` takes a boolean mask or 0-based indices** (deliberate
  change), where `misc/setupvars_vbmc.m:14-17` reads 1-based indices; an
  integer array of length `D` holding only zeros and ones is refused as
  ambiguous. Integer variables are experimental (FAQ).
- **Two MATLAB input checks work as written** (deliberate change).
  `_normalize_bounds` repairs every coordinate whose plausible bounds
  coincide, where `misc/boundscheck_vbmc.m:27-30` repairs the first only; and
  `VBMC` checks that the hard bounds of an integer variable lie half an
  integer outside its range, which `misc/setupvars_vbmc.m:19-22` tests with
  an expression that always holds (MATLAB-side defects 10 and 11).
- **`display` takes `"off"`, `"iter"` and `"full"`** (deliberate change),
  the levels of Python's `logging`; MATLAB has `'iter'`, `'notify'`,
  `'final'` and `'off'`. Other values act as `"iter"`.
- **Declared options that nothing reads** (deliberate change). The names of
  `INERT_OPTIONS` (`options.py`) stay declared with their defaults, so that
  recorded option sets still load, and a value other than the default draws a
  warning at construction and in `load`; the `# description` line of each in
  `advanced_vbmc_options.ini` says why it is inert. They cover options that
  MATLAB deleted in 2021 (`2044530`), options MATLAB declares and never reads
  (`AnnealedGPMean`, `ConstrainedGPMean`, `OutputFcn`, `NonlinearScaling`,
  `ProposalFcn`), and the knobs of features PyVBMC never ported; a few more,
  such as `diagnostics` and `active_sample_fess_thresh`, which MATLAB reads,
  have no reader in PyVBMC's design, as their `.ini` lines explain. A test
  recomputes the set by scanning the package for option reads.

### Initial design and active sampling (`active_sample.py`)

- **The `cma` package replaces `cmaes_modded.m`** (substituted library).
  The search runs `cma.fmin` with its draws bound to the run's generator, so
  neither the population draws nor the restarts are MATLAB's. Agreeing with
  MATLAB: the step size `max(insigma)` and the per-coordinate scaling
  (`CMA_stds`), no noise handler, `tolfun` and `maxfevals`. Left at cma's
  defaults on the PI's ruling: `tolx` (absolute, against MATLAB's
  `1e-11*max(insigma)`) and `tolfunhist` (1e-12 against 1e-13), both far
  below `tolfun`; the starting point unevaluated inside cma (its value is the
  sieve's, and the result is kept only when it beats it); and the best-ever
  point returned, where MATLAB returns the best point of the last generation
  (`utils/cmaes_modded.m:1708`), which makes `search_cmaes_best` inert.
  Differing in form: PyVBMC starts the search isotropic at `max(insigma)`
  where an entry of `insigma` is zero, where MATLAB refuses such an
  `insigma` or a ratio above 1e6 and keeps the sieve's point; cma folds candidates into the bounds and caps each coordinate's step
  at a third of its range; and its `tolfacupx` scales per coordinate
  (`utils/cmaes_modded.m`; `verification/wave1_M_P2.md`;
  `verification/cmaes_side_by_side/`).
- **The CMA-ES population is evaluated in one acquisition call**
  (deliberate change), as MATLAB's `EvalParallel` does
  (`utils/cmaes_modded.m:880`, `:917`). One wrapper serves the batched and
  the pointwise route, and the batched route carries the batch dependence of
  `_sq_dist`'s centering, about 2e-15 relative, which can flip a near-tie
  (`dev/plans/stage2-batched-acquisition.md`).
- **`search_optimizer` takes `"cmaes"` or `"none"`, and a one-dimensional
  search is a bounded scalar search** (substituted library). MATLAB's
  `fmincon`, `bads` and `slicesample` branches
  (`private/activesample_vbmc.m:265-315`) have no counterpart. Under
  `"cmaes"` with `D == 1` PyVBMC runs `minimize_scalar(method="bounded")`
  over the search interval, with `maxiter = search_max_fun_evals` and
  `xatol = 1e-11`, because `cma` does not support one dimension; under
  `"none"` no local search runs. `load` replaces a stored `"Nelder-Mead"`,
  which release 1.0.4 wrote into every one-dimensional run, by `"cmaes"` for
  a problem of one dimension and refuses it otherwise
  (`verification/wave5.md`, W5-25).
- **The initial design does not cluster surplus starting points** (unported
  feature). With more than `fun_eval_start` starting points MATLAB picks the
  design by k-means (`misc/initdesign_vbmc.m:28-43`, `utils/fastkmeans.m`);
  PyVBMC takes the first rows. Both leave the other rows in the starting
  cache with their values, where the sieve can take them.
- **An acquired starting point leaves the cache whether or not it had a
  value** (deliberate change); `private/activesample_vbmc.m:380-392`
  deletes the row only when it reuses a stored value, so a row evaluated
  through the target can be drawn and evaluated again (commit `a7f323e`).
- **The stored value of a cached starting point is reused at that point
  alone** (deliberate change). The value is recorded without a target call
  only when the acquired candidate equals the row the sieve made of the
  cached point before its clip; a candidate that the clip or the snap to the
  integer grid moved is evaluated. MATLAB records the stored value at the
  moved candidate (`private/activesample_vbmc.m:388`; MATLAB-side defect 40;
  `verification/wave5.md`, W5-7).
- **The fractions of the acquisition search are checked, and the search
  cache holds no training input** (deliberate change). `VBMC` checks at
  construction and in `load` that the five search fractions lie in `[0, 1]`
  and sum to at most 1, and that `cache_frac` lies in `[0, 1]`;
  `_get_search_points` caps each source at what the sources before it left,
  so it returns the number of candidates asked for, and draws the rows a
  source cannot give (the empty search cache of the first step) from the
  variational posterior. `getSearchPoints` in
  `private/activesample_vbmc.m:565-633` checks nothing and rounds without a
  cap, so its search set can be longer than asked, and at the first step it
  is a quarter short with `search_cache_frac = 0.25` (MATLAB-side defect
  35; `verification/wave5.md`, W5-24). With repeated observations on, the
  training inputs that head the search set as repeat candidates are kept
  out of `optim_state["search_cache"]`; MATLAB never puts a training input
  into its search set.
- **Repeated observations are selected inside the sieve** (deliberate
  change). With `max_repeated_observations > 0` on a noisy target the
  training inputs join the search set and a chosen repeat is pooled into its
  row; MATLAB compares the best training input with the search result after
  the search (`private/activesample_vbmc.m:334-365`), so
  `RepeatedAcqDiscount` and the cost model `t_algoperfuneval` have no
  counterpart. Off by default on both sides.
- **The rank-one GP update is taken for a fresh observation, noisy or not**
  (deliberate change). PyVBMC extends the GP by rank one whenever the
  observation adds a training row, with its noise variance; MATLAB's
  condition (`private/activesample_vbmc.m:481`) also takes it for a repeat on
  a noiseless target, appending a duplicate row (MATLAB-side defect 5), and
  `gplite/gplite_post.m:76-79` recomputes in full whenever a noise variance
  is given. gpyreg's extension reproduces a rebuild to about 1e-15 in the
  Cholesky representation, the one a run enters; after a retried
  factorization it keeps the stored `sn2_mult` (`verification/wave4.md`,
  W4-11; `verification/wave6.md`, W6-9, W6-20).
- **The acquisition-portfolio hedge is not ported** (unported feature).
  `acq_hedge=True` is refused at construction and in `load`; with several
  entries of `search_acq_fcn` PyVBMC picks one uniformly at random per step,
  where `private/acqhedge_vbmc.m` tracks each one's improvement. MATLAB
  itself, with the hedge turned on (`AcqHedge` defaults to `'no'`,
  `vbmc.m:323`) and its default single acquisition, never sets `idxAcq`
  and reads it unset (MATLAB-side defect 33).
- **Variational active sampling (`VarActiveSample`) is not ported**
  (unported feature; `misc/vpsample_vbmc.m`, marked unused in `vbmc.m:652`),
  and PyVBMC declares no option for it.
- **`AbstractAcqFcn._real2int` snaps its input in place** (deliberate
  change), where `misc/real2int_vbmc.m` returns a new array; the batched
  CMA-ES objective reproduces the side effect deliberately. The rounding is
  MATLAB's `round`, a half away from zero.
- **Starting points that leave the initial design without spread are
  refused** (deliberate change, interface). With at least `fun_eval_start`
  starting points, a coordinate that takes one value across the first
  `fun_eval_start` of them raises `ValueError` at construction; MATLAB
  reaches the first GP fit with infinite length-scale bounds
  (`gplite/gplite_covfun.m:105`, `:121-124`; MATLAB-side defect 61;
  `verification/wave7.md`, W7-17).

### Acquisition functions

- **The EIG acquisition and two experimental VIQR losses were removed**
  (removed feature). There is no `AcqFcnEIG` (MATLAB's
  `acq/acqeig_vbmc.m` and its helper `misc/intkernel.m` have no
  counterpart), and `AcqFcnVIQR` offers `loss="iqr"`, MATLAB's acquisition
  and the default, and `loss="iqr_reduction"`, which MATLAB does not have.
  The branch `retain/experimental-acquisitions` holds the removed code
  (`dev/results/2026-09-08-noisy-acquisition-experiments.md`).
- **The quantile of VIQR and IMIQR is an argument, and `u` is computed from
  it** (deliberate change): `u = norm.ppf(quantile)`, a quantile strictly
  between 0.5 and 1, where `acq/acqviqr_vbmc.m:4` and
  `acq/acqimiqr_vbmc.m:4` have the literal `0.6745`. The literal accounts
  for the whole difference between the two implementations' values: on the
  stored noisy state of the oracles they move by up to 1.7e-5 (VIQR) and
  8e-5 (IMIQR), with the same best candidate (pull request 80;
  `verification/wave4.md`, W4-8).
- **A NaN acquisition value stays NaN** (deliberate change), where
  `acq/acqwrapper_vbmc.m:47` turns it into `-realmax`, the best value for the
  search. No shipped acquisition returns one (`verification/wave4.md`,
  W4-22).
- **The variance regularization reads a canonical key with a legacy alias**
  (deliberate change): `variance_regularized_acq_fcn`, falling back to
  `variance_regularized_acqfcn` for saved runs and stored states; the result
  is one value per point before the masks (`dev/plans/latent-bug-fixes.md`,
  Phase 4).

### Noisy importance sampling (`active_importance_sampling.py`)

- **The MCMC step of IMIQR uses gpyreg's `SliceSampler`** (substituted
  library). PyVBMC runs one chain per GP hyperparameter sample from one
  starting point drawn among the proposal samples by importance weight;
  `private/activeimportancesampling_vbmc.m` runs `eissample_lite` with
  `2(D+1)` walkers drawn the same way without replacement (`:206-214`). Only
  the target density is shared.
- **The MCMC refinement of the variational importance samples is not
  ported** (unported feature). An acquisition that sets
  `acq_info["mcmc_importance_sampling"]` is refused, at construction and in
  the importance sampler; no MATLAB acquisition sets the flag
  (`private/activeimportancesampling_vbmc.m:57-92`), and
  `active_importance_sampling_fess_thresh` is inert (`verification/wave4.md`,
  W4-6).
- **The importance log weights are normalized** (deliberate change): one
  log-sum-exp over the whole array is subtracted, a constant that moves the
  values of IMIQR and of VIQR with `loss="iqr_reduction"` and no comparison
  between candidates (`verification/wave4.md`, W4-9).

### Variational optimization and the ELBO (`variational_optimization.py`)

- **The soft bound on the weights' `eta` was removed** (deliberate change).
  `_vp_bound_loss` replaces the `eta` entries of the bounds with infinities
  in a private copy, so the `eta` loss and its gradient are zero, where `misc/vpbndloss.m` penalizes `eta` against `[log(0.5
  TolWeight), 0]`; the location and scale losses and the small-weight
  penalty match MATLAB, and neither `theta` nor the bounds are modified.
  MATLAB's absolute-`eta` bounds penalize differently vectors that represent
  one posterior (`dev/results/2026-09-08-eta-bound-fix.md`,
  `dev/plans/latent-bug-fixes.md`, Phase 6).
- **The diagonal approximation of the variance of the expected log joint is
  not ported** (unported feature): `compute_var == 2` raises, and there is no
  gradient of the variance (`misc/gplogjoint.m`, its `dvarG` accumulators).
- **The weight-only fast paths are not ported** (unported feature):
  `misc/gplogjoint_weights.m` and `misc/vpoptimizeweights_vbmc.m`, a speed
  optimization. MATLAB takes them only with the means and both scales
  frozen (`misc/negelcbo_vbmc.m:54`), which only `vpoptimizeweights_vbmc.m`
  sets, and its one call site (`vbmc.m:716-718`) is commented out, so no
  MATLAB run reaches them.
- **`ELCBOWeight` is not ported; `elcbo_beta` is 0** (unported feature,
  inactive at MATLAB's default of 0).
- **The repository of earlier variational posteriors (`vp_repo`) is not
  ported** (unported feature): `misc/vpsieve_vbmc.m` can seed the sieve from
  it; PyVBMC's sieve builds its candidates afresh.
- **Sampling of the variational parameters is not ported** (unported
  feature): `misc/vpsample_vbmc.m`, `utils/slicesample_vbmc.m`,
  `utils/malasample_vbmc.m`, `VarParamsBack`; the options
  `variational_sampler`, `active_variational_samples` and
  `scale_lower_bound` are inert.
- **`_vb_init` type 3 with a frozen `sigma` copies the existing widths**
  (deliberate change, a dormant path): for the new components when `K`
  grows, where `misc/vbinit_vbmc.m` has no such case
  (`dev/plans/latent-bug-fixes.md`, Q3).
- **The sieve asks for candidates in proportion to the current `K`**
  (deliberate change): `ns_elbo` is evaluated at `vp.K`, 500 candidates at
  `K = 10`. In the main loop `vp.K` is the number of components of the
  posterior entering the optimization, not the count `Knew` that the
  optimization is asked for, whether the means are free or fixed to the
  training inputs: `update_K` returns the new count without changing the
  posterior, and fixed means are replaced by the training inputs without
  changing `vp.K`. `vbmc.m:699` evaluates `NSelbo` at a `K` that
  `vbmc.m:459` sets once to
  `Kwarmup` and never changes, so MATLAB's main loop always asks for 100 (10
  on an incremental iteration); the two agree through warm-up. The warp
  branch and the final boost use the new number of components on both sides
  (`verification/wave1_P6.md`, cmp F3).
- **GP smoothing (`Bandwidth`, `vp.delta`) is not ported** (unported
  feature, inactive at MATLAB's default `Bandwidth = 0`).
- **The full ELCBO evaluation uses the exact entropy of a one-component
  posterior** (deliberate change). `_eval_full_elcbo` asks for no entropy
  samples when `vp.K == 1`, so the reported ELBO, the ranking of the
  optimizations and the pruning rest on the closed-form entropy, where
  `misc/vpoptimize_vbmc.m:279`, `:288-289` estimates it from 4096 samples.
  The comparison of the variational update inside active sampling uses the
  same estimator on both of its sides, where
  `private/activesample_vbmc.m:525-526` samples (commits `7331841`,
  `831b024`; the sampling error removed is measured in
  `verification/wave1_P6.md`, section (b)).
- **The deterministic-entropy optimization is SciPy's BFGS** (substituted
  library): `tol=det_entropy_tol_opt` acts as a gradient tolerance where
  MATLAB's `TolFun` is a tolerance of the value, there is no cap of
  `50*(D+2)` evaluations and no CMA-ES fallback, and a non-success keeps the
  iterate with a warning (`misc/vpoptimize_vbmc.m:73-101`; commit `04d15ff`;
  `verification/wave1_P6.md`, int F7 + cmp F4).
- **The variance of `_neg_elcbo` is computed by default only for a nonzero
  `beta`** (deliberate change, interface). Python has no `nargout`; MATLAB's
  default (`misc/negelcbo_vbmc.m:10`, `:16`) computes it for a caller that
  takes it and passes `compute_grad = 0` (MATLAB-side defect 56). Every
  caller on both sides passes `compute_var` (`verification/wave7.md`,
  W7-9).

### Variational posterior, entropies and statistics

- **Posterior tempering is not ported** (unported feature). `vbmc_power.m`
  and `misc/vptrain2real.m` have no counterpart, `temperature` is inert, and
  the function logger divides nothing by a temperature where
  `misc/funlogger_vbmc.m` does; at `T = 1`, the only temperature a run has,
  `vptrain2real` is the identity. A port of tempering has the logger to
  complete (`verification/wave3.md`, W3-30).
- **The upper-bound entropy (`ent/entub_vbmc.m`) and `AltMCEntropy` are not
  ported** (unported features).
- **`vp.pdf` refuses original-space gradients** (deliberate change) in both
  modes, where `vbmc_pdf.m` returns an uncorrected gradient in the non-log
  case (`dev/plans/latent-bug-fixes.md`, Q2).
- **The soft bounds are the box of the training inputs of the call**
  (deliberate change). `get_bounds` computes the box afresh at every call,
  where `misc/vpbounds.m:9-24` widens a box that the posterior carries, so
  MATLAB's never shrinks after the warm-up trim or a warp. The fitted
  posteriors stay within a small margin of the rebuilt box
  (`verification/scripts/soft_bounds_trace.py`).
- **`kl_div_mvn` takes the log determinants from `slogdet`** (deliberate
  change). The formula is `shared/mvnkl.m`'s, but the log
  determinants come from `np.linalg.slogdet`, where MATLAB's raw
  determinants underflow or overflow for a well-conditioned covariance at
  moderate dimension (MATLAB-side defect 37; `verification/wave5.md`, W5-2).
- **`vp.stats["J_sjk"]` is pruned on both component axes** (deliberate
  change), keeping its shape `(Ns, K, K)`; `misc/vpoptimize_vbmc.m:237`
  prunes the last axis only (`dev/plans/latent-bug-fixes.md`, Phase 1;
  `verification/wave1_P6.md`, cmp sheet note 1).
- **The corner plot uses the `corner` package** (substituted library) for
  `vbmc_plot.m`, `utils/cornerplot.m` and `utils/kde2d.m`.
- **`qtrapz.m` is SciPy's trapezoid rule** (substituted library), the same
  formula: `mtv` integrates with `scipy.integrate.trapezoid`, which agrees
  with `shared/qtrapz.m` bit for bit on short vectors and to about 1e-14 on
  thousands of points, by the order of the sum
  (`verification/wave5_P7.md`, P7-10).
- **The mode search starts from draws of the posterior** (deliberate
  change). `mode` runs `ceil(sqrt(K))` optimizations, each from the densest
  of 1e5 draws (the component means among the candidates of the first),
  drawn with a copy of the generator; L-BFGS-B with finite differences in the
  original space, BFGS with the analytic gradient in the transformed space.
  `vbmc_mode.m:21-47` starts one optimization at each of up to 20 component
  means. A call without `n_opts` returns a stored original-space mode, or
  searches and stores one; a call with `n_opts` neither reads nor writes
  the store; `set_parameters` and `get_parameters` clear it
  (pull request 115, 2022; `verification/wave5.md`, W5-14).
- **`vp.pdf` gives a point on or outside the original bounds a density of
  zero** (deliberate change), where `vbmc_pdf.m` returns NaN or a complex
  value; where the Jacobian underflows the density is the exponential of the
  difference of the logs, and a working copy of the points is float64
  (`verification/wave5.md`, W5-4, W5-18).
- **`kde_1d` is an independent implementation of Botev's estimator**
  (substituted library): it bins to the nearest grid point, where
  `shared/kde1d.m` bins half a step low (MATLAB-side defect 31), falls back
  on Scott's rule where MATLAB minimizes, and floors at 0; `mtv` normalizes
  both, and the two estimates are 2e-4 apart in total variation at
  `nkde = 2^13` (`verification/wave5_P7.md`, P7-6a to P7-6d).
- **The interface of `vp.sample`** (deliberate change; the draws are
  MATLAB's). The component indices are a 0-based `(N,)` array; a whole float
  count is taken and a fractional one refused; a negative `df` is refused;
  weights that do not sum to one raise (`verification/wave5.md`, W5-15,
  W5-16).
- **`vbmc_isavp.m` has no counterpart** (deliberate change): Python checks
  the class.
- **The log density is taken in log space where the density underflows**
  (deliberate change). Under `log_flag`, a row whose Gaussian-mixture
  density is below the smallest normal double takes a log-sum-exp of the
  components and its gradient from their responsibilities;
  `vbmc_pdf.m:58-66`, `:107-110` takes the log of the linear sum, which
  loses precision among subnormal densities and gives `-Inf` with a NaN
  gradient beyond about 38.6 standard deviations from every component
  (MATLAB-side defect 58). The Student-t branches (finite `df`) take the
  log of the linear sum on both sides. The acquisitions
  floor the log density at `log(realmin)`, and the proposal of the active
  importance sampling takes the log of the density as it is held, so no run
  changes; `vp.mode()` refines its start on narrow posteriors
  (`verification/wave7.md`, W7-3).
- **`pdf` refuses points whose width is not `D`** (deliberate change,
  interface): for a one-parameter posterior a row of several points is
  refused, where `vbmc_pdf.m:41` returns one number (MATLAB-side defect 60;
  `verification/wave7.md`, W7-4).
- **`set_parameters` rescales the two scales only when both are optimized**
  (deliberate change, a dormant configuration). With one scale not
  optimized, the objective depends on θ alone on both sides and the
  posterior `optimize_vp` returns is not rescaled at the end, where
  `misc/vpoptimize_vbmc.m:189` rescales it, with the same density
  (`verification/wave7.md`, W7-8).

### Parameter transformer, warping and function logger

- **`warp_cov_reg` is checked** (deliberate change): a function or a finite
  real number, refused otherwise at construction and in `load`, where
  `misc/warp_input_vbmc.m:59-64` checks nothing (commit `cc0ef2f`;
  `verification/wave1.md`).
- **Nonlinear input warping is not ported** (unported feature): only
  `warp_rotoscaling`; MATLAB's `WarpNonlinear` has no counterpart, neither
  an option nor code.
- **`warp_input` maps the search state back with the current transform**
  (deliberate change), the function logger's, where
  `misc/warp_input_vbmc.m:8`, `:133` uses the transform of the posterior
  handed in, which is an older one when that posterior predates an earlier
  warp (MATLAB-side defect 3; `verification/wave2.md`, W2-6).
- **The default bounded transform is probit** (deliberate change), where
  `BoundedTransform` defaults to `'logit'` (`vbmc.m:344`); both are
  available on both sides. Commit `6cee9bb5` (2022-11-24) records the change
  and no reason beyond it.
- **MATLAB's `'g'` action of `warpvars_vbmc.m` is not ported** (unported
  feature, dead in MATLAB). The action of `shared/warpvars_vbmc.m` returns
  the derivatives of the coordinate-wise
  inverse map, not the gradient of the log Jacobian, and its one caller,
  `vbmc_pdf.m:119`, follows an unconditional `error` (MATLAB-side defect 59).
  No PyVBMC computation needs the gradient of the log Jacobian
  (`verification/wave7.md`, W7-12).
- **Rotation matrices are checked to be orthogonal, and their determinant is
  taken to be one** (deliberate change): a non-orthogonal or singular
  `R_mat` is refused at construction (`dev/plans/latent-bug-fixes.md`,
  Phase 1).
- **Transformers compare equal only with equal `scale`, and the logit
  log-Jacobian stays finite in its tail** (deliberate change). MATLAB's
  transforms are structs with no equality, and `shared/warpvars_vbmc.m:500`
  evaluates `-log1p(exp(-y))`, which overflows to `-Inf` below
  `y = -709.78`; PyVBMC switches to a stable form there
  (`dev/plans/latent-bug-fixes.md`, Phase 1).
- **`FunctionLogger.finalize()` is explicit** (deliberate change):
  `optimize` does not call it, so a continued run needs no reallocation; it
  trims every row array of the logger (`dev/plans/latent-bug-fixes.md`,
  decision D4).
- **The warp keeps the covariance when its thresholded form is not positive
  definite** (deliberate change), where `misc/warp_input_vbmc.m:52-71` takes
  the SVD of the thresholded matrix in every case (`verification/wave3.md`,
  W3-6).
- **A variable with one finite bound is refused** (unported feature): the
  log transforms of `shared/warpvars_vbmc.m` for such variables were never
  ported, and `VBMC` rejects such bounds itself, as
  `misc/boundscheck_vbmc.m:138-143` does (`verification/wave3.md`, W3-23).
- **Points within rounding of a bound** (deliberate change). The direct
  transform moves a unit-interval image that rounds to 0 or 1 to the
  adjacent number, where MATLAB returns an infinity (pull request 89); the
  clamp of the inverse uses `nextafter` where MATLAB uses `eps(bound)`, one
  ulp apart at a bound that is a power of two and identical elsewhere; and
  the Student-t inverse groups one product differently, one ulp. Everything
  else in the bounded transforms is bit-identical to a transcription of
  `shared/warpvars_vbmc.m` (`verification/wave3_P8.md`, P8-15).
- **The pooling of a repeated noisy observation has a fallback for extreme
  SDs** (Python-only addition): relative weights where the precisions
  overflow (an SD below about 1e-154), where `misc/funlogger_vbmc.m:234-238`
  gives Inf or NaN; the ordinary path is MATLAB's arithmetic (commit
  `6769a9a`; `dev/plans/pymc-target-adapter.md`).
- **A cached value of a target that provides its noise needs its SD**
  (deliberate change): an `f_vals` that holds a value is refused with
  `specify_target_noise`, and `FunctionLogger.add` raises without an SD at
  uncertainty level 2; `precomputed_evaluations` carries SDs.
  `misc/funlogger_vbmc.m:159-162` means to record SD 1 for a missing SD at
  every noisy level but raises before that line, since both of its callers
  pass the value alone (MATLAB-side defect 16; `verification/wave3.md`,
  W3-28).
- **The evaluation time of a repeated point leaves out unknown times**
  (deliberate change): an unknown time is NaN and stays out of the average,
  where `misc/funlogger_vbmc.m:187`, `:245` records 0. Nothing reads the
  times of single points on either side (`verification/wave3.md`, W3-21).
- **`scale` and the logger's noise flag are checked** (Python-only
  addition): a finite positive `scale` per dimension, and a `noise_flag`
  true exactly above uncertainty level 0 (`verification/wave3.md`, W3-26,
  W3-32).
- **A zero-mean GP is warped** (deliberate change): `warp_gp_and_vp` warps
  the length scales whatever the mean and re-expresses the hyperparameters
  of a constant and a negative quadratic mean; `misc/warp_gpandvp_vbmc.m:37-66`
  misnumbers its cases, so a zero-mean or constant-mean run stops at its
  first warp in MATLAB (MATLAB-side defect 57; `verification/wave7.md`,
  W7-1).
- **The 0-D/1-D input decorator** (Python-only addition): public methods
  promote 1-D inputs to the rigid 2-D shapes; it assumes a `self` first
  argument.

### Priors (`pyvbmc/priors/`)

- **PyVBMC has a prior API** (Python-only addition, with MATLAB's density
  functions inside it). MATLAB supplies the functions `shared/m*pdf.m`,
  `m*logpdf.m` and `m*rnd.m` for the user to assemble a log joint
  (`lpostfun.m`); PyVBMC wraps them as classes (`UniformBox`, `Trapezoidal`,
  `SplineTrapezoidal`, `SmoothBox`), adds `SciPy`, `UserFunction` and
  `Product`, and combines the prior with a log likelihood inside `VBMC`. A
  frozen SciPy distribution's support is read from its `support()`, and a
  prior reads its input as float64.
- **The prior classes check their arguments, and `VBMC` checks a prior
  against the hard bounds** (deliberate change, stricter). The constructors
  refuse non-finite arguments and, for the trapezoids, `a >= u`, `u >= v`
  and `v >= b`. Of these limits MATLAB computes the tent prior `u == v` and
  `v == b` correctly and returns NaN for `u == a`; the PI kept the refusal
  of all three. `VBMC` refuses a prior whose support
  does not contain the hard bounds, with a slack of `1e-9` of the range at
  finite bounds. MATLAB checks less and has no prior object to hold against
  the bounds (MATLAB-side defects 25 and 39; `verification/wave5.md`, W5-5,
  W5-33, W5-38 to W5-40).
- **A NaN coordinate has density zero in every box prior** (deliberate
  change), where `shared/munifboxlogpdf.m:51` gives it the full density
  (MATLAB-side defect 38; `verification/wave5.md`, W5-32).

### Additions with no MATLAB counterpart

S-VBMC (`pyvbmc/svbmc/`, from the standalone `svbmc` package 0.1.1), the
PyMC adapter (`pyvbmc/pymc/`), the Torch and ArviZ exports of the posterior
and the machine-local calibration (`pyvbmc/calibration/`) are additions of
the Python line (`dev/plans/svbmc-integration.md`,
`dev/plans/pymc-target-adapter.md`,
`dev/plans/machine-local-calibration.md`).

## Behaviors that look like differences and are MATLAB's

- **`misc/vbmc_gphyp.m` is an empty file**; `vbmc_gphyp` is a local function
  of `misc/gptrain_vbmc.m:109`, and `_gp_hyp` is its counterpart.
- **The two entropies differ in the weight gradient of one component, on
  both sides.** Without the Jacobian, `entlb_vbmc` returns 0 and
  `entmc_vbmc` returns `H - 1` for `K = 1` (`ent/entlb_vbmc.m:45-47`,
  `ent/entmc_vbmc.m:96-101`); every production caller sets the Jacobian
  flag, under which both are zero. The Monte Carlo gradient keeps its extra
  term rightly: it estimates the derivative of the mixture's total mass.
  MATLAB's fix of that gradient (`1b72896`) is in the Python. Both
  entropies return NaN for a weight of exactly zero with separated
  components, on both sides (`verification/wave5.md`, W5-3, W5-22).
- **Lines of `active_sample.py` that read as defects are MATLAB's own**:
  `recompute_var_post` saved and restored around a block that never writes
  it; `cov(X_hpd, 1)` with the fallback `cov(X)`; the unreachable fallback box
  with the literal 3; candidates clipped into the search box before the
  integer coordinates are snapped, so a snapped candidate can lie outside it;
  an initial design that is not snapped to the integer grid; and training
  inputs not kept out of the candidates (`private/activesample_vbmc.m`,
  `misc/initdesign_vbmc.m`; `verification/wave5.md`, W5-26, W5-27, W5-29).
- **The outer search limits are unused on both sides**: MATLAB computes
  `LB_searchmin` and `UB_searchmin` and comments out their assignment to
  the search bounds (`private/activesample_vbmc.m:503-508`); `active_sample`
  computes them and uses them nowhere.
- **The spread of the per-sample means in `predict` and `quad` divides by
  `Ns - 1`**, the number of hyperparameter samples less one, as `gplite/gplite_pred.m:158`, `:160` and `gplite_quad.m:115`
  do; it reaches `lcb_max`.
- **The quiet omissions of the noise function are MATLAB's**: `s2_star`
  ignored without a user-provided noise feature and the output-dependent
  term dropped without a target (`gplite/gplite_noisefun.m:186-198`), the
  noise evaluated at the test points with neither target nor variance by
  `random_function`, as by `gplite/gplite_rnd.m:67`, and an inverted pair of
  recommended bounds repaired by `UB = max(LB, UB)`
  (`gplite/gplite_train.m:142`). `f_min_fill` with no more evaluations than
  starting points evaluates the first `N` of them and returns those, sorted
  by value, as `gplite/private/fminfill.m:98-114` does
  (`verification/wave6.md`, W6-25, W6-36).
- **`optim_state["N"]` counts the logged distinct locations**, rows made
  inactive by the warm-up trim included, as `vbmc.m` sets `N` from `Xn`, and
  the GP-sampling termination is `private/vbmc_termination.m:42-47`, with
  guards of PyVBMC's own: `N` is back-filled for older saves, and a missing
  or non-finite history never stops the sampling. The counts are refreshed
  after every evaluation, as `misc/funlogger_vbmc.m:278-279` refreshes
  them.
- **`AbstractAcqFcn._sq_dist` centres both point sets on their size-weighted
  mean**, as `gplite/private/sq_dist.m` does.
- **`minimize_adam`'s point and value tables are offset by one iteration**,
  as in `utils/fminadam.m:48-63`, and the returned parameters are paired the
  same way on both sides.
- **`var_ss` adds a standard deviation to a variance**, as
  `misc/gplogjoint.m:404` does; the term has no counterpart in the paper.
- **Adam's constants differ from the paper on both sides**: `beta_2 = 0.999`
  and the maximum step sizes 0.05 and 0.005 (`utils/fminadam.m:22-23`,
  `misc/vpoptimize_vbmc.m:110-118`), against the paper's 0.99, 0.1 and 0.01.
- **`update_K`'s recent window is six iterations at the defaults**, as
  `private/updateK.m:16` computes it, against the paper's four, and the mask
  of its two oldest entries is MATLAB's.
- **The full ELBO of each candidate of the variational optimization comes
  with its variance**, as in MATLAB, whose guard that would skip the
  variance (`misc/vpoptimize_vbmc.m:281`) is dead: no declared option holds
  its key, `SkipELBOVariance`. PyVBMC has no such guard.
- **`final_boost` and `determine_best_vp` leave the run's state alone**, as
  `misc/finalboost_vbmc.m` and `misc/best_vbmc.m` work on by-value structs:
  `final_boost` works on a copy of `optim_state`, and `determine_best_vp`
  returns a copy of the recorded posterior with the stability flag written
  into the copy, so the recorded posteriors never carry the flag, on
  either side.
- **The iteration history stores GPs without their posterior factors**, as
  MATLAB's does: `savestats` in `vbmc.m` records `gplite_clean(gp)`
  (`vbmc.m:1044`), which keeps the hyperparameter samples and empties
  `alpha`, `sW`, `L`, `sn2_mult` and `Lchol`, and
  `misc/finalboost_vbmc.m:36` rebuilds them with `gplite_post`. `_lean_gp`
  keeps the training data, the hyperparameter samples and the model, and
  `VBMC.get_gp` rebuilds identical factors on demand
  (`dev/plans/stage2-memory.md`). MATLAB removes the GPs from the returned
  `stats` unless `Diagnostics` is set (`vbmc.m:959-966`); PyVBMC's history
  keeps them, and `diagnostics` is inert.
- **The noise of a candidate point is one value for all GP hyperparameter
  samples**, their mean, and leaves out the `sn2_mult` of a retried
  factorization, as `private/activesample_vbmc.m:172-174` does
  (`verification/wave4.md`, W4-14, W4-15).
- **The target of IMIQR's MCMC step adds the constant noise term**, as
  MATLAB's `log_isbasefun` does, while the other densities of the importance
  sampling use the latent variance (`verification/wave4.md`, W4-13).
- **Each row of IMIQR's MCMC weights estimates a relative reduction**, so
  the rows' total weights spread widely while their contributions to the
  acquisition agree, on both sides (`verification/wave4.md`, W4-10).
- **Three properties of the warps and the transforms are MATLAB's**: the
  search box after a warp is the extent of 1000 mapped draws of the old box,
  short of the exact image by a volume of at most about `2D/1001`
  (`misc/warp_input_vbmc.m:143-148`); every bounded transform resolves the
  distance to an upper bound only to `(b - a) eps/2`, a loss beyond the
  representation of the point where `abs(b) < b - a`, while a lower bound
  keeps full precision (`shared/warpvars_vbmc.m:106-108`, `:256-258`,
  `:442-459`); and the Student-t(4) forward transform loses up to about
  1.6e-8 near the midpoint by cancellation (`:265-269`). The Torch export
  computes the distance to each bound separately. All three are left as they
  are (`verification/wave7.md`, W7-5 to W7-7).
- **The end of warm-up follows `private/vbmc_warmup.m`**: the window of the
  no-recent-improvement test, the recomputed LCB maxima
  (`private/recompute_lcbmax.m`), whose entries can be NaN at the start of
  the sequence and are passed over as MATLAB's `max` does, and the warping
  clocks set at its end; `LastNonlinearWarping` has no reader in MATLAB and
  no counterpart.

## Open notes from the port

- `VBMC` exposes `final_boost` and `determine_best_vp`, which `optimize` calls
  as steps of a run, beside the user interface (the constructor, `optimize`,
  which continues a finished or a loaded run, `save`, `load` and `get_gp`).
  Whether they should stay public, or the steps of a run should be opened to
  users in a designed way, is open.
- The logging was noted during the port as not working well, to be
  redesigned around a logging concept; its shortcomings are not recorded in
  this repository.

## Design notes

- **Options class.** In order to prevent setting options after
  initialization, but allowing for an override to this behavior, I added an
  `is_initialized` flag to the Options class and a special keyword to the
  `Options.__setitem__` method, which means that `options['foo'] = 'bar'`
  will raise an error after initialization, but
  `options.__setitem__('foo', 'bar', force=True)` will change the setting
  without error. This mean also adding a custom `Options.deepcopy()` method,
  and for completeness I added a similar `Options.copy()` method. I added
  simple tests for these copy methods, but am adding a warning here in case
  something breaks down the line. (Bobby H., 23.03.2022) In the code the
  copy methods are `__copy__` and `__deepcopy__`, and the freeze also
  covers removal, through `__delitem__`.
- **The import order of `pyvbmc/priors/__init__.py`** is fixed by
  `# isort:skip` markers, which avoid circular imports; it is not to be
  reordered.

## MATLAB references

The main functions of this subpackage and their MATLAB counterparts; the
links point at MATLAB's `master` branch, which is at `396d649`, and
`dev/experiments/port_review_20260919/counterpart_map.md` has the full map.

### active_sample.py
- active_sample(): [activesample_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/private/activesample_vbmc.m)
- active_sample() (part where gp is None): [initdesign_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/initdesign_vbmc.m)
- _get_search_points(): getSearchPoints in [activesample_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/private/activesample_vbmc.m)

### iteration_history.py
- the local function savestats in: [vbmc.m](https://github.com/acerbilab/vbmc/blob/master/vbmc.m)

### vbmc.py and related functions
- initialization:
     - parts of the __init__(): [vbmc.m](https://github.com/acerbilab/vbmc/blob/master/vbmc.m)
     - parts of the __init__(): [setupvars_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/setupvars_vbmc.m)
     - _bounds_check(): [boundscheck_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/boundscheck_vbmc.m)
- VBMC loop (optimize(): loop in [vbmc.m](https://github.com/acerbilab/vbmc/blob/master/vbmc.m)):
     - Active Sampling:
          - see active_sample.py above
          - reupdate_gp() in gaussian_process_train.py: [gpreupdate.m](https://github.com/acerbilab/vbmc/blob/master/misc/gpreupdate.m)
     - Gaussian Process training:
          - train_gp(): [gptrain_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/gptrain_vbmc.m)
          - _gp_hyp(): [gptrain_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/gptrain_vbmc.m)
          - _get_gp_training_options(): [get_GPTrainOptions.m](https://github.com/acerbilab/vbmc/blob/master/misc/get_GPTrainOptions.m)
          - _get_hyp_cov(): [get_GPTrainOptions.m](https://github.com/acerbilab/vbmc/blob/master/misc/get_GPTrainOptions.m)
          - get_hpd() in pyvbmc/stats/get_hpd.py: [gethpd_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/gethpd_vbmc.m)
          - _get_training_data(): [get_traindata_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/get_traindata_vbmc.m)
          - _estimate_noise(): [gptrain_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/gptrain_vbmc.m)
     - Variational optimization / training of VP:
          - update_K(): [updateK.m](https://github.com/acerbilab/vbmc/blob/master/private/updateK.m)
          - optimize_vp(): [vpoptimize_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/vpoptimize_vbmc.m)
          - _initialize_full_elcbo(): [vpoptimize_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/vpoptimize_vbmc.m)
          - _eval_full_elcbo(): [vpoptimize_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/vpoptimize_vbmc.m)
          - _vp_bound_loss(): [vpbndloss.m](https://github.com/acerbilab/vbmc/blob/master/misc/vpbndloss.m)
          - _soft_bound_loss(): [softbndloss.m](https://github.com/acerbilab/vbmc/blob/master/utils/softbndloss.m)
          - _sieve(): [vpsieve_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/vpsieve_vbmc.m)
          - _vb_init(): [vbinit_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/vbinit_vbmc.m)
          - _neg_elcbo(): [negelcbo_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/negelcbo_vbmc.m)
          - _gp_log_joint(): [gplogjoint.m](https://github.com/acerbilab/vbmc/blob/master/misc/gplogjoint.m)
     - Loop termination:
          - _check_warmup_end_conditions(): [vbmc_warmup.m](https://github.com/acerbilab/vbmc/blob/master/private/vbmc_warmup.m)
          - _setup_vbmc_after_warmup(): [vbmc_warmup.m](https://github.com/acerbilab/vbmc/blob/master/private/vbmc_warmup.m)
          - _recompute_lcb_max(): [recompute_lcbmax.m](https://github.com/acerbilab/vbmc/blob/master/private/recompute_lcbmax.m)
          - _check_termination_conditions(): [vbmc_termination.m](https://github.com/acerbilab/vbmc/blob/master/private/vbmc_termination.m)
          - _compute_reliability_index(): [vbmc_termination.m](https://github.com/acerbilab/vbmc/blob/master/private/vbmc_termination.m)
          - _is_gp_sampling_finished(): [vbmc_termination.m](https://github.com/acerbilab/vbmc/blob/master/private/vbmc_termination.m)
- Finalizing:
     - final_boost(): [finalboost_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/finalboost_vbmc.m)
     - determine_best_vp(): [best_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/best_vbmc.m)
     - _create_result_dict(): [vbmc_output.m](https://github.com/acerbilab/vbmc/blob/master/private/vbmc_output.m)
