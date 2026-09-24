# Changelog

All notable changes to PyVBMC are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Changes since PyVBMC 1.0.4, to be released as PyVBMC 1.5.

Many of the changes and fixes come from a systematic comparison of PyVBMC with
VBMC, the original MATLAB implementation.

### Upgrading from 1.0.4

What can stop an existing script, or change what it returns. Each point has
its entry below.

- Results differ from 1.0.4, also with a fixed seed.
- PyVBMC needs Python 3.10 or later, and no longer installs `pytest` and
  `plotly`.
- Option values are checked: an unknown option name in an options file,
  `uncertainty_handling=[1]`, a `specify_target_noise` that is not `True` or
  `False`, an ambiguous `integer_vars`, a `max_fun_evals` or `max_iter`
  that is not a positive integer, or a `min_iter` that is not a finite
  non-negative integer raises an error.
  So do a `gp_mean_fun` or a `gp_hyp_sampler` that PyVBMC does not implement,
  `f_vals` together with `specify_target_noise`, a `quantile` of `AcqFcnVIQR`
  or `AcqFcnIMIQR` outside (0.5, 1), and a `search_acq_fcn` string that
  cannot be read as a call. `search_optimizer` takes `"cmaes"` or
  `"none"` (the `"Nelder-Mead"` value is removed), `acq_hedge=True` and
  `noise_shaping=True` raise an error, also when `VBMC.load` finds them in
  a saved run (the message gives the `new_options` that loads it), and so
  do fractions of the acquisition search that add up to
  more than one and a `cache_frac` outside [0, 1]. `integer_vars` given as
  a list of indices marks those
  variables, where 1.0.4 marked every variable. A `warp_cov_reg` that is
  neither a finite number nor a function (`True`, which 1.0.4 read as 1,
  among them) raises an error, and so does an `hpd_frac` that is not a
  fraction in (0, 1] or that leaves fewer than two points of the initial
  design. A `noise_size` that is neither empty nor a positive finite number
  (zero or a negative value, which 1.0.4 read as `tol_gp_noise`, among
  them) raises an error, also when `VBMC.load` finds it in a saved run.
  An `active_importance_sampling_mcmc_samples` that is neither a finite
  number nor a function (`True` and NaN among them) raises an error, also
  for a run that does not read it and when `VBMC.load` finds it in a saved
  run. So does a `gp_sample_thin` that is not a whole number greater than
  zero, whatever `ns_gp_max`: 1.0.4 ran a boolean as its integer, and any
  value with `ns_gp_max=0`.
  A `log_file_level` other than `"off"`, `"iter"`, `"full"` or a
  non-negative integer raises an error, also when no `log_file_name` is
  given. `None` and `False`, with which 1.0.4 wrote no log file, are among
  them, and so are the names `"notify"` and `"final"`, which 1.0.4's
  description of the option listed, and a whole float such as `10.0`: 1.0.4
  took these when no log file was named. `log_file_level=0` writes the log
  file, where 1.0.4 wrote none.
- `VBMC.load(file, new_options=...)` raises for an option that only
  construction reads (`uncertainty_handling`, `gp_mean_fun`, `integer_vars`,
  `warmup` and their kin; the message names them) given a value other than
  the one the run stores, where such a value used to be stored and ignored;
  construct a new `VBMC` object to run with one. The stored value itself is
  taken, so the options a run was built with can be given back with a new
  budget.
  A saved run whose stored `gp_mean_fun` names a mean function PyVBMC does
  not implement raises on `load` as it does at construction.
- With `uncertainty_handling=True`, or a noisy setting in an options file, the
  defaults for noisy targets apply, a larger budget of evaluations among
  them.
- `vbmc.determine_best_vp()` without arguments follows the options
  `rank_criterion` (on by default), `best_safe_sd` and `best_frac_back`, so
  it can return another iteration than in 1.0.4 when the last iteration is
  not stable. It returns a copy of the selected posterior, so changing that
  posterior no longer changes the one recorded in
  `vbmc.iteration_history["vp"]`, and `vbmc.final_boost(vp, gp)` leaves
  `vbmc.optim_state` as it is, where 1.0.4 changed it. A variational
  optimization in which every candidate has a NaN ELBO raises `ValueError`,
  where 1.0.4 went on with a posterior whose ELBO was NaN.
- Once a `VBMC` object is constructed, `del vbmc.options[name]` and
  `vbmc.options.pop(name)` raise an error, as assigning to an option already
  did in 1.0.4.
- `results["iterations"]` is the number of iterations, one more than in 1.0.4.
  `results["problem_type"]` is `"bounded"` for a problem with bounds, where
  1.0.4 said `"unconstrained"` for every problem.
- An entry of `vbmc.iteration_history["gp"]` cannot make predictions; call
  `vbmc.get_gp(iteration)`. An entry of
  `vbmc.iteration_history["gp_hyp_full"]` holds the hyperparameter samples
  from before thinning, five times as many rows at the default
  `gp_sample_thin`. The `vbmc.hyp_dict` of a run started by this release
  has no `logp` entry.
- `vp.pdf(x, grad_flag=True)` raises an error in the original parameter space.
  `vp.pdf` and `vp.log_pdf` raise an error for an `x` whose rows do not have
  `D` coordinates; for a posterior of one parameter, several points are
  passed as a column (`x.reshape(-1, 1)`).
- Other methods of the variational posterior return something else in
  some calls: `vp.kl_div(samples=...)`, `vp.kl_div(gauss_flag=False)` with
  densities that are not finite, `vp.pdf` at integer or float32
  coordinates,
  `vp.moments` of a posterior of one parameter (a 1-by-1 covariance), the
  component indices of `vp.sample` (always a flat array of integers), and
  `vp.mode()`, which no longer advances the random stream of the run,
  runs a search of its own when `n_opts` is given, searches again
  after `vp.get_parameters()`, which discards the stored mode, and refines
  its starting point on a narrow posterior. `vp.log_pdf` is finite far in
  the tails, where 1.0.4 returned `-inf`, and `vp.set_parameters` leaves
  `sigma` or `lambd` as it is when its `optimize_sigma` or `optimize_lambd`
  flag is off.
  `vp.moments` and `vp.kl_div` refuse a number of samples that is not
  whole, which 1.0.4 truncated.
  `vp.set_parameters(theta, raw_flag=False)` refuses a negative scale or
  weight and accepts negative means. `pyvbmc.stats.get_hpd` rounds the size
  of its subset away from zero at a tie. `pyvbmc.stats.kl_div_mvn`, and
  with it the `sKL` of a run, is finite for covariances whose
  determinants leave the range of a double, where 1.0.4 gave zero or
  infinity.
- `VBMC` refuses at least `fun_eval_start` starting points (10 up to nine
  variables) that take one value in a coordinate across the first
  `fun_eval_start` of them; give points that vary in every coordinate, or
  fewer of them.
- Priors: `VBMC` refuses a prior whose support does not cover the hard
  bounds. The prior classes refuse a bound, a pivot or a scale that is NaN
  or infinite, an array argument whose shape disagrees with `D` (as
  `pyvbmc.priors.tile_inputs` does for its `size`), and a one-dimensional
  `scipy.stats` distribution whose parameters are arrays. `a` and `b` of
  a `SciPy` or `Product` prior are read-only.
  `support()` of a shifted or scaled SciPy prior, the density of a prior at
  integer or float32 points, and the density of `UniformBox` at a point
  with a NaN coordinate return other values.
- `results["rng_state"]` and `vbmc.random_state` hold the state of `vbmc.rng`.
- Classes of your own: a prior needs `sample(self, n, rng=None)` to be part of
  a `Product` prior, and an acquisition function must return one value per
  input point and must not set `acq_info["mcmc_importance_sampling"]`.
- A function given for `adaptive_k` receives the number of components by the
  keyword `K`, or by position when it takes no argument of that name, where
  1.0.4 passed it by the keyword `unkn`. One that takes `unkn` by keyword
  alone (`lambda *, unkn: ...`, `lambda **kw: kw["unkn"]`) raises an error,
  and one with parameters named both `K` and `unkn` receives the number in
  `K`.
- `ParameterTransformer`, `FunctionLogger` and `unscent_warp` used on their
  own: a variable with one finite bound, a `scale` that is not positive, a
  noise flag that contradicts the uncertainty handling level, and `add`
  without an SD for a target that provides its noise raise an error. The
  logger returns a float for a repeated point, `unscent_warp` returns
  floating-point values, and a transformer given a rotation or a scale
  together with plausible bounds centres the plausible box differently.

### Added

- **Seeded runs.** `VBMC(..., seed=...)` fixes all the random draws of a run.
  `seed` takes what `numpy.random.default_rng` takes: an integer, a
  `SeedSequence` or a `Generator`. The generator is `vbmc.rng`, and the
  posterior shares it as `vp.rng`, so `vp.sample()` continues the same stream.
  On one machine, two runs with the same seed give the same result whatever
  happens to NumPy's global random state, which a run neither reads nor writes
  once the `VBMC` object exists. With `seed=None`, the default, the generator
  is derived from NumPy's global state, so `np.random.seed(...)` before
  creating the `VBMC` object still fixes a run, as it did in 1.0.4. In 1.0.4
  every draw came from the global state, and any other code that used that
  state changed the run.
  - `seed=` does not reach inside the target. A target that draws random
    numbers of its own, such as a simulator, has to be seeded separately.
  - A copy of a posterior (`copy.deepcopy`) shares the generator of the
    original, so sampling from the copy advances the stream of both.
  - `VariationalPosterior(..., rng=)` and the `sample(n, rng=)` method of the
    priors accept a generator as well. A prior class of your own that
    overrides `sample(self, n)` needs the `rng=None` argument too before it
    can be part of a `Product` prior; `UserFunction` priors need no change.
- **S-VBMC: stack the posteriors of several runs.** `pyvbmc.SVBMC` takes the
  variational posteriors of several VBMC runs on the same problem and combines
  all their components into one posterior, by optimizing the weights of the
  components. It needs no further evaluations of the target. Use it when
  repeated runs disagree, or when each run captures a part of a multimodal
  posterior. S-VBMC is the method of Silvestrin, Li & Acerbi (2025), so far
  available as the standalone `svbmc` package; PyVBMC includes that package
  in an improved and faster version, described under "What differs" below.

  ```python
  from pyvbmc import SVBMC

  stacked = SVBMC(vps, seed=0)     # vps: the posteriors returned by the runs
  stacked.optimize()
  samples = stacked.sample(10000)  # in the original parameter space
  print(stacked.elbo, stacked.elbo_sd)
  ```

  - S-VBMC needs PyTorch: `pip install "pyvbmc[torch]"`. `import pyvbmc` does
    not import Torch, and creating an `SVBMC` object without it raises an
    error that names the extra.
  - The runs must have the same number of variables and the same hard bounds.
    Runs that did not end stable, or whose estimate of the expected log joint
    is too uncertain (`SVBMC(..., s_max=)`), are left out, and an error is
    raised when too few remain (`SVBMC(..., M_min=)`).
  - `stacked.elbo` is the ELBO of the stacked posterior, an approximate lower
    bound of the log evidence, and `stacked.elbo_details` holds every estimate
    that was computed. When any of the runs that are kept had a noisy target,
    the expected log joint that enters `elbo` is limited to the median of its
    values over the components, because weights chosen to maximize noisy
    estimates make the plain value optimistic; the limit lowers `elbo` only
    when the plain value exceeds it. `stacked.elbo_sd` is the uncertainty of
    the plain value, without that limit, and it does not account for the
    choice of the weights, so `elbo ± 2 * elbo_sd` is not a calibrated
    interval.
  - A posterior records whether its run handled a noisy target, in
    `vp.stats["uncertainty_handling_level"]`. For a posterior saved by an
    earlier version, S-VBMC guesses it from `vp.stats["elbo_sd"]`; pass
    `SVBMC(..., noisy=True)` or `noisy=False` to settle it.
  - See the `SVBMC` page of the documentation and Example 7. Please cite the
    S-VBMC paper along with those of VBMC and PyVBMC when you use it.
  - What differs from the standalone `svbmc` package (0.1.1). The method and
    the optimizer of the weights are the same, and on our benchmark the two
    reach the same weights.
    - It is faster: the optimization ran 1.9 to 4 times faster on our
      benchmark, a provisional figure measured on one machine. The work that
      the components of one run share, its parameter transform above all, is
      done once per run and no longer once per component.
    - The reported ELBO is more precise. The corrections that bring the
      expected log joint of each run to the original parameter space are
      computed once, when the object is created, exactly or by numerical
      quadrature; the standalone package estimated them by Monte Carlo at
      every step. After the optimization the ELBO is evaluated again with
      more draws (`optimize(n_samples_final=)`), and it comes with an
      uncertainty, `elbo_sd`.
    - `elbo` is a number. The three entries of the old `elbo` dictionary are
      in `elbo_details` under new names, which the `SVBMC` page of the
      documentation lists.
    - `seed=` takes the place of `testing=`. All the random draws come from
      the generator of the `SVBMC` object, and the input posteriors are
      neither modified nor advanced.
    - `sample(n)` returns exactly `n` independent draws. The standalone
      package rounded the share of each run separately and could return a
      few more or fewer.
    - All results are float64, problems with one variable work, and malformed
      inputs raise errors.
- **PyMC models as targets.** `PyMCTarget` wraps a PyMC model so that PyVBMC
  can fit it. It provides the log joint, the bounds, a starting point and the
  plausible box.

  ```python
  from pyvbmc import VBMC, PyMCTarget

  target = PyMCTarget(model, seed=7)
  vp, results = VBMC(target, seed=7).optimize()
  data = target.to_arviz(vp)       # draws in the model's own variables
  ```

  - Install it with `pip install "pyvbmc[pymc]"`. It needs PyMC 6.3 or later
    and Python 3.12 or later.
  - `target.to_arviz(vp)` exports draws under the names and shapes of the
    model's variables; `vp.to_arviz()`, below, knows the flat parameters
    only.
  - A free variable of the model must be continuous, and either carry one of
    PyMC's standard log, log-odds or interval transforms or have one of the
    recognized distributions on the whole real line (normal, Student's t,
    Cauchy, Laplace, logistic, Gumbel, skew-normal, multivariate and matrix
    normal). Anything else is refused with `pyvbmc.pymc.UnsupportedModel`: a
    discrete variable, another transform (a simplex, an ordered or a custom
    one), another distribution without a transform, and bounds that depend on
    other random variables.
  - The adapter copies the model's data when it is created, so a later
    `pm.set_data` does not reach it. Create a new `PyMCTarget` for new data.
  - Creating the adapter spends a limited budget of model evaluations
    (`PyMCTarget(..., setup_budget=)`, by default `20 + 5 * D`, where a
    Hessian costs `D`). The run reuses these evaluations, and their cost
    counts once against `max_fun_evals`.
  - See the `PyMCTarget` page of the documentation and Example 8.
- **Vectorized targets.** With `options={"vectorized_target": True}`, the
  target receives an array of shape `(N, D)` and returns one value per row. A
  noisy target with `specify_target_noise` returns a pair of arrays, the
  values and their standard deviations. PyVBMC then evaluates the whole
  initial design in one call. Later evaluations are still made one point at a
  time, as calls with shape `(1, D)`. A separate prior (`prior=`,
  `log_prior=`) keeps its one-point interface. The option is not available
  with a `PyMCTarget`.
- **Posterior exports.**
  - `vp.to_torch()` returns the posterior as a `torch.distributions` object in
    the original parameter space, with a `log_prob` that can be
    differentiated with respect to its input. It needs
    `pip install "pyvbmc[torch]"` (Torch 2.7 or later). Tensors are float64 on
    the CPU unless `dtype=` and `device=` say otherwise. For a problem with
    bounds, `log_prob` takes points strictly inside the bounds.
  - `vp.to_arviz()` draws samples and returns them as an ArviZ `DataTree`. It
    needs `pip install "pyvbmc[arviz]"` and Python 3.12 or later. `var_names=`
    names the parameters; `variables=`, `dims=` and `coords=` group them into
    named vector and matrix variables. The draws are independent samples from
    the variational posterior, so MCMC convergence diagnostics do not apply
    to them.
- **Evaluations made before the run.**
  `VBMC(..., precomputed_evaluations=(X, y))` gives a run target values that
  were computed beforehand (`(X, y, y_sd)` with `specify_target_noise`). They
  enter as observations, they are independent of `x0`, and they do not count
  as function evaluations of the run. With a separate prior, `y` holds
  log-likelihood values and VBMC adds the prior; `options["f_vals"]`, the
  values at `x0`, holds log-joint values, as in 1.0.4. `VBMC(..., initialization_cost=k)`
  charges `k` evaluations against `max_fun_evals` for work done before the
  run. `results` reports the first in `precomputed_observations` and
  `precomputed_locations`, present when at least one evaluation was given,
  and the second in `evaluation_budget`, present when `initialization_cost`
  is above zero.
- **Machine calibration (optional).** `pyvbmc.calibrate()` measures, on your
  machine, the block sizes that PyVBMC uses when it evaluates the posterior
  density and the Monte Carlo entropy, and saves them in a cache that later
  runs read. It takes tens of seconds, never evaluates your model, and runs
  only when you call it. It changes speed, not results. Unless the display is
  off, a run prints one line saying whether it uses the standard settings or
  a saved calibration, and the first run of a Python session that finds no
  saved calibration prints one more line suggesting it.
  `options={"performance_calibration": "off"}` keeps the standard settings, a
  `CalibrationProfile` passed there fixes them for a run, and
  `results["performance_calibration"]` records what a run used.
- **Tips.** A new run may print a short tip before the iteration display, for
  example on how many runs to make or how to choose the plausible bounds. A
  run that prints the suggestion to calibrate shows no tip. Of the other runs
  of a Python session, the first shows a tip and then every third one does,
  each tip at most once per session. `options={"show_tips": False}` turns the
  tips off, and so does `display="off"`. `SVBMC(..., show_tips=False)` does
  the same for S-VBMC.
- **Repeated observations of a noisy target.** With
  `max_repeated_observations` set to a positive number, active sampling may
  evaluate a noisy target again at a point it has already evaluated, up to
  that many times in a row. The observations made at one point are pooled
  into one, which also updates the largest observed value and the total
  evaluation time that the function logger keeps (1.0.4 updated neither for
  a pooled observation). The training inputs offered as candidates for a
  repeat stay out of the search cache that `search_cache_frac` keeps. In
  1.0.4 the option changed the iteration display and nothing else. Its
  default, 0, is unchanged.
- **Smaller additions.** The option `ns_gp_max_active` caps the number of GP
  hyperparameter samples of the GP fits that a run on a noisy target makes
  between the evaluations of one iteration; the default leaves them as they
  were. `AcqFcnVIQR(loss="iqr_reduction")` scores the reduction of the
  integrated interquantile range; the default `loss="iqr"` is the acquisition
  function of 1.0.4.
- **Documentation.** A FAQ page. Example 7 (S-VBMC), Example 8 (a PyMC model)
  and Example 9 (targets written in Torch or JAX, and the posterior exports).
  New quickstart sections on reproducible runs, Torch and JAX targets,
  vectorized targets, PyMC models and the use of a fitted posterior. The
  noisy target of Example 6 computes its noise point by point, so it also
  works when it is called with several points at once.
  `skills/pyvbmc/SKILL.md` in the repository guides a coding agent through the
  documentation.

### Changed

- **Requirements.** PyVBMC needs Python 3.10 or later (1.0.4 accepted 3.9),
  SciPy 1.15 or later and gpyreg 1.3.2 or later. gpyreg 1.3.0 takes the
  bounds of the location and scale of the GP mean function, and the
  starting length scales, per input dimension, where it pooled the
  statistics of the training inputs over all dimensions; the
  hyperparameter samples of every GP fit that samples, and with them the
  results of every run, move. gpyreg 1.3.1 repairs parts of its hyperprior
  code and of the space-filling design of its fit, which a run meets only
  on a noiseless target whose values span less than about 3e-3, with the
  option `noise_size` set. gpyreg 1.3.2 forms the predictive variances of a
  GP whose noise variance is below 1e-6 from a Cholesky factor, where they
  carried a rounding error that grows as the noise shrinks; a run reaches
  such a GP only with `tol_gp_noise` below 1e-3. It also refuses, with a
  message, inputs that it could not use. See the release notes of gpyreg
  for the rest of what changed there. `filelock`, `platformdirs` and
  `threadpoolctl` are new dependencies, used by the machine calibration.
  `pytest`, its plugins and `plotly` are no longer installed with PyVBMC: they
  are in the extras `test` and `examples`. `plotly` is used by example
  notebook 2 alone; run `pip install "pyvbmc[examples]"` before opening it.
  The test suite is no longer part of the wheel; run it from a checkout of
  the repository with `python -m pytest --reruns=5 -x`.
- **Results differ from 1.0.4, also with a fixed seed.** Several steps of the
  algorithm have been corrected, most of them to do what MATLAB VBMC does.
  Each correction changes the course of a run, so a script that fixes the
  seed gives different numbers than it gave with 1.0.4. Some entries under
  Fixed change results as well: the gradient of the soft-bound penalty, the
  conversion of `x0` and the bounds to double precision, the SciPy priors,
  which no longer consume a random draw when they are created, and the
  starting points beyond those of the initial design, which stay available
  to the acquisition search.
  - The initial design has `10 * ceil((D + 1) / 10)` points, where `D` is the
    number of variables: 10 points up to `D = 9`, 20 for `D` from 10 to 19,
    and so on. In 1.0.4 the number was `max(D, 10)`. The option is
    `fun_eval_start`.
  - The CMA-ES search that refines each new point before it is evaluated
    starts with a separate step size for each variable, and does not use the
    noise handling of the `cma` package. For a problem with one variable, a
    bounded one-dimensional search replaces the unbounded Nelder-Mead search
    of 1.0.4.
  - Warm-up ends by the same rule as in MATLAB VBMC. The test for a recent
    improvement of the target looks at fewer iterations, and it compares the
    running maximum of the lower confidence bound, recomputed with the
    current GP (option `recompute_lcb_max`, which had no effect in 1.0.4).
    After warm-up ends, a few iterations pass before the first input warp and
    before the run can stop for stability.
  - The initial variational posterior is centred on the starting point `x0`.
    For a problem with bounds it was centred somewhere else, because `x0` was
    read in the wrong coordinate system.
  - `min_iter` is the minimum number of iterations a run performs. In 1.0.4
    the minimum was in effect `min_iter + 1`.
  - A part of the candidate points of the acquisition search, a quarter by
    default (`box_search_frac`), is drawn uniformly in a box around the
    evaluated points, as in MATLAB VBMC. 1.0.4 drew these points from a normal
    distribution centred on a corner of the box, and most of them fell
    outside it.
  - The acquisition functions apply their variance regularization, which
    discourages points where the GP is already nearly certain. In 1.0.4 it
    never ran, because of a misspelt internal name, and the option
    `tol_gp_var` had no effect.
  - The sampling of GP hyperparameters stops once it no longer matters for
    the expected log joint (option `tol_gp_var_mcmc`). In 1.0.4 this rule
    could not be reached, and the option had no effect.
  - The slice sampler of the GP hyperparameters takes its step widths from a
    weighted covariance of the samples of recent iterations. 1.0.4 computed
    that covariance wrongly, and from a fifth of the samples: each iteration
    records the samples drawn before thinning, as MATLAB VBMC does, so an
    entry of `vbmc.iteration_history["gp_hyp_full"]` holds `gp_sample_thin`
    times as many rows as in 1.0.4. An iteration whose fit draws no samples
    records the one optimized vector, as in 1.0.4.
  - `vbmc.hyp_dict` no longer carries a `logp` entry. It held an array of
    zeros, the log prior densities the GP hyperparameter sampler
    reports rather than the log posterior MATLAB VBMC keeps there, and
    nothing read it. A run loaded from a file saved by 1.0.4 keeps the
    entry it was saved with.
  - The fit of the GP hyperparameters starts as in MATLAB VBMC. Its starting
    points include the hyperparameters of the GPs of the later half of the
    iterations; 1.0.4 left out the oldest of them whenever an even number of
    iterations had been recorded. The constant of the GP mean has a lower
    bound and the observation noise an upper bound, both recommended from
    the training set, where 1.0.4 left them unbounded, which also kept the
    starting points for that constant inside a narrow range.
  - For a target with inferred noise (`uncertainty_handling=True` without
    `specify_target_noise`), the GP models the noise as MATLAB VBMC does: a
    constant term plus the noise recorded for each point, scaled by a fitted
    factor, so that a point evaluated several times weighs more than one
    evaluated once. 1.0.4 fitted a single noise level for all points. Every
    run on such a target changes from its first GP fit, the model having one
    more hyperparameter; the weighting shows with repeated observations
    (`max_repeated_observations`), without which every point is recorded
    with the same noise.
  - Points with exactly equal values of the target, which a quantized
    log-likelihood can return, are ordered as in MATLAB VBMC, the earlier
    one first, when PyVBMC selects the points of highest density
    (`pyvbmc.stats.get_hpd`) and when it keeps the best points at the end of
    warm-up. So are iterations with equal scores when the best posterior of
    a run is selected, and candidates of the acquisition search with equal
    values when a search cache is kept (`search_cache_frac`), which
    candidates snapped to one point of an integer grid can have. 1.0.4
    ordered them arbitrarily. In that selection an
    iteration whose ELCBO or reliability index is NaN ranks last, and
    without the ranking criterion an iteration whose ELCBO is NaN is passed
    over; if the ELCBO of every candidate iteration is NaN, the last
    iteration considered is returned. 1.0.4 could return a posterior whose
    ELBO was NaN.
  - In a long run, from about 1380 evaluations on, the number of starting
    points tried in the fit of the GP hyperparameters goes down to zero, as
    in MATLAB VBMC, where 1.0.4 kept nine.
  - The variational optimization puts no soft bound on the mixture weights.
    Small weights are still penalized (`weight_penalty`) and removed
    (`tol_weight`). The bound of 1.0.4 came with a gradient that did not
    match it.
  - A balanced draw from a variational posterior whose components have
    unequal weights takes the samples that are left over, once every
    component has its whole share, in proportion to the fractional parts of
    those shares, as MATLAB VBMC does, so that a component contributes
    `w * N` draws on average. 1.0.4 drew them from other probabilities, and
    a component could count up to about one draw more or fewer on average.
    The candidates of every acquisition search and the Monte Carlo moments
    of the posterior come from such draws.
  - On a noisy target, the refit of the GP between the new points of an
    iteration counts the point just evaluated, as MATLAB VBMC does; 1.0.4
    counted the training set without it.
  - Smaller changes: the initial widths of the components in the variational
    optimization; the entropy of a posterior with a single component, which
    is computed exactly, also where the variational update between the new
    points of an iteration on a noisy target is compared with the posterior
    from before it; and which posterior a run returns when its last
    iteration is not stable, and which posterior an input warp starts from.
    That choice follows the options `rank_criterion`, `best_safe_sd` and
    `best_frac_back`, which 1.0.4 ignored. `VBMC.determine_best_vp()`,
    called without `safe_sd`, `frac_back` or `rank_criterion_flag`, uses
    the same options, so on a finished run it returns the iteration the run
    selected. In 1.0.4 these arguments defaulted to 5, 0.25 and no ranking
    criterion, and without the ranking criterion the method looked back one
    or two iterations fewer than MATLAB VBMC does.
  - Code rewritten for speed changes the last digits of intermediate results,
    and that is enough to change the course of a run.
- **The final boost is checked before it is accepted.** At the end of a run,
  PyVBMC fits the posterior once more with a larger number of components.
  The new posterior is kept only if it loses less than `tol_elcbo_boost` (0.1
  by default), both in the ELBO and in the ELBO minus five times its standard
  deviation. Otherwise PyVBMC warns and returns the posterior of the main
  loop. 1.0.4 kept the new posterior in every case, which on some runs
  replaced a good posterior with a much worse one; in our benchmark the check
  rejected the new posterior in 5 of 303 runs. With the check on, this last
  fit also runs without the penalty on small weights (`weight_penalty`).
  `tol_elcbo_boost=None` restores the behaviour of 1.0.4.
- **Option values are checked.** A wrong value raises an error that says what
  is expected. In 1.0.4 it could be accepted and then ignored or misread.
  - An option name that PyVBMC does not know raises an error wherever it is
    given: in the `options=` dictionary (as in 1.0.4), in an `options_path=`
    file, and in `VBMC.load(new_options=...)`. A misspelt name in an options
    file used to be ignored. The values given to `VBMC.load` are checked as
    those given at construction are.
  - `VBMC.load(file, new_options=...)` refuses a value of an option that
    PyVBMC reads only while it builds a `VBMC` object (`uncertainty_handling`,
    `specify_target_noise`, `gp_mean_fun`, `integer_vars`, `warmup`,
    `k_warmup`, `entropy_switch`, `active_search_bound` and their kin) that
    differs from the one the run stores, with a message that names the
    option and says to construct a new `VBMC` object: the saved run carries
    the state that was built from such an option, so a value given to `load`
    was stored and then ignored, leaving the options and the state in
    disagreement. The stored value, in any form construction reads alike
    (`1` for `True`, `"norminv"` for `"probit"`, the indices of the integer
    variables for their mask, and `uncertainty_handling=True` for one left
    empty beside `specify_target_noise=True`, since construction reads the
    two together), is taken and changes nothing. The options a continued
    run reads, `max_fun_evals` and `max_iter` among them, are taken as
    before, and `gp_mean_fun` and `integer_vars` are checked for a value no
    run can use whichever way they are supplied. A run saved by 1.0.4 whose
    stored `integer_vars`, `uncertainty_handling` or `specify_target_noise`
    is refused now or read otherwise than 1.0.4 read it (an `integer_vars`
    of zeros, an `uncertainty_handling` of `[1]`) loads with the values that
    state what the run was made with: the mask of its integer variables, and
    `True` or `False` for its noise handling.
  - `uncertainty_handling` takes `True` or `False` (`1` and `0` are accepted).
    Left empty, it follows `specify_target_noise`. A list such as `[1]`, which
    used to switch it on, raises an error, and so does `False` combined with
    `specify_target_noise=True`. `specify_target_noise` takes `True` or
    `False` as well (`1` and `0` are accepted); any other value was read by
    its truth, so that `"no"` or `[0]` turned the noise handling on. A script
    that sets only `specify_target_noise=True` needs no change.
  - `integer_vars` takes either a boolean mask with one entry per variable or
    the 0-based indices of the integer variables. In 1.0.4 a plain Python list
    made every variable an integer variable, without warning. An array of `D`
    integers that are all 0 or 1 could be a mask or a list of indices, and
    raises an error: write the mask with `True` and `False`.
  - `max_fun_evals` and `max_iter` must be positive integers, or `np.inf`
    for no limit, and `min_iter` a finite non-negative integer, 0 for a run
    without a minimum. A `max_iter` smaller than `min_iter` is raised to
    `min_iter`, with a warning.
  - `gp_mean_fun` takes `"zero"`, `"const"` or `"negquad"`, and
    `gp_hyp_sampler` takes `"slicesample"`: the mean functions and the
    sampler that PyVBMC implements. 1.0.4 accepted the other names of MATLAB
    VBMC and failed part-way through the run, after the first evaluations of
    the target.
  - `f_vals` cannot be combined with `specify_target_noise`, because it
    carries no noise for the values it supplies; 1.0.4 gave them a noise of
    1. Pass such observations through the `precomputed_evaluations` argument.
    An `f_vals` of NaN alone supplies no value and is accepted.
  - `AcqFcnVIQR` and `AcqFcnIMIQR` take a `quantile` strictly between 0.5 and
    1. 1.0.4 accepted any value; outside that range the values of VIQR were
    all NaN, or all the same, and the search took the first candidate it was
    offered, without a message.
  - An entry of `search_acq_fcn` given as a string, such as
    `"AcqFcnVIQR(quantile=0.9, loss='iqr_reduction')"`, is read as Python
    reads a call with literal arguments. 1.0.4 failed on a space around the
    name of a keyword and dropped a value that held an `=` without a message.
    A string that cannot be read raises an error that quotes it.
  - An acquisition function of your own that sets
    `acq_info["mcmc_importance_sampling"]` is refused when the `VBMC` object
    is created. The step it asks for, MATLAB VBMC's refinement of the
    importance samples by an ensemble sampler, is not ported: in 1.0.4 the
    flag either did nothing or, together with
    `variational_importance_sampling`, made the run fail part-way through.
    `search_acq_fcn` itself must be a list, of acquisition objects or of
    strings; a single acquisition outside a list raises an error that names
    the option.
  - `search_optimizer` takes `"cmaes"` or `"none"`. See Removed for the
    `"Nelder-Mead"` value.
  - `acq_hedge` must be `False`: the portfolio of acquisition functions it
    asks for (MATLAB VBMC's `acqhedge_vbmc.m`) is not ported. In 1.0.4 a
    run with the option ended in an `UnboundLocalError` at its first
    active-sampling step, after the initial design.
  - The five fractions that divide the candidates of the acquisition search
    among their sources (`search_cache_frac`, `heavy_tail_search_frac`,
    `mvn_search_frac`, `hpd_search_frac`, `box_search_frac`) must each lie
    in [0, 1] and add up to at most one; the default values leave a
    quarter of the search set, which `search_cache_frac` can take beside
    them. In 1.0.4 fractions that claimed more than the whole raised an
    error part-way through the run, and so could fractions that add up
    to one, since each share is rounded to a whole number of points.
    Each source takes its rounded share or what the sources before it
    left, whichever is smaller, and the variational posterior draws the
    rest. The descriptions of the five options give the range and the
    sum. `cache_frac`, the share of the search set that the starting
    cache gives, must lie in [0, 1] as well; it stands outside that
    sum. 1.0.4 took any value: a negative one took all but so many
    rows of the cache, and one above one could make the search set
    larger than asked.
  - `noise_shaping=True` raises an error, because noise shaping is not
    implemented.
  - The errors that refuse a value of `search_optimizer`, `acq_hedge`,
    `noise_shaping`, `warp_cov_reg`, `hpd_frac`, `noise_size`,
    `gp_sample_thin`, `active_importance_sampling_mcmc_samples`,
    `cache_frac`, the fractions, `min_iter` or `log_file_level` say how a
    saved run that carries it is loaded:
    `VBMC.load(file, new_options={...})`.
  - Setting an option that has no effect gives a warning that names the
    option, whether it is given at construction or to
    `VBMC.load(new_options=...)`. Most such options come from MATLAB VBMC
    and belong to features that PyVBMC does not have, `gp_int_mean_fun` and
    `proposal_fcn` among them; `nonlinear_scaling` and `search_cmaes_best`
    name behavior that PyVBMC always has. Their descriptions in the options
    files say which. Options given to `load` are listed with the user's
    options (`print(vbmc.options)`).
  - A `noise_size` given together with `specify_target_noise` has no effect,
    since the target returns the noise of each evaluation, and gives a
    warning that says so, at construction and in
    `VBMC.load(new_options=...)`, as MATLAB VBMC warns. 1.0.4 ignored it
    silently.
  - Once a `VBMC` object is constructed, its options cannot be removed (`del`,
    `pop`). Assigning to them was already an error.
  - The options of one run can be given to another, `VBMC(...,
    options=vbmc.options)`, without changing the first: 1.0.4 shared their
    set of user options, and building the second made the first list every
    option as set by the user.
- **Priors are checked.**
  - `VBMC` refuses a prior whose support does not cover the hard bounds,
    with a message that names the coordinates, the two intervals and the
    remedy: hard bounds inside the support of the prior. In 1.0.4 such a
    run stopped at its first evaluation outside the support with
    `FunctionLogger:InvalidFuncValue`, which named neither the prior nor
    the bounds. The hard bounds may coincide with the ends of the
    support, and the comparison allows a gap of a billionth of each
    coordinate's range, so that a support computed from the hard bounds
    themselves, as `uniform(loc=low, scale=high - low)`, passes; an
    infinite hard bound is compared exactly. With `integer_vars` the
    hard bounds are the half-integer ones, so a uniform prior over the
    values 0 to 10 is `UniformBox(-0.5, 10.5)`. Inside the hard bounds
    the prior is used as it is given:
    the model evidence is that of the prior restricted to the hard bounds,
    not of a prior truncated to them and normalized again.
  - `UniformBox`, `Trapezoidal`, `SplineTrapezoidal` and `SmoothBox` refuse
    a bound, a pivot or a scale that is NaN or infinite, and an array
    argument whose shape disagrees with the `D` given. Such arguments used
    to build a prior whose density is not a density, or one whose
    parameters sat on the wrong coordinates. `pyvbmc.priors.tile_inputs`,
    which the constructors share, refuses an array that is not a row, a
    column or a flat array of `size` values; 1.0.4 flattened a 2-by-2
    array given with `size=4`. A uniform prior needs finite
    bounds; for an unbounded parameter use `SmoothBox` or a `scipy.stats`
    distribution.
  - A one-dimensional `scipy.stats` distribution whose parameters are
    arrays, such as `norm(loc=[0, 10, 100])`, is refused as a prior. For
    several variables give a list of one-dimensional distributions or a
    multivariate one.
- **`ParameterTransformer` and `FunctionLogger` check what they are given.**
  `VBMC` builds both with arguments that pass; the checks concern code that
  uses the classes on its own.
  - `ParameterTransformer` refuses a variable with one finite bound, for
    which it has no transform. 1.0.4 treated such a variable as unbounded, so
    the inverse transform could return a value beyond the bound. `VBMC`
    refuses such bounds, as it did in 1.0.4.
  - `scale=` must hold one finite positive number per variable. 1.0.4 stored
    any value, and a zero or a negative one made the log-Jacobian infinite
    or NaN.
  - `FunctionLogger` refuses a `noise_flag` that contradicts
    `uncertainty_handling_level`: the flag goes with levels 1 and 2. 1.0.4
    accepted the pair, and with the flag at level 0 it recorded an SD of 1
    for the values of a noiseless target.
  - At uncertainty handling level 2, where the target provides its noise,
    `FunctionLogger.add` needs the SD of the value it is given. 1.0.4
    recorded an SD of 1.
- **Defaults for noisy targets.** A noisy target gets its own defaults: a
  larger budget of evaluations, a larger stability count, updates of the GP
  and of the posterior within active sampling, and the VIQR acquisition
  function. They apply whenever noise handling is on. In 1.0.4 they applied
  only when `specify_target_noise` was set in the `options=` dictionary: not
  with `uncertainty_handling=True`, and not when the setting came from an
  options file. An option set in an options file counts as the user's choice,
  and these defaults do not replace it.
- **The `results` dictionary.** `results["iterations"]` is the number of
  iterations performed. In 1.0.4 it was the index of the last iteration, which
  is one less. `results["best_iter"]` remains a 0-based index.
  `results["problem_type"]` is `"bounded"` for a problem with bounds; in 1.0.4
  it was `"unconstrained"` for every problem.
- **The recorded random state.** `results["rng_state"]` holds the state of
  `vbmc.rng` at the end of the run; in 1.0.4 it held the string `"rng"`.
  `vbmc.random_state` and the `random_state` entries of the iteration history
  have the same form; in 1.0.4 they held NumPy's global state.
  `VBMC.load(..., set_random_state=True)` restores `vbmc.rng`, and for a file
  saved by this version it leaves NumPy's global state alone. A file saved by
  an earlier version holds the global state only: that state is restored as
  before, the generator is derived from it, and a warning says that the
  continued run will not repeat the original one exactly.
- `vp.pdf(x, grad_flag=True)` raises `NotImplementedError` in the original
  parameter space (`orig_flag=True`, the default): gradients exist in the
  transformed space only. 1.0.4 raised for the log-density and returned a
  wrong gradient for the density.
- **The iteration history holds the GPs without their factors, and a run uses
  less memory.** An entry of `vbmc.iteration_history["gp"]` holds the training
  data and the hyperparameters of the GP of that iteration, without the
  posterior factors, whose size grows with the square of the number of
  evaluations. Such an entry raises an error when it is asked to predict:
  call `vbmc.get_gp(iteration)`, which returns the complete GP of a recorded
  iteration with the factors recomputed. The importance samples of the noisy
  acquisition functions are left out of the history as well, unless the
  option `record_full_history_details` is set. On a noisy problem with 5
  variables, the history shrank from 117 MB to 4.6 MB and the peak memory of
  the process from 427 MB to 273 MB. A file saved by 1.0.4 holds complete
  GPs, and `vbmc.get_gp` returns them unchanged.
- An acquisition function of your own must return one value per input point,
  with shape `(M,)`, `(1, M)` or `(M, 1)`; another shape raises `ValueError`.
- `VBMC.determine_best_vp()` returns a copy of the selected posterior, no
  longer the object stored in the iteration history. `VBMC.final_boost()`
  leaves the state of the run unchanged.
- The iteration display ends with a `finalize` line whenever the returned
  posterior is not that of the last iteration, including when it is the
  posterior of an earlier iteration.
- **Runs are faster.** The acquisition function is evaluated for a whole
  CMA-ES generation in one call, and `vp.pdf`, the expected log joint and the
  Monte Carlo entropy work on whole arrays where they looped over mixture
  components and GP hyperparameter samples. The prediction and the
  hyperparameter sampling of gpyreg are faster as well. With a noisy target,
  the first observation at a new point updates the GP by a rank-one step;
  1.0.4 recomputed the GP in full, with the same result up to rounding. On
  our noiseless benchmark problems with 4 to 15 variables, a run took about
  two to three times less time than before (283 → about 100 seconds on a
  problem with 10 variables, for example), and about 20 per cent less on
  noisy targets. These whole-run timings come from one machine and were
  taken before the corrections listed above, which change how long a run
  takes. The VIQR acquisition function, the default for noisy targets, is 1.1
  to 1.4 times faster per call on large candidate sets.
- `vp.mode(orig_flag=False)` ranks its candidate starting points by the
  density alone, where 1.0.4 also computed the gradient and discarded it,
  with the same result; on a posterior with 5 variables and 20 components
  it took 0.40 seconds instead of 0.69, about 1.7 times faster.

### Fixed

- **Saved posteriors work across Python versions.** For a problem with bounds,
  a posterior saved under one minor version of Python and loaded under another
  (3.11 and 3.12, say) crashed the interpreter, with no Python error, as soon
  as it was sampled or evaluated, because the file stored functions as
  bytecode. Files written by earlier versions of PyVBMC work too. You can run
  VBMC on one machine and analyze the posteriors, or stack them with S-VBMC,
  on another. A saved *run* (`VBMC.save`) still contains the target function:
  under another Python version it can be loaded and inspected, but should not
  be continued or saved again there.
- **Long runs.** With NumPy 2.4 or later, a run that went past
  `stable_gp_sampling` evaluations of the target (`200 + 10 * D` by default)
  raised `ValueError: setting an array element with a sequence`. From that
  point PyVBMC works with a single set of GP hyperparameters, and one variance
  term kept a dimension too many, which earlier versions of NumPy accepted
  with a deprecation warning.
- **Integer variables.** A run with `integer_vars` and a search optimizer,
  the default one included, raised `IndexError` right after its initial
  design: the point that the local search returned could not be snapped to
  the integer grid. Such runs complete. A coordinate exactly halfway between
  two integers is rounded away from zero, as in MATLAB VBMC. The feature is
  experimental, and the FAQ and the description of the option say what it
  does: the points of the active-sampling search are snapped to the integer
  grid; the initial design and a provided `x0` are not, as in MATLAB VBMC;
  and on a grid the search can return a point that has been evaluated
  already, which on a noiseless target spends an evaluation and adds
  nothing. A starting point with a provided value that the grid, or the
  search box, moves is evaluated where it lands; 1.0.4 recorded its value
  at the moved point.
- With `AcqFcnIMIQR`, each chain of the importance sampler starts from a
  sample drawn in proportion to its importance weight, as in MATLAB VBMC. The
  draw was uniform. Runs with `AcqFcnIMIQR` give different results.
- With `AcqFcnIMIQR`, a function given for
  `active_importance_sampling_mcmc_samples` sets the length of the MCMC
  chains of the importance sampler, as it sets the number of samples drawn
  from the posterior with `AcqFcnVIQR`. So does a count given as a
  floating-point number (`100.0`, or `50.5`, which is rounded up to 51, as
  with `AcqFcnVIQR`). Both raised `TypeError` in the first step of active
  sampling. The function receives `K`, `n_vars` and `D` as
  keyword arguments, and the description of the option says what the
  number is under each acquisition. A value that is neither a finite number
  nor a function is refused at construction and by `VBMC.load`, and a
  function that does not return a finite number raises an error at the
  first importance sampling, each with a message that names the option;
  1.0.4 read `True` as 1 with `AcqFcnVIQR`, and NaN as 0 with
  `AcqFcnIMIQR`, which leaves out the MCMC step.
- The penalty that keeps the width of the mixture components within its soft
  bounds has the right gradient. In 1.0.4 the entries of that gradient were in
  the wrong order, so the variational optimization followed a wrong gradient
  whenever a width left its bounds.
- Calling `optimize()` again on a finished run continues from copies of the
  last recorded iteration. 1.0.4 continued from the history entries
  themselves, so the record of that iteration changed as the run went on, and
  `VBMC.load(iteration=...)` could restore a wrong state.
- A run built with `ns_gp_max=0`, saved, and loaded with a positive
  `ns_gp_max` (`VBMC.load(file, new_options=...)`) samples the GP
  hyperparameters when it is continued. 1.0.4 stored the new value and went
  on fitting them by optimization alone. A run whose sampling has stopped in
  the stable regime stays there.
- `x0` and the bounds are converted to double precision whatever floating-point
  type they come in. 1.0.4 converted integer inputs only, and kept `float32`
  or `float16` values in the state of the run and in the parameter transform.
- With at least `fun_eval_start` starting points, the initial design is the
  first `fun_eval_start` of them. When they took one value in a coordinate,
  the run stopped at its first GP fit with `KeyError: (-inf, -inf)`, after
  the evaluations of the design. `VBMC` refuses such starting points at
  construction, with a message that names the coordinates and says that
  fewer points leave the rest of the design to the plausible box.
- The first GP fit of a run completes when the points of highest density in
  the initial design (the fraction `hpd_frac` of it, 80% by default) share
  one value in a coordinate, as when the best 8 of 10 starting points do.
  Such a run stopped at that fit with `ValueError: The widths vector needs
  to be all positive real numbers`, after the evaluations of the design, or,
  with `ns_gp_max=0`, gave a GP with non-finite hyperparameters. In such a
  coordinate the starting length scale of the GP, the lower bound of that
  length scale and the starting scale of the GP mean come from the whole
  training set. Later fits whose points of highest density share a value,
  which is common with `integer_vars`, take that lower bound from the whole
  training set as well, so their results differ from 1.0.4. Such fits print
  no `RuntimeWarning: divide by zero encountered in log`.
- A bound given as a single number applies to every variable, as documented.
  It raised an error for problems with more than one variable. Without `x0`
  the number of variables comes from the plausible bounds, so one of them
  needs an entry per variable; two single numbers raise an error that says
  so.
- `x0` can be given as a list or a number, as the bounds can: a list of
  numbers is one starting point, a list of lists holds several, and a number
  is the starting point of a problem of one variable. 1.0.4 raised an error
  for each of them.
- The title of the final plot (`plot=True`, `create_vbmc_animation`) gave one
  iteration fewer than the run had performed.
- After `VBMC.load`, the run, its posterior and its function logger share one
  parameter transformer, that of the loaded iteration. For a run that had
  warped its input space, `vbmc.parameter_transformer` was the transformer of
  a different iteration.
- `print(vbmc)` shows the Gaussian process, the prior and the log-density,
  which always printed as `None`, and shows the starting point in original
  coordinates, which was wrong after an input warp. The starting points in
  original coordinates are available as `vbmc.x0_orig`.
- `entropy_switch=True` raised `TypeError` in the first iteration of any
  problem with five or more variables.
- When SciPy's optimizer did not converge in a variational optimization with
  the deterministic entropy (a posterior with one component, or
  `entropy_switch=True`), the run stopped with `RuntimeError`. It now goes on
  from the optimizer's last iterate and logs a warning.
- The variational optimization passes over NaN. The best iterate of a
  stochastic optimization is taken among the objective values that are not
  NaN, as in MATLAB VBMC, and a candidate solution whose ELBO is NaN is not
  selected; if the ELBO of every candidate is NaN, `optimize_vp` raises
  `ValueError`. 1.0.4 took the first NaN it met in both places and could go
  on with a posterior whose ELBO was NaN.
- A function given as an option of one argument (`ns_ent`, `k_fun_max`,
  `adaptive_k` and the like) receives the argument by keyword when it takes
  an argument of the name PyVBMC passes (`K`, and `N` for `k_fun_max`), as
  in 1.0.4, and by position otherwise, as MATLAB VBMC passes it, so the
  parameter of a function of one positional parameter may have any name.
  1.0.4 passed the argument by keyword alone, and as `unkn` for
  `adaptive_k`, so a function whose parameter had another name was accepted
  at construction and raised `TypeError` at its first use in the run.
- `variable_means=False` raised an error in the final boost of any run that
  ended with fewer than `min_final_components` components. With that
  setting, `vbmc.final_boost(vp, gp)` needs a `gp` with at least as many
  training inputs as `vp` has components, and says so otherwise.
- With `variable_means=False`, the variational fit that follows an input warp,
  and decides whether the warp is kept, tries a number of candidate
  posteriors (`ns_elbo`) in proportion to the number of components it
  optimizes, one per training input, as MATLAB VBMC does. 1.0.4 sized it by
  the number of components of the posterior from before the warp.
- A noisy target with `max_fun_evals=np.inf` raised `OverflowError`.
- A `tol_stable_warmup` no larger than `fun_evals_per_iter` raised an error in
  the third iteration.
- With `log_file_name` set, a handler on the `"VBMC"` logger that was not a
  file handler raised `AttributeError`, and duplicate file handlers could be
  left behind.
- With `log_file_name` set, the log file is written at the level that
  `log_file_level` gives: `"off"`, `"iter"`, `"full"`, or any level of the
  `logging` module, that is any non-negative integer, custom levels and
  `logging.NOTSET` (0) included. 1.0.4 refused a level other than 0, 10, 20,
  30, 40 and 50, and wrote no log file for 0, `None` or `False`. Any other
  value raises an error at construction and in `VBMC.load`, whether or not a
  log file is named.
- Active sampling:
  - With `search_cache_frac` above 0 (it is 0 by default), the first step of
    active sampling raised an error, because the cache of search points is
    empty until a step has run.
  - When the local search of the acquisition function fails, the run goes on
    with the best candidate found before it. The failure used to end the run.
  - When `x0` holds more points than the initial design uses, the others stay
    available as candidates for later evaluations. One that is acquired
    where it lies is recorded with its value from `f_vals`, without a call
    to the target, also after an input warp, and a starting point that has
    been evaluated is not proposed again. 1.0.4 discarded the surplus
    points.
  - A number given for `ns_elbo` or `ns_ent_fine_active`, in place of a
    function of the number of components, raised `TypeError` in the
    variational update between the new points of an iteration, which a
    noisy target runs by default.
  - A run with `warmup=False` takes the updates of the GP and of the
    posterior between the new points of an iteration
    (`active_sample_gp_update`, `active_sample_vp_update`, on by default for
    a noisy target) in its first `active_sample_full_update_past_warmup`
    iterations, as MATLAB VBMC does, also when it was saved by 1.0.4 and is
    continued. 1.0.4 took them for one iteration more.
- A run whose `max_fun_evals` equals the size of its initial design raised an
  error in its first GP fit; with the initial design of
  `10 * ceil((D + 1) / 10)` points, `max_fun_evals=20` does this for `D` from
  10 to 19. The fit now starts without the space-filling design of the GP
  hyperparameters, as in MATLAB VBMC.
- An `hpd_frac` that leaves fewer than two points of the initial design, below
  0.15 of ten points, made the first GP fit fail with an error about an
  empty array or zero widths; such a value is refused at construction, with
  a message that names the option, and so is one outside (0, 1].
- `noise_size` takes a positive finite number of any numeric type, or an
  empty value (`[]`, an empty array or `None`) that leaves it unset. For a
  target that does not return its own noise (`specify_target_noise`), the
  only one whose GP fit reads the option, `None` raised `TypeError` at the
  first GP fit; an empty array or a NumPy number raised `ValueError` there
  with recent versions of NumPy, and older ones ignored a NumPy number;
  zero or a negative value was replaced by `tol_gp_noise` without a word.
  Any other value is refused at construction and by `VBMC.load`, with a
  message that names the option, as MATLAB VBMC refuses a value that is not
  positive.
- `gp_sample_thin` takes a whole number greater than zero, of an integer or
  a floating-point type (`5.0` is read as 5). Any other value is refused at
  construction and by `VBMC.load`, whatever `ns_gp_max`, with a message
  that names the option. 1.0.4 raised an error at the first GP fit that
  sampled the hyperparameters, after the evaluations of the initial design,
  for a value such as 0, 2.5 or 5.0, ran a boolean as its integer, and took
  any value with `ns_gp_max=0`, which fits the hyperparameters without
  sampling them.
- After a second or later input warp, the bounds of the acquisition search
  could be mapped through the transform of an earlier iteration. We have not
  seen this happen in a run.
- The input warp keeps the covariance of the posterior as it is when setting
  its weak correlations to zero would leave a matrix that is not positive
  definite, which could leave the warped posterior far from unit variance
  along one direction. We have not seen this happen in a run.
- After an input warp, `function_logger.X` and `function_logger.y` are
  expressed in the new coordinates for every recorded evaluation. The points
  set aside at the end of warm-up kept the coordinates of the space the run
  had left.
- With `gp_mean_fun="zero"`, a run with two or more variables stopped with
  `ValueError` ("Unsupported GP mean function for input warping.") at its
  first input warp. The warp re-expresses the length scales of a zero-mean
  GP as it does for the other mean functions, and the GP fit that follows
  the warp adapts the rest.
- `VBMC.load(file, iteration=k)` at an iteration that kept an input warp,
  and `VBMC.load(file)` of a run that stopped on one, restore that
  iteration's GP hyperparameters, from which a resumed run starts its GP
  fit. The record of such an iteration held the hyperparameters of the
  iteration before, from before the warp; files saved by earlier versions
  keep that record.
- `upper_gp_length_factor` caps the length scales of the GP, as documented.
  In 1.0.4 it had no effect.
- With `weighted_hyp_cov=False`, the running covariance of the GP
  hyperparameters is cleared at the end of warm-up and takes in only fits
  that drew samples. 1.0.4 did neither.
- `vp.stats["J_sjk"]`, the covariance between the expected log joints of the
  mixture components, has the right shape after the variational optimization
  removed a component. In 1.0.4 the component was removed along one of the
  array's two component axes only.
- `vp.eta`, the unbounded parameters of the mixture weights, agrees with
  `vp.w` on a returned posterior. In 1.0.4 `set_parameters` wrote `w` alone,
  and a returned posterior could hold an `eta` left over from other weights.
- The options `true_mean` and `true_cov`, which record the divergence from a
  known posterior in the iteration history, work and leave the run as it is.
  In 1.0.4 NumPy arrays raised `ValueError`, and lists made every iteration
  draw a million samples from the random stream of the run, which changed its
  result.
- `ParameterTransformer`: `rotation_matrix=` must be an orthogonal matrix,
  which the density of the transform assumes; 1.0.4 accepted any matrix.
  Comparing two transformers with `==` compares their scales, which 1.0.4
  compared with themselves. The log-Jacobian of the `logit` transform stays
  finite far into the lower tail, where 1.0.4 returned `-inf`.
- `pyvbmc.priors.SciPy` recognizes SciPy distributions through SciPy's public
  interface; 1.0.4 imported private SciPy classes, which a SciPy release may
  move. Creating a multivariate prior no longer draws a random sample to find
  its dimension.
- `pyvbmc.stats.kl_div_mvn` accepts its means and covariances as 1-D or 2-D
  arrays, by position or by keyword. In 1.0.4 a call that gave all four as
  keywords failed. It works from the log determinants of the two
  covariances, and with it the symmetrized KL divergence between the
  posteriors of successive iterations, which a run reports and tests for
  stability. The determinants themselves leave the range of a double for a
  problem of twenty parameters whose posterior standard deviations are,
  in geometric mean, below about 8e-9 or above about 5e7 (2e-11 and 2e10
  for fifteen parameters); 1.0.4 then reported an
  infinite divergence, so that the run was never stable, or zero, the value
  of a posterior that has stopped moving.
- **Methods of the variational posterior.**
  - `vp.kl_div(samples=..., gauss_flag=True)` compares the posterior with
    the mean of the samples in each coordinate. 1.0.4 used one average over
    the whole sample matrix for every coordinate.
  - `vp.kl_div(gauss_flag=False)` sets aside a density that is infinite or
    NaN, as it sets aside a zero one, and returns a number where 1.0.4
    returned NaN. `vp.pdf` in the original space keeps a density near the
    top of the range of a double, which 1.0.4 reported as infinite.
  - `vp.pdf` and `vp.log_pdf` evaluate the density at the point they are
    given when its coordinates are integers. In the original space 1.0.4
    truncated the transformed coordinates to the integer type and reported
    the density of another point. A float32 point is transformed in
    double precision, which moves its density in the last digits of
    single precision.
  - `vp.log_pdf`, and `vp.pdf` with `log_flag=True`, are finite far in the
    tails of the posterior, where the density is too small for a double:
    there the log density and its gradient are computed over the
    components in log space. 1.0.4 returned `-inf` with a NaN gradient, and
    lost precision just before. `vp.mode()` therefore refines its starting
    point on a narrow posterior (a standard deviation below about 0.01 in
    the units of the parameters), where 1.0.4 returned the starting point
    unrefined, with a NumPy `RuntimeWarning`.
  - `vp.pdf` and `vp.log_pdf` raise `ValueError` when the rows of `x` do
    not have `D` coordinates. In the transformed space (`orig_flag=False`)
    1.0.4 returned values for points it had misread: for a posterior of one
    parameter, a flat array of several points was read as a single point
    and gave one number, and a column of points given to a posterior of
    more parameters gave one number per row.
  - `vp.mode()` works for a posterior of one parameter, where it raised an
    `AxisError`. It draws its starting points from a copy of the random
    generator, so a call leaves the random stream of a run where it was and
    two calls return the same mode. `vp.mode(n_opts=...)` runs a search of
    its own: it does not return the mode stored by an earlier call, as
    it did in 1.0.4, and does not replace it, so `vp.mode()` keeps
    answering with the result of the default search.
    `vp.get_parameters()` discards a stored mode, which its normalization
    of the weights may have moved, and gives a zero weight the raw
    parameter minus infinity without NumPy's warning of a division by zero.
  - `vp.set_parameters` rescales `sigma` and `lambd` against each other
    only when both are optimized; a scale whose `optimize_sigma` or
    `optimize_lambd` flag is off keeps its value. 1.0.4 rescaled whatever
    the flags, so that with `sigma` fixed and `lambd` optimized the
    variational optimization rescaled the fixed `sigma` at every evaluation
    of its objective.
  - `vp.set_parameters(theta, raw_flag=False)` requires the entries that
    hold `sigma`, `lambd` and the weights to be positive, and those alone.
    1.0.4 checked other entries: a negative scale could pass, and a
    negative component mean could be refused.
  - `vp.moments(cov_flag=True)` returns a 1-by-1 covariance matrix for a
    posterior of one parameter, where 1.0.4 returned a scalar array.
    `vp.sample`, and through it `vp.moments`, `vp.kl_div` and `vp.mtv`,
    accept a number of samples given as any scalar that holds a whole
    number, a float such as `1e5` included, and refuse another count
    with a message that says so. In 1.0.4 `vp.sample` failed on `1e5`,
    and `vp.moments` and `vp.kl_div(gauss_flag=True)` truncated a
    fractional count without a word. The
    component indices that `vp.sample` returns are a flat array of
    integers in every case. A negative `df`, which `vp.pdf` reads as a
    product of univariate `t` densities, is refused by `vp.sample` with a
    message that says so.
  - `VariationalPosterior(D, K, x0)` accepts a single starting point given
    as a column.
- **Priors.**
  - `support()` of a `SciPy` prior built from a shifted or scaled
    one-dimensional `scipy.stats` distribution, and the bounds of a
    `Product` that holds one, give the interval the distribution lives on.
    1.0.4 gave the interval of its standard form: `uniform(loc=2, scale=3)`
    reported `[0, 1]`. The interval is read from the distribution at every
    call, so a prior loaded from a file that 1.0.4 wrote reports it too;
    `prior.a` and `prior.b` of a `SciPy` or `Product` prior are read-only
    arrays, which a script can no longer assign to.
  - The density of a prior is computed in double precision whatever the
    type of the point. In 1.0.4 an array of integers truncated the log
    density, and gave a trapezoidal or spline-trapezoidal prior a density
    of one, or an infinite one, at points outside its support; a float32
    array gave a float32 result.
  - A point with a NaN coordinate has density zero under `UniformBox`, as
    under the other box priors. 1.0.4 gave it the full density.
  - A list of one-dimensional priors that holds a `UserFunction` works as a
    prior, whether the function returns a float or an array of one
    element. In 1.0.4 the first evaluation of the target raised a
    `TypeError`.
  - `Trapezoidal` no longer warns of a division by zero at a point on its
    lower bound, and neither trapezoid leaves NumPy's floating-point error
    settings changed when an error interrupts it.
  - The documentation of the priors gives the shapes that the code uses,
    `convert_to_prior` documents `log_prior`, and the description of the
    `log_prior` argument of `VBMC` is complete.
- A quantity exactly halfway between two integers is rounded away from
  zero, as in MATLAB VBMC, wherever PyVBMC sizes something from a fraction:
  the high-posterior-density subset (`pyvbmc.stats.get_hpd(X, y, 0.1)` on 5
  points returns 1 point, where 1.0.4 returned none), the shares of the
  acquisition search among its sources, the number of GP hyperparameter
  samples, the number of starting points of the GP hyperparameter
  search, and the bonus of mixture components
  (`adaptive_k`). A run at the default options that starts from no more
  points than its initial design takes has no such tie; a run with
  another `hpd_frac`, with `hpd_search_frac` above zero or with more
  starting points than the initial design takes can.
- `FunctionLogger.finalize()` trims the count of evaluations per point along
  with the other arrays.
- `FunctionLogger`: recording a repeated evaluation whose duration is unknown
  leaves the stored average duration of that point as it is, and a known
  duration takes the place of an unknown average; in 1.0.4 one unknown
  duration left the point's average at NaN. The value returned for a
  repeated point is a float, as for a new one, where 1.0.4 returned a
  one-element array.
- `ParameterTransformer` keeps copies of the bound, scale and rotation arrays
  it is given, so changing one of them afterwards does not change the
  transform. Built with a rotation or a scale and with plausible bounds, it
  centres the plausible box as a transformer without them does. Its
  documentation names the keyword for the bounded transform,
  `transform_type`.
- `pyvbmc.whitening.unscent_warp` works in floating point whatever the type
  of the points it is given, where an array of integers was truncated, and
  accepts a single point with several scales, which raised an error.
- Example 1 gave −2.272 as the true log evidence of its target; the value is
  −2.2598.
- `warp_cov_reg` takes a finite number of any numeric type, NumPy's
  included, or a function of `N`, the number of points logged so far; the
  function may return its number in an array of one element. Any other
  value (a boolean, NaN, a string) is refused at construction and by
  `VBMC.load`, and a function's result that is not a finite number is
  refused at the warp, with a message that names the option. 1.0.4 took
  only a Python `int` or `float`, read `True` as 1, and failed at the first
  warp on other values.
- Option descriptions are printed in full (`print(options)`,
  `repr(options)`). Descriptions containing `:` or `=` were cut short, and
  an option of the advanced set that the user had given showed `None` for
  its description.

### Removed

- The separate search GP (`separate_search_gp=True`). It never worked: a run
  with it failed in its second iteration. The option is still accepted, with a
  warning, and has no effect.
- The `"Nelder-Mead"` value of `search_optimizer`, with the search it named.
  That search ignored the search bounds and `search_max_fun_evals`, and on
  an unbounded variable it could return a point far outside the search box.
  1.0.4 used it for every problem of one variable; with `"cmaes"` such a
  problem is searched by a bounded one-dimensional method in place of
  CMA-ES, and with `"none"` it has no local search, like any other. For
  more variables only a script that set the option reached it. A file saved by
  1.0.4 for a problem of one variable holds the value and loads as before.
  A saved run of more variables that set it is refused by `VBMC.load`, which
  says how to continue it: `new_options={"search_optimizer": "cmaes"}`.

[Unreleased]: https://github.com/acerbilab/pyvbmc/compare/v1.0.4...HEAD
