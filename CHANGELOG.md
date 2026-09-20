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
  `uncertainty_handling=[1]` or an ambiguous `integer_vars` raises an error.
  So do a `gp_mean_fun` or a `gp_hyp_sampler` that PyVBMC does not implement,
  and `f_vals` together with `specify_target_noise`.
- With `uncertainty_handling=True`, or a noisy setting in an options file, the
  defaults for noisy targets apply, a larger budget of evaluations among
  them.
- `results["iterations"]` is the number of iterations, one more than in 1.0.4.
- An entry of `vbmc.iteration_history["gp"]` cannot make predictions; call
  `vbmc.get_gp(iteration)`.
- `vp.pdf(x, grad_flag=True)` raises an error in the original parameter space.
- `results["rng_state"]` and `vbmc.random_state` hold the state of `vbmc.rng`.
- Classes of your own: a prior needs `sample(self, n, rng=None)` to be part of
  a `Product` prior, and an acquisition function must return one value per
  input point.
- `ParameterTransformer` and `FunctionLogger` used on their own: a variable
  with one finite bound, a `scale` that is not positive, a noise flag that
  contradicts the uncertainty handling level, and `add` without an SD for a
  target that provides its noise raise an error.

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
  as function evaluations of the run. `VBMC(..., initialization_cost=k)`
  charges `k` evaluations against `max_fun_evals` for work done before the
  run. `results` reports the first in `precomputed_observations` and
  `precomputed_locations`, and the second in `evaluation_budget`; each key is
  present only when its argument was used.
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
  a pooled observation). In 1.0.4 the option changed the iteration display
  and nothing else. Its default, 0, is unchanged.
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
  SciPy 1.15 or later and gpyreg 1.2.1 or later; see the release notes of
  gpyreg for what changed there. `filelock`, `platformdirs` and
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
  conversion of `x0` and the bounds to double precision, and the SciPy
  priors, which no longer consume a random draw when they are created.
  - The initial design has `10 * ceil((D + 1) / 10)` points, where `D` is the
    number of variables: 10 points up to `D = 9`, 20 for `D` from 10 to 19,
    and so on. In 1.0.4 the number was `max(D, 10)`. The option is
    `fun_eval_start`.
  - The CMA-ES search that refines each new point before it is evaluated
    starts with a separate step size for each variable, and does not use the
    noise handling of the `cma` package. For a problem with one variable, a
    bounded one-dimensional search replaces CMA-ES.
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
    records the samples drawn before thinning, as MATLAB VBMC does, so
    `vbmc.iteration_history["gp_hyp_full"]` holds `gp_sample_thin` times as
    many rows as in 1.0.4.
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
    evaluated once. 1.0.4 fitted a single noise level for all points. The
    difference shows with repeated observations
    (`max_repeated_observations`).
  - Points with exactly equal values of the target, which a quantized
    log-likelihood can return, are ordered as in MATLAB VBMC, the earlier
    one first, when PyVBMC selects the points of highest density
    (`pyvbmc.stats.get_hpd`). 1.0.4 ordered them arbitrarily.
  - In a run of more than 1000 evaluations, the number of starting points
    tried in the fit of the GP hyperparameters goes down to zero, as in
    MATLAB VBMC, where 1.0.4 kept nine.
  - The variational optimization puts no soft bound on the mixture weights.
    Small weights are still penalized (`weight_penalty`) and removed
    (`tol_weight`). The bound of 1.0.4 came with a gradient that did not
    match it.
  - Smaller changes: the initial widths of the components in the variational
    optimization; the entropy of a posterior with a single component, which
    is computed exactly; and which posterior is returned when no iteration
    was stable. That choice follows the options `rank_criterion`,
    `best_safe_sd` and `best_frac_back`, which 1.0.4 ignored.
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
    file used to be ignored.
  - `uncertainty_handling` takes `True` or `False` (`1` and `0` are accepted).
    Left empty, it follows `specify_target_noise`. A list such as `[1]`, which
    used to switch it on, raises an error, and so does `False` combined with
    `specify_target_noise=True`. A script that sets only
    `specify_target_noise` needs no change.
  - `integer_vars` takes either a boolean mask with one entry per variable or
    the 0-based indices of the integer variables. In 1.0.4 a plain Python list
    made every variable an integer variable, without warning. An array of `D`
    integers that are all 0 or 1 could be a mask or a list of indices, and
    raises an error: write the mask with `True` and `False`.
  - `max_fun_evals` and `max_iter` must be positive integers (`np.inf` is
    allowed). A `max_iter` smaller than `min_iter` is raised to `min_iter`,
    with a warning.
  - `gp_mean_fun` takes `"zero"`, `"const"` or `"negquad"`, and
    `gp_hyp_sampler` takes `"slicesample"`: the mean functions and the
    sampler that PyVBMC implements. 1.0.4 accepted the other names of MATLAB
    VBMC and failed part-way through the run, after the first evaluations of
    the target.
  - `f_vals` cannot be combined with `specify_target_noise`, because it
    carries no noise for the values it supplies; 1.0.4 gave them a noise of
    1. Pass such observations through the `precomputed_evaluations` argument.
  - Setting an option that has no effect gives a warning that names the
    option. Such options come from MATLAB VBMC and belong to features that
    PyVBMC does not have; their descriptions in the options files say so.
    `noise_shaping=True` raises an error, because noise shaping is not
    implemented.
  - Once a `VBMC` object is constructed, its options cannot be removed (`del`,
    `pop`). Assigning to them was already an error.
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
- The penalty that keeps the width of the mixture components within its soft
  bounds has the right gradient. In 1.0.4 the entries of that gradient were in
  the wrong order, so the variational optimization followed a wrong gradient
  whenever a width left its bounds.
- Calling `optimize()` again on a finished run continues from copies of the
  last recorded iteration. 1.0.4 continued from the history entries
  themselves, so the record of that iteration changed as the run went on, and
  `VBMC.load(iteration=...)` could restore a wrong state.
- `x0` and the bounds are converted to double precision whatever floating-point
  type they come in. 1.0.4 converted integer inputs only, and kept `float32`
  or `float16` values in the state of the run and in the parameter transform.
- A bound given as a single number applies to every variable, as documented.
  It raised an error for problems with more than one variable.
- After `VBMC.load`, the run, its posterior and its function logger share one
  parameter transformer, that of the loaded iteration. For a run that had
  warped its input space, `vbmc.parameter_transformer` was the transformer of
  a different iteration.
- `print(vbmc)` shows the Gaussian process, the prior and the log-density,
  which always printed as `None`, and shows the starting point in original
  coordinates, which was wrong after an input warp.
- `entropy_switch=True` raised `TypeError` in the first iteration of any
  problem with five or more variables.
- `variable_means=False` raised an error in the final boost of any run that
  ended with fewer than `min_final_components` components.
- A noisy target with `max_fun_evals=np.inf` raised `OverflowError`.
- A `tol_stable_warmup` no larger than `fun_evals_per_iter` raised an error in
  the third iteration.
- With `log_file_name` set, a handler on the `"VBMC"` logger that was not a
  file handler raised `AttributeError`, and duplicate file handlers could be
  left behind.
- Active sampling:
  - With `search_cache_frac` above 0 (it is 0 by default), the first step of
    active sampling raised an error, because the cache of search points is
    empty until a step has run.
  - When the local search of the acquisition function fails, the run goes on
    with the best candidate found before it. The failure used to end the run.
  - When `x0` holds more points than the initial design uses, the others stay
    available as candidates for later evaluations, and a starting point that
    has been evaluated is not proposed again. 1.0.4 discarded the surplus
    points.
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
  keywords failed.
- `FunctionLogger.finalize()` trims the count of evaluations per point along
  with the other arrays.
- `FunctionLogger`: recording a repeated evaluation whose duration is unknown
  leaves the stored average duration of that point as it is, where 1.0.4
  replaced it with NaN; and the value returned for a repeated point is a
  float, as for a new one, where 1.0.4 returned a one-element array.
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
- `warp_cov_reg` accepts any number, or a callable that receives the number
  of training points. Only a Python `int` or `float` worked.
- Option descriptions are printed in full (`print(options)`,
  `repr(options)`). Descriptions containing `:` or `=` were cut short.

### Removed

- The separate search GP (`separate_search_gp=True`). It never worked: a run
  with it failed in its second iteration. The option is still accepted, with a
  warning, and has no effect.

[Unreleased]: https://github.com/acerbilab/pyvbmc/compare/v1.0.4...HEAD
