# Changelog

All notable changes to PyVBMC are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Changes since PyVBMC 1.0.4, to be released as PyVBMC 1.5.

Many of the changes and fixes come from a systematic comparison of PyVBMC with
VBMC, the original MATLAB implementation. The
[porting log](pyvbmc/vbmc/README.md) lists where PyVBMC deliberately differs
from MATLAB VBMC, and the
[ledger of the comparison](dev/results/2026-09-23-port-correctness-review.md)
lists its findings and what was done about each.

### Upgrading from 1.0.4

What can stop an existing script, or change what it returns. Each point has
its entry below.

- PyVBMC needs Python 3.10 or later, SciPy 1.15 or later and gpyreg 1.4.0 or
  later, and no longer installs `pytest` and `plotly`.

A script can get other values, without an error:

- Results differ from 1.0.4, also with a fixed seed.
- `integer_vars` given as a list of indices marks those variables, where
  1.0.4 marked every variable.
- With `uncertainty_handling=True`, or a noisy setting in an options file, the
  defaults for noisy targets apply, a larger budget of evaluations among them.
- `results["iterations"]` counts the iterations, one more than 1.0.4 gave, and
  `results["problem_type"]` is `"bounded"` for a problem with bounds.
  `results["rng_state"]` and `vbmc.random_state` hold the state of `vbmc.rng`.
- `vbmc.determine_best_vp()` without arguments follows the run's options, so it
  can select another iteration, and it returns a copy of the posterior.
  `vbmc.final_boost(vp, gp)` leaves the state of the run unchanged.
- Methods of the variational posterior and of the priors, and functions of
  `pyvbmc.stats`, return other values in some calls: `vp.log_pdf` far in the
  tails, `vp.mode()`, `vp.kl_div`, `vp.moments` for one parameter, `vp.pdf` at
  integer points, `support()` of a shifted or scaled SciPy prior, `get_hpd`
  and `kl_div_mvn`, among them. A change to the array that `a` or `b` of a
  `SciPy` or `Product` prior returns does not reach the prior.
- A function given for `ns_elbo` or `ns_ent_fine_active` that takes `K` after
  another parameter receives the number of components in `K` in every call.
- With `log_file_name` set, `log_file_level=0` writes the log file, where
  1.0.4 wrote none.
- An entry of `vbmc.iteration_history["gp_hyp_full"]` for a fit that samples
  holds `gp_sample_thin` times as many rows (five by default), and the
  `vbmc.hyp_dict` of a run started by this release has no `logp` entry.
- Used on their own, `FunctionLogger` returns a float for a repeated point,
  `unscent_warp` computes in floating point for integer points, and a
  `ParameterTransformer` given a rotation or a scale with plausible bounds
  centres the plausible box as one without them does.

An error is raised for calls and values that 1.0.4 accepted:

- Option names and values are checked: an option name PyVBMC does not know,
  also in an options file or in `VBMC.load(new_options=...)`, and a value it
  cannot use raise an error that says what is expected.
  `search_optimizer="Nelder-Mead"`, `acq_hedge=True`, `noise_shaping=True`,
  `uncertainty_handling=[1]` and an `integer_vars` of `D` integers that are
  all 0 or 1, which could be a mask or indices, are among them. A file saved by
  1.0.4 that holds `uncertainty_handling=[1]`, or `"Nelder-Mead"` for a
  problem of one variable, loads with the value restated.
- An option given as a function of the number of components receives it by
  the keyword `K` or by position, so one that takes `unkn` by keyword alone,
  as 1.0.4 passed it to `adaptive_k`, raises an error when it is called.
- `VBMC.load(file, new_options=...)` refuses a changed value of an option that
  only construction reads, and options cannot be removed from a constructed
  `VBMC` object.
- An entry of `vbmc.iteration_history["gp"]` cannot make predictions; call
  `vbmc.get_gp(iteration)`.
- `vp.pdf(x, grad_flag=True)` raises in the original parameter space, and
  `vp.pdf` and `vp.log_pdf` raise for an `x` whose rows do not have `D`
  coordinates (for one parameter, pass several points as a column).
- A few methods and constructors refuse arguments that 1.0.4 misread: a
  number of samples that is not whole, a negative scale or weight in
  `vp.set_parameters(theta, raw_flag=False)`, and a `quantile` of
  `AcqFcnVIQR` or `AcqFcnIMIQR` outside (0.5, 1), among them.
- Priors are checked: `VBMC` refuses a prior whose support does not cover the
  hard bounds, the box and trapezoidal priors refuse a parameter that is NaN,
  infinite or of the wrong shape, a one-dimensional `scipy.stats`
  distribution with array parameters is refused, and `a` and `b` of a
  `SciPy` or `Product` prior cannot be assigned.
- At least `fun_eval_start` starting points that take one value in a
  coordinate across the first `fun_eval_start` of them are refused.
- A variational optimization in which every candidate has a NaN ELBO raises
  `ValueError`.
- Classes of your own: a prior needs `sample(self, n, rng=None)` to be part of
  a `Product` prior, and an acquisition function must return one value per
  input point and must not set `acq_info["mcmc_importance_sampling"]`.
- `ParameterTransformer` and `FunctionLogger`, used on their own, refuse
  inconsistent arguments.

### Added

- **Seeded runs.** `VBMC(..., seed=...)` fixes every random draw of a run. The
  generator is `vbmc.rng`, shared by the posterior as `vp.rng`, and a run
  neither reads nor writes NumPy's global random state once the `VBMC` object
  exists; with `seed=None`, the default, the generator is derived from that
  state, so `np.random.seed(...)` before creating the object still fixes a
  run. `seed=` does not reach a target that draws random numbers of its own,
  such as a simulator. See the reproducibility section of the quickstart.
- **S-VBMC: stack the posteriors of several runs.** `pyvbmc.SVBMC` combines the
  variational posteriors of several runs on the same problem into one
  posterior by optimizing the weights of their components, with no further
  evaluations of the target (Silvestrin, Li & Acerbi, 2025). Use it when
  repeated runs disagree, or when each captures a part of a multimodal
  posterior.
  - It needs `pip install "pyvbmc[torch]"`. A stack saved with `stacked.save`
    loads with `SVBMC.load` under another Python version and without Torch.
  - When any of the runs it keeps had a noisy target, the expected log joint
    that enters `stacked.elbo` is capped at its median over the components,
    since weights chosen on noisy estimates make the plain value optimistic;
    `stacked.elbo_details` holds every estimate.
  - It replaces the standalone `svbmc` package (0.1.1), with the same method
    and optimizer of the weights: on our benchmark the two reach the same
    weights, and the optimization ran 1.9 to 4 times faster (a provisional
    figure from one machine). `elbo` is a number. The `SVBMC` page of the
    documentation lists the differences; Example 7 shows a complete use.
- **PyMC models as targets.** `PyMCTarget` wraps a PyMC model so that PyVBMC
  can fit it: it provides the log joint and the bounds, and finds a starting
  point and the plausible box with a budget of model evaluations that the run
  reuses (`setup_budget=`). `target.to_arviz(vp)` exports draws under the
  names and shapes of the model's variables. It needs
  `pip install "pyvbmc[pymc]"`, PyMC 6.3 or later and Python 3.12 or later; a
  model it cannot handle, such as one with a discrete variable, is refused
  with `pyvbmc.pymc.UnsupportedModel`. See the `PyMCTarget` page of the
  documentation and Example 8.
- **Vectorized targets.** With `options={"vectorized_target": True}`, the
  target takes an `(N, D)` array and returns one value per row (and one noise
  SD per row with `specify_target_noise`), and PyVBMC evaluates the whole
  initial design in one call. Later evaluations are one point at a time.
- **Posterior exports.** `vp.to_torch()` returns the posterior as a
  `torch.distributions` object in the original parameter space, with a
  differentiable `log_prob` (`pyvbmc[torch]`, Torch 2.7 or later).
  `vp.to_arviz()` returns draws as an ArviZ `DataTree` (`pyvbmc[arviz]`,
  Python 3.12 or later).
- **Evaluations made before the run.**
  `VBMC(..., precomputed_evaluations=(X, y))` (`(X, y, y_sd)` with
  `specify_target_noise`) gives a run target values computed beforehand; they
  do not count as evaluations of the run, and with a separate prior `y` holds
  log-likelihood values. `initialization_cost=k` charges `k` evaluations
  against `max_fun_evals` for work done before the run. `results` counts the
  evaluations in `precomputed_observations` and their distinct points in
  `precomputed_locations`, and reports the charge in `evaluation_budget`.
- **Machine calibration (optional).** `pyvbmc.calibrate()` measures, in tens
  of seconds, the block sizes that suit your machine for the posterior density
  and the Monte Carlo entropy, and saves them for later runs. It changes
  speed, not results. Unless the display is off, a run prints one line saying
  which settings it uses, and while none are saved the first run of a session
  suggests calibrating. `options={"performance_calibration": "off"}` keeps the
  standard settings, and `results["performance_calibration"]` records what a
  run used.
- **Tips.** A run may print a short tip before the iteration display, each tip
  at most once per Python session. `options={"show_tips": False}`, or
  `display="off"`, turns them off; `SVBMC(..., show_tips=False)` does the same
  for S-VBMC.
- **Update reminders.** In an interactive session, a new run of a release more
  than a year old starts with a note that a newer version may exist, at most
  three times for each installed version; the note goes by the release date
  shipped with PyVBMC, without a network request, and
  `options={"show_tips": False}` turns it off with the tips.
  `pyvbmc.check_for_updates()` asks PyPI whether a newer version exists and
  gives the command that installs it.
- **Repeated observations of a noisy target.** With `max_repeated_observations`
  above 0, active sampling may evaluate a noisy target again at a point it has
  already evaluated, and pools the observations into one. In 1.0.4 the option
  changed only the iteration display; its default, 0, is unchanged.
- **Smaller additions.** The option `ns_gp_max_active` caps the GP
  hyperparameter samples of the fits made between the evaluations of one
  iteration on a noisy target, `AcqFcnVIQR(loss="iqr_reduction")` scores the
  reduction of the integrated interquantile range, and `vbmc.x0_orig` holds
  the starting points in the original coordinates. The first two change
  nothing at their defaults.
- **Documentation.** A FAQ page; Example 7 (S-VBMC), Example 8 (a PyMC model)
  and Example 9 (Torch and JAX targets, and the posterior exports); quickstart
  sections on reproducible runs, Torch and JAX targets, vectorized targets,
  PyMC models and the use of a fitted posterior. Every example shows the
  output of this release. `skills/pyvbmc/SKILL.md` guides a coding agent
  through the documentation.

### Changed

- **Requirements.** PyVBMC needs Python 3.10 or later (1.0.4 accepted 3.9),
  SciPy 1.15 or later and gpyreg 1.4.0 or later. From 1.3.0 on, gpyreg takes
  the bounds of the GP mean function and the starting length scales per input
  dimension, which moves the results of every run; see its release notes.
  `filelock`, `platformdirs` and `threadpoolctl` are new dependencies.
  `pytest` and `plotly` are in the extras `test` and `examples` (Example 2
  needs `pip install "pyvbmc[examples]"`), and the test suite is no longer
  part of the wheel. PyVBMC 1.0.4 fails at the first GP fit of a noisy target
  without `specify_target_noise` when it gets gpyreg 1.3.0 or later.
- **Results differ from 1.0.4, also with a fixed seed.** Several steps of the
  algorithm have been corrected, most of them to do what MATLAB VBMC does, and
  each changes the course of a run. The main ones:
  - The initial design has `10 * ceil((D + 1) / 10)` points
    (`fun_eval_start`), where 1.0.4 had `max(D, 10)`.
  - The initial variational posterior is centred on `x0`, which 1.0.4 read in
    the wrong coordinates for a problem with bounds.
  - Warm-up ends by MATLAB VBMC's rule (`recompute_lcb_max` takes effect), and
    a few iterations pass after it before the first input warp and before the
    run can stop for stability. `min_iter` is the minimum number of
    iterations, where 1.0.4's minimum was `min_iter + 1`.
  - The CMA-ES search of each new point starts with a step size per variable
    and without the noise handling of `cma`; a problem with one variable gets
    a bounded one-dimensional search in place of Nelder-Mead.
  - The box share of the acquisition search's candidates (`box_search_frac`) is
    drawn uniformly in the box, where 1.0.4 drew most of them outside it.
    Starting points beyond the initial design stay available to the search.
  - The GP hyperparameter fit starts as in MATLAB VBMC and bounds the constant
    of the mean from below and the noise from above, and in a long run its
    number of starting points goes down to zero from about 1380 evaluations
    on, where 1.0.4 kept nine. Its slice sampler takes its step widths from a
    correctly computed covariance of recent samples.
  - For a target with inferred noise, the GP adds to a constant noise the
    noise recorded for each point, scaled by a fitted factor, where 1.0.4
    fitted a single noise level. A run saved by 1.0.4 on such a target
    continues with this model.
  - Options that had no effect in 1.0.4 take effect: the variance
    regularization of the acquisition functions (`tol_gp_var`), the stopping
    rule of the hyperparameter sampling (`tol_gp_var_mcmc`) and the cap on the
    GP length scales (`upper_gp_length_factor`).
  - The gradient of the soft bounds on the component widths is correct, and
    the mixture weights have no soft bound. With `AcqFcnIMIQR`, each chain of
    the importance sampler starts from a sample drawn by its importance weight.
  - On a noisy target, the refit of the GP between the new points of an
    iteration counts the point just evaluated.
  - A balanced draw from the posterior gives each component `w * N` draws on
    average. Ties between equal values are broken as in MATLAB VBMC, the
    earlier first, and a half is rounded away from zero wherever PyVBMC sizes
    something from a fraction (`pyvbmc.stats.get_hpd` among them).
  - The posterior a run returns when its last iteration is not stable, and the
    posterior an input warp starts from, are chosen by `rank_criterion`,
    `best_safe_sd` and `best_frac_back`, which 1.0.4 ignored; an iteration
    whose ELCBO is NaN ranks last by ELCBO, and without the ranking criterion
    it is passed over unless every one's is.
  - Smaller corrections: the initial widths of the components in the
    variational optimization; the exact entropy of a posterior with one
    component; `x0` and the bounds converted to double precision whatever
    their type; no random draw when a multivariate SciPy prior is created.
  - Under options that are not the default: with `variable_means=False`, the
    fit after an input warp tries a number of candidate posteriors in
    proportion to the components it optimizes; with `warmup=False`, the
    updates within active sampling last one iteration fewer; with
    `weighted_hyp_cov=False`, the covariance of the GP hyperparameters
    restarts at the end of warm-up and takes in only fits that sample.
  - Code rewritten for speed changes the last digits of intermediate results,
    which is enough to change the course of a run.
- **The final boost is checked before it is accepted.** The posterior refit
  with more components at the end of a run is kept only if it loses less than
  `tol_elcbo_boost` (0.1 by default) in the ELBO and in the ELBO minus five
  times its SD; otherwise PyVBMC warns and returns the posterior of the main
  loop. 1.0.4 kept it in every case, which on some runs replaced a good
  posterior with a much worse one. The checked refit runs without the penalty
  on small weights; `tol_elcbo_boost=None` restores 1.0.4's boost, penalty
  included.
- **Options are checked.** An option name PyVBMC does not know, or a value it
  cannot use, raises an error that says what is expected, at construction and
  in `VBMC.load(new_options=...)`, and a value also when `VBMC.load` finds it
  in a saved run. 1.0.4 could ignore such an option, misread it, or fail on it
  part-way through a run. The options page of the documentation, and
  `repr(vbmc.options)`, give what each option accepts.
  - `integer_vars` takes a boolean mask or the 0-based indices of the integer
    variables, where 1.0.4 made every variable an integer variable when given
    a plain list; a list or array of `D` integers that are all 0 or 1 is
    ambiguous and refused (write the mask with `True` and `False`).
  - `uncertainty_handling` and `specify_target_noise` take `True` or `False`
    (or 1 and 0). 1.0.4 took other values: a list such as `[1]` switched the
    noise handling on, and `specify_target_noise` was read by its truth.
  - `max_fun_evals` and `max_iter` are positive integers or `np.inf`, and
    `min_iter` a finite non-negative integer; a `max_iter` below `min_iter` is
    raised to it, with a warning.
  - `log_file_level` takes `"off"`, `"iter"`, `"full"` or a level of the
    `logging` module, and is checked also when no log file is named; with one
    named, `0` writes it.
  - An option given as a function of one argument (`ns_ent`, `k_fun_max`,
    `adaptive_k` and the like) receives it by keyword when it takes an
    argument of the name PyVBMC passes (`K`, or `N` for `k_fun_max`), and by
    position otherwise, so the name of a single parameter does not matter.
    1.0.4 passed it to `adaptive_k` as `unkn`, and by position to `ns_elbo`
    and `ns_ent_fine_active` in some calls, so that one taking `K` after
    another parameter received the number in that other parameter.
  - Settings that PyVBMC does not implement are refused: a `gp_mean_fun` other
    than `"zero"`, `"const"` and `"negquad"`, a `gp_hyp_sampler` other than
    `"slicesample"`, `acq_hedge=True`, `noise_shaping=True`, and an
    acquisition function that sets `acq_info["mcmc_importance_sampling"]`.
  - `f_vals` cannot be combined with `specify_target_noise`; pass such values
    through `precomputed_evaluations`.
  - `AcqFcnVIQR` and `AcqFcnIMIQR` take a `quantile` strictly between 0.5 and
    1; outside that range the values of 1.0.4's VIQR were NaN or all equal.
  - Values that PyVBMC can use and 1.0.4 failed on part-way through a run are
    accepted, such as `None` for `noise_size`, `5.0` for `gp_sample_thin`, a
    function or a float count for `active_importance_sampling_mcmc_samples`
    with `AcqFcnIMIQR`, and a `search_acq_fcn` string with spaces around a
    keyword.
  - An option that has no effect in PyVBMC gives a warning, as does
    `noise_size` together with `specify_target_noise`.
  - `VBMC.load(file, new_options=...)` refuses a changed value of an option
    that only construction reads (`uncertainty_handling`, `gp_mean_fun`,
    `integer_vars`, `warmup` and their kin), which 1.0.4 stored and ignored;
    construct a new `VBMC` object to run with one. Where a file saved by 1.0.4
    holds a value in a form 1.0.4 wrote (`uncertainty_handling=[1]`, an
    `integer_vars` of zeros, `search_optimizer="Nelder-Mead"` for one
    variable), `VBMC.load` restates it, so the run loads with the values it
    was run with.
  - The options of a constructed `VBMC` object cannot be removed, as they
    could not be assigned. The options of one run can be given to another
    (`options=vbmc.options`) without changing the first.
- **Priors are checked.** `VBMC` refuses a prior whose support does not cover
  the hard bounds, with a message that names the coordinates; 1.0.4 stopped
  at the first evaluation outside the support. The box and trapezoidal priors
  refuse a NaN or infinite parameter and an array whose shape disagrees with
  `D`, and a one-dimensional `scipy.stats` distribution with array parameters
  is refused: give a list of distributions or a multivariate one.
- **Defaults for noisy targets** (a larger budget of evaluations, a larger
  stability count, updates of the GP and of the posterior within active
  sampling, and the VIQR acquisition function) apply whenever noise handling
  is on. In 1.0.4 they applied only when `specify_target_noise` was set in the
  `options=` dictionary. An option set in an options file counts as the user's
  choice.
- **The `results` dictionary.** `results["iterations"]` is the number of
  iterations performed, one more than 1.0.4's index of the last one, and
  `results["problem_type"]` is `"bounded"` for a problem with bounds, which
  1.0.4 called `"unconstrained"`. `results["rng_state"]`, `vbmc.random_state`
  and the history's `random_state` entries hold the state of `vbmc.rng`;
  `VBMC.load(..., set_random_state=True)` restores it, and for a file saved by
  1.0.4 derives the generator from the saved global state, with a warning that
  the continued run will not repeat the original.
- **Smaller run histories.** An entry of `vbmc.iteration_history["gp"]` holds
  the GP without its posterior factors and cannot make predictions;
  `vbmc.get_gp(iteration)` returns the complete GP. The importance samples of
  the noisy acquisition functions are left out unless
  `record_full_history_details` is set. On a noisy problem with 5 variables
  the history shrank from 117 MB to 4.6 MB. An entry of
  `iteration_history["gp_hyp_full"]` holds the hyperparameter samples from
  before thinning, and the `vbmc.hyp_dict` of a run started by this release
  has no `logp` entry, which held zeros.
  A run saved by 1.0.4 holds complete GPs, which `vbmc.get_gp` returns as
  they are.
- `vbmc.determine_best_vp()`, called without arguments, uses the run's
  `rank_criterion`, `best_safe_sd` and `best_frac_back`, so on a finished run
  it returns the iteration the run selected, and it returns a copy of the
  selected posterior. `vbmc.final_boost(vp, gp)` leaves the state of the run
  unchanged.
- **Classes of your own.** An acquisition function must return one value per
  input point, with shape `(M,)`, `(1, M)` or `(M, 1)`, and a prior class
  needs `sample(self, n, rng=None)` to be part of a `Product` prior.
- `ParameterTransformer` and `FunctionLogger`, used on their own, refuse
  inconsistent arguments (a variable with one finite bound, a scale that is
  not positive, a rotation that is not orthogonal, a noise flag or a missing
  SD that contradicts the uncertainty handling level); `VBMC` builds them with
  arguments that pass. `FunctionLogger` returns a float for a repeated point,
  `unscent_warp` computes in floating point for integer points, and a
  `ParameterTransformer` given a rotation or a scale with plausible bounds
  centres the plausible box as one without them does.
- **Display.** With `display="off"`, a run prints warnings only; why it ended
  and its ELBO are in `results["message"]` and `results["elbo"]`. The
  iteration display ends with a `finalize` line whenever the returned
  posterior is not that of the last iteration.
- **Runs are faster.** On our noiseless benchmark problems with 4 to 15
  variables a run took two to three times less time (283 → about 100 seconds
  with 10 variables), and about 20 per cent less on noisy targets. These
  timings come from one machine and were taken before the corrections above,
  which change how long a run takes.

### Fixed

- **Runs that stopped with an error complete:**
  - long runs with NumPy 2.4 or later, past `stable_gp_sampling` evaluations
    (`ValueError: setting an array element with a sequence`);
  - runs with `integer_vars` (`IndexError` after the initial design); the
    feature is experimental, and the FAQ says what it does;
  - `entropy_switch=True` with five or more variables;
  - a variational optimization whose SciPy optimizer does not converge, which
    goes on from the optimizer's last iterate with a warning;
  - a `max_fun_evals` equal to the size of the initial design, a
    `tol_stable_warmup` no larger than `fun_evals_per_iter`, a
    `search_cache_frac` above 0, and `max_fun_evals=np.inf` on a noisy target;
  - a failed local search of the acquisition function, after which the run
    goes on with the best candidate found before it;
  - a number given for `ns_elbo` or `ns_ent_fine_active` on a noisy target;
  - the final boost with `variable_means=False` of a run that ended with fewer
    than `min_final_components` components;
  - `gp_mean_fun="zero"` with two or more variables, at the first input warp;
  - a first GP fit in which the best points of the initial design share a
    value in a coordinate, which 1.0.4 fails with gpyreg 1.3.0 or later;
  - `log_file_name` with a handler on the `"VBMC"` logger that is not a file
    handler;
  - `create_vbmc_animation` with current NumPy.
- **Inputs.** A bound given as a single number applies to every variable, as
  documented. `x0` can be given as a list or a number, as the bounds can. A
  target with `specify_target_noise` may return its noise SD as an array of
  one element. A list of one-dimensional priors may hold a `UserFunction`
  made with `D=1`. `VBMC` refuses at least `fun_eval_start` starting points
  that take one value in a coordinate across the first `fun_eval_start` of
  them, on which 1.0.4 stopped at its first GP fit, after evaluating them.
- The variational optimization passes over NaN values of its objective and
  candidates whose ELBO is NaN, and raises `ValueError` if every candidate's
  ELBO is NaN; 1.0.4 could go on with a posterior whose ELBO was NaN.
- **Saving, loading and continuing.**
  - A posterior of a problem with bounds, saved under one Python version and
    loaded under another, crashed the interpreter when it was sampled or
    evaluated; files written by earlier versions work too. A saved run
    (`VBMC.save`) still holds Python bytecode: under another Python version it
    can be loaded and inspected, but should not be continued or saved again.
  - Calling `optimize()` again on a finished run no longer changes the record
    of its last iteration, which could make `VBMC.load(iteration=...)` restore
    a wrong state.
  - After `VBMC.load`, the run, its posterior and its function logger share
    the transformer of the loaded iteration, and an iteration that kept an
    input warp, in a file saved by this release, is restored with its own GP
    hyperparameters.
  - A run built with `ns_gp_max=0` and loaded with a positive `ns_gp_max`
    samples the GP hyperparameters when it is continued.
- **The variational posterior.**
  - `vp.log_pdf` is finite far in the tails, where 1.0.4 returned `-inf`, and
    `vp.mode()` refines its starting point on a narrow posterior.
  - `vp.pdf` and `vp.log_pdf` evaluate a point with integer coordinates where
    it is, which 1.0.4 truncated in the original space, and raise
    `ValueError` when the rows of `x` do not have `D` coordinates, where 1.0.4
    could misread the points. `vp.pdf(x, grad_flag=True)` raises
    `NotImplementedError` in the original space, where 1.0.4 returned a wrong
    gradient.
  - `vp.mode()` works for a posterior of one parameter, leaves the random
    stream of the run where it was, and with `n_opts` runs a search of its
    own.
  - `vp.kl_div(samples=...)` uses the mean of each coordinate, and
    `vp.kl_div(gauss_flag=False)` returns a number where 1.0.4 returned NaN.
  - `vp.moments` of a posterior of one parameter returns a 1-by-1 covariance.
    `vp.sample` accepts a count such as `1e5` and returns its component
    indices as a flat array of integers; it, `vp.moments` and `vp.kl_div`
    refuse a count that is not whole.
  - `vp.set_parameters` keeps `sigma` or `lambd` fixed when its optimize flag
    is off, and with `raw_flag=False` refuses a negative scale or weight and
    accepts a negative mean.
  - `vp.stats["J_sjk"]` keeps its shape when the variational optimization
    removes a component, which 1.0.4 removed along one of its two component
    axes only.
- **Priors.** `support()` of a SciPy prior built from a shifted or scaled
  distribution gives the interval it lives on (1.0.4 gave `[0, 1]` for
  `uniform(loc=2, scale=3)`). `a` and `b` of a `SciPy` or `Product` prior are
  read-only and recomputed from the distribution at each access, so a change
  to the array one of them returns does not reach the prior. Densities are
  computed in double precision whatever the type of the point; in 1.0.4
  integer points could give a trapezoidal prior a density of one outside its
  support. `UniformBox` gives a point with a NaN coordinate density zero.
- `pyvbmc.stats.kl_div_mvn`, and with it the `sKL` a run tests for stability,
  is finite for covariances whose determinants leave the range of a double.
  1.0.4 gave infinity, so that the run never became stable, or zero.
- The options `true_mean` and `true_cov` work and leave the run as it is; in
  1.0.4 arrays raised an error and lists changed the result of the run.
- PyVBMC leaves the configuration of logging to your program. Creating a
  `VBMC` object configured Python's root logger, which sent the log messages of
  other libraries to standard output. PyVBMC's messages still appear there
  when your program configures no logging.

### Removed

- The separate search GP (`separate_search_gp=True`), which never worked. The
  option is accepted, with a warning, and has no effect.
- The `"Nelder-Mead"` value of `search_optimizer`. Its search ignored the
  search bounds; a problem of one variable, for which 1.0.4 used it, gets a
  bounded one-dimensional search. `VBMC.load` refuses a saved run of more
  variables that set it and says how to continue it.

[Unreleased]: https://github.com/acerbilab/pyvbmc/compare/v1.0.4...HEAD
