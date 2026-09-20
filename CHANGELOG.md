# Changelog

Changes to PyVBMC that a user can notice, newest first, in the layout of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). An entry says what
changed and, where a script may need it, what to do. The records behind the
entries are under `dev/` (start from `dev/README.md`).

## [Unreleased]: PyVBMC 1.5

Compared with 1.0.4. The section is assembled as the work lands: it holds the
changes that came out of the correctness review of the port of MATLAB VBMC
([plan and worklog](dev/plans/port-correctness-review.md)), waves 0 to 2. The
new features of 1.5 are still to be listed.

### Changed

- **A seeded run does not reproduce its 1.0.4 numbers.** Several steps of the
  algorithm follow MATLAB VBMC again where the port had drifted from it, and
  each of them changes the trajectory of a run:
  - The initial design holds `10 * ceil((D + 1) / 10)` points, which is 20
    from ten dimensions on (it was `max(D, 10)`).
  - The local search of the acquisition function (CMA-ES) starts with one
    step size per coordinate and runs without `cma`'s noise handling; with a
    single variable a bounded scalar search takes its place.
  - Warm-up ends as it does in MATLAB: the recent-improvement test looks at
    the intended window, the running maximum of the lower confidence bound
    is recomputed with the current GP (the option `recompute_lcb_max`, which
    had no effect), and the first input warp and the earliest stable
    termination wait for a few iterations after warm-up.
  - The initial variational posterior sits at the starting point. For a
    bounded problem it sat somewhere else, because the starting point was
    used in original coordinates where transformed ones belong.
  - `min_iter` counts iterations: a run on which it binds performs
    `min_iter` of them, where it performed one more.
  - Smaller ones: the starting widths of the variational optimization, the
    exact entropy of a one-component posterior, and the choice of the
    returned posterior when no iteration is stable, which the options
    `rank_criterion`, `best_safe_sd` and `best_frac_back` govern (they were
    ignored).
- **Options are checked, and a wrong value fails with a message** instead of
  being accepted and ignored or misread.
  - An option name that PyVBMC does not declare raises wherever it is
    supplied: the `options=` dictionary (as before), an `options_path=` file
    and `VBMC.load(new_options=...)`. A misspelt name in a file used to be
    ignored.
  - `uncertainty_handling` is `True` or `False` (1 and 0 are accepted; leave
    it empty to follow `specify_target_noise`). A list such as `[1]`, which
    used to switch it on, raises, and so does turning it off while
    `specify_target_noise` is set. Most scripts set `specify_target_noise`
    alone and are unaffected.
  - `integer_vars` is a boolean array with one entry per variable, or an
    array of 0-based indices. A plain list used to mark every variable as an
    integer, silently. An integer array of length `D` holding only zeros and
    ones could be either and raises: write the mask with booleans.
  - `max_fun_evals` and `max_iter` must be positive integers (infinity is
    allowed), and a `max_iter` below `min_iter` is raised to it with a
    warning.
  - Setting an option that has no effect warns and names the option. These
    are options inherited from MATLAB VBMC whose features were never ported;
    their descriptions in the options file say so. `noise_shaping=True`
    raises, because noise shaping is not implemented.
  - The options of an initialized run can no more be removed (`pop`, `del`)
    than they can be set.
- **Noisy targets.** The defaults for a noisy target (a larger evaluation
  budget and stability count, GP and posterior updates within active
  sampling, the VIQR acquisition) also apply with
  `uncertainty_handling=True`, where they followed `specify_target_noise`
  alone, and when the noisy setting comes from an options file. An option set
  in an options file counts as set by the user and is not replaced by those
  defaults.
- **`results`.** `results["iterations"]` is the number of iterations the run
  performed; it held the index of the last one, which is one less.
  `results["best_iter"]` is still a 0-based index. `results["problem_type"]`
  says `"bounded"` for a problem with bounds; it always said
  `"unconstrained"`.
- A bound given as a single number holds for every variable, as the
  documentation says. It raised for problems of more than one dimension.
- `VBMC.determine_best_vp()` returns a copy of the selected posterior, not the
  entry of the iteration history, and `VBMC.final_boost()` leaves the state
  of the run as it found it.
- The display closes with a "finalize" line whenever the returned posterior is
  not the one of the last iteration, also when it comes from an earlier
  iteration.

### Fixed

- **A saved variational posterior can be used under another version of
  Python.** For a problem with bounds, sampling from a posterior saved under
  another minor version of Python, or evaluating it, ended the interpreter
  without a Python error, because the file held functions as bytecode. Files
  written by earlier versions of PyVBMC are covered too. This is what makes
  it possible to run VBMC on one machine and to analyze the posteriors, or
  to stack them with S-VBMC, on another. A saved *run* (`VBMC.save`) still
  holds the target function: under another Python version it can be loaded
  and inspected, and should not be continued or saved again there.
- After `VBMC.load`, the run, its posterior and its function logger share one
  parameter transformer, that of the loaded iteration; for a run that had
  warped its input space, `vbmc.parameter_transformer` was the map of another
  iteration.
- `print(vbmc)` shows the starting point in original coordinates, the Gaussian
  process, the prior and the log-density; the last three always printed
  `None`, and the starting point was wrong after an input warp.
- `entropy_switch=True` raised `TypeError` in the first iteration of any
  problem of five or more dimensions.
- `variable_means=False` raised in the final boost of any run that ended with
  fewer than `min_final_components` components.
- A noisy target with `max_fun_evals=np.inf` raised `OverflowError`.
- A `tol_stable_warmup` no larger than `fun_evals_per_iter` raised in the
  third iteration.
- With `log_file_name` set, a handler of another kind on the `"VBMC"` logger
  raised `AttributeError`, and duplicate file handlers could survive.
- Active sampling: the first noisy observation at a point updates the GP by a
  rank-one step; an empty search cache no longer raises; a local search that
  fails keeps the best candidate found so far; starting points beyond the
  initial design stay available, and one that has been evaluated is not
  proposed again.
- After a second or later input warp, the bounds of the acquisition search
  could be mapped through the transform of an earlier iteration (no stored
  run shows it happening).
- `warp_cov_reg` accepts a number or a callable.
- `print(options)` shows option descriptions in full; those holding a `:` or
  an `=` were cut short.

### Removed

- The separate search GP (`separate_search_gp=True`) never worked: the run
  failed in its second iteration. The option is still accepted and has no
  effect, with a warning.
