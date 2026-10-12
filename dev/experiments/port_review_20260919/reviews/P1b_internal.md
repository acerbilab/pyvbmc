# P1b — setup, options, defaults, bounds, history, save/load, internal-correctness track

Raw reviewer report, wave 2 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent,
fresh context, read-only on the repositories; it did not read the MATLAB
source. Its check scripts ran in the session scratchpad and are not
retained. The text below is the agent's final message, unedited.

---

# Slice P1b — internal correctness review

Repository `C:\Users\luigi\Documents\GitHub\pyvbmc`, branch `dev-port-review` at `51451dc`.
Track: internal correctness (no MATLAB source read).

(Note: the harness refused the `REPORT.md` write in my scratch directory — "Subagents should return findings as text, not write report files" — so the report exists only here, complete and unedited.)

## 1. Coverage

### Read completely, line by line

- `pyvbmc/vbmc/vbmc.py`: class docstring (`:62-226`), `__init__` (`:228-492`), `target`
  property, `_normalized_hard_bound`, `_resolve_pymc_target`,
  `_validate_initialization_cost`, `_precomputed_float64_array`,
  `_precomputed_values_agree`, `_initialize_precomputed_evaluations`,
  `_validate_initial_fresh_budget`, `_fresh_evaluations_for_batch`,
  `_bounds_check`, `_init_optim_state` (`:851-1112`), `save` (`:2795`),
  `load` (`:2833`), `_optim_state_record` (`:3035`), `_get_random_state`,
  `_set_random_state`, `_ensure_gp_sampling_history` (`:2331`),
  `_init_log_joint` (`:3336`), `_rebuild_log_joint` (`:3453`), every
  `_validate_*_option`, and (skimmed as non-numerical) `_init_logger`,
  `_log_column_headers`, `_setup_logging_display_format`, `__str__`, `__repr__`.
- `pyvbmc/vbmc/options.py`, `pyvbmc/vbmc/_bounds.py`,
  `pyvbmc/vbmc/iteration_history.py`, `pyvbmc/rng.py`, `pyvbmc/__init__.py`.
- `pyvbmc/vbmc/option_configs/basic_vbmc_options.ini` and
  `advanced_vbmc_options.ini`: every declared name, every default expression and
  every `# description` line.

### Read as an interface (to see how the state this slice builds is consumed)

`optimize()`'s consumption of `optim_state` and `options`
(`vbmc.py:1114-1830`, `_check_termination_conditions`,
`_compute_reliability_index`, `_check_warmup_end_conditions`,
`_is_gp_sampling_finished`, `_create_result_dict`);
`active_sample.py:100-330` (initial design, `skip_logger`, cache removal);
`gaussian_process_train.py:460-640` (`optim_state["N"]`, `n_eff`,
`max_fun_evals`); `variational_optimization.py:740-960` (`_sieve`, `_vb_init`);
`whitening.py:176-260` (transformer reinstallation, `temperature`);
`function_logger.py:265-340`, `:468-560` (`__call__` and `add`);
`variational_posterior.py:27-175` (constructor and the space in which `mu` lives);
`priors/convert_to_prior.py`. gpyreg was not needed beyond `GP.fit(rng=)`.

### Not reached

Warmup/termination/`final_boost`/`determine_best_vp` internals beyond the state
they read (slice P1a); the acquisition functions; the PyMC adapter itself
(only the `VBMC` side of `_resolve_pymc_target`).

### Checks run

All from the repository root with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/Scripts/python.exe -u`;
scripts and outputs are in the scratch directory. `pyvbmc.__file__` printed as
`C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\__init__.py`. No `optimize()` run,
no test suite, nothing written inside the repository.

| script | what it settled |
| --- | --- |
| `scan_keys.py`, `scan_keys2.py` | every `optim_state` key's readers and writers across `pyvbmc/` (excluding `testing/`), with whitespace flattened so multi-line subscripts are seen. Produced the dead-key list and found `entropy_force_switch` (read, never written). |
| `scan_opts.py`, `scan_opts3.py` | every declared option name against every `options[...]` / `options.get(...)` / `options.eval(...)` read. Confirms `INERT_OPTIONS` equals the set of names that appear nowhere else (22), and that three further declared names — `diagnostics`, `entropy_force_switch`, `temperature` — are read from no options mapping at all. Also lists names read from an options mapping but never declared (`varactivesample`, `warp_nonlinear`, `skip_elbo_variance`, plus two in comments). |
| `check1.py` | `pyvbmc` imports no torch/PyMC/PyTensor/corner; `entropy_force_switch` absent from `optim_state` and the reader's product raises `TypeError`; `temperature` absent from `optim_state`; scalar bounds raise at `D=3`; `VBMC(seed=None)` advances NumPy's global state (position 624 → 4); seeded construction reproducible and `vp.rng is vbmc.rng`; bound repairs for x0 on a hard bound, x0 on PLB, three starting points, LB==UB, integer and float32 inputs. |
| `check2.py` | save/load round trip in the scratch directory; transformer and generator sharing before and after `load`, for the last and for an earlier iteration; the stored finished run `test_vbmc_save_static.pkl` loaded read-only. |
| `check3.py` | `specify_target_noise` through `options=` versus through `options_path=`; unknown names accepted from `options_path=` and rejected from `options=`; the descriptions the option parser actually stores. |
| `check4.py` | `integer_vars` as mask versus indices; `get_rng` pass-through and reproducibility; `IterationHistory` growth, deep-copying, declared-key check; precomputed evaluations (dedup, counts, budget, `skip_logger`). |
| `check5.py` | all five log-joint configurations (joint alone, `prior=`, `log_prior=`, `sample_prior=`, scipy list), noisy level-2 pair, vectorized batch with a prior. |
| `check6.py` | `load(new_options={"max_fun_evals": 500})` leaves `optim_state["max_fun_evals"]` at 40, and the resulting GP training-schedule collapse; `new_options` typo accepted silently; `__str__` content. |
| `check7.py` | the initial `vp.mu` against the transformed `x0`, bounded and unbounded. |
| `check8.py` | options freeze versus `pop`; `x0` as a list; scalar bounds with several starting rows; dead `optim_state` entries; `iteration_history["data_trim_list"]` never recorded. |

### Verified correct, no finding

- **Bounds** (`_bounds.py` in full): real-valuedness check; float64 widening that
  leaves the caller's arrays untouched (verified for `int64` and `float32`, and
  the repairs work on detached copies, `:53-59`); reshaping to `(1,D)`;
  `(n0,D)` starting sets; the plausible box derived from a multi-row `x0`
  (`x0.min(0) - width/N0`, `x0.max(0) + width/N0`, clipped to the hard bounds,
  falling back to the hard bounds where the derived pair coincides); the
  fall-back to hard bounds for a single starting row; the finite-PLB
  requirement; the fixed-variable and matching-PLB rejections; the `x0`-inside-
  hard-bounds check; the effective-bound margin `1e-3*(UB-LB)` with `1e3`
  substituted for infinite ranges and the `realmin` special cases; moving `x0`
  inward before the plausible box is expanded to it (so the expanded box can
  never fall below the effective bound); the permissive and the strict ordering
  checks; the half-bounded rejection. Infinite hard bounds, equal hard bounds,
  and `x0` exactly on a hard bound all behave as documented.
- `self.x0` is rebound to transformed coordinates at `vbmc.py:460` *after*
  `optim_state["cache"]["x_orig"]` has taken the original-coordinate array
  (`:869`), so the starting cache is in the right space and is not aliased.
- **`_init_optim_state`**: transformed hard/plausible bounds and the
  `tol_bound_x` margins (the `errstate` guards are appropriate); the expanding
  search bounds; `gp_cov_fun`; the `gp_noise_fun` triple per uncertainty level;
  the `gp_mean_fun` whitelist; the uncertainty-level ladder (2 for
  `specify_target_noise`, 1 for a non-empty `uncertainty_handling`, else 0);
  `stop_sampling` keyed on `ns_gp_max`; `last_warmup`; the `det_entropy_min_d`
  gate on `entropy_switch`; `out_warp_delta` as `[]` when `fitness_shaping` is
  off, which is what its reader at `:1677` compares against; the `f_vals`
  length check. `optim_state["N"]` and `["n_eff"]` are not initialized here but
  are written at `vbmc.py:1463-1466` before their first reader (`train_gp`).
- **Options**: the layering order basic → advanced → `options_path=` with the
  `options=` dict winning (`load_options_file` skips `useroptions`, `:173`);
  freezing after `validate_option_names` and the `force=True` escape; unknown
  keys in the `options=` dict raising; `INERT_OPTIONS` equals exactly the set of
  declared names that appear nowhere else in the package (22 names, recomputed
  independently); `_warn_inert_options` skipping callable defaults and comparing
  the rest with `_equals_default`; no option declared twice within or across the
  two files; every option preceded by a `# description` line; every D-dependent
  default sensible and positive at `D = 1, 2, 20`; every `options.eval` call
  site's keyword names matching the parameter names of the declared lambda
  (`ns_elbo`/`ns_ent*`/`adaptive_k`/`pruning_threshold_multiplier` take `K`,
  `k_fun_max` takes `N`, `active_importance_sampling_mcmc_samples` is a
  constant); `Options.eval` returning a plain value unchanged;
  `__copy__`/`__deepcopy__` preserving `descriptions` and the frozen flag.
- **`IterationHistory`**: `record` deep-copies the value and pads with `None`
  through an object array when the iteration jumps ahead; `_expand_array`
  deliberately bypasses `__setitem__` so the stored objects are not copied
  again; the declared-key check fires from both `record` and `__setitem__`;
  negative iterations rejected; `__reduce__`/`__getstate__`/`__setstate__`
  round-trip; every key recorded by `optimize()` is in the declared list.
- **`_optim_state_record`**: returns the live dict unless there are importance
  samples and `record_full_history_details` is off, in which case only that one
  key is blanked on a shallow copy; the live dict is not modified.
- **Log-joint construction**: all five configurations produce the documented
  callable; the noisy level-2 wrapper returns the `(value, sd)` pair with the
  prior added to the value only; the vectorized wrapper validates the
  likelihood output, evaluates the prior row by row with a finiteness check and
  returns `(values, noise)` only at level 2; `_rebuild_log_joint` restores the
  wrapper, the prior, `self.log_joint` and `function_logger.fun`, and leaves
  `options["vectorized_target"]` at the value `load` decided.
- **Precomputed evaluations**: shape, real-valuedness, finiteness and
  positive-`y_sd` validation with row numbers; the level-2/level-0 consistency
  rules; the strict-interior check against the hard bounds; deterministic
  duplicates collapsed to one row after a four-ULP agreement test (verified,
  including the conflicting-value rejection); noisy repeats kept; the prior
  added exactly once, with a scalar/finite check; `skip_logger` marking the
  first cache row that coincides with a supplied location (verified);
  `func_count` untouched and `initialization_cost` charged once
  (`optim_state["max_fun_evals"] = 193` for `200 - 7`).
- **Save/load**: round trip of a freshly constructed instance keeps the
  transformer shared by `vbmc`, `vp` and `function_logger` and the generator
  shared by `vbmc` and `vp`; `iteration=` bounds-checked; every history key
  truncated to `iteration+1`; `get_gp(iteration)` returns a copy with the
  posterior factors rebuilt; the backfills for `performance_calibration`,
  `show_tips`, `vectorized_target`, the `N` history, a missing `rng` (created
  without touching the global state and shared with every stored VP) and the
  budget attributes; `_set_random_state` accepting the current dict format, the
  older dict with a legacy entry, and the bare global-state tuple, restoring the
  global state it consumed and warning.
- **`rng.py`**: a `Generator` is returned unchanged (so it can be shared); an
  int seed is reproducible; `np.random.seed(...)` before construction fixes the
  derived generator. (But see F9 for what the derivation does to the global
  state.)
- **`pyvbmc/__init__.py`**: importing `pyvbmc` imports neither torch, PyMC,
  PyTensor nor corner; `SVBMC` and `PyMCTarget` resolve lazily through
  `__getattr__`, and `__dir__` advertises them.

## 2. Findings

### F1. The initial variational posterior's component means are set from the untransformed `x0`
- Location: `pyvbmc/vbmc/vbmc.py`:409-416 (and `:460`); MATLAB: "no counterpart read" (internal track)
- Category: indexing/shape (wrong coordinate space)
- Proposed classification: suspected defect
- Confidence: high
- `VBMC.__init__` builds the variational posterior at `:409-416` with
  `x0=self.x0`, and only afterwards, at `:460`, replaces `self.x0` with
  `self.parameter_transformer(self.x0)`. So the `x0` handed to
  `VariationalPosterior.__init__` is in the *original* (constrained) coordinates,
  while `VariationalPosterior.__init__` writes it straight into `self.mu`
  (`pyvbmc/variational_posterior/variational_posterior.py:133-146`), and `mu`
  is by definition in the transformed space: the class docstring states "Note
  that q(θ) is defined in an unconstrained space. Constrained variables in the
  posterior are mapped to a transformed, unconstrained space via a nonlinear
  mapping" (`variational_posterior.py:105-110`), and every consumer treats it
  that way (`vp.sample`/`vp.pdf` apply `parameter_transformer.inverse` to points
  built from `mu`). The two component means are therefore placed at the point
  whose *transformed* coordinate equals the *original* `x0`.
- Consequence if real: fires at default options for every problem whose
  plausible box is not centred on the origin. Measured (`check7.py`): with
  `LB=0, UB=10, PLB=4, PUB=6, x0=5`, the transformed `x0` is `0` but
  `vp.mu = 5`, which maps back to the original coordinate `9.94` — the initial
  components sit next to the upper bound instead of at the starting point. With
  unbounded bounds, `PLB=2, PUB=4, x0=3`, `vp.mu = 3` maps back to `9.0`, three
  plausible-box widths away. Two effects follow. (i) `vbmc.vp` before
  `optimize()` is a user-visible object (`vbmc.vp.sample()`, `vbmc.vp.plot()`)
  that describes the wrong region. (ii) In iteration 0, one third of the sieve's
  candidates are generated from these means
  (`variational_optimization.py:805-807`, type 1) and are wasted; the other two
  thirds are built from high-posterior-density training points and are
  unaffected, so the optimized posterior recovers. It is also the `vp_old`
  against which iteration 0's `sKL` is computed (`vbmc.py:1215`, `:1548-1565`),
  though that entry never reaches the reliability index, which is `inf` for
  `iter < 2` (`:2214-2217`). The error vanishes exactly when the transformed and
  original coordinates of `x0` coincide, which is the case in the example
  notebooks (symmetric plausible box, `x0` at its centre) — which is why it has
  not been noticed.
- Suggested reproduction: already run.
  `VBMC(f, np.array([[5.0]]), np.array([[0.0]]), np.array([[10.0]]), np.array([[4.0]]), np.array([[6.0]]))`
  then compare `vbmc.x0` (`[[0.]]`) with `vbmc.vp.mu` (`[[5.0, 5.0]]` up to the
  1e-6 jitter) and `vbmc.parameter_transformer.inverse(vbmc.vp.mu.T)`
  (`[[9.94], [9.94]]`).
- Test adequacy: no. `pyvbmc/testing/vbmc/test_vbmc_init.py` never inspects
  `vbmc.vp.mu` (its only VP assertion is `vbmc.optim_state["vp_K"] == vbmc.vp.K`
  at `:525`), and the end-to-end runs in `test_vbmc_optimize.py` use targets
  whose plausible box is centred at the origin, where the bug is invisible.

### F2. `optim_state["entropy_force_switch"]` is never written, so `entropy_switch=True` makes `optimize()` raise
- Location: `pyvbmc/vbmc/vbmc.py`:1224-1228 (reader) against `:851-1112` (`_init_optim_state`, no writer); option at `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini`:211; MATLAB: "no counterpart read" (internal track)
- Category: cross-module (a key read under a name that is never written)
- Proposed classification: suspected defect
- Confidence: high
- `optimize()` computes
  `self.optim_state.get("entropy_force_switch") * self.optim_state.get("max_fun_evals")`
  inside `if self.optim_state.get("entropy_switch") and (...)`. Every other
  option that the main loop reads out of `optim_state` is copied there by
  `_init_optim_state` (`tol_gp_var` at `:995`, `entropy_alpha` at `:1035`,
  `max_fun_evals` at `:1001`, ...), but `entropy_force_switch` is not, and no
  other module writes it (whole-package scan). `optim_state.get` therefore
  returns `None` and the multiplication raises
  `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`.
  The option's own description — "Force switch to stochastic entropy at this
  fraction of total fcn evals" — describes a rule that consequently never runs.
- Consequence if real: the whole `entropy_switch` feature is unusable. At the
  shipped default `entropy_switch = False` (ini `:209`) the guard is false and
  nothing happens, so the default path is unaffected; but a user who sets
  `entropy_switch=True` on a problem with `D >= det_entropy_min_d` (5) gets a
  crash in the first iteration rather than the documented behaviour. It also
  means the deterministic-entropy start can only ever be left through the
  stability route at `:2169-2173`, never at the 80 %-of-budget mark.
- Suggested reproduction: already run (`check1.py`):
  `v = VBMC(f, np.zeros((1,5)), ..., options={"entropy_switch": True})`;
  `"entropy_force_switch" in v.optim_state` is `False`, and
  `v.optim_state.get("entropy_force_switch") * v.optim_state.get("max_fun_evals")`
  raises `TypeError`.
- Test adequacy: no. `test_vbmc_optimstate_entropy_switch`
  (`test_vbmc_init.py:572-584`) checks only that `optim_state["entropy_switch"]`
  takes the right value; no test enables the option and iterates, and the
  `INERT_OPTIONS` scan test cannot see the problem (see F8).

### F3. `load(new_options={"max_fun_evals": N})` leaves the stale budget in `optim_state`, collapsing the GP training design
- Location: `pyvbmc/vbmc/vbmc.py`:2956-2999 (`load`) against `:1001` (`_init_optim_state`) and `pyvbmc/vbmc/gaussian_process_train.py`:624-626; MATLAB: "no counterpart read" (internal track)
- Category: state/caching (and cross-module)
- Proposed classification: suspected defect
- Confidence: high
- `_init_optim_state` copies the budget into `optim_state["max_fun_evals"]`
  because, as its own comment says, it is "used by some schedules and
  acquisition functions". `load` re-derives `_configured_max_fun_evals` and
  `_effective_max_fun_evals` from the new options (`:2981-2984`) but updates
  `optim_state["max_fun_evals"]` **only** on the opted-in budget path
  (`:2985-2997`). On the ordinary path the loaded `optim_state` keeps the value
  the run was constructed with. The one live reader is
  `gaussian_process_train.py:624-626`,
  `schedule_limit = min(optim_state.get("max_fun_evals", options["max_fun_evals"]), 1e3)`,
  which sets the horizon of the cubic schedule for `init_N`, the number of
  starting points of the GP hyperparameter fit. Past the stale horizon,
  `x = (n_eff - fun_eval_start)/schedule_span` exceeds 1, the cubic goes
  negative and `init_N = max(round(f(x)), 9)` pins to its floor 9. (The clip to
  `[0,1]` at `:635-636` is applied only when `budget_active`.)
- Consequence if real: it fires in exactly the use `new_options` is documented
  for — "to continue a previous run with a larger budget of function
  evaluations" (`vbmc.py:2851-2855`), which the runtime tip at
  `pyvbmc/vbmc/_tip_catalog.py:93` also recommends. Measured on the stored run
  (`check6.py`): after `load(..., new_options={"max_fun_evals": 500})`,
  `options["max_fun_evals"] = 500` while `optim_state["max_fun_evals"] = 40`;
  the GP fit's initial design is then 9 points for the rest of the run where the
  correct horizon gives 414 (at `n_eff = 150`), 273, 129 and 65. A hyperparameter
  optimization started from 9 instead of tens or hundreds of design points is
  markedly more likely to land in a poor local optimum, which propagates into
  every downstream quantity. Termination is unaffected (it reads
  `options["max_fun_evals"]` at `:2102-2107`), so the run really does continue to
  the new budget with the degraded schedule.
- Suggested reproduction: already run —
  `VBMC.load("pyvbmc/testing/vbmc/test_vbmc_save_static.pkl", new_options={"max_fun_evals": 500})`
  and compare `options["max_fun_evals"]` with `optim_state["max_fun_evals"]`.
- Test adequacy: no. `test_vbmc_load_static`
  (`test_vbmc_save_and_load.py:113-121`) loads with
  `new_options={"max_fun_evals": 42}` and asserts only
  `vbmc.options["max_fun_evals"] == 42`; it mirrors the implementation's one
  update instead of the invariant that the two copies agree.

### F4. Options supplied through `options_path=` never reach `update_defaults()`, so an `.ini`-configured noisy run keeps the noiseless defaults
- Location: `pyvbmc/vbmc/vbmc.py`:306-324 against `pyvbmc/vbmc/options.py`:100-112; MATLAB: "no counterpart read" (internal track)
- Category: control flow (and defaults)
- Proposed classification: suspected defect
- Confidence: high
- `__init__` loads the advanced defaults, calls `self.options.update_defaults()`
  (`:311`), and only *then* loads `options_path` (`:312-317`).
  `update_defaults` is the sole place where `specify_target_noise` adjusts the
  other defaults (1.5× `max_fun_evals` and `tol_stable_count`,
  `active_sample_gp_update`/`vp_update` on, and `search_acq_fcn = [AcqFcnVIQR()]`).
  A `specify_target_noise = True` written in the user's `.ini` file arrives
  after that call and never triggers it, while `_init_optim_state` (`:1016`)
  still reads it and sets `uncertainty_handling_level = 2`. The docstring for
  `options_path` (`:124-128`) promises only that the `options=` dict takes
  precedence over the file, not that the file is a second-class route.
- Consequence if real: a run that is treated as noisy everywhere else searches
  with `AcqFcnLog`, the acquisition for noiseless targets, instead of VIQR; it
  gets two thirds of the intended evaluation budget and stability window, and it
  skips the per-sample GP and VP updates. Measured (`check3.py`): through the
  dict, `max_fun_evals 300 / tol_stable_count 90 / AcqFcnVIQR / gp_update True`;
  through the file, `200 / 60 / AcqFcnLog / False`, with
  `uncertainty_handling_level = 2` in both. This is a large difference in the
  quality of a noisy run, and it fires at default options for anyone who
  configures through a file.
- Suggested reproduction: already run. An `.ini` file containing
  `specify_target_noise = True` under a section header, passed as
  `options_path=`, versus `options={"specify_target_noise": True}`.
- Test adequacy: no. `test_init_options_path` (`test_vbmc_init.py:1133-1189`)
  exercises `options_path` only with the neutral keys of
  `option_configs/test_options.ini`, and `test_options_update_defaults`
  (`test_options.py:~120-161`) only through the `options=` dict.

### F5. The running-moments update is unreachable: a misplaced parenthesis makes its guard always true
- Location: `pyvbmc/vbmc/vbmc.py`:1580-1597; option `moments_run_weight` at `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini`:189; MATLAB: "no counterpart read" (internal track)
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- The guard reads

  ```python
  if len(self.optim_state.get("run_mean")) == 0 or len(
      self.optim_state.get("run_cov") == 0
  ):
  ```

  The `== 0` sits inside the `len(...)` call: the second operand is
  `len(run_cov == 0)`, not `len(run_cov) == 0`. At iteration 0 the first operand
  is true. From iteration 1 on, `run_cov` is the `(D,D)` covariance,
  `run_cov == 0` is a `(D,D)` boolean array and `len(...)` is `D`, which is
  truthy for every `D >= 1`. The `if` branch is therefore taken on every
  iteration and the `else` branch (`:1586-1597`), which applies the
  exponential weight `moments_run_weight ** (N - last_run_avg)`, never executes.
- Consequence if real: `optim_state["run_mean"]`, `["run_cov"]` and
  `["last_run_avg"]` hold the current iteration's moments instead of a running
  average, and the option `moments_run_weight` ("Weight of previous trials (per
  trial) for running avg of variational posterior moments") has no effect.
  Nothing else in `pyvbmc/` reads those three keys — the only other readers are
  the reset at `pyvbmc/whitening/whitening.py:244-246` — so no number the
  algorithm computes changes today; what is lost is the recorded quantity and
  any future use of it, and the code as written cannot do what its option says.
  Note also that the intended `else` branch would be reached with `run_cov`
  still `[]` if the two keys ever went out of step, in which case
  `wRun * []` would fail; the guard as written hides that too.
- Suggested reproduction: in a Python session,
  `import numpy as np; len(np.eye(3) == 0)` returns `3`, whereas
  `len(np.eye(3)) == 0` returns `False`.
- Test adequacy: no test covers the branch;
  `test_vbmc_init.py:522-524` only checks the initial `[]`/`[]`/`nan`.

### F6. Scalar bounds are not replicated across dimensions, though the docstring says they are
- Location: `pyvbmc/vbmc/vbmc.py`:98-104 (docstring) against `pyvbmc/vbmc/_bounds.py`:137-157; MATLAB: "no counterpart read" (internal track)
- Category: defaults (documented contract versus code)
- Proposed classification: suspected defect
- Confidence: high
- The `lower_bounds, upper_bounds` docstring states "If scalars, the bound is
  replicated in each dimension." `_normalize_bounds` only `atleast_1d`s and
  reshapes to `(1,D)`; a scalar becomes shape `(1,)` and
  `reshape((1, D))` raises for every `D > 1`. Measured (`check1.py`): with
  `D=3`, `VBMC(f, np.zeros((1,3)), -10, 10, -1, 1)` raises
  `ValueError('Bounds must match problem dimension D=%d.', 3)`; the same call
  with `D=1` succeeds. The same promise is implicitly made for the plausible
  bounds, which are treated identically.
- Consequence if real: a documented call form fails. It is a hard failure, not a
  silent wrong number, so no result is corrupted; but the raised message is
  itself malformed (the `%d` template and its argument are passed as two
  `ValueError` arguments and never formatted), so the user is told
  `('Bounds must match problem dimension D=%d.', 3)`. A second, sharper edge
  exists for `D == 1`, where a scalar *is* accepted: there the scalar survives
  as a 0-d array into the plausible-bound derivation, and with more than one
  starting row and no plausible bounds supplied, `lower_bounds[idx]` at
  `_bounds.py:116-117` raises `IndexError: too many indices for array: array is
  0-dimensional` (reproduced in `check8.py`).
- Suggested reproduction: already run; both calls above.
- Test adequacy: no. `test_vbmc_bounds_check_not_D`
  (`test_vbmc_init.py:116-139`) checks mismatched *vector* lengths, and
  `test_vbmc_bounds_check_not_vectors` / `..._not_row_vectors` use arrays; no
  test passes a scalar bound.

### F7. `integer_vars` is used as a per-dimension mask while its documentation calls it a list of indices
- Location: `pyvbmc/vbmc/vbmc.py`:878-892; description at `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini`:4-5; MATLAB: "no counterpart read" (internal track)
- Category: indexing/shape (and defaults)
- Proposed classification: suspected defect
- Confidence: high
- The `.ini` description, which is the user documentation of the option, reads
  "Array with indices of integer variables". The code computes
  `integeridx = self.options.get("integer_vars") != 0` and then uses that as a
  boolean mask over dimensions, both to fill `optim_state["integer_vars"]`
  (`:881`) and to select the columns of the hard bounds (`:883-886`). That is a
  length-`D` 0/1 mask, not a list of indices.
- Consequence if real: a user who follows the description gets either a crash or
  silently wrong behaviour, depending on how many indices they list. Measured
  (`check4.py`, `D=3`): `[0,1,0]` and `[True,False,True]` work as masks;
  `[1]`, `[0,1]` and `[0,2]` raise `IndexError: boolean index did not match
  indexed array`. The silent case is the dangerous one: with `D = 2` and
  `integer_vars = [0, 1]` meaning "variables 0 and 1 are integers", the mask
  reading marks variable 1 only, so variable 0 is never snapped to the integer
  grid by `AbstractAcqFcn._real2int` and the hard-bound check at `:882-892`
  never applies to it — a wrong inference with no message. (`D = 2` with
  `[1, 2]` marks both, by coincidence.)
- Suggested reproduction: already run;
  `VBMC(f, x0_D2, lb, ub, plb, pub, options={"integer_vars": np.array([0, 1])})`
  and inspect `optim_state["integer_vars"]`.
- Test adequacy: no. `test_vbmc_optimstate_integer_vars`
  (`test_vbmc_init.py:397-425`) passes `np.array([1, 0, 0])` for `D = 3` — a
  mask — and asserts the mask that the implementation produces. It mirrors the
  implementation rather than the documented contract.

### F8. Four more declared options have no effect, and the `INERT_OPTIONS` scan cannot see them
- Location: `pyvbmc/vbmc/options.py`:21-46 (`INERT_OPTIONS`) with `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini`:27 (`diagnostics`), `:35` (`proposal_fcn`), `:257` (`temperature`), `:309` (`recompute_lcb_max`); MATLAB: "no counterpart read" (internal track)
- Category: defaults (and cross-module)
- Proposed classification: suspected defect
- Confidence: high for the four names; medium for whether each should be
  registered or repaired
- A whole-package scan of every `options[...]`, `options.get(...)` and
  `options.eval(...)` read gives 25 declared names that no module reads;
  `INERT_OPTIONS` lists 22 of them. The three extra are `diagnostics`,
  `temperature` and `entropy_force_switch` (F2). A fourth, `proposal_fcn`, is
  read once but only into a dead `optim_state` key, and a fifth,
  `recompute_lcb_max`, is read but gates a stub:
  - `temperature` (ini `:257`, "Temperature for posterior tempering"): the only
    readers are `pyvbmc/whitening/whitening.py:196-197` and `:291-292`, and they
    read `optim_state["temperature"]`, which `_init_optim_state` never writes,
    so `optim_state.get("temperature")` is always `None` and the code always
    falls back to `T = 1` (verified: `"temperature" in vbmc.optim_state` is
    `False` even with `options={"temperature": 4}`).
  - `diagnostics` (ini `:27`, "Run in diagnostics mode get additional info"):
    no reader at all. The only quoted occurrence of the name in the package is
    `sampler_opts["diagnostics"] = False` in
    `pyvbmc/vbmc/active_importance_sampling.py:437`, an unrelated slice-sampler
    option.
  - `proposal_fcn` (ini `:35`): read at `vbmc.py:963-966` into
    `optim_state["proposal_fcn"]`, which nothing reads — the same shape as the
    `gp_int_mean_fun` → `optim_state["int_mean_fun"]` entry that the sheet lists
    first.
  - `recompute_lcb_max` (ini `:309`, "Recompute LCB max for each iteration based
    True current GP estimate"): read at `:1642`, but `_recompute_lcb_max`
    (`:2403-2408`) is a stub with a `# ToDo` that returns `np.array([])`, and its
    result goes into `optim_state["lcb_max_vec"]`, which nothing reads —
    `_check_warmup_end_conditions` uses `iteration_history["lcb_max"]` instead
    (`:1975`).
  The registry's regression test cannot catch any of these: it searches the
  whole package text for the *quoted name*
  (`test_options.py:200-205`), so `optim_state["temperature"]`,
  `sampler_opts["diagnostics"]`, `optim_state["proposal_fcn"]` and
  `options.get("recompute_lcb_max")` all count as reads.
- Consequence if real: four options are accepted at a non-default value with no
  warning and no effect. The user-visible harm is confined to the false
  documentation (the descriptions promise behaviour that does not exist) plus
  the missing "no effect" warning that `_warn_inert_options` gives for the other
  22, since none of the four is on by default and the features behind them are
  unported anyway. `temperature` is the one with teeth: posterior tempering is
  not ported (sheet, slice P7), yet the option looks live.
- Suggested reproduction: already run (`scan_opts3.py`, `check1.py`,
  `check8.py`).
- Test adequacy: the test exists and passes while three of these are dead —
  see §3.
- Sheet note: this contradicts the sheet's P7 entry "Posterior tempering
  (`vbmc_power`, `vptrain2real`) is not ported", which states "The option
  `temperature = 1` exists ... and is read only by
  `pyvbmc/whitening/whitening.py:196`, `:291`". Those two lines read an
  `optim_state` key of the same name that nothing ever writes, so the option is
  read nowhere.

### F9. Constructing `VBMC` with `seed=None` advances NumPy's global random state, which the docstring says is never written
- Location: `pyvbmc/rng.py`:23-25; docstring claim at `pyvbmc/vbmc/vbmc.py`:152-154; MATLAB: "no counterpart read" (internal track)
- Category: random draws
- Proposed classification: suspected defect (documentation against behaviour)
- Confidence: high
- `get_rng(None)` derives the seed with
  `np.random.randint(0, 2**32, size=4, dtype=np.uint32)`, which draws from —
  and therefore advances — NumPy's global legacy generator. The `VBMC` `seed`
  docstring says "Every random draw of a run comes from ``vbmc.rng``; NumPy's
  global random state is never written", and `optimize()`'s Notes repeat
  "NumPy's global random state is neither read nor written" (`:1132-1134`).
  `get_rng`'s own docstring says only "seeded from NumPy's global legacy random
  state". That the intent is to leave the global state untouched is visible in
  `_set_random_state` (`vbmc.py:3106-3111`), which carefully restores it after
  calling `get_rng()`, and in `load`'s fresh-generator path (`:3017-3018`),
  which uses `np.random.default_rng()` precisely so as not to touch it.
- Consequence if real: measured (`check1.py`), a single unseeded `VBMC(...)`
  moves the global MT19937 position from 624 to 4 and rewrites its key — four
  32-bit draws consumed. Anything the user draws from `np.random` around a
  construction is therefore shifted, and a script that constructs a `VBMC`
  object between two global draws is not reproducible from the global seed alone
  in the way the docstring implies. The intended property ("`np.random.seed`
  beforehand still fixes a run") does hold, and two successive unseeded
  constructions correctly get different streams. This is a claim that is false,
  not a wrong number.
- Suggested reproduction: already run —
  `np.random.seed(0); s = np.random.get_state(); VBMC(...); np.random.get_state() != s`.
- Test adequacy: no. `test_vbmc_seed.py` checks reproducibility of a run, not
  the untouched-global-state claim; `test_vbmc_load_static` asserts the global
  state is unchanged across `load`, but nothing asserts it across construction.

### F10. After `load`, the parameter transformer is no longer the shared object, and `vbmc.parameter_transformer` is not the chosen iteration's
- Location: `pyvbmc/vbmc/vbmc.py`:2915-2935; MATLAB: "no counterpart read" (internal track)
- Category: state/caching
- Proposed classification: suspected defect
- Confidence: high on the behaviour, medium on the consequence
- `load` restores `vp`, `function_logger` and `optim_state` from the iteration
  history (`:2918-2927`), each of which carries its own deep copy of the
  transformer made at record time, and leaves `vbmc.parameter_transformer` as
  the pickled live attribute. Measured on the stored run (`check2.py`): after
  `VBMC.load(...)`, `vbmc.parameter_transformer is vbmc.vp.parameter_transformer`
  is `False`, likewise for the logger, and all three are pairwise distinct
  (they compare equal by value). Before the save they are one object, and
  `AGENTS.md` records the invariant that the three share it. The generator is
  fine: `vbmc.vp.rng is vbmc.rng` survives the round trip, because
  `VariationalPosterior.__deepcopy__` shares it.
- Consequence if real: no numerical consequence in a continued run, because the
  main loop transforms through `vp.parameter_transformer` and
  `function_logger.parameter_transformer` and re-establishes identity at
  `:1385-1388` on the first iteration. Two visible effects remain. (i) The
  invariant that the tests assert with `is` no longer holds for a loaded
  instance. (ii) When `iteration=` selects an iteration before a warp,
  `vbmc.parameter_transformer` is the *later*, warped transformer while `vp`,
  `function_logger` and `optim_state` are pre-warp, and `vbmc.x0` (stored in the
  pre-warp transformed space, never re-transformed by the warp) is inverted with
  it in `__str__` (`:3496`), which then prints a wrong starting point. A user who
  reaches for `vbmc.parameter_transformer` to convert coordinates after such a
  load gets the wrong map.
- Suggested reproduction: already run —
  `s = VBMC.load("pyvbmc/testing/vbmc/test_vbmc_save_static.pkl"); s.parameter_transformer is s.vp.parameter_transformer` → `False`.
- Test adequacy: no. `test_vbmc_load_dynamic`/`test_vbmc_load_static`
  (`test_vbmc_save_and_load.py:51-142`) check bounds, the iteration and option
  values, never the sharing invariant.

### F11. Option descriptions are truncated at the first `=` or `:`
- Location: `pyvbmc/vbmc/options.py`:401-437 (`_read_config_file`, the parser configured at `:415`); MATLAB: "no counterpart read" (internal track)
- Category: defaults (user documentation of the options)
- Proposed classification: suspected defect
- Confidence: high
- `_read_config_file` parses the `.ini` files with
  `configparser.ConfigParser(comment_prefixes="", allow_no_value=True)` so that
  `#` lines survive as entries and become descriptions. But configparser still
  splits every line on its delimiters, `=` and `:`; a description containing
  either keeps only the part before it. Measured (`check3.py`): the stored
  descriptions are "Number of GP samples when GP is stable (0" for
  `stable_gp_samples`, "Multiplier to widths from previous posterior for GP
  sampling (Inf" for `gp_sample_widths`, "Explicit noise handling (0" for
  `uncertainty_handling`, "Upper bound True GP input lengths based True
  plausible box (0" for `upper_gp_length_factor`, "Local optimizer of the
  acquisition search" for `search_optimizer`, "Fixed numerical chunk profile"
  for `performance_calibration`, "Cap on the number of GP hyperparameter samples
  of the GP refits within active sampling (0" for `ns_gp_max_active`, and
  "Temperature for posterior tempering (allowed values T" for `temperature`.
- Consequence if real: `Options.descriptions`, and therefore `repr(options)` and
  `str(options)` (`options.py:330-380`), drop exactly the part of the
  documentation that lists the accepted values or the meaning of the sentinel —
  including the values of `search_optimizer` and the `adaptive_k` argument that
  commit `e6a21a4` added for that purpose. The published documentation is
  unaffected, because `docsrc/source/api/options/vbmc_options.rst` includes the
  two `.ini` files literally. No numerical effect.
- Suggested reproduction: already run —
  `from pyvbmc.vbmc.options import _read_config_file;`
  `[d for k,v,d in _read_config_file("option_configs/advanced_vbmc_options.ini") if k=="search_optimizer"]`.
- Test adequacy: no. `test__str__and__repr__` (`test_options.py:258-264`) checks
  one description, `"bar: 40 (Bar description)"`, from the test fixture, whose
  text contains neither delimiter.

### F12. Option names are validated for the `options=` dict but not for `options_path=` or `load(new_options=)`
- Location: `pyvbmc/vbmc/options.py`:177-212 with `pyvbmc/vbmc/vbmc.py`:312-324 and `:2956-2959`; MATLAB: "no counterpart read" (internal track)
- Category: control flow
- Proposed classification: possibly intentional
- Confidence: high on the behaviour
- `validate_option_names` checks that every key of the object appears in one of
  the paths it is given, and `__init__` passes `options_path` among them, so any
  name a user writes in their own `.ini` file is declared by construction and is
  accepted. `load` sets `is_initialized = False`, calls
  `vbmc.options.update(new_options)` and sets the flag back, never validating.
  Measured (`check3.py`, `check6.py`): `options_path` containing
  `max_fun_eval = 123` (a typo) is accepted and stored, while
  `options={"max_fun_eval": 123}` raises "The option max_fun_eval does not
  exist."; `VBMC.load(..., new_options={"max_fun_eval": 999})` is accepted
  silently and leaves `max_fun_evals` at its old value.
- Consequence if real: a misspelled option silently has no effect on the two
  routes where the user cannot see the default list next to what they typed. The
  `load` case is the worse one: a user extending a run with a typo'd budget gets
  a run that terminates immediately at the old budget, with no message.
- Suggested reproduction: already run.
- Test adequacy: no test supplies an unknown name through either route;
  `test_options.py` covers only the dict.

### F13. Passing `initialization_cost=0` or `precomputed_evaluations=None` explicitly changes the run
- Location: `pyvbmc/vbmc/vbmc.py`:335-347 with `:821-829`, `:1138-1145`, `:2193-2201`; MATLAB: "no counterpart read" (internal track)
- Category: control flow (defaults)
- Proposed classification: possibly intentional
- Confidence: high on the behaviour, low on whether it matters
- `_budget_active` is set from *whether the arguments were supplied*, not from
  their values: `bool(precomputed_was_provided or initialization_cost_was_provided)`,
  where "provided" means "not the private sentinel". So
  `VBMC(..., initialization_cost=0)` or `VBMC(..., precomputed_evaluations=None)`
  — both of which the docstring describes as the defaults ("Default is zero",
  "Evaluations available before the run") — take the opted-in path, where
  `_fresh_evaluations_for_batch` caps each acquisition batch at the remaining
  allowance (`:1394`), `optimize()` refuses to start with no allowance left, and
  the "prevent early termination" clause at `:2197-2201` gains an extra
  condition. On the ordinary path the last batch may overshoot `max_fun_evals`
  by up to `fun_evals_per_iter - 1`.
- Consequence if real: the two calls that a reader would take to be equivalent
  give runs with a different number of target evaluations in the final
  iteration. No wrong number, but a surprise, and it makes the budget semantics
  depend on an argument's presence rather than its value.
- Suggested reproduction: compare
  `VBMC(f, ...)._budget_active` (False) with
  `VBMC(f, ..., initialization_cost=0)._budget_active` (True).
- Test adequacy: `test_vbmc_precomputed.py` exercises the opted-in path with
  real data; nothing pins the equivalence of the explicit-default call with the
  omitted one.

### Minor observations

- `VBMC.__str__` (`vbmc.py:3487-3506`) reads the GP from
  `getattr(getattr(self, "vp", None), "gp", None)`, but `VariationalPosterior`
  has no `gp` attribute, so the summary always prints "Gaussian process = None"
  even after a run (verified). The same method prints
  `log-prior = {getattr(self, "log_prior", None)}` and
  `prior sampler = {getattr(self, "sample_prior", None)}`, and `__init__` never
  assigns either attribute (it assigns `log_joint`, `log_likelihood`, `prior`),
  so both always print `None` (verified).
- `_init_logger` (`vbmc.py:3295-3301`) removes handlers while iterating over
  `logger.handlers`, which skips an element when two match, and reads
  `handler.baseFilename` on every handler, which raises `AttributeError` if a
  non-file handler was attached to the `"VBMC"` logger (a process-wide
  singleton shared by all instances). `log_file_level` accepts only the six
  standard numeric levels (`:3318`), so `logging.DEBUG + 5` is rejected although
  the description says "a level from logging module".
- Fourteen `optim_state` entries are written and never read anywhere in
  `pyvbmc/`: `H` (`:1346`, `:1538`), `hedge` (`:1025`), `int_mean_fun`
  (`:1101`, already in the sheet), `iter_list` (`:1028-1032`), `lb_orig` and
  `ub_orig` (`:895-896`; `plb_orig`/`pub_orig` *are* read by the warp),
  `lcb_max_vec` (`:1643`), `proposal_fcn` (`:964-966`), `pruned` (`:985`),
  `redo_roto_scaling` (`:1214`), `vp_repo` (`:1663`), `warmup_stable_count`
  (`:960`), `initialization_cost` and `max_fun_evals_total` (`:1004`, `:1003`).
  Most are harmless leftovers; the two that carry a user option are covered by
  F8.
- `iteration_history["data_trim_list"]` is declared at `vbmc.py:473` and never
  recorded, so it stays `None` for the life of the object (the trim list lives
  in `optim_state` instead).
- `Options` freezes `__setitem__` after initialization but not `__delitem__`, so
  `vbmc.options.pop("max_iter")` succeeds on an initialized object and removes a
  key the algorithm reads (verified). `validate_option_names`'s docstring says
  initialization is flagged complete "to prevent further modification of
  options".
- `_bounds.py:155-157` raises `ValueError("Bounds must match problem dimension
  D=%d.", D)`: the template and the argument are two separate `ValueError` args,
  so the message is never formatted.
- `_bounds.py:252-261` warns "The starting points X0 are not inside the provided
  plausible bounds PLB and PUB. Expanding the plausible bounds..." when `x0` is
  *exactly on* a plausible bound (the test uses `<=`/`>=`), in which case the
  following `minimum`/`maximum` expands nothing; the warning is then misleading.
- `VBMC.__init__` calls `x0.ndim` (`:294`) before converting, so a list or a
  float for `x0` raises `AttributeError: 'list' object has no attribute 'ndim'`
  rather than a bounds message (verified), although the signature annotates
  `np.ndarray`.
- A single non-finite entry anywhere in `x0` replaces the whole starting set
  with one row at the centre of the plausible box (`:391-396`), silently
  discarding the finite starting points and reducing `n0` to 1.
- `IterationHistory.__setitem__` deep-copies, so `load`'s truncation loop
  (`vbmc.py:2931-2935`) copies the entire history a second time (every VP, every
  function logger, every recorded `optim_state`); `__setstate__` does the same
  on unpickling. Cost only, no correctness effect.
- `vbmc.py:1409` reads `self.options.get("varactivesample")`, a name no `.ini`
  declares (the declared one is `active_variational_samples`), so the term is
  always `None`; the branch is dead anyway at `separate_search_gp = False`.
  `warp_nonlinear` (`:1243`, `whitening.py:123`) is the same shape and is
  covered by the sheet's P8 entry.

## 3. Test adequacy notes

Tests of this slice that mirror the implementation rather than the
specification:

- `pyvbmc/testing/vbmc/test_options.py:183-205`,
  `test_inert_options_are_the_declared_options_nothing_reads`. It decides that an
  option is read if its name appears in quotes anywhere in the package text.
  That is the implementation's own spelling, not the specification "some module
  reads this option": it counts `optim_state["temperature"]` (a key nothing
  writes), `sampler_opts["diagnostics"]` (an unrelated slice-sampler option) and
  `optim_state["proposal_fcn"]` (a dead key) as reads, which is why F2 and F8
  pass unnoticed. A specification-shaped version would resolve reads against an
  options mapping and would additionally require that every option copied into
  `optim_state` has a reader for the copy.
- `test_vbmc_init.py:397-425`, `test_vbmc_optimstate_integer_vars`: builds the
  option as a mask and asserts the mask the code produces (F7).
- `test_vbmc_init.py:563-570`, `test_vbmc_optimstate_proposal_fcn`: asserts that
  a dead `optim_state` key holds what was put in it, which will keep passing
  however inert the option is.
- `test_vbmc_save_and_load.py:113-121`: asserts `options["max_fun_evals"] == 42`
  after `load(new_options=...)` and stops there, so the stale
  `optim_state["max_fun_evals"]` of F3 is invisible.
- `test_vbmc_init.py:1192-1195`, `test__str__and__repr__`: calls `__str__` and
  `__repr__` and asserts nothing about their content, so the always-`None` GP
  and prior lines survive.

Good models on this slice:

- `test_vbmc_init.py:985-1033`, `test_init_narrow_float_input` and
  `test_init_does_not_modify_repaired_input_arrays`: they state a property
  (everything widens to float64, the caller's arrays are never touched) and
  check it over parameterized dtypes, independently of how the widening is
  coded.
- `test_options.py:164-180`, `test_fun_eval_start_default`: parameterized over
  `D` with the expected values spelled out and the rule quoted in the docstring;
  it would catch a change of the formula.
- `test_vbmc_init.py:650-880`, the four `test_vbmc_init_log_joint*` tests:
  they check the constructed callable's *values* against independently computed
  sums for each configuration, rather than its identity or its closure.
- `test_optim_state_record.py` and `test_gp_records.py`: they state what the
  record must and must not contain and what a rebuild must reproduce.

## 4. Sheet notes

- **P7, "Posterior tempering (`vbmc_power`, `vptrain2real`) is not ported"** —
  the entry says the option `temperature` "is read only by
  `pyvbmc/whitening/whitening.py:196`, `:291`". Those lines read
  `optim_state["temperature"]`, and no module writes that key, so the option is
  read nowhere and `whitening` always takes `T = 1`. See F8. The entry's
  conclusion (tempering is not ported and is inert at the default) still holds;
  its description of how the option reaches the code does not.
- **P1b, "Declared options that nothing reads"** — the entry is accurate for the
  22 registered names (I recomputed the set independently and it matches
  exactly), but its closing claim that a test "recomputes the set by scanning
  the package for reads, so a newly dead option, or a newly read registered one,
  fails it" overstates what the test can detect: the scan matches any quoted
  occurrence of the name, so an option whose name also appears as a dictionary
  key elsewhere is counted as read. Three declared options are dead today
  without failing it (F8).
