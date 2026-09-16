# PyMC target adapter

Implementation plan for the PyMC integration chosen for PyVBMC 1.5.
Created 2026-09-14 on `dev-next` at `613f2a8`. Status: **pending
approval**, after the two investigations of
[Before approval](#before-approval-the-setup-probe-and-evaluation-reuse)
have run and their answers are written into this plan; code work runs on
the feature branch `dev-pymc-adapter` and merges back into `dev-next`.
Design inputs: the
[PyMC proposal](../2026-09-13-pymc-integration.md), the
[feasibility report](../results/2026-09-14-pymc-feasibility.md) and the
prototype `dev/scripts/pymc_feasibility.py`, whose module docstring is the
specification of the route through PyMC. The PI chose the scope on
2026-09-14 on the feasibility record, and `dev/TODO.md` carries the
decision; this plan settles the design questions that decision left to
the implementation and lays out the work.

## Pickup: 2026-09-16

Resume from `dev-next` with the two investigations in
[Before approval](#before-approval-the-setup-probe-and-evaluation-reuse).
The implementation plan remains **pending approval**. Neither
investigation has run; `dev/scripts/pymc_setup_probe.py` has not been
written, and Phase 0 has not started. No job or watcher needs reattachment.

Read `dev/TODO.md`, this plan, the
[PyMC proposal](../2026-09-13-pymc-integration.md), the
[feasibility report](../results/2026-09-14-pymc-feasibility.md) and the
prototype's module docstring before writing the probe. Preserve
`dev/scripts/pymc_feasibility.py` as the historical feasibility record:
its `find_MAP` and finite-difference Hessian are superseded by this
plan's gradient search and exact Hessian.

1. **Part A:** implement the separate setup probe over the specified
   models. Measure the `20 + 5 D` budget, stopping rules and prior-based
   location check, including models without a mode or usable gradients.
   Record the evidence and any design questions before Part B.
2. **Part B:** compare the three evaluation-reuse arms on the three
   specified models, five seeds each, at equal total evaluation budgets
   including setup. The probe emulates logger seeding without changing
   package code. Generate and assess the required NUTS references and
   run VBMC sequentially in the heavy-compute slot.
3. Write the findings into this plan's design, decisions, affected
   phases and execution record. Present the setup and reuse choices to
   the PI for approval; only then start Phase 0 and package implementation.

Both investigations belong to the orchestrator. In Codex, the executor
names below follow `AGENTS.md`: Astra fills the Fable orchestrator role;
Sol fills the Opus implementation/review role after approval. At most
one heavy process runs at a time.

The analysis environment is local by design and listed under "PyMC
feasibility check" in `dev/scripts/runs/LOCAL.md`. Its metadata was
checked on 2026-09-16: Python 3.12.6, PyMC 6.3.2, PyTensor 3.3.1,
ArviZ 1.3.0 and gpyreg 1.2.1. PyVBMC resolves to this checkout; gpyreg
resolves to the sibling checkout, so verify its revision before probing.
On a fresh machine, create a Python 3.12 environment, install those
PyMC/PyTensor/ArviZ versions, install gpyreg from tag `v1.2.1`
(`9e70e6b`) and this checkout editable with its development dependencies.
The tracked feasibility outputs under `dev/experiments/pymc_feasibility/`
remain readable without that environment. The planned probe's raw and
tracked output locations are specified below in Before approval.

The GP investigation is complete and preserved in the
[tail-acquisition report](../results/2026-09-16-gp-tail-acquisition.md)
and its linked box-sampler study; algorithmic remedies are outside 1.5.
The S-VBMC campaign is closed. Its headline switch remains tied to the
release gate, which follows the remaining 1.5 work in `dev/TODO.md`.

## Summary

A PyMC model with continuous free variables becomes a box-bounded log
joint for `VBMC`, with the mapping between the model's variables and
VBMC's flat parameter vector retained: a variable bounded on one side
keeps PyMC's transform and Jacobian and travels as an unbounded
coordinate, a variable bounded on both sides or unbounded is handed over
in the model's own coordinates with its interval as hard bounds, and
everything else is rejected by variable name before anything is compiled.
The adapter proposes the starting point and the plausible box within a
budget of `20 + 5 D` log-density evaluations: a gradient search for the
mode of its own log joint and a Laplace box from the exact Hessian there,
both from PyTensor's derivatives, with fallbacks that cost no evaluations;
explicit values are accepted and cost one evaluation, a finiteness check.
Every evaluation the setup spends is retained for VBMC to use as training
data. The adapter exports a fitted posterior as an ArviZ `DataTree` over
the model's variables with their shapes, dimensions and coordinates, and
the
user computes deterministics and posterior predictions with PyMC's own
functions on that export. The structured export is a generic extension of
`VariationalPosterior.to_arviz` (vector and matrix parameters with names,
dimensions and coordinates) that is useful without PyMC. PyMC is an
optional extra, imported only when the adapter is constructed. The
inference defaults, the numerical core and gpyreg are untouched; whether
`VBMC`'s interface gains a way to receive precomputed evaluations is
settled before approval.

## Scope

In scope:

- `pyvbmc.pymc` with the class `PyMCTarget` and the exception
  `UnsupportedModel`; `pyvbmc.PyMCTarget` resolved lazily.
- The `pymc` extra, a CI cell that installs it, and the tested PyMC and
  ArviZ version range.
- The generic structured export in `VariationalPosterior.to_arviz`
  (`variables`, `dims`, `coords`), backwards compatible.
- Tests, including `VBMC.save` and `VBMC.load` on an adapter target and
  one short seeded end-to-end run.
- The route by which the setup evaluations reach `VBMC`, as settled
  before approval (possibly an extension of `VBMC`'s interface, which
  would then be in scope on its own branch).
- User documentation (quickstart section, installation, API page, FAQ,
  README and index bullets, agent skill row) and Example 8.

Out of scope, per the PI's decision: automatic initialization policies,
running several fits, returning everything from one call, prior-based
plausible bounds as a user-facing route, vectorized (batched) evaluation
of the PyMC density, discrete or transform-changing-dimension variables,
PyMC 5.x, and any change to VBMC's defaults or numerics. The feasibility
prototype stays as it is under `dev/scripts/` as the record of the check.

## What the plan rests on

Verified on 2026-09-14 by reading the code, by probes in the PyMC
environment that ran the feasibility check (Python 3.12, PyMC 6.3.2,
PyTensor 3.3.1, ArviZ 1.3.0, this checkout and gpyreg 1.2.1 installed
editable, no C++ compiler; its location is listed in the gitignored
`dev/scripts/runs/LOCAL.md`, section "PyMC feasibility check"), and by
reading PyMC's, ArviZ's and PyTensor's published sources, release notes
and package metadata:

- **The target contract.** `FunctionLogger.__call__` hands the target a
  float64 `(D,)` vector in original coordinates and requires a finite real
  scalar back: `-inf`, `nan` and `inf` raise
  `FunctionLogger:InvalidFuncValue` at once
  (`pyvbmc/function_logger/function_logger.py`, the value check after the
  call and the same check in `add`). VBMC never proposes a point on or
  outside the hard box: `ParameterTransformer.inverse` clamps a bounded
  coordinate to the next representable number inside the bound, and the
  effective bounds of `VBMC.__init__` keep the plausible box off the
  edges. A density that is finite on the whole open box is therefore
  safe; a density that is `-inf` one ULP inside a bound (a removed
  two-sided variable feeding a scale or a probability elsewhere in the
  model) would abort a run through the adapter's own error, which names
  the variables.
- **Deep copies.** `IterationHistory.record` deep-copies the function
  logger every iteration, but `FunctionLogger.__deepcopy__` assigns `fun`
  by reference, so the adapter is never copied by the history.
- **Save and load.** `VBMC.save` is `dill.dump(self, recurse=True)`.
  Probe: a `VBMC` built on the prototype adapter (regression model,
  D = 3, a PyMC model and two compiled PyTensor functions on the
  instance) saves in 0.6 s to 366 KB and loads back with
  `function_logger.fun(x)` bit-identical, for a bound-method target and
  for a closure; `copy.deepcopy` of the adapter and `dill` of the adapter
  alone also round-trip. The one failure seen was the prototype keeping
  the `pymc` module itself as an instance attribute, which no shipped
  code will do. The standard-library `pickle` cannot serialize a PyMC
  `Model` (a local function inside PyMC's transform machinery), so
  `dill` is load-bearing here. PyTensor drops a function's compiled
  thunk on pickling and recompiles on unpickling (pymc-devs/pytensor
  issue 2108, open), so a load costs seconds, not a failure.
- **PyTensor's backend.** PyTensor 3's default linker is `auto`, which
  resolves to numba on every platform regardless of a C++ compiler
  (`pytensor/compile/mode.py`); the feasibility environment and the
  probes above therefore ran the numba backend, not the Python
  interpreter (the feasibility report's "Python mode" is a misreading of
  the recorded `linker: auto`, corrected on 2026-09-14). A compiled log
  density there builds in 0.06 s, pays about 0.4 s of JIT on its first
  call and then costs 0.006 ms per call. `PYTENSOR_FLAGS=mode=FAST_COMPILE`
  (or `linker=py`) selects the Python interpreter; `cxx=` does not
  change the linker.
- **Bounds handling in `VBMC.__init__`.** Plausible bounds must be finite
  and strictly inside the hard bounds. `_bounds_check` then adjusts and
  warns: a plausible bound closer than `1e-3` of the range to a finite
  hard bound is moved to that distance (`vbmc:TooCloseBounds`); a
  starting point closer than that to a hard bound is moved inside
  (`vbmc:InitialPointsTooClosePB`); a starting point on or outside a
  plausible bound (`x0 <= plb` or `x0 >= pub`) expands the plausible box
  to it (`vbmc:InitialPointsOutsidePB`); a coordinate bounded on one side
  raises `vbmc:HalfBounds`. Arrays are copied and cast to float64. These
  warnings go through the named logger `VBMC_init`.
- **Dtype canary.** `pyvbmc/testing/_dtype.py` enters only instances of
  `pyvbmc` and `gpyreg` classes; a bound method or a function is a
  boundary. A `VBMC` whose `function_logger.fun` is the adapter's bound
  method is therefore walkable, and the adapter's own object (which holds
  PyTensor variables whose `dtype` is a string) must not be walked; a
  dictionary of its NumPy arrays is, and its arrays are checked.
- **Packaging.** `[tool.setuptools] packages` lists subpackages
  explicitly, but a wheel built from this checkout ships every
  git-tracked subpackage (setuptools_scm's file finder), so listing
  `pyvbmc.pymc` follows the `svbmc` and `calibration` precedent rather
  than being required. `pyvbmc/testing/` is excluded from the wheel and
  ships in the sdist only. `pyvbmc/testing/pymc/` is collected by
  `testpaths` with no registration, and its `.py` files reach the sdist
  through the same finder; `MANIFEST.in` already carries an explicit
  `recursive-include` for one testing subpackage's `.py` files, the
  fallback if the sdist check of Phase 6 finds the new directory missing.
- **ArviZ.** `arviz_base.from_dict(data, *, name, sample_dims,
  save_warmup, index_origin, coords, dims, pred_dims, pred_coords,
  check_conventions, attrs)` returns an `xarray.DataTree`; `arviz.from_dict`
  is the same function in ArviZ 1.x. The existing `to_arviz` passes
  `sample_dims`, `coords` and `attrs`; the structured export adds `dims`.
  ArviZ 1.x and arviz-base 1.x require Python 3.12.
- **PyMC releases.** PyMC 6.0.0 (2026-05-13) requires Python 3.12,
  `arviz >= 1.1, < 2` and PyTensor 3; it replaced `InferenceData` by
  `xarray.DataTree` in every return value and made
  `sample_posterior_predictive` accept a `DataTree` (PyMC 5.x raises
  `TypeError` on one and pins ArviZ below 1.0). `compute_deterministics`
  accepts a `DataTree` from 6.3.0 (2026-08-12; before, its `posterior`
  `Dataset`) and returns a `Dataset`. Each PyMC minor moves its PyTensor
  pin (6.3.2 requires `pytensor >= 3.2.2, < 3.4`). The latest release at
  the time of writing is 6.3.2 (2026-09-08).
- **PyMC surface used**, and its status in the public reference:
  documented: `Model.compile_logp(jacobian=)`, `Model.initial_point()`,
  `Model.coords`, `remove_value_transforms` (public since 5.8.0, exported
  by `pymc.model.transform` and `pymc.model.transform.conditioning`;
  works inside and outside a model context, accepts a variable that has
  no transform, preserves `named_vars_to_dims` and `coords`, and returns
  a model whose `free_RVs` may be in a different order), `pymc.draw`
  (`random_seed=` takes a `Generator` and leaves NumPy's global state
  untouched), `Model.compile_dlogp(vars=, jacobian=)` and
  `Model.compile_d2logp(vars=, jacobian=, negate_output=)` (the exact
  gradient and Hessian of the log density with respect to the value
  variables of `vars`, in the order of `vars`, each variable flattened;
  the Hessian is the Jacobian of the symbolic gradient, a
  `pytensor.map` over the gradient's components, so one call costs about
  `D` gradient passes; `negate_output=True`, the default, negates the
  result and emits a `FutureWarning`), `compute_deterministics`,
  `sample_posterior_predictive`, the transform classes `LogTransform`,
  `LogOddsTransform`, `IntervalTransform` (`pymc.logprob.transforms`) and
  `Interval` (`pymc.distributions.transforms`, a subclass of
  `IntervalTransform` and the class PyMC attaches), all importable from
  `pymc.distributions.transforms`, with `forward(value, *inputs)` and
  `backward(value, *inputs)` whose `*inputs` are the variable's
  `rv.owner.inputs` by PyMC's own convention; undocumented but used by
  PyMC's own utilities: the instance attributes `Model.free_RVs`,
  `rvs_to_values`, `rvs_to_transforms`, `named_vars_to_dims`,
  `deterministics`, `observed_RVs`, and `IntervalTransform.args_fn` (the
  attribute; `Interval`'s constructor keyword is `bounds_fn`); private:
  `pymc.distributions.transforms._default_transform`, a `singledispatch`
  documented only as an extension point for distribution implementers,
  and the only way to learn a distribution's default transform.
- **Not used: `constrain_values` and `unconstrain_values`.** PyMC 6.1
  added these public functions (`pymc.model.transform_values`) to map an
  `xarray.Dataset` between a model's transformed and constrained values.
  On PyMC 6.3.2 they fail under the default numba linker, PyMC's own
  docstring example included (`transform_values.py` hands `DataArray`
  objects to a function compiled with `trust_input=True`, which numba
  cannot type); they work only under `compile_kwargs={"mode":
  "FAST_COMPILE"}`, recompile a fresh graph on every call and cannot map
  a single point (`sample_dims=[]` raises). The adapter therefore keeps
  the prototype's compiled `transform.forward` and `transform.backward`
  maps, which the feasibility check validated on five models.
- **PyTensor's float32 constants.** A Python float that float32 holds
  exactly is stored as a float32 constant even under `floatX = float64`
  (documented autocasting; `autocast_float_as("float64")` at model
  construction changes it). This is the constant offset of about 1e-8
  nats per constant-scale term the feasibility report measured; it is a
  property of how the user wrote the model and the adapter cannot remove
  it, so density checks compare the shape of the density and allow the
  offset.
- **Precomputed evaluations in `VBMC`.** `x0` may have `n0` rows, and the
  advanced option `f_vals` (length `n0`) supplies their values;
  `_init_optim_state` stores both in `optim_state["cache"]` (a length
  mismatch raises `vbmc:MismatchedStartingInputs`). The initial design
  (`active_sample` before the first GP, `fun_eval_start = max(D, 10)`
  points) takes the cached rows, evaluates those whose value is `nan`,
  and fills the remainder with uniform draws in the plausible box. Rows
  beyond `fun_eval_start` are dropped from the cache (the first ones are
  kept; MATLAB's clustering selection is not ported). `_bounds_check`
  expands the plausible box to contain every row of `x0`
  (`vbmc:InitialPointsOutsidePB`) and moves a row closer than `1e-3` of
  the range to a finite hard bound inside without changing its entry in
  `f_vals`, after which that value no longer belongs to its point.
  Supplied values enter through `FunctionLogger.add`, which raises
  `cache_count`, not `func_count`, so they do not count towards
  `max_fun_evals` or the reported `func_count`.

## Design

### Public surface

```python
from pyvbmc import VBMC, PyMCTarget          # PyMCTarget resolves lazily
target = PyMCTarget(model)                    # or PyMCTarget(model, plausible_bounds=..., start=..., seed=...)
print(target)                                 # variable, VBMC coordinate(s), hard and plausible bounds
vbmc = VBMC(target.log_joint, target.x0, target.lb, target.ub, target.plb, target.pub)
                                              # plus the setup evaluations, by the route settled before approval
vp, results = vbmc.optimize()
data = target.to_arviz(vp, n_samples=2000)    # DataTree over the model's variables
pm.compute_deterministics(data, model=model)  # a Dataset of the deterministics
pm.sample_posterior_predictive(data, model=model)
```

`pyvbmc/pymc/__init__.py` exports `PyMCTarget` and `UnsupportedModel`
(a `ValueError`). Importing `pyvbmc` or `pyvbmc.pymc` imports neither
PyMC nor PyTensor; constructing a `PyMCTarget` without PyMC raises an
`ImportError` naming `pyvbmc[pymc]`, as `SVBMC` does for `pyvbmc[torch]`.

`PyMCTarget(model, *, plausible_bounds=None, start=None, seed=None)`:

- `model`: a `pymc.Model`. Its free variables, in the order of the given
  model's `free_RVs` (the partially untransformed model may list them in
  another order; the adapter looks them up by name), become consecutive
  blocks of the flat vector, each flattened in C order.
- `plausible_bounds`: `None` (the default) computes the Laplace box
  below. Otherwise a mapping from every free variable's name to a pair
  `(lower, upper)` in the model's own variables, each broadcastable to
  the variable's shape and strictly inside the variable's support as
  recorded in `support` (checked for every variable, kept or not; a
  violation raises `ValueError` naming the variable). The adapter maps a
  kept variable's pair to value space through `forward` and sorts it per
  coordinate, since a one-sided interval transform bounded above is
  decreasing. The mapping must cover every free variable (a missing or
  unknown name raises `ValueError` naming it).
- `start`: `None` (the default) takes the best point of the mode search
  below. Otherwise a mapping from every free variable's name to a value
  in the model's variables, strictly inside the support, mapped to VBMC's
  coordinates the same way; no search runs.
- `seed`: an `int`, a `numpy.random.Generator` or `None`, through
  `pyvbmc.rng.get_rng`. Only the prior draws (the curvature fallback, the
  location check and the route without gradients) use it
  (`pymc.draw(random_seed=...)`); the mode search and the Hessian are
  deterministic. A `PyMCTarget` never reads or writes NumPy's global
  state.

Attributes, all float64 NumPy arrays of shape `(1, D)` where they are
arrays: `x0`, `lb`, `ub`, `plb`, `pub`; `D`; `names` (free variables in
the given model's order), `shapes` (tuples), `sizes`; `value_names`
(PyMC's value-variable name per free variable: the variable's own name,
or `sigma_log__` and the like for a kept one); `coordinate_names` (D
strings, one per flat coordinate, built from the value name and the
C-order multi-index of the coordinate within the variable: a scalar
gives the value name alone, `mu` or `sigma_log__`; an array gives the
name followed by the index in brackets, `u[0]`, `w[1,2]`,
`tau_log__[0]`; an axis whose dimension has declared coordinates uses
the coordinate label in place of the index on that axis,
`beta[intercept]`); `kept` (variable name to transform class name, for
the variables whose transform is kept); `support` (variable name to the
`(lower, upper)` arrays of the hard bounds in the model's variables, of
the variable's shape); `plausible_info` (a dictionary with exactly the
keys `route`, one of `"laplace"` (the box from the Hessian),
`"prior"` (the model has no gradient, the box from prior quantiles) or
`"explicit"`; `start`, one of `"mode"` (the best point of the search),
`"initial"` (the model's initial point, when there is no gradient) or
`"user"`; `mode`, the best point of the search as a dictionary over the
model's variables when the search ran, else `None`; `n_evaluations`, the
log-density evaluations spent, the Hessian counted as `D`; `cap_reached`,
whether the search stopped at its budget; `start_moved`, `curvature`,
`relocated` and `clipped`, lists of coordinate names, empty when the
corresponding step did not run or changed nothing);
`setup_evaluations` (the pair `(X, y)`: the `(n, D)` points in VBMC's
coordinates at which the setup evaluated `log_joint`, the search's calls
and the finiteness check, and the `(n,)` finite values, in call order;
how they reach `VBMC` is settled before approval); `model` (the model as
given; the partially untransformed model the adapter compiles is
private).

Methods:

- `log_joint(x)`: the callable for `VBMC`. Takes a `(D,)` or `(1, D)`
  array in VBMC's coordinates, returns a Python `float`: the model's log
  density in its own variables plus the log-Jacobian of every kept
  transform (`compile_logp(jacobian=True)` on the partially untransformed
  model), so that the ELBO estimates the model's evidence. On or outside
  the hard box (`x <= lb` or `x >= ub` in any coordinate, as in the
  prototype) it returns `-inf`, the density's value there, which VBMC
  never requests. Inside the box a non-finite density raises `ValueError`
  naming the model variables and their values at the point, so that the
  failure names the model's parameters instead of a flat vector.
- `log_joint_no_jacobian(x)`: the model's log density in its own
  variables at a point of VBMC's coordinates, without the Jacobian
  terms (`compile_logp(jacobian=False)`; the prototype's
  `log_density_plain`). The tests use the difference of the two.
- `flatten(values)` and `unflatten(x)`: a dictionary over `value_names`
  (arrays of the value variables' shapes) to the flat `(D,)` vector and
  back, as in the prototype.
- `to_model_variables(X)`: `(n, D)` draws in VBMC's coordinates to a
  dictionary of `(n, *shape)` arrays over the model's variables, kept
  variables mapped through `backward`. `from_model_variables(point)`: a
  dictionary over the model's variables, one value per variable of the
  variable's shape, to the flat `(D,)` vector, kept variables mapped
  through `forward`; what `start` and each side of `plausible_bounds`
  go through.
- `to_arviz(vp, n_samples=1000)`: checks `vp.D == target.D` (else
  `ValueError`) and the ArviZ import guard, then draws `vp.sample(n_samples,
  orig_flag=True)`, maps the draws with `to_model_variables`, and
  packages the result with the shared helper of the structured export:
  one variable per free variable with dimensions `(chain, draw, *dims)`
  and shape `(1, n_samples, *shape)`, the model's dimension names and
  coordinates, and the attributes of the existing export with
  `parameter_space="model"`. It advances `vp.rng` exactly as one `sample`
  call does, and not at all when it raises.
- `__str__`: a table with one row per free variable: name, shape, VBMC
  coordinate names, whether the transform is kept, hard bounds,
  plausible bounds, start.

### Coordinates and bounds

Unchanged from the prototype, which every accepted and rejected model of
the feasibility check exercised. Per free variable, in the given model's
order:

1. A non-floating dtype raises `UnsupportedModel` (`k is int64: VBMC
   needs continuous parameters`).
2. The transform PyMC attached (`rvs_to_transforms`) gives the hard
   bounds of every coordinate in the model's variables: `LogTransform`
   means `(0, inf)`; `LogOddsTransform` means `(0, 1)`; an interval
   transform gives its two limits through `args_fn(*rv.owner.inputs)`,
   evaluated per coordinate, a missing limit being infinite; a limit with
   a random variable among its ancestors raises `UnsupportedModel` (`an
   interval limit depends on another random variable`); any other
   transform raises `UnsupportedModel` (`unsupported transform
   <ClassName>`). No transform means unbounded only when
   `_default_transform(rv.owner.op, rv)` is `None` as well; a transform
   suppressed at construction raises `UnsupportedModel` (`its
   <ClassName> was suppressed at construction`).
3. A variable that has no transform, or whose coordinates are all
   two-sided, is untransformed (`remove_value_transforms(model,
   vars=[...])`) and handed over in its own coordinates with the interval
   as hard bounds; a variable with a transform whose coordinates are all
   one-sided keeps its transform, travels as PyMC's value variable,
   unbounded, and contributes its Jacobian; any other combination (a
   transformed variable with an unbounded coordinate, or coordinates that
   mix one-sided with two-sided or unbounded) raises `UnsupportedModel`
   (`its coordinates mix one-sided and other bounds`).
4. Shapes come from the initial point of the partially untransformed
   model, read at the value names; the flat vector is the concatenation
   in the given model's order.

The maps between value space and the model's variables for the kept
variables are PyTensor functions compiled once at construction from
`transform.backward(v, *rv.owner.inputs)` and `transform.forward(v,
*rv.owner.inputs)` with a leading axis of draws, as in the prototype.

### Starting point and plausible box

The setup spends at most `20 + 5 D` evaluations of the log joint (25 at
`D = 1`, 70 at `D = 10`, 120 at `D = 20`), the Hessian counted as `D`;
one, the finiteness check, when the model has no gradient or the user
gives both `start` and `plausible_bounds`. Every evaluation is recorded
in `setup_evaluations`. The budget, the stopping rule and the width of
the location check are provisional until the setup probe of
[Before approval](#before-approval-the-setup-probe-and-evaluation-reuse)
has run. The default route, `laplace`:

1. **Search start.** The initial point of the partially untransformed
   model (`initial_point()`, read at the adapter's value names), moved by
   the start-point rule of step 5.
2. **Mode search.** L-BFGS-B (`scipy.optimize.minimize` with `jac=True`)
   on `-log_joint` in VBMC's coordinates, through one compiled function
   of the partially untransformed model returning the log joint and its
   gradient (`logp(jacobian=True)` and `dlogp(vars=<free variables in
   the adapter's order>, jacobian=True)`), with the start-point rule's
   interval as bounds for a coordinate with finite hard bounds and none
   otherwise, at most `19 + 4 D` calls, and a loose stopping rule (the
   box needs a point well inside the posterior, not the mode to many
   digits). The search maximizes the target VBMC sees, Jacobian
   included, so for a kept variable the point is the stationary point of
   the adapter's log joint. `x0` is the best point found. A search that
   exhausts its calls sets `cap_reached`; the steps below run at the best
   point.
3. **Curvature.** The exact Hessian at `x0` (`compile_d2logp(vars=...,
   jacobian=True, negate_output=False)`), counted as `D` evaluations.
   The box is `x0 ± 3` marginal standard deviations from the inverse of
   the negative Hessian.
4. **Fallbacks without evaluations.** The prior box below is the 5 % to
   95 % quantiles of 4000 prior draws (`pymc.draw`, seeded by `seed`,
   mapped to value space).
   - *Curvature*: a coordinate whose marginal variance is not finite and
     positive (a ridge, a saddle, a singular Hessian), and every
     coordinate when the graph has a gradient but no second derivative,
     takes as its standard deviation half the width of the prior box
     divided by 3; listed in `curvature`.
   - *Location*: a coordinate of `x0` outside the prior's `q` to `1 - q`
     quantile interval (`q` from the probe) means the search ran away,
     as it does on a density without a mode (a centered hierarchical
     model whose group scale goes to zero). That coordinate of `x0` moves
     to the midpoint of the prior box, takes the prior width, and is
     listed in `relocated`.
   - *No gradient*: when the gradient cannot be built (a custom `Op`
     without `grad` in the graph), no search and no Hessian run: `route`
     is `"prior"`, `start` is `"initial"`, `x0` is the model's initial
     point after the start-point rule, the box is the prior box, and a
     warning recommends explicit `plausible_bounds`.
5. **Start-point rule and clipping.** A coordinate of `x0` (the search
   start, the best point, a relocated point or a user-supplied `start`)
   closer than `2e-3` of the range to a finite hard bound, or on it (a
   mode of a bounded density on its boundary maps back onto the bound),
   is moved to that distance and listed in `start_moved`; the search's
   bounds are the same interval, so no search point is one that
   `VBMC.__init__` would move. Each plausible bound is kept at least 1 %
   of a finite range inside its hard bound, and a coordinate whose
   plausible interval collapses or falls outside the box is widened to
   the middle 98 % of the box; the coordinates touched are listed in
   `clipped`. Where the 1 % margin would put a plausible bound at or
   beyond `x0` (a start point within 1 % of a hard bound), that bound is
   placed halfway between the hard bound and `x0` instead, which is at
   least `1e-3` of the range inside the hard bound and strictly on the
   far side of `x0`.
6. **Finiteness.** `log_joint(x0)` must be finite; the search's value is
   reused when `x0` is its best point, and otherwise the check is one
   more evaluation, which the search's `19 + 4 D` leaves room for. A
   non-finite value raises `ValueError` naming the variables and values,
   before a user reaches VBMC's less specific error.

With these rules none of the `VBMC.__init__` adjustments (`TooCloseBounds`,
`InitialPointsTooClosePB`, `InitialPointsOutsidePB`) fires on an adapter
target, and `test_plausible.py` asserts it. The lists (`start_moved`,
`curvature`, `relocated`, `clipped`), `cap_reached` and the `prior`
route are reported through the logger `logging.getLogger("pyvbmc.pymc")`
at `WARNING` (Python's last-resort handler shows warnings when no logging
is configured, and a configured root logger receives them), because each
is worth the user's attention. With `plausible_bounds` given, no Hessian
is computed and no prior draws happen; with `start` given, no search
runs; the clipping and the start-point rule apply in every case.

A model whose observed data are `pm.Data` containers reads them when the
compiled functions are called: `pm.set_data` after construction changes
`log_joint` and leaves the box and `setup_evaluations` describing the old
data. The documentation says to build a new target after changing data.

### The structured ArviZ export

`VariationalPosterior.to_arviz(self, n_samples=1000, *, var_names=None,
orig_flag=True, variables=None, dims=None, coords=None)`:

- `variables`: a `Mapping` (a `dict` in practice; its iteration order is
  the layout) from variable name to shape, an `int` or a sequence of
  `int`s converted to a tuple, `()` for a scalar. The sizes must sum to
  `D`; the variables take consecutive column blocks of the draws in
  mapping order, each reshaped in C order to `(1, n_samples, *shape)`.
  Names obey the rules `var_names` obeys today (non-empty, unique, not
  `chain` or `draw`) and must differ from every dimension name. Giving
  both `var_names` and `variables` raises `ValueError`.
- `dims`: a mapping from variable name to a sequence of dimension names,
  of length equal to the variable's number of axes; only for names in
  `variables`. Dimension names must not be `chain` or `draw`.
- `coords`: a mapping from dimension name to its coordinate values, of
  length equal to that axis in every variable that uses the dimension;
  a dimension no variable uses raises `ValueError`.
- `dims` or `coords` without `variables` raises `ValueError`.
- Order inside the method: argument validation (`ValueError`), then the
  import guard (`ImportError` for Python below 3.12 or a missing
  `arviz_base`), then the draw. Every error therefore leaves the
  generator untouched, as today (`test_vp_export_dependencies.py`
  asserts it for the import guard and `test_vp_arviz.py` for the
  arguments).
- Without `variables`, `dims` and `coords` the method behaves exactly as
  today; the existing tests are unchanged.

The `from_dict` call and its import guard move into a private helper
module `pyvbmc/variational_posterior/_arviz.py`: `require_from_dict()`
performs the guard and returns `from_dict`; `datatree_from_arrays(arrays,
*, dims=None, coords=None, attrs=None)` takes a mapping of name to
`(1, n_samples, *shape)` array (the sample count is read from the
arrays), the `dims` and `coords` mappings, and the attribute dictionary
of the `posterior` group, and returns the `DataTree`. `to_arviz` and
`PyMCTarget.to_arviz` build their trees through it. The `parameter_space`
attribute gains the value `"model"` for the adapter's export, documented
on the method's page next to `"original"` and `"internal"`.

### Version range and guards

The extra is `pymc = ["pymc >= 6.3; python_version >= '3.12'"]` together
with the two entries of the `arviz` extra repeated verbatim with a
comment saying so (the export imports `arviz_base` directly, and PyMC
declares only the `arviz` metapackage). The floor is the oldest PyMC on
which the documented workflow above works as written
(`compute_deterministics` on the exported `DataTree`) and the version the
adapter was tested on; 6.0 to 6.2 would need the `posterior` group as a
`Dataset` and were not tested. There is no upper bound, following the
`torch` and `arviz` extras: the CI cell installs the newest PyMC that
satisfies the extra, so a release that breaks the adapter shows up at
the next scheduled run (and in PR runs) with the message of the guard it
tripped. `pyvbmc/pymc/_compat.py` holds the guarded imports and one
constant, `TESTED_RANGE`, that the error messages and the installation
page quote; Phase 2 sets it to `"PyMC 6.3 with ArviZ 1.3"` and Phase 6 is
its only later editor.

The adapter reaches past PyMC's documented surface in four places. Each
is guarded by capability, not by version number, so that an incompatible
PyMC fails at construction with a message naming the tested range rather
than misbehaving. A guard that finds PyMC's surface changed raises
`ImportError` quoting `pymc.__version__` and `TESTED_RANGE`; a guard that
finds a particular model outside the supported scope raises
`UnsupportedModel` naming the variable.

1. **Transform classes.** `LogTransform`, `LogOddsTransform`, `Interval`
   and `IntervalTransform` are imported from
   `pymc.distributions.transforms` at construction; an `ImportError`
   becomes the guard's `ImportError`. Dispatch is by exact type identity
   (`type(transform) in {...}`), never by class name, so a subclass or a
   look-alike is rejected as unsupported instead of being read as a
   known transform. PyMC's transform module changes between minors,
   which is why the guard runs on every construction.
2. **`args_fn`, `forward` and `backward` with `rv.owner.inputs`.**
   `args_fn` is an undocumented attribute and the `*inputs` convention is
   PyMC's own. A missing `args_fn` attribute on an interval transform is
   the guard's `ImportError`; a `TypeError` from calling any of the three
   is `UnsupportedModel` naming the variable; each limit must broadcast
   to the variable's shape. At construction the compiled maps of every
   kept variable are checked at the initial point against the closed
   forms (`exp` for a log transform, `lower + exp` and `upper - exp` for
   a one-sided interval) to `1e-10`, and `forward(backward(v))` against
   `v`; a mismatch is the guard's `ImportError`, since it means PyMC's
   parameterization of a value variable changed.
3. **`_default_transform`.** If the private function cannot be imported,
   the guard's `ImportError`. The alternative, treating a missing
   transform as unbounded, was rejected because a suppressed transform
   would then hand VBMC a density that is `-inf` on part of the box, the
   failure the check exists to prevent.
4. **The undocumented `Model` instance attributes** (`free_RVs`,
   `rvs_to_values`, `rvs_to_transforms`, `named_vars_to_dims`): one
   `hasattr` check on the model instance at construction, the guard's
   `ImportError` when one is missing.

### Save and copy

Nothing in `VBMC.save`, `VBMC.load` or `IterationHistory` changes. The
adapter keeps only picklable state (NumPy arrays, the PyMC model, the
partially untransformed model, compiled PyTensor functions, the
generator) and no module objects; `dill` pickles it once because the
history shares `fun`. The documentation notes that a saved run carries
the PyMC model and its compiled functions, loads only under a compatible
PyMC and PyTensor (the caveat `VBMC.save` already states for complex
attributes), and recompiles the density on loading, which takes seconds.

### Layout

```
pyvbmc/pymc/__init__.py                       PyMCTarget, UnsupportedModel
pyvbmc/pymc/_compat.py                        guarded imports, TESTED_RANGE
pyvbmc/pymc/_target.py                        PyMCTarget: mapping, log joint, export
pyvbmc/pymc/_plausible.py                     mode, Hessian, fallbacks, clipping
pyvbmc/variational_posterior/_arviz.py        shared DataTree packaging and import guard
pyvbmc/testing/pymc/                          tests (see Phase 2)
docsrc/source/api/classes/pymc_target.rst     API page
examples/pyvbmc_example_8_pymc.ipynb          Example 8 (+ scripts/pyvbmc_example_8_full_code.py)
```

`pyproject.toml`: `pyvbmc.pymc` in `packages`; the `pymc` extra.

## Before approval: the setup probe and evaluation reuse

Two investigations settle what the setup design leaves provisional. Both
run before the plan is approved, both are the orchestrator's, and
neither changes the package; their answers are written into the design,
Phase 2 and the decisions, and summarized in the execution record. They
run in the PyMC environment named by `LOCAL.md`, from one script,
`dev/scripts/pymc_setup_probe.py`, with raw outputs under
`dev/scripts/runs/pymc_setup_probe_<date>/`, tracked copies under
`dev/experiments/pymc_setup_probe/` and a report
`dev/results/<date>-pymc-setup-probe.md`.

### Part A: the evaluation budget

**Question.** Does `20 + 5 D` evaluations (a search of `19 + 4 D` calls,
the Hessian counted as `D`, one finiteness check) give a box close to the
one at the converged mode, and which stopping rule and location-check
width `q` should the search use?

**Method.** Part A counts evaluations and takes seconds per model. For
each model the script builds the prototype's `PyMCTarget` for the
coordinate mapping, compiles the log joint with its gradient and the
exact Hessian on the partially untransformed model (`jacobian=True`), and
runs L-BFGS-B from the model's initial point to tight convergence (at
most 2000 calls), logging every call. Models: the five accepted models of
the feasibility check; the eight-schools model, centered (a density
without a mode as the group scale goes to zero) and non-centered; a
linear regression with uncentred covariates on very different scales
(`D = 5`, badly conditioned); a logistic regression with 20 correlated
coefficients; a non-centered varying-intercept model with `D = 20`; and a
model whose likelihood is a custom `Op` without `grad`.

**Recorded per model.** The calls until the best point so far is within
1 and 0.1 nats of the optimum, and within 1 and 0.3 standard deviations
of it (Mahalanobis distance under the exact Hessian at the optimum); at
`19 + 4 D` calls, that distance, the largest shift of the box centre in
the optimum's marginal standard deviations and the range of the ratios of
the marginal standard deviations; whether the Hessian at the optimum is
negative definite; which coordinates of the best point fall outside the
prior's quantile intervals for `q` in 0.005, 0.01 and 0.05; the calls
each candidate stopping rule (relative and absolute improvement in nats,
projected gradient) would have stopped at; compilation and search
times; and, for the custom `Op`, that the gradient cannot be built and
how PyTensor reports it.

**Decides.** The budget (kept, or changed with the evidence), the
stopping rule (the loosest that keeps the box centre within 0.3 marginal
standard deviations of the converged one on every model with a mode),
`q` (the widest that relocates nothing on the models with a mode and
catches the centered eight-schools model), and the expectations
`test_plausible.py` asserts.

### Part B: handing the setup evaluations to VBMC

**Question.** How should the `setup_evaluations` of a target reach
`VBMC`, and does it help?

**What exists.** `VBMC` accepts precomputed values through a
multi-row `x0` and the advanced option `f_vals` ("What the plan rests
on"). For this use it has four limits: rows beyond `fun_eval_start`
(`max(D, 10)`) are dropped, where the search yields up to `19 + 4 D`; the
plausible box is expanded to contain every row, so the early points of a
search path far from the mode would widen it; a row within `1e-3` of the
range of a finite hard bound is moved without its value, a defect for any
caller of `f_vals` that the adapter avoids by keeping its search inside
the start-point rule's interval; and supplied values count towards
neither `max_fun_evals` nor the reported `func_count`.

**Options.**

1. No change to `VBMC`: the adapter's documentation passes the setup
   points inside the plausible box as `x0` (best point first), their
   values as `f_vals`, and `fun_eval_start` raised to the number of
   points; points outside the box are dropped.
2. An extension of `VBMC`'s interface that seeds the function logger with
   evaluated points as training data, independent of `x0`, of the
   plausible box and of `fun_eval_start` (a constructor argument such as
   `initial_evaluations=(X, y)`), with a documented rule on whether they
   count towards `max_fun_evals`, and the moved-row defect fixed. Unused,
   it must leave every oracle (`--check --exact`) and the golden replay
   bit-identical. It would be implemented and reviewed on its own branch
   before Phase 2, with its own tests and documentation.
3. Option 1 or 2, plus a method on the target that returns the
   arguments for `VBMC` (`x0`, bounds, the evaluations, the options), so
   that the documented call cannot pass them inconsistently.

**Also to decide.** Whether supplied evaluations count towards the
budget; whether VBMC's uniform initial design still runs in full on top
of them (a search path clusters along its way to the mode and is not
space-filling) or is shortened by their number; whether points far below
the best value (the first calls of a path started in the tail) are
passed at all.

**Method.** Part B runs VBMC and takes the heavy slot. On the vector
model, the non-centered eight-schools model and the badly conditioned
regression, with five seeds each, three arms at equal total evaluations,
the setup's included: no reuse (the setup's evaluations discarded, VBMC's
default initial design); option 1 as documented (the points inside the
box, the uniform design shortened by their number); every setup point
with VBMC's full uniform initial design on top, which only option 2
allows and the probe script emulates by filling the function logger
before the first iteration, without changing the package. Scored by the
ELBO's error where the evidence is known and by posterior distances
against a long NUTS reference, with VBMC's iterations to stability and
wall time.

**Decides.** The PI chooses among the options on the recorded evidence;
the design's public surface, Phase 2's steps and tests, Phase 4's
documentation and Example 8's call are revised to the choice.

## Phases

Executors: the orchestrating Fable session keeps the design, the
integration and the heavy verification; Opus sub-agents implement the
phases whose steps are spelled out below. At most one heavy process runs
at a time. Sub-agents run only the test files they own, in the PyMC
environment named by `LOCAL.md`, one sub-agent at a time
except where a phase says otherwise; the full suite runs once, in
Phase 6. Every phase ends with the formatting hooks (black 79, isort,
pycln, black-jupyter) and conventional-commit messages: `feat(pymc):`
and `test(pymc):` for the adapter, `feat(vp):` for the export,
`build:` for `pyproject.toml` and `MANIFEST.in`, `ci:` for the workflow,
`docs:` for user documentation and the notebook, `docs(dev):` for the
records.

### Phase 0: branch and records

**Executor**: Fable (orchestrator).
**Goal**: the branch exists and the plan's status says so.

The plan, the status pointers (the PyMC item in `dev/TODO.md`, the plans
list of `dev/README.md`, the header of the proposal) and the correction
of the feasibility report's environment description were committed on
`dev-next` on 2026-09-14, when the plan was handed off pending approval.
The investigations of
[Before approval](#before-approval-the-setup-probe-and-evaluation-reuse)
run first, and their answers are written into the design, the phases and
the decisions. On approval:

1. Set this plan's status line to in progress and commit it on
   `dev-next` (`docs(dev):`).
2. `git switch -c dev-pymc-adapter`.

### Phase 1: the structured ArviZ export

**Executor**: Opus sub-agent.
**Goal**: `VariationalPosterior.to_arviz` accepts `variables`, `dims`
and `coords`; the packaging lives in a helper the adapter reuses.

**Steps**:

1. Create `pyvbmc/variational_posterior/_arviz.py` with
   `require_from_dict()` (the Python 3.12 check and the guarded
   `from arviz_base import from_dict` currently inside `to_arviz`, with
   their messages unchanged, returning `from_dict`) and
   `datatree_from_arrays(arrays, *, dims=None, coords=None, attrs=None)`
   as specified in the design section (`sample_dims=["chain", "draw"]`,
   `coords` extended with `chain: [0]` and `draw: arange(n)`, `dims`, and
   `{"posterior": attrs}` passed to `from_dict`). Import the module
   inside `to_arviz`, not at module level.
2. In `variational_posterior.py`, extend `to_arviz` with the three
   keyword-only parameters. A private `_structured_layout(D, variables,
   dims, coords)` validates and returns the list of `(name, offset,
   shape, dims)`; every error is a `ValueError` whose message contains
   the parameter's name. Order: `n_samples` check, `var_names` or
   `variables` validation, `require_from_dict()`, `self.sample`, then
   the arrays and `datatree_from_arrays`. With `variables=None` the
   output is unchanged.
3. Update the docstring (Parameters, Raises, Notes: the structured layout
   takes consecutive column blocks in C order; the export is independent
   of the posterior's arrays afterwards; `parameter_space` values).
4. Tests in `pyvbmc/testing/variational_posterior/test_vp_arviz.py`
   (existing tests untouched): a D = 3 posterior exported as
   `variables={"beta": (2,), "sigma": ()}` with `dims={"beta": ["coef"]}`
   and `coords={"coef": ["a", "b"]}` gives `beta.dims == ("chain",
   "draw", "coef")`, shape `(1, n, 2)`, `sigma.dims == ("chain",
   "draw")`, coordinates `["a", "b"]`, values equal to the matching
   columns of a fresh `sample` on an identical posterior, and the
   generator state equal afterwards; a matrix variable `(2, 2)` on a
   D = 4 posterior reshapes in C order (`values[0, :, i, j] ==
   samples[:, 2 i + j]`); a shape given as a list is accepted;
   `orig_flag=False` works; `var_names` together with `variables`,
   sizes not summing to `D`, a `dims` entry of the wrong length or for
   an unknown variable, a `coords` entry of the wrong length or for an
   unused dimension, a variable name equal to a dimension name, and
   `dims` or `coords` without `variables` each raise `ValueError`
   mentioning the parameter and leave the generator state unchanged.
5. `docsrc/source/api/methods/variational_posterior_to_arviz.rst`: a
   paragraph on the structured layout and the `parameter_space` values;
   `docsrc/source/quickstart.rst`, subsection "ArviZ DataTree": a short
   second example with `variables`, `dims`, `coords`.
6. Run the formatting hooks and, in the PyMC environment,
   `python -m pytest pyvbmc/testing/variational_posterior/test_vp_arviz.py pyvbmc/testing/variational_posterior/test_vp_export_dependencies.py -vv`
   (the project venv has no ArviZ and skips the first file; the second
   runs everywhere and pins the guard order).

**Verification**:
- [ ] Both test files pass, the existing tests unchanged.
- [ ] `git diff` of `variational_posterior.py` touches only `to_arviz`
  and its docstring.

### Phase 2: the adapter and its tests

**Executor**: Opus sub-agent, after Phase 1.
**Goal**: `pyvbmc.pymc.PyMCTarget` as designed, tested against
hand-written densities.

**Steps**:

1. `pyvbmc/pymc/_compat.py`: `TESTED_RANGE = "PyMC 6.3 with ArviZ 1.3"`;
   `import_pymc()` (imports `pymc`, `pytensor`, `pytensor.tensor`;
   `ModuleNotFoundError` for those names becomes `ImportError("PyMC
   targets require the pymc extra; install pyvbmc[pymc] on Python
   3.12+.")`); `transform_classes()`, `default_transform()`,
   `remove_value_transforms()` and `check_model(model)` (the four guards
   of the design section, each raising `ImportError` that quotes
   `pymc.__version__` and `TESTED_RANGE`); `closed_form(kind, lower,
   upper)` returning the NumPy `backward` map the construction-time
   check compares against.
2. `pyvbmc/pymc/_target.py`: port `PyMCTarget` from the prototype with
   these changes: no module stored on the instance; dispatch by type
   identity through `_compat`; the `args_fn`/`forward`/`backward`
   guards and the construction-time closed-form check; the order of
   `names` taken from the given model; `value_names` and
   `coordinate_names`; `plausible_bounds` (support check for every
   variable, mapping, per-coordinate sort), `start` and `seed`; the
   compiled log joint with its gradient and the compiled Hessian
   (`compile_d2logp`), both with the free variables in the adapter's
   order, and the detection of a graph without a gradient or without a
   second derivative (the exception PyTensor raises while building the
   graph, caught narrowly and turned into the `prior` route or the
   curvature fallback); the evaluation counter and `setup_evaluations`;
   the start-point rule and the finiteness check at `x0`; `log_joint`
   accepting `(1, D)` and raising the named `ValueError` on a non-finite
   density inside the box; `log_joint_no_jacobian`, `flatten`,
   `unflatten`, `to_model_variables`, `from_model_variables`;
   `to_arviz(vp, n_samples)` in the order of the design section through
   `datatree_from_arrays`; `plausible_info` with exactly the nine keys;
   the `pyvbmc.pymc` logger; `__str__`. Numpydoc docstrings in the
   `VariationalPosterior` style, with Raises sections.
3. `pyvbmc/pymc/_plausible.py`, NumPy and SciPy over callables, no PyMC
   import: `move_inside(x0, lb, ub, fraction=2e-3) -> (x0, moved_mask)`;
   `search_mode(value_and_grad, x_start, lb, ub, max_calls) -> (x_best,
   X, y, cap_reached)` (L-BFGS-B with the start-point rule's interval as
   bounds, the stopping rule set by the probe, every call's point and
   finite value returned in call order, the best point by value);
   `marginal_sd(H) -> (sd, usable_mask)` (inverse of `-H`, `sd` finite
   and positive where usable; a `LinAlgError` makes nothing usable);
   `quantile_box(draws, quantiles=(0.05, 0.95)) -> (plb, pub)` on `(n,
   D)` value-space draws; `relocate(x0, draws, q) -> (x0, moved_mask)`
   (the location check); `clip_inside(plb, pub, lb, ub, x0,
   margin=0.01) -> (plb, pub, clipped_mask)` implementing the clipping
   including the halfway rule; `laplace_box(x0, hessian, lb, ub,
   prior_draws, k=3.0) -> (x0, plb, pub, curvature_mask,
   relocated_mask, clipped_mask)` composing them, where `hessian()`
   returns the `(D, D)` Hessian at `x0` or `None` when the graph has no
   second derivative, and `prior_draws()` is a callable returning the
   `(4000, D)` value-space prior draws, called at most once and only
   when a coordinate needs a fallback or the location check. `_target.py`
   turns the masks into coordinate-name lists and logs them. The
   budget, the stopping rule and `q` are module constants with the
   values the probe settled.
4. `pyvbmc/pymc/__init__.py` (docstring naming the extra and the lazy
   import) and the `PyMCTarget` branch in `pyvbmc/__init__.py`'s
   `__getattr__` and `__dir__`.
5. `pyproject.toml`: `pyvbmc.pymc` in `packages`; the `pymc` extra.
6. Tests under `pyvbmc/testing/pymc/` with an `__init__.py`; every file
   except `test_imports.py` begins with `pytest.importorskip("pymc")`
   and `pytest.importorskip("arviz_base")`. Every random draw is
   seeded: check points from `np.random.default_rng(<fixed seed>)`,
   targets built with `seed=`. A `models.py` module holds the five
   accepted models of the prototype (`scalar`, `positive`, `vector`,
   `bounded`, `one_sided`, data simulated from a seeded generator as the
   prototype does) with their hand-written densities, the `scalar`
   model's analytic posterior mean and standard deviation, three models
   for the setup (the eight-schools model, centered and non-centered,
   and a model whose likelihood is a custom `Op` without `grad`), and
   eight rejected models: the prototype's four (`discrete`, `simplex`,
   `suppressed`, `random_bounds`), an ordered transform
   (`Normal(..., shape=3, transform=pm.distributions.transforms.ordered)`),
   a zero-sum variable (`ZeroSumNormal`), an interval variable mixing a
   one-sided and a two-sided coordinate (`Uniform("w", lower=[0, -inf],
   upper=[1, 2])`), and a model whose observed likelihood is not finite
   at the mode. Models are built once per module through module-scoped
   fixtures, to keep the number of PyTensor compilations small in CI.
   - `test_imports.py` (runs without PyMC): a subprocess `import pyvbmc;
     import pyvbmc.pymc` leaves `pymc` and `pytensor` out of
     `sys.modules`; with a `MetaPathFinder` hiding `pymc`,
     `PyMCTarget(object())` raises `ImportError` matching
     `pyvbmc\[pymc\]` (the pattern of
     `pyvbmc/testing/svbmc/test_svbmc_imports.py`).
   - `test_target.py`: for each accepted model, `names` (in the given
     model's order, on a model whose partially untransformed order
     differs, for instance `beta, sigma, t, p`), `shapes`, `sizes`,
     `value_names`, `coordinate_names` (including `beta[intercept]`,
     `u[0]`, `sigma_log__`), `kept`, `support`, `model is` the given
     model, `lb`/`ub` per coordinate; the density check: `d_i =
     log_joint_no_jacobian(x_i) - hand(x_i)` at 25 seeded uniform points
     inside the plausible box, `offset = mean(d)` with `|offset| <=
     1e-6` and `max |d - offset| <= 1e-8`; the Jacobian increment
     `log_joint - log_joint_no_jacobian` equal to the sum of the kept
     value variables to `1e-8` at the same points; `-inf` one unit
     outside a finite bound and on it; a finite value at
     `np.nextafter(lb, inf)` and `np.nextafter(ub, -inf)` for every
     finite bound; `(1, D)` input accepted; `flatten`/`unflatten` round
     trip on the initial point; `to_model_variables` against the closed
     forms of the kept transforms to `1e-12` and `from_model_variables`
     inverting it; `assert_float64` on a dictionary of `x0, lb, ub,
     plb, pub` and of the export's arrays; the non-finite-at-mode model
     raises `ValueError` at construction naming its variable.
   - `test_rejections.py`: each rejected model raises `UnsupportedModel`
     whose message contains the variable's name and, respectively,
     `int64`, `unsupported transform SimplexTransform`, `suppressed`,
     `depends on another random variable`, `unsupported transform
     Ordered`, `unsupported transform ZeroSumTransform`, `mix`.
   - `test_guards.py`: with `_compat.default_transform` monkeypatched to
     raise `ImportError`, construction raises `ImportError` containing
     `TESTED_RANGE`; the same for a `transform_classes()` that lacks a
     class, for `check_model` on an object without `free_RVs`, and for
     an interval transform whose `args_fn` attribute is deleted; a
     transform object of an unregistered subclass of `Interval`
     attached through `transform=` raises `UnsupportedModel`; a
     `closed_form` monkeypatched to return a wrong map makes
     construction raise `ImportError` (the closed-form check fires).
   - `test_plausible.py`: on the `scalar` model the box equals the
     analytic posterior mean `± 3` standard deviations to `1e-3`
     relative and `plausible_info` is `{"route": "laplace", "start":
     "mode", "mode": {...}, "n_evaluations": n, "cap_reached": False,
     "start_moved": [], "curvature": [], "relocated": [], "clipped":
     []}` with `n <= 25` equal to the number of rows of
     `setup_evaluations` plus 1 for the Hessian (`D = 1`); for every
     model `n_evaluations <= 20 + 5 D`, and the rows of
     `setup_evaluations` satisfy `y[i] == log_joint(X[i])` exactly and
     lie inside the start-point rule's interval; the `bounded` model's
     `curvature` and `clipped` lists and the `cap_reached` and
     `relocated` entries of both eight-schools models are asserted as
     the probe recorded them, and each fallback emits a warning from the
     `pyvbmc.pymc` logger (`caplog`); the custom-`Op` model gives
     `route == "prior"`, `start == "initial"`, `n_evaluations == 1` and
     a warning; a `Beta(1, 3)` variable with no data, whose mode is on
     the bound, lists its coordinate under `start_moved`, has finite
     `log_joint(x0)`, and building
     `VBMC(target.log_joint, target.x0, target.lb, target.ub,
     target.plb, target.pub, options={"display": "off"})` on it and on
     the five accepted models emits no record from the `VBMC_init`
     logger (`caplog`); explicit `plausible_bounds` on the `vector` model
     give `plb`/`pub` equal to the mapping with `log` applied to
     `sigma`'s pair, `plausible_info["route"] == "explicit"`, and a
     bound on the support boundary raises `ValueError` for `sigma`
     (kept) and for `p` of the `bounded` model (removed); `start` sets
     `x0` without a search (monkeypatch `_plausible.search_mode` to
     fail) and gives `plausible_info["start"] == "user"`, `mode` None,
     `n_evaluations == 1 + D`; `start` together with `plausible_bounds`
     gives `n_evaluations == 1`;
     a mapping missing a variable or naming an unknown one raises
     `ValueError` naming it; two targets built with `seed=3` on the
     `bounded` model have identical boxes, and NumPy's global state is
     unchanged by construction.
   - `test_export.py`: on a hand-built `VariationalPosterior` with
     `D = target.D` (no optimization), `to_arviz(vp, 40)` has the
     variables' dims, shapes `(1, 40, *shape)`, the model's coordinate
     values, kept variables inside their support, attributes with
     `parameter_space == "model"`, and advances `vp.rng` exactly as one
     `sample(40)`; `pymc.compute_deterministics(data, model=...)` on the
     `vector` model equals `X @ beta` to `1e-12`;
     `pymc.sample_posterior_predictive` returns the observed variable
     with shape `(1, 40, n)`; a posterior of the wrong `D` raises
     `ValueError` and leaves `vp.rng` untouched.
   - `test_save_load.py`: build `VBMC(target.log_joint, target.x0,
     target.lb, target.ub, target.plb, target.pub, options={"display":
     "off"})`, `save` to `tmp_path`, `load`, and assert
     `loaded.function_logger.fun(x) == target.log_joint(x)` at three
     points; `copy.deepcopy(target)` and `dill.loads(dill.dumps(target,
     recurse=True))` give the same values (the pattern of
     `pyvbmc/testing/vbmc/test_vbmc_save_and_load.py`).
   - `test_optimize_short.py`: one seeded run on the `scalar` model with
     `max_iter=3`, `max_fun_evals=40`, `display="off"`, asserting
     completion, a finite `elbo`, and a structured export of the
     returned posterior with the model's variable name. This is the
     only test that exercises the adapter inside the live loop (initial
     design, transformer inverse, the logger's finiteness check); it
     runs only where PyMC is installed.
7. Formatting hooks; `python -m pytest pyvbmc/testing/pymc -x -vv` in the
   PyMC environment; `python -m pytest pyvbmc/testing/pymc -vv` in the
   project venv (everything but `test_imports.py` skips).

**Verification**:
- [ ] All `pyvbmc/testing/pymc` tests pass in the PyMC environment.
- [ ] `python -c "import pyvbmc, pyvbmc.pymc, sys; assert 'pymc' not in sys.modules"`.
- [ ] The prototype still runs:
  `python dev/scripts/pymc_feasibility.py --no-fit --out dev/scripts/runs/pymc_feasibility_20260914/nofit_phase2`
  in the PyMC environment (the output directory is gitignored; the
  tracked `dev/experiments/pymc_feasibility/` is not overwritten).

### Phase 3: CI

**Executor**: Opus sub-agent, after Phase 2 step 5 (the extra must
exist).
**Goal**: the optional cell installs and tests the PyMC extra.

**Steps**:

1. `.github/workflows/test-matrix.yml`: rename the optional-coverage step
   and the extras install to cover PyMC (`Select optional coverage
   (posterior exports, S-VBMC, PyMC)`); the extras install becomes
   `pip install -e "./pyvbmc[test,torch,arviz,pymc]"`. The gate stays
   Ubuntu with the newest Python (3.12 today; PyMC 6 requires 3.12, so
   a future 3.13 cell must be checked against PyMC's support before the
   matrix moves). Add a step in that cell that prints
   `pymc.__version__`, `pytensor.__version__`, `arviz.__version__`,
   `numpy.__version__`, `scipy.__version__` and
   `type(pytensor.compile.mode.get_default_mode().linker).__name__`, so
   the log records the resolved linker and that installing PyMC left
   NumPy and SciPy at or above the floors in `pyproject.toml`. If the
   cell's runtime grows past the suite's by more than a few minutes
   because of PyTensor's compilation, set
   `PYTENSOR_FLAGS=mode=FAST_COMPILE` on the test step of that cell and
   record the choice here; the default is to leave the default linker on.
2. Push the branch; the `dev*` smoke runs the cell.

**Verification**:
- [ ] The smoke run is green and its log shows the PyMC tests running,
  not skipping, and the installed versions and linker; record them in
  the execution record (Phase 6 carries them into `TESTED_RANGE` and the
  installation page).

### Phase 4: documentation

**Executor**: Opus sub-agent, in parallel with Phase 5 for the writing;
its docs-build gate runs after Phase 5 has committed the notebook.
Nothing heavy runs in this phase besides the docs build.
**Goal**: every user-facing surface that lists integrations covers the
adapter.

**Steps** (the sites and their analogues were located on 2026-09-14):

1. `docsrc/source/quickstart.rst`: a new top-level section `Bring a PyMC
   model into PyVBMC` before `Use a fitted posterior downstream`, 60 to
   90 lines in the style of the torch/JAX section: install line, a
   regression model with a positive scale and a named coefficient
   vector, `PyMCTarget`, what `print(target)` shows (which coordinate is
   which, the kept transform), the `VBMC` call, the structured export
   and the two PyMC calls, then the supported scope and the rejections,
   the default box, its evaluation budget, how its evaluations reach
   VBMC and how to override it, building a new target after
   `pm.set_data`, float64, the generator, and that saving works under a
   compatible PyMC. Run every code block of the section in the PyMC
   environment (as a script; no full `optimize()` is needed beyond one
   short run) and paste the actual `print(target)` output; do not
   invent it.
2. `docsrc/source/installation.rst`: a `PyMC` subsection after `ArviZ`
   (pip, the Python 3.12 requirement in the ArviZ paragraph's words,
   the tested versions as `TESTED_RANGE` states them, `conda install
   --channel=conda-forge pyvbmc pymc`); the closing sentence names three
   extras.
3. `docsrc/source/api/classes/pymc_target.rst` in the shape of
   `svbmc.rst` (note block with the extra and the lazy import; sections
   on coordinates and bounds, the plausible box, the export and the
   return path, unsupported models, saving; `autoclass` for
   `PyMCTarget` and `autoexception` for `UnsupportedModel` at the end),
   with section titles that no other page uses (Sphinx's
   `autosectionlabel` is global here, so a title such as "Basic usage"
   would collide with `svbmc.rst`); its line in
   `docsrc/source/documentation.rst` after `api/classes/svbmc` and in
   the alphabetical toctree of `docsrc/source/api/classes/classes.rst`.
4. `docsrc/source/index.rst` and `README.md`: a `What's new` bullet
   after the torch/JAX adapters bullet; the sentence recommending Stan or
   PyMC for fast closed-form models gains a clause pointing PyMC users
   with expensive models at the adapter (the same sentence also appears
   in the FAQ's "What do I do if VBMC is not suited for my problem?");
   README's optional-integrations paragraph names the `pymc` extra.
5. `docsrc/source/faq.md` (its table of contents is hand-maintained): a
   new entry `Can I use a PyMC model with PyVBMC?` after `What is the
   target function?`; one sentence in `Can I have parameters bounded
   only on one side?` (the adapter does the reparameterization and the
   Jacobian for a PyMC model); one clause in `How do I prevent VBMC from
   evaluating certain inputs or regions of input space?`, whose rule
   against returning `-inf` needs the carve-out that the adapter's
   target returns `-inf` only outside the hard box, where VBMC never
   evaluates; the extras and Python-version entries name the `pymc`
   extra.
6. `skills/pyvbmc/SKILL.md`: a table row `Fit a PyMC model` linking the
   quickstart section, the API page and Example 8; `PyMC` in the
   front-matter description.
7. `docsrc/source/development.rst`: one sentence in the testing section
   saying that the torch, ArviZ and PyMC tests skip unless the matching
   extra is installed.
8. `AGENTS.md`: the `Extras` line in "Setup and commands" and the extras
   sentence of the "Posterior exports" bullet; a bullet on the adapter's
   contract (coordinates, `-inf` outside the box only, the four guarded
   reaches, where the tests run); the lazy-import sentence.
9. After Phase 5 has committed the notebook, build the docs (`cd docsrc
   && make github`, or `make.bat github` from `cmd` on Windows) and
   check that the new and changed pages render without warnings.

**Verification**:
- [ ] `grep -rn -i pymc docsrc/source README.md skills AGENTS.md` shows
  every site above, `classes.rst` and the three FAQ entries included.
- [ ] The docs build is warning-free for the new and changed pages.

### Phase 5: Example 8

**Executor**: Opus sub-agent, in parallel with Phase 4's writing; this
phase holds the heavy slot (the notebook's VBMC run and NUTS sampling).
**Goal**: `examples/pyvbmc_example_8_pymc.ipynb`, executed once, with its
generated script.

**Steps**:

1. Title `# PyVBMC Example 8: Fitting a PyMC model` (the exact text every
   `:ref:` uses). Intro in the Example 7 pattern, naming the extra.
   Sections: 1. a regression model in PyMC (named coefficients, a
   half-normal scale, a deterministic mean, simulated data); 2. from the
   model to a PyVBMC target (`print(target)`, which coordinate is which,
   the kept transform and its Jacobian, the default box); 3. run
   PyVBMC; 4. back to PyMC (`to_arviz`, `az.summary`,
   `pm.compute_deterministics`, `pm.sample_posterior_predictive`, a
   posterior-predictive plot); 5. a comparison with `pm.sample` (NUTS,
   short) through `az.plot_forest` on both posteriors; 6. conclusions
   (what is supported, what is rejected, when to prefer PyMC's own
   samplers); acknowledgments.
2. Execute it in the PyMC environment (a temporary kernelspec bound to
   that interpreter, removed afterwards, as the Example 7 record in
   `dev/plans/svbmc-integration.md` describes), strip execution timing
   metadata, commit outputs.
3. `examples/scripts/Makefile`: `EXAMPLES = 1 2 3 4 5 6 7 8`; generate and
   commit `examples/scripts/pyvbmc_example_8_full_code.py`.

**Verification**:
- [ ] The notebook runs top to bottom in the PyMC environment in a few
  minutes and the PyVBMC and NUTS marginals overlap in the forest plot.
- [ ] The `docsrc` build lists the example (checked in Phase 4 step 9).

### Phase 6: verification

**Executor**: Fable (orchestrator).
**Goal**: the branch is release-quality before it merges.

- In the PyMC environment: `pyvbmc/testing/pymc`,
  `test_vp_arviz.py` and `test_vp_export_dependencies.py`. In the
  project venv with single-threaded BLAS: the full suite (the PyMC and
  ArviZ tests skip there; every other test must pass as before, since
  only `to_arviz` and new modules changed) and
  `python dev/scripts/make_oracle_fixtures.py --check --exact` (nothing
  numerical changed, so every oracle must be bit-identical). The golden
  replay (`dev/scripts/golden_replay.py`) runs only if the diff touches
  a file the solver imports; the adapter and the export do not, and the
  exact oracle check stands in for it.
- A wheel and an sdist built and inspected: `pyvbmc/pymc/` in both,
  `pyvbmc/testing/pymc/` in the sdist only; if the sdist lacks it, add
  `recursive-include pyvbmc/testing/pymc *.py` to `MANIFEST.in` as the
  calibration tests have.
- The docs build; the smoke run green with the PyMC tests executed.
- `/doublecheck` with read-only Opus reviewers (implementation and
  guards; tests; docs and example); findings folded in; the smoke rerun.
- `TESTED_RANGE` and the installation page updated to the versions the
  CI cell and the PyMC environment actually ran (this phase is the only
  editor of both after Phase 2 and Phase 4 wrote them).

### Phase 7: merge and records

**Executor**: Fable (orchestrator).

- Merge `dev-pymc-adapter` into `dev-next` (fast-forward or merge commit
  as the branch allows); delete the branch.
- `dev/TODO.md`: the PyMC item becomes complete or moves its remaining
  action to the release item (the reference sweep already covers the
  skill's `dev-next` links); `dev/plans/modernization-roadmap.md` and
  `dev/2026-09-06-pyvbmc-1.5-overview.md` gain the adapter;
  `dev/README.md` index and plans list; the proposal header's status.
- The execution record below.

## Documentation

Existing documents updated: `docsrc/source/quickstart.rst`,
`installation.rst`, `index.rst`, `documentation.rst`,
`api/classes/classes.rst`, `faq.md`, `development.rst`,
`api/methods/variational_posterior_to_arviz.rst`, `README.md`,
`skills/pyvbmc/SKILL.md`, `AGENTS.md`, `dev/TODO.md`, `dev/README.md`,
`dev/plans/modernization-roadmap.md`,
`dev/2026-09-06-pyvbmc-1.5-overview.md`,
`dev/2026-09-13-pymc-integration.md` (status line),
`dev/results/2026-09-14-pymc-feasibility.md` (the environment's
linker). New documents, each with a role no existing one has:
`docsrc/source/api/classes/pymc_target.rst` (the API page every public
class has), `examples/pyvbmc_example_8_pymc.ipynb` with its script (the
worked example), `dev/results/<date>-pymc-setup-probe.md` with
`dev/scripts/pymc_setup_probe.py` and `dev/experiments/pymc_setup_probe/`
(the evidence for the setup budget and the reuse of its evaluations),
and this plan (the design decisions and the execution record).

## Decisions

- **A class, `PyMCTarget`, constructed directly, exposed lazily as
  `pyvbmc.PyMCTarget`** — the adapter is an object with a callable, arrays
  and an export method, and the `SVBMC` precedent already resolves an
  optional-extra class lazily. The name says what the object is to
  `VBMC`; the PI confirmed it on 2026-09-14 over `PyMCModel` and
  `PyMCAdapter`. Rejected: a function `from_pymc(model)` returning the
  same object (adds a name for no behavior) and a `pyvbmc.pymc_fit`
  wrapper (the deferred orchestration).
- **Subpackage `pyvbmc/pymc/`** — mirrors `pyvbmc/svbmc/` and leaves room
  for the compatibility and plausible-box modules. Rejected: a flat
  module (no room) and `pyvbmc/integrations/` (a new convention with no
  precedent; the torch and JAX adapters are documentation, not code).
- **Floor `pymc >= 6.3`, no upper bound** — see "Version range and
  guards". Rejected: `>= 6.0` (untested, and the documented workflow
  would need a different form of one call there) and an upper bound
  (maintenance on every PyMC release against the convention of the
  other extras).
- **Kept-variable maps through compiled `transform.forward` and
  `transform.backward`** — the route the feasibility check validated,
  guarded by a closed-form check at construction. Rejected: PyMC's
  public `constrain_values` and `unconstrain_values`, which fail under
  the default linker on the tested PyMC (see "What the plan rests on").
- **A setup budget of `20 + 5 D` evaluations, spent on a gradient search
  and one exact Hessian of the adapter's own log joint** — the PI's
  decision of 2026-09-14: VBMC exists for targets whose evaluations are
  expensive, so the setup must be nearly free of them, and PyTensor
  supplies exact first and second derivatives of every PyMC model built
  from differentiable operations. The budget is provisional until the
  setup probe has run. Rejected: a finite-difference Hessian (about
  `2 D²` evaluations, 800 at `D = 20`, where the exact one costs about
  `D`), `pymc.find_MAP` (a search on the density without the Jacobian,
  allowed 5000 evaluations by default and falling back to Powell's
  derivative-free search on a model without a gradient), and prior
  quantiles alone as the
  default (no evaluations, but on the feasibility check's regression
  with `Normal(0, 5)` coefficients their box spanned log joints from
  −93 827 to −63 and the initial GP fit failed).
- **Fallbacks that cost no evaluations** — prior-quantile widths for a
  coordinate without usable curvature, a location check against the
  prior for a search that ran away, and the prior box for a model without
  a gradient, each reported. Explicit `plausible_bounds` and `start` are
  accepted; the mapping form of `plausible_bounds` is all-or-nothing
  (every free variable), because a partial mapping would have to combine
  user intervals with Laplace intervals coordinate by coordinate and the
  fallbacks already handle the ridge case that would motivate it.
- **The setup's evaluations go to VBMC** — every evaluation the search
  spends is a valid training point, and discarding them would waste
  evaluations of an expensive target. The route (the existing `f_vals`
  option or an extension of `VBMC`'s interface) is settled before
  approval.
- **The adapter applies VBMC's own start-point and margin rules before
  handing over** — a start point moved inside by `2e-3` of the range and
  plausible bounds never clipped past it, so that the finiteness check
  and the curvature are evaluated at the point VBMC uses and none of the
  constructor's adjustments fires. Rejected: leaving a boundary mode to
  VBMC, which would have made the construction-time finiteness check
  reject models VBMC accepts.
- **`-inf` on and outside the hard box, a named `ValueError` inside it** —
  the first is the density's value at points VBMC never requests; the
  second gives the user the model's variables where VBMC's own error
  would give a flat vector. Rejected: clamping to a large negative
  number (the FAQ forbids it for good reason).
- **Guards by capability, not version number** — the reaches either work
  or fail loudly at construction with the tested range in the message.
  Rejected: a version-based warning on newer PyMC (noise without
  information).
- **Fallback reports through a named logger at `WARNING`** — visible by
  default through Python's last-resort handler and through any
  configured root logger, the level VBMC uses for its own bound
  adjustments on its `VBMC_init` logger; `warnings.warn` is reserved in
  this codebase for argument misuse.
- **The structured export as new keyword-only parameters, not an overload
  of `var_names`** — `var_names` accepts any iterable today, so a mapping
  passed there is already legal input with a different meaning.
- **One short end-to-end run in the PyMC tests** — the repository's rule
  against more `optimize()` runs protects the suite's runtime; this run
  is three iterations in one dimension, executes only where PyMC is
  installed (one CI cell), and is the only check that the adapter's
  callable survives the live loop.
- **PyTensor's default linker in CI** — it is what users get, and the
  save/load test then exercises pickling of numba-compiled functions;
  `PYTENSOR_FLAGS=mode=FAST_COMPILE` is the recorded fallback if the
  cell becomes slow.
- **Example 8 compares with NUTS** — a PyMC user's first question is
  whether the answers agree; the cost is a short `pm.sample` in a
  notebook that is never executed in CI or the docs build.

## Execution record

- 2026-09-14: plan drafted from the feasibility record, the probes and
  the API research above, reviewed by three read-only Opus reviewers
  (codebase facts and scope; PyMC API claims and design consistency,
  with probes in the PyMC environment; executor precision), and revised
  on their findings: the value maps returned to the prototype's compiled
  transforms after `constrain_values` was found to fail under PyTensor's
  default linker; the environment's linker was identified as numba, not
  the Python interpreter; the start-point rule, the halfway clipping
  rule and the support check for explicit bounds were added; every
  method, `plausible_info` key, coordinate-name rule, rejection message,
  `_plausible` signature and test seed was pinned down; the guard tests,
  `classes.rst`, the `UnsupportedModel` documentation, the FAQ's `-inf`
  carve-out, the docs-build ordering after Example 8 and the single
  owner of `TESTED_RANGE` were added. The PI confirmed the name
  `PyMCTarget`. The plan awaits approval before Phase 0 runs.
- 2026-09-14: the plan, the status pointers and the feasibility report's
  corrected environment description (numba linker, not the Python
  interpreter; confirmed in the PyMC environment: PyTensor 3.3.1 with
  numba 0.67.0 resolves `linker = auto` to `NumbaLinker` with no C++
  compiler present) committed on `dev-next` at the handoff. No
  implementation has started.
- 2026-09-14: the PI's review, before approval, changed the setup. The
  finite-difference Hessian and `pymc.find_MAP` gave way to a budget of
  `20 + 5 D` evaluations spent on an L-BFGS-B search with PyTensor's
  gradient and one exact Hessian (`compile_d2logp`) of the adapter's own
  log joint, with fallbacks that cost no evaluations, among them a
  location check against the prior for a search that runs away (a
  concern for hierarchical models, none of which the feasibility check
  covered; the eight-schools models join the tests). The evaluations
  the setup spends are to be handed to `VBMC` rather than discarded.
  Two investigations were added before approval: the setup probe (Part
  A: the budget, the stopping rule, the location check) and the reuse of
  the setup evaluations (Part B: `f_vals` or an extension of `VBMC`'s
  interface, measured in short VBMC runs). The PI kept the short
  end-to-end run in the PyMC tests; the documentation gains the
  `pm.set_data` caveat.
