# PyMC adapter feasibility check

The bounded prototype that the
[PyMC integration proposal](../2026-09-13-pymc-integration.md#feasibility-check)
asks for is `dev/scripts/pymc_feasibility.py`: a PyMC model becomes a
box-bounded log joint for `VBMC` with the parameter mapping retained, the
fitted posterior is exported with the model's variable names, shapes and
coordinates, and PyMC's own functions compute deterministic quantities and
posterior predictions on the exported draws. Three models (an ordinary
scalar model, a positive scale, a two-coefficient regression with a
deterministic mean) and two unsupported ones (a discrete latent, a simplex)
exercise it against hand-written densities and analytic or quadrature
evidences. The package is unchanged. The prototype settles the coordinate
convention and the bounds question the proposal left open, shows that the
adapter can be written against public PyMC APIs with two small exceptions,
and turns up one hard constraint on the plausible bounds. Tracked outputs:
[`experiments/pymc_feasibility/`](../experiments/pymc_feasibility/)
(`fit_laplace/`, `fit_prior/`, `nofit/`, each with `report.json` and
`report.md`). Environment: PyMC 6.3.2, PyTensor 3.3.1 (no C++ compiler,
Python mode, `floatX` float64), ArviZ 1.3.0, Python 3.12, in a dedicated
virtual environment with this checkout of PyVBMC and gpyreg 1.2.1 installed
editable (the project venv has neither PyMC nor ArviZ; the environment's
location is listed in the gitignored `dev/scripts/runs/LOCAL.md`).

## What the adapter does

- **Coordinates and bounds.** VBMC accepts a parameter that is unbounded
  or bounded on both sides, and refuses one bounded on one side
  (`vbmc:HalfBounds`). PyMC attaches a value transform to every constrained
  variable. The adapter keeps PyMC's transform for a variable bounded on one
  side (a positive scale travels as `sigma_log__`, unbounded for VBMC, and
  `Model.compile_logp(jacobian=True)` adds the Jacobian, so VBMC's ELBO
  still estimates the model's evidence) and removes it for a variable
  bounded on both sides or unbounded
  (`pymc.model.transform.conditioning.remove_value_transforms(model, vars=...)`),
  handing VBMC the interval as hard bounds and the model's own density
  there. Draws come back through the kept transforms' `backward` maps, so
  the exported posterior is over `sigma`, not `sigma_log__`. The bounds
  themselves are read from the transform PyMC attached: none, log,
  log-odds, or an interval whose limits come from the transform's `args_fn`
  applied to the variable's inputs.
- **Mapping.** Free variables in model order, shapes from the initial
  point of the partially transformed model, flattened into the vector VBMC
  sees; the inverse rebuilds named arrays and hands them to
  `arviz.from_dict` (the ArviZ 1.x signature: a dict of groups, returning a
  `DataTree`) with the dimensions and coordinates the model declares.
- **Return path.** `pymc.compute_deterministics(posterior, model=model)`
  and `pymc.sample_posterior_predictive(data, model=model)` on the
  exported draws.
- **Rejection.** A discrete free variable (dtype) or a transform whose
  support is not a box (simplex, ordered, zero-sum, Cholesky) raises
  `UnsupportedModel` before anything is compiled.
- **Starting point and plausible box.** Two routes: `prior` (the model's
  initial point and the 5 % and 95 % quantiles of prior draws through
  `pymc.draw`) and `laplace` (the mode from `pymc.find_MAP` on the fully
  transformed model, whose returned point carries every coordinate the
  adapter uses, and a finite-difference Laplace box of three marginal SDs
  around it in the adapter's own coordinates).

## Results

Every fit is `VBMC` with PyVBMC defaults, seed 0. The density check
compares the adapter's compiled density with the hand-written one at 25
random points inside the plausible box, split into a constant offset and
the deviation from it; the Jacobian check compares `logp(jacobian=True)
− logp(jacobian=False)` with `log sigma`.

| Model | D | VBMC coordinates | density: offset; shape | Jacobian increment | ELBO − ln Z (± sd) | evaluations | seconds | exported |
|---|---:|---|---|---|---:|---:|---:|---|
| scalar | 1 | `mu` unbounded | +2.2e-7; 0 | n/a | +0.000 (± 0.000) | 65 | 3.5 | `mu` (chain, draw); predictive `y` (1, 2000, 20) |
| positive | 1 | `sigma_log__` (log transform kept) | −1.9e-9; 0 | 1.5e-14 | −0.002 (± 0.001) | 65 | 3.5 | `sigma` (chain, draw); predictive `y` (1, 2000, 15) |
| vector | 3 | `beta` unbounded; `sigma_log__` | −6.2e-8; 4e-12 | 3.9e-12 | −0.007 (± 0.001) | 65 | 13.5 | `beta` (chain, draw, coef) with `coef = [intercept, slope]`, `sigma`; `mu` (1, 2000, 30) equal to `X @ beta` to 0; predictive `y` (1, 2000, 30) |

These are the `laplace` route. For the scalar model the posterior mean is
within 0.004 of the analytic one and the posterior SD within 1.5 % of it.
Both rejections fire with a message naming the variable and the reason
(`k is int64`; `w: the support of a variable with a SimplexTransform is
not a box`).

The `prior` route fits the scalar and positive models the same way
(ELBO − ln Z of −0.001 and −0.001) and fails on the regression: the box
from the 5 % and 95 % quantiles of `Normal(0, 5)` coefficients and a
`HalfNormal(2)` scale puts the initial design where the log likelihood
spans tens of thousands of nats, and the GP fit of the initial design
ends in `LinAlgError: Singular matrix for L Cholesky decomposition`
(`gpyreg.gaussian_process.fit` through `train_gp`). The proposal's caution
about prior-based bounds is confirmed in the form that matters: prior
quantiles are not a usable default for the plausible box, while the mode
and its Laplace box are, on these three models.

## What the prototype establishes

1. **The coordinate convention.** Handing VBMC the model's own
   coordinates is possible only for unbounded and two-sided variables;
   one-sided variables must keep PyMC's transform. This mixed convention
   is one call to `remove_value_transforms` with the variables to
   untransform, and PyMC supplies the Jacobian for the kept ones. The
   Jacobian check passes to 1e-12, the evidences agree to 0.007 nats, and
   the exported posterior is over the model's variables.
2. **Public APIs suffice, with two reaches.** Everything runs through
   `Model.free_RVs`, `rvs_to_transforms`, `rvs_to_values`,
   `initial_point`, `compile_logp`, `remove_value_transforms`, `pymc.draw`,
   `pymc.find_MAP`, `pymc.compute_deterministics`,
   `pymc.sample_posterior_predictive` and the transforms' `forward` and
   `backward`. The two reaches past documented attributes are the
   transform classes' names (dispatch on `LogTransform`,
   `LogOddsTransform`, `Interval`) and an interval transform's `args_fn`
   evaluated on `rv.owner.inputs` to read its limits. Both are stable in
   PyMC 6.3 and both would need a version guard in a shipped adapter.
3. **Normalization is exact to a float32 constant.** PyTensor folds the
   logarithm of a Python-float scale (`Normal(mu, 1.5)`) into a float32
   constant, so a model with constant scales carries an offset of about
   1e-8 nats per such term against the exact density (2.2e-7 over the
   scalar model's 21 terms); the shape of the density agrees to 1e-11 and
   a symbolic scale is exact. This is irrelevant to inference and to
   evidence at any precision VBMC reports, and a density check has to
   separate it from real errors.
4. **The return path works on the current ArviZ.** ArviZ 1.3's
   `from_dict` takes a dict of groups and returns a `DataTree`; PyMC 6.3's
   `compute_deterministics` and `sample_posterior_predictive` accept it,
   the model's coordinates propagate (`coef`), the deterministic equals
   `X @ beta` exactly and the predictive has the observed variable's
   shape. The `arviz.from_dict(posterior=...)` form the proposal links is
   the 0.x signature and fails on 1.x.
5. **Plausible bounds are the user's or the mode's, not the prior's.**
   Prior quantiles work for well-scaled scalar models and break a
   regression with vague priors; the Laplace box works for all three. An
   adapter that offers a default should offer the Laplace route (one
   `find_MAP` and a finite-difference Hessian, or PyMC's `find_hessian`
   with the coordinate convention matched) and accept explicit bounds.
6. **Cost.** Without a C++ compiler the compiled density runs in
   PyTensor's Python mode; the three fits took 3.5, 3.5 and 13.5 seconds
   for 65 evaluations each, so the target evaluation is not the
   bottleneck at this size.

## Supported scope this implies for 1.5

Continuous free variables whose value transforms are none, log, log-odds
or an interval; every other transform and every discrete variable
rejected with the variable's name. Deterministics and observed variables
as the model defines them. Starting point and plausible box from the
Laplace route by default, with explicit bounds accepted. The automatic
wrapper of the proposal (initialization policy, several fits, everything
returned) stays deferred. The structured export is a small extension of
`VariationalPosterior.to_arviz` (dimensions and coordinates per variable)
and is useful without PyMC. Whether the adapter ships in 1.5 with this
scope, or only the structured export, is the PI's decision on this record.

## Files

- `dev/scripts/pymc_feasibility.py`: the prototype (`--out DIR
  [--plausible laplace|prior] [--no-fit]`); its module docstring is the
  specification of the adapter's route through PyMC.
- `dev/experiments/pymc_feasibility/fit_laplace/`, `fit_prior/`,
  `nofit/`: `report.json` (environment, every check with its numbers,
  the fits, the return path, the rejections) and `report.md`.
