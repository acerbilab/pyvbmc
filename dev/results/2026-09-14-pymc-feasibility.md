# PyMC adapter feasibility check

The bounded prototype that the
[PyMC integration proposal](../2026-09-13-pymc-integration.md#feasibility-check)
asks for is `dev/scripts/pymc_feasibility.py`: a PyMC model becomes a
box-bounded log joint for `VBMC` with the parameter mapping retained, the
fitted posterior is exported with the model's variable names, shapes and
coordinates, and PyMC's own functions compute deterministic quantities and
posterior predictions on the exported draws. Five models (an ordinary
scalar model, a positive scale, a two-coefficient regression with a
deterministic mean, a model with a unit-interval and a per-coordinate
interval variable, and a model with a variable bounded below at a nonzero
value and one bounded above only) and four unsupported ones (a discrete
latent, a simplex, a suppressed transform, an interval whose limit is
another random variable) exercise it against hand-written densities and
analytic or quadrature evidences. The package is unchanged. The prototype
settles the coordinate convention and the bounds question the proposal
left open, shows that the adapter can be written against public PyMC APIs
with three small exceptions, and turns up two constraints on the plausible
box. Tracked outputs:
[`experiments/pymc_feasibility/`](../experiments/pymc_feasibility/)
(`fit_laplace/`, `fit_prior/`, `nofit/`, each with `report.json` and
`report.md`; every report records the PyVBMC and gpyreg commits and the
script's hash). Environment: PyMC 6.3.2, PyTensor 3.3.1 (no C++ compiler,
Python mode, `floatX` float64), ArviZ 1.3.0, Python 3.12, in a dedicated
virtual environment with this checkout of PyVBMC and gpyreg 1.2.1
installed editable (the project venv has neither PyMC nor ArviZ; the
environment's location is listed in the gitignored
`dev/scripts/runs/LOCAL.md`). These are the only versions tested.

## What the adapter does

- **Coordinates and bounds.** VBMC accepts a parameter that is unbounded
  or bounded on both sides, and refuses one bounded on one side
  (`vbmc:HalfBounds`). PyMC attaches a value transform to every constrained
  variable. The adapter keeps PyMC's transform for a variable bounded on one
  side (a positive scale travels as `sigma_log__`, a variable bounded below
  at 1 as `t_interval__`, both unbounded for VBMC, and
  `Model.compile_logp(jacobian=True)` adds the Jacobian, so VBMC's ELBO
  still estimates the model's evidence) and removes it for a variable
  bounded on both sides or unbounded
  (`pymc.model.transform.conditioning.remove_value_transforms(model, vars=...)`),
  handing VBMC the interval as hard bounds, coordinate by coordinate, and
  the model's own density there. Draws come back through the kept
  transforms' `backward` maps, so the exported posterior is over `sigma`
  and `t`, not their transformed value variables. The bounds are read from
  the transform PyMC attached: log, log-odds, or an interval whose limits
  come from the transform's `args_fn` applied to the variable's inputs and
  are evaluated per coordinate; a limit that depends on another random
  variable is rejected. A variable with no transform is unbounded only if
  the distribution's default transform is none as well; a transform
  suppressed at construction (`default_transform=None`) is rejected.
- **Mapping.** Free variables in model order, shapes from the initial
  point of the partially transformed model, flattened into the vector VBMC
  sees; the inverse rebuilds named arrays and hands them to
  `arviz.from_dict` (the ArviZ 1.x signature: a dict of groups, returning a
  `DataTree`) with the dimensions and coordinates the model declares.
- **Return path.** `pymc.compute_deterministics(posterior, model=model)`
  and `pymc.sample_posterior_predictive(data, model=model)` on the
  exported draws.
- **Rejection.** A discrete free variable (dtype), a transform whose
  support is not a box (simplex, ordered, zero-sum, Cholesky, softplus), a
  suppressed transform or a random interval limit raises
  `UnsupportedModel`, naming the variable, before anything is compiled.
- **Starting point and plausible box.** Two routes: `prior` (the model's
  initial point and the 5 % and 95 % quantiles of prior draws through
  `pymc.draw`) and `laplace` (the mode from `pymc.find_MAP` on the fully
  transformed model, whose returned point carries every coordinate the
  adapter uses, and a box of three marginal SDs from a finite-difference
  Hessian of the adapter's log joint at that point). `find_MAP` maximizes
  the density without the Jacobian, so for a kept transform the point is
  the mode of the model's density in its own variables rather than the
  stationary point of the adapter's target; a coordinate whose curvature
  is not negative there falls back to the prior-quantile width, and one
  whose Laplace interval leaves the hard box is clipped to the middle 98 %
  of it, both recorded under `laplace_fallback`.

## Results

Every fit is `VBMC` with PyVBMC defaults, seed 0; every model's data come
from a generator of its own, so both routes see the same models. The
density check compares the adapter's compiled density with the
hand-written one at 25 random points inside the plausible box, split into
a constant offset and the deviation from it, and probes a point one unit
outside a finite bound; the Jacobian check compares `logp(jacobian=True)
− logp(jacobian=False)` with the sum of the kept value variables (the
log-Jacobian of a log transform and of a one-sided interval transform is
the value variable itself). `ms per call` is the compiled density's cost
at the starting point.

| Model | D | VBMC coordinates | density: offset; shape | outside a bound | Jacobian increment | ms per call | ELBO − ln Z (± sd) | evaluations | setup + fit seconds | exported |
|---|---:|---|---|---|---|---:|---:|---:|---:|---|
| scalar | 1 | `mu` unbounded | +2.2e-7; 4e-14 | n/a | n/a | 0.09 | +0.0003 (± 0.0000) | 65 | 0.2 + 5 | `mu` (chain, draw); predictive `y` (1, 2000, 20) |
| positive | 1 | `sigma_log__` (log transform kept) | −1.9e-9; 8e-14 | n/a | 1.3e-14 | 0.02 | −0.0008 (± 0.0008) | 65 | 1.0 + 5 | `sigma` (chain, draw); predictive `y` (1, 2000, 15) |
| vector | 3 | `beta` unbounded; `sigma_log__` | −6.2e-8; 3e-11 | n/a | 3.3e-12 | 0.02 | −0.0037 (± 0.0009) | 65 | 2.4 + 23 | `beta` (chain, draw, coef) with `coef = [intercept, slope]`, `sigma`; `mu` (1, 2000, 30) equal to `X @ beta` to 0; predictive `y` (1, 2000, 30) |
| bounded | 3 | `p` in [0, 1]; `u` in [−1, 2] × [0, 1] | +2.2e-8; 1e-14 | −inf | n/a | 0.16 | −0.0073 (± 0.0029) | 75 | 3.7 + 36 | `p` (chain, draw), `u` (chain, draw, u_dim_0); predictive `y` (1, 2000, 12) |
| one_sided | 2 | `t_interval__`, `v_interval__` (interval transforms kept) | −8.0e-9; 8e-15 | n/a | 6.2e-15 | 0.02 | −0.0048 (± 0.0008) | 75 | 2.8 + 19 | `t`, `v` (chain, draw); predictive `y_t` (1, 2000, 8) |

These are the `laplace` route. The evidences are analytic (scalar) or
quadrature with relative errors below 1e-8 (the `ln_Z_error` field). For
the scalar model the posterior mean is within 0.0042 of the analytic one
and the posterior SD within 1.5 % of it. All four rejections fire with a
message naming the variable and the reason (`k is int64`; `w: unsupported
transform SimplexTransform`; `s: its LogTransform was suppressed at
construction`; `h: an interval limit depends on another random variable`).

The `prior` route fits the scalar, positive, bounded and one-sided models
the same way (ELBO − ln Z of +0.0017, −0.0015, −0.0035 and −0.0043, with
standard deviations of 0.0001 to 0.0031) and fails on the regression: the
box from the 5 % and 95 % quantiles of `Normal(0, 5)` coefficients and a
`HalfNormal(2)` scale is `beta` in about [−8.2, 8.4] and `log sigma` in
[−2.0, 1.4], the log joint at its lower corner is −93 827 against −63 at
the initial point (recorded with the error), and the GP fit of the initial
design ends in `LinAlgError: Singular matrix for L Cholesky decomposition`
(`gpyreg.gaussian_process.fit` through `train_gp`). On the bounded model
the Laplace route met the other constraint: the likelihood depends only on
`p + u[0] + u[1]`, the posterior is a ridge, the finite-difference Hessian
at the mode is not negative definite, so all three coordinates took the
prior-quantile width and one of `u`'s was then clipped to the hard box
(both recorded); the fit succeeded, at 75 evaluations against 65 for the
unimodal models.

## What the prototype establishes

1. **The coordinate convention.** Handing VBMC the model's own
   coordinates is possible only for unbounded and two-sided variables;
   one-sided variables must keep PyMC's transform. This mixed convention
   is one call to `remove_value_transforms` with the variables to
   untransform, and PyMC supplies the Jacobian for the kept ones. Every
   kind was exercised: unbounded (`mu`, `beta`); one-sided kept through
   the log transform (`sigma`) and through interval transforms with one
   infinite limit, bounded below at 1 (`t`) and above at −0.5 (`v`);
   two-sided removed with the unit interval (`p`) and with per-coordinate
   limits (`u`). The Jacobian increments agree to 4e-12, the evidences to
   0.008 nats, a point outside a bound gives `−inf`, and the exported
   posteriors are over the model's variables.
2. **Public APIs suffice, with three reaches.** Everything runs through
   `Model.free_RVs`, `rvs_to_transforms`, `rvs_to_values`,
   `initial_point`, `compile_logp`, `remove_value_transforms`, `pymc.draw`,
   `pymc.find_MAP`, `pymc.compute_deterministics`,
   `pymc.sample_posterior_predictive` and the transforms' `forward` and
   `backward`. The reaches past documented attributes are: dispatch on the
   transform classes' names (`LogTransform`, `LogOddsTransform`,
   `Interval`; a subclass would be rejected as unsupported), the
   variable's `rv.owner.inputs` passed to the transform's `args_fn`,
   `forward` and `backward`, and the private
   `pymc.distributions.transforms._default_transform` to recognise a
   suppressed transform. Each would need a version guard in a shipped
   adapter.
3. **Normalization is exact to a float32 constant.** PyTensor folds the
   logarithm of a Python-float scale (`Normal(mu, 1.5)`) into a float32
   constant, so a model with constant scales carries an offset of about
   1e-8 nats per such term against the exact density (2.2e-7 over the
   scalar model's 21 terms); the shape of the density agrees to 3e-11 and
   a symbolic scale is exact. This is irrelevant to inference and to
   evidence at any precision VBMC reports, and a density check has to
   separate it from real errors.
4. **The return path works on the current ArviZ.** ArviZ 1.3's
   `from_dict` takes a dict of groups and returns a `DataTree`; PyMC 6.3's
   `compute_deterministics` and `sample_posterior_predictive` accept it,
   the model's coordinates propagate (`coef`), the deterministic equals
   `X @ beta` exactly and the predictive has the observed variable's
   shape. The `arviz.from_dict(posterior=...)` form the proposal links is
   the 0.x signature and is not supported by the prototype.
5. **Plausible bounds are the user's or the mode's, not the prior's, and
   the mode's need a guard.** Prior quantiles work for well-scaled models
   and break a regression with vague priors; the Laplace box works on all
   five models, provided that a curvature that is not negative or an
   interval that leaves the hard box (a ridge) falls back to the prior
   width or the box, which the bounded model required. An adapter that
   offers a default should offer the Laplace route with these fallbacks
   and accept explicit bounds.
6. **Cost.** Without a C++ compiler the compiled density runs in
   PyTensor's Python mode at 0.02–0.16 ms per call, so the target is not
   what VBMC's few hundred evaluations spend their time on; the fits took
   5 to 36 seconds and the mode-plus-Hessian setup 0.2 to 3.7 seconds on
   a laptop in mixed use.

## Supported scope this implies for 1.5

Continuous free variables whose value transforms are none (with none as
the distribution's default), log, log-odds or an interval with fixed
limits on one or both sides; every other transform, a suppressed
transform, a random interval limit and every discrete variable rejected
with the variable's name. Deterministics and observed variables as the
model defines them. Starting point and plausible box from the Laplace
route with its fallbacks by default, with explicit bounds accepted. The
automatic wrapper of the proposal (initialization policy, several fits,
everything returned) stays deferred. The structured export is a small
extension of `VariationalPosterior.to_arviz` (dimensions and coordinates
per variable) and is useful without PyMC. Two things a shipped adapter
would still have to settle: a version range for PyMC and ArviZ beyond the
one pair tested here, and `VBMC.save`, which pickles the target callable
and has not been tried on an adapter holding compiled PyTensor functions
and a PyMC model. The PI chose, on this record (2026-09-14), the target
adapter with this scope for PyVBMC 1.5.

## Files

- `dev/scripts/pymc_feasibility.py`: the prototype (`--out DIR
  [--plausible laplace|prior] [--no-fit]`); its module docstring is the
  specification of the adapter's route through PyMC.
- `dev/experiments/pymc_feasibility/fit_laplace/`, `fit_prior/`,
  `nofit/`: `report.json` (environment with commits and the script's
  hash, every check with its numbers, the fits with their boxes, the
  return path, the rejections, the errors with the box and the log joint
  at its corners) and `report.md`.
