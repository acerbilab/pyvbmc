# PyMC adapter feasibility check

The bounded prototype that the
[PyMC integration proposal](../2026-09-13-pymc-integration.md#feasibility-check)
asks for is `dev/scripts/pymc_feasibility.py`: a PyMC model becomes a
box-bounded log joint for `VBMC` with the parameter mapping retained, the
fitted posterior is exported with the model's variable names, shapes and
coordinates, and PyMC's own functions compute deterministic quantities and
posterior predictions on the exported draws. Four models (an ordinary
scalar model, a positive scale, a two-coefficient regression with a
deterministic mean, and a model with a unit-interval and a per-coordinate
interval variable) and four unsupported ones (a discrete latent, a
simplex, a suppressed transform, an interval whose limit is another random
variable) exercise it against hand-written densities and analytic or
quadrature evidences. The package is unchanged. The prototype settles the
coordinate convention and the bounds question the proposal left open,
shows that the adapter can be written against public PyMC APIs with three
small exceptions, and turns up two constraints on the plausible box.
Tracked outputs:
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
  side (a positive scale travels as `sigma_log__`, unbounded for VBMC, and
  `Model.compile_logp(jacobian=True)` adds the Jacobian, so VBMC's ELBO
  still estimates the model's evidence) and removes it for a variable
  bounded on both sides or unbounded
  (`pymc.model.transform.conditioning.remove_value_transforms(model, vars=...)`),
  handing VBMC the interval as hard bounds, coordinate by coordinate, and
  the model's own density there. Draws come back through the kept
  transforms' `backward` maps, so the exported posterior is over `sigma`,
  not `sigma_log__`. The bounds are read from the transform PyMC
  attached: log, log-odds, or an interval whose limits come from the
  transform's `args_fn` applied to the variable's inputs and are
  evaluated per coordinate; a limit that depends on another random
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
  of it, both recorded.

## Results

Every fit is `VBMC` with PyVBMC defaults, seed 0. The density check
compares the adapter's compiled density with the hand-written one at 25
random points inside the plausible box, split into a constant offset and
the deviation from it, and probes a point outside a finite bound; the
Jacobian check compares `logp(jacobian=True) − logp(jacobian=False)` with
`log sigma`. `ms per call` is the compiled density's cost at the starting
point.

| Model | D | VBMC coordinates | density: offset; shape | outside a bound | Jacobian increment | ms per call | ELBO − ln Z (± sd) | evaluations | setup + fit seconds | exported |
|---|---:|---|---|---|---|---:|---:|---:|---:|---|
| scalar | 1 | `mu` unbounded | +2.2e-7; 3e-14 | n/a | n/a | 0.06 | +0.0003 (± 0.0000) | 65 | 0.2 + 3.3 | `mu` (chain, draw); predictive `y` (1, 2000, 20) |
| positive | 1 | `sigma_log__` (log transform kept) | −1.9e-9; 3e-14 | n/a | 1.5e-14 | 0.01 | −0.0017 (± 0.0009) | 65 | 0.2 + 3.1 | `sigma` (chain, draw); predictive `y` (1, 2000, 15) |
| vector | 3 | `beta` unbounded; `sigma_log__` | −6.2e-8; 4e-12 | n/a | 3.9e-12 | 0.02 | −0.0070 (± 0.0008) | 65 | 0.3 + 12.4 | `beta` (chain, draw, coef) with `coef = [intercept, slope]`, `sigma`; `mu` (1, 2000, 30) equal to `X @ beta` to 0; predictive `y` (1, 2000, 30) |
| bounded | 3 | `p` in [0, 1]; `u` in [−1, 2] × [0, 1] | +2.2e-8; 7e-15 | −inf | n/a | 0.07 | −0.0111 (± 0.0020) | 80 | 0.5 + 20.1 | `p` (chain, draw), `u` (chain, draw, u_dim_0); predictive `y` (1, 2000, 12) |

These are the `laplace` route. The evidences are analytic (scalar) or
quadrature with relative errors below 3e-8 (the `ln_Z_error` field). For
the scalar model the posterior mean is within 0.0042 of the analytic one
and the posterior SD within 1.5 % of it. All four rejections fire with a
message naming the variable and the reason (`k is int64`; `w: unsupported
transform SimplexTransform`; `s: its LogTransform was suppressed at
construction`; `h: an interval limit depends on another random variable`).

The `prior` route fits the scalar, positive and bounded models the same
way (ELBO − ln Z of −0.0005, −0.0009 and −0.0196 ± 0.0027) and fails on
the regression: the box from the 5 % and 95 % quantiles of `Normal(0, 5)`
coefficients and a `HalfNormal(2)` scale is `beta` in about [−8.2, 8.3]
and `log sigma` in [−2.0, 1.4], the log joint at its lower corner is
−117 369 against −60 at the initial point (recorded with the error), and
the GP fit of the initial design ends in `LinAlgError: Singular matrix for
L Cholesky decomposition` (`gpyreg.gaussian_process.fit` through
`train_gp`). On the bounded model the Laplace route met the other
constraint: the likelihood depends only on `p + u[0] + u[1]`, the
posterior is a ridge, the curvature along it gives a marginal SD of about
27, and the Laplace interval of `u` and `p` had to be clipped to the hard
box (recorded under `laplace_fallback`); the fit then succeeded, at 80
evaluations against 65 for the other models.

## What the prototype establishes

1. **The coordinate convention.** Handing VBMC the model's own
   coordinates is possible only for unbounded and two-sided variables;
   one-sided variables must keep PyMC's transform. This mixed convention
   is one call to `remove_value_transforms` with the variables to
   untransform, and PyMC supplies the Jacobian for the kept ones. All
   three kinds were exercised: unbounded (`mu`, `beta`), one-sided kept
   (`sigma`), two-sided removed with the unit interval (`p`) and with
   per-coordinate limits (`u`); the Jacobian increment agrees to 4e-12,
   the evidences to 0.011 nats, a point outside a bound gives `−inf`, and
   the exported posterior is over the model's variables.
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
   scalar model's 21 terms); the shape of the density agrees to 4e-12 and
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
   four models, provided that a curvature that is not negative or an
   interval that leaves the hard box (a ridge) falls back to the prior
   width or the box, which the bounded model required. An adapter that
   offers a default should offer the Laplace route with these fallbacks
   and accept explicit bounds.
6. **Cost.** Without a C++ compiler the compiled density runs in
   PyTensor's Python mode at 0.01–0.07 ms per call, so the target is not
   what VBMC's few hundred evaluations spend their time on; the fits took
   3 to 20 seconds and the mode-plus-Hessian setup under a second.

## Supported scope this implies for 1.5

Continuous free variables whose value transforms are none (with none as
the distribution's default), log, log-odds or an interval with fixed
limits; every other transform, a suppressed transform, a random interval
limit and every discrete variable rejected with the variable's name.
Deterministics and observed variables as the model defines them. Starting
point and plausible box from the Laplace route with its fallbacks by
default, with explicit bounds accepted. The automatic wrapper of the
proposal (initialization policy, several fits, everything returned) stays
deferred. The structured export is a small extension of
`VariationalPosterior.to_arviz` (dimensions and coordinates per variable)
and is useful without PyMC. Two things a shipped adapter would still have
to settle: a version range for PyMC and ArviZ beyond the one pair tested
here, and `VBMC.save`, which pickles the target callable and has not been
tried on an adapter holding compiled PyTensor functions and a PyMC model.
Whether the adapter ships in 1.5 with this scope, or only the structured
export, is the PI's decision on this record.

## Files

- `dev/scripts/pymc_feasibility.py`: the prototype (`--out DIR
  [--plausible laplace|prior] [--no-fit]`); its module docstring is the
  specification of the adapter's route through PyMC.
- `dev/experiments/pymc_feasibility/fit_laplace/`, `fit_prior/`,
  `nofit/`: `report.json` (environment with commits and the script's
  hash, every check with its numbers, the fits with their boxes, the
  return path, the rejections, the errors with the box and the log joint
  at its corners) and `report.md`.
