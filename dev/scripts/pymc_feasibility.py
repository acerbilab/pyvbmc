"""Bounded feasibility check of a PyMC model adapter for PyVBMC.

The proposal ``dev/2026-09-13-pymc-integration.md`` scopes a possible PyMC
integration for PyVBMC 1.5 as a *target adapter* (a PyMC model becomes a
callable log joint with the parameter mapping retained), a *structured
export* of the fitted posterior (draws with the model's variable names,
shapes and coordinates), and the *return path* into PyMC (deterministic
quantities and posterior predictions on those draws). Its "Feasibility
check" section asks for a small prototype over an ordinary scalar model, a
positive parameter and a vector parameter with a deterministic quantity,
checking pointwise log densities with normalization and Jacobians against
known expressions, sample shapes, names and coordinates, deterministic and
predictive calculations, and a clear rejection of unsupported models. This
script is that prototype. It changes nothing in the package: the adapter
lives here, and what it needs from PyMC is recorded so that the supported
model scope can be decided.

The adapter's route through PyMC:

- **Coordinates.** VBMC needs every parameter either unbounded or bounded
  on both sides (``vbmc:HalfBounds`` refuses a variable bounded on one
  side only), and a log joint in those coordinates. PyMC attaches a value
  transform to every constrained variable (``sigma_log__`` for a positive
  scale, an interval transform for a bounded one). The adapter keeps
  PyMC's transform for a variable bounded on one side (its value variable
  is unbounded, and ``Model.compile_logp(jacobian=True)`` adds the
  Jacobian, so VBMC's ELBO still estimates the model's evidence) and
  removes it for a variable bounded on both sides or unbounded
  (``pymc.model.transform.conditioning.remove_value_transforms`` with the
  variables to untransform), handing VBMC the interval as hard bounds and
  the density in the model's own coordinates there. Draws come back
  through the kept transforms' ``backward`` maps, so the exported
  posterior is over the model's variables (``sigma``, not
  ``sigma_log__``).
- **Bounds.** Read from the transform PyMC attached: none means
  unbounded, a log transform means positive (kept), a log-odds transform
  means the unit interval (removed), an interval transform gives its two
  limits (evaluated through the transform's ``args_fn`` on the variable's
  inputs, the one place the adapter reaches past documented attributes;
  removed when both limits are finite, kept otherwise). Any other
  transform (simplex, ordered, zero-sum, Cholesky) describes a support
  that is not a box, and any discrete free variable is not continuous:
  both are rejected before anything is compiled.
- **Mapping.** Free variables in model order, shapes from the initial
  point of the partially transformed model, flattened in C order into the
  vector VBMC sees; the inverse rebuilds named arrays from draws, maps
  kept transforms back, and hands the result to ``arviz.from_dict`` with
  the dimensions and coordinates the model declares.
- **Return path.** ``pymc.compute_deterministics`` on the posterior group
  and ``pymc.sample_posterior_predictive`` with the model, both on the
  structured draws.

Models (data simulated from fixed seeds; every density has a closed form
and every evidence a closed form or a one-dimensional quadrature):

1. ``scalar``: ``mu ~ Normal(0, 3)``, ``y ~ Normal(mu, 1.5)``, 20 data;
   conjugate, so the posterior and the evidence are analytic.
2. ``positive``: ``sigma ~ HalfNormal(2)``, ``y ~ Normal(0, sigma)``, 15
   data; the evidence by quadrature over ``sigma``; the log transform is
   kept and its Jacobian checked.
3. ``vector``: ``beta ~ Normal(0, 5, shape=2)`` with a named coordinate,
   ``sigma ~ HalfNormal(2)``, ``mu = Deterministic(X @ beta)``,
   ``y ~ Normal(mu, sigma)``, 30 data; the evidence by integrating the
   conjugate marginal over ``beta`` against the half-normal prior on
   ``sigma`` by quadrature.
4. ``discrete`` (``k ~ Poisson``) and ``simplex`` (``w ~ Dirichlet``):
   must be rejected with an informative error.

For every accepted model the script checks the adapter's density against
the hand-written one at random points (the shape to 1e-8 nats; a constant
offset of about 1e-8 nats per constant-scale term is PyTensor folding the
logarithm of a Python-float scale in float32, allowed up to 1e-6 and
reported), checks the Jacobian increment of the kept transforms against
``log sigma``, runs ``VBMC`` with PyVBMC defaults from a starting point
and plausible box chosen by ``--plausible`` (the proposal leaves both as
explicit user steps; the two routes here are the candidates an adapter
could offer), compares the ELBO with the evidence, exports draws through
the mapping, and runs the return path, checking names, dimensions,
shapes and, for the deterministic, the values against ``X @ beta``.

Usage (from a Python environment with PyMC, ArviZ and this checkout of
PyVBMC installed; the project venv has neither PyMC nor ArviZ)::

    python dev/scripts/pymc_feasibility.py --out DIR [--seed 0] [--no-fit] \\
        [--plausible laplace|prior]

``--plausible`` chooses the starting point and plausible box of every
fit: ``laplace`` (the default) takes the mode from ``pymc.find_MAP`` and
the box from a finite-difference Laplace approximation at the mode;
``prior`` takes the model's initial point and the 5 % and 95 % quantiles
of prior draws, which for a vague regression prior puts the box where the
log likelihood spans tens of thousands of nats and the GP fit of the
initial design fails. Outputs under ``--out``: ``report.json`` (versions,
the PyTensor compiler state, every check with its numbers, the VBMC
results, the rejections) and ``report.md`` (the same as tables).
``--no-fit`` runs every check that needs no VBMC fit.
"""

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

TOL_LOGP = 1e-8
#: Allowed constant offset of the compiled density from the hand-written
#: one (PyTensor's float32 folding of constant scales, see check_density).
TOL_OFFSET = 1e-6
N_CHECK_POINTS = 25
N_DRAWS = 2000
PRIOR_DRAWS = 4000
PLAUSIBLE_QUANTILES = (0.05, 0.95)


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


class UnsupportedModel(ValueError):
    """A PyMC model the adapter cannot represent as a box-bounded target."""


def _remove_value_transforms():
    try:
        from pymc.model.transform.conditioning import remove_value_transforms
    except ImportError:  # older layout
        from pymc.model.transform import remove_value_transforms
    return remove_value_transforms


def _support(rv, transform):
    """Hard bounds of one variable from the transform PyMC attached to it."""
    if transform is None:
        return -np.inf, np.inf
    name = type(transform).__name__
    if name == "LogTransform":
        return 0.0, np.inf
    if name == "LogOddsTransform":
        return 0.0, 1.0
    if name in ("IntervalTransform", "Interval"):
        args_fn = getattr(transform, "args_fn", None)
        if args_fn is None:
            raise UnsupportedModel(
                f"{rv.name}: interval transform without readable limits"
            )
        lower, upper = args_fn(*rv.owner.inputs)
        lower = -np.inf if lower is None else float(np.asarray(lower.eval()))
        upper = np.inf if upper is None else float(np.asarray(upper.eval()))
        return lower, upper
    raise UnsupportedModel(
        f"{rv.name}: the support of a variable with a {name} is not a box "
        "of independent bounds, which VBMC cannot represent"
    )


class PyMCTarget:
    """A PyMC model as a box-bounded log joint over a flat parameter vector.

    Attributes
    ----------
    names, shapes, sizes : list
        The free variables in model order, their shapes and sizes.
    value_names : list of str
        The name of each variable's coordinate as VBMC sees it: the
        variable's own name, or PyMC's transformed value variable
        (``sigma_log__``) for a variable bounded on one side.
    kept : dict
        For every variable whose transform is kept, the transform and the
        compiled forward and backward maps.
    D : int
        Length of the flat vector.
    lb, ub : np.ndarray, shape (1, D)
        Hard bounds per coordinate, in VBMC's coordinates.
    dims, coords : dict
        Dimension names per variable and coordinate values per dimension,
        as the model declares them (empty when it does not).
    """

    def __init__(self, model):
        import pymc as pm
        import pytensor
        import pytensor.tensor as pt

        self.model = model
        self.names = [rv.name for rv in model.free_RVs]
        for rv in model.free_RVs:
            if not np.issubdtype(np.dtype(rv.dtype), np.floating):
                raise UnsupportedModel(
                    f"{rv.name} is {rv.dtype}: VBMC needs continuous "
                    "parameters"
                )
        support = {
            rv.name: _support(rv, model.rvs_to_transforms[rv])
            for rv in model.free_RVs
        }
        # A variable bounded on one side keeps PyMC's transform (VBMC
        # cannot take a half-bounded box); every other one is handed over
        # in its own coordinates.
        untransform = [
            rv
            for rv in model.free_RVs
            if model.rvs_to_transforms[rv] is None
            or np.isfinite(support[rv.name]).all()
        ]
        self.partial = _remove_value_transforms()(model, vars=untransform)
        self.kept = {}
        for rv in self.partial.free_RVs:
            transform = self.partial.rvs_to_transforms[rv]
            if transform is None:
                continue
            ndim = rv.ndim + 1  # a leading axis of draws
            v = pt.TensorType("float64", shape=(None,) * ndim)(f"{rv.name}_v")
            self.kept[rv.name] = {
                "transform": type(transform).__name__,
                "backward": pytensor.function(
                    [v], transform.backward(v, *rv.owner.inputs)
                ),
                "forward": pytensor.function(
                    [v], transform.forward(v, *rv.owner.inputs)
                ),
            }
        by_name = {rv.name: rv for rv in self.partial.free_RVs}
        self.value_names = [
            self.partial.rvs_to_values[by_name[n]].name for n in self.names
        ]
        point = self.partial.initial_point()
        missing = [v for v in self.value_names if v not in point]
        if missing:
            raise UnsupportedModel(
                f"the initial point lacks {missing}: the value variables "
                "were not laid out as expected"
            )
        self.shapes = [np.asarray(point[v]).shape for v in self.value_names]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.D = int(sum(self.sizes))
        bounds = [
            (-np.inf, np.inf) if n in self.kept else support[n]
            for n in self.names
        ]
        self.support = support
        self.lb = np.concatenate(
            [np.full(k, b[0]) for b, k in zip(bounds, self.sizes)]
        ).reshape(1, -1)
        self.ub = np.concatenate(
            [np.full(k, b[1]) for b, k in zip(bounds, self.sizes)]
        ).reshape(1, -1)
        self.x0 = self.flatten(point).reshape(1, -1)
        self._logp = self.partial.compile_logp(jacobian=True)
        self._logp_plain = self.partial.compile_logp(jacobian=False)
        named_dims = getattr(model, "named_vars_to_dims", {})
        self.dims = {
            n: list(named_dims[n]) for n in self.names if n in named_dims
        }
        self.coords = {
            d: list(np.asarray(v).tolist())
            for d, v in getattr(model, "coords", {}).items()
            if v is not None
        }
        self.pm = pm

    # -- the flat vector and the two coordinate systems --------------------

    def flatten(self, point):
        """A dict over value names (VBMC's coordinates) to the flat vector."""
        return np.concatenate(
            [
                np.asarray(point[v], dtype=float).ravel()
                for v in self.value_names
            ]
        )

    def unflatten(self, x):
        """The flat vector to a dict over value names."""
        x = np.asarray(x, dtype=float).ravel()
        out, start = {}, 0
        for v, shape, k in zip(self.value_names, self.shapes, self.sizes):
            out[v] = x[start : start + k].reshape(shape)
            start += k
        return out

    def to_original(self, values, draws_axis=False):
        """Value-space arrays (dict over value names) to the model's variables.

        With ``draws_axis`` every array carries a leading axis of draws.
        """
        out = {}
        for n, v in zip(self.names, self.value_names):
            a = np.asarray(values[v], dtype=float)
            if n in self.kept:
                a = np.asarray(
                    self.kept[n]["backward"](a if draws_axis else a[None])
                )
                a = a if draws_axis else a[0]
            out[n] = a
        return out

    def to_values(self, original, draws_axis=False):
        """The model's variables to value space (the inverse of the above)."""
        out = {}
        for n, v in zip(self.names, self.value_names):
            a = np.asarray(original[n], dtype=float)
            if n in self.kept:
                a = np.asarray(
                    self.kept[n]["forward"](a if draws_axis else a[None])
                )
                a = a if draws_axis else a[0]
            out[v] = a
        return out

    # -- what VBMC calls ----------------------------------------------------

    def log_joint(self, x):
        """Log joint at one flat point in VBMC's coordinates.

        The model's density in its own variables plus the Jacobian of every
        kept transform; ``-inf`` outside the hard bounds.
        """
        x = np.asarray(x, dtype=float).ravel()
        if np.any(x <= self.lb.ravel()) or np.any(x >= self.ub.ravel()):
            return -np.inf
        return float(self._logp(self.unflatten(x)))

    def log_density_plain(self, x):
        """The model's density in its own variables, at a flat point."""
        return float(self._logp_plain(self.unflatten(np.asarray(x))))

    def plausible_bounds(self, rng, quantiles=PLAUSIBLE_QUANTILES):
        """Plausible bounds from prior quantiles, strictly inside the box."""
        draws = self.pm.draw(
            self.model.free_RVs, draws=PRIOR_DRAWS, random_seed=rng
        )
        original = {
            n: np.asarray(d, dtype=float) for n, d in zip(self.names, draws)
        }
        values = self.to_values(original, draws_axis=True)
        flat = np.column_stack(
            [values[v].reshape(PRIOR_DRAWS, -1) for v in self.value_names]
        )
        plb = np.quantile(flat, quantiles[0], axis=0)
        pub = np.quantile(flat, quantiles[1], axis=0)
        lb, ub = self.lb.ravel(), self.ub.ravel()
        width = pub - plb
        plb = np.where(
            np.isfinite(lb), np.maximum(plb, lb + 0.01 * width), plb
        )
        pub = np.where(
            np.isfinite(ub), np.minimum(pub, ub - 0.01 * width), pub
        )
        return plb.reshape(1, -1), pub.reshape(1, -1)

    def laplace_box(self, k=3.0, step=1e-4):
        """Starting point and plausible bounds from the mode and its curvature.

        ``pymc.find_MAP`` on the original model (whose variables are all
        transformed, so the optimizer runs unconstrained) gives the mode;
        with ``include_transformed=True`` the returned point carries every
        name this adapter's coordinates use. The curvature is a central
        finite-difference Hessian of the adapter's own log joint at the
        mode, so no coordinate convention has to be matched with PyMC's;
        the plausible box is the mode plus and minus ``k`` marginal
        standard deviations of the resulting Gaussian, clipped strictly
        inside the hard bounds.
        """
        point = self.pm.find_MAP(
            model=self.model, include_transformed=True, progressbar=False
        )
        x = self.flatten({v: point[v] for v in self.value_names})
        D = self.D
        H = np.empty((D, D))
        f0 = self.log_joint(x)
        h = step * np.maximum(1.0, np.abs(x))
        for i in range(D):
            for j in range(i, D):
                e_i, e_j = np.zeros(D), np.zeros(D)
                e_i[i], e_j[j] = h[i], h[j]
                if i == j:
                    H[i, i] = (
                        self.log_joint(x + e_i)
                        - 2 * f0
                        + self.log_joint(x - e_i)
                    ) / h[i] ** 2
                else:
                    H[i, j] = H[j, i] = (
                        self.log_joint(x + e_i + e_j)
                        - self.log_joint(x + e_i - e_j)
                        - self.log_joint(x - e_i + e_j)
                        + self.log_joint(x - e_i - e_j)
                    ) / (4 * h[i] * h[j])
        cov = np.linalg.inv(-H)
        sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
        plb, pub = x - k * sd, x + k * sd
        lb, ub = self.lb.ravel(), self.ub.ravel()
        width = pub - plb
        plb = np.where(
            np.isfinite(lb), np.maximum(plb, lb + 0.01 * width), plb
        )
        pub = np.where(
            np.isfinite(ub), np.minimum(pub, ub - 0.01 * width), pub
        )
        return x.reshape(1, -1), plb.reshape(1, -1), pub.reshape(1, -1)

    # -- the structured export ---------------------------------------------

    def to_arviz(self, X):
        """Structured posterior draws over the model's variables.

        One ``(chain, draw, *shape)`` array per variable, in the model's
        coordinates, with the model's dimension names and coordinates.
        """
        import arviz as az

        X = np.asarray(X, dtype=float)
        values, start = {}, 0
        for v, shape, k in zip(self.value_names, self.shapes, self.sizes):
            values[v] = X[:, start : start + k].reshape((len(X),) + shape)
            start += k
        original = self.to_original(values, draws_axis=True)
        posterior = {n: a[None] for n, a in original.items()}
        dims = self.dims or None
        coords = self.coords or None
        try:  # ArviZ 1.x: a dict of groups
            return az.from_dict(
                {"posterior": posterior}, dims=dims, coords=coords
            )
        except TypeError:  # ArviZ 0.x: one keyword per group
            return az.from_dict(posterior=posterior, dims=dims, coords=coords)


def posterior_dataset(data):
    """The posterior group of an InferenceData or DataTree as a Dataset."""
    group = data["posterior"] if "posterior" in data else data.posterior
    return group.to_dataset() if hasattr(group, "to_dataset") else group


# --------------------------------------------------------------------------
# Models with hand-written densities and evidences
# --------------------------------------------------------------------------


def _log_normal(x, mean, sd):
    x = np.asarray(x, dtype=float)
    return -0.5 * np.log(2 * np.pi) - np.log(sd) - 0.5 * ((x - mean) / sd) ** 2


def _log_halfnormal(x, sd):
    x = np.asarray(x, dtype=float)
    return np.where(
        x > 0,
        0.5 * np.log(2 / np.pi) - np.log(sd) - 0.5 * (x / sd) ** 2,
        -np.inf,
    )


def _mvn_logpdf(y, cov):
    sign, logdet = np.linalg.slogdet(cov)
    solve = np.linalg.solve(cov, y)
    return float(-0.5 * (len(y) * np.log(2 * np.pi) + logdet + y @ solve))


def model_scalar(rng):
    import pymc as pm

    y = rng.normal(1.2, 1.5, size=20)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 3.0)
        pm.Normal("y", mu, 1.5, observed=y)

    def hand(point):
        return float(
            _log_normal(point["mu"], 0.0, 3.0)
            + np.sum(_log_normal(y, point["mu"], 1.5))
        )

    ln_Z = _mvn_logpdf(
        y, 1.5**2 * np.eye(len(y)) + 9.0 * np.ones((len(y), len(y)))
    )
    precision = 1 / 9.0 + len(y) / 1.5**2
    return {
        "name": "scalar",
        "model": model,
        "hand": hand,
        "ln_Z": ln_Z,
        "truth": {
            "mu_mean": float(np.sum(y) / 1.5**2 / precision),
            "mu_sd": float(np.sqrt(1 / precision)),
        },
        "reference": "analytic (conjugate normal-normal)",
    }


def model_positive(rng):
    import pymc as pm
    from scipy.integrate import quad

    y = rng.normal(0.0, 0.8, size=15)
    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", 2.0)
        pm.Normal("y", 0.0, sigma, observed=y)

    def hand(point):
        s = float(point["sigma"])
        if s <= 0:
            return -np.inf
        return float(_log_halfnormal(s, 2.0) + np.sum(_log_normal(y, 0.0, s)))

    peak = hand({"sigma": float(np.std(y))})
    value, error = quad(
        lambda s: np.exp(hand({"sigma": s}) - peak), 1e-9, 20.0, limit=200
    )
    return {
        "name": "positive",
        "model": model,
        "hand": hand,
        "ln_Z": float(np.log(value) + peak),
        "ln_Z_error": float(error / value),
        "truth": {},
        "reference": "one-dimensional quadrature over sigma",
    }


def model_vector(rng):
    import pymc as pm
    from scipy.integrate import quad

    n = 30
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    beta_true = np.array([0.5, -1.0])
    y = X @ beta_true + rng.normal(0.0, 0.7, size=n)
    with pm.Model(coords={"coef": ["intercept", "slope"]}) as model:
        beta = pm.Normal("beta", 0.0, 5.0, dims="coef")
        sigma = pm.HalfNormal("sigma", 2.0)
        mu = pm.Deterministic("mu", X @ beta)
        pm.Normal("y", mu, sigma, observed=y)

    def hand(point):
        b, s = np.asarray(point["beta"], dtype=float), float(point["sigma"])
        if s <= 0:
            return -np.inf
        return float(
            np.sum(_log_normal(b, 0.0, 5.0))
            + _log_halfnormal(s, 2.0)
            + np.sum(_log_normal(y, X @ b, s))
        )

    def marginal(s):
        return _mvn_logpdf(y, s**2 * np.eye(n) + 25.0 * X @ X.T) + float(
            _log_halfnormal(s, 2.0)
        )

    peak = marginal(0.7)
    value, error = quad(
        lambda s: np.exp(marginal(s) - peak), 1e-9, 20.0, limit=200
    )
    return {
        "name": "vector",
        "model": model,
        "hand": hand,
        "ln_Z": float(np.log(value) + peak),
        "ln_Z_error": float(error / value),
        "truth": {},
        "reference": "beta marginalized analytically, quadrature over sigma",
        "X": X,
    }


def model_discrete(rng):
    import pymc as pm

    with pm.Model() as model:
        k = pm.Poisson("k", 3.0)
        pm.Normal("y", k, 1.0, observed=rng.normal(3.0, 1.0, size=5))
    return {"name": "discrete", "model": model}


def model_simplex(rng):
    import pymc as pm

    with pm.Model() as model:
        w = pm.Dirichlet("w", np.ones(3))
        pm.Multinomial("counts", 20, w, observed=np.array([5, 7, 8]))
    return {"name": "simplex", "model": model}


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_density(spec, target, rng):
    """The adapter's density against the hand-written one at random points.

    The plain density (no Jacobian) at a point of VBMC's coordinates must
    equal the hand-written log joint at the corresponding model variables.
    The difference is split into a constant ``offset`` (its mean over the
    points, a normalization constant) and the deviation from it (the shape
    of the density). PyTensor folds the logarithm of a Python-float scale
    such as ``Normal(mu, 1.5)`` into a float32 constant, so a model with
    constant scales carries an offset of about 1e-8 nats per such term;
    the shape must agree to ``TOL_LOGP`` and the offset to ``TOL_OFFSET``.
    """
    plb, pub = target.plausible_bounds(rng)
    points = rng.uniform(plb, pub, size=(N_CHECK_POINTS, target.D))
    differences = np.asarray(
        [
            target.log_density_plain(x)
            - spec["hand"](target.to_original(target.unflatten(x)))
            for x in points
        ]
    )
    offset = float(np.mean(differences))
    deviation = float(np.max(np.abs(differences - offset)))
    lb = target.lb.ravel()
    below = None
    if np.any(np.isfinite(lb)):
        x = points[0].copy()
        i = int(np.flatnonzero(np.isfinite(lb))[0])
        x[i] = lb[i] - 1.0
        below = target.log_joint(x)
    return {
        "points": int(len(points)),
        "max_abs_difference": float(np.max(np.abs(differences))),
        "offset": offset,
        "max_abs_deviation_from_offset": deviation,
        "passes": bool(deviation <= TOL_LOGP and abs(offset) <= TOL_OFFSET),
        "outside_bounds_value": None if below is None else float(below),
    }


def check_jacobian(spec, target, rng):
    """The Jacobian increment of the kept transforms against ``log sigma``.

    ``log_joint - log_density_plain`` is what ``compile_logp(jacobian=True)``
    adds for the kept transforms; for a log transform of a scalar it is the
    value variable itself, ``log sigma``. The constant offsets of the
    folded scales cancel in the increment.
    """
    if not target.kept:
        return {"applicable": False}
    kinds = {n: k["transform"] for n, k in target.kept.items()}
    if any(kind != "LogTransform" for kind in kinds.values()):
        return {"applicable": True, "kept": kinds, "checked": False}
    plb, pub = target.plausible_bounds(rng)
    points = rng.uniform(plb, pub, size=(N_CHECK_POINTS, target.D))
    worst = 0.0
    for x in points:
        values = target.unflatten(x)
        expected = sum(
            float(np.sum(values[v]))
            for n, v in zip(target.names, target.value_names)
            if n in target.kept
        )
        increment = target.log_joint(x) - target.log_density_plain(x)
        worst = max(worst, abs(increment - expected))
    return {
        "applicable": True,
        "kept": kinds,
        "checked": True,
        "max_abs_difference": float(worst),
        "passes": bool(worst <= TOL_LOGP),
    }


def fit(spec, target, rng, seed, plausible):
    """One VBMC run from the chosen starting point and plausible box.

    ``plausible="prior"`` starts from the model's initial point inside a
    box of prior quantiles; ``plausible="laplace"`` starts from the mode
    inside the mode's Laplace box (:meth:`PyMCTarget.laplace_box`).
    """
    from pyvbmc import VBMC

    started = time.perf_counter()
    if plausible == "prior":
        x0 = target.x0
        plb, pub = target.plausible_bounds(rng)
    else:
        x0, plb, pub = target.laplace_box()
    setup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vbmc = VBMC(
        target.log_joint,
        x0,
        target.lb,
        target.ub,
        plb,
        pub,
        options={"display": "off", "plot": False},
        seed=seed,
    )
    vp, results = vbmc.optimize()
    seconds = time.perf_counter() - started
    out = {
        "plausible": plausible,
        "setup_seconds": setup_seconds,
        "seconds": seconds,
        "func_count": int(results["func_count"]),
        "iterations": int(results["iterations"]),
        "success_flag": bool(results["success_flag"]),
        "elbo": float(results["elbo"]),
        "elbo_sd": float(results["elbo_sd"]),
        "ln_Z": spec["ln_Z"],
        "elbo_minus_ln_Z": float(results["elbo"] - spec["ln_Z"]),
        "K": int(vp.K),
        "plb": plb.ravel().tolist(),
        "pub": pub.ravel().tolist(),
        "x0": np.ravel(x0).tolist(),
    }
    draws = vp.sample(N_DRAWS)[0]
    values = {
        v: draws[:, start : start + k].reshape((len(draws),) + shape)
        for v, shape, k, start in zip(
            target.value_names,
            target.shapes,
            target.sizes,
            np.concatenate([[0], np.cumsum(target.sizes)[:-1]]).astype(int),
        )
    }
    original = target.to_original(values, draws_axis=True)
    out["posterior_mean"] = {
        n: np.asarray(a.mean(axis=0)).tolist() for n, a in original.items()
    }
    out["posterior_sd"] = {
        n: np.asarray(a.std(axis=0)).tolist() for n, a in original.items()
    }
    if spec["truth"]:
        out["truth"] = spec["truth"]
        out["mu_mean_error"] = float(
            original["mu"].mean() - spec["truth"]["mu_mean"]
        )
        out["mu_sd_ratio"] = float(
            original["mu"].std() / spec["truth"]["mu_sd"]
        )
    return out, vp, draws


def check_return_path(spec, target, draws):
    """Structured export, deterministics and posterior predictions."""
    import pymc as pm

    data = target.to_arviz(draws)
    posterior = posterior_dataset(data)
    report = {
        "container": type(data).__name__,
        "variables": {
            n: {
                "dims": list(posterior[n].dims),
                "shape": list(posterior[n].shape),
            }
            for n in target.names
        },
        "coords": {
            d: np.asarray(posterior[d].values).tolist()
            for d in posterior.dims
            if d not in ("chain", "draw")
        },
    }
    model = spec["model"]
    if model.deterministics:
        det = pm.compute_deterministics(
            posterior, model=model, progressbar=False
        )
        name = model.deterministics[0].name
        report["deterministic"] = {
            "name": name,
            "dims": list(det[name].dims),
            "shape": list(det[name].shape),
        }
        if "X" in spec:
            expected = np.einsum(
                "nk,dk->dn", spec["X"], posterior["beta"].values[0]
            )
            report["deterministic"]["max_abs_difference_from_X_beta"] = float(
                np.max(np.abs(det[name].values[0] - expected))
            )
    predictive = pm.sample_posterior_predictive(
        data, model=model, progressbar=False, random_seed=1
    )
    group = predictive["posterior_predictive"]
    group = group.to_dataset() if hasattr(group, "to_dataset") else group
    observed = model.observed_RVs[0].name
    report["posterior_predictive"] = {
        "name": observed,
        "dims": list(group[observed].dims),
        "shape": list(group[observed].shape),
    }
    return report


def check_rejection(spec):
    try:
        PyMCTarget(spec["model"])
    except UnsupportedModel as error:
        return {"rejected": True, "message": str(error)}
    except Exception as error:  # noqa: BLE001
        return {
            "rejected": False,
            "message": f"unexpected {type(error).__name__}: {error}",
        }
    return {"rejected": False, "message": "accepted"}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def environment():
    import arviz
    import gpyreg
    import pymc
    import pytensor
    import scipy

    import pyvbmc

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pymc": pymc.__version__,
        "pytensor": pytensor.__version__,
        "pytensor_cxx": repr(pytensor.config.cxx),
        "pytensor_mode": str(pytensor.config.mode),
        "pytensor_linker": str(pytensor.config.linker),
        "pytensor_floatX": str(pytensor.config.floatX),
        "arviz": arviz.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyvbmc": str(Path(pyvbmc.__file__).resolve().parent),
        "gpyreg": str(Path(gpyreg.__file__).resolve().parent),
        "executable": sys.executable,
    }


def markdown(report):
    env = report["environment"]
    lines = [
        "# PyMC adapter feasibility check",
        "",
        f"Generated {report['generated']}; PyMC {env['pymc']}, PyTensor "
        f"{env['pytensor']} (C++ compiler {env['pytensor_cxx']}, mode "
        f"{env['pytensor_mode']}, floatX {env['pytensor_floatX']}), ArviZ "
        f"{env['arviz']}, Python {env['python']}. Starting point and "
        f"plausible box: `{report.get('plausible', 'prior')}` (prior "
        "quantiles from the model's initial point, or the mode and its "
        "Laplace box).",
        "",
        "| Model | D | VBMC coordinates and bounds | density (offset; "
        "max |Δ − offset|) | Jacobian increment | ELBO − ln Z (± sd) | "
        "evaluations | seconds | return path |",
        "|---|---:|---|---:|---|---:|---:|---:|---|",
    ]
    for m in report["models"]:
        density = m.get("density", {})
        jac = m.get("jacobian", {})
        fitted = m.get("fit")
        rp = m.get("return_path", {})
        if not jac.get("applicable"):
            jac_text = "n/a (no transform kept)"
        elif not jac.get("checked"):
            jac_text = f"kept {jac['kept']}, not checked"
        else:
            jac_text = f"{jac['max_abs_difference']:.1e}" + (
                " ok" if jac["passes"] else " FAIL"
            )
        if rp:
            rp_text = "; ".join(
                f"{n} {v['dims']} {v['shape']}"
                for n, v in rp["variables"].items()
            )
            if "deterministic" in rp:
                rp_text += (
                    f"; {rp['deterministic']['name']} "
                    f"{rp['deterministic']['shape']}"
                )
                if "max_abs_difference_from_X_beta" in rp["deterministic"]:
                    rp_text += (
                        " (max |Δ| from X beta "
                        f"{rp['deterministic']['max_abs_difference_from_X_beta']:.1e})"
                    )
            rp_text += f"; predictive {rp['posterior_predictive']['shape']}"
        else:
            rp_text = "not run"
        lines.append(
            f"| {m['name']} | {m.get('D', '-')} | {m.get('coordinates', '-')} | "
            f"{density.get('offset', float('nan')):+.1e}; "
            f"{density.get('max_abs_deviation_from_offset', float('nan')):.1e}"
            + (" ok" if density.get("passes") else " FAIL")
            + f" | {jac_text} | "
            + (
                f"{fitted['elbo_minus_ln_Z']:+.3f} (± {fitted['elbo_sd']:.3f})"
                if fitted
                else "not run"
            )
            + " | "
            + (str(fitted["func_count"]) if fitted else "-")
            + " | "
            + (f"{fitted['seconds']:.0f}" if fitted else "-")
            + f" | {rp_text} |"
        )
    lines += ["", "## Rejections", ""]
    for r in report["rejections"]:
        lines.append(
            f"- `{r['name']}`: "
            + ("rejected: " if r["rejected"] else "NOT rejected: ")
            + r["message"]
        )
    if report.get("errors"):
        lines += ["", "## Errors", ""]
        for e in report["errors"]:
            lines.append(f"- {e['where']}: `{e['error']}`")
    return "\n".join(lines) + "\n"


def describe_coordinates(target):
    parts = []
    for n, v, (lo, hi) in zip(
        target.names,
        target.value_names,
        zip(
            target.lb.ravel()[np.cumsum([0] + target.sizes[:-1])],
            target.ub.ravel()[np.cumsum([0] + target.sizes[:-1])],
        ),
    ):
        if n in target.kept:
            parts.append(
                f"{v} ({target.kept[n]['transform']} kept, unbounded)"
            )
        else:
            parts.append(f"{n} [{lo:g}, {hi:g}]")
    return "; ".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-fit", action="store_true")
    parser.add_argument(
        "--plausible", choices=("prior", "laplace"), default="laplace"
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": environment(),
        "seed": args.seed,
        "plausible": args.plausible,
        "models": [],
        "rejections": [],
        "errors": [],
    }
    rng = np.random.default_rng(args.seed)
    for build in (model_scalar, model_positive, model_vector):
        spec = build(rng)
        entry = {"name": spec["name"], "reference": spec["reference"]}
        try:
            target = PyMCTarget(spec["model"])
            entry.update(
                {
                    "D": target.D,
                    "names": target.names,
                    "value_names": target.value_names,
                    "kept_transforms": {
                        n: k["transform"] for n, k in target.kept.items()
                    },
                    "shapes": [list(s) for s in target.shapes],
                    "support": {
                        n: [float(lo), float(hi)]
                        for n, (lo, hi) in target.support.items()
                    },
                    "coordinates": describe_coordinates(target),
                    "dims": target.dims,
                    "coords": target.coords,
                    "ln_Z": spec["ln_Z"],
                    "density": check_density(spec, target, rng),
                    "jacobian": check_jacobian(spec, target, rng),
                }
            )
            print(
                f"[{spec['name']}] D={target.D}; {entry['coordinates']}; "
                f"density offset {entry['density']['offset']:+.1e}, max "
                f"|diff - offset| "
                f"{entry['density']['max_abs_deviation_from_offset']:.1e}"
                + (
                    f"; Jacobian increment max |diff| "
                    f"{entry['jacobian']['max_abs_difference']:.1e}"
                    if entry["jacobian"].get("checked")
                    else ""
                ),
                flush=True,
            )
            if not args.no_fit:
                fitted, vp, draws = fit(
                    spec, target, rng, args.seed, args.plausible
                )
                entry["fit"] = fitted
                print(
                    f"[{spec['name']}] VBMC: elbo {fitted['elbo']:.3f} ± "
                    f"{fitted['elbo_sd']:.3f}, ln Z {spec['ln_Z']:.3f}, "
                    f"{fitted['func_count']} evaluations, "
                    f"{fitted['seconds']:.0f} s",
                    flush=True,
                )
                entry["return_path"] = check_return_path(spec, target, draws)
                print(
                    f"[{spec['name']}] return path: {entry['return_path']}",
                    flush=True,
                )
        except Exception as error:  # noqa: BLE001
            report["errors"].append(
                {
                    "where": spec["name"],
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                f"[{spec['name']}] ERROR {type(error).__name__}: {error}",
                flush=True,
            )
        report["models"].append(entry)
    for build in (model_discrete, model_simplex):
        spec = build(rng)
        outcome = check_rejection(spec)
        outcome["name"] = spec["name"]
        report["rejections"].append(outcome)
        print(f"[{spec['name']}] {outcome}", flush=True)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8"
    )
    (args.out / "report.md").write_text(markdown(report), encoding="utf-8")
    print(f"report under {args.out}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
