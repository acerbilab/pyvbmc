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
  variables to untransform), handing VBMC the interval as hard bounds,
  coordinate by coordinate, and the density in the model's own
  coordinates there. Draws come back through the kept transforms'
  ``backward`` maps, so the exported posterior is over the model's
  variables (``sigma``, not ``sigma_log__``).
- **Bounds.** Read from the transform PyMC attached: a log transform
  means positive (kept), a log-odds transform means the unit interval
  (removed), an interval transform gives its two limits, evaluated per
  coordinate through the transform's ``args_fn`` on the variable's inputs
  (removed when both limits are finite everywhere, kept when exactly one
  limit is finite everywhere, rejected when the coordinates disagree). A
  limit that depends on another random variable is rejected, since its
  value would be a draw. No transform means unbounded only when the
  distribution's default transform is none too; a variable whose default
  transform was suppressed at construction (``default_transform=None``)
  is rejected, since its density would be handed over as unbounded while
  its support is not. Any other transform (simplex, ordered, zero-sum,
  Cholesky, softplus) and any discrete free variable are rejected before
  anything is compiled.
- **Mapping.** Free variables in model order, shapes from the initial
  point of the partially transformed model, flattened in C order into the
  vector VBMC sees; the inverse rebuilds named arrays from draws, maps
  kept transforms back, and hands the result to ``arviz.from_dict`` (the
  ArviZ 1.x signature: a dict of groups, returning a ``DataTree``) with
  the dimensions and coordinates the model declares.
- **Return path.** ``pymc.compute_deterministics`` on the posterior group
  and ``pymc.sample_posterior_predictive`` with the model, both on the
  structured draws.
- **Starting point and plausible box** (``--plausible``). ``laplace``
  (the default): ``pymc.find_MAP`` on the fully transformed model, whose
  returned point (``include_transformed=True``) carries every coordinate
  this adapter uses, then a central finite-difference Hessian of the
  adapter's own log joint at that point and a box of three marginal
  standard deviations, clipped strictly inside the hard bounds. The
  point is the mode of the model's density in its own variables
  (``find_MAP`` maximizes the density without the Jacobian), so for a
  kept transform it is not exactly the stationary point of the adapter's
  target; a coordinate whose curvature is not negative there falls back
  to the prior-quantile width, and one whose Laplace interval leaves the
  hard box (a ridge of the posterior) is clipped to the middle 98 % of
  the box, both recorded (``laplace_fallback``). ``prior``: the model's
  initial point and the 5 % and 95 % quantiles of prior draws
  (``pymc.draw``), which for a vague regression prior puts the box where
  the log likelihood spans tens of thousands of nats and the GP fit of
  the initial design fails.

Models (data simulated from a generator per model keyed by ``--seed`` and
the model's position, so the same data appear on both routes; every
density has a closed form and every evidence a closed form or a
low-dimensional quadrature):

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
4. ``bounded``: ``p ~ Beta(2, 2)`` (log-odds transform, removed),
   ``u ~ Uniform([-1, 0], [2, 1], shape=2)`` (interval transform with
   per-coordinate limits, removed), ``y ~ Normal(p + u[0] + u[1], 1)``,
   12 data; the evidence by integrating over ``u[0]`` in closed form and
   over ``p`` and ``u[1]`` by quadrature.
5. ``one_sided``: ``t ~ TruncatedNormal(0, 2, lower=1)`` and
   ``v ~ TruncatedNormal(0, 2, upper=-0.5)`` (interval transforms with
   one infinite limit, kept), each with 8 normal observations; the
   evidence is a product of two one-dimensional quadratures and the
   Jacobian increment of both kept transforms is checked.
6. Rejected: ``discrete`` (``k ~ Poisson``), ``simplex`` (``w ~
   Dirichlet``), ``suppressed`` (``HalfNormal(default_transform=None)``)
   and ``random_bounds`` (``Uniform(lower=mu, upper=mu + 1)`` with ``mu``
   a free variable).

For every accepted model the script checks the adapter's density against
the hand-written one at random points (the shape to 1e-8 nats; a constant
offset of about 1e-8 nats per constant-scale term is PyTensor folding the
logarithm of a Python-float scale in float32, allowed up to 1e-6 and
reported), probes a point outside a finite bound, checks the Jacobian
increment of the kept transforms against ``log sigma``, times the compiled
density, runs ``VBMC`` with PyVBMC defaults from the chosen starting point
and plausible box, compares the ELBO with the evidence, exports draws
through the mapping, and runs the return path, checking names,
dimensions, shapes and, for the deterministic, the values against
``X @ beta``. A fit that fails records the box it was given and the log
joint at its corners with the error.

Usage (from a Python environment with PyMC 6.3, ArviZ 1.x and this
checkout of PyVBMC installed; the project venv has neither PyMC nor
ArviZ; the versions the script was written against are the tested
ones)::

    python dev/scripts/pymc_feasibility.py --out DIR [--seed 0] [--no-fit] \\
        [--plausible laplace|prior]

Outputs under ``--out``: ``report.json`` (versions, the PyTensor compiler
state, the PyVBMC and gpyreg commits and this file's hash, every check
with its numbers, the fits, the return path, the rejections, the errors
with their context) and ``report.md`` (the same as tables). ``--no-fit``
runs every check that needs no VBMC fit.
"""

import argparse
import hashlib
import json
import platform
import subprocess
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
N_TIMING_CALLS = 200
LAPLACE_K = 3.0


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


class UnsupportedModel(ValueError):
    """A PyMC model the adapter cannot represent as a box-bounded target."""


class FitFailure(RuntimeError):
    """A VBMC fit that failed, carrying the box it was given."""

    def __init__(self, context, error):
        super().__init__(f"{type(error).__name__}: {error}")
        self.context = context
        self.error = error


def _has_random_ancestor(expression):
    from pytensor.graph.traversal import ancestors
    from pytensor.tensor.random.op import RandomVariable

    return any(
        node.owner is not None and isinstance(node.owner.op, RandomVariable)
        for node in ancestors([expression])
    )


def _limit(rv, bound, shape, default):
    """One interval limit as an array of the variable's shape."""
    if bound is None:
        return np.full(shape, default, dtype=float)
    if hasattr(bound, "eval"):
        if _has_random_ancestor(bound):
            raise UnsupportedModel(
                f"{rv.name}: an interval limit depends on another random "
                "variable, so it has no fixed value"
            )
        bound = bound.eval()
    return np.broadcast_to(np.asarray(bound, dtype=float), shape).copy()


def _support(rv, transform, default_transform, shape):
    """Hard bounds of one variable, per coordinate, from its transform."""
    if transform is None:
        if default_transform is not None:
            raise UnsupportedModel(
                f"{rv.name}: its {type(default_transform).__name__} was "
                "suppressed at construction, so its support is not the "
                "unbounded one an untransformed variable would be handed "
                "over with"
            )
        return np.full(shape, -np.inf), np.full(shape, np.inf)
    name = type(transform).__name__
    if name == "LogTransform":
        return np.zeros(shape), np.full(shape, np.inf)
    if name == "LogOddsTransform":
        return np.zeros(shape), np.ones(shape)
    if name in ("IntervalTransform", "Interval"):
        args_fn = getattr(transform, "args_fn", None)
        if args_fn is None:
            raise UnsupportedModel(
                f"{rv.name}: interval transform without readable limits"
            )
        lower, upper = args_fn(*rv.owner.inputs)
        return (
            _limit(rv, lower, shape, -np.inf),
            _limit(rv, upper, shape, np.inf),
        )
    raise UnsupportedModel(f"{rv.name}: unsupported transform {name}")


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
        For every variable whose transform is kept, the transform's class
        name and the compiled forward and backward maps.
    support : dict
        Per variable, the hard bounds of every coordinate in the model's
        own variables, as ``(lower, upper)`` arrays of the variable's
        shape.
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
        from pymc.distributions.transforms import _default_transform
        from pymc.model.transform.conditioning import remove_value_transforms

        self.model = model
        self.names = [rv.name for rv in model.free_RVs]
        for rv in model.free_RVs:
            if not np.issubdtype(np.dtype(rv.dtype), np.floating):
                raise UnsupportedModel(
                    f"{rv.name} is {rv.dtype}: VBMC needs continuous "
                    "parameters"
                )
        point = model.initial_point()
        self.support, untransform = {}, []
        for rv in model.free_RVs:
            transform = model.rvs_to_transforms[rv]
            shape = np.asarray(point[model.rvs_to_values[rv].name]).shape
            lower, upper = _support(
                rv, transform, _default_transform(rv.owner.op, rv), shape
            )
            self.support[rv.name] = (lower, upper)
            two_sided = np.isfinite(lower) & np.isfinite(upper)
            unbounded = ~np.isfinite(lower) & ~np.isfinite(upper)
            if transform is None or np.all(two_sided):
                # Handed over in the model's own coordinates: VBMC takes
                # the box (or no box) itself.
                untransform.append(rv)
            elif np.any(two_sided | unbounded):
                raise UnsupportedModel(
                    f"{rv.name}: its coordinates mix one-sided and other "
                    "bounds, which one transform cannot represent for VBMC"
                )
        self.partial = remove_value_transforms(model, vars=untransform)
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
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)]).astype(int)
        lb, ub = [], []
        for n, shape, k in zip(self.names, self.shapes, self.sizes):
            if n in self.kept:
                lb.append(np.full(k, -np.inf))
                ub.append(np.full(k, np.inf))
            else:
                lb.append(self.support[n][0].reshape(k))
                ub.append(self.support[n][1].reshape(k))
        self.lb = np.concatenate(lb).reshape(1, -1)
        self.ub = np.concatenate(ub).reshape(1, -1)
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
        return {
            v: x[self.offsets[i] : self.offsets[i + 1]].reshape(shape)
            for i, (v, shape) in enumerate(zip(self.value_names, self.shapes))
        }

    def values_from_draws(self, X):
        """Flat draws ``(n, D)`` to a dict of ``(n, *shape)`` value arrays."""
        X = np.asarray(X, dtype=float)
        return {
            v: X[:, self.offsets[i] : self.offsets[i + 1]].reshape(
                (len(X),) + shape
            )
            for i, (v, shape) in enumerate(zip(self.value_names, self.shapes))
        }

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

    def _clip(self, plb, pub):
        """Plausible bounds strictly inside the hard box, 1 % of it away.

        A coordinate whose plausible interval collapses or falls outside
        the box after clipping is widened to the middle 98 % of the box
        (both bounds finite) or left as given (unbounded).
        """
        lb, ub = self.lb.ravel(), self.ub.ravel()
        plb, pub = (
            np.array(plb, dtype=float).ravel(),
            np.array(pub, dtype=float).ravel(),
        )
        finite = np.isfinite(lb) & np.isfinite(ub)
        margin = np.where(finite, 0.01 * (ub - lb), 0.0)
        clipped = finite & ((plb < lb + margin) | (pub > ub - margin))
        plb = np.where(finite, np.maximum(plb, lb + margin), plb)
        pub = np.where(finite, np.minimum(pub, ub - margin), pub)
        bad = finite & ~(pub - plb > 0)
        plb = np.where(bad, lb + margin, plb)
        pub = np.where(bad, ub - margin, pub)
        return plb.reshape(1, -1), pub.reshape(1, -1), clipped | bad

    def _coordinate_names(self, mask):
        return [
            self.value_names[
                int(np.searchsorted(self.offsets, i, side="right")) - 1
            ]
            for i in np.flatnonzero(mask)
        ]

    def prior_box(self, rng, quantiles=PLAUSIBLE_QUANTILES):
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
        plb, pub, _ = self._clip(
            np.quantile(flat, quantiles[0], axis=0),
            np.quantile(flat, quantiles[1], axis=0),
        )
        return plb, pub

    def laplace_box(self, rng, k=LAPLACE_K, step=1e-4):
        """Starting point and plausible bounds from the mode and curvature.

        ``pymc.find_MAP`` on the original model (whose variables are all
        transformed, so the optimizer runs unconstrained) gives the mode
        of the model's density in its own variables; with
        ``include_transformed=True`` the returned point carries every name
        this adapter's coordinates use. ``find_MAP`` maximizes the density
        without the Jacobian, so for a kept transform the point is not
        exactly the stationary point of the adapter's log joint; the
        curvature is a central finite-difference Hessian of that log joint
        at the point anyway, and a coordinate whose curvature is not
        negative there takes the prior-quantile width instead (recorded
        in ``fallback``). The box is the point plus and minus ``k``
        marginal standard deviations, clipped strictly inside the hard
        bounds.
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
        fallback = []
        try:
            cov = np.linalg.inv(-H)
            sd = np.sqrt(np.diag(cov))
            usable = np.isfinite(sd) & (sd > 0)
        except np.linalg.LinAlgError:
            sd = np.zeros(D)
            usable = np.zeros(D, dtype=bool)
        if not np.all(usable):
            plb_prior, pub_prior = self.prior_box(rng)
            half = 0.5 * (pub_prior - plb_prior).ravel() / k
            sd = np.where(usable, sd, half)
            fallback = self._coordinate_names(~usable)
        plb, pub, clipped = self._clip(x - k * sd, x + k * sd)
        # A coordinate whose Laplace interval leaves the hard box (a ridge
        # of the posterior gives a huge curvature-based width) is clipped
        # to the box; recorded next to the curvature fallbacks.
        return (
            x.reshape(1, -1),
            plb,
            pub,
            {
                "curvature": fallback,
                "clipped": self._coordinate_names(clipped),
            },
        )

    # -- the structured export ---------------------------------------------

    def to_arviz(self, X):
        """Structured posterior draws over the model's variables.

        One ``(chain, draw, *shape)`` array per variable, in the model's
        coordinates, with the model's dimension names and coordinates
        (ArviZ 1.x: ``from_dict`` takes a dict of groups and returns a
        ``DataTree``).
        """
        import arviz as az

        original = self.to_original(self.values_from_draws(X), draws_axis=True)
        return az.from_dict(
            {"posterior": {n: a[None] for n, a in original.items()}},
            dims=self.dims or None,
            coords=self.coords or None,
        )


def posterior_dataset(data):
    """The posterior group of a DataTree (or InferenceData) as a Dataset."""
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
        "ln_Z_error": 0.0,
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


def model_bounded(rng):
    """A unit-interval and a per-coordinate interval variable, both removed."""
    import pymc as pm
    from scipy.integrate import dblquad
    from scipy.special import betaln, ndtr

    n = 12
    lower, upper = np.array([-1.0, 0.0]), np.array([2.0, 1.0])
    y = rng.normal(0.6 + 0.4 + 0.5, 1.0, size=n)
    with pm.Model() as model:
        p = pm.Beta("p", 2.0, 2.0)
        u = pm.Uniform("u", lower=lower, upper=upper, shape=2)
        pm.Normal("y", p + u[0] + u[1], 1.0, observed=y)
    log_uniform = float(-np.sum(np.log(upper - lower)))

    def log_beta(p):
        return float(np.log(p) + np.log(1 - p) - betaln(2.0, 2.0))

    def hand(point):
        p, u = float(point["p"]), np.asarray(point["u"], dtype=float)
        if not (0 < p < 1) or np.any(u <= lower) or np.any(u >= upper):
            return -np.inf
        return float(
            log_beta(p)
            + log_uniform
            + np.sum(_log_normal(y, p + u.sum(), 1.0))
        )

    # The likelihood depends on s = p + u0 + u1 only: prod N(y_i; s, 1) =
    # A exp(-n (s - ybar)^2 / 2); the integral over u0 in its interval is
    # a difference of normal CDFs, the rest is a two-dimensional
    # quadrature over p and u1.
    ybar = float(np.mean(y))
    log_A = float(-0.5 * n * np.log(2 * np.pi) - 0.5 * np.sum((y - ybar) ** 2))
    root_n = np.sqrt(n)

    def inner(u1, p):
        c = p + u1 - ybar
        return (
            np.exp(log_beta(p))
            * np.sqrt(2 * np.pi / n)
            * (ndtr(root_n * (upper[0] + c)) - ndtr(root_n * (lower[0] + c)))
        )

    value, error = dblquad(inner, 0.0, 1.0, lower[1], upper[1])
    return {
        "name": "bounded",
        "model": model,
        "hand": hand,
        "ln_Z": float(log_A + log_uniform + np.log(value)),
        "ln_Z_error": float(error / value),
        "truth": {},
        "reference": "u[0] integrated in closed form, quadrature over p, u[1]",
    }


def model_one_sided(rng):
    """Two one-sided interval variables, transforms kept.

    ``t`` is bounded below at 1 and ``v`` above at -0.5; PyMC attaches an
    interval transform with one infinite limit to each (``lower + exp``
    and ``upper - exp``), which the adapter keeps, so VBMC sees two
    unbounded coordinates and the Jacobian increment is the value variable
    itself, as for the log transform. Each variable has its own data, so
    the evidence is a product of two one-dimensional quadratures.
    """
    import pymc as pm
    from scipy.integrate import quad
    from scipy.stats import norm

    y_t = rng.normal(1.8, 1.0, size=8)
    y_v = rng.normal(-1.2, 1.0, size=8)
    with pm.Model() as model:
        t = pm.TruncatedNormal("t", 0.0, 2.0, lower=1.0)
        v = pm.TruncatedNormal("v", 0.0, 2.0, upper=-0.5)
        pm.Normal("y_t", t, 1.0, observed=y_t)
        pm.Normal("y_v", v, 1.0, observed=y_v)
    log_norm_t = float(norm.logsf(1.0, 0.0, 2.0))
    log_norm_v = float(norm.logcdf(-0.5, 0.0, 2.0))

    def log_t(t):
        if t <= 1.0:
            return -np.inf
        return float(
            _log_normal(t, 0.0, 2.0)
            - log_norm_t
            + np.sum(_log_normal(y_t, t, 1.0))
        )

    def log_v(v):
        if v >= -0.5:
            return -np.inf
        return float(
            _log_normal(v, 0.0, 2.0)
            - log_norm_v
            + np.sum(_log_normal(y_v, v, 1.0))
        )

    def hand(point):
        return log_t(float(point["t"])) + log_v(float(point["v"]))

    peak_t, peak_v = log_t(max(1.0, y_t.mean()) + 1e-3), log_v(
        min(-0.5, y_v.mean()) - 1e-3
    )
    value_t, error_t = quad(
        lambda x: np.exp(log_t(x) - peak_t), 1.0, 30.0, limit=200
    )
    value_v, error_v = quad(
        lambda x: np.exp(log_v(x) - peak_v), -30.0, -0.5, limit=200
    )
    return {
        "name": "one_sided",
        "model": model,
        "hand": hand,
        "ln_Z": float(np.log(value_t) + peak_t + np.log(value_v) + peak_v),
        "ln_Z_error": float(error_t / value_t + error_v / value_v),
        "truth": {},
        "reference": "two one-dimensional quadratures (t and v independent)",
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


def model_suppressed(rng):
    import pymc as pm

    with pm.Model() as model:
        s = pm.HalfNormal("s", 2.0, default_transform=None)
        pm.Normal("y", 0.0, s, observed=rng.normal(0.0, 1.0, size=5))
    return {"name": "suppressed", "model": model}


def model_random_bounds(rng):
    import pymc as pm

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        h = pm.Uniform("h", lower=mu, upper=mu + 1.0)
        pm.Normal("y", h, 1.0, observed=rng.normal(0.5, 1.0, size=5))
    return {"name": "random_bounds", "model": model}


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
    A point one unit outside a finite bound must give ``-inf``.
    """
    plb, pub = target.prior_box(rng)
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
    outside = None
    if np.any(np.isfinite(lb)):
        x = points[0].copy()
        i = int(np.flatnonzero(np.isfinite(lb))[0])
        x[i] = lb[i] - 1.0
        outside = target.log_joint(x)
    return {
        "points": int(len(points)),
        "max_abs_difference": float(np.max(np.abs(differences))),
        "offset": offset,
        "max_abs_deviation_from_offset": deviation,
        "outside_bounds_value": None if outside is None else float(outside),
        "passes": bool(
            deviation <= TOL_LOGP
            and abs(offset) <= TOL_OFFSET
            and (outside is None or outside == -np.inf)
        ),
    }


def check_jacobian(spec, target, rng):
    """The Jacobian increment of the kept transforms against the value.

    ``log_joint - log_density_plain`` is what ``compile_logp(jacobian=True)``
    adds for the kept transforms. For a log transform (``exp``) and for a
    one-sided interval transform (``lower + exp`` or ``upper - exp``) of a
    scalar the log-Jacobian is the value variable itself, so the expected
    increment is the sum of the kept value variables. The constant
    offsets of the folded scales cancel in the increment.
    """
    if not target.kept:
        return {"applicable": False}
    kinds = {n: k["transform"] for n, k in target.kept.items()}
    if any(
        kind not in ("LogTransform", "Interval", "IntervalTransform")
        for kind in kinds.values()
    ):
        return {"applicable": True, "kept": kinds, "checked": False}
    plb, pub = target.prior_box(rng)
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


def time_density(target, calls=N_TIMING_CALLS):
    """Milliseconds per call of the compiled log joint at the start point."""
    x = target.x0.ravel()
    started = time.perf_counter()
    for _ in range(calls):
        target.log_joint(x)
    return 1000.0 * (time.perf_counter() - started) / calls


def choose_box(target, rng, plausible):
    if plausible == "prior":
        plb, pub = target.prior_box(rng)
        return target.x0, plb, pub, {"curvature": [], "clipped": []}
    return target.laplace_box(rng)


def fit(spec, target, rng, seed, plausible):
    """One VBMC run from the chosen starting point and plausible box.

    Raises :class:`FitFailure` carrying the box and the log joint at its
    corners and centre when VBMC fails.
    """
    from pyvbmc import VBMC

    started = time.perf_counter()
    x0, plb, pub, fallback = choose_box(target, rng, plausible)
    setup_seconds = time.perf_counter() - started
    corners = np.array(
        [plb.ravel(), pub.ravel(), 0.5 * (plb + pub).ravel(), x0.ravel()]
    )
    context = {
        "plausible": plausible,
        "laplace_fallback": fallback,
        "setup_seconds": setup_seconds,
        "plb": plb.ravel().tolist(),
        "pub": pub.ravel().tolist(),
        "x0": np.ravel(x0).tolist(),
        "log_joint_at_plb_pub_centre_x0": [
            target.log_joint(c) for c in corners
        ],
    }
    started = time.perf_counter()
    try:
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
    except Exception as error:  # noqa: BLE001
        raise FitFailure(context, error) from error
    seconds = time.perf_counter() - started
    out = dict(context)
    out.update(
        {
            "seconds": seconds,
            "func_count": int(results["func_count"]),
            "iterations": int(results["iterations"]),
            "success_flag": bool(results["success_flag"]),
            "elbo": float(results["elbo"]),
            "elbo_sd": float(results["elbo_sd"]),
            "ln_Z": spec["ln_Z"],
            "ln_Z_error": spec["ln_Z_error"],
            "elbo_minus_ln_Z": float(results["elbo"] - spec["ln_Z"]),
            "K": int(vp.K),
        }
    )
    draws = vp.sample(N_DRAWS)[0]
    original = target.to_original(
        target.values_from_draws(draws), draws_axis=True
    )
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


def _git(path, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    import arviz
    import gpyreg
    import pymc
    import pytensor
    import scipy

    import pyvbmc

    gp_dir = Path(gpyreg.__file__).resolve().parent
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
        "pyvbmc_commit": _git(ROOT, "rev-parse", "HEAD"),
        "pyvbmc_dirty": _git(
            ROOT, "status", "--porcelain", "--", "pyvbmc", "dev/scripts"
        ),
        "gpyreg": str(gp_dir),
        "gpyreg_commit": _git(gp_dir.parent, "rev-parse", "HEAD"),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
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
        f"{env['arviz']}, Python {env['python']}; PyVBMC commit "
        f"{(env['pyvbmc_commit'] or '?')[:7]}, gpyreg commit "
        f"{(env['gpyreg_commit'] or '?')[:7]}, script "
        f"{env['script_sha256'][:12]}. Starting point and plausible box: "
        f"`{report['plausible']}` "
        + (
            "(the mode from `find_MAP` and its Laplace box)"
            if report["plausible"] == "laplace"
            else "(the model's initial point and prior quantiles)"
        )
        + ".",
        "",
        "| Model | D | VBMC coordinates and bounds | density: offset; max "
        "abs deviation | Jacobian increment | ms per call | ELBO − ln Z "
        "(± sd) | evaluations | setup + fit seconds | return path |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---|",
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
                    delta = rp["deterministic"][
                        "max_abs_difference_from_X_beta"
                    ]
                    rp_text += f" (max abs difference from X beta {delta:.1e})"
            rp_text += f"; predictive {rp['posterior_predictive']['shape']}"
        else:
            rp_text = "not run"
        lines.append(
            f"| {m['name']} | {m.get('D', '-')} | "
            f"{m.get('coordinates', '-')} | "
            f"{density.get('offset', float('nan')):+.1e}; "
            f"{density.get('max_abs_deviation_from_offset', float('nan')):.1e}"
            + (" ok" if density.get("passes") else " FAIL")
            + f" | {jac_text} | "
            + (f"{m['ms_per_call']:.2f}" if "ms_per_call" in m else "-")
            + " | "
            + (
                f"{fitted['elbo_minus_ln_Z']:+.4f} "
                f"(± {fitted['elbo_sd']:.4f})"
                if fitted
                else "not run"
            )
            + " | "
            + (str(fitted["func_count"]) if fitted else "-")
            + " | "
            + (
                f"{fitted['setup_seconds']:.1f} + {fitted['seconds']:.1f}"
                if fitted
                else "-"
            )
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
            context = e.get("context")
            if context:
                lines.append(
                    f"  - box: plb {context['plb']}, pub {context['pub']}, "
                    f"x0 {context['x0']}; log joint at plb, pub, centre, "
                    f"x0: {context['log_joint_at_plb_pub_centre_x0']}"
                )
    return "\n".join(lines) + "\n"


def describe_coordinates(target):
    parts = []
    for i, (n, v) in enumerate(zip(target.names, target.value_names)):
        lo = target.lb.ravel()[target.offsets[i] : target.offsets[i + 1]]
        hi = target.ub.ravel()[target.offsets[i] : target.offsets[i + 1]]
        if n in target.kept:
            parts.append(
                f"{v} ({target.kept[n]['transform']} kept, unbounded)"
            )
        else:
            parts.append(
                f"{n} " + " ".join(f"[{a:g}, {b:g}]" for a, b in zip(lo, hi))
            )
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
    # Every model's data come from a generator of their own, keyed by the
    # model's position, so that the models are the same whichever route
    # is run and however many draws the checks and boxes consume.
    rng = np.random.default_rng([args.seed, 1])
    for index, build in enumerate(
        (
            model_scalar,
            model_positive,
            model_vector,
            model_bounded,
            model_one_sided,
        )
    ):
        spec = build(np.random.default_rng([args.seed, 0, index]))
        entry = {
            "name": spec["name"],
            "reference": spec["reference"],
            "ln_Z": spec["ln_Z"],
            "ln_Z_error": spec["ln_Z_error"],
        }
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
                        n: [lo.tolist(), hi.tolist()]
                        for n, (lo, hi) in target.support.items()
                    },
                    "coordinates": describe_coordinates(target),
                    "dims": target.dims,
                    "coords": target.coords,
                    "density": check_density(spec, target, rng),
                    "jacobian": check_jacobian(spec, target, rng),
                    "ms_per_call": time_density(target),
                }
            )
            print(
                f"[{spec['name']}] D={target.D}; {entry['coordinates']}; "
                f"density offset {entry['density']['offset']:+.1e}, max "
                f"abs deviation "
                f"{entry['density']['max_abs_deviation_from_offset']:.1e}, "
                f"outside {entry['density']['outside_bounds_value']}"
                + (
                    "; Jacobian increment max abs difference "
                    f"{entry['jacobian']['max_abs_difference']:.1e}"
                    if entry["jacobian"].get("checked")
                    else ""
                )
                + f"; {entry['ms_per_call']:.2f} ms per call",
                flush=True,
            )
            if not args.no_fit:
                try:
                    fitted, vp, draws = fit(
                        spec, target, rng, args.seed, args.plausible
                    )
                except FitFailure as failure:
                    report["errors"].append(
                        {
                            "where": spec["name"],
                            "error": str(failure),
                            "context": failure.context,
                            "traceback": "".join(
                                traceback.format_exception(failure.error)
                            ),
                        }
                    )
                    print(
                        f"[{spec['name']}] FIT FAILED {failure}; box "
                        f"{failure.context['plb']} .. "
                        f"{failure.context['pub']}",
                        flush=True,
                    )
                else:
                    entry["fit"] = fitted
                    print(
                        f"[{spec['name']}] VBMC: elbo {fitted['elbo']:.4f} +- "
                        f"{fitted['elbo_sd']:.4f}, ln Z {spec['ln_Z']:.4f} "
                        f"(quadrature error {spec['ln_Z_error']:.1e}), "
                        f"{fitted['func_count']} evaluations, "
                        f"{fitted['setup_seconds']:.1f} + "
                        f"{fitted['seconds']:.1f} s"
                        + (
                            f", Laplace fallback {fitted['laplace_fallback']}"
                            if any(fitted["laplace_fallback"].values())
                            else ""
                        ),
                        flush=True,
                    )
                    entry["return_path"] = check_return_path(
                        spec, target, draws
                    )
                    print(
                        f"[{spec['name']}] return path: "
                        f"{entry['return_path']}",
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
    for index, build in enumerate(
        (
            model_discrete,
            model_simplex,
            model_suppressed,
            model_random_bounds,
        )
    ):
        spec = build(np.random.default_rng([args.seed, 2, index]))
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
