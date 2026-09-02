"""Benchmark target suite for PyVBMC developer tooling.

One place that defines every benchmark target (log density, ground truth,
plausible box, VBMC options) and the named suites built from them, so that
``profile_run.py`` / ``profile_suite.py`` (profiling) and ``golden_trace.py``
(regression population) never drift apart. Plan and rationale:
``dev/plans/benchmark-suite-and-golden-traces.md``; the suite was asked for in
``dev/2026-09-02-modernization-discussion.md`` section 10.

Targets (``make_problem(name, D)``), hard bounds infinite unless stated:

``normal``      independent Gaussian, SDs 1..D (the historical profile target)
``corr``        rotated Gaussian, SDs linspace(0.2, 1)
``halfnormal``  Gaussian restricted to the negative orthant (bounded, probit
                path); ln Z = -D ln 2
``rosenbrock``  the test/notebook Rosenbrock with a N(0, 3^2) prior, D = 2;
                truth by 1-D quadrature (the x2 integral is analytic)
``banana``      volume-preserving transform of a Gaussian, D >= 2: exact
                truth at any D, curvature comparable to the notebook Rosenbrock
``cigar``       one axis 100x longer than the others, seeded orthogonal mixing
``lumpy``       12-component Gaussian mixture (Acerbi 2018, section 4.1)
``student``     product of Student-t likelihoods, nu in [2.5, 2 + D/2], times
                the paper's broad normal prior; truth by 1-D quadratures
``logreg``      Bayesian logistic regression on fixed synthetic data, D = 5;
                truth by Laplace + importance sampling, stored as constants

Any target can be made noisy with ``noise_sd``: the callable then returns
``(y + sd * eps, sd)`` and ``options["specify_target_noise"] = True``.

Plausible boxes: legacy targets keep the boxes of the tests / notebooks; the
new targets use the per-coordinate 0.5 % and 99.5 % quantiles of 10^6 exact
draws, rounded to one decimal (about +-2.6 SD for Gaussian marginals, and
following skew where there is some).

Command line::

    python dev/scripts/benchmark_targets.py --list
    python dev/scripts/benchmark_targets.py --check [--suite all]
    python dev/scripts/benchmark_targets.py --smoke [--suite smoke]

``--check`` verifies each implementation against an independent reference
density and against the moments of exact samples, and integrates numerically
only where ln Z is not analytic by construction. ``--smoke`` runs every config
of a suite through two VBMC iterations.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy import stats
from scipy.special import expit, log_expit, logsumexp

REPO_ROOT = Path(__file__).resolve().parents[2]

# Seeds that freeze the *structure* of randomly generated targets. They are
# independent of the run seed on purpose: every run of every version sees
# the same target.
STRUCTURE_SEED = 20260900
LOGREG_SEED = 20260905
BOX_SEED = 20260901
BOX_DRAWS = 1_000_000
BOX_QUANTILES = (0.005, 0.995)


# --------------------------------------------------------------------------
# Problem and Config
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Problem:
    """A benchmark target with its ground truth and VBMC setup.

    ``log_density_vec`` maps an ``(n, D)`` array to ``(n,)`` log densities;
    ``fun`` is the scalar callable VBMC receives (a thin wrapper, adding
    noise when ``noise_sd`` is set). ``sampler(n, rng)`` draws exact samples
    when the generative process is known; ``reference_logpdf`` is an
    independent implementation used by ``--check``. Truth entries are
    ``None`` when unknown. Truth is never passed to VBMC.
    """

    name: str
    D: int
    log_density_vec: Callable[[np.ndarray], np.ndarray]
    x0: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    plb: np.ndarray
    pub: np.ndarray
    ln_Z: Optional[float] = None
    true_mean: Optional[np.ndarray] = None
    true_cov: Optional[np.ndarray] = None
    sampler: Optional[Callable[[int, np.random.Generator], np.ndarray]] = None
    reference_logpdf: Optional[Callable[[np.ndarray], np.ndarray]] = None
    options: dict = dataclasses.field(default_factory=dict)
    noise_sd: Optional[float] = None
    notes: str = ""
    _noise_rng: Optional[np.random.Generator] = dataclasses.field(
        default=None, repr=False
    )

    def fun(self, x):
        """Scalar log density (or ``(y, sd)`` when noisy) at one point."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        y = float(self.log_density_vec(x)[0])
        if self.noise_sd is None:
            return y
        eps = float(self._noise_rng.standard_normal())
        return y + self.noise_sd * eps, self.noise_sd

    def vbmc_args(self):
        """``(positional_args, options)`` for ``VBMC(*args, options=...)``."""
        args = (self.fun, self.x0, self.lb, self.ub, self.plb, self.pub)
        return args, dict(self.options)

    @property
    def all_unbounded(self):
        return bool(np.all(np.isinf(self.lb)) and np.all(np.isinf(self.ub)))

    @property
    def noisy(self):
        return self.noise_sd is not None


@dataclasses.dataclass(frozen=True)
class Config:
    """One entry of a suite: a target at a dimension, plus run options."""

    name: str
    D: int
    noise_sd: Optional[float] = None
    options: tuple = ()  # tuple of (key, value) pairs so the Config hashes
    tag: str = ""

    @property
    def label(self):
        s = f"{self.name}_D{self.D}"
        if self.noise_sd is not None:
            s += f"_noise{self.noise_sd:g}"
        if self.tag:
            s += f"_{self.tag}"
        return s

    def options_dict(self):
        return dict(self.options)

    def make(self, seed=None):
        return make_problem(
            self.name,
            self.D,
            noise_sd=self.noise_sd,
            seed=seed,
            options=self.options_dict(),
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _row(v, D):
    return (
        np.full((1, D), float(v)) if np.ndim(v) == 0 else np.reshape(v, (1, D))
    )


def _inf_bounds(D):
    return np.full((1, D), -np.inf), np.full((1, D), np.inf)


def _quantile_box(sampler, D, n=BOX_DRAWS, seed=BOX_SEED):
    """Plausible box from the 0.5 % / 99.5 % quantiles of exact draws."""
    rng = np.random.default_rng(seed)
    X = sampler(n, rng)
    lo, hi = np.quantile(X, BOX_QUANTILES, axis=0)
    lo = np.floor(lo * 10) / 10
    hi = np.ceil(hi * 10) / 10
    return lo.reshape(1, D), hi.reshape(1, D)


def _mvn_logpdf_chol(X, mean, L):
    """Log density of N(mean, L L^T) at the rows of X."""
    diff = X - mean
    z = np.linalg.solve(L, diff.T)  # (D, n)
    D = len(mean)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * np.sum(z**2, axis=0) - 0.5 * (
        logdet + D * np.log(2 * np.pi)
    )


class _Grid1D:
    """A 1-D density tabulated on a fine grid: normalization, moments,
    inverse-CDF sampling. Used for the quadrature-truth targets."""

    def __init__(self, log_f, lo, hi, n=40001):
        self.x = np.linspace(lo, hi, n)
        lf = log_f(self.x)
        self.lmax = lf.max()
        f = np.exp(lf - self.lmax)
        self.dx = self.x[1] - self.x[0]
        Z = np.trapezoid(f, self.x)
        self.ln_Z = self.lmax + np.log(Z)
        p = f / Z
        self.p = p
        self.mean = np.trapezoid(self.x * p, self.x)
        self.var = np.trapezoid((self.x - self.mean) ** 2 * p, self.x)
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * self.dx)]
        )
        self.cdf = cdf / cdf[-1]

    def expect(self, g):
        return np.trapezoid(g(self.x) * self.p, self.x)

    def sample(self, n, rng):
        u = rng.random(n)
        return np.interp(u, self.cdf, self.x)


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------


def _normal(D):
    scales = np.arange(1, D + 1, dtype=float)
    log_norm = -np.sum(np.log(scales)) - 0.5 * D * np.log(2 * np.pi)

    def logp(X):
        X = np.atleast_2d(X)
        return np.sum(-0.5 * (X / scales) ** 2, axis=1) + log_norm

    def ref(X):
        return np.sum(stats.norm.logpdf(np.atleast_2d(X), 0.0, scales), axis=1)

    def sampler(n, rng):
        return rng.standard_normal((n, D)) * scales

    lb, ub = _inf_bounds(D)
    return Problem(
        name="normal",
        D=D,
        log_density_vec=logp,
        x0=-np.ones((1, D)),
        lb=lb,
        ub=ub,
        plb=np.full((1, D), -2.0 * D),
        pub=np.full((1, D), 2.0 * D),
        ln_Z=0.0,
        true_mean=np.zeros((1, D)),
        true_cov=np.diag(scales**2),
        sampler=sampler,
        reference_logpdf=ref,
        notes="independent Gaussian, SDs 1..D; legacy box -+2D",
    )


def _corr(D):
    rng = np.random.default_rng(12345)  # as in the original profile_run.py
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    scales = np.linspace(0.2, 1.0, D)
    cov = Q @ np.diag(scales**2) @ Q.T
    mean = np.linspace(-0.5, 0.5, D)
    L = np.linalg.cholesky(cov)
    mvn = stats.multivariate_normal(mean, cov)

    def logp(X):
        return _mvn_logpdf_chol(np.atleast_2d(X), mean, L)

    def sampler(n, rng):
        return mean + rng.standard_normal((n, D)) @ L.T

    lb, ub = _inf_bounds(D)
    return Problem(
        name="corr",
        D=D,
        log_density_vec=logp,
        x0=np.zeros((1, D)),
        lb=lb,
        ub=ub,
        plb=np.full((1, D), -2.5),
        pub=np.full((1, D), 2.5),
        ln_Z=0.0,
        true_mean=mean.reshape(1, D),
        true_cov=cov,
        sampler=sampler,
        reference_logpdf=lambda X: mvn.logpdf(np.atleast_2d(X)),
        notes="rotated Gaussian, SDs linspace(0.2, 1); legacy box -+2.5",
    )


def _halfnormal(D):
    scales = np.arange(1, D + 1, dtype=float)
    log_norm = -np.sum(np.log(scales)) - 0.5 * D * np.log(2 * np.pi)

    def logp(X):
        X = np.atleast_2d(X)
        return np.sum(-0.5 * (X / scales) ** 2, axis=1) + log_norm

    def ref(X):
        return np.sum(stats.norm.logpdf(np.atleast_2d(X), 0.0, scales), axis=1)

    def sampler(n, rng):
        return -np.abs(rng.standard_normal((n, D)) * scales)

    return Problem(
        name="halfnormal",
        D=D,
        log_density_vec=logp,
        x0=-np.ones((1, D)),
        lb=np.full((1, D), -10.0 * D),
        ub=np.zeros((1, D)),
        plb=np.full((1, D), -6.0),
        pub=np.full((1, D), -0.05),
        ln_Z=-D * np.log(2.0),
        true_mean=(-scales * np.sqrt(2 / np.pi)).reshape(1, D),
        true_cov=np.diag(scales**2 * (1 - 2 / np.pi)),
        sampler=sampler,
        reference_logpdf=ref,
        notes=(
            "Gaussian (SDs 1..D) on the negative orthant, lb=-10D, ub=0;"
            " ln Z exact up to Phi(-10); legacy box [-6, -0.05]"
        ),
    )


_ROSENBROCK_CACHE = {}


def _rosenbrock(D):
    if D != 2:
        raise ValueError("rosenbrock is defined for D = 2 only")
    prior_sd = 3.0

    def logp(X):
        X = np.atleast_2d(X)
        ll = -np.sum(
            (X[:, :-1] ** 2 - X[:, 1:]) ** 2 + (X[:, :-1] - 1) ** 2 / 100,
            axis=1,
        )
        return ll + np.sum(stats.norm.logpdf(X, 0.0, prior_sd), axis=1)

    # Truth. For fixed x1 the x2 factor is exp(-a x2^2 + b x2 + c) with
    # a = 1 + 1/(2 s^2), b = 2 x1^2, c = -x1^4, so the x2 integral is analytic
    # and a single 1-D quadrature in x1 remains (lazy, cached per process).
    if "grid" not in _ROSENBROCK_CACHE:
        a = 1.0 + 1.0 / (2 * prior_sd**2)
        const = -np.log(2 * np.pi * prior_sd**2)

        def log_g(x1):
            b = 2 * x1**2
            c = -(x1**4)
            return (
                const
                - (x1 - 1) ** 2 / 100
                - x1**2 / (2 * prior_sd**2)
                + 0.5 * np.log(np.pi / a)
                + b**2 / (4 * a)
                + c
            )

        g = _Grid1D(log_g, -8.0, 8.0, n=80001)
        m1 = g.mean
        e_x2_given = lambda x1: (2 * x1**2) / (2 * a)  # b / (2a)
        var_x2_given = 1.0 / (2 * a)
        m2 = g.expect(e_x2_given)
        v11 = g.var
        c12 = g.expect(lambda x1: x1 * e_x2_given(x1)) - m1 * m2
        v22 = g.expect(lambda x1: var_x2_given + e_x2_given(x1) ** 2) - m2**2
        _ROSENBROCK_CACHE.update(
            grid=g,
            ln_Z=float(g.ln_Z),
            mean=np.array([[m1, m2]]),
            cov=np.array([[v11, c12], [c12, v22]]),
            a=a,
        )
    cache = _ROSENBROCK_CACHE

    def sampler(n, rng):
        x1 = cache["grid"].sample(n, rng)
        mu2 = (2 * x1**2) / (2 * cache["a"])
        x2 = mu2 + rng.standard_normal(n) / np.sqrt(2 * cache["a"])
        return np.c_[x1, x2]

    lb, ub = _inf_bounds(D)
    return Problem(
        name="rosenbrock",
        D=D,
        log_density_vec=logp,
        x0=np.zeros((1, D)),
        lb=lb,
        ub=ub,
        plb=np.full((1, D), -3.0),
        pub=np.full((1, D), 3.0),
        ln_Z=cache["ln_Z"],
        true_mean=cache["mean"],
        true_cov=cache["cov"],
        sampler=sampler,
        reference_logpdf=None,
        notes=(
            "test/notebook Rosenbrock + N(0, 3^2) prior; truth by 1-D"
            " quadrature; notebook-1 box -+3"
        ),
    )


def _banana(D, sig1=2.0, b=0.5):
    if D < 2:
        raise ValueError("banana needs D >= 2")
    sig = np.ones(D)
    sig[0] = sig1

    def to_z(X):
        Z = np.array(X, dtype=float, copy=True)
        Z[:, 1] = X[:, 1] - b * (X[:, 0] ** 2 - sig1**2)
        return Z

    log_norm = -np.sum(np.log(sig)) - 0.5 * D * np.log(2 * np.pi)

    def logp(X):
        Z = to_z(np.atleast_2d(X))
        return np.sum(-0.5 * (Z / sig) ** 2, axis=1) + log_norm

    mvn = stats.multivariate_normal(np.zeros(D), np.diag(sig**2))

    def ref(X):
        return mvn.logpdf(to_z(np.atleast_2d(X)))

    def sampler(n, rng):
        Z = rng.standard_normal((n, D)) * sig
        X = Z.copy()
        X[:, 1] = Z[:, 1] + b * (Z[:, 0] ** 2 - sig1**2)
        return X

    cov = np.diag(sig**2)
    cov[1, 1] = sig[1] ** 2 + 2 * b**2 * sig1**4
    plb, pub = _quantile_box(sampler, D)
    lb, ub = _inf_bounds(D)
    return Problem(
        name="banana",
        D=D,
        log_density_vec=logp,
        x0=np.zeros((1, D)),
        lb=lb,
        ub=ub,
        plb=plb,
        pub=pub,
        ln_Z=0.0,
        true_mean=np.zeros((1, D)),
        true_cov=cov,
        sampler=sampler,
        reference_logpdf=ref,
        notes=(
            f"z ~ N(0, diag(sig^2)), sig1={sig1}, x2 = z2 + {b}(z1^2 - sig1^2);"
            " unit Jacobian; true cov is diagonal so only elbo_err sees the"
            " ridge"
        ),
    )


def _cigar(D):
    rng = np.random.default_rng(STRUCTURE_SEED + D)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    ell = np.full(D, 0.01)
    ell[-1] = 1.0
    cov = Q @ np.diag(ell**2) @ Q.T  # the one expression used everywhere
    mean = np.linspace(-0.5, 0.5, D)
    L = np.linalg.cholesky(cov)
    mvn = stats.multivariate_normal(mean, cov)

    def logp(X):
        return _mvn_logpdf_chol(np.atleast_2d(X), mean, L)

    def sampler(n, rng):
        return mean + (rng.standard_normal((n, D)) * ell) @ Q.T

    plb, pub = _quantile_box(sampler, D)
    lb, ub = _inf_bounds(D)
    return Problem(
        name="cigar",
        D=D,
        log_density_vec=logp,
        x0=np.full((1, D), 0.5),
        lb=lb,
        ub=ub,
        plb=plb,
        pub=pub,
        ln_Z=0.0,
        true_mean=mean.reshape(1, D),
        true_cov=cov,
        sampler=sampler,
        reference_logpdf=lambda X: mvn.logpdf(np.atleast_2d(X)),
        notes=(
            "SDs 0.01 on D-1 axes and 1 on one, mixed by a seeded orthogonal"
            f" matrix (det {np.linalg.det(Q):+.0f}); cov = Q diag(ell^2) Q^T"
        ),
    )


def _lumpy(D, n_comp=12):
    rng = np.random.default_rng(STRUCTURE_SEED + D)
    mus = rng.random((n_comp, D))
    sds = rng.uniform(0.2, 0.6, size=(n_comp, D))
    w = rng.dirichlet(np.ones(n_comp))
    log_w = np.log(w)
    comp_log_norm = -np.sum(np.log(sds), axis=1) - 0.5 * D * np.log(2 * np.pi)

    def logp(X):
        X = np.atleast_2d(X)
        z = (X[:, None, :] - mus[None, :, :]) / sds[None, :, :]
        lc = -0.5 * np.sum(z**2, axis=2) + comp_log_norm + log_w  # (n, K)
        return logsumexp(lc, axis=1)

    comps = [
        stats.multivariate_normal(mus[k], np.diag(sds[k] ** 2))
        for k in range(n_comp)
    ]

    def ref(X):
        X = np.atleast_2d(X)
        lc = np.stack([c.logpdf(X) + log_w[k] for k, c in enumerate(comps)])
        return logsumexp(np.atleast_2d(lc), axis=0)

    def sampler(n, rng):
        k = rng.choice(n_comp, size=n, p=w)
        return mus[k] + rng.standard_normal((n, D)) * sds[k]

    mean = w @ mus
    cov = np.zeros((D, D))
    for k in range(n_comp):
        cov += w[k] * (np.diag(sds[k] ** 2) + np.outer(mus[k], mus[k]))
    cov -= np.outer(mean, mean)
    plb, pub = _quantile_box(sampler, D)
    lb, ub = _inf_bounds(D)
    return Problem(
        name="lumpy",
        D=D,
        log_density_vec=logp,
        x0=mean.reshape(1, D).copy(),
        lb=lb,
        ub=ub,
        plb=plb,
        pub=pub,
        ln_Z=0.0,
        true_mean=mean.reshape(1, D),
        true_cov=cov,
        sampler=sampler,
        reference_logpdf=ref,
        notes=(
            f"{n_comp} Gaussians, means U[0,1]^D, SDs U[0.2,0.6], Dirichlet(1)"
            f" weights (top weight {w.max():.2f}); Acerbi 2018 section 4.1"
        ),
    )


_STUDENT_CACHE = {}


def _student(D):
    nu = np.linspace(2.5, 2.0 + D / 2.0, D)
    sd_t = np.sqrt(nu / (nu - 2.0))
    prior_sd = 3.0 * sd_t  # the paper's broad normal prior

    def logp(X):
        X = np.atleast_2d(X)
        return np.sum(
            stats.t.logpdf(X, nu) + stats.norm.logpdf(X, 0.0, prior_sd), axis=1
        )

    if D not in _STUDENT_CACHE:
        grids = []
        for d in range(D):
            lo_hi = 12.0 * prior_sd[d]

            def log_f(x, d=d):
                return stats.t.logpdf(x, nu[d]) + stats.norm.logpdf(
                    x, 0.0, prior_sd[d]
                )

            grids.append(_Grid1D(log_f, -lo_hi, lo_hi, n=80001))
        _STUDENT_CACHE[D] = grids
    grids = _STUDENT_CACHE[D]
    ln_Z = float(sum(g.ln_Z for g in grids))
    var = np.array([g.var for g in grids])

    def sampler(n, rng):
        return np.column_stack([g.sample(n, rng) for g in grids])

    plb, pub = _quantile_box(sampler, D)
    lb, ub = _inf_bounds(D)
    return Problem(
        name="student",
        D=D,
        log_density_vec=logp,
        x0=np.zeros((1, D)),
        lb=lb,
        ub=ub,
        plb=plb,
        pub=pub,
        ln_Z=ln_Z,
        true_mean=np.zeros((1, D)),
        true_cov=np.diag(var),
        sampler=sampler,
        reference_logpdf=None,  # logp is already the reference composition
        notes=(
            f"product of t(nu), nu = {np.round(nu, 2).tolist()}, unit scale,"
            " times N(0, (3 sd_t)^2) priors; truth by per-coordinate"
            " quadrature"
        ),
    )


# Logistic regression: design and data are frozen by LOGREG_SEED; the truth
# constants below were produced by ``--check`` (Laplace + importance
# sampling, t(4) proposal with 2x the Laplace covariance, 2e6 draws). Rerun
# ``python dev/scripts/benchmark_targets.py --check --only logreg`` to
# regenerate them; the standard error and ESS are printed alongside.
LOGREG_N = 50
LOGREG_RHO = 0.95
LOGREG_N_RARE = 6
LOGREG_PRIOR_SD = 5.0
LOGREG_W_TRUE = np.array([0.3, 1.0, -1.0, 0.8, 1.5])
LOGREG_TRUTH = {
    # --check on 2026-09-02: ln_Z = -33.34232 +- 0.00081 (IS standard error),
    # ESS = 868302 of 2e6 draws; t(4) proposal, 2x Laplace covariance.
    "ln_Z": -33.34232,
    "mean": [0.94379, -1.46096, 1.50319, 1.49473, 4.58708],
    "cov": [
        [0.15855, -0.06968, 0.07649, 0.07091, -0.09458],
        [-0.06968, 1.84493, -1.54365, -0.10568, 0.05864],
        [0.07649, -1.54365, 1.45224, 0.14413, -0.08335],
        [0.07091, -0.10568, 0.14413, 0.30443, -0.02085],
        [-0.09458, 0.05864, -0.08335, -0.02085, 9.12125],
    ],
}


def _logreg_data():
    rng = np.random.default_rng(LOGREG_SEED)
    n = LOGREG_N
    z = rng.standard_normal((n, 2))
    x1 = z[:, 0]
    x2 = LOGREG_RHO * z[:, 0] + np.sqrt(1 - LOGREG_RHO**2) * z[:, 1]
    x3 = rng.standard_normal(n)
    x4 = np.zeros(n)
    x4[rng.choice(n, LOGREG_N_RARE, replace=False)] = 1.0
    X = np.c_[np.ones(n), x1, x2, x3, x4]
    p = expit(X @ LOGREG_W_TRUE)
    y = (rng.random(n) < p).astype(float)
    y[x4 == 1] = 1.0  # the rare predictor's coefficient is prior-identified
    return X, y


def _logreg(D):
    if D != 5:
        raise ValueError("logreg is defined for D = 5 only")
    Xd, y = _logreg_data()

    def logp(W):
        W = np.atleast_2d(W)
        eta = W @ Xd.T
        ll = np.sum(y * log_expit(eta) + (1 - y) * log_expit(-eta), axis=1)
        lp = np.sum(stats.norm.logpdf(W, 0.0, LOGREG_PRIOR_SD), axis=1)
        return ll + lp

    truth = LOGREG_TRUTH
    lb, ub = _inf_bounds(D)
    return Problem(
        name="logreg",
        D=D,
        log_density_vec=logp,
        x0=np.zeros((1, D)),
        lb=lb,
        ub=ub,
        plb=np.full((1, D), -LOGREG_PRIOR_SD),
        pub=np.full((1, D), LOGREG_PRIOR_SD),
        ln_Z=truth["ln_Z"],
        true_mean=None if truth["mean"] is None else np.array([truth["mean"]]),
        true_cov=None if truth["cov"] is None else np.array(truth["cov"]),
        sampler=None,
        reference_logpdf=None,
        notes=(
            f"logistic regression, {LOGREG_N} trials, predictors 1-2 at rho"
            f" {LOGREG_RHO}, predictor 4 rare ({LOGREG_N_RARE} ones, all"
            f" successes), N(0, {LOGREG_PRIOR_SD}^2) prior; box -+1 prior SD"
        ),
    )


def logreg_reference(m=2_000_000, chunk=20_000, df=4, scale=2.0, seed=1):
    """Laplace + importance-sampling truth for ``logreg`` (ln Z, mean, cov)."""
    from scipy import optimize

    Xd, y = _logreg_data()
    D = Xd.shape[1]
    prob = _logreg(D)

    def f(w):
        return -prob.log_density_vec(w)[0]

    def g(w):
        p = expit(Xd @ w)
        return -(Xd.T @ (y - p) - w / LOGREG_PRIOR_SD**2)

    res = optimize.minimize(f, np.zeros(D), jac=g, method="BFGS")
    p = expit(Xd @ res.x)
    H = Xd.T @ (Xd * (p * (1 - p))[:, None]) + np.eye(D) / LOGREG_PRIOR_SD**2
    lap_cov = np.linalg.inv(H)
    prop = stats.multivariate_t(
        loc=res.x, shape=lap_cov * scale, df=df, seed=seed
    )
    Ws, lws = [], []
    for _ in range(m // chunk):
        W = prop.rvs(chunk)
        lws.append(prob.log_density_vec(W) - prop.logpdf(W))
        Ws.append(W)
    W = np.vstack(Ws)
    lw = np.concatenate(lws)
    lmax = lw.max()
    w = np.exp(lw - lmax)
    ln_Z = lmax + np.log(w.mean())
    se_ln_Z = np.std(w) / w.mean() / np.sqrt(len(w))
    ess = w.sum() ** 2 / np.sum(w**2)
    wn = w / w.sum()
    mean = wn @ W
    cov = (W - mean).T @ ((W - mean) * wn[:, None])
    return dict(
        ln_Z=float(ln_Z),
        se_ln_Z=float(se_ln_Z),
        ess=float(ess),
        mean=mean,
        cov=cov,
        laplace_mode=res.x,
        laplace_cov=lap_cov,
        n_draws=len(w),
    )


_REGISTRY = {
    "normal": _normal,
    "corr": _corr,
    "halfnormal": _halfnormal,
    "rosenbrock": _rosenbrock,
    "banana": _banana,
    "cigar": _cigar,
    "lumpy": _lumpy,
    "student": _student,
    "logreg": _logreg,
}

TARGET_NAMES = tuple(_REGISTRY)


def make_problem(name, D, noise_sd=None, seed=None, options=None):
    """Build a benchmark ``Problem``.

    ``noise_sd`` makes the target noisy (homoskedastic Gaussian noise on the
    log density, returned as the second output; ``specify_target_noise`` is
    set). ``seed`` seeds only that noise stream, through a spawned
    ``SeedSequence`` so it is not the same stream as ``VBMC(seed=seed)``;
    ``None`` means fresh entropy. ``options`` are merged into the problem's
    VBMC options (caller wins).
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown target {name!r}; known: {TARGET_NAMES}")
    prob = _REGISTRY[name](int(D))
    if noise_sd is not None:
        if not noise_sd > 0:
            raise ValueError("noise_sd must be positive")
        prob.noise_sd = float(noise_sd)
        if seed is None:
            prob._noise_rng = np.random.default_rng()
        else:
            prob._noise_rng = np.random.default_rng(
                np.random.SeedSequence(seed).spawn(1)[0]
            )
        prob.options["specify_target_noise"] = True
    if options:
        prob.options.update(options)
    return prob


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------

SUITES = {
    "smoke": [
        Config("normal", 2),
        Config("normal", 5),
        Config("corr", 5),
    ],
    "profile": [
        Config("banana", 4),
        Config("cigar", 4),
        Config("lumpy", 4),
        Config("student", 4),
        Config("logreg", 5),
        Config("banana", 2, noise_sd=1.0),
        Config(
            "normal",
            5,
            options=(("tol_stable_excpt_frac", -(10**6)),),
            tag="exhaust",
        ),
    ],
    "golden": [
        Config("normal", 5),
        Config("corr", 5),
        Config("halfnormal", 2),
        Config("rosenbrock", 2),
        Config("banana", 2),
        Config("banana", 6),
        Config("cigar", 4),
        Config("lumpy", 4),
        Config("student", 4),
        Config("logreg", 5),
        Config(
            "banana",
            2,
            noise_sd=1.0,
            options=(("max_fun_evals", 150),),
            tag="mfe150",
        ),
    ],
}


def suite_configs(suite):
    if suite == "all":
        seen = {}
        for s in SUITES.values():
            for c in s:
                seen.setdefault(c.label, c)
        return list(seen.values())
    return list(SUITES[suite])


def find_config(label):
    for c in suite_configs("all"):
        if c.label == label:
            return c
    raise ValueError(f"unknown config label {label!r}")


# --------------------------------------------------------------------------
# Posterior moments and metrics (shared by profile_run and golden_trace)
# --------------------------------------------------------------------------

DIAG_SEED = 2026
DIAG_MC_SAMPLES = 200_000


def posterior_moments(vp, n_mc=DIAG_MC_SAMPLES, seed=DIAG_SEED):
    """Original-space posterior mean ``(1, D)`` and covariance ``(D, D)``.

    Exact through the transformer's affine map when every coordinate is
    unbounded (``x = mu + A u``, ``A = diag(delta) R_mat diag(scale)``,
    including any rotoscale warp); otherwise Monte Carlo on a deep copy with
    a dedicated generator, so the run's stream is never consumed. Returns
    ``(mean, cov, method)``.
    """
    pt = vp.parameter_transformer
    if np.all(pt.type == 0):
        mean_u, cov_u = vp.moments(orig_flag=False, cov_flag=True)
        mean_u = np.reshape(mean_u, (1, -1))
        D = mean_u.shape[1]
        R = np.eye(D) if pt.R_mat is None else np.asarray(pt.R_mat)
        scale = np.ones(D) if pt.scale is None else np.asarray(pt.scale)
        A = np.diag(pt.delta) @ R @ np.diag(scale)
        mean = np.reshape(pt.inverse(mean_u), (1, D))
        check = (A @ mean_u.T).T + np.reshape(
            pt.inverse(np.zeros((1, D))), (1, D)
        )
        if not np.allclose(mean, check, rtol=1e-8, atol=1e-10):
            raise RuntimeError("transformer is not affine as assumed")
        cov = A @ cov_u @ A.T
        return mean, cov, "affine"
    vp2 = copy.deepcopy(vp)
    vp2.rng = np.random.default_rng(seed)
    mean, cov = vp2.moments(N=int(n_mc), orig_flag=True, cov_flag=True)
    return np.reshape(mean, (1, -1)), cov, "mc"


def metrics(problem, vp, elbo):
    """``elbo_err``, ``rmse``, ``gskl`` against the problem's truth (NaN when
    unknown), plus the posterior moments used."""
    from pyvbmc.stats import kl_div_mvn

    mean, cov, method = posterior_moments(vp)
    out = {
        "elbo_err": float("nan"),
        "rmse": float("nan"),
        "gskl": float("nan"),
        "moment_method": method,
        "post_mean": mean,
        "post_cov": cov,
    }
    if problem.ln_Z is not None:
        out["elbo_err"] = float(abs(elbo - problem.ln_Z))
    if problem.true_mean is not None:
        out["rmse"] = float(np.sqrt(np.mean((mean - problem.true_mean) ** 2)))
    if problem.true_cov is not None and problem.true_mean is not None:
        kl = kl_div_mvn(mean, cov, problem.true_mean, problem.true_cov)
        out["gskl"] = float(0.5 * np.sum(kl))
    return out


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_problem(prob, n_ref=200, n_draws=2_000_000, seed=7):
    """Verify one problem; returns a dict of diagnostics."""
    rng = np.random.default_rng(seed)
    res = {"label": f"{prob.name}_D{prob.D}", "ok": True, "msgs": []}
    D = prob.D
    # (a) implementation vs independent reference density
    if prob.reference_logpdf is not None:
        X = rng.uniform(prob.plb, prob.pub, size=(n_ref, D))
        d = np.max(np.abs(prob.log_density_vec(X) - prob.reference_logpdf(X)))
        res["max_abs_logpdf_diff"] = float(d)
        # 1e-6: the cigar's covariance has condition number 1e8 and scipy's
        # eigendecomposition path differs from our Cholesky by ~3e-8.
        if d > 1e-6:
            res["ok"] = False
            res["msgs"].append(f"density differs from reference by {d:.2e}")
    # (b) moments vs exact samples
    if prob.sampler is not None and prob.true_mean is not None:
        S = prob.sampler(n_draws, rng)
        m = S.mean(0)
        c = np.cov(S.T)
        sd = np.sqrt(np.diag(prob.true_cov))
        z = (m - prob.true_mean.ravel()) / (sd / np.sqrt(n_draws))
        res["mean_zscore_max"] = float(np.max(np.abs(z)))
        rel = np.max(np.abs(c - prob.true_cov)) / np.max(np.abs(prob.true_cov))
        res["cov_rel_err_max"] = float(rel)
        if np.max(np.abs(z)) > 4.5:
            res["ok"] = False
            res["msgs"].append(
                f"sampled mean off: max |z| = {np.max(np.abs(z)):.1f}"
            )
        if rel > 0.01:
            res["ok"] = False
            res["msgs"].append(f"sampled cov off: rel err {rel:.3f}")
        # box coverage of the samples
        inside = np.mean(np.all((S >= prob.plb) & (S <= prob.pub), axis=1))
        res["box_mass"] = float(inside)
    # (c) quadrature stability where ln Z comes from a grid
    if prob.name == "rosenbrock":
        g = _ROSENBROCK_CACHE["grid"]
        g2 = _Grid1D(
            lambda x: np.interp(x, g.x, np.log(g.p + 1e-300) + g.ln_Z),
            g.x[0],
            g.x[-1],
            n=2 * len(g.x) - 1,
        )
        res["ln_Z_grid_refine_diff"] = float(abs(g2.ln_Z - g.ln_Z))
    if prob.name == "student":
        d = 0.0
        for gr in _STUDENT_CACHE[D]:
            g2 = _Grid1D(
                lambda x, gr=gr: np.interp(
                    x, gr.x, np.log(gr.p + 1e-300) + gr.ln_Z
                ),
                gr.x[0],
                gr.x[-1],
                n=2 * len(gr.x) - 1,
            )
            d = max(d, abs(g2.ln_Z - gr.ln_Z))
        res["ln_Z_grid_refine_diff"] = float(d)
    return res


def run_check(configs, only=None):
    seen = set()
    all_ok = True
    for cfg in configs:
        key = (cfg.name, cfg.D)
        if key in seen or (only and cfg.name != only):
            continue
        seen.add(key)
        t0 = time.time()
        prob = make_problem(cfg.name, cfg.D)
        r = check_problem(prob)
        r["seconds"] = round(time.time() - t0, 1)
        r["ln_Z"] = prob.ln_Z
        r["plb"] = np.round(prob.plb.ravel(), 2).tolist()
        r["pub"] = np.round(prob.pub.ravel(), 2).tolist()
        all_ok &= r["ok"]
        status = "ok " if r["ok"] else "FAIL"
        extras = {
            k: v
            for k, v in r.items()
            if k not in ("label", "ok", "msgs", "plb", "pub")
        }
        print(f"[check] {status} {r['label']:14s} {extras}", flush=True)
        print(f"        box plb={r['plb']} pub={r['pub']}", flush=True)
        for m in r["msgs"]:
            print(f"        ! {m}", flush=True)
    if only in (None, "logreg"):
        t0 = time.time()
        ref = logreg_reference()
        print(
            f"[check] logreg reference ({time.time() - t0:.0f} s): ln_Z ="
            f" {ref['ln_Z']:.5f} +- {ref['se_ln_Z']:.5f}, ESS ="
            f" {ref['ess']:.0f} of {ref['n_draws']}",
            flush=True,
        )
        print("        mean =", np.round(ref["mean"], 5).tolist(), flush=True)
        print("        cov  =", np.round(ref["cov"], 5).tolist(), flush=True)
        if LOGREG_TRUTH["ln_Z"] is not None:
            d = abs(ref["ln_Z"] - LOGREG_TRUTH["ln_Z"])
            dm = np.max(np.abs(ref["mean"] - np.array(LOGREG_TRUTH["mean"])))
            print(
                f"        stored constants: |d ln_Z| = {d:.5f}, max |d mean| ="
                f" {dm:.5f}",
                flush=True,
            )
            if d > 5 * ref["se_ln_Z"] + 1e-3 or dm > 0.02:
                all_ok = False
                print("        ! stored logreg constants disagree", flush=True)
        else:
            print("        ! LOGREG_TRUTH not filled in yet", flush=True)
    return all_ok


def run_smoke(configs, seed=0):
    import psutil

    from pyvbmc import VBMC

    proc = psutil.Process()
    all_ok = True
    for cfg in configs:
        prob = cfg.make(seed=seed)
        args, options = prob.vbmc_args()
        options.update(
            max_iter=2,
            min_iter=0,
            min_fun_evals=0,
            do_final_boost=False,
            display="off",
            plot=False,
            print_iteration_header=False,
        )
        t0 = time.time()
        try:
            if prob.noisy:
                out = prob.fun(prob.x0)
                assert isinstance(out, tuple) and len(out) == 2
                assert np.isfinite(out[0]) and out[1] > 0
            vbmc = VBMC(*args, options=options, seed=seed)
            vp, results = vbmc.optimize()
            ok = np.isfinite(results["elbo"])
            met = metrics(prob, vp, results["elbo"])
            msg = (
                f"iters={results['iterations'] + 1} evals={results['func_count']}"
                f" elbo={results['elbo']:.3f} elbo_err={met['elbo_err']:.3f}"
                f" rmse={met['rmse']:.3f} gskl={met['gskl']:.3f}"
                f" ({met['moment_method']})"
            )
        except Exception as e:  # noqa: BLE001
            ok = False
            msg = f"EXCEPTION {type(e).__name__}: {e}"
        all_ok &= ok
        rss = (
            proc.memory_info().peak_wset / 2**20
            if hasattr(proc.memory_info(), "peak_wset")
            else proc.memory_info().rss / 2**20
        )
        print(
            f"[smoke] {'ok  ' if ok else 'FAIL'} {cfg.label:28s}"
            f" {time.time() - t0:6.1f} s  peakRSS {rss:6.0f} MB  {msg}",
            flush=True,
        )
    return all_ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="list suites")
    ap.add_argument("--check", action="store_true", help="verify targets")
    ap.add_argument("--smoke", action="store_true", help="2-iteration runs")
    ap.add_argument("--suite", default=None, help="smoke|profile|golden|all")
    ap.add_argument(
        "--only", default=None, help="restrict --check to a target"
    )
    args = ap.parse_args(argv)
    if args.list or not (args.check or args.smoke):
        for s, cfgs in SUITES.items():
            print(f"{s}:")
            for c in cfgs:
                print(f"    {c.label:28s} options={c.options_dict()}")
        return 0
    ok = True
    if args.check:
        ok &= run_check(suite_configs(args.suite or "all"), only=args.only)
    if args.smoke:
        ok &= run_smoke(suite_configs(args.suite or "smoke"))
    print("[benchmark_targets]", "all ok" if ok else "FAILURES", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
