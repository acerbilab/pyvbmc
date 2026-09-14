"""Benchmark target suite for PyVBMC developer tooling.

One place that defines every benchmark target (log density, ground truth,
plausible box, VBMC options) and the named suites built from them, so that
``profile_run.py`` / ``profile_suite.py`` (profiling) and ``golden_trace.py``
(regression population) never drift apart. Plan and rationale:
``dev/plans/benchmark-suite-and-golden-traces.md``; the suite was asked for in
``dev/2026-09-02-modernization-discussion.md`` section 10.

Targets (``make_problem(name, D)``), hard bounds infinite unless stated:

The paper set (Acerbi 2018, section 4.1; each likelihood times the papers'
broad normal prior, centred at the family's expected mean with SD 3x the
likelihood's marginal SD; plausible box = prior mean -+ 1 prior SD):

``lumpy``       12-component Gaussian mixture; truth analytic
``student``     product of Student-t, nu in [2.5, 2 + D/2]; truth by 1-D
                quadratures
``cigar``       one axis 100x longer than the others, seeded orthogonal
                mixing; truth analytic

Extras, declared as such (not paper targets):

``banana``      volume-preserving transform of a Gaussian, D >= 2, no prior:
                exact truth at any D, curvature comparable to the notebook
                Rosenbrock; box = 0 -+ 3 marginal SD
``rosenbrock``  the test/notebook Rosenbrock with a N(0, 3^2) prior, D = 2,
                which is also the 2020 paper's Fig. 1 toy target (LML -2.27);
                truth by 1-D quadrature (the x2 integral is analytic)
``logreg``      Bayesian logistic regression on fixed synthetic data, D = 5,
                bounded with a uniform prior in the 2020 style; truth by
                defensive importance sampling, stored as constants

The S-VBMC paper's two synthetic targets (Silvestrin, Li & Acerbi 2025),
ported from the standalone ``svbmc`` package at commit 13a78f6; D = 2,
unnormalized, unbounded, plausible box [-10, 10]^2:

``gmm``         20 Gaussian components in four well-separated clusters;
                truth analytic, ln Z = log 20
``ring``        thin circular ridge, radius 8 and radial SD 0.1; truth by
                radial quadrature. The pinned source adds a ``log r`` term
                that the paper's own runs did not use; ``_ring`` records
                how that was settled

Real-data targets (problems of the 2020 noisy paper), bounded, spline-
trapezoidal prior with pivots at the plausible bounds:

``timing``      Bayesian time-interval reproduction (Acerbi, Wolpert &
                Vijayakumar 2012), D = 5, 1512 trials of one subject; the
                observer's response distribution is integrated numerically,
                about 40 to 50 ms per evaluation
``multisensory_s1``, ``multisensory_s2``
                visuo-vestibular causal inference (Acerbi, Dokka, Angelaki &
                Ma 2018), D = 6, one target per subject; analytic likelihood,
                vectorized over rows, about 7 ms per 100 rows

Their data are the plain archives under ``data/`` (layout and provenance in
``data/README.md``), and their ground truths the files ``data/truths/``
holds once ``make_benchmark_truths.py`` has run: until then ``ln_Z``, the
moments and the sampler stay ``None`` and the metrics that need them are
NaN. Both likelihoods are ports of the lab's benchflow implementations and
are pinned in ``--check`` to values computed by those implementations (for
timing, by the original MATLAB code).

Smoke / legacy targets with the tests' and notebooks' boxes (not benchmark
entries): ``normal`` (independent Gaussian, SDs 1..D), ``corr`` (rotated
Gaussian), ``halfnormal`` (Gaussian on the negative orthant, bounded; ln Z =
-D ln 2).

Start point: drawn uniformly inside the plausible box from a stream spawned
off the run seed (``make_problem(..., seed=)``), as in the papers; it never
uses the truth. The plausible boxes of the paper set and of banana follow
the papers' prior rule and therefore scale with the target's marginal SD,
as the papers' do; they do not use the realized posterior otherwise.

Any target can be made noisy with ``noise_sd``: the callable then returns
``(y + sd * eps, sd)`` and ``options["specify_target_noise"] = True``.

Command line::

    python dev/scripts/benchmark_targets.py --list
    python dev/scripts/benchmark_targets.py --check [--suite all]
    python dev/scripts/benchmark_targets.py --smoke [--suite smoke]

``--check`` verifies each implementation against an independent reference
density, against the moments of exact samples and against the pinned values,
and integrates numerically only where ln Z is not analytic by construction.
For the real-data targets the moments and the sampler both come from the
stored importance-weighted population, so that comparison only checks the
truth file against itself; the pins and the generator's own ``--check``
are the real gates.
``--smoke`` runs every config of a suite through two VBMC iterations. Both
take ``--only``, a comma-separated list of target names or config labels.
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
from scipy import integrate, stats
from scipy.special import expit, log_expit, logsumexp

REPO_ROOT = Path(__file__).resolve().parents[2]

# Seeds that freeze the *structure* of randomly generated targets. They are
# independent of the run seed on purpose: every run of every version sees
# the same target.
STRUCTURE_SEED = 20260900
LOGREG_SEED = 20260905


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
    ``None`` when unknown. What VBMC receives: ``fun``, the bounds, a
    plausible box that follows the papers' prior rule (it scales with the
    target's marginal SD, nothing else), and an ``x0`` drawn uniformly in
    that box by ``make_problem``; ``true_mean``, ``true_cov`` and ``ln_Z``
    are used only by the metrics.

    Three fields serve targets whose truth is not analytic.
    ``log_likelihood_vec`` is the likelihood alone, where the density is a
    likelihood times an explicit prior. ``pins`` are reference values from
    an independent implementation, ``(x, expected, which, tol)`` with
    ``which`` naming the function evaluated at ``x`` (``"loglik"`` or
    ``"logp"``); ``--check`` fails if any is off by more than ``tol``.
    ``sampler_n_eff`` is the number of independent draws a resampling
    ``sampler`` can offer, which is what the z-scores of ``--check`` must
    use instead of the number of draws requested.
    """

    name: str
    D: int
    log_density_vec: Callable[[np.ndarray], np.ndarray]
    x0: Optional[np.ndarray]
    lb: np.ndarray
    ub: np.ndarray
    plb: np.ndarray
    pub: np.ndarray
    ln_Z: Optional[float] = None
    true_mean: Optional[np.ndarray] = None
    true_cov: Optional[np.ndarray] = None
    sampler: Optional[Callable[[int, np.random.Generator], np.ndarray]] = None
    reference_logpdf: Optional[Callable[[np.ndarray], np.ndarray]] = None
    log_likelihood_vec: Optional[Callable[[np.ndarray], np.ndarray]] = None
    pins: tuple = ()
    sampler_n_eff: Optional[int] = None
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
    # always a fresh array: a view of a module-level constant or of a cached
    # data archive would let one in-place write reach every problem
    return (
        np.full((1, D), float(v))
        if np.ndim(v) == 0
        else np.array(v, dtype=float).reshape(1, D)
    )


def _inf_bounds(D):
    return np.full((1, D), -np.inf), np.full((1, D), np.inf)


PRIOR_SD_FACTOR = 3.0  # the papers' "3-4 times the SD in each dimension"


def _paper_box(center, sd):
    """The papers' plausible box: prior mean -+ 1 prior SD, where the prior is
    a broad normal centred at the *expected* mean of the target family with
    SD = PRIOR_SD_FACTOR x the target's marginal SD (Acerbi 2018, sections 3.5
    and 4). ``center`` is the family's expected mean, ``sd`` the likelihood's
    marginal SDs; neither depends on the realized posterior beyond its scale,
    which is the papers' own convention."""
    center = np.reshape(center, (1, -1))
    half = PRIOR_SD_FACTOR * np.reshape(sd, (1, -1))
    return center - half, center + half


def _draw_x0(plb, pub, rng):
    """Start point drawn uniformly inside the plausible box (Acerbi 2018,
    section 4: "we draw the starting point x0 uniformly from a box within 1
    prior standard deviation from the prior mean")."""
    return plb + rng.random(plb.shape) * (pub - plb)


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
        x0=None,
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
        x0=None,
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
        x0=None,
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
        x0=None,
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
    # Family mean 0 (exact); box = 0 -+ 3 marginal SD. No prior: a normal
    # prior in x-space would break the closed form, and this is not a paper
    # target (the 2020 paper's Fig. 1 "banana" is the Rosenbrock above).
    plb, pub = _paper_box(np.zeros(D), np.sqrt(np.diag(cov)))
    lb, ub = _inf_bounds(D)
    return Problem(
        name="banana",
        D=D,
        log_density_vec=logp,
        x0=None,
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
            " unit Jacobian, no prior; not a paper target. The true cov is"
            " diagonal, so gsKL and RMSE cannot see the ridge; MMTV sees the"
            " skewed x2 marginal, only elbo_err sees the joint"
        ),
    )


def _cigar(D):
    rng = np.random.default_rng(STRUCTURE_SEED + D)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    ell = np.full(D, 0.01)
    ell[-1] = 1.0
    cov_lik = Q @ np.diag(ell**2) @ Q.T  # the one expression used everywhere
    mean = np.linspace(-0.5, 0.5, D)  # the family's centre (deterministic)
    L_lik = np.linalg.cholesky(cov_lik)
    # The papers' prior: normal at the family mean with SD 3 x the marginal
    # SD of the likelihood, per dimension. Gaussian x Gaussian stays Gaussian.
    prior_sd = PRIOR_SD_FACTOR * np.sqrt(np.diag(cov_lik))
    P = np.diag(prior_sd**2)
    cov_post = np.linalg.inv(np.linalg.inv(cov_lik) + np.linalg.inv(P))
    cov_post = 0.5 * (cov_post + cov_post.T)
    # both factors are centred at `mean`, so the posterior mean is `mean` and
    # ln Z = ln N(mean; mean, cov_lik + P) = -0.5 ln det(2 pi (cov_lik + P))
    ln_Z = float(-0.5 * np.linalg.slogdet(2 * np.pi * (cov_lik + P))[1])
    L_post = np.linalg.cholesky(cov_post)
    mvn_lik = stats.multivariate_normal(mean, cov_lik)

    def logp(X):
        X = np.atleast_2d(X)
        return _mvn_logpdf_chol(X, mean, L_lik) + np.sum(
            stats.norm.logpdf(X, mean, prior_sd), axis=1
        )

    def ref(X):
        X = np.atleast_2d(X)
        return mvn_lik.logpdf(X) + np.sum(
            stats.norm.logpdf(X, mean, prior_sd), axis=1
        )

    def sampler(n, rng):
        return mean + rng.standard_normal((n, D)) @ L_post.T

    plb, pub = _paper_box(mean, np.sqrt(np.diag(cov_lik)))
    lb, ub = _inf_bounds(D)
    return Problem(
        name="cigar",
        D=D,
        log_density_vec=logp,
        x0=None,
        lb=lb,
        ub=ub,
        plb=plb,
        pub=pub,
        ln_Z=ln_Z,
        true_mean=mean.reshape(1, D),
        true_cov=cov_post,
        sampler=sampler,
        reference_logpdf=ref,
        notes=(
            "likelihood: SDs 0.01 on D-1 axes and 1 on one, mixed by a seeded"
            f" orthogonal matrix (det {np.linalg.det(Q):+.0f}), centred at"
            " linspace(-0.5, 0.5); times the papers' prior N(centre,"
            f" ({PRIOR_SD_FACTOR:g} marginal SD)^2); box = prior mean -+ 1 prior SD"
        ),
    )


def _mixture_moments(w, mus, covs):
    """Mean and covariance of a Gaussian mixture.

    ``w`` are the weights ``(K,)``, ``mus`` the component means ``(K, D)``
    and ``covs`` either the per-coordinate variances ``(K, D)`` of diagonal
    components or the full component covariances ``(K, D, D)``.
    """
    covs = np.asarray(covs, dtype=float)
    if covs.ndim not in (2, 3):
        raise ValueError(
            "covs must be (K, D) variances or (K, D, D) covariances, got"
            f" shape {covs.shape}"
        )
    mean = w @ mus
    cov = np.zeros((mus.shape[1], mus.shape[1]))
    for k in range(len(w)):
        cov_k = np.diag(covs[k]) if covs.ndim == 2 else covs[k]
        cov += w[k] * (cov_k + np.outer(mus[k], mus[k]))
    cov -= np.outer(mean, mean)
    return mean, cov


def _lumpy(D, n_comp=12):
    rng = np.random.default_rng(STRUCTURE_SEED + D)
    mus = rng.random((n_comp, D))
    sds = rng.uniform(0.2, 0.6, size=(n_comp, D))
    w = rng.dirichlet(np.ones(n_comp))
    log_w = np.log(w)
    comp_log_norm = -np.sum(np.log(sds), axis=1) - 0.5 * D * np.log(2 * np.pi)
    # The papers' prior: normal at the family's expected mean (0.5 per
    # coordinate for means U[0, 1]) with SD 3 x the likelihood's marginal SD.
    lik_mean, lik_cov = _mixture_moments(w, mus, sds**2)
    center = np.full(D, 0.5)
    prior_sd = PRIOR_SD_FACTOR * np.sqrt(np.diag(lik_cov))
    # mixture x Gaussian is a mixture: component k becomes N(m_k, V_k) with
    # weight w_k N(mu_k; c, S_k + P); ln Z is the log of the weight sum
    post_var = 1.0 / (1.0 / sds**2 + 1.0 / prior_sd**2)  # (K, D)
    post_mu = post_var * (mus / sds**2 + center / prior_sd**2)
    log_wpost = log_w + np.sum(
        stats.norm.logpdf(mus, center, np.sqrt(sds**2 + prior_sd**2)),
        axis=1,
    )
    ln_Z = float(logsumexp(log_wpost))
    w_post = np.exp(log_wpost - ln_Z)
    post_mean, post_cov = _mixture_moments(w_post, post_mu, post_var)

    def logp(X):
        X = np.atleast_2d(X)
        z = (X[:, None, :] - mus[None, :, :]) / sds[None, :, :]
        lc = -0.5 * np.sum(z**2, axis=2) + comp_log_norm + log_w  # (n, K)
        return logsumexp(lc, axis=1) + np.sum(
            stats.norm.logpdf(X, center, prior_sd), axis=1
        )

    comps = [
        stats.multivariate_normal(mus[k], np.diag(sds[k] ** 2))
        for k in range(n_comp)
    ]

    def ref(X):
        X = np.atleast_2d(X)
        lc = np.stack([c.logpdf(X) + log_w[k] for k, c in enumerate(comps)])
        return logsumexp(np.atleast_2d(lc), axis=0) + np.sum(
            stats.norm.logpdf(X, center, prior_sd), axis=1
        )

    def sampler(n, rng):
        k = rng.choice(n_comp, size=n, p=w_post)
        return post_mu[k] + rng.standard_normal((n, D)) * np.sqrt(post_var[k])

    plb, pub = _paper_box(center, np.sqrt(np.diag(lik_cov)))
    lb, ub = _inf_bounds(D)
    return Problem(
        name="lumpy",
        D=D,
        log_density_vec=logp,
        x0=None,
        lb=lb,
        ub=ub,
        plb=plb,
        pub=pub,
        ln_Z=ln_Z,
        true_mean=post_mean.reshape(1, D),
        true_cov=post_cov,
        sampler=sampler,
        reference_logpdf=ref,
        notes=(
            f"likelihood: {n_comp} Gaussians, means U[0,1]^D, SDs U[0.2,0.6],"
            f" Dirichlet(1) weights (top weight {w.max():.2f}), Acerbi 2018"
            f" section 4.1; times the papers' prior N(0.5, ({PRIOR_SD_FACTOR:g}"
            " marginal SD)^2); box = prior mean -+ 1 prior SD"
        ),
    )


_STUDENT_CACHE = {}


def _student(D):
    nu = np.linspace(2.5, 2.0 + D / 2.0, D)
    sd_t = np.sqrt(nu / (nu - 2.0))
    prior_sd = PRIOR_SD_FACTOR * sd_t  # the papers' broad normal prior

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

    plb, pub = _paper_box(np.zeros(D), sd_t)  # = prior mean -+ 1 prior SD
    lb, ub = _inf_bounds(D)
    return Problem(
        name="student",
        D=D,
        log_density_vec=logp,
        x0=None,
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
            f" times the papers' prior N(0, ({PRIOR_SD_FACTOR:g} sd_t)^2); truth"
            " by per-coordinate quadrature; box = prior mean -+ 1 prior SD"
        ),
    )


# Logistic regression in the style of the 2020 noisy benchmark: bounded
# parameters with a uniform prior over the box (Acerbi 2020, section 4.1),
# so it also exercises the probit-transformed bounded path. Design and data
# are frozen by LOGREG_SEED. The truth constants below are produced by
# ``--check`` (defensive importance sampling: half a t(4) around the
# constrained mode, half uniform over the box; 2e6 draws). Rerun
# ``python dev/scripts/benchmark_targets.py --check --only logreg`` to
# regenerate them; the standard error and ESS are printed alongside.
LOGREG_N = 50
LOGREG_RHO = 0.95
LOGREG_N_RARE = 6
LOGREG_W_TRUE = np.array([0.3, 1.0, -1.0, 0.8, 1.5])
LOGREG_LB, LOGREG_UB = -10.0, 10.0  # hard bounds; uniform prior over the box
LOGREG_PLB, LOGREG_PUB = -5.0, 5.0  # a modeller's plausible logit effects
LOGREG_TRUTH = {
    # --check on 2026-09-03: ln_Z = -34.89349 +- 0.00209 (IS standard error),
    # ESS = 205544 of 2e6 draws (the uniform half of the defensive proposal
    # costs ESS but bounds the weights; w4's posterior is a plateau running
    # into the upper bound, mean 5.5, SD 2.7).
    "ln_Z": -34.89349,
    "mean": [0.95809, -1.70466, 1.72039, 1.53800, 5.47709],
    "cov": [
        [0.16504, -0.09694, 0.10047, 0.07820, -0.07712],
        [-0.09694, 2.18496, -1.84405, -0.14672, 0.06923],
        [0.10047, -1.84405, 1.72262, 0.18190, -0.08630],
        [0.07820, -0.14672, 0.18190, 0.32108, -0.01763],
        [-0.07712, 0.06923, -0.08630, -0.01763, 7.41547],
    ],
}


def _logreg_loglik(W, Xd, y):
    W = np.atleast_2d(W)
    eta = W @ Xd.T
    return np.sum(y * log_expit(eta) + (1 - y) * log_expit(-eta), axis=1)


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
    log_prior = -D * np.log(LOGREG_UB - LOGREG_LB)  # uniform over the box

    def logp(W):
        W = np.atleast_2d(W)
        out = _logreg_loglik(W, Xd, y) + log_prior
        inside = np.all((W >= LOGREG_LB) & (W <= LOGREG_UB), axis=1)
        return np.where(inside, out, -np.inf)

    truth = LOGREG_TRUTH
    return Problem(
        name="logreg",
        D=D,
        log_density_vec=logp,
        x0=None,
        lb=np.full((1, D), LOGREG_LB),
        ub=np.full((1, D), LOGREG_UB),
        plb=np.full((1, D), LOGREG_PLB),
        pub=np.full((1, D), LOGREG_PUB),
        ln_Z=truth["ln_Z"],
        true_mean=None if truth["mean"] is None else np.array([truth["mean"]]),
        true_cov=None if truth["cov"] is None else np.array(truth["cov"]),
        sampler=_logreg_sampler,
        reference_logpdf=None,
        notes=(
            f"logistic regression, {LOGREG_N} trials, predictors 1-2 at rho"
            f" {LOGREG_RHO}, predictor 4 rare ({LOGREG_N_RARE} ones, all"
            f" successes: its coefficient is identified only by the bound);"
            f" uniform prior on [{LOGREG_LB:g}, {LOGREG_UB:g}]^D (2020 style),"
            f" plausible box -+{LOGREG_PUB:g}; not a paper problem"
        ),
    )


_LOGREG_IS_CACHE = {}


def _logreg_sampler(n, rng, m=200_000):
    """Approximate exact draws for ``logreg``: importance resampling (with
    replacement) from a cached defensive-IS population of ``m`` draws. Good
    enough for marginal total-variation distances on a 2^13 KDE grid; the
    ESS is printed by ``--check`` and sets a floor on logreg's MMTV/gsKL."""
    if "pop" not in _LOGREG_IS_CACHE:
        ref = logreg_reference(m=m, seed=11, return_draws=True)
        _LOGREG_IS_CACHE["pop"] = (ref["draws"], ref["weights"])
    W, wn = _LOGREG_IS_CACHE["pop"]
    return W[rng.choice(len(wn), size=n, p=wn)]


def logreg_reference(
    m=2_000_000, chunk=20_000, df=4, scale=2.0, seed=1, return_draws=False
):
    """Defensive importance-sampling truth for ``logreg`` (ln Z, mean, cov).

    Proposal: an equal mixture of a t(``df``) centred at the constrained
    posterior mode with ``scale`` x the inverse Hessian (per-coordinate scale
    capped at the box half-width, since the rare predictor's coefficient
    has a flat likelihood towards the upper bound) and the uniform prior over
    the box, which guarantees bounded weights.
    """
    from scipy import optimize

    Xd, y = _logreg_data()
    D = Xd.shape[1]
    prob = _logreg(D)
    lb, ub = LOGREG_LB, LOGREG_UB

    def f(w):
        return -_logreg_loglik(w, Xd, y)[0]

    def g(w):
        p = expit(Xd @ w)
        return -(Xd.T @ (y - p))

    res = optimize.minimize(
        f,
        np.zeros(D),
        jac=g,
        method="L-BFGS-B",
        bounds=[(lb + 1e-6, ub - 1e-6)] * D,
    )
    p = expit(Xd @ res.x)
    H = Xd.T @ (Xd * (p * (1 - p))[:, None]) + 1e-8 * np.eye(D)
    cov_t = np.linalg.inv(H) * scale
    s = np.sqrt(np.diag(cov_t))
    fac = np.minimum(1.0, 0.5 * (ub - lb) / s)
    cov_t = cov_t * np.outer(fac, fac)
    prop = stats.multivariate_t(loc=res.x, shape=cov_t, df=df, seed=seed)
    rng = np.random.default_rng(seed)
    log_vol = D * np.log(ub - lb)
    Ws, lws = [], []
    for _ in range(m // chunk):
        n_t = chunk // 2
        W = np.vstack(
            [prop.rvs(n_t), rng.uniform(lb, ub, size=(chunk - n_t, D))]
        )
        log_q = np.logaddexp(
            np.log(0.5) + prop.logpdf(W), np.log(0.5) - log_vol
        )
        lws.append(prob.log_density_vec(W) - log_q)  # -inf outside the box
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
    out = dict(
        ln_Z=float(ln_Z),
        se_ln_Z=float(se_ln_Z),
        ess=float(ess),
        mean=mean,
        cov=cov,
        laplace_mode=res.x,
        proposal_cov=cov_t,
        n_draws=len(w),
    )
    if return_draws:
        out["draws"] = W
        out["weights"] = wn
    return out


# --------------------------------------------------------------------------
# The two synthetic targets of the S-VBMC paper (Silvestrin, Li & Acerbi
# 2025): a 20-component Gaussian mixture and a thin ring, both
# two-dimensional, unnormalized and unbounded. They are ported from
# ``src/svbmc/targets.py`` of the standalone ``svbmc`` package at commit
# 13a78f6c4a3ffe9c4557fee4f4b8f67c98b29e01; the mixture reproduces that
# source bit for bit, the ring deliberately does not (see ``_ring``). Three
# values of each log density are pinned, so that the definitions are
# checkable without that source; the samplers and the ground truths below
# are this suite's own. Both targets are used on the plausible box
# [-10, 10]^2, the box of the upstream example notebook
# (``examples/svbmc_example_1_basic_usage.ipynb``) and of the fitted
# posteriors the paper's runs left behind.
# --------------------------------------------------------------------------

SVBMC_PLB, SVBMC_PUB = -10.0, 10.0

# The paper's 20 component means and covariances: five components around
# each of four centres, unit variances and correlations of +-0.5.
GMM_MUS = np.array(
    [
        [-8.26197841, -8.75505571],
        [-6.99244831, -8.41215022],
        [-7.94352608, -7.96066128],
        [-6.94653653, -6.25663673],
        [-8.31800964, -8.86116925],
        [-5.35377745, 9.09808750],
        [-6.57295220, 7.03574990],
        [-6.51023244, 7.15063425],
        [-6.45464607, 6.41256552],
        [-5.08644145, 8.34791521],
        [5.39972878, -5.16531348],
        [6.84830038, -4.83860713],
        [6.79133501, -4.86439174],
        [6.04858289, -7.71908624],
        [6.67275726, -4.80883357],
        [5.30248146, 4.88622069],
        [6.22996358, 4.04107658],
        [4.86277661, 6.56073311],
        [4.38009601, 6.32750016],
        [5.11056247, 6.32234096],
    ]
)
_GMM_RHO = np.array(
    # one correlation per component, in the order of GMM_MUS: five
    # components per row, one row per cluster
    # fmt: off
    [+0.5, +0.5, -0.5, -0.5, -0.5,
     -0.5, +0.5, +0.5, -0.5, +0.5,
     +0.5, -0.5, -0.5, +0.5, +0.5,
     -0.5, -0.5, -0.5, +0.5, -0.5]
    # fmt: on
)
GMM_SIGMAS = np.array([[[1.0, rho], [rho, 1.0]] for rho in _GMM_RHO])
# Values of the upstream log densities, to full precision.
GMM_PINS = (
    ((0.0, 0.0), -22.64583186718372),
    ((5.39972878, -5.16531348), -1.1124777729834854),
    ((-7.0, 7.0), -0.690501796850522),
)

RING_R, RING_SIGMA = 8.0, 0.1
RING_CENTER = (1.0, -2.0)
# Values of the ring log density as ``_ring`` defines it, at the centre,
# on the ridge and at the origin.
RING_PINS = (
    ((1.0, -2.0), -3200.0),  # the centre, r = 0
    ((9.0, -2.0), 0.0),  # on the ridge, r = R exactly
    ((0.0, 0.0), -1661.1456180001683),
)
# The pinned source's ``Ring().log_pdf`` values at the same three points,
# to full precision: the density above plus ``log r`` -- the term dropped
# here -- with ``r`` clamped to ``np.finfo(float).tiny`` at the centre.
# ``check_problem`` adds that term back and compares, which is the check
# on the port that ``RING_PINS`` alone cannot give.
RING_UPSTREAM_PINS = (
    -3908.396418532264,
    2.0794415416798357,
    -1660.3408990439511,
)
SVBMC_PIN_TOL = 1e-10


def _gmm(D):
    """The S-VBMC paper's multimodal target: 20 Gaussian components in two
    dimensions, in four well-separated clusters of five.

    The log density is the log-sum-exp of the 20 *normalized* component
    densities with no ``- log K`` term, as the upstream ``GMM.log_pdf``
    computes it, so the target integrates to ``K = 20`` over the plane and
    ``ln Z = log 20``. Mean and covariance are those of the equally
    weighted mixture; the sampler picks a component uniformly.
    """
    if D != 2:
        raise ValueError("gmm is defined for D = 2 only")
    mus, sigmas = GMM_MUS, GMM_SIGMAS
    K = mus.shape[0]
    inv_sigmas = np.linalg.inv(sigmas)
    log_norm = -0.5 * (
        np.linalg.slogdet(sigmas)[1] + D * np.log(2 * np.pi)
    )  # (K,)
    chols = np.linalg.cholesky(sigmas)

    def logp(X):
        X = np.atleast_2d(X)
        diff = mus[None, :, :] - X[:, None, :]  # (n, K, D)
        quad = np.einsum("nki,kij,nkj->nk", diff, inv_sigmas, diff)
        return logsumexp(-0.5 * quad + log_norm, axis=1)

    comps = [stats.multivariate_normal(mus[k], sigmas[k]) for k in range(K)]

    def ref(X):
        X = np.atleast_2d(X)
        lc = np.stack([np.atleast_1d(c.logpdf(X)) for c in comps])  # (K, n)
        return logsumexp(lc, axis=0)

    def sampler(n, rng):
        k = rng.integers(K, size=n)
        z = rng.standard_normal((n, D))
        return mus[k] + np.einsum("nij,nj->ni", chols[k], z)

    mean, cov = _mixture_moments(np.full(K, 1.0 / K), mus, sigmas)
    lb, ub = _inf_bounds(D)
    return Problem(
        name="gmm",
        D=D,
        log_density_vec=logp,
        x0=None,
        lb=lb,
        ub=ub,
        plb=np.full((1, D), SVBMC_PLB),
        pub=np.full((1, D), SVBMC_PUB),
        ln_Z=float(np.log(K)),
        true_mean=mean.reshape(1, D),
        true_cov=cov,
        sampler=sampler,
        reference_logpdf=ref,
        pins=tuple(
            (x, expected, "logp", SVBMC_PIN_TOL) for x, expected in GMM_PINS
        ),
        notes=(
            f"{K} unit-variance Gaussians with correlations -+0.5, five"
            " around each of four centres about 15 apart (the S-VBMC"
            " paper's multimodal target); unnormalized, ln Z = log"
            f" {K}; box [{SVBMC_PLB:g}, {SVBMC_PUB:g}]^2 as upstream"
        ),
    )


_RING_CACHE = {}


def _ring(D):
    """The S-VBMC paper's ring: a thin circular ridge of radius 8 and radial
    width 0.1 around ``(1, -2)``, in two dimensions.

    The log density is ``-(r - R)^2 / (2 sigma^2)``, with ``r`` the distance
    to the centre. The upstream ``Ring.log_pdf`` adds ``log r`` (clamped
    away from zero at the centre), which the posteriors of the paper's runs
    say the runs themselves did not: recomputing the ELBO of each of the ten
    fitted ring posteriors kept as fixtures (group ``upstream_Ring`` of
    ``pyvbmc/testing/svbmc/fixtures/``) by Monte Carlo under the two
    definitions reproduces the stored value to within 0.24 nats without the
    term -- a near-uniform residual of about +0.23 nats, the surrogate's own
    optimism on a ridge this thin -- and misses by 2.31 with it: the two
    residuals differ by exactly ``log R = 2.08``, the value of the dropped
    term on the ridge where those posteriors sit. The definition here is
    therefore the paper's, not the pinned source's; ``check_problem`` adds
    the term back at the three pinned points and compares with the source's
    values there (``RING_UPSTREAM_PINS``), so the departure stays a
    deliberate one and not a porting error.

    In polar coordinates the radial marginal is proportional to
    ``r exp(-(r - R)^2 / 2 sigma^2)`` (the factor of ``r`` is the area
    element) and the angle is uniform, which gives ``ln Z``, the mean (the
    centre) and the covariance ``(E[r^2] / 2) I`` below. Upstream's own
    sampler draws the radius from ``N(R, sigma)``, which is not this
    marginal; the sampler here inverts the radial CDF on a fine grid.
    """
    if D != 2:
        raise ValueError("ring is defined for D = 2 only")
    R, sigma = RING_R, RING_SIGMA
    center = np.array(RING_CENTER, dtype=float)

    def logp(X):
        r = np.linalg.norm(np.atleast_2d(X) - center, axis=-1)
        return -0.5 * ((r - R) / sigma) ** 2

    def ref(X):
        # a different route to the same value: the log density of the
        # radius under N(R, sigma), less that normal's normalizing constant
        X = np.atleast_2d(X)
        r = np.hypot(X[:, 0] - center[0], X[:, 1] - center[1])
        return stats.norm.logpdf(r, R, sigma) + 0.5 * np.log(
            2 * np.pi * sigma**2
        )

    if "grid" not in _RING_CACHE:
        # Z = 2 pi int_0^inf r e(r) dr and E[r^2] = int r^3 e / int r e,
        # with e(r) = exp(-(r - R)^2 / 2 sigma^2). The integrand is a narrow
        # peak at r = R, so the quadrature is told where it is; 20 sigma of
        # tail is all there is to double precision.
        def e(r):
            return np.exp(-0.5 * ((r - R) / sigma) ** 2)

        hi = R + 20 * sigma
        i1 = integrate.quad(lambda r: r * e(r), 0.0, hi, points=(R,))[0]
        i3 = integrate.quad(lambda r: r**3 * e(r), 0.0, hi, points=(R,))[0]
        # Over the whole line int r e(r) dr = sigma sqrt(2 pi) R, and the
        # mass below r = 0 is exp(-R^2 / 2 sigma^2), zero in doubles: the
        # closed form is the check on the quadrature.
        _RING_CACHE.update(
            ln_Z=float(np.log(2 * np.pi * i1)),
            ln_Z_closed=float(
                np.log(2 * np.pi * sigma * np.sqrt(2 * np.pi) * R)
            ),
            e_r2=float(i3 / i1),
            grid=_Grid1D(
                lambda r: np.log(r) - 0.5 * ((r - R) / sigma) ** 2,
                max(0.0, R - 10 * sigma),
                R + 10 * sigma,
            ),
        )
    cache = _RING_CACHE

    def sampler(n, rng):
        r = cache["grid"].sample(n, rng)
        theta = 2 * np.pi * rng.random(n)
        return center + r[:, None] * np.c_[np.cos(theta), np.sin(theta)]

    lb, ub = _inf_bounds(D)
    return Problem(
        name="ring",
        D=D,
        log_density_vec=logp,
        x0=None,
        lb=lb,
        ub=ub,
        plb=np.full((1, D), SVBMC_PLB),
        pub=np.full((1, D), SVBMC_PUB),
        ln_Z=cache["ln_Z"],
        true_mean=center.reshape(1, D),
        true_cov=0.5 * cache["e_r2"] * np.eye(D),
        sampler=sampler,
        reference_logpdf=ref,
        pins=tuple(
            (x, expected, "logp", SVBMC_PIN_TOL) for x, expected in RING_PINS
        ),
        notes=(
            f"ring of radius {R:g} and radial SD {sigma:g} centred at"
            f" {RING_CENTER}, density -(r - R)^2 / 2 sigma^2 (the S-VBMC"
            " paper's hardest target, without the log r term of the pinned"
            f" upstream source); unnormalized, ln Z = {cache['ln_Z']:.4f} by"
            f" quadrature; box [{SVBMC_PLB:g}, {SVBMC_PUB:g}]^2 as upstream"
        ),
    )


# --------------------------------------------------------------------------
# Real-data targets (the two problems of the 2020 noisy paper that are pure
# NumPy/SciPy). Data in data/, ground truths in data/truths/; see
# data/README.md for both layouts.
# --------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
TRUTH_DIR = DATA_DIR / "truths"
PIN_TOL = 1e-6  # tolerance of the pinned reference values in --check
REAL_DATA_TARGETS = ("timing", "multisensory_s1", "multisensory_s2")
TRUTH_KEYS = (
    "samples",
    "log_weights",
    "sample_logp",
    "mean",
    "cov",
    "is_ess",
    "ln_z",
    "ln_z_se",
)

_DATA_CACHE = {}


def _load_data(name):
    """The arrays of ``data/{name}.npz``, read once per process."""
    if name not in _DATA_CACHE:
        path = DATA_DIR / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; write it with"
                " dev/scripts/export_benchflow_data.py"
            )
        with np.load(path, allow_pickle=False) as z:
            _DATA_CACHE[name] = {k: z[k] for k in z.files}
    return _DATA_CACHE[name]


def _spline_trapezoid_logpdf(X, a, u, v, b):
    """Log density of the spline-trapezoidal prior at the rows of ``X``.

    Per dimension the density is uniform between the pivots ``u`` and ``v``
    and tapers to zero at the hard bounds ``a`` and ``b`` as the cubic
    ``3 z^2 - 2 z^3`` of the rescaled distance ``z`` from the bound, so that
    both the density and its derivative are continuous; the marginals are
    independent. ``X`` is ``(n, D)``, the four bound arrays are ``(D,)``, the
    result ``(n,)``.

    This is the density of ``pyvbmc.priors.SplineTrapezoidal``, written out
    here so that the benchmark targets do not depend on the package under
    test.
    """
    X = np.atleast_2d(X)
    left = (X >= a) & (X < u)
    plateau = (X >= u) & (X < v)
    right = (X >= v) & (X <= b)
    z = np.zeros(X.shape)
    # the quotients are formed on the whole array and masked afterwards, so
    # a degenerate box (a == u or v == b) only produces discarded values
    with np.errstate(divide="ignore", invalid="ignore"):
        z[left] = ((X - a) / (u - a))[left]
        z[right] = (1.0 - (X - v) / (b - v))[right]
        taper = np.log(3.0 * z**2 - 2.0 * z**3)  # z = 0 at a and b
    # each taper integrates to half its width, so before normalization the
    # marginal integrates to (v - u) + (u - a) / 2 + (b - v) / 2
    log_norm = np.log(0.5 * (v - u + b - a))
    log_pdf = np.where(
        left | plateau | right,
        np.where(plateau, 0.0, taper) - log_norm,
        -np.inf,
    )
    return np.sum(log_pdf, axis=1)


def _bounded_log_joint(log_likelihood_vec, a, u, v, b):
    """Log joint of a real-data target: its likelihood times the
    spline-trapezoidal prior on the box, and -inf wherever that prior
    vanishes (the likelihoods are defined only inside the box, so they are
    never called there)."""

    def logp(X):
        X = np.atleast_2d(X)
        log_prior = _spline_trapezoid_logpdf(X, a, u, v, b)
        out = np.full(X.shape[0], -np.inf)
        inside = np.isfinite(log_prior)
        if np.any(inside):
            out[inside] = log_likelihood_vec(X[inside]) + log_prior[inside]
        return out

    return logp


_TRUTH_CACHE = {}


def _load_truth(name, D, lb, ub):
    """The arrays of ``data/truths/{name}.npz``, read once per process and
    checked against the target: an importance-weighted population of draws
    ``(n, D)`` inside the hard bounds (so in the original space) with
    normalized log weights, moments of matching shape, a scalar ``ln_z``
    and the weights' effective sample size ``is_ess``."""
    if name in _TRUTH_CACHE:
        return _TRUTH_CACHE[name]
    path = TRUTH_DIR / f"{name}.npz"
    if not path.is_file():
        _TRUTH_CACHE[name] = None
        return None
    with np.load(path, allow_pickle=False) as z:
        truth = {k: np.asarray(z[k], dtype=float) for k in z.files}
    missing = [k for k in TRUTH_KEYS if k not in truth]
    if missing:
        raise ValueError(
            f"{path} lacks {', '.join(missing)}: an older layout; regenerate"
            " it with dev/scripts/make_benchmark_truths.py"
        )
    samples = truth["samples"]
    if samples.ndim != 2 or samples.shape[1] != D:
        raise ValueError(f"{path}: samples must be (n, {D})")
    if not np.all((samples >= lb) & (samples <= ub)):
        raise ValueError(f"{path}: draws outside the hard bounds")
    log_w = truth["log_weights"]
    if log_w.shape != (len(samples),):
        raise ValueError(f"{path}: log_weights must be (n,)")
    if abs(logsumexp(log_w)) > 1e-8:
        raise ValueError(f"{path}: log_weights are not normalized")
    if truth["mean"].shape != (D,) or truth["cov"].shape != (D, D):
        raise ValueError(f"{path}: mean must be ({D},) and cov ({D}, {D})")
    if truth["ln_z"].shape != () or truth["is_ess"].shape != ():
        raise ValueError(f"{path}: ln_z and is_ess must be scalars")
    truth["weights"] = np.exp(log_w)
    _TRUTH_CACHE[name] = truth
    return truth


def _attach_truth(prob):
    """Fill a real-data target's truth from ``data/truths/{name}.npz``, or
    record in its notes that the file is not there yet.

    The stored population is importance-weighted (draws from the
    generator's mixture proposal), so the sampler resamples it with its
    weights and ``sampler_n_eff`` is the weights' effective sample size,
    not the draw count."""
    truth = _load_truth(prob.name, prob.D, prob.lb, prob.ub)
    if truth is None:
        prob.notes += (
            "; ground truth not generated yet (write it with"
            " dev/scripts/make_benchmark_truths.py): ln Z and the moments"
            " are unknown and the metrics that need them are NaN"
        )
        return prob
    samples, weights = truth["samples"], truth["weights"]
    prob.ln_Z = float(truth["ln_z"])
    prob.true_mean = np.reshape(truth["mean"], (1, prob.D))
    prob.true_cov = truth["cov"]
    prob.sampler = lambda n, rng: samples[
        rng.choice(len(samples), size=n, p=weights)
    ]
    prob.sampler_n_eff = int(truth["is_ess"])
    return prob


def missing_truths(configs):
    """Labels of the configs whose target has no ground truth (a real-data
    target whose truth file has not been generated), for harnesses that
    must not record a population without one."""
    missing, seen = [], set()
    for cfg in configs:
        if (cfg.name, cfg.D) in seen:
            continue
        seen.add((cfg.name, cfg.D))
        if make_problem(cfg.name, cfg.D).ln_Z is None:
            missing.append(cfg.label)
    return missing


# Bayesian time-interval reproduction (Acerbi, Wolpert & Vijayakumar 2012),
# one subject of Experiment 3. Parameters: the sensory and motor Weber
# fractions w_s and w_m, the observer's prior mean mu_p and SD sigma_p (in
# seconds), and the lapse rate. The likelihood below is a port of
# benchflow's ``BayesianTiming.log_likelihood``, on the same grids and with
# the same scipy calls.
TIMING_N_S = 101  # points of the interval grid, over [0, 2] s
TIMING_N_X = 401  # points of the measurement grid, one per interval
TIMING_MAX_SD = 5  # half-width of the measurement grid, in sensory SDs
# benchflow's test value, computed there with the original MATLAB code
TIMING_PIN_X = (0.15, 0.15, 0.7875, 0.225, 0.035)
TIMING_PIN_LOGLIK = -4586.122592352263


def _timing(D):
    if D != 5:
        raise ValueError("timing is defined for D = 5 only")
    data = _load_data("timing")
    stim_index = data["stim_index"]
    response = data["response"]
    stimuli = data["stimuli"]
    dr = float(data["bin_size"])  # responses are binned, so dr > 0
    n_trials = response.size
    # The paper's box, Table S2 of the 2020 paper, as exported. The
    # plausible box is kept exactly as published although the posterior mass
    # of the motor Weber fraction w_m (median 0.031) lies below its lower
    # plausible bound of 0.05; benchflow lowered that bound to 0.02. This is
    # the box an imperfect modeller would set, deliberately, and under the
    # prior below it also shapes the prior's taper over that parameter.
    lb, ub = data["lb"], data["ub"]
    plb, pub = data["plb"], data["pub"]

    def loglik_one(x):
        ws, wm, mu_prior, sigma_prior, lambd = x
        srange = np.linspace(0.0, 2.0, TIMING_N_S)[:, None]
        ds = srange[1, 0] - srange[0, 0]
        ll = np.zeros((n_trials, 1))
        for i_stim, mu_s in enumerate(stimuli):
            sigma_s = ws * mu_s
            xrange = np.linspace(
                max(0.0, mu_s - TIMING_MAX_SD * sigma_s),
                mu_s + TIMING_MAX_SD * sigma_s,
                TIMING_N_X,
            )[None, :]
            dx = xrange[0, 1] - xrange[0, 0]
            xpdf = stats.norm.pdf(xrange, mu_s, sigma_s)
            xpdf = xpdf / np.trapezoid(xpdf, dx=dx)
            # the observer's posterior over the interval given each
            # measurement, on the (interval, measurement) grid, and the
            # estimate it produces: the posterior mean, shrunk by the motor
            # noise (the model's optimal reproduction target)
            like = stats.norm.pdf(
                xrange, srange, ws * srange + np.finfo(float).eps
            )
            prior = stats.norm.pdf(srange, mu_prior, sigma_prior)
            post = like * prior
            post = post / np.trapezoid(post, axis=0, dx=ds)
            s_hat = np.trapezoid(post * srange, axis=0, dx=ds) / (1 + wm**2)
            s_hat = s_hat[None, :]
            # probability of each observed response bin under motor noise,
            # marginalized over the measurement
            idx = stim_index == i_stim
            sigma_m = wm * s_hat
            r = response[idx][:, None]
            pr = stats.norm.cdf(r + 0.5 * dr, s_hat, sigma_m) - stats.norm.cdf(
                r - 0.5 * dr, s_hat, sigma_m
            )
            ll[idx] = np.trapezoid(xpdf * pr, axis=1, dx=dx)[:, None]
        # a lapse responds uniformly over the bins of the interval grid
        n_bins = (srange[-1, 0] - srange[0, 0]) / dr
        return float(np.sum(np.log(ll * (1 - lambd) + lambd / n_bins)))

    def loglik_vec(X):
        # about 40 ms per row: the rows are evaluated one by one, which is
        # what VBMC asks for anyway (one point per call)
        return np.array([loglik_one(row) for row in np.atleast_2d(X)])

    return Problem(
        name="timing",
        D=D,
        log_density_vec=_bounded_log_joint(loglik_vec, lb, plb, pub, ub),
        log_likelihood_vec=loglik_vec,
        pins=((TIMING_PIN_X, TIMING_PIN_LOGLIK, "loglik", PIN_TOL),),
        x0=None,
        lb=_row(lb, D),
        ub=_row(ub, D),
        plb=_row(plb, D),
        pub=_row(pub, D),
        notes=(
            "Bayesian time-interval reproduction (Acerbi, Wolpert &"
            f" Vijayakumar 2012), {n_trials} trials of one subject over"
            f" {stimuli.size} intervals, responses binned at {dr:g} s;"
            " the 2020 paper's problem and box, times a"
            " spline-trapezoidal prior with the pivots at the plausible"
            " bounds; likelihood ported from benchflow"
        ),
    )


# Visuo-vestibular unity judgments (Acerbi, Dokka, Angelaki & Ma 2018): the
# observer reports one source when the two noisy measurements differ by less
# than kappa, with a lapse, at three visual coherence levels. Parameters in
# the paper's order, which the likelihood below remaps to the order of
# benchflow's ``Multisensory_6D`` (sigma_vis x 3, sigma_vest, lambda, kappa),
# the implementation it is ported from.
MULTISENSORY_PARAMS = (
    "sigma_vest",
    "sigma_vis_low",
    "sigma_vis_med",
    "sigma_vis_high",
    "kappa",
    "lambda",
)
MULTISENSORY_LB = np.array([0.5, 0.5, 0.5, 0.5, 0.25, 0.005])
MULTISENSORY_UB = np.array([80.0, 80.0, 80.0, 80.0, 180.0, 0.5])
MULTISENSORY_PLB = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.01])
MULTISENSORY_PUB = np.array([40.0, 40.0, 40.0, 40.0, 45.0, 0.2])
# benchflow's stored posterior mode of subject 1 under this prior, in the
# paper's order, and the log joint there: it pins the likelihood and the
# prior together.
MULTISENSORY_S1_PIN_X = (
    7.00615081,
    2.3969857,
    1.37271846,
    8.43661978,
    10.13859551,
    0.02525963,
)
MULTISENSORY_S1_PIN_LOGP = -503.4863062430452
# Regression pin for subject 2, which benchflow stores no value for: the
# log-likelihood of this implementation at the same point on 2026-09-11,
# when a transcription of benchflow's expression agreed with it bit for bit
# on 200 random points of the plausible box.
MULTISENSORY_S2_PIN_LOGLIK = -600.9543713193984
# benchflow's stored log normalizing constant for subject 1 under this
# prior. It is not a gate for the ground-truth generator: three estimators
# sharing no code (Geyer's and importance sampling in the transformed
# space, defensive importance sampling in the original space with an
# effective sample size above 10^5) agree on -502.19 +- 0.01 on
# 2026-09-11, while the log joint at benchflow's stored mode is reproduced
# to 2e-10, so the stored constant is off by about 0.29.
MULTISENSORY_S1_LN_Z_BENCHFLOW = -502.4790984812846


def _multisensory(D, subject):
    if D != 6:
        raise ValueError("multisensory is defined for D = 6 only")
    data = _load_data("multisensory")
    # benchflow's three per-subject cells are taken in their stored order as
    # the low, medium and high coherence levels; the source file does not
    # label them, and the four noise parameters share one set of bounds, so
    # the order only matters for naming the marginals
    trials = [
        (data[f"s{subject}_c{c}_stim"], data[f"s{subject}_c{c}_resp"] == 2)
        for c in (1, 2, 3)
    ]
    n_trials = sum(len(report_two) for _, report_two in trials)

    def loglik_vec(X):
        X = np.atleast_2d(X)
        sigma_vest = X[:, 0:1]
        sigma_vis = X[:, 1:4]
        kappa = X[:, 4:5]
        lambd = X[:, 5:6]
        out = np.zeros(X.shape[0])
        for level, (stim, report_two) in enumerate(trials):
            s_vest, s_vis = stim[:, 0], stim[:, 1]
            sigma_v = sigma_vis[:, level : level + 1]
            # the difference of the two measurements is Gaussian around the
            # difference of the directions with SD sqrt(sigma_vest^2 +
            # sigma_vis^2), written here in units of sigma_vis; p_one is the
            # probability that it falls within -+ kappa, mixed with the lapse
            scale = np.sqrt(1.0 + (sigma_vest / sigma_v) ** 2)
            a_plus = (s_vest - s_vis + kappa) / sigma_v
            a_minus = (s_vest - s_vis - kappa) / sigma_v
            p_one = 0.5 * lambd + (1 - lambd) * (
                stats.norm.cdf(a_plus / scale)
                - stats.norm.cdf(a_minus / scale)
            )
            # the response is coded 2 when the subject reported two sources
            out += stats.bernoulli.logpmf(report_two, p=1 - p_one).sum(1)
        return out

    name = f"multisensory_s{subject}"
    if subject == 1:
        pins = (
            (MULTISENSORY_S1_PIN_X, MULTISENSORY_S1_PIN_LOGP, "logp", PIN_TOL),
        )
    else:
        pins = (
            (
                MULTISENSORY_S1_PIN_X,
                MULTISENSORY_S2_PIN_LOGLIK,
                "loglik",
                PIN_TOL,
            ),
        )
    return Problem(
        name=name,
        D=D,
        log_density_vec=_bounded_log_joint(
            loglik_vec,
            MULTISENSORY_LB,
            MULTISENSORY_PLB,
            MULTISENSORY_PUB,
            MULTISENSORY_UB,
        ),
        log_likelihood_vec=loglik_vec,
        pins=pins,
        x0=None,
        lb=_row(MULTISENSORY_LB, D),
        ub=_row(MULTISENSORY_UB, D),
        plb=_row(MULTISENSORY_PLB, D),
        pub=_row(MULTISENSORY_PUB, D),
        notes=(
            "visuo-vestibular causal inference (Acerbi, Dokka, Angelaki &"
            f" Ma 2018), subject {subject} of the 2020 paper,"
            f" {n_trials} unity judgments over three visual coherence"
            " levels; the 'Fixed' rule with a lapse, parameters"
            f" {', '.join(MULTISENSORY_PARAMS)}, times a"
            " spline-trapezoidal prior with the pivots at the plausible"
            " bounds; likelihood ported from benchflow"
        ),
    )


def _multisensory_s1(D):
    return _multisensory(D, 1)


def _multisensory_s2(D):
    return _multisensory(D, 2)


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
    "gmm": _gmm,
    "ring": _ring,
    "timing": _timing,
    "multisensory_s1": _multisensory_s1,
    "multisensory_s2": _multisensory_s2,
}

TARGET_NAMES = tuple(_REGISTRY)


def make_problem(
    name, D, noise_sd=None, seed=None, options=None, attach_truth=True
):
    """Build a benchmark ``Problem``.

    ``noise_sd`` makes the target noisy (homoskedastic Gaussian noise on the
    log density, returned as the second output; ``specify_target_noise`` is
    set). ``seed`` seeds only that noise stream, through a spawned
    ``SeedSequence`` so it is not the same stream as ``VBMC(seed=seed)``;
    ``None`` means fresh entropy. ``options`` are merged into the problem's
    VBMC options (caller wins). ``attach_truth=False`` leaves a real-data
    target's stored truth file unread (the truth generator builds the
    target it is about to write the truth for).
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown target {name!r}; known: {TARGET_NAMES}")
    prob = _REGISTRY[name](int(D))
    if attach_truth and name in REAL_DATA_TARGETS:
        prob = _attach_truth(prob)
    # Two streams spawned from the run seed: one for the noise, one for the
    # start point; neither is the stream VBMC(seed=seed) uses.
    ss = np.random.SeedSequence(seed) if seed is not None else None
    noise_ss, x0_ss = ss.spawn(2) if ss is not None else (None, None)
    prob.x0 = _draw_x0(prob.plb, prob.pub, np.random.default_rng(x0_ss))
    if noise_sd is not None:
        if not noise_sd > 0:
            raise ValueError("noise_sd must be positive")
        prob.noise_sd = float(noise_sd)
        prob._noise_rng = np.random.default_rng(noise_ss)
        prob.options["specify_target_noise"] = True
    if options:
        prob.options.update(options)
    return prob


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------


def _paper_budget(D):
    """The papers' budget, 50 (D + 2), set explicitly on noisy configs so
    that PyVBMC's x1.5 default for noisy targets does not apply (the 2020
    paper used the same budget for noisy and noiseless problems)."""
    return (("max_fun_evals", 50 * (D + 2)),)


# Noisy configs: rosenbrock_D2 at sigma = 1 reproduces the 2020 paper's
# Fig. 1 toy problem (Rosenbrock + N(0, 3^2), LML -2.27, sigma_obs = 1);
# sigma = 3 isolates the effect of stronger noise on the same target.
# logreg_D5 at sigma = 3 is a bounded problem at the top of the 2020
# benchmark's noise range (1.3-3.2) on the probit-transformed path. The
# student_D8 and lumpy_D10 sigma = 3 entries add broad-tailed and mixture
# structure at higher dimension. The three real-data entries carry the 2020
# paper's own noise levels, the standard deviation of its IBS estimator at
# the MAP: 2.2 for timing and 1.3 for both multisensory subjects. Of the
# three, multisensory subject 1 is the hardest posterior (broad skewed
# visual-noise marginals against their lower bound, correlations of 0.8),
# and it is the one that also runs noiseless. Every noisy entry uses the
# paper budget.
# The budget-exhausting configuration: 750 evaluations at D = 15 with early
# termination disabled, the one run that spends long in the optimize-only
# regime (a single GP hyperparameter sample from N >= 350, K around 30, the
# occasional full refit at N >= 500). About 13 minutes per seed; in the
# golden population it is run with 10 seeds, for regime coverage rather than
# statistics (`golden_trace.py run --only cigar_D15_exhaust --seeds 0-9`).
_EXHAUST = Config(
    "cigar",
    15,
    options=(
        ("max_fun_evals", 750),
        ("tol_stable_excpt_frac", -(10**6)),
    ),
    tag="exhaust",
)

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
        Config("rosenbrock", 2, noise_sd=1.0, options=_paper_budget(2)),
        Config("logreg", 5, noise_sd=3.0, options=_paper_budget(5)),
        Config("lumpy", 10),
        Config("banana", 10),
        _EXHAUST,
    ],
    "golden": [
        Config("normal", 5),
        Config("corr", 5),
        Config("halfnormal", 2),
        Config("rosenbrock", 2),
        Config("banana", 2),
        Config("banana", 6),
        Config("banana", 10),
        Config("cigar", 4),
        Config("lumpy", 4),
        Config("lumpy", 10),
        Config("student", 4),
        Config("logreg", 5),
        Config("rosenbrock", 2, noise_sd=1.0, options=_paper_budget(2)),
        Config("rosenbrock", 2, noise_sd=3.0, options=_paper_budget(2)),
        Config("logreg", 5, noise_sd=3.0, options=_paper_budget(5)),
        # The two hard shapes (ill-conditioned, heavy-tailed) at an
        # intermediate-high dimension; the other shapes cover D = 6 and 10.
        Config("cigar", 8),
        Config("student", 8),
        Config("student", 8, noise_sd=3.0, options=_paper_budget(8)),
        Config("lumpy", 10, noise_sd=3.0, options=_paper_budget(10)),
        # The real-data problems of the 2020 paper, at its noise levels.
        Config("multisensory_s1", 6),
        Config("timing", 5, noise_sd=2.2, options=_paper_budget(5)),
        Config("multisensory_s1", 6, noise_sd=1.3, options=_paper_budget(6)),
        Config("multisensory_s2", 6, noise_sd=1.3, options=_paper_budget(6)),
        _EXHAUST,
    ],
    # The pool inputs of the S-VBMC run-pool campaign
    # (dev/plans/svbmc-benchmark-campaign.md): independent VBMC runs whose
    # finished posteriors are stacked, in the paper's conditions plus a
    # noiseless control. They run at PyVBMC's default evaluation budget,
    # 75 (D + 2) on a noisy target, which is the S-VBMC paper's convention
    # and what users get, so no entry pins the 2020 paper's budget; the
    # `svbmc` tag gives them their own labels, so that `find_config` cannot
    # return a golden entry, which does pin that budget, for the same
    # target, dimension and noise level. Six noisy conditions (the
    # multisensory model at the paper's noise 3 and at the 2020 paper's
    # IBS level 1.3, noisy Rosenbrock, the paper's GMM and ring, and the
    # heavy-tailed Student target at D = 8) and two noiseless controls.
    "svbmc_pool": [
        Config("multisensory_s1", 6, noise_sd=3.0, tag="svbmc"),
        Config("multisensory_s1", 6, noise_sd=1.3, tag="svbmc"),
        Config("rosenbrock", 2, noise_sd=3.0, tag="svbmc"),
        Config("gmm", 2, noise_sd=3.0, tag="svbmc"),
        Config("ring", 2, noise_sd=3.0, tag="svbmc"),
        Config("student", 8, noise_sd=3.0, tag="svbmc"),
        Config("gmm", 2, tag="svbmc"),
        Config("multisensory_s1", 6, tag="svbmc"),
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


def posterior_moments(
    vp, n_mc=DIAG_MC_SAMPLES, seed=DIAG_SEED, force_mc=False
):
    """Original-space posterior mean ``(1, D)`` and covariance ``(D, D)``.

    Exact through the transformer's affine map when every coordinate is
    unbounded (``x = mu + A u``, ``A = diag(delta) R_mat diag(scale)``,
    including any rotoscale warp); otherwise Monte Carlo on a deep copy with
    a dedicated generator, so the run's stream is never consumed. Returns
    ``(mean, cov, method)``.
    """
    pt = vp.parameter_transformer
    if np.all(pt.type == 0) and not force_mc:
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


DIAG_TV_SAMPLES = 100_000


def metrics(problem, vp, elbo):
    """``elbo_err``, ``gskl``, ``mmtv`` and ``rmse`` against the problem's
    truth (NaN when unknown), plus the posterior moments used.

    ``elbo_err`` (|ELBO - ln Z|), ``gskl`` (Gaussianized symmetrized KL to
    the true mean and covariance) and ``mmtv`` (mean marginal total variation
    distance to exact samples, via ``VariationalPosterior.mtv``) are the
    metrics of the VBMC papers; ``rmse`` of the posterior mean is what the
    end-to-end tests assert and is kept for that reason only. All draws use
    dedicated generators on a copy of the VP, so the run's stream is never
    consumed.
    """
    from pyvbmc.stats import kl_div_mvn

    mean, cov, method = posterior_moments(vp)
    out = {
        "elbo_err": float("nan"),
        "gskl": float("nan"),
        "mmtv": float("nan"),
        "rmse": float("nan"),
        "moment_method": method,
        "post_mean": mean,
        "post_cov": cov,
    }
    if problem.sampler is not None:
        ref = problem.sampler(
            DIAG_TV_SAMPLES, np.random.default_rng(DIAG_SEED + 1)
        )
        vp2 = copy.deepcopy(vp)
        vp2.rng = np.random.default_rng(DIAG_SEED + 2)
        out["mmtv"] = float(np.mean(vp2.mtv(samples=ref, N=DIAG_TV_SAMPLES)))
    if problem.ln_Z is not None:
        out["elbo_err"] = float(abs(elbo - problem.ln_Z))
    if problem.true_mean is not None:
        out["rmse"] = float(np.sqrt(np.mean((mean - problem.true_mean) ** 2)))
    if problem.true_cov is not None and problem.true_mean is not None:
        kl = kl_div_mvn(mean, cov, problem.true_mean, problem.true_cov)
        out["gskl"] = float(0.5 * np.sum(kl))
    return out


def _normalized_kde(x, nkde, lower, upper):
    """A 1-D kernel density estimate normalized over its own grid."""
    from pyvbmc.stats import kde_1d

    yy, mesh, _ = kde_1d(x, nkde, lower, upper)
    return yy / (np.trapezoid(yy) * (mesh[1] - mesh[0])), mesh


def _marginal_tv(xx1, xx2, bounds1, bounds2, nkde=2**13, n_grid=100_000):
    """Per-dimension total variation distance between two sets of samples.

    The procedure of ``VariationalPosterior.mtv``, applied to two sample
    sets rather than to a posterior and a sample set: in each dimension a
    ``nkde``-point kernel density estimate over the range of that set's own
    draws widened by a tenth and clipped to its bounds, each estimate
    normalized by its trapezoid integral, then half the integral of the
    absolute difference of the two cubic interpolants, taken piecewise
    between the sorted endpoints of the two grids so that the region where
    only one density is defined counts as well.
    """
    from scipy.interpolate import interp1d

    def limits(xx, bounds):
        low, high = np.amin(xx, axis=0), np.amax(xx, axis=0)
        span = high - low
        lb, ub = bounds
        return (
            np.maximum(low - span / 10, np.ravel(lb)),
            np.minimum(high + span / 10, np.ravel(ub)),
        )

    lb1, ub1 = limits(xx1, bounds1)
    lb2, ub2 = limits(xx2, bounds2)
    tv = np.zeros(xx1.shape[1])
    for d in range(xx1.shape[1]):
        yy1, mesh1 = _normalized_kde(xx1[:, d], nkde, lb1[d], ub1[d])
        yy2, mesh2 = _normalized_kde(xx2[:, d], nkde, lb2[d], ub2[d])
        curves = [
            interp1d(
                mesh,
                yy,
                kind="cubic",
                fill_value=np.array([0]),
                bounds_error=False,
            )
            for mesh, yy in ((mesh1, yy1), (mesh2, yy2))
        ]
        edges = np.sort([mesh1[0], mesh1[-1], mesh2[0], mesh2[-1]])
        for j in range(3):
            grid = np.linspace(edges[j], edges[j + 1], num=int(n_grid))
            difference = np.abs(curves[0](grid) - curves[1](grid))
            tv[d] += 0.5 * np.trapezoid(difference) * (grid[1] - grid[0])
    return tv


def sample_metrics(
    problem, samples, elbos, reference=None, seed=DIAG_SEED + 1
):
    """Metrics of a posterior that is only available as draws.

    The counterpart of :func:`metrics` for a posterior with no closed-form
    moments and no ``log_pdf``, such as a stacked S-VBMC mixture. The
    moments come from ``samples``; ``gskl`` is the house convention (half
    the sum of the two Kullback-Leibler directions between the Gaussians
    with those moments and the truth, no ``1/D`` factor) and
    ``gskl_normalized`` the S-VBMC paper's ``1/(2D) sum KL``; ``mmtv`` is
    the mean over dimensions of the marginal total variation distance to
    exact draws from the problem, computed as
    ``VariationalPosterior.mtv`` computes it: the density estimate of the
    posterior draws is taken over a range clipped to the problem's bounds,
    that of the exact draws over an unclipped one. One
    ``elbo_err_<name>`` is returned per entry of the ``elbos`` mapping.
    Entries whose truth is unknown are NaN.

    ``reference`` are exact draws to compare against; without them
    ``DIAG_TV_SAMPLES`` are drawn from ``problem.sampler`` with a dedicated
    generator seeded by ``seed``, so a caller that scores many posteriors
    against one problem can draw the reference once. The default ``seed``
    is the one :func:`metrics` uses for the same draws, so that posteriors
    scored by either function are compared against one reference.
    """
    from pyvbmc.stats import kl_div_mvn

    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    mean = np.mean(samples, axis=0, keepdims=True)
    cov = np.cov(samples, rowvar=False).reshape(problem.D, problem.D)
    out = {
        "n_samples": int(samples.shape[0]),
        "post_mean": mean,
        "post_cov": cov,
        "mmtv": float("nan"),
        "gskl": float("nan"),
        "gskl_normalized": float("nan"),
    }
    for name, elbo in dict(elbos).items():
        out[f"elbo_err_{name}"] = (
            float("nan")
            if problem.ln_Z is None or elbo is None
            else float(abs(float(elbo) - problem.ln_Z))
        )
    if reference is None and problem.sampler is not None:
        reference = problem.sampler(
            DIAG_TV_SAMPLES, np.random.default_rng(seed)
        )
    if reference is not None:
        unbounded = (
            np.full((1, problem.D), -np.inf),
            np.full((1, problem.D), np.inf),
        )
        out["mmtv"] = float(
            np.mean(
                _marginal_tv(
                    samples,
                    np.atleast_2d(np.asarray(reference, dtype=float)),
                    (problem.lb, problem.ub),
                    unbounded,
                )
            )
        )
    if problem.true_cov is not None and problem.true_mean is not None:
        kl = kl_div_mvn(mean, cov, problem.true_mean, problem.true_cov)
        out["gskl"] = float(0.5 * np.sum(kl))
        out["gskl_normalized"] = out["gskl"] / problem.D
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
    # (b) moments vs exact samples (for a stored importance-weighted
    # population, whose moments were computed from the same draws, a
    # consistency check of the truth file: it cannot fail unless the file
    # is inconsistent)
    if prob.sampler is not None and prob.true_mean is not None:
        S = prob.sampler(n_draws, rng)
        m = S.mean(0)
        c = np.cov(S.T)
        sd = np.sqrt(np.diag(prob.true_cov))
        n_eff = n_draws
        if prob.sampler_n_eff is not None:
            # resampling from a stored population: no more independent draws
            # than the population holds
            n_eff = min(n_draws, prob.sampler_n_eff)
            res["sampler_n_eff"] = float(n_eff)
        elif prob.name == "logreg":
            # importance *re*sampling: the draws are not independent; the
            # effective size is that of the cached weighted population
            _, wn = _LOGREG_IS_CACHE["pop"]
            n_eff = min(n_draws, 1.0 / np.sum(wn**2))
            res["sampler_ess"] = float(n_eff)
        z = (m - prob.true_mean.ravel()) / (sd / np.sqrt(n_eff))
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
    if prob.name == "ring":
        # the quadrature that produced ln Z against the closed form of the
        # same integral over the whole line
        d = abs(prob.ln_Z - _RING_CACHE["ln_Z_closed"])
        res["ln_Z_closed_form_diff"] = float(d)
        if d > 1e-9:
            res["ok"] = False
            res["msgs"].append(f"ln Z quadrature off by {d:.2e}")
        # the port against the pinned upstream source: the density here
        # drops upstream's ``log r`` term, so ``pins`` cannot compare with
        # that source. Adding the term back at the ``RING_PINS`` points
        # must reproduce its values there.
        Xp = np.array([x for x, _ in RING_PINS], dtype=float)
        r = np.linalg.norm(Xp - np.asarray(RING_CENTER, dtype=float), axis=-1)
        log_r = np.log(np.maximum(r, np.finfo(float).tiny))
        d = float(
            np.max(
                np.abs(
                    prob.log_density_vec(Xp)
                    + log_r
                    - np.asarray(RING_UPSTREAM_PINS)
                )
            )
        )
        res["upstream_pin_diff"] = d
        if d > SVBMC_PIN_TOL:
            res["ok"] = False
            res["msgs"].append(
                f"upstream ring values differ by {d:.2e}"
                f" (tolerance {SVBMC_PIN_TOL:.0e})"
            )
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
    # (d) pinned values from the reference implementation
    for i, (x, expected, which, tol) in enumerate(prob.pins):
        X = np.reshape(np.asarray(x, dtype=float), (1, D))
        f = prob.log_density_vec
        if which == "loglik":
            f = prob.log_likelihood_vec
        elif which != "logp":
            raise ValueError(f"unknown pin kind {which!r}")
        d = abs(float(f(X)[0]) - expected)
        res[f"pin{i}_{which}_diff"] = float(d)
        if d > tol:
            res["ok"] = False
            res["msgs"].append(
                f"pinned {which} differs by {d:.2e} (tolerance {tol:.0e})"
            )
    return res


def _selected(cfg, only):
    """Whether a config passes an ``--only`` filter: ``None`` for everything,
    otherwise a set (not a string: ``in`` on a string matches substrings)
    of target names or config labels."""
    return only is None or cfg.name in only or cfg.label in only


def run_check(configs, only=None):
    seen = set()
    all_ok = True
    for cfg in configs:
        key = (cfg.name, cfg.D)
        if key in seen or not _selected(cfg, only):
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
    if _selected(Config("logreg", 5), only):
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


def run_smoke(configs, seed=0, only=None):
    import psutil

    from pyvbmc import VBMC

    proc = psutil.Process()
    all_ok = True
    for cfg in configs:
        if not _selected(cfg, only):
            continue
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
            if prob.all_unbounded:
                # the exact affine route must agree with Monte Carlo
                m2, c2, _ = posterior_moments(vp, force_mc=True)
                sd = np.sqrt(np.diag(met["post_cov"]))
                dm = np.max(np.abs(met["post_mean"] - m2) / sd)
                dc = np.max(np.abs(met["post_cov"] - c2)) / np.max(
                    np.abs(met["post_cov"])
                )
                ok &= dm < 0.05 and dc < 0.05
                met[
                    "moment_method"
                ] += f"; mc diff mean {dm:.3f} sd, cov {dc:.3f}"
            msg = (
                f"iters={results['iterations'] + 1} evals={results['func_count']}"
                f" elbo={results['elbo']:.3f} elbo_err={met['elbo_err']:.3f}"
                f" gskl={met['gskl']:.3f} mmtv={met['mmtv']:.3f}"
                f" rmse={met['rmse']:.3f} ({met['moment_method']})"
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
    ap.add_argument(
        "--suite", default=None, help="smoke|profile|golden|svbmc_pool|all"
    )
    ap.add_argument(
        "--only",
        default=None,
        help="restrict --check / --smoke to a comma-separated list of"
        " target names or config labels",
    )
    args = ap.parse_args(argv)
    only = None
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}
        known = {c.label for c in suite_configs("all")} | set(TARGET_NAMES)
        unknown = sorted(only - known)
        if unknown:
            ap.error(f"unknown --only entries: {', '.join(unknown)}")
        for flag, suite in (
            (args.check, args.suite or "all"),
            (args.smoke, args.suite or "smoke"),
        ):
            if flag and not any(
                _selected(c, only) for c in suite_configs(suite)
            ):
                ap.error(f"--only selects no config of suite {suite!r}")
    if args.list or not (args.check or args.smoke):
        for s, cfgs in SUITES.items():
            print(f"{s}:")
            for c in cfgs:
                print(f"    {c.label:28s} options={c.options_dict()}")
        return 0
    ok = True
    if args.check:
        ok &= run_check(suite_configs(args.suite or "all"), only=only)
    if args.smoke:
        ok &= run_smoke(suite_configs(args.suite or "smoke"), only=only)
    print("[benchmark_targets]", "all ok" if ok else "FAILURES", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
