"""Finite-difference checks of the hand-derived gradients in
``pyvbmc.vbmc.variational_optimization``.

Stage 0 of the modernization plan (see
``dev/2026-09-02-modernization-discussion.md``, section 10): before any of
these gradients is touched or replaced by autodiff, each one is checked
against numerical differentiation, not only against stored MATLAB arrays.

The GP fixture (``X.txt``, ``y.txt``, ``hyp.txt`` with 8 hyperparameter
samples, ``mu.txt``) is shared with ``test_variational_optimization.py``.

Parameterization note: the checks use the raw parameterization
``theta = (mu, ln sigma, ln lambd, eta)`` with ``jacobian_flag=True``, which is
what ``_neg_elcbo`` uses in production; one check takes ``jacobian_flag=False``
and differentiates with respect to ``(mu, sigma, lambd, w)`` directly (until
2026-09-04 that path returned only the ``mu`` block of the gradient, devlog
section 9). ``VariationalPosterior.set_parameters`` renormalizes ``lambd`` to
``||lambd|| = sqrt(D)`` and rescales ``sigma`` to compensate; the objective is
invariant along that ray, so finite differences in raw coordinates agree with
the analytic gradient evaluated at the renormalized point.
"""

from pathlib import Path

import gpyreg as gpr
import numpy as np
import pytest

from pyvbmc.testing import check_grad
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import (
    _gp_log_joint,
    _neg_elcbo,
    _soft_bound_loss,
    _vp_bound_loss,
)

BASE_PATH = Path(__file__).parent
D, K = 2, 2
OPTIONS = {
    "tol_con_loss": 0.01,
    "tol_weight": 1e-2,
    "weight_penalty": 0.1,
    "tol_length": 1e-6,
}


@pytest.fixture(autouse=True)
def _restore_global_rng():
    """``VariationalPosterior.__init__`` draws from the global ``np.random``
    state and these tests build a VP per objective evaluation; leave the
    stream as we found it for the rest of the suite."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _load(name):
    return np.loadtxt(open(BASE_PATH.joinpath(name), "rb"), delimiter=",")


def _fixture_gp():
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    X = _load("X.txt")
    y = _load("y.txt").reshape((-1, 1))
    hyp = _load("hyp.txt")
    gp.update(X_new=X, y_new=y, hyp=hyp)
    return gp, X


def _raw_theta0(seed):
    """Raw parameter vector (mu, ln sigma, ln lambd, eta), mu near the data.

    Uses a local Generator so the point does not depend on the global
    ``np.random`` stream.
    """
    rng = np.random.default_rng(seed)
    mu = _load("mu.txt") + 0.1 * rng.standard_normal((D, K))
    ln_sigma = np.log(0.5 + 0.5 * rng.random(K))
    ln_lambd = np.log(0.7 + 0.6 * rng.random(D))
    eta = 0.5 * rng.standard_normal(K)
    return np.concatenate([mu.ravel(order="F"), ln_sigma, ln_lambd, eta])


def _vp_from_raw_theta(theta):
    """Build a VP from raw theta without the gauge renormalization that
    ``set_parameters`` applies, so the map theta -> vp is a plain bijection."""
    vp = VariationalPosterior(D, K)
    vp.mu = theta[: D * K].reshape((D, K), order="F")
    vp.sigma = np.exp(theta[D * K : D * K + K]).reshape(1, -1)
    vp.lambd = np.exp(theta[D * K + K : D * K + K + D]).reshape(-1, 1)
    eta = theta[-K:] - np.max(theta[-K:])
    vp.eta = eta.reshape(1, -1)
    vp.w = (np.exp(eta) / np.sum(np.exp(eta))).reshape(1, -1)
    return vp


def test_gp_log_joint_grad_fd():
    """dG of the expected log joint wrt (mu, ln sigma, ln lambd, eta),
    averaged over the 8 GP hyperparameter samples."""
    gp, _ = _fixture_gp()

    def f(theta):
        G, *_ = _gp_log_joint(_vp_from_raw_theta(theta), gp, False)
        return G

    def grad(theta):
        _, dG, *_ = _gp_log_joint(_vp_from_raw_theta(theta), gp, True)
        return dG

    theta0 = _raw_theta0(seed=0)
    assert grad(theta0).shape == theta0.shape
    assert check_grad(f, grad, theta0, rtol=1e-5, atol=1e-8)


def test_gp_log_joint_grad_fd_single_sample():
    """Same check with a single hyperparameter sample (``Ns == 1`` path)."""
    gp, _ = _fixture_gp()
    gp.update(hyp=_load("hyp.txt")[0:1])

    def f(theta):
        G, *_ = _gp_log_joint(_vp_from_raw_theta(theta), gp, False)
        return G

    def grad(theta):
        _, dG, *_ = _gp_log_joint(_vp_from_raw_theta(theta), gp, True)
        return dG

    theta0 = _raw_theta0(seed=3)
    assert grad(theta0).shape == theta0.shape
    assert check_grad(f, grad, theta0, rtol=1e-5, atol=1e-8)


def _random_gp(D, N, Ns, seed):
    """A small GP with ``Ns`` hyperparameter samples at any ``D``, from a
    local Generator (the shared fixture is ``D = 2`` only)."""
    rng = np.random.default_rng(seed)
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    X = rng.uniform(-2.0, 2.0, size=(N, D))
    y = -0.5 * np.sum(X**2, axis=1) + 0.1 * rng.standard_normal(N)
    cov_N = gp.covariance.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()
    hyp = np.zeros((Ns, cov_N + noise_N + gp.mean.hyperparameter_count(D)))
    hyp[:, :D] = rng.normal(0.0, 0.3, size=(Ns, D))  # ln ell
    hyp[:, D] = rng.normal(0.0, 0.3, size=Ns)  # ln sf
    hyp[:, cov_N : cov_N + noise_N] = -2.0  # ln sn
    hyp[:, cov_N + noise_N] = rng.normal(0.0, 0.5, size=Ns)  # m0
    hyp[:, cov_N + noise_N + 1 : cov_N + noise_N + 1 + D] = rng.normal(
        0.0, 0.3, size=(Ns, D)
    )  # xm
    hyp[:, cov_N + noise_N + 1 + D :] = rng.normal(
        0.3, 0.3, size=(Ns, D)
    )  # ln omega
    gp.update(X_new=X, y_new=y.reshape(-1, 1), hyp=hyp)
    return gp


def test_gp_log_joint_grad_fd_D_ne_K():
    """``D = 3, K = 2``: a wrong packing order of the ``mu`` block
    (``d`` fastest) or a transposed sigma/lambd block is a scramble here,
    not a symmetry."""
    D3, K2 = 3, 2
    gp = _random_gp(D3, N=25, Ns=4, seed=5)
    rng = np.random.default_rng(6)

    def vp_from(theta):
        vp = VariationalPosterior(D3, K2)
        vp.mu = theta[: D3 * K2].reshape((D3, K2), order="F")
        vp.sigma = np.exp(theta[D3 * K2 : D3 * K2 + K2]).reshape(1, -1)
        vp.lambd = np.exp(theta[D3 * K2 + K2 : D3 * K2 + K2 + D3]).reshape(
            -1, 1
        )
        eta = theta[-K2:] - np.max(theta[-K2:])
        vp.eta = eta.reshape(1, -1)
        vp.w = (np.exp(eta) / np.sum(np.exp(eta))).reshape(1, -1)
        return vp

    theta0 = np.concatenate(
        [
            rng.uniform(-1.0, 1.0, size=D3 * K2),
            np.log(0.5 + 0.5 * rng.random(K2)),
            np.log(0.7 + 0.6 * rng.random(D3)),
            0.5 * rng.standard_normal(K2),
        ]
    )

    def f(theta):
        return _gp_log_joint(vp_from(theta), gp, False)[0]

    def grad(theta):
        return _gp_log_joint(vp_from(theta), gp, True)[1]

    assert grad(theta0).shape == theta0.shape
    assert check_grad(f, grad, theta0, rtol=1e-5, atol=1e-8)


def test_gp_log_joint_grad_fd_no_jacobian():
    """``jacobian_flag=False``: the gradient with respect to the untransformed
    ``(mu, sigma, lambd, w)``, with ``w`` free (the function uses the weights
    as given, so ``dG/dw_k = I_k``). All four blocks must be present."""
    gp, _ = _fixture_gp()
    rng = np.random.default_rng(4)

    def vp_from(theta):
        vp = VariationalPosterior(D, K)
        vp.mu = theta[: D * K].reshape((D, K), order="F")
        vp.sigma = theta[D * K : D * K + K].reshape(1, -1)
        vp.lambd = theta[D * K + K : D * K + K + D].reshape(-1, 1)
        vp.w = theta[-K:].reshape(1, -1)
        return vp

    theta0 = np.concatenate(
        [
            (_load("mu.txt") + 0.1 * rng.standard_normal((D, K))).ravel(
                order="F"
            ),
            0.5 + 0.5 * rng.random(K),
            0.7 + 0.6 * rng.random(D),
            0.3 + 0.5 * rng.random(K),
        ]
    )

    def f(theta):
        return _gp_log_joint(vp_from(theta), gp, False, True, False)[0]

    def grad(theta):
        return _gp_log_joint(vp_from(theta), gp, True, True, False)[1]

    dG = grad(theta0)
    assert dG.shape == (D * K + K + D + K,)
    # The w block is the per-component expected log joint.
    _, _, _, _, _, I_sk, _ = _gp_log_joint(
        vp_from(theta0), gp, False, True, False, False, True
    )
    assert np.allclose(dG[-K:], I_sk.mean(axis=0))
    assert check_grad(f, grad, theta0, rtol=1e-5, atol=1e-8)


def test_neg_elcbo_grad_fd_deterministic_entropy():
    """dF of the negative ELBO with the entropy lower bound (``Ns == 0``),
    with and without the soft-bound penalty, at a point where the penalty is
    active.

    ``_neg_elcbo`` subtracts ``max(eta)`` from the eta block of ``theta`` in
    place before calling ``_vp_bound_loss``, so the weight upper bound
    (``ub = 0``) can never fire through this entry point; the violated bounds
    here are on mu and on ln_scale. The wrappers pass copies so that in-place
    shift does not leak into the finite-difference abscissae.
    """
    gp, X = _fixture_gp()
    vp0 = VariationalPosterior(D, K)
    vp0.mu = _load("mu.txt")
    theta_bnd = vp0.get_bounds(X, OPTIONS, K)

    theta0 = _raw_theta0(seed=1)
    theta_viol = theta0.copy()
    theta_viol[0] = theta_bnd["lb"][0] - 0.7  # mu below its bound
    theta_viol[D * K] = 1.0  # ln sigma_1: ln_scale[:, 0] above its bound

    for bnd, theta in ((None, theta0), (theta_bnd, theta_viol)):

        def f(th):
            vp = VariationalPosterior(D, K)
            return _neg_elcbo(th.copy(), gp, vp, 0.0, 0, False, False, bnd)[0]

        def grad(th):
            vp = VariationalPosterior(D, K)
            return _neg_elcbo(th.copy(), gp, vp, 0.0, 0, True, False, bnd)[1]

        if bnd is not None:
            L, dL = _vp_bound_loss(
                VariationalPosterior(D, K), theta, bnd, bnd["tol_con"]
            )
            assert L > 0.0, "soft-bound penalty should be active"
            assert np.any(dL[D * K : D * K + K + D] != 0.0)
        assert grad(theta).shape == theta.shape
        assert check_grad(f, grad, theta, rtol=1e-5, atol=1e-8)


def test_neg_elcbo_grad_fd_mc_entropy():
    """dF with the Monte Carlo entropy (``Ns > 0``), using common random
    numbers so the objective is a deterministic function of theta.

    The tolerance is loose on purpose: the reparameterization gradient in
    ``entmc_vbmc`` is an unbiased estimator of the true gradient, not the
    exact derivative of the sample-based value estimate, so the two agree
    only up to Monte Carlo error (relative ~1e-3 at ``Ns = 1e4``). This
    still catches wrong signs, wrong Jacobians, or a missing block.
    """
    gp, _ = _fixture_gp()
    Ns = int(1e4)

    # A fresh VP with the same generator seed per call gives the value and
    # gradient evaluations common random numbers.
    def f(th):
        return _neg_elcbo(
            th, gp, VariationalPosterior(D, K, rng=7), 0.0, Ns, False, False
        )[0]

    def grad(th):
        return _neg_elcbo(
            th, gp, VariationalPosterior(D, K, rng=7), 0.0, Ns, True, False
        )[1]

    theta0 = _raw_theta0(seed=2)
    assert check_grad(f, grad, theta0, rtol=1e-2, atol=1e-2)


def test_vp_bound_loss_grad_fd():
    """dL of the soft-bound penalty, including the fold of the ``ln_scale``
    gradient back onto ``ln sigma`` and ``ln lambd``."""
    vp = VariationalPosterior(D, K)
    vp.mu = _load("mu.txt")
    theta_bnd = vp.get_bounds(_load("X.txt"), OPTIONS, K)

    theta0 = vp.get_parameters()
    theta0[0] = theta_bnd["lb"][0] - 0.7  # mu below its bound
    theta0[D * K] = 1.0  # ln sigma_1: pushes ln_scale[:, 0] above its bound
    theta0[-1] = theta_bnd["ub"][-1] + 1.0  # eta above its bound

    L, dL = _vp_bound_loss(vp, theta0, theta_bnd, tol_con=0.01)
    assert L > 0.0
    assert dL.shape == theta0.shape
    assert np.any(dL[D * K : D * K + K] != 0.0)  # sigma block active
    assert np.any(dL[D * K + K : D * K + K + D] != 0.0)  # lambd block active

    def f(th):
        return _vp_bound_loss(
            vp, th, theta_bnd, tol_con=0.01, compute_grad=False
        )

    def grad(th):
        return _vp_bound_loss(vp, th, theta_bnd, tol_con=0.01)[1]

    # atol covers the platform-dependent finite-difference noise on the
    # many coordinates where the penalty is exactly flat (see
    # test_soft_bound_loss_grad_fd).
    assert check_grad(f, grad, theta0, rtol=1e-4, atol=1e-5)


def test_soft_bound_loss_grad_fd():
    """Two coordinates outside the soft bounds, one inside.

    The loss is exactly flat in the inside coordinate, where
    ``scipy.differentiate.jacobian`` does not converge and returns roundoff
    noise of order ``eps * L / h`` that differs between platforms (6.5e-7 on
    Windows, above 1e-6 on Linux for ``L ~ 4e4``). Small bound excesses keep
    ``L`` small, and ``atol = 1e-5`` sits well above that noise and far below
    the gradients on the violated coordinates (hundreds).
    """
    slb = np.full((3,), -10.0)
    sub = np.full((3,), 10.0)
    x0 = np.array([10.5, -10.7, 3.0])

    def f(x):
        return _soft_bound_loss(x, slb, sub)

    def grad(x):
        return _soft_bound_loss(x, slb, sub, compute_grad=True)[1]

    L, dL = _soft_bound_loss(x0, slb, sub, compute_grad=True)
    assert L > 0.0
    assert dL[0] > 0.0 and dL[1] < 0.0 and dL[2] == 0.0
    assert check_grad(f, grad, x0, rtol=1e-4, atol=1e-5)


def test_vp_bound_loss_grad_fd_D_ne_K():
    """As ``test_vp_bound_loss_grad_fd`` but with ``D != K``, so a wrong
    reshape order of the ln_scale block is a genuine scramble rather than a
    transpose, and the mu block packing is pinned too."""
    D3, K2 = 3, 2
    rng = np.random.default_rng(11)
    X = rng.uniform(-1.0, 1.0, size=(12, D3))
    vp = VariationalPosterior(D3, K2)
    vp.mu = rng.uniform(-0.5, 0.5, size=(D3, K2))
    theta_bnd = vp.get_bounds(X, OPTIONS, K2)

    theta0 = vp.get_parameters()
    theta0[1] = theta_bnd["lb"][1] - 0.5  # mu[1, 0] below its bound
    theta0[D3 * K2 + 1] = 1.0  # ln sigma_2: ln_scale[:, 1] above its bound
    theta0[D3 * K2 + K2] = 1.5  # ln lambd_1: makes ln_scale[0, 1] the worst
    theta0[-1] = theta_bnd["ub"][-1] + 1.0  # eta above its bound

    L, dL = _vp_bound_loss(vp, theta0, theta_bnd, tol_con=0.01)
    assert L > 0.0
    assert dL.shape == theta0.shape
    # Column 1 of ln_scale is out of bounds for every d, so sigma_2 and every
    # lambd_d receive a nonzero penalty gradient; sigma_1 does not.
    assert dL[D3 * K2] == 0.0 and dL[D3 * K2 + 1] != 0.0
    assert np.all(dL[D3 * K2 + K2 : D3 * K2 + K2 + D3] != 0.0)

    def f(th):
        return _vp_bound_loss(
            vp, th, theta_bnd, tol_con=0.01, compute_grad=False
        )

    def grad(th):
        return _vp_bound_loss(vp, th, theta_bnd, tol_con=0.01)[1]

    assert check_grad(f, grad, theta0, rtol=1e-4, atol=1e-5)
