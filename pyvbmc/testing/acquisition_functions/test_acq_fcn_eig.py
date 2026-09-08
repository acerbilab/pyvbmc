"""``AcqFcnEIG``: the integrated posterior covariance against grid
quadrature, the scalar information gain against its definition from the
joint Gaussian of the expected log joint and a new observation, and the
per-component variant against the scalar one at ``K = 1``."""

import gpyreg as gpr
import numpy as np
import pytest

from pyvbmc.acquisition_functions import AcqFcnEIG
from pyvbmc.acquisition_functions.acq_fcn_eig import (
    integrated_posterior_covariance,
)
from pyvbmc.acquisition_functions.utilities import string_to_acq
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import _gp_log_joint


def _gp(D, ns_gp, seed=0, N=12, noisy=True):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.5, 2.5, size=(N, D))
    y = -0.5 * np.sum(X**2, axis=1, keepdims=True) + 0.3 * rng.normal(
        size=(N, 1)
    )
    s2 = rng.uniform(0.2, 1.0, size=(N, 1)) if noisy else None
    base = np.concatenate(
        [
            0.1 * np.arange(D),  # log ell
            [0.4],  # log sf
            [-1.5],  # log base noise sd
            [-(D / 2) * np.log(2 * np.pi)],  # mean function maximum
            np.zeros(D),  # mean function location
            0.3 * np.ones(D),  # log mean function scale
        ]
    )
    hyp = np.array([base, base + 0.1 * rng.normal(size=base.size)][:ns_gp])
    gp = gpr.GP(
        D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=noisy
        ),
    )
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp)
    return gp


def _vp(D, K, seed=0):
    rng = np.random.default_rng(seed)
    vp = VariationalPosterior(D, K)
    vp.mu = rng.uniform(-1, 1, size=(D, K))
    vp.sigma = rng.uniform(0.5, 1.5, size=(1, K))
    vp.lambd = rng.uniform(0.7, 1.3, size=(D, 1))
    w = rng.uniform(0.5, 1.5, size=(1, K))
    vp.w = w / np.sum(w)
    return vp


def _prepare(gp, vp, s2=None):
    """What ``active_sample`` stores before evaluating the acquisition."""
    D = gp.D
    Ns = len(gp.posteriors)
    optim_state = {
        "integer_vars": None,
        "lb_eps_orig": -np.inf,
        "ub_eps_orig": np.inf,
    }
    ln_ell = np.column_stack([p.hyp[:D] for p in gp.posteriors])
    optim_state["gp_length_scale"] = np.exp(ln_ell.mean(1))
    gp.temporary_data["X_rescaled"] = gp.X / optim_state["gp_length_scale"]
    cov_N = gp.covariance.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()
    sn2_new = np.column_stack(
        [
            np.broadcast_to(
                gp.noise.compute(
                    p.hyp[cov_N : cov_N + noise_N], gp.X, gp.y, gp.s2
                ).reshape(-1),
                (gp.X.shape[0],),
            )
            for p in gp.posteriors
        ]
    )
    gp.temporary_data["sn2_new"] = sn2_new.mean(1)
    out = _gp_log_joint(vp, gp, 0, 0, 0, 1, separate_K=True)
    optim_state["var_log_joint_samples"] = out[2]
    optim_state["cov_log_joint_components"] = out[6]
    return optim_state


@pytest.mark.parametrize("ns_gp", [1, 2])
def test_integrated_covariance_matches_grid_quadrature(ns_gp):
    """D = 1: integrate the posterior covariance from ``predict_full`` over
    each component on a fine grid."""
    D, K = 1, 3
    gp = _gp(D, ns_gp)
    vp = _vp(D, K)
    Xs = np.array([[-1.5], [0.2], [3.0]])
    c = integrated_posterior_covariance(Xs, vp, gp)
    assert c.shape == (3, ns_gp, K)

    grid = np.linspace(-14, 14, 14001).reshape(-1, 1)
    dx = grid[1, 0] - grid[0, 0]
    for i, x in enumerate(Xs):
        # C(f(grid), f(x)) per sample, from full covariances of chunks of
        # the grid with the candidate appended (a full 14001-point
        # covariance would be 1.6 GB per sample).
        cross = np.empty((grid.shape[0], ns_gp))
        for lo in range(0, grid.shape[0], 2000):
            chunk = grid[lo : lo + 2000]
            __, cov = gp.predict_full(np.vstack([chunk, x[None, :]]))
            cross[lo : lo + chunk.shape[0]] = cov[:-1, -1, :]
        for s in range(ns_gp):
            for k in range(K):
                sd = vp.sigma[0, k] * vp.lambd[0, 0]
                dens = np.exp(
                    -0.5 * ((grid[:, 0] - vp.mu[0, k]) / sd) ** 2
                ) / (sd * np.sqrt(2 * np.pi))
                expected = np.sum(dens * cross[:, s]) * dx
                assert np.isclose(c[i, s, k], expected, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("ns_gp", [1, 2])
def test_eig_matches_definition(ns_gp):
    """-1/2 log(1 - rho^2) with rho the correlation of G and y_*, from the
    same integrals as the expected log joint."""
    D, K = 2, 3
    gp = _gp(D, ns_gp)
    vp = _vp(D, K)
    optim_state = _prepare(gp, vp)
    Xs = np.array([[0.0, 0.0], [1.0, -1.0], [-2.0, 2.0]])
    acq = AcqFcnEIG()
    actual = acq(Xs.copy(), gp, vp, None, optim_state)
    assert actual.shape == (3,)
    assert np.all(actual <= 0)

    c = integrated_posterior_covariance(Xs, vp, gp)
    cov_Gy = c @ vp.w.reshape(-1)  # (Nx, Ns)
    __, f_s2 = gp.predict(Xs, separate_samples=True)
    sn2 = acq._estimate_observation_noise(Xs, gp, optim_state)
    var_y = f_s2 + sn2[:, None]
    var_G = np.atleast_1d(optim_state["var_log_joint_samples"])
    rho2 = cov_Gy**2 / (var_y * var_G[None, :])
    assert np.all(rho2 <= 1 + 1e-12)
    expected = np.mean(0.5 * np.log(1 - rho2), axis=1)
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-14)


def test_more_information_near_the_mass():
    """A candidate in the posterior mass is more informative than one far
    from it and from every training point."""
    gp = _gp(2, 1)
    vp = _vp(2, 2)
    optim_state = _prepare(gp, vp)
    near = np.mean(vp.mu, axis=1)[None, :]
    far = np.array([[40.0, -40.0]])
    for components in (False, True):
        acq = AcqFcnEIG(components=components)
        a = acq(np.vstack([near, far]), gp, vp, None, optim_state)
        assert a[0] < a[1]
        assert np.isclose(a[1], 0.0, atol=1e-12)


@pytest.mark.parametrize("ns_gp", [1, 2])
def test_components_equals_scalar_at_one_component(ns_gp):
    gp = _gp(2, ns_gp)
    vp = _vp(2, 1)
    optim_state = _prepare(gp, vp)
    Xs = np.array([[0.0, 0.0], [1.0, -1.0], [-2.0, 2.0]])
    a_scalar = AcqFcnEIG()(Xs.copy(), gp, vp, None, optim_state)
    a_comp = AcqFcnEIG(components=True)(Xs.copy(), gp, vp, None, optim_state)
    assert np.allclose(a_scalar, a_comp, rtol=1e-8, atol=1e-14)


def test_components_information_is_at_least_scalar():
    """The information about the vector of component integrals is at least
    the information about their weighted sum (a function of the vector)."""
    gp = _gp(2, 2)
    vp = _vp(2, 4)
    optim_state = _prepare(gp, vp)
    rng = np.random.default_rng(3)
    Xs = rng.uniform(-2, 2, size=(20, 2))
    a_scalar = AcqFcnEIG()(Xs.copy(), gp, vp, None, optim_state)
    a_comp = AcqFcnEIG(components=True)(Xs.copy(), gp, vp, None, optim_state)
    assert np.all(a_comp <= a_scalar + 1e-12)


@pytest.mark.parametrize("ns_gp", [1, 2])
@pytest.mark.parametrize("components", [False, True])
def test_variance_regularization_batch_matches_pointwise(ns_gp, components):
    """A non-log acquisition is scaled by exp(-(tol / var - 1)) where the
    GP variance is below the threshold, in a batch as pointwise."""
    gp = _gp(2, ns_gp)
    vp = _vp(2, 3)
    optim_state = _prepare(gp, vp)
    X_eval = np.vstack([gp.X[:2] + 1e-3, np.array([[1.5, -1.5]])])
    acq_fcn = AcqFcnEIG(components=components)
    optim_state["variance_regularized_acq_fcn"] = False
    base = acq_fcn(X_eval.copy(), gp, vp, None, optim_state)
    f_mu, f_s2 = gp.predict(X_eval, separate_samples=True)
    var_tot = np.mean(f_s2, axis=1)
    if ns_gp > 1:
        var_tot += np.var(f_mu, axis=1, ddof=1)
    tol_var = 0.5 * (np.min(var_tot) + np.max(var_tot))
    mask = var_tot < tol_var
    assert np.any(mask) and np.any(~mask)
    expected = base.copy()
    expected[mask] *= np.exp(-(tol_var / var_tot[mask] - 1))
    optim_state["variance_regularized_acq_fcn"] = True
    optim_state["tol_gp_var"] = tol_var
    batch = acq_fcn(X_eval.copy(), gp, vp, None, optim_state)
    pointwise = np.array(
        [acq_fcn(x.copy(), gp, vp, None, optim_state)[0] for x in X_eval]
    )
    assert batch.shape == (3,)
    assert np.allclose(batch, expected)
    assert np.allclose(batch, pointwise)


def test_acq_info_and_string():
    acq = AcqFcnEIG()
    assert not acq.acq_info["log_flag"]
    assert acq.acq_info["compute_var_log_joint"]
    assert not acq.acq_info["components"]
    assert string_to_acq("AcqFcnEIG(components=True)").components
    assert string_to_acq("AcqFcnEIG()").components is False
