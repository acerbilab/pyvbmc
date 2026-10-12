"""Checks for deterministic expected parameter-transform Jacobians."""

import copy

import numpy as np
import pytest
from scipy.integrate import quad

from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.svbmc._jacobian import expected_log_jacobian
from pyvbmc.variational_posterior import VariationalPosterior


def _vp(transform, mu, sigma, lambd):
    mu = np.asarray(mu, dtype=np.float64)
    vp = VariationalPosterior(
        mu.shape[0],
        mu.shape[1],
        x0=np.zeros(mu.shape[0]),
        parameter_transformer=transform,
        rng=91,
    )
    vp.mu = mu.copy()
    vp.sigma = np.asarray(sigma, dtype=np.float64).reshape(1, -1)
    vp.lambd = np.asarray(lambd, dtype=np.float64).reshape(-1, 1)
    return vp


def _pretransform_marginals(vp):
    """Independent row-vector calculation of pre-transform marginals."""
    transform = vp.parameter_transformer
    scale = (
        np.ones(vp.D)
        if transform.scale is None
        else np.asarray(transform.scale).reshape(vp.D)
    )
    rotation = np.eye(vp.D) if transform.R_mat is None else transform.R_mat
    component_sd = vp.lambd * vp.sigma
    means = (vp.mu.T * scale) @ rotation.T
    variances = (component_sd.T**2 * scale**2) @ (rotation.T**2)
    return means, variances


def test_affine_expectation_is_exact_and_float64(monkeypatch):
    D, K = 3, 2
    transform = ParameterTransformer(
        D,
        plb_orig=np.array([[-2.0, -1.0, 0.5]]),
        pub_orig=np.array([[4.0, 3.0, 5.5]]),
    )
    transform.R_mat = np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transform.scale = np.array([0.5, 2.0, 1.5])
    vp = _vp(
        transform,
        [[-8.0, 9.0], [3.0, -4.0], [1.0, 20.0]],
        [0.2, 7.0],
        [0.4, 2.0, 3.0],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("analytic transforms must not use quadrature")

    monkeypatch.setattr(
        "pyvbmc.svbmc._jacobian.roots_hermitenorm", fail_if_called
    )
    actual = expected_log_jacobian(vp)
    expected = np.sum(np.log(transform.delta)) + np.sum(
        np.log(transform.scale)
    )
    assert actual.shape == (K,)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, np.full(K, expected))


def test_mixed_warped_probit_matches_closed_form():
    transform = ParameterTransformer(
        4,
        lb_orig=np.array([[-2.0, -np.inf, 0.5, -np.inf]]),
        ub_orig=np.array([[3.0, np.inf, 7.5, np.inf]]),
        plb_orig=np.array([[-1.2, -4.0, 1.0, -2.0]]),
        pub_orig=np.array([[2.2, 6.0, 6.0, 3.0]]),
        transform_type="probit",
    )
    transform.R_mat = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
        ]
    )
    transform.scale = np.array([0.4, 1.7, 2.3, 0.8])
    vp = _vp(
        transform,
        [[-1.0, 2.0], [0.2, -3.0], [1.5, 0.7], [4.0, -0.5]],
        [0.6, 2.5],
        [0.5, 1.2, 2.0, 0.8],
    )

    means, variances = _pretransform_marginals(vp)
    expected = np.zeros(vp.K)
    bounded = transform.type == 12
    for k in range(vp.K):
        expected[k] = np.sum(np.log(transform.scale))
        expected[k] += np.sum(np.log(transform.delta[~bounded]))
        for d in np.flatnonzero(bounded):
            y_mean = transform.delta[d] * means[k, d] + transform.mu[d]
            y_var = transform.delta[d] ** 2 * variances[k, d]
            expected[k] += (
                np.log(transform.ub_orig[0, d] - transform.lb_orig[0, d])
                + np.log(transform.delta[d])
                - 0.5 * np.log(2.0 * np.pi)
                - 0.5 * (y_mean**2 + y_var)
            )

    np.testing.assert_allclose(
        expected_log_jacobian(vp), expected, rtol=2e-15, atol=1e-13
    )


@pytest.mark.parametrize("kind", ["logit", "student4"])
def test_scalar_quadrature_matches_independent_integral(kind):
    transform = ParameterTransformer(
        1,
        lb_orig=np.array([[-3.0]]),
        ub_orig=np.array([[5.0]]),
        plb_orig=np.array([[-2.4]]),
        pub_orig=np.array([[2.7]]),
        transform_type=kind,
    )
    vp = _vp(transform, [[-1.7, 3.2]], [0.35, 2.8], [1.4])

    def reference(k):
        mean = vp.mu[0, k]
        sd = vp.lambd[0, 0] * vp.sigma[0, k]

        def integrand(z):
            u = mean + sd * z
            return (
                transform.log_abs_det_jacobian(np.array([[u]]))[0]
                * np.exp(-0.5 * z**2)
                / np.sqrt(2.0 * np.pi)
            )

        return quad(integrand, -np.inf, np.inf, epsabs=2e-11, epsrel=2e-11)[0]

    expected = np.array([reference(k) for k in range(vp.K)])
    np.testing.assert_allclose(expected_log_jacobian(vp), expected, atol=1e-9)


@pytest.mark.parametrize("kind", ["logit", "student4"])
def test_broad_mixed_warped_components_match_seeded_monte_carlo(kind):
    transform = ParameterTransformer(
        3,
        lb_orig=np.array([[-4.0, -np.inf, 0.0]]),
        ub_orig=np.array([[6.0, np.inf, 20.0]]),
        plb_orig=np.array([[-3.0, -2.0, 2.0]]),
        pub_orig=np.array([[4.0, 5.0, 16.0]]),
        transform_type=kind,
    )
    angle = 0.63
    transform.R_mat = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform.scale = np.array([0.45, 2.1, 1.3])
    vp = _vp(
        transform,
        [[-4.0, 3.0], [1.5, -2.0], [0.7, -1.2]],
        [1.8, 4.5],
        [0.8, 2.3, 1.6],
    )
    expected = expected_log_jacobian(vp)

    rng = np.random.default_rng(20260912)
    n = 300_000
    for k in range(vp.K):
        z = rng.standard_normal((n, vp.D))
        u = vp.mu[:, k] + z * (vp.lambd[:, 0] * vp.sigma[0, k])
        values = transform.log_abs_det_jacobian(u)
        error = 6.0 * values.std(ddof=1) / np.sqrt(n) + 2e-3
        assert abs(values.mean() - expected[k]) <= error


def test_expected_log_jacobian_does_not_advance_rng():
    transform = ParameterTransformer(
        2,
        lb_orig=np.array([[-1.0, -np.inf]]),
        ub_orig=np.array([[2.0, np.inf]]),
        transform_type="student4",
    )
    vp = _vp(transform, [[0.4, -0.8], [1.0, 2.0]], [0.7, 1.5], [1.2, 0.5])
    state = copy.deepcopy(vp.rng.bit_generator.state)
    expected_log_jacobian(vp)
    assert vp.rng.bit_generator.state == state


def test_quadrature_nonconvergence_names_components(monkeypatch):
    transform = ParameterTransformer(
        1,
        lb_orig=np.array([[-1.0]]),
        ub_orig=np.array([[1.0]]),
        transform_type="logit",
    )
    vp = _vp(transform, [[0.0, 0.5]], [1.0, 2.0], [1.0])
    calls = []

    def alternating_rule(order):
        calls.append(order)
        nodes = np.full(order, float(len(calls) % 2))
        weights = np.full(order, np.sqrt(2.0 * np.pi) / order)
        return nodes, weights

    monkeypatch.setattr(
        "pyvbmc.svbmc._jacobian.roots_hermitenorm", alternating_rule
    )
    monkeypatch.setattr(
        "pyvbmc.svbmc._jacobian.quad", lambda *args, **kwargs: (0.0, np.inf)
    )
    with pytest.raises(RuntimeError, match=r"2048.*component\(s\).*0"):
        expected_log_jacobian(vp)
    assert calls == [32, 64, 128, 256, 512, 1024, 2048]


def test_convergence_uses_absolute_coordinate_changes(monkeypatch):
    transform = ParameterTransformer(
        2,
        lb_orig=np.array([[-1.0, -1.0]]),
        ub_orig=np.array([[1.0, 1.0]]),
        transform_type="logit",
    )
    vp = _vp(transform, [[-1.0], [1.0]], [1.0], [1.0, 1.0])
    calls = []

    def cancelling_rule(order):
        calls.append(order)
        # The two symmetric coordinates exchange their contributions at
        # every level. Their signed changes cancel exactly, while the sum of
        # their absolute changes stays nonzero.
        nodes = np.full(order, 0.5 if len(calls) % 2 == 0 else -0.5)
        weights = np.full(order, np.sqrt(2.0 * np.pi) / order)
        return nodes, weights

    monkeypatch.setattr(
        "pyvbmc.svbmc._jacobian.roots_hermitenorm", cancelling_rule
    )
    monkeypatch.setattr(
        "pyvbmc.svbmc._jacobian.quad", lambda *args, **kwargs: (0.0, np.inf)
    )
    with pytest.raises(RuntimeError):
        expected_log_jacobian(vp)
    assert calls[-1] == 2048
