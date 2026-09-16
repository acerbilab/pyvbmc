import numpy as np
import pytest

pytest.importorskip("pymc")
pytest.importorskip("arviz_base")

from pyvbmc.pymc._plausible import (
    clip_inside,
    laplace_box,
    location_outside_prior,
    marginal_sd,
    move_inside,
    quantile_box,
    search_mode,
)


def test_move_inside_uses_automatic_margin_without_mutating_input():
    x0 = np.array([0.0, 10.0, 3.0])
    original = x0.copy()
    moved, mask = move_inside(
        x0,
        np.array([0.0, 0.0, -np.inf]),
        np.array([10.0, 10.0, np.inf]),
    )
    np.testing.assert_array_equal(x0, original)
    np.testing.assert_allclose(moved, [0.02, 9.98, 3.0])
    np.testing.assert_array_equal(mask, [True, True, False])


def test_search_mode_reports_attempts_and_finite_observations():
    def value_and_grad(x):
        value = -((x[0] - 2.0) ** 2)
        gradient = np.array([-2.0 * (x[0] - 2.0)])
        return value, gradient

    x_best, X, y, n_calls, cap_reached = search_mode(
        value_and_grad,
        np.array([0.0]),
        np.array([-np.inf]),
        np.array([np.inf]),
        max_calls=20,
    )
    assert n_calls == X.shape[0] == y.shape[0]
    assert n_calls <= 20
    assert not cap_reached
    np.testing.assert_allclose(x_best, [2.0], atol=1e-8)
    np.testing.assert_allclose(y, -((X[:, 0] - 2.0) ** 2))


def test_search_mode_strict_cap():
    def value_and_grad(x):
        return -((x[0] - 2.0) ** 2), np.array([-2.0 * (x[0] - 2.0)])

    _, X, y, n_calls, cap_reached = search_mode(
        value_and_grad,
        np.array([0.0]),
        np.array([-np.inf]),
        np.array([np.inf]),
        max_calls=1,
    )
    assert n_calls == 1
    assert X.shape == (1, 1)
    assert y.shape == (1,)
    assert cap_reached


def test_search_mode_charges_nonfinite_attempt_without_retaining_it(
    monkeypatch,
):
    def fake_minimize(objective, x_start, **kwargs):
        objective(x_start)
        objective(np.array([1.0]))
        objective(np.array([2.0]))

    monkeypatch.setattr("pyvbmc.pymc._plausible.minimize", fake_minimize)

    def value_and_grad(x):
        if x[0] == 1.0:
            return np.nan, np.array([0.0])
        return -x[0] ** 2, np.array([-2.0 * x[0]])

    x_best, X, y, n_calls, cap_reached = search_mode(
        value_and_grad,
        np.array([0.0]),
        np.array([-np.inf]),
        np.array([np.inf]),
        max_calls=2,
    )
    np.testing.assert_array_equal(x_best, [0.0])
    np.testing.assert_array_equal(X, [[0.0]])
    np.testing.assert_array_equal(y, [0.0])
    assert n_calls == 2
    assert cap_reached


def test_marginal_sd_requires_positive_definite_precision():
    sd, usable = marginal_sd(np.diag([-4.0, -9.0]))
    np.testing.assert_allclose(sd, [0.5, 1 / 3])
    np.testing.assert_array_equal(usable, [True, True])

    sd, usable = marginal_sd(np.diag([-1.0, 1.0]))
    assert np.all(np.isnan(sd))
    assert not np.any(usable)


def test_prior_quantiles_and_location_are_diagnostic_only():
    draws = np.arange(100, dtype=np.float64).reshape(50, 2)
    lower, upper = quantile_box(draws)
    np.testing.assert_allclose(
        [lower, upper], np.quantile(draws, [0.05, 0.95], axis=0)
    )
    x0 = np.array([1000.0, 50.0])
    original = x0.copy()
    mask = location_outside_prior(x0, draws)
    np.testing.assert_array_equal(mask, [True, False])
    np.testing.assert_array_equal(x0, original)


def test_clip_inside_uses_halfway_rule_and_repairs_collapsed_box():
    plb, pub, clipped = clip_inside(
        np.array([-1.0, 8.0]),
        np.array([9.9, 2.0]),
        np.array([0.0, 0.0]),
        np.array([10.0, 10.0]),
        np.array([0.02, 5.0]),
    )
    np.testing.assert_allclose(plb, [0.01, 0.1])
    np.testing.assert_allclose(pub, [9.9, 9.9])
    np.testing.assert_array_equal(clipped, [True, True])


def test_laplace_box_fallback_draws_once_and_never_moves_location():
    calls = 0

    def prior_draws():
        nonlocal calls
        calls += 1
        return np.column_stack(
            [np.linspace(-1.0, 1.0, 4000), np.linspace(-2.0, 2.0, 4000)]
        )

    x0 = np.array([1000 / 101, 0.0])
    result = laplace_box(
        x0,
        lambda: np.diag([-101.0, 1.0]),
        np.array([-np.inf, -1.0]),
        np.array([np.inf, 1.0]),
        prior_draws,
    )
    actual_x0, plb, pub, curvature, location, clipped = result
    np.testing.assert_array_equal(actual_x0, x0)
    np.testing.assert_array_equal(curvature, [True, True])
    np.testing.assert_array_equal(location, [True, False])
    assert calls == 1
    assert np.all(plb < actual_x0)
    assert np.all(actual_x0 < pub)
    assert clipped.shape == (2,)


def test_laplace_box_skips_prior_draws_when_not_needed():
    def prior_draws():
        raise AssertionError("prior draws should not be requested")

    result = laplace_box(
        np.array([0.0]),
        lambda: np.array([[-4.0]]),
        np.array([-np.inf]),
        np.array([np.inf]),
        prior_draws,
        check_location=False,
    )
    _, plb, pub, curvature, location, clipped = result
    np.testing.assert_allclose([plb, pub], [[-1.5], [1.5]])
    assert not curvature[0]
    assert not location[0]
    assert not clipped[0]
