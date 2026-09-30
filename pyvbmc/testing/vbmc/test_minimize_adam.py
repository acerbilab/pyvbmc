import numpy as np
import scipy as sp
import scipy.optimize

from pyvbmc.vbmc.minimize_adam import minimize_adam


def test_minimize_adam_sphere():
    f = lambda x_: (np.sum(x_**2), 2 * x_)
    x0 = np.array([-3.0, -4.0])

    x, y, _, _, _ = minimize_adam(f, x0)

    assert np.all(np.abs(x) < 0.1)
    assert np.abs(y) < 0.001


def test_minimize_adam_sphere_with_noise():
    # ADAM is supposed to work for noisy (unbiased) estimates
    # of the gradient, so it is OK but not particularly good for
    # deterministic optimization as in the other tests.
    # Here we have i.i.d. Gaussian noise at each gadient with
    # variance not too small.
    rng = np.random.default_rng(20260921)
    f = lambda x_: (
        np.sum(x_**2),
        2 * x_ + rng.normal(scale=3, size=x_.shape),
    )
    x0 = np.array([-0.3, -0.4])

    x, y, _, _, _ = minimize_adam(f, x0, use_early_stopping=False)

    assert np.all(np.abs(x) < 0.5)
    assert np.abs(y) < 0.1


def test_minimize_adam_matyas():
    def f(x_):
        val = 0.26 * (x_[0] ** 2 + x_[1] ** 2) - 0.48 * x_[0] * x_[1]
        grad_1 = 0.52 * x_[0] - 0.48 * x_[1]
        grad_2 = 0.52 * x_[1] - 0.48 * x_[0]
        grad = np.array([grad_1, grad_2])
        return val, grad

    x0 = np.array([-0.3, -0.4])
    lb = np.array([-10.0, -10.0])
    ub = np.array([10.0, 10.0])

    x, y, _, _, _ = minimize_adam(f, x0, lb, ub, use_early_stopping=False)

    assert np.all(np.abs(x) < 1.0)
    assert np.abs(y) < 0.1


def test_minimize_adam_matyas_with_noise():
    """The Matyas function is ``0.5 * u**2 + 0.02 * v**2``, with ``u`` the
    coordinate across the line ``x[0] == x[1]`` and ``v`` the one along it.
    Gradient noise of SD 3 swamps the slope along that valley, where the
    iterates perform a random walk, so what ADAM holds is the position
    across the valley, and the bounds differ by direction. Over 300 seeds of
    the noise the largest values are 0.13 across, 2.8 along and 0.16 for the
    function, which is 3.8 or more on the boundary of the box."""
    rng = np.random.default_rng(20260921)

    def f(x_):
        val = 0.26 * (x_[0] ** 2 + x_[1] ** 2) - 0.48 * x_[0] * x_[1]
        grad_1 = 0.52 * x_[0] - 0.48 * x_[1]
        grad_2 = 0.52 * x_[1] - 0.48 * x_[0]
        grad = np.array([grad_1, grad_2]) + rng.normal(scale=3, size=(2,))
        return val, grad

    x0 = np.array([-0.3, -0.4])
    lb = np.array([-10.0, -10.0])
    ub = np.array([10.0, 10.0])

    x, y, _, _, _ = minimize_adam(f, x0, lb, ub, use_early_stopping=False)

    across = np.abs(x[0] - x[1]) / np.sqrt(2)
    along = np.abs(x[0] + x[1]) / np.sqrt(2)
    assert across < 0.5
    assert along < 5.0
    assert np.abs(y) < 0.5


def test_minimize_adam_leaves_the_starting_point_unchanged():
    f = lambda x_: (np.sum(x_**2), 2 * x_)
    x0 = np.array([1.0, -2.0])
    expected = x0.copy()

    minimize_adam(f, x0, max_iter=5)

    assert np.array_equal(x0, expected)


def test_minimize_adam_short_run_averages_every_iterate():
    """A run shorter than the trailing window of 20 iterations returns the
    average over all the iterates it performed."""
    f = lambda x_: (np.sum(x_**2), 2 * x_)
    x0 = np.array([1.0, -2.0])

    x, y, x_tab, y_tab, iterations = minimize_adam(f, x0, max_iter=15)

    assert iterations == 15
    assert x_tab.shape == (2, 15)
    assert np.allclose(x, np.mean(x_tab, axis=1))
    assert np.isclose(y, np.mean(y_tab))


def test_minimize_adam_rosen():
    f = lambda x_: (sp.optimize.rosen(x_), sp.optimize.rosen_der(x_))
    x0 = np.array([-3.0, -4.0])

    # Here the early stopping makes us have results that are not so good.
    x, y, _, _, _ = minimize_adam(
        f, x0, max_iter=50000, use_early_stopping=False
    )

    assert np.all(np.isclose(x, 1))
    assert np.isclose(y, 0.0)
