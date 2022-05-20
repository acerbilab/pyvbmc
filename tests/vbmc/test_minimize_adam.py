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
    f = lambda x_: (
        np.sum(x_**2),
        2 * x_ + np.random.normal(scale=3, size=x_.shape),
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
    def f(x_):
        val = 0.26 * (x_[0] ** 2 + x_[1] ** 2) - 0.48 * x_[0] * x_[1]
        grad_1 = 0.52 * x_[0] - 0.48 * x_[1]
        grad_2 = 0.52 * x_[1] - 0.48 * x_[0]
        grad = np.array([grad_1, grad_2]) + np.random.normal(
            scale=3, size=(2,)
        )
        return val, grad

    x0 = np.array([-0.3, -0.4])
    lb = np.array([-10.0, -10.0])
    ub = np.array([10.0, 10.0])

    x, y, _, _, _ = minimize_adam(f, x0, lb, ub, use_early_stopping=False)

    assert np.all(np.abs(x) < 1.0)
    assert np.abs(y) < 0.1


def test_minimize_adam_rosen():
    f = lambda x_: (sp.optimize.rosen(x_), sp.optimize.rosen_der(x_))
    x0 = np.array([-3.0, -4.0])

    # Here the early stopping makes us have results that are not so good.
    x, y, _, _, _ = minimize_adam(
        f, x0, max_iter=50000, use_early_stopping=False
    )

    assert np.all(np.isclose(x, 1))
    assert np.isclose(y, 0.0)
