r"""Utilities for testing analytical versus numerical gradients.
"""
import numpy as np


def _partial_eval(f, x0_orig, x0_i, i):
    x0 = x0_orig.copy()
    x0[i] = x0_i
    return f(x0)


def _compute_gradient(f, x0, dx):
    """Compute the gradient of a function via finite differences"""
    num_grad = np.zeros(x0.shape)
    weights = (
        np.array([1, -8, 0, 8, -1]) / 12.0
    )  # O5 finite difference stencil

    for i in range(0, np.size(x0)):
        f_i = lambda x0_i: _partial_eval(f, x0, x0_i, i)
        tmp = 0.0
        for k in range(-2, 3):
            tmp += weights[k + 2] * f_i(x0[i] + k * dx)
        num_grad[i] = tmp / dx

    return num_grad


def check_grad(
    f,
    grad,
    x0,
    rtol=1e-3,
    atol=1e-6,
    dx=np.finfo(float).eps ** 0.5,
):
    analytical_grad = grad(x0)
    numerical_grad = _compute_gradient(f, x0, dx)
    print(analytical_grad)
    print(numerical_grad)
    return np.allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol)
