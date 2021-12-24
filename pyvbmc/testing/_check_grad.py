r"""Utilities for testing analytical versus numerical gradients.
"""
import numpy as np
from scipy.misc import derivative


def _partial_eval(f, x0_orig, x0_i, i):
    x0 = x0_orig.copy()
    x0[i] = x0_i
    return f(x0)


def _compute_gradient(f, x0, dx):
    num_grad = np.zeros(x0.shape)

    for i in range(0, np.size(x0)):
        f_i = lambda x0_i: _partial_eval(f, x0, x0_i, i)
        tmp = derivative(f_i, x0[i], dx=dx, order=5)
        num_grad[i] = tmp

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
    return np.allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol)
