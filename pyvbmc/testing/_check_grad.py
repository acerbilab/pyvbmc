r"""Utilities for testing analytical versus numerical gradients.
"""
import numpy as np
from scipy.differentiate import jacobian


def check_grad(
    f_not_vectorized,
    grad,
    x0,
    initial_step=1e-2,
    rtol=1e-3,
    atol=1e-6,
):
    analytical_grad = grad(x0)

    def f(x):
        return np.apply_along_axis(f_not_vectorized, axis=0, arr=x)

    numerical_grad = jacobian(f, x0, initial_step=initial_step).df
    return np.allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol)
