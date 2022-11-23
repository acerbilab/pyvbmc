import numpy as np

from pyvbmc.testing import check_grad


def test_check_grad():
    f = lambda x: np.sum(x**2 + np.exp(x) + np.sin(x))
    f_grad = lambda x: 2 * x + np.exp(x) + np.cos(x)
    x0 = np.array([0.1, 0.3, 0.6])
    assert check_grad(f, f_grad, x0)
