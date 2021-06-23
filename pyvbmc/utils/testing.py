import numpy as np
from scipy.special import erfinv
from scipy.misc import derivative


def randn2(*args, **kwargs):
    """
    For reproducing the same random normal samples with MATLAB.
    Modified from: https://github.com/jonasrauber/randn-matlab-python.
    """
    uniform = np.random.rand(*args, **kwargs)
    normal = np.sqrt(2.0) * erfinv(2 * uniform - 1)
    return np.reshape(normal.ravel(), args, "F")


def partial_eval(f, x0_orig, x0_i, i):
    x0 = x0_orig.copy()
    x0[i] = x0_i
    return f(x0)


def compute_gradient(f, x0):
    num_grad = np.zeros(x0.shape)

    for i in range(0, np.size(x0)):
        f_i = lambda x0_i: partial_eval(f, x0, x0_i, i)
        tmp = derivative(f_i, x0[i], dx=np.finfo(float).eps ** 0.5, order=5)
        num_grad[i] = tmp

    return num_grad


def check_grad(f, grad, x0, rtol=1e-3, atol=1e-6):
    analytical_grad = grad(x0)
    numerical_grad = compute_gradient(f, x0)
    return np.allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol)
