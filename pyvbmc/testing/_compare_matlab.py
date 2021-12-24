r"""Helpers for testing python results versus the original MATLAB version.
"""
import numpy as np
from scipy.special import erfinv


def randn2(*args, **kwargs):
    """
    For reproducing the same random normal samples with MATLAB.
    Modified from: https://github.com/jonasrauber/randn-matlab-python.
    MATLAB randn should be replaced by randn2 defined in randn2.m.
    """
    uniform = np.random.rand(*args, **kwargs)
    normal = np.sqrt(2.0) * erfinv(2 * uniform - 1)
    return np.reshape(normal.ravel(), args, "F")


def fisher_yates_shuffle(a):
    """
    For reproducing same random shuffle with MATLAB.
    MATLAB needs to use the same algorithm as this.
    """
    b = a.copy()
    left = b.size

    while left > 1:
        i = int(np.floor(np.random.rand() * left))
        left -= 1
        b[i], b[left] = b[left], b[i]
    return b


def rand_perm(n):
    """
    For reproducing same random permutations with MATLAB.
    MATLAB needs to use the same algorithm as this.
    """
    return fisher_yates_shuffle(np.array(range(0, n)))


def rand_int(hi):
    """
    For reproducing same random integer with MATLAB.
    MATLAB already uses this algorithm.
    """
    proportion = 1.0 / hi
    tmp = np.random.rand()
    res = lo
    while res * proportion < tmp:
        res += 1
    return res
