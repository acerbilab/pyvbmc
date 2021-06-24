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
