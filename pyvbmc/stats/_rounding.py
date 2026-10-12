"""Rounding to the nearest integer, a half away from zero."""

import numpy as np


def round_half_away_from_zero(x):
    """Round to the nearest integer, sending a half away from zero.

    MATLAB's ``round``, which stands at the counterpart of every call of
    this function, sends a half away from zero, where Python's built-in
    ``round`` and :py:func:`numpy.round` send it to the nearer even
    integer: MATLAB's ``round(2.5)`` is 3 and Python's is 2.

    The decision is read from the exact fractional part. The shorter
    ``floor(abs(x) + 0.5)`` states the same rule but is not it: the sum
    rounds the largest double below a half up to one, which would send
    that value to 1 instead of 0, and it loses the last bit of an integer
    large enough for the addition to be inexact.

    Parameters
    ----------
    x : float or array_like
        The finite value, or array of values, to round.

    Returns
    -------
    rounded : int or np.ndarray
        A Python ``int`` for a scalar argument, so that a count stays a
        count; an array of floats of the shape of the argument otherwise.
    """
    values = np.asarray(x, dtype=float)
    whole = np.trunc(values)
    rounded = whole + np.sign(values) * (np.abs(values - whole) >= 0.5)
    if rounded.ndim == 0:
        return int(rounded)
    return rounded
