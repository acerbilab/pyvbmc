import numpy as np


def get_hpd(X: np.ndarray, y: np.ndarray, hpd_frac: float = 0.8):
    """
    Get high-posterior density dataset.

    Parameters
    ==========
    X : ndarray, shape (N, D)
        The training points.
    y : ndarray, shape (N, 1)
        The training targets.
    hpd_frac : float
        The portion of the training set to consider, by default 0.8.

    Returns
    =======
    hpd_X : ndarray
        High-posterior density training points.
    hpd_y : ndarray
        High-posterior density training targets.
    hpd_range : ndarray, shape (D,)
        The range of values of hpd_X in each dimension.
    indices : ndarray
        The indices of the points returned with respect to the original data
        being passed to the function.
    """

    N, D = X.shape

    # Subsample high posterior density dataset.
    # Sort by descending order, not ascending.
    order = np.argsort(y, axis=None)[::-1]
    hpd_N = round(hpd_frac * N)
    indices = order[0:hpd_N]
    hpd_X = X[indices]
    hpd_y = y[indices]

    if hpd_N > 0:
        hpd_range = np.max(hpd_X, axis=0) - np.min(hpd_X, axis=0)
    else:
        hpd_range = np.full((D), np.NaN)

    return hpd_X, hpd_y, hpd_range, indices
