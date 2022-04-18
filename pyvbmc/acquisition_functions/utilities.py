import numpy as np


def sq_dist(a, b):
    """
    Compute matrix of all pairwise squared distances between two sets
    of vectors, stored in the columns of the two matrices `a` and `b`.

    Parameters
    ----------
    a : np.array, shape (n, D)
        First set of vectors.
    b : np.array, shape (m, D)
        Second set of vectors.

    Returns
    -------
    c: np.array, shape(n, m)
        The matrix of all pairwise squared distances.
    """
    n = a.shape[0]
    m = b.shape[0]
    mu = (m / (n + m)) * np.mean(b, axis=0) + (n / (n + m)) * np.mean(
        a, axis=0
    )
    a = a - mu
    b = b - mu
    c = np.sum(a * a, axis=1, keepdims=True) + (
        np.sum(b * b, axis=1, keepdims=True).T - (2 * a @ b.T)
    )
    return np.maximum(c, 0)
