import numpy as np
from scipy import fftpack
from scipy.optimize import brentq

"""
Created on Wed May 19 22:20:51 2021
​
@author: Luigi Acerbi
"""


def _linear_binning(samples: np.ndarray, grid_points: np.ndarray):
    """
    linear_binning [summary]

    Parameters
    ----------
    samples : np.ndarray
        the samples to assign weights to
    grid_points : np.ndarray
        [description]

    Returns
    -------
    np.ndarray
        [description]
    """
    samples = samples[
        np.logical_and(samples >= grid_points[0], samples <= grid_points[-1])
    ]
    dx = grid_points[1] - grid_points[0]
    idx = np.floor((samples - (grid_points[0] - 0.5 * dx)) / dx)
    u, u_counts = np.unique(idx, return_counts=True)
    counts = np.zeros(len(grid_points))
    counts[u.astype(np.int)] = u_counts

    return counts


def _fixed_point(t: float, N: int, irange_squared: np.ndarray, a2: np.ndarray):
    """
    _fixed_point Compute the fixed point according to Botev et al. (2010).
    This implements the function t-zeta*gamma^[l](t).
    Based on an implementation by Daniel B. Smith, PhD:
    https://github.com/Daniel-B-Smith/KDE-for-SciPy/blob/master/kde.py


    Parameters
    ----------
    t : float
        [description]
    N : int
        [description]
    irange_squared : np.ndarray
        [description]
    a2 : np.ndarray
        [description]

    Returns
    -------
    [type]
        [description]
    """

    irange_squared = np.asfarray(irange_squared, dtype=np.float64)
    a2 = np.asfarray(a2, dtype=np.float64)
    ell = 7
    f = (
        2.0
        * np.pi ** (2 * ell)
        * np.sum(
            np.power(irange_squared, ell)
            * a2
            * np.exp(-irange_squared * np.pi ** 2.0 * t)
        )
    )

    if f <= 0:
        return -1

    for s in reversed(range(2, ell)):
        odd_numbers_prod = np.product(
            np.arange(1, 2 * s + 1, 2, dtype=np.float64)
        )
        K0 = odd_numbers_prod / np.sqrt(2.0 * np.pi)
        const = (1.0 + (1.0 / 2.0) ** (s + 1.0 / 2.0)) / 3.0
        time = np.power((2 * const * K0 / (N * f)), (2.0 / (3.0 + 2.0 * s)))
        f = (
            2.0
            * np.pi ** (2.0 * s)
            * np.sum(
                np.power(irange_squared, s)
                * a2
                * np.exp(-irange_squared * np.pi ** 2.0 * time)
            )
        )

    t_opt = np.power(2.0 * N * np.sqrt(np.pi) * f, -2.0 / 5.0)

    return t - t_opt


def _root(function: callable, N: int, args: tuple):
    """
    _root Root finding algorithm based on MATLAB implementation by Botev et al. (2010)
    Try to find smallest root whenever there is more than one

    Parameters
    ----------
    function : callable
        [description]
    N : int
        [description]
    args : tuple
        extra arguments for the function

    Returns
    -------
    root: float or none
        [description]
    """
    N = max(min(1050.0, N), 50.0)
    tol = 1e-12 + 0.01 * (N - 50.0) / 1000.0
    converged = False
    while not converged:
        try:
            x, res = brentq(
                function, 0, tol, args=args, full_output=True, disp=False
            )
            converged = np.bool(res.converged)
        except ValueError:
            x = 0.0
            tol *= 2.0
            converged = False
        if x <= 0.0:
            converged = False
        if tol >= 1:
            return None

    if x <= 0.0:
        return None
    return x


def _scottrule1d(samples: np.ndarray):
    """
    _scottrule1d Compute the scotts rule for 1D samples

    Parameters
    ----------
    samples : np.ndarray
        the 1D samples for which the scott rule is being computed

    Returns
    -------
    float
        Scott's factor
    """
    sigma = np.std(samples, ddof=1)
    sigma_iqr = (
        np.quantile(samples, q=0.75) - np.quantile(samples, q=0.25)
    ) / 1.3489795003921634
    sigma = min(sigma, sigma_iqr)
    return sigma * np.power(len(samples), -1.0 / 5.0)


def kde1d(
    samples: np.ndarray,
    n: int = 2 ** 14,
    lower_bound: float = None,
    upper_bound: float = None,
):
    """
    kde1d Reliable and extremely fast kernel density estimator for one-dimensional data
    Gaussian kernel is assumed and the bandwidth is chosen automatically

    Notes
    -----
    Unlike many other implementations, this one is immune to problems
    caused by multimodal densities with widely separated modes (see example). The
    estimation does not deteriorate for multimodal densities, because we never assume
    a parametric model for the data.

    Example:
    
    .. code-block:: python
        
        import numpy as np
        from numpy.random import randn
        samples = np.concatenate((randn(100,1),randn(100,1)*2+35,randn(100,1)+55))
        kde1d(samples,2^14,min(samples)-5,max(samples)+5)

    Parameters
    ----------
    samples : np.ndarray
        the points from which the density estimate is computed
    n : int, optional
        the number of mesh points used in the uniform discretization of the
        interval [lower_bound, upper_bound]; n has to be a power of two;
        if n is not a power of two, then n is rounded up to the next power of two,
        i.e., n is set to n=2^ceil(log2(n)), by default 2**14
    lower_bound : float, optional
        the lower bound of the interval in which the density is being computed,
        if not given the default value is lower_bound=min(samples)-Range/10,
        where Range=max(samples)-min(samples), by default None
    upper_bound : float, optional
        the lower bound of the interval in which the density is being computed,
        if not given the default value is lower_bound=max(data)+Range/10,
        where Range=max(samples)-min(samples), by default None

    Returns
    -------
    density: np.ndarray
        column vector of length n with the values of the density
        estimate at the grid points;
    xmesh: np.ndarray
        the grid over which the density estimate is computed
    bandwidth: np.ndarray
        the optimal bandwidth (Gaussian kernel assumed)
    """
    samples = samples.ravel()  # make samples a 1D array
    n = np.int(2 ** np.ceil(np.log2(n)))  # round up to the next power of 2
    if lower_bound is None or upper_bound is None:
        minimum = np.min(samples)
        maximum = np.max(samples)
        delta = maximum - minimum
        if lower_bound is None:
            lower_bound = minimum - 0.1 * delta
        if upper_bound is None:
            upper_bound = maximum + 0.1 * delta

    delta = upper_bound - lower_bound
    xmesh = np.linspace(lower_bound, upper_bound, n)
    N = len(np.unique(samples))

    initial_data = _linear_binning(samples, xmesh)
    initial_data = initial_data / np.sum(initial_data)

    # Compute the Discrete Cosine Transform (DCT) of the data
    a = fftpack.dct(initial_data, type=2)

    # Compute the bandwidth
    irange_squared = np.arange(1, n, dtype=np.float64) ** 2.0
    a2 = a[1:] ** 2.0 / 4.0
    t_star = _root(_fixed_point, N, args=(N, irange_squared, a2))

    if t_star is None:
        # Automated bandwidth selection failed, use Scott's rule
        bandwidth = _scottrule1d(samples)
        t_star = (bandwidth / delta) ** 2.0
    else:
        bandwidth = np.sqrt(t_star) * delta

    # Smooth the discrete cosine transform of initial data using t_star
    a_t = a * np.exp(
        -np.arange(n, dtype=float) ** 2 * np.pi ** 2.0 * t_star / 2.0
    )

    # Diving by 2 because of the implementation of fftpack.idct
    density = fftpack.idct(a_t) / (2.0 * delta)
    density[density < 0] = 0.0  # remove negatives due to round-off error

    return density, xmesh, bandwidth