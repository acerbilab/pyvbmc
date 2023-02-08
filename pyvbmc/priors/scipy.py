from textwrap import indent

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from scipy.stats._multivariate import (
    multivariate_normal_frozen,
    multivariate_t_frozen,
)

from pyvbmc.formatting import full_repr
from pyvbmc.priors import Prior


class SciPy(Prior):
    """Wrapper class for `scipy.stats` distributions.

    Attributes
    ----------
    D : int
        The dimension of the prior distribution.
    distribution : scipy.stats._multivariate.multivariate_normal_frozen\
            or scipy.stats._multivariate.multivariate_t_frozen\
            or scipy.stats._distn_infrastructure.rv_continuous_frozen
        The underlying `scipy.stats` distributions.
    """

    def __init__(self, distribution):
        """Initialize `pyvbmc.priors.Prior` from `scipy.stats` distribution(s).

        Parameters
        ----------
        distribution : scipy.stats.multivariate_normal\
                or scipy.stats.multivariate_t\
                or scipy.stats.rv_continuous\
            The underlying `scipy.stats` distributions. Should be a
            multivariate normal distribution, a multivariate t distribution, or
            a univariate continuous distribution (for 1-D models).

        Raises
        ------
        TypeError
            If the provided distribution is not of the appropriate type.
        """
        if isinstance(distribution, multivariate_normal_frozen) or isinstance(
            distribution, multivariate_t_frozen
        ):
            x = np.atleast_1d(distribution.rvs(1))
            self.D = len(x)
            self.a = np.full(self.D, -np.inf)
            self.b = np.full(self.D, np.inf)
        elif isinstance(distribution, rv_continuous_frozen):
            self.D = 1
            self.a = np.atleast_1d(distribution.a)
            self.b = np.atleast_1d(distribution.b)
        else:
            raise TypeError(
                f'A SciPy prior should be initialized from a "frozen" multivariate normal, multivariate t, or univariate SciPy distribution, but got `distribution` of type {type(distribution)}.'
            )
        self.distribution = distribution

    def _log_pdf(self, x):
        """Compute the log-pdf of the multivariate uniform-box prior.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension `(D,)` or `(n,D)`, where
            `D` is the distribution dimension.

        Returns
        -------
        log_pdf : np.ndarray
            The log-density of the prior at the input point(s), of dimension
            `(n, 1)`.
        """
        n, D = x.shape
        log_pdf = self.distribution.logpdf(x).reshape((n, 1))
        return log_pdf

    def sample(self, n):
        """Sample random variables from the uniform-box distribution.

        Parameters
        ----------
        n : int
            The number of points to sample.

        Returns
        -------
        rvs : np.ndarray
            The samples points, of shape `(n, D)`, where `D` is the dimension.
        """
        rvs = self.distribution.rvs(n).reshape((n, self.D))
        return rvs

    @classmethod
    def _generic(cls, D=1):
        """Return a generic instance of the class (used for tests)."""
        return SciPy(multivariate_normal(np.zeros(D)))

    def _support(self):
        """Returns the support of the distribution.

        Used to test that the distribution integrates to one, so it is also
        acceptable to return a box which bounds the support of the
        distribution.

        Returns
        -------
        a, b : tuple(np.ndarray, np.ndarray)
            A tuple of lower and upper bounds of the support, such that
            [``a[i]``, ``b[i]``] bounds the support of the `i`th marginal.
        """
        return self.a, self.b

    def __str__(self):
        """Print a string summary."""
        return "SciPy prior:" + indent(
            f"""
dimension = {self.D},
lower bounds = {self.a},
upper bounds = {self.b},
distribution(s) = {self.distribution}""",
            "    ",
        )

    def __repr__(self, arr_size_thresh=10, expand=False):
        """Construct a detailed string summary.

        Parameters
        ----------
        arr_size_thresh : float, optional
            If ``obj`` is an array whose product of dimensions is less than
            ``arr_size_thresh``, print the full array. Otherwise print only the
            shape. Default `10`.
        expand : bool, optional
            If ``expand`` is `False`, then describe any complex child
            attributes of the object by their name and memory location.
            Otherwise, recursively expand the child attributes into their own
            representations. Default `False`.

        Returns
        -------
        string : str
            The string representation of ``self``.
        """
        return full_repr(
            self,
            "SciPy",
            order=[
                "D",
                "a",
                "b",
                "distribution",
            ],
            expand=expand,
            arr_size_thresh=arr_size_thresh,
        )


def is_valid_scipy_dist(obj):
    """Assess whether ``obj`` is a valid SciPy distribution for a PyVBMC prior.

    A valid SciPy distribution is a frozen multivariate normal, multivariate t,
    or continuous univariate distribution.

    Parameters
    ----------
    obj : any
        The object to type-check.

    Returns
    -------
    is_valid : bool
    """
    return (
        isinstance(obj, multivariate_normal_frozen)
        or isinstance(obj, multivariate_t_frozen)
        or isinstance(obj, rv_continuous_frozen)
    )
