from collections.abc import Iterable
from textwrap import indent

import numpy as np
from scipy.stats._distn_infrastructure import (
    rv_continuous_frozen as scipy_univariate,
)

from pyvbmc.formatting import full_repr
from pyvbmc.priors import (
    Prior,
    SciPy,
    SmoothBox,
    SplineTrapezoidal,
    Trapezoidal,
    UniformBox,
    is_valid_scipy_dist,
)


class Product(Prior):
    """A prior which is an product of independent univariate priors.

    Attributes
    ----------
    D : int
        The dimension of the product distribution.
    marginals : pyvbmc.priors.Prior
        The underlying marginal prior distribution(s).
    """

    def __init__(self, marginals):
        """Initialize a `Product` prior from a list of marginal priors.

        Parameters
        ----------
        marginals : list of pyvbmc.priors.Prior
            The underlying marginal prior distribution(s). Each must have
            dimension 1.

        Raises
        ------
        TypeError
            If the provided marginals are not a list, or if they are not all
            PyVBMC priors or appropriate SciPy distributions.
        ValueError
            If the provided marginals are not all univariate.
        """
        if not isinstance(marginals, list):
            raise TypeError(
                f"`Product` should be initialized from a list of distributions, but received type {type(marginals)}."
            )
        self.D = len(marginals)
        self.a = np.full(self.D, -np.inf)
        self.b = np.full(self.D, np.inf)
        self.marginals = []
        for (m, marginal) in enumerate(marginals):
            if is_valid_scipy_dist(marginal):
                marginal = SciPy(marginal)
            elif not isinstance(marginal, Prior):
                raise TypeError(
                    f"All marginals should be subclasses of `pyvbmc.priors.Prior`, or valid continuous SciPy distributions, but found type {type(marginal)}."
                )
            if marginal.D != 1:
                raise ValueError(
                    f"All marginals of a product distribution should have dimension 1, but marginal {marginal} has dimension {marginal.D}"
                )
            a_m, b_m = marginal.support()
            self.a[m], self.b[m] = a_m.item(), b_m.item()
            self.marginals.append(marginal)

    def _log_pdf(self, x):
        """Compute the log-pdf of the multivariate uniform-box prior.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension `(n,D)`, where `D` is the
            distribution dimension.

        Returns
        -------
        log_pdf : np.ndarray
            The log-density of the prior at the input point(s), of dimension
            `(n, 1)`.
        """
        n, D = x.shape
        log_pdf = np.zeros((n, D))
        for (m, marginal) in enumerate(self.marginals):
            log_pdf[:, m] = marginal.log_pdf(x[:, m], keepdims=False)
        log_pdf = np.sum(log_pdf, axis=1, keepdims=True)
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
        rvs = np.zeros((n, self.D))
        for (m, marginal) in enumerate(self.marginals):
            rvs[:, m] = marginal.sample(n).ravel()
        return rvs

    @classmethod
    def _generic(cls, D=1):
        """Return a generic instance of the class (used for tests)."""
        return Product(
            [
                np.random.choice(
                    [
                        UniformBox,
                        Trapezoidal,
                        SplineTrapezoidal,
                        SmoothBox,
                        SciPy,
                    ]
                )._generic(1)
                for __ in range(D)
            ]
        )

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
        return "Product prior:" + indent(
            f"""
dimension = {self.D},
lower bounds = {self.a},
upper bounds = {self.b},
marginals = {self.marginals}""",
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
            "Product",
            order=[
                "D",
                "a",
                "b",
                "marginals",
            ],
            expand=expand,
            arr_size_thresh=arr_size_thresh,
        )
