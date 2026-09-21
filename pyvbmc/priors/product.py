from textwrap import indent

import numpy as np

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
from pyvbmc.priors.user_function import UserFunction
from pyvbmc.rng import get_rng


class Product(Prior):
    """A prior which is an product of independent univariate priors.

    Attributes
    ----------
    D : int
        The dimension of the product distribution.
    marginals : pyvbmc.priors.Prior
        The underlying marginal prior distribution(s).
    a : np.ndarray
        The lower bound(s) of the support, shape `(D,)`, read from the
        marginals.
    b : np.ndarray
        The upper bound(s) of the support, shape `(D,)`, read from the
        marginals.
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
        self.marginals = []
        for m, marginal in enumerate(marginals):
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
        for m, marginal in enumerate(self.marginals):
            if isinstance(marginal, UserFunction):
                # `log_pdf` is the user's own callable, which takes one
                # point as a one-dimensional array and returns its
                # log-density.
                for row in range(n):
                    log_pdf[row, m] = marginal.log_pdf(x[row, m : m + 1])
            else:
                log_pdf[:, m] = marginal.log_pdf(x[:, m], keepdims=False)
        log_pdf = np.sum(log_pdf, axis=1, keepdims=True)
        return log_pdf

    def sample(self, n, rng=None):
        """Sample random variables from the uniform-box distribution.

        Parameters
        ----------
        n : int
            The number of points to sample.
        rng : None, int, SeedSequence or Generator, optional
            Random generator or seed; if None a generator is derived from
            NumPy's global random state. The same generator is shared by all
            marginals, so the whole product draws from one stream.

        Returns
        -------
        rvs : np.ndarray
            The samples points, of shape `(n, D)`, where `D` is the dimension.
        """
        rng = get_rng(rng)
        rvs = np.zeros((n, self.D))
        for m, marginal in enumerate(self.marginals):
            if isinstance(marginal, UserFunction):
                # `sample` is the user's own callable and takes only `n`.
                rvs[:, m] = marginal.sample(n).ravel()
            else:
                rvs[:, m] = marginal.sample(n, rng=rng).ravel()
        return rvs

    @classmethod
    def _generic(cls, D=1, rng=None):
        """Return a generic instance of the class (used for tests).

        The class of each marginal is drawn from ``rng`` (a generator or a
        seed; if None a generator is derived from NumPy's global random
        state).
        """
        rng = get_rng(rng)
        classes = [
            UniformBox,
            Trapezoidal,
            SplineTrapezoidal,
            SmoothBox,
            SciPy,
        ]
        return Product(
            [
                classes[rng.integers(len(classes))]._generic(1)
                for __ in range(D)
            ]
        )

    def _support_box(self):
        """The box of the support, read from ``self.marginals``.

        The box is read at every call, so an object restored from a file
        describes the marginals it carries whatever the file stored beside
        them.

        Returns
        -------
        a, b : tuple(np.ndarray, np.ndarray)
            The lower and upper bounds of the support, each of shape `(D,)`
            and of dtype `float64`.
        """
        a = np.empty(self.D, dtype=np.float64)
        b = np.empty(self.D, dtype=np.float64)
        for m, marginal in enumerate(self.marginals):
            a_m, b_m = marginal.support()
            a[m] = np.asarray(a_m, dtype=np.float64).item()
            b[m] = np.asarray(b_m, dtype=np.float64).item()
        return a, b

    @property
    def a(self):
        """np.ndarray: The lower bound(s) of the support, shape `(D,)`."""
        return self._support_box()[0]

    @property
    def b(self):
        """np.ndarray: The upper bound(s) of the support, shape `(D,)`."""
        return self._support_box()[1]

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
        return self._support_box()

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
