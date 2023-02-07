from textwrap import indent

import numpy as np

from pyvbmc.formatting import full_repr
from pyvbmc.priors import Prior, tile_inputs


class UniformBox(Prior):
    """Multivariate uniform-box prior.

    A prior distribution represented by box of uniform dimension, with lower
    bound(s) ``a`` and upper bound(s) ``b``.

    Attributes
    ----------
    D : int
        The dimension of the prior distribution.
    a : np.ndarray
        The lower bound(s), shape `(1, D)`.
    b : np.ndarray
        The upper bound(s), shape `(1, D)`.
    """

    def __init__(self, a, b, D=None):
        """Initialize a multivariate uniform-box prior.

        Parameters
        ----------
        a : np.ndarray | float
            The lower bound(s), shape `(D,)` where `D` is the dimension
            (parameters of type ``float`` will be tiled to this shape).
        b : np.ndarray | float
            The upper bound(s), shape `(D,)` where `D` is the dimension
            (parameters of type ``float`` will be tiled to this shape).

        Raises
        ------
        ValueError
            If ``a[i] >= b[i]``, for any `i`.
        """
        self.a, self.b = tile_inputs(a, b, size=D, squeeze=True)
        if np.any(self.a >= self.b):
            raise ValueError(
                f"All elements of a={a} should be strictly less than b={b}."
            )
        self.D = self.a.size

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
        log_norm_factor = np.sum(np.log(self.b - self.a))
        log_pdf = np.full((n, 1), -log_norm_factor)

        mask = np.any((x < self.a) | (x > self.b), axis=1)
        log_pdf[mask] = -np.inf

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
        return np.random.uniform(self.a, self.b, size=(n, self.D))

    @classmethod
    def _generic(cls, D=1):
        """Return a generic instance of the class (used for tests)."""
        return UniformBox(
            np.zeros(D),
            np.ones(D),
        )

    def _support(self):
        """Returns the support of the distribution.

        Used to test that the distribution integrates to one, so it is also
        acceptable to return a box which bounds the support of the
        distribution.

        Returns
        -------
        lb, ub : tuple(np.ndarray, np.ndarray)
            A tuple of lower and upper bounds of the support, such that
            [``lb[i]``, ``ub[i]``] bounds the support of the `i`th marginal.
        """
        return self.a, self.b

    def __str__(self):
        """Print a string summary."""
        return "UniformBox prior:" + indent(
            f"""
dimension = {self.D},
lower bounds = {self.a},
upper bounds = {self.b}""",
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
            "UniformBox",
            order=[
                "D",
                "a",
                "b",
            ],
            expand=expand,
            arr_size_thresh=arr_size_thresh,
        )
