from textwrap import indent

import numpy as np

from pyvbmc.formatting import full_repr
from pyvbmc.priors import Prior, tile_inputs


class Trapezoidal(Prior):
    r"""Multivariate trapezoidal prior.

    A prior distribution represented by a density with external bounds ``a``
    and ``b`` and internal points ``u`` and ``v``. Each marginal distribution
    has a trapezoidal density which is uniform between ``u[i]`` and ``v[i]``
    and falls of linearly to zero at ``a[i]`` and ``b[i]``::

                 ______________________
                |       ________       |
                |      /|      |\      |
        p(X(i)) |     / |      | \     |
                |    /  |      |  \    |
                |___/___|______|___\___|
                  a[i] u[i]  v[i] b[i]
                          X(i)

    The overall density is a product of these marginals.

    Attributes
    ----------
    D : int
        The dimension of the prior distribution.
    a : np.ndarray
        The lower bound(s), shape `(1, D)`.
    u : np.ndarray
        The lower pivot(s), shape `(1, D)`.
    v : np.ndarray
        The upper pivot(s), shape `(1, D)`.
    b : np.ndarray
        The upper bound(s), shape `(1, D)`.
    """

    def __init__(self, a, u, v, b, D=None):
        """Initialize a multivariate trapezoidal prior.

        Parameters
        ----------
        a : np.ndarray | float
            The lower bound(s), shape `(D,)` where `D` is the dimension
            (parameters of type ``float`` will be tiled to this shape).
        u : np.ndarray | float
            The lower pivot(s), shape `(D,)` where `D` is the dimension
            (parameters of type ``float`` will be tiled to this shape).
        v : np.ndarray | float
            The upper pivot(s), shape `(D,)` where `D` is the dimension
            (parameters of type ``float`` will be tiled to this shape).
        b : np.ndarray | float
            The upper bound(s), shape `(D,)` where `D` is the dimension
            (parameters of type ``float`` will be tiled to this shape).
        D : int, optional
            The distribution dimension. If given, will convert scalar `a`, `u`,
            `v`, and `b` to this dimension.

        Raises
        ------
        ValueError
            If the order ``a[i] < u[i] < v[i] < b[i]`` is not respected, for any `i`.
        """
        self.a, self.u, self.v, self.b = tile_inputs(
            a, u, v, b, size=D, squeeze=True
        )
        if np.any(
            (self.a >= self.u) | (self.u >= self.v) | (self.v >= self.b)
        ):
            raise ValueError(
                "Bounds and pivots should respect the order a < u < v < b."
            )
        self.D = self.a.size

    def _log_pdf(self, x):
        """Compute the log-pdf of the multivariate trapezoidal prior.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension `(D,)` or `(n, D)`, where
            `D` is the distribution dimension.

        Returns
        -------
        log_pdf : np.ndarray
            The log-density of the prior at the input point(s), of dimension
            `(n, 1)`.
        """
        n, D = x.shape
        log_pdf = np.full_like(x, -np.inf)
        log_norm_factor = np.log(0.5) + np.log(
            self.b - self.a + self.v - self.u
        )

        for d in range(D):
            # Left tail
            mask = (x[:, d] >= self.a[d]) & (x[:, d] < self.u[d])
            log_pdf[mask, d] = (
                np.log(x[mask, d] - self.a[d])
                - np.log(self.u[d] - self.a[d])
                - log_norm_factor[d]
            )

            # Plateau
            mask = (x[:, d] >= self.u[d]) & (x[:, d] < self.v[d])
            log_pdf[mask, d] = -log_norm_factor[d]

            # Right tail
            mask = (x[:, d] >= self.v[d]) & (x[:, d] < self.b[d])
            log_pdf[mask, d] = (
                np.log(self.b[d] - x[mask, d])
                - np.log(self.b[d] - self.v[d])
                - log_norm_factor[d]
            )

        return np.sum(log_pdf, axis=1, keepdims=True)

    def sample(self, n):
        """Sample random variables from the trapezoidal distribution.

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

        # Sample one dimension at a time
        for d in range(self.D):
            one_d_dist = Trapezoidal(
                self.a[d], self.u[d], self.v[d], self.b[d]
            )
            # Compute maximum of one-dimensional pdf
            x0 = 0.5 * (self.u[d] + self.v[d])
            y_max = one_d_dist.pdf(x0, keepdims=False)

            mask = np.full(n, True)
            r1 = np.zeros(n)
            n1 = n

            # Rejection sampling
            while n1 > 0:
                # Uniform sampling in the bounding box
                r1[mask] = np.random.uniform(self.a[d], self.b[d], size=n1)

                # Rejection sampling
                z1 = np.random.uniform(0.0, y_max, size=n1)
                y1 = one_d_dist.pdf(r1[mask].reshape(-1, 1), keepdims=False)

                mask_new = np.full(n, False)
                mask_new[mask] = z1 > y1  # Resample these points

                mask = mask_new
                n1 = np.sum(mask)

            # Assign d-th dimension
            rvs[:, d] = r1

        return rvs

    @classmethod
    def _generic(cls, D=1):
        """Return a generic instance of the class (used for tests)."""
        return Trapezoidal(
            np.zeros(D),
            np.full((D,), 0.25),
            np.full((D,), 0.75),
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
        return "Trapezoidal prior:" + indent(
            f"""
dimension = {self.D},
lower bounds = {self.a},
lower pivots = {self.u},
upper pivots = {self.v},
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
            "Trapezoidal",
            order=[
                "D",
                "a",
                "u",
                "v",
                "b",
            ],
            expand=expand,
            arr_size_thresh=arr_size_thresh,
        )
