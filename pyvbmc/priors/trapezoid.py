import numpy as np

from pyvbmc.priors import Prior, tile_inputs


class Trapezoid(Prior):
    """Multivariate trapezoid prior.

    A prior distribution represented by a trapezoidal density with external
    bounds ``a`` and ``b`` and internal points ``u`` and ``v``. Each marginal
    distribution has a trapezoidal density which is uniform between ``u[i]``
    and ``v[i]`` and falls of linearly to ``a[i]`` and ``b[i]``::

                |      ________
                |     /|      |\
        p(X(i)) |    / |      | \
                |   /  |      |  \
                |__/___|______|___\__
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
        """Initialize a multivariate trapezoid prior.

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
        if np.any((a >= u) | (u >= v) | (v >= b)):
            raise ValueError(
                "Bounds and pivots should respect the order a < u < v < b."
            )

        self.a, self.b = tile_inputs(a, b, size=D)
        self.D = self.a.shape[1]

    def _logpdf(self, x):
        """Compute the log-pdf of the multivariate trapezoid prior.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension ``(D,)`` or ``(n,D)``, where
            ``D`` is the distribution dimension.

        Returns
        -------
        logpdf : np.ndarray
            The log-density of the prior at the input point(s), of dimension
            ``(n,1)``.
        """
        n, D = x.shape
        logpdf = np.full_like(x, -np.inf)
        log_norm_factor = (
            np.log(0.5)
            + np.log(self.b - self.a + self.v - self.u)
            + np.log(self.u - self.a)
        )

        for d in range(D):
            # Left tail
            mask = (x[:, d] >= self.a[d]) & (x[:, d] < self.u[d])
            logpdf[mask, d] = (
                np.log(x[mask, d] - self.a[d]) - log_norm_factor[d]
            )

            # Plateau
            mask = (x[:, d] >= self.u[d]) & (x[:, d] < self.v[d])
            logpdf[mask, d] = (
                np.log(self.u[d] - self.a[d]) - log_norm_factor[d]
            )

            # Right tail
            mask = (x[:, d] >= self.v[d]) & (x[:, d] < self.b[d])
            logpdf[mask, d] = (
                np.log(self.b[d] - x[mask, d])
                - np.log(self.b[d] - self.v[d])
                + np.log(self.u[d] - self.a[d])
                - log_norm_factor[d]
            )

        return np.sum(logpdf, axis=1, keepdims=True)

    def sample(self, n):
        """Sample random variables from the trapezoid distribution.

        Parameters
        ----------
        n : int
            The number of points to sample.

        Returns
        -------
        rvs : np.ndarray
            The samples points, of shape ``(n, D)``, where ``D`` is the dimension.
        """
        rvs = np.zeros((n, self.D))

        # Sample one dimension at a time
        for d in range(self.D):
            one_d_dist = Trapezoid(self.a[d], self.u[d], self.v[d], self.b[d])
            # Compute maximum of one-dimensional pdf
            x0 = 0.5 * (self.u[d] + self.v[d])
            y_max = one_d_dist.pdf(x0, keepdims=False)

            mask = np.full(n, True)
            r1 = np.zeros(n)
            n1 = n

            # Rejection sampling
            while n1 > 0:
                # Uniform sampling in the bounding box
                r1[mask] = np.random.uniform(self.a, self.b, size=n1)

                # Rejection sampling
                z1 = np.random.uniform(0.0, y_max[mask], size=(n1, 1))
                y1 = one_d_dist.pdf(r1[mask])

                mask_new = np.full(n, False)
                mask_new[mask] = z1 > y1  # Resample these points

                mask = mask_new
                n1 = np.sum(mask)

            # Assign d-th dimension
            rvs[:, d] = r1

        return rvs
