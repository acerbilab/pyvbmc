import numpy as np

from pyvbmc.priors import Prior, tile_inputs


class SmoothBox(Prior):
    """Multivariate smooth-box prior.

    For each dimension `i`, the univariate smooth-box pdf is defined as a
    uniform distribution between pivots ``a[i]``, ``b[i]`` with Gaussian tails
    that fall starting from `p(a[i])` to the left (resp., `p(b[i])` to the
    right) with standard deviation ``sigma[i]``.

    Attributes
    ----------
    D : int
        The dimension of the prior distribution.
    a : np.ndarray
        The lower pivot(s), shape `(1, D)`.
    b : np.ndarray
        The upper pivot(s), shape `(1, D)`.
    sigma : np.ndarray
        The standard deviation of the Gaussian tails, shape `(1, D)`.
    """

    def __init__(self, a, b, sigma, D=None):
        """Initialize a multivariate smooth-box prior.

        Parameters
        ----------
        a : np.ndarray | float
            The lower pivot(s), shape `(D,)` where `D` is the dimension
            (parameters of type ``float`` will be tiled to this shape).
        b : np.ndarray | float
            The upper pivot(s), shape `(D,)` where `D` is the dimension
            (parameters of type ``float`` will be tiled to this shape).
        sigma : np.ndarray
            The standard deviation of the Gaussian tails, shape `(D,)` where
            `D` is the dimension (parameters of type ``float`` will be tiled to
            this shape).

        Raises
        ------
        ValueError
            If ``sigma[i] <= 0`` or if ``a[i] >= b[i]``, for any `i`.
        """
        self.a, self.b, self.sigma = tile_inputs(a, b, sigma, size=D)
        if np.any(self.sigma <= 0.0):
            raise ValueError(
                f"All elements of sigma={sigma} should be positive."
            )
        if np.any(self.a >= self.b):
            raise ValueError(
                f"All elements of a={a} should be strictly less than b={b}."
            )
        self.D = self.a.size

    def _log_pdf(self, x):
        """Compute the log-pdf of the multivariate smooth-box prior.

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
        log_pdf = np.full_like(x, -np.inf)
        log_norm_factor = -np.log(np.sqrt(2 * np.pi) * self.sigma) - np.log1p(
            (self.b - self.a) / (np.sqrt(2 * np.pi) * self.sigma)
        )

        for d in range(self.D):
            mask = x[:, d] < self.a[d]
            log_pdf[mask, d] = (
                log_norm_factor[d]
                - 0.5 * ((x[mask, d] - self.a[d]) / self.sigma[d]) ** 2
            )

            mask = (x[:, d] >= self.a[d]) & (x[:, d] <= self.b[d])
            log_pdf[mask, d] = log_norm_factor[d]

            mask = x[:, d] > self.b[d]
            log_pdf[mask, d] = (
                log_norm_factor[d]
                - 0.5 * ((x[mask, d] - self.b[d]) / self.sigma[d]) ** 2
            )

        return np.sum(log_pdf, axis=1, keepdims=True)

    def sample(self, n):
        """Sample random variables from the smooth-box distribution.

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
        norm_factor = 1 + 1 / np.sqrt(2 * np.pi) * (
            (self.b - self.a) / self.sigma
        )

        for d in range(self.D):
            # Draw component (left/right tails or plateau)
            u = np.random.uniform(0.0, norm_factor[d], size=n)

            # Left Gaussian tails
            mask = u < 0.5
            if np.any(mask):
                z1 = np.abs(
                    np.random.normal(0.0, self.sigma[d], size=np.sum(mask))
                )
                rvs[mask, d] = self.a[d] - z1

            # Right Gaussian tails
            mask = (u >= 0.5) & (u < 1.0)
            if np.any(mask):
                z1 = np.abs(
                    np.random.normal(0.0, self.sigma[d], size=np.sum(mask))
                )
                rvs[mask, d] = self.b[d] + z1

            # Plateau
            mask = u >= 1.0
            if np.any(mask):
                rvs[mask, d] = np.random.uniform(
                    self.a[d], self.b[d], size=np.sum(mask)
                )

        return rvs

    @classmethod
    def _generic(cls, D=1):
        """Return a generic instance of the class (used for tests)."""
        return SmoothBox(
            np.zeros(D),
            np.ones(D),
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
        return (
            np.full_like(self.a, -np.inf),
            np.full_like(self.b, np.inf),
        )