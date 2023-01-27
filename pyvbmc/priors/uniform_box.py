import numpy as np

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
        self.a, self.b = tile_inputs(a, b, size=D)
        if np.any(self.a >= self.b):
            raise ValueError(
                f"All elements of a={a} should be strictly less than b={b}."
            )
        self.D = self.a.size

    def _logpdf(self, x):
        """Compute the log-pdf of the multivariate uniform-box prior.

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
        log_norm_factor = np.sum(np.log(self.b - self.a))
        logpdf = np.full((n, 1), -log_norm_factor)

        mask = np.any((x < self.a) | (x > self.b), axis=1)
        logpdf[mask] = -np.inf

        return logpdf

    def sample(self, n):
        """Sample random variables from the uniform-box distribution.

        Parameters
        ----------
        n : int
            The number of points to sample.

        Returns
        -------
        rvs : np.ndarray
            The samples points, of shape ``(n, D)``, where ``D`` is the dimension.
        """
        return np.random.uniform(self.a, self.b, size=(n, self.D))

    @classmethod
    def _generic(cls, D=1):
        return UniformBox(
            np.zeros(D),
            np.ones(D),
        )

    def _support(self):
        return self.a, self.b
