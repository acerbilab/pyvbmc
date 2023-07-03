from abc import ABC, abstractmethod

import numpy as np


class Prior(ABC):
    """Abstract base class for PyVBMC prior distributions."""

    def log_pdf(self, x, keepdims=True):
        """Compute the log-pdf of the distribution.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension `(D,)` or `(n, D)`, where
            `d` is the distribution dimension.
        keepdims : bool
            Whether to keep the input dimensions and return an array of shape
            `(1, D)`, or discard them and return an array of shape `(D,)`.

        Returns
        -------
        log_pdf : np.ndarray
            The log-density of the prior at the input point(s), of dimension
            `(n, 1)` or `(n,)` (depending on ``keepdims``).
        """
        x_orig_shape = x.shape
        x = np.atleast_2d(x)
        n, D = x.shape
        if self.D == 1 and n == 1:
            x = x.T
        elif D != self.D:
            raise ValueError(
                f"x should have shape ({self.D},) or (n, {self.D}) but has shape {x_orig_shape}."
            )
        log_pdf = self._log_pdf(x)
        if keepdims:
            return log_pdf
        else:
            return log_pdf.ravel()

    def pdf(self, x, keepdims=True):
        """Compute the pdf of the distribution.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension `(D,)` or `(n, D)`, where
            `D` is the distribution dimension.
        keepdims : bool
            Whether to keep the input dimensions and return an array of shape
            `(1, D)`, or discard them and return an array of shape `(D,)`.

        returns
        -------
        pdf : np.ndarray
            The density of the prior at the input point(s), of dimension `(n,
            1)` or `(n,)` (depending on ``keepdims``).
        """
        return np.exp(self.log_pdf(x, keepdims=keepdims))

    def support(self):
        r"""Returns the support of the distribution.

        Used to test that the distribution integrates to one, so it is also
        acceptable to return a box which bounds the support of the
        distribution.

        Returns
        -------
        lb, ub : tuple(np.ndarray, np.ndarray)
            A tuple of lower and upper bounds of the support, such that
            [``lb[i]``, ``ub[i]``] bounds the support of the ``i``\ th marginal.
        """
        if hasattr(self, "_support"):
            return self._support()
        else:
            return np.full(self.D, -np.inf), np.full(self.D, np.inf)

    @abstractmethod
    def __init__(self):
        """Initialize the distribution."""
        self.D = 1

    @abstractmethod
    def _log_pdf(self, x):
        """Compute the log-pdf of the distribution.

        This private method is wrapped by ``self.log_pdf()``, which handles
        input validation and output shape.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension `(D,)` or `(n, D)`, where
            `d` is the distribution dimension.

        Returns
        -------
        log_pdf : np.ndarray
            The log-density of the prior at the input point(s), of dimension
            `(n, 1)`.
        """
        pass

    @abstractmethod
    def sample(self, n):
        """Sample random variables from the distribution.

        Parameters
        ----------
        n : int
            The number of points to sample.

        Returns
        -------
        rvs : np.ndarray
            The samples points, of shape `(n, D)`, where `D` is the dimension.
        """
        pass

    @classmethod
    @abstractmethod
    def _generic(cls, D=1):
        """Return a generic instance of the class (used for tests)."""
        return cls(D=D)
