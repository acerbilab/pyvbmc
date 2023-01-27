from abc import ABC, abstractmethod

import numpy as np


class Prior(ABC):
    """Abstract base class for PyVBMC prior distributions."""

    def logpdf(self, x, keepdims=True):
        x_orig_shape = x.shape
        x = np.atleast_2d(x)
        n, D = x.shape
        if D != self.D:
            raise ValueError(
                f"Shape of x should have shape ({self.D},) or (n, {self.D}) but has shape {x_orig_shape}."
            )
        logpdf = self._logpdf(x)
        if keepdims:
            return logpdf
        else:
            return logpdf.ravel()

    def pdf(self, x, keepdims=True):
        """Compute the pdf of the distribution.

        parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension ``(D,)`` or ``(n,D)``, where
            ``d`` is the distribution dimension.
        keepdims : bool
            Keep the dimensions as-is and return a ``(n,1)`` vector of
            densities if ``true`` (default), otherwise return vector of shape
            ``(n,)``.

        returns
        -------
        pdf : np.ndarray
            The density of the prior at the input point(s), of dimension
            ``(n,1)``.
        """
        return np.exp(self.logpdf(x, keepdims=keepdims))

    def _support(self):
        return np.full(self.D, -np.inf), np.full(self.D, np.inf)

    @abstractmethod
    def __init__(self):
        """Initialize the distribution."""
        self.D = 1

    @abstractmethod
    def _logpdf(self, x):
        """Compute the log-pdf of the distribution.

        Parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension ``(D,)`` or ``(n,D)``, where
            ``d`` is the distribution dimension.

        Returns
        -------
        logpdf : np.ndarray
            The log-density of the prior at the input point(s), of dimension
            ``(n,1)``.
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
            The samples points, of shape ``(n, D)``, where ``D`` is the dimension.
        """
        pass

    @classmethod
    @abstractmethod
    def _generic(cls, D=1):
        return cls(D=D)


def tile_inputs(*args, size=None):
    """Tile scalar inputs to have the same dimension as array inputs.

    If all inputs are given as scalars, returned arrays will have shape `size`
    if `size` is a tuple, or shape `(size,)` if `size` is an integer.

    Parameters
    ----------
    *args : [Union[float, np.ndarray]]
        The inputs to tile.
    size : Union[int, tuple], optional
        The desired size/shape of the output, default `(1,)`.

    Raises
    ------
    ValueError
        If the non-scalar arguments do not have the same shape, or if they do not agree with `size`.
    """
    if type(size) == int:
        size = (size,)
    shape = None

    # Check that all non-scalar inputs have the same shape
    args = list(args)
    for i, arg in enumerate(args):
        if not (np.isscalar(arg)):
            arg = args[i] = np.array(arg)
            if shape is None:
                shape = arg.shape
            elif arg.shape != shape:
                raise ValueError(
                    f"All inputs should have the same shape, but found inputs with shapes {shape} and {arg.shape}."
                )
    # Check that size agrees with input shape
    if size is not None and shape is not None and shape != size:
        raise ValueError(
            f"Requested shape {size} but some arguments have shape {shape}."
        )
    if shape is None:
        # Default to shape (1,)
        if size is None:
            shape = (1,)
        # Or use provided size
        else:
            shape = size

    for i, arg in enumerate(args):
        if np.isscalar(arg):
            args[i] = np.full(shape, arg)

    return args
