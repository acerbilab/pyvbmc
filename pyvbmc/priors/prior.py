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

        parameters
        ----------
        x : np.ndarray
            The array of input point(s), of dimension `(D,)` or `(n, D)`, where
            `d` is the distribution dimension.
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


def tile_inputs(*args, size=None, squeeze=False):
    """Tile scalar inputs to have the same dimension as array inputs.

    If all inputs are given as scalars, returned arrays will have shape `size`
    if `size` is a tuple, or shape `(size,)` if `size` is an integer.

    Parameters
    ----------
    *args : [Union[float, np.ndarray]]
        The inputs to tile.
    size : Union[int, tuple], optional
        The desired size/shape of the output, default `(1,)`.
    squeeze : bool
        If `True`, then drop 1-d axes from inputs. Default `False`.

    Raises
    ------
    ValueError
        If the non-scalar arguments do not have the same shape, or if they do
        not agree with `size`.
    """
    if type(size) == int:
        size = (size,)
    shape = None

    # Check that all non-scalar inputs have the same shape
    args = list(args)
    for i, arg in enumerate(args):
        if not (np.isscalar(arg)):
            if squeeze:
                arg = args[i] = np.atleast_1d(np.squeeze(np.array(arg)))
            else:
                arg = args[i] = np.array(arg)
            if shape is None:
                shape = arg.shape
            elif arg.shape != shape:
                raise ValueError(
                    f"All inputs should have the same shape, but found inputs with shapes {shape} and {arg.shape}."
                )

    if size is None:
        if shape is None:
            # Default to shape (1,)
            size = (1,)
        else:
            # Or use inferred shape
            size = shape

    for i, arg in enumerate(args):
        if np.isscalar(arg):
            args[i] = np.full(size, arg)
        else:
            args[i] = args[i].reshape(size)

    return args
