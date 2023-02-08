from textwrap import indent

import numpy as np
from scipy.stats import multivariate_normal

from pyvbmc.formatting import full_repr
from pyvbmc.priors import Prior


class UserFunction(Prior):
    """Lightweight wrapper for user-defined priors.

    Attributes
    ----------
    log_pdf : callable
        The user-provided function representing the log-density of the prior.
    sample : callable or None
        A function for sampling from the prior, if provided.
    D : int or None
        The dimension of the prior, if provided.
    """

    def __init__(self, log_prior, sample_prior=None, D=None):
        """Initialize a user-specified prior from a function.

        Parameters
        ----------
        log_prior : callable, optional
            The user-provided function. Should take a one-dimensional array as
            a single argument, and return the log-density of the prior
            evaluated at that point.
        sample_prior : callable, optional
            An optional user-provided function for sampling form the prior.
            Should take an integer `n` as a single argument, and return `n`
            samples from the prior distribution as an `(n, D)` array.
        D : int, optional
            Specified dimension of the prior (optional).
        """
        self.D = D
        if (log_prior is not None) and (not callable(log_prior)):
            raise TypeError("`log_prior` must be callable.")
        self.log_pdf = log_prior
        if (sample_prior is not None) and (not callable(sample_prior)):
            raise TypeError(
                f"Optional keyword `sample_prior` must be callable."
            )
        self.sample = sample_prior

    def pdf(self, *args, **kwargs):
        """Compute the pdf of the distribution."""
        return np.exp(self.log_pdf(*args, **kwargs))

    @classmethod
    def _generic(cls, D=1):
        """Return a generic instance of the class (used for tests)."""
        log_prior = lambda x: multivariate_normal(np.zeros(D)).logpdf(x)
        return cls(log_prior, D=D)

    def sample(self, n):
        """Unused"""
        pass

    def _log_pdf(self):
        """Unused"""
        pass

    def __str__(self):
        """Print a string summary."""
        return "UserFunction prior:" + indent(
            f"""
dimension = {self.D},
log pdf = {self.log_pdf},
sample function = {self.sample}""",
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
            "UserFunction",
            order=[
                "D",
                "log_pdf",
                "sample",
            ],
            expand=expand,
            arr_size_thresh=arr_size_thresh,
        )
