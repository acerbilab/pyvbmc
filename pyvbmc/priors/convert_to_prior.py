from pyvbmc.priors import (
    Prior,
    Product,
    SciPy,
    UserFunction,
    is_valid_scipy_dist,
)


def convert_to_prior(prior=None, log_prior=None, sample_prior=None, D=None):
    """Convert an object to a pyvbmc Prior instance.

    Parameters
    ----------
    prior, optional
        The object to convert to a PyVBMC ``Prior``. May be one of:

        #. a ``PyVBMC`` prior, which will remain unchanged,
        #. a frozen SciPy multivariate normal, multivariate t, or
           one-dimensional continuous distribution, which will be converted to
           a PyVBMC ``SciPy`` prior, or
        #. a list of one-dimensional continuous SciPy distributions and/or
           PyVBMC ``Prior`` objects, which will be treated as independent
           marginals of a ``Product`` prior.
        #. `None`, in which case `log_prior` (and optionally `sample_prior`)
           are used to build a user-defined `Prior`.

    sample_prior : callable, optional
        A function of a single argument `x` which returns the log-density of
        the prior at `x`. Optional, should agree with `prior.log_pdf` if both
        are provided.
    sample_prior : callable, optional
        A function of a single argument `n` which returns `n` samples from the
        prior distribution. Optional, should agree with `prior.sample` if both
        are provided.
    D : int, optional
        The dimension of the prior distribution. Optional, used only for
        user-defined priors.

    Returns
    -------
    prior : PyVBMC.priors.Prior
    """
    if isinstance(prior, list):
        prior = Product(prior)
    elif isinstance(prior, Prior):
        pass
    elif is_valid_scipy_dist(prior):
        prior = SciPy(prior)
    elif prior is None and (callable(log_prior) or callable(sample_prior)):
        prior = UserFunction(log_prior, sample_prior, D)
    elif callable(prior):
        raise TypeError(
            f"Optional keyword `prior` should be a subclass of `pyvbmc.priors.Prior`, an appropriate `scipy.stats` distribution, or a list of these, not a function. Perhaps you meant to use `log_prior` or `sample_prior`?"
        )
    else:
        raise TypeError(
            f"Optional keyword `prior` should be a subclass of `pyvbmc.priors.Prior`, an appropriate `scipy.stats` distribution, or a list of these. Optional keyword `log_prior` should be a function."
        )
    if (sample_prior) is not None and (sample_prior != prior.sample):
        raise ValueError(
            "If `prior` is provided then `sample_prior` should be `None` or `prior.sample`."
        )
    if (log_prior is not None) and (log_prior != prior.log_pdf):
        raise ValueError(
            "If `prior` is provided then `log_prior` should be `None` or `prior.log_pdf`."
        )
    return prior
