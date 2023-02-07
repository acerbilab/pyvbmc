from pyvbmc.priors import Prior, Product, SciPy, UserFunction


def convert_to_prior(prior, sample_prior=None, D=None):
    if isinstance(prior, list):
        prior = Product(prior)
    elif callable(prior):
        prior = UserFunction(prior, sample_prior, D)
    elif isinstance(prior, Prior):
        pass
    else:
        try:
            prior = SciPy(prior)
        except TypeError as err:
            raise TypeError(
                f"Optional keyword `prior` should be a subclass of `pyvbmc.priors.Prior`, an appropriate `scipy.stats` distribution, a list of these, or a function. ({err})"
            ) from err
    if sample_prior is not None and sample_prior != prior.sample:
        raise ValueError(
            "If `prior` is provided then `sample_prior` should be `None` or `prior.sample`."
        )
    return prior
