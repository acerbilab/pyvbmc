import numpy as np
from pytest import raises
from scipy.stats import (
    beta,
    lognorm,
    multivariate_normal,
    multivariate_t,
    norm,
)
from scipy.stats._distn_infrastructure import (
    rv_continuous_frozen as scipy_univariate,
)

from pyvbmc.priors import (
    Product,
    SciPy,
    SmoothBox,
    SplineTrapezoidal,
    Trapezoidal,
    UniformBox,
    UserFunction,
    convert_to_prior,
)

pyvbmc_priors = [
    UniformBox,
    Trapezoidal,
    SplineTrapezoidal,
    SmoothBox,
    SciPy,
    Product,
    UserFunction,
]
scipy_univariate_priors = [norm(), lognorm(1.0), beta(2, 1.5)]
scipy_multivariate_priors = [
    multivariate_normal(np.zeros(2)),
    multivariate_t(np.zeros(2), df=7),
]


def test_convert_to_prior():
    for prior in pyvbmc_priors:
        D = np.random.randint(1, 4)
        prior_in = prior._generic(D=D)
        prior_out = convert_to_prior(prior_in)
        assert prior_out is prior_in
        assert prior_out.D == D

    for prior_in in scipy_univariate_priors:
        prior_out = convert_to_prior(prior_in)
        assert isinstance(prior_out, SciPy)
        assert prior_out.distribution is prior_in
        assert prior_out.D == 1

    for prior_in in scipy_multivariate_priors:
        prior_out = convert_to_prior(prior_in)
        assert isinstance(prior_out, SciPy)
        assert prior_out.distribution is prior_in
        assert prior_out.D == 2

    prior_in = scipy_univariate_priors
    prior_out = convert_to_prior(prior_in)
    assert isinstance(prior_out, Product)
    assert prior_out.D == 3
    for (m, marginal) in enumerate(prior_out.marginals):
        assert isinstance(marginal, SciPy)
        assert isinstance(marginal.distribution, scipy_univariate)
        assert marginal.distribution is scipy_univariate_priors[m]
        assert marginal.D == 1

    prior_in = lambda x: np.sum(x)
    prior_out = convert_to_prior(log_prior=prior_in)
    assert isinstance(prior_out, UserFunction)
    assert prior_out.log_pdf is prior_in
    assert prior_out.sample is None
    assert prior_out.D is None

    sample_prior = lambda n: np.random.normal(size=n)
    prior_out = convert_to_prior(log_prior=prior_in, sample_prior=sample_prior)
    assert isinstance(prior_out, UserFunction)
    assert prior_out.log_pdf is prior_in
    assert prior_out.sample is sample_prior
    assert prior_out.D is None

    D = np.random.randint(1, 4)
    prior_out = convert_to_prior(None, prior_in, sample_prior, D=D)
    assert isinstance(prior_out, UserFunction)
    assert prior_out.log_pdf is prior_in
    assert prior_out.sample is sample_prior
    assert prior_out.D == D


def test_convert_to_prior_error_handling():
    prior_in = 1.0
    with raises(TypeError) as err:
        prior_out = convert_to_prior(prior_in)
    assert (
        "Optional keyword `prior` should be a subclass of `pyvbmc.priors.Prior`, an appropriate `scipy.stats` distribution,"
        in err.value.args[0]
    )

    prior_in = UniformBox._generic()
    log_prior = lambda x: np.sum(x)
    with raises(ValueError) as err:
        prior_out = convert_to_prior(prior_in, log_prior)
    assert (
        "If `prior` is provided then `log_prior` should be `None` or `prior.log_pdf`."
        in err.value.args[0]
    )
    sample_prior = lambda n: np.random.normal(size=n)
    with raises(ValueError) as err:
        prior_out = convert_to_prior(prior_in, sample_prior=sample_prior)
    assert (
        "If `prior` is provided then `sample_prior` should be `None` or `prior.sample`."
        in err.value.args[0]
    )
