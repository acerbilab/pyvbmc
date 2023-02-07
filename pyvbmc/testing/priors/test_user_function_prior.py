import numpy as np
from pytest import raises
from scipy.special import gamma
from scipy.stats import (
    beta,
    binom,
    lognorm,
    multivariate_normal,
    multivariate_t,
    norm,
    t,
)

from pyvbmc.priors import UserFunction


def test_function_prior():
    log_prior = lambda x: np.sum(x)
    prior = UserFunction(log_prior)
    assert prior.D is None
    assert prior.log_pdf is log_prior
    assert prior.sample is None
    x = np.random.normal(size=10)
    assert np.isclose(prior.log_pdf(x), np.log(prior.pdf(x)))
    assert np.isclose(prior.pdf(x), np.prod(np.exp(x)))

    sample_prior = lambda n: np.random.normal(size=n)
    prior = UserFunction(log_prior, sample_prior)
    assert prior.D is None
    assert prior.log_pdf is log_prior
    assert prior.sample is sample_prior
    x = np.random.normal(size=10)
    assert np.isclose(prior.log_pdf(x), np.log(prior.pdf(x)))
    assert np.isclose(prior.pdf(x), np.prod(np.exp(x)))

    prior = UserFunction(log_prior, sample_prior, D=2)
    assert prior.D == 2
    assert prior.log_pdf is log_prior
    assert prior.sample is sample_prior
    x = np.random.normal(size=10)
    assert np.isclose(prior.log_pdf(x), np.log(prior.pdf(x)))
    assert np.isclose(prior.pdf(x), np.prod(np.exp(x)))


def test_function_prior_error_handling():
    with raises(TypeError) as err:
        UserFunction(1.0)
    assert "`log_prior` must be callable." in err.value.args[0]
    with raises(TypeError) as err:
        log_prior = lambda x: np.sum(x)
        UserFunction(log_prior, 1.0)
    assert (
        "Optional keyword `sample_prior` must be callable."
        in err.value.args[0]
    )


def test__str__and__repr__():
    D = 4
    log_prior = lambda x: np.sum(x)
    prior = UserFunction(log_prior, D=D)
    string = prior.__str__()
    assert f"{prior.__class__.__name__} prior:" in string
    assert f"dimension = {D}" in string

    repr_ = prior.__repr__()
    assert f"self.D = {D}" in repr_
