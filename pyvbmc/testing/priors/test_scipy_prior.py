from copy import deepcopy

import numpy as np
import pytest
from scipy.special import gamma
from scipy.stats import (
    beta,
    binom,
    expon,
    lognorm,
    multivariate_normal,
    multivariate_t,
    norm,
    t,
    uniform,
    wishart,
)

from pyvbmc.priors import Product, SciPy


def test_scipy_support_follows_loc_and_scale():
    """The support of a univariate marginal is the interval the
    distribution lives on, so a shift or a scale moves it."""
    a, b = SciPy(uniform(loc=2, scale=3)).support()
    assert np.allclose(a, [2.0]) and np.allclose(b, [5.0])

    a, b = SciPy(beta(2, 3, loc=-1, scale=4)).support()
    assert np.allclose(a, [-1.0]) and np.allclose(b, [3.0])

    a, b = SciPy(expon(loc=5)).support()
    assert np.allclose(a, [5.0]) and np.all(np.isposinf(b))

    # An unshifted distribution keeps the support of its standard form.
    a, b = SciPy(norm()).support()
    assert np.all(np.isneginf(a)) and np.all(np.isposinf(b))


def test_product_support_follows_loc_and_scale_of_its_marginals():
    """A `Product` reads the support of each marginal, so a shifted or
    scaled marginal contributes the interval it lives on."""
    prior = Product([uniform(loc=2, scale=3), beta(2, 3, loc=-1, scale=4)])
    a, b = prior.support()
    assert np.allclose(a, [2.0, -1.0])
    assert np.allclose(b, [5.0, 3.0])


def test_a_univariate_distribution_over_several_variables_is_refused():
    """A frozen univariate distribution with array-valued parameters stands
    for one distribution per element, so its sampler and its density
    describe different things."""
    with pytest.raises(ValueError) as err:
        SciPy(norm(loc=np.array([0.0, 10.0, 100.0])))
    message = err.value.args[0]
    assert "loc holds 3 values" in message
    assert "should be scalars" in message

    with pytest.raises(ValueError) as err:
        SciPy(t(np.array([3.0, 7.0])))
    assert "positional parameter 0 holds 2 values" in err.value.args[0]

    # The same distribution as a marginal of a product.
    with pytest.raises(ValueError) as err:
        Product([norm(loc=np.array([0.0, 10.0])), norm()])
    assert "should be scalars" in err.value.args[0]


@pytest.mark.parametrize(
    "distribution",
    [norm(loc=0.0), norm(loc=np.array(0.0)), norm(loc=np.array([0.0]))],
)
def test_a_scalar_parameter_is_taken_however_it_is_written(distribution):
    """SciPy treats a 0-d array and a one-element array as scalars, and so
    does the prior built from them."""
    prior = SciPy(distribution)
    assert prior.D == 1
    assert prior.log_pdf(np.zeros((4, 1))).shape == (4, 1)
    assert prior.sample(4, rng=np.random.default_rng(0)).shape == (4, 1)


def test_scipy_mv_normal_pdf():
    D = np.random.randint(1, 21)

    # Multivariate version
    base_dist = multivariate_normal(np.zeros(D))
    prior1 = SciPy(base_dist)
    assert np.isclose(prior1.pdf(np.zeros(D)), (2 * np.pi) ** (-D / 2))

    n = 1000
    x = base_dist.rvs(n).reshape(-1, D)
    y0 = base_dist.logpdf(x).reshape((n, 1))
    y1 = prior1.log_pdf(x)
    assert np.allclose(y1, y0)


def test_scipy_mv_t_pdf():
    D = np.random.randint(1, 21)
    nu = np.random.randint(1, 31)

    # Multivariate version
    base_dist = multivariate_t(np.zeros(D), shape=np.eye(D), df=nu)
    prior = SciPy(base_dist)
    nf = gamma((nu + D) / 2) / (
        gamma(nu / 2) * nu ** (D / 2) * np.pi ** (D / 2)
    )
    assert np.isclose(prior.pdf(np.zeros(D)), nf)

    n = 1000
    x = base_dist.rvs(n).reshape(-1, D)
    y0 = base_dist.logpdf(x).reshape((n, 1))
    y1 = prior.log_pdf(x)
    assert np.allclose(y1, y0)


def test_scipy_type_checking():
    with pytest.raises(TypeError) as err:
        SciPy(1.0)
    assert (
        "A SciPy prior should be initialized from a \"frozen\" multivariate normal, multivariate t, or univariate SciPy distribution, but got `distribution` of type <class 'float'>."
        in err.value.args[0]
    )
    with pytest.raises(TypeError) as err:
        SciPy(wishart())
    assert (
        "A SciPy prior should be initialized from a \"frozen\" multivariate normal, multivariate t, or univariate SciPy distribution, but got `distribution` of type <class 'scipy.stats._multivariate.wishart_frozen'>."
        in err.value.args[0]
    )
    with pytest.raises(TypeError) as err:
        SciPy(binom(10, 0.5))
    assert (
        "A SciPy prior should be initialized from a \"frozen\" multivariate normal, multivariate t, or univariate SciPy distribution, but got `distribution` of type <class 'scipy.stats._distn_infrastructure.rv_discrete_frozen'>."
        in err.value.args[0]
    )
    with pytest.raises(TypeError) as err:
        SciPy([norm(), norm(), norm()])
    assert (
        "A SciPy prior should be initialized from a \"frozen\" multivariate normal, multivariate t, or univariate SciPy distribution, but got `distribution` of type <class 'list'>."
        in err.value.args[0]
    )
    with pytest.raises(TypeError):
        SciPy(norm)

    class Unsupported:
        dist = norm

    with pytest.raises(TypeError):
        SciPy(Unsupported())


@pytest.mark.parametrize(
    "distribution",
    [
        multivariate_normal(np.zeros(3), seed=np.random.RandomState(12345)),
        multivariate_t(np.zeros(3), df=7, seed=np.random.RandomState(12345)),
    ],
)
def test_scipy_constructor_does_not_advance_rng(distribution):
    global_state = deepcopy(np.random.get_state())
    frozen_state = deepcopy(distribution.random_state.get_state())

    prior = SciPy(distribution)

    assert prior.D == distribution.dim
    _assert_random_state_equal(np.random.get_state(), global_state)
    _assert_random_state_equal(
        distribution.random_state.get_state(), frozen_state
    )


def _assert_random_state_equal(first, second):
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert first[2:] == second[2:]


def test_scipy_sample_rng():
    prior = SciPy._generic(D=3)
    s1 = prior.sample(5, rng=np.random.default_rng(0))
    s2 = prior.sample(5, rng=np.random.default_rng(0))
    assert np.array_equal(s1, s2)
    assert prior.sample(5).shape == (5, prior.D)
