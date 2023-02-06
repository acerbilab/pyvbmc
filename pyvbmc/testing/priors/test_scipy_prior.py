import numpy as np
import pytest
from scipy.special import gamma
from scipy.stats import (
    beta,
    binom,
    lognorm,
    multivariate_normal,
    multivariate_t,
    norm,
    t,
    wishart,
)

from pyvbmc.priors import SciPy


def test_scipy_mv_normal_pdf():
    D = np.random.randint(1, 21)

    # Multivariate version
    base_dist = multivariate_normal(np.zeros(D))
    prior1 = SciPy(base_dist)
    assert not prior1._product
    assert np.isclose(prior1.pdf(np.zeros(D)), (2 * np.pi) ** (-D / 2))

    # Product of independent univariates:
    prior2 = SciPy([norm(0.0) for __ in range(D)])
    assert prior2._product
    assert np.isclose(prior2.pdf(np.zeros(D)), (2 * np.pi) ** (-D / 2))

    n = 1000
    x = base_dist.rvs(n).reshape(-1, D)
    y0 = base_dist.logpdf(x).reshape((n, 1))
    y1 = prior1.log_pdf(x)
    y2 = prior2.log_pdf(x)
    assert np.allclose(y1, y0)
    assert np.allclose(y2, y1)


def test_scipy_mv_t_pdf():
    D = np.random.randint(1, 21)
    nu = np.random.randint(1, 31)

    # Multivariate version
    base_dist = multivariate_t(np.zeros(D), shape=np.eye(D), df=nu)
    prior = SciPy(base_dist)
    assert not prior._product
    nf = gamma((nu + D) / 2) / (
        gamma(nu / 2) * nu ** (D / 2) * np.pi ** (D / 2)
    )
    assert np.isclose(prior.pdf(np.zeros(D)), nf)

    n = 1000
    x = base_dist.rvs(n).reshape(-1, D)
    y0 = base_dist.logpdf(x).reshape((n, 1))
    y1 = prior.log_pdf(x)
    assert np.allclose(y1, y0)


def test_scipy_indep_t_pdf():
    D = np.random.randint(1, 21)
    nu = np.random.randint(1, 31)

    # Product of independent univariates:
    base_dist = t(nu)
    prior = SciPy([base_dist for __ in range(D)])
    assert prior._product
    nf = (
        gamma((nu + 1) / 2)
        / (gamma(nu / 2) * nu ** (1 / 2) * np.pi ** (1 / 2))
    ) ** D
    assert np.isclose(prior.pdf(np.zeros(D)), nf)

    n = 1000
    x = multivariate_t(np.zeros(D), df=nu).rvs(n).reshape(-1, D)
    y0 = np.sum(
        np.hstack(
            [base_dist.logpdf(x[:, d]).reshape((n, 1)) for d in range(D)]
        ),
        axis=1,
        keepdims=True,
    )
    y1 = prior.log_pdf(x)
    assert np.allclose(y1, y0)


def test_scipy_product_distribution():
    m = 1.0
    s = 0.5
    a, b = 2.0, 1.5
    prior = SciPy([norm(m), lognorm(s), beta(a, b)])
    n = 200000
    samples = prior.sample(n)
    assert np.isclose(np.mean(samples[:, 0]), m, atol=0.01)
    assert np.isclose(
        np.mean(samples[:, 1]), np.exp(0 + s**2 / 2), atol=0.01
    )
    assert np.isclose(np.mean(samples[:, 2]), a / (a + b), atol=0.01)
    assert np.isclose(np.var(samples[:, 0]), 1.0, atol=0.01)
    assert np.isclose(
        np.var(samples[:, 1]), (np.exp(s**2) - 1) * np.exp(s**2), atol=0.01
    )
    assert np.isclose(
        np.var(samples[:, 2]), a * b / ((a + b) ** 2 * (a + b + 1)), atol=0.01
    )


def test_scipy_type_checking():
    with pytest.raises(TypeError) as err:
        SciPy(1.0)
    assert (
        "A SciPy prior should be initialized from a \"frozen\" multivariate normal or multivariate t distribution, or an iterable of univariate scipy distributions, but got `distribution` of type <class 'float'>."
        in err.value.args[0]
    )
    with pytest.raises(TypeError) as err:
        SciPy(wishart())
    assert (
        "A SciPy prior should be initialized from a \"frozen\" multivariate normal or multivariate t distribution, or an iterable of univariate scipy distributions, but got `distribution` of type <class 'scipy.stats._multivariate.wishart_frozen'>."
        in err.value.args[0]
    )
    with pytest.raises(TypeError) as err:
        SciPy(binom(10, 0.5))
    assert (
        "A SciPy prior should be initialized from a \"frozen\" multivariate normal or multivariate t distribution, or an iterable of univariate scipy distributions, but got `distribution` of type <class 'scipy.stats._distn_infrastructure.rv_discrete_frozen'>."
        in err.value.args[0]
    )
    with pytest.raises(TypeError) as err:
        SciPy([norm(), multivariate_normal(np.zeros(2)), norm()])
    assert (
        "Each element of ``Prior`` should be a \"frozen\" continuous univariate distribution, but an element has type <class 'scipy.stats._multivariate.multivariate_normal_frozen'>"
        in err.value.args[0]
    )
