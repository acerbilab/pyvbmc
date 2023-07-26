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

from pyvbmc.priors import Product, SciPy, SmoothBox, UniformBox


def test_product_prior_scipy_mv_normal_pdf():
    D = np.random.randint(1, 21)

    # Multivariate version
    base_dist = multivariate_normal(np.zeros(D))
    prior1 = SciPy(base_dist)
    assert np.isclose(prior1.pdf(np.zeros(D)), (2 * np.pi) ** (-D / 2))

    # Product of independent univariates:
    prior2 = Product([SciPy(norm(0.0)) for __ in range(D)])
    assert np.isclose(prior2.pdf(np.zeros(D)), (2 * np.pi) ** (-D / 2))

    n = 1000
    x = base_dist.rvs(n).reshape(-1, D)
    y0 = base_dist.logpdf(x).reshape((n, 1))
    y1 = prior1.log_pdf(x)
    y2 = prior2.log_pdf(x)
    assert np.allclose(y1, y0)
    assert np.allclose(y2, y1)


def test_product_prior_scipy_indep_t_pdf():
    D = np.random.randint(1, 21)
    nu = np.random.randint(1, 31)

    # Product of independent univariates:
    base_dist = t(nu)
    prior = Product([SciPy(base_dist) for __ in range(D)])
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


def test_product_mixed_distribution_type():
    m = 1.0
    s = 0.5
    a, b = 2.0, 1.5
    prior = Product(
        [
            norm(m),
            lognorm(s),
            UniformBox(0, 1),
            SmoothBox(0.0, np.finfo(np.float64).eps),
            SciPy(beta(a, b)),
        ]
    )
    n = 200000
    samples = prior.sample(n)
    assert np.isclose(np.mean(samples[:, 0]), m, atol=0.01)
    assert np.isclose(
        np.mean(samples[:, 1]), np.exp(0 + s**2 / 2), atol=0.01
    )
    assert np.isclose(np.mean(samples[:, 2]), 0.5, atol=0.01)
    assert np.isclose(np.mean(samples[:, 3]), 0.0, atol=0.01)
    assert np.isclose(np.mean(samples[:, 4]), a / (a + b), atol=0.01)

    assert np.isclose(np.var(samples[:, 0]), 1.0, atol=0.01)
    assert np.isclose(
        np.var(samples[:, 1]), (np.exp(s**2) - 1) * np.exp(s**2), atol=0.01
    )
    assert np.isclose(np.var(samples[:, 2]), 1 / 12, atol=0.01)
    assert np.isclose(np.var(samples[:, 3]), 1.0, atol=0.01)
    assert np.isclose(
        np.var(samples[:, 4]), a * b / ((a + b) ** 2 * (a + b + 1)), atol=0.01
    )


def test_product_type_checking():
    with raises(TypeError) as err:
        Product(1.0)
    assert (
        "`Product` should be initialized from a list of distributions, but received type <class 'float'>."
        in err.value.args[0]
    )
    with raises(TypeError) as err:
        Product([1.0, 2.0])
    assert (
        "All marginals should be subclasses of `pyvbmc.priors.Prior`, or valid continuous SciPy distributions, but found type <class 'float'>."
        in err.value.args[0]
    )
    with raises(TypeError) as err:
        Product([norm(), binom(10, 0.5)])
    assert (
        "All marginals should be subclasses of `pyvbmc.priors.Prior`, or valid continuous SciPy distributions, but found type <class 'scipy.stats._distn_infrastructure.rv_discrete_frozen'>."
        in err.value.args[0]
    )
    with raises(ValueError) as err:
        Product([norm(), multivariate_normal(np.zeros(2)), norm()])
    assert (
        "All marginals of a product distribution should have dimension 1, but marginal SciPy prior:"
        in err.value.args[0]
    )
