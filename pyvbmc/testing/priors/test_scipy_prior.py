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
