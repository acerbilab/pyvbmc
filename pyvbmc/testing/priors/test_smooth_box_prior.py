import numpy as np
import pytest
from scipy.stats import multivariate_normal

from pyvbmc.priors import SmoothBox


def test_smooth_box_basic_pdf():
    D = np.random.randint(1, 21)
    prior = SmoothBox(0, 1, 1, D=D)

    area = 1 + np.sqrt(2 * np.pi)
    h = 1 / area
    midpoint = np.full((1, D), 0.5)
    assert np.isclose(prior.pdf(midpoint), h**D)


def test_smooth_box_random_pdf():
    D = np.random.randint(1, 21)
    a = np.random.normal(0, 10, size=D)
    b = a + np.abs(np.random.normal(0, 10, size=D))
    scale = np.random.lognormal(size=D)
    prior = SmoothBox(a, b, scale, D=D)

    # sample some points inside and outside of support
    points = np.random.uniform(a - 4 / D, b + 4 / D, size=(10000, D))
    constant = np.all((points >= a) & (points < b), axis=1)

    h = 1 / (b - a + scale * np.sqrt(2 * np.pi))  # heights of marginal pdfs
    max_pdf = np.prod(h)
    # pdf inside [u,v) should be constant
    const_ps = prior.pdf(points[constant])
    assert np.all(const_ps == const_ps[0])
    assert np.allclose(const_ps, max_pdf)


def test_smooth_box_like_mv_normal():
    D = np.random.randint(1, 21)
    a = 0.0
    b = a + np.finfo(np.float64).eps
    scale = np.random.lognormal(size=D)
    prior = SmoothBox(a, b, scale, D=D)
    mv_normal = multivariate_normal(np.zeros(D), np.diag(scale**2))
    points = mv_normal.rvs(10000).reshape(-1, D)
    assert np.allclose(prior.pdf(points).ravel(), mv_normal.pdf(points))


def test_smooth_box_error_handling():
    D = 3
    a = np.array([0.0, 0.5, 0.0])
    b = np.array([1.0, 0.5, 1.0])
    scale = np.array([1.0, 0.0, 1.0])
    with pytest.raises(ValueError) as err:
        prior = SmoothBox(a, b, scale)
    assert (
        f"All elements of scale={scale} should be positive."
        in err.value.args[0]
    )
    with pytest.raises(ValueError) as err:
        prior = SmoothBox(a, b, np.ones(D))
    assert (
        f"All elements of a={a} should be strictly less than b={b}."
        in err.value.args[0]
    )
    with pytest.raises(ValueError) as err:
        prior = SmoothBox(np.zeros(D + 1), b, np.ones(D))
    assert (
        f"All inputs should have the same shape, but found inputs with shapes ({D+1},) and ({D},)."
        in err.value.args[0]
    )
