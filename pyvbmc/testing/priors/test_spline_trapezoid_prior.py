import numpy as np
import pytest

from pyvbmc.priors import SplineTrapezoidal


def test_spline_trapezoidal_basic_pdf():
    D = np.random.randint(1, 21)
    prior = SplineTrapezoidal(0, 0.25, 0.75, 1, D=D)

    midpoint = np.full((1, D), 0.5)
    assert np.isclose(prior.pdf(midpoint), (4 / 3) ** D)


def test_spline_trapezoidal_random_pdf():
    D = np.random.randint(1, 21)
    a = np.random.normal(0, 10, size=D)
    u = a + 1 + np.abs(np.random.normal(0, 1, size=D))
    v = u + 2 * D + np.abs(np.random.normal(0, 1, size=D))
    b = v + 1 + np.abs(np.random.normal(0, 1, size=D))
    prior = SplineTrapezoidal(a, u, v, b, D=D)

    # sample some points inside and outside of support
    points = np.random.uniform(a - 4 / D, b + 4 / D, size=(10000, D))
    constant = np.all((points >= u) & (points < v), axis=1)
    outside = ~np.all((points >= a) & (points < b), axis=1)

    h = 1 / (v - u + 0.5 * (u - a + b - v))  # heights of marginal pdfs
    max_pdf = np.prod(h)
    # pdf inside [u,v) should be constant
    const_ps = prior.pdf(points[constant])
    assert np.all(const_ps == const_ps[0])
    assert np.allclose(const_ps, max_pdf)
    # pdf outside support should be zero
    assert np.all(prior.pdf(points[outside]) == 0)


def test_spline_trapezoidal_error_handling():
    D = 3
    a = np.zeros(D)
    u = np.zeros(D) + 0.25
    u[1] = 0.9
    v = np.zeros(D) + 0.75
    b = np.ones(D)
    with pytest.raises(ValueError) as err:
        prior = SplineTrapezoidal(a, u, v, b)
    assert (
        "Bounds and pivots should respect the order a < u < v < b."
        in err.value.args[0]
    )
    u = np.zeros(D + 1) + 0.25
    with pytest.raises(ValueError) as err:
        prior = SplineTrapezoidal(a, u, v, b)
    assert (
        f"All inputs should have the same shape, but found inputs with shapes ({D},) and ({D+1},)."
        in err.value.args[0]
    )
