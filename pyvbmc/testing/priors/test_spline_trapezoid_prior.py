import numpy as np
import pytest

from pyvbmc.priors import SplineTrapezoid


def test_spline_trapezoid_basic_pdf():
    D = np.random.randint(1, 21)
    prior = SplineTrapezoid(0, 0.25, 0.75, 1, D=D)

    midpoint = np.full((1, D), 0.5)
    assert np.isclose(prior.pdf(midpoint), (4 / 3) ** D)


def test_spline_trapezoid_random_pdf():
    D = np.random.randint(1, 21)
    a = np.random.normal(0, 10, size=D)
    u = a + 1 + np.abs(np.random.normal(0, 1, size=D))
    v = u + 2 * D + np.abs(np.random.normal(0, 1, size=D))
    b = v + 1 + np.abs(np.random.normal(0, 1, size=D))
    prior = SplineTrapezoid(a, u, v, b, D=D)

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
