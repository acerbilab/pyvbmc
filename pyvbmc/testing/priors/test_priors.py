from itertools import product

import numpy as np
import pytest
from scipy.integrate import nquad
from scipy.stats import multivariate_normal

from pyvbmc.priors import (
    Prior,
    Product,
    SciPy,
    SmoothBox,
    SplineTrapezoidal,
    Trapezoidal,
    UniformBox,
)

classes = [
    UniformBox,
    Trapezoidal,
    SmoothBox,
    SplineTrapezoidal,
    SciPy,
    Product,
]
from pyvbmc import VBMC


def integrate(prior, epsabs=1.49e-08):
    lb, ub = prior.support()
    lb, ub = lb.reshape(-1, 1), ub.reshape(-1, 1)

    def func(*x):
        return prior.pdf(np.array(x), keepdims=False).item()

    return nquad(func, np.hstack([lb, ub]), opts={"epsabs": epsabs})[0]


def test_prior_init():
    for cls in classes:
        prior = cls._generic()
        assert isinstance(prior, Prior), f"Failed for {cls.__name__}!"


def test_unit_integral_1d():
    for cls in classes:
        prior = cls._generic()
        integral = integrate(prior)
        assert np.isclose(integral, 1.0)


def test_unit_integral_2d():
    for cls in classes:
        prior = cls._generic(D=2)
        integral = integrate(prior)
        assert np.isclose(integral, 1.0)


def test_shape():
    for cls in classes:
        for D in [1, 4]:
            prior = cls._generic(D)

            x1 = np.random.normal(size=(D,))
            y1 = prior.log_pdf(x1)
            y1t = prior.log_pdf(x1, keepdims=True)
            y1f = prior.log_pdf(x1, keepdims=False)
            assert y1.shape == (1, 1)
            assert y1t.shape == (1, 1)
            assert y1f.shape == (1,)
            assert np.array_equal(y1, y1t)
            assert np.array_equal(y1, y1f.reshape(1, 1))

            x2 = x1.reshape((1, D))
            y2 = prior.log_pdf(x2)
            y2t = prior.log_pdf(x2, keepdims=True)
            y2f = prior.log_pdf(x2, keepdims=False)
            assert y2.shape == (1, 1)
            assert y2t.shape == (1, 1)
            assert y2f.shape == (1,)
            assert np.array_equal(y1, y2)
            assert np.array_equal(y2, y2t)
            assert np.array_equal(y2, y2f.reshape(1, 1))

            n = 20
            x = np.random.normal(size=(n, D))
            y = prior.log_pdf(x)
            yt = prior.log_pdf(x, keepdims=True)
            yf = prior.log_pdf(x, keepdims=False)
            assert y.shape == (n, 1)
            assert yt.shape == (n, 1)
            assert yf.shape == (n,)
            assert np.array_equal(y, yt)
            assert np.array_equal(y, yf.reshape(n, 1))


def test_sample():
    for cls in classes:
        D = np.random.randint(1, 5)
        prior = cls._generic(D=D)
        n = np.random.randint(0, 10000)
        samples = prior.sample(n)
        assert samples.shape == (n, prior.D)

        lb, ub = prior.support()
        assert np.all(samples > lb) and np.all(samples < ub)


def test__str__and__repr__():
    D = 4
    for prior in classes:
        new_prior = prior._generic(D)
        string = new_prior.__str__()
        assert f"{prior.__name__} prior:" in string
        assert f"dimension = {D}" in string
        assert f"lower bounds = {new_prior.a}" in string
        assert f"upper bounds = {new_prior.b}" in string

        repr_ = new_prior.__repr__()
        assert f"self.D = {D}" in repr_
        assert f"self.a = [" in repr_
        assert f"self.b = [" in repr_
