import numpy as np
import pytest
from scipy.integrate import nquad

from pyvbmc.priors import (
    Prior,
    SmoothBox,
    SplineTrapezoid,
    Trapezoid,
    UniformBox,
)

classes = [UniformBox, Trapezoid, SmoothBox, SplineTrapezoid]


def integrate(prior, epsabs=1.49e-08):
    lb, ub = prior._support()
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
        integral = integrate(prior, epsabs=0.01)
        assert np.isclose(integral, 1.0, atol=0.001)


def test_sample():
    for cls in classes:
        D = np.random.randint(1, 5)
        prior = cls._generic(D=D)
        n = np.random.randint(0, 10000)
        samples = prior.sample(n)
        assert samples.shape == (n, prior.D)

        lb, ub = prior._support()
        assert np.all(samples > lb) and np.all(samples < ub)
