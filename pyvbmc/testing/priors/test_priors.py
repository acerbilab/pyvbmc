import numpy as np
import pytest
from scipy.integrate import nquad

from pyvbmc.priors import (
    Prior,
    SciPy,
    SmoothBox,
    SplineTrapezoidal,
    Trapezoidal,
    UniformBox,
)

classes = [UniformBox, Trapezoidal, SmoothBox, SplineTrapezoidal, SciPy]


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
        integral = integrate(prior)
        assert np.isclose(integral, 1.0)


def test_sample():
    for cls in classes:
        D = np.random.randint(1, 5)
        prior = cls._generic(D=D)
        n = np.random.randint(0, 10000)
        samples = prior.sample(n)
        assert samples.shape == (n, prior.D)

        lb, ub = prior._support()
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
