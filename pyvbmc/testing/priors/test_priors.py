import warnings
from itertools import product

import numpy as np
import pytest
import scipy as sp
import scipy.stats
from scipy.integrate import nquad
from scipy.stats import beta, multivariate_normal, uniform

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


def test_unit_integral_shifted_scipy():
    """``integrate`` covers the density over ``support()``, so a shifted or
    scaled SciPy marginal integrates to one only where it lives."""
    prior = SciPy(uniform(loc=2, scale=3))
    assert np.isclose(integrate(prior), 1.0)

    product = Product([uniform(loc=2, scale=3), beta(2, 3, loc=-1, scale=4)])
    assert np.isclose(integrate(product), 1.0)


_BOX_ARGUMENTS = [
    (UniformBox, {"a": 0.0, "b": 1.0}),
    (Trapezoidal, {"a": 0.0, "u": 0.25, "v": 0.75, "b": 1.0}),
    (SplineTrapezoidal, {"a": 0.0, "u": 0.25, "v": 0.75, "b": 1.0}),
    (SmoothBox, {"a": 0.0, "b": 1.0, "scale": 1.0}),
]


@pytest.mark.parametrize("cls, arguments", _BOX_ARGUMENTS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_an_argument_which_is_not_finite_is_refused(cls, arguments, bad):
    """Every comparison with a NaN is false and an infinity satisfies a
    strict order, so a bound or a pivot which is not finite would pass the
    order checks and leave a density which is not a density."""
    for name in arguments:
        scalars = dict(arguments)
        scalars[name] = bad
        with pytest.raises(ValueError) as err:
            cls(**scalars)
        assert f"All elements of {name}=" in err.value.args[0]
        assert "should be finite" in err.value.args[0]

        # One coordinate of one argument, the others well formed.
        arrays = {key: np.full(3, value) for key, value in arguments.items()}
        arrays[name][1] = bad
        with pytest.raises(ValueError) as err:
            cls(**arrays)
        assert f"All elements of {name}=" in err.value.args[0]
        assert "should be finite" in err.value.args[0]


def test_the_finiteness_check_is_private_to_the_subpackage():
    """The helper the four box constructors share is not part of the
    interface of `pyvbmc.priors`, which exports and documents its names."""
    import pyvbmc.priors
    from pyvbmc.priors.prior import _check_finite

    assert not hasattr(pyvbmc.priors, "check_finite")
    with pytest.raises(ValueError) as err:
        _check_finite({"a": np.array([0.0, np.nan])})
    assert "All elements of a=" in err.value.args[0]
    assert "should be finite" in err.value.args[0]


def test_infinite_hard_bounds_are_refused_by_a_uniform_box():
    """The message says why an unbounded problem's hard bounds do not make
    a uniform-box prior."""
    with pytest.raises(ValueError) as err:
        UniformBox(np.full(2, -np.inf), np.full(2, np.inf))
    assert "uniform-box prior needs finite bounds" in err.value.args[0]


def _dtype_cases():
    """One instance of each class, with a point whose coordinates are whole
    numbers, so that the three dtypes hold the same point exactly."""
    return [
        UniformBox(0.0, 10.0, D=2),
        Trapezoidal(0.0, 2.0, 8.0, 10.0, D=2),
        SmoothBox(0.0, 10.0, 1.0, D=2),
        SplineTrapezoidal(0.0, 2.0, 8.0, 10.0, D=2),
        SciPy(multivariate_normal(np.zeros(2))),
        Product([UniformBox(0.0, 10.0), Trapezoidal(0.0, 2.0, 8.0, 10.0)]),
    ]


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.float32])
def test_log_pdf_is_float64_whatever_the_input_dtype(dtype):
    """The log-density is a double whatever the input is (MATLAB's
    ``-inf(size(x))`` and its arithmetic are), so a narrower input gives the
    float64 value and not a truncated or a narrower one."""
    point = np.array([[3, 5]])
    for prior in _dtype_cases():
        expected = prior.log_pdf(point.astype(np.float64))
        assert expected.dtype == np.float64

        y = prior.log_pdf(point.astype(dtype))
        assert y.dtype == np.float64, f"Failed for {type(prior).__name__}!"
        assert np.array_equal(
            y, expected
        ), f"Failed for {type(prior).__name__}!"

        p = prior.pdf(point.astype(dtype))
        assert p.dtype == np.float64
        assert np.array_equal(p, prior.pdf(point.astype(np.float64)))


@pytest.mark.parametrize("cls", [Trapezoidal, SplineTrapezoidal])
def test_log_pdf_of_an_integer_point_outside_the_support(cls):
    """Outside the support the log-density is ``-inf`` and the density is
    zero, for a point of integer dtype as for one of floats: the fill is a
    double and its sum over the coordinates cannot wrap."""
    for D in (1, 2, 3, 4):
        prior = cls(0.0, 0.25, 0.75, 1.0, D=D)
        x = np.full((1, D), 20)
        assert x.dtype.kind == "i"
        assert np.all(np.isneginf(prior.log_pdf(x))), f"Failed for D={D}!"
        assert prior.pdf(x).item() == 0.0, f"Failed for D={D}!"

    # A point with some coordinates inside the support and some outside.
    mixed = cls(0.0, 2.0, 8.0, 10.0, D=3)
    assert np.all(np.isneginf(mixed.log_pdf(np.array([[20, 5, 5]]))))
    assert mixed.pdf(np.array([[20, 5, 5]])).item() == 0.0
    assert np.all(np.isneginf(mixed.log_pdf(np.array([[20, 20, 5]]))))
    assert mixed.pdf(np.array([[20, 20, 5]])).item() == 0.0


@pytest.mark.parametrize("cls", [Trapezoidal, SplineTrapezoidal])
def test_no_warning_at_a_bound(cls):
    """The density of both trapezoids is zero at either bound, so the
    logarithm of zero there is the answer and is reported as such, without
    a warning, as MATLAB's ``log(0)`` is."""
    prior = cls(0.0, 0.25, 0.75, 1.0, D=2)
    for x in (np.zeros((1, 2)), np.ones((1, 2))):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            log_pdf = prior.log_pdf(x)
        assert np.all(np.isneginf(log_pdf))


def test_the_error_state_survives_a_failure_in_the_spline_loop():
    """The suppression of the logarithm of zero is undone however the
    density leaves, so a failure does not leave it installed."""
    before = np.geterr()
    prior = SplineTrapezoidal(0.0, 0.25, 0.75, 1.0, D=2)
    # `_log_pdf` loops over the columns of `x` and indexes the prior's own
    # bounds, so a point with more columns than the prior has dimensions
    # fails inside the loop.
    with pytest.raises(IndexError):
        prior._log_pdf(np.full((2, 3), 0.5))
    assert np.geterr() == before


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


# Bounds, pivots and scales that differ in every dimension, so that a
# sampler which mixed up its coordinates would show.
_A = np.array([-3.0, 0.0, 10.0])
_U = np.array([-2.0, 0.5, 11.0])
_V = np.array([1.0, 0.75, 18.0])
_B = np.array([2.0, 4.0, 20.0])
_SCALE = np.array([0.2, 1.0, 5.0])


def _uniform_box_cdf(x, a, b):
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def _trapezoidal_cdf(x, a, u, v, b):
    """The integral of the trapezoidal density, ramp by ramp."""
    nf = 0.5 * (b - a + v - u)
    mass_left = (u - a) / (2 * nf)
    mass_plateau = mass_left + (v - u) / nf
    left = (x - a) ** 2 / (2 * (u - a) * nf)
    plateau = mass_left + (x - u) / nf
    right = mass_plateau + ((b - v) ** 2 - (b - x) ** 2) / (2 * (b - v) * nf)
    return np.select(
        [x < a, x < u, x < v, x < b],
        [0.0, left, plateau, right],
        default=1.0,
    )


def _spline_trapezoidal_cdf(x, a, u, v, b):
    """The integral of the spline-trapezoidal density. Its ramp is
    ``-2 z**3 + 3 z**2``, whose integral over ``z`` is ``-z**4 / 2 + z**3``,
    one half over the whole ramp."""
    nf = 0.5 * (v - u + b - a)
    mass_left = (u - a) / (2 * nf)
    mass_plateau = mass_left + (v - u) / nf
    z_left = (x - a) / (u - a)
    left = (u - a) / nf * (-(z_left**4) / 2 + z_left**3)
    plateau = mass_left + (x - u) / nf
    z_right = 1 - (x - v) / (b - v)
    right = mass_plateau + (b - v) / nf * (
        0.5 - (-(z_right**4) / 2 + z_right**3)
    )
    return np.select(
        [x < a, x < u, x < v, x < b],
        [0.0, left, plateau, right],
        default=1.0,
    )


def _smooth_box_cdf(x, a, b, scale):
    """The integral of the smooth-box density: half a Gaussian, a plateau,
    and half a Gaussian."""
    height = 1 / ((b - a) + np.sqrt(2 * np.pi) * scale)
    tail_mass = height * np.sqrt(2 * np.pi) * scale
    left = tail_mass * sp.stats.norm.cdf((x - a) / scale)
    plateau = tail_mass / 2 + height * (x - a)
    right = (
        tail_mass / 2
        + height * (b - a)
        + tail_mass * (sp.stats.norm.cdf((x - b) / scale) - 0.5)
    )
    return np.select([x < a, x <= b], [left, plateau], default=right)


_SAMPLER_CASES = [
    (
        UniformBox(_A, _B),
        lambda x, d: _uniform_box_cdf(x, _A[d], _B[d]),
    ),
    (
        Trapezoidal(_A, _U, _V, _B),
        lambda x, d: _trapezoidal_cdf(x, _A[d], _U[d], _V[d], _B[d]),
    ),
    (
        SplineTrapezoidal(_A, _U, _V, _B),
        lambda x, d: _spline_trapezoidal_cdf(x, _A[d], _U[d], _V[d], _B[d]),
    ),
    (
        SmoothBox(_A, _B, _SCALE),
        lambda x, d: _smooth_box_cdf(x, _A[d], _B[d], _SCALE[d]),
    ),
]
_SAMPLER_IDS = ["UniformBox", "Trapezoidal", "SplineTrapezoidal", "SmoothBox"]
_SAMPLER_SEED = 20260926
_SAMPLER_N = 20000


@pytest.mark.parametrize("prior, cdf", _SAMPLER_CASES, ids=_SAMPLER_IDS)
def test_sampler_draws_from_its_own_distribution(prior, cdf):
    """Each marginal of a draw is compared with the distribution function
    of the family, derived from the density by hand."""
    samples = prior.sample(
        _SAMPLER_N, rng=np.random.default_rng(_SAMPLER_SEED)
    )
    assert samples.shape == (_SAMPLER_N, 3)
    for d in range(3):
        p = sp.stats.kstest(samples[:, d], lambda x, d=d: cdf(x, d)).pvalue
        assert p > 1e-3, f"Failed for column {d} (p = {p})!"


@pytest.mark.parametrize(
    "cdf",
    [case[1] for case in _SAMPLER_CASES[1:3]],
    ids=_SAMPLER_IDS[1:3],
)
def test_the_sampler_check_sees_a_uniform_draw(cdf):
    """Uniform draws on the bounding box are what the rejection sampler of
    either trapezoid would give if its rejection test never fired, and the
    check refuses them."""
    rng = np.random.default_rng(_SAMPLER_SEED)
    for d in range(3):
        samples = rng.uniform(_A[d], _B[d], size=_SAMPLER_N)
        p = sp.stats.kstest(samples, lambda x, d=d: cdf(x, d)).pvalue
        assert p < 1e-3, f"Failed for column {d} (p = {p})!"


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
