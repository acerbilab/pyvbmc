import numpy as np
import pytest

from pyvbmc.priors import SmoothBox, SplineTrapezoidal, Trapezoidal, UniformBox


def test_uniform_box_unit_pdf():
    D = np.random.randint(1, 21)
    prior = UniformBox(0, 1, D=D)

    midpoint = np.full((1, D), 0.5)
    assert prior.pdf(midpoint) == 1.0


def test_uniform_box_random_pdf():
    D = np.random.randint(1, 21)
    lb = np.random.normal(0, 10, size=D)
    ub = lb + np.abs(np.random.normal(0, 10, size=D))
    prior = UniformBox(lb, ub, D=D)

    # sample some points inside and outside of support
    points = np.random.uniform(lb - 1 / D, ub + 1 / D, size=(10000, D))
    inside = np.all((points >= lb) & (points < ub), axis=1)

    volume = np.prod(ub - lb)
    # pdf inside support should be reciprocal of volume
    inside_ps = prior.pdf(points[inside])
    assert np.all(inside_ps == inside_ps[0])
    assert np.allclose(inside_ps, 1 / volume)
    # pdf outside support should be zero
    assert np.all(prior.pdf(points[~inside]) == 0)


def test_uniform_box_includes_its_bounds():
    """The support is closed: a point on a bound has the density of the
    plateau."""
    D = 3
    a = np.array([-2.0, 0.0, 1.5])
    b = np.array([1.0, 4.0, 3.0])
    prior = UniformBox(a, b)

    plateau = 1 / np.prod(b - a)
    assert np.isclose(prior.pdf(a.reshape(1, D)).item(), plateau)
    assert np.isclose(prior.pdf(b.reshape(1, D)).item(), plateau)
    assert np.isclose(
        prior.pdf(np.array([[a[0], b[1], a[2]]])).item(), plateau
    )


@pytest.mark.parametrize(
    "prior",
    [
        UniformBox(0.0, 1.0, D=2),
        Trapezoidal(0.0, 0.25, 0.75, 1.0, D=2),
        SplineTrapezoidal(0.0, 0.25, 0.75, 1.0, D=2),
        SmoothBox(0.0, 1.0, 1.0, D=2),
    ],
)
def test_a_nan_coordinate_has_density_zero(prior):
    """A row that holds a NaN coordinate is not in the support of any of the
    four box families."""
    x = np.array([[np.nan, 0.5]])
    assert np.all(np.isneginf(prior.log_pdf(x)))
    assert prior.pdf(x).item() == 0.0

    x = np.array([[0.5, np.nan]])
    assert np.all(np.isneginf(prior.log_pdf(x)))
    assert prior.pdf(x).item() == 0.0


def test_uniform_box_error_handling():
    D = 3
    a = np.array([0.0, 0.5, 0.0])
    b = np.array([1.0, 0.5, 1.0])
    with pytest.raises(ValueError) as err:
        prior = UniformBox(a, b)
    assert (
        f"All elements of a={a} should be strictly less than b={b}."
        in err.value.args[0]
    )
    with pytest.raises(ValueError) as err:
        prior = UniformBox(np.zeros(D + 1), b)
    assert (
        f"All inputs should have the same shape, but found inputs with shapes ({D+1},) and ({D},)."
        in err.value.args[0]
    )


def test_uniform_box_sample_rng():
    prior = UniformBox._generic(D=3)
    s1 = prior.sample(5, rng=np.random.default_rng(0))
    s2 = prior.sample(5, rng=np.random.default_rng(0))
    assert np.array_equal(s1, s2)
    assert prior.sample(5).shape == (5, prior.D)
