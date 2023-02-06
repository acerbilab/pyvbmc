import numpy as np
import pytest

from pyvbmc.priors import UniformBox


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
