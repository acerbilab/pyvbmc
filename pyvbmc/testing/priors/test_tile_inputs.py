import random

import numpy as np
import pytest

from pyvbmc.priors import UniformBox, tile_inputs


def test_tile_inputs_all_scalars():
    a, b, c = 1, 2, 3
    x, y, z = tile_inputs(a, b, c)
    assert x == np.array([a]) and y == np.array([b]) and z == np.array([c])


def test_tile_inputs_explicit_size():
    n = np.random.randint(0, 10)
    a, b, c = 1, 2, 3
    x1, y1, z1 = tile_inputs(a, b, c, size=n)
    x2, y2, z2 = tile_inputs(a, b, c, size=(n,))
    assert (
        np.array_equal(x1, x2)
        and np.array_equal(y1, y2)
        and np.array_equal(z1, z2)
    )
    assert np.array_equal(x1, np.full((n,), a))
    assert np.array_equal(y1, np.full((n,), b))
    assert np.array_equal(z1, np.full((n,), c))

    n, m = np.random.randint(0, 10, size=2)
    x, y, z = tile_inputs(a, b, c, size=(n, m))
    assert np.array_equal(x, np.full((n, m), a))
    assert np.array_equal(y, np.full((n, m), b))
    assert np.array_equal(z, np.full((n, m), c))


def test_tile_inputs_implicit_size():
    n = np.random.randint(0, 10)
    a, b, c = 1, np.full((n,), 2), 3
    a, b, c = random.sample([a, b, c], 3)
    x, y, z = tile_inputs(a, b, c)
    assert np.array_equal(x, np.full((n,), a))
    assert np.array_equal(y, np.full((n,), b))
    assert np.array_equal(z, np.full((n,), c))

    n, m = np.random.randint(0, 10, size=2)
    a, b, c = 1, np.full((n, m), 2), 3
    a, b, c = random.sample([a, b, c], 3)
    x, y, z = tile_inputs(a, b, c)
    assert np.array_equal(x, np.full((n, m), a))
    assert np.array_equal(y, np.full((n, m), b))
    assert np.array_equal(z, np.full((n, m), c))


def test_tile_inputs_wrong_size():
    n = np.random.randint(0, 10)
    a, b, c = 1, np.full((n,), 2), 3
    a, b, c = random.sample([a, b, c], 3)
    with pytest.raises(ValueError) as e:
        x, y, z = tile_inputs(a, b, c, size=(n + 1,))
    assert (
        f"All inputs should agree with size=({n+1},), but found an input "
        f"with shape ({n},)." in e.value.args[0]
    )


@pytest.mark.parametrize("squeeze", [False, True])
def test_tile_inputs_shape_disagreeing_with_size(squeeze):
    """An array whose shape differs from `size` in more than its axes of
    length one is refused even where its number of elements would let
    `reshape` succeed."""
    with pytest.raises(ValueError) as e:
        tile_inputs(np.zeros((2, 2)), np.ones((2, 2)), size=4, squeeze=squeeze)
    assert (
        "All inputs should agree with size=(4,), but found an input with "
        "shape (2, 2)." in e.value.args[0]
    )

    with pytest.raises(ValueError) as e:
        UniformBox(np.zeros((2, 2)), np.ones((2, 2)), D=4)
    assert "should agree with size=(4,)" in e.value.args[0]


@pytest.mark.parametrize("squeeze", [False, True])
@pytest.mark.parametrize("shape", [(3,), (1, 3), (3, 1), (1, 3, 1)])
def test_tile_inputs_takes_a_shape_that_squeezes_to_size(squeeze, shape):
    """A row, a column and a flat array of the right length all describe
    the same `D` values, so all three agree with `size=D`."""
    a, b = np.arange(3.0).reshape(shape), np.ones(shape)
    x, y = tile_inputs(a, b, size=3, squeeze=squeeze)
    assert x.shape == (3,) and y.shape == (3,)
    assert np.array_equal(x, np.arange(3.0))
    assert np.array_equal(y, np.ones(3))


@pytest.mark.parametrize("shape", [(3,), (1, 3), (3, 1)])
def test_tile_inputs_takes_the_shapes_the_priors_pass(shape):
    """The prior constructors squeeze their arguments, so a row, a column
    and a flat array of the right length all agree with `size`."""
    a, b = np.zeros(shape), np.ones(shape)

    assert UniformBox(a, b, D=3).a.shape == (3,)
    assert UniformBox(a, b).a.shape == (3,)
    assert UniformBox(0.0, 1.0, D=3).a.shape == (3,)

    x, y = tile_inputs(a, b, size=3, squeeze=True)
    assert x.shape == (3,) and y.shape == (3,)


def test_tile_inputs_implicit_size_mismatch():
    n = np.random.randint(0, 10)
    a, b, c = 1, np.full((n,), 2), np.full((n + 1,), 3)
    a, b, c = random.sample([a, b, c], 3)
    with pytest.raises(ValueError) as e:
        x, y, z = tile_inputs(a, b, c)
    assert (
        "All inputs should have the same shape, but found inputs with shape"
        in e.value.args[0]
    )
    with pytest.raises(ValueError) as e:
        x, y, z = tile_inputs(a, b, c, size=(n + 2,))
    assert (
        "All inputs should have the same shape, but found inputs with shape"
        in e.value.args[0]
    )
