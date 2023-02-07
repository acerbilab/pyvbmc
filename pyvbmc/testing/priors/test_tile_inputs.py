import random

import numpy as np
import pytest

from pyvbmc.priors import tile_inputs


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
        f"cannot reshape array of size {n} into shape ({n+1},)"
        in e.value.args[0]
    )


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
