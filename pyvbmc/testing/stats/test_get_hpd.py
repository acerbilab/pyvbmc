import numpy as np
import pytest

from pyvbmc.stats import get_hpd


def test_get_hpd():
    order = np.random.permutation(range(0, 100))
    X = np.reshape(order.copy(), (-1, 1))
    y = X.copy()

    hpd_X, hpd_y, hpd_range, indices = get_hpd(X, y)

    assert np.all(hpd_X == hpd_y)
    assert np.all(hpd_X.flatten() == np.array(list(reversed(range(20, 100)))))
    assert hpd_range == np.array([79])
    assert indices.shape == (80,)
    assert np.all(hpd_X == X[indices])

    hpd_X, hpd_y, hpd_range, indices = get_hpd(X, y, hpd_frac=0.5)

    assert np.all(hpd_X == hpd_y)
    assert np.all(hpd_X.flatten() == np.array(list(reversed(range(50, 100)))))
    assert hpd_range == np.array([49])
    assert indices.shape == (50,)
    assert np.all(hpd_X == X[indices])

    hpd_X, hpd_y, hpd_range, indices = get_hpd(X, y, hpd_frac=0.01)

    assert np.all(hpd_X == hpd_y)
    assert np.all(hpd_X == np.array([99]))
    assert np.all(X[indices] == np.array([99]))
    assert hpd_range == np.array([0])
    assert indices.shape == (1,)
    assert np.all(hpd_X == X[indices])
    assert np.all(indices[0] == np.argsort(order, axis=None)[::-1][0])


def _descending_order(y):
    """The order of ``sort(y,'descend')``: descending, and among equal
    values the earlier index first. Python's ``sorted`` is stable, and
    ``reverse=True`` leaves the order of equal elements alone."""
    values = np.asarray(y).flatten()
    return sorted(range(values.size), key=lambda i: values[i], reverse=True)


def test_get_hpd_orders_distinct_values_by_value_alone():
    """With no two targets equal the descending order is unique, so the
    subset and its order are decided by the values."""
    rng = np.random.default_rng(20260920)
    for __ in range(50):
        X = rng.standard_normal((12, 2))
        y = rng.standard_normal((12, 1))
        assert np.unique(y).size == y.size

        __, __, __, indices = get_hpd(X, y, hpd_frac=0.5)

        assert list(indices) == _descending_order(y)[:6]


def test_get_hpd_breaks_a_tie_at_the_cut_by_index():
    """``misc/gethpd_vbmc.m:9`` sorts with MATLAB's ``sort(y,'descend')``,
    which is stable: of several points with the same target value, the
    earlier one is taken into the subset first."""
    X = np.reshape(np.arange(5.0), (-1, 1))
    y = np.reshape([2.0, 2.0, 2.0, 1.0, 0.0], (-1, 1))

    hpd_X, hpd_y, __, indices = get_hpd(X, y, hpd_frac=0.4)

    assert list(indices) == [0, 1]
    assert np.all(hpd_X.flatten() == np.array([0.0, 1.0]))
    assert np.all(hpd_y.flatten() == np.array([2.0, 2.0]))


def test_get_hpd_orders_integer_targets_by_value():
    """The subset holds the largest values whatever the dtype of ``y``. The
    negation of an unsigned integer wraps around, zero staying zero, and so
    does that of the smallest value of a signed type, so an order taken from
    the negated integers would put those values first."""
    X = np.reshape(np.arange(3.0), (-1, 1))
    for values, dtype in (([3, 0, 2], np.uint8), ([-128, 0, 127], np.int8)):
        y = np.reshape(np.array(values, dtype=dtype), (-1, 1))

        __, hpd_y, __, indices = get_hpd(X, y, hpd_frac=1.0)

        assert list(indices) == _descending_order(y)
        assert hpd_y.dtype == dtype


@pytest.mark.parametrize("N, size", [(5, 1), (25, 3)])
def test_get_hpd_sizes_a_tie_away_from_zero(N, size):
    """``misc/gethpd_vbmc.m:10`` sizes the subset with MATLAB's ``round``,
    which sends a half away from zero: a tenth of 5 points is 1 and a
    tenth of 25 is 3, where Python's built-in ``round`` gives 0 and 2."""
    X = np.reshape(np.arange(float(N)), (-1, 1))
    y = X.copy()

    hpd_X, hpd_y, __, indices = get_hpd(X, y, hpd_frac=0.1)

    assert hpd_X.shape == (size, 1)
    assert hpd_y.shape == (size, 1)
    assert indices.shape == (size,)


def test_get_hpd_hpd_frac_zero():
    """
    Test that function also works with hpd_frac = 0.
    """
    order = np.random.permutation(range(0, 100))
    X = np.reshape(order.copy(), (-1, 1))
    y = X.copy()
    hpd_X, hpd_y, hpd_range, indices = get_hpd(X, y, hpd_frac=0)

    assert hpd_X.shape == (0, 1)
    assert hpd_y.shape == (0, 1)
    assert hpd_range.shape == (1,)
    assert np.all(np.isnan(hpd_range))
    assert indices.shape == (0,)
