import numpy as np

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
