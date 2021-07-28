import numpy as np
from pyvbmc.stats import get_hpd

def test_get_hpd():
    order = np.random.permutation(range(0, 100))
    X = np.reshape(order.copy(), (-1, 1))
    y = X.copy()

    hpd_X, hpd_y, hpd_range = get_hpd(X, y)

    assert np.all(hpd_X == hpd_y)
    assert np.all(hpd_X.flatten() == np.array(list(reversed(range(20, 100)))))
    assert hpd_range == np.array([79])

    hpd_X, hpd_y, hpd_range = get_hpd(X, y, hpd_frac=0.5)

    assert np.all(hpd_X == hpd_y)
    assert np.all(hpd_X.flatten() == np.array(list(reversed(range(50, 100)))))
    assert hpd_range == np.array([49])
