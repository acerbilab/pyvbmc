import numpy as np

from pyvbmc.stats import kl_div_mvn


def test_kl_div_1d_identical():
    mu = np.ones((1, 1))
    sigma = np.ones((1, 1))
    assert np.all(kl_div_mvn(mu, sigma, mu, sigma) == 0)


def test_kl_div_1d_seperated():
    mu = np.ones((1, 1))
    sigma = np.ones((1, 1))
    assert np.all(kl_div_mvn(mu, sigma, mu * 100, sigma) == 4900.5)


def test_kl_div_3d_identical():
    mu = np.ones((3, 1))
    sigma = np.eye(3)
    assert np.all(kl_div_mvn(mu, sigma, mu, sigma) == 0)


def test_kl_div_3d_seperated():
    mu = np.ones((3, 1))
    sigma = np.eye(3)
    assert np.all(kl_div_mvn(mu, sigma, mu * 100, sigma) == 14701.5)


def test_kl_div_input_forms_agree():
    expected = kl_div_mvn(
        np.array([[1.0]]),
        np.array([[2.0]]),
        np.array([[3.0]]),
        np.array([[4.0]]),
    )

    assert np.array_equal(kl_div_mvn(1.0, 2.0, 3.0, 4.0), expected)
    assert np.array_equal(
        kl_div_mvn(
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            np.array([4.0]),
        ),
        expected,
    )
    assert np.array_equal(
        kl_div_mvn(
            mu1=1.0,
            sigma1=np.array([[2.0]]),
            mu2=np.array([3.0]),
            sigma2=4.0,
        ),
        expected,
    )
    assert np.array_equal(
        kl_div_mvn(
            1.0,
            sigma1=np.array([2.0]),
            mu2=np.array([[3.0]]),
            sigma2=4.0,
        ),
        expected,
    )

    mu1 = np.array([1.0, 2.0, 3.0])
    mu2 = np.array([3.0, 2.0, 1.0])
    sigma1 = np.diag([1.0, 2.0, 3.0])
    sigma2 = np.diag([3.0, 2.0, 1.0])
    assert np.array_equal(
        kl_div_mvn(mu1, sigma1, mu2, sigma2),
        kl_div_mvn(mu1[:, None], sigma1, mu2[:, None], sigma2),
    )
