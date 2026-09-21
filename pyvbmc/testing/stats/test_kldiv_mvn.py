import numpy as np
import pytest

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


@pytest.mark.parametrize("sd", [1e-9, 1e-8, 1.0, 1e9])
def test_kl_div_does_not_depend_on_the_scale(sd):
    """For ``N(0, S)`` against ``N(0, 4S)`` the two divergences are the
    same whatever ``S``, and at ``D = 20`` the determinant of ``S`` is
    ``sd ** 40``: zero below a scale of about 8e-9 and infinite above
    about 5e7."""
    D = 20
    sigma1 = sd**2 * np.eye(D)
    sigma2 = 4 * sigma1
    mu = np.zeros(D)
    expected = np.array(
        [
            0.5 * (D / 4 - D + D * np.log(4.0)),
            0.5 * (4 * D - D - D * np.log(4.0)),
        ]
    )

    assert np.allclose(
        kl_div_mvn(mu, sigma1, mu, sigma2), expected, rtol=1e-9, atol=0
    )


@pytest.mark.parametrize(
    "sigma2",
    [
        np.zeros((2, 2)),
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        np.diag([1.0, -1.0]),
    ],
)
def test_kl_div_infinite_without_a_positive_determinant(sigma2):
    """A singular covariance matrix, or one of negative determinant, is
    not the covariance of a distribution with a density."""
    mu = np.zeros(2)

    assert np.all(np.isinf(kl_div_mvn(mu, np.eye(2), mu, sigma2)))
    assert np.all(np.isinf(kl_div_mvn(mu, sigma2, mu, np.eye(2))))


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
