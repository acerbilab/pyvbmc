import numpy as np
import pytest
from parameter_transformer import ParameterTransformer
from variational_posterior import VariationalPosterior


def mock_init_vbmc(k=2, nvars=3):
    vp = VariationalPosterior()
    vp.d = nvars
    vp.k = k
    x0 = np.array([5])
    x0_start = np.tile(x0, int(np.ceil(vp.k / x0.shape[0])))
    vp.w = np.ones((1, k)) / k
    vp.mu = x0_start.T + 1e-6 * np.random.randn(vp.d, vp.k)
    vp.sigma = 1e-3 * np.ones((1, k))
    vp.lamb = np.ones((vp.d, 1))
    vp.parameter_transformer = ParameterTransformer(nvars)
    vp.optimize_sigma = True
    vp.optimize_lamb = True
    vp.optimize_mu = True
    vp.optimize_weights = False
    vp.bounds = list()
    return vp


def test_mtv_not_enough_arguments():
    vp = mock_init_vbmc()
    with pytest.raises(ValueError):
        vp.mtv()


def test_mtv_vp_identical():
    vp1 = mock_init_vbmc(k=1, nvars=1)
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = mock_init_vbmc(k=2, nvars=1)
    vp2.mu = np.array([[0, 100]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[1, 0]])
    mtv = vp1.mtv(vp2)
    assert np.isclose(0, mtv, atol=1e-2)


def test_mtv_vp_no_overlap():
    vp1 = mock_init_vbmc(k=1, nvars=1)
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = mock_init_vbmc(k=2, nvars=1)
    vp2.mu = np.array([[0, 100]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[0, 1]])
    mtv = vp1.mtv(vp2)
    assert np.isclose(1, mtv, atol=1e-2)


def test_mtv_sample_identical():
    vp1 = mock_init_vbmc(k=1, nvars=1)
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = mock_init_vbmc(k=2, nvars=1)
    vp2.mu = np.array([[0, 100]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[1, 0]])
    samples, _ = vp2.sample(int(1e5))
    mtv = vp1.mtv(samples=samples, N=int(1e5))
    assert np.isclose(0, mtv, atol=1e-2)


def test_mtv_sample_no_overlap():
    vp1 = mock_init_vbmc(k=1, nvars=1)
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = mock_init_vbmc(k=2, nvars=1)
    vp2.mu = np.array([[0, 100]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[0, 1]])
    samples, _ = vp2.sample(int(1e5))
    mtv = vp1.mtv(samples=samples, N=int(1e5))
    assert np.isclose(1, mtv, atol=1e-2)


def test_mtv_sample_some_overlap():
    vp1 = mock_init_vbmc(k=1, nvars=1)
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = mock_init_vbmc(k=2, nvars=1)
    vp2.mu = np.array([[0, 5]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[0.5, 0.5]])
    samples, k = vp2.sample(int(1e5))
    mtv = vp1.mtv(samples=samples, N=int(1e5))
    assert np.isclose(0.5, mtv, atol=1e-2)