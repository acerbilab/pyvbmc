import numpy as np
import pytest
from parameter_transformer import ParameterTransformer
from variational_posterior import VariationalPosterior


def test_mtv_not_enough_arguments():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    with pytest.raises(ValueError):
        vp.mtv()


def test_mtv_vp_identical():
    vp1 = VariationalPosterior(1, 1, np.array([[5]]))
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = VariationalPosterior(1, 2, np.array([[5]]))
    vp2.mu = np.array([[0, 100]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[1, 0]])
    mtv = vp1.mtv(vp2)
    assert np.isclose(0, mtv, atol=1e-2)


def test_mtv_vp_no_overlap():
    vp1 = VariationalPosterior(1, 1, np.array([[5]]))
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = VariationalPosterior(1, 2, np.array([[5]]))
    vp2.mu = np.array([[0, 100]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[0, 1]])
    mtv = vp1.mtv(vp2)
    assert np.isclose(1, mtv, atol=1e-2)


def test_mtv_sample_identical():
    vp1 = VariationalPosterior(1, 1, np.array([[5]]))
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = VariationalPosterior(1, 2, np.array([[5]]))
    vp2.mu = np.array([[0, 100]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[1, 0]])
    samples, _ = vp2.sample(int(1e5))
    mtv = vp1.mtv(samples=samples, N=int(1e5))
    assert np.isclose(0, mtv, atol=1e-2)


def test_mtv_sample_no_overlap():
    vp1 = VariationalPosterior(1, 1, np.array([[5]]))
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = VariationalPosterior(1, 2, np.array([[5]]))
    vp2.mu = np.array([[0, 100]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[0, 1]])
    samples, _ = vp2.sample(int(1e5))
    mtv = vp1.mtv(samples=samples, N=int(1e5))
    assert np.isclose(1, mtv, atol=1e-2)


def test_mtv_sample_some_overlap():
    vp1 = VariationalPosterior(1, 1, np.array([[5]]))
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = VariationalPosterior(1, 2, np.array([[5]]))
    vp2.mu = np.array([[0, 5]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[0.5, 0.5]])
    samples, k = vp2.sample(int(1e5))
    mtv = vp1.mtv(samples=samples, N=int(1e5))
    assert np.isclose(0.5, mtv, atol=1e-2)