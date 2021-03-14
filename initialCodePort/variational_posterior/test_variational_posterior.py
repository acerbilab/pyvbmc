import pytest
import numpy as np
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
    vp.parameter_transformer = ParameterTransformer(nvars=nvars)
    vp.optimize_sigma = True
    vp.optimize_lamb = True
    vp.optimize_mu = True
    vp.optimize_weights = False
    vp.bounds = list()
    return vp


def test_sample_n_lower_1():
    vp = mock_init_vbmc(k=2, nvars=3)
    x, i = vp.sample(0)
    assert np.all(x.shape == np.zeros((0, 3)).shape)
    assert np.all(i.shape == np.zeros((0, 1)).shape)
    assert np.all(x == np.zeros((0, 3)))
    assert np.all(i == np.zeros((0, 1)))


def test_sample_default():
    vp = mock_init_vbmc(k=2, nvars=3)
    n = int(1e6)
    x, i = vp.sample(n)
    assert np.all(x.shape == (n, 3))
    assert np.all(i.shape[0] == n)
    assert 0 in i
    assert 1 in i


def test_sample_balance_no_extra():
    vp = mock_init_vbmc(k=2, nvars=3)
    n = 10
    x, i = vp.sample(n, balanceflag=True)
    assert np.all(x.shape == (n, 3))
    assert np.all(i.shape[0] == n)
    unique, counts = np.unique(i, return_counts=True)
    assert np.all(counts == n / 2)


def test_sample_balance_extra():
    vp = mock_init_vbmc(k=2, nvars=3)
    n = 11
    x, i = vp.sample(n, balanceflag=True)
    assert np.all(x.shape == (n, 3))
    assert np.all(i.shape[0] == n)
    unique, counts = np.unique(i, return_counts=True)
    assert np.all(np.isin(counts, np.array([n // 2, n // 2 + 1])))


def test_sample_one_k():
    vp = mock_init_vbmc(k=1, nvars=3)
    n = 11
    x, i = vp.sample(n)
    assert np.all(x.shape == (n, 3))
    assert np.all(i.shape[0] == n)
    unique, counts = np.unique(i, return_counts=True)
    assert counts[0] == n


def test_sample_one_k_df():
    vp = mock_init_vbmc(k=1, nvars=3)
    n = 11
    x, i = vp.sample(n, df=20)
    assert np.all(x.shape == (n, 3))
    assert np.all(i.shape[0] == n)
    unique, counts = np.unique(i, return_counts=True)
    assert counts[0] == n


def test_sample_df():
    vp = mock_init_vbmc(k=2, nvars=3)
    n = int(1e4)
    x, i = vp.sample(n, df=20)
    assert np.all(x.shape == (n, 3))
    assert np.all(i.shape[0] == n)
    assert 0 in i
    assert 1 in i

def test_sample_no_origflag():
    vp = mock_init_vbmc(k=1, nvars=3)
    n = 11
    x, i = vp.sample(n, origflag=False)
    assert np.all(x.shape == (n, 3))
    assert np.all(i.shape[0] == n)
    unique, counts = np.unique(i, return_counts=True)
    assert counts[0] == n