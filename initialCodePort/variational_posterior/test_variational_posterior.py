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
    _, counts = np.unique(i, return_counts=True)
    assert np.all(counts == n / 2)


def test_sample_balance_extra():
    vp = mock_init_vbmc(k=2, nvars=3)
    n = 11
    x, i = vp.sample(n, balanceflag=True)
    assert np.all(x.shape == (n, 3))
    assert np.all(i.shape[0] == n)
    _, counts = np.unique(i, return_counts=True)
    assert np.all(np.isin(counts, np.array([n // 2, n // 2 + 1])))


def test_sample_one_k():
    vp = mock_init_vbmc(k=1, nvars=3)
    n = 11
    x, i = vp.sample(n)
    assert np.all(x.shape == (n, 3))
    assert np.all(i.shape[0] == n)
    _, counts = np.unique(i, return_counts=True)
    assert counts[0] == n


def test_sample_one_k_df():
    vp = mock_init_vbmc(k=1, nvars=3)
    n = 11
    x, i = vp.sample(n, df=20)
    assert np.all(x.shape == (n, 3))
    assert np.all(i.shape[0] == n)
    _, counts = np.unique(i, return_counts=True)
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
    _, counts = np.unique(i, return_counts=True)
    assert counts[0] == n


def test_pdf_default_no_origflag():
    n = 20
    d = 3
    vp = mock_init_vbmc(k=2, nvars=d)
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((n, d)) * 4.996
    y = vp.pdf(x, origflag=False)
    assert y.shape == (n, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(
        np.isclose(
            y, 0.002396970183585 * np.ones((n, 1)), rtol=1e-12, atol=1e-14
        )
    )


def test_pdf_grad_default_no_origflag():
    n = 20
    d = 3
    vp = mock_init_vbmc(k=2, nvars=d)
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((n, d)) * 4.996
    y, dy = vp.pdf(x, origflag=False, gradflag=True)
    assert y.shape == (n, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(
        np.isclose(
            y, 0.002396970183585 * np.ones((n, 1)), rtol=1e-12, atol=1e-14
        )
    )
    assert dy.shape == x.shape
    assert np.isscalar(dy[0, 0])
    assert np.all(
        np.isclose(
            dy, 9.58788073433898 * np.ones((n, 3)), rtol=1e-12, atol=1e-14
        )
    )


def test_pdf_grad_logflag_no_origflag():
    n = 20
    d = 3
    vp = mock_init_vbmc(k=2, nvars=d)
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((n, d)) * 4.996
    y, dy = vp.pdf(x, origflag=False, logflag=True, gradflag=True)
    assert y.shape == (n, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(
        np.isclose(
            y,
            np.log(0.002396970183585 * np.ones((n, 1))),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    assert dy.shape == x.shape
    assert np.isscalar(dy[0, 0])
    assert np.all(
        np.isclose(
            dy,
            9.58788073433898 * np.ones((n, 3)) / np.exp(y),
            rtol=1e-12,
            atol=1e-14,
        )
    )


def test_pdf_grad_origflag():
    n = 20
    d = 3
    vp = mock_init_vbmc(k=2, nvars=d)
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((n, d)) * 4.996
    y, dy = vp.pdf(x, gradflag=True)
    assert y.shape == (n, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(
        np.isclose(
            y, 0.002396970183585 * np.ones((n, 1)), rtol=1e-12, atol=1e-14
        )
    )
    assert dy.shape == x.shape
    assert np.isscalar(dy[0, 0])
    assert np.all(
        np.isclose(
            dy, 9.58788073433898 * np.ones((n, 3)), rtol=1e-12, atol=1e-14
        )
    )


def test_pdf_df_real_positive():
    n = 20
    d = 3
    vp = mock_init_vbmc(k=2, nvars=d)
    vp.mu = np.ones((3, 2)) * 5
    x = np.repeat([4.99, 4.996], [10, 10])[:, np.newaxis] * np.ones((1, d))
    y = vp.pdf(x, origflag=False, df=10)
    assert y.shape == (n, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(np.isclose(y[:10], 0.01378581338784, rtol=1e-12, atol=1e-14))
    assert np.all(np.isclose(y[10:], 743.0216137262, rtol=1e-12, atol=1e-14))


def test_pdf_df_real_negative():
    n = 20
    d = 3
    vp = mock_init_vbmc(k=2, nvars=d)
    vp.mu = np.ones((3, 2)) * 5
    x = np.repeat([4.995, 4.99], [10, 10])[:, np.newaxis] * np.ones((1, d))
    y = vp.pdf(x, origflag=False, df=-2)
    assert y.shape == (n, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(np.isclose(y[:10], 362.1287964795, rtol=1e-12, atol=1e-14))
    assert np.all(np.isclose(y[10:], 0.914743278670, rtol=1e-12, atol=1e-14))


def test_set_parameters_raw():
    k = 2
    d = 3
    vp = mock_init_vbmc(k=k, nvars=d)
    theta_size = d * k + 2 * k + d
    rng = np.random.default_rng()
    theta = rng.random(theta_size)
    vp.optimize_weights = True
    vp.set_parameters(theta)
    assert vp.mu.shape == (d, k)
    assert np.all(vp.mu[: d * k] == np.reshape(theta[: d * k], (d, k)))
    lamb = np.exp(theta[d * k + k : d * k + k + d])
    nl = np.sqrt(np.sum(lamb ** 2) / d)
    assert vp.sigma.shape == (k,)
    assert np.all(vp.sigma == np.exp(theta[d * k : d * k + k]).conj().T * nl)
    assert vp.lamb.shape == (d,)
    assert np.all(vp.lamb == lamb / nl)
    assert vp.w.shape == (1, k)
    w = np.exp(theta[-k:] - np.amax(theta[-k:]))
    w = w.conj().T / np.sum(w)
    assert np.all(vp.w == w)


def test_set_parameters_not_raw():
    k = 2
    d = 3
    vp = mock_init_vbmc(k=k, nvars=d)
    theta_size = d * k + 2 * k + d
    rng = np.random.default_rng()
    theta = rng.random(theta_size)
    vp.optimize_weights = True
    vp.set_parameters(theta, rawflag=False)
    assert vp.mu.shape == (d, k)
    assert np.all(vp.mu[: d * k] == np.reshape(theta[: d * k], (d, k)))
    lamb = theta[d * k + k : d * k + k + d]
    nl = np.sqrt(np.sum(lamb ** 2) / d)
    assert vp.sigma.shape == (k,)
    assert np.all(vp.sigma == theta[d * k : d * k + k].conj().T * nl)
    assert vp.lamb.shape == (d,)
    assert np.all(vp.lamb == lamb / nl)
    assert vp.w.shape == (1, k)
    w = theta[-k:]
    w = w.conj().T / np.sum(w)
    assert np.all(vp.w == w)


def test_set_parameters_not_raw_negative_error():
    k = 2
    d = 3
    vp = mock_init_vbmc(k=k, nvars=d)
    vp.optimize_weights = True
    theta_size = d * k + 2 * k + d
    rng = np.random.default_rng()
    theta = rng.random(theta_size) * -1
    with pytest.raises(ValueError):
        vp.set_parameters(theta, rawflag=False)


def test_get_parameters_raw():
    k = 2
    d = 3
    vp = mock_init_vbmc(k=k, nvars=d)
    vp.optimize_weights = True
    theta = vp.get_parameters(rawflag=True)
    assert np.all(vp.mu[: d * k] == np.reshape(theta[: d * k], (d, k)))
    assert np.all(
        np.isclose(
            vp.sigma.flatten(),
            np.exp(theta[d * k : d * k + k]),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    assert np.all(
        vp.lamb.flatten() == np.exp(theta[d * k + k : d * k + k + d])
    )
    assert np.all(vp.w.flatten() == np.exp(theta[-k:]))


def test_get_parameters_not_raw():
    k = 2
    d = 3
    vp = mock_init_vbmc(k=k, nvars=d)
    vp.optimize_weights = True
    theta = vp.get_parameters(rawflag=False)
    assert np.all(vp.mu[: d * k] == np.reshape(theta[: d * k], (d, k)))
    assert np.all(vp.sigma.flatten() == theta[d * k : d * k + k])
    assert np.all(vp.lamb.flatten() == theta[d * k + k : d * k + k + d])
    assert np.all(vp.w.flatten() == theta[-k:])


def test_get_set_parameters_roundtrip():
    k = 2
    d = 3
    vp = mock_init_vbmc(k=k, nvars=d)
    vp.optimize_weights = True
    theta = vp.get_parameters(rawflag=True)
    vp.set_parameters(theta, rawflag=True)
    theta2 = vp.get_parameters(rawflag=True)
    assert theta.shape == theta2.shape
    assert np.all(theta == theta2)


def test_get_set_parameters_roundtrip_non_raw():
    k = 2
    d = 3
    vp = mock_init_vbmc(k=k, nvars=d)
    vp.optimize_weights = True
    theta = vp.get_parameters(rawflag=False)
    vp.set_parameters(theta, rawflag=False)
    theta2 = vp.get_parameters(rawflag=False)
    assert theta.shape == theta2.shape
    assert np.all(theta == theta2)


@pytest.fixture
def test_moments_origflag(mocker):
    rng = np.random.default_rng()
    mocker.patch("vp.sample", return_value=rng.random((20, 3)))
    vp = mock_init_vbmc(k=2, nvars=3)
    mubar, sigma = vp.moments(n=1e6, covflag=True)
    x2, _ = vp.sample(n=1e6, origflag=True, balanceflag=True)
    assert mubar.shape == (3,)
    assert np.all(mubar == np.mean(x2, axis=0))
    assert sigma.shape == (3, 3)
    assert np.all(sigma == np.cov(x2.T))


def test_moments_no_origflag():
    vp = mock_init_vbmc(k=2, nvars=3)
    vp.mu = np.ones((3, 2)) * [1, 4]
    mubar, sigma = vp.moments(n=1e6, covflag=True, origflag=False)
    assert mubar.shape == (3,)
    assert sigma.shape == (3, 3)
    assert np.all(mubar == 2.5)
    sigma2 = np.ones((3, 3)) * 2.25 + np.eye(3) * 1e-3 ** 2
    assert np.all(sigma == sigma2)


def test_moments_no_covflag():
    vp = mock_init_vbmc(k=2, nvars=3)
    mubar = vp.moments(n=1e6, origflag=False)
    assert mubar.shape == (3,)


def test_mode_exists_already():
    vp = mock_init_vbmc()
    vp._mode = np.ones(3)
    mode2 = vp.mode()
    assert np.all(mode2 == vp._mode)


def test_mode_no_origflag():
    vp = mock_init_vbmc()
    vp.mu = np.ones((3, 2)) * [1, 4]
    mode2 = vp.mode(origflag=False)
    assert np.all(1 == mode2)


def test_mode_origflag():
    vp = mock_init_vbmc()
    vp.mu = np.ones((3, 2)) * [1, 4]
    with pytest.raises(NotImplementedError):
        vp.mode(origflag=True)


def test_kldiv_missing_params():
    vp = mock_init_vbmc()
    with pytest.raises(ValueError):
        vp.kldiv()


def test_kldiv_no_gaussianflag_and_samples():
    vp = mock_init_vbmc()
    with pytest.raises(ValueError):
        vp.kldiv(samples=np.ones(3), gaussflag=False)


def test_kldiv_two_vp_identical_gaussflag():
    vp = mock_init_vbmc()
    kldivs = vp.kldiv(vp2=vp, gaussflag=True, N=int(1e6))
    assert np.all(np.isclose(np.zeros(2), kldivs, atol=1e-4))


def test_kldiv_two_vp_identical_samples_gaussflag():
    vp = mock_init_vbmc()
    samples, _ = vp.sample(int(1e5))
    kldivs = vp.kldiv(samples=samples, gaussflag=True, N=int(1e6))
    assert np.all(np.isclose(np.zeros(2), kldivs, atol=1e-3))


def test_kldiv_two_vp_gaussflag():
    vp = mock_init_vbmc(k=1, nvars=1)
    vp2 = mock_init_vbmc(k=1, nvars=1)
    vp.mu = np.zeros((1, 1))
    vp.sigma = np.ones((1, 1))
    vp2.mu = np.ones((1, 1)) * 10
    vp2.sigma = np.ones((1, 1))
    kldivs = vp.kldiv(vp2=vp2, gaussflag=True, N=int(1e6))
    assert np.all(np.isclose(50, kldivs, atol=5e-1))


def test_kldiv_two_vp_samples_gaussflag():
    vp = mock_init_vbmc(k=1, nvars=1)
    vp2 = mock_init_vbmc(k=1, nvars=1)
    vp.mu = np.ones((1, 1)) * 0.5
    vp.sigma = np.ones((1, 1))
    vp2.mu = np.zeros((1, 1))
    vp2.sigma = np.ones((1, 1))
    samples, _ = vp2.sample(int(1e6))
    kldivs = vp.kldiv(samples=samples, gaussflag=True, N=int(1e6))
    assert np.all(np.isclose(np.ones(2) * 0.1244, kldivs, atol=1e-2))


def test_kldiv_two_vp_identical_no_gaussflag():
    vp = mock_init_vbmc()
    kldivs = vp.kldiv(vp2=vp, gaussflag=False, N=int(1e6))
    assert np.all(np.isclose(np.zeros(2), kldivs, atol=1e-4))


def test_kldiv_two_vp_no_gaussflag():
    vp = mock_init_vbmc(k=1, nvars=1)
    vp2 = mock_init_vbmc(k=1, nvars=1)
    vp.mu = np.ones((1, 1)) * 10
    vp.sigma = np.ones((1, 1))
    vp2.mu = np.zeros((1, 1))
    vp2.sigma = np.ones((1, 1))
    kldivs = vp.kldiv(vp2=vp2, gaussflag=False, N=int(1e6))
    assert np.all(np.isclose(50, kldivs, atol=5e-1))
