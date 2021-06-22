import numpy as np
import pytest
from pyvbmc.parameter_transformer import ParameterTransformer
from scipy.io import loadmat

from pyvbmc.variational_posterior import VariationalPosterior


def get_matlab_vp():
    mat = loadmat("./pyvbmc/variational_posterior/vp-test.mat")
    vp = VariationalPosterior(2, 2, np.array([[5]]))
    vp.D = mat["D"][0, 0]
    vp.K = mat["K"][0, 0]
    vp.w = mat["w"]
    vp.mu = mat["mu"]
    vp.sigma = mat["sigma"]
    vp.lamb = mat["lambda"]
    vp.optimize_lamb = mat["optimize_lambda"][0, 0] == 1
    vp.optimize_mu = mat["optimize_mu"][0, 0] == 1
    vp.optimize_sigma = mat["optimize_sigma"][0, 0] == 1
    vp.optimize_weights = mat["optimize_weights"][0, 0] == 1
    vp.parameter_transformer = ParameterTransformer(vp.D)
    return vp


def test_sample_n_lower_1():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    x, i = vp.sample(0)
    assert np.all(x.shape == np.zeros((0, 3)).shape)
    assert np.all(i.shape == np.zeros((0, 1)).shape)
    assert np.all(x == np.zeros((0, 3)))
    assert np.all(i == np.zeros((0, 1)))


def test_sample_default():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    N = int(1e6)
    x, i = vp.sample(N)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    assert 0 in i
    assert 1 in i


def test_sample_balance_no_extra():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    N = 10
    x, i = vp.sample(N, balanceflag=True)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    _, counts = np.unique(i, return_counts=True)
    assert np.all(counts == N / 2)


def test_sample_balance_extra():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    N = 11
    x, i = vp.sample(N, balanceflag=True)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    _, counts = np.unique(i, return_counts=True)
    assert np.all(np.isin(counts, np.array([N // 2, N // 2 + 1])))


def test_sample_one_k():
    vp = VariationalPosterior(3, 1, np.array([[5]]))
    N = 11
    x, i = vp.sample(N)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    _, counts = np.unique(i, return_counts=True)
    assert counts[0] == N


def test_sample_one_k_df():
    vp = VariationalPosterior(3, 1, np.array([[5]]))
    N = 11
    x, i = vp.sample(N, df=20)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    _, counts = np.unique(i, return_counts=True)
    assert counts[0] == N


def test_sample_df():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    N = int(1e4)
    x, i = vp.sample(N, df=20)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    assert 0 in i
    assert 1 in i


def test_sample_no_origflag():
    vp = VariationalPosterior(3, 1, np.array([[5]]))
    N = 11
    x, i = vp.sample(N, origflag=False)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    _, counts = np.unique(i, return_counts=True)
    assert counts[0] == N


def test_pdf_default_no_origflag():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((N, D)) * 4.996
    y = vp.pdf(x, origflag=False)
    assert y.shape == (N, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(
        np.isclose(
            y, 0.002396970183585 * np.ones((N, 1)), rtol=1e-12, atol=1e-14
        )
    )


def test_pdf_grad_default_no_origflag():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((N, D)) * 4.996
    y, dy = vp.pdf(x, origflag=False, gradflag=True)
    assert y.shape == (N, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(
        np.isclose(
            y, 0.002396970183585 * np.ones((N, 1)), rtol=1e-12, atol=1e-14
        )
    )
    assert dy.shape == x.shape
    assert np.isscalar(dy[0, 0])
    assert np.all(
        np.isclose(
            dy, 9.58788073433898 * np.ones((N, 3)), rtol=1e-12, atol=1e-14
        )
    )


def test_pdf_grad_logflag_no_origflag():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((N, D)) * 4.996
    y, dy = vp.pdf(x, origflag=False, logflag=True, gradflag=True)
    assert y.shape == (N, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(
        np.isclose(
            y,
            np.log(0.002396970183585 * np.ones((N, 1))),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    assert dy.shape == x.shape
    assert np.isscalar(dy[0, 0])
    assert np.all(
        np.isclose(
            dy,
            9.58788073433898 * np.ones((N, 3)) / np.exp(y),
            rtol=1e-12,
            atol=1e-14,
        )
    )


def test_pdf_grad_origflag():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((N, D)) * 4.996
    y, dy = vp.pdf(x, gradflag=True)
    assert y.shape == (N, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(
        np.isclose(
            y, 0.002396970183585 * np.ones((N, 1)), rtol=1e-12, atol=1e-14
        )
    )
    assert dy.shape == x.shape
    assert np.isscalar(dy[0, 0])
    assert np.all(
        np.isclose(
            dy, 9.58788073433898 * np.ones((N, 3)), rtol=1e-12, atol=1e-14
        )
    )


def test_pdf_df_real_positive():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.repeat([4.99, 4.996], [10, 10])[:, np.newaxis] * np.ones((1, D))
    y = vp.pdf(x, origflag=False, df=10)
    assert y.shape == (N, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(np.isclose(y[:10], 0.01378581338784, rtol=1e-12, atol=1e-14))
    assert np.all(np.isclose(y[10:], 743.0216137262, rtol=1e-12, atol=1e-14))


def test_pdf_df_real_negative():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.repeat([4.995, 4.99], [10, 10])[:, np.newaxis] * np.ones((1, D))
    y = vp.pdf(x, origflag=False, df=-2)
    assert y.shape == (N, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(np.isclose(y[:10], 362.1287964795, rtol=1e-12, atol=1e-14))
    assert np.all(np.isclose(y[10:], 0.914743278670, rtol=1e-12, atol=1e-14))


def test_pdf_heavy_tailed_pdf_gradient():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    x = np.ones((1, 3))
    with pytest.raises(NotImplementedError):
        vp.pdf(x, df=300, gradflag=True)
    with pytest.raises(NotImplementedError):
        vp.pdf(x, df=-300, gradflag=True)


def test_pdf_origflag_gradient():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    with pytest.raises(NotImplementedError):
        vp.pdf(vp.mu.T, origflag=True, logflag=True, gradflag=True)


def test_set_parameters_raw():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    theta_size = D * K + 2 * K + D
    rng = np.random.default_rng()
    theta = rng.random(theta_size)
    vp.optimize_weights = True
    vp.set_parameters(theta)
    assert vp.mu.shape == (D, K)
    assert np.all(vp.mu[: D * K] == np.reshape(theta[: D * K], (D, K)))
    lamb = np.exp(theta[D * K + K : D * K + K + D])
    nl = np.sqrt(np.sum(lamb ** 2) / D)
    assert vp.sigma.shape == (1, K)
    assert np.all(
        vp.sigma == np.exp(theta[D * K : D * K + K]).reshape(1, -1) * nl
    )
    assert vp.lamb.shape == (D, 1)
    assert np.all(vp.lamb == np.array([lamb]).reshape(-1, 1) / nl)
    assert vp.w.shape == (1, K)
    w = np.exp(theta[-K:] - np.amax(theta[-K:]))
    w = w.reshape(1, -1) / np.sum(w)
    assert np.all(vp.w == w)


def test_set_parameters_not_raw():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    theta_size = D * K + 2 * K + D
    rng = np.random.default_rng()
    theta = rng.random(theta_size)
    vp.optimize_weights = True
    vp.set_parameters(theta, rawflag=False)
    assert vp.mu.shape == (D, K)
    assert np.all(vp.mu[: D * K] == np.reshape(theta[: D * K], (D, K)))
    lamb = theta[D * K + K : D * K + K + D]
    nl = np.sqrt(np.sum(lamb ** 2) / D)
    assert vp.sigma.shape == (1, K)
    assert np.all(vp.sigma == theta[D * K : D * K + K].reshape(1, -1) * nl)
    assert vp.lamb.shape == (D, 1)
    assert np.all(vp.lamb == np.array([lamb]).reshape(-1, 1) / nl)
    assert vp.w.shape == (1, K)
    w = theta[-K:]
    w = w.reshape(1, -1) / np.sum(w)
    assert np.all(vp.w == w)


def test_set_parameters_not_raw_negative_error():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta_size = D * K + 2 * K + D
    rng = np.random.default_rng()
    theta = rng.random(theta_size) * -1
    with pytest.raises(ValueError):
        vp.set_parameters(theta, rawflag=False)


def test_get_parameters_raw():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta = vp.get_parameters(rawflag=True)
    assert np.all(vp.mu[: D * K] == np.reshape(theta[: D * K], (D, K)))
    assert np.all(
        np.isclose(
            vp.sigma.flatten(),
            np.exp(theta[D * K : D * K + K]),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    assert np.all(
        vp.lamb.flatten() == np.exp(theta[D * K + K : D * K + K + D])
    )
    assert np.all(vp.w.flatten() == np.exp(theta[-K:]))


def test_get_parameters_not_raw():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta = vp.get_parameters(rawflag=False)
    assert np.all(vp.mu[: D * K] == np.reshape(theta[: D * K], (D, K)))
    assert np.all(vp.sigma.flatten() == theta[D * K : D * K + K])
    assert np.all(vp.lamb.flatten() == theta[D * K + K : D * K + K + D])
    assert np.all(vp.w.flatten() == theta[-K:])


def test_get_set_parameters_roundtrip():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta = vp.get_parameters(rawflag=True)
    vp.set_parameters(theta, rawflag=True)
    theta2 = vp.get_parameters(rawflag=True)
    assert theta.shape == theta2.shape
    assert np.all(theta == theta2)


def test_get_set_parameters_roundtrip_no_mu():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_mu = False
    theta = vp.get_parameters(rawflag=True)
    vp.set_parameters(theta, rawflag=True)
    theta2 = vp.get_parameters(rawflag=True)
    assert theta.shape == theta2.shape
    assert np.all(theta == theta2)


def test_get_set_parameters_delete_mode():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    theta = vp.get_parameters(rawflag=True)
    vp._mode = np.ones(D)
    assert hasattr(vp, "_mode")
    vp.set_parameters(theta, rawflag=True)
    assert not hasattr(vp, "_mode")


def test_get_set_parameters_roundtrip_non_raw():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta = vp.get_parameters(rawflag=False)
    vp.set_parameters(theta, rawflag=False)
    theta2 = vp.get_parameters(rawflag=False)
    assert theta.shape == theta2.shape
    assert np.all(theta == theta2)


def test_moments_origflag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    mubar, sigma = vp.moments(N=int(1e6), covflag=True)
    x2, _ = vp.sample(N=int(1e6), origflag=True, balanceflag=True)
    assert mubar.shape == (1, 3)
    assert np.all(np.isclose(mubar, np.mean(x2, axis=0)))
    assert sigma.shape == (3, 3)
    assert np.all(np.isclose(sigma, np.cov(x2.T)))


def test_moments_no_origflag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * [1, 4]
    mubar, sigma = vp.moments(N=1e6, covflag=True, origflag=False)
    assert mubar.shape == (1, 3)
    assert sigma.shape == (3, 3)
    assert np.all(mubar == 2.5)
    sigma2 = np.ones((3, 3)) * 2.25 + np.eye(3) * 1e-3 ** 2
    assert np.all(sigma == sigma2)


def test_moments_no_covflag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    mubar = vp.moments(N=1e6, origflag=False)
    assert mubar.shape == (1, 3)


def test_mode_exists_already():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    vp._mode = np.ones(3)
    mode2 = vp.mode()
    assert np.all(mode2 == vp._mode)


def test_mode_no_origflag():
    vp = get_matlab_vp()
    assert np.all(
        np.isclose([0.0540, -0.1818], vp.mode(origflag=False), atol=1e-4)
    )


def test_mode_origflag():
    vp = get_matlab_vp()
    assert np.all(
        np.isclose([0.0540, -0.1818], vp.mode(origflag=True), atol=1e-4)
    )


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
    vp2.mu = np.array([[0, 100000]])
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
    vp2.mu = np.array([[0, 10000]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[0.5, 0.5]])
    samples, _ = vp2.sample(int(1e5))
    mtv = vp1.mtv(samples=samples, N=int(1e5))
    assert np.isclose(0.5, mtv, atol=1e-2)


def test_kldiv_missing_params():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    with pytest.raises(ValueError):
        vp.kldiv()


def test_kldiv_no_gaussianflag_and_samples():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    with pytest.raises(ValueError):
        vp.kldiv(samples=np.ones(3), gaussflag=False)


def test_kldiv_two_vp_identical_gaussflag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    kldivs = vp.kldiv(vp2=vp, gaussflag=True, N=int(1e6))
    assert np.all(np.isclose(np.zeros(2), kldivs, atol=1e-4))


def test_kldiv_two_vp_identical_samples_gaussflag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    samples, _ = vp.sample(int(1e5))
    kldivs = vp.kldiv(samples=samples, gaussflag=True, N=int(1e6))
    assert np.all(np.isclose(np.zeros(2), kldivs, atol=1e-3))


def test_kldiv_two_vp_gaussflag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    vp2 = VariationalPosterior(1, 1, np.array([[5]]))
    vp.mu = np.zeros((1, 1))
    vp.sigma = np.ones((1, 1))
    vp2.mu = np.ones((1, 1)) * 10
    vp2.sigma = np.ones((1, 1))
    kldivs = vp.kldiv(vp2=vp2, gaussflag=True, N=int(1e6))
    assert np.all(np.isclose(50, kldivs, atol=5e-1))


def test_kldiv_two_vp_samples_gaussflag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    vp2 = VariationalPosterior(1, 1, np.array([[5]]))
    vp.mu = np.ones((1, 1)) * 0.5
    vp.sigma = np.ones((1, 1))
    vp2.mu = np.zeros((1, 1))
    vp2.sigma = np.ones((1, 1))
    samples, _ = vp2.sample(int(1e6))
    kldivs = vp.kldiv(samples=samples, gaussflag=True, N=int(1e6))
    assert np.all(np.isclose(np.ones(2) * 0.1244, kldivs, atol=1e-2))


def test_kldiv_two_vp_identical_no_gaussflag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    kldivs = vp.kldiv(vp2=vp, gaussflag=False, N=int(1e6))
    assert np.all(np.isclose(np.zeros(2), kldivs, atol=1e-4))


def test_kldiv_two_vp_no_gaussflag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    vp2 = VariationalPosterior(1, 1, np.array([[5]]))
    vp.mu = np.ones((1, 1)) * 10
    vp.sigma = np.ones((1, 1))
    vp2.mu = np.zeros((1, 1))
    vp2.sigma = np.ones((1, 1))
    kldivs = vp.kldiv(vp2=vp2, gaussflag=False, N=int(1e6))
    assert np.all(np.isclose(50, kldivs, atol=5e-1))


def test_kldiv_no_samples_gaussflag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    with pytest.raises(ValueError):
        vp.kldiv(vp, gaussflag=True, N=0)