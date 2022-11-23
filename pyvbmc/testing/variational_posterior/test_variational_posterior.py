from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior


def get_matlab_vp():
    path = Path(__file__).parent.joinpath("vp-test.mat")
    mat = loadmat(path)
    vp = VariationalPosterior(2, 2, np.array([[5]]))
    vp.D = mat["D"][0, 0]
    vp.K = mat["K"][0, 0]
    vp.w = mat["w"]
    vp.mu = mat["mu"]
    vp.sigma = mat["sigma"]
    vp.lambd = mat["lambda"]
    vp.optimize_lambd = mat["optimize_lambda"][0, 0] == 1
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
    x, i = vp.sample(N, balance_flag=True)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    _, counts = np.unique(i, return_counts=True)
    assert np.all(counts == N / 2)


def test_sample_balance_extra():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    N = 11
    x, i = vp.sample(N, balance_flag=True)
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


def test_sample_no_orig_flag():
    vp = VariationalPosterior(3, 1, np.array([[5]]))
    N = 11
    x, i = vp.sample(N, orig_flag=False)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    _, counts = np.unique(i, return_counts=True)
    assert counts[0] == N


def test_pdf_default_no_orig_flag():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((N, D)) * 4.996
    y = vp.pdf(x, orig_flag=False)
    assert y.shape == (N, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(
        np.isclose(
            y, 0.002396970183585 * np.ones((N, 1)), rtol=1e-12, atol=1e-14
        )
    )
    log_y = vp.log_pdf(x, orig_flag=False)
    assert log_y.shape == (N, 1)
    assert np.isscalar(log_y[0, 0])
    assert np.all(
        np.isclose(
            log_y,
            np.log(0.002396970183585) * np.ones((N, 1)),
            rtol=1e-12,
            atol=1e-14,
        )
    )


def test_pdf_grad_default_no_orig_flag():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((N, D)) * 4.996
    y, dy = vp.pdf(x, orig_flag=False, grad_flag=True)
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
    log_y, dlog_y = vp.log_pdf(x, orig_flag=False, grad_flag=True)
    assert log_y.shape == (N, 1)
    assert np.isscalar(log_y[0, 0])
    assert np.all(
        np.isclose(
            log_y,
            np.log(0.002396970183585) * np.ones((N, 1)),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    assert dlog_y.shape == x.shape
    assert np.isscalar(dlog_y[0, 0])
    assert np.all(
        np.isclose(
            dlog_y * y,
            9.58788073433898 * np.ones((N, 3)),
            rtol=1e-12,
            atol=1e-14,
        )
    )


def test_pdf_grad_log_flag_no_orig_flag():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((N, D)) * 4.996
    log_y, dlog_y = vp.pdf(x, orig_flag=False, log_flag=True, grad_flag=True)
    assert log_y.shape == (N, 1)
    assert np.isscalar(log_y[0, 0])
    assert np.all(
        np.isclose(
            log_y,
            np.log(0.002396970183585 * np.ones((N, 1))),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    assert dlog_y.shape == x.shape
    assert np.isscalar(dlog_y[0, 0])
    assert np.all(
        np.isclose(
            dlog_y,
            9.58788073433898 * np.ones((N, 3)) / np.exp(log_y),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    log_y_2, dlog_y_2 = vp.log_pdf(x, orig_flag=False, grad_flag=True)
    assert np.all(log_y_2 == log_y)
    assert np.all(dlog_y_2 == dlog_y)


def test_pdf_grad_orig_flag():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((N, D)) * 4.996
    y, dy = vp.pdf(x, grad_flag=True)
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
    with pytest.raises(NotImplementedError) as err:
        log_y, dlog_y = vp.log_pdf(x, grad_flag=True)
        assert "vbmc_pdf:NoOriginalGrad" in err


def test_pdf_df_real_positive():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.repeat([4.99, 4.996], [10, 10])[:, np.newaxis] * np.ones((1, D))
    y = vp.pdf(x, orig_flag=False, df=10)
    assert y.shape == (N, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(np.isclose(y[:10], 0.01378581338784, rtol=1e-12, atol=1e-14))
    assert np.all(np.isclose(y[10:], 743.0216137262, rtol=1e-12, atol=1e-14))
    log_y = vp.log_pdf(x, orig_flag=False, df=10)
    assert log_y.shape == (N, 1)
    assert np.isscalar(log_y[0, 0])
    assert np.all(
        np.isclose(
            log_y[:10], np.log(0.01378581338784), rtol=1e-12, atol=1e-14
        )
    )
    assert np.all(
        np.isclose(log_y[10:], np.log(743.0216137262), rtol=1e-12, atol=1e-14)
    )


def test_pdf_df_real_negative():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.repeat([4.995, 4.99], [10, 10])[:, np.newaxis] * np.ones((1, D))
    y = vp.pdf(x, orig_flag=False, df=-2)
    assert y.shape == (N, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(np.isclose(y[:10], 362.1287964795, rtol=1e-12, atol=1e-14))
    assert np.all(np.isclose(y[10:], 0.914743278670, rtol=1e-12, atol=1e-14))
    log_y = vp.log_pdf(x, orig_flag=False, df=-2)
    assert log_y.shape == (N, 1)
    assert np.isscalar(log_y[0, 0])
    assert np.all(
        np.isclose(log_y[:10], np.log(362.1287964795), rtol=1e-11, atol=1e-13)
    )
    assert np.all(
        np.isclose(log_y[10:], np.log(0.914743278670), rtol=1e-11, atol=1e-13)
    )


def test_pdf_heavy_tailed_pdf_gradient():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    x = np.ones((1, 3))
    with pytest.raises(NotImplementedError):
        vp.pdf(x, df=300, grad_flag=True)
    with pytest.raises(NotImplementedError):
        vp.pdf(x, df=-300, grad_flag=True)
    with pytest.raises(NotImplementedError):
        vp.log_pdf(x, df=300, grad_flag=True)
    with pytest.raises(NotImplementedError):
        vp.log_pdf(x, df=-300, grad_flag=True)


def test_pdf_orig_flag_gradient():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    with pytest.raises(NotImplementedError):
        vp.pdf(vp.mu.T, orig_flag=True, log_flag=True, grad_flag=True)
    with pytest.raises(NotImplementedError):
        vp.log_pdf(vp.mu.T, orig_flag=True, grad_flag=True)


def test_pdf_outside_bounds():
    D = 2
    lb = np.ones((1, D)) * -3
    ub = np.ones((1, D)) * 3
    x0 = np.array([[2, 2], [-2, -2]])
    parameter_transformer = ParameterTransformer(D, lb, ub)

    vp = VariationalPosterior(D, 2, x0, parameter_transformer)
    vp.sigma = np.ones((1, 2))

    # outside or on bounds should be 0
    assert vp.pdf(lb, orig_flag=True) == 0
    assert vp.pdf(lb - 1e-3, orig_flag=True) == 0
    assert vp.pdf(ub, orig_flag=True) == 0
    assert vp.pdf(ub + 1e-3, orig_flag=True) == 0

    # inside should be more than 0
    assert vp.pdf(lb + 0.5, orig_flag=True) > 0
    assert vp.pdf(ub - 0.5, orig_flag=True) > 0

    # outside or on bounds should be -inf
    assert vp.log_pdf(lb, orig_flag=True) == -np.inf
    assert vp.log_pdf(lb - 1e-3, orig_flag=True) == -np.inf
    assert vp.log_pdf(ub, orig_flag=True) == -np.inf
    assert vp.log_pdf(ub + 1e-3, orig_flag=True) == -np.inf

    # inside should be finite
    assert np.all(np.isfinite(vp.log_pdf(lb + 0.5, orig_flag=True)))
    assert np.all(np.isfinite(vp.log_pdf(ub - 0.5, orig_flag=True)))


def test_pdf_duplicate_log_flag():
    D = 2
    lb = np.ones((1, D)) * -3
    ub = np.ones((1, D)) * 3
    x0 = np.array([[2, 2], [-2, -2]])
    parameter_transformer = ParameterTransformer(D, lb, ub)

    vp = VariationalPosterior(D, 2, x0, parameter_transformer)
    vp.sigma = np.ones((1, 2))

    with pytest.raises(TypeError) as err:
        y = vp.log_pdf(lb + 0.5, log_flag=True)
        assert "got multiple values for keyword argument 'log_flag'" in err


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
    assert np.all(
        vp.mu[: D * K] == np.reshape(theta[: D * K], (D, K), order="F")
    )
    lamb = np.exp(theta[D * K + K : D * K + K + D])
    nl = np.sqrt(np.sum(lamb**2) / D)
    assert vp.sigma.shape == (1, K)
    assert np.all(
        vp.sigma == np.exp(theta[D * K : D * K + K]).reshape(1, -1) * nl
    )
    assert vp.lambd.shape == (D, 1)
    assert np.all(vp.lambd == np.array([lamb]).reshape(-1, 1) / nl)
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
    vp.set_parameters(theta, raw_flag=False)
    assert vp.mu.shape == (D, K)
    assert np.all(
        vp.mu[: D * K] == np.reshape(theta[: D * K], (D, K), order="F")
    )
    lamb = theta[D * K + K : D * K + K + D]
    nl = np.sqrt(np.sum(lamb**2) / D)
    assert vp.sigma.shape == (1, K)
    assert np.all(vp.sigma == theta[D * K : D * K + K].reshape(1, -1) * nl)
    assert vp.lambd.shape == (D, 1)
    assert np.all(vp.lambd == np.array([lamb]).reshape(-1, 1) / nl)
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
        vp.set_parameters(theta, raw_flag=False)


def test_get_parameters_raw():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta = vp.get_parameters(raw_flag=True)
    assert np.all(
        vp.mu[: D * K] == np.reshape(theta[: D * K], (D, K), order="F")
    )
    assert np.all(
        np.isclose(
            vp.sigma.flatten(),
            np.exp(theta[D * K : D * K + K]),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    assert np.all(
        vp.lambd.flatten() == np.exp(theta[D * K + K : D * K + K + D])
    )
    assert np.all(vp.w.flatten() == np.exp(theta[-K:]))


def test_get_parameters_not_raw():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta = vp.get_parameters(raw_flag=False)
    assert np.all(
        vp.mu[: D * K] == np.reshape(theta[: D * K], (D, K), order="F")
    )
    assert np.all(vp.sigma.flatten() == theta[D * K : D * K + K])
    assert np.all(vp.lambd.flatten() == theta[D * K + K : D * K + K + D])
    assert np.all(vp.w.flatten() == theta[-K:])


def test_get_set_parameters_roundtrip():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta = vp.get_parameters(raw_flag=True)
    vp.set_parameters(theta, raw_flag=True)
    theta2 = vp.get_parameters(raw_flag=True)
    assert theta.shape == theta2.shape
    assert np.all(theta == theta2)


def test_get_set_parameters_roundtrip_no_mu():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_mu = False
    theta = vp.get_parameters(raw_flag=True)
    vp.set_parameters(theta, raw_flag=True)
    theta2 = vp.get_parameters(raw_flag=True)
    assert theta.shape == theta2.shape
    assert np.all(theta == theta2)


def test_get_set_parameters_delete_mode():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    theta = vp.get_parameters(raw_flag=True)
    vp._mode = np.ones(D)
    assert hasattr(vp, "_mode")
    vp.set_parameters(theta, raw_flag=True)
    assert vp._mode is None


def test_get_set_parameters_roundtrip_non_raw():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta = vp.get_parameters(raw_flag=False)
    vp.set_parameters(theta, raw_flag=False)
    theta2 = vp.get_parameters(raw_flag=False)
    assert theta.shape == theta2.shape
    assert np.all(theta == theta2)


def test_set_parameters_reference_regression():
    K = 2
    D = 2
    vp = VariationalPosterior(D, K)
    theta = vp.get_parameters().copy()
    theta[0] = -1e-7
    vp.set_parameters(theta)

    assert vp.mu[0, 0] == -1e-7

    # Make sure we don't get accidental reference to theta in the VP.
    theta[0] = -2e-7
    assert vp.mu[0, 0] == -1e-7


def test_moments_orig_flag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    mubar, sigma = vp.moments(N=int(1e6), cov_flag=True)
    x2, _ = vp.sample(N=int(1e6), orig_flag=True, balance_flag=True)
    assert mubar.shape == (1, 3)
    assert np.all(np.isclose(mubar, np.mean(x2, axis=0)))
    assert sigma.shape == (3, 3)
    assert np.all(np.isclose(sigma, np.cov(x2.T)))


def test_moments_no_orig_flag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * [1, 4]
    mubar, sigma = vp.moments(N=1e6, cov_flag=True, orig_flag=False)
    assert mubar.shape == (1, 3)
    assert sigma.shape == (3, 3)
    assert np.all(mubar == 2.5)
    sigma2 = np.ones((3, 3)) * 2.25 + np.eye(3) * 1e-3**2
    assert np.all(sigma == sigma2)


def test_moments_no_orig_flag_2():
    # A second test with more unusual (non-ones) vp.lambd
    D = 6
    K = 3
    vp = VariationalPosterior(D, K)
    vp.mu = np.linspace(-3, 3, D * K).reshape([D, K], order="F")
    vp.sigma = np.atleast_2d(np.array(range(2, 5)))
    vp.lambd = np.atleast_2d(np.array(range(3, 9))).T
    vp.w = np.atleast_2d(np.array(range(1, 4)))
    vp.w = vp.w / np.sum(vp.w)

    mubar, sigma = vp.moments(N=1e6, cov_flag=True, orig_flag=False)
    path = Path(__file__).parent.joinpath(
        "test_moments_no_orig_flag_2_MATLAB.mat"
    )
    matlab = loadmat(path)

    assert mubar.shape == (1, 6)
    assert sigma.shape == (6, 6)
    assert np.allclose(mubar, matlab["mubar"])
    assert np.allclose(sigma, matlab["sigma"])


def test_moments_no_cov_flag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    mubar = vp.moments(N=1e6, orig_flag=False)
    assert mubar.shape == (1, 3)


def test_mode_exists_already():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    vp._mode = np.ones(3)
    mode2 = vp.mode()
    assert np.all(mode2 == vp._mode)


def test_mode_no_orig_flag():
    vp = get_matlab_vp()
    assert np.all(
        np.isclose([0.0540, -0.1818], vp.mode(orig_flag=False), atol=1e-4)
    )


def test_mode_orig_flag():
    vp = get_matlab_vp()
    assert np.all(
        np.isclose([0.0540, -0.1818], vp.mode(orig_flag=True), atol=1e-4)
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


def test_kl_div_missing_params():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    with pytest.raises(ValueError):
        vp.kl_div()


def test_kl_div_no_gaussianflag_and_samples():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    with pytest.raises(ValueError):
        vp.kl_div(samples=np.ones(3), gauss_flag=False)


def test_kl_div_two_vp_identical_gauss_flag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    kl_divs = vp.kl_div(vp2=vp, gauss_flag=True, N=int(1e6))
    assert np.all(np.isclose(np.zeros(2), kl_divs, atol=1e-4))


def test_kl_div_two_vp_identical_samples_gauss_flag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    samples, _ = vp.sample(int(1e5))
    kl_divs = vp.kl_div(samples=samples, gauss_flag=True, N=int(1e6))
    assert np.all(np.isclose(np.zeros(2), kl_divs, atol=1e-3))


def test_kl_div_two_vp_gauss_flag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    vp2 = VariationalPosterior(1, 1, np.array([[5]]))
    vp.mu = np.zeros((1, 1))
    vp.sigma = np.ones((1, 1))
    vp2.mu = np.ones((1, 1)) * 10
    vp2.sigma = np.ones((1, 1))
    kl_divs = vp.kl_div(vp2=vp2, gauss_flag=True, N=int(1e6))
    assert np.all(np.isclose(50, kl_divs, atol=5e-1))


def test_kl_div_two_vp_samples_gauss_flag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    vp2 = VariationalPosterior(1, 1, np.array([[5]]))
    vp.mu = np.ones((1, 1)) * 0.5
    vp.sigma = np.ones((1, 1))
    vp2.mu = np.zeros((1, 1))
    vp2.sigma = np.ones((1, 1))
    samples, _ = vp2.sample(int(1e6))
    kl_divs = vp.kl_div(samples=samples, gauss_flag=True, N=int(1e6))
    assert np.all(np.isclose(np.ones(2) * 0.1244, kl_divs, atol=1e-2))


def test_kl_div_two_vp_identical_no_gauss_flag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    kl_divs = vp.kl_div(vp2=vp, gauss_flag=False, N=int(1e6))
    assert np.all(np.isclose(np.zeros(2), kl_divs, atol=1e-4))


def test_kl_div_two_vp_no_gauss_flag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    vp2 = VariationalPosterior(1, 1, np.array([[5]]))
    vp.mu = np.ones((1, 1)) * 10
    vp.sigma = np.ones((1, 1))
    vp2.mu = np.zeros((1, 1))
    vp2.sigma = np.ones((1, 1))
    kl_divs = vp.kl_div(vp2=vp2, gauss_flag=False, N=int(1e6))
    assert np.all(np.isclose(50, kl_divs, atol=5e-1))


def test_kl_div_no_samples_gauss_flag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    with pytest.raises(ValueError):
        vp.kl_div(vp, gauss_flag=True, N=0)


def test_soft_bounds_1():
    D = 2
    K = 1
    vp = VariationalPosterior(D, K)
    assert vp.bounds is None

    # use a fake options struct
    options = {
        "tol_con_loss": 0.01,
        "tol_weight": 1e-2,
        "weight_penalty": 0.1,
        "tol_length": 1e-6,
    }

    # Make up some fake data.
    X = np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T

    theta_bnd = vp.get_bounds(X, options)

    assert vp.bounds is not None
    assert np.all(vp.bounds["mu_lb"] == 0)
    assert np.all(vp.bounds["mu_ub"] == 1)
    assert np.all(vp.bounds["lnscale_lb"] == np.log(options["tol_length"]))
    assert np.all(vp.bounds["lnscale_ub"] == 0)
    assert vp.bounds["eta_lb"] == np.log(0.5 * options["tol_weight"])
    assert vp.bounds["eta_ub"] == 0

    assert theta_bnd["tol_con"] == options["tol_con_loss"]
    assert theta_bnd["weight_threshold"] == max(
        1 / (4 * K), options["tol_weight"]
    )
    assert theta_bnd["weight_penalty"] == options["weight_penalty"]


def test_soft_bounds_2():
    D = 2
    K = 2
    vp = VariationalPosterior(D, K)

    options = {
        "tol_con_loss": 0.01,
        "tol_weight": 1e-2,
        "weight_penalty": 0.1,
        "tol_length": 1e-6,
    }
    base_path = Path(__file__).parent
    X = np.loadtxt(open(base_path.joinpath("X.txt"), "rb"), delimiter=",")
    path = Path(__file__).parent.joinpath("mu.txt")
    vp.mu = np.loadtxt(open(base_path.joinpath("mu.txt"), "rb"), delimiter=",")

    theta_bnd = vp.get_bounds(X, options)

    bnd_lb = np.loadtxt(
        open(base_path.joinpath("bnd_lb.txt"), "rb"), delimiter=","
    )
    assert np.allclose(theta_bnd["lb"], bnd_lb)

    bnd_ub = np.loadtxt(
        open(base_path.joinpath("bnd_ub.txt"), "rb"), delimiter=","
    )
    assert np.allclose(theta_bnd["ub"], bnd_ub)

    assert theta_bnd["tol_con"] == 0.0100
    assert theta_bnd["weight_threshold"] == 0.1250
    assert theta_bnd["weight_penalty"] == 0.1000


def test_plot():
    """
    This is a really naive test of the plotting as everything else is
    complicated.
    """
    D = 2
    K = 2
    vp = VariationalPosterior(D, K)
    test_title = "Test title"
    fig = vp.plot(title=test_title)
    assert fig._suptitle.get_text() == test_title
    assert len(fig.axes) == D * D


def test__str__and__repr__():
    D = 2
    K = 2
    vp = VariationalPosterior(D, K)
    assert "num. components = 2" in vp.__str__()
    assert "self.K = 2" in vp.__repr__()
