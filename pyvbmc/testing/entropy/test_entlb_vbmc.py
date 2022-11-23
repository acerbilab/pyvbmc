from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pyvbmc.entropy import entlb_vbmc
from pyvbmc.testing import check_grad
from pyvbmc.variational_posterior import VariationalPosterior


def entlb_vbmc_wrapper(theta, D, K, ret="H"):
    assert theta.shape[0] == D * K + K + D + K
    vp = VariationalPosterior(D, K)
    vp.mu = np.reshape(theta[: D * K], (D, K), "F")
    vp.sigma = theta[D * K : D * K + K]
    vp.lambd = theta[D * K + K : D * K + K + D]
    vp.w = theta[D * K + K + D :]

    if ret == "H":
        H, _ = entlb_vbmc(
            vp, grad_flags=tuple([False] * 4), jacobian_flag=False
        )
        return H
    else:
        _, dH = entlb_vbmc(vp, grad_flags=(True,) * 4, jacobian_flag=False)
        return dH


def test_entlb_vbmc_single_gaussian():
    # Check gradients with a single Gaussian
    D, K = 3, 1
    vp = VariationalPosterior(D, K)
    vp.mu = np.ones((D, K))
    vp.sigma = np.ones((1, K))

    Hl, dHl = entlb_vbmc(vp, jacobian_flag=False)
    theta0 = np.concatenate(
        [x.flatten() for x in [vp.mu.transpose(), vp.sigma, vp.lambd, vp.w]]
    )
    f = lambda theta: entlb_vbmc_wrapper(theta, D, K, "H")
    f_grad = lambda theta: entlb_vbmc_wrapper(theta, D, K, "dH")
    assert check_grad(f, f_grad, theta0, rtol=0.01)


def test_entlb_vbmc_nonoverlapping_mixture():
    # Check with multiple Gaussians that nearly have non-overlapping supports
    D, K = 3, 2
    vp = VariationalPosterior(D, K)
    vp.mu = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    vp.sigma = np.array([1.0, 1.0])
    vp.lambd = np.ones(D)
    vp.w = np.ones(K) / K

    nconst = 1 / (2 * np.pi) ** (D / 2) / np.prod(vp.lambd)
    H_appro = -np.sum(
        vp.w * np.log(vp.w * nconst / (2 * vp.sigma**2) ** (D / 2))
    )
    dH_appro_mu = np.zeros(D * K)
    dH_appro_lambd = (vp.w[:, None] / vp.lambd).sum(0)
    dH_appro_sigma = vp.w / vp.sigma * D
    dH_appro_w = -np.log(vp.w * nconst / (2 * vp.sigma**2) ** (D / 2)) - 1
    dH_appro = np.concatenate(
        [dH_appro_mu, dH_appro_sigma, dH_appro_lambd, dH_appro_w]
    )

    H, dH = entlb_vbmc(vp, jacobian_flag=False)

    assert np.isclose(H, H_appro, rtol=0.01)
    assert np.allclose(dH, dH_appro, rtol=0.01)

    # Check gradients
    theta0 = np.concatenate(
        [x.flatten() for x in [vp.mu.transpose(), vp.sigma, vp.lambd, vp.w]]
    )
    f = lambda theta: entlb_vbmc_wrapper(theta, D, K, "H")
    f_grad = lambda theta: entlb_vbmc_wrapper(theta, D, K, "dH")
    assert check_grad(f, f_grad, theta0, rtol=0.01)


def test_entlb_vbmc_overlapping_mixture():
    # Check gradients with multiple Gaussians that have overlapping supports
    state = np.random.get_state()
    np.random.seed(42)
    D, K = 3, 2
    vp = VariationalPosterior(D, K)
    vp.mu = np.random.uniform(-1, 1, size=(D, K))
    vp.sigma = np.ones(K) + 0.2 * np.random.rand(K)
    vp.lambd = np.ones(D) + 0.2 * np.random.rand(D)
    vp.eta = np.random.rand(K)
    vp.w = np.exp(vp.eta) / np.exp(vp.eta).sum()

    theta0 = np.concatenate(
        [x.flatten() for x in [vp.mu.transpose(), vp.sigma, vp.lambd, vp.w]]
    )
    f = lambda theta: entlb_vbmc_wrapper(theta, D, K, "H")
    f_grad = lambda theta: entlb_vbmc_wrapper(theta, D, K, "dH")
    np.random.set_state(state)
    assert check_grad(f, f_grad, theta0, rtol=0.01)


def test_entlb_vbmc_matlab():
    path = Path(__file__).parent.joinpath("entropy-test.mat")
    mat = loadmat(path)
    D = mat["D"].item()
    K = mat["K"].item()
    vp = VariationalPosterior(D, K)
    vp.w = mat["vp"]["w"].item()
    vp.mu = mat["vp"]["mu"].item()
    vp.sigma = mat["vp"]["sigma"].item()
    vp.lambd = mat["vp"]["lambda"].item()
    vp.eta = mat["vp"]["eta"].item()
    Hlm = mat["Hl"].item()
    dHlm = mat["dHl"].squeeze()
    jacobian_flag = mat["jacobian_flag"].item()

    Hl, dHl = entlb_vbmc(vp, jacobian_flag=jacobian_flag)

    assert np.isclose(Hl, Hlm)
    assert np.allclose(dHl, dHlm)


def test_entlb_vbmc_grad_flags():
    D, K = 4, 3
    vp = VariationalPosterior(D, K)
    grad_flags = tuple([False] * 4)
    _, dH = entlb_vbmc(vp, grad_flags=grad_flags)
    assert dH.shape == (0,)

    grad_flags = tuple([False] * 3) + (True,)
    _, dH = entlb_vbmc(vp, grad_flags=grad_flags)
    assert dH.shape == (K,)
