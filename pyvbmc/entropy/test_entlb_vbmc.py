import numpy as np
from scipy.io import loadmat
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.entropy import entlb_vbmc
from pyvbmc.utils.testing import check_grad


def entlb_vbmc_wrapper(theta, D, K, ret="H"):
    assert theta.shape[0] == D * K + K + D + K
    vp = VariationalPosterior(D, K)
    vp.mu = np.reshape(theta[: D * K], (D, K))
    vp.sigma = theta[D * K : D * K + K]
    vp.lamb = theta[D * K + K : D * K + K + D]
    vp.w = theta[D * K + K + D :]

    if ret == "H":
        H, _ = entlb_vbmc(
            vp, grad_flags=tuple([False] * 4), jacobian_flag=False
        )
        return H
    else:
        _, dH = entlb_vbmc(
            vp, grad_flags=tuple([True] * 4), jacobian_flag=False
        )
        return dH


def test_entlb_vbmc_multi():
    # Check with multiple Gaussians that nearly have non-overlapping supports
    D, K = 3, 2
    vp = VariationalPosterior(D, K)
    vp.mu = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    vp.sigma = np.array([1.0, 1.0])
    vp.lamb = np.ones(D)
    vp.w = np.ones(K) / K

    nconst = 1 / (2 * np.pi) ** (D / 2) / np.prod(vp.lamb)
    H_appro = -np.sum(
        vp.w * np.log(vp.w * nconst / (2 * vp.sigma ** 2) ** (D / 2))
    )
    dH_appro_mu = np.zeros(D * K)
    dH_appro_lambd = (vp.w[:, None] / vp.lamb).sum(0)
    dH_appro_sigma = vp.w / vp.sigma * D
    dH_appro_w = -np.log(vp.w * nconst / (2 * vp.sigma ** 2) ** (D / 2)) - 1
    dH_appro = np.concatenate(
        [dH_appro_mu, dH_appro_sigma, dH_appro_lambd, dH_appro_w]
    )

    H, dH = entlb_vbmc(vp, jacobian_flag=False)

    assert np.isclose(H, H_appro, rtol=0.01)
    assert np.allclose(dH, dH_appro, rtol=0.01)

    # Check gradients
    theta0 = np.concatenate(
        [x.flatten() for x in [vp.mu, vp.sigma, vp.lamb, vp.w]]
    )
    f = lambda theta: entlb_vbmc_wrapper(theta, D, K, "H")
    f_grad = lambda theta: entlb_vbmc_wrapper(theta, D, K, "dH")
    assert check_grad(f, f_grad, theta0, rtol=0.01)


def test_entlb_vbmc_matlab():
    mat = loadmat("./pyvbmc/entropy/entropy-test.mat")
    D = mat["D"].item()
    K = mat["K"].item()
    vp = VariationalPosterior(D, K)
    vp.w = mat["vp"]["w"].item()
    vp.mu = mat["vp"]["mu"].item()
    vp.sigma = mat["vp"]["sigma"].item()
    vp.lamb = mat["vp"]["lambda"].item()
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
