import importlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pyvbmc.entropy import entmc_vbmc
from pyvbmc.testing import check_grad
from pyvbmc.variational_posterior import VariationalPosterior


def single_gaussian_entropy(D, sigma, lambd):
    H = (
        0.5 * D * (1 + np.log(2 * np.pi))
        + D * np.log(sigma).sum()
        + np.log(lambd).sum()
    )
    dH = np.concatenate(
        [
            np.zeros(D),
            D / sigma.flatten(),
            1 / lambd.flatten(),
            np.array([H - 1]),
        ]
    )
    return H, dH


def entmc_vbmc_wrapper(theta, D, K, Ns=1e5, ret="H"):
    assert theta.shape[0] == D * K + K + D + K
    vp = VariationalPosterior(D, K)
    vp.mu = np.reshape(theta[: D * K], (D, K), "F")
    vp.sigma = theta[D * K : D * K + K]
    vp.lambd = theta[D * K + K : D * K + K + D]
    vp.w = theta[D * K + K + D :]

    # A fixed seed per call gives common random numbers to the value and
    # gradient evaluations (important for numerical gradient testing).
    if ret == "H":
        H, _ = entmc_vbmc(
            vp,
            Ns,
            grad_flags=tuple([False] * 4),
            jacobian_flag=False,
            rng=42,
        )
        return H
    else:
        _, dH = entmc_vbmc(
            vp,
            Ns,
            grad_flags=tuple([True] * 4),
            jacobian_flag=False,
            rng=42,
        )
        return dH


def test_entmc_vbmc_single_gaussian():
    # Check with a single Gaussian. For K = 1 the numerical gradient of the
    # sample-based entropy is exact, while the reparameterization gradient
    # carries Monte Carlo error of relative order sqrt(4 / Ns) per
    # coordinate (0.6% at Ns = 1e5), so the gradient check below uses
    # rtol = 0.03, about five standard deviations.
    D, K, Ns = 3, 1, 1e5
    vp = VariationalPosterior(D, K)
    vp.mu = np.ones((D, K))
    vp.sigma = np.ones((1, K))

    H_exact, dH_exact = single_gaussian_entropy(D, vp.sigma, vp.lambd)
    H, dH = entmc_vbmc(vp, Ns, jacobian_flag=False)

    assert np.isclose(H, H_exact, rtol=0.01, atol=0.01)
    assert np.allclose(dH, dH_exact, rtol=0.01, atol=0.01)

    # Check gradients
    theta0 = np.concatenate(
        [x.flatten() for x in [vp.mu.transpose(), vp.sigma, vp.lambd, vp.w]]
    )
    f = lambda theta: entmc_vbmc_wrapper(theta, D, K, Ns, "H")
    f_grad = lambda theta: entmc_vbmc_wrapper(theta, D, K, Ns, "dH")
    assert check_grad(f, f_grad, theta0, rtol=0.03)


def test_entmc_vbmc_nonoverlapping_mixture():
    # Check with multiple Gaussians that nearly have non-overlapping supports
    Ns = 1e5
    for D in range(1, 3):
        for K in range(2, 4):
            vp = VariationalPosterior(D, K)
            vp.mu = np.stack([np.ones(D) * 10 * i for i in range(K)], 1)
            vp.sigma = np.ones(K)
            vp.lambd = np.ones(D)
            vp.w = np.ones(K) / K

            H_appro = 0
            dH_appro = np.zeros(D * K + K + D + K)
            for k in range(K):
                H_appro_k, _ = single_gaussian_entropy(
                    D, vp.sigma[k], vp.lambd
                )
                H_appro += vp.w[k] * H_appro_k - vp.w[k] * np.log(vp.w[k])
                dH_appro[D * k : D * (k + 1)] = 0  # mu
                dH_appro[D * K + k] = D / vp.sigma[k] * vp.w[k]  # sigma
                dH_appro[D * K + K : D * K + K + D] += (
                    1 / vp.lambd.flatten() * vp.w[k]
                )  # lambda
                dH_appro[D * K + K + D + k] = (
                    H_appro_k - 1 - np.log(vp.w[k])
                )  # w

            H, dH = entmc_vbmc(vp, Ns, jacobian_flag=False)

            assert np.isclose(H, H_appro, rtol=0.01, atol=0.01)
            assert np.allclose(dH, dH_appro, rtol=0.01, atol=0.01)

            # Check gradients
            theta0 = np.concatenate(
                [
                    x.flatten()
                    for x in [vp.mu.transpose(), vp.sigma, vp.lambd, vp.w]
                ]
            )
            f = lambda theta: entmc_vbmc_wrapper(theta, D, K, Ns, "H")
            f_grad = lambda theta: entmc_vbmc_wrapper(theta, D, K, Ns, "dH")
            check_grad(f, f_grad, theta0, rtol=0.01)


def test_entmc_vbmc_overlapping_mixture():
    # Check gradients with multiple Gaussians that have overlapping supports
    state = np.random.get_state()
    np.random.seed(42)
    D, K, Ns = 3, 2, 1e5
    vp = VariationalPosterior(D, K)
    vp.mu = np.random.uniform(-1, 1, size=(D, K))
    vp.sigma = np.abs(np.ones(K) + 0.2 * np.random.rand(K))
    vp.lambd = np.abs(np.ones(D) + 0.2 * np.random.rand(D))
    vp.eta = [0.6, 0.4]
    vp.w = np.exp(vp.eta) / np.exp(vp.eta).sum()

    theta0 = np.concatenate(
        [x.flatten() for x in [vp.mu.transpose(), vp.sigma, vp.lambd, vp.w]]
    )

    f = lambda theta: entmc_vbmc_wrapper(theta, D, K, Ns, "H")
    f_grad = lambda theta: entmc_vbmc_wrapper(theta, D, K, Ns, "dH")
    np.random.set_state(state)
    assert check_grad(f, f_grad, theta0, rtol=0.01, atol=0.01)


def test_entmc_vbmc_matlab():
    # If exact is True, random seeds and samples should be the same
    # with MATLAB version, i.e. entmc_vbmc.py need to be modified a
    # bit: epsilon[: Ns // 2, :] = randn2(D, Ns // 2).transpose()
    exact = False
    path = Path(__file__).parent.joinpath("entropy-test.mat")
    mat = loadmat(path)
    D = mat["D"].item()
    K = mat["K"].item()
    Ns = mat["Ns"].item()
    vp = VariationalPosterior(D, K)
    vp.w = mat["vp"]["w"].item().astype(float)
    vp.mu = mat["vp"]["mu"].item().astype(float)
    vp.sigma = mat["vp"]["sigma"].item().astype(float)
    vp.lambd = mat["vp"]["lambda"].item().astype(float)
    vp.eta = mat["vp"]["eta"].item().astype(float)
    Hm = mat["H"].item()
    dHm = mat["dH"].squeeze()
    jacobian_flag = mat["jacobian_flag"].item()

    H, dH = entmc_vbmc(
        vp,
        Ns,
        grad_flags=tuple([True] * 4),
        jacobian_flag=jacobian_flag,
        rng=42,  # Random seed used in MATLAB
    )
    if exact:
        assert np.isclose(H, Hm)
        assert np.allclose(dH, dHm)
    else:
        assert np.isclose(H, Hm, rtol=0.01)
        assert np.allclose(dH, dHm, rtol=0.01, atol=0.01)


def test_entmc_vbmc_grad_flags():
    D, K = 4, 3
    vp = VariationalPosterior(D, K)
    grad_flags = tuple([False] * 4)
    _, dH = entmc_vbmc(vp, Ns=1e5, grad_flags=grad_flags)
    assert dH.shape == (0,)

    grad_flags = tuple([False] * 3) + (True,)
    _, dH = entmc_vbmc(vp, Ns=1e5, grad_flags=grad_flags)
    assert dH.shape == (K,)


def test_entmc_vbmc_block_size_invariance(monkeypatch):
    """The computation runs in blocks of components (and of samples when
    one component's distance tensor exceeds the budget); the block size
    changes only the order of sums, not the estimate."""
    # The package re-exports the function under the module's name, so the
    # module itself is fetched from the import system.
    mod = importlib.import_module("pyvbmc.entropy.entmc_vbmc")

    rs = np.random.default_rng(11)
    D, K = 3, 7
    vp = VariationalPosterior(D, K)
    vp.mu = rs.normal(0, 1, (D, K))
    vp.sigma = np.exp(rs.normal(0, 0.3, (1, K)))
    vp.lambd = np.exp(rs.normal(0, 0.3, (D, 1)))
    vp.eta = rs.normal(0, 0.5, (1, K))
    vp.w = np.exp(vp.eta) / np.exp(vp.eta).sum()

    # Ns * D * K = 40 * 3 * 7 = 840 elements per component: the default
    # budget takes all components in one block; 2600 elements give blocks
    # of 3, 3 and 1 components (a partial last block, as K = 50 in
    # production: 11, 11, 11, 11, 6); 1000 force one component per block;
    # 500 force blocks of samples within a component of 23 and 17 samples
    # (500 // (D * K) = 23); 100 give ten blocks of 4 samples.
    Ns = 40
    H_ref, dH_ref = entmc_vbmc(vp, Ns, rng=5)
    _, dH_ref_raw = entmc_vbmc(vp, Ns, jacobian_flag=False, rng=5)
    for budget in (2600, 1000, 500, 100):
        monkeypatch.setattr(mod, "_MAX_TENSOR_ELEMENTS", budget)
        H, dH = entmc_vbmc(vp, Ns, rng=5)
        assert np.isclose(H, H_ref, rtol=1e-12, atol=0)
        assert np.allclose(dH, dH_ref, rtol=1e-12, atol=1e-14)
        _, dH_raw = entmc_vbmc(vp, Ns, jacobian_flag=False, rng=5)
        assert np.allclose(dH_raw, dH_ref_raw, rtol=1e-12, atol=1e-14)
        # Partial gradient requests: only the requested blocks, same values
        _, dH_w = entmc_vbmc(
            vp, Ns, grad_flags=(False, False, False, True), rng=5
        )
        assert np.allclose(dH_w, dH_ref[-K:], rtol=1e-12, atol=1e-14)
        _, dH_s = entmc_vbmc(
            vp, Ns, grad_flags=(False, True, False, False), rng=5
        )
        assert np.allclose(
            dH_s, dH_ref[D * K : D * K + K], rtol=1e-12, atol=1e-14
        )


if __name__ == "__main__":
    test_entmc_vbmc_overlapping_mixture()
