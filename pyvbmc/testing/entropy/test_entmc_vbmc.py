import itertools
from pathlib import Path

import numpy as np
import pytest

from pyvbmc.calibration._campaign import CANDIDATE_BUDGETS
from pyvbmc.calibration.profile import DEFAULT_CHUNK_ELEMENTS
from pyvbmc.entropy import entmc_vbmc
from pyvbmc.entropy.entmc_vbmc import _entmc_vbmc
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


def _draws(K, Ns, D, seed):
    """The antithetic standard normal draws of a call seeded with `seed`,
    as a `(K, Ns, D)` array."""
    Ns = int(np.ceil(Ns / 2)) * 2
    half = np.random.default_rng(seed).standard_normal((K, Ns // 2, D))
    return np.concatenate([half, -half], axis=1)


def _estimate(epsilon, samples, density, components=None):
    """The Monte Carlo estimate of the entropy at the draws `epsilon`.

    The samples are built from ``samples = (mu, sigma, lambd)`` and
    evaluated under the mixture ``density = (mu, sigma, lambd, w)``, whose
    weights also weight the average; only the terms of the components in
    `components` (all by default) are summed. Complex arguments give the
    complex-step derivative.
    """
    K, Ns, D = epsilon.shape
    mu_s, sigma_s, lambd_s = samples
    mu_d, sigma_d, lambd_d, w_d = density
    norm = w_d / ((2 * np.pi) ** (D / 2) * np.prod(lambd_d) * sigma_d**D)
    scale = sigma_d[:, np.newaxis] * lambd_d  # (K, D)
    H = 0.0
    for j in range(K) if components is None else components:
        x = mu_s[:, j] + sigma_s[j] * lambd_s * epsilon[j]  # (Ns, D)
        d2 = np.sum(((x[:, np.newaxis, :] - mu_d.T) / scale) ** 2, axis=2)
        H = H - w_d[j] * np.sum(np.log(np.exp(-0.5 * d2) @ norm)) / Ns
    return H


def _estimate_and_path_derivative(vp, Ns, jacobian_flag, seed):
    """The entropy estimate of a call seeded with `seed` and the gradient
    it returns, computed independently: with the draws fixed, the location
    and scale parameters move the samples while the density they are
    evaluated under stays fixed (the reparameterization gradient), and the
    weights move the density and the average, not the samples. The
    derivatives are taken by complex step, with respect to ``log sigma``,
    ``log lambda`` and the softmax parameters where `jacobian_flag` is set.
    """
    D, K = vp.mu.shape
    mu = np.asarray(vp.mu, dtype=float)
    sigma = np.asarray(vp.sigma, dtype=float).ravel()
    lambd = np.asarray(vp.lambd, dtype=float).ravel()
    w = np.asarray(vp.w, dtype=float).ravel()
    epsilon = _draws(K, Ns, D, seed)
    density = (mu, sigma, lambd, w)
    h = 1e-30

    def derivative(f, t):
        return np.imag(f(t + 1j * h)) / h

    def unpack(t):
        return np.exp(t) if jacobian_flag else t

    def pack(value):
        return np.log(value) if jacobian_flag else value

    dmu = np.zeros((D, K))
    dsigma = np.zeros(K)
    dlambd = np.zeros(D)
    dw = np.zeros(K)
    for j in range(K):
        for d in range(D):

            def f(t, j=j, d=d):
                moved = mu.astype(complex)
                moved[d, j] = t
                return _estimate(epsilon, (moved, sigma, lambd), density, [j])

            dmu[d, j] = derivative(f, mu[d, j])

        def f(t, j=j):
            moved = sigma.astype(complex)
            moved[j] = unpack(t)
            return _estimate(epsilon, (mu, moved, lambd), density, [j])

        dsigma[j] = derivative(f, pack(sigma[j]))
    for d in range(D):

        def f(t, d=d):
            moved = lambd.astype(complex)
            moved[d] = unpack(t)
            return _estimate(epsilon, (mu, sigma, moved), density)

        dlambd[d] = derivative(f, pack(lambd[d]))
    eta = np.log(w)
    for j in range(K):

        def f(t, j=j):
            if jacobian_flag:
                moved_eta = eta.astype(complex)
                moved_eta[j] = t
                moved = np.exp(moved_eta) / np.sum(np.exp(moved_eta))
            else:
                moved = w.astype(complex)
                moved[j] = t
            return _estimate(
                epsilon, (mu, sigma, lambd), (mu, sigma, lambd, moved)
            )

        dw[j] = derivative(f, eta[j] if jacobian_flag else w[j])
    H = _estimate(epsilon, (mu, sigma, lambd), density)
    return H, np.concatenate([dmu.ravel("F"), dsigma, dlambd, dw])


def _assert_is_the_path_derivative(vp, Ns, jacobian_flag, seed):
    H, dH = entmc_vbmc(
        vp,
        Ns,
        grad_flags=(True,) * 4,
        jacobian_flag=jacobian_flag,
        rng=seed,
    )
    H_ref, dH_ref = _estimate_and_path_derivative(vp, Ns, jacobian_flag, seed)
    assert np.isclose(H, H_ref, rtol=1e-13, atol=0.0)
    assert np.allclose(
        dH, dH_ref, rtol=1e-11, atol=1e-13 * np.max(np.abs(dH_ref))
    )


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

            # The gradient is the reparameterization gradient at the
            # call's draws. The derivative of the estimate at fixed draws,
            # which also moves the density, agrees with it only in
            # expectation (about 1% apart at this Ns).
            _assert_is_the_path_derivative(vp, Ns, False, seed=42)


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


def _mixture(mu, sigma, lambd, w):
    mu = np.asarray(mu, dtype=float)
    D, K = mu.shape
    vp = VariationalPosterior(D, K, rng=20260923)
    vp.mu = mu
    vp.sigma = np.asarray(sigma, dtype=float).reshape(1, K)
    vp.lambd = np.asarray(lambd, dtype=float).reshape(D, 1)
    w = np.asarray(w, dtype=float).reshape(1, K)
    vp.w = w / w.sum()
    vp.eta = np.log(vp.w)
    return vp


_rs = np.random.default_rng(11)
_PATH_CASES = {
    "K1_D1": ([[0.2]], [0.8], [1.3], [1.0]),
    "D1_K3": ([[0.0, 0.9, -1.4]], [1.0, 0.6, 0.8], [0.7], [0.2, 0.5, 0.3]),
    "D5_K6": (
        _rs.normal(size=(5, 6)),
        _rs.uniform(0.5, 1.5, 6),
        _rs.uniform(0.5, 1.5, 5),
        _rs.dirichlet(np.ones(6)),
    ),
    "weight_1e-10": (
        _rs.normal(size=(3, 3)) * 0.3,
        [1.0, 0.8, 1.2],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 1e-10],
    ),
    "near_coincident": (
        [[0.0, 1e-7, 0.5], [0.0, 0.0, 0.2]],
        [1.0, 1.0, 0.9],
        [1.0, 1.0],
        [0.3, 0.3, 0.4],
    ),
    "coincident": ([[0.2, 0.2], [0.1, 0.1]], [1, 1], [1, 1], [0.4, 0.6]),
    "scales_1e-3_to_1e3": (
        [[0.0, 0.001, 5.0], [0.0, 0.0, 1.0]],
        [1e-3, 1.0, 1e3],
        [1.0, 1.0],
        [0.3, 0.3, 0.4],
    ),
    "40_sd_apart": ([[0.0, 40.0], [0.0, 0.0]], [1, 1], [1, 1], [0.5, 0.5]),
}


@pytest.mark.parametrize("jacobian_flag", [False, True])
@pytest.mark.parametrize("case", sorted(_PATH_CASES))
def test_entmc_vbmc_gradient_is_the_path_derivative(case, jacobian_flag):
    """The estimate and its gradient are those of an independent
    implementation of the estimator at the same draws, to rounding: the
    reparameterization gradient with the density held fixed, and the full
    derivative for the weights. Checked with and without the
    reparameterization of the scales and the weights."""
    vp = _mixture(*_PATH_CASES[case])
    _assert_is_the_path_derivative(vp, 40, jacobian_flag, seed=404)


def test_entmc_vbmc_matlab():
    # If exact is True, random seeds and samples should be the same
    # with MATLAB version, i.e. entmc_vbmc.py need to be modified a
    # bit: epsilon[: Ns // 2, :] = randn2(D, Ns // 2).transpose()
    exact = False
    path = Path(__file__).parent.joinpath("entropy-test.npz")
    with np.load(path, allow_pickle=False) as fixture:
        D = fixture["D"].item()
        K = fixture["K"].item()
        Ns = fixture["Ns"].item()
        vp = VariationalPosterior(D, K)
        vp.w = fixture["vp_w"].astype(float)
        vp.mu = fixture["vp_mu"].astype(float)
        vp.sigma = fixture["vp_sigma"].astype(float)
        vp.lambd = fixture["vp_lambd"].astype(float)
        vp.eta = fixture["vp_eta"].astype(float)
        Hm = fixture["H"].item()
        dHm = fixture["dH"].squeeze()
        jacobian_flag = fixture["jacobian_flag"].item()

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


def _budget_case_vp(D, K, seed):
    """A mixture with a distinct location and scale in every direction."""
    rs = np.random.default_rng(seed)
    vp = VariationalPosterior(D, K)
    vp.mu = rs.normal(0, 1, (D, K))
    vp.sigma = np.exp(rs.normal(0, 0.3, (1, K)))
    vp.lambd = np.exp(rs.normal(0, 0.3, (D, 1)))
    vp.eta = rs.normal(0, 0.5, (1, K))
    vp.w = np.exp(vp.eta) / np.exp(vp.eta).sum()
    return vp


def _entmc_at_budget(vp, Ns, grad_flags, budget):
    """Entropy, gradient and generator state of one seeded call."""
    rng = np.random.default_rng(404)
    H, dH = _entmc_vbmc(vp, Ns, grad_flags, True, rng, budget=budget)
    return H, dH, rng.bit_generator.state


_ALL_GRADS = (True, True, True, True)
_NO_GRADS = (False, False, False, False)

# One component's distance tensor holds Ns * D * K doubles, so every budget
# below that figure splits a component's samples.  The extra budgets of each
# case are below the smallest candidate and leave partial blocks: 2600 with
# (4, 20, 38) gives sample blocks of 32 and 6, 800 with (3, 7, 12) component
# blocks of 3, 3 and 1, and 110 sample blocks of 5, 5 and 2. (8, 20, 738) is
# a gradient call of the size a run makes (738 samples per component at
# K = 20), in which the canonical blocks themselves split each component's
# samples, into 409 and 329.
_BUDGET_CASES = (
    (4, 20, 38, _ALL_GRADS, (2600, 1000, 300)),
    (15, 50, 56, _ALL_GRADS, (7000, 2000, 400)),
    (4, 20, 4096, _NO_GRADS, (5000, 1100)),
    (15, 26, 4096, _NO_GRADS, (9000, 2500)),
    (3, 7, 12, _ALL_GRADS, (800, 300, 110, 37)),
    (3, 1, 40, _ALL_GRADS, (60, 30, 11)),
    (4, 20, 38, (True, False, False, False), (2600, 1000, 300)),
    (4, 20, 38, (False, True, False, False), (2600, 1000, 300)),
    (4, 20, 38, (False, False, True, False), (2600, 1000, 300)),
    (4, 20, 38, (False, False, False, True), (2600, 1000, 300)),
    (8, 20, 738, _ALL_GRADS, (7777, 3000, 1000)),
)


@pytest.mark.parametrize("D, K, Ns, grad_flags, budgets", _BUDGET_CASES)
def test_entmc_vbmc_is_independent_of_the_chunk_budget(
    D, K, Ns, grad_flags, budgets
):
    """Every budget returns the bits the default budget returns."""
    vp = _budget_case_vp(D, K, seed=1000 * D + K)
    H_ref, dH_ref, state_ref = _entmc_at_budget(
        vp, Ns, grad_flags, DEFAULT_CHUNK_ELEMENTS
    )
    assert np.asarray(H_ref).dtype == np.float64
    assert dH_ref.dtype == np.float64
    for budget in tuple(CANDIDATE_BUDGETS) + budgets:
        H, dH, state = _entmc_at_budget(vp, Ns, grad_flags, budget)
        assert np.array_equal(H, H_ref), budget
        assert np.array_equal(dH, dH_ref), budget
        assert state == state_ref, budget


@pytest.mark.parametrize("jacobian_flag", [False, True])
@pytest.mark.parametrize("D, K, Ns", [(4, 5, 60), (8, 20, 738)])
def test_entmc_vbmc_blocks_do_not_depend_on_the_other_flags(
    D, K, Ns, jacobian_flag
):
    """The estimate and each gradient block are the bits of the call that
    requests every block, whichever blocks a call requests, none
    included."""
    vp = _budget_case_vp(D, K, seed=1000 * D + K)
    sizes = (D * K, K, D, K)
    H_all, dH_all = entmc_vbmc(vp, Ns, _ALL_GRADS, jacobian_flag, rng=404)
    blocks = np.split(dH_all, np.cumsum(sizes)[:-1])
    for grad_flags in itertools.product((False, True), repeat=4):
        H, dH = entmc_vbmc(vp, Ns, grad_flags, jacobian_flag, rng=404)
        requested = [b for b, flag in zip(blocks, grad_flags) if flag]
        assert np.array_equal(H, H_all), grad_flags
        assert np.array_equal(
            dH, np.concatenate(requested) if requested else np.empty(0)
        ), grad_flags


if __name__ == "__main__":
    test_entmc_vbmc_overlapping_mixture()
