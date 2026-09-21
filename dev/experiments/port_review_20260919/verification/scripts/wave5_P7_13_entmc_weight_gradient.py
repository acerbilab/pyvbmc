"""P7-13: the Chengkun Li weight-gradient fix of entmc, and the kept score term.

(i) Reproduces entmc_vbmc's weight gradient with a loop transcription of
    ent/entmc_vbmc.m at 1b72896 (the fixed line) and at its parent (the old
    line), on the same Monte Carlo draws, to show which one the Python has.
(ii) Checks the claim that the weight gradient keeps a term (the average of
     norm_k / q over the mixture) that the mu, sigma and lambda gradients
     drop, and that the dropped term has expectation zero while the kept
     one has expectation one.
"""

import numpy as np

import pyvbmc
from pyvbmc.entropy import entmc_vbmc
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc.__file__ =", pyvbmc.__file__)

D, K, Ns = 2, 3, 400
vp = VariationalPosterior(
    D, K, x0=np.zeros((1, D)), rng=np.random.default_rng(2)
)
vp.mu = np.array([[0.0, 1.5, -1.0], [0.0, -1.0, 2.0]])
vp.sigma = np.array([[0.7, 1.1, 0.5]])
vp.lambd = np.array([[1.3], [0.8]])
w = np.array([0.2, 0.5, 0.3])
vp.w = w.reshape(1, K)
vp.eta = np.log(vp.w).reshape(1, K)

seed = 4242


def draw_epsilon(seed, K, Ns, D):
    """Exactly the draw _entmc_vbmc makes."""
    rng = np.random.default_rng(seed)
    half = Ns // 2
    eps_half = rng.standard_normal((K, half, D))
    return np.concatenate([eps_half, -eps_half], axis=1)


def matlab_entmc_wgrad(vp, Ns, epsilon, fixed):
    """ent/entmc_vbmc.m, the w_grad block only, in a loop.

    fixed=True  -> the 1b72896 line (sum over every component k)
    fixed=False -> the line it replaced (the j-th component only)
    """
    D, K = int(vp.D), int(vp.K)
    mu = np.asarray(vp.mu, float)
    sigma = np.asarray(vp.sigma, float).ravel()
    lambd = np.asarray(vp.lambd, float).ravel()
    w = np.asarray(vp.w, float).ravel()
    nconst = 1 / (2 * np.pi) ** (D / 2) / np.prod(lambd)
    w_grad = np.zeros(K)
    H = 0.0
    for j in range(K):
        xi = mu[:, j][None, :] + epsilon[j] * (sigma[j] * lambd)[None, :]
        # norm_jl[n, l] = N(xi_n ; mu_l, sigma_l lambda)
        norm_jl = np.empty((Ns, K))
        for l in range(K):
            d2 = np.sum(
                ((xi - mu[:, l][None, :]) / (sigma[l] * lambd)) ** 2, 1
            )
            norm_jl[:, l] = nconst / sigma[l] ** D * np.exp(-0.5 * d2)
        q_j = norm_jl @ w
        H -= w[j] * np.sum(np.log(q_j)) / Ns
        w_grad[j] -= np.sum(np.log(q_j)) / Ns
        if fixed:
            w_grad -= w[j] * np.sum(norm_jl / q_j[:, None], axis=0) / Ns
        else:
            w_grad -= w[j] * np.sum(norm_jl[:, j] / q_j) / Ns
    return H, w_grad


eps = draw_epsilon(seed, K, Ns, D)
H_py, dH_py = entmc_vbmc(
    vp, Ns, tuple([True] * 4), False, rng=np.random.default_rng(seed)
)
w_py = dH_py[-K:]
H_fix, w_fix = matlab_entmc_wgrad(vp, Ns, eps, fixed=True)
H_old, w_old = matlab_entmc_wgrad(vp, Ns, eps, fixed=False)

print(
    "\n  H python     =",
    H_py,
    "  transcription =",
    H_fix,
    "  diff =",
    H_py - H_fix,
)
print("  w_grad python        =", w_py)
print(
    "  w_grad 1b72896 (new) =",
    w_fix,
    "  max|diff| =",
    np.max(np.abs(w_py - w_fix)),
)
print(
    "  w_grad pre-fix (old) =",
    w_old,
    "  max|diff| =",
    np.max(np.abs(w_py - w_old)),
)

print("\n== (ii) the kept and the dropped term ==")
# The kept term for w_k is an MC estimate of  int norm_k(x) dx = 1.
D_, K_ = D, K
mu = np.asarray(vp.mu, float)
sigma = np.asarray(vp.sigma, float).ravel()
lambd = np.asarray(vp.lambd, float).ravel()
wv = np.asarray(vp.w, float).ravel()
nconst = 1 / (2 * np.pi) ** (D_ / 2) / np.prod(lambd)
kept = np.zeros(K_)
dropped_mu = np.zeros((D_, K_))
for j in range(K_):
    xi = mu[:, j][None, :] + eps[j] * (sigma[j] * lambd)[None, :]
    norm_jl = np.empty((Ns, K_))
    for l in range(K_):
        d2 = np.sum(((xi - mu[:, l][None, :]) / (sigma[l] * lambd)) ** 2, 1)
        norm_jl[:, l] = nconst / sigma[l] ** D_ * np.exp(-0.5 * d2)
    q_j = norm_jl @ wv
    kept += wv[j] * np.sum(norm_jl / q_j[:, None], axis=0) / Ns
    # the term the mu gradient drops: E_q[ d log q / d mu_k ] at fixed x
    for k in range(K_):
        g = (wv[k] * norm_jl[:, k] / q_j)[:, None] * (
            (xi - mu[:, k][None, :]) / (sigma[k] * lambd) ** 2
        )
        dropped_mu[:, k] += wv[j] * g.sum(0) / Ns
print("  kept term for w (MC estimate of int norm_k = 1):", kept)
print("  dropped term for mu (MC estimate of d/dmu int q = 0):")
print("  ", dropped_mu)
print(
    "  reason: int q dx = sum_k w_k, so d/dw_k of it is 1 and d/dmu, "
    "d/dsigma, d/dlambda of it are 0"
)
