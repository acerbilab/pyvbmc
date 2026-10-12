"""First question: entmc_vbmc's gradient against the exact path derivative.

With the draws of the call reproduced (standard_normal((K, Ns/2, D)) from
the same seed, antithetic), define
    F(theta_s; theta_d) = -sum_j w_j(theta_d) / Ns sum_n
                           log q(mu_j^s + sigma_j^s lambda^s eps_jn; theta_d).
The code's H must equal F(theta; theta); its mu, sigma, lambda gradients
must equal dF/dtheta_s (samples moved, density frozen); its weight gradient
must equal dF/dw through theta_d (the samples do not depend on w). With the
Jacobian, the same in (log sigma, log lambda, eta). All derivatives of F are
taken by complex step on this independent implementation. Configurations
include those the tests leave out, and production-sized gradient calls in
which the canonical blocks split one component's samples, at several
budgets.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wave7_O2_common import banner

from pyvbmc.calibration.profile import DEFAULT_CHUNK_ELEMENTS
from pyvbmc.entropy.entmc_vbmc import _entmc_vbmc
from pyvbmc.variational_posterior import VariationalPosterior

banner()
print("DEFAULT_CHUNK_ELEMENTS", DEFAULT_CHUNK_ELEMENTS, flush=True)
H_STEP = 1e-30
SEED = 404


def draws(K, Ns, D):
    Ns = int(np.ceil(Ns / 2)) * 2
    e = np.random.default_rng(SEED).standard_normal((K, Ns // 2, D))
    return np.concatenate([e, -e], axis=1)  # (K, Ns, D)


def F_j(j, mu_sj, sig_sj, lam_s, mu_d, sig_d, lam_d, w_d, eps):
    """Component j's term of the estimator (complex-safe)."""
    K, Ns, D = eps.shape
    X = mu_sj[None, :] + sig_sj * lam_s[None, :] * eps[j]  # (Ns, D)
    diff = X[:, None, :] - mu_d.T[None, :, :]  # (Ns, K, D)
    sl = sig_d[None, :, None] * lam_d[None, None, :]
    d2 = np.sum((diff / sl) ** 2, axis=2)  # (Ns, K)
    nf = 1 / (2 * np.pi) ** (D / 2) / np.prod(lam_d) / sig_d**D
    q = np.sum(w_d * nf * np.exp(-0.5 * d2), axis=1)  # (Ns,)
    return -w_d[j] * np.sum(np.log(q)) / Ns


def F(mu_s, sig_s, lam_s, mu_d, sig_d, lam_d, w_d, eps):
    return sum(
        F_j(j, mu_s[:, j], sig_s[j], lam_s, mu_d, sig_d, lam_d, w_d, eps)
        for j in range(eps.shape[0])
    )


def cs1(fun, x0):
    h = H_STEP * min(1.0, abs(x0)) if x0 != 0 else H_STEP
    return np.imag(fun(x0 + 1j * h)) / h


def reference(mu, sig, lam, w, eps, jac, idx_k, idx_d):
    """H and the gradient entries at components idx_k and coordinates
    idx_d, in the layout [mu (D,K) column-major, sigma, lambda, w]."""
    D, K = mu.shape
    H = F(mu, sig, lam, mu, sig, lam, w, eps)
    g = {}
    for j in idx_k:
        for d in range(D):

            def f(t, j=j, d=d):
                m = mu[:, j].astype(complex)
                m[d] = t
                return F_j(j, m, sig[j], lam, mu, sig, lam, w, eps)

            g[j * D + d] = cs1(f, mu[d, j])
        if jac:
            g[D * K + j] = cs1(
                lambda t, j=j: F_j(
                    j, mu[:, j], np.exp(t), lam, mu, sig, lam, w, eps
                ),
                np.log(sig[j]),
            )
        else:
            g[D * K + j] = cs1(
                lambda t, j=j: F_j(j, mu[:, j], t, lam, mu, sig, lam, w, eps),
                sig[j],
            )
    for d in idx_d:

        def f(t, d=d):
            l = lam.astype(complex)
            l[d] = np.exp(t) if jac else t
            return F(mu, sig, l, mu, sig, lam, w, eps)

        g[D * K + K + d] = cs1(f, np.log(lam[d]) if jac else lam[d])
    eta = np.log(w)
    for j in idx_k:

        def f(t, j=j):
            if jac:
                e = eta.astype(complex)
                e[j] = t
                ww = np.exp(e) / np.sum(np.exp(e))
            else:
                ww = w.astype(complex)
                ww[j] = t
            return F(mu, sig, lam, mu, sig, lam, ww, eps)

        g[D * K + K + D + j] = cs1(f, eta[j] if jac else w[j])
    return H, g


def check(name, mu, sig, lam, w, Ns, budgets=(None,), n_sub=None):
    mu = np.asarray(mu, float)
    D, K = mu.shape
    sig = np.asarray(sig, float).ravel()
    lam = np.asarray(lam, float).ravel()
    w = np.asarray(w, float).ravel()
    w = w / w.sum()
    vp = VariationalPosterior(D, K, rng=0)
    vp.mu, vp.sigma, vp.lambd = mu, sig.reshape(1, K), lam.reshape(D, 1)
    vp.w, vp.eta = w.reshape(1, K), np.log(w).reshape(1, K)
    eps = draws(K, Ns, D)
    Ns_even = eps.shape[1]
    splits = Ns_even * D * K > DEFAULT_CHUNK_ELEMENTS
    out = []
    sub = np.random.default_rng(5)
    idx_k = range(K) if n_sub is None else sub.choice(K, n_sub, replace=False)
    idx_d = (
        range(D) if n_sub is None else sub.choice(D, min(D, 3), replace=False)
    )
    for jac in (False, True):
        Hr, gd = reference(mu, sig, lam, w, eps, jac, idx_k, idx_d)
        keys = np.array(sorted(gd))
        gr = np.array([gd[k] for k in keys])
        for b in budgets:
            budget = DEFAULT_CHUNK_ELEMENTS if b is None else b
            H, dH = _entmc_vbmc(
                vp,
                Ns,
                (True,) * 4,
                jac,
                np.random.default_rng(SEED),
                budget=budget,
            )
            dHs = dH[keys]
            e = np.max(
                np.abs(dHs - gr)
                / np.maximum(np.abs(gr), 1e-8 * np.max(np.abs(gr)))
            )
            out.append(
                f"jac={int(jac)} b={budget}: dH {e:.1e} (abs "
                f"{np.max(np.abs(dHs - gr)):.1e}, max|g| "
                f"{np.max(np.abs(gr)):.1e}), "
                f"H {abs(H - Hr) / abs(Hr):.1e}"
            )
    print(
        f"{name} (D={D}, K={K}, Ns={Ns_even}, canonical split: {splits}, "
        f"entries checked: {len(keys)} of {D * K + 2 * K + D})",
        flush=True,
    )
    for o in out:
        print("    " + o, flush=True)


rs = np.random.default_rng(11)
check("K=1, D=1", [[0.2]], [0.8], [1.3], [1.0], 40)
check(
    "D=1, K=3", [[0.0, 0.9, -1.4]], [1.0, 0.6, 0.8], [0.7], [0.2, 0.5, 0.3], 40
)
check(
    "K=6, D=5 random",
    rs.normal(size=(5, 6)),
    rs.uniform(0.5, 1.5, 6),
    rs.uniform(0.5, 1.5, 5),
    rs.dirichlet(np.ones(6)),
    40,
)
check(
    "weight 1e-10",
    rs.normal(size=(3, 3)) * 0.3,
    [1, 0.8, 1.2],
    [1, 1, 1],
    [0.5, 0.5, 1e-10],
    40,
)
check(
    "near-coincident (1e-7)",
    [[0, 1e-7, 0.5], [0, 0, 0.2]],
    [1, 1, 0.9],
    [1, 1],
    [0.3, 0.3, 0.4],
    40,
)
check(
    "exactly coincident",
    [[0.2, 0.2], [0.1, 0.1]],
    [1, 1],
    [1, 1],
    [0.4, 0.6],
    40,
)
check(
    "scales 1e-3..1e3",
    [[0, 0.001, 5], [0, 0, 1]],
    [1e-3, 1, 1e3],
    [1, 1],
    [0.3, 0.3, 0.4],
    40,
)
check(
    "components 40 sd apart", [[0, 40], [0, 0]], [1, 1], [1, 1], [0.5, 0.5], 40
)
# production-sized gradient calls: Ns = ns_ent = 100 K^(2/3)
K = 50
check(
    "K=50, D=10, Ns = ns_ent",
    rs.normal(size=(10, K)),
    rs.uniform(0.3, 1.0, K),
    rs.uniform(0.5, 1.5, 10),
    rs.dirichlet(np.ones(K)),
    int(np.ceil(100 * K ** (2 / 3))),
    budgets=(None, 20000, 5000, 1000, 2**18),
    n_sub=4,
)
K = 20
check(
    "K=20, D=8, Ns = ns_ent",
    rs.normal(size=(8, K)),
    rs.uniform(0.3, 1.0, K),
    rs.uniform(0.5, 1.5, 8),
    rs.dirichlet(np.ones(K)),
    int(np.ceil(100 * K ** (2 / 3))),
    budgets=(None, 7777, 300, 2**20),
    n_sub=4,
)
