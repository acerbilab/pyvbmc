"""First question: the gradient of entlb_vbmc, by complex-step differentiation.

entlb_vbmc reads vp.mu, vp.sigma, vp.lambd, vp.w and vp.eta as they are
and uses only arithmetic, sqrt, exp and log, so assigning complex arrays
gives the derivative of the code's own value to machine precision
(step 1e-30). Both parameterizations are checked: (mu, sigma, lambda, w
free) with jacobian_flag=False, and (mu, log sigma, log lambda, eta) with
jacobian_flag=True, in the configurations that the tests leave out.
The value is also compared with an independent log-sum-exp formula.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import logsumexp
from wave7_O2_common import banner

from pyvbmc.entropy import entlb_vbmc
from pyvbmc.variational_posterior import VariationalPosterior

banner()
warnings.simplefilter("error", np.exceptions.ComplexWarning)
H_STEP = 1e-30


def set_vp(vp, mu, sig, lam, w, eta):
    vp.mu, vp.sigma, vp.lambd, vp.w, vp.eta = mu, sig, lam, w, eta


def unpack(theta, D, K, jac):
    mu = theta[: D * K].reshape((D, K), order="F")
    a = theta[D * K : D * K + K]
    b = theta[D * K + K : D * K + K + D]
    c = theta[D * K + K + D :]
    if jac:
        sig, lam = np.exp(a), np.exp(b)
        eta = c - np.max(c.real)
        w = np.exp(eta) / np.sum(np.exp(eta))
    else:
        sig, lam, w = a, b, c
        eta = np.log(w.astype(complex)) if np.iscomplexobj(w) else np.log(w)
    return (
        mu,
        sig.reshape(1, K),
        lam.reshape(D, 1),
        w.reshape(1, K),
        eta.reshape(1, K),
    )


def cs_grad(vp, theta, D, K, jac):
    g = np.zeros(theta.size)
    for i in range(theta.size):
        # the step must be small against the coordinate itself (a weight
        # of 1e-100 has a Taylor radius of 1e-100 in log w)
        h = H_STEP * min(1.0, abs(theta[i])) if theta[i] != 0 else H_STEP
        t = theta.astype(complex)
        t[i] += 1j * h
        set_vp(vp, *unpack(t, D, K, jac))
        H, _ = entlb_vbmc(vp, grad_flags=(False,) * 4, jacobian_flag=jac)
        g[i] = np.imag(H) / h
    return g


def ref_H(mu, sig, lam, w):
    """Jensen bound -sum_j w_j log sum_k w_k N(mu_j; mu_k, (s_j^2+s_k^2)L)."""
    D, K = mu.shape
    s, lam, w = sig.ravel(), lam.ravel(), w.ravel()
    if K == 1:
        return (
            0.5 * D * (1 + np.log(2 * np.pi))
            + D * np.log(s[0])
            + np.sum(np.log(lam))
        )
    L = np.empty((K, K))
    for j in range(K):
        for k in range(K):
            s2 = s[j] ** 2 + s[k] ** 2
            d2 = np.sum((mu[:, j] - mu[:, k]) ** 2 / (s2 * lam**2))
            L[j, k] = (
                np.log(w[k])
                - 0.5 * D * np.log(2 * np.pi * s2)
                - np.sum(np.log(lam))
                - 0.5 * d2
            )
    return -np.sum(w * logsumexp(L, axis=1))


def config(name, mu, sig, lam, w):
    mu = np.asarray(mu, float)
    D, K = mu.shape
    sig = np.asarray(sig, float)
    lam = np.asarray(lam, float)
    w = np.asarray(w, float)
    w = w / w.sum()
    vp = VariationalPosterior(D, K, rng=0)
    out = []
    for jac in (False, True):
        if jac:
            eta = np.log(w)
            theta = np.concatenate(
                [mu.ravel("F"), np.log(sig), np.log(lam), eta - eta.max()]
            )
        else:
            theta = np.concatenate([mu.ravel("F"), sig, lam, w])
        set_vp(vp, *unpack(theta, D, K, jac))
        H, dH = entlb_vbmc(vp, jacobian_flag=jac)
        g = cs_grad(vp, theta, D, K, jac)
        scale = np.maximum(np.abs(g), np.max(np.abs(g)) * 1e-8)
        err = np.max(np.abs(dH - g) / scale)
        out.append(
            f"jac={jac!s:5s} max rel err {err:.1e} "
            f"(abs {np.max(np.abs(dH - g)):.1e}, max|g| "
            f"{np.max(np.abs(g)):.1e})"
        )
        if jac:
            Href = ref_H(mu, sig, lam, w)
            out.append(f"|H-ref|={abs(H - Href):.1e}")
    print(f"{name:40s} " + "; ".join(out), flush=True)


rs = np.random.default_rng(7)
config("K=1, D=1", [[0.3]], [0.7], [1.3], [1.0])
config("K=1, D=4", rs.normal(size=(4, 1)), [0.9], rs.uniform(0.5, 2, 4), [1.0])
config("D=1, K=2", [[0.0, 0.9]], [1.0, 0.6], [1.2], [0.3, 0.7])
config("D=1, K=3", [[0.0, 0.9, -1.4]], [1.0, 0.6, 0.8], [0.7], [0.2, 0.5, 0.3])
config(
    "K=4, D=3 random",
    rs.normal(size=(3, 4)),
    rs.uniform(0.5, 1.5, 4),
    rs.uniform(0.5, 1.5, 3),
    rs.dirichlet(np.ones(4)),
)
config(
    "K=6, D=5 random",
    rs.normal(size=(5, 6)),
    rs.uniform(0.5, 1.5, 6),
    rs.uniform(0.5, 1.5, 5),
    rs.dirichlet(np.ones(6)),
)
config(
    "weight 1e-10, overlapping",
    rs.normal(size=(3, 3)) * 0.3,
    [1, 0.8, 1.2],
    [1, 1, 1],
    [0.5, 0.5, 1e-10],
)
config(
    "weight 1e-100, separated (40 sd)",
    [[0, 40, 80], [0, 0, 0]],
    [1, 1, 1],
    [1, 1],
    [0.5, 0.5, 1e-100],
)
config(
    "near-coincident (1e-7)",
    [[0, 1e-7, 0.5], [0, 0, 0.2]],
    [1, 1, 0.9],
    [1, 1],
    [0.3, 0.3, 0.4],
)
config(
    "exactly coincident", [[0.2, 0.2], [0.1, 0.1]], [1, 1], [1, 1], [0.4, 0.6]
)
config(
    "scales 1e-3..1e3",
    [[0, 0.001, 5], [0, 0, 1]],
    [1e-3, 1, 1e3],
    [1, 1],
    [0.3, 0.3, 0.4],
)
config("scales 1e-4 vs 1", [[0, 0.01], [0, 0]], [1e-4, 1], [1, 1], [0.5, 0.5])
config("components 40 sd apart", [[0, 40], [0, 0]], [1, 1], [1, 1], [0.5, 0.5])
config(
    "D=20, small scales",
    rs.normal(size=(20, 4)) * 1e-3,
    rs.uniform(0.5, 1.5, 4),
    np.full(20, 1e-3),
    rs.dirichlet(np.ones(4)),
)
config(
    "K=50, D=10 random",
    rs.normal(size=(10, 50)),
    rs.uniform(0.3, 1.0, 50),
    rs.uniform(0.5, 1.5, 10),
    rs.dirichlet(np.ones(50)),
)
