"""Helpers of the O2 verification: independent reference formulas."""
import gpyreg
import numpy as np
import scipy
from scipy.special import logsumexp

import pyvbmc
from pyvbmc.variational_posterior import VariationalPosterior


def banner():
    print("pyvbmc:", pyvbmc.__file__, flush=True)
    print("gpyreg:", gpyreg.__file__, flush=True)
    print("numpy", np.__version__, "scipy", scipy.__version__, flush=True)


def build_vp(mu, sigma, lambd, w, transformer=None):
    mu = np.asarray(mu, float)
    D, K = mu.shape
    kw = {"rng": 0}
    if transformer is not None:
        kw["parameter_transformer"] = transformer
    vp = VariationalPosterior(D, K, **kw)
    vp.mu = mu.copy()
    vp.sigma = np.asarray(sigma, float).reshape(1, K).copy()
    vp.lambd = np.asarray(lambd, float).reshape(D, 1).copy()
    w = np.asarray(w, float).reshape(1, K)
    vp.w = w / w.sum()
    with np.errstate(divide="ignore"):
        vp.eta = np.log(vp.w)
    return vp


def ref_logq_and_grad(X, mu, sigma, lambd, w):
    """log q(x) and grad_x log q(x) by log-sum-exp and responsibilities."""
    X = np.atleast_2d(np.asarray(X, float))
    mu = np.asarray(mu, float)
    D, K = mu.shape
    s = np.ravel(sigma)
    lam = np.ravel(lambd)
    w = np.ravel(w)
    L = np.empty((X.shape[0], K))
    G = np.empty((X.shape[0], K, D))
    for k in range(K):
        diff = X - mu[:, k]
        z2 = np.sum((diff / (s[k] * lam)) ** 2, axis=1)
        L[:, k] = (
            np.log(w[k])
            - 0.5 * D * np.log(2 * np.pi)
            - D * np.log(s[k])
            - np.sum(np.log(lam))
            - 0.5 * z2
        )
        G[:, k, :] = -diff / (s[k] ** 2 * lam**2)
    lq = logsumexp(L, axis=1)
    r = np.exp(L - lq[:, None])
    g = np.einsum("nk,nkd->nd", r, G)
    return lq, g
