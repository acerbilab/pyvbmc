"""First question: the gradient of vp.pdf, with and without log_flag.

Central differences with one Richardson step on the code's own value, row
by row, and the independent log-sum-exp formula, in configurations the
tests leave out: K = 1, D = 1, many rows at once, a tiny weight, disparate
scales, coincident components, and points in the tails.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wave7_O2_common import banner, build_vp, ref_logq_and_grad

banner()
warnings.simplefilter("ignore")


def fd_rows(f, X, rel=1e-4):
    G = np.zeros_like(X)
    for n in range(X.shape[0]):
        for d in range(X.shape[1]):
            h = rel * max(1.0, abs(X[n, d]))

            def c(step):
                xp = X[n].copy()
                xp[d] += step
                xm = X[n].copy()
                xm[d] -= step
                return (f(xp[None, :]) - f(xm[None, :])) / (2 * step)

            G[n, d] = (4 * c(h / 2) - c(h)) / 3
    return G


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    floor = 1e-10 * np.max(np.abs(b))
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), floor))


def check(name, mu, sig, lam, w, X):
    mu = np.asarray(mu, float)
    vp = build_vp(mu, sig, lam, w)
    X = np.asarray(X, float)
    y, dy = vp.pdf(X, orig_flag=False, grad_flag=True)
    ly, dly = vp.pdf(X, orig_flag=False, log_flag=True, grad_flag=True)
    fd_y = fd_rows(lambda x: vp.pdf(x, orig_flag=False)[0, 0], X)
    fd_ly = fd_rows(
        lambda x: vp.pdf(x, orig_flag=False, log_flag=True)[0, 0], X
    )
    lq, g = ref_logq_and_grad(X, mu, sig, lam, vp.w)
    print(
        f"{name:34s} rows={X.shape[0]:3d}: pdf grad vs FD {rel(dy, fd_y):.1e}"
        f", vs ref {rel(dy, np.exp(lq)[:, None] * g):.1e}; log grad vs FD "
        f"{rel(dly, fd_ly):.1e}, vs ref {rel(dly, g):.1e}; "
        f"log q vs ref {np.max(np.abs(ly.ravel() - lq)):.1e}",
        flush=True,
    )
    # chunked evaluation: rows are independent
    same = all(
        np.array_equal(
            vp._pdf(X, False, True, True, np.inf, chunk_elements=c)[1], dly
        )
        for c in (1, 3, 17, 1000)
    )
    print(
        f"{'':34s} chunk sizes 1, 3, 17, 1000 bit-identical: {same}",
        flush=True,
    )


rs = np.random.default_rng(21)
check("K=1, D=1", [[0.3]], [0.8], [1.3], [1.0], rs.normal(size=(7, 1)))
check(
    "D=1, K=3",
    [[0.0, 0.9, -1.4]],
    [1.0, 0.6, 0.8],
    [0.7],
    [0.2, 0.5, 0.3],
    rs.normal(size=(9, 1)),
)
check(
    "K=1, D=4",
    rs.normal(size=(4, 1)),
    [0.9],
    rs.uniform(0.5, 2, 4),
    [1.0],
    rs.normal(size=(6, 4)),
)
check(
    "K=5, D=4, 50 rows",
    rs.normal(size=(4, 5)),
    rs.uniform(0.5, 1.5, 5),
    rs.uniform(0.5, 1.5, 4),
    rs.dirichlet(np.ones(5)),
    rs.normal(size=(50, 4)) * 1.5,
)
check(
    "tiny weight 1e-12",
    [[0, 2], [0, 1]],
    [1, 0.5],
    [1, 1],
    [1, 1e-12],
    rs.normal(size=(8, 2)) + [2, 1],
)
check(
    "scales 1e-3..1e3",
    [[0, 0, 1], [0, 0, 1]],
    [1e-3, 1, 1e3],
    [1, 1],
    [0.3, 0.3, 0.4],
    np.vstack([rs.normal(size=(4, 2)) * 1e-3, rs.normal(size=(4, 2)) * 50]),
)
check(
    "coincident components",
    [[0.2, 0.2], [0.1, 0.1]],
    [1, 0.5],
    [1, 1],
    [0.4, 0.6],
    rs.normal(size=(6, 2)),
)
check(
    "tails, 20-30 sd",
    [[0, 1], [0, 0]],
    [1, 0.7],
    [1, 1],
    [0.5, 0.5],
    np.array([[20.0, 3.0], [-25.0, 1.0], [0.0, 30.0], [18.0, -18.0]]),
)
