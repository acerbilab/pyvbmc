"""O4, first question: the low-noise representation of the posterior (min
sn2 < 1e-6: L = chol(K + diag(sn2)), sl = 1). (A) Just below the threshold on
a well-conditioned set, where double-precision differences are reliable, and
the same set just above it (Cholesky representation) for continuity. (B) An
ill-conditioned low-noise set against an 80-digit reference (Decimal, a hand
Cholesky, central differences with step 1e-25), beside double-precision
differences."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from wave7_O4_common import attach, banner, fd_grad, hp_grad, hp_nlZ, make_gp

banner()
np.set_printoptions(precision=6, linewidth=130)


def nlZ_and_grad(gp, h):
    lz, dlz = gp.log_likelihood(h, compute_grad=True)
    return -lz, -dlz


def dense_nlZ(gp, h):
    """Dense evaluation (slogdet), SE ARD with scalar constant noise."""
    X, y = gp.X, gp.y
    N, D = X.shape
    Z = X / np.exp(h[:D])
    K = np.exp(2 * h[D]) * np.exp(
        -0.5 * np.sum((Z[:, None] - Z[None]) ** 2, 2)
    )
    C = K + np.exp(2 * h[D + 1]) * np.eye(N)
    m = gp.mean.compute(h[D + 2 :], X).reshape(-1, 1)
    r = y - m
    return (
        0.5 * (r.T @ np.linalg.solve(C, r))[0, 0]
        + 0.5 * np.linalg.slogdet(C)[1]
        + 0.5 * N * np.log(2 * np.pi)
    )


print("== (A) well-conditioned set, just below / just above sn2 = 1e-6 ==")
D, N = 1, 8
X = np.linspace(-1, 1, N).reshape(-1, 1)
y = np.sin(3 * X)
for mean in ("const", "negquad"):
    gp = make_gp(D, mean, (1, 0, 0))
    attach(gp, X, y)
    mh = [0.1] if mean == "const" else [0.5, 0.1, np.log(0.8)]
    for sn2 in (0.9e-6, 1.1e-6, 0.99e-6, 1.01e-6):
        h = np.array([np.log(0.08), np.log(1.0), 0.5 * np.log(sn2)] + mh)
        post = gp._GP__core_computation(h, 0, False)
        v, g = nlZ_and_grad(gp, h)
        fd = fd_grad(lambda hh: -gp.log_likelihood(hh), h, 1e-4)
        print(
            f"  mean={mean:7s} sn2={sn2:.2e} L_chol={post.L_chol!s:5s} nlZ={v:.12f} "
            f"(dense {dense_nlZ(gp, h):.12f}) max|g-fd|/max|fd|={np.max(np.abs(g - fd)) / np.max(np.abs(fd)):.1e}"
        )
        print(f"      analytic {g}\n      FD       {fd}")

print(
    "\n== (B) ill-conditioned low-noise sets against the 80-digit reference =="
)
cases = []
Xb = np.linspace(-1, 1, 10).reshape(-1, 1)
yb = np.sin(2 * Xb) + 0.3 * Xb**2
cases.append(
    (
        "D=1 N=10 dense, sn2=1e-10, const",
        Xb,
        yb,
        "const",
        [np.log(0.5), np.log(1.0), 0.5 * np.log(1e-10), 0.2],
    )
)
cases.append(
    (
        "D=1 N=10 dense, sn2=1e-10, negquad",
        Xb,
        yb,
        "negquad",
        [np.log(0.5), np.log(1.0), 0.5 * np.log(1e-10), 0.5, 0.1, np.log(0.9)],
    )
)
Xd = np.linspace(-1, 1, 14).reshape(-1, 1)
yd = np.sin(2 * Xd) + 0.3 * Xd**2
cases.append(
    (
        "D=1 N=14 dense, ell=0.8, sn2=1e-12, negquad",
        Xd,
        yd,
        "negquad",
        [np.log(0.8), np.log(1.0), 0.5 * np.log(1e-12), 0.5, 0.1, np.log(0.9)],
    )
)
rng = np.random.default_rng(8)
Xc = rng.uniform(-1, 1, (9, 2))
Xc[1] = Xc[0]
yc = np.sin(Xc[:, :1] * 2) + Xc[:, 1:] ** 2
yc[1] = yc[0]
cases.append(
    (
        "D=2 N=9 one repeated input, sn2=1e-9, zero",
        Xc,
        yc,
        "zero",
        [np.log(0.6), np.log(0.7), np.log(1.0), 0.5 * np.log(1e-9)],
    )
)
for label, Xs, ys, mean, h in cases:
    t0 = time.time()
    Dd = Xs.shape[1]
    gp = make_gp(Dd, mean, (1, 0, 0))
    attach(gp, Xs, ys)
    h = np.array(h)
    post = gp._GP__core_computation(h, 0, False)
    v, g = nlZ_and_grad(gp, h)
    ref_v = float(hp_nlZ(Xs, ys[:, 0], h, Dd, (1, 0, 0), mean))
    ref_g = hp_grad(Xs, ys[:, 0], h, Dd, (1, 0, 0), mean)
    fd = fd_grad(lambda hh: -gp.log_likelihood(hh), h, 1e-5)
    C = gp.covariance.compute(h[: Dd + 1], gp.X) + np.exp(
        2 * h[Dd + 1]
    ) * np.eye(Xs.shape[0])
    print(
        f"  {label}: L_chol={post.L_chol}, cond(C)={np.linalg.cond(C):.1e}, "
        f"nlZ rel err vs ref={abs(v - ref_v) / abs(ref_v):.1e}  ({time.time() - t0:.1f}s)"
    )
    print(
        f"      reference {ref_g}\n      analytic  {g}\n      double FD {fd}"
    )
    per = np.abs(g - ref_g) / np.abs(ref_g)
    print(
        f"      analytic: max err / max|ref| = {np.max(np.abs(g - ref_g)) / np.max(np.abs(ref_g)):.1e}, "
        f"per-component rel err = {np.array2string(per, precision=1)}"
    )
    print(
        f"      double FD: max err / max|ref| = {np.max(np.abs(fd - ref_g)) / np.max(np.abs(ref_g)):.1e}",
        flush=True,
    )
