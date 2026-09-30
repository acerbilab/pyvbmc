"""Helpers shared by the O4 verification scripts (wave 7)."""

import sys
from decimal import Decimal, getcontext

import gpyreg as gpr
import numpy as np
import scipy

import pyvbmc


def banner():
    print("pyvbmc:", pyvbmc.__file__)
    print("gpyreg:", gpr.__file__)
    print(
        "numpy",
        np.__version__,
        "scipy",
        scipy.__version__,
        "python",
        sys.version.split()[0],
        flush=True,
    )


def make_gp(D, mean="negquad", noise=(1, 0, 0), cov="se"):
    covf = {
        "se": gpr.covariance_functions.SquaredExponential,
    }[cov]()
    meanf = {
        "zero": gpr.mean_functions.ZeroMean,
        "const": gpr.mean_functions.ConstantMean,
        "negquad": gpr.mean_functions.NegativeQuadratic,
    }[mean]()
    c, u, r = noise
    noisef = gpr.noise_functions.GaussianNoise(
        constant_add=c == 1,
        user_provided_add=u > 0,
        scale_user_provided=u == 2,
        rectified_linear_output_dependent_add=r == 1,
    )
    return gpr.GP(D=D, covariance=covf, mean=meanf, noise=noisef)


def attach(gp, X, y, s2=None):
    X, y, s2 = gp._convert_shapes(X, y, s2)
    gp.X, gp.y, gp.s2 = X, y, s2


def central(f, x, i, h):
    xp = x.copy()
    xm = x.copy()
    xp[i] += h
    xm[i] -= h
    return (f(xp) - f(xm)) / (2 * h)


def fd_grad(f, x, h=1e-5):
    """Central differences with one Richardson step (O(h^4))."""
    g = np.empty_like(x, dtype=float)
    for i in range(x.size):
        g[i] = (4 * central(f, x, i, h / 2) - central(f, x, i, h)) / 3
    return g


def max_rel(a, b):
    """Largest absolute difference relative to the largest component."""
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


# ---------------------------------------------------------------------------
# High-precision negative log marginal likelihood (SE ARD kernel, constant
# noise with an optional recorded s2, zero / constant / negquad mean), written
# for this verification: Decimal arithmetic and a hand Cholesky.
# ---------------------------------------------------------------------------


def _dec(x):
    # Exact binary value of the double, so that the reference sees the very
    # inputs the double-precision code sees.
    return x if isinstance(x, Decimal) else Decimal(float(x))


def hp_nlZ(X, y, h, D, noise, mean, s2=None, prec=80):
    getcontext().prec = prec
    N = X.shape[0]
    Xd = [[_dec(X[i, d]) for d in range(D)] for i in range(N)]
    yd = [_dec(y[i]) for i in range(N)]
    h = [_dec(v) for v in h]
    k = 0
    ell = [h[d].exp() for d in range(D)]
    sf2 = (2 * h[D]).exp()
    k = D + 1
    sn2 = [Decimal(0)] * N
    if noise[0] == 1:
        sn2 = [(2 * h[k]).exp()] * N
        k += 1
    if noise[1] == 1:
        sn2 = [sn2[i] + _dec(s2[i]) for i in range(N)]
    elif noise[1] == 2:
        mult = h[k].exp()
        sn2 = [sn2[i] + mult * _dec(s2[i]) for i in range(N)]
        k += 1
    mh = h[k:]
    if mean == "zero":
        m = [Decimal(0)] * N
    elif mean == "const":
        m = [mh[0]] * N
    else:
        m = []
        for i in range(N):
            s = Decimal(0)
            for d in range(D):
                s += ((Xd[i][d] - mh[1 + d]) / mh[1 + D + d].exp()) ** 2
            m.append(mh[0] - s / 2)
    C = [[None] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            r2 = Decimal(0)
            for d in range(D):
                r2 += ((Xd[i][d] - Xd[j][d]) / ell[d]) ** 2
            C[i][j] = sf2 * (-r2 / 2).exp()
        C[i][i] += sn2[i]
    # Cholesky, lower
    L = [[Decimal(0)] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1):
            s = C[i][j] - sum(L[i][p] * L[j][p] for p in range(j))
            if i == j:
                L[i][i] = s.sqrt()
            else:
                L[i][j] = s / L[j][j]
    r = [yd[i] - m[i] for i in range(N)]
    # forward solve L z = r
    z = []
    for i in range(N):
        z.append((r[i] - sum(L[i][p] * z[p] for p in range(i))) / L[i][i])
    quad = sum(zi * zi for zi in z)
    logdet = 2 * sum(L[i][i].ln() for i in range(N))
    two_pi = Decimal(2) * Decimal(
        "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"
    )
    return quad / 2 + logdet / 2 + N * two_pi.ln() / 2


def hp_grad(X, y, h, D, noise, mean, s2=None, step="1e-25", prec=80):
    """Central differences of hp_nlZ in Decimal (truncation ~ step^2)."""
    getcontext().prec = prec
    hs = Decimal(step)
    h = [_dec(v) for v in h]
    g = []
    for i in range(len(h)):
        hp_ = list(h)
        hm_ = list(h)
        hp_[i] += hs
        hm_[i] -= hs
        fp = hp_nlZ(X, y, hp_, D, noise, mean, s2, prec)
        fm = hp_nlZ(X, y, hm_, D, noise, mean, s2, prec)
        g.append(float((fp - fm) / (2 * hs)))
    return np.array(g)
