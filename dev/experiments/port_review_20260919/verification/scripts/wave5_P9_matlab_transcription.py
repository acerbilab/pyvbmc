"""My own line-by-line transcription of the MATLAB prior functions of
`../vbmc` at 396d649 (`shared/m*{logpdf,pdf,rnd}.m`), written from the
sources directly and used by P9_11, P9_12 and P9_13.

MATLAB is not installed here: these are Python functions that execute the
same arithmetic and the same branch tests, with MATLAB's column-major
linear indexing reproduced where it matters (`msmoothboxrnd`). Nothing
here executes MATLAB.

Conventions: `x` is (N, D); `a, u, v, b, sigma` are scalars, (1, D) or
(N, D), expanded as the .m files expand them.
"""

import numpy as np


def _prep(x, params):
    """MATLAB's scalar expansion + repmat to (N, D)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    N, D = x.shape
    out = []
    for p in params:
        p = np.asarray(p, dtype=float)
        if p.ndim == 0:  # isscalar(p) -> p*ones(1,D)
            p = np.full((1, D), float(p))
        else:
            p = np.atleast_2d(p)
        if p.shape[1] != D:
            raise ValueError("SizeError")
        if p.shape[0] == 1:  # repmat(p, [N,1])
            p = np.repeat(p, N, axis=0)
        out.append(p)
    return x, N, D, out


def munifboxlogpdf(x, a, b):
    """shared/munifboxlogpdf.m"""
    x, N, D, (a, b) = _prep(x, (a, b))
    if np.any(a >= b):
        raise ValueError("munifboxlogpdf:OrderError")
    lnf = np.sum(np.log(b - a), axis=1)  # (N,)
    y = -lnf * np.ones(N)
    idx = np.any(x < a, axis=1) | np.any(x > b, axis=1)
    y[idx] = -np.inf
    return y


def munifboxpdf(x, a, b):
    return np.exp(munifboxlogpdf(x, a, b))


def mtrapezlogpdf(x, a, u, v, b):
    """shared/mtrapezlogpdf.m (no order check in MATLAB)."""
    x, N, D, (a, u, v, b) = _prep(x, (a, u, v, b))
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.full((N, D), -np.inf)
        lnf = np.log(0.5) + np.log(b - a + v - u) + np.log(u - a)
        for ii in range(D):
            idx = (x[:, ii] >= a[:, ii]) & (x[:, ii] < u[:, ii])
            y[idx, ii] = np.log(x[idx, ii] - a[idx, ii]) - lnf[idx, ii]

            idx = (x[:, ii] >= u[:, ii]) & (x[:, ii] < v[:, ii])
            y[idx, ii] = np.log(u[idx, ii] - a[idx, ii]) - lnf[idx, ii]

            idx = (x[:, ii] >= v[:, ii]) & (x[:, ii] < b[:, ii])
            y[idx, ii] = (
                np.log(b[idx, ii] - x[idx, ii])
                - np.log(b[idx, ii] - v[idx, ii])
                + np.log(u[idx, ii] - a[idx, ii])
                - lnf[idx, ii]
            )
        return np.sum(y, axis=1)


def mtrapezpdf(x, a, u, v, b):
    return np.exp(mtrapezlogpdf(x, a, u, v, b))


def msplinetrapezlogpdf(x, a, b, c, d):
    """shared/msplinetrapezlogpdf.m.  (a,b,c,d) == PyVBMC's (a,u,v,b)."""
    x, N, D, (a, b, c, d) = _prep(x, (a, b, c, d))
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.full((N, D), -np.inf)
        lnf = np.log(0.5 * (c - b + d - a))
        for ii in range(D):
            idx = (x[:, ii] >= a[:, ii]) & (x[:, ii] < b[:, ii])
            z = (x[idx, ii] - a[idx, ii]) / (b[idx, ii] - a[idx, ii])
            y[idx, ii] = np.log(-2 * z**3 + 3 * z**2) - lnf[idx, ii]

            idx = (x[:, ii] >= b[:, ii]) & (x[:, ii] < c[:, ii])
            y[idx, ii] = -lnf[idx, ii]

            idx = (x[:, ii] >= c[:, ii]) & (x[:, ii] < d[:, ii])
            z = 1 - (x[idx, ii] - c[idx, ii]) / (d[idx, ii] - c[idx, ii])
            y[idx, ii] = np.log(-2 * z**3 + 3 * z**2) - lnf[idx, ii]
        return np.sum(y, axis=1)


def msplinetrapezpdf(x, a, b, c, d):
    return np.exp(msplinetrapezlogpdf(x, a, b, c, d))


def msmoothboxlogpdf(x, a, b, sigma):
    """shared/msmoothboxlogpdf.m"""
    x, N, D, (a, b, sigma) = _prep(x, (a, b, sigma))
    if np.any(sigma <= 0):
        raise ValueError("msmoothboxpdf:NonPositiveSigma")
    if np.any(a >= b):
        raise ValueError("msmoothboxpdf:OrderError")
    y = np.full((N, D), -np.inf)
    lnf = np.log(1 / np.sqrt(2 * np.pi) / sigma) - np.log1p(
        1 / np.sqrt(2 * np.pi) / sigma * (b - a)
    )
    for ii in range(D):
        idx = x[:, ii] < a[:, ii]
        y[idx, ii] = (
            lnf[idx, ii]
            - 0.5 * ((x[idx, ii] - a[idx, ii]) / sigma[idx, ii]) ** 2
        )
        idx = (x[:, ii] >= a[:, ii]) & (x[:, ii] <= b[:, ii])
        y[idx, ii] = lnf[idx, ii]
        idx = x[:, ii] > b[:, ii]
        y[idx, ii] = (
            lnf[idx, ii]
            - 0.5 * ((x[idx, ii] - b[idx, ii]) / sigma[idx, ii]) ** 2
        )
    return np.sum(y, axis=1)


def msmoothboxpdf(x, a, b, sigma):
    return np.exp(msmoothboxlogpdf(x, a, b, sigma))


# ---------------------------------------------------------------- samplers


def _rows(p, n):
    p = np.atleast_2d(np.asarray(p, dtype=float))
    if p.shape[0] == 1:
        p = np.repeat(p, n, axis=0)
    return p


def munifboxrnd(a, b, n, rng):
    """shared/munifboxrnd.m (the 3-argument call)."""
    a = np.atleast_2d(np.asarray(a, float))
    b = np.atleast_2d(np.asarray(b, float))
    D = a.shape[1]
    if np.any(a >= b):
        raise ValueError("munifboxpdf:OrderError")
    return a + rng.random((n, D)) * (b - a)


def mtrapezrnd(a, u, v, b, n, rng):
    """shared/mtrapezrnd.m (the 5-argument call)."""
    D = np.atleast_2d(np.asarray(a, float)).shape[1]
    a, u, v, b = (_rows(p, n) for p in (a, u, v, b))
    r = np.zeros((n, D))
    for d in range(D):
        x0 = 0.5 * (u[:, d] + v[:, d])
        y_max = mtrapezpdf(
            x0.reshape(-1, 1),
            a[:, [d]],
            u[:, [d]],
            v[:, [d]],
            b[:, [d]],
        )
        idx = np.ones(n, dtype=bool)
        r1 = np.zeros(n)
        n1 = int(idx.sum())
        while n1 > 0:
            r1[idx] = a[idx, d] + rng.random(n1) * (b[idx, d] - a[idx, d])
            z1 = rng.random(n1) * y_max[idx]
            y1 = mtrapezpdf(
                r1[idx].reshape(-1, 1),
                a[idx][:, [d]],
                u[idx][:, [d]],
                v[idx][:, [d]],
                b[idx][:, [d]],
            )
            idx_new = np.zeros(n, dtype=bool)
            idx_new[idx] = z1 > y1
            idx = idx_new
            n1 = int(idx.sum())
        r[:, d] = r1
    return r


def msplinetrapezrnd(a, u, v, b, n, rng):
    """shared/msplinetrapezrnd.m (the 5-argument call)."""
    D = np.atleast_2d(np.asarray(a, float)).shape[1]
    a, u, v, b = (_rows(p, n) for p in (a, u, v, b))
    r = np.zeros((n, D))
    for d in range(D):
        x0 = 0.5 * (u[:, d] + v[:, d])
        y_max = msplinetrapezpdf(
            x0.reshape(-1, 1),
            a[:, [d]],
            u[:, [d]],
            v[:, [d]],
            b[:, [d]],
        )
        idx = np.ones(n, dtype=bool)
        r1 = np.zeros(n)
        n1 = int(idx.sum())
        while n1 > 0:
            r1[idx] = a[idx, d] + rng.random(n1) * (b[idx, d] - a[idx, d])
            z1 = rng.random(n1) * y_max[idx]
            y1 = msplinetrapezpdf(
                r1[idx].reshape(-1, 1),
                a[idx][:, [d]],
                u[idx][:, [d]],
                v[idx][:, [d]],
                b[idx][:, [d]],
            )
            idx_new = np.zeros(n, dtype=bool)
            idx_new[idx] = z1 > y1
            idx = idx_new
            n1 = int(idx.sum())
        r[:, d] = r1
    return r


def msmoothboxrnd(a, b, sigma, n, rng, repaired=False):
    """shared/msmoothboxrnd.m (the 4-argument call).

    With ``repaired=False`` the tail branches use ``a(idx)`` / ``b(idx)``
    exactly as lines 59 and 66 do: a logical (n,)-vector index into the
    (n, D) matrix, which MATLAB resolves as column-major linear indexing
    and so reads column 1.  ``repaired=True`` uses ``a(idx, d)``.
    """
    D = np.atleast_2d(np.asarray(a, float)).shape[1]
    a, b, sigma = (_rows(p, n) for p in (a, b, sigma))
    if np.any(sigma <= 0):
        raise ValueError("msmoothboxrnd:NonPositiveSigma")
    r = np.zeros((n, D))
    nf = 1 + 1 / np.sqrt(2 * np.pi) / sigma * (b - a)
    for d in range(D):
        u = nf[:, d] * rng.random(n)

        idx = u < 0.5
        if idx.any():
            z1 = np.abs(rng.standard_normal(int(idx.sum())) * sigma[idx, d])
            # a(idx): column-major linear index -> column 0
            src = (
                a[idx, d]
                if repaired
                else a.ravel(order="F")[np.flatnonzero(idx)]
            )
            r[idx, d] = src - z1

        idx = (u >= 0.5) & (u < 1.0)
        if idx.any():
            z1 = np.abs(rng.standard_normal(int(idx.sum())) * sigma[idx, d])
            src = (
                b[idx, d]
                if repaired
                else b.ravel(order="F")[np.flatnonzero(idx)]
            )
            r[idx, d] = src + z1

        idx = u >= 1.0
        if idx.any():
            r[idx, d] = a[idx, d] + (b[idx, d] - a[idx, d]) * rng.random(
                int(idx.sum())
            )
    return r
