"""Python transcription of the MATLAB gplite routines that slice G2 compares.

Transcribed by reading C:\\Users\\luigi\\Documents\\GitHub\\vbmc\\gplite at
revision 396d649; MATLAB itself was not available, so every routine below is a
line-by-line reading of the .m file it names, with MATLAB's column-wise
reductions (max/min/std/median of a matrix act along the first non-singleton
dimension) written out explicitly.

Covered:
  gplite_core.m            -> core_posterior   (no intmeanfun, no outwarp)
  gplite_pred.m            -> pred             (no intmeanfun, no outwarp)
  gplite_quad.m            -> quad             (meanfun 0, 1, 4)
  gplite_noisefun.m        -> noisefun, noisefun_info
  gplite_covfun.m          -> covfun_se, covfun_matern, covfun_info
  gplite_meanfun.m         -> meanfun_negquad, meanfun_info (cases 0, 1, 4)
  private/quantile1.m      -> quantile1
  private/sq_dist.m        -> sq_dist
  gplite_rnd.m robustchol  -> robustchol

MATLAB's `noisefun` is the 3-vector [constant, user_provided, output_dep].
"""

import numpy as np
import scipy as sp

EPS = np.spacing(1.0)


# --------------------------------------------------------------- helpers


def sq_dist(a, b=None):
    """private/sq_dist.m: squared distances between columns of a and b.

    MATLAB's `a` is D-by-n. The file centers both sets on a size-weighted
    mean of the two before forming the distances.
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    d, n = a.shape
    if b is None:
        mu = np.mean(a, axis=1, keepdims=True)
        a = a - mu
        sa = np.sum(a * a, axis=0)
        C = sa[:, None] + sa[None, :] - 2.0 * (a.T @ a)
    else:
        b = np.atleast_2d(np.asarray(b, dtype=float))
        m = b.shape[1]
        mu = (m / (n + m)) * np.mean(b, axis=1, keepdims=True) + (
            n / (n + m)
        ) * np.mean(a, axis=1, keepdims=True)
        a = a - mu
        b = b - mu
        C = (
            np.sum(a * a, axis=0)[:, None]
            + np.sum(b * b, axis=0)[None, :]
            - 2.0 * (a.T @ b)
        )
    return np.maximum(C, 0.0)


def quantile1(x, p):
    """private/quantile1.m. `x` is flattened (x = x(:)) before sorting."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    if p == 0.5:
        if n % 2:
            return x[(n + 1) // 2 - 1]
        return 0.5 * (x[n // 2 - 1] + x[n // 2])
    r = p * n
    k = int(np.floor(r + 0.5))
    kp1 = min(k + 1, n)
    r = r - k
    k = max(k, 1)
    y = (0.5 + r) * x[kp1 - 1] + (0.5 - r) * x[k - 1]
    if r == -0.5:
        y = x[k - 1]
    if x[k - 1] == x[kp1 - 1]:
        y = x[k - 1]
    return y


def _squeeze1(v):
    """MATLAB returns a scalar where a column-wise reduction has one column."""
    v = np.asarray(v, dtype=float)
    return float(v.ravel()[0]) if v.size == 1 else v


def _mstd(A):
    """MATLAB std(A) for a 2-D array: per column, N-1 normalization; for a
    single row it acts along the row (and std of one scalar is 0)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] == 1:
        row = A[0]
        if row.size == 1:
            return 0.0
        return float(np.std(row, ddof=1))
    return _squeeze1(np.std(A, axis=0, ddof=1))


def _mmax(A):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] == 1:
        return float(np.max(A[0]))
    return _squeeze1(np.max(A, axis=0))


def _mmin(A):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] == 1:
        return float(np.min(A[0]))
    return _squeeze1(np.min(A, axis=0))


def _mmedian(A):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] == 1:
        return float(np.median(A[0]))
    return _squeeze1(np.median(A, axis=0))


# --------------------------------------------------------------- noise


def noisefun(hyp, X, noise_parameters, y=None, s2=None):
    """gplite_noisefun.m compute branch (lines 175-210)."""
    hyp = np.asarray(hyp, dtype=float).ravel()
    p = list(noise_parameters) + [0] * (3 - len(noise_parameters))
    if s2 is None:
        s2 = 0.0
    i = 0
    if p[0] == 0:
        sn2 = EPS
    else:
        sn2 = np.exp(2 * hyp[i])
        i += 1
    if p[1] == 1:
        sn2 = sn2 + s2
    elif p[1] == 2:
        sn2 = sn2 + np.exp(hyp[i]) * s2
        i += 1
    if p[2] == 1:
        if y is not None:
            ythresh = hyp[i]
            w2 = np.exp(2 * hyp[i + 1])
            zz = np.maximum(0.0, ythresh - y)
            sn2 = sn2 + w2 * zz**2
        i += 2
    return sn2


def noisefun_info(X, noise_parameters, y, s2=None):
    """gplite_noisefun.m info branch (lines 84-150)."""
    p = list(noise_parameters) + [0] * (3 - len(noise_parameters))
    n_noise = (1 if p[0] == 1 else 0) + (1 if p[1] == 2 else 0)
    n_noise += 2 if p[2] == 1 else 0
    D = np.atleast_2d(X).shape[1]
    tol = 1e-6
    out = {
        k: np.full((n_noise,), v)
        for k, v in (
            ("LB", -np.inf),
            ("UB", np.inf),
            ("PLB", -np.inf),
            ("PUB", np.inf),
            ("x0", np.nan),
        )
    }
    y = np.asarray(y, dtype=float)
    if y.size <= 1:
        y = np.array([0.0, 1.0])
    height = np.max(y) - np.min(y)
    i = 0
    if p[0] == 1:
        out["LB"][i] = np.log(tol)
        out["UB"][i] = np.log(height)
        out["PLB"][i] = 0.5 * np.log(tol)
        out["PUB"][i] = np.log(_mstd(y.reshape(-1, 1)))
        out["x0"][i] = np.log(1e-3)
        i += 1
    if p[1] == 2:
        out["LB"][i] = np.log(1e-3)
        out["UB"][i] = np.log(1e3)
        out["PLB"][i] = np.log(0.5)
        out["PUB"][i] = np.log(2)
        out["x0"][i] = np.log(1)
        i += 1
    if p[2] == 1:
        miny, maxy = np.min(y), np.max(y)
        out["LB"][i] = miny
        out["UB"][i] = maxy
        out["PLB"][i] = miny
        out["PUB"][i] = max(maxy - 5 * D, miny)
        out["x0"][i] = max(maxy - 10 * D, miny)
        i += 1
        out["LB"][i] = np.log(1e-3)
        out["UB"][i] = np.log(0.1)
        out["PLB"][i] = np.log(0.01)
        out["PUB"][i] = np.log(0.1)
        out["x0"][i] = np.log(0.1)
        i += 1
    nan = np.isnan(out["x0"])
    out["x0"][nan] = 0.5 * (out["PLB"][nan] + out["PUB"][nan])
    return out


# --------------------------------------------------------------- covariance


def covfun_se(hyp, X, X_star=None, diag=False):
    """gplite_covfun.m case 1 (SE-ARD), lines 176-191."""
    hyp = np.asarray(hyp, dtype=float).ravel()
    X = np.atleast_2d(X)
    D = X.shape[1]
    ell = np.exp(hyp[0:D])
    sf2 = np.exp(2 * hyp[D])
    if diag:
        return sf2 * np.ones((X.shape[0], 1))
    if X_star is None:
        K = sq_dist((X / ell).T)
    else:
        K = sq_dist((X / ell).T, (np.atleast_2d(X_star) / ell).T)
    return sf2 * np.exp(-K / 2)


def covfun_matern(hyp, X, degree, compute_grad=False):
    """gplite_covfun.m case 3 (Matern ARD), lines 193-222, self-covariance."""
    hyp = np.asarray(hyp, dtype=float).ravel()
    X = np.atleast_2d(X)
    N, D = X.shape
    ell = np.exp(hyp[0:D])
    sf2 = np.exp(2 * hyp[D])
    if degree == 1:
        f = lambda t: np.ones_like(t)
        df = lambda t: 1.0 / t
    elif degree == 3:
        f = lambda t: 1 + t
        df = lambda t: np.ones_like(t)
    else:
        f = lambda t: 1 + t * (1 + t / 3)
        df = lambda t: (1 + t) / 3
    tmp = np.sqrt(sq_dist((X * np.sqrt(degree) / ell).T))
    K = sf2 * f(tmp) * np.exp(-tmp)
    if not compute_grad:
        return K
    dK = np.zeros((N, N, D + 1))
    with np.errstate(all="ignore"):
        for i in range(D):
            Ki = sq_dist((np.sqrt(degree) / ell[i] * X[:, i]).reshape(1, -1))
            dK[:, :, i] = sf2 * (df(tmp) * np.exp(-tmp)) * Ki
    dK[:, :, D] = 2 * K
    return K, dK


def covfun_info(X, y, isoflag=False, n_cov=None):
    """gplite_covfun.m info branch (lines 94-148)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    D = X.shape[1]
    n_cov = (2 if isoflag else D + 1) if n_cov is None else n_cov
    tol = 1e-6
    out = {
        "LB": np.full((n_cov,), -np.inf),
        "UB": np.full((n_cov,), np.inf),
        "PLB": np.full((n_cov,), -np.inf),
        "PUB": np.full((n_cov,), np.inf),
        "x0": np.full((n_cov,), np.nan),
    }
    width = _mmax(X) - _mmin(X)
    y = np.asarray(y, dtype=float)
    if y.size <= 1:
        y = np.array([0.0, 1.0])
    height = np.max(y) - np.min(y)
    with np.errstate(all="ignore"):
        if isoflag:
            out["LB"][0] = np.mean(np.log(width)) + np.log(tol)
            out["UB"][0] = np.mean(np.log(width * 10))
            out["PLB"][0] = np.mean(np.log(width)) + 0.5 * np.log(tol)
            out["PUB"][0] = np.mean(np.log(width))
            out["x0"][0] = np.mean(np.log(_mstd(X)))
            n_ell = 1
        else:
            out["LB"][0:D] = np.log(width) + np.log(tol)
            out["UB"][0:D] = np.log(width * 10)
            out["PLB"][0:D] = np.log(width) + 0.5 * np.log(tol)
            out["PUB"][0:D] = np.log(width)
            out["x0"][0:D] = np.log(_mstd(X))
            n_ell = D
        out["LB"][n_ell] = np.log(height) + np.log(tol)
        out["UB"][n_ell] = np.log(height * 10)
        out["PLB"][n_ell] = np.log(height) + 0.5 * np.log(tol)
        out["PUB"][n_ell] = np.log(height)
        out["x0"][n_ell] = np.log(_mstd(y.reshape(-1, 1)))
    nan = np.isnan(out["x0"])
    out["x0"][nan] = 0.5 * (out["PLB"][nan] + out["PUB"][nan])
    return out


# --------------------------------------------------------------- mean


def meanfun_negquad(hyp, X):
    """gplite_meanfun.m case 4 (negative quadratic)."""
    hyp = np.asarray(hyp, dtype=float).ravel()
    X = np.atleast_2d(X)
    D = X.shape[1]
    m0 = hyp[0]
    xm = hyp[1 : 1 + D]
    omega = np.exp(hyp[1 + D : 1 + 2 * D])
    z2 = ((X - xm) / omega) ** 2
    return m0 - 0.5 * np.sum(z2, axis=1)


def meanfun_info(X, y, meanfun):
    """gplite_meanfun.m info branch (lines 136-345) for meanfun 0, 1, 4."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    D = X.shape[1]
    tol = 1e-6
    big = np.exp(3)
    n_mean = {0: 0, 1: 1, 4: 1 + 2 * D}[meanfun]
    out = {
        "LB": np.full((n_mean,), -np.inf),
        "UB": np.full((n_mean,), np.inf),
        "PLB": np.full((n_mean,), -np.inf),
        "PUB": np.full((n_mean,), np.inf),
        "x0": np.full((n_mean,), np.nan),
    }
    if n_mean == 0:
        return out
    w = _mmax(X) - _mmin(X)
    y = np.asarray(y, dtype=float)
    if y.size <= 1:
        y = np.array([0.0, 1.0])
    h = np.max(y) - np.min(y)
    if meanfun == 1:
        out["LB"][0] = np.min(y) - 0.5 * h
        out["UB"][0] = np.max(y) + 0.5 * h
        out["PLB"][0] = quantile1(y, 0.1)
        out["PUB"][0] = quantile1(y, 0.9)
        out["x0"][0] = np.median(y)
    else:  # meanfun == 4
        out["LB"][0] = np.min(y)
        out["UB"][0] = np.max(y) + h
        out["PLB"][0] = np.median(y)
        out["PUB"][0] = np.max(y)
        out["x0"][0] = quantile1(y, 0.9)
        out["LB"][1 : D + 1] = _mmin(X) - 0.5 * w
        out["UB"][1 : D + 1] = _mmax(X) + 0.5 * w
        out["PLB"][1 : D + 1] = _mmin(X)
        out["PUB"][1 : D + 1] = _mmax(X)
        out["x0"][1 : D + 1] = _mmedian(X)
        with np.errstate(all="ignore"):
            out["LB"][D + 1 : 2 * D + 1] = np.log(w) + np.log(tol)
            out["UB"][D + 1 : 2 * D + 1] = np.log(w) + np.log(big)
            out["PLB"][D + 1 : 2 * D + 1] = np.log(w) + 0.5 * np.log(tol)
            out["PUB"][D + 1 : 2 * D + 1] = np.log(w)
            out["x0"][D + 1 : 2 * D + 1] = np.log(_mstd(X))
    nan = np.isnan(out["x0"])
    out["x0"][nan] = 0.5 * (out["PLB"][nan] + out["PUB"][nan])
    return out


# --------------------------------------------------------------- posterior


def core_posterior(hyp, X, y, noise_parameters, meanfun, s2=None):
    """gplite_core.m, the posterior branch (lines 32-102, 277-291)."""
    hyp = np.asarray(hyp, dtype=float).ravel()
    X = np.atleast_2d(X)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    N, D = X.shape
    n_cov = D + 1
    p = list(noise_parameters) + [0] * (3 - len(noise_parameters))
    n_noise = (1 if p[0] == 1 else 0) + (1 if p[1] == 2 else 0)
    n_noise += 2 if p[2] == 1 else 0
    n_mean = {0: 0, 1: 1, 4: 1 + 2 * D}[meanfun]
    sn2 = noisefun(hyp[n_cov : n_cov + n_noise], X, p, y, s2)
    sn2_mult = 1
    if meanfun == 0:
        m = np.zeros((N, 1))
    elif meanfun == 1:
        m = np.full((N, 1), hyp[n_cov + n_noise])
    else:
        m = meanfun_negquad(
            hyp[n_cov + n_noise : n_cov + n_noise + n_mean], X
        ).reshape(-1, 1)
    K = covfun_se(hyp[0:n_cov], X)
    L_chol = np.min(sn2) >= 1e-6
    if L_chol:
        if np.isscalar(sn2) or np.size(sn2) == 1:
            sn2div = float(np.asarray(sn2).ravel()[0])
            sn2_mat = np.eye(N)
        else:
            sn2div = float(np.min(sn2))
            sn2_mat = np.diag(np.asarray(sn2).ravel() / sn2div)
        for _ in range(10):
            try:
                L = sp.linalg.cholesky(
                    K / (sn2div * sn2_mult) + sn2_mat, check_finite=False
                )
                break
            except sp.linalg.LinAlgError:
                sn2_mult *= 10
        sl = sn2div * sn2_mult
        pL = L
    else:
        if np.isscalar(sn2) or np.size(sn2) == 1:
            sn2_mat = float(np.asarray(sn2).ravel()[0]) * np.eye(N)
        else:
            sn2_mat = np.diag(np.asarray(sn2).ravel())
        for _ in range(10):
            try:
                L = sp.linalg.cholesky(
                    K + sn2_mult * sn2_mat, check_finite=False
                )
                break
            except sp.linalg.LinAlgError:
                sn2_mult *= 10
        sl = 1
        pL = -sp.linalg.solve_triangular(
            L,
            sp.linalg.solve_triangular(L, np.eye(N), trans=1),
            trans=0,
        )
    alpha = (
        sp.linalg.solve_triangular(
            L, sp.linalg.solve_triangular(L, y - m, trans=1), trans=0
        )
        / sl
    )
    return {
        "hyp": hyp,
        "alpha": alpha,
        "sW": np.ones((N, 1)) / np.sqrt(np.min(sn2) * sn2_mult),
        "L": pL,
        "sn2_mult": sn2_mult,
        "Lchol": bool(L_chol),
        "sl": sl,
    }


# --------------------------------------------------------------- predict


def pred(
    posts,
    X,
    y,
    noise_parameters,
    meanfun,
    X_star,
    y_star=None,
    s2_star=None,
    ssflag=False,
):
    """gplite_pred.m without intmeanfun and without output warping.

    Returns (ymu, ys2, fmu, fs2, lp) with MATLAB's shapes: (Nstar, Ns)
    columns when ssflag, single columns otherwise (lp is never averaged).
    """
    X = np.atleast_2d(X)
    N, D = X.shape
    Ns = len(posts)
    X_star = np.atleast_2d(X_star)
    N_star = X_star.shape[0]
    n_cov = D + 1
    p = list(noise_parameters) + [0] * (3 - len(noise_parameters))
    n_noise = (1 if p[0] == 1 else 0) + (1 if p[1] == 2 else 0)
    n_noise += 2 if p[2] == 1 else 0
    n_mean = {0: 0, 1: 1, 4: 1 + 2 * D}[meanfun]

    fmu = np.zeros((N_star, Ns))
    fs2 = np.zeros((N_star, Ns))
    ys2 = np.zeros((N_star, Ns))
    lp = None if y_star is None else np.zeros((N_star, Ns))

    for s in range(Ns):
        post = posts[s]
        hyp = np.asarray(post["hyp"], dtype=float).ravel()
        alpha = post["alpha"]
        L = post["L"]
        Lchol = post["Lchol"]
        sW = post["sW"]
        sn2_mult = post["sn2_mult"]
        sn2_star = noisefun(
            hyp[n_cov : n_cov + n_noise], X_star, p, y_star, s2_star
        )
        if meanfun == 0:
            mstar = np.zeros((N_star, 1))
        elif meanfun == 1:
            mstar = np.full((N_star, 1), hyp[n_cov + n_noise])
        else:
            mstar = meanfun_negquad(
                hyp[n_cov + n_noise : n_cov + n_noise + n_mean], X_star
            ).reshape(-1, 1)
        ell = np.exp(hyp[0:D])
        sf2 = np.exp(2 * hyp[D])
        Ks = sq_dist((X / ell).T, (X_star / ell).T)
        Ks = sf2 * np.exp(-Ks / 2)
        kss = sf2 * np.ones((N_star, 1))
        if N > 0:
            fmu[:, s : s + 1] = mstar + Ks.T @ alpha
        else:
            fmu[:, s : s + 1] = mstar
        if N > 0:
            if Lchol:
                V = sp.linalg.solve_triangular(
                    L, np.tile(sW, (1, N_star)) * Ks, trans=1
                )
                fs2[:, s : s + 1] = kss - np.sum(V * V, axis=0).reshape(-1, 1)
            else:
                fs2[:, s : s + 1] = kss + np.sum(
                    Ks * (L @ Ks), axis=0
                ).reshape(-1, 1)
        else:
            fs2[:, s : s + 1] = kss
        fs2[:, s] = np.maximum(fs2[:, s], 0)
        sn2_col = np.asarray(sn2_star, dtype=float)
        if sn2_col.size > 1:
            sn2_col = sn2_col.reshape(-1, 1)
        else:
            sn2_col = float(sn2_col.ravel()[0])
        ys2[:, s : s + 1] = fs2[:, s : s + 1] + sn2_col * sn2_mult
        if y_star is not None:
            ys = np.asarray(y_star, dtype=float).reshape(-1, 1)
            lp[:, s : s + 1] = -0.5 * (ys - fmu[:, s : s + 1]) ** 2 / ys2[
                :, s : s + 1
            ] - 0.5 * np.log(2 * np.pi * ys2[:, s : s + 1])
    ymu = fmu.copy()
    if Ns > 1 and not ssflag:
        fbar = np.sum(fmu, axis=1, keepdims=True) / Ns
        ybar = np.sum(ymu, axis=1, keepdims=True) / Ns
        vf = np.sum((fmu - fbar) ** 2, axis=1, keepdims=True) / (Ns - 1)
        fs2_o = np.sum(fs2, axis=1, keepdims=True) / Ns + vf
        vy = np.sum((ymu - ybar) ** 2, axis=1, keepdims=True) / (Ns - 1)
        ys2_o = np.sum(ys2, axis=1, keepdims=True) / Ns + vy
        return ybar, ys2_o, fbar, fs2_o, lp
    return ymu, ys2, fmu, fs2, lp


# --------------------------------------------------------------- quadrature


def quad(
    posts,
    X,
    y,
    noise_parameters,
    meanfun,
    mu,
    sigma,
    compute_var=False,
    ssflag=False,
):
    """gplite_quad.m for meanfun 0, 1 and 4 (SE-ARD kernel only)."""
    X = np.atleast_2d(X)
    N, D = X.shape
    Ns = len(posts)
    n_cov = D + 1
    p = list(noise_parameters) + [0] * (3 - len(noise_parameters))
    n_noise = (1 if p[0] == 1 else 0) + (1 if p[1] == 2 else 0)
    n_noise += 2 if p[2] == 1 else 0
    mu = np.atleast_2d(np.asarray(mu, dtype=float))
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    N_star = mu.shape[0]
    if sigma.shape[0] == 1:
        sigma = np.tile(sigma, (N_star, 1))
    F = np.zeros((N_star, Ns))
    varF = np.zeros((N_star, Ns)) if compute_var else None
    for s in range(Ns):
        post = posts[s]
        hyp = np.asarray(post["hyp"], dtype=float).ravel()
        ell = np.exp(hyp[0:D]).reshape(1, -1)
        ln_sf2 = 2 * hyp[D]
        sum_lnell = np.sum(hyp[0:D])
        m0 = hyp[n_cov + n_noise] if meanfun > 0 else 0.0
        if meanfun == 4:
            xm = hyp[n_cov + n_noise + 1 : n_cov + n_noise + 1 + D].reshape(
                1, -1
            )
            omega = np.exp(
                hyp[n_cov + n_noise + D + 1 : n_cov + n_noise + 2 * D + 1]
            ).reshape(1, -1)
        alpha = post["alpha"]
        L = post["L"]
        Lchol = post["Lchol"]
        # gplite_quad.m:66-67 -- the CONSTANT noise hyperparameter
        sn2 = np.exp(2 * hyp[n_cov])
        sn2_eff = sn2 * post["sn2_mult"]
        tau = np.sqrt(sigma**2 + ell**2)
        lnnf = ln_sf2 + sum_lnell - np.sum(np.log(tau), axis=1)
        sumdelta2 = np.zeros((N_star, N))
        for i in range(D):
            sumdelta2 += (
                (mu[:, i : i + 1] - X[:, i].reshape(1, -1)) / tau[:, i : i + 1]
            ) ** 2
        z = np.exp(lnnf.reshape(-1, 1) - 0.5 * sumdelta2)
        F[:, s : s + 1] = z @ alpha + m0
        if meanfun == 4:
            nu_k = -0.5 * np.sum(
                1
                / omega**2
                * (mu**2 + sigma**2 - 2 * mu * xm + xm**2),
                axis=1,
            )
            F[:, s] += nu_k
        if compute_var:
            tau_kk = np.sqrt(2 * sigma**2 + ell**2)
            nf_kk = np.exp(ln_sf2 + sum_lnell - np.sum(np.log(tau_kk), 1))
            if Lchol:
                invKzk = (
                    sp.linalg.solve_triangular(
                        L,
                        sp.linalg.solve_triangular(L, z.T, trans=1),
                        trans=0,
                    )
                    / sn2_eff
                )
            else:
                invKzk = -L @ z.T
            J_kk = nf_kk - np.sum(z * invKzk.T, axis=1)
            varF[:, s] = np.maximum(EPS, J_kk)
    if Ns > 1 and not ssflag:
        Fbar = np.sum(F, axis=1, keepdims=True) / Ns
        if compute_var:
            varFss = np.sum((F - Fbar) ** 2, axis=1) / (Ns - 1)
            varF = (np.sum(varF, axis=1) / Ns + varFss).reshape(-1, 1)
        F = Fbar
    if compute_var:
        return F, varF
    return F


# --------------------------------------------------------------- robustchol


def robustchol(Sigma):
    """gplite_rnd.m local robustchol (lines 88-112).

    MATLAB's `eig` of a symmetric matrix returns real eigenvectors; the
    transcription uses `eigh`, whose eigenvalues are sorted ascending. The
    sign convention and the tolerance test are transcribed literally, with
    MATLAB's linear index picking one entry per COLUMN.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    n, m = Sigma.shape
    try:
        T = sp.linalg.cholesky(Sigma, check_finite=False)
        return T, 0
    except sp.linalg.LinAlgError:
        pass
    D, U = np.linalg.eigh((Sigma + Sigma.T) / 2)
    maxidx = np.argmax(np.abs(U), axis=0)  # one row index per column
    negidx = U[maxidx, np.arange(m)] < 0  # a length-m mask
    U[:, negidx] *= -1
    tol = np.spacing(np.max(D)) * D.size
    t = np.abs(D) > tol
    Dk = D[t]
    p = int(np.sum(Dk < 0))
    if p == 0:
        T = np.diag(np.sqrt(Dk)) @ U[:, t].T
    else:
        T = np.zeros((0, m))
    return T, p
