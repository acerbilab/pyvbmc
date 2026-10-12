"""P4-5: gpyreg's rank-one GP update with a supplied noise variance.

MATLAB's ``gplite_post.m:76-79`` turns a requested rank-one update into a
full recomputation whenever ``s2`` is given, so on a noisy target MATLAB
never takes the cheap path.  This script settles whether gpyreg's
rank-one update with ``s2_new`` reproduces a GP rebuilt from the enlarged
training set with the same hyperparameters, at uncertainty levels 0, 1
and 2, including a new point whose noise lies below every existing one,
and on a posterior whose Cholesky factorization had to be retried
(``sn2_mult > 1``).  It compares what the P4 slice actually reads off the
posterior: ``alpha``, ``sW``, the predictions, ``sn2_eff = 1/sW[0]**2``
and the ``C_tmp`` of ``active_importance_sampling.py:310-320``.
"""

import copy
import warnings

import numpy as np
from scipy.linalg import solve_triangular
from wave4_P4_common import banner

banner()

import gpyreg as gpr  # noqa: E402


def make_gp(X, y, s2, noise_params, hyp):
    gp = gpr.GP(
        D=X.shape[1],
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(*noise_params),
    )
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp, compute_posterior=True)
    return gp


def c_tmp(gp, Xa, s):
    cov_N = gp.covariance.hyperparameter_count(gp.D)
    hyp = gp.posteriors[s].hyp[0:cov_N]
    L = gp.posteriors[s].L
    L_chol = gp.posteriors[s].L_chol
    sn2_eff = 1 / gp.posteriors[s].sW[0] ** 2
    K = gp.covariance.compute(hyp, Xa, gp.X)
    if L_chol:
        return (
            solve_triangular(
                L,
                solve_triangular(L, K.T, trans=True, check_finite=False),
                check_finite=False,
            )
            / sn2_eff
        )
    return L @ K.T


def compare(tag, gp_r1, gp_full, Xt, Xa):
    print(f"  [{tag}]")
    for s in range(len(gp_r1.posteriors)):
        p1, p2 = gp_r1.posteriors[s], gp_full.posteriors[s]
        da = np.abs(p1.alpha - p2.alpha).max()
        dsw = np.abs(p1.sW - p2.sW).max()
        sl1 = 1 / p1.sW[0, 0] ** 2
        sl2 = 1 / p2.sW[0, 0] ** 2
        dc = np.abs(c_tmp(gp_r1, Xa, s) - c_tmp(gp_full, Xa, s)).max()
        print(
            f"    s={s}  |dalpha|={da:.3e}  |dsW|={dsw:.3e}  "
            f"sn2_eff r1={sl1:.6e} full={sl2:.6e}  |dC_tmp|={dc:.3e}  "
            f"sn2_mult r1={p1.sn2_mult} full={p2.sn2_mult}  "
            f"L_chol={p1.L_chol}/{p2.L_chol}"
        )
    m1, v1 = gp_r1.predict(Xt, separate_samples=True)
    m2, v2 = gp_full.predict(Xt, separate_samples=True)
    print(
        f"    |dmu|={np.abs(m1 - m2).max():.3e}  "
        f"|ds2|={np.abs(v1 - v2).max():.3e}  "
        f"|ds2 stored|={np.abs(np.asarray(gp_r1.s2) - np.asarray(gp_full.s2)).max() if gp_r1.s2 is not None else 'n/a'}"
    )


rng = np.random.default_rng(0)
D, N = 2, 14
X = rng.standard_normal((N, D))
y = (-0.5 * np.sum(X**2, 1) + 0.1 * rng.standard_normal(N)).reshape(-1, 1)
Xt = rng.standard_normal((6, D))
Xa = rng.standard_normal((9, D))
x_new = rng.standard_normal((1, D))
y_new = np.array([[-0.8]])

# hyp = [log ell (D), log sf, noise..., mean const, mean mu (D), mean ln omega (D)]
cov_hyp = [np.log(1.0), np.log(1.2), np.log(2.0)]
mean_hyp = [0.3] + [0.0] * D + [np.log(1.0)] * D

cases = {
    # level 0: constant noise only
    "level 0": ((1, 0, 0), None, None, cov_hyp + [np.log(0.1)] + mean_hyp),
    # level 1: constant + scaled user variance, s2 == 1 everywhere
    "level 1": (
        (1, 2, 0),
        np.ones((N, 1)),
        np.ones((1, 1)),
        cov_hyp + [np.log(1e-3), np.log(0.5)] + mean_hyp,
    ),
    # level 2: constant + user variance as is, heteroskedastic
    "level 2 (new noise inside the range)": (
        (1, 1, 0),
        (0.05 + 0.4 * rng.random((N, 1))),
        np.array([[0.2]]),
        cov_hyp + [np.log(1e-3)] + mean_hyp,
    ),
}

s2_het = 0.05 + 0.4 * rng.random((N, 1))
cases["level 2 (new noise below every existing one)"] = (
    (1, 1, 0),
    s2_het,
    np.array([[s2_het.min() * 0.01]]),
    cov_hyp + [np.log(1e-3)] + mean_hyp,
)

for tag, (params, s2, s2_new, hyp) in cases.items():
    hyp_arr = np.array(hyp, dtype=np.float64).reshape(1, -1)
    gp0 = make_gp(X, y, s2, params, hyp_arr)
    gp_r1 = copy.deepcopy(gp0)
    gp_r1.update(
        X_new=x_new, y_new=y_new, s2_new=s2_new, compute_posterior=True
    )
    s2_full = None if s2 is None else np.vstack([s2, s2_new])
    gp_full = make_gp(
        np.vstack([X, x_new]), np.vstack([y, y_new]), s2_full, params, hyp_arr
    )
    compare(tag, gp_r1, gp_full, Xt, Xa)
    print()

# ----------------------------------------------------------- sn2_mult > 1
print("Forcing a retried factorization (sn2_mult > 1):")
found = False
for sf, noise, dup in (
    (np.log(1e6), np.log(1e-3), 6),
    (np.log(1e8), np.log(1e-3), 8),
    (np.log(1e10), np.log(1e-3), 10),
):
    Xd = np.vstack([X[:3]] * dup)[: 3 * dup]
    Xd = Xd + 1e-13 * rng.standard_normal(Xd.shape)
    yd = (-0.5 * np.sum(Xd**2, 1)).reshape(-1, 1)
    s2d = np.full((Xd.shape[0], 1), 1e-3)
    hyp_arr = np.array(
        [np.log(1.0), np.log(1.0), sf, noise] + mean_hyp, dtype=np.float64
    ).reshape(1, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp0 = make_gp(Xd, yd, s2d, (1, 1, 0), hyp_arr)
    mult = gp0.posteriors[0].sn2_mult
    print(f"  sf2=exp(2*{sf:.2f}), N={Xd.shape[0]} -> sn2_mult = {mult}")
    if mult is not None and mult > 1:
        found = True
        gp_r1 = copy.deepcopy(gp0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp_r1.update(
                X_new=x_new,
                y_new=y_new,
                s2_new=np.array([[1e-3]]),
                compute_posterior=True,
            )
            gp_full = make_gp(
                np.vstack([Xd, x_new]),
                np.vstack([yd, y_new]),
                np.vstack([s2d, [[1e-3]]]),
                (1, 1, 0),
                hyp_arr,
            )
        compare("sn2_mult > 1", gp_r1, gp_full, Xt, Xa)
        break
if not found:
    print("  no retried factorization produced by these states")
