"""Settles the first question of slice G2: which variance the log predictive
density of `GP.predict` carries, in each noise parameterization, with and
without `s2_star`, with `add_noise` both ways, over one and several
hyperparameter samples, and with a retried factorization (sn2_mult > 1);
and how the combination over hyperparameter samples compares with
`gplite_pred.m` (the ddof=1 sample variance included).

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python G2_0_predict_lpd.py
Needs `matlab_gplite_g2.py` next to it.
"""

import os
import sys

import numpy as np
import scipy.stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpyreg as gpr
import wave6_G2_matlab_gplite_g2 as M

print("gpyreg:", gpr.__file__)
np.set_printoptions(precision=12, suppress=False, linewidth=140)


def build(params, Ns, D=2, N=20, seed=7, s2=None, jitter=0.0):
    """A GP with hyperparameters set by hand (no fit), PyVBMC's components."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(N, D))
    y = np.sum(np.sin(X), axis=1).reshape(-1, 1) + 0.1 * rng.standard_normal(
        (N, 1)
    )
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=params[0] == 1,
            user_provided_add=params[1] > 0,
            scale_user_provided=params[1] == 2,
            rectified_linear_output_dependent_add=params[2] == 1,
        ),
    )
    n_cov = D + 1
    n_noise = gp.noise.hyperparameter_count()
    n_mean = 1 + 2 * D
    hyps = np.zeros((Ns, n_cov + n_noise + n_mean))
    for s in range(Ns):
        h = np.zeros(n_cov + n_noise + n_mean)
        h[0:D] = np.log(0.8 + 0.1 * s + 0.05 * np.arange(D))
        h[D] = np.log(1.3 + 0.2 * s) + jitter
        i = n_cov
        if params[0] == 1:
            h[i] = np.log(0.2 + 0.05 * s)
            i += 1
        if params[1] == 2:
            h[i] = np.log(1.5)
            i += 1
        if params[2] == 1:
            h[i] = 0.3
            h[i + 1] = np.log(0.05)
            i += 2
        h[n_cov + n_noise] = 0.5 + 0.1 * s
        h[n_cov + n_noise + 1 : n_cov + n_noise + 1 + D] = 0.1 * s
        h[n_cov + n_noise + 1 + D :] = np.log(2.0)
        hyps[s] = h
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyps)
    return gp, X, y


def matlab_posts(gp, params, X, y, s2):
    return [
        M.core_posterior(p.hyp, X, y, params, 4, s2) for p in gp.posteriors
    ]


print("\n=== A. predict against the gplite_pred.m transcription ===")
rng = np.random.default_rng(3)
Xs = rng.uniform(-2, 2, size=(6, 2))
ys = rng.standard_normal((6, 1))
s2s = np.abs(rng.standard_normal((6, 1))) * 0.3 + 0.05
worst = 0.0
for params, name in [
    ((1, 0, 0), "level 0: constant"),
    ((1, 1, 0), "level 2: constant+user"),
    ((1, 2, 0), "level 1: constant+scaled user"),
    ((1, 0, 1), "constant+rectified (not used by PyVBMC)"),
    ((0, 1, 0), "no constant + user"),
]:
    s2_tr = (
        np.abs(rng.standard_normal((20, 1))) * 0.1 + 0.02
        if params[1] > 0
        else None
    )
    for Ns in (1, 3):
        gp, X, y = build(params, Ns, s2=s2_tr)
        posts = matlab_posts(gp, params, X, y, s2_tr)
        for ss in (False, True):
            for an in (False, True):
                for use_s2 in (False, True):
                    s2_arg = s2s if use_s2 else None
                    mu, s2o, lpd = gp.predict(
                        Xs,
                        ys,
                        s2_arg,
                        add_noise=an,
                        separate_samples=ss,
                        return_lpd=True,
                    )
                    ymu, ys2, fmu, fs2, lp = M.pred(
                        posts, X, y, params, 4, Xs, ys, s2_arg, ssflag=ss
                    )
                    ref_mu = fmu
                    ref_s2 = ys2 if an else fs2
                    e1 = np.max(np.abs(mu - ref_mu))
                    e2 = np.max(np.abs(s2o - ref_s2))
                    worst = max(worst, e1, e2)
                    if ss:
                        e3 = np.max(np.abs(lpd - lp))
                        worst = max(worst, e3)
print("max |predict - gplite_pred| over mu and s2 (and lp when ss):", worst)

print("\n=== B. lpd is always the total-variance density ===")
for params, name in [
    ((1, 0, 0), "level 0"),
    ((1, 1, 0), "level 2"),
    ((1, 2, 0), "level 1"),
]:
    s2_tr = (
        np.abs(rng.standard_normal((20, 1))) * 0.1 + 0.02
        if params[1] > 0
        else None
    )
    gp, X, y = build(params, 1, s2=s2_tr)
    for use_s2 in (False, True):
        s2_arg = s2s if use_s2 else None
        mu, s2l, lpd = gp.predict(
            Xs,
            ys,
            s2_arg,
            add_noise=False,
            separate_samples=True,
            return_lpd=True,
        )
        _, s2t, _ = gp.predict(
            Xs,
            ys,
            s2_arg,
            add_noise=True,
            separate_samples=True,
            return_lpd=True,
        )
        d_lat = np.max(np.abs(lpd - st.norm.logpdf(ys, mu, np.sqrt(s2l))))
        d_tot = np.max(np.abs(lpd - st.norm.logpdf(ys, mu, np.sqrt(s2t))))
        print(
            f"  {name} s2_star={'yes' if use_s2 else 'no ':3s}"
            f" |lpd - logN(latent)|={d_lat:.3e}"
            f"  |lpd - logN(total)|={d_tot:.3e}"
        )

print("\n=== C. add_noise does not change lpd; s2_star ignored at level 0 ===")
gp, X, y = build((1, 0, 0), 3)
_, _, l_f = gp.predict(Xs, ys, None, add_noise=False, return_lpd=True)
_, _, l_t = gp.predict(Xs, ys, None, add_noise=True, return_lpd=True)
print(
    "  averaged, |lpd(add_noise=F) - lpd(add_noise=T)|:",
    np.max(np.abs(l_f - l_t)),
)
_, _, l_s = gp.predict(Xs, ys, s2s, add_noise=False, return_lpd=True)
print("  level 0: |lpd(s2_star) - lpd(None)| =", np.max(np.abs(l_s - l_f)))

print("\n=== D. separate_samples=False: moment matching vs MATLAB matrix ===")
gp, X, y = build((1, 0, 0), 3)
posts = matlab_posts(gp, (1, 0, 0), X, y, None)
mu, s2o, lpd = gp.predict(Xs, ys, None, add_noise=False, return_lpd=True)
_, ys2, fmu, fs2, lp = M.pred(
    posts, X, y, (1, 0, 0), 4, Xs, ys, None, ssflag=False
)
_, ys2s, fmus, fs2s, lps = M.pred(
    posts, X, y, (1, 0, 0), 4, Xs, ys, None, ssflag=True
)
print("  MATLAB lp shape (ssflag=False):", lp.shape, " python lpd:", lpd.shape)
mm = st.norm.logpdf(ys, mu, np.sqrt(ys2))
print(
    "  |python lpd - logN(y; mu_bar, mean(y_s2)+var_ddof1(mu))| =",
    np.max(np.abs(lpd - mm)),
)
print(
    "  |python lpd - mean over MATLAB columns| =",
    np.max(np.abs(lpd - np.mean(lps, axis=1, keepdims=True))),
)
print(
    "  |python lpd - log mean of densities| =",
    np.max(
        np.abs(lpd - (np.log(np.mean(np.exp(lps), axis=1, keepdims=True))))
    ),
)

print("\n=== E. ddof of the across-sample spread (predict and quad) ===")
mu_sep, s2_sep = gp.predict(
    Xs, None, None, add_noise=False, separate_samples=True
)
mu_avg, s2_avg = gp.predict(Xs, None, None, add_noise=False)
v1 = np.var(mu_sep, axis=1, ddof=1)
v0 = np.var(mu_sep, axis=1, ddof=0)
lhs = s2_avg.ravel() - np.mean(s2_sep, axis=1)
print(
    "  predict: max|returned spread - var(ddof=1)| =", np.max(np.abs(lhs - v1))
)
print(
    "  predict: max|returned spread - var(ddof=0)| =", np.max(np.abs(lhs - v0))
)
Fq_s = gp.quad(0.0, 1.0, compute_var=True, separate_samples=True)
Fq_a = gp.quad(0.0, 1.0, compute_var=True, separate_samples=False)
vq1 = np.var(Fq_s[0], axis=1, ddof=1)
vq0 = np.var(Fq_s[0], axis=1, ddof=0)
lhsq = Fq_a[1].ravel() - np.mean(Fq_s[1], axis=1)
print(
    "  quad:    max|returned spread - var(ddof=1)| =",
    np.max(np.abs(lhsq - vq1)),
)
print(
    "  quad:    max|returned spread - var(ddof=0)| =",
    np.max(np.abs(lhsq - vq0)),
)

print(
    "\n=== F. retried factorization: sn2_mult > 1, both L representations ==="
)
D = 2
for lnell, lnsf, lnsn, tag in [
    (np.log(1e5), 20.0, np.log(1e-3), "L_chol=True"),
    (np.log(1e4), 10.0, np.log(1e-6), "L_chol=False"),
]:
    rng2 = np.random.default_rng(11)
    Xd = rng2.uniform(-1, 1, size=(40, D))
    yd = np.sum(Xd, axis=1).reshape(-1, 1)
    gpd = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    h = np.zeros(D + 1 + 1 + 1 + 2 * D)
    h[0:D] = lnell
    h[D] = lnsf
    h[D + 1] = lnsn
    h[D + 3 : D + 3 + D] = 0.0
    h[D + 3 + D :] = np.log(3.0)
    gpd.update(X_new=Xd, y_new=yd, hyp=h[None, :])
    post = gpd.posteriors[0]
    print(
        f"  [{tag}] sn2_mult = {post.sn2_mult:g}  L_chol = {post.L_chol}"
        f"  sl = {post.sl:g}"
    )
    postd = [M.core_posterior(post.hyp, Xd, yd, (1, 0, 0), 4, None)]
    print(
        "    MATLAB transcription sn2_mult =",
        postd[0]["sn2_mult"],
        " L_chol =",
        postd[0]["Lchol"],
    )
    Xt = rng2.uniform(-1, 1, size=(5, D))
    yt = rng2.standard_normal((5, 1))
    mu, s2o, lpd = gpd.predict(Xt, yt, None, add_noise=True, return_lpd=True)
    ymu, ys2, fmu, fs2, lp = M.pred(
        postd, Xd, yd, (1, 0, 0), 4, Xt, yt, None, ssflag=True
    )
    print(
        "    |mu - MATLAB| =",
        np.max(np.abs(mu - fmu)),
        " |s2(add_noise) - MATLAB ys2| =",
        np.max(np.abs(s2o - ys2)),
        " |lpd - MATLAB lp| =",
        np.max(np.abs(lpd - lp)),
    )
    _, s2lat = gpd.predict(Xt, add_noise=False)
    sn2 = np.exp(2 * h[D + 1])
    ratio = (s2o - s2lat) / (sn2 * post.sn2_mult)
    print(
        "    (s2_total - s2_latent) / (sn2 * sn2_mult) =",
        np.round(ratio.ravel(), 12),
    )
    print(
        "    lpd carries the total variance:",
        bool(np.allclose(lpd, st.norm.logpdf(yt, mu, np.sqrt(s2o)))),
    )
