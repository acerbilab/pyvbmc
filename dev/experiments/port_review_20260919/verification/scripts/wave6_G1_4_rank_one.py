"""Settles question 2 of both G1 briefs: does the rank-one extension of
``GP.update`` equal a full recomputation in every state ``update`` accepts?

Covered states, all against a fresh GP rebuilt on the enlarged training set
at the same hyperparameters:

  * ``L_chol`` True with each noise parameterization, one and three
    hyperparameter samples;
  * a new point whose total noise crosses the ``1e-6`` threshold
    (G1 comparison, "departures", item 1);
  * ``L_chol`` False with a duplicated input and eps noise, the unguarded
    branch of ``gaussian_process.py:920-928`` (G1 internal F4);
  * a stored ``sn2_mult > 1``, forced by making the first Cholesky attempts
    fail (G1 comparison M3, which could not produce it);
  * posteriors emptied by ``clean()`` (G1 internal F8 / comparison M13);
  * the ``sqrt_arg <= 0`` guard of ``:885-899``.

It also transcribes ``gplite_post.m:226-244`` (read at vbmc 396d649) and
compares the two rank-one formulas, which differ once the noise is
heteroskedastic: MATLAB uses ``sn2_eff`` of the new point where the stored
factorization is scaled by ``sl = min(sn2)*sn2_mult``
(``gplite_core.m:82``).

Run from anywhere.
"""

import warnings

import gpyreg as gpr
import numpy as np
import scipy as sp

print("gpyreg:", gpr.__file__)

RNG = np.random.default_rng(11)


def make_gp(noise, D=2, n=12, s2=None, seed=5, log_sn=np.log(0.1)):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, D))
    y = np.sum(np.sin(X), axis=1, keepdims=True) + 1.0
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=noise,
    )
    n_noise = noise.hyperparameter_count()
    cov = np.concatenate([np.zeros(D), [0.0]])
    noise_h = [] if n_noise == 0 else [log_sn] + [0.0] * (n_noise - 1)
    mean_h = np.concatenate([[0.0], np.zeros(D), np.zeros(D)])
    hyp = np.concatenate([cov, noise_h, mean_h]).reshape(1, -1)
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp)
    return gp, hyp


def rebuild(gp_src, X_all, y_all, s2_all, hyp):
    gp = gpr.GP(
        D=gp_src.D,
        covariance=gp_src.covariance,
        mean=gp_src.mean,
        noise=gp_src.noise,
    )
    gp.update(X_new=X_all, y_new=y_all, s2_new=s2_all, hyp=hyp)
    return gp


def compare(label, gp_rank1, gp_ref, X_test):
    rows = []
    for s in range(gp_rank1.posteriors.size):
        p1, p2 = gp_rank1.posteriors[s], gp_ref.posteriors[s]
        da = np.max(np.abs(p1.alpha - p2.alpha)) / max(
            np.max(np.abs(p2.alpha)), 1e-300
        )
        rows.append(
            (
                s,
                da,
                p1.L_chol,
                p2.L_chol,
                p1.sl,
                p2.sl,
                p1.sn2_mult,
                p2.sn2_mult,
            )
        )
    m1, v1 = gp_rank1.predict(X_test, add_noise=True)
    m2, v2 = gp_ref.predict(X_test, add_noise=True)
    dm = np.max(np.abs(m1 - m2)) / max(np.max(np.abs(m2)), 1e-300)
    dv = np.max(np.abs(v1 - v2)) / max(np.max(np.abs(v2)), 1e-300)
    print(f"\n-- {label}")
    for s, da, lc1, lc2, sl1, sl2, sm1, sm2 in rows:
        print(
            f"   sample {s}: rel|dalpha|={da:.3g}  L_chol r1={lc1} ref={lc2}"
            f"  sl r1={sl1:.6g} ref={sl2:.6g}"
            f"  sn2_mult r1={sm1} ref={sm2}"
        )
    print(f"   rel predict mean={dm:.3g}  var={dv:.3g}")
    return max(r[1] for r in rows), dm, dv


X_TEST = RNG.uniform(-2.0, 2.0, size=(6, 2))

print("\n############ 1. L_chol True, the three noise levels ############")
configs = [
    (
        "level 0: constant only",
        gpr.noise_functions.GaussianNoise(constant_add=True),
        None,
    ),
    (
        "level 2: constant + s2 as is",
        gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=1
        ),
        "s2",
    ),
    (
        "level 1: constant + scaled s2",
        gpr.noise_functions.GaussianNoise(
            constant_add=True, scale_user_provided=True
        ),
        "s2",
    ),
]
for label, noise, want_s2 in configs:
    rng = np.random.default_rng(21)
    s2 = rng.uniform(0.01, 0.2, size=(12, 1)) if want_s2 else None
    gp, hyp = make_gp(noise, s2=s2)
    X_new = RNG.uniform(-2.0, 2.0, size=(1, 2))
    y_new = np.array([[0.3]])
    s2_new = np.array([[0.05]]) if want_s2 else None
    X_all = np.vstack([gp.X, X_new])
    y_all = np.vstack([gp.y, y_new])
    s2_all = np.vstack([s2, s2_new]) if want_s2 else None
    ref = rebuild(gp, X_all, y_all, s2_all, hyp)
    gp.update(X_new=X_new, y_new=y_new, s2_new=s2_new)
    compare(label, gp, ref, X_TEST)

print("\n############ 2. three hyperparameter samples ############")
noise = gpr.noise_functions.GaussianNoise(constant_add=True)
gp, hyp1 = make_gp(noise)
hyp3 = np.vstack([hyp1, hyp1 * 1.05, hyp1 * 0.95])
gp.update(hyp=hyp3)
X_new = RNG.uniform(-2.0, 2.0, size=(1, 2))
y_new = np.array([[0.3]])
ref = rebuild(
    gp, np.vstack([gp.X, X_new]), np.vstack([gp.y, y_new]), None, hyp3
)
gp.update(X_new=X_new, y_new=y_new)
compare("3 samples, constant noise", gp, ref, X_TEST)

print("\n############ 3. the new point's noise crosses 1e-6 ############")
# Training noise above the threshold; the new point carries a user-provided
# variance so small that the total noise of the enlarged set falls below it.
noise = gpr.noise_functions.GaussianNoise(
    constant_add=True, user_provided_add=1
)
rng = np.random.default_rng(31)
s2 = rng.uniform(1e-5, 2e-5, size=(12, 1))
gp, hyp = make_gp(noise, s2=s2, log_sn=0.5 * np.log(1e-7))
print(
    "   min total noise of the training set:",
    float(np.min(noise.compute(hyp[0, 3:4], gp.X, gp.y, s2))),
    " stored L_chol =",
    gp.posteriors[0].L_chol,
)
X_new = RNG.uniform(-2.0, 2.0, size=(1, 2))
y_new = np.array([[0.3]])
s2_new = np.array([[0.0]])
X_all, y_all = np.vstack([gp.X, X_new]), np.vstack([gp.y, y_new])
s2_all = np.vstack([s2, s2_new])
print(
    "   min total noise after the append:",
    float(np.min(noise.compute(hyp[0, 3:4], X_all, y_all, s2_all))),
)
ref = rebuild(gp, X_all, y_all, s2_all, hyp)
gp.update(X_new=X_new, y_new=y_new, s2_new=s2_new)
compare("noise crossing the 1e-6 threshold", gp, ref, X_TEST)

print(
    "\n############ 4. L_chol False, duplicated input (internal F4) ############"
)
noise = gpr.noise_functions.GaussianNoise(constant_add=False)
gp, hyp = make_gp(noise)
print(
    "   sn2 =",
    float(np.ravel(noise.compute(np.array([]), gp.X, gp.y, None))[0]),
    " L_chol =",
    gp.posteriors[0].L_chol,
)
X_new = gp.X[3:4, :].copy()  # an exact copy of an existing row
y_new = gp.y[3:4, :].copy()
X_all, y_all = np.vstack([gp.X, X_new]), np.vstack([gp.y, y_new])
ref = rebuild(gp, X_all, y_all, None, hyp)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    gp.update(X_new=X_new, y_new=y_new)
    print("   warnings raised:", [str(x.message) for x in w])
compare("L_chol False, exact duplicate", gp, ref, X_TEST)
# Which of the two is wrong?  Compare both with alpha = C^-1 (y - m) built
# from scratch at extended precision-free arithmetic.
K = gp.covariance.compute(hyp[0, :3], X_all)
sn2v = float(np.ravel(gp.noise.compute(np.array([]), X_all, y_all, None))[0])
m = np.reshape(gp.mean.compute(hyp[0, 3:], X_all), (-1, 1))
C = K + sn2v * np.eye(X_all.shape[0])
alpha_direct = np.linalg.lstsq(C, y_all - m, rcond=None)[0]
for nm, g in (("rank-one", gp), ("rebuild", ref)):
    e = np.max(np.abs(g.posteriors[0].alpha - alpha_direct)) / np.max(
        np.abs(alpha_direct)
    )
    print(f"   rel|alpha - lstsq(C, y-m)| for the {nm}: {e:.3g}")
print("   cond(C) =", np.linalg.cond(C))

print("\n#### 4b. how large the low-noise error gets (sweep) ####")
for log_sf, dup in (
    (0.0, "exact"),
    (3.0, "exact"),
    (6.0, "exact"),
    (0.0, "near"),
    (3.0, "near"),
):
    noise = gpr.noise_functions.GaussianNoise(constant_add=False)
    gp, hyp = make_gp(noise)
    hyp = hyp.copy()
    hyp[0, 2] = log_sf  # covariance_log_outputscale
    gp.update(hyp=hyp)
    X_new = gp.X[3:4, :].copy()
    if dup == "near":
        X_new = X_new + 1e-9
    y_new = gp.y[3:4, :].copy() + 0.01
    X_all, y_all = np.vstack([gp.X, X_new]), np.vstack([gp.y, y_new])
    ref = rebuild(gp, X_all, y_all, None, hyp)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gp.update(X_new=X_new, y_new=y_new)
        nwarn = len(w)
    da = np.max(
        np.abs(gp.posteriors[0].alpha - ref.posteriors[0].alpha)
    ) / np.max(np.abs(ref.posteriors[0].alpha))
    m1, v1 = gp.predict(X_TEST, add_noise=True)
    m2, v2 = ref.predict(X_TEST, add_noise=True)
    dv = np.max(np.abs(v1 - v2)) / np.max(np.abs(v2))
    print(
        f"   log_sf={log_sf:4.1f} {dup:5s} duplicate: rel|dalpha|={da:.3g}"
        f"  rel dvar={dv:.3g}  warnings={nwarn}  L_chol={gp.posteriors[0].L_chol}"
    )

print("\n############ 5. a stored sn2_mult > 1 (comparison M3) ############")
# Force the retry ladder by making the first two Cholesky attempts fail.
real_chol = sp.linalg.cholesky
state = {"n": 0}


def flaky_chol(a, **kw):
    state["n"] += 1
    if state["n"] <= 2:
        raise sp.linalg.LinAlgError("forced failure")
    return real_chol(a, **kw)


noise = gpr.noise_functions.GaussianNoise(constant_add=True)
gp, hyp = make_gp(noise)
sp.linalg.cholesky = flaky_chol
try:
    gp.update(hyp=hyp)  # recompute the posterior with the flaky factorization
finally:
    sp.linalg.cholesky = real_chol
p = gp.posteriors[0]
print("   stored sn2_mult =", p.sn2_mult, " sl =", p.sl, " L_chol =", p.L_chol)
X_new = RNG.uniform(-2.0, 2.0, size=(1, 2))
y_new = np.array([[0.3]])
ref = rebuild(
    gp, np.vstack([gp.X, X_new]), np.vstack([gp.y, y_new]), None, hyp
)
print("   rebuild sn2_mult =", ref.posteriors[0].sn2_mult)
gp.update(X_new=X_new, y_new=y_new)
compare("stored sn2_mult = 100, rebuild sn2_mult = 1", gp, ref, X_TEST)
# and against C and alpha built from scratch with the STORED sn2_mult
gp2, hyp2 = make_gp(gpr.noise_functions.GaussianNoise(constant_add=True))
X_all = np.vstack([gp2.X, X_new])
y_all = np.vstack([gp2.y, y_new])
K = gp2.covariance.compute(hyp2[0, :3], X_all)
sn2 = float(np.ravel(gp2.noise.compute(hyp2[0, 3:4], X_all, y_all, None))[0])
m = np.reshape(gp2.mean.compute(hyp2[0, 4:], X_all), (-1, 1))
for mult in (1.0, 100.0):
    C = K + mult * sn2 * np.eye(X_all.shape[0])
    alpha_direct = np.linalg.solve(C, y_all - m)
    err = np.max(np.abs(gp.posteriors[0].alpha - alpha_direct)) / np.max(
        np.abs(alpha_direct)
    )
    print(
        f"   rank-one alpha vs C^-1(y-m) with sn2_mult={mult:g}: rel {err:.3g}"
    )

print(
    "\n############ 6. posteriors emptied by clean() (internal F8) ############"
)
noise = gpr.noise_functions.GaussianNoise(constant_add=True)
gp, hyp = make_gp(noise)
gp.clean()
try:
    gp.update(X_new=np.array([[0.2, 0.2]]), y_new=np.array([[0.1]]))
    print("   no error")
except Exception as exc:  # noqa: BLE001
    print(f"   {type(exc).__name__}: {exc}")
gp2, hyp2 = make_gp(noise)
gp2.clean()
gp2.update(compute_posterior=True)
print(
    "   update(compute_posterior=True) after clean(): alpha restored =",
    gp2.posteriors[0].alpha is not None,
)
# compute_posterior=False then a one-point update
gp3, hyp3b = make_gp(noise)
gp3.update(hyp=hyp3b, compute_posterior=False)
try:
    gp3.update(X_new=np.array([[0.2, 0.2]]), y_new=np.array([[0.1]]))
    print("   after compute_posterior=False: no error")
except Exception as exc:  # noqa: BLE001
    print(f"   after compute_posterior=False: {type(exc).__name__}: {exc}")

print("\n############ 7. MATLAB's formula against gpyreg's ############")


# gplite_post.m:226-244 uses sn2_eff of the new point as the scale of the
# extended factor and appends 1/sqrt(sn2_eff) to sW; gplite_core.m:82 sets
# the stored scale to sl = min(sn2)*sn2_mult.  Compare both extensions with
# the represented matrix (K + sn2_mult*diag(sn2)) / sl.
def matlab_rank_one(gp, hyp, X_new, y_new, s2_new, s=0):
    D = gp.D
    cov_N = gp.covariance.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()
    post = gp.posteriors[s]
    sn2 = float(
        np.ravel(
            gp.noise.compute(
                hyp[cov_N : cov_N + noise_N],
                X_new,
                y_new,
                0 if s2_new is None else s2_new,
            )
        )[0]
    )
    sn2_eff = sn2 * post.sn2_mult
    K = gp.covariance.compute(hyp[0:cov_N], X_new)
    Ks = gp.covariance.compute(hyp[0:cov_N], gp.X, X_new)
    L = post.L
    c = (
        sp.linalg.solve_triangular(L, Ks, trans=1, check_finite=False)
        / sn2_eff
    )
    d = np.sqrt(1.0 + K / sn2_eff - c.T @ c)
    L_new = np.block([[L, c], [np.zeros((1, L.shape[0])), d]])
    sW_new = np.concatenate((post.sW, np.array([[1 / np.sqrt(sn2_eff)]])))
    return L_new, sW_new


for label, noise, s2 in (
    (
        "homoskedastic (constant noise only)",
        gpr.noise_functions.GaussianNoise(constant_add=True),
        None,
    ),
    (
        "heteroskedastic (constant + user s2)",
        gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=1
        ),
        np.random.default_rng(41).uniform(0.01, 0.4, size=(12, 1)),
    ),
):
    gp, hyp = make_gp(noise, s2=s2)
    X_new = RNG.uniform(-2.0, 2.0, size=(1, 2))
    y_new = np.array([[0.3]])
    s2_new = np.array([[0.05]]) if s2 is not None else None
    sl = gp.posteriors[0].sl
    L_ml, sW_ml = matlab_rank_one(gp, hyp[0], X_new, y_new, s2_new)
    gp_py = gp
    gp_py.update(X_new=X_new, y_new=y_new, s2_new=s2_new)
    L_py = gp_py.posteriors[0].L
    # the matrix the representation must reproduce
    X_all, y_all = gp_py.X, gp_py.y
    s2_all = gp_py.s2
    K = gp_py.covariance.compute(hyp[0, :3], X_all)
    sn2_all = gp_py.noise.compute(
        hyp[0, 3 : 3 + noise.hyperparameter_count()], X_all, y_all, s2_all
    )
    sn2_all = np.ravel(sn2_all) * np.ones(X_all.shape[0])
    C = K + gp_py.posteriors[0].sn2_mult * np.diag(sn2_all)
    err_py = np.max(np.abs(L_py.T @ L_py * sl - C)) / np.max(np.abs(C))
    err_ml = np.max(np.abs(L_ml.T @ L_ml * sl - C)) / np.max(np.abs(C))
    sW_py = gp_py.posteriors[0].sW
    print(f"   {label}")
    print(
        f"     sl = {sl:.6g}; new point's sn2_eff = "
        f"{float(np.ravel(noise.compute(hyp[0, 3:3+noise.hyperparameter_count()], X_new, y_new, 0 if s2_new is None else s2_new))[0]) * gp_py.posteriors[0].sn2_mult:.6g}"
    )
    print(
        f"     rel|L^T L * sl - C|  gpyreg {err_py:.3g}   MATLAB {err_ml:.3g}"
    )
    print(
        f"     last sW entry        gpyreg {sW_py[-1, 0]:.8g}  "
        f"MATLAB {sW_ml[-1, 0]:.8g}  (all others {sW_py[0, 0]:.8g})"
    )
