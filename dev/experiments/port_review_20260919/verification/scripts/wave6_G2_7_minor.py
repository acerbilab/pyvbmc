"""Checks the minor observations of both G2 reports, one block each, and two
observations of my own.

Blocks: the compute_diag return shape against the abstract docstring; the
noise arguments `random_function` drops; the silent omissions of the
output-dependent term and of `s2_star`; `quad`'s guards (prior-only GP, a 1-D
`mu`, `sn2_mult is None`, an unvalidated mean function); the clipping of
negative variances in `predict` but not `predict_full`; the hidden
`np.spacing(1)` nugget with `constant_add=False`; `_convert_shapes` input
types; `return_lpd` without `y_star`; the `SquaredExponential` diag+grad
combination; the docstring defaults; the `sq_dist`/`cdist` rounding gap; and
the error the non-Cholesky branch raises when every factorization attempt
fails.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python G2_7_minor.py
Needs `matlab_gplite_g2.py` next to it.
"""

import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpyreg as gpr
import gpyreg.isotropic_covariance_functions as iso
import wave6_G2_matlab_gplite_g2 as M

print("gpyreg:", gpr.__file__)
np.set_printoptions(precision=8, suppress=False, linewidth=170)

D = 2
rng = np.random.default_rng(31)
X = rng.uniform(-2, 2, size=(16, D))
y = np.sum(np.sin(X), 1).reshape(-1, 1)
s2 = np.abs(rng.standard_normal((16, 1))) * 0.05 + 0.01
Xs = rng.uniform(-2, 2, size=(4, D))
ys = rng.standard_normal((4, 1))
s2s = np.abs(rng.standard_normal((4, 1))) * 0.3 + 0.05


def gp_with(params, s2_train=None, mean=None, cov=None):
    gp = gpr.GP(
        D=D,
        covariance=cov or gpr.covariance_functions.SquaredExponential(),
        mean=mean or gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=params[0] == 1,
            user_provided_add=params[1] > 0,
            scale_user_provided=params[1] == 2,
            rectified_linear_output_dependent_add=params[2] == 1,
        ),
    )
    n_cov = gp.covariance.hyperparameter_count(D)
    n_noise = gp.noise.hyperparameter_count()
    n_mean = gp.mean.hyperparameter_count(D)
    h = np.zeros(n_cov + n_noise + n_mean)
    h[0:n_cov] = np.log(1.0)
    i = n_cov
    if params[0] == 1:
        h[i] = np.log(0.2)
        i += 1
    if params[1] == 2:
        h[i] = np.log(1.2)
        i += 1
    if params[2] == 1:
        h[i] = 0.4
        h[i + 1] = np.log(0.3)
    if n_mean:
        h[n_cov + n_noise] = 0.5
        if n_mean > 1:
            h[n_cov + n_noise + 1 + D :] = np.log(2.0)
    gp.update(X_new=X, y_new=y, s2_new=s2_train, hyp=h[None, :])
    return gp


print("\n--- compute_diag return shape vs AbstractKernel's docstring ---")
doc = gpr.covariance_functions.AbstractKernel.compute.__doc__
print(
    "  docstring says:",
    [l.strip() for l in doc.splitlines() if "compute_diag = True" in l][0],
)
for k in (
    gpr.covariance_functions.SquaredExponential(),
    gpr.covariance_functions.Matern(3),
    gpr.covariance_functions.RationalQuadraticARD(),
    iso.SquaredExponentialIsotropic(),
    iso.MaternIsotropic(3),
):
    n = k.hyperparameter_count(D)
    out = k.compute(np.zeros(n), X, compute_diag=True)
    print(f"  {type(k).__name__:32s} returns shape {np.shape(out)}")
print(
    "  predict indexes the result with [:, 0] (gaussian_process.py:1994),"
    " so a (N,) return raises IndexError:"
)
try:
    np.zeros(4)[:, 0]
except IndexError as e:
    print("   ", type(e).__name__, e)

print("\n--- random_function's noise arguments ---")
src = inspect.getsource(gpr.GP.random_function)
line = [l.strip() for l in src.splitlines() if "hyp[cov_N : cov_N" in l][0]
print("  call:", line, "-> y and s2 are both None")
gp = gp_with((1, 1, 0), s2)
hn = gp.posteriors[0].hyp[D + 1 : D + 1 + gp.noise.hyperparameter_count()]
print(
    "  noise with (None, None):",
    gp.noise.compute(hn, Xs, None, None),
    " with s2_star:",
    np.ravel(gp.noise.compute(hn, Xs, None, s2s)),
)
print(
    "  MATLAB gplite_rnd.m:67 also calls gplite_noisefun(hyp,Xstar,fun)"
    " with no y and no s2 -> the same omission:",
    M.noisefun(hn, Xs, (1, 1, 0)),
)

print("\n--- silent omissions in predict ---")
gp = gp_with((1, 0, 1))
_, s2_with_y = gp.predict(Xs, ys, None, add_noise=True)
_, s2_no_y = gp.predict(Xs, None, None, add_noise=True)
print("  output-dependent noise, y_star given :", s2_with_y.ravel())
print("  output-dependent noise, y_star = None:", s2_no_y.ravel())
print("  MATLAB's `if ~isempty(y)` does the same (gplite_noisefun.m:198)")
gp = gp_with((1, 0, 0))
_, a = gp.predict(Xs, None, s2s, add_noise=True)
_, b = gp.predict(Xs, None, None, add_noise=True)
print("  constant-only noise: s2_star ignored:", bool(np.array_equal(a, b)))

print("\n--- quad's guards ---")
gp_prior = gpr.GP(
    D=D,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.NegativeQuadratic(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
gp_prior.update(hyp=np.zeros((1, D + 1 + 1 + 1 + 2 * D)))
try:
    gp_prior.quad(0.0, 1.0)
except Exception as e:
    print("  prior-only GP:", type(e).__name__, "-", e)
print(
    "  (predict on the same GP returns the prior:",
    gp_prior.predict(Xs)[0].ravel(),
    ")",
)
gp = gp_with((1, 0, 0))
try:
    gp.quad(np.zeros(D), np.ones(D))
except Exception as e:
    print("  1-D mu of shape (D,):", type(e).__name__, "-", e)
print("  scalar mu/sigma:", gp.quad(0.0, 1.0).ravel())
print("  (1,D) mu:", gp.quad(np.zeros((1, D)), np.ones((1, D))).ravel())
gp.clean()
try:
    gp.quad(0.0, 1.0, compute_var=True)
except Exception as e:
    print("  after clean(), compute_var=True:", type(e).__name__, "-", e)
gp = gp_with((1, 0, 0), mean=gpr.mean_functions.ConstantMean())
print(
    "  quad accepts ConstantMean (MATLAB allows meanfun 0,1,4,6,8):",
    gp.quad(0.0, 1.0).ravel(),
)
for cov in (
    gpr.covariance_functions.Matern(3),
    gpr.covariance_functions.RationalQuadraticARD(),
    iso.MaternIsotropic(3),
):
    gp2 = gp_with((1, 0, 0), cov=cov)
    try:
        gp2.quad(0.0, 1.0)
        print(f"  quad with {type(cov).__name__}: accepted")
    except ValueError as e:
        print(f"  quad with {type(cov).__name__}: ValueError - {e}")
gp3 = gp_with((1, 0, 0), cov=iso.SquaredExponentialIsotropic())
print("  quad with SquaredExponentialIsotropic:", gp3.quad(0.0, 1.0).ravel())

print("\n--- clipping of negative variances ---")
gpn = gpr.GP(
    D=1,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.ZeroMean(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
Xn = np.linspace(-1, 1, 30).reshape(-1, 1)
gpn.update(
    X_new=Xn,
    y_new=np.sin(Xn),
    hyp=np.array([[np.log(3.0), np.log(6.0), np.log(1e-3)]]),
)
Xt = np.linspace(-1, 1, 60).reshape(-1, 1)
_, s2p = gpn.predict(Xt, add_noise=False)
_, cvp = gpn.predict_full(Xt, add_noise=False)
dg = np.diag(cvp[:, :, 0])
print(
    "  entries where predict_full's diagonal is negative:",
    int(np.sum(dg < 0)),
    " most negative:",
    float(np.min(dg)),
)
print(
    "  the same entries in predict's s2 (clipped at 0):",
    s2p.ravel()[dg < 0][:5],
)
print(
    "  max|predict s2 - max(diag, 0)| =",
    float(np.max(np.abs(s2p.ravel() - np.maximum(dg, 0)))),
)
print(
    "  gplite_pred.m:120 clamps fs2; gplite_rnd/gplite_pred return no"
    " full matrix, so MATLAB has no counterpart for predict_full"
)

print("\n--- constant_add=False: the hidden nugget ---")
nf = gpr.noise_functions.GaussianNoise(
    constant_add=False, user_provided_add=True
)
print(
    "  noise_N =",
    nf.hyperparameter_count(),
    " compute(s2=None) =",
    nf.compute(np.array([]), X, y, None),
    " = np.spacing(1):",
    nf.compute(np.array([]), X, y, None) == np.spacing(1.0),
)
print(
    "  MATLAB `sn2 = eps` (gplite_noisefun.m:179) =",
    M.noisefun(np.array([]), X, (0, 1, 0), y, None),
)

print("\n--- _convert_shapes input types ---")
gp = gp_with((1, 1, 0), s2)
for val, tag in [
    (3.0, "python float"),
    (3, "python int"),
    (np.float64(3.0), "np.float64"),
    (np.int64(3), "np.int64"),
    (np.array(3.0), "0-d ndarray"),
]:
    try:
        gp.predict(Xs, None, val)
        print(f"  s2_star as {tag:14s}: accepted")
    except Exception as e:
        print(
            f"  s2_star as {tag:14s}: {type(e).__name__} - "
            f"{str(e).split(chr(10))[0]}"
        )
try:
    gp.predict(Xs, None, np.zeros((1, 4)))
    print("  s2_star of shape (1, M): accepted silently (reshaped)")
except Exception as e:
    print("  s2_star of shape (1, M):", type(e).__name__)
try:
    gp.predict(np.zeros((3, 5)))
except Exception as e:
    print("  wrong D:", type(e).__name__, "-", repr(str(e)))

print("\n--- return_lpd without y_star ---")
try:
    gp.predict(Xs, None, None, return_lpd=True)
except Exception as e:
    print(" ", type(e).__name__, "-", e)
print("  gplite_pred.m:32-36 returns lp = [] instead")

print("\n--- SquaredExponential compute_diag and compute_grad together ---")
k = gpr.covariance_functions.SquaredExponential()
out = k.compute(np.zeros(D + 1), X, compute_diag=True, compute_grad=True)
print("  returns shapes:", [np.shape(o) for o in out])

print("\n--- docstring defaults and declared shapes ---")
sig = inspect.signature(gpr.GP.predict_full)
print(
    "  predict_full signature default add_noise =",
    sig.parameters["add_noise"].default,
)
print(
    "  docstring line:",
    [
        l.strip()
        for l in gpr.GP.predict_full.__doc__.splitlines()
        if "add_noise" in l and "defaults" in l
    ][0],
)
sig = inspect.signature(gpr.GP.predict)
print(
    "  predict signature default add_noise =",
    sig.parameters["add_noise"].default,
    " docstring:",
    [
        l.strip()
        for l in gpr.GP.predict.__doc__.splitlines()
        if "add_noise" in l and "defaults" in l
    ][0],
)
for obj, nm in [
    (gpr.covariance_functions.AbstractKernel.get_bounds_info, "covariance"),
    (gpr.mean_functions.ConstantMean.get_bounds_info, "mean"),
    (gpr.noise_functions.GaussianNoise.get_bounds_info, "noise"),
    (iso.AbstractIsotropicKernel.get_bounds_info, "isotropic"),
]:
    d = obj.__doc__ or ""
    decl = [l.strip() for l in d.splitlines() if "**LB**" in l]
    shp = (
        gpr.covariance_functions.SquaredExponential()
        .get_bounds_info(X, y)["LB"]
        .shape
    )
    print(f"  {nm:10s} docstring declares {decl[:1]}; actual LB shape {shp}")

print("\n--- quad docstring typo and the stale comment ---")
print("  'conputed' in quad's docstring:", "conputed" in gpr.GP.quad.__doc__)
srcfile = gpr.gaussian_process.__file__
with open(srcfile, encoding="utf-8") as f:
    txt = f.read()
print(
    '  "sigma doesn\'t work, requires gplite_quad implementation" present:',
    "sigma doesn't work, requires gplite_quad implementation" in txt,
)

print("\n--- sq_dist (MATLAB) against cdist (gpyreg) ---")
hyp = np.array([np.log(0.9), np.log(0.8), np.log(1.3)])
Kpy = gpr.covariance_functions.SquaredExponential().compute(hyp, X)
Kml = M.covfun_se(hyp, X)
print("  self-covariance max|K_py - K_ml| =", np.max(np.abs(Kpy - Kml)))
Kpy2 = gpr.covariance_functions.SquaredExponential().compute(hyp, X, Xs)
Kml2 = M.covfun_se(hyp, X, Xs)
print("  cross-covariance max|K_py - K_ml| =", np.max(np.abs(Kpy2 - Kml2)))

print("\n--- the negative-quadratic mean and its gradient vs MATLAB ---")
hm = np.array([0.7, 0.1, -0.2, np.log(1.3), np.log(0.8)])
m_py, dm_py = gpr.mean_functions.NegativeQuadratic().compute(
    hm, X, compute_grad=True
)
m_ml = M.meanfun_negquad(hm, X)
print("  max|m_py - m_ml| =", np.max(np.abs(m_py - m_ml)))
print(
    "  dm columns: [1, (X-xm)/omega^2, z2] reproduced:",
    bool(np.allclose(dm_py[:, 0], 1.0))
    and bool(
        np.allclose(
            dm_py[:, 1 : 1 + D], (X - hm[1 : 1 + D]) / np.exp(hm[1 + D :]) ** 2
        )
    )
    and bool(
        np.allclose(
            dm_py[:, 1 + D :], ((X - hm[1 : 1 + D]) / np.exp(hm[1 + D :])) ** 2
        )
    ),
)
print(
    "  compute_batched equals compute column by column:",
    bool(
        np.array_equal(
            gpr.mean_functions.NegativeQuadratic().compute_batched(
                np.stack([hm, hm + 0.1]), X
            ),
            np.stack(
                [
                    gpr.mean_functions.NegativeQuadratic().compute(hm, X),
                    gpr.mean_functions.NegativeQuadratic().compute(
                        hm + 0.1, X
                    ),
                ],
                axis=1,
            ),
        )
    ),
)

print("\n--- MY OWN: the error when every factorization attempt fails ---")
print(
    "  __core_computation's non-Cholesky branch forms"
    " `pL = solve_triangular(-L, ...)` before the `if L is None: raise"
    " LinAlgError` guard, so a total failure surfaces as a TypeError:"
)
gbad = gpr.GP(
    D=2,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.NegativeQuadratic(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
rb = np.random.default_rng(1)
Xb = rb.uniform(-1, 1, size=(40, 2))
hb = np.zeros(2 + 1 + 1 + 1 + 4)
hb[0:2] = np.log(1e5)
hb[2] = 20.0
hb[3] = np.log(1e-7)
hb[6:] = np.log(3.0)
try:
    gbad.update(X_new=Xb, y_new=np.sum(Xb, 1).reshape(-1, 1), hyp=hb[None, :])
except Exception as e:
    print("   ", type(e).__name__, "-", e)
hb[3] = np.log(3e-3)  # L_chol = True: the guard is reached
hb[0:2] = np.log(1e6)
hb[2] = 25.0
gbad2 = gpr.GP(
    D=2,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.NegativeQuadratic(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
try:
    gbad2.update(X_new=Xb, y_new=np.sum(Xb, 1).reshape(-1, 1), hyp=hb[None, :])
except Exception as e:
    print("    Cholesky branch, same situation:", type(e).__name__, "-", e)
