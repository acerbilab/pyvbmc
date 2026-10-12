"""Settles every claim about the bound recommendations of gpyreg's components
against a transcription of the `info` branches of `gplite_covfun.m`,
`gplite_meanfun.m` and `gplite_noisefun.m`:

  * covariance_functions.py:451 and :401  -- `np.log(np.std(X, ddof=1))` with
    no axis, where gplite_covfun.m:126 has `log(std(X))` per column;
  * mean_functions.py:488, :511-515, :518-524 -- `np.max/min/median/std` of
    the whole of X, where gplite_meanfun.m:142, :220-230 are per column;
  * covariance_functions.py:414 -- `plausible_upper_bounds[D] = 5.0` in a
    block whose other four lines index [-1];
  * isotropic_covariance_functions.py:233-246 -- `np.min`/`np.max` of a
    scalar, and log-of-mean where gplite_covfun.m:112-118 has mean-of-log;
  * the noise recommendations, which are claimed to match exactly;
  * the single-training-point edge, where MATLAB's reductions act along the
    row and the two sides swap places.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python G2_3_bounds.py
Needs `matlab_gplite_g2.py` next to it.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpyreg as gpr
import gpyreg.isotropic_covariance_functions as iso
import wave6_G2_matlab_gplite_g2 as M

print("gpyreg:", gpr.__file__)
np.set_printoptions(precision=6, suppress=True, linewidth=170)

# A training set whose coordinates differ in scale.
rng = np.random.default_rng(12)
N, D = 30, 3
X = np.column_stack(
    [
        rng.normal(0.0, 0.4, N),
        rng.normal(1.0, 1.0, N),
        rng.normal(-3.0, 4.0, N),
    ]
)
y = (np.sum(X, 1) + rng.standard_normal(N)).reshape(-1, 1)
print("per-dimension std  :", np.std(X, axis=0, ddof=1))
print("pooled std         :", np.std(X, ddof=1))
print("per-dimension width:", np.max(X, 0) - np.min(X, 0))
print("pooled width       :", np.max(X) - np.min(X))


def show(tag, py, ml):
    print(f"\n  {tag}")
    for k in ("LB", "PLB", "PUB", "UB", "x0"):
        p = np.asarray(py[k], dtype=float)
        m = np.asarray(ml[k], dtype=float)
        same = np.allclose(p, m, rtol=1e-12, atol=1e-12, equal_nan=True)
        print(f"    {k:4s} py={p}  ml={m}  {'EQUAL' if same else 'DIFFERS'}")


print("\n=== A. SE-ARD covariance recommendations ===")
py = gpr.covariance_functions.SquaredExponential().get_bounds_info(X, y)
ml = M.covfun_info(X, y, isoflag=False)
show("SquaredExponential vs gplite_covfun case 1", py, ml)
print(
    "    exp(x0) py:",
    np.exp(py["x0"][:D]),
    " ml:",
    np.exp(ml["x0"][:D]),
    " factor:",
    np.exp(py["x0"][:D]) / np.exp(ml["x0"][:D]),
)
print(
    "    py x0 inside [LB,UB]? ",
    np.all((py["x0"] >= py["LB"]) & (py["x0"] <= py["UB"])),
    " inside [PLB,PUB]? ",
    np.all((py["x0"] >= py["PLB"]) & (py["x0"] <= py["PUB"])),
)

print("\n=== B. NegativeQuadratic mean recommendations ===")
py = gpr.mean_functions.NegativeQuadratic().get_bounds_info(X, y)
ml = M.meanfun_info(X, y, 4)
show("NegativeQuadratic vs gplite_meanfun case 4", py, ml)
print("\n=== B2. ConstantMean recommendations (the quantile convention) ===")
pyc = gpr.mean_functions.ConstantMean().get_bounds_info(X, y)
mlc = M.meanfun_info(X, y, 1)
show("ConstantMean vs gplite_meanfun case 1", pyc, mlc)
print(
    "    np.quantile(y,0.1) =",
    float(np.quantile(y, 0.1)),
    " quantile1(y,0.1) =",
    M.quantile1(y, 0.1),
    " difference =",
    float(np.quantile(y, 0.1)) - M.quantile1(y, 0.1),
)
print(
    "    np.quantile(y,0.9) =",
    float(np.quantile(y, 0.9)),
    " quantile1(y,0.9) =",
    M.quantile1(y, 0.9),
    " difference =",
    float(np.quantile(y, 0.9)) - M.quantile1(y, 0.9),
)

print("\n=== C. noise recommendations, four parameterizations ===")
for params in [(1, 0, 0), (1, 1, 0), (1, 2, 0), (1, 0, 1)]:
    nf = gpr.noise_functions.GaussianNoise(
        constant_add=params[0] == 1,
        user_provided_add=params[1] > 0,
        scale_user_provided=params[1] == 2,
        rectified_linear_output_dependent_add=params[2] == 1,
    )
    py = nf.get_bounds_info(X, y)
    ml = M.noisefun_info(X, params, y)
    ok = all(
        np.allclose(py[k], ml[k], rtol=1e-12, atol=1e-12, equal_nan=True)
        for k in ("LB", "UB", "PLB", "PUB", "x0")
    )
    print(f"  {params}: {'EQUAL' if ok else 'DIFFERS'}")
    if not ok:
        show(str(params), py, ml)

print("\n=== D. RationalQuadraticARD: the PUB index slip ===")
py = gpr.covariance_functions.RationalQuadraticARD().get_bounds_info(X, y)
print(
    "  cov_N =",
    gpr.covariance_functions.RationalQuadraticARD().hyperparameter_count(D),
)
print(
    "  hyperparameter_info:",
    gpr.covariance_functions.RationalQuadraticARD().hyperparameter_info(D),
)
for k in ("LB", "PLB", "PUB", "UB", "x0"):
    print(f"    {k:4s} {np.asarray(py[k], dtype=float)}")
h = np.max(y) - np.min(y)
print(
    "  log(height) =",
    float(np.log(h)),
    "  (should be PUB[D]); PUB[D] =",
    float(py["PUB"][D]),
    "  PUB[-1] =",
    float(py["PUB"][-1]),
)
print("  after fit's PUB = max(min(PUB,UB),LB):")
LB, UB = py["LB"].copy(), py["UB"].copy()
PUB = np.maximum(np.minimum(py["PUB"], UB), LB)
PLB = np.minimum(np.maximum(py["PLB"], LB), UB)
print("    PLB =", PLB, "\n    PUB =", PUB)

print("\n=== E. isotropic recommendations vs gplite_covfun isoflag ===")
pyi = iso.SquaredExponentialIsotropic().get_bounds_info(X, y)
mli = M.covfun_info(X, y, isoflag=True)
show("SquaredExponentialIsotropic vs gplite_covfun isoflag", pyi, mli)
w = np.max(X, 0) - np.min(X, 0)
print("    widths:", w)
print(
    "    log(mean(w)) =",
    float(np.log(np.mean(w))),
    "  mean(log(w)) =",
    float(np.mean(np.log(w))),
    "  log(min(w)) =",
    float(np.log(np.min(w))),
    "  log(max(w)) =",
    float(np.log(np.max(w))),
)
print(
    "    np.min(np.mean(w)) == np.max(np.mean(w)):",
    np.min(np.mean(w)) == np.max(np.mean(w)),
)

print("\n=== F. the single-training-point edge ===")
for Dn in (1, 3):
    X1 = np.array([[0.7, 2.0, -5.0][:Dn]])
    y1 = np.array([[1.25]])
    print(f"\n  --- N = 1, D = {Dn} ---")
    with np.errstate(all="ignore"):
        pyc = gpr.covariance_functions.SquaredExponential().get_bounds_info(
            X1, y1
        )
        mlc = M.covfun_info(X1, y1, isoflag=False)
        pym = gpr.mean_functions.NegativeQuadratic().get_bounds_info(X1, y1)
        mlm = M.meanfun_info(X1, y1, 4)
    print(
        "   covariance: MATLAB width (row reduction) =",
        M._mmax(X1) - M._mmin(X1),
        " python per-column width =",
        np.max(X1, 0) - np.min(X1, 0),
    )
    print(
        "   MATLAB std(X) =",
        M._mstd(X1),
        " np.std(X, ddof=1) =",
        np.std(X1, ddof=1),
    )
    show("covariance", pyc, mlc)
    show("negative-quadratic mean", pym, mlm)

print("\n=== G. degenerate training sets (constant y) ===")
Xg = X.copy()
yg = np.full((N, 1), 3.0)
with np.errstate(all="ignore"):
    pyn = gpr.noise_functions.GaussianNoise(constant_add=True).get_bounds_info(
        Xg, yg
    )
    mln = M.noisefun_info(Xg, (1, 0, 0), yg)
    pyc = gpr.covariance_functions.SquaredExponential().get_bounds_info(Xg, yg)
    mlc = M.covfun_info(Xg, yg, isoflag=False)
print("  noise  py LB/UB:", pyn["LB"], pyn["UB"], " ml:", mln["LB"], mln["UB"])
print(
    "  cov    py LB/UB (output scale):",
    pyc["LB"][D],
    pyc["UB"][D],
    " ml:",
    mlc["LB"][D],
    mlc["UB"][D],
)
print("  cov    py x0 (output scale):", pyc["x0"][D], " ml:", mlc["x0"][D])
print(
    "  (MATLAB's own guard `if numel(y) <= 1; y = [0;1]` does not fire:",
    "numel(y) =",
    yg.size,
    ")",
)
