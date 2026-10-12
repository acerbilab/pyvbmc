"""Reproduces the specific numbers the internal G2 report cites for its F5,
F6 and M9, to check its arithmetic, and compares the isotropic bound branch
with the MATLAB transcription so that the direction of the difference is on
record (a uniform upward shift of log(mean w) - mean(log w)).

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python G2_10_report_examples.py
Needs `matlab_gplite_g2.py` next to it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpyreg as gpr
import gpyreg.isotropic_covariance_functions as iso
import wave6_G2_matlab_gplite_g2 as M

np.set_printoptions(precision=4, suppress=True, linewidth=160)

# Internal F6's example: dimension 0 in [0,1], dimension 1 in [0,1000]
rng = np.random.default_rng(0)
N = 30
X = np.column_stack([rng.uniform(0, 1, N), rng.uniform(0, 1000, N)])
y = (rng.standard_normal(N)).reshape(-1, 1)
info = gpr.covariance_functions.SquaredExponential().get_bounds_info(X, y)
print(
    "per-dim std:",
    np.std(X, axis=0, ddof=1),
    " pooled std:",
    np.std(X, ddof=1),
)
print(
    "x0[0:2] =", info["x0"][0:2], " = log(pooled) =", np.log(np.std(X, ddof=1))
)
print(
    "LB =",
    info["LB"],
    "\nUB =",
    info["UB"],
    "\nPLB =",
    info["PLB"],
    "\nPUB =",
    info["PUB"],
)
print(
    "x0 inside [LB,UB]?",
    np.all((info["x0"] >= info["LB"]) & (info["x0"] <= info["UB"])),
)
print(
    "x0 inside [PLB,PUB]?",
    np.all((info["x0"] >= info["PLB"]) & (info["x0"] <= info["PUB"])),
)

# Internal F5's example: widths [0.912, 935.07]
X2 = np.array([[0.0, 0.0], [0.912, 935.07]])
X2 = np.vstack([X2, rng.uniform([0, 0], [0.912, 935.07], size=(20, 2))])
w = np.max(X2, 0) - np.min(X2, 0)
i2 = iso.SquaredExponentialIsotropic().get_bounds_info(X2, y[:22])
m2 = M.covfun_info(X2, y[:22], isoflag=True)
print("\nwidths:", w)
print(
    "gpyreg LB[0] =",
    i2["LB"][0],
    " log(mean)+log(tol) =",
    np.log(np.mean(w)) + np.log(1e-6),
    " log(min)+log(tol) =",
    np.log(np.min(w)) + np.log(1e-6),
)
print(
    "gpyreg UB[0] =",
    i2["UB"][0],
    " log(10*mean) =",
    np.log(10 * np.mean(w)),
    " log(10*max) =",
    np.log(10 * np.max(w)),
)
print(
    "MATLAB LB[0] =",
    m2["LB"][0],
    " = mean(log(w))+log(tol) =",
    np.mean(np.log(w)) + np.log(1e-6),
)
print(
    "MATLAB UB[0] =",
    m2["UB"][0],
    " = mean(log(10*w)) =",
    np.mean(np.log(10 * w)),
)
print(
    "gpyreg - MATLAB shift, all four bounds:",
    np.array([i2[k][0] - m2[k][0] for k in ("LB", "PLB", "PUB", "UB")]),
    " log(mean w) - mean(log w) =",
    np.log(np.mean(w)) - np.mean(np.log(w)),
)

# Internal M9: degenerate sets
print("\n--- M9 edges ---")
with np.errstate(all="ignore"):
    yc = np.full((30, 1), 2.0)
    nfo = gpr.noise_functions.GaussianNoise(constant_add=True).get_bounds_info(
        X, yc
    )
    print(
        "constant y: noise LB =",
        nfo["LB"],
        " UB =",
        nfo["UB"],
        " LB > UB:",
        nfo["LB"][0] > nfo["UB"][0],
    )
    mnfo = M.noisefun_info(X, (1, 0, 0), yc)
    print("            MATLAB   LB =", mnfo["LB"], " UB =", mnfo["UB"])
    cfo = gpr.covariance_functions.SquaredExponential().get_bounds_info(X, yc)
    print(
        "constant y: cov output-scale LB/UB/x0 =",
        cfo["LB"][2],
        cfo["UB"][2],
        cfo["x0"][2],
    )
    X1 = np.array([[0.7]])
    y1 = np.array([[1.25]])
    c1 = gpr.covariance_functions.SquaredExponential().get_bounds_info(X1, y1)
    print(
        "N=1 D=1: cov x0 =",
        c1["x0"],
        " (x0[0] came from the NaN fill 0.5*(PLB+PUB))",
    )
    print(
        "         np.std(X1, ddof=1) =",
        np.std(X1, ddof=1),
        " -> log is",
        np.log(np.std(X1, ddof=1)),
    )
    # `fit` then does UB = max(LB, UB); does that repair the inversion?
    gp = gpr.GP(
        D=2,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp.X, gp.y = X, yc
    b = gp.get_recommended_bounds("recommended", "recommended")
    print(
        "after get_recommended_bounds' `ub = max(lb, ub)`: noise bounds =",
        b["noise_log_scale"],
    )
    print("MATLAB gplite_train.m:141 does `UB = max(LB,UB)` as well")
