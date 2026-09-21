"""Settles the claim that the degree-1 Matern length-scale gradient is NaN on
the diagonal in gpyreg and in gplite alike (covariance_functions.py:221, :289;
isotropic_covariance_functions.py:156; gplite_covfun.m:198, :218-219), and
that the NaN propagates into the marginal-likelihood gradient so that
`GP.fit` cannot optimize a Matern-1 kernel.

Also checks degrees 3 and 5, and the effect of the fix MATLAB carries as a
commented-out line (`dK = dK .* (Ki > 1e-12)`).

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python G2_6_matern1_grad.py
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
np.set_printoptions(precision=6, suppress=False, linewidth=170)

D = 1
rng = np.random.default_rng(9)
X = np.sort(rng.uniform(-2, 2, size=(15, 1)), axis=0)
y = np.sin(2 * X) + 0.05 * rng.standard_normal((15, 1))
hyp_cov = np.array([np.log(0.8), np.log(1.1)])

print("\n=== A. the kernel gradient on the diagonal ===")
for deg in (1, 3, 5):
    k = gpr.covariance_functions.Matern(deg)
    with np.errstate(all="ignore"):
        K, dK = k.compute(hyp_cov, X, compute_grad=True)
        Km, dKm = M.covfun_matern(hyp_cov, X, deg, compute_grad=True)
    d_py = np.diag(dK[:, :, 0])
    d_ml = np.diag(dKm[:, :, 0])
    print(f"  degree {deg}: gpyreg diag(dK/dlog ell)[:4] = {d_py[:4]}")
    print(f"             gplite diag(dK/dlog ell)[:4] = {d_ml[:4]}")
    print(
        f"             NaNs: gpyreg {int(np.sum(np.isnan(dK)))},"
        f" gplite {int(np.sum(np.isnan(dKm)))}"
        f"   df(0) = {k.df(0.0) if deg != 1 else 'inf'}"
    )
    print(
        f"             off-diagonal agreement max|dK - dKm| ="
        f" {np.nanmax(np.abs(dK - dKm)):.3e}"
    )

print("\n=== B. the marginal-likelihood gradient ===")
for deg in (1, 3, 5):
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(deg),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp.update(X_new=X, y_new=y, compute_posterior=False)
    gp.set_bounds(None)
    gp.set_priors(None)
    h = np.array(
        [np.log(0.8), np.log(1.1), np.log(0.1), 0.2, 0.0, np.log(1.5)]
    )
    with np.errstate(all="ignore"):
        nlZ, dnlZ = gp.log_likelihood(h, compute_grad=True)
    print(f"  degree {deg}: nlZ = {nlZ:.6f}  dnlZ = {dnlZ}")

print("\n=== C. the isotropic Matern ===")
for deg in (1, 3):
    k = iso.MaternIsotropic(deg)
    with np.errstate(all="ignore"):
        K, dK = k.compute(hyp_cov, X, compute_grad=True)
    print(
        f"  degree {deg}: NaNs in dK = {int(np.sum(np.isnan(dK)))}"
        f"  diag(dK[...,0])[:3] = {np.diag(dK[:, :, 0])[:3]}"
    )

print("\n=== D. MATLAB's commented-out fix ===")
k = gpr.covariance_functions.Matern(1)
with np.errstate(all="ignore"):
    K, dK = k.compute(hyp_cov, X, compute_grad=True)
ell = np.exp(hyp_cov[0])
from scipy.spatial.distance import pdist, squareform

Ki = squareform(pdist((1.0 / ell * X[:, 0]).reshape(-1, 1), "sqeuclidean"))
fixed = np.where(Ki > 1e-12, np.nan_to_num(dK[:, :, 0]), 0.0)
print(
    "  with `dK = dK .* (Ki > 1e-12)` the diagonal becomes",
    np.diag(fixed)[:4],
    " NaNs:",
    int(np.sum(np.isnan(fixed))),
)
print(
    "  the true derivative at coincident inputs is 0 (K = sf2 there,",
    "independent of the length scale)",
)
print("\n=== E. would a fit reach it? ===")
print("  gpyreg's own tests instantiate Matern with degree:", end=" ")

print("(see G2_7_tests.py)")
