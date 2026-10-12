"""P4-12 (and the MATLAB half of P4-7): what the stored MATLAB fixture
``compare_MATLAB/log_isbasefun.npz`` pins.

MATLAB's ``log_isbasefun`` (``activeimportancesampling_vbmc.m:348``)
takes the *first two* outputs of ``gplite_pred``, which are ``ymu`` and
``ys2`` -- noise-inclusive -- not ``fmu``/``fs2``.  This script
recomputes ``is_log_full`` on the test's own scenario with and without
the observation noise and compares each with the stored MATLAB array, so
that the question "does MATLAB add the constant noise term here?" is
settled against a MATLAB number rather than by reading alone.

It also reproduces the ``test_acq_log_f`` quirk: the values computed at
lines 352-356 are overwritten at 357-363.
"""

import os.path
from sys import float_info

import gpyreg as gpr
import numpy as np
import scipy.stats as sps
from wave4_P4_common import banner

import pyvbmc

banner()

from pyvbmc.acquisition_functions import AcqFcnIMIQR, AcqFcnVIQR  # noqa: E402
from pyvbmc.variational_posterior import VariationalPosterior  # noqa: E402

D, K = 3, 2
vp = VariationalPosterior(D=D, K=K)
vp.mu = np.array([[-1.5, -1.0, -0.5], [0.0, 1.0, 2.0]]).T
vp.w = np.array([[0.7, 0.3]])
vp.sigma = np.ones(vp.sigma.shape)
vp.lambd = np.ones(vp.lambd.shape)
X = np.arange(-7, 8).reshape((5, 3), order="F")
y = np.array(
    [sps.multivariate_normal.logpdf(x, mean=np.zeros(D)) for x in X]
).reshape((-1, 1))
hyp = np.array(
    [
        -2.0,
        -3.0,
        -4.0,
        1.0,
        0.0,
        -(D / 2) * np.log(2 * np.pi),
        0.0,
        0.25,
        0.5,
        -0.5,
        0.0,
        0.5,
    ]
)
hyp = np.vstack([hyp, 2 * hyp])
gp = gpr.GP(
    D,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.NegativeQuadratic(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
gp.update(X_new=X, y_new=y, hyp=hyp)
Xa = 2 * np.arange(-4, 5).reshape((3, 3), order="F") / np.pi

fixture = os.path.join(
    os.path.dirname(pyvbmc.__file__),
    "testing",
    "vbmc",
    "compare_MATLAB",
    "log_isbasefun.npz",
)
M = np.load(fixture, allow_pickle=False)
print("fixture keys:", list(M.keys()))

print(
    "constant noise variance of each hyperparameter sample:",
    [
        float(np.exp(2 * gp.posteriors[s].hyp[4]))
        for s in range(len(gp.posteriors))
    ],
)

imiqr, viqr = AcqFcnIMIQR(), AcqFcnVIQR()

# noise-inclusive (what the code does)
y_imiqr_noise = imiqr.is_log_full(Xa, gp=gp, vp=vp)
y_viqr_noise = viqr.is_log_full(Xa, gp=gp, vp=vp) + np.maximum(
    vp.pdf(Xa, orig_flag=False, log_flag=True), np.log(float_info.min)
)

# latent only (what the code would do with add_noise=False)
f_mu, f_s2 = gp.predict(Xa)
y_imiqr_latent = imiqr.is_log_base(
    Xa, f_mu=f_mu, f_s2=f_s2
) + imiqr.is_log_added(f_mu=f_mu, f_s2=f_s2)
y_viqr_latent = viqr.is_log_added(f_s2=f_s2) + np.maximum(
    vp.pdf(Xa, orig_flag=False, log_flag=True), np.log(float_info.min)
)

for name, noisy, latent, ref in (
    ("imiqr", y_imiqr_noise, y_imiqr_latent, M["y_imiqr"]),
    ("viqr", y_viqr_noise, y_viqr_latent, M["y_viqr"]),
):
    print(f"  {name}: MATLAB {np.ravel(ref)}")
    print(
        f"        add_noise=True  {np.ravel(noisy)}  "
        f"max|d| = {np.abs(np.ravel(noisy) - np.ravel(ref)).max():.3e}"
    )
    print(
        f"        latent only     {np.ravel(latent)}  "
        f"max|d| = {np.abs(np.ravel(latent) - np.ravel(ref)).max():.3e}"
    )

print()
print("test_acq_log_f quirk: the flagged VIQR value at line 355 equals the")
v1 = AcqFcnVIQR()
v1.acq_info["importance_sampling_vp"] = True
flagged = v1.is_log_full(Xa, gp=gp, vp=vp)
plain = AcqFcnVIQR().is_log_full(Xa, gp=gp, vp=vp)
print(
    "  plain one (the flag changes nothing in PyVBMC):",
    np.array_equal(flagged, plain),
)
print("  so lines 352-356 are dead; the comparison is of the hand-assembled")
print("  is_log_added + vp log pdf built at lines 357-362.")
