"""P4-4: ``active_sample_proposal_pdf`` when every mixture component is zero.

MATLAB's ``activesample_proposalpdf`` has no check: with every
``templpdf`` entry ``-Inf`` it computes ``mmax = -Inf``, ``templpdf -
mmax`` is ``NaN``, and the caller's line 148
(``lnw(~isfinite(lnw)) = -Inf``) turns the point into a zero weight.
PyVBMC raises ``ValueError("Invalid value.")`` at
``active_importance_sampling.py:396-398`` and the run ends.

This script (i) reproduces the raise, (ii) shows how far outside the
training boxes and how flat the smoothed variational posterior a point
must be for it, and (iii) checks whether the shipped path can reach it
by sampling from the smoothed posterior the module itself builds.
"""

import copy

import numpy as np
from wave4_P4_common import banner, load

banner()

from pyvbmc.acquisition_functions import AcqFcnIMIQR  # noqa: E402
from pyvbmc.vbmc.active_importance_sampling import (  # noqa: E402
    active_sample_proposal_pdf,
)

st = load("normal_D2_singlesample", seed=2)
vp, gp = st["vp"], st["gp"]
acq = AcqFcnIMIQR()
rect_delta = 2 * np.std(gp.X, ddof=1, axis=0)
w_vp = 0.5

print("gp.X box:", gp.X.min(0), gp.X.max(0))
print("rect_delta (box half-widths):", rect_delta)
print("vp.sigma:", vp.sigma.ravel(), " vp.lambd:", vp.lambd.ravel())

# The smoothed copy the module builds (scale_vec = [0.05, 0.2, 1.0])
vp_is = copy.deepcopy(vp)
for sc in (0.05, 0.2, 1.0):
    vp_is.K = vp_is.K + vp.K
    vp_is.w = np.hstack((vp_is.w, vp.w))
    vp_is.mu = np.hstack((vp_is.mu, vp.mu))
    vp_is.sigma = np.hstack((vp_is.sigma, np.sqrt(vp.sigma**2 + sc**2)))
vp_is.w = vp_is.w / np.sum(vp_is.w)

# How far out must a point be for the smoothed VP's log density to
# underflow to -inf, and to leave every training box?
centre = vp.mu.mean(1)
direction = np.array([1.0, 0.0])
print()
print("  distance  log vp_is pdf   in any box?")
for r in (5, 20, 50, 100, 200, 400, 800, 2000, 1e4, 1e5):
    x = (centre + r * direction).reshape(1, -1)
    lp = float(np.ravel(vp_is.pdf(x, orig_flag=False, log_flag=True))[0])
    inbox = bool(np.any(np.all(np.abs(x - gp.X) < rect_delta, axis=1)))
    print(f"  {r:>9.4g}  {lp:>13.4g}   {inbox}")

# the raise
far = (centre + 1e5 * direction).reshape(1, -1)
try:
    active_sample_proposal_pdf(far, gp, vp_is, w_vp, rect_delta, acq)
    print("\n  no raise at distance 1e5")
except Exception as exc:
    print(f"\n  distance 1e5 -> {type(exc).__name__}: {exc}")

# MATLAB's arithmetic on the same input
templpdf = np.full((1, 1 + gp.X.shape[0]), -np.inf)
with np.errstate(invalid="ignore"):
    mmax = np.amax(templpdf, axis=1)
    lpdf = np.log(np.sum(np.exp(templpdf - mmax.reshape(-1, 1)), axis=1))
    lnw = 0.0 - (lpdf + mmax)
print(
    "  MATLAB arithmetic on an all -inf row: mmax =",
    mmax,
    " lpdf =",
    lpdf,
    " lnw =",
    lnw,
    " -> caller line 148 sets it to -inf:",
    not np.isfinite(lnw).all(),
)

# can the shipped draw reach it?
vp_is.rng = np.random.default_rng(0)
Xa, _ = vp_is.sample(20000, orig_flag=False)
lp = vp_is.pdf(Xa, orig_flag=False, log_flag=True).ravel()
print()
print(
    "  20000 draws from the smoothed VP: min log density =",
    lp.min(),
    " any -inf:",
    np.any(~np.isfinite(lp)),
)
inbox = np.array(
    [np.any(np.all(np.abs(x - gp.X) < rect_delta, axis=1)) for x in Xa]
)
print("  fraction of draws inside no training box:", 1 - inbox.mean())
print(
    "  min log density among those:",
    lp[~inbox].min() if np.any(~inbox) else "n/a",
)
