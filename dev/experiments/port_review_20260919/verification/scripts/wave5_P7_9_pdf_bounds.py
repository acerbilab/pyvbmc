"""P7-9: vp.pdf sets rows on or outside the original bounds to 0 / -inf.

Settles what PyVBMC returns on a bound, outside a bound and for a NaN
coordinate (variational_posterior.py:791-795, :913, :915), and reproduces
in NumPy what vbmc_pdf.m:36-39 with shared/warpvars_vbmc.m:104-110 would
compute on the same rows, MATLAB having no mask.
"""

import numpy as np

import pyvbmc
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc.__file__ =", pyvbmc.__file__)

D, K = 1, 2
lb = np.array([[0.0]])
ub = np.array([[1.0]])
pt = ParameterTransformer(D, lb, ub)
print("  transform type:", pt.type)
vp = VariationalPosterior(
    D,
    K,
    x0=np.zeros((1, D)),
    parameter_transformer=pt,
    rng=np.random.default_rng(4),
)
vp.mu = np.zeros((D, K))
vp.sigma = np.full((1, K), 1.0)
vp.lambd = np.ones((D, 1))
vp.w = np.ones((1, K)) / K

pts = np.array([[0.5], [0.0], [1.0], [-0.5], [1.5], [np.nan]])
labels = ["interior", "on lb", "on ub", "below lb", "above ub", "NaN"]
y = vp.pdf(pts)
ly = vp.pdf(pts, log_flag=True)
for lab, p, yy, ll in zip(labels, pts.ravel(), y.ravel(), ly.ravel()):
    print(f"  {lab:>9} x={p}: pdf={yy}  log_pdf={ll}")

print("\n== what MATLAB's arithmetic gives on the same rows ==")
# warpvars_vbmc.m 'dir' for a logit ('probit' uses norminv) bounded variable.
# PyVBMC's default bounded transform and MATLAB's differ (sheet entry); the
# point here is only the behavior of log(z/(1-z)) on and outside a bound.
for lab, x in zip(labels, pts.ravel()):
    z = (x - lb[0, 0]) / (ub[0, 0] - lb[0, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.log(z) - np.log1p(-z)
    print(
        f"  {lab:>9} x={x}: z={z}  log(z)-log1p(-z) = {u}"
        f"   (MATLAB: log of a negative number is complex)"
    )

print("\n== a NaN coordinate: which branch of the mask takes it ==")
xn = np.array([[np.nan]])
m_lo = np.all(xn > pt.lb_orig, axis=1)
m_hi = np.all(xn < pt.ub_orig, axis=1)
print(
    "  x > lb_orig ->",
    m_lo,
    "  x < ub_orig ->",
    m_hi,
    "  mask ->",
    np.logical_and(m_lo, m_hi),
)
print("  so a NaN row is treated as out of bounds and returns 0 / -inf")

print("\n== an unbounded transform: no mask fires ==")
pt2 = ParameterTransformer(D)
vp2 = VariationalPosterior(
    D,
    K,
    x0=np.zeros((1, D)),
    parameter_transformer=pt2,
    rng=np.random.default_rng(4),
)
vp2.mu = np.zeros((D, K))
vp2.sigma = np.full((1, K), 1.0)
vp2.lambd = np.ones((D, 1))
vp2.w = np.ones((1, K)) / K
print("  lb_orig/ub_orig =", pt2.lb_orig, pt2.ub_orig)
print("  pdf at NaN, unbounded:", vp2.pdf(np.array([[np.nan]])))
print("  pdf at 0.5, unbounded:", vp2.pdf(np.array([[0.5]])))
