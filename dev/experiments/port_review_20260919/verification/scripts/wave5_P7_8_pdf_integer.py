"""P7-8: vp.pdf(orig_flag=True) on an integer array truncates the coordinates.

Settles whether the x.copy() at variational_posterior.py:787 keeps the
caller's integer dtype, so that the float coordinates the transformer
returns at :801 are cast back to integers, and what the
handle_0D_1D_input decorator does to the dtype of a 1-D integer input.
"""

import numpy as np

import pyvbmc
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc.__file__ =", pyvbmc.__file__)

D, K = 2, 2
lb = np.array([[0.0, 0.0]])
ub = np.array([[10.0, 10.0]])
pt = ParameterTransformer(D, lb, ub)
vp = VariationalPosterior(
    D,
    K,
    x0=np.zeros((1, D)),
    parameter_transformer=pt,
    rng=np.random.default_rng(9),
)
vp.mu = np.zeros((D, K))
vp.sigma = np.full((1, K), 1.0)
vp.lambd = np.ones((D, 1))
vp.w = np.ones((1, K)) / K

xi = np.array([[3, 4]])
xf = np.array([[3.0, 4.0]])
print("  integer input dtype:", xi.dtype, " float input dtype:", xf.dtype)
print("  pdf(int)   =", vp.pdf(xi))
print("  pdf(float) =", vp.pdf(xf))
print("  ratio      =", (vp.pdf(xi) / vp.pdf(xf)).ravel())
print(
    "  transformed [3,4] ->",
    pt(xf),
    " truncated to int ->",
    pt(xf).astype(np.int64),
)
print(
    "  pdf at the truncated transformed point, orig_flag=False:",
    vp.pdf(pt(xf).astype(np.int64).astype(float), orig_flag=False),
)

print("\n  orig_flag=False with an integer input (no assignment happens):")
print("   pdf(int, orig_flag=False)   =", vp.pdf(xi, orig_flag=False))
print("   pdf(float, orig_flag=False) =", vp.pdf(xf, orig_flag=False))

print("\n== what handle_0D_1D_input does to a 1-D integer input ==")
x1d_int = np.array([3, 4])
print("   pdf(np.array([3,4]))   =", vp.pdf(x1d_int))
print("   pdf(np.array([3.,4.])) =", vp.pdf(np.array([3.0, 4.0])))
print(
    "   (the decorator calls np.atleast_2d, which preserves the dtype:",
    np.atleast_2d(x1d_int).dtype,
    ")",
)

print("\n== a Python list of ints ==")
try:
    print("   pdf([[3,4]]) =", vp.pdf([[3, 4]]))
except Exception as e:
    print(f"   pdf([[3,4]]) raises {type(e).__name__}: {e}")

print("\n== log_flag: same truncation? ==")
print("   log_pdf(int)   =", vp.pdf(xi, log_flag=True))
print("   log_pdf(float) =", vp.pdf(xf, log_flag=True))

print("\n== unbounded transform (identity): is the density still wrong? ==")
pt2 = ParameterTransformer(D)
vp2 = VariationalPosterior(
    D,
    K,
    x0=np.zeros((1, D)),
    parameter_transformer=pt2,
    rng=np.random.default_rng(9),
)
vp2.mu = np.zeros((D, K))
vp2.sigma = np.full((1, K), 1.0)
vp2.lambd = np.ones((D, 1))
vp2.w = np.ones((1, K)) / K
print(
    "   identity transform, pdf(int) =",
    vp2.pdf(np.array([[1, 2]])),
    " pdf(float) =",
    vp2.pdf(np.array([[1.0, 2.0]])),
)
