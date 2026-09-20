"""Settles P8i F1, F9 and F10 (and the P8c F5 half-bounded half of F10).

F1  - a transformer built with ``scale``/``rotation_matrix`` and plausible
      bounds tighter than the hard bounds derives ``mu``/``delta`` in the
      rotated-and-scaled coordinates but applies them before the rotation,
      so the plausible box no longer maps to [-0.5, 0.5].
F9  - ``scale`` is not validated; ``log_abs_det_jacobian`` takes log(scale),
      which is NaN for a negative entry and -inf for a zero entry.
F10 - the constructor stores the caller's arrays by reference; the class
      docstring names a keyword ``bounded_transform_type`` that the
      signature does not have; the constructor default is "logit" while
      the package option default is "probit"; a half-bounded variable is
      taken as unbounded and the inverse returns points outside the
      declared support.
"""

import numpy as np

import pyvbmc
from pyvbmc.parameter_transformer import ParameterTransformer

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

# ---------------------------------------------------------------- F1
print("=== F1: mu/delta derived after the rotation, applied before it ===")
theta = np.deg2rad(40.0)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
scale = np.array([2.0, 0.5])
lb = np.array([[-10.0, -10.0]])
ub = np.array([[10.0, 10.0]])
plb = np.array([[-1.0, -3.0]])
pub = np.array([[1.0, 5.0]])

pt_warp = ParameterTransformer(
    2, lb, ub, plb, pub, scale=scale, rotation_matrix=R
)
pt_plain = ParameterTransformer(2, lb, ub, plb, pub)
print(
    "with scale+rotation : pt(plb) =", pt_warp(plb), " pt(pub) =", pt_warp(pub)
)
print(
    "without             : pt(plb) =",
    pt_plain(plb),
    " pt(pub) =",
    pt_plain(pub),
)
print("mu =", pt_warp.mu, " delta =", pt_warp.delta)

# scale alone is enough to break it
pt_s = ParameterTransformer(2, lb, ub, plb, pub, scale=np.array([2.0, 2.0]))
print("scale only          : pt(plb) =", pt_s(plb), " pt(pub) =", pt_s(pub))

# rotation alone with an unbounded problem
lbi = np.array([[-np.inf, -np.inf]])
ubi = np.array([[np.inf, np.inf]])
pt_r = ParameterTransformer(2, lbi, ubi, plb, pub, rotation_matrix=R)
print("rotation only, unbd : pt(plb) =", pt_r(plb), " pt(pub) =", pt_r(pub))

# round trip still exact?
pts = np.array([[0.3, -1.2], [2.0, 4.0]])
print(
    "round-trip max |err| =",
    np.max(np.abs(pt_warp.inverse(pt_warp(pts)) - pts)),
)
print()

# ---------------------------------------------------------------- F9
print("=== F9: scale is unvalidated; log(scale) not log|scale| ===")
for s in ([1.0, -2.0], [1.0, 0.0], [1.0, 2.0]):
    with np.errstate(all="ignore"):
        pt = ParameterTransformer(2, scale=np.array(s, dtype=float))
        u = pt(np.array([[0.5, 0.5]]))
        ld = pt.log_abs_det_jacobian(np.array([[0.5, 0.5]]))
        back = pt.inverse(u)
    print(f"  scale={s}: log_abs_det={ld}, forward={u}, inverse={back}")
# a reflection in R_mat is handled correctly
Rref = np.array([[1.0, 0.0], [0.0, -1.0]])
pt = ParameterTransformer(2, rotation_matrix=Rref)
print(
    "  reflection in R_mat: log_abs_det =",
    pt.log_abs_det_jacobian(np.array([[0.5, 0.5]])),
    " det(R) =",
    np.linalg.det(Rref),
)
print()

# ---------------------------------------------------------------- F10a
print("=== F10a: constructor keeps references to the caller's arrays ===")
lb2 = np.array([[-5.0, -5.0]])
ub2 = np.array([[5.0, 5.0]])
sc = np.array([1.0, 1.0])
pt = ParameterTransformer(2, lb2, ub2)
print(
    "  pt.lb_orig is lb2:",
    pt.lb_orig is lb2,
    " pt.ub_orig is ub2:",
    pt.ub_orig is ub2,
)
before = pt(np.array([[0.5, 0.5]])).copy()
lb2[0, 0] = -100.0
after = pt(np.array([[0.5, 0.5]]))
print("  pt([[0.5,0.5]]) before mutating lb2:", before)
print("  pt([[0.5,0.5]]) after  mutating lb2:", after)
pt2 = ParameterTransformer(2, scale=sc)
print("  pt.scale is the caller's array:", pt2.scale is sc)
import copy as _copy

pt3 = _copy.deepcopy(pt2)
print("  deepcopy detaches scale:", pt3.scale is not sc)
print()

# ---------------------------------------------------------------- F10b
print("=== F10b: docstring keyword vs signature; constructor default ===")
import inspect

sig = inspect.signature(ParameterTransformer.__init__)
print("  signature parameters:", list(sig.parameters))
print(
    "  'bounded_transform_type' in docstring:",
    "bounded_transform_type" in ParameterTransformer.__doc__,
)
try:
    ParameterTransformer(1, bounded_transform_type="probit")
except TypeError as exc:
    print("  documented keyword raises:", type(exc).__name__, exc)
print(
    "  constructor default transform_type =",
    sig.parameters["transform_type"].default,
    "-> bounded type",
    ParameterTransformer(1, np.array([[0.0]]), np.array([[1.0]])).type,
)
print()

# ---------------------------------------------------------------- F10c/P8c F5
print("=== F10c: half-bounded variable taken as unbounded ===")
pt = ParameterTransformer(
    1,
    np.array([[0.0]]),
    np.array([[np.inf]]),
    np.array([[1.0]]),
    np.array([[3.0]]),
)
print("  type =", pt.type, " mu =", pt.mu, " delta =", pt.delta)
u = np.array([[-2.0], [0.0], [2.0]])
print("  inverse([-2, 0, 2]) =", pt.inverse(u).ravel(), "  (lb_orig = 0)")
print("  log_abs_det_jacobian =", pt.log_abs_det_jacobian(u))
print(
    "  MATLAB type 1 would give exp(y)+a =",
    np.exp(u.ravel()) + 0.0,
    " and log|J| = y =",
    u.ravel(),
)
print()

print("=== VBMC rejects half-bounds ===")
try:
    pyvbmc.VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.array([[1.0, 1.0]]),
        np.array([[0.0, 0.0]]),
        np.array([[np.inf, np.inf]]),
        np.array([[0.5, 0.5]]),
        np.array([[2.0, 2.0]]),
    )
except ValueError as exc:
    print("  VBMC(...) raises:", str(exc).splitlines()[0])
