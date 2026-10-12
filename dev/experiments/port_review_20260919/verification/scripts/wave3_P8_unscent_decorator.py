"""Settles P8i F2, F3 (``unscent_warp``) and P8i F12 (the shape decorator).

F2  - ``unscent_warp`` tiles ``x`` without casting, so an integer ``x``
      truncates the sigma points.  MATLAB's ``utils/unscent_warp.m`` builds
      ``xx`` from ``repmat`` of a double and ``bsxfun(@plus, ...)``, so its
      arithmetic is double whatever the caller passes.
F3  - the single-row-``x``/many-row-``sigma`` branch: PyVBMC reshapes the
      results back to ``x.shape`` captured before ``atleast_2d``, i.e.
      ``(1, D)``, and raises.  MATLAB reshapes by ``N > 1`` instead
      (``unscent_warp.m:29-35``).
F12 - the decorator: a 0-D input is not unwrapped; ``input_dims`` can be
      unbound; one variable is shared by several patched arguments.
"""

import numpy as np

import pyvbmc
from pyvbmc.decorators import handle_0D_1D_input
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.whitening import unscent_warp

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

identity = lambda x: x

print("=== F2: integer x truncates the sigma points ===")
xi = np.array([[1, 2], [3, 4]])
sig = np.array([0.25, 0.25])
m, s, __ = unscent_warp(identity, xi, sig)
print("  integer x:", xi.dtype, "-> mean", m.ravel(), " sigma", s.ravel())
xf = xi.astype(float)
m, s, __ = unscent_warp(identity, xf, sig)
print("  float   x:", xf.dtype, "-> mean", m.ravel(), " sigma", s.ravel())
print("  the exact answer is mean = x, sigma = 0.25 (the identity map)")
print("  in-tree callers (whitening.py:324, :350, :376) pass float64 only")
print()

print("=== F3: x with one row and sigma with many ===")
x1 = np.array([[0.0, 0.0]])
sigN = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
try:
    out = unscent_warp(identity, x1, sigN)
    print("  returned shapes:", [np.shape(o) for o in out])
except Exception as exc:
    print("  raises:", type(exc).__name__, exc)
print("  the mirror branch (one row of sigma, many of x) works:")
out = unscent_warp(identity, sigN, x1)
print("   shapes:", [np.shape(o) for o in out])
print("  MATLAB reshapes on N > 1, so it returns the (N, D) result")
print()

print("=== F12a: a 0-D input is not unwrapped ===")
pt = ParameterTransformer(1, np.array([[0.0]]), np.array([[1.0]]))
r0 = pt.log_abs_det_jacobian(np.float64(0.3))
r1 = pt.log_abs_det_jacobian(np.array([0.3]))
r2 = pt.log_abs_det_jacobian(np.array([[0.3]]))
print("  0-D input ->", type(r0).__name__, np.shape(r0), r0)
print("  1-D input ->", type(r1).__name__, np.shape(r1), r1)
print("  2-D input ->", type(r2).__name__, np.shape(r2), r2)
print(
    "  same for __call__:",
    np.shape(pt(np.float64(0.3))),
    np.shape(pt(np.array([0.3]))),
)
print()

print("=== F12b: input_dims unbound when the patched argument is absent ===")


class Toy:
    @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
    def f(self, x=None):
        return np.zeros((1, 2))


try:
    Toy().f()
except Exception as exc:
    print(
        "  a decorated method with a default raises:", type(exc).__name__, exc
    )
try:
    ParameterTransformer(1).log_abs_det_jacobian()
except Exception as exc:
    print(
        "  in-tree methods require the argument, so they raise first:",
        type(exc).__name__,
        exc,
    )
print()

print("=== F12c: one input_dims shared by several patched arguments ===")


class Toy2:
    @handle_0D_1D_input(patched_kwargs=["a", "b"], patched_argpos=[0, 1])
    def g(self, a, b):
        return a + b


print(
    "  g(1-D, 2-D) ->",
    np.shape(Toy2().g(np.array([1.0, 2.0]), np.array([[3.0, 4.0]]))),
    " (the last patched argument decides)",
)
print(
    "  g(2-D, 1-D) ->",
    np.shape(Toy2().g(np.array([[1.0, 2.0]]), np.array([3.0, 4.0]))),
)
print()

print("=== the decorated methods in the package ===")
import subprocess

out = subprocess.run(
    [
        "git",
        "grep",
        "-n",
        "-A1",
        "handle_0D_1D_input(",
        "--",
        "pyvbmc/",
        ":!pyvbmc/testing/",
        ":!pyvbmc/decorators/",
    ],
    cwd=r"C:\Users\luigi\Documents\GitHub\pyvbmc",
    capture_output=True,
    text=True,
)
print(out.stdout)
