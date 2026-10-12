"""P9-6. Do the documented shapes and parameter names match the code?

Settles: the actual shapes returned by `Prior.log_pdf`/`pdf` under both
values of `keepdims`, against the `keepdims` description; the stored shapes
of `a`, `u`, `v`, `b`, `scale` against the `Attributes` blocks; and the
duplicated `sample_prior` name in `convert_to_prior`'s docstring.
"""

import numpy as np
from scipy.stats import norm

import pyvbmc
from pyvbmc.priors import (
    Prior,
    Product,
    SciPy,
    SmoothBox,
    SplineTrapezoidal,
    Trapezoidal,
    UniformBox,
    convert_to_prior,
)

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

print("--- documented `keepdims` text (prior.py) ---")
for meth in (Prior.log_pdf, Prior.pdf):
    doc = meth.__doc__
    i = doc.find("keepdims : bool")
    print(f"{meth.__name__}:")
    print("   ", " ".join(doc[i : i + 200].split())[:190])
print()

print("--- actual shapes ---")
p = UniformBox(np.zeros(3), np.ones(3))
for n in (1, 5):
    x = np.full((n, 3), 0.5)
    print(
        f"n={n}: keepdims=True  -> {p.log_pdf(x, keepdims=True).shape}, "
        f"keepdims=False -> {p.log_pdf(x, keepdims=False).shape}, "
        f"pdf -> {p.pdf(x, keepdims=True).shape} / "
        f"{p.pdf(x, keepdims=False).shape}"
    )
x1 = np.full(3, 0.5)
print(
    f"x of shape (D,): keepdims=True -> "
    f"{p.log_pdf(x1, keepdims=True).shape}, False -> "
    f"{p.log_pdf(x1, keepdims=False).shape}"
)
print()

print("--- stored attribute shapes against the documented `(1, D)` ---")
objs = {
    "UniformBox": UniformBox(np.zeros((1, 3)), np.ones((1, 3))),
    "Trapezoidal": Trapezoidal(
        np.zeros(3), np.full(3, 0.25), np.full(3, 0.75), np.ones(3)
    ),
    "SplineTrapezoidal": SplineTrapezoidal(
        np.zeros(3), np.full(3, 0.25), np.full(3, 0.75), np.ones(3)
    ),
    "SmoothBox": SmoothBox(np.zeros(3), np.ones(3), np.ones(3)),
    "SciPy(norm())": SciPy(norm()),
    "Product": Product([norm(), norm(), norm()]),
}
for name, o in objs.items():
    shapes = {
        k: getattr(o, k).shape
        for k in ("a", "u", "v", "b", "scale")
        if hasattr(o, k) and isinstance(getattr(o, k), np.ndarray)
    }
    doc = type(o).__doc__ or ""
    documented = "shape `(1, D)`" in doc
    print(
        f"{name:20s} stored {shapes}   docstring says (1, D): " f"{documented}"
    )
print()

print("--- Attributes blocks that say (1, D) ---")
for cls in (UniformBox, Trapezoidal, SplineTrapezoidal, SmoothBox):
    lines = [
        " ".join(l.split())
        for l in (cls.__doc__ or "").splitlines()
        if "(1, D)" in l
    ]
    print(f"{cls.__name__}: {lines}")
print()

print("--- convert_to_prior docstring parameter names ---")
doc = convert_to_prior.__doc__
for line in doc.splitlines():
    if ": callable" in line or line.strip().startswith(("prior,", "D : int")):
        print("   ", line.strip())
print(
    "   occurrences of 'sample_prior :' =",
    doc.count("sample_prior : callable"),
)
print(
    "   occurrences of 'log_prior :'    =", doc.count("log_prior : callable")
)
print()

print("--- the spline-trapezoid normalizer comment ---")
import inspect

src = inspect.getsource(SplineTrapezoidal._log_pdf)
for line in src.splitlines():
    if "norm_factor" in line:
        print("   ", line.strip())
v, u, b, a = 0.75, 0.25, 1.0, 0.0
print(
    f"   comment expression  u - v + 0.5*(b - v + u - a) = "
    f"{u - v + 0.5 * (b - v + u - a)}"
)
print(
    f"   MATLAB's comment    v - u + 0.5*(b - v + u - a) = "
    f"{v - u + 0.5 * (b - v + u - a)}"
)
print(
    f"   the code            0.5*(v - u + b - a)         = "
    f"{0.5 * (v - u + b - a)}"
)
