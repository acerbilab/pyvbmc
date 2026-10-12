"""P9-2. Does `np.full_like(x, -np.inf)` inherit a non-float64 input dtype,
and what does that do to the log-density?

Settles: the returned dtype for int64 and float32 inputs in all six
families; the truncation of an in-support value; the two's-complement wrap
of the -inf fill for D = 1..4 out-of-support coordinates; the same through a
`Product` with a trapezoidal marginal; and the comparison with MATLAB's
`-inf(size(x))`, which is always double.
"""

import matlab_transcription as M
import numpy as np
from scipy.stats import norm

import pyvbmc
from pyvbmc.priors import (
    Product,
    SciPy,
    SmoothBox,
    SplineTrapezoidal,
    Trapezoidal,
    UniformBox,
)

print("pyvbmc.__file__ =", pyvbmc.__file__)
print("INT64_MIN =", np.iinfo(np.int64).min)
print()

fams = {
    "UniformBox(0,10,D=2)": UniformBox(0.0, 10.0, D=2),
    "Trapezoidal(0,2,8,10,D=2)": Trapezoidal(0.0, 2.0, 8.0, 10.0, D=2),
    "SplineTrapezoidal(0,2,8,10,D=2)": SplineTrapezoidal(
        0.0, 2.0, 8.0, 10.0, D=2
    ),
    "SmoothBox(0,10,1,D=2)": SmoothBox(0.0, 10.0, 1.0, D=2),
    "SciPy(norm())_D1": SciPy(norm()),
    "Product[norm,Trapezoidal]": Product(
        [norm(), Trapezoidal(0.0, 2.0, 8.0, 10.0)]
    ),
}

print("--- in-support point [3, 5] (D=2 families) ---")
pt64 = np.array([[3.0, 5.0]])
pti = np.array([[3, 5]])
pt32 = np.array([[3.0, 5.0]], dtype=np.float32)
for name, p in fams.items():
    if p.D != 2:
        continue
    try:
        a = p.log_pdf(pt64)
        b = p.log_pdf(pti)
        c = p.log_pdf(pt32)
        print(
            f"{name:34s} f64={a.item():.6f} ({a.dtype})   "
            f"int={b.ravel()[0]!r} ({b.dtype})   "
            f"f32={c.ravel()[0]!r} ({c.dtype})"
        )
    except Exception as e:
        print(f"{name:34s} raised {type(e).__name__}: {e}")
print()

print("--- out-of-support wrap for D = 1..4 (Trapezoidal(0,.25,.75,1)) ---")
for D in (1, 2, 3, 4):
    p = Trapezoidal(0.0, 0.25, 0.75, 1.0, D=D)
    x = np.full((1, D), 20)
    y = p.log_pdf(x)
    yf = p.log_pdf(x.astype(float))
    print(
        f"D={D}: int log_pdf = {y.ravel()[0]!r} ({y.dtype}), pdf = "
        f"{p.pdf(x).ravel()[0]!r}   |  float log_pdf = {yf.ravel()[0]!r}"
    )
print()

print("--- same for SplineTrapezoidal and SmoothBox, D = 1..4 ---")
for cls, args in (
    (SplineTrapezoidal, (0.0, 0.25, 0.75, 1.0)),
    (SmoothBox, (0.0, 1.0, 1.0)),
):
    for D in (1, 2, 3, 4):
        p = cls(*args, D=D)
        x = np.full((1, D), 20)
        y = p.log_pdf(x)
        print(
            f"{cls.__name__:20s} D={D}: {y.ravel()[0]!r} ({y.dtype})  "
            f"pdf={p.pdf(x).ravel()[0]!r}"
        )
print()

print("--- UniformBox and SciPy are not affected ---")
for name, p in (
    ("UniformBox(0,10,D=2)", UniformBox(0.0, 10.0, D=2)),
    ("SciPy(norm())", SciPy(norm())),
):
    for x in (np.array([[3, 5]]), np.array([[30, 50]])):
        xx = x[:, : p.D]
        y = p.log_pdf(xx)
        print(f"{name:22s} x={xx.tolist()} -> {y.ravel()[0]!r} ({y.dtype})")
print()

print("--- Product with a trapezoidal marginal ---")
prod = Product([norm(), Trapezoidal(0.0, 2.0, 8.0, 10.0)])
for x in (np.array([[3.0, 5.0]]), np.array([[3, 5]])):
    y = prod.log_pdf(x)
    print(f"x dtype {x.dtype}: {y.ravel()[0]!r} ({y.dtype})")
print()

print("--- MATLAB: -inf(size(x)) is always double ---")
for name, f, args in (
    ("mtrapezlogpdf", M.mtrapezlogpdf, (0.0, 2.0, 8.0, 10.0)),
    ("msplinetrapezlogpdf", M.msplinetrapezlogpdf, (0.0, 2.0, 8.0, 10.0)),
    ("msmoothboxlogpdf", M.msmoothboxlogpdf, (0.0, 10.0, 1.0)),
):
    for x in (
        np.array([[3.0, 5.0]]),
        np.array([[3, 5]]),
        np.array([[20, 20]]),
    ):
        y = f(x, *args)
        print(f"{name:22s} x dtype {str(x.dtype):8s} -> {y[0]!r} ({y.dtype})")
print()

print("--- can a package caller hand a prior a non-float64 array? ---")
import inspect

from pyvbmc.function_logger import FunctionLogger

src = inspect.getsource(FunctionLogger.__call__)
print("FunctionLogger.__call__ passes x_orig from:")
for line in src.splitlines():
    if "x_orig" in line and "=" in line:
        print("   ", line.strip())
print()
print("parameter_transformer.inverse returns dtype:")
from pyvbmc.parameter_transformer import ParameterTransformer

for kind, kw in (
    ("unbounded", dict(D=2)),
    (
        "bounded",
        dict(
            D=2,
            lb_orig=np.array([[-5.0, -5.0]]),
            ub_orig=np.array([[5.0, 5.0]]),
        ),
    ),
):
    pt = ParameterTransformer(**kw)
    for dt in (np.float64, np.float32, np.int64):
        xx = np.array([[0, 0]]).astype(dt)
        try:
            print(
                f"   {kind:10s} input {dt.__name__:8s} ->",
                pt.inverse(xx).dtype,
            )
        except Exception as e:
            print(
                f"   {kind:10s} input {dt.__name__:8s} raised",
                type(e).__name__,
                e,
            )
print()

print("--- mixed out-of-support counts, int input, Trapezoidal D=3 ---")
p3 = Trapezoidal(0.0, 2.0, 8.0, 10.0, D=3)
for x in ([[20, 20, 5]], [[20, 5, 5]], [[20, 20, 20]], [[5, 5, 5]]):
    xi = np.array(x)
    print(
        f"  x={x} int -> log {p3.log_pdf(xi).ravel()[0]!r}, "
        f"pdf {p3.pdf(xi).ravel()[0]!r}  | float -> "
        f"{p3.log_pdf(xi.astype(float)).ravel()[0]!r}"
    )
print()

print("--- Product with an out-of-support trapezoidal marginal, int input ---")
prod2 = Product(
    [Trapezoidal(0.0, 0.25, 0.75, 1.0), Trapezoidal(0.0, 0.25, 0.75, 1.0)]
)
for x in (np.array([[20, 20]]), np.array([[20.0, 20.0]])):
    print(
        f"  dtype {x.dtype}: {prod2.log_pdf(x).ravel()[0]!r}, "
        f"pdf {prod2.pdf(x).ravel()[0]!r}"
    )
print()

print("--- vectorized-target path: dtype reaching the prior ---")
src2 = inspect.getsource(FunctionLogger.batch_call)
for line in src2.splitlines():
    if "asarray" in line or "astype" in line or "float" in line:
        print("   batch_call:", line.strip())
