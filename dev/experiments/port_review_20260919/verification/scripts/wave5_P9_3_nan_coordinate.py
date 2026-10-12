"""P9-3. What does each family return for a row that holds a NaN coordinate,
in PyVBMC and in the MATLAB functions?

Settles: the three different PyVBMC answers (UniformBox full density,
the other box families -inf, SciPy NaN), the same question for
`Product` and `SmoothBox`, the MATLAB value of all four `m*logpdf`
functions on the same row, and what `FunctionLogger` does with a NaN
log-joint.
"""

import warnings

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
print()

nan = np.nan
x = np.array([[nan, 0.5]])

py = {
    "UniformBox(0,1,D=2)": UniformBox(0.0, 1.0, D=2),
    "Trapezoidal(0,.25,.75,1,D=2)": Trapezoidal(0.0, 0.25, 0.75, 1.0, D=2),
    "SplineTrapezoidal(0,.25,.75,1,D=2)": SplineTrapezoidal(
        0.0, 0.25, 0.75, 1.0, D=2
    ),
    "SmoothBox(0,1,1,D=2)": SmoothBox(0.0, 1.0, 1.0, D=2),
    "Product[norm,norm]": Product([norm(), norm()]),
    "Product[UniformBox,UniformBox]": Product(
        [UniformBox(0.0, 1.0), UniformBox(0.0, 1.0)]
    ),
}
print("--- PyVBMC, x = [[nan, 0.5]] ---")
for name, p in py.items():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = p.log_pdf(x)
        pdf = p.pdf(x)
    msgs = sorted({str(ww.message) for ww in w})
    print(
        f"{name:36s} log_pdf = {y.ravel()[0]!r:24s} pdf = "
        f"{pdf.ravel()[0]!r:22s} warnings = {msgs}"
    )

p1 = SciPy(norm())
print(
    f"{'SciPy(norm()) at [[nan]]':36s} log_pdf = "
    f"{p1.log_pdf(np.array([[nan]])).ravel()[0]!r}"
)
print()

print("--- MATLAB transcription, x = [[nan, 0.5]] ---")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("munifboxlogpdf(x, 0, 1)        =", M.munifboxlogpdf(x, 0.0, 1.0))
    print(
        "mtrapezlogpdf(x, 0,.25,.75,1)  =",
        M.mtrapezlogpdf(x, 0.0, 0.25, 0.75, 1.0),
    )
    print(
        "msplinetrapezlogpdf(x,0,.25,.75,1) =",
        M.msplinetrapezlogpdf(x, 0.0, 0.25, 0.75, 1.0),
    )
    print(
        "msmoothboxlogpdf(x, 0, 1, 1)   =",
        M.msmoothboxlogpdf(x, 0.0, 1.0, 1.0),
    )
print()
print("(MATLAB's munifbox mask is `any(x<a,2) | any(x>b,2)`, the same")
print(" two comparisons PyVBMC's uniform_box.py:69 makes; the other three")
print(" fill -inf and overwrite only positive membership tests.)")
print()

print("--- what a NaN does inside a run ---")
import inspect

from pyvbmc.function_logger import FunctionLogger

src = inspect.getsource(FunctionLogger.__call__)
i = src.find("# Check function value")
print(src[i : i + 420])
