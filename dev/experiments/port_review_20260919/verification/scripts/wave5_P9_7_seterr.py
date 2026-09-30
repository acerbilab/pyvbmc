"""P9-7. Does `Trapezoidal.log_pdf` warn at `x == a` where
`SplineTrapezoidal.log_pdf` does not, and what does the spline's
`np.seterr` pair leave behind when the loop raises?

Settles: the warning on one side and not the other; that the restore at
spline_trapezoidal.py:124 is skipped when an exception escapes the loop,
leaving `divide="ignore"` installed; and whether NumPy's error state is
per thread.
"""

import threading
import warnings

import numpy as np

import pyvbmc
from pyvbmc.priors import SplineTrapezoidal, Trapezoidal

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

x = np.array([[0.0], [0.25], [0.5], [1.0]])

for cls in (Trapezoidal, SplineTrapezoidal):
    p = cls(0.0, 0.25, 0.75, 1.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = p.log_pdf(x, keepdims=False)
    print(
        f"{cls.__name__:20s} log_pdf(x) = {y}  warnings = "
        f"{sorted({str(ww.message) for ww in w})}"
    )
print()

print("--- MATLAB: log(0) is -Inf with no warning (read from the source) ---")
print("  mtrapezlogpdf.m:54  y(idx,ii) = log(x(idx,ii) - a(idx,ii)) - lnf")
print("  msplinetrapezlogpdf.m:60  log(-2*z.^3 + 3*z.^2) - lnf")
print("  Neither file sets or restores any warning state.")
print()

print("--- what an exception inside the spline loop leaves behind ---")
print("initial np.geterr() =", np.geterr())
np.seterr(divide="raise")
print("after np.seterr(divide='raise'):", np.geterr())

p = SplineTrapezoidal(0.0, 0.25, 0.75, 1.0, D=2)
# `_log_pdf` loops over the columns of `x` and indexes `self.a[d]`, so an
# `x` with more columns than the prior has dimensions raises inside the
# loop -- the shape mismatch that the public `log_pdf` wrapper would have
# refused, reached here the way any exception inside the loop would be.
try:
    p._log_pdf(np.full((2, 3), 0.5))
except Exception as e:
    print("forced failure inside _log_pdf:", type(e).__name__, str(e)[:90])
print("np.geterr() after the failure =", np.geterr())

# restore for the rest of the script
np.seterr(divide="warn")
print("restored:", np.geterr())
print()

print("--- is NumPy's error state per thread? ---")
np.seterr(divide="raise")
seen = {}


def worker():
    seen["thread"] = np.geterr()["divide"]
    np.seterr(divide="ignore")
    seen["thread_after"] = np.geterr()["divide"]


t = threading.Thread(target=worker)
t.start()
t.join()
print(
    "main thread set divide='raise'; the new thread saw divide =",
    seen["thread"],
)
print("the new thread then set divide='ignore'; it saw", seen["thread_after"])
print("back in the main thread, divide =", np.geterr()["divide"])
np.seterr(divide="warn")
