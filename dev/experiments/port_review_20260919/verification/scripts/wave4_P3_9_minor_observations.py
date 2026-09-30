"""P3-9: the minor observations of the comparison report.

Settles the Python halves of: (a) `np.maximum` propagating NaN where
MATLAB's two-argument `max` drops it, and what `np.argmin` then selects;
(b) the `-inf` guards of the log sums producing `-inf` where MATLAB
produces NaN; (c) `string_to_acq` splitting the argument string on `,`
and `=`.
"""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from wave4_P3_common import banner  # noqa: E402

banner()

REALMAX = np.finfo(np.float64).max

print("--- (a) NaN through np.maximum and np.argmin ---")
print("np.maximum(nan, -realmax) =", np.maximum(np.nan, -REALMAX))
for a in (
    np.array([1.0, np.nan, 2.0]),
    np.array([-REALMAX, np.nan]),
    np.array([np.nan, -REALMAX]),
    np.array([np.nan, np.nan]),
):
    m = np.maximum(a, -REALMAX)
    ml = np.where(np.isnan(a), -REALMAX, np.maximum(a, -REALMAX))
    print(
        f"  acq={a} -> python {m} argmin {np.argmin(m)} | "
        f"MATLAB {ml} argmin {np.argmin(ml)}"
    )

print("\n--- (b) the -inf guard of a log sum ---")
zz = np.full((2, 4), -np.inf)
zz[1, 0] = -1.0
ln_max = np.amax(zz, axis=1)
ml_out = ln_max + np.log(np.sum(np.exp(zz - ln_max.reshape(-1, 1)), axis=1))
guarded = ln_max.copy()
guarded[guarded == -np.inf] = 0.0
py_out = guarded + np.log(np.sum(np.exp(zz - guarded.reshape(-1, 1)), axis=1))
print(
    "  MATLAB (no guard):",
    ml_out,
    "-> max(.,-realmax):",
    np.where(np.isnan(ml_out), -REALMAX, np.maximum(ml_out, -REALMAX)),
)
print(
    "  Python (guarded) :",
    py_out,
    "-> np.maximum(.,-realmax):",
    np.maximum(py_out, -REALMAX),
)

print("\n--- (c) string_to_acq splitting ---")
from pyvbmc.acquisition_functions.utilities import string_to_acq  # noqa: E402

for s in (
    "AcqFcnVIQR",
    "AcqFcnVIQR()",
    "AcqFcnVIQR(0.666)",
    "AcqFcnVIQR(quantile=0.666)",
    "AcqFcnVIQR(0.666, 'iqr_reduction')",
    "AcqFcnVIQR(quantile=0.666, loss='iqr_reduction')",
    "AcqFcnVIQR(loss='iqr,reduction')",
    "AcqFcnVIQR(loss='a=b')",
):
    try:
        a = string_to_acq(s)
        print(f"  {s!r:55s} -> u={a.u:.6f} loss={getattr(a,'loss',None)!r}")
    except Exception as exc:
        print(f"  {s!r:55s} -> {type(exc).__name__}: {exc}")
