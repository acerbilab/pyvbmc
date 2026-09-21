"""W4-3: `string_to_acq` and the names of keyword arguments.

The parser splits the argument string on "," and strips trailing whitespace
alone, so the name of every keyword argument after the first keeps the space
that follows the comma. This script calls it on the argument forms a user can
write in `options["search_acq_fcn"]` and prints what each gives.
"""

import pyvbmc
from pyvbmc.acquisition_functions.utilities import string_to_acq

print("pyvbmc:", pyvbmc.__file__, flush=True)

forms = [
    "AcqFcnLog()",
    "AcqFcnVIQR",
    "AcqFcnVIQR(0.9)",
    "AcqFcnVIQR(quantile=0.9)",
    "AcqFcnVIQR(0.9, 'iqr_reduction')",
    "AcqFcnVIQR(0.9, loss='iqr_reduction')",
    "AcqFcnVIQR(quantile=0.9,loss='iqr_reduction')",
    "AcqFcnVIQR(quantile=0.9, loss='iqr_reduction')",
    "AcqFcnVIQR( quantile=0.9)",
    "AcqFcnVIQR(quantile = 0.9)",
]
for form in forms:
    try:
        acq = string_to_acq(form)
        print(
            f"{form!r:55} -> {type(acq).__name__}, u={getattr(acq, 'u', None)}, loss={getattr(acq, 'loss', None)}"
        )
    except Exception as exc:
        print(f"{form!r:55} -> {type(exc).__name__}: {exc}")
