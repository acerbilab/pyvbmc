"""P5-12 (P5c F13): the `slicelite` burn-in formula.

Settles that PyVBMC computes
    ceil(thin * log(rindex / log(threshold)))
where MATLAB computes
    ceil(thin * log(rindex) / log(threshold)),
that the two disagree in general, and what each does at the shipped
default `gp_retrain_threshold = 1` in the region the branch is entered in
(`rindex < threshold`, i.e. `rindex < 1`).
"""

import math

import numpy as np

import pyvbmc

print("pyvbmc.__file__ =", pyvbmc.__file__)


def matlab(thin, rindex, thresh):
    # MATLAB: max(1, ceil(Thin*log(rindex)/log(GPRetrainThreshold)))
    with np.errstate(all="ignore"):
        inner = thin * np.log(rindex) / np.log(thresh)
    # MATLAB's ceil(-Inf) = -Inf, ceil(Inf) = Inf, ceil(NaN) = NaN;
    # max(1, -Inf) = 1, max(1, Inf) = Inf, max(1, NaN) = 1.
    if np.isnan(inner):
        return 1.0
    return max(1.0, math.ceil(inner) if np.isfinite(inner) else inner)


def python(thin, rindex, thresh):
    try:
        with np.errstate(all="ignore"):
            return max(1, math.ceil(thin * np.log(rindex / np.log(thresh))))
    except (OverflowError, ValueError) as exc:
        return type(exc).__name__ + ": " + str(exc)


print("\n thin rindex thresh |        MATLAB |        PyVBMC")
for thin, rindex, thresh in [
    (5, 3.0, 2.0),
    (5, 3.0, 10.0),
    (5, 0.5, 2.0),
    (5, 0.9, 10.0),
    (5, 0.5, 1.0),  # the shipped default threshold, inside the branch
    (5, 0.9, 1.0),
    (5, 0.0, 1.0),
]:
    print(
        f" {thin:4d} {rindex:6.2f} {thresh:6.2f} | {str(matlab(thin, rindex, thresh)):>13} "
        f"| {str(python(thin, rindex, thresh)):>13}"
    )

print(
    "\nNote: the branch is entered only when rindex < threshold, so with the"
    "\nshipped threshold 1 MATLAB's numerator log(rindex) is negative and"
    "\nnegative/0 is -Inf, which max(1, .) turns into 1 -- not Inf or NaN."
)
