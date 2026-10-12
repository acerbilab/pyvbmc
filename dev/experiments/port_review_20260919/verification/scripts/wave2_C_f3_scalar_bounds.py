"""Finding 3 (P1b internal F6, P1b comparison F7): scalar bounds.

Settles: whether a scalar hard/plausible bound is replicated across
dimensions as the `VBMC` docstring (`vbmc.py:98-104`) and MATLAB
(`misc/boundscheck_vbmc.m:6-10`) say; what the raised message looks like;
and the D == 1 case, where the scalar survives as a 0-d array into the
plausible-bound derivation.
"""

import logging

import numpy as np

import pyvbmc
from pyvbmc import VBMC

logging.getLogger("VBMC").setLevel(logging.ERROR)
print("pyvbmc:", pyvbmc.__file__)

f = lambda x: -0.5 * np.sum(np.atleast_2d(x) ** 2, axis=1)


def attempt(label, *args, **kwargs):
    try:
        v = VBMC(*args, **kwargs)
        print(
            f"{label:60s} OK  lb={v.lower_bounds} ub={v.upper_bounds} "
            f"plb={v.plausible_lower_bounds} pub={v.plausible_upper_bounds}"
        )
        return v
    except Exception as exc:
        print(f"{label:60s} {type(exc).__name__}: {exc!r}")
        return None


for D in (1, 2, 3, 5):
    x0 = np.zeros((1, D))
    attempt(
        f"D={D}, scalar LB/UB, array PLB/PUB",
        f,
        x0,
        -10.0,
        10.0,
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
    )
for D in (1, 3):
    x0 = np.zeros((1, D))
    attempt(f"D={D}, all four bounds scalar", f, x0, -10.0, 10.0, -1.0, 1.0)

print()
print(
    "--- D == 1, scalar bounds, several starting rows, no plausible bounds ---"
)
attempt(
    "D=1, x0 = [[1.],[3.]], scalar LB/UB, no PLB/PUB",
    f,
    np.array([[1.0], [3.0]]),
    -10.0,
    10.0,
)
attempt(
    "D=1, x0 = [[2.],[2.]] (zero width), scalar LB/UB, no PLB/PUB",
    f,
    np.array([[2.0], [2.0]]),
    -10.0,
    10.0,
)
attempt(
    "D=1, x0 = [[2.],[2.]] (zero width), array LB/UB, no PLB/PUB",
    f,
    np.array([[2.0], [2.0]]),
    np.array([[-10.0]]),
    np.array([[10.0]]),
)
print()
print(
    "--- D == 3, array bounds, zero-width multi-row x0, no PLB/PUB (MATLAB M5 counterpart) ---"
)
attempt(
    "D=3, x0 two identical rows, array LB/UB, no PLB/PUB",
    f,
    np.tile(np.array([[1.0, 2.0, 3.0]]), (2, 1)),
    np.full((1, 3), -10.0),
    np.full((1, 3), 10.0),
)
attempt(
    "D=3, x0 rows differ in coords 0,1 only, array LB/UB, no PLB/PUB",
    f,
    np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 3.0]]),
    np.full((1, 3), -10.0),
    np.full((1, 3), 10.0),
)
