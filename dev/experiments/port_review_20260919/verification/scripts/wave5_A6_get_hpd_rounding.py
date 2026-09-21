"""W5, the rounding of the size of the high-posterior-density subset.

`pyvbmc/stats/get_hpd.py:38` has `hpd_N = round(hpd_frac * N)` with Python's
built-in `round`, which sends a half to the even integer, where
`misc/gethpd_vbmc.m:10` has MATLAB's `round`, which sends a half away from
zero. This script settles, in floating point as the code computes it:

1. for which `N` up to 2000 the two conventions give different sizes, at the
   fractions a run can pass: the default `hpd_frac = 0.8` of every default
   call (`gaussian_process_train.py:327`, `variational_optimization.py:798`,
   `active_sample.py:536`), its eighth, 0.1, the exact lower endpoint of the
   fractions of `active_sample.py:983-1004` (reached with
   `hpd_search_frac > 0` alone; the default is 0), and 0.5 as an example of a
   user's `hpd_frac`;
2. `get_hpd` itself at `N = 5`, `hpd_frac = 0.1`;
3. `np.round` at `active_sample.py:994`, the split of the draws among six
   fractions, for the counts a run can ask for.
"""

import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import pyvbmc
from pyvbmc.stats import get_hpd

print("pyvbmc:", pyvbmc.__file__, flush=True)


def away_from_zero(x):
    return math.floor(abs(x) + 0.5) * (1 if x >= 0 else -1)


print(
    "\n--- 1. N up to 2000 at which round() and MATLAB's round differ",
    flush=True,
)
for frac in (0.8, 0.8 / 8, 0.5):
    differ = [
        N
        for N in range(1, 2001)
        if round(frac * N) != away_from_zero(frac * N)
    ]
    print(
        f"   hpd_frac = {frac:g}: {len(differ)} values of N; first ones "
        f"{differ[:8]}"
    )

print("\n--- 2. get_hpd at N = 5, hpd_frac = 0.1", flush=True)
X = np.arange(10.0).reshape(5, 2)
y = np.array([[3.0], [1.0], [4.0], [1.5], [5.0]])
hpd_X, hpd_y, hpd_range, idx = get_hpd(X, y, 0.8 / 8)
print(
    "   points returned:",
    hpd_X.shape[0],
    "(MATLAB's round gives",
    away_from_zero(0.1 * 5),
    "), range",
    hpd_range,
)

print(
    "\n--- 3. np.round(np.linspace(0, N_hpd, 7)) against rounding away from"
    " zero",
    flush=True,
)
differ = []
for N_hpd in range(1, 8193):
    edges = np.linspace(0, N_hpd, 7)
    a = np.diff(np.round(edges))
    b = np.diff([away_from_zero(e) for e in edges])
    if not np.array_equal(a, b):
        differ.append(N_hpd)
print(
    f"   N_hpd in 1..8192 with a different split: {len(differ)}; first ones "
    f"{differ[:8]}"
)
if differ:
    N_hpd = differ[0]
    edges = np.linspace(0, N_hpd, 7)
    print(
        f"   N_hpd = {N_hpd}: np.round split",
        np.diff(np.round(edges)),
        " away-from-zero split",
        np.diff([away_from_zero(e) for e in edges]),
    )
