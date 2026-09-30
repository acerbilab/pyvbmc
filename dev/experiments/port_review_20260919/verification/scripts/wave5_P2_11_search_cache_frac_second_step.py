"""Adjacent to P2i F3: what ``search_cache_frac > 0`` does on the second step.

The shares of the sieve add up to ``search_cache_frac + 0.25 + 0.25 + 0 +
0.25`` at the shipped fractions.  The search cache is empty on the first
step, so its share is clipped to zero there; from the second step on it is
full.  This settles at which value of ``search_cache_frac`` the second step
raises the guard of ``_get_search_points`` (:1050-1057), and what MATLAB's
``getSearchPoints`` does with the same numbers.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from _common import Cheap, banner, build, set_acq, with_gp  # noqa: E402

from pyvbmc.vbmc.active_sample import active_sample  # noqa: E402

banner()

for frac in (0.1, 0.25, 0.26, 0.5, 1.0):
    v = build(
        2,
        {
            "ns_search": 64,
            "search_cache_frac": frac,
            "search_optimizer": "none",
            "cache_frac": 0,
        },
    )
    gp, fl, os_ = with_gp(v, n=12)
    set_acq(v, Cheap())
    os_["cache"]["x_orig"] = np.empty((0, 2))
    os_["cache"]["y_orig"] = np.empty(0)
    outcome = []
    for step in (1, 2):
        try:
            fl, os_, _, gp = active_sample(
                gp, 1, os_, fl, v.iteration_history, v.vp, v.options
            )
        except Exception as exc:  # noqa: BLE001
            outcome.append(f"step {step}: {type(exc).__name__}")
            break
        else:
            outcome.append(f"step {step}: ok")
    print(f"  search_cache_frac={frac:<5} -> {'; '.join(outcome)}")

print("\n  shipped fractions: heavy 0.25 + mvn 0.25 + hpd 0 + box 0.25 = 0.75")
print("  so the parts exceed the request once round(frac*N) > 0.25*N")
