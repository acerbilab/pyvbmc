"""N2 (static review) F1: does the default PyMCTarget construction take prior
draws whose only use is a location warning?

Counts calls to PyMCTarget._prior_draws and the draws they take, on a model
whose Hessian at the mode is positive definite (so the curvature fallback,
the other consumer of the draws, is not needed), with and without `start`.
Run from the repository root with the PyMC environment.
"""
import logging
import sys
import time

import numpy as np

sys.path.insert(0, r"C:/Users/luigi/Documents/GitHub/pyvbmc")
logging.disable(logging.WARNING)

from pyvbmc.pymc import PyMCTarget, _target  # noqa: E402
from pyvbmc.testing.pymc import models as M  # noqa: E402

specs = M.accepted_models()
print("accepted models:", [s["name"] for s in specs], flush=True)
print("_PRIOR_DRAWS =", getattr(_target, "_PRIOR_DRAWS", "?"), flush=True)

calls = []
orig = PyMCTarget._prior_draws


def counting(self, *a, **k):
    t0 = time.perf_counter()
    out = orig(self, *a, **k)
    calls.append(
        (
            time.perf_counter() - t0,
            {n: np.shape(v) for n, v in out.items()}
            if isinstance(out, dict)
            else np.shape(out),
        )
    )
    return out


PyMCTarget._prior_draws = counting

for spec in specs:
    calls.clear()
    t0 = time.perf_counter()
    t = PyMCTarget(spec["model"], seed=0)
    total = time.perf_counter() - t0
    info = t.plausible_info
    print(
        f"{spec['name']:>14s}: default construction {total:6.2f}s; "
        f"_prior_draws calls={len(calls)} "
        f"time={sum(c[0] for c in calls):6.2f}s; route={info.get('route')} "
        f"curvature={info.get('curvature')} "
        f"location_outside_prior={info.get('location_outside_prior')}",
        flush=True,
    )
    if calls:
        print("               draw shapes:", calls[0][1], flush=True)
