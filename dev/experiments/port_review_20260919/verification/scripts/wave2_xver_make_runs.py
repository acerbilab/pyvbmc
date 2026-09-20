"""Three converged bounded runs of one problem; save each posterior.

Usage: python wave2_xver_make_runs.py <directory>

Part of the two-interpreter experiment behind W2-30 of `../wave2.md`: a
posterior of a bounded problem saved under one minor version of Python and
used under another. Each interpreter needs PyVBMC's dependencies and, with
`PYTHONPATH`, this checkout and the gpyreg checkout; every step runs in a
process of its own, because the failing ones end the interpreter.
"""

import logging
import sys

import numpy as np

logging.disable(logging.CRITICAL)
from pyvbmc import VBMC

out = sys.argv[1]
D = 2
for seed in (1, 2, 3):
    vbmc = VBMC(
        lambda x: -0.5 * np.sum((np.atleast_2d(x) - 3.0) ** 2),
        np.full((1, D), 3.0),
        np.full((1, D), 0.0),
        np.full((1, D), 10.0),
        np.full((1, D), 2.0),
        np.full((1, D), 4.0),
        options={"display": "off"},
        seed=seed,
    )
    vp, _ = vbmc.optimize()
    vp.save(f"{out}/run{seed}_vp.pkl", overwrite=True)
    print(
        "run", seed, "stable", bool(vp.stats["stable"]), "K", vp.K, flush=True
    )
data = open(f"{out}/run1_vp.pkl", "rb").read()
print(
    "saved under Python",
    sys.version.split()[0],
    "| files hold functions by value:",
    b"_create_function" in data,
)
