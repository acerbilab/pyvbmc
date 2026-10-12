"""Save a short bounded run and its posterior under this interpreter.

Usage: python wave2_xver_make.py <tag> <directory>

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

tag, out = sys.argv[1], sys.argv[2]
D = 2
vbmc = VBMC(
    lambda x: -0.5 * np.sum((np.atleast_2d(x) - 3.0) ** 2),
    np.full((1, D), 3.0),
    np.full((1, D), 0.0),
    np.full((1, D), 10.0),
    np.full((1, D), 2.0),
    np.full((1, D), 4.0),
    options={"display": "off", "max_iter": 3},
    seed=7,
)
vp, _ = vbmc.optimize()
vp.save(f"{out}/vp_{tag}.pkl", overwrite=True)
vbmc.save(f"{out}/vbmc_{tag}.pkl", overwrite=True)
mu, _ = vp.moments(orig_flag=True, cov_flag=True)
print(
    f"saved under Python {sys.version.split()[0]}: K = {vp.K}, mean {np.round(mu.ravel(), 4)}"
)
