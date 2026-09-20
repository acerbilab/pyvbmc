import logging
import sys

import numpy as np

sys.path.insert(0, r"C:/Users/luigi/Documents/GitHub/pyvbmc")
logging.disable(logging.WARNING)
import pymc as pm

from pyvbmc import VBMC
from pyvbmc.pymc import PyMCTarget

with pm.Model() as m:
    pm.Wald("x", mu=1.0, lam=2.0, alpha=2.0)  # true support (2, inf)
t = PyMCTarget(m, seed=0)
print("claimed support:", t.support, flush=True)
try:
    PyMCTarget(m, start={"x": 1.0}, seed=0)
except Exception as e:
    print(
        "start=1.0 (inside claimed, outside true):",
        type(e).__name__,
        str(e)[:110],
        flush=True,
    )
try:
    t2 = PyMCTarget(
        m, start={"x": 3.0}, plausible_bounds={"x": (0.5, 1.5)}, seed=0
    )
    print(
        "plausible_bounds entirely outside true support accepted; plb/pub =",
        t2.plb,
        t2.pub,
        "x0",
        t2.x0,
        flush=True,
    )
    v = VBMC(
        t2,
        options={
            "max_iter": 2,
            "min_iter": 1,
            "max_fun_evals": 40,
            "display": "off",
        },
        seed=2,
    )
    vp, res = v.optimize()
    print("run finished", res["func_count"], flush=True)
except Exception as e:
    print("OPTIMIZE/SETUP FAILED:", type(e).__name__, str(e)[:130], flush=True)
