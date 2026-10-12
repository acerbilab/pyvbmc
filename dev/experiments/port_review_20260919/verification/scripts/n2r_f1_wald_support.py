import logging
import sys

import numpy as np

sys.path.insert(0, r"C:/Users/luigi/Documents/GitHub/pyvbmc")
logging.disable(logging.WARNING)
import pymc as pm

from pyvbmc import VBMC
from pyvbmc.pymc import PyMCTarget

rng = np.random.default_rng(0)
obs = rng.normal(3.0, 0.5, size=10)
with pm.Model() as m:
    x = pm.Wald("x", mu=1.0, lam=2.0, alpha=2.0)
    pm.Normal("y", x, 0.5, observed=obs)
t = PyMCTarget(m, seed=0)
print("support", t.support, "kept", t.kept, "lb", t.lb, "ub", t.ub, flush=True)
print(
    "x0",
    t.x0,
    "plb",
    t.plb,
    "pub",
    t.pub,
    "mode",
    t.plausible_info["mode"],
    flush=True,
)
print(
    "true support lower = alpha = 2.0; coordinate log(2) =",
    np.log(2.0),
    flush=True,
)
print("log_joint at log(1.9):", end=" ", flush=True)
try:
    print(t.log_joint(np.array([np.log(1.9)])), flush=True)
except Exception as e:
    print(f"{type(e).__name__}: {str(e)[:100]}", flush=True)
v = VBMC(
    t,
    options={
        "max_iter": 2,
        "min_iter": 1,
        "max_fun_evals": 40,
        "display": "off",
    },
    seed=1,
)
try:
    vp, res = v.optimize()
    print("run finished; func_count", res["func_count"], flush=True)
except Exception as e:
    print("OPTIMIZE FAILED:", type(e).__name__, str(e)[:160], flush=True)
