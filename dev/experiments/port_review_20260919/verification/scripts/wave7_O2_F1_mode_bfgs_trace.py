"""F1: why BFGS inside mode(orig_flag=False) sometimes recovers.

Same mixtures as O2_F1_mode_start_error.py at scale 1e-3. The optimizer
call is wrapped in-process to print every trial point's distance from the
start (in units of the smallest component SD) and the objective there.
"""
import importlib
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize as _minimize
from wave7_O2_common import banner, build_vp

banner()
warnings.simplefilter("ignore")
VPM = importlib.import_module(
    "pyvbmc.variational_posterior.variational_posterior"
)
recs = []


def traced(fun, x0, **kw):
    seen = []

    def f(x):
        out = fun(x)
        seen.append(
            (
                np.array(x),
                float(np.ravel(out[0])[0]),
                float(np.linalg.norm(np.ravel(out[1]))),
            )
        )
        return out

    res = _minimize(f, x0, **kw)
    recs.append((np.array(x0), res, seen))
    return res


VPM.minimize = traced
for D in (2, 4, 6, 10):
    rs = np.random.default_rng(100 + D)
    K = 5
    mu0 = rs.normal(0, 0.7, (D, K))
    sig = rs.uniform(0.7, 1.3, K)
    lam0 = rs.uniform(0.6, 1.4, D)
    w = rs.dirichlet(np.full(K, 3.0))
    scale = 1e-3
    vp = build_vp(mu0 * scale, sig, lam0 * scale, w)
    recs.clear()
    vp.mode(orig_flag=False)
    c = np.min(vp.sigma) * vp.lambd.ravel()
    for x0, res, seen in recs:
        print(
            f"D={D}: nit={res.nit} nfev={res.nfev} success={res.success} "
            f"msg={res.message!r}",
            flush=True,
        )
        for x, v, gn in seen[:8]:
            print(
                f"    step/sd={np.max(np.abs(x - x0) / c):10.3e} "
                f"step/unit={np.linalg.norm(x - x0):9.3e} f={v:.6g} "
                f"|g|={gn:.3g}",
                flush=True,
            )
VPM.minimize = _minimize
