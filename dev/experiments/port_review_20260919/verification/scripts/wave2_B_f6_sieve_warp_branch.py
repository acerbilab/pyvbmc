"""Finding B-6: the warp-undo refit sizes its sieve at `self.vp.K` where
`vbmc.m:584` uses `Knew`.

Settles:
  * that `variable_means` defaults to True, so `vp.optimize_mu` is True after
    warm-up (the only state in which the warp branch runs) and the two
    expressions coincide at the defaults;
  * with `variable_means=False`, the candidate counts the two toolboxes
    request: `ceil(ns_elbo(vp.K))` against `ceil(ns_elbo(Knew))` with
    `Knew = gp.X.shape[0]`, the training-set size;
  * that `final_boost` evaluates `ns_elbo` at `K_new`, as
    `misc/finalboost_vbmc.m:33` does.
"""

import math

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc.__file__ =", pyvbmc.__file__)

f = lambda x: -0.5 * np.sum(np.asarray(x).ravel() ** 2)
D = 2


def make(options):
    return VBMC(
        f,
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -3.0),
        np.full((1, D), 3.0),
        options=dict(options, display="off"),
    )


v = make({})
print("variable_means default:", v.options.get("variable_means"))
print("vp.optimize_mu at construction:", v.vp.optimize_mu)
print("k_warmup:", v.options.get("k_warmup"), " vp.K:", v.vp.K)
print("ns_elbo:", v.options.get("ns_elbo"))

vf = make({"variable_means": False})
print(
    "\nwith variable_means=False, optimize_mu after warm-up:",
    vf.options.get("variable_means"),
)
print(
    "vp.K (stale component count) | Knew = training-set size | "
    "python N_fastopts = ceil(ns_elbo(vp.K)) | matlab = ceil(ns_elbo(Knew))"
)
for vpK, n_train in ((2, 30), (2, 60), (5, 80), (10, 120)):
    py = math.ceil(vf.options.eval("ns_elbo", {"K": vpK}))
    ml = math.ceil(vf.options.eval("ns_elbo", {"K": n_train}))
    print(f"  {vpK:3d} | {n_train:4d} | {py:6d} | {ml:6d}")
