"""W5, `entmc_vbmc` with a mixture weight that is zero or close to it.

`entmc_vbmc.py:217-218` takes `np.log(E @ wnf)` at samples drawn from every
component, those of weight zero included, and `:234` multiplies the sum of
logs by the weight, as `ent/entmc_vbmc.m:63-67` does (`H = H -
w(j)*sum(log(ys))/Ns`). Where the other components' densities underflow at
the samples of a component of weight zero, that is `0 * (-inf)`. This script
settles:

1. the value and the gradient for a separated two-component posterior as the
   weight of one component goes to zero, through the softmax of `eta` as
   `set_parameters` computes it, and at which gap of `eta` the first
   non-finite number appears;
2. whether the same happens when the components overlap;
3. what the lower bound `entlb_vbmc` returns on the same posteriors.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import pyvbmc
from pyvbmc.entropy import entlb_vbmc, entmc_vbmc
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc:", pyvbmc.__file__, flush=True)


def make_vp(separation, eta_gap):
    vp = VariationalPosterior(1, 2, np.zeros((1, 1)))
    vp.parameter_transformer = ParameterTransformer(1)
    vp.mu = np.array([[0.0, separation]])
    vp.sigma = np.ones((1, 2))
    vp.lambd = np.ones((1, 1))
    eta = np.array([[0.0, -eta_gap]])
    vp.eta = eta
    w = np.exp(eta - eta.max())
    vp.w = w / w.sum()
    vp.rng = np.random.default_rng(0)
    return vp


def describe(a):
    a = np.asarray(a, dtype=float)
    if np.all(np.isfinite(a)):
        return "finite"
    return f"{int(np.sum(~np.isfinite(a)))} of {a.size} non-finite"


for separation in (60.0, 3.0):
    print(
        f"\n--- component means 0 and {separation:g}, unit scales", flush=True
    )
    for gap in (0.0, 30.0, 300.0, 700.0, 740.0, 745.0, 746.0, 800.0):
        vp = make_vp(separation, gap)
        with np.errstate(all="ignore"):
            H, dH = entmc_vbmc(vp, 100, rng=np.random.default_rng(1))
            Hlb, dHlb = entlb_vbmc(vp)
        print(
            f"   eta gap {gap:5.0f}: w2 = {vp.w[0, 1]:9.3e}; entmc H = {H!r:>22}, "
            f"gradient {describe(dH)}; entlb H = {Hlb!r:>20}, gradient "
            f"{describe(dHlb)}"
        )
