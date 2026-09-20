"""Second warp with a posterior recorded before the first warp.

Wave-2 finding P1a internal F8. warp_input inverts the search bounds with
the transformer of the posterior it is handed (whitening.py, `warpfun`;
MATLAB misc/warp_input_vbmc.m:8 and :133 do the same with vp.trinfo). The
bounds live in the current inference space. This script warps a stored
state once, then warps it a second time handing in (a) the current
posterior and (b) the posterior recorded before the first warp, and
compares the search box each call returns with the box obtained from the
same uniform draws and the same new transformer when the bounds are
inverted with the current transformer.
"""

import copy

import dill
import numpy as np

import pyvbmc
from pyvbmc.whitening.whitening import warp_gp_and_vp, warp_input

print("pyvbmc:", pyvbmc.__file__, flush=True)
np.set_printoptions(precision=4, suppress=True, linewidth=120)

with open("pyvbmc/testing/vbmc/test_vbmc_save_static.pkl", "rb") as fh:
    vbmc = dill.load(fh)
vbmc.gp = vbmc.get_gp(len(vbmc.iteration_history["iter"]) - 1)
D = vbmc.D
opts = vbmc.options
vp0 = copy.deepcopy(vbmc.vp)  # recorded before any warp: transformer T0
T0 = vp0.parameter_transformer
print("D =", D, " K =", vp0.K)
print("T0: mu", T0.mu, "delta", T0.delta, "R_mat", T0.R_mat is not None)
print(
    "search box in T0 space:\n ",
    vbmc.optim_state["lb_search"],
    "\n ",
    vbmc.optim_state["ub_search"],
)

# ---- first warp, as optimize() does it --------------------------------
T1, optim1, flog1, _ = warp_input(
    vp0, vbmc.optim_state, vbmc.function_logger, opts
)
vbmc.optim_state, vbmc.function_logger = optim1, flog1
vp1, _ = warp_gp_and_vp(T1, vbmc.gp, vbmc.vp, vbmc)  # current vp, T1
print("\nafter warp 1: T1 scale", T1.scale, "\nT1 R_mat\n", T1.R_mat)
print(
    "search box in T1 space:\n ",
    optim1["lb_search"],
    "\n ",
    optim1["ub_search"],
)


def second_warp(vp_in, label):
    rng = vp_in.rng
    state = copy.deepcopy(rng.bit_generator.state)
    T2, optim2, _, _ = warp_input(vp_in, optim1, flog1, opts)
    # replay the draws of warp_input to recover the uniform points it used
    rng.bit_generator.state = state
    rng.random((100000, D))
    u = rng.random((1000, D))
    xx = u * (optim1["ub_search"] - optim1["lb_search"]) + optim1["lb_search"]

    def box(inv_transformer):
        yy = T2(inv_transformer.inverse(xx))
        lo, hi = yy.min(0), yy.max(0)
        d = hi - lo
        return lo - d / 1000, hi + d / 1000

    lo_used, hi_used = box(vp_in.parameter_transformer)
    lo_right, hi_right = box(T1)  # T1 is the space the bounds live in
    assert np.allclose(lo_used, optim2["lb_search"])
    assert np.allclose(hi_used, optim2["ub_search"])
    width_right = hi_right - lo_right
    print(f"\n--- second warp handed {label} ---")
    print(
        "transformer of the posterior == current (T1):",
        vp_in.parameter_transformer == T1,
    )
    print("returned box lb :", optim2["lb_search"].ravel())
    print("returned box ub :", optim2["ub_search"].ravel())
    print("correct  box lb :", lo_right)
    print("correct  box ub :", hi_right)
    print(
        "centre shift / correct width :",
        (0.5 * (lo_used + hi_used) - 0.5 * (lo_right + hi_right))
        / width_right,
    )
    print("returned width / correct width:", (hi_used - lo_used) / width_right)
    overlap = np.maximum(
        0, np.minimum(hi_used, hi_right) - np.maximum(lo_used, lo_right)
    )
    print("overlap / correct width       :", overlap / width_right)


second_warp(copy.deepcopy(vp1), "the current posterior (T1)")
second_warp(copy.deepcopy(vp0), "the posterior recorded before warp 1 (T0)")
