"""O3 F6: what the optim_state recorded at a warp iteration holds under
"hyp_dict", and whether an undone warp leaves the same staleness.

No optimize() run. The real warp_input and warp_gp_and_vp are called on a
constructed state, and the statements of VBMC.optimize that touch
self.hyp_dict and self.optim_state are replayed in their order
(vbmc.py:1426-1447 warp block, 1457 train_gp in the undo check, 1527-1531
undo, 1560-1586 active sampling with the relink at 1571 and 1586, 1599
train_gp, 1939-1945 record through IterationHistory, 3252-3253 load).
train_gp is replaced by a stand-in that does what gaussian_process_train.py
:168 and :208 do to its argument: it assigns a new "hyp" into the dict it
is given and returns that same dict.
"""
import copy

import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc.vbmc.iteration_history import IterationHistory
from pyvbmc.whitening import warp_gp_and_vp, warp_input

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpr.__file__)

# Reuse the constructed state of the F2 script.
import importlib.util
import pathlib

here = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "f2", here / "O3_F2_zero_mean_warp.py"
)
f2 = importlib.util.module_from_spec(spec)
import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(f2)


def fake_train_gp(hyp_dict, tag):
    hyp_dict["hyp"] = np.full_like(np.atleast_2d(hyp_dict["hyp"]), tag)
    return hyp_dict


def replay(kept):
    pt, fl, vp, gp, optim_state = f2.state("negquad")
    optim_state["skip_active_sampling"] = False
    hist = IterationHistory(["optim_state"])
    # State at the start of the warp iteration k: the previous iteration's
    # active sampling linked the two (vbmc.py:1571, 1586) and its train_gp
    # wrote the fit h_{k-1} into the shared dict.
    self_hyp_dict = {
        "hyp": gp.posteriors[0].hyp[None, :].copy(),
        "run_cov": None,
    }
    optim_state["hyp_dict"] = self_hyp_dict
    h_prev = self_hyp_dict["hyp"].copy()
    self_optim_state = optim_state
    print(
        f"\n[{'kept' if kept else 'undone'}] at start: optim_state['hyp_dict'] "
        f"is self.hyp_dict: {self_optim_state['hyp_dict'] is self_hyp_dict}"
    )
    # vbmc.py:1426, 1431
    optim_state_old = copy.deepcopy(self_optim_state)
    hyp_dict_old = copy.deepcopy(self_hyp_dict)
    # vbmc.py:1433-1447
    pt_w, self_optim_state, fl_w, _ = warp_input(
        vp, self_optim_state, fl, f2.options
    )
    print(
        "  after warp_input: optim_state['hyp_dict'] is self.hyp_dict:",
        self_optim_state["hyp_dict"] is self_hyp_dict,
        "| equal to h_{k-1}:",
        np.array_equal(self_optim_state["hyp_dict"]["hyp"], h_prev),
    )
    vp_w, hyp_warped = warp_gp_and_vp(
        pt_w, gp, vp, f2._Owner(self_optim_state)
    )
    self_hyp_dict["hyp"] = hyp_warped
    # vbmc.py:1457 (warp_undo_check, default True)
    self_hyp_dict = fake_train_gp(self_hyp_dict, 111.0)
    if not kept:
        # vbmc.py:1527-1531
        self_optim_state = optim_state_old
        self_hyp_dict = hyp_dict_old
    # vbmc.py:1560-1586
    if self_optim_state.get("skip_active_sampling"):
        self_optim_state["skip_active_sampling"] = False
        relinked = False
    else:
        self_optim_state["hyp_dict"] = self_hyp_dict
        self_hyp_dict = self_optim_state["hyp_dict"]
        relinked = True
    # vbmc.py:1599
    self_hyp_dict = fake_train_gp(self_hyp_dict, 222.0)
    # vbmc.py:1939-1945
    hist.record("optim_state", self_optim_state, 0)
    rec = hist["optim_state"][0]["hyp_dict"]["hyp"]
    live = self_hyp_dict["hyp"]
    print(
        f"  active sampling ran (relink): {relinked}; live self.hyp_dict "
        f"holds the fit of this iteration: {np.all(live == 222.0)}"
    )
    print(
        "  recorded optim_state['hyp_dict']['hyp']: equals h_{k-1} (pre-warp):",
        np.array_equal(rec, h_prev),
        "| equals this iteration's fit:",
        bool(np.all(rec == 222.0)),
    )
    # vbmc.py:3252-3253: VBMC.load(file, iteration=k)
    loaded_hyp = hist["optim_state"][0]["hyp_dict"]["hyp"]
    if kept:
        # What the stale vector means in the new space: compare its
        # length scales and m0 with the warped ones.
        D = 3
        print(
            "  stale ell (old space):",
            np.round(np.exp(loaded_hyp[0, :D]), 4),
            " warped ell (new space):",
            np.round(np.exp(hyp_warped[0, :D]), 4),
        )
        print(
            f"  stale m0 {loaded_hyp[0, D + 2]:.4f} against warped m0 "
            f"{hyp_warped[0, D + 2]:.4f}"
        )


replay(kept=True)
replay(kept=False)
