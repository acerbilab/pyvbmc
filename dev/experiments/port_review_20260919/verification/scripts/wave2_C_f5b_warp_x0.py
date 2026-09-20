"""Finding 5, second half: what a warp does to `vbmc.x0` and `__str__`.

`VBMC.x0` is written once, at `vbmc.py:460`, in the *initial* transformed
space, and afterwards read only by `_init_optim_state` (`:859-869`, before
any warp) and by `__str__` (`:3496`, through
`self.parameter_transformer.inverse`).  `optimize()` replaces
`self.parameter_transformer` with the (possibly warped) `vp` transformer at
`:1385` and never re-transforms `x0`.  This script warps the stored run's
state once with `whitening.warp_input` (read-only on the repository) and
prints the two inversions of the stored `x0`.
"""

import logging
from pathlib import Path

import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.whitening import warp_input

logging.getLogger("VBMC").setLevel(logging.ERROR)
print("pyvbmc:", pyvbmc.__file__)

static = (
    Path(__file__).resolve().parents[5]
    / "pyvbmc/testing/vbmc/test_vbmc_save_static.pkl"
)
s = VBMC.load(static)
print("x0 (stored, initial transformed space):", s.x0)
print(
    "inverse with the initial transformer:  ",
    s.parameter_transformer.inverse(s.x0),
)

pt_new, optim_state_new, fl_new, action = warp_input(
    s.vp, s.optim_state, s.function_logger, s.options
)
print("warp action:", action)
print(
    "new transformer R_mat is None:",
    pt_new.R_mat is None,
    "| scale:",
    None if pt_new.scale is None else np.round(pt_new.scale, 4),
)
print(
    "inverse of the SAME x0 with the warped transformer:", pt_new.inverse(s.x0)
)
print(
    "(these are the two values `__str__` would print before and after the warp;"
)
print(" the starting point itself never moved)")
