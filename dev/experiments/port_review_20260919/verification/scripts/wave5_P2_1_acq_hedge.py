"""P2i F2: does ``acq_hedge=True`` raise at the first active-sampling step?

Settles: (a) whether construction accepts the option, (b) what
``optim_state["hedge"]`` holds, (c) what happens on one ``active_sample``
step with one acquisition function and with two, (d) whether the option is
registered inert.
"""
import sys
import traceback

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from _common import Cheap, banner, build, set_acq, with_gp  # noqa: E402

from pyvbmc.vbmc.active_sample import active_sample  # noqa: E402
from pyvbmc.vbmc.options import INERT_OPTIONS  # noqa: E402

banner()

print("\n=== (a) is acq_hedge in INERT_OPTIONS? ===")
for name in ("acq_hedge", "acq_hedge_decay", "acq_hedge_iter_window"):
    print(f"  {name!r:28s} inert={name in INERT_OPTIONS}")

print("\n=== (b) construction with acq_hedge=True ===")
v = build(2, {"acq_hedge": True, "search_optimizer": "none"})
print(
    "  constructed, no error.  optim_state['hedge'] =",
    repr(v.optim_state.get("hedge", "<absent>")),
)
print("  options['acq_hedge'] =", v.options["acq_hedge"])

gp, fl, os_ = with_gp(v)
set_acq(v, Cheap())

print("\n=== (c) one active_sample step with acq_hedge=True, 1 acq fcn ===")
try:
    active_sample(gp, 1, os_, fl, v.iteration_history, v.vp, v.options)
except Exception as exc:  # noqa: BLE001
    print("  RAISED", type(exc).__name__, ":", exc)
    tb = traceback.extract_tb(sys.exc_info()[2])[-1]
    print(
        "  at",
        tb.filename.split("pyvbmc")[-1],
        "line",
        tb.lineno,
        ":",
        tb.line,
    )
else:
    print("  no exception (unexpected)")

print("\n=== (c2) same with two acquisition functions ===")
v2 = build(2, {"acq_hedge": True, "search_optimizer": "none"}, seed=9)
gp2, fl2, os2 = with_gp(v2)
v2.options.__setitem__("search_acq_fcn", [Cheap(), Cheap()], force=True)
try:
    active_sample(gp2, 1, os2, fl2, v2.iteration_history, v2.vp, v2.options)
except Exception as exc:  # noqa: BLE001
    print("  RAISED", type(exc).__name__, ":", exc)
else:
    print("  no exception (unexpected)")

print("\n=== (d) control: acq_hedge=False takes a step ===")
v3 = build(2, {"acq_hedge": False, "search_optimizer": "none"}, seed=9)
gp3, fl3, os3 = with_gp(v3)
set_acq(v3, Cheap())
n0 = fl3.Xn
fl3, os3, _, gp3 = active_sample(
    gp3, 1, os3, fl3, v3.iteration_history, v3.vp, v3.options
)
print("  rows", n0, "->", fl3.Xn)
