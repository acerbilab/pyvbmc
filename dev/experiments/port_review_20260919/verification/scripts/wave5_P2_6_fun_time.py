"""P2i F7: the initial design is not charged to ``main_timer``'s ``fun_time``.

Settles: after ``active_sample(gp=None, ...)`` the process-wide timer holds
no ``fun_time`` entry, while after an active-sampling step it does; that
``Timer.get_duration`` on a missing key warns and returns ``None``; and
that a target raising inside the active-sampling branch leaves the timer
running (no ``finally``).
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from _common import Cheap, banner, build, set_acq, with_gp  # noqa: E402

from pyvbmc.timer import main_timer  # noqa: E402
from pyvbmc.vbmc.active_sample import active_sample  # noqa: E402

banner()

print("\n=== initial design ===")
main_timer.reset()
v = build(2, {"search_optimizer": "none"})
fl, os_, _, _ = active_sample(
    None,
    6,
    v.optim_state,
    v.function_logger,
    v.iteration_history,
    v.vp,
    v.options,
)
print(
    "  main_timer._durations after the initial design:", main_timer._durations
)
print("  main_timer._start_times:", main_timer._start_times)
print("  get_duration('fun_time') ->", main_timer.get_duration("fun_time"))
print(
    "  function_logger.total_fun_eval_time (its own local Timer) =",
    round(float(fl.total_fun_eval_time), 6),
)

print("\n=== one active-sampling step ===")
v2 = build(2, {"search_optimizer": "none"})
gp2, fl2, os2 = with_gp(v2)
set_acq(v2, Cheap())
main_timer.reset()
fl2, os2, _, gp2 = active_sample(
    gp2, 1, os2, fl2, v2.iteration_history, v2.vp, v2.options
)
print(
    "  main_timer._durations:",
    {k: round(x, 6) for k, x in main_timer._durations.items()},
)

print("\n=== a target that raises inside the active-sampling branch ===")
calls = {"n": 0}


def blows_up(x):
    calls["n"] += 1
    if calls["n"] > 12:
        raise RuntimeError("target failure")
    return -0.5 * float(np.sum(np.asarray(x) ** 2))


v3 = build(2, {"search_optimizer": "none"}, target=blows_up)
gp3, fl3, os3 = with_gp(v3)
set_acq(v3, Cheap())
main_timer.reset()
try:
    active_sample(gp3, 1, os3, fl3, v3.iteration_history, v3.vp, v3.options)
except Exception as exc:  # noqa: BLE001
    print("  raised", type(exc).__name__)
print("  _start_times left running:", list(main_timer._start_times))
print("  _durations:", list(main_timer._durations))
main_timer.reset()
