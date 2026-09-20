"""Finding B-1: the forced entropy switch reads `entropy_force_switch` from
`optim_state`, where nothing ever writes it.

Settles, without running `optimize()`:
  * that a `VBMC` built with `options={"entropy_switch": True}` at D >= 5 has
    `optim_state["entropy_switch"] is True` and no `entropy_force_switch` key;
  * that the exact expression the main loop evaluates at vbmc.py:1224-1227
    raises `TypeError`;
  * that `self.options.get("entropy_force_switch")` (the MATLAB read) is 0.8;
  * that D < 5 forces the flag off, so the expression is never reached there.
"""

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc.__file__ =", pyvbmc.__file__)


def build(D, entropy_switch):
    lb = np.full((1, D), -np.inf)
    ub = np.full((1, D), np.inf)
    plb = np.full((1, D), -3.0)
    pub = np.full((1, D), 3.0)
    x0 = np.zeros((1, D))
    f = lambda x: -0.5 * np.sum(np.asarray(x).ravel() ** 2)
    return VBMC(
        f,
        x0,
        lb,
        ub,
        plb,
        pub,
        options={"entropy_switch": entropy_switch, "display": "off"},
    )


for D in (2, 5):
    v = build(D, True)
    os_ = v.optim_state
    print(
        f"\nD={D}, options['entropy_switch'] =",
        v.options.get("entropy_switch"),
    )
    print("  optim_state['entropy_switch'] =", os_.get("entropy_switch"))
    print(
        "  'entropy_force_switch' in optim_state:",
        "entropy_force_switch" in os_,
    )
    print(
        "  optim_state.get('entropy_force_switch') =",
        os_.get("entropy_force_switch"),
    )
    print(
        "  options.get('entropy_force_switch') =",
        v.options.get("entropy_force_switch"),
    )
    print("  optim_state['max_fun_evals'] =", os_.get("max_fun_evals"))
    # exactly the expression at vbmc.py:1224-1227
    try:
        gate = os_.get("entropy_switch") and (
            v.function_logger.func_count
            >= os_.get("entropy_force_switch") * os_.get("max_fun_evals")
        )
        print("  loop expression ->", gate)
    except Exception as exc:
        print("  loop expression raises:", type(exc).__name__, exc)

# The stability branch (vbmc.py:2169) against vbmc_termination.m:80.
v = build(5, True)
print(
    "\nstability branch reads only entropy_switch; "
    "options['entropy_force_switch'] is finite by default:",
    np.isfinite(v.options.get("entropy_force_switch")),
)
