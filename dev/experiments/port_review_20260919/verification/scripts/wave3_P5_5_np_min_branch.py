"""P5-5 (P5i F5 / P5c F14): `np.min` called with two positional arguments.

Settles that `np.min(np.min(y), np.max(y) - 20 * D)` at
`gaussian_process_train.py:427` passes the second value as `axis` and
raises, that `np.minimum` is what MATLAB's
`min(min(y_all), max(y_all) - 20*D)` means, and that the branch guarded by
`optim_state["gp_noise_fun"][2] == 1` cannot be entered through any
supported route.
"""

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc.__file__ =", pyvbmc.__file__)

D = 2
y = np.array([[-3.0], [-1.0], [-7.0]])
try:
    np.min(np.min(y), np.max(y) - 20 * D)
except Exception as exc:  # noqa: BLE001
    print(
        "np.min(np.min(y), np.max(y) - 20*D) ->", type(exc).__name__ + ":", exc
    )
print(
    "np.minimum(np.min(y), np.max(y) - 20*D) =",
    np.minimum(np.min(y), np.max(y) - 20 * D),
    "  (MATLAB: min(min(y_all), max(y_all) - 20*D))",
)

# Reachability of the branch.
f = lambda x: -np.sum(np.atleast_2d(x) ** 2, axis=1)
x0 = np.zeros((1, D))
plb, pub = np.full((1, D), -1.0), np.full((1, D), 1.0)
for name, opts in (
    ("default (level 0)", {}),
    ("uncertainty_handling=True (level 1)", {"uncertainty_handling": True}),
    ("specify_target_noise=True (level 2)", {"specify_target_noise": True}),
):
    v = VBMC(f, x0, None, None, plb, pub, opts)
    print(f"{name:40s} gp_noise_fun = {v.optim_state['gp_noise_fun']}")

try:
    VBMC(f, x0, None, None, plb, pub, {"noise_shaping": True})
except Exception as exc:  # noqa: BLE001
    print(
        "noise_shaping=True at construction ->",
        type(exc).__name__ + ":",
        str(exc).split("\n")[0],
    )
