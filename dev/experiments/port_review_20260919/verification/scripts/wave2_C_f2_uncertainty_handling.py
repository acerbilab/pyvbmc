"""Finding 2 (P1b comparison F4): `uncertainty_handling` is tested by length.

Settles which values of the option `VBMC.__init__` accepts and what
uncertainty level each produces (`vbmc.py:1016-1021`), against MATLAB's
truth test (`misc/setupvars_vbmc.m:230-236` with `utils/evalbool.m`).
Also records what `integer_vars`-style lists do for the neighbouring
`[0, 2]` case in f1.
"""

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc:", pyvbmc.__file__)

D = 2
f = lambda x: -0.5 * np.sum(np.atleast_2d(x) ** 2, axis=1)
lb = np.full((1, D), -10.0)
ub = np.full((1, D), 10.0)
plb = np.full((1, D), -1.0)
pub = np.full((1, D), 1.0)
x0 = np.zeros((1, D))

values = [
    ("default (absent)", "ABSENT"),
    ("[]", []),
    ("np.array([])", np.array([])),
    ("True", True),
    ("False", False),
    ("1", 1),
    ("0", 0),
    ("2", 2),
    ("None", None),
    ("[0]", [0]),
    ("[False]", [False]),
    ("[1]", [1]),
    ("[2]", [2]),
    ("'yes'", "yes"),
    ("'no'", "no"),
    ("'off'", "off"),
    ("np.True_", np.True_),
]
for name, val in values:
    opts = (
        {}
        if isinstance(val, str) and val == "ABSENT"
        else {"uncertainty_handling": val}
    )
    try:
        v = VBMC(f, x0, lb, ub, plb, pub, options=opts)
        print(
            f"{name:18s} -> level {v.optim_state['uncertainty_handling_level']}"
            f"  gp_noise_fun={v.optim_state['gp_noise_fun']}"
        )
    except Exception as exc:
        print(
            f"{name:18s} -> {type(exc).__name__}: {' '.join(str(exc).split())[:90]}"
        )

print()
print("--- f1 addendum: plain lists that the internal report called masks ---")
D3 = 3
lb3 = np.full((1, D3), -0.5)
ub3 = np.full((1, D3), 10.5)
plb3 = np.full((1, D3), 1.0)
pub3 = np.full((1, D3), 9.0)
x03 = np.full((1, D3), 5.0)
for val in ([1], [0, 1], [0, 2], [0, 1, 0]):
    v = VBMC(f, x03, lb3, ub3, plb3, pub3, options={"integer_vars": val})
    print(f"integer_vars={str(val):12s} -> {v.optim_state['integer_vars']}")
