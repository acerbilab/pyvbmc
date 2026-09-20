"""Finding 1 (P1b internal F7, P1b comparison F3): how `integer_vars` is read.

Settles, at D=3 with half-integer hard bounds so the bound check can pass:
what `optim_state["integer_vars"]` becomes for a plain Python list of
indices, for a NumPy index array, and for a length-D mask; and whether the
list case is silent.  MATLAB (`misc/setupvars_vbmc.m:14-24`) reads 1-based
indices into a length-nvars logical vector.
"""

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc:", pyvbmc.__file__)

D = 3
f = lambda x: -0.5 * np.sum(np.atleast_2d(x) ** 2, axis=1)
lb = np.full((1, D), -0.5)
ub = np.full((1, D), 10.5)
plb = np.full((1, D), 1.0)
pub = np.full((1, D), 9.0)
x0 = np.full((1, D), 5.0)

cases = [
    ("default []", None),
    ("list [0, 2]  (0-based indices)", [0, 2]),
    ("list [1, 3]  (MATLAB 1-based indices)", [1, 3]),
    ("list [0, 1, 0]  (mask written as a list)", [0, 1, 0]),
    ("list [True, False, True]", [True, False, True]),
    ("np.array([0, 2])", np.array([0, 2])),
    ("np.array([1, 3])", np.array([1, 3])),
    ("np.array([1, 0, 1])  (mask)", np.array([1, 0, 1])),
    ("np.array([True, False, True])  (mask)", np.array([True, False, True])),
    ("np.array([0, 1, 2])  (all three, 0-based)", np.array([0, 1, 2])),
]
for name, val in cases:
    opts = {} if val is None else {"integer_vars": val}
    try:
        v = VBMC(f, x0, lb, ub, plb, pub, options=opts)
        print(f"{name:45s} -> {v.optim_state['integer_vars']}")
    except Exception as exc:
        print(f"{name:45s} -> {type(exc).__name__}: {str(exc)[:110]}")

# The half-integer / infinite bound check, with non-half-integer bounds.
print()
print("--- with plain integer hard bounds 0 / 10 (not half-integer) ---")
lb2 = np.full((1, D), 0.0)
ub2 = np.full((1, D), 10.0)
for name, val in [
    ("list [0, 2]", [0, 2]),
    ("np.array([1,0,1])", np.array([1, 0, 1])),
]:
    try:
        v = VBMC(f, x0, lb2, ub2, plb, pub, options={"integer_vars": val})
        print(f"{name:25s} -> {v.optim_state['integer_vars']}")
    except Exception as exc:
        print(
            f"{name:25s} -> {type(exc).__name__}: {' '.join(str(exc).split())[:110]}"
        )

print()
print("--- with infinite hard bounds ---")
lb3 = np.full((1, D), -np.inf)
ub3 = np.full((1, D), np.inf)
for name, val in [("np.array([1,0,0])", np.array([1, 0, 0]))]:
    try:
        v = VBMC(f, x0, lb3, ub3, plb, pub, options={"integer_vars": val})
        print(f"{name:25s} -> {v.optim_state['integer_vars']}")
    except Exception as exc:
        print(
            f"{name:25s} -> {type(exc).__name__}: {' '.join(str(exc).split())[:110]}"
        )

print()
print("--- raw numpy semantics behind the list case ---")
print("[0, 2] != 0 ->", repr([0, 2] != 0))
a = np.full(3, False)
a[[0, 2] != 0] = True
print("np.full(3, False)[True] = True ->", a)
print(
    "np.full((1,3), 1.0)[:, True].shape ->",
    np.full((1, 3), 1.0)[:, True].shape,
)
