"""P5-8 (P5i F7): `gp_mean_fun` names accepted at construction that the
first GP training call rejects.

Settles how many of the names `VBMC.__init__` accepts
(`vbmc.py:1103-1123`) `_meanfun_name_to_mean_function` implements, that
construction of a `VBMC` object with one of the others succeeds, and that
the failure comes only from inside `train_gp`.
"""

import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import (
    _meanfun_name_to_mean_function,
    train_gp,
)

print("pyvbmc.__file__ =", pyvbmc.__file__)

valid = [
    "zero",
    "const",
    "negquad",
    "se",
    "negquadse",
    "negquadfixiso",
    "negquadfix",
    "negquadsefix",
    "negquadonly",
    "negquadfixonly",
    "negquadlinonly",
    "negquadmix",
]
print("names VBMC.__init__ accepts:", len(valid))
ok, bad = [], []
for name in valid:
    try:
        _meanfun_name_to_mean_function(name)
        ok.append(name)
    except ValueError:
        bad.append(name)
print("  implemented by _meanfun_name_to_mean_function:", ok)
print(
    "  rejected there                               :",
    bad,
    f"({len(bad)} of {len(valid)})",
)

D = 2
N = 6
f = lambda x: -np.sum(np.atleast_2d(x) ** 2, axis=1)
v = VBMC(
    f,
    np.zeros((1, D)),
    None,
    None,
    np.full((1, D), -1.0),
    np.full((1, D), 1.0),
    {"gp_mean_fun": "se"},
)
print(
    '\nVBMC(..., options={"gp_mean_fun": "se"}) constructed:',
    type(v).__name__,
    "; optim_state['gp_mean_fun'] =",
    v.optim_state["gp_mean_fun"],
)

rng = np.random.default_rng(0)
window = v.optim_state["pub_tran"] - v.optim_state["plb_tran"]
Xs = window * rng.random((N, D)) + v.optim_state["plb_tran"]
for i in range(N):
    v.function_logger.X_flag[i] = True
    v.function_logger.X[i] = Xs[i]
    v.function_logger.y[i] = f(Xs[i])
    v.function_logger.fun_eval_time[i] = 1e-5
v.optim_state["N"] = N
v.optim_state["n_eff"] = N
try:
    train_gp(
        {},
        v.optim_state,
        v.function_logger,
        v.iteration_history,
        v.options,
        v.optim_state["plb_tran"],
        v.optim_state["pub_tran"],
        rng=0,
    )
except Exception as exc:  # noqa: BLE001
    print("train_gp ->", type(exc).__name__ + ":", exc)
