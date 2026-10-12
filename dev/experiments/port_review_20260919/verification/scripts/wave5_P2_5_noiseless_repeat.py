"""P2i, last paragraph of the first-question answer: an acquired point that
is already a training input on a noiseless target.

Settles: the function logger pools the evaluation into the existing row, no
training row is added, the target call is spent, and the in-loop GP update
takes the full recomputation rather than the rank-one extension.  Also
checks whether anything on either side guards against acquiring an input
that is already in the training set.
"""
import importlib
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
import _common  # noqa: E402
from _common import banner, build, with_gp  # noqa: E402

from pyvbmc.vbmc.active_sample import active_sample  # noqa: E402

banner()
asmod = importlib.import_module("pyvbmc.vbmc.active_sample")

print("\n=== D=1, one integer variable, the bounded search lands on x=0 ===")
v = build(
    1,
    {"integer_vars": np.array([True]), "search_optimizer": "cmaes"},
    lb=np.array([[-10.5]]),
    ub=np.array([[10.5]]),
)
gp, fl, os_ = with_gp(v, n=12)
print(
    "  training inputs (original):",
    np.round(np.ravel(fl.X_orig[fl.X_flag]), 6),
)

# Record which GP update path the second sample takes.
calls = {"reupdate": 0, "rank_one": 0}
real_reupdate = asmod.reupdate_gp


def spy_reupdate(function_logger, gp_):
    calls["reupdate"] += 1
    return real_reupdate(function_logger, gp_)


asmod.reupdate_gp = spy_reupdate
real_update = type(gp).update


def spy_update(self, *a, **kw):
    calls["rank_one"] += 1
    return real_update(self, *a, **kw)


type(gp).update = spy_update

before = {
    "func_count": fl.func_count,
    "Xn": fl.Xn,
    "live": int(np.sum(fl.X_flag)),
    "gp_rows": gp.X.shape[0],
    "n_evals": fl.n_evals[fl.X_flag].ravel().copy(),
    "y_max": float(fl.y_max),
}
_common.TARGET_CALLS.clear()
try:
    fl, os_, _, gp = active_sample(
        gp, 2, os_, fl, v.iteration_history, v.vp, v.options
    )
finally:
    asmod.reupdate_gp = real_reupdate
    type(gp).update = real_update

after = {
    "func_count": fl.func_count,
    "Xn": fl.Xn,
    "live": int(np.sum(fl.X_flag)),
    "gp_rows": gp.X.shape[0],
    "n_evals": fl.n_evals[fl.X_flag].ravel().copy(),
    "y_max": float(fl.y_max),
}
print(
    "  target calls this step:", [np.round(c, 8) for c in _common.TARGET_CALLS]
)
for k in ("func_count", "Xn", "live", "gp_rows", "y_max"):
    print(f"  {k:10s}: {before[k]} -> {after[k]}")
print("  n_evals before:", before["n_evals"])
print("  n_evals after :", after["n_evals"])
print(
    "  GP update path: rank-one calls =",
    calls["rank_one"],
    " full recomputations =",
    calls["reupdate"],
)

print("\n=== control: a fresh point takes the rank-one extension ===")
v0 = build(2, {"search_optimizer": "none"})
gp0, fl0, os0 = with_gp(v0, n=12)
calls0 = {"reupdate": 0, "update": 0}
real_reupdate0 = asmod.reupdate_gp


def spy_reupdate0(function_logger, gp_):
    calls0["reupdate"] += 1
    return real_reupdate0(function_logger, gp_)


asmod.reupdate_gp = spy_reupdate0
real_update0 = type(gp0).update


def spy_update0(self, *a, **kw):
    calls0["update"] += 1
    return real_update0(self, *a, **kw)


type(gp0).update = spy_update0
n0 = fl0.Xn
try:
    fl0, os0, _, gp0 = active_sample(
        gp0, 2, os0, fl0, v0.iteration_history, v0.vp, v0.options
    )
finally:
    asmod.reupdate_gp = real_reupdate0
    type(gp0).update = real_update0
print(
    "  rows",
    n0,
    "->",
    fl0.Xn,
    "| gp.update calls =",
    calls0["update"],
    "| reupdate_gp calls =",
    calls0["reupdate"],
)
print(
    "  (so in the colliding run above the single gp.update call came from "
    "inside reupdate_gp)"
)

print(
    "\n=== the same collision with the sieve alone (search_optimizer='none') ==="
)
v2 = build(
    2,
    {
        "integer_vars": np.array([True, False]),
        "search_optimizer": "none",
        "ns_search": 64,
    },
    lb=np.array([[-10.5, -np.inf]]),
    ub=np.array([[10.5, np.inf]]),
)
gp2, fl2, os2 = with_gp(v2, n=12)


class HitTrainRow(_common.Cheap):
    """Prefer exactly the snapped image of training row 0."""


from pyvbmc.acquisition_functions import AbstractAcqFcn  # noqa: E402

row0 = fl2.X[fl2.X_flag][0].copy()
snapped0 = AbstractAcqFcn._real2int(
    row0.copy()[None, :],
    fl2.parameter_transformer,
    v2.optim_state["integer_vars"],
)[0]
print(
    "  training row 0 (original):",
    fl2.parameter_transformer.inverse(row0[None, :]),
    " snapped:",
    fl2.parameter_transformer.inverse(snapped0[None, :]),
)
print(
    "  is the snapped image already a training input?",
    bool(np.any(np.all(np.isclose(fl2.X[fl2.X_flag], snapped0), axis=1))),
)

print("\n=== is there any guard against acquiring an existing input? ===")
src = open(asmod.__file__, encoding="utf8").read()
for needle in ("duplicate", "already", "X_flag]", "np.isclose"):
    print(
        f"  occurrences of {needle!r} in active_sample.py:", src.count(needle)
    )
