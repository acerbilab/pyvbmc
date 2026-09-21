"""P2i F5 and its first-question table: which paths put a point on the grid.

Settles, with one integer variable: which of the initial-design paths
(provided x0, the 'plausible' design, the 'narrow' design, a starting-cache
row the design consumes) reach the target off the integer grid, and whether
each local-search branch ('none', 'cmaes', 'Nelder-Mead', the D=1 bounded
search) delivers a snapped point; plus the idempotence of the snap.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
import _common  # noqa: E402
from _common import (  # noqa: E402
    Cheap,
    banner,
    build,
    noisy_target,
    quad_target,
    set_acq,
    with_gp,
)

from pyvbmc.acquisition_functions import AbstractAcqFcn  # noqa: E402
from pyvbmc.vbmc.active_sample import active_sample  # noqa: E402

banner()

INT_LB = np.array([[-10.5, -np.inf]])
INT_UB = np.array([[10.5, np.inf]])
INT_OPT = {"integer_vars": np.array([True, False])}


def on_grid(x):
    return bool(np.all(np.abs(x[0] - np.round(x[0])) < 1e-9))


print("\n=== initial design: provided x0 off the grid ===")
_common.TARGET_CALLS.clear()
v = build(
    2,
    dict(INT_OPT, search_optimizer="none"),
    lb=INT_LB,
    ub=INT_UB,
    x0=np.array([[3.4, 0.7]]),
)
fl, os_, _, _ = active_sample(
    None,
    4,
    v.optim_state,
    v.function_logger,
    v.iteration_history,
    v.vp,
    v.options,
)
calls = np.array(_common.TARGET_CALLS)
print("  init_design =", v.options["init_design"])
print("  target saw (original coordinates):")
for row in calls:
    print("   ", np.round(row, 6), " first coord on grid:", on_grid(row))

print("\n=== initial design: 'narrow' ===")
_common.TARGET_CALLS.clear()
v = build(
    2,
    dict(INT_OPT, search_optimizer="none", init_design="narrow"),
    lb=INT_LB,
    ub=INT_UB,
    x0=np.array([[3.4, 0.7]]),
)
active_sample(
    None,
    4,
    v.optim_state,
    v.function_logger,
    v.iteration_history,
    v.vp,
    v.options,
)
for row in np.array(_common.TARGET_CALLS):
    print("   ", np.round(row, 6), " first coord on grid:", on_grid(row))

print("\n=== initial design: a starting-cache row with no stored value ===")
_common.TARGET_CALLS.clear()
v = build(2, dict(INT_OPT, search_optimizer="none"), lb=INT_LB, ub=INT_UB)
v.optim_state["cache"]["x_orig"] = np.array([[2.4, 0.1], [-1.7, 0.3]])
v.optim_state["cache"]["y_orig"] = np.array([np.nan, np.nan])
active_sample(
    None,
    2,
    v.optim_state,
    v.function_logger,
    v.iteration_history,
    v.vp,
    v.options,
)
for row in np.array(_common.TARGET_CALLS):
    print("   ", np.round(row, 6), " first coord on grid:", on_grid(row))

print("\n=== the local-search branches ===")
for optimizer in ("none", "cmaes", "Nelder-Mead"):
    _common.TARGET_CALLS.clear()
    v = build(
        2, dict(INT_OPT, search_optimizer=optimizer), lb=INT_LB, ub=INT_UB
    )
    gp, fl, os_ = with_gp(v, n=12)
    n0 = fl.Xn
    _common.TARGET_CALLS.clear()
    fl, os_, _, gp = active_sample(
        gp, 1, os_, fl, v.iteration_history, v.vp, v.options
    )
    x_new = fl.X_orig[fl.Xn]
    seen = _common.TARGET_CALLS[-1] if _common.TARGET_CALLS else None
    print(
        f"  [{optimizer:11s}] rows {n0}->{fl.Xn}  x_orig={np.round(x_new, 6)}"
        f"  on grid={on_grid(x_new)}"
        f"  target saw={None if seen is None else np.round(seen, 6)}"
    )

# D == 1: the bounded scalar search
_common.TARGET_CALLS.clear()
v = build(
    1,
    {"integer_vars": np.array([True]), "search_optimizer": "cmaes"},
    lb=np.array([[-10.5]]),
    ub=np.array([[10.5]]),
)
gp, fl, os_ = with_gp(v, n=12)
n0 = fl.Xn
_common.TARGET_CALLS.clear()
fl, os_, _, gp = active_sample(
    gp, 1, os_, fl, v.iteration_history, v.vp, v.options
)
x_new = fl.X_orig[fl.Xn]
seen = _common.TARGET_CALLS[-1] if _common.TARGET_CALLS else None
print(
    f"  [bounded D=1] rows {n0}->{fl.Xn}  x_orig={np.round(x_new, 6)}"
    f"  on grid={on_grid(x_new)}"
    f"  target saw={None if seen is None else np.round(seen, 6)}"
)

print("\n=== the snap is idempotent ===")
v = build(2, INT_OPT, lb=INT_LB, ub=INT_UB)
pt = v.parameter_transformer
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2))
X1 = AbstractAcqFcn._real2int(X.copy(), pt, v.optim_state["integer_vars"])
X2 = AbstractAcqFcn._real2int(X1.copy(), pt, v.optim_state["integer_vars"])
print(
    "  _real2int(_real2int(X)) == _real2int(X):", bool(np.array_equal(X1, X2))
)
print(
    "  every snapped row on the grid:",
    bool(
        np.all(
            np.abs(pt.inverse(X1)[:, 0] - np.round(pt.inverse(X1)[:, 0]))
            < 1e-9
        )
    ),
)

print("\n=== a starting-cache row drawn by the sieve is snapped ===")
v = build(
    2,
    dict(INT_OPT, ns_search=1, cache_frac=1.0, search_optimizer="none"),
    lb=INT_LB,
    ub=INT_UB,
)
gp, fl, os_ = with_gp(v, n=12)
set_acq(v, Cheap())
os_["cache"]["x_orig"] = np.array([[2.4, 0.1]])
os_["cache"]["y_orig"] = np.array([-77.0])
_common.TARGET_CALLS.clear()
fl, os_, _, gp = active_sample(
    gp, 1, os_, fl, v.iteration_history, v.vp, v.options
)
print(
    "  cached x_orig [2.4, 0.1] with stored y -77.0 recorded at",
    np.round(fl.X_orig[fl.Xn], 8),
    " y_orig =",
    fl.y_orig[fl.Xn, 0],
)

print("\n=== a repeated observation is NOT snapped (by design) ===")
_common.TARGET_CALLS.clear()
v = build(
    2,
    dict(
        INT_OPT,
        search_optimizer="none",
        specify_target_noise=True,
        max_repeated_observations=3,
    ),
    target=noisy_target,
    lb=INT_LB,
    ub=INT_UB,
)
gp, fl, os_ = with_gp(v, n=12)
X_train = fl.X[fl.X_flag].copy()


TARGET_ROW = X_train[3].copy()  # an off-grid row of the initial design
# `AbstractAcqFcn.__call__` snaps its input in place, so the acquisition
# sees the snapped copy of the training row; aim at that.
SNAPPED_ROW = AbstractAcqFcn._real2int(
    TARGET_ROW.copy()[None, :],
    fl.parameter_transformer,
    v.optim_state["integer_vars"],
)[0]
print(
    "  the row aimed at, original coordinates:",
    fl.parameter_transformer.inverse(TARGET_ROW[None, :]),
    " snapped:",
    fl.parameter_transformer.inverse(SNAPPED_ROW[None, :]),
)


class PreferOneTrainRow(AbstractAcqFcn):
    def _compute_acquisition_function(
        self, Xs, vp, gp, function_logger, optim_state, *a
    ):
        Xs = np.atleast_2d(Xs)
        hit = np.all(np.isclose(Xs, SNAPPED_ROW), axis=1)
        return np.where(hit, 0.0, 1.0)


set_acq(v, PreferOneTrainRow())
n0 = fl.Xn
_common.TARGET_CALLS.clear()
fl, os_, _, gp = active_sample(
    gp, 1, os_, fl, v.iteration_history, v.vp, v.options
)
seen = _common.TARGET_CALLS[-1] if _common.TARGET_CALLS else None
print(
    "  rows",
    n0,
    "->",
    fl.Xn,
    "| streak",
    os_["repeated_observations_streak"],
    "| target saw",
    None if seen is None else np.round(seen, 6),
    "| on grid:",
    None if seen is None else on_grid(seen),
)
