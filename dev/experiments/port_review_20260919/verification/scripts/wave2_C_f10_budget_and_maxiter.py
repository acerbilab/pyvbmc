"""Finding 10 (P1b comparison F8, P1b internal F13).

F8: MATLAB errors for a non-positive or non-integer `MaxFunEvals`/`MaxIter`
and raises `MaxIter` to `MinIter` with a warning
(`misc/setupoptions_vbmc.m:109-124`); PyVBMC does none of it.

F13: `_budget_active` is set from whether `precomputed_evaluations=` and
`initialization_cost=` were supplied, not from their values, so passing
either at its documented default switches on the budget path.
"""

import logging

import numpy as np

import pyvbmc
from pyvbmc import VBMC

logging.getLogger("VBMC").setLevel(logging.ERROR)
print("pyvbmc:", pyvbmc.__file__)

D = 2
f = lambda x: -0.5 * np.sum(np.atleast_2d(x) ** 2, axis=1)
lb, ub = np.full((1, D), -10.0), np.full((1, D), 10.0)
plb, pub = np.full((1, D), -1.0), np.full((1, D), 1.0)
x0 = np.zeros((1, D))

print("\n=== F8: no validation of max_iter / max_fun_evals ===")
for opts in (
    {"max_iter": 0},
    {"max_iter": 2.5},
    {"max_iter": -3},
    {"max_fun_evals": 0},
    {"max_fun_evals": -1},
    {"max_fun_evals": 7.5},
    {"max_iter": 1, "min_iter": 5},
    {"max_fun_evals": 3, "min_fun_evals": 20},
):
    try:
        v = VBMC(f, x0, lb, ub, plb, pub, options=opts)
        print(
            f"    {str(opts):46s} accepted: max_iter={v.options['max_iter']}, "
            f"min_iter={v.options['min_iter']}, "
            f"max_fun_evals={v.options['max_fun_evals']}, "
            f"min_fun_evals={v.options['min_fun_evals']}"
        )
    except Exception as exc:
        print(f"    {str(opts):46s} {type(exc).__name__}: {exc}")

print("\n=== F13: _budget_active from presence, not value ===")
cases = [
    ("VBMC(f, ...)", {}),
    ("initialization_cost=0", {"initialization_cost": 0}),
    ("precomputed_evaluations=None", {"precomputed_evaluations": None}),
    (
        "both at their documented defaults",
        {"initialization_cost": 0, "precomputed_evaluations": None},
    ),
]
for name, kwargs in cases:
    v = VBMC(f, x0, lb, ub, plb, pub, **kwargs)
    print(
        f"    {name:36s} _budget_active={v._budget_active}"
        f"  effective_max_fun_evals={v._effective_max_fun_evals}"
        f"  initialization_cost={v.initialization_cost}"
    )
    print(
        f"        _fresh_evaluations_for_batch(5) with func_count=0 -> "
        f"{v._fresh_evaluations_for_batch(5)}"
    )
