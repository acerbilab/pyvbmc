"""Finding B-3: the two result fields `problem_type` and `iterations`.

Settles:
  * that `optim_state["lb_tran"]` / `["ub_tran"]` are all infinite for a
    bounded, a mixed and an unbounded problem under both
    `bounded_transform` settings, so the `problem_type` test at
    `vbmc.py:3127-3132` always yields "unconstrained";
  * that the same is true of MATLAB's `optimState.LB` / `.UB`, which
    `misc/setupvars_vbmc.m:49-50` fills with the transformed bounds (read,
    not run: the logit branch `shared/warpvars_vbmc.m:104-110` maps the lower
    bound to `log(0) = -Inf`, the probit branch `:254-260` to
    `-sqrt(2)*erfcinv(0) = -Inf`);
  * that the original-space bounds are available in the same dictionary;
  * that `optim_state["iter"]` is the 0-based index of the last iteration,
    one less than the number of recorded iterations.
"""

import os

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc.__file__ =", pyvbmc.__file__)

f = lambda x: -0.5 * np.sum(np.asarray(x).ravel() ** 2)
D = 2
cases = {
    "bounded": (np.full((1, D), -5.0), np.full((1, D), 5.0)),
    # PyVBMC rejects variables bounded on one side only (_bounds.py:282),
    # so the mixed case is one bounded and one unbounded coordinate.
    "mixed": (np.array([[-5.0, -np.inf]]), np.array([[5.0, np.inf]])),
    "unbounded": (np.full((1, D), -np.inf), np.full((1, D), np.inf)),
}
for transform in ("probit", "logit"):
    print(f"\nbounded_transform = {transform}")
    for name, (lb, ub) in cases.items():
        plb = np.full((1, D), -1.0)
        pub = np.full((1, D), 1.0)
        v = VBMC(
            f,
            np.zeros((1, D)),
            lb,
            ub,
            plb,
            pub,
            options={"bounded_transform": transform, "display": "off"},
        )
        os_ = v.optim_state
        test = np.all(np.isinf(os_["lb_tran"])) and np.all(
            np.isinf(os_["ub_tran"])
        )
        print(
            f"  {name:22s} lb_tran={np.ravel(os_['lb_tran'])} "
            f"ub_tran={np.ravel(os_['ub_tran'])} -> problem_type="
            f"{'unconstrained' if test else 'bounded'!r}  "
            f"(lb_orig={np.ravel(os_['lb_orig'])}, "
            f"ub_orig={np.ravel(os_['ub_orig'])})"
        )

import dill

pkl = os.path.join(
    os.path.dirname(pyvbmc.__file__),
    "testing",
    "vbmc",
    "test_vbmc_save_static.pkl",
)
with open(pkl, "rb") as fh:
    stored = dill.load(fh)
print(
    "\nstored run: len(iteration_history['iter']) =",
    len(stored.iteration_history["iter"]),
    "| recorded iter values =",
    list(stored.iteration_history["iter"]),
    "| optim_state['iter'] (what results['iterations'] copies) =",
    stored.optim_state["iter"],
)
