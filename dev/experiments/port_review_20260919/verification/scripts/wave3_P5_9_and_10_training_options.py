"""P5-9 and P5-10: two branches of `_get_gp_training_options`.

Settles:
  * P5-9 (P5c F10): `init_N = max(round(f(x)), 9)` against MATLAB's
    `Ninit = max(round(f(x)), 0)` — where the clamp binds, and what MATLAB
    would give there.
  * P5-10 (P5c F11): the retrain-threshold branch tests `iteration > 1`
    where the same function's other translation of MATLAB's `iter > 1` is
    `iteration > 0`; whether the reliability index masks the difference.
"""

import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import _get_gp_training_options

print("pyvbmc.__file__ =", pyvbmc.__file__)

D = 2
f = lambda x: -np.sum(np.atleast_2d(x) ** 2, axis=1)
x0 = np.zeros((1, D))
plb, pub = np.full((1, D), -1.0), np.full((1, D), 1.0)


def make(opts=None):
    v = VBMC(f, x0, None, None, plb, pub, opts or {})
    v.options.__setitem__("weighted_hyp_cov", False, force=True)
    return v


# --- P5-9: the floor of the N-dependent design size ---
print("\n[P5-9] init_N schedule, gp_train_n_init=1024 -> _final=64")
v = make({"max_fun_evals": 5000})
a = -(v.options["gp_train_n_init"] - v.options["gp_train_n_init_final"])
b, c, d = -3 * a, 3 * a, v.options["gp_train_n_init"]
cubic = lambda x_: a * x_**3 + b * x_**2 + c * x_ + d
fes = v.options["fun_eval_start"]
lim = min(v.optim_state.get("max_fun_evals", v.options["max_fun_evals"]), 1e3)
print(f"  fun_eval_start={fes}  schedule_limit={lim}  span={lim - fes}")
for n_eff in (10, 100, 500, 1000, 1400, 1450, 1600, 2500):
    v.optim_state["n_eff"] = n_eff
    v.optim_state["iter"] = 0
    v.optim_state["recompute_var_post"] = True
    x = (n_eff - fes) / (lim - fes)
    gt = _get_gp_training_options(
        v.optim_state, v.iteration_history, v.options, {"run_cov": None}, 8
    )
    print(
        f"  n_eff={n_eff:5d}  x={x:6.3f}  round(f(x))={round(cubic(x)):6d}"
        f"   python init_N={gt['init_N']:5d}   matlab Ninit="
        f"{max(round(cubic(x)), 0):5d}"
    )

# --- P5-10: the iteration guard ---
print("\n[P5-10] retrain-threshold branch: MATLAB `iter > 1` (1-based)")
v = make()
v.options.__setitem__("gp_retrain_threshold", 10, force=True)
v.optim_state["n_eff"] = 10
v.optim_state["recompute_var_post"] = False
for it in (1, 2, 3):
    v.optim_state["iter"] = it
    v.iteration_history.record("r_index", 1.0, it - 1)
    gt = _get_gp_training_options(
        v.optim_state, v.iteration_history, v.options, {"run_cov": None}, 8
    )
    # MATLAB's own predicate, in 0-based terms: iteration > 0.
    matlab_branch = it > 0 and 1.0 < v.options["gp_retrain_threshold"]
    print(
        f"  python iter={it} (MATLAB iter={it + 1}): r_index[{it - 1}]=1.0 "
        f"< threshold  ->  python init_N={gt['init_N']:5d} opts_N="
        f"{gt['opts_N']}; MATLAB would take the branch: {matlab_branch}"
    )

print("\n  first iteration at which `r_index` can be finite:")
src = "pyvbmc/vbmc/vbmc.py _compute_reliability_index"
print(
    "   PyVBMC returns inf for optim_state['iter'] < 2 (0-based);"
    " MATLAB vbmc_termination.m returns early for optimState.iter < 3"
)
