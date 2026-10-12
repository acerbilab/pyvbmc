"""A run continued with a larger max_fun_evals: the budget copy in optim_state.

Wave-2 findings P1a internal F7 and P1b internal F3. VBMC.load applies
new_options to the options, and refreshes optim_state["max_fun_evals"] only
on the opted-in budget path. The GP training options read that copy as the
horizon of the schedule that sets the number of starting points of the
hyperparameter fit (gaussian_process_train.py, "N-dependent initial
training points"); termination reads the option.
"""

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc:", pyvbmc.__file__, flush=True)

path = "pyvbmc/testing/vbmc/test_vbmc_save_static.pkl"
plain = VBMC.load(path)
more = VBMC.load(path, new_options={"max_fun_evals": 500})
for name, v in (("loaded as saved", plain), ("loaded with 500", more)):
    print(
        f"{name:18s} options max_fun_evals = {v.options['max_fun_evals']:4d}"
        f"   optim_state copy = {v.optim_state['max_fun_evals']}"
        f"   budget path active = {v._budget_active}"
    )

o = more.options
n_init, n_final = o["gp_train_n_init"], o["gp_train_n_init_final"]
start = o["fun_eval_start"]
print(
    f"\nschedule: gp_train_n_init = {n_init}, final = {n_final}, "
    f"fun_eval_start = {start}"
)


def init_n(n_eff, limit):
    a = -(n_init - n_final)
    x = (n_eff - start) / (min(limit, 1e3) - start)
    return max(round(a * x**3 - 3 * a * x**2 + 3 * a * x + n_init), 9), x


print("n_eff | starting points with the stale horizon (x) | with the option")
for n_eff in (40, 60, 100, 200, 350, 500):
    stale, xs = init_n(n_eff, more.optim_state["max_fun_evals"])
    fresh, xf = init_n(n_eff, o["max_fun_evals"])
    print(
        f"{n_eff:5d} | {stale:6d}  (x = {xs:5.2f})"
        f"                   | {fresh:6d}  (x = {xf:4.2f})"
    )
