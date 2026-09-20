"""Finding B-4: `_check_warmup_end_conditions` reduces an empty slice when the
stable-warmup window is one iteration.

Settles:
  * that with `tol_stable_warmup <= fun_evals_per_iter` (so
    `tol_stable_warmup_iters == 1`) the first call that passes the guard
    (0-based iteration 2, a three-entry history) raises
    `ValueError: zero-size array to reduction operation maximum ...`;
  * that the following iteration (index 3, four entries) is fine, so the
    window is empty only on the first call;
  * that the shipped defaults (15 / 5, window 3) never produce an empty
    slice;
  * and it tabulates the index sets of both toolboxes' slices so the MATLAB
    side (`private/vbmc_warmup.m:39-40`) can be read off:
        MATLAB  max_now    = elcbo_vec(max(4, end-T+1) : end)     [1-based]
                max_before = elcbo_vec(3 : max(3, end-T))         [1-based]
        Python  max_now    = elcbo_vec[max(3, len-T) :]           [0-based]
                max_before = elcbo_vec[2 : max(3, len-T)]         [0-based]
"""

import math

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc.__file__ =", pyvbmc.__file__)

fun = lambda x: np.sum(x + 2)


def create_vbmc(D, options):
    lb = np.ones((1, D)) * 1.0
    ub = np.ones((1, D)) * 5.0
    plb = np.ones((1, D)) * 2.0
    pub = np.ones((1, D)) * 4.0
    x0 = np.ones((2, D)) * 3.0
    return VBMC(fun, x0, lb, ub, plb, pub, options)


def drive(tol_stable_warmup, fun_evals_per_iter, iteration):
    options = {
        "tol_stable_warmup": tol_stable_warmup,
        "fun_evals_per_iter": fun_evals_per_iter,
    }
    v = create_vbmc(2, options)
    n = iteration + 1
    v.optim_state["iter"] = iteration
    v.optim_state["N"] = 10 * n
    v.optim_state["data_trim_list"] = []
    v.function_logger.func_count = 5 * n
    v.iteration_history["elbo"] = np.linspace(-3.0, -1.0, n)
    v.iteration_history["elbo_sd"] = np.ones(n) * 1e-3
    v.iteration_history["lcb_max"] = np.linspace(-4.0, -2.0, n)
    v.iteration_history["func_count"] = np.arange(1, n + 1) * 5.0
    T = math.ceil(tol_stable_warmup / fun_evals_per_iter)
    guard = iteration > T
    try:
        out = v._check_warmup_end_conditions()
        res = f"returned {out}"
    except Exception as exc:
        res = f"raises {type(exc).__name__}: {exc}"
    print(
        f"tol_stable_warmup={tol_stable_warmup:3d} fun_evals_per_iter="
        f"{fun_evals_per_iter:3d}  T={T}  iteration={iteration} "
        f"(history {n})  guard={guard}  -> {res}"
    )


print("\n-- short window (tol_stable_warmup <= fun_evals_per_iter, T = 1) --")
drive(5, 5, 2)
drive(5, 5, 3)
drive(15, 15, 2)
print("\n-- shipped defaults (15 / 5, T = 3): first call is iteration 4 --")
drive(15, 5, 4)
drive(15, 5, 5)

print("\n-- index sets, both toolboxes, T = 1 --")
for n in (3, 4, 5):
    T = 1
    py_now = list(range(max(3, n - T), n))
    py_before = list(range(2, max(3, n - T)))
    ml_now = list(range(max(4, n - T + 1), n + 1))
    ml_before = list(range(3, max(3, n - T) + 1))
    print(
        f"history {n}: python max_now idx (0-based) {py_now}, "
        f"max_before {py_before} | matlab max_now idx (1-based) {ml_now}, "
        f"max_before {ml_before}"
    )
