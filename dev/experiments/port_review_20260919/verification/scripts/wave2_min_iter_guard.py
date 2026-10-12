"""Minimum-iteration guard: how many iterations does min_iter = k buy?

Wave-2 finding (P1a internal F4, P1a comparison F4, P1b comparison F9).
MATLAB (private/vbmc_termination.m:98-99) blocks termination while the
1-based iteration counter is below MinIter, so a run whose only active
stopping rule is the iteration limit performs max(MinIter, MaxIter)
iterations. The reliability index is stubbed out: only the iteration
limits are exercised.
"""

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc:", pyvbmc.__file__, flush=True)


def iterations_performed(min_iter, max_iter, D=3):
    v = VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        -10 * np.ones((1, D)),
        10 * np.ones((1, D)),
        -np.ones((1, D)),
        np.ones((1, D)),
        options={
            "display": "off",
            "min_iter": min_iter,
            "max_iter": max_iter,
            "min_fun_evals": 0,
        },
        seed=0,
    )
    v._compute_reliability_index = lambda tol: (np.inf, np.nan)
    v.function_logger.func_count = 0  # the evaluation limits stay inactive
    for index in range(50):  # index is the 0-based iteration, as in optimize
        v.optim_state["iter"] = index
        finished, _ = v._check_termination_conditions()
        if finished:
            return index + 1
    return None


print("min_iter max_iter | PyVBMC iterations | MATLAB rule max(min,max)")
for mn, mx in [(3, 3), (3, 1), (5, 5), (1, 1), (0, 4), (4, 10)]:
    print(
        f"{mn:8d} {mx:8d} | {iterations_performed(mn, mx):17d} | "
        f"{max(mn, mx):10d}"
    )
