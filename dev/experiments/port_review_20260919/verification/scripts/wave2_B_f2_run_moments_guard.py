"""Finding B-2: the running average of the variational moments never runs.

Settles:
  * that `len(run_cov == 0)` (the expression at vbmc.py:1580-1582) is `D`,
    so the guard is true at every iteration for every D >= 1;
  * that on the stored run `test_vbmc_save_static.pkl` every recorded
    `optim_state` has `last_run_avg == N`, which only the reset branch can
    produce, and that the recorded `run_mean` swings instead of smoothing.

Read-only: the pickle is opened for reading and nothing is written.
"""

import os

import numpy as np

import pyvbmc

print("pyvbmc.__file__ =", pyvbmc.__file__)

for D in (1, 2, 3, 5, 10):
    run_cov = np.eye(D)
    print(
        f"D={D:2d}  len(run_cov == 0) = {len(run_cov == 0):2d} (truthy: "
        f"{bool(len(run_cov == 0))})   len(run_cov) == 0 -> "
        f"{len(run_cov) == 0}"
    )

pkl = os.path.join(
    os.path.dirname(pyvbmc.__file__),
    "testing",
    "vbmc",
    "test_vbmc_save_static.pkl",
)
import dill

with open(pkl, "rb") as fh:
    vbmc = dill.load(fh)

hist = vbmc.iteration_history
n = len(hist["iter"])
print(
    "\nstored run: recorded iterations =",
    n,
    " optim_state['iter'] =",
    vbmc.optim_state["iter"],
)
print("iter | N | last_run_avg | run_mean[0] | weight would be 0.9**(N-last)")
for i in range(n):
    os_i = hist["optim_state"][i]
    if os_i is None:
        continue
    rm = np.ravel(os_i["run_mean"])
    lra = os_i["last_run_avg"]
    N = os_i["N"]
    print(
        f"{i:4d} | {N} | {lra} | {rm[0]: .4f} | "
        f"{0.9 ** (N - lra) if np.isfinite(lra) else float('nan'):.3f}"
    )
