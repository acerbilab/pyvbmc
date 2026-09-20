"""Settles the warping findings of slice P8.

P8c F4 / P8i F5 (second half) - ``warp_input`` rewrites only the rows with
    ``X_flag`` true (``whitening.py:213-220``); ``misc/warp_input_vbmc.m:112-119``
    rewrites rows ``1:Xn``.  Shows the stale rows and what still reads them.
P8i F11 - ``warp_input`` divides the log-Jacobian by the temperature;
    ``FunctionLogger._record`` does not.  MATLAB divides in both places
    (``warp_input_vbmc.m:117``; ``funlogger_vbmc.m:243``, ``:269``).
P8c F6 - the low-correlation mask: MATLAB keeps ``abs(corr) > thresh`` and
    zeroes the complement, so a NaN entry is zeroed; PyVBMC zeroes
    ``abs(corr) <= thresh``, so a NaN entry survives.
P8c F10 - ``np.quantile`` against MATLAB's ``quantile`` convention.
P8i F8 - the "Reset GP Hyperparameters" block.
"""

import copy

import numpy as np

import pyvbmc
from pyvbmc.whitening import warp_input

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()


def target(x):
    x = np.atleast_2d(x)
    return float(-0.5 * np.sum(x**2))


vbmc = pyvbmc.VBMC(
    target,
    np.array([[0.1, 0.2]]),
    np.array([[-np.inf, -np.inf]]),
    np.array([[np.inf, np.inf]]),
    np.array([[-2.0, -2.0]]),
    np.array([[2.0, 2.0]]),
    options={"max_fun_evals": 20},
    seed=7,
)
fl = vbmc.function_logger
pts = np.array([[0.1, -0.2], [0.5, 0.4], [-0.7, 0.3], [0.0, 0.9]])
for p in pts:
    fl.add(p, target(p))
fl.X_flag[1] = False  # what the warm-up trim does
vbmc.optim_state["N"] = 4

print("=== P8c F4: the warp rewrites only the active rows ===")
X_before = fl.X[:4].copy()
y_before = fl.y[:4].copy()
pt_new, optim_state, fl_new, action = warp_input(
    vbmc.vp, vbmc.optim_state, fl, vbmc.options
)
print("  warp action:", action)
print("  row : X before -> X after            | rewritten?")
for i in range(4):
    print(
        f"   {i} (flag {fl_new.X_flag[i]!s:5s}): {X_before[i]} ->"
        f" {fl_new.X[i]}  {not np.array_equal(X_before[i], fl_new.X[i])}"
    )
print("  what the new space wants for row 1:", pt_new(fl_new.X_orig[1:2])[0])
print(
    "  y row 1 before/after:",
    y_before[1],
    fl_new.y[1],
    " X_orig row 1 (untouched):",
    fl_new.X_orig[1],
)
print()

print("=== what reads inactive rows ===")
import subprocess

out = subprocess.run(
    [
        "git",
        "grep",
        "-n",
        "-E",
        r"function_logger\.(X|y)\b|self\.X\b|self\.y\b",
        "--",
        "pyvbmc/",
        ":!pyvbmc/testing/",
    ],
    cwd=r"C:\Users\luigi\Documents\GitHub\pyvbmc",
    capture_output=True,
    text=True,
)
for line in out.stdout.splitlines():
    if "X_flag" in line or "function_logger.py" in line:
        print("  ", line)
print()

print("=== the duplicate scan after the warp sees a stale row ===")
stale = fl_new.X[1].copy()
print("  a point equal to the stale row 1 in the NEW space:", stale)
f_val, f_sd, idx = fl_new.add(stale, -9.0)
print(
    "  -> matched idx =",
    idx,
    " X_flag[idx] =",
    fl_new.X_flag[idx],
    " n_evals[idx] =",
    fl_new.n_evals[idx, 0],
)
print()

print("=== P8i F11: the temperature divisor ===")
os2 = copy.deepcopy(vbmc.optim_state)
os2["temperature"] = 2.0
fl2 = copy.deepcopy(fl)
fl2.X_flag[:4] = True
pt2, os2, fl2, __ = warp_input(vbmc.vp, os2, fl2, vbmc.options)
dy = pt2.log_abs_det_jacobian(fl2.X[:4])
print("  warp_input wrote y = y_orig + dy/T:")
print(
    "   y_orig + dy/2 =",
    (fl2.y_orig[:4].ravel() + dy / 2)[:2],
    " stored y =",
    fl2.y[:4].ravel()[:2],
)
p = np.array([1.1, 1.3])
fl2.add(p, -3.0)
i = fl2.Xn
print(
    "  _record wrote y = y_orig + dy (no /T):",
    " stored",
    fl2.y[i, 0],
    " y_orig + dy =",
    fl2.y_orig[i, 0] + pt2.log_abs_det_jacobian(np.atleast_2d(fl2.X[i]))[0],
    " y_orig + dy/2 =",
    fl2.y_orig[i, 0]
    + pt2.log_abs_det_jacobian(np.atleast_2d(fl2.X[i]))[0] / 2,
)
print("  MATLAB funlogger_vbmc.m:243 and :269 both divide by T")
print(
    "  temperature key present in a default run's optim_state:",
    "temperature" in vbmc.optim_state,
)
print()

print("=== P8c F6: the low-correlation mask and NaN ===")
cov = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.02], [0.0, 0.02, 0.0]])
corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
thresh = 0.05
py_mask = np.abs(corr) <= thresh  # PyVBMC zeroes these
ml_keep = np.abs(corr) > thresh  # MATLAB keeps these,
ml_mask = ~ml_keep  # zeroes the complement
print("  corr =\n", corr)
print("  PyVBMC zeroes:\n", py_mask)
print("  MATLAB zeroes:\n", ml_mask)
print("  they differ where corr is NaN:", np.any(py_mask != ml_mask))
py_cov, ml_cov = cov.copy(), cov.copy()
py_cov[py_mask] = 0
ml_cov[ml_mask] = 0
print("  PyVBMC result:\n", py_cov)
print("  MATLAB result:\n", ml_cov)
print()

print("=== P8c F10: quantile conventions ===")
rng = np.random.default_rng(1)
s = rng.normal(size=100000)


def matlab_quantile(a, q):
    """quantile1.m / MATLAB quantile: interpolate at (i-0.5)/n."""
    a = np.sort(np.asarray(a))
    n = a.size
    pos = (np.arange(1, n + 1) - 0.5) / n
    return np.interp(q, pos, a, left=a[0], right=a[-1])


for q in (0.05, 0.95):
    print(
        f"  q={q}: numpy {np.quantile(s, q):.10f}"
        f"  MATLAB {matlab_quantile(s, q):.10f}"
        f"  diff {np.quantile(s, q) - matlab_quantile(s, q):.3e}"
    )
print(
    "  (both read the empirical distribution about half an order"
    " statistic apart; at n = 1e5 the gap is ~1e-5 of a sigma)"
)
print()

print("=== P8i F8: the reset block ===")
print("  keys warp_input clears:", "run_mean", "run_cov", "last_run_avg")
print(
    "  post-warp optim_state values:",
    optim_state["run_mean"],
    optim_state["run_cov"],
    optim_state["last_run_avg"],
)
print(
    "  hyp_dict keys that survive the warp:", sorted(k for k in vbmc.hyp_dict)
)
