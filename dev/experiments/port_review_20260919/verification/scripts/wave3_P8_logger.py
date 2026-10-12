"""Settles the FunctionLogger findings of slice P8.

P8i F4 / P8c F7 - a repeat whose evaluation time is unknown (the ``add``
    default ``np.nan``) turns the row's ``fun_eval_time`` into NaN for
    good; MATLAB's ``'add'`` passes ``t = 0`` (``misc/funlogger_vbmc.m:187``)
    so the same repeat averages to 0.  Second half: PyVBMC charges
    ``total_fun_eval_time`` inside ``_record``, so an ``add`` with an
    explicit time charges the total, where MATLAB charges it only in the
    ``'iter'``/``'single'`` branch (``:151``).
P8i F5 (first half) - a repeat at an input whose row has been deactivated
    is pooled into the dead row, which stays dead.
P8i F13 / P8c F8 - a repeat returns a ``(1,)`` array, a new point a scalar.
P8i F6 - ``add`` without an SD records ``S = 1`` at uncertainty level 2.
P8i F14 - ``batch_call`` writes NaN into ``S`` where ``__call__`` writes 1,
    when ``noise_flag`` is true at uncertainty level 0.
P8c F9 - the extreme-SD fallback of the duplicate pooling; the ordinary
    case reproduces MATLAB's three-line expression to the last bit.
"""

import numpy as np

import pyvbmc
from pyvbmc.function_logger import FunctionLogger

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()


def make(noise=False, level=0, vectorized=False, fun=None):
    if fun is None:
        fun = lambda x: -0.5 * float(np.sum(np.asarray(x) ** 2))
    return FunctionLogger(fun, 2, noise, level, vectorized_target=vectorized)


# ------------------------------------------------- F4 / P8c F7
print("=== F4: a repeat with an unknown time makes fun_eval_time NaN ===")
fl = make()
x = np.array([0.3, -0.2])
fl.add(x, -1.0, None, 0.5)  # a known time
print(
    "  after add(t=0.5): fun_eval_time[0] =",
    fl.fun_eval_time[0, 0],
    " total =",
    fl.total_fun_eval_time,
)
fl.add(x, -1.1)  # default t = NaN -> repeat
print(
    "  after add(default NaN): fun_eval_time[0] =",
    fl.fun_eval_time[0, 0],
    " total =",
    fl.total_fun_eval_time,
    " n_evals =",
    fl.n_evals[0, 0],
)
fl.add(x, -1.2, None, 2.0)  # a later known time cannot recover it
print(
    "  after add(t=2.0): fun_eval_time[0] =",
    fl.fun_eval_time[0, 0],
    " total =",
    fl.total_fun_eval_time,
)
print("  MATLAB would hold (N*t_prev + 0)/(N+1) at every 'add'")

print("  --- the new-row branch, by contrast ---")
fl2 = make()
fl2.add(np.array([1.0, 1.0]), -2.0)  # fresh row, NaN time
print(
    "  fresh row with the NaN default: fun_eval_time[0] =",
    fl2.fun_eval_time[0, 0],
    " total =",
    fl2.total_fun_eval_time,
)

print("  --- total_fun_eval_time charged by add (MATLAB does not) ---")
fl3 = make()
fl3.add(np.array([2.0, 2.0]), -3.0, None, 7.0)
print(
    "  add(..., fun_eval_time=7.0): total_fun_eval_time =",
    fl3.total_fun_eval_time,
    " (MATLAB: 0, 'add' never touches it)",
)
print()

# ------------------------------------------------- readers of the fields
print("=== readers of fun_eval_time / total_fun_eval_time ===")
import subprocess

out = subprocess.run(
    [
        "git",
        "grep",
        "-n",
        "-E",
        r"fun_eval_time|total_fun_eval_time",
        "--",
        "pyvbmc/",
        ":!pyvbmc/testing/",
    ],
    cwd=r"C:\Users\luigi\Documents\GitHub\pyvbmc",
    capture_output=True,
    text=True,
)
print(out.stdout)
print()

# ------------------------------------------------- F5 first half
print("=== F5: a repeat at a deactivated row is pooled into the dead row ===")
fl = make()
a = np.array([0.1, 0.1])
b = np.array([0.9, 0.9])
fl.add(a, -1.0)
fl.add(b, -2.0)
fl.X_flag[0] = False  # what the warm-up trim does
before = fl.func_count, fl.cache_count
f_val, f_sd, idx = fl.add(a, -1.5)
print(
    "  idx =",
    idx,
    " X_flag[0] =",
    fl.X_flag[0],
    " n_evals[0] =",
    fl.n_evals[0, 0],
    " y_orig[0] =",
    fl.y_orig[0, 0],
    " Xn =",
    fl.Xn,
)
print(
    "  cache_count before/after =",
    before[1],
    "/",
    fl.cache_count,
    " (a charged evaluation no consumer can see)",
)
print("  y_max (over X_flag) =", fl.y_max)
print()

# ------------------------------------------------- F13 / P8c F8
print("=== F13: return type, repeat vs new point ===")
fl = make()
v_new, _, _ = fl.add(np.array([0.5, 0.5]), -1.0)
v_rep, _, _ = fl.add(np.array([0.5, 0.5]), -1.0)
print("  new point:", type(v_new).__name__, np.shape(v_new), repr(v_new))
print("  repeat   :", type(v_rep).__name__, np.shape(v_rep), repr(v_rep))
flv = make(
    vectorized=True, fun=lambda X: -0.5 * np.sum(np.asarray(X) ** 2, axis=1)
)
vals, sds, idxs = flv.batch_call(np.array([[0.5, 0.5]]))
print("  batch_call value:", type(vals[0]).__name__, repr(vals[0]))
v_call, _, _ = flv(np.array([0.7, 0.7]))
print("  vectorized __call__ value:", type(v_call).__name__, repr(v_call))
print(
    "  the only consumer of the returned value is active_sample.py:808,"
    " guarded by n_evals[idx_new] == 1"
)
print()

# ------------------------------------------------- F6
print("=== F6: add() without an SD on a level-2 logger records S = 1 ===")
fl = make(noise=True, level=2, fun=lambda x: (-1.0, 0.01))
f_val, f_sd, idx = fl.add(np.array([0.2, 0.2]), -1.0)
print("  level 2, add(x, y) -> f_sd =", f_sd, " S[0] =", fl.S[0, 0])
fl1 = make(noise=True, level=1)
f_val, f_sd, idx = fl1.add(np.array([0.2, 0.2]), -1.0)
print("  level 1, add(x, y) -> f_sd =", f_sd, "  (the level-1 convention)")
print()

print("=== F6 reachability: options['f_vals'] with specify_target_noise ===")
target = lambda x: (-0.5 * float(np.sum(np.asarray(x) ** 2)), 0.01)
vbmc = pyvbmc.VBMC(
    target,
    np.array([[0.1, 0.1], [0.2, 0.2]]),
    np.array([[-5.0, -5.0]]),
    np.array([[5.0, 5.0]]),
    np.array([[-1.0, -1.0]]),
    np.array([[1.0, 1.0]]),
    options={"specify_target_noise": True, "f_vals": [-0.01, -0.04]},
)
print(
    "  uncertainty_handling_level =",
    vbmc.optim_state["uncertainty_handling_level"],
)
print("  cache y_orig =", vbmc.optim_state["cache"]["y_orig"])
print("  logger noise_flag =", vbmc.function_logger.noise_flag)
print("  -> active_sample.py:218 will call add(Xs[idx], ys[idx]) with no SD")
print()

print("=== F6: what precomputed_evaluations requires for the same case ===")
try:
    pyvbmc.VBMC(
        target,
        np.array([[0.1, 0.1]]),
        np.array([[-5.0, -5.0]]),
        np.array([[5.0, 5.0]]),
        np.array([[-1.0, -1.0]]),
        np.array([[1.0, 1.0]]),
        options={"specify_target_noise": True},
        precomputed_evaluations=(np.array([[0.3, 0.3]]), np.array([-0.05])),
    )
except ValueError as exc:
    print("  precomputed_evaluations without y_sd raises:", exc)
print()

# ------------------------------------------------- F14
print("=== F14: noise_flag true at uncertainty level 0 ===")
flv = make(
    noise=True,
    level=0,
    vectorized=True,
    fun=lambda X: -0.5 * np.sum(np.asarray(X) ** 2, axis=1),
)
vals, sds, idxs = flv.batch_call(np.array([[0.4, 0.4]]))
print("  batch_call sds =", sds, " S[0] =", flv.S[0, 0])
fls = make(noise=True, level=0)
v, sd, i = fls(np.array([0.4, 0.4]))
print("  sequential __call__ f_sd =", sd, " S[0] =", fls.S[0, 0])
print(
    "  VBMC sets noise_flag = (level > 0) at vbmc.py:465, so the pair is"
    " not built inside the package"
)
print()

# ------------------------------------------------- P8c F9
print("=== P8c F9: duplicate pooling, ordinary case vs MATLAB ===")
fl = make(noise=True, level=2, fun=lambda x: (-1.0, 1.0))
p = np.array([0.6, 0.6])
fl.add(p, -2.0, 2.0)
fl.add(p, -2.1, 0.5)
tau_n, tau_1 = 1 / 2.0**2, 1 / 0.5**2
m_y = (tau_n * -2.0 + tau_1 * -2.1) / (tau_n + tau_1)
m_s = 1 / np.sqrt(tau_n + tau_1)
print(f"  PyVBMC y_orig = {fl.y_orig[0, 0]!r}  S = {fl.S[0, 0]!r}")
print(f"  MATLAB  y_orig = {m_y!r}  S = {m_s!r}")
print("  bit-identical:", fl.y_orig[0, 0] == m_y and fl.S[0, 0] == m_s)

print("  --- extreme SDs, where the MATLAB expression overflows ---")
fl = make(noise=True, level=2, fun=lambda x: (-1.0, 1.0))
fl.add(p, -2.0, 1e-170)
fl.add(p, -2.1, 1e-160)
print(f"  PyVBMC y_orig = {fl.y_orig[0, 0]!r}  S = {fl.S[0, 0]!r}")
with np.errstate(all="ignore"):
    tn = 1 / np.float64(1e-170) ** 2
    t1 = 1 / np.float64(1e-160) ** 2
    print(
        f"  MATLAB tau_n = {tn!r}  tau_1 = {t1!r}"
        f"  -> y = {(tn * -2.0 + t1 * -2.1) / (tn + t1)!r}"
    )
