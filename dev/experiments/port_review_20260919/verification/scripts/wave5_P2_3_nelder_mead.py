"""P2i F4: the Nelder-Mead branch of the acquisition search.

Settles: (i) no ``bounds`` reach ``scipy.optimize.minimize``; (ii) no
``search_max_fun_evals``; (iii) ``tol`` is turned into both ``xatol`` and
``fatol`` by SciPy; (iv) the acquisition's hard-bound mask with an
unbounded variable, where ``lb_eps_orig`` is ``nan``; (v) whether the
branch can leave the search box.
"""
import importlib
import sys

import numpy as np
import scipy.optimize as sopt

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from _common import banner, build, set_acq, with_gp  # noqa: E402

from pyvbmc.acquisition_functions import AbstractAcqFcn  # noqa: E402
from pyvbmc.vbmc.active_sample import active_sample  # noqa: E402

banner()
asmod = importlib.import_module("pyvbmc.vbmc.active_sample")


class Linear(AbstractAcqFcn):
    """Monotone in the first coordinate: the simplex never stops improving."""

    def _compute_acquisition_function(self, Xs, *args):
        return np.atleast_2d(Xs)[:, 0].copy()


print("\n=== (i)-(iii) the kwargs handed to scipy.optimize.minimize ===")
captured = {}
real_minimize = sopt.minimize


def spy_minimize(fun, x0, *args, **kwargs):
    captured.update({"args": args, "kwargs": dict(kwargs), "x0": np.array(x0)})
    return real_minimize(fun, x0, *args, **kwargs)


v = build(2, {"search_optimizer": "Nelder-Mead"})
gp, fl, os_ = with_gp(v)
set_acq(v, Linear())
sopt.minimize = spy_minimize
try:
    fl, os_, _, gp = active_sample(
        gp, 1, os_, fl, v.iteration_history, v.vp, v.options
    )
finally:
    sopt.minimize = real_minimize
print("  positional args:", captured["args"])
print("  keyword args   :", captured["kwargs"])
print("  'bounds' passed:", "bounds" in captured["kwargs"])
print("  'options' passed:", "options" in captured["kwargs"])
print("  search_max_fun_evals option =", v.options["search_max_fun_evals"])
print("  lb_search:", os_["lb_search"], " ub_search:", os_["ub_search"])
x_tran = fl.X[fl.Xn]
print("  acquired (transformed):", x_tran)
print(
    "  inside the search box:",
    bool(
        np.all(x_tran >= os_["lb_search"])
        and np.all(x_tran <= os_["ub_search"])
    ),
)

print("\n=== (iii) how SciPy turns tol into xatol/fatol ===")
seen = {}
real_nm = sopt._optimize._minimize_neldermead


def spy_nm(func, x0, args=(), callback=None, **unknown):
    seen.update(unknown)
    return real_nm(func, x0, args, callback, **unknown)


sopt._optimize._minimize_neldermead = spy_nm
try:
    real_minimize(
        lambda x: float(np.sum(x**2)),
        np.zeros(2),
        method="Nelder-Mead",
        tol=1e-2,
    )
finally:
    sopt._optimize._minimize_neldermead = real_nm
print(
    "  _minimize_neldermead received:",
    {k: seen[k] for k in ("xatol", "fatol", "maxiter", "maxfev") if k in seen},
)
print("  scipy version:", __import__("scipy").__version__)

print(
    "\n=== (iv) the acquisition hard-bound mask with an unbounded variable ==="
)
print(
    "  lb_eps_orig:",
    v.optim_state["lb_eps_orig"],
    " ub_eps_orig:",
    v.optim_state["ub_eps_orig"],
)
X_probe = np.array([[1e30, 1e30], [-1e30, -1e30]])
print(
    "  (x < lb_eps_orig).any(axis=1):",
    np.any(X_probe < v.optim_state["lb_eps_orig"], axis=1),
)
print(
    "  (x > ub_eps_orig).any(axis=1):",
    np.any(X_probe > v.optim_state["ub_eps_orig"], axis=1),
)

vb = build(
    2,
    {"search_optimizer": "Nelder-Mead"},
    lb=np.array([[-10.0, -np.inf]]),
    ub=np.array([[10.0, np.inf]]),
)
print(
    "  mixed bounds: lb_eps_orig:",
    vb.optim_state["lb_eps_orig"],
    " ub_eps_orig:",
    vb.optim_state["ub_eps_orig"],
)
Xm = np.array([[11.0, 1e30], [0.0, 1e30]])
print(
    "  mask on [[11, 1e30], [0, 1e30]]:",
    np.logical_or(
        np.any(Xm < vb.optim_state["lb_eps_orig"], axis=1),
        np.any(Xm > vb.optim_state["ub_eps_orig"], axis=1),
    ),
)

print("\n=== (iv-b) does the mask bite for a BOUNDED variable? ===")
pt = vb.parameter_transformer
far = np.array([[1e3, 1e3], [-1e3, -1e3]])
print("  inverse of a runaway transformed point:", pt.inverse(far))
print(
    "  masked:",
    np.logical_or(
        np.any(pt.inverse(far) < vb.optim_state["lb_eps_orig"], axis=1),
        np.any(pt.inverse(far) > vb.optim_state["ub_eps_orig"], axis=1),
    ),
)

print("\n=== (iv-c) SciPy's own default budget for Nelder-Mead ===")
res = real_minimize(lambda x: float(x[0]), np.zeros(2), method="Nelder-Mead")
print(
    "  D=2 unbounded linear objective: nfev =",
    res.nfev,
    " nit =",
    res.nit,
    " status:",
    res.message,
)

print("\n=== (v) CMA-ES on the same state, for contrast ===")
v2 = build(2, {"search_optimizer": "cmaes"})
gp2, fl2, os2 = with_gp(v2)
set_acq(v2, Linear())
fl2, os2, _, gp2 = active_sample(
    gp2, 1, os2, fl2, v2.iteration_history, v2.vp, v2.options
)
x2 = fl2.X[fl2.Xn]
print(
    "  acquired (transformed):",
    x2,
    " inside search box:",
    bool(np.all(x2 >= os2["lb_search"]) and np.all(x2 <= os2["ub_search"])),
)
