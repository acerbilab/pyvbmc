"""F1: is the underflow what stops the refinement?

For case A of O2_F1_mode_narrow.py at scale 0.003, repeat the exact
optimizer calls of mode() (L-BFGS-B with infinite bounds and no gradient
for orig_flag=True; BFGS with the analytic gradient for orig_flag=False)
from the same start, once on the code's log density and once on a
log-sum-exp log density with the log-sum-exp gradient.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
from wave7_O2_common import banner, build_vp, ref_logq_and_grad

banner()
warnings.simplefilter("ignore")
rs = np.random.default_rng(3)
base_mu = rs.normal(0, 0.6, (2, 3))
scale = 0.003
vp = build_vp(
    base_mu * scale, [1.0, 1.3, 0.8], np.full(2, scale), [0.4, 0.33, 0.27]
)
mu, s, lam, w = vp.mu, vp.sigma, vp.lambd, vp.w

# the starts mode() uses: captured from its own optimizer calls
import importlib

VPM = importlib.import_module(
    "pyvbmc.variational_posterior.variational_posterior"
)
starts = {}


def capture(flag):
    got = []

    def fake(fun, x0, **kw):
        got.append(np.array(x0))
        return minimize(fun, x0, **kw)

    VPM.minimize = fake
    vp.mode(orig_flag=flag, n_opts=1)
    VPM.minimize = minimize
    return got[0]


starts[True] = capture(True)
starts[False] = capture(False)
print(
    "starts / scale:", starts[True] / scale, starts[False] / scale, flush=True
)
x0 = starts[True]


def code_f(x):
    return -vp.pdf(x, orig_flag=True, log_flag=True)


def code_fg(x):
    y, dy = vp.pdf(x, orig_flag=False, log_flag=True, grad_flag=True)
    return -y, -dy


def lse_f(x):
    return -ref_logq_and_grad(np.atleast_2d(x), mu, s, lam, w)[0]


def lse_fg(x):
    lq, g = ref_logq_and_grad(np.atleast_2d(x), mu, s, lam, w)
    return -lq[0], -g[0]


bounds = np.array([[-np.inf, np.inf]] * 2)
ref = minimize(lse_fg, x0, method="BFGS", jac=True, options={"gtol": 1e-12})
print("reference mode / scale:", ref.x / scale, flush=True)
for name, f, kw in [
    ("orig_flag=True  code", code_f, dict(bounds=bounds, jac=False)),
    ("orig_flag=True  lse ", lse_f, dict(bounds=bounds, jac=False)),
    ("orig_flag=False code", code_fg, dict(bounds=None, jac=True)),
    ("orig_flag=False lse ", lse_fg, dict(bounds=None, jac=True)),
]:
    x0 = starts["True" in name]
    r = minimize(fun=f, x0=x0, **kw)
    print(
        f"{name}: moved {np.linalg.norm(r.x - x0)/scale:.3e} scale units, "
        f"|x - mode|/scale {np.max(np.abs(r.x - ref.x))/scale:.2e}, "
        f"nit {r.nit}, success {r.success}, {r.message}",
        flush=True,
    )
