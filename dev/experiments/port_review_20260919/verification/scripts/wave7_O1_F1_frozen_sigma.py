"""O1 F1: optimize_sigma=False with optimize_lambd=True.

(a) _neg_elcbo on a reused posterior is not a function of theta alone:
    set_parameters multiplies the frozen sigma by rms(exp(theta_lambda)).
(b) On a fresh copy, G and H equal those of a transcription of MATLAB's
    negelcbo_vbmc.m, but the soft-bound term reads ln(sigma*nl) and its
    gradient misses d ln(nl)/d theta_lambda.
(c) The same objective inside scipy BFGS, reusing one posterior as
    optimize_vp does, against the MATLAB transcription.
(d) optimize_vp itself (K = 1, deterministic branch), with _neg_elcbo
    wrapped to record the frozen sigma it sees on each call.
(e) The mirror case, sigma free and lambda frozen at a non-unit RMS.
"""

import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import scipy as sp

sys.path.insert(0, str(Path(__file__).parent))
from wave7_O1_common import OPTS, banner, build_gp, central_fd, matlab_negelcbo

from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.options import Options
from pyvbmc.vbmc.variational_optimization import _neg_elcbo, optimize_vp

banner()
np.set_printoptions(precision=5, suppress=False, linewidth=120)

D, K = 2, 2
gp = build_gp(D, N=25, Ns=2, seed=11)
vp0 = VariationalPosterior(D, K, rng=0)
vp0.mu = np.array([[0.2, 0.6], [0.1, 0.5]])
vp0.sigma = np.array([[0.6, 0.4]])
vp0.lambd = np.array([[1.0], [1.0]])
vp0.optimize_sigma = False
bnd = vp0.get_bounds(gp.X, OPTS, K)
theta0 = vp0.get_parameters()  # (mu, ln lambd, eta)
print("theta layout: mu(4), ln lambd(2), eta(2); theta0 =", theta0)

print("\n(a) call-history dependence on one reused posterior")
vp = copy.deepcopy(vp0)
F_first = _neg_elcbo(theta0, gp, vp, 0.0, 0, False, False, bnd)[0]
theta_a = theta0.copy()
theta_a[D * K : D * K + D] += np.array([0.5, 0.3])
for i in range(5):
    _neg_elcbo(theta_a, gp, vp, 0.0, 0, False, False, bnd)
F_again = _neg_elcbo(theta0, gp, vp, 0.0, 0, False, False, bnd)[0]
print(f"  F(theta0), fresh posterior        = {F_first:.6f}")
print(f"  F(theta0), after 5 calls at theta_a = {F_again:.6f}")
print("  frozen sigma: start", vp0.sigma.ravel(), "-> now", vp.sigma.ravel())
nl_a = np.sqrt(np.mean(np.exp(theta_a[D * K : D * K + D]) ** 2))
print(
    f"  nl(theta_a) = {nl_a:.5f}; nl^5 = {nl_a**5:.5f}; "
    f"ratio sigma_now/sigma_0 = {(vp.sigma / vp0.sigma).ravel()}"
)
Fm0 = matlab_negelcbo(theta0, vp0, gp, bnd, False)
print(f"  MATLAB transcription F(theta0) = {Fm0:.6f} (any call history)")

print("\n(b) one call on a fresh copy, log-scale bound active")
theta_b = theta0.copy()
theta_b[D * K : D * K + D] = np.array([2.2, 0.9])
lnsc_true = (
    np.log(vp0.sigma.ravel())[None, :] + theta_b[D * K : D * K + D][:, None]
)
vpb = copy.deepcopy(vp0)
Fp, dFp = _neg_elcbo(theta_b, gp, vpb, 0.0, 0, True, False, bnd)[:2]
lnsc_read = (
    np.log(vpb.sigma.ravel())[None, :] + theta_b[D * K : D * K + D][:, None]
)
nl_b = np.sqrt(np.mean(np.exp(theta_b[D * K : D * K + D]) ** 2))
print("  ln-scale upper bound:", bnd["ub"][D * K : D * K + D])
print("  true ln scale (D x K):\n", lnsc_true)
print("  ln scale _vp_bound_loss reads:\n", lnsc_read)
print(f"  difference = ln nl = {np.log(nl_b):.5f}")
# G and H alone (no bounds), fresh copies
Gp_noB = _neg_elcbo(
    theta_b, gp, copy.deepcopy(vp0), 0.0, 0, False, False, None
)
vpm = copy.deepcopy(vp0)
Fm, dFm = matlab_negelcbo(theta_b, vp0, gp, bnd, True)
# MATLAB -G-H without bounds: strip bound terms by passing huge bounds
bnd_inf = dict(bnd)
bnd_inf["lb"] = np.full_like(bnd["lb"], -np.inf)
bnd_inf["ub"] = np.full_like(bnd["ub"], np.inf)
bnd_inf["weight_penalty"] = 0.0
Fm_noB = matlab_negelcbo(theta_b, vp0, gp, bnd_inf, False)
print(
    f"  -G-H: Python (fresh) {Gp_noB[0]:.12f}  MATLAB transcr. {Fm_noB:.12f}"
    f"  diff {Gp_noB[0]-Fm_noB:.2e}"
)
print(f"  F with bounds: Python {Fp:.6f}  MATLAB transcr. {Fm:.6f}")


def f_py_fresh(th):
    return _neg_elcbo(th, gp, copy.deepcopy(vp0), 0.0, 0, False, False, bnd)[0]


fd_py = central_fd(f_py_fresh, theta_b, 1e-5)
fd_m = central_fd(
    lambda th: matlab_negelcbo(th, vp0, gp, bnd, False), theta_b, 1e-5
)
print("  Python analytic dF :", dFp)
print("  FD of Python F     :", fd_py)
print("  MATLAB analytic dF :", dFm)
print("  FD of MATLAB F     :", fd_m)
print(f"  max|Py analytic - Py FD| = {np.max(np.abs(dFp - fd_py)):.3e}")
print(f"  max|M analytic - M FD|   = {np.max(np.abs(dFm - fd_m)):.3e}")

print(
    "\n(c) BFGS as optimize_vp's deterministic branch runs it (one reused vp0)"
)
theta_c0 = theta0.copy()
vpc = copy.deepcopy(vp0)
calls = {"n": 0}


def py_obj(th):
    calls["n"] += 1
    r = _neg_elcbo(th, gp, vpc, 0.0, 0, True, False, bnd)
    return r[0], r[1]


res_p = sp.optimize.minimize(py_obj, theta_c0, jac=True, tol=1e-3)
res_m = sp.optimize.minimize(
    lambda th: matlab_negelcbo(th, vp0, gp, bnd, True),
    theta_c0,
    jac=True,
    tol=1e-3,
)
print(
    f"  Python: {calls['n']} calls, sigma of reused vp0 now {vpc.sigma.ravel()}"
    f" (frozen value {vp0.sigma.ravel()})"
)
for name, r in (("Python", res_p), ("MATLAB", res_m)):
    th = r.x
    Ftrue = matlab_negelcbo(th, vp0, gp, bnd, False)
    print(
        f"  {name:6s} optimum: F_true(theta*) = {Ftrue:.6f}, "
        f"ln lambda* = {th[D*K:D*K+D]}, eta* = {th[-K:]}"
    )
# scale matrices sigma_k * lambda_d of the posterior each would return
lam_m = np.exp(res_m.x[D * K : D * K + D])
P_m = vp0.sigma.ravel()[None, :] * lam_m[:, None]
vp_ret = copy.deepcopy(vpc)
vp_ret.set_parameters(res_p.x)  # as optimize_vp:362 does on the fine copy
P_p = vp_ret.sigma.ravel()[None, :] * vp_ret.lambd.ravel()[:, None]
print(
    "  scale sigma_k*lambda_d, MATLAB:\n", P_m, "\n  Python returned:\n", P_p
)

print("\n(d) optimize_vp, K = 1, deterministic branch, frozen sigma")
base = Path(importlib.import_module("pyvbmc").__file__).parent
options = Options(
    base / "vbmc" / "option_configs" / "basic_vbmc_options.ini",
    evaluation_parameters={"D": D},
    user_options={},
)
options.load_options_file(
    base / "vbmc" / "option_configs" / "advanced_vbmc_options.ini",
    evaluation_parameters={"D": D},
)
vo = importlib.import_module("pyvbmc.vbmc.variational_optimization")
orig = vo._neg_elcbo
log = []


def wrapped(theta, gp_, vp_, *a, **k):
    s_before = vp_.sigma.copy().ravel()
    out = orig(theta, gp_, vp_, *a, **k)
    log.append((id(vp_), s_before, vp_.sigma.copy().ravel(), theta.copy()))
    return out


vo._neg_elcbo = wrapped
try:
    vp1 = VariationalPosterior(D, 1, rng=np.random.default_rng(3))
    vp1.mu = np.array([[0.2], [0.1]])
    vp1.sigma = np.array([[0.5]])
    vp1.optimize_sigma = False
    optim_state = {"warmup": False, "entropy_switch": False}
    out_vp, _, _ = optimize_vp(options, optim_state, vp1, gp, 3, 1)
finally:
    vo._neg_elcbo = orig
ids = [e[0] for e in log]
# the slow optimization's vp0 is the object called most often
main_id = max(set(ids), key=ids.count)
seq = [e for e in log if e[0] == main_id]
print(f"  calls on the reused vp0: {len(seq)}")
print(
    "  frozen sigma seen by call 1, 2, 3, ..., last:",
    [float(np.round(e[1][0], 5)) for e in seq[:4]],
    "...",
    float(np.round(seq[-1][2][0], 5)),
)
th_last = seq[-1][3]
sig_start = seq[0][1][0]
print(
    f"  sigma at start of the slow optimization {sig_start:.5f}; "
    f"returned vp sigma {out_vp.sigma.item():.5f}, lambda {out_vp.lambd.ravel()}"
)
print(
    f"  a MATLAB run would return sigma = sigma_start * nl(theta_final) ="
    f" {sig_start*np.sqrt(np.mean(np.exp(th_last[D:D+D])**2)):.5f}"
    f" (theta_final = last evaluated theta)"
)

print("\n(e) mirror case: sigma free, lambda frozen with RMS != 1, fresh vp")
vpe0 = VariationalPosterior(D, K, rng=0)
vpe0.mu = np.array([[0.2, 0.6], [0.1, 0.5]])
vpe0.sigma = np.array([[0.6, 0.4]])
vpe0.lambd = np.array([[3.0], [1.5]])  # RMS 2.37
vpe0.optimize_lambd = False
bnde = vpe0.get_bounds(gp.X, OPTS, K)
th_e = np.concatenate(
    [vpe0.mu.ravel(order="F"), np.log([0.6, 0.4]), [0.0, 0.0]]
)
vpe = copy.deepcopy(vpe0)
Fe1 = _neg_elcbo(th_e, gp, vpe, 0.0, 0, False, False, bnde)[0]
Fe2 = _neg_elcbo(th_e, gp, vpe, 0.0, 0, False, False, bnde)[0]
Fme = matlab_negelcbo(th_e, vpe0, gp, bnde, False)
print(
    f"  F first call {Fe1:.6f}, second call {Fe2:.6f}, MATLAB transcr. {Fme:.6f}"
)
print(
    "  lambda after the first call:",
    vpe.lambd.ravel(),
    "(frozen value",
    vpe0.lambd.ravel(),
    ")",
)
