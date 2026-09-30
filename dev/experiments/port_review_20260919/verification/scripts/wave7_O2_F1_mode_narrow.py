"""F1: vp.mode() on narrow posteriors, both values of orig_flag.

The optimizer call inside mode() is wrapped in-process (the module's
`minimize` name is rebound for the duration of the script; no file is
touched) to record the start, the result, and the objective values seen.
The true mode is found independently by BFGS on the log-sum-exp density in
coordinates scaled by the posterior's size, started from the same point.
Warnings are recorded with the "always" filter to see what a user would be
shown.
"""
import importlib
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize as _minimize
from wave7_O2_common import banner, build_vp, ref_logq_and_grad

from pyvbmc.parameter_transformer import ParameterTransformer

banner()
np.set_printoptions(precision=4, linewidth=150)
VPM = importlib.import_module(
    "pyvbmc.variational_posterior.variational_posterior"
)
records = []


def traced(fun, x0, **kw):
    seen = []

    def f(x):
        out = fun(x)
        v = out[0] if isinstance(out, tuple) else out
        seen.append(float(np.ravel(v)[0]))
        return out

    res = _minimize(f, x0, **kw)
    records.append(dict(x0=np.array(x0), res=res, seen=seen, kw=kw))
    return res


VPM.minimize = traced


def true_mode_transformed(vp, u0):
    """Mode of q in the transformed space, by BFGS in scaled coordinates."""
    mu, s, lam, w = vp.mu, vp.sigma, vp.lambd.ravel(), vp.w
    c = np.min(s) * lam  # per-coordinate scale

    def f(y):
        u = y * c
        lq, g = ref_logq_and_grad(u[None, :], mu, s, lam, w)
        return -lq[0], -(g[0] * c)

    r = _minimize(f, u0 / c, jac=True, method="BFGS", options={"gtol": 1e-10})
    return r.x * c, -r.fun


def run_case(label, vp, orig_flag, scale_note):
    records.clear()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = vp.mode(orig_flag=orig_flag)
    wmsgs = sorted({f"{r.category.__name__}: {r.message}" for r in rec})
    # all optimizations of this call
    n_inf = [sum(1 for v in r["seen"] if not np.isfinite(v)) for r in records]
    nits = [r["res"].nit for r in records]
    succ = [bool(r["res"].success) for r in records]
    msgs = sorted({str(r["res"].message) for r in records})
    moved = [np.max(np.abs(r["res"].x - r["x0"])) for r in records]
    method = (
        "L-BFGS-B" if records[0]["kw"].get("bounds") is not None else "BFGS"
    )
    # compare with the true mode (transformed-space mode for orig_flag=False;
    # for orig_flag=True with an identity transform the two coincide)
    out = f"{label:44s} {method:8s} nit={nits} success={succ} inf-evals={n_inf} moved={np.array(moved)}"
    if (
        not orig_flag
        or vp.parameter_transformer.type.sum() == 0
        and np.all(vp.parameter_transformer.delta == 1)
    ):
        u_code = m if not orig_flag else m
        u_true, lq_true = true_mode_transformed(vp, u_code)
        c = np.min(vp.sigma) * vp.lambd.ravel()
        lq_code = ref_logq_and_grad(
            u_code[None, :], vp.mu, vp.sigma, vp.lambd, vp.w
        )[0][0]
        out += (
            f" |mode-true|/scale={np.max(np.abs(u_code - u_true) / c):.2e}"
            f" dlogq={lq_true - lq_code:.2e}"
        )
    print(out, flush=True)
    print(
        f"{'':44s} {scale_note}; messages {msgs}; warnings {wmsgs}", flush=True
    )
    return m


# --- A: default (identity) transformer, D = 2, K = 3 ----------------------
rs = np.random.default_rng(3)
base_mu = rs.normal(0, 0.6, (2, 3))
print("\nA. identity transformer, D=2, K=3, lambda = scale", flush=True)
for scale in (0.3, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001):
    for orig_flag in (False, True):
        vp = build_vp(
            base_mu * scale,
            [1.0, 1.3, 0.8],
            np.full(2, scale),
            [0.4, 0.33, 0.27],
        )
        run_case(
            f"scale {scale:g}, orig_flag={orig_flag}",
            vp,
            orig_flag,
            f"sigma*lambda in [{0.8*scale:.3g}, {1.3*scale:.3g}]",
        )

# --- B: unbounded variables with plausible bounds [-5, 5] ----------------
print(
    "\nB. unbounded variables, plausible box [-5, 5] (transformed unit = "
    "10 original units), D=2, K=3",
    flush=True,
)
ptB = ParameterTransformer(
    2,
    np.full((1, 2), -np.inf),
    np.full((1, 2), np.inf),
    np.full((1, 2), -5.0),
    np.full((1, 2), 5.0),
)
print("   transformer delta:", ptB.delta, flush=True)
for sd_orig in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
    s = sd_orig / 10.0  # posterior scale in transformed units
    for orig_flag in (False, True):
        vp = build_vp(
            base_mu * s + 0.1,
            [1.0, 1.3, 0.8],
            np.full(2, s),
            [0.4, 0.33, 0.27],
            transformer=ptB,
        )
        m = run_case(
            f"sd_orig {sd_orig:g} ({sd_orig/10:.1%} of box), "
            f"orig={orig_flag}",
            vp,
            orig_flag,
            f"posterior SD ~{sd_orig:g} original units",
        )
        if orig_flag:
            u_true, _ = true_mode_transformed(vp, ptB(m[None, :]).ravel())
            # orig-space mode of a linear transform equals the image of the
            # transformed-space mode (constant Jacobian)
            x_true = ptB.inverse(u_true[None, :]).ravel()
            print(
                f"{'':44s} orig-space |mode - true| / SD = "
                f"{np.max(np.abs(m - x_true)) / sd_orig:.2e}",
                flush=True,
            )

# --- C: bounded variables [0, 10], plausible [1, 9], orig_flag=True -------
print(
    "\nC. bounded variables [0, 10], plausible [1, 9], logit transform, "
    "D=2, K=3, orig_flag=True",
    flush=True,
)
ptC = ParameterTransformer(
    2,
    np.zeros((1, 2)),
    np.full((1, 2), 10.0),
    np.ones((1, 2)),
    np.full((1, 2), 9.0),
)
for s in (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001):
    vp = build_vp(
        base_mu * s + 0.05,
        [1.0, 1.3, 0.8],
        np.full(2, s),
        [0.4, 0.33, 0.27],
        transformer=ptC,
    )
    xs, _ = vp.sample(20000, orig_flag=True)
    sd = np.std(xs, axis=0)
    run_case(
        f"transformed scale {s:g}",
        vp,
        True,
        f"posterior SD in original units {sd}",
    )

# --- D: one bounded [0, 1] (plausible [0.05, 0.5]) and one unbounded
# variable (plausible [-5, 5]); not every variable boxed ------------------
print(
    "\nD. x1 in [0, 1] (plausible [0.05, 0.5]), x2 unbounded (plausible "
    "[-5, 5]), D=2, K=3, orig_flag=True",
    flush=True,
)
ptD = ParameterTransformer(
    2,
    np.array([[0.0, -np.inf]]),
    np.array([[1.0, np.inf]]),
    np.array([[0.05, -5.0]]),
    np.array([[0.5, 5.0]]),
)
for s in (0.1, 0.03, 0.01, 0.003, 0.001):
    vp = build_vp(
        base_mu * s,
        [1.0, 1.3, 0.8],
        np.full(2, s),
        [0.4, 0.33, 0.27],
        transformer=ptD,
    )
    xs, _ = vp.sample(20000, orig_flag=True)
    sd = np.std(xs, axis=0)
    run_case(
        f"transformed scale {s:g}",
        vp,
        True,
        f"posterior SD in original units {sd}",
    )
VPM.minimize = _minimize
