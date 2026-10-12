"""Do the soft optimization bounds of the variational posterior expand in
practice, and does rebuilding them every iteration ever bind?

`VariationalPosterior.get_bounds` is written to accumulate the box of the
training inputs over calls, but every posterior `optimize_vp` returns has
`bounds = None`, so each iteration rebuilds the box from the current GP
training set. This script runs short seeded VBMC fits and records, at every
`optimize_vp` call: the box rebuilt from the current training set (what
PyVBMC uses), the box accumulated over the run (what MATLAB VBMC's
`vpbounds.m` keeps), and afterwards where the fitted posterior sits relative
to each: the largest overshoot of a component mean beyond the box, and the
margin of the largest log scale below its upper bound, in units of the box
width (the soft-bound loss is zero inside the box).

Usage: python -u soft_bounds_trace.py
"""

import logging

import numpy as np

import pyvbmc.vbmc.variational_optimization as vo
from pyvbmc import VBMC

logging.disable(logging.CRITICAL)


def rosenbrock(x):
    x = np.atleast_2d(x)
    return float(
        -np.sum(
            100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2
        )
        - 0.5 * np.sum(x**2) / 9.0
    )


def two_blobs(x):
    x = np.atleast_2d(x)
    a = -0.5 * np.sum((x - 2.0) ** 2) / 0.3**2
    b = -0.5 * np.sum((x + 2.0) ** 2) / 0.6**2
    return float(np.logaddexp(a, b))


targets = {"rosenbrock_D2": (rosenbrock, 2), "two_blobs_D3": (two_blobs, 3)}
orig_optimize_vp = vo.optimize_vp

for name, (fun, D) in targets.items():
    print(f"\n================ {name} ================", flush=True)
    acc = {"lb": np.full(D, np.inf), "ub": np.full(D, -np.inf)}
    acc_ln_ub = np.full(D, -np.inf)
    rows = []

    def traced(options, optim_state, vp, gp, *args, **kwargs):
        X = gp.X
        lb, ub = X.min(0), X.max(0)
        ln_ub = np.log(ub - lb)
        acc["lb"] = np.minimum(acc["lb"], lb)
        acc["ub"] = np.maximum(acc["ub"], ub)
        np.maximum(acc_ln_ub, ln_ub, out=acc_ln_ub)
        out = orig_optimize_vp(options, optim_state, vp, gp, *args, **kwargs)
        v = out[0]
        mu = v.mu  # (D, K)
        width = ub - lb
        over_now = np.max(
            np.maximum(lb[:, None] - mu, mu - ub[:, None]) / width[:, None]
        )
        acc_w = acc["ub"] - acc["lb"]
        over_acc = np.max(
            np.maximum(acc["lb"][:, None] - mu, mu - acc["ub"][:, None])
            / acc_w[:, None]
        )
        ln_scale = np.log(v.sigma * v.lambd)  # (D, K)
        scale_margin_now = np.min(ln_ub[:, None] - ln_scale)
        scale_margin_acc = np.min(acc_ln_ub[:, None] - ln_scale)
        rows.append(
            (
                X.shape[0],
                bool(optim_state.get("warmup")),
                float(np.max(acc_w / width)),
                float(over_now),
                float(over_acc),
                float(scale_margin_now),
                float(scale_margin_acc),
                v.K,
            )
        )
        return out

    vo.optimize_vp = traced
    import pyvbmc.vbmc.vbmc as vbmc_module

    vbmc_module.optimize_vp = traced
    try:
        lb = np.full((1, D), -10.0)
        ub = np.full((1, D), 10.0)
        vbmc = VBMC(
            fun,
            np.zeros((1, D)),
            lb,
            ub,
            np.full((1, D), -3.0),
            np.full((1, D), 3.0),
            options={"display": "off"},
            seed=3,
        )
        vbmc.optimize()
    finally:
        vo.optimize_vp = orig_optimize_vp
        vbmc_module.optimize_vp = orig_optimize_vp

    print(
        "call  N_train warmup  acc/now_width  mean_overshoot(now, acc)  "
        "lnscale_margin(now, acc)  K",
        flush=True,
    )
    for i, r in enumerate(rows):
        print(
            f"{i:4d}  {r[0]:7d} {str(r[1]):>6s}  {r[2]:13.3f}  "
            f"{r[3]:+10.3f} {r[4]:+10.3f}    {r[5]:+9.3f} {r[6]:+9.3f}  {r[7]:3d}",
            flush=True,
        )
    a = np.array([r[2:7] for r in rows])
    print(
        f"\nmax ratio of accumulated to rebuilt box width: {a[:, 0].max():.3f}; "
        f"calls where they differ: {int(np.sum(a[:, 0] > 1 + 1e-12))} of {len(rows)}",
        flush=True,
    )
    print(
        f"largest overshoot of a component mean beyond the rebuilt box: "
        f"{a[:, 1].max():+.3f} box widths (positive means outside)",
        flush=True,
    )
    print(
        f"smallest margin of a log scale below its rebuilt upper bound: "
        f"{a[:, 3].min():+.3f} (negative means above the bound)",
        flush=True,
    )
