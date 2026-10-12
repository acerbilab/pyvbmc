"""The five-variant grid over every stored oracle state with D >= 2."""

import csv
import sys
import time

import numpy as np
from harness import FIX, RUN_OF, VARIANTS, capture, run_once
from scipy.stats import binomtest

from pyvbmc.testing.oracles._state import snapshot_names

P = lambda *a: print(*a, flush=True)
OUT = sys.argv[1] if len(sys.argv) > 1 else "runs.csv"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEEDS = list(range(1, N_SEEDS + 1))

names = snapshot_names(FIX)
rows = []
caps = {}

P(f"states={names}")
P(f"seeds={SEEDS}")
P("")
P("=== states ===")
P(
    f"{'state':28s} {'D':>2s} {'K':>3s} {'N':>4s} {'Ns':>3s} {'noisy':>5s} "
    f"{'acq':12s} {'aniso':>8s} {'sigma0':>9s} {'maxfev':>7s} {'f(x0)':>12s}"
)
for n in names:
    c = capture(n)
    caps[n] = c
    ins = c["insigma"]
    P(
        f"{n:28s} {c['D']:2d} {c['K']:3d} {c['n_train']:4d} {c['Ns']:3d} "
        f"{str(c['noisy']):>5s} {c['acq']:12s} {ins.max()/ins.min():8.3f} "
        f"{c['sigma0']:9.4g} {int(c['cma_options']['maxfevals']):7d} {c['f_x0']:12.6g}"
    )
    P(f"{'':28s} insigma = {np.array2string(ins, precision=5)}")
P("")

t_start = time.perf_counter()
for n in names:
    c = caps[n]
    acq = c["acq_fun"]
    ins = c["insigma"]
    for seed in SEEDS:
        runs = {}
        for cfg in [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]:
            runs[cfg] = run_once(c, seed, *cfg)
        cur = runs[(False, False)]["bestever"]
        for v in VARIANTS:
            cfg, which = RUN_OF[v]
            r = runs[cfg]
            x = r[which]
            f = float(acq(x))
            rows.append(
                dict(
                    state=n,
                    D=c["D"],
                    seed=seed,
                    variant=v,
                    f_ret=f,
                    f_x0=c["f_x0"],
                    improved=int(f < c["f_x0"]),
                    n_evals=r["n_evals"],
                    n_iters=r["n_iters"],
                    wall=r["wall"],
                    dist_insigma=float(np.linalg.norm((x - cur) / ins)),
                    mean_used=int(r["mean_used"]),
                    lastgen_aligned=int(r["aligned"]),
                    stop=r["stop"],
                )
            )
        P(
            f"{n:28s} seed={seed:3d} "
            + " ".join(
                f"{v}={[r for r in rows if r['state']==n and r['seed']==seed and r['variant']==v][0]['f_ret']:.8g}"
                for v in VARIANTS
            )
        )
P(f"\ngrid wall time {time.perf_counter()-t_start:.1f}s, {len(rows)} records")

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
P(f"wrote {OUT}")


# ------------------------------------------------------------------ summary
def get(state, variant, field):
    return np.array(
        [
            r[field]
            for r in sorted(
                (
                    r
                    for r in rows
                    if r["state"] == state and r["variant"] == variant
                ),
                key=lambda r: r["seed"],
            )
        ],
        dtype=float,
    )


def iqr(a):
    q1, q3 = np.percentile(a, [25, 75])
    return q3 - q1


P("\n\n=== acquisition value at the returned point (lower is better) ===")
P(
    f"{'state':28s} {'variant':8s} {'median':>14s} {'IQR':>12s} "
    f"{'med diff vs current':>20s} {'IQR(diff)':>12s} {'lower/seeds':>12s} {'sign p':>8s} "
    f"{'med dist(insigma)':>18s}"
)
for n in names:
    base = get(n, "current", "f_ret")
    for v in VARIANTS:
        a = get(n, v, "f_ret")
        d = a - base
        lower = int(np.sum(d < 0))
        ties = int(np.sum(d == 0))
        nz = len(d) - ties
        p = binomtest(lower, nz).pvalue if nz else float("nan")
        P(
            f"{n:28s} {v:8s} {np.median(a):14.8g} {iqr(a):12.4g} "
            f"{np.median(d):20.4g} {iqr(d):12.4g} {lower:5d}/{len(d):<6d} "
            f"{p:8.3g} {np.median(get(n, v, 'dist_insigma')):18.4g}"
        )
    P("")

P("\n=== evaluations, wall time, improvement over the sieve ===")
P(
    f"{'state':28s} {'variant':8s} {'med evals':>10s} {'IQR':>8s} {'med wall(s)':>12s} "
    f"{'improved/seeds':>15s} {'med f(x0)-f_ret':>16s}"
)
for n in names:
    for v in VARIANTS:
        e = get(n, v, "n_evals")
        w_ = get(n, v, "wall")
        imp = get(n, v, "improved")
        gain = get(n, v, "f_x0") - get(n, v, "f_ret")
        P(
            f"{n:28s} {v:8s} {np.median(e):10.1f} {iqr(e):8.1f} {np.median(w_):12.4f} "
            f"{int(imp.sum()):6d}/{len(imp):<8d} {np.median(gain):16.6g}"
        )
    P("")

P("\n=== pooled over all states x seeds (paired by state and seed) ===")
P(
    f"{'variant':8s} {'lower than current':>19s} {'sign p':>10s} {'med paired diff':>16s} "
    f"{'med |diff|':>12s} {'med evals ratio':>16s} {'med wall ratio':>15s} "
    f"{'improved/total':>15s} {'med dist(insigma)':>18s}"
)
for v in VARIANTS:
    d, er, wr, di, im = [], [], [], [], []
    for n in names:
        b = get(n, "current", "f_ret")
        a = get(n, v, "f_ret")
        d += list(a - b)
        er += list(get(n, v, "n_evals") / get(n, "current", "n_evals"))
        wr += list(get(n, v, "wall") / get(n, "current", "wall"))
        di += list(get(n, v, "dist_insigma"))
        im += list(get(n, v, "improved"))
    d = np.array(d)
    lower = int(np.sum(d < 0))
    ties = int(np.sum(d == 0))
    nz = len(d) - ties
    p = binomtest(lower, nz).pvalue if nz else float("nan")
    P(
        f"{v:8s} {lower:6d}/{len(d):<12d} {p:10.3g} {np.median(d):16.5g} "
        f"{np.median(np.abs(d)):12.5g} {np.median(er):16.4f} {np.median(wr):15.4f} "
        f"{int(np.sum(im)):6d}/{len(im):<8d} {np.median(di):18.5g}"
    )
P("")
P("ties (identical returned acquisition value) per variant:")
for v in VARIANTS:
    d = np.concatenate(
        [get(n, v, "f_ret") - get(n, "current", "f_ret") for n in names]
    )
    P(
        f"  {v:8s} exact ties {int(np.sum(d == 0)):4d}/{len(d)}   |diff| < 1e-9: {int(np.sum(np.abs(d) < 1e-9))}"
    )
