"""Post-checks: instrumentation inertness, the B anomaly, scale-free summaries."""

import csv
import sys

import cma
import numpy as np
from harness import FIX, RUN_OF, VARIANTS, capture, run_once

from pyvbmc.testing.oracles._state import snapshot_names

P = lambda *a: print(*a, flush=True)
CSV = sys.argv[1]

names = snapshot_names(FIX)

# --- 1. is the instrumentation (obj wrapper + callback) inert? ------------
P("=== instrumentation check: wrapped+callback run vs a bare cma.fmin ===")
for n in ["normal_D2_warmup", "cigar_D4_boosted", "rosenbrock_D2_noise1_viqr"]:
    c = capture(n)
    for seed in (1, 2, 3):
        r = run_once(c, seed, False, False)
        rng = np.random.default_rng(seed)
        import pyvbmc.vbmc.active_sample  # noqa

        asmod = sys.modules["pyvbmc.vbmc.active_sample"]
        opts = dict(c["cma_options"])
        opts["randn"] = lambda *sh: rng.standard_normal(sh)
        bare = cma.fmin(
            c["acq_fun"],
            c["x0"],
            c["sigma0"],
            options=opts,
            parallel_objective=c["acq_fun"],
            noise_handler=asmod._BatchedNoiseHandler(
                c["x0"].size, c["acq_fun"], rng
            ),
        )
        same_x = np.array_equal(np.asarray(bare[0], float), r["bestever"])
        P(
            f"  {n:28s} seed={seed} identical x_best={same_x} "
            f"evals {int(bare[3])} vs {r['n_evals']} iters {int(bare[4])} vs {r['n_iters']}"
        )

# --- 2. read the grid ------------------------------------------------------
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
for r in rows:
    for k in ("f_ret", "f_x0", "wall", "dist_insigma"):
        r[k] = float(r[k])
    for k in (
        "seed",
        "improved",
        "n_evals",
        "n_iters",
        "mean_used",
        "lastgen_aligned",
    ):
        r[k] = int(r[k])


def sel(state, variant, field):
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


P(
    "\n=== last-generation alignment (best index == argmin of the generation) ==="
)
bad = [r for r in rows if not r["lastgen_aligned"]]
P(f"  misaligned records: {len(bad)} / {len(rows)}")

P("\n=== where variant B returns a strictly lower value than current ===")
for n in names:
    b, cur = sel(n, "B", "f_ret"), sel(n, "current", "f_ret")
    d = b - cur
    idx = np.where(d < 0)[0]
    for i in idx:
        P(
            f"  {n} seed={i+1} f_B-f_current = {d[i]:.3e} (relative {d[i]/abs(cur[i]):.2e})"
        )
P("  (B takes a point cma also evaluated, so by the bookkept values it can")
P("   never beat the best-ever; a flip can only come from the batched-vs-")
P("   pointwise evaluation gap of the acquisition.)")

P("\n=== how often each variant's point differs from current's at all ===")
P(f"{'state':28s} " + " ".join(f"{v:>9s}" for v in VARIANTS))
for n in names:
    P(
        f"{n:28s} "
        + " ".join(
            f"{int(np.sum(sel(n, v, 'dist_insigma') > 0)):9d}"
            for v in VARIANTS
        )
    )

P(
    "\n=== scale-free view: paired difference as a fraction of the search's own gain ==="
)
P(
    "gain = median over seeds of f(x0) - f_current  (how much the local search buys)"
)
P(
    f"{'state':28s} {'gain':>12s} {'seed IQR of':>13s} "
    + " ".join(f"{'d'+v:>12s}" for v in VARIANTS[1:])
)
P(
    f"{'':28s} {'':12s} {'f_current':>13s} "
    + " ".join(f"{'/gain':>12s}" for v in VARIANTS[1:])
)
for n in names:
    cur = sel(n, "current", "f_ret")
    gain = float(np.median(sel(n, "current", "f_x0") - cur))
    q1, q3 = np.percentile(cur, [25, 75])
    line = f"{n:28s} {gain:12.5g} {q3-q1:13.4g} "
    for v in VARIANTS[1:]:
        d = float(np.median(sel(n, v, "f_ret") - cur))
        line += f"{d/abs(gain):12.4g} "
    P(line)

P(
    "\n=== evaluations and wall time relative to current (median of the ratio) ==="
)
P(
    f"{'state':28s} "
    + " ".join(f"{v+' ev':>10s} {v+' t':>8s}" for v in VARIANTS[1:])
)
for n in names:
    line = f"{n:28s} "
    for v in VARIANTS[1:]:
        er = np.median(sel(n, v, "n_evals") / sel(n, "current", "n_evals"))
        wr = np.median(sel(n, v, "wall") / sel(n, "current", "wall"))
        line += f"{er:10.3f} {wr:8.3f}"
    P(line)

P("\n=== budget use: median evals vs maxfevals, and the stop reason mix ===")
for n in names:
    c_stop = {}
    for r in rows:
        if r["state"] == n:
            c_stop[(r["variant"], r["stop"])] = (
                c_stop.get((r["variant"], r["stop"]), 0) + 1
            )
    P(
        f"  {n:28s} "
        + "; ".join(f"{k[0]}:{k[1]}x{v}" for k, v in sorted(c_stop.items()))
    )

P(
    "\n=== how often MATLAB's rule takes the final mean rather than the last-gen best ==="
)
for n in names:
    P(
        f"  {n:28s} B {int(sel(n,'B','mean_used').sum())}/20   SNB {int(sel(n,'SNB','mean_used').sum())}/20"
    )
