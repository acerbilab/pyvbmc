"""Read the records of `wave6_A1d_mean_bounds_probe.py`, the pooled and the
per-column form side by side.

Per target and hyperparameter, over every vector returned by every fit of
the recorded runs: the median, and the fraction that lie within 1% of the
box's width of each hard bound in effect, in each form; and, for the pooled
form, the fraction of the mean's location and log scale that lie outside
gplite's per-column box, split by part of the run (thirds by the number of
training points) with the median distance beyond it (in column widths for
the location, in log units for the log scale).

Usage: python wave6_A1d_read.py POOLED.npz PER_COLUMN.npz
"""

import collections
import sys

import numpy as np

NEAR = 0.01


def load(path):
    z = np.load(path)
    fits = collections.defaultdict(dict)
    for key in z.files:
        if key == "form":
            continue
        i, name = key.split("/", 1)
        fits[int(i)][name] = z[key]
    by_label = collections.defaultdict(list)
    for i in sorted(fits):
        by_label[str(fits[i]["label"])].append(fits[i])
    return str(z["form"]), by_label


def stack(rows, key):
    return np.concatenate([np.atleast_2d(r[key]) for r in rows])


def bounds_stats(rows):
    hyp = stack(rows, "hyp")
    lb = np.concatenate(
        [np.broadcast_to(r["lb"], r["hyp"].shape) for r in rows]
    )
    ub = np.concatenate(
        [np.broadcast_to(r["ub"], r["hyp"].shape) for r in rows]
    )
    near = NEAR * (ub - lb)
    return (
        np.median(hyp, 0),
        (hyp <= lb + near).mean(0),
        (hyp >= ub - near).mean(0),
        hyp.shape[0],
    )


form_a, pooled = load(sys.argv[1])
form_b, per_column = load(sys.argv[2])
assert (form_a, form_b) == ("pooled", "per_column"), (form_a, form_b)

for label in pooled:
    rows_a, rows_b = pooled[label], per_column.get(label, [])
    seeds = sorted({int(r["seed"]) for r in rows_a})
    print(
        f"\n== {label}, seeds {seeds}: {len(rows_a)} fits pooled, "
        f"{len(rows_b)} per column"
    )
    names = rows_a[0]["names"]
    med_a, lo_a, hi_a, n_a = bounds_stats(rows_a)
    med_b, lo_b, hi_b, n_b = bounds_stats(rows_b)
    print(
        f"  {'hyperparameter':32s} {'median':>17s}   "
        f"{'at lower bound':>15s}   {'at upper bound':>15s}"
        f"   (pooled -> per column; {n_a} and {n_b} vectors)"
    )
    for j, name in enumerate(names):
        print(
            f"  {name:32s} {med_a[j]:7.2f} -> {med_b[j]:7.2f}   "
            f"{lo_a[j]:5.2f} -> {lo_b[j]:5.2f}   "
            f"{hi_a[j]:5.2f} -> {hi_b[j]:5.2f}"
        )

    if "xm_index" not in rows_a[0]:
        continue
    Ns = np.array([int(r["N"]) for r in rows_a])
    edges = np.quantile(Ns, [0, 1 / 3, 2 / 3, 1])
    print(
        "  pooled form, outside gplite's per-column box, by part of the run:"
    )
    for group, unit in (("xm", "column widths"), ("lw", "log units")):
        for part, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            last = part == len(edges) - 2
            sel = [
                r
                for r in rows_a
                if lo <= r["N"] and (r["N"] <= hi if last else r["N"] < hi)
            ]
            vals = np.concatenate(
                [np.atleast_2d(r["hyp"])[:, r[f"{group}_index"]] for r in sel]
            )
            shape = lambda r: (
                np.atleast_2d(r["hyp"]).shape[0],
                len(r[f"{group}_index"]),
            )
            clb = np.concatenate(
                [np.broadcast_to(r[f"col_{group}_lb"], shape(r)) for r in sel]
            )
            cub = np.concatenate(
                [np.broadcast_to(r[f"col_{group}_ub"], shape(r)) for r in sel]
            )
            if group == "xm":
                scale = (cub - clb) / 2  # the column's width w_d
            else:
                scale = np.ones_like(cub)
            beyond = np.where(
                vals < clb,
                (clb - vals) / scale,
                np.where(vals > cub, (vals - cub) / scale, 0.0),
            )
            out = beyond > 0
            far = [
                np.median(beyond[out[:, d], d]) if out[:, d].any() else 0.0
                for d in range(vals.shape[1])
            ]
            print(
                f"    {group} N {int(lo)}-{int(hi)} ({vals.shape[0]} vectors): "
                "outside per dimension "
                + " ".join(f"{x:.2f}" for x in out.mean(0))
                + f"; median distance beyond, {unit}: "
                + " ".join(f"{x:.2f}" for x in far)
            )
