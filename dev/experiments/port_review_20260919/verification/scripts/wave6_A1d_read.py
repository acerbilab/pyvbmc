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


# The mean's location, per target and dimension: in the pooled form, the
# share of the vectors outside gplite's per-column box and how far beyond
# it (median, 90th percentile and maximum, in column widths); the width of
# the quadratic, exp(log omega_d) / w_d, at the median for the vectors
# inside the box and for those outside, and the share of the latter wider
# than gplite's cap of e^3 widths; the change of the quadratic term across
# the data, 0.5 * ((x - xm_d) / omega_d)^2 between the column's minimum and
# maximum, at the median for the vectors outside; and, in each form, where
# the location lies in the box (0 its lower bound, 1 its upper, the data
# spanning 0.25 to 0.75): outside it, in its outer 2% and 10%, and within
# the span of the data. And the share of the pooled form's log scale above
# and below gplite's per-column box.
print("\n== the mean's location, per target and dimension")
for label in pooled:
    rows_a, rows_b = pooled[label], per_column.get(label, [])
    if "xm_index" not in rows_a[0]:
        continue
    for d in range(len(rows_a[0]["xm_index"])):
        beyond, ratio_in, ratio_out, change, pos_a = [], [], [], [], []
        for r in rows_a:
            h = np.atleast_2d(r["hyp"])
            xm, lw = h[:, r["xm_index"][d]], h[:, r["lw_index"][d]]
            lb, ub = r["col_xm_lb"][d], r["col_xm_ub"][d]
            w = (ub - lb) / 2
            low, high = lb + 0.5 * w, ub - 0.5 * w
            ratio = np.exp(lw) / w
            out = (xm < lb) | (xm > ub)
            beyond.extend(np.where(xm < lb, lb - xm, xm - ub)[out] / w)
            ratio_in.extend(ratio[~out])
            ratio_out.extend(ratio[out])
            term = lambda x: 0.5 * ((x - xm[out]) / np.exp(lw[out])) ** 2
            change.extend(np.abs(term(low) - term(high)))
            pos_a.extend((xm - lb) / (ub - lb))
        lw_vals = np.concatenate(
            [np.atleast_2d(r["hyp"])[:, r["lw_index"][d]] for r in rows_a]
        )
        lw_lb = np.concatenate(
            [
                np.full(np.atleast_2d(r["hyp"]).shape[0], r["col_lw_lb"][d])
                for r in rows_a
            ]
        )
        lw_ub = np.concatenate(
            [
                np.full(np.atleast_2d(r["hyp"]).shape[0], r["col_lw_ub"][d])
                for r in rows_a
            ]
        )
        pos_b = []
        for r in rows_b:
            xm = np.atleast_2d(r["hyp"])[:, r["xm_index"][d]]
            lb, ub = r["col_xm_lb"][d], r["col_xm_ub"][d]
            pos_b.extend((xm - lb) / (ub - lb))

        def where(p):
            p = np.asarray(p)
            inside = (p >= 0) & (p <= 1)
            return (
                (~inside).mean(),
                (inside & ((p < 0.02) | (p > 0.98))).mean(),
                (inside & ((p < 0.1) | (p > 0.9))).mean(),
                ((p >= 0.25) & (p <= 0.75)).mean(),
            )

        b, ro = np.asarray(beyond), np.asarray(ratio_out)
        n = len(pos_a)
        print(
            f"  {label} location[{d}]: outside {b.size}/{n} "
            f"({b.size / n:.3f})",
            end="",
        )
        if b.size:
            print(
                f"; beyond, column widths: median {np.median(b):.2f}, "
                f"90th {np.quantile(b, 0.9):.2f}, max {b.max():.2f}; width "
                f"inside {np.median(ratio_in):.2f}, outside "
                f"{np.median(ro):.2f}, outside above e^3 "
                f"{np.mean(ro > np.exp(3)):.3f}; change across the data, "
                f"outside {np.median(change):.2f} nats",
                end="",
            )
        print()
        print(
            f"    log scale of the pooled form: above its per-column box "
            f"{np.mean(lw_vals > lw_ub):.3f}, below "
            f"{np.mean(lw_vals < lw_lb):.3f}"
        )
        print(
            "    outside | outer 2% | outer 10% | within the data span "
            "(pooled -> per column): "
            + " | ".join(
                f"{x:.3f} -> {y:.3f}"
                for x, y in zip(where(pos_a), where(pos_b))
            )
        )
