"""F3 figures: S0 versus S2 trajectories on one E5 configuration.

Reads the same golden traces, sidecars and selection records as
``noisy_acq_e5_trajectory_diagnostic.py`` and draws three PNG figures with a
non-interactive backend: the ELBO error against target evaluations for every
seed, the reliability index against iteration with the stopping threshold,
and the live evaluated points of selected seeds over the moment-matched
Gaussian ellipses of the reference posterior.  Read-only; no target calls,
no GP fits.

The trace's ``X_orig`` holds the logger's live rows after warmup trimming;
the initial design is taken from ``X_init`` and the first post-warmup row
from the selection records, and each panel states how many live points lie
outside its frame.

Example::

    python dev/scripts/noisy_acq_e5_trajectory_figures.py \
        --pilot <pilot campaign> --continuation <continuation campaign> \
        --label rosenbrock_D2_noise3 --scatter-seeds 2008,2004 --out <dir>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from scipy import stats  # noqa: E402

PILOT_SEEDS = (2000, 2001)
CONTINUATION_SEEDS = tuple(range(2002, 2010))
ARMS = ("S0", "S2")
# Validated default categorical slots 1 and 2 (light surface), text and
# surface tokens from the charting reference palette.
COLOR = {"S0": "#2a78d6", "S2": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e1"
FRAME_X = (-4.0, 4.0)
FRAME_Y = (-4.0, 8.0)
R_INDEX_TOP = 30.0


def load(campaign, label, seed, arm):
    tag = f"{label}_seed{seed}"
    with np.load(
        campaign / "arms" / arm / f"{tag}.npz", allow_pickle=False
    ) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    side = json.loads(
        (campaign / "arms" / arm / f"{tag}.json").read_text(encoding="utf-8")
    )
    records = json.loads(
        (
            campaign / "records" / f"{label}__seed{seed}__{arm}.selection.json"
        ).read_text(encoding="utf-8")
    )
    return arrays, side, records


def style(ax, title=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")


def warmup_end(a):
    warm = a["warmup"].astype(bool)
    return int(np.argmax(~warm)) if (~warm).any() else None


def live_masks(a, records):
    X, X_init = a["X_orig"], a["X_init"]
    init = np.zeros(X.shape[0], dtype=bool)
    for row in X_init:
        init |= np.all(X == row, axis=1)
    w = warmup_end(a)
    recs = records["records"]
    if w is None:
        first_post = X.shape[0]
    else:
        k = int(a["func_count"][w]) - int(X_init.shape[0])
        first_post = (
            int(recs[k]["logger_live_rows"])
            if 0 <= k < len(recs)
            else X.shape[0]
        )
    post = np.arange(X.shape[0]) >= first_post
    return init, ~post & ~init, post


def figure_elbo(pairs, label, out):
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (seed, data) in zip(axes.ravel(), pairs):
        for arm in ARMS:
            a, side, _ = data[arm]
            err = np.abs(a["elbo"] - side["ln_Z"])
            ax.plot(
                a["func_count"], err, color=COLOR[arm], linewidth=2, label=arm
            )
            w = warmup_end(a)
            if w is not None:
                ax.plot(
                    a["func_count"][w],
                    err[w],
                    marker="o",
                    markersize=5,
                    color=COLOR[arm],
                    markeredgecolor=SURFACE,
                    markeredgewidth=1,
                )
            if not side["final"]["success_flag"]:
                ax.plot(
                    a["func_count"][-1],
                    err[-1],
                    marker="x",
                    markersize=8,
                    color=COLOR[arm],
                    markeredgewidth=2,
                )
        ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=":")
        ax.set_yscale("log")
        ax.set_xlim(0, 205)
        style(ax, f"seed {seed}")
    for ax in axes[1]:
        ax.set_xlabel("target evaluations", fontsize=8, color=INK_SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel(
            "|ELBO − ln Z| per iteration", fontsize=8, color=INK_SECONDARY
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        ["S0 (production search)", "S2 (1024 sieve, re-score, refine)"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"{label}: |ELBO − ln Z| of each iteration's variational posterior "
        "along the paired trajectories\n"
        "dot: warmup end; x: stopped at the evaluation budget; dotted: "
        "|ELBO − ln Z| = 1, one of the three usability conditions; the "
        "reported final error is the boosted best iteration's",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(out / f"{label}_elbo_error.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_r_index(pairs, label, out):
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    clipped = 0
    for ax, (seed, data) in zip(axes.ravel(), pairs):
        for arm in ARMS:
            a, side, _ = data[arm]
            r = a["r_index"].copy()
            finite = np.isfinite(r)
            clipped += int(np.sum(r[finite] > R_INDEX_TOP))
            r[~finite] = np.nan
            ax.plot(a["iter"], r, color=COLOR[arm], linewidth=2, label=arm)
            w = warmup_end(a)
            if w is not None:
                ax.axvline(
                    a["iter"][w],
                    color=COLOR[arm],
                    linewidth=1,
                    linestyle="--",
                    alpha=0.7,
                )
        ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=":")
        ax.set_yscale("log")
        ax.set_ylim(0.1, R_INDEX_TOP)
        style(ax, f"seed {seed}")
    for ax in axes[1]:
        ax.set_xlabel("iteration", fontsize=8, color=INK_SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("reliability index", fontsize=8, color=INK_SECONDARY)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        ["S0 (production search)", "S2 (1024 sieve, re-score, refine)"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"{label}: reliability index per iteration\n"
        "dotted: stability threshold 1; dashed: warmup end; the axis is "
        f"clipped at {R_INDEX_TOP:g} ({clipped} finite values above it, plus the "
        "undefined first two iterations of each run)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(out / f"{label}_r_index.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_points(pairs, label, seeds, out):
    chosen = [(s, d) for s, d in pairs if s in seeds]
    fig, axes = plt.subplots(
        len(chosen), 2, figsize=(9, 4.6 * len(chosen)), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for row, (seed, data) in zip(axes, chosen):
        for ax, arm in zip(row, ARMS):
            a, side, records = data[arm]
            mean = np.asarray(side["true_mean"], dtype=float).ravel()
            cov = np.asarray(side["true_cov"], dtype=float)
            vals, vecs = np.linalg.eigh(cov)
            angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
            for q, ls in ((0.5, "-"), (0.99, "--")):
                k = np.sqrt(stats.chi2.ppf(q, 2))
                ax.add_patch(
                    Ellipse(
                        mean,
                        2 * k * np.sqrt(vals[0]),
                        2 * k * np.sqrt(vals[1]),
                        angle=angle,
                        fill=False,
                        edgecolor=MUTED,
                        linewidth=1,
                        linestyle=ls,
                    )
                )
            X = a["X_orig"]
            init, warm, post = live_masks(a, records)
            outside = np.sum(
                (X[:, 0] < FRAME_X[0])
                | (X[:, 0] > FRAME_X[1])
                | (X[:, 1] < FRAME_Y[0])
                | (X[:, 1] > FRAME_Y[1])
            )
            ax.scatter(
                a["X_init"][:, 0],
                a["X_init"][:, 1],
                s=24,
                facecolor="none",
                edgecolor=INK_SECONDARY,
                linewidth=0.8,
                label="initial design",
            )
            ax.scatter(
                X[warm, 0],
                X[warm, 1],
                s=18,
                color=COLOR[arm],
                alpha=0.35,
                edgecolor=SURFACE,
                linewidth=0.5,
                label="warmup (live rows)",
            )
            ax.scatter(
                X[post, 0],
                X[post, 1],
                s=26,
                color=COLOR[arm],
                alpha=0.95,
                edgecolor=SURFACE,
                linewidth=0.5,
                label="after warmup",
            )
            fin = side["final"]
            status = (
                "converged" if fin["success_flag"] else "stopped at budget"
            )
            style(
                ax,
                f"seed {seed}, {arm}: {status}\n"
                f"{fin['func_count']} evals, {X.shape[0]} live "
                f"({int(post.sum())} post-warmup), {int(outside)} off-frame\n"
                f"gsKL {fin['gskl']:.3f}, MMTV {fin['mmtv']:.3f}",
            )
            ax.set_xlim(*FRAME_X)
            ax.set_ylim(*FRAME_Y)
            ax.set_aspect("equal")
            ax.set_xlabel("x1", fontsize=8, color=INK_SECONDARY)
            ax.set_ylabel("x2", fontsize=8, color=INK_SECONDARY)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"{label}: live evaluated points\n"
        "ellipses: moment-matched Gaussian of the reference posterior at the\n"
        "nominal 50% (solid) and 99% (dashed) levels, holding about 0.60 and "
        "0.97 of the true mass;\nwarmup rows trimmed at warmup end are absent",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(out / f"{label}_points.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--label", default="rosenbrock_D2_noise3")
    parser.add_argument("--scatter-seeds", default="2008,2004")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for seed, campaign in [(s, args.pilot) for s in PILOT_SEEDS] + [
        (s, args.continuation) for s in CONTINUATION_SEEDS
    ]:
        pairs.append(
            (
                seed,
                {arm: load(campaign, args.label, seed, arm) for arm in ARMS},
            )
        )
    figure_elbo(pairs, args.label, args.out)
    figure_r_index(pairs, args.label, args.out)
    seeds = tuple(int(s) for s in args.scatter_seeds.split(","))
    figure_points(pairs, args.label, seeds, args.out)
    for p in sorted(args.out.glob(f"{args.label}_*.png")):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
