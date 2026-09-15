"""Weight-aware variants of the S-VBMC component-median cap, on recorded cells.

The integrated class reports, on a noisy stack, ``min(G, median_k I_k) + H``:
the stacked expected log joint ``G`` (the optimized weights times each
component's expected log joint ``I_k``, Jacobian-corrected) capped at the
median of ``I_k`` over every component of the stack, whatever its weight.
On a heavy-tailed target the median component sits far below the stack's
own expected log joint and the cap over-corrects (the stacking comparison
of ``dev/results/2026-09-15-svbmc-pool-comparison.md``, Student D8). This
script re-scores every cell of a comparison under caps that respect the
weights, without refitting anything:

- ``kappa`` caps: order the components by weight, walk down until the
  cumulative mass reaches ``kappa`` and include the component that crosses
  it (so the set is never empty), and cap ``G`` at the plain median of the
  ``I_k`` of that set. ``kappa = 1`` includes every component, which is the
  cap the class applies.
- the weighted median: the ``I_k`` at which the cumulative weight, with
  the components ordered by ``I_k``, reaches one half;
- run-level caps: ``E_max`` caps ``G`` at the largest of the runs' own
  expected log joints ``E_m`` (each run's posterior weights times its
  ``I_k``, the class's ``E_corrected``), so that a stack cannot report
  more than its best input run, and ``E_top`` at the ``E_m`` of the run
  carrying the largest stacking mass. A cap at the largest component
  ``I_k`` can never bind, since ``G`` is a convex combination of the
  ``I_k``; the script checks that on every cell and reports it.

``G`` and ``H`` are the cell's own (``raw − entropy`` and ``entropy`` of the
integrated arm), so only the cap level moves; the bias of every variant
is scored against the cell's recorded ``elbo_mc``, as the comparison
scores its headlines. The per-component ``I_k`` are not in the results
file, so each cell's stack is rebuilt from the pool artifacts with the
cell's entry seeds and cell seed and constructed, not optimized (the
construction is deterministic and yields the same retained runs and
component order as the recorded weights); the rebuild is checked against
the cell's raw value and its recorded cap. Needs Torch on ``PYTHONPATH``
(the overlay recorded in ``baseline_environment.json``) and the pool's
gpyreg through ``--gpyreg-source``::

    PYTHONPATH="<TORCH_PATH>" python -u dev/scripts/svbmc_cap_kappa.py \\
        --pool DIR --cells RESULTS.json --out DIR \\
        --gpyreg-source PATH [--kappa 0.5,0.6,0.7,0.8,0.9,0.95,0.99,1] \\
        [--conditions L1,L2]

Outputs under ``--out``: ``cells.jsonl`` (one line per cell: the levels,
biases and whether each cap binds), ``summary.json`` and ``summary.md``
(per condition and ``M``, the median bias over cells of the raw value,
the class's cap, every ``kappa`` cap and the weighted median, with the
fraction of cells each cap binds on).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from svbmc_pool_run import (  # noqa: E402
    DEFAULT_GPYREG,
    THREAD_KEYS,
    activate_gpyreg,
    write_json,
)

# isort: split
import numpy as np  # noqa: E402

DEFAULT_KAPPA = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)


def kappa_level(I, w, kappa):
    """The cap level: the median ``I_k`` over the top-weight components
    whose cumulative mass reaches ``kappa``, the crossing one included."""
    order = np.argsort(-w, kind="stable")
    cumulative = np.cumsum(w[order]) / np.sum(w)
    count = int(np.searchsorted(cumulative, kappa - 1e-12, side="left")) + 1
    count = min(count, w.size)
    chosen = order[:count]
    return float(np.median(I[chosen])), int(count)


def weighted_median_level(I, w):
    """The ``I_k`` at which the cumulative weight, in ``I_k`` order, reaches
    one half."""
    order = np.argsort(I, kind="stable")
    cumulative = np.cumsum(w[order]) / np.sum(w)
    index = int(np.searchsorted(cumulative, 0.5, side="left"))
    return float(I[order[min(index, I.size - 1)]])


def score_cell(cell, pool, kappas):
    from svbmc_pool_io import load_run

    from pyvbmc.svbmc import SVBMC

    arm = cell["arms"]["integrated"]
    vps = [
        load_run(pool / name, rng=seed)["vp"]
        for name, seed in zip(cell["entries"], cell["entry_seeds"])
    ]
    stacked = SVBMC(vps, seed=cell["cell_seed"], show_tips=False)
    I = np.ravel(stacked.I_corrected).astype(float)
    w = np.asarray(arm["w"], dtype=float)
    if w.size != I.size or [int(k) for k in stacked.K] != list(arm["K"]):
        raise RuntimeError(
            f"{cell['condition']} M={cell['M']} r={cell['repetition']}: the "
            f"rebuilt stack has {I.size} components in runs {list(stacked.K)}, "
            f"the recorded weights {w.size} in runs {list(arm['K'])}"
        )
    H = float(arm["entropy"])
    G = float(arm["elbos"]["raw"]) - H
    if abs(float(np.dot(w, I)) - G) > 1e-6:
        raise RuntimeError(
            f"{cell['condition']} M={cell['M']} r={cell['repetition']}: the "
            f"rebuilt expected log joint {np.dot(w, I):.6f} differs from the "
            f"recorded {G:.6f}"
        )
    elbo_mc = float(arm["elbo_mc"])
    record = {
        "condition": cell["condition"],
        "M": int(cell["M"]),
        "repetition": int(cell["repetition"]),
        "K_total": int(I.size),
        "G": G,
        "H": H,
        "elbo_mc": elbo_mc,
        "bias_raw": G + H - elbo_mc,
        "bias_class_cap": float(arm["elbos"]["capped_I_median"]) - elbo_mc,
        "caps": {},
    }
    for kappa in kappas:
        level, count = kappa_level(I, w, kappa)
        record["caps"][f"kappa_{kappa:g}"] = {
            "level": level,
            "components": count,
            "bias": min(G, level) + H - elbo_mc,
            "binds": bool(level < G),
        }
    level = weighted_median_level(I, w)
    record["caps"]["weighted_median"] = {
        "level": level,
        "components": None,
        "bias": min(G, level) + H - elbo_mc,
        "binds": bool(level < G),
    }
    E = np.ravel(stacked.E_corrected).astype(float)
    offsets = np.concatenate([[0], np.cumsum(arm["K"])])
    run_mass = np.array(
        [np.sum(w[offsets[m] : offsets[m + 1]]) for m in range(len(arm["K"]))]
    )
    for name, level in (
        ("E_max", float(np.max(E))),
        ("E_top", float(E[int(np.argmax(run_mass))])),
    ):
        record["caps"][name] = {
            "level": level,
            "components": None,
            "bias": min(G, level) + H - elbo_mc,
            "binds": bool(level < G),
        }
    record["bias_class_cap_E"] = (
        float(arm["elbos"]["capped_E_median"]) - elbo_mc
    )
    # A cap at the largest component I_k is the raw value: G is a convex
    # combination of the I_k.
    record["component_max_binds"] = bool(float(np.max(I)) < G - 1e-9)
    # kappa = 1 is the class's own cap; the rebuild must reproduce it.
    if 1.0 in kappas:
        difference = abs(
            record["caps"]["kappa_1"]["bias"] - record["bias_class_cap"]
        )
        if difference > 1e-6:
            raise RuntimeError(
                f"{cell['condition']} M={cell['M']} r={cell['repetition']}: "
                f"the kappa = 1 cap differs from the class's by {difference:.3e}"
            )
    return record


def cap_names(kappas):
    return [f"kappa_{k:g}" for k in kappas] + [
        "weighted_median",
        "E_max",
        "E_top",
    ]


def summarize(records, kappas):
    names = cap_names(kappas)
    conditions = []
    for condition in dict.fromkeys(r["condition"] for r in records):
        entry = {"condition": condition, "M": []}
        for M in sorted(
            {r["M"] for r in records if r["condition"] == condition}
        ):
            rows = [
                r
                for r in records
                if r["condition"] == condition and r["M"] == M
            ]
            item = {
                "M": M,
                "cells": len(rows),
                "bias_raw": float(np.median([r["bias_raw"] for r in rows])),
                "bias_class_cap": float(
                    np.median([r["bias_class_cap"] for r in rows])
                ),
                "bias_class_cap_E": float(
                    np.median([r["bias_class_cap_E"] for r in rows])
                ),
                "component_max_binds": int(
                    sum(r["component_max_binds"] for r in rows)
                ),
                "caps": {
                    name: {
                        "bias": float(
                            np.median([r["caps"][name]["bias"] for r in rows])
                        ),
                        "abs_bias": float(
                            np.median(
                                [abs(r["caps"][name]["bias"]) for r in rows]
                            )
                        ),
                        "binds": float(
                            np.mean([r["caps"][name]["binds"] for r in rows])
                        ),
                    }
                    for name in names
                },
            }
            entry["M"].append(item)
        conditions.append(entry)
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kappa": list(kappas),
        "cells": len(records),
        "conditions": conditions,
    }


def markdown(summary):
    names = cap_names(summary["kappa"])
    never = sum(
        item["component_max_binds"]
        for entry in summary["conditions"]
        for item in entry["M"]
    )
    lines = [
        "# Weight-aware caps on the stacked expected log joint",
        "",
        f"Generated {summary['generated']} from {summary['cells']} cells of "
        "the integrated arm. Each entry is the median over the cells of a "
        "condition and `M` of the bias against the cell's `elbo_mc`; `raw` "
        "is the uncapped value, `class` the cap the integrated class "
        "applies (every component; equal to `kappa_1`), `kappa_x` the cap "
        "at the median of the top-weight components carrying mass `x`, "
        "`wmed` the weighted median of the components' expected log "
        "joints, `E_max` the cap at the largest run-level expected log "
        "joint, `E_top` at that of the run carrying the largest stacking "
        "mass, `classE` the class's run-median cap (`capped_E_median`). "
        "In brackets, the fraction of cells on which the cap binds. A cap "
        "at the largest component expected log joint would bind on "
        f"{never} of {summary['cells']} cells: it is the raw value.",
        "",
    ]
    header = (
        "| condition | M | raw | class | classE | "
        + " | ".join(
            n.replace("kappa_", "κ ").replace("weighted_median", "wmed")
            for n in names
        )
        + " |"
    )
    lines += [header, "|" + "---|" * (5 + len(names))]
    for entry in summary["conditions"]:
        for item in entry["M"]:
            cells = [
                f"{item['caps'][n]['bias']:+.2f} [{item['caps'][n]['binds']:.2f}]"
                for n in names
            ]
            lines.append(
                f"| {entry['condition'].replace('_svbmc', '')} | {item['M']} | "
                f"{item['bias_raw']:+.2f} | {item['bias_class_cap']:+.2f} | "
                f"{item['bias_class_cap_E']:+.2f} | "
                + " | ".join(cells)
                + " |"
            )
    lines += [
        "",
        "Median absolute bias over the cells of each condition and `M`, the "
        "quantity a headline should make small:",
        "",
        header,
        "|" + "---|" * (5 + len(names)),
    ]
    for entry in summary["conditions"]:
        for item in entry["M"]:
            lines.append(
                f"| {entry['condition'].replace('_svbmc', '')} | {item['M']} | "
                f"{abs(item['bias_raw']):.2f} | {abs(item['bias_class_cap']):.2f} | "
                f"{abs(item['bias_class_cap_E']):.2f} | "
                + " | ".join(
                    f"{item['caps'][n]['abs_bias']:.2f}" for n in names
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpyreg-source", type=Path, default=DEFAULT_GPYREG)
    parser.add_argument(
        "--kappa", default=",".join(str(k) for k in DEFAULT_KAPPA)
    )
    parser.add_argument("--conditions", default=None)
    args = parser.parse_args(argv)
    kappas = tuple(float(k) for k in args.kappa.split(",") if k.strip())
    # The summary carries a Greek kappa; a console with a narrow encoding
    # must not end the run after the files are written.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    for key in THREAD_KEYS:
        os.environ.setdefault(key, "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    activate_gpyreg(args.gpyreg_source)

    results = json.loads(args.cells.read_text(encoding="utf-8"))
    only = [c.strip() for c in (args.conditions or "").split(",") if c.strip()]
    cells = [
        c
        for c in results["cells"]
        if "integrated" in c["arms"] and (not only or c["condition"] in only)
    ]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pool = args.pool.resolve()
    records = []
    started = time.perf_counter()
    with (out / "cells.jsonl").open("w", encoding="utf-8") as progress:
        for number, cell in enumerate(cells, start=1):
            record = score_cell(cell, pool, kappas)
            records.append(record)
            progress.write(json.dumps(record) + "\n")
            progress.flush()
            if number % 20 == 0 or number == len(cells):
                print(
                    f"{number}/{len(cells)} cells, "
                    f"{(time.perf_counter() - started) / 60:.1f} min",
                    flush=True,
                )
    summary = summarize(records, kappas)
    summary["cells_file"] = str(args.cells.resolve())
    summary["pool"] = str(pool)
    write_json(out / "summary.json", summary)
    text = markdown(summary)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
