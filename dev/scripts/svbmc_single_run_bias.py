"""The bias of a single VBMC run's ELBO, and what stacking adds to it.

The stacking comparison scores every stacked estimate by its bias against
the stack's own Monte Carlo ELBO. The runs that go into a stack carry a
bias of their own, from VBMC's variational optimization selecting
components on noisy GP estimates, and S-VBMC's job is not to remove it
but not to add to it. This script measures the baseline: every filtered
run of a pool is scored as a stack of one, with the same reference the
comparison uses (the mean of the target's noiseless log density over
draws from the posterior, plus the entropy of the mixture by the
comparison's estimator), and its reported ELBO is compared with that
reference. Two values are scored per run: ``elbo_vbmc``, the ELBO the run
itself reports (``vp.stats["elbo"]``, with VBMC's own entropy estimate),
and ``elbo_raw``, the integrated class's value for the run alone at its
own weights, without optimization (``G + H`` with the class's entropy
at ``N_SAMPLES_FINAL`` draws per component); the component-median cap
at those weights is scored too.

With ``--cells`` (the comparison's ``results.json``, repeatable) the
script also joins the runs' biases to every cell that holds an
integrated arm: the mean bias of the cell's input runs, the bias of the
input with the highest reported ELBO, and the stack's mass-weighted mean
over its inputs; and for every stacked estimate the **added bias**, the
stack's bias minus the mean bias of its inputs. With ``--shrink`` (a
``cells.jsonl`` of ``svbmc_shrink_elbo.py``, repeatable) the shrinkage
variants of the same cells are joined too. Needs Torch on
``PYTHONPATH`` and the pool's gpyreg through ``--gpyreg-source``::

    PYTHONPATH="<TORCH_PATH>" python -u dev/scripts/svbmc_single_run_bias.py \\
        --pool DIR --out DIR --gpyreg-source PATH \\
        [--conditions L1,L2] [--limit N] [--reuse] \\
        [--cells RESULTS.json ...] [--shrink CELLS.jsonl ...]

Outputs under ``--out``: ``runs.jsonl`` (one record per run: the
condition, name and seed, ``K``, the three ELBOs, ``e_log_joint_mc``,
``entropy_ref``, ``elbo_mc`` with their standard errors, the biases, the
KL gap where ``ln Z`` is known, the run's ``elbo_sd`` and the single-run
metrics), ``summary.json`` / ``summary.md`` (per condition, the median
with a bootstrap interval, the mean and the quartiles of every bias and
of the KL gap), ``sources.json``, and with ``--cells`` also
``cells.jsonl`` (per cell: the input biases and every estimate's bias and
added bias) and ``added.json`` / ``added.md`` (per condition and ``M``,
medians over cells of the inputs' mean bias, of each estimate's bias
and added bias, a bootstrap interval on the added bias of ``raw``,
``run_level``, ``two_level_full``, ``two_level_anchored`` and
``two_level_anchored_mean``, and the fraction of cells whose raw added
bias is positive). ``--limit N`` scores the first ``N`` runs of every
condition, for a smoke test. ``--reuse`` reads the ``runs.jsonl``
already under ``--out`` instead of scoring the runs again and redoes
the summaries and the join; every pass rewrites ``summary.*`` and
``sources.json`` (which hashes the script, the runs file and every
``--cells`` and ``--shrink`` file it read).
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
    THREAD_KEYS,
    activate_gpyreg,
    identity,
    pinned_gpyreg_source,
    write_json,
)

# isort: split
import numpy as np  # noqa: E402
from svbmc_pool_stack import (  # noqa: E402
    ENTROPY_REF_STREAM,
    N_DRAWS,
    N_SAMPLES_FINAL,
    Problems,
    bootstrap_median,
    entropy_reference,
    load_entry,
    pool_conditions,
    stacked_outcome,
)
from svbmc_shrink_elbo import VARIANTS  # noqa: E402

BIASES = ("bias_vbmc", "bias_raw", "bias_cap")
#: Stacked estimates joined per cell, from the comparison's record and,
#: when given, the shrinkage cells (every variant the shrinkage script
#: writes).
STACK_ESTIMATES = ("raw", "capped_I_median")
SHRINK_ESTIMATES = VARIANTS
#: The estimates whose added bias gets a bootstrap interval.
INTERVAL_ESTIMATES = (
    "raw",
    "two_level_full",
    "run_level",
    "two_level_anchored",
    "two_level_anchored_mean",
)


def score_run(entry, problem, reference):
    """One filtered run as a stack of one, scored as the comparison scores
    a cell's integrated arm."""
    import torch

    from pyvbmc.svbmc import SVBMC

    started = time.perf_counter()
    seed = int(entry["seed"])
    vp = load_entry(entry, rng=seed)
    stacked = SVBMC([vp], seed=seed, show_tips=False)
    if stacked.M != 1:
        raise RuntimeError(f"{entry['name']}: the run did not pass the class")
    w = np.ravel(stacked.w).astype(float)
    I = np.ravel(stacked.I_corrected).astype(float)
    G = float(np.dot(w, I))
    tensor = torch.as_tensor(w.reshape(1, -1), dtype=torch.float64)
    with torch.no_grad():
        H, _ = stacked.stacked_entropy(tensor, N_SAMPLES_FINAL)
    H = float(H.item())
    raw = G + H
    capped = float(min(G, np.median(I))) + H
    elbo_vbmc = float(vp.stats["elbo"])
    headline = capped if stacked.noisy else raw
    samples = stacked.sample(N_DRAWS)
    outcome = stacked_outcome(
        problem,
        reference,
        samples,
        {
            "headline": headline,
            "raw": raw,
            "capped_I_median": capped,
            "vbmc": elbo_vbmc,
        },
        seed,
        "integrated",
    )
    rng = np.random.default_rng([seed, ENTROPY_REF_STREAM])
    outcome.update(entropy_reference(stacked, w, rng))
    elbo_mc = float(outcome["e_log_joint_mc"] + outcome["entropy_ref"])
    ln_Z = problem.ln_Z
    record = {
        "condition": entry["label"],
        "name": entry["name"],
        "seed": seed,
        "K": int(stacked.K[0]),
        "noisy": bool(stacked.noisy),
        "elbo_vbmc": elbo_vbmc,
        "elbo_raw": raw,
        "elbo_cap": capped,
        "G": G,
        "H": H,
        "e_log_joint_mc": float(outcome["e_log_joint_mc"]),
        "e_log_joint_mc_sd": float(outcome["e_log_joint_mc_sd"]),
        "entropy_ref": float(outcome["entropy_ref"]),
        "entropy_ref_sd": float(outcome["entropy_ref_sd"]),
        "elbo_mc": elbo_mc,
        "elbo_mc_sd": float(
            np.hypot(outcome["e_log_joint_mc_sd"], outcome["entropy_ref_sd"])
        ),
        "bias_vbmc": elbo_vbmc - elbo_mc,
        "bias_raw": raw - elbo_mc,
        "bias_cap": capped - elbo_mc,
        "kl_gap": float("nan") if ln_Z is None else float(ln_Z - elbo_mc),
        "elbo_sd": float(vp.stats.get("elbo_sd", float("nan"))),
        "metrics": outcome["metrics"],
        "seconds": time.perf_counter() - started,
    }
    return record


def quartiles(values):
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return {"n": 0, "mean": None, "q25": None, "q75": None}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "q25": float(np.percentile(finite, 25)),
        "q75": float(np.percentile(finite, 75)),
    }


def summarize_runs(records, rng):
    conditions = list(dict.fromkeys(r["condition"] for r in records))
    out = {"conditions": []}
    for condition in conditions:
        rows = [r for r in records if r["condition"] == condition]
        entry = {"condition": condition, "n": len(rows)}
        for key in BIASES + ("kl_gap",):
            values = [r[key] for r in rows]
            entry[key] = dict(
                bootstrap_median(values, rng), **quartiles(values)
            )
        entry["elbo_sd"] = quartiles([r["elbo_sd"] for r in rows])
        entry["seconds"] = float(sum(r["seconds"] for r in rows))
        out["conditions"].append(entry)
    return out


def interval(entry, digits=2):
    if entry["median"] is None:
        return "n/a"
    return (
        f"{entry['median']:+.{digits}f} "
        f"[{entry['lo']:+.{digits}f}, {entry['hi']:+.{digits}f}]"
    )


def runs_markdown(summary):
    lines = [
        "Single VBMC runs scored against their own Monte Carlo ELBO "
        "(`elbo_mc`, the target's noiseless log density averaged over "
        "draws from the run's posterior plus the mixture's entropy by the "
        "comparison's estimator). `bias_vbmc` is the ELBO the run reports "
        "minus `elbo_mc`; `bias_raw` the integrated class's raw value for "
        "the run alone at its own weights; `bias_cap` the component-median "
        "cap at those weights. Medians over the filtered runs with a "
        "bootstrap 95 % interval, then the mean and the quartiles.",
        "",
        "| condition | n | bias_vbmc, median [CI] | mean [q25, q75] | "
        "bias_raw, median | bias_cap, median | KL gap, median | "
        "elbo_sd, mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in summary["conditions"]:
        v = c["bias_vbmc"]
        lines.append(
            f"| {c['condition']} | {c['n']} | {interval(v)} | "
            f"{v['mean']:+.2f} [{v['q25']:+.2f}, {v['q75']:+.2f}] | "
            f"{interval(c['bias_raw'])} | {interval(c['bias_cap'])} | "
            f"{interval(c['kl_gap'])} | {c['elbo_sd']['mean']:.2f} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# The join with the comparison's cells
# --------------------------------------------------------------------------


def load_cells(paths):
    cells = []
    for path in paths:
        results = json.loads(Path(path).read_text(encoding="utf-8"))
        for cell in results["cells"]:
            if "integrated" in cell["arms"]:
                cells.append(cell)
    return cells


def load_shrink(paths):
    by_key = {}
    for path in paths:
        for line in Path(path).open(encoding="utf-8"):
            record = json.loads(line)
            key = (
                record["condition"],
                int(record["M"]),
                int(record["repetition"]),
            )
            by_key[key] = record
    return by_key


def join_cell(cell, runs_by_name, shrink):
    """The input biases of one cell and every estimate's added bias."""
    arm = cell["arms"]["integrated"]
    inputs = [runs_by_name[name] for name in cell["entries"]]
    input_bias = np.array([r["bias_vbmc"] for r in inputs])
    input_raw = np.array([r["bias_raw"] for r in inputs])
    reported = np.array([r["elbo_vbmc"] for r in inputs])
    K = [int(k) for k in arm["K"]]
    w = np.asarray(arm["w"], dtype=float)
    offsets = np.concatenate([[0], np.cumsum(K)])
    mass = np.array(
        [float(np.sum(w[a:b])) for a, b in zip(offsets[:-1], offsets[1:])]
    )
    if len(inputs) != len(K):
        raise RuntimeError(
            f"{cell['condition']} M={cell['M']} r={cell['repetition']}: "
            f"{len(inputs)} inputs but the stack retained {len(K)} runs"
        )
    mean_bias = float(np.mean(input_bias))
    record = {
        "condition": cell["condition"],
        "M": int(cell["M"]),
        "repetition": int(cell["repetition"]),
        "inputs": {
            "names": list(cell["entries"]),
            "bias_vbmc": input_bias.tolist(),
            "bias_raw": input_raw.tolist(),
            "mass": mass.tolist(),
            "mean_bias": mean_bias,
            "mean_bias_raw": float(np.mean(input_raw)),
            "best_bias": float(input_bias[int(np.argmax(reported))]),
            "mass_bias": float(np.dot(mass, input_bias) / np.sum(mass)),
        },
        "bias": {},
        "added": {},
    }
    for name in STACK_ESTIMATES:
        record["bias"][name] = float(arm["bias"][name])
    key = (cell["condition"], int(cell["M"]), int(cell["repetition"]))
    if key in shrink:
        for name in SHRINK_ESTIMATES:
            record["bias"][name] = float(shrink[key]["variants"][name]["bias"])
    for name, value in record["bias"].items():
        record["added"][name] = value - mean_bias
    return record


def summarize_added(records, rng):
    conditions = list(dict.fromkeys(r["condition"] for r in records))
    out = {"conditions": []}
    for condition in conditions:
        rows = [r for r in records if r["condition"] == condition]
        Ms = sorted({r["M"] for r in rows})
        entry = {"condition": condition, "by_M": []}
        for M in Ms:
            cells = [r for r in rows if r["M"] == M]
            names = list(cells[0]["bias"])
            for c in cells:
                if list(c["bias"]) != names:
                    raise RuntimeError(
                        f"{condition} M={M}: the cells carry different "
                        "estimate sets; a summary needs the same shrinkage "
                        "coverage on every cell"
                    )
            row = {
                "M": M,
                "n": len(cells),
                "inputs_mean_bias": float(
                    np.median([c["inputs"]["mean_bias"] for c in cells])
                ),
                "inputs_best_bias": float(
                    np.median([c["inputs"]["best_bias"] for c in cells])
                ),
                "inputs_mass_bias": float(
                    np.median([c["inputs"]["mass_bias"] for c in cells])
                ),
                "bias": {
                    name: float(np.median([c["bias"][name] for c in cells]))
                    for name in names
                },
                "added": {
                    name: float(np.median([c["added"][name] for c in cells]))
                    for name in names
                },
                "added_interval": {
                    name: bootstrap_median(
                        [c["added"][name] for c in cells], rng
                    )
                    for name in names
                    if name in INTERVAL_ESTIMATES
                },
                "raw_added_positive": float(
                    np.mean([c["added"]["raw"] > 0 for c in cells])
                ),
            }
            entry["by_M"].append(row)
        out["conditions"].append(entry)
    return out


def added_markdown(summary):
    lines = [
        "Per condition and `M`, medians over cells: the mean bias of the "
        "cell's input runs (each run's reported ELBO against its own "
        "`elbo_mc`), the bias of the input with the highest reported ELBO, "
        "the stack's raw bias, and the **added bias** of every estimate "
        "(the stack's bias minus the mean bias of its inputs; a positive "
        "value means the stacked estimate is more optimistic than the runs "
        "it was built from). The interval on the raw added bias is a "
        "percentile bootstrap over cells, and the last column the fraction "
        "of cells whose raw added bias is positive.",
        "",
    ]
    for c in summary["conditions"]:
        names = list(c["by_M"][0]["bias"])
        lines.append(f"## {c['condition']}")
        lines.append("")
        header = (
            "| M | n | inputs mean | inputs best | stack raw | raw added [CI] | "
            + " | ".join(f"{name} added" for name in names if name != "raw")
            + " | raw added > 0 |"
        )
        lines.append(header)
        # M, n, inputs mean, inputs best, stack raw, raw added [CI], one
        # column per other estimate, raw added > 0.
        lines.append("|" + "---|" * (len(names) + 6))
        for row in c["by_M"]:
            lines.append(
                f"| {row['M']} | {row['n']} | {row['inputs_mean_bias']:+.2f} | "
                f"{row['inputs_best_bias']:+.2f} | {row['bias']['raw']:+.2f} | "
                f"{interval(row['added_interval']['raw'])} | "
                + " | ".join(
                    f"{row['added'][name]:+.2f}"
                    for name in names
                    if name != "raw"
                )
                + f" | {row['raw_added_positive']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--gpyreg-source",
        type=Path,
        required=True,
        help="the gpyreg checkout the pool is read against: a clean git "
        "checkout at the commit the pool's manifest pins",
    )
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cells", type=Path, action="append", default=[])
    parser.add_argument("--shrink", type=Path, action="append", default=[])
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="read runs.jsonl under --out instead of scoring the runs again",
    )
    args = parser.parse_args(argv)
    for key in THREAD_KEYS:
        os.environ.setdefault(key, "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    pool = args.pool.resolve()
    manifest = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))
    gpyreg_source = activate_gpyreg(
        pinned_gpyreg_source(args.gpyreg_source, manifest)
    )
    import torch

    torch.set_num_threads(1)
    import logging

    logging.getLogger("SVBMC").setLevel(logging.WARNING)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    from svbmc_pool_io import sha256

    only = [c.strip() for c in (args.conditions or "").split(",") if c.strip()]
    conditions, identities, _ = pool_conditions(
        [pool], only, gpyreg_source=args.gpyreg_source
    )
    if args.reuse:
        records = [
            json.loads(line)
            for line in (out / "runs.jsonl").open(encoding="utf-8")
        ]
    else:
        problems = Problems()
        records = []
        total = sum(
            len(entries[: args.limit]) for entries in conditions.values()
        )
        started = time.perf_counter()
        number = 0
        with (out / "runs.jsonl").open("w", encoding="utf-8") as progress:
            for condition, entries in conditions.items():
                problem = problems.get(condition, "run")
                reference = problems.reference(condition, "run")
                for entry in entries[: args.limit]:
                    record = score_run(entry, problem, reference)
                    records.append(record)
                    progress.write(json.dumps(record) + "\n")
                    progress.flush()
                    number += 1
                    if number % 25 == 0 or number == total:
                        print(
                            f"{number}/{total} runs, "
                            f"{(time.perf_counter() - started) / 60:.1f} min "
                            f"({condition})",
                            flush=True,
                        )
    # The summaries and the provenance record are rewritten on every
    # pass, so that they describe the files under --out whether the
    # runs were scored now or read back.
    summary = summarize_runs(records, np.random.default_rng(0))
    summary["pool"] = str(pool)
    summary["pools"] = identities
    summary["settings"] = {
        "n_draws": N_DRAWS,
        "n_samples_final": N_SAMPLES_FINAL,
        "limit": args.limit,
        "reuse": bool(args.reuse),
    }
    write_json(out / "summary.json", summary)
    text = runs_markdown(summary)
    (out / "summary.md").write_text(text, encoding="utf-8")

    def hashed(paths):
        return [
            {"path": str(Path(p).resolve()), "sha256": sha256(p)}
            for p in paths
        ]

    write_json(
        out / "sources.json",
        {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(__file__),
            },
            "pool": str(pool),
            "reuse": bool(args.reuse),
            "runs": (hashed([out / "runs.jsonl"])[0] if args.reuse else None),
            "cells": hashed(args.cells),
            "shrink": hashed(args.shrink),
            "environment": identity(gpyreg_source),
            "torch_version": torch.__version__,
            "threads": {key: os.environ.get(key) for key in THREAD_KEYS},
        },
    )
    if not args.reuse:
        print(text, flush=True)

    if args.cells:
        runs_by_name = {r["name"]: r for r in records}
        shrink = load_shrink(args.shrink)
        joined = []
        skipped = 0
        with (out / "cells.jsonl").open("w", encoding="utf-8") as progress:
            for cell in load_cells(args.cells):
                if any(name not in runs_by_name for name in cell["entries"]):
                    skipped += 1
                    continue
                record = join_cell(cell, runs_by_name, shrink)
                joined.append(record)
                progress.write(json.dumps(record) + "\n")
        added = summarize_added(joined, np.random.default_rng(1))
        added["cells"] = [str(Path(p).resolve()) for p in args.cells]
        added["shrink"] = [str(Path(p).resolve()) for p in args.shrink]
        added["skipped_cells"] = skipped
        write_json(out / "added.json", added)
        text = added_markdown(added)
        (out / "added.md").write_text(text, encoding="utf-8")
        print(text, flush=True)
        if skipped:
            print(f"{skipped} cells skipped: inputs not scored", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
