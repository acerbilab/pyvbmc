"""Re-optimize the stacking weights on the shrunken expected log joints.

``svbmc_shrink_elbo.py`` shrinks the components' expected log joints and
re-evaluates the stack at the weights the raw optimization chose, so only
the reported value changes. This script runs the optimization itself on
the shrunken values: for every recorded cell it rebuilds the stack from
the pool, replaces the class's corrected expected log joints by the
two-level full-covariance shrinkage of ``svbmc_shrink_elbo.py`` (within
each run with the full estimation covariance, plus the run-level shift),
optimizes the weights with the comparison's settings (the optimizer's
warm start, which the class takes from the runs' own reported ELBOs,
is shifted by the change of each run's own level under the shrinkage,
so that nothing the optimization reads is unshrunk), and scores the
result as the comparison scores a cell: draws from the new stack, the
target's noiseless log density over them, the entropy reference at the
new weights, ``elbo_mc``, the biases, the KL gap and the posterior
metrics. Recorded alongside: the raw optimization's biases and metrics
from the cell's record, the value-only shrinkage's bias from
``--shrink`` cells, the largest weight change between the two
optimizations, and with ``--single-runs`` (the ``runs.jsonl`` of
``svbmc_single_run_bias.py``) the added bias of every value, the bias
minus the mean bias of the cell's input runs. Needs Torch on
``PYTHONPATH`` and the pool's gpyreg through ``--gpyreg-source``::

    PYTHONPATH="<TORCH_PATH>" python -u dev/scripts/svbmc_shrink_optimize.py \\
        --pool DIR --cells RESULTS.json [--cells ...] --out DIR \\
        --gpyreg-source PATH [--shrink CELLS.jsonl ...] \\
        [--single-runs RUNS.jsonl] [--conditions L1,L2] [--M 3,5] [--limit N]

Outputs under ``--out``: ``cells.jsonl`` (per cell: the new weights'
``G``, ``H``, ``elbo_mc``, the biases ``shrunk_opt`` (the shrunken value
at the shrunken-optimized weights, the headline this variant would
report), ``raw_at_opt`` (the raw value at those weights) and, from the
records, ``raw`` (the raw optimization) and ``two_level_full`` (the
value-only shrinkage); the posterior metrics and KL gap of both
optimizations; ``max_abs_dw``; the inputs' mean bias and every added
bias), ``summary.json`` / ``summary.md`` (per condition and ``M``,
medians over cells with bootstrap intervals on the added bias, and the
fraction of cells where the new weights improve gsKL and MMTV) and
``sources.json``.
"""

import argparse
import json
import logging
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
    identity,
    pinned_gpyreg_source,
    write_json,
)

# isort: split
import numpy as np  # noqa: E402
from svbmc_pool_stack import (  # noqa: E402
    ENTROPY_REF_STREAM,
    LEARNING_RATE,
    N_DRAWS,
    N_SAMPLES,
    N_SAMPLES_FINAL,
    VERSION,
    Problems,
    bootstrap_median,
    entropy_reference,
    stacked_outcome,
)
from svbmc_shrink_elbo import (  # noqa: E402
    moments,
    run_estimates,
    shrink_diagonal,
    shrink_full,
)

VALUES = ("raw", "two_level_full", "shrunk_opt", "raw_at_opt")


def two_level_full(stacked):
    """The two-level full-covariance shrinkage of the stack's corrected
    expected log joints, as ``svbmc_shrink_elbo.py`` computes it."""
    offsets = np.concatenate([[0], np.cumsum(stacked.K)])
    jacobian = np.ravel(stacked._jacobian_corrections)
    runs = [
        run_estimates(vp, jacobian[offsets[m] : offsets[m + 1]])
        for m, vp in enumerate(stacked.vp_list)
    ]
    own = [np.ravel(vp.w) / np.sum(vp.w) for vp in stacked.vp_list]
    levels = np.array([float(np.dot(o, I)) for o, (I, _, _) in zip(own, runs)])
    level_variance = np.array(
        [float(o @ C @ o) for o, (_, _, C) in zip(own, runs)]
    )
    mu_r, tau2_r, _, _ = moments(levels, np.diag(level_variance))
    shrunk_levels, _ = shrink_diagonal(levels, level_variance, mu_r, tau2_r)
    shifts = shrunk_levels - levels
    shrunk = np.empty(int(np.sum(stacked.K)))
    for m, (I, V, C) in enumerate(runs):
        mu, tau2, _, _ = moments(I, C)
        values, _ = shrink_full(I, C, mu, tau2)
        shrunk[offsets[m] : offsets[m + 1]] = values + shifts[m]
    return shrunk


def fit_cell(cell, pool, problem, reference, max_steps, shrink, runs_by_name):
    from svbmc_pool_io import load_run

    from pyvbmc.svbmc import SVBMC

    arm = cell["arms"]["integrated"]
    started = time.perf_counter()
    vps = [
        load_run(pool / name, rng=seed)["vp"]
        for name, seed in zip(cell["entries"], cell["entry_seeds"])
    ]
    stacked = SVBMC(vps, seed=cell["cell_seed"], show_tips=False)
    if [int(k) for k in stacked.K] != list(arm["K"]):
        raise RuntimeError(
            f"{cell['condition']} M={cell['M']} r={cell['repetition']}: the "
            f"rebuilt stack retained runs {list(stacked.K)}, the recorded "
            f"weights {list(arm['K'])}"
        )
    I_raw = np.ravel(stacked.I_corrected).astype(float).copy()
    w_recorded = np.asarray(arm["w"], dtype=float)
    G_recorded = float(arm["elbos"]["raw"]) - float(arm["entropy"])
    if abs(float(np.dot(w_recorded, I_raw)) - G_recorded) > 1e-6:
        raise RuntimeError(
            f"{cell['condition']} M={cell['M']} r={cell['repetition']}: the "
            f"rebuilt expected log joint {np.dot(w_recorded, I_raw):.6f} "
            f"differs from the recorded {G_recorded:.6f}"
        )
    I_shrunk = two_level_full(stacked)
    # The objective and the reported value read ``I_corrected`` at call
    # time, so the optimization runs on the shrunken estimates.
    offsets = np.concatenate([[0], np.cumsum(stacked.K)])
    own = [np.ravel(vp.w) / np.sum(vp.w) for vp in stacked.vp_list]
    E_raw = np.array(
        [
            float(np.dot(I_raw[offsets[m] : offsets[m + 1]], o))
            for m, o in enumerate(own)
        ]
    )
    E_shrunk = np.array(
        [
            float(np.dot(I_shrunk[offsets[m] : offsets[m + 1]], o))
            for m, o in enumerate(own)
        ]
    )
    stacked.I_corrected = I_shrunk.reshape(1, -1)
    stacked.I = stacked.I_corrected + np.reshape(
        stacked._jacobian_corrections, (1, -1)
    )
    stacked.E_corrected = E_shrunk
    # The class warm-starts the logits from the runs' own reported
    # ELBOs; shift each by the change of its run's level, so that the
    # start favours the runs the shrunken values favour.
    stacked.individual_elbos = [
        float(elbo + E_shrunk[m] - E_raw[m])
        for m, elbo in enumerate(stacked.individual_elbos)
    ]
    construction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    stacked.optimize(
        n_samples=N_SAMPLES,
        lr=LEARNING_RATE,
        max_steps=max_steps,
        version=VERSION,
        n_samples_final=N_SAMPLES_FINAL,
    )
    optimize_seconds = time.perf_counter() - started
    w = np.ravel(stacked.w).astype(float)
    H = float(stacked.entropy)
    G_shrunk = float(np.dot(w, I_shrunk))
    G_raw_at_opt = float(np.dot(w, I_raw))
    samples = stacked.sample(N_DRAWS)
    outcome = stacked_outcome(
        problem,
        reference,
        samples,
        {"shrunk_opt": G_shrunk + H, "raw_at_opt": G_raw_at_opt + H},
        cell["cell_seed"],
        "integrated",
    )
    rng = np.random.default_rng([int(cell["cell_seed"]), ENTROPY_REF_STREAM])
    outcome.update(entropy_reference(stacked, w, rng))
    elbo_mc = float(outcome["e_log_joint_mc"] + outcome["entropy_ref"])
    ln_Z = problem.ln_Z
    key = (cell["condition"], int(cell["M"]), int(cell["repetition"]))
    record = {
        "condition": cell["condition"],
        "M": int(cell["M"]),
        "repetition": int(cell["repetition"]),
        "K_total": int(I_raw.size),
        "w": w.tolist(),
        "G_shrunk": G_shrunk,
        "G_raw_at_opt": G_raw_at_opt,
        "H": H,
        "elbo_mc": elbo_mc,
        "elbo_mc_sd": float(
            np.hypot(outcome["e_log_joint_mc_sd"], outcome["entropy_ref_sd"])
        ),
        "kl_gap": float("nan") if ln_Z is None else float(ln_Z - elbo_mc),
        "kl_gap_recorded": float(arm["kl_gap"]),
        "bias": {
            "shrunk_opt": G_shrunk + H - elbo_mc,
            "raw_at_opt": G_raw_at_opt + H - elbo_mc,
            "raw": float(arm["bias"]["raw"]),
            "two_level_full": (
                float(shrink[key]["variants"]["two_level_full"]["bias"])
                if key in shrink
                else float("nan")
            ),
        },
        "metrics": {
            "gskl": float(outcome["metrics"]["gskl"]),
            "mmtv": float(outcome["metrics"]["mmtv"]),
            "gskl_recorded": float(arm["metrics"]["gskl"]),
            "mmtv_recorded": float(arm["metrics"]["mmtv"]),
        },
        "level_shift": (E_shrunk - E_raw).tolist(),
        "max_abs_dw": float(np.max(np.abs(w - w_recorded))),
        "mass_shift": float(0.5 * np.sum(np.abs(w - w_recorded))),
        "construction_seconds": construction_seconds,
        "optimize_seconds": optimize_seconds,
        "reference_seconds": outcome["metrics_seconds"],
    }
    if runs_by_name:
        if len(cell["entries"]) != len(arm["K"]):
            raise RuntimeError(
                f"{cell['condition']} M={cell['M']} r={cell['repetition']}: "
                f"{len(cell['entries'])} inputs but the stack retained "
                f"{len(arm['K'])} runs"
            )
        inputs = [runs_by_name[name]["bias_vbmc"] for name in cell["entries"]]
        record["inputs_mean_bias"] = float(np.mean(inputs))
        record["added"] = {
            name: value - record["inputs_mean_bias"]
            for name, value in record["bias"].items()
        }
    return record


def summarize(records, rng):
    conditions = list(dict.fromkeys(r["condition"] for r in records))
    out = {"conditions": []}
    for condition in conditions:
        rows = [r for r in records if r["condition"] == condition]
        entry = {"condition": condition, "by_M": []}
        for M in sorted({r["M"] for r in rows}):
            cells = [r for r in rows if r["M"] == M]
            row = {
                "M": M,
                "n": len(cells),
                "bias": {
                    name: float(np.nanmedian([c["bias"][name] for c in cells]))
                    for name in VALUES
                },
                "gskl": float(
                    np.median([c["metrics"]["gskl"] for c in cells])
                ),
                "gskl_recorded": float(
                    np.median([c["metrics"]["gskl_recorded"] for c in cells])
                ),
                "mmtv": float(
                    np.median([c["metrics"]["mmtv"] for c in cells])
                ),
                "mmtv_recorded": float(
                    np.median([c["metrics"]["mmtv_recorded"] for c in cells])
                ),
                "gskl_improved": float(
                    np.mean(
                        [
                            c["metrics"]["gskl"]
                            < c["metrics"]["gskl_recorded"]
                            for c in cells
                        ]
                    )
                ),
                "mmtv_improved": float(
                    np.mean(
                        [
                            c["metrics"]["mmtv"]
                            < c["metrics"]["mmtv_recorded"]
                            for c in cells
                        ]
                    )
                ),
                "kl_gap": float(np.nanmedian([c["kl_gap"] for c in cells])),
                "kl_gap_recorded": float(
                    np.nanmedian([c["kl_gap_recorded"] for c in cells])
                ),
                "max_abs_dw": float(
                    np.median([c["max_abs_dw"] for c in cells])
                ),
                "mass_shift": float(
                    np.median([c["mass_shift"] for c in cells])
                ),
            }
            if "added" in cells[0]:
                row["inputs_mean_bias"] = float(
                    np.median([c["inputs_mean_bias"] for c in cells])
                )
                row["added"] = {
                    name: float(
                        np.nanmedian([c["added"][name] for c in cells])
                    )
                    for name in VALUES
                }
                row["added_interval"] = {
                    name: bootstrap_median(
                        [c["added"][name] for c in cells], rng
                    )
                    for name in ("shrunk_opt", "raw")
                }
            entry["by_M"].append(row)
        out["conditions"].append(entry)
    return out


def markdown(summary):
    lines = [
        "Weights re-optimized on the two-level full-covariance shrinkage of "
        "the expected log joints, scored against the new stack's own "
        "`elbo_mc`. Per condition and `M`, medians over cells: the bias of "
        "the raw optimization (from the record), of the value-only "
        "shrinkage at the raw weights, of the shrunken value at the "
        "shrunken-optimized weights (`shrunk_opt`, the headline this "
        "variant would report) and of the raw value at those weights; the "
        "added bias of `shrunk_opt` and of raw (bias minus the mean bias of "
        "the cell's input runs) with a bootstrap interval; gsKL and MMTV of "
        "both optimizations with the fraction of cells the new weights "
        "improve; the KL gap of both; and the largest weight change.",
        "",
    ]
    for c in summary["conditions"]:
        lines.append(f"## {c['condition']}")
        lines.append("")
        lines.append(
            "| M | n | raw | value-only | shrunk_opt | raw_at_opt | "
            "shrunk_opt added [CI] | raw added | gsKL new / recorded "
            "(improved) | MMTV new / recorded (improved) | KL gap new / "
            "recorded | max abs dw |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in c["by_M"]:
            b = r["bias"]
            if "added" in r:
                ci = r["added_interval"]["shrunk_opt"]
                added = (
                    f"{r['added']['shrunk_opt']:+.2f} "
                    f"[{ci['lo']:+.2f}, {ci['hi']:+.2f}] | "
                    f"{r['added']['raw']:+.2f}"
                )
            else:
                added = "n/a | n/a"
            lines.append(
                f"| {r['M']} | {r['n']} | {b['raw']:+.2f} | "
                f"{b['two_level_full']:+.2f} | {b['shrunk_opt']:+.2f} | "
                f"{b['raw_at_opt']:+.2f} | {added} | "
                f"{r['gskl']:.2f} / {r['gskl_recorded']:.2f} "
                f"({r['gskl_improved']:.2f}) | "
                f"{r['mmtv']:.3f} / {r['mmtv_recorded']:.3f} "
                f"({r['mmtv_improved']:.2f}) | "
                f"{r['kl_gap']:+.2f} / {r['kl_gap_recorded']:+.2f} | "
                f"{r['max_abs_dw']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--cells", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpyreg-source", type=Path, default=DEFAULT_GPYREG)
    parser.add_argument("--shrink", type=Path, action="append", default=[])
    parser.add_argument("--single-runs", type=Path, default=None)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--M", default=None)
    parser.add_argument("--limit", type=int, default=None)
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
    logging.getLogger("SVBMC").setLevel(logging.WARNING)

    only = [c.strip() for c in (args.conditions or "").split(",") if c.strip()]
    Ms = [int(m) for m in (args.M or "").split(",") if m.strip()]
    cells, max_steps = [], None
    for path in args.cells:
        results = json.loads(Path(path).read_text(encoding="utf-8"))
        steps = int(results["settings"]["max_steps"])
        if max_steps is not None and steps != max_steps:
            raise RuntimeError(
                "the cells files were run with different max_steps"
            )
        max_steps = steps
        for cell in results["cells"]:
            if "integrated" not in cell["arms"]:
                continue
            if only and cell["condition"] not in only:
                continue
            if Ms and int(cell["M"]) not in Ms:
                continue
            cells.append(cell)
    if args.limit:
        kept, count = [], {}
        for cell in cells:
            key = (cell["condition"], int(cell["M"]))
            if count.get(key, 0) < args.limit:
                kept.append(cell)
                count[key] = count.get(key, 0) + 1
        cells = kept
    shrink = {}
    for path in args.shrink:
        for line in Path(path).open(encoding="utf-8"):
            record = json.loads(line)
            shrink[
                (
                    record["condition"],
                    int(record["M"]),
                    int(record["repetition"]),
                )
            ] = record
    runs_by_name = None
    if args.single_runs:
        runs_by_name = {
            r["name"]: r
            for r in (
                json.loads(line)
                for line in args.single_runs.open(encoding="utf-8")
            )
        }
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    problems = Problems()
    if runs_by_name is not None:
        # A cell whose inputs were not scored has no yardstick; skip it
        # as the join in svbmc_single_run_bias.py does, and say so.
        unscored = [
            c
            for c in cells
            if any(name not in runs_by_name for name in c["entries"])
        ]
        if unscored:
            print(
                f"{len(unscored)} cells skipped: inputs not in the single-run "
                "file",
                flush=True,
            )
            cells = [c for c in cells if c not in unscored]
    records = []
    started = time.perf_counter()
    with (out / "cells.jsonl").open("w", encoding="utf-8") as progress:
        for number, cell in enumerate(cells, start=1):
            problem = problems.get(cell["condition"], "run")
            reference = problems.reference(cell["condition"], "run")
            record = fit_cell(
                cell, pool, problem, reference, max_steps, shrink, runs_by_name
            )
            records.append(record)
            progress.write(json.dumps(record) + "\n")
            progress.flush()
            if number % 20 == 0 or number == len(cells):
                print(
                    f"{number}/{len(cells)} cells, "
                    f"{(time.perf_counter() - started) / 60:.1f} min "
                    f"({cell['condition']} M={cell['M']})",
                    flush=True,
                )
    summary = summarize(records, np.random.default_rng(0))
    summary["settings"] = {
        "max_steps": max_steps,
        "n_samples": N_SAMPLES,
        "lr": LEARNING_RATE,
        "version": VERSION,
        "n_samples_final": N_SAMPLES_FINAL,
        "n_draws": N_DRAWS,
        "M": Ms or "all",
        "limit": args.limit,
        "warm_start": "the runs' reported ELBOs shifted by the change of "
        "their own level under the shrinkage",
    }
    write_json(out / "summary.json", summary)
    text = markdown(summary)
    (out / "summary.md").write_text(text, encoding="utf-8")
    from svbmc_pool_io import sha256

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
            "cells": hashed(args.cells),
            "shrink": hashed(args.shrink),
            "single_runs": (
                hashed([args.single_runs])[0] if args.single_runs else None
            ),
            "environment": identity(gpyreg_source),
            "torch_version": torch.__version__,
            "threads": {key: os.environ.get(key) for key in THREAD_KEYS},
        },
    )
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
