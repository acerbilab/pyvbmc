"""Empirical-Bayes shrinkage of the stacked expected log joint, on recorded cells.

A stack's expected log joint ``G = Σ_k w_k I_k`` is optimistic on a noisy
target because the ``I_k`` are noisy GP estimates and the weights, VBMC's
own and then the stacking's, favour the components whose estimates came
out high. The classical correction for selecting on noisy estimates is to
shrink each estimate toward the population mean by the share of its
spread that is estimation noise: with ``I_k ~ N(θ_k, V_k)`` and
``θ_k ~ N(μ, τ²)``, the posterior mean of ``θ_k`` is
``μ + τ² / (τ² + V_k) (I_k − μ)``. Here ``V_k`` is the GP's own variance
of the estimate (the mean over hyperparameter samples of the diagonal of
``J_sjk`` plus the between-sample variance of ``I_sk``, the class's
``_expected_log_joint_variance`` at a unit weight), ``μ`` the mean of the
``I_k`` over the population and ``τ²`` their excess variance over the mean
``V_k`` (method of moments, floored at zero). Three variants:

- ``within``: the population is the run's own components, so every run is
  shrunk toward its own level with its own noise;
- ``within_full``: as ``within`` with the full estimation covariance
  ``Σ`` of the run's components (the estimates share one GP), the
  posterior mean ``μ + τ² (τ² I + Σ)⁻¹ (I − μ)``;
- ``stack``: the population is every component of the stack.

Only the value changes: ``G`` is re-evaluated at the cell's recorded
weights with the shrunken ``I_k``, and every variant's bias is scored
against the cell's ``elbo_mc`` with the cell's own entropy, as
``svbmc_cap_kappa.py`` scores the caps. Each cell's stack is rebuilt from
the pool artifacts with the recorded seeds and constructed, not
optimized, and the rebuild is checked against the cell's raw value.
Needs Torch on ``PYTHONPATH`` and the pool's gpyreg through
``--gpyreg-source``::

    PYTHONPATH="<TORCH_PATH>" python -u dev/scripts/svbmc_shrink_elbo.py \\
        --pool DIR --cells RESULTS.json --out DIR --gpyreg-source PATH \\
        [--conditions L1,L2]

Outputs under ``--out``: ``cells.jsonl`` (per cell: ``G``, ``H``,
``elbo_mc``, the raw and class-cap biases, and per variant the shrunken
``G``, its bias, the weight-averaged shrinkage factor and, per run,
``μ``, ``τ²`` and the mean ``V_k``), ``summary.json`` and ``summary.md``
(per condition and ``M``, medians over cells of every bias and of the
shrinkage factor and the noise share ``mean V / var I``).
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

VARIANTS = ("within", "within_full", "stack")


def run_estimates(vp, jacobian):
    """A run's Jacobian-corrected estimates, their variances and covariance.

    ``I_sk`` and ``J_sjk`` are per hyperparameter sample; the estimate is
    the mean over samples, its covariance the mean of the samples'
    covariances plus the between-sample covariance of the means.
    """
    I_sk = np.asarray(vp.stats["I_sk"], dtype=float)
    J_sjk = np.asarray(vp.stats["J_sjk"], dtype=float)
    I = I_sk.mean(axis=0) - jacobian
    covariance = J_sjk.mean(axis=0)
    if I_sk.shape[0] > 1:
        covariance = covariance + np.cov(I_sk, rowvar=False, ddof=1)
    covariance = 0.5 * (covariance + covariance.T)
    variance = np.maximum(np.diag(covariance), np.spacing(1.0))
    return I, variance, covariance


def moments(I, variance):
    """Population mean and excess variance (method of moments)."""
    mu = float(np.mean(I))
    spread = float(np.var(I, ddof=1)) if I.size > 1 else 0.0
    tau2 = max(spread - float(np.mean(variance)), 0.0)
    return mu, tau2, spread


def shrink_diagonal(I, variance, mu, tau2):
    factor = tau2 / (tau2 + variance)
    return mu + factor * (I - mu), factor


def shrink_full(I, covariance, mu, tau2):
    n = I.size
    factor = tau2 * np.linalg.solve(tau2 * np.eye(n) + covariance, np.eye(n))
    return mu + factor @ (I - mu), np.diag(factor)


def score_cell(cell, pool):
    from svbmc_pool_io import load_run

    from pyvbmc.svbmc import SVBMC

    arm = cell["arms"]["integrated"]
    vps = [
        load_run(pool / name, rng=seed)["vp"]
        for name, seed in zip(cell["entries"], cell["entry_seeds"])
    ]
    stacked = SVBMC(vps, seed=cell["cell_seed"], show_tips=False)
    w = np.asarray(arm["w"], dtype=float)
    if [int(k) for k in stacked.K] != list(arm["K"]):
        raise RuntimeError(
            f"{cell['condition']} M={cell['M']} r={cell['repetition']}: the "
            f"rebuilt stack retained runs {list(stacked.K)}, the recorded "
            f"weights {list(arm['K'])}"
        )
    H = float(arm["entropy"])
    G = float(arm["elbos"]["raw"]) - H
    elbo_mc = float(arm["elbo_mc"])
    offsets = np.concatenate([[0], np.cumsum(stacked.K)])
    jacobian = np.ravel(stacked._jacobian_corrections)
    runs = [
        run_estimates(vp, jacobian[offsets[m] : offsets[m + 1]])
        for m, vp in enumerate(stacked.vp_list)
    ]
    I_all = np.concatenate([I for I, _, _ in runs])
    V_all = np.concatenate([V for _, V, _ in runs])
    if abs(float(np.dot(w, I_all)) - G) > 1e-6:
        raise RuntimeError(
            f"{cell['condition']} M={cell['M']} r={cell['repetition']}: the "
            f"rebuilt expected log joint {np.dot(w, I_all):.6f} differs "
            f"from the recorded {G:.6f}"
        )
    record = {
        "condition": cell["condition"],
        "M": int(cell["M"]),
        "repetition": int(cell["repetition"]),
        "K_total": int(I_all.size),
        "G": G,
        "H": H,
        "elbo_mc": elbo_mc,
        "bias_raw": G + H - elbo_mc,
        "bias_class_cap": float(arm["elbos"]["capped_I_median"]) - elbo_mc,
        "variants": {},
        "runs": [],
    }
    shrunk = {name: np.empty_like(I_all) for name in VARIANTS}
    factors = {name: np.empty_like(I_all) for name in VARIANTS}
    mu_stack, tau2_stack, spread_stack = moments(I_all, V_all)
    for m, (I, V, C) in enumerate(runs):
        sl = slice(offsets[m], offsets[m + 1])
        mu, tau2, spread = moments(I, V)
        shrunk["within"][sl], factors["within"][sl] = shrink_diagonal(
            I, V, mu, tau2
        )
        shrunk["within_full"][sl], factors["within_full"][sl] = shrink_full(
            I, C, mu, tau2
        )
        shrunk["stack"][sl], factors["stack"][sl] = shrink_diagonal(
            I, V, mu_stack, tau2_stack
        )
        record["runs"].append(
            {
                "K": int(I.size),
                "mass": float(np.sum(w[sl])),
                "mu": mu,
                "tau2": tau2,
                "spread": spread,
                "mean_variance": float(np.mean(V)),
                "noise_share": float(np.mean(V) / spread) if spread else None,
            }
        )
    record["stack_population"] = {
        "mu": mu_stack,
        "tau2": tau2_stack,
        "spread": spread_stack,
        "mean_variance": float(np.mean(V_all)),
        "noise_share": (
            float(np.mean(V_all) / spread_stack) if spread_stack else None
        ),
    }
    for name in VARIANTS:
        G_shrunk = float(np.dot(w, shrunk[name]))
        record["variants"][name] = {
            "G": G_shrunk,
            "bias": G_shrunk + H - elbo_mc,
            "factor": float(np.dot(w, factors[name]) / np.sum(w)),
        }
    return record


def summarize(records):
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
            shares = [
                run["noise_share"]
                for r in rows
                for run in r["runs"]
                if run["noise_share"] is not None
            ]
            entry["M"].append(
                {
                    "M": M,
                    "cells": len(rows),
                    "bias_raw": float(
                        np.median([r["bias_raw"] for r in rows])
                    ),
                    "bias_class_cap": float(
                        np.median([r["bias_class_cap"] for r in rows])
                    ),
                    "noise_share_within": (
                        float(np.median(shares)) if shares else None
                    ),
                    "noise_share_stack": float(
                        np.median(
                            [
                                r["stack_population"]["noise_share"]
                                for r in rows
                                if r["stack_population"]["noise_share"]
                                is not None
                            ]
                        )
                    ),
                    "variants": {
                        name: {
                            "bias": float(
                                np.median(
                                    [r["variants"][name]["bias"] for r in rows]
                                )
                            ),
                            "abs_bias": float(
                                np.median(
                                    [
                                        abs(r["variants"][name]["bias"])
                                        for r in rows
                                    ]
                                )
                            ),
                            "factor": float(
                                np.median(
                                    [
                                        r["variants"][name]["factor"]
                                        for r in rows
                                    ]
                                )
                            ),
                        }
                        for name in VARIANTS
                    },
                }
            )
        conditions.append(entry)
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variants": list(VARIANTS),
        "cells": len(records),
        "conditions": conditions,
    }


def markdown(summary):
    lines = [
        "# Empirical-Bayes shrinkage of the stacked expected log joint",
        "",
        f"Generated {summary['generated']} from {summary['cells']} cells of "
        "the integrated arm. Medians over the cells of a condition and `M` "
        "of the bias against `elbo_mc`: `raw` the uncapped value, `class` "
        "the component-median cap the class applies, then each shrinkage "
        "variant with, in brackets, the weight-averaged shrinkage factor "
        "(1 leaves the estimates unchanged, 0 replaces them by the "
        "population mean). `noise share` is the mean estimation variance "
        "of the components over the spread of their estimates, the share "
        "of the spread that is noise, within a run and over the stack.",
        "",
        "| condition | M | raw | class | within [factor] | within full [factor] | stack [factor] | noise share within / stack |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in summary["conditions"]:
        for item in entry["M"]:
            v = item["variants"]
            share = item["noise_share_within"]
            lines.append(
                f"| {entry['condition'].replace('_svbmc', '')} | {item['M']} | "
                f"{item['bias_raw']:+.2f} | {item['bias_class_cap']:+.2f} | "
                f"{v['within']['bias']:+.2f} [{v['within']['factor']:.2f}] | "
                f"{v['within_full']['bias']:+.2f} "
                f"[{v['within_full']['factor']:.2f}] | "
                f"{v['stack']['bias']:+.2f} [{v['stack']['factor']:.2f}] | "
                f"{share if share is None else round(share, 2)} / "
                f"{item['noise_share_stack']:.2f} |"
            )
    lines += [
        "",
        "Median absolute bias over the cells of each condition and `M`:",
        "",
        "| condition | M | raw | class | within | within full | stack |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in summary["conditions"]:
        for item in entry["M"]:
            v = item["variants"]
            lines.append(
                f"| {entry['condition'].replace('_svbmc', '')} | {item['M']} | "
                f"{abs(item['bias_raw']):.2f} | {abs(item['bias_class_cap']):.2f} | "
                f"{v['within']['abs_bias']:.2f} | "
                f"{v['within_full']['abs_bias']:.2f} | "
                f"{v['stack']['abs_bias']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpyreg-source", type=Path, default=DEFAULT_GPYREG)
    parser.add_argument("--conditions", default=None)
    args = parser.parse_args(argv)
    for key in THREAD_KEYS:
        os.environ.setdefault(key, "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
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
            record = score_cell(cell, pool)
            records.append(record)
            progress.write(json.dumps(record) + "\n")
            progress.flush()
            if number % 100 == 0 or number == len(cells):
                print(
                    f"{number}/{len(cells)} cells, "
                    f"{(time.perf_counter() - started) / 60:.1f} min",
                    flush=True,
                )
    summary = summarize(records)
    summary["cells_file"] = str(args.cells.resolve())
    summary["pool"] = str(pool)
    write_json(out / "summary.json", summary)
    text = markdown(summary)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
