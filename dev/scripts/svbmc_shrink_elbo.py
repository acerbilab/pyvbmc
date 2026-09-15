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
``I_k`` over the population and ``τ²`` their excess variance over the
noise (method of moments, floored at zero). One GP estimates a run's
components jointly, so their errors are correlated, and the noise that
the sample variance of ``K`` estimates carries is
``(tr Σ − 1ᵀ Σ 1 / K) / (K − 1)`` for the estimation covariance ``Σ``,
which is the mean ``V_k`` only when ``Σ`` is diagonal; a shared error
moves every estimate together and adds nothing to their spread. The
variants:

- ``within``: the population is the run's own components, so every run is
  shrunk toward its own level with its own noise;
- ``within_full``: as ``within`` with the full estimation covariance
  ``Σ`` of the run's components (the estimates share one GP), the
  posterior mean ``μ + τ² (τ² I + Σ)⁻¹ (I − μ)``;
- ``stack``: the population is every component of the stack;
- ``run_level``: the runs are the population. Each run's level is its
  own weighted expected log joint ``E_m`` (its posterior weights times
  its ``I_k``) with the estimation variance ``w_mᵀ Σ_m w_m``; the level is
  shrunk toward the runs' mean by the share of the runs' spread that is
  estimation noise, and every component of the run is shifted by the
  change of its level. This targets the selection among runs, which the
  within-run populations cannot see;
- ``two_level`` and ``two_level_full``: the run-level shift composed with
  ``within`` or ``within_full``.

Each cell also records its noise share, the mass-weighted mean over its
runs of the within-run share, and the ``hybrid`` rule built on it: the
class's component-median cap when the share is at least
``HYBRID_SHARE``, the ``within_full`` shrinkage otherwise. The share is
what separates the targets the cap is right on (components of similar
density, a large share of their spread being estimation noise) from
those it over-corrects (a heavy-tailed target, whose components' spread
is real).

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
``G``, its bias, the weight-averaged shrinkage factor (for the
full-covariance forms the row sums of the shrinkage matrix, the factor a
deviation shared by every component is kept by) and, per run,
``μ``, ``τ²`` and the mean ``V_k``), ``summary.json`` and ``summary.md``
(per condition and ``M``, medians over cells of every bias and of the
shrinkage factor and the noise share, the noise in the spread over the
spread) and ``sources.json`` (the script's and the cells file's hashes,
the pool, the process's identity).

The method, its derivation and a worked example are in
``dev/2026-09-15-svbmc-shrinkage-explained.md``.
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
    analysis_sources,
    write_json,
)

# isort: split
import numpy as np  # noqa: E402

VARIANTS = (
    "within",
    "within_full",
    "stack",
    "run_level",
    "two_level",
    "two_level_full",
    "hybrid",
)
#: The noise share at or above which the ``hybrid`` rule applies the
#: class's cap instead of the within-run shrinkage.
HYBRID_SHARE = 0.2


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


def noise_in_spread(covariance):
    """The part of the sample variance of correlated estimates that their
    estimation covariance accounts for: ``(tr Σ − 1ᵀ Σ 1 / K) / (K − 1)``."""
    K = covariance.shape[0]
    if K < 2:
        return 0.0
    return float((np.trace(covariance) - np.sum(covariance) / K) / (K - 1))


def moments(I, covariance):
    """Population mean, excess variance and the noise in the spread."""
    mu = float(np.mean(I))
    spread = float(np.var(I, ddof=1)) if I.size > 1 else 0.0
    noise = noise_in_spread(covariance)
    tau2 = max(spread - noise, 0.0)
    return mu, tau2, spread, noise


def block_diagonal(blocks):
    total = sum(block.shape[0] for block in blocks)
    matrix = np.zeros((total, total))
    offset = 0
    for block in blocks:
        n = block.shape[0]
        matrix[offset : offset + n, offset : offset + n] = block
        offset += n
    return matrix


def shrink_diagonal(I, variance, mu, tau2):
    factor = tau2 / (tau2 + variance)
    return mu + factor * (I - mu), factor


def shrink_full(I, covariance, mu, tau2):
    """The posterior mean under the full estimation covariance; the
    reported factor is the row sum of the shrinkage matrix."""
    n = I.size
    if tau2 <= 0.0:
        return np.full(n, mu), np.zeros(n)
    factor = tau2 * np.linalg.solve(tau2 * np.eye(n) + covariance, np.eye(n))
    return mu + factor @ (I - mu), factor.sum(axis=1)


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
    shrunk = {
        name: np.empty_like(I_all) for name in VARIANTS if name != "hybrid"
    }
    factors = {
        name: np.empty_like(I_all) for name in VARIANTS if name != "hybrid"
    }
    mu_stack, tau2_stack, spread_stack, noise_stack = moments(
        I_all, block_diagonal([C for _, _, C in runs])
    )
    # Run level: each run's own weighted expected log joint and its
    # estimation variance, shrunk toward the runs' mean.
    own = [np.ravel(vp.w) / np.sum(vp.w) for vp in stacked.vp_list]
    levels = np.array([float(np.dot(o, I)) for o, (I, _, _) in zip(own, runs)])
    level_variance = np.array(
        [float(o @ C @ o) for o, (_, _, C) in zip(own, runs)]
    )
    mu_runs, tau2_runs, spread_runs, noise_runs = moments(
        levels, np.diag(level_variance)
    )
    shrunk_levels, level_factors = shrink_diagonal(
        levels, level_variance, mu_runs, tau2_runs
    )
    shifts = shrunk_levels - levels
    record["run_population"] = {
        "mu": mu_runs,
        "tau2": tau2_runs,
        "spread": spread_runs,
        "mean_variance": float(np.mean(level_variance)),
        "noise": noise_runs,
        "noise_share": noise_runs / spread_runs if spread_runs else None,
    }
    for m, (I, V, C) in enumerate(runs):
        sl = slice(offsets[m], offsets[m + 1])
        mu, tau2, spread, noise = moments(I, C)
        shrunk["within"][sl], factors["within"][sl] = shrink_diagonal(
            I, V, mu, tau2
        )
        shrunk["within_full"][sl], factors["within_full"][sl] = shrink_full(
            I, C, mu, tau2
        )
        shrunk["stack"][sl], factors["stack"][sl] = shrink_diagonal(
            I, V, mu_stack, tau2_stack
        )
        shrunk["run_level"][sl] = I + shifts[m]
        factors["run_level"][sl] = level_factors[m]
        shrunk["two_level"][sl] = shrunk["within"][sl] + shifts[m]
        factors["two_level"][sl] = factors["within"][sl] * level_factors[m]
        shrunk["two_level_full"][sl] = shrunk["within_full"][sl] + shifts[m]
        factors["two_level_full"][sl] = (
            factors["within_full"][sl] * level_factors[m]
        )
        record["runs"].append(
            {
                "K": int(I.size),
                "mass": float(np.sum(w[sl])),
                "mu": mu,
                "tau2": tau2,
                "spread": spread,
                "mean_variance": float(np.mean(V)),
                "noise": noise,
                "noise_share": noise / spread if spread else None,
                "level": float(levels[m]),
                "level_variance": float(level_variance[m]),
                "level_shift": float(shifts[m]),
                "level_factor": float(level_factors[m]),
            }
        )
    record["stack_population"] = {
        "mu": mu_stack,
        "tau2": tau2_stack,
        "spread": spread_stack,
        "mean_variance": float(np.mean(V_all)),
        "noise": noise_stack,
        "noise_share": noise_stack / spread_stack if spread_stack else None,
    }
    # The cell's share: mass-weighted over the runs whose share is
    # defined (a run of one component, or of identical estimates, has no
    # spread to attribute); a cell with no defined share has none.
    defined = [run for run in record["runs"] if run["noise_share"] is not None]
    masses = np.array([run["mass"] for run in defined])
    record["noise_share"] = (
        float(
            np.sum(masses * np.array([run["noise_share"] for run in defined]))
            / np.sum(masses)
        )
        if defined and np.sum(masses) > 0
        else None
    )
    for name in VARIANTS:
        if name == "hybrid":
            continue
        G_shrunk = float(np.dot(w, shrunk[name]))
        record["variants"][name] = {
            "G": G_shrunk,
            "bias": G_shrunk + H - elbo_mc,
            "factor": float(np.dot(w, factors[name]) / np.sum(w)),
        }
    use_cap = (
        record["noise_share"] is not None
        and record["noise_share"] >= HYBRID_SHARE
    )
    record["variants"]["hybrid"] = {
        "G": (
            float(arm["elbos"]["capped_I_median"]) - H
            if use_cap
            else record["variants"]["within_full"]["G"]
        ),
        "bias": (
            record["bias_class_cap"]
            if use_cap
            else record["variants"]["within_full"]["bias"]
        ),
        "factor": 0.0
        if use_cap
        else record["variants"]["within_full"]["factor"],
        "cap": bool(use_cap),
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
            run_shares = [
                r["run_population"]["noise_share"]
                for r in rows
                if r["run_population"]["noise_share"] is not None
            ]
            stack_shares = [
                r["stack_population"]["noise_share"]
                for r in rows
                if r["stack_population"]["noise_share"] is not None
            ]
            cell_shares = [
                r["noise_share"] for r in rows if r["noise_share"] is not None
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
                    "noise_share_stack": (
                        float(np.median(stack_shares))
                        if stack_shares
                        else None
                    ),
                    "noise_share_runs": (
                        float(np.median(run_shares)) if run_shares else None
                    ),
                    "noise_share_cell": (
                        float(np.median(cell_shares)) if cell_shares else None
                    ),
                    "hybrid_cap_fraction": float(
                        np.mean([r["variants"]["hybrid"]["cap"] for r in rows])
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
        "hybrid_share": HYBRID_SHARE,
        "cells": len(records),
        "conditions": conditions,
    }


def markdown(summary):
    names = list(summary["variants"])
    lines = [
        "# Empirical-Bayes shrinkage of the stacked expected log joint",
        "",
        f"Generated {summary['generated']} from {summary['cells']} cells of "
        "the integrated arm. Medians over the cells of a condition and `M` "
        "of the bias against `elbo_mc`: `raw` the uncapped value, `class` "
        "the component-median cap the class applies, then each shrinkage "
        "variant with, in brackets, the weight-averaged shrinkage factor "
        "(for the full-covariance forms the row sum of the shrinkage "
        "matrix; 1 leaves the estimates unchanged, 0 replaces them by the "
        "population mean). `noise share` is the mean estimation variance "
        "of the components over the spread of their estimates, the share "
        "of the spread that is noise, within a run, over the stack's "
        "components and over the runs' levels. `hybrid` is the class's cap "
        f"when the cell's noise share is at least {summary['hybrid_share']} "
        "and `within_full` otherwise; its bracket is the fraction of cells "
        "on which it chose the cap.",
        "",
        "| condition | M | raw | class | "
        + " | ".join(f"{name} [factor]" for name in names)
        + " | noise share within / stack / runs |",
        "|" + "---|" * (5 + len(names)),
    ]

    def share(value):
        return "-" if value is None else f"{value:.2f}"

    for entry in summary["conditions"]:
        for item in entry["M"]:
            v = item["variants"]
            lines.append(
                f"| {entry['condition'].replace('_svbmc', '')} | {item['M']} | "
                f"{item['bias_raw']:+.2f} | {item['bias_class_cap']:+.2f} | "
                + " | ".join(
                    f"{v[name]['bias']:+.2f} "
                    + (
                        f"[{item['hybrid_cap_fraction']:.2f}]"
                        if name == "hybrid"
                        else f"[{v[name]['factor']:.2f}]"
                    )
                    for name in names
                )
                + f" | {share(item['noise_share_within'])} / "
                f"{share(item['noise_share_stack'])} / "
                f"{share(item.get('noise_share_runs'))} |"
            )
    lines += [
        "",
        "Median absolute bias over the cells of each condition and `M`:",
        "",
        "| condition | M | raw | class | " + " | ".join(names) + " |",
        "|" + "---|" * (4 + len(names)),
    ]
    for entry in summary["conditions"]:
        for item in entry["M"]:
            v = item["variants"]
            lines.append(
                f"| {entry['condition'].replace('_svbmc', '')} | {item['M']} | "
                f"{abs(item['bias_raw']):.2f} | {abs(item['bias_class_cap']):.2f} | "
                + " | ".join(f"{v[name]['abs_bias']:.2f}" for name in names)
                + " |"
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
    write_json(
        out / "sources.json",
        analysis_sources(__file__, args.cells, pool, args.gpyreg_source),
    )
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
