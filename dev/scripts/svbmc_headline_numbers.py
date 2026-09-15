"""Every number the S-VBMC headline documents quote, from the tracked cells.

The headline note (``dev/2026-09-15-svbmc-headline-shrinkage.md``) and the
shrinkage and cap sections of the stage D report
(``dev/results/2026-09-15-svbmc-pool-comparison.md``) quote medians over
cells, worst and mean cases over the noisy conditions, threshold sweeps
and paired bootstrap intervals. This prints all of them from the tracked
per-cell records of ``svbmc_shrink_elbo.py`` and ``svbmc_cap_kappa.py``
under ``dev/experiments/svbmc_pool/``, so that a sentence in those
documents can be checked or updated without the raw directories::

    python dev/scripts/svbmc_headline_numbers.py [--experiments DIR]

Reads ``shrink_20260915/cells.jsonl``, ``shrink_M35_20260915/cells.jsonl``,
``cap_kappa_20260915/cells.jsonl``, ``cap_kappa_M35_20260915/cells.jsonl``
and the ``summary.json`` and ``added.json`` of ``single_run_20260915/``
(the single-run biases and the added-bias ranges). Needs NumPy only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_EXPERIMENTS = ROOT / "dev" / "experiments" / "svbmc_pool"
SHRINK_DIRS = ("shrink_20260915", "shrink_M35_20260915")
CAP_DIRS = ("cap_kappa_20260915", "cap_kappa_M35_20260915")
GRID = (2, 3, 4, 5, 8, 16)
SHORT = {
    "multisensory_s1_D6_noise3_svbmc": "multisensory noise 3",
    "multisensory_s1_D6_noise1.3_svbmc": "multisensory noise 1.3",
    "rosenbrock_D2_noise3_svbmc": "Rosenbrock noise 3",
    "gmm_D2_noise3_svbmc": "GMM noise 3",
    "ring_D2_noise3_svbmc": "ring noise 3",
    "student_D8_noise3_svbmc": "Student D8 noise 3",
    "gmm_D2_svbmc": "GMM (noiseless)",
    "multisensory_s1_D6_svbmc": "multisensory (noiseless)",
}
RULES = {
    "raw": lambda r: r["bias_raw"],
    "cap": lambda r: r["bias_class_cap"],
    "within": lambda r: r["variants"]["within"]["bias"],
    "within_full": lambda r: r["variants"]["within_full"]["bias"],
    "stack": lambda r: r["variants"]["stack"]["bias"],
    "run_level": lambda r: r["variants"]["run_level"]["bias"],
    "two_level": lambda r: r["variants"]["two_level"]["bias"],
    "two_level_full": lambda r: r["variants"]["two_level_full"]["bias"],
    "hybrid": lambda r: r["variants"]["hybrid"]["bias"],
}


def load(experiments, names):
    rows = []
    for name in names:
        path = Path(experiments) / name / "cells.jsonl"
        rows += [json.loads(line) for line in path.open(encoding="utf-8")]
    return rows


def median(values):
    return float(np.median(values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--experiments", type=Path, default=DEFAULT_EXPERIMENTS
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    rows = load(args.experiments, SHRINK_DIRS)
    caps = load(args.experiments, CAP_DIRS)
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    noisy = [c for c in conditions if "noise" in c]
    controls = [c for c in conditions if "noise" not in c]
    rng = np.random.default_rng(0)

    def cells(c, M):
        return [r for r in rows if r["condition"] == c and r["M"] == M]

    def bias(c, M, rule):
        return median([RULES[rule](r) for r in cells(c, M)])

    def factor(c, M, name):
        return median([r["variants"][name]["factor"] for r in cells(c, M)])

    def share(c, M):
        return median(
            [
                r["noise_share"]
                for r in cells(c, M)
                if r["noise_share"] is not None
            ]
        )

    print(
        "# Report, M = 16 table: raw | cap | within [f] | within full [f] | stack [f] | noise share"
    )
    for c in conditions:
        print(
            f"| {SHORT[c]} | {bias(c, 16, 'raw'):+.2f} | {bias(c, 16, 'cap'):+.2f} | "
            f"{bias(c, 16, 'within'):+.2f} [{factor(c, 16, 'within'):.2f}] | "
            f"{bias(c, 16, 'within_full'):+.2f} [{factor(c, 16, 'within_full'):.2f}] | "
            f"{bias(c, 16, 'stack'):+.2f} [{factor(c, 16, 'stack'):.2f}] | {share(c, 16):.2f} |"
        )
    print(
        "controls, max |median bias| over M:",
        {
            rule: round(
                max(abs(bias(c, M, rule)) for c in controls for M in GRID), 3
            )
            for rule in (
                "raw",
                "cap",
                "within",
                "within_full",
                "stack",
                "two_level_full",
                "hybrid",
            )
        },
    )

    print(
        "\n# Report, two-level table at M = 2 / 4 / 8 / 16: two_level_full | two_level | run_level | cap"
    )
    for c in conditions:
        print(
            f"| {SHORT[c]} | "
            + " | ".join(
                " / ".join(f"{bias(c, M, rule):+.2f}" for M in (2, 4, 8, 16))
                for rule in ("two_level_full", "two_level", "run_level", "cap")
            )
            + " |"
        )

    print(
        "\n# Report, M = 2 to 5 table: raw / cap / within_full / two_level_full"
    )
    for c in conditions:
        print(
            f"| {SHORT[c]} | "
            + " | ".join(
                " / ".join(
                    f"{bias(c, M, rule):+.2f}"
                    for rule in ("raw", "cap", "within_full", "two_level_full")
                )
                for M in (2, 3, 4, 5)
            )
            + " |"
        )

    print(
        "\n# Worst and mean over the six noisy conditions of the median absolute bias, per M"
    )
    for rule in (
        "raw",
        "cap",
        "within",
        "within_full",
        "two_level_full",
        "hybrid",
    ):
        worst = [max(abs(bias(c, M, rule)) for c in noisy) for M in GRID]
        mean = [np.mean([abs(bias(c, M, rule)) for c in noisy]) for M in GRID]
        print(
            f"  {rule:15s} worst "
            + " / ".join(f"{x:.2f}" for x in worst)
            + "   mean "
            + " / ".join(f"{x:.2f}" for x in mean)
        )

    print("\n# Prose numbers")
    for M in GRID:
        d = {
            c: abs(bias(c, M, "two_level_full") - bias(c, M, "within_full"))
            for c in conditions
        }
        worst = max(d, key=d.get)
        print(
            f"  max |two_level_full - within_full| at M={M}: {d[worst]:.3f} ({SHORT[worst]})"
        )
    print(
        "  run-level tau2 == 0 fraction and factor 10th-90th percentile, per condition at M = 2 to 5:"
    )
    for c in conditions:
        parts = []
        for M in (2, 3, 4, 5):
            rs = cells(c, M)
            zero = np.mean([r["run_population"]["tau2"] == 0 for r in rs])
            f = [r["variants"]["run_level"]["factor"] for r in rs]
            parts.append(
                f"M={M}: zero {zero:.2f}, factor {np.percentile(f, 10):.2f}-{np.percentile(f, 90):.2f}"
            )
        print(f"    {SHORT[c]}: " + "; ".join(parts))
    print("  hybrid chooses the cap, cells per condition:")
    for c in conditions:
        rs = [r for r in rows if r["condition"] == c]
        print(
            f"    {SHORT[c]}: {sum(r['variants']['hybrid']['cap'] for r in rs)}/{len(rs)}"
        )
    print(
        "  cell noise share: median over all cells, 5th-95th percentile, min, max:"
    )
    for c in conditions:
        s = np.array(
            [
                r["noise_share"]
                for r in rows
                if r["condition"] == c and r["noise_share"] is not None
            ]
        )
        print(
            f"    {SHORT[c]}: median {np.median(s):.3f}, 5-95% {np.percentile(s, 5):.3f}-{np.percentile(s, 95):.3f}, "
            f"min {s.min():.3f}, max {s.max():.3f}"
        )

    def hybrid(r, t):
        s = r["noise_share"]
        if s is not None and s >= t:
            return r["bias_class_cap"]
        return r["variants"]["within_full"]["bias"]

    print(
        "  hybrid threshold sweep, worst and mean over the noisy conditions at M = 2 to 16:"
    )
    for t in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30):
        worst = [
            max(
                abs(median([hybrid(r, t) for r in cells(c, M)])) for c in noisy
            )
            for M in GRID
        ]
        mean = [
            np.mean(
                [
                    abs(median([hybrid(r, t) for r in cells(c, M)]))
                    for c in noisy
                ]
            )
            for M in GRID
        ]
        print(
            f"    t={t:.2f}: worst "
            + " ".join(f"{x:.2f}" for x in worst)
            + "  mean "
            + " ".join(f"{x:.2f}" for x in mean)
        )
    print(
        "  |cap| - |two_level_full| (median biases), min..max over M, per noisy condition (positive: shrinkage closer):"
    )
    for c in noisy:
        d = [
            abs(bias(c, M, "cap")) - abs(bias(c, M, "two_level_full"))
            for M in GRID
        ]
        print(f"    {SHORT[c]}: {min(d):+.2f} .. {max(d):+.2f}")
    print(
        "  Student: |two_level_full| and |raw| at M = 2 to 5, fraction of cells where two_level_full is worse:"
    )
    for M in (2, 3, 4, 5):
        rs = cells("student_D8_noise3_svbmc", M)
        worse = np.mean(
            [
                abs(r["variants"]["two_level_full"]["bias"])
                > abs(r["bias_raw"])
                for r in rs
            ]
        )
        print(
            f"    M={M}: {abs(bias('student_D8_noise3_svbmc', M, 'two_level_full')):.2f} against "
            f"{abs(bias('student_D8_noise3_svbmc', M, 'raw')):.2f}; worse in {worse:.2f} of cells"
        )
    print(
        "  interval [two_level_full, raw] contains the reference, fraction of cells per noisy condition:"
    )
    for c in noisy:
        rs = [r for r in rows if r["condition"] == c]
        inside = np.mean(
            [
                min(r["variants"]["two_level_full"]["bias"], r["bias_raw"])
                <= 0
                <= max(r["variants"]["two_level_full"]["bias"], r["bias_raw"])
                for r in rs
            ]
        )
        print(f"    {SHORT[c]}: {inside:.2f}")

    print(
        "\n# Caps (svbmc_cap_kappa.py): max bind fraction and max |median move from raw| per cap"
    )
    names = [f"kappa_{k:g}" for k in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)] + [
        "E_max",
        "E_top",
    ]
    for name in names:
        binds, moves = [], []
        for c in conditions:
            for M in GRID:
                rs = [r for r in caps if r["condition"] == c and r["M"] == M]
                binds.append(np.mean([r["caps"][name]["binds"] for r in rs]))
                moves.append(
                    abs(
                        median([r["caps"][name]["bias"] for r in rs])
                        - median([r["bias_raw"] for r in rs])
                    )
                )
        print(
            f"  {name}: max binds {max(binds):.2f}, max |median move| {max(moves):.2f}"
        )
    for name in ("kappa_0.9", "kappa_0.95", "kappa_0.99"):
        typical = [
            median(
                [
                    r["caps"][name]["bias"]
                    for r in caps
                    if r["condition"] == c and r["M"] == M
                ]
            )
            for c in noisy
            if "student" not in c
            for M in GRID
        ]
        student = [
            median(
                [
                    r["caps"][name]["bias"]
                    for r in caps
                    if r["condition"] == "student_D8_noise3_svbmc"
                    and r["M"] == M
                ]
            )
            for M in GRID
        ]
        print(
            f"  {name}: typical residual {min(typical):+.2f}..{max(typical):+.2f}; Student {min(student):+.2f}..{max(student):+.2f}"
        )

    print(
        "\n# Paired bootstrap over cells (10000 resamples) of the difference of median |bias|, at M = 3 / 5 / 16"
    )
    for a_name, b_name in (
        ("cap", "two_level_full"),
        ("hybrid", "two_level_full"),
    ):
        print(f"  {a_name} minus {b_name} (positive: {b_name} closer):")
        for c in noisy:
            parts = []
            for M in (3, 5, 16):
                rs = cells(c, M)
                a = np.array([abs(RULES[a_name](r)) for r in rs])
                b = np.array([abs(RULES[b_name](r)) for r in rs])
                index = rng.integers(0, len(rs), size=(10000, len(rs)))
                d = np.median(a[index], axis=1) - np.median(b[index], axis=1)
                parts.append(
                    f"M={M}: {np.median(a) - np.median(b):+.2f} [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]"
                )
            print(f"    {SHORT[c]}: " + "; ".join(parts))
    print("\n# Single runs and what stacking adds (single_run_20260915)")
    single_dir = Path(args.experiments) / "single_run_20260915"
    single = json.loads(
        (single_dir / "summary.json").read_text(encoding="utf-8")
    )
    for c in single["conditions"]:
        v = c["bias_vbmc"]
        print(
            f"  {SHORT[c['condition']]}: reported-ELBO bias median "
            f"{v['median']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}], quartiles "
            f"{v['q25']:+.2f}..{v['q75']:+.2f}; class raw "
            f"{c['bias_raw']['median']:+.2f}; cap {c['bias_cap']['median']:+.2f}"
        )
    added = json.loads((single_dir / "added.json").read_text(encoding="utf-8"))
    noisy_added = [c for c in added["conditions"] if "noise" in c["condition"]]
    for name in ("raw", "capped_I_median", "run_level", "two_level_full"):
        parts = []
        for M in GRID:
            v = [
                r["added"][name]
                for c in noisy_added
                for r in c["by_M"]
                if r["M"] == M
            ]
            parts.append(f"M={M}: {min(v):+.2f}..{max(v):+.2f}")
        print(f"  added bias of {name}, noisy conditions: " + "; ".join(parts))
    for M in GRID:
        rows = [r for c in noisy_added for r in c["by_M"] if r["M"] == M]
        pos = [r["raw_added_positive"] for r in rows]
        best = [r["inputs_best_bias"] - r["bias"]["raw"] for r in rows]
        fr = [1 - r["added"]["run_level"] / r["added"]["raw"] for r in rows]
        print(
            f"  M={M}: raw added > 0 in {min(pos):.2f}..{max(pos):.2f} of "
            f"cells; best input minus stack raw {min(best):+.2f}..{max(best):+.2f}; "
            f"run_level removes {min(fr):.2f}..{max(fr):.2f} of the addition"
        )
    controls_added = [
        abs(r["added"][n])
        for c in added["conditions"]
        if "noise" not in c["condition"]
        for r in c["by_M"]
        for n in r["added"]
    ]
    print(
        f"  controls: max |added| over estimates and M {max(controls_added):.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
