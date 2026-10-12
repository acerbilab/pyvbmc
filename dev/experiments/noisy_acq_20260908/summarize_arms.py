"""Markdown tables of the end-to-end arms from the per-run JSON files in
``arms/``: per configuration, medians over seeds, the usable fraction and
the paired columns against the baseline arm on the same seeds.

    python summarize_arms.py [ARMS_DIR]
"""

import json
import sys
from pathlib import Path

import numpy as np

ORDER = [
    "baseline_here",
    "refits_off",
    "gp_off_vp_on",
    "gp_map_inloop",
    "var_reduction",
    "ns_search_2048",
    "eig",
    "eig_components",
    "repeat3",
    "combo",
    "hard_baseline",
    "hard_var_reduction",
    "hard_repeat3",
    "hard_combo",
]
BASELINE = {"baseline_here", "hard_baseline"}
LABEL = {
    "baseline_here": "defaults (VIQR, both refits)",
    "refits_off": "both refits off",
    "gp_off_vp_on": "GP refit off, VP refit on",
    "gp_map_inloop": "in-loop GP refit MAP-only (`ns_gp_max_active = 0`)",
    "var_reduction": '`AcqFcnVIQR(loss="var_reduction")`',
    "ns_search_2048": "`ns_search = 2048`",
    "eig": "`AcqFcnEIG()`",
    "eig_components": "`AcqFcnEIG(components=True)`",
    "repeat3": "`max_repeated_observations = 3`",
    "combo": "`var_reduction` + repeats",
    "hard_baseline": "defaults (VIQR, both refits)",
    "hard_var_reduction": '`AcqFcnVIQR(loss="var_reduction")`',
    "hard_repeat3": "`max_repeated_observations = 3`",
    "hard_combo": "`var_reduction` + repeats",
}


def load(path):
    rows = {}
    for p in sorted(path.glob("*_seed*.json")):
        d = json.loads(p.read_text())
        rows[(d["label"], d["seed"])] = d["final"]
    return rows


def main(arms_dir):
    arms_dir = Path(arms_dir)
    data = {a: load(arms_dir / a) for a in ORDER if (arms_dir / a).is_dir()}
    configs = sorted({k[0] for rows in data.values() for k in rows})
    for cfg in configs:
        base = {}
        for a in BASELINE:
            base.update(
                {k: v for k, v in data.get(a, {}).items() if k[0] == cfg}
            )
        seeds = sorted({k[1] for k in base})
        print(
            f"\n**{cfg}** (seeds {seeds[0]}–{seeds[-1]}; paired columns"
            " against the defaults on the same seeds)\n"
        )
        print(
            "| arm | n | gsKL | MMTV | ELBO err | usable | evals | GP rows |"
            " wall min | gsKL ratio | wins | wall ratio |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a, r_all in data.items():
            r = {k: v for k, v in r_all.items() if k[0] == cfg}
            if not r:
                continue
            g = np.array([v["gskl"] for v in r.values()])
            m = np.array([v["mmtv"] for v in r.values()])
            e = np.array([v["elbo_err"] for v in r.values()])
            ev = np.array([v["func_count"] for v in r.values()])
            n_gp = np.array([v["final_N"] for v in r.values()])
            w = np.array([v["wall_s"] for v in r.values()]) / 60
            usable = np.mean((e < 1) & (g < 1) & (m < 0.2))
            if a in BASELINE:
                paired = " | | |"
            else:
                common = [k for k in r if k in base]
                gr = np.exp(
                    np.median(
                        [
                            np.log(r[k]["gskl"] / base[k]["gskl"])
                            for k in common
                        ]
                    )
                )
                wr = np.median(
                    [r[k]["wall_s"] / base[k]["wall_s"] for k in common]
                )
                wins = sum(r[k]["gskl"] < base[k]["gskl"] for k in common)
                paired = f" {gr:.2f} | {wins}/{len(common)} | {wr:.2f} |"
            print(
                f"| {LABEL.get(a, a)} | {len(r)} | {np.median(g):.3f} |"
                f" {np.median(m):.3f} | {np.median(e):.3f} | {usable:.2f} |"
                f" {np.median(ev):.0f} | {np.median(n_gp):.0f} |"
                f" {np.median(w):.1f} |{paired}"
            )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "arms")
