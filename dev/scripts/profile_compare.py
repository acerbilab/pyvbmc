"""Compare two ``profile_suite.py`` campaigns config by config.

Developer tooling; see ``dev/plans/stage2-gpyreg-predict-and-sampler.md``
§Results for the first use. Reads the two ``aggregate.json`` files and
prints, for every plain run present in both, the wall and per-stage seconds
with their new/old ratios, whether the trajectory is the same (iterations,
evaluations, ``N``, ``K``, ``Ns`` and the final metrics equal), and the
control ratio: the time of a stage the change under test does not touch
(``--control variational_fit`` by default). A control ratio far from 1.0
means the machine, not the code, ran at a different speed during that
config (a laptop in desktop use slowed three configs by 1.3–2.4× on
2026-09-05), so that row measures nothing. The speed probe of
``profile_suite.py`` brackets the campaign; this brackets each config.

For the cProfile runs present in both it prints each bucket's seconds
(share × profiled wall) and call counts, and the per-call time of
``GP.predict``, ``SliceSampler.sample``, the acquisition and ``train_gp``.

Example::

    python dev/scripts/profile_compare.py dev/scripts/runs/profile_20260905 \
        dev/scripts/runs/profile_20260905_item8
"""

import argparse
import json
from pathlib import Path

IDENT = [
    "iterations",
    "evals",
    "final_N",
    "final_K",
    "min_Ns_gp",
    "elbo_err",
    "gskl",
    "mmtv",
    "rmse",
]
STAGES = ["active_sampling", "gp_train", "variational_fit", "finalize"]
PER_CALL = [
    "GP.predict",
    "SliceSampler.sample",
    "acquisition __call__",
    "train_gp",
]


def load(path):
    rows = json.loads((Path(path) / "aggregate.json").read_text())
    return {r["tag"]: r for r in rows}


def same_trajectory(b, n):
    return all(b[k] == n[k] for k in IDENT)


def table(header, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for row in [header] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def compare_plain(base, new, control):
    print(
        "## Plain runs: seconds old -> new (ratio new/old);"
        f" control = {control}"
    )
    header = ["config", "wall", "ratio"]
    for s in STAGES:
        header += [s, "ratio"]
    header += ["control", "trajectory"]
    rows = []
    total_b = total_n = 0.0
    for tag, b in base.items():
        if b["mode"] != "plain" or tag not in new:
            continue
        n = new[tag]
        if not tag.startswith("probe"):
            total_b += b["wall_s"]
            total_n += n["wall_s"]
        cells = [
            tag,
            f'{b["wall_s"]:.1f} -> {n["wall_s"]:.1f}',
            f'{n["wall_s"] / b["wall_s"]:.2f}',
        ]
        for s in STAGES:
            bs, ns = b[s + "_s"], n[s + "_s"]
            cells += [f"{bs:.1f} -> {ns:.1f}", f"{ns / bs:.2f}" if bs else "-"]
        bc, nc = b[control + "_s"], n[control + "_s"]
        cells.append(f"{nc / bc:.2f}" if bc else "-")
        if same_trajectory(b, n):
            cells.append("same")
        else:
            cells.append(
                "DIFFERENT: " + ", ".join(k for k in IDENT if b[k] != n[k])
            )
        rows.append(cells)
    table(header, rows)
    if total_b:
        print(
            f"\nSuite wall without probes: {total_b:.0f} -> {total_n:.0f} s"
            f" (ratio {total_n / total_b:.2f}, {total_b / total_n:.2f}x)"
        )
    missing = [
        t for t, r in base.items() if r["mode"] == "plain" and t not in new
    ]
    if missing:
        print("not in the new campaign:", ", ".join(missing))


def compare_cprof(base, new):
    configs = [
        r["config"]
        for r in base.values()
        if r["mode"] == "cprof" and r["tag"] in new
    ]
    if not configs:
        return
    print(
        "\n## cProfile: bucket seconds (share x profiled wall) old -> new [calls]"
    )
    print(
        "profiled wall: "
        + ", ".join(
            f'{c} {base[c + "_cprof"]["wall_s"]:.1f} -> '
            f'{new[c + "_cprof"]["wall_s"]:.1f} s'
            for c in configs
        )
    )
    any_cprof = next(r for r in base.values() if r["mode"] == "cprof")
    buckets = [k[5:] for k in any_cprof if k.startswith("attr:")]
    rows = []
    for bk in buckets:
        cells = [bk]
        for c in configs:
            b, n = base[c + "_cprof"], new[c + "_cprof"]
            bs = b["attr:" + bk] / 100 * b["wall_s"]
            ns = n["attr:" + bk] / 100 * n["wall_s"]
            cb, cn = b["calls:" + bk], n["calls:" + bk]
            calls = f"{cb}" if cb == cn else f"{cb}->{cn}"
            ratio = f" ({ns / bs:.2f})" if bs > 0.05 else ""
            cells.append(f"{bs:.1f}->{ns:.1f}s{ratio} [{calls}]")
        rows.append(cells)
    for bk in PER_CALL:
        if "attr:" + bk not in any_cprof:
            continue
        cells = [bk + " per call"]
        for c in configs:
            b, n = base[c + "_cprof"], new[c + "_cprof"]
            pb = b["attr:" + bk] / 100 * b["wall_s"] / max(b["calls:" + bk], 1)
            pn = n["attr:" + bk] / 100 * n["wall_s"] / max(n["calls:" + bk], 1)
            unit, f = ("ms", 1e3) if pb > 1e-3 else ("us", 1e6)
            ratio = f" ({pn / pb:.2f})" if pb else ""
            cells.append(f"{pb * f:.2f}->{pn * f:.2f} {unit}{ratio}")
        rows.append(cells)
    table(["bucket"] + configs, rows)
    for c in configs:
        b, n = base[c + "_cprof"], new[c + "_cprof"]
        verdict = "same" if same_trajectory(b, n) else "DIFFERENT"
        print(f"{c}: trajectory {verdict}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", help="campaign directory of the baseline")
    ap.add_argument("new", help="campaign directory of the new code")
    ap.add_argument(
        "--control",
        default="variational_fit",
        choices=STAGES,
        help="stage the change does not touch; its ratio is the per-config"
        " machine-speed control (default: variational_fit)",
    )
    args = ap.parse_args(argv)
    base, new = load(args.base), load(args.new)
    compare_plain(base, new, args.control)
    compare_cprof(base, new)


if __name__ == "__main__":
    main()
