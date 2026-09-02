"""Run ``profile_run.py`` over a benchmark suite and aggregate the results.

Developer tooling; see ``dev/plans/benchmark-suite-and-golden-traces.md``.
Each config of a suite (``benchmark_targets.SUITES``) is run in its own
subprocess, sequentially, plain and/or under cProfile, each streaming to its
own log under the campaign directory. Finished runs (``summary.json``
present) are skipped, so an interrupted campaign can be resumed with the
same ``--out``.

Examples::

    python -u dev/scripts/profile_suite.py --suite profile --mode plain
    python -u dev/scripts/profile_suite.py --suite profile --mode cprof \
        --out dev/scripts/runs/profile_1756850000
    python dev/scripts/profile_suite.py --aggregate dev/scripts/runs/profile_1756850000

``--aggregate`` writes ``aggregate.md`` (two tables: wall time and stage
balance from the plain runs; cProfile attribution from the cprof runs) and
``aggregate.json`` next to the run directories.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from benchmark_targets import SUITES, suite_configs

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_RUNS = REPO_ROOT / "dev" / "scripts" / "runs"
PROFILE_RUN = HERE / "profile_run.py"

STAGES = (
    "active_sampling",
    "gp_train",
    "variational_fit",
    "finalize",
    "warping",
)

# Attribution labels reported in the aggregate (must match profile_run.py).
BUCKETS = [
    "active_sample",
    "cma.fmin",
    "acquisition __call__",
    "GP.predict",
    "vp.pdf",
    "active_importance_sampling",
    "train_gp",
    "SliceSampler.sample",
    "scipy.optimize.minimize",
    "optimize_vp",
    "_gp_log_joint",
    "_eval_full_elcbo",
    "entmc_vbmc",
    "final_boost",
    "determine_best_vp (incl. in-loop warping calls)",
    "copy.deepcopy",
]


def run_one(cfg, mode, out_dir, seed, extra):
    tag = f"{cfg.label}_{mode}"
    run_dir = out_dir / tag
    if (run_dir / "summary.json").exists():
        print(f"[suite] skip {tag} (summary.json exists)", flush=True)
        return True
    log = out_dir / f"{tag}.log"
    cmd = [
        sys.executable,
        "-u",
        str(PROFILE_RUN),
        "--config",
        cfg.label,
        "--seed",
        str(seed),
        "--out",
        str(out_dir),
        "--tag",
        tag,
    ]
    if mode == "cprof":
        cmd.append("--cprofile")
    cmd += extra
    print(
        f"[suite] start {tag} at {time.strftime('%H:%M:%S')} -> {log.name}",
        flush=True,
    )
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(
            cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO_ROOT
        )
    dt = time.time() - t0
    status = "done" if rc == 0 else f"FAILED rc={rc}"
    print(f"[suite] {status} {tag} in {dt / 60:.1f} min", flush=True)
    return rc == 0


def aggregate(out_dir):
    rows = []
    for summ in sorted(out_dir.glob("*/summary.json")):
        s = json.loads(summ.read_text())
        r = s["result"]
        wall = r["wall_s"]
        row = {
            "tag": summ.parent.name,
            "config": s["meta"].get("config"),
            "mode": "cprof" if s["meta"].get("cprofile") else "plain",
            "wall_s": wall,
            "untimed_s": r.get("untimed_s"),
            "iterations": r["recorded_iterations"],
            "evals": r["func_count"],
            "final_N": r.get("final_N"),
            "final_K": r["final_K"],
            "min_Ns_gp": r.get("min_Ns_gp"),
            "elbo_err": r.get("elbo_err"),
            "gskl": r.get("gskl"),
            "mmtv": r.get("mmtv"),
            "rmse": r.get("posterior_mean_rmse"),
            "message": r.get("message"),
        }
        for st in STAGES:
            v = s["stage_totals_s"].get(st, 0.0)
            row[f"{st}_s"] = v
            row[f"{st}_pct"] = 100 * v / wall if wall else float("nan")
        if "attribution" in s:
            att = {a["label"]: a for a in s["attribution"]}
            opt = att["VBMC.optimize"]["cumtime"]
            for b in BUCKETS:
                a = att.get(b)
                row[f"attr:{b}"] = (
                    100 * a["cumtime"] / opt if a and opt else None
                )
                row[f"calls:{b}"] = a["ncalls"] if a else None
        rows.append(row)
    (out_dir / "aggregate.json").write_text(json.dumps(rows, indent=2))

    def fmt(v, nd=1):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    lines = [f"# Profile campaign {out_dir.name}", ""]
    plain = [r for r in rows if r["mode"] == "plain"]
    if plain:
        lines += [
            "## Plain runs (true wall time; stages nest, untimed = wall - sum)",
            "",
            "| config | wall s | untimed s | iters | evals | N | K | min Ns | "
            "act.samp % | gp_train % | var.fit % | finalize % | warping % | "
            "elbo_err | gskl | mmtv | rmse | termination |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in plain:
            lines.append(
                f"| {r['config']} | {fmt(r['wall_s'])} | {fmt(r['untimed_s'])} |"
                f" {r['iterations']} | {r['evals']} | {fmt(r['final_N'])} |"
                f" {r['final_K']} | {fmt(r['min_Ns_gp'])} |"
                + "".join(f" {fmt(r[f'{st}_pct'])} |" for st in STAGES)
                + f" {fmt(r['elbo_err'], 3)} | {fmt(r['gskl'], 3)} |"
                f" {fmt(r['mmtv'], 3)} | {fmt(r['rmse'], 3)} |"
                f" {(r['message'] or '')[:40]} |"
            )
        lines.append("")
    cprof = [r for r in rows if r["mode"] == "cprof"]
    if cprof:
        lines += [
            "## cProfile attribution (% of profiled VBMC.optimize; calls in"
            " parentheses)",
            "",
            "| bucket | " + " | ".join(r["config"] for r in cprof) + " |",
            "|---|" + "---|" * len(cprof),
        ]
        for b in BUCKETS:
            cells = []
            for r in cprof:
                v = r.get(f"attr:{b}")
                c = r.get(f"calls:{b}")
                cells.append("-" if v is None else f"{v:.1f} ({c})")
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "| profiled wall s | "
            + " | ".join(fmt(r["wall_s"]) for r in cprof)
            + " |",
        ]
        lines.append("")
    (out_dir / "aggregate.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[suite] wrote {out_dir / 'aggregate.md'}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--suite", default="profile", choices=list(SUITES) + ["all"]
    )
    ap.add_argument(
        "--mode", default="plain", choices=["plain", "cprof", "both"]
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--only", default=None, help="comma-separated config labels"
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="campaign directory"
    )
    ap.add_argument("--aggregate", type=Path, default=None)
    ap.add_argument(
        "extra", nargs="*", help="extra args passed to profile_run"
    )
    args = ap.parse_args(argv)

    if args.aggregate:
        aggregate(args.aggregate)
        return 0

    out_dir = args.out or (DEFAULT_RUNS / f"profile_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfgs = suite_configs(args.suite)
    if args.only:
        wanted = set(args.only.split(","))
        cfgs = [c for c in cfgs if c.label in wanted]
    modes = ["plain", "cprof"] if args.mode == "both" else [args.mode]
    print(f"[suite] {len(cfgs)} configs x {modes} -> {out_dir}", flush=True)
    t0 = time.time()
    ok = True
    for mode in modes:
        for cfg in cfgs:
            ok &= run_one(cfg, mode, out_dir, args.seed, args.extra)
    print(
        f"[suite] campaign finished in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    aggregate(out_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
