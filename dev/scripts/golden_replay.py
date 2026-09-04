"""Replay golden configurations with the current code and compare each run
with its stored trace.

Developer tooling for Stage 2 of the modernization plan: the per-step
trajectory gate of ``dev/plans/stage2-batched-acquisition.md`` (roadmap
pickup point 2). Targets, options and the trace format come from
``golden_trace.py``, so a replayed run is built exactly as the baseline
population was::

    python -u dev/scripts/golden_replay.py
    python -u dev/scripts/golden_replay.py --configs cigar_D4,logreg_D5 --seeds 0-1
    python dev/scripts/golden_replay.py --report-only --out <dir>   # re-render

Run it as a script from the repository root (it imports its neighbours by
module name). For each (config, seed) the script runs VBMC in this process
with one BLAS thread (as the baseline was run), writes the new trace under
``--out`` and reports, against the stored trace of the same (config, seed)
under ``--baseline``:

- the agreement horizon: the first iteration at which the ELBO path
  differs (exactly, and beyond 1e-6), and how many leading *live* evaluated
  points are identical. The trace stores only the rows that survive
  warm-up trimming, so the point horizon is a lower bound on the true
  evaluation horizon; the ELBO horizon is the primary measure, and an
  identical iteration 0 certifies an identical initial design;
- the final metrics side by side, and whether the new run's ΔLML, gsKL and
  MMTV lie inside the baseline population's envelope for that config
  (Tukey far-out fence, ``Q3 + 3 IQR`` over the seeds' sidecars under
  ``--sidecars``, in git; the plain maximum is vacuous where a seed is a
  known failure, e.g. ``student_D4`` seed 19).

An arithmetic-preserving refactor is expected to *part* from the stored
trajectory at some point: a few-ulp change in an acquisition value flips a
CMA-ES ranking and the search ends elsewhere. What must hold is that
iteration 0 is identical (the initial design is drawn from the generator
before any numerics run) and that a parted run's finals stay inside the
envelope (an identical run is exempt: its own seed may be the
population's far outlier). Exit code 1 if any run fails, an iteration 0
differs, a final is not finite, a parted run's accuracy metric exceeds
the envelope, or nothing was compared. Without the baseline ``.npz``
traces (a fresh checkout; they are gitignored) only the final-metric
comparison runs. Timers, wall times and memory figures are never
compared.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_BASELINE = REPO_ROOT / "dev" / "scripts" / "runs" / "golden"
DEFAULT_BASELINE = DEFAULT_BASELINE / "baseline_20260903"
DEFAULT_SIDECARS = REPO_ROOT / "dev" / "golden" / "baseline"
DEFAULT_OUT_ROOT = REPO_ROOT / "dev" / "scripts" / "runs" / "golden"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

# Cheap configurations covering the regimes the Stage 2 items touch: a
# Gaussian at D = 5, the D = 2 banana, the bounded (probit) half-normal,
# the warped large-K cigar and the noisy VIQR path. About 7 minutes at
# seed 0 on the reference laptop.
DEFAULT_CONFIGS = (
    "normal_D5",
    "banana_D2",
    "halfnormal_D2",
    "cigar_D4",
    "rosenbrock_D2_noise1",
)
ACCURACY = ("elbo_err", "gskl", "mmtv")
FINAL_KEYS = ACCURACY + ("func_count", "iterations", "final_K", "wall_s")
RTOL_CLOSE = 1e-6
ATOL_ELBO = 1e-6
FENCE_IQR = 3.0  # Tukey far-out fence: Q3 + 3 IQR


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help="comma-separated golden labels",
    )
    ap.add_argument("--seeds", default="0", help="e.g. 0 or 0-2 or 0,3")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--sidecars", type=Path, default=DEFAULT_SIDECARS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--threads",
        type=int,
        default=1,
        help="BLAS threads (the baseline was made with 1)",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="do not run; re-render the report for the traces in --out",
    )
    return ap.parse_args(argv)


def _horizon(a, b, rtol, atol):
    """Number of leading rows of `a` and `b` that agree to (rtol, atol)."""
    import numpy as np

    n = min(len(a), len(b))
    if n == 0:
        return 0
    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)
    ok = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    if ok.ndim > 1:
        ok = ok.reshape(n, -1).all(axis=1)
    bad = np.flatnonzero(~ok)
    return int(bad[0]) if len(bad) else n


def compare_traces(ref, new):
    """Agreement horizons between two trace archives (dict-like)."""
    out = {}
    for key in ("X_orig", "y_orig"):
        out[f"{key}_exact"] = _horizon(ref[key], new[key], 0.0, 0.0)
        out[f"{key}_close"] = _horizon(ref[key], new[key], RTOL_CLOSE, 0.0)
    out["n_live_ref"] = int(len(ref["X_orig"]))
    out["n_live_new"] = int(len(new["X_orig"]))
    out["elbo_exact_iter"] = _horizon(ref["elbo"], new["elbo"], 0.0, 0.0)
    out["elbo_iter"] = _horizon(ref["elbo"], new["elbo"], 0.0, ATOL_ELBO)
    for key in ("func_count", "K", "Ns_gp"):
        out[f"{key}_iter"] = _horizon(ref[key], new[key], 0.0, 0.0)
    out["n_iter_ref"] = int(len(ref["elbo"]))
    out["n_iter_new"] = int(len(new["elbo"]))
    out["identical"] = bool(
        out["n_live_ref"] == out["n_live_new"]
        and out["X_orig_exact"] == out["n_live_ref"]
        and out["y_orig_exact"] == out["n_live_ref"]  # noisy targets
        and out["n_iter_ref"] == out["n_iter_new"]
        and out["elbo_exact_iter"] == out["n_iter_ref"]
    )
    out["initial_design_ok"] = out["elbo_exact_iter"] >= 1
    return out


def envelope(values):
    """Tukey far-out fence ``Q3 + 3 IQR`` of the finite values."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")
    q1, q3 = np.percentile(v, [25, 75])
    return float(q3 + FENCE_IQR * (q3 - q1))


def compare_run(label, seed, out_dir, baseline, sidecars, pop):
    """One row of the report for a finished (label, seed) under out_dir."""
    import numpy as np
    from golden_trace import _tag

    tag = _tag(label, seed)
    row = {"label": label, "seed": seed, "ok": True}
    side_new = json.loads((out_dir / f"{tag}.json").read_text())
    fin_new = side_new["final"]
    row["final_new"] = {k: fin_new.get(k) for k in FINAL_KEYS}

    ref_json = sidecars / f"{tag}.json"
    if not ref_json.exists():
        ref_json = baseline / f"{tag}.json"
    if ref_json.exists():
        fin_ref = json.loads(ref_json.read_text())["final"]
        row["final_ref"] = {k: fin_ref.get(k) for k in FINAL_KEYS}

    ref_npz, new_npz = baseline / f"{tag}.npz", out_dir / f"{tag}.npz"
    if ref_npz.exists() and new_npz.exists():
        with np.load(ref_npz) as ref, np.load(new_npz) as new:
            row.update(compare_traces(ref, new))

    outside = []
    pop_ok = label in pop and np.isfinite(pop[label]["func_count"]).any()
    if pop_ok:
        row["pop_fence"] = {m: envelope(pop[label][m]) for m in ACCURACY}
        row["pop_evals"] = [
            int(np.nanmin(pop[label]["func_count"])),
            int(np.nanmax(pop[label]["func_count"])),
        ]
        # An identical replay cannot be an outlier of its own population
        # (the reference seed may itself be the population's far outlier,
        # e.g. student_D4 seed 19), so the envelope applies only to runs
        # that parted. A non-finite final is always a flag.
        for m in ACCURACY:
            v = fin_new.get(m)
            v = float("nan") if v is None else float(v)
            fence = row["pop_fence"][m]
            if not np.isfinite(v) or (
                not row.get("identical") and np.isfinite(fence) and v > fence
            ):
                outside.append(m)
    row["outside"] = outside

    if row.get("identical"):
        verdict = "identical"
    elif "elbo_exact_iter" in row:
        # Iterations are 0-based: "parted at iteration i" means iterations
        # 0..i-1 are bit-identical and iteration i is the first to differ.
        verdict = f"parted at iteration {row['elbo_exact_iter']}"
        if row["elbo_iter"] > row["elbo_exact_iter"]:
            verdict += f" (beyond 1e-6 at {row['elbo_iter']})"
        verdict += f"; live points identical: {row['X_orig_exact']}"
        if not row["initial_design_ok"]:
            verdict = (
                "ITERATION 0 DIFFERS (initial design or GP fit); " + verdict
            )
    else:
        verdict = "finals only"
    if outside:
        verdict += "; OUTSIDE envelope: " + ", ".join(outside)
    row["flagged"] = bool(outside) or not row.get("initial_design_ok", True)
    row["verdict"] = verdict
    return row


def _fmt(v, nd=3):
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return "nan"
    if f.is_integer() and abs(f) >= 10:
        return f"{int(f)}"
    return f"{f:.{nd}g}"


def render(rows, git, args, minutes):
    lines = [
        f"# Golden replay {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Code `{git['sha']}`{' (dirty)' if git['dirty'] else ''};"
        f" baseline `{args.baseline.name}`; threads {args.threads};"
        f" {minutes:.1f} min.",
        "",
        "| config | seed | verdict | identical iterations / iters ref →"
        " new | live points identical / ref → new | ΔLML ref →"
        " new (fence) | gsKL ref → new (fence) | MMTV ref → new (fence) |"
        " evals ref → new [pop] | wall min ref → new |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    n_flag = 0
    for row in rows:
        n_flag += int(row.get("flagged", not row["ok"]))
        fr = row.get("final_ref", {})
        fn = row.get("final_new", {})
        fence = row.get("pop_fence", {})
        pe = row.get("pop_evals")

        def pair(m, nd=3):
            s = f"{_fmt(fr.get(m), nd)} → {_fmt(fn.get(m), nd)}"
            if m in fence:
                s += f" ({_fmt(fence[m], nd)})"
            return s

        if "elbo_exact_iter" in row:
            it = (
                f"{row['elbo_exact_iter']} / {row['n_iter_ref']} →"
                f" {row['n_iter_new']}"
            )
            ev = (
                f"{row['X_orig_exact']} / {row['n_live_ref']} →"
                f" {row['n_live_new']}"
            )
        else:
            ev = it = "-"
        evals = f"{_fmt(fr.get('func_count'))} → {_fmt(fn.get('func_count'))}"
        if pe:
            evals += f" [{pe[0]}, {pe[1]}]"
        wall = (
            f"{_fmt((fr.get('wall_s') or float('nan')) / 60, 2)} →"
            f" {_fmt((fn.get('wall_s') or float('nan')) / 60, 2)}"
        )
        lines.append(
            f"| {row['label']} | {row['seed']} | {row['verdict']} | {it} |"
            f" {ev} | {pair('elbo_err')} | {pair('gskl')} | {pair('mmtv')}"
            f" | {evals} | {wall} |"
        )
    lines.append("")
    lines.append(
        f"{n_flag} flagged of {len(rows)}."
        " `identical` = same live points and ELBO path. `parted at"
        " iteration i` (0-based) = iterations 0..i−1 are bit-identical and"
        " iteration i is the first to differ (expected for an"
        " arithmetic-preserving change once a CMA-ES ranking flips); the"
        " live-point count is a lower bound on the evaluation horizon"
        " because warm-up trimming removes rows. Flags: a run failed,"
        " iteration 0 differs, a final is not finite, or a parted run's"
        " ΔLML/gsKL/MMTV exceeds the population's Q3 + 3 IQR fence."
    )
    return "\n".join(lines) + "\n", n_flag


def main(argv=None):
    args = parse_args(argv)
    for k in THREAD_KEYS:
        os.environ[k] = str(args.threads)
    os.environ.setdefault("MPLBACKEND", "Agg")
    if hasattr(sys.stdout, "reconfigure"):  # the report is not cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sys.path.insert(0, str(HERE))
    from golden_trace import _tag, load_population, parse_seeds, run_task
    from profile_run import git_info

    try:
        import psutil  # noqa: F401
    except ImportError:
        sys.exit("golden_replay.py needs psutil (pip install psutil)")

    labels = [c for c in args.configs.split(",") if c]
    seeds = parse_seeds(args.seeds)
    if args.report_only and args.out is None:
        sys.exit("--report-only needs --out")
    out_dir = args.out or (
        DEFAULT_OUT_ROOT / f"replay_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pop = load_population(args.sidecars) if args.sidecars.exists() else {}
    have_traces = args.baseline.exists()
    git = git_info()
    if not args.report_only:
        print(
            f"[replay] {len(labels)} configs x {len(seeds)} seeds, code"
            f" {git['sha']}{' (dirty)' if git['dirty'] else ''}, threads"
            f" {args.threads}, baseline traces"
            f" {'found' if have_traces else 'ABSENT (finals only)'} ->"
            f" {out_dir}",
            flush=True,
        )

    rows = []
    t_all = time.time()
    for label in labels:
        for seed in seeds:
            tag = _tag(label, seed)
            if args.report_only:
                if not (out_dir / f"{tag}.json").exists():
                    continue
            else:
                print(f"[replay] {tag} ...", flush=True)
                r = run_task(label, seed, {}, out_dir)
                if not r["ok"]:
                    rows.append(
                        {
                            "label": label,
                            "seed": seed,
                            "ok": False,
                            "flagged": True,
                            "verdict": "FAILED (see .error.txt)",
                        }
                    )
                    print(f"[replay] {tag:32s} FAILED", flush=True)
                    continue
            row = compare_run(
                label, seed, out_dir, args.baseline, args.sidecars, pop
            )
            rows.append(row)
            fn = row["final_new"]
            if not args.report_only:
                print(
                    f"[replay] {tag:32s} {fn['wall_s'] / 60:4.1f} min "
                    f" {row['verdict']}  elbo_err={fn['elbo_err']:.3g}"
                    f" gskl={fn['gskl']:.3g} mmtv={fn['mmtv']:.3g}"
                    f" evals={fn['func_count']}",
                    flush=True,
                )

    if not rows:
        print(f"[replay] nothing to report under {out_dir}", flush=True)
        return 1
    minutes = (time.time() - t_all) / 60
    if args.report_only:  # keep the provenance of the run being re-rendered
        prev = out_dir / "replay.json"
        if prev.exists():
            saved = json.loads(prev.read_text())
            git = saved.get("git", git)
            minutes = saved.get("minutes", minutes)
            args.threads = saved.get("threads", args.threads)
    report, n_flag = render(rows, git, args, minutes)
    (out_dir / "replay.md").write_text(report, encoding="utf-8")
    (out_dir / "replay.json").write_text(
        json.dumps(
            {
                "git": git,
                "threads": args.threads,
                "minutes": minutes,
                "rows": rows,
            },
            indent=1,
            default=str,
        )
    )
    print(report, flush=True)
    return 1 if n_flag else 0


if __name__ == "__main__":
    sys.exit(main())
