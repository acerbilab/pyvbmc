"""Profile one PyVBMC run and report where the wall-clock time goes.

Developer tooling for the modernization work (see
``dev/2026-09-02-modernization-discussion.md``, sections 2 and 10, and
``dev/plans/benchmark-suite-and-golden-traces.md``). Runs VBMC on a target
from ``benchmark_targets.py`` under a fixed seed and reports:

* the per-stage timers VBMC already records (``pyvbmc.timer.main_timer``,
  snapshotted into ``iteration_history["timer"]`` every iteration): totals
  and a per-iteration table. Note that ``active_sampling`` wraps the whole
  active-sampling stage, which internally starts ``gp_train`` and
  ``variational_fit`` timers for intermediate refits, so stage totals do not
  sum to the wall-clock time; ``untimed_s`` (wall minus the sum) is therefore
  a lower bound on the post-loop work (``determine_best_vp``, ``final_boost``);
* with ``--cprofile``, a cProfile of ``VBMC.optimize()``: top functions by
  cumulative and internal time, plus cumulative time attributed to a curated
  list of hot-path functions so the numbers line up with the devlog tables.

Output goes to ``dev/scripts/runs/<tag>/`` (gitignored):
``summary.json`` (metadata, results with truth-based metrics where the target
has ground truth, stage totals, per-iteration table, attribution),
``profile.prof`` (open with ``snakeviz`` or ``pstats``) and ``profile.txt``.

Examples::

    python -u dev/scripts/profile_run.py --D 5 --seed 0 --cprofile
    python -u dev/scripts/profile_run.py --problem lumpy --D 4
    python -u dev/scripts/profile_run.py --config banana_D2_noise1 --cprofile
    python -u dev/scripts/profile_run.py --problem normal --D 5 \
        --options '{"tol_stable_excpt_frac": -1000000}' --tag exhaust

``--config`` takes a label from ``benchmark_targets.SUITES`` (see
``benchmark_targets.py --list``) and sets problem, dimension, noise and
options at once; the other flags then override.

The targets are deliberately cheap so what is measured is algorithm overhead,
not target cost.
"""

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from benchmark_targets import TARGET_NAMES, find_config, make_problem, metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "dev" / "scripts" / "runs"

# Options whose effective value is worth recording because VBMC rewrites them
# (noisy targets) or because a suite config sets them.
EFFECTIVE_OPTION_KEYS = (
    "max_fun_evals",
    "max_iter",
    "tol_stable_count",
    "tol_stable_excpt_frac",
    "specify_target_noise",
    "search_acq_fcn",
    "active_sample_gp_update",
    "active_sample_vp_update",
    "min_final_components",
    "do_final_boost",
    "display",
)


# --------------------------------------------------------------------------
# Stage timers
# --------------------------------------------------------------------------


def stage_tables(vbmc):
    """Per-iteration values and run totals of the VBMC stage timers.

    ``VBMC.optimize`` calls ``main_timer.reset()`` at the top of every
    iteration and records a deep copy of the timer at the end of it, so each
    snapshot in ``iteration_history["timer"]`` already holds that iteration's
    durations; totals are the sum over snapshots. Work after the loop
    (``determine_best_vp``, ``final_boost``) is not timed by the stage timers
    and shows up only in the wall-clock time and the cProfile attribution.
    """
    hist = vbmc.iteration_history
    timers = list(hist["timer"])
    names = sorted({k for t in timers for k in t._durations})
    rows = []
    totals = {k: 0.0 for k in names}
    for i, t in enumerate(timers):
        row = {"iter": i}
        for k in names:
            row[k] = float(t._durations.get(k, 0.0))
            totals[k] += row[k]
        row["K"] = int(hist["vp"][i].K)
        row["N"] = int(hist["optim_state"][i]["N"])
        row["n_eff"] = int(hist["n_eff"][i])
        row["Ns_gp"] = int(hist["Ns_gp"][i])
        row["func_count"] = int(hist["func_count"][i])
        row["elbo"] = float(hist["elbo"][i])
        row["elbo_sd"] = float(hist["elbo_sd"][i])
        row["warmup"] = bool(hist["warmup"][i])
        rows.append(row)
    return totals, rows, len(timers)


# --------------------------------------------------------------------------
# cProfile attribution
# --------------------------------------------------------------------------

# (label, filename substring, function name). Filenames are normalized to
# forward slashes before matching. Cumulative times are summed over all
# matching code objects.
ATTRIBUTION = [
    # whole run
    ("VBMC.optimize", "vbmc/vbmc.py", "optimize"),
    # active sampling
    ("active_sample", "vbmc/active_sample.py", "active_sample"),
    ("cma.fmin", "cma/", "fmin"),
    ("acquisition __call__", "acquisition_functions/", "__call__"),
    ("GP.predict", "gpyreg/gaussian_process.py", "predict"),
    ("GP.predict_full", "gpyreg/gaussian_process.py", "predict_full"),
    ("GP.update (rank-1)", "gpyreg/gaussian_process.py", "update"),
    ("vp.pdf", "variational_posterior.py", "pdf"),
    (
        "active_importance_sampling",
        "active_importance_sampling.py",
        "active_importance_sampling",
    ),
    # GP training
    ("train_gp", "gaussian_process_train.py", "train_gp"),
    ("GP.fit", "gpyreg/gaussian_process.py", "fit"),
    ("SliceSampler.sample", "gpyreg/slice_sample.py", "sample"),
    (
        "GP.__core_computation",
        "gpyreg/gaussian_process.py",
        "__core_computation",
    ),
    ("f_min_fill", "gpyreg/f_min_fill.py", "f_min_fill"),
    ("scipy.optimize.minimize", "scipy/optimize/_minimize.py", "minimize"),
    ("scipy cholesky", "scipy/linalg/_decomp_cholesky.py", "cholesky"),
    ("solve_triangular", "scipy/linalg/_basic.py", "solve_triangular"),
    # variational optimization
    ("optimize_vp", "variational_optimization.py", "optimize_vp"),
    ("update_K", "variational_optimization.py", "update_K"),
    ("_sieve", "variational_optimization.py", "_sieve"),
    ("_vb_init", "variational_optimization.py", "_vb_init"),
    ("minimize_adam", "vbmc/minimize_adam.py", "minimize_adam"),
    ("_neg_elcbo", "variational_optimization.py", "_neg_elcbo"),
    ("_gp_log_joint", "variational_optimization.py", "_gp_log_joint"),
    ("_eval_full_elcbo", "variational_optimization.py", "_eval_full_elcbo"),
    ("entmc_vbmc", "entropy/entmc_vbmc.py", "entmc_vbmc"),
    ("entlb_vbmc", "entropy/entlb_vbmc.py", "entlb_vbmc"),
    # bookkeeping and finalization
    ("record_iteration", "iteration_history.py", "record_iteration"),
    ("copy.deepcopy", "copy.py", "deepcopy"),
    (
        "_check_termination_conditions",
        "vbmc/vbmc.py",
        "_check_termination_conditions",
    ),
    (
        "determine_best_vp (incl. in-loop warping calls)",
        "vbmc/vbmc.py",
        "determine_best_vp",
    ),
    ("final_boost", "vbmc/vbmc.py", "final_boost"),
    ("vp.kl_div", "variational_posterior.py", "kl_div"),
    ("vp.sample", "variational_posterior.py", "sample"),
    ("vp.moments", "variational_posterior.py", "moments"),
]


def attribute(stats: pstats.Stats):
    """Cumulative time and call count for each ATTRIBUTION entry."""
    out = []
    for label, fsub, fname in ATTRIBUTION:
        ct = 0.0
        ncalls = 0
        hits = []
        for (fn, lineno, name), (
            cc,
            nc,
            tt,
            cum,
            callers,
        ) in stats.stats.items():
            fn_norm = fn.replace("\\", "/")
            if name == fname and fsub in fn_norm:
                ct += cum
                ncalls += nc
                hits.append(f"{fn_norm.split('/')[-1]}:{lineno}")
        out.append(
            {"label": label, "cumtime": ct, "ncalls": ncalls, "where": hits}
        )
    return out


def profile_text(stats: pstats.Stats, n: int = 40):
    buf = io.StringIO()
    stats.stream = buf
    stats.sort_stats("cumulative").print_stats(n)
    stats.sort_stats("tottime").print_stats(n)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Metadata helpers
# --------------------------------------------------------------------------


def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_info():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=REPO_ROOT,
                text=True,
            ).strip()
        )
        return {"sha": sha, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"sha": None, "dirty": None}


def thread_env():
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    return {k: os.environ.get(k) for k in keys}


def jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


def effective_options(vbmc, extra_keys=()):
    keys = list(EFFECTIVE_OPTION_KEYS) + [
        k for k in extra_keys if k not in EFFECTIVE_OPTION_KEYS
    ]
    return {k: jsonable(vbmc.options.get(k)) for k in keys}


def build_meta(problem, cfg_label, args, requested_options, prof):
    return {
        "problem": problem.name,
        "D": problem.D,
        "config": cfg_label,
        "noise_sd": problem.noise_sd,
        "seed": args.seed,
        "requested_options": jsonable(requested_options),
        "cprofile": bool(prof),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "threads": thread_env(),
        "git": git_info(),
        "numpy": np.__version__,
        "scipy": pkg_version("scipy"),
        "pyvbmc": pkg_version("pyvbmc"),
        "gpyreg": pkg_version("gpyreg"),
        "cma": pkg_version("cma"),
        "plb": problem.plb.ravel().tolist(),
        "pub": problem.pub.ravel().tolist(),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--D", type=int, default=None, help="dimension (default 5)"
    )
    ap.add_argument(
        "--problem",
        choices=list(TARGET_NAMES),
        default=None,
        help="target name (default normal)",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="suite config label from benchmark_targets --list; sets problem,"
        " D, noise and options (other flags override)",
    )
    ap.add_argument("--noise-sd", type=float, default=None)
    ap.add_argument(
        "--options",
        default=None,
        help="JSON dict of VBMC options merged last (scalars only)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cprofile", action="store_true")
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="set display='off' (default: 'iter')",
    )
    ap.add_argument(
        "--max-fun-evals",
        type=int,
        default=None,
        help="override max_fun_evals (default: 50*(2+D), x1.5 if noisy)",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tag", default=None)
    return ap.parse_args(argv)


def resolve_problem(args):
    """Turn the CLI into ``(problem, config_label, requested_options)``."""
    name, D, noise_sd, options, label = "normal", 5, None, {}, None
    if args.config:
        cfg = find_config(args.config)
        name, D, noise_sd = cfg.name, cfg.D, cfg.noise_sd
        options = cfg.options_dict()
        label = cfg.label
    if args.problem:
        name = args.problem
    if args.D is not None:
        D = args.D
    if args.noise_sd is not None:
        noise_sd = args.noise_sd
    if args.options:
        options.update(json.loads(args.options))
    if args.quiet:
        options["display"] = "off"
    if args.max_fun_evals is not None:
        options["max_fun_evals"] = args.max_fun_evals
    problem = make_problem(
        name, D, noise_sd=noise_sd, seed=args.seed, options=options
    )
    if label is None:
        label = f"{name}_D{D}" + (f"_noise{noise_sd:g}" if noise_sd else "")
    return problem, label, options


def main(argv=None):
    args = parse_args(argv)
    problem, label, requested = resolve_problem(args)
    vbmc_args, options = problem.vbmc_args()

    tag = args.tag or f"{label}_seed{args.seed}_{int(time.time())}"
    out_dir = args.out / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[profile_run] {label} seed {args.seed} -> {out_dir}", flush=True)

    from pyvbmc import VBMC

    vbmc = VBMC(*vbmc_args, options=options, seed=args.seed)

    prof = cProfile.Profile() if args.cprofile else None
    t0 = time.perf_counter()
    if prof is not None:
        prof.enable()
    vp, results = vbmc.optimize()
    if prof is not None:
        prof.disable()
    wall = time.perf_counter() - t0

    totals, rows, n_iter = stage_tables(vbmc)
    untimed = wall - sum(totals.values())
    met = metrics(problem, vp, results["elbo"])

    summary = {
        "meta": build_meta(problem, label, args, requested, prof),
        "effective_options": effective_options(vbmc, requested.keys()),
        "result": {
            "wall_s": wall,
            "untimed_s": untimed,
            "target_eval_s": float(vbmc.function_logger.total_fun_eval_time),
            "iterations": int(results["iterations"]),
            "recorded_iterations": n_iter,
            "func_count": int(results["func_count"]),
            "final_N": int(vbmc.optim_state["N"]),
            "final_K": int(vp.K),
            "min_Ns_gp": int(min(r["Ns_gp"] for r in rows)),
            "elbo": float(results["elbo"]),
            "elbo_sd": float(results["elbo_sd"]),
            "ln_Z": problem.ln_Z,
            "elbo_err": met["elbo_err"],
            "gskl": met["gskl"],
            "mmtv": met["mmtv"],
            "posterior_mean_rmse": met["rmse"],
            "moment_method": met["moment_method"],
            "message": results["message"],
        },
        "stage_totals_s": totals,
        "per_iteration": rows,
    }

    print("", flush=True)
    print(
        f"[profile_run] wall {wall:.1f} s (untimed {untimed:.1f} s), {n_iter}"
        f" iterations, {results['func_count']} evals, N={summary['result']['final_N']},"
        f" K={vp.K}, min Ns_gp={summary['result']['min_Ns_gp']},"
        f" |elbo-lnZ|={met['elbo_err']:.3f}, gsKL={met['gskl']:.3f},"
        f" MMTV={met['mmtv']:.3f}, RMSE={met['rmse']:.3f}",
        flush=True,
    )
    print(
        "[profile_run] stage totals (s, % of wall; nested, see docstring):",
        flush=True,
    )
    for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"    {k:22s} {v:8.2f}  {100 * v / wall:5.1f}%", flush=True)

    if prof is not None:
        stats = pstats.Stats(prof)
        stats.dump_stats(str(out_dir / "profile.prof"))
        (out_dir / "profile.txt").write_text(
            profile_text(stats), encoding="utf-8"
        )
        attr = attribute(stats)
        summary["attribution"] = attr
        print(
            "[profile_run] cProfile attribution (cumtime s, % of optimize, "
            "ncalls):",
            flush=True,
        )
        opt = next(a["cumtime"] for a in attr if a["label"] == "VBMC.optimize")
        for a in attr:
            pct = 100 * a["cumtime"] / opt if opt else float("nan")
            print(
                f"    {a['label']:48s} {a['cumtime']:8.2f}  {pct:5.1f}%  "
                f"{a['ncalls']:>9d}",
                flush=True,
            )

    (out_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2)
    )
    print(f"[profile_run] done -> {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
