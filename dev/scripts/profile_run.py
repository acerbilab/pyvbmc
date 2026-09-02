"""Profile one PyVBMC run and report where the wall-clock time goes.

Developer tooling for the modernization work (see
``dev/2026-09-02-modernization-discussion.md``, sections 2 and 14). Runs VBMC
on a cheap synthetic target with known normalizing constant under a fixed
seed and reports:

* the per-stage timers VBMC already records (``pyvbmc.timer.main_timer``,
  snapshotted into ``iteration_history["timer"]`` every iteration): totals
  and a per-iteration table. Note that ``active_sampling`` wraps the whole
  active-sampling stage, which internally starts ``gp_train`` and
  ``variational_fit`` timers for intermediate refits, so stage totals do not
  sum to the wall-clock time;
* with ``--cprofile``, a cProfile of ``VBMC.optimize()``: top functions by
  cumulative and internal time, plus cumulative time attributed to a curated
  list of hot-path functions so the numbers line up with the devlog tables.

Output goes to ``dev/scripts/runs/<tag>/`` (gitignored):
``summary.json`` (metadata, stage totals, per-iteration table, attribution),
``profile.prof`` (open with ``snakeviz`` or ``pstats``) and ``profile.txt``.

Examples::

    python -u dev/scripts/profile_run.py --D 5 --seed 0 --cprofile
    python -u dev/scripts/profile_run.py --D 10 --problem corr --cprofile

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
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "dev" / "scripts" / "runs"


# --------------------------------------------------------------------------
# Synthetic targets (log joint, known ln Z, known posterior mean)
# --------------------------------------------------------------------------


def make_problem(name: str, D: int):
    """Return (f, x0, lb, ub, plb, pub, ln_Z, mu_bar) for a named target."""
    if name == "normal":
        # Same target as test_vbmc_multivariate_normal: independent Gaussian
        # with standard deviations 1..D, normalized (ln Z = 0).
        scales = np.arange(1, D + 1, dtype=float)
        log_norm = -np.sum(np.log(scales)) - 0.5 * D * np.log(2 * np.pi)

        def f(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            return float(np.sum(-0.5 * (x / scales) ** 2) + log_norm)

        x0 = -np.ones((1, D))
        plb = np.full((1, D), -2.0 * D)
        pub = np.full((1, D), 2.0 * D)
        lb = np.full((1, D), -np.inf)
        ub = np.full((1, D), np.inf)
        return f, x0, lb, ub, plb, pub, 0.0, np.zeros((1, D))

    if name == "corr":
        # Correlated Gaussian: fixed random rotation (independent of the run
        # seed) times a diagonal of standard deviations in [0.2, 1].
        rng = np.random.default_rng(12345)
        Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        scales = np.linspace(0.2, 1.0, D)
        cov = Q @ np.diag(scales**2) @ Q.T
        prec = np.linalg.inv(cov)
        mean = np.linspace(-0.5, 0.5, D)
        log_norm = -0.5 * np.linalg.slogdet(cov)[1] - 0.5 * D * np.log(
            2 * np.pi
        )

        def f(x):
            x = np.asarray(x, dtype=float).reshape(-1) - mean
            return float(-0.5 * x @ prec @ x + log_norm)

        x0 = np.zeros((1, D))
        plb = np.full((1, D), -2.5)
        pub = np.full((1, D), 2.5)
        lb = np.full((1, D), -np.inf)
        ub = np.full((1, D), np.inf)
        return f, x0, lb, ub, plb, pub, 0.0, mean.reshape(1, -1)

    raise ValueError(f"unknown problem {name!r}")


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
        row["n_eff"] = int(hist["n_eff"][i])
        row["Ns_gp"] = int(hist["Ns_gp"][i])
        row["func_count"] = int(hist["func_count"][i])
        row["elbo"] = float(hist["elbo"][i])
        row["elbo_sd"] = float(hist["elbo_sd"][i])
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
# Main
# --------------------------------------------------------------------------


def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--D", type=int, default=5)
    ap.add_argument("--problem", choices=["normal", "corr"], default="normal")
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
        help="override max_fun_evals (default: 50*(2+D))",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from pyvbmc import VBMC

    f, x0, lb, ub, plb, pub, ln_Z, mu_bar = make_problem(args.problem, args.D)
    options = {}
    if args.quiet:
        options["display"] = "off"
    if args.max_fun_evals is not None:
        options["max_fun_evals"] = args.max_fun_evals

    tag = args.tag or (
        f"{args.problem}_D{args.D}_seed{args.seed}_{int(time.time())}"
    )
    out_dir = args.out / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[profile_run] writing to {out_dir}", flush=True)

    np.random.seed(args.seed)
    vbmc = VBMC(f, x0, lb, ub, plb, pub, options=options)

    prof = cProfile.Profile() if args.cprofile else None
    t0 = time.perf_counter()
    if prof is not None:
        prof.enable()
    vp, results = vbmc.optimize()
    if prof is not None:
        prof.disable()
    wall = time.perf_counter() - t0

    totals, rows, n_iter = stage_tables(vbmc)
    vmu = vp.moments()
    rmse = float(np.sqrt(np.mean((vmu - mu_bar) ** 2)))
    elbo_err = float(abs(results["elbo"] - ln_Z))

    summary = {
        "meta": {
            "problem": args.problem,
            "D": args.D,
            "seed": args.seed,
            "cprofile": bool(prof),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "scipy": pkg_version("scipy"),
            "pyvbmc": pkg_version("pyvbmc"),
            "gpyreg": pkg_version("gpyreg"),
            "cma": pkg_version("cma"),
        },
        "result": {
            "wall_s": wall,
            "target_eval_s": float(vbmc.function_logger.total_fun_eval_time),
            "iterations": int(results["iterations"]),
            "recorded_iterations": n_iter,
            "func_count": int(results["func_count"]),
            "final_K": int(vp.K),
            "elbo": float(results["elbo"]),
            "elbo_sd": float(results["elbo_sd"]),
            "ln_Z": ln_Z,
            "elbo_err": elbo_err,
            "posterior_mean_rmse": rmse,
            "message": results["message"],
        },
        "stage_totals_s": totals,
        "per_iteration": rows,
    }

    print("", flush=True)
    print(
        f"[profile_run] wall {wall:.1f} s, {n_iter} iterations, "
        f"{results['func_count']} evals, K={vp.K}, "
        f"|elbo-lnZ|={elbo_err:.3f}, RMSE={rmse:.3f}",
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
                f"    {a['label']:30s} {a['cumtime']:8.2f}  {pct:5.1f}%  "
                f"{a['ncalls']:>9d}",
                flush=True,
            )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[profile_run] done -> {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
