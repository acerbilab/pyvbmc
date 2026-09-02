"""Golden-trace harness: run a benchmark suite over many seeds, store one
compact trace per run, summarize a population, compare two populations.

Developer tooling for Stage 0 of the modernization plan
(``dev/2026-09-02-modernization-discussion.md`` section 10; plan and design
in ``dev/plans/benchmark-suite-and-golden-traces.md``). Targets come from
``benchmark_targets.py``.

Sub-commands::

    python -u dev/scripts/golden_trace.py run --suite golden --seeds 0-9 \
        --workers 6 --out dev/scripts/runs/golden/baseline
    python dev/scripts/golden_trace.py summary dev/scripts/runs/golden/baseline
    python dev/scripts/golden_trace.py compare REF_DIR NEW_DIR
    python dev/scripts/golden_trace.py compare --split dev/scripts/runs/golden/baseline

``run`` executes one VBMC per task in a pool of *spawned processes* (the
``pyvbmc.timer.main_timer`` singleton and the gpyreg/cma global random state
make threads unsafe), BLAS pinned to one thread per worker, longest tasks
first, skipping (config, seed) pairs whose ``.npz`` already exists, so a
sweep can be interrupted and resumed or extended with more seeds. A failing
run writes ``<tag>.error.txt`` and the sweep continues.

Each run produces ``<label>_seed<seed>.npz`` (per-iteration vectors, ragged
blocks with index vectors, final arrays) and ``<label>_seed<seed>.json``
(config, seed, options as requested and as effective, git SHA, versions,
final scalar metrics). ``summary`` and ``compare`` read only the sidecars.

``compare`` runs a two-sample Kolmogorov-Smirnov test per (config, metric)
on the final ``elbo_err``, ``rmse`` and ``gskl``, applies a Holm correction
over the whole family at alpha 0.05, and flags a config whose median
``func_count`` ratio leaves +-5 %. Exit code 1 if anything is flagged.
Per-iteration traces are stored for diagnosis, never compared.
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_RUNS = REPO_ROOT / "dev" / "scripts" / "runs" / "golden"

# Rough solo minutes per run, used only to order tasks longest-first.
EST_MINUTES = {
    "banana_D2_noise1": 24,
    "banana_D2_noise1_mfe150": 12,
    "banana_D6": 15,
    "logreg_D5": 10,
    "cigar_D4": 7.5,
    "lumpy_D4": 7.5,
    "student_D4": 7.5,
    "banana_D2": 4,
    "rosenbrock_D2": 4,
    "corr_D5": 2.5,
    "normal_D5": 2,
    "halfnormal_D2": 2,
}

# Gated metrics (the VBMC papers' three); rmse is recorded but not gated.
METRICS = ("elbo_err", "gskl", "mmtv")
EXTRA_SCALARS = (
    "rmse",
    "func_count",
    "iterations",
    "final_K",
    "wall_s",
    "n_warps",
)


# --------------------------------------------------------------------------
# One run (executed inside a worker process)
# --------------------------------------------------------------------------


def _nan(v):
    return float("nan") if v is None else float(v)


def _tag(label, seed):
    return f"{label}_seed{seed}"


def run_task(label, seed, extra_options, out_dir):
    """Run one (config, seed); write ``.npz`` + ``.json``; return a row."""
    out_dir = Path(out_dir)
    tag = _tag(label, seed)
    t_start = time.time()
    try:
        import psutil
        from benchmark_targets import find_config, metrics
        from profile_run import (
            effective_options,
            git_info,
            jsonable,
            pkg_version,
            thread_env,
        )

        from pyvbmc import VBMC

        cfg = find_config(label)
        prob = cfg.make(seed=seed)
        args, options = prob.vbmc_args()
        options.update(display="off", plot=False, print_iteration_header=False)
        options.update(extra_options or {})
        requested = dict(options)

        vbmc = VBMC(*args, options=options, seed=seed)
        t0 = time.perf_counter()
        vp, results = vbmc.optimize()
        wall = time.perf_counter() - t0

        hist = vbmc.iteration_history
        n_it = len(hist["elbo"])
        it = np.arange(n_it)

        def col(key, cast=_nan):
            return np.array([cast(hist[key][i]) for i in range(n_it)])

        per_iter = {
            "iter": it,
            "elbo": col("elbo"),
            "elbo_sd": col("elbo_sd"),
            "sKL": col("sKL"),
            "r_index": col("r_index"),
            "stable": col("stable", lambda v: float(bool(v))),
            "warmup": col("warmup", lambda v: float(bool(v))),
            "Ns_gp": col("Ns_gp"),
            "func_count": col("func_count"),
            "n_eff": col("n_eff"),
            "pruned": col("pruned"),
            "K": np.array([hist["vp"][i].K for i in range(n_it)], dtype=float),
            "N": np.array(
                [_nan(hist["optim_state"][i].get("N")) for i in range(n_it)]
            ),
            "warped": np.array(
                [
                    float(
                        any(
                            "rotoscale" in str(a)
                            for a in (hist["logging_action"][i] or [])
                        )
                    )
                    for i in range(n_it)
                ]
            ),
        }
        # stage timers over the union of keys, zero-filled
        timers = [hist["timer"][i] for i in range(n_it)]
        timer_keys = sorted({k for t in timers for k in t._durations})
        timer_mat = np.array(
            [
                [float(t._durations.get(k, 0.0)) for k in timer_keys]
                for t in timers
            ]
        ).reshape(n_it, len(timer_keys))
        # transformer state per iteration
        D = prob.D
        pt_mu = np.zeros((n_it, D))
        pt_delta = np.zeros((n_it, D))
        pt_scale = np.ones((n_it, D))
        pt_R = np.zeros((n_it, D, D))
        for i in range(n_it):
            pt = hist["vp"][i].parameter_transformer
            pt_mu[i] = pt.mu
            pt_delta[i] = pt.delta
            if pt.scale is not None:
                pt_scale[i] = pt.scale
            pt_R[i] = np.eye(D) if pt.R_mat is None else pt.R_mat
        # ragged blocks
        gp_blocks, gp_iter = [], []
        for i in range(n_it):
            h = np.atleast_2d(np.asarray(hist["gp_hyp_full"][i], dtype=float))
            gp_blocks.append(h)
            gp_iter.append(np.full(h.shape[0], i))
        gp_hyp = np.vstack(gp_blocks)
        gp_hyp_iter = np.concatenate(gp_iter)
        vp_w, vp_mu, vp_sigma, vp_iter, vp_lambd = [], [], [], [], []
        for i in range(n_it):
            v = hist["vp"][i]
            vp_w.append(np.ravel(v.w))
            vp_mu.append(np.asarray(v.mu).T)  # (K, D)
            vp_sigma.append(np.ravel(v.sigma))
            vp_iter.append(np.full(v.K, i))
            vp_lambd.append(np.ravel(v.lambd))
        # final quantities
        met = metrics(prob, vp, results["elbo"])
        fl = vbmc.function_logger
        live = fl.X_flag
        proc = psutil.Process()
        mi = proc.memory_info()
        peak_mb = getattr(mi, "peak_wset", mi.rss) / 2**20

        arrays = dict(per_iter)
        arrays.update(
            timer=timer_mat,
            pt_mu=pt_mu,
            pt_delta=pt_delta,
            pt_scale=pt_scale,
            pt_R=pt_R,
            gp_hyp=gp_hyp,
            gp_hyp_iter=gp_hyp_iter,
            vp_w=np.concatenate(vp_w),
            vp_mu=np.vstack(vp_mu),
            vp_sigma=np.concatenate(vp_sigma),
            vp_iter=np.concatenate(vp_iter),
            vp_lambd=np.vstack(vp_lambd),
            X_orig=np.asarray(fl.X_orig)[live],
            y_orig=np.ravel(np.asarray(fl.y_orig))[live],
            final_w=np.ravel(vp.w),
            final_mu=np.asarray(vp.mu),
            final_sigma=np.ravel(vp.sigma),
            final_lambd=np.ravel(vp.lambd),
            post_mean=np.asarray(met["post_mean"]),
            post_cov=np.asarray(met["post_cov"]),
        )
        np.savez_compressed(out_dir / f"{tag}.npz", **arrays)

        side = {
            "label": label,
            "seed": seed,
            "problem": prob.name,
            "D": D,
            "noise_sd": prob.noise_sd,
            "requested_options": jsonable(requested),
            "effective_options": effective_options(vbmc, requested.keys()),
            "timer_keys": timer_keys,
            "ln_Z": prob.ln_Z,
            "true_mean": jsonable(prob.true_mean),
            "true_cov": jsonable(prob.true_cov),
            "plb": prob.plb.ravel().tolist(),
            "pub": prob.pub.ravel().tolist(),
            "notes": prob.notes,
            "final": {
                "elbo": float(results["elbo"]),
                "elbo_sd": float(results["elbo_sd"]),
                "best_iter": int(results["best_iter"]),
                "iterations": int(n_it),
                "func_count": int(results["func_count"]),
                "final_K": int(vp.K),
                "final_N": int(vbmc.optim_state["N"]),
                "min_Ns_gp": int(np.nanmin(per_iter["Ns_gp"])),
                "n_warps": int(np.nansum(per_iter["warped"])),
                "success_flag": bool(results["success_flag"]),
                "message": results["message"],
                "wall_s": wall,
                "target_eval_s": float(fl.total_fun_eval_time),
                "elbo_err": met["elbo_err"],
                "gskl": met["gskl"],
                "mmtv": met["mmtv"],
                "rmse": met["rmse"],
                "moment_method": met["moment_method"],
                "peak_rss_mb": peak_mb,
            },
            "meta": {
                "git": git_info(),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "scipy": pkg_version("scipy"),
                "pyvbmc": pkg_version("pyvbmc"),
                "gpyreg": pkg_version("gpyreg"),
                "cma": pkg_version("cma"),
                "threads": thread_env(),
                "started": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(t_start)
                ),
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pid": os.getpid(),
            },
        }
        (out_dir / f"{tag}.json").write_text(json.dumps(side, indent=1))
        return {
            "tag": tag,
            "ok": True,
            "wall_s": wall,
            "peak_rss_mb": peak_mb,
            **{
                k: side["final"][k]
                for k in ("elbo_err", "gskl", "mmtv", "func_count")
            },
        }
    except Exception:  # noqa: BLE001
        (out_dir / f"{tag}.error.txt").write_text(traceback.format_exc())
        return {"tag": tag, "ok": False, "wall_s": time.time() - t_start}


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def parse_seeds(spec):
    seeds = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return sorted(set(seeds))


def cmd_run(args):
    from benchmark_targets import suite_configs

    # Environment for the spawned workers: one BLAS thread each, headless.
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")

    out_dir = args.out or (DEFAULT_RUNS / f"sweep_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfgs = suite_configs(args.suite)
    if args.only:
        wanted = set(args.only.split(","))
        cfgs = [c for c in cfgs if c.label in wanted]
    extra = json.loads(args.options) if args.options else {}
    seeds = parse_seeds(args.seeds)
    tasks = [
        (c.label, s)
        for c in cfgs
        for s in seeds
        if not (out_dir / f"{_tag(c.label, s)}.npz").exists()
    ]
    tasks.sort(key=lambda t: -EST_MINUTES.get(t[0], 5.0))
    n_skip = len(cfgs) * len(seeds) - len(tasks)
    print(
        f"[golden] {len(tasks)} tasks ({n_skip} already done), {args.workers}"
        f" workers -> {out_dir}",
        flush=True,
    )
    if not tasks:
        return 0
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    t0 = time.time()
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = {
            ex.submit(run_task, label, seed, extra, str(out_dir)): (
                label,
                seed,
            )
            for label, seed in tasks
        }
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["ok"]:
                n_ok += 1
                print(
                    f"[golden] {k}/{len(tasks)} ok   {r['tag']:32s}"
                    f" {r['wall_s'] / 60:5.1f} min  peakRSS {r['peak_rss_mb']:5.0f} MB"
                    f"  elbo_err={r['elbo_err']:.3f} gskl={r['gskl']:.3f}"
                    f" mmtv={r['mmtv']:.3f} evals={r['func_count']}"
                    f"  [{(time.time() - t0) / 60:.0f} min elapsed]",
                    flush=True,
                )
            else:
                n_fail += 1
                print(
                    f"[golden] {k}/{len(tasks)} FAIL {r['tag']:32s}"
                    f" after {r['wall_s'] / 60:.1f} min (see .error.txt)",
                    flush=True,
                )
    print(
        f"[golden] done: {n_ok} ok, {n_fail} failed, {(time.time() - t0) / 60:.1f}"
        f" min wall",
        flush=True,
    )
    return 0 if n_fail == 0 else 1


# --------------------------------------------------------------------------
# summary / compare
# --------------------------------------------------------------------------


def load_population(d):
    """{label: {"seeds": [...], metric: array, ...}} from the sidecars."""
    pop = {}
    for p in sorted(Path(d).glob("*_seed*.json")):
        s = json.loads(p.read_text())
        e = pop.setdefault(
            s["label"],
            {"seeds": [], "rows": [], "fails": 0},
        )
        e["seeds"].append(s["seed"])
        e["rows"].append(s["final"])
    for p in Path(d).glob("*.error.txt"):
        label = p.name.rsplit("_seed", 1)[0]
        pop.setdefault(label, {"seeds": [], "rows": [], "fails": 0})[
            "fails"
        ] += 1
    for e in pop.values():
        for m in METRICS + EXTRA_SCALARS:
            e[m] = np.array([r.get(m, np.nan) for r in e["rows"]], dtype=float)
    return pop


def _med_iqr(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "-"
    q = np.percentile(x, [25, 50, 75])
    return f"{q[1]:.3g} [{q[0]:.3g}, {q[2]:.3g}]"


def cmd_summary(args):
    pop = load_population(args.dir)
    lines = [
        f"# Golden population {Path(args.dir).name}",
        "",
        "| config | n | failed | elbo_err | gskl | mmtv | rmse | usable |"
        " evals | iters | K | warps | wall min |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label in sorted(pop):
        e = pop[label]
        usable = (
            np.mean((e["elbo_err"] < 1) & (e["gskl"] < 1))
            if len(e["rows"])
            else np.nan
        )
        lines.append(
            f"| {label} | {len(e['rows'])} | {e['fails']} |"
            f" {_med_iqr(e['elbo_err'])} | {_med_iqr(e['gskl'])} |"
            f" {_med_iqr(e['mmtv'])} | {_med_iqr(e['rmse'])} | {usable:.2f} |"
            f" {_med_iqr(e['func_count'])} | {_med_iqr(e['iterations'])} |"
            f" {_med_iqr(e['final_K'])} | {np.nanmean(e['n_warps']):.1f} |"
            f" {_med_iqr(e['wall_s'] / 60)} |"
        )
    text = "\n".join(lines)
    print(text)
    (Path(args.dir) / "summary.md").write_text(text + "\n", encoding="utf-8")
    return 0


def _holm(pvals, alpha):
    """Holm step-down: returns a boolean array of rejections."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def compare_populations(ref, new, alpha=0.05, ratio_tol=0.05):
    from scipy import stats

    tests = []  # (label, metric, n_ref, n_new, ks, p, median shift)
    ratios = {}
    only_ref = sorted(set(ref) - set(new))
    only_new = sorted(set(new) - set(ref))
    for label in sorted(set(ref) & set(new)):
        a, b = ref[label], new[label]
        for m in METRICS:
            x = a[m][np.isfinite(a[m])]
            y = b[m][np.isfinite(b[m])]
            if len(x) >= 3 and len(y) >= 3:
                ks = stats.ks_2samp(x, y)
                tests.append(
                    (
                        label,
                        m,
                        len(x),
                        len(y),
                        ks.statistic,
                        ks.pvalue,
                        float(np.median(y) - np.median(x)),
                    )
                )
        fx, fy = a["func_count"], b["func_count"]
        if np.isfinite(fx).sum() and np.isfinite(fy).sum():
            ratios[label] = float(np.nanmedian(fy) / np.nanmedian(fx))
    reject = (
        _holm([t[5] for t in tests], alpha) if tests else np.array([], bool)
    )
    flagged = set()
    lines = [
        f"| config | metric | n ref | n new | KS | p | Holm | median shift |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t, rj in zip(tests, reject):
        if rj:
            flagged.add(t[0])
        lines.append(
            f"| {t[0]} | {t[1]} | {t[2]} | {t[3]} | {t[4]:.3f} | {t[5]:.3g} |"
            f" {'REJECT' if rj else 'ok'} | {t[6]:+.3g} |"
        )
    lines += [
        "",
        "| config | median func_count ratio (new/ref) | flag |",
        "|---|---|---|",
    ]
    for label, r in sorted(ratios.items()):
        bad = abs(r - 1) > ratio_tol
        if bad:
            flagged.add(label)
        lines.append(f"| {label} | {r:.3f} | {'FLAG' if bad else 'ok'} |")
    if only_ref or only_new:
        lines += [
            "",
            f"Only in reference: {only_ref}; only in new: {only_new}.",
        ]
    verdict = (
        f"{len(flagged)} config(s) flagged: {sorted(flagged)}"
        if flagged
        else f"no config flagged ({len(tests)} KS tests, Holm alpha {alpha})"
    )
    lines += ["", f"**{verdict}**"]
    return "\n".join(lines), flagged


def cmd_compare(args):
    if args.split:
        pop = load_population(args.ref)
        ref, new = {}, {}
        for label, e in pop.items():
            seeds = np.array(e["seeds"])
            for tgt, mask in ((ref, seeds % 2 == 0), (new, seeds % 2 == 1)):
                sub = {"seeds": list(seeds[mask]), "rows": [], "fails": 0}
                for m in e:
                    if isinstance(e[m], np.ndarray):
                        sub[m] = e[m][mask]
                tgt[label] = sub
        title = f"# Null check (even vs odd seeds) on {Path(args.ref).name}"
    else:
        ref, new = load_population(args.ref), load_population(args.new)
        title = f"# Compare {Path(args.ref).name} (ref) vs {Path(args.new).name} (new)"
    text, flagged = compare_populations(ref, new, alpha=args.alpha)
    print(title + "\n\n" + text)
    return 1 if flagged else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--suite", default="golden")
    r.add_argument("--seeds", default="0-9")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--only", default=None, help="comma-separated labels")
    r.add_argument(
        "--options", default=None, help="JSON merged into every run"
    )
    s = sub.add_parser("summary")
    s.add_argument("dir")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("new", nargs="?", default=None)
    c.add_argument(
        "--split", action="store_true", help="even vs odd seeds of REF"
    )
    c.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "summary":
        return cmd_summary(args)
    if args.cmd == "compare":
        if not args.split and args.new is None:
            ap.error("compare needs NEW_DIR or --split")
        return cmd_compare(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
