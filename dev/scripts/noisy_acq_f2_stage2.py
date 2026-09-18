"""F2 Stage 2: paired inference comparison on the production suite.

Two arms run through ``golden_trace.run_task`` on the ``production`` suite
of ``benchmark_targets.py``, so every run is a production-budget run at the
package's defaults for a specified-noise target: the baseline with no
extra options, and the treatment with the quasi-Monte Carlo importance
nodes switched on (``active_importance_sampling_qmc`` at 96 nodes). The
baseline arm's traces are production-reference runs. Six configurations
on the production labels, twenty seeds on high-noise Rosenbrock and
logistic regression and ten elsewhere, the same seeds in both arms; the two
fits of a pair run back to back in alternating order so that machine drift
falls on both arms alike.

    prepare  --manifest M.json --out-baseline DIR --out-treatment DIR
    run      --manifest M.json [--max-fits N] [--max-wall-seconds S]
    summary  --manifest M.json --out REPORT.json

``run`` is resumable: a fit whose sidecar exists is skipped, a fit that
left an error file is never rerun, and every fit runs in a fresh
single-threaded process. ``summary`` computes the pre-registered outcomes:
per configuration the paired log-ratios of gsKL, MMTV and evidence error
with a t interval and a sign test, usability and convergence transitions,
target-call, fit-time and active-sampling-time ratios, the review
triggers, and the post-warmup spatial statistics of the F3 mechanism check
for both arms.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

SEEDS = {
    "rosenbrock_D2_noise1_production": list(range(2, 12)),
    "rosenbrock_D2_noise3_production": list(range(2, 22)),
    "logreg_D5_noise3_production": list(range(2, 22)),
    "student_D8_noise3_production": list(range(2, 12)),
    "timing_D5_noise2.2_production": list(range(2, 12)),
    "multisensory_s1_D6_noise1.3_production": list(range(2, 12)),
}
ARMS = {
    "baseline": {},
    "treatment": {
        "active_importance_sampling_qmc": True,
        "active_importance_sampling_qmc_samples": 96,
    },
}
# The E5 usability thresholds (strict) and the metrics they apply to.
USABLE_LIMITS = {"elbo_err": 1.0, "gskl": 1.0, "mmtv": 0.2}
METRICS = ("gskl", "mmtv", "elbo_err")
# The F3 spatial statistics' constants.
TOP_OBSERVATIONS_FOR_REFERENCE = 5
DEEP_TAIL_NATS = 20.0
# Review triggers: a worse median with at least this many pairs worse.
TRIGGER_WORSE = {10: 7, 20: 14}


def _tag(label: str, seed: int) -> str:
    return f"{label}_seed{seed}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    ).stdout.strip()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _environment() -> dict[str, Any]:
    import gpyreg
    import numpy
    import scipy

    gp_dir = Path(gpyreg.__file__).resolve().parent
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "gpyreg_import": str(gp_dir),
        "gpyreg_commit": _git("rev-parse", "HEAD", cwd=gp_dir.parent),
        "pyvbmc_commit": _git("rev-parse", "HEAD"),
        "pyvbmc_dirty": bool(
            _git("status", "--porcelain", "--untracked-files=no")
        ),
        "thread_environment": {k: os.environ.get(k) for k in THREAD_KEYS},
    }


def prepare(
    manifest_path: Path, out_baseline: Path, out_treatment: Path
) -> dict[str, Any]:
    pairs = []
    index = 0
    for label, seeds in SEEDS.items():
        for seed in seeds:
            order = (
                ["baseline", "treatment"]
                if index % 2 == 0
                else ["treatment", "baseline"]
            )
            pairs.append(
                {"label": label, "seed": seed, "index": index, "order": order}
            )
            index += 1
    manifest = {
        "kind": "f2_stage2_manifest",
        "schema_version": 1,
        "purpose": __doc__.split("\n\n")[1].strip(),
        "launch_ready": False,
        "suite": "production",
        "seeds": SEEDS,
        "arms": ARMS,
        "outputs": {
            "baseline": str(out_baseline.resolve()),
            "treatment": str(out_treatment.resolve()),
        },
        "pairs": pairs,
        "pair_count": len(pairs),
        "fit_count": 2 * len(pairs),
        "usable_limits": USABLE_LIMITS,
        "review_triggers": {
            "worse_median_and_worse_pairs_at_least": TRIGGER_WORSE,
            "every_convergence_or_usability_loss": True,
        },
        "environment": _environment(),
        "created_utc": _now(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "f2_stage2_manifest":
        raise RuntimeError("not an F2 Stage 2 manifest")
    return manifest


def _fit_status(out_dir: Path, tag: str) -> str:
    if (out_dir / f"{tag}.json").exists() and (
        out_dir / f"{tag}.npz"
    ).exists():
        return "done"
    if (out_dir / f"{tag}.error.txt").exists():
        return "failed"
    return "pending"


def _run_fit(
    manifest: dict[str, Any], arm: str, label: str, seed: int, timeout: int
) -> dict[str, Any]:
    out_dir = Path(manifest["outputs"][arm])
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = manifest["arms"][arm]
    # The arm's options travel as a JSON string literal and are decoded in
    # the child: JSON booleans are not Python literals.
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "import golden_trace\n"
        f"extra = json.loads({json.dumps(json.dumps(extra))})\n"
        f"r = golden_trace.run_task({label!r}, {seed}, extra,"
        f" {str(out_dir)!r})\n"
        "print(json.dumps(r))\n"
    )
    env = dict(os.environ)
    for key in THREAD_KEYS:
        env[key] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=ROOT,
        )
        last = (
            proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        )
        result = json.loads(last) if last.startswith("{") else {"ok": False}
        result["returncode"] = proc.returncode
        if not result.get("ok"):
            result["stderr_tail"] = proc.stderr[-2000:]
    except subprocess.TimeoutExpired:
        result = {"ok": False, "timeout": True}
        (out_dir / f"{_tag(label, seed)}.error.txt").write_text(
            f"timed out after {timeout} seconds\n", encoding="utf-8"
        )
    result["seconds"] = time.time() - started
    return result


def run(
    manifest_path: Path,
    max_fits: int | None,
    max_wall_seconds: float | None,
    timeout: int,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    if not manifest.get("launch_ready"):
        raise RuntimeError("Stage 2 manifest is not marked launch_ready")
    if any(os.environ.get(k) != "1" for k in THREAD_KEYS):
        raise RuntimeError("Set every BLAS thread environment variable to 1")
    if manifest["environment"]["pyvbmc_commit"] != _git("rev-parse", "HEAD"):
        raise RuntimeError("the checkout moved since the manifest was frozen")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(
            "tracked files are modified; the runs must record a clean commit"
        )
    started = time.time()
    launched = failed = skipped = 0
    batch = {
        "kind": "f2_stage2_batch",
        "manifest_sha256": _sha256(manifest_path),
        "started_utc": _now(),
        "fits": [],
    }
    stop = False
    for pair in manifest["pairs"]:
        for arm in pair["order"]:
            tag = _tag(pair["label"], pair["seed"])
            status = _fit_status(Path(manifest["outputs"][arm]), tag)
            if status != "pending":
                skipped += 1
                continue
            if max_fits is not None and launched >= max_fits:
                stop = True
                break
            if (
                max_wall_seconds is not None
                and time.time() - started > max_wall_seconds
            ):
                stop = True
                break
            result = _run_fit(
                manifest, arm, pair["label"], pair["seed"], timeout
            )
            launched += 1
            failed += 0 if result.get("ok") else 1
            batch["fits"].append(
                {
                    "arm": arm,
                    "tag": tag,
                    **{k: v for k, v in result.items() if k != "stderr_tail"},
                }
            )
            print(
                f"[stage2] {tag:44s} {arm:9s}"
                f" {'ok  ' if result.get('ok') else 'FAIL'}"
                f" {result['seconds'] / 60:5.1f} min"
                + (
                    f"  gskl={result['gskl']:.3f} mmtv={result['mmtv']:.3f}"
                    f" elbo_err={result['elbo_err']:.3f} evals={result['func_count']}"
                    if result.get("ok")
                    else ""
                )
                + f"  [{(time.time() - started) / 60:.0f} min elapsed]",
                flush=True,
            )
        if stop:
            break
    batch.update(
        finished_utc=_now(),
        launched=launched,
        failed=failed,
        skipped=skipped,
        stopped_early=stop,
    )
    out = Path(manifest["outputs"]["treatment"]).parent / (
        f"f2_stage2_batch_{int(started)}.json"
    )
    out.write_text(
        json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[stage2] batch done: {launched} launched, {failed} failed,"
        f" {skipped} already terminal, {(time.time() - started) / 60:.1f} min",
        flush=True,
    )
    return batch


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def _usable(final: dict[str, Any]) -> bool:
    return all(
        np.isfinite(float(final[k])) and float(final[k]) < limit
        for k, limit in USABLE_LIMITS.items()
    )


def _load_run(out_dir: Path, tag: str) -> dict[str, Any] | None:
    side_path = out_dir / f"{tag}.json"
    npz_path = out_dir / f"{tag}.npz"
    if not side_path.exists() or not npz_path.exists():
        return None
    side = json.loads(side_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as a:
        timer = a["timer"]
        keys = side["timer_keys"]
        active = (
            float(timer[:, keys.index("active_sampling")].sum())
            if "active_sampling" in keys
            else float("nan")
        )
        spatial = _post_warmup_spatial(a, side)
    final = side["final"]
    return {
        "final": final,
        "usable": _usable(final),
        "converged": bool(final["success_flag"]),
        "func_count": int(final["func_count"]),
        "wall_s": float(final["wall_s"]),
        "active_sampling_s": active,
        "started": side["meta"]["started"],
        "git": side["meta"]["git"],
        "spatial": spatial,
        "sha256": {"json": _sha256(side_path), "npz": _sha256(npz_path)},
    }


def _post_warmup_spatial(a, side: dict[str, Any]) -> dict[str, Any] | None:
    """The F3 post-warmup statistics from the trace alone.

    Rows appended after warmup end are never trimmed, so the post-warmup
    evaluations are the last ``func_count[-1] - func_count[last warmup
    iteration]`` live rows of the logger.
    """
    X, y = a["X_orig"], a["y_orig"]
    warm = a["warmup"].astype(bool)
    fc = a["func_count"].astype(int)
    if not warm.any():
        return None
    last_warm = int(np.max(np.flatnonzero(warm)))
    n_post = int(fc[-1] - fc[last_warm])
    if n_post <= 0 or n_post > X.shape[0]:
        return None
    mean = np.asarray(side["true_mean"], dtype=float).ravel()
    cov = np.asarray(side["true_cov"], dtype=float)
    prec = np.linalg.inv(cov)
    sd = np.sqrt(np.diag(cov))
    plb = np.asarray(side["plb"], dtype=float).ravel()
    pub = np.asarray(side["pub"], dtype=float).ravel()
    diff = X - mean
    d2 = np.einsum("ij,jk,ik->i", diff, prec, diff)
    D = X.shape[1]
    q50, q99 = stats.chi2.ppf([0.5, 0.99], D)
    Z = X / sd
    nn = np.full(X.shape[0], np.nan)
    for i in range(1, X.shape[0]):
        nn[i] = float(np.min(np.linalg.norm(Z[:i] - Z[i], axis=1)))
    reference_y = float(np.mean(np.sort(y)[-TOP_OBSERVATIONS_FOR_REFERENCE:]))
    outside = np.any((X < plb) | (X > pub), axis=1)
    mask = np.zeros(X.shape[0], dtype=bool)
    mask[-n_post:] = True
    return {
        "post_warmup_points": n_post,
        "median_mahalanobis_sq": float(np.median(d2[mask])),
        "fraction_inside_50pct_ellipse": float(np.mean(d2[mask] <= q50)),
        "fraction_beyond_99pct_ellipse": float(np.mean(d2[mask] > q99)),
        "fraction_outside_plausible_box": float(np.mean(outside[mask])),
        "median_nearest_earlier_live_point_sd_units": float(
            np.nanmedian(nn[mask])
        ),
        "median_observed_log_density": float(np.median(y[mask])),
        "fraction_deep_tail": float(
            np.mean(y[mask] < reference_y - DEEP_TAIL_NATS)
        ),
    }


def _paired(values: list[float]) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": n,
        "median": float(np.median(v)),
        "mean": float(np.mean(v)),
        "worse": int(np.sum(v > 0)),
        "better": int(np.sum(v < 0)),
        "ties": int(np.sum(v == 0)),
    }
    if n > 1:
        half = float(stats.t.ppf(0.975, n - 1) * stats.sem(v))
        out["t_interval_of_mean"] = [out["mean"] - half, out["mean"] + half]
    trials = out["worse"] + out["better"]
    out["sign_test_p_two_sided"] = (
        float(stats.binomtest(out["worse"], trials, 0.5).pvalue)
        if trials
        else None
    )
    return out


def _transitions(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    both = gained = lost = neither = 0
    losses = []
    for row in rows:
        b, t = row["baseline"][key], row["treatment"][key]
        if b and t:
            both += 1
        elif t and not b:
            gained += 1
        elif b and not t:
            lost += 1
            losses.append(row["seed"])
        else:
            neither += 1
    return {
        "both": both,
        "treatment_gained": gained,
        "treatment_lost": lost,
        "neither": neither,
        "seeds_lost": losses,
        "baseline_count": both + lost,
        "treatment_count": both + gained,
    }


def summarize(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    outs = {arm: Path(p) for arm, p in manifest["outputs"].items()}
    report: dict[str, Any] = {
        "kind": "f2_stage2_summary",
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "created_utc": _now(),
        "configurations": {},
        "pairs_complete": 0,
        "pairs_scheduled": manifest["pair_count"],
        "fits_failed": [],
    }
    for label, seeds in manifest["seeds"].items():
        rows = []
        for seed in seeds:
            tag = _tag(label, seed)
            runs = {arm: _load_run(outs[arm], tag) for arm in outs}
            for arm in outs:
                if (outs[arm] / f"{tag}.error.txt").exists():
                    report["fits_failed"].append({"arm": arm, "tag": tag})
            if all(runs.values()):
                rows.append({"seed": seed, **runs})
        report["pairs_complete"] += len(rows)
        n = len(rows)
        entry: dict[str, Any] = {
            "pairs_complete": n,
            "pairs_scheduled": len(seeds),
        }
        if not n:
            report["configurations"][label] = entry
            continue
        metrics: dict[str, Any] = {}
        triggers = []
        for metric in METRICS:
            diffs = [
                float(np.log(r["treatment"]["final"][metric]))
                - float(np.log(r["baseline"]["final"][metric]))
                for r in rows
            ]
            paired = _paired(diffs)
            paired["per_seed_log_ratio"] = {
                str(r["seed"]): d for r, d in zip(rows, diffs)
            }
            threshold = TRIGGER_WORSE.get(
                len(seeds), int(np.ceil(0.7 * len(seeds)))
            )
            paired["review_trigger"] = bool(
                paired.get("median", 0.0) > 0
                and paired.get("worse", 0) >= threshold
            )
            if paired["review_trigger"]:
                triggers.append(metric)
            metrics[metric] = paired
        ratios = {}
        for key in ("func_count", "wall_s", "active_sampling_s"):
            values = np.array(
                [r["treatment"][key] / r["baseline"][key] for r in rows],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            ratios[key] = {
                "median_paired_ratio": float(np.median(values))
                if values.size
                else None,
                "range": [float(values.min()), float(values.max())]
                if values.size
                else None,
                "geometric_mean": float(np.exp(np.mean(np.log(values))))
                if values.size
                else None,
            }
        spatial = {}
        stat_keys = (
            "median_mahalanobis_sq",
            "fraction_inside_50pct_ellipse",
            "fraction_beyond_99pct_ellipse",
            "fraction_outside_plausible_box",
            "median_nearest_earlier_live_point_sd_units",
            "median_observed_log_density",
            "fraction_deep_tail",
        )
        for key in stat_keys:
            b = [
                r["baseline"]["spatial"][key]
                for r in rows
                if r["baseline"]["spatial"]
            ]
            t = [
                r["treatment"]["spatial"][key]
                for r in rows
                if r["treatment"]["spatial"]
            ]
            more_central = sum(
                1
                for r in rows
                if r["baseline"]["spatial"]
                and r["treatment"]["spatial"]
                and (
                    r["treatment"]["spatial"][key]
                    < r["baseline"]["spatial"][key]
                    if key
                    in (
                        "median_mahalanobis_sq",
                        "fraction_beyond_99pct_ellipse",
                        "fraction_outside_plausible_box",
                        "fraction_deep_tail",
                        "median_nearest_earlier_live_point_sd_units",
                    )
                    else r["treatment"]["spatial"][key]
                    > r["baseline"]["spatial"][key]
                )
            )
            spatial[key] = {
                "baseline_median": float(np.median(b)) if b else None,
                "treatment_median": float(np.median(t)) if t else None,
                "seeds_treatment_more_central": more_central,
            }
        entry.update(
            metrics=metrics,
            review_triggers=triggers,
            usability=_transitions(rows, "usable"),
            convergence=_transitions(rows, "converged"),
            ratios=ratios,
            spatial_post_warmup=spatial,
            per_seed={
                str(r["seed"]): {
                    arm: {
                        **{k: r[arm]["final"][k] for k in METRICS},
                        "usable": r[arm]["usable"],
                        "converged": r[arm]["converged"],
                        "func_count": r[arm]["func_count"],
                        "wall_s": r[arm]["wall_s"],
                        "active_sampling_s": r[arm]["active_sampling_s"],
                        "git": r[arm]["git"],
                        "sha256": r[arm]["sha256"],
                    }
                    for arm in ("baseline", "treatment")
                }
                for r in rows
            },
        )
        report["configurations"][label] = entry
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "| Configuration | Pairs | gsKL median log-ratio (worse/n) | MMTV (worse/n) |"
        " Evidence error (worse/n) | Usable base -> treat | Converged base -> treat |"
        " Calls | Fit time | Active-sampling time | Triggers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, e in report["configurations"].items():
        if not e.get("pairs_complete"):
            lines.append(f"| {label} | 0 | | | | | | | | | |")
            continue

        def cell(metric):
            m = e["metrics"][metric]
            return f"{m['median']:+.3f} ({m['worse']}/{m['n']})"

        u, c, r = e["usability"], e["convergence"], e["ratios"]
        lines.append(
            f"| {label} | {e['pairs_complete']} | {cell('gskl')} | {cell('mmtv')} |"
            f" {cell('elbo_err')} | {u['baseline_count']} -> {u['treatment_count']} |"
            f" {c['baseline_count']} -> {c['treatment_count']} |"
            f" {r['func_count']['median_paired_ratio']:.3f} |"
            f" {r['wall_s']['median_paired_ratio']:.3f} |"
            f" {r['active_sampling_s']['median_paired_ratio']:.3f} |"
            f" {', '.join(e['review_triggers']) or 'none'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-baseline", type=Path, required=True)
    p.add_argument("--out-treatment", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--max-fits", type=int)
    r.add_argument("--max-wall-seconds", type=float)
    r.add_argument("--timeout", type=int, default=5400)
    s = sub.add_parser("summary")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare(
            args.manifest, args.out_baseline, args.out_treatment
        )
        print(
            f"prepared {manifest['fit_count']} fits in {manifest['pair_count']} pairs"
        )
    elif args.command == "run":
        run(args.manifest, args.max_fits, args.max_wall_seconds, args.timeout)
    else:
        report = summarize(args.manifest)
        args.out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(markdown(report))
        print(
            f"\npairs complete {report['pairs_complete']} of {report['pairs_scheduled']};"
            f" failed fits {len(report['fits_failed'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
