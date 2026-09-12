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

- exact stored-state identity: every non-timer NPZ array (including its
  shape), separated into main-loop and returned-final state, plus the
  semantic final fields in the JSON sidecar.  The report distinguishes
  ``same loop, changed final`` from a main-loop difference.  Historical
  traces which do not store the returned posterior's transformer say that
  this part is not certifiable; they are never credited with equality for
  state they do not contain;
- the agreement horizon: the first iteration at which the ELBO path
  differs (exactly, and beyond 1e-6), and how many leading *live* evaluated
  points are identical. The trace stores only the rows that survive
  warm-up trimming, so the point horizon is a lower bound on the true
  evaluation horizon; the ELBO horizon is the primary measure;
- whether the initial design (every evaluation before the first GP fit,
  drawn from the generator before any numerics run) is identical: exactly,
  when both traces store it (``X_init``, written since commit 9d92c7f of
  2026-09-04; the 2026-09-03 baseline lacks it), else
  by finding the new run's generator-drawn design points among the
  reference's live rows (row 0 is the benchmark's start point ``x0``,
  drawn from a stream spawned off the run seed and identical by
  construction, so it does not count; warm-up trimming can remove the
  whole design on a target like cigar, and then the stored trace cannot
  certify it: reported, not flagged);
- the final metrics side by side, and whether the new run's ΔLML, gsKL and
  MMTV lie inside the baseline population's envelope for that config
  (Tukey far-out fence, ``Q3 + 3 IQR`` over the seeds' sidecars under
  ``--sidecars``, in git; the plain maximum is vacuous where a seed is a
  known failure, e.g. ``student_D4`` seed 19).

The exact verdict excludes only the NPZ ``timer`` array and timing, memory,
and provenance fields in the sidecar.  The toleranced horizons and population
accuracy fences are separate diagnostics.  In particular, a run with the same
loop but a changed returned final is subject to the accuracy fences.

An arithmetic-preserving refactor is expected to *part* from the stored
trajectory at some point: a few-ulp change in an acquisition value flips a
CMA-ES ranking and the search ends elsewhere; a change to the ELBO
arithmetic itself (e.g. ``_gp_log_joint``) parts at iteration 0, and on a
target whose warm-up variational optimization is chaotic (cigar) by far
more than rounding. What must hold is that the initial design is the same
and that a parted run's finals stay inside the envelope (an identical run
is exempt: its own seed may be the population's far outlier). Exit code 1
if any run fails, the initial design differs, a final is not finite, a
parted run's accuracy metric exceeds
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
DEFAULT_BASELINE = DEFAULT_BASELINE / "reference_990_20260912"
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
DISPLAY_FINAL_KEYS = ACCURACY + (
    "func_count",
    "iterations",
    "final_K",
    "wall_s",
)
SEMANTIC_FINAL_KEYS = (
    "elbo",
    "elbo_sd",
    "final_K",
    "best_iter",
    "success_flag",
    "message",
    "iterations",
    "func_count",
    "final_N",
    "min_Ns_gp",
    "n_warps",
)
FINAL_KEYS = tuple(dict.fromkeys(DISPLAY_FINAL_KEYS + SEMANTIC_FINAL_KEYS))
FINAL_ARRAY_KEYS = {
    "final_w",
    "final_mu",
    "final_sigma",
    "final_lambd",
    "post_mean",
    "post_cov",
}
RETURNED_TRANSFORMER_KEYS = {
    "final_pt_mu",
    "final_pt_delta",
    "final_pt_scale",
    "final_pt_R",
}
# ``golden_trace.col`` encodes absent iteration-history scalars as NaN for
# these arrays.  NaNs elsewhere are data, not missing-value sentinels, and
# therefore cannot establish exact equality.
NAN_PERMITTED_ARRAYS = {
    "elbo",
    "elbo_sd",
    "sKL",
    "r_index",
    "Ns_gp",
    "func_count",
    "n_eff",
    "pruned",
    "N",
}
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
        "--calibration-budget",
        type=int,
        default=None,
        help="pin all three chunk budgets; omitted uses historical defaults",
    )
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


def _archive_keys(archive):
    """Return keys from an NPZ archive or a synthetic dict-like archive."""
    return set(getattr(archive, "files", archive.keys()))


def _array_difference(key, ref_value, new_value):
    """Describe an exact shape/value difference, or return ``None``."""
    import numpy as np

    ref_value = np.asarray(ref_value)
    new_value = np.asarray(new_value)
    if ref_value.shape != new_value.shape:
        return f"shape {ref_value.shape} -> {new_value.shape}"
    equal_nan = key in NAN_PERMITTED_ARRAYS
    if not np.array_equal(ref_value, new_value, equal_nan=equal_nan):
        return "value"
    return None


def _is_final_array(key):
    return (
        key in FINAL_ARRAY_KEYS
        or key in RETURNED_TRANSFORMER_KEYS
        or key.startswith("final_pt_")
    )


def _returned_transformer_coverage(ref_keys, new_keys):
    """Describe whether the returned VP transformer can be compared."""
    ref_have = RETURNED_TRANSFORMER_KEYS & ref_keys
    new_have = RETURNED_TRANSFORMER_KEYS & new_keys
    if (
        ref_have == RETURNED_TRANSFORMER_KEYS
        and new_have == RETURNED_TRANSFORMER_KEYS
    ):
        return True, "compared"
    missing = []
    if ref_have != RETURNED_TRANSFORMER_KEYS:
        missing.append("reference")
    if new_have != RETURNED_TRANSFORMER_KEYS:
        missing.append("new trace")
    return (
        False,
        "returned transformer not certifiable (absent from "
        + " and ".join(missing)
        + ")",
    )


def compare_traces(ref, new):
    """Exact stored-state comparison and separate agreement diagnostics."""
    import numpy as np

    out = {}
    ref_keys = _archive_keys(ref)
    new_keys = _archive_keys(new)
    compared_keys = (ref_keys | new_keys) - {"timer"}
    loop_differences = {}
    final_differences = {}
    for key in sorted(compared_keys):
        destination = (
            final_differences if _is_final_array(key) else loop_differences
        )
        if key not in ref_keys:
            destination[key] = "absent from reference"
        elif key not in new_keys:
            destination[key] = "absent from new trace"
        else:
            difference = _array_difference(key, ref[key], new[key])
            if difference is not None:
                destination[key] = difference
    out["loop_differences"] = loop_differences
    out["final_array_differences"] = final_differences
    out["loop_identical"] = not loop_differences
    out["final_arrays_identical"] = not final_differences
    transformer_ok, transformer_coverage = _returned_transformer_coverage(
        ref_keys, new_keys
    )
    out["returned_transformer_certifiable"] = transformer_ok
    out["returned_transformer_coverage"] = transformer_coverage

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
    out["stored_arrays_identical"] = bool(
        out["loop_identical"] and out["final_arrays_identical"]
    )
    # The initial design (every evaluation before the first GP fit) is
    # drawn from the generator before any numerics run, so it must be
    # identical. Traces written since commit 9d92c7f store it as ``X_init``;
    # the 2026-09-03 baseline does not, so the fallback looks for the new
    # run's design points among the reference's live rows. Warm-up trimming
    # removes low-density rows, on some targets (cigar) the whole design:
    # then the stored trace cannot certify the design and the check is
    # reported as such, not flagged. The ELBO of iteration 0 is numerics
    # run on that design (a change to the ELBO arithmetic moves it) and is
    # not part of the certificate.
    if "X_init" in ref and "X_init" in new:
        ok = bool(
            ref["X_init"].shape == new["X_init"].shape
            and np.array_equal(ref["X_init"], new["X_init"])
        )
        out["design"] = (
            "identical" if ok else "DIFFERENT"
        ) + " (X_init in both traces)"
        out["initial_design_ok"] = ok
    elif "X_init" in new:
        # Row 0 is the start point x0, drawn by the benchmark from a stream
        # spawned off the run seed (benchmark_targets.py), so it is
        # identical whatever the code did: only the rows VBMC's generator
        # drew can certify the design.
        drawn = new["X_init"][1:]
        n0 = len(drawn)
        found = sum(
            bool(np.any(np.all(ref["X_orig"] == x, axis=1))) for x in drawn
        )
        if found:
            # One identical generator-drawn design point means the same
            # generator stream, hence the same design.
            out["design"] = (
                f"{found} of {n0} generator-drawn design points live in"
                " the ref"
            )
            out["initial_design_ok"] = True
        else:
            out["design"] = (
                f"not certifiable (none of the {n0} generator-drawn design"
                " points is live in the reference trace)"
            )
            out["initial_design_ok"] = None
    else:
        out["design"] = "not certifiable (no X_init in the new trace)"
        out["initial_design_ok"] = None
    return out


def _semantic_final_differences(ref, new):
    """Compare the final fields whose values describe solver semantics."""
    differences = {}
    for key in SEMANTIC_FINAL_KEYS:
        if key not in ref:
            differences[key] = "absent from reference"
        elif key not in new:
            differences[key] = "absent from new sidecar"
        elif type(ref[key]) is not type(new[key]) or ref[key] != new[key]:
            differences[key] = "value"
    return differences


def _trace_consistency_issues(trace, final):
    """Cross-check duplicated sidecar counts against the NPZ schema."""
    import numpy as np

    issues = {}

    def require_equal(key, actual):
        expected = final.get(key)
        if expected is None or expected != actual:
            issues[key] = f"sidecar {expected!r}, NPZ {actual!r}"

    keys = _archive_keys(trace)
    if "iter" in keys:
        n_iter = len(trace["iter"])
        require_equal("iterations", n_iter)
        if n_iter and not np.array_equal(trace["iter"], np.arange(n_iter)):
            issues["iter"] = "not the stored 0..iterations-1 mapping"
        best_iter = final.get("best_iter")
        if type(best_iter) is not int or best_iter < 0 or best_iter >= n_iter:
            issues[
                "best_iter"
            ] = f"sidecar {best_iter!r}, outside NPZ iteration mapping"
    if "func_count" in keys and len(trace["func_count"]):
        require_equal("func_count", int(trace["func_count"][-1]))
    if "N" in keys and len(trace["N"]):
        require_equal("final_N", int(trace["N"][-1]))
    if "Ns_gp" in keys and len(trace["Ns_gp"]):
        require_equal("min_Ns_gp", int(np.nanmin(trace["Ns_gp"])))
    if "warped" in keys:
        require_equal("n_warps", int(np.nansum(trace["warped"])))

    final_shape_keys = FINAL_ARRAY_KEYS & keys
    if {"final_w", "final_sigma", "final_mu"} <= final_shape_keys:
        final_w = np.asarray(trace["final_w"])
        final_sigma = np.asarray(trace["final_sigma"])
        final_mu = np.asarray(trace["final_mu"])
        final_k = final.get("final_K")
        if (
            final_w.ndim != 1
            or final_sigma.ndim != 1
            or final_mu.ndim != 2
            or final_w.shape[0] != final_k
            or final_sigma.shape[0] != final_k
            or final_mu.shape[1] != final_k
        ):
            issues["final_K"] = (
                f"sidecar {final_k!r}, final_w {final_w.shape}, "
                f"final_sigma {final_sigma.shape}, final_mu {final_mu.shape}"
            )
    return issues


def _final_validity_issues(trace, final):
    """Return non-finite semantic scalars and returned-array fields."""
    import numpy as np

    issues = {}
    for key in ("elbo", "elbo_sd"):
        value = final.get(key)
        try:
            valid = (
                not isinstance(value, bool)
                and np.isscalar(value)
                and bool(np.isreal(value))
                and bool(np.isfinite(value))
            )
        except TypeError:
            valid = False
        if not valid:
            issues[key] = f"non-finite or non-numeric value {value!r}"

    keys = _archive_keys(trace)
    returned_array_keys = (FINAL_ARRAY_KEYS | RETURNED_TRANSFORMER_KEYS) & keys
    returned_array_keys |= {key for key in keys if key.startswith("final_pt_")}
    for key in sorted(returned_array_keys):
        value = np.asarray(trace[key])
        try:
            finite = np.isfinite(value)
        except TypeError:
            issues[key] = "non-numeric returned array"
            continue
        if not np.all(finite):
            issues[key] = "non-finite returned array"
    return issues


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

    ref_npz, new_npz = baseline / f"{tag}.npz", out_dir / f"{tag}.npz"
    baseline_json = baseline / f"{tag}.json"
    population_json = sidecars / f"{tag}.json"
    # Bind semantic finals to the trace selected by --baseline. The tracked
    # --sidecars population supplies accuracy fences, and is only a semantic
    # fallback when the selected baseline has no trace/sidecar pair.
    ref_json = baseline_json if ref_npz.exists() else population_json
    if not ref_npz.exists() and not ref_json.exists():
        ref_json = baseline_json
    if ref_json.exists():
        fin_ref = json.loads(ref_json.read_text())["final"]
        row["final_ref"] = {k: fin_ref.get(k) for k in FINAL_KEYS}
        semantic_differences = _semantic_final_differences(fin_ref, fin_new)
        row["semantic_final_certifiable"] = True
        row["semantic_final_differences"] = semantic_differences
        row["semantic_final_identical"] = not semantic_differences
    else:
        fin_ref = None
        row["semantic_final_certifiable"] = False
        row["semantic_final_differences"] = {}
        row["semantic_final_identical"] = None

    if ref_npz.exists() and new_npz.exists():
        with np.load(ref_npz) as ref, np.load(new_npz) as new:
            row.update(compare_traces(ref, new))
            consistency = {
                "reference": (
                    _trace_consistency_issues(ref, fin_ref)
                    if fin_ref is not None
                    else {}
                ),
                "new": _trace_consistency_issues(new, fin_new),
            }
            row["consistency_issues"] = {
                side: issues for side, issues in consistency.items() if issues
            }
            validity = {
                "reference": (
                    _final_validity_issues(ref, fin_ref)
                    if fin_ref is not None
                    else {}
                ),
                "new": _final_validity_issues(new, fin_new),
            }
            row["final_validity_issues"] = {
                side: issues for side, issues in validity.items() if issues
            }
        row["final_identical"] = bool(
            row["final_arrays_identical"]
            and row["semantic_final_identical"] is True
        )
        row["identical"] = bool(
            row["loop_identical"] and row["final_identical"]
        )
        not_certifiable = []
        if not row["returned_transformer_certifiable"]:
            not_certifiable.append(row["returned_transformer_coverage"])
        if not row["semantic_final_certifiable"]:
            not_certifiable.append("semantic reference sidecar absent")
        row["identity_not_certifiable"] = not_certifiable
    else:
        # Even a finals-only comparison must reject an invalid returned
        # result. Inspect the new arrays when that archive is available and
        # always validate the semantic ELBO scalars from its sidecar.
        if new_npz.exists():
            with np.load(new_npz) as new:
                new_validity = _final_validity_issues(new, fin_new)
        else:
            new_validity = _final_validity_issues({}, fin_new)
        row["final_validity_issues"] = (
            {"new": new_validity} if new_validity else {}
        )

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
        verdict = "identical stored loop and final"
    elif row.get("loop_identical") and "elbo_exact_iter" in row:
        verdict = "same loop, changed final"
        changed = sorted(
            set(row.get("final_array_differences", {}))
            | set(row.get("semantic_final_differences", {}))
        )
        if changed:
            verdict += ": " + ", ".join(changed)
    elif "elbo_exact_iter" in row:
        # Iterations are 0-based: "parted at iteration i" means iterations
        # 0..i-1 are bit-identical and iteration i is the first to differ.
        if row["elbo_exact_iter"] < min(row["n_iter_ref"], row["n_iter_new"]):
            verdict = f"parted at iteration {row['elbo_exact_iter']}"
            if row["elbo_iter"] > row["elbo_exact_iter"]:
                verdict += f" (beyond 1e-6 at {row['elbo_iter']})"
        elif row["n_iter_ref"] != row["n_iter_new"]:
            verdict = "loop state differs: ELBO length changed"
        else:
            verdict = "loop state differs: ELBO path identical"
        verdict += f"; live points identical: {row['X_orig_exact']}"
    else:
        verdict = "finals only"
    if "design" in row:
        # Whenever the traces were compared, identical runs included (an
        # identical live set with a different trimmed design row would
        # otherwise be flagged without saying why).
        verdict += f"; initial design {row['design']}"
        if row["initial_design_ok"] is False:
            verdict = "INITIAL DESIGN DIFFERS; " + verdict
    if outside:
        verdict += "; OUTSIDE envelope: " + ", ".join(outside)
    if row.get("identity_not_certifiable"):
        verdict += "; " + "; ".join(row["identity_not_certifiable"])
    if row.get("consistency_issues"):
        verdict += "; INCONSISTENT sidecar/NPZ counts"
    if row.get("final_validity_issues"):
        verdict += "; NONFINITE final output"
    row["flagged"] = (
        bool(outside)
        or row.get("initial_design_ok", True) is False
        or bool(row.get("consistency_issues"))
        or bool(row.get("final_validity_issues"))
    )
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
    calibration = (
        "historical default budgets"
        if args.calibration_budget is None
        else f"all calibration budgets {args.calibration_budget}"
    )
    lines = [
        f"# Golden replay {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Code `{git['sha']}`{' (dirty)' if git['dirty'] else ''};"
        f" baseline `{args.baseline.name}`; threads {args.threads};"
        f" {calibration}; {minutes:.1f} min.",
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
        " `identical stored loop and final` = exact shape and value equality"
        " for every stored non-timer NPZ array and all 11 semantic final"
        " sidecar fields. `same loop, changed final` separates a returned"
        " posterior/result change from the main loop. A historical trace"
        " without the returned posterior's transformer is explicitly not"
        " certifiable for that state. `parted at iteration i` (0-based) ="
        " iterations 0..i−1 are bit-identical and iteration i is the first"
        " ELBO difference; the JSON report lists every differing loop and"
        " final array. Toleranced horizons are diagnostics only. The"
        " live-point count is a lower bound on the evaluation horizon because"
        " warm-up trimming removes rows. Flags: a run failed,"
        " the initial design differs (exact where both traces store it,"
        " else a generator-drawn design point of the new run found live in"
        " the reference; where none is live, e.g. cigar, the trace cannot"
        " certify the design and no flag is raised),"
        " a sidecar count contradicts its NPZ, a final is not finite, or a"
        " changed loop/final's ΔLML/gsKL/MMTV exceeds the population's"
        " Q3 + 3 IQR fence."
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

    extra_options = {}
    if args.calibration_budget is not None:
        from pyvbmc import CalibrationProfile

        extra_options["performance_calibration"] = CalibrationProfile(
            pdf_chunk_elements=args.calibration_budget,
            entropy_grad_chunk_elements=args.calibration_budget,
            entropy_value_chunk_elements=args.calibration_budget,
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
                r = run_task(label, seed, extra_options, out_dir)
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
            if args.calibration_budget is None:
                args.calibration_budget = saved.get("calibration_budget")
    report, n_flag = render(rows, git, args, minutes)
    (out_dir / "replay.md").write_text(report, encoding="utf-8")
    (out_dir / "replay.json").write_text(
        json.dumps(
            {
                "git": git,
                "threads": args.threads,
                "calibration_budget": args.calibration_budget,
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
