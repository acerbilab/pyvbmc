"""F3: read-only diagnosis of S0/S2 inference trajectories from E5 traces.

Loads the golden traces (``.npz``), sidecars and per-selection records of the
E5 pilot and continuation campaigns for one configuration and compares the
two arms seed by seed: ELBO and reliability-index trajectories, the stopping
rule replayed from the stored series, warmup and warps (applied and undone),
GP hyperparameters, mixture size, where the acquired points fall relative to
the reference posterior, near-repeats, S2's refinement diagnostics and the
scatter of its coarse scores, and the association between concentration and
final accuracy.  Writes one JSON record and prints a compact table.  Makes
no target calls and fits no GP; the traces are read with
``allow_pickle=False`` and never modified.

The trace's ``X_orig`` holds the logger's live rows: warmup points more than
``warmup_keep_threshold`` below the best observation are removed when warmup
ends, so row positions are not evaluation counts.  The initial design is
recovered from ``X_init`` and the first post-warmup row from the selection
records' ``logger_live_rows``.

Example::

    python dev/scripts/noisy_acq_e5_trajectory_diagnostic.py \
        --pilot dev/scripts/runs/noisy_acq_efficiency_20260916/e5_pilot_20260917/campaign \
        --continuation dev/scripts/runs/noisy_acq_efficiency_20260916/e5_continuation_20260917/campaign \
        --label rosenbrock_D2_noise3 --out <record.json>
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

PILOT_SEEDS = (2000, 2001)
CONTINUATION_SEEDS = tuple(range(2002, 2010))
ARMS = ("S0", "S2")
NEAR_REPEAT_SD_UNITS = 0.02
DEEP_TAIL_NATS = 20.0
TOP_OBSERVATIONS_FOR_REFERENCE = 5
# Package defaults for the stopping rule; a sidecar's effective options
# override them (the package raises tol_stable_count to 90 for a target
# with specified noise; the benchmark contract sets only the budget).
STOPPING_DEFAULTS = {
    "tol_sd": 0.1,
    "tol_stable_count": 60,
    "fun_evals_per_iter": 5,
    "tol_stable_excpt_frac": 0.2,
    "tol_improvement": 0.01,
    "elcbo_impro_weight": 3.0,
}


def stopping_options(side):
    effective = side.get("effective_options") or {}
    opts = {}
    for key, default in STOPPING_DEFAULTS.items():
        value = effective.get(key)
        opts[key] = default if value is None else value
    D = int(side["D"])
    tol_skl = effective.get("tol_skl")
    opts["tol_skl"] = 0.01 * np.sqrt(D) if tol_skl is None else tol_skl
    opts["tol_stable_iters"] = int(
        np.ceil(opts["tol_stable_count"] / opts["fun_evals_per_iter"])
    )
    opts["window_needed_below_one"] = (
        opts["tol_stable_iters"]
        - int(
            np.floor(opts["tol_stable_iters"] * opts["tol_stable_excpt_frac"])
        )
        - 1
    )
    opts["improvement_window_iterations"] = int(
        np.ceil(0.5 * opts["tol_stable_iters"])
    )
    opts["warp_distance_iterations"] = opts["tol_stable_iters"] / 3
    return opts


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arm(campaign, label, seed, arm):
    tag = f"{label}_seed{seed}"
    npz = campaign / "arms" / arm / f"{tag}.npz"
    side = campaign / "arms" / arm / f"{tag}.json"
    rec = campaign / "records" / f"{label}__seed{seed}__{arm}.selection.json"
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    return (
        arrays,
        json.loads(side.read_text(encoding="utf-8")),
        json.loads(rec.read_text(encoding="utf-8")),
        {"npz": sha256(npz), "json": sha256(side), "selection": sha256(rec)},
    )


def _finite_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def _iqr(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    q1, q3 = np.percentile(values, [25, 75])
    return float(q3 - q1)


def stopping_terms(a, D, noise_sd, opts):
    """Rebuild the reliability-index terms and the ELCBO improvement.

    VBMC's index is the mean of |dELBO|/tol_sd, ELBO_sd/tol_sd and sKL/tol_skl,
    where tol_sd = min(max(0.1, sqrt(sn / 0.1) * 0.1), 1.0) and sn is the
    estimated observation noise at high-posterior-density points.  These
    targets supply a constant observation noise, and inverting the stored
    index shows the estimate equals it, so ``noise_sd`` reproduces the index
    to a few 1e-5.  The ELCBO improvement is the least-squares slope of
    ELBO - elcbo_impro_weight * ELBO_sd against the evaluation count over the
    last ceil(tol_stable_iters / 2) iterations, as in
    ``VBMC._compute_reliability_index``, which leaves the first two
    iterations undefined.
    """
    iters = a["iter"]
    n = len(iters)
    base = opts["tol_sd"]
    tol_sn = np.sqrt(noise_sd / base) * base
    tol_sd = float(min(max(base, tol_sn), base * 10))
    d_elbo = np.abs(np.diff(a["elbo"], prepend=np.nan))
    window = opts["improvement_window_iterations"]
    improvement = np.full(n, np.nan)
    fc = a["func_count"].astype(float)
    yy_all = a["elbo"] - opts["elcbo_impro_weight"] * a["elbo_sd"]
    for i in range(2, n):
        idx0 = max(0, i - window + 1)
        xx, yy = fc[idx0 : i + 1], yy_all[idx0 : i + 1]
        if len(xx) >= 2 and np.ptp(xx) > 0:
            improvement[i] = np.polyfit(xx, yy, 1)[0]
    terms = {
        "delta_elbo": d_elbo / tol_sd,
        "elbo_sd": a["elbo_sd"] / tol_sd,
        "skl": a["sKL"] / opts["tol_skl"],
    }
    rebuilt = np.mean([terms[k] for k in terms], axis=0)
    finite = np.isfinite(a["r_index"]) & np.isfinite(rebuilt)
    deviation = (
        float(np.max(np.abs(rebuilt[finite] - a["r_index"][finite])))
        if finite.any()
        else None
    )
    return {
        "tol_sd_effective": tol_sd,
        "term_delta_elbo": terms["delta_elbo"],
        "term_elbo_sd": terms["elbo_sd"],
        "term_skl": terms["skl"],
        "rebuilt_index_max_abs_deviation": deviation,
        "elcbo_improvement_per_eval": improvement,
    }


def applied_warps(a):
    """Split flagged rotoscale iterations into applied and undone.

    The trace flags an iteration whenever its logging actions mention a
    rotoscale, which includes a warp that ``warp_undo_check`` reverted.  A
    warp that was kept changes the recorded parameter transformer, so the
    two are told apart by comparing the transformer with the previous
    iteration.
    """
    flagged = np.flatnonzero(a["warped"].astype(bool))
    applied, undone = [], []
    for i in flagged:
        if i == 0:
            applied.append(int(a["iter"][i]))
            continue
        changed = any(
            not np.array_equal(a[key][i], a[key][i - 1])
            for key in ("pt_mu", "pt_delta", "pt_scale", "pt_R")
        )
        (applied if changed else undone).append(int(a["iter"][i]))
    return applied, undone


def stability_conditions(a, terms, opts, applied):
    """Replay the termination rule iteration by iteration from the trace.

    The recorded ``stable`` flag is set when the reliability index is below
    1, the ELCBO improvement per evaluation is below ``tol_improvement`` and
    at least ``window_needed_below_one`` of the previous
    ``tol_stable_iters - 1`` iterations had index below 1 (the entropy switch
    is off for these runs).  Those three conditions are replayed here and
    validated against the recorded flag.  Whether a stable iteration also
    terminates depends on the distance from the last kept warp; that gate is
    evaluated from the applied warps and reported separately.  Also rolls
    the window forward assuming every further iteration keeps the index
    below 1, to count how many more iterations a non-terminated run needed.
    """
    r = a["r_index"]
    n = len(r)
    tol_stable_iters = opts["tol_stable_iters"]
    needed = opts["window_needed_below_one"]
    impro = terms["elcbo_improvement_per_eval"]
    iter_values = list(a["iter"])
    cond = {"index": [], "improvement": [], "window": [], "warp_distance": []}
    window_counts = []
    for i in range(n):
        cond["index"].append(bool(r[i] < 1.0))
        cond["improvement"].append(
            bool(np.isfinite(impro[i]) and impro[i] < opts["tol_improvement"])
        )
        past = r[max(0, i - tol_stable_iters + 1) : i]
        count = int(np.sum(past < 1.0))
        window_counts.append(count)
        cond["window"].append(
            bool(i + 1 >= tol_stable_iters and count >= needed)
        )
        kept = [w for w in applied if w <= iter_values[i]]
        distance = iter_values[i] - (kept[-1] if kept else -np.inf)
        cond["warp_distance"].append(
            bool(distance >= opts["warp_distance_iterations"])
        )
    stable_replay = [
        cond["index"][i] and cond["improvement"][i] and cond["window"][i]
        for i in range(n)
    ]
    terminate_replay = [
        stable_replay[i] and cond["warp_distance"][i] for i in range(n)
    ]
    # Roll forward: further iterations with index below 1 until the window
    # condition holds (improvement and index assumed satisfied).
    extra = 0
    future = list(r)
    while extra < 200:
        i = len(future) - 1
        past = np.asarray(future[max(0, i - tol_stable_iters + 1) : i])
        if i + 1 >= tol_stable_iters and int(np.sum(past < 1.0)) >= needed:
            break
        future.append(0.5)
        extra += 1
    last = slice(max(0, n - 12), n)
    return {
        "last_12_iterations_failing": {
            k: int(sum(not v for v in vals[last])) for k, vals in cond.items()
        },
        "last_12_elcbo_improvement_per_eval": [
            None if not np.isfinite(v) else round(float(v), 4)
            for v in impro[last]
        ],
        "last_12_window_counts_below_one": window_counts[last],
        "window_needed_below_one": needed,
        "final_iteration": {
            "index_below_one": cond["index"][-1],
            "improvement_below_tol": cond["improvement"][-1],
            "window_met": cond["window"][-1],
            "warp_distance_met": cond["warp_distance"][-1],
            "window_count_below_one": window_counts[-1],
            "elcbo_improvement_per_eval": (
                None
                if not np.isfinite(impro[-1])
                else round(float(impro[-1]), 4)
            ),
        },
        "further_iterations_needed_if_index_stays_below_one": int(extra),
        "replay_matches_recorded_stable_flags": bool(
            np.array_equal(
                np.asarray(stable_replay, dtype=bool), a["stable"].astype(bool)
            )
        ),
        "terminate_replay_matches_final_flag": bool(
            terminate_replay[-1] == bool(a["stable"][-1])
        ),
    }


def gp_hyperparameter_summary(a, D, index):
    hyp, hyp_iter = a["gp_hyp"], a["gp_hyp_iter"]
    rows = hyp[hyp_iter == a["iter"][index]]
    if not rows.size:
        return None
    mean = rows.mean(axis=0)
    return {
        "length_scales": [float(np.exp(v)) for v in mean[:D]],
        "output_scale": float(np.exp(mean[D])),
        "noise_sd": float(np.exp(mean[D + 1])),
        "samples": int(rows.shape[0]),
    }


def live_row_masks(a, records, warmup_end_count):
    """Classify live logger rows: initial design, warmup, post-warmup."""
    X, X_init = a["X_orig"], a["X_init"]
    n_live = X.shape[0]
    init_mask = np.zeros(n_live, dtype=bool)
    for row in X_init:
        init_mask |= np.all(X == row, axis=1)
    n0 = int(X_init.shape[0])
    recs = records["records"]
    k = warmup_end_count - n0
    if 0 <= k < len(recs):
        first_post = int(recs[k]["logger_live_rows"])
    else:
        first_post = n_live
    post_mask = np.arange(n_live) >= first_post
    warm_mask = ~post_mask & ~init_mask
    return {
        "initial": init_mask,
        "warmup": warm_mask,
        "post_warmup": post_mask,
        "initial_rows_surviving": int(init_mask.sum()),
        "initial_rows": n0,
        "live_rows": n_live,
        "first_post_warmup_live_row": first_post,
        "trimmed_rows": int(a["func_count"][-1]) - n_live,
    }


def spatial(a, side, masks):
    X_all, y_all = a["X_orig"], a["y_orig"]
    mean = np.asarray(side["true_mean"], dtype=float).ravel()
    cov = np.asarray(side["true_cov"], dtype=float)
    prec = np.linalg.inv(cov)
    sd = np.sqrt(np.diag(cov))
    plb = np.asarray(side["plb"], dtype=float).ravel()
    pub = np.asarray(side["pub"], dtype=float).ravel()
    diff = X_all - mean
    d2 = np.einsum("ij,jk,ik->i", diff, prec, diff)
    D = X_all.shape[1]
    q50, q99, q999 = stats.chi2.ppf([0.5, 0.99, 0.999], D)
    Z_all = X_all / sd
    nn = np.full(X_all.shape[0], np.nan)
    for i in range(1, X_all.shape[0]):
        nn[i] = float(np.min(np.linalg.norm(Z_all[:i] - Z_all[i], axis=1)))
    top = np.sort(y_all)[-TOP_OBSERVATIONS_FOR_REFERENCE:]
    reference_y = float(np.mean(top))
    outside_box = np.any((X_all < plb) | (X_all > pub), axis=1)

    def block(mask):
        if not mask.any():
            return None
        return {
            "points": int(mask.sum()),
            "median_mahalanobis_sq": float(np.median(d2[mask])),
            "fraction_inside_50pct_ellipse": float(np.mean(d2[mask] <= q50)),
            "fraction_beyond_99pct_ellipse": float(np.mean(d2[mask] > q99)),
            "fraction_beyond_99_9pct_ellipse": float(np.mean(d2[mask] > q999)),
            "fraction_outside_plausible_box": float(
                np.mean(outside_box[mask])
            ),
            "max_abs_coordinate": float(np.max(np.abs(X_all[mask]))),
            "median_nearest_earlier_live_point_sd_units": _finite_median(
                nn[mask]
            ),
            "near_repeats": int(np.sum(nn[mask] < NEAR_REPEAT_SD_UNITS)),
            "median_observed_log_density": float(np.median(y_all[mask])),
            "fraction_deep_tail": float(
                np.mean(y_all[mask] < reference_y - DEEP_TAIL_NATS)
            ),
        }

    return {
        "reference_posterior_sd": [float(v) for v in sd],
        "ellipse_note": (
            "Ellipses are the moment-matched Gaussian of the reference "
            "posterior; on this banana-shaped target the 50 and 99 percent "
            "ellipses hold about 0.60 and 0.97 of the true mass, so they "
            "compare the arms against a common region rather than measure "
            "posterior mass."
        ),
        "deep_tail_reference_log_density": reference_y,
        "live_rows": {
            k: v for k, v in masks.items() if not isinstance(v, np.ndarray)
        },
        "all_acquired": block(masks["warmup"] | masks["post_warmup"]),
        "warmup_acquired_live": block(masks["warmup"]),
        "post_warmup_acquired": block(masks["post_warmup"]),
    }


def selection_summary(records):
    recs = records["records"]
    out = {
        "selections": len(recs),
        "selected_source": dict(
            Counter(r.get("selected_source") for r in recs)
        ),
        "repeats": int(sum(bool(r.get("repeat")) for r in recs)),
        "cache_hits": int(sum(r.get("cache_index") is not None for r in recs)),
        "median_search_seconds": _finite_median(
            [r["search_seconds"] for r in recs]
        ),
    }
    if recs and recs[0]["arm"] == "S2":
        local = [r["local_iterations"] for r in recs]
        improvements, rel_improvements, spreads = [], [], []
        offsets, coarse_is_best = [], []
        messages = Counter()
        for r in recs:
            diag = r.get("refinement_diagnostics") or {}
            fallback = diag.get("coarse_fallback_score")
            chosen = r.get("selected_accurate_score")
            scale = diag.get("objective_scale")
            if r.get("selected_source") == "refined" and None not in (
                fallback,
                chosen,
            ):
                improvements.append(fallback - chosen)
                if scale:
                    rel_improvements.append((fallback - chosen) / scale)
            scores = r.get("shortlist_scores") or []
            indices = r.get("shortlist_indices") or []
            if scores:
                spreads.append(max(scores) - min(scores))
                winner = r.get("coarse_winner_index")
                if (
                    winner in indices
                    and r.get("coarse_winner_score") is not None
                ):
                    accurate = scores[indices.index(winner)]
                    offsets.append(r["coarse_winner_score"] - accurate)
                    coarse_is_best.append(
                        indices[int(np.argmin(scores))] == winner
                    )
            for run in diag.get("local_runs") or []:
                messages[run.get("message")] += 1
        spread_median = _finite_median(spreads)
        offset_iqr = _iqr(offsets)
        out.update(
            {
                "fallback_reasons": dict(
                    Counter(str(r.get("fallback_reason")) for r in recs)
                ),
                "refinement_stop_reasons": dict(
                    Counter(str(r.get("refinement_stop_reason")) for r in recs)
                ),
                "local_iterations": {
                    "median": _finite_median(local),
                    "max": int(max(local)) if local else None,
                    "zero_count": int(sum(v == 0 for v in local)),
                },
                "solver_messages": dict(messages),
                "accurate_rows_median": _finite_median(
                    [r["accurate_candidate_rows"] for r in recs]
                ),
                "refined_fraction": float(
                    sum(r.get("selected_source") == "refined" for r in recs)
                    / len(recs)
                ),
                "score_improvement_over_shortlist_best_refined_only": {
                    "count": len(improvements),
                    "median": _finite_median(improvements),
                    "max": float(max(improvements)) if improvements else None,
                    "median_in_objective_scale_units": _finite_median(
                        rel_improvements
                    ),
                },
                "shortlist_score_spread_median": spread_median,
                "coarse_versus_accurate": {
                    "selections": len(offsets),
                    "coarse_minus_accurate_offset_median": _finite_median(
                        offsets
                    ),
                    "coarse_minus_accurate_offset_iqr": offset_iqr,
                    "offset_iqr_over_shortlist_spread": (
                        float(offset_iqr / spread_median)
                        if offset_iqr is not None and spread_median
                        else None
                    ),
                    "coarse_winner_is_accurate_best_fraction": (
                        float(np.mean(coarse_is_best))
                        if coarse_is_best
                        else None
                    ),
                    "chance_fraction": 1.0 / 8.0,
                },
            }
        )
    return out


def arm_diagnostics(a, side, records, D):
    final = side["final"]
    iters = a["iter"]
    n_iter = len(iters)
    warm = a["warmup"].astype(bool)
    warmup_end_index = int(np.argmax(~warm)) if (~warm).any() else None
    warmup_end_count = (
        int(a["func_count"][warmup_end_index])
        if warmup_end_index is not None
        else int(final["func_count"])
    )
    stable = a["stable"].astype(bool)
    stable_indices = np.flatnonzero(stable)
    r = a["r_index"]
    post = np.arange(n_iter) >= (
        warmup_end_index if warmup_end_index is not None else n_iter
    )
    opts = stopping_options(side)
    terms = stopping_terms(a, D, float(side["noise_sd"]), opts)
    applied, undone = applied_warps(a)
    conditions = stability_conditions(a, terms, opts, applied)
    last = slice(max(0, n_iter - 12), n_iter)
    spikes = post & (r >= 1.0)
    quiet = post & (r < 1.0)

    def share(mask):
        if not mask.any():
            return None
        total = (
            terms["term_delta_elbo"][mask]
            + terms["term_elbo_sd"][mask]
            + terms["term_skl"][mask]
        )
        return {
            "iterations": int(mask.sum()),
            "median_skl_share_of_index": _finite_median(
                terms["term_skl"][mask] / total
            ),
            "median_terms": {
                "delta_elbo": _finite_median(terms["term_delta_elbo"][mask]),
                "elbo_sd": _finite_median(terms["term_elbo_sd"][mask]),
                "skl": _finite_median(terms["term_skl"][mask]),
            },
        }

    masks = live_row_masks(a, records, warmup_end_count)
    return {
        "final": {
            k: final[k]
            for k in (
                "elbo",
                "elbo_sd",
                "elbo_err",
                "gskl",
                "mmtv",
                "rmse",
                "iterations",
                "func_count",
                "final_K",
                "n_warps",
                "success_flag",
                "message",
            )
        },
        "warmup": {
            "end_iteration_index": warmup_end_index,
            "end_func_count": warmup_end_count,
            "flagged_rotoscale_iterations": [
                int(v) for v in iters[a["warped"].astype(bool)]
            ],
            "applied_warp_iterations": applied,
            "undone_warp_iterations": undone,
        },
        "stability": {
            "stable_iterations": int(stable.sum()),
            "first_stable_iteration": (
                int(iters[stable_indices[0]]) if stable_indices.size else None
            ),
            "stopping_options": opts,
            "post_warmup_iterations": int(post.sum()),
            "post_warmup_r_index_at_or_above_one": int(spikes.sum()),
            "post_warmup_r_index_median": (
                _finite_median(r[post]) if post.any() else None
            ),
            "post_warmup_spike_iterations": share(spikes),
            "post_warmup_quiet_iterations": share(quiet),
            "last_12_r_index": [round(float(v), 3) for v in r[last]],
            "last_12_terms_median": {
                "delta_elbo": _finite_median(terms["term_delta_elbo"][last]),
                "elbo_sd": _finite_median(terms["term_elbo_sd"][last]),
                "skl": _finite_median(terms["term_skl"][last]),
                "tol_sd_effective": terms["tol_sd_effective"],
            },
            "rebuilt_index_max_abs_deviation": terms[
                "rebuilt_index_max_abs_deviation"
            ],
            "termination_conditions": conditions,
        },
        "mixture": {
            "K_at_warmup_end": (
                int(a["K"][warmup_end_index])
                if warmup_end_index is not None
                else None
            ),
            "K_final_iteration": int(a["K"][-1]),
            "K_max": int(a["K"].max()),
            "pruned_total": int(np.nansum(a["pruned"])),
            "n_eff_final": float(a["n_eff"][-1]),
            "Ns_gp_final": int(a["Ns_gp"][-1]),
        },
        "gp_hyperparameters": {
            "at_warmup_end": (
                gp_hyperparameter_summary(a, D, warmup_end_index)
                if warmup_end_index is not None
                else None
            ),
            "final": gp_hyperparameter_summary(a, D, n_iter - 1),
        },
        "spatial": spatial(a, side, masks),
        "selection": selection_summary(records),
        "elbo_error_trajectory": {
            "func_count": [int(v) for v in a["func_count"]],
            "elbo_err": [
                round(float(abs(v - side["ln_Z"])), 4) for v in a["elbo"]
            ],
            "r_index": [round(float(v), 3) for v in r],
        },
    }


def elbo_error_at(traj, checkpoints):
    fc = np.asarray(traj["func_count"])
    err = np.asarray(traj["elbo_err"])
    out = {}
    for c in checkpoints:
        idx = np.flatnonzero(fc <= c)
        out[str(c)] = float(err[idx[-1]]) if idx.size else None
    return out


def _post(d, key):
    block = d["spatial"]["post_warmup_acquired"] or {}
    return block.get(key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--label", default="rosenbrock_D2_noise3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    seeds = [(s, args.pilot, "pilot") for s in PILOT_SEEDS] + [
        (s, args.continuation, "continuation") for s in CONTINUATION_SEEDS
    ]
    pairs = []
    for seed, campaign, phase in seeds:
        entry = {"seed": seed, "phase": phase, "arms": {}, "inputs_sha256": {}}
        for arm in ARMS:
            a, side, records, hashes = load_arm(
                campaign, args.label, seed, arm
            )
            D = int(side["D"])
            entry["arms"][arm] = arm_diagnostics(a, side, records, D)
            entry["inputs_sha256"][arm] = hashes
            entry["noise_sd"] = side.get("noise_sd")
        checkpoints = (50, 100, 150, 200)
        entry["elbo_error_at_func_count"] = {
            arm: elbo_error_at(
                entry["arms"][arm]["elbo_error_trajectory"], checkpoints
            )
            for arm in ARMS
        }
        pairs.append(entry)

    def agg(path_fn):
        out = {}
        for arm in ARMS:
            vals = [path_fn(p["arms"][arm]) for p in pairs]
            vals = [v for v in vals if v is not None]
            out[arm] = {"median": _finite_median(vals), "values": vals}
        return out

    def seeds_where(key, s2_lower):
        count = 0
        for p in pairs:
            v0, v2 = _post(p["arms"]["S0"], key), _post(p["arms"]["S2"], key)
            if v0 is None or v2 is None:
                continue
            count += (v2 < v0) if s2_lower else (v2 > v0)
        return int(count)

    # Association between post-warmup concentration and final accuracy.
    d2_all, gskl_all, d2_s2, gskl_s2 = [], [], [], []
    for p in pairs:
        for arm in ARMS:
            d = p["arms"][arm]
            m = _post(d, "median_mahalanobis_sq")
            if m is None:
                continue
            d2_all.append(m)
            gskl_all.append(d["final"]["gskl"])
            if arm == "S2":
                d2_s2.append(m)
                gskl_s2.append(d["final"]["gskl"])
    rho_all = stats.spearmanr(d2_all, gskl_all)
    rho_s2 = stats.spearmanr(d2_s2, gskl_s2)
    gskl_pairs = [
        (p["arms"]["S0"]["final"]["gskl"], p["arms"]["S2"]["final"]["gskl"])
        for p in pairs
    ]
    wilcoxon = stats.wilcoxon(
        [b for _, b in gskl_pairs], [a for a, _ in gskl_pairs]
    )
    spikes_total = {
        arm: int(
            sum(
                p["arms"][arm]["stability"][
                    "post_warmup_r_index_at_or_above_one"
                ]
                for p in pairs
            )
        )
        for arm in ARMS
    }
    post_iters_total = {
        arm: int(
            sum(
                p["arms"][arm]["stability"]["post_warmup_iterations"]
                for p in pairs
            )
        )
        for arm in ARMS
    }
    aggregate = {
        "iterations": agg(lambda d: d["final"]["iterations"]),
        "func_count": agg(lambda d: d["final"]["func_count"]),
        "converged": {
            arm: int(
                sum(p["arms"][arm]["final"]["success_flag"] for p in pairs)
            )
            for arm in ARMS
        },
        "warmup_end_func_count": agg(lambda d: d["warmup"]["end_func_count"]),
        "flagged_rotoscales": agg(
            lambda d: len(d["warmup"]["flagged_rotoscale_iterations"])
        ),
        "applied_warps": agg(
            lambda d: len(d["warmup"]["applied_warp_iterations"])
        ),
        "post_warmup_iterations_total": post_iters_total,
        "post_warmup_r_index_at_or_above_one_total": spikes_total,
        "post_warmup_spike_rate": {
            arm: spikes_total[arm] / post_iters_total[arm] for arm in ARMS
        },
        "seeds_with_more_post_warmup_spikes_S2": int(
            sum(
                p["arms"]["S2"]["stability"][
                    "post_warmup_r_index_at_or_above_one"
                ]
                > p["arms"]["S0"]["stability"][
                    "post_warmup_r_index_at_or_above_one"
                ]
                for p in pairs
            )
        ),
        "post_warmup_r_index_median": agg(
            lambda d: d["stability"]["post_warmup_r_index_median"]
        ),
        "last_12_term_delta_elbo": agg(
            lambda d: d["stability"]["last_12_terms_median"]["delta_elbo"]
        ),
        "last_12_term_elbo_sd": agg(
            lambda d: d["stability"]["last_12_terms_median"]["elbo_sd"]
        ),
        "last_12_term_skl": agg(
            lambda d: d["stability"]["last_12_terms_median"]["skl"]
        ),
        "spike_skl_share_median": agg(
            lambda d: (
                d["stability"]["post_warmup_spike_iterations"] or {}
            ).get("median_skl_share_of_index")
        ),
        "rebuilt_index_max_abs_deviation": agg(
            lambda d: d["stability"]["rebuilt_index_max_abs_deviation"]
        ),
        "K_final": agg(lambda d: d["mixture"]["K_final_iteration"]),
        "gp_length_scale_2_final": agg(
            lambda d: (d["gp_hyperparameters"]["final"] or {}).get(
                "length_scales", [None, None]
            )[-1]
        ),
        "gp_output_scale_final": agg(
            lambda d: (d["gp_hyperparameters"]["final"] or {}).get(
                "output_scale"
            )
        ),
        "post_warmup_fraction_inside_50pct": agg(
            lambda d: _post(d, "fraction_inside_50pct_ellipse")
        ),
        "post_warmup_fraction_beyond_99pct": agg(
            lambda d: _post(d, "fraction_beyond_99pct_ellipse")
        ),
        "post_warmup_fraction_outside_plausible_box": agg(
            lambda d: _post(d, "fraction_outside_plausible_box")
        ),
        "post_warmup_max_abs_coordinate": agg(
            lambda d: _post(d, "max_abs_coordinate")
        ),
        "post_warmup_median_mahalanobis_sq": agg(
            lambda d: _post(d, "median_mahalanobis_sq")
        ),
        "post_warmup_median_nn_sd_units": agg(
            lambda d: _post(d, "median_nearest_earlier_live_point_sd_units")
        ),
        "post_warmup_median_observed_log_density": agg(
            lambda d: _post(d, "median_observed_log_density")
        ),
        "post_warmup_fraction_deep_tail": agg(
            lambda d: _post(d, "fraction_deep_tail")
        ),
        "seeds_with_S2_more_central": {
            "fraction_inside_50pct_ellipse": seeds_where(
                "fraction_inside_50pct_ellipse", False
            ),
            "fraction_beyond_99pct_ellipse": seeds_where(
                "fraction_beyond_99pct_ellipse", True
            ),
            "median_nearest_earlier_live_point_sd_units": seeds_where(
                "median_nearest_earlier_live_point_sd_units", True
            ),
            "median_observed_log_density": seeds_where(
                "median_observed_log_density", False
            ),
            "fraction_deep_tail": seeds_where("fraction_deep_tail", True),
            "median_mahalanobis_sq": seeds_where(
                "median_mahalanobis_sq", True
            ),
        },
        "association_post_warmup_mahalanobis_vs_gskl": {
            "spearman_all_20_runs": {
                "rho": float(rho_all.statistic),
                "p": float(rho_all.pvalue),
            },
            "spearman_S2_only": {
                "rho": float(rho_s2.statistic),
                "p": float(rho_s2.pvalue),
            },
            "note": (
                "Direction predicted by the concentration reading; the "
                "acquisition proposes from the current VP, so a narrow VP also "
                "proposes narrowly, and the association does not settle "
                "causation."
            ),
        },
        "paired_gskl_wilcoxon_S2_minus_S0": {
            "statistic": float(wilcoxon.statistic),
            "p": float(wilcoxon.pvalue),
            "seeds_S2_lower": int(sum(b < a for a, b in gskl_pairs)),
        },
        "S2_coarse_winner_is_accurate_best_fraction_pooled": (
            float(
                np.average(
                    [
                        p["arms"]["S2"]["selection"]["coarse_versus_accurate"][
                            "coarse_winner_is_accurate_best_fraction"
                        ]
                        for p in pairs
                    ],
                    weights=[
                        p["arms"]["S2"]["selection"]["coarse_versus_accurate"][
                            "selections"
                        ]
                        for p in pairs
                    ],
                )
            )
        ),
        "S2_offset_iqr_over_shortlist_spread": agg(
            lambda d: (d["selection"].get("coarse_versus_accurate") or {}).get(
                "offset_iqr_over_shortlist_spread"
            )
        )["S2"],
    }
    record = {
        "schema_version": 2,
        "kind": "noisy_acq_e5_f3_trajectory_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "seeds": [s for s, _, _ in seeds],
        "method": (
            "Read-only comparison of the stored golden traces, sidecars and "
            "selection records of the paired S0/S2 fits. The reliability-index "
            "terms use the target's constant observation noise, which "
            "reproduces the stored index to a few 1e-5; the ELCBO improvement "
            "and the stability conditions are replayed from the stored series "
            "and validated against the recorded stable flags. Live logger rows "
            "exclude warmup points trimmed at warmup end; the initial design "
            "comes from X_init and the first post-warmup row from the selection "
            "records. Spatial statistics use the moment-matched Gaussian of the "
            "reference posterior as a common region, not as a mass measure; "
            "observed log densities are noisy target values."
        ),
        "constants": {
            "near_repeat_threshold_sd_units": NEAR_REPEAT_SD_UNITS,
            "deep_tail_nats_below_reference": DEEP_TAIL_NATS,
            "deep_tail_reference": (
                f"mean of the top {TOP_OBSERVATIONS_FOR_REFERENCE} observed values"
            ),
            "stopping_rule_defaults": STOPPING_DEFAULTS,
            "stopping_rule_note": (
                "Each arm's stopping options are read from its sidecar's "
                "effective options and recorded under stability.stopping_options."
            ),
        },
        "replay_matches_recorded_stable_flags": all(
            p["arms"][arm]["stability"]["termination_conditions"][
                "replay_matches_recorded_stable_flags"
            ]
            for p in pairs
            for arm in ARMS
        ),
        "terminate_replay_matches_final_flags": all(
            p["arms"][arm]["stability"]["termination_conditions"][
                "terminate_replay_matches_final_flag"
            ]
            for p in pairs
            for arm in ARMS
        ),
        "aggregate": aggregate,
        "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def fmt(v, nd=2):
        if v is None:
            return "-"
        return f"{v:.{nd}f}" if isinstance(v, float) else str(v)

    print(
        f"{'seed':>4} {'arm':3} {'conv':4} {'it':>3} {'calls':>5} {'live':>4} "
        f"{'warm@':>5} {'kept':>4} {'spk':>5} {'win':>3} {'need':>4} {'more':>4} "
        f"{'in50':>5} {'>99':>5} {'box':>5} {'nnSD':>6} {'yMed':>7} {'deep':>5} "
        f"{'gskl':>6}"
    )
    for p in pairs:
        for arm in ARMS:
            d = p["arms"][arm]
            st, fin = d["stability"], d["final"]
            tc = st["termination_conditions"]
            sp = d["spatial"]["post_warmup_acquired"] or {}
            print(
                f"{p['seed']:>4} {arm:3} {str(fin['success_flag'])[:4]:4} "
                f"{fin['iterations']:>3} {fin['func_count']:>5} "
                f"{d['spatial']['live_rows']['live_rows']:>4} "
                f"{d['warmup']['end_func_count']:>5} "
                f"{len(d['warmup']['applied_warp_iterations']):>4} "
                f"{st['post_warmup_r_index_at_or_above_one']:>2}/"
                f"{st['post_warmup_iterations']:<2} "
                f"{tc['final_iteration']['window_count_below_one']:>3} "
                f"{tc['window_needed_below_one']:>4} "
                f"{tc['further_iterations_needed_if_index_stays_below_one']:>4} "
                f"{fmt(sp.get('fraction_inside_50pct_ellipse')):>5} "
                f"{fmt(sp.get('fraction_beyond_99pct_ellipse')):>5} "
                f"{fmt(sp.get('fraction_outside_plausible_box')):>5} "
                f"{fmt(sp.get('median_nearest_earlier_live_point_sd_units'), 3):>6} "
                f"{fmt(sp.get('median_observed_log_density'), 1):>7} "
                f"{fmt(sp.get('fraction_deep_tail')):>5} "
                f"{fmt(fin['gskl'], 3):>6}"
            )
    print()
    for k, v in aggregate.items():
        if isinstance(v, dict) and "S0" in v and isinstance(v["S0"], dict):
            print(
                f"{k:46s} S0 median {fmt(v['S0']['median'], 3):>8}   "
                f"S2 median {fmt(v['S2']['median'], 3):>8}"
            )
        else:
            print(f"{k:46s} {json.dumps(v)}")
    print(
        f"\nreplay matches stable flags: {record['replay_matches_recorded_stable_flags']}; "
        f"termination replay matches final flags: "
        f"{record['terminate_replay_matches_final_flags']}"
    )
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
