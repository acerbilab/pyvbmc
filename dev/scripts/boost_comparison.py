"""Analyze stored final-boost scores and optionally reconstruct diagnostics.

This script never runs VBMC or variational optimization.  The compact golden
traces contain enough information to compare the stored pre-boost and final
scores and to rebuild their variational posteriors for benchmark diagnostics.
They do not retain exact boost restart state.  ``boost_reconstruction.py``
checks reconstructed inputs for new paired experiments, including the
numerical sensitivity introduced by recovering transformed GP coordinates.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_targets import find_config, metrics

from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior

THRESHOLDS = (0.1, 0.2)
NEAR_WIDTH = 0.025
NEW_NOISY_LABELS = {"rosenbrock_D2_noise3", "student_D8_noise3"}


def _threshold_key(value):
    return f"{value:g}"


def score_record(sidecar, trace, thresholds=THRESHOLDS):
    """Return endpoint score changes for one stored boosted candidate."""
    final = sidecar["final"]
    best_iter = int(final["best_iter"])
    pre_elbo = float(trace["elbo"][best_iter])
    pre_sd = float(trace["elbo_sd"][best_iter])
    final_elbo = float(final["elbo"])
    final_sd = float(final["elbo_sd"])
    valid = bool(
        np.all(np.isfinite([pre_elbo, pre_sd, final_elbo, final_sd]))
        and pre_sd >= 0
        and final_sd >= 0
    )
    if valid:
        delta_elbo = final_elbo - pre_elbo
        delta_sd = final_sd - pre_sd
        endpoint_b0 = delta_elbo
        endpoint_b5 = delta_elbo - 5 * delta_sd
        worst_delta = min(endpoint_b0, endpoint_b5)
        worst_b = 0 if endpoint_b0 <= endpoint_b5 else 5
        rejected = {
            _threshold_key(t): bool(worst_delta <= -t) for t in thresholds
        }
    else:
        delta_elbo = delta_sd = endpoint_b0 = endpoint_b5 = worst_delta = None
        worst_b = None
        rejected = {_threshold_key(t): None for t in thresholds}
    return {
        "tag": Path(sidecar.get("tag", "")).stem or None,
        "label": sidecar["label"],
        "seed": int(sidecar["seed"]),
        "best_iter": best_iter,
        "valid": valid,
        "pre_elbo": pre_elbo,
        "pre_elbo_sd": pre_sd,
        "final_elbo": final_elbo,
        "final_elbo_sd": final_sd,
        "delta_elbo": delta_elbo,
        "delta_sd": delta_sd,
        "endpoint_b0": endpoint_b0,
        "endpoint_b5": endpoint_b5,
        "worst_delta": worst_delta,
        "score_drop": None if worst_delta is None else -worst_delta,
        "worst_b": worst_b,
        "rejected": rejected,
    }


def scan_reference(trace_dir, thresholds=THRESHOLDS, near_width=NEAR_WIDTH):
    """Scan all JSON/NPZ pairs and summarize threshold decisions."""
    trace_dir = Path(trace_dir)
    rows = []
    for side_path in sorted(trace_dir.glob("*.json")):
        trace_path = side_path.with_suffix(".npz")
        if not trace_path.is_file():
            raise FileNotFoundError(f"missing trace for {side_path.name}")
        side = json.loads(side_path.read_text(encoding="utf-8"))
        side["tag"] = side_path.stem
        with np.load(trace_path, allow_pickle=False) as trace:
            row = score_record(side, trace, thresholds)
        row["tag"] = side_path.stem
        rows.append(row)

    reject_ids = {
        _threshold_key(t): [
            row["tag"] for row in rows if row["rejected"][_threshold_key(t)]
        ]
        for t in thresholds
    }
    near_ids = {
        _threshold_key(t): [
            row["tag"]
            for row in rows
            if row["valid"] and abs(row["score_drop"] - t) < near_width
        ]
        for t in thresholds
    }
    by_config = {}
    for label in sorted({row["label"] for row in rows}):
        group = [row for row in rows if row["label"] == label]
        by_config[label] = {
            "runs": len(group),
            "invalid": sum(not row["valid"] for row in group),
            "max_score_drop": max(
                (row["score_drop"] for row in group if row["valid"]),
                default=None,
            ),
            "rejected": {
                _threshold_key(t): sum(
                    row["rejected"][_threshold_key(t)] is True for row in group
                )
                for t in thresholds
            },
        }
    selected = set().union(*reject_ids.values(), *near_ids.values())
    selected.update(
        row["tag"] for row in rows if row["label"] in NEW_NOISY_LABELS
    )
    return {
        "trace_dir": str(trace_dir.resolve()),
        "runs": len(rows),
        "thresholds": list(thresholds),
        "near_width": near_width,
        "reject_ids": reject_ids,
        "near_ids": near_ids,
        "selected_metric_tags": sorted(selected),
        "by_config": by_config,
        "rows": rows,
    }


def _build_transformer(problem, trace, iteration, transform_type):
    pt = ParameterTransformer(
        problem.D,
        problem.lb,
        problem.ub,
        problem.plb,
        problem.pub,
        transform_type=transform_type,
    )
    pt.mu = np.asarray(trace["pt_mu"][iteration]).copy()
    pt.delta = np.asarray(trace["pt_delta"][iteration]).copy()
    pt.scale = np.asarray(trace["pt_scale"][iteration]).copy()
    pt.R_mat = np.asarray(trace["pt_R"][iteration]).copy()
    return pt


def _build_vp(trace, pt, iteration=None):
    if iteration is None:
        w = np.asarray(trace["final_w"])
        mu = np.asarray(trace["final_mu"])
        sigma = np.asarray(trace["final_sigma"])
        lambd = np.asarray(trace["final_lambd"])
    else:
        mask = np.asarray(trace["vp_iter"]) == iteration
        w = np.asarray(trace["vp_w"])[mask]
        mu = np.asarray(trace["vp_mu"])[mask].T
        sigma = np.asarray(trace["vp_sigma"])[mask]
        lambd = np.asarray(trace["vp_lambd"])[iteration]
    vp = VariationalPosterior(
        mu.shape[0],
        w.size,
        x0=np.zeros((1, mu.shape[0])),
        parameter_transformer=pt,
        rng=np.random.default_rng(0),
    )
    vp.w = w.reshape(1, -1).copy()
    vp.eta = np.log(np.maximum(vp.w, np.finfo(float).tiny))
    vp.mu = mu.copy()
    vp.sigma = sigma.reshape(1, -1).copy()
    vp.lambd = lambd.reshape(-1, 1).copy()
    vp.stats = None
    return vp


def _metric_scalars(values):
    return {
        key: values[key]
        for key in ("elbo_err", "gskl", "mmtv", "rmse", "moment_method")
    }


def reconstruct_metrics(trace_dir, tags):
    """Rebuild stored pre/final VPs and evaluate fixed-seed diagnostics."""
    output = {}
    for tag in tags:
        side_path = Path(trace_dir) / f"{tag}.json"
        trace_path = side_path.with_suffix(".npz")
        if not side_path.is_file() or not trace_path.is_file():
            raise FileNotFoundError(f"missing JSON/NPZ pair for {tag}")
        side = json.loads(side_path.read_text(encoding="utf-8"))
        final = side["final"]
        iteration = int(final["best_iter"])
        problem = find_config(side["label"]).make(seed=int(side["seed"]))
        transform_type = side["requested_options"].get(
            "bounded_transform", "probit"
        )
        with np.load(trace_path, allow_pickle=False) as trace:
            pt = _build_transformer(problem, trace, iteration, transform_type)
            before = metrics(
                problem,
                _build_vp(trace, pt, iteration),
                float(trace["elbo"][iteration]),
            )
            after = metrics(
                problem, _build_vp(trace, pt), float(final["elbo"])
            )
            for key in ("elbo_err", "gskl", "mmtv", "rmse"):
                if not np.isclose(
                    after[key],
                    final[key],
                    rtol=1e-9,
                    atol=1e-11,
                    equal_nan=True,
                ):
                    raise RuntimeError(
                        f"{tag}: reconstructed final {key}={after[key]!r} "
                        f"does not match sidecar {final[key]!r}"
                    )
            if after["moment_method"] != final["moment_method"]:
                raise RuntimeError(
                    f"{tag}: reconstructed moment method differs"
                )
            for key, actual in (
                ("post_mean", after["post_mean"]),
                ("post_cov", after["post_cov"]),
            ):
                if not np.allclose(
                    actual, trace[key], rtol=1e-9, atol=1e-11, equal_nan=True
                ):
                    raise RuntimeError(
                        f"{tag}: reconstructed final {key} differs"
                    )
        before_out = _metric_scalars(before)
        after_out = _metric_scalars(after)
        output[tag] = {
            "transformer": "best_iter assumption; absent as final trace state",
            "final_sidecar_validation": "passed rtol=1e-9 atol=1e-11",
            "before": before_out,
            "after": after_out,
            "delta": {
                key: after_out[key] - before_out[key]
                for key in ("elbo_err", "gskl", "mmtv", "rmse")
            },
        }
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--metrics-tags",
        nargs="+",
        metavar="TAG",
        help="evaluate listed tags, or 'selected' for rejects/near/new-noisy",
    )
    args = parser.parse_args(argv)
    report = scan_reference(args.trace_dir)
    if args.metrics_tags:
        tags = (
            report["selected_metric_tags"]
            if args.metrics_tags == ["selected"]
            else args.metrics_tags
        )
        report["metrics"] = reconstruct_metrics(args.trace_dir, tags)
    text = json.dumps(report, indent=2, allow_nan=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
