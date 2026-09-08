"""Analyze the paired 870-endpoint final-boost campaign offline.

This script only reads JSON sidecars.  It never imports PyVBMC, loads the
retained dill captures, fits a GP, or runs a boost.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binomtest

PRIMARY = ("elbo_err", "gskl", "mmtv")
TRUTH_METRICS = PRIMARY + ("rmse",)
SCORES = ("elbo", "gp_sd", "elcbo_beta5")
ARMS = {
    "penalty_on": ("weight_penalty_0.1", 0.1),
    "penalty_off": ("weight_penalty_0", 0.0),
}
TOLERANCES = (0.0, 0.05, 0.1, 0.2, 0.5)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260910
ALPHA = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        required=True,
        type=Path,
        help="Campaign directory, or its campaign.json manifest",
    )
    parser.add_argument(
        "--golden", required=True, type=Path, help="Golden JSON directory"
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_number(value, where, *, nonnegative=False):
    if isinstance(value, str):
        if value.strip().lower() in {"", "nan", "unknown", "n/a", "na"}:
            raise ValueError(f"{where} is explicitly unavailable ({value!r})")
        raise ValueError(f"{where} is a string, expected a finite number")
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{where} is unavailable, expected a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{where} is non-finite")
    if nonnegative and result < 0:
        raise ValueError(f"{where} must be nonnegative")
    return result


def optional_number(value, where):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "nan",
        "unknown",
        "n/a",
        "na",
    }:
        return None
    return required_number(value, where)


def score_block(raw, where):
    scores = {
        name: required_number(
            raw.get(name), f"{where}.{name}", nonnegative=name == "gp_sd"
        )
        for name in SCORES
    }
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{where}.metrics is missing")
    for metric in PRIMARY:
        scores[metric] = required_number(
            metrics.get(metric), f"{where}.metrics.{metric}"
        )
    scores["rmse"] = optional_number(
        metrics.get("rmse"), f"{where}.metrics.rmse"
    )
    return scores


def golden_block(raw, tag, label, seed):
    if raw.get("label") != label or raw.get("seed") != seed:
        raise ValueError(f"{tag}: golden label/seed does not match manifest")
    final = raw.get("final")
    if not isinstance(final, dict):
        raise ValueError(f"{tag}: golden final block is missing")
    result = {
        metric: required_number(final.get(metric), f"{tag}.golden.{metric}")
        for metric in PRIMARY
    }
    result["rmse"] = optional_number(final.get("rmse"), f"{tag}.golden.rmse")
    return result


def validate_options(report, tag):
    arms = report.get("arms")
    if not isinstance(arms, dict) or set(arms) != {
        "weight_penalty_0.1",
        "weight_penalty_0",
    }:
        raise ValueError(f"{tag}: arm set is not exactly the expected pair")
    for _, (source_name, penalty) in ARMS.items():
        actual = arms[source_name]
        expected = {
            "actual_weight_penalty": penalty,
            "actual_tol_weight": 0,
            "actual_tol_elcbo_boost": None,
        }
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(
                    f"{tag}.{source_name}.{key}={actual.get(key)!r}, "
                    f"expected {value!r}"
                )


def load_cases(campaign_arg, golden_dir):
    manifest_path = (
        campaign_arg
        if campaign_arg.name == "campaign.json"
        else campaign_arg / "campaign.json"
    )
    campaign_dir = manifest_path.parent
    manifest = load_json(manifest_path)
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 870:
        raise ValueError("campaign manifest must contain exactly 870 cases")
    if manifest.get("penalties") != [0.1, 0.0]:
        raise ValueError("manifest penalties must be exactly [0.1, 0.0]")
    config_hash = manifest.get("config_sha256")
    if not isinstance(config_hash, str) or len(config_hash) != 64:
        raise ValueError("manifest config_sha256 is missing or malformed")

    records = []
    tags = set()
    for spec in cases:
        tag, label, seed = spec.get("tag"), spec.get("label"), spec.get("seed")
        if not isinstance(tag, str) or tag in tags:
            raise ValueError(f"duplicate or invalid campaign tag {tag!r}")
        if not isinstance(label, str) or not isinstance(seed, int):
            raise ValueError(f"{tag}: invalid manifest label or seed")
        tags.add(tag)
        report = load_json(campaign_dir / f"{tag}.json")
        case = report.get("case", {})
        expected_case = {
            "tag": tag,
            "label": label,
            "seed": seed,
            "state_source": spec.get("state_source"),
        }
        if any(case.get(k) != v for k, v in expected_case.items()):
            raise ValueError(
                f"{tag}: report case metadata does not match manifest"
            )
        if report.get("completion", {}).get("config_sha256") != config_hash:
            raise ValueError(f"{tag}: completion config hash does not match")
        campaign_capture_hash = report.get("completion", {}).get(
            "capture_sha256"
        )
        if (
            not isinstance(campaign_capture_hash, str)
            or len(campaign_capture_hash) != 64
        ):
            raise ValueError(
                f"{tag}: campaign capture hash is missing or malformed"
            )
        if not isinstance(report.get("paired_entry_hash"), str):
            raise ValueError(f"{tag}: paired entry hash is missing")
        validate_options(report, tag)
        diagnostics = report.get("diagnostic_scores")
        if not isinstance(diagnostics, dict):
            raise ValueError(f"{tag}: diagnostic_scores is missing")
        pre = score_block(diagnostics.get("pre", {}), f"{tag}.pre")
        arms = {
            name: score_block(
                diagnostics.get(source_name, {}), f"{tag}.{source_name}"
            )
            for name, (source_name, _) in ARMS.items()
        }
        timings = {
            name: {
                timer: required_number(
                    report["arms"][source_name].get(timer),
                    f"{tag}.{source_name}.{timer}",
                    nonnegative=True,
                )
                for timer in ("optimizer_s", "final_boost_s")
            }
            for name, (source_name, _) in ARMS.items()
        }
        golden_path = golden_dir / f"{tag}.json"
        golden_hash = file_sha256(golden_path)
        if golden_hash != spec.get("json_sha256"):
            raise ValueError(
                f"{tag}: golden JSON hash does not match manifest"
            )
        golden = golden_block(load_json(golden_path), tag, label, seed)
        records.append(
            {
                "tag": tag,
                "config": label,
                "seed": seed,
                "state_source": spec.get("state_source"),
                "golden_json_sha256": golden_hash,
                "source_capture_sha256": spec.get("capture_sha256"),
                "campaign_capture_sha256": campaign_capture_hash,
                "paired_entry_hash": report["paired_entry_hash"],
                "pre": pre,
                "arms": arms,
                "timings": timings,
                "golden": golden,
            }
        )
    configs = sorted({record["config"] for record in records})
    if len(configs) != 19:
        raise ValueError(f"expected 19 configurations, found {len(configs)}")
    return manifest_path, manifest, records, configs


def quantiles(values):
    array = np.asarray(values, dtype=float)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return float(q1), float(median), float(q3)


def summary(values):
    array = np.asarray(values, dtype=float)
    q1, median, q3 = quantiles(array)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def bootstrap_ci(values, indices, statistic):
    sampled = np.asarray(values, dtype=float)[indices]
    estimates = statistic(sampled, axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def holm_adjust(tests):
    order = sorted(range(len(tests)), key=lambda index: tests[index]["sign_p"])
    running = 0.0
    continue_rejecting = True
    for rank, index in enumerate(order):
        test = tests[index]
        multiplier = len(tests) - rank
        running = max(running, min(1.0, test["sign_p"] * multiplier))
        test["holm_adjusted_p"] = running
        threshold = ALPHA / multiplier
        continue_rejecting = continue_rejecting and test["sign_p"] <= threshold
        test["holm_reject"] = bool(continue_rejecting)


def usable(block):
    return block["elbo_err"] < 1 and block["gskl"] < 1 and block["mmtv"] < 0.2


def rule_decisions(record, arm):
    pre, post = record["pre"], record["arms"][arm]
    delta_elbo = post["elbo"] - pre["elbo"]
    delta_elcbo = post["elcbo_beta5"] - pre["elcbo_beta5"]
    decisions = {"always": True, "pre": False}
    for tol in TOLERANCES:
        suffix = f"{tol:g}"
        decisions[f"joint_tol_{suffix}"] = (
            delta_elbo > -tol and delta_elcbo > -tol
        )
        decisions[f"elbo_only_tol_{suffix}"] = delta_elbo > -tol
        decisions[f"elcbo_only_tol_{suffix}"] = delta_elcbo > -tol
    return decisions


def endpoint_rows(records):
    rows = []
    for record in records:
        for arm in ARMS:
            pre, post = record["pre"], record["arms"][arm]
            row = {
                "tag": record["tag"],
                "config": record["config"],
                "seed": record["seed"],
                "state_source": record["state_source"],
                "arm": arm,
                "paired_entry_hash": record["paired_entry_hash"],
                "golden_json_sha256": record["golden_json_sha256"],
                "source_capture_sha256": record["source_capture_sha256"],
                "campaign_capture_sha256": record["campaign_capture_sha256"],
                "delta_elbo": post["elbo"] - pre["elbo"],
                "delta_gp_sd": post["gp_sd"] - pre["gp_sd"],
                "delta_elcbo_beta5": post["elcbo_beta5"] - pre["elcbo_beta5"],
                "post_usable": usable(post),
                "pre_usable": usable(pre),
                "golden_usable": usable(record["golden"]),
                "optimizer_s": record["timings"][arm]["optimizer_s"],
                "final_boost_s": record["timings"][arm]["final_boost_s"],
            }
            for score in SCORES:
                row[f"pre_{score}"] = pre[score]
                row[f"post_{score}"] = post[score]
            for metric in TRUTH_METRICS:
                row[f"pre_{metric}"] = pre[metric]
                row[f"post_{metric}"] = post[metric]
                row[f"golden_{metric}"] = record["golden"][metric]
            row.update(rule_decisions(record, arm))
            rows.append(row)
    return rows


def paired_rows(records):
    rows = []
    for record in records:
        row = {
            key: record[key]
            for key in ("tag", "config", "seed", "state_source")
        }
        for metric in TRUTH_METRICS:
            on = record["arms"]["penalty_on"][metric]
            off = record["arms"]["penalty_off"][metric]
            row[f"penalty_on_{metric}"] = on
            row[f"penalty_off_{metric}"] = off
            row[f"off_minus_on_{metric}"] = (
                off - on if off is not None and on is not None else None
            )
        rows.append(row)
    return rows


def effect_summary(config, metric, diffs, indices):
    return {
        "config": config,
        "metric": metric,
        **summary(diffs),
        "mean_ci95": bootstrap_ci(diffs, indices, np.mean),
        "median_ci95": bootstrap_ci(diffs, indices, np.median),
        "penalty_on_wins": int(np.sum(diffs > 0)),
        "penalty_off_wins": int(np.sum(diffs < 0)),
        "exact_ties": int(np.sum(diffs == 0)),
    }


def analyze_effects(records, configs, bootstrap_indices):
    by_config = {
        config: [r for r in records if r["config"] == config]
        for config in configs
    }
    tests = []
    for config in configs:
        group = by_config[config]
        indices = bootstrap_indices[config]
        for metric in PRIMARY:
            diffs = np.asarray(
                [
                    r["arms"]["penalty_off"][metric]
                    - r["arms"]["penalty_on"][metric]
                    for r in group
                ]
            )
            nonzero = diffs[diffs != 0]
            test = {
                **effect_summary(config, metric, diffs, indices),
                "sign_n_nonzero": int(nonzero.size),
                "sign_p": (
                    float(
                        binomtest(
                            int(np.sum(nonzero > 0)), nonzero.size
                        ).pvalue
                    )
                    if nonzero.size
                    else 1.0
                ),
            }
            tests.append(test)
    if len(tests) != 57:
        raise AssertionError(f"expected 57 primary tests, built {len(tests)}")
    holm_adjust(tests)

    secondary = []
    for config in configs:
        group = by_config[config]
        diffs = np.asarray(
            [
                r["arms"]["penalty_off"]["rmse"]
                - r["arms"]["penalty_on"]["rmse"]
                for r in group
                if r["arms"]["penalty_off"]["rmse"] is not None
                and r["arms"]["penalty_on"]["rmse"] is not None
            ]
        )
        if diffs.size != len(group):
            raise ValueError(
                f"{config}: complete finite paired RMSE was expected"
            )
        secondary.append(
            effect_summary(config, "rmse", diffs, bootstrap_indices[config])
        )

    aggregate = []
    for metric in TRUTH_METRICS:
        replicate_config_means = []
        config_means = []
        for config in configs:
            diffs = np.asarray(
                [
                    r["arms"]["penalty_off"][metric]
                    - r["arms"]["penalty_on"][metric]
                    for r in by_config[config]
                ]
            )
            config_means.append(float(np.mean(diffs)))
            replicate_config_means.append(
                np.mean(diffs[bootstrap_indices[config]], axis=1)
            )
        replicates = np.mean(np.stack(replicate_config_means), axis=0)
        low, high = np.quantile(replicates, [0.025, 0.975])
        aggregate.append(
            {
                "metric": metric,
                "role": (
                    "primary" if metric in PRIMARY else "secondary_descriptive"
                ),
                "config_count": len(configs),
                "equal_config_weighted_mean": float(np.mean(config_means)),
                "stratified_bootstrap_ci95": [float(low), float(high)],
            }
        )
    return tests, secondary, aggregate


def endpoint_quality(records, configs):
    result = []
    scopes = [(config, "configuration") for config in configs]
    scopes.append(("__whole_panel__", "whole_panel_descriptive"))
    for config, scope in scopes:
        group = (
            records
            if config == "__whole_panel__"
            else [record for record in records if record["config"] == config]
        )
        for endpoint in ("pre", "penalty_on", "penalty_off", "golden"):
            blocks = [
                (
                    record["arms"][endpoint]
                    if endpoint in ARMS
                    else record[endpoint]
                )
                for record in group
            ]
            entry = {
                "config": config,
                "scope": scope,
                "endpoint": endpoint,
                "n": len(blocks),
                "usable_count": sum(usable(block) for block in blocks),
                "metrics": {},
            }
            for metric in TRUTH_METRICS:
                available = [
                    block[metric]
                    for block in blocks
                    if block[metric] is not None
                ]
                entry["metrics"][metric] = {
                    "available": len(available),
                    **summary(available),
                }
            result.append(entry)
    return result


def diagnostic_summaries(records, configs):
    result = []
    for config in configs:
        group = [record for record in records if record["config"] == config]
        for endpoint in ("pre", "penalty_on", "penalty_off"):
            blocks = [
                (
                    record["arms"][endpoint]
                    if endpoint in ARMS
                    else record[endpoint]
                )
                for record in group
            ]
            for score in SCORES:
                result.append(
                    {
                        "config": config,
                        "endpoint": endpoint,
                        "score": score,
                        **summary([block[score] for block in blocks]),
                    }
                )
        for score in SCORES:
            result.append(
                {
                    "config": config,
                    "endpoint": "penalty_off_minus_penalty_on",
                    "score": score,
                    **summary(
                        [
                            record["arms"]["penalty_off"][score]
                            - record["arms"]["penalty_on"][score]
                            for record in group
                        ]
                    ),
                }
            )
    return result


def timing_summaries(records, configs):
    result = []
    scopes = [(config, "configuration") for config in configs]
    scopes.append(("__whole_panel__", "whole_panel_descriptive"))
    for config, scope in scopes:
        group = (
            records
            if config == "__whole_panel__"
            else [record for record in records if record["config"] == config]
        )
        for arm in ARMS:
            for timer in ("optimizer_s", "final_boost_s"):
                values = [record["timings"][arm][timer] for record in group]
                result.append(
                    {
                        "config": config,
                        "scope": scope,
                        "arm": arm,
                        "timer": timer,
                        **summary(values),
                        "total_s": float(np.sum(values)),
                    }
                )
    return result


def acceptance_analysis(records, configs):
    summaries = []
    rules = ["always", "pre"] + [
        f"{kind}_tol_{tol:g}"
        for tol in TOLERANCES
        for kind in ("joint", "elbo_only", "elcbo_only")
    ]
    for config in configs:
        group = [record for record in records if record["config"] == config]
        for arm in ARMS:
            decisions = [rule_decisions(record, arm) for record in group]
            always_blocks = [record["arms"][arm] for record in group]
            for rule in rules:
                accepted = np.asarray(
                    [item[rule] for item in decisions], dtype=bool
                )
                returned = [
                    record["arms"][arm] if keep else record["pre"]
                    for record, keep in zip(group, accepted)
                ]
                entry = {
                    "config": config,
                    "arm": arm,
                    "rule": rule,
                    "n": len(group),
                    "accepted_count": int(np.sum(accepted)),
                    "rejected_count": int(np.sum(~accepted)),
                    "usable_count": sum(usable(block) for block in returned),
                    "always_usable_count": sum(
                        usable(block) for block in always_blocks
                    ),
                    "metrics": {},
                }
                for metric in PRIMARY:
                    values = np.asarray([block[metric] for block in returned])
                    always = np.asarray(
                        [block[metric] for block in always_blocks]
                    )
                    rejected_changes = np.asarray(
                        [
                            record["arms"][arm][metric] - record["pre"][metric]
                            for record, keep in zip(group, accepted)
                            if not keep
                        ]
                    )
                    entry["metrics"][metric] = {
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "always_mean": float(np.mean(always)),
                        "always_median": float(np.median(always)),
                        "mean_minus_always": float(
                            np.mean(values) - np.mean(always)
                        ),
                        "median_minus_always": float(
                            np.median(values) - np.median(always)
                        ),
                        "rejected_candidate_benefit_count": int(
                            np.sum(rejected_changes < 0)
                        ),
                        "rejected_candidate_loss_count": int(
                            np.sum(rejected_changes > 0)
                        ),
                        "rejected_candidate_exact_ties": int(
                            np.sum(rejected_changes == 0)
                        ),
                        "rejected_benefit_total": float(
                            np.sum(-rejected_changes[rejected_changes < 0])
                        ),
                        "rejected_harm_total": float(
                            np.sum(rejected_changes[rejected_changes > 0])
                        ),
                    }
                summaries.append(entry)
    return summaries


def guarded_penalty_comparisons(records, configs):
    """Compare returned OFF and ON results under the same joint guard."""
    result = []
    for config in configs:
        group = [record for record in records if record["config"] == config]
        for tol in (0.1, 0.2):
            rule = f"joint_tol_{tol:g}"
            returned = {arm: [] for arm in ARMS}
            for record in group:
                for arm in ARMS:
                    keep = rule_decisions(record, arm)[rule]
                    returned[arm].append(
                        record["arms"][arm] if keep else record["pre"]
                    )
            entry = {
                "config": config,
                "rule": rule,
                "n": len(group),
                "penalty_on_usable_count": sum(
                    usable(block) for block in returned["penalty_on"]
                ),
                "penalty_off_usable_count": sum(
                    usable(block) for block in returned["penalty_off"]
                ),
                "metrics": {},
            }
            for metric in PRIMARY:
                diffs = np.asarray(
                    [
                        off[metric] - on[metric]
                        for off, on in zip(
                            returned["penalty_off"], returned["penalty_on"]
                        )
                    ]
                )
                entry["metrics"][metric] = {
                    **summary(diffs),
                    "penalty_on_wins": int(np.sum(diffs > 0)),
                    "penalty_off_wins": int(np.sum(diffs < 0)),
                    "exact_ties": int(np.sum(diffs == 0)),
                }
            result.append(entry)
    return result


def acceptance_case_rows(records):
    rows = []
    for record in records:
        for arm in ARMS:
            pre, post = record["pre"], record["arms"][arm]
            delta_elbo = post["elbo"] - pre["elbo"]
            delta_sd = post["gp_sd"] - pre["gp_sd"]
            delta_elcbo = post["elcbo_beta5"] - pre["elcbo_beta5"]
            for tol in (0.1, 0.2):
                if delta_elbo > -tol and delta_elcbo > -tol:
                    continue
                row = {
                    "tag": record["tag"],
                    "config": record["config"],
                    "seed": record["seed"],
                    "state_source": record["state_source"],
                    "arm": arm,
                    "joint_tolerance": tol,
                    "delta_elbo": delta_elbo,
                    "delta_gp_sd": delta_sd,
                    "delta_elcbo_beta5": delta_elcbo,
                    "worst_score_change": min(delta_elbo, delta_elcbo),
                    "failed_elbo": delta_elbo <= -tol,
                    "failed_elcbo_beta5": delta_elcbo <= -tol,
                }
                for score in SCORES:
                    row[f"pre_{score}"] = pre[score]
                    row[f"post_{score}"] = post[score]
                for metric in PRIMARY:
                    row[f"pre_{metric}"] = pre[metric]
                    row[f"post_{metric}"] = post[metric]
                    row[f"golden_{metric}"] = record["golden"][metric]
                rows.append(row)
    return rows


def flatten_grouped(
    effect_tests,
    secondary_effects,
    aggregate,
    quality,
    diagnostics,
    timings,
    acceptance,
    guarded_comparisons,
):
    rows = []
    for item in effect_tests:
        rows.append({"section": "paired_effect", **item})
    for item in secondary_effects:
        rows.append({"section": "paired_effect_secondary", **item})
    for item in aggregate:
        rows.append({"section": "aggregate_effect", **item})
    for item in quality:
        base = {k: v for k, v in item.items() if k != "metrics"}
        for metric, stats in item["metrics"].items():
            rows.append(
                {
                    "section": "endpoint_quality",
                    **base,
                    "metric": metric,
                    **stats,
                }
            )
    for item in diagnostics:
        rows.append({"section": "diagnostic_scores", **item})
    for item in timings:
        rows.append({"section": "timing", **item})
    for item in acceptance:
        base = {k: v for k, v in item.items() if k != "metrics"}
        for metric, stats in item["metrics"].items():
            rows.append(
                {"section": "acceptance", **base, "metric": metric, **stats}
            )
    for item in guarded_comparisons:
        base = {k: v for k, v in item.items() if k != "metrics"}
        for metric, stats in item["metrics"].items():
            rows.append(
                {
                    "section": "guarded_penalty_comparison",
                    **base,
                    "metric": metric,
                    **stats,
                }
            )
    return rows


def csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), allow_nan=False)
    return value


def write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="raise"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: csv_value(value) for key, value in row.items()}
            )


def write_json(path, value, *, compact=False):
    separators = (",", ":") if compact else None
    path.write_text(
        json.dumps(
            value,
            indent=None if compact else 2,
            separators=separators,
            sort_keys=compact,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    manifest_path, manifest, records, configs = load_cases(
        args.campaign, args.golden
    )
    print(
        f"Validated 870 unique cases in 19 configurations; "
        f"config_sha256={manifest['config_sha256']}"
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = {}
    for config in configs:
        n = sum(record["config"] == config for record in records)
        bootstrap_indices[config] = rng.integers(
            0, n, size=(BOOTSTRAP_SAMPLES, n)
        )

    endpoints = endpoint_rows(records)
    pairs = paired_rows(records)
    effects, secondary_effects, aggregate = analyze_effects(
        records, configs, bootstrap_indices
    )
    quality = endpoint_quality(records, configs)
    diagnostics = diagnostic_summaries(records, configs)
    timings = timing_summaries(records, configs)
    acceptance = acceptance_analysis(records, configs)
    guarded_comparisons = guarded_penalty_comparisons(records, configs)
    rejected_cases = acceptance_case_rows(records)
    grouped = {
        "paired_effects": effects,
        "paired_effects_secondary": secondary_effects,
        "equal_config_panel": aggregate,
        "endpoint_quality": quality,
        "diagnostic_scores": diagnostics,
        "timing": timings,
        "acceptance_rules": acceptance,
        "guarded_penalty_comparisons": guarded_comparisons,
    }
    analysis = {
        "schema_version": 1,
        "campaign_manifest": str(manifest_path.resolve()),
        "golden_directory": str(args.golden.resolve()),
        "config_sha256": manifest["config_sha256"],
        "campaign_protocol_version": manifest.get("protocol_version"),
        "campaign_sources": manifest.get("sources", []),
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__)),
        },
        "packages": {"numpy": np.__version__, "scipy": scipy.__version__},
        "campaign_seeds": {
            "optimization": manifest.get("optimization_seed"),
            "diagnostic": manifest.get("diagnostic_seed"),
        },
        "case_count": len(records),
        "statistical_unit": "paired endpoint",
        "statistical_unit_count": len(pairs),
        "endpoint_arm_row_count": len(endpoints),
        "configuration_count": len(configs),
        "configuration_counts": dict(
            sorted(
                (config, sum(r["config"] == config for r in records))
                for config in configs
            )
        ),
        "state_source_counts": dict(
            sorted(
                (source, sum(r["state_source"] == source for r in records))
                for source in {r["state_source"] for r in records}
            )
        ),
        "metric_availability": {
            endpoint: {
                metric: sum(
                    (r["arms"][endpoint] if endpoint in ARMS else r[endpoint])[
                        metric
                    ]
                    is not None
                    for r in records
                )
                for metric in TRUTH_METRICS
            }
            for endpoint in ("pre", "penalty_on", "penalty_off", "golden")
        },
        "methods": {
            "paired_difference": "penalty_off minus penalty_on; lower truth error is better",
            "primary_metrics": list(PRIMARY),
            "secondary_metric": "rmse (descriptive only)",
            "bootstrap": {
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "ci": "percentile 95%; paired within configuration",
            },
            "sign_test": "exact two-sided binomial test of direction probability 0.5 on nonzero paired differences; it does not directly test the mean",
            "multiplicity": "Holm step-down at alpha 0.05 across 57 primary tests",
            "confidence_intervals": "pointwise bootstrap intervals; not multiplicity-adjusted",
            "usable": "elbo_err < 1 and gskl < 1 and mmtv < 0.2",
            "acceptance": "strict score delta > -tolerance; score delta is post minus pre",
            "acceptance_grid_status": "post hoc exploratory threshold grid",
            "aggregate": "equal weight per configuration; stratified paired bootstrap; finite benchmark panel only",
        },
        "interpretation_notes": [
            "A non-significant test does not prove equivalence.",
            "Exact ties are counted exactly; very small roundoff-scale differences are not scientific benefits.",
            "Golden results use historical boosts, while campaign states may be reconstructed and use fresh RNG; golden comparisons are contextual, not causal.",
            "Reported GP SD excludes entropy Monte Carlo uncertainty.",
            "Pre/post component counts differ, limiting common-draw alignment in diagnostic rescoring.",
            "No production penalty, acceptance rule, or tolerance is selected by this analysis.",
        ],
        "outputs": {
            "endpoints": ["endpoints.csv", "endpoints.json"],
            "paired_effects": ["paired_effects.csv", "paired_effects.json"],
            "grouped_stats": ["grouped_stats.csv", "grouped_stats.json"],
            "acceptance_cases": "acceptance_cases.csv",
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "endpoints.csv", endpoints)
    write_json(args.out / "endpoints.json", endpoints, compact=True)
    write_csv(args.out / "paired_effects.csv", pairs)
    write_json(args.out / "paired_effects.json", pairs, compact=True)
    write_csv(
        args.out / "grouped_stats.csv",
        flatten_grouped(
            effects,
            secondary_effects,
            aggregate,
            quality,
            diagnostics,
            timings,
            acceptance,
            guarded_comparisons,
        ),
    )
    write_json(args.out / "grouped_stats.json", grouped)
    write_csv(args.out / "acceptance_cases.csv", rejected_cases)
    write_json(args.out / "analysis.json", analysis)
    print(f"Wrote analysis artifacts to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
