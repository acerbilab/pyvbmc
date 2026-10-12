"""Reduce frozen E2 experiment summaries to compact decision evidence.

This developer utility reads completed summary JSON files only.  It preserves
their scheduled denominators, unresolved classifications, missing cells, and
ordinary-MC disagreements.  The output is descriptive evidence and never an
automatic promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
E2_REPLICATES = 8


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state_dimensions(state_id: str) -> dict[str, str]:
    trajectory, checkpoint = state_id.rsplit("_", 1)
    target, seed = trajectory.rsplit("_seed", 1)
    return {
        "state_id": state_id,
        "target": target,
        "trajectory": trajectory,
        "seed": seed,
        "checkpoint": checkpoint,
    }


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _worst_record(item: dict[str, Any]) -> dict[str, Any]:
    cell = item["cell"]
    dimensions = _state_dimensions(cell["state_id"])
    raw_applicable = not item["comparison_penalty_active"]
    return {
        **dimensions,
        "replicate": int(cell["replicate"]),
        "mean_F_difference": float(item["judge"]["mean_difference"]),
        "F_ci": item["judge"].get("ci"),
        "F_classification": item["judge"]["classification"],
        "eps_F": float(item["band"]["eps_F"]),
        "raw_loss_applicable": raw_applicable,
        "raw_loss_classification": (
            item["raw_loss"]["classification"]
            if raw_applicable
            else "not_applicable_penalty_active"
        ),
        "raw_material_loss": bool(
            raw_applicable
            and item["raw_loss"]["classification"] == "material_loss"
        ),
    }


def _worst_by(
    comparisons: list[dict[str, Any]], dimension: str
) -> list[dict[str, Any]]:
    worst: dict[str, dict[str, Any]] = {}
    for item in comparisons:
        mean = _finite(item.get("judge", {}).get("mean_difference"))
        if mean is None:
            continue
        record = _worst_record(item)
        key = record[dimension]
        if key not in worst or mean > worst[key]["mean_F_difference"]:
            worst[key] = record
    return [worst[key] for key in sorted(worst)]


def _raw_loss_record(item: dict[str, Any]) -> dict[str, Any]:
    raw = item["raw_loss"]
    cell = item["cell"]
    return {
        **_state_dimensions(cell["state_id"]),
        "replicate": int(cell["replicate"]),
        "classification": raw["classification"],
        "guarded_arm_to_baseline_ratio": raw.get("arm_to_baseline_ratio"),
        "decision_ci": raw.get("ci"),
        "mean_loss_margin": raw.get("mean_difference"),
        "reason": raw.get("reason"),
    }


def _validate_counts(row: dict[str, Any]) -> None:
    scheduled = int(row["scheduled"])
    score = row["score_counts"]
    if sum(score.values()) != scheduled:
        raise RuntimeError(
            "score counts do not preserve the scheduled denominator"
        )
    raw = row["raw_loss_gate_counts"]
    if sum(raw.values()) != scheduled:
        raise RuntimeError(
            "raw-loss counts do not preserve the scheduled denominator"
        )


def _validate_observations(
    row: dict[str, Any], comparisons: list[dict[str, Any]]
) -> None:
    if len(comparisons) != row["observed"]:
        raise RuntimeError("setting observed count differs from comparisons")
    score = row["score_counts"]
    for key in ("beneficial", "harmful", "practical_tie", "unresolved"):
        if score[key] != sum(
            item["judge"]["classification"] == key for item in comparisons
        ):
            raise RuntimeError("setting score classification count changed")
    raw = row["raw_loss_gate_counts"]
    for key in ("material_loss", "no_material_loss", "unresolved"):
        if raw[key] != sum(
            not item["comparison_penalty_active"]
            and item["raw_loss"]["classification"] == key
            for item in comparisons
        ):
            raise RuntimeError("setting raw-loss classification count changed")
    if raw["not_applicable_penalty_active"] != sum(
        item["comparison_penalty_active"] for item in comparisons
    ):
        raise RuntimeError("setting raw-loss applicability count changed")


def _timing_by_setting(
    timing: dict[str, Any],
    setting_rows: list[dict[str, Any]],
    integration_manifest_sha256: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    if timing.get("mode") != "timing":
        raise RuntimeError("timing input is not a warmed timing summary")
    rows_by_setting: dict[tuple[str, int], list[dict[str, Any]]] = {}
    seen_cells: set[tuple[str, str, int]] = set()
    bound_rows = 0
    timing_hashes = set()
    for row in timing["rows"]:
        if row["status"] not in {"succeeded", "failed", "missing"}:
            raise RuntimeError("timing row status is invalid")
        key = (row["method"], int(row["budget"]))
        cell_key = (row["state_id"], *key)
        if cell_key in seen_cells:
            raise RuntimeError("timing contains a duplicate state setting")
        seen_cells.add(cell_key)
        rows_by_setting.setdefault(key, []).append(row)
        if row["status"] != "missing":
            bound_rows += 1
            timing_hashes.add(row["identity"]["timing_source_sha256"])
            identity_setting = row["identity"]["setting"]
            if identity_setting != {"method": key[0], "budget": key[1]}:
                raise RuntimeError("timing row setting identity changed")
            if (
                row["identity"]["integration_manifest_sha256"]
                != integration_manifest_sha256
            ):
                raise RuntimeError(
                    "timing belongs to another integration manifest"
                )
    if bound_rows == 0:
        raise RuntimeError("all-missing timing summary has no source identity")
    if len(timing_hashes) != 1:
        raise RuntimeError("timing rows use mixed timing implementations")
    output = {}
    total_scheduled = 0
    for judge_row in setting_rows:
        key = (judge_row["method"], int(judge_row["budget"]))
        if judge_row["scheduled"] % E2_REPLICATES:
            raise RuntimeError(
                "judge setting denominator is not eight replicates"
            )
        scheduled_states = judge_row["scheduled"] // E2_REPLICATES
        total_scheduled += scheduled_states
        rows = rows_by_setting.pop(key, [])
        if len(rows) > scheduled_states:
            raise RuntimeError("timing contains too many state cells")
        succeeded = [row for row in rows if row["status"] == "succeeded"]
        failed = [row for row in rows if row["status"] == "failed"]
        explicit_missing = [row for row in rows if row["status"] == "missing"]
        for row in succeeded:
            ratio = row["result"]["median_paired_ratio"]
            ratio_range = row["result"]["paired_ratio_range"]
            if (
                not isinstance(ratio_range, list)
                or len(ratio_range) != 2
                or not all(
                    _finite(value) is not None and value > 0
                    for value in ratio_range
                )
                or ratio_range[0] > ratio_range[1]
                or _finite(ratio) is None
                or ratio <= 0
                or not ratio_range[0] <= ratio <= ratio_range[1]
            ):
                raise RuntimeError("timing ratio evidence is invalid")
        ratios = [
            float(row["result"]["median_paired_ratio"]) for row in succeeded
        ]
        state_ratios = [
            {
                "state_id": row["state_id"],
                "median_paired_ratio": float(
                    row["result"]["median_paired_ratio"]
                ),
                "paired_ratio_range": [
                    float(value)
                    for value in row["result"]["paired_ratio_range"]
                ],
            }
            for row in succeeded
        ]
        state_ratios.sort(key=lambda item: item["state_id"])
        output[key] = {
            "scheduled_state_count": int(scheduled_states),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "missing": int(scheduled_states - len(succeeded) - len(failed)),
            "explicit_missing": len(explicit_missing),
            "unrepresented_missing": int(scheduled_states - len(rows)),
            "median_of_state_median_paired_ratios": (
                None if not ratios else float(np.median(ratios))
            ),
            "state_median_ratio_range": (
                None
                if not ratios
                else [float(min(ratios)), float(max(ratios))]
            ),
            "state_ratios": state_ratios,
            "failed_states": [
                {
                    "state_id": row["state_id"],
                    "error": row.get("result"),
                }
                for row in failed
            ],
        }
    if rows_by_setting:
        raise RuntimeError("timing input contains an unknown setting")
    if int(timing["scheduled"]) != total_scheduled:
        raise RuntimeError("timing scheduled denominator differs from judge")
    return output


def _mc_diagnostic(crosscheck: dict[str, Any]) -> dict[str, Any]:
    if crosscheck.get("automatic_override") is not False:
        raise RuntimeError("MC crosscheck must remain diagnostic")
    scheduled = int(crosscheck["scheduled_pair_count"])
    counts = dict(crosscheck["status_counts"])
    raw_counts = dict(crosscheck["raw_status_counts"])
    if (
        sum(counts.values()) != scheduled
        or sum(raw_counts.values()) != scheduled
        or sum(
            value
            for key, value in counts.items()
            if key != "failed_or_missing"
        )
        != len(crosscheck["comparisons"])
        or sum(
            value
            for key, value in raw_counts.items()
            if key != "failed_or_missing"
        )
        != len(crosscheck["comparisons"])
    ):
        raise RuntimeError("MC crosscheck denominator is inconsistent")
    disagreements = [
        {
            "pair_id": item["pair_id"],
            "state_id": item["state_id"],
            "comparison_tags": item["comparison_tags"],
            "score_status": item["cross_method_status"],
            "raw_status": item["raw_cross_method_status"],
        }
        for item in crosscheck["comparisons"]
        if item["cross_method_status"] == "disagree"
        or item["raw_cross_method_status"] == "disagree"
    ]
    return {
        "budget": int(crosscheck["budget"]),
        "scheduled_pair_count": scheduled,
        "score_status_counts": counts,
        "raw_status_counts": raw_counts,
        "disagreements": disagreements,
        "failed_states": crosscheck["failed_states"],
        "automatic_override": False,
    }


def reduce_evidence(
    judge: dict[str, Any],
    *,
    judge_source: dict[str, Any],
    timing: dict[str, Any] | None = None,
    timing_source: dict[str, Any] | None = None,
    crosscheck: dict[str, Any] | None = None,
    crosscheck_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build compact tables without changing any scientific classification."""
    if (timing is None) != (timing_source is None):
        raise RuntimeError(
            "timing data and source identity must be supplied together"
        )
    if (crosscheck is None) != (crosscheck_source is None):
        raise RuntimeError(
            "MC data and source identity must be supplied together"
        )
    grouped = judge["grouped"]
    setting_rows = grouped["by_setting"]
    setting_keys = [
        (row["method"], int(row["budget"])) for row in setting_rows
    ]
    if len(setting_keys) != len(set(setting_keys)):
        raise RuntimeError("judge contains a duplicate method/budget setting")
    timing_rows = (
        {}
        if timing is None
        else _timing_by_setting(
            timing, setting_rows, judge["integration_manifest_sha256"]
        )
    )
    comparisons_by_setting: dict[tuple[str, int], list[dict[str, Any]]] = {}
    seen_cells = set()
    seen_tags = set()
    for item in judge["comparisons"]:
        cell = item["cell"]
        key = (cell["method"], int(cell["budget"]))
        if key not in set(setting_keys):
            raise RuntimeError(
                "judge comparison belongs to an unknown setting"
            )
        identity = (cell["state_id"], *key, int(cell["replicate"]))
        tag = item.get("integration_cell_tag")
        if identity in seen_cells or (tag is not None and tag in seen_tags):
            raise RuntimeError("judge contains a duplicate comparison")
        seen_cells.add(identity)
        if tag is not None:
            seen_tags.add(tag)
        comparisons_by_setting.setdefault(key, []).append(item)
    trajectory_rows = {
        (row["method"], int(row["budget"])): [] for row in setting_rows
    }
    seen_trajectories = set()
    for row in grouped["by_trajectory"]:
        _validate_counts(row)
        key = (row["method"], int(row["budget"]))
        trajectory_key = (*key, row["trajectory"])
        if key not in trajectory_rows or trajectory_key in seen_trajectories:
            raise RuntimeError("judge trajectory allocation is inconsistent")
        seen_trajectories.add(trajectory_key)
        trajectory_rows[key].append(row)

    tables = []
    for row in setting_rows:
        _validate_counts(row)
        key = (row["method"], int(row["budget"]))
        selected = comparisons_by_setting.get(key, [])
        _validate_observations(row, selected)
        score = dict(row["score_counts"])
        raw = dict(row["raw_loss_gate_counts"])
        scheduled = int(row["scheduled"])
        if (
            sum(item["scheduled"] for item in trajectory_rows[key])
            != scheduled
        ):
            raise RuntimeError("trajectory denominators differ from setting")
        for count_key in ("score_counts", "raw_loss_gate_counts"):
            for classification, value in row[count_key].items():
                if (
                    sum(
                        item[count_key][classification]
                        for item in trajectory_rows[key]
                    )
                    != value
                ):
                    raise RuntimeError("trajectory counts differ from setting")
        worst = sorted(
            (
                _worst_record(item)
                for item in selected
                if _finite(item.get("judge", {}).get("mean_difference"))
                is not None
            ),
            key=lambda item: item["mean_F_difference"],
            reverse=True,
        )[:5]
        tables.append(
            {
                "method": key[0],
                "budget": key[1],
                "scheduled": scheduled,
                "observed": int(row["observed"]),
                "score_counts": score,
                "fractions_of_scheduled": {
                    "harmful": score["harmful"] / scheduled,
                    "unresolved": score["unresolved"] / scheduled,
                    "failed_or_missing": score["failed_or_missing"]
                    / scheduled,
                },
                "raw_loss_gate": {
                    "applicable_observed": raw["material_loss"]
                    + raw["no_material_loss"]
                    + raw["unresolved"],
                    "not_applicable_penalty_active": raw[
                        "not_applicable_penalty_active"
                    ],
                    "counts": raw,
                    "material_loss_fraction_of_scheduled": raw["material_loss"]
                    / scheduled,
                },
                "trajectory_counts": trajectory_rows[key],
                "worst_by_target": _worst_by(selected, "target"),
                "worst_by_trajectory": _worst_by(selected, "trajectory"),
                "top_worst_5_F_differences": worst,
                "material_raw_loss_cells": sorted(
                    (
                        _raw_loss_record(item)
                        for item in selected
                        if not item["comparison_penalty_active"]
                        and item["raw_loss"]["classification"]
                        == "material_loss"
                    ),
                    key=lambda item: (item["state_id"], item["replicate"]),
                ),
                "timing": timing_rows.get(key),
            }
        )
    sources = {
        "judge": {
            **judge_source,
            "integration_manifest_sha256": judge[
                "integration_manifest_sha256"
            ],
            "judge_manifest_sha256": judge["judge_manifest_sha256"],
        }
    }
    if timing_source is not None:
        timing_hashes = {
            row["identity"]["timing_source_sha256"]
            for row in timing["rows"]
            if row["status"] != "missing"
        }
        sources["timing"] = {
            **timing_source,
            "integration_manifest_sha256": judge[
                "integration_manifest_sha256"
            ],
            "timing_source_sha256": next(iter(timing_hashes)),
        }
    if crosscheck_source is not None:
        if (
            crosscheck.get("source_judge_summary_sha256")
            != judge_source["sha256"]
        ):
            raise RuntimeError(
                "MC crosscheck belongs to another judge summary"
            )
        sources["mc_crosscheck"] = {
            **crosscheck_source,
            "crosscheck_manifest_sha256": crosscheck.get("manifest_sha256"),
            "source_judge_summary_sha256": crosscheck.get(
                "source_judge_summary_sha256"
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "noisy_acquisition_evidence",
        "automatic_promotion": False,
        "judge_budget": int(judge["judge_budget"]),
        "sources": sources,
        "settings": tables,
        "mc_crosscheck": (
            None if crosscheck is None else _mc_diagnostic(crosscheck)
        ),
    }


def _read_source(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")), {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--timing", type=Path)
    parser.add_argument("--mc-crosscheck", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    judge, judge_source = _read_source(args.judge)
    timing, timing_source = (None, None)
    if args.timing is not None:
        timing, timing_source = _read_source(args.timing)
    crosscheck, crosscheck_source = (None, None)
    if args.mc_crosscheck is not None:
        crosscheck, crosscheck_source = _read_source(args.mc_crosscheck)
        if crosscheck["source_judge_summary_sha256"] != judge_source["sha256"]:
            raise RuntimeError(
                "MC crosscheck belongs to another judge summary"
            )
    report = reduce_evidence(
        judge,
        judge_source=judge_source,
        timing=timing,
        timing_source=timing_source,
        crosscheck=crosscheck,
        crosscheck_source=crosscheck_source,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
