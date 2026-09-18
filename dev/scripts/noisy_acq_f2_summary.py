"""Summarize one F2 frozen-state stage from its judge ladder and records.

Reads the frozen search manifest, the completed judge summary of the
ladder's last budget, the selection records and the paired timing records
of one stage, and writes one evidence record plus Markdown tables: per
contrast and per trajectory the beneficial, practical-tie, harmful and
unresolved counts with the material raw losses; the node count every
selection actually used against the count its arm prescribes; the paired
timing ratios per treatment as medians of state medians; the E2 screening
gate applied to each contrast; and, for the development stage, the
component order chosen for holdout confirmation. It performs no numerical
work of its own and can run on any completed E3 or F2 stage.

    python dev/scripts/noisy_acq_f2_summary.py --manifest M.json \
        --results RESULTS_DIR --judge-summary bNNNN_summary.json \
        --out f2_stage_evidence.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import noisy_acq_integration as integration
import noisy_acq_search_experiment as campaign

COST_GATE = 1.10
HARMFUL_GATE = 0.05
UNRESOLVED_CLAIM_LIMIT = 0.20
SCORE_CLASSES = ("beneficial", "practical_tie", "harmful", "unresolved")


def _label(state_id: str) -> str:
    return campaign._state_dimensions(state_id)["label"]


def _checkpoint(state_id: str) -> str:
    return campaign._state_dimensions(state_id)["checkpoint"]


def node_counts(manifest: dict[str, Any], results: Path) -> dict[str, Any]:
    """Every succeeded selection must have used its arm's node count."""
    prescribed = {}
    for treatment in manifest["treatments"]:
        config = campaign.search.SearchConfig(**treatment["config"])
        prescribed[treatment["tag"]] = (
            int(config.importance_qmc_samples)
            if config.importance_qmc
            else 100
        )
    observed: dict[str, dict[int, int]] = {tag: {} for tag in prescribed}
    mismatches = []
    succeeded = 0
    for path in sorted((results / "selection").glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "succeeded":
            continue
        succeeded += 1
        tag = report["treatment"]["tag"]
        count = report["details"].get("importance_node_count")
        if count is None:
            # Records written before the node count was recorded (E3).
            observed[tag]["unrecorded"] = (
                observed[tag].get("unrecorded", 0) + 1
            )
            continue
        count = int(count)
        observed[tag][count] = observed[tag].get(count, 0) + 1
        if count != prescribed[tag]:
            mismatches.append({"selection": path.stem, "count": count})
    return {
        "prescribed": prescribed,
        "observed": {
            tag: {str(k): v for k, v in counts.items()}
            for tag, counts in observed.items()
        },
        "succeeded_selections": succeeded,
        "mismatches": mismatches,
    }


def _empty_counts() -> dict[str, int]:
    return {name: 0 for name in SCORE_CLASSES}


def contrast_tables(
    manifest: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    scheduled_per_contrast = len(manifest["states"]) * manifest["replicates"]
    specs = [
        spec
        for spec in campaign._contrast_specs(manifest)
        if spec["kind"] == "primary_vs_s0"
    ]
    tables: dict[str, Any] = {}
    for spec in specs:
        contrast = spec["contrast"]
        rows = [
            item
            for item in summary["comparisons"]
            if item["cell"]["contrast"] == contrast
        ]
        counts = _empty_counts()
        raw = {"material_loss": 0, "no_material_loss": 0, "unresolved": 0}
        by_label: dict[str, dict[str, int]] = {}
        by_checkpoint: dict[str, dict[str, int]] = {}
        losses = []
        for item in rows:
            classification = item["judge"]["classification"]
            counts[classification] = counts.get(classification, 0) + 1
            label = _label(item["cell"]["state_id"])
            checkpoint = _checkpoint(item["cell"]["state_id"])
            by_label.setdefault(label, _empty_counts())
            by_label[label][classification] = (
                by_label[label].get(classification, 0) + 1
            )
            by_checkpoint.setdefault(checkpoint, _empty_counts())
            by_checkpoint[checkpoint][classification] = (
                by_checkpoint[checkpoint].get(classification, 0) + 1
            )
            raw_class = item["raw_loss"].get("classification")
            if raw_class in raw:
                raw[raw_class] += 1
            if raw_class == "material_loss":
                losses.append(
                    {
                        "state_id": item["cell"]["state_id"],
                        "replicate": item["cell"]["replicate"],
                        "arm_to_baseline_ratio": item["raw_loss"].get(
                            "arm_to_baseline_ratio"
                        ),
                        "mean_score_difference": item["judge"].get(
                            "mean_difference"
                        ),
                    }
                )
        observed = len(rows)
        gate = {
            "scheduled": scheduled_per_contrast,
            "observed": observed,
            "harmful_fraction_of_scheduled": counts["harmful"]
            / scheduled_per_contrast,
            "unresolved_fraction_of_scheduled": (
                counts["unresolved"] + scheduled_per_contrast - observed
            )
            / scheduled_per_contrast,
            "material_raw_losses": raw["material_loss"],
        }
        gate["harmful_within_gate"] = (
            gate["harmful_fraction_of_scheduled"] <= HARMFUL_GATE
        )
        gate["positive_claim_allowed"] = (
            gate["unresolved_fraction_of_scheduled"] <= UNRESOLVED_CLAIM_LIMIT
        )
        tables[contrast] = {
            "arm_tag": spec["arm_tag"],
            "score_counts": counts,
            "raw_loss_counts": raw,
            "material_losses": losses,
            "by_label": by_label,
            "by_checkpoint": by_checkpoint,
            "gate": gate,
        }
    return tables


def timing_tables(manifest: dict[str, Any], results: Path) -> dict[str, Any]:
    ratios: dict[str, dict[str, float]] = {}
    for path in sorted((results / "timing").glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "succeeded":
            continue
        tag = report["treatment"]["tag"]
        ratios.setdefault(tag, {})[report["cell"]["state_id"]] = report[
            "timing"
        ]["median_paired_ratio"]
    tables = {}
    for tag, per_state in ratios.items():
        values = list(per_state.values())
        tables[tag] = {
            "state_median_ratios": per_state,
            "states_timed": len(values),
            "states_scheduled": len(manifest["states"]),
            "median_of_state_medians": statistics.median(values),
            "range": [min(values), max(values)],
            "cost_within_gate": statistics.median(values) <= COST_GATE,
        }
    return tables


def development_choice(
    tables: dict[str, Any], timing: dict[str, Any]
) -> dict[str, Any]:
    """Prefer fewer harmful selections, then more beneficial ones, then
    fewer material raw losses, then the lower cost ratio; index order
    names the tags so the rule is explicit in the record."""
    candidates = []
    for contrast, table in tables.items():
        tag = table["arm_tag"]
        counts = table["score_counts"]
        candidates.append(
            (
                counts["harmful"],
                -counts["beneficial"],
                table["raw_loss_counts"]["material_loss"],
                timing.get(tag, {}).get(
                    "median_of_state_medians", float("inf")
                ),
                tag,
            )
        )
    candidates.sort()
    ordered = [item[-1] for item in candidates]
    return {
        "rule": (
            "fewest harmful, then most beneficial, then fewest material raw"
            " losses, then lowest median cost ratio"
        ),
        "ranking": ordered,
        "chosen_tag": ordered[0] if ordered else None,
        "tie_on_harmful_and_beneficial": len(candidates) > 1
        and candidates[0][:2] == candidates[1][:2],
    }


def markdown(record: dict[str, Any]) -> str:
    lines = []
    lines.append(
        "| Contrast | Beneficial | Practical tie | Harmful | Unresolved |"
        " Material raw loss | Harmful / scheduled | Cost ratio |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for contrast, table in record["contrasts"].items():
        counts = table["score_counts"]
        timing = record["timing"].get(table["arm_tag"], {})
        cost = timing.get("median_of_state_medians")
        lines.append(
            f"| {contrast} | {counts['beneficial']} | {counts['practical_tie']}"
            f" | {counts['harmful']} | {counts['unresolved']} |"
            f" {table['raw_loss_counts']['material_loss']} |"
            f" {table['gate']['harmful_fraction_of_scheduled']:.3f} |"
            f" {'n/a' if cost is None else f'{cost:.3f}'} |"
        )
    for contrast, table in record["contrasts"].items():
        lines.append("")
        lines.append(f"Per trajectory, {contrast}:")
        lines.append("")
        lines.append(
            "| Trajectory | Beneficial | Practical tie | Harmful | Unresolved |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for label in sorted(table["by_label"]):
            counts = table["by_label"][label]
            lines.append(
                f"| {label} | {counts['beneficial']} | {counts['practical_tie']}"
                f" | {counts['harmful']} | {counts['unresolved']} |"
            )
    return "\n".join(lines)


def summarize(
    manifest_path: Path, results: Path, judge_summary: Path
) -> dict[str, Any]:
    manifest = campaign._load_manifest(manifest_path)
    summary = json.loads(judge_summary.read_text(encoding="utf-8"))
    tables = contrast_tables(manifest, summary)
    timing = timing_tables(manifest, results)
    record = {
        "kind": "f2_stage_evidence",
        "schema_version": 1,
        "stage": manifest["stage"],
        "split": manifest["split"],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": integration.sha256_file(manifest_path),
        "judge_summary": str(judge_summary.resolve()),
        "judge_summary_sha256": integration.sha256_file(judge_summary),
        "judge_budget": summary.get("judge_budget"),
        "inventory": campaign.inventory(manifest, results),
        "node_counts": node_counts(manifest, results),
        "contrasts": tables,
        "timing": timing,
        "gates": {
            "cost_ratio_at_most": COST_GATE,
            "harmful_fraction_at_most": HARMFUL_GATE,
            "unresolved_fraction_for_positive_claim_at_most": (
                UNRESOLVED_CLAIM_LIMIT
            ),
        },
    }
    if manifest["stage"] == "f2_development":
        record["development_choice"] = development_choice(tables, timing)
    if manifest["stage"] == "f2_holdout":
        record["holdout_confirmation"] = manifest["holdout_confirmation"]
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--judge-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    record = summarize(args.manifest, args.results, args.judge_summary)
    integration.write_json(args.out, record)
    print(markdown(record))
    print()
    print(
        json.dumps(
            {
                "node_count_mismatches": record["node_counts"]["mismatches"],
                "succeeded_selections": record["node_counts"][
                    "succeeded_selections"
                ],
                "development_choice": record.get("development_choice"),
                "timing": {
                    tag: (
                        item["median_of_state_medians"],
                        item["range"],
                        item["states_timed"],
                    )
                    for tag, item in record["timing"].items()
                },
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
