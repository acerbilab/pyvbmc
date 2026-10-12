"""Execute predeclared judging ladders for completed E2/E3 allocations.

This controller makes no treatment-selection decisions. It verifies the
parent allocation, publishes immutable per-budget manifests, and delegates
each numerical cell to the existing sequential worker controllers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import noisy_acq_integration as integration


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_immutable(path, value):
    if Path(path).exists():
        if read_json(path) != integration.capture.canonical(value):
            raise RuntimeError(f"existing campaign artifact differs: {path}")
    else:
        integration.write_json(path, value)


def _identity(manifest_path, kind, tags):
    manifest = read_json(manifest_path)
    integration.validate_manifest(manifest, require_ready=True)
    return {
        "kind": kind,
        "integration_manifest": str(Path(manifest_path).resolve()),
        "integration_manifest_sha256": integration.sha256_file(manifest_path),
        "integration_identity": integration.runtime_identity(manifest),
        "controller_sha256": integration.sha256_file(__file__),
        "explicit_comparison_tags": sorted(tags),
    }


def _terminal_snapshot(manifest, results, kind):
    """Bind validated terminal selections, including numerical failures."""
    records = {}
    if kind == "search":
        import noisy_acq_search_experiment as search

        for cell in manifest["cells"]:
            tag = search.cell_tag(cell)
            paths = search._paths(results, tag, "selection")
            if not paths["complete"].exists():
                raise RuntimeError(
                    "search selection allocation has pending cells"
                )
            search._validate_terminal(manifest, paths, cell)
            records[tag] = integration.sha256_file(paths["complete"])
    else:
        for cells_key, tag_function, paths_function, validate in (
            (
                "cells",
                integration.cell_tag,
                integration._cell_paths,
                integration.validate_cell,
            ),
            (
                "baseline_cells",
                integration.baseline_tag,
                integration._baseline_paths,
                integration.validate_baseline_cell,
            ),
        ):
            for cell in manifest.get(cells_key, []):
                tag = tag_function(cell)
                paths = paths_function(results, tag)
                validate(manifest, results, cell)
                records[tag] = integration.sha256_file(paths["complete"])
    return records


def integration_ladder(manifest_path, results, out, allow_holdout=False):
    manifest_path, results, out = map(Path, (manifest_path, results, out))
    parent = read_json(manifest_path)

    def current_identity():
        return {
            **_identity(manifest_path, "integration_judge", []),
            "results": str(results.resolve()),
            "selection_completions": _terminal_snapshot(
                parent, results, "integration"
            ),
        }

    if parent["split"] == "holdout" and not allow_holdout:
        raise RuntimeError("holdout judging requires --allow-holdout")
    inventory = integration.inventory(parent, results)
    for record in (inventory, inventory["production_baseline"]):
        if record["counts"]["missing"] or record["counts"]["partial"]:
            raise RuntimeError("selection allocation still has pending cells")
    identity = current_identity()
    write_immutable(out / "launch.json", identity)
    previous = None
    rounds = []
    for budget in integration.JUDGE_BUDGETS:
        if current_identity() != identity:
            raise RuntimeError(
                "campaign identity changed before judging round"
            )
        manifest = integration.prepare_judge_manifest(
            manifest_path, results, budget, previous
        )
        if not manifest["unions"]:
            break
        manifest["launch_ready"] = True
        if parent["split"] == "holdout":
            manifest["holdout_locked"] = False
        integration.validate_judge_manifest(manifest, require_ready=True)
        integration.judge_identity(manifest)
        path = out / f"b{budget}_manifest.json"
        cell_out = out / f"b{budget}"
        write_immutable(path, manifest)
        integration.run_judge_controller(path, manifest, cell_out)
        previous = cell_out / "summary.json"
        summary = read_json(previous)
        pending = sum(
            integration.comparison_needs_escalation(item)
            for item in summary["comparisons"]
        )
        record = {
            "budget": budget,
            "summary": str(previous.resolve()),
            "summary_sha256": integration.sha256_file(previous),
            "counts": summary["counts"],
            "raw_loss_gate_counts": summary["raw_loss_gate_counts"],
            "pending_comparisons": pending,
        }
        rounds.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if not pending:
            break
    if current_identity() != identity:
        raise RuntimeError("campaign identity changed during judging")
    report = {"identity": identity, "rounds": rounds}
    write_immutable(out / "completed_ladder.json", report)
    return report


def mc_ladder(manifest_path, judge_summary, out, tags=(), allow_holdout=False):
    import noisy_acq_crosscheck as crosscheck

    manifest_path, judge_summary, out = map(
        Path, (manifest_path, judge_summary, out)
    )
    parent = read_json(manifest_path)
    identity = _identity(manifest_path, "ordinary_mc_crosscheck", tags)
    identity["judge_summary_sha256"] = integration.sha256_file(judge_summary)
    identity["crosscheck_source_hashes"] = crosscheck.source_hashes()
    if parent["split"] == "holdout" and not allow_holdout:
        raise RuntimeError("holdout crosscheck requires --allow-holdout")
    write_immutable(out / "launch.json", identity)
    previous = None
    rounds = []
    for budget in crosscheck.BUDGETS:
        manifest = crosscheck.prepare_manifest(
            judge_summary, manifest_path, budget, previous, list(tags)
        )
        if not manifest["unions"]:
            break
        manifest["launch_ready"] = True
        if parent["split"] == "holdout":
            manifest["holdout_locked"] = False
        crosscheck.validate_manifest(manifest, require_ready=True)
        crosscheck.runtime_identity(manifest)
        path = out / f"b{budget}_manifest.json"
        cell_out = out / f"b{budget}"
        write_immutable(path, manifest)
        crosscheck.run_controller(path, manifest, cell_out)
        previous = cell_out / "summary.json"
        summary = read_json(previous)
        record = {
            "budget": budget,
            "summary": str(previous.resolve()),
            "summary_sha256": integration.sha256_file(previous),
            "status_counts": summary["status_counts"],
            "raw_status_counts": summary["raw_status_counts"],
        }
        rounds.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    actual = _identity(manifest_path, "ordinary_mc_crosscheck", tags)
    actual["judge_summary_sha256"] = integration.sha256_file(judge_summary)
    actual["crosscheck_source_hashes"] = crosscheck.source_hashes()
    if actual != identity:
        raise RuntimeError("campaign identity changed during crosscheck")
    report = {"identity": identity, "rounds": rounds}
    write_immutable(out / "completed_ladder.json", report)
    return report


def search_ladder(manifest_path, results, out, allow_holdout=False):
    import noisy_acq_search_experiment as search

    manifest_path, results, out = map(Path, (manifest_path, results, out))
    manifest = read_json(manifest_path)
    search.validate_manifest(manifest, require_ready=True)

    def current_identity():
        return {
            "kind": "search_judge",
            "search_manifest_sha256": integration.sha256_file(manifest_path),
            "search_identity": search.runtime_identity(manifest),
            "controller_sha256": integration.sha256_file(__file__),
            "results": str(results.resolve()),
            "selection_completions": _terminal_snapshot(
                manifest, results, "search"
            ),
        }

    if manifest["split"] == "holdout" and not allow_holdout:
        raise RuntimeError("holdout judging requires --allow-holdout")
    identity = current_identity()
    write_immutable(out / "launch.json", identity)
    previous = None
    pending_states = {state["state_id"] for state in manifest["states"]}
    rounds = []
    for budget in manifest["judge_budgets"]:
        if not pending_states:
            break
        if current_identity() != identity:
            raise RuntimeError(
                "campaign identity changed before search judging round"
            )
        for state_id in sorted(pending_states):
            command = [
                sys.executable,
                "-u",
                str(Path(search.__file__).resolve()),
                "judge-cell",
                "--manifest",
                str(manifest_path.resolve()),
                "--out",
                str(results.resolve()),
                "--state-id",
                state_id,
                "--budget",
                str(budget),
            ]
            if previous is not None:
                command.extend(["--previous-summary", str(previous.resolve())])
            subprocess.run(command, check=True)
            print(f"JUDGED {state_id} budget={budget}", flush=True)
        summary = search.judge_summary(manifest, results, budget, previous)
        previous = out / f"b{budget}_summary.json"
        write_immutable(previous, summary)
        pending = [
            item
            for item in summary["comparisons"]
            if search.comparison_needs_escalation(item)
        ]
        pending_states = {item["cell"]["state_id"] for item in pending}
        record = {
            "budget": budget,
            "summary": str(previous.resolve()),
            "summary_sha256": integration.sha256_file(previous),
            "primary_screen_counts": summary["primary_screen_counts"],
            "component_diagnostic_counts": summary[
                "component_diagnostic_counts"
            ],
            "direct_control_counts": summary.get("direct_control_counts"),
            "pending_comparisons": len(pending),
        }
        rounds.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    if current_identity() != identity:
        raise RuntimeError("campaign identity changed during search judging")
    report = {"identity": identity, "rounds": rounds}
    write_immutable(out / "completed_ladder.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("integration", "mc", "search"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--judge-summary", type=Path)
    parser.add_argument("--comparison-tag", action="append", default=[])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-holdout", action="store_true")
    args = parser.parse_args()
    if args.kind in {"integration", "search"}:
        if args.results is None or args.comparison_tag:
            parser.error("judging needs --results and no comparison tags")
        ladder = (
            integration_ladder if args.kind == "integration" else search_ladder
        )
        ladder(args.manifest, args.results, args.out, args.allow_holdout)
    else:
        if args.judge_summary is None:
            parser.error("mc needs --judge-summary")
        mc_ladder(
            args.manifest,
            args.judge_summary,
            args.out,
            args.comparison_tag,
            args.allow_holdout,
        )


if __name__ == "__main__":
    main()
