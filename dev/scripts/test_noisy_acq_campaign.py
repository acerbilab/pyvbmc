"""Protocol-only checks for the noisy-acquisition campaign controller."""

import json
from pathlib import Path

import noisy_acq_campaign as campaign
import noisy_acq_search_experiment as search
import pytest


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _patch_integration_ladder(
    monkeypatch,
    *,
    classifications,
    inventory_counts=None,
    runtime_identities=None,
):
    prepared = []
    runtime_values = iter(runtime_identities or [{"source": "fixed"}] * 20)
    inventory_counts = inventory_counts or {
        "succeeded": 1,
        "failed": 0,
        "partial": 0,
        "missing": 0,
    }
    monkeypatch.setattr(campaign.integration, "JUDGE_BUDGETS", (10, 20))
    monkeypatch.setattr(
        campaign.integration,
        "validate_manifest",
        lambda manifest, require_ready: None,
    )
    monkeypatch.setattr(
        campaign.integration,
        "runtime_identity",
        lambda manifest: next(runtime_values),
    )
    monkeypatch.setattr(
        campaign.integration,
        "inventory",
        lambda manifest, results: {
            "counts": dict(inventory_counts),
            "production_baseline": {"counts": dict(inventory_counts)},
        },
    )

    def prepare(manifest_path, results, budget, previous):
        prepared.append((budget, previous))
        return {
            "unions": [{"state_id": "state"}],
            "budget": budget,
            "launch_ready": False,
            "holdout_locked": True,
        }

    monkeypatch.setattr(
        campaign.integration, "prepare_judge_manifest", prepare
    )
    monkeypatch.setattr(
        campaign.integration,
        "validate_judge_manifest",
        lambda manifest, require_ready: None,
    )
    monkeypatch.setattr(
        campaign.integration, "judge_identity", lambda manifest: None
    )

    def run(path, manifest, out):
        index = len(prepared) - 1
        classification = classifications[index]
        summary = {
            "comparisons": [
                {
                    "judge": {"classification": classification},
                    "raw_loss": {"classification": "no_material_loss"},
                    "comparison_penalty_active": False,
                }
            ],
            "counts": {classification: 1, "failed_or_missing": 0},
            "raw_loss_gate_counts": {
                "no_material_loss": 1,
                "failed_or_missing": 0,
            },
        }
        _write(Path(out) / "summary.json", summary)
        return 0

    monkeypatch.setattr(campaign.integration, "run_judge_controller", run)
    monkeypatch.setattr(
        campaign.integration,
        "comparison_needs_escalation",
        lambda item: item["judge"]["classification"] == "unresolved",
    )
    return prepared


def test_integration_ladder_completes_and_resumes_identically(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"split": "development", "launch_ready": True})
    prepared = _patch_integration_ladder(
        monkeypatch, classifications=("unresolved", "practical_tie") * 2
    )
    out = tmp_path / "campaign"
    first = campaign.integration_ladder(manifest, tmp_path / "results", out)
    second = campaign.integration_ladder(manifest, tmp_path / "results", out)
    assert first == second
    assert [item["budget"] for item in first["rounds"]] == [10, 20]
    assert prepared[0][1] is None
    assert Path(prepared[1][1]).name == "summary.json"
    assert (out / "completed_ladder.json").is_file()


def test_integration_ladder_rejects_pending_allocation(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"split": "development", "launch_ready": True})
    _patch_integration_ladder(
        monkeypatch,
        classifications=("unresolved",),
        inventory_counts={
            "succeeded": 0,
            "failed": 0,
            "partial": 1,
            "missing": 1,
        },
    )
    with pytest.raises(RuntimeError, match="pending cells"):
        campaign.integration_ladder(
            manifest, tmp_path / "results", tmp_path / "campaign"
        )


def test_integration_ladder_does_not_escalate_resolved(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"split": "development", "launch_ready": True})
    prepared = _patch_integration_ladder(
        monkeypatch, classifications=("practical_tie",)
    )
    report = campaign.integration_ladder(
        manifest, tmp_path / "results", tmp_path / "campaign"
    )
    assert len(prepared) == 1
    assert len(report["rounds"]) == 1
    assert report["rounds"][0]["pending_comparisons"] == 0


def test_integration_ladder_rejects_source_identity_change(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"split": "development", "launch_ready": True})
    _patch_integration_ladder(
        monkeypatch,
        classifications=("practical_tie",),
        runtime_identities=[{"source": "before"}, {"source": "after"}],
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        campaign.integration_ladder(
            manifest, tmp_path / "results", tmp_path / "campaign"
        )


def _patch_search_ladder(monkeypatch, tmp_path):
    calls = {"workers": [], "summaries": []}
    marker = tmp_path / "selection.complete.json"
    marker.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        search, "validate_manifest", lambda manifest, require_ready: None
    )
    monkeypatch.setattr(
        search, "runtime_identity", lambda manifest: {"source": "fixed"}
    )
    monkeypatch.setattr(search, "cell_tag", lambda cell: cell["state_id"])
    monkeypatch.setattr(
        search,
        "_paths",
        lambda out, tag, kind: {"complete": marker},
    )
    monkeypatch.setattr(
        search,
        "_validate_terminal",
        lambda manifest, paths, cell: {"status": "succeeded"},
    )
    monkeypatch.setattr(
        search,
        "comparison_needs_escalation",
        lambda item: item["judge"]["classification"] == "unresolved",
    )

    def run(command, check):
        assert check is True
        calls["workers"].append(list(command))

    monkeypatch.setattr(campaign.subprocess, "run", run)

    def summary(manifest, results, budget, previous):
        calls["summaries"].append((budget, previous))
        comparisons = [
            {
                "comparison_tag": "state-a",
                "cell": {"state_id": "state-a"},
                "judge": {"classification": "practical_tie"},
            },
            {
                "comparison_tag": "state-b",
                "cell": {"state_id": "state-b"},
                "judge": {
                    "classification": (
                        "unresolved" if budget == 10 else "practical_tie"
                    )
                },
            },
        ]
        return {
            "comparisons": comparisons,
            "primary_screen_counts": {"scheduled": 2},
            "component_diagnostic_counts": None,
        }

    monkeypatch.setattr(search, "judge_summary", summary)
    return calls


def _search_manifest(path, split="development"):
    value = {
        "split": split,
        "launch_ready": True,
        "states": [{"state_id": "state-a"}, {"state_id": "state-b"}],
        "cells": [{"state_id": "state-a"}, {"state_id": "state-b"}],
        "judge_budgets": [10, 20],
    }
    _write(path, value)
    return value


def test_search_ladder_runs_only_pending_states_with_predecessor(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "search.json"
    _search_manifest(manifest)
    calls = _patch_search_ladder(monkeypatch, tmp_path)
    report = campaign.search_ladder(
        manifest, tmp_path / "results", tmp_path / "campaign"
    )
    worker_states = [
        command[command.index("--state-id") + 1]
        for command in calls["workers"]
    ]
    worker_budgets = [
        command[command.index("--budget") + 1] for command in calls["workers"]
    ]
    assert worker_states == ["state-a", "state-b", "state-b"]
    assert worker_budgets == ["10", "10", "20"]
    assert "--previous-summary" not in calls["workers"][0]
    predecessor = calls["workers"][2]
    previous_path = predecessor[predecessor.index("--previous-summary") + 1]
    assert Path(previous_path).name == "b10_summary.json"
    assert calls["summaries"][0] == (10, None)
    assert Path(calls["summaries"][1][1]).name == "b10_summary.json"
    assert [item["pending_comparisons"] for item in report["rounds"]] == [1, 0]


def test_search_ladder_resume_is_immutable(tmp_path, monkeypatch):
    manifest = tmp_path / "search.json"
    _search_manifest(manifest)
    _patch_search_ladder(monkeypatch, tmp_path)
    out = tmp_path / "campaign"
    first = campaign.search_ladder(manifest, tmp_path / "results", out)
    second = campaign.search_ladder(manifest, tmp_path / "results", out)
    assert first == second
    _write(out / "b10_summary.json", {"tampered": True})
    with pytest.raises(
        RuntimeError, match="existing campaign artifact differs"
    ):
        campaign.search_ladder(manifest, tmp_path / "results", out)


def test_search_ladder_refuses_holdout_before_workers(tmp_path, monkeypatch):
    manifest = tmp_path / "search.json"
    _search_manifest(manifest, split="holdout")
    calls = _patch_search_ladder(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="requires --allow-holdout"):
        campaign.search_ladder(
            manifest, tmp_path / "results", tmp_path / "campaign"
        )
    assert calls["workers"] == []


def test_search_cli_routes_required_paths_and_holdout_flag(
    tmp_path, monkeypatch
):
    received = []
    monkeypatch.setattr(
        campaign,
        "search_ladder",
        lambda manifest, results, out, allow: received.append(
            (manifest, results, out, allow)
        ),
    )
    monkeypatch.setattr(
        campaign.sys,
        "argv",
        [
            "noisy_acq_campaign.py",
            "search",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--results",
            str(tmp_path / "results"),
            "--out",
            str(tmp_path / "out"),
            "--allow-holdout",
        ],
    )
    campaign.main()
    assert received == [
        (
            tmp_path / "manifest.json",
            tmp_path / "results",
            tmp_path / "out",
            True,
        )
    ]


def test_integration_ladder_rejects_replaced_selections(tmp_path, monkeypatch):
    manifest = tmp_path / "integration.json"
    _write(manifest, {"split": "development"})
    prepared = _patch_integration_ladder(
        monkeypatch, classifications=["unresolved", "practical_tie"]
    )
    marker = tmp_path / "selection.complete.json"
    marker.write_text("original", encoding="utf-8")
    monkeypatch.setattr(
        campaign,
        "_terminal_snapshot",
        lambda manifest, results, kind: {
            "selected": campaign.integration.sha256_file(marker)
        },
    )
    original_run = campaign.integration.run_judge_controller

    def run(path, allocated, out):
        result = original_run(path, allocated, out)
        marker.write_text("replacement", encoding="utf-8")
        return result

    monkeypatch.setattr(campaign.integration, "run_judge_controller", run)
    with pytest.raises(RuntimeError, match="identity changed before judging"):
        campaign.integration_ladder(
            manifest, tmp_path / "results", tmp_path / "campaign"
        )
    assert len(prepared) == 1


def test_search_ladder_rejects_replaced_selections(tmp_path, monkeypatch):
    manifest = tmp_path / "search.json"
    _search_manifest(manifest)
    calls = _patch_search_ladder(monkeypatch, tmp_path)
    original_run = campaign.subprocess.run

    def run(command, check):
        original_run(command, check)
        (tmp_path / "selection.complete.json").write_text(
            "replacement", encoding="utf-8"
        )

    monkeypatch.setattr(campaign.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="identity changed before search"):
        campaign.search_ladder(
            manifest, tmp_path / "results", tmp_path / "campaign"
        )
    assert len(calls["workers"]) == 2
