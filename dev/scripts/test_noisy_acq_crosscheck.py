"""Protocol checks for the ordinary-MC crosscheck runner."""

import copy
import json

import noisy_acq_crosscheck as crosscheck
import numpy as np
import pytest


def comparison(
    tag,
    state,
    mean,
    *,
    eps=0.1,
    arm=1,
    baseline=2,
    raw="no_material_loss",
):
    return {
        "integration_cell_tag": tag,
        "cell": {"state_id": state},
        "selected_global_index": arm,
        "baseline_global_index": baseline,
        "band": {"eps_F": eps},
        "judge": {
            "mean_difference": mean,
            "classification": "harmful" if mean > eps else "beneficial",
        },
        "raw_loss": {"classification": raw},
    }


def test_development_policy_selects_extrema_threshold_and_raw_loss():
    items = [
        comparison("low", "s1", -0.2, arm=1),
        comparison("middle", "s1", 0.0, arm=3),
        comparison("high", "s1", 0.3, arm=4),
        comparison("far", "s1", 1.1, arm=5),
        comparison("raw", "s1", 0.1, arm=6, raw="material_loss"),
        comparison("only", "s2", 0.05, arm=7),
    ]
    tags, reasons = crosscheck.select_development_tags(items)
    assert tags == ["far", "low", "only", "raw"]
    assert "state_min_mean_difference" in reasons["low"]
    assert "state_max_mean_difference" in reasons["far"]
    assert "ten_practical_bands" in reasons["far"]
    assert "confident_raw_material_loss" in reasons["raw"]
    assert set(reasons["only"]) == {
        "state_min_mean_difference",
        "state_max_mean_difference",
    }


def test_pair_deduplication_retains_tags_and_reasons():
    first = comparison("a", "s", -1.0, arm=4, baseline=9)
    second = comparison("b", "s", 1.0, arm=4, baseline=9)
    pairs = crosscheck._deduplicate_pairs(
        [first, second], {"a": ["low"], "b": ["high"]}
    )
    assert len(pairs) == 1
    assert pairs[0]["comparison_tags"] == ["a", "b"]
    assert pairs[0]["selection_reasons"] == ["high", "low"]
    assert len(pairs[0]["source_comparisons"]) == 2


def minimal_manifest(split="development"):
    indices = [1, 2]
    union = {
        "state_id": "target_seed0_early",
        "candidate_indices": indices,
        "candidate_union_sha256": crosscheck.capture.digest(
            (
                "target_seed0_early",
                np.asarray(indices, dtype=np.int64),
            )
        ),
        "replicate_seeds": list(range(crosscheck.REPLICATES)),
        "pairs": [
            {
                "pair_id": "pair",
                "arm_global_index": 1,
                "baseline_global_index": 2,
            }
        ],
    }
    tags = ["explicit_tag"] if split == "holdout" else []
    return {
        "schema_version": crosscheck.SCHEMA_VERSION,
        "kind": "ordinary_mc_crosscheck",
        "launch_ready": True,
        "split": split,
        "holdout_locked": split == "holdout",
        "budget": crosscheck.BUDGETS[0],
        "replicates": crosscheck.REPLICATES,
        "selection_policy": (
            "explicit_holdout_only"
            if split == "holdout"
            else "development_extrema_and_catastrophes"
        ),
        "unions": [union],
        "state_cell_count": 1,
        "scheduled_pair_count": 1,
        "carried_comparisons": [],
        "frozen_comparison_tags": tags,
        "explicit_comparison_tags": tags,
    }


def test_manifest_enforces_holdout_lock_and_union_identity():
    manifest = minimal_manifest("holdout")
    with pytest.raises(RuntimeError, match="locked"):
        crosscheck.validate_manifest(manifest, require_ready=True)
    manifest["holdout_locked"] = False
    crosscheck.validate_manifest(manifest, require_ready=True)
    manifest["unions"][0]["candidate_indices"] = [2, 1]
    with pytest.raises(RuntimeError, match="identity"):
        crosscheck.validate_manifest(manifest, require_ready=False)


def test_manifest_rejects_explicit_development_expansion():
    manifest = minimal_manifest()
    manifest["explicit_comparison_tags"] = ["post_outcome_addition"]
    with pytest.raises(RuntimeError, match="development.*explicit"):
        crosscheck.validate_manifest(manifest, require_ready=False)


def test_manifest_rejects_unknown_split():
    manifest = minimal_manifest()
    manifest["split"] = "review"
    with pytest.raises(RuntimeError, match="split is invalid"):
        crosscheck.validate_manifest(manifest, require_ready=False)


def test_runtime_binds_split_to_parent_manifest(tmp_path, monkeypatch):
    manifest = minimal_manifest()
    parent = {"split": "holdout"}
    parent_path = tmp_path / "integration.json"
    summary_path = tmp_path / "summary.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    summary_path.write_text("{}", encoding="utf-8")
    manifest.update(
        {
            "integration_manifest": str(parent_path),
            "integration_manifest_sha256": crosscheck.sha256_file(parent_path),
            "source_judge_summary": str(summary_path),
            "source_judge_summary_sha256": crosscheck.sha256_file(
                summary_path
            ),
            "integration_identity": {},
            "source_hashes": crosscheck.source_hashes(),
            "previous_summary": None,
        }
    )
    monkeypatch.setattr(
        crosscheck.integration,
        "validate_manifest",
        lambda parent, require_ready: None,
    )
    with pytest.raises(RuntimeError, match="split differs"):
        crosscheck.runtime_identity(manifest)


def test_escalation_uses_only_applicable_unresolved_gates():
    resolved = {
        "mc_judge": {"classification": "practical_tie"},
        "mc_raw_loss": {"classification": "no_material_loss"},
        "comparison_penalty_active": False,
    }
    assert not crosscheck.comparison_needs_escalation(resolved)
    score = copy.deepcopy(resolved)
    score["mc_judge"]["classification"] = "unresolved"
    assert crosscheck.comparison_needs_escalation(score)
    raw = copy.deepcopy(resolved)
    raw["mc_raw_loss"]["classification"] = "unresolved"
    assert crosscheck.comparison_needs_escalation(raw)
    raw["comparison_penalty_active"] = True
    assert not crosscheck.comparison_needs_escalation(raw)


def test_completion_is_hash_and_manifest_bound(tmp_path):
    manifest = minimal_manifest()
    union = manifest["unions"][0]
    paths = crosscheck._paths(tmp_path, union["state_id"])
    crosscheck.write_json(paths["failure"], {"message": "expected"})
    completion = {
        "schema_version": crosscheck.SCHEMA_VERSION,
        "status": "failed",
        "manifest_sha256": crosscheck.manifest_digest(manifest),
        "state_id": union["state_id"],
        "hashes": {"failure": crosscheck.sha256_file(paths["failure"])},
    }
    crosscheck.write_json(paths["complete"], completion)
    assert (
        crosscheck.validate_cell(manifest, tmp_path, union)["status"]
        == "failed"
    )
    paths["failure"].write_text(
        json.dumps({"changed": True}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="changed"):
        crosscheck.validate_cell(manifest, tmp_path, union)
