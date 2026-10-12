"""Protocol checks for the E3 selected-coordinate MC crosscheck."""

import copy
import json

import noisy_acq_search_crosscheck as crosscheck
import numpy as np
import pytest


def comparison(
    tag,
    state,
    mean,
    *,
    eps=0.1,
    selected="arm",
    baseline="baseline",
    raw="no_material_loss",
):
    return {
        "comparison_tag": tag,
        "cell": {"state_id": state},
        "selected_row_id": selected,
        "baseline_row_id": baseline,
        "band": {"eps_F": eps, "pilot": "frozen"},
        "judge": {
            "mean_difference": mean,
            "classification": "harmful" if mean > eps else "beneficial",
        },
        "raw_loss": {"classification": raw},
    }


def test_development_policy_selects_extrema_threshold_and_raw_loss():
    items = [
        comparison("low", "s1", -0.2),
        comparison("middle", "s1", 0.0),
        comparison("high", "s1", 0.3),
        comparison("far", "s1", 1.1),
        comparison("raw", "s1", 0.1, raw="material_loss"),
        comparison("only", "s2", 0.05),
    ]
    tags, reasons = crosscheck.select_development_tags(items)
    assert tags == ["far", "low", "only", "raw"]
    assert "state_min_mean_difference" in reasons["low"]
    assert "state_max_mean_difference" in reasons["far"]
    assert "ten_practical_bands" in reasons["far"]
    assert "confident_raw_material_loss" in reasons["raw"]


def test_frozen_development_tags_cannot_substitute_another_valid_source():
    items = [
        comparison("low", "s", -1.0),
        comparison("middle", "s", 0.0),
        comparison("high", "s", 1.0),
    ]
    source = {item["comparison_tag"]: item for item in items}
    manifest = {
        "split": "development",
        "frozen_comparison_tags": ["middle"],
        "explicit_comparison_tags": [],
        "unions": [],
        "carried_comparisons": [],
        "carried_failed_pairs": [],
    }
    with pytest.raises(RuntimeError, match="frozen source selection changed"):
        crosscheck._validate_frozen_selection(manifest, source)


def test_coordinate_pair_deduplication_retains_provenance(monkeypatch):
    candidates = np.array([[1.0, 2.0], [3.0, 4.0]])
    row_ids = [crosscheck.search_experiment._row_id(row) for row in candidates]
    union = {"candidate_row_ids": row_ids}
    artifact = {"state_id": "s"}
    monkeypatch.setattr(
        crosscheck,
        "_load_initial_union",
        lambda *args: (union, candidates.copy(), artifact),
    )
    first = comparison(
        "a", "s", -1.0, selected=row_ids[0], baseline=row_ids[1]
    )
    second = comparison(
        "b", "s", 1.0, selected=row_ids[0], baseline=row_ids[1]
    )
    pairs, artifacts = crosscheck._deduplicate_coordinate_pairs(
        [first, second], {"a": ["low"], "b": ["high"]}, {}, None
    )
    assert len(pairs) == 1
    assert pairs[0]["comparison_tags"] == ["a", "b"]
    assert pairs[0]["selection_reasons"] == ["high", "low"]
    assert pairs[0]["arm"] == [1.0, 2.0]
    assert artifacts == {"s": artifact}


def test_union_binding_rejects_coherent_coordinate_and_snapshot_substitution():
    candidates = np.array([[1.0, 2.0], [3.0, 4.0]])
    row_ids = [crosscheck.search_experiment._row_id(row) for row in candidates]
    source = comparison(
        "tag", "s", 0.2, selected=row_ids[0], baseline=row_ids[1]
    )
    pair = _pair()
    pair.update(
        {
            "band": copy.deepcopy(source["band"]),
            "source_comparisons": [copy.deepcopy(source)],
        }
    )
    descriptor = {"state_id": "s", "snapshot": "frozen"}
    union = crosscheck._build_union(
        "s", [pair], descriptor, {"state_id": "s"}, 32768
    )
    initial = {"candidate_row_ids": row_ids}
    sources = {"tag": source}
    crosscheck._validate_union_source_binding(
        union, descriptor, initial, candidates, sources
    )

    changed = copy.deepcopy(union)
    changed_arm = np.array([9.0, 2.0])
    changed_id = crosscheck.search_experiment._row_id(changed_arm)
    changed_pair = changed["pairs"][0]
    changed_pair["arm"] = changed_arm.tolist()
    changed_pair["arm_row_id"] = changed_id
    changed_pair["pair_id"] = crosscheck._pair_id(
        "s", changed_arm, np.asarray(changed_pair["baseline"])
    )
    changed_pair["source_comparisons"][0]["selected_row_id"] = changed_id
    changed["candidate_row_ids"][0] = changed_id
    changed["candidates"][0] = changed_arm.tolist()
    with pytest.raises(RuntimeError, match="source comparison changed"):
        crosscheck._validate_union_source_binding(
            changed, descriptor, initial, candidates, sources
        )

    changed = copy.deepcopy(union)
    changed["state"] = {"state_id": "s", "snapshot": "substituted"}
    with pytest.raises(RuntimeError, match="state descriptor changed"):
        crosscheck._validate_union_source_binding(
            changed, descriptor, initial, candidates, sources
        )


def _pair(state="s", tags=None):
    arm = np.array([1.0, 2.0])
    baseline = np.array([3.0, 4.0])
    return {
        "pair_id": crosscheck._pair_id(state, arm, baseline),
        "state_id": state,
        "arm_row_id": crosscheck.search_experiment._row_id(arm),
        "baseline_row_id": crosscheck.search_experiment._row_id(baseline),
        "arm": arm.tolist(),
        "baseline": baseline.tolist(),
        "comparison_tags": ["tag"] if tags is None else tags,
        "selection_reasons": ["test"],
    }


def minimal_manifest(split="development"):
    pair = _pair()
    candidates = np.asarray([pair["arm"], pair["baseline"]])
    row_ids = [pair["arm_row_id"], pair["baseline_row_id"]]
    artifact = {"state_id": "s"}
    tags = ["tag"]
    explicit = tags if split == "holdout" else []
    union = {
        "state_id": "s",
        "candidate_row_ids": row_ids,
        "candidates": candidates.tolist(),
        "candidate_union_sha256": crosscheck.hashlib.sha256(
            np.ascontiguousarray(candidates, dtype="<f8").tobytes()
        ).hexdigest(),
        "pairs": [pair],
        "initial_judge_artifact": artifact,
        "replicate_seeds": [
            crosscheck.integration.derive_seed(
                crosscheck.MASTER_SEED,
                "s",
                crosscheck.BUDGETS[0],
                replicate,
                "search_ordinary_mc_crosscheck",
            )
            for replicate in range(crosscheck.REPLICATES)
        ],
    }
    return {
        "schema_version": crosscheck.SCHEMA_VERSION,
        "kind": "search_ordinary_mc_crosscheck",
        "source_judge_summary_sha256": "frozen-judge",
        "search_manifest_sha256": "frozen-search",
        "launch_ready": True,
        "split": split,
        "holdout_locked": split == "holdout",
        "budget": crosscheck.BUDGETS[0],
        "replicates": crosscheck.REPLICATES,
        "master_seed": crosscheck.MASTER_SEED,
        "selection_policy": (
            "explicit_holdout_only"
            if split == "holdout"
            else "development_extrema_and_catastrophes"
        ),
        "frozen_comparison_tags": tags,
        "explicit_comparison_tags": explicit,
        "initial_judge_artifacts": {"s": artifact},
        "unions": [union],
        "state_cell_count": 1,
        "scheduled_pair_count": 1,
        "scheduled_pairs_by_state": {"s": 1},
        "carried_comparisons": [],
        "carried_failed_pairs": [],
        "carried_failed_states": [],
    }


def test_manifest_locks_holdout_and_coordinate_identity():
    manifest = minimal_manifest("holdout")
    with pytest.raises(RuntimeError, match="locked"):
        crosscheck.validate_manifest(manifest, require_ready=True)
    manifest["holdout_locked"] = False
    crosscheck.validate_manifest(manifest, require_ready=True)
    manifest["unions"][0]["pairs"][0]["arm"] = [9.0, 2.0]
    with pytest.raises(RuntimeError, match="coordinate identity"):
        crosscheck.validate_manifest(manifest, require_ready=False)


def test_manifest_rejects_duplicate_state_workers():
    manifest = minimal_manifest()
    manifest["unions"].append(copy.deepcopy(manifest["unions"][0]))
    manifest["state_cell_count"] = 2
    with pytest.raises(RuntimeError, match="state cells are duplicated"):
        crosscheck.validate_manifest(manifest, require_ready=False)


def test_manifest_rejects_explicit_development_expansion():
    manifest = minimal_manifest()
    manifest["explicit_comparison_tags"] = ["post_outcome_addition"]
    with pytest.raises(RuntimeError, match="development.*explicit"):
        crosscheck.validate_manifest(manifest, require_ready=False)


def test_prepared_holdout_cannot_mutate_split_to_bypass_lock(
    tmp_path, monkeypatch
):
    manifest = minimal_manifest("holdout")
    manifest["split"] = "development"
    manifest["selection_policy"] = "development_extrema_and_catastrophes"
    manifest["explicit_comparison_tags"] = []
    crosscheck.validate_manifest(manifest, require_ready=True)

    search_manifest = {"split": "holdout"}
    search_path = tmp_path / "search.json"
    summary_path = tmp_path / "summary.json"
    search_path.write_text(json.dumps(search_manifest), encoding="utf-8")
    summary_path.write_text("{}", encoding="utf-8")
    manifest.update(
        {
            "search_manifest": str(search_path),
            "search_manifest_sha256": crosscheck.sha256_file(search_path),
            "search_manifest_digest": crosscheck.manifest_digest(
                search_manifest
            ),
            "source_judge_summary": str(summary_path),
            "source_judge_summary_sha256": crosscheck.sha256_file(
                summary_path
            ),
            "search_identity": {},
            "source_hashes": crosscheck.source_hashes(),
            "search_output": str(tmp_path),
            "previous_summary": None,
        }
    )
    monkeypatch.setattr(
        crosscheck.search_experiment, "runtime_identity", lambda source: {}
    )
    monkeypatch.setattr(
        crosscheck.search_experiment,
        "validate_manifest",
        lambda source, require_ready: None,
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


def test_escalated_partition_is_exactly_bound_to_previous_summary():
    resolved = {
        "pair_id": "resolved",
        "mc_judge": {"classification": "practical_tie"},
        "mc_raw_loss": {"classification": "no_material_loss"},
        "comparison_penalty_active": False,
    }
    pending = copy.deepcopy(resolved)
    pending["pair_id"] = "pending"
    pending["mc_judge"]["classification"] = "unresolved"
    failed = {"pair_id": "failed"}
    prior = {
        "comparisons": [resolved, pending],
        "failed_pairs": [failed],
        "failed_states": ["failed_state"],
    }
    manifest = {
        "unions": [{"pairs": [{"pair_id": "pending"}]}],
        "carried_comparisons": [copy.deepcopy(resolved)],
        "carried_failed_pairs": [copy.deepcopy(failed)],
        "carried_failed_states": ["failed_state"],
    }
    crosscheck._validate_previous_partition(manifest, prior)
    manifest["carried_comparisons"][0]["mc_judge"][
        "classification"
    ] = "harmful"
    with pytest.raises(RuntimeError, match="carried_comparisons changed"):
        crosscheck._validate_previous_partition(manifest, prior)


def test_completion_is_hash_and_manifest_bound(tmp_path):
    manifest = minimal_manifest()
    union = manifest["unions"][0]
    paths = crosscheck._paths(tmp_path, union["state_id"])
    crosscheck.write_json(paths["failure"], {"message": "expected"})
    crosscheck.write_json(
        paths["complete"],
        {
            "schema_version": crosscheck.SCHEMA_VERSION,
            "status": "failed",
            "manifest_sha256": crosscheck.manifest_digest(manifest),
            "state_id": union["state_id"],
            "hashes": {"failure": crosscheck.sha256_file(paths["failure"])},
        },
    )
    assert (
        crosscheck.validate_cell(manifest, tmp_path, union)["status"]
        == "failed"
    )
    paths["failure"].write_text(
        json.dumps({"changed": True}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="changed"):
        crosscheck.validate_cell(manifest, tmp_path, union)


def test_failed_pair_is_retained_in_denominator(tmp_path, monkeypatch):
    manifest = minimal_manifest()
    pair = manifest["unions"][0]["pairs"][0]
    paths = crosscheck._paths(tmp_path, "s")
    crosscheck.write_json(paths["failure"], {"message": "legitimate"})
    crosscheck.write_json(
        paths["complete"],
        {
            "schema_version": crosscheck.SCHEMA_VERSION,
            "status": "failed",
            "manifest_sha256": crosscheck.manifest_digest(manifest),
            "state_id": "s",
            "hashes": {"failure": crosscheck.sha256_file(paths["failure"])},
        },
    )
    monkeypatch.setattr(crosscheck, "runtime_identity", lambda manifest: {})
    summary = crosscheck.summarize(manifest, tmp_path)
    assert summary["scheduled_pair_count"] == 1
    assert summary["counts"]["observed"] == 0
    assert summary["counts"]["score"]["failed_or_missing"] == 1
    assert summary["failed_pairs"][0]["pair_id"] == pair["pair_id"]


def test_second_budget_cannot_expand_frozen_tag_set(tmp_path, monkeypatch):
    source = {
        "kind": "search_judge_summary",
        "source_identity_sha256": "identity",
        "comparisons": [comparison("tag", "s", 0.0)],
    }
    search_manifest = {
        "split": "development",
        "identity": {},
        "states": [{"state_id": "s"}],
    }
    source["search_manifest_sha256"] = crosscheck.manifest_digest(
        search_manifest
    )
    source_path = tmp_path / "judge.json"
    manifest_path = tmp_path / "search.json"
    previous_path = tmp_path / "previous.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    manifest_path.write_text(json.dumps(search_manifest), encoding="utf-8")
    previous_path.write_text(
        json.dumps(
            {
                "budget": crosscheck.BUDGETS[0],
                "source_judge_summary_sha256": crosscheck.sha256_file(
                    source_path
                ),
                "frozen_comparison_tags": ["expanded"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        crosscheck.search_experiment,
        "validate_manifest",
        lambda manifest, require_ready: None,
    )
    monkeypatch.setattr(
        crosscheck.search_experiment, "runtime_identity", lambda manifest: {}
    )
    monkeypatch.setattr(crosscheck.capture, "digest", lambda value: "identity")
    monkeypatch.setattr(
        crosscheck,
        "_deduplicate_coordinate_pairs",
        lambda *args: ([_pair()], {"s": {"state_id": "s"}}),
    )
    with pytest.raises(RuntimeError, match="reuse the frozen"):
        crosscheck.prepare_manifest(
            source_path,
            manifest_path,
            tmp_path,
            crosscheck.BUDGETS[1],
            previous_path,
            [],
        )
