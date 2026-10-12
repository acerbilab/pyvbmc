"""Protocol checks for the E2 integration runner; no VBMC fits."""

import json

import noisy_acq_integration as runner
import numpy as np
import pytest


def test_named_seed_derivation_is_stable_and_role_separated():
    first = runner.derive_seed(1, "state", "mc", 128, 0, "rule")
    assert first == runner.derive_seed(1, "state", "mc", 128, 0, "rule")
    assert first != runner.derive_seed(1, "state", "mc", 128, 0, "judge")
    assert first != runner.derive_seed(1, "state", "mc", 128, 1, "rule")
    assert 0 <= first < 2**64


def test_panel_contains_baseline_top_and_seeded_unique_remainder():
    candidates = np.arange(8192 * 2, dtype=np.float64).reshape(8192, 2)
    baseline = np.linspace(1.0, -1.0, 8192)
    state = {
        "cand": {
            "sieve_Xs": candidates,
            "n_repeat_candidates": 0,
        },
        "ref": {"acq": baseline},
    }
    descriptor = {"panel_seed": 123}
    panel, indices, scores, order = runner.candidate_panel(
        state, descriptor, "panel"
    )
    assert panel.shape == (runner.PANEL_SIZE, 2)
    assert len(np.unique(indices)) == runner.PANEL_SIZE
    np.testing.assert_array_equal(
        indices[: runner.PANEL_BASELINE_TOP],
        order[: runner.PANEL_BASELINE_TOP],
    )
    np.testing.assert_array_equal(panel, candidates[indices])
    np.testing.assert_array_equal(scores, baseline[indices])
    second = runner.candidate_panel(state, descriptor, "panel")
    np.testing.assert_array_equal(indices, second[1])


def test_panel_slices_prepended_repeat_reference_rows():
    candidates = np.arange(8192, dtype=np.float64)[:, None]
    state = {
        "cand": {
            "sieve_Xs": candidates,
            "n_repeat_candidates": 3,
        },
        "ref": {
            "acq": np.concatenate(([99.0, 98.0, 97.0], -candidates[:, 0]))
        },
    }
    full, indices, baseline, order = runner.candidate_panel(
        state, {"panel_seed": 1}, "full"
    )
    np.testing.assert_array_equal(full, candidates)
    np.testing.assert_array_equal(indices, np.arange(8192))
    np.testing.assert_array_equal(baseline, -candidates[:, 0])
    assert order[0] == 8191


@pytest.mark.parametrize(
    "selection,match",
    [
        (
            {"frozen": False, "settings": [{"method": "mc", "budget": 128}]},
            "frozen",
        ),
        ({"frozen": True, "settings": []}, "between one and 2"),
        (
            {
                "frozen": True,
                "settings": [
                    {"method": "mc", "budget": 128},
                    {"method": "mc", "budget": 512},
                ],
            },
            "one setting per method",
        ),
        (
            {
                "frozen": True,
                "settings": [{"method": "mc", "budget": 999}],
            },
            "untested",
        ),
    ],
)
def test_selection_lock_rejects_unfrozen_or_expanded_choices(
    tmp_path, selection, match
):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(RuntimeError, match=match):
        runner._read_selection(path, "holdout")


def test_selection_lock_accepts_at_most_one_budget_per_method(tmp_path):
    selection = {
        "frozen": True,
        "settings": [
            {"method": "mc", "budget": 512},
            {"method": "stratified_rqmc", "budget": 128},
        ],
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    settings, decoded = runner._read_selection(path, "holdout")
    assert settings == selection["settings"]
    assert decoded == selection


def test_full_selection_accepts_two_settings_per_method(tmp_path):
    selection = {
        "frozen": True,
        "settings": [
            {"method": "mc", "budget": 128},
            {"method": "mc", "budget": 512},
            {"method": "stratified_mc", "budget": 128},
            {"method": "stratified_mc", "budget": 512},
            {"method": "stratified_rqmc", "budget": 128},
            {"method": "stratified_rqmc", "budget": 512},
        ],
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    settings, _ = runner._read_selection(path, "full")
    assert settings == selection["settings"]


def test_full_selection_rejects_third_setting_for_method(tmp_path):
    selection = {
        "frozen": True,
        "settings": [
            {"method": "mc", "budget": 128},
            {"method": "mc", "budget": 512},
            {"method": "mc", "budget": 2048},
        ],
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(RuntimeError, match="at most two"):
        runner._read_selection(path, "full")


def minimal_manifest(stage="panel"):
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "kind": "integration",
        "launch_ready": True,
        "stage": stage,
        "holdout_locked": stage != "holdout",
        "replicates": runner.REPLICATES,
        "cells": [
            {
                "state_id": "x_seed0_early",
                "method": "mc",
                "budget": 128,
                "replicate": 0,
                "rule_seed": 1,
            }
        ],
        "cell_count": 1,
        "baseline_cells": [
            {
                "state_id": "x_seed0_early",
                "replicate": 0,
                "baseline_seed": 2,
            }
        ],
        "baseline_cell_count": 1,
    }


def test_holdout_cannot_launch_while_lock_is_present():
    manifest = minimal_manifest("holdout")
    manifest["holdout_locked"] = True
    with pytest.raises(RuntimeError, match="locked"):
        runner.validate_manifest(manifest, require_ready=True)
    manifest["holdout_locked"] = False
    runner.validate_manifest(manifest, require_ready=True)


def test_manifest_rejects_duplicate_cells_and_unreviewed_launch():
    manifest = minimal_manifest()
    manifest["launch_ready"] = False
    with pytest.raises(RuntimeError, match="launch_ready"):
        runner.validate_manifest(manifest, require_ready=True)
    manifest["launch_ready"] = True
    manifest["cells"] *= 2
    manifest["cell_count"] = 2
    with pytest.raises(RuntimeError, match="duplicated"):
        runner.validate_manifest(manifest, require_ready=False)


def test_cell_completion_is_hash_and_manifest_bound(tmp_path):
    manifest = minimal_manifest()
    cell = manifest["cells"][0]
    tag = runner.cell_tag(cell)
    paths = runner._cell_paths(tmp_path, tag)
    runner.write_json(paths["failure"], {"message": "expected test failure"})
    record = {
        "schema_version": runner.SCHEMA_VERSION,
        "status": "failed",
        "manifest_sha256": runner.manifest_digest(manifest),
        "cell": cell,
        "hashes": {"failure": runner.sha256_file(paths["failure"])},
    }
    runner.write_json(paths["complete"], record)
    assert runner.validate_cell(manifest, tmp_path, cell)["status"] == "failed"
    paths["failure"].write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        runner.validate_cell(manifest, tmp_path, cell)


def test_baseline_completion_is_hash_and_manifest_bound(tmp_path):
    manifest = minimal_manifest()
    cell = manifest["baseline_cells"][0]
    tag = runner.baseline_tag(cell)
    paths = runner._baseline_paths(tmp_path, tag)
    runner.write_json(paths["failure"], {"message": "expected test failure"})
    record = {
        "schema_version": runner.SCHEMA_VERSION,
        "status": "failed",
        "manifest_sha256": runner.manifest_digest(manifest),
        "cell": cell,
        "hashes": {"failure": runner.sha256_file(paths["failure"])},
    }
    runner.write_json(paths["complete"], record)
    assert (
        runner.validate_baseline_cell(manifest, tmp_path, cell)["status"]
        == "failed"
    )
    paths["failure"].write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        runner.validate_baseline_cell(manifest, tmp_path, cell)


def test_escalation_keeps_either_unresolved_gate():
    resolved = {
        "judge": {"classification": "practical_tie"},
        "raw_loss": {"classification": "no_material_loss"},
        "comparison_penalty_active": False,
    }
    assert not runner.comparison_needs_escalation(resolved)
    unresolved_score = json.loads(json.dumps(resolved))
    unresolved_score["judge"]["classification"] = "unresolved"
    assert runner.comparison_needs_escalation(unresolved_score)
    unresolved_loss = json.loads(json.dumps(resolved))
    unresolved_loss["raw_loss"]["classification"] = "unresolved"
    assert runner.comparison_needs_escalation(unresolved_loss)
    unresolved_loss["comparison_penalty_active"] = True
    assert not runner.comparison_needs_escalation(unresolved_loss)


def test_judge_manifest_rejects_locked_holdout():
    manifest = {
        "schema_version": runner.SCHEMA_VERSION,
        "kind": "judge",
        "launch_ready": True,
        "split": "holdout",
        "holdout_locked": True,
        "judge_budget": runner.JUDGE_BUDGETS[0],
        "replicates": runner.REPLICATES,
        "unions": [],
        "cell_count": 0,
    }
    with pytest.raises(RuntimeError, match="locked"):
        runner.validate_judge_manifest(manifest, require_ready=True)
    manifest["holdout_locked"] = False
    runner.validate_judge_manifest(manifest, require_ready=True)


def test_judge_completion_is_hash_and_manifest_bound(tmp_path):
    union = {
        "state_id": "x_seed0_early",
        "candidate_indices": [1],
        "candidate_union_sha256": runner.capture.digest(
            ("x_seed0_early", np.asarray([1], dtype=np.int64))
        ),
        "union_origin_budget": runner.JUDGE_BUDGETS[0],
        "fixed_baseline_global_index": 1,
        "comparison_seeds": list(range(runner.REPLICATES)),
        "band_pilot_seeds": list(range(100, 100 + runner.REPLICATES)),
    }
    manifest = {
        "schema_version": runner.SCHEMA_VERSION,
        "kind": "judge",
        "launch_ready": True,
        "split": "development",
        "holdout_locked": True,
        "judge_budget": runner.JUDGE_BUDGETS[0],
        "replicates": runner.REPLICATES,
        "unions": [union],
        "cell_count": 1,
    }
    runner.validate_judge_manifest(manifest, require_ready=True)
    paths = runner._judge_paths(tmp_path, union["state_id"])
    runner.write_json(paths["failure"], {"message": "expected test failure"})
    completion = {
        "schema_version": runner.SCHEMA_VERSION,
        "status": "failed",
        "manifest_sha256": runner.manifest_digest(manifest),
        "state_id": union["state_id"],
        "hashes": {"failure": runner.sha256_file(paths["failure"])},
    }
    runner.write_json(paths["complete"], completion)
    assert (
        runner.validate_judge_cell(manifest, tmp_path, union)["status"]
        == "failed"
    )
    paths["failure"].write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        runner.validate_judge_cell(manifest, tmp_path, union)


def test_grouped_summary_keeps_missing_cells_in_every_denominator():
    integration = {
        "settings": [{"method": "mc", "budget": 128}],
        "states": [
            {
                "state_id": "rosenbrock_D2_noise1_seed0_early",
            }
        ],
        "missing_states": [
            {
                "state_id": "rosenbrock_D2_noise1_seed0_late",
                "reason": "checkpoint_missing",
            }
        ],
        "expected_state_count": 2,
        "cells": [
            {
                "state_id": "rosenbrock_D2_noise1_seed0_early",
                "method": "mc",
                "budget": 128,
                "replicate": replicate,
            }
            for replicate in range(runner.REPLICATES)
        ],
    }
    comparison = {
        "cell": {
            "state_id": "rosenbrock_D2_noise1_seed0_early",
            "method": "mc",
            "budget": 128,
            "replicate": 0,
        },
        "judge": {"classification": "harmful"},
        "raw_loss": {"classification": "material_loss"},
        "comparison_penalty_active": False,
    }
    grouped = runner.grouped_judge_summary(integration, [comparison])
    setting = grouped["by_setting"][0]
    assert setting["scheduled"] == 2 * runner.REPLICATES
    assert setting["observed"] == 1
    assert setting["score_counts"] == {
        "beneficial": 0,
        "harmful": 1,
        "practical_tie": 0,
        "unresolved": 0,
        "failed_or_missing": 15,
    }
    assert setting["raw_loss_gate_counts"]["material_loss"] == 1
    assert setting["fractions_of_scheduled"]["harmful"] == 1 / 16
    trajectory = grouped["by_trajectory"][0]
    assert trajectory["trajectory"] == "rosenbrock_D2_noise1_seed0"
    assert trajectory["scheduled"] == 16
    checkpoints = {
        item["checkpoint"]: item for item in grouped["by_checkpoint"]
    }
    assert checkpoints["early"]["scheduled"] == 8
    assert checkpoints["early"]["score_counts"]["harmful"] == 1
    assert checkpoints["late"]["scheduled"] == 8
    assert checkpoints["late"]["score_counts"]["failed_or_missing"] == 8
