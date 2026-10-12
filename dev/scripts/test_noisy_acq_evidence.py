"""Synthetic checks for the E2 evidence reducer."""

import copy

import noisy_acq_evidence as evidence
import pytest


def _comparison(
    state_id,
    replicate,
    mean,
    classification,
    raw_classification,
    *,
    penalty=False,
):
    return {
        "cell": {
            "state_id": state_id,
            "method": "stratified_rqmc",
            "budget": 256,
            "replicate": replicate,
        },
        "judge": {
            "mean_difference": mean,
            "ci": [mean - 0.1, mean + 0.1],
            "classification": classification,
        },
        "band": {"eps_F": 0.05},
        "raw_loss": {"classification": raw_classification},
        "comparison_penalty_active": penalty,
    }


def _count_row(**extra):
    return {
        "method": "stratified_rqmc",
        "budget": 256,
        "scheduled": 24,
        "observed": 3,
        "score_counts": {
            "beneficial": 1,
            "harmful": 1,
            "practical_tie": 0,
            "unresolved": 1,
            "failed_or_missing": 21,
        },
        "raw_loss_gate_counts": {
            "material_loss": 1,
            "no_material_loss": 1,
            "unresolved": 0,
            "not_applicable_penalty_active": 1,
            "failed_or_missing": 21,
        },
        **extra,
    }


def _trajectory_row(trajectory, score_class, raw_class, *, penalty=False):
    score = {
        key: 0
        for key in ("beneficial", "harmful", "practical_tie", "unresolved")
    }
    score[score_class] = 1
    score["failed_or_missing"] = 7
    raw = {
        key: 0
        for key in (
            "material_loss",
            "no_material_loss",
            "unresolved",
            "not_applicable_penalty_active",
        )
    }
    raw["not_applicable_penalty_active" if penalty else raw_class] = 1
    raw["failed_or_missing"] = 7
    return {
        "method": "stratified_rqmc",
        "budget": 256,
        "trajectory": trajectory,
        "scheduled": 8,
        "observed": 1,
        "score_counts": score,
        "raw_loss_gate_counts": raw,
    }


def _judge():
    comparisons = [
        _comparison(
            "target_seed0_early",
            0,
            -0.4,
            "beneficial",
            "no_material_loss",
        ),
        _comparison(
            "target_seed1_late",
            0,
            0.7,
            "harmful",
            "material_loss",
        ),
        _comparison(
            "other_seed0_early",
            0,
            0.2,
            "unresolved",
            "material_loss",
            penalty=True,
        ),
    ]
    trajectories = [
        _trajectory_row("target_seed0", "beneficial", "no_material_loss"),
        _trajectory_row("target_seed1", "harmful", "material_loss"),
        _trajectory_row(
            "other_seed0", "unresolved", "material_loss", penalty=True
        ),
    ]
    return {
        "judge_budget": 16384,
        "integration_manifest_sha256": "integration",
        "judge_manifest_sha256": "judge-manifest",
        "grouped": {
            "by_setting": [_count_row()],
            "by_trajectory": trajectories,
        },
        "comparisons": comparisons,
    }


def _timing():
    def succeeded(state, ratio, low, high):
        return {
            "state_id": state,
            "method": "stratified_rqmc",
            "budget": 256,
            "status": "succeeded",
            "identity": {
                "integration_manifest_sha256": "integration",
                "timing_source_sha256": "timing-source",
                "setting": {"method": "stratified_rqmc", "budget": 256},
            },
            "result": {
                "median_paired_ratio": ratio,
                "paired_ratio_range": [low, high],
            },
        }

    return {
        "mode": "timing",
        "scheduled": 3,
        "rows": [
            succeeded("target_seed0_early", 0.5, 0.4, 0.7),
            succeeded("target_seed1_late", 1.5, 1.2, 1.8),
            {
                "state_id": "other_seed0_early",
                "method": "stratified_rqmc",
                "budget": 256,
                "status": "failed",
                "identity": {
                    "integration_manifest_sha256": "integration",
                    "timing_source_sha256": "timing-source",
                    "setting": {
                        "method": "stratified_rqmc",
                        "budget": 256,
                    },
                },
                "result": {"type": "RuntimeError", "message": "expected"},
            },
        ],
    }


def _crosscheck():
    return {
        "source_judge_summary_sha256": "judge-file",
        "manifest_sha256": "crosscheck-manifest",
        "budget": 65536,
        "scheduled_pair_count": 2,
        "status_counts": {
            "agree": 0,
            "disagree": 1,
            "unresolved": 0,
            "failed_or_missing": 1,
        },
        "raw_status_counts": {
            "agree": 0,
            "disagree": 0,
            "unresolved": 1,
            "not_applicable_penalty_active": 0,
            "failed_or_missing": 1,
        },
        "comparisons": [
            {
                "pair_id": "pair",
                "state_id": "target_seed1_late",
                "comparison_tags": ["tag"],
                "cross_method_status": "disagree",
                "raw_cross_method_status": "unresolved",
            }
        ],
        "failed_states": ["other_seed0_early"],
        "automatic_override": False,
    }


def test_reducer_preserves_denominators_uncertainty_and_worst_cells():
    report = evidence.reduce_evidence(
        _judge(),
        judge_source={"path": "judge.json", "sha256": "judge-file"},
        timing=_timing(),
        timing_source={"path": "timing.json", "sha256": "timing-file"},
    )
    setting = report["settings"][0]
    assert not report["automatic_promotion"]
    assert setting["scheduled"] == 24
    assert setting["score_counts"]["unresolved"] == 1
    assert setting["fractions_of_scheduled"]["failed_or_missing"] == 21 / 24
    assert setting["raw_loss_gate"]["applicable_observed"] == 2
    assert setting["raw_loss_gate"]["not_applicable_penalty_active"] == 1
    assert len(setting["trajectory_counts"]) == 3
    assert setting["worst_by_target"][1]["mean_F_difference"] == 0.7
    worst = setting["top_worst_5_F_differences"]
    assert worst[0]["mean_F_difference"] == 0.7
    assert worst[0]["raw_material_loss"]
    assert (
        setting["material_raw_loss_cells"][0]["state_id"]
        == "target_seed1_late"
    )
    assert (
        setting["material_raw_loss_cells"][0]["guarded_arm_to_baseline_ratio"]
        is None
    )
    unresolved = next(
        item for item in worst if item["F_classification"] == "unresolved"
    )
    assert (
        unresolved["raw_loss_classification"]
        == "not_applicable_penalty_active"
    )


def test_timing_uses_median_of_state_medians_and_retains_failure():
    report = evidence.reduce_evidence(
        _judge(),
        judge_source={"path": "judge.json", "sha256": "judge-file"},
        timing=_timing(),
        timing_source={"path": "timing.json", "sha256": "timing-file"},
    )
    timing = report["settings"][0]["timing"]
    assert timing["median_of_state_median_paired_ratios"] == 1.0
    assert timing["state_median_ratio_range"] == [0.5, 1.5]
    assert timing["failed"] == 1
    assert timing["missing"] == 0
    assert len(timing["state_ratios"]) == 2
    assert timing["state_ratios"][0]["paired_ratio_range"] == [0.4, 0.7]


def test_mc_crosscheck_stays_separate_and_cannot_override():
    report = evidence.reduce_evidence(
        _judge(),
        judge_source={"path": "judge.json", "sha256": "judge-file"},
        crosscheck=_crosscheck(),
        crosscheck_source={"path": "mc.json", "sha256": "mc-file"},
    )
    diagnostic = report["mc_crosscheck"]
    assert diagnostic["score_status_counts"]["disagree"] == 1
    assert diagnostic["raw_status_counts"]["unresolved"] == 1
    assert diagnostic["disagreements"][0]["pair_id"] == "pair"
    assert not diagnostic["automatic_override"]


def test_reducer_rejects_changed_scheduled_counts():
    judge = _judge()
    judge["grouped"]["by_setting"][0]["score_counts"]["unresolved"] = 0
    with pytest.raises(RuntimeError, match="scheduled denominator"):
        evidence.reduce_evidence(
            judge,
            judge_source={"path": "judge.json", "sha256": "judge-file"},
        )


def test_timing_preserves_unrepresented_missing_states():
    timing = _timing()
    timing["rows"] = timing["rows"][:1]
    report = evidence.reduce_evidence(
        _judge(),
        judge_source={"path": "judge.json", "sha256": "judge-file"},
        timing=timing,
        timing_source={"path": "timing.json", "sha256": "timing-file"},
    )
    reduced = report["settings"][0]["timing"]
    assert reduced["missing"] == 2
    assert reduced["unrepresented_missing"] == 2


def test_timing_rejects_cross_manifest():
    timing = _timing()
    timing["rows"][0]["identity"]["integration_manifest_sha256"] = "other"
    with pytest.raises(RuntimeError, match="another integration manifest"):
        evidence.reduce_evidence(
            _judge(),
            judge_source={"path": "judge.json", "sha256": "judge-file"},
            timing=timing,
            timing_source={"path": "timing.json", "sha256": "timing-file"},
        )


def test_timing_rejects_mixed_implementation_and_duplicate_cells():
    timing = _timing()
    timing["rows"][1]["identity"]["timing_source_sha256"] = "other-source"
    with pytest.raises(RuntimeError, match="mixed timing implementations"):
        evidence.reduce_evidence(
            _judge(),
            judge_source={"path": "judge.json", "sha256": "judge-file"},
            timing=timing,
            timing_source={"path": "timing.json", "sha256": "timing-file"},
        )
    timing = _timing()
    timing["rows"].append(copy.deepcopy(timing["rows"][0]))
    with pytest.raises(RuntimeError, match="duplicate state setting"):
        evidence.reduce_evidence(
            _judge(),
            judge_source={"path": "judge.json", "sha256": "judge-file"},
            timing=timing,
            timing_source={"path": "timing.json", "sha256": "timing-file"},
        )


def test_all_missing_timing_without_identity_fails_closed():
    timing = _timing()
    timing["rows"] = [
        {
            "state_id": "target_seed0_early",
            "method": "stratified_rqmc",
            "budget": 256,
            "status": "missing",
        }
    ]
    with pytest.raises(RuntimeError, match="no source identity"):
        evidence.reduce_evidence(
            _judge(),
            judge_source={"path": "judge.json", "sha256": "judge-file"},
            timing=timing,
            timing_source={"path": "timing.json", "sha256": "timing-file"},
        )


def test_crosscheck_count_inconsistency_is_rejected():
    crosscheck = copy.deepcopy(_crosscheck())
    crosscheck["status_counts"]["failed_or_missing"] = 0
    with pytest.raises(RuntimeError, match="denominator"):
        evidence.reduce_evidence(
            _judge(),
            judge_source={"path": "judge.json", "sha256": "judge-file"},
            crosscheck=crosscheck,
            crosscheck_source={"path": "mc.json", "sha256": "mc-file"},
        )
