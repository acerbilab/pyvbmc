"""Focused protocol checks for the E3 accepted-refinement diagnostic."""

import copy

import noisy_acq_search_loss_diagnostic as diagnostic
import numpy as np
import pytest


def _rqmc(tag, *, raw="material_loss", kind="primary_vs_s0"):
    return {
        "comparison_tag": tag,
        "cell": {
            "kind": kind,
            "arm_tag": "finalist",
            "reference_tag": "S0",
            "state_id": "state",
            "replicate": 0,
        },
        "raw_loss": {"classification": raw},
        "comparison_penalty_active": False,
    }


def _mc(tags, classification):
    return {
        "comparison_tags": tags,
        "mc_raw_loss": {"classification": classification},
        "comparison_penalty_active": False,
    }


def test_eligibility_keeps_only_material_losses_confirmed_or_unresolved_by_mc():
    rqmc = [
        _rqmc("confirmed"),
        _rqmc("unresolved"),
        _rqmc("rejected"),
        _rqmc("not_material", raw="no_material_loss"),
        _rqmc("component", kind="component_diagnostic"),
        {**_rqmc("penalized"), "comparison_penalty_active": True},
    ]
    mc = [
        _mc(["confirmed"], "material_loss"),
        _mc(["unresolved"], "unresolved"),
        _mc(["rejected"], "no_material_loss"),
        _mc(["not_material"], "material_loss"),
        _mc(["component"], "material_loss"),
        _mc(["penalized"], "material_loss"),
    ]
    original = copy.deepcopy((rqmc, mc))
    selected = diagnostic.eligible_comparisons(rqmc, mc)
    assert [item["comparison_tag"] for item in selected] == [
        "confirmed",
        "unresolved",
    ]
    assert [item["mc_eligibility"] for item in selected] == [
        "material_loss",
        "unresolved",
    ]
    assert (rqmc, mc) == original


def test_parent_schema_distinguishes_raw_search_file_hash_from_digest():
    raw_search_sha = "1" * 64
    semantic_search_digest = "2" * 64
    rqmc_file_sha = "3" * 64
    rqmc = {
        "kind": "search_judge_summary",
        "search_manifest_sha256": semantic_search_digest,
    }
    mc_summary = {
        "kind": "search_ordinary_mc_crosscheck_summary",
        "search_manifest_sha256": raw_search_sha,
        "source_judge_summary_sha256": rqmc_file_sha,
    }
    explicit = {
        "kind": "search_ordinary_mc_holdout_explicit_selection",
        "search_manifest_sha256": raw_search_sha,
        "source_judge_summary_sha256": rqmc_file_sha,
    }
    diagnostic.validate_parent_hash_contracts(
        rqmc,
        mc_summary,
        explicit,
        search_manifest_file_sha256=raw_search_sha,
        search_manifest_digest=semantic_search_digest,
        rqmc_summary_file_sha256=rqmc_file_sha,
    )
    changed = {**mc_summary, "search_manifest_sha256": semantic_search_digest}
    with pytest.raises(RuntimeError, match="MC parent identity"):
        diagnostic.validate_parent_hash_contracts(
            rqmc,
            changed,
            explicit,
            search_manifest_file_sha256=raw_search_sha,
            search_manifest_digest=semantic_search_digest,
            rqmc_summary_file_sha256=rqmc_file_sha,
        )


def test_reconstruct_own_winner_uses_first_argmin_and_rejects_repeats():
    candidates = np.array(
        [[-2.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        dtype=np.float64,
    )
    arrays = {
        "coarse_candidates": candidates,
        "shortlist_indices": np.array([3, 2, 4], dtype=np.int64),
        "shortlist_scores": np.array([1.0, 1.0, 2.0], dtype=np.float64),
        "selected": np.array([[3.25, 0.0]], dtype=np.float64),
    }
    details = {
        "generated_count": 3,
        "coarse_fallback_score": 1.0,
        "selected_accurate_score": 0.5,
        "fallback_reason": None,
        "selected_shortlist_index": None,
        "target_called": False,
    }
    assert np.array_equal(
        diagnostic.reconstruct_own_winner(arrays, details), candidates[3]
    )
    repeated = {**arrays, "shortlist_indices": np.array([1, 3, 4])}
    with pytest.raises(RuntimeError, match="substituted repeat"):
        diagnostic.reconstruct_own_winner(repeated, details)
    nonfinite = arrays.copy()
    nonfinite["coarse_candidates"] = candidates.copy()
    nonfinite["coarse_candidates"][3, 0] = np.inf
    with pytest.raises(RuntimeError, match="finite"):
        diagnostic.reconstruct_own_winner(nonfinite, details)


def test_build_union_has_three_roles_three_contrasts_and_fresh_seed_domain(
    monkeypatch,
):
    monkeypatch.setattr(
        diagnostic.search_experiment,
        "_state_descriptor",
        lambda manifest, state_id: {"state_id": state_id, "snapshot": "s"},
    )
    coordinates = {
        "S0": [0.0, 0.0],
        "own_winner": [1.0, 0.0],
        "selected": [2.0, 0.0],
    }
    row_ids = {
        role: diagnostic._row_id(np.asarray(row, dtype=np.float64))
        for role, row in coordinates.items()
    }
    case = {
        "comparison_tag": "case",
        "state_id": "state",
        "band": {"eps_F": 0.1},
        "coordinates": coordinates,
        "row_ids": row_ids,
    }
    union = diagnostic._build_unions(
        [case], {"states": []}, diagnostic.BUDGETS[0]
    )[0]
    assert union["candidate_row_ids"] == [
        row_ids["S0"],
        row_ids["own_winner"],
        row_ids["selected"],
    ]
    assert [item["name"] for item in union["contrasts"]] == [
        "selected_minus_own_winner",
        "own_winner_minus_S0",
        "selected_minus_S0",
    ]
    assert union["replicate_seeds"] == [
        diagnostic.integration.derive_seed(
            diagnostic.MASTER_SEED,
            "state",
            diagnostic.BUDGETS[0],
            replicate,
            diagnostic.SEED_DOMAIN,
        )
        for replicate in range(diagnostic.REPLICATES)
    ]


@pytest.mark.parametrize(
    ("selected_own", "own_s0", "selected_s0", "expected"),
    [
        (
            "harmful",
            "practical_tie",
            "harmful",
            "accepted_refinement_regression",
        ),
        ("harmful", "harmful", "harmful", "mixed_stage_regressions"),
        (
            "harmful",
            "unresolved",
            "harmful",
            "direct_regression_stage_unresolved",
        ),
        (
            "practical_tie",
            "harmful",
            "harmful",
            "loss_located_before_refinement",
        ),
        (
            "practical_tie",
            "practical_tie",
            "harmful",
            "distributed_threshold_accumulation",
        ),
        (
            "beneficial",
            "unresolved",
            "unresolved",
            "unresolved_selected_vs_S0",
        ),
    ],
)
def test_mechanism_classification_is_conservative(
    selected_own, own_s0, selected_s0, expected
):
    assert (
        diagnostic.mechanism_classification(selected_own, own_s0, selected_s0)
        == expected
    )


def test_raw_mechanism_uses_material_loss_threshold_names():
    assert (
        diagnostic.raw_mechanism_classification(
            "material_loss", "no_material_loss", "material_loss"
        )
        == "accepted_refinement_material_loss"
    )
    assert (
        diagnostic.raw_mechanism_classification(
            "no_material_loss", "material_loss", "material_loss"
        )
        == "material_loss_before_refinement"
    )
    assert (
        diagnostic.raw_mechanism_classification(
            "no_material_loss", "no_material_loss", "material_loss"
        )
        == "distributed_raw_threshold_accumulation"
    )
    assert (
        diagnostic.raw_mechanism_classification(
            "material_loss", "no_material_loss", "no_material_loss"
        )
        == "direct_MC_does_not_confirm_material_loss"
    )


def test_penalty_keeps_raw_diagnostics_but_disables_raw_localization():
    records = {
        "selected_minus_own_winner": {
            "judge": {"classification": "harmful"},
            "raw_loss": {"classification": "material_loss"},
            "penalty_active": True,
        },
        "own_winner_minus_S0": {
            "judge": {"classification": "practical_tie"},
            "raw_loss": {"classification": "no_material_loss"},
            "penalty_active": False,
        },
        "selected_minus_S0": {
            "judge": {"classification": "harmful"},
            "raw_loss": {"classification": "material_loss"},
            "penalty_active": True,
        },
    }
    score, raw = diagnostic.classify_case_records(records)
    assert score == "accepted_refinement_regression"
    assert raw == "not_applicable_penalty_active"
    assert records["selected_minus_own_winner"]["raw_loss"] == {
        "classification": "material_loss"
    }
