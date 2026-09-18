"""Protocol tests for the manifest-driven E3 search campaign."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import noisy_acq_search_experiment as campaign
import numpy as np
import pytest


def _pin_threads(monkeypatch):
    for name in campaign.THREAD_KEYS:
        monkeypatch.setenv(name, "1")


def _selection(stage="development", method="stratified_rqmc", budget=512):
    record = {
        "kind": "noisy_acq_search_selection",
        "frozen": True,
        "stage": stage,
        "accurate_rule": {
            "method": method,
            "budget": budget,
            "provenance": "externally_frozen_test_decision",
        },
    }
    if stage == "development_control":
        config = campaign.search.SearchConfig(
            arm="S3", accurate_method=method, accurate_budget=budget
        )
        record["strongest_arm_config"] = campaign.asdict(config)
        record["development_manifest_sha256"] = "frozen-development"
    elif stage == "holdout":
        config = campaign.search.SearchConfig(
            arm="S2", accurate_method=method, accurate_budget=budget
        )
        record["finalist_config"] = campaign.asdict(config)
        record["include_mc1600_control"] = True
    elif stage in campaign.F2_STAGES:
        record["accurate_rule"] = {
            "method": "mc",
            "budget": 1600,
            "provenance": "unused_by_production_arms",
        }
        if stage == "f2_control_development":
            record["control_rule"] = {
                "kind": campaign.F2_CONTROL_RULE,
                "offset": campaign.F2_CONTROL_OFFSET,
            }
        else:
            record["node_rule"] = {
                "kind": campaign.F2_NODE_RULE,
                "samples": campaign.F2_NODE_SAMPLES,
                "orders": list(campaign.F2_ORDERS),
            }
        if stage == "f2_holdout":
            record["development_choice"] = "axis"
            record["development_manifest_sha256"] = "frozen-development"
    return record


def _canned_transition():
    """A declared F2 source transition, as prepare_manifest would record."""
    return {
        "scope": campaign.F2_TRANSITION_SCOPE,
        "numerical_base_commit": campaign.capture.NUMERICAL_BASE,
        "changed_sources": {
            "pyvbmc/vbmc/active_importance_sampling.py": {
                "capture": "old",
                "runtime": "new",
            }
        },
        "numerical_diff_exclude": list(campaign.F2_NUMERICAL_DIFF_EXCLUDE),
    }


def _prepare_inputs(tmp_path, monkeypatch, selection):
    _pin_threads(monkeypatch)
    capture_manifest = tmp_path / "capture.json"
    capture_manifest.write_text(
        json.dumps({"allocation": [{"label": "case", "seeds": [0, 1]}]}),
        encoding="utf-8",
    )
    if selection["stage"] in ("development_control", "f2_holdout"):
        development_manifest = tmp_path / "development-manifest.json"
        source = {
            "stage": "development"
            if selection["stage"] == "development_control"
            else "f2_development",
            "capture_manifest_sha256": campaign.integration.sha256_file(
                capture_manifest
            ),
            "identity": {"frozen": True},
        }
        if selection["stage"] == "development_control":
            source["treatments"] = [
                {
                    "tag": selection["strongest_arm_config"]["arm"],
                    "config": selection["strongest_arm_config"],
                }
            ]
        development_manifest.write_text(json.dumps(source), encoding="utf-8")
        selection["development_manifest"] = str(development_manifest)
        selection[
            "development_manifest_sha256"
        ] = campaign.integration.sha256_file(development_manifest)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    monkeypatch.setattr(
        campaign.capture, "validate_manifest", lambda *a, **k: None
    )
    monkeypatch.setattr(
        campaign, "_identity", lambda *args, **kwargs: {"frozen": True}
    )
    monkeypatch.setattr(
        campaign, "_f2_source_transition", lambda _: _canned_transition()
    )
    trajectory_seed = 1 if selection["stage"] in campaign.HOLDOUT_STAGES else 0
    state = {
        "state_id": f"case_seed{trajectory_seed}_early",
        "label": "case",
        "seed": trajectory_seed,
        "snapshot": str(tmp_path / "snapshot"),
        "snapshot_hashes": {".json": "j", ".npz": "n"},
    }
    monkeypatch.setattr(
        campaign.integration,
        "discover_states",
        lambda *args: ([state], []),
    )
    return capture_manifest, selection_path


def test_development_manifest_pairs_seeds_without_naming_winner(
    tmp_path, monkeypatch
):
    capture_manifest, selection_path = _prepare_inputs(
        tmp_path, monkeypatch, _selection()
    )
    manifest = campaign.prepare_manifest(
        capture_manifest,
        tmp_path,
        selection_path,
        "development",
    )
    assert [item["tag"] for item in manifest["treatments"]] == [
        "S0",
        "S1",
        "S2",
        "S3",
    ]
    assert "finalist_config" not in manifest["selection"]
    replicate_zero = [
        cell for cell in manifest["cells"] if cell["replicate"] == 0
    ]
    assert len({cell["search_seed"] for cell in replicate_zero}) == 1
    assert len({cell["accurate_seed"] for cell in replicate_zero}) == 1
    assert len(manifest["cells"]) == 4 * campaign.SELECTION_REPLICATES
    assert not manifest["launch_ready"]
    assert not manifest["holdout_locked"]


def test_f2_development_pairs_three_production_arms_on_fresh_streams(
    tmp_path, monkeypatch
):
    e3_dir = tmp_path / "e3"
    e3_dir.mkdir()
    e3_capture, e3_selection = _prepare_inputs(
        e3_dir, monkeypatch, _selection()
    )
    e3_manifest = campaign.prepare_manifest(
        e3_capture, e3_dir, e3_selection, "development"
    )
    capture_manifest, selection_path = _prepare_inputs(
        tmp_path, monkeypatch, _selection(stage="f2_development")
    )
    manifest = campaign.prepare_manifest(
        capture_manifest, tmp_path, selection_path, "f2_development"
    )
    assert [item["tag"] for item in manifest["treatments"]] == [
        "S0",
        "S0_qmc_sorted",
        "S0_qmc_unsorted",
    ]
    configs = {item["tag"]: item["config"] for item in manifest["treatments"]}
    assert all(config["arm"] == "S0" for config in configs.values())
    assert configs["S0"]["importance_qmc"] is False
    assert configs["S0_qmc_sorted"] == {
        **configs["S0"],
        "importance_qmc": True,
        "importance_qmc_samples": 96,
        "importance_qmc_order": "axis",
    }
    assert configs["S0_qmc_unsorted"]["importance_qmc_order"] == "index"
    assert manifest["split"] == "development"
    assert manifest["stage"] == "f2_development"
    assert not manifest["holdout_locked"] and not manifest["launch_ready"]
    assert manifest["selection"]["node_rule"]["samples"] == 96
    assert len(manifest["cells"]) == 3 * campaign.SELECTION_REPLICATES
    assert manifest["timing_cell_count"] == 2
    assert manifest["allocation_cell_count"] == 4
    replicate_zero = [
        cell for cell in manifest["cells"] if cell["replicate"] == 0
    ]
    assert len({cell["search_seed"] for cell in replicate_zero}) == 1
    # The F2 stage seeds its own streams: the same state and replicate
    # draw a different candidate-search seed than the E3 development stage.
    e3_zero = [cell for cell in e3_manifest["cells"] if cell["replicate"] == 0]
    assert replicate_zero[0]["search_seed"] != e3_zero[0]["search_seed"]
    campaign.validate_manifest(manifest, require_ready=False)
    assert [
        spec["contrast"] for spec in campaign._contrast_specs(manifest)
    ] == ["S0_qmc_sorted_minus_S0", "S0_qmc_unsorted_minus_S0"]


def test_f2_control_stage_pairs_the_baseline_with_a_reseeded_copy(
    tmp_path, monkeypatch
):
    capture_manifest, selection_path = _prepare_inputs(
        tmp_path, monkeypatch, _selection(stage="f2_control_development")
    )
    manifest = campaign.prepare_manifest(
        capture_manifest, tmp_path, selection_path, "f2_control_development"
    )
    assert [item["tag"] for item in manifest["treatments"]] == [
        "S0",
        "S0_reseeded",
    ]
    control = manifest["treatments"][1]
    assert control["role"] == "search_stream_null_control"
    assert control["config"] == {
        **manifest["treatments"][0]["config"],
        "search_stream_offset": 1,
    }
    assert manifest["split"] == "development"
    assert manifest["purpose"].endswith("search-stream null control")
    assert len(manifest["cells"]) == 2 * campaign.SELECTION_REPLICATES
    campaign.validate_manifest(manifest, require_ready=False)
    assert [
        spec["contrast"] for spec in campaign._contrast_specs(manifest)
    ] == ["S0_reseeded_minus_S0"]
    # Its streams are its own: the same state and replicate draw another
    # search seed than the F2 development stage.
    other = tmp_path / "nodes"
    other.mkdir()
    node_capture, node_selection = _prepare_inputs(
        other, monkeypatch, _selection(stage="f2_development")
    )
    node_manifest = campaign.prepare_manifest(
        node_capture, other, node_selection, "f2_development"
    )
    assert (
        manifest["cells"][0]["search_seed"]
        != node_manifest["cells"][0]["search_seed"]
    )
    bad = _selection(stage="f2_control_development")
    bad["control_rule"]["offset"] = 2
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    bad_capture, bad_selection = _prepare_inputs(wrong, monkeypatch, bad)
    with pytest.raises(RuntimeError, match="one-draw"):
        campaign.prepare_manifest(
            bad_capture, wrong, bad_selection, "f2_control_development"
        )


def test_f2_holdout_binds_the_development_choice_and_stays_locked(
    tmp_path, monkeypatch
):
    capture_manifest, selection_path = _prepare_inputs(
        tmp_path, monkeypatch, _selection(stage="f2_holdout")
    )
    manifest = campaign.prepare_manifest(
        capture_manifest, tmp_path, selection_path, "f2_holdout"
    )
    assert manifest["split"] == "holdout"
    assert manifest["holdout_locked"] is True
    assert manifest["selection"]["development_choice"] == "axis"
    assert [item["tag"] for item in manifest["treatments"]] == [
        "S0",
        "S0_qmc_sorted",
        "S0_qmc_unsorted",
    ]
    roles = {item["tag"]: item["role"] for item in manifest["treatments"]}
    assert roles["S0_qmc_sorted"] == "qmc_node_treatment_confirmation"
    assert roles["S0_qmc_unsorted"] == "qmc_node_control_descriptive"
    assert manifest["holdout_confirmation"] == {
        "development_choice": "axis",
        "primary_contrast": "S0_qmc_sorted_minus_S0",
        "descriptive_contrasts": ["S0_qmc_unsorted_minus_S0"],
    }
    with pytest.raises(RuntimeError, match="locked"):
        campaign.validate_manifest(
            {**manifest, "launch_ready": True}, require_ready=True
        )
    unlocked = {**manifest, "launch_ready": True, "holdout_locked": False}
    with pytest.raises(RuntimeError, match="split does not match"):
        campaign.validate_manifest(
            {**unlocked, "split": "development"}, require_ready=True
        )
    wrong_seed = json.loads(json.dumps(unlocked))
    wrong_seed["states"][0]["seed"] = 0
    with pytest.raises(RuntimeError, match="other split"):
        campaign.validate_manifest(wrong_seed, require_ready=True)
    with pytest.raises(RuntimeError, match="confirmation record"):
        campaign.validate_manifest(
            {**unlocked, "holdout_confirmation": None}, require_ready=True
        )
    unlocked = {**manifest, "launch_ready": True, "holdout_locked": False}
    campaign.validate_manifest(unlocked, require_ready=True)


def test_f2_selection_rejects_another_node_rule_or_a_search_arm(
    tmp_path, monkeypatch
):
    selection = _selection(stage="f2_development")
    selection["node_rule"]["samples"] = 128
    capture_manifest, selection_path = _prepare_inputs(
        tmp_path, monkeypatch, selection
    )
    with pytest.raises(RuntimeError, match="frozen 96-node"):
        campaign.prepare_manifest(
            capture_manifest, tmp_path, selection_path, "f2_development"
        )
    selection = _selection(stage="f2_development")
    selection["finalist_config"] = campaign.asdict(
        campaign.search.SearchConfig(arm="S2")
    )
    other = tmp_path / "other"
    other.mkdir()
    capture_manifest, selection_path = _prepare_inputs(
        other, monkeypatch, selection
    )
    with pytest.raises(RuntimeError, match="names no search arm"):
        campaign.prepare_manifest(
            capture_manifest, other, selection_path, "f2_development"
        )


def test_f2_source_transition_declares_only_the_allowed_changes(
    tmp_path, monkeypatch
):
    pinned = {"pyvbmc/vbmc/vbmc.py": "same", "pyvbmc/other.py": "same"}
    for path in campaign.F2_ALLOWED_SOURCE_CHANGES:
        pinned[path] = "old"
    current = dict(pinned)
    current["pyvbmc/vbmc/active_importance_sampling.py"] = "new"
    current["pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini"] = "new"
    monkeypatch.setattr(campaign.capture, "source_hashes", lambda: current)
    transition = campaign._f2_source_transition({"source_hashes": pinned})
    assert transition["scope"] == "f2_importance_nodes"
    assert sorted(transition["changed_sources"]) == [
        "pyvbmc/vbmc/active_importance_sampling.py",
        "pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini",
    ]
    assert transition["changed_sources"][
        "pyvbmc/vbmc/active_importance_sampling.py"
    ] == {"capture": "old", "runtime": "new"}
    current["pyvbmc/vbmc/vbmc.py"] = "drifted"
    with pytest.raises(RuntimeError, match="beyond the declared"):
        campaign._f2_source_transition({"source_hashes": pinned})

    # validate_manifest accepts the declared transition on an F2 manifest,
    # refuses a misdeclared one, and refuses any transition on E3 stages.
    capture_manifest, selection_path = _prepare_inputs(
        tmp_path, monkeypatch, _selection(stage="f2_development")
    )
    manifest = campaign.prepare_manifest(
        capture_manifest, tmp_path, selection_path, "f2_development"
    )
    good = _canned_transition()
    assert manifest["source_transition"] == good
    campaign.validate_manifest(
        {**manifest, "source_transition": good}, require_ready=False
    )
    with pytest.raises(RuntimeError, match="lacks its source transition"):
        campaign.validate_manifest(
            {k: v for k, v in manifest.items() if k != "source_transition"},
            require_ready=False,
        )
    bad = dict(good)
    bad["changed_sources"] = {
        "pyvbmc/vbmc/vbmc.py": {"capture": "old", "runtime": "new"}
    }
    with pytest.raises(RuntimeError, match="not the declared one"):
        campaign.validate_manifest(
            {**manifest, "source_transition": bad}, require_ready=False
        )
    e3_dir = tmp_path / "e3"
    e3_dir.mkdir()
    e3_capture, e3_selection = _prepare_inputs(
        e3_dir, monkeypatch, _selection()
    )
    e3_manifest = campaign.prepare_manifest(
        e3_capture, e3_dir, e3_selection, "development"
    )
    assert "source_transition" not in e3_manifest
    with pytest.raises(RuntimeError, match="only F2 stages"):
        campaign.validate_manifest(
            {**e3_manifest, "source_transition": good}, require_ready=False
        )


def test_timing_and_allocation_records_project_their_results(
    tmp_path, monkeypatch
):
    manifest = _minimal_manifest(tmp_path)
    monkeypatch.setattr(campaign, "runtime_identity", lambda _: {})
    state = {"vp": SimpleNamespace(D=2)}
    monkeypatch.setattr(campaign.capture, "restore_capture", lambda _: state)
    monkeypatch.setattr(campaign, "_state_digest", lambda _: "unchanged")
    monkeypatch.setattr(
        campaign.search,
        "prepare_selection",
        lambda state, config, **kwargs: nullcontext(
            lambda: _fake_search_result(0)
        ),
    )

    def time_pair(baseline, treatment, *, seed):
        results = {}
        for name, factory in (
            ("baseline", baseline),
            ("treatment", treatment),
        ):
            with factory(seed) as operation:
                results[name] = operation()
                results[name]["cache_indices"] = np.array([np.nan, 1.0])
                results[name]["importance_node_count"] = 96
        first_round = {"round": 0}
        for name, result in results.items():
            first_round[name] = {
                "seed": seed,
                "seconds": 0.1,
                "result": result,
            }
        return {"seed": seed, "rounds": [first_round], "quality_claim": False}

    monkeypatch.setattr(campaign.timing, "time_pair", time_pair)
    cell = manifest["timing_cells"][0]
    assert campaign.run_timing_cell(manifest, tmp_path, cell)["status"] == (
        "succeeded"
    )
    written = json.loads(
        campaign._paths(tmp_path, campaign.timing_tag(cell), "timing")[
            "json"
        ].read_text(encoding="utf-8")
    )
    for name in ("baseline", "treatment"):
        result = written["timing"]["rounds"][0][name]["result"]
        assert "cache_indices" not in result
        assert result["importance_node_count"] == 96
        assert result["selected"] == [0.0, 0.0]

    def measure_allocations(prepare, *, seed):
        with prepare(seed) as operation:
            result = operation()
        result["cache_indices"] = np.full(8192, np.nan)
        return {"seed": seed, "result": result, "traced_peak_bytes": 1}

    monkeypatch.setattr(
        campaign.timing, "measure_allocations", measure_allocations
    )
    allocation = manifest["allocation_cells"][0]
    assert campaign.run_allocation_cell(manifest, tmp_path, allocation)[
        "status"
    ] == ("succeeded")
    written = json.loads(
        campaign._paths(
            tmp_path, campaign.allocation_tag(allocation), "allocation"
        )["json"].read_text(encoding="utf-8")
    )
    assert "cache_indices" not in written["allocation"]["result"]
    assert written["allocation"]["result"]["selected"] == [0.0, 0.0]


def test_timing_projection_keeps_scalars_and_the_selected_row():
    result = _fake_search_result(1)
    result["cache_indices"] = np.array([np.nan, 1.0])
    result["importance_node_count"] = 96
    projected = campaign._project_timing_result(result)
    assert "cache_indices" not in projected
    assert "coarse_candidates" not in projected
    assert projected["importance_node_count"] == 96
    assert projected["selected"] == [1.0, 1.0]
    assert projected["selected_row_sha256"] == campaign._row_id(
        np.array([1.0, 1.0])
    )
    json.dumps(campaign.capture.canonical(projected), allow_nan=False)


def test_mc1600_is_allowed_as_an_external_fallback(tmp_path, monkeypatch):
    capture_manifest, selection_path = _prepare_inputs(
        tmp_path, monkeypatch, _selection(method="mc", budget=1600)
    )
    manifest = campaign.prepare_manifest(
        capture_manifest, tmp_path, selection_path, "development"
    )
    for treatment in manifest["treatments"][1:]:
        assert treatment["config"]["accurate_method"] == "mc"
        assert treatment["config"]["accurate_budget"] == 1600


def test_development_control_freezes_one_arm_and_reuses_search_streams(
    tmp_path, monkeypatch
):
    capture_manifest, selection_path = _prepare_inputs(
        tmp_path, monkeypatch, _selection(stage="development_control")
    )
    control = campaign.prepare_manifest(
        capture_manifest, tmp_path, selection_path, "development_control"
    )
    development_selection = _selection()
    development_path = tmp_path / "development-selection.json"
    development_path.write_text(
        json.dumps(development_selection), encoding="utf-8"
    )
    development = campaign.prepare_manifest(
        capture_manifest, tmp_path, development_path, "development"
    )
    assert [item["tag"] for item in control["treatments"]] == [
        "S0",
        "original_arm",
        "mc1600_control",
    ]
    assert control["split"] == "development"
    control_zero = [
        cell for cell in control["cells"] if cell["replicate"] == 0
    ]
    development_zero = [
        cell for cell in development["cells"] if cell["replicate"] == 0
    ]
    assert control_zero[0]["search_seed"] == development_zero[0]["search_seed"]
    assert (
        control_zero[1]["accurate_seed"]
        == development_zero[1]["accurate_seed"]
    )
    assert control_zero[2]["accurate_seed"] != control_zero[1]["accurate_seed"]
    development_s3 = next(
        item for item in development["treatments"] if item["tag"] == "S3"
    )
    assert control["treatments"][1]["config"] == development_s3["config"]
    assert control["selection"]["development_manifest_sha256"]
    assert campaign._contrast_specs(control)[-1] == {
        "contrast": "mc1600_control_minus_original_arm",
        "kind": "direct_integration_control",
        "arm_tag": "mc1600_control",
        "reference_tag": "original_arm",
    }
    assert (
        control["direct_control_timing"]["classification"]
        == "descriptive_only"
    )
    monkeypatch.setattr(campaign, "validate_manifest", lambda *a, **k: None)
    Path(control["selection"]["development_manifest"]).write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="preceding development manifest"):
        campaign.runtime_identity(control)


def test_development_control_skips_duplicate_mc1600_arm(tmp_path, monkeypatch):
    capture_manifest, selection_path = _prepare_inputs(
        tmp_path,
        monkeypatch,
        _selection(stage="development_control", method="mc", budget=1600),
    )
    control = campaign.prepare_manifest(
        capture_manifest, tmp_path, selection_path, "development_control"
    )
    assert [item["tag"] for item in control["treatments"]] == [
        "S0",
        "original_arm",
    ]
    assert not any(
        item["kind"] == "direct_integration_control"
        for item in campaign._contrast_specs(control)
    )
    assert control["direct_control_timing"]["classification"] == (
        "not_applicable_duplicate_rule"
    )


def test_development_control_rejects_nonintegration_config_drift(
    tmp_path, monkeypatch
):
    capture_manifest, selection_path = _prepare_inputs(
        tmp_path, monkeypatch, _selection(stage="development_control")
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    development_path = Path(selection["development_manifest"])
    development = json.loads(development_path.read_text(encoding="utf-8"))
    development["treatments"][0]["config"]["finite_difference_step"] *= 2
    development_path.write_text(json.dumps(development), encoding="utf-8")
    selection[
        "development_manifest_sha256"
    ] = campaign.integration.sha256_file(development_path)
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(RuntimeError, match="config differs"):
        campaign.prepare_manifest(
            capture_manifest,
            tmp_path,
            selection_path,
            "development_control",
        )


def test_holdout_contains_only_frozen_finalist_and_plain_mc_control(
    tmp_path, monkeypatch
):
    capture_manifest, selection_path = _prepare_inputs(
        tmp_path, monkeypatch, _selection(stage="holdout")
    )
    manifest = campaign.prepare_manifest(
        capture_manifest, tmp_path, selection_path, "holdout"
    )
    assert [item["tag"] for item in manifest["treatments"]] == [
        "S0",
        "finalist",
        "mc1600_control",
    ]
    control = manifest["treatments"][-1]
    assert control["config"]["arm"] == "S2"
    assert control["config"]["accurate_method"] == "mc"
    assert control["config"]["accurate_budget"] == 1600
    replicate_zero = [
        cell for cell in manifest["cells"] if cell["replicate"] == 0
    ]
    assert len({cell["search_seed"] for cell in replicate_zero}) == 1
    primary = next(
        cell for cell in replicate_zero if cell["treatment_tag"] == "finalist"
    )
    control_cell = next(
        cell
        for cell in replicate_zero
        if cell["treatment_tag"] == "mc1600_control"
    )
    assert primary["accurate_seed"] != control_cell["accurate_seed"]
    assert campaign._contrast_specs(manifest)[-1] == {
        "contrast": "mc1600_control_minus_finalist",
        "kind": "direct_integration_control",
        "arm_tag": "mc1600_control",
        "reference_tag": "finalist",
    }
    manifest["launch_ready"] = True
    with pytest.raises(RuntimeError, match="locked"):
        campaign.validate_manifest(manifest, require_ready=True)


def test_holdout_skips_duplicate_mc1600_control(tmp_path, monkeypatch):
    capture_manifest, selection_path = _prepare_inputs(
        tmp_path,
        monkeypatch,
        _selection(stage="holdout", method="mc", budget=1600),
    )
    manifest = campaign.prepare_manifest(
        capture_manifest, tmp_path, selection_path, "holdout"
    )
    assert [item["tag"] for item in manifest["treatments"]] == [
        "S0",
        "finalist",
    ]
    assert manifest["direct_control_timing"]["classification"] == (
        "not_applicable_duplicate_rule"
    )


def test_manifests_frozen_before_new_config_fields_still_validate(
    tmp_path, monkeypatch
):
    capture_manifest, selection_path = _prepare_inputs(
        tmp_path, monkeypatch, _selection()
    )
    manifest = campaign.prepare_manifest(
        capture_manifest, tmp_path, selection_path, "development"
    )
    older = json.loads(json.dumps(manifest))
    for treatment in older["treatments"]:
        for key in (
            "importance_qmc",
            "importance_qmc_samples",
            "importance_qmc_order",
        ):
            del treatment["config"][key]
    campaign.validate_manifest(older, require_ready=False)
    older["treatments"][1]["config"]["sieve_size"] = 2048
    with pytest.raises(RuntimeError, match="differ from the frozen"):
        campaign.validate_manifest(older, require_ready=False)


def _minimal_manifest(tmp_path, treatments=None):
    treatments = treatments or [
        {
            "tag": "S0",
            "role": "production_baseline",
            "config": campaign.asdict(campaign.search.SearchConfig(arm="S0")),
            "accurate_seed_role": "unused_baseline",
        },
        {
            "tag": "S1",
            "role": "search_treatment",
            "config": campaign.asdict(campaign.search.SearchConfig(arm="S1")),
            "accurate_seed_role": "shared_primary_rule",
        },
    ]
    cells = [
        {
            "state_id": "case_seed0_early",
            "treatment_tag": treatment["tag"],
            "replicate": replicate,
            "search_seed": 100 + replicate,
            "accurate_seed": 200 + replicate,
        }
        for replicate in range(campaign.SELECTION_REPLICATES)
        for treatment in treatments
    ]
    timing_cells = [
        {
            "state_id": "case_seed0_early",
            "treatment_tag": "S1",
            "timing_seed": 301,
        }
    ]
    allocation_cells = [
        {
            **timing_cells[0],
            "role": role,
            "allocation_seed": campaign.integration.derive_seed(
                301, role, "allocation_measurement"
            ),
        }
        for role in ("baseline", "treatment")
    ]
    return {
        "schema_version": campaign.SCHEMA_VERSION,
        "kind": "search",
        "stage": "development",
        "split": "development",
        "launch_ready": True,
        "holdout_locked": False,
        "replicates": campaign.SELECTION_REPLICATES,
        "capture_manifest": str(tmp_path / "capture.json"),
        "capture_manifest_sha256": "capture",
        "identity": {},
        "thread_environment": {name: "1" for name in campaign.THREAD_KEYS},
        "states": [
            {
                "state_id": "case_seed0_early",
                "snapshot": str(tmp_path / "snapshot"),
                "snapshot_hashes": {},
            }
        ],
        "missing_states": [],
        "expected_state_count": 1,
        "treatments": treatments,
        "cells": cells,
        "cell_count": len(cells),
        "scheduled_cell_count_including_missing_states": len(cells),
        "timing_cells": timing_cells,
        "timing_cell_count": len(timing_cells),
        "scheduled_timing_count_including_missing_states": len(timing_cells),
        "allocation_cells": allocation_cells,
        "allocation_cell_count": len(allocation_cells),
        "scheduled_allocation_count_including_missing_states": len(
            allocation_cells
        ),
        "judge_budgets": list(campaign.JUDGE_BUDGETS),
        "initial_judge_budget": campaign.JUDGE_BUDGETS[0],
        "judge": {
            "case_seed0_early": {
                "comparison_seeds": {
                    str(budget): list(range(8))
                    for budget in campaign.JUDGE_BUDGETS
                },
                "band_pilot_budget": 4096,
                "band_pilot_seeds": list(range(8, 16)),
            }
        },
    }


def _fake_search_result(seed):
    candidates = np.array([[0.0, 0.0], [1.0, 1.0]])
    return {
        "arm": "S1",
        "selected": candidates[[seed % 2]],
        "coarse_candidates": candidates,
        "coarse_scores": np.array([0.0, 1.0]),
        "coarse_nodes": np.array([[0.2, 0.3]]),
        "coarse_winner": candidates[0],
        "cache_indices": np.array([0.0, 1.0]),
        "target_called": False,
        "cache_index": 0.0,
        "elapsed_seconds": 0.1,
    }


def test_selection_cell_is_atomic_hash_checked_and_rejects_partials(
    tmp_path, monkeypatch
):
    manifest = _minimal_manifest(tmp_path)
    cell = manifest["cells"][1]
    monkeypatch.setattr(campaign, "runtime_identity", lambda _: {})
    state = {"vp": SimpleNamespace(D=2)}
    monkeypatch.setattr(campaign.capture, "restore_capture", lambda _: state)
    monkeypatch.setattr(campaign, "_state_digest", lambda _: "unchanged")
    monkeypatch.setattr(
        campaign.search,
        "select_candidate",
        lambda *args, **kwargs: _fake_search_result(cell["replicate"]),
    )
    terminal = campaign.run_cell(manifest, tmp_path, cell)
    assert terminal["status"] == "succeeded"
    paths = campaign._paths(tmp_path, campaign.cell_tag(cell), "selection")
    paths["json"].write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        campaign.run_cell(manifest, tmp_path, cell)

    other = manifest["cells"][3]
    other_paths = campaign._paths(
        tmp_path, campaign.cell_tag(other), "selection"
    )
    campaign.integration.write_json(other_paths["json"], {"partial": True})
    with pytest.raises(RuntimeError, match="partial"):
        campaign.run_cell(manifest, tmp_path, other)


def test_timing_cell_prepares_contexts_outside_clock_without_panels(
    tmp_path, monkeypatch
):
    manifest = _minimal_manifest(tmp_path)
    cell = manifest["timing_cells"][0]
    monkeypatch.setattr(campaign, "runtime_identity", lambda _: {})
    state = {"vp": SimpleNamespace(D=2)}
    monkeypatch.setattr(campaign.capture, "restore_capture", lambda _: state)
    monkeypatch.setattr(campaign, "_state_digest", lambda _: "unchanged")
    preparation = []

    def prepare_selection(state, config, **kwargs):
        preparation.append(kwargs)
        return nullcontext(lambda: {"selected": np.zeros((1, 2))})

    def time_pair(baseline, treatment, *, seed):
        for factory in (baseline, treatment):
            with factory(seed) as operation:
                operation()
        return {"seed": seed, "rounds": [], "quality_claim": False}

    monkeypatch.setattr(
        campaign.search, "prepare_selection", prepare_selection
    )
    monkeypatch.setattr(campaign.timing, "time_pair", time_pair)
    terminal = campaign.run_timing_cell(manifest, tmp_path, cell)
    assert terminal["status"] == "succeeded"
    assert len(preparation) == 2
    assert all(item["retain_panel"] is False for item in preparation)
    assert preparation[0]["search_seed"] == preparation[1]["search_seed"]


def test_allocation_cell_uses_one_fresh_arm_context(tmp_path, monkeypatch):
    manifest = _minimal_manifest(tmp_path)
    cell = manifest["allocation_cells"][1]
    monkeypatch.setattr(campaign, "runtime_identity", lambda _: {})
    state = {"vp": SimpleNamespace(D=2)}
    monkeypatch.setattr(campaign.capture, "restore_capture", lambda _: state)
    monkeypatch.setattr(campaign, "_state_digest", lambda _: "unchanged")
    prepared = []

    def prepare_selection(state, config, **kwargs):
        prepared.append((config.arm, kwargs))
        return nullcontext(lambda: {"selected": np.zeros((1, 2))})

    def measure(prepare, *, seed):
        with prepare(seed) as operation:
            operation()
        return {"seed": seed, "timing_evidence": False}

    monkeypatch.setattr(
        campaign.search, "prepare_selection", prepare_selection
    )
    monkeypatch.setattr(campaign.timing, "measure_allocations", measure)
    terminal = campaign.run_allocation_cell(manifest, tmp_path, cell)
    assert terminal["status"] == "succeeded"
    assert prepared[0][0] == "S1"
    assert prepared[0][1]["retain_panel"] is False


def _publish_selected(manifest, out, cell, selected):
    tag = campaign.cell_tag(cell)
    paths = campaign._paths(out, tag, "selection")
    campaign.integration.write_npz(
        paths["npz"],
        {
            "selected": np.asarray(selected, dtype=np.float64).reshape(1, -1),
            "coarse_candidates": np.asarray(
                selected, dtype=np.float64
            ).reshape(1, -1),
            "coarse_scores": np.array([0.0]),
            "coarse_nodes": np.asarray(selected, dtype=np.float64).reshape(
                1, -1
            ),
        },
    )
    campaign.integration.write_json(paths["json"], {"cell": cell})
    campaign._publish_terminal(
        manifest,
        paths,
        cell,
        "succeeded",
        {
            "npz": campaign.integration.sha256_file(paths["npz"]),
            "json": campaign.integration.sha256_file(paths["json"]),
        },
    )


def _patch_reference_shortlist(monkeypatch):
    candidates = np.column_stack(
        (np.arange(8192, dtype=np.float64), -np.arange(8192, dtype=np.float64))
    )
    # Exact-byte deduplication needs the fixture's shared zero row to agree.
    candidates[candidates == 0] = 0.0
    monkeypatch.setattr(
        campaign.capture,
        "restore_capture",
        lambda _: {"vp": SimpleNamespace(D=2)},
    )
    monkeypatch.setattr(campaign, "_state_digest", lambda _: "unchanged")
    monkeypatch.setattr(campaign, "runtime_identity", lambda _: {})
    monkeypatch.setattr(
        campaign.integration,
        "candidate_panel",
        lambda *args: (
            candidates,
            np.arange(8192),
            np.arange(8192, dtype=np.float64),
            np.arange(8192),
        ),
    )


def test_judge_union_deduplicates_exact_rows_and_tracks_provenance(
    tmp_path, monkeypatch
):
    manifest = _minimal_manifest(tmp_path)
    _patch_reference_shortlist(monkeypatch)
    for cell in manifest["cells"]:
        if cell["treatment_tag"] == "S0" or cell["replicate"] >= 4:
            selected = np.array([0.0, 0.0])
        else:
            selected = np.array([1.0, 1.0])
        _publish_selected(manifest, tmp_path, cell, selected)
    union = campaign.build_judge_union(manifest, tmp_path, "case_seed0_early")
    assert union["candidates"].dtype == np.float64
    assert union["candidates"].shape == (33, 2)
    assert set(union["baseline_by_replicate"]) == {
        str(index) for index in range(8)
    }
    assert set(union["treatment_by_tag"]["S1"]) == {
        str(index) for index in range(8)
    }
    assert len(union["candidate_row_ids"]) == 33
    assert len(union["fixed_reference_row_ids"]) == 32
    assert set(union["candidate_row_ids"]) == set(
        union["candidate_provenance"]
    )
    selected_sources = [
        source
        for sources in union["candidate_provenance"].values()
        for source in sources
        if source["source"] == "selected_candidate"
    ]
    assert len(selected_sources) == 16
    assert union["fixed_band_candidate"].shape == (1, 2)


def test_judge_union_retains_valid_pairs_when_one_selection_is_missing(
    tmp_path, monkeypatch
):
    manifest = _minimal_manifest(tmp_path)
    _patch_reference_shortlist(monkeypatch)
    missing = next(
        cell
        for cell in manifest["cells"]
        if cell["treatment_tag"] == "S1" and cell["replicate"] == 0
    )
    for cell in manifest["cells"]:
        if cell != missing:
            _publish_selected(
                manifest,
                tmp_path,
                cell,
                np.array([cell["replicate"], cell["treatment_tag"] == "S1"]),
            )
    union = campaign.build_judge_union(manifest, tmp_path, "case_seed0_early")
    assert "0" in union["baseline_by_replicate"]
    assert "0" not in union["treatment_by_tag"]["S1"]
    assert union["missing_selections"][0]["cell"] == missing
    assert len(union["fixed_reference_row_ids"]) == 32


def test_judge_union_rejects_corrupt_selection_terminal(tmp_path, monkeypatch):
    manifest = _minimal_manifest(tmp_path)
    _patch_reference_shortlist(monkeypatch)
    cell = manifest["cells"][0]
    _publish_selected(manifest, tmp_path, cell, np.zeros(2))
    paths = campaign._paths(tmp_path, campaign.cell_tag(cell), "selection")
    paths["json"].write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        campaign.build_judge_union(manifest, tmp_path, "case_seed0_early")


def test_development_contrasts_separate_refinement_effects(tmp_path):
    treatments = [
        {
            "tag": tag,
            "role": "production_baseline"
            if tag == "S0"
            else "search_treatment",
            "config": campaign.asdict(campaign.search.SearchConfig(arm=tag)),
            "accurate_seed_role": (
                "unused_baseline" if tag == "S0" else "shared_primary_rule"
            ),
        }
        for tag in ("S0", "S1", "S2", "S3")
    ]
    manifest = _minimal_manifest(tmp_path, treatments)
    specs = campaign._contrast_specs(manifest)
    assert [item["contrast"] for item in specs] == [
        "S1_minus_S0",
        "S2_minus_S0",
        "S3_minus_S0",
        "S2_minus_S1",
        "S3_minus_S2",
        "S3_minus_S1",
    ]


def test_judge_cell_accepts_arbitrary_selected_coordinate_union(
    tmp_path, monkeypatch
):
    manifest = _minimal_manifest(tmp_path)
    _patch_reference_shortlist(monkeypatch)
    for cell in manifest["cells"]:
        selected = np.array(
            [float(cell["replicate"]), cell["treatment_tag"] == "S1"]
        )
        _publish_selected(manifest, tmp_path, cell, selected)
    monkeypatch.setattr(campaign, "runtime_identity", lambda _: {})
    state = {"vp": SimpleNamespace(D=2)}
    monkeypatch.setattr(campaign.capture, "restore_capture", lambda _: state)
    seen = {}

    def evaluate(state, candidates, budget, seeds, **kwargs):
        seen["candidates"] = candidates.copy()
        seen["budget"] = budget
        return {
            "full_score": np.zeros((8, len(candidates))),
            "log_residual": np.zeros((8, len(candidates))),
            "log_reduction": np.zeros((8, len(candidates))),
            "penalty": np.zeros((8, len(candidates))),
            "valid": np.ones((8, len(candidates)), dtype=bool),
            "reference_log_residual": np.zeros(8),
            "actual_nodes": np.full(8, budget),
            "pilot_full_score": np.zeros(8),
            "pilot_log_residual": np.zeros(8),
            "pilot_log_reduction": np.zeros(8),
            "pilot_reference_log_residual": np.ones(8),
        }

    monkeypatch.setattr(
        campaign.integration, "evaluate_judge_matrix", evaluate
    )
    terminal = campaign.run_judge_cell(
        manifest, tmp_path, "case_seed0_early", 4096
    )
    assert terminal["status"] == "succeeded"
    assert seen["budget"] == 4096
    assert seen["candidates"].shape == (47, 2)
    with np.load(
        tmp_path / "judge" / "case_seed0_early__judge_b4096.npz"
    ) as data:
        np.testing.assert_array_equal(data["candidates"], seen["candidates"])
    summary = campaign.judge_summary(manifest, tmp_path, 4096)
    assert summary["counts"]["score_counts"]["unresolved"] == 8
    assert summary["comparisons"][0]["judge"]["replicate_seeds"] == list(
        range(8)
    )
    previous = tmp_path / "summary-4096.json"
    campaign.integration.write_json(previous, summary)
    escalated = campaign.run_judge_cell(
        manifest, tmp_path, "case_seed0_early", 8192, previous
    )
    assert escalated["status"] == "succeeded"
    final = campaign.judge_summary(manifest, tmp_path, 8192, previous)
    assert final["counts"]["score_counts"]["practical_tie"] == 8
    assert final["counts"]["raw_loss_gate_counts"]["no_material_loss"] == 8
    final_paths = campaign._paths(
        tmp_path, campaign.judge_tag("case_seed0_early", 8192), "judge"
    )
    final_paths["json"].write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        campaign.judge_summary(manifest, tmp_path, 8192, previous)


def test_inventory_counts_failed_missing_and_unscheduled_capture_states(
    tmp_path,
):
    manifest = _minimal_manifest(tmp_path)
    manifest["missing_states"] = [
        {"state_id": "absent", "reason": "checkpoint_missing"}
    ]
    manifest["expected_state_count"] += 1
    manifest["scheduled_cell_count_including_missing_states"] += 16
    manifest["scheduled_allocation_count_including_missing_states"] += 2
    report = campaign.inventory(manifest, tmp_path)
    assert report["counts"]["selection"]["missing"] == (
        len(manifest["cells"]) + 16
    )
    assert report["counts"]["timing"]["missing"] == 1
    assert report["counts"]["allocation"]["missing"] == 4
    assert report["counts"]["judge"]["missing"] == 2
    assert report["missing_states"][0]["state_id"] == "absent"
    assert report["scheduled_selection_including_missing_states"] > len(
        manifest["cells"]
    )
