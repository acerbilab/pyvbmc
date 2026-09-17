import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import noisy_acq_inference as runner

TARGET_CONTRACTS = {label: {"standard_options": {}} for label in runner.LABELS}
REFERENCE = {
    "labels": {
        label: {"envelopes": {metric: 10.0 for metric in runner.QUALITY}}
        for label in runner.LABELS
    }
}
ENVIRONMENT = {"python": "3.12.6", "dependencies": {"numpy": "2.5.2"}}


def identity(runner_hash="runner-pilot", pyvbmc_hash="core-a"):
    return {
        "sources": {
            runner.RUNNER_SOURCE_KEY: runner_hash,
            "pyvbmc/vbmc/active_sample.py": pyvbmc_hash,
            "dev/scripts/data/timing.npz": "data-a",
        },
        "environment": copy.deepcopy(ENVIRONMENT),
    }


def manifest(ready=True, selection=None):
    semantic = {
        "schema_version": 1,
        "kind": "noisy_acq_e5_inference",
        "arms": {
            "S0": {"description": "unmodified production search"},
            "S2": copy.deepcopy(runner.S2_CONFIG),
        },
        "labels": list(runner.LABELS),
        "full_seeds": list(runner.FULL_SEEDS),
        "pilot_seeds": list(runner.PILOT_SEEDS),
        "full_design_fit_count": 120,
        "pilot_fit_count": 24,
        "cells": runner._cells(),
        "executable_cell_ids": [
            c["cell_id"] for c in runner._cells() if c["phase"] == "pilot"
        ],
        "validation_replay": {
            "cell_id": runner.cell_id("rosenbrock_D2_noise1", 2000, "S2"),
            "purpose": "operational reproducibility check before the remaining pilot fits",
            "excluded_from_all_scientific_and_timing_denominators": True,
        },
        "options": {},
        "target_contracts": copy.deepcopy(TARGET_CONTRACTS),
        "selection": {
            "path": str(selection) if selection else "unused",
            "raw_sha256": "unused",
            "finalist_config": copy.deepcopy(runner.S2_CONFIG),
        },
        "identity": identity(),
        "planning": {"pilot_operational_ceiling_seconds": 10800},
        "reference": copy.deepcopy(REFERENCE),
        "measurement": {"search_seconds": "InferencePolicy"},
    }
    return {
        "semantic": semantic,
        "semantic_digest": runner._digest(semantic),
        "reviewed_ready": {"approved": ready},
    }


def continuation_manifest(parent, parent_path, parent_campaign, ready=True):
    """Structurally valid continuation fixture bound to ``parent``."""
    cells = runner._cells()
    semantic = {
        "schema_version": 1,
        "kind": runner.CONTINUATION_KIND,
        "arms": copy.deepcopy(parent["semantic"]["arms"]),
        "labels": list(runner.LABELS),
        "full_seeds": list(runner.FULL_SEEDS),
        "pilot_seeds": list(runner.PILOT_SEEDS),
        "continuation_seeds": list(runner.CONTINUATION_SEEDS),
        "full_design_fit_count": 120,
        "pilot_fit_count": 24,
        "continuation_fit_count": 96,
        "cells": cells,
        "executable_cell_ids": [
            c["cell_id"] for c in cells if c["phase"] == "continuation"
        ],
        "validation_replay": None,
        "options": {},
        "target_contracts": copy.deepcopy(TARGET_CONTRACTS),
        "selection": copy.deepcopy(parent["semantic"]["selection"]),
        "identity": identity(runner_hash="runner-continuation"),
        "identity_provenance": {
            "allowed_changed_sources": [runner.RUNNER_SOURCE_KEY],
            "changed_sources": {
                runner.RUNNER_SOURCE_KEY: {
                    "parent": "runner-pilot",
                    "current": "runner-continuation",
                }
            },
            "source_keys_identical": True,
            "unchanged_source_count": 2,
            "environment_unchanged": True,
        },
        "parent": {
            "path": str(parent_path),
            "raw_sha256": (
                runner.sha256(parent_path)
                if Path(parent_path).is_file()
                else "absent"
            ),
            "semantic_digest": parent["semantic_digest"],
            "campaign_dir": str(parent_campaign),
            "record_sha256": {
                name: (
                    runner.sha256(Path(parent_path).parent / name)
                    if (Path(parent_path).parent / name).is_file()
                    else "absent"
                )
                for name in runner.PARENT_RECORDS
            },
        },
        "reference": copy.deepcopy(REFERENCE),
        "measurement": {"search_seconds": "InferencePolicy"},
        "planning": {
            "minimum_per_fit_timeout_seconds": 1200,
            "batch_operational_ceiling_seconds": 21600,
        },
    }
    return {
        "semantic": semantic,
        "semantic_digest": runner._digest(semantic),
        "reviewed_ready": {"approved": ready},
    }


def redigest(prepared):
    prepared["semantic_digest"] = runner._digest(prepared["semantic"])
    return prepared


def cell(label=None, seed=2000, arm="S0"):
    label = label or runner.LABELS[0]
    return next(
        c
        for c in runner._cells()
        if c["label"] == label and c["seed"] == seed and c["arm"] == arm
    )


def write_success(
    prepared, campaign, one_cell, x_init=None, y_init=None, final=None
):
    paths = runner._paths(campaign, one_cell)
    paths["arm_dir"].mkdir(parents=True, exist_ok=True)
    x_init = np.array([[0.0, 1.0]]) if x_init is None else np.asarray(x_init)
    y_init = np.array([2.0]) if y_init is None else np.asarray(y_init)
    np.savez_compressed(paths["npz"], X_init=x_init, y_init=y_init)
    final = {
        "success_flag": True,
        "wall_s": 4.0,
        "target_eval_s": 0.5,
        "func_count": 100,
        "elbo_err": 0.2,
        "gskl": 0.3,
        "mmtv": 0.1,
        **(final or {}),
    }
    runner.atomic_json(
        paths["json"],
        {
            "label": one_cell["label"],
            "seed": one_cell["seed"],
            "requested_options": {
                "display": "off",
                "plot": False,
                "print_iteration_header": False,
                "performance_calibration": "off",
            },
            "effective_options": {
                "display": "off",
                "plot": False,
                "print_iteration_header": False,
                "performance_calibration": "off",
            },
            "final": final,
        },
    )
    record = {
        "status": "complete",
        "arm": one_cell["arm"],
        "label": one_cell["label"],
        "selection_index": 0,
        "search_seconds": 1.0,
        "accurate_seed": 123 if one_cell["arm"] == "S2" else None,
    }
    runner.atomic_json(
        paths["selection"],
        {
            "cell": one_cell,
            "records": [record],
            "summary": {
                "arm": one_cell["arm"],
                "label": one_cell["label"],
                "selection_count": 1,
                "complete_selection_count": 1,
                "failed_selection_count": 0,
                "total_search_seconds": 1.0,
            },
        },
    )
    terminal = runner._terminal_record(
        prepared,
        one_cell,
        campaign,
        "success",
        (paths["npz"], paths["json"], paths["selection"]),
        outer_wall_s=5.0,
    )
    runner.atomic_json(paths["terminal"], terminal)


def test_manifest_locks_frozen_config_and_pilot_allocation():
    prepared = manifest()
    runner.validate_manifest(prepared, require_ready=True)

    changed = copy.deepcopy(prepared)
    changed["semantic"]["arms"]["S2"]["sieve_size"] = 2048
    changed["semantic_digest"] = runner._digest(changed["semantic"])
    with pytest.raises(RuntimeError, match="S2 configuration"):
        runner.validate_manifest(changed)

    expanded = copy.deepcopy(prepared)
    expanded["semantic"]["executable_cell_ids"].append(
        runner.cell_id(runner.LABELS[0], 2002, "S0")
    )
    expanded["semantic_digest"] = runner._digest(expanded["semantic"])
    with pytest.raises(RuntimeError, match="24-fit pilot"):
        runner.validate_manifest(expanded)

    with pytest.raises(RuntimeError, match="outside the executable"):
        runner._cell(prepared, runner.cell_id(runner.LABELS[0], 2002, "S0"))


def test_partial_case_is_refused_before_subprocess(tmp_path):
    prepared = manifest()
    manifest_path = tmp_path / "manifest.json"
    runner.atomic_json(manifest_path, prepared)
    campaign = tmp_path / "campaign"
    one_cell = cell()
    paths = runner._paths(campaign, one_cell)
    paths["arm_dir"].mkdir(parents=True)
    paths["json"].write_text("partial", encoding="utf-8")

    with pytest.raises(RuntimeError, match="partial prior attempt"):
        runner.run_cell(manifest_path, campaign, one_cell["cell_id"], 1200)


def test_terminal_rejects_artifact_and_record_tampering(tmp_path):
    prepared = manifest()
    campaign = tmp_path / "campaign"
    one_cell = cell()
    write_success(prepared, campaign, one_cell)
    runner.validate_terminal(prepared, campaign, one_cell["cell_id"])

    paths = runner._paths(campaign, one_cell)
    side = json.loads(paths["json"].read_text())
    side["final"]["wall_s"] = 99
    runner.atomic_json(paths["json"], side)
    with pytest.raises(RuntimeError, match="changed or missing artifact"):
        runner.validate_terminal(prepared, campaign, one_cell["cell_id"])

    write_success(prepared, campaign, one_cell)
    terminal = json.loads(paths["terminal"].read_text())
    terminal["outer_wall_s"] = 99
    runner.atomic_json(paths["terminal"], terminal)
    with pytest.raises(RuntimeError, match="terminal record was changed"):
        runner.validate_terminal(prepared, campaign, one_cell["cell_id"])


def test_summary_keeps_missing_and_failure_denominators(tmp_path):
    prepared = manifest()
    campaign = tmp_path / "campaign"
    s0, s2 = cell(arm="S0"), cell(arm="S2")
    write_success(prepared, campaign, s0)
    paths = runner._paths(campaign, s2)
    runner._write_failure(
        prepared,
        s2,
        campaign,
        "RuntimeError: deliberate",
        runner._utc_now(),
        paths,
    )

    report = runner.summarize(prepared, campaign)
    assert report["scheduled_fits"] == 24
    assert report["planned_continuation"]["fit_count"] == 96
    assert report["fit_status_counts"] == {
        "success": 1,
        "failure": 1,
        "missing": 22,
    }
    assert report["scheduled_pairs"] == 12
    assert report["pair_status_counts"] == {
        "complete": 0,
        "failure": 1,
        "missing": 11,
        "invalid_initial_design": 0,
        "nonfinite": 0,
    }
    first = report["targets"][runner.LABELS[0]]
    assert (
        first["log_ratio_intervals_S2_over_S0"]["wall_s"]["n_available_pairs"]
        == 0
    )
    assert (
        first["log_ratio_intervals_S2_over_S0"]["wall_s"]["n_scheduled_pairs"]
        == 2
    )
    assert first["convergence"]["excluded"] == {
        "missing": 1,
        "failure": 1,
        "invalid_initial_design": 0,
        "nonfinite": 0,
    }
    assert report["scientific_review"]["new_failure"] is True


def test_summary_flags_exact_initial_design_mismatch(tmp_path):
    prepared = manifest()
    campaign = tmp_path / "campaign"
    s0, s2 = cell(arm="S0"), cell(arm="S2")
    write_success(prepared, campaign, s0, [[0.0, 1.0]], [2.0])
    write_success(
        prepared, campaign, s2, [[0.0, 1.0]], [np.nextafter(2.0, 3.0)]
    )

    report = runner.summarize(prepared, campaign)
    assert report["pair_status_counts"]["invalid_initial_design"] == 1
    assert report["initial_design_mismatches"] == [
        {"label": runner.LABELS[0], "seed": 2000}
    ]
    pair = report["targets"][runner.LABELS[0]]["pairs"][0]
    assert pair["initial_design_equal"] is False
    interval = report["targets"][runner.LABELS[0]][
        "log_ratio_intervals_S2_over_S0"
    ]["wall_s"]
    assert interval["n_available_pairs"] == 0
    assert interval["df"] is None
    assert interval["ci95"] is None


def test_pilot_stops_before_remaining_cells_until_replay(
    tmp_path, monkeypatch
):
    prepared = manifest()
    manifest_path = tmp_path / "manifest.json"
    runner.atomic_json(manifest_path, prepared)
    campaign = tmp_path / "campaign"
    launched = []

    def fake_cell(_manifest_path, _out, requested, _timeout):
        launched.append(requested)
        return 0

    monkeypatch.setattr(runner, "_run_cell_locked", fake_cell)
    assert runner.run_pilot(manifest_path, campaign, 1200) == 1
    assert launched == prepared["semantic"]["executable_cell_ids"][:2]
    status = json.loads((campaign / "status.json").read_text())
    assert status["stop_reason"] == "validation_replay_required"
    assert status["cells"][-1]["status"] == (
        "not_launched_validation_replay_required"
    )


def test_validation_replay_obeys_primary_campaign_lock(tmp_path):
    prepared = manifest()
    manifest_path = tmp_path / "manifest.json"
    runner.atomic_json(manifest_path, prepared)
    campaign = tmp_path / "campaign"
    validation = cell(arm="S2")
    write_success(prepared, campaign, validation)
    lock_path = campaign / "execution.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with runner.FileLock(str(lock_path), timeout=0):
        with pytest.raises(RuntimeError, match="supervisor is active"):
            runner.run_validation_replay(manifest_path, campaign, 1200)


def test_validation_replay_mismatch_returns_failure(tmp_path, monkeypatch):
    prepared = manifest()
    manifest_path = tmp_path / "manifest.json"
    runner.atomic_json(manifest_path, prepared)
    campaign = tmp_path / "campaign"
    validation = cell(arm="S2")
    write_success(prepared, campaign, validation, [[0.0, 1.0]], [2.0])
    write_success(
        prepared,
        campaign / "validation_replay",
        validation,
        [[0.0, 2.0]],
        [2.0],
    )
    monkeypatch.setattr(runner, "_run_cell_locked", lambda *args: 0)

    assert runner.run_validation_replay(manifest_path, campaign, 1200) == 1
    with pytest.raises(RuntimeError, match="does not exactly match"):
        runner.validate_replay_report(prepared, campaign)


def test_validation_report_tampering_is_rejected(tmp_path, monkeypatch):
    prepared = manifest()
    manifest_path = tmp_path / "manifest.json"
    runner.atomic_json(manifest_path, prepared)
    campaign = tmp_path / "campaign"
    validation = cell(arm="S2")
    write_success(prepared, campaign, validation)
    write_success(prepared, campaign / "validation_replay", validation)
    monkeypatch.setattr(runner, "_run_cell_locked", lambda *args: 0)

    assert runner.run_validation_replay(manifest_path, campaign, 1200) == 0
    runner.validate_replay_report(prepared, campaign)
    report_path = campaign / "validation_replay" / "validation_replay.json"
    report = json.loads(report_path.read_text())
    report["exact_non_timing_match"] = False
    runner.atomic_json(report_path, report)
    with pytest.raises(RuntimeError, match="report was changed"):
        runner.validate_replay_report(prepared, campaign)


def complete_pilot(tmp_path, monkeypatch, final_by_arm=None):
    """Write a complete, reviewed 24-fit pilot tree with immutable records."""
    root = tmp_path / "pilot"
    root.mkdir()
    selection = tmp_path / "selection.json"
    selection.write_text('{"frozen": true}', encoding="utf-8")
    monkeypatch.setattr(runner, "SELECTION_SHA256", runner.sha256(selection))
    prepared = manifest(selection=selection)
    manifest_path = root / "manifest.json"
    runner.atomic_json(manifest_path, prepared)
    campaign = root / "campaign"
    for one_cell in runner._cells():
        if one_cell["phase"] == "pilot":
            write_success(
                prepared,
                campaign,
                one_cell,
                final=(final_by_arm or {}).get(one_cell["arm"]),
            )
    replay_cell = cell(arm="S2")
    write_success(prepared, campaign / "validation_replay", replay_cell)
    monkeypatch.setattr(runner, "_run_cell_locked", lambda *args: 0)
    assert runner.run_validation_replay(manifest_path, campaign, 1200) == 0
    summary = runner.summarize(prepared, campaign)
    runner.atomic_json(root / "summary.json", summary)
    runner.atomic_json(
        root / "launch_clearance.json",
        {
            "manifest_semantic_digest": prepared["semantic_digest"],
            "manifest_sha256": runner.sha256(manifest_path),
        },
    )
    runner.atomic_json(
        root / "completion.json",
        {
            "manifest_semantic_digest": prepared["semantic_digest"],
            "manifest_sha256": runner.sha256(manifest_path),
            "summary_sha256": runner.sha256(root / "summary.json"),
            "fit_status_counts": summary["fit_status_counts"],
            "pilot_worker_seconds": 120.0,
            "remaining_96_fits_point_estimate_seconds": 480.0,
            "remaining_96_fits_planning_seconds": [600.0, 720.0],
            "cost_estimate_method": "test",
        },
    )
    runner.atomic_json(
        root / "independent_review.json",
        {
            "manifest_semantic_digest": prepared["semantic_digest"],
            "verdict": "pass",
            "remaining_findings": [],
            "operational_continuation_criteria_satisfied": True,
            "artifact_sha256": {
                "summary.json": runner.sha256(root / "summary.json")
            },
        },
    )
    return prepared, manifest_path, campaign


def prepare_continuation(tmp_path, monkeypatch, current=None):
    prepared, manifest_path, campaign = complete_pilot(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "runtime_identity",
        lambda: current or identity(runner_hash="runner-continuation"),
    )
    monkeypatch.setattr(
        runner, "_target_contracts", lambda: copy.deepcopy(TARGET_CONTRACTS)
    )
    monkeypatch.setattr(
        runner, "_reference_envelopes", lambda *a: copy.deepcopy(REFERENCE)
    )
    monkeypatch.setattr(runner, "_git", lambda *a: "deadbeef")
    continuation = runner.prepare_continuation_manifest(
        manifest_path, campaign
    )
    return prepared, manifest_path, campaign, continuation


def test_continuation_manifest_allocates_exactly_the_remaining_fits(
    tmp_path,
):
    parent = manifest()
    prepared = continuation_manifest(
        parent, tmp_path / "pilot" / "manifest.json", tmp_path / "campaign"
    )
    runner.validate_manifest(prepared, require_ready=True)
    allowed = prepared["semantic"]["executable_cell_ids"]
    assert len(allowed) == 96
    assert {
        c["seed"] for c in runner._cells() if c["cell_id"] in allowed
    } == set(range(2002, 2010))
    assert runner._cell(prepared, runner.cell_id(runner.LABELS[0], 2002, "S0"))
    with pytest.raises(RuntimeError, match="outside the executable"):
        runner._cell(prepared, runner.cell_id(runner.LABELS[0], 2000, "S0"))
    with pytest.raises(RuntimeError, match="outside the executable"):
        runner._cell(parent, runner.cell_id(runner.LABELS[0], 2002, "S0"))

    widened = copy.deepcopy(prepared)
    widened["semantic"]["executable_cell_ids"].insert(
        0, runner.cell_id(runner.LABELS[0], 2000, "S0")
    )
    with pytest.raises(RuntimeError, match="96 remaining fits"):
        runner.validate_manifest(redigest(widened))

    narrowed = copy.deepcopy(prepared)
    narrowed["semantic"]["executable_cell_ids"].pop()
    with pytest.raises(RuntimeError, match="96 remaining fits"):
        runner.validate_manifest(redigest(narrowed))

    replayed = copy.deepcopy(prepared)
    replayed["semantic"]["validation_replay"] = {"cell_id": "x"}
    with pytest.raises(RuntimeError, match="no validation replay"):
        runner.validate_manifest(redigest(replayed))

    loosened = copy.deepcopy(prepared)
    loosened["semantic"]["identity_provenance"]["changed_sources"][
        "pyvbmc/vbmc/active_sample.py"
    ] = {"parent": "a", "current": "b"}
    with pytest.raises(RuntimeError, match="more than the runner"):
        runner.validate_manifest(redigest(loosened))

    raised = copy.deepcopy(prepared)
    raised["semantic"]["planning"]["batch_operational_ceiling_seconds"] = 1
    with pytest.raises(RuntimeError, match="batch limits"):
        runner.validate_manifest(redigest(raised))

    unknown = copy.deepcopy(prepared)
    unknown["semantic"]["kind"] = "noisy_acq_e5_other"
    with pytest.raises(RuntimeError, match="Unknown E5 manifest kind"):
        runner.validate_manifest(redigest(unknown))


def test_identity_provenance_admits_only_the_runner():
    parent = identity()
    unchanged = runner.identity_provenance(parent, identity())
    assert unchanged["changed_sources"] == {}
    assert unchanged["unchanged_source_count"] == 3

    runner_only = runner.identity_provenance(
        parent, identity(runner_hash="runner-continuation")
    )
    assert set(runner_only["changed_sources"]) == {runner.RUNNER_SOURCE_KEY}
    assert runner_only["environment_unchanged"] is True

    with pytest.raises(RuntimeError, match="Numerical/helper/data source"):
        runner.identity_provenance(parent, identity(pyvbmc_hash="core-b"))

    extra = identity()
    extra["sources"]["pyvbmc/new_module.py"] = "x"
    with pytest.raises(RuntimeError, match="Source inventory differs"):
        runner.identity_provenance(parent, extra)

    environment = identity()
    environment["environment"]["dependencies"]["numpy"] = "2.6.0"
    with pytest.raises(RuntimeError, match="environment differs"):
        runner.identity_provenance(parent, environment)


def test_run_continuation_reuses_completed_and_obeys_limits(
    tmp_path, monkeypatch
):
    parent = manifest()
    parent_path = tmp_path / "pilot" / "manifest.json"
    prepared = continuation_manifest(parent, parent_path, tmp_path / "pc")
    manifest_path = tmp_path / "continuation.json"
    runner.atomic_json(manifest_path, prepared)
    campaign = tmp_path / "campaign"
    allowed = prepared["semantic"]["executable_cell_ids"]
    for requested in allowed[:2]:
        write_success(prepared, campaign, runner._cell(prepared, requested))
    monkeypatch.setattr(runner, "_load_parent", lambda m: parent)
    monkeypatch.setattr(runner, "verify_parent_campaign", lambda *a: {})
    launched = []

    def fake_cell(_manifest_path, out, requested, _timeout):
        launched.append(requested)
        write_success(prepared, out, runner._cell(prepared, requested))
        return 0

    monkeypatch.setattr(runner, "_run_cell_locked", fake_cell)

    code = runner.run_continuation(
        manifest_path, campaign, 1200, 7200, max_fits=2
    )
    assert code == 2
    assert launched == allowed[2:4]
    batches = sorted((campaign / "batches").glob("batch_*.json"))
    assert len(batches) == 1
    record = json.loads(batches[0].read_text())
    assert record["stop_reason"] == "fit_limit_reached"
    assert record["state_before"] == {
        "success": 2,
        "failure": 0,
        "missing": 94,
    }
    assert record["state_after"] == {
        "success": 4,
        "failure": 0,
        "missing": 92,
    }
    assert [c["action"] for c in record["cells"]] == [
        "reused",
        "reused",
        "launched",
        "launched",
        "not_launched",
    ]
    assert record["complete"] is False
    status = json.loads((campaign / "status.json").read_text())
    assert status["stop_reason"] == "fit_limit_reached"

    launched.clear()
    clock = itertools.count(0, 5)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    code = runner.run_continuation(manifest_path, campaign, 1200, 1201)
    assert code == 2 and launched == []
    latest = json.loads(
        sorted((campaign / "batches").glob("batch_*.json"))[-1].read_text()
    )
    assert latest["stop_reason"] == "insufficient_time_for_next_fit"
    assert [c["action"] for c in latest["cells"]].count("reused") == 4

    with pytest.raises(RuntimeError, match="frozen ceiling"):
        runner.run_continuation(manifest_path, campaign, 1200, 21601)
    with pytest.raises(RuntimeError, match="at least 1200"):
        runner.run_continuation(manifest_path, campaign, 600, 7200)
    with pytest.raises(RuntimeError, match="on_failure"):
        runner.run_continuation(
            manifest_path, campaign, 1200, 7200, on_failure="retry"
        )
    pilot_path = tmp_path / "pilot_manifest.json"
    runner.atomic_json(pilot_path, parent)
    with pytest.raises(RuntimeError, match="only the continuation manifest"):
        runner.run_continuation(pilot_path, campaign, 1200, 7200)
    with pytest.raises(RuntimeError, match="only the pilot manifest"):
        runner.run_pilot(manifest_path, campaign, 1200)


def test_run_continuation_stops_at_failures_unless_told_to_continue(
    tmp_path, monkeypatch
):
    parent = manifest()
    prepared = continuation_manifest(
        parent, tmp_path / "pilot" / "manifest.json", tmp_path / "pc"
    )
    manifest_path = tmp_path / "continuation.json"
    runner.atomic_json(manifest_path, prepared)
    campaign = tmp_path / "campaign"
    allowed = prepared["semantic"]["executable_cell_ids"]
    monkeypatch.setattr(runner, "_load_parent", lambda m: parent)
    monkeypatch.setattr(runner, "verify_parent_campaign", lambda *a: {})
    launched = []

    def fake_cell(_manifest_path, out, requested, _timeout):
        launched.append(requested)
        one_cell = runner._cell(prepared, requested)
        if requested == allowed[1]:
            runner._write_failure(
                prepared,
                one_cell,
                out,
                "RuntimeError: deliberate",
                runner._utc_now(),
                runner._paths(out, one_cell),
            )
            return 1
        write_success(prepared, out, one_cell)
        return 0

    monkeypatch.setattr(runner, "_run_cell_locked", fake_cell)

    assert runner.run_continuation(manifest_path, campaign, 1200, 7200) == 1
    assert launched == allowed[:2]
    latest = json.loads(
        sorted((campaign / "batches").glob("batch_*.json"))[-1].read_text()
    )
    assert latest["stop_reason"] == "fit_failed"

    launched.clear()
    assert runner.run_continuation(manifest_path, campaign, 1200, 7200) == 1
    assert launched == []
    latest = json.loads(
        sorted((campaign / "batches").glob("batch_*.json"))[-1].read_text()
    )
    assert latest["stop_reason"] == "prior_failed_terminal"
    assert latest["cells"][-1]["action"] == "stopped_at_failed"

    launched.clear()
    code = runner.run_continuation(
        manifest_path, campaign, 1200, 7200, max_fits=2, on_failure="continue"
    )
    assert code == 1
    assert launched == allowed[2:4]
    latest = json.loads(
        sorted((campaign / "batches").glob("batch_*.json"))[-1].read_text()
    )
    assert latest["cells"][1]["action"] == "skipped_failed"
    assert latest["failures"] == 1
    assert latest["state_after"]["failure"] == 1
    report = runner.summarize(prepared, campaign)
    assert report["phase"] == "continuation"
    assert report["scheduled_fits"] == 96 and report["scheduled_pairs"] == 48
    assert report["fit_status_counts"] == {
        "success": 3,
        "failure": 1,
        "missing": 92,
    }
    assert report["pair_status_counts"]["failure"] == 1
    assert report["targets"][runner.LABELS[0]]["scheduled_pairs"] == 8
    assert report["scientific_review"]["new_failure"] is True


def test_prepare_continuation_binds_a_complete_reviewed_pilot(
    tmp_path, monkeypatch
):
    prepared, manifest_path, campaign, continuation = prepare_continuation(
        tmp_path, monkeypatch
    )
    runner.validate_manifest(continuation)
    assert continuation["reviewed_ready"]["approved"] is False
    semantic = continuation["semantic"]
    assert len(semantic["executable_cell_ids"]) == 96
    assert set(semantic["identity_provenance"]["changed_sources"]) == {
        runner.RUNNER_SOURCE_KEY
    }
    assert semantic["identity_provenance"]["unchanged_source_count"] == 2
    assert semantic["parent"]["semantic_digest"] == prepared["semantic_digest"]
    assert semantic["parent"]["raw_sha256"] == runner.sha256(manifest_path)
    assert set(semantic["parent"]["record_sha256"]) == set(
        runner.PARENT_RECORDS
    )
    assert semantic["planning"]["remaining_96_fits_planning_seconds"] == [
        600.0,
        720.0,
    ]
    assert runner._load_parent(continuation)["semantic_digest"] == (
        prepared["semantic_digest"]
    )

    monkeypatch.setattr(
        runner,
        "runtime_identity",
        lambda: identity(
            runner_hash="runner-continuation", pyvbmc_hash="core-b"
        ),
    )
    with pytest.raises(RuntimeError, match="Numerical/helper/data source"):
        runner.prepare_continuation_manifest(manifest_path, campaign)

    monkeypatch.setattr(
        runner,
        "runtime_identity",
        lambda: identity(runner_hash="runner-continuation"),
    )
    summary_path = manifest_path.parent / "summary.json"
    stored = json.loads(summary_path.read_text())
    stored["fit_status_counts"]["success"] = 23
    runner.atomic_json(summary_path, stored)
    with pytest.raises(RuntimeError, match="not a complete 24-fit record"):
        runner.prepare_continuation_manifest(manifest_path, campaign)
    with pytest.raises(RuntimeError, match="record changed"):
        runner._load_parent(continuation)

    missing = runner._paths(campaign, cell(arm="S0"))["terminal"]
    missing.unlink()
    with pytest.raises(RuntimeError, match="no terminal record"):
        runner.prepare_continuation_manifest(manifest_path, campaign)


def test_combined_summary_reports_ten_pairs_per_target(tmp_path, monkeypatch):
    prepared, manifest_path, campaign, continuation = prepare_continuation(
        tmp_path, monkeypatch
    )
    continuation["reviewed_ready"] = {"approved": True}
    out = tmp_path / "continuation_campaign"
    first, second = runner.LABELS[0], runner.LABELS[1]
    for one_cell in runner._executable_cells(continuation):
        if one_cell["label"] == first:
            final = None
            if one_cell["arm"] == "S2":
                final = {"elbo_err": 0.3, "wall_s": 2.0}
            write_success(continuation, out, one_cell, final=final)
        elif one_cell["label"] == second and one_cell["seed"] == 2002:
            write_success(continuation, out, one_cell)

    report = runner.summarize_combined(continuation, out)
    assert report["scheduled_fits"] == 120 and report["scheduled_pairs"] == 60
    assert report["fit_status_counts"] == {
        "success": 24 + 16 + 2,
        "failure": 0,
        "missing": 78,
    }
    assert report["pilot_consistency"]["mismatches"] == []
    assert report["pilot_consistency"]["pilot_pairs_checked"] == 12
    target = report["targets"][first]
    assert target["scheduled_pairs"] == 10
    assert target["status_counts"]["complete"] == 10
    assert target["pilot_pairs"] == 2 and target["continuation_pairs"] == 8
    assert [p["phase"] for p in target["pairs"]] == ["pilot"] * 2 + [
        "continuation"
    ] * 8
    wall = target["log_ratio_intervals_S2_over_S0"]["wall_s"]
    assert wall["n_available_pairs"] == 10 and wall["df"] == 9
    assert wall["ci95"][0] < wall["geometric_mean_ratio"] < wall["ci95"][1]
    accuracy = target["accuracy_differences"]["elbo_err"]
    assert accuracy["worse_count"] == 8 and accuracy["median"] > 0
    assert target["full_ten_pair_review_metrics"] == ["elbo_err"]
    assert report["scientific_review"]["full_ten_pair_accuracy_triggers"] == {
        first: ["elbo_err"]
    }
    assert report["scientific_review"]["incomplete_targets"] == sorted(
        runner.LABELS[1:]
    )
    other = report["targets"][second]
    assert other["status_counts"] == {
        "complete": 3,
        "failure": 0,
        "missing": 7,
        "invalid_initial_design": 0,
        "nonfinite": 0,
    }
    assert other["full_ten_pair_review_metrics"] == []
    pooled = report["pooled_equal_weight"]["wall_s"]
    assert pooled["targets_available"] == sorted(runner.LABELS)
    assert pooled["targets_scheduled"] == 6

    stored = json.loads((manifest_path.parent / "summary.json").read_text())
    stored["targets"][first]["pairs"][0]["status"] = "failure"
    runner.atomic_json(manifest_path.parent / "summary.json", stored)
    with pytest.raises(RuntimeError, match="record changed"):
        runner.summarize_combined(continuation, out)
    with pytest.raises(RuntimeError, match="needs the continuation manifest"):
        runner.summarize_combined(prepared, campaign)


def test_runtime_check_rederives_continuation_provenance(
    tmp_path, monkeypatch
):
    prepared, manifest_path, campaign, continuation = prepare_continuation(
        tmp_path, monkeypatch
    )
    continuation["reviewed_ready"] = {"approved": True}
    runner.validate_manifest(
        continuation, require_ready=True, check_runtime=True
    )

    recorded = copy.deepcopy(continuation)
    recorded["semantic"]["identity_provenance"]["unchanged_source_count"] = 1
    with pytest.raises(RuntimeError, match="provenance no longer holds"):
        runner.validate_manifest(
            redigest(recorded), require_ready=True, check_runtime=True
        )

    monkeypatch.setattr(
        runner,
        "runtime_identity",
        lambda: identity(runner_hash="runner-edited-after-approval"),
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        runner.validate_manifest(
            continuation, require_ready=True, check_runtime=True
        )

    unapproved = copy.deepcopy(prepared)
    unapproved["reviewed_ready"] = {"approved": False}
    runner.atomic_json(manifest_path, unapproved)
    with pytest.raises(RuntimeError, match="changed or is missing"):
        runner._load_parent(continuation)
    rebound = copy.deepcopy(continuation)
    rebound["semantic"]["parent"]["raw_sha256"] = runner.sha256(manifest_path)
    with pytest.raises(RuntimeError, match="not the reviewed pilot"):
        runner._load_parent(redigest(rebound))


def test_batch_verifies_parent_and_leaves_pilot_untouched(
    tmp_path, monkeypatch
):
    prepared, manifest_path, campaign, continuation = prepare_continuation(
        tmp_path, monkeypatch
    )
    continuation["reviewed_ready"] = {"approved": True}
    continuation_path = tmp_path / "continuation.json"
    runner.atomic_json(continuation_path, continuation)
    pilot_root = manifest_path.parent

    def snapshot():
        return {
            str(p.relative_to(pilot_root)): runner.sha256(p)
            for p in sorted(pilot_root.rglob("*"))
            if p.is_file()
        }

    before = snapshot()

    def fake_cell(_manifest_path, out, requested, _timeout):
        write_success(continuation, out, runner._cell(continuation, requested))
        return 0

    monkeypatch.setattr(runner, "_run_cell_locked", fake_cell)
    out = tmp_path / "continuation_campaign"
    assert (
        runner.run_continuation(continuation_path, out, 1200, 7200, max_fits=1)
        == 2
    )
    assert snapshot() == before
    assert not (out / "validation_replay").exists()

    def refuse(*args):
        raise RuntimeError("partial prior attempt exists")

    monkeypatch.setattr(runner, "_run_cell_locked", refuse)
    with pytest.raises(RuntimeError, match="partial prior attempt"):
        runner.run_continuation(continuation_path, out, 1200, 7200)
    latest = json.loads(
        sorted((out / "batches").glob("batch_*.json"))[-1].read_text()
    )
    assert latest["stop_reason"] == "aborted"
    assert "partial prior attempt" in latest["abort_error"]
    assert latest["state_after"] == {
        "success": 1,
        "failure": 0,
        "missing": 95,
    }
    assert snapshot() == before

    runner._paths(campaign, cell(arm="S0"))["terminal"].unlink()
    with pytest.raises(RuntimeError, match="no terminal record"):
        runner.run_continuation(continuation_path, out, 1200, 7200)
