import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import noisy_acq_inference as runner


def manifest(ready=True):
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
        "target_contracts": {
            label: {"standard_options": {}} for label in runner.LABELS
        },
        "planning": {"pilot_operational_ceiling_seconds": 10800},
        "reference": {
            "labels": {
                label: {
                    "envelopes": {metric: 10.0 for metric in runner.QUALITY}
                }
                for label in runner.LABELS
            }
        },
    }
    return {
        "semantic": semantic,
        "semantic_digest": runner._digest(semantic),
        "reviewed_ready": {"approved": ready},
    }


def cell(label=None, seed=2000, arm="S0"):
    label = label or runner.LABELS[0]
    return next(
        c
        for c in runner._cells()
        if c["label"] == label and c["seed"] == seed and c["arm"] == arm
    )


def write_success(prepared, campaign, one_cell, x_init=None, y_init=None):
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
