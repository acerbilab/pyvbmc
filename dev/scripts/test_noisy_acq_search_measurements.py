"""Protocol checks for versioned E3 search measurement reporting."""

import copy
import hashlib
import json
from types import SimpleNamespace

import noisy_acq_search_measurements as measurements
import numpy as np
import pytest


def _selection_result(arm="S3", seed=11):
    result = {
        "arm": arm,
        "selected": np.array([[1.25, -2.5]]),
        "elapsed_seconds": 0.125,
        "search_seed": seed,
        "accurate_seed": seed + 1,
        "target_called": False,
        "cache_index": np.nan,
        "cache_indices": np.array([np.nan, 4.0]),
        "shortlist_scores": np.array([np.inf, 2.0]),
        "coarse_winner": np.array([3.0, 4.0]),
        "generated_count": 1024,
        "coarse_candidate_rows": 1027,
        "production_candidate_rows": 1027,
    }
    if arm != "S0":
        result.update(
            accurate_candidate_rows=41,
            local_iterations=17,
            fallback_reason=None,
            refinement_stop_reason="iteration_budget",
        )
    return result


def _timing_report():
    rounds = []
    for index in range(7):
        baseline_seconds = 1.0 + index
        treatment_seconds = 2.0 + index
        rounds.append(
            {
                "round": index,
                "order": (
                    ["baseline", "treatment"]
                    if index % 2 == 0
                    else ["treatment", "baseline"]
                ),
                "paired_seed": 100 + index,
                "baseline": {
                    "seed": 100 + index,
                    "seconds": baseline_seconds,
                    "result": _selection_result("S0", 100 + index),
                },
                "treatment": {
                    "seed": 100 + index,
                    "seconds": treatment_seconds,
                    "result": _selection_result("S3", 100 + index),
                },
                "ratio_treatment_over_baseline": (
                    treatment_seconds / baseline_seconds
                ),
            }
        )
    return {
        "seed": 99,
        "thread_environment": {"OMP_NUM_THREADS": "1"},
        "warmup_seeds": {"baseline": 10, "treatment": 10},
        "rounds": rounds,
        "baseline_median_seconds": 4.0,
        "treatment_median_seconds": 5.0,
        "median_paired_ratio": 1.25,
        "paired_ratio_range": [1.125, 2.0],
        "quality_claim": False,
    }


def test_projection_is_compact_nonmutating_and_strict_json():
    source = _selection_result()
    before = copy.deepcopy(source)
    projected = measurements.project_selection_result(source, dimension=2)

    expected = (
        "x_"
        + hashlib.sha256(
            np.ascontiguousarray(source["selected"], dtype="<f8").tobytes()
        ).hexdigest()
    )
    assert projected["selected_row_id"] == expected
    assert projected["cache_index"] is None
    assert projected["accurate_candidate_rows"] == 41
    assert projected["local_iterations"] == 17
    assert projected["operation_elapsed_seconds"] == 0.125
    assert not {
        "selected",
        "cache_indices",
        "shortlist_scores",
        "coarse_winner",
    }.intersection(projected)
    json.dumps(measurements.capture.canonical(projected), allow_nan=False)
    assert source.keys() == before.keys()
    for name in ("selected", "cache_indices", "shortlist_scores"):
        np.testing.assert_equal(source[name], before[name])

    baseline = measurements.project_selection_result(
        _selection_result("S0"), dimension=2
    )
    assert "accurate_candidate_rows" not in baseline
    assert "local_iterations" not in baseline


def _cell_fixture(tmp_path, monkeypatch):
    timing_cell = {
        "state_id": "state",
        "treatment_tag": "S3",
        "timing_seed": 99,
    }
    allocation_cell = {
        **timing_cell,
        "role": "treatment",
        "allocation_seed": 101,
    }
    manifest = {
        "adapter": {"sha256": "adapter"},
        "parent_manifest_sha256": "parent",
        "timing_cells": [timing_cell],
        "allocation_cells": [allocation_cell],
    }
    manifest_path = tmp_path / "measurement.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    parent = {
        "states": [],
        "treatments": [
            {"tag": "S0", "config": {"arm": "S0"}},
            {"tag": "S3", "config": {"arm": "S3"}},
        ],
    }
    monkeypatch.setattr(
        measurements,
        "validate_manifest",
        lambda value, **kwargs: (parent, {"runtime": "frozen"}),
    )
    descriptor = {"state_id": "state", "snapshot": "snapshot"}
    monkeypatch.setattr(
        measurements.campaign, "_state_descriptor", lambda *args: descriptor
    )
    monkeypatch.setattr(
        measurements.campaign,
        "_treatment",
        lambda parent, tag: {"tag": tag, "config": {"arm": tag}},
    )
    state = {"vp": SimpleNamespace(D=2)}
    monkeypatch.setattr(
        measurements.capture, "restore_capture", lambda path: state
    )
    monkeypatch.setattr(
        measurements.campaign, "_state_digest", lambda value: "unchanged"
    )
    monkeypatch.setattr(
        measurements.campaign,
        "_timing_factory",
        lambda state, state_id, treatment: (state_id, treatment["tag"]),
    )
    return manifest_path, timing_cell, allocation_cell


def test_timing_cell_projects_after_measurement_and_refuses_existing(
    tmp_path, monkeypatch
):
    manifest_path, cell, _ = _cell_fixture(tmp_path, monkeypatch)
    report = _timing_report()
    events = []

    def time_pair(baseline, treatment, *, seed):
        events.append(("timed", baseline, treatment, seed))
        return copy.deepcopy(report)

    original_project = measurements.project_timing_report

    def project(value, *, dimension):
        events.append(("projected", dimension))
        return original_project(value, dimension=dimension)

    monkeypatch.setattr(measurements.timing, "time_pair", time_pair)
    monkeypatch.setattr(measurements, "project_timing_report", project)
    old = tmp_path / "timing" / "old.failure.json"
    old.parent.mkdir()
    old.write_text("old-v1", encoding="utf-8")

    terminal = measurements.run_timing_cell(manifest_path, tmp_path, cell)
    assert terminal["status"] == "succeeded"
    assert [event[0] for event in events] == ["timed", "projected"]
    path = measurements._paths(
        tmp_path, measurements.campaign.timing_tag(cell), "timing"
    )["payload"]
    saved = json.loads(path.read_text(encoding="utf-8"))["timing"]
    assert saved["seed"] == report["seed"]
    assert saved["rounds"][1]["order"] == report["rounds"][1]["order"]
    assert saved["rounds"][2]["paired_seed"] == 102
    assert saved["rounds"][3]["baseline"]["seconds"] == 4.0
    assert saved["rounds"][4]["ratio_treatment_over_baseline"] == (
        report["rounds"][4]["ratio_treatment_over_baseline"]
    )
    assert old.read_text(encoding="utf-8") == "old-v1"
    reused = measurements.run_timing_cell(manifest_path, tmp_path, cell)
    assert reused == terminal
    assert [event[0] for event in events] == ["timed", "projected"]
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="payload changed"):
        measurements.run_timing_cell(manifest_path, tmp_path, cell)

    foreign = tmp_path / "foreign"
    paths = measurements._paths(
        foreign, measurements.campaign.timing_tag(cell), "timing"
    )
    paths["payload"].parent.mkdir(parents=True)
    paths["payload"].write_text("foreign", encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial or foreign"):
        measurements.run_timing_cell(manifest_path, foreign, cell)

    temporary = tmp_path / "temporary"
    paths = measurements._paths(
        temporary, measurements.campaign.timing_tag(cell), "timing"
    )
    temp = paths["payload"].with_name(paths["payload"].name + ".tmp")
    temp.parent.mkdir(parents=True)
    temp.write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match="temporary"):
        measurements.run_timing_cell(manifest_path, temporary, cell)


def test_allocation_cell_preserves_memory_measurements_and_projects_after(
    tmp_path, monkeypatch
):
    manifest_path, _, cell = _cell_fixture(tmp_path, monkeypatch)
    events = []
    raw = {
        "seed": 101,
        "result": _selection_result("S3", 101),
        "traced_current_bytes": 1200,
        "traced_peak_bytes": 3400,
        "windows_before": {"resident_bytes": 5000},
        "windows_after": {"resident_bytes": 5500},
        "timing_evidence": False,
    }

    def measure(factory, *, seed):
        events.append(("measured", factory, seed))
        return copy.deepcopy(raw)

    original_project = measurements.project_allocation_report

    def project(value, *, dimension):
        events.append(("projected", dimension))
        return original_project(value, dimension=dimension)

    monkeypatch.setattr(measurements.timing, "measure_allocations", measure)
    monkeypatch.setattr(measurements, "project_allocation_report", project)
    terminal = measurements.run_allocation_cell(manifest_path, tmp_path, cell)

    assert terminal["status"] == "succeeded"
    assert [event[0] for event in events] == ["measured", "projected"]
    path = measurements._paths(
        tmp_path, measurements.campaign.allocation_tag(cell), "allocation"
    )["payload"]
    saved = json.loads(path.read_text(encoding="utf-8"))["allocation"]
    assert saved["seed"] == 101
    assert saved["traced_current_bytes"] == 1200
    assert saved["traced_peak_bytes"] == 3400
    assert saved["windows_before"] == {"resident_bytes": 5000}
    assert saved["windows_after"] == {"resident_bytes": 5500}
    assert saved["result"]["selected_row_id"].startswith("x_")
    np.testing.assert_equal(raw["result"]["cache_indices"], [np.nan, 4.0])


def test_cell_refuses_manifest_bytes_changed_during_measurement(
    tmp_path, monkeypatch
):
    manifest_path, cell, _ = _cell_fixture(tmp_path, monkeypatch)

    def time_pair(*args, **kwargs):
        report = _timing_report()
        manifest_path.write_text(
            manifest_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(measurements.timing, "time_pair", time_pair)
    with pytest.raises(RuntimeError, match="manifest changed during cell"):
        measurements.run_timing_cell(manifest_path, tmp_path, cell)
    paths = measurements._paths(
        tmp_path, measurements.campaign.timing_tag(cell), "timing"
    )
    assert not any(path.exists() for path in paths.values())


def test_manifest_refuses_parent_snapshot_adapter_and_allocation_drift(
    tmp_path, monkeypatch
):
    timing_cells = [
        {"state_id": f"state{i}", "treatment_tag": "S1", "timing_seed": i}
        for i in range(measurements.EXPECTED_TIMING_CELLS)
    ]
    timing_cells[0] = copy.deepcopy(measurements.FAILED_V1_CELL)
    allocation_cells = [
        {
            **timing_cells[index // 2],
            "role": "baseline" if index % 2 == 0 else "treatment",
            "allocation_seed": 1000 + index,
        }
        for index in range(measurements.EXPECTED_ALLOCATION_CELLS)
    ]
    source_hashes = {"frozen.py": "abc"}
    parent = {
        "identity": {"search_source_hashes": source_hashes},
        "thread_environment": {"OMP_NUM_THREADS": "1"},
        "timing_cells": timing_cells,
        "allocation_cells": allocation_cells,
        "cells": [
            {
                "state_id": f"selection{index // 32}",
                "treatment_tag": f"S{index % 4}",
                "replicate": (index // 4) % 8,
            }
            for index in range(measurements.EXPECTED_SELECTION_TERMINALS)
        ],
    }
    parent_path = tmp_path / "parent.json"
    parent_text = json.dumps(parent)
    parent_path.write_text(parent_text, encoding="utf-8")
    parent_sha = measurements._sha256(parent_path)
    snapshot = {
        "kind": "noisy_acq_search_selection_completion_snapshot",
        "manifest_sha256": parent_sha,
        "counts": {
            "expected": 384,
            "failed": 0,
            "missing": 0,
            "succeeded": 384,
        },
        "terminals": [
            {"tag": measurements.campaign.cell_tag(cell)}
            for cell in parent["cells"]
        ],
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_text = json.dumps(snapshot)
    snapshot_path.write_text(snapshot_text, encoding="utf-8")
    failed_terminal = tmp_path / "old.complete.json"
    failed_artifact = tmp_path / "old.failure.json"
    failed_artifact.write_text(
        json.dumps({"status": "failed", "cell": measurements.FAILED_V1_CELL})
    )
    failed_terminal.write_text(
        json.dumps(
            {
                "status": "failed",
                "cell": measurements.FAILED_V1_CELL,
                "manifest_sha256": measurements.integration.manifest_digest(
                    parent
                ),
                "hashes": {"failure": measurements._sha256(failed_artifact)},
            }
        )
    )
    runtime = {
        "integration_identity": {
            "capture_identity": {"gpyreg": {"commit": "frozen"}}
        }
    }
    monkeypatch.setattr(
        measurements.campaign, "validate_manifest", lambda *args, **kw: None
    )
    monkeypatch.setattr(
        measurements.campaign, "runtime_identity", lambda value: runtime
    )
    monkeypatch.setattr(
        measurements, "_adapter_sha256", lambda: "adapter-frozen"
    )
    manifest = measurements.prepare_manifest(
        parent_path, snapshot_path, failed_terminal, failed_artifact
    )
    manifest["launch_ready"] = True
    measurements.validate_manifest(manifest, require_ready=True)

    parent_path.write_text(parent_text + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent search manifest"):
        measurements.validate_manifest(manifest, require_ready=True)
    parent_path.write_text(parent_text, encoding="utf-8")
    snapshot_path.write_text(snapshot_text + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="snapshot"):
        measurements.validate_manifest(manifest, require_ready=True)
    snapshot_path.write_text(snapshot_text, encoding="utf-8")
    monkeypatch.setattr(measurements, "_adapter_sha256", lambda: "changed")
    with pytest.raises(RuntimeError, match="adapter"):
        measurements.validate_manifest(manifest, require_ready=True)
    monkeypatch.setattr(
        measurements, "_adapter_sha256", lambda: "adapter-frozen"
    )
    drifted = copy.deepcopy(manifest)
    drifted["allocation_cells"].pop()
    with pytest.raises(RuntimeError, match="72-cell"):
        measurements.validate_manifest(drifted, require_ready=True)
