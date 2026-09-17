"""Checks for pairing, warmup and allocation-measurement separation."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from noisy_acq_timing import measure_allocations, time_pair


def pin_threads(monkeypatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")


def test_timing_preparation_is_outside_interval_and_rounds_alternate(
    monkeypatch,
):
    pin_threads(monkeypatch)
    events = []
    ticks = iter(range(28))

    def factory(arm):
        def prepare(seed):
            events.append(("prepare", arm, seed))

            def run():
                events.append(("run", arm, seed))
                return {"seed": seed}

            return run

        return prepare

    def clock():
        events.append(("clock",))
        return next(ticks)

    report = time_pair(
        factory("baseline"), factory("treatment"), seed=18, clock=clock
    )
    assert len(report["rounds"]) == 7
    assert report["median_paired_ratio"] == 1
    assert len({event[2] for event in events if event[0] == "prepare"}) == 8
    for index, row in enumerate(report["rounds"]):
        assert row["order"][0] == ("treatment" if index % 2 else "baseline")
    timed = events[4:]
    assert len(timed) == 14 * 4
    for start in range(0, len(timed), 4):
        assert [event[0] for event in timed[start : start + 4]] == [
            "prepare",
            "clock",
            "run",
            "clock",
        ]


def test_allocation_measurement_is_separate_from_timing(monkeypatch):
    pin_threads(monkeypatch)

    def prepare(seed):
        def run():
            block = np.ones(100_000)
            return {"sum": float(block.sum())}

        return run

    result = measure_allocations(prepare, seed=19)
    assert result["traced_peak_bytes"] >= 800_000
    assert not result["timing_evidence"]


def test_timing_refuses_unpinned_threads(monkeypatch):
    pin_threads(monkeypatch)
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="thread"):
        time_pair(lambda _: None, lambda _: None, seed=1)


def test_completed_timing_rejects_status_identity_and_report_corruption(
    tmp_path,
):
    import noisy_acq_integration as integration
    import noisy_acq_timing as timing

    identity = {"source": "frozen"}
    report_path = tmp_path / "report.json"
    complete_path = tmp_path / "complete.json"
    integration.write_json(
        report_path, {"identity": identity, "status": "failed", "result": {}}
    )
    completion = {
        "identity": identity,
        "status": "failed",
        "report_sha256": integration.sha256_file(report_path),
    }
    integration.write_json(complete_path, completion)
    assert (
        timing._read_completed(report_path, complete_path, identity)[0][
            "status"
        ]
        == "failed"
    )
    completion["status"] = "succeeded"
    integration.write_json(complete_path, completion)
    with pytest.raises(RuntimeError, match="inconsistent"):
        timing._read_completed(report_path, complete_path, identity)
    completion["status"] = "failed"
    integration.write_json(complete_path, completion)
    with pytest.raises(RuntimeError, match="identity"):
        timing._read_completed(report_path, complete_path, {"source": "other"})
    report_path.write_text(json.dumps({"changed": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        timing._read_completed(report_path, complete_path, identity)


def test_prepared_search_adapter_excludes_copy_and_patch_setup(monkeypatch):
    import noisy_acq_search as search

    from pyvbmc.function_logger import FunctionLogger
    from pyvbmc.testing.oracles._state import build_state, load_snapshot

    pin_threads(monkeypatch)
    fixture = (
        Path(__file__).resolve().parents[2]
        / "pyvbmc/testing/oracles/fixtures/rosenbrock_D2_noise1_viqr"
    )
    state = build_state(load_snapshot(fixture))
    events = []
    original_copy = search._copy_state
    original_logger_call = FunctionLogger.__call__

    def private_copy(state, seed):
        events.append("copy")
        return original_copy(state, seed)

    def select(gp, count, optim, logger, history, vp, options):
        events.append("select")
        logger(np.zeros((1, vp.D)))

    ticks = iter(range(28))

    def clock():
        events.append("clock")
        assert FunctionLogger.__call__ is not original_logger_call
        return next(ticks)

    monkeypatch.setattr(search, "_copy_state", private_copy)
    active = importlib.import_module("pyvbmc.vbmc.active_sample")
    monkeypatch.setattr(active, "active_sample", select)

    def prepare(seed):
        return search.prepare_selection(
            state,
            search.SearchConfig(arm="S0"),
            search_seed=seed,
            accurate_seed=seed,
            retain_panel=False,
        )

    report = time_pair(prepare, prepare, seed=41, clock=clock)
    assert len(report["rounds"]) == 7
    assert events[:4] == ["copy", "select", "copy", "select"]
    for start in range(4, len(events), 4):
        assert events[start : start + 4] == [
            "copy",
            "clock",
            "select",
            "clock",
        ]
    assert FunctionLogger.__call__ is original_logger_call
