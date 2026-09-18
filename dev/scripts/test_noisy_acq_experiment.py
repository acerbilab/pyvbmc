"""Protocol tests for the noisy-acquisition experiment harness.

These checks do not run VBMC fits.  The numerical evaluator has its focused
tests in ``test_noisy_acq_quadrature.py``.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import noisy_acq_experiment as runner
import numpy as np
import pytest


def small_manifest():
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "launch_ready": True,
        "numerical_base_commit": runner.NUMERICAL_BASE,
        "allocation": [{"label": "example_D2_noise1", "seeds": [0, 1]}],
        "case_count": 2,
        "state_target_count": 4,
    }


def test_manifest_requires_review_and_consistent_allocation():
    manifest = small_manifest()
    runner.validate_manifest(manifest, require_ready=True)
    manifest["launch_ready"] = False
    with pytest.raises(RuntimeError, match="reviewed"):
        runner.validate_manifest(manifest, require_ready=True)
    manifest["launch_ready"] = True
    manifest["case_count"] = 3
    with pytest.raises(RuntimeError, match="case_count"):
        runner.validate_manifest(manifest, require_ready=False)


def test_manifest_digest_is_order_independent_and_content_sensitive():
    manifest = small_manifest()
    reordered = dict(reversed(list(manifest.items())))
    assert runner.manifest_digest(manifest) == runner.manifest_digest(
        reordered
    )
    changed = copy.deepcopy(manifest)
    changed["allocation"][0]["seeds"] = [0]
    assert runner.manifest_digest(manifest) != runner.manifest_digest(changed)


def test_canonical_callable_has_no_process_specific_repr():
    def local(value=1):
        return value

    encoded = runner.canonical(local)
    assert encoded["@callable"].endswith(
        "test_canonical_callable_has_no_process_specific_repr.<locals>.local"
    )
    assert "0x" not in json.dumps(encoded)
    assert runner.canonical(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert runner.canonical(float("inf")) == {"@float": "inf"}
    # Non-finite entries inside arrays are encoded like scalars, so a
    # record holding them still serializes as strict JSON.
    assert runner.canonical(np.array([0.0, np.nan])) == [
        0.0,
        {"@float": "nan"},
    ]
    json.dumps(
        runner.canonical({"cache_indices": np.array([[np.nan, 1.0]])}),
        allow_nan=False,
    )


def test_runtime_identity_honours_a_declared_source_transition(monkeypatch):
    import gpyreg

    import pyvbmc

    pinned = {"pyvbmc/a.py": "a0", "pyvbmc/b.py": "b0"}
    current = {"pyvbmc/a.py": "a1", "pyvbmc/b.py": "b0"}
    identity = {
        "gpyreg": {"g": 1},
        "python": "py",
        "platform": "plat",
        "dependencies": {"d": 1},
        "thread_environment": {"OMP_NUM_THREADS": "1"},
        "numpy_build": {"n": 1},
        "data_hashes": {"h": 1},
    }
    manifest = {
        **identity,
        "gpyreg": {"import": gpyreg.__file__, **identity["gpyreg"]},
        "source_hashes": pinned,
    }
    assert Path(pyvbmc.__file__).resolve().parent == runner.ROOT / "pyvbmc"
    monkeypatch.setattr(runner, "gpyreg_identity", lambda: manifest["gpyreg"])
    monkeypatch.setattr(runner.platform, "python_version", lambda: "py")
    monkeypatch.setattr(runner.platform, "platform", lambda: "plat")
    monkeypatch.setattr(runner, "_package_versions", lambda: {"d": 1})
    monkeypatch.setattr(runner, "numpy_build", lambda: {"n": 1})
    monkeypatch.setattr(runner, "data_hashes", lambda: {"h": 1})
    monkeypatch.setattr(runner, "source_hashes", lambda: current)
    monkeypatch.setattr(runner, "manifest_digest", lambda m: "digest")
    for key in runner.THREAD_KEYS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(
        runner,
        "THREAD_KEYS",
        ("OMP_NUM_THREADS",),
    )
    seen = []

    def numerical_diff(exclude=()):
        seen.append(tuple(exclude))
        return "" if exclude else "pyvbmc/a.py changed"

    monkeypatch.setattr(runner, "numerical_diff", numerical_diff)

    with pytest.raises(RuntimeError, match="source_hashes differs"):
        runner.runtime_identity(manifest)
    transition = {
        "changed_sources": {"pyvbmc/a.py": {"capture": "a0", "runtime": "a1"}},
        "numerical_diff_exclude": ["pyvbmc/testing"],
    }
    actual = runner.runtime_identity(manifest, transition)
    assert actual["source_hashes"] == current
    assert actual["source_transition"] == transition
    assert seen[-1] == ("pyvbmc/a.py", "pyvbmc/testing")
    with pytest.raises(RuntimeError, match="misstates the captured hash"):
        runner.runtime_identity(
            manifest,
            {
                "changed_sources": {
                    "pyvbmc/a.py": {"capture": "wrong", "runtime": "a1"}
                }
            },
        )
    with pytest.raises(RuntimeError, match="differs from the declared"):
        runner.runtime_identity(
            manifest,
            {
                "changed_sources": {
                    "pyvbmc/a.py": {"capture": "a0", "runtime": "a2"}
                }
            },
        )
    current["pyvbmc/b.py"] = "b1"
    with pytest.raises(RuntimeError, match="source_hashes differs"):
        runner.runtime_identity(manifest, transition)


def write_completed_case(tmp_path, manifest, *, damage=None):
    label, seed = "example_D2_noise1", 0
    tag = runner._case_tag(label, seed)
    capture = {
        "early_present": True,
        "late_present": False,
        "early_func_count": 20,
        "late_func_count": None,
        "late_trigger": None,
    }
    case = tmp_path / "records" / f"{tag}.case.json"
    runner.write_json(
        case,
        {
            "label": label,
            "seed": seed,
            "status": "succeeded",
            "capture": capture,
            "result": {"success_flag": True},
            "failure": None,
        },
    )
    artifact = tmp_path / f"{tag}_early.npz"
    np.savez_compressed(artifact, value=np.array([1.0]))
    sidecar = tmp_path / f"{tag}_early.json"
    runner.write_json(sidecar, {"meta": {"checkpoint": "early"}})
    artifacts = {
        case.relative_to(tmp_path).as_posix(): runner.sha256_file(case),
        artifact.relative_to(tmp_path).as_posix(): runner.sha256_file(
            artifact
        ),
        sidecar.relative_to(tmp_path).as_posix(): runner.sha256_file(sidecar),
    }
    complete = {
        "schema_version": runner.SCHEMA_VERSION,
        "status": "succeeded",
        "label": label,
        "seed": seed,
        "manifest_sha256": runner.manifest_digest(manifest),
        "identity": {},
        "artifacts": artifacts,
        "capture": capture,
    }
    runner.write_json(runner._completion_path(tmp_path, tag), complete)
    if damage == "artifact":
        artifact.write_bytes(b"changed")
    elif damage == "identity":
        complete["manifest_sha256"] = "0" * 64
        runner.write_json(runner._completion_path(tmp_path, tag), complete)
    return label, seed


@pytest.mark.parametrize("damage", ["artifact", "identity"])
def test_completed_case_rejects_tampering(tmp_path, damage):
    manifest = small_manifest()
    label, seed = write_completed_case(tmp_path, manifest, damage=damage)
    with pytest.raises(RuntimeError):
        runner.validate_completed_case(manifest, tmp_path, label, seed)


def test_inventory_counts_missing_cells_in_denominator(tmp_path):
    manifest = small_manifest()
    label, seed = write_completed_case(tmp_path, manifest)
    runner.validate_completed_case(manifest, tmp_path, label, seed)
    report = runner.inventory(manifest, tmp_path)
    assert report["expected_cases"] == 2
    assert report["expected_states"] == 4
    assert report["counts"] == {
        "complete_cases": 1,
        "missing_cases": 1,
        "failed_cases": 0,
        "states": 1,
    }
    assert [row["status"] for row in report["rows"]] == [
        "succeeded",
        "missing",
    ]


def test_late_fallback_must_be_distinct_from_early():
    collector = runner.CaptureCollector.__new__(runner.CaptureCollector)
    collector.calls_seen = 1
    collector.late = None
    collector.early = ({}, {"meta": {"func_count": 20, "iteration": 2}})
    collector.last = collector.early
    report = collector.finish()
    assert not report["late_present"]

    collector.last = ({}, {"meta": {"func_count": 25, "iteration": 2}})
    report = collector.finish()
    assert report["late_present"]
    assert report["late_trigger"] == "early_termination_fallback"
    assert collector.late[1]["meta"]["checkpoint"] == "late"


def test_partial_case_is_never_reported_as_resumable(tmp_path):
    manifest = small_manifest()
    tag = runner._case_tag("example_D2_noise1", 0)
    (tmp_path / f"{tag}_early.npz").write_bytes(b"partial")
    report = runner.inventory(manifest, tmp_path)
    assert report["rows"][0]["status"] == "partial"
    assert report["counts"]["failed_cases"] == 1


def test_failed_trajectory_is_terminal_and_keeps_a_hashed_failure_record(
    tmp_path,
):
    manifest = small_manifest()
    capture = {
        "calls_seen": 0,
        "early_present": False,
        "late_present": False,
        "late_trigger": None,
        "early_func_count": None,
        "late_func_count": None,
    }
    collector = SimpleNamespace(
        early=None,
        late=None,
        finish=lambda: capture,
    )
    failure = {
        "type": "RuntimeError",
        "message": "trajectory failed",
        "traceback": "test traceback",
    }
    complete = runner._publish_capture_case(
        manifest,
        tmp_path,
        "example_D2_noise1",
        0,
        {"manifest_sha256": runner.manifest_digest(manifest)},
        collector,
        0.0,
        None,
        failure,
    )
    assert complete["status"] == "failed"
    assert (
        "records/example_D2_noise1_seed0.failure.json" in complete["artifacts"]
    )
    runner.validate_completed_case(manifest, tmp_path, "example_D2_noise1", 0)
    report = runner.inventory(manifest, tmp_path)
    assert report["rows"][0]["status"] == "failed"
    assert report["counts"]["failed_cases"] == 1
    assert report["counts"]["missing_cases"] == 1


@pytest.mark.parametrize("damage", ["schema", "count", "factor"])
def test_restore_rejects_malformed_capture_extension(damage):
    factor = {key: None for key in runner.FACTOR_FIELDS}
    snapshot = {
        "gp": {"Ns": 1},
        "vp": {},
        "logger": {},
        "optim_state": {},
        "options": {},
        "meta": {"capture_schema_version": runner.SCHEMA_VERSION},
        "cand": {},
        "ref": {},
        "live_factors": [factor],
        "temporary_data": {},
        "rng_state": {"vp": {}, "numpy_legacy": [], "target": None},
        "effective_options": {},
        "requested_options": {},
    }
    if damage == "schema":
        snapshot["meta"]["capture_schema_version"] += 1
        match = "unsupported schema"
    elif damage == "count":
        snapshot["live_factors"] = []
        match = "factor count"
    else:
        snapshot["live_factors"][0].pop(runner.FACTOR_FIELDS[0])
        match = "missing factors"
    with pytest.raises(RuntimeError, match=match):
        runner._validate_capture_snapshot(snapshot)


def test_real_oracle_snapshot_roundtrip_preserves_options_factors_and_rng(
    tmp_path,
):
    from pyvbmc.acquisition_functions import AcqFcnVIQR
    from pyvbmc.testing.oracles._oracles import (
        prepare_gp_for_acq,
        prepare_importance_sampling,
    )
    from pyvbmc.testing.oracles._state import (
        build_state,
        encode,
        load_snapshot,
    )

    fixture = (
        runner.ROOT
        / "pyvbmc/testing/oracles/fixtures/rosenbrock_D2_noise1_viqr"
    )
    state = build_state(load_snapshot(fixture), rng=np.random.default_rng(17))
    prepare_gp_for_acq(state["gp"], state["logger"], state["optim_state"])
    candidates = np.array(state["cand"]["Xs"], copy=True)
    problem = SimpleNamespace(
        D=state["vp"].D, _noise_rng=np.random.default_rng(18)
    )
    identity = {"manifest_sha256": "test-manifest"}
    collector = runner.CaptureCollector(
        "rosenbrock_D2_noise1",
        0,
        problem,
        state["options"],
        {},
        identity,
    )
    collector.ns_search = candidates.shape[0]
    acquisition = AcqFcnVIQR()
    prepare_importance_sampling(state, acquisition, seed=19)
    rng_before = copy.deepcopy(state["vp"].rng.bit_generator.state)
    arrays, tree = collector._snapshot(
        acquisition,
        candidates,
        state["gp"],
        state["vp"],
        state["logger"],
        state["optim_state"],
        "early",
        "test",
    )
    value = acquisition(
        candidates.copy(),
        state["gp"],
        state["vp"],
        state["logger"],
        copy.deepcopy(state["optim_state"]),
    )
    tree["ref"] = {"acq": encode(value, "ref/acq", arrays)}
    path = tmp_path / "roundtrip"
    runner._save_snapshot_atomic(path, arrays, tree)
    assert runner.verify_capture(path)["public_acquisition_exact"]
    restored = runner.restore_capture(path)
    assert runner.canonical(restored["effective_options"]) == runner.canonical(
        dict(state["options"])
    )
    assert restored["vp"].rng.bit_generator.state == rng_before
    assert (
        restored["gp"].temporary_data.keys()
        == state["gp"].temporary_data.keys()
    )
