import copy
import json

import pytest
from filelock import FileLock

from pyvbmc.calibration import CalibrationProfile, _cache


@pytest.fixture(autouse=True)
def isolated_process_state(monkeypatch, tmp_path):
    monkeypatch.setenv("PYVBMC_CACHE_DIR", str(tmp_path / "cache"))
    _cache._clear_process_state()
    yield
    _cache._clear_process_state()


@pytest.fixture
def identity():
    return {
        "cpu_model": "Test CPU",
        "architecture": "test64",
        "host_hash": "a" * 64,
        "operating_system": {"system": "TestOS", "release": "1"},
        "python": "3.10",
        "numpy": "2.0.0",
        "scipy": "1.15.0",
        "backends": [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "prefix": "libopenblas",
                "version": "1.0",
                "num_threads": 2,
                "threading_layer": "pthreads",
                "architecture": "Test CPU",
            }
        ],
        "thread_environment": {
            name: None for name in _cache._THREAD_ENVIRONMENT
        },
        "device": "cpu",
        "dtype": "float64",
    }


def make_test_record(identity, **settings):
    values = {
        "pdf_chunk_elements": 2**14,
        "entropy_grad_chunk_elements": 2**15,
        "entropy_value_chunk_elements": 2**16,
    }
    values.update(settings)
    summary = {
        name: {
            "selected": value,
            "reason": "test selection",
            "heldout_speedup": 1.2,
        }
        for name, value in values.items()
    }
    groups = {}
    for group, workload_names in _cache._GROUP_WORKLOADS.items():
        setting = _cache._GROUP_SETTINGS[group]
        selected = values[setting]
        discovery = []
        for name in workload_names:
            D, K, count, effective = _cache._WORKLOAD_SHAPES[name]
            discovery.append(
                {
                    "name": name,
                    "D": D,
                    "K": K,
                    "requested_count": count,
                    "effective_count": effective,
                    "aliases": {
                        str(candidate): 2**16
                        for candidate in _cache.CANDIDATE_BUDGETS
                    },
                    "layouts": {
                        str(candidate): [1, 1, 0]
                        for candidate in _cache.CANDIDATE_BUDGETS
                    },
                    "workspace_estimates": {
                        str(candidate): {}
                        for candidate in _cache.CANDIDATE_BUDGETS
                    },
                    "first_timing_call_seconds": {"65536": 0.01},
                    "round_seconds": {"65536": [0.01] * 5},
                    "default_control_seconds": [0.01] * 5,
                }
            )
        heldout = [
            {
                "name": name,
                "selected_budget": selected,
                "default_seconds": [0.01] * 4,
                "selected_seconds": [0.01] * 4,
                "default_control_seconds": [0.01] * 4,
                "selected_control_seconds": [0.01] * 4,
            }
            for name in workload_names
        ]
        groups[group] = {
            "setting": setting,
            "discovery": discovery,
            "discovery_selection": {
                "budget": selected,
                "accepted": selected != 2**16,
                "reason": "test selection",
                "candidates": {
                    str(candidate): {"pass": candidate == selected}
                    for candidate in _cache.CANDIDATE_BUDGETS
                    if candidate != 2**16
                },
            },
            "heldout": heldout,
            "heldout_validation": {
                "pass": True,
                "accepted_budget": selected,
                "reason": "test validation",
            },
        }
    workload_names = {
        name for names in _cache._GROUP_WORKLOADS.values() for name in names
    }
    valid_budgets = {
        str(candidate): True for candidate in _cache.CANDIDATE_BUDGETS
    }
    numerical_workloads = {
        name: {
            "checks": [
                {
                    "budget": candidate,
                    "diagnostic_call_seconds": 0.01,
                    "pass": True,
                    "rng_advancement_exact": True,
                }
                for candidate in _cache.CANDIDATE_BUDGETS
            ],
            "first_public_call_seconds": 0.01,
            "valid_budgets": valid_budgets.copy(),
            "inputs_vp_and_vp_rng_unchanged": True,
        }
        for name in workload_names
    }
    memory_estimates = {
        name: {
            str(candidate): {
                "bytes": 1024,
                "limit_bytes": 64 * 2**20,
                "within_limit": True,
                "label": "test estimate",
            }
            for candidate in _cache.CANDIDATE_BUDGETS
        }
        for name in workload_names
    }
    report = {
        "schema_version": 1,
        "recipe_version": 1,
        "status": "complete",
        "estimated_seconds": 30.0,
        "watchdog_seconds": 300.0,
        "candidate_budgets": sorted(_cache.CANDIDATE_BUDGETS),
        "discovery_rounds": 5,
        "heldout_rounds": 4,
        "memory_limit_bytes": 64 * 2**20,
        "groups": groups,
        "numerical": {
            "dedicated": {
                "pass": True,
                "checks": [{"name": "test", "pass": True}],
                "inputs_vp_and_vp_rng_unchanged": True,
            },
            "workloads": numerical_workloads,
            "valid_by_group": {
                group: valid_budgets.copy()
                for group in _cache._GROUP_WORKLOADS
            },
        },
        "timings_seconds": {
            "setup": 0.0,
            "numerical": 1.0,
            "discovery": 1.0,
            "heldout": 1.0,
            "total": 3.0,
        },
        "memory_estimates": memory_estimates,
        "watchdog": {
            "deadline": 300.0,
            "last_checkpoint": "campaign_complete",
            "maximum_kernel_call_seconds": 0.01,
            "stopped": False,
        },
        "numpy_global_rng_unchanged": True,
        "settings": values.copy(),
        "summary": summary,
    }
    return _cache.make_record(
        identity=identity,
        settings=values,
        report=report,
        elapsed_seconds=3.0,
    )


def test_cache_round_trip_is_strict_and_read_only(monkeypatch, identity):
    monkeypatch.setattr(_cache, "machine_identity", lambda: identity)
    record = make_test_record(identity)
    path = _cache.write_record(record)

    profile = _cache.resolve_cached_profile()

    assert profile.settings == record["settings"]
    assert profile.source == "cache"
    assert profile.status == "complete"
    assert profile.fingerprint == record["fingerprint"]
    assert profile.cache_path == str(path)
    assert _cache._load_report(profile) == record["report"]


def test_cache_miss_does_not_create_directory(monkeypatch, tmp_path, identity):
    root = tmp_path / "does-not-exist"
    monkeypatch.setenv("PYVBMC_CACHE_DIR", str(root))
    monkeypatch.setattr(_cache, "machine_identity", lambda: identity)

    profile = _cache.resolve_cached_profile()

    assert profile.settings == {
        "pdf_chunk_elements": 2**16,
        "entropy_grad_chunk_elements": 2**16,
        "entropy_value_chunk_elements": 2**16,
    }
    assert profile.status == "cache_miss"
    assert profile.cache_path is None
    assert not root.exists()


@pytest.mark.parametrize("invalid", [True, 0, -1, 2**13, 2**19])
def test_cache_rejects_invalid_or_unsupported_budget(identity, invalid):
    record = make_test_record(identity)
    record["settings"]["pdf_chunk_elements"] = invalid

    with pytest.raises(ValueError, match="invalid calibrated value"):
        _cache.validate_record(record)


def test_cache_rejects_nonfinite_report_and_wrong_fingerprint(identity):
    record = make_test_record(identity)
    record["report"]["timings_seconds"]["total"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        _cache.validate_record(record)

    record = make_test_record(identity)
    record["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        _cache.validate_record(record)


@pytest.mark.parametrize(
    "provenance",
    [
        {"elapsed_seconds": 1.0},
        {
            "pyvbmc_version": None,
            "calibrated_at_utc": "now",
            "elapsed_seconds": {"nested": 1.0},
        },
    ],
)
def test_cache_rejects_malformed_provenance(identity, provenance):
    record = make_test_record(identity)
    record["provenance"] = provenance

    with pytest.raises(ValueError, match="provenance"):
        _cache.validate_record(record)


def test_malformed_cached_provenance_is_a_cache_miss(monkeypatch, identity):
    monkeypatch.setattr(_cache, "machine_identity", lambda: identity)
    record = make_test_record(identity)
    record["provenance"].pop("calibrated_at_utc")
    path = _cache.cache_path(record["fingerprint"])
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding="utf-8")

    profile = _cache.resolve_cached_profile()

    assert profile.status == "cache_miss"
    assert "provenance" in profile.provenance["reason"]


def test_cache_rejects_unrecognized_candidate_list(identity):
    record = make_test_record(identity)
    record["report"]["candidate_budgets"].append(2**19)

    with pytest.raises(ValueError, match="candidates"):
        _cache.validate_record(record)


def test_complete_cache_requires_global_rng_preservation(identity):
    record = make_test_record(identity)
    record["report"]["numpy_global_rng_unchanged"] = False

    with pytest.raises(ValueError, match="global RNG"):
        _cache.validate_record(record)


@pytest.mark.parametrize(
    "section", ["groups", "numerical", "memory_estimates"]
)
def test_complete_cache_requires_campaign_coverage(identity, section):
    record = make_test_record(identity)
    record["report"][section] = {}

    with pytest.raises(ValueError, match="incomplete|invalid"):
        _cache.validate_record(record)


def test_complete_cache_requires_positive_kernel_timings(identity):
    record = make_test_record(identity)
    record["report"]["watchdog"]["maximum_kernel_call_seconds"] = 0.0

    with pytest.raises(ValueError, match="kernel timing"):
        _cache.validate_record(record)


def test_complete_cache_requires_selected_budget_numerically_valid(identity):
    record = make_test_record(identity)
    record["report"]["numerical"]["valid_by_group"]["pdf"]["16384"] = False

    with pytest.raises(ValueError, match="candidate numerics"):
        _cache.validate_record(record)


def test_complete_cache_reconciles_numerical_check_results(identity):
    record = make_test_record(identity)
    check = record["report"]["numerical"]["workloads"]["small_value"][
        "checks"
    ][0]
    check["rng_advancement_exact"] = False

    with pytest.raises(ValueError, match="numerical checks"):
        _cache.validate_record(record)


def test_complete_cache_rejects_failed_nondefault_heldout(identity):
    record = make_test_record(identity)
    validation = record["report"]["groups"]["pdf"]["heldout_validation"]
    validation["pass"] = False

    with pytest.raises(ValueError, match="held-out"):
        _cache.validate_record(record)


def test_corrupt_deep_and_oversize_records_are_cache_misses(
    monkeypatch, identity
):
    monkeypatch.setattr(_cache, "machine_identity", lambda: identity)
    path = _cache.cache_path(_cache.fingerprint(identity))
    path.parent.mkdir(parents=True)

    path.write_text("{not JSON", encoding="utf-8")
    assert _cache.read_record(identity)[0] is None

    nested = "null"
    for _ in range(2000):
        nested = f"[{nested}]"
    path.write_text(nested, encoding="utf-8")
    assert _cache.read_record(identity)[0] is None

    path.write_bytes(b"x" * (_cache.MAX_CACHE_BYTES + 1))
    record, reason, _ = _cache.read_record(identity)
    assert record is None
    assert "size limit" in reason


def test_changed_identity_invalidates_record(monkeypatch, identity):
    monkeypatch.setattr(_cache, "machine_identity", lambda: identity)
    _cache.write_record(make_test_record(identity))
    changed = copy.deepcopy(identity)
    changed["thread_environment"]["OMP_NUM_THREADS"] = "8"

    record, reason, _ = _cache.read_record(changed)

    assert record is None
    assert reason == "no compatible calibration cache"


def test_fresh_lookup_observes_external_record_replacement(
    monkeypatch, identity
):
    monkeypatch.setattr(_cache, "machine_identity", lambda: identity)
    first = make_test_record(identity, pdf_chunk_elements=2**14)
    _cache.write_record(first)
    assert _cache.resolve_cached_profile().pdf_chunk_elements == 2**14

    second = make_test_record(identity, pdf_chunk_elements=2**18)
    _cache.write_record(second)

    assert _cache.resolve_cached_profile().pdf_chunk_elements == 2**18


def test_memory_registry_is_scoped_to_cache_root(
    monkeypatch, tmp_path, identity
):
    monkeypatch.setattr(_cache, "machine_identity", lambda: identity)
    record = make_test_record(identity)
    profile = CalibrationProfile(
        **record["settings"],
        source="memory",
        fingerprint=_cache.fingerprint(identity),
    )
    _cache.register_success(profile, record["report"])
    assert _cache.resolve_cached_profile() is profile

    monkeypatch.setenv("PYVBMC_CACHE_DIR", str(tmp_path / "other-cache"))

    resolved = _cache.resolve_cached_profile()
    assert resolved.source == "default"
    assert resolved.pdf_chunk_elements == 2**16


def test_insufficient_backend_identity_disables_reuse(identity):
    identity["backends"] = []
    assert "backend" in _cache.identity_reuse_reason(identity)
    with pytest.raises(ValueError, match="insufficient"):
        _cache.make_record(
            identity=identity,
            settings={name: 2**16 for name in _cache._SETTING_NAMES},
            report={},
            elapsed_seconds=1.0,
        )


def test_atomic_write_failure_preserves_previous_record(monkeypatch, identity):
    previous = make_test_record(identity, pdf_chunk_elements=2**14)
    path = _cache.write_record(previous)
    replacement = make_test_record(identity, pdf_chunk_elements=2**18)

    def fail_replace(source, destination):
        raise PermissionError("read only")

    monkeypatch.setattr(_cache.os, "replace", fail_replace)
    with pytest.raises(PermissionError):
        _cache.write_record(replacement)

    assert json.loads(path.read_text(encoding="utf-8")) == previous
    assert not list(path.parent.glob("*.tmp"))


def test_native_lock_is_nonblocking_and_shared_across_fingerprints(
    monkeypatch, identity
):
    monkeypatch.setattr(_cache, "machine_identity", lambda: identity)
    lock_path = (
        _cache.cache_root()
        / "calibration"
        / f"campaign-{identity['host_hash'][:32]}.lock"
    )
    lock_path.parent.mkdir(parents=True)
    outer = FileLock(lock_path)
    with outer.acquire():
        with _cache.campaign_guard() as guard:
            assert not guard.acquired
            assert guard.persistent
            assert "machine lock" in guard.reason


def test_unwritable_lock_falls_back_to_process_guard(monkeypatch, identity):
    monkeypatch.setattr(_cache, "machine_identity", lambda: identity)

    def unavailable(path):
        raise PermissionError("read only")

    monkeypatch.setattr(_cache, "FileLock", unavailable)
    with _cache.campaign_guard() as guard:
        assert guard.acquired
        assert not guard.persistent
        assert "cache unavailable" in guard.reason


def test_suggestion_is_once_per_identity_and_respects_display(
    monkeypatch, capsys, identity
):
    monkeypatch.setattr(_cache, "machine_identity", lambda: identity)
    assert not _cache.suggest_calibration_once(display=False)
    assert _cache.suggest_calibration_once("missing")
    assert not _cache.suggest_calibration_once("missing")
    output = capsys.readouterr().out
    assert output.strip() == (
        "PyVBMC is using standard performance settings. You can optionally "
        "tune these for your machine by running pyvbmc.calibrate(). "
        "Calibration takes tens of seconds and does not evaluate your model."
    )
