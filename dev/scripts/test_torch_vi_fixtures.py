"""Lightweight contract checks for the bounded Stage 4 experiment harness.

These tests deliberately do not build a GP, import Torch, or run an
optimization.  Numerical acceptance belongs to the emitted campaign artifacts.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from torch_vi_benchmark import (  # noqa: E402
    MAX_CLEAN_COMPLETE_OVERALL,
    MAX_CLEAN_COMPLETE_PER_BACKEND,
    MAX_COMPLETE_OVERALL,
    MAX_COMPLETE_PER_BACKEND,
    _append_ledger,
    _numpy_kernel_values,
    _parse_cases,
    _parse_seeds,
    _parser,
    _peak_rss,
    _replay_divergence,
    _save_artifact,
    _strict_json,
    _summary,
)
from torch_vi_fixtures import (  # noqa: E402
    GPYREG_PIN,
    MEASUREMENT_SEEDS,
    ORACLE_KERNEL_CASES,
    WORKLOADS,
    _config_manifest,
    _gpyreg_checkout,
    _resolved_options,
)


def test_bounded_workload_matrix_and_caps():
    complete = [spec for spec in WORKLOADS.values() if spec.complete]
    assert len(complete) == 8
    assert len(complete) * len(MEASUREMENT_SEEDS) == MAX_COMPLETE_PER_BACKEND
    assert 3 * MAX_COMPLETE_PER_BACKEND == MAX_COMPLETE_OVERALL
    assert len(complete) == MAX_CLEAN_COMPLETE_PER_BACKEND
    assert 3 * MAX_CLEAN_COMPLETE_PER_BACKEND == MAX_CLEAN_COMPLETE_OVERALL
    assert len(ORACLE_KERNEL_CASES) == 8

    medium = WORKLOADS["medium_synthetic"]
    assert (medium.D, medium.N, medium.input_K, medium.gp_samples) == (
        10,
        250,
        25,
        5,
    )
    boost = WORKLOADS["boost_single"]
    assert (
        boost.D,
        boost.N,
        boost.input_K,
        boost.target_K,
        boost.gp_samples,
    ) == (
        15,
        750,
        25,
        50,
        1,
    )
    stress = WORKLOADS["kernel_stress"]
    assert (stress.D, stress.N, stress.input_K, stress.gp_samples) == (
        20,
        500,
        60,
        1,
    )
    assert not stress.complete


def test_case_and_seed_selection_is_bounded():
    assert _parse_seeds("1703,1701,1703") == (1703, 1701)
    with pytest.raises(ValueError):
        _parse_seeds("1701,9999")
    with pytest.raises(ValueError):
        _parse_seeds("")

    complete = _parse_cases("all_complete")
    assert len(complete) == 8
    assert all(WORKLOADS[name].complete for name in complete)
    kernels = _parse_cases("all_kernels")
    assert len(kernels) == len(ORACLE_KERNEL_CASES) + 3
    assert {"medium_synthetic", "boost_single", "kernel_stress"} <= set(
        kernels
    )
    assert len(complete) == len(set(complete))


def test_clean_and_kernel_only_flags_are_explicit():
    common = [
        "--out",
        "run",
        "--cases",
        "all_complete",
        "--backend",
        "numpy",
        "--device",
        "cpu",
    ]
    clean = _parser().parse_args(
        common + ["--trace-disabled", "--skip-auxiliary", "--warm-kernels"]
    )
    assert clean.trace_disabled
    assert clean.skip_auxiliary
    assert clean.warm_kernels
    assert not clean.kernel_only
    kernel = _parser().parse_args(common + ["--kernel-only"])
    assert kernel.kernel_only
    assert not kernel.trace_disabled


class _DummyVP:
    def __init__(self, theta, mean, covariance, K):
        self._theta = np.asarray(theta, dtype=np.float64)
        self._mean = np.asarray(mean, dtype=np.float64)
        self._covariance = np.asarray(covariance, dtype=np.float64)
        self.K = K

    def get_parameters(self):
        return self._theta.copy()

    def moments(self, *, orig_flag, cov_flag):
        assert not orig_flag and cov_flag
        return self._mean.copy(), self._covariance.copy()


def test_replay_divergence_is_shape_safe():
    complete = {"vp": _DummyVP([1.0, 2.0], [0.0], [[1.0]], K=1)}
    fixed = {"vp": _DummyVP([1.0, 2.0, 3.0], [0.25], [[1.5]], K=2)}
    result = _replay_divergence(complete, fixed)
    assert result["available"]
    assert not result["theta_shape_match"]
    assert result["theta_max_abs"] is None
    assert result["mean_max_abs"] == pytest.approx(0.25)
    assert result["covariance_max_abs"] == pytest.approx(0.5)


def test_standalone_numpy_kernel_refreshes_eta_after_parameter_setter():
    source = inspect.getsource(_numpy_kernel_values)
    set_offset = source.index("vp.set_parameters(theta)")
    eta_offset = source.index("vp.eta = eta.reshape(1, -1)")
    assert set_offset < eta_offset
    assert "eta -= np.max(eta)" in source[set_offset:eta_offset]


def test_peak_rss_reports_value_or_explicit_error():
    record = _peak_rss()
    assert set(record) == {"bytes", "source", "error"}
    if record["bytes"] is None:
        assert isinstance(record["error"], str) and record["error"]
    else:
        assert isinstance(record["bytes"], int) and record["bytes"] > 0
        assert record["error"] is None


class _DummyOptions(dict):
    def eval(self, key, parameters):
        value = self[key]
        return value(**parameters) if callable(value) else value


def test_all_options_and_config_hashes_enter_provenance():
    options = _DummyOptions(
        alpha=3,
        custom_option=lambda K: K + 1,
        useroptions={"custom_option"},
    )
    resolved = _resolved_options(options, (2, 5))
    assert set(resolved) == set(options)
    assert resolved["custom_option"]["evaluated_by_K"] == {"2": 3, "5": 6}
    configs = _config_manifest()
    assert len(configs) == 2
    assert all(len(item["sha256"]) == 64 for item in configs.values())


def test_gpyreg_git_provenance_contract(monkeypatch, tmp_path):
    origin = tmp_path / "gpyreg" / "__init__.py"
    origin.parent.mkdir()
    origin.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "torch_vi_fixtures._module_origin", lambda name: str(origin)
    )

    def fake_git(path, args):
        assert path == origin.parent
        return {
            ("status", "--porcelain"): "",
            ("rev-parse", "HEAD"): GPYREG_PIN,
            ("branch", "--show-current"): "main",
        }[tuple(args)]

    monkeypatch.setattr("torch_vi_fixtures._git_at", fake_git)
    record = _gpyreg_checkout()
    assert record["head"] == GPYREG_PIN
    assert record["matches_pin"]
    assert record["dirty"] is False


def test_invocation_ledger_appends_and_gate_ratios_are_quarantined(tmp_path):
    ledger = tmp_path / "invocations.jsonl"
    _append_ledger(ledger, {"event": "start", "invocation_id": "one"})
    _append_ledger(ledger, {"event": "end", "invocation_id": "one"})
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["start", "end"]

    runs = tmp_path / "runs"
    for backend, device, status, gate in (
        ("numpy", "cpu", "ok", True),
        ("torch", "cpu", "gate_failed", False),
    ):
        run = runs / f"{backend}_{device}"
        run.mkdir(parents=True)
        (run / "result.json").write_text(
            json.dumps(
                {
                    "case": "warped",
                    "seed": 1701,
                    "backend": backend,
                    "device": device,
                    "mode": "traced",
                    "status": status,
                    "complete": True,
                    "requested_complete": True,
                    "complete_run": {"timing": {"total": 1.0}},
                    "timing": {"complete_fit_host_wall_s": 1.2},
                    "kernel": {"all_pass": gate},
                    "memory": {"process_peak_rss_bytes": 100},
                }
            ),
            encoding="utf-8",
        )
    summary = _summary(tmp_path)
    assert summary["paired_ratios"] == []
    assert len(summary["quarantined_ratios"]) == 1
    assert summary["counts"]["gate_failed"] == 1


def test_strict_json_tags_nonfinite_values():
    value = _strict_json(
        {"finite": 1.25, "nan": float("nan"), "inf": float("inf")}
    )
    encoded = json.dumps(value, allow_nan=False)
    assert '"finite": 1.25' in encoded
    assert "nan" in value["nan"]["nonfinite"].lower()
    assert "inf" in value["inf"]["nonfinite"].lower()


def test_artifact_arrays_round_trip_with_named_archive(tmp_path):
    payload = {
        "event": {"epsilon": np.arange(12, dtype=np.float64).reshape(2, 2, 3)},
        "nonfinite": float("nan"),
    }
    _save_artifact(tmp_path, payload, stem="raw_primary")
    tree = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    reference = tree["event"]["epsilon"]
    assert reference["archive"] == "raw_primary.npz"
    assert reference["dtype"] == "float64"
    assert reference["shape"] == [2, 2, 3]
    with np.load(
        tmp_path / reference["archive"], allow_pickle=False
    ) as archive:
        np.testing.assert_array_equal(
            archive[reference["npz"]], payload["event"]["epsilon"]
        )
