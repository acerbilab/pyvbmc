"""Lightweight acceptance checks for the noisy reference preparation."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "dev" / "scripts"
pytestmark = pytest.mark.skipif(
    not (SCRIPTS / "reference_noisy_extension.py").is_file(),
    reason="developer scripts are available only in a repository checkout",
)


def _load_script(name):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get(name, missing)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is missing:
            del sys.modules[name]
        else:
            sys.modules[name] = previous
    return module


def _valid_trace(extension):
    arrays = {key: np.zeros(1) for key in extension.ITERATION_ARRAYS}
    arrays.update(
        timer=np.zeros((1, 0)),
        pt_mu=np.zeros((1, 2)),
        pt_delta=np.zeros((1, 2)),
        pt_scale=np.ones((1, 2)),
        pt_R=np.eye(2)[None, :, :],
        gp_hyp=np.zeros((1, 3)),
        gp_hyp_iter=np.zeros(1),
        vp_w=np.ones(1),
        vp_mu=np.zeros((1, 2)),
        vp_sigma=np.ones(1),
        vp_iter=np.zeros(1),
        vp_lambd=np.ones((1, 2)),
        X_orig=np.zeros((1, 2)),
        y_orig=np.zeros(1),
        X_init=np.zeros((1, 2)),
        y_init=np.zeros(1),
        final_w=np.ones(1),
        final_mu=np.zeros((2, 1)),
        final_sigma=np.ones(1),
        final_lambd=np.ones(2),
        post_mean=np.zeros((1, 2)),
        post_cov=np.eye(2),
    )
    return arrays


def test_golden_extension_registration_and_unchanged_profile_suite():
    targets = _load_script("benchmark_targets")
    profile = tuple(config.label for config in targets.SUITES["profile"])
    golden = tuple(config.label for config in targets.SUITES["golden"])

    assert profile == (
        "banana_D4",
        "cigar_D4",
        "lumpy_D4",
        "student_D4",
        "logreg_D5",
        "rosenbrock_D2_noise1",
        "logreg_D5_noise3",
        "lumpy_D10",
        "banana_D10",
        "cigar_D15_exhaust",
    )
    assert golden == (
        "normal_D5",
        "corr_D5",
        "halfnormal_D2",
        "rosenbrock_D2",
        "banana_D2",
        "banana_D6",
        "banana_D10",
        "cigar_D4",
        "lumpy_D4",
        "lumpy_D10",
        "student_D4",
        "logreg_D5",
        "rosenbrock_D2_noise1",
        "rosenbrock_D2_noise3",
        "logreg_D5_noise3",
        "cigar_D8",
        "student_D8",
        "student_D8_noise3",
        "lumpy_D10_noise3",
        "cigar_D15_exhaust",
    )

    expected = {
        "rosenbrock_D2_noise3": 200,
        "student_D8_noise3": 500,
        "lumpy_D10_noise3": 600,
    }
    configs = {config.label: config for config in targets.SUITES["golden"]}
    for label, budget in expected.items():
        config = configs[label]
        assert config.noise_sd == 3.0
        assert config.options_dict() == {"max_fun_evals": budget}
        problem = config.make(seed=0)
        _, options = problem.vbmc_args()
        assert options == {
            "max_fun_evals": budget,
            "specify_target_noise": True,
        }


def test_extension_allocation_and_run_gate():
    extension = _load_script("reference_noisy_extension")
    tasks = extension._tasks()
    assert len(tasks) == 150
    assert {(task["label"], task["seed"]) for task in tasks} == {
        (label, seed) for label in extension.CONFIGS for seed in range(50)
    }

    args = type("Args", (), {"confirm_run_150": False})()
    with pytest.raises(extension.PreparationError, match="--confirm-run-150"):
        extension.cmd_run(args)

    corrupted = [dict(task) for task in tasks]
    corrupted[0]["tag"] = "wrong_seed0"
    with pytest.raises(extension.PreparationError, match="canonical task tag"):
        extension._verify_task_allocation(corrupted)


def test_historical_manifest_is_pinned_after_preparation(tmp_path):
    extension = _load_script("reference_noisy_extension")
    historical = tmp_path / "historical.json"
    historical.write_text('{"population": "original"}\n', encoding="utf-8")
    paths = {"historical_manifest": historical}
    data = {
        "historical": {
            "manifest_sha256": extension._sha256(
                historical, normalize_text=True
            )
        }
    }

    extension._verify_historical_manifest_pin(data, paths)
    historical.write_text('{"population": "changed"}\n', encoding="utf-8")
    with pytest.raises(
        extension.PreparationError, match="changed since preparation"
    ):
        extension._verify_historical_manifest_pin(data, paths)


def test_direct_worker_task_requires_live_parent_lock(monkeypatch):
    extension = _load_script("reference_noisy_extension")
    missing_manifest = REPO_ROOT / "never-created-preparation.json"
    monkeypatch.setattr(
        extension,
        "_load_manifest",
        lambda unused: (missing_manifest, {}, {}),
    )
    args = type(
        "Args",
        (),
        {
            "manifest": missing_manifest,
            "launch_token": "not-authorized",
            "label": "rosenbrock_D2_noise3",
            "seed": 0,
        },
    )()
    with pytest.raises(extension.PreparationError, match="parent launch lock"):
        extension.cmd_task(args)


def test_resume_state_requires_sidecar_trace_and_runtime(tmp_path):
    extension = _load_script("reference_noisy_extension")
    tag = "rosenbrock_D2_noise3_seed0"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (tmp_path / f"{tag}.json").write_text("{}", encoding="utf-8")
    (tmp_path / f"{tag}.npz").write_bytes(b"placeholder")
    (runtime / f"{tag}.json").write_text("{}", encoding="utf-8")
    assert extension._extension_state(tmp_path, {tag}) == {tag}

    (tmp_path / f"{tag}.npz").unlink()
    with pytest.raises(
        extension.PreparationError, match="not safely resumable"
    ):
        extension._extension_state(tmp_path, {tag})


def test_trace_schema_rejects_missing_and_misaligned_arrays(tmp_path):
    extension = _load_script("reference_noisy_extension")
    valid = _valid_trace(extension)
    valid_path = tmp_path / "valid.npz"
    np.savez(valid_path, **valid)
    extension._validate_trace(valid_path, 2, [], 1, 1)

    missing_path = tmp_path / "missing.npz"
    np.savez(missing_path, iter=np.zeros(1))
    with pytest.raises(extension.PreparationError, match="lacks arrays"):
        extension._validate_trace(missing_path, 2, [], 1, 1)

    invalid_path = tmp_path / "invalid.npz"
    invalid = dict(valid, final_mu=np.zeros((1, 2)))
    np.savez(invalid_path, **invalid)
    with pytest.raises(extension.PreparationError, match="final_mu"):
        extension._validate_trace(invalid_path, 2, [], 1, 1)


def test_completed_record_validates_effective_options_and_worker_runtime(
    tmp_path,
):
    extension = _load_script("reference_noisy_extension")
    tag = "rosenbrock_D2_noise3_seed0"
    row = {"label": "rosenbrock_D2_noise3", "seed": 0, "tag": tag}
    options = {
        "vectorized_target": False,
        "max_fun_evals": 200,
        "specify_target_noise": True,
    }
    side = {
        "label": row["label"],
        "seed": 0,
        "problem": "rosenbrock",
        "D": 2,
        "noise_sd": 3.0,
        "requested_options": dict(options),
        "effective_options": dict(options),
        "timer_keys": [],
        "final": {
            "elbo": 1.0,
            "elbo_sd": 0.1,
            "elbo_err": 0.2,
            "gskl": 0.3,
            "mmtv": 0.4,
            "rmse": 0.5,
            "iterations": 1,
            "final_K": 1,
            "func_count": 100,
            "success_flag": True,
            "message": "ok",
        },
        "meta": {
            **extension.EXPECTED_VERSIONS,
            "git": {"sha": "abcdef0", "dirty": False},
            "threads": extension.THREADS,
        },
    }
    (tmp_path / f"{tag}.json").write_text(json.dumps(side), encoding="utf-8")
    np.savez(tmp_path / f"{tag}.npz", **_valid_trace(extension))
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    runtime_probe = {"interpreter": "python", "modules": {"pyvbmc": "pinned"}}
    runtime = {**runtime_probe, "label": row["label"], "seed": 0}
    runtime_path = runtime_dir / f"{tag}.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    data = {"expected_sha": "abcdef012345", "runtime_probe": runtime_probe}
    paths = {"extension_output": tmp_path}

    assert extension._validate_records(data, paths, [row])["pairs"] == 1
    side["effective_options"]["vectorized_target"] = True
    (tmp_path / f"{tag}.json").write_text(json.dumps(side), encoding="utf-8")
    with pytest.raises(
        extension.PreparationError, match="effective vectorized"
    ):
        extension._validate_records(data, paths, [row])

    side["effective_options"]["vectorized_target"] = False
    (tmp_path / f"{tag}.json").write_text(json.dumps(side), encoding="utf-8")
    runtime["modules"] = {"pyvbmc": "wrong"}
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    with pytest.raises(
        extension.PreparationError, match="worker runtime differs"
    ):
        extension._validate_records(data, paths, [row])
