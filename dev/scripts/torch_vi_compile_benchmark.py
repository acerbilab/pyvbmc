"""Bounded complete-fit runner for the compiled-Torch Stage 4 follow-up.

The public CLI is a serial coordinator.  It starts one fresh worker for each
case/backend arm, gives compiled workers a unique persistent compiler cache,
and checkpoints every outcome without retrying it.  A worker constructs the
compile adapter without executing it, measures one cold seed-1701 fit, then
three same-process fits for seeds 1701--1703.  Fixed-input gates and independent
NumPy rescoring run only after those fits so they cannot warm the measurements.

This script does not change production PyVBMC or the completed eager evidence.
It requires explicit output and overlay paths and never launches on import.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
CASES = ("deterministic", "warped", "boost_sampled", "boost_single")
SEEDS = (1701, 1702, 1703)
RESCORE_SEEDS = (2701, 2702, 2703, 2704, 2705)
ARM_NAMES = (
    "numpy_cpu",
    "eager_cpu",
    "compiled_cpu",
    "eager_cuda",
    "compiled_cuda",
)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
}
DEFAULT_WORKER_TIMEOUT_S = 30 * 60
DEFAULT_SMOKE_TIMEOUT_S = 10 * 60
DEFAULT_CAMPAIGN_CAP_S = 3 * 60 * 60


@dataclass(frozen=True)
class Arm:
    name: str
    backend: str
    device: str
    compiled: bool
    overlay_kind: str | None


ARMS = {
    "numpy_cpu": Arm("numpy_cpu", "numpy", "cpu", False, None),
    "eager_cpu": Arm("eager_cpu", "torch", "cpu", False, "cpu"),
    "compiled_cpu": Arm("compiled_cpu", "torch", "cpu", True, "cpu"),
    "eager_cuda": Arm("eager_cuda", "torch", "cuda", False, "cuda"),
    "compiled_cuda": Arm("compiled_cuda", "torch", "cuda", True, "cuda"),
}

SOURCE_FILES = (
    "torch_vi_compile_benchmark.py",
    "torch_vi_compile.py",
    "torch_compile_toolchain.py",
    "torch_vi_core.py",
    "torch_vi_step.py",
    "torch_vi_benchmark.py",
    "torch_vi_fixtures.py",
)


def _strict(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": repr(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if hasattr(value, "item"):
        with contextlib.suppress(Exception):
            return _strict(value.item())
    return repr(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_strict(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _replace_with_retry(temporary, path)


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Tolerate brief Windows scanner locks, then expose the real failure."""
    attempts = 20
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        _strict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: Any, np: Any) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(repr(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *arguments],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _source_manifest(out: Path) -> dict[str, Any]:
    archive_dir = out / "source"
    archive_dir.mkdir(parents=True, exist_ok=False)
    archive = archive_dir / "measured-source.zip"
    files: dict[str, str] = {}
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        for name in SOURCE_FILES:
            source = SCRIPT_DIR / name
            if not source.is_file():
                raise FileNotFoundError(
                    f"required experiment source is absent: {source}"
                )
            relative = f"dev/scripts/{name}"
            bundle.write(source, relative)
            files[relative] = _sha256(source)
    manifest = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "files": files,
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
    }
    _atomic_json(archive_dir / "manifest.json", manifest)
    return manifest


def _parse_csv(value: str, allowed: Sequence[str], label: str) -> list[str]:
    selected = list(
        dict.fromkeys(
            part.strip() for part in value.split(",") if part.strip()
        )
    )
    unknown = set(selected).difference(allowed)
    if not selected or unknown:
        raise ValueError(
            f"{label} must be a nonempty comma-separated subset of {tuple(allowed)}; "
            f"got {selected}"
        )
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "campaign"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pilot-case", choices=CASES)
    parser.add_argument("--arms", default=",".join(ARM_NAMES))
    parser.add_argument("--cpu-overlay", type=Path)
    parser.add_argument("--cuda-overlay", type=Path)
    parser.add_argument("--compiler-overlay", type=Path)
    parser.add_argument("--cpu-smoke", type=Path)
    parser.add_argument("--cuda-smoke", type=Path)
    parser.add_argument(
        "--worker-timeout", type=float, default=DEFAULT_WORKER_TIMEOUT_S
    )
    parser.add_argument(
        "--campaign-cap", type=float, default=DEFAULT_CAMPAIGN_CAP_S
    )
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode")
    parser.add_argument(
        "--worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARM_NAMES, help=argparse.SUPPRESS)
    return parser


def _toolchain_record(path: Path | None, device: str) -> dict[str, Any]:
    if path is None:
        return {"required": True, "valid": False, "error": "path not supplied"}
    path = path.resolve()
    if not path.is_file():
        return {
            "required": True,
            "valid": False,
            "path": str(path),
            "error": "file absent",
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    valid = value.get("status") == "passed" and value.get("device") == device
    return {
        "required": True,
        "valid": valid,
        "path": str(path),
        "sha256": _sha256(path),
        "status": value.get("status"),
        "device": value.get("device"),
        "torch": value.get("torch"),
        "compiler": value.get("compiler"),
        "error": None if valid else "smoke did not pass for requested device",
    }


def _validate_request(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    missing = [
        name for name in ("phase", "out") if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(f"missing required coordinator arguments: {missing}")
    arms = _parse_csv(args.arms, ARM_NAMES, "--arms")
    cases = [args.pilot_case] if args.phase == "pilot" else list(CASES)
    if args.phase == "pilot" and args.pilot_case is None:
        raise ValueError("--phase pilot requires --pilot-case")
    if (
        args.worker_timeout <= 0
        or args.worker_timeout > DEFAULT_WORKER_TIMEOUT_S
    ):
        raise ValueError(
            f"--worker-timeout must be in (0, {DEFAULT_WORKER_TIMEOUT_S}]"
        )
    if args.campaign_cap <= 0 or args.campaign_cap > DEFAULT_CAMPAIGN_CAP_S:
        raise ValueError(
            f"--campaign-cap must be in (0, {DEFAULT_CAMPAIGN_CAP_S}]"
        )
    if any(ARMS[name].overlay_kind == "cpu" for name in arms):
        if args.cpu_overlay is None or not args.cpu_overlay.resolve().is_dir():
            raise ValueError("selected CPU Torch arms require --cpu-overlay")
    if any(ARMS[name].overlay_kind == "cuda" for name in arms):
        if (
            args.cuda_overlay is None
            or not args.cuda_overlay.resolve().is_dir()
        ):
            raise ValueError("selected CUDA Torch arms require --cuda-overlay")
    if "compiled_cuda" in arms:
        if (
            args.compiler_overlay is None
            or not args.compiler_overlay.resolve().is_dir()
        ):
            raise ValueError("compiled CUDA requires --compiler-overlay")
    return cases, arms


def _balanced_schedule(
    cases: Sequence[str], arms: Sequence[str]
) -> list[tuple[str, str]]:
    schedule = []
    for case_index, case in enumerate(cases):
        offset = case_index % len(arms)
        order = list(arms[offset:]) + list(arms[:offset])
        schedule.extend((case, arm) for arm in order)
    return schedule


def _summary_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Compiled Torch Stage 4 campaign status",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Case | Arm | Status | Cold fit s | Hot fit s | Recompilations |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in report.get("workers", []):
        hot = row.get("hot_fit_wall_s", [])
        hot_text = ", ".join(f"{float(value):.6g}" for value in hot)
        lines.append(
            f"| {row.get('case')} | {row.get('arm')} | {row.get('status')} | "
            f"{row.get('cold_fit_wall_s', '')} | {hot_text} | "
            f"{row.get('hot_recompilation_count', '')} |"
        )
    if report.get("failures"):
        lines.extend(
            (
                "",
                "Failures are retained in `campaign.json` and worker artifacts.",
            )
        )
    return "\n".join(lines) + "\n"


def _checkpoint_campaign(out: Path, report: dict[str, Any]) -> None:
    report["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_json(out / "campaign.json", report)
    temporary = out / "summary.md.tmp"
    temporary.write_text(
        _summary_markdown(report), encoding="utf-8", newline="\n"
    )
    _replace_with_retry(temporary, out / "summary.md")


def _configure_worker_paths(
    args: argparse.Namespace, arm: Arm
) -> dict[str, Any]:
    for key, value in THREAD_ENV.items():
        os.environ[key] = value
    os.environ["PYTHONHASHSEED"] = "0"
    run_dir = args.run_dir.resolve()
    cache_token = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()[:12]
    cache_root = SCRIPT_DIR / "runs" / "tcc" / cache_token
    if cache_root.exists():
        raise FileExistsError(f"compiler cache is not fresh: {cache_root}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "i")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "t")
    inserted = []
    if arm.overlay_kind == "cpu":
        inserted.append(args.cpu_overlay.resolve())
    elif arm.overlay_kind == "cuda":
        inserted.append(args.cuda_overlay.resolve())
    if arm.name == "compiled_cuda":
        inserted.insert(0, args.compiler_overlay.resolve())
        cuda_path = args.compiler_overlay.resolve() / "triton/backends/nvidia"
        os.environ["CUDA_PATH"] = str(cuda_path)
    inserted.extend((SCRIPT_DIR, REPO_ROOT))
    for path in reversed(inserted):
        sys.path.insert(0, str(path))
    return {
        "sys_path_prefix": [str(path) for path in inserted],
        "thread_environment": dict(THREAD_ENV),
        "inductor_cache": os.environ["TORCHINDUCTOR_CACHE_DIR"],
        "triton_cache": os.environ["TRITON_CACHE_DIR"],
        "cache_token": cache_token,
        "cuda_path": os.environ.get("CUDA_PATH"),
    }


def _configure_torch(torch: Any) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    return {
        "version": str(torch.__version__),
        "path": str(torch.__file__),
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "default_dtype": str(torch.get_default_dtype()),
        "intraop_threads": int(torch.get_num_threads()),
        "interop_threads": int(torch.get_num_interop_threads()),
    }


def _cuda_context(torch: Any, device: str) -> dict[str, Any]:
    if not device.startswith("cuda"):
        return {"requested": False, "wall_s": 0.0}
    started = time.perf_counter()
    value = torch.ones(1, dtype=torch.float64, device=device) + 1.0
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {
        "requested": True,
        "wall_s": time.perf_counter() - started,
        "probe": float(value.cpu().item()),
        "name": properties.name,
        "total_memory": int(properties.total_memory),
        "capability": list(torch.cuda.get_device_capability(device)),
    }


def _counter_signal(delta: Any, path: str = "") -> dict[str, float]:
    """Extract graph/code-generation increments from a compiler delta tree."""
    signals: dict[str, float] = {}
    interesting = {
        "unique_graphs",
        "generated_kernel_count",
        "generated_cpp_vec_kernel_count",
        "fxgraph_cache_hit",
        "fxgraph_cache_miss",
    }
    if isinstance(delta, Mapping):
        for key, value in delta.items():
            child = f"{path}.{key}" if path else str(key)
            if (
                key in interesting
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value > 0
            ):
                signals[child] = float(value)
            signals.update(_counter_signal(value, child))
    elif isinstance(delta, (list, tuple)):
        for index, value in enumerate(delta):
            signals.update(_counter_signal(value, f"{path}[{index}]"))
    return signals


def _array_record(value: Any, np: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    finite = (
        bool(np.all(np.isfinite(array)))
        if np.issubdtype(array.dtype, np.number)
        else None
    )
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _array_sha256(array, np),
        "finite": finite,
    }


def _summarize_value(value: Any, np: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _summarize_value(item, np) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_summarize_value(item, np) for item in value]
    if isinstance(value, np.ndarray):
        return _array_record(value, np)
    if isinstance(value, np.generic):
        return _strict(value.item())
    return _strict(value)


def _summarize_fit(
    run: Mapping[str, Any],
    label: str,
    seed: int,
    wall_s: float,
    cpu_s: float,
    np: Any,
) -> dict[str, Any]:
    vp = copy.deepcopy(run["vp"])
    theta = np.asarray(vp.get_parameters(), dtype=np.float64)
    mean, covariance = vp.moments(orig_flag=False, cov_flag=True)
    clean_wall = float(run.get("timing", {}).get("total", wall_s))
    return {
        "label": label,
        "seed": seed,
        "wall_s": clean_wall,
        "api_wall_s": wall_s,
        "cpu_s": cpu_s,
        "trace_enabled": run.get("trace_enabled"),
        "complete": run["result"].get("complete"),
        "result": _summarize_value(run["result"], np),
        "timing": _summarize_value(run.get("timing", {}), np),
        "source_hashes": run.get("source_hashes"),
        "theta": _array_record(theta, np),
        "mu": _array_record(np.asarray(vp.mu), np),
        "sigma": _array_record(np.asarray(vp.sigma), np),
        "lambd": _array_record(np.asarray(vp.lambd), np),
        "weights": _array_record(np.asarray(vp.w), np),
        "stats": _summarize_value(vp.stats, np),
        "eta": _array_record(np.asarray(vp.eta), np),
        "rng_state": _summarize_value(vp.rng.bit_generator.state, np),
        "transformed_mean": _array_record(np.asarray(mean), np),
        "transformed_covariance": _array_record(np.asarray(covariance), np),
    }


def _fit_gate(fit: Mapping[str, Any]) -> dict[str, Any]:
    result = fit.get("result", {})
    expected_transformer_identity = not isinstance(
        result.get("boost"), Mapping
    )
    array_keys = (
        "theta",
        "mu",
        "sigma",
        "lambd",
        "weights",
        "eta",
        "transformed_mean",
        "transformed_covariance",
    )
    checks = {
        "complete": result.get("complete") is True,
        "complete_fit_mode": result.get("mode") == "complete_fit",
        "production_policy": result.get("production_policy_complete") is True,
        "trace_disabled": fit.get("trace_enabled") is False,
        "transformer_identity": result.get("transformer_identity")
        is expected_transformer_identity,
        "finite_arrays": all(
            fit.get(key, {}).get("finite") is True for key in array_keys
        ),
        "positive_clean_wall": isinstance(fit.get("wall_s"), (int, float))
        and fit["wall_s"] > 0,
    }
    return {
        "checks": checks,
        "expected_transformer_identity": expected_transformer_identity,
        "recorded_transformer_identity": result.get("transformer_identity"),
        "all_pass": all(checks.values()),
    }


def _run_complete_fit(
    state: Mapping[str, Any], arm: Arm, seed: int, run_step: Any, torch: Any
) -> tuple[dict[str, Any], float, float]:
    kwargs = {
        "backend": arm.backend,
        "device": arm.device,
        "seed": seed,
        "boost": state["boost"],
        "fixed_updates": None,
        "trace_enabled": False,
        "fast_opts_N": state.get("fast_opts_N"),
        "slow_opts_N": state.get("slow_opts_N"),
        "K": state["K"],
    }
    if torch is not None and arm.device.startswith("cuda"):
        torch.cuda.synchronize(arm.device)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    result = run_step(state, **kwargs)
    if torch is not None and arm.device.startswith("cuda"):
        torch.cuda.synchronize(arm.device)
    return (
        result,
        time.perf_counter() - wall_started,
        time.process_time() - cpu_started,
    )


def _worker_checkpoint(
    run_dir: Path,
    summary: dict[str, Any],
    raw: dict[str, Any] | None,
    save_artifact: Any | None,
    stem: str,
) -> None:
    summary["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_json(run_dir / "summary.json", summary)
    if raw is not None and save_artifact is not None:
        save_artifact(run_dir, raw, stem=stem)


def _worker(args: argparse.Namespace) -> int:
    arm = ARMS[args.arm]
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema": 1,
        "status": "initializing",
        "case": args.case,
        "arm": arm.name,
        "backend": arm.backend,
        "device": arm.device,
        "compiled": arm.compiled,
        "cold_seed": SEEDS[0],
        "hot_seeds": list(SEEDS),
        "fits": [],
        "errors": [],
    }
    _worker_checkpoint(run_dir, summary, None, None, "initial")
    worker_started = time.perf_counter()
    raw: dict[str, Any] = {"fits": []}
    save_artifact = None
    try:
        summary["paths"] = _configure_worker_paths(args, arm)
        msvc = None
        if arm.compiled:
            from torch_compile_toolchain import msvc_environment

            msvc_started = time.perf_counter()
            msvc = msvc_environment()
            summary["msvc_setup_s"] = time.perf_counter() - msvc_started
            summary["msvc"] = msvc
            summary["cl"] = shutil.which("cl")

        import_started = time.perf_counter()
        import numpy as np
        from torch_vi_benchmark import _kernel_diagnostics, _save_artifact
        from torch_vi_fixtures import (
            environment_manifest,
            load_workload,
            workload_manifest,
        )
        from torch_vi_step import rescore_candidates, run_step, source_hashes

        save_artifact = _save_artifact
        torch = None
        compile_module = None
        core = None
        if arm.backend == "torch":
            import torch as imported_torch
            import torch_vi_core as imported_core

            torch = imported_torch
            core = imported_core
            summary["torch"] = _configure_torch(torch)
            if arm.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA arm requested but torch.cuda.is_available() is false"
                )
            if arm.compiled:
                import torch_vi_compile as imported_compile

                compile_module = imported_compile
        summary["import_s"] = time.perf_counter() - import_started
        summary["environment"] = environment_manifest()
        if not summary["environment"]["provenance_validation"]["valid"]:
            raise RuntimeError(
                "invalid fixture provenance: "
                + repr(
                    summary["environment"]["provenance_validation"]["errors"]
                )
            )
        summary["workload_manifest"] = workload_manifest([args.case])
        summary["step_source_hashes"] = source_hashes(check=True)
        summary["cuda_context"] = (
            _cuda_context(torch, arm.device)
            if torch is not None
            else {"requested": False, "wall_s": 0.0}
        )

        session = None
        install_context = contextlib.nullcontext()
        if arm.compiled:
            reset_started = time.perf_counter()
            summary["compiler_reset"] = compile_module.reset_compiler_state(
                torch
            )
            summary["compiler_reset_s"] = time.perf_counter() - reset_started
            construct_started = time.perf_counter()
            session = compile_module.CompiledForward(
                core,
                backend=args.compile_backend,
                fullgraph=True,
                dynamic=False,
                mode=args.compile_mode,
                eager_full_variance=True,
                torch_module=torch,
            )
            summary["compile_wrapper_construction_s"] = (
                time.perf_counter() - construct_started
            )
            summary[
                "compiler_capabilities"
            ] = compile_module.compiler_capabilities(torch)
            summary[
                "compile_adapter_source"
            ] = compile_module.source_metadata()
            summary["compiled_callable_type_before_observer"] = (
                f"{type(session.compiled).__module__}."
                f"{type(session.compiled).__qualname__}"
            )
            compiled_observation = {"capture": False, "records": []}
            compiled_callable = session.compiled

            def observe_compiled_forward(
                *forward_args: Any, **forward_kwargs: Any
            ) -> Any:
                output = compiled_callable(*forward_args, **forward_kwargs)
                if (
                    compiled_observation["capture"]
                    and not compiled_observation["records"]
                ):
                    scalar = output[0]
                    compiled_observation["records"].append(
                        {
                            "requires_grad": bool(scalar.requires_grad),
                            "dtype": str(scalar.dtype),
                            "device": str(scalar.device),
                            "grad_fn_type": (
                                None
                                if scalar.grad_fn is None
                                else type(scalar.grad_fn).__name__
                            ),
                        }
                    )
                    session.mark("gate_after_compiled_forward_before_backward")
                return output

            session.compiled = observe_compiled_forward
            install_context = session.install()

        with install_context:
            # No objective call occurs before this cold complete fit.
            cold_state_started = time.perf_counter()
            cold_state = load_workload(args.case, SEEDS[0])
            cold_state_s = time.perf_counter() - cold_state_started
            before = (
                compile_module.snapshot_compile_evidence(torch)
                if session is not None
                else None
            )
            if session is not None:
                session.mark("before_cold_fit")
            cold_run, cold_wall, cold_cpu = _run_complete_fit(
                cold_state, arm, SEEDS[0], run_step, torch
            )
            after = (
                compile_module.snapshot_compile_evidence(torch)
                if session is not None
                else None
            )
            cold_delta = (
                compile_module.evidence_delta(before, after)
                if session is not None
                else None
            )
            if session is not None:
                session.mark("after_cold_fit")
            cold_summary = _summarize_fit(
                cold_run, "cold", SEEDS[0], cold_wall, cold_cpu, np
            )
            cold_summary.update(
                {
                    "state_rebuild_s": cold_state_s,
                    "compiler_delta": cold_delta,
                    "compile_signals": _counter_signal(cold_delta),
                    "classification": "cold_compile_and_fit"
                    if arm.compiled
                    else "cold_fit",
                }
            )
            summary["fits"].append(cold_summary)
            cold_summary["fit_gate"] = _fit_gate(cold_summary)
            raw["fits"].append(cold_run)
            _worker_checkpoint(
                run_dir, summary, raw, save_artifact, "checkpoint_cold"
            )

            for hot_index, seed in enumerate(SEEDS, start=1):
                state_started = time.perf_counter()
                state = load_workload(args.case, seed)
                state_s = time.perf_counter() - state_started
                before = (
                    compile_module.snapshot_compile_evidence(torch)
                    if session is not None
                    else None
                )
                if session is not None:
                    session.mark(f"before_hot_seed_{seed}")
                fit, wall_s, cpu_s = _run_complete_fit(
                    state, arm, seed, run_step, torch
                )
                after = (
                    compile_module.snapshot_compile_evidence(torch)
                    if session is not None
                    else None
                )
                delta = (
                    compile_module.evidence_delta(before, after)
                    if session is not None
                    else None
                )
                signals = _counter_signal(delta)
                if session is not None:
                    session.mark(f"after_hot_seed_{seed}")
                fit_summary = _summarize_fit(
                    fit, "hot", seed, wall_s, cpu_s, np
                )
                fit_summary.update(
                    {
                        "hot_index": hot_index,
                        "state_rebuild_s": state_s,
                        "compiler_delta": delta,
                        "compile_signals": signals,
                        "classification": "recompiled" if signals else "hot",
                    }
                )
                summary["fits"].append(fit_summary)
                fit_summary["fit_gate"] = _fit_gate(fit_summary)
                raw["fits"].append(fit)
                _worker_checkpoint(
                    run_dir,
                    summary,
                    raw,
                    save_artifact,
                    f"checkpoint_hot_{hot_index}",
                )

            # Corrected gate: current _numpy_kernel_values refreshes vp.eta.
            gate_before = (
                compile_module.snapshot_compile_evidence(torch)
                if session is not None
                else None
            )
            gate_started = time.perf_counter()
            if session is not None:
                compiled_observation["capture"] = True
            gate = _kernel_diagnostics(
                cold_state, arm.backend, arm.device, SEEDS[0]
            )
            if session is not None:
                session.mark("gate_after_backward")
            if torch is not None and arm.device.startswith("cuda"):
                torch.cuda.synchronize(arm.device)
            summary["kernel_gate_s"] = time.perf_counter() - gate_started
            gate_after = (
                compile_module.snapshot_compile_evidence(torch)
                if session is not None
                else None
            )
            summary["kernel_gate"] = {
                "all_pass": gate.get("all_pass"),
                "errors": _summarize_value(gate.get("errors", {}), np),
                "epsilon": gate.get("epsilon"),
                "Ns": gate.get("Ns"),
                "production_Ns": gate.get("production_Ns"),
                "timing": gate.get("timing"),
                "compiled_delta": (
                    compile_module.evidence_delta(gate_before, gate_after)
                    if session is not None
                    else None
                ),
                "compiled_scope_note": (
                    "ordinary neg_elcbo optimization forward/backward uses the installed "
                    "compiled boundary; full-variance/per-component fine scoring and "
                    "direct GP/entropy component diagnostics deliberately remain eager"
                    if arm.compiled
                    else None
                ),
            }
            if session is not None:
                gradient = gate.get("raw", {}).get("torch", {}).get("dF_mc")
                gradient_record = (
                    _array_record(gradient, np)
                    if gradient is not None
                    else None
                )
                observation = compiled_observation["records"]
                summary["compiled_backward_observation"] = {
                    "forward": observation[0] if observation else None,
                    "gradient": gradient_record,
                    "executed": bool(
                        observation
                        and observation[0].get("requires_grad")
                        and observation[0].get("dtype") == "torch.float64"
                        and observation[0].get("grad_fn_type")
                        == "CompiledFunctionBackward"
                        and gradient_record is not None
                        and gradient_record["finite"]
                        and gradient_record["dtype"] == "float64"
                    ),
                    "evidence_marks": (
                        "gate_after_compiled_forward_before_backward, gate_after_backward"
                    ),
                }
            raw["kernel_gate"] = gate

        # Rescoring is deliberately outside the compile installation and fit walls.
        rescore_started = time.perf_counter()
        hot_vps = [fit["vp"] for fit in raw["fits"][1:]]
        rescore = rescore_candidates(
            hot_vps,
            gp=cold_state["gp"],
            options=cold_state["options"],
            seeds=RESCORE_SEEDS,
        )
        summary["rescore_s"] = time.perf_counter() - rescore_started
        summary["rescore"] = _summarize_value(rescore, np)
        raw["rescore"] = rescore
        if session is not None:
            summary["compiler_evidence"] = session.evidence()
        summary["cold_hot_seed1701_theta_equal"] = (
            summary["fits"][0]["theta"]["sha256"]
            == summary["fits"][1]["theta"]["sha256"]
        )
        fit_gates_pass = all(
            fit["fit_gate"]["all_pass"] for fit in summary["fits"]
        )
        backward_pass = (
            not arm.compiled
            or summary.get("compiled_backward_observation", {}).get("executed")
            is True
        )
        summary["validation"] = {
            "fits": fit_gates_pass,
            "kernel": summary["kernel_gate"]["all_pass"] is True,
            "compiled_backward": backward_pass,
        }
        summary["validation"]["all_pass"] = all(summary["validation"].values())
        summary["status"] = (
            "ok" if summary["validation"]["all_pass"] else "gate_failed"
        )
        summary["worker_wall_s"] = time.perf_counter() - worker_started
        _worker_checkpoint(run_dir, summary, raw, save_artifact, "raw")
        return 0 if summary["status"] == "ok" else 2
    except BaseException as exc:
        summary["status"] = "failed"
        summary["worker_wall_s"] = time.perf_counter() - worker_started
        summary["errors"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        with contextlib.suppress(Exception):
            _worker_checkpoint(run_dir, summary, raw, save_artifact, "failed")
        return 1


def _worker_row(
    case: str,
    arm: str,
    run_dir: Path,
    returncode: int | None,
    launcher_wall_s: float,
    launcher_error: str | None = None,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    worker = None
    if summary_path.is_file():
        with contextlib.suppress(Exception):
            worker = json.loads(summary_path.read_text(encoding="utf-8"))
    fits = worker.get("fits", []) if isinstance(worker, Mapping) else []
    cold = fits[0] if fits else {}
    hot = fits[1:] if len(fits) > 1 else []
    status = (
        worker.get("status", "launcher_failed")
        if worker
        else "launcher_failed"
    )
    if returncode not in (0, None) and status == "ok":
        status = "launcher_failed"
    return {
        "case": case,
        "arm": arm,
        "status": status,
        "returncode": returncode,
        "launcher_wall_s": launcher_wall_s,
        "launcher_error": launcher_error,
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path)
        if summary_path.is_file()
        else None,
        "cold_fit_wall_s": cold.get("wall_s"),
        "cold_api_wall_s": cold.get("api_wall_s"),
        "hot_fit_wall_s": [fit.get("wall_s") for fit in hot],
        "hot_api_wall_s": [fit.get("api_wall_s") for fit in hot],
        "hot_classifications": [fit.get("classification") for fit in hot],
        "hot_recompilation_count": sum(
            fit.get("classification") == "recompiled" for fit in hot
        ),
        "kernel_gate_pass": (
            worker.get("kernel_gate", {}).get("all_pass") if worker else None
        ),
    }


def _worker_command(
    args: argparse.Namespace, case: str, arm_name: str, run_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--run-dir",
        str(run_dir),
        "--case",
        case,
        "--arm",
        arm_name,
        "--compile-backend",
        args.compile_backend,
    ]
    if args.compile_mode is not None:
        command.extend(("--compile-mode", args.compile_mode))
    for option, value in (
        ("--cpu-overlay", args.cpu_overlay),
        ("--cuda-overlay", args.cuda_overlay),
        ("--compiler-overlay", args.compiler_overlay),
    ):
        if value is not None:
            command.extend((option, str(value.resolve())))
    return command


def _coordinator(args: argparse.Namespace) -> int:
    cases, arms = _validate_request(args)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"use a fresh output directory: {out}")
    out.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema": 1,
        "status": "initializing",
        "phase": args.phase,
        "cases": cases,
        "arms": arms,
        "seeds": list(SEEDS),
        "cold_policy": "first seed-1701 complete fit in a fresh worker",
        "hot_policy": "same-session complete fits for seeds 1701, 1702, 1703",
        "fit_timing_policy": (
            "primary wall time is run_step timing.total; launcher/API wall retains "
            "source-hash checking and synchronization around the call"
        ),
        "compiled_scope": (
            "ordinary optimization objective and backward are compiled; exact "
            "full-variance/per-component fine scoring is deliberately eager"
        ),
        "limits_s": {
            "toolchain_smoke": DEFAULT_SMOKE_TIMEOUT_S,
            "worker": args.worker_timeout,
            "campaign": args.campaign_cap,
        },
        "schedule": _balanced_schedule(cases, arms),
        "workers": [],
        "failures": [],
        "source_manifest": _source_manifest(out),
        "toolchain": {},
        "command": sys.argv,
    }
    compiled_devices = {
        ARMS[name].device for name in arms if ARMS[name].compiled
    }
    if "cpu" in compiled_devices:
        report["toolchain"]["cpu"] = _toolchain_record(args.cpu_smoke, "cpu")
    if "cuda" in compiled_devices:
        report["toolchain"]["cuda"] = _toolchain_record(
            args.cuda_smoke, "cuda"
        )
    invalid = [
        device
        for device, record in report["toolchain"].items()
        if not record["valid"]
    ]
    if invalid:
        report["status"] = "failed_preflight"
        report["failures"].append(
            {"stage": "toolchain", "invalid_devices": invalid}
        )
        _checkpoint_campaign(out, report)
        return 2

    report["status"] = "running"
    _checkpoint_campaign(out, report)
    campaign_started = time.perf_counter()
    runs_root = out / "runs"
    runs_root.mkdir()
    for sequence, (case, arm_name) in enumerate(report["schedule"]):
        elapsed = time.perf_counter() - campaign_started
        remaining = args.campaign_cap - elapsed
        if remaining <= 0:
            report["status"] = "stopped_campaign_cap"
            report["failures"].append(
                {"stage": "campaign", "error": "campaign wall cap reached"}
            )
            break
        run_dir = runs_root / f"{sequence:02d}-{case}-{arm_name}"
        command = _worker_command(args, case, arm_name, run_dir)
        log_path = runs_root / f"{sequence:02d}-{case}-{arm_name}.log"
        env = os.environ.copy()
        env.update(THREAD_ENV)
        started = time.perf_counter()
        returncode: int | None = None
        launcher_error = None
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                returncode = process.wait(
                    timeout=min(float(args.worker_timeout), remaining)
                )
            except subprocess.TimeoutExpired:
                launcher_error = "worker timeout"
                if os.name == "nt":
                    subprocess.run(
                        [
                            "taskkill.exe",
                            "/PID",
                            str(process.pid),
                            "/T",
                            "/F",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        text=True,
                    )
                else:
                    process.terminate()
                try:
                    returncode = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait()
        row = _worker_row(
            case,
            arm_name,
            run_dir,
            returncode,
            time.perf_counter() - started,
            launcher_error,
        )
        row["log"] = str(log_path)
        row["log_sha256"] = _sha256(log_path)
        report["workers"].append(row)
        _append_jsonl(out / "workers.jsonl", row)
        if returncode != 0 or row["status"] != "ok":
            report["failures"].append(
                {
                    "stage": "worker",
                    "case": case,
                    "arm": arm_name,
                    "status": row["status"],
                    "returncode": returncode,
                    "launcher_error": launcher_error,
                }
            )
        _checkpoint_campaign(out, report)
    else:
        report["status"] = (
            "completed"
            if not report["failures"]
            else "completed_with_failures"
        )
    report["campaign_wall_s"] = time.perf_counter() - campaign_started
    _checkpoint_campaign(out, report)
    return 0 if report["status"] == "completed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker:
        missing = [
            name
            for name in ("run_dir", "case", "arm")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(f"missing worker arguments: {missing}")
        return _worker(args)
    return _coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
