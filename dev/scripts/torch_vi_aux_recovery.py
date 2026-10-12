"""Recover auxiliary diagnostics from preserved Stage 4 primary fits.

This runner exists for the three ``warped``/seed-1703 workers whose complete
fits were committed to ``raw_primary.npz`` before an auxiliary divergence
report failed on unequal theta shapes.  It reads those primary artifacts,
runs only the corrected kernel gate, a fixed-100 replay, and five-stream
candidate rescoring, and writes a separate recovery artifact.

The coordinator starts a fresh process for each requested arm.  There is no
code path that calls ``run_step`` with ``fixed_updates=None``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

REPOSITORY = Path(__file__).resolve().parents[2]
STAGE4_RUNS = Path("dev/scripts/runs/stage4_torch")
DEFAULT_CAMPAIGN = STAGE4_RUNS / "campaign"
DEFAULT_OUTPUT = STAGE4_RUNS / "aux_recovery"
DEFAULT_CPU_TORCH_OVERLAY = Path("dev/scripts/runs/svbmc_compat_20260908/deps")
DEFAULT_CUDA_TORCH_OVERLAY = STAGE4_RUNS / "deps_cuda"
DEFAULT_TIMEOUT_S = 5 * 60
CASE = "warped"
SEED = 1703
RESCORE_SEEDS = tuple(SEED * 10_000 + 700 + i for i in range(5))
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
ARMS = {
    "numpy_cpu": ("numpy", "cpu"),
    "torch_cpu": ("torch", "cpu"),
    "torch_cuda": ("torch", "cuda"),
}
RETAINED_PREFIXES = (
    "selected_start:",
    "midpoint:",
    "endpoint:",
    "prune_attempt:",
    "final:",
    "boost_pre:",
    "boost_returned:",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_files(run_dir: Path) -> dict[str, Path]:
    files = {
        "result.json": run_dir / "result.json",
        "raw_primary.npz": run_dir / "raw_primary.npz",
    }
    failure = run_dir / "failure.json"
    if failure.is_file():
        files["failure.json"] = failure
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"primary artifact {run_dir} lacks required files {missing}"
        )
    return files


def _fingerprints(files: Mapping[str, Path]) -> dict[str, Any]:
    return {
        name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for name, path in files.items()
    }


def _same_fingerprints(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return {
        key: (value["bytes"], value["sha256"]) for key, value in left.items()
    } == {
        key: (value["bytes"], value["sha256"]) for key, value in right.items()
    }


class _ArtifactLoader:
    """Decode and validate numerical leaves from one primary artifact."""

    def __init__(self, run_dir: Path, np: Any) -> None:
        self.run_dir = run_dir.resolve()
        self.np = np
        self.archives: dict[str, Any] = {}
        self.loaded_arrays = 0

    def close(self) -> None:
        for archive in self.archives.values():
            archive.close()

    def _array(self, descriptor: Mapping[str, Any]) -> Any:
        from torch_vi_benchmark import _hash_array

        archive_name = descriptor.get("archive")
        if archive_name != "raw_primary.npz":
            raise ValueError(
                f"unexpected primary array archive {archive_name!r}"
            )
        archive_path = (self.run_dir / archive_name).resolve()
        if archive_path.parent != self.run_dir:
            raise ValueError(
                f"array archive escapes primary run: {archive_path}"
            )
        archive = self.archives.get(archive_name)
        if archive is None:
            archive = self.np.load(archive_path, allow_pickle=False)
            self.archives[archive_name] = archive
        key = descriptor["npz"]
        if key not in archive.files:
            raise KeyError(f"{archive_name} lacks stored array {key}")
        array = self.np.array(archive[key], copy=True)
        expected_shape = tuple(int(item) for item in descriptor["shape"])
        if array.shape != expected_shape:
            raise ValueError(
                f"{key} shape {array.shape} does not match {expected_shape}"
            )
        if str(array.dtype) != descriptor["dtype"]:
            raise ValueError(
                f"{key} dtype {array.dtype} does not match "
                f"{descriptor['dtype']}"
            )
        actual_hash = _hash_array(self.np.ascontiguousarray(array))
        if actual_hash != descriptor["sha256"]:
            raise ValueError(
                f"{key} hash {actual_hash} does not match saved descriptor"
            )
        self.loaded_arrays += 1
        return array

    def decode(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            if {"npz", "archive", "shape", "dtype", "sha256"}.issubset(value):
                return self._array(value)
            return {key: self.decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.decode(item) for item in value]
        return value


def _restore_vp(payload: Mapping[str, Any], template: Any, np: Any) -> Any:
    import copy

    vp = copy.deepcopy(template)
    D = int(payload["D"] if "D" in payload else template.D)
    K = int(payload["K"])
    if D != int(template.D):
        raise ValueError(f"saved VP D={D} differs from fixture D={template.D}")
    expected = {
        "mu": (D, K),
        "sigma": (1, K),
        "lambd": (D, 1),
        "w": (1, K),
        "eta": (1, K),
    }
    arrays = {}
    for name, shape in expected.items():
        arrays[name] = np.asarray(payload[name], dtype=np.float64).copy()
        if arrays[name].shape != shape and not (
            name == "eta" and arrays[name].shape == (K,)
        ):
            raise ValueError(
                f"saved VP {name} shape {arrays[name].shape} differs from {shape}"
            )
    vp.K = K
    for name, value in arrays.items():
        setattr(vp, name, value)
    vp._mode = None
    return vp


def _restore_candidates(
    rows: Sequence[Mapping[str, Any]], template: Any, np: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = {}
    checks = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if not candidate_id.startswith(RETAINED_PREFIXES):
            continue
        candidate = _restore_vp(row, template, np)
        stored_theta = np.asarray(row["theta"], dtype=np.float64)
        rebuilt_theta = np.asarray(
            __import__("copy").deepcopy(candidate).get_parameters(),
            dtype=np.float64,
        )
        shape_match = stored_theta.shape == rebuilt_theta.shape
        checks.append(
            {
                "candidate_id": candidate_id,
                "kind": row.get("kind"),
                "K": int(candidate.K),
                "stored_theta_shape": list(stored_theta.shape),
                "rebuilt_theta_shape": list(rebuilt_theta.shape),
                "theta_shape_match": shape_match,
                "theta_max_abs": (
                    float(np.max(np.abs(stored_theta - rebuilt_theta)))
                    if shape_match and stored_theta.size
                    else None
                ),
            }
        )
        if not shape_match:
            raise ValueError(
                f"candidate {candidate_id} theta shape changed during rebuild"
            )
        candidates[candidate_id] = candidate
    if not candidates:
        raise ValueError("primary artifact contains no retained candidates")
    return candidates, checks


def _max_abs(left: Any, right: Any, np: Any) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    shape_match = left.shape == right.shape
    return {
        "shape_match": shape_match,
        "complete_shape": list(left.shape),
        "fixed_shape": list(right.shape),
        "max_abs": (
            float(np.max(np.abs(left - right)))
            if shape_match and left.size
            else 0.0
            if shape_match
            else None
        ),
    }


def _shape_safe_divergence(
    complete_vp: Any, fixed_vp: Any, np: Any
) -> dict[str, Any]:
    import copy

    complete_theta = np.asarray(
        copy.deepcopy(complete_vp).get_parameters(), dtype=np.float64
    )
    fixed_theta = np.asarray(
        copy.deepcopy(fixed_vp).get_parameters(), dtype=np.float64
    )
    complete_mean, complete_cov = complete_vp.moments(
        orig_flag=False, cov_flag=True
    )
    fixed_mean, fixed_cov = fixed_vp.moments(orig_flag=False, cov_flag=True)
    fields = {
        name: _max_abs(getattr(complete_vp, name), getattr(fixed_vp, name), np)
        for name in ("mu", "sigma", "lambd", "w", "eta")
    }
    return {
        "available": True,
        "complete_K": int(complete_vp.K),
        "fixed_K": int(fixed_vp.K),
        "same_K": int(complete_vp.K) == int(fixed_vp.K),
        "theta": _max_abs(complete_theta, fixed_theta, np),
        "mean": _max_abs(complete_mean, fixed_mean, np),
        "covariance": _max_abs(complete_cov, fixed_cov, np),
        "parameters": fields,
        "interpretation": (
            "endpoint divergence from one common input and seed; max_abs is "
            "omitted whenever shapes differ"
        ),
    }


def _validate_primary_tree(
    tree: Mapping[str, Any], arm: str, backend: str, device: str
) -> None:
    expected = {
        "case": CASE,
        "seed": SEED,
        "backend": backend,
        "device": device,
        "phase": "complete_fit_saved",
    }
    differences = {
        key: (tree.get(key), value)
        for key, value in expected.items()
        if tree.get(key) != value
    }
    if differences:
        raise ValueError(f"primary {arm} metadata mismatch: {differences}")
    if tree.get("status") not in {
        "partial",
        "failed_after_complete",
        "timeout_after_complete",
    }:
        raise ValueError(
            f"primary {arm} has unexpected status {tree.get('status')!r}"
        )
    if not tree.get("complete_run"):
        raise ValueError(f"primary {arm} has no saved complete_run")


def _worker(args: argparse.Namespace) -> int:
    for key, value in THREAD_ENV.items():
        os.environ[key] = value
    started = time.perf_counter()
    run_dir = Path(args.run_dir).resolve()
    campaign = Path(args.campaign).resolve()
    source_dir = campaign / "runs" / f"{args.arm}_{CASE}_seed{SEED}"
    run_dir.mkdir(parents=True, exist_ok=False)
    before = None
    try:
        import numpy as np
        from torch_vi_benchmark import (
            _atomic_json,
            _configure_torch_threads,
            _kernel_diagnostics,
            _save_artifact,
            _sync_cuda,
        )
        from torch_vi_fixtures import environment_manifest, load_workload
        from torch_vi_step import rescore_candidates, run_step

        backend, device = ARMS[args.arm]
        torch = None
        torch_threads = {"applicable": False, "reason": "NumPy backend"}
        if backend == "torch":
            import torch as torch_module
            import torch_vi_core  # noqa: F401 - validate selected environment

            torch = torch_module
            torch_threads = _configure_torch_threads(torch)
            if not torch_threads["matches_request"]:
                raise RuntimeError(
                    f"invalid Torch thread settings: {torch_threads}"
                )
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA recovery arm requested but CUDA is unavailable"
                )

        files = _source_files(source_dir)
        before = _fingerprints(files)
        tree = json.loads(files["result.json"].read_text(encoding="utf-8"))
        _validate_primary_tree(tree, args.arm, backend, device)

        loader = _ArtifactLoader(source_dir, np)
        try:
            vp_payload = loader.decode(tree["complete_run"]["vp"])
            candidate_rows = loader.decode(tree["complete_run"]["candidates"])
        finally:
            loader.close()

        state_started = time.perf_counter()
        state = load_workload(CASE, SEED)
        state_wall = time.perf_counter() - state_started
        complete_vp = _restore_vp(vp_payload, state["vp"], np)
        candidate_vps, reconstruction = _restore_candidates(
            candidate_rows, state["vp"], np
        )
        reconstructed_run = {
            "_context": {
                "gp": state["gp"],
                "options": state["options"],
                "candidate_vps": candidate_vps,
            }
        }

        kernel_started = time.perf_counter()
        kernel = _kernel_diagnostics(state, backend, device, SEED)
        if torch is not None:
            _sync_cuda(torch, device)
        kernel_wall = time.perf_counter() - kernel_started

        fixed_started = time.perf_counter()
        fixed_run = run_step(
            state,
            backend=backend,
            device=device,
            seed=SEED,
            boost=state["boost"],
            fixed_updates=100,
            fast_opts_N=state.get("fast_opts_N"),
            slow_opts_N=state.get("slow_opts_N"),
            K=state["K"],
        )
        if torch is not None:
            _sync_cuda(torch, device)
        fixed_wall = time.perf_counter() - fixed_started

        rescore_started = time.perf_counter()
        rescored = rescore_candidates(
            reconstructed_run,
            seeds=RESCORE_SEEDS,
        )
        rescore_wall = time.perf_counter() - rescore_started

        divergence = _shape_safe_divergence(complete_vp, fixed_run["vp"], np)
        after = _fingerprints(files)
        preserved = _same_fingerprints(before, after)
        if not preserved:
            raise RuntimeError(
                "a preserved primary artifact changed during recovery"
            )

        environment = environment_manifest()
        environment["torch_runtime_threads"] = torch_threads
        result = {
            "schema": 1,
            "status": "gate_failed" if not kernel["all_pass"] else "ok",
            "mode": "auxiliary_recovery",
            "arm": args.arm,
            "case": CASE,
            "seed": SEED,
            "backend": backend,
            "device": device,
            "primary_fit_executed": False,
            "operations": [
                "corrected_kernel_diagnostics",
                "fixed_100_replay",
                "five_stream_saved_candidate_rescore",
                "shape_safe_divergence",
            ],
            "source_primary": {
                "run_dir": str(source_dir),
                "status": tree.get("status"),
                "phase": tree.get("phase"),
                "fingerprints_before": before,
                "fingerprints_after": after,
                "preserved": preserved,
                "saved_source_hashes": tree.get("environment", {}).get(
                    "source_hashes"
                ),
            },
            "environment": environment,
            "kernel": kernel,
            "fixed_100_run": fixed_run,
            "rescored_candidates": rescored,
            "fixed_100_divergence": divergence,
            "candidate_reconstruction": {
                "saved_count": len(candidate_rows),
                "retained_count": len(candidate_vps),
                "loaded_array_descriptors": loader.loaded_arrays,
                "checks": reconstruction,
            },
            "timing": {
                "state_rebuild_s": state_wall,
                "kernel_host_wall_s": kernel_wall,
                "fixed_100_host_wall_s": fixed_wall,
                "candidate_rescore_host_wall_s": rescore_wall,
                "worker_wall_s": time.perf_counter() - started,
            },
        }
        _save_artifact(run_dir, result)
        return 1 if result["status"] == "gate_failed" else 0
    except Exception as exc:
        failure = {
            "schema": 1,
            "status": "failed",
            "mode": "auxiliary_recovery",
            "arm": args.arm,
            "case": CASE,
            "seed": SEED,
            "primary_fit_executed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "worker_wall_s": time.perf_counter() - started,
        }
        if before is not None:
            failure["source_fingerprints_before"] = before
            try:
                after = _fingerprints(_source_files(source_dir))
                failure["source_fingerprints_after"] = after
                failure["source_primary_preserved"] = _same_fingerprints(
                    before, after
                )
            except Exception as fingerprint_exc:
                failure["fingerprint_error"] = repr(fingerprint_exc)
        try:
            from torch_vi_benchmark import _atomic_json

            _atomic_json(run_dir / "failure.json", failure)
        except Exception:
            (run_dir / "failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return 1


def _selected_arms(text: str) -> tuple[str, ...]:
    if text == "all":
        return tuple(ARMS)
    selected = tuple(
        dict.fromkeys(part.strip() for part in text.split(",") if part.strip())
    )
    unknown = set(selected) - set(ARMS)
    if not selected or unknown:
        raise ValueError(
            f"--arms must be all or a subset of {tuple(ARMS)}; got {selected}"
        )
    return selected


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _strip_pythonpath_overlays(
    environment: dict[str, str], overlays: Sequence[Path]
) -> None:
    """Remove experiment Torch overlays before selecting a worker runtime."""
    existing = environment.get("PYTHONPATH", "")
    overlay_keys = {os.path.normcase(str(path.resolve())) for path in overlays}
    entries = [entry for entry in existing.split(os.pathsep) if entry]
    retained = [
        entry
        for entry in entries
        if os.path.normcase(str(Path(entry).resolve())) not in overlay_keys
    ]
    if retained:
        environment["PYTHONPATH"] = os.pathsep.join(retained)
    else:
        environment.pop("PYTHONPATH", None)


def _coordinator(args: argparse.Namespace) -> int:
    campaign = Path(args.campaign).resolve()
    out = Path(args.out).resolve()
    if out == campaign or _inside(out, campaign):
        raise ValueError(
            "recovery output must be separate from the preserved campaign"
        )
    if not campaign.is_dir():
        raise FileNotFoundError(
            f"campaign directory does not exist: {campaign}"
        )
    arms = _selected_arms(args.arms)
    cpu_overlay = Path(args.cpu_torch_overlay).resolve()
    cuda_overlay = Path(args.cuda_torch_overlay).resolve()
    requested_overlays = {
        "torch_cpu": cpu_overlay,
        "torch_cuda": cuda_overlay,
    }
    for arm in arms:
        overlay = requested_overlays.get(arm)
        if overlay is not None and not overlay.is_dir():
            raise FileNotFoundError(
                f"Torch overlay for {arm} does not exist: {overlay}"
            )
    out.mkdir(parents=True, exist_ok=True)

    records = []
    failures = 0
    for arm in arms:
        backend, device = ARMS[arm]
        source_dir = campaign / "runs" / f"{arm}_{CASE}_seed{SEED}"
        _source_files(source_dir)
        run_id = f"aux_{arm}_{CASE}_seed{SEED}"
        run_dir = out / "runs" / run_id
        if run_dir.exists():
            raise FileExistsError(
                f"{run_dir} exists; recovery runs are never overwritten or retried"
            )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--campaign",
            str(campaign),
            "--run-dir",
            str(run_dir),
            "--arm",
            arm,
        ]
        environment = os.environ.copy()
        environment.update(THREAD_ENV)
        _strip_pythonpath_overlays(environment, (cpu_overlay, cuda_overlay))
        overlay = requested_overlays.get(arm)
        if overlay is not None:
            existing = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(overlay)
                if not existing
                else str(overlay) + os.pathsep + existing
            )
        log_path = out / "runs" / f"{run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        arm_started = time.perf_counter()
        timed_out = False
        returncode = None
        try:
            with log_path.open("w", encoding="utf-8", newline="\n") as log:
                completed = subprocess.run(
                    command,
                    cwd=REPOSITORY,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
            returncode = completed.returncode
            failures += int(returncode != 0)
        except subprocess.TimeoutExpired:
            timed_out = True
            failures += 1
        records.append(
            {
                "arm": arm,
                "backend": backend,
                "device": device,
                "source_dir": str(source_dir),
                "run_dir": str(run_dir),
                "log": str(log_path),
                "returncode": returncode,
                "timed_out": timed_out,
                "wall_s": time.perf_counter() - arm_started,
                "retry": False,
            }
        )
        from torch_vi_benchmark import _atomic_json

        _atomic_json(
            out / "summary.json",
            {
                "schema": 1,
                "status": "failed" if failures else "running",
                "mode": "auxiliary_recovery",
                "case": CASE,
                "seed": SEED,
                "primary_fits_executed": 0,
                "records": records,
            },
        )
    from torch_vi_benchmark import _atomic_json

    _atomic_json(
        out / "summary.json",
        {
            "schema": 1,
            "status": "failed" if failures else "ok",
            "mode": "auxiliary_recovery",
            "case": CASE,
            "seed": SEED,
            "primary_fits_executed": 0,
            "records": records,
        },
    )
    return 1 if failures else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=str(DEFAULT_CAMPAIGN))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--arms",
        default="all",
        help="all or comma-separated numpy_cpu,torch_cpu,torch_cuda",
    )
    parser.add_argument(
        "--cpu-torch-overlay",
        default=str(DEFAULT_CPU_TORCH_OVERLAY),
        help="CPU-only Torch overlay prepended for the torch_cpu worker",
    )
    parser.add_argument(
        "--cuda-torch-overlay",
        default=str(DEFAULT_CUDA_TORCH_OVERLAY),
        help="CUDA Torch overlay prepended for the torch_cuda worker",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="per-arm auxiliary-only timeout in seconds (maximum 300)",
    )
    parser.add_argument(
        "--worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--run-dir", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=tuple(ARMS), help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout <= 0 or args.timeout > DEFAULT_TIMEOUT_S:
        raise SystemExit(f"--timeout must be in (0, {DEFAULT_TIMEOUT_S}]")
    if args.worker:
        if args.run_dir is None or args.arm is None:
            raise SystemExit("recovery worker requires --run-dir and --arm")
        return _worker(args)
    return _coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
