"""Bounded runner for the Stage 4 Torch variational-step experiment.

The normal CLI is a coordinator.  It launches one fresh worker process for
each requested case/seed, enforces the five-minute fit and 45-minute backend
caps, retains failures without retrying, and rebuilds aggregate JSON/Markdown
after every worker.  Numerical libraries are imported only in workers so the
thread environment and import/context costs are observable.

Examples (the output directory and case selection are intentionally required)::

    python dev/scripts/torch_vi_benchmark.py --out RUNS --cases all_complete \
        --backend numpy --device cpu
    python dev/scripts/torch_vi_benchmark.py --out RUNS --cases all_kernels \
        --backend torch --device cuda

Raw arrays are stored in per-run ``raw.npz`` files and referenced from strict
``result.json`` trees.  Existing run directories are never overwritten and a
failed or timed-out run is never retried automatically.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_FIT_TIMEOUT_S = 5 * 60
DEFAULT_BACKEND_CAP_S = 45 * 60
MAX_COMPLETE_PER_BACKEND = 24
MAX_COMPLETE_OVERALL = 72
MAX_CLEAN_COMPLETE_PER_BACKEND = 8
MAX_CLEAN_COMPLETE_OVERALL = 24
MICRO_BLOCKS = 5
MICRO_WARMUPS = 3
MICRO_BLOCK_CAP_S = 1.0
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
DEFAULT_PRODUCTION_CONTROL_CASES: tuple[str, ...] = ()


def _strict_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"nonfinite": repr(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(v) for v in value]
    if hasattr(value, "item"):
        with contextlib.suppress(TypeError, ValueError, RuntimeError):
            return _strict_json(value.item())
    return repr(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            _strict_json(value), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    tmp.replace(path)


def _append_ledger(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        _strict_json(value),
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _hash_array(array: Any) -> str:
    view = memoryview(array).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), 1024 * 1024):
        digest.update(view[start : start + 1024 * 1024])
    return digest.hexdigest()


class ArtifactEncoder:
    """Move numerical leaves to NPZ while preserving a reviewable JSON tree."""

    def __init__(self, archive: str = "raw.npz") -> None:
        self.arrays: dict[str, Any] = {}
        self._counter = 0
        self.archive = archive

    def _key(self, hint: str) -> str:
        clean = "".join(c if c.isalnum() or c in "_-" else "_" for c in hint)
        self._counter += 1
        return f"{self._counter:05d}_{clean[-80:]}"

    def encode(self, value: Any, hint: str = "root") -> Any:
        import numpy as np

        # Torch remains optional for a NumPy worker.  Detect by module/type
        # rather than importing it solely for serialization.
        if value.__class__.__module__.startswith("torch") and hasattr(
            value, "detach"
        ):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            key = self._key(hint)
            self.arrays[key] = array
            return {
                "npz": key,
                "archive": self.archive,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _hash_array(array),
            }
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            return _strict_json(value)
        if isinstance(value, np.generic):
            return self.encode(value.item(), hint)
        if is_dataclass(value):
            return self.encode(asdict(value), hint)
        if isinstance(value, Mapping):
            return {
                str(key): self.encode(item, f"{hint}/{key}")
                for key, item in value.items()
                if key != "_context"
            }
        if isinstance(value, (list, tuple)):
            return [
                self.encode(item, f"{hint}/{i}")
                for i, item in enumerate(value)
            ]
        # Preserve a VP outcome without pickling implementation objects.
        if all(
            hasattr(value, name)
            for name in ("D", "K", "mu", "sigma", "lambd", "w")
        ):
            payload = {
                "type": f"{value.__class__.__module__}.{value.__class__.__name__}",
                "D": int(value.D),
                "K": int(value.K),
                "mu": value.mu,
                "sigma": value.sigma,
                "lambd": value.lambd,
                "w": value.w,
                "eta": getattr(value, "eta", None),
                "stats": getattr(value, "stats", None),
            }
            with contextlib.suppress(Exception):
                mean, covariance = value.moments(
                    orig_flag=False, cov_flag=True
                )
                payload["transformed_moments"] = {
                    "mean": mean,
                    "covariance": covariance,
                }
            return self.encode(payload, hint)
        if callable(value):
            return {
                "callable": f"{getattr(value, '__module__', '')}."
                f"{getattr(value, '__qualname__', type(value).__name__)}"
            }
        return {
            "type": f"{value.__class__.__module__}.{value.__class__.__name__}",
            "repr": repr(value),
        }


def _save_artifact(
    run_dir: Path, result: Mapping[str, Any], *, stem: str = "raw"
) -> None:
    import numpy as np

    archive = f"{stem}.npz"
    encoder = ArtifactEncoder(archive)
    tree = encoder.encode(result)
    npz_tmp = run_dir / f"{stem}.tmp.npz"
    np.savez_compressed(npz_tmp, **encoder.arrays)
    npz_tmp.replace(run_dir / archive)
    _atomic_json(run_dir / "result.json", tree)


def _sync_cuda(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _cuda_context(torch: Any, device: str) -> dict[str, Any]:
    if not device.startswith("cuda"):
        return {"requested": False, "initialized": False, "wall_s": 0.0}
    started = time.perf_counter()
    probe = torch.ones(1, dtype=torch.float64, device=device)
    probe = probe + 1.0
    _sync_cuda(torch, device)
    properties = torch.cuda.get_device_properties(device)
    return {
        "requested": True,
        "initialized": True,
        "wall_s": time.perf_counter() - started,
        "name": properties.name,
        "total_memory": int(properties.total_memory),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch_cuda": torch.version.cuda,
    }


def _configure_torch_threads(torch: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "requested_intraop": 1,
        "requested_interop": 1,
        "errors": [],
    }
    try:
        torch.set_num_threads(1)
    except RuntimeError as exc:
        record["errors"].append(f"set_num_threads: {exc}")
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        record["errors"].append(f"set_num_interop_threads: {exc}")
    try:
        record["actual_intraop"] = int(torch.get_num_threads())
    except RuntimeError as exc:
        record["actual_intraop"] = None
        record["errors"].append(f"get_num_threads: {exc}")
    try:
        record["actual_interop"] = int(torch.get_num_interop_threads())
    except RuntimeError as exc:
        record["actual_interop"] = None
        record["errors"].append(f"get_num_interop_threads: {exc}")
    record["matches_request"] = (
        record.get("actual_intraop") == 1
        and record.get("actual_interop") == 1
        and not record["errors"]
    )
    return record


def _memory_record(torch: Any, device: str) -> dict[str, Any]:
    rss = _peak_rss()
    cuda = None
    if torch is not None and device.startswith("cuda"):
        _sync_cuda(torch, device)
        cuda = {
            "max_allocated": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved": int(torch.cuda.max_memory_reserved(device)),
        }
    return {
        "process_peak_rss_bytes": rss["bytes"],
        "process_peak_rss": rss,
        "cuda": cuda,
    }


def _peak_rss() -> dict[str, Any]:
    """Process peak RSS with explicit native signatures and error capture."""

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            get_process = kernel32.GetCurrentProcess
            get_process.argtypes = []
            get_process.restype = wintypes.HANDLE
            get_memory = psapi.GetProcessMemoryInfo
            get_memory.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            ]
            get_memory.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not get_memory(
                get_process(), ctypes.byref(counters), counters.cb
            ):
                code = ctypes.get_last_error()
                raise ctypes.WinError(code)
            return {
                "bytes": int(counters.PeakWorkingSetSize),
                "source": "GetProcessMemoryInfo.PeakWorkingSetSize",
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - platform API failure
            return {
                "bytes": None,
                "source": "GetProcessMemoryInfo.PeakWorkingSetSize",
                "error": f"{type(exc).__name__}: {exc}",
            }
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return {
            "bytes": value if sys.platform == "darwin" else value * 1024,
            "source": "resource.RUSAGE_SELF.ru_maxrss",
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - platform API failure
        return {
            "bytes": None,
            "source": "resource.RUSAGE_SELF.ru_maxrss",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _numpy_control(np: Any) -> dict[str, Any]:
    """Short unchanged NumPy control used before and after each fit."""

    rng = np.random.default_rng(916_247)
    left = rng.standard_normal((192, 192))
    right = rng.standard_normal((192, 192))
    walls = []
    checksum = None
    for _ in range(3):
        started = time.perf_counter()
        product = left @ right
        walls.append(time.perf_counter() - started)
        checksum = float(product[0, 0] + product[-1, -1])
    return {
        "walls_s": walls,
        "median_s": float(np.median(walls)),
        "checksum": checksum,
    }


def _native_rng_timing(
    torch: Any, state: Mapping[str, Any], device: str
) -> dict[str, Any]:
    vp = state["vp"]
    K, D = int(state.get("K", vp.K)), int(vp.D)
    options = state["options"]
    if state.get("boost"):
        total = options.eval("ns_ent_boost", {"K": K})
    else:
        total = options.eval("ns_ent", {"K": K})
    per_component = int(math.ceil(float(total) / K))
    per_component += per_component % 2
    shape = (K, per_component // 2, D)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(state["seed"]))
    for _ in range(MICRO_WARMUPS):
        torch.randn(
            shape, dtype=torch.float64, device=device, generator=generator
        )
    _sync_cuda(torch, device)
    walls = []
    for _ in range(MICRO_BLOCKS):
        started = time.perf_counter()
        torch.randn(
            shape, dtype=torch.float64, device=device, generator=generator
        )
        _sync_cuda(torch, device)
        elapsed = time.perf_counter() - started
        walls.append(elapsed)
        if elapsed > MICRO_BLOCK_CAP_S:
            break
    return {
        "shape": list(shape),
        "dtype": "torch.float64",
        "device": device,
        "warmups": MICRO_WARMUPS,
        "walls_s": walls,
        "block_cap_s": MICRO_BLOCK_CAP_S,
    }


def _error_rows(
    expected: Any, actual: Any, rtol: float, atol: float
) -> dict[str, Any]:
    import numpy as np

    from pyvbmc.testing.oracles._oracles import compare

    # Check the original arrays before the oracle comparator performs its
    # documented float conversion.  A numerically equal float32 output is a
    # prototype failure because Stage 4 is explicitly float64-only.
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    expected_float64 = bool(expected_array.dtype == np.float64)
    actual_float64 = bool(actual_array.dtype == np.float64)
    _, max_abs, max_scaled, numerical_pass = compare(
        {"value": expected_array},
        {"value": actual_array},
        rtol,
        atol,
    )[0]
    return {
        "shape_expected": list(expected_array.shape),
        "shape_actual": list(actual_array.shape),
        "dtype_expected": str(expected_array.dtype),
        "dtype_actual": str(actual_array.dtype),
        "float64_expected": expected_float64,
        "float64_actual": actual_float64,
        "max_abs": max_abs,
        "max_scaled": max_scaled,
        "pass": bool(numerical_pass and expected_float64 and actual_float64),
        "rtol": rtol,
        "atol": atol,
        "comparison": "oracle_lower_quartile_floor",
    }


class _ReplayRNG:
    """One exact NumPy standard-normal draw for common-random diagnostics."""

    def __init__(self, epsilon: Any) -> None:
        import numpy as np

        self._epsilon = np.array(epsilon, dtype=np.float64, copy=True)
        self.calls = 0

    def standard_normal(self, shape: Any) -> Any:
        if tuple(shape) != self._epsilon.shape:
            raise ValueError(
                f"entropy requested epsilon shape {tuple(shape)}, expected "
                f"{self._epsilon.shape}"
            )
        self.calls += 1
        return self._epsilon.copy()


def _kernel_inputs(state: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Freeze theta, soft bounds, and one raw half-antithetic draw."""

    import copy

    import numpy as np

    vp = copy.deepcopy(state["vp"])
    vp.rng = np.random.default_rng(seed)
    theta = np.asarray(vp.get_parameters(), dtype=np.float64).copy()
    K = int(vp.K)
    total = state["options"].eval("ns_ent", {"K": K})
    Ns = max(2, int(math.ceil(float(total) / K)))
    half = int(math.ceil(Ns / 2))
    epsilon = np.random.default_rng(seed + 91_337).standard_normal(
        (K, half, int(vp.D))
    )
    bounds = vp.get_bounds(state["gp"].X, state["options"], K)
    production_Ns = (
        0
        if K == 1 or state["optim_state"].get("entropy_switch", False)
        else Ns
    )
    return {
        "vp": vp,
        "theta": theta,
        "bounds": bounds,
        "Ns": Ns,
        "production_Ns": production_Ns,
        "epsilon": epsilon,
        "epsilon_meta": {
            "shape": list(epsilon.shape),
            "dtype": str(epsilon.dtype),
            "sha256": _hash_array(np.ascontiguousarray(epsilon)),
            "draw_seed": seed + 91_337,
        },
    }


def _numpy_kernel_values(
    state: Mapping[str, Any], inputs: Mapping[str, Any]
) -> tuple[dict[str, Any], Any]:
    """Evaluate the current NumPy formulas with an injected epsilon draw."""

    import copy

    import numpy as np

    from pyvbmc.entropy.entlb_vbmc import entlb_vbmc
    from pyvbmc.entropy.entmc_vbmc import entmc_vbmc
    from pyvbmc.vbmc.variational_optimization import _gp_log_joint, _neg_elcbo

    gp = state["gp"]
    theta = inputs["theta"]
    bounds = inputs["bounds"]
    Ns = int(inputs["Ns"])
    epsilon = inputs["epsilon"]
    vp = copy.deepcopy(inputs["vp"])
    vp.set_parameters(theta)
    if vp.optimize_weights:
        eta = np.asarray(theta[-vp.K :], dtype=np.float64).copy()
        eta -= np.max(eta)
        vp.eta = eta.reshape(1, -1)
    flags = (
        bool(vp.optimize_mu),
        bool(vp.optimize_sigma),
        bool(vp.optimize_lambd),
        bool(vp.optimize_weights),
    )

    G, dG, _, _, _ = _gp_log_joint(vp, gp, flags, True, True, False, False)
    _, _, varG, _, var_ss, I_sk, J_sjk = _gp_log_joint(
        vp, gp, (False, False, False, False), True, True, True, True
    )
    H_det, dH_det = entlb_vbmc(vp, flags, True)
    vp._rng = _ReplayRNG(epsilon)
    H_mc, dH_mc = entmc_vbmc(vp, Ns, flags, True)

    def objective(entropy_samples: int) -> tuple[Any, ...]:
        objective_vp = copy.deepcopy(inputs["vp"])
        objective_vp._rng = _ReplayRNG(epsilon)
        return _neg_elcbo(
            theta,
            gp,
            objective_vp,
            0.0,
            entropy_samples,
            True,
            False,
            bounds,
        )

    F_mc, dF_mc, _, _, _ = objective(Ns)
    F_det, dF_det, _, _, _ = objective(0)
    production_Ns = int(inputs["production_Ns"])
    F_production, dF_production, _, _, _ = objective(production_Ns)
    values = {
        "G": np.asarray(G),
        "dG": np.asarray(dG),
        "varG": np.asarray(varG),
        "var_ss": np.asarray(var_ss),
        "I_sk": np.asarray(I_sk),
        "J_sjk": np.asarray(J_sjk),
        "H_mc": np.asarray(H_mc),
        "dH_mc": np.asarray(dH_mc),
        "H_det": np.asarray(H_det),
        "dH_det": np.asarray(dH_det),
        "F_mc": np.asarray(F_mc),
        "dF_mc": np.asarray(dF_mc),
        "F_det": np.asarray(F_det),
        "dF_det": np.asarray(dF_det),
        "F_production": np.asarray(F_production),
        "dF_production": np.asarray(dF_production),
    }

    timing_vp = copy.deepcopy(inputs["vp"])

    def timed_objective() -> None:
        timing_vp._rng = _ReplayRNG(epsilon)
        _neg_elcbo(
            theta,
            gp,
            timing_vp,
            0.0,
            production_Ns,
            True,
            False,
            bounds,
        )

    return values, timed_objective


def _torch_kernel_values(
    state: Mapping[str, Any], inputs: Mapping[str, Any], device: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    """Evaluate the resident Torch kernels and account for conversion."""

    import torch
    from torch_vi_core import (
        TorchBounds,
        TorchGP,
        TorchVPTemplate,
        as_float64_tensor,
        decode_theta,
        entropy_lower_bound,
        entropy_mc,
        gp_log_joint,
        neg_elcbo,
    )

    conversion: dict[str, Any] = {}
    _sync_cuda(torch, device)
    started = time.perf_counter()
    tgp = TorchGP.from_gp(state["gp"], device)
    _sync_cuda(torch, device)
    conversion["gp_s"] = time.perf_counter() - started
    started = time.perf_counter()
    tvp = TorchVPTemplate.from_vp(inputs["vp"], device)
    _sync_cuda(torch, device)
    conversion["vp_s"] = time.perf_counter() - started
    started = time.perf_counter()
    tbounds = TorchBounds.from_dict(inputs["bounds"], device)
    theta = as_float64_tensor(inputs["theta"], device)
    epsilon = as_float64_tensor(inputs["epsilon"], device)
    _sync_cuda(torch, device)
    conversion["bounds_theta_epsilon_s"] = time.perf_counter() - started
    conversion["total_s"] = sum(conversion.values())

    def value_and_grad(kind: str) -> tuple[Any, Any]:
        theta_work = theta.detach().clone().requires_grad_(True)
        decoded = decode_theta(theta_work, tvp)
        if kind == "G":
            value = gp_log_joint(decoded, tgp)[0]
        elif kind == "H_mc":
            value = entropy_mc(decoded, int(inputs["Ns"]), epsilon)
        elif kind == "H_det":
            value = entropy_lower_bound(decoded)
        else:  # pragma: no cover - internal misuse guard
            raise ValueError(kind)
        gradient = torch.autograd.grad(value, theta_work)[0]
        return value.detach().cpu().numpy(), gradient.detach().cpu().numpy()

    G, dG = value_and_grad("G")
    H_mc, dH_mc = value_and_grad("H_mc")
    H_det, dH_det = value_and_grad("H_det")
    theta_var = theta.detach()
    decoded_var = decode_theta(theta_var, tvp)
    raw_variance: dict[str, Any] = {}
    _, varG, var_ss, I_sk, J_sjk = gp_log_joint(
        decoded_var,
        tgp,
        compute_var=True,
        separate_K=True,
        diagnostics=raw_variance,
    )

    def objective(entropy_samples: int) -> tuple[Any, Any]:
        result = neg_elcbo(
            theta,
            tgp,
            tvp,
            0.0,
            entropy_samples,
            True,
            False,
            tbounds,
            False,
            epsilon if entropy_samples > 0 else None,
        )
        return result[0].cpu().numpy(), result[1].cpu().numpy()

    F_mc, dF_mc = objective(int(inputs["Ns"]))
    F_det, dF_det = objective(0)
    F_production, dF_production = objective(int(inputs["production_Ns"]))

    def cpu(value: Any) -> Any:
        return value.detach().cpu().numpy()

    values = {
        "G": G,
        "dG": dG,
        "varG": cpu(varG),
        "var_ss": cpu(var_ss),
        "I_sk": cpu(I_sk),
        "J_sjk": cpu(J_sjk),
        "H_mc": H_mc,
        "dH_mc": dH_mc,
        "H_det": H_det,
        "dH_det": dH_det,
        "F_mc": F_mc,
        "dF_mc": dF_mc,
        "F_det": F_det,
        "dF_det": dF_det,
        "F_production": F_production,
        "dF_production": dF_production,
    }

    def timed_objective() -> None:
        neg_elcbo(
            theta,
            tgp,
            tvp,
            0.0,
            int(inputs["production_Ns"]),
            True,
            False,
            tbounds,
            False,
            epsilon if int(inputs["production_Ns"]) > 0 else None,
        )

    return values, conversion, raw_variance, timed_objective


def _microtime(call: Any, sync: Any = None) -> dict[str, Any]:
    for _ in range(MICRO_WARMUPS):
        call()
        if sync is not None:
            sync()
    walls = []
    for _ in range(MICRO_BLOCKS):
        if sync is not None:
            sync()
        started = time.perf_counter()
        call()
        if sync is not None:
            sync()
        walls.append(time.perf_counter() - started)
    return {
        "warmups": MICRO_WARMUPS,
        "blocks": MICRO_BLOCKS,
        "calls_per_block": 1,
        "walls_s": walls,
        "median_s": float(sorted(walls)[len(walls) // 2]),
        "block_cap_s": MICRO_BLOCK_CAP_S,
        "cap_exceeded": bool(any(wall > MICRO_BLOCK_CAP_S for wall in walls)),
    }


def _kernel_diagnostics(
    state: Mapping[str, Any], backend: str, device: str, seed: int
) -> dict[str, Any]:
    """Run same-input value/gradient gates and resident microtimings.

    NumPy and Torch receive the same fixed GP factors, VP theta, bounds, and
    raw half-antithetic epsilon.  A NumPy worker never imports Torch.
    """

    inputs = _kernel_inputs(state, seed)
    expected, numpy_call = _numpy_kernel_values(state, inputs)
    timing = {"numpy_resident": _microtime(numpy_call)}
    conversion = {}
    raw_variance = {}
    if backend == "torch":
        actual, conversion, raw_variance, torch_call = _torch_kernel_values(
            state, inputs, device
        )
        import torch

        timing["torch_resident"] = _microtime(
            torch_call, lambda: _sync_cuda(torch, device)
        )
    else:
        actual = expected
    variance = {"varG", "var_ss", "J_sjk"}
    solve = {
        "G",
        "dG",
        "F_mc",
        "dF_mc",
        "F_det",
        "dF_det",
        "F_production",
        "dF_production",
        "I_sk",
    }
    errors = {}
    for key in sorted(set(expected) & set(actual)):
        if key in variance:
            tolerance = (1e-3, 1e-8)
        elif key in solve:
            tolerance = (1e-6, 1e-10)
        else:
            tolerance = (1e-10, 1e-12)
        errors[key] = _error_rows(expected[key], actual[key], *tolerance)
    return {
        "errors": errors,
        "all_pass": bool(
            errors and all(row["pass"] for row in errors.values())
        ),
        "timing": timing,
        "conversion": conversion,
        "raw_variance": raw_variance,
        "epsilon": inputs["epsilon_meta"],
        "Ns": int(inputs["Ns"]),
        "production_Ns": int(inputs["production_Ns"]),
        "raw": {"numpy": expected, backend: actual},
    }


def _warm_objective_calls(
    state: Mapping[str, Any], backend: str, device: str, seed: int
) -> dict[str, Any]:
    """Warm exactly three common-input production-shape objective calls."""

    import copy

    import numpy as np

    input_started = time.perf_counter()
    inputs = _kernel_inputs(state, seed)
    input_wall = time.perf_counter() - input_started
    Ns = int(inputs["production_Ns"])
    theta = inputs["theta"]
    epsilon = inputs["epsilon"]
    transfer_wall = 0.0

    if backend == "numpy":
        from pyvbmc.vbmc.variational_optimization import _neg_elcbo

        vp = copy.deepcopy(inputs["vp"])

        def call() -> None:
            vp._rng = _ReplayRNG(epsilon)
            _neg_elcbo(
                theta,
                state["gp"],
                vp,
                0.0,
                Ns,
                True,
                False,
                inputs["bounds"],
            )

        sync = None
    else:
        import torch
        from torch_vi_core import (
            TorchBounds,
            TorchGP,
            TorchVPTemplate,
            as_float64_tensor,
            neg_elcbo,
        )

        transfer_started = time.perf_counter()
        tgp = TorchGP.from_gp(state["gp"], device)
        tvp = TorchVPTemplate.from_vp(inputs["vp"], device)
        tbounds = TorchBounds.from_dict(inputs["bounds"], device)
        ttheta = as_float64_tensor(theta, device)
        tepsilon = as_float64_tensor(epsilon, device)
        _sync_cuda(torch, device)
        transfer_wall = time.perf_counter() - transfer_started

        def call() -> None:
            neg_elcbo(
                ttheta,
                tgp,
                tvp,
                0.0,
                Ns,
                True,
                False,
                tbounds,
                False,
                tepsilon if Ns > 0 else None,
            )

        sync = lambda: _sync_cuda(torch, device)

    walls = []
    for _ in range(3):
        if sync is not None:
            sync()
        started = time.perf_counter()
        call()
        if sync is not None:
            sync()
        walls.append(time.perf_counter() - started)
    return {
        "calls": 3,
        "backend": backend,
        "device": device,
        "input_setup_s": input_wall,
        "fixed_state_transfer_s": transfer_wall,
        "walls_s": walls,
        "epsilon": inputs["epsilon_meta"],
        "Ns": Ns,
        "common_private_input": True,
    }


def _rescore_seeds(seed: int) -> tuple[int, ...]:
    # Same list for NumPy/Torch arms of one case/seed, disjoint from fit RNG.
    return tuple(seed * 10_000 + 700 + i for i in range(5))


def _replay_divergence(
    complete_run: Any, fixed_run: Any
) -> dict[str, Any] | None:
    if complete_run is None or fixed_run is None:
        return None
    import copy

    import numpy as np

    left = copy.deepcopy(complete_run.get("vp"))
    right = copy.deepcopy(fixed_run.get("vp"))
    if left is None or right is None:
        return {"available": False}
    left_theta = np.asarray(left.get_parameters(), dtype=np.float64)
    right_theta = np.asarray(right.get_parameters(), dtype=np.float64)
    left_mean, left_cov = left.moments(orig_flag=False, cov_flag=True)
    right_mean, right_cov = right.moments(orig_flag=False, cov_flag=True)
    theta_shape_match = left_theta.shape == right_theta.shape
    return {
        "available": True,
        "theta_shape_match": theta_shape_match,
        "complete_theta_shape": list(left_theta.shape),
        "fixed_theta_shape": list(right_theta.shape),
        "theta_max_abs": (
            float(np.max(np.abs(left_theta - right_theta)))
            if theta_shape_match
            else None
        ),
        "mean_max_abs": float(np.max(np.abs(left_mean - right_mean))),
        "covariance_max_abs": float(np.max(np.abs(left_cov - right_cov))),
        "complete_K": int(left.K),
        "fixed_K": int(right.K),
        "interpretation": "endpoint divergence from one common input and seed",
    }


def _worker(args: argparse.Namespace) -> int:
    worker_started = time.perf_counter()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    import_started = time.perf_counter()
    requested_complete = False
    thread_record = None
    try:
        import numpy as np
        from torch_vi_fixtures import environment_manifest, load_workload

        if args.trace_disabled:
            from torch_vi_step import run_step

            rescore_candidates = None
            run_numpy_control = None
        else:
            from torch_vi_step import (
                rescore_candidates,
                run_numpy_control,
                run_step,
            )

        torch = None
        if args.backend == "torch":
            import torch as _torch
            import torch_vi_core  # noqa: F401 - charged to prototype import

            torch = _torch
        import_wall = time.perf_counter() - import_started

        load_started = time.perf_counter()
        state = load_workload(args.case, args.seed)
        load_wall = time.perf_counter() - load_started
        spec = state["spec"]
        requested_complete = bool(spec.complete and not args.kernel_only)
        mode = (
            "kernel_only"
            if args.kernel_only
            else "clean_control"
            if args.trace_disabled
            else "traced"
        )
        if args.kernel_only and (
            args.trace_disabled or args.skip_auxiliary or args.warm_kernels
        ):
            raise ValueError(
                "--kernel-only cannot be combined with clean-control flags"
            )
        if args.trace_disabled and not (
            args.skip_auxiliary and args.warm_kernels
        ):
            raise ValueError(
                "--trace-disabled requires --skip-auxiliary and --warm-kernels"
            )
        if (
            args.skip_auxiliary or args.warm_kernels
        ) and not args.trace_disabled:
            raise ValueError(
                "--skip-auxiliary/--warm-kernels require --trace-disabled"
            )
        if args.backend == "numpy" and args.device != "cpu":
            raise ValueError("the NumPy backend only supports --device cpu")
        if args.backend == "torch":
            thread_record = _configure_torch_threads(torch)
            if not thread_record["matches_request"]:
                raise RuntimeError(
                    f"invalid Torch thread configuration: {thread_record}"
                )
            if (
                args.device.startswith("cuda")
                and not torch.cuda.is_available()
            ):
                raise RuntimeError(
                    "CUDA requested but torch.cuda.is_available() is false"
                )
            context_record = _cuda_context(torch, args.device)
        else:
            thread_record = {
                "applicable": False,
                "reason": "NumPy backend",
            }
            context_record = {
                "requested": False,
                "initialized": False,
                "wall_s": 0.0,
            }

        environment = environment_manifest()
        environment["torch_runtime_threads"] = thread_record
        if not environment["provenance_validation"]["valid"]:
            raise RuntimeError(
                "invalid experiment provenance: "
                f"{environment['provenance_validation']['errors']}"
            )
        warmup = (
            _warm_objective_calls(state, args.backend, args.device, args.seed)
            if args.warm_kernels
            else {"enabled": False, "calls": 0}
        )
        if torch is not None and args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        control_before = None if args.skip_auxiliary else _numpy_control(np)

        complete_run = None
        fixed_run = None
        production_control = None
        rescored = None
        moments = None
        fit_wall = 0.0
        fixed_wall = 0.0
        post_fit_memory = None
        if requested_complete:
            fit_started = time.perf_counter()
            run_kwargs = {
                "backend": args.backend,
                "device": args.device,
                "seed": args.seed,
                "boost": state["boost"],
                "fixed_updates": None,
                "fast_opts_N": state.get("fast_opts_N"),
                "slow_opts_N": state.get("slow_opts_N"),
                "K": state["K"],
            }
            if args.trace_disabled:
                complete_run = run_step(
                    state, trace_enabled=False, **run_kwargs
                )
            else:
                complete_run = run_step(state, **run_kwargs)
            if torch is not None:
                _sync_cuda(torch, args.device)
            fit_wall = time.perf_counter() - fit_started
            post_fit_memory = _memory_record(torch, args.device)
            vp = complete_run.get("vp")
            if vp is not None:
                mean, covariance = vp.moments(orig_flag=False, cov_flag=True)
                moments = {"mean": mean, "covariance": covariance}

            # Commit the complete fit before auxiliary gates.  If the outer
            # watchdog stops a later replay/rescore, raw events and candidate
            # outcomes from the primary fit remain reviewable.
            _save_artifact(
                run_dir,
                {
                    "schema": 1,
                    "status": "partial",
                    "phase": "complete_fit_saved",
                    "case": args.case,
                    "seed": args.seed,
                    "backend": args.backend,
                    "device": args.device,
                    "mode": mode,
                    "complete": True,
                    "requested_complete": True,
                    "contract": state["contract"],
                    "environment": environment,
                    "timing": {
                        "import_s": import_wall,
                        "state_rebuild_s": load_wall,
                        "cuda_context_s": context_record["wall_s"],
                        "complete_fit_host_wall_s": fit_wall,
                        "instrumented_complete_fit_s": complete_run.get(
                            "timing", {}
                        ).get("total"),
                        "warmup": warmup,
                    },
                    "cuda_context": context_record,
                    "complete_run": complete_run,
                    "post_fit_memory_before_diagnostics": post_fit_memory,
                    "transformed_moments": moments,
                },
                stem="raw_primary",
            )

        kernel = (
            _kernel_diagnostics(state, args.backend, args.device, args.seed)
            if args.kernel_only or not args.skip_auxiliary
            else None
        )

        if requested_complete and not args.skip_auxiliary:
            if args.production_control:
                assert run_numpy_control is not None
                control_started = time.perf_counter()
                production_control = run_numpy_control(
                    state,
                    seed=args.seed,
                    boost=state["boost"],
                    fast_opts_N=state.get("fast_opts_N"),
                    slow_opts_N=state.get("slow_opts_N"),
                    K=state["K"],
                )
                production_control["timing"]["host_wall_s"] = (
                    time.perf_counter() - control_started
                )
            if args.case != "deterministic":
                fixed_started = time.perf_counter()
                fixed_run = run_step(
                    state,
                    backend=args.backend,
                    device=args.device,
                    seed=args.seed,
                    boost=state["boost"],
                    fixed_updates=100,
                    fast_opts_N=state.get("fast_opts_N"),
                    slow_opts_N=state.get("slow_opts_N"),
                    K=state["K"],
                )
                if torch is not None:
                    _sync_cuda(torch, args.device)
                fixed_wall = time.perf_counter() - fixed_started
            assert rescore_candidates is not None
            rescored = rescore_candidates(
                complete_run, seeds=_rescore_seeds(args.seed)
            )

        native_rng = (
            _native_rng_timing(torch, state, args.device)
            if args.backend == "torch" and not args.skip_auxiliary
            else {
                "applicable": False,
                "reason": (
                    "auxiliary work disabled"
                    if args.skip_auxiliary
                    else "NumPy backend"
                ),
            }
        )
        control_after = None if args.skip_auxiliary else _numpy_control(np)
        final_memory = _memory_record(torch, args.device)
        gate_failed = kernel is not None and not kernel.get("all_pass", False)
        result = {
            "schema": 1,
            "status": "gate_failed" if gate_failed else "ok",
            "case": args.case,
            "seed": args.seed,
            "backend": args.backend,
            "device": args.device,
            "mode": mode,
            "complete": requested_complete,
            "requested_complete": requested_complete,
            "contract": state["contract"],
            "environment": environment,
            "timing": {
                "worker_wall_s": time.perf_counter() - worker_started,
                "import_s": import_wall,
                "state_rebuild_s": load_wall,
                "cuda_context_s": context_record["wall_s"],
                "complete_fit_host_wall_s": fit_wall,
                "instrumented_complete_fit_s": (
                    None
                    if complete_run is None
                    else complete_run.get("timing", {}).get("total")
                ),
                "fixed_100_fit_host_wall_s": fixed_wall,
                "warmup": warmup,
                "control_before": control_before,
                "control_after": control_after,
            },
            "cuda_context": context_record,
            "memory": final_memory,
            "post_fit_memory_before_diagnostics": post_fit_memory,
            "kernel": kernel,
            "native_rng": native_rng,
            "complete_run": complete_run,
            "bare_numpy_production_control": production_control,
            "fixed_100_run": fixed_run,
            "fixed_100_divergence": _replay_divergence(
                complete_run, fixed_run
            ),
            "fine_rescores": rescored,
            "transformed_moments": moments,
        }
        _save_artifact(run_dir, result)
        return 2 if gate_failed else 0
    except BaseException as exc:
        elapsed = time.perf_counter() - worker_started
        memory = _peak_rss()
        failure = {
            "schema": 1,
            "status": "failed",
            "case": args.case,
            "seed": args.seed,
            "backend": args.backend,
            "device": args.device,
            "mode": (
                "kernel_only"
                if args.kernel_only
                else "clean_control"
                if args.trace_disabled
                else "traced"
            ),
            "complete": requested_complete,
            "requested_complete": requested_complete,
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "worker_wall_s": elapsed,
            "failure_elapsed_s": elapsed,
            "process_peak_rss_bytes": memory["bytes"],
            "process_peak_rss": memory,
            "torch_runtime_threads": thread_record,
        }
        _atomic_json(run_dir / "failure.json", failure)
        return 1


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _existing_records(out: Path) -> list[dict[str, Any]]:
    records = []
    runs = out / "runs"
    if not runs.exists():
        return records
    for run_dir in sorted(path for path in runs.iterdir() if path.is_dir()):
        record = _read_json(run_dir / "result.json")
        if record is None:
            record = _read_json(run_dir / "failure.json")
        if record is not None:
            records.append(record)
    return records


def _summary(out: Path) -> dict[str, Any]:
    records = _existing_records(out)
    rows = []
    for record in records:
        timing = record.get("timing", {})
        complete_timing = (record.get("complete_run") or {}).get("timing", {})
        fit = complete_timing.get("total")
        kernel = record.get("kernel", {})
        kernel_pass = None if kernel is None else kernel.get("all_pass")
        status = record.get("status")
        ratio_eligible = bool(
            status == "ok"
            and record.get("complete", False)
            and isinstance(fit, (int, float))
            and kernel_pass is not False
        )
        rows.append(
            {
                "case": record.get("case"),
                "seed": record.get("seed"),
                "backend": record.get("backend"),
                "device": record.get("device"),
                "mode": record.get("mode", "traced"),
                "status": status,
                "complete": record.get("complete", False),
                "requested_complete": record.get(
                    "requested_complete", record.get("complete", False)
                ),
                "instrumented_fit_s": fit,
                "external_fit_host_wall_s": timing.get(
                    "complete_fit_host_wall_s"
                ),
                "kernel_all_pass": kernel_pass,
                "ratio_eligible": ratio_eligible,
                "peak_rss_bytes": record.get("memory", {}).get(
                    "process_peak_rss_bytes"
                ),
                "failure_elapsed_s": record.get(
                    "failure_elapsed_s", record.get("worker_wall_s")
                ),
            }
        )
    lookup = {
        (
            row["case"],
            row["seed"],
            row["backend"],
            row["device"],
            row["mode"],
        ): row
        for row in rows
    }
    ratios = []
    quarantined_ratios = []
    for row in rows:
        if row["backend"] != "torch" or not isinstance(
            row["instrumented_fit_s"], (int, float)
        ):
            continue
        baseline = lookup.get(
            (row["case"], row["seed"], "numpy", "cpu", row["mode"])
        )
        pair = {
            "case": row["case"],
            "seed": row["seed"],
            "mode": row["mode"],
            "device": row["device"],
        }
        if (
            baseline
            and isinstance(baseline["instrumented_fit_s"], (int, float))
            and baseline["instrumented_fit_s"] > 0
        ):
            pair["torch_over_numpy"] = (
                row["instrumented_fit_s"] / baseline["instrumented_fit_s"]
            )
            if row["ratio_eligible"] and baseline["ratio_eligible"]:
                ratios.append(pair)
            else:
                pair["reason"] = {
                    "torch_status": row["status"],
                    "torch_kernel_gate": row["kernel_all_pass"],
                    "numpy_status": baseline["status"],
                    "numpy_kernel_gate": baseline["kernel_all_pass"],
                }
                quarantined_ratios.append(pair)
        else:
            pair["reason"] = {"missing_eligible_numpy_cpu_baseline": True}
            quarantined_ratios.append(pair)
    return {
        "schema": 1,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "rows": rows,
        "paired_ratios": ratios,
        "quarantined_ratios": quarantined_ratios,
        "counts": {
            "records": len(rows),
            "ok": sum(row["status"] == "ok" for row in rows),
            "failed": sum(row["status"] != "ok" for row in rows),
            "complete": sum(bool(row["complete"]) for row in rows),
            "requested_complete": sum(
                bool(row["requested_complete"]) for row in rows
            ),
            "gate_failed": sum(row["status"] == "gate_failed" for row in rows),
            "failure_elapsed_s": sum(
                row["failure_elapsed_s"]
                for row in rows
                if isinstance(row["failure_elapsed_s"], (int, float))
            ),
        },
        "limits": {
            "complete_per_backend": MAX_COMPLETE_PER_BACKEND,
            "complete_overall": MAX_COMPLETE_OVERALL,
            "clean_complete_per_backend": MAX_CLEAN_COMPLETE_PER_BACKEND,
            "clean_complete_overall": MAX_CLEAN_COMPLETE_OVERALL,
        },
    }


def _write_summary(out: Path) -> None:
    summary = _summary(out)
    _atomic_json(out / "summary.json", summary)
    lines = [
        "# Stage 4 Torch variational-step measurements",
        "",
        "| case | seed | mode | backend | device | status | run_step total s | external host s | kernel gate |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in summary["rows"]:
        wall = row["instrumented_fit_s"]
        wall_text = "" if not isinstance(wall, (int, float)) else f"{wall:.6g}"
        host = row["external_fit_host_wall_s"]
        host_text = "" if not isinstance(host, (int, float)) else f"{host:.6g}"
        lines.append(
            f"| {row['case']} | {row['seed']} | {row['mode']} | "
            f"{row['backend']} | {row['device']} | {row['status']} | "
            f"{wall_text} | {host_text} | "
            f"{row['kernel_all_pass']} |"
        )
    if summary["paired_ratios"]:
        lines += [
            "",
            "## Paired complete-step ratios",
            "",
            "| case | seed | mode | Torch device | Torch / NumPy CPU |",
            "| --- | ---: | --- | --- | ---: |",
        ]
        for row in summary["paired_ratios"]:
            lines.append(
                f"| {row['case']} | {row['seed']} | {row['mode']} | "
                f"{row['device']} | "
                f"{row['torch_over_numpy']:.6g} |"
            )
    if summary["quarantined_ratios"]:
        lines += [
            "",
            "## Quarantined ratios",
            "",
            "Ratios with a failed numerical gate or failed worker are retained here and excluded from performance evidence.",
            "",
        ]
        for row in summary["quarantined_ratios"]:
            lines.append(
                f"- {row['case']} seed {row['seed']} {row['device']}: "
                f"{json.dumps(row['reason'], sort_keys=True)}"
            )
    lines += [
        "",
        "Stage timing is not an end-to-end VBMC runtime measurement. Around 1.2x is not an agreed cutoff.",
        "GPU results do not replace the independent CPU acceptance requirement.",
        "",
    ]
    (out / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def _parse_cases(text: str) -> list[str]:
    from torch_vi_fixtures import ALL_CASES, ORACLE_KERNEL_CASES, WORKLOADS

    aliases = {
        "all": list(WORKLOADS) + list(ORACLE_KERNEL_CASES),
        "all_complete": [
            name for name, spec in WORKLOADS.items() if spec.complete
        ],
        "all_kernels": list(ORACLE_KERNEL_CASES)
        + ["medium_synthetic", "boost_single", "kernel_stress"],
    }
    selected = []
    for token in (part.strip() for part in text.split(",")):
        if not token:
            continue
        selected.extend(aliases.get(token, [token]))
    selected = list(dict.fromkeys(selected))
    unknown = set(selected) - set(ALL_CASES)
    if unknown:
        raise ValueError(
            f"unknown cases {sorted(unknown)}; choose from {sorted(ALL_CASES)}"
        )
    if not selected:
        raise ValueError("--cases selected no workloads")
    return selected


def _parse_seeds(text: str) -> tuple[int, ...]:
    from torch_vi_fixtures import MEASUREMENT_SEEDS

    try:
        selected = tuple(
            dict.fromkeys(
                int(part.strip()) for part in text.split(",") if part.strip()
            )
        )
    except ValueError as exc:
        raise ValueError(
            "--seeds must be a comma-separated integer subset"
        ) from exc
    unknown = set(selected) - set(MEASUREMENT_SEEDS)
    if unknown or not selected:
        raise ValueError(
            f"--seeds must be a nonempty subset of {MEASUREMENT_SEEDS}; "
            f"got {selected}"
        )
    return selected


def _run_id(
    backend: str, device: str, case: str, seed: int, mode: str = "traced"
) -> str:
    clean_device = device.replace(":", "-")
    return f"{mode}_{backend}_{clean_device}_{case}_seed{seed}"


def _coordinator(args: argparse.Namespace) -> int:
    from torch_vi_fixtures import ALL_CASES, workload_manifest

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = _parse_cases(args.cases)
    selected_seeds = _parse_seeds(args.seeds)
    if args.kernel_only:
        mode = "kernel_only"
    elif args.trace_disabled:
        mode = "clean_control"
    else:
        mode = "traced"
    if args.kernel_only and (
        args.trace_disabled or args.skip_auxiliary or args.warm_kernels
    ):
        raise ValueError(
            "--kernel-only cannot be combined with clean-control flags"
        )
    if args.trace_disabled and not (args.skip_auxiliary and args.warm_kernels):
        raise ValueError(
            "--trace-disabled requires --skip-auxiliary and --warm-kernels"
        )
    if (args.skip_auxiliary or args.warm_kernels) and not args.trace_disabled:
        raise ValueError(
            "--skip-auxiliary/--warm-kernels require --trace-disabled"
        )
    control_cases = tuple(
        token.strip()
        for token in args.production_controls.split(",")
        if token.strip()
    )
    unknown_controls = set(control_cases) - set(ALL_CASES)
    if unknown_controls:
        raise ValueError(
            f"unknown --production-controls cases {sorted(unknown_controls)}"
        )
    if args.backend == "numpy" and args.device != "cpu":
        raise ValueError("the NumPy backend only supports --device cpu")
    complete_requested = sum(
        len(selected_seeds)
        for case in cases
        if ALL_CASES[case].complete and not args.kernel_only
    )
    existing = _existing_records(out)
    existing_backend = [
        row
        for row in existing
        if row.get("backend") == args.backend
        and row.get("device") == args.device
    ]
    existing_backend_complete = sum(
        bool(row.get("requested_complete", row.get("complete")))
        and row.get("mode", "traced") == mode
        for row in existing_backend
    )
    per_backend_cap = (
        MAX_CLEAN_COMPLETE_PER_BACKEND
        if mode == "clean_control"
        else MAX_COMPLETE_PER_BACKEND
    )
    overall_cap = (
        MAX_CLEAN_COMPLETE_OVERALL
        if mode == "clean_control"
        else MAX_COMPLETE_OVERALL
    )
    if existing_backend_complete + complete_requested > per_backend_cap:
        raise ValueError(
            f"campaign arm already has {existing_backend_complete} complete fit "
            f"attempts; {complete_requested} more exceeds the "
            f"{per_backend_cap} per-backend cap for {mode}"
        )
    existing_complete = sum(
        bool(row.get("requested_complete", row.get("complete")))
        and row.get("mode", "traced") == mode
        for row in existing
    )
    if existing_complete + complete_requested > overall_cap:
        raise ValueError(
            f"campaign would exceed {overall_cap} complete fits overall for {mode}"
        )
    invocation_id = (
        time.strftime("%Y%m%dT%H%M%S")
        + f"_{time.time_ns() % 1_000_000_000:09d}_{os.getpid()}"
    )
    manifest = workload_manifest(cases)
    if not manifest["environment"]["provenance_validation"]["valid"]:
        raise RuntimeError(
            "invalid experiment provenance: "
            f"{manifest['environment']['provenance_validation']['errors']}"
        )
    manifest["request"] = {
        "invocation_id": invocation_id,
        "mode": mode,
        "cases": cases,
        "seeds": list(selected_seeds),
        "backend": args.backend,
        "device": args.device,
        "fit_timeout_s": args.fit_timeout,
        "backend_cap_s": args.backend_cap,
        "production_controls": list(control_cases),
        "kernel_only": args.kernel_only,
        "trace_disabled": args.trace_disabled,
        "skip_auxiliary": args.skip_auxiliary,
        "warm_kernels": args.warm_kernels,
        "prior_backend_wall_s": args.prior_backend_wall,
        "requested_complete": complete_requested,
    }
    _atomic_json(
        out
        / (
            f"manifest_{mode}_{args.backend}_"
            f"{args.device.replace(':', '-')}_{invocation_id}.json"
        ),
        manifest,
    )
    _append_ledger(
        out / "invocations.jsonl",
        {
            "schema": 1,
            "event": "start",
            "invocation_id": invocation_id,
            "mode": mode,
            "backend": args.backend,
            "device": args.device,
            "cases": cases,
            "seeds": list(selected_seeds),
            "requested_complete": complete_requested,
            "wall_time_ns": time.time_ns(),
        },
    )

    prior_backend_wall = float(args.prior_backend_wall)
    for row in existing_backend:
        candidate = row.get("timing", {}).get("process_spawn_to_exit_s")
        if not isinstance(candidate, (int, float)):
            candidate = row.get("spawn_wall_s")
        if isinstance(candidate, (int, float)):
            prior_backend_wall += candidate
    if prior_backend_wall >= args.backend_cap:
        raise ValueError(
            f"persisted {args.backend}/{args.device} worker time "
            f"{prior_backend_wall:.3f}s already reaches the backend cap"
        )
    coordinator_started = time.perf_counter()
    stopped_for_cap = False
    failures = 0
    failure_elapsed_s = 0.0
    scheduled = 0
    for case in cases:
        seeds = selected_seeds
        for seed in seeds:
            if (
                prior_backend_wall + time.perf_counter() - coordinator_started
                >= args.backend_cap
            ):
                stopped_for_cap = True
                break
            run_id = _run_id(args.backend, args.device, case, seed, mode)
            run_dir = out / "runs" / run_id
            if run_dir.exists():
                raise FileExistsError(
                    f"{run_dir} already exists; no automatic retry or overwrite"
                )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--run-dir",
                str(run_dir),
                "--case",
                case,
                "--seed",
                str(seed),
                "--backend",
                args.backend,
                "--device",
                args.device,
            ]
            if args.kernel_only:
                command.append("--kernel-only")
            if args.trace_disabled:
                command.append("--trace-disabled")
            if args.skip_auxiliary:
                command.append("--skip-auxiliary")
            if args.warm_kernels:
                command.append("--warm-kernels")
            if (
                mode == "traced"
                and not args.kernel_only
                and args.backend == "numpy"
                and args.device == "cpu"
                and case in control_cases
                and seed == 1701
                and ALL_CASES[case].complete
            ):
                command.append("--production-control")
            environment = os.environ.copy()
            environment.update(THREAD_ENV)
            log_path = out / "runs" / f"{run_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            spawn_started = time.perf_counter()
            remaining_backend_s = (
                args.backend_cap
                - prior_backend_wall
                - (spawn_started - coordinator_started)
            )
            worker_timeout_s = min(args.fit_timeout, remaining_backend_s)
            if worker_timeout_s <= 0:
                stopped_for_cap = True
                break
            scheduled += 1
            completed = None
            timed_out = False
            try:
                with log_path.open("w", encoding="utf-8", newline="\n") as log:
                    completed = subprocess.run(
                        command,
                        cwd=Path(__file__).resolve().parents[2],
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=worker_timeout_s,
                        check=False,
                    )
                if completed.returncode != 0:
                    failures += 1
            except subprocess.TimeoutExpired:
                timed_out = True
                failures += 1
                run_dir.mkdir(parents=True, exist_ok=True)
                _atomic_json(
                    run_dir / "failure.json",
                    {
                        "schema": 1,
                        "status": "timeout",
                        "case": case,
                        "seed": seed,
                        "backend": args.backend,
                        "device": args.device,
                        "mode": mode,
                        "complete": bool(
                            ALL_CASES[case].complete and not args.kernel_only
                        ),
                        "requested_complete": bool(
                            ALL_CASES[case].complete and not args.kernel_only
                        ),
                        "timeout_s": worker_timeout_s,
                        "spawn_wall_s": time.perf_counter() - spawn_started,
                        "failure_elapsed_s": time.perf_counter()
                        - spawn_started,
                        "retry": False,
                    },
                )
                failure_elapsed_s += time.perf_counter() - spawn_started
            record_path = run_dir / "result.json"
            record = _read_json(record_path)
            if record is not None:
                worker_elapsed_s = time.perf_counter() - spawn_started
                if timed_out:
                    record["status"] = "timeout_after_complete"
                    record["phase"] = "auxiliary_work_incomplete"
                    record["failure_elapsed_s"] = worker_elapsed_s
                elif completed is not None and completed.returncode != 0:
                    if record.get("status") not in {"gate_failed"}:
                        record["status"] = "failed_after_complete"
                    record["failure_elapsed_s"] = worker_elapsed_s
                record.setdefault("timing", {})[
                    "process_spawn_to_exit_s"
                ] = worker_elapsed_s
                _atomic_json(record_path, record)
            if completed is not None and completed.returncode != 0:
                failure_elapsed_s += time.perf_counter() - spawn_started
            _write_summary(out)
        if stopped_for_cap:
            break
    _atomic_json(
        out
        / (
            f"campaign_{mode}_{args.backend}_"
            f"{args.device.replace(':', '-')}_{invocation_id}.json"
        ),
        {
            "schema": 1,
            "invocation_id": invocation_id,
            "mode": mode,
            "backend": args.backend,
            "device": args.device,
            "cases": cases,
            "seeds": list(selected_seeds),
            "wall_s": time.perf_counter() - coordinator_started,
            "prior_backend_wall_s": prior_backend_wall,
            "cumulative_backend_wall_s": (
                prior_backend_wall + time.perf_counter() - coordinator_started
            ),
            "backend_cap_s": args.backend_cap,
            "stopped_for_cap": stopped_for_cap,
            "failures": failures,
            "failure_elapsed_s": failure_elapsed_s,
            "requested_complete": complete_requested,
            "scheduled_workers": scheduled,
            "automatic_retries": 0,
            "production_controls": list(control_cases),
        },
    )
    _append_ledger(
        out / "invocations.jsonl",
        {
            "schema": 1,
            "event": "end",
            "invocation_id": invocation_id,
            "mode": mode,
            "backend": args.backend,
            "device": args.device,
            "requested_complete": complete_requested,
            "scheduled_workers": scheduled,
            "failures": failures,
            "failure_elapsed_s": failure_elapsed_s,
            "stopped_for_cap": stopped_for_cap,
            "wall_s": time.perf_counter() - coordinator_started,
            "wall_time_ns": time.time_ns(),
        },
    )
    _write_summary(out)
    return 1 if failures else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="mandatory campaign output directory")
    parser.add_argument(
        "--cases", help="comma-separated cases or all/all_complete/all_kernels"
    )
    parser.add_argument("--backend", required=True, choices=("numpy", "torch"))
    parser.add_argument("--device", required=True, help="cpu, cuda, or cuda:N")
    parser.add_argument(
        "--fit-timeout", type=float, default=DEFAULT_FIT_TIMEOUT_S
    )
    parser.add_argument(
        "--backend-cap", type=float, default=DEFAULT_BACKEND_CAP_S
    )
    parser.add_argument(
        "--prior-backend-wall",
        type=float,
        default=0.0,
        help="seconds already spent in the arm and charged to its 45-minute cap",
    )
    parser.add_argument(
        "--seeds",
        default="1701,1702,1703",
        help="comma-separated subset of the fixed seeds 1701,1702,1703",
    )
    parser.add_argument(
        "--production-controls",
        default=",".join(DEFAULT_PRODUCTION_CONTROL_CASES),
        help=(
            "NumPy-only representative cases whose seed-1701 worker also "
            "runs the bare production orchestration; empty disables"
        ),
    )
    parser.add_argument(
        "--kernel-only",
        action="store_true",
        help="run corrected kernel diagnostics without a complete fit",
    )
    parser.add_argument(
        "--trace-disabled",
        action="store_true",
        help="use the thin trace-disabled step adapter for clean controls",
    )
    parser.add_argument(
        "--skip-auxiliary",
        action="store_true",
        help="skip diagnostics, fixed replay, rescoring, and control probes",
    )
    parser.add_argument(
        "--warm-kernels",
        action="store_true",
        help="warm exactly three common-input objectives before the fit",
    )
    parser.add_argument(
        "--worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--run-dir", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--production-control", action="store_true", help=argparse.SUPPRESS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker:
        missing = [
            name
            for name in ("run_dir", "case", "seed")
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit(f"worker missing arguments: {missing}")
        return _worker(args)
    missing = [
        name for name in ("out", "cases") if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(f"coordinator missing required arguments: {missing}")
    if args.fit_timeout <= 0 or args.fit_timeout > DEFAULT_FIT_TIMEOUT_S:
        raise SystemExit(
            f"--fit-timeout must be in (0, {DEFAULT_FIT_TIMEOUT_S}]"
        )
    if args.backend_cap <= 0 or args.backend_cap > DEFAULT_BACKEND_CAP_S:
        raise SystemExit(
            f"--backend-cap must be in (0, {DEFAULT_BACKEND_CAP_S}]"
        )
    if (
        args.prior_backend_wall < 0
        or args.prior_backend_wall >= args.backend_cap
    ):
        raise SystemExit(
            "--prior-backend-wall must be nonnegative and below --backend-cap"
        )
    return _coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
