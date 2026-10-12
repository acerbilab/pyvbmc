"""Reconcile bounded Stage 4 Torch feasibility campaign artifacts.

This reader is deliberately independent of the numerical prototype.  It uses
only the Python standard library, resolves the small set of required NPZ
leaves lazily, and writes compact evidence for incorporation into the Stage 4
plan.  It does not make a feasibility recommendation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import statistics
import struct
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ARM_ORDER = ("numpy_cpu", "torch_cpu", "torch_cuda")
CASE_ORDER = (
    "warmup",
    "deterministic",
    "bounded",
    "warped",
    "noisy",
    "boost_sampled",
    "medium_synthetic",
    "boost_single",
    "kernel_stress",
    "kernel_normal_D2_warmup",
    "kernel_normal_D2_K1",
    "kernel_normal_D2_singlesample",
    "kernel_halfnormal_D2_bounded",
    "kernel_corr_D5_warped",
    "kernel_rosenbrock_D2_noise1_viqr",
    "kernel_cigar_D4_largeK",
    "kernel_cigar_D4_boosted",
)
TIMING_FIELDS = (
    "primary_complete_total_s",
    "external_complete_host_wall_s",
    "source_validation_report_construction_s",
    "complete_setup_s",
    "complete_gp_extract_transfer_s",
    "complete_optimizer_s",
    "complete_fine_scoring_pruning_s",
    "complete_candidate_sieve_s",
    "complete_output_transfer_s",
    "complete_unattributed_orchestration_s",
    "worker_wall_s",
    "process_spawn_to_exit_s",
    "import_s",
    "state_rebuild_s",
    "cuda_context_s",
    "fixed_100_host_wall_s",
    "warmup_calls",
    "warmup_total_s",
    "warmup_input_setup_s",
    "warmup_fixed_state_transfer_s",
    "clean_pre_sync_outside_total_s",
    "clean_source_guard_outside_total_s",
    "process_peak_rss_bytes",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
)
MISSING = object()
VP_RTOL = 1e-10
VP_ATOL = 1e-12
SOLVE_RTOL = 1e-6
SOLVE_ATOL = 1e-10
VAR_RTOL = 1e-3
VAR_ATOL = 1e-8


@dataclass(frozen=True)
class ArrayData:
    """A decoded numeric NPY payload, flattened in stored C order."""

    values: tuple[float | int | bool, ...]
    shape: tuple[int, ...]
    dtype: str
    raw_sha256: str


def _strict(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _npy_payload(raw: bytes) -> tuple[dict[str, Any], bytes]:
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("archive member is not an NPY payload")
    major = raw[6]
    if major == 1:
        header_length = struct.unpack_from("<H", raw, 8)[0]
        offset = 10
    elif major in (2, 3):
        header_length = struct.unpack_from("<I", raw, 8)[0]
        offset = 12
    else:
        raise ValueError(f"unsupported NPY version {major}.{raw[7]}")
    header = ast.literal_eval(
        raw[offset : offset + header_length].decode("latin1").strip()
    )
    if not isinstance(header, dict):
        raise ValueError("invalid NPY header")
    return header, raw[offset + header_length :]


def _decode_npy(raw: bytes) -> ArrayData:
    header, payload = _npy_payload(raw)
    shape = tuple(int(item) for item in header.get("shape", ()))
    if header.get("fortran_order"):
        raise ValueError("Fortran-order NPY payloads are unsupported")
    descr = header.get("descr")
    formats = {
        "<f8": "<d",
        ">f8": ">d",
        "<f4": "<f",
        ">f4": ">f",
        "<i8": "<q",
        ">i8": ">q",
        "<i4": "<i",
        ">i4": ">i",
        "<u8": "<Q",
        ">u8": ">Q",
        "<u4": "<I",
        ">u4": ">I",
        "|b1": "<?",
        "|u1": "<B",
        "|i1": "<b",
    }
    fmt = formats.get(descr)
    if fmt is None:
        raise ValueError(f"unsupported NPY dtype {descr!r}")
    item_size = struct.calcsize(fmt)
    expected_count = math.prod(shape) if shape else 1
    if len(payload) != expected_count * item_size:
        raise ValueError(
            f"NPY payload length {len(payload)} does not match shape {shape}"
        )
    values = tuple(item[0] for item in struct.iter_unpack(fmt, payload))
    return ArrayData(
        values, shape, str(descr), hashlib.sha256(payload).hexdigest()
    )


def _decode_reference(
    reference: Mapping[str, Any], run_dir: Path
) -> ArrayData:
    archive = reference.get("archive")
    key = reference.get("npz")
    if not isinstance(archive, str) or not isinstance(key, str):
        raise ValueError("array reference lacks archive or npz key")
    archive_path = run_dir / archive
    with zipfile.ZipFile(archive_path) as bundle:
        raw = bundle.read(f"{key}.npy")
    decoded = _decode_npy(raw)
    expected_shape = tuple(reference.get("shape", ()))
    if expected_shape and decoded.shape != expected_shape:
        raise ValueError(
            f"decoded shape {decoded.shape} differs from {expected_shape}"
        )
    expected_hash = reference.get("sha256")
    if isinstance(expected_hash, str) and decoded.raw_sha256 != expected_hash:
        raise ValueError("decoded payload SHA-256 differs from JSON reference")
    return decoded


def _decode_selected(value: Any, run_dir: Path) -> Any:
    """Recursively decode only the selected artifact subtree passed here."""
    if isinstance(value, dict):
        if isinstance(value.get("archive"), str) and isinstance(
            value.get("npz"), str
        ):
            return _decode_reference(value, run_dir)
        return {
            key: _decode_selected(item, run_dir) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_decode_selected(item, run_dir) for item in value]
    return value


def _arm(record: Mapping[str, Any]) -> str:
    backend = str(record.get("backend") or "unknown")
    device = str(record.get("device") or "unknown").replace(":", "-")
    return f"{backend}_{device}"


def _ordered(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    found = set(values)
    return [value for value in preferred if value in found] + sorted(
        found.difference(preferred)
    )


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def _median_range(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [float(value) for value in values if _number(value) is not None]
    if not numbers:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {
        "n": len(numbers),
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def _mean_spread(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [float(value) for value in values if _number(value) is not None]
    if not numbers:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "max": max(numbers),
    }


def _difference(left: ArrayData, right: ArrayData) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "comparable": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "max_abs": None,
            "rms": None,
        }
    deltas = [float(a) - float(b) for a, b in zip(left.values, right.values)]
    return {
        "comparable": True,
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "max_abs": max((abs(value) for value in deltas), default=0.0),
        "rms": (
            math.sqrt(statistics.fmean(value * value for value in deltas))
            if deltas
            else 0.0
        ),
    }


def _tolerance_comparison(
    actual: Any,
    expected: Any,
    actual_dir: Path,
    expected_dir: Path,
    *,
    rtol: float,
    atol: float,
    canonicalize_singleton_vector: bool = False,
    center_logweights: bool = False,
) -> dict[str, Any]:
    """Compare one retained clean-control value to its traced counterpart."""
    if actual is MISSING or expected is MISSING:
        return {
            "status": "unknown",
            "reason": "field missing from one or both artifacts",
            "rtol": rtol,
            "atol": atol,
        }
    if actual is None or expected is None:
        matched = actual is None and expected is None
        return {
            "status": "matched" if matched else "mismatched",
            "actual": actual,
            "expected": expected,
            "rtol": rtol,
            "atol": atol,
        }
    try:
        actual_value = _decode_selected(actual, actual_dir)
        expected_value = _decode_selected(expected, expected_dir)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        return {
            "status": "unknown",
            "reason": f"{type(exc).__name__}: {exc}",
            "rtol": rtol,
            "atol": atol,
        }
    if isinstance(actual_value, ArrayData) or isinstance(
        expected_value, ArrayData
    ):
        if not isinstance(actual_value, ArrayData) or not isinstance(
            expected_value, ArrayData
        ):
            return {
                "status": "mismatched",
                "reason": "array/scalar representation differs",
                "rtol": rtol,
                "atol": atol,
            }
        shape_canonicalized = False
        if actual_value.shape != expected_value.shape:
            shape_canonicalized = (
                canonicalize_singleton_vector
                and len(actual_value.values) == len(expected_value.values)
                and sum(size != 1 for size in actual_value.shape) <= 1
                and sum(size != 1 for size in expected_value.shape) <= 1
            )
        if (
            actual_value.shape != expected_value.shape
            and not shape_canonicalized
        ):
            return {
                "status": "mismatched",
                "actual_shape": list(actual_value.shape),
                "expected_shape": list(expected_value.shape),
                "rtol": rtol,
                "atol": atol,
            }
        actual_values = actual_value.values
        expected_values = expected_value.values
        raw_max_abs = max(
            (
                abs(float(a) - float(b))
                for a, b in zip(actual_values, expected_values)
            ),
            default=0.0,
        )
        shifts = None
        if center_logweights:
            shifts = (max(actual_values), max(expected_values))
            actual_values = tuple(
                float(value) - shifts[0] for value in actual_values
            )
            expected_values = tuple(
                float(value) - shifts[1] for value in expected_values
            )
        differences = [
            abs(float(one) - float(two))
            for one, two in zip(actual_values, expected_values)
        ]
        scales = [atol + rtol * abs(float(value)) for value in expected_values]
        passed = all(diff <= scale for diff, scale in zip(differences, scales))
        return {
            "status": "matched" if passed else "mismatched",
            "actual_shape": list(actual_value.shape),
            "expected_shape": list(expected_value.shape),
            "actual_dtype": actual_value.dtype,
            "expected_dtype": expected_value.dtype,
            "shape_canonicalized": shape_canonicalized,
            "logweight_gauge_centered": center_logweights,
            "raw_max_abs_before_gauge": (
                raw_max_abs if center_logweights else None
            ),
            "subtracted_logweight_maxima": shifts,
            "max_abs": max(differences, default=0.0),
            "max_scaled": (
                max(
                    (diff / scale if scale else math.inf)
                    for diff, scale in zip(differences, scales)
                )
                if differences
                else 0.0
            ),
            "rtol": rtol,
            "atol": atol,
        }
    if isinstance(actual_value, bool) or isinstance(expected_value, bool):
        matched = actual_value == expected_value
        return {
            "status": "matched" if matched else "mismatched",
            "actual": actual_value,
            "expected": expected_value,
            "rtol": 0.0,
            "atol": 0.0,
        }
    if (
        _number(actual_value) is not None
        and _number(expected_value) is not None
    ):
        difference = abs(float(actual_value) - float(expected_value))
        scale = atol + rtol * abs(float(expected_value))
        return {
            "status": "matched" if difference <= scale else "mismatched",
            "actual": actual_value,
            "expected": expected_value,
            "max_abs": difference,
            "max_scaled": difference / scale if scale else math.inf,
            "rtol": rtol,
            "atol": atol,
        }
    matched = actual_value == expected_value
    return {
        "status": "matched" if matched else "mismatched",
        "actual": actual_value,
        "expected": expected_value,
        "rtol": 0.0,
        "atol": 0.0,
    }


def _mapping_value(value: Any, key: str) -> Any:
    return value.get(key, MISSING) if isinstance(value, Mapping) else MISSING


def _final_equivalence(
    reference: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Reconcile retained final outputs from two same-input complete fits."""
    traced_run = reference.get("complete_run") or {}
    clean_run = actual.get("complete_run") or {}
    traced_result = traced_run.get("result") or {}
    clean_result = clean_run.get("result") or {}
    traced_vp = traced_run.get("vp") or {}
    clean_vp = clean_run.get("vp") or {}
    arrays = {
        name: _tolerance_comparison(
            _mapping_value(clean_vp, name),
            _mapping_value(traced_vp, name),
            actual["run_dir"],
            reference["run_dir"],
            rtol=VP_RTOL,
            atol=VP_ATOL,
            canonicalize_singleton_vector=name == "eta",
            center_logweights=name == "eta",
        )
        for name in ("mu", "sigma", "lambd", "w", "eta")
    }
    traced_stats = traced_vp.get("stats") or traced_result.get("stats") or {}
    clean_stats = clean_vp.get("stats") or clean_result.get("stats") or {}
    statistic_tolerances = {
        "e_log_joint": (SOLVE_RTOL, SOLVE_ATOL),
        "entropy": (VP_RTOL, VP_ATOL),
        "elbo": (SOLVE_RTOL, SOLVE_ATOL),
        "e_log_joint_sd": (VAR_RTOL, VAR_ATOL),
        "entropy_sd": (VAR_RTOL, VAR_ATOL),
        "elbo_sd": (VAR_RTOL, VAR_ATOL),
        "I_sk": (SOLVE_RTOL, SOLVE_ATOL),
        "J_sjk": (VAR_RTOL, VAR_ATOL),
        "stable": (0.0, 0.0),
    }
    statistics_comparison = {
        name: _tolerance_comparison(
            _mapping_value(clean_stats, name),
            _mapping_value(traced_stats, name),
            actual["run_dir"],
            reference["run_dir"],
            rtol=tolerance[0],
            atol=tolerance[1],
        )
        for name, tolerance in statistic_tolerances.items()
    }
    statistics_comparison["var_ss"] = _tolerance_comparison(
        clean_run.get("var_ss", MISSING),
        traced_run.get("var_ss", MISSING),
        actual["run_dir"],
        reference["run_dir"],
        rtol=VAR_RTOL,
        atol=VAR_ATOL,
    )
    structural_fields = {
        "K": (
            _mapping_value(clean_result, "K"),
            _mapping_value(traced_result, "K"),
        ),
        "pruned": (
            clean_result.get("pruned", clean_run.get("pruned", MISSING)),
            traced_result.get("pruned", traced_run.get("pruned", MISSING)),
        ),
        "budgets": (
            _mapping_value(clean_result, "budgets"),
            _mapping_value(traced_result, "budgets"),
        ),
        "iterations_by_start": (
            _mapping_value(clean_result, "iterations_by_start"),
            _mapping_value(traced_result, "iterations_by_start"),
        ),
        "stopping_reasons": (
            _mapping_value(clean_result, "stopping_reasons"),
            _mapping_value(traced_result, "stopping_reasons"),
        ),
        "boost_accepted": (
            (
                _mapping_value(clean_result.get("boost") or {}, "changed")
                if clean_result.get("boost") is not None
                else None
            ),
            (
                _mapping_value(traced_result.get("boost") or {}, "changed")
                if traced_result.get("boost") is not None
                else None
            ),
        ),
    }
    structure = {
        name: _tolerance_comparison(
            actual_value,
            expected_value,
            actual["run_dir"],
            reference["run_dir"],
            rtol=0.0,
            atol=0.0,
        )
        for name, (actual_value, expected_value) in structural_fields.items()
    }
    retained = [
        *arrays.values(),
        *statistics_comparison.values(),
        *structure.values(),
    ]
    statuses = {item["status"] for item in retained}
    overall = (
        "mismatched"
        if "mismatched" in statuses
        else "unknown"
        if "unknown" in statuses
        else "matched"
    )
    return {
        "case": actual.get("case"),
        "seed": actual.get("seed"),
        "reference_arm": reference.get("arm"),
        "actual_arm": actual.get("arm"),
        "reference_run_id": reference.get("run_id"),
        "actual_run_id": actual.get("run_id"),
        "context": context,
        "status": overall,
        "posterior_parameters": arrays,
        "statistics": statistics_comparison,
        "structure": structure,
        "trace_only_fields": {
            "status": "not_compared",
            "fields": [
                "trace",
                "candidates",
                "optimizer_runs",
                "dispatch_counts",
                "fine_rescores",
                "fixed_100_run",
            ],
            "reason": (
                "clean controls intentionally omit tracing and auxiliary diagnostics"
                if context == "clean_control_vs_instrumented"
                else "trace diagnostics are reconciled in the fixed-100 and rescore sections"
            ),
        },
        "tolerance_policy": {
            "posterior_parameters": {"rtol": VP_RTOL, "atol": VP_ATOL},
            "solve_derived": {"rtol": SOLVE_RTOL, "atol": SOLVE_ATOL},
            "variance_derived": {"rtol": VAR_RTOL, "atol": VAR_ATOL},
            "structure": {"rtol": 0.0, "atol": 0.0},
        },
    }


def _find_candidate(
    run: Mapping[str, Any], candidate_id: str
) -> Mapping[str, Any] | None:
    for candidate in run.get("candidates") or []:
        if (
            isinstance(candidate, dict)
            and candidate.get("candidate_id") == candidate_id
        ):
            return candidate
    return None


def _resolve_alias(rescores: Mapping[str, Any], candidate_id: str) -> str:
    aliases = rescores.get("aliases") or {}
    seen: set[str] = set()
    while candidate_id in aliases and candidate_id not in seen:
        seen.add(candidate_id)
        candidate_id = aliases[candidate_id]
    return candidate_id


def _timing_row(record: Mapping[str, Any]) -> dict[str, Any]:
    outer = record.get("timing") or {}
    complete = record.get("complete_run") or {}
    inner = complete.get("timing") or {}
    primary = _number(inner.get("total"))
    host = _number(outer.get("complete_fit_host_wall_s"))
    residual = (
        max(0.0, float(host) - float(primary))
        if host is not None and primary is not None
        else None
    )
    warmup = outer.get("warmup") or {}
    warmup_walls = warmup.get("walls_s") or []
    memory = (
        record.get("post_fit_memory_before_diagnostics")
        or record.get("memory")
        or {}
    )
    cuda_memory = memory.get("cuda") or {}
    row = {
        "run_id": record.get("run_id"),
        "case": record.get("case"),
        "seed": record.get("seed"),
        "arm": record.get("arm"),
        "measurement_kind": record.get("measurement_kind", "instrumented"),
        "status": record.get("status"),
        "primary_saved": primary is not None,
        "primary_complete_total_s": primary,
        "external_complete_host_wall_s": host,
        "source_validation_report_construction_s": residual,
        "complete_setup_s": _number(inner.get("setup")),
        "complete_gp_extract_transfer_s": _number(
            inner.get("gp_extract_transfer")
        ),
        "complete_optimizer_s": _number(inner.get("optimizer")),
        "complete_fine_scoring_pruning_s": _number(
            inner.get("fine_scoring_pruning")
        ),
        "complete_candidate_sieve_s": _number(inner.get("candidate_sieve")),
        "complete_output_transfer_s": _number(inner.get("output_transfer")),
        "complete_unattributed_orchestration_s": _number(
            inner.get("unattributed_orchestration")
        ),
        "worker_wall_s": _number(
            outer.get("worker_wall_s", record.get("worker_wall_s"))
        ),
        "process_spawn_to_exit_s": _number(
            outer.get("process_spawn_to_exit_s", record.get("spawn_wall_s"))
        ),
        "import_s": _number(outer.get("import_s")),
        "state_rebuild_s": _number(outer.get("state_rebuild_s")),
        "cuda_context_s": _number(outer.get("cuda_context_s")),
        "fixed_100_host_wall_s": _number(
            outer.get("fixed_100_fit_host_wall_s")
        ),
        "warmup_calls": _number(warmup.get("calls")),
        "warmup_total_s": (
            sum(
                float(value)
                for value in warmup_walls
                if _number(value) is not None
            )
            if warmup_walls
            else None
        ),
        "warmup_input_setup_s": _number(warmup.get("input_setup_s")),
        "warmup_fixed_state_transfer_s": _number(
            warmup.get("fixed_state_transfer_s")
        ),
        "clean_pre_sync_outside_total_s": _number(
            inner.get("pre_sync_outside_total")
        ),
        "clean_source_guard_outside_total_s": _number(
            inner.get("source_guard_outside_total")
        ),
        "process_peak_rss_bytes": _number(
            memory.get("process_peak_rss_bytes")
        ),
        "cuda_peak_allocated_bytes": _number(cuda_memory.get("max_allocated")),
        "cuda_peak_reserved_bytes": _number(cuda_memory.get("max_reserved")),
        "paired_torch_over_numpy": None,
        "clean_over_instrumented": None,
    }
    control = record.get("bare_numpy_production_control") or {}
    control_timing = control.get("timing") or {}
    bare_fit = _number(control_timing.get("fit"))
    bare_host = _number(control_timing.get("host_wall_s"))
    row.update(
        {
            "bare_numpy_fit_s": bare_fit,
            "bare_numpy_host_wall_s": bare_host,
            "instrumented_total_over_bare_fit": (
                float(primary) / float(bare_fit)
                if primary is not None and bare_fit not in (None, 0)
                else None
            ),
            "instrumented_host_over_bare_host": (
                float(host) / float(bare_host)
                if host is not None and bare_host not in (None, 0)
                else None
            ),
        }
    )
    kernel_pass = (record.get("kernel") or {}).get("all_pass")
    row["kernel_gate_status"] = (
        "pass"
        if kernel_pass is True
        else "fail"
        if kernel_pass is False
        else "not_recorded"
    )
    row["quarantined_original_kernel_gate"] = bool(
        record.get("measurement_kind") == "instrumented"
        and kernel_pass is False
    )
    return row


def _compact_run(
    record: Mapping[str, Any], run_dir: Path, campaign: Path
) -> dict[str, Any]:
    complete = record.get("complete_run") or {}
    result = complete.get("result") or {}
    final = _find_candidate(complete, "final:0")
    stats = (
        result.get("stats") or (complete.get("vp") or {}).get("stats") or {}
    )
    boost = result.get("boost") or {}
    environment = record.get("environment") or {}
    contract = record.get("contract") or {}
    shape = contract.get("shape") or {}
    fixture_meta = contract.get("fixture_meta") or {}
    failure_path = run_dir / "failure.json"
    raw_paths: set[str] = set()

    def collect_archives(value: Any) -> None:
        if isinstance(value, dict):
            archive = value.get("archive")
            if isinstance(archive, str):
                raw_paths.add(archive)
            for item in value.values():
                collect_archives(item)
        elif isinstance(value, list):
            for item in value:
                collect_archives(item)

    collect_archives(record)
    artifacts = []
    for path in sorted(
        {run_dir / "result.json", failure_path}
        | {run_dir / archive for archive in raw_paths}
    ):
        if path.is_file():
            artifacts.append(
                {
                    "path": path.relative_to(campaign).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    kernel = record.get("kernel") or {}
    failed_kernel_errors = {
        name: error
        for name, error in (kernel.get("errors") or {}).items()
        if not error.get("pass")
    }
    failure = _read_json(failure_path)
    raw_bytes = sum(
        item["bytes"]
        for item in artifacts
        if str(item["path"]).endswith(".npz")
    )
    return {
        "run_id": run_dir.name,
        "case": record.get("case"),
        "seed": record.get("seed"),
        "backend": record.get("backend"),
        "device": record.get("device"),
        "arm": _arm(record),
        "measurement_kind": record.get("measurement_kind", "instrumented"),
        "status": record.get("status"),
        "complete_workload": record.get("complete"),
        "primary_saved": bool(complete),
        "auxiliary_complete": record.get("status") == "ok" and failure is None,
        "outcome": {
            "iterations_by_start": result.get("iterations_by_start"),
            "stopping_reasons": result.get("stopping_reasons"),
            "selected_candidate_id": result.get("selected_candidate_id"),
            "K": result.get("K"),
            "pruned": result.get("pruned"),
            "boost_accepted": boost.get("changed") if boost else None,
            "elbo": _number(stats.get("elbo")),
            "elbo_sd": _number(stats.get("elbo_sd")),
            "valid_weights": final.get("valid_weights") if final else None,
            "valid_scales": final.get("valid_scales") if final else None,
            "transformer_identity": result.get("transformer_identity"),
        },
        "kernel": {
            "all_pass": kernel.get("all_pass"),
            "failed_errors": failed_kernel_errors,
            "epsilon": kernel.get("epsilon"),
            "conversion": kernel.get("conversion"),
        },
        "memory": record.get("memory"),
        "post_fit_memory_before_diagnostics": record.get(
            "post_fit_memory_before_diagnostics"
        ),
        "warmup": (record.get("timing") or {}).get("warmup"),
        "cuda_context": record.get("cuda_context"),
        "environment": {
            "git": environment.get("git"),
            "python": environment.get("python"),
            "platform": environment.get("platform"),
            "threads": environment.get("threads"),
            "packages": environment.get("packages"),
            "module_origins": environment.get("module_origins"),
            "plan_hash": environment.get("plan_hash"),
            "source_hashes": environment.get("source_hashes"),
        },
        "fixture_contract": {
            "spec": contract.get("spec"),
            "shape": {
                key: shape.get(key) for key in ("D", "N", "K", "Ns", "theta")
            },
            "parameter_flags": contract.get("parameter_flags"),
            "regime": contract.get("regime"),
            "resolved_options": contract.get("resolved_options"),
            "fixture_hashes": fixture_meta.get("fixture_hashes"),
            "fixture_git": fixture_meta.get("git"),
            "synthetic": fixture_meta.get("synthetic"),
        },
        "lineage": {
            key: record.get(key)
            for key in (
                "mode",
                "source_run",
                "source_run_id",
                "source_artifact",
                "source_artifacts",
                "corrected_from",
                "recovery_of",
                "recovered_fields",
                "correction",
            )
            if key in record
        },
        "artifacts": artifacts,
        "compactness": {
            "result_json_bytes": (
                (run_dir / "result.json").stat().st_size
                if (run_dir / "result.json").is_file()
                else None
            ),
            "raw_artifact_bytes": raw_bytes,
            "arrays_embedded_in_report": False,
        },
        "failure": failure,
    }


def _epsilon_signature(run: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    signature = []
    for event in run.get("trace") or []:
        if (
            event.get("phase") != "adam"
            or event.get("kind") != "rng_standard_normal"
        ):
            continue
        signature.append(
            (
                event.get("start_id"),
                event.get("evaluation_id"),
                tuple(event.get("shape") or ()),
                event.get("value_hash"),
            )
        )
    return signature


def _fixed_comparison(
    baseline: Mapping[str, Any], torch_run: Mapping[str, Any]
) -> dict[str, Any]:
    left = baseline.get("fixed_100_run") or {}
    right = torch_run.get("fixed_100_run") or {}
    output = {
        "case": baseline.get("case"),
        "seed": baseline.get("seed"),
        "torch_arm": torch_run.get("arm"),
        "available": bool(left and right),
        "note": (
            "Theta-hash mismatches are reported as exact-hash differences only; "
            "they do not establish structural branch divergence."
        ),
    }
    if not left or not right:
        output["reason"] = "fixed replay auxiliary data missing"
        return output
    left_runs = (left.get("result") or {}).get("optimizer_runs") or []
    right_runs = (right.get("result") or {}).get("optimizer_runs") or []
    left_epsilon = _epsilon_signature(left)
    right_epsilon = _epsilon_signature(right)
    output["epsilon"] = {
        "left_events": len(left_epsilon),
        "right_events": len(right_epsilon),
        "same_hashes_and_shapes": left_epsilon == right_epsilon,
        "matching_events": sum(
            one == two for one, two in zip(left_epsilon, right_epsilon)
        ),
    }
    comparisons = []
    for index, (one, two) in enumerate(zip(left_runs, right_runs)):
        item: dict[str, Any] = {
            "start_index": index,
            "numpy_start_id": one.get("start_id"),
            "torch_start_id": two.get("start_id"),
            "numpy_updates": one.get("iterations"),
            "torch_updates": two.get("iterations"),
        }
        try:
            one_f = _decode_selected(
                one.get("pre_update_objectives", one.get("F")),
                baseline["run_dir"],
            )
            two_f = _decode_selected(
                two.get("pre_update_objectives", two.get("F")),
                torch_run["run_dir"],
            )
            item["objective_difference"] = _difference(two_f, one_f)
            one_g = _decode_selected(one.get("gradients"), baseline["run_dir"])
            two_g = _decode_selected(
                two.get("gradients"), torch_run["run_dir"]
            )
            item["gradient_difference"] = _difference(two_g, one_g)
        except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
            item["decode_error"] = f"{type(exc).__name__}: {exc}"
        one_hashes = one.get("post_update_theta_hashes") or []
        two_hashes = two.get("post_update_theta_hashes") or []
        item["post_update_theta_hashes"] = {
            "numpy_count": len(one_hashes),
            "torch_count": len(two_hashes),
            "matching_positions": sum(
                a == b for a, b in zip(one_hashes, two_hashes)
            ),
            "all_match": one_hashes == two_hashes,
        }
        one_candidate = _find_candidate(left, f"selected_start:{index}")
        two_candidate = _find_candidate(right, f"selected_start:{index}")
        if one_candidate and two_candidate:
            try:
                one_theta = _decode_selected(
                    one_candidate.get("theta"), baseline["run_dir"]
                )
                two_theta = _decode_selected(
                    two_candidate.get("theta"), torch_run["run_dir"]
                )
                item["initial_start_theta_difference"] = _difference(
                    two_theta, one_theta
                )
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                item[
                    "initial_start_theta_decode_error"
                ] = f"{type(exc).__name__}: {exc}"
        else:
            item["initial_start_theta_difference"] = None
        comparisons.append(item)
    output["start_count"] = {
        "numpy": len(left_runs),
        "torch": len(right_runs),
        "same": len(left_runs) == len(right_runs),
    }
    output["starts"] = comparisons
    theta_inputs_match = bool(
        comparisons
        and all(
            (item.get("initial_start_theta_difference") or {}).get(
                "comparable"
            )
            and (
                item["initial_start_theta_difference"].get("max_abs")
                is not None
            )
            and item["initial_start_theta_difference"]["max_abs"] <= VP_RTOL
            for item in comparisons
        )
    )
    output["matched_input_requirements"] = {
        "same_start_count": output["start_count"]["same"],
        "same_epsilon_hashes_and_shapes": output["epsilon"][
            "same_hashes_and_shapes"
        ],
        "every_initial_theta_comparable_and_max_abs_le": VP_RTOL,
        "initial_theta_requirement_met": theta_inputs_match,
    }
    output["matched_input_numeric_comparison_valid"] = bool(
        output["start_count"]["same"]
        and output["epsilon"]["same_hashes_and_shapes"]
        and theta_inputs_match
        and comparisons
        and all(
            (item.get("objective_difference") or {}).get("comparable")
            and (item.get("gradient_difference") or {}).get("comparable")
            for item in comparisons
        )
    )
    return output


def _structural_comparison(
    baseline: Mapping[str, Any], torch_run: Mapping[str, Any]
) -> dict[str, Any]:
    one = (baseline.get("complete_run") or {}).get("result") or {}
    two = (torch_run.get("complete_run") or {}).get("result") or {}
    fields = {
        "iterations_by_start": (
            one.get("iterations_by_start"),
            two.get("iterations_by_start"),
        ),
        "selected_candidate_id": (
            one.get("selected_candidate_id"),
            two.get("selected_candidate_id"),
        ),
        "pruned": (one.get("pruned"), two.get("pruned")),
        "K": (one.get("K"), two.get("K")),
        "boost_accepted": (
            (one.get("boost") or {}).get("changed"),
            (two.get("boost") or {}).get("changed"),
        ),
    }
    differences = {
        name: {"numpy": values[0], "torch": values[1]}
        for name, values in fields.items()
        if values[0] != values[1]
    }
    return {
        "case": baseline.get("case"),
        "seed": baseline.get("seed"),
        "torch_arm": torch_run.get("arm"),
        "structural_branch_difference": bool(differences),
        "differences": differences,
    }


def _final_scores(
    record: Mapping[str, Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    rescores = record.get("fine_rescores") or {}
    final_id = _resolve_alias(rescores, "final:0")
    scores = (rescores.get("scores") or {}).get(final_id)
    return final_id, scores if isinstance(scores, list) else []


def _fine_comparison(
    baseline: Mapping[str, Any], torch_run: Mapping[str, Any]
) -> dict[str, Any]:
    left_id, left_scores = _final_scores(baseline)
    right_id, right_scores = _final_scores(torch_run)
    output = {
        "case": baseline.get("case"),
        "seed": baseline.get("seed"),
        "torch_arm": torch_run.get("arm"),
        "numpy_final_alias": left_id,
        "torch_final_alias": right_id,
        "numpy": _mean_spread(row.get("elbo") for row in left_scores),
        "torch": _mean_spread(row.get("elbo") for row in right_scores),
    }
    left_by_seed = {row.get("seed"): row for row in left_scores}
    right_by_seed = {row.get("seed"): row for row in right_scores}
    common = sorted(set(left_by_seed).intersection(right_by_seed))
    common_streams_declared = bool(
        (baseline.get("fine_rescores") or {}).get("common_streams")
        and (torch_run.get("fine_rescores") or {}).get("common_streams")
    )
    compatibility = []
    paired = []
    for seed in common:
        left_row = left_by_seed[seed]
        right_row = right_by_seed[seed]
        checks = {
            name: left_row.get(name) == right_row.get(name)
            for name in (
                "samples_per_component",
                "samples_total",
                "I_sk_shape",
                "J_sjk_shape",
            )
        }
        # Recovered artifacts retain sample dimensions but no complete_run.
        # Derive K from those explicit dimensions instead of comparing None.
        left_per = left_row.get("samples_per_component")
        right_per = right_row.get("samples_per_component")
        left_total = left_row.get("samples_total")
        right_total = right_row.get("samples_total")
        checks["final_K"] = bool(
            all(
                isinstance(value, int) and value > 0
                for value in (left_per, right_per, left_total, right_total)
            )
            and left_total % left_per == 0
            and right_total % right_per == 0
            and left_total // left_per == right_total // right_per
        )
        compatible = common_streams_declared and all(checks.values())
        compatibility.append(
            {"stream": seed, "compatible": compatible, "checks": checks}
        )
        if (
            compatible
            and _number(right_row.get("elbo")) is not None
            and _number(left_row.get("elbo")) is not None
        ):
            paired.append(
                (seed, float(right_row["elbo"]) - float(left_row["elbo"]))
            )
    deltas = [delta for _, delta in paired]
    output["paired_torch_minus_numpy"] = {
        **_mean_spread(deltas),
        "max_abs": max((abs(value) for value in deltas), default=None),
        "per_stream": [
            {"stream": seed, "elbo_delta": delta} for seed, delta in paired
        ],
    }
    output["common_streams_declared"] = common_streams_declared
    output["stream_compatibility"] = compatibility
    output["paired_claim_valid"] = bool(paired) and len(paired) == len(common)
    output["available"] = bool(left_scores and right_scores)
    return output


def _load_runs(
    campaign: Path, measurement_kind: str = "instrumented"
) -> list[dict[str, Any]]:
    runs_root = campaign / "runs"
    records = []
    if not runs_root.is_dir():
        return records
    for run_dir in sorted(
        path for path in runs_root.iterdir() if path.is_dir()
    ):
        result = _read_json(run_dir / "result.json")
        failure = _read_json(run_dir / "failure.json")
        record = result or failure
        if record is None:
            continue
        row = dict(record)
        if not row.get("fine_rescores") and row.get("rescored_candidates"):
            row["fine_rescores"] = row["rescored_candidates"]
            row["fine_rescores_source_field"] = "rescored_candidates"
        row["run_id"] = run_dir.name
        row["arm"] = _arm(row)
        row["measurement_kind"] = measurement_kind
        row["run_dir"] = run_dir
        row["result_present"] = result is not None
        row["failure_present"] = failure is not None
        records.append(row)
    return records


def _campaign_metadata(campaign: Path, arms: Sequence[str]) -> dict[str, Any]:
    ledger_path = campaign / "invocations.jsonl"
    ledger = []
    if ledger_path.is_file():
        for line_number, line in enumerate(
            ledger_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                ledger.append(
                    {
                        "line": line_number,
                        "status": "invalid",
                        "error": str(exc),
                    }
                )
            else:
                if isinstance(item, dict):
                    ledger.append(item)
    repository = Path(__file__).resolve().parents[2]
    experiment_root = repository / "dev" / "experiments" / "stage4_torch"

    session_order_path = campaign / "session-order.json"
    session_order = _read_json(session_order_path) or {}
    session_rows = (
        session_order.get("rows", [])
        if isinstance(session_order, dict)
        else []
    )

    def artifact(path: Path, record: Any = None) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            shown_path = path.resolve().relative_to(repository).as_posix()
        except ValueError:
            shown_path = str(path)
        output = {
            "path": shown_path,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if record is not None:
            output["record"] = record
        return output

    tracked = {}
    tracked_ledger_path = experiment_root / "campaign-ledger.json"
    if session_order_path.is_file() or campaign.name == "campaign":
        tracked_ledger = _read_json(tracked_ledger_path)
        tracked["campaign_ledger"] = artifact(
            tracked_ledger_path,
            (
                {
                    key: tracked_ledger.get(key)
                    for key in (
                        "note",
                        "worker_wall_by_arm_s",
                        "statuses",
                        "primary_fits",
                    )
                }
                if isinstance(tracked_ledger, dict)
                else None
            ),
        )
        if tracked["campaign_ledger"] is not None:
            tracked["campaign_ledger"][
                "matches_session_order_sha256"
            ] = session_order_path.is_file() and _sha256(
                tracked_ledger_path
            ) == _sha256(
                session_order_path
            )
        measured_source_record_path = experiment_root / "measured-source.json"
        tracked["measured_source"] = artifact(
            measured_source_record_path,
            _read_json(measured_source_record_path),
        )
    control_source_path = experiment_root / "control-source.json"
    if campaign.name == "controls":
        control_source = _read_json(control_source_path)
        tracked["control_source"] = artifact(
            control_source_path, control_source
        )
    kernel_diagnostics_path = experiment_root / "kernel-diagnostics.json"
    if campaign.name == "diagnostics":
        kernel_diagnostics = _read_json(kernel_diagnostics_path)
        tracked["kernel_diagnostics"] = artifact(
            kernel_diagnostics_path,
            (
                {
                    "note": kernel_diagnostics.get("note"),
                    "run_count": len(kernel_diagnostics.get("runs", [])),
                }
                if isinstance(kernel_diagnostics, dict)
                else None
            ),
        )

    measured_source_path = campaign / "measured_source.zip"
    measured_source = artifact(measured_source_path)
    if measured_source is not None and isinstance(
        tracked.get("control_source"), dict
    ):
        declared = (tracked["control_source"].get("record") or {}).get(
            "sha256"
        )
        measured_source["matches_tracked_control_source_sha256"] = (
            declared == measured_source["sha256"]
        )
    if measured_source is not None and isinstance(
        tracked.get("measured_source"), dict
    ):
        declared = (tracked["measured_source"].get("record") or {}).get(
            "archive_sha256"
        )
        measured_source["matches_tracked_measured_source_sha256"] = (
            declared == measured_source["sha256"]
        )

    manifests = []
    for path in sorted(campaign.glob("manifest_*.json")):
        value = _read_json(path) or {}
        environment = value.get("environment") or {}
        manifests.append(
            {
                "path": path.relative_to(campaign).as_posix(),
                "sha256": _sha256(path),
                "request": value.get("request"),
                "measured_source_hashes": environment.get("source_hashes"),
                "plan_hash": environment.get("plan_hash"),
                "provenance_validation": environment.get(
                    "provenance_validation"
                ),
            }
        )
    coordinators = []
    for path in sorted(campaign.glob("campaign_*.json")):
        coordinators.append(
            {
                "path": path.relative_to(campaign).as_posix(),
                "sha256": _sha256(path),
                "record": _read_json(path),
            }
        )
    entries = []
    for arm in arms:
        backend, device = arm.rsplit("_", 1)
        invocation_events = [
            item
            for item in ledger
            if item.get("backend") == backend and item.get("device") == device
        ]
        session_events = [
            item
            for item in session_rows
            if item.get("backend") == backend and item.get("device") == device
        ]
        arm_manifests = [
            item
            for item in manifests
            if (item.get("request") or {}).get("backend") == backend
            and (item.get("request") or {}).get("device") == device
        ]
        entries.append(
            {
                "arm": arm,
                "order_source": (
                    "canonical session-order.json"
                    if session_events
                    else (
                        "canonical invocations.jsonl ledger"
                        if invocation_events
                        else "explicit report arm order"
                    )
                ),
                "session_order_events": session_events,
                "ledger_events": invocation_events,
                "manifests": arm_manifests,
            }
        )
    return {
        "arm_order": entries,
        "session_order": artifact(
            session_order_path,
            (
                {
                    key: session_order.get(key)
                    for key in (
                        "note",
                        "worker_wall_by_arm_s",
                        "statuses",
                        "primary_fits",
                    )
                }
                if isinstance(session_order, dict)
                else None
            ),
        ),
        "canonical_ledger": (
            {
                "path": ledger_path.relative_to(campaign).as_posix(),
                "sha256": _sha256(ledger_path),
                "events": ledger,
            }
            if ledger_path.is_file()
            else None
        ),
        "manifest_artifacts": manifests,
        "coordinator_artifacts": coordinators,
        "measured_source_archive": measured_source,
        "tracked_provenance_artifacts": tracked,
    }


def _supplement_lane(
    root: Path | None,
    measurement_kind: str,
    instrumented_lookup: Mapping[tuple[Any, Any, str], Mapping[str, Any]],
) -> dict[str, Any]:
    if root is None:
        return {
            "requested": False,
            "directory": None,
            "available_records": 0,
            "runs": [],
        }
    records = _load_runs(root, measurement_kind)
    supplement_lookup = {
        (record.get("case"), record.get("seed"), record["arm"]): record
        for record in records
    }
    compact = [
        _compact_run(record, record["run_dir"], root) for record in records
    ]
    links = []
    recovered_comparisons = []
    for record in records:
        original = instrumented_lookup.get(
            (record.get("case"), record.get("seed"), record["arm"])
        )
        links.append(
            {
                "supplement_run_id": record.get("run_id"),
                "case": record.get("case"),
                "seed": record.get("seed"),
                "arm": record.get("arm"),
                "original_run_id": (
                    original.get("run_id") if original else None
                ),
                "original_match_status": "matched" if original else "unknown",
                "lineage": {
                    key: record.get(key)
                    for key in (
                        "source_run",
                        "source_run_id",
                        "source_artifact",
                        "source_artifacts",
                        "corrected_from",
                        "recovery_of",
                        "recovered_fields",
                        "correction",
                    )
                    if key in record
                },
            }
        )
        if measurement_kind == "auxiliary_recovery":
            numpy_baseline = supplement_lookup.get(
                (record.get("case"), record.get("seed"), "numpy_cpu")
            )
            recovered = {
                "recovery_run_id": record.get("run_id"),
                "case": record.get("case"),
                "seed": record.get("seed"),
                "arm": record.get("arm"),
                "fixed_100_present": bool(record.get("fixed_100_run")),
                "fine_rescores_present": bool(record.get("fine_rescores")),
            }
            if record["arm"] != "numpy_cpu" and numpy_baseline is not None:
                recovered["numpy_reference_run_id"] = numpy_baseline.get(
                    "run_id"
                )
                recovered["numpy_reference_source"] = "auxiliary_recovery"
                recovered["fixed_100_vs_numpy"] = _fixed_comparison(
                    numpy_baseline, record
                )
                recovered["fine_rescores_vs_numpy"] = _fine_comparison(
                    numpy_baseline, record
                )
            else:
                recovered["cross_arm_comparison"] = {
                    "status": "not_compared",
                    "reason": (
                        "NumPy recovery is the cross-arm reference"
                        if record["arm"] == "numpy_cpu"
                        else "matching frozen NumPy artifact missing"
                    ),
                }
            recovered_comparisons.append(recovered)
    return {
        "requested": True,
        "directory": str(root),
        "available_records": len(records),
        "campaign_metadata": _campaign_metadata(
            root,
            _ordered((record["arm"] for record in records), ARM_ORDER),
        ),
        "runs": compact,
        "links_to_frozen_instrumented_runs": links,
        "recovered_auxiliary_comparisons": recovered_comparisons,
    }


def _make_report(
    campaign: Path,
    controls: Path | None = None,
    diagnostics: Path | None = None,
    recovery: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = _load_runs(campaign)
    control_records = _load_runs(controls, "clean_control") if controls else []
    arms = _ordered((row["arm"] for row in records), ARM_ORDER)
    cases = _ordered((str(row.get("case")) for row in records), CASE_ORDER)
    records.sort(
        key=lambda row: (
            cases.index(str(row.get("case"))),
            int(row.get("seed") or -1),
            arms.index(row["arm"]),
        )
    )
    timing_rows = [_timing_row(row) for row in records]
    control_timing_rows = [_timing_row(row) for row in control_records]
    lookup = {
        (row.get("case"), row.get("seed"), row["arm"]): row for row in records
    }
    timing_lookup = {
        (row.get("case"), row.get("seed"), row["arm"]): row
        for row in timing_rows
    }
    for row in timing_rows:
        if row["arm"] == "numpy_cpu":
            continue
        baseline = timing_lookup.get((row["case"], row["seed"], "numpy_cpu"))
        if row["quarantined_original_kernel_gate"] or (
            baseline and baseline["quarantined_original_kernel_gate"]
        ):
            continue
        left = baseline and baseline.get("primary_complete_total_s")
        right = row.get("primary_complete_total_s")
        if _number(left) not in (None, 0) and _number(right) is not None:
            row["paired_torch_over_numpy"] = float(right) / float(left)

    instrumented_timing_lookup = {
        (row.get("case"), row.get("seed"), row["arm"]): row
        for row in timing_rows
    }
    control_timing_lookup = {
        (row.get("case"), row.get("seed"), row["arm"]): row
        for row in control_timing_rows
    }
    clean_equivalence = []
    for row in control_timing_rows:
        instrumented = instrumented_timing_lookup.get(
            (row.get("case"), row.get("seed"), row["arm"])
        )
        clean = row.get("primary_complete_total_s")
        traced = instrumented and instrumented.get("primary_complete_total_s")
        if _number(traced) not in (None, 0) and _number(clean) is not None:
            row["clean_over_instrumented"] = float(clean) / float(traced)
        if row["arm"] != "numpy_cpu":
            baseline = control_timing_lookup.get(
                (row.get("case"), row.get("seed"), "numpy_cpu")
            )
            numpy_clean = baseline and baseline.get("primary_complete_total_s")
            if (
                _number(numpy_clean) not in (None, 0)
                and _number(clean) is not None
            ):
                row["paired_torch_over_numpy"] = float(clean) / float(
                    numpy_clean
                )
    for clean_record in control_records:
        instrumented = lookup.get(
            (
                clean_record.get("case"),
                clean_record.get("seed"),
                clean_record["arm"],
            )
        )
        if instrumented is None:
            clean_equivalence.append(
                {
                    "case": clean_record.get("case"),
                    "seed": clean_record.get("seed"),
                    "reference_arm": clean_record["arm"],
                    "actual_arm": clean_record["arm"],
                    "actual_run_id": clean_record.get("run_id"),
                    "reference_run_id": None,
                    "context": "clean_control_vs_instrumented",
                    "status": "unknown",
                    "reason": "matching frozen instrumented artifact missing",
                }
            )
        else:
            clean_equivalence.append(
                _final_equivalence(
                    instrumented,
                    clean_record,
                    context="clean_control_vs_instrumented",
                )
            )

    control_summary = []
    clean_ratio_summary = []
    for case in cases:
        for arm in arms:
            selected = [
                row
                for row in control_timing_rows
                if row["case"] == case and row["arm"] == arm
            ]
            if selected:
                control_summary.append(
                    {
                        "case": case,
                        "arm": arm,
                        "clean_primary_complete_total_s": _median_range(
                            row["primary_complete_total_s"] for row in selected
                        ),
                        "clean_over_instrumented": _median_range(
                            row["clean_over_instrumented"] for row in selected
                        ),
                        "seeds": [row["seed"] for row in selected],
                    }
                )
                if arm != "numpy_cpu":
                    clean_ratio_summary.append(
                        {
                            "case": case,
                            "arm": arm,
                            "paired_clean_torch_over_numpy": _median_range(
                                row["paired_torch_over_numpy"]
                                for row in selected
                            ),
                            "note": "one paired control seed per case and arm",
                        }
                    )

    timing_summary = []
    ratio_summary = []
    for case in cases:
        for arm in arms:
            selected = [
                row
                for row in timing_rows
                if row["case"] == case and row["arm"] == arm
            ]
            if selected:
                timing_summary.append(
                    {
                        "case": case,
                        "arm": arm,
                        "primary_complete_total_s": _median_range(
                            row["primary_complete_total_s"] for row in selected
                        ),
                        "external_complete_host_wall_s": _median_range(
                            row["external_complete_host_wall_s"]
                            for row in selected
                        ),
                        "seeds": [row["seed"] for row in selected],
                    }
                )
                if arm != "numpy_cpu":
                    quarantined_pairs = sum(
                        row["quarantined_original_kernel_gate"]
                        or bool(
                            timing_lookup.get(
                                (row["case"], row["seed"], "numpy_cpu"), {}
                            ).get("quarantined_original_kernel_gate")
                        )
                        for row in selected
                    )
                    ratio_summary.append(
                        {
                            "case": case,
                            "arm": arm,
                            "paired_torch_over_numpy": _median_range(
                                row["paired_torch_over_numpy"]
                                for row in selected
                            ),
                            "quarantined_original_pairs_excluded": quarantined_pairs,
                        }
                    )

    paired_records = []
    structural = []
    fixed = []
    fine = []
    for record in records:
        if record["arm"] == "numpy_cpu":
            continue
        baseline = lookup.get(
            (record.get("case"), record.get("seed"), "numpy_cpu")
        )
        if baseline is None:
            continue
        paired_records.append(
            _final_equivalence(
                baseline,
                record,
                context="torch_vs_numpy_instrumented_complete_fit",
            )
        )
        structural.append(_structural_comparison(baseline, record))
        fixed.append(_fixed_comparison(baseline, record))
        fine.append(_fine_comparison(baseline, record))

    compact_runs = [
        _compact_run(record, record["run_dir"], campaign) for record in records
    ]
    compact_controls = (
        [
            _compact_run(record, record["run_dir"], controls)
            for record in control_records
        ]
        if controls
        else []
    )
    diagnostics_lane = _supplement_lane(
        diagnostics, "corrected_diagnostic", lookup
    )
    recovery_lane = _supplement_lane(recovery, "auxiliary_recovery", lookup)
    recovered_by_key = {
        (item.get("case"), item.get("seed"), item.get("arm")): item
        for item in recovery_lane.get("recovered_auxiliary_comparisons", [])
        if item.get("arm") != "numpy_cpu"
    }
    effective_fixed = []
    effective_fine = []
    for frozen_fixed, frozen_fine in zip(fixed, fine):
        key = (
            frozen_fixed.get("case"),
            frozen_fixed.get("seed"),
            frozen_fixed.get("torch_arm"),
        )
        recovered = recovered_by_key.get(key) or {}
        recovered_fixed = recovered.get("fixed_100_vs_numpy")
        recovered_fine = recovered.get("fine_rescores_vs_numpy")
        selected_fixed = (
            recovered_fixed
            if not frozen_fixed.get("available") and recovered_fixed
            else frozen_fixed
        )
        selected_fine = (
            recovered_fine
            if not frozen_fine.get("available") and recovered_fine
            else frozen_fine
        )
        effective_fixed.append(
            {
                **selected_fixed,
                "comparison_source": (
                    "auxiliary_recovery"
                    if selected_fixed is recovered_fixed
                    else "frozen_instrumented"
                ),
            }
        )
        effective_fine.append(
            {
                **selected_fine,
                "comparison_source": (
                    "auxiliary_recovery"
                    if selected_fine is recovered_fine
                    else "frozen_instrumented"
                ),
            }
        )
    failures = []
    kernel_failures = []
    for run in compact_runs:
        if run["failure"] is not None:
            failures.append(
                {
                    "run_id": run["run_id"],
                    "primary_saved": run["primary_saved"],
                    **run["failure"],
                }
            )
        if run["kernel"]["all_pass"] is False:
            kernel_failures.append(
                {
                    "run_id": run["run_id"],
                    "case": run["case"],
                    "seed": run["seed"],
                    "arm": run["arm"],
                    "errors": run["kernel"]["failed_errors"],
                }
            )

    missing = []
    for run in compact_runs:
        if run["primary_saved"] and not run["auxiliary_complete"]:
            missing.append(
                {
                    "run_id": run["run_id"],
                    "missing": "auxiliary diagnostics after saved primary fit",
                    "failure_message": (run.get("failure") or {}).get(
                        "message"
                    ),
                }
            )
    for case in cases:
        for seed in (1701, 1702, 1703):
            present = {arm for c, s, arm in lookup if c == case and s == seed}
            if present and set(arms).difference(present):
                missing.append(
                    {
                        "case": case,
                        "seed": seed,
                        "missing_arms": [
                            arm for arm in arms if arm not in present
                        ],
                    }
                )

    coordinator = _campaign_metadata(campaign, arms)
    control_metadata = (
        _campaign_metadata(
            controls,
            _ordered((record["arm"] for record in control_records), ARM_ORDER),
        )
        if controls
        else None
    )
    process_totals = []
    for arm in arms:
        arm_rows = [row for row in timing_rows if row["arm"] == arm]
        process_totals.append(
            {
                "arm": arm,
                "process_spawn_to_exit_s": sum(
                    float(row["process_spawn_to_exit_s"])
                    for row in arm_rows
                    if _number(row["process_spawn_to_exit_s"]) is not None
                ),
                "worker_wall_s": sum(
                    float(row["worker_wall_s"])
                    for row in arm_rows
                    if _number(row["worker_wall_s"]) is not None
                ),
                "record_count": len(arm_rows),
                "campaign_arm_metadata": next(
                    (
                        item
                        for item in coordinator["arm_order"]
                        if item["arm"] == arm
                    ),
                    None,
                ),
            }
        )

    valid_fixed = [
        item
        for item in effective_fixed
        if item.get("matched_input_numeric_comparison_valid")
    ]
    fixed_starts = [
        start for item in valid_fixed for start in item.get("starts", [])
    ]
    quality_summary = {
        "paired_complete_fit_status_counts": {
            status: sum(
                item.get("status") == status for item in paired_records
            )
            for status in ("matched", "mismatched", "unknown")
        },
        "structural_branch_difference_count": sum(
            item.get("structural_branch_difference") for item in structural
        ),
        "fixed_100_matched_input_comparison_count": len(valid_fixed),
        "fixed_100_max_objective_abs": max(
            (
                (start.get("objective_difference") or {}).get("max_abs")
                for start in fixed_starts
            ),
            default=None,
        ),
        "fixed_100_max_gradient_abs": max(
            (
                (start.get("gradient_difference") or {}).get("max_abs")
                for start in fixed_starts
            ),
            default=None,
        ),
        "fine_rescore_max_abs_elbo_delta": max(
            (
                (item.get("paired_torch_minus_numpy") or {}).get("max_abs")
                for item in effective_fine
                if (item.get("paired_torch_minus_numpy") or {}).get("max_abs")
                is not None
            ),
            default=None,
        ),
    }

    report = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign),
        "scope": {
            "decision": None,
            "recommendation": None,
            "measurement": "one complete variational fit with frozen GP factors",
            "production_evidence": False,
            "traceless_evidence": bool(control_records),
            "trace_disabled_control_evidence": bool(control_records),
            "stage_is_end_to_end_solver_timing": False,
            "primary_timing": "complete_run.timing.total",
            "host_residual_definition": (
                "external_complete_host_wall_s minus primary_complete_total_s; "
                "a composite of source validation and report construction"
            ),
        },
        "prototype_scope": {
            "instrumented_campaign": str(campaign),
            "clean_controls": str(controls) if controls else None,
            "corrected_diagnostics": str(diagnostics) if diagnostics else None,
            "auxiliary_recovery": str(recovery) if recovery else None,
            "complete_fit_limit": "72 instrumented plus at most 24 clean controls",
            "full_vbmc_trajectory": False,
            "production_backend": False,
        },
        "ordering": coordinator,
        "cases": cases,
        "counts": {
            "records": len(records),
            "primary_saved": sum(run["primary_saved"] for run in compact_runs),
            "auxiliary_complete": sum(
                run["auxiliary_complete"] for run in compact_runs
            ),
            "failure_logs": len(failures),
            "kernel_gate_failures": len(kernel_failures),
        },
        "timing": {
            "per_case_arm": timing_summary,
            "paired_ratios": ratio_summary,
            "numpy_bare_controls": [
                row
                for row in timing_rows
                if row.get("bare_numpy_fit_s") is not None
            ],
            "process_wall_budget": process_totals,
        },
        "clean_controls": {
            "requested": controls is not None,
            "directory": str(controls) if controls else None,
            "available_records": len(control_records),
            "campaign_metadata": control_metadata,
            "note": (
                "Clean controls are separately measured trace-off fits. No "
                "subtraction is used to estimate an untraced runtime."
            ),
            "per_case_arm": control_summary,
            "paired_torch_over_numpy": clean_ratio_summary,
            "trace_equivalence": clean_equivalence,
            "equivalence_counts": {
                status: sum(
                    item.get("status") == status for item in clean_equivalence
                )
                for status in ("matched", "mismatched", "unknown")
            },
            "runs": compact_controls,
        },
        "corrected_diagnostics": diagnostics_lane,
        "auxiliary_recovery": recovery_lane,
        "paired_complete_fits": paired_records,
        "quality_summary": quality_summary,
        "structural_comparisons": structural,
        "fixed_100_comparisons": effective_fixed,
        "fine_rescore_comparisons": effective_fine,
        "kernel_gate_failures": kernel_failures,
        "failures": failures,
        "missing_data": missing,
        "runs": compact_runs,
    }
    return _strict(report), [
        _strict(row) for row in timing_rows + control_timing_rows
    ]


def _fmt(value: Any, digits: int = 6) -> str:
    return "" if _number(value) is None else f"{float(value):.{digits}g}"


def _format_range(summary: Mapping[str, Any]) -> str:
    if not summary or not summary.get("n"):
        return "missing"
    return (
        f"{_fmt(summary.get('median'))} "
        f"[{_fmt(summary.get('min'))}, {_fmt(summary.get('max'))}] "
        f"(n={summary.get('n')})"
    )


def _tables(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 4 Torch campaign evidence tables",
        "",
        "Instrumented tables use the complete variational-fit total (`complete_run.timing.total`) as their primary timing. Clean performance comparisons use the separately measured trace-disabled totals and paired clean Torch/NumPy ratios. External host wall, setup, GP extraction/transfer, optimizer, fine scoring/pruning, and the source-validation/report-construction residual remain separate. Both lanes are prototype measurements with frozen GP factors, outside an end-to-end solver or production backend.",
        "",
        "## Complete-fit timing",
        "",
        "| Case | Arm | Primary seconds, median [range] | External host seconds, median [range] |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in report["timing"]["per_case_arm"]:
        lines.append(
            f"| {row['case']} | {row['arm']} | "
            f"{_format_range(row['primary_complete_total_s'])} | "
            f"{_format_range(row['external_complete_host_wall_s'])} |"
        )
    lines += [
        "",
        "## Paired complete-fit ratios",
        "",
        "| Case | Torch arm | Torch / NumPy, median [range] |",
        "| --- | --- | ---: |",
    ]
    for row in report["timing"]["paired_ratios"]:
        lines.append(
            f"| {row['case']} | {row['arm']} | "
            f"{_format_range(row['paired_torch_over_numpy'])} |"
        )
    quarantined_pairs = sum(
        row.get("quarantined_original_pairs_excluded", 0)
        for row in report["timing"]["paired_ratios"]
    )
    if quarantined_pairs:
        lines += [
            "",
            f"The instrumented ratio summaries exclude {quarantined_pairs} paired observations whose original raw artifacts failed a kernel gate. Those failures remain recorded in the raw-failure sections; corrected diagnostics do not overwrite them.",
        ]
    if report["clean_controls"]["available_records"]:
        lines += [
            "",
            "## Separately measured clean controls",
            "",
            "| Case | Arm | Clean primary seconds, median [range] | Clean / instrumented, median [range] |",
            "| --- | --- | ---: | ---: |",
        ]
        for row in report["clean_controls"]["per_case_arm"]:
            lines.append(
                f"| {row['case']} | {row['arm']} | "
                f"{_format_range(row['clean_primary_complete_total_s'])} | "
                f"{_format_range(row['clean_over_instrumented'])} |"
            )
        lines += [
            "",
            "Clean-control values are independent trace-off measurements; the report does not subtract traced and clean timings to estimate an untraced runtime.",
            "",
            "### Clean-control paired backend ratios",
            "",
            "| Case | Torch arm | Clean Torch / clean NumPy | Replicates |",
            "| --- | --- | ---: | ---: |",
        ]
        for row in report["clean_controls"]["paired_torch_over_numpy"]:
            summary = row["paired_clean_torch_over_numpy"]
            lines.append(
                f"| {row['case']} | {row['arm']} | "
                f"{_format_range(summary)} | {summary.get('n')} |"
            )
        lines += [
            "",
            "Each clean backend ratio has one paired seed (`n=1`); it has no three-seed spread.",
            "",
            "### Clean-control parity with the same frozen fit",
            "",
            "| Clean run | Frozen run | Status | Posterior mismatches | Statistic mismatches | Structure mismatches |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in report["clean_controls"]["trace_equivalence"]:
            parameter_mismatches = ", ".join(
                name
                for name, comparison in row.get(
                    "posterior_parameters", {}
                ).items()
                if comparison.get("status") != "matched"
            )
            statistic_mismatches = ", ".join(
                name
                for name, comparison in row.get("statistics", {}).items()
                if comparison.get("status") != "matched"
            )
            structure_mismatches = ", ".join(
                name
                for name, comparison in row.get("structure", {}).items()
                if comparison.get("status") != "matched"
            )
            lines.append(
                f"| {row.get('actual_run_id')} | "
                f"{row.get('reference_run_id')} | {row.get('status')} | "
                f"{parameter_mismatches} | {statistic_mismatches} | "
                f"{structure_mismatches} |"
            )
        lines += [
            "",
            "Trace events, retained candidates, optimizer traces, fixed replay, and fine rescoring are intentionally absent from clean controls and are recorded as `not_compared` rather than inferred.",
            "",
            "### Clean-control warmup and memory",
            "",
            "| Run | Warmup calls | Warmup walls s | Post-fit peak RSS bytes | CUDA allocated/reserved bytes |",
            "| --- | ---: | --- | ---: | --- |",
        ]
        for run in report["clean_controls"]["runs"]:
            warmup = run.get("warmup") or {}
            memory = run.get("post_fit_memory_before_diagnostics") or {}
            cuda = memory.get("cuda") or {}
            lines.append(
                f"| {run['run_id']} | {warmup.get('calls')} | "
                f"{warmup.get('walls_s')} | "
                f"{memory.get('process_peak_rss_bytes')} | "
                f"{cuda.get('max_allocated')}/{cuda.get('max_reserved')} |"
            )
    lines += [
        "",
        "## Paired complete-fit final-output parity",
        "",
        "| NumPy reference | Torch run | Status | Posterior mismatches | Statistic mismatches | Structure mismatches |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["paired_complete_fits"]:
        parameter_mismatches = ", ".join(
            name
            for name, comparison in row["posterior_parameters"].items()
            if comparison.get("status") != "matched"
        )
        statistic_mismatches = ", ".join(
            name
            for name, comparison in row["statistics"].items()
            if comparison.get("status") != "matched"
        )
        structure_mismatches = ", ".join(
            name
            for name, comparison in row["structure"].items()
            if comparison.get("status") != "matched"
        )
        lines.append(
            f"| {row['reference_run_id']} | {row['actual_run_id']} | "
            f"{row['status']} | {parameter_mismatches} | "
            f"{statistic_mismatches} | {structure_mismatches} |"
        )
    lines += [
        "",
        "## Complete-fit structure and quality",
        "",
        "| Run | Status | Iterations | K | Pruned | Selected | Boost accepted | ELBO | Weights/scales valid |",
        "| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for run in report["runs"]:
        if not run["primary_saved"]:
            continue
        outcome = run["outcome"]
        validity = f"{outcome['valid_weights']}/{outcome['valid_scales']}"
        lines.append(
            f"| {run['run_id']} | {run['status']} | "
            f"{outcome['iterations_by_start']} | {outcome['K']} | "
            f"{outcome['pruned']} | {outcome['selected_candidate_id']} | "
            f"{outcome['boost_accepted']} | {_fmt(outcome['elbo'])} | {validity} |"
        )
    lines += [
        "",
        "## Fixed 100-update replay",
        "",
        "| Case/seed | Torch arm | Starts | Same epsilon hashes/shapes | Max objective abs | Objective RMS | Max gradient abs | Gradient RMS | Initial theta max abs | Exact post-theta hashes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["fixed_100_comparisons"]:
        starts = row.get("starts") or []
        for index, start in enumerate(starts or [{}]):
            objective = start.get("objective_difference") or {}
            gradient = start.get("gradient_difference") or {}
            theta = start.get("initial_start_theta_difference") or {}
            hashes = start.get("post_update_theta_hashes") or {}
            lines.append(
                f"| {row['case']}/{row['seed']} | {row['torch_arm']} | "
                f"{index + 1}/{len(starts)} | "
                f"{(row.get('epsilon') or {}).get('same_hashes_and_shapes')} | "
                f"{_fmt(objective.get('max_abs'))} | {_fmt(objective.get('rms'))} | "
                f"{_fmt(gradient.get('max_abs'))} | {_fmt(gradient.get('rms'))} | "
                f"{_fmt(theta.get('max_abs'))} | "
                f"{hashes.get('matching_positions')}/{hashes.get('numpy_count')} |"
            )
    lines += [
        "",
        "Exact theta-hash mismatches are not classified as structural branch divergence. Structural differences are limited to iteration counts, selected candidate, pruning, returned K, and boost acceptance.",
        "",
        "## Common-stream final-candidate rescoring",
        "",
        "| Case/seed | Torch arm | NumPy ELBO mean ± sd | Torch ELBO mean ± sd | Torch - NumPy mean ± sd | Max abs delta | Streams |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["fine_rescore_comparisons"]:
        one = row["numpy"]
        two = row["torch"]
        delta = row["paired_torch_minus_numpy"]
        lines.append(
            f"| {row['case']}/{row['seed']} | {row['torch_arm']} | "
            f"{_fmt(one.get('mean'))} ± {_fmt(one.get('std'))} | "
            f"{_fmt(two.get('mean'))} ± {_fmt(two.get('std'))} | "
            f"{_fmt(delta.get('mean'))} ± {_fmt(delta.get('std'))} | "
            f"{_fmt(delta.get('max_abs'))} | {delta.get('n')} |"
        )
    lines += [
        "",
        "## Kernel gates and failures",
        "",
        "| Run | Kernel gate | Failed quantities | Failure log |",
        "| --- | --- | --- | --- |",
    ]
    for run in report["runs"]:
        failed = ", ".join(run["kernel"]["failed_errors"]) or ""
        failure = run.get("failure") or {}
        message = str(failure.get("message") or "").replace("|", "\\|")
        lines.append(
            f"| {run['run_id']} | {run['kernel']['all_pass']} | {failed} | {message} |"
        )
    if report["corrected_diagnostics"]["available_records"]:
        lines += [
            "",
            "### Corrected kernel-only diagnostics",
            "",
            "| Supplemental run | Status | Kernel gate | Failed quantities | Frozen source link |",
            "| --- | --- | --- | --- | --- |",
        ]
        links = {
            item["supplement_run_id"]: item
            for item in report["corrected_diagnostics"].get(
                "links_to_frozen_instrumented_runs", []
            )
        }
        for run in report["corrected_diagnostics"]["runs"]:
            link = links.get(run["run_id"], {})
            lines.append(
                f"| {run['run_id']} | {run['status']} | "
                f"{run['kernel']['all_pass']} | "
                f"{', '.join(run['kernel']['failed_errors'])} | "
                f"{link.get('original_run_id')} |"
            )
        lines += [
            "",
            "Corrected diagnostics are supplemental. Original gate failures and their raw errors remain in the frozen campaign evidence.",
        ]
    if report["auxiliary_recovery"]["available_records"]:
        lines += [
            "",
            "### Recovered auxiliary diagnostics",
            "",
            "| Recovery run | Case/seed | Arm | Fixed 100 present | Fine rescores present |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in report["auxiliary_recovery"].get(
            "recovered_auxiliary_comparisons", []
        ):
            lines.append(
                f"| {row['recovery_run_id']} | {row['case']}/{row['seed']} | "
                f"{row['arm']} | {row['fixed_100_present']} | "
                f"{row['fine_rescores_present']} |"
            )
    lines += [
        "",
        "Kernel failures retain their raw error records in `results.json`; a later kernel-only correction does not silently turn an earlier failed artifact green.",
        "",
        "## Missing auxiliary data",
        "",
    ]
    if report["missing_data"]:
        for item in report["missing_data"]:
            lines.append(f"- `{item}`")
    else:
        lines.append("None recorded.")
    lines += [
        "",
        "No feasibility recommendation is made by this artifact reconciler.",
        "",
    ]
    return "\n".join(lines)


def _write_outputs(
    out: Path,
    report: Mapping[str, Any],
    timing_rows: Sequence[Mapping[str, Any]],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    columns = (
        "run_id",
        "case",
        "seed",
        "arm",
        "measurement_kind",
        "status",
        "primary_saved",
        *TIMING_FIELDS,
        "paired_torch_over_numpy",
        "clean_over_instrumented",
        "bare_numpy_fit_s",
        "bare_numpy_host_wall_s",
        "instrumented_total_over_bare_fit",
        "instrumented_host_over_bare_host",
    )
    with (out / "timings.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(timing_rows)
    (out / "tables.md").write_text(
        _tables(report), encoding="utf-8", newline="\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", required=True, help="campaign directory containing runs/"
    )
    parser.add_argument(
        "--out", required=True, help="directory for results.json and tables"
    )
    parser.add_argument(
        "--controls",
        help=(
            "optional clean-control campaign directory containing runs/; "
            "absence does not fail reconciliation"
        ),
    )
    parser.add_argument(
        "--diagnostics",
        help=(
            "optional corrected kernel-only diagnostics directory containing "
            "runs/"
        ),
    )
    parser.add_argument(
        "--recovery",
        help=(
            "optional recovered auxiliary diagnostics directory containing runs/"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign = Path(args.campaign).resolve()
    out = Path(args.out).resolve()
    controls = Path(args.controls).resolve() if args.controls else None
    diagnostics = (
        Path(args.diagnostics).resolve() if args.diagnostics else None
    )
    recovery = Path(args.recovery).resolve() if args.recovery else None
    if not (campaign / "runs").is_dir():
        raise SystemExit(f"campaign lacks runs directory: {campaign}")
    if controls is not None and not (controls / "runs").is_dir():
        raise SystemExit(f"controls directory lacks runs: {controls}")
    if diagnostics is not None and not (diagnostics / "runs").is_dir():
        raise SystemExit(f"diagnostics directory lacks runs: {diagnostics}")
    if recovery is not None and not (recovery / "runs").is_dir():
        raise SystemExit(f"recovery directory lacks runs: {recovery}")
    report, timings = _make_report(
        campaign,
        controls,
        diagnostics,
        recovery,
    )
    _write_outputs(out, report, timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
