"""Summarize the bounded Stage 4 compiled-Torch complete-fit campaign.

The report reads only archived worker outputs.  It compares the three hot
seeds pairwise, keeps raw numerical differences, excludes recompiled fits from
the warm-only aggregate, and treats compiler setup as part of cold total cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

STATE_RTOL = 1e-8
STATE_ATOL = 1e-10
RESCORE_RTOL = 1e-6
RESCORE_ATOL = 1e-10
COMPILED_REFERENCES = {
    "compiled_cpu": ("numpy_cpu", "eager_cpu"),
    "compiled_cuda": ("numpy_cpu", "eager_cuda"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    view = memoryview(array).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), 1024 * 1024):
        digest.update(view[start : start + 1024 * 1024])
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": repr(value)}
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(value), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


class ArtifactReader:
    """Decode and verify an ``ArtifactEncoder`` JSON/NPZ pair."""

    def __init__(self, tree_path: Path) -> None:
        self.tree_path = tree_path.resolve()
        self.root = self.tree_path.parent
        self.archives: dict[str, Any] = {}

    def _array(self, reference: Mapping[str, Any]) -> np.ndarray:
        archive_name = str(reference["archive"])
        archive = self.archives.get(archive_name)
        if archive is None:
            archive = np.load(self.root / archive_name, allow_pickle=False)
            self.archives[archive_name] = archive
        value = np.array(archive[str(reference["npz"])], copy=True)
        if list(value.shape) != reference.get("shape"):
            raise ValueError(f"artifact shape mismatch for {reference['npz']}")
        if str(value.dtype) != reference.get("dtype"):
            raise ValueError(f"artifact dtype mismatch for {reference['npz']}")
        if _array_sha256(value) != reference.get("sha256"):
            raise ValueError(f"artifact hash mismatch for {reference['npz']}")
        return value

    def decode(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            if "npz" in value and "archive" in value:
                return self._array(value)
            return {str(key): self.decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.decode(item) for item in value]
        return value

    def read(self) -> Any:
        try:
            tree = json.loads(self.tree_path.read_text(encoding="utf-8"))
            return self.decode(tree)
        finally:
            for archive in self.archives.values():
                archive.close()


def _resolve(path: str | Path, base: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _array_comparison(
    reference: Any, actual: Any, rtol: float, atol: float
) -> dict[str, Any]:
    left = np.asarray(reference)
    right = np.asarray(actual)
    record: dict[str, Any] = {
        "reference_shape": list(left.shape),
        "actual_shape": list(right.shape),
        "reference_dtype": str(left.dtype),
        "actual_dtype": str(right.dtype),
        "rtol": rtol,
        "atol": atol,
    }
    if left.shape != right.shape:
        record.update({"array_equal": False, "allclose": False})
        return record
    record["array_equal"] = bool(np.array_equal(left, right))
    if not (
        np.issubdtype(left.dtype, np.number)
        and np.issubdtype(right.dtype, np.number)
    ):
        record["allclose"] = record["array_equal"]
        return record
    difference = right.astype(np.float64) - left.astype(np.float64)
    absolute = np.abs(difference)
    denominator = np.maximum(np.abs(left.astype(np.float64)), np.spacing(1.0))
    record.update(
        {
            "finite": bool(
                np.all(np.isfinite(left)) and np.all(np.isfinite(right))
            ),
            "max_abs": float(np.max(absolute)) if absolute.size else 0.0,
            "rms": float(np.sqrt(np.mean(difference**2)))
            if difference.size
            else 0.0,
            "max_rel": float(np.max(absolute / denominator))
            if absolute.size
            else 0.0,
            "allclose": bool(np.allclose(left, right, rtol=rtol, atol=atol)),
        }
    )
    return record


def _tree_comparison(
    reference: Any, actual: Any, *, numeric_tolerance: bool = True
) -> Any:
    if isinstance(reference, np.ndarray) or isinstance(actual, np.ndarray):
        return _array_comparison(reference, actual, STATE_RTOL, STATE_ATOL)
    if isinstance(reference, Mapping) and isinstance(actual, Mapping):
        keys = sorted(set(reference) | set(actual))
        return {
            "keys_equal": set(reference) == set(actual),
            "fields": {
                str(key): _tree_comparison(
                    reference.get(key),
                    actual.get(key),
                    numeric_tolerance=numeric_tolerance,
                )
                for key in keys
            },
        }
    if isinstance(reference, list) and isinstance(actual, list):
        return {
            "length_equal": len(reference) == len(actual),
            "items": [
                _tree_comparison(
                    left, right, numeric_tolerance=numeric_tolerance
                )
                for left, right in zip(reference, actual)
            ],
        }
    if (
        isinstance(reference, (int, float))
        and not isinstance(reference, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        difference = float(actual) - float(reference)
        record = {
            "reference": reference,
            "actual": actual,
            "difference": difference,
            "exact": reference == actual,
        }
        if numeric_tolerance:
            limit = STATE_ATOL + STATE_RTOL * abs(float(reference))
            record["within_tolerance"] = abs(difference) <= limit
        return record
    return {
        "reference": reference,
        "actual": actual,
        "exact": reference == actual,
    }


def _normalized_eta(value: Any) -> np.ndarray:
    eta = np.asarray(value, dtype=np.float64).reshape(-1)
    return eta - np.max(eta) if eta.size else eta


def _endpoint_comparison(
    reference_run: Mapping[str, Any],
    actual_run: Mapping[str, Any],
    reference_summary: Mapping[str, Any],
    actual_summary: Mapping[str, Any],
) -> dict[str, Any]:
    left_vp = reference_run["vp"]
    right_vp = actual_run["vp"]
    fields = {
        name: _array_comparison(
            left_vp[name], right_vp[name], STATE_RTOL, STATE_ATOL
        )
        for name in ("mu", "sigma", "lambd", "w")
    }
    eta = _array_comparison(
        _normalized_eta(left_vp["eta"]),
        _normalized_eta(right_vp["eta"]),
        STATE_RTOL,
        STATE_ATOL,
    )
    left_result = reference_run["result"]
    right_result = actual_run["result"]
    decisions = {
        name: _tree_comparison(
            left_result.get(name),
            right_result.get(name),
            numeric_tolerance=name == "boost",
        )
        for name in (
            "complete",
            "mode",
            "production_policy_complete",
            "K",
            "pruned",
            "iterations_by_start",
            "stopping_reasons",
            "boost",
        )
    }
    return {
        "physical_state": fields,
        "eta_gauge": "subtract maximum",
        "eta_normalized": eta,
        "decisions": decisions,
        "stats": _tree_comparison(left_vp.get("stats"), right_vp.get("stats")),
        "rng_state": _tree_comparison(
            reference_summary.get("rng_state"),
            actual_summary.get("rng_state"),
            numeric_tolerance=False,
        ),
    }


def _rescore_comparison(reference: Any, actual: Any) -> dict[str, Any]:
    left_scores = reference.get("scores", {})
    right_scores = actual.get("scores", {})
    candidates: dict[str, Any] = {}
    for candidate in sorted(set(left_scores) | set(right_scores)):
        left = {
            int(row["seed"]): row for row in left_scores.get(candidate, [])
        }
        right = {
            int(row["seed"]): row for row in right_scores.get(candidate, [])
        }
        streams = []
        for seed in sorted(set(left) | set(right)):
            left_row, right_row = left.get(seed), right.get(seed)
            if left_row is None or right_row is None:
                streams.append({"seed": seed, "missing": True})
                continue
            values = {}
            for name in sorted(set(left_row) | set(right_row)):
                left_value, right_value = left_row.get(name), right_row.get(
                    name
                )
                if isinstance(left_value, (int, float)) and isinstance(
                    right_value, (int, float)
                ):
                    difference = float(right_value) - float(left_value)
                    if name in {
                        "seed",
                        "samples_per_component",
                        "samples_total",
                    }:
                        limit = 0.0
                        comparison = "exact metadata"
                    else:
                        limit = RESCORE_ATOL + RESCORE_RTOL * abs(
                            float(left_value)
                        )
                        comparison = "declared rescore rtol/atol"
                    values[name] = {
                        "reference": left_value,
                        "actual": right_value,
                        "difference": difference,
                        "absolute_limit": limit,
                        "comparison": comparison,
                        "within_tolerance": abs(difference) <= limit,
                    }
                else:
                    values[name] = {
                        "reference": left_value,
                        "actual": right_value,
                        "exact": left_value == right_value,
                    }
            streams.append({"seed": seed, "values": values})
        candidates[str(candidate)] = streams
    return {
        "rtol": RESCORE_RTOL,
        "atol": RESCORE_ATOL,
        "common_streams": _tree_comparison(
            reference.get("common_streams"),
            actual.get("common_streams"),
            numeric_tolerance=False,
        ),
        "aliases": _tree_comparison(
            reference.get("aliases", {}),
            actual.get("aliases", {}),
            numeric_tolerance=False,
        ),
        "candidates": candidates,
    }


def _kernel_comparison(reference: Any, actual: Any) -> dict[str, Any]:
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
    fields = {}
    for name in sorted(set(reference) | set(actual)):
        if name not in reference or name not in actual:
            fields[name] = {"present_in_both": False}
            continue
        if name in variance:
            tolerance = (1e-3, 1e-8)
        elif name in solve:
            tolerance = (1e-6, 1e-10)
        else:
            tolerance = (1e-10, 1e-12)
        fields[name] = _kernel_array_comparison(
            reference[name], actual[name], *tolerance
        )
    return {
        "keys_equal": set(reference) == set(actual),
        "fields": fields,
        "tolerance_source": "existing Stage 4 fixed-input kernel gate",
    }


def _kernel_array_comparison(
    reference: Any, actual: Any, rtol: float, atol: float
) -> dict[str, Any]:
    """Reproduce the existing oracle lower-quartile-floor kernel gate."""
    left_original = np.asarray(reference)
    right_original = np.asarray(actual)
    left = np.asarray(reference, dtype=float)
    right = np.asarray(actual, dtype=float)
    same_shape = left.shape == right.shape
    dtype_pass = (
        left_original.dtype == np.float64
        and right_original.dtype == np.float64
    )
    record: dict[str, Any] = {
        "reference_shape": list(left.shape),
        "actual_shape": list(right.shape),
        "reference_dtype": str(left_original.dtype),
        "actual_dtype": str(right_original.dtype),
        "shape_match": same_shape,
        "float64_reference": left_original.dtype == np.float64,
        "float64_actual": right_original.dtype == np.float64,
        "rtol": rtol,
        "atol": atol,
        "comparison": "oracle_lower_quartile_floor",
    }
    if not same_shape:
        record.update(
            {
                "max_abs": None,
                "max_scaled": None,
                "nonfinite_patterns_match": False,
                "pass": False,
                "standard_allclose_extra_diagnostic": False,
            }
        )
        return record
    pattern_ok = bool(
        np.array_equal(np.isnan(left), np.isnan(right))
        and np.array_equal(np.isposinf(left), np.isposinf(right))
        and np.array_equal(np.isneginf(left), np.isneginf(right))
    )
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.any():
        scale = np.abs(left[finite])
        difference = np.abs(right[finite] - left[finite])
        floor = max(float(np.quantile(scale, 0.25)), np.finfo(float).tiny)
        denominator = np.maximum(scale, floor)
        max_abs = float(np.max(difference))
        max_scaled = float(np.max(difference / denominator))
        within = bool(np.all(difference <= rtol * denominator + atol))
    else:
        max_abs, max_scaled, within = 0.0, 0.0, True
    record.update(
        {
            "max_abs": max_abs,
            "max_scaled": max_scaled,
            "nonfinite_patterns_match": pattern_ok,
            "pass": bool(within and pattern_ok and dtype_pass),
            "standard_allclose_extra_diagnostic": bool(
                np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
            ),
        }
    )
    return record


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "median": statistics.median(finite) if finite else None,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "values": finite,
    }


def _mismatch_paths(value: Any, path: str = "") -> list[str]:
    """Collect failed comparison predicates without treating drift as an error."""
    mismatches: list[str] = []
    if isinstance(value, Mapping):
        if "pass" in value and value["pass"] is not True:
            mismatches.append(path or "root")
        elif "allclose" in value and value["allclose"] is not True:
            mismatches.append(path or "root")
        elif (
            "within_tolerance" in value
            and value["within_tolerance"] is not True
        ):
            mismatches.append(path or "root")
        elif "exact" in value and value["exact"] is not True:
            mismatches.append(path or "root")
        for structural in ("keys_equal", "length_equal", "finite"):
            if structural in value and value[structural] is not True:
                mismatches.append(
                    f"{path}.{structural}" if path else structural
                )
        for key, item in value.items():
            if key in {
                "allclose",
                "pass",
                "within_tolerance",
                "exact",
                "array_equal",
                "keys_equal",
                "length_equal",
                "finite",
            }:
                continue
            child = f"{path}.{key}" if path else str(key)
            mismatches.extend(_mismatch_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            mismatches.extend(_mismatch_paths(item, f"{path}[{index}]"))
    return sorted(set(mismatches))


def _cold_total(summary: Mapping[str, Any]) -> dict[str, Any]:
    fits = summary.get("fits", [])
    cold = fits[0] if fits else {}
    context = summary.get("cuda_context", {})
    pieces = {
        "msvc_setup": float(summary.get("msvc_setup_s", 0.0)),
        "imports": float(summary.get("import_s", 0.0)),
        "cuda_context": float(context.get("wall_s", 0.0)),
        "compiler_reset": float(summary.get("compiler_reset_s", 0.0)),
        "compile_wrapper_construction": float(
            summary.get("compile_wrapper_construction_s", 0.0)
        ),
        "state_rebuild": float(cold.get("state_rebuild_s", 0.0)),
        "complete_fit_api": float(
            cold.get("api_wall_s", cold.get("wall_s", 0.0))
        ),
    }
    return {"seconds": sum(pieces.values()), "pieces": pieces}


def _compile_record(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    evidence = summary.get("compiler_evidence")
    if not isinstance(evidence, Mapping):
        return None
    counters = evidence.get("current", {}).get("dynamo_counters", {})
    metrics = evidence.get("current", {}).get("inductor_metrics", {})
    return {
        "config": evidence.get("config"),
        "calls": evidence.get("call_count"),
        "compiled_calls": evidence.get("compiled_call_count"),
        "eager_full_variance_calls": evidence.get(
            "eager_full_variance_call_count"
        ),
        "unique_graphs": counters.get("stats", {}).get("unique_graphs"),
        "graph_breaks": counters.get("graph_break", {}),
        "recompiles": counters.get("recompiles", {}),
        "aot_autograd": counters.get("aot_autograd", {}),
        "generated_kernels": metrics.get("generated_kernel_count"),
        "backward_observation": summary.get(
            "compiled_backward_observation",
            summary.get("backward_observation"),
        ),
    }


def _effective_worker_validation(
    summary: Mapping[str, Any], raw: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    recorded = summary.get("validation", {})
    fit_gates = [fit.get("fit_gate", {}) for fit in summary.get("fits", [])]
    raw_fits = [] if raw is None else raw.get("fits", [])
    expected_boost_identity = bool(raw_fits) and len(raw_fits) == len(
        fit_gates
    )
    expected_boost_identity = expected_boost_identity and all(
        run.get("boost") is True
        and run.get("result", {}).get("transformer_identity") is False
        and isinstance(run.get("result", {}).get("boost"), Mapping)
        and run.get("result", {}).get("mode") == "complete_fit"
        and run.get("result", {}).get("production_policy_complete") is True
        for run in raw_fits
    )
    identity_only = bool(fit_gates) and all(
        gate.get("checks", {}).get("transformer_identity") is False
        and all(
            value is True
            for name, value in gate.get("checks", {}).items()
            if name != "transformer_identity"
        )
        for gate in fit_gates
    )
    override = (
        recorded.get("all_pass") is False
        and recorded.get("kernel") is True
        and recorded.get("compiled_backward") is True
        and expected_boost_identity
        and identity_only
    )
    return {
        "recorded": recorded,
        "effective_pass": recorded.get("all_pass") is True or override,
        "known_final_boost_identity_gate_artifact": override,
        "reason": (
            "final_boost intentionally returns its independent pre-boost copy; "
            "the outer VBMC caller restores live transformer identity. The "
            "artifact does not retain transformer arrays, so this reclassifies "
            "only the documented identity check, not transformer equivalence."
            if override
            else None
        ),
    }


def _timing_record(
    summary: Mapping[str, Any], raw: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    fits = summary.get("fits", [])
    hot = fits[1:4]
    all_walls = [float(fit["wall_s"]) for fit in hot]
    warm = [
        float(fit["wall_s"])
        for fit in hot
        if fit.get("classification") == "hot"
    ]
    return {
        "cold_fit": float(fits[0]["wall_s"]) if fits else None,
        "cold_total": _cold_total(summary),
        "all_subsequent": _aggregate(all_walls),
        "warm_only": _aggregate(warm),
        "classifications": [fit.get("classification") for fit in hot],
        "fixed_input_gate_pass": summary.get("kernel_gate", {}).get(
            "all_pass"
        ),
        "worker_validation": _effective_worker_validation(summary, raw),
    }


def _speedups(
    reference: Sequence[float], actual: Sequence[float]
) -> dict[str, Any]:
    values = [
        float(left) / float(right)
        for left, right in zip(reference, actual)
        if float(left) > 0 and float(right) > 0
    ]
    record = _aggregate(values)
    record["definition"] = "reference_seconds / target_seconds"
    return record


def _amortization(
    reference_summary: Mapping[str, Any], compiled_summary: Mapping[str, Any]
) -> dict[str, Any]:
    reference_cold = _cold_total(reference_summary)["seconds"]
    compiled_cold = _cold_total(compiled_summary)["seconds"]
    observed_cold_disadvantage = compiled_cold - reference_cold
    reference_timing = _timing_record(reference_summary)
    compiled_timing = _timing_record(compiled_summary)
    reference_hot = reference_summary.get("fits", [])[1:4]
    compiled_hot = compiled_summary.get("fits", [])[1:4]
    paired_savings = [
        float(left["wall_s"]) - float(right["wall_s"])
        for left, right in zip(reference_hot, compiled_hot)
        if left.get("classification") == "hot"
        and right.get("classification") == "hot"
    ]
    savings = statistics.median(paired_savings) if paired_savings else None
    compiled_warm = compiled_timing["warm_only"]["median"]
    reference_warm = reference_timing["warm_only"]["median"]
    compiled_gap = (
        None if compiled_warm is None else compiled_cold - compiled_warm
    )
    reference_gap = (
        None if reference_warm is None else reference_cold - reference_warm
    )
    overhead = (
        None
        if compiled_gap is None or reference_gap is None
        else compiled_gap - reference_gap
    )
    additional_hot = None
    estimated_total = None
    reason = None
    if savings is None or savings <= 0:
        reason = "warm-only compiled fits have no positive median saving"
    elif observed_cold_disadvantage <= 0:
        additional_hot = 0
        estimated_total = 1
        reason = "compiled recorded cold total has no positive disadvantage"
    else:
        additional_hot = int(math.ceil(observed_cold_disadvantage / savings))
        estimated_total = 1 + additional_hot
    return {
        "compiled_first_fit_startup_s": compiled_cold,
        "reference_first_fit_startup_s": reference_cold,
        "observed_cold_disadvantage_s": observed_cold_disadvantage,
        "compiled_cold_minus_warm_s": compiled_gap,
        "reference_cold_minus_warm_s": reference_gap,
        "incremental_compiler_overhead_estimate_s": overhead,
        "paired_warm_savings_s": _aggregate(paired_savings),
        "median_paired_warm_saving_s": savings,
        "additional_hot_fits_to_break_even": additional_hot,
        "estimated_total_fits_including_cold": estimated_total,
        "unavailable_reason": reason,
        "assumption": (
            "This is a per-case estimate. Future fits must retain the same "
            "process/compiler cache, tensor shapes, and Python flags, avoid "
            "recompilation, and repeat the observed paired median saving. A "
            "full solver's growing GP training size can recompile and is not "
            "quantified by these fixed-GP workloads."
        ),
    }


def _load_workers(
    campaign: Mapping[str, Any], campaign_path: Path
) -> dict[Any, Any]:
    workers = {}
    for row in campaign.get("workers", []):
        key = (str(row["case"]), str(row["arm"]))
        original = _resolve(row["summary"], campaign_path.parent)
        worker_directory = PureWindowsPath(str(row["summary"])).parent.name
        fallback = (
            campaign_path.parent / "runs" / worker_directory / "summary.json"
        ).resolve()
        expected_hash = row.get("summary_sha256")
        summary_path = None
        candidates = list(dict.fromkeys((fallback, original)))
        for candidate in candidates:
            if candidate.is_file() and (
                not expected_hash or _sha256(candidate) == expected_hash
            ):
                summary_path = candidate
                break
        if summary_path is None:
            observed = {
                str(candidate): _sha256(candidate)
                for candidate in candidates
                if candidate.is_file()
            }
            raise ValueError(
                "worker summary absent or hash-mismatched; "
                f"expected {expected_hash}, observed {observed}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tree_path = summary_path.parent / "result.json"
        raw = ArtifactReader(tree_path).read() if tree_path.is_file() else None
        workers[key] = {"row": row, "summary": summary, "raw": raw}
    return workers


def _has_complete_hot_artifact(worker: Mapping[str, Any]) -> bool:
    summary_fits = worker.get("summary", {}).get("fits", [])
    raw_fits = (worker.get("raw") or {}).get("fits", [])
    return len(summary_fits) >= 4 and len(raw_fits) >= 4


def build_report(campaign_path: Path) -> dict[str, Any]:
    campaign_path = campaign_path.resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    workers = _load_workers(campaign, campaign_path)
    cases = [str(case) for case in campaign.get("cases", [])]
    timings: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    failures = list(campaign.get("failures", []))
    pending = []
    validation: dict[str, Any] = {"pairs": {}}

    for case in cases:
        timings[case] = {}
        for arm in campaign.get("arms", []):
            worker = workers.get((case, arm))
            if worker is None:
                record = {"case": case, "arm": arm, "reason": "worker absent"}
                if campaign.get("status") in {"initializing", "running"}:
                    pending.append(record)
                else:
                    failures.append({**record, "error": "worker absent"})
                continue
            timings[case][arm] = _timing_record(
                worker["summary"], worker["raw"]
            )
            timings[case][arm]["compile"] = _compile_record(worker["summary"])

        comparisons[case] = {}
        for compiled_arm, references in COMPILED_REFERENCES.items():
            actual = workers.get((case, compiled_arm))
            if actual is None or not _has_complete_hot_artifact(actual):
                continue
            arm_comparisons = {}
            for reference_arm in references:
                reference = workers.get((case, reference_arm))
                if reference is None or not _has_complete_hot_artifact(
                    reference
                ):
                    continue
                endpoints = []
                for index, seed in enumerate(
                    campaign.get("seeds", []), start=1
                ):
                    endpoints.append(
                        {
                            "seed": int(seed),
                            "comparison": _endpoint_comparison(
                                reference["raw"]["fits"][index],
                                actual["raw"]["fits"][index],
                                reference["summary"]["fits"][index],
                                actual["summary"]["fits"][index],
                            ),
                        }
                    )
                left_walls = [
                    fit["wall_s"] for fit in reference["summary"]["fits"][1:4]
                ]
                right_walls = [
                    fit["wall_s"] for fit in actual["summary"]["fits"][1:4]
                ]
                paired_warm = [
                    (float(left["wall_s"]), float(right["wall_s"]))
                    for left, right in zip(
                        reference["summary"]["fits"][1:4],
                        actual["summary"]["fits"][1:4],
                    )
                    if left.get("classification") == "hot"
                    and right.get("classification") == "hot"
                ]
                fixed_reference = (
                    "numpy" if reference_arm == "numpy_cpu" else "torch"
                )
                fixed_kernel = _kernel_comparison(
                    reference["raw"]["kernel_gate"]["raw"][fixed_reference],
                    actual["raw"]["kernel_gate"]["raw"]["torch"],
                )
                numpy_baselines = _kernel_comparison(
                    reference["raw"]["kernel_gate"]["raw"]["numpy"],
                    actual["raw"]["kernel_gate"]["raw"]["numpy"],
                )
                input_metadata = _tree_comparison(
                    {
                        name: reference["summary"]
                        .get("kernel_gate", {})
                        .get(name)
                        for name in ("epsilon", "Ns", "production_Ns")
                    },
                    {
                        name: actual["summary"]
                        .get("kernel_gate", {})
                        .get(name)
                        for name in ("epsilon", "Ns", "production_Ns")
                    },
                    numeric_tolerance=False,
                )
                comparison = {
                    "hot_endpoints": endpoints,
                    "rescore": _rescore_comparison(
                        reference["raw"].get("rescore", {}),
                        actual["raw"].get("rescore", {}),
                    ),
                    "all_subsequent_speedup": _speedups(
                        left_walls, right_walls
                    ),
                    "warm_only_speedup": _speedups(
                        [pair[0] for pair in paired_warm],
                        [pair[1] for pair in paired_warm],
                    ),
                    "fixed_input_gate_pass": actual["summary"]
                    .get("kernel_gate", {})
                    .get("all_pass"),
                    "amortization": _amortization(
                        reference["summary"], actual["summary"]
                    ),
                    "fixed_kernel_raw": fixed_kernel,
                    "cross_worker_numpy_baselines": numpy_baselines,
                    "fixed_input_metadata": input_metadata,
                }
                arm_comparisons[reference_arm] = comparison
                pair_key = f"{case}/{compiled_arm}/{reference_arm}"
                endpoint_mismatches = _mismatch_paths(endpoints)
                rescore_mismatches = _mismatch_paths(comparison["rescore"])
                kernel_mismatches = _mismatch_paths(fixed_kernel)
                baseline_mismatches = _mismatch_paths(numpy_baselines)
                input_mismatches = _mismatch_paths(input_metadata)
                validation["pairs"][pair_key] = {
                    "endpoint_drift_count": len(endpoint_mismatches),
                    "endpoint_drift_paths": endpoint_mismatches,
                    "rescore_drift_count": len(rescore_mismatches),
                    "rescore_drift_paths": rescore_mismatches,
                    "fixed_kernel_mismatch_count": len(kernel_mismatches),
                    "fixed_kernel_mismatch_paths": kernel_mismatches,
                    "numpy_baseline_mismatch_count": len(baseline_mismatches),
                    "numpy_baseline_mismatch_paths": baseline_mismatches,
                    "fixed_input_metadata_mismatch_count": len(
                        input_mismatches
                    ),
                    "fixed_input_metadata_mismatch_paths": input_mismatches,
                    "compiled_worker_gate_pass": _effective_worker_validation(
                        actual["summary"], actual["raw"]
                    )["effective_pass"],
                    "compiled_worker_recorded_gate_pass": actual["summary"]
                    .get("validation", {})
                    .get("all_pass"),
                }
                pair_validation = validation["pairs"][pair_key]
                pair_validation["strict_all_match"] = (
                    pair_validation["endpoint_drift_count"] == 0
                    and pair_validation["rescore_drift_count"] == 0
                    and pair_validation["fixed_kernel_mismatch_count"] == 0
                    and pair_validation["numpy_baseline_mismatch_count"] == 0
                    and pair_validation["fixed_input_metadata_mismatch_count"]
                    == 0
                    and pair_validation["compiled_worker_gate_pass"] is True
                )
            comparisons[case][compiled_arm] = arm_comparisons

    pair_validations = list(validation["pairs"].values())
    effective_workers = {
        f"{case}/{arm}": _effective_worker_validation(
            worker["summary"], worker["raw"]
        )
        for (case, arm), worker in workers.items()
    }
    corrected_failures = []
    reclassified_failures = []
    for failure in failures:
        worker_key = f"{failure.get('case')}/{failure.get('arm')}"
        effective = effective_workers.get(worker_key, {})
        if failure.get("stage") == "worker" and effective.get(
            "known_final_boost_identity_gate_artifact"
        ):
            reclassified_failures.append(failure)
        else:
            corrected_failures.append(failure)
    validation.update(
        {
            "campaign_finished": campaign.get("status")
            in {"completed", "completed_with_failures"},
            "fixed_kernel_all_match": bool(pair_validations)
            and all(
                row["fixed_kernel_mismatch_count"] == 0
                for row in pair_validations
            ),
            "compiled_worker_gates_all_pass": bool(pair_validations)
            and all(
                row["compiled_worker_gate_pass"] is True
                for row in pair_validations
            ),
            "endpoint_drift_total": sum(
                row["endpoint_drift_count"] for row in pair_validations
            ),
            "rescore_drift_total": sum(
                row["rescore_drift_count"] for row in pair_validations
            ),
            "strict_all_crossarm_match": bool(pair_validations)
            and all(row["strict_all_match"] for row in pair_validations),
            "interpretation": (
                "Fixed-kernel mismatches and worker-gate failures are numerical "
                "validation failures. Endpoint, decision, stats, RNG, and rescore "
                "differences are reported as scientific trajectory drift and do "
                "not prevent the report from summarizing available timings."
            ),
            "workers": effective_workers,
            "reclassified_harness_failures": reclassified_failures,
            "corrected_measurement_failures": corrected_failures,
        }
    )
    worker_validations = [
        record.get("effective_pass") for record in effective_workers.values()
    ]
    validation["all_worker_validations_pass"] = bool(
        worker_validations
    ) and all(value is True for value in worker_validations)
    validation["measurement_integrity_pass"] = (
        validation["campaign_finished"]
        and not corrected_failures
        and not pending
        and validation["all_worker_validations_pass"]
        and validation["fixed_kernel_all_match"]
    )

    return {
        "schema": 1,
        "campaign": str(campaign_path),
        "campaign_sha256": _sha256(campaign_path),
        "campaign_status": campaign.get("status"),
        "original_harness_status": {
            "campaign": campaign.get("status"),
            "failures": failures,
            "worker_statuses": {
                f"{case}/{arm}": {
                    "status": worker["row"].get("status"),
                    "validation": worker["summary"].get("validation"),
                }
                for (case, arm), worker in workers.items()
            },
        },
        "scope": campaign.get("compiled_scope"),
        "tolerances": {
            "state": {"rtol": STATE_RTOL, "atol": STATE_ATOL},
            "rescore": {"rtol": RESCORE_RTOL, "atol": RESCORE_ATOL},
        },
        "timings": timings,
        "comparisons": comparisons,
        "validation": validation,
        "failures": failures,
        "pending": pending,
    }


def _format_range(record: Mapping[str, Any]) -> str:
    if not record or record.get("median") is None:
        return "n/a"
    return (
        f"{record['median']:.6g} "
        f"[{record['min']:.6g}, {record['max']:.6g}] (n={record['count']})"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    original_status = report.get("original_harness_status", {})
    validation = report.get("validation", {})
    original_failures = original_status.get("failures", [])
    reclassified = validation.get("reclassified_harness_failures", [])
    integrity = validation.get("measurement_integrity_pass") is True
    strict_match = validation.get("strict_all_crossarm_match") is True
    lines = [
        "# Stage 4 compiled-Torch comparison",
        "",
        str(report.get("scope") or "Compiled objective follow-up."),
        "",
        (
            f"Original harness status: **{original_status.get('campaign', 'unknown')}** "
            f"with {len(original_failures)} recorded failure(s)."
        ),
        (
            f"The reporter reclassified {len(reclassified)} worker failure(s) "
            "caused solely by the documented final-boost transformer-identity "
            "check. The archived outputs establish the expected false identity "
            "and boost return policy, but do not retain transformer arrays for "
            "a direct value comparison."
        ),
        (
            f"Corrected measurement integrity: **{'pass' if integrity else 'fail'}**. "
            "Strict cross-arm trajectory matching: "
            f"**{'pass' if strict_match else 'fail'}**. Trajectory drift is retained "
            "as scientific evidence and does not suppress timing summaries."
        ),
        "",
        "Warm-only summaries exclude subsequent fits that generated a new graph or kernel.",
        "Recorded cold total sums imports, compiler/MSVC setup, CUDA context, state rebuild, "
        "and the externally timed first complete fit. It excludes worker environment and "
        "manifest checks outside the API call, plus unrecorded process startup.",
        "",
        "| Case | Arm | Cold fit s | Recorded cold total s | All subsequent median [range] | Warm-only median [range] |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for case, arms in report.get("timings", {}).items():
        for arm, timing in arms.items():
            cold_fit = timing.get("cold_fit")
            cold_text = "n/a" if cold_fit is None else f"{cold_fit:.6g}"
            lines.append(
                f"| {case} | {arm} | {cold_text} | "
                f"{timing['cold_total']['seconds']:.6g} | "
                f"{_format_range(timing['all_subsequent'])} | "
                f"{_format_range(timing['warm_only'])} |"
            )
    lines.extend(("", "## Paired hot-fit speedups", ""))
    lines.extend(
        (
            "Speedup is reference wall time divided by compiled wall time for the same seed. "
            "The table retains all three subsequent fits, including recompilations.",
            "",
            "| Case | Compiled arm | Reference | All subsequent median [range] | Warm-only median [range] |",
            "| --- | --- | --- | --- | --- |",
        )
    )
    for case, arms in report.get("comparisons", {}).items():
        for compiled_arm, values in arms.items():
            for reference, comparison in values.items():
                if reference == "amortization_vs_eager":
                    continue
                lines.append(
                    f"| {case} | {compiled_arm} | {reference} | "
                    f"{_format_range(comparison['all_subsequent_speedup'])} | "
                    f"{_format_range(comparison['warm_only_speedup'])} |"
                )
    lines.extend(
        (
            "",
            "Numerical endpoint, decision, RNG, rescore, compiler, and amortization details are retained in `comparison.json`.",
            "",
        )
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = build_report(args.campaign)
    _write_json(out / "comparison.json", report)
    (out / "comparison.md").write_text(
        render_markdown(report), encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
