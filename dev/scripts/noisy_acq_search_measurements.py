"""Versioned E3 timing adapter for frozen search selections.

The numerical selection and paired-timing operations remain in the frozen
E3 modules.  This adapter projects their completed return values to compact,
strict-JSON provenance only after ``time_pair`` has stopped every clock.  It
writes to a separate namespace and never repairs or overwrites v1 artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any

import noisy_acq_experiment as capture
import noisy_acq_integration as integration
import noisy_acq_search_experiment as campaign
import noisy_acq_timing as timing
import numpy as np

SCHEMA_VERSION = 2
MANIFEST_KIND = "noisy_acq_search_timing_measurement_manifest_v2"
OUTPUT_NAMESPACE = "search_timing_v2"
EXPECTED_TIMING_CELLS = 36
EXPECTED_ALLOCATION_CELLS = 72
EXPECTED_SELECTION_TERMINALS = 384
FAILED_V1_CELL = {
    "state_id": "logreg_D5_noise3_seed0_late",
    "timing_seed": 15108013424671789913,
    "treatment_tag": "S3",
}
ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path | str) -> str:
    return integration.sha256_file(path)


def _adapter_sha256() -> str:
    return _sha256(Path(__file__).resolve())


def _as_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _optional_count(result: dict[str, Any], name: str) -> int | None:
    if name not in result:
        return None
    return _as_int(name, result[name])


def _optional_reason(result: dict[str, Any], name: str) -> str | None:
    value = result.get(name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


def _nullable_cache_index(value: Any) -> int | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("cache_index must be scalar")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError("cache_index must be integral or null")
    if isinstance(scalar, (float, np.floating)) and math.isnan(float(scalar)):
        return None
    number = float(scalar)
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError("cache_index must be a nonnegative integer or NaN")
    return int(number)


def _selected_row_id(value: Any, dimension: int) -> str:
    selected = np.asarray(value, dtype=np.float64)
    if selected.shape != (1, dimension):
        raise ValueError("selected must have shape (1, dimension)")
    if not np.all(np.isfinite(selected)):
        raise ValueError("selected must contain only finite coordinates")
    row = np.ascontiguousarray(selected.reshape(-1), dtype="<f8")
    return "x_" + hashlib.sha256(row.tobytes()).hexdigest()


def project_selection_result(
    result: dict[str, Any], *, dimension: int
) -> dict[str, Any]:
    """Return validated compact provenance without mutating ``result``."""

    if not isinstance(result, dict):
        raise ValueError("selection result must be a mapping")
    arm = result.get("arm")
    if arm not in {"S0", "S1", "S2", "S3"}:
        raise ValueError("selection result has an invalid arm")
    if result.get("target_called") is not False:
        raise ValueError("timed selection must stop before a target call")
    projected = {
        "arm": arm,
        "selected_row_id": _selected_row_id(result.get("selected"), dimension),
        "search_seed": _as_int("search_seed", result.get("search_seed")),
        "accurate_seed": _as_int("accurate_seed", result.get("accurate_seed")),
        "target_called": False,
        "cache_index": _nullable_cache_index(result.get("cache_index")),
    }
    for name in (
        "generated_count",
        "coarse_candidate_rows",
        "production_candidate_rows",
        "accurate_candidate_rows",
        "local_iterations",
    ):
        value = _optional_count(result, name)
        if value is not None:
            projected[name] = value
    for name in ("fallback_reason", "refinement_stop_reason"):
        if name in result:
            projected[name] = _optional_reason(result, name)
    if "elapsed_seconds" in result:
        elapsed = float(result["elapsed_seconds"])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("elapsed_seconds must be finite and positive")
        projected["operation_elapsed_seconds"] = elapsed
    return projected


def project_timing_report(
    report: dict[str, Any], *, dimension: int
) -> dict[str, Any]:
    """Replace only completed selection results in a copied timing report."""

    projected = copy.deepcopy(report)
    rounds = projected.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 7:
        raise ValueError("paired timing must contain seven rounds")
    for row in rounds:
        if not isinstance(row, dict):
            raise ValueError("paired timing round must be a mapping")
        for role in ("baseline", "treatment"):
            measurement = row.get(role)
            if (
                not isinstance(measurement, dict)
                or "result" not in measurement
            ):
                raise ValueError(f"paired timing round lacks {role} result")
            measurement["result"] = project_selection_result(
                measurement["result"], dimension=dimension
            )
    # This is intentionally a strict check of the allowlisted projection,
    # rather than a generic sanitizer for arbitrary numerical return values.
    json.dumps(projected, allow_nan=False)
    return projected


def project_allocation_report(
    report: dict[str, Any], *, dimension: int
) -> dict[str, Any]:
    """Project one allocation result after frozen measurement completes."""

    projected = copy.deepcopy(report)
    if "result" not in projected:
        raise ValueError("allocation measurement lacks a selection result")
    projected["result"] = project_selection_result(
        projected["result"], dimension=dimension
    )
    json.dumps(projected, allow_nan=False)
    return projected


def _validate_timing_projection(
    report: dict[str, Any],
    *,
    timing_seed: int,
    baseline_arm: str,
    treatment_arm: str,
) -> None:
    if report.get("seed") != timing_seed:
        raise RuntimeError("paired timing returned the wrong root seed")
    for row in report["rounds"]:
        paired_seed = row.get("paired_seed")
        for role, expected_arm in (
            ("baseline", baseline_arm),
            ("treatment", treatment_arm),
        ):
            measurement = row[role]
            result = measurement["result"]
            if (
                measurement.get("seed") != paired_seed
                or result.get("search_seed") != paired_seed
            ):
                raise RuntimeError("paired timing search seed changed")
            if result.get("arm") != expected_arm:
                raise RuntimeError("paired timing returned the wrong arm")


def _validate_allocation_projection(
    report: dict[str, Any], *, allocation_seed: int, expected_arm: str
) -> None:
    if (
        report.get("seed") != allocation_seed
        or report["result"].get("search_seed") != allocation_seed
    ):
        raise RuntimeError("allocation measurement seed changed")
    if report["result"].get("arm") != expected_arm:
        raise RuntimeError("allocation measurement returned the wrong arm")


def _load_json(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


def _load_hashed_json(path: Path | str) -> tuple[dict[str, Any], str]:
    """Parse a JSON object and hash the exact bytes that were parsed."""

    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _snapshot_contract(
    snapshot: dict[str, Any],
    parent_sha256: str,
    parent: dict[str, Any],
) -> None:
    if (
        snapshot.get("kind")
        != "noisy_acq_search_selection_completion_snapshot"
    ):
        raise RuntimeError("invalid selection completion snapshot")
    if snapshot.get("manifest_sha256") != parent_sha256:
        raise RuntimeError("selection snapshot belongs to another manifest")
    counts = snapshot.get("counts", {})
    if counts != {
        "expected": EXPECTED_SELECTION_TERMINALS,
        "failed": 0,
        "missing": 0,
        "succeeded": EXPECTED_SELECTION_TERMINALS,
    }:
        raise RuntimeError(
            "selection snapshot is not the complete 384-cell set"
        )
    terminals = snapshot.get("terminals", [])
    tags = [item.get("tag") for item in terminals if isinstance(item, dict)]
    if len(tags) != EXPECTED_SELECTION_TERMINALS or len(set(tags)) != len(
        tags
    ):
        raise RuntimeError("selection snapshot terminal inventory is invalid")
    expected_tags = {
        campaign.cell_tag(cell) for cell in parent.get("cells", [])
    }
    if set(tags) != expected_tags:
        raise RuntimeError("selection snapshot does not match parent cells")


def _failed_v1_contract(
    terminal_path: Path,
    failure_path: Path,
    parent: dict[str, Any],
) -> None:
    terminal = _load_json(terminal_path)
    failure = _load_json(failure_path)
    if terminal.get("status") != "failed" or failure.get("status") != "failed":
        raise RuntimeError("bound v1 pilot artifacts must both be failures")
    if (
        terminal.get("cell") != FAILED_V1_CELL
        or failure.get("cell") != FAILED_V1_CELL
    ):
        raise RuntimeError("bound v1 failure is not the frozen timing pilot")
    if FAILED_V1_CELL not in parent.get("timing_cells", []):
        raise RuntimeError("frozen failed pilot is outside parent allocation")
    if terminal.get("manifest_sha256") != integration.manifest_digest(parent):
        raise RuntimeError("bound v1 terminal belongs to another manifest")
    if terminal.get("hashes") != {"failure": _sha256(failure_path)}:
        raise RuntimeError(
            "bound v1 terminal does not authenticate its failure"
        )


def _gpyreg_contract(identity: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(
        identity["integration_identity"]["capture_identity"]["gpyreg"]
    )


def prepare_manifest(
    parent_manifest_path: Path,
    selection_snapshot_path: Path,
    old_failed_terminal: Path,
    old_failed_artifact: Path,
) -> dict[str, Any]:
    """Build an unready v2 allocation after the adapter source is frozen."""

    parent_manifest_path = parent_manifest_path.resolve()
    selection_snapshot_path = selection_snapshot_path.resolve()
    old_failed_terminal = old_failed_terminal.resolve()
    old_failed_artifact = old_failed_artifact.resolve()
    parent = _load_json(parent_manifest_path)
    campaign.validate_manifest(parent, require_ready=True)
    runtime = campaign.runtime_identity(parent)
    parent_sha256 = _sha256(parent_manifest_path)
    snapshot = _load_json(selection_snapshot_path)
    _snapshot_contract(snapshot, parent_sha256, parent)
    if len(parent.get("timing_cells", [])) != EXPECTED_TIMING_CELLS:
        raise RuntimeError("parent does not contain exactly 36 timing cells")
    if len(parent.get("allocation_cells", [])) != EXPECTED_ALLOCATION_CELLS:
        raise RuntimeError(
            "parent does not contain exactly 72 allocation cells"
        )
    _failed_v1_contract(old_failed_terminal, old_failed_artifact, parent)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "launch_ready": False,
        "quality_claim": False,
        "parent_manifest": str(parent_manifest_path),
        "parent_manifest_sha256": parent_sha256,
        "parent_runtime_identity": runtime,
        "parent_source_hashes": copy.deepcopy(
            parent["identity"]["search_source_hashes"]
        ),
        "selection_snapshot": str(selection_snapshot_path),
        "selection_snapshot_sha256": _sha256(selection_snapshot_path),
        "selection_terminal_count": EXPECTED_SELECTION_TERMINALS,
        "timing_cells": copy.deepcopy(parent["timing_cells"]),
        "timing_cell_count": EXPECTED_TIMING_CELLS,
        "allocation_cells": copy.deepcopy(parent["allocation_cells"]),
        "allocation_cell_count": EXPECTED_ALLOCATION_CELLS,
        "thread_environment": copy.deepcopy(parent["thread_environment"]),
        "gpyreg": _gpyreg_contract(runtime),
        "adapter": {
            "path": str(Path(__file__).resolve()),
            "sha256": _adapter_sha256(),
        },
        "source_transition": {
            "scope": "measurement_reporting_only",
            "numerical_sources_unchanged": True,
        },
        "output_namespace": OUTPUT_NAMESPACE,
        "old_failed_v1_pilot": {
            "terminal": str(old_failed_terminal),
            "terminal_sha256": _sha256(old_failed_terminal),
            "failure": str(old_failed_artifact),
            "failure_sha256": _sha256(old_failed_artifact),
        },
        "memory_protocol": {
            "separate_from_timing": True,
            "cells_predeclared": True,
            "operational_launch_requires_separate_allocation": True,
        },
    }


def validate_manifest(
    manifest: dict[str, Any], *, require_ready: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every parent, source, snapshot, and failed-pilot binding."""

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported timing measurement schema")
    if manifest.get("kind") != MANIFEST_KIND:
        raise RuntimeError("not an E3 timing measurement manifest")
    if require_ready and manifest.get("launch_ready") is not True:
        raise RuntimeError("timing measurement manifest is not launch_ready")
    if manifest.get("output_namespace") != OUTPUT_NAMESPACE:
        raise RuntimeError("timing output namespace changed")
    if manifest.get("source_transition") != {
        "scope": "measurement_reporting_only",
        "numerical_sources_unchanged": True,
    }:
        raise RuntimeError("invalid measurement-only source transition")
    adapter = manifest.get("adapter", {})
    if Path(adapter.get("path", "")).resolve() != Path(__file__).resolve() or (
        adapter.get("sha256") != _adapter_sha256()
    ):
        raise RuntimeError("timing measurement adapter changed")

    parent_path = Path(manifest["parent_manifest"])
    if _sha256(parent_path) != manifest.get("parent_manifest_sha256"):
        raise RuntimeError("parent search manifest changed")
    parent = _load_json(parent_path)
    campaign.validate_manifest(parent, require_ready=True)
    runtime = campaign.runtime_identity(parent)
    if capture.canonical(runtime) != capture.canonical(
        manifest.get("parent_runtime_identity")
    ):
        raise RuntimeError("parent runtime identity changed")
    if parent["identity"]["search_source_hashes"] != manifest.get(
        "parent_source_hashes"
    ):
        raise RuntimeError("frozen parent source hashes changed")
    if _gpyreg_contract(runtime) != manifest.get("gpyreg"):
        raise RuntimeError("gpyreg identity changed")
    if parent["thread_environment"] != manifest.get("thread_environment"):
        raise RuntimeError("thread contract changed")
    if (
        parent.get("timing_cells") != manifest.get("timing_cells")
        or len(manifest.get("timing_cells", [])) != EXPECTED_TIMING_CELLS
    ):
        raise RuntimeError("frozen 36-cell timing allocation changed")
    tags = [campaign.timing_tag(cell) for cell in manifest["timing_cells"]]
    if len(set(tags)) != EXPECTED_TIMING_CELLS:
        raise RuntimeError("timing allocation has duplicate tags")
    if parent.get("allocation_cells") != manifest.get(
        "allocation_cells"
    ) or len(manifest.get("allocation_cells", [])) != (
        EXPECTED_ALLOCATION_CELLS
    ):
        raise RuntimeError("frozen 72-cell allocation measurement changed")
    allocation_tags = [
        campaign.allocation_tag(cell) for cell in manifest["allocation_cells"]
    ]
    if len(set(allocation_tags)) != EXPECTED_ALLOCATION_CELLS:
        raise RuntimeError("allocation measurement has duplicate tags")
    memory = manifest.get("memory_protocol", {})
    if memory != {
        "separate_from_timing": True,
        "cells_predeclared": True,
        "operational_launch_requires_separate_allocation": True,
    }:
        raise RuntimeError("separate allocation-measurement protocol changed")

    snapshot_path = Path(manifest["selection_snapshot"])
    if _sha256(snapshot_path) != manifest.get("selection_snapshot_sha256"):
        raise RuntimeError("selection completion snapshot changed")
    snapshot = _load_json(snapshot_path)
    _snapshot_contract(snapshot, manifest["parent_manifest_sha256"], parent)
    if manifest.get("selection_terminal_count") != (
        EXPECTED_SELECTION_TERMINALS
    ):
        raise RuntimeError("selection terminal count changed")

    failed = manifest.get("old_failed_v1_pilot", {})
    for role in ("terminal", "failure"):
        path = Path(failed.get(role, ""))
        if not path.is_file() or _sha256(path) != failed.get(f"{role}_sha256"):
            raise RuntimeError(f"old failed v1 {role} changed")
        if _load_json(path).get("status") != "failed":
            raise RuntimeError(f"old v1 {role} is not a failed artifact")
    _failed_v1_contract(
        Path(failed["terminal"]), Path(failed["failure"]), parent
    )
    return parent, runtime


def _paths(out: Path, tag: str, kind: str) -> dict[str, Path]:
    if kind not in {"timing", "allocation"}:
        raise ValueError("unknown v2 measurement kind")
    root = out / OUTPUT_NAMESPACE
    return {
        "payload": root / kind / f"{tag}.json",
        "failure": root / "records" / f"{tag}.{kind}.failure.json",
        "complete": root / "records" / f"{tag}.{kind}.complete.json",
    }


def _temporary_paths(paths: dict[str, Path]) -> list[Path]:
    return [path.with_name(path.name + ".tmp") for path in paths.values()]


def _terminal_basis(
    manifest_sha256: str, manifest: dict[str, Any], cell: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_manifest_sha256": manifest_sha256,
        "parent_manifest_sha256": manifest["parent_manifest_sha256"],
        "adapter_sha256": manifest["adapter"]["sha256"],
        "cell": cell,
    }


def _existing_terminal(
    manifest_sha256: str,
    manifest: dict[str, Any],
    paths: dict[str, Path],
    cell: dict[str, Any],
) -> dict[str, Any] | None:
    temporary = [path for path in _temporary_paths(paths) if path.exists()]
    if temporary:
        raise RuntimeError("partial v2 temporary artifact requires inspection")
    if not paths["complete"].exists():
        partial = [
            paths[role]
            for role in ("payload", "failure")
            if paths[role].exists()
        ]
        if partial:
            raise RuntimeError(
                "partial or foreign v2 output requires inspection"
            )
        return None
    terminal = _load_json(paths["complete"])
    basis = _terminal_basis(manifest_sha256, manifest, cell)
    if any(terminal.get(key) != value for key, value in basis.items()):
        raise RuntimeError("completed v2 cell has foreign identity")
    status = terminal.get("status")
    expected_role = "payload" if status == "succeeded" else "failure"
    if status not in {"succeeded", "failed"} or set(
        terminal.get("hashes", {})
    ) != {expected_role}:
        raise RuntimeError("completed v2 cell has invalid status or artifacts")
    artifact = paths[expected_role]
    if (
        not artifact.is_file()
        or _sha256(artifact) != terminal["hashes"][expected_role]
    ):
        raise RuntimeError("completed v2 cell payload changed")
    other_role = "failure" if expected_role == "payload" else "payload"
    if paths[other_role].exists():
        raise RuntimeError("completed v2 cell leaves an untracked artifact")
    payload = _load_json(artifact)
    if any(payload.get(key) != value for key, value in basis.items()) or (
        payload.get("status") != status
    ):
        raise RuntimeError("completed v2 payload has foreign identity")
    return terminal


def _validate_unchanged(
    manifest_path: Path,
    manifest_sha256: str,
    manifest: dict[str, Any],
) -> None:
    if _sha256(manifest_path) != manifest_sha256:
        raise RuntimeError("timing measurement manifest changed during cell")
    validate_manifest(manifest, require_ready=True)


def _clear_current_success_attempt(paths: dict[str, Path]) -> None:
    """Remove only success artifacts created after the initial clean check."""

    candidates = [paths["payload"]]
    candidates.extend(
        path.with_name(path.name + ".tmp")
        for path in (paths["payload"], paths["complete"])
    )
    for path in candidates:
        if path.exists():
            path.unlink()


def run_timing_cell(
    manifest_path: Path, out: Path, cell: dict[str, Any]
) -> dict[str, Any]:
    """Run one frozen paired timing and publish projected v2 provenance."""

    manifest_path = manifest_path.resolve()
    manifest, manifest_sha256 = _load_hashed_json(manifest_path)
    parent, _ = validate_manifest(manifest, require_ready=True)
    matches = [item for item in manifest["timing_cells"] if item == cell]
    if len(matches) != 1:
        raise RuntimeError("timing cell is outside the frozen allocation")
    tag = campaign.timing_tag(cell)
    paths = _paths(out, tag, "timing")
    existing = _existing_terminal(manifest_sha256, manifest, paths, cell)
    if existing is not None:
        return existing
    descriptor = campaign._state_descriptor(parent, cell["state_id"])
    baseline = campaign._treatment(parent, "S0")
    treatment = campaign._treatment(parent, cell["treatment_tag"])
    state = capture.restore_capture(descriptor["snapshot"])
    before = campaign._state_digest(state)
    try:
        raw = timing.time_pair(
            campaign._timing_factory(state, cell["state_id"], baseline),
            campaign._timing_factory(state, cell["state_id"], treatment),
            seed=cell["timing_seed"],
        )
        report = project_timing_report(raw, dimension=state["vp"].D)
        _validate_timing_projection(
            report,
            timing_seed=cell["timing_seed"],
            baseline_arm=baseline["config"]["arm"],
            treatment_arm=treatment["config"]["arm"],
        )
        if campaign._state_digest(state) != before:
            raise RuntimeError("paired timing mutated captured state or RNG")
        _validate_unchanged(manifest_path, manifest_sha256, manifest)
        payload = {
            **_terminal_basis(manifest_sha256, manifest, cell),
            "status": "succeeded",
            "state": descriptor,
            "baseline": baseline,
            "treatment": treatment,
            "timing": report,
            "quality_claim": False,
        }
        json.dumps(payload, allow_nan=False)
        integration.write_json(paths["payload"], payload)
        _validate_unchanged(manifest_path, manifest_sha256, manifest)
        terminal = {
            **_terminal_basis(manifest_sha256, manifest, cell),
            "status": "succeeded",
            "hashes": {"payload": _sha256(paths["payload"])},
        }
        integration.write_json(paths["complete"], terminal)
        return terminal
    except Exception as error:  # noqa: BLE001 - failed cells are evidence
        _validate_unchanged(manifest_path, manifest_sha256, manifest)
        _clear_current_success_attempt(paths)
        failure = {
            **_terminal_basis(manifest_sha256, manifest, cell),
            "status": "failed",
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        integration.write_json(paths["failure"], failure)
        _validate_unchanged(manifest_path, manifest_sha256, manifest)
        terminal = {
            **_terminal_basis(manifest_sha256, manifest, cell),
            "status": "failed",
            "hashes": {"failure": _sha256(paths["failure"])},
        }
        integration.write_json(paths["complete"], terminal)
        return terminal


def run_allocation_cell(
    manifest_path: Path, out: Path, cell: dict[str, Any]
) -> dict[str, Any]:
    """Run one frozen allocation measurement and publish projected output."""

    manifest_path = manifest_path.resolve()
    manifest, manifest_sha256 = _load_hashed_json(manifest_path)
    parent, _ = validate_manifest(manifest, require_ready=True)
    matches = [item for item in manifest["allocation_cells"] if item == cell]
    if len(matches) != 1:
        raise RuntimeError("allocation cell is outside the frozen allocation")
    tag = campaign.allocation_tag(cell)
    paths = _paths(out, tag, "allocation")
    existing = _existing_terminal(manifest_sha256, manifest, paths, cell)
    if existing is not None:
        return existing
    descriptor = campaign._state_descriptor(parent, cell["state_id"])
    treatment = campaign._treatment(
        parent,
        "S0" if cell["role"] == "baseline" else cell["treatment_tag"],
    )
    state = capture.restore_capture(descriptor["snapshot"])
    before = campaign._state_digest(state)
    try:
        raw = timing.measure_allocations(
            campaign._timing_factory(state, cell["state_id"], treatment),
            seed=cell["allocation_seed"],
        )
        report = project_allocation_report(raw, dimension=state["vp"].D)
        _validate_allocation_projection(
            report,
            allocation_seed=cell["allocation_seed"],
            expected_arm=treatment["config"]["arm"],
        )
        if campaign._state_digest(state) != before:
            raise RuntimeError("allocation measurement mutated state or RNG")
        _validate_unchanged(manifest_path, manifest_sha256, manifest)
        payload = {
            **_terminal_basis(manifest_sha256, manifest, cell),
            "status": "succeeded",
            "state": descriptor,
            "measured_treatment": treatment,
            "allocation": report,
            "timing_evidence": False,
        }
        json.dumps(payload, allow_nan=False)
        integration.write_json(paths["payload"], payload)
        _validate_unchanged(manifest_path, manifest_sha256, manifest)
        terminal = {
            **_terminal_basis(manifest_sha256, manifest, cell),
            "status": "succeeded",
            "hashes": {"payload": _sha256(paths["payload"])},
        }
        integration.write_json(paths["complete"], terminal)
        return terminal
    except Exception as error:  # noqa: BLE001 - failed cells are evidence
        _validate_unchanged(manifest_path, manifest_sha256, manifest)
        _clear_current_success_attempt(paths)
        failure = {
            **_terminal_basis(manifest_sha256, manifest, cell),
            "status": "failed",
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        integration.write_json(paths["failure"], failure)
        _validate_unchanged(manifest_path, manifest_sha256, manifest)
        terminal = {
            **_terminal_basis(manifest_sha256, manifest, cell),
            "status": "failed",
            "hashes": {"failure": _sha256(paths["failure"])},
        }
        integration.write_json(paths["complete"], terminal)
        return terminal


def mark_ready(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    validate_manifest(manifest, require_ready=False)
    if manifest.get("launch_ready") is True:
        raise RuntimeError("measurement manifest is already launch_ready")
    manifest["launch_ready"] = True
    integration.write_json(path, manifest)
    validate_manifest(_load_json(path), require_ready=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-manifest")
    prepare.add_argument("--parent-manifest", type=Path, required=True)
    prepare.add_argument("--selection-snapshot", type=Path, required=True)
    prepare.add_argument("--old-failed-terminal", type=Path, required=True)
    prepare.add_argument("--old-failed-artifact", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    ready = sub.add_parser("mark-ready")
    ready.add_argument("--manifest", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    cell = sub.add_parser("cell")
    cell.add_argument("--manifest", type=Path, required=True)
    cell.add_argument("--out", type=Path, required=True)
    cell.add_argument("--tag", required=True)
    allocation = sub.add_parser("allocation-cell")
    allocation.add_argument("--manifest", type=Path, required=True)
    allocation.add_argument("--out", type=Path, required=True)
    allocation.add_argument("--tag", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-manifest":
        manifest = prepare_manifest(
            args.parent_manifest,
            args.selection_snapshot,
            args.old_failed_terminal,
            args.old_failed_artifact,
        )
        if args.manifest.exists():
            raise RuntimeError("refusing to overwrite measurement manifest")
        integration.write_json(args.manifest, manifest)
    elif args.command == "mark-ready":
        mark_ready(args.manifest)
    else:
        manifest = _load_json(args.manifest)
        validate_manifest(manifest, require_ready=True)
        if args.command == "cell":
            matches = [
                item
                for item in manifest["timing_cells"]
                if campaign.timing_tag(item) == args.tag
            ]
            if len(matches) != 1:
                raise RuntimeError("unknown v2 timing cell tag")
            run_timing_cell(args.manifest, args.out, matches[0])
        elif args.command == "allocation-cell":
            matches = [
                item
                for item in manifest["allocation_cells"]
                if campaign.allocation_tag(item) == args.tag
            ]
            if len(matches) != 1:
                raise RuntimeError("unknown v2 allocation cell tag")
            run_allocation_cell(args.manifest, args.out, matches[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
