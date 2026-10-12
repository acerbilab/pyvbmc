"""Independent ordinary-MC crosschecks for selected E2 judge comparisons.

The development manifest selects, per captured state, the smallest and
largest recorded paired mean difference, every comparison at least ten
practical bands from zero, and every confidently flagged material raw loss.
Held-out manifests never inspect outcomes automatically and require explicit
comparison tags.  Cells use eight fresh ordinary-MC rules with common nodes
for every candidate in a state's deduplicated union.  Results are diagnostic:
they record agreement or disagreement with the frozen RQMC judge and never
replace its verdict automatically.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
for _key in THREAD_KEYS:
    os.environ.setdefault(_key, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "dev" / "scripts"
sys.path.insert(0, str(ROOT))
if os.environ.get("PYVBMC_GPYREG_SOURCE"):
    sys.path.insert(1, os.environ["PYVBMC_GPYREG_SOURCE"])
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

import noisy_acq_experiment as capture
import noisy_acq_integration as integration
import numpy as np

SCHEMA_VERSION = 1
BUDGETS = (32768, 65536)
REPLICATES = 8


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path | str, value: Any) -> None:
    integration.write_json(path, value)


def write_npz(path: Path | str, arrays: dict[str, np.ndarray]) -> None:
    integration.write_npz(path, arrays)


def manifest_digest(manifest: dict[str, Any]) -> str:
    return integration.manifest_digest(manifest)


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        SCRIPTS / "noisy_acq_integration.py",
        SCRIPTS / "noisy_acq_quadrature.py",
        SCRIPTS / "noisy_acq_experiment.py",
    ]
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths
    }


def comparison_needs_escalation(comparison: dict[str, Any]) -> bool:
    return comparison["mc_judge"]["classification"] == "unresolved" or (
        not comparison["comparison_penalty_active"]
        and comparison["mc_raw_loss"]["classification"] == "unresolved"
    )


def _finite_mean(comparison: dict[str, Any]) -> float | None:
    value = comparison["judge"].get("mean_difference")
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def select_development_tags(
    comparisons: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[str]]]:
    """Apply the frozen extrema/catastrophe policy without changing it."""

    by_state: dict[str, list[dict[str, Any]]] = {}
    for item in comparisons:
        by_state.setdefault(item["cell"]["state_id"], []).append(item)
    reasons: dict[str, list[str]] = {}
    for state_items in by_state.values():
        finite = [
            item for item in state_items if _finite_mean(item) is not None
        ]
        if finite:
            low = min(finite, key=lambda item: _finite_mean(item))
            high = max(finite, key=lambda item: _finite_mean(item))
            reasons.setdefault(low["integration_cell_tag"], []).append(
                "state_min_mean_difference"
            )
            reasons.setdefault(high["integration_cell_tag"], []).append(
                "state_max_mean_difference"
            )
        for item in state_items:
            mean = _finite_mean(item)
            eps_f = item["band"]["eps_F"]
            tag = item["integration_cell_tag"]
            if mean is not None and abs(mean) >= 10 * eps_f:
                reasons.setdefault(tag, []).append("ten_practical_bands")
            if item["raw_loss"]["classification"] == "material_loss":
                reasons.setdefault(tag, []).append(
                    "confident_raw_material_loss"
                )
    return sorted(reasons), reasons


def _deduplicate_pairs(
    selected: list[dict[str, Any]], reasons: dict[str, list[str]]
) -> list[dict[str, Any]]:
    pairs: dict[tuple[str, int, int], dict[str, Any]] = {}
    for item in selected:
        state_id = item["cell"]["state_id"]
        arm = int(item["selected_global_index"])
        baseline = int(item["baseline_global_index"])
        key = (state_id, arm, baseline)
        pair = pairs.setdefault(
            key,
            {
                "pair_id": hashlib.sha256(
                    json.dumps(key, separators=(",", ":")).encode()
                ).hexdigest()[:20],
                "state_id": state_id,
                "arm_global_index": arm,
                "baseline_global_index": baseline,
                "comparison_tags": [],
                "selection_reasons": [],
                "source_comparisons": [],
                "band": copy.deepcopy(item["band"]),
            },
        )
        if pair["band"]["eps_F"] != item["band"]["eps_F"]:
            raise RuntimeError(
                f"{state_id}: practical band changed within state"
            )
        tag = item["integration_cell_tag"]
        pair["comparison_tags"].append(tag)
        pair["selection_reasons"].extend(reasons[tag])
        pair["source_comparisons"].append(copy.deepcopy(item))
    for pair in pairs.values():
        pair["comparison_tags"].sort()
        pair["selection_reasons"] = sorted(set(pair["selection_reasons"]))
    return sorted(pairs.values(), key=lambda item: item["pair_id"])


def prepare_manifest(
    judge_summary_path: Path,
    integration_manifest_path: Path,
    budget: int,
    previous_summary_path: Path | None,
    explicit_tags: list[str],
) -> dict[str, Any]:
    source_summary = json.loads(judge_summary_path.read_text(encoding="utf-8"))
    integration_manifest = json.loads(
        integration_manifest_path.read_text(encoding="utf-8")
    )
    integration.validate_manifest(integration_manifest, require_ready=True)
    integration_identity = integration.runtime_identity(integration_manifest)
    if source_summary.get("integration_manifest_sha256") != manifest_digest(
        integration_manifest
    ):
        raise RuntimeError(
            "judge summary belongs to another integration manifest"
        )
    budget = int(budget)
    if budget not in BUDGETS:
        raise RuntimeError(f"crosscheck budget must be one of {BUDGETS}")
    split = integration_manifest["split"]
    source_comparisons = source_summary["comparisons"]
    known = {item["integration_cell_tag"]: item for item in source_comparisons}
    if len(known) != len(source_comparisons):
        raise RuntimeError(
            "source judge summary contains duplicate comparisons"
        )
    if split == "development":
        if explicit_tags:
            raise RuntimeError(
                "explicit comparison tags are reserved for holdout crosschecks"
            )
        tags, reasons = select_development_tags(source_comparisons)
        policy = "development_extrema_and_catastrophes"
    else:
        if not explicit_tags:
            raise RuntimeError(
                "holdout crosschecks require explicit comparison tags"
            )
        unknown = set(explicit_tags) - set(known)
        if unknown:
            raise RuntimeError(
                f"unknown explicit comparison tags: {sorted(unknown)}"
            )
        tags = sorted(set(explicit_tags))
        reasons = {tag: ["explicit"] for tag in tags}
        policy = "explicit_holdout_only"
    selected = [known[tag] for tag in tags]
    initial_pairs = _deduplicate_pairs(selected, reasons)

    previous = None
    carried: list[dict[str, Any]] = []
    previous_by_pair: dict[str, dict[str, Any]] = {}
    if budget == BUDGETS[1]:
        if previous_summary_path is None:
            raise RuntimeError(
                "65536-node crosscheck requires --previous-summary"
            )
        previous = json.loads(
            previous_summary_path.read_text(encoding="utf-8")
        )
        if previous.get("budget") != BUDGETS[0]:
            raise RuntimeError("crosscheck budgets must be consecutive")
        if previous.get("source_judge_summary_sha256") != sha256_file(
            judge_summary_path
        ):
            raise RuntimeError(
                "previous crosscheck used another judge summary"
            )
        if previous.get("frozen_comparison_tags") != tags:
            raise RuntimeError(
                "65536-node crosscheck must reuse the frozen 32768-node tags"
            )
        previous_by_pair = {
            item["pair_id"]: item for item in previous["comparisons"]
        }
        pending_ids = {
            pair_id
            for pair_id, item in previous_by_pair.items()
            if comparison_needs_escalation(item)
        }
        carried = [
            item
            for pair_id, item in previous_by_pair.items()
            if pair_id not in pending_ids
        ]
        pairs = [
            pair for pair in initial_pairs if pair["pair_id"] in pending_ids
        ]
        if set(pending_ids) - {pair["pair_id"] for pair in initial_pairs}:
            raise RuntimeError(
                "previous crosscheck names a pair absent from source"
            )
        scheduled_pair_count = previous["scheduled_pair_count"]
    else:
        if previous_summary_path is not None:
            raise RuntimeError(
                "32768-node crosscheck cannot use a previous summary"
            )
        pairs = initial_pairs
        scheduled_pair_count = len(initial_pairs)

    descriptors = {
        item["state_id"]: item for item in integration_manifest["states"]
    }
    unions = []
    for state_id in sorted({pair["state_id"] for pair in pairs}):
        state_pairs = [pair for pair in pairs if pair["state_id"] == state_id]
        if state_id not in descriptors:
            raise RuntimeError(f"crosscheck state is unavailable: {state_id}")
        indices = sorted(
            {
                index
                for pair in state_pairs
                for index in (
                    pair["arm_global_index"],
                    pair["baseline_global_index"],
                )
            }
        )
        unions.append(
            {
                "state_id": state_id,
                "state": descriptors[state_id],
                "candidate_indices": indices,
                "candidate_union_sha256": capture.digest(
                    (state_id, np.asarray(indices, dtype=np.int64))
                ),
                "pairs": state_pairs,
                "replicate_seeds": [
                    integration.derive_seed(
                        integration.MASTER_SEED,
                        state_id,
                        budget,
                        replicate,
                        "ordinary_mc_crosscheck",
                    )
                    for replicate in range(REPLICATES)
                ],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "ordinary_mc_crosscheck",
        "purpose": "E1 ordinary-MC diagnostic crosscheck",
        "launch_ready": False,
        "split": split,
        "holdout_locked": split == "holdout",
        "budget": budget,
        "replicates": REPLICATES,
        "selection_policy": policy,
        "explicit_comparison_tags": sorted(set(explicit_tags)),
        "frozen_comparison_tags": tags,
        "source_judge_summary": str(judge_summary_path.resolve()),
        "source_judge_summary_sha256": sha256_file(judge_summary_path),
        "integration_manifest": str(integration_manifest_path.resolve()),
        "integration_manifest_sha256": sha256_file(integration_manifest_path),
        "integration_identity": integration_identity,
        "source_hashes": source_hashes(),
        "unions": unions,
        "state_cell_count": len(unions),
        "scheduled_pair_count": scheduled_pair_count,
        "carried_comparisons": carried,
        "previous_summary": None
        if previous_summary_path is None
        else str(previous_summary_path.resolve()),
        "previous_summary_sha256": None
        if previous_summary_path is None
        else sha256_file(previous_summary_path),
    }


def validate_manifest(
    manifest: dict[str, Any], *, require_ready: bool
) -> None:
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != "ordinary_mc_crosscheck"
    ):
        raise RuntimeError("unsupported ordinary-MC crosscheck manifest")
    if require_ready and not manifest.get("launch_ready"):
        raise RuntimeError("crosscheck manifest is not marked launch_ready")
    if manifest.get("split") not in {"development", "holdout"}:
        raise RuntimeError("crosscheck split is invalid")
    if manifest.get("split") == "holdout" and manifest.get(
        "holdout_locked", True
    ):
        raise RuntimeError("holdout crosscheck remains locked")
    if (
        manifest.get("budget") not in BUDGETS
        or manifest.get("replicates") != REPLICATES
    ):
        raise RuntimeError("crosscheck numerical allocation changed")
    tags = manifest.get("frozen_comparison_tags", [])
    if tags != sorted(set(tags)):
        raise RuntimeError(
            "frozen comparison tags are duplicated or reordered"
        )
    explicit = manifest.get("explicit_comparison_tags", [])
    if manifest["split"] == "development":
        if explicit:
            raise RuntimeError(
                "development crosscheck cannot add explicit tags"
            )
        if (
            manifest.get("selection_policy")
            != "development_extrema_and_catastrophes"
        ):
            raise RuntimeError("development crosscheck policy changed")
    if manifest["split"] == "holdout":
        if not tags:
            raise RuntimeError("holdout crosscheck requires explicit tags")
        if tags != sorted(set(explicit)):
            raise RuntimeError(
                "holdout frozen tags differ from explicit allocation"
            )
        if manifest.get("selection_policy") != "explicit_holdout_only":
            raise RuntimeError("holdout crosscheck policy changed")
    unions = manifest.get("unions", [])
    if len(unions) != manifest.get("state_cell_count"):
        raise RuntimeError("crosscheck state allocation is inconsistent")
    state_ids = [union["state_id"] for union in unions]
    if len(state_ids) != len(set(state_ids)):
        raise RuntimeError("crosscheck state cells are duplicated")
    pair_ids = []
    for union in unions:
        indices = union["candidate_indices"]
        if len(indices) != len(set(indices)):
            raise RuntimeError("crosscheck candidate union is duplicated")
        expected = capture.digest(
            (union["state_id"], np.asarray(indices, dtype=np.int64))
        )
        if union["candidate_union_sha256"] != expected:
            raise RuntimeError("crosscheck candidate union identity changed")
        if len(union["replicate_seeds"]) != REPLICATES:
            raise RuntimeError("crosscheck replicate allocation changed")
        for pair in union["pairs"]:
            if not {
                pair["arm_global_index"],
                pair["baseline_global_index"],
            } <= set(indices):
                raise RuntimeError("crosscheck pair is absent from its union")
            pair_ids.append(pair["pair_id"])
    carried_ids = [item["pair_id"] for item in manifest["carried_comparisons"]]
    all_ids = pair_ids + carried_ids
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("crosscheck pairs are duplicated")
    if len(all_ids) > manifest["scheduled_pair_count"]:
        raise RuntimeError("crosscheck pair count exceeds its frozen schedule")
    if (
        manifest["budget"] == BUDGETS[0]
        and len(all_ids) != manifest["scheduled_pair_count"]
    ):
        raise RuntimeError("initial crosscheck omits a frozen pair")


def runtime_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    integration_path = Path(manifest["integration_manifest"])
    source_summary = Path(manifest["source_judge_summary"])
    if (
        sha256_file(integration_path)
        != manifest["integration_manifest_sha256"]
    ):
        raise RuntimeError("integration manifest changed")
    if sha256_file(source_summary) != manifest["source_judge_summary_sha256"]:
        raise RuntimeError("source judge summary changed")
    integration_manifest = json.loads(
        integration_path.read_text(encoding="utf-8")
    )
    integration.validate_manifest(integration_manifest, require_ready=True)
    if manifest["split"] != integration_manifest["split"]:
        raise RuntimeError(
            "crosscheck split differs from its integration source"
        )
    actual_integration = integration.runtime_identity(integration_manifest)
    if capture.canonical(actual_integration) != capture.canonical(
        manifest["integration_identity"]
    ):
        raise RuntimeError("integration identity changed")
    actual_sources = source_hashes()
    if actual_sources != manifest["source_hashes"]:
        raise RuntimeError("crosscheck source identity changed")
    previous = manifest.get("previous_summary")
    if (
        previous is not None
        and sha256_file(previous) != manifest["previous_summary_sha256"]
    ):
        raise RuntimeError("previous crosscheck summary changed")
    return {
        "integration_identity": actual_integration,
        "source_hashes": actual_sources,
    }


def _paths(out: Path, state_id: str) -> dict[str, Path]:
    return {
        "npz": out / "cells" / f"{state_id}.npz",
        "json": out / "cells" / f"{state_id}.json",
        "failure": out / "records" / f"{state_id}.failure.json",
        "complete": out / "records" / f"{state_id}.complete.json",
    }


def validate_cell(
    manifest: dict[str, Any], out: Path, union: dict[str, Any]
) -> dict[str, Any]:
    state_id = union["state_id"]
    paths = _paths(out, state_id)
    record = json.loads(paths["complete"].read_text(encoding="utf-8"))
    if record.get("manifest_sha256") != manifest_digest(manifest):
        raise RuntimeError(f"{state_id}: crosscheck manifest changed")
    if record.get("state_id") != state_id or record.get("status") not in {
        "succeeded",
        "failed",
    }:
        raise RuntimeError(f"{state_id}: invalid crosscheck completion")
    expected = (
        {"json", "npz"} if record["status"] == "succeeded" else {"failure"}
    )
    if set(record.get("hashes", {})) != expected:
        raise RuntimeError(f"{state_id}: incorrect crosscheck artifact set")
    for role, digest in record["hashes"].items():
        if not paths[role].is_file() or sha256_file(paths[role]) != digest:
            raise RuntimeError(f"{state_id}: changed crosscheck {role}")
    return record


def _state_digest(state: dict[str, Any], candidates: np.ndarray) -> str:
    return capture._observed_digest(
        candidates,
        state["gp"],
        state["vp"],
        state["logger"],
        state["optim_state"],
        SimpleNamespace(_noise_rng=None),
    )


def evaluate_mc_union(
    state: dict[str, Any],
    candidates: np.ndarray,
    budget: int,
    seeds: list[int],
) -> dict[str, np.ndarray]:
    from noisy_acq_quadrature import make_rule, prepare_rule

    rows: dict[str, list[Any]] = {
        "full_score": [],
        "log_residual": [],
        "log_reduction": [],
        "penalty": [],
        "valid": [],
        "reference_log_residual": [],
        "actual_nodes": [],
    }
    for seed in seeds:
        rule = make_rule(state["vp"], "mc", budget, seed)
        evaluator = prepare_rule(state, rule)
        score = evaluator.score(
            np.array(candidates, dtype=np.float64, copy=True), diagnostics=True
        )
        for key in (
            "full_score",
            "log_residual",
            "log_reduction",
            "penalty",
            "valid",
        ):
            rows[key].append(score[key])
        rows["reference_log_residual"].append(score["reference_log_residual"])
        rows["actual_nodes"].append(rule["metadata"]["actual_nodes"])
    return {key: np.asarray(value) for key, value in rows.items()}


def run_cell(
    manifest: dict[str, Any], out: Path, union: dict[str, Any]
) -> dict[str, Any]:
    validate_manifest(manifest, require_ready=True)
    runtime_identity(manifest)
    state_id = union["state_id"]
    paths = _paths(out, state_id)
    if paths["complete"].exists():
        return validate_cell(manifest, out, union)
    if any(path.exists() for key, path in paths.items() if key != "complete"):
        raise RuntimeError(f"{state_id}: partial crosscheck attempt exists")
    try:
        state = capture.restore_capture(union["state"]["snapshot"])
        sieve = np.asarray(state["cand"]["sieve_Xs"], dtype=np.float64)
        indices = np.asarray(union["candidate_indices"], dtype=np.int64)
        candidates = sieve[indices].copy()
        before = _state_digest(state, candidates)
        arrays = evaluate_mc_union(
            state, candidates, manifest["budget"], union["replicate_seeds"]
        )
        if before != _state_digest(state, candidates):
            raise RuntimeError("ordinary-MC crosscheck mutated captured state")
        arrays["candidate_indices"] = indices
        write_npz(paths["npz"], arrays)
        write_json(
            paths["json"],
            {
                "schema_version": SCHEMA_VERSION,
                "status": "succeeded",
                "state_id": state_id,
                "budget": manifest["budget"],
                "replicate_seeds": union["replicate_seeds"],
                "candidate_indices": union["candidate_indices"],
                "candidate_union_sha256": union["candidate_union_sha256"],
                "pairs": union["pairs"],
                "actual_nodes": arrays["actual_nodes"].tolist(),
            },
        )
        status = "succeeded"
        hashes = {
            "npz": sha256_file(paths["npz"]),
            "json": sha256_file(paths["json"]),
        }
    except Exception as error:  # noqa: BLE001
        write_json(
            paths["failure"],
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "state_id": state_id,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        status = "failed"
        hashes = {"failure": sha256_file(paths["failure"])}
    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "manifest_sha256": manifest_digest(manifest),
        "state_id": state_id,
        "hashes": hashes,
    }
    if status == "succeeded":
        runtime_identity(manifest)
    write_json(paths["complete"], completion)
    return validate_cell(manifest, out, union)


def _logmeanexp(values: np.ndarray, axis: int | None = None):
    values = np.asarray(values, dtype=np.float64)
    count = values.size if axis is None else values.shape[axis]
    return np.logaddexp.reduce(values, axis=axis) - np.log(count)


def summarize(manifest: dict[str, Any], out: Path) -> dict[str, Any]:
    from noisy_acq_quadrature import judge_paired, raw_loss_assessment

    validate_manifest(manifest, require_ready=True)
    runtime_identity(manifest)
    previous = None
    if manifest.get("previous_summary") is not None:
        previous = json.loads(
            Path(manifest["previous_summary"]).read_text(encoding="utf-8")
        )
    previous_by_pair = (
        {}
        if previous is None
        else {item["pair_id"]: item for item in previous["comparisons"]}
    )
    comparisons = copy.deepcopy(manifest["carried_comparisons"])
    failed_states = []
    for union in manifest["unions"]:
        record = validate_cell(manifest, out, union)
        if record["status"] != "succeeded":
            failed_states.append(union["state_id"])
            continue
        with np.load(
            _paths(out, union["state_id"])["npz"], allow_pickle=False
        ) as stored:
            arrays = {key: stored[key] for key in stored.files}
        positions = {
            int(index): position
            for position, index in enumerate(arrays["candidate_indices"])
        }
        for pair in union["pairs"]:
            arm_position = positions[pair["arm_global_index"]]
            baseline_position = positions[pair["baseline_global_index"]]
            arm_scores = arrays["full_score"][:, arm_position]
            baseline_scores = arrays["full_score"][:, baseline_position]
            pooled_difference = float(
                _logmeanexp(arrays["log_residual"][:, arm_position])
                + np.mean(arrays["penalty"][:, arm_position])
                - _logmeanexp(arrays["log_residual"][:, baseline_position])
                - np.mean(arrays["penalty"][:, baseline_position])
            )
            prior = previous_by_pair.get(pair["pair_id"])
            judged = judge_paired(
                arm_scores,
                baseline_scores,
                pair["band"]["eps_F"],
                previous=None if prior is None else prior["mc_judge"],
                replicate_seeds=union["replicate_seeds"],
                pooled_difference=pooled_difference,
            )
            raw_loss = raw_loss_assessment(
                arrays["log_reduction"][:, arm_position],
                arrays["log_reduction"][:, baseline_position],
                arrays["reference_log_residual"],
                previous=None if prior is None else prior["mc_raw_loss"],
            )
            penalty_active = bool(
                np.any(arrays["penalty"][:, arm_position] > 0)
                or np.any(arrays["penalty"][:, baseline_position] > 0)
            )
            source_classes = sorted(
                {
                    item["judge"]["classification"]
                    for item in pair["source_comparisons"]
                }
            )
            resolved_source = (
                len(source_classes) == 1 and source_classes[0] != "unresolved"
            )
            resolved_mc = judged["classification"] != "unresolved"
            source_raw_classes = sorted(
                {
                    item["raw_loss"]["classification"]
                    for item in pair["source_comparisons"]
                }
            )
            resolved_source_raw = (
                len(source_raw_classes) == 1
                and source_raw_classes[0] != "unresolved"
            )
            resolved_mc_raw = raw_loss["classification"] != "unresolved"
            comparisons.append(
                {
                    "pair_id": pair["pair_id"],
                    "state_id": union["state_id"],
                    "comparison_tags": pair["comparison_tags"],
                    "selection_reasons": pair["selection_reasons"],
                    "arm_global_index": pair["arm_global_index"],
                    "baseline_global_index": pair["baseline_global_index"],
                    "band": pair["band"],
                    "budget": manifest["budget"],
                    "source_rqmc_classifications": source_classes,
                    "source_rqmc_raw_loss_classifications": source_raw_classes,
                    "mc_judge": judged,
                    "mc_raw_loss": raw_loss,
                    "comparison_penalty_active": penalty_active,
                    "cross_method_status": (
                        "unresolved"
                        if not (resolved_source and resolved_mc)
                        else (
                            "agree"
                            if source_classes[0] == judged["classification"]
                            else "disagree"
                        )
                    ),
                    "raw_cross_method_status": (
                        "not_applicable_penalty_active"
                        if penalty_active
                        else (
                            "unresolved"
                            if not (resolved_source_raw and resolved_mc_raw)
                            else (
                                "agree"
                                if source_raw_classes[0]
                                == raw_loss["classification"]
                                else "disagree"
                            )
                        )
                    ),
                    "automatic_override": False,
                }
            )
    scheduled = manifest["scheduled_pair_count"]
    statuses = {
        key: sum(item["cross_method_status"] == key for item in comparisons)
        for key in ("agree", "disagree", "unresolved")
    }
    statuses["failed_or_missing"] = scheduled - len(comparisons)
    raw_statuses = {
        key: sum(
            item["raw_cross_method_status"] == key for item in comparisons
        )
        for key in (
            "agree",
            "disagree",
            "unresolved",
            "not_applicable_penalty_active",
        )
    }
    raw_statuses["failed_or_missing"] = scheduled - len(comparisons)
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest_digest(manifest),
        "source_judge_summary_sha256": manifest["source_judge_summary_sha256"],
        "budget": manifest["budget"],
        "scheduled_pair_count": scheduled,
        "frozen_comparison_tags": manifest["frozen_comparison_tags"],
        "status_counts": statuses,
        "raw_status_counts": raw_statuses,
        "failed_states": failed_states,
        "comparisons": comparisons,
        "automatic_override": False,
    }


def run_controller(
    manifest_path: Path, manifest: dict[str, Any], out: Path
) -> int:
    validate_manifest(manifest, require_ready=True)
    identity = runtime_identity(manifest)
    out.mkdir(parents=True, exist_ok=True)
    launch_path = out / "launch.json"
    launch = {"manifest": manifest, "identity": identity}
    if launch_path.exists():
        if json.loads(
            launch_path.read_text(encoding="utf-8")
        ) != capture.canonical(launch):
            raise RuntimeError("crosscheck output belongs to another manifest")
    else:
        write_json(launch_path, launch)
    failures = 0
    for union in manifest["unions"]:
        state_id = union["state_id"]
        complete = _paths(out, state_id)["complete"]
        if complete.exists():
            record = validate_cell(manifest, out, union)
        else:
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "cell",
                "--manifest",
                str(manifest_path.resolve()),
                "--out",
                str(out.resolve()),
                "--state-id",
                state_id,
            ]
            log = out / "logs" / f"{state_id}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
                )
            if result.returncode:
                raise RuntimeError(
                    f"{state_id}: crosscheck worker failed; see {log}"
                )
            record = validate_cell(manifest, out, union)
        failures += record["status"] == "failed"
        print(f"{record['status'].upper()} {state_id}", flush=True)
    write_json(out / "summary.json", summarize(manifest, out))
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--judge-summary", type=Path, required=True)
    prepare.add_argument("--integration-manifest", type=Path, required=True)
    prepare.add_argument("--budget", type=int, choices=BUDGETS, required=True)
    prepare.add_argument("--previous-summary", type=Path)
    prepare.add_argument("--comparison-tag", action="append", default=[])
    prepare.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    cell = sub.add_parser("cell")
    cell.add_argument("--manifest", type=Path, required=True)
    cell.add_argument("--out", type=Path, required=True)
    cell.add_argument("--state-id", required=True)
    summary_parser = sub.add_parser("summary")
    summary_parser.add_argument("--manifest", type=Path, required=True)
    summary_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_manifest(
            args.judge_summary,
            args.integration_manifest,
            args.budget,
            args.previous_summary,
            args.comparison_tag,
        )
        write_json(args.out, manifest)
        print(f"Wrote review draft {args.out}; set launch_ready after review.")
        return 0
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest, require_ready=False)
    if args.command == "run":
        return run_controller(args.manifest, manifest, args.out)
    if args.command == "cell":
        unions = {union["state_id"]: union for union in manifest["unions"]}
        if args.state_id not in unions:
            raise RuntimeError(f"unknown crosscheck state {args.state_id}")
        run_cell(manifest, args.out, unions[args.state_id])
        return 0
    report = summarize(manifest, args.out)
    write_json(args.out / "summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)
