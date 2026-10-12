"""Run E2 fixed-budget integration cells on captured VIQR states.

This runner is deliberately separate from ``noisy_acq_experiment.py`` so an
active state-capture campaign retains a fixed source hash.  It prepares one
manifest per stage and split, runs every cell in a fresh process, and resumes
only from hash-verified terminal completion records.

The development panel stage evaluates the predeclared method/budget grid on
512 candidates: the production baseline's top 32 plus 480 seeded candidates
from the remaining captured sieve.  Each state and replicate also receives a
fresh exact production MC100 baseline selection.  Full-sieve and holdout
manifests require an explicit frozen selection file and cannot reopen the
development grid.  Independent judge manifests pair each treatment with its
replicate's production winner and escalate only unresolved score or raw-loss
comparisons.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
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
import numpy as np

SCHEMA_VERSION = 1
METHOD_BUDGETS = {
    "mc": (128, 512, 2048),
    "stratified_mc": (128, 512, 2048),
    "stratified_rqmc": (128, 512, 2048),
}
REPLICATES = 8
MASTER_SEED = 2026091601
PANEL_SIZE = 512
PANEL_BASELINE_TOP = 32
JUDGE_BUDGETS = (4096, 8192, 16384, 32768, 65536)


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            capture.canonical(value),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    temporary.replace(path)


def write_npz(path: Path | str, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def manifest_digest(manifest: dict[str, Any]) -> str:
    text = json.dumps(
        capture.canonical(manifest), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_seed(master: int, *parts: Any) -> int:
    """Derive a stable independent uint64 seed from named stream roles."""

    payload = json.dumps([int(master), *parts], separators=(",", ":"))
    return int.from_bytes(
        hashlib.sha256(payload.encode()).digest()[:8], "little"
    )


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        SCRIPTS / "noisy_acq_quadrature.py",
        SCRIPTS / "noisy_acq_experiment.py",
        ROOT / "pyvbmc/testing/oracles/_state.py",
        ROOT / "pyvbmc/acquisition_functions/acq_fcn_viqr.py",
        ROOT / "pyvbmc/acquisition_functions/abstract_acq_fcn.py",
        ROOT / "pyvbmc/vbmc/active_importance_sampling.py",
    ]
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths
    }


def integration_identity(capture_manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the capture source lock, then identify E2-only sources."""

    capture_identity = capture.runtime_identity(capture_manifest)
    return {
        "capture_identity": capture_identity,
        "integration_source_hashes": source_hashes(),
    }


def _snapshot_hashes(base: Path) -> dict[str, str]:
    return {
        suffix: sha256_file(base.parent / f"{base.name}{suffix}")
        for suffix in (".json", ".npz")
    }


def discover_states(
    capture_manifest: dict[str, Any], captures: Path, split: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return existing states and explicit missing cells for one trajectory split."""

    wanted_seed = 0 if split == "development" else 1
    states = []
    missing = []
    for allocation in capture_manifest["allocation"]:
        label = allocation["label"]
        if wanted_seed not in allocation["seeds"]:
            raise RuntimeError(
                f"capture manifest lacks {label} seed {wanted_seed}"
            )
        tag = f"{label}_seed{wanted_seed}"
        completion_path = captures / "records" / f"{tag}.complete.json"
        if not completion_path.exists():
            for checkpoint in ("early", "late"):
                missing.append(
                    {
                        "state_id": f"{tag}_{checkpoint}",
                        "reason": "capture_case_missing",
                    }
                )
            continue
        complete = capture.validate_completed_case(
            capture_manifest, captures, label, wanted_seed
        )
        for checkpoint in ("early", "late"):
            state_id = f"{tag}_{checkpoint}"
            if not complete["capture"][f"{checkpoint}_present"]:
                missing.append(
                    {"state_id": state_id, "reason": "checkpoint_missing"}
                )
                continue
            base = captures / state_id
            states.append(
                {
                    "state_id": state_id,
                    "label": label,
                    "seed": wanted_seed,
                    "split": split,
                    "checkpoint": checkpoint,
                    "trajectory_status": complete["status"],
                    "snapshot": str(base.resolve()),
                    "snapshot_hashes": _snapshot_hashes(base),
                    "capture_completion_sha256": sha256_file(completion_path),
                    "panel_seed": derive_seed(
                        MASTER_SEED, state_id, "candidate_panel"
                    ),
                }
            )
    return states, missing


def _read_selection(
    path: Path | None, stage: str
) -> tuple[list[dict[str, Any]], Any]:
    if path is None:
        raise RuntimeError("full-sieve and holdout stages require --selection")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if not selection.get("frozen"):
        raise RuntimeError("selection is not marked frozen")
    settings = selection.get("settings", [])
    limit = 6 if stage == "full" else 2
    if not 1 <= len(settings) <= limit:
        raise RuntimeError(
            f"{stage} selection must contain between one and {limit} settings"
        )
    normalized = []
    seen_methods = set()
    seen_settings = set()
    for setting in settings:
        method = str(setting["method"])
        budget = int(setting["budget"])
        if (
            method not in METHOD_BUDGETS
            or budget not in METHOD_BUDGETS[method]
        ):
            raise RuntimeError(
                f"selection contains untested setting {setting}"
            )
        if (method, budget) in seen_settings:
            raise RuntimeError("selection contains a duplicate setting")
        if stage == "holdout" and method in seen_methods:
            raise RuntimeError(
                "holdout selection may retain only one setting per method"
            )
        if (
            stage == "full"
            and sum(item["method"] == method for item in normalized) >= 2
        ):
            raise RuntimeError(
                "full selection may retain at most two settings per method"
            )
        seen_methods.add(method)
        seen_settings.add((method, budget))
        normalized.append({"method": method, "budget": budget})
    return normalized, selection


def prepare_manifest(
    capture_manifest_path: Path,
    captures: Path,
    stage: str,
    selection_path: Path | None,
) -> dict[str, Any]:
    capture_manifest = json.loads(
        capture_manifest_path.read_text(encoding="utf-8")
    )
    capture.validate_manifest(capture_manifest, require_ready=True)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise RuntimeError("Set every BLAS thread environment variable to 1")
    if stage == "panel":
        split = "development"
        settings = [
            {"method": method, "budget": budget}
            for method, budgets in METHOD_BUDGETS.items()
            for budget in budgets
        ]
        selection = None
    else:
        split = "development" if stage == "full" else "holdout"
        settings, selection = _read_selection(selection_path, stage)
    states, missing = discover_states(capture_manifest, captures, split)
    cells = []
    for state in states:
        for setting in settings:
            for replicate in range(REPLICATES):
                cells.append(
                    {
                        "state_id": state["state_id"],
                        "method": setting["method"],
                        "budget": setting["budget"],
                        "replicate": replicate,
                        "rule_seed": derive_seed(
                            MASTER_SEED,
                            stage,
                            state["state_id"],
                            setting["method"],
                            setting["budget"],
                            replicate,
                            "integration_rule",
                        ),
                    }
                )
    baseline_cells = [
        {
            "state_id": state["state_id"],
            "replicate": replicate,
            "baseline_seed": derive_seed(
                MASTER_SEED,
                stage,
                state["state_id"],
                replicate,
                "production_mc100",
            ),
        }
        for state in states
        for replicate in range(REPLICATES)
    ]
    expected_states = 2 * len(capture_manifest["allocation"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "integration",
        "purpose": "noisy-acquisition E2 fixed-budget integration",
        "launch_ready": False,
        "stage": stage,
        "split": split,
        # A holdout manifest requires a second explicit edit in addition to
        # launch_ready, after the frozen selection has been reviewed.
        "holdout_locked": True,
        "capture_manifest": str(capture_manifest_path.resolve()),
        "capture_manifest_sha256": sha256_file(capture_manifest_path),
        "captures": str(captures.resolve()),
        "identity": integration_identity(capture_manifest),
        "settings": settings,
        "selection": selection,
        "selection_sha256": None
        if selection_path is None
        else sha256_file(selection_path),
        "replicates": REPLICATES,
        "master_seed": MASTER_SEED,
        "candidate_policy": {
            "panel_size": PANEL_SIZE if stage == "panel" else 8192,
            "baseline_top": PANEL_BASELINE_TOP if stage == "panel" else None,
            "sampled_remainder": PANEL_SIZE - PANEL_BASELINE_TOP
            if stage == "panel"
            else None,
            "selection_bias": stage == "panel",
        },
        "expected_state_count": expected_states,
        "states": states,
        "missing_states": missing,
        "cells": cells,
        "cell_count": len(cells),
        "baseline_cells": baseline_cells,
        "baseline_cell_count": len(baseline_cells),
        "scheduled_baseline_cell_count_including_missing_states": (
            expected_states * REPLICATES
        ),
        "scheduled_cell_count_including_missing_states": expected_states
        * len(settings)
        * REPLICATES,
    }


def validate_manifest(
    manifest: dict[str, Any], *, require_ready: bool
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported E2 manifest schema")
    if manifest.get("kind") != "integration":
        raise RuntimeError("manifest is not an integration manifest")
    if require_ready and not manifest.get("launch_ready"):
        raise RuntimeError("manifest is not marked launch_ready")
    if manifest.get("stage") not in {"panel", "full", "holdout"}:
        raise RuntimeError("unknown E2 stage")
    if manifest["stage"] == "holdout" and manifest.get("holdout_locked", True):
        raise RuntimeError("holdout manifest remains locked")
    if manifest.get("replicates") != REPLICATES:
        raise RuntimeError("replicate allocation changed")
    cells = manifest.get("cells", [])
    keys = [
        (
            cell["state_id"],
            cell["method"],
            cell["budget"],
            cell["replicate"],
        )
        for cell in cells
    ]
    if len(cells) != manifest.get("cell_count") or len(keys) != len(set(keys)):
        raise RuntimeError("cell allocation is duplicated or inconsistent")
    baseline_cells = manifest.get("baseline_cells", [])
    baseline_keys = [
        (cell["state_id"], cell["replicate"]) for cell in baseline_cells
    ]
    if len(baseline_cells) != manifest.get("baseline_cell_count") or len(
        baseline_keys
    ) != len(set(baseline_keys)):
        raise RuntimeError(
            "production baseline allocation is duplicated or inconsistent"
        )


def runtime_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    capture_manifest_path = Path(manifest["capture_manifest"])
    if (
        sha256_file(capture_manifest_path)
        != manifest["capture_manifest_sha256"]
    ):
        raise RuntimeError("capture manifest changed")
    capture_manifest = json.loads(
        capture_manifest_path.read_text(encoding="utf-8")
    )
    actual = integration_identity(capture_manifest)
    if capture.canonical(actual) != capture.canonical(manifest["identity"]):
        raise RuntimeError("integration or capture source identity changed")
    for state in manifest["states"]:
        base = Path(state["snapshot"])
        if _snapshot_hashes(base) != state["snapshot_hashes"]:
            raise RuntimeError(f"captured state changed: {state['state_id']}")
        capture_completion = (
            Path(manifest["captures"])
            / "records"
            / f"{state['label']}_seed{state['seed']}.complete.json"
        )
        if (
            not capture_completion.is_file()
            or sha256_file(capture_completion)
            != state["capture_completion_sha256"]
        ):
            raise RuntimeError(
                f"capture completion changed: {state['state_id']}"
            )
    return actual


def candidate_panel(
    state: dict[str, Any], descriptor: dict[str, Any], stage: str
):
    candidates = np.asarray(state["cand"]["sieve_Xs"], dtype=np.float64)
    if candidates.shape[0] != 8192:
        raise RuntimeError(
            "captured generated sieve does not contain 8192 rows"
        )
    repeat = int(state["cand"]["n_repeat_candidates"])
    baseline = np.asarray(state["ref"]["acq"], dtype=np.float64)[repeat:]
    if baseline.shape != (8192,):
        raise RuntimeError(
            "captured public reference does not match the sieve"
        )
    order = np.argsort(baseline, kind="stable")
    if stage != "panel":
        return candidates.copy(), np.arange(8192), baseline, order
    top = order[:PANEL_BASELINE_TOP]
    remaining = order[PANEL_BASELINE_TOP:]
    rng = np.random.default_rng(descriptor["panel_seed"])
    sampled = rng.choice(
        remaining,
        size=PANEL_SIZE - PANEL_BASELINE_TOP,
        replace=False,
    )
    indices = np.concatenate((top, sampled)).astype(np.int64)
    return candidates[indices].copy(), indices, baseline[indices], order


def state_digest(state: dict[str, Any], candidates: np.ndarray) -> str:
    problem = SimpleNamespace(_noise_rng=None)
    return capture._observed_digest(
        candidates,
        state["gp"],
        state["vp"],
        state["logger"],
        state["optim_state"],
        problem,
    )


def cell_tag(cell: dict[str, Any]) -> str:
    return (
        f"{cell['state_id']}__{cell['method']}__b{cell['budget']}"
        f"__r{cell['replicate']}"
    )


def _cell_paths(out: Path, tag: str) -> dict[str, Path]:
    return {
        "npz": out / "cells" / f"{tag}.npz",
        "json": out / "cells" / f"{tag}.json",
        "complete": out / "records" / f"{tag}.complete.json",
        "failure": out / "records" / f"{tag}.failure.json",
    }


def baseline_tag(cell: dict[str, Any]) -> str:
    return f"{cell['state_id']}__production_mc100__r{cell['replicate']}"


def _baseline_paths(out: Path, tag: str) -> dict[str, Path]:
    return {
        "npz": out / "baselines" / f"{tag}.npz",
        "json": out / "baselines" / f"{tag}.json",
        "complete": out / "records" / f"{tag}.complete.json",
        "failure": out / "records" / f"{tag}.failure.json",
    }


def prepare_production(state: dict[str, Any], seed: int) -> dict[str, Any]:
    """Make private production objects outside a warmed timing interval."""

    from pyvbmc.acquisition_functions import AcqFcnVIQR

    private_vp = copy.deepcopy(state["vp"])
    private_vp.rng = np.random.default_rng(int(seed))
    return {
        "vp": private_vp,
        "optim_state": copy.deepcopy(state["optim_state"]),
        "acquisition": AcqFcnVIQR(),
    }


def evaluate_production(
    state: dict[str, Any],
    candidates: np.ndarray,
    seed: int | None = None,
    *,
    prepared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate an exact fresh production VIQR MC100 selection.

    The private VP receives a new generator because ``deepcopy`` otherwise
    shares the captured generator.  The public score is normalized by its
    actual node count so it is on the same complete-score scale as the
    explicit-weight treatments.
    """

    from pyvbmc.vbmc.active_importance_sampling import (
        active_importance_sampling,
    )

    if prepared is None:
        if seed is None:
            raise ValueError(
                "seed is required when prepared objects are absent"
            )
        prepared = prepare_production(state, seed)
    elif seed is not None:
        raise ValueError("pass either seed or prepared objects, not both")
    private_vp = prepared["vp"]
    private_optim = prepared["optim_state"]
    acq = prepared["acquisition"]
    started = time.perf_counter()
    t0 = time.perf_counter()
    active_is = active_importance_sampling(
        private_vp, state["gp"], acq, state["options"]
    )
    preparation_seconds = time.perf_counter() - t0
    nodes = np.asarray(active_is["X"], dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[0] != 100:
        raise RuntimeError(
            "production VIQR baseline did not generate exactly 100 MC nodes"
        )
    private_optim["active_importance_sampling"] = active_is
    candidate_copy = np.array(candidates, dtype=np.float64, copy=True)
    t0 = time.perf_counter()
    raw_score = acq(
        candidate_copy,
        state["gp"],
        private_vp,
        state["logger"],
        private_optim,
    )
    evaluation_seconds = time.perf_counter() - t0
    normalized = np.asarray(raw_score, dtype=np.float64) - np.log(
        nodes.shape[0]
    )
    return {
        "full_score": normalized,
        "raw_public_score": np.asarray(raw_score, dtype=np.float64),
        "X_used": candidate_copy.copy(),
        "nodes": nodes.copy(),
        "ln_weights": np.asarray(active_is["ln_weights"]).copy(),
        "f_s2": np.asarray(active_is["f_s2"]).copy(),
        "actual_nodes": int(nodes.shape[0]),
        "private_rng_final": copy.deepcopy(private_vp.rng.bit_generator.state),
        "timing_seconds": {
            "node_and_cache_preparation": preparation_seconds,
            "evaluation": evaluation_seconds,
            "total_baseline": time.perf_counter() - started,
        },
    }


def evaluate_treatment(
    state: dict[str, Any],
    candidates: np.ndarray,
    method: str,
    budget: int,
    seed: int,
    *,
    diagnostics: bool = False,
) -> dict[str, Any]:
    """Build and evaluate one private explicit-weight integration rule."""

    from noisy_acq_quadrature import make_rule, prepare_rule

    started = time.perf_counter()
    t0 = time.perf_counter()
    rule = make_rule(state["vp"], method, budget, seed)
    generation_seconds = time.perf_counter() - t0
    if not rule["available"]:
        raise RuntimeError(
            f"integration rule unavailable: {rule['metadata']['reason']}"
        )
    t0 = time.perf_counter()
    evaluator = prepare_rule(state, rule)
    preparation_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    score = evaluator.score(
        np.array(candidates, dtype=np.float64, copy=True),
        diagnostics=diagnostics,
    )
    evaluation_seconds = time.perf_counter() - t0
    return {
        "rule": rule,
        "evaluator": evaluator,
        "score": score,
        "timing_seconds": {
            "node_generation": generation_seconds,
            "preparation": preparation_seconds,
            "evaluation": evaluation_seconds,
            "total_treatment": time.perf_counter() - started,
        },
    }


def prepare_production_operation(
    state: dict[str, Any], candidates: np.ndarray, seed: int
):
    """Return a small-result production operation for warmed timing."""

    from pyvbmc.vbmc.active_importance_sampling import (
        active_importance_sampling,
    )

    prepared = prepare_production(state, seed)
    candidate_copy = np.array(candidates, dtype=np.float64, copy=True)

    def operation():
        active_is = active_importance_sampling(
            prepared["vp"],
            state["gp"],
            prepared["acquisition"],
            state["options"],
        )
        nodes = np.asarray(active_is["X"])
        if nodes.ndim != 2 or nodes.shape[0] != 100:
            raise RuntimeError(
                "production VIQR baseline did not generate exactly 100 MC nodes"
            )
        prepared["optim_state"]["active_importance_sampling"] = active_is
        score = prepared["acquisition"](
            candidate_copy,
            state["gp"],
            prepared["vp"],
            state["logger"],
            prepared["optim_state"],
        ) - np.log(nodes.shape[0])
        selected = int(np.argmin(score))
        return {
            "selected_local_index": selected,
            "selected_full_score": float(score[selected]),
            "actual_nodes": int(nodes.shape[0]),
        }

    return operation


def prepare_treatment_operation(
    state: dict[str, Any],
    candidates: np.ndarray,
    method: str,
    budget: int,
    seed: int,
):
    """Return a small-result fixed-rule operation for warmed timing."""

    from noisy_acq_quadrature import make_rule, prepare_rule

    candidate_copy = np.array(candidates, dtype=np.float64, copy=True)

    def operation():
        rule = make_rule(state["vp"], method, budget, seed)
        if not rule["available"]:
            raise RuntimeError(
                f"integration rule unavailable: {rule['metadata']['reason']}"
            )
        evaluator = prepare_rule(state, rule)
        score = evaluator.score(candidate_copy, diagnostics=False)
        valid = np.asarray(score["valid"], dtype=bool)
        if not np.any(valid):
            raise RuntimeError(
                "integration treatment produced no valid candidate"
            )
        selected = int(np.argmin(np.where(valid, score["full_score"], np.inf)))
        return {
            "selected_local_index": selected,
            "selected_full_score": float(score["full_score"][selected]),
            "actual_nodes": int(rule["metadata"]["actual_nodes"]),
        }

    return operation


def validate_cell(manifest: dict[str, Any], out: Path, cell: dict[str, Any]):
    tag = cell_tag(cell)
    paths = _cell_paths(out, tag)
    record = json.loads(paths["complete"].read_text(encoding="utf-8"))
    if record.get("manifest_sha256") != manifest_digest(manifest):
        raise RuntimeError(f"{tag}: manifest changed")
    if record.get("cell") != cell or record.get("status") not in {
        "succeeded",
        "failed",
    }:
        raise RuntimeError(f"{tag}: invalid completion record")
    expected = (
        {"json", "npz"} if record["status"] == "succeeded" else {"failure"}
    )
    if set(record.get("hashes", {})) != expected:
        raise RuntimeError(f"{tag}: incorrect artifact set")
    for role, expected_hash in record["hashes"].items():
        if (
            not paths[role].is_file()
            or sha256_file(paths[role]) != expected_hash
        ):
            raise RuntimeError(f"{tag}: changed {role} artifact")
    return record


def validate_baseline_cell(
    manifest: dict[str, Any], out: Path, cell: dict[str, Any]
) -> dict[str, Any]:
    tag = baseline_tag(cell)
    paths = _baseline_paths(out, tag)
    record = json.loads(paths["complete"].read_text(encoding="utf-8"))
    if record.get("manifest_sha256") != manifest_digest(manifest):
        raise RuntimeError(f"{tag}: manifest changed")
    if record.get("cell") != cell or record.get("status") not in {
        "succeeded",
        "failed",
    }:
        raise RuntimeError(f"{tag}: invalid baseline completion record")
    expected = (
        {"json", "npz"} if record["status"] == "succeeded" else {"failure"}
    )
    if set(record.get("hashes", {})) != expected:
        raise RuntimeError(f"{tag}: incorrect baseline artifact set")
    for role, expected_hash in record["hashes"].items():
        if (
            not paths[role].is_file()
            or sha256_file(paths[role]) != expected_hash
        ):
            raise RuntimeError(f"{tag}: changed baseline {role} artifact")
    return record


def run_baseline_cell(
    manifest: dict[str, Any], out: Path, cell: dict[str, Any]
) -> dict[str, Any]:
    validate_manifest(manifest, require_ready=True)
    runtime_identity(manifest)
    tag = baseline_tag(cell)
    paths = _baseline_paths(out, tag)
    if paths["complete"].exists():
        return validate_baseline_cell(manifest, out, cell)
    partial = [
        path
        for key, path in paths.items()
        if key != "complete" and path.exists()
    ]
    if partial:
        raise RuntimeError(f"{tag}: partial prior baseline artifacts exist")
    descriptors = {state["state_id"]: state for state in manifest["states"]}
    descriptor = descriptors[cell["state_id"]]
    try:
        state = capture.restore_capture(descriptor["snapshot"])
        candidates, indices, captured_scores, captured_order = candidate_panel(
            state, descriptor, manifest["stage"]
        )
        before = state_digest(state, candidates)
        evaluated = evaluate_production(
            state, candidates, cell["baseline_seed"]
        )
        if before != state_digest(state, candidates):
            raise RuntimeError(
                "production baseline mutated captured state or RNG"
            )
        full_score = evaluated["full_score"]
        valid = np.isfinite(full_score)
        if not np.any(valid):
            raise RuntimeError(
                "production baseline produced no valid candidate"
            )
        selected_local = int(np.argmin(np.where(valid, full_score, np.inf)))
        selected_global = int(indices[selected_local])
        write_npz(
            paths["npz"],
            {
                "candidate_indices": np.asarray(indices, dtype=np.int64),
                "captured_public_score": np.asarray(captured_scores),
                "full_score": np.asarray(full_score),
                "raw_public_score": evaluated["raw_public_score"],
                "nodes": evaluated["nodes"],
                "ln_weights": evaluated["ln_weights"],
                "f_s2": evaluated["f_s2"],
                "selected_X": evaluated["X_used"][[selected_local]],
            },
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "succeeded",
            "cell": cell,
            "state": descriptor,
            "candidate_panel_sha256": capture.digest(
                (indices, candidates, captured_scores)
            ),
            "fixed_captured_winner_global_index": int(captured_order[0]),
            "selected_local_index": selected_local,
            "selected_global_index": selected_global,
            "selected_full_score": float(full_score[selected_local]),
            "actual_nodes": evaluated["actual_nodes"],
            "node_sha256": capture.digest(evaluated["nodes"]),
            "private_rng_final_sha256": capture.digest(
                evaluated["private_rng_final"]
            ),
            "timing_kind": "single_shot_exploratory",
            "timing_seconds": evaluated["timing_seconds"],
        }
        write_json(paths["json"], report)
        status = "succeeded"
        hashes = {
            "npz": sha256_file(paths["npz"]),
            "json": sha256_file(paths["json"]),
        }
    except Exception as error:  # noqa: BLE001
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "cell": cell,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(paths["failure"], failure)
        status = "failed"
        hashes = {"failure": sha256_file(paths["failure"])}
    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "manifest_sha256": manifest_digest(manifest),
        "cell": cell,
        "hashes": hashes,
    }
    if status == "succeeded":
        runtime_identity(manifest)
    write_json(paths["complete"], completion)
    return validate_baseline_cell(manifest, out, cell)


def run_cell(
    manifest: dict[str, Any], out: Path, cell: dict[str, Any]
) -> dict[str, Any]:
    validate_manifest(manifest, require_ready=True)
    runtime_identity(manifest)
    tag = cell_tag(cell)
    paths = _cell_paths(out, tag)
    if paths["complete"].exists():
        return validate_cell(manifest, out, cell)
    partial = [
        path
        for key, path in paths.items()
        if key != "complete" and path.exists()
    ]
    if partial:
        raise RuntimeError(f"{tag}: partial prior artifacts exist")
    descriptors = {state["state_id"]: state for state in manifest["states"]}
    descriptor = descriptors[cell["state_id"]]
    started = time.perf_counter()
    try:
        state = capture.restore_capture(descriptor["snapshot"])
        candidates, indices, baseline_panel, baseline_order = candidate_panel(
            state, descriptor, manifest["stage"]
        )
        before = state_digest(state, candidates)
        evaluated = evaluate_treatment(
            state,
            candidates,
            cell["method"],
            cell["budget"],
            cell["rule_seed"],
            diagnostics=False,
        )
        rule = evaluated["rule"]
        evaluator = evaluated["evaluator"]
        score = evaluated["score"]
        t0 = time.perf_counter()
        diagnostics = evaluator.score(
            np.array(candidates, dtype=np.float64, copy=True), diagnostics=True
        )
        diagnostics_seconds = time.perf_counter() - t0
        np.testing.assert_array_equal(
            score["full_score"], diagnostics["full_score"]
        )
        if before != state_digest(state, candidates):
            raise RuntimeError(
                "integration treatment mutated captured state or RNG"
            )
        valid = np.asarray(score["valid"], dtype=bool)
        if not np.any(valid):
            raise RuntimeError(
                "integration treatment produced no valid candidate"
            )
        selectable = np.where(valid, score["full_score"], np.inf)
        selected_local = int(np.argmin(selectable))
        selected_global = int(indices[selected_local])
        arrays = {
            "candidate_indices": np.asarray(indices, dtype=np.int64),
            "baseline_public_score": np.asarray(baseline_panel),
            "full_score": np.asarray(score["full_score"]),
            "log_residual": np.asarray(score["log_residual"]),
            "residual": np.asarray(score["residual"]),
            "log_reduction": np.asarray(diagnostics["log_reduction"]),
            "reduction": np.asarray(diagnostics["reduction"]),
            "penalty": np.asarray(score["penalty"]),
            "valid": valid,
            "selected_X": np.asarray(score["X_used"])[[selected_local]],
            "rule_nodes": np.asarray(rule["nodes"]),
            "rule_weights": np.asarray(rule["weights"]),
        }
        write_npz(paths["npz"], arrays)
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "succeeded",
            "cell": cell,
            "state": descriptor,
            "candidate_panel_sha256": capture.digest(
                (indices, candidates, baseline_panel)
            ),
            "baseline_winner_global_index": int(baseline_order[0]),
            "selected_local_index": selected_local,
            "selected_global_index": selected_global,
            "selected_full_score": float(score["full_score"][selected_local]),
            "selected_log_residual": float(
                score["log_residual"][selected_local]
            ),
            "selected_reduction": float(
                diagnostics["reduction"][selected_local]
            ),
            "rule": {
                key: capture.canonical(value)
                for key, value in rule.items()
                if key not in {"nodes", "weights"}
            },
            "evaluator": capture.canonical(evaluator.metadata),
            "timing_kind": "single_shot_exploratory",
            "timing_seconds": {
                **evaluated["timing_seconds"],
                "diagnostics_excluded": diagnostics_seconds,
                "worker": time.perf_counter() - started,
            },
            "failures": {
                str(reason): int(
                    np.count_nonzero(score["failure_reason"] == reason)
                )
                for reason in set(score["failure_reason"])
                if reason is not None
            },
        }
        write_json(paths["json"], report)
        status = "succeeded"
        hashes = {
            "npz": sha256_file(paths["npz"]),
            "json": sha256_file(paths["json"]),
        }
    except (
        Exception
    ) as error:  # noqa: BLE001 - every scheduled cell is retained
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "cell": cell,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(paths["failure"], failure)
        status = "failed"
        hashes = {"failure": sha256_file(paths["failure"])}
    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "manifest_sha256": manifest_digest(manifest),
        "cell": cell,
        "hashes": hashes,
    }
    if status == "succeeded":
        runtime_identity(manifest)
    write_json(paths["complete"], completion)
    return validate_cell(manifest, out, cell)


def inventory(manifest: dict[str, Any], out: Path) -> dict[str, Any]:
    counts = {"succeeded": 0, "failed": 0, "partial": 0, "missing": 0}
    rows = []
    for cell in manifest["cells"]:
        tag = cell_tag(cell)
        paths = _cell_paths(out, tag)
        if paths["complete"].exists():
            try:
                record = validate_cell(manifest, out, cell)
                status = record["status"]
                counts[status] += 1
                rows.append({"tag": tag, "status": status, "cell": cell})
            except Exception as error:  # noqa: BLE001
                counts["partial"] += 1
                rows.append(
                    {
                        "tag": tag,
                        "status": "partial",
                        "cell": cell,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        else:
            partial = any(path.exists() for path in paths.values())
            status = "partial" if partial else "missing"
            counts[status] += 1
            rows.append({"tag": tag, "status": status, "cell": cell})
    scheduled = manifest["scheduled_cell_count_including_missing_states"]
    unavailable = scheduled - manifest["cell_count"]
    baseline_counts = {"succeeded": 0, "failed": 0, "partial": 0, "missing": 0}
    baseline_rows = []
    for cell in manifest["baseline_cells"]:
        tag = baseline_tag(cell)
        paths = _baseline_paths(out, tag)
        if paths["complete"].exists():
            try:
                record = validate_baseline_cell(manifest, out, cell)
                status = record["status"]
            except Exception as error:  # noqa: BLE001
                status = "partial"
                baseline_rows.append(
                    {
                        "tag": tag,
                        "status": status,
                        "cell": cell,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                baseline_counts[status] += 1
                continue
        else:
            status = (
                "partial"
                if any(path.exists() for path in paths.values())
                else "missing"
            )
        baseline_counts[status] += 1
        baseline_rows.append({"tag": tag, "status": status, "cell": cell})
    scheduled_baselines = manifest[
        "scheduled_baseline_cell_count_including_missing_states"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest_digest(manifest),
        "scheduled_cells": scheduled,
        "available_state_cells": manifest["cell_count"],
        "missing_state_cells": unavailable,
        "counts": counts,
        "rows": rows,
        "production_baseline": {
            "scheduled_cells": scheduled_baselines,
            "available_state_cells": manifest["baseline_cell_count"],
            "missing_state_cells": scheduled_baselines
            - manifest["baseline_cell_count"],
            "counts": baseline_counts,
            "rows": baseline_rows,
        },
    }


def summarize(manifest: dict[str, Any], out: Path) -> dict[str, Any]:
    report = inventory(manifest, out)
    baseline_selected = []
    baseline_times = []
    for cell in manifest["baseline_cells"]:
        tag = baseline_tag(cell)
        completion = _baseline_paths(out, tag)["complete"]
        if not completion.exists():
            continue
        record = validate_baseline_cell(manifest, out, cell)
        if record["status"] != "succeeded":
            continue
        side = json.loads(
            _baseline_paths(out, tag)["json"].read_text(encoding="utf-8")
        )
        baseline_times.append(side["timing_seconds"]["total_baseline"])
        baseline_selected.append(
            {
                "state_id": cell["state_id"],
                "replicate": cell["replicate"],
                "global_index": side["selected_global_index"],
            }
        )
    settings = []
    for setting in manifest["settings"]:
        cells = [
            cell
            for cell in manifest["cells"]
            if cell["method"] == setting["method"]
            and cell["budget"] == setting["budget"]
        ]
        times = []
        selected = []
        failures = 0
        for cell in cells:
            tag = cell_tag(cell)
            complete = _cell_paths(out, tag)["complete"]
            if not complete.exists():
                failures += 1
                continue
            record = validate_cell(manifest, out, cell)
            if record["status"] != "succeeded":
                failures += 1
                continue
            side = json.loads(
                _cell_paths(out, tag)["json"].read_text(encoding="utf-8")
            )
            times.append(side["timing_seconds"]["total_treatment"])
            selected.append(
                {
                    "state_id": cell["state_id"],
                    "replicate": cell["replicate"],
                    "global_index": side["selected_global_index"],
                }
            )
        denominator = manifest["expected_state_count"] * REPLICATES
        settings.append(
            {
                **setting,
                "scheduled": denominator,
                "succeeded": len(selected),
                "failed_or_missing": denominator - len(selected),
                "observed_cell_failures": failures,
                "median_treatment_seconds": None
                if not times
                else float(np.median(times)),
                "selected": selected,
            }
        )
    baseline_denominator = manifest["expected_state_count"] * REPLICATES
    return {
        "inventory": report,
        "production_baseline": {
            "scheduled": baseline_denominator,
            "succeeded": len(baseline_selected),
            "failed_or_missing": baseline_denominator - len(baseline_selected),
            "median_single_shot_exploratory_seconds": None
            if not baseline_times
            else float(np.median(baseline_times)),
            "selected": baseline_selected,
        },
        "settings": settings,
    }


def prepare_judge_manifest(
    integration_manifest_path: Path,
    results: Path,
    budget: int,
    previous_summary: Path | None,
) -> dict[str, Any]:
    """Freeze each state's selected-candidate union for independent judging."""

    integration = json.loads(
        integration_manifest_path.read_text(encoding="utf-8")
    )
    validate_manifest(integration, require_ready=True)
    runtime_identity(integration)
    budget = int(budget)
    if budget not in JUDGE_BUDGETS:
        raise RuntimeError(f"judge budget must be one of {JUDGE_BUDGETS}")
    if budget != JUDGE_BUDGETS[0] and previous_summary is None:
        raise RuntimeError(
            "escalated judge budgets require --previous-summary"
        )
    previous = None
    previous_by_tag: dict[str, dict[str, Any]] = {}
    previous_band_by_state: dict[str, dict[str, Any]] = {}
    previous_best_by_state: dict[str, dict[str, Any]] = {}
    carried_comparisons: list[dict[str, Any]] = []
    pending_tags: set[str] | None = None
    if previous_summary is not None:
        previous = json.loads(previous_summary.read_text(encoding="utf-8"))
        if previous.get("integration_manifest_sha256") != manifest_digest(
            integration
        ):
            raise RuntimeError(
                "previous judge summary belongs to another integration"
            )
        previous_budget = int(previous["judge_budget"])
        if JUDGE_BUDGETS.index(previous_budget) + 1 != JUDGE_BUDGETS.index(
            budget
        ):
            raise RuntimeError("judge budgets must be consecutive")
        previous_by_tag = {
            item["integration_cell_tag"]: item
            for item in previous["comparisons"]
        }
        pending_tags = {
            tag
            for tag, item in previous_by_tag.items()
            if comparison_needs_escalation(item)
        }
        carried_comparisons = [
            item
            for tag, item in previous_by_tag.items()
            if tag not in pending_tags
        ]
        for item in previous["comparisons"]:
            prior_state = item["cell"]["state_id"]
            previous_band_by_state.setdefault(prior_state, item["band"])
            if item["pooled_union_best_global_index"] is not None:
                previous_best_by_state.setdefault(
                    prior_state,
                    {
                        "global_index": item["pooled_union_best_global_index"],
                        "union": item["assessed_union"],
                    },
                )

    unions = []
    missing_selections = (
        []
        if previous is None
        else copy.deepcopy(previous.get("missing_selections", []))
    )
    by_state: dict[str, list[dict[str, Any]]] = {}
    for cell in integration["cells"]:
        tag = cell_tag(cell)
        if pending_tags is None or tag in pending_tags:
            by_state.setdefault(cell["state_id"], []).append(cell)
    descriptors = {state["state_id"]: state for state in integration["states"]}
    for state_id, cells in by_state.items():
        descriptor = descriptors[state_id]
        state = capture.restore_capture(descriptor["snapshot"])
        _, _, _, baseline_order = candidate_panel(
            state, descriptor, integration["stage"]
        )
        fixed_baseline = int(baseline_order[0])
        provenance: dict[int, list[str]] = {
            fixed_baseline: ["captured_fixed_baseline"]
        }
        treatment_by_tag: dict[str, int] = {}
        if previous is None:
            for index in baseline_order[:PANEL_BASELINE_TOP]:
                provenance.setdefault(int(index), []).append(
                    "fixed_baseline_shortlist"
                )
        previous_best = previous_best_by_state.get(state_id)
        if previous_best is not None:
            provenance.setdefault(
                int(previous_best["global_index"]), []
            ).append("previous_union_best_reference")
        baseline_by_replicate: dict[str, int] = {}
        pending_replicates = {cell["replicate"] for cell in cells}
        for baseline_cell in integration["baseline_cells"]:
            if (
                baseline_cell["state_id"] != state_id
                or baseline_cell["replicate"] not in pending_replicates
            ):
                continue
            baseline_name = baseline_tag(baseline_cell)
            baseline_paths = _baseline_paths(results, baseline_name)
            if not baseline_paths["complete"].exists():
                missing_selections.append(
                    {
                        "cell": baseline_cell,
                        "kind": "production_baseline",
                        "reason": "baseline_cell_missing",
                    }
                )
                continue
            completion = validate_baseline_cell(
                integration, results, baseline_cell
            )
            if completion["status"] != "succeeded":
                missing_selections.append(
                    {
                        "cell": baseline_cell,
                        "kind": "production_baseline",
                        "reason": "baseline_cell_failed",
                    }
                )
                continue
            report = json.loads(
                baseline_paths["json"].read_text(encoding="utf-8")
            )
            index = int(report["selected_global_index"])
            replicate = str(baseline_cell["replicate"])
            baseline_by_replicate[replicate] = index
            provenance.setdefault(index, []).append(baseline_name)
        for cell in cells:
            tag = cell_tag(cell)
            paths = _cell_paths(results, tag)
            if not paths["complete"].exists():
                missing_selections.append(
                    {"cell": cell, "reason": "integration_cell_missing"}
                )
                continue
            completion = validate_cell(integration, results, cell)
            if completion["status"] != "succeeded":
                missing_selections.append(
                    {"cell": cell, "reason": "integration_cell_failed"}
                )
                continue
            report = json.loads(paths["json"].read_text(encoding="utf-8"))
            index = int(report["selected_global_index"])
            treatment_by_tag[tag] = index
            provenance.setdefault(index, []).append(tag)
        indices = sorted(provenance)
        candidate_union_sha256 = capture.digest(
            (state_id, np.asarray(indices, dtype=np.int64))
        )
        unions.append(
            {
                "state_id": state_id,
                "state": descriptor,
                "candidate_indices": indices,
                "candidate_provenance": {
                    str(index): provenance[index] for index in indices
                },
                "candidate_union_sha256": candidate_union_sha256,
                "union_origin_budget": budget,
                "previous_union_best_reference": previous_best,
                "fixed_baseline_global_index": fixed_baseline,
                "baseline_by_replicate": baseline_by_replicate,
                "treatment_by_tag": treatment_by_tag,
                "comparison_seeds": [
                    derive_seed(
                        MASTER_SEED,
                        state_id,
                        budget,
                        replicate,
                        "comparison_judge",
                    )
                    for replicate in range(REPLICATES)
                ],
                "band_pilot_budget": JUDGE_BUDGETS[0],
                "band_pilot_seeds": []
                if previous is not None
                else [
                    derive_seed(
                        MASTER_SEED,
                        state_id,
                        JUDGE_BUDGETS[0],
                        replicate,
                        "band_pilot",
                    )
                    for replicate in range(REPLICATES)
                ],
                "frozen_band": previous_band_by_state.get(state_id),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "judge",
        "purpose": "noisy-acquisition E2 independent selected-union judge",
        "launch_ready": False,
        "split": integration["split"],
        "holdout_locked": integration["split"] == "holdout",
        "judge_budget": budget,
        "replicates": REPLICATES,
        "integration_manifest": str(integration_manifest_path.resolve()),
        "integration_manifest_sha256": sha256_file(integration_manifest_path),
        "integration_results": str(results.resolve()),
        "identity": integration["identity"],
        "unions": unions,
        "missing_selections": missing_selections,
        "carried_comparisons": carried_comparisons,
        "carried_missing_comparisons": []
        if previous is None
        else copy.deepcopy(previous.get("missing_comparisons", [])),
        "carried_from_summary_sha256": None
        if previous_summary is None
        else sha256_file(previous_summary),
        "cell_count": len(unions),
        "previous_summary": None
        if previous_summary is None
        else str(previous_summary.resolve()),
        "previous_summary_sha256": None
        if previous_summary is None
        else sha256_file(previous_summary),
    }


def comparison_needs_escalation(comparison: dict[str, Any]) -> bool:
    """Whether either predeclared gate remains unresolved at this budget."""

    return comparison["judge"]["classification"] == "unresolved" or (
        not comparison["comparison_penalty_active"]
        and comparison["raw_loss"]["classification"] == "unresolved"
    )


def validate_judge_manifest(
    manifest: dict[str, Any], *, require_ready: bool
) -> None:
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != "judge"
    ):
        raise RuntimeError("unsupported judge manifest")
    if require_ready and not manifest.get("launch_ready"):
        raise RuntimeError("judge manifest is not marked launch_ready")
    if manifest.get("split") == "holdout" and manifest.get(
        "holdout_locked", True
    ):
        raise RuntimeError("holdout judge remains locked")
    if manifest.get("judge_budget") not in JUDGE_BUDGETS:
        raise RuntimeError("invalid judge budget")
    if manifest.get("replicates") != REPLICATES:
        raise RuntimeError("judge replicate allocation changed")
    unions = manifest.get("unions", [])
    if len(unions) != manifest.get("cell_count"):
        raise RuntimeError("judge cell count is inconsistent")
    state_ids = [union["state_id"] for union in unions]
    if len(state_ids) != len(set(state_ids)):
        raise RuntimeError("judge states are duplicated")
    for union in unions:
        indices = union["candidate_indices"]
        if len(indices) != len(set(indices)):
            raise RuntimeError("judge candidate union contains duplicates")
        expected_union_hash = capture.digest(
            (union["state_id"], np.asarray(indices, dtype=np.int64))
        )
        if union.get("candidate_union_sha256") != expected_union_hash:
            raise RuntimeError("judge candidate union identity changed")
        if union.get("union_origin_budget") != manifest["judge_budget"]:
            raise RuntimeError("judge union budget identity changed")
        if union["fixed_baseline_global_index"] not in indices:
            raise RuntimeError("judge union omits the fixed band baseline")
        if len(union["comparison_seeds"]) != REPLICATES:
            raise RuntimeError("judge comparison seed allocation changed")
        if len(union["band_pilot_seeds"]) not in {0, REPLICATES}:
            raise RuntimeError("judge band-pilot seed allocation changed")


def judge_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    integration_path = Path(manifest["integration_manifest"])
    if (
        sha256_file(integration_path)
        != manifest["integration_manifest_sha256"]
    ):
        raise RuntimeError("integration manifest changed")
    integration = json.loads(integration_path.read_text(encoding="utf-8"))
    actual = runtime_identity(integration)
    if capture.canonical(actual) != capture.canonical(manifest["identity"]):
        raise RuntimeError("judge source identity changed")
    previous = manifest.get("previous_summary")
    if (
        previous is not None
        and sha256_file(previous) != manifest["previous_summary_sha256"]
    ):
        raise RuntimeError("previous judge summary changed")
    return actual


def _judge_paths(out: Path, state_id: str) -> dict[str, Path]:
    return {
        "npz": out / "judge" / f"{state_id}.npz",
        "json": out / "judge" / f"{state_id}.json",
        "failure": out / "records" / f"{state_id}.failure.json",
        "complete": out / "records" / f"{state_id}.complete.json",
    }


def validate_judge_cell(
    manifest: dict[str, Any], out: Path, union: dict[str, Any]
) -> dict[str, Any]:
    state_id = union["state_id"]
    paths = _judge_paths(out, state_id)
    record = json.loads(paths["complete"].read_text(encoding="utf-8"))
    if record.get("manifest_sha256") != manifest_digest(manifest):
        raise RuntimeError(f"{state_id}: judge manifest changed")
    if record.get("state_id") != state_id or record.get("status") not in {
        "succeeded",
        "failed",
    }:
        raise RuntimeError(f"{state_id}: invalid judge completion")
    expected = (
        {"json", "npz"} if record["status"] == "succeeded" else {"failure"}
    )
    if set(record.get("hashes", {})) != expected:
        raise RuntimeError(f"{state_id}: incorrect judge artifact set")
    for role, expected_hash in record["hashes"].items():
        if (
            not paths[role].is_file()
            or sha256_file(paths[role]) != expected_hash
        ):
            raise RuntimeError(f"{state_id}: changed judge {role} artifact")
    return record


def evaluate_judge_matrix(
    state: dict[str, Any],
    candidates: np.ndarray,
    budget: int,
    comparison_seeds: list[int],
    *,
    band_candidate: np.ndarray,
    band_pilot_seeds: list[int],
    band_pilot_budget: int = JUDGE_BUDGETS[0],
) -> dict[str, np.ndarray]:
    """Judge any candidate matrix with common independent RQMC streams."""

    from noisy_acq_quadrature import make_rule, prepare_rule

    candidates = np.asarray(candidates, dtype=np.float64)
    band_candidate = np.asarray(band_candidate, dtype=np.float64).reshape(
        1, -1
    )
    rows: dict[str, list[Any]] = {
        "full_score": [],
        "log_residual": [],
        "log_reduction": [],
        "penalty": [],
        "valid": [],
        "reference_log_residual": [],
        "actual_nodes": [],
    }
    for seed in comparison_seeds:
        rule = make_rule(state["vp"], "stratified_rqmc", budget, seed)
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

    pilot: dict[str, list[Any]] = {
        "full_score": [],
        "log_residual": [],
        "log_reduction": [],
        "reference_log_residual": [],
    }
    for seed in band_pilot_seeds:
        rule = make_rule(
            state["vp"], "stratified_rqmc", band_pilot_budget, seed
        )
        evaluator = prepare_rule(state, rule)
        score = evaluator.score(
            np.array(band_candidate, dtype=np.float64, copy=True),
            diagnostics=True,
        )
        for key in ("full_score", "log_residual", "log_reduction"):
            pilot[key].append(score[key][0])
        pilot["reference_log_residual"].append(score["reference_log_residual"])
    return {
        **{key: np.asarray(value) for key, value in rows.items()},
        **{f"pilot_{key}": np.asarray(value) for key, value in pilot.items()},
    }


def run_judge_cell(
    manifest: dict[str, Any], out: Path, union: dict[str, Any]
) -> dict[str, Any]:
    validate_judge_manifest(manifest, require_ready=True)
    judge_identity(manifest)
    state_id = union["state_id"]
    paths = _judge_paths(out, state_id)
    if paths["complete"].exists():
        return validate_judge_cell(manifest, out, union)
    partial = [
        path
        for key, path in paths.items()
        if key != "complete" and path.exists()
    ]
    if partial:
        raise RuntimeError(f"{state_id}: partial judge artifacts exist")
    try:
        state = capture.restore_capture(union["state"]["snapshot"])
        sieve = np.asarray(state["cand"]["sieve_Xs"], dtype=np.float64)
        indices = np.asarray(union["candidate_indices"], dtype=np.int64)
        candidates = sieve[indices].copy()
        fixed_baseline_position = int(
            np.flatnonzero(indices == union["fixed_baseline_global_index"])[0]
        )
        before = state_digest(state, candidates)
        arrays = evaluate_judge_matrix(
            state,
            candidates,
            manifest["judge_budget"],
            union["comparison_seeds"],
            band_candidate=candidates[[fixed_baseline_position]],
            band_pilot_seeds=union["band_pilot_seeds"],
            band_pilot_budget=union["band_pilot_budget"],
        )
        if before != state_digest(state, candidates):
            raise RuntimeError("judge mutated captured state or RNG")
        arrays["candidate_indices"] = indices
        write_npz(paths["npz"], arrays)
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "succeeded",
            "state_id": state_id,
            "judge_budget": manifest["judge_budget"],
            "comparison_seeds": union["comparison_seeds"],
            "band_pilot_budget": union["band_pilot_budget"],
            "band_pilot_seeds": union["band_pilot_seeds"],
            "actual_nodes": arrays["actual_nodes"].tolist(),
            "candidate_indices": union["candidate_indices"],
            "candidate_provenance": union["candidate_provenance"],
            "fixed_baseline_global_index": union[
                "fixed_baseline_global_index"
            ],
            "fixed_baseline_position": fixed_baseline_position,
            "baseline_by_replicate": union["baseline_by_replicate"],
        }
        write_json(paths["json"], report)
        status = "succeeded"
        hashes = {
            "npz": sha256_file(paths["npz"]),
            "json": sha256_file(paths["json"]),
        }
    except Exception as error:  # noqa: BLE001
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "state_id": state_id,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(paths["failure"], failure)
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
        judge_identity(manifest)
    write_json(paths["complete"], completion)
    return validate_judge_cell(manifest, out, union)


def _state_dimensions(state_id: str) -> dict[str, str]:
    """Decode the capture allocation dimensions from a stable state id."""

    trajectory, checkpoint = state_id.rsplit("_", 1)
    label, seed = trajectory.rsplit("_seed", 1)
    return {
        "state_id": state_id,
        "trajectory": trajectory,
        "label": label,
        "seed": seed,
        "checkpoint": checkpoint,
    }


def _comparison_counts(
    comparisons: list[dict[str, Any]], scheduled: int
) -> dict[str, Any]:
    score = {
        key: sum(
            item["judge"]["classification"] == key for item in comparisons
        )
        for key in ("beneficial", "harmful", "practical_tie", "unresolved")
    }
    score["failed_or_missing"] = int(scheduled - len(comparisons))
    raw_gate = {
        key: sum(
            not item["comparison_penalty_active"]
            and item["raw_loss"]["classification"] == key
            for item in comparisons
        )
        for key in ("material_loss", "no_material_loss", "unresolved")
    }
    raw_gate["not_applicable_penalty_active"] = sum(
        item["comparison_penalty_active"] for item in comparisons
    )
    raw_gate["failed_or_missing"] = score["failed_or_missing"]
    denominator = float(scheduled)
    return {
        "scheduled": int(scheduled),
        "observed": len(comparisons),
        "score_counts": score,
        "raw_loss_gate_counts": raw_gate,
        "fractions_of_scheduled": {
            "harmful": score["harmful"] / denominator,
            "unresolved": score["unresolved"] / denominator,
            "failed_or_missing": score["failed_or_missing"] / denominator,
            "raw_material_loss": raw_gate["material_loss"] / denominator,
            "raw_unresolved": raw_gate["unresolved"] / denominator,
        },
    }


def grouped_judge_summary(
    integration: dict[str, Any], comparisons: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize every setting with full scheduled denominators."""

    allocated = {
        (
            cell["state_id"],
            cell["method"],
            cell["budget"],
            cell["replicate"],
        )
        for cell in integration["cells"]
    }
    observed = [
        (
            item["cell"]["state_id"],
            item["cell"]["method"],
            item["cell"]["budget"],
            item["cell"]["replicate"],
        )
        for item in comparisons
    ]
    if len(observed) != len(set(observed)) or not set(observed) <= allocated:
        raise RuntimeError("judge comparisons are duplicated or unallocated")
    expected_ids = [state["state_id"] for state in integration["states"]]
    expected_ids.extend(
        state["state_id"] for state in integration["missing_states"]
    )
    if len(expected_ids) != integration["expected_state_count"] or len(
        expected_ids
    ) != len(set(expected_ids)):
        raise RuntimeError("expected state allocation is inconsistent")
    dimensions = {
        state_id: _state_dimensions(state_id) for state_id in expected_ids
    }

    def rows_for_dimension(name: str) -> list[dict[str, Any]]:
        values = sorted({item[name] for item in dimensions.values()})
        rows = []
        for setting in integration["settings"]:
            for value in values:
                state_ids = {
                    state_id
                    for state_id, item in dimensions.items()
                    if item[name] == value
                }
                selected = [
                    item
                    for item in comparisons
                    if item["cell"]["method"] == setting["method"]
                    and item["cell"]["budget"] == setting["budget"]
                    and item["cell"]["state_id"] in state_ids
                ]
                rows.append(
                    {
                        **setting,
                        name: value,
                        **_comparison_counts(
                            selected, len(state_ids) * REPLICATES
                        ),
                    }
                )
        return rows

    by_setting = []
    for setting in integration["settings"]:
        selected = [
            item
            for item in comparisons
            if item["cell"]["method"] == setting["method"]
            and item["cell"]["budget"] == setting["budget"]
        ]
        by_setting.append(
            {
                **setting,
                **_comparison_counts(
                    selected, integration["expected_state_count"] * REPLICATES
                ),
            }
        )
    return {
        "by_setting": by_setting,
        "by_trajectory": rows_for_dimension("trajectory"),
        "by_label": rows_for_dimension("label"),
        "by_checkpoint": rows_for_dimension("checkpoint"),
    }


def judge_summary(manifest: dict[str, Any], out: Path) -> dict[str, Any]:
    from noisy_acq_quadrature import (
        baseline_gain_resolution,
        judge_paired,
        practical_band,
        raw_loss_assessment,
    )

    def logmeanexp(values: np.ndarray, axis: int | None = None):
        values = np.asarray(values, dtype=np.float64)
        count = values.size if axis is None else values.shape[axis]
        return np.logaddexp.reduce(values, axis=axis) - np.log(count)

    integration = json.loads(
        Path(manifest["integration_manifest"]).read_text(encoding="utf-8")
    )
    previous = None
    if manifest.get("previous_summary") is not None:
        previous = json.loads(
            Path(manifest["previous_summary"]).read_text(encoding="utf-8")
        )
    previous_by_tag = (
        {}
        if previous is None
        else {
            item["integration_cell_tag"]: item
            for item in previous["comparisons"]
        }
    )
    comparisons = copy.deepcopy(manifest.get("carried_comparisons", []))
    failed_states = []
    missing_comparisons = copy.deepcopy(
        manifest.get("carried_missing_comparisons", [])
    )
    unions = {item["state_id"]: item for item in manifest["unions"]}
    for state_id, union in unions.items():
        record = validate_judge_cell(manifest, out, union)
        if record["status"] != "succeeded":
            failed_states.append(state_id)
            continue
        paths = _judge_paths(out, state_id)
        with np.load(paths["npz"], allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        indices = arrays["candidate_indices"]
        if union.get("frozen_band") is None:
            pilot_gain = baseline_gain_resolution(
                arrays["pilot_log_reduction"],
                arrays["pilot_reference_log_residual"],
            )
            log_reference = float(
                logmeanexp(arrays["pilot_reference_log_residual"])
            )
            log_baseline = float(logmeanexp(arrays["pilot_log_residual"]))
            band = practical_band(
                log_reference,
                log_baseline,
                baseline_gain_resolved=pilot_gain["resolved"],
            )
            band["pilot_gain"] = pilot_gain
        else:
            band = copy.deepcopy(union["frozen_band"])
        mean_log_union_scores = np.mean(arrays["full_score"], axis=0)
        pooled_union_scores = logmeanexp(
            arrays["log_residual"], axis=0
        ) + np.mean(arrays["penalty"], axis=0)
        union_valid = np.all(arrays["valid"], axis=0)
        mean_log_selectable = np.where(
            union_valid & np.isfinite(mean_log_union_scores),
            mean_log_union_scores,
            np.inf,
        )
        pooled_selectable = np.where(
            union_valid & np.isfinite(pooled_union_scores),
            pooled_union_scores,
            np.inf,
        )
        mean_log_best_position = (
            None
            if not np.any(np.isfinite(mean_log_selectable))
            else int(np.argmin(mean_log_selectable))
        )
        pooled_best_position = (
            None
            if not np.any(np.isfinite(pooled_selectable))
            else int(np.argmin(pooled_selectable))
        )
        mean_log_best_index = (
            None
            if mean_log_best_position is None
            else int(indices[mean_log_best_position])
        )
        pooled_best_index = (
            None
            if pooled_best_position is None
            else int(indices[pooled_best_position])
        )
        cells = [
            cell
            for cell in integration["cells"]
            if cell["state_id"] == state_id
        ]
        for cell in cells:
            tag = cell_tag(cell)
            selected = union["treatment_by_tag"].get(tag)
            if selected is None:
                continue
            position = int(np.flatnonzero(indices == selected)[0])
            baseline_index = union["baseline_by_replicate"].get(
                str(cell["replicate"])
            )
            if baseline_index is None:
                missing_comparisons.append(
                    {"cell": cell, "reason": "paired_baseline_missing"}
                )
                continue
            baseline_position = int(
                np.flatnonzero(indices == baseline_index)[0]
            )
            arm_scores = arrays["full_score"][:, position]
            reference_scores = arrays["full_score"][:, baseline_position]
            arm_pooled = float(
                logmeanexp(arrays["log_residual"][:, position])
                + np.mean(arrays["penalty"][:, position])
            )
            reference_pooled = float(
                logmeanexp(arrays["log_residual"][:, baseline_position])
                + np.mean(arrays["penalty"][:, baseline_position])
            )
            prior = previous_by_tag.get(tag)
            judged = judge_paired(
                arm_scores,
                reference_scores,
                band["eps_F"],
                previous=None if prior is None else prior["judge"],
                replicate_seeds=union["comparison_seeds"],
                pooled_difference=arm_pooled - reference_pooled,
            )
            raw_loss = raw_loss_assessment(
                arrays["log_reduction"][:, position],
                arrays["log_reduction"][:, baseline_position],
                arrays["reference_log_residual"],
                previous=None if prior is None else prior["raw_loss"],
            )
            comparisons.append(
                {
                    "integration_cell_tag": tag,
                    "judge_budget": manifest["judge_budget"],
                    "cell": cell,
                    "selected_global_index": int(selected),
                    "baseline_global_index": int(baseline_index),
                    "fixed_band_baseline_global_index": union[
                        "fixed_baseline_global_index"
                    ],
                    "band": band,
                    "judge": judged,
                    "raw_loss": raw_loss,
                    "penalty": {
                        "arm": arrays["penalty"][:, position].tolist(),
                        "baseline": arrays["penalty"][
                            :, baseline_position
                        ].tolist(),
                        "arm_active": bool(
                            np.any(arrays["penalty"][:, position] > 0)
                        ),
                        "baseline_active": bool(
                            np.any(arrays["penalty"][:, baseline_position] > 0)
                        ),
                    },
                    "comparison_penalty_active": bool(
                        np.any(arrays["penalty"][:, position] > 0)
                        or np.any(arrays["penalty"][:, baseline_position] > 0)
                    ),
                    "pooled_log_reduction": {
                        "arm": float(
                            logmeanexp(arrays["log_reduction"][:, position])
                        ),
                        "baseline": float(
                            logmeanexp(
                                arrays["log_reduction"][:, baseline_position]
                            )
                        ),
                    },
                    "pooled_union_best_global_index": pooled_best_index,
                    "mean_log_union_best_global_index": mean_log_best_index,
                    "union_best_disagreement": bool(
                        pooled_best_index != mean_log_best_index
                    ),
                    "pooled_score_regret_within_union": None
                    if pooled_best_position is None
                    or not np.isfinite(pooled_union_scores[position])
                    else float(
                        pooled_union_scores[position]
                        - pooled_union_scores[pooled_best_position]
                    ),
                    "assessed_union": {
                        "candidate_union_sha256": union[
                            "candidate_union_sha256"
                        ],
                        "judge_budget": manifest["judge_budget"],
                        "candidate_count": len(indices),
                    },
                }
            )
    counts = {
        key: sum(
            item["judge"]["classification"] == key for item in comparisons
        )
        for key in ("beneficial", "harmful", "practical_tie", "unresolved")
    }
    scheduled = integration["scheduled_cell_count_including_missing_states"]
    counts["failed_or_missing"] = scheduled - len(comparisons)
    raw_loss_counts = {
        key: sum(
            item["raw_loss"]["classification"] == key for item in comparisons
        )
        for key in ("material_loss", "no_material_loss", "unresolved")
    }
    raw_loss_counts["failed_or_missing"] = scheduled - len(comparisons)
    raw_loss_gate_counts = {
        key: sum(
            not item["comparison_penalty_active"]
            and item["raw_loss"]["classification"] == key
            for item in comparisons
        )
        for key in ("material_loss", "no_material_loss", "unresolved")
    }
    raw_loss_gate_counts["not_applicable_penalty_active"] = sum(
        item["comparison_penalty_active"] for item in comparisons
    )
    raw_loss_gate_counts["failed_or_missing"] = scheduled - len(comparisons)
    return {
        "schema_version": SCHEMA_VERSION,
        "integration_manifest_sha256": manifest_digest(integration),
        "judge_manifest_sha256": manifest_digest(manifest),
        "judge_budget": manifest["judge_budget"],
        "counts": counts,
        "raw_loss_counts": raw_loss_counts,
        "raw_loss_gate_counts": raw_loss_gate_counts,
        "grouped": grouped_judge_summary(integration, comparisons),
        "failed_states": failed_states,
        "missing_selections": manifest["missing_selections"],
        "missing_comparisons": missing_comparisons,
        "comparisons": comparisons,
    }


def run_judge_controller(
    manifest_path: Path, manifest: dict[str, Any], out: Path
) -> int:
    validate_judge_manifest(manifest, require_ready=True)
    identity = judge_identity(manifest)
    out.mkdir(parents=True, exist_ok=True)
    launch = {"manifest": manifest, "identity": identity}
    launch_path = out / "launch.json"
    if launch_path.exists():
        if json.loads(
            launch_path.read_text(encoding="utf-8")
        ) != capture.canonical(launch):
            raise RuntimeError("judge output belongs to another manifest")
    else:
        write_json(launch_path, launch)
    failures = 0
    for union in manifest["unions"]:
        state_id = union["state_id"]
        completion = _judge_paths(out, state_id)["complete"]
        if completion.exists():
            record = validate_judge_cell(manifest, out, union)
        else:
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "judge-cell",
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
                    f"{state_id}: judge worker infrastructure failed; see {log}"
                )
            record = validate_judge_cell(manifest, out, union)
        failures += record["status"] == "failed"
        print(f"{record['status'].upper()} {state_id}", flush=True)
    write_json(out / "summary.json", judge_summary(manifest, out))
    return 1 if failures else 0


def run_controller(
    manifest_path: Path, manifest: dict[str, Any], out: Path, limit: int | None
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
            raise RuntimeError(
                "output directory belongs to another E2 manifest"
            )
    else:
        write_json(launch_path, launch)
    failures = 0
    baseline_cells = (
        manifest["baseline_cells"][:limit]
        if limit
        else manifest["baseline_cells"]
    )
    for cell in baseline_cells:
        tag = baseline_tag(cell)
        complete = _baseline_paths(out, tag)["complete"]
        if complete.exists():
            record = validate_baseline_cell(manifest, out, cell)
            failures += record["status"] == "failed"
            print(f"[resume] {record['status']} {tag}", flush=True)
            continue
        partial = [
            path
            for key, path in _baseline_paths(out, tag).items()
            if key != "complete" and path.exists()
        ]
        if partial:
            raise RuntimeError(f"{tag}: incomplete prior baseline attempt")
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "baseline-cell",
            "--manifest",
            str(manifest_path.resolve()),
            "--out",
            str(out.resolve()),
            "--tag",
            tag,
        ]
        log = out / "logs" / f"{tag}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(
                f"{tag}: baseline worker infrastructure failed; see {log}"
            )
        record = validate_baseline_cell(manifest, out, cell)
        failures += record["status"] == "failed"
        print(f"{record['status'].upper()} {tag}", flush=True)
        write_json(out / "inventory.json", inventory(manifest, out))
    cells = manifest["cells"][:limit] if limit else manifest["cells"]
    for cell in cells:
        tag = cell_tag(cell)
        complete = _cell_paths(out, tag)["complete"]
        if complete.exists():
            record = validate_cell(manifest, out, cell)
            failures += record["status"] == "failed"
            print(f"[resume] {record['status']} {tag}", flush=True)
            continue
        partial = [
            path
            for key, path in _cell_paths(out, tag).items()
            if key != "complete" and path.exists()
        ]
        if partial:
            raise RuntimeError(f"{tag}: incomplete prior attempt")
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "cell",
            "--manifest",
            str(manifest_path.resolve()),
            "--out",
            str(out.resolve()),
            "--tag",
            tag,
        ]
        log = out / "logs" / f"{tag}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(
                f"{tag}: worker infrastructure failed; see {log}"
            )
        record = validate_cell(manifest, out, cell)
        failures += record["status"] == "failed"
        print(f"{record['status'].upper()} {tag}", flush=True)
        write_json(out / "inventory.json", inventory(manifest, out))
    write_json(out / "summary.json", summarize(manifest, out))
    return 1 if failures else 0


def _load_manifest(path: Path, kind: str = "integration") -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if kind == "integration":
        validate_manifest(manifest, require_ready=False)
    elif kind == "judge":
        validate_judge_manifest(manifest, require_ready=False)
    else:
        raise ValueError(f"unknown manifest kind {kind}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--capture-manifest", type=Path, required=True)
    prepare.add_argument("--captures", type=Path, required=True)
    prepare.add_argument(
        "--stage", choices=("panel", "full", "holdout"), required=True
    )
    prepare.add_argument("--selection", type=Path)
    prepare.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--limit", type=int)
    cell = sub.add_parser("cell")
    cell.add_argument("--manifest", type=Path, required=True)
    cell.add_argument("--out", type=Path, required=True)
    cell.add_argument("--tag", required=True)
    baseline = sub.add_parser("baseline-cell")
    baseline.add_argument("--manifest", type=Path, required=True)
    baseline.add_argument("--out", type=Path, required=True)
    baseline.add_argument("--tag", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--manifest", type=Path, required=True)
    inv.add_argument("--out", type=Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    prepare_judge = sub.add_parser("prepare-judge")
    prepare_judge.add_argument(
        "--integration-manifest", type=Path, required=True
    )
    prepare_judge.add_argument("--results", type=Path, required=True)
    prepare_judge.add_argument(
        "--budget", type=int, choices=JUDGE_BUDGETS, required=True
    )
    prepare_judge.add_argument("--previous-summary", type=Path)
    prepare_judge.add_argument("--out", type=Path, required=True)
    judge = sub.add_parser("judge")
    judge.add_argument("--manifest", type=Path, required=True)
    judge.add_argument("--out", type=Path, required=True)
    judge_cell = sub.add_parser("judge-cell")
    judge_cell.add_argument("--manifest", type=Path, required=True)
    judge_cell.add_argument("--out", type=Path, required=True)
    judge_cell.add_argument("--state-id", required=True)
    judge_summary_parser = sub.add_parser("judge-summary")
    judge_summary_parser.add_argument("--manifest", type=Path, required=True)
    judge_summary_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        manifest = prepare_manifest(
            args.capture_manifest, args.captures, args.stage, args.selection
        )
        write_json(args.out, manifest)
        print(f"Wrote review draft {args.out}; set launch_ready after review.")
        return 0
    if args.command == "prepare-judge":
        manifest = prepare_judge_manifest(
            args.integration_manifest,
            args.results,
            args.budget,
            args.previous_summary,
        )
        write_json(args.out, manifest)
        print(f"Wrote review draft {args.out}; set launch_ready after review.")
        return 0
    if args.command in {"judge", "judge-cell", "judge-summary"}:
        manifest = _load_manifest(args.manifest, "judge")
        if args.command == "judge":
            return run_judge_controller(args.manifest, manifest, args.out)
        if args.command == "judge-cell":
            unions = {item["state_id"]: item for item in manifest["unions"]}
            if args.state_id not in unions:
                raise RuntimeError(f"unknown judge state {args.state_id}")
            run_judge_cell(manifest, args.out, unions[args.state_id])
            return 0
        report = judge_summary(manifest, args.out)
        write_json(args.out / "summary.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    manifest = _load_manifest(args.manifest)
    if args.command == "run":
        return run_controller(args.manifest, manifest, args.out, args.limit)
    if args.command == "cell":
        cells = {cell_tag(item): item for item in manifest["cells"]}
        if args.tag not in cells:
            raise RuntimeError(f"unknown cell tag {args.tag}")
        run_cell(manifest, args.out, cells[args.tag])
        return 0
    if args.command == "baseline-cell":
        cells = {
            baseline_tag(item): item for item in manifest["baseline_cells"]
        }
        if args.tag not in cells:
            raise RuntimeError(f"unknown baseline cell tag {args.tag}")
        run_baseline_cell(manifest, args.out, cells[args.tag])
        return 0
    if args.command == "inventory":
        print(
            json.dumps(inventory(manifest, args.out), indent=2, sort_keys=True)
        )
        return 0
    if args.command == "summarize":
        report = summarize(manifest, args.out)
        write_json(args.out / "summary.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)
