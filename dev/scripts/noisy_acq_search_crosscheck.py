"""Independent ordinary-MC crosschecks for selected E3 search comparisons.

Development allocations are frozen from the completed E3 judge summary: for
each captured state they include the smallest and largest paired mean, every
comparison at least ten practical bands from zero, and every confidently
classified raw material loss.  Holdout allocations accept only explicitly
named comparisons.  Exact selected-coordinate pairs are deduplicated before
eight common-node ordinary-MC replicates are evaluated at 32768 nodes and,
only while a gate remains unresolved, 65536 nodes.

The crosscheck is diagnostic.  It records agreement with the E3 RQMC judge
and never changes the frozen search verdict.
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

import noisy_acq_crosscheck as ordinary
import noisy_acq_experiment as capture
import noisy_acq_integration as integration
import noisy_acq_search_experiment as search_experiment
import numpy as np

SCHEMA_VERSION = 1
BUDGETS = (32768, 65536)
REPLICATES = 8
MASTER_SEED = 2026091702


def sha256_file(path: Path | str) -> str:
    return integration.sha256_file(path)


def write_json(path: Path | str, value: Any) -> None:
    integration.write_json(path, value)


def write_npz(path: Path | str, arrays: dict[str, np.ndarray]) -> None:
    integration.write_npz(path, arrays)


def manifest_digest(manifest: dict[str, Any]) -> str:
    return integration.manifest_digest(manifest)


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        SCRIPTS / "noisy_acq_crosscheck.py",
        SCRIPTS / "noisy_acq_search_experiment.py",
        SCRIPTS / "noisy_acq_integration.py",
        SCRIPTS / "noisy_acq_quadrature.py",
        SCRIPTS / "noisy_acq_experiment.py",
    ]
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths
    }


def comparison_needs_escalation(comparison: dict[str, Any]) -> bool:
    return ordinary.comparison_needs_escalation(comparison)


def _finite_mean(comparison: dict[str, Any]) -> float | None:
    value = comparison.get("judge", {}).get("mean_difference")
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def select_development_tags(
    comparisons: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[str]]]:
    """Apply the predeclared E3 extrema and catastrophe policy."""
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
            reasons.setdefault(low["comparison_tag"], []).append(
                "state_min_mean_difference"
            )
            reasons.setdefault(high["comparison_tag"], []).append(
                "state_max_mean_difference"
            )
        for item in state_items:
            mean = _finite_mean(item)
            tag = item["comparison_tag"]
            eps_f = float(item["band"]["eps_F"])
            if mean is not None and abs(mean) >= 10 * eps_f:
                reasons.setdefault(tag, []).append("ten_practical_bands")
            if item["raw_loss"]["classification"] == "material_loss":
                reasons.setdefault(tag, []).append(
                    "confident_raw_material_loss"
                )
    return sorted(reasons), reasons


def _row_bytes(row: np.ndarray) -> bytes:
    return np.ascontiguousarray(row, dtype="<f8").reshape(-1).tobytes()


def _pair_id(state_id: str, arm: np.ndarray, baseline: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(state_id.encode("utf-8"))
    digest.update(b"\0arm\0")
    digest.update(_row_bytes(arm))
    digest.update(b"\0baseline\0")
    digest.update(_row_bytes(baseline))
    return "pair_" + digest.hexdigest()


def _load_initial_union(
    search_manifest: dict[str, Any], search_out: Path, state_id: str
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    budget = int(search_manifest["initial_judge_budget"])
    tag = search_experiment.judge_tag(state_id, budget)
    paths = search_experiment._paths(search_out, tag, "judge")
    if not paths["complete"].is_file():
        if search_experiment._partial(paths):
            raise RuntimeError(f"{state_id}: partial initial judge artifacts")
        raise RuntimeError(f"{state_id}: initial judge completion is missing")
    terminal = search_experiment._validate_terminal(
        search_manifest,
        paths,
        search_experiment._judge_cell(state_id, budget, None),
    )
    if terminal["status"] != "succeeded":
        raise RuntimeError(f"{state_id}: initial judge did not succeed")
    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    union = report["union"]
    with np.load(paths["npz"], allow_pickle=False) as stored:
        candidates = np.asarray(stored["candidates"], dtype=np.float64).copy()
    candidate_hash = hashlib.sha256(
        np.ascontiguousarray(candidates, dtype="<f8").tobytes()
    ).hexdigest()
    row_ids = list(union["candidate_row_ids"])
    if (
        candidate_hash != union["candidate_union_sha256"]
        or len(row_ids) != len(candidates)
        or len(row_ids) != len(set(row_ids))
    ):
        raise RuntimeError(f"{state_id}: inconsistent initial judge union")
    for row_id, row in zip(row_ids, candidates, strict=True):
        if row_id != search_experiment._row_id(row):
            raise RuntimeError(f"{state_id}: changed initial judge coordinate")
    artifact = {
        "state_id": state_id,
        "budget": budget,
        "union_id": union["union_id"],
        "candidate_union_sha256": candidate_hash,
        "paths": {
            role: str(paths[role].resolve())
            for role in ("npz", "json", "complete")
        },
        "hashes": {
            role: sha256_file(paths[role])
            for role in ("npz", "json", "complete")
        },
    }
    return union, candidates, artifact


def _deduplicate_coordinate_pairs(
    selected: list[dict[str, Any]],
    reasons: dict[str, list[str]],
    search_manifest: dict[str, Any],
    search_out: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    selected_by_state: dict[str, list[dict[str, Any]]] = {}
    for item in selected:
        selected_by_state.setdefault(item["cell"]["state_id"], []).append(item)
    pairs: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for state_id, state_items in selected_by_state.items():
        union, candidates, artifact = _load_initial_union(
            search_manifest, search_out, state_id
        )
        artifacts[state_id] = artifact
        positions = {
            row_id: index
            for index, row_id in enumerate(union["candidate_row_ids"])
        }
        for item in state_items:
            arm_id = item["selected_row_id"]
            baseline_id = item["baseline_row_id"]
            if arm_id not in positions or baseline_id not in positions:
                raise RuntimeError(
                    f"{item['comparison_tag']}: selected coordinate is absent "
                    "from the initial judge union"
                )
            arm = candidates[positions[arm_id]].copy()
            baseline = candidates[positions[baseline_id]].copy()
            pair_id = _pair_id(state_id, arm, baseline)
            pair = pairs.setdefault(
                pair_id,
                {
                    "pair_id": pair_id,
                    "state_id": state_id,
                    "arm_row_id": search_experiment._row_id(arm),
                    "baseline_row_id": search_experiment._row_id(baseline),
                    "arm": arm.tolist(),
                    "baseline": baseline.tolist(),
                    "comparison_tags": [],
                    "selection_reasons": [],
                    "source_comparisons": [],
                    "band": copy.deepcopy(item["band"]),
                },
            )
            if capture.canonical(pair["band"]) != capture.canonical(
                item["band"]
            ):
                raise RuntimeError(
                    f"{state_id}: practical band changed within state"
                )
            tag = item["comparison_tag"]
            pair["comparison_tags"].append(tag)
            pair["selection_reasons"].extend(reasons[tag])
            pair["source_comparisons"].append(copy.deepcopy(item))
    for pair in pairs.values():
        pair["comparison_tags"].sort()
        pair["selection_reasons"] = sorted(set(pair["selection_reasons"]))
    return sorted(pairs.values(), key=lambda item: item["pair_id"]), artifacts


def _build_union(
    state_id: str,
    pairs: list[dict[str, Any]],
    descriptor: dict[str, Any],
    artifact: dict[str, Any],
    budget: int,
) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    row_ids: list[str] = []
    lookup: dict[str, int] = {}
    for pair in pairs:
        for role in ("arm", "baseline"):
            row = np.asarray(pair[role], dtype=np.float64)
            row_id = search_experiment._row_id(row)
            if row_id not in lookup:
                lookup[row_id] = len(rows)
                rows.append(row.copy())
                row_ids.append(row_id)
    candidates = np.vstack(rows).astype(np.float64, copy=False)
    candidate_hash = hashlib.sha256(
        np.ascontiguousarray(candidates, dtype="<f8").tobytes()
    ).hexdigest()
    return {
        "state_id": state_id,
        "state": descriptor,
        "candidate_row_ids": row_ids,
        "candidates": candidates.tolist(),
        "candidate_union_sha256": candidate_hash,
        "pairs": pairs,
        "initial_judge_artifact": artifact,
        "replicate_seeds": [
            integration.derive_seed(
                MASTER_SEED,
                state_id,
                budget,
                replicate,
                "search_ordinary_mc_crosscheck",
            )
            for replicate in range(REPLICATES)
        ],
    }


def prepare_manifest(
    judge_summary_path: Path,
    search_manifest_path: Path,
    search_out: Path,
    budget: int,
    previous_summary_path: Path | None,
    explicit_tags: list[str],
) -> dict[str, Any]:
    """Prepare a review-locked E3 ordinary-MC crosscheck allocation."""
    source_summary = json.loads(judge_summary_path.read_text(encoding="utf-8"))
    search_manifest = json.loads(
        search_manifest_path.read_text(encoding="utf-8")
    )
    search_experiment.validate_manifest(search_manifest, require_ready=True)
    search_identity = search_experiment.runtime_identity(search_manifest)
    if source_summary.get("kind") != "search_judge_summary":
        raise RuntimeError("source is not an E3 judge summary")
    if source_summary.get("search_manifest_sha256") != manifest_digest(
        search_manifest
    ):
        raise RuntimeError("judge summary belongs to another E3 manifest")
    if source_summary.get("source_identity_sha256") != capture.digest(
        search_manifest["identity"]
    ):
        raise RuntimeError("judge summary has another frozen source identity")
    budget = int(budget)
    if budget not in BUDGETS:
        raise RuntimeError(f"crosscheck budget must be one of {BUDGETS}")
    source_comparisons = source_summary["comparisons"]
    known = {item["comparison_tag"]: item for item in source_comparisons}
    if len(known) != len(source_comparisons):
        raise RuntimeError(
            "source judge summary contains duplicate comparisons"
        )
    split = search_manifest["split"]
    if split == "development":
        if explicit_tags:
            raise RuntimeError(
                "explicit comparison tags are reserved for holdout crosschecks"
            )
        tags, reasons = select_development_tags(source_comparisons)
        policy = "development_extrema_and_catastrophes"
    elif split == "holdout":
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
    else:
        raise RuntimeError(f"unsupported E3 split {split!r}")
    selected = [known[tag] for tag in tags]
    initial_pairs, artifacts = _deduplicate_coordinate_pairs(
        selected, reasons, search_manifest, search_out
    )

    previous = None
    carried: list[dict[str, Any]] = []
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
        if (
            previous.get("kind") != "search_ordinary_mc_crosscheck_summary"
            or previous.get("search_manifest_sha256")
            != sha256_file(search_manifest_path)
            or previous.get("automatic_override") is not False
        ):
            raise RuntimeError("previous crosscheck provenance changed")
        previous_pairs = {
            item["pair_id"]: item for item in previous["comparisons"]
        }
        previous_failed = {
            item["pair_id"]: item for item in previous["failed_pairs"]
        }
        pending_ids = {
            pair_id
            for pair_id, item in previous_pairs.items()
            if comparison_needs_escalation(item)
        }
        available_ids = {pair["pair_id"] for pair in initial_pairs}
        if set(previous_pairs) | set(previous_failed) != available_ids:
            raise RuntimeError("previous crosscheck pair allocation changed")
        if set(previous_pairs) & set(previous_failed):
            raise RuntimeError("previous crosscheck duplicated failed pairs")
        source_pairs = {pair["pair_id"]: pair for pair in initial_pairs}
        for pair_id, prior in {**previous_pairs, **previous_failed}.items():
            source_pair = source_pairs[pair_id]
            for key in ("state_id", "comparison_tags", "selection_reasons"):
                if prior.get(key) != source_pair[key]:
                    raise RuntimeError(
                        "previous crosscheck pair provenance changed"
                    )
            if pair_id in previous_pairs and (
                prior.get("arm_row_id") != source_pair["arm_row_id"]
                or prior.get("baseline_row_id")
                != source_pair["baseline_row_id"]
                or capture.canonical(prior.get("band"))
                != capture.canonical(source_pair["band"])
            ):
                raise RuntimeError(
                    "previous crosscheck coordinate or band changed"
                )
        expected_by_state: dict[str, int] = {}
        for pair in initial_pairs:
            expected_by_state[pair["state_id"]] = (
                expected_by_state.get(pair["state_id"], 0) + 1
            )
        if (
            previous.get("scheduled_pair_count") != len(initial_pairs)
            or previous.get("scheduled_pairs_by_state") != expected_by_state
        ):
            raise RuntimeError("previous crosscheck denominator changed")
        carried = [
            copy.deepcopy(item)
            for pair_id, item in previous_pairs.items()
            if pair_id not in pending_ids
        ]
        pairs = [
            pair for pair in initial_pairs if pair["pair_id"] in pending_ids
        ]
        scheduled_pair_count = int(previous["scheduled_pair_count"])
        scheduled_by_state = copy.deepcopy(
            previous["scheduled_pairs_by_state"]
        )
    else:
        if previous_summary_path is not None:
            raise RuntimeError(
                "32768-node crosscheck cannot use a previous summary"
            )
        pairs = initial_pairs
        scheduled_pair_count = len(initial_pairs)
        scheduled_by_state: dict[str, int] = {}
        for pair in initial_pairs:
            scheduled_by_state[pair["state_id"]] = (
                scheduled_by_state.get(pair["state_id"], 0) + 1
            )

    descriptors = {
        item["state_id"]: item for item in search_manifest["states"]
    }
    unions = []
    for state_id in sorted({pair["state_id"] for pair in pairs}):
        if state_id not in descriptors:
            raise RuntimeError(f"crosscheck state is unavailable: {state_id}")
        state_pairs = [pair for pair in pairs if pair["state_id"] == state_id]
        unions.append(
            _build_union(
                state_id,
                state_pairs,
                descriptors[state_id],
                artifacts[state_id],
                budget,
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "search_ordinary_mc_crosscheck",
        "purpose": "E1 ordinary-MC diagnostic crosscheck for E3 search",
        "launch_ready": False,
        "split": split,
        "holdout_locked": split == "holdout",
        "budget": budget,
        "replicates": REPLICATES,
        "master_seed": MASTER_SEED,
        "selection_policy": policy,
        "explicit_comparison_tags": sorted(set(explicit_tags)),
        "frozen_comparison_tags": tags,
        "source_judge_summary": str(judge_summary_path.resolve()),
        "source_judge_summary_sha256": sha256_file(judge_summary_path),
        "search_manifest": str(search_manifest_path.resolve()),
        "search_manifest_sha256": sha256_file(search_manifest_path),
        "search_manifest_digest": manifest_digest(search_manifest),
        "search_output": str(search_out.resolve()),
        "search_identity": search_identity,
        "source_hashes": source_hashes(),
        "initial_judge_artifacts": artifacts,
        "unions": unions,
        "state_cell_count": len(unions),
        "scheduled_pair_count": scheduled_pair_count,
        "scheduled_pairs_by_state": scheduled_by_state,
        "carried_comparisons": carried,
        "carried_failed_pairs": (
            [] if previous is None else copy.deepcopy(previous["failed_pairs"])
        ),
        "carried_failed_states": (
            []
            if previous is None
            else copy.deepcopy(previous["failed_states"])
        ),
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
        or manifest.get("kind") != "search_ordinary_mc_crosscheck"
    ):
        raise RuntimeError("unsupported E3 ordinary-MC crosscheck manifest")
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
        or manifest.get("master_seed") != MASTER_SEED
    ):
        raise RuntimeError("crosscheck numerical allocation changed")
    if manifest["budget"] == BUDGETS[0]:
        if manifest.get("previous_summary") is not None or any(
            manifest.get(key)
            for key in (
                "carried_comparisons",
                "carried_failed_pairs",
                "carried_failed_states",
            )
        ):
            raise RuntimeError("initial crosscheck cannot carry prior results")
    elif manifest.get("previous_summary") is None:
        raise RuntimeError("escalated crosscheck requires its prior summary")
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
        if not tags or tags != sorted(set(explicit)):
            raise RuntimeError(
                "holdout crosscheck requires its explicit frozen tags"
            )
        if manifest.get("selection_policy") != "explicit_holdout_only":
            raise RuntimeError("holdout crosscheck policy changed")
    unions = manifest.get("unions", [])
    if len(unions) != manifest.get("state_cell_count"):
        raise RuntimeError("crosscheck state allocation is inconsistent")
    state_ids = [union["state_id"] for union in unions]
    if len(state_ids) != len(set(state_ids)):
        raise RuntimeError("crosscheck state cells are duplicated")
    pair_ids: list[str] = []
    represented_tags: set[str] = set()
    represented_by_state: dict[str, int] = {}
    artifacts = manifest.get("initial_judge_artifacts", {})
    for union in unions:
        candidates = np.asarray(union["candidates"], dtype=np.float64)
        row_ids = union["candidate_row_ids"]
        if (
            candidates.ndim != 2
            or len(row_ids) != len(candidates)
            or len(row_ids) != len(set(row_ids))
        ):
            raise RuntimeError("crosscheck candidate union is inconsistent")
        expected = hashlib.sha256(
            np.ascontiguousarray(candidates, dtype="<f8").tobytes()
        ).hexdigest()
        if union["candidate_union_sha256"] != expected:
            raise RuntimeError("crosscheck candidate union identity changed")
        if any(
            row_id != search_experiment._row_id(row)
            for row_id, row in zip(row_ids, candidates, strict=True)
        ):
            raise RuntimeError("crosscheck candidate row identity changed")
        if len(union["replicate_seeds"]) != REPLICATES:
            raise RuntimeError("crosscheck replicate allocation changed")
        expected_seeds = [
            integration.derive_seed(
                MASTER_SEED,
                union["state_id"],
                manifest["budget"],
                replicate,
                "search_ordinary_mc_crosscheck",
            )
            for replicate in range(REPLICATES)
        ]
        if union["replicate_seeds"] != expected_seeds:
            raise RuntimeError("crosscheck replicate seed changed")
        if union.get("initial_judge_artifact") != artifacts.get(
            union["state_id"]
        ):
            raise RuntimeError("crosscheck initial judge binding changed")
        for pair in union["pairs"]:
            if pair["state_id"] != union["state_id"]:
                raise RuntimeError("crosscheck pair belongs to another state")
            if (
                pair["arm_row_id"] not in row_ids
                or pair["baseline_row_id"] not in row_ids
            ):
                raise RuntimeError("crosscheck pair is absent from its union")
            arm = np.asarray(pair["arm"], dtype=np.float64)
            baseline = np.asarray(pair["baseline"], dtype=np.float64)
            if (
                pair["arm_row_id"] != search_experiment._row_id(arm)
                or pair["baseline_row_id"]
                != search_experiment._row_id(baseline)
                or pair["pair_id"]
                != _pair_id(union["state_id"], arm, baseline)
            ):
                raise RuntimeError(
                    "crosscheck pair coordinate identity changed"
                )
            pair_ids.append(pair["pair_id"])
            represented_tags.update(pair["comparison_tags"])
            represented_by_state[union["state_id"]] = (
                represented_by_state.get(union["state_id"], 0) + 1
            )
    carried_ids = [item["pair_id"] for item in manifest["carried_comparisons"]]
    failed_ids = [item["pair_id"] for item in manifest["carried_failed_pairs"]]
    for item in (
        manifest["carried_comparisons"] + manifest["carried_failed_pairs"]
    ):
        represented_tags.update(item["comparison_tags"])
        represented_by_state[item["state_id"]] = (
            represented_by_state.get(item["state_id"], 0) + 1
        )
    all_ids = pair_ids + carried_ids + failed_ids
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("crosscheck pairs are duplicated")
    if len(all_ids) > manifest["scheduled_pair_count"]:
        raise RuntimeError("crosscheck pair count exceeds its frozen schedule")
    if (
        manifest["budget"] == BUDGETS[0]
        and len(all_ids) != manifest["scheduled_pair_count"]
    ):
        raise RuntimeError("initial crosscheck omits a frozen pair")
    if (
        sum(manifest["scheduled_pairs_by_state"].values())
        != manifest["scheduled_pair_count"]
    ):
        raise RuntimeError("crosscheck per-state denominator changed")
    if represented_by_state != manifest["scheduled_pairs_by_state"]:
        raise RuntimeError("crosscheck per-state allocation changed")
    if represented_tags != set(tags):
        raise RuntimeError("crosscheck frozen comparison allocation changed")
    if set(artifacts) != set(manifest["scheduled_pairs_by_state"]):
        raise RuntimeError("crosscheck initial judge bindings are incomplete")


def _validate_initial_artifact(
    search_manifest: dict[str, Any], search_out: Path, artifact: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    expected_paths = search_experiment._paths(
        search_out,
        search_experiment.judge_tag(artifact["state_id"], artifact["budget"]),
        "judge",
    )
    if set(artifact.get("paths", {})) != {"npz", "json", "complete"} or set(
        artifact.get("hashes", {})
    ) != {"npz", "json", "complete"}:
        raise RuntimeError(
            f"{artifact['state_id']}: incomplete initial judge binding"
        )
    paths = {role: Path(path) for role, path in artifact["paths"].items()}
    if any(
        paths[role].resolve() != expected_paths[role].resolve()
        for role in paths
    ):
        raise RuntimeError(
            f"{artifact['state_id']}: initial judge artifact path changed"
        )
    for role, expected in artifact["hashes"].items():
        if not paths[role].is_file() or sha256_file(paths[role]) != expected:
            raise RuntimeError(
                f"{artifact['state_id']}: initial judge {role} changed"
            )
    terminal = search_experiment._validate_terminal(
        search_manifest,
        expected_paths,
        search_experiment._judge_cell(
            artifact["state_id"], artifact["budget"], None
        ),
    )
    if terminal["status"] != "succeeded":
        raise RuntimeError(
            f"{artifact['state_id']}: initial judge is not successful"
        )
    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    union = report["union"]
    with np.load(paths["npz"], allow_pickle=False) as stored:
        candidates = np.asarray(stored["candidates"], dtype=np.float64)
    candidate_hash = hashlib.sha256(
        np.ascontiguousarray(candidates, dtype="<f8").tobytes()
    ).hexdigest()
    if (
        artifact["union_id"] != union["union_id"]
        or artifact["candidate_union_sha256"] != candidate_hash
        or candidate_hash != union["candidate_union_sha256"]
    ):
        raise RuntimeError(
            f"{artifact['state_id']}: initial judge union identity changed"
        )
    return union, candidates.copy()


def _validate_union_source_binding(
    union: dict[str, Any],
    descriptor: dict[str, Any],
    initial_union: dict[str, Any],
    initial_candidates: np.ndarray,
    source_comparisons: dict[str, dict[str, Any]],
) -> None:
    state_id = union["state_id"]
    if capture.canonical(union["state"]) != capture.canonical(descriptor):
        raise RuntimeError(f"{state_id}: crosscheck state descriptor changed")
    initial_positions = {
        row_id: index
        for index, row_id in enumerate(initial_union["candidate_row_ids"])
    }
    expected_rows: list[np.ndarray] = []
    expected_row_ids: list[str] = []
    seen_rows: set[str] = set()
    if [pair["pair_id"] for pair in union["pairs"]] != sorted(
        pair["pair_id"] for pair in union["pairs"]
    ):
        raise RuntimeError(f"{state_id}: crosscheck pair order changed")
    for pair in union["pairs"]:
        tags = pair["comparison_tags"]
        if tags != sorted(set(tags)):
            raise RuntimeError(f"{state_id}: pair source tags changed")
        pair_sources = {
            item["comparison_tag"]: item for item in pair["source_comparisons"]
        }
        if set(pair_sources) != set(tags) or len(pair_sources) != len(
            pair["source_comparisons"]
        ):
            raise RuntimeError(f"{state_id}: pair source comparisons changed")
        for tag in tags:
            source = source_comparisons.get(tag)
            if source is None or capture.canonical(
                pair_sources[tag]
            ) != capture.canonical(source):
                raise RuntimeError(f"{state_id}: source comparison changed")
            if (
                source["cell"]["state_id"] != state_id
                or source["selected_row_id"] != pair["arm_row_id"]
                or source["baseline_row_id"] != pair["baseline_row_id"]
                or capture.canonical(source["band"])
                != capture.canonical(pair["band"])
            ):
                raise RuntimeError(f"{state_id}: pair source mapping changed")
        for role, row_id_key in (
            ("arm", "arm_row_id"),
            ("baseline", "baseline_row_id"),
        ):
            row_id = pair[row_id_key]
            if row_id not in initial_positions:
                raise RuntimeError(
                    f"{state_id}: source coordinate left initial judge union"
                )
            row = np.asarray(pair[role], dtype=np.float64)
            frozen = initial_candidates[initial_positions[row_id]]
            if _row_bytes(row) != _row_bytes(frozen):
                raise RuntimeError(f"{state_id}: source coordinate changed")
            if row_id not in seen_rows:
                seen_rows.add(row_id)
                expected_row_ids.append(row_id)
                expected_rows.append(frozen.copy())
    expected_candidates = np.vstack(expected_rows).astype(
        np.float64, copy=False
    )
    actual_candidates = np.asarray(union["candidates"], dtype=np.float64)
    if (
        union["candidate_row_ids"] != expected_row_ids
        or np.ascontiguousarray(actual_candidates, dtype="<f8").tobytes()
        != np.ascontiguousarray(expected_candidates, dtype="<f8").tobytes()
    ):
        raise RuntimeError(f"{state_id}: crosscheck candidate union changed")


def _validate_previous_partition(
    manifest: dict[str, Any], prior: dict[str, Any]
) -> None:
    prior_comparisons = prior["comparisons"]
    expected_carried = [
        copy.deepcopy(item)
        for item in prior_comparisons
        if not comparison_needs_escalation(item)
    ]
    pending_ids = {
        item["pair_id"]
        for item in prior_comparisons
        if comparison_needs_escalation(item)
    }
    active_ids = {
        pair["pair_id"]
        for union in manifest["unions"]
        for pair in union["pairs"]
    }
    if active_ids != pending_ids:
        raise RuntimeError("escalated crosscheck active pair set changed")
    for key, expected in (
        ("carried_comparisons", expected_carried),
        ("carried_failed_pairs", prior["failed_pairs"]),
        ("carried_failed_states", prior["failed_states"]),
    ):
        if capture.canonical(manifest[key]) != capture.canonical(expected):
            raise RuntimeError(f"escalated crosscheck {key} changed")


def _validate_frozen_selection(
    manifest: dict[str, Any], source_comparisons: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    if manifest["split"] == "development":
        expected_tags, reasons = select_development_tags(
            list(source_comparisons.values())
        )
    else:
        expected_tags = sorted(manifest["explicit_comparison_tags"])
        reasons = {tag: ["explicit"] for tag in expected_tags}
    if manifest["frozen_comparison_tags"] != expected_tags:
        raise RuntimeError("crosscheck frozen source selection changed")
    records = [pair for union in manifest["unions"] for pair in union["pairs"]]
    records.extend(manifest["carried_comparisons"])
    records.extend(manifest["carried_failed_pairs"])
    for record in records:
        expected_reasons = sorted(
            {
                reason
                for tag in record["comparison_tags"]
                for reason in reasons[tag]
            }
        )
        if record["selection_reasons"] != expected_reasons:
            raise RuntimeError("crosscheck source selection reasons changed")
    return reasons


def runtime_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    search_path = Path(manifest["search_manifest"])
    summary_path = Path(manifest["source_judge_summary"])
    if sha256_file(search_path) != manifest["search_manifest_sha256"]:
        raise RuntimeError("E3 search manifest changed")
    if sha256_file(summary_path) != manifest["source_judge_summary_sha256"]:
        raise RuntimeError("source E3 judge summary changed")
    search_manifest = json.loads(search_path.read_text(encoding="utf-8"))
    search_experiment.validate_manifest(search_manifest, require_ready=True)
    if manifest_digest(search_manifest) != manifest["search_manifest_digest"]:
        raise RuntimeError("E3 search manifest identity changed")
    if manifest["split"] != search_manifest["split"]:
        raise RuntimeError("crosscheck split differs from its E3 source")
    actual_search = search_experiment.runtime_identity(search_manifest)
    if capture.canonical(actual_search) != capture.canonical(
        manifest["search_identity"]
    ):
        raise RuntimeError("E3 search identity changed")
    if source_hashes() != manifest["source_hashes"]:
        raise RuntimeError("crosscheck source identity changed")
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_comparisons = {
        item["comparison_tag"]: item for item in source_summary["comparisons"]
    }
    if len(source_comparisons) != len(source_summary["comparisons"]):
        raise RuntimeError(
            "source E3 judge summary contains duplicate comparisons"
        )
    _validate_frozen_selection(manifest, source_comparisons)
    seen = set()
    frozen_unions: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    search_out = Path(manifest["search_output"])
    for state_id, artifact in manifest["initial_judge_artifacts"].items():
        if state_id != artifact["state_id"]:
            raise RuntimeError("initial judge artifact state identity changed")
        frozen_unions[state_id] = _validate_initial_artifact(
            search_manifest, search_out, artifact
        )
        seen.add(state_id)
    descriptors = {
        item["state_id"]: item for item in search_manifest["states"]
    }
    for union in manifest["unions"]:
        state_id = union["state_id"]
        if state_id not in descriptors or state_id not in frozen_unions:
            raise RuntimeError("crosscheck union refers to an unfrozen state")
        initial_union, initial_candidates = frozen_unions[state_id]
        _validate_union_source_binding(
            union,
            descriptors[state_id],
            initial_union,
            initial_candidates,
            source_comparisons,
        )
    previous = manifest.get("previous_summary")
    if (
        previous is not None
        and sha256_file(previous) != manifest["previous_summary_sha256"]
    ):
        raise RuntimeError("previous crosscheck summary changed")
    if previous is not None:
        prior = json.loads(Path(previous).read_text(encoding="utf-8"))
        _validate_previous_partition(manifest, prior)
    return {
        "search_identity": actual_search,
        "source_hashes": source_hashes(),
        "initial_judge_artifact_states": sorted(seen),
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
    for role, expected_hash in record["hashes"].items():
        if (
            not paths[role].is_file()
            or sha256_file(paths[role]) != expected_hash
        ):
            raise RuntimeError(f"{state_id}: changed crosscheck {role}")
    for role in ("npz", "json", "failure"):
        if paths[role].exists() and role not in record["hashes"]:
            raise RuntimeError(f"{state_id}: untracked crosscheck artifact")
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


def run_cell(
    manifest: dict[str, Any], out: Path, union: dict[str, Any]
) -> dict[str, Any]:
    validate_manifest(manifest, require_ready=True)
    runtime_identity(manifest)
    state_id = union["state_id"]
    paths = _paths(out, state_id)
    if paths["complete"].exists():
        return validate_cell(manifest, out, union)
    if any(
        path.exists() for role, path in paths.items() if role != "complete"
    ):
        raise RuntimeError(f"{state_id}: partial crosscheck attempt exists")
    try:
        state = capture.restore_capture(union["state"]["snapshot"])
        candidates = np.asarray(union["candidates"], dtype=np.float64).copy()
        before = _state_digest(state, candidates)
        arrays = ordinary.evaluate_mc_union(
            state, candidates, manifest["budget"], union["replicate_seeds"]
        )
        if before != _state_digest(state, candidates):
            raise RuntimeError("ordinary-MC crosscheck mutated captured state")
        arrays["candidates"] = candidates
        write_npz(paths["npz"], arrays)
        write_json(
            paths["json"],
            {
                "schema_version": SCHEMA_VERSION,
                "status": "succeeded",
                "state_id": state_id,
                "budget": manifest["budget"],
                "replicate_seeds": union["replicate_seeds"],
                "candidate_row_ids": union["candidate_row_ids"],
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
        for role in ("npz", "json"):
            if paths[role].exists():
                paths[role].unlink()
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
    if status == "succeeded":
        runtime_identity(manifest)
    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "manifest_sha256": manifest_digest(manifest),
        "state_id": state_id,
        "hashes": hashes,
    }
    write_json(paths["complete"], completion)
    return validate_cell(manifest, out, union)


def _logmeanexp(values: np.ndarray, axis: int | None = None):
    values = np.asarray(values, dtype=np.float64)
    count = values.size if axis is None else values.shape[axis]
    return np.logaddexp.reduce(values, axis=axis) - np.log(count)


def _status_counts(
    comparisons: list[dict[str, Any]], scheduled: int
) -> dict[str, Any]:
    score = {
        key: sum(item["cross_method_status"] == key for item in comparisons)
        for key in ("agree", "disagree", "unresolved")
    }
    raw = {
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
    missing = int(scheduled - len(comparisons))
    score["failed_or_missing"] = missing
    raw["failed_or_missing"] = missing
    return {
        "scheduled": int(scheduled),
        "observed": len(comparisons),
        "score": score,
        "raw": raw,
    }


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
    failed_pairs = copy.deepcopy(manifest["carried_failed_pairs"])
    failed_states = copy.deepcopy(manifest["carried_failed_states"])
    for union in manifest["unions"]:
        record = validate_cell(manifest, out, union)
        if record["status"] != "succeeded":
            if union["state_id"] not in failed_states:
                failed_states.append(union["state_id"])
            failed_pairs.extend(
                {
                    "pair_id": pair["pair_id"],
                    "state_id": union["state_id"],
                    "comparison_tags": pair["comparison_tags"],
                    "selection_reasons": pair["selection_reasons"],
                    "reason": "ordinary_mc_cell_failed",
                    "budget": manifest["budget"],
                }
                for pair in union["pairs"]
            )
            continue
        with np.load(
            _paths(out, union["state_id"])["npz"], allow_pickle=False
        ) as stored:
            arrays = {key: stored[key] for key in stored.files}
        candidates = np.asarray(arrays["candidates"], dtype=np.float64)
        candidate_hash = hashlib.sha256(
            np.ascontiguousarray(candidates, dtype="<f8").tobytes()
        ).hexdigest()
        if candidate_hash != union["candidate_union_sha256"]:
            raise RuntimeError("crosscheck candidate coordinates changed")
        positions = {
            row_id: index
            for index, row_id in enumerate(union["candidate_row_ids"])
        }
        for pair in union["pairs"]:
            arm_position = positions[pair["arm_row_id"]]
            baseline_position = positions[pair["baseline_row_id"]]
            pooled_difference = float(
                _logmeanexp(arrays["log_residual"][:, arm_position])
                + np.mean(arrays["penalty"][:, arm_position])
                - _logmeanexp(arrays["log_residual"][:, baseline_position])
                - np.mean(arrays["penalty"][:, baseline_position])
            )
            prior = previous_by_pair.get(pair["pair_id"])
            judged = judge_paired(
                arrays["full_score"][:, arm_position],
                arrays["full_score"][:, baseline_position],
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
            source_raw_classes = sorted(
                {
                    item["raw_loss"]["classification"]
                    for item in pair["source_comparisons"]
                }
            )
            source_resolved = (
                len(source_classes) == 1 and source_classes[0] != "unresolved"
            )
            mc_resolved = judged["classification"] != "unresolved"
            source_raw_resolved = (
                len(source_raw_classes) == 1
                and source_raw_classes[0] != "unresolved"
            )
            mc_raw_resolved = raw_loss["classification"] != "unresolved"
            comparisons.append(
                {
                    "pair_id": pair["pair_id"],
                    "state_id": union["state_id"],
                    "comparison_tags": pair["comparison_tags"],
                    "selection_reasons": pair["selection_reasons"],
                    "arm_row_id": pair["arm_row_id"],
                    "baseline_row_id": pair["baseline_row_id"],
                    "band": pair["band"],
                    "budget": manifest["budget"],
                    "source_rqmc_classifications": source_classes,
                    "source_rqmc_raw_loss_classifications": source_raw_classes,
                    "mc_judge": judged,
                    "mc_raw_loss": raw_loss,
                    "comparison_penalty_active": penalty_active,
                    "cross_method_status": (
                        "unresolved"
                        if not (source_resolved and mc_resolved)
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
                            if not (source_raw_resolved and mc_raw_resolved)
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
    if len({item["pair_id"] for item in comparisons}) != len(comparisons):
        raise RuntimeError("crosscheck summary contains duplicated pairs")
    observed_ids = {item["pair_id"] for item in comparisons}
    failed_ids = {item["pair_id"] for item in failed_pairs}
    if (
        len(failed_ids) != len(failed_pairs)
        or observed_ids & failed_ids
        or len(observed_ids | failed_ids) != manifest["scheduled_pair_count"]
    ):
        raise RuntimeError(
            "crosscheck summary does not account for its allocation"
        )
    scheduled = int(manifest["scheduled_pair_count"])
    by_state = []
    for state_id, count in sorted(
        manifest["scheduled_pairs_by_state"].items()
    ):
        state_comparisons = [
            item for item in comparisons if item["state_id"] == state_id
        ]
        by_state.append(
            {"state_id": state_id, **_status_counts(state_comparisons, count)}
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "search_ordinary_mc_crosscheck_summary",
        "manifest_sha256": manifest_digest(manifest),
        "source_judge_summary_sha256": manifest["source_judge_summary_sha256"],
        "search_manifest_sha256": manifest["search_manifest_sha256"],
        "budget": manifest["budget"],
        "scheduled_pair_count": scheduled,
        "scheduled_pairs_by_state": manifest["scheduled_pairs_by_state"],
        "frozen_comparison_tags": manifest["frozen_comparison_tags"],
        "counts": _status_counts(comparisons, scheduled),
        "by_state": by_state,
        "failed_states": failed_states,
        "failed_pairs": failed_pairs,
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
    report = summarize(manifest, out)
    write_json(out / "summary.json", report)
    return 1 if failures or report["failed_pairs"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--judge-summary", type=Path, required=True)
    prepare.add_argument("--search-manifest", type=Path, required=True)
    prepare.add_argument("--search-out", type=Path, required=True)
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
            args.search_manifest,
            args.search_out,
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
