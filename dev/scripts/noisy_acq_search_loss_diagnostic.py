"""Diagnose where accepted-refinement holdout losses enter E3 search.

This post-hoc diagnostic is separate from the 96 primary holdout
comparisons.  It evaluates the S0 point, the finalist's accurately rescored
pre-refinement winner, and its accepted refined point on fresh common
ordinary-MC rules.  Results are explanatory only and cannot change the
frozen finalist, the primary holdout verdict, or an E4 entry gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

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
import noisy_acq_search_crosscheck as search_crosscheck
import noisy_acq_search_experiment as search_experiment
import numpy as np

SCHEMA_VERSION = 1
KIND = "search_refinement_loss_diagnostic"
SUMMARY_KIND = "search_refinement_loss_diagnostic_summary"
BUDGETS = (32768, 65536)
REPLICATES = 8
MASTER_SEED = 2026091703
SEED_DOMAIN = "search_refinement_loss_diagnostic"
PRIMARY_HOLDOUT_COMPARISONS = 96
CONTRASTS = (
    ("selected_minus_own_winner", "selected", "own_winner"),
    ("own_winner_minus_S0", "own_winner", "S0"),
    ("selected_minus_S0", "selected", "S0"),
)


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_hashed_json(path: Path | str) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _row_bytes(row: np.ndarray) -> bytes:
    return np.ascontiguousarray(row, dtype="<f8").reshape(-1).tobytes()


def _row_id(row: np.ndarray) -> str:
    return search_experiment._row_id(np.asarray(row, dtype=np.float64))


def _require_point(value: Any, *, dimension: int | None = None) -> np.ndarray:
    point = np.asarray(value)
    if point.dtype != np.float64:
        raise RuntimeError("diagnostic coordinates must be stored as float64")
    point = point.reshape(-1)
    if dimension is not None and point.shape != (dimension,):
        raise RuntimeError("diagnostic coordinate dimension changed")
    if point.size == 0 or not np.all(np.isfinite(point)):
        raise RuntimeError(
            "diagnostic coordinates must be finite and nonempty"
        )
    return point.copy()


def reconstruct_own_winner(
    arrays: Mapping[str, np.ndarray], details: Mapping[str, Any]
) -> np.ndarray:
    """Reconstruct the accurately rescored winner before refinement."""

    candidates = np.asarray(arrays["coarse_candidates"])
    indices = np.asarray(arrays["shortlist_indices"])
    scores = np.asarray(arrays["shortlist_scores"])
    selected = np.asarray(arrays["selected"])
    if candidates.dtype != np.float64 or selected.dtype != np.float64:
        raise RuntimeError("selection coordinates are not float64")
    if candidates.ndim != 2 or selected.shape != (1, candidates.shape[1]):
        raise RuntimeError("selection coordinate shapes changed")
    if indices.ndim != 1 or scores.ndim != 1 or len(indices) != len(scores):
        raise RuntimeError("shortlist arrays are not aligned")
    if not np.issubdtype(indices.dtype, np.integer) or len(indices) == 0:
        raise RuntimeError("shortlist indices are missing or nonintegral")
    if np.any(np.isnan(scores)) or not np.any(np.isfinite(scores)):
        raise RuntimeError("shortlist scores have no finite winner")
    generated = details.get("generated_count")
    if not isinstance(generated, int) or isinstance(generated, bool):
        raise RuntimeError("generated candidate count is not integral")
    repeat_count = len(candidates) - generated
    if repeat_count < 0:
        raise RuntimeError("generated candidate count exceeds the panel")
    shortlist_position = int(np.argmin(scores))
    row = int(indices[shortlist_position])
    if row < repeat_count:
        raise RuntimeError(
            "pre-refinement winner is a substituted repeat coordinate"
        )
    if row < 0 or row >= len(candidates):
        raise RuntimeError("pre-refinement winner index is outside the panel")
    fallback = details.get("coarse_fallback_score")
    selected_score = details.get("selected_accurate_score")
    if not isinstance(fallback, (int, float)) or not np.isfinite(fallback):
        raise RuntimeError("coarse fallback score is not finite")
    if float(scores[shortlist_position]) != float(fallback):
        raise RuntimeError("reconstructed winner score changed")
    if (
        not isinstance(selected_score, (int, float))
        or not np.isfinite(selected_score)
        or not float(selected_score) < float(fallback)
    ):
        raise RuntimeError(
            "refinement did not strictly improve its own winner"
        )
    if details.get("fallback_reason") is not None:
        raise RuntimeError(
            "accepted-refinement diagnostic received a fallback"
        )
    if details.get("selected_shortlist_index") is not None:
        raise RuntimeError(
            "selection did not retain an accepted refined point"
        )
    if details.get("target_called") is not False:
        raise RuntimeError("selection reached the target")
    winner = _require_point(candidates[row], dimension=candidates.shape[1])
    refined = _require_point(selected[0], dimension=candidates.shape[1])
    if _row_bytes(winner) == _row_bytes(refined):
        raise RuntimeError("accepted refined point equals its own winner")
    return winner


def eligible_comparisons(
    rqmc_comparisons: list[dict[str, Any]],
    mc_comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the frozen conditional diagnostic predicate."""

    by_tag: dict[str, dict[str, Any]] = {}
    for item in mc_comparisons:
        for tag in item.get("comparison_tags", []):
            if tag in by_tag:
                raise RuntimeError("MC summary maps a comparison tag twice")
            by_tag[tag] = item
    eligible = []
    for item in rqmc_comparisons:
        cell = item.get("cell", {})
        if (
            cell.get("kind") != "primary_vs_s0"
            or cell.get("arm_tag") != "finalist"
            or cell.get("reference_tag") != "S0"
            or item.get("raw_loss", {}).get("classification")
            != "material_loss"
            or item.get("comparison_penalty_active") is not False
        ):
            continue
        tag = item.get("comparison_tag")
        crosscheck = by_tag.get(tag)
        if crosscheck is None:
            raise RuntimeError(
                f"material-loss comparison lacks MC evidence: {tag}"
            )
        mc_class = crosscheck.get("mc_raw_loss", {}).get("classification")
        if mc_class not in {"material_loss", "unresolved", "no_material_loss"}:
            raise RuntimeError(f"invalid MC raw-loss classification: {tag}")
        if (
            mc_class in {"material_loss", "unresolved"}
            and crosscheck.get("comparison_penalty_active") is False
        ):
            eligible.append(
                {
                    "comparison_tag": tag,
                    "source_comparison": copy.deepcopy(item),
                    "source_mc_comparison": copy.deepcopy(crosscheck),
                    "mc_eligibility": mc_class,
                }
            )
    return sorted(eligible, key=lambda item: item["comparison_tag"])


def _accepted_refinement(details: Mapping[str, Any]) -> bool:
    return (
        "fallback_reason" in details
        and details["fallback_reason"] is None
        and "selected_shortlist_index" in details
        and details["selected_shortlist_index"] is None
    )


def mechanism_classification(
    selected_vs_own: str, own_vs_s0: str, selected_vs_s0: str
) -> str:
    """Classify score evidence without claiming a unique causal mechanism."""

    resolved_nonharmful = {"beneficial", "practical_tie"}
    allowed = resolved_nonharmful | {"harmful", "unresolved"}
    if (
        selected_vs_own not in allowed
        or own_vs_s0 not in allowed
        or selected_vs_s0 not in allowed
    ):
        raise ValueError("unknown diagnostic score classification")
    if selected_vs_s0 == "unresolved":
        return "unresolved_selected_vs_S0"
    if selected_vs_s0 == "harmful":
        if selected_vs_own == "unresolved" or own_vs_s0 == "unresolved":
            return "direct_regression_stage_unresolved"
        if selected_vs_own == "harmful" and own_vs_s0 == "harmful":
            return "mixed_stage_regressions"
        if selected_vs_own == "harmful":
            return "accepted_refinement_regression"
        if own_vs_s0 == "harmful":
            return "loss_located_before_refinement"
        return "distributed_threshold_accumulation"
    return "direct_score_not_harmful"


def raw_mechanism_classification(
    selected_vs_own: str, own_vs_s0: str, selected_vs_s0: str
) -> str:
    """Localize the separate 10% raw-reduction loss threshold."""

    allowed = {"material_loss", "no_material_loss", "unresolved"}
    if (
        selected_vs_own not in allowed
        or own_vs_s0 not in allowed
        or selected_vs_s0 not in allowed
    ):
        raise ValueError("unknown diagnostic raw-loss classification")
    if selected_vs_s0 == "unresolved":
        return "unresolved_selected_vs_S0"
    if selected_vs_s0 == "material_loss":
        if selected_vs_own == "unresolved" or own_vs_s0 == "unresolved":
            return "direct_material_loss_stage_unresolved"
        if selected_vs_own == "material_loss" and own_vs_s0 == "material_loss":
            return "mixed_stage_material_losses"
        if selected_vs_own == "material_loss":
            return "accepted_refinement_material_loss"
        if own_vs_s0 == "material_loss":
            return "material_loss_before_refinement"
        return "distributed_raw_threshold_accumulation"
    return "direct_MC_does_not_confirm_material_loss"


def classify_case_records(
    records: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    if set(records) != {name for name, _, _ in CONTRASTS}:
        raise ValueError("diagnostic case does not have three exact contrasts")
    selected_own = records["selected_minus_own_winner"]
    own_s0 = records["own_winner_minus_S0"]
    selected_s0 = records["selected_minus_S0"]
    score_status = mechanism_classification(
        selected_own["judge"]["classification"],
        own_s0["judge"]["classification"],
        selected_s0["judge"]["classification"],
    )
    if any(record["penalty_active"] for record in records.values()):
        raw_status = "not_applicable_penalty_active"
    else:
        raw_status = raw_mechanism_classification(
            selected_own["raw_loss"]["classification"],
            own_s0["raw_loss"]["classification"],
            selected_s0["raw_loss"]["classification"],
        )
    return score_status, raw_status


def _holdout_selection_reasons(
    comparisons: list[dict[str, Any]],
) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {}
    for item in comparisons:
        tag = item["comparison_tag"]
        mean = item.get("judge", {}).get("mean_difference")
        if (
            isinstance(mean, (int, float))
            and np.isfinite(mean)
            and abs(float(mean)) >= 10 * float(item["band"]["eps_F"])
        ):
            reasons.setdefault(tag, []).append("ten_practical_bands")
        if item.get("raw_loss", {}).get("classification") == "material_loss":
            reasons.setdefault(tag, []).append("confident_raw_material_loss")
    return {
        tag: sorted(set(values)) for tag, values in sorted(reasons.items())
    }


def validate_parent_hash_contracts(
    rqmc: Mapping[str, Any],
    mc_summary: Mapping[str, Any],
    explicit_selection: Mapping[str, Any],
    *,
    search_manifest_file_sha256: str,
    search_manifest_digest: str,
    rqmc_summary_file_sha256: str,
) -> None:
    """Validate the raw and semantic hashes used by the frozen parents."""

    if (
        rqmc.get("kind") != "search_judge_summary"
        or rqmc.get("search_manifest_sha256") != search_manifest_digest
    ):
        raise RuntimeError("holdout RQMC parent identity changed")
    if (
        mc_summary.get("kind") != "search_ordinary_mc_crosscheck_summary"
        or mc_summary.get("search_manifest_sha256")
        != search_manifest_file_sha256
        or mc_summary.get("source_judge_summary_sha256")
        != rqmc_summary_file_sha256
    ):
        raise RuntimeError("holdout MC parent identity changed")
    if (
        explicit_selection.get("kind")
        != "search_ordinary_mc_holdout_explicit_selection"
        or explicit_selection.get("search_manifest_sha256")
        != search_manifest_file_sha256
        or explicit_selection.get("source_judge_summary_sha256")
        != rqmc_summary_file_sha256
    ):
        raise RuntimeError("holdout explicit-selection identity changed")


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        SCRIPTS / "noisy_acq_crosscheck.py",
        SCRIPTS / "noisy_acq_search_crosscheck.py",
        SCRIPTS / "noisy_acq_search_experiment.py",
        SCRIPTS / "noisy_acq_quadrature.py",
        SCRIPTS / "noisy_acq_integration.py",
        SCRIPTS / "noisy_acq_experiment.py",
    ]
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths
    }


def archive_adapter(path: Path) -> str:
    source = Path(__file__).resolve()
    digest = sha256_file(source)
    if path.exists():
        if sha256_file(path) != digest:
            raise RuntimeError("existing diagnostic adapter archive changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            raise RuntimeError("diagnostic adapter archive temporary exists")
        shutil.copyfile(source, temporary)
        temporary.replace(path)
    return digest


def _selection_artifact(
    search_manifest: dict[str, Any],
    search_out: Path,
    state_id: str,
    treatment_tag: str,
    replicate: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    cells = [
        cell
        for cell in search_manifest["cells"]
        if cell["state_id"] == state_id
        and cell["treatment_tag"] == treatment_tag
        and cell["replicate"] == replicate
    ]
    if len(cells) != 1:
        raise RuntimeError("diagnostic selection cell is not unique")
    cell = cells[0]
    paths = search_experiment._paths(
        search_out, search_experiment.cell_tag(cell), "selection"
    )
    terminal = search_experiment._validate_terminal(
        search_manifest, paths, cell
    )
    if terminal["status"] != "succeeded":
        raise RuntimeError("diagnostic selection did not succeed")
    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    with np.load(paths["npz"], allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    selected = _require_point(arrays["selected"][0])
    artifact = {
        "cell": copy.deepcopy(cell),
        "paths": {
            role: str(paths[role].resolve())
            for role in ("npz", "json", "complete")
        },
        "hashes": {
            role: sha256_file(paths[role])
            for role in ("npz", "json", "complete")
        },
    }
    return selected, report, arrays, artifact


def _contrast_id(case_tag: str, name: str) -> str:
    payload = json.dumps([case_tag, name], separators=(",", ":"))
    return "contrast_" + hashlib.sha256(payload.encode()).hexdigest()


def _build_unions(
    cases: list[dict[str, Any]],
    search_manifest: dict[str, Any],
    budget: int,
) -> list[dict[str, Any]]:
    by_state: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_state.setdefault(case["state_id"], []).append(case)
    unions = []
    for state_id, state_cases in sorted(by_state.items()):
        rows: list[np.ndarray] = []
        row_ids: list[str] = []
        lookup: dict[str, int] = {}

        def include(point: Any) -> str:
            row = _require_point(point)
            row_id = _row_id(row)
            if row_id in lookup:
                if _row_bytes(rows[lookup[row_id]]) != _row_bytes(row):
                    raise RuntimeError(
                        "diagnostic coordinate digest collision"
                    )
            else:
                lookup[row_id] = len(rows)
                rows.append(row)
                row_ids.append(row_id)
            return row_id

        contrasts = []
        for case in sorted(
            state_cases, key=lambda item: item["comparison_tag"]
        ):
            role_ids = {
                role: include(case["coordinates"][role])
                for role in ("S0", "own_winner", "selected")
            }
            if role_ids != case["row_ids"]:
                raise RuntimeError("frozen diagnostic row identifiers changed")
            for name, arm_role, baseline_role in CONTRASTS:
                contrasts.append(
                    {
                        "contrast_id": _contrast_id(
                            case["comparison_tag"], name
                        ),
                        "comparison_tag": case["comparison_tag"],
                        "name": name,
                        "arm_role": arm_role,
                        "baseline_role": baseline_role,
                        "arm_row_id": role_ids[arm_role],
                        "baseline_row_id": role_ids[baseline_role],
                        "band": copy.deepcopy(case["band"]),
                    }
                )
        candidates = np.vstack(rows).astype(np.float64, copy=False)
        unions.append(
            {
                "state_id": state_id,
                "state": copy.deepcopy(
                    search_experiment._state_descriptor(
                        search_manifest, state_id
                    )
                ),
                "case_tags": sorted(
                    case["comparison_tag"] for case in state_cases
                ),
                "candidate_row_ids": row_ids,
                "candidates": candidates.tolist(),
                "candidate_union_sha256": hashlib.sha256(
                    np.ascontiguousarray(candidates, dtype="<f8").tobytes()
                ).hexdigest(),
                "contrasts": contrasts,
                "replicate_seeds": [
                    integration.derive_seed(
                        MASTER_SEED, state_id, budget, replicate, SEED_DOMAIN
                    )
                    for replicate in range(REPLICATES)
                ],
            }
        )
    return unions


def prepare_manifest(
    *,
    search_manifest_path: Path,
    search_out: Path,
    rqmc_summary_path: Path,
    mc_manifest_path: Path,
    mc_summary_path: Path,
    explicit_selection_path: Path,
    adapter_archive: Path,
    budget: int,
    previous_summary_path: Path | None,
    launch_ready: bool,
) -> dict[str, Any]:
    if budget not in BUDGETS:
        raise RuntimeError("diagnostic budget is not prescribed")
    search_manifest, search_manifest_file_sha = _load_hashed_json(
        search_manifest_path
    )
    search_experiment.validate_manifest(search_manifest, require_ready=True)
    search_identity = search_experiment.runtime_identity(search_manifest)
    if search_manifest.get("split") != "holdout":
        raise RuntimeError("loss diagnostic requires the frozen holdout")
    rqmc, rqmc_sha = _load_hashed_json(rqmc_summary_path)
    mc_manifest, mc_manifest_sha = _load_hashed_json(mc_manifest_path)
    mc_summary, mc_summary_sha = _load_hashed_json(mc_summary_path)
    explicit, explicit_sha = _load_hashed_json(explicit_selection_path)
    search_digest = integration.manifest_digest(search_manifest)
    validate_parent_hash_contracts(
        rqmc,
        mc_summary,
        explicit,
        search_manifest_file_sha256=search_manifest_file_sha,
        search_manifest_digest=search_digest,
        rqmc_summary_file_sha256=rqmc_sha,
    )
    if (
        rqmc.get("primary_screen_counts", {}).get("scheduled")
        != PRIMARY_HOLDOUT_COMPARISONS
    ):
        raise RuntimeError("holdout RQMC summary identity changed")
    search_crosscheck.validate_manifest(mc_manifest, require_ready=True)
    search_crosscheck.runtime_identity(mc_manifest)
    if (
        mc_summary.get("manifest_sha256")
        != integration.manifest_digest(mc_manifest)
        or mc_manifest.get("budget") != BUDGETS[1]
        or mc_summary.get("budget") != BUDGETS[1]
        or mc_summary.get("automatic_override") is not False
        or mc_summary.get("failed_pairs")
        or mc_summary.get("failed_states")
    ):
        raise RuntimeError("final holdout MC summary identity changed")
    expected_reasons = _holdout_selection_reasons(rqmc["comparisons"])
    expected_tags = sorted(expected_reasons)
    explicit_tags = explicit.get("comparison_tags")
    explicit_reasons = explicit.get("selection_reasons")
    reasons_match = (
        isinstance(explicit_reasons, dict)
        and sorted(explicit_reasons) == expected_tags
        and all(
            isinstance(explicit_reasons[tag], list)
            and all(
                isinstance(reason, str) for reason in explicit_reasons[tag]
            )
            and len(explicit_reasons[tag]) == len(set(explicit_reasons[tag]))
            and sorted(explicit_reasons[tag]) == expected_reasons[tag]
            for tag in expected_tags
        )
    )
    if (
        explicit.get("schema_version") != 1
        or explicit.get("policy")
        != (
            "abs(score_mean_difference) >= 10 * eps_F OR confidently "
            "classified raw material loss"
        )
        or explicit.get("automatic_override") is not False
        or explicit.get("comparison_count") != len(expected_tags)
        or explicit_tags != expected_tags
        or explicit_tags != mc_summary.get("frozen_comparison_tags")
        or not reasons_match
    ):
        raise RuntimeError("holdout MC explicit selection identity changed")
    selected = eligible_comparisons(
        rqmc["comparisons"], mc_summary["comparisons"]
    )
    cases = []
    for item in selected:
        source = item["source_comparison"]
        cell = source["cell"]
        state_id = cell["state_id"]
        replicate = int(cell["replicate"])
        (
            refined,
            refined_report,
            refined_arrays,
            refined_artifact,
        ) = _selection_artifact(
            search_manifest, search_out, state_id, cell["arm_tag"], replicate
        )
        baseline, _, _, baseline_artifact = _selection_artifact(
            search_manifest, search_out, state_id, "S0", replicate
        )
        details = refined_report.get("details", {})
        if (
            refined_report.get("treatment", {}).get("config", {}).get("arm")
            != "S2"
        ):
            raise RuntimeError(
                "holdout finalist is not the frozen S2 treatment"
            )
        if not _accepted_refinement(details):
            continue
        own = reconstruct_own_winner(refined_arrays, details)
        row_ids = {
            "S0": _row_id(baseline),
            "own_winner": _row_id(own),
            "selected": _row_id(refined),
        }
        if (
            row_ids["S0"] != source["baseline_row_id"]
            or row_ids["selected"] != source["selected_row_id"]
        ):
            raise RuntimeError(
                "diagnostic selection differs from primary holdout rows"
            )
        cases.append(
            {
                "comparison_tag": item["comparison_tag"],
                "state_id": state_id,
                "replicate": replicate,
                "band": copy.deepcopy(source["band"]),
                "mc_eligibility": item["mc_eligibility"],
                "row_ids": row_ids,
                "coordinates": {
                    "S0": baseline.tolist(),
                    "own_winner": own.tolist(),
                    "selected": refined.tolist(),
                },
                "source_rqmc_comparison": source,
                "source_mc_comparison": item["source_mc_comparison"],
                "selection_artifacts": {
                    "S0": baseline_artifact,
                    "finalist": refined_artifact,
                },
            }
        )
    if not cases:
        raise RuntimeError("no eligible accepted-refinement loss remains")
    unions = _build_unions(cases, search_manifest, budget)
    case_basis_sha = integration.manifest_digest(cases)
    previous = None
    previous_sha = None
    if budget == BUDGETS[0]:
        if previous_summary_path is not None:
            raise RuntimeError(
                "initial diagnostic budget cannot cite a predecessor"
            )
    else:
        if previous_summary_path is None:
            raise RuntimeError("65536 diagnostic requires its 32768 summary")
        previous, previous_sha = _load_hashed_json(previous_summary_path)
        if (
            previous.get("kind") != SUMMARY_KIND
            or previous.get("budget") != BUDGETS[0]
            or previous.get("case_tags")
            != [case["comparison_tag"] for case in cases]
            or previous.get("case_basis_sha256") != case_basis_sha
            or previous.get("search_manifest_sha256") != search_digest
            or previous.get("rqmc_summary_sha256") != rqmc_sha
            or previous.get("mc_summary_sha256") != mc_summary_sha
            or previous.get("automatic_override") is not False
            or previous.get("tuning_use") is not False
            or previous.get("failed_states")
        ):
            raise RuntimeError("previous diagnostic summary identity changed")
    adapter_sha = archive_adapter(adapter_archive)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "launch_ready": bool(launch_ready),
        "purpose": "post_hoc_stage_localization_only",
        "split": "holdout",
        "budget": budget,
        "replicates": REPLICATES,
        "master_seed": MASTER_SEED,
        "seed_domain": SEED_DOMAIN,
        "primary_holdout_comparison_count": PRIMARY_HOLDOUT_COMPARISONS,
        "automatic_override": False,
        "tuning_use": False,
        "changes_e4_entry_gate": False,
        "search_manifest": str(search_manifest_path.resolve()),
        "search_manifest_file_sha256": search_manifest_file_sha,
        "search_manifest_sha256": search_digest,
        "search_runtime_identity": search_identity,
        "search_out": str(search_out.resolve()),
        "rqmc_summary": str(rqmc_summary_path.resolve()),
        "rqmc_summary_sha256": rqmc_sha,
        "mc_manifest": str(mc_manifest_path.resolve()),
        "mc_manifest_file_sha256": mc_manifest_sha,
        "mc_summary": str(mc_summary_path.resolve()),
        "mc_summary_sha256": mc_summary_sha,
        "explicit_selection": str(explicit_selection_path.resolve()),
        "explicit_selection_sha256": explicit_sha,
        "previous_summary": None
        if previous_summary_path is None
        else str(previous_summary_path.resolve()),
        "previous_summary_sha256": previous_sha,
        "adapter": str(Path(__file__).resolve()),
        "adapter_sha256": adapter_sha,
        "adapter_archive": str(adapter_archive.resolve()),
        "source_hashes": source_hashes(),
        "case_tags": [case["comparison_tag"] for case in cases],
        "case_basis_sha256": case_basis_sha,
        "cases": cases,
        "unions": unions,
    }


def validate_manifest(
    manifest: dict[str, Any], *, require_ready: bool
) -> None:
    if (
        manifest.get("kind") != KIND
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise RuntimeError("not a search refinement-loss diagnostic manifest")
    if require_ready and manifest.get("launch_ready") is not True:
        raise RuntimeError("diagnostic manifest is not launch_ready")
    if (
        manifest.get("budget") not in BUDGETS
        or manifest.get("replicates") != REPLICATES
        or manifest.get("master_seed") != MASTER_SEED
        or manifest.get("seed_domain") != SEED_DOMAIN
        or manifest.get("primary_holdout_comparison_count")
        != PRIMARY_HOLDOUT_COMPARISONS
        or manifest.get("automatic_override") is not False
        or manifest.get("tuning_use") is not False
        or manifest.get("changes_e4_entry_gate") is not False
    ):
        raise RuntimeError("diagnostic scientific contract changed")
    actual_source_hashes = source_hashes()
    if manifest.get("source_hashes") != actual_source_hashes:
        raise RuntimeError("diagnostic source identity changed")
    if sha256_file(manifest["adapter_archive"]) != manifest["adapter_sha256"]:
        raise RuntimeError("diagnostic adapter archive changed")
    adapter_key = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if (
        manifest.get("source_hashes", {}).get(adapter_key)
        != manifest["adapter_sha256"]
    ):
        raise RuntimeError("diagnostic adapter source binding changed")
    for path_key, hash_key in (
        ("search_manifest", "search_manifest_file_sha256"),
        ("rqmc_summary", "rqmc_summary_sha256"),
        ("mc_manifest", "mc_manifest_file_sha256"),
        ("mc_summary", "mc_summary_sha256"),
        ("explicit_selection", "explicit_selection_sha256"),
    ):
        if sha256_file(manifest[path_key]) != manifest[hash_key]:
            raise RuntimeError(f"bound diagnostic source changed: {path_key}")
    if manifest["budget"] == BUDGETS[0]:
        if manifest.get("previous_summary") is not None:
            raise RuntimeError(
                "initial diagnostic manifest cites a predecessor"
            )
    elif sha256_file(manifest["previous_summary"]) != manifest.get(
        "previous_summary_sha256"
    ):
        raise RuntimeError("previous diagnostic summary changed")
    case_tags = [case["comparison_tag"] for case in manifest.get("cases", [])]
    if (
        not case_tags
        or case_tags != sorted(set(case_tags))
        or case_tags != manifest.get("case_tags")
    ):
        raise RuntimeError("diagnostic case allocation changed")
    if integration.manifest_digest(manifest["cases"]) != manifest.get(
        "case_basis_sha256"
    ):
        raise RuntimeError("diagnostic case basis changed")
    union_cases = []
    for union in manifest.get("unions", []):
        candidates = np.asarray(union["candidates"], dtype=np.float64)
        row_ids = union["candidate_row_ids"]
        if (
            candidates.ndim != 2
            or len(candidates) != len(row_ids)
            or len(row_ids) != len(set(row_ids))
        ):
            raise RuntimeError("diagnostic candidate union changed")
        if any(
            _row_id(row) != row_id
            for row, row_id in zip(candidates, row_ids, strict=True)
        ):
            raise RuntimeError("diagnostic candidate row identifier changed")
        digest = hashlib.sha256(
            np.ascontiguousarray(candidates, dtype="<f8").tobytes()
        ).hexdigest()
        if digest != union["candidate_union_sha256"]:
            raise RuntimeError("diagnostic candidate union digest changed")
        expected_seeds = [
            integration.derive_seed(
                MASTER_SEED,
                union["state_id"],
                manifest["budget"],
                replicate,
                SEED_DOMAIN,
            )
            for replicate in range(REPLICATES)
        ]
        if union["replicate_seeds"] != expected_seeds:
            raise RuntimeError("diagnostic seed allocation changed")
        union_cases.extend(union["case_tags"])
    if sorted(union_cases) != case_tags:
        raise RuntimeError("diagnostic union case partition changed")


def runtime_identity(manifest: dict[str, Any]) -> None:
    validate_manifest(manifest, require_ready=True)
    search_manifest = json.loads(
        Path(manifest["search_manifest"]).read_text(encoding="utf-8")
    )
    search_experiment.validate_manifest(search_manifest, require_ready=True)
    if (
        integration.manifest_digest(search_manifest)
        != manifest["search_manifest_sha256"]
    ):
        raise RuntimeError("search manifest canonical identity changed")
    if (
        search_experiment.runtime_identity(search_manifest)
        != manifest["search_runtime_identity"]
    ):
        raise RuntimeError("search runtime identity changed")
    expected_unions = _build_unions(
        manifest["cases"], search_manifest, manifest["budget"]
    )
    if capture.canonical(expected_unions) != capture.canonical(
        manifest["unions"]
    ):
        raise RuntimeError("diagnostic state unions changed")
    for case in manifest["cases"]:
        for artifact in case["selection_artifacts"].values():
            for role, expected in artifact["hashes"].items():
                if sha256_file(artifact["paths"][role]) != expected:
                    raise RuntimeError("bound selection artifact changed")


def _paths(out: Path, state_id: str) -> dict[str, Path]:
    return {
        "npz": out / "cells" / f"{state_id}.npz",
        "json": out / "cells" / f"{state_id}.json",
        "failure": out / "records" / f"{state_id}.failure.json",
        "complete": out / "records" / f"{state_id}.complete.json",
    }


def _partial(paths: dict[str, Path]) -> list[Path]:
    found = [
        path
        for role, path in paths.items()
        if role != "complete" and path.exists()
    ]
    found.extend(
        temporary
        for path in paths.values()
        for temporary in (
            path.with_name(path.name + ".tmp"),
            path.with_name(path.name + ".tmp.npz"),
        )
        if temporary.exists()
    )
    return found


def validate_cell(
    manifest: dict[str, Any], out: Path, union: dict[str, Any]
) -> dict[str, Any]:
    paths = _paths(out, union["state_id"])
    temporary = [
        candidate
        for path in paths.values()
        for candidate in (
            path.with_name(path.name + ".tmp"),
            path.with_name(path.name + ".tmp.npz"),
        )
        if candidate.exists()
    ]
    if temporary:
        raise RuntimeError("diagnostic cell has temporary artifacts")
    record = json.loads(paths["complete"].read_text(encoding="utf-8"))
    expected_roles = (
        {"npz", "json"} if record.get("status") == "succeeded" else {"failure"}
    )
    if (
        record.get("manifest_sha256") != integration.manifest_digest(manifest)
        or record.get("state_id") != union["state_id"]
        or record.get("status") not in {"succeeded", "failed"}
        or set(record.get("hashes", {})) != expected_roles
    ):
        raise RuntimeError("invalid diagnostic completion record")
    for role, expected in record["hashes"].items():
        if not paths[role].is_file() or sha256_file(paths[role]) != expected:
            raise RuntimeError("diagnostic cell artifact changed")
    for role in ("npz", "json", "failure"):
        if paths[role].exists() and role not in expected_roles:
            raise RuntimeError("diagnostic terminal leaves a mixed artifact")
    return record


def run_cell(
    manifest: dict[str, Any], out: Path, state_id: str
) -> dict[str, Any]:
    runtime_identity(manifest)
    matches = [
        union for union in manifest["unions"] if union["state_id"] == state_id
    ]
    if len(matches) != 1:
        raise RuntimeError("unknown diagnostic state cell")
    union = matches[0]
    paths = _paths(out, state_id)
    if paths["complete"].exists():
        return validate_cell(manifest, out, union)
    if _partial(paths):
        raise RuntimeError("partial prior diagnostic cell exists")
    try:
        state = capture.restore_capture(union["state"]["snapshot"])
        candidates = np.asarray(union["candidates"], dtype=np.float64).copy()
        before = ordinary._state_digest(state, candidates)
        arrays = ordinary.evaluate_mc_union(
            state, candidates, manifest["budget"], union["replicate_seeds"]
        )
        if before != ordinary._state_digest(state, candidates):
            raise RuntimeError("diagnostic evaluation mutated captured state")
        arrays["candidates"] = candidates
        integration.write_npz(paths["npz"], arrays)
        integration.write_json(
            paths["json"],
            {
                "schema_version": SCHEMA_VERSION,
                "status": "succeeded",
                "state_id": state_id,
                "budget": manifest["budget"],
                "replicate_seeds": union["replicate_seeds"],
                "candidate_row_ids": union["candidate_row_ids"],
                "candidate_union_sha256": union["candidate_union_sha256"],
                "case_tags": union["case_tags"],
                "contrasts": union["contrasts"],
                "actual_nodes": arrays["actual_nodes"].tolist(),
            },
        )
        status = "succeeded"
        hashes = {role: sha256_file(paths[role]) for role in ("npz", "json")}
    except Exception as error:  # noqa: BLE001
        for role in ("npz", "json"):
            if paths[role].exists():
                paths[role].unlink()
        integration.write_json(
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
    integration.write_json(
        paths["complete"],
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "manifest_sha256": integration.manifest_digest(manifest),
            "state_id": state_id,
            "hashes": hashes,
        },
    )
    return validate_cell(manifest, out, union)


def _logmeanexp(values: np.ndarray, axis: int | None = None):
    values = np.asarray(values, dtype=np.float64)
    count = values.size if axis is None else values.shape[axis]
    return np.logaddexp.reduce(values, axis=axis) - np.log(count)


def summarize(manifest: dict[str, Any], out: Path) -> dict[str, Any]:
    from noisy_acq_quadrature import judge_paired, raw_loss_assessment

    runtime_identity(manifest)
    previous_by_id: dict[str, dict[str, Any]] = {}
    if manifest.get("previous_summary") is not None:
        previous = json.loads(
            Path(manifest["previous_summary"]).read_text(encoding="utf-8")
        )
        previous_by_id = {
            item["contrast_id"]: item for item in previous["contrasts"]
        }
        expected_count = len(manifest["cases"]) * len(CONTRASTS)
        if (
            len(previous_by_id) != len(previous["contrasts"])
            or len(previous_by_id) != expected_count
        ):
            raise RuntimeError(
                "previous diagnostic contrast partition changed"
            )
    contrasts = []
    failed_states = []
    for union in manifest["unions"]:
        terminal = validate_cell(manifest, out, union)
        if terminal["status"] != "succeeded":
            failed_states.append(union["state_id"])
            continue
        with np.load(
            _paths(out, union["state_id"])["npz"], allow_pickle=False
        ) as stored:
            arrays = {key: stored[key] for key in stored.files}
        candidates = np.asarray(arrays["candidates"], dtype=np.float64)
        digest = hashlib.sha256(
            np.ascontiguousarray(candidates, dtype="<f8").tobytes()
        ).hexdigest()
        if digest != union["candidate_union_sha256"]:
            raise RuntimeError("diagnostic result candidate union changed")
        positions = {
            row_id: index
            for index, row_id in enumerate(union["candidate_row_ids"])
        }
        for spec in union["contrasts"]:
            arm = positions[spec["arm_row_id"]]
            baseline = positions[spec["baseline_row_id"]]
            pooled = float(
                _logmeanexp(arrays["log_residual"][:, arm])
                + np.mean(arrays["penalty"][:, arm])
                - _logmeanexp(arrays["log_residual"][:, baseline])
                - np.mean(arrays["penalty"][:, baseline])
            )
            prior = previous_by_id.get(spec["contrast_id"])
            if manifest["budget"] == BUDGETS[1] and prior is None:
                raise RuntimeError(
                    "diagnostic contrast lacks its exact predecessor"
                )
            if prior is not None and any(
                prior.get(key) != spec[key]
                for key in (
                    "comparison_tag",
                    "name",
                    "arm_role",
                    "baseline_role",
                    "arm_row_id",
                    "baseline_row_id",
                    "band",
                )
            ):
                raise RuntimeError("diagnostic predecessor contrast changed")
            judged = judge_paired(
                arrays["full_score"][:, arm],
                arrays["full_score"][:, baseline],
                spec["band"]["eps_F"],
                previous=None if prior is None else prior["judge"],
                replicate_seeds=union["replicate_seeds"],
                pooled_difference=pooled,
            )
            raw = raw_loss_assessment(
                arrays["log_reduction"][:, arm],
                arrays["log_reduction"][:, baseline],
                arrays["reference_log_residual"],
                previous=None if prior is None else prior["raw_loss"],
            )
            contrasts.append(
                {
                    **copy.deepcopy(spec),
                    "state_id": union["state_id"],
                    "budget": manifest["budget"],
                    "replicate_seeds": union["replicate_seeds"],
                    "pooled_difference": pooled,
                    "judge": judged,
                    "raw_loss": raw,
                    "penalty_active": bool(
                        np.any(arrays["penalty"][:, arm] > 0)
                        or np.any(arrays["penalty"][:, baseline] > 0)
                    ),
                }
            )
    by_case: dict[str, dict[str, Any]] = {}
    for case in manifest["cases"]:
        records = {
            item["name"]: item
            for item in contrasts
            if item["comparison_tag"] == case["comparison_tag"]
        }
        row = {
            "comparison_tag": case["comparison_tag"],
            "state_id": case["state_id"],
            "replicate": case["replicate"],
            "row_ids": case["row_ids"],
            "mc_eligibility": case["mc_eligibility"],
            "contrasts": records,
            "mechanism_status": "failed_or_missing",
            "raw_mechanism_status": "failed_or_missing",
            "raw_threshold": "loss_of_more_than_10_percent_of_baseline_raw_reduction",
            "causal_claim": False,
        }
        if len(records) == len(CONTRASTS):
            (
                row["mechanism_status"],
                row["raw_mechanism_status"],
            ) = classify_case_records(records)
        by_case[case["comparison_tag"]] = row
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SUMMARY_KIND,
        "manifest_sha256": integration.manifest_digest(manifest),
        "budget": manifest["budget"],
        "search_manifest_sha256": manifest["search_manifest_sha256"],
        "rqmc_summary_sha256": manifest["rqmc_summary_sha256"],
        "mc_summary_sha256": manifest["mc_summary_sha256"],
        "case_basis_sha256": manifest["case_basis_sha256"],
        "primary_holdout_comparison_count": PRIMARY_HOLDOUT_COMPARISONS,
        "case_tags": manifest["case_tags"],
        "scheduled_case_count": len(manifest["case_tags"]),
        "observed_case_count": sum(
            row["mechanism_status"] != "failed_or_missing"
            for row in by_case.values()
        ),
        "failed_states": failed_states,
        "cases": [by_case[tag] for tag in manifest["case_tags"]],
        "contrasts": sorted(contrasts, key=lambda item: item["contrast_id"]),
        "automatic_override": False,
        "tuning_use": False,
        "changes_e4_entry_gate": False,
        "causal_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--search-manifest", type=Path, required=True)
    prepare.add_argument("--search-out", type=Path, required=True)
    prepare.add_argument("--rqmc-summary", type=Path, required=True)
    prepare.add_argument("--mc-manifest", type=Path, required=True)
    prepare.add_argument("--mc-summary", type=Path, required=True)
    prepare.add_argument("--explicit-selection", type=Path, required=True)
    prepare.add_argument("--adapter-archive", type=Path, required=True)
    prepare.add_argument("--budget", type=int, choices=BUDGETS, required=True)
    prepare.add_argument("--previous-summary", type=Path)
    prepare.add_argument("--launch-ready", action="store_true")
    prepare.add_argument("--out-manifest", type=Path, required=True)
    cell = subparsers.add_parser("cell")
    cell.add_argument("--manifest", type=Path, required=True)
    cell.add_argument("--out", type=Path, required=True)
    cell.add_argument("--state-id", required=True)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare_manifest(
            search_manifest_path=args.search_manifest,
            search_out=args.search_out,
            rqmc_summary_path=args.rqmc_summary,
            mc_manifest_path=args.mc_manifest,
            mc_summary_path=args.mc_summary,
            explicit_selection_path=args.explicit_selection,
            adapter_archive=args.adapter_archive,
            budget=args.budget,
            previous_summary_path=args.previous_summary,
            launch_ready=args.launch_ready,
        )
        if (
            args.out_manifest.exists()
            or args.out_manifest.with_name(
                args.out_manifest.name + ".tmp"
            ).exists()
        ):
            raise RuntimeError("refusing to overwrite diagnostic manifest")
        integration.write_json(args.out_manifest, manifest)
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.command == "cell":
            run_cell(manifest, args.out, args.state_id)
        else:
            report = summarize(manifest, args.out)
            summary_path = args.out / "summary.json"
            if (
                summary_path.exists()
                or summary_path.with_name(summary_path.name + ".tmp").exists()
            ):
                raise RuntimeError("refusing to overwrite diagnostic summary")
            integration.write_json(summary_path, report)


if __name__ == "__main__":
    main()
