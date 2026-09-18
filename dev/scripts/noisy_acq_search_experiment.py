"""Manifest-driven E3 frozen-state search campaign.

This developer driver schedules paired S0--S3 selections, controlled timing,
and independent judging without fitting a GP or evaluating a target.  Search
policy lives in :mod:`noisy_acq_search`; this module only freezes allocation,
publishes hash-checked artifacts, and accounts for every scheduled cell.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import noisy_acq_experiment as capture
import noisy_acq_integration as integration
import noisy_acq_quadrature as quadrature
import noisy_acq_search as search
import noisy_acq_timing as timing
import numpy as np

SCHEMA_VERSION = 1
SELECTION_REPLICATES = 8
MASTER_SEED = 2026091701
JUDGE_BUDGETS = (4096, 8192, 16384, 32768, 65536)
# The F2 stages evaluate the package's quasi-Monte Carlo importance nodes
# as a drop-in for the production selection: every arm is the production
# search (S0), the treatments differ from the baseline only in the node
# rule switched on through its option. Each stage maps to the capture
# split it reads and seeds its own streams under its own name.
F2_STAGES = {"f2_development": "development", "f2_holdout": "holdout"}
F2_NODE_RULE = "scrambled_sobol_mixture"
F2_NODE_SAMPLES = 96
F2_ORDERS = ("axis", "index")
F2_TAGS = {"axis": "S0_qmc_sorted", "index": "S0_qmc_unsorted"}
HOLDOUT_STAGES = {"holdout", "f2_holdout"}
STAGES = {"development", "development_control", *HOLDOUT_STAGES, *F2_STAGES}
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent


def _source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        SCRIPTS / "noisy_acq_search.py",
        SCRIPTS / "noisy_acq_quadrature.py",
        SCRIPTS / "noisy_acq_integration.py",
        SCRIPTS / "noisy_acq_timing.py",
        SCRIPTS / "noisy_acq_experiment.py",
    ]
    return {
        path.relative_to(ROOT).as_posix(): integration.sha256_file(path)
        for path in paths
    }


def _identity(capture_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "integration_identity": integration.integration_identity(
            capture_manifest
        ),
        "search_source_hashes": _source_hashes(),
    }


def _thread_environment() -> dict[str, str | None]:
    values = {name: os.environ.get(name) for name in THREAD_KEYS}
    if any(value != "1" for value in values.values()):
        raise RuntimeError("Set every BLAS thread environment variable to 1")
    return values


def _read_selection(path: Path, stage: str) -> dict[str, Any]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("kind") != "noisy_acq_search_selection":
        raise RuntimeError("selection is not an E3 search selection")
    if not selection.get("frozen"):
        raise RuntimeError("search selection is not marked frozen")
    if selection.get("stage") != stage:
        raise RuntimeError("search selection stage does not match manifest")
    rule = selection.get("accurate_rule", {})
    method = str(rule.get("method"))
    budget = int(rule.get("budget", 0))
    if method not in {"mc", "stratified_mc", "stratified_rqmc"}:
        raise RuntimeError("unsupported frozen accurate integration method")
    if budget <= 0:
        raise RuntimeError(
            "frozen accurate integration budget must be positive"
        )
    normalized = copy.deepcopy(selection)
    normalized["accurate_rule"] = {**rule, "method": method, "budget": budget}
    if stage in F2_STAGES:
        if (
            selection.get("finalist_config") is not None
            or selection.get("strongest_arm_config") is not None
        ):
            raise RuntimeError("an F2 selection names no search arm")
        node_rule = selection.get("node_rule")
        if not isinstance(node_rule, dict):
            raise RuntimeError("an F2 selection requires node_rule")
        if (
            node_rule.get("kind") != F2_NODE_RULE
            or int(node_rule.get("samples", 0)) != F2_NODE_SAMPLES
            or list(node_rule.get("orders", [])) != list(F2_ORDERS)
        ):
            raise RuntimeError(
                "F2 node rule must be the frozen 96-node scrambled Sobol'"
                " mixture rule with both component orders"
            )
        normalized["node_rule"] = {
            **node_rule,
            "kind": F2_NODE_RULE,
            "samples": F2_NODE_SAMPLES,
            "orders": list(F2_ORDERS),
        }
        if stage == "f2_holdout":
            if selection.get("development_choice") not in F2_ORDERS:
                raise RuntimeError(
                    "F2 holdout must name the component order chosen on"
                    " development"
                )
            source_name = selection.get("development_manifest")
            if not source_name or not selection.get(
                "development_manifest_sha256"
            ):
                raise RuntimeError(
                    "F2 holdout must bind the preceding F2 development"
                    " manifest"
                )
            source_path = Path(source_name)
            if not source_path.is_absolute():
                source_path = (path.parent / source_path).resolve()
            if (
                not source_path.is_file()
                or integration.sha256_file(source_path)
                != selection["development_manifest_sha256"]
            ):
                raise RuntimeError("preceding F2 development manifest changed")
            normalized["development_manifest"] = str(source_path)
    elif stage == "development":
        if selection.get("finalist_config") is not None:
            raise RuntimeError("development selection cannot name a finalist")
    elif stage == "development_control":
        strongest = selection.get("strongest_arm_config")
        if not isinstance(strongest, dict):
            raise RuntimeError(
                "development control selection requires strongest_arm_config"
            )
        config = search.SearchConfig(**strongest)
        config.validate()
        if config.arm not in {"S1", "S2", "S3"}:
            raise RuntimeError(
                "development control must name one strongest search arm"
            )
        if (
            config.accurate_method != method
            or config.accurate_budget != budget
        ):
            raise RuntimeError(
                "strongest development arm must use the frozen accurate rule"
            )
        source_name = selection.get("development_manifest")
        if not source_name or not selection.get("development_manifest_sha256"):
            raise RuntimeError(
                "development control must bind the preceding development manifest"
            )
        source_path = Path(source_name)
        if not source_path.is_absolute():
            source_path = (path.parent / source_path).resolve()
        if (
            not source_path.is_file()
            or integration.sha256_file(source_path)
            != selection["development_manifest_sha256"]
        ):
            raise RuntimeError("preceding development manifest changed")
        normalized["development_manifest"] = str(source_path)
        normalized["strongest_arm_config"] = asdict(config)
    else:
        finalist = selection.get("finalist_config")
        if not isinstance(finalist, dict):
            raise RuntimeError("holdout selection requires finalist_config")
        config = search.SearchConfig(**finalist)
        config.validate()
        if config.arm not in {"S1", "S2", "S3"}:
            raise RuntimeError("holdout finalist must be S1, S2, or S3")
        if (
            config.accurate_method != method
            or config.accurate_budget != budget
        ):
            raise RuntimeError(
                "holdout finalist and accurate_rule must name one rule"
            )
        if selection.get("include_mc1600_control") is not True:
            raise RuntimeError(
                "holdout selection must retain the MC1600 control"
            )
        normalized["finalist_config"] = asdict(config)
    return normalized


def _config_record(
    arm: str, rule: dict[str, Any], *, template: dict[str, Any] | None = None
) -> dict[str, Any]:
    values = dict(template or {})
    values.update(
        arm=arm,
        accurate_method=rule["method"],
        accurate_budget=int(rule["budget"]),
    )
    config = search.SearchConfig(**values)
    config.validate()
    return asdict(config)


def _treatments(stage: str, selection: dict[str, Any]) -> list[dict[str, Any]]:
    rule = selection["accurate_rule"]
    baseline = {
        "tag": "S0",
        "role": "production_baseline",
        "config": _config_record("S0", rule),
        "accurate_seed_role": "unused_baseline",
    }
    if stage in F2_STAGES:
        node_rule = selection["node_rule"]
        return [
            baseline,
            *[
                {
                    "tag": F2_TAGS[order],
                    "role": "qmc_node_treatment",
                    "config": _config_record(
                        "S0",
                        rule,
                        template={
                            "importance_qmc": True,
                            "importance_qmc_samples": int(
                                node_rule["samples"]
                            ),
                            "importance_qmc_order": order,
                        },
                    ),
                    "accurate_seed_role": "unused_baseline",
                }
                for order in node_rule["orders"]
            ],
        ]
    if stage == "development":
        return [
            baseline,
            *[
                {
                    "tag": arm,
                    "role": "search_treatment",
                    "config": _config_record(arm, rule),
                    "accurate_seed_role": "shared_primary_rule",
                }
                for arm in ("S1", "S2", "S3")
            ],
        ]
    if stage == "development_control":
        original = dict(selection["strongest_arm_config"])
        treatments = [
            baseline,
            {
                "tag": "original_arm",
                "role": "frozen_development_strongest_arm",
                "config": original,
                "accurate_seed_role": "shared_primary_rule",
            },
        ]
        if not (
            original["accurate_method"] == "mc"
            and original["accurate_budget"] == 1600
        ):
            control = dict(original)
            control.update(accurate_method="mc", accurate_budget=1600)
            treatments.append(
                {
                    "tag": "mc1600_control",
                    "role": "frozen_development_plain_mc_control",
                    "config": control,
                    "accurate_seed_role": "method_specific_mc1600",
                }
            )
        return treatments

    finalist = dict(selection["finalist_config"])
    finalist_config = search.SearchConfig(**finalist)
    treatments = [
        baseline,
        {
            "tag": "finalist",
            "role": "frozen_finalist",
            "config": asdict(finalist_config),
            "accurate_seed_role": "shared_primary_rule",
        },
    ]
    if not (
        finalist_config.accurate_method == "mc"
        and finalist_config.accurate_budget == 1600
    ):
        control = asdict(finalist_config)
        control.update(accurate_method="mc", accurate_budget=1600)
        treatments.append(
            {
                "tag": "mc1600_control",
                "role": "frozen_plain_mc_control",
                "config": control,
                "accurate_seed_role": "method_specific_mc1600",
            }
        )
    return treatments


def prepare_manifest(
    capture_manifest_path: Path,
    captures: Path,
    selection_path: Path,
    stage: str,
) -> dict[str, Any]:
    """Prepare a locked E3 or F2 development or holdout allocation."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}")
    threads = _thread_environment()
    capture_manifest = json.loads(
        capture_manifest_path.read_text(encoding="utf-8")
    )
    capture.validate_manifest(capture_manifest, require_ready=True)
    selection = _read_selection(selection_path, stage)
    if stage == "f2_holdout":
        development = json.loads(
            Path(selection["development_manifest"]).read_text(encoding="utf-8")
        )
        if (
            development.get("stage") != "f2_development"
            or development.get("capture_manifest_sha256")
            != integration.sha256_file(capture_manifest_path)
            or capture.canonical(development.get("identity"))
            != capture.canonical(_identity(capture_manifest))
        ):
            raise RuntimeError(
                "F2 holdout does not match its F2 development source"
            )
    if stage == "development_control":
        development = json.loads(
            Path(selection["development_manifest"]).read_text(encoding="utf-8")
        )
        if (
            development.get("stage") != "development"
            or development.get("capture_manifest_sha256")
            != integration.sha256_file(capture_manifest_path)
            or capture.canonical(development.get("identity"))
            != capture.canonical(_identity(capture_manifest))
        ):
            raise RuntimeError(
                "development control does not match its development source"
            )
        strongest = selection["strongest_arm_config"]
        matching_treatments = [
            item
            for item in development.get("treatments", [])
            if item.get("tag") == strongest["arm"]
        ]
        if len(matching_treatments) != 1 or capture.canonical(
            matching_treatments[0].get("config")
        ) != capture.canonical(strongest):
            raise RuntimeError(
                "strongest arm config differs from the bound development treatment"
            )
    if stage in F2_STAGES:
        split = F2_STAGES[stage]
        seed_stage = stage
    else:
        split = "development" if stage == "development_control" else stage
        seed_stage = split
    states, missing = integration.discover_states(
        capture_manifest, captures, split
    )
    treatments = _treatments(stage, selection)
    cells = []
    for state in states:
        for replicate in range(SELECTION_REPLICATES):
            search_seed = integration.derive_seed(
                MASTER_SEED,
                seed_stage,
                state["state_id"],
                replicate,
                "candidate_search",
            )
            primary_seed = integration.derive_seed(
                MASTER_SEED,
                seed_stage,
                state["state_id"],
                replicate,
                "accurate_primary_rule",
            )
            for treatment in treatments:
                if treatment["accurate_seed_role"] == "method_specific_mc1600":
                    accurate_seed = integration.derive_seed(
                        MASTER_SEED,
                        seed_stage,
                        state["state_id"],
                        replicate,
                        "accurate_mc1600_control",
                    )
                else:
                    accurate_seed = primary_seed
                cells.append(
                    {
                        "state_id": state["state_id"],
                        "treatment_tag": treatment["tag"],
                        "replicate": replicate,
                        "search_seed": search_seed,
                        "accurate_seed": accurate_seed,
                    }
                )
    nonbaseline = [item for item in treatments if item["tag"] != "S0"]
    timing_cells = [
        {
            "state_id": state["state_id"],
            "treatment_tag": treatment["tag"],
            "timing_seed": integration.derive_seed(
                MASTER_SEED,
                seed_stage,
                state["state_id"],
                treatment["tag"],
                "paired_timing",
            ),
        }
        for state in states
        for treatment in nonbaseline
    ]
    allocation_cells = [
        {
            **cell,
            "role": role,
            "allocation_seed": integration.derive_seed(
                cell["timing_seed"], role, "allocation_measurement"
            ),
        }
        for cell in timing_cells
        for role in ("baseline", "treatment")
    ]
    judge = {
        state["state_id"]: {
            "comparison_seeds": {
                str(budget): [
                    integration.derive_seed(
                        MASTER_SEED,
                        seed_stage,
                        state["state_id"],
                        budget,
                        replicate,
                        "search_judge",
                    )
                    for replicate in range(SELECTION_REPLICATES)
                ]
                for budget in JUDGE_BUDGETS
            },
            "band_pilot_budget": JUDGE_BUDGETS[0],
            "band_pilot_seeds": [
                integration.derive_seed(
                    MASTER_SEED,
                    seed_stage,
                    state["state_id"],
                    replicate,
                    "search_band_pilot",
                )
                for replicate in range(SELECTION_REPLICATES)
            ],
        }
        for state in states
    }
    expected_states = 2 * len(capture_manifest["allocation"])
    direct_control_available = any(
        spec["kind"] == "direct_integration_control"
        for spec in _contrast_specs({"treatments": treatments})
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "search",
        "purpose": (
            "noisy-acquisition F2 frozen-state importance-node evaluation"
            if stage in F2_STAGES
            else "noisy-acquisition E3 frozen-state search"
        ),
        "launch_ready": False,
        "stage": stage,
        "split": split,
        "holdout_locked": stage in HOLDOUT_STAGES,
        "capture_manifest": str(capture_manifest_path.resolve()),
        "capture_manifest_sha256": integration.sha256_file(
            capture_manifest_path
        ),
        "captures": str(captures.resolve()),
        "selection": selection,
        "selection_file": str(selection_path.resolve()),
        "selection_sha256": integration.sha256_file(selection_path),
        "identity": _identity(capture_manifest),
        "thread_environment": threads,
        "master_seed": MASTER_SEED,
        "replicates": SELECTION_REPLICATES,
        "states": states,
        "missing_states": missing,
        "expected_state_count": expected_states,
        "treatments": treatments,
        "cells": cells,
        "cell_count": len(cells),
        "scheduled_cell_count_including_missing_states": (
            expected_states * len(treatments) * SELECTION_REPLICATES
        ),
        "timing_cells": timing_cells,
        "timing_cell_count": len(timing_cells),
        "scheduled_timing_count_including_missing_states": (
            expected_states * len(nonbaseline)
        ),
        "allocation_cells": allocation_cells,
        "allocation_cell_count": len(allocation_cells),
        "scheduled_allocation_count_including_missing_states": (
            expected_states * len(nonbaseline) * 2
        ),
        "judge_budgets": list(JUDGE_BUDGETS),
        "initial_judge_budget": JUDGE_BUDGETS[0],
        "judge": judge,
        "direct_control_timing": {
            "classification": (
                "descriptive_only"
                if direct_control_available
                else "not_applicable_duplicate_rule"
            ),
            "reason": (
                "Each search arm is timed against S0; those separate pairings do "
                "not estimate a direct paired MC1600-versus-original speed effect."
                if direct_control_available
                else "The frozen original arm already uses ordinary MC1600."
            ),
        },
    }


def validate_manifest(
    manifest: dict[str, Any], *, require_ready: bool
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported E3 manifest schema")
    if manifest.get("kind") != "search":
        raise RuntimeError("manifest is not an E3 search manifest")
    if manifest.get("stage") not in STAGES:
        raise RuntimeError("unknown search stage")
    if require_ready and not manifest.get("launch_ready"):
        raise RuntimeError("search manifest is not marked launch_ready")
    if manifest["stage"] in HOLDOUT_STAGES and manifest.get(
        "holdout_locked", True
    ):
        raise RuntimeError("holdout search manifest remains locked")
    if manifest.get("replicates") != SELECTION_REPLICATES:
        raise RuntimeError("search replicate allocation changed")
    if manifest.get("master_seed") != MASTER_SEED:
        raise RuntimeError("search master seed changed")
    if manifest.get("judge_budgets") != list(JUDGE_BUDGETS):
        raise RuntimeError("judge escalation ladder changed")
    expected_treatments = _treatments(
        manifest["stage"], manifest.get("selection", {})
    )
    if capture.canonical(manifest.get("treatments")) != capture.canonical(
        expected_treatments
    ):
        raise RuntimeError(
            "search treatments differ from the frozen selection"
        )
    treatment_tags = [item["tag"] for item in manifest.get("treatments", [])]
    if not treatment_tags or len(treatment_tags) != len(set(treatment_tags)):
        raise RuntimeError("search treatments are missing or duplicated")
    for treatment in manifest["treatments"]:
        config = search.SearchConfig(**treatment["config"])
        config.validate()
        if config.arm != treatment["config"]["arm"]:
            raise RuntimeError("search treatment config is inconsistent")
    state_ids = {state["state_id"] for state in manifest.get("states", [])}
    cell_keys = [
        (cell["state_id"], cell["treatment_tag"], cell["replicate"])
        for cell in manifest.get("cells", [])
    ]
    if len(cell_keys) != manifest.get("cell_count") or len(cell_keys) != len(
        set(cell_keys)
    ):
        raise RuntimeError("search cell allocation is inconsistent")
    if any(
        cell["state_id"] not in state_ids
        or cell["treatment_tag"] not in treatment_tags
        or cell["replicate"] not in range(SELECTION_REPLICATES)
        for cell in manifest["cells"]
    ):
        raise RuntimeError(
            "search cell refers to an unfrozen state or treatment"
        )
    grouped_cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for cell in manifest["cells"]:
        grouped_cells.setdefault(
            (cell["state_id"], cell["replicate"]), []
        ).append(cell)
    expected_groups = {
        (state_id, replicate)
        for state_id in state_ids
        for replicate in range(SELECTION_REPLICATES)
    }
    if set(grouped_cells) != expected_groups:
        raise RuntimeError("paired search replicate allocation is incomplete")
    for group in grouped_cells.values():
        if {cell["treatment_tag"] for cell in group} != set(treatment_tags):
            raise RuntimeError("paired search replicate is incomplete")
        if len({cell["search_seed"] for cell in group}) != 1:
            raise RuntimeError("paired search arms do not share a search seed")
        primary = [
            cell["accurate_seed"]
            for cell in group
            if _treatment(manifest, cell["treatment_tag"])[
                "accurate_seed_role"
            ]
            == "shared_primary_rule"
        ]
        if primary and len(set(primary)) != 1:
            raise RuntimeError(
                "primary search arms do not share an accurate rule"
            )
    timing_keys = [
        (cell["state_id"], cell["treatment_tag"])
        for cell in manifest.get("timing_cells", [])
    ]
    if len(timing_keys) != manifest.get("timing_cell_count") or len(
        timing_keys
    ) != len(set(timing_keys)):
        raise RuntimeError("search timing allocation is inconsistent")
    if any(
        cell["state_id"] not in state_ids
        or cell["treatment_tag"] not in treatment_tags
        or cell["treatment_tag"] == "S0"
        for cell in manifest["timing_cells"]
    ):
        raise RuntimeError("timing cell refers to an invalid paired treatment")
    expected_timing = {
        (state_id, tag)
        for state_id in state_ids
        for tag in treatment_tags
        if tag != "S0"
    }
    if set(timing_keys) != expected_timing:
        raise RuntimeError("paired timing allocation is incomplete")
    allocation_keys = [
        (cell["state_id"], cell["treatment_tag"], cell["role"])
        for cell in manifest.get("allocation_cells", [])
    ]
    expected_allocation = {
        (state_id, tag, role)
        for state_id, tag in expected_timing
        for role in ("baseline", "treatment")
    }
    if (
        len(allocation_keys) != manifest.get("allocation_cell_count")
        or len(allocation_keys) != len(set(allocation_keys))
        or set(allocation_keys) != expected_allocation
    ):
        raise RuntimeError("allocation-measurement schedule is inconsistent")
    if any(
        cell["allocation_seed"]
        != integration.derive_seed(
            cell["timing_seed"], cell["role"], "allocation_measurement"
        )
        for cell in manifest["allocation_cells"]
    ):
        raise RuntimeError("allocation-measurement seed changed")
    for state_id in state_ids:
        seed_record = manifest.get("judge", {}).get(state_id)
        if seed_record is None:
            raise RuntimeError("judge seed allocation is missing a state")
        for budget in manifest.get("judge_budgets", []):
            if len(seed_record["comparison_seeds"].get(str(budget), [])) != 8:
                raise RuntimeError("judge comparison seed allocation changed")
        if len(seed_record.get("band_pilot_seeds", [])) != 8:
            raise RuntimeError("judge pilot seed allocation changed")


def runtime_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    validate_manifest(manifest, require_ready=True)
    capture_manifest_path = Path(manifest["capture_manifest"])
    if (
        integration.sha256_file(capture_manifest_path)
        != manifest["capture_manifest_sha256"]
    ):
        raise RuntimeError("capture manifest changed")
    capture_manifest = json.loads(
        capture_manifest_path.read_text(encoding="utf-8")
    )
    if (
        integration.sha256_file(manifest["selection_file"])
        != manifest["selection_sha256"]
    ):
        raise RuntimeError("frozen search selection changed")
    if manifest["stage"] == "development_control":
        development_path = manifest["selection"]["development_manifest"]
        if (
            integration.sha256_file(development_path)
            != manifest["selection"]["development_manifest_sha256"]
        ):
            raise RuntimeError("preceding development manifest changed")
    actual = _identity(capture_manifest)
    if capture.canonical(actual) != capture.canonical(manifest["identity"]):
        raise RuntimeError("search, integration, or capture source changed")
    if _thread_environment() != manifest["thread_environment"]:
        raise RuntimeError("thread environment changed")
    for state in manifest["states"]:
        base = Path(state["snapshot"])
        actual_hashes = {
            suffix: integration.sha256_file(
                base.parent / f"{base.name}{suffix}"
            )
            for suffix in (".json", ".npz")
        }
        if actual_hashes != state["snapshot_hashes"]:
            raise RuntimeError(f"captured state changed: {state['state_id']}")
        capture_completion = (
            Path(manifest["captures"])
            / "records"
            / f"{state['label']}_seed{state['seed']}.complete.json"
        )
        if (
            not capture_completion.is_file()
            or integration.sha256_file(capture_completion)
            != state["capture_completion_sha256"]
        ):
            raise RuntimeError(
                f"capture completion changed: {state['state_id']}"
            )
    return actual


def _treatment(manifest: dict[str, Any], tag: str) -> dict[str, Any]:
    matches = [item for item in manifest["treatments"] if item["tag"] == tag]
    if len(matches) != 1:
        raise RuntimeError(f"unknown search treatment {tag!r}")
    return matches[0]


def cell_tag(cell: dict[str, Any]) -> str:
    return (
        f"{cell['state_id']}__{cell['treatment_tag']}"
        f"__r{cell['replicate']}"
    )


def timing_tag(cell: dict[str, Any]) -> str:
    return f"{cell['state_id']}__{cell['treatment_tag']}__timing"


def allocation_tag(cell: dict[str, Any]) -> str:
    return (
        f"{cell['state_id']}__{cell['treatment_tag']}"
        f"__allocation_{cell['role']}"
    )


def _paths(out: Path, tag: str, kind: str) -> dict[str, Path]:
    directory = out / kind
    return {
        "npz": directory / f"{tag}.npz",
        "json": directory / f"{tag}.json",
        "failure": out / "records" / f"{tag}.{kind}.failure.json",
        "complete": out / "records" / f"{tag}.{kind}.complete.json",
    }


def _validate_terminal(
    manifest: dict[str, Any],
    paths: dict[str, Path],
    expected_cell: dict[str, Any],
) -> dict[str, Any]:
    record = json.loads(paths["complete"].read_text(encoding="utf-8"))
    if record.get("manifest_sha256") != integration.manifest_digest(manifest):
        raise RuntimeError("terminal record belongs to another manifest")
    if record.get("cell") != expected_cell or record.get("status") not in {
        "succeeded",
        "failed",
    }:
        raise RuntimeError("invalid terminal search record")
    expected = (
        {"json", "npz"}
        if record["status"] == "succeeded" and paths.get("npz") is not None
        else ({"json"} if record["status"] == "succeeded" else {"failure"})
    )
    if set(record.get("hashes", {})) != expected:
        raise RuntimeError("terminal artifact set is inconsistent")
    for role, expected_hash in record["hashes"].items():
        if (
            not paths[role].is_file()
            or integration.sha256_file(paths[role]) != expected_hash
        ):
            raise RuntimeError("terminal search artifact changed")
    for role in ("npz", "json", "failure"):
        path = paths.get(role)
        if path is not None and path.exists() and role not in record["hashes"]:
            raise RuntimeError("terminal record leaves an untracked artifact")
    return record


def _publish_terminal(
    manifest: dict[str, Any],
    paths: dict[str, Path],
    cell: dict[str, Any],
    status: str,
    hashes: dict[str, str],
) -> dict[str, Any]:
    if status == "succeeded":
        runtime_identity(manifest)
    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "manifest_sha256": integration.manifest_digest(manifest),
        "cell": cell,
        "hashes": hashes,
    }
    integration.write_json(paths["complete"], completion)
    return _validate_terminal(manifest, paths, cell)


def _partial(paths: dict[str, Path]) -> list[Path]:
    return [
        path
        for role, path in paths.items()
        if role != "complete" and path is not None and path.exists()
    ]


def _clear_success_artifacts(paths: dict[str, Path]) -> None:
    """Remove artifacts from a cell that failed before terminal publication."""
    for role in ("npz", "json"):
        path = paths.get(role)
        if path is not None and path.exists():
            path.unlink()


def _state_descriptor(
    manifest: dict[str, Any], state_id: str
) -> dict[str, Any]:
    matches = [
        state for state in manifest["states"] if state["state_id"] == state_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"unknown captured state {state_id!r}")
    return matches[0]


def _state_digest(state: dict[str, Any]) -> str:
    return integration.state_digest(
        state, np.empty((0, state["vp"].D), dtype=np.float64)
    )


def run_cell(
    manifest: dict[str, Any], out: Path, cell: dict[str, Any]
) -> dict[str, Any]:
    """Run and atomically publish one untimed, panel-retaining selection."""
    runtime_identity(manifest)
    tag = cell_tag(cell)
    paths = _paths(out, tag, "selection")
    if paths["complete"].exists():
        return _validate_terminal(manifest, paths, cell)
    if _partial(paths):
        raise RuntimeError(f"{tag}: partial prior selection artifacts exist")
    descriptor = _state_descriptor(manifest, cell["state_id"])
    treatment = _treatment(manifest, cell["treatment_tag"])
    try:
        state = capture.restore_capture(descriptor["snapshot"])
        before = _state_digest(state)
        config = search.SearchConfig(**treatment["config"])
        result = search.select_candidate(
            state,
            config,
            search_seed=cell["search_seed"],
            accurate_seed=cell["accurate_seed"],
            retain_panel=True,
        )
        if before != _state_digest(state):
            raise RuntimeError(
                "search selection mutated captured state or RNG"
            )
        arrays = {
            "selected": np.asarray(result["selected"], dtype=np.float64),
            "coarse_candidates": np.asarray(
                result["coarse_candidates"], dtype=np.float64
            ),
            "coarse_scores": np.asarray(result["coarse_scores"]),
            "coarse_nodes": np.asarray(result["coarse_nodes"]),
        }
        for key in (
            "shortlist_indices",
            "shortlist_scores",
            "cache_indices",
            "coarse_winner",
        ):
            if key in result:
                arrays[key] = np.asarray(result[key])
        integration.write_npz(paths["npz"], arrays)
        details = {
            key: value
            for key, value in result.items()
            if key not in arrays and not isinstance(value, np.ndarray)
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "succeeded",
            "cell": cell,
            "state": descriptor,
            "treatment": treatment,
            "selected_row_sha256": _row_id(arrays["selected"][0]),
            "coarse_panel_sha256": capture.digest(
                (arrays["coarse_candidates"], arrays["coarse_scores"])
            ),
            "timing_kind": "untimed_selection_provenance",
            "details": details,
        }
        integration.write_json(paths["json"], report)
        return _publish_terminal(
            manifest,
            paths,
            cell,
            "succeeded",
            {
                "npz": integration.sha256_file(paths["npz"]),
                "json": integration.sha256_file(paths["json"]),
            },
        )
    except Exception as error:  # noqa: BLE001 - failed cells remain scheduled
        _clear_success_artifacts(paths)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "cell": cell,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        integration.write_json(paths["failure"], failure)
        return _publish_terminal(
            manifest,
            paths,
            cell,
            "failed",
            {"failure": integration.sha256_file(paths["failure"])},
        )


def _timing_factory(
    state: dict[str, Any],
    state_id: str,
    treatment: dict[str, Any],
):
    config = search.SearchConfig(**treatment["config"])

    def prepare(paired_seed: int):
        accurate_seed = integration.derive_seed(
            int(paired_seed),
            state_id,
            treatment["accurate_seed_role"],
            config.accurate_method,
            config.accurate_budget,
            "timing_rule",
        )
        return search.prepare_selection(
            state,
            config,
            search_seed=int(paired_seed),
            accurate_seed=accurate_seed,
            retain_panel=False,
        )

    return prepare


TIMING_RESULT_KEYS = (
    "arm",
    "elapsed_seconds",
    "search_seed",
    "accurate_seed",
    "target_called",
    "cache_index",
    "coarse_candidate_rows",
    "production_candidate_rows",
    "coarse_winner_index",
    "coarse_minimum",
    "generated_count",
    "importance_node_count",
    "accurate_candidate_rows",
    "local_iterations",
    "fallback_reason",
    "refinement_stop_reason",
)


def _project_timing_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the small provenance of a timed selection, not its arrays.

    A timing round retains the selected row and scalar counts; the panel,
    candidate cache indices and other arrays belong to the untimed
    selection cells.
    """
    projected = {
        key: result[key] for key in TIMING_RESULT_KEYS if key in result
    }
    if "selected" in result:
        selected = np.asarray(result["selected"], dtype=np.float64).reshape(-1)
        projected["selected"] = selected.tolist()
        projected["selected_row_sha256"] = _row_id(selected)
    return projected


def run_timing_cell(
    manifest: dict[str, Any], out: Path, cell: dict[str, Any]
) -> dict[str, Any]:
    """Run seven quiet paired rounds against exact S0."""
    runtime_identity(manifest)
    tag = timing_tag(cell)
    paths = _paths(out, tag, "timing")
    paths["npz"] = None
    if paths["complete"].exists():
        return _validate_terminal(manifest, paths, cell)
    if _partial(paths):
        raise RuntimeError(f"{tag}: partial prior timing artifacts exist")
    descriptor = _state_descriptor(manifest, cell["state_id"])
    baseline = _treatment(manifest, "S0")
    treatment = _treatment(manifest, cell["treatment_tag"])
    try:
        state = capture.restore_capture(descriptor["snapshot"])
        before = _state_digest(state)
        report = timing.time_pair(
            _timing_factory(state, cell["state_id"], baseline),
            _timing_factory(state, cell["state_id"], treatment),
            seed=cell["timing_seed"],
        )
        if before != _state_digest(state):
            raise RuntimeError("paired timing mutated captured state or RNG")
        for row in report.get("rounds", []):
            for name in ("baseline", "treatment"):
                if name in row and "result" in row[name]:
                    row[name]["result"] = _project_timing_result(
                        row[name]["result"]
                    )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "succeeded",
            "cell": cell,
            "state": descriptor,
            "baseline": baseline,
            "treatment": treatment,
            "timing": report,
        }
        integration.write_json(paths["json"], payload)
        return _publish_terminal(
            manifest,
            paths,
            cell,
            "succeeded",
            {"json": integration.sha256_file(paths["json"])},
        )
    except Exception as error:  # noqa: BLE001
        _clear_success_artifacts(paths)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "cell": cell,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        integration.write_json(paths["failure"], failure)
        return _publish_terminal(
            manifest,
            paths,
            cell,
            "failed",
            {"failure": integration.sha256_file(paths["failure"])},
        )


def run_allocation_cell(
    manifest: dict[str, Any], out: Path, cell: dict[str, Any]
) -> dict[str, Any]:
    """Measure one arm's allocations in its own fresh worker process."""
    runtime_identity(manifest)
    tag = allocation_tag(cell)
    paths = _paths(out, tag, "allocation")
    paths["npz"] = None
    if paths["complete"].exists():
        return _validate_terminal(manifest, paths, cell)
    if _partial(paths):
        raise RuntimeError(f"{tag}: partial prior allocation artifacts exist")
    descriptor = _state_descriptor(manifest, cell["state_id"])
    treatment = _treatment(
        manifest,
        "S0" if cell["role"] == "baseline" else cell["treatment_tag"],
    )
    try:
        state = capture.restore_capture(descriptor["snapshot"])
        before = _state_digest(state)
        report = timing.measure_allocations(
            _timing_factory(state, cell["state_id"], treatment),
            seed=cell["allocation_seed"],
        )
        if before != _state_digest(state):
            raise RuntimeError(
                "allocation measurement mutated captured state or RNG"
            )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "succeeded",
            "cell": cell,
            "state": descriptor,
            "measured_treatment": treatment,
            "allocation": report,
        }
        integration.write_json(paths["json"], payload)
        return _publish_terminal(
            manifest,
            paths,
            cell,
            "succeeded",
            {"json": integration.sha256_file(paths["json"])},
        )
    except Exception as error:  # noqa: BLE001
        _clear_success_artifacts(paths)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "cell": cell,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        integration.write_json(paths["failure"], failure)
        return _publish_terminal(
            manifest,
            paths,
            cell,
            "failed",
            {"failure": integration.sha256_file(paths["failure"])},
        )


def _row_bytes(row: np.ndarray) -> bytes:
    return np.ascontiguousarray(row, dtype="<f8").reshape(-1).tobytes()


def _row_id(row: np.ndarray) -> str:
    return "x_" + hashlib.sha256(_row_bytes(row)).hexdigest()


def build_judge_union(
    manifest: dict[str, Any], out: Path, state_id: str
) -> dict[str, Any]:
    """Build one shared union without discarding successful selection pairs."""
    rows: list[np.ndarray] = []
    row_lookup: dict[str, int] = {}
    provenance: dict[str, list[dict[str, Any]]] = {}
    baseline_by_replicate: dict[str, str] = {}
    treatment_by_tag: dict[str, dict[str, str]] = {
        item["tag"]: {}
        for item in manifest["treatments"]
        if item["tag"] != "S0"
    }
    accurate_seeds: dict[str, dict[str, int]] = {}
    missing_selections: list[dict[str, Any]] = []
    descriptor = _state_descriptor(manifest, state_id)
    state = capture.restore_capture(descriptor["snapshot"])
    before = _state_digest(state)
    reference_candidates, _, _, reference_order = integration.candidate_panel(
        state, descriptor, "full"
    )
    if before != _state_digest(state):
        raise RuntimeError(
            "reference-shortlist construction mutated captured state"
        )

    def include(row: np.ndarray, source: dict[str, Any]) -> str:
        selected = np.asarray(row, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(selected)):
            raise RuntimeError("judge union contains a nonfinite coordinate")
        row_id = _row_id(selected)
        if row_id in row_lookup:
            if _row_bytes(rows[row_lookup[row_id]]) != _row_bytes(selected):
                raise RuntimeError("selected-coordinate digest collision")
        else:
            row_lookup[row_id] = len(rows)
            rows.append(selected.copy())
            provenance[row_id] = []
        provenance[row_id].append(source)
        return row_id

    fixed_reference_row_ids = []
    for rank, global_index in enumerate(reference_order[:32]):
        fixed_reference_row_ids.append(
            include(
                reference_candidates[int(global_index)],
                {
                    "source": "captured_fixed_reference_shortlist",
                    "rank": rank,
                    "captured_sieve_index": int(global_index),
                },
            )
        )
    fixed_row_id = fixed_reference_row_ids[0]
    cells = sorted(
        (cell for cell in manifest["cells"] if cell["state_id"] == state_id),
        key=lambda cell: (cell["replicate"], cell["treatment_tag"]),
    )
    for cell in cells:
        tag = cell_tag(cell)
        paths = _paths(out, tag, "selection")
        if not paths["complete"].is_file():
            missing_selections.append(
                {"cell": cell, "cell_tag": tag, "reason": "selection_missing"}
            )
            continue
        terminal = _validate_terminal(manifest, paths, cell)
        if terminal["status"] != "succeeded":
            missing_selections.append(
                {"cell": cell, "cell_tag": tag, "reason": "selection_failed"}
            )
            continue
        with np.load(paths["npz"], allow_pickle=False) as data:
            selected = np.asarray(data["selected"], dtype=np.float64)[0]
        row_id = include(
            selected,
            {
                "source": "selected_candidate",
                "replicate": cell["replicate"],
                "treatment_tag": cell["treatment_tag"],
                "cell_tag": tag,
            },
        )
        replicate = str(cell["replicate"])
        if cell["treatment_tag"] == "S0":
            baseline_by_replicate[replicate] = row_id
        else:
            treatment_by_tag[cell["treatment_tag"]][replicate] = row_id
        accurate_seeds.setdefault(replicate, {})[cell["treatment_tag"]] = cell[
            "accurate_seed"
        ]
    candidates = np.vstack(rows).astype(np.float64, copy=False)
    candidate_row_ids = [None] * len(rows)
    for row_id, index in row_lookup.items():
        candidate_row_ids[index] = row_id
    if any(row_id is None for row_id in candidate_row_ids):
        raise RuntimeError("judge union candidate identifiers are incomplete")
    fixed_band_candidate = candidates[[row_lookup[fixed_row_id]]].copy()
    candidate_hash = hashlib.sha256(
        np.ascontiguousarray(candidates, dtype="<f8").tobytes()
    ).hexdigest()
    seed_record = manifest["judge"][state_id]
    union_basis = {
        "state_id": state_id,
        "candidate_union_sha256": candidate_hash,
        "baseline_by_replicate": baseline_by_replicate,
        "treatment_by_tag": treatment_by_tag,
    }
    union_id = integration.manifest_digest(union_basis)
    return {
        "union_id": union_id,
        "state_id": state_id,
        "split": manifest["split"],
        "holdout_locked": manifest.get("holdout_locked", False),
        "manifest_sha256": integration.manifest_digest(manifest),
        "source_identity_sha256": capture.digest(manifest["identity"]),
        "candidates": candidates,
        "candidate_row_ids": candidate_row_ids,
        "candidate_union_sha256": candidate_hash,
        "candidate_provenance": provenance,
        "fixed_reference_row_ids": fixed_reference_row_ids,
        "fixed_band_candidate": fixed_band_candidate,
        "fixed_band_row_id": fixed_row_id,
        "baseline_by_replicate": baseline_by_replicate,
        "treatment_by_tag": treatment_by_tag,
        "accurate_seed_roles": accurate_seeds,
        "missing_selections": missing_selections,
        "comparison_seeds": seed_record["comparison_seeds"],
        "band_pilot_budget": seed_record["band_pilot_budget"],
        "band_pilot_seeds": seed_record["band_pilot_seeds"],
        "prior_union_id": None,
        "resolved_records": [],
    }


def _judge_cell(
    state_id: str, budget: int, previous_summary_sha256: str | None = None
) -> dict[str, Any]:
    return {
        "state_id": state_id,
        "budget": int(budget),
        "previous_summary_sha256": previous_summary_sha256,
    }


def judge_tag(state_id: str, budget: int) -> str:
    return f"{state_id}__judge_b{budget}"


def _restrict_escalated_union(
    union: dict[str, Any], previous: dict[str, Any]
) -> dict[str, Any]:
    """Retain only pending pairs and frozen reference rows at a higher budget."""
    state_comparisons = [
        item
        for item in previous["comparisons"]
        if item["cell"]["state_id"] == union["state_id"]
    ]
    pending = [
        item for item in state_comparisons if comparison_needs_escalation(item)
    ]
    if not pending:
        raise RuntimeError("state has no unresolved comparison to escalate")
    keep = set(union["fixed_reference_row_ids"])
    resolved_records = []
    for item in state_comparisons:
        if item in pending:
            keep.update((item["selected_row_id"], item["baseline_row_id"]))
            for key in (
                "pooled_union_best_row_id",
                "mean_log_union_best_row_id",
            ):
                if item.get(key) is not None:
                    keep.add(item[key])
        else:
            resolved_records.append(
                {
                    "comparison_tag": item["comparison_tag"],
                    "judge_budget": item["judge_budget"],
                    "assessed_union": item["assessed_union"],
                }
            )
    row_ids = union["candidate_row_ids"]
    absent = keep - set(row_ids)
    if absent:
        raise RuntimeError("escalated union lost a prior selected or best row")
    indices = [index for index, row_id in enumerate(row_ids) if row_id in keep]
    restricted = copy.deepcopy(union)
    restricted["candidates"] = union["candidates"][indices].copy()
    restricted["candidate_row_ids"] = [row_ids[index] for index in indices]
    restricted["candidate_provenance"] = {
        row_id: union["candidate_provenance"][row_id]
        for row_id in restricted["candidate_row_ids"]
    }
    restricted["baseline_by_replicate"] = {}
    restricted["treatment_by_tag"] = {
        tag: {} for tag in union["treatment_by_tag"]
    }
    for item in pending:
        replicate = str(item["cell"]["replicate"])
        for tag, row_id in (
            (item["cell"]["arm_tag"], item["selected_row_id"]),
            (item["cell"]["reference_tag"], item["baseline_row_id"]),
        ):
            if tag == "S0":
                restricted["baseline_by_replicate"][replicate] = row_id
            else:
                restricted["treatment_by_tag"][tag][replicate] = row_id
    candidate_hash = hashlib.sha256(
        np.ascontiguousarray(restricted["candidates"], dtype="<f8").tobytes()
    ).hexdigest()
    prior_union_ids = sorted(
        {item["assessed_union"]["union_id"] for item in state_comparisons}
    )
    restricted["candidate_union_sha256"] = candidate_hash
    restricted["prior_union_id"] = prior_union_ids
    restricted["resolved_records"] = resolved_records
    restricted["union_id"] = integration.manifest_digest(
        {
            "state_id": union["state_id"],
            "candidate_union_sha256": candidate_hash,
            "baseline_by_replicate": restricted["baseline_by_replicate"],
            "treatment_by_tag": restricted["treatment_by_tag"],
            "prior_union_id": prior_union_ids,
        }
    )
    return restricted


def run_judge_cell(
    manifest: dict[str, Any],
    out: Path,
    state_id: str,
    budget: int,
    previous_summary: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one fixed selected-coordinate union with common judge rules."""
    runtime_identity(manifest)
    if budget not in manifest["judge_budgets"]:
        raise ValueError(
            "judge budget is outside the frozen escalation ladder"
        )
    if budget == manifest["initial_judge_budget"]:
        if previous_summary is not None:
            raise ValueError(
                "initial judge budget cannot use a previous summary"
            )
        previous_sha256 = None
    else:
        if previous_summary is None:
            raise ValueError(
                "escalated judge budget requires a previous summary"
            )
        previous = _load_previous_summary(manifest, budget, previous_summary)
        previous_sha256 = integration.sha256_file(previous_summary)
    if budget == manifest["initial_judge_budget"]:
        previous = None
    cell = _judge_cell(state_id, budget, previous_sha256)
    paths = _paths(out, judge_tag(state_id, budget), "judge")
    if paths["complete"].exists():
        return _validate_terminal(manifest, paths, cell)
    if _partial(paths):
        raise RuntimeError("partial prior judge artifacts exist")
    descriptor = _state_descriptor(manifest, state_id)
    try:
        union = build_judge_union(manifest, out, state_id)
        if previous is not None:
            union = _restrict_escalated_union(union, previous)
        union["prior_summary_sha256"] = previous_sha256
        state = capture.restore_capture(descriptor["snapshot"])
        before = _state_digest(state)
        arrays = integration.evaluate_judge_matrix(
            state,
            union["candidates"],
            budget,
            union["comparison_seeds"][str(budget)],
            band_candidate=union["fixed_band_candidate"],
            band_pilot_seeds=(
                union["band_pilot_seeds"]
                if budget == manifest["initial_judge_budget"]
                else []
            ),
            band_pilot_budget=union["band_pilot_budget"],
        )
        arrays["candidates"] = union["candidates"].copy()
        if before != _state_digest(state):
            raise RuntimeError("search judge mutated captured state or RNG")
        integration.write_npz(paths["npz"], arrays)
        report_union = {
            key: value
            for key, value in union.items()
            if key not in {"candidates", "fixed_band_candidate"}
        }
        report_union["fixed_band_candidate"] = union[
            "fixed_band_candidate"
        ].tolist()
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "succeeded",
            "cell": cell,
            "state": descriptor,
            "union": report_union,
            "source_identity_sha256": capture.digest(manifest["identity"]),
            "actual_nodes": arrays["actual_nodes"].tolist(),
        }
        integration.write_json(paths["json"], report)
        return _publish_terminal(
            manifest,
            paths,
            cell,
            "succeeded",
            {
                "npz": integration.sha256_file(paths["npz"]),
                "json": integration.sha256_file(paths["json"]),
            },
        )
    except Exception as error:  # noqa: BLE001
        _clear_success_artifacts(paths)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "cell": cell,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        integration.write_json(paths["failure"], failure)
        return _publish_terminal(
            manifest,
            paths,
            cell,
            "failed",
            {"failure": integration.sha256_file(paths["failure"])},
        )


def comparison_needs_escalation(comparison: dict[str, Any]) -> bool:
    """Return whether either applicable predeclared gate is unresolved."""
    return comparison["judge"]["classification"] == "unresolved" or (
        not comparison["comparison_penalty_active"]
        and comparison["raw_loss"]["classification"] == "unresolved"
    )


def _load_previous_summary(
    manifest: dict[str, Any], budget: int, path: Path
) -> dict[str, Any]:
    previous = json.loads(Path(path).read_text(encoding="utf-8"))
    if previous.get("search_manifest_sha256") != integration.manifest_digest(
        manifest
    ):
        raise RuntimeError("previous judge summary belongs to another search")
    previous_budget = int(previous.get("judge_budget", -1))
    budgets = manifest["judge_budgets"]
    if budget not in budgets or budgets.index(
        previous_budget
    ) + 1 != budgets.index(budget):
        raise RuntimeError(
            "judge summaries must use consecutive frozen budgets"
        )
    if previous.get("source_identity_sha256") != capture.digest(
        manifest["identity"]
    ):
        raise RuntimeError(
            "previous judge summary has another source identity"
        )
    return previous


def _logmeanexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    count = values.size if axis is None else values.shape[axis]
    return np.logaddexp.reduce(values, axis=axis) - np.log(count)


def _state_dimensions(state_id: str) -> dict[str, str]:
    trajectory, checkpoint = state_id.rsplit("_", 1)
    label, seed = trajectory.rsplit("_seed", 1)
    return {
        "state_id": state_id,
        "trajectory": trajectory,
        "label": label,
        "seed": seed,
        "checkpoint": checkpoint,
    }


def _contrast_specs(manifest: dict[str, Any]) -> list[dict[str, str]]:
    treatment_tags = {
        item["tag"] for item in manifest["treatments"] if item["tag"] != "S0"
    }
    specs = [
        {
            "contrast": f"{tag}_minus_S0",
            "kind": "primary_vs_s0",
            "arm_tag": tag,
            "reference_tag": "S0",
        }
        for tag in sorted(treatment_tags)
    ]
    if {"S1", "S2", "S3"} <= treatment_tags:
        specs.extend(
            [
                {
                    "contrast": "S2_minus_S1",
                    "kind": "component_diagnostic",
                    "arm_tag": "S2",
                    "reference_tag": "S1",
                },
                {
                    "contrast": "S3_minus_S2",
                    "kind": "component_diagnostic",
                    "arm_tag": "S3",
                    "reference_tag": "S2",
                },
                {
                    "contrast": "S3_minus_S1",
                    "kind": "component_diagnostic",
                    "arm_tag": "S3",
                    "reference_tag": "S1",
                },
            ]
        )
    direct_pairs = (
        ("original_arm", "mc1600_control"),
        ("finalist", "mc1600_control"),
    )
    for reference_tag, arm_tag in direct_pairs:
        if {reference_tag, arm_tag} <= treatment_tags:
            specs.append(
                {
                    "contrast": f"{arm_tag}_minus_{reference_tag}",
                    "kind": "direct_integration_control",
                    "arm_tag": arm_tag,
                    "reference_tag": reference_tag,
                }
            )
    return specs


def _comparison_cell(
    state_id: str, replicate: int, spec: dict[str, str]
) -> dict[str, Any]:
    return {
        "state_id": state_id,
        "replicate": int(replicate),
        **spec,
    }


def _comparison_tag(cell: dict[str, Any]) -> str:
    return f"{cell['state_id']}__{cell['contrast']}__r{cell['replicate']}"


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
    raw = {
        key: sum(
            not item["comparison_penalty_active"]
            and item["raw_loss"]["classification"] == key
            for item in comparisons
        )
        for key in ("material_loss", "no_material_loss", "unresolved")
    }
    raw["not_applicable_penalty_active"] = sum(
        item["comparison_penalty_active"] for item in comparisons
    )
    raw["failed_or_missing"] = score["failed_or_missing"]
    denominator = float(scheduled)
    return {
        "scheduled": int(scheduled),
        "observed": len(comparisons),
        "score_counts": score,
        "raw_loss_gate_counts": raw,
        "fractions_of_scheduled": {
            "harmful": score["harmful"] / denominator,
            "unresolved": score["unresolved"] / denominator,
            "failed_or_missing": score["failed_or_missing"] / denominator,
            "raw_material_loss": raw["material_loss"] / denominator,
            "raw_unresolved": raw["unresolved"] / denominator,
        },
    }


def grouped_judge_summary(
    manifest: dict[str, Any], comparisons: list[dict[str, Any]]
) -> dict[str, Any]:
    """Report every arm and state dimension against its full allocation."""
    specs = _contrast_specs(manifest)
    primary_specs = [item for item in specs if item["kind"] == "primary_vs_s0"]
    expected_ids = [state["state_id"] for state in manifest["states"]]
    expected_ids.extend(
        state["state_id"] for state in manifest["missing_states"]
    )
    dimensions = {
        state_id: _state_dimensions(state_id) for state_id in expected_ids
    }

    def grouped(name: str) -> list[dict[str, Any]]:
        rows = []
        for value in sorted({item[name] for item in dimensions.values()}):
            state_ids = {
                state_id
                for state_id, item in dimensions.items()
                if item[name] == value
            }
            for spec in specs:
                selected = [
                    item
                    for item in comparisons
                    if item["cell"]["state_id"] in state_ids
                    and item["cell"]["contrast"] == spec["contrast"]
                ]
                rows.append(
                    {
                        name: value,
                        **spec,
                        **_comparison_counts(
                            selected, len(state_ids) * SELECTION_REPLICATES
                        ),
                    }
                )
        return rows

    return {
        "by_arm": [
            {
                **spec,
                **_comparison_counts(
                    [
                        item
                        for item in comparisons
                        if item["cell"]["contrast"] == spec["contrast"]
                    ],
                    manifest["expected_state_count"] * SELECTION_REPLICATES,
                ),
            }
            for spec in primary_specs
        ],
        "by_component_contrast": [
            {
                **spec,
                **_comparison_counts(
                    [
                        item
                        for item in comparisons
                        if item["cell"]["contrast"] == spec["contrast"]
                    ],
                    manifest["expected_state_count"] * SELECTION_REPLICATES,
                ),
            }
            for spec in specs
            if spec["kind"] == "component_diagnostic"
        ],
        "by_direct_control": [
            {
                **spec,
                **_comparison_counts(
                    [
                        item
                        for item in comparisons
                        if item["cell"]["contrast"] == spec["contrast"]
                    ],
                    manifest["expected_state_count"] * SELECTION_REPLICATES,
                ),
            }
            for spec in specs
            if spec["kind"] == "direct_integration_control"
        ],
        "by_trajectory": grouped("trajectory"),
        "by_label": grouped("label"),
        "by_checkpoint": grouped("checkpoint"),
    }


def judge_summary(
    manifest: dict[str, Any],
    out: Path,
    budget: int,
    previous_summary: Path | None = None,
) -> dict[str, Any]:
    """Classify existing E3 judge arrays and retain resolved prior records."""
    runtime_identity(manifest)
    if budget == manifest["initial_judge_budget"]:
        if previous_summary is not None:
            raise ValueError("initial judge summary cannot have a predecessor")
        previous = None
        previous_sha256 = None
    else:
        if previous_summary is None:
            raise ValueError(
                "escalated judge summary requires its predecessor"
            )
        previous = _load_previous_summary(manifest, budget, previous_summary)
        previous_sha256 = integration.sha256_file(previous_summary)
    previous_by_tag = (
        {}
        if previous is None
        else {item["comparison_tag"]: item for item in previous["comparisons"]}
    )
    pending_tags = (
        None
        if previous is None
        else {
            tag
            for tag, item in previous_by_tag.items()
            if comparison_needs_escalation(item)
        }
    )
    comparisons = (
        []
        if previous is None
        else [
            copy.deepcopy(item)
            for tag, item in previous_by_tag.items()
            if tag not in pending_tags
        ]
    )
    bands_by_state = (
        {} if previous is None else copy.deepcopy(previous["bands_by_state"])
    )
    missing_comparisons = (
        []
        if previous is None
        else copy.deepcopy(previous.get("missing_comparisons", []))
    )
    failed_states = (
        []
        if previous is None
        else copy.deepcopy(previous.get("failed_states", []))
    )
    contrast_specs = _contrast_specs(manifest)
    for state in manifest["states"]:
        state_id = state["state_id"]
        state_cells = [
            _comparison_cell(state_id, replicate, spec)
            for replicate in range(SELECTION_REPLICATES)
            for spec in contrast_specs
        ]
        assessed_cells = [
            cell
            for cell in state_cells
            if pending_tags is None or _comparison_tag(cell) in pending_tags
        ]
        if not assessed_cells:
            continue
        cell_identity = _judge_cell(state_id, budget, previous_sha256)
        paths = _paths(out, judge_tag(state_id, budget), "judge")
        if not paths["complete"].is_file():
            failed_states.append(
                {"state_id": state_id, "reason": "judge_missing"}
            )
            missing_comparisons.extend(
                {"cell": cell, "reason": "judge_missing"}
                for cell in assessed_cells
            )
            continue
        terminal = _validate_terminal(manifest, paths, cell_identity)
        if terminal["status"] != "succeeded":
            failed_states.append(
                {"state_id": state_id, "reason": "judge_failed"}
            )
            missing_comparisons.extend(
                {"cell": cell, "reason": "judge_failed"}
                for cell in assessed_cells
            )
            continue
        report = json.loads(paths["json"].read_text(encoding="utf-8"))
        union = report["union"]
        with np.load(paths["npz"], allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        candidates = np.asarray(arrays["candidates"], dtype=np.float64)
        candidate_hash = hashlib.sha256(
            np.ascontiguousarray(candidates, dtype="<f8").tobytes()
        ).hexdigest()
        if candidate_hash != union["candidate_union_sha256"]:
            raise RuntimeError("judge candidate coordinates changed")
        row_ids = list(union["candidate_row_ids"])
        if len(row_ids) != len(candidates) or len(row_ids) != len(
            set(row_ids)
        ):
            raise RuntimeError(
                "judge candidate row identifiers are inconsistent"
            )
        positions = {row_id: index for index, row_id in enumerate(row_ids)}
        if state_id not in bands_by_state:
            pilot_gain = quadrature.baseline_gain_resolution(
                arrays["pilot_log_reduction"],
                arrays["pilot_reference_log_residual"],
            )
            band = quadrature.practical_band(
                float(_logmeanexp(arrays["pilot_reference_log_residual"])),
                float(_logmeanexp(arrays["pilot_log_residual"])),
                baseline_gain_resolved=pilot_gain["resolved"],
            )
            band["pilot_gain"] = pilot_gain
            band["fixed_band_row_id"] = union["fixed_band_row_id"]
            bands_by_state[state_id] = band
        band = bands_by_state[state_id]
        mean_log_scores = np.mean(arrays["full_score"], axis=0)
        pooled_scores = _logmeanexp(arrays["log_residual"], axis=0) + np.mean(
            arrays["penalty"], axis=0
        )
        valid = np.all(arrays["valid"], axis=0)
        mean_selectable = np.where(
            valid & np.isfinite(mean_log_scores), mean_log_scores, np.inf
        )
        pooled_selectable = np.where(
            valid & np.isfinite(pooled_scores), pooled_scores, np.inf
        )
        mean_best = (
            None
            if not np.any(np.isfinite(mean_selectable))
            else int(np.argmin(mean_selectable))
        )
        pooled_best = (
            None
            if not np.any(np.isfinite(pooled_selectable))
            else int(np.argmin(pooled_selectable))
        )
        for cell in assessed_cells:
            tag = _comparison_tag(cell)
            replicate = str(cell["replicate"])
            selected = (
                union["baseline_by_replicate"].get(replicate)
                if cell["arm_tag"] == "S0"
                else union["treatment_by_tag"]
                .get(cell["arm_tag"], {})
                .get(replicate)
            )
            baseline = (
                union["baseline_by_replicate"].get(replicate)
                if cell["reference_tag"] == "S0"
                else union["treatment_by_tag"]
                .get(cell["reference_tag"], {})
                .get(replicate)
            )
            if selected not in positions or baseline not in positions:
                missing_comparisons.append(
                    {"cell": cell, "reason": "paired_selection_missing"}
                )
                continue
            position = positions[selected]
            baseline_position = positions[baseline]
            arm_pooled = float(pooled_scores[position])
            baseline_pooled = float(pooled_scores[baseline_position])
            prior = previous_by_tag.get(tag)
            judged = quadrature.judge_paired(
                arrays["full_score"][:, position],
                arrays["full_score"][:, baseline_position],
                band["eps_F"],
                previous=None if prior is None else prior["judge"],
                replicate_seeds=union["comparison_seeds"][str(budget)],
                pooled_difference=arm_pooled - baseline_pooled,
            )
            raw_loss = quadrature.raw_loss_assessment(
                arrays["log_reduction"][:, position],
                arrays["log_reduction"][:, baseline_position],
                arrays["reference_log_residual"],
                previous=None if prior is None else prior["raw_loss"],
            )
            arm_penalty = arrays["penalty"][:, position]
            baseline_penalty = arrays["penalty"][:, baseline_position]
            penalty_active = bool(
                np.any(arm_penalty > 0) or np.any(baseline_penalty > 0)
            )
            comparisons.append(
                {
                    "comparison_tag": tag,
                    "judge_budget": budget,
                    "cell": cell,
                    "selected_row_id": selected,
                    "baseline_row_id": baseline,
                    "band": band,
                    "judge": judged,
                    "raw_loss": raw_loss,
                    "penalty": {
                        "arm": arm_penalty.tolist(),
                        "baseline": baseline_penalty.tolist(),
                        "arm_active": bool(np.any(arm_penalty > 0)),
                        "baseline_active": bool(np.any(baseline_penalty > 0)),
                    },
                    "comparison_penalty_active": penalty_active,
                    "pooled_log_reduction": {
                        "arm": float(
                            _logmeanexp(arrays["log_reduction"][:, position])
                        ),
                        "baseline": float(
                            _logmeanexp(
                                arrays["log_reduction"][:, baseline_position]
                            )
                        ),
                    },
                    "pooled_union_best_row_id": (
                        None if pooled_best is None else row_ids[pooled_best]
                    ),
                    "mean_log_union_best_row_id": (
                        None if mean_best is None else row_ids[mean_best]
                    ),
                    "union_best_disagreement": mean_best != pooled_best,
                    "pooled_score_regret_within_union": None
                    if pooled_best is None
                    or not np.isfinite(pooled_scores[position])
                    else float(
                        pooled_scores[position] - pooled_scores[pooled_best]
                    ),
                    "assessed_union": {
                        "union_id": union["union_id"],
                        "candidate_union_sha256": candidate_hash,
                        "judge_budget": budget,
                        "candidate_count": len(row_ids),
                    },
                }
            )
    scheduled = (
        manifest["expected_state_count"]
        * len(contrast_specs)
        * SELECTION_REPLICATES
    )
    if len({item["comparison_tag"] for item in comparisons}) != len(
        comparisons
    ):
        raise RuntimeError("judge summary contains duplicated comparisons")
    primary = [
        item for item in comparisons if item["cell"]["kind"] == "primary_vs_s0"
    ]
    component = [
        item
        for item in comparisons
        if item["cell"]["kind"] == "component_diagnostic"
    ]
    direct_control = [
        item
        for item in comparisons
        if item["cell"]["kind"] == "direct_integration_control"
    ]
    primary_scheduled = (
        manifest["expected_state_count"]
        * sum(spec["kind"] == "primary_vs_s0" for spec in contrast_specs)
        * SELECTION_REPLICATES
    )
    component_scheduled = scheduled - primary_scheduled
    direct_control_scheduled = (
        manifest["expected_state_count"]
        * sum(
            spec["kind"] == "direct_integration_control"
            for spec in contrast_specs
        )
        * SELECTION_REPLICATES
    )
    component_scheduled -= direct_control_scheduled
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "search_judge_summary",
        "search_manifest_sha256": integration.manifest_digest(manifest),
        "source_identity_sha256": capture.digest(manifest["identity"]),
        "judge_budget": int(budget),
        "previous_summary_sha256": previous_sha256,
        "bands_by_state": bands_by_state,
        "comparisons": comparisons,
        "counts": _comparison_counts(comparisons, scheduled),
        "primary_screen_counts": _comparison_counts(
            primary, primary_scheduled
        ),
        "component_diagnostic_counts": _comparison_counts(
            component, component_scheduled
        )
        if component_scheduled
        else None,
        "direct_control_counts": _comparison_counts(
            direct_control, direct_control_scheduled
        )
        if direct_control_scheduled
        else None,
        "grouped": grouped_judge_summary(manifest, comparisons),
        "failed_states": failed_states,
        "missing_comparisons": missing_comparisons,
    }


def _artifact_status(
    manifest: dict[str, Any], paths: dict[str, Path], cell: dict[str, Any]
) -> tuple[str, str | None]:
    if paths["complete"].exists():
        try:
            record = _validate_terminal(manifest, paths, cell)
            return record["status"], None
        except Exception as error:  # noqa: BLE001
            return "corrupt", str(error)
    if _partial(paths):
        return "partial", None
    return "missing", None


def inventory(manifest: dict[str, Any], out: Path) -> dict[str, Any]:
    """Account for succeeded, failed, partial, corrupt, and missing cells."""
    counts = {
        kind: {
            status: 0
            for status in (
                "succeeded",
                "failed",
                "partial",
                "corrupt",
                "missing",
            )
        }
        for kind in ("selection", "timing", "allocation", "judge")
    }
    rows = []
    for cell in manifest["cells"]:
        tag = cell_tag(cell)
        status, error = _artifact_status(
            manifest, _paths(out, tag, "selection"), cell
        )
        counts["selection"][status] += 1
        rows.append(
            {
                "kind": "selection",
                "tag": tag,
                "status": status,
                "error": error,
            }
        )
    for cell in manifest["timing_cells"]:
        tag = timing_tag(cell)
        paths = _paths(out, tag, "timing")
        paths["npz"] = None
        status, error = _artifact_status(manifest, paths, cell)
        counts["timing"][status] += 1
        rows.append(
            {"kind": "timing", "tag": tag, "status": status, "error": error}
        )
    for cell in manifest["allocation_cells"]:
        tag = allocation_tag(cell)
        paths = _paths(out, tag, "allocation")
        paths["npz"] = None
        status, error = _artifact_status(manifest, paths, cell)
        counts["allocation"][status] += 1
        rows.append(
            {
                "kind": "allocation",
                "tag": tag,
                "status": status,
                "error": error,
            }
        )
    for state in manifest["states"]:
        budget = manifest["initial_judge_budget"]
        cell = _judge_cell(state["state_id"], budget)
        tag = judge_tag(state["state_id"], budget)
        status, error = _artifact_status(
            manifest, _paths(out, tag, "judge"), cell
        )
        counts["judge"][status] += 1
        rows.append(
            {"kind": "judge", "tag": tag, "status": status, "error": error}
        )
    counts["selection"]["missing"] += max(
        0,
        manifest["scheduled_cell_count_including_missing_states"]
        - len(manifest["cells"]),
    )
    counts["timing"]["missing"] += max(
        0,
        manifest["scheduled_timing_count_including_missing_states"]
        - len(manifest["timing_cells"]),
    )
    counts["allocation"]["missing"] += max(
        0,
        manifest["scheduled_allocation_count_including_missing_states"]
        - len(manifest["allocation_cells"]),
    )
    expected_states = manifest.get(
        "expected_state_count",
        len(manifest["states"]) + len(manifest["missing_states"]),
    )
    counts["judge"]["missing"] += max(
        0, expected_states - len(manifest["states"])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": integration.manifest_digest(manifest),
        "stage": manifest["stage"],
        "counts": counts,
        "scheduled_selection_including_missing_states": manifest[
            "scheduled_cell_count_including_missing_states"
        ],
        "scheduled_timing_including_missing_states": manifest[
            "scheduled_timing_count_including_missing_states"
        ],
        "scheduled_allocation_including_missing_states": manifest[
            "scheduled_allocation_count_including_missing_states"
        ],
        "missing_states": manifest["missing_states"],
        "rows": rows,
    }


def _load_manifest(
    path: Path, *, require_ready: bool = True
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest, require_ready=require_ready)
    return manifest


def run_controller(manifest_path: Path, out: Path) -> dict[str, Any]:
    """Run every terminal cell in a fresh process and retain failures."""
    manifest = _load_manifest(manifest_path)
    runtime_identity(manifest)
    tasks = (
        [("cell", cell_tag(cell)) for cell in manifest["cells"]]
        + [
            ("timing-cell", timing_tag(cell))
            for cell in manifest["timing_cells"]
        ]
        + [
            ("allocation-cell", allocation_tag(cell))
            for cell in manifest["allocation_cells"]
        ]
    )
    for command, tag in tasks:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                command,
                "--manifest",
                str(manifest_path),
                "--out",
                str(out),
                "--tag",
                tag,
            ],
            check=True,
        )
    for state in manifest["states"]:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "judge-cell",
                "--manifest",
                str(manifest_path),
                "--out",
                str(out),
                "--state-id",
                state["state_id"],
                "--budget",
                str(manifest["initial_judge_budget"]),
            ],
            check=True,
        )
    summary = judge_summary(manifest, out, manifest["initial_judge_budget"])
    integration.write_json(
        out / f"judge_summary_b{manifest['initial_judge_budget']}.json",
        summary,
    )
    report = inventory(manifest, out)
    integration.write_json(out / "inventory.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-manifest")
    prepare.add_argument("--capture-manifest", type=Path, required=True)
    prepare.add_argument("--captures", type=Path, required=True)
    prepare.add_argument("--selection", type=Path, required=True)
    prepare.add_argument(
        "--stage",
        choices=tuple(sorted(STAGES)),
        required=True,
    )
    prepare.add_argument("--manifest", type=Path, required=True)
    for name in ("cell", "timing-cell", "allocation-cell"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--tag", required=True)
    judge = sub.add_parser("judge-cell")
    judge.add_argument("--manifest", type=Path, required=True)
    judge.add_argument("--out", type=Path, required=True)
    judge.add_argument("--state-id", required=True)
    judge.add_argument("--budget", type=int, required=True)
    judge.add_argument("--previous-summary", type=Path)
    judge_summary_parser = sub.add_parser("judge-summary")
    judge_summary_parser.add_argument("--manifest", type=Path, required=True)
    judge_summary_parser.add_argument("--out", type=Path, required=True)
    judge_summary_parser.add_argument("--budget", type=int, required=True)
    judge_summary_parser.add_argument("--previous-summary", type=Path)
    judge_summary_parser.add_argument("--summary", type=Path, required=True)
    for name in ("inventory", "run"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-manifest":
        manifest = prepare_manifest(
            args.capture_manifest, args.captures, args.selection, args.stage
        )
        integration.write_json(args.manifest, manifest)
        return 0
    manifest = _load_manifest(args.manifest)
    if args.command == "cell":
        matches = [
            cell for cell in manifest["cells"] if cell_tag(cell) == args.tag
        ]
        if len(matches) != 1:
            raise RuntimeError("unknown selection cell tag")
        run_cell(manifest, args.out, matches[0])
    elif args.command == "timing-cell":
        matches = [
            cell
            for cell in manifest["timing_cells"]
            if timing_tag(cell) == args.tag
        ]
        if len(matches) != 1:
            raise RuntimeError("unknown timing cell tag")
        run_timing_cell(manifest, args.out, matches[0])
    elif args.command == "allocation-cell":
        matches = [
            cell
            for cell in manifest["allocation_cells"]
            if allocation_tag(cell) == args.tag
        ]
        if len(matches) != 1:
            raise RuntimeError("unknown allocation cell tag")
        run_allocation_cell(manifest, args.out, matches[0])
    elif args.command == "judge-cell":
        run_judge_cell(
            manifest,
            args.out,
            args.state_id,
            args.budget,
            args.previous_summary,
        )
    elif args.command == "judge-summary":
        report = judge_summary(
            manifest, args.out, args.budget, args.previous_summary
        )
        integration.write_json(args.summary, report)
    elif args.command == "inventory":
        print(
            json.dumps(inventory(manifest, args.out), indent=2, sort_keys=True)
        )
    elif args.command == "run":
        run_controller(args.manifest, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
