"""Prepare, execute, and summarize the bounded E5 S0/S2 inference campaign.

Two manifest kinds describe the same full paired design (six targets, seeds
2000--2009).  The pilot manifest makes only the two-seed pilot (24 fits)
executable.  The continuation manifest binds a completed, reviewed pilot
manifest and makes exactly the remaining 96 fits (seeds 2002--2009)
executable; every numerical source, data archive and environment entry must
equal the pilot's, and only this runner's own hash may differ.  Cases run
sequentially in fresh processes, write complete golden traces, and become
resumable only after an atomic, hash-validated terminal record is present.
Continuation batches have explicit wall-time and fit-count limits, reuse
validated success terminals and never rerun a failed fit.  The combined
summary reports all ten paired seeds per target from both campaigns.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from filelock import FileLock, Timeout
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if os.environ.get("PYVBMC_GPYREG_SOURCE"):
    sys.path.insert(1, str(Path(os.environ["PYVBMC_GPYREG_SOURCE"]).resolve()))

import golden_trace
from benchmark_targets import find_config
from golden_replay import SEMANTIC_FINAL_KEYS, envelope
from profile_run import jsonable

LABELS = (
    "rosenbrock_D2_noise1",
    "rosenbrock_D2_noise3",
    "logreg_D5_noise3",
    "student_D8_noise3",
    "multisensory_s1_D6_noise1.3",
    "timing_D5_noise2.2",
)
ARMS = ("S0", "S2")
FULL_SEEDS = tuple(range(2000, 2010))
PILOT_SEEDS = (2000, 2001)
CONTINUATION_SEEDS = tuple(range(2002, 2010))
PILOT_KIND = "noisy_acq_e5_inference"
CONTINUATION_KIND = "noisy_acq_e5_continuation"
PILOT_CEILING_SECONDS = 10800
CONTINUATION_BATCH_CEILING_SECONDS = 21600
MINIMUM_FIT_TIMEOUT_SECONDS = 1200
# Immutable pilot records bound by a continuation manifest, relative to the
# directory holding the pilot manifest.
PARENT_RECORDS = (
    "launch_clearance.json",
    "summary.json",
    "completion.json",
    "independent_review.json",
)
SELECTION_SHA256 = (
    "98c5d773bd83c6b1954e7d5d5f79b83a47d196c6916925f335897350d66aab40"
)
S2_CONFIG = {
    "arm": "S2",
    "sieve_size": 1024,
    "shortlist_size": 8,
    "accurate_method": "mc",
    "accurate_budget": 1600,
    "finite_difference_step": 0.0001,
    "max_iterations": 50,
    "ftol": 1e-9,
    "gtol": 1e-5,
    "start_separation": 0.5,
    "gradient_relative_tolerance": 0.25,
    "max_candidate_rows": 1000,
}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MPLBACKEND": "Agg",
}
DEPENDENCIES = (
    "numpy",
    "scipy",
    "gpyreg",
    "cma",
    "psutil",
    "filelock",
    "threadpoolctl",
)
QUALITY = ("elbo_err", "gskl", "mmtv")
POSITIVE_PAIRED = ("wall_s", "search_s", "target_eval_s", "func_count")
USABLE_LIMITS = {"elbo_err": 1.0, "gskl": 1.0, "mmtv": 0.2}
REFERENCE_DEFAULT = ROOT / "dev/scripts/runs/golden/reference_990_20260913"
PROMOTION_DEFAULT = ROOT / "dev/golden/promotion_20260913/sha256_manifest.json"
# The runner's own entry in the source inventory (same key format as
# ``_source_inventory``); the one source a continuation may change.
RUNNER_SOURCE_KEY = str(Path(__file__).resolve().relative_to(ROOT))


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lf_sha256(path):
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cell_id(label, seed, arm):
    return f"{label}__seed{seed}__{arm}"


def _cells():
    cells = []
    order = 0
    for label in LABELS:
        for seed in FULL_SEEDS:
            arms = (
                ARMS
                if (seed - FULL_SEEDS[0]) % 2 == 0
                else tuple(reversed(ARMS))
            )
            for arm in arms:
                cells.append(
                    {
                        "cell_id": cell_id(label, seed, arm),
                        "label": label,
                        "seed": seed,
                        "arm": arm,
                        "order": order,
                        "phase": "pilot"
                        if seed in PILOT_SEEDS
                        else "continuation",
                    }
                )
                order += 1
    return cells


def _git(repo, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={Path(repo).resolve()}",
            "-C",
            str(Path(repo).resolve()),
            *args,
        ],
        text=True,
    ).strip()


def _source_inventory():
    package = ROOT / "pyvbmc"
    paths = [
        path
        for pattern in ("*.py", "*.ini")
        for path in package.rglob(pattern)
        if "__pycache__" not in path.parts
        and path.relative_to(package).parts[0] != "testing"
    ]
    paths += [
        HERE / "benchmark_targets.py",
        HERE / "golden_trace.py",
        HERE / "golden_replay.py",
        HERE / "profile_run.py",
        HERE / "noisy_acq_inference_policy.py",
        HERE / "noisy_acq_search.py",
        HERE / "noisy_acq_quadrature.py",
        Path(__file__).resolve(),
    ]
    data_root = HERE / "data"
    paths += sorted(data_root.glob("*.npz"))
    paths += sorted((data_root / "truths").glob("*.npz"))
    paths += sorted((data_root / "truths").glob("*.json"))
    paths.append(data_root / "provenance.json")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError(f"Required source files are missing: {missing}")
    return {
        str(p.resolve().relative_to(ROOT)): sha256(p)
        for p in sorted(set(paths), key=lambda p: str(p))
    }


def _gpyreg_identity():
    import gpyreg

    package = Path(gpyreg.__file__).resolve().parent
    requested = os.environ.get("PYVBMC_GPYREG_SOURCE")
    repo = Path(requested).resolve() if requested else package.parent
    if not (repo / ".git").exists():
        raise RuntimeError(
            "Set PYVBMC_GPYREG_SOURCE to the frozen gpyreg repository before preparation"
        )
    expected_package = repo / "gpyreg"
    if package != expected_package:
        raise RuntimeError(
            f"gpyreg imported from {package}, expected frozen source {expected_package}"
        )
    return {
        "import": str(Path(gpyreg.__file__).resolve()),
        "repository": str(repo),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "source_hashes": {
            str(path.relative_to(repo)).replace("\\", "/"): sha256(path)
            for path in sorted(package.rglob("*.py"))
        },
    }


def _import_identity():
    if any(os.environ.get(key) != value for key, value in THREAD_ENV.items()):
        raise RuntimeError(
            "Set all BLAS thread variables to 1 and MPLBACKEND=Agg before preparation/execution"
        )
    imports = {}
    for name in (
        "pyvbmc",
        "gpyreg",
        "golden_trace",
        "benchmark_targets",
        "profile_run",
        "noisy_acq_inference_policy",
        "noisy_acq_search",
        "noisy_acq_quadrature",
    ):
        module = importlib.import_module(name)
        imports[name] = str(Path(module.__file__).resolve())
    expected = {
        "pyvbmc": ROOT / "pyvbmc/__init__.py",
        "golden_trace": HERE / "golden_trace.py",
        "benchmark_targets": HERE / "benchmark_targets.py",
        "profile_run": HERE / "profile_run.py",
        "noisy_acq_inference_policy": HERE / "noisy_acq_inference_policy.py",
        "noisy_acq_search": HERE / "noisy_acq_search.py",
        "noisy_acq_quadrature": HERE / "noisy_acq_quadrature.py",
    }
    for name, path in expected.items():
        if Path(imports[name]) != path.resolve():
            raise RuntimeError(f"{name} imported outside the frozen workspace")
    versions = {
        name: importlib.metadata.version(name) for name in DEPENDENCIES
    }
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependencies": versions,
        "imports": imports,
        "numpy_build": jsonable(getattr(np.__config__, "CONFIG", None)),
        "gpyreg": _gpyreg_identity(),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV},
    }


def runtime_identity():
    """Return the exact execution identity without using older private seams."""
    return {"sources": _source_inventory(), "environment": _import_identity()}


def _target_contracts():
    contracts = {}
    for label in LABELS:
        problem = find_config(label).make(seed=PILOT_SEEDS[0])
        args, options = problem.vbmc_args()
        contracts[label] = {
            "dimension": int(np.asarray(args[1]).size),
            "noise_sd": float(problem.noise_sd),
            "standard_options": jsonable(options),
            "extra_options": {},
        }
    return contracts


def _reference_envelopes(folder, promotion_manifest):
    folder, promotion_manifest = Path(folder), Path(promotion_manifest)
    promoted = json.loads(promotion_manifest.read_text(encoding="utf-8"))
    result = {
        "path": str(folder.resolve()),
        "promotion_manifest": str(promotion_manifest.resolve()),
        "promotion_manifest_sha256": sha256(promotion_manifest),
        "labels": {},
    }
    for label in LABELS:
        files = sorted(folder.glob(f"{label}_seed*.json"))
        if not files:
            raise RuntimeError(f"No promoted reference sidecars for {label}")
        values = {metric: [] for metric in QUALITY}
        hashes = {}
        for path in files:
            expected = promoted["files"].get(path.stem, {}).get("json_sha256")
            actual = lf_sha256(path)
            if expected != actual:
                raise RuntimeError(
                    f"Promoted reference hash mismatch: {path.name}"
                )
            side = json.loads(path.read_text(encoding="utf-8"))
            hashes[path.name] = actual
            for metric in QUALITY:
                value = float(side["final"][metric])
                if np.isfinite(value):
                    values[metric].append(value)
        result["labels"][label] = {
            "sidecar_lf_sha256": hashes,
            "finite_counts": {k: len(v) for k, v in values.items()},
            "envelopes": {
                k: float(envelope(np.asarray(v))) for k, v in values.items()
            },
        }
    return result


def prepare_manifest(
    selection, reference=REFERENCE_DEFAULT, promotion=PROMOTION_DEFAULT
):
    selection = Path(selection).resolve()
    if sha256(selection) != SELECTION_SHA256:
        raise RuntimeError(
            "S2 selection artifact does not have the frozen raw SHA256"
        )
    selected = json.loads(selection.read_text(encoding="utf-8"))
    if (
        not selected.get("frozen")
        or selected.get("finalist_config") != S2_CONFIG
    ):
        raise RuntimeError(
            "S2 selection artifact does not contain the frozen SearchConfig"
        )
    cells = _cells()
    pilot_ids = [c["cell_id"] for c in cells if c["phase"] == "pilot"]
    semantic = {
        "schema_version": 1,
        "kind": "noisy_acq_e5_inference",
        "arms": {
            "S0": {"description": "unmodified production search"},
            "S2": S2_CONFIG,
        },
        "labels": list(LABELS),
        "full_seeds": list(FULL_SEEDS),
        "pilot_seeds": list(PILOT_SEEDS),
        "full_design_fit_count": 120,
        "pilot_fit_count": 24,
        "cells": cells,
        "executable_cell_ids": pilot_ids,
        "validation_replay": {
            "cell_id": cell_id("rosenbrock_D2_noise1", 2000, "S2"),
            "purpose": "operational reproducibility check before the remaining pilot fits",
            "excluded_from_all_scientific_and_timing_denominators": True,
        },
        "options": {},
        "target_contracts": _target_contracts(),
        "selection": {
            "path": str(selection),
            "raw_sha256": SELECTION_SHA256,
            "finalist_config": S2_CONFIG,
        },
        "identity": runtime_identity(),
        "reference": _reference_envelopes(reference, promotion),
        "measurement": {
            "optimizer_wall_seconds": "golden trace wall_s around VBMC.optimize",
            "search_seconds": "InferencePolicy total_search_seconds",
            "outer_wall_seconds": "case worker including metrics and artifact writes",
            "usable_limits_strict": USABLE_LIMITS,
        },
        "planning": {
            "pilot_minutes": [90, 120],
            "full_design_hours": [7, 10],
            "minimum_per_fit_timeout_seconds": 1200,
            "pilot_operational_ceiling_seconds": 10800,
        },
    }
    return {
        "created_utc": _utc_now(),
        "prepared_from_commit": _git(ROOT, "rev-parse", "HEAD"),
        "semantic_digest": _digest(semantic),
        "semantic": semantic,
        "reviewed_ready": {
            "approved": False,
            "reviewer": None,
            "reviewed_utc": None,
            "note": "Set only after independent implementation/allocation review.",
        },
    }


def verify_parent_campaign(parent, parent_campaign):
    """Require every pilot fit and the replay to be complete and valid.

    Returns the pilot terminal records keyed by cell id.
    """
    terminals = {}
    for requested in parent["semantic"]["executable_cell_ids"]:
        terminal = validate_terminal(parent, parent_campaign, requested)
        if terminal["status"] != "success":
            raise RuntimeError(f"Pilot fit did not succeed: {requested}")
        terminals[requested] = terminal
    validate_replay_report(parent, parent_campaign)
    return terminals


def _parent_records(parent_path, parent, parent_campaign):
    """Check the pilot's immutable records against each other and hash them."""
    root = Path(parent_path).parent
    records = {}
    for name in PARENT_RECORDS:
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"Pilot record is missing: {name}")
        records[name] = json.loads(path.read_text(encoding="utf-8"))
    digest = parent["semantic_digest"]
    raw = sha256(parent_path)
    if (
        records["launch_clearance.json"].get("manifest_semantic_digest")
        != (digest)
        or records["launch_clearance.json"].get("manifest_sha256") != raw
    ):
        raise RuntimeError("Pilot launch clearance binds another manifest")
    summary = records["summary.json"]
    if summary.get("manifest_semantic_digest") != digest or summary.get(
        "fit_status_counts"
    ) != {"success": 24, "failure": 0, "missing": 0}:
        raise RuntimeError("Pilot summary is not a complete 24-fit record")
    completion = records["completion.json"]
    if (
        completion.get("manifest_semantic_digest") != digest
        or completion.get("manifest_sha256") != raw
        or completion.get("summary_sha256") != sha256(root / "summary.json")
        or completion.get("fit_status_counts") != summary["fit_status_counts"]
    ):
        raise RuntimeError("Pilot completion record does not bind the summary")
    review = records["independent_review.json"]
    if (
        review.get("manifest_semantic_digest") != digest
        or review.get("verdict") != "pass"
        or review.get("remaining_findings") != []
        or review.get("operational_continuation_criteria_satisfied")
        is not True
    ):
        raise RuntimeError(
            "Pilot independent review did not clear operational continuation"
        )
    for name, expected in review.get("artifact_sha256", {}).items():
        path = root / name
        if path.is_file() and sha256(path) != expected:
            raise RuntimeError(f"Pilot artifact changed since review: {name}")
    return {name: sha256(root / name) for name in PARENT_RECORDS}, completion


def prepare_continuation_manifest(
    parent_path,
    parent_campaign,
    reference=REFERENCE_DEFAULT,
    promotion=PROMOTION_DEFAULT,
):
    """Allocate exactly the 96 remaining fits under a reviewed pilot.

    The pilot manifest, its complete campaign and its immutable records are
    bound by hash.  The current runtime identity may differ from the pilot's
    only in this runner's own source hash.
    """
    parent_path = Path(parent_path).resolve()
    parent_campaign = Path(parent_campaign).resolve()
    parent = validate_manifest(
        json.loads(parent_path.read_text(encoding="utf-8")),
        require_ready=True,
    )
    if _kind(parent) != PILOT_KIND:
        raise RuntimeError(
            "The continuation parent must be the pilot manifest"
        )
    verify_parent_campaign(parent, parent_campaign)
    record_hashes, completion = _parent_records(
        parent_path, parent, parent_campaign
    )
    parent_semantic = parent["semantic"]
    selection = Path(parent_semantic["selection"]["path"])
    if sha256(selection) != SELECTION_SHA256:
        raise RuntimeError(
            "S2 selection artifact does not have the frozen raw SHA256"
        )
    current = runtime_identity()
    provenance = identity_provenance(parent_semantic["identity"], current)
    contracts = _target_contracts()
    if contracts != parent_semantic["target_contracts"]:
        raise RuntimeError("Target contracts differ from the pilot")
    envelopes = _reference_envelopes(reference, promotion)
    if envelopes != parent_semantic["reference"]:
        raise RuntimeError(
            "Promoted reference envelopes differ from the pilot"
        )
    cells = _cells()
    continuation_ids = [
        c["cell_id"] for c in cells if c["phase"] == "continuation"
    ]
    semantic = {
        "schema_version": 1,
        "kind": CONTINUATION_KIND,
        "arms": parent_semantic["arms"],
        "labels": list(LABELS),
        "full_seeds": list(FULL_SEEDS),
        "pilot_seeds": list(PILOT_SEEDS),
        "continuation_seeds": list(CONTINUATION_SEEDS),
        "full_design_fit_count": 120,
        "pilot_fit_count": 24,
        "continuation_fit_count": 96,
        "cells": cells,
        "executable_cell_ids": continuation_ids,
        "validation_replay": None,
        "options": {},
        "target_contracts": contracts,
        "selection": parent_semantic["selection"],
        "identity": current,
        "identity_provenance": provenance,
        "parent": {
            "path": str(parent_path),
            "raw_sha256": sha256(parent_path),
            "semantic_digest": parent["semantic_digest"],
            "campaign_dir": str(parent_campaign),
            "record_sha256": record_hashes,
            "pilot_fit_count": 24,
            "pilot_pair_count": 12,
        },
        "reference": envelopes,
        "measurement": parent_semantic["measurement"],
        "planning": {
            "pilot_worker_seconds": completion.get("pilot_worker_seconds"),
            "remaining_96_fits_point_estimate_seconds": completion.get(
                "remaining_96_fits_point_estimate_seconds"
            ),
            "remaining_96_fits_planning_seconds": completion.get(
                "remaining_96_fits_planning_seconds"
            ),
            "cost_estimate_method": completion.get("cost_estimate_method"),
            "minimum_per_fit_timeout_seconds": MINIMUM_FIT_TIMEOUT_SECONDS,
            "batch_operational_ceiling_seconds": (
                CONTINUATION_BATCH_CEILING_SECONDS
            ),
            "scheduling": (
                "Sequential batches in manifest cell order (target-major, "
                "arm order alternating by seed block); each batch declares "
                "its wall-time limit and optional fit-count limit, stops "
                "launching when less than one per-fit timeout remains, "
                "reuses validated success terminals and never reruns a "
                "failed fit."
            ),
            "source_freeze": (
                "The runner's own source hash is part of this manifest's "
                "identity. Do not modify the runner once any continuation "
                "fit has run: its terminals would then belong to another "
                "manifest and could neither be reused nor rerun. A "
                "correction requires a declared successor allocation with "
                "explicit handling of the fits it invalidates."
            ),
            "interrupted_fit_handling": (
                "A fit interrupted before its terminal record leaves partial "
                "artifacts that the runner refuses on resume. Move them out "
                "of the campaign directory, record the interruption, and "
                "let the next batch rerun that cell from scratch; a partial "
                "fit is never completed evidence."
            ),
        },
    }
    return {
        "created_utc": _utc_now(),
        "prepared_from_commit": _git(ROOT, "rev-parse", "HEAD"),
        "semantic_digest": _digest(semantic),
        "semantic": semantic,
        "reviewed_ready": {
            "approved": False,
            "reviewer": None,
            "reviewed_utc": None,
            "note": "Set only after independent implementation/allocation review.",
        },
    }


def _kind(manifest):
    kind = manifest["semantic"].get("kind")
    if kind not in (PILOT_KIND, CONTINUATION_KIND):
        raise RuntimeError(f"Unknown E5 manifest kind: {kind!r}")
    return kind


def _seeds_for(manifest):
    return PILOT_SEEDS if _kind(manifest) == PILOT_KIND else CONTINUATION_SEEDS


def _phase_for(manifest):
    return "pilot" if _kind(manifest) == PILOT_KIND else "continuation"


def identity_provenance(parent_identity, current_identity):
    """Compare a pilot identity with the current one.

    Every source hash except this runner's own entry must be identical, the
    inventories must list the same files, and the environment (Python,
    dependency versions, import paths, NumPy build, gpyreg commit and source
    hashes, thread variables) must be unchanged.  Raise on any other change.
    """
    parent_sources = parent_identity.get("sources", {})
    current_sources = current_identity.get("sources", {})
    if set(parent_sources) != set(current_sources):
        raise RuntimeError(
            "Source inventory differs from the pilot: "
            f"{sorted(set(parent_sources) ^ set(current_sources))}"
        )
    changed = {
        key: {"parent": parent_sources[key], "current": current_sources[key]}
        for key in sorted(parent_sources)
        if parent_sources[key] != current_sources[key]
    }
    disallowed = sorted(set(changed) - {RUNNER_SOURCE_KEY})
    if disallowed:
        raise RuntimeError(
            f"Numerical/helper/data source changed since the pilot: {disallowed}"
        )
    if parent_identity.get("environment") != current_identity.get(
        "environment"
    ):
        raise RuntimeError("Execution environment differs from the pilot")
    return {
        "allowed_changed_sources": [RUNNER_SOURCE_KEY],
        "changed_sources": changed,
        "source_keys_identical": True,
        "unchanged_source_count": len(parent_sources) - len(changed),
        "environment_unchanged": True,
    }


def _validate_pilot_semantic(semantic, expected_cells):
    allowed = semantic.get("executable_cell_ids")
    expected_allowed = [
        c["cell_id"] for c in expected_cells if c["phase"] == "pilot"
    ]
    if allowed != expected_allowed or len(allowed) != 24:
        raise RuntimeError(
            "Executable allocation is not exactly the 24-fit pilot"
        )
    if semantic.get("validation_replay") != {
        "cell_id": cell_id("rosenbrock_D2_noise1", 2000, "S2"),
        "purpose": "operational reproducibility check before the remaining pilot fits",
        "excluded_from_all_scientific_and_timing_denominators": True,
    }:
        raise RuntimeError("Validation replay contract changed")
    if (
        semantic.get("planning", {}).get("pilot_operational_ceiling_seconds")
        != PILOT_CEILING_SECONDS
    ):
        raise RuntimeError("Pilot operational ceiling changed")


def _validate_continuation_semantic(semantic, expected_cells):
    if semantic.get("continuation_seeds") != list(CONTINUATION_SEEDS):
        raise RuntimeError(
            "Manifest continuation seeds differ from the approved design"
        )
    allowed = semantic.get("executable_cell_ids")
    expected_allowed = [
        c["cell_id"] for c in expected_cells if c["phase"] == "continuation"
    ]
    if allowed != expected_allowed or len(allowed) != 96:
        raise RuntimeError(
            "Executable allocation is not exactly the 96 remaining fits"
        )
    if semantic.get("continuation_fit_count") != 96:
        raise RuntimeError("Continuation fit count changed")
    if semantic.get("validation_replay") is not None:
        raise RuntimeError(
            "The continuation allocates no validation replay of its own"
        )
    parent = semantic.get("parent")
    required = {
        "path",
        "raw_sha256",
        "semantic_digest",
        "campaign_dir",
        "record_sha256",
    }
    if not isinstance(parent, dict) or not required <= set(parent):
        raise RuntimeError("Continuation manifest does not bind its parent")
    if set(parent["record_sha256"]) != set(PARENT_RECORDS):
        raise RuntimeError(
            "Continuation manifest does not bind every pilot record"
        )
    provenance = semantic.get("identity_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("allowed_changed_sources") != [RUNNER_SOURCE_KEY]
        or not set(provenance.get("changed_sources", {"?": None}))
        <= {RUNNER_SOURCE_KEY}
        or provenance.get("source_keys_identical") is not True
        or provenance.get("environment_unchanged") is not True
    ):
        raise RuntimeError(
            "Continuation provenance permits more than the runner to change"
        )
    planning = semantic.get("planning", {})
    if (
        planning.get("batch_operational_ceiling_seconds")
        != CONTINUATION_BATCH_CEILING_SECONDS
        or planning.get("minimum_per_fit_timeout_seconds")
        != MINIMUM_FIT_TIMEOUT_SECONDS
    ):
        raise RuntimeError("Continuation batch limits changed")


def _load_parent(manifest):
    """Return the bound pilot manifest after checking its bytes and digest."""
    parent = manifest["semantic"]["parent"]
    path = Path(parent["path"])
    if not path.is_file() or sha256(path) != parent["raw_sha256"]:
        raise RuntimeError("Bound pilot manifest changed or is missing")
    loaded = validate_manifest(json.loads(path.read_text(encoding="utf-8")))
    if (
        _kind(loaded) != PILOT_KIND
        or loaded["semantic_digest"] != parent["semantic_digest"]
        or not loaded.get("reviewed_ready", {}).get("approved")
    ):
        raise RuntimeError("Bound pilot manifest is not the reviewed pilot")
    for name, expected in parent["record_sha256"].items():
        record = path.parent / name
        if not record.is_file() or sha256(record) != expected:
            raise RuntimeError(
                f"Bound pilot record changed or missing: {name}"
            )
    return loaded


def validate_manifest(manifest, require_ready=False, check_runtime=False):
    semantic = manifest.get("semantic")
    if not isinstance(semantic, dict) or manifest.get(
        "semantic_digest"
    ) != _digest(semantic):
        raise RuntimeError("Manifest semantic digest is invalid")
    kind = _kind(manifest)
    if semantic.get("labels") != list(LABELS) or semantic.get(
        "full_seeds"
    ) != list(FULL_SEEDS):
        raise RuntimeError(
            "Manifest design differs from the approved E5 design"
        )
    if semantic.get("pilot_seeds") != list(PILOT_SEEDS):
        raise RuntimeError(
            "Manifest pilot seeds differ from the approved allocation"
        )
    if semantic.get("arms", {}).get("S2") != S2_CONFIG:
        raise RuntimeError("Manifest S2 configuration is not frozen")
    cells = semantic.get("cells", [])
    expected = _cells()
    if cells != expected:
        raise RuntimeError("Manifest cells or sequential order changed")
    if kind == PILOT_KIND:
        _validate_pilot_semantic(semantic, expected)
    else:
        _validate_continuation_semantic(semantic, expected)
    if semantic.get("options") != {}:
        raise RuntimeError(
            "E5 must use the standard target options without overrides"
        )
    if set(semantic.get("target_contracts", {})) != set(LABELS):
        raise RuntimeError("Manifest does not bind every target contract")
    if require_ready and not manifest.get("reviewed_ready", {}).get(
        "approved"
    ):
        raise RuntimeError("Manifest has not been marked reviewed and ready")
    if check_runtime:
        selection = Path(semantic["selection"]["path"])
        if not selection.is_file() or sha256(selection) != SELECTION_SHA256:
            raise RuntimeError(
                "Frozen S2 selection artifact changed or is missing"
            )
        current = runtime_identity()
        if semantic.get("identity") != current:
            raise RuntimeError(
                "Numerical/helper/data/dependency/environment identity changed"
            )
        if kind == CONTINUATION_KIND:
            parent = _load_parent(manifest)
            provenance = identity_provenance(
                parent["semantic"]["identity"], current
            )
            if provenance != semantic.get("identity_provenance"):
                raise RuntimeError(
                    "Recorded pilot/continuation provenance no longer holds"
                )
    return manifest


def _cell(manifest, requested):
    matches = [
        c for c in manifest["semantic"]["cells"] if c["cell_id"] == requested
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown cell id: {requested}")
    if requested not in manifest["semantic"]["executable_cell_ids"]:
        raise RuntimeError(
            f"Cell is outside the executable allocation: {requested}"
        )
    return matches[0]


def _paths(out, cell):
    out = Path(out)
    tag = golden_trace._tag(cell["label"], cell["seed"])
    arm_dir = out / "arms" / cell["arm"]
    return {
        "arm_dir": arm_dir,
        "npz": arm_dir / f"{tag}.npz",
        "json": arm_dir / f"{tag}.json",
        "error": arm_dir / f"{tag}.error.txt",
        "selection": out / "records" / f"{cell['cell_id']}.selection.json",
        "terminal": out / "records" / f"{cell['cell_id']}.terminal.json",
        "log": out / "logs" / f"{cell['cell_id']}.log",
    }


def _relative(path, out):
    return str(Path(path).resolve().relative_to(Path(out).resolve())).replace(
        "\\", "/"
    )


def _terminal_record(manifest, cell, out, status, artifacts, **details):
    hashed = {
        _relative(path, out): sha256(path)
        for path in artifacts
        if Path(path).is_file()
    }
    record = {
        "schema_version": 1,
        "manifest_semantic_digest": manifest["semantic_digest"],
        "cell": cell,
        "status": status,
        "artifacts": hashed,
        **details,
    }
    record["record_digest"] = _digest(record)
    return record


def validate_terminal(manifest, out, requested):
    cell = _cell(manifest, requested)
    paths = _paths(out, cell)
    if not paths["terminal"].is_file():
        raise RuntimeError(f"{requested}: no terminal record")
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    recorded_digest = terminal.pop("record_digest", None)
    if recorded_digest != _digest(terminal):
        raise RuntimeError(f"{requested}: terminal record was changed")
    terminal["record_digest"] = recorded_digest
    if terminal.get("manifest_semantic_digest") != manifest["semantic_digest"]:
        raise RuntimeError(
            f"{requested}: terminal belongs to another manifest"
        )
    if terminal.get("cell") != cell or terminal.get("status") not in (
        "success",
        "failure",
    ):
        raise RuntimeError(f"{requested}: invalid terminal semantics")
    for relative, expected in terminal.get("artifacts", {}).items():
        path = Path(out) / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(
                f"{requested}: changed or missing artifact {relative}"
            )
    required = {paths["selection"]}
    if terminal["status"] == "success":
        required |= {paths["npz"], paths["json"]}
    actual = {Path(out) / p for p in terminal.get("artifacts", {})}
    if not required <= actual:
        raise RuntimeError(f"{requested}: incomplete terminal artifact set")
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    if selection.get("cell") != cell:
        raise RuntimeError(f"{requested}: selection sidecar has wrong cell")
    if terminal["status"] == "success":
        records = selection.get("records")
        summary = selection.get("summary")
        if (
            not isinstance(records, list)
            or not records
            or not isinstance(summary, dict)
        ):
            raise RuntimeError(
                f"{requested}: incomplete selection instrumentation"
            )
        expected_indices = list(range(len(records)))
        if [
            record.get("selection_index") for record in records
        ] != expected_indices:
            raise RuntimeError(f"{requested}: nonsequential selection records")
        if any(
            record.get("status") != "complete"
            or record.get("arm") != cell["arm"]
            or record.get("label") != cell["label"]
            or (
                cell["arm"] == "S0" and record.get("accurate_seed") is not None
            )
            or (cell["arm"] == "S2" and record.get("accurate_seed") is None)
            for record in records
        ):
            raise RuntimeError(
                f"{requested}: invalid or failed selection record"
            )
        expected_summary = {
            "arm": cell["arm"],
            "label": cell["label"],
            "selection_count": len(records),
            "complete_selection_count": len(records),
            "failed_selection_count": 0,
        }
        if any(
            summary.get(key) != value
            for key, value in expected_summary.items()
        ):
            raise RuntimeError(f"{requested}: inconsistent selection summary")
        measured = sum(float(record["search_seconds"]) for record in records)
        if not np.isclose(
            float(summary.get("total_search_seconds", -1)),
            measured,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError(f"{requested}: inconsistent search timing")
        side = json.loads(paths["json"].read_text(encoding="utf-8"))
        if (
            side.get("label") != cell["label"]
            or side.get("seed") != cell["seed"]
        ):
            raise RuntimeError(f"{requested}: golden sidecar has wrong case")
        requested_options = {
            **manifest["semantic"]["target_contracts"][cell["label"]][
                "standard_options"
            ],
            "display": "off",
            "plot": False,
            "print_iteration_header": False,
            "performance_calibration": "off",
        }
        if side.get("requested_options") != requested_options:
            raise RuntimeError(
                f"{requested}: requested target options changed"
            )
        effective = side.get("effective_options", {})
        if any(
            effective.get(key) != value
            for key, value in requested_options.items()
        ):
            raise RuntimeError(
                f"{requested}: effective target options changed"
            )
        with np.load(paths["npz"], allow_pickle=False) as trace:
            for key in trace.files:
                _ = trace[key]
    return terminal


def _existing_case_artifacts(paths):
    return [
        path
        for name, path in paths.items()
        if name not in ("arm_dir", "terminal") and path.exists()
    ]


def _write_failure(
    manifest, cell, out, error, started, paths, extra_artifacts=()
):
    if not paths["selection"].exists():
        atomic_json(
            paths["selection"],
            {"cell": cell, "records": [], "summary": None, "error": error},
        )
    artifacts = [paths["selection"], *extra_artifacts]
    for key in ("npz", "json", "error"):
        if paths[key].is_file():
            artifacts.append(paths[key])
    terminal = _terminal_record(
        manifest,
        cell,
        out,
        "failure",
        artifacts,
        started_utc=started,
        finished_utc=_utc_now(),
        outer_wall_s=max(
            0.0, time.time() - datetime.fromisoformat(started).timestamp()
        ),
        error=error,
    )
    atomic_json(paths["terminal"], terminal)
    return terminal


def worker(manifest_path, out, requested):
    manifest = validate_manifest(
        json.loads(Path(manifest_path).read_text(encoding="utf-8")),
        require_ready=True,
        check_runtime=True,
    )
    cell = _cell(manifest, requested)
    paths = _paths(out, cell)
    paths["arm_dir"].mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    began = time.perf_counter()
    policy = None
    try:
        from noisy_acq_inference_policy import InferencePolicy

        with InferencePolicy(
            cell["arm"], cell["seed"], cell["label"]
        ) as policy:
            result = golden_trace.run_task(
                cell["label"],
                cell["seed"],
                manifest["semantic"]["options"],
                paths["arm_dir"],
            )
        selection = {
            "cell": cell,
            "records": jsonable(policy.records),
            "summary": jsonable(policy.summary()),
        }
        atomic_json(paths["selection"], selection)
        if not result.get("ok"):
            raise RuntimeError(
                "golden_trace.run_task reported an execution failure"
            )
        if manifest["semantic"]["identity"] != runtime_identity():
            raise RuntimeError("Execution identity changed during the fit")
        artifacts = [paths["npz"], paths["json"], paths["selection"]]
        terminal = _terminal_record(
            manifest,
            cell,
            out,
            "success",
            artifacts,
            started_utc=started,
            finished_utc=_utc_now(),
            outer_wall_s=time.perf_counter() - began,
        )
        atomic_json(paths["terminal"], terminal)
        validate_terminal(manifest, out, requested)
        return 0
    except Exception as exc:
        if policy is not None and not paths["selection"].exists():
            atomic_json(
                paths["selection"],
                {
                    "cell": cell,
                    "records": jsonable(policy.records),
                    "summary": jsonable(policy.summary()),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        _write_failure(manifest, cell, out, error, started, paths)
        return 1


def _run_cell_locked(manifest_path, out, requested, timeout):
    if timeout < 1200:
        raise RuntimeError("Per-fit timeout must be at least 1200 seconds")
    manifest = validate_manifest(
        json.loads(Path(manifest_path).read_text(encoding="utf-8")),
        require_ready=True,
    )
    cell = _cell(manifest, requested)
    paths = _paths(out, cell)
    try:
        lock_path = Path(out) / "locks" / f"{requested}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock_path), timeout=0):
            if paths["terminal"].exists():
                return (
                    0
                    if validate_terminal(manifest, out, requested)["status"]
                    == "success"
                    else 1
                )
            existing = _existing_case_artifacts(paths)
            if existing:
                raise RuntimeError(
                    f"{requested}: partial prior attempt exists: {existing}"
                )
            paths["log"].parent.mkdir(parents=True, exist_ok=True)
            started = _utc_now()
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "_worker",
                "--manifest",
                str(Path(manifest_path).resolve()),
                "--out",
                str(Path(out).resolve()),
                "--cell-id",
                requested,
            ]
            env = dict(os.environ)
            env.update(THREAD_ENV)
            try:
                with paths["log"].open("w", encoding="utf-8") as stream:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                    )
            except subprocess.TimeoutExpired:
                _write_failure(
                    manifest,
                    cell,
                    out,
                    f"TimeoutExpired: exceeded {timeout} seconds",
                    started,
                    paths,
                    (paths["log"],),
                )
                return 1
            if not paths["terminal"].exists():
                _write_failure(
                    manifest,
                    cell,
                    out,
                    f"WorkerExit: return code {completed.returncode} without terminal record",
                    started,
                    paths,
                    (paths["log"],),
                )
            terminal = validate_terminal(manifest, out, requested)
            return 0 if terminal["status"] == "success" else 1
    except Timeout as exc:
        raise RuntimeError(
            f"{requested}: another process owns the cell lock"
        ) from exc


def run_cell(manifest_path, out, requested, timeout):
    campaign_lock = Path(out) / "execution.lock"
    campaign_lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(campaign_lock), timeout=0):
            return _run_cell_locked(manifest_path, out, requested, timeout)
    except Timeout as exc:
        raise RuntimeError(
            "Another E5 fit or pilot supervisor is active"
        ) from exc


def run_pilot(manifest_path, out, timeout, campaign_max_seconds=10800):
    manifest = validate_manifest(
        json.loads(Path(manifest_path).read_text(encoding="utf-8")),
        require_ready=True,
    )
    if _kind(manifest) != PILOT_KIND:
        raise RuntimeError("run-pilot accepts only the pilot manifest")
    ceiling = manifest["semantic"]["planning"][
        "pilot_operational_ceiling_seconds"
    ]
    if campaign_max_seconds > ceiling:
        raise RuntimeError(
            "Requested pilot wall time exceeds the frozen operational ceiling"
        )
    campaign_lock = Path(out) / "execution.lock"
    campaign_lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(campaign_lock), timeout=0):
            statuses = []
            started = time.monotonic()
            allowed = manifest["semantic"]["executable_cell_ids"]
            validation_id = manifest["semantic"]["validation_replay"][
                "cell_id"
            ]
            validation_index = allowed.index(validation_id)
            for index, requested in enumerate(allowed):
                if index > validation_index:
                    try:
                        validate_replay_report(manifest, out)
                    except RuntimeError as exc:
                        statuses.append(
                            {
                                "cell_id": requested,
                                "returncode": None,
                                "status": "not_launched_validation_replay_required",
                            }
                        )
                        atomic_json(
                            Path(out) / "status.json",
                            {
                                "updated_utc": _utc_now(),
                                "stop_reason": "validation_replay_required",
                                "detail": str(exc),
                                "cells": statuses,
                            },
                        )
                        return 1
                remaining = campaign_max_seconds - (time.monotonic() - started)
                if remaining < timeout:
                    statuses.append(
                        {
                            "cell_id": requested,
                            "returncode": None,
                            "status": "not_launched_campaign_deadline",
                        }
                    )
                    atomic_json(
                        Path(out) / "status.json",
                        {
                            "updated_utc": _utc_now(),
                            "stop_reason": "insufficient_time_for_next_fit",
                            "remaining_seconds": remaining,
                            "cells": statuses,
                        },
                    )
                    return 1
                code = _run_cell_locked(manifest_path, out, requested, timeout)
                statuses.append({"cell_id": requested, "returncode": code})
                atomic_json(
                    Path(out) / "status.json",
                    {"updated_utc": _utc_now(), "cells": statuses},
                )
                if code:
                    break
            return int(any(row["returncode"] for row in statuses))
    except Timeout as exc:
        raise RuntimeError(
            "Another E5 fit or pilot supervisor is active"
        ) from exc


def _batch_state(manifest, out):
    """Classify every executable cell from its terminal record."""
    state = {}
    for requested in manifest["semantic"]["executable_cell_ids"]:
        cell = _cell(manifest, requested)
        if _paths(out, cell)["terminal"].is_file():
            state[requested] = validate_terminal(manifest, out, requested)[
                "status"
            ]
        else:
            state[requested] = "missing"
    return state


def run_continuation(
    manifest_path,
    out,
    timeout,
    max_wall_seconds,
    max_fits=None,
    on_failure="stop",
):
    """Run one bounded, resumable batch of the 96 continuation fits.

    Cells are visited in manifest order.  A validated success terminal is
    reused without launching; a failed terminal is never rerun and, unless
    ``on_failure="continue"``, ends the batch so the failure can be
    investigated first.  No fit is launched when fewer than ``timeout``
    seconds of the batch allowance remain or when ``max_fits`` launches have
    occurred.  Return 0 when every continuation cell has a success terminal,
    1 when a failure was encountered or skipped, and 2 when the batch ended
    cleanly with cells still missing.
    """
    manifest = validate_manifest(
        json.loads(Path(manifest_path).read_text(encoding="utf-8")),
        require_ready=True,
    )
    if _kind(manifest) != CONTINUATION_KIND:
        raise RuntimeError(
            "run-continuation accepts only the continuation manifest"
        )
    if on_failure not in ("stop", "continue"):
        raise RuntimeError("on_failure must be 'stop' or 'continue'")
    planning = manifest["semantic"]["planning"]
    if timeout < planning["minimum_per_fit_timeout_seconds"]:
        raise RuntimeError("Per-fit timeout must be at least 1200 seconds")
    ceiling = planning["batch_operational_ceiling_seconds"]
    if not timeout <= max_wall_seconds <= ceiling:
        raise RuntimeError(
            "Batch wall time must lie between one per-fit timeout and the "
            f"frozen ceiling of {ceiling} seconds"
        )
    if max_fits is not None and max_fits < 1:
        raise RuntimeError("max_fits must be a positive integer")
    parent = _load_parent(manifest)
    verify_parent_campaign(
        parent, manifest["semantic"]["parent"]["campaign_dir"]
    )
    campaign_lock = Path(out) / "execution.lock"
    campaign_lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(campaign_lock), timeout=0):
            started_utc = _utc_now()
            started = time.monotonic()
            batch_path = (
                Path(out)
                / "batches"
                / f"batch_{started_utc.replace(':', '').replace('-', '')}.json"
            )
            record = {
                "schema_version": 1,
                "manifest_semantic_digest": manifest["semantic_digest"],
                "started_utc": started_utc,
                "timeout_seconds": timeout,
                "max_wall_seconds": max_wall_seconds,
                "max_fits": max_fits,
                "on_failure": on_failure,
                "cells": [],
                "stop_reason": None,
            }
            # One validation pass over the terminals; launches update it.
            state = _batch_state(manifest, out)
            record["state_before"] = _state_counts(state)
            launched = failures = 0

            def checkpoint(stop_reason=None):
                record["stop_reason"] = stop_reason
                record["updated_utc"] = _utc_now()
                record["batch_wall_s"] = time.monotonic() - started
                record["launched"] = launched
                record["failures"] = failures
                atomic_json(batch_path, record)
                atomic_json(
                    Path(out) / "status.json",
                    {
                        "updated_utc": record["updated_utc"],
                        "batch_record": _relative(batch_path, out),
                        "stop_reason": stop_reason,
                        "cells": record["cells"],
                    },
                )

            def finish(stop_reason):
                record["state_after"] = _state_counts(state)
                record["complete"] = record["state_after"]["success"] == len(
                    state
                )
                record["finished_utc"] = _utc_now()
                checkpoint(stop_reason)

            checkpoint()
            stop_reason = None
            try:
                for requested in manifest["semantic"]["executable_cell_ids"]:
                    status = state[requested]
                    if status == "success":
                        record["cells"].append(
                            {"cell_id": requested, "action": "reused"}
                        )
                        continue
                    if status == "failure":
                        failures += 1
                        if on_failure == "stop":
                            record["cells"].append(
                                {
                                    "cell_id": requested,
                                    "action": "stopped_at_failed",
                                }
                            )
                            stop_reason = "prior_failed_terminal"
                            break
                        record["cells"].append(
                            {"cell_id": requested, "action": "skipped_failed"}
                        )
                        continue
                    if max_fits is not None and launched >= max_fits:
                        record["cells"].append(
                            {"cell_id": requested, "action": "not_launched"}
                        )
                        stop_reason = "fit_limit_reached"
                        break
                    remaining = max_wall_seconds - (time.monotonic() - started)
                    if remaining < timeout:
                        record["cells"].append(
                            {
                                "cell_id": requested,
                                "action": "not_launched",
                                "remaining_seconds": remaining,
                            }
                        )
                        stop_reason = "insufficient_time_for_next_fit"
                        break
                    launch_started = time.monotonic()
                    code = _run_cell_locked(
                        manifest_path, out, requested, timeout
                    )
                    # ``_run_cell_locked`` returns 0 only after validating a
                    # success terminal and 1 only after a failure terminal.
                    state[requested] = "success" if code == 0 else "failure"
                    launched += 1
                    record["cells"].append(
                        {
                            "cell_id": requested,
                            "action": "launched",
                            "returncode": code,
                            "outer_seconds": time.monotonic() - launch_started,
                        }
                    )
                    checkpoint()
                    if code:
                        failures += 1
                        if on_failure == "stop":
                            stop_reason = "fit_failed"
                            break
            except BaseException as exc:
                # A refused partial attempt, a tampered terminal or an
                # interrupt leaves an explicit aborted record, never an
                # open-ended one; the cell's own artifacts are untouched.
                record["abort_error"] = f"{type(exc).__name__}: {exc}"
                finish("aborted")
                raise
            if stop_reason is None:
                stop_reason = (
                    "all_cells_complete"
                    if _state_counts(state)["success"] == len(state)
                    else "allocation_visited"
                )
            finish(stop_reason)
            if record["complete"]:
                return 0
            return 1 if failures else 2
    except Timeout as exc:
        raise RuntimeError(
            "Another E5 fit or continuation batch is active"
        ) from exc


def _state_counts(state):
    return {
        status: sum(v == status for v in state.values())
        for status in ("success", "failure", "missing")
    }


def _without_selection_timing(records):
    return [
        {
            key: value
            for key, value in record.items()
            if key != "search_seconds"
        }
        for record in records
    ]


def _replay_differences(manifest, out):
    requested = manifest["semantic"]["validation_replay"]["cell_id"]
    cell = _cell(manifest, requested)
    auxiliary = Path(out) / "validation_replay"
    primary_paths = _paths(out, cell)
    replay_paths = _paths(auxiliary, cell)
    differences = []
    with np.load(primary_paths["npz"], allow_pickle=False) as primary, np.load(
        replay_paths["npz"], allow_pickle=False
    ) as replay:
        keys = sorted((set(primary.files) | set(replay.files)) - {"timer"})
        for key in keys:
            if (
                key not in primary.files
                or key not in replay.files
                or not np.array_equal(
                    primary[key], replay[key], equal_nan=True
                )
            ):
                differences.append(f"npz:{key}")
    primary_side = json.loads(
        primary_paths["json"].read_text(encoding="utf-8")
    )
    replay_side = json.loads(replay_paths["json"].read_text(encoding="utf-8"))
    for key in SEMANTIC_FINAL_KEYS:
        if primary_side["final"].get(key) != replay_side["final"].get(key):
            differences.append(f"final:{key}")
    primary_selection = json.loads(
        primary_paths["selection"].read_text(encoding="utf-8")
    )
    replay_selection = json.loads(
        replay_paths["selection"].read_text(encoding="utf-8")
    )
    if _without_selection_timing(
        primary_selection["records"]
    ) != _without_selection_timing(replay_selection["records"]):
        differences.append("selection_records_except_timing")
    return differences


def validate_replay_report(manifest, out):
    requested = manifest["semantic"]["validation_replay"]["cell_id"]
    auxiliary = Path(out) / "validation_replay"
    report_path = auxiliary / "validation_replay.json"
    if not report_path.is_file():
        raise RuntimeError("Validation replay report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    record_digest = report.pop("record_digest", None)
    if record_digest != _digest(report):
        raise RuntimeError("Validation replay report was changed")
    report["record_digest"] = record_digest
    primary = validate_terminal(manifest, out, requested)
    replay = validate_terminal(manifest, auxiliary, requested)
    differences = _replay_differences(manifest, out)
    expected = {
        "manifest_semantic_digest": manifest["semantic_digest"],
        "cell_id": requested,
        "excluded_from_all_scientific_and_timing_denominators": True,
        "exact_non_timing_match": True,
        "differences": [],
        "primary_terminal_record_digest": primary["record_digest"],
        "replay_terminal_record_digest": replay["record_digest"],
    }
    if differences or any(
        report.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError(
            "Validation replay does not exactly match the primary fit"
        )
    return report


def run_validation_replay(manifest_path, out, timeout):
    """Repeat the frozen S2 case and require exact non-timing reproducibility."""
    manifest = validate_manifest(
        json.loads(Path(manifest_path).read_text(encoding="utf-8")),
        require_ready=True,
    )
    requested = manifest["semantic"]["validation_replay"]["cell_id"]
    auxiliary = Path(out) / "validation_replay"
    report_path = auxiliary / "validation_replay.json"
    if report_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite validation replay: {report_path}"
        )
    campaign_lock = Path(out) / "execution.lock"
    campaign_lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(campaign_lock), timeout=0):
            primary = validate_terminal(manifest, out, requested)
            if primary["status"] != "success":
                raise RuntimeError("Primary validation case did not succeed")
            code = _run_cell_locked(
                manifest_path, auxiliary, requested, timeout
            )
            if code:
                return code
            replay = validate_terminal(manifest, auxiliary, requested)
            differences = _replay_differences(manifest, out)
            report = {
                "schema_version": 1,
                "created_utc": _utc_now(),
                "manifest_semantic_digest": manifest["semantic_digest"],
                "cell_id": requested,
                "excluded_from_all_scientific_and_timing_denominators": True,
                "exact_non_timing_match": not differences,
                "differences": differences,
                "primary_terminal_record_digest": primary["record_digest"],
                "replay_terminal_record_digest": replay["record_digest"],
            }
            report["record_digest"] = _digest(report)
            atomic_json(report_path, report)
            return int(bool(differences))
    except Timeout as exc:
        raise RuntimeError(
            "Another E5 fit or pilot supervisor is active"
        ) from exc


def _finite(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def usable(final):
    return bool(
        all(_finite(final.get(k)) for k in QUALITY)
        and all(float(final[k]) < limit for k, limit in USABLE_LIMITS.items())
    )


def _load_outcome(manifest, out, cell):
    paths = _paths(out, cell)
    if not paths["terminal"].exists():
        return {"status": "missing", "cell": cell}
    terminal = validate_terminal(manifest, out, cell["cell_id"])
    result = {"status": terminal["status"], "cell": cell, "terminal": terminal}
    if terminal["status"] == "failure":
        result["error"] = terminal.get("error")
        if paths["json"].is_file():
            try:
                partial = json.loads(paths["json"].read_text(encoding="utf-8"))
                result["partial_final"] = partial.get("final")
            except (OSError, ValueError):
                result["partial_final"] = None
        return result
    side = json.loads(paths["json"].read_text(encoding="utf-8"))
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    final = side["final"]
    result.update(
        final=final,
        selection_summary=selection["summary"],
        search_s=float(selection["summary"]["total_search_seconds"]),
        outer_wall_s=float(terminal["outer_wall_s"]),
        converged=bool(final["success_flag"]),
        usable=usable(final),
    )
    reference = manifest["semantic"]["reference"]["labels"][cell["label"]][
        "envelopes"
    ]
    result["reference_exceedances"] = [
        metric
        for metric in QUALITY
        if _finite(final.get(metric))
        and float(final[metric]) > reference[metric]
    ]
    return result


def _log_ratio_interval(rows, metric):
    ratios = []
    for pair in rows:
        if pair["status"] != "complete":
            continue
        a, b = pair["S0"], pair["S2"]
        av = a["final"].get(metric) if metric != "search_s" else a.get(metric)
        bv = b["final"].get(metric) if metric != "search_s" else b.get(metric)
        if _finite(av) and _finite(bv) and float(av) > 0 and float(bv) > 0:
            ratios.append(math.log(float(bv) / float(av)))
    n = len(ratios)
    result = {
        "n_available_pairs": n,
        "n_scheduled_pairs": len(rows),
        "df": n - 1 if n else None,
    }
    if not n:
        return result | {"geometric_mean_ratio": None, "ci95": None}
    mean = float(np.mean(ratios))
    if n < 2:
        ci = None
    else:
        half = float(stats.t.ppf(0.975, n - 1) * stats.sem(ratios))
        ci = [math.exp(mean - half), math.exp(mean + half)]
    return result | {"geometric_mean_ratio": math.exp(mean), "ci95": ci}


def _accuracy(rows, metric):
    values = []
    for pair in rows:
        if pair["status"] == "complete":
            a, b = pair["S0"]["final"].get(metric), pair["S2"]["final"].get(
                metric
            )
            if _finite(a) and _finite(b):
                values.append(float(b) - float(a))
    return {
        "n_available_pairs": len(values),
        "n_scheduled_pairs": len(rows),
        "differences_S2_minus_S0": values,
        "median": float(np.median(values)) if values else None,
        "range": [min(values), max(values)] if values else None,
        "worse_count": sum(v > 0 for v in values),
    }


def _transitions(rows, key):
    counts = {
        "false_false": 0,
        "false_true": 0,
        "true_false": 0,
        "true_true": 0,
    }
    excluded = {
        "missing": 0,
        "failure": 0,
        "invalid_initial_design": 0,
        "nonfinite": 0,
    }
    for pair in rows:
        if pair["status"] != "complete":
            excluded[pair["status"]] += 1
            continue
        old, new = bool(pair["S0"][key]), bool(pair["S2"][key])
        counts[f"{str(old).lower()}_{str(new).lower()}"] += 1
    return {
        "transitions_S0_to_S2": counts,
        "excluded": excluded,
        "scheduled": len(rows),
    }


PAIR_STATUSES = (
    "complete",
    "failure",
    "missing",
    "invalid_initial_design",
    "nonfinite",
)


def _executable_cells(manifest):
    executable = set(manifest["semantic"]["executable_cell_ids"])
    return [
        cell
        for cell in manifest["semantic"]["cells"]
        if cell["cell_id"] in executable
    ]


def _outcome_index(campaigns):
    """Load outcomes for ``(manifest, out)`` campaigns, keyed by cell id.

    Also returns the campaign directory that holds each cell's artifacts, so
    pilot and continuation cells can be paired from different campaigns.
    """
    outcomes, locations = {}, {}
    for manifest, out in campaigns:
        for cell in _executable_cells(manifest):
            if cell["cell_id"] in outcomes:
                raise RuntimeError(f"Duplicate cell: {cell['cell_id']}")
            outcomes[cell["cell_id"]] = _load_outcome(manifest, out, cell)
            locations[cell["cell_id"]] = out
    return outcomes, locations


def _pairs(outcomes, locations, seeds):
    pairs = []
    initial_mismatches = []
    for label in LABELS:
        for seed in seeds:
            arms = {arm: outcomes[cell_id(label, seed, arm)] for arm in ARMS}
            statuses = {v["status"] for v in arms.values()}
            status = (
                "complete"
                if statuses == {"success"}
                else ("failure" if "failure" in statuses else "missing")
            )
            pair = {"label": label, "seed": seed, "status": status, **arms}
            pair["initial_design_equal"] = None
            if status == "complete":
                traces = [
                    _paths(
                        locations[arms[a]["cell"]["cell_id"]], arms[a]["cell"]
                    )["npz"]
                    for a in ARMS
                ]
                with np.load(traces[0], allow_pickle=False) as s0, np.load(
                    traces[1], allow_pickle=False
                ) as s2:
                    equal = bool(
                        np.array_equal(s0["X_init"], s2["X_init"])
                        and np.array_equal(s0["y_init"], s2["y_init"])
                    )
                pair["initial_design_equal"] = equal
                if not equal:
                    initial_mismatches.append({"label": label, "seed": seed})
                    pair["status"] = "invalid_initial_design"
                elif any(
                    not _finite(arms[arm]["final"].get(metric))
                    for arm in ARMS
                    for metric in (
                        *QUALITY,
                        "wall_s",
                        "target_eval_s",
                        "func_count",
                    )
                ) or any(
                    not _finite(arms[arm].get("search_s")) for arm in ARMS
                ):
                    pair["status"] = "nonfinite"
            pairs.append(pair)
    return pairs, initial_mismatches


def _target_report(rows, scheduled_pairs):
    """Per-target paired summaries; the ten-pair review trigger needs all ten."""
    accuracy = {metric: _accuracy(rows, metric) for metric in QUALITY}
    full_ten = all(p["status"] == "complete" for p in rows) and len(rows) == 10
    review_metrics = [
        metric
        for metric, report in accuracy.items()
        if full_ten and report["median"] > 0 and report["worse_count"] >= 7
    ]
    return {
        "scheduled_pairs": scheduled_pairs,
        "status_counts": {
            s: sum(p["status"] == s for p in rows) for s in PAIR_STATUSES
        },
        "pairs": rows,
        "log_ratio_intervals_S2_over_S0": {
            metric: _log_ratio_interval(rows, metric)
            for metric in POSITIVE_PAIRED
        },
        "accuracy_differences": accuracy,
        "convergence": _transitions(rows, "converged"),
        "usability": _transitions(rows, "usable"),
        "full_ten_pair_review_metrics": review_metrics,
    }


def _campaign_flags(outcomes, pairs):
    new_failures = [
        {"label": p["label"], "seed": p["seed"]}
        for p in pairs
        if p["S0"]["status"] == "success" and p["S2"]["status"] == "failure"
    ]
    usable_losses = [
        {"label": p["label"], "seed": p["seed"]}
        for p in pairs
        if p["status"] == "complete"
        and p["S0"]["usable"]
        and not p["S2"]["usable"]
    ]
    nonfinite_outcomes = []
    for key, outcome in outcomes.items():
        if outcome["status"] != "success":
            continue
        metrics = {
            metric: outcome["final"].get(metric)
            for metric in (*QUALITY, "wall_s", "target_eval_s", "func_count")
        } | {"search_s": outcome.get("search_s")}
        bad = [
            metric for metric, value in metrics.items() if not _finite(value)
        ]
        if bad:
            nonfinite_outcomes.append({"cell_id": key, "metrics": bad})
    return {
        "fit_status_counts": {
            s: sum(o["status"] == s for o in outcomes.values())
            for s in ("success", "failure", "missing")
        },
        "pair_status_counts": {
            s: sum(p["status"] == s for p in pairs) for s in PAIR_STATUSES
        },
        "nonfinite_outcomes": nonfinite_outcomes,
        "new_S2_failures": new_failures,
        "S0_usable_to_S2_unusable": usable_losses,
        "reference_exceedances": [
            {"cell_id": key, "metrics": value.get("reference_exceedances", [])}
            for key, value in outcomes.items()
            if value.get("reference_exceedances")
        ],
    }


def _review_block(flags, initial_mismatches, by_target):
    return {
        "automatic_adoption": False,
        "new_failure": bool(flags["new_S2_failures"]),
        "usability_loss": bool(flags["S0_usable_to_S2_unusable"]),
        "invalid_initial_design": bool(initial_mismatches),
        "nonfinite_outcome": bool(flags["nonfinite_outcomes"]),
        "full_ten_pair_accuracy_triggers": {
            label: row["full_ten_pair_review_metrics"]
            for label, row in by_target.items()
            if row["full_ten_pair_review_metrics"]
        },
    }


def summarize(manifest, out):
    """Summarize one campaign: the 24-fit pilot or the 96-fit continuation."""
    validate_manifest(manifest)
    kind = _kind(manifest)
    seeds = _seeds_for(manifest)
    outcomes, locations = _outcome_index([(manifest, out)])
    pairs, initial_mismatches = _pairs(outcomes, locations, seeds)
    by_target = {}
    for label in LABELS:
        rows = [p for p in pairs if p["label"] == label]
        by_target[label] = _target_report(rows, len(seeds))
        if kind == PILOT_KIND:
            by_target[label]["pilot_quality_gate"] = "descriptive_only"
    flags = _campaign_flags(outcomes, pairs)
    report = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "manifest_semantic_digest": manifest["semantic_digest"],
        "scheduled_pairs": len(pairs),
        "scheduled_fits": len(outcomes),
    }
    if kind == PILOT_KIND:
        executable = set(manifest["semantic"]["executable_cell_ids"])
        continuation_ids = [
            cell["cell_id"]
            for cell in manifest["semantic"]["cells"]
            if cell["cell_id"] not in executable
        ]
        report["planned_continuation"] = {
            "allocated": False,
            "fit_count": len(continuation_ids),
            "cell_ids": continuation_ids,
        }
    else:
        report["phase"] = "continuation"
        report["parent_manifest_semantic_digest"] = manifest["semantic"][
            "parent"
        ]["semantic_digest"]
    report.update(
        fit_status_counts=flags["fit_status_counts"],
        pair_status_counts=flags["pair_status_counts"],
        initial_design_mismatches=initial_mismatches,
        nonfinite_outcomes=flags["nonfinite_outcomes"],
        new_S2_failures=flags["new_S2_failures"],
        S0_usable_to_S2_unusable=flags["S0_usable_to_S2_unusable"],
        reference_exceedances=flags["reference_exceedances"],
        targets=by_target,
    )
    review = _review_block(flags, initial_mismatches, by_target)
    descriptive_key = (
        "pilot_is_descriptive_only"
        if kind == PILOT_KIND
        else "phase_summary_is_descriptive_only"
    )
    report["scientific_review"] = {
        "automatic_adoption": False,
        descriptive_key: True,
        **{k: v for k, v in review.items() if k != "automatic_adoption"},
    }
    return report


def _json_text(value):
    """Serialize for equality checks; NaN compares equal to NaN."""
    return json.dumps(value, sort_keys=True, allow_nan=True)


def _pilot_consistency(parent_root, pairs):
    """Check recomputed pilot pairs against the immutable pilot summary."""
    stored = json.loads(
        (Path(parent_root) / "summary.json").read_text(encoding="utf-8")
    )
    mismatches = []
    for pair in pairs:
        if pair["seed"] not in PILOT_SEEDS:
            continue
        stored_pair = next(
            p
            for p in stored["targets"][pair["label"]]["pairs"]
            if p["seed"] == pair["seed"]
        )
        compared = [
            (stored_pair["status"], pair["status"]),
            (
                stored_pair["initial_design_equal"],
                pair["initial_design_equal"],
            ),
        ]
        for arm in ARMS:
            compared += [
                (stored_pair[arm].get("final"), pair[arm].get("final")),
                (stored_pair[arm].get("search_s"), pair[arm].get("search_s")),
                (
                    stored_pair[arm].get("terminal", {}).get("record_digest"),
                    pair[arm].get("terminal", {}).get("record_digest"),
                ),
            ]
        if any(_json_text(a) != _json_text(b) for a, b in compared):
            mismatches.append({"label": pair["label"], "seed": pair["seed"]})
    return {
        "pilot_summary_sha256": sha256(Path(parent_root) / "summary.json"),
        "pilot_pairs_checked": sum(p["seed"] in PILOT_SEEDS for p in pairs),
        "mismatches": mismatches,
    }


def _pooled(by_target):
    """Equal-weight geometric means of per-target ratios (descriptive)."""
    pooled = {}
    for metric in POSITIVE_PAIRED:
        intervals = {
            label: row["log_ratio_intervals_S2_over_S0"][metric]
            for label, row in by_target.items()
        }
        available = {
            label: interval["geometric_mean_ratio"]
            for label, interval in intervals.items()
            if interval["geometric_mean_ratio"] is not None
        }
        pooled[metric] = {
            "targets_available": sorted(available),
            "targets_scheduled": len(intervals),
            "pairs_available_by_target": {
                label: interval["n_available_pairs"]
                for label, interval in sorted(intervals.items())
            },
            "equal_weight_geometric_mean_of_target_ratios": (
                math.exp(
                    float(np.mean([math.log(v) for v in available.values()]))
                )
                if available
                else None
            ),
        }
    return pooled


def summarize_combined(manifest, out):
    """Report all ten paired seeds per target from the pilot and continuation.

    Pilot outcomes are read from the bound pilot campaign under the pilot
    manifest; continuation outcomes from ``out`` under the continuation
    manifest.  Missing and failed fits stay in every denominator.
    """
    validate_manifest(manifest)
    if _kind(manifest) != CONTINUATION_KIND:
        raise RuntimeError(
            "The combined summary needs the continuation manifest"
        )
    parent = _load_parent(manifest)
    parent_info = manifest["semantic"]["parent"]
    parent_campaign = Path(parent_info["campaign_dir"])
    verify_parent_campaign(parent, parent_campaign)
    outcomes, locations = _outcome_index(
        [(parent, parent_campaign), (manifest, out)]
    )
    pairs, initial_mismatches = _pairs(outcomes, locations, FULL_SEEDS)
    for pair in pairs:
        pair["phase"] = (
            "pilot" if pair["seed"] in PILOT_SEEDS else "continuation"
        )
    by_target = {}
    for label in LABELS:
        rows = [p for p in pairs if p["label"] == label]
        by_target[label] = _target_report(rows, len(FULL_SEEDS))
        by_target[label]["pilot_pairs"] = sum(
            p["phase"] == "pilot" for p in rows
        )
        by_target[label]["continuation_pairs"] = sum(
            p["phase"] == "continuation" for p in rows
        )
    flags = _campaign_flags(outcomes, pairs)
    consistency = _pilot_consistency(Path(parent_info["path"]).parent, pairs)
    if consistency["mismatches"]:
        raise RuntimeError(
            "Recomputed pilot pairs differ from the immutable pilot summary: "
            f"{consistency['mismatches']}"
        )
    incomplete = sorted(
        label
        for label, row in by_target.items()
        if row["status_counts"]["complete"] != len(FULL_SEEDS)
    )
    review = _review_block(flags, initial_mismatches, by_target)
    return {
        "schema_version": 1,
        "kind": "noisy_acq_e5_combined_summary",
        "created_utc": _utc_now(),
        "continuation_manifest_semantic_digest": manifest["semantic_digest"],
        "pilot_manifest_semantic_digest": parent["semantic_digest"],
        "phases": {
            "pilot": {
                "seeds": list(PILOT_SEEDS),
                "fits": 24,
                "pairs": 12,
                "campaign_dir": str(parent_campaign),
            },
            "continuation": {
                "seeds": list(CONTINUATION_SEEDS),
                "fits": 96,
                "pairs": 48,
                "campaign_dir": str(Path(out).resolve()),
            },
        },
        "scheduled_pairs": len(pairs),
        "scheduled_fits": len(outcomes),
        "pilot_consistency": consistency,
        **flags,
        "initial_design_mismatches": initial_mismatches,
        "targets": by_target,
        "pooled_equal_weight": _pooled(by_target),
        "scientific_review": {
            "automatic_adoption": False,
            "descriptive_only": True,
            "incomplete_targets": incomplete,
            "planned_degrees_of_freedom_for_ten_complete_pairs": 9,
            **{k: v for k, v in review.items() if k != "automatic_adoption"},
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--selection", type=Path, required=True)
    prepare.add_argument("--reference", type=Path, default=REFERENCE_DEFAULT)
    prepare.add_argument(
        "--promotion-manifest", type=Path, default=PROMOTION_DEFAULT
    )
    prepare.add_argument("--out", type=Path, required=True)
    continuation = commands.add_parser(
        "prepare-continuation",
        help="allocate the 96 remaining fits under a completed pilot",
    )
    continuation.add_argument("--parent-manifest", type=Path, required=True)
    continuation.add_argument("--parent-campaign", type=Path, required=True)
    continuation.add_argument(
        "--reference", type=Path, default=REFERENCE_DEFAULT
    )
    continuation.add_argument(
        "--promotion-manifest", type=Path, default=PROMOTION_DEFAULT
    )
    continuation.add_argument("--out", type=Path, required=True)
    for name in (
        "cell",
        "run-pilot",
        "run-continuation",
        "validation-replay",
        "_worker",
        "summary",
        "summary-combined",
    ):
        epilog = None
        if name == "run-continuation":
            epilog = (
                "Exit status: 0 when every continuation cell has a success "
                "terminal; 1 when a failed fit was encountered or skipped; "
                "2 when the batch ended cleanly (time or fit limit) with "
                "cells still missing. Each batch writes batches/batch_*.json "
                "and status.json under --out."
            )
        command = commands.add_parser(name, epilog=epilog)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name in ("cell", "_worker"):
            command.add_argument("--cell-id", required=True)
        if name in (
            "cell",
            "run-pilot",
            "run-continuation",
            "validation-replay",
        ):
            command.add_argument("--timeout", type=int, default=1200)
        if name == "run-pilot":
            command.add_argument("--max-wall-seconds", type=int, default=10800)
        if name == "run-continuation":
            command.add_argument(
                "--max-wall-seconds",
                type=int,
                required=True,
                help="explicit batch allowance; no fit starts once less than "
                "one per-fit timeout remains",
            )
            command.add_argument(
                "--max-fits",
                type=int,
                default=None,
                help="optional cap on fits launched by this batch",
            )
            command.add_argument(
                "--on-failure",
                choices=("stop", "continue"),
                default="stop",
                help="stop (default) ends the batch at a failed fit so it "
                "can be investigated; continue records it and proceeds",
            )
        if name in ("summary", "summary-combined"):
            command.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        atomic_json(
            args.out,
            prepare_manifest(
                args.selection, args.reference, args.promotion_manifest
            ),
        )
        return 0
    if args.command == "prepare-continuation":
        if args.out.exists():
            raise RuntimeError(
                f"Refusing to overwrite an existing manifest: {args.out}"
            )
        prepared = prepare_continuation_manifest(
            args.parent_manifest,
            args.parent_campaign,
            args.reference,
            args.promotion_manifest,
        )
        atomic_json(args.out, prepared)
        semantic = prepared["semantic"]
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "semantic_digest": prepared["semantic_digest"],
                    "prepared_from_commit": prepared["prepared_from_commit"],
                    "executable_fits": len(semantic["executable_cell_ids"]),
                    "continuation_seeds": semantic["continuation_seeds"],
                    "changed_sources": sorted(
                        semantic["identity_provenance"]["changed_sources"]
                    ),
                    "unchanged_source_count": semantic["identity_provenance"][
                        "unchanged_source_count"
                    ],
                    "parent_semantic_digest": semantic["parent"][
                        "semantic_digest"
                    ],
                    "reviewed_ready": prepared["reviewed_ready"]["approved"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "_worker":
        return worker(args.manifest, args.out, args.cell_id)
    if args.command == "cell":
        manifest = validate_manifest(
            json.loads(args.manifest.read_text(encoding="utf-8")),
            require_ready=True,
        )
        if _kind(manifest) == CONTINUATION_KIND:
            verify_parent_campaign(
                _load_parent(manifest),
                manifest["semantic"]["parent"]["campaign_dir"],
            )
        return run_cell(args.manifest, args.out, args.cell_id, args.timeout)
    if args.command == "run-pilot":
        if args.max_wall_seconds < 1200 or args.max_wall_seconds > 10800:
            raise RuntimeError(
                "Pilot max wall seconds must be between 1200 and 10800"
            )
        return run_pilot(
            args.manifest, args.out, args.timeout, args.max_wall_seconds
        )
    if args.command == "run-continuation":
        return run_continuation(
            args.manifest,
            args.out,
            args.timeout,
            args.max_wall_seconds,
            args.max_fits,
            args.on_failure,
        )
    if args.command == "validation-replay":
        return run_validation_replay(args.manifest, args.out, args.timeout)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.command == "summary-combined":
        report = summarize_combined(manifest, args.out)
        target = args.report or (args.out / "summary_combined.json")
    else:
        report = summarize(manifest, args.out)
        target = args.report or (args.out / "summary.json")
    if target.exists():
        raise RuntimeError(
            f"Refusing to overwrite immutable summary: {target}"
        )
    atomic_json(target, report)
    print(json.dumps(report["fit_status_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
