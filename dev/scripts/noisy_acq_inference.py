"""Prepare, execute, and summarize the bounded E5 S0/S2 inference campaign.

The manifest describes the full paired design (six targets, seeds 2000--2009)
but only the two-seed pilot is executable.  Cases run sequentially in fresh
processes, write complete golden traces, and become resumable only after an
atomic, hash-validated terminal record is present.
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


def validate_manifest(manifest, require_ready=False, check_runtime=False):
    semantic = manifest.get("semantic")
    if not isinstance(semantic, dict) or manifest.get(
        "semantic_digest"
    ) != _digest(semantic):
        raise RuntimeError("Manifest semantic digest is invalid")
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
    allowed = semantic.get("executable_cell_ids")
    expected_allowed = [
        c["cell_id"] for c in expected if c["phase"] == "pilot"
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
    if semantic.get("options") != {}:
        raise RuntimeError(
            "E5 must use the standard target options without overrides"
        )
    if set(semantic.get("target_contracts", {})) != set(LABELS):
        raise RuntimeError("Manifest does not bind every target contract")
    if (
        semantic.get("planning", {}).get("pilot_operational_ceiling_seconds")
        != 10800
    ):
        raise RuntimeError("Pilot operational ceiling changed")
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
        if semantic.get("identity") != runtime_identity():
            raise RuntimeError(
                "Numerical/helper/data/dependency/environment identity changed"
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
            f"Cell is outside the executable pilot allocation: {requested}"
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


def summarize(manifest, out):
    validate_manifest(manifest)
    executable = set(manifest["semantic"]["executable_cell_ids"])
    scheduled_cells = [
        cell
        for cell in manifest["semantic"]["cells"]
        if cell["cell_id"] in executable
    ]
    outcomes = {
        c["cell_id"]: _load_outcome(manifest, out, c) for c in scheduled_cells
    }
    pairs = []
    initial_mismatches = []
    for label in LABELS:
        for seed in PILOT_SEEDS:
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
                traces = [_paths(out, arms[a]["cell"])["npz"] for a in ARMS]
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
    by_target = {}
    for label in LABELS:
        rows = [p for p in pairs if p["label"] == label]
        accuracy = {metric: _accuracy(rows, metric) for metric in QUALITY}
        full_ten = (
            all(p["status"] == "complete" for p in rows) and len(rows) == 10
        )
        review_metrics = [
            metric
            for metric, report in accuracy.items()
            if full_ten and report["median"] > 0 and report["worse_count"] >= 7
        ]
        by_target[label] = {
            "scheduled_pairs": len(PILOT_SEEDS),
            "status_counts": {
                s: sum(p["status"] == s for p in rows)
                for s in (
                    "complete",
                    "failure",
                    "missing",
                    "invalid_initial_design",
                    "nonfinite",
                )
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
            "pilot_quality_gate": "descriptive_only",
        }
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
    continuation_ids = [
        cell["cell_id"]
        for cell in manifest["semantic"]["cells"]
        if cell["cell_id"] not in executable
    ]
    report = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "manifest_semantic_digest": manifest["semantic_digest"],
        "scheduled_pairs": len(pairs),
        "scheduled_fits": len(outcomes),
        "planned_continuation": {
            "allocated": False,
            "fit_count": len(continuation_ids),
            "cell_ids": continuation_ids,
        },
        "fit_status_counts": {
            s: sum(o["status"] == s for o in outcomes.values())
            for s in ("success", "failure", "missing")
        },
        "pair_status_counts": {
            s: sum(p["status"] == s for p in pairs)
            for s in (
                "complete",
                "failure",
                "missing",
                "invalid_initial_design",
                "nonfinite",
            )
        },
        "initial_design_mismatches": initial_mismatches,
        "nonfinite_outcomes": nonfinite_outcomes,
        "new_S2_failures": new_failures,
        "S0_usable_to_S2_unusable": usable_losses,
        "reference_exceedances": [
            {"cell_id": key, "metrics": value.get("reference_exceedances", [])}
            for key, value in outcomes.items()
            if value.get("reference_exceedances")
        ],
        "targets": by_target,
        "scientific_review": {
            "automatic_adoption": False,
            "pilot_is_descriptive_only": True,
            "new_failure": bool(new_failures),
            "usability_loss": bool(usable_losses),
            "invalid_initial_design": bool(initial_mismatches),
            "nonfinite_outcome": bool(nonfinite_outcomes),
            "full_ten_pair_accuracy_triggers": {
                label: row["full_ten_pair_review_metrics"]
                for label, row in by_target.items()
                if row["full_ten_pair_review_metrics"]
            },
        },
    }
    return report


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
    for name in (
        "cell",
        "run-pilot",
        "validation-replay",
        "_worker",
        "summary",
    ):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name in ("cell", "_worker"):
            command.add_argument("--cell-id", required=True)
        if name in ("cell", "run-pilot", "validation-replay"):
            command.add_argument("--timeout", type=int, default=1200)
        if name == "run-pilot":
            command.add_argument("--max-wall-seconds", type=int, default=10800)
        if name == "summary":
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
    if args.command == "_worker":
        return worker(args.manifest, args.out, args.cell_id)
    if args.command == "cell":
        return run_cell(args.manifest, args.out, args.cell_id, args.timeout)
    if args.command == "run-pilot":
        if args.max_wall_seconds < 1200 or args.max_wall_seconds > 10800:
            raise RuntimeError(
                "Pilot max wall seconds must be between 1200 and 10800"
            )
        return run_pilot(
            args.manifest, args.out, args.timeout, args.max_wall_seconds
        )
    if args.command == "validation-replay":
        return run_validation_replay(args.manifest, args.out, args.timeout)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
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
