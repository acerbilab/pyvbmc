"""Machine-local calibration identity, cache, and campaign locking.

This module deliberately contains no numerical calibration code. Cache reads
are side-effect free, while directory creation and locking are reserved for
the explicit :func:`pyvbmc.calibrate` entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import scipy
from filelock import FileLock, Timeout
from platformdirs import user_cache_dir
from threadpoolctl import threadpool_info

from .profile import CalibrationProfile, default_profile

CACHE_SCHEMA_VERSION = 1
KERNEL_REVISION = "chunk-kernels-v1"
WORKLOAD_REVISION = "machine-calibration-v1"
CANDIDATE_BUDGETS = frozenset(2**power for power in range(14, 19))
MAX_CACHE_BYTES = 1024 * 1024
MAX_REPORT_DEPTH = 8
MAX_REPORT_ITEMS = 10_000
MAX_STRING_LENGTH = 16_384

_SETTING_NAMES = (
    "pdf_chunk_elements",
    "entropy_grad_chunk_elements",
    "entropy_value_chunk_elements",
)
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_DYNAMIC",
    "MKL_CBWR",
    "NPY_DISABLE_CPU_FEATURES",
    "NPY_ENABLE_CPU_FEATURES",
)
_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "kernel_revision",
        "workload_revision",
        "fingerprint",
        "identity",
        "settings",
        "status",
        "provenance",
        "report",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "cpu_model",
        "architecture",
        "host_hash",
        "operating_system",
        "python",
        "numpy",
        "scipy",
        "backends",
        "thread_environment",
        "device",
        "dtype",
    }
)
_BACKEND_FIELDS = frozenset(
    {
        "user_api",
        "internal_api",
        "prefix",
        "version",
        "num_threads",
        "threading_layer",
        "architecture",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {"pyvbmc_version", "calibrated_at_utc", "elapsed_seconds"}
)
_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "recipe_version",
        "status",
        "estimated_seconds",
        "watchdog_seconds",
        "candidate_budgets",
        "discovery_rounds",
        "heldout_rounds",
        "memory_limit_bytes",
        "groups",
        "numerical",
        "timings_seconds",
        "memory_estimates",
        "watchdog",
        "numpy_global_rng_unchanged",
        "settings",
        "summary",
    }
)
_GROUP_WORKLOADS = {
    "pdf": (
        "small_value",
        "sieve_value",
        "sieve_gradient",
        "large_density",
    ),
    "entropy_grad": (
        "adam_d4_k20",
        "boost_d4_k50",
        "boost_d15_k50",
        "active_d4_k20",
    ),
    "entropy_value": ("fine_d4_k20", "fine_d15_k26"),
}
_GROUP_SETTINGS = {
    "pdf": "pdf_chunk_elements",
    "entropy_grad": "entropy_grad_chunk_elements",
    "entropy_value": "entropy_value_chunk_elements",
}
_WORKLOAD_SHAPES = {
    "small_value": (4, 20, 8, 8),
    "sieve_value": (4, 20, 8192, 8192),
    "sieve_gradient": (4, 20, 8192, 8192),
    "large_density": (15, 26, 100_000, 100_000),
    "adam_d4_k20": (4, 20, 37, 38),
    "boost_d4_k50": (4, 50, 55, 56),
    "boost_d15_k50": (15, 50, 55, 56),
    "active_d4_k20": (4, 20, 200, 200),
    "fine_d4_k20": (4, 20, 4096, 4096),
    "fine_d15_k26": (15, 26, 4096, 4096),
}
_TIMING_FIELDS = frozenset(
    {"setup", "numerical", "discovery", "heldout", "total"}
)
# These tables define report schema v1. Keep them aligned with the workload
# recipe when WORKLOAD_REVISION changes.

_CAMPAIGN_GUARD = threading.Lock()
_REGISTRY_LOCK = threading.Lock()
_PROFILE_REGISTRY: dict[tuple[str, str], CalibrationProfile] = {}
_REPORT_REGISTRY: dict[tuple[str, str], Mapping[str, Any]] = {}
_SUGGESTED_FINGERPRINTS: set[str] = set()


def _package_version() -> str | None:
    try:
        return version("pyvbmc")
    except PackageNotFoundError:
        return None


def _cpu_model() -> str | None:
    model = platform.processor().strip()
    if not model:
        model = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    if not model and sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as stream:
                for line in stream:
                    if line.lower().startswith("model name"):
                        model = line.partition(":")[2].strip()
                        break
        except OSError:
            pass
    return model or None


def _backend_identity() -> list[dict[str, Any]]:
    """Return a stable, path-free description of loaded native runtimes."""
    backends = []
    for item in threadpool_info():
        backend = {
            name: item.get(name)
            for name in (
                "user_api",
                "internal_api",
                "prefix",
                "version",
                "num_threads",
                "threading_layer",
                "architecture",
            )
        }
        backends.append(backend)
    return sorted(
        backends,
        key=lambda item: tuple(str(item[name]) for name in sorted(item)),
    )


def machine_identity() -> dict[str, Any]:
    """Collect the compatibility identity without changing thread settings."""
    hostname = socket.gethostname().strip()
    host_hash = (
        hashlib.sha256(hostname.encode("utf-8")).hexdigest()
        if hostname
        else None
    )
    return {
        "cpu_model": _cpu_model(),
        "architecture": platform.machine().strip() or None,
        "host_hash": host_hash,
        "operating_system": {
            "system": platform.system().strip() or None,
            "release": platform.release().strip() or None,
        },
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "backends": _backend_identity(),
        "thread_environment": {
            name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
        },
        "device": "cpu",
        "dtype": "float64",
    }


def identity_reuse_reason(identity: Mapping[str, Any]) -> str | None:
    """Return why *identity* is unsafe for persistent reuse, if applicable."""
    for field in ("cpu_model", "architecture", "host_hash"):
        if not identity.get(field):
            return f"missing {field.replace('_', ' ')}"
    operating_system = identity.get("operating_system")
    if not isinstance(operating_system, Mapping) or not operating_system.get(
        "system"
    ):
        return "missing operating system identity"
    backends = identity.get("backends")
    if not isinstance(backends, list) or not backends:
        return "missing numerical backend identity"
    if not any(
        isinstance(item, Mapping)
        and item.get("internal_api")
        and item.get("version")
        and _is_positive_int(item.get("num_threads"))
        for item in backends
    ):
        return "incomplete numerical backend identity"
    return None


def fingerprint(identity: Mapping[str, Any] | None = None) -> str:
    """Return the stable compatibility fingerprint for an identity."""
    if identity is None:
        identity = machine_identity()
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kernel_revision": KERNEL_REVISION,
        "workload_revision": WORKLOAD_REVISION,
        "identity": identity,
    }
    encoded = json.dumps(
        payload, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_root() -> Path:
    """Return the application cache root without creating it."""
    override = os.environ.get("PYVBMC_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(user_cache_dir("pyvbmc", appauthor=False, opinion=False))


def cache_path(current_fingerprint: str) -> Path:
    return (
        cache_root()
        / "calibration"
        / f"v{CACHE_SCHEMA_VERSION}"
        / f"{current_fingerprint}.json"
    )


def _registry_key(current_fingerprint: str) -> tuple[str, str]:
    root = os.path.abspath(os.path.normpath(cache_root()))
    return (root, current_fingerprint)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_json_value(value: Any, *, depth: int = 0) -> int:
    if depth > MAX_REPORT_DEPTH:
        raise ValueError("calibration report is nested too deeply")
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
            raise ValueError("calibration report string is too long")
        return 1
    if isinstance(value, int) and not isinstance(value, bool):
        return 1
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("calibration report contains a non-finite number")
        return 1
    if isinstance(value, list):
        count = 1
        for item in value:
            count += _validate_json_value(item, depth=depth + 1)
            if count > MAX_REPORT_ITEMS:
                raise ValueError("calibration report has too many items")
        return count
    if isinstance(value, dict):
        count = 1
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > MAX_STRING_LENGTH:
                raise ValueError("calibration report has an invalid key")
            count += _validate_json_value(item, depth=depth + 1)
            if count > MAX_REPORT_ITEMS:
                raise ValueError("calibration report has too many items")
        return count
    raise ValueError(
        f"calibration report contains unsupported {type(value).__name__}"
    )


def _validate_settings(settings: Any) -> dict[str, int]:
    if not isinstance(settings, dict) or set(settings) != set(_SETTING_NAMES):
        raise ValueError("calibration settings have invalid fields")
    validated = {}
    for name in _SETTING_NAMES:
        value = settings[name]
        if not _is_positive_int(value) or value not in CANDIDATE_BUDGETS:
            raise ValueError(f"invalid calibrated value for {name}")
        validated[name] = value
    return validated


def _validate_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_FIELDS:
        raise ValueError("calibration identity has invalid fields")
    _validate_json_value(identity)
    for name in ("cpu_model", "architecture", "host_hash"):
        if identity[name] is not None and not isinstance(identity[name], str):
            raise ValueError(f"calibration identity has invalid {name}")
    for name in ("python", "numpy", "scipy"):
        if not isinstance(identity[name], str) or not identity[name]:
            raise ValueError(f"calibration identity has invalid {name}")
    if identity["device"] != "cpu" or identity["dtype"] != "float64":
        raise ValueError("calibration identity has invalid device or dtype")
    operating_system = identity["operating_system"]
    if not isinstance(operating_system, dict) or set(operating_system) != {
        "system",
        "release",
    }:
        raise ValueError("calibration operating system identity is invalid")
    for value in operating_system.values():
        if value is not None and not isinstance(value, str):
            raise ValueError(
                "calibration operating system identity is invalid"
            )
    environment = identity["thread_environment"]
    if not isinstance(environment, dict) or set(environment) != set(
        _THREAD_ENVIRONMENT
    ):
        raise ValueError("calibration thread environment is invalid")
    if any(
        value is not None and not isinstance(value, str)
        for value in environment.values()
    ):
        raise ValueError("calibration thread environment is invalid")
    backends = identity["backends"]
    if not isinstance(backends, list):
        raise ValueError("calibration backend identity is invalid")
    for backend in backends:
        if not isinstance(backend, dict) or set(backend) != _BACKEND_FIELDS:
            raise ValueError("calibration backend identity is invalid")
        for name, value in backend.items():
            if name == "num_threads":
                if value is not None and not _is_positive_int(value):
                    raise ValueError(
                        "calibration backend thread count is invalid"
                    )
            elif value is not None and not isinstance(value, str):
                raise ValueError("calibration backend identity is invalid")
    return identity


def _validate_provenance(provenance: Any) -> dict[str, Any]:
    if (
        not isinstance(provenance, dict)
        or set(provenance) != _PROVENANCE_FIELDS
    ):
        raise ValueError("calibration provenance has invalid fields")
    package_version = provenance["pyvbmc_version"]
    if package_version is not None and (
        not isinstance(package_version, str) or not package_version
    ):
        raise ValueError("calibration provenance version is invalid")
    calibrated_at = provenance["calibrated_at_utc"]
    if not isinstance(calibrated_at, str) or not calibrated_at:
        raise ValueError("calibration provenance timestamp is invalid")
    elapsed = provenance["elapsed_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("calibration provenance elapsed time is invalid")
    return provenance


def _validate_report(
    report: Any, expected_settings: Mapping[str, int]
) -> dict[str, Any]:
    if not isinstance(report, dict) or set(report) != _REPORT_FIELDS:
        raise ValueError("calibration report has invalid fields")
    _validate_json_value(report)
    for name in ("schema_version", "recipe_version"):
        if not _is_positive_int(report[name]) or report[name] != 1:
            raise ValueError("calibration report has an invalid version")
    if report["status"] != "complete":
        raise ValueError("calibration report is incomplete")
    if report["candidate_budgets"] != sorted(CANDIDATE_BUDGETS):
        raise ValueError("calibration report has invalid candidates")
    if report["settings"] != dict(expected_settings):
        raise ValueError("calibration report settings do not match the record")
    for name in (
        "estimated_seconds",
        "watchdog_seconds",
        "discovery_rounds",
        "heldout_rounds",
        "memory_limit_bytes",
    ):
        value = report[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"calibration report has invalid {name}")
    if report["numpy_global_rng_unchanged"] is not True:
        raise ValueError("calibration report did not preserve the global RNG")
    for name in (
        "groups",
        "numerical",
        "timings_seconds",
        "memory_estimates",
        "watchdog",
        "summary",
    ):
        if not isinstance(report[name], dict):
            raise ValueError(f"calibration report has invalid {name}")
    _validate_report_timings(report["timings_seconds"])
    _validate_report_watchdog(report["watchdog"])
    _validate_report_groups(report["groups"], expected_settings)
    _validate_report_numerics(report["numerical"], expected_settings)
    _validate_report_memory(report["memory_estimates"])
    if set(report["summary"]) != set(_SETTING_NAMES):
        raise ValueError("calibration report summary has invalid settings")
    for name, summary in report["summary"].items():
        if not isinstance(summary, dict) or set(summary) != {
            "selected",
            "reason",
            "heldout_speedup",
        }:
            raise ValueError("calibration report summary is invalid")
        if summary["selected"] != expected_settings[name]:
            raise ValueError("calibration report summary has wrong selection")
        if not isinstance(summary["reason"], str) or not summary["reason"]:
            raise ValueError("calibration report summary has invalid reason")
        benefit = summary["heldout_speedup"]
        if benefit is not None and (
            isinstance(benefit, bool)
            or not isinstance(benefit, (int, float))
            or not math.isfinite(benefit)
            or benefit <= 0
        ):
            raise ValueError("calibration report summary has invalid speedup")
    return report


def _positive_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def _validate_report_timings(timings: Mapping[str, Any]) -> None:
    if set(timings) != _TIMING_FIELDS:
        raise ValueError("calibration report has invalid timing fields")
    for name, value in timings.items():
        valid = _positive_finite_number(value)
        if name == "setup":
            valid = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                and value >= 0
            )
        if not valid:
            raise ValueError(f"calibration report has invalid {name} timing")
    stage_total = sum(
        timings[name]
        for name in ("setup", "numerical", "discovery", "heldout")
    )
    tolerance = max(1e-12, abs(timings["total"]) * 1e-12)
    if timings["total"] + tolerance < stage_total:
        raise ValueError("calibration report total timing is inconsistent")


def _validate_report_watchdog(watchdog: Mapping[str, Any]) -> None:
    if set(watchdog) != {
        "deadline",
        "last_checkpoint",
        "maximum_kernel_call_seconds",
        "stopped",
    }:
        raise ValueError("calibration report has invalid watchdog fields")
    if not _positive_finite_number(watchdog["deadline"]):
        raise ValueError("calibration report has invalid watchdog deadline")
    if (
        not isinstance(watchdog["last_checkpoint"], str)
        or not watchdog["last_checkpoint"]
    ):
        raise ValueError("calibration report has invalid watchdog checkpoint")
    if not _positive_finite_number(watchdog["maximum_kernel_call_seconds"]):
        raise ValueError("calibration report has invalid kernel timing")
    if watchdog["stopped"] is not False:
        raise ValueError("completed calibration report has a stopped watchdog")


def _positive_timing_list(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(_positive_finite_number(item) for item in value)
    )


def _record_names(
    records: Any, expected: tuple[str, ...], section: str
) -> None:
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError(f"calibration report has incomplete {section}")
    names = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(
            record.get("name"), str
        ):
            raise ValueError(f"calibration report has invalid {section}")
        names.append(record["name"])
    if tuple(names) != expected:
        raise ValueError(f"calibration report has wrong {section} workloads")
    for record in records:
        if section == "discovery":
            first = record.get("first_timing_call_seconds")
            rounds = record.get("round_seconds")
            controls = record.get("default_control_seconds")
            aliases = record.get("aliases")
            layouts = record.get("layouts")
            estimates = record.get("workspace_estimates")
            shape = (
                record.get("D"),
                record.get("K"),
                record.get("requested_count"),
                record.get("effective_count"),
            )
            candidate_keys = {str(value) for value in CANDIDATE_BUDGETS}
            if (
                shape != _WORKLOAD_SHAPES[record["name"]]
                or not isinstance(aliases, dict)
                or set(aliases) != candidate_keys
                or any(
                    not _is_positive_int(value)
                    or value not in CANDIDATE_BUDGETS
                    for value in aliases.values()
                )
                or not isinstance(layouts, dict)
                or set(layouts) != candidate_keys
                or not isinstance(estimates, dict)
                or set(estimates) != candidate_keys
                or not isinstance(first, dict)
                or not first
                or set(first)
                != {str(value) for value in set(aliases.values())}
                or any(
                    not _positive_finite_number(value)
                    for value in first.values()
                )
                or not isinstance(rounds, dict)
                or not rounds
                or set(rounds) != set(first)
                or any(
                    not _positive_timing_list(value, 5)
                    for value in rounds.values()
                )
                or not _positive_timing_list(controls, 5)
            ):
                raise ValueError(
                    "calibration report has incomplete discovery timings"
                )
        else:
            for name in (
                "default_seconds",
                "selected_seconds",
                "default_control_seconds",
                "selected_control_seconds",
            ):
                if not _positive_timing_list(record.get(name), 4):
                    raise ValueError(
                        "calibration report has incomplete held-out timings"
                    )


def _validate_report_groups(
    groups: Mapping[str, Any], expected_settings: Mapping[str, int]
) -> None:
    if set(groups) != set(_GROUP_WORKLOADS):
        raise ValueError("calibration report has incomplete timing groups")
    for group, workloads in _GROUP_WORKLOADS.items():
        entry = groups[group]
        if not isinstance(entry, dict) or set(entry) != {
            "setting",
            "discovery",
            "discovery_selection",
            "heldout",
            "heldout_validation",
        }:
            raise ValueError("calibration report has invalid timing group")
        setting = _GROUP_SETTINGS[group]
        if entry["setting"] != setting:
            raise ValueError("calibration report group has wrong setting")
        _record_names(entry["discovery"], workloads, "discovery")
        _record_names(entry["heldout"], workloads, "held-out")
        selection = entry["discovery_selection"]
        validation = entry["heldout_validation"]
        if (
            not isinstance(selection, dict)
            or not isinstance(selection.get("accepted"), bool)
            or not _is_positive_int(selection.get("budget"))
            or selection["budget"] not in CANDIDATE_BUDGETS
            or not isinstance(selection.get("reason"), str)
            or not selection["reason"]
        ):
            raise ValueError("calibration report has invalid discovery result")
        if selection["accepted"] != (selection["budget"] != 2**16):
            raise ValueError(
                "calibration discovery acceptance is inconsistent"
            )
        if any(
            record.get("selected_budget") != selection["budget"]
            for record in entry["heldout"]
        ):
            raise ValueError("calibration held-out budget is inconsistent")
        candidates = selection.get("candidates")
        if not isinstance(candidates, dict) or set(candidates) != {
            str(value) for value in CANDIDATE_BUDGETS if value != 2**16
        }:
            raise ValueError("calibration discovery candidates are incomplete")
        if (
            not isinstance(validation, dict)
            or not isinstance(validation.get("pass"), bool)
            or validation.get("accepted_budget") != expected_settings[setting]
            or not isinstance(validation.get("reason"), str)
            or not validation["reason"]
        ):
            raise ValueError("calibration report has invalid held-out result")
        if selection["budget"] == 2**16 and validation["pass"] is not True:
            raise ValueError("default selection failed held-out validation")
        accepted = selection["budget"] if validation["pass"] else 2**16
        if validation["accepted_budget"] != accepted:
            raise ValueError("calibration held-out result is inconsistent")
        if (
            expected_settings[setting] != 2**16
            and validation["pass"] is not True
        ):
            raise ValueError("nondefault selection failed held-out validation")


def _validate_report_numerics(
    numerical: Mapping[str, Any], expected_settings: Mapping[str, int]
) -> None:
    if set(numerical) != {"dedicated", "workloads", "valid_by_group"}:
        raise ValueError("calibration report has invalid numerical fields")
    dedicated = numerical["dedicated"]
    if (
        not isinstance(dedicated, dict)
        or dedicated.get("pass") is not True
        or dedicated.get("inputs_vp_and_vp_rng_unchanged") is not True
        or not isinstance(dedicated.get("checks"), list)
        or not dedicated["checks"]
        or any(
            not isinstance(check, dict) or check.get("pass") is not True
            for check in dedicated["checks"]
        )
    ):
        raise ValueError(
            "calibration report failed dedicated numerical checks"
        )
    workload_records = numerical["workloads"]
    expected_workloads = {
        name for workloads in _GROUP_WORKLOADS.values() for name in workloads
    }
    if not isinstance(workload_records, dict) or set(workload_records) != (
        expected_workloads
    ):
        raise ValueError(
            "calibration report has incomplete numerical workloads"
        )
    default_key = str(2**16)
    for record in workload_records.values():
        if not isinstance(record, dict):
            raise ValueError("calibration report failed baseline numerics")
        budgets = record.get("valid_budgets")
        checks = record.get("checks")
        if (
            record.get("inputs_vp_and_vp_rng_unchanged") is not True
            or not isinstance(budgets, dict)
            or set(budgets) != {str(value) for value in CANDIDATE_BUDGETS}
            or any(not isinstance(value, bool) for value in budgets.values())
            or budgets.get(default_key) is not True
            or not _positive_finite_number(
                record.get("first_public_call_seconds")
            )
            or not isinstance(checks, list)
        ):
            raise ValueError("calibration report failed baseline numerics")
        check_budgets = []
        for check in checks:
            if (
                not isinstance(check, dict)
                or not _is_positive_int(check.get("budget"))
                or check["budget"] not in CANDIDATE_BUDGETS
                or not _positive_finite_number(
                    check.get("diagnostic_call_seconds")
                )
                or not isinstance(check.get("pass"), bool)
                or not isinstance(check.get("rng_advancement_exact"), bool)
                or check["pass"] != budgets[str(check["budget"])]
                or (check["pass"] and not check["rng_advancement_exact"])
            ):
                raise ValueError(
                    "calibration report has invalid numerical checks"
                )
            check_budgets.append(check["budget"])
        if set(check_budgets) != CANDIDATE_BUDGETS or len(
            check_budgets
        ) != len(CANDIDATE_BUDGETS):
            raise ValueError(
                "calibration report has incomplete numerical checks"
            )
    validity = numerical["valid_by_group"]
    if not isinstance(validity, dict) or set(validity) != set(
        _GROUP_WORKLOADS
    ):
        raise ValueError("calibration report has incomplete numerical groups")
    expected_candidates = {str(value) for value in CANDIDATE_BUDGETS}
    for group in _GROUP_WORKLOADS:
        values = validity[group]
        expected_validity = {
            key: all(
                workload_records[name]["valid_budgets"][key]
                for name in _GROUP_WORKLOADS[group]
            )
            for key in expected_candidates
        }
        if (
            not isinstance(values, dict)
            or set(values) != expected_candidates
            or any(not isinstance(value, bool) for value in values.values())
            or values.get(default_key) is not True
            or values != expected_validity
            or values.get(str(expected_settings[_GROUP_SETTINGS[group]]))
            is not True
        ):
            raise ValueError(
                "calibration report has invalid candidate numerics"
            )


def _validate_report_memory(memory: Mapping[str, Any]) -> None:
    expected_workloads = {
        name for workloads in _GROUP_WORKLOADS.values() for name in workloads
    }
    if set(memory) != expected_workloads:
        raise ValueError("calibration report has incomplete memory coverage")
    expected_candidates = {str(value) for value in CANDIDATE_BUDGETS}
    for workload in memory.values():
        if (
            not isinstance(workload, dict)
            or set(workload) != expected_candidates
        ):
            raise ValueError(
                "calibration report has invalid memory candidates"
            )
        for estimate in workload.values():
            if (
                not isinstance(estimate, dict)
                or set(estimate)
                != {"bytes", "limit_bytes", "within_limit", "label"}
                or estimate.get("within_limit") is not True
                or not _is_positive_int(estimate.get("bytes"))
                or not _is_positive_int(estimate.get("limit_bytes"))
                or estimate["bytes"] > estimate["limit_bytes"]
                or not isinstance(estimate.get("label"), str)
                or not estimate["label"]
            ):
                raise ValueError(
                    "calibration report has invalid memory estimate"
                )


def validate_record(
    record: Any,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Validate and return a cache record suitable for trusted reuse."""
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        raise ValueError("calibration cache record has invalid fields")
    if (
        not _is_positive_int(record["schema_version"])
        or record["schema_version"] != CACHE_SCHEMA_VERSION
        or record["kernel_revision"] != KERNEL_REVISION
        or record["workload_revision"] != WORKLOAD_REVISION
        or record["status"] != "complete"
    ):
        raise ValueError("unsupported or incomplete calibration cache record")
    identity = _validate_identity(record["identity"])
    if identity_reuse_reason(identity) is not None:
        raise ValueError("calibration identity is insufficient for reuse")
    actual_fingerprint = fingerprint(identity)
    if record["fingerprint"] != actual_fingerprint:
        raise ValueError("calibration fingerprint does not match its identity")
    if (
        expected_fingerprint is not None
        and actual_fingerprint != expected_fingerprint
    ):
        raise ValueError("calibration fingerprint is incompatible")
    if expected_identity is not None and identity != dict(expected_identity):
        raise ValueError("calibration identity is incompatible")
    _validate_settings(record["settings"])
    _validate_provenance(record["provenance"])
    _validate_report(record["report"], record["settings"])
    return record


def make_record(
    *,
    identity: Mapping[str, Any],
    settings: Mapping[str, Any],
    report: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build and validate a complete persistent cache record."""
    current_fingerprint = fingerprint(identity)
    record = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kernel_revision": KERNEL_REVISION,
        "workload_revision": WORKLOAD_REVISION,
        "fingerprint": current_fingerprint,
        "identity": dict(identity),
        "settings": dict(settings),
        "status": "complete",
        "provenance": {
            "pyvbmc_version": _package_version(),
            "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": float(elapsed_seconds),
        },
        "report": dict(report),
    }
    return validate_record(record)


def _profile_from_record(
    record: Mapping[str, Any], path: Path
) -> CalibrationProfile:
    provenance = dict(record["provenance"])
    provenance.update(
        {
            "kernel_revision": record["kernel_revision"],
            "workload_revision": record["workload_revision"],
        }
    )
    return CalibrationProfile(
        **record["settings"],
        source="cache",
        status="complete",
        fingerprint=record["fingerprint"],
        cache_path=str(path),
        provenance=provenance,
    )


def read_record(
    identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None, Path]:
    """Read a compatible record, returning ``(record, miss_reason, path)``."""
    if identity is None:
        identity = machine_identity()
    current_fingerprint = fingerprint(identity)
    path = cache_path(current_fingerprint)
    insufficient = identity_reuse_reason(identity)
    if insufficient is not None:
        return None, insufficient, path
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_CACHE_BYTES + 1)
        if len(payload) > MAX_CACHE_BYTES:
            return None, "cache record exceeds the size limit", path
        record = json.loads(payload.decode("utf-8"))
        validate_record(
            record,
            expected_identity=identity,
            expected_fingerprint=current_fingerprint,
        )
    except FileNotFoundError:
        return None, "no compatible calibration cache", path
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as error:
        return None, f"invalid calibration cache: {error}", path
    return record, None, path


def _resolve_cached_profile() -> tuple[CalibrationProfile, str | None]:
    identity = machine_identity()
    current_fingerprint = fingerprint(identity)
    key = _registry_key(current_fingerprint)
    with _REGISTRY_LOCK:
        registered = _PROFILE_REGISTRY.get(key)
    # A campaign which could not persist remains authoritative in this
    # process. Persisted entries are refreshed from disk so another process's
    # later atomic replacement is observed by each unresolved run.
    if registered is not None and registered.source == "memory":
        return registered, None
    record, reason, path = read_record(identity)
    if record is not None:
        profile = _profile_from_record(record, path)
        with _REGISTRY_LOCK:
            _PROFILE_REGISTRY[key] = profile
            _REPORT_REGISTRY[key] = record["report"]
        return profile, None
    return (
        default_profile(
            source="default",
            status="cache_miss",
            fingerprint=current_fingerprint,
            cache_path=None,
            provenance={"reason": reason or "cache miss"},
        ),
        reason,
    )


def resolve_cached_profile() -> CalibrationProfile:
    """Resolve a compatible cached profile or historical defaults.

    This function never creates directories, waits for locks, or starts a
    calibration campaign.
    """
    return _resolve_cached_profile()[0]


def suggest_calibration_once(
    reason: str | None = None, *, display: bool = True
) -> bool:
    """Print the cache-miss suggestion at most once per process identity."""
    if not display:
        return False
    identity = machine_identity()
    current_fingerprint = fingerprint(identity)
    with _REGISTRY_LOCK:
        if current_fingerprint in _SUGGESTED_FINGERPRINTS:
            return False
        _SUGGESTED_FINGERPRINTS.add(current_fingerprint)
    del reason
    print(
        "PyVBMC is using standard performance settings. You can optionally "
        "tune these for your machine by running pyvbmc.calibrate(). "
        "Calibration takes tens of seconds and does not evaluate your model."
    )
    return True


def write_record(record: Mapping[str, Any]) -> Path:
    """Atomically persist a fully validated complete record."""
    validated = validate_record(dict(record))
    path = cache_path(validated["fingerprint"])
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{validated['fingerprint']}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(
                validated,
                stream,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.flush()
            os.fsync(stream.fileno())
        if temporary_path.stat().st_size > MAX_CACHE_BYTES:
            raise ValueError("calibration cache record exceeds the size limit")
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return path


def register_success(
    profile: CalibrationProfile, report: Mapping[str, Any]
) -> None:
    """Register a successful profile/report for this process."""
    if profile.status != "complete" or not profile.fingerprint:
        raise ValueError(
            "only complete fingerprinted profiles can be registered"
        )
    _validate_report(dict(report), profile.settings)
    key = _registry_key(profile.fingerprint)
    with _REGISTRY_LOCK:
        _PROFILE_REGISTRY[key] = profile
        _REPORT_REGISTRY[key] = dict(report)


def _load_report(profile: CalibrationProfile) -> Mapping[str, Any] | None:
    """Return a validated detailed report for internal inspection/tests."""
    if not profile.fingerprint:
        return None
    key = _registry_key(profile.fingerprint)
    with _REGISTRY_LOCK:
        registered = _REPORT_REGISTRY.get(key)
    if registered is not None:
        return registered
    if not profile.cache_path:
        return None
    record, _, _ = read_record()
    if record is None or record["fingerprint"] != profile.fingerprint:
        return None
    return record["report"]


@dataclass
class CampaignGuard:
    """Result of a nonblocking campaign-lock attempt."""

    acquired: bool
    persistent: bool
    reason: str | None
    identity: dict[str, Any]
    fingerprint: str
    path: Path


@contextmanager
def campaign_guard() -> Iterator[CampaignGuard]:
    """Acquire process and native campaign locks without waiting."""
    identity = machine_identity()
    current_fingerprint = fingerprint(identity)
    path = cache_path(current_fingerprint)
    if not _CAMPAIGN_GUARD.acquire(blocking=False):
        yield CampaignGuard(
            False,
            False,
            "another calibration is active in this process",
            identity,
            current_fingerprint,
            path,
        )
        return

    native_lock = None
    acquired_native = False
    try:
        host_key = identity.get("host_hash") or "unknown-host"
        lock_path = (
            cache_root()
            / "calibration"
            / f"campaign-{str(host_key)[:32]}.lock"
        )
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            native_lock = FileLock(lock_path)
            native_lock.acquire(timeout=0)
            acquired_native = True
        except Timeout:
            yield CampaignGuard(
                False,
                True,
                "another calibration process holds the machine lock",
                identity,
                current_fingerprint,
                path,
            )
            return
        except OSError as error:
            yield CampaignGuard(
                True,
                False,
                f"cache unavailable ({error})",
                identity,
                current_fingerprint,
                path,
            )
            return
        yield CampaignGuard(
            True,
            True,
            None,
            identity,
            current_fingerprint,
            path,
        )
    finally:
        if acquired_native and native_lock is not None:
            native_lock.release()
        _CAMPAIGN_GUARD.release()


def _clear_process_state() -> None:
    """Reset process caches for isolated tests."""
    with _REGISTRY_LOCK:
        _PROFILE_REGISTRY.clear()
        _REPORT_REGISTRY.clear()
        _SUGGESTED_FINGERPRINTS.clear()
