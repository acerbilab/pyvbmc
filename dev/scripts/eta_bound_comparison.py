"""Run the bounded, paired Phase 6 eta-bound comparison.

This is a development experiment, not a production API.  For every selected
saved state and replicate it generates one candidate pool with the production
(``C``) sieve, captures the post-sieve random state, and then supplies exact
copies of that pool to the three isolated variants in
``eta_bound_variants.py``.  The three fits therefore differ only in their
eta-bound objective/gradient treatment.

The runner is deliberately resumable at the state/replicate level.  An
``input.dill`` file is the authoritative paired input, arm captures are
written independently, and ``complete.json`` is written last.  Existing
artifacts are checked against the manifest and source configuration before
they are reused.  A recorded failure is never retried or overwritten.

Examples
--------
Run the timed pilot (the resulting pair is reused by the full invocation)::

    python dev/scripts/eta_bound_comparison.py \
        --manifest dev/experiments/eta_bound_20260908.json \
        --out dev/scripts/runs/latent_fixes/eta_bound_20260908 \
        --states normal_D2_singlesample --replicates 0

Resume with all eight states and all three paired replicates::

    python dev/scripts/eta_bound_comparison.py \
        --manifest dev/experiments/eta_bound_20260908.json \
        --out dev/scripts/runs/latent_fixes/eta_bound_20260908 \
        --replicates 0,1,2
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

import dill
import gpyreg
import numpy as np
import scipy
from eta_bound_variants import ARM_LABELS, ARMS, get_variant, penalty_terms

import pyvbmc
from pyvbmc.testing.oracles._state import build_state, load_snapshot
from pyvbmc.vbmc import variational_optimization as baseline_vo

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
MAX_STATES = 8
MAX_REPLICATES = 3
PAIR_INPUT_VERSION = 1
ARM_CAPTURE_VERSION = 1
SCORE_CAPTURE_VERSION = 1
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _jsonable(value: Any) -> Any:
    """Convert experiment metadata to strict, stable JSON values."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if callable(value):
        return {
            "callable": f"{getattr(value, '__module__', '')}."
            f"{getattr(value, '__qualname__', type(value).__qualname__)}"
        }
    if hasattr(value, "__dict__"):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": _jsonable(value.__dict__),
        }
    return {
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    data = (
        json.dumps(
            _jsonable(value), indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_bytes(path, data)


def _atomic_dill(path: Path, value: Any) -> None:
    stream = io.BytesIO()
    dill.dump(value, stream, recurse=True)
    _atomic_bytes(path, stream.getvalue())


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    _atomic_bytes(path, stream.getvalue())


def _load_dill(path: Path) -> Any:
    with path.open("rb") as stream:
        return dill.load(stream)


def _resolved_repo_path(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"source path leaves repository: {path}") from error
    return path


def _update_digest(
    digest: Any, value: Any, seen: set[int] | None = None
) -> None:
    """Hash nested numerical state without relying on pickle byte layout."""
    if seen is None:
        seen = set()
    if value is None:
        digest.update(b"none;")
    elif isinstance(value, (bool, np.bool_)):
        digest.update(b"bool:")
        digest.update(b"1;" if bool(value) else b"0;")
    elif isinstance(value, (int, np.integer)):
        digest.update(f"int:{int(value)};".encode())
    elif isinstance(value, (float, np.floating)):
        array = np.asarray(float(value), dtype=np.float64)
        digest.update(b"float64:")
        digest.update(array.tobytes())
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode())
        digest.update(encoded)
    elif isinstance(value, bytes):
        digest.update(f"bytes:{len(value)}:".encode())
        digest.update(value)
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(f"array:{array.dtype.str}:{array.shape}:".encode())
        if array.dtype.hasobject:
            for item in array.flat:
                _update_digest(digest, item, seen)
        else:
            digest.update(array.tobytes())
    elif isinstance(value, np.generic):
        _update_digest(digest, value.item(), seen)
    elif isinstance(value, dict):
        digest.update(b"dict{")
        for key in sorted(value, key=lambda item: str(item)):
            _update_digest(digest, str(key), seen)
            _update_digest(digest, value[key], seen)
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(f"seq:{type(value).__name__}:{len(value)}[".encode())
        for item in value:
            _update_digest(digest, item, seen)
        digest.update(b"]")
    elif isinstance(value, (set, frozenset)):
        pieces = []
        for item in value:
            item_digest = hashlib.sha256()
            _update_digest(item_digest, item, set())
            pieces.append(item_digest.digest())
        digest.update(b"set[")
        for piece in sorted(pieces):
            digest.update(piece)
        digest.update(b"]")
    elif callable(value):
        _update_digest(
            digest,
            (
                "callable",
                getattr(value, "__module__", ""),
                getattr(value, "__qualname__", type(value).__qualname__),
            ),
            seen,
        )
    elif hasattr(value, "__dict__"):
        object_id = id(value)
        if object_id in seen:
            digest.update(b"cycle;")
            return
        seen.add(object_id)
        _update_digest(
            digest,
            (
                f"object:{type(value).__module__}.{type(value).__qualname__}",
                value.__dict__,
            ),
            seen,
        )
        seen.remove(object_id)
    else:
        _update_digest(
            digest,
            (
                f"opaque:{type(value).__module__}.{type(value).__qualname__}",
                repr(value),
            ),
            seen,
        )


def _digest(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _transformer_state(transformer: Any) -> dict[str, Any]:
    return {
        "D": int(np.size(transformer.lb_orig)),
        "lb_orig": transformer.lb_orig,
        "ub_orig": transformer.ub_orig,
        "mu": transformer.mu,
        "delta": transformer.delta,
        "type": transformer.type,
        "bounded_types": transformer.bounded_types,
        "R_mat": transformer.R_mat,
        "scale": transformer.scale,
    }


def _vp_state(vp: Any) -> dict[str, Any]:
    return {
        "D": vp.D,
        "K": vp.K,
        "w": vp.w,
        "eta": vp.eta,
        "mu": vp.mu,
        "sigma": vp.sigma,
        "lambd": vp.lambd,
        "bounds": vp.bounds,
        "stats": vp.stats,
        "optimize_mu": vp.optimize_mu,
        "optimize_sigma": vp.optimize_sigma,
        "optimize_lambd": vp.optimize_lambd,
        "optimize_weights": vp.optimize_weights,
        "parameters": _peek_parameters(vp),
        "transformer": _transformer_state(vp.parameter_transformer),
    }


def _peek_parameters(vp: Any) -> np.ndarray:
    """Return what ``get_parameters`` would return without mutating ``vp``."""
    norm = np.sqrt(np.sum(vp.lambd**2) / vp.D)
    lambd = np.asarray(vp.lambd).reshape(-1, 1) / norm
    sigma = np.asarray(vp.sigma).reshape(1, -1) * norm
    pieces = []
    if vp.optimize_mu:
        pieces.append(np.asarray(vp.mu).ravel(order="F"))
    if vp.optimize_sigma:
        pieces.append(np.log(sigma.ravel()))
    if vp.optimize_lambd:
        pieces.append(np.log(lambd.ravel()))
    if vp.optimize_weights:
        weights = np.asarray(vp.w).reshape(1, -1)
        weights = weights / np.sum(weights)
        pieces.append(np.log(weights.ravel()))
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=float)


def _gp_state(gp: Any) -> dict[str, Any]:
    posterior_fields = ("hyp", "alpha", "L", "L_chol", "sW")
    return {
        "D": gp.D,
        "X": gp.X,
        "y": gp.y,
        "s2": gp.s2,
        "covariance": gp.covariance,
        "mean": gp.mean,
        "noise": gp.noise,
        "posteriors": [
            {name: getattr(posterior, name, None) for name in posterior_fields}
            for posterior in gp.posteriors
        ],
    }


def _options_state(options: Any) -> dict[str, Any]:
    return {str(key): options[key] for key in options}


def _input_hashes(vp: Any, gp: Any, optim_state: dict, options: Any) -> dict:
    return {
        "vp": _digest(_vp_state(vp)),
        "gp": _digest(_gp_state(gp)),
        "optim_state": _digest(optim_state),
        "options": _digest(_options_state(options)),
    }


def _sieve_hashes(sieve_result: tuple) -> dict[str, Any]:
    candidates, candidate_types, *tail = sieve_result
    candidate_hashes = [_digest(_vp_state(vp)) for vp in candidates]
    start_theta_hashes = [_digest(_peek_parameters(vp)) for vp in candidates]
    return {
        "bundle": _digest(
            {
                "candidates": [_vp_state(vp) for vp in candidates],
                "types": candidate_types,
                "tail": tail,
            }
        ),
        "candidate_vp": candidate_hashes,
        "start_theta": start_theta_hashes,
    }


def _clone_rng(state: dict) -> np.random.Generator:
    generator = np.random.default_rng()
    generator.bit_generator.state = copy.deepcopy(state)
    return generator


def _bind_rng(vp: Any, rng: np.random.Generator) -> None:
    # VariationalPosterior.__deepcopy__ intentionally shares the generator.
    # Direct rebinding is required for independent paired arms.
    vp._rng = rng


def _clone_vp(vp: Any, rng: np.random.Generator) -> Any:
    cloned = copy.deepcopy(vp)
    _bind_rng(cloned, rng)
    return cloned


def _clone_sieve_result(
    sieve_result: tuple, rng: np.random.Generator
) -> tuple:
    candidates, candidate_types, *tail = sieve_result
    cloned_candidates = np.empty(candidates.shape, dtype=object)
    for index, candidate in np.ndenumerate(candidates):
        cloned_candidates[index] = _clone_vp(candidate, rng)
    return (
        cloned_candidates,
        np.array(candidate_types, copy=True),
        *(copy.deepcopy(item) for item in tail),
    )


def _apply_option_overrides(options: Any, overrides: dict) -> None:
    for key, value in overrides.items():
        if key not in options:
            raise ValueError(f"unknown option override {key!r}")
        options.__setitem__(key, value, force=True)


def _validate_source_hashes(state_spec: dict) -> None:
    expected = state_spec.get("source_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError(
            f"state {state_spec.get('label')!r} has no source_sha256 map"
        )
    for relative, wanted in expected.items():
        path = _resolved_repo_path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _sha256_file(path)
        if observed != wanted:
            raise ValueError(
                f"source hash mismatch for {relative}: {observed} != {wanted}"
            )


def _load_source_state(
    state_spec: dict, rng: np.random.Generator
) -> dict[str, Any]:
    _validate_source_hashes(state_spec)
    source_kind = state_spec["source_kind"]
    source_path = _resolved_repo_path(state_spec["path"])
    if source_kind == "oracle":
        rebuilt = build_state(load_snapshot(source_path), rng=rng)
        vp = rebuilt["vp"]
        gp = rebuilt["gp"]
        optim_state = rebuilt["optim_state"]
        options = rebuilt["options"]
        source_metadata = rebuilt["meta"]
    elif source_kind == "boost_pre":
        captured = _load_dill(source_path)
        required = {"vp", "gp", "optim_state", "options"}
        missing = required.difference(captured)
        if missing:
            raise ValueError(
                f"boost capture {source_path} lacks {sorted(missing)}"
            )
        # Preserve the authentic, already factorized GP.  In particular, do
        # not rebuild it from rounded trace data.
        vp = copy.deepcopy(captured["vp"])
        gp = copy.deepcopy(captured["gp"])
        optim_state = copy.deepcopy(captured["optim_state"])
        options = copy.deepcopy(captured["options"])
        source_metadata = {
            key: copy.deepcopy(captured.get(key))
            for key in ("label", "seed", "provenance", "rng_state")
        }
    else:
        raise ValueError(
            f"unknown source_kind {source_kind!r} for {state_spec['label']}"
        )
    _bind_rng(vp, rng)
    _apply_option_overrides(options, state_spec.get("options", {}))
    coverage = state_spec.get("coverage", {})
    checks = {
        "D": int(vp.D),
        "K": int(vp.K),
        "N_gp": int(gp.X.shape[0]),
        "Ns_gp": int(len(gp.posteriors)),
        "warmup": bool(optim_state["warmup"]),
        "optimize_weights": bool(vp.optimize_weights),
    }
    for key, observed in checks.items():
        if key in coverage and coverage[key] != observed:
            raise ValueError(
                f"{state_spec['label']} coverage {key}={coverage[key]!r}, "
                f"rebuilt {observed!r}"
            )
    if int(state_spec["slow_opts_N"]) != 1:
        raise ValueError("Phase 6 manifest must use one slow optimizer start")
    return {
        "vp": vp,
        "gp": gp,
        "optim_state": optim_state,
        "options": options,
        "source_metadata": source_metadata,
    }


def _load_manifest(path: Path) -> tuple[dict, str]:
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manifest version {manifest.get('version')!r}"
        )
    states = manifest.get("states")
    if not isinstance(states, list) or not 1 <= len(states) <= MAX_STATES:
        raise ValueError(f"manifest must contain 1..{MAX_STATES} states")
    labels = []
    for state in states:
        required = {
            "label",
            "source_kind",
            "path",
            "options",
            "fast_opts_N",
            "slow_opts_N",
            "source_sha256",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(
                f"manifest state lacks required keys {sorted(missing)}"
            )
        label = state["label"]
        if not isinstance(label, str) or not _LABEL_RE.fullmatch(label):
            raise ValueError(f"unsafe state label {label!r}")
        labels.append(label)
        if state["source_kind"] not in {"oracle", "boost_pre"}:
            raise ValueError(f"unsupported source_kind in {label}")
        if not isinstance(state["options"], dict):
            raise ValueError(f"options for {label} must be a JSON object")
        if int(state["fast_opts_N"]) < 1:
            raise ValueError(f"fast_opts_N for {label} must be positive")
    if len(labels) != len(set(labels)):
        raise ValueError("manifest state labels are not unique")
    for key in ("seed_base", "diagnostic_seed_base", "diagnostic_samples"):
        if not isinstance(manifest.get(key), int) or manifest[key] < 1:
            raise ValueError(f"manifest {key} must be a positive integer")
    return manifest, _sha256_bytes(manifest_bytes)


def _module_record(module: Any) -> dict[str, Any]:
    path_text = inspect.getsourcefile(module)
    path = Path(path_text).resolve() if path_text else None
    return {
        "name": module.__name__,
        "path": str(path) if path else None,
        "sha256": _sha256_file(path) if path and path.is_file() else None,
        "version": getattr(module, "__version__", None),
    }


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _configuration(
    manifest_path: Path, manifest: dict, manifest_hash: str
) -> dict[str, Any]:
    import eta_bound_variants

    import pyvbmc.testing.oracles._state as oracle_state
    import pyvbmc.vbmc.minimize_adam as adam_module

    modules = {
        name: _module_record(module)
        for name, module in {
            "runner": sys.modules[__name__],
            "eta_bound_variants": eta_bound_variants,
            "variational_optimization": baseline_vo,
            "minimize_adam": adam_module,
            "oracle_state": oracle_state,
            "pyvbmc": pyvbmc,
            "gpyreg": gpyreg,
            "numpy": np,
            "scipy": scipy,
            "dill": dill,
        }.items()
    }
    config = {
        "schema": "pyvbmc-eta-bound-comparison-config-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "manifest_protocol": manifest.get("protocol"),
        "source_base_commit": manifest.get("source_base_commit"),
        "git_head": _git_head(),
        "arms": list(ARMS),
        "arm_labels": ARM_LABELS,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "modules": modules,
    }
    config["config_sha256"] = _sha256_bytes(_canonical_json(config))
    return config


def _prepare_output(out: Path, config: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != _jsonable(config):
            raise ValueError(
                f"output configuration differs from existing {path}; "
                "use a different output directory"
            )
    else:
        _atomic_json(path, config)


def _replicate_rng(seed_base: int, state_index: int, replicate: int):
    sequence = np.random.SeedSequence([seed_base, state_index, replicate])
    return np.random.default_rng(sequence), {
        "algorithm": "numpy.random.default_rng(SeedSequence)",
        "entropy_words": [seed_base, state_index, replicate],
        "state_index": state_index,
        "replicate": replicate,
    }


def _theta_bound_summary(theta: np.ndarray, vp: Any, bounds: dict) -> dict:
    if not vp.optimize_weights:
        return {
            "optimize_weights": False,
            "raw_eta": [],
            "lower_count": 0,
            "upper_count": 0,
        }
    raw_eta = np.asarray(theta[-vp.K :], dtype=float)
    lower = np.asarray(bounds["lb"][-vp.K :], dtype=float).reshape(-1)
    upper = np.asarray(bounds["ub"][-vp.K :], dtype=float).reshape(-1)
    lower_violation = np.maximum(lower - raw_eta, 0.0)
    upper_violation = np.maximum(raw_eta - upper, 0.0)
    relative = raw_eta - np.max(raw_eta)
    return {
        "optimize_weights": True,
        "raw_eta": raw_eta,
        "relative_eta": relative,
        "raw_min": float(np.min(raw_eta)),
        "raw_max": float(np.max(raw_eta)),
        "raw_mean": float(np.mean(raw_eta)),
        "raw_range": float(np.ptp(raw_eta)),
        "relative_min": float(np.min(relative)),
        "lower_count": int(np.count_nonzero(lower_violation)),
        "upper_count": int(np.count_nonzero(upper_violation)),
        "lower_max": float(np.max(lower_violation, initial=0.0)),
        "upper_max": float(np.max(upper_violation, initial=0.0)),
    }


def _optimizer_settings(
    options: Any, optim_state: dict, vp: Any, state_spec: dict
) -> dict[str, Any]:
    K = vp.K
    keys = (
        "stochastic_optimizer",
        "max_iter_stochastic",
        "tol_fun_stochastic",
        "sgd_step_size",
        "det_entropy_tol_opt",
        "elcbo_midpoint",
        "tol_weight",
        "weight_penalty",
        "tol_improvement",
        "elcbo_impro_weight",
        "skip_elbo_variance",
    )
    evaluated = {}
    for key in ("ns_elbo", "ns_ent", "ns_ent_fast", "ns_ent_fine"):
        evaluated[key] = options.eval(key, {"K": K})
    return {
        "K": int(K),
        "fast_opts_N": int(state_spec["fast_opts_N"]),
        "slow_opts_N": int(state_spec["slow_opts_N"]),
        "warmup": bool(optim_state["warmup"]),
        "entropy_switch": bool(optim_state["entropy_switch"]),
        "values": {key: options[key] for key in keys if key in options},
        "evaluated": evaluated,
        "all_options": _options_state(options),
    }


def _make_pair_input(
    state_spec: dict,
    state_index: int,
    replicate: int,
    manifest: dict,
    config: dict,
) -> dict[str, Any]:
    rng, seed_record = _replicate_rng(
        manifest["seed_base"], state_index, replicate
    )
    initial_rng_state = copy.deepcopy(rng.bit_generator.state)
    state = _load_source_state(state_spec, rng)
    vp = state["vp"]
    gp = state["gp"]
    optim_state = state["optim_state"]
    options = state["options"]

    # Rebuilding an oracle VP invokes its public constructor, which may draw
    # while creating state that is immediately overwritten by the snapshot.
    # Those reconstruction draws are not part of the experiment stream.
    rng.bit_generator.state = copy.deepcopy(initial_rng_state)
    _bind_rng(vp, rng)

    # Match optimize_vp's only pre-sieve state change.
    if optim_state["warmup"]:
        vp.optimize_weights = False

    pre_vp = _clone_vp(vp, _clone_rng(initial_rng_state))
    pre_hashes = _input_hashes(vp, gp, optim_state, options)
    sieve_result = baseline_vo._sieve(
        options,
        optim_state,
        vp,
        gp,
        K=vp.K,
        init_N=int(state_spec["fast_opts_N"]),
        best_N=int(state_spec["slow_opts_N"]),
    )
    post_sieve_rng_state = copy.deepcopy(rng.bit_generator.state)
    post_sieve_hashes = _input_hashes(vp, gp, optim_state, options)
    sieve_hashes = _sieve_hashes(sieve_result)
    bounds_vp = _clone_vp(vp, _clone_rng(post_sieve_rng_state))
    theta_bounds = bounds_vp.get_bounds(gp.X, options, vp.K)
    candidate_starts = []
    for index, candidate in enumerate(sieve_result[0]):
        theta = _peek_parameters(candidate)
        candidate_starts.append(
            {
                "candidate_index": index,
                "candidate_type": int(sieve_result[1][index]),
                "theta": theta,
                "theta_sha256": _digest(theta),
                "eta_bounds": _theta_bound_summary(
                    theta, candidate, theta_bounds
                ),
            }
        )
    return {
        "version": PAIR_INPUT_VERSION,
        "config_sha256": config["config_sha256"],
        "state_label": state_spec["label"],
        "state_index": state_index,
        "replicate": replicate,
        "state_spec": copy.deepcopy(state_spec),
        "source_metadata": state["source_metadata"],
        "seed": seed_record,
        "initial_rng_state": initial_rng_state,
        "post_sieve_rng_state": post_sieve_rng_state,
        "pre_vp": pre_vp,
        "post_sieve_vp": vp,
        "gp": gp,
        "optim_state": optim_state,
        "options": options,
        "sieve_result": sieve_result,
        "pre_hashes": pre_hashes,
        "post_sieve_hashes": post_sieve_hashes,
        "sieve_hashes": sieve_hashes,
        "candidate_starts": candidate_starts,
        "optimizer_settings": _optimizer_settings(
            options, optim_state, vp, state_spec
        ),
    }


def _pair_input_report(pair_input: dict, dill_path: Path) -> dict[str, Any]:
    return {
        "schema": "pyvbmc-eta-bound-pair-input-v1",
        "config_sha256": pair_input["config_sha256"],
        "state_label": pair_input["state_label"],
        "state_index": pair_input["state_index"],
        "replicate": pair_input["replicate"],
        "source_kind": pair_input["state_spec"]["source_kind"],
        "source_path": pair_input["state_spec"]["path"],
        "source_sha256": pair_input["state_spec"]["source_sha256"],
        "source_metadata": pair_input["source_metadata"],
        "seed": pair_input["seed"],
        "initial_rng_sha256": _digest(pair_input["initial_rng_state"]),
        "post_sieve_rng_sha256": _digest(pair_input["post_sieve_rng_state"]),
        "pre_hashes": pair_input["pre_hashes"],
        "post_sieve_hashes": pair_input["post_sieve_hashes"],
        "sieve_hashes": pair_input["sieve_hashes"],
        "candidate_starts": pair_input["candidate_starts"],
        "optimizer_settings": pair_input["optimizer_settings"],
        "input_dill": dill_path.name,
        "input_dill_sha256": _sha256_file(dill_path),
    }


def _load_or_create_pair_input(
    pair_dir: Path,
    state_spec: dict,
    state_index: int,
    replicate: int,
    manifest: dict,
    config: dict,
) -> dict[str, Any]:
    dill_path = pair_dir / "input.dill"
    json_path = pair_dir / "input.json"
    if dill_path.exists():
        pair_input = _load_dill(dill_path)
        expected = (
            PAIR_INPUT_VERSION,
            config["config_sha256"],
            state_spec["label"],
            state_index,
            replicate,
        )
        observed = (
            pair_input.get("version"),
            pair_input.get("config_sha256"),
            pair_input.get("state_label"),
            pair_input.get("state_index"),
            pair_input.get("replicate"),
        )
        if observed != expected:
            raise ValueError(f"incompatible saved pair input in {dill_path}")
        if pair_input["sieve_hashes"] != _sieve_hashes(
            pair_input["sieve_result"]
        ):
            raise ValueError(f"saved candidate pool changed in {dill_path}")
    else:
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_input = _make_pair_input(
            state_spec, state_index, replicate, manifest, config
        )
        _atomic_dill(dill_path, pair_input)
    report = _pair_input_report(pair_input, dill_path)
    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if existing != _jsonable(report):
            raise ValueError(f"saved pair input report changed in {json_path}")
    else:
        _atomic_json(json_path, report)
    return pair_input


def _as_array(value: Any, like: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if like is not None and array.size == 0:
        return np.zeros_like(like, dtype=float)
    return array


class _ObjectiveTracker:
    """Capture objective calls already made by the original optimizer."""

    def __init__(self, arm: str):
        self.arm = arm
        self.phase = "full_evaluation"
        self.records: list[dict[str, Any]] = []
        self.optimizer_events: list[dict[str, Any]] = []
        self.sieve_calls = 0
        self.sieve_supplied_hashes: dict[str, Any] | None = None

    def record(
        self,
        theta_before: np.ndarray,
        theta_after: np.ndarray,
        result: tuple,
        vp: Any,
        theta_bnd: dict | None,
        terms: dict | None,
    ) -> None:
        objective = float(np.asarray(result[0]))
        gradient = None
        if len(result) > 1 and result[1] is not None:
            gradient = _as_array(result[1])
        item: dict[str, Any] = {
            "phase": self.phase,
            "objective": objective,
            "theta_mutation_norm": float(
                np.linalg.norm(theta_after - theta_before)
            ),
            "theta_mutated": not np.array_equal(theta_before, theta_after),
            "theta_offset": (
                float(np.max(theta_before[-vp.K :]))
                if vp.optimize_weights
                else 0.0
            ),
            "eta_min": (
                float(np.min(theta_before[-vp.K :]))
                if vp.optimize_weights
                else 0.0
            ),
            "eta_max": (
                float(np.max(theta_before[-vp.K :]))
                if vp.optimize_weights
                else 0.0
            ),
            "eta_range": (
                float(np.ptp(theta_before[-vp.K :]))
                if vp.optimize_weights
                else 0.0
            ),
            "has_bounds": theta_bnd is not None,
        }
        if gradient is not None:
            item["total_gradient_norm"] = float(np.linalg.norm(gradient))
        if terms is not None:
            supplied_bound = _as_array(
                terms.get("bound_gradient", []), like=gradient
            )
            correct_bound = _as_array(
                terms.get("correct_bound_gradient", supplied_bound),
                like=gradient,
            )
            small_weight = _as_array(
                terms.get("small_weight_gradient", []), like=gradient
            )
            supplied_regularizer = supplied_bound + small_weight
            correct_regularizer = correct_bound + small_weight
            inconsistency_norm = float(
                np.linalg.norm(supplied_regularizer - correct_regularizer)
            )
            item.update(
                {
                    "bound_loss": float(terms.get("bound_loss", 0.0)),
                    "non_eta_loss": float(terms.get("non_eta_loss", 0.0)),
                    "eta_loss": float(terms.get("eta_loss", 0.0)),
                    "small_weight_loss": float(
                        terms.get("small_weight_loss", 0.0)
                    ),
                    "supplied_bound_gradient_norm": float(
                        np.linalg.norm(supplied_bound)
                    ),
                    "correct_bound_gradient_norm": float(
                        np.linalg.norm(correct_bound)
                    ),
                    "small_weight_gradient_norm": float(
                        np.linalg.norm(small_weight)
                    ),
                    "regularizer_gradient_norm": float(
                        np.linalg.norm(supplied_regularizer)
                    ),
                    "correct_regularizer_gradient_norm": float(
                        np.linalg.norm(correct_regularizer)
                    ),
                    "gradient_inconsistency_norm": inconsistency_norm,
                    "correct_gradient_defined": terms[
                        "correct_gradient_defined"
                    ],
                }
            )
            if gradient is not None:
                elbo_gradient = gradient - supplied_regularizer
                elbo_norm = float(np.linalg.norm(elbo_gradient))
                item["elbo_gradient_norm"] = elbo_norm
                item["regularizer_to_elbo_gradient_ratio"] = (
                    float(np.linalg.norm(supplied_regularizer)) / elbo_norm
                    if elbo_norm > 0
                    else None
                )
                item["correct_regularizer_to_elbo_gradient_ratio"] = (
                    float(np.linalg.norm(correct_regularizer)) / elbo_norm
                    if elbo_norm > 0
                    else None
                )
                item["gradient_inconsistency_to_elbo_gradient_ratio"] = (
                    inconsistency_norm / elbo_norm if elbo_norm > 0 else None
                )
                item["unpenalized_objective"] = (
                    objective - item["bound_loss"] - item["small_weight_loss"]
                )
            for key in (
                "lower_violation_count",
                "upper_violation_count",
                "lower_violation_max",
                "upper_violation_max",
            ):
                if key in terms:
                    item[key] = terms[key]
            # Accept the variants helper's array names without putting the
            # arrays themselves in JSON/NPZ traces.
            for prefix, key in (
                ("lower", "eta_lower_violation"),
                ("upper", "eta_upper_violation"),
            ):
                if key in terms:
                    values = _as_array(terms[key])
                    item.setdefault(
                        f"{prefix}_violation_count",
                        int(np.count_nonzero(values)),
                    )
                    item.setdefault(
                        f"{prefix}_violation_max",
                        float(np.max(values, initial=0.0)),
                    )
        self.records.append(item)

    def arrays(self) -> dict[str, np.ndarray]:
        keys = sorted({key for item in self.records for key in item})
        arrays: dict[str, np.ndarray] = {}
        for key in keys:
            values = [item.get(key) for item in self.records]
            if key == "phase":
                arrays[key] = np.asarray(values, dtype="U32")
            elif all(
                value is None or isinstance(value, (bool, np.bool_))
                for value in values
            ):
                arrays[key] = np.asarray(
                    [False if value is None else value for value in values],
                    dtype=bool,
                )
            else:
                arrays[key] = np.asarray(
                    [np.nan if value is None else value for value in values],
                    dtype=float,
                )
        return arrays

    def summary(self) -> dict[str, Any]:
        optimizer_records = [
            item
            for item in self.records
            if item["phase"].startswith("optimizer")
        ]
        return {
            "objective_calls": len(self.records),
            "optimizer_objective_calls": len(optimizer_records),
            "sieve_calls": self.sieve_calls,
            "sieve_supplied_hashes": self.sieve_supplied_hashes,
            "optimizer_events": self.optimizer_events,
            "optimizer_first_objective": (
                optimizer_records[0]["objective"]
                if optimizer_records
                else None
            ),
            "optimizer_last_objective": (
                optimizer_records[-1]["objective"]
                if optimizer_records
                else None
            ),
            "optimizer_best_objective": min(
                (item["objective"] for item in optimizer_records),
                default=None,
            ),
            "calls_with_lower_eta_violation": sum(
                item.get("lower_violation_count", 0) > 0
                for item in optimizer_records
            ),
            "calls_with_upper_eta_violation": sum(
                item.get("upper_violation_count", 0) > 0
                for item in optimizer_records
            ),
            "max_lower_eta_violation": max(
                (
                    item.get("lower_violation_max", 0.0)
                    for item in optimizer_records
                ),
                default=0.0,
            ),
            "max_upper_eta_violation": max(
                (
                    item.get("upper_violation_max", 0.0)
                    for item in optimizer_records
                ),
                default=0.0,
            ),
            "theta_mutation_calls": sum(
                item["theta_mutated"] for item in optimizer_records
            ),
            "calls_with_active_eta_penalty": sum(
                item.get("eta_loss", 0.0) > 0.0 for item in optimizer_records
            ),
            "violation_semantics": (
                "A reports counterfactual raw-eta distances although its "
                "eta penalty is disabled; B distances are raw eta; C "
                "distances are max-relative eta. Active incidence is "
                "calls_with_active_eta_penalty."
            ),
        }


def _scipy_result_summary(result: Any) -> dict[str, Any]:
    return {
        key: _jsonable(getattr(result, key, None))
        for key in (
            "success",
            "status",
            "message",
            "fun",
            "nit",
            "nfev",
            "njev",
            "maxcv",
        )
    }


def _instrument_variant(
    variant: types.ModuleType,
    arm: str,
    pair_input: dict,
    arm_rng: np.random.Generator,
    tracker: _ObjectiveTracker,
) -> Any:
    expected_input = pair_input["post_sieve_hashes"]
    expected_sieve = pair_input["sieve_hashes"]

    def supplied_sieve(options, optim_state, vp, gp, *args, **kwargs):
        tracker.sieve_calls += 1
        if tracker.sieve_calls != 1:
            raise RuntimeError("optimize_vp called the supplied sieve twice")
        observed_input = _input_hashes(vp, gp, optim_state, options)
        if observed_input != expected_input:
            raise RuntimeError(
                f"arm {arm} optimizer input differs before shared sieve"
            )
        if _digest(arm_rng.bit_generator.state) != _digest(
            pair_input["post_sieve_rng_state"]
        ):
            raise RuntimeError(f"arm {arm} RNG differs before shared sieve")
        supplied = _clone_sieve_result(pair_input["sieve_result"], arm_rng)
        observed_sieve = _sieve_hashes(supplied)
        if observed_sieve != expected_sieve:
            raise RuntimeError(f"arm {arm} candidate pool differs")
        tracker.sieve_supplied_hashes = observed_sieve
        return supplied

    original_sieve = variant._sieve
    variant._sieve = supplied_sieve

    original_neg_elcbo = variant._neg_elcbo

    def tracked_neg_elcbo(theta, gp, vp, *args, **kwargs):
        theta_before = np.asarray(theta).copy()
        bound = kwargs.get("theta_bnd")
        if len(args) >= 5:
            bound = args[4]
        terms = None
        if bound is not None:
            terms = penalty_terms(theta_before, bound, vp, arm)
        result = original_neg_elcbo(theta, gp, vp, *args, **kwargs)
        tracker.record(
            theta_before,
            np.asarray(theta).copy(),
            result,
            vp,
            bound,
            terms,
        )
        return result

    variant._neg_elcbo = tracked_neg_elcbo

    original_adam = variant.minimize_adam

    def tracked_adam(*args, **kwargs):
        previous = tracker.phase
        tracker.phase = "optimizer_adam"
        started = time.perf_counter()
        try:
            result = original_adam(*args, **kwargs)
        finally:
            tracker.phase = previous
        iterations = int(result[-1])
        max_iter = int(kwargs.get("max_iter", 10000))
        tracker.optimizer_events.append(
            {
                "kind": "adam",
                "iterations": iterations,
                "max_iter": max_iter,
                "stop_reason": (
                    "early_stopping"
                    if iterations < max_iter
                    else "budget_reached_or_last_iteration_stop"
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return result

    variant.minimize_adam = tracked_adam

    original_sp = variant.sp
    original_minimize = original_sp.optimize.minimize

    def tracked_minimize(*args, **kwargs):
        previous = tracker.phase
        tracker.phase = "optimizer_scipy"
        started = time.perf_counter()
        try:
            result = original_minimize(*args, **kwargs)
        finally:
            tracker.phase = previous
        tracker.optimizer_events.append(
            {
                "kind": "scipy",
                "elapsed_seconds": time.perf_counter() - started,
                "result": _scipy_result_summary(result),
                "x": np.asarray(result.x).copy(),
            }
        )
        return result

    variant.sp = types.SimpleNamespace(
        linalg=original_sp.linalg,
        optimize=types.SimpleNamespace(minimize=tracked_minimize),
    )

    def restore() -> None:
        variant._sieve = original_sieve
        variant._neg_elcbo = original_neg_elcbo
        variant.minimize_adam = original_adam
        variant.sp = original_sp

    return restore


def _weight_summary(vp: Any, tol_weight: float) -> dict[str, Any]:
    weights = np.asarray(vp.w, dtype=float).reshape(-1)
    return {
        "K": int(vp.K),
        "weights": weights,
        "minimum": float(np.min(weights)),
        "maximum": float(np.max(weights)),
        "quantiles": np.quantile(weights, [0.0, 0.25, 0.5, 0.75, 1.0]),
        "below_tol_weight": int(np.count_nonzero(weights < tol_weight)),
        "effective_components": float(1.0 / np.sum(weights**2)),
    }


def _arm_report(capture: dict, dill_path: Path, trace_path: Path) -> dict:
    return {
        "schema": "pyvbmc-eta-bound-arm-v1",
        "config_sha256": capture["config_sha256"],
        "state_label": capture["state_label"],
        "replicate": capture["replicate"],
        "arm": capture["arm"],
        "arm_label": capture["arm_label"],
        "pair_input_sha256": capture["pair_input_sha256"],
        "input_hashes": capture["input_hashes"],
        "sieve_hashes": capture["sieve_hashes"],
        "rng_before_sha256": _digest(capture["rng_before"]),
        "rng_after_sha256": _digest(capture["rng_after"]),
        "elapsed_seconds": capture["elapsed_seconds"],
        "var_ss": capture["var_ss"],
        "pruned": capture["pruned"],
        "K_before": capture["K_before"],
        "K_after": capture["K_after"],
        "weights": capture["weights"],
        "tracker": capture["tracker_summary"],
        "capture_dill": dill_path.name,
        "capture_dill_sha256": _sha256_file(dill_path),
        "trace_npz": trace_path.name,
        "trace_npz_sha256": _sha256_file(trace_path),
    }


def _run_arm(
    arm: str, pair_input: dict, pair_dir: Path, config: dict
) -> dict[str, Any]:
    dill_path = pair_dir / f"arm_{arm}.dill"
    json_path = pair_dir / f"arm_{arm}.json"
    trace_path = pair_dir / f"arm_{arm}_trace.npz"
    pair_input_sha = _sha256_file(pair_dir / "input.dill")
    if dill_path.exists():
        capture = _load_dill(dill_path)
        expected = (
            ARM_CAPTURE_VERSION,
            config["config_sha256"],
            pair_input["state_label"],
            pair_input["replicate"],
            arm,
            pair_input_sha,
        )
        observed = (
            capture.get("version"),
            capture.get("config_sha256"),
            capture.get("state_label"),
            capture.get("replicate"),
            capture.get("arm"),
            capture.get("pair_input_sha256"),
        )
        if observed != expected:
            raise ValueError(f"incompatible saved arm in {dill_path}")
    else:
        arm_rng = _clone_rng(pair_input["post_sieve_rng_state"])
        vp = _clone_vp(pair_input["post_sieve_vp"], arm_rng)
        gp = copy.deepcopy(pair_input["gp"])
        optim_state = copy.deepcopy(pair_input["optim_state"])
        options = copy.deepcopy(pair_input["options"])
        input_hashes = _input_hashes(vp, gp, optim_state, options)
        if input_hashes != pair_input["post_sieve_hashes"]:
            raise RuntimeError(f"arm {arm} input copy differs from pair input")
        tracker = _ObjectiveTracker(arm)
        variant = get_variant(arm)
        restore_variant = _instrument_variant(
            variant, arm, pair_input, arm_rng, tracker
        )
        rng_before = copy.deepcopy(arm_rng.bit_generator.state)
        started = time.perf_counter()
        try:
            output_vp, var_ss, pruned = variant.optimize_vp(
                options,
                optim_state,
                vp,
                gp,
                int(pair_input["state_spec"]["fast_opts_N"]),
                int(pair_input["state_spec"]["slow_opts_N"]),
                vp.K,
            )
        finally:
            restore_variant()
        elapsed = time.perf_counter() - started
        if tracker.sieve_calls != 1:
            raise RuntimeError(f"arm {arm} did not consume shared sieve once")
        if tracker.sieve_supplied_hashes != pair_input["sieve_hashes"]:
            raise RuntimeError(f"arm {arm} did not receive exact candidates")
        trace_arrays = tracker.arrays()
        capture = {
            "version": ARM_CAPTURE_VERSION,
            "config_sha256": config["config_sha256"],
            "state_label": pair_input["state_label"],
            "replicate": pair_input["replicate"],
            "arm": arm,
            "arm_label": ARM_LABELS[arm],
            "pair_input_sha256": pair_input_sha,
            "input_hashes": input_hashes,
            "sieve_hashes": tracker.sieve_supplied_hashes,
            "rng_before": rng_before,
            "rng_after": copy.deepcopy(arm_rng.bit_generator.state),
            "elapsed_seconds": elapsed,
            "output_vp": output_vp,
            "var_ss": var_ss,
            "pruned": int(pruned),
            "K_before": int(vp.K),
            "K_after": int(output_vp.K),
            "weights": _weight_summary(
                output_vp, float(options["tol_weight"])
            ),
            "tracker_records": tracker.records,
            "tracker_summary": tracker.summary(),
            "optimizer_events": tracker.optimizer_events,
        }
        _atomic_npz(trace_path, trace_arrays)
        _atomic_dill(dill_path, capture)
    if not trace_path.exists():
        tracker = _ObjectiveTracker(arm)
        tracker.records = capture["tracker_records"]
        _atomic_npz(trace_path, tracker.arrays())
    report = _arm_report(capture, dill_path, trace_path)
    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if existing != _jsonable(report):
            raise ValueError(f"saved arm report changed in {json_path}")
    else:
        _atomic_json(json_path, report)
    return capture


def _score_vp(
    vp: Any,
    gp: Any,
    diagnostic_rng_state: dict,
    diagnostic_samples_total: int,
) -> dict[str, Any]:
    rng = _clone_rng(diagnostic_rng_state)
    scoring_vp = _clone_vp(vp, rng)
    theta = scoring_vp.get_parameters().copy()
    # entmc_vbmc uses antithetic pairs and rounds an odd request up to even;
    # make the effective draw count explicit in the retained metadata.
    samples_per_component = int(
        2 * math.ceil(diagnostic_samples_total / (2 * scoring_vp.K))
    )
    result = baseline_vo._neg_elcbo(
        theta,
        gp,
        scoring_vp,
        beta=0.0,
        Ns=samples_per_component,
        compute_grad=False,
        compute_var=True,
        theta_bnd=None,
        separate_K=True,
    )
    nelbo = float(result[0])
    variance = float(result[4])
    if not math.isfinite(nelbo) or not math.isfinite(variance) or variance < 0:
        raise ValueError(f"invalid unpenalized score: {nelbo=}, {variance=}")
    elbo_sd = math.sqrt(variance)
    return {
        "K": int(scoring_vp.K),
        "samples_total_requested": diagnostic_samples_total,
        "samples_per_component": samples_per_component,
        "samples_total_actual": samples_per_component * scoring_vp.K,
        "theta_sha256": _digest(theta),
        "rng_before_sha256": _digest(diagnostic_rng_state),
        "rng_after_sha256": _digest(rng.bit_generator.state),
        "elbo": -nelbo,
        "elbo_sd": elbo_sd,
        "elbo_variance": variance,
        "uncertainty_note": "GP variance; current entropy variance is zero",
        "elcbo_beta5": -nelbo - 5.0 * elbo_sd,
        "expected_log_joint": float(result[2]),
        "entropy": float(result[3]),
        "var_ss": result[6],
        "varG": float(result[7]),
        "varH": float(result[8]),
    }


def _score_report(capture: dict, dill_path: Path) -> dict[str, Any]:
    return {
        "schema": "pyvbmc-eta-bound-scores-v1",
        "config_sha256": capture["config_sha256"],
        "state_label": capture["state_label"],
        "replicate": capture["replicate"],
        "diagnostic_seed": capture["diagnostic_seed"],
        "common_draw_limit": capture["common_draw_limit"],
        "scores": capture["scores"],
        "paired_differences": capture["paired_differences"],
        "scores_dill": dill_path.name,
        "scores_dill_sha256": _sha256_file(dill_path),
    }


def _load_or_score(
    pair_input: dict,
    arms: dict[str, dict],
    pair_dir: Path,
    manifest: dict,
    config: dict,
) -> dict[str, Any]:
    dill_path = pair_dir / "scores.dill"
    json_path = pair_dir / "scores.json"
    arm_hashes = {
        arm: _sha256_file(pair_dir / f"arm_{arm}.dill") for arm in ARMS
    }
    if dill_path.exists():
        capture = _load_dill(dill_path)
        expected = (
            SCORE_CAPTURE_VERSION,
            config["config_sha256"],
            pair_input["state_label"],
            pair_input["replicate"],
            arm_hashes,
        )
        observed = (
            capture.get("version"),
            capture.get("config_sha256"),
            capture.get("state_label"),
            capture.get("replicate"),
            capture.get("arm_capture_sha256"),
        )
        if observed != expected:
            raise ValueError(f"incompatible scores in {dill_path}")
    else:
        diag_rng, seed_record = _replicate_rng(
            manifest["diagnostic_seed_base"],
            pair_input["state_index"],
            pair_input["replicate"],
        )
        diagnostic_state = copy.deepcopy(diag_rng.bit_generator.state)
        posteriors = {"pre": pair_input["pre_vp"]}
        posteriors.update({arm: arms[arm]["output_vp"] for arm in ARMS})
        scores = {
            label: _score_vp(
                vp,
                pair_input["gp"],
                diagnostic_state,
                int(manifest["diagnostic_samples"]),
            )
            for label, vp in posteriors.items()
        }
        differences = {}
        for left, right in (
            ("A", "pre"),
            ("B", "pre"),
            ("C", "pre"),
            ("A", "B"),
            ("A", "C"),
            ("B", "C"),
        ):
            differences[f"{left}_minus_{right}"] = {
                key: scores[left][key] - scores[right][key]
                for key in ("elbo", "elbo_sd", "elcbo_beta5")
            }
        k_values = {label: score["K"] for label, score in scores.items()}
        common_limit = {
            "same_initial_generator_state": True,
            "K_by_posterior": k_values,
            "identical_draw_shapes": len(set(k_values.values())) == 1,
            "note": (
                "Resetting the generator gives common entropy draws only "
                "where K (and therefore the draw shape) agrees; pruning can "
                "make cross-posterior draws only prefix-coupled."
            ),
        }
        capture = {
            "version": SCORE_CAPTURE_VERSION,
            "config_sha256": config["config_sha256"],
            "state_label": pair_input["state_label"],
            "replicate": pair_input["replicate"],
            "arm_capture_sha256": arm_hashes,
            "diagnostic_seed": seed_record,
            "diagnostic_rng_state": diagnostic_state,
            "scores": scores,
            "paired_differences": differences,
            "common_draw_limit": common_limit,
        }
        _atomic_dill(dill_path, capture)
    report = _score_report(capture, dill_path)
    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if existing != _jsonable(report):
            raise ValueError(f"saved scores report changed in {json_path}")
    else:
        _atomic_json(json_path, report)
    return capture


def _artifact_hashes(pair_dir: Path) -> dict[str, str]:
    names = ["input.dill", "input.json", "scores.dill", "scores.json"]
    for arm in ARMS:
        names.extend(
            [f"arm_{arm}.dill", f"arm_{arm}.json", f"arm_{arm}_trace.npz"]
        )
    return {name: _sha256_file(pair_dir / name) for name in names}


def _validate_completion(
    path: Path, state_label: str, replicate: int, config: dict
) -> dict:
    completion = json.loads(path.read_text(encoding="utf-8"))
    expected = (
        "pyvbmc-eta-bound-pair-complete-v1",
        config["config_sha256"],
        state_label,
        replicate,
    )
    observed = (
        completion.get("schema"),
        completion.get("config_sha256"),
        completion.get("state_label"),
        completion.get("replicate"),
    )
    if observed != expected:
        raise ValueError(f"incompatible completion marker {path}")
    for name, wanted in completion["artifact_sha256"].items():
        artifact = path.parent / name
        if not artifact.is_file() or _sha256_file(artifact) != wanted:
            raise ValueError(f"completion artifact mismatch: {artifact}")
    return completion


def _record_failure(
    pair_dir: Path,
    arm: str,
    pair_input: dict,
    config: dict,
    error: BaseException,
) -> None:
    failure_path = pair_dir / f"failure_{arm}.json"
    if failure_path.exists():
        return
    _atomic_json(
        failure_path,
        {
            "schema": "pyvbmc-eta-bound-failure-v1",
            "config_sha256": config["config_sha256"],
            "state_label": pair_input["state_label"],
            "replicate": pair_input["replicate"],
            "arm": arm,
            "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "error": str(error),
            "traceback": traceback.format_exc(),
            "recorded_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        },
    )


def _run_pair(
    out: Path,
    state_spec: dict,
    state_index: int,
    replicate: int,
    manifest: dict,
    config: dict,
) -> str:
    pair_dir = out / "pairs" / state_spec["label"] / f"replicate_{replicate}"
    completion_path = pair_dir / "complete.json"
    if completion_path.exists():
        _validate_completion(
            completion_path, state_spec["label"], replicate, config
        )
        return "skipped-complete"
    failures = (
        sorted(pair_dir.glob("failure_*.json")) if pair_dir.exists() else []
    )
    if failures:
        raise RuntimeError(
            "refusing to replace recorded failure(s): "
            + ", ".join(str(path) for path in failures)
        )
    pair_input = _load_or_create_pair_input(
        pair_dir,
        state_spec,
        state_index,
        replicate,
        manifest,
        config,
    )
    arm_captures = {}
    for arm in ARMS:
        try:
            arm_captures[arm] = _run_arm(arm, pair_input, pair_dir, config)
        except BaseException as error:
            _record_failure(pair_dir, arm, pair_input, config, error)
            raise
    scores = _load_or_score(
        pair_input, arm_captures, pair_dir, manifest, config
    )
    completion = {
        "schema": "pyvbmc-eta-bound-pair-complete-v1",
        "config_sha256": config["config_sha256"],
        "state_label": state_spec["label"],
        "state_index": state_index,
        "replicate": replicate,
        "arms": list(ARMS),
        "arm_labels": ARM_LABELS,
        "pruned": {arm: arm_captures[arm]["pruned"] for arm in ARMS},
        "K_after": {arm: arm_captures[arm]["K_after"] for arm in ARMS},
        "paired_differences": scores["paired_differences"],
        "artifact_sha256": _artifact_hashes(pair_dir),
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(completion_path, completion)
    return "completed"


def _parse_replicates(text: str) -> list[int]:
    try:
        values = [
            int(piece.strip()) for piece in text.split(",") if piece.strip()
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "replicates must be comma-separated integers"
        ) from error
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            "replicates must be nonempty and unique"
        )
    if any(value < 0 or value >= MAX_REPLICATES for value in values):
        raise argparse.ArgumentTypeError(
            f"replicates must be in 0..{MAX_REPLICATES - 1}"
        )
    return values


def _parse_states(text: str | None, manifest: dict) -> list[str]:
    available = [state["label"] for state in manifest["states"]]
    if text is None:
        return available
    selected = [piece.strip() for piece in text.split(",") if piece.strip()]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("states must be nonempty and unique")
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"unknown state labels: {unknown}")
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--replicates",
        default="0,1,2",
        help="comma-separated replicate indices (allowed: 0,1,2)",
    )
    parser.add_argument(
        "--states",
        help="comma-separated state labels (default: every manifest state)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    out = args.out.resolve()
    manifest, manifest_hash = _load_manifest(manifest_path)
    replicates = _parse_replicates(args.replicates)
    selected_labels = _parse_states(args.states, manifest)
    config = _configuration(manifest_path, manifest, manifest_hash)
    _prepare_output(out, config)
    state_by_label = {
        state["label"]: (index, state)
        for index, state in enumerate(manifest["states"])
    }
    counts = {"completed": 0, "skipped-complete": 0}
    started = time.perf_counter()
    for label in selected_labels:
        state_index, state_spec = state_by_label[label]
        for replicate in replicates:
            status = _run_pair(
                out,
                state_spec,
                state_index,
                replicate,
                manifest,
                config,
            )
            counts[status] += 1
            print(f"{label} replicate {replicate}: {status}", flush=True)
    print(
        json.dumps(
            {
                "completed": counts["completed"],
                "skipped_complete": counts["skipped-complete"],
                "elapsed_seconds": time.perf_counter() - started,
                "out": str(out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
