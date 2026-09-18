"""Manifest-driven noisy-acquisition experiments.

The capture commands in this module run the unmodified benchmark targets and
observe the public VIQR sieve call.  They store portable state snapshots, the
live GP factors, the public acquisition values, and every relevant random
generator state without making an extra target call or random draw.

Typical use (PowerShell)::

    $env:PYVBMC_GPYREG_SOURCE = "dev/scripts/runs/.../gpyreg_1.2.1"
    python dev/scripts/noisy_acq_experiment.py prepare-manifest --out manifest.json
    # Review the manifest and set launch_ready to true.
    python dev/scripts/noisy_acq_experiment.py capture --manifest manifest.json --out RUN_DIR
    python dev/scripts/noisy_acq_experiment.py inventory --manifest manifest.json --out RUN_DIR

The controller launches one fresh process per configuration/seed pair.  A
case is resumable only through a completion record whose identity and artifact
hashes still validate.  Partial output is never silently skipped.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# Pin BLAS before importing NumPy or PyVBMC.  A caller may inspect the values in
# every manifest/case record; workers reject a manifest prepared differently.
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

import numpy as np

SCHEMA_VERSION = 1
NUMERICAL_BASE = "9cc6882768ff682ae892a6b453c42e0f2d03d5fa"
GPYREG_BASE = "9e70e6ba53f7607d05c2d9cc2fa9f41cd12b8f3b"
LABELS = (
    "rosenbrock_D2_noise1",
    "rosenbrock_D2_noise3",
    "logreg_D5_noise3",
    "student_D8_noise3",
    "multisensory_s1_D6_noise1.3",
    "timing_D5_noise2.2",
)
SEEDS = (0, 1)
FACTOR_FIELDS = ("alpha", "L", "L_chol", "sW", "sn2_mult")
DEPENDENCIES = ("numpy", "scipy", "cma", "pyvbmc", "gpyreg")
CAPTURE_OPTIONS = {
    "display": "off",
    "plot": False,
    "print_iteration_header": False,
    "performance_calibration": "off",
}


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value: Any) -> str:
    """Hash in-memory state without advancing any generator."""

    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def canonical(value: Any) -> Any:
    """Convert provenance to stable, strict-JSON data.

    Callables are identified by module and qualified name.  Their source files
    are separately hashed by the manifest, avoiding unstable object reprs.
    """

    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, np.generic):
        return canonical(value.item())
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(canonical(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "@float": "nan"
            if math.isnan(value)
            else ("inf" if value > 0 else "-inf")
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if callable(value):
        cls = value if isinstance(value, type) else type(value)
        if hasattr(value, "__module__") and hasattr(value, "__qualname__"):
            name = f"{value.__module__}.{value.__qualname__}"
        else:
            name = f"{cls.__module__}.{cls.__qualname__}"
        state = getattr(value, "__dict__", None)
        return {
            "@callable": name,
            "state": canonical(state) if state else None,
        }
    cls = type(value)
    state = getattr(value, "__dict__", None)
    return {
        "@object": f"{cls.__module__}.{cls.__qualname__}",
        "state": canonical(state) if state else None,
    }


def write_json(path: Path | str, value: Any) -> None:
    """Atomically write strict JSON with stable key order."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            canonical(value), stream, indent=2, sort_keys=True, allow_nan=False
        )
        stream.write("\n")
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={repo.resolve()}",
            "-C",
            str(repo),
            *args,
        ],
        text=True,
    ).strip()


def _package_versions() -> dict[str, str | None]:
    values = {}
    for name in DEPENDENCIES:
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = None
    return values


def _hash_paths(paths: list[Path], base: Path) -> dict[str, str]:
    return {
        path.relative_to(base).as_posix(): sha256_file(path)
        for path in sorted(set(paths), key=lambda item: item.as_posix())
    }


def source_hashes() -> dict[str, str]:
    """Hash every numerical or experiment source loaded by capture."""

    package = ROOT / "pyvbmc"
    paths = [
        path
        for pattern in ("*.py", "*.ini")
        for path in package.rglob(pattern)
        if "__pycache__" not in path.parts
        and path.relative_to(package).parts[0] != "testing"
    ]
    for name in (
        "benchmark_targets.py",
        "noisy_acq_experiment.py",
    ):
        path = SCRIPTS / name
        if path.exists():
            paths.append(path)
    paths.append(ROOT / "pyvbmc/testing/oracles/_state.py")
    return _hash_paths(paths, ROOT)


def data_hashes() -> dict[str, str]:
    data = SCRIPTS / "data"
    paths = list(data.glob("*.npz")) + list((data / "truths").glob("*.npz"))
    return _hash_paths(paths, ROOT)


def gpyreg_identity() -> dict[str, Any]:
    import gpyreg

    package = Path(gpyreg.__file__).resolve().parent
    repo = package.parent
    return {
        "import": str(Path(gpyreg.__file__).resolve()),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "source_hashes": _hash_paths(list(package.rglob("*.py")), repo),
    }


def numpy_build() -> dict[str, Any]:
    config = getattr(np.__config__, "CONFIG", None)
    if config is not None:
        return canonical(config)
    # Older NumPy exposes no structured build dictionary.  The absence is
    # explicit; thread environment pins remain the reproducibility contract.
    return {"structured_config_available": False}


def numerical_diff(exclude: tuple[str, ...] = ()) -> str:
    """Return tracked or untracked deviations from the numerical base.

    ``exclude`` lists repository-relative files or directories whose
    deviations are not reported, for a campaign that declares them.
    """

    paths = ("pyvbmc", "dev/scripts/benchmark_targets.py", "dev/scripts/data")
    pathspec = (*paths, *(f":(exclude){path}" for path in exclude))
    content = _git(ROOT, "diff", NUMERICAL_BASE, "--", *pathspec)
    status = _git(
        ROOT,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *pathspec,
    )
    return "\n".join(part for part in (content, status) if part)


def default_manifest() -> dict[str, Any]:
    """Build the reviewable E0 capture manifest for the current process."""

    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise RuntimeError("Set every BLAS thread environment variable to 1")
    gp = gpyreg_identity()
    if numerical_diff():
        raise RuntimeError(
            "Numerical source differs from the frozen 9cc6882 baseline"
        )
    if gp["commit"] != GPYREG_BASE:
        raise RuntimeError(f"gpyreg is {gp['commit']}, expected {GPYREG_BASE}")
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "noisy-acquisition E0 state capture",
        "launch_ready": False,
        "numerical_base_commit": NUMERICAL_BASE,
        "prepared_from_commit": _git(ROOT, "rev-parse", "HEAD"),
        "gpyreg": gp,
        "allocation": [
            {"label": label, "seeds": list(SEEDS)} for label in LABELS
        ],
        "case_count": len(LABELS) * len(SEEDS),
        "state_target_count": 2 * len(LABELS) * len(SEEDS),
        "checkpoint_policy": {
            "early_charged_evaluations": "max(20, 2*D)",
            "late_budget_fraction": 0.6,
            "late_fallback": "last distinct acquisition checkpoint on early termination",
            "candidate_count": 8192,
        },
        "options": dict(CAPTURE_OPTIONS),
        "thread_environment": {
            key: os.environ.get(key) for key in THREAD_KEYS
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": _package_versions(),
        "numpy_build": numpy_build(),
        "source_hashes": source_hashes(),
        "data_hashes": data_hashes(),
    }


def manifest_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(
        canonical(manifest), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_manifest(
    manifest: dict[str, Any], *, require_ready: bool
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported manifest schema")
    if require_ready and not manifest.get("launch_ready"):
        raise RuntimeError(
            "Manifest has not been reviewed and marked launch_ready"
        )
    if manifest.get("numerical_base_commit") != NUMERICAL_BASE:
        raise RuntimeError("Manifest names a different numerical baseline")
    allocation = [
        (entry["label"], int(seed))
        for entry in manifest.get("allocation", [])
        for seed in entry.get("seeds", [])
    ]
    if len(allocation) != manifest.get("case_count") or len(
        set(allocation)
    ) != len(allocation):
        raise RuntimeError(
            "Manifest allocation has duplicates or an incorrect case_count"
        )
    if 2 * len(allocation) != manifest.get("state_target_count"):
        raise RuntimeError("Manifest state_target_count is inconsistent")


def runtime_identity(
    manifest: dict[str, Any],
    source_transition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and return the load-bearing identity of this invocation.

    ``source_transition`` lets a later campaign run on the captured states
    with declared source changes: ``changed_sources`` maps each
    repository-relative path to its ``capture`` hash (the manifest's pin)
    and its ``runtime`` hash (the version the campaign froze), and
    ``numerical_diff_exclude`` lists further paths, such as test
    directories, that the numerical diff ignores. Every other source stays
    pinned to the manifest, and the transition is echoed in the returned
    identity so that a campaign manifest binds it.
    """

    import gpyreg

    import pyvbmc

    if Path(pyvbmc.__file__).resolve().parent != ROOT / "pyvbmc":
        raise RuntimeError("PyVBMC imported outside this workspace")
    expected_gp = Path(manifest["gpyreg"]["import"]).resolve()
    if Path(gpyreg.__file__).resolve() != expected_gp:
        raise RuntimeError(
            "gpyreg imported from a path different from the manifest"
        )
    actual = {
        "numerical_base_commit": NUMERICAL_BASE,
        "gpyreg": gpyreg_identity(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": _package_versions(),
        "thread_environment": {
            key: os.environ.get(key) for key in THREAD_KEYS
        },
        "numpy_build": numpy_build(),
        "source_hashes": source_hashes(),
        "data_hashes": data_hashes(),
        "manifest_sha256": manifest_digest(manifest),
    }
    changed: dict[str, dict[str, str]] = {}
    exclude: tuple[str, ...] = ()
    if source_transition is not None:
        changed = dict(source_transition["changed_sources"])
        for path, hashes in changed.items():
            if manifest["source_hashes"].get(path) != hashes["capture"]:
                raise RuntimeError(
                    f"source transition misstates the captured hash of {path}"
                )
            if actual["source_hashes"].get(path) != hashes["runtime"]:
                raise RuntimeError(
                    f"Runtime source {path} differs from the declared"
                    " transition"
                )
        exclude = (
            *changed,
            *source_transition.get("numerical_diff_exclude", ()),
        )
    for key in (
        "gpyreg",
        "python",
        "platform",
        "dependencies",
        "thread_environment",
        "numpy_build",
        "source_hashes",
        "data_hashes",
    ):
        expected, observed = manifest[key], actual[key]
        if key == "source_hashes" and changed:
            expected = {k: v for k, v in expected.items() if k not in changed}
            observed = {k: v for k, v in observed.items() if k not in changed}
        if canonical(observed) != canonical(expected):
            raise RuntimeError(
                f"Runtime {key} differs from the frozen manifest"
            )
    if numerical_diff(exclude):
        raise RuntimeError(
            "Runtime numerical source differs from 9cc6882"
            + (" beyond the declared transition" if changed else "")
        )
    if source_transition is not None:
        actual["source_transition"] = canonical(source_transition)
    return actual


def _case_tag(label: str, seed: int) -> str:
    return f"{label}_seed{seed}"


def _completion_path(out: Path, tag: str) -> Path:
    return out / "records" / f"{tag}.complete.json"


def _case_path(out: Path, tag: str) -> Path:
    return out / "records" / f"{tag}.case.json"


def _failure_path(out: Path, tag: str) -> Path:
    return out / "records" / f"{tag}.failure.json"


def _partial_artifacts(out: Path, tag: str) -> list[Path]:
    """Return case artifacts that prevent retry; a diagnostic log does not."""

    paths = list(out.glob(f"{tag}*")) + list((out / "records").glob(f"{tag}*"))
    return [path for path in paths if path.suffix not in {".log", ".txt"}]


def _target_rng_state(problem: Any) -> Any:
    rng = getattr(problem, "_noise_rng", None)
    return None if rng is None else copy.deepcopy(rng.bit_generator.state)


def _observed_digest(
    xs: np.ndarray,
    gp: Any,
    vp: Any,
    logger: Any,
    optim: dict,
    problem: Any,
    *,
    include_candidates: bool = True,
) -> str:
    """Digest every mutable numerical leaf observed by acquisition capture."""

    live_n = logger.Xn + 1
    transformer = vp.parameter_transformer
    posterior_state = [
        (
            posterior.hyp,
            posterior.alpha,
            posterior.L,
            posterior.L_chol,
            posterior.sW,
            posterior.sn2_mult,
        )
        for posterior in gp.posteriors
    ]
    transformer_state = (
        transformer.lb_orig,
        transformer.ub_orig,
        transformer.mu,
        transformer.delta,
        transformer.type,
        transformer.R_mat,
        transformer.scale,
    )
    logger_state = (
        logger.X_orig[:live_n],
        logger.y_orig[:live_n],
        logger.X[:live_n],
        logger.y[:live_n],
        None
        if not hasattr(logger, "S") or logger.S is None
        else logger.S[:live_n],
        logger.n_evals[:live_n],
        logger.X_flag[:live_n],
        logger.fun_eval_time[:live_n],
        logger.Xn,
        logger.func_count,
        logger.cache_count,
        logger.y_max,
        logger.total_fun_eval_time,
        logger.noise_flag,
        logger.uncertainty_handling_level,
    )
    return digest(
        (
            xs if include_candidates else None,
            gp.X,
            gp.y,
            gp.s2,
            gp.temporary_data,
            posterior_state,
            vp.mu,
            vp.sigma,
            vp.lambd,
            vp.w,
            vp.eta,
            transformer_state,
            logger_state,
            optim,
            vp.rng.bit_generator.state,
            np.random.get_state(),
            _target_rng_state(problem),
        )
    )


class CaptureCollector:
    """Observe authentic full-sieve VIQR calls during one baseline fit."""

    def __init__(
        self,
        label: str,
        seed: int,
        problem: Any,
        options: Any,
        requested_options: dict[str, Any],
        identity: dict[str, Any],
    ) -> None:
        self.label = label
        self.seed = int(seed)
        self.problem = problem
        self.options = options
        self.requested_options = requested_options
        self.identity = identity
        self.early_threshold = max(20, 2 * int(problem.D))
        self.late_threshold = int(math.ceil(0.6 * options["max_fun_evals"]))
        self.ns_search = int(options["ns_search"])
        self.early: tuple[dict[str, np.ndarray], dict[str, Any]] | None = None
        self.late: tuple[dict[str, np.ndarray], dict[str, Any]] | None = None
        self.last: tuple[dict[str, np.ndarray], dict[str, Any]] | None = None
        self.calls_seen = 0

    def _snapshot(
        self,
        acq: Any,
        xs: np.ndarray,
        gp: Any,
        vp: Any,
        logger: Any,
        optim: dict,
        role: str,
        trigger: str,
    ):
        from pyvbmc.testing.oracles._state import encode, snapshot_from_objects

        rng_state = {
            "vp": copy.deepcopy(vp.rng.bit_generator.state),
            "numpy_legacy": copy.deepcopy(np.random.get_state()),
            "target": _target_rng_state(self.problem),
        }
        meta = {
            "capture_schema_version": SCHEMA_VERSION,
            "label": self.label,
            "seed": self.seed,
            "split": "development" if self.seed == 0 else "holdout",
            "checkpoint": role,
            "checkpoint_trigger": trigger,
            "D": int(self.problem.D),
            "func_count": int(logger.func_count),
            "early_threshold": self.early_threshold,
            "late_threshold": self.late_threshold,
            "max_fun_evals": int(self.options["max_fun_evals"]),
            "N": int(len(gp.X)),
            "Ns": int(len(gp.posteriors)),
            "K": int(vp.K),
            "candidate_count": int(np.atleast_2d(xs).shape[0]),
            "loss": getattr(acq, "loss", None),
            "manifest_sha256": self.identity["manifest_sha256"],
            "source_identity_sha256": digest(self.identity),
        }
        before = _observed_digest(xs, gp, vp, logger, optim, self.problem)
        arrays, tree = snapshot_from_objects(
            vp,
            gp,
            logger,
            optim,
            self.options,
            meta=meta,
            iteration=int(optim["iter"]),
        )
        n_repeat = int(xs.shape[0] - self.ns_search)
        tree["cand"] = {
            "Xs": encode(np.array(xs, copy=True), "cand/Xs", arrays),
            "sieve_Xs": encode(
                np.array(xs[n_repeat:], copy=True), "cand/sieve_Xs", arrays
            ),
            "n_repeat_candidates": n_repeat,
        }
        tree["meta"]["repeat_candidate_count"] = n_repeat
        factors = [
            {
                key: copy.deepcopy(getattr(posterior, key))
                for key in FACTOR_FIELDS
            }
            for posterior in gp.posteriors
        ]
        tree["live_factors"] = encode(factors, "live_factors", arrays)
        tree["temporary_data"] = encode(
            copy.deepcopy(gp.temporary_data), "temporary_data", arrays
        )
        tree["rng_state"] = encode(rng_state, "rng_state", arrays)
        tree["effective_options"] = canonical(dict(self.options))
        tree["requested_options"] = canonical(self.requested_options)
        arrays = {
            key: np.array(value, copy=True) for key, value in arrays.items()
        }
        if before != _observed_digest(xs, gp, vp, logger, optim, self.problem):
            raise RuntimeError(
                "State capture mutated the live acquisition state or RNG"
            )
        return arrays, tree

    def call(
        self,
        original: Any,
        acq: Any,
        xs: np.ndarray,
        gp: Any,
        vp: Any,
        logger: Any,
        optim: dict,
    ):
        xs = np.asarray(xs)
        full_sieve = (
            xs.ndim == 2
            and xs.shape[0] >= self.ns_search
            and getattr(acq, "loss", "iqr") == "iqr"
        )
        if not full_sieve:
            return original(acq, xs, gp, vp, logger, optim)

        self.calls_seen += 1
        func_count = int(logger.func_count)
        role = None
        trigger = None
        if self.early is None and func_count >= self.early_threshold:
            role, trigger = "early", "threshold"
        elif self.early is not None and self.late is None:
            role = (
                "late"
                if func_count >= self.late_threshold
                else "late_candidate"
            )
            trigger = "threshold" if role == "late" else "fallback_candidate"

        capture = None
        if role is not None:
            capture = self._snapshot(
                acq, xs, gp, vp, logger, optim, role, trigger
            )
        state_before = _observed_digest(
            xs,
            gp,
            vp,
            logger,
            optim,
            self.problem,
            include_candidates=False,
        )
        candidates_before = digest(xs)
        value = original(acq, xs, gp, vp, logger, optim)
        if state_before != _observed_digest(
            xs,
            gp,
            vp,
            logger,
            optim,
            self.problem,
            include_candidates=False,
        ):
            raise RuntimeError(
                "Public acquisition evaluation changed captured algorithm state"
            )
        integer_vars = np.asarray(optim.get("integer_vars", []), dtype=bool)
        if not np.any(integer_vars) and candidates_before != digest(xs):
            raise RuntimeError(
                "Public acquisition changed continuous candidates"
            )
        if capture is not None:
            arrays, tree = capture
            from pyvbmc.testing.oracles._state import encode

            tree["ref"] = {
                "acq": encode(np.asarray(value).copy(), "ref/acq", arrays)
            }
            if role == "early":
                self.early = (arrays, tree)
                self.last = (arrays, tree)
            elif role == "late":
                self.late = (arrays, tree)
                self.last = None
            else:
                self.last = (arrays, tree)
        return value

    def finish(self) -> dict[str, Any]:
        if (
            self.late is None
            and self.early is not None
            and self.last is not None
        ):
            early_meta = self.early[1]["meta"]
            last_meta = self.last[1]["meta"]
            distinct = (
                last_meta["func_count"] != early_meta["func_count"]
                or last_meta["iteration"] != early_meta["iteration"]
            )
            if distinct:
                arrays, tree = self.last
                tree["meta"]["checkpoint"] = "late"
                tree["meta"][
                    "checkpoint_trigger"
                ] = "early_termination_fallback"
                self.late = arrays, tree
        return {
            "calls_seen": self.calls_seen,
            "early_present": self.early is not None,
            "late_present": self.late is not None,
            "late_trigger": None
            if self.late is None
            else self.late[1]["meta"]["checkpoint_trigger"],
            "early_func_count": None
            if self.early is None
            else self.early[1]["meta"]["func_count"],
            "late_func_count": None
            if self.late is None
            else self.late[1]["meta"]["func_count"],
        }


def _save_snapshot_atomic(
    path: Path, arrays: dict[str, np.ndarray], tree: dict[str, Any]
) -> None:
    from pyvbmc.testing.oracles._state import save_snapshot

    temporary = path.with_name(path.name + ".tmp")
    save_snapshot(temporary, arrays, tree)
    for suffix in (".npz", ".json"):
        source = temporary.parent / f"{temporary.name}{suffix}"
        destination = path.parent / f"{path.name}{suffix}"
        source.replace(destination)


def _validate_capture_snapshot(snap: dict[str, Any]) -> None:
    """Reject incomplete capture extensions before rebuilding any objects."""

    required = {
        "gp",
        "vp",
        "logger",
        "optim_state",
        "options",
        "meta",
        "cand",
        "ref",
        "live_factors",
        "temporary_data",
        "rng_state",
        "effective_options",
        "requested_options",
    }
    missing = required - set(snap)
    if missing:
        raise RuntimeError(
            f"capture snapshot is missing keys: {sorted(missing)}"
        )
    if snap["meta"].get("capture_schema_version") != SCHEMA_VERSION:
        raise RuntimeError("capture snapshot has an unsupported schema")
    if not {"vp", "numpy_legacy", "target"} <= set(snap["rng_state"]):
        raise RuntimeError("capture snapshot has incomplete RNG state")
    factors = snap["live_factors"]
    expected = int(snap["gp"]["Ns"])
    if not isinstance(factors, list) or len(factors) != expected:
        raise RuntimeError("capture snapshot has the wrong live-factor count")
    for index, factor in enumerate(factors):
        missing_factors = set(FACTOR_FIELDS) - set(factor)
        if missing_factors:
            raise RuntimeError(
                f"capture posterior {index} is missing factors: "
                f"{sorted(missing_factors)}"
            )


def restore_capture(path: Path | str) -> dict[str, Any]:
    """Restore a capture with its live factors, temporary data, and exact RNG."""

    from pyvbmc.testing.oracles._state import build_state, load_snapshot

    snap = load_snapshot(path)
    _validate_capture_snapshot(snap)
    rng = np.random.default_rng()
    state = build_state(snap, rng=rng)
    state["vp"].rng.bit_generator.state = copy.deepcopy(
        snap["rng_state"]["vp"]
    )
    for posterior, factors in zip(
        state["gp"].posteriors, snap["live_factors"]
    ):
        for key, value in factors.items():
            setattr(posterior, key, copy.deepcopy(value))
    state["gp"].temporary_data = copy.deepcopy(snap["temporary_data"])
    state["rng_state"] = copy.deepcopy(snap["rng_state"])
    state["effective_options"] = copy.deepcopy(snap["effective_options"])
    state["requested_options"] = copy.deepcopy(snap["requested_options"])
    return state


def verify_capture(path: Path | str) -> dict[str, Any]:
    """Reproduce the captured public VIQR output without changing its RNG."""

    from pyvbmc.acquisition_functions import AcqFcnVIQR

    state = restore_capture(path)
    rng_before = digest(state["vp"].rng.bit_generator.state)
    candidates = np.array(state["cand"]["Xs"], copy=True)
    actual = AcqFcnVIQR()(
        candidates,
        state["gp"],
        state["vp"],
        state["logger"],
        copy.deepcopy(state["optim_state"]),
    )
    np.testing.assert_array_equal(actual, state["ref"]["acq"])
    if digest(state["vp"].rng.bit_generator.state) != rng_before:
        raise RuntimeError("Capture verification changed the restored VP RNG")
    return {
        "public_acquisition_exact": True,
        "candidate_count": int(candidates.shape[0]),
        "output_sha256": digest(np.asarray(actual)),
    }


def _publish_capture_case(
    manifest: dict[str, Any],
    out: Path,
    label: str,
    seed: int,
    identity: dict[str, Any],
    collector: CaptureCollector,
    started: float,
    results: dict[str, Any] | None,
    failure: dict[str, Any] | None,
) -> dict[str, Any]:
    """Publish a terminal success or failure and every state it captured."""

    tag = _case_tag(label, seed)
    capture_summary = collector.finish()
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    parity = {}
    for role, snapshot in (
        ("early", collector.early),
        ("late", collector.late),
    ):
        if snapshot is None:
            continue
        base = out / f"{tag}_{role}"
        _save_snapshot_atomic(base, *snapshot)
        parity[role] = verify_capture(base)
        for suffix in (".npz", ".json"):
            path = out / f"{tag}_{role}{suffix}"
            artifacts[path.relative_to(out).as_posix()] = sha256_file(path)

    status = "failed" if failure is not None else "succeeded"
    if failure is not None:
        failure_path = _failure_path(out, tag)
        write_json(failure_path, failure)
        artifacts[failure_path.relative_to(out).as_posix()] = sha256_file(
            failure_path
        )
        result_record = None
    else:
        result_record = {
            "success_flag": bool(results["success_flag"]),
            "message": str(results["message"]),
            "func_count": int(results["func_count"]),
            "iterations": int(results["iterations"]),
            "elbo": float(results["elbo"]),
            "elbo_sd": float(results["elbo_sd"]),
        }
    case = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "label": label,
        "seed": int(seed),
        "split": "development" if seed == 0 else "holdout",
        "capture": capture_summary,
        "parity": parity,
        "result": result_record,
        "failure": failure,
        "elapsed_seconds": time.time() - started,
        "identity": identity,
    }
    case_path = _case_path(out, tag)
    write_json(case_path, case)
    artifacts[case_path.relative_to(out).as_posix()] = sha256_file(case_path)
    complete = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "label": label,
        "seed": int(seed),
        "manifest_sha256": manifest_digest(manifest),
        "identity": identity,
        "artifacts": artifacts,
        "capture": capture_summary,
    }
    completion = _completion_path(out, tag)
    write_json(completion, complete)
    validate_completed_case(manifest, out, label, seed)
    return complete


def run_capture_case(
    manifest: dict[str, Any], out: Path, label: str, seed: int
) -> dict[str, Any]:
    """Run one authentic baseline trajectory and atomically publish its record."""

    from unittest.mock import patch

    from benchmark_targets import find_config

    from pyvbmc import VBMC
    from pyvbmc.acquisition_functions import AcqFcnVIQR

    validate_manifest(manifest, require_ready=True)
    identity_before = runtime_identity(manifest)
    allocated = {
        (entry["label"], int(item))
        for entry in manifest["allocation"]
        for item in entry["seeds"]
    }
    if (label, int(seed)) not in allocated:
        raise RuntimeError(
            f"Case {(label, seed)!r} is outside the manifest allocation"
        )
    tag = _case_tag(label, seed)
    completion = _completion_path(out, tag)
    if completion.exists():
        return validate_completed_case(manifest, out, label, seed)
    existing = _partial_artifacts(out, tag)
    if existing:
        raise RuntimeError(
            f"{tag}: incomplete prior artifacts exist; inspect before retry"
        )

    problem = find_config(label).make(seed=seed)
    args, options = problem.vbmc_args()
    options.update(manifest["options"])
    requested_options = copy.deepcopy(options)
    vbmc = VBMC(*args, options=options, seed=seed)
    collector = CaptureCollector(
        label, seed, problem, vbmc.options, requested_options, identity_before
    )
    original_call = AcqFcnVIQR.__call__

    def observed_call(acq, xs, gp, vp, logger, optim):
        return collector.call(original_call, acq, xs, gp, vp, logger, optim)

    started = time.time()
    results = None
    failure = None
    try:
        with patch.object(AcqFcnVIQR, "__call__", observed_call):
            _, results = vbmc.optimize()
    except Exception as error:  # noqa: BLE001 - failed trajectories are data
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    identity_after = runtime_identity(manifest)
    if identity_after != identity_before:
        raise RuntimeError(
            "Load-bearing source or environment changed during the fit"
        )
    return _publish_capture_case(
        manifest,
        out,
        label,
        seed,
        identity_before,
        collector,
        started,
        results,
        failure,
    )


def validate_completed_case(
    manifest: dict[str, Any], out: Path, label: str, seed: int
) -> dict[str, Any]:
    tag = _case_tag(label, seed)
    path = _completion_path(out, tag)
    if not path.exists():
        raise RuntimeError(f"{tag}: completion record is missing")
    complete = json.loads(path.read_text(encoding="utf-8"))
    if complete.get("manifest_sha256") != manifest_digest(manifest):
        raise RuntimeError(f"{tag}: manifest identity differs")
    if (complete.get("label"), complete.get("seed")) != (label, int(seed)):
        raise RuntimeError(f"{tag}: completion record names another case")
    if complete.get("status") not in {"succeeded", "failed"}:
        raise RuntimeError(f"{tag}: completion record has no terminal status")
    artifacts = complete.get("artifacts", {})
    expected_roles = [
        role
        for role in ("early", "late")
        if complete["capture"][f"{role}_present"]
    ]
    expected_names = {f"records/{tag}.case.json"}
    for role in expected_roles:
        expected_names.update({f"{tag}_{role}.json", f"{tag}_{role}.npz"})
    if complete["status"] == "failed":
        expected_names.add(f"records/{tag}.failure.json")
    if set(artifacts) != expected_names:
        raise RuntimeError(
            f"{tag}: completion artifact set is incomplete or unexpected"
        )
    for relative, expected in artifacts.items():
        artifact = out / relative
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise RuntimeError(
                f"{tag}: missing or changed artifact {relative}"
            )
    case = json.loads(_case_path(out, tag).read_text(encoding="utf-8"))
    if (
        case.get("status") != complete["status"]
        or case.get("capture") != complete["capture"]
    ):
        raise RuntimeError(f"{tag}: case and completion records disagree")
    if complete["status"] == "failed":
        failure = json.loads(
            _failure_path(out, tag).read_text(encoding="utf-8")
        )
        if not failure.get("type") or case.get("failure") != failure:
            raise RuntimeError(
                f"{tag}: failure record is empty or inconsistent"
            )
    elif case.get("failure") is not None or case.get("result") is None:
        raise RuntimeError(
            f"{tag}: successful case has inconsistent result fields"
        )
    return complete


def inventory(manifest: dict[str, Any], out: Path) -> dict[str, Any]:
    """Validate all present cases and report every expected state cell."""

    rows = []
    counts = {
        "complete_cases": 0,
        "missing_cases": 0,
        "failed_cases": 0,
        "states": 0,
    }
    for entry in manifest["allocation"]:
        for seed in entry["seeds"]:
            label, seed = entry["label"], int(seed)
            tag = _case_tag(label, seed)
            completion = _completion_path(out, tag)
            if not completion.exists():
                partial = _partial_artifacts(out, tag)
                status = "partial" if partial else "missing"
                counts["failed_cases" if partial else "missing_cases"] += 1
                rows.append(
                    {
                        "label": label,
                        "seed": seed,
                        "status": status,
                        "states": [],
                    }
                )
                continue
            try:
                done = validate_completed_case(manifest, out, label, seed)
                states = [
                    role
                    for role in ("early", "late")
                    if done["capture"][f"{role}_present"]
                ]
                counts["states"] += len(states)
                status = done["status"]
                counts[
                    "complete_cases"
                    if status == "succeeded"
                    else "failed_cases"
                ] += 1
                rows.append(
                    {
                        "label": label,
                        "seed": seed,
                        "split": "development" if seed == 0 else "holdout",
                        "status": status,
                        "states": states,
                        "late_trigger": done["capture"]["late_trigger"],
                        "func_count": {
                            "early": done["capture"]["early_func_count"],
                            "late": done["capture"]["late_func_count"],
                        },
                    }
                )
            except (
                Exception
            ) as error:  # noqa: BLE001 - inventory must retain failures
                counts["failed_cases"] += 1
                rows.append(
                    {
                        "label": label,
                        "seed": seed,
                        "status": "invalid",
                        "error": f"{type(error).__name__}: {error}",
                        "states": [],
                    }
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest_digest(manifest),
        "expected_cases": manifest["case_count"],
        "expected_states": manifest["state_target_count"],
        "counts": counts,
        "rows": rows,
    }


def run_controller(
    manifest_path: Path,
    manifest: dict[str, Any],
    out: Path,
    limit: int | None,
) -> int:
    validate_manifest(manifest, require_ready=True)
    identity = runtime_identity(manifest)
    out.mkdir(parents=True, exist_ok=True)
    launch_path = out / "launch.json"
    launch = {"manifest": manifest, "identity": identity}
    if launch_path.exists():
        if json.loads(launch_path.read_text(encoding="utf-8")) != canonical(
            launch
        ):
            raise RuntimeError(
                "Output directory belongs to another launch identity"
            )
    else:
        write_json(launch_path, launch)
    tasks = [
        (entry["label"], int(seed))
        for entry in manifest["allocation"]
        for seed in entry["seeds"]
    ]
    if limit is not None:
        tasks = tasks[:limit]
    terminal_failures = []
    for label, seed in tasks:
        tag = _case_tag(label, seed)
        completion = _completion_path(out, tag)
        if completion.exists():
            done = validate_completed_case(manifest, out, label, seed)
            if done["status"] == "failed":
                terminal_failures.append(tag)
                print(f"[resume] retained failed {tag}", flush=True)
            else:
                print(f"[resume] verified {tag}", flush=True)
            continue
        partial = _partial_artifacts(out, tag)
        if partial:
            raise RuntimeError(
                f"{tag}: incomplete prior attempt; inspect before retry"
            )
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "capture-one",
            "--manifest",
            str(manifest_path.resolve()),
            "--out",
            str(out.resolve()),
            "--label",
            label,
            "--seed",
            str(seed),
        ]
        print(f"START {tag}", flush=True)
        log_path = out / f"{tag}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            print(f"FAILED {tag}; see {log_path}", flush=True)
            return result.returncode
        done = validate_completed_case(manifest, out, label, seed)
        if done["status"] == "failed":
            terminal_failures.append(tag)
            print(f"FAILED {tag}; terminal record retained", flush=True)
        else:
            print(f"DONE {tag}", flush=True)
        write_json(out / "inventory.json", inventory(manifest, out))
    return 1 if terminal_failures else 0


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest, require_ready=False)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-manifest")
    prepare.add_argument("--out", type=Path, required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--manifest", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--limit", type=int)
    one = sub.add_parser("capture-one")
    one.add_argument("--manifest", type=Path, required=True)
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--label", required=True)
    one.add_argument("--seed", type=int, required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--manifest", type=Path, required=True)
    inv.add_argument("--out", type=Path, required=True)
    inv.add_argument("--write", type=Path)
    args = parser.parse_args(argv)

    if args.command == "prepare-manifest":
        write_json(args.out, default_manifest())
        print(
            f"Wrote review draft {args.out}; set launch_ready only after review."
        )
        return 0
    manifest = _load_manifest(args.manifest)
    if args.command == "capture":
        return run_controller(args.manifest, manifest, args.out, args.limit)
    if args.command == "capture-one":
        run_capture_case(manifest, args.out, args.label, args.seed)
        return 0
    if args.command == "inventory":
        report = inventory(manifest, args.out)
        if args.write:
            write_json(args.write, report)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 1 if report["counts"]["failed_cases"] else 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001 - CLI records a complete diagnostic
        traceback.print_exc()
        raise SystemExit(1)
