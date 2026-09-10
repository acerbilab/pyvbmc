"""Frozen inputs and provenance for the Stage 4 Torch feasibility prototype.

This module deliberately lives outside :mod:`pyvbmc`.  It rebuilds the eight
committed oracle *inputs* through their existing codec and creates three
deterministic synthetic fixed-GP states for the larger shape probes.  It never
fits a GP or calls a user target.  Rebuilding a synthetic state only computes
the posterior factors for explicitly supplied hyperparameters.

The public entry points are :func:`load_workload`, :func:`workload_manifest`
and :func:`environment_manifest`.  Imports of NumPy, gpyreg and PyVBMC are
kept inside functions so the benchmark coordinator can measure worker import
and process startup separately.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_FIXTURES = REPO_ROOT / "pyvbmc" / "testing" / "oracles" / "fixtures"
PLAN_PATH = REPO_ROOT / "dev" / "plans" / "stage4-torch-feasibility.md"
GPYREG_PIN = "a2f8ddce867f502e29717959cf0ff3529f598618"
MEASUREMENT_SEEDS = (1701, 1702, 1703)
RESCORE_STREAMS = 5
SYNTHETIC_RECIPE_SEED = 20260908


@dataclass(frozen=True)
class WorkloadSpec:
    """One bounded experiment input.

    ``mode`` is ``"main"``, ``"boost"`` or ``"kernel"``.  ``snapshot``
    names a committed oracle fixture; otherwise the dimensions define a
    deterministic synthetic state.  ``input_K`` is the VP stored in the
    input and ``target_K`` is the component count handed to the step.
    """

    name: str
    mode: str
    description: str
    snapshot: str | None = None
    D: int | None = None
    N: int | None = None
    input_K: int | None = None
    gp_samples: int | None = None
    target_K: int | None = None
    complete: bool = True
    authentic: bool = True


_SPECS = (
    WorkloadSpec(
        "warmup",
        "main",
        "Committed warm-up state; weights fixed, MC Adam.",
        snapshot="normal_D2_warmup",
    ),
    WorkloadSpec(
        "deterministic",
        "main",
        "Committed K=1 state; deterministic entropy and SciPy optimizer.",
        snapshot="normal_D2_K1",
    ),
    WorkloadSpec(
        "bounded",
        "main",
        "Committed half-normal state in transformed coordinates.",
        snapshot="halfnormal_D2_bounded",
    ),
    WorkloadSpec(
        "warped",
        "main",
        "Committed rotated/scaled transformed-coordinate state.",
        snapshot="corr_D5_warped",
    ),
    WorkloadSpec(
        "noisy",
        "main",
        "Committed noisy Rosenbrock state with a fixed fitted GP.",
        snapshot="rosenbrock_D2_noise1_viqr",
    ),
    WorkloadSpec(
        "boost_sampled",
        "boost",
        "Committed cigar input grown from K=14 to final-boost K=50.",
        snapshot="cigar_D4_largeK",
        target_K=50,
    ),
    WorkloadSpec(
        "medium_synthetic",
        "main",
        "Synthetic fixed-GP shape probe D=10,N=250,K=25,Ns=5.",
        D=10,
        N=250,
        input_K=25,
        gp_samples=5,
        target_K=25,
        authentic=False,
    ),
    WorkloadSpec(
        "boost_single",
        "boost",
        "Synthetic fixed-GP shape probe D=15,N=750,K=25->50,Ns=1.",
        D=15,
        N=750,
        input_K=25,
        gp_samples=1,
        target_K=50,
        authentic=False,
    ),
    WorkloadSpec(
        "kernel_stress",
        "kernel",
        "Synthetic kernel-only shape probe D=20,N=500,K=60,Ns=1.",
        D=20,
        N=500,
        input_K=60,
        gp_samples=1,
        target_K=60,
        complete=False,
        authentic=False,
    ),
)

WORKLOADS = {spec.name: spec for spec in _SPECS}

# All committed oracle states participate in kernel gates even when they are
# not complete-fit workloads.  Prefixing keeps CLI names unambiguous.
ORACLE_KERNEL_CASES = {
    f"kernel_{name}": WorkloadSpec(
        f"kernel_{name}",
        "kernel",
        f"Kernel gate for committed oracle input {name}.",
        snapshot=name,
        complete=False,
    )
    for name in (
        "normal_D2_warmup",
        "normal_D2_K1",
        "normal_D2_singlesample",
        "halfnormal_D2_bounded",
        "corr_D5_warped",
        "rosenbrock_D2_noise1_viqr",
        "cigar_D4_largeK",
        "cigar_D4_boosted",
    )
}
ALL_CASES = {**WORKLOADS, **ORACLE_KERNEL_CASES}


_OPTION_KEYS = (
    "ns_ent",
    "ns_ent_fast",
    "ns_ent_fine",
    "ns_ent_boost",
    "ns_ent_fast_boost",
    "ns_ent_fine_boost",
    "ns_elbo",
    "ns_elbo_incr",
    "elbo_starts",
    "elcbo_midpoint",
    "stochastic_optimizer",
    "sgd_step_size",
    "max_iter_stochastic",
    "tol_fun_stochastic",
    "det_entropy_tol_opt",
    "tol_weight",
    "weight_penalty",
    "tol_con_loss",
    "tol_length",
    "tol_improvement",
    "pruning_threshold_multiplier",
    "elcbo_impro_weight",
    "min_final_components",
    "tol_elcbo_boost",
    "variable_means",
    "variable_weights",
)

_SOURCE_PATHS = (
    "pyvbmc/vbmc/variational_optimization.py",
    "pyvbmc/vbmc/minimize_adam.py",
    "pyvbmc/entropy/entmc_vbmc.py",
    "pyvbmc/entropy/entlb_vbmc.py",
    "pyvbmc/variational_posterior/variational_posterior.py",
    "pyvbmc/vbmc/vbmc.py",
    "pyvbmc/testing/oracles/_state.py",
    "dev/scripts/torch_vi_core.py",
    "dev/scripts/torch_vi_step.py",
    "dev/scripts/torch_vi_fixtures.py",
    "dev/scripts/torch_vi_benchmark.py",
    "dev/plans/stage4-torch-feasibility.md",
)

_OPTION_CONFIG_PATHS = (
    "pyvbmc/vbmc/option_configs/basic_vbmc_options.ini",
    "pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini",
)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Strict-JSON representation for options and environment metadata."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return {"__float__": "nan"}
        return {"__float__": "+inf" if value > 0 else "-inf"}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(v) for v in sorted(value, key=repr)]
    if callable(value):
        module = getattr(value, "__module__", None)
        name = getattr(value, "__qualname__", getattr(value, "__name__", None))
        return {"callable": ".".join(x for x in (module, name) if x)}
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_origin(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    return None if spec is None or spec.origin is None else str(spec.origin)


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_at(path: Path, args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _gpyreg_checkout() -> dict[str, Any]:
    origin = _module_origin("gpyreg")
    if origin is None:
        return {
            "origin": None,
            "head": None,
            "dirty": None,
            "matches_pin": False,
        }
    package_dir = Path(origin).resolve().parent
    status = _git_at(package_dir, ["status", "--porcelain"])
    head = _git_at(package_dir, ["rev-parse", "HEAD"])
    return {
        "origin": origin,
        "checkout": str(package_dir),
        "head": head,
        "matches_pin": head == GPYREG_PIN,
        "branch": _git_at(package_dir, ["branch", "--show-current"]),
        "dirty": None if status is None else bool(status),
    }


def _config_manifest() -> dict[str, Any]:
    return {
        rel: {"path": str(REPO_ROOT / rel), "sha256": _sha256(REPO_ROOT / rel)}
        for rel in _OPTION_CONFIG_PATHS
    }


def environment_manifest() -> dict[str, Any]:
    """Return lightweight, non-mutating runtime and source provenance."""

    packages = ("pyvbmc", "gpyreg", "numpy", "scipy", "torch")
    gpyreg_git = _gpyreg_checkout()
    configs = _config_manifest()
    provenance_errors = []
    if not gpyreg_git.get("matches_pin"):
        provenance_errors.append("gpyreg HEAD does not match GPYREG_PIN")
    if gpyreg_git.get("dirty") is not False:
        provenance_errors.append("gpyreg checkout is dirty or unreadable")
    if any(item["sha256"] is None for item in configs.values()):
        provenance_errors.append("an option config is missing")
    return {
        "schema": 1,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node(),
        },
        "packages": {name: _package_version(name) for name in packages},
        "module_origins": {name: _module_origin(name) for name in packages},
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "git": {
            "head": _git(["rev-parse", "HEAD"]),
            "branch": _git(["branch", "--show-current"]),
            "dirty": bool(_git(["status", "--porcelain"])),
        },
        "gpyreg_pin": GPYREG_PIN,
        "gpyreg_git": gpyreg_git,
        "option_configs": configs,
        "provenance_validation": {
            "valid": not provenance_errors,
            "errors": provenance_errors,
        },
        "source_hashes": {
            rel: _sha256(REPO_ROOT / rel) for rel in _SOURCE_PATHS
        },
        "plan_hash": _sha256(PLAN_PATH),
    }


def _set_option(options: Any, key: str, value: Any) -> None:
    options.__setitem__(key, value, force=True)


def _build_synthetic(spec: WorkloadSpec, run_seed: int) -> dict[str, Any]:
    """Construct a deterministic fixed-GP shape probe without GP fitting."""

    import gpyreg as gpr
    import numpy as np

    from pyvbmc.parameter_transformer import ParameterTransformer
    from pyvbmc.testing.oracles._state import build_options
    from pyvbmc.variational_posterior import VariationalPosterior

    if None in (spec.D, spec.N, spec.input_K, spec.gp_samples):
        raise ValueError(
            f"incomplete synthetic workload specification: {spec}"
        )
    D, N, K, Ns = spec.D, spec.N, spec.input_K, spec.gp_samples
    recipe_rng = np.random.default_rng(SYNTHETIC_RECIPE_SEED + D + N + K + Ns)

    # Deterministic, well-spread transformed-space design.  The diagonal
    # scales keep each coordinate O(1), while the small shared latent creates
    # enough correlation to exercise nontrivial ARD integrals.
    X = recipe_rng.standard_normal((N, D))
    X += 0.15 * recipe_rng.standard_normal((N, 1))
    y = (
        -0.5 * np.sum(X**2, axis=1) - 0.02 * np.sum(X, axis=1) ** 2
    ).reshape(-1, 1)

    covariance = gpr.covariance_functions.SquaredExponential()
    mean = gpr.mean_functions.NegativeQuadratic()
    noise = gpr.noise_functions.GaussianNoise(constant_add=True)
    gp = gpr.GP(D=D, covariance=covariance, mean=mean, noise=noise)
    cov_N = covariance.hyperparameter_count(D)
    noise_N = noise.hyperparameter_count()
    mean_N = mean.hyperparameter_count(D)
    hyp = np.zeros((Ns, cov_N + noise_N + mean_N), dtype=np.float64)
    for s in range(Ns):
        offset = 0.04 * (s - (Ns - 1) / 2)
        hyp[s, :D] = np.log(1.2 + 0.03 * np.arange(D)) + offset
        hyp[s, D] = offset  # log signal scale
        hyp[s, cov_N : cov_N + noise_N] = -1.5 + 0.25 * offset
        mean_start = cov_N + noise_N
        hyp[s, mean_start] = float(np.max(y))
        hyp[s, mean_start + 1 : mean_start + 1 + D] = 0.0
        hyp[s, mean_start + 1 + D :] = np.log(3.0 + 0.05 * np.arange(D))
    # Explicit hyperparameters: posterior computation, never optimization.
    gp.update(X_new=X, y_new=y, hyp=hyp)

    pt = ParameterTransformer(D)
    vp = VariationalPosterior(
        D,
        K,
        x0=np.zeros((1, D)),
        parameter_transformer=pt,
        rng=np.random.default_rng(run_seed),
    )
    pick = np.linspace(0, N - 1, K, dtype=int)
    vp.mu = np.array(X[pick].T, dtype=np.float64, copy=True)
    vp.sigma = np.full((1, K), 0.45 / max(1.0, K**0.15), dtype=np.float64)
    vp.lambd = np.ones((D, 1), dtype=np.float64)
    vp.w = np.full((1, K), 1.0 / K, dtype=np.float64)
    vp.eta = np.log(vp.w)
    vp.stats = None
    vp.bounds = None

    options = build_options({}, D)
    # These are main-loop inputs.  The step adapter performs the documented
    # boost overrides on a private copy when spec.mode == "boost".
    optim_state = {
        "warmup": False,
        "entropy_switch": False,
        "vp_K": K,
        "entropy_alpha": 0.0,
        # This is the one sanctioned full-refit regime in the bounded suite.
        # All committed snapshots currently exercise the incremental path.
        "recompute_var_post": spec.name == "medium_synthetic",
    }
    return {
        "pt": pt,
        "vp": vp,
        "gp": gp,
        "options": options,
        "optim_state": optim_state,
        "meta": {
            "synthetic": True,
            "recipe_seed": SYNTHETIC_RECIPE_SEED,
            "D": D,
            "N": N,
            "K": K,
            "Ns": Ns,
            "note": spec.description,
        },
    }


def _build_committed(spec: WorkloadSpec, run_seed: int) -> dict[str, Any]:
    import numpy as np

    from pyvbmc.testing.oracles._state import build_state, load_snapshot

    if spec.snapshot is None:
        raise ValueError(f"{spec.name} has no committed snapshot")
    path = ORACLE_FIXTURES / spec.snapshot
    snap = load_snapshot(path)
    state = build_state(snap, rng=np.random.default_rng(run_seed))
    state["meta"] = dict(state["meta"])
    state["meta"]["synthetic"] = False
    state["meta"]["fixture_hashes"] = {
        "json": _sha256(path.parent / f"{path.name}.json"),
        "npz": _sha256(path.parent / f"{path.name}.npz"),
    }
    return state


def _shape_record(state: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    vp, gp = state["vp"], state["gp"]
    return {
        "D": int(vp.D),
        "N": int(gp.X.shape[0]),
        "K": int(vp.K),
        "Ns": int(len(gp.posteriors)),
        "theta": int(np.size(vp.get_parameters())),
        "X": list(gp.X.shape),
        "mu": list(vp.mu.shape),
        "sigma": list(vp.sigma.shape),
        "lambd": list(vp.lambd.shape),
        "w": list(vp.w.shape),
        "factor_representations": [
            "chol" if bool(post.L_chol) else "inverse"
            for post in gp.posteriors
        ],
    }


def _resolved_options(
    options: Any, K_values: tuple[int, ...]
) -> dict[str, Any]:
    """Record option source values plus callable results at relevant K."""

    result: dict[str, Any] = {}
    for key in sorted(options.keys()):
        value = options.get(key)
        entry = _jsonable(value)
        if callable(value):
            evaluated: dict[str, Any] = {}
            for K in K_values:
                try:
                    evaluated[str(K)] = _jsonable(options.eval(key, {"K": K}))
                except (
                    Exception
                ) as exc:  # pragma: no cover - provenance fallback
                    evaluated[str(K)] = {
                        "error": f"{type(exc).__name__}: {exc}"
                    }
            entry = {**entry, "evaluated_by_K": evaluated}
        result[key] = entry
    return result


def _validate_float64_state(state: Mapping[str, Any]) -> None:
    import numpy as np

    vp, gp = state["vp"], state["gp"]
    arrays = {
        "gp.X": gp.X,
        "gp.y": gp.y,
        "vp.mu": vp.mu,
        "vp.sigma": vp.sigma,
        "vp.lambd": vp.lambd,
        "vp.w": vp.w,
        "vp.eta": vp.eta,
    }
    for s, post in enumerate(gp.posteriors):
        arrays[f"gp.posteriors[{s}].hyp"] = post.hyp
        arrays[f"gp.posteriors[{s}].alpha"] = post.alpha
        arrays[f"gp.posteriors[{s}].L"] = post.L
        arrays[f"gp.posteriors[{s}].sW"] = post.sW
    bad = {
        name: str(np.asarray(value).dtype)
        for name, value in arrays.items()
        if np.asarray(value).dtype != np.float64
    }
    if bad:
        raise TypeError(f"non-float64 fixed state: {bad}")


def load_workload(name: str, seed: int) -> dict[str, Any]:
    """Build one private workload state and attach its resolved contract."""

    if name not in ALL_CASES:
        raise KeyError(
            f"unknown workload {name!r}; choose from {sorted(ALL_CASES)}"
        )
    spec = ALL_CASES[name]
    state = (
        _build_committed(spec, seed)
        if spec.snapshot is not None
        else _build_synthetic(spec, seed)
    )
    # Never let one backend arm consume another arm's source objects or RNG.
    state = copy.deepcopy(state)
    import numpy as np

    state["vp"].rng = np.random.default_rng(seed)
    if name == "warmup":
        state["optim_state"]["warmup"] = True
        state["vp"].optimize_weights = False
    state["label"] = name
    state["seed"] = int(seed)
    state["boost"] = spec.mode == "boost"
    state["K"] = spec.target_K or int(state["vp"].K)
    state["fast_opts_N"] = None
    state["slow_opts_N"] = None
    state["spec"] = spec
    _validate_float64_state(state)
    target_K = int(state["K"])
    input_K = int(state["vp"].K)
    K_values = tuple(dict.fromkeys((input_K, target_K)))
    recompute = bool(state["optim_state"].get("recompute_var_post", False))
    always_refit = bool(state["options"].get("always_refit_vp"))
    base_budget = int(
        np.ceil(state["options"].eval("ns_elbo", {"K": input_K}))
    )
    if recompute or always_refit:
        fast_budget = base_budget
        slow_budget = int(state["options"].get("elbo_starts"))
        sieve_regime = "full_refit"
    else:
        fast_budget = int(
            np.ceil(base_budget * state["options"].get("ns_elbo_incr"))
        )
        slow_budget = 1
        sieve_regime = "incremental"
    state["contract"] = {
        "spec": asdict(spec),
        "seed": int(seed),
        "shape": _shape_record(state),
        "parameter_flags": {
            "mu": bool(state["vp"].optimize_mu),
            "sigma": bool(state["vp"].optimize_sigma),
            "lambd": bool(state["vp"].optimize_lambd),
            "weights": bool(state["vp"].optimize_weights),
        },
        "regime": {
            "warmup": bool(state["optim_state"].get("warmup", False)),
            "entropy_switch": bool(
                state["optim_state"].get("entropy_switch", False)
            ),
            "boost": spec.mode == "boost",
            "complete": spec.complete,
            "recompute_var_post": recompute,
            "sieve_regime": sieve_regime,
            "fast_opts_N": fast_budget,
            "slow_opts_N": slow_budget,
        },
        "resolved_options": _resolved_options(state["options"], K_values),
        "option_configs": _config_manifest(),
        "fixture_meta": _jsonable(state.get("meta", {})),
    }
    return state


def workload_manifest(
    names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Describe selected inputs without running any numerical objective."""

    selected = list(WORKLOADS if names is None else names)
    unknown = set(selected) - set(ALL_CASES)
    if unknown:
        raise KeyError(f"unknown workloads: {sorted(unknown)}")
    cases: dict[str, Any] = {}
    for name in selected:
        spec = ALL_CASES[name]
        item = asdict(spec)
        if spec.snapshot is not None:
            base = ORACLE_FIXTURES / spec.snapshot
            item["fixture_hashes"] = {
                "json": _sha256(base.parent / f"{base.name}.json"),
                "npz": _sha256(base.parent / f"{base.name}.npz"),
            }
        cases[name] = item
    return {
        "schema": 1,
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "rescore_streams": RESCORE_STREAMS,
        "cases": cases,
        "environment": environment_manifest(),
    }


def write_manifest(
    path: str | os.PathLike[str], manifest: Mapping[str, Any]
) -> None:
    """Write strict JSON atomically; raw numerical arrays belong in NPZ."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(manifest), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)


__all__ = [
    "ALL_CASES",
    "GPYREG_PIN",
    "MEASUREMENT_SEEDS",
    "ORACLE_KERNEL_CASES",
    "RESCORE_STREAMS",
    "WORKLOADS",
    "WorkloadSpec",
    "environment_manifest",
    "load_workload",
    "workload_manifest",
    "write_manifest",
]
