"""Reproducible NumPy versus upstream Torch CPU comparison for S-VBMC.

This is a bounded developer experiment.  It imports the pinned, unchanged
upstream S-VBMC checkout and compares it with ``svbmc_numpy_core`` without
changing either PyVBMC or the upstream source.  Numerical imports deliberately
live below argument parsing/path setup so that the Torch overlay is explicit.

Examples
--------
Run the early, ten-step paired smoke check::

    python dev/scripts/svbmc_numpy_prototype.py --phase smoke \
        --output dev/experiments/svbmc_numpy/smoke.json

Run fixed-input correctness and kernel timings::

    python dev/scripts/svbmc_numpy_prototype.py --phase kernels \
        --output dev/experiments/svbmc_numpy/kernels.json

Run the clean three-group, two-mode, three-seed fit campaign::

    python dev/scripts/svbmc_numpy_prototype.py --phase fits \
        --output dev/experiments/svbmc_numpy/fits.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import pickle
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# These must be fixed before importing NumPy, SciPy, or Torch.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "dev/scripts/runs/svbmc_compat_20260908/source"
DEFAULT_DEPS = REPO_ROOT / "dev/scripts/runs/svbmc_compat_20260908/deps"

EXPECTED_UPSTREAM_COMMIT = "13a78f6c4a3ffe9c4557fee4f4b8f67c98b29e01"
EXPECTED_UPSTREAM_VERSION = "0.1.1"
EXPECTED_SOURCE_SHA256 = (
    "98465fbb75de8636afde5c96b538483ff02fa457da0e6421356907a82a3a9607"
)
EXPECTED_DEPENDENCIES = {
    "numpy": "2.5.2",
    "scipy": "1.18.1",
    "torch": "2.14.0+cpu",
}
EXPECTED_GPYREG_COMMIT = "a2f8ddce867f502e29717959cf0ff3529f598618"

GROUPS = {
    "GMM": "vp_gmm_try_*.pkl",
    "GMM_noisy": "vp_noisy_gmm_try_*.pkl",
    "Ring": "vp_ring_try_*.pkl",
}

# The experiment must fail rather than silently accepting a different corpus.
EXPECTED_CORPUS_SHA256 = {
    "GMM/vp_gmm_try_0.pkl": "91481b2de648c2068dd9c868f9c46e9d4ccd5c0ad99af24a38b6548695bf2b38",
    "GMM/vp_gmm_try_1.pkl": "12561667ffcce01e128fb9ac91dc1b996ca78a3dc49148748d0aff06f49f59be",
    "GMM/vp_gmm_try_2.pkl": "b1702da3416324881b74d2536d7ab0a21de21c7e2a22d85469193346423c3eaa",
    "GMM/vp_gmm_try_3.pkl": "a0ec33be111f8b09aa2e029e6f6eeb25f5a9b4ff1737f6c119c921424b503eab",
    "GMM/vp_gmm_try_4.pkl": "e30ccd4eee676bbf1218ef549555a48584b542703fb22a6b76c169fd6fc6247d",
    "GMM/vp_gmm_try_5.pkl": "4d3d751978ba5c6314e616ddd56012c52cded1b8792a5ce9fd6d16fe122193f3",
    "GMM/vp_gmm_try_6.pkl": "4e148e675eb61eca83e439a64c61ee442ada8465f86c531724a9d0996514fa75",
    "GMM/vp_gmm_try_7.pkl": "a8bf7a3e7d40f9b765708462123e1d4050a57e9300be64429fe49b1b4ea00b55",
    "GMM/vp_gmm_try_8.pkl": "e753c600db874edaf1812cbdb960d7bcd9064deeb147593c02ff4537f48b7df7",
    "GMM/vp_gmm_try_9.pkl": "6e79bce9a0b57be3069cf7f5165f41ac57a00b7c35a1606dc89500e6e1927578",
    "GMM_noisy/vp_noisy_gmm_try_0.pkl": "c162197357e69c74bccc3ae74862bbf0703fd561d43ced89827034fe2d40f844",
    "GMM_noisy/vp_noisy_gmm_try_1.pkl": "b7fa0daa0e9357f1545694143f0f5567061baf30f62ce361a4cc415286925649",
    "GMM_noisy/vp_noisy_gmm_try_2.pkl": "e6101d9f03b566cce2422071216403103358a19531c6e925d2c2d5d05b22fb33",
    "GMM_noisy/vp_noisy_gmm_try_3.pkl": "7e7fe1ba23ec132a6fefa6038339da0d811cdc89a585b387ef2c242ae5291717",
    "GMM_noisy/vp_noisy_gmm_try_4.pkl": "92cab0a4134fc694cbc8079ab8b5fb4f5b8a6fe499d0c1eab04f820045852d59",
    "GMM_noisy/vp_noisy_gmm_try_5.pkl": "809fc43ff3030ef4b1c2f52b1835462091c83005f4b91d164ba367c4c4496b1e",
    "GMM_noisy/vp_noisy_gmm_try_6.pkl": "5e5daece5e03fdfc69a97194dbd8bc6c803963057ad906c057ce82714caac203",
    "GMM_noisy/vp_noisy_gmm_try_7.pkl": "476432b08f197e5043094f255f7930868355e80b4b96735654be4ea9843f5c19",
    "GMM_noisy/vp_noisy_gmm_try_8.pkl": "79920b041fcf1c65596dd87d67fb042747199d9c1fe5d2dc0485f0d40087acd6",
    "GMM_noisy/vp_noisy_gmm_try_9.pkl": "2460f70ed58cce30c582438e0970d080fa5f02eedc8437be22b7bf4f6575e0db",
    "Ring/vp_ring_try_0.pkl": "843bc342789838c13e3211a22c29cc0a25e28b55d0e427bae8e355631e35abb5",
    "Ring/vp_ring_try_1.pkl": "965c242d62b5f6e900992b6af03b0004f2572c4c10e9be97115f7e5289a9b58f",
    "Ring/vp_ring_try_2.pkl": "4e4e63e311cfdf6085861dfa190131d489f5a6e7a72e1865606c6323d344fc25",
    "Ring/vp_ring_try_3.pkl": "d10cad32432d057044a6784a2c6d2bbdf50e984b26900a090b35529b34ec5868",
    "Ring/vp_ring_try_4.pkl": "40f69ab44253f1abcf864651350b15a9627782ae5c7b36778ce3b71812c22c44",
    "Ring/vp_ring_try_5.pkl": "ea887f6eb7e267cfe12d9566823b23840743a97d4113fc1340e3aa6c1a697c05",
    "Ring/vp_ring_try_6.pkl": "2a84ce6a853e7545bd4fddbb1983d108f6024ce55ec1045fcf8cf780eb1e5b4c",
    "Ring/vp_ring_try_7.pkl": "a1fa8afe2092400037ad8cfe02508d80f720dd786d3244330e58eb92cd9db3e2",
    "Ring/vp_ring_try_8.pkl": "75ea52fc3482244fc305fe3d3b8244914f43015c33a892989ec6df67cece8fc1",
    "Ring/vp_ring_try_9.pkl": "2d87c0fb72799fca0d945603c7f63076311b1406133d96cc085923f917ca19d2",
}

# Declared before any measurement and copied verbatim into every report.
TOLERANCES = {
    "preparation_logq": {"rtol": 2e-12, "atol": 2e-11},
    "jacobian_correction": {"rtol": 5e-13, "atol": 5e-13},
    "fixed_objective": {"rtol": 2e-12, "atol": 2e-11},
    "fixed_gradient": {"rtol": 2e-11, "atol": 2e-10},
    "fit_scalar": {"rtol": 2e-9, "atol": 2e-8},
    "fit_weights": {"rtol": 2e-9, "atol": 2e-8},
    "trace_scalar": {"rtol": 2e-9, "atol": 2e-8},
    "trace_weights": {"rtol": 2e-9, "atol": 2e-8},
}


class _NullWriter(io.TextIOBase):
    """Discard upstream progress output while retaining its code path."""

    def write(self, text: str) -> int:
        return len(text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("smoke", "kernels", "fits", "all"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--deps", type=Path, default=DEFAULT_DEPS)
    parser.add_argument(
        "--groups", nargs="+", choices=tuple(GROUPS), default=list(GROUPS)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1701, 1702, 1703]
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=("all-weights", "posterior-only"),
        default=["all-weights", "posterior-only"],
    )
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--smoke-max-steps", type=int, default=10)
    parser.add_argument("--prep-repeats", type=int, default=3)
    parser.add_argument("--kernel-repeats", type=int, default=10)
    parser.add_argument(
        "--diagnostic-seed",
        type=int,
        default=1701,
        help="One seed used for separate instrumented full-fit diagnostics.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing report."
    )
    args = parser.parse_args(argv)
    if args.n_samples <= 0 or args.max_steps <= 0 or args.smoke_max_steps <= 0:
        parser.error("sample and step counts must be positive")
    if args.prep_repeats <= 0 or args.kernel_repeats <= 0:
        parser.error("timing repeat counts must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    return args


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def atomic_write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            _jsonable(report),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(array: Any, np: Any) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(repr(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def rng_state_digest(state: tuple[Any, ...], np: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(state[0]).encode("ascii"))
    digest.update(np.ascontiguousarray(state[1]).tobytes())
    digest.update(str((state[2], state[3], state[4])).encode("ascii"))
    return digest.hexdigest()


def rng_states_equal(
    left: tuple[Any, ...], right: tuple[Any, ...], np: Any
) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def compare_arrays(
    left: Any, right: Any, tolerance: dict[str, float], np: Any
) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    same_shape = left_array.shape == right_array.shape
    if same_shape and left_array.size:
        difference = np.abs(left_array - right_array)
        max_abs = float(np.max(difference))
        denominator = np.maximum(
            np.abs(right_array), np.finfo(np.float64).tiny
        )
        max_rel = float(np.max(difference / denominator))
    elif same_shape:
        max_abs = 0.0
        max_rel = 0.0
    else:
        max_abs = None
        max_rel = None
    passed = bool(
        same_shape
        and np.allclose(
            left_array,
            right_array,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
        )
    )
    return {
        "passed": passed,
        "left_shape": list(left_array.shape),
        "right_shape": list(right_array.shape),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "rtol": tolerance["rtol"],
        "atol": tolerance["atol"],
    }


def timing_summary(values: list[float]) -> dict[str, Any]:
    return {
        "repeats": len(values),
        "median_seconds": float(statistics.median(values)),
        "minimum_seconds": float(min(values)),
        "maximum_seconds": float(max(values)),
        "all_seconds": [float(value) for value in values],
    }


def archive_experiment_sources(output: Path) -> dict[str, Any]:
    """Freeze the two experiment scripts beside the report before measuring."""
    archive = output.parent / f"{output.stem}.sources"
    archive.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__).resolve(),
        REPO_ROOT / "dev/scripts/svbmc_numpy_core.py",
    ]
    records = {}
    for source in sources:
        destination = archive / f"{source.name}.txt"
        shutil.copy2(source, destination)
        records[source.name] = {
            "live_path": str(source),
            "archive_path": str(destination),
            "sha256": sha256_file(destination),
        }
    return {"directory": str(archive), "files": records}


def import_backends(args: argparse.Namespace) -> dict[str, Any]:
    source_src = args.source.resolve() / "src"
    deps = args.deps.resolve()
    for path in (REPO_ROOT, source_src, deps):
        sys.path.insert(0, str(path))

    started = time.perf_counter()
    np = importlib.import_module("numpy")
    scipy = importlib.import_module("scipy")
    torch = importlib.import_module("torch")
    pyvbmc = importlib.import_module("pyvbmc")
    gpyreg = importlib.import_module("gpyreg")
    svbmc = importlib.import_module("svbmc")
    core = importlib.import_module("svbmc_numpy_core")
    import_seconds = time.perf_counter() - started

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # It can only be set before inter-op work begins; record the actual value.
        pass
    normal_default = str(torch.get_default_dtype())
    torch.set_default_dtype(torch.float64)

    return {
        "np": np,
        "scipy": scipy,
        "torch": torch,
        "pyvbmc": pyvbmc,
        "gpyreg": gpyreg,
        "svbmc": svbmc,
        "SVBMC": svbmc.SVBMC,
        "core": core,
        "import_seconds": import_seconds,
        "normal_torch_default_dtype": normal_default,
    }


def verify_environment(
    args: argparse.Namespace, modules: dict[str, Any]
) -> dict[str, Any]:
    np = modules["np"]
    scipy = modules["scipy"]
    torch = modules["torch"]
    source = args.source.resolve()
    source_file = source / "src/svbmc/svbmc.py"
    corpus_root = source / "vbmc_runs"

    actual_commit = git_output(source, "rev-parse", "HEAD")
    source_status = git_output(
        source, "status", "--porcelain", "--untracked-files=no"
    )
    actual_source_hash = sha256_file(source_file)
    corpus_hashes = {
        str(path.relative_to(corpus_root)).replace("\\", "/"): sha256_file(
            path
        )
        for path in sorted(corpus_root.rglob("*.pkl"))
    }

    svbmc_file = Path(modules["svbmc"].__file__).resolve()
    expected_svbmc_init = (source / "src/svbmc/__init__.py").resolve()
    pyvbmc_file = Path(modules["pyvbmc"].__file__).resolve()
    expected_pyvbmc_init = (REPO_ROOT / "pyvbmc/__init__.py").resolve()
    gpyreg_file = Path(modules["gpyreg"].__file__).resolve()
    gpyreg_repo = gpyreg_file.parent.parent
    gpyreg_commit = git_output(gpyreg_repo, "rev-parse", "HEAD")

    checks = {
        "upstream_commit": actual_commit == EXPECTED_UPSTREAM_COMMIT,
        "upstream_tracked_source_clean": source_status == "",
        "upstream_source_sha256": actual_source_hash == EXPECTED_SOURCE_SHA256,
        "corpus_exact_paths_and_sha256": corpus_hashes
        == EXPECTED_CORPUS_SHA256,
        "svbmc_import_path": svbmc_file == expected_svbmc_init,
        "pyvbmc_import_path": pyvbmc_file == expected_pyvbmc_init,
        "svbmc_version": modules["svbmc"].__version__
        == EXPECTED_UPSTREAM_VERSION,
        "numpy_version": np.__version__ == EXPECTED_DEPENDENCIES["numpy"],
        "scipy_version": scipy.__version__ == EXPECTED_DEPENDENCIES["scipy"],
        "torch_version": torch.__version__ == EXPECTED_DEPENDENCIES["torch"],
        "torch_cpu_build": torch.version.cuda is None,
        "torch_default_float64": torch.get_default_dtype() == torch.float64,
        "torch_intraop_one_thread": torch.get_num_threads() == 1,
        "torch_interop_one_thread": torch.get_num_interop_threads() == 1,
        "blas_environment_one_thread": all(
            os.environ[name] == "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ),
        "gpyreg_commit": gpyreg_commit == EXPECTED_GPYREG_COMMIT,
    }
    evidence = {
        "checks": checks,
        "all_passed": all(checks.values()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "import_seconds": modules["import_seconds"],
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "svbmc": modules["svbmc"].__version__,
        },
        "imports": {
            "numpy": str(Path(np.__file__).resolve()),
            "scipy": str(Path(scipy.__file__).resolve()),
            "torch": str(Path(torch.__file__).resolve()),
            "svbmc": str(svbmc_file),
            "pyvbmc": str(pyvbmc_file),
            "gpyreg": str(gpyreg_file),
            "numpy_core": str(Path(modules["core"].__file__).resolve()),
        },
        "git": {
            "upstream_commit": actual_commit,
            "upstream_expected_commit": EXPECTED_UPSTREAM_COMMIT,
            "upstream_status": source_status,
            "gpyreg_commit": gpyreg_commit,
            "gpyreg_expected_commit": EXPECTED_GPYREG_COMMIT,
            "pyvbmc_commit": git_output(REPO_ROOT, "rev-parse", "HEAD"),
            "pyvbmc_branch": git_output(REPO_ROOT, "branch", "--show-current"),
        },
        "hashes": {
            "upstream_source": actual_source_hash,
            "upstream_source_expected": EXPECTED_SOURCE_SHA256,
            "corpus": corpus_hashes,
        },
        "threads": {
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
        | {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "dtype_policy": {
            "normal_upstream_torch_default": modules[
                "normal_torch_default_dtype"
            ],
            "comparison_torch_default": str(torch.get_default_dtype()),
            "upstream_optimize_arithmetic": (
                "float64 because initial weights originate in NumPy float64"
            ),
            "normal_upstream_return": (
                "maximize_ELBO casts best weights to the global default float32; "
                "optimize widens the rounded values back to NumPy float64"
            ),
            "comparison_return": (
                "the explicit float64 global default prevents that final float32 "
                "round-trip and makes non-Tensor stacked_ELBO inputs float64"
            ),
            "normal_return_probe_dtype": str(
                torch.ones(1, dtype=torch.float64).to(torch.float32).dtype
            ),
            "comparison_return_probe_dtype": str(
                torch.ones(1, dtype=torch.float64)
                .to(torch.get_default_dtype())
                .dtype
            ),
        },
    }
    return evidence


def load_group(source: Path, group: str) -> list[Any]:
    root = source.resolve() / "vbmc_runs" / group
    paths = sorted(root.glob(GROUPS[group]))
    if len(paths) != 10:
        raise RuntimeError(
            f"{group}: expected 10 posterior files, found {len(paths)}"
        )
    values = []
    for path in paths:
        with path.open("rb") as stream:
            values.append(pickle.load(stream))
    return values


def make_stacker(
    args: argparse.Namespace, modules: dict[str, Any], group: str
) -> Any:
    with contextlib.redirect_stdout(_NullWriter()):
        return modules["SVBMC"](load_group(args.source, group), testing=False)


def seed_all(seed: int, np: Any, torch: Any) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_upstream_entropy(
    stacker: Any,
    weights: Any,
    n_samples: int,
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    """Capture unchanged upstream's NumPy matrix at its Torch boundary."""

    original_tensor = torch.tensor
    captured: dict[str, Any] = {}
    call_started = time.perf_counter()

    def tensor_wrapper(data: Any, *positional: Any, **keywords: Any) -> Any:
        is_matrix = (
            not captured
            and isinstance(data, np.ndarray)
            and data.ndim == 2
            and data.shape[0] == int(np.sum(stacker.K)) * n_samples
            and data.shape[1] == int(np.sum(stacker.K))
        )
        if is_matrix:
            captured["preparation_seconds"] = (
                time.perf_counter() - call_started
            )
            copy_started = time.perf_counter()
            captured["logq_matrix"] = np.array(
                data, dtype=np.float64, copy=True
            )
            captured["instrumentation_copy_seconds"] = (
                time.perf_counter() - copy_started
            )
            conversion_started = time.perf_counter()
            result = original_tensor(data, *positional, **keywords)
            captured["conversion_seconds"] = (
                time.perf_counter() - conversion_started
            )
            return result
        return original_tensor(data, *positional, **keywords)

    torch.tensor = tensor_wrapper
    try:
        entropy, jacobian = stacker.stacked_entropy(
            weights, n_samples=n_samples
        )
    finally:
        torch.tensor = original_tensor
    total = time.perf_counter() - call_started
    if "logq_matrix" not in captured:
        raise RuntimeError("failed to capture upstream log-density matrix")
    captured.update(
        {
            "entropy": float(entropy.detach().cpu()),
            "jacobian": np.asarray(jacobian, dtype=np.float64),
            "total_seconds": total,
            "post_conversion_forward_seconds": total
            - captured["preparation_seconds"]
            - captured["conversion_seconds"]
            - captured["instrumentation_copy_seconds"],
        }
    )
    return captured


def torch_weight_objective(
    weights: Any,
    logq_tensor: Any,
    corrected_tensor: Any,
    n_samples: int,
    torch: Any,
    vectorized: bool,
) -> tuple[Any, Any]:
    component_count = logq_tensor.shape[1]
    normalized = weights / weights.sum()
    log_mixture = torch.logsumexp(
        logq_tensor + torch.log(normalized.reshape(1, -1) + 1e-40), dim=1
    )
    if vectorized:
        component_means = log_mixture.reshape(component_count, n_samples).mean(
            dim=1
        )
    else:
        component_index = (
            torch.arange(component_count, device=logq_tensor.device)
            .repeat_interleave(n_samples)
            .cpu()
            .numpy()
        )
        sums = torch.zeros(
            component_count, dtype=weights.dtype, device=weights.device
        )
        counts = torch.zeros(
            component_count, dtype=weights.dtype, device=weights.device
        )
        for component in range(component_count):
            mask = component_index == component
            count = mask.sum()
            if count > 0:
                sums[component] = log_mixture[mask].sum()
                counts[component] = count
        component_means = sums / (counts + 1e-40)
    entropy = -(normalized.reshape(1, -1) @ component_means).squeeze()
    expected_log_joint = (weights @ corrected_tensor).squeeze()
    return expected_log_joint + entropy, entropy


def torch_logits_objective(
    logits: Any,
    base_weights: Any,
    counts: list[int],
    version: str,
    logq_tensor: Any,
    corrected_tensor: Any,
    n_samples: int,
    torch: Any,
    vectorized: bool,
) -> tuple[Any, Any, Any]:
    if version == "all-weights":
        weight_logits = logits
    else:
        repeats = torch.tensor(counts, dtype=torch.int64, device=logits.device)
        weight_logits = torch.repeat_interleave(
            logits, repeats=repeats
        ) + torch.log(base_weights)
    weights = torch.softmax(weight_logits, dim=-1)
    objective, entropy = torch_weight_objective(
        weights,
        logq_tensor,
        corrected_tensor,
        n_samples,
        torch,
        vectorized,
    )
    return objective, entropy, weights


def initial_logits_reference(stacker: Any, version: str, np: Any) -> Any:
    if version == "all-weights":
        broadcasted = np.concatenate(
            [
                np.full(stacker.K[index], stacker.individual_elbos[index])
                for index in range(stacker.M)
            ]
        )
        logits = (
            np.log(np.asarray(stacker.w, dtype=np.float64).ravel())
            + broadcasted
        )
    else:
        logits = np.asarray(stacker.individual_elbos, dtype=np.float64)
    return logits - np.max(logits)


def optimizer_parity_check(modules: dict[str, Any]) -> dict[str, Any]:
    """Compare ten stateful NumPy Adam steps with real torch.optim.Adam."""
    np, torch, core = modules["np"], modules["torch"], modules["core"]
    initial = np.asarray([0.25, -2.0, 12.0, -20.0, 1e-9], dtype=np.float64)
    gradients = np.asarray(
        [
            [1e-12, -1e-12, 3.0, -4.0, 0.25],
            [0.5, -0.25, -2.0, 8e-12, -1.5],
            [-3.0, 4.0, 1e-10, -2e-10, 0.75],
            [7.0, -8.0, 0.125, -0.0625, 1e-12],
            [-1e-12, 2e-12, -0.5, 0.25, -6.0],
            [0.03125, -0.015625, 5.0, -4.0, 3.0],
            [-9.0, 1e-12, -1e-12, 2.0, -0.125],
            [1.25, -2.5, 3.75, -5.0, 6.25],
            [-0.75, 0.5, -0.25, 0.125, -0.0625],
            [1e-12, 4.0, -3.0, 2.0, -1.0],
        ],
        dtype=np.float64,
    )
    np_parameters = initial.copy()
    first = np.zeros_like(initial)
    second = np.zeros_like(initial)
    torch_parameters = torch.tensor(
        initial, dtype=torch.float64, requires_grad=True
    )
    torch_optimizer = torch.optim.Adam([torch_parameters], lr=0.1)
    numpy_trajectory = []
    torch_trajectory = []
    moment_checks = []
    for step, gradient in enumerate(gradients, start=1):
        np_parameters, first, second = core._adam_step(
            np_parameters, gradient, first, second, step, 0.1
        )
        torch_optimizer.zero_grad()
        torch_parameters.grad = torch.tensor(gradient, dtype=torch.float64)
        torch_optimizer.step()
        torch_state = torch_optimizer.state[torch_parameters]
        numpy_trajectory.append(np_parameters.copy())
        torch_trajectory.append(torch_parameters.detach().cpu().numpy().copy())
        moment_checks.append(
            {
                "step": step,
                "first_moment": compare_arrays(
                    first,
                    torch_state["exp_avg"].detach().cpu().numpy(),
                    TOLERANCES["fixed_gradient"],
                    np,
                ),
                "second_moment": compare_arrays(
                    second,
                    torch_state["exp_avg_sq"].detach().cpu().numpy(),
                    TOLERANCES["fixed_gradient"],
                    np,
                ),
            }
        )
    return {
        "description": (
            "Ten supplied nonconstant gradients include 1e-12 entries to "
            "exercise Adam's epsilon and accumulated moment state."
        ),
        "parameter_trajectory": compare_arrays(
            numpy_trajectory,
            torch_trajectory,
            TOLERANCES["fixed_gradient"],
            np,
        ),
        "moment_state_each_step": moment_checks,
    }


def fixed_objective_checks(
    stacker: Any,
    logq_matrix: Any,
    jacobian: Any,
    sampling_state: tuple[Any, ...],
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> dict[str, Any]:
    np, torch, core = modules["np"], modules["torch"], modules["core"]
    corrected = (np.asarray(stacker.I, dtype=np.float64) - jacobian).ravel()
    base_weights = np.asarray(stacker.w, dtype=np.float64).ravel()
    # A deterministic interior point exercises normalization and every component.
    perturbation = np.linspace(-0.15, 0.15, base_weights.size)
    direct_weights = base_weights * np.exp(perturbation)
    direct_weights *= 1.37

    logq_tensor = torch.tensor(logq_matrix, dtype=torch.float64)
    corrected_tensor = torch.tensor(corrected, dtype=torch.float64)
    direct_tensor = torch.tensor(
        direct_weights, dtype=torch.float64, requires_grad=True
    )
    upstream_value, upstream_entropy = torch_weight_objective(
        direct_tensor,
        logq_tensor,
        corrected_tensor,
        args.n_samples,
        torch,
        vectorized=False,
    )
    upstream_value.backward()
    upstream_gradient = direct_tensor.grad.detach().cpu().numpy().copy()

    numpy_value, numpy_entropy, numpy_gradient = core.objective_weights(
        direct_weights,
        logq_matrix,
        corrected,
        args.n_samples,
    )

    # This gate calls the pinned implementation itself. Resetting the exact
    # pre-draw state makes it consume the samples represented by logq_matrix.
    np.random.set_state(sampling_state)
    actual_upstream_weights = torch.tensor(
        direct_weights, dtype=torch.float64, requires_grad=True
    )
    actual_upstream_value, actual_upstream_entropy = stacker.stacked_ELBO(
        actual_upstream_weights, n_samples=args.n_samples
    )
    actual_upstream_value.backward()

    vector_weights = torch.tensor(
        direct_weights, dtype=torch.float64, requires_grad=True
    )
    vector_value, vector_entropy = torch_weight_objective(
        vector_weights,
        logq_tensor,
        corrected_tensor,
        args.n_samples,
        torch,
        vectorized=True,
    )
    vector_value.backward()

    result: dict[str, Any] = {
        "direct_weights": {
            "numpy_vs_actual_upstream_value": compare_arrays(
                numpy_value,
                float(actual_upstream_value.detach()),
                TOLERANCES["fixed_objective"],
                np,
            ),
            "numpy_vs_actual_upstream_entropy": compare_arrays(
                numpy_entropy,
                float(actual_upstream_entropy.detach()),
                TOLERANCES["fixed_objective"],
                np,
            ),
            "numpy_vs_actual_upstream_gradient": compare_arrays(
                numpy_gradient,
                actual_upstream_weights.grad.detach().cpu().numpy(),
                TOLERANCES["fixed_gradient"],
                np,
            ),
            "numpy_vs_upstream_style_loop_control_value": compare_arrays(
                numpy_value,
                float(upstream_value.detach()),
                TOLERANCES["fixed_objective"],
                np,
            ),
            "numpy_vs_upstream_style_loop_control_entropy": compare_arrays(
                numpy_entropy,
                float(upstream_entropy.detach()),
                TOLERANCES["fixed_objective"],
                np,
            ),
            "numpy_vs_upstream_style_loop_control_gradient": compare_arrays(
                numpy_gradient,
                upstream_gradient,
                TOLERANCES["fixed_gradient"],
                np,
            ),
            "vector_torch_vs_upstream_style_loop_control_value": compare_arrays(
                float(vector_value.detach()),
                float(upstream_value.detach()),
                TOLERANCES["fixed_objective"],
                np,
            ),
            "vector_torch_vs_upstream_style_loop_control_entropy": compare_arrays(
                float(vector_entropy.detach()),
                float(upstream_entropy.detach()),
                TOLERANCES["fixed_objective"],
                np,
            ),
            "vector_torch_vs_upstream_style_loop_control_gradient": compare_arrays(
                vector_weights.grad.detach().cpu().numpy(),
                upstream_gradient,
                TOLERANCES["fixed_gradient"],
                np,
            ),
        },
        "versions": {},
    }

    for version in args.versions:
        numpy_initial = np.asarray(
            core.initial_logits(stacker, version), dtype=np.float64
        )
        reference_initial = initial_logits_reference(stacker, version, np)
        result["versions"][version] = {
            "initial_logits": compare_arrays(
                numpy_initial,
                reference_initial,
                TOLERANCES["fixed_gradient"],
                np,
            ),
            "cases": {},
        }
        cases = {
            "upstream_initial": reference_initial,
            "moderate": np.linspace(-4.0, 4.0, reference_initial.size),
            "extreme": np.linspace(-80.0, 80.0, reference_initial.size),
        }
        for case_name, case_logits in cases.items():
            (
                numpy_elbo,
                numpy_h,
                numpy_grad,
                numpy_weights,
            ) = core.objective_logits(
                case_logits,
                base_weights,
                stacker.K,
                version,
                logq_matrix,
                corrected,
                args.n_samples,
            )
            logits_tensor = torch.tensor(
                case_logits, dtype=torch.float64, requires_grad=True
            )
            torch_elbo, torch_h, torch_weights = torch_logits_objective(
                logits_tensor,
                torch.tensor(base_weights, dtype=torch.float64),
                list(stacker.K),
                version,
                logq_tensor,
                corrected_tensor,
                args.n_samples,
                torch,
                vectorized=False,
            )
            torch_elbo.backward()
            result["versions"][version]["cases"][case_name] = {
                "elbo": compare_arrays(
                    numpy_elbo,
                    float(torch_elbo.detach()),
                    TOLERANCES["fixed_objective"],
                    np,
                ),
                "entropy": compare_arrays(
                    numpy_h,
                    float(torch_h.detach()),
                    TOLERANCES["fixed_objective"],
                    np,
                ),
                "gradient": compare_arrays(
                    numpy_grad,
                    logits_tensor.grad.detach().cpu().numpy(),
                    TOLERANCES["fixed_gradient"],
                    np,
                ),
                "weights": compare_arrays(
                    numpy_weights,
                    torch_weights.detach().cpu().numpy(),
                    TOLERANCES["fixed_gradient"],
                    np,
                ),
            }
    return result


def time_fixed_objectives(
    stacker: Any,
    logq_matrix: Any,
    jacobian: Any,
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> dict[str, Any]:
    np, torch, core = modules["np"], modules["torch"], modules["core"]
    corrected = (np.asarray(stacker.I, dtype=np.float64) - jacobian).ravel()
    base = np.asarray(stacker.w, dtype=np.float64).ravel()

    conversion_times = []
    for _ in range(args.kernel_repeats):
        started = time.perf_counter()
        torch.tensor(logq_matrix, dtype=torch.float64)
        torch.tensor(corrected, dtype=torch.float64)
        conversion_times.append(time.perf_counter() - started)

    logq_tensor = torch.tensor(logq_matrix, dtype=torch.float64)
    corrected_tensor = torch.tensor(corrected, dtype=torch.float64)
    base_tensor = torch.tensor(base, dtype=torch.float64)
    output: dict[str, Any] = {
        "torch_conversion": timing_summary(conversion_times)
    }

    for version in args.versions:
        initial = np.asarray(
            core.initial_logits(stacker, version), dtype=np.float64
        )
        # Warm every implementation once; these calls are not measured.
        core.objective_logits(
            initial,
            base,
            stacker.K,
            version,
            logq_matrix,
            corrected,
            args.n_samples,
        )
        for vectorized in (False, True):
            warm_logits = torch.tensor(
                initial, dtype=torch.float64, requires_grad=True
            )
            warm_value, _, _ = torch_logits_objective(
                warm_logits,
                base_tensor,
                list(stacker.K),
                version,
                logq_tensor,
                corrected_tensor,
                args.n_samples,
                torch,
                vectorized,
            )
            warm_value.backward()

        numpy_objective_times: list[float] = []
        numpy_adam_times: list[float] = []
        parameters = initial.copy()
        first = np.zeros_like(parameters)
        second = np.zeros_like(parameters)
        for step in range(1, args.kernel_repeats + 1):
            started = time.perf_counter()
            _, _, gradient, _ = core.objective_logits(
                parameters,
                base,
                stacker.K,
                version,
                logq_matrix,
                corrected,
                args.n_samples,
            )
            numpy_objective_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            parameters, first, second = core._adam_step(
                parameters, -gradient, first, second, step, args.lr
            )
            numpy_adam_times.append(time.perf_counter() - started)

        version_timing: dict[str, Any] = {
            "numpy_objective_and_analytic_gradient": timing_summary(
                numpy_objective_times
            ),
            "numpy_adam": timing_summary(numpy_adam_times),
        }
        for label, vectorized in (
            ("upstream_style_loop_torch_control", False),
            ("vectorized_torch_control", True),
        ):
            forward_times: list[float] = []
            backward_times: list[float] = []
            adam_times: list[float] = []
            logits = torch.tensor(
                initial, dtype=torch.float64, requires_grad=True
            )
            optimizer = torch.optim.Adam([logits], lr=args.lr)
            for _ in range(args.kernel_repeats):
                optimizer.zero_grad()
                started = time.perf_counter()
                value, _, _ = torch_logits_objective(
                    logits,
                    base_tensor,
                    list(stacker.K),
                    version,
                    logq_tensor,
                    corrected_tensor,
                    args.n_samples,
                    torch,
                    vectorized,
                )
                loss = -value
                forward_times.append(time.perf_counter() - started)
                started = time.perf_counter()
                loss.backward()
                backward_times.append(time.perf_counter() - started)
                started = time.perf_counter()
                optimizer.step()
                adam_times.append(time.perf_counter() - started)
            version_timing[label] = {
                "objective_forward": timing_summary(forward_times),
                "backward": timing_summary(backward_times),
                "adam": timing_summary(adam_times),
                "forward_plus_backward": timing_summary(
                    [
                        forward + backward
                        for forward, backward in zip(
                            forward_times, backward_times
                        )
                    ]
                ),
                "forward_plus_backward_seconds_per_sample": timing_summary(
                    [
                        (forward + backward) / logq_matrix.shape[0]
                        for forward, backward in zip(
                            forward_times, backward_times
                        )
                    ]
                ),
            }
        output[version] = version_timing
    return output


def run_kernel_case(
    group: str,
    seed: int,
    args: argparse.Namespace,
    modules: dict[str, Any],
    measure: bool,
) -> dict[str, Any]:
    np, torch, core = modules["np"], modules["torch"], modules["core"]
    stacker = make_stacker(args, modules, group)
    weights = torch.tensor(
        np.asarray(stacker.w, dtype=np.float64).ravel(), dtype=torch.float64
    )

    seed_all(seed, np, torch)
    initial_state = np.random.get_state()
    upstream = capture_upstream_entropy(
        stacker, weights, args.n_samples, np, torch
    )
    upstream_state = np.random.get_state()

    np.random.set_state(initial_state)
    numpy_started = time.perf_counter()
    numpy_logq, numpy_jacobian = core.prepare_entropy(stacker, args.n_samples)
    numpy_seconds = time.perf_counter() - numpy_started
    numpy_state = np.random.get_state()

    logq_check = compare_arrays(
        numpy_logq,
        upstream["logq_matrix"],
        TOLERANCES["preparation_logq"],
        np,
    )
    jacobian_check = compare_arrays(
        numpy_jacobian,
        upstream["jacobian"],
        TOLERANCES["jacobian_correction"],
        np,
    )
    result: dict[str, Any] = {
        "group": group,
        "seed": seed,
        "shape": list(np.asarray(numpy_logq).shape),
        "n_samples_per_component": args.n_samples,
        "matrix_sha256": {
            "upstream": array_digest(upstream["logq_matrix"], np),
            "numpy": array_digest(numpy_logq, np),
        },
        "jacobian_sha256": {
            "upstream": array_digest(upstream["jacobian"], np),
            "numpy": array_digest(numpy_jacobian, np),
        },
        "preparation_correctness": {
            "logq_matrix": logq_check,
            "jacobian": jacobian_check,
            "rng_state_equal": rng_states_equal(
                upstream_state, numpy_state, np
            ),
            "upstream_final_rng_sha256": rng_state_digest(upstream_state, np),
            "numpy_final_rng_sha256": rng_state_digest(numpy_state, np),
        },
        "fixed_objective_correctness": fixed_objective_checks(
            stacker,
            upstream["logq_matrix"],
            upstream["jacobian"],
            initial_state,
            args,
            modules,
        ),
        "single_call_preparation_seconds": {
            "upstream_numpy_before_torch_boundary": upstream[
                "preparation_seconds"
            ],
            "upstream_torch_conversion": upstream["conversion_seconds"],
            "capture_copy_overhead_excluded": upstream[
                "instrumentation_copy_seconds"
            ],
            "upstream_post_conversion_forward": upstream[
                "post_conversion_forward_seconds"
            ],
            "optimized_numpy": numpy_seconds,
            "note": (
                "The upstream split is diagnostic instrumentation at the "
                "unchanged torch.tensor boundary; clean complete-fit timings are "
                "reported separately."
            ),
        },
    }

    if measure:
        upstream_prep_times = []
        upstream_conversion_times = []
        upstream_reduction_times = []
        numpy_prep_times = []
        preparation_order = []
        for repeat in range(args.prep_repeats):
            order = ["upstream", "numpy"]
            if repeat % 2:
                order.reverse()
            preparation_order.append(order)
            for backend in order:
                seed_all(seed, np, torch)
                if backend == "upstream":
                    captured = capture_upstream_entropy(
                        stacker, weights, args.n_samples, np, torch
                    )
                    upstream_prep_times.append(captured["preparation_seconds"])
                    upstream_conversion_times.append(
                        captured["conversion_seconds"]
                    )
                    upstream_reduction_times.append(
                        captured["post_conversion_forward_seconds"]
                    )
                else:
                    started = time.perf_counter()
                    core.prepare_entropy(stacker, args.n_samples)
                    numpy_prep_times.append(time.perf_counter() - started)
        result["preparation_timings"] = {
            "balanced_order": preparation_order,
            "upstream_numpy": timing_summary(upstream_prep_times),
            "upstream_torch_conversion": timing_summary(
                upstream_conversion_times
            ),
            "upstream_torch_reduction": timing_summary(
                upstream_reduction_times
            ),
            "optimized_numpy": timing_summary(numpy_prep_times),
        }
        result["fixed_matrix_timings"] = time_fixed_objectives(
            stacker,
            upstream["logq_matrix"],
            upstream["jacobian"],
            args,
            modules,
        )

    del numpy_logq
    del upstream["logq_matrix"]
    return result


def normalize_fit_result(result: dict[str, Any], np: Any) -> dict[str, Any]:
    return {
        "weights": np.asarray(result["weights"], dtype=np.float64).ravel(),
        "elbo": float(result["elbo"]),
        "entropy": float(result["entropy"]),
        "elbo_debiased_I_median": float(result["elbo_debiased_I_median"]),
        "elbo_debiased_E_median": float(result["elbo_debiased_E_median"]),
        "steps": int(result["steps"]),
        "stopped": bool(result["stopped"]),
        "best_iteration": int(result["best_iteration"]),
        "version": result["version"],
        "timings": result.get("timings"),
        **({"trace": result["trace"]} if "trace" in result else {}),
    }


def run_numpy_fit(
    group: str,
    seed: int,
    version: str,
    max_steps: int,
    args: argparse.Namespace,
    modules: dict[str, Any],
    trace: bool,
) -> dict[str, Any]:
    np, torch, core = modules["np"], modules["torch"], modules["core"]
    construct_wall_started = time.perf_counter()
    construct_cpu_started = time.process_time()
    stacker = make_stacker(args, modules, group)
    construct_cpu_seconds = time.process_time() - construct_cpu_started
    construct_wall_seconds = time.perf_counter() - construct_wall_started
    seed_all(seed, np, torch)
    rng_before = np.random.get_state()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    result = core.optimize_numpy(
        stacker,
        n_samples=args.n_samples,
        lr=args.lr,
        max_steps=max_steps,
        version=version,
        trace=trace,
    )
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    rng_after = np.random.get_state()
    normalized = normalize_fit_result(result, np)
    normalized.update(
        {
            "backend": "numpy",
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "load_and_construction_wall_seconds": construct_wall_seconds,
            "load_and_construction_cpu_seconds": construct_cpu_seconds,
            "rng_before_sha256": rng_state_digest(rng_before, np),
            "rng_after_sha256": rng_state_digest(rng_after, np),
        }
    )
    return normalized


def upstream_result(stacker: Any, np: Any) -> dict[str, Any]:
    return {
        "weights": np.asarray(stacker.w, dtype=np.float64).ravel(),
        "elbo": float(stacker.elbo["estimated"]),
        "entropy": float(stacker.entropy),
        "elbo_debiased_I_median": float(stacker.elbo["debiased_I_median"]),
        "elbo_debiased_E_median": float(stacker.elbo["debiased_E_median"]),
    }


def run_upstream_fit(
    group: str,
    seed: int,
    version: str,
    max_steps: int,
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> dict[str, Any]:
    np, torch = modules["np"], modules["torch"]
    construct_wall_started = time.perf_counter()
    construct_cpu_started = time.process_time()
    stacker = make_stacker(args, modules, group)
    construct_cpu_seconds = time.process_time() - construct_cpu_started
    construct_wall_seconds = time.perf_counter() - construct_wall_started
    seed_all(seed, np, torch)
    rng_before = np.random.get_state()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    with contextlib.redirect_stdout(_NullWriter()):
        stacker.optimize(
            n_samples=args.n_samples,
            lr=args.lr,
            max_steps=max_steps,
            version=version,
        )
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    rng_after = np.random.get_state()
    result = upstream_result(stacker, np)
    result.update(
        {
            "backend": "upstream_torch",
            "steps": None,
            "stopped": None,
            "best_iteration": None,
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "load_and_construction_wall_seconds": construct_wall_seconds,
            "load_and_construction_cpu_seconds": construct_cpu_seconds,
            "rng_before_sha256": rng_state_digest(rng_before, np),
            "rng_after_sha256": rng_state_digest(rng_after, np),
            "note": (
                "This is an uninstrumented timing fit. Step and best-iterate "
                "evidence comes from the separate diagnostic fit."
            ),
        }
    )
    return result


def compare_fit_results(
    numpy_fit: dict[str, Any], torch_fit: dict[str, Any], np: Any
) -> dict[str, Any]:
    scalar_fields = (
        "elbo",
        "entropy",
        "elbo_debiased_I_median",
        "elbo_debiased_E_median",
    )
    comparison = {
        field: compare_arrays(
            numpy_fit[field], torch_fit[field], TOLERANCES["fit_scalar"], np
        )
        for field in scalar_fields
    }
    comparison["weights"] = compare_arrays(
        numpy_fit["weights"],
        torch_fit["weights"],
        TOLERANCES["fit_weights"],
        np,
    )
    comparison["final_rng_state_equal"] = (
        numpy_fit["rng_after_sha256"] == torch_fit["rng_after_sha256"]
    )
    comparison["all_numeric_passed"] = all(
        item["passed"]
        for item in comparison.values()
        if isinstance(item, dict)
    )
    return comparison


def rescore_weights_on_identical_draws(
    group: str,
    seed: int,
    numpy_weights: Any,
    torch_weights: Any,
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> dict[str, Any]:
    np, torch, core = modules["np"], modules["torch"], modules["core"]
    stacker = make_stacker(args, modules, group)
    seed_all(seed, np, torch)
    captured = capture_upstream_entropy(
        stacker,
        torch.tensor(np.asarray(stacker.w).ravel(), dtype=torch.float64),
        args.n_samples,
        np,
        torch,
    )
    matrix = captured["logq_matrix"]
    corrected = (np.asarray(stacker.I) - captured["jacobian"]).ravel()
    matrix_tensor = torch.tensor(matrix, dtype=torch.float64)
    corrected_tensor = torch.tensor(corrected, dtype=torch.float64)
    result: dict[str, Any] = {
        "draw_matrix_sha256": array_digest(matrix, np),
        "weights": {},
    }
    for label, weights in (
        ("numpy_fit", numpy_weights),
        ("upstream_fit", torch_weights),
    ):
        weights_array = np.asarray(weights, dtype=np.float64).ravel()
        np_elbo, np_entropy, _ = core.objective_weights(
            weights_array, matrix, corrected, args.n_samples
        )
        weights_tensor = torch.tensor(
            weights_array, dtype=torch.float64, requires_grad=True
        )
        t_elbo, t_entropy = torch_weight_objective(
            weights_tensor,
            matrix_tensor,
            corrected_tensor,
            args.n_samples,
            torch,
            vectorized=True,
        )
        result["weights"][label] = {
            "numpy_elbo": float(np_elbo),
            "vector_torch_elbo": float(t_elbo.detach()),
            "numpy_entropy": float(np_entropy),
            "vector_torch_entropy": float(t_entropy.detach()),
            "elbo_backend_check": compare_arrays(
                np_elbo,
                float(t_elbo.detach()),
                TOLERANCES["fixed_objective"],
                np,
            ),
        }
    return result


def run_clean_fit_pair(
    group: str,
    seed: int,
    version: str,
    max_steps: int,
    pair_index: int,
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> dict[str, Any]:
    np = modules["np"]
    order = ["numpy", "upstream_torch"]
    if pair_index % 2:
        order.reverse()
    fits: dict[str, Any] = {}
    for backend in order:
        if backend == "numpy":
            fits[backend] = run_numpy_fit(
                group, seed, version, max_steps, args, modules, trace=False
            )
        else:
            fits[backend] = run_upstream_fit(
                group, seed, version, max_steps, args, modules
            )
    comparison = compare_fit_results(fits["numpy"], fits["upstream_torch"], np)
    result: dict[str, Any] = {
        "group": group,
        "seed": seed,
        "version": version,
        "max_steps": max_steps,
        "execution_order": order,
        "fits": fits,
        "comparison": comparison,
    }
    if not comparison["all_numeric_passed"]:
        result["independent_rescore"] = rescore_weights_on_identical_draws(
            group,
            seed,
            fits["numpy"]["weights"],
            fits["upstream_torch"]["weights"],
            args,
            modules,
        )
    return result


def instrumented_upstream_fit(
    group: str,
    seed: int,
    version: str,
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> dict[str, Any]:
    np, torch = modules["np"], modules["torch"]
    stacker = make_stacker(args, modules, group)
    original = stacker.stacked_ELBO
    trace: dict[str, list[Any]] = {
        "iteration": [],
        "elbo": [],
        "entropy": [],
        "rounded_loss": [],
        "convergence_counter": [],
        "is_best": [],
        "weights": [],
        "rng_state_sha256": [],
    }
    loss_old = 1e8
    convergence_counter = 0
    best_iteration = -1

    def wrapper(weights: Any, n_samples: int = 20) -> tuple[Any, Any]:
        nonlocal loss_old, convergence_counter, best_iteration
        elbo, entropy = original(weights, n_samples=n_samples)
        rounded = float((torch.round((-elbo) * 1e5) / 1e5).detach())
        is_best = rounded < loss_old
        if is_best:
            convergence_counter = 0
            loss_old = rounded
            best_iteration = len(trace["iteration"])
        else:
            convergence_counter += 1
        trace["iteration"].append(len(trace["iteration"]))
        trace["elbo"].append(float(elbo.detach()))
        trace["entropy"].append(float(entropy.detach()))
        trace["rounded_loss"].append(rounded)
        trace["convergence_counter"].append(convergence_counter)
        trace["is_best"].append(is_best)
        trace["weights"].append(
            weights.detach().cpu().numpy().reshape(-1).tolist()
        )
        trace["rng_state_sha256"].append(
            rng_state_digest(np.random.get_state(), np)
        )
        return elbo, entropy

    stacker.stacked_ELBO = wrapper
    seed_all(seed, np, torch)
    before = np.random.get_state()
    with contextlib.redirect_stdout(_NullWriter()):
        stacker.optimize(
            n_samples=args.n_samples,
            lr=args.lr,
            max_steps=args.max_steps,
            version=version,
        )
    after = np.random.get_state()
    result = upstream_result(stacker, np)
    result.update(
        {
            "backend": "upstream_torch_instrumented",
            "steps": len(trace["iteration"]),
            "stopped": bool(
                trace["convergence_counter"]
                and trace["convergence_counter"][-1] >= 5
            ),
            "best_iteration": best_iteration,
            "trace": trace,
            "rng_before_sha256": rng_state_digest(before, np),
            "rng_after_sha256": rng_state_digest(after, np),
        }
    )
    return result


def instrumented_numpy_fit(
    group: str,
    seed: int,
    version: str,
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> dict[str, Any]:
    np, torch, core = modules["np"], modules["torch"], modules["core"]
    stacker = make_stacker(args, modules, group)
    original_prepare = core.prepare_entropy
    rng_states: list[str] = []
    matrix_hashes: list[str] = []

    def prepare_wrapper(subject: Any, n_samples: int) -> tuple[Any, Any]:
        matrix, jacobian = original_prepare(subject, n_samples)
        rng_states.append(rng_state_digest(np.random.get_state(), np))
        matrix_hashes.append(array_digest(matrix, np))
        return matrix, jacobian

    seed_all(seed, np, torch)
    before = np.random.get_state()
    core.prepare_entropy = prepare_wrapper
    try:
        raw = core.optimize_numpy(
            stacker,
            n_samples=args.n_samples,
            lr=args.lr,
            max_steps=args.max_steps,
            version=version,
            trace=True,
        )
    finally:
        core.prepare_entropy = original_prepare
    after = np.random.get_state()
    result = normalize_fit_result(raw, np)
    result.update(
        {
            "backend": "numpy_instrumented",
            "rng_before_sha256": rng_state_digest(before, np),
            "rng_after_sha256": rng_state_digest(after, np),
            "per_iteration_rng_state_sha256": rng_states,
            "per_iteration_matrix_sha256": matrix_hashes,
        }
    )
    return result


def compare_diagnostic_traces(
    numpy_fit: dict[str, Any], upstream_fit: dict[str, Any], np: Any
) -> dict[str, Any]:
    np_trace = numpy_fit["trace"]
    torch_trace = upstream_fit["trace"]
    shared = min(numpy_fit["steps"], upstream_fit["steps"])
    comparisons = {
        field: compare_arrays(
            np.asarray(np_trace[field])[:shared],
            np.asarray(torch_trace[field])[:shared],
            TOLERANCES[
                "trace_weights" if field == "weights" else "trace_scalar"
            ],
            np,
        )
        for field in ("elbo", "entropy", "rounded_loss", "weights")
    }
    numpy_rng = numpy_fit["per_iteration_rng_state_sha256"][:shared]
    torch_rng = torch_trace["rng_state_sha256"][:shared]
    return {
        "shared_iterations": shared,
        "steps_equal": numpy_fit["steps"] == upstream_fit["steps"],
        "stopped_equal": numpy_fit["stopped"] == upstream_fit["stopped"],
        "best_iteration_equal": (
            numpy_fit["best_iteration"] == upstream_fit["best_iteration"]
        ),
        "per_iteration": comparisons,
        "per_iteration_rng_all_equal": numpy_rng == torch_rng,
        "per_iteration_rng_equal": [
            left == right for left, right in zip(numpy_rng, torch_rng)
        ],
        "final_rng_equal": (
            numpy_fit["rng_after_sha256"] == upstream_fit["rng_after_sha256"]
        ),
    }


def run_diagnostic_pair(
    group: str,
    seed: int,
    version: str,
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> dict[str, Any]:
    numpy_fit = instrumented_numpy_fit(group, seed, version, args, modules)
    upstream_fit = instrumented_upstream_fit(
        group, seed, version, args, modules
    )
    comparison = compare_diagnostic_traces(
        numpy_fit, upstream_fit, modules["np"]
    )
    return {
        "group": group,
        "seed": seed,
        "version": version,
        "warning": (
            "Instrumentation captures every iterate and RNG state; these elapsed "
            "times are intentionally excluded from clean timing results."
        ),
        "numpy": numpy_fit,
        "upstream_torch": upstream_fit,
        "comparison": comparison,
    }


def warm_up(
    args: argparse.Namespace, modules: dict[str, Any], group: str, seed: int
) -> dict[str, Any]:
    started = time.perf_counter()
    # One correctness case warms SciPy transforms, NumPy kernels, Torch reductions,
    # autograd, and Adam. Its timings are discarded by callers.
    temporary_args = argparse.Namespace(**vars(args))
    temporary_args.versions = [args.versions[0]]
    temporary_args.kernel_repeats = 1
    temporary_args.prep_repeats = 1
    run_kernel_case(group, seed, temporary_args, modules, measure=False)
    return {
        "group": group,
        "seed": seed,
        "wall_seconds": time.perf_counter() - started,
        "excluded_from_measurements": True,
    }


def collect_validation_failures(
    value: Any, path: str = "results"
) -> list[str]:
    """Collect explicit numerical/equality failures without treating stopped=False as failure."""
    failures: list[str] = []
    if isinstance(value, dict):
        if "passed" in value and value["passed"] is False:
            failures.append(path)
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if (
                isinstance(item, bool)
                and item is False
                and (
                    key.endswith("_equal")
                    or key.endswith("_all_equal")
                    or key == "all_numeric_passed"
                )
            ):
                failures.append(child_path)
            elif isinstance(item, (dict, list)):
                failures.extend(collect_validation_failures(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(
                collect_validation_failures(item, f"{path}[{index}]")
            )
    return sorted(set(failures))


def checkpoint(report: dict[str, Any], output: Path) -> None:
    report["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(output, report)


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "phase": args.phase,
        "config": {
            "source": str(args.source.resolve()),
            "deps": str(args.deps.resolve()),
            "groups": args.groups,
            "seeds": args.seeds,
            "versions": args.versions,
            "n_samples": args.n_samples,
            "lr": args.lr,
            "max_steps": args.max_steps,
            "smoke_max_steps": args.smoke_max_steps,
            "prep_repeats": args.prep_repeats,
            "kernel_repeats": args.kernel_repeats,
            "diagnostic_seed": args.diagnostic_seed,
        },
        "tolerances_declared_before_measurement": TOLERANCES,
        "design": {
            "upstream": "pinned unchanged S-VBMC on CPU with explicit float64",
            "sampling": "ordinary fresh global-NumPy draws on every objective call",
            "fit_order": "backend order alternates across paired cells",
            "timing": (
                "perf_counter wall and process_time CPU; imports, warmup, and "
                "instrumented diagnostics excluded from clean complete fits"
            ),
            "kernel_controls": (
                "optimized NumPy analytic gradient, upstream-style Torch Python "
                "stratum loop, and vectorized Torch stratum reshape on one matrix"
            ),
            "failure_policy": (
                "all tolerance failures remain visible; mismatched fits receive "
                "an independent fixed-draw rescore"
            ),
        },
        "results": {"smoke": [], "kernels": [], "fits": [], "diagnostics": []},
        "errors": [],
    }
    report["frozen_experiment_sources"] = archive_experiment_sources(output)
    checkpoint(report, output)

    modules = import_backends(args)
    report["environment"] = verify_environment(args, modules)
    if not report["environment"]["all_passed"]:
        failed = [
            name
            for name, passed in report["environment"]["checks"].items()
            if not passed
        ]
        report["status"] = "failed_provenance"
        checkpoint(report, output)
        raise RuntimeError(
            "environment/provenance checks failed: " + ", ".join(failed)
        )

    report["status"] = "running"
    report["optimizer_parity"] = optimizer_parity_check(modules)
    checkpoint(report, output)
    report["warmup"] = warm_up(args, modules, args.groups[0], args.seeds[0])
    checkpoint(report, output)

    if args.phase == "smoke":
        kernel = run_kernel_case(
            args.groups[0], args.seeds[0], args, modules, measure=False
        )
        report["results"]["smoke"].append({"kernel": kernel})
        checkpoint(report, output)
        pair = run_clean_fit_pair(
            args.groups[0],
            args.seeds[0],
            args.versions[0],
            args.smoke_max_steps,
            0,
            args,
            modules,
        )
        report["results"]["smoke"].append({"paired_fit": pair})
        checkpoint(report, output)

    if args.phase in ("kernels", "all"):
        for group in args.groups:
            result = run_kernel_case(
                group, args.seeds[0], args, modules, measure=True
            )
            report["results"]["kernels"].append(result)
            checkpoint(report, output)

    if args.phase in ("fits", "all"):
        pair_index = 0
        for group in args.groups:
            for version in args.versions:
                for seed in args.seeds:
                    result = run_clean_fit_pair(
                        group,
                        seed,
                        version,
                        args.max_steps,
                        pair_index,
                        args,
                        modules,
                    )
                    report["results"]["fits"].append(result)
                    pair_index += 1
                    checkpoint(report, output)
        for group in args.groups:
            for version in args.versions:
                result = run_diagnostic_pair(
                    group,
                    args.diagnostic_seed,
                    version,
                    args,
                    modules,
                )
                report["results"]["diagnostics"].append(result)
                checkpoint(report, output)

    failures = collect_validation_failures(report["results"])
    failures.extend(
        collect_validation_failures(
            report["optimizer_parity"], "optimizer_parity"
        )
    )
    report["validation"] = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }
    report["status"] = "complete" if not failures else "complete_with_failures"
    report["completed_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    checkpoint(report, output)
    return 0 if not failures else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        output = args.output.resolve()
        if not isinstance(exc, FileExistsError) and output.exists():
            try:
                with output.open("r", encoding="utf-8") as stream:
                    report = json.load(stream)
                report["status"] = "failed_exception"
                report.setdefault("errors", []).append(
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                checkpoint(report, output)
            except Exception as checkpoint_error:
                print(
                    f"failed to checkpoint exception: {checkpoint_error}",
                    file=sys.stderr,
                )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
