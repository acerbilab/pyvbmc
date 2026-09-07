"""Prepare and verify the approved 150-run noisy reference extension.

The default invocation only prints help. ``prepare`` performs fail-closed
source, dependency, historical-population, allocation, and spawned-import
checks, then writes a launch manifest. PyVBMC is imported only in the isolated
probe; no optimization is run. A future run requires the frozen launcher's explicit
``run --confirm-run-150`` command. ``validate`` checks the isolated extension
output without publishing or changing the historical population.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_SHA = "7314a6afe158c5673775ca79601b88b3913ae383"
DEFAULT_GPYREG_SHA = "a2f8ddce867f502e29717959cf0ff3529f598618"
THREADS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
EXPECTED_VERSIONS = {
    "python": "3.12.6",
    "numpy": "2.5.2",
    "scipy": "1.18.1",
    "gpyreg": "1.1.0",
    "cma": "4.4.4",
}
CONFIGS = {
    "rosenbrock_D2_noise3": {
        "problem": "rosenbrock",
        "D": 2,
        "noise_sd": 3.0,
        "max_fun_evals": 200,
    },
    "student_D8_noise3": {
        "problem": "student",
        "D": 8,
        "noise_sd": 3.0,
        "max_fun_evals": 500,
    },
    "lumpy_D10_noise3": {
        "problem": "lumpy",
        "D": 10,
        "noise_sd": 3.0,
        "max_fun_evals": 600,
    },
}
SEEDS = tuple(range(50))
SOURCE_FILES = (
    "dev/scripts/benchmark_targets.py",
    "dev/scripts/golden_trace.py",
    "dev/scripts/profile_run.py",
)
ALLOWED_REFERENCE_DIFF = {"dev/scripts/benchmark_targets.py"}
SCHEMA = 1
ITERATION_ARRAYS = {
    "iter",
    "elbo",
    "elbo_sd",
    "sKL",
    "r_index",
    "stable",
    "warmup",
    "Ns_gp",
    "func_count",
    "n_eff",
    "pruned",
    "K",
    "N",
    "warped",
}
TRACE_ARRAYS = ITERATION_ARRAYS | {
    "timer",
    "pt_mu",
    "pt_delta",
    "pt_scale",
    "pt_R",
    "gp_hyp",
    "gp_hyp_iter",
    "vp_w",
    "vp_mu",
    "vp_sigma",
    "vp_iter",
    "vp_lambd",
    "X_orig",
    "y_orig",
    "X_init",
    "y_init",
    "final_w",
    "final_mu",
    "final_sigma",
    "final_lambd",
    "post_mean",
    "post_cov",
}


class PreparationError(RuntimeError):
    """A fail-closed preparation, launch, or validation error."""


def _resolved(path):
    return Path(path).expanduser().resolve()


def _git(path, *args):
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = "\n".join(
            part.strip()
            for part in (result.stdout, result.stderr)
            if part.strip()
        )
        raise PreparationError(detail)
    return result.stdout.strip()


def _git_state(path, expected_sha):
    head = _git(path, "rev-parse", "HEAD")
    if head != expected_sha:
        raise PreparationError(
            f"{path}: HEAD {head} does not equal expected {expected_sha}"
        )
    status = _git(path, "status", "--porcelain")
    if status:
        raise PreparationError(f"{path}: checkout is not clean:\n{status}")
    return {"head": head, "clean": True}


def _verify_reference_source(path, expected_sha):
    state = _git_state(path, expected_sha)
    ancestor = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={path}",
            "-C",
            str(path),
            "merge-base",
            "--is-ancestor",
            BASE_SHA,
            "HEAD",
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise PreparationError(
            f"reference HEAD is not based on required {BASE_SHA}"
        )
    count = int(_git(path, "rev-list", "--count", f"{BASE_SHA}..HEAD"))
    if count != 1:
        raise PreparationError(
            f"reference must be one additive commit after {BASE_SHA}; got {count}"
        )
    changed = set(
        filter(
            None,
            _git(
                path, "diff", "--name-only", f"{BASE_SHA}..HEAD"
            ).splitlines(),
        )
    )
    if changed != ALLOWED_REFERENCE_DIFF:
        raise PreparationError(
            "reference diff must contain only the benchmark registration; "
            f"got {sorted(changed)}"
        )
    state["base"] = BASE_SHA
    state["changed_files"] = sorted(changed)
    return state


def _sha256(path, normalize_text=False):
    data = Path(path).read_bytes()
    if normalize_text:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _verify_historical(population, historical_manifest):
    population = _resolved(population)
    manifest_path = _resolved(historical_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 810:
        raise PreparationError(
            "historical manifest must describe exactly 810 pairs"
        )
    expected = set(files)
    json_tags = {path.stem for path in population.glob("*_seed*.json")}
    npz_tags = {path.stem for path in population.glob("*_seed*.npz")}
    errors = sorted(path.name for path in population.glob("*.error.txt"))
    if errors:
        raise PreparationError(
            f"historical population has error files: {errors[:3]}"
        )
    if json_tags != expected or npz_tags != expected:
        raise PreparationError(
            "historical population allocation differs from its 810-pair manifest"
        )
    for tag, hashes in files.items():
        json_hash = _sha256(population / f"{tag}.json", normalize_text=True)
        npz_hash = _sha256(population / f"{tag}.npz")
        if (
            json_hash != hashes["json_sha256"]
            or npz_hash != hashes["npz_sha256"]
        ):
            raise PreparationError(f"historical hash mismatch for {tag}")
    return {
        "pairs": len(expected),
        "manifest_sha256": _sha256(manifest_path, normalize_text=True),
    }


def _require_under(module_file, root, name):
    actual = _resolved(module_file)
    root = _resolved(root)
    try:
        actual.relative_to(root)
    except ValueError as exc:
        raise PreparationError(
            f"{name} imported from {actual}, outside {root}"
        ) from exc
    return str(actual)


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise PreparationError(f"package {name!r} is not installed") from exc


def _runtime_record(
    reference, expected_sha, gpyreg_checkout, expected_gpyreg_sha
):
    import benchmark_targets
    import cma
    import golden_trace
    import gpyreg
    import numpy
    import profile_run
    import scipy

    import pyvbmc

    reference = _resolved(reference)
    gpyreg_checkout = _resolved(gpyreg_checkout)
    versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "gpyreg": _package_version("gpyreg"),
        "cma": _package_version("cma"),
        "pyvbmc": _package_version("pyvbmc"),
    }
    for name, expected in EXPECTED_VERSIONS.items():
        if versions[name] != expected:
            raise PreparationError(
                f"{name} version {versions[name]} does not equal expected {expected}"
            )
    modules = {
        "pyvbmc": _require_under(pyvbmc.__file__, reference, "pyvbmc"),
        "benchmark_targets": _require_under(
            benchmark_targets.__file__,
            reference / "dev" / "scripts",
            "benchmark_targets",
        ),
        "golden_trace": _require_under(
            golden_trace.__file__,
            reference / "dev" / "scripts",
            "golden_trace",
        ),
        "profile_run": _require_under(
            profile_run.__file__, reference / "dev" / "scripts", "profile_run"
        ),
        "gpyreg": _require_under(gpyreg.__file__, gpyreg_checkout, "gpyreg"),
        "numpy": str(_resolved(numpy.__file__)),
        "scipy": str(_resolved(scipy.__file__)),
        "cma": str(_resolved(cma.__file__)),
    }
    reference_state = _verify_reference_source(reference, expected_sha)
    gpyreg_state = _git_state(gpyreg_checkout, expected_gpyreg_sha)
    threads = {name: os.environ.get(name) for name in THREADS}
    if threads != THREADS:
        raise PreparationError(f"thread environment mismatch: {threads}")

    golden = {cfg.label: cfg for cfg in benchmark_targets.SUITES["golden"]}
    for label, expected in CONFIGS.items():
        if label not in golden:
            raise PreparationError(f"golden suite is missing {label}")
        cfg = golden[label]
        observed = {
            "problem": cfg.name,
            "D": cfg.D,
            "noise_sd": cfg.noise_sd,
            "max_fun_evals": cfg.options_dict().get("max_fun_evals"),
        }
        if observed != expected:
            raise PreparationError(
                f"configuration mismatch for {label}: {observed} != {expected}"
            )
        problem = cfg.make(seed=0)
        _, options = problem.vbmc_args()
        if options.get("specify_target_noise") is not True:
            raise PreparationError(
                f"{label} does not use the existing noisy wrapper"
            )

    return {
        "interpreter": str(_resolved(sys.executable)),
        "versions": versions,
        "modules": modules,
        "reference": reference_state,
        "gpyreg": gpyreg_state,
        "threads": threads,
        "execution_options": {"vectorized_target": False, "workers": 1},
    }


_BOOTSTRAP = r"""
import runpy
import sys
from pathlib import Path

reference = str(Path(sys.argv.pop(1)).resolve())
gpyreg = str(Path(sys.argv.pop(1)).resolve())
script = str(Path(sys.argv.pop(1)).resolve())
scripts = str(Path(reference, "dev", "scripts"))
sys.path[:0] = [reference, gpyreg, scripts]
sys.argv = [script, *sys.argv[1:]]
runpy.run_path(script, run_name="__main__")
"""


def _worker_command(python, reference, gpyreg_checkout, launcher, *arguments):
    return [
        str(python),
        "-I",
        "-c",
        _BOOTSTRAP,
        str(reference),
        str(gpyreg_checkout),
        str(launcher),
        *map(str, arguments),
    ]


def _worker_env(reference, gpyreg_checkout):
    env = os.environ.copy()
    env.update(THREADS)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (env.get("PYTHONPATH"),) if part
    )
    env["GIT_CONFIG_COUNT"] = "2"
    env["GIT_CONFIG_KEY_0"] = "safe.directory"
    env["GIT_CONFIG_VALUE_0"] = str(_resolved(reference))
    env["GIT_CONFIG_KEY_1"] = "safe.directory"
    env["GIT_CONFIG_VALUE_1"] = str(_resolved(gpyreg_checkout))
    return env


def _probe_runtime(
    python, reference, expected_sha, gpyreg_checkout, gpyreg_sha, launcher
):
    command = _worker_command(
        python,
        reference,
        gpyreg_checkout,
        launcher,
        "_probe",
        "--reference-checkout",
        reference,
        "--expected-sha",
        expected_sha,
        "--gpyreg-checkout",
        gpyreg_checkout,
        "--expected-gpyreg-sha",
        gpyreg_sha,
    )
    result = subprocess.run(
        command,
        env=_worker_env(reference, gpyreg_checkout),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise PreparationError(
            "spawned runtime probe failed:\n" + result.stdout + result.stderr
        )
    marker = "REFERENCE_RUNTIME="
    lines = [
        line for line in result.stdout.splitlines() if line.startswith(marker)
    ]
    if len(lines) != 1:
        raise PreparationError(
            "spawned runtime probe returned no unique record"
        )
    record = json.loads(lines[0][len(marker) :])
    if not os.path.samefile(record["interpreter"], python):
        raise PreparationError(
            f"spawned interpreter {record['interpreter']} differs from {python}"
        )
    return record


def _tasks():
    return [
        {"label": label, "seed": seed, "tag": f"{label}_seed{seed}"}
        for label in CONFIGS
        for seed in SEEDS
    ]


def _source_hashes(reference):
    return {name: _sha256(reference / name) for name in SOURCE_FILES}


def _ensure_distinct_empty_output(out, population):
    out = _resolved(out)
    population = _resolved(population)
    if (
        out == population
        or population in out.parents
        or out in population.parents
    ):
        raise PreparationError(
            "extension output and historical population must be separate"
        )
    if out.exists() and any(out.iterdir()):
        raise PreparationError(
            f"extension output must be empty during prepare: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


def _manifest_paths(data):
    paths = data["paths"]
    return {name: _resolved(value) for name, value in paths.items()}


def _verify_historical_manifest_pin(data, paths):
    observed = _sha256(paths["historical_manifest"], normalize_text=True)
    expected = data["historical"]["manifest_sha256"]
    if observed != expected:
        raise PreparationError(
            "historical manifest changed since preparation; refusing to change "
            "the prepared 810-pair baseline"
        )


def _load_manifest(path):
    path = _resolved(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise PreparationError(
            f"unsupported preparation manifest schema in {path}"
        )
    _verify_task_allocation(data.get("tasks"))
    paths = _manifest_paths(data)
    _verify_historical_manifest_pin(data, paths)
    launcher = paths["launcher"]
    if _resolved(__file__) != launcher:
        raise PreparationError(f"run the frozen launcher directly: {launcher}")
    if _sha256(launcher) != data["launcher_sha256"]:
        raise PreparationError("frozen launcher hash changed")
    if _source_hashes(paths["reference"]) != data["source_sha256"]:
        raise PreparationError("frozen supporting source hash changed")
    return path, data, paths


def _verify_task_allocation(tasks):
    if tasks != _tasks():
        raise PreparationError(
            "preparation manifest allocation, order, or canonical task tag "
            "is not the approved 3 x 50"
        )


def cmd_prepare(args):
    python = _resolved(args.python)
    reference = _resolved(args.reference_checkout)
    gpyreg_checkout = _resolved(args.gpyreg_checkout)
    population = _resolved(args.population)
    historical_manifest = _resolved(args.historical_manifest)
    manifest_path = _resolved(args.manifest)
    launcher = manifest_path.with_name("reference_noisy_extension_frozen.py")
    if manifest_path.exists() or launcher.exists():
        raise PreparationError(
            f"refusing to overwrite preparation artifact beside {manifest_path}"
        )
    if not python.is_file():
        raise PreparationError(f"Python interpreter does not exist: {python}")
    reference_state = _verify_reference_source(reference, args.expected_sha)
    gpyreg_state = _git_state(gpyreg_checkout, args.expected_gpyreg_sha)
    historical = _verify_historical(population, historical_manifest)
    out = _ensure_distinct_empty_output(args.out, population)
    runtime = _probe_runtime(
        python,
        reference,
        args.expected_sha,
        gpyreg_checkout,
        args.expected_gpyreg_sha,
        _resolved(__file__),
    )
    tasks = _tasks()
    data = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "base_sha": BASE_SHA,
        "expected_sha": args.expected_sha,
        "expected_gpyreg_sha": args.expected_gpyreg_sha,
        "paths": {
            "python": str(python),
            "reference": str(reference),
            "gpyreg": str(gpyreg_checkout),
            "historical_population": str(population),
            "historical_manifest": str(historical_manifest),
            "extension_output": str(out),
            "launcher": str(launcher),
        },
        "reference": reference_state,
        "gpyreg": gpyreg_state,
        "historical": historical,
        "source_sha256": _source_hashes(reference),
        "launcher_sha256": _sha256(__file__),
        "runtime_probe": runtime,
        "configs": CONFIGS,
        "seeds": list(SEEDS),
        "tasks": tasks,
        "execution": {
            "workers": 1,
            "threads": THREADS,
            "options": {"vectorized_target": False},
            "launcher": "run --manifest <path> --confirm-run-150",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, launcher)
    manifest_path.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"prepared 150 tasks; zero optimizations run\nmanifest: {manifest_path}"
    )
    return 0


def _extension_state(out, expected_tags):
    json_tags = {path.stem for path in out.glob("*_seed*.json")}
    npz_tags = {path.stem for path in out.glob("*_seed*.npz")}
    runtime_dir = out / "runtime"
    runtime_tags = {path.stem for path in runtime_dir.glob("*_seed*.json")}
    errors = sorted(path.name for path in out.glob("*.error.txt"))
    unexpected = (json_tags | npz_tags | runtime_tags) - expected_tags
    present = json_tags | npz_tags | runtime_tags
    complete = json_tags & npz_tags & runtime_tags
    incomplete = present - complete
    if unexpected or incomplete or errors:
        raise PreparationError(
            "extension output is not safely resumable: "
            f"unexpected={sorted(unexpected)}, incomplete={sorted(incomplete)}, "
            f"errors={errors}"
        )
    return complete


def cmd_run(args):
    if not args.confirm_run_150:
        raise PreparationError(
            "run requires the literal --confirm-run-150 flag"
        )
    manifest_path, data, paths = _load_manifest(args.manifest)
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".run.lock")
    launch_token = secrets.token_hex(32)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise PreparationError(
            f"launch lock exists: {lock_path}. Inspect it and confirm no launcher "
            "is active before removing a stale lock manually"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as lock:
        lock.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                    "manifest": str(manifest_path),
                    "token": launch_token,
                },
                indent=2,
            )
            + "\n"
        )
    try:
        return _run_locked(manifest_path, data, paths, launch_token)
    finally:
        lock_path.unlink(missing_ok=True)


def _run_locked(manifest_path, data, paths, launch_token):
    _verify_reference_source(paths["reference"], data["expected_sha"])
    _git_state(paths["gpyreg"], data["expected_gpyreg_sha"])
    _verify_historical(
        paths["historical_population"], paths["historical_manifest"]
    )
    expected_tags = {row["tag"] for row in data["tasks"]}
    done = _extension_state(paths["extension_output"], expected_tags)
    completed_rows = [row for row in data["tasks"] if row["tag"] in done]
    if completed_rows:
        _validate_records(data, paths, completed_rows)
    pending = [row for row in data["tasks"] if row["tag"] not in done]
    print(f"approved extension: {len(pending)} pending, {len(done)} complete")
    for index, row in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] {row['tag']}", flush=True)
        command = _worker_command(
            paths["python"],
            paths["reference"],
            paths["gpyreg"],
            paths["launcher"],
            "_task",
            "--manifest",
            manifest_path,
            "--label",
            row["label"],
            "--seed",
            row["seed"],
            "--launch-token",
            launch_token,
        )
        result = subprocess.run(
            command,
            env=_worker_env(paths["reference"], paths["gpyreg"]),
            check=False,
        )
        if result.returncode != 0:
            raise PreparationError(
                f"worker failed for {row['tag']}; launch stopped"
            )
        _validate_records(data, paths, [row])
    _verify_historical(
        paths["historical_population"], paths["historical_manifest"]
    )
    print("150 extension tasks complete; run validate before any publication")
    return 0


def _validate_trace(path, dimension, timer_keys, iterations, final_components):
    try:
        return _validate_trace_contents(
            path, dimension, timer_keys, iterations, final_components
        )
    except PreparationError:
        raise
    except (
        Exception
    ) as exc:  # malformed ZIP/NPY data or forbidden object arrays
        raise PreparationError(
            f"trace {Path(path).name} is unreadable: {exc}"
        ) from exc


def _validate_trace_contents(
    path, dimension, timer_keys, iterations, final_components
):
    import numpy as np

    with np.load(path, allow_pickle=False) as trace:
        for key in trace.files:
            np.asarray(trace[key])
        missing = TRACE_ARRAYS - set(trace.files)
        if missing:
            raise PreparationError(
                f"trace {path.name} lacks arrays {sorted(missing)}"
            )
        n_it = int(iterations)
        if n_it <= 0:
            raise PreparationError(f"trace {path.name} has no iterations")
        for key in ITERATION_ARRAYS:
            if trace[key].shape != (n_it,):
                raise PreparationError(
                    f"trace {path.name} has misaligned {key}"
                )
        if trace["timer"].shape != (n_it, len(timer_keys)):
            raise PreparationError(f"trace {path.name} has misaligned timer")
        for key in ("pt_mu", "pt_delta", "pt_scale", "vp_lambd"):
            if trace[key].shape != (n_it, dimension):
                raise PreparationError(
                    f"trace {path.name} has invalid {key} shape"
                )
        if trace["pt_R"].shape != (n_it, dimension, dimension):
            raise PreparationError(f"trace {path.name} has invalid pt_R shape")
        if trace["gp_hyp"].ndim != 2 or trace["gp_hyp"].shape[0] == 0:
            raise PreparationError(f"trace {path.name} has empty GP history")
        gp_rows = trace["gp_hyp"].shape[0]
        if trace["gp_hyp_iter"].shape != (gp_rows,):
            raise PreparationError(
                f"trace {path.name} has misaligned GP history"
            )
        vp_rows = trace["vp_w"].size
        if vp_rows == 0 or trace["vp_w"].shape != (vp_rows,):
            raise PreparationError(f"trace {path.name} has empty VP history")
        for key in ("vp_sigma", "vp_iter"):
            if trace[key].shape != (vp_rows,):
                raise PreparationError(
                    f"trace {path.name} has misaligned {key}"
                )
        if trace["vp_mu"].shape != (vp_rows, dimension):
            raise PreparationError(
                f"trace {path.name} has invalid vp_mu shape"
            )
        for x_key, y_key in (("X_orig", "y_orig"), ("X_init", "y_init")):
            if trace[x_key].ndim != 2 or trace[x_key].shape[1] != dimension:
                raise PreparationError(
                    f"trace {path.name} has invalid {x_key} shape"
                )
            if trace[y_key].shape != (trace[x_key].shape[0],):
                raise PreparationError(
                    f"trace {path.name} has misaligned {y_key}"
                )
        k = int(final_components)
        if k <= 0 or trace["final_w"].shape != (k,):
            raise PreparationError(
                f"trace {path.name} has invalid final weights"
            )
        if trace["final_sigma"].shape != (k,):
            raise PreparationError(
                f"trace {path.name} has invalid final sigma"
            )
        if trace["final_mu"].shape != (dimension, k):
            raise PreparationError(
                f"trace {path.name} has invalid final_mu shape"
            )
        if trace["final_lambd"].shape != (dimension,):
            raise PreparationError(
                f"trace {path.name} has invalid final_lambd shape"
            )
        if trace["post_mean"].shape != (1, dimension):
            raise PreparationError(
                f"trace {path.name} has invalid post_mean shape"
            )
        if trace["post_cov"].shape != (dimension, dimension):
            raise PreparationError(
                f"trace {path.name} has invalid post_cov shape"
            )


def _validate_records(data, paths, rows):
    statuses = {}
    for row in rows:
        tag = row["tag"]
        expected = CONFIGS[row["label"]]
        side = json.loads(
            (paths["extension_output"] / f"{tag}.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            side.get("label") != row["label"]
            or side.get("seed") != row["seed"]
        ):
            raise PreparationError(f"sidecar allocation mismatch for {tag}")
        for key in ("problem", "D", "noise_sd"):
            if side.get(key) != expected[key]:
                raise PreparationError(f"sidecar {key} mismatch for {tag}")
        requested = side.get("requested_options", {})
        effective = side.get("effective_options", {})
        for options_name, options in (
            ("requested", requested),
            ("effective", effective),
        ):
            if options.get("vectorized_target") is not False:
                raise PreparationError(
                    f"{options_name} vectorized_target is not false for {tag}"
                )
            if options.get("max_fun_evals") != expected["max_fun_evals"]:
                raise PreparationError(
                    f"{options_name} paper budget mismatch for {tag}"
                )
            if options.get("specify_target_noise") is not True:
                raise PreparationError(
                    f"{options_name} target-noise option mismatch for {tag}"
                )
        final = side.get("final", {})
        for key in ("elbo", "elbo_sd", "elbo_err", "gskl", "mmtv", "rmse"):
            if not math.isfinite(final.get(key, math.nan)):
                raise PreparationError(f"nonfinite {key} for {tag}")
        count = final.get("func_count", 0)
        if not 0 < count <= expected["max_fun_evals"]:
            raise PreparationError(
                f"invalid function count for {tag}: {count}"
            )
        meta = side.get("meta", {})
        recorded_sha = meta.get("git", {}).get("sha", "")
        if (
            not data["expected_sha"].startswith(recorded_sha)
            or not recorded_sha
        ):
            raise PreparationError(f"source provenance mismatch for {tag}")
        if meta.get("git", {}).get("dirty") is not False:
            raise PreparationError(f"dirty source provenance for {tag}")
        for name, expected_version in EXPECTED_VERSIONS.items():
            observed = meta.get(name)
            if observed != expected_version:
                raise PreparationError(
                    f"{name} provenance mismatch for {tag}: {observed}"
                )
        if meta.get("threads") != THREADS:
            raise PreparationError(f"thread provenance mismatch for {tag}")
        _validate_trace(
            paths["extension_output"] / f"{tag}.npz",
            expected["D"],
            side.get("timer_keys", []),
            final.get("iterations", 0),
            final.get("final_K", 0),
        )
        runtime = json.loads(
            (paths["extension_output"] / "runtime" / f"{tag}.json").read_text(
                encoding="utf-8"
            )
        )
        if runtime.pop("label", None) != row["label"]:
            raise PreparationError(f"runtime label mismatch for {tag}")
        if runtime.pop("seed", None) != row["seed"]:
            raise PreparationError(f"runtime seed mismatch for {tag}")
        if runtime != data["runtime_probe"]:
            raise PreparationError(
                f"worker runtime differs from prepared runtime for {tag}"
            )
        key = (
            bool(side["final"]["success_flag"]),
            str(side["final"].get("message", "")),
        )
        statuses[key] = statuses.get(key, 0) + 1
    return {"pairs": len(rows), "statuses": statuses}


def _validate_outputs(data, paths):
    expected_tags = {row["tag"] for row in data["tasks"]}
    complete = _extension_state(paths["extension_output"], expected_tags)
    if complete != expected_tags:
        missing = sorted(expected_tags - complete)
        raise PreparationError(
            f"extension is incomplete: {len(missing)} tasks missing"
        )
    return _validate_records(data, paths, data["tasks"])


def cmd_validate(args):
    _, data, paths = _load_manifest(args.manifest)
    _verify_reference_source(paths["reference"], data["expected_sha"])
    _git_state(paths["gpyreg"], data["expected_gpyreg_sha"])
    _verify_historical(
        paths["historical_population"], paths["historical_manifest"]
    )
    report = _validate_outputs(data, paths)
    print(
        f"validated {report['pairs']} extension pairs; historical 810 remain unchanged"
    )
    print("convergence statuses:")
    for (success, message), count in sorted(report["statuses"].items()):
        print(f"  success={success}: {count} ({message})")
    print("output remains isolated; no files were published")
    return 0


def cmd_probe(args):
    record = _runtime_record(
        args.reference_checkout,
        args.expected_sha,
        args.gpyreg_checkout,
        args.expected_gpyreg_sha,
    )
    print("REFERENCE_RUNTIME=" + json.dumps(record, sort_keys=True))
    return 0


def cmd_task(args):
    manifest_path, data, paths = _load_manifest(args.manifest)
    _verify_launch_token(manifest_path, args.launch_token)
    pair = (args.label, args.seed)
    approved = {(row["label"], row["seed"]) for row in data["tasks"]}
    if pair not in approved:
        raise PreparationError(
            f"worker task is outside approved allocation: {pair}"
        )
    tag = f"{args.label}_seed{args.seed}"
    for suffix in (".json", ".npz", ".error.txt"):
        if (paths["extension_output"] / f"{tag}{suffix}").exists():
            raise PreparationError(
                f"worker refuses to overwrite {tag}{suffix}"
            )
    runtime_dir = paths["extension_output"] / "runtime"
    runtime_dir.mkdir(exist_ok=True)
    runtime_path = runtime_dir / f"{tag}.json"
    if runtime_path.exists():
        raise PreparationError(f"worker refuses to overwrite {runtime_path}")
    runtime = _runtime_record(
        paths["reference"],
        data["expected_sha"],
        paths["gpyreg"],
        data["expected_gpyreg_sha"],
    )
    runtime["label"] = args.label
    runtime["seed"] = args.seed
    runtime_path.write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    import golden_trace

    result = golden_trace.run_task(
        args.label,
        args.seed,
        {"vectorized_target": False},
        str(paths["extension_output"]),
    )
    if not result.get("ok"):
        return 1
    return 0


def _verify_launch_token(manifest_path, launch_token):
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".run.lock")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreparationError(
            "worker has no readable live parent launch lock"
        ) from exc
    if lock.get("manifest") != str(
        manifest_path
    ) or not secrets.compare_digest(
        str(lock.get("token", "")), str(launch_token)
    ):
        raise PreparationError(
            "worker launch token does not match the live parent lock"
        )


def _common_probe_arguments(parser):
    parser.add_argument("--reference-checkout", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--gpyreg-checkout", required=True)
    parser.add_argument("--expected-gpyreg-sha", default=DEFAULT_GPYREG_SHA)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command")
    prepare = sub.add_parser(
        "prepare", help="verify and write a launch manifest"
    )
    prepare.add_argument("--python", required=True)
    _common_probe_arguments(prepare)
    prepare.add_argument("--population", required=True)
    prepare.add_argument("--historical-manifest", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.set_defaults(func=cmd_prepare)

    run = sub.add_parser(
        "run", help="future explicit execution of exactly 150 tasks"
    )
    run.add_argument("--manifest", required=True)
    run.add_argument("--confirm-run-150", action="store_true")
    run.set_defaults(func=cmd_run)

    validate = sub.add_parser(
        "validate", help="validate isolated completed output"
    )
    validate.add_argument("--manifest", required=True)
    validate.set_defaults(func=cmd_validate)

    probe = sub.add_parser("_probe", help=argparse.SUPPRESS)
    _common_probe_arguments(probe)
    probe.set_defaults(func=cmd_probe)
    task = sub.add_parser("_task", help=argparse.SUPPRESS)
    task.add_argument("--manifest", required=True)
    task.add_argument("--label", required=True)
    task.add_argument("--seed", required=True, type=int)
    task.add_argument("--launch-token", required=True)
    task.set_defaults(func=cmd_task)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except PreparationError as exc:
        print(f"reference extension refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
