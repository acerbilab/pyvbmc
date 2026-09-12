"""Run a fixed population sequentially, retaining production boost decisions.

The manifest supplies configuration/seed pairs and options. Each case runs in
a fresh process; a hash-verified completion record permits safe resumption.
Use a frozen checkout and set PYVBMC_GPYREG_SOURCE to a frozen gpyreg checkout.
"""

import argparse
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(ROOT))
if os.environ.get("PYVBMC_GPYREG_SOURCE"):
    sys.path.insert(1, os.environ["PYVBMC_GPYREG_SOURCE"])

import dill
import golden_trace
import numpy as np
from benchmark_targets import find_config, metrics
from filelock import FileLock
from profile_run import jsonable

from pyvbmc import VBMC

VBMC_MODULE = importlib.import_module("pyvbmc.vbmc.vbmc")
SUFFIXES = (".npz", ".json", ".boost.pkl", ".boost.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def case_path(out, tag, suffix):
    # Keep auxiliary JSON outside golden_trace's *_seed*.json discovery.
    folder = (
        Path(out) / "records"
        if suffix in (".boost.json", ".complete.json")
        else Path(out)
    )
    return folder / f"{tag}{suffix}"


def clone_vp(vp):
    result = copy.deepcopy(vp)
    result.rng = copy.deepcopy(vp.rng)
    return result


def scores(vp):
    if vp is None:
        return None
    return {k: float(vp.stats[k]) for k in ("elbo", "elbo_sd")}


class BoostCapture:
    """Observe the real final boost without rescoring or making random draws."""

    def __init__(self):
        self.state = None

    def __enter__(self):
        original_boost = VBMC.final_boost
        original_optimize = VBMC_MODULE.optimize_vp
        owner = self

        def boost(vbmc, vp, gp):
            if owner.state is not None:
                raise RuntimeError("More than one final boost in a run")
            state = {
                "pre": clone_vp(vp),
                "candidate": None,
                "rng_before": copy.deepcopy(vbmc.rng.bit_generator.state),
                "tolerance": vbmc.options.get("tol_elcbo_boost"),
                "boost_options": None,
                "boost_call": None,
            }
            owner.state = state

            def optimize(options, *args, **kwargs):
                if state["candidate"] is not None:
                    raise RuntimeError("More than one optimizer call in boost")
                state["boost_options"] = {
                    k: jsonable(options.get(k)) for k in options
                }
                state["boost_call"] = dict(
                    zip(("n_fast_opts", "n_slow_opts", "K_new"), args[3:])
                )
                result = original_optimize(options, *args, **kwargs)
                state["candidate"] = clone_vp(result[0])
                return result

            with patch.object(VBMC_MODULE, "optimize_vp", optimize):
                result = original_boost(vbmc, vp, gp)
            state["returned"] = clone_vp(result[0])
            state["attempted"] = state["candidate"] is not None
            state["accepted"] = bool(result[3])
            state["rng_after"] = copy.deepcopy(vbmc.rng.bit_generator.state)
            return result

        self.context = patch.object(VBMC, "final_boost", boost)
        self.context.__enter__()
        return self

    def __exit__(self, *exc):
        return self.context.__exit__(*exc)


def git(path, *args):
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True
    ).strip()


def identity(manifest):
    from importlib.metadata import version

    import gpyreg

    import pyvbmc

    gp_root = Path(os.environ["PYVBMC_GPYREG_SOURCE"]).resolve()
    if Path(pyvbmc.__file__).resolve().parent != ROOT / "pyvbmc":
        raise RuntimeError("PyVBMC imported outside the frozen checkout")
    if Path(gpyreg.__file__).resolve().parent != gp_root / "gpyreg":
        raise RuntimeError("gpyreg imported outside the frozen checkout")
    helper_imports = {}
    for name in ("golden_trace", "benchmark_targets", "profile_run"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if path != ROOT / "dev" / "scripts" / f"{name}.py":
            raise RuntimeError(f"{name} imported outside frozen tooling")
        helper_imports[name] = str(path)
    for repo in (ROOT, gp_root):
        if git(repo, "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError(f"Uncommitted files in {repo}")
    if git(
        ROOT,
        "diff",
        manifest["candidate_code_commit"],
        "HEAD",
        "--",
        "pyvbmc",
        "dev/scripts/benchmark_targets.py",
        "dev/scripts/data",  # the real-data targets' data and truths
    ):
        raise RuntimeError(
            "Numerical source differs from the prepared candidate"
        )
    gp_sha = git(gp_root, "rev-parse", "HEAD")
    if gp_sha != manifest["gpyreg_commit"]:
        raise RuntimeError("gpyreg commit differs from manifest")
    versions = {k: version(k) for k in manifest["dependencies"]}
    if versions != manifest["dependencies"]:
        raise RuntimeError("Dependency versions differ from manifest")
    if sys.version.split()[0] != manifest["python"]:
        raise RuntimeError("Python version differs from manifest")
    return {
        "candidate_sha": git(ROOT, "rev-parse", "HEAD"),
        "numerical_base_sha": manifest["candidate_code_commit"],
        "gpyreg_sha": gp_sha,
        "pyvbmc_import": pyvbmc.__file__,
        "gpyreg_import": gpyreg.__file__,
        "helper_imports": helper_imports,
        "python": sys.version.split()[0],
        "dependencies": versions,
        "threads": {k: os.environ[k] for k in manifest["thread_environment"]},
    }


def validate_case(out, tag, expected):
    """Fail on incomplete or changed artifacts; never silently skip them."""
    out = Path(out)
    done = json.loads(case_path(out, tag, ".complete.json").read_text())
    if done["identity"] != expected:
        raise RuntimeError(f"{tag}: source/environment/options changed")
    if set(done["hashes"]) != set(SUFFIXES):
        raise RuntimeError(f"{tag}: incomplete file manifest")
    for suffix, digest in done["hashes"].items():
        if sha256(case_path(out, tag, suffix)) != digest:
            raise RuntimeError(f"{tag}: changed {suffix}")
    side = json.loads((out / f"{tag}.json").read_text())
    if f"{side['label']}_seed{side['seed']}" != tag:
        raise RuntimeError(f"{tag}: wrong configuration or seed")
    for metric in (*golden_trace.METRICS, "elbo", "elbo_sd"):
        if not np.isfinite(side["final"][metric]):
            raise RuntimeError(f"{tag}: nonfinite {metric}")
    if side["label"] == "cigar_D15_exhaust":
        if side["final"]["func_count"] != 750:
            raise RuntimeError(f"{tag}: incorrect exhaust budget")
    for key, value in expected["options"].items():
        if side["requested_options"].get(key) != value:
            raise RuntimeError(f"{tag}: incorrect requested {key}")
        if side["effective_options"].get(key) != value:
            raise RuntimeError(f"{tag}: incorrect effective {key}")
    with np.load(out / f"{tag}.npz", allow_pickle=False) as data:
        for key in data.files:
            _ = data[key]
    with (out / f"{tag}.boost.pkl").open("rb") as stream:
        capture = dill.load(stream)
    for key in ("pre", "returned"):
        if capture[key].parameter_transformer is None:
            raise RuntimeError(f"{tag}: missing {key} transformer")
    if capture["attempted"]:
        if capture["candidate"] is None:
            raise RuntimeError(f"{tag}: missing candidate")
        if capture["boost_options"]["weight_penalty"] != 0:
            raise RuntimeError(f"{tag}: boost penalty was not disabled")
    report = json.loads(case_path(out, tag, ".boost.json").read_text())
    for key in (
        "attempted",
        "accepted",
        "tolerance",
        "boost_options",
        "boost_call",
    ):
        if key not in report or report[key] != jsonable(capture[key]):
            raise RuntimeError(f"{tag}: inconsistent boost {key}")
    if (
        type(report["attempted"]) is not bool
        or type(report["accepted"]) is not bool
    ):
        raise RuntimeError(f"{tag}: invalid boost decision type")
    if report["tolerance"] != expected["options"]["tol_elcbo_boost"]:
        raise RuntimeError(f"{tag}: incorrect boost tolerance")
    keys = (
        ("pre", "candidate", "returned")
        if capture["attempted"]
        else ("pre", "returned")
    )
    if set(report.get("metrics", {})) != set(keys):
        raise RuntimeError(f"{tag}: incomplete boost metrics")
    if set(report.get("scores", {})) != {"pre", "candidate", "returned"}:
        raise RuntimeError(f"{tag}: incomplete boost scores")
    for key in keys:
        vp = capture[key]
        if vp.parameter_transformer is None:
            raise RuntimeError(f"{tag}: missing {key} transformer")
        np.testing.assert_equal(report["scores"][key], scores(vp))
        met = report["metrics"][key]
        if key != "returned" and isinstance(met.get("error"), str):
            continue
        for metric in (
            "elbo_err",
            "gskl",
            "mmtv",
            "rmse",
            "post_mean",
            "post_cov",
            "moment_method",
        ):
            if metric not in met:
                raise RuntimeError(f"{tag}: missing {key}/{metric}")
        for metric in ("elbo_err", "gskl", "mmtv", "rmse"):
            if key == "returned" and not np.isfinite(met[metric]):
                raise RuntimeError(f"{tag}: nonfinite {key}/{metric}")
    if not capture["attempted"] and (
        capture["accepted"] or capture["candidate"] is not None
    ):
        raise RuntimeError(f"{tag}: inconsistent skipped boost")
    chosen = capture["candidate"] if capture["accepted"] else capture["pre"]
    for attr in ("w", "mu", "sigma", "lambd"):
        np.testing.assert_array_equal(
            getattr(chosen, attr), getattr(capture["returned"], attr)
        )
    np.testing.assert_equal(
        chosen.parameter_transformer.__dict__,
        capture["returned"].parameter_transformer.__dict__,
    )
    np.testing.assert_equal(
        scores(capture["returned"]),
        {k: side["final"][k] for k in ("elbo", "elbo_sd")},
    )
    return done


def diagnostic_metrics(problem, vp, required=False):
    try:
        return jsonable(metrics(problem, vp, vp.stats["elbo"]))
    except Exception as error:
        if required:
            raise
        return {"error": f"{type(error).__name__}: {error}"}


def worker(manifest, out, label, seed, expected):
    actual = identity(manifest)
    actual["options"] = manifest["options"]
    if actual != expected:
        raise RuntimeError("Worker identity differs from launcher")
    tag = f"{label}_seed{seed}"
    started = time.time()
    with BoostCapture() as capture:
        result = golden_trace.run_task(label, seed, manifest["options"], out)
    if not result["ok"] or capture.state is None:
        raise RuntimeError(f"{tag}: inference or boost capture failed")
    state = capture.state
    with (out / f"{tag}.boost.pkl").open("wb") as stream:
        dill.dump(state, stream)
    problem = find_config(label).make(seed=seed)
    report = {
        "attempted": state["attempted"],
        "accepted": state["accepted"],
        "tolerance": state["tolerance"],
        "boost_options": state["boost_options"],
        "boost_call": jsonable(state["boost_call"]),
        "scores": {
            k: scores(state[k]) for k in ("pre", "candidate", "returned")
        },
        "metrics": {
            k: diagnostic_metrics(problem, state[k], required=k == "returned")
            for k in ("pre", "candidate", "returned")
            if state[k] is not None
        },
    }
    write_json(case_path(out, tag, ".boost.json"), report)
    done = {
        "identity": actual,
        "elapsed_seconds": time.time() - started,
        "hashes": {
            suffix: sha256(case_path(out, tag, suffix)) for suffix in SUFFIXES
        },
    }
    write_json(case_path(out, tag, ".complete.json"), done)
    validate_case(out, tag, actual)


def run(args, manifest):
    if not manifest.get("launch_ready"):
        raise RuntimeError("Manifest is not marked ready for launch")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    expected = identity(manifest)
    expected["options"] = manifest["options"]
    launch = out / "launch.json"
    if launch.exists():
        previous = json.loads(launch.read_text())
        if (
            previous["identity"] != expected
            or previous["manifest"] != manifest
        ):
            raise RuntimeError(
                "Existing campaign belongs to a different setup"
            )
    else:
        write_json(launch, {"identity": expected, "manifest": manifest})
    tasks = [
        (c["label"], s) for c in manifest["allocation"] for s in c["seeds"]
    ]
    if (
        len(set(tasks)) != len(tasks)
        or len(tasks) != manifest["candidate_run_count"]
    ):
        raise RuntimeError("Duplicate tasks or incorrect task count")
    # Early ordinary/noisy cases provide recording checks and timing feedback.
    priority = ["normal_D5", "rosenbrock_D2_noise1", "cigar_D15_exhaust"]
    tasks.sort(
        key=lambda t: (
            (0, priority.index(t[0]))
            if t[1] == 0 and t[0] in priority
            else (1, t[0], t[1])
        )
    )
    selected = tasks[: args.limit] if args.limit else tasks
    completed, failed = [], []
    started = time.time()
    for label, seed in selected:
        tag = f"{label}_seed{seed}"
        if case_path(out, tag, ".complete.json").exists():
            validate_case(out, tag, expected)
            completed.append(tag)
            continue
        if list(out.glob(f"{tag}.*")):
            raise RuntimeError(
                f"{tag}: incomplete prior attempt; inspect before retry"
            )
        status = {
            "pid": os.getpid(),
            "started": started,
            "active": tag,
            "completed": completed,
            "failed": failed,
            "total": len(tasks),
        }
        write_json(out / "status.json", status)
        print(
            f"START {tag} ({len(completed)}/{len(tasks)} complete)", flush=True
        )
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "worker",
            "--manifest",
            str(args.manifest.resolve()),
            "--out",
            str(out),
            "--label",
            label,
            "--seed",
            str(seed),
        ]
        with (out / f"{tag}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            failed.append(tag)
            print(f"FAILED {tag}; see case log", flush=True)
        else:
            validate_case(out, tag, expected)
            completed.append(tag)
            print(
                f"DONE {tag}; elapsed {(time.time()-started)/3600:.2f} h",
                flush=True,
            )
        status.update(
            active=None,
            completed=completed,
            failed=failed,
            updated=time.time(),
        )
        write_json(out / "status.json", status)
        if failed:
            return 1
    if len(completed) == len(tasks):
        golden_trace.cmd_summary(argparse.Namespace(dir=str(out)))
        text, flagged = golden_trace.compare_populations(
            golden_trace.load_population(ROOT / "dev/golden/baseline"),
            golden_trace.load_population(out),
        )
        (out / "comparison.md").write_text(text + "\n", encoding="utf-8")
        write_json(
            out / "finished.json",
            {
                "completed": completed,
                "flagged_configurations": sorted(flagged),
                "elapsed_seconds": time.time() - started,
                "finished": time.time(),
            },
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    manifest = json.loads(args.manifest.read_text())
    if args.command == "run":
        # Process-scoped request: permit display sleep, prevent idle system
        # sleep while the overnight supervisor is working.
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            args.out.mkdir(parents=True, exist_ok=True)
            with FileLock(str(args.out / "campaign.lock"), timeout=0):
                return run(args, manifest)
        finally:
            if sys.platform == "win32":
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    expected = json.loads((args.out / "launch.json").read_text())["identity"]
    worker(manifest, args.out, args.label, args.seed, expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
