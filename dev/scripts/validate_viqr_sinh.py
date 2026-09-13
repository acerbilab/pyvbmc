"""Capture and time VIQR states during the approved bounded golden replay.

Run ``replay --out DIR`` for the 18-run allocation, or ``timing --out DIR
--before-source CHECKOUT`` to compare the captured states in one process.
The timing comparison loads only the original VIQR module from CHECKOUT;
it verifies that the shared acquisition wrapper and GP code have not changed.
No target is evaluated by the timing command.

Use ``--equivalent-viqr`` to time an output-equivalent VIQR revision against
existing captures. Both source hashes are recorded, and every captured
public acquisition output must still match exactly.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from pyvbmc import VBMC
from pyvbmc.acquisition_functions import AcqFcnVIQR
from pyvbmc.testing.oracles._state import (
    build_state,
    encode,
    load_snapshot,
    save_snapshot,
    snapshot_from_objects,
    snapshot_names,
)

LABELS = (
    "rosenbrock_D2_noise1",
    "rosenbrock_D2_noise3",
    "logreg_D5_noise3",
    "student_D8_noise3",
    "multisensory_s1_D6_noise1.3",
    "timing_D5_noise2.2",
)
SEEDS = (0, 1, 2)
FACTOR_FIELDS = ("alpha", "L", "L_chol", "sW", "sn2_mult")
REPLAY_DEPENDENCIES = (
    Path("dev/scripts/validate_viqr_sinh.py"),
    Path("dev/scripts/golden_replay.py"),
    Path("dev/scripts/golden_trace.py"),
    Path("dev/scripts/benchmark_targets.py"),
    Path("dev/scripts/profile_run.py"),
    Path("pyvbmc/testing/oracles/_state.py"),
)


def digest(value):
    """Hash in-memory state without drawing from any generator."""
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shared_source_hashes():
    import gpyreg

    package = Path(gpyreg.__file__).parent
    paths = [
        package / name
        for name in (
            "gaussian_process.py",
            "covariance_functions.py",
            "mean_functions.py",
            "noise_functions.py",
        )
    ]
    paths.append(ROOT / "pyvbmc/acquisition_functions/abstract_acq_fcn.py")
    return {
        path.name: hashlib.sha256(
            path.read_text(encoding="utf-8").encode()
        ).hexdigest()
        for path in paths
    }


def replay_source_hashes():
    """Hash production PyVBMC and the sources that define this replay."""
    package = ROOT / "pyvbmc"
    paths = [
        path
        for pattern in ("*.py", "*.ini")
        for path in package.rglob(pattern)
        if path.relative_to(package).parts[0] != "testing"
    ]
    paths.extend(ROOT / path for path in REPLAY_DEPENDENCIES)
    return {
        path.relative_to(ROOT).as_posix(): file_hash(path)
        for path in sorted(paths, key=lambda value: value.as_posix())
    }


def execution_inputs():
    """Describe installed numerical code and benchmark data inputs."""
    from importlib.metadata import version

    import cma
    import gpyreg
    import scipy

    package = Path(gpyreg.__file__).resolve().parent
    data = ROOT / "dev/scripts/data"
    archives = list(data.glob("*.npz"))
    archives.extend((data / "truths").glob("*.npz"))
    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cma": cma.__version__,
            "gpyreg": version("gpyreg"),
        },
        "gpyreg_package_hashes": {
            path.relative_to(package).as_posix(): file_hash(path)
            for path in sorted(
                package.rglob("*.py"), key=lambda value: value.as_posix()
            )
        },
        "data_hashes": {
            path.relative_to(ROOT).as_posix(): file_hash(path)
            for path in sorted(archives, key=lambda value: value.as_posix())
        },
    }


def replay_manifest():
    return {
        "manifest_version": 2,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "allocation": {"labels": list(LABELS), "seeds": list(SEEDS)},
        "viqr_sha256": file_hash(
            ROOT / "pyvbmc/acquisition_functions/acq_fcn_viqr.py"
        ),
        "shared_source_hashes": shared_source_hashes(),
        "source_hashes": replay_source_hashes(),
        "execution_inputs": execution_inputs(),
    }


class Capture:
    """Copy two acquisition states; serialize them after optimization."""

    def __init__(self, label, directory, options):
        self.label, self.directory, self.options = label, directory, options
        self.pending = {}

    def call(self, original, acq, xs, gp, vp, logger, optim):
        checkpoint = None
        if (
            len(np.atleast_2d(xs)) == 8192
            and getattr(acq, "loss", "iqr") == "iqr"
        ):
            if "early" not in self.pending:
                checkpoint = "early"
            elif "late" not in self.pending and (
                len(gp.posteriors) == 1
                or len(gp.X) >= 0.75 * self.options["max_fun_evals"]
            ):
                checkpoint = "late"
        if checkpoint is None:
            return original(acq, xs, gp, vp, logger, optim)
        rng_before = digest(
            (vp.rng.bit_generator.state, np.random.get_state())
        )
        live_before = digest((xs, gp.X, gp.y, vp.mu, vp.w, optim))
        arrays, tree = snapshot_from_objects(
            vp,
            gp,
            logger,
            optim,
            self.options,
            meta={
                "label": self.label,
                "seed": 0,
                "checkpoint": checkpoint,
                "N": len(gp.X),
                "Ns": len(gp.posteriors),
                "shared_source_hashes": shared_source_hashes(),
            },
            iteration=optim["iter"],
        )
        tree["cand"] = {"Xs": encode(np.array(xs), "cand/Xs", arrays)}
        # Preserve the live factors, including incremental-update rounding.
        factors = [
            {key: copy.deepcopy(getattr(p, key)) for key in FACTOR_FIELDS}
            for p in gp.posteriors
        ]
        tree["live_factors"] = encode(factors, "live_factors", arrays)
        tree["temporary_data"] = encode(
            copy.deepcopy(gp.temporary_data), "temporary_data", arrays
        )
        # The codec retains references; freeze arrays before the live run
        # changes its VP, logger or bookkeeping.
        arrays = {
            key: np.array(value, copy=True) for key, value in arrays.items()
        }
        assert rng_before == digest(
            (vp.rng.bit_generator.state, np.random.get_state())
        )
        assert live_before == digest((xs, gp.X, gp.y, vp.mu, vp.w, optim))
        value = original(acq, xs, gp, vp, logger, optim)
        assert rng_before == digest(
            (vp.rng.bit_generator.state, np.random.get_state())
        )
        tree["ref"] = {"acq": encode(np.array(value), "ref/acq", arrays)}
        self.pending[checkpoint] = arrays, tree
        return value

    def flush(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for checkpoint, (arrays, tree) in self.pending.items():
            save_snapshot(
                self.directory / f"{self.label}_{checkpoint}", arrays, tree
            )
            for ext in (".npz", ".json"):
                filename = f"{self.label}_{checkpoint}{ext}"
                hashes[filename] = file_hash(self.directory / filename)
        status = {
            "label": self.label,
            "captured": list(self.pending),
            "late_checkpoint_reached": "late" in self.pending,
            "file_hashes": hashes,
        }
        (self.directory / f"{self.label}.status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )


def replay(out):
    import golden_replay
    import golden_trace

    original_task = golden_trace.run_task
    original_optimize = VBMC.optimize
    original_call = AcqFcnVIQR.__call__
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run_manifest.json"
    manifest = replay_manifest()
    if manifest_path.exists():
        assert (
            json.loads(manifest_path.read_text()) == manifest
        ), "Resume provenance differs"
    else:
        assert not list(
            out.glob("*_seed*.npz")
        ), "Existing traces have no validation manifest; use a fresh output directory"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def task(label, seed, options, directory):
        tag = f"{label}_seed{seed}"
        if (Path(directory) / f"{tag}.npz").exists() and (
            Path(directory) / f"{tag}.json"
        ).exists():
            if seed == 0:
                status_path = out / "snapshots" / f"{label}.status.json"
                assert (
                    status_path.exists()
                ), f"Missing capture status for {tag}"
                status = json.loads(status_path.read_text())
                assert (
                    status["label"] == label and "early" in status["captured"]
                )
                for filename, expected in status["file_hashes"].items():
                    assert file_hash(out / "snapshots" / filename) == expected
            print(f"[resume] retained {tag}", flush=True)
            return {"ok": True}

        def optimize(vbmc, *args, **kwargs):
            capture = Capture(label, out / "snapshots", vbmc.options)

            def call(acq, xs, gp, vp, logger, optim):
                return capture.call(
                    original_call, acq, xs, gp, vp, logger, optim
                )

            AcqFcnVIQR.__call__ = call
            try:
                return original_optimize(vbmc, *args, **kwargs)
            finally:
                del AcqFcnVIQR.__call__
                capture.flush()

        if seed == 0:
            VBMC.optimize = optimize
        try:
            return original_task(label, seed, options, directory)
        finally:
            VBMC.optimize = original_optimize

    golden_trace.run_task = task
    try:
        return golden_replay.main(
            [
                "--configs",
                ",".join(LABELS),
                "--seeds",
                ",".join(str(seed) for seed in SEEDS),
                "--out",
                str(out),
            ]
        )
    finally:
        golden_trace.run_task = original_task


def restore(path):
    snap = load_snapshot(path)
    state = build_state(snap, rng=np.random.default_rng(20260913))
    for posterior, factors in zip(
        state["gp"].posteriors, snap["live_factors"]
    ):
        for key, value in factors.items():
            setattr(posterior, key, copy.deepcopy(value))
    state["gp"].temporary_data = copy.deepcopy(snap["temporary_data"])
    return state


def timing(out, before, equivalent_viqr=False):
    import platform
    import tracemalloc

    import gpyreg
    import scipy

    from pyvbmc.acquisition_functions import acq_fcn_viqr as current_module
    from pyvbmc.vbmc.active_importance_sampling import (
        active_importance_sampling,
    )

    rel = Path("pyvbmc/acquisition_functions")
    manifest = json.loads((out / "run_manifest.json").read_text())
    manifest_version = manifest.get("manifest_version", 1)
    assert manifest_version in (1, 2), "Unknown replay manifest version"
    if not equivalent_viqr:
        assert manifest["viqr_sha256"] == file_hash(
            ROOT / rel / "acq_fcn_viqr.py"
        )
    assert manifest["shared_source_hashes"] == shared_source_hashes()
    if manifest_version == 2:
        current = replay_manifest()
        assert manifest["allocation"] == current["allocation"]
        expected_sources = manifest["source_hashes"].copy()
        if equivalent_viqr:
            key = "pyvbmc/acquisition_functions/acq_fcn_viqr.py"
            expected_sources[key] = current["source_hashes"][key]
        assert expected_sources == current["source_hashes"]
        assert manifest["execution_inputs"] == current["execution_inputs"]
        provenance = {
            "manifest_version": 2,
            "coverage": (
                "Git SHA, labels/seeds, runner, production PyVBMC/options, "
                "replay dependencies, full gpyreg package, runtime/library "
                "versions, platform, data, and VIQR/shared source hashes"
            ),
            "git_sha": manifest["git_sha"],
            "allocation": manifest["allocation"],
        }
    else:
        provenance = {
            "manifest_version": 1,
            "coverage": (
                "VIQR and shared acquisition/gpyreg source hashes only"
            ),
        }
    provenance["capture_viqr_sha256"] = manifest["viqr_sha256"]
    provenance["equivalent_viqr_requested"] = equivalent_viqr
    assert (before / rel / "abstract_acq_fcn.py").read_text(
        encoding="utf-8"
    ) == (ROOT / rel / "abstract_acq_fcn.py").read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "pyvbmc.acquisition_functions._viqr_before",
        before / rel / "acq_fcn_viqr.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    methods = {"before": module.AcqFcnVIQR(), "after": AcqFcnVIQR()}
    viqr_hashes = {
        "before": file_hash(before / rel / "acq_fcn_viqr.py"),
        "after": file_hash(ROOT / rel / "acq_fcn_viqr.py"),
    }
    assert (before / rel / "acq_fcn_viqr.py").read_text() != (
        ROOT / rel / "acq_fcn_viqr.py"
    ).read_text()
    rows = []
    statuses = [
        json.loads(path.read_text())
        for path in sorted((out / "snapshots").glob("*.status.json"))
    ]
    names = []
    for status in statuses:
        for filename, expected in status["file_hashes"].items():
            assert file_hash(out / "snapshots" / filename) == expected
        names.extend(
            f"{status['label']}_{checkpoint}"
            for checkpoint in status["captured"]
        )
    for name in names:
        state = restore(out / "snapshots" / name)
        assert state["meta"]["shared_source_hashes"] == shared_source_hashes()
        args = (
            state["gp"],
            state["vp"],
            state["logger"],
            state["optim_state"],
        )
        candidates = state["cand"]["Xs"]
        live = methods["after"](candidates.copy(), *args)
        np.testing.assert_array_equal(live, state["ref"]["acq"])
        setup_ms = []
        for _ in range(5):
            vp_copy = copy.deepcopy(state["vp"])
            vp_copy.rng = np.random.default_rng(20260913)
            start = time.perf_counter()
            active_importance_sampling(
                vp_copy, state["gp"], methods["after"], state["options"]
            )
            setup_ms.append(1000 * (time.perf_counter() - start))
        for count in (1, 4 + int(3 * np.log(state["gp"].D)), len(candidates)):
            xs = candidates[:count].copy()
            values = {
                key: method(xs.copy(), *args)
                for key, method in methods.items()
            }
            np.testing.assert_allclose(
                values["after"], values["before"], rtol=1e-12, atol=1e-12
            )
            rng_before = digest(state["vp"].rng.bit_generator.state)
            elapsed = {key: [] for key in methods}
            inner = 30 if count < 100 else 1
            for repeat in range(7):
                for key in (
                    ("before", "after")
                    if repeat % 2 == 0
                    else ("after", "before")
                ):
                    start = time.perf_counter()
                    for _ in range(inner):
                        methods[key](xs.copy(), *args)
                    elapsed[key].append(
                        (time.perf_counter() - start) * 1000 / inner
                    )
            assert rng_before == digest(state["vp"].rng.bit_generator.state)
            medians = {key: float(np.median(v)) for key, v in elapsed.items()}
            peak_bytes = {}
            for key, method in methods.items():
                tracemalloc.start()
                method(xs.copy(), *args)
                peak_bytes[key] = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
            count_rows = {"total": 0, "fallback": 0}
            helper = current_module._log_viqr_sum

            def counted(a):
                lower, upper = np.min(a, axis=1), np.max(a, axis=1)
                limit = np.log(np.finfo(np.float64).max) - np.log(
                    2 * a.shape[1]
                )
                safe = (
                    np.isfinite(lower)
                    & np.isfinite(upper)
                    & (lower >= 0)
                    & (upper <= limit)
                )
                count_rows["total"] += len(a)
                count_rows["fallback"] += int(np.count_nonzero(~safe))
                return helper(a)

            current_module._log_viqr_sum = counted
            try:
                methods["after"](xs.copy(), *args)
            finally:
                current_module._log_viqr_sum = helper
            finite = np.isfinite(values["after"]) & np.isfinite(
                values["before"]
            )
            error = (
                float(
                    np.max(
                        np.abs(
                            values["after"][finite] - values["before"][finite]
                        )
                    )
                )
                if np.any(finite)
                else None
            )
            row = {
                "snapshot": name,
                "Nc": count,
                "N": len(state["gp"].X),
                "Ns": len(state["gp"].posteriors),
                "milliseconds": elapsed,
                "median_ms": medians,
                "speedup": medians["before"] / medians["after"],
                "max_abs_difference": error,
                "setup_ms_unchanged_path": setup_ms,
                "peak_traced_bytes_call": peak_bytes,
                "sum_rows": count_rows,
            }
            rows.append(row)
            print(name, count, medians, "speedup", row["speedup"], flush=True)
    report = {
        "rows": rows,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "gpyreg_path": gpyreg.__file__,
        "threads": 1,
        "before_source": str(before),
        "after_source": str(ROOT),
        "viqr_sha256": viqr_hashes,
        "capture_provenance": provenance,
        "capture_status": statuses,
    }
    (out / "timing.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    assert rows, "No snapshots timed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "timing"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--before-source", type=Path)
    parser.add_argument(
        "--equivalent-viqr",
        action="store_true",
        help="allow a changed VIQR source; every captured output must still match exactly",
    )
    args = parser.parse_args()
    if args.mode == "replay":
        return replay(args.out)
    if args.before_source is None:
        parser.error("timing requires --before-source")
    return timing(args.out, args.before_source, args.equivalent_viqr)


if __name__ == "__main__":
    sys.exit(main())
