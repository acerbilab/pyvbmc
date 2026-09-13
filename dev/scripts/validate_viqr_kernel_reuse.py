"""Compare actual before/after PyVBMC and gpyreg on captured VIQR states.

Two warmed workers import explicit source checkouts. The controller sends
one request at a time, so only one worker computes. Individual public-call
timings exclude imports, IPC, restoration and instrumentation. Captures and
the loaded production sources are hashed; no target is evaluated and no
importance points are redrawn. This is a developer-only measurement tool.
"""

import argparse
import base64
import hashlib
import json
import os
import pickle
import platform
import statistics
import subprocess
import sys
import time
import traceback
import tracemalloc
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
CAP_NAME = "_VIQR_KERNEL_CACHE_MAX_BYTES"
ALLOWED_CHANGES = {
    "pyvbmc/pyvbmc/acquisition_functions/abstract_acq_fcn.py",
    "pyvbmc/pyvbmc/acquisition_functions/acq_fcn_viqr.py",
    "gpyreg/gpyreg/gaussian_process.py",
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "-C",
            str(root),
            *args,
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def source_hashes(pyvbmc_root, gpyreg_root):
    # Git checkouts may differ only in CRLF conversion on Windows. Hash
    # normalized source text; capture archives retain their raw byte hashes.
    def source_hash(path):
        return hashlib.sha256(
            path.read_text(encoding="utf-8").encode()
        ).hexdigest()

    hashes = {}
    for label, root in (("pyvbmc", pyvbmc_root), ("gpyreg", gpyreg_root)):
        package = root / label
        for pattern in ("*.py", "*.ini"):
            for path in sorted(package.rglob(pattern)):
                if "testing" in path.relative_to(package).parts:
                    continue
                if path.name == "_version.py":
                    continue  # setuptools_scm-generated metadata, not numerics
                hashes[
                    f"{label}/{path.relative_to(root).as_posix()}"
                ] = source_hash(path)
    for relative in (
        "dev/scripts/validate_viqr_sinh.py",
        "pyvbmc/testing/oracles/_state.py",
    ):
        hashes[f"pyvbmc/{relative}"] = source_hash(pyvbmc_root / relative)
    return hashes


def captures(directories):
    entries = []
    for directory in directories:
        for path in sorted((directory / "snapshots").glob("*.status.json")):
            status = json.loads(path.read_text())
            for filename, expected in status["file_hashes"].items():
                assert file_hash(path.parent / filename) == expected, filename
            for checkpoint in status["captured"]:
                name = f"{status['label']}_{checkpoint}"
                entries.append(
                    {
                        "name": name,
                        "path": str((path.parent / name).resolve()),
                        "hashes": {
                            ext: file_hash(Path(str(path.parent / name) + ext))
                            for ext in (".npz", ".json")
                        },
                    }
                )
    assert entries, "No captures found"
    assert len({entry["name"] for entry in entries}) == len(entries)
    return entries


def packed(array):
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def pin_worker_cpu():
    """Keep both source pairs on the same logical CPU when supported.

    Separate workers can otherwise land on different core types on hybrid
    laptops, creating an apparent speed difference even for unchanged code.
    Both inherit the controller's allowed CPU set and select its first CPU.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetProcessAffinityMask.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel.SetProcessAffinityMask.argtypes = [
            wintypes.HANDLE,
            ctypes.c_size_t,
        ]
        handle = kernel.GetCurrentProcess()
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(
            handle, ctypes.byref(allowed), ctypes.byref(system)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        selected = allowed.value & -allowed.value
        if not kernel.SetProcessAffinityMask(handle, selected):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"logical_cpu": selected.bit_length() - 1, "mask": selected}
    if hasattr(os, "sched_getaffinity"):
        selected = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {selected})
        return {"logical_cpu": selected}
    return {"unsupported_platform": sys.platform}


def worker(args):
    affinity = pin_worker_cpu()
    # Import from the requested pair before importing any project code.
    sys.path[:0] = [
        str(args.pyvbmc_root / "dev/scripts"),
        str(args.pyvbmc_root),
        str(args.gpyreg_root),
    ]
    import gpyreg
    import numpy as np
    import scipy
    import validate_viqr_sinh as validation
    from threadpoolctl import threadpool_info

    import pyvbmc
    from pyvbmc.acquisition_functions import AcqFcnLog, AcqFcnVIQR
    from pyvbmc.acquisition_functions import acq_fcn_viqr as viqr_module

    assert (
        Path(pyvbmc.__file__).resolve().parent == args.pyvbmc_root / "pyvbmc"
    )
    assert (
        Path(gpyreg.__file__).resolve().parent == args.gpyreg_root / "gpyreg"
    )
    pools = threadpool_info()
    assert pools and all(
        p["num_threads"] == 1 for p in pools if p["user_api"] == "blas"
    )
    original_cap = getattr(viqr_module, CAP_NAME, None)
    if args.role == "after":
        assert original_cap == 128 * 1024**2

    def digest_state(state, xs):
        gp, vp = state["gp"], state["vp"]
        value = (
            xs,
            gp.X,
            gp.y,
            gp.posteriors,
            gp.temporary_data,
            vp.mu,
            vp.w,
            vp.sigma,
            vp.lambd,
            state["optim_state"],
            vp.rng.bit_generator.state,
            np.random.get_state(),
        )
        return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()

    info = {
        "role": args.role,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "threadpools": pools,
        "cpu_affinity": affinity,
        "module_paths": {"pyvbmc": pyvbmc.__file__, "gpyreg": gpyreg.__file__},
        "git": {
            name: {
                "sha": git(root, "rev-parse", "HEAD"),
                "describe": git(root, "describe", "--tags", "--always"),
                "status": git(root, "status", "--short"),
            }
            for name, root in (
                ("pyvbmc", args.pyvbmc_root),
                ("gpyreg", args.gpyreg_root),
            )
        },
        "source_hashes": source_hashes(args.pyvbmc_root, args.gpyreg_root),
        "cache_cap_bytes": original_cap,
    }
    print(json.dumps({"ok": True, "result": info}), flush=True)
    state = xs = call = before_digest = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request["op"]
            if operation == "close":
                break
            if operation == "prepare":
                if original_cap is not None:
                    setattr(
                        viqr_module,
                        CAP_NAME,
                        0
                        if request["method"] == "viqr_fallback"
                        else original_cap,
                    )
                path = Path(request["capture"])
                for ext, expected in request["capture_hashes"].items():
                    assert file_hash(Path(str(path) + ext)) == expected
                state = validation.restore(path)
                if args.role == "before":
                    assert (
                        state["meta"]["shared_source_hashes"]
                        == validation.shared_source_hashes()
                    )
                gp, vp = state["gp"], state["vp"]
                count = request["count"]
                xs = state["cand"]["Xs"][:count].copy()
                before_digest = digest_state(state, xs)
                acq = (
                    AcqFcnLog()
                    if request["method"] == "acq_log"
                    else AcqFcnVIQR()
                )
                if request["method"] == "gp_predict":
                    call = lambda: gp.predict(xs, separate_samples=True)
                else:
                    call = lambda: acq(
                        xs, gp, vp, state["logger"], state["optim_state"]
                    )
                requested = []

                def profile(frame, event, arg):
                    if (
                        event == "call"
                        and frame.f_code is gpyreg.GP.predict.__code__
                    ):
                        requested.append(
                            bool(
                                frame.f_locals.get(
                                    "return_cross_covariance", False
                                )
                            )
                        )

                previous = sys.getprofile()
                assert previous is None
                sys.setprofile(profile)
                try:
                    value = call()
                finally:
                    sys.setprofile(previous)
                use = (
                    args.role == "after"
                    and request["method"] == "viqr"
                    and 8 * len(gp.X) * count * len(gp.posteriors)
                    <= original_cap
                )
                assert requested == [use], requested
                outputs = value if isinstance(value, tuple) else (value,)
                assert all(a.dtype == np.float64 for a in outputs)
                if count == len(state["cand"]["Xs"]) and request[
                    "method"
                ].startswith("viqr"):
                    if args.role == "before":
                        np.testing.assert_array_equal(
                            outputs[0], state["ref"]["acq"]
                        )
                for _ in range(2):
                    call()  # untimed warmup
                result = {
                    "N": len(gp.X),
                    "Ns": len(gp.posteriors),
                    "D": gp.D,
                    "Nc": count,
                    "cache_requested": requested[0],
                    "retained_kernel_bytes": 8
                    * len(gp.X)
                    * count
                    * len(gp.posteriors)
                    if use
                    else 0,
                    "values": [packed(a) for a in outputs],
                }
            elif operation == "measure":
                times = []
                for _ in range(request.get("repetitions", 1)):
                    start = time.perf_counter_ns()
                    call()
                    times.append((time.perf_counter_ns() - start) / 1e6)
                result = times
            elif operation == "finish":
                tracemalloc.start()
                call()
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                assert (
                    digest_state(state, xs) == before_digest
                ), "State or RNG changed"
                result = {
                    "peak_traced_bytes": peak,
                    "state_and_rng_unchanged": True,
                }
                state = xs = call = before_digest = None
            else:
                raise ValueError(operation)
            print(json.dumps({"ok": True, "result": result}), flush=True)
        except Exception:
            print(
                json.dumps({"ok": False, "error": traceback.format_exc()}),
                flush=True,
            )
            return 1
    assert info["source_hashes"] == source_hashes(
        args.pyvbmc_root, args.gpyreg_root
    )
    return 0


class Worker:
    def __init__(self, role, pyvbmc_root, gpyreg_root, out):
        self.log = (out / f"{role}.stderr.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--role",
                role,
                "--pyvbmc-root",
                str(pyvbmc_root),
                "--gpyreg-root",
                str(gpyreg_root),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        try:
            self.info = self.receive()
        except BaseException:
            self.process.kill()
            self.process.wait()
            self.log.close()
            raise

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Worker exited: see {self.log.name}")
        response = json.loads(line)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def request(self, **request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.write('{"op":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=30)
            assert self.process.returncode == 0, self.log.name
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.log.close()


def compare_values(before, after):
    import numpy as np

    checks = []
    assert len(before) == len(after)
    for b, a in zip(before, after):
        assert b["shape"] == a["shape"] and b["dtype"] == a["dtype"]
        expected = np.frombuffer(
            base64.b64decode(b["data"]), dtype=b["dtype"]
        ).reshape(b["shape"])
        actual = np.frombuffer(
            base64.b64decode(a["data"]), dtype=a["dtype"]
        ).reshape(a["shape"])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        finite = np.isfinite(expected)
        checks.append(
            {
                "shape": b["shape"],
                "dtype": b["dtype"],
                "exact": b["data"] == a["data"],
                "max_abs_difference": float(
                    np.max(
                        np.abs(actual[finite] - expected[finite]), initial=0
                    )
                ),
                "argmin_equal": bool(np.argmin(actual) == np.argmin(expected)),
                "before_sha256": hashlib.sha256(
                    expected.tobytes()
                ).hexdigest(),
                "after_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            }
        )
    return checks


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    entries = captures(args.captures)
    workers = {}
    report = {
        "runner_sha256": file_hash(__file__),
        "captures": entries,
        "rounds": args.rounds,
        "small_pairs_per_round": args.small_pairs,
        "rows": [],
        "complete": False,
        "method": "Warmed isolated source pairs; individual calls alternate; one active worker",
    }
    try:
        for role in ("before", "after"):
            workers[role] = Worker(
                role,
                getattr(args, f"{role}_pyvbmc"),
                getattr(args, f"{role}_gpyreg"),
                args.out,
            )
        report["workers"] = {role: w.info for role, w in workers.items()}
        old, new = (
            workers[role].info["source_hashes"] for role in ("before", "after")
        )
        changed = {
            key
            for key in old.keys() | new.keys()
            if old.get(key) != new.get(key)
        }
        assert (
            changed <= ALLOWED_CHANGES
        ), f"Unexpected production changes: {changed - ALLOWED_CHANGES}"
        assert (
            changed == ALLOWED_CHANGES
        ), f"Expected both-package implementation: {changed}"
        report["changed_sources"] = sorted(changed)
        for key in (
            "python",
            "numpy",
            "scipy",
            "platform",
            "threadpools",
            "cpu_affinity",
        ):
            assert (
                workers["before"].info[key] == workers["after"].info[key]
            ), key
        for entry in entries:
            # Capture JSON carries D through the snapshot schema; prepare a
            # singleton first and derive population size from the restored GP.
            initial = workers["before"].request(
                op="prepare",
                capture=entry["path"],
                capture_hashes=entry["hashes"],
                count=1,
                method="viqr",
            )
            workers["before"].request(op="finish")
            import math

            population = 4 + int(math.floor(3 * math.log(initial["D"])))
            for count in (1, population, 8192):
                for method in (
                    "viqr",
                    "viqr_fallback",
                    "gp_predict",
                    "acq_log",
                ):
                    ready = {
                        role: w.request(
                            op="prepare",
                            capture=entry["path"],
                            capture_hashes=entry["hashes"],
                            count=count,
                            method=method,
                        )
                        for role, w in workers.items()
                    }
                    numerical = compare_values(
                        ready["before"].pop("values"),
                        ready["after"].pop("values"),
                    )
                    samples = {role: [] for role in workers}
                    pairs = args.small_pairs if count < 100 else 1
                    for round_id in range(args.rounds):
                        for pair in range(pairs):
                            order = (
                                ("before", "after")
                                if (round_id + pair) % 2 == 0
                                else ("after", "before")
                            )
                            for role in order:
                                samples[role].extend(
                                    workers[role].request(op="measure")
                                )
                    peaks = {
                        role: w.request(op="finish")
                        for role, w in workers.items()
                    }
                    row = {
                        "snapshot": entry["name"],
                        "method": method,
                        "Nc": count,
                        "N": ready["before"]["N"],
                        "Ns": ready["before"]["Ns"],
                        "D": initial["D"],
                        "numerical": numerical,
                        "calls": ready,
                        "memory": peaks,
                        "milliseconds": samples,
                        "paired_median_speedup": statistics.median(
                            b / a
                            for b, a in zip(
                                samples["before"], samples["after"]
                            )
                        ),
                    }
                    report["rows"].append(row)
                    (args.out / "results.json").write_text(
                        json.dumps(report, indent=2) + "\n", encoding="utf-8"
                    )
                    print(
                        entry["name"],
                        count,
                        method,
                        round(row["paired_median_speedup"], 4),
                        "exact=" + str(all(c["exact"] for c in numerical)),
                        flush=True,
                    )
        report["complete"] = True
    finally:
        close_errors = []
        for worker_instance in workers.values():
            try:
                worker_instance.close()
            except Exception as error:
                close_errors.append(str(error))
        if close_errors:
            report["complete"] = False
            report["close_errors"] = close_errors
        (args.out / "results.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        if close_errors:
            raise RuntimeError(
                "Worker close failed: " + "; ".join(close_errors)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--role", choices=("before", "after"))
    parser.add_argument("--pyvbmc-root", type=Path)
    parser.add_argument("--gpyreg-root", type=Path)
    parser.add_argument("--before-pyvbmc", type=Path)
    parser.add_argument("--before-gpyreg", type=Path)
    parser.add_argument("--after-pyvbmc", type=Path, default=ROOT)
    parser.add_argument("--after-gpyreg", type=Path)
    parser.add_argument("--captures", type=Path, action="append")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--small-pairs", type=int, default=31)
    parsed = parser.parse_args()
    required = (
        ("role", "pyvbmc_root", "gpyreg_root")
        if parsed.worker
        else (
            "before_pyvbmc",
            "before_gpyreg",
            "after_pyvbmc",
            "after_gpyreg",
            "captures",
            "out",
        )
    )
    for name in required:
        if getattr(parsed, name) is None:
            parser.error("required: --" + name.replace("_", "-"))
    if parsed.rounds < 1 or parsed.small_pairs < 1:
        parser.error("rounds and small-pairs must be positive")
    for name, value in vars(parsed).items():
        if isinstance(value, Path):
            setattr(parsed, name, value.resolve())
    if parsed.worker:
        sys.exit(worker(parsed))
    run(parsed)
