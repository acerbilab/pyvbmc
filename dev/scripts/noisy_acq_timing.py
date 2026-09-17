"""Paired selection timings with separate allocation measurements.

A preparation callback creates private experiment state and returns a
zero-argument selection operation or a context manager yielding one.
That operation must include every node
draw, integration-cache construction and acquisition/search evaluation.
Snapshot restoration, copying for isolation and result serialization belong
in preparation or the caller. No cache built by the operation is reused in
another round.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import tracemalloc
from contextlib import nullcontext
from pathlib import Path

import numpy as np


def _thread_contract():
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    values = {name: os.environ.get(name) for name in names}
    if any(value != "1" for value in values.values()):
        raise RuntimeError(
            "paired timings require all three BLAS thread pins at one"
        )
    return values


def _seeds(seed, count):
    return [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in np.random.SeedSequence(int(seed)).spawn(count)
    ]


def _prepared_context(prepared):
    return (
        prepared if hasattr(prepared, "__enter__") else nullcontext(prepared)
    )


def time_pair(
    prepare_baseline, prepare_treatment, *, seed, clock=time.perf_counter
):
    """Warm both arms, then measure seven rounds in alternating order.

    Each round receives a fresh seed shared by the two preparation callbacks.
    Factories may derive additional seeds for method-specific integration
    streams while retaining the shared candidate-generation seed. Return
    values from selection operations are retained as provenance; callers
    should return only small records such as the selected point and actual
    node count. Quality judging is a separate experiment.
    """
    threads = _thread_contract()
    factories = {"baseline": prepare_baseline, "treatment": prepare_treatment}
    seeds = _seeds(seed, 8)
    warmup = {}
    for name in factories:
        with _prepared_context(factories[name](seeds[0])) as operation:
            operation()
        warmup[name] = seeds[0]
        del operation
    rows = []
    for round_index in range(7):
        order = ["baseline", "treatment"]
        if round_index % 2:
            order.reverse()
        row = {
            "round": round_index,
            "order": order,
            "paired_seed": seeds[round_index + 1],
        }
        for name in order:
            role_seed = seeds[round_index + 1]
            with _prepared_context(factories[name](role_seed)) as operation:
                started = clock()
                result = operation()
                elapsed = clock() - started
            if elapsed <= 0 or not np.isfinite(elapsed):
                raise RuntimeError("invalid selection timing")
            row[name] = {
                "seed": role_seed,
                "seconds": elapsed,
                "result": result,
            }
            del operation
        row["ratio_treatment_over_baseline"] = (
            row["treatment"]["seconds"] / row["baseline"]["seconds"]
        )
        rows.append(row)
    baseline = np.array([row["baseline"]["seconds"] for row in rows])
    treatment = np.array([row["treatment"]["seconds"] for row in rows])
    ratios = treatment / baseline
    return {
        "seed": int(seed),
        "thread_environment": threads,
        "warmup_seeds": warmup,
        "rounds": rows,
        "baseline_median_seconds": float(np.median(baseline)),
        "treatment_median_seconds": float(np.median(treatment)),
        "median_paired_ratio": float(np.median(ratios)),
        "paired_ratio_range": [float(np.min(ratios)), float(np.max(ratios))],
        "quality_claim": False,
    }


def _windows_memory():
    """Return process resident memory, when the Windows API is available."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counter = Counters()
    counter.cb = ctypes.sizeof(counter)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "resident_bytes": int(counter.WorkingSetSize),
        "process_lifetime_peak_resident_bytes": int(
            counter.PeakWorkingSetSize
        ),
    }


def measure_allocations(prepare, *, seed):
    """Measure allocations in an untimed selection call.

    Tracemalloc reports traced Python/NumPy allocations; it does not cover
    every native-library allocation. Windows resident-memory readings are
    also retained, with the peak explicitly labelled as process lifetime.
    Run each arm in a fresh process for comparable process-peak figures.
    """
    _thread_contract()
    if tracemalloc.is_tracing():
        raise RuntimeError(
            "memory measurement requires an inactive tracemalloc tracer"
        )
    with _prepared_context(prepare(int(seed))) as operation:
        gc.collect()
        before = _windows_memory()
        tracemalloc.start()
        try:
            result = operation()
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    return {
        "seed": int(seed),
        "result": result,
        "traced_current_bytes": int(current),
        "traced_peak_bytes": int(peak),
        "windows_before": before,
        "windows_after": _windows_memory(),
        "timing_evidence": False,
    }


def _cell_identity(manifest, descriptor, method, budget, mode):
    import noisy_acq_integration as integration

    integration.validate_manifest(manifest, require_ready=True)
    identity = integration.runtime_identity(manifest)
    setting = {"method": method, "budget": int(budget)}
    if setting not in manifest["settings"]:
        raise ValueError(
            "timing setting is outside the frozen integration allocation"
        )
    return {
        "integration_manifest_sha256": integration.manifest_digest(manifest),
        "source_identity": identity,
        "timing_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "state": descriptor,
        "setting": setting,
        "mode": mode,
    }


def _paths(out, state_id, method, budget, mode):
    tag = f"{state_id}__{method}__b{budget}__{mode}"
    return out / f"{tag}.json", out / f"{tag}.complete.json"


def _read_completed(report_path, complete_path, identity):
    import noisy_acq_integration as integration

    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete["identity"] != integration.capture.canonical(identity):
        raise RuntimeError("timing completion belongs to another identity")
    if (
        not report_path.exists()
        or integration.sha256_file(report_path) != complete["report_sha256"]
    ):
        raise RuntimeError("timing report is missing or changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report["identity"] != complete["identity"]
        or report["status"] != complete["status"]
        or complete["status"] not in {"succeeded", "failed"}
    ):
        raise RuntimeError("inconsistent timing completion and report")
    return complete, report


def run_integration_cell(
    manifest, out, state_id, method, budget, mode="timing"
):
    """Measure one frozen integration setting with hash-checked resumption."""
    import noisy_acq_experiment as capture
    import noisy_acq_integration as integration

    descriptor = next(
        item for item in manifest["states"] if item["state_id"] == state_id
    )
    identity = _cell_identity(manifest, descriptor, method, budget, mode)
    report_path, complete_path = _paths(out, state_id, method, budget, mode)
    if complete_path.exists():
        complete, _ = _read_completed(report_path, complete_path, identity)
        return complete
    if report_path.exists():
        raise RuntimeError("partial timing report requires inspection")
    state = capture.restore_capture(descriptor["snapshot"])
    candidates, _, _, _ = integration.candidate_panel(
        state, descriptor, manifest["stage"]
    )
    before = integration.state_digest(state, candidates)
    seed = integration.derive_seed(
        integration.MASTER_SEED,
        manifest["stage"],
        state_id,
        method,
        budget,
        mode,
    )

    def baseline(role_seed):
        return integration.prepare_production_operation(
            state, candidates, role_seed
        )

    def treatment(role_seed):
        return integration.prepare_treatment_operation(
            state, candidates, method, budget, role_seed
        )

    try:
        if mode == "timing":
            result = time_pair(baseline, treatment, seed=seed)
        elif mode in {"memory_baseline", "memory_treatment"}:
            # Load the same numerical implementation in both fresh workers
            # before recording process memory or traced allocations.
            __import__("noisy_acq_quadrature")
            result = measure_allocations(
                baseline if mode == "memory_baseline" else treatment, seed=seed
            )
        else:
            raise ValueError("unknown measurement mode")
        if integration.state_digest(state, candidates) != before:
            raise RuntimeError("timing changed captured state or its RNG")
        if (
            _cell_identity(manifest, descriptor, method, budget, mode)
            != identity
        ):
            raise RuntimeError("load-bearing source changed during timing")
        status = "succeeded"
    except Exception as error:
        status = "failed"
        result = {"type": type(error).__name__, "message": str(error)}
    integration.write_json(
        report_path, {"identity": identity, "status": status, "result": result}
    )
    complete = {
        "identity": identity,
        "status": status,
        "report_sha256": integration.sha256_file(report_path),
    }
    integration.write_json(complete_path, complete)
    return complete


def main(argv=None):
    """Run integration timing or memory cells in fresh sequential workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "cell", "summary"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-id")
    parser.add_argument("--method")
    parser.add_argument("--budget", type=int)
    parser.add_argument(
        "--mode",
        choices=["timing", "memory_baseline", "memory_treatment"],
        default="timing",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    import noisy_acq_integration as integration

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    integration.validate_manifest(manifest, require_ready=True)
    integration.runtime_identity(manifest)
    if args.command == "cell":
        if None in (args.state_id, args.method, args.budget):
            parser.error("cell requires --state-id, --method and --budget")
        result = run_integration_cell(
            manifest,
            args.out,
            args.state_id,
            args.method,
            args.budget,
            args.mode,
        )
        return int(result["status"] != "succeeded")
    allocation = [
        (state["state_id"], setting["method"], setting["budget"])
        for state in manifest["states"]
        for setting in manifest["settings"]
    ]
    run_allocation = allocation
    if args.limit is not None:
        run_allocation = allocation[: args.limit]
    failures = 0
    if args.command == "run":
        for state_id, method, budget in run_allocation:
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "cell",
                "--manifest",
                str(args.manifest.resolve()),
                "--out",
                str(args.out.resolve()),
                "--state-id",
                state_id,
                "--method",
                method,
                "--budget",
                str(budget),
                "--mode",
                args.mode,
            ]
            result = subprocess.run(command)
            failures += result.returncode != 0
            print(
                f"{'FAILED' if result.returncode else 'DONE'} {state_id} {method} {budget} {args.mode}",
                flush=True,
            )
    rows = []
    for state_id, method, budget in allocation:
        report_path, complete_path = _paths(
            args.out, state_id, method, budget, args.mode
        )
        if not complete_path.exists():
            rows.append(
                {
                    "state_id": state_id,
                    "method": method,
                    "budget": budget,
                    "status": "missing",
                }
            )
            continue
        descriptor = next(
            item for item in manifest["states"] if item["state_id"] == state_id
        )
        expected = _cell_identity(
            manifest, descriptor, method, budget, args.mode
        )
        _, report = _read_completed(report_path, complete_path, expected)
        rows.append(
            {
                "state_id": state_id,
                "method": method,
                "budget": budget,
                **report,
            }
        )
    scheduled = manifest["expected_state_count"] * len(manifest["settings"])
    unavailable = scheduled - len(allocation)
    integration.write_json(
        args.out / f"summary_{args.mode}.json",
        {
            "mode": args.mode,
            "rows": rows,
            "scheduled": scheduled,
            "observed_cells": sum(row["status"] != "missing" for row in rows),
            "counts": {
                status: sum(row["status"] == status for row in rows)
                + (unavailable if status == "missing" else 0)
                for status in ("succeeded", "failed", "missing")
            },
            "unavailable_state_cells": unavailable,
            "quality_claim": False,
        },
    )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
