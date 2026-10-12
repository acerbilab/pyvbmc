"""Paired before/after timing of complete S-VBMC fits from two source trees.

Every cell is one ``SVBMC(vps, seed=seed).optimize(...)`` call on a fixture
group, run once from each source tree. Each tree is imported by its own
long-lived worker process (``PYTHONPATH`` set to the tree, one BLAS thread,
one Torch thread, warmed with an unrecorded two-step fit on ``upstream_GMM``
before any cell is timed); the controller alternates which source runs
first in every cell and never runs the two workers at once. Loading the
fixtures and constructing the object are timed separately from the
``optimize()`` call, which is the timed region the speedups refer to.

Every cell records wall and CPU time, the number of entropy evaluations,
the optimized weights, ELBO and entropy, and a digest of the generator
state after the fit. A pair passes when the weights agree within
``WEIGHT_TOLERANCE`` and the evaluation counts and generator digests are
identical. The report carries the provenance of both trees (commit,
working-tree state, imported module path, interpreter, library versions,
thread settings, a digest of the fixture files each worker read) and a
digest of this script.

Example
-------
::

    python dev/scripts/svbmc_speedup_benchmark.py \\
        --before dev/scripts/runs/svbmc_speedups_20260913/before \\
        --after . \\
        --output dev/scripts/runs/svbmc_speedups_20260913/benchmark.json

The plan is ``dev/plans/svbmc-speedups.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

GROUPS = ("upstream_GMM", "upstream_GMM_noisy", "upstream_Ring")
MODES = ("all-weights", "posterior-only")
SEEDS = (0, 1, 2)
WEIGHT_TOLERANCE = 1e-8
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------- #
# worker                                                                #
# --------------------------------------------------------------------- #
def _worker(source: Path) -> None:
    """Serve fit requests from stdin for the pyvbmc tree at ``source``."""
    # The protocol keeps a private copy of the original stdout; file
    # descriptor 1 is then pointed at stderr so that neither the library's
    # logging (``SVBMC`` configures logging on ``sys.stdout``) nor a C-level
    # write can reach the protocol.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    import logging

    import numpy as np
    import scipy
    import torch

    # Set before pyvbmc (and so the S-VBMC module) is imported.
    torch.set_num_threads(1)
    import pyvbmc
    from pyvbmc.svbmc import SVBMC
    from pyvbmc.testing.svbmc import _fixtures
    from pyvbmc.testing.svbmc._fixtures import load_group

    module = Path(pyvbmc.__file__).resolve()
    if module.parent.parent != source.resolve():
        raise RuntimeError(f"imported {module}, not the tree at {source}")
    logging.getLogger("SVBMC").setLevel(logging.WARNING)

    fixtures_dir = Path(_fixtures.FIXTURES_DIR)
    fixtures = hashlib.sha256()
    for path in sorted(fixtures_dir.iterdir()):
        if path.is_file():
            fixtures.update(path.name.encode())
            fixtures.update(path.read_bytes())

    def send(payload):
        protocol.write(json.dumps(payload) + "\n")
        protocol.flush()

    send(
        {
            "ready": True,
            "pyvbmc_file": str(module),
            "executable": sys.executable,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
            },
            "torch_threads": torch.get_num_threads(),
            "thread_variables": {
                name: os.environ.get(name) for name in THREAD_VARIABLES
            },
            "fixtures_dir": str(fixtures_dir),
            "fixtures_sha256": fixtures.hexdigest(),
        }
    )

    def fit(request):
        construction_started = time.perf_counter()
        vps = load_group(request["group"], rng=0)[0]
        stacked = SVBMC(vps, seed=int(request["seed"]))
        construction_seconds = time.perf_counter() - construction_started
        calls = 0
        evaluate = stacked._stacked_entropy

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return evaluate(*args, **kwargs)

        stacked._stacked_entropy = counted
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        stacked.optimize(
            n_samples=int(request["n_samples"]),
            lr=float(request["lr"]),
            max_steps=int(request["max_steps"]),
            version=request["mode"],
            n_samples_final=int(request["n_samples_final"]),
        )
        cpu_seconds = time.process_time() - cpu_started
        wall_seconds = time.perf_counter() - wall_started
        state = json.dumps(stacked.rng.bit_generator.state, sort_keys=True)
        details = {
            key: value
            for key, value in stacked.elbo_details.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        return {
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "load_and_construction_seconds": construction_seconds,
            "entropy_calls": calls,
            "w": np.ravel(stacked.w).tolist(),
            "elbo": float(stacked.elbo),
            "elbo_sd": float(stacked.elbo_sd),
            "entropy": float(stacked.entropy),
            "elbo_details": details,
            "rng_state_sha256": hashlib.sha256(state.encode()).hexdigest(),
        }

    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "quit":
            send({"bye": True})
            break
        if request["op"] == "warmup":
            fit(
                {
                    "group": "upstream_GMM",
                    "seed": 0,
                    "mode": "all-weights",
                    "n_samples": 20,
                    "n_samples_final": 2,
                    "lr": 0.1,
                    "max_steps": 2,
                }
            )
            send({"warm": True})
            continue
        send(fit(request))


class Worker:
    """One source tree served by a subprocess."""

    def __init__(self, label: str, source: Path):
        self.label = label
        self.source = source.resolve()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.source)
        env["PYTHONUNBUFFERED"] = "1"
        for name in THREAD_VARIABLES:
            env[name] = "1"
        self.process = subprocess.Popen(
            [sys.executable, "-u", __file__, "--worker", str(self.source)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        self.info = self._read()
        if not self.info.get("ready"):
            raise RuntimeError(f"{label}: worker did not start: {self.info}")

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.label}: worker exited")
        return json.loads(line)

    def request(self, payload):
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        return self._read()

    def close(self):
        """Ask the worker to quit; a dead worker is not an error here."""
        try:
            self.request({"op": "quit"})
        except (OSError, ValueError, RuntimeError):
            pass
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


# --------------------------------------------------------------------- #
# controller                                                            #
# --------------------------------------------------------------------- #
def _git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _provenance(source: Path, info: dict) -> dict:
    """Commit and working-tree state of ``source`` plus the worker's report.

    ``dirty`` inspects the package directory only; the script and the
    developer notes may differ from the commit without affecting a fit.
    """
    return {
        "path": str(source.resolve()),
        "commit": _git(source, "rev-parse", "HEAD"),
        "dirty": bool(_git(source, "status", "--porcelain", "--", "pyvbmc")),
        **info,
    }


def _compare(before: dict, after: dict) -> dict:
    w_before = before["w"]
    w_after = after["w"]
    same_length = len(w_before) == len(w_after)
    max_abs_w = (
        max(abs(a - b) for a, b in zip(w_before, w_after))
        if same_length
        else float("inf")
    )
    same_calls = before["entropy_calls"] == after["entropy_calls"]
    same_rng = before["rng_state_sha256"] == after["rng_state_sha256"]
    return {
        "speedup": before["wall_seconds"] / after["wall_seconds"],
        "max_abs_weight_difference": max_abs_w,
        "elbo_difference": after["elbo"] - before["elbo"],
        "entropy_difference": after["entropy"] - before["entropy"],
        "elbo_sd_difference": after["elbo_sd"] - before["elbo_sd"],
        "same_entropy_calls": same_calls,
        "same_rng_state": same_rng,
        "passed": (
            same_length
            and max_abs_w <= WEIGHT_TOLERANCE
            and same_calls
            and same_rng
        ),
    }


def _summarize(cells: list) -> dict:
    by_config = {}
    for cell in cells:
        key = f"{cell['group']} / {cell['mode']}"
        entry = by_config.setdefault(
            key, {"before_wall": [], "after_wall": [], "speedups": []}
        )
        entry["before_wall"].append(cell["before"]["wall_seconds"])
        entry["after_wall"].append(cell["after"]["wall_seconds"])
        entry["speedups"].append(cell["comparison"]["speedup"])
    for entry in by_config.values():
        entry["mean_before_wall"] = statistics.mean(entry["before_wall"])
        entry["mean_after_wall"] = statistics.mean(entry["after_wall"])
        entry["ratio_of_means"] = (
            entry["mean_before_wall"] / entry["mean_after_wall"]
        )
    speedups = [cell["comparison"]["speedup"] for cell in cells]
    total_before = sum(cell["before"]["wall_seconds"] for cell in cells)
    total_after = sum(cell["after"]["wall_seconds"] for cell in cells)
    return {
        "configurations": by_config,
        "total_before_wall": total_before,
        "total_after_wall": total_after,
        "aggregate_speedup": total_before / total_after,
        "min_speedup": min(speedups),
        "median_speedup": statistics.median(speedups),
        "max_speedup": max(speedups),
        "median_construction_before": statistics.median(
            cell["before"]["load_and_construction_seconds"] for cell in cells
        ),
        "median_construction_after": statistics.median(
            cell["after"]["load_and_construction_seconds"] for cell in cells
        ),
        "pairs": len(cells),
        "no_cell_slower": min(speedups) >= 1.0,
        "all_passed": all(cell["comparison"]["passed"] for cell in cells),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", type=Path, help="source tree before")
    parser.add_argument("--after", type=Path, help="source tree after")
    parser.add_argument("--output", type=Path, help="JSON report to write")
    parser.add_argument("--groups", nargs="+", default=list(GROUPS))
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--n-samples-final", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker is not None:
        _worker(args.worker)
        return 0
    if args.before is None or args.after is None or args.output is None:
        parser.error("--before, --after and --output are required")
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} exists; pass --overwrite to replace it")

    settings = {
        "n_samples": args.n_samples,
        "n_samples_final": args.n_samples_final,
        "lr": args.lr,
        "max_steps": args.max_steps,
    }
    report = {
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "harness": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__)),
        },
        "settings": settings,
        "weight_tolerance": WEIGHT_TOLERANCE,
        "sources": {},
        "cells": [],
    }

    def checkpoint():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))

    workers = {}
    try:
        for label, source in (("before", args.before), ("after", args.after)):
            workers[label] = Worker(label, source)
            report["sources"][label] = _provenance(
                workers[label].source, workers[label].info
            )
            workers[label].request({"op": "warmup"})
        cells = list(product(args.groups, args.modes, args.seeds))
        for index, (group, mode, seed) in enumerate(cells):
            order = ["before", "after"]
            if index % 2:
                order.reverse()
            request = {"op": "fit", "group": group, "mode": mode, "seed": seed}
            request.update(settings)
            results = {}
            for label in order:
                results[label] = workers[label].request(request)
            comparison = _compare(results["before"], results["after"])
            report["cells"].append(
                {
                    "group": group,
                    "mode": mode,
                    "seed": seed,
                    "order": order,
                    "before": results["before"],
                    "after": results["after"],
                    "comparison": comparison,
                }
            )
            checkpoint()
            print(
                f"[{index + 1}/{len(cells)}] {group} {mode} seed {seed}: "
                f"before {results['before']['wall_seconds']:.2f}s, "
                f"after {results['after']['wall_seconds']:.2f}s, "
                f"x{comparison['speedup']:.2f}, "
                f"max|dw| {comparison['max_abs_weight_difference']:.1e}, "
                f"calls {results['before']['entropy_calls']}/"
                f"{results['after']['entropy_calls']}, "
                f"{'ok' if comparison['passed'] else 'MISMATCH'}",
                flush=True,
            )
    finally:
        for worker in workers.values():
            worker.close()

    summary = _summarize(report["cells"])
    report["summary"] = summary
    report["status"] = "complete"
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    checkpoint()
    print(
        f"\n{summary['pairs']} pairs: "
        f"aggregate x{summary['aggregate_speedup']:.2f} "
        f"(min x{summary['min_speedup']:.2f}, "
        f"median x{summary['median_speedup']:.2f}, "
        f"max x{summary['max_speedup']:.2f}); "
        f"no cell slower: {summary['no_cell_slower']}; numerical checks "
        f"{'passed' if summary['all_passed'] else 'FAILED'}"
    )
    return 0 if summary["all_passed"] and summary["no_cell_slower"] else 1


if __name__ == "__main__":
    sys.exit(main())
