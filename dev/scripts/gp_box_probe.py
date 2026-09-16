"""Measure GP conditioning and tail evaluations in one benchmark run.

Run each case in a fresh process, selecting a frozen package or the working
checkout with ``--source-root``. For the box-sampler comparison, use the
same gpyreg source and seeds in both arms. Example::

    python dev/scripts/gp_box_probe.py --source-root . --gpyreg-source ../gpyreg \
        --seed 1000 --out dev/scripts/runs/gp_box_20260916/uniform

The JSON records per-iteration and returned-GP conditioning, live training
target ranges, search bounds in transformed coordinates, inference metrics,
and source hashes. The NPZ retains every target call in original coordinates,
including evaluations later trimmed from the logger. A saved VBMC instance
permits examination of the histories. Diagnostics run after optimization.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path


def source_hash(root):
    """Hash Python and option sources, excluding tests and bytecode."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.suffix not in (".py", ".ini") or "testing" in relative.parts:
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gpyreg-source", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", default="rosenbrock_D2_noise3_svbmc")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path[:0] = [
        str(args.source_root.resolve()),
        str(args.gpyreg_source.resolve()),
    ]

    import gpyreg
    import numpy as np
    import scipy
    from benchmark_targets import find_config, metrics
    from profile_run import effective_options, jsonable

    import pyvbmc

    for module, root in (
        (pyvbmc, args.source_root),
        (gpyreg, args.gpyreg_source),
    ):
        expected = root.resolve() / module.__name__
        if Path(module.__file__).resolve().parent != expected:
            raise RuntimeError(f"Wrong import path for {module.__name__}")

    # This helper adds the working checkout to sys.path on import, so load
    # the selected package before importing it.
    from svbmc_pool_io import gp_condition_number

    tag = f"{args.config}_seed{args.seed}"
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.glob(f"{tag}.*")):
        raise FileExistsError(f"Outputs already exist for {tag}")
    problem = find_config(args.config).make(seed=args.seed)
    vbmc_args, options = problem.vbmc_args()
    options.update(
        display="off",
        plot=False,
        print_iteration_header=False,
        performance_calibration="off",
    )
    calls_x, calls_y = [], []
    target = vbmc_args[0]

    def recorded_target(x):
        value = target(x)
        calls_x.append(np.asarray(x).reshape(-1).copy())
        calls_y.append(float(value[0] if isinstance(value, tuple) else value))
        return value

    vbmc = pyvbmc.VBMC(
        recorded_target, *vbmc_args[1:], options=options, seed=args.seed
    )
    sources = {
        "pyvbmc": source_hash(Path(pyvbmc.__file__).parent),
        "gpyreg": source_hash(Path(gpyreg.__file__).parent),
        "probe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_targets": hashlib.sha256(
            Path(__file__).with_name("benchmark_targets.py").read_bytes()
        ).hexdigest(),
    }
    started = time.perf_counter()
    vp, results = vbmc.optimize()
    wall = time.perf_counter() - started
    vbmc.save(args.out / f"{tag}.vbmc.pkl")
    np.savez_compressed(
        args.out / f"{tag}.npz", X=np.array(calls_x), y=np.array(calls_y)
    )

    def gp_diagnostics(gp):
        y = gp.y.reshape(-1)
        return {
            "N": len(y),
            "condition": gp_condition_number(gp),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "tail_count": int(np.sum(y < y.max() - 1000)),
        }

    history = vbmc.iteration_history
    iterations = []
    for i in range(len(history["gp"])):
        state = history["optim_state"][i]
        iterations.append(
            {
                "iteration": i,
                "func_count": int(history["func_count"][i]),
                "warmup": bool(history["warmup"][i]),
                "lb_search": jsonable(state["lb_search"]),
                "ub_search": jsonable(state["ub_search"]),
                **gp_diagnostics(vbmc.get_gp(i)),
            }
        )
    measured = metrics(problem, vp, results["elbo"])
    y = np.array(calls_y)
    tail_indices = np.flatnonzero(y < y.max() - 1000)
    result = {
        "config": args.config,
        "seed": args.seed,
        "sources": sources,
        "source_root": str(args.source_root.resolve()),
        "gpyreg_source": str(args.gpyreg_source.resolve()),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "blas_threads": 1,
        },
        "options": effective_options(vbmc, options.keys()),
        "results": results,
        "metrics": measured,
        "wall_s": wall,
        "all_calls": {
            "count": len(y),
            "y_min": float(y.min()),
            "tail_count": len(tail_indices),
            "first_tail_call": (
                int(tail_indices[0] + 1) if len(tail_indices) else None
            ),
        },
        "returned_gp": gp_diagnostics(vbmc.get_gp(results["best_iter"])),
        "iterations": iterations,
    }
    (args.out / f"{tag}.json").write_text(
        json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            jsonable(
                {
                    "seed": args.seed,
                    "wall_s": wall,
                    "all_calls": result["all_calls"],
                    "returned_gp": result["returned_gp"],
                    "metrics": measured,
                }
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
