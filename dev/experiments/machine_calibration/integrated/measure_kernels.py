import importlib.util
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np

from pyvbmc.calibration import _campaign as c

root = Path("dev/scripts/runs/calibration_integration_20260909")


def module(name, relative):
    spec = importlib.util.spec_from_file_location(
        name, root / "baseline_source" / relative
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


oldvp = module(
    "baseline_vp", "pyvbmc/variational_posterior/variational_posterior.py"
)
oldent = module("baseline_entropy", "pyvbmc/entropy/entmc_vbmc.py")
api = c._load_kernel_api()
rows = []
for i, workload in enumerate(c._workloads()):
    problem = c._make_problem(workload, api, 50000 + i)
    vp = problem[0]

    def call(old, rng):
        if workload.group == "pdf":
            f = oldvp.VariationalPosterior.pdf if old else type(vp).pdf
            return f(
                vp,
                problem[1],
                orig_flag=False,
                grad_flag=workload.grad_flags is not None,
            )
        f = oldent.entmc_vbmc if old else api.public_entropy
        return f(
            vp,
            workload.count,
            grad_flags=workload.grad_flags,
            jacobian_flag=workload.jacobian_flag,
            rng=rng,
        )

    call(True, np.random.default_rng(123))
    call(False, np.random.default_rng(123))
    repeats = 30 if workload.name == "small_value" else 1
    durations = {
        name: [] for name in ("old", "new", "old_control", "new_control")
    }
    for round_index in range(12):
        for arm in c._heldout_orders(12)[round_index]:
            name = ("old", "new", "old_control", "new_control")[arm]
            old = arm in (0, 2)
            rngs = [np.random.default_rng(9876 + j) for j in range(repeats)]
            start = time.perf_counter()
            for rng in rngs:
                call(old, rng)
            durations[name].append((time.perf_counter() - start) / repeats)
    memory = {}
    for budget in (65536, 262144):
        tracemalloc.start()
        c._invoke(workload, problem, api, budget, 12345)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory[str(budget)] = {
            "peak_traced_bytes": peak,
            "estimated_bytes": c._workspace_estimate(workload, budget)[
                "bytes"
            ],
        }
    rows.append(
        {
            "workload": workload.name,
            "timings_seconds": durations,
            "median_old_over_new": statistics.median(
                a / b for a, b in zip(durations["old"], durations["new"])
            ),
            "same_version_controls": {
                name: statistics.median(
                    a / b
                    for a, b in zip(
                        durations[name], durations[name + "_control"]
                    )
                )
                for name in ("old", "new")
            },
            "memory": memory,
        }
    )
    print(workload.name, rows[-1]["median_old_over_new"], memory, flush=True)
report = {
    "baseline_commit": "f7f0ce1af1664571dc55f2a7c69f14124b85db88",
    "method": "same-process public kernels, 12 position- and pair-balanced four-arm rounds (old/new plus same-version controls); warmup separate; private RNG creation outside timing; incremental tracemalloc peak separate from timings, not RSS; workloads preallocated",
    "rows": rows,
}
(root / "kernel_overhead_memory.json").write_text(json.dumps(report, indent=2))
