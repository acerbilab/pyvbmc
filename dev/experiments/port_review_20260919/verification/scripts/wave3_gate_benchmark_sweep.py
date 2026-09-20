"""Seed sweep on benchmark targets with known truth, before and after the
moving fixes of the wave-3 pass (W3-2, W3-3, W3-4, W3-5, W3-7).

The four seeded runs of `wave2_fixpass_gate_runs.py` are a fingerprint of the
trajectory, made for bit-for-bit comparison between commits; their targets
have no recorded truth, and one of them, the unscaled Rosenbrock function, is
far sharper than any problem of the benchmark. Whether a pass that moves
default trajectories changes the quality of the results is a question for
targets whose log evidence and posterior moments are known. This script runs
targets of `dev/scripts/benchmark_targets.py` at the default options for
several seeds and reports the metrics of the VBMC papers against the truth:
the error of the ELBO as an estimate of the log evidence, the Gaussianized
symmetrized KL divergence and the mean marginal total variation distance.

Run it once per code state, from the root of the main checkout, with
PYTHONPATH naming the checkout under test, and compare the tables.

Usage: python -u wave3_phase2_benchmark_sweep.py LABEL N_SEEDS [LABEL N_SEEDS ...]
Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.
"""

import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "dev/scripts")

from benchmark_targets import find_config, metrics  # noqa: E402

import pyvbmc  # noqa: E402
from pyvbmc import VBMC  # noqa: E402

print("pyvbmc:", pyvbmc.__file__, flush=True)
print(
    "HEAD:",
    subprocess.run(
        ["git", "log", "-1", "--format=%h %s"],
        capture_output=True,
        text=True,
        cwd=pyvbmc.__file__.rsplit("pyvbmc", 1)[0],
    ).stdout.strip(),
    flush=True,
)

pairs = sys.argv[1:]
for label, n_seeds in zip(pairs[0::2], pairs[1::2]):
    rows = []
    for seed in range(1, int(n_seeds) + 1):
        cfg = find_config(label)
        prob = cfg.make(seed=seed)
        args, options = prob.vbmc_args()
        options.update(
            display="off",
            plot=False,
            print_iteration_header=False,
            performance_calibration="off",
        )
        t0 = time.time()
        vbmc = VBMC(*args, options=options, seed=seed)
        vp, results = vbmc.optimize()
        m = metrics(prob, vp, results["elbo"])
        rows.append(
            (
                results["elbo"] - prob.ln_Z,
                m["gskl"],
                m["mmtv"],
                results["func_count"],
                results["iterations"],
            )
        )
        print(
            f"RESULT {label} seed {seed}: elbo - lnZ {rows[-1][0]:+.4f} "
            f"(elbo_sd {results['elbo_sd']:.4f})  gsKL {m['gskl']:.4f}  "
            f"MMTV {m['mmtv']:.4f}  evaluations {results['func_count']}  "
            f"iterations {results['iterations']}  stable "
            f"{results['convergence_status']}  ({time.time() - t0:.0f} s)",
            flush=True,
        )
    r = np.array(rows, dtype=float)
    print(
        f"SUMMARY {label} over {n_seeds} seeds: |elbo - lnZ| mean "
        f"{np.mean(np.abs(r[:, 0])):.4f} median "
        f"{np.median(np.abs(r[:, 0])):.4f} max {np.max(np.abs(r[:, 0])):.4f}; "
        f"gsKL mean {np.mean(r[:, 1]):.4f} median {np.median(r[:, 1]):.4f}; "
        f"MMTV mean {np.mean(r[:, 2]):.4f}; evaluations mean "
        f"{np.mean(r[:, 3]):.1f}",
        flush=True,
    )
