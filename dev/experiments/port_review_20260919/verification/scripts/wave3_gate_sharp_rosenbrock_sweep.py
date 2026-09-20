"""Seed sweep on the noiseless Rosenbrock gate run, before and after the
moving fixes of the wave-3 pass.

The single seeded gate run of `wave2_fixpass_gate_runs.py` on this target
ended with an ELBO 0.054 lower after the fixes of the hyperparameter fit
(W3-2, W3-3, W3-4, W3-5, W3-7) than before, against a reported SD of 0.0006,
and stopped 20 evaluations earlier. One seed cannot tell a shift of the
distribution from the luck of a trajectory, so this script runs the same
problem for several seeds. Run it once per code state, with PYTHONPATH naming
the checkout under test, and compare the two tables.

The true log evidence of the target comes from a quadrature on a grid and is
printed first.

Usage: python -u wave3_phase2_seed_sweep.py [n_seeds]
Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.
"""

import subprocess
import sys
import time

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc:", pyvbmc.__file__, flush=True)
root = pyvbmc.__file__.rsplit("pyvbmc", 1)[0]
print(
    "HEAD:",
    subprocess.run(
        ["git", "log", "-1", "--format=%h %s"],
        capture_output=True,
        text=True,
        cwd=root,
    ).stdout.strip(),
    flush=True,
)


def rosenbrock(x):
    x = np.atleast_2d(x)
    return float(
        -np.sum(
            100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2
        )
        - 0.5 * np.sum(x**2) / 9.0
    )


# True log evidence. The integral over the second variable is Gaussian and
# done in closed form: with a = x1**2, A = 100 + 1/18 and B = 200*a,
# int exp(-100*(x2 - a)**2 - x2**2/18) dx2 = sqrt(pi/A)*exp(B**2/(4*A) - 100*a**2).
# The integral over the first variable is a trapezoid rule on a fine grid.
g1 = np.linspace(-12, 12, 480001)
a = g1**2
A = 100.0 + 1.0 / 18.0
log_inner = (
    0.5 * np.log(np.pi / A) + (200.0 * a) ** 2 / (4 * A) - 100.0 * a**2
)
log_outer = log_inner - (1 - g1) ** 2 - g1**2 / 18.0
m = log_outer.max()
log_z = m + np.log(np.trapezoid(np.exp(log_outer - m), g1))
print(
    f"true log evidence (closed form and quadrature): {log_z:.4f}", flush=True
)

n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
D = 2
full = lambda v: np.full((1, D), float(v))  # noqa: E731
rows = []
for seed in range(1, n_seeds + 1):
    t0 = time.time()
    vbmc = VBMC(
        rosenbrock,
        full(0.0),
        full(-np.inf),
        full(np.inf),
        full(-3.0),
        full(3.0),
        options={"display": "off"},
        seed=seed,
    )
    vp, results = vbmc.optimize()
    warm = [
        i
        for i, a in enumerate(vbmc.iteration_history["logging_action"])
        if a is not None
        and any("end warm-up" in str(b) for b in np.atleast_1d(a))
    ]
    rows.append(
        (
            seed,
            results["elbo"],
            results["elbo_sd"],
            results["iterations"],
            results["func_count"],
            warm[0] if warm else -1,
            vp.K,
        )
    )
    print(
        f"seed {seed}: elbo {results['elbo']:.4f} +- {results['elbo_sd']:.4f}  "
        f"gap to truth {log_z - results['elbo']:.4f}  iterations "
        f"{results['iterations']}  evaluations {results['func_count']}  "
        f"end of warm-up {warm[0] if warm else -1}  K {vp.K}  "
        f"({time.time() - t0:.0f} s)",
        flush=True,
    )
elbo = np.array([r[1] for r in rows])
print(
    f"\nELBO over {n_seeds} seeds: mean {elbo.mean():.4f}, median "
    f"{np.median(elbo):.4f}, min {elbo.min():.4f}, max {elbo.max():.4f}; "
    f"mean gap to truth {np.mean(log_z - elbo):.4f}; mean evaluations "
    f"{np.mean([r[4] for r in rows]):.1f}",
    flush=True,
)
