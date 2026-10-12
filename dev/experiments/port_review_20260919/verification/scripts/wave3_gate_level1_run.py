"""A short seeded run at uncertainty level 1, the check of the W3-1 fix.

With `uncertainty_handling=True` and no `specify_target_noise` the GP noise
is inferred: MATLAB's model is `exp(2*h1) + exp(h2)*s2` per training row,
with `s2 = 1/n_evals` from the function logger. Until the fix PyVBMC built
the noise function of a noiseless run for this level, so no PyVBMC run has
exercised the scaled provided-noise term through a whole run: the noisy
acquisition, the GP updates inside active sampling, the variational
optimization and the termination checks all see it here for the first time.

The target is a two-dimensional Gaussian log-density with standard
deviations 1 and 2, observed with Gaussian noise of SD 1. The script
reports, per iteration, the noise function of the recorded GP and its
fitted noise hyperparameters, and at the end the posterior moments beside
the truth. Run it on the commit before the fix and on the one after.

Usage: python -u wave3_level1_run.py [max_fun_evals]
Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 and
PYTHONPATH naming the checkout under test.
"""

import subprocess
import sys
import time

import numpy as np

import pyvbmc
from pyvbmc import VBMC

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

D = 2
SD_TRUE = np.array([1.0, 2.0])
NOISE_SD = 1.0
max_fun_evals = int(sys.argv[1]) if len(sys.argv) > 1 else 150
noise_rng = np.random.default_rng(20260920)


def target(x):
    x = np.atleast_2d(x)
    clean = -0.5 * np.sum((x / SD_TRUE) ** 2) - np.sum(np.log(SD_TRUE))
    return float(clean + NOISE_SD * noise_rng.normal())


vbmc = VBMC(
    target,
    np.array([[0.5, -0.5]]),
    np.full((1, D), -12.0),
    np.full((1, D), 12.0),
    np.full((1, D), -4.0),
    np.full((1, D), 4.0),
    options={
        "uncertainty_handling": True,
        "max_fun_evals": max_fun_evals,
        "display": "off",
    },
    seed=31,
)
print(
    "level:",
    vbmc.optim_state["uncertainty_handling_level"],
    "gp_noise_fun:",
    vbmc.optim_state["gp_noise_fun"],
    "search acquisition:",
    [type(a).__name__ for a in vbmc.options["search_acq_fcn"]],
    flush=True,
)

t0 = time.time()
vp, results = vbmc.optimize()
print(f"optimize() returned after {time.time() - t0:.0f} s", flush=True)

h = vbmc.iteration_history
print(
    "\niter  N  noise.parameters  n_noise_hyp  mean noise hyp  elbo  elbo_sd  r_index",
    flush=True,
)
for t in range(len(h["iter"])):
    gp = h["gp"][t]
    n_cov = gp.covariance.hyperparameter_count(D)
    n_noise = gp.noise.hyperparameter_count()
    hyp = gp.get_hyperparameters(as_array=True)
    noise_hyp = np.mean(hyp[:, n_cov : n_cov + n_noise], axis=0)
    print(
        f"{t:3d} {int(h['N'][t]):4d}  {gp.noise.parameters.tolist()}  "
        f"{n_noise}  {np.round(noise_hyp, 3)}  {h['elbo'][t]:8.3f}  "
        f"{h['elbo_sd'][t]:6.3f}  {h['r_index'][t]:7.3f}",
        flush=True,
    )

mean, cov = vp.moments(cov_flag=True)
print("\nmessage:", results["message"], flush=True)
print(
    "func_count:",
    results["func_count"],
    "iterations:",
    results["iterations"],
    flush=True,
)
print(
    "elbo:",
    results["elbo"],
    "+/-",
    results["elbo_sd"],
    "(true log evidence: log(2*pi) =",
    round(float(np.log(2 * np.pi)), 3),
    "since the target omits the Gaussian's constant)",
    flush=True,
)
print(
    "posterior mean:", np.round(np.ravel(mean), 3), "(truth 0, 0)", flush=True
)
print(
    "posterior SD:",
    np.round(np.sqrt(np.diag(cov)), 3),
    "(truth 1, 2)",
    flush=True,
)
fl = vbmc.function_logger
print("max n_evals of a row:", int(np.max(fl.n_evals[fl.X_flag])), flush=True)
