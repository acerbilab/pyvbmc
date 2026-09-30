"""W4-1: the weight each GP hyperparameter sample gets in IMIQR.

The MCMC step of `active_importance_sampling` returns one row of log
weights per GP hyperparameter sample, and `AcqFcnIMIQR` adds the rows'
contributions before it takes the log, so a row whose weights sum to far
less than another's counts for nothing in the acquisition. This script
measures the spread of the rows' totals, `log sum_a exp(ln_weights[s, a])`,
on the stored noisy state of the oracles (`Ns_gp = 8`), with the starting
point of each chain drawn uniformly, as PyVBMC draws it (W4-1), and drawn in
proportion to the importance weights, MATLAB's rule, over seeds, the same
seed for both. It also reports the effective number of rows, the inverse of
the sum of the squared shares of the rows, and the same two numbers for the
first step alone (no MCMC), where every row weighs the same points.

Usage: python -u wave4_A1c_isr_row_totals.py [n_seeds]
Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import copy
import importlib
import sys
import types
from pathlib import Path

import gpyreg
import numpy as np
from scipy.special import logsumexp

import pyvbmc
from pyvbmc.acquisition_functions import AcqFcnIMIQR
from pyvbmc.testing.oracles import _state

print("pyvbmc:", pyvbmc.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)

n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20

ais_module = importlib.import_module("pyvbmc.vbmc.active_importance_sampling")
source = Path(ais_module.__file__).read_text(encoding="utf-8")
OLD = "ln_weights_max = np.amax(ln_weights, axis=1).reshape(-1, 1)"
assert source.count(OLD) == 1
matlab_rule = types.ModuleType("ais_matlab_rule")
exec(
    compile(
        source.replace(OLD, "ln_weights_max = np.amax(ln_weights)"),
        "ais_matlab_rule.py",
        "exec",
    ),
    matlab_rule.__dict__,
)

fixtures = Path(pyvbmc.__file__).parent / "testing" / "oracles" / "fixtures"
state = _state.build_state(
    _state.load_snapshot(fixtures / "rosenbrock_D2_noise1_viqr")
)
vp, gp, options = state["vp"], state["gp"], state["options"]
options_step1 = copy.deepcopy(options)
options_step1.__setitem__(
    "active_importance_sampling_mcmc_samples", 0, force=True
)
print(f"state: D={gp.D}, N={gp.X.shape[0]}, Ns_gp={len(gp.posteriors)}")


def row_stats(fun, opts, seed):
    vp._rng = np.random.default_rng(seed)
    lnw = fun(vp, gp, AcqFcnIMIQR(), opts)["ln_weights"]
    totals = logsumexp(lnw, axis=1)
    share = np.exp(totals - logsumexp(totals))
    return totals.max() - totals.min(), 1.0 / np.sum(share**2)


cases = {
    "MCMC, uniform start (PyVBMC)": (
        ais_module.active_importance_sampling,
        options,
    ),
    "MCMC, weighted start (MATLAB)": (
        matlab_rule.active_importance_sampling,
        options,
    ),
    "first step alone, no MCMC": (
        ais_module.active_importance_sampling,
        options_step1,
    ),
}
out = {k: [] for k in cases}
for seed in range(n_seeds):
    for label, (fun, opts) in cases.items():
        out[label].append(row_stats(fun, opts, 3000 + seed))
    print(
        f"seed {seed:2d}: "
        + "; ".join(
            f"{k.split(',')[1].strip() if ',' in k else k}: "
            f"{v[-1][0]:.2f} nats, {v[-1][1]:.2f} rows"
            for k, v in out.items()
        ),
        flush=True,
    )

# What a row contributes to the acquisition is not its total weight but
# sum_a w[s, a] * 2 sinh(u * s_pred[s, a]): the acquisition evaluated with
# every other row's weights set to zero shows each row's level.
from pyvbmc.testing.oracles._oracles import prepare_gp_for_acq  # noqa: E402

prepare_gp_for_acq(gp, state["logger"], state["optim_state"])
vp._rng = np.random.default_rng(5)
CAND, _ = vp.sample(64, orig_flag=False)
print("\n=== level of each row's contribution to IMIQR (mean over 64")
print("    candidates of the acquisition with the other rows switched off)")
for label in list(cases)[:2]:
    fun, opts = cases[label]
    for seed in (3000, 3001, 3002):
        vp._rng = np.random.default_rng(seed)
        acq = AcqFcnIMIQR()
        ais = fun(vp, gp, acq, opts)
        totals = logsumexp(ais["ln_weights"], axis=1)
        levels = []
        for s in range(len(gp.posteriors)):
            only = copy.deepcopy(ais)
            mask = np.arange(len(gp.posteriors)) != s
            only["ln_weights"][mask, :] = -np.inf
            st = copy.copy(state["optim_state"])
            st["active_importance_sampling"] = only
            a = acq(np.array(CAND), gp, vp, state["logger"], st).reshape(-1)
            levels.append(np.mean(a[np.isfinite(a) & (np.abs(a) < 1e100)]))
        levels = np.array(levels)
        print(
            f"{label}, seed {seed}:\n   log totals of the rows' weights: "
            f"{np.array2string(totals, precision=1)}\n   levels of the "
            f"rows' contributions:    "
            f"{np.array2string(levels, precision=2)}  (spread "
            f"{levels.max() - levels.min():.2f} nats)",
            flush=True,
        )

print("\n=== spread of the rows' log totals, and effective number of rows")
for label, vals in out.items():
    a = np.array(vals)
    print(
        f"{label:32s} spread: median {np.median(a[:, 0]):.2f}, mean "
        f"{a[:, 0].mean():.2f}, max {a[:, 0].max():.2f} nats;  effective "
        f"rows of {len(gp.posteriors)}: median {np.median(a[:, 1]):.2f}, "
        f"min {a[:, 1].min():.2f}"
    )
