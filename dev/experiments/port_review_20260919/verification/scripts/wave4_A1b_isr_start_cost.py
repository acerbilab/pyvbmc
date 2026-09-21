"""W4-1: what the uniform starting point costs the IMIQR acquisition.

The MCMC step of `active_importance_sampling` starts each slice-sampling
chain at one of the proposal samples of its first step. PyVBMC draws that
sample uniformly (its resampling weights come out equal, W4-1); MATLAB draws
it in proportion to the importance weights. The chain is short at the
defaults: 100 recorded draws after a burn-in of 50. The two reviewers who
found the defect disagree on whether the burn-in absorbs a poor start.

This script measures it on two states: the stored noisy state of the oracles
(`rosenbrock_D2_noise1_viqr`, D = 2) and the state a short noisy run on a
six-dimensional benchmark target ends in. For single GP hyperparameter
samples it computes the IMIQR acquisition at a fixed set of candidate points
from the importance samples of (a) the function as it is and (b) a copy of
the function whose one line takes the maximum over the samples, MATLAB's
rule, over several seeds, the same seed for both. The reference is the same
acquisition from 20 000 importance samples of the first step alone (exact
importance weights under a normalized proposal, no MCMC), computed twice to
show its own error. The IMIQR value is a log and is defined up to a constant
that differs between the three, so every vector of values is centred over
the candidates before it is compared; what is left is what the search for
the next point sees.

Usage: python -u wave4_A1b_isr_start_cost.py stored|run6 [n_seeds]
Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import copy
import sys
import time
import types
from pathlib import Path

import gpyreg
import numpy as np

import pyvbmc
from pyvbmc.acquisition_functions import AcqFcnIMIQR
from pyvbmc.testing.oracles import _state
from pyvbmc.testing.oracles._oracles import prepare_gp_for_acq

print("pyvbmc:", pyvbmc.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)

which = sys.argv[1] if len(sys.argv) > 1 else "stored"
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20

# --- the function as it is, and a copy with MATLAB's rule ------------------
import importlib

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
RULES = {
    "uniform (PyVBMC)": ais_module.active_importance_sampling,
    "weighted (MATLAB)": matlab_rule.active_importance_sampling,
}


# --- the state --------------------------------------------------------------
def stored_state():
    fixtures = (
        Path(pyvbmc.__file__).parent / "testing" / "oracles" / "fixtures"
    )
    snap = _state.load_snapshot(fixtures / "rosenbrock_D2_noise1_viqr")
    st = _state.build_state(snap)
    return st["vp"], st["gp"], st["logger"], st["optim_state"], st["options"]


def run6_state():
    sys.path.insert(
        0, str(Path(pyvbmc.__file__).parent.parent / "dev" / "scripts")
    )
    from benchmark_targets import make_problem

    from pyvbmc import VBMC

    prob = make_problem("corr", 6, noise_sd=1.0, seed=20260920)
    options = dict(prob.options)
    options.update({"max_fun_evals": 130, "display": "off", "plot": False})
    vbmc = VBMC(
        prob.fun,
        prob.x0,
        prob.lb,
        prob.ub,
        prob.plb,
        prob.pub,
        options=options,
        seed=20260920,
    )
    t0 = time.time()
    vbmc.optimize()
    last = vbmc.iteration_history["iter"][-1]
    print(
        f"run6: corr_D6_noise1, {vbmc.function_logger.func_count} "
        f"evaluations, {int(last) + 1} iterations, {time.time() - t0:.0f} s",
        flush=True,
    )
    vp = copy.deepcopy(vbmc.iteration_history["vp"][int(last)])
    gp = vbmc.get_gp(int(last))
    return vp, gp, vbmc.function_logger, vbmc.optim_state, vbmc.options


vp, gp, logger, optim_state, options = (
    stored_state() if which == "stored" else run6_state()
)
D, N, Ns_gp = gp.D, gp.X.shape[0], len(gp.posteriors)
assert N == int(np.sum(logger.X_flag))
print(
    f"state {which}: D={D}, N={N}, Ns_gp={Ns_gp}, K={vp.K}; "
    f"mcmc_samples={options['active_importance_sampling_mcmc_samples']}, "
    f"thin={options['active_importance_sampling_mcmc_thin']}, "
    f"vp_samples={options['active_importance_sampling_vp_samples']}, "
    f"box_samples={options['active_importance_sampling_box_samples']}",
    flush=True,
)
prepare_gp_for_acq(gp, logger, optim_state)

# --- candidates: where a search would evaluate the acquisition -------------
cand_rng = np.random.default_rng(4)
vp._rng = np.random.default_rng(5)
X_vp, _ = vp.sample(128, orig_flag=False)
lo, hi = gp.X.min(0), gp.X.max(0)
X_box = lo + (hi - lo) * cand_rng.random((64, D))
X_near = gp.X[cand_rng.integers(0, N, 64)] + 0.1 * gp.X.std(
    0, ddof=1
) * cand_rng.standard_normal((64, D))
CAND = np.vstack([X_vp, X_box, X_near])

options_ref = copy.deepcopy(options)
for key, val in (
    ("active_importance_sampling_vp_samples", 10000),
    ("active_importance_sampling_box_samples", 10000),
    ("active_importance_sampling_mcmc_samples", 0),
):
    options_ref.__setitem__(key, val, force=True)


def acquisition(fun, gp_s, opts, seed):
    """IMIQR at the candidates from the importance samples `fun` draws."""
    acq = AcqFcnIMIQR()
    vp._rng = np.random.default_rng(seed)
    state = copy.copy(optim_state)
    state["active_importance_sampling"] = fun(vp, gp_s, acq, opts)
    return acq(np.array(CAND), gp_s, vp, logger, state).reshape(-1)


def centred(a, mask):
    return a[mask] - a[mask].mean()


samples = list(range(min(Ns_gp, 4)))
rows = []
for s in samples:
    gp_s = copy.deepcopy(gp)
    gp_s.posteriors = np.array([gp.posteriors[s]])
    t0 = time.time()
    ref_a = acquisition(RULES["uniform (PyVBMC)"], gp_s, options_ref, 101)
    ref_b = acquisition(RULES["uniform (PyVBMC)"], gp_s, options_ref, 202)
    mask = np.isfinite(ref_a) & np.isfinite(ref_b) & (np.abs(ref_a) < 1e100)
    ref = 0.5 * (centred(ref_a, mask) + centred(ref_b, mask))
    ref_err = np.sqrt(
        np.mean((centred(ref_a, mask) - centred(ref_b, mask)) ** 2)
    )
    print(
        f"\nhyperparameter sample {s}: {int(mask.sum())} candidates; spread "
        f"of the reference over them {ref.std():.3f} nats; the two "
        f"references differ by {ref_err:.4f} RMS ({time.time() - t0:.0f} s)",
        flush=True,
    )
    for seed in range(n_seeds):
        out = {}
        for label, fun in RULES.items():
            a = acquisition(fun, gp_s, options, 1000 + seed)
            a_c = centred(a, mask)
            out[label] = (
                np.sqrt(np.mean((a_c - ref) ** 2)),
                ref[np.argmin(a_c)] - ref.min(),
            )
        rows.append((s, seed, out))
        u, w = out["uniform (PyVBMC)"], out["weighted (MATLAB)"]
        print(
            f"  s={s} seed={seed:2d}  RMS error: uniform {u[0]:.4f}, "
            f"weighted {w[0]:.4f};  regret: uniform {u[1]:.4f}, "
            f"weighted {w[1]:.4f}",
            flush=True,
        )

print("\n=== summary over hyperparameter samples and seeds", flush=True)
for k, name in (
    (0, "RMS error of the centred acquisition (nats)"),
    (1, "regret at the estimate's best candidate (nats)"),
):
    u = np.array([r[2]["uniform (PyVBMC)"][k] for r in rows])
    w = np.array([r[2]["weighted (MATLAB)"][k] for r in rows])
    d = u - w
    print(
        f"{name}:\n  uniform  mean {u.mean():.4f}, median "
        f"{np.median(u):.4f}, max {u.max():.4f}\n  weighted mean "
        f"{w.mean():.4f}, median {np.median(w):.4f}, max {w.max():.4f}\n"
        f"  paired difference uniform - weighted: {d.mean():+.4f} "
        f"(SE {d.std(ddof=1) / np.sqrt(d.size):.4f}, n = {d.size}); "
        f"uniform worse in {int(np.sum(d > 0))} of {d.size}",
        flush=True,
    )
