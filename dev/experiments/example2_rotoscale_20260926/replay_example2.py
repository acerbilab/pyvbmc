"""Replay Example 2 and record what the rotoscaling of iteration 10 does.

Runs the notebook's seeded problem with its options, with ``train_gp``,
``optimize_vp`` and ``warp_gp_and_vp`` wrapped in the namespace of
``pyvbmc.vbmc.vbmc`` so that every GP fit, every variational optimization
and the warp are copied as they happen. The wrappers draw no random numbers,
so the run follows the notebook's trajectory; the script checks the trace
against the notebook's stored output. After the run it evaluates, for the
fits of iterations 8 to 14, the true ELBO of each posterior (Monte Carlo
against the real log joint), the share of its mass far from the training
inputs, its Gaussianized symmetrized KL divergence from the posterior of
the iteration before (the measure of the trace's ``sKL-iter[q]``), and the
GP hyperparameters, and writes them to ``summary.json`` beside this script,
with the commit of the checkout that the imported ``pyvbmc`` comes from.
Run it with BLAS single-threaded and a checkout on ``PYTHONPATH``, as
``dev/scripts/execute_notebooks.py`` runs the notebook::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        PYTHONPATH=. python -u \
        dev/experiments/example2_rotoscale_20260926/replay_example2.py
"""

import copy
import json
import os
import platform
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import gpyreg
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as scs

import pyvbmc
import pyvbmc.vbmc.vbmc as vbmc_module
from pyvbmc import VBMC

HERE = Path(__file__).resolve().parent
D = 2
PRIOR_TAU = 3 * np.ones((1, D))
LML_TRUE = -1.836
# The trace of the notebook as stored on 2026-09-26: iteration, Mean[ELBO].
NOTEBOOK_TRACE = {9: -1.98, 10: -0.82, 11: -104.79, 12: -2.44, 13: -2.03}
FIRST, LAST = 8, 14

print("pyvbmc", pyvbmc.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)

# The notebook shows each iteration's figure; closing it keeps the draws of
# the plot (which samples the posterior with the run's generator) and
# frees the memory.
plt.show = lambda *args, **kwargs: plt.close("all")

RECORDS = []
_train_gp = vbmc_module.train_gp
_optimize_vp = vbmc_module.optimize_vp
_warp_gp_and_vp = vbmc_module.warp_gp_and_vp
RUN = {}


def train_gp(*args, **kwargs):
    out = _train_gp(*args, **kwargs)
    RECORDS.append(
        {
            "kind": "train_gp",
            "iter": RUN["vbmc"].iteration,
            "gp": copy.deepcopy(out[0]),
        }
    )
    return out


def optimize_vp(*args, **kwargs):
    out = _optimize_vp(*args, **kwargs)
    RECORDS.append(
        {
            "kind": "optimize_vp",
            "iter": RUN["vbmc"].iteration,
            "vp": copy.deepcopy(out[0]),
            "fast": int(args[4]),
            "slow": int(args[5]),
        }
    )
    return out


def warp_gp_and_vp(parameter_transformer, gp_old, vp_old, vbmc):
    out = _warp_gp_and_vp(parameter_transformer, gp_old, vp_old, vbmc)
    RECORDS.append(
        {
            "kind": "warp",
            "iter": vbmc.iteration,
            "gp_old": copy.deepcopy(gp_old),
            "vp_old": copy.deepcopy(vp_old),
            "vp_new": copy.deepcopy(out[0]),
            "hyp_warped": out[1].copy(),
            "scale": np.array(parameter_transformer.scale),
            "R": np.array(parameter_transformer.R_mat),
        }
    )
    return out


vbmc_module.train_gp = train_gp
vbmc_module.optimize_vp = optimize_vp
vbmc_module.warp_gp_and_vp = warp_gp_and_vp


def log_likelihood(theta):
    theta = np.atleast_2d(theta)
    x, y = theta[:, :-1], theta[:, 1:]
    return -np.sum((x**2 - y) ** 2 + (x - 1) ** 2 / 100, axis=1)


def log_prior(x):
    return np.sum(scs.expon.logpdf(x, scale=PRIOR_TAU))


def log_joint(x):
    return log_likelihood(x) + log_prior(x)


def log_joint_rows(theta):
    theta = np.atleast_2d(theta)
    return log_likelihood(theta) + np.sum(
        scs.expon.logpdf(theta, scale=PRIOR_TAU), axis=1
    )


# The notebook's cells, in order.
np.random.seed(42)
LB = np.zeros((1, D))
UB = 10 * PRIOR_TAU
PLB = scs.expon.ppf(0.159, scale=PRIOR_TAU)
PUB = scs.expon.ppf(0.841, scale=PRIOR_TAU)
x0 = np.ones((1, D))
vbmc = VBMC(log_joint, x0, LB, UB, PLB, PUB, {"plot": True})
RUN["vbmc"] = vbmc
vp, results = vbmc.optimize()

elbo_trace = [float(e) for e in vbmc.iteration_history["elbo"]]
for iteration, value in NOTEBOOK_TRACE.items():
    assert round(elbo_trace[iteration], 2) == value, (iteration, value)
print("trace matches the notebook", flush=True)


# The analysis draws with generators of its own, after the run.
def with_rng(vp, seed):
    vp = copy.deepcopy(vp)
    vp.rng = np.random.default_rng(seed)
    return vp


def true_elbo(vp, n=200_000):
    """ELBO of the posterior against the real log joint, with its SE."""
    x, _ = with_rng(vp, 0).sample(n, orig_flag=True)
    log_q = vp.pdf(x, orig_flag=True, log_flag=True).ravel()
    values = log_joint_rows(x) - log_q
    return float(values.mean()), float(values.std() / np.sqrt(n))


def true_log_joint_in(transformer, X):
    """The real log joint in the inference space of ``transformer``."""
    return log_joint_rows(transformer.inverse(X)) + np.ravel(
        transformer.log_abs_det_jacobian(X)
    )


def hyperparameters(gp):
    """The samples of the SE-ARD, Gaussian-noise, negative-quadratic GP."""
    H = np.array([p.hyp for p in gp.posteriors])
    return {
        "samples": len(H),
        "ell_mean": np.exp(H[:, :D]).mean(0).tolist(),
        "sf": np.exp(H[:, D]).tolist(),
        "m0": H[:, D + 2].tolist(),
        "xm": H[:, D + 3 : 2 * D + 3].tolist(),
        "omega": np.exp(H[:, 2 * D + 3 : 3 * D + 3]).tolist(),
        "omega_upper_bound": np.exp(
            np.asarray(gp.upper_bounds)[2 * D + 3 : 3 * D + 3]
        ).tolist(),
    }


def against_gp(vp, gp, n=20_000):
    """Where the posterior puts its mass, and what the GP says there."""
    X, _ = with_rng(vp, 1).sample(n, orig_flag=False)
    f_gp = gp.predict(X, separate_samples=True, add_noise=False)[0].mean(1)
    f_true = true_log_joint_in(vp.parameter_transformer, X)
    ell = np.exp(np.mean([p.hyp[:D] for p in gp.posteriors], axis=0))
    distance = np.sqrt(
        (((X[:, None, :] - gp.X[None, :, :]) / ell) ** 2).sum(-1)
    ).min(1)
    far = distance > 3
    return {
        "mean_gp_under_q": float(f_gp.mean()),
        "mean_true_under_q": float(f_true.mean()),
        "share_beyond_3_ell": float(far.mean()),
        "mean_gp_beyond_3_ell": (
            float(f_gp[far].mean()) if far.any() else None
        ),
        "mean_true_beyond_3_ell": (
            float(f_true[far].mean()) if far.any() else None
        ),
        "distance_quantiles_50_90_99": np.quantile(
            distance, [0.5, 0.9, 0.99]
        ).tolist(),
        "training_inputs": int(len(gp.X)),
        "training_y_range": [float(gp.y.min()), float(gp.y.max())],
        "max_abs_y_error": float(
            np.max(
                np.abs(
                    gp.y.ravel()
                    - true_log_joint_in(vp.parameter_transformer, gp.X)
                )
            )
        ),
    }


fits = []
last_gp = None
for record in RECORDS:
    if not FIRST <= record["iter"] <= LAST:
        if record["kind"] == "train_gp":
            last_gp = record["gp"]
        continue
    if record["kind"] == "train_gp":
        last_gp = record["gp"]
        continue
    if record["kind"] == "warp":
        fits.append(
            {
                "kind": "warp",
                "iter": record["iter"],
                "scale": record["scale"].tolist(),
                "R": record["R"].tolist(),
                "true_elbo_before": true_elbo(record["vp_old"])[0],
                "true_elbo_after": true_elbo(record["vp_new"])[0],
                "gp_hyperparameters_before": hyperparameters(record["gp_old"]),
            }
        )
        continue
    vp_fit = record["vp"]
    value, se = true_elbo(vp_fit)
    previous = vbmc.iteration_history["vp"][record["iter"] - 1]
    skl = 0.5 * np.sum(
        with_rng(vp_fit, 2).kl_div(
            vp2=previous, N=int(1e5), gauss_flag=vbmc.options["kl_gauss"]
        )
    )
    fits.append(
        {
            "kind": "fit",
            "iter": record["iter"],
            "fast_starts": record["fast"],
            "slow_starts": record["slow"],
            "K": int(vp_fit.K),
            "reported_elbo": float(vp_fit.stats["elbo"]),
            "reported_elbo_sd": float(vp_fit.stats["elbo_sd"]),
            "true_elbo": value,
            "true_elbo_se": se,
            "skl_to_previous_iteration": float(skl),
            "gp": hyperparameters(last_gp),
            **against_gp(vp_fit, last_gp),
        }
    )
    print(
        f"iter {record['iter']:2d}: reported {vp_fit.stats['elbo']:9.3f} "
        f"+/- {vp_fit.stats['elbo_sd']:6.3f}, true {value:10.3f}, "
        f"far share {fits[-1]['share_beyond_3_ell']:.3f}, sKL {skl:.2f}",
        flush=True,
    )


def git(*args):
    """Git in the checkout that the imported ``pyvbmc`` comes from."""
    return subprocess.run(
        ["git", *args],
        cwd=Path(pyvbmc.__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    ).stdout.strip()


summary = {
    "provenance": {
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "pyvbmc")),
        "pyvbmc_file": pyvbmc.__file__,
        "gpyreg_file": gpyreg.__file__,
        "gpyreg_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(gpyreg.__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "host": platform.node(),
    },
    "true_log_evidence": LML_TRUE,
    "final_elbo": float(results["elbo"]),
    "final_elbo_sd": float(results["elbo_sd"]),
    "elbo_trace": elbo_trace,
    "elbo_sd_trace": [float(e) for e in vbmc.iteration_history["elbo_sd"]],
    "func_count_trace": [int(n) for n in vbmc.iteration_history["func_count"]],
    "fits": fits,
}
out = HERE / "summary.json"
out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
print("wrote", out, flush=True)
