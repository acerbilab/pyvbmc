"""Shared helpers for the wave-5 verification of slice P2.

Builds a small VBMC instance, runs the initial design through
``active_sample(gp=None, ...)`` and fits one GP, so that a single
active-sampling step can be taken on a synthetic state.  Every script that
imports this prints the imported checkout once.
"""
import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.acquisition_functions import AbstractAcqFcn
from pyvbmc.vbmc.active_sample import active_sample
from pyvbmc.vbmc.gaussian_process_train import train_gp


def banner():
    print("pyvbmc.__file__ =", pyvbmc.__file__)
    print("gpyreg.__file__ =", gpr.__file__)
    print("numpy", np.__version__)


TARGET_CALLS = []


def quad_target(x):
    TARGET_CALLS.append(np.asarray(x).copy())
    return -0.5 * float(np.sum(np.asarray(x) ** 2))


def noisy_target(x):
    TARGET_CALLS.append(np.asarray(x).copy())
    return -0.5 * float(np.sum(np.asarray(x) ** 2)), 1.0


class Cheap(AbstractAcqFcn):
    """Acquisition that prefers points near the origin (deterministic)."""

    def _compute_acquisition_function(self, Xs, *args):
        return np.sum(np.atleast_2d(Xs) ** 2, axis=1)


def build(
    D=2,
    options=None,
    target=quad_target,
    lb=None,
    ub=None,
    plb=-3.0,
    pub=3.0,
    x0=None,
    seed=7,
):
    lb = np.full((1, D), -np.inf) if lb is None else np.asarray(lb, float)
    ub = np.full((1, D), np.inf) if ub is None else np.asarray(ub, float)
    opts = {
        "active_sample_gp_update": False,
        "active_sample_vp_update": False,
    }
    opts.update(options or {})
    if x0 is None:
        x0 = np.zeros((1, D))
    return VBMC(
        target,
        np.asarray(x0, float),
        lb,
        ub,
        np.full((1, D), plb),
        np.full((1, D), pub),
        opts,
        seed=seed,
    )


def with_gp(v, n=12):
    """Initial design of ``n`` points plus one GP fit."""
    fl, os_, _, _ = active_sample(
        None,
        n,
        v.optim_state,
        v.function_logger,
        v.iteration_history,
        v.vp,
        v.options,
    )
    os_["N"] = fl.Xn + 1
    os_["n_eff"] = np.sum(fl.n_evals[fl.X_flag])
    gp, _, sn2hpd, hyp = train_gp(
        {},
        os_,
        fl,
        v.iteration_history,
        v.options,
        v.plausible_lower_bounds,
        v.plausible_upper_bounds,
        rng=v.vp.rng,
    )
    os_["hyp_dict"] = hyp
    os_["sn2_hpd"] = sn2hpd
    return gp, fl, os_


def set_acq(v, acq):
    v.options.__setitem__("search_acq_fcn", [acq], force=True)
