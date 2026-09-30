"""W5, an acquired point equal to a logger row that is not live.

`FunctionLogger._record` looks for an earlier evaluation of the same input
over the whole array of inputs, `(self.X == x).all(axis=1)`
(`function_logger.py:685`), as `misc/funlogger_vbmc.m:220` does, and the trim
at the end of warm-up switches rows off through `X_flag` and leaves their
inputs in place (`vbmc.py`, `_trim`; `private/vbmc_warmup.m:126`). This
script settles what one active-sampling step does when the point it acquires
equals a row that is switched off: a state with 12 evaluated points, one of
them switched off, and a search whose one candidate is that input.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging

import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.acquisition_functions import AbstractAcqFcn
from pyvbmc.vbmc import active_sample
from pyvbmc.vbmc.gaussian_process_train import train_gp

print("pyvbmc:", pyvbmc.__file__, flush=True)
logging.disable(logging.CRITICAL)

calls = []


def target(x):
    calls.append(np.array(x, dtype=float))
    return float(-0.5 * np.sum(np.asarray(x) ** 2))


class DistanceFromOrigin(AbstractAcqFcn):
    def _compute_acquisition_function(self, Xs, *args):
        return np.sum(np.atleast_2d(Xs) ** 2, axis=1)


D = 2
vbmc = VBMC(
    target,
    np.zeros((1, D)),
    np.full((1, D), -np.inf),
    np.full((1, D), np.inf),
    np.full((1, D), -3.0),
    np.full((1, D), 3.0),
    {
        "active_sample_gp_update": False,
        "active_sample_vp_update": False,
        "ns_search": 1,
        "cache_frac": 1,
        "search_optimizer": "none",
    },
    seed=7,
)
fl, optim_state, _, _ = active_sample(
    None,
    12,
    vbmc.optim_state,
    vbmc.function_logger,
    vbmc.iteration_history,
    vbmc.vp,
    vbmc.options,
)
dead = 3
fl.X_flag[dead] = False  # what the trim at the end of warm-up does to a row
optim_state["N"] = fl.Xn + 1
optim_state["n_eff"] = np.sum(fl.n_evals[fl.X_flag])
gp, _, _, hyp = train_gp(
    {},
    optim_state,
    fl,
    vbmc.iteration_history,
    vbmc.options,
    vbmc.plausible_lower_bounds,
    vbmc.plausible_upper_bounds,
    rng=vbmc.vp.rng,
)
optim_state["hyp_dict"] = hyp
vbmc.options.__setitem__("search_acq_fcn", [DistanceFromOrigin()], force=True)

# The one candidate of the search: the input of the row that is switched off,
# with no stored value, so that the target is called.
optim_state["cache"]["x_orig"] = np.copy(fl.X_orig[[dead]])
optim_state["cache"]["y_orig"] = np.array([np.nan])

before = dict(
    func_count=fl.func_count,
    Xn=fl.Xn,
    live=int(np.sum(fl.X_flag)),
    n_evals_dead=int(np.ravel(fl.n_evals[dead])[0]),
    gp_rows=gp.X.shape[0],
    y_max=float(fl.y_max),
    n_calls=len(calls),
)
fl, optim_state, _, gp = active_sample(
    gp, 1, optim_state, fl, vbmc.iteration_history, vbmc.vp, vbmc.options
)
after = dict(
    func_count=fl.func_count,
    Xn=fl.Xn,
    live=int(np.sum(fl.X_flag)),
    n_evals_dead=int(np.ravel(fl.n_evals[dead])[0]),
    gp_rows=gp.X.shape[0],
    y_max=float(fl.y_max),
    n_calls=len(calls),
)
for key in before:
    print(f"   {key:13s}: {before[key]} -> {after[key]}")
print("   the row is live after the step:", bool(fl.X_flag[dead]))
print("   the target was called at      :", calls[-1])
