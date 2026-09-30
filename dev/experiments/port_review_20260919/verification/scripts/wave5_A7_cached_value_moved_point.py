"""W5, the stored value of a cached starting point and the point it is
recorded at.

`_get_search_points` transforms the cached starting points and clips every
candidate into the search box (`active_sample.py:939`, `:1072`), and
`active_sample` snaps integer-valued coordinates (`:383`). A cached row that
wins the sieve keeps its cache index, so its stored value is recorded at the
candidate as clipped and snapped (`:679-698`, `function_logger.add(xnew,
y_orig)`). `private/activesample_vbmc.m` does the same (`:555`, `:637`,
`:219`, `:388`). This script settles:

1. the mechanism, on a state built so that the cached row is the one
   candidate: a cached point outside the search box, and a cached point off
   the grid of an integer variable;
2. the reach of the clip. At construction the starting points lie in the
   plausible box, which is widened to hold them, and the search box is the
   plausible box with `active_search_bound` widths added on each side. A warp
   replaces the search box by the bounding box of the image of 1000 uniform
   draws from it (`whitening.py:299-310`) and the cache is transformed afresh
   by the new transformer. The check puts the corners of the plausible box in
   the cache of stored oracle states, applies `warp_input`, and reports
   whether any cached corner falls outside the new search box.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import copy
import itertools
import logging
import sys
from pathlib import Path

import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.acquisition_functions import AbstractAcqFcn
from pyvbmc.vbmc import active_sample
from pyvbmc.vbmc.gaussian_process_train import train_gp
from pyvbmc.whitening.whitening import warp_input

print("pyvbmc:", pyvbmc.__file__, flush=True)
logging.disable(logging.CRITICAL)
ROOT = Path(pyvbmc.__file__).resolve().parents[1]


def target(x):
    return float(-0.5 * np.sum(np.asarray(x) ** 2))


class DistanceFromOrigin(AbstractAcqFcn):
    """A cheap acquisition, so that the one candidate is taken as it is."""

    def _compute_acquisition_function(self, Xs, *args):
        return np.sum(np.atleast_2d(Xs) ** 2, axis=1)


def state_with_gp(options, lb, ub):
    D = 2
    opts = {
        "active_sample_gp_update": False,
        "active_sample_vp_update": False,
        "ns_search": 1,
        "cache_frac": 1,
        "search_optimizer": "none",
    }
    opts.update(options)
    vbmc = VBMC(
        target,
        np.zeros((1, D)),
        lb,
        ub,
        np.full((1, D), -3.0),
        np.full((1, D), 3.0),
        opts,
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
    vbmc.options.__setitem__(
        "search_acq_fcn", [DistanceFromOrigin()], force=True
    )
    return vbmc, gp, fl, optim_state


def one_step(label, options, lb, ub, x_cached, y_cached):
    vbmc, gp, fl, optim_state = state_with_gp(options, lb, ub)
    optim_state["cache"]["x_orig"] = np.copy(x_cached)
    optim_state["cache"]["y_orig"] = np.array([y_cached])
    calls_before = fl.func_count
    fl, optim_state, _, _ = active_sample(
        gp, 1, optim_state, fl, vbmc.iteration_history, vbmc.vp, vbmc.options
    )
    print(f"   {label}")
    print("      cached point and value :", x_cached[0], y_cached)
    print("      true value at the recorded point:", target(fl.X_orig[fl.Xn]))
    print(
        "      recorded point and value:",
        fl.X_orig[fl.Xn],
        fl.y_orig[fl.Xn, 0],
    )
    print("      target calls made by the step:", fl.func_count - calls_before)


inf = np.inf
print("\n--- 1. the mechanism", flush=True)
one_step(
    "a cached point outside the search box",
    {},
    np.full((1, 2), -inf),
    np.full((1, 2), inf),
    np.array([[50.0, 0.0]]),
    target([50.0, 0.0]),
)
one_step(
    "a cached point off the grid of an integer variable",
    {"integer_vars": np.array([True, False])},
    np.array([[-10.5, -inf]]),
    np.array([[10.5, inf]]),
    np.array([[2.4, 0.1]]),
    target([2.4, 0.1]),
)

print(
    "\n--- 2. the reach of the clip: corners of the plausible box through a"
    " warp",
    flush=True,
)
sys.path.insert(0, str(ROOT))
from pyvbmc.testing.oracles._state import (  # noqa: E402
    build_state,
    load_snapshot,
    snapshot_names,
)

FIXTURES = ROOT / "pyvbmc" / "testing" / "oracles" / "fixtures"
for name in snapshot_names(FIXTURES):
    snap = load_snapshot(FIXTURES / name)
    state = build_state(snap)
    vp, optim_state, fl = state["vp"], state["optim_state"], state["logger"]
    options = state["options"]
    D = vp.D
    pt_old = fl.parameter_transformer
    plb = np.ravel(optim_state["plb_tran"])
    pub = np.ravel(optim_state["pub_tran"])
    corners_tran = np.array(list(itertools.product(*zip(plb, pub))))
    corners_orig = pt_old.inverse(corners_tran)
    inside_before = np.all(
        (corners_tran >= optim_state["lb_search"])
        & (corners_tran <= optim_state["ub_search"])
    )
    try:
        pt_new, os_new, _, _ = warp_input(
            copy.deepcopy(vp),
            copy.deepcopy(optim_state),
            copy.deepcopy(fl),
            options,
        )
    except Exception as exc:  # the record is the exception itself
        print(f"   {name}: warp_input raised {type(exc).__name__}: {exc}")
        continue
    moved = pt_new(corners_orig)
    lo, hi = os_new["lb_search"], os_new["ub_search"]
    outside = int(np.sum(np.any((moved < lo) | (moved > hi), axis=1)))
    margin = np.min(np.minimum(moved - lo, hi - moved) / (hi - lo))
    print(
        f"   {name} (D = {D}): corners inside the search box before the warp: "
        f"{bool(inside_before)}; after it, {outside} of {len(moved)} outside, "
        f"smallest margin {margin:.3f} of the box width"
    )
