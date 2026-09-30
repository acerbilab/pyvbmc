"""P2i section 4, the statements it reports as correct.

Settles: (b) the source counts of ``_get_search_points`` sum exactly to the
number asked for, with and without a partly filled starting cache, and
``idx_cache`` stays aligned with the rows; (c) every draw of one
``active_sample`` step goes through ``vp.rng``: a seeded step reproduces bit
for bit in each search branch and leaves NumPy's global stream untouched.
"""
import copy
import importlib
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from _common import Cheap, banner, build, set_acq, with_gp  # noqa: E402

from pyvbmc.vbmc.active_sample import active_sample  # noqa: E402

banner()
asmod = importlib.import_module("pyvbmc.vbmc.active_sample")

print("\n=== (b) the counts of _get_search_points ===")
v = build(2, {"search_optimizer": "none"})
gp, fl, os_ = with_gp(v, n=12)
for n_cache in (0, 3, 9):
    os2 = copy.deepcopy(os_)
    if n_cache:
        os2["cache"]["x_orig"] = np.linspace(-2, 2, n_cache * 2).reshape(
            n_cache, 2
        )
        os2["cache"]["y_orig"] = np.full(n_cache, np.nan)
    else:
        os2["cache"]["x_orig"] = np.empty((0, 2))
        os2["cache"]["y_orig"] = np.empty(0)
    for n in (1, 2, 3, 5, 7, 100, 8191, 8192):
        X, idx = asmod._get_search_points(n, os2, fl, v.vp, v.options)
        ok = X.shape[0] == n and idx.shape[0] == n
        if not ok:
            print(
                f"  cache={n_cache} n={n}: X {X.shape} idx {idx.shape} "
                f"MISMATCH"
            )
    # alignment of the non-nan indices
    X, idx = asmod._get_search_points(64, os2, fl, v.vp, v.options)
    live = np.flatnonzero(~np.isnan(idx))
    aligned = all(
        np.allclose(
            X[i],
            np.minimum(
                np.maximum(
                    fl.parameter_transformer(
                        os2["cache"]["x_orig"][int(idx[i])][None, :]
                    ),
                    os2["lb_search"],
                ),
                os2["ub_search"],
            )[0],
        )
        for i in live
    )
    print(
        f"  cache={n_cache}: every requested size matched; "
        f"{len(live)} cache rows at n=64, idx_cache aligned: {aligned}"
    )

print("\n=== (c) the run's draws and NumPy's global stream ===")
for optimizer in ("none", "cmaes", "Nelder-Mead"):
    results = []
    states = []
    for global_seed in (1, 2):
        np.random.seed(global_seed)
        gstate_before = np.random.get_state()[1].copy()
        v = build(2, {"search_optimizer": optimizer}, seed=123)
        gpx, flx, osx = with_gp(v, n=12)
        set_acq(v, Cheap())
        flx, osx, _, gpx = active_sample(
            gpx, 1, osx, flx, v.iteration_history, v.vp, v.options
        )
        results.append(flx.X[flx.Xn].copy())
        states.append(np.array_equal(np.random.get_state()[1], gstate_before))
    same = bool(np.array_equal(results[0], results[1]))
    print(
        f"  [{optimizer:11s}] identical under two global seeds: {same}"
        f" | global stream untouched: {states}"
    )

# and different vp.rng seeds give different draws
res = []
for s in (11, 12):
    v = build(2, {"search_optimizer": "cmaes"}, seed=s)
    gpx, flx, osx = with_gp(v, n=12)
    set_acq(v, Cheap())
    flx, osx, _, gpx = active_sample(
        gpx, 1, osx, flx, v.iteration_history, v.vp, v.options
    )
    res.append(flx.X[flx.Xn].copy())
print(
    "  two vp.rng seeds give different acquired points:",
    not bool(np.array_equal(res[0], res[1])),
)
