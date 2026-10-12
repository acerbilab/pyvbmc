"""P2i F3: the search cache is written before the 'removal' of the acquired point.

Settles, with ``search_cache_frac > 0``:
 (a) ``optim_state["search_cache"][0]`` equals the point just acquired;
 (b) the two ``np.delete`` calls at :467-468 rebind locals that are never
     read again (checked by re-running the step with the deletes disabled:
     bitwise-identical outcome);
 (c) the cached rows contain the training inputs appended for repeated
     observations when that mechanism is on;
 (d) a row that came from the starting cache carries ``nan`` as its cache
     index once it is re-offered through the search cache.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
import importlib  # noqa: E402

from _common import (  # noqa: E402
    Cheap,
    banner,
    build,
    noisy_target,
    set_acq,
    with_gp,
)

from pyvbmc.vbmc.active_sample import active_sample  # noqa: E402

# The package exports the function under the module's name, so the module
# object has to be fetched explicitly.
asmod = importlib.import_module("pyvbmc.vbmc.active_sample")

banner()

print("\n=== (a) search_cache[0] after one step ===")
v = build(
    2,
    {
        "ns_search": 32,
        "search_cache_frac": 0.5,
        "search_optimizer": "none",
        "cache_frac": 0,
    },
)
gp, fl, os_ = with_gp(v)
set_acq(v, Cheap())
os_["cache"]["x_orig"] = np.empty((0, 2))
os_["cache"]["y_orig"] = np.empty(0)
fl, os_, _, gp = active_sample(
    gp, 1, os_, fl, v.iteration_history, v.vp, v.options
)
acquired = fl.X[fl.Xn].copy()
cache = np.asarray(os_["search_cache"])
print("  search_cache shape:", cache.shape, " ns_search =", 32)
print(
    "  search_cache[0] == acquired point:",
    bool(np.array_equal(cache[0], acquired)),
)
print(
    "  acquired point appears in search_cache at row(s):",
    np.flatnonzero((cache == acquired).all(axis=1)).tolist(),
)

print("\n=== (b) is the np.delete pair dead? ===")
# Re-run the identical step with the module's np.delete replaced by a
# version that records its calls and returns the input unchanged for the
# two search-set deletions; everything else is untouched.
recorded = []
real_delete = asmod.np.delete


def spy_delete(arr, obj, axis=None):
    recorded.append((np.shape(arr), np.shape(np.atleast_1d(obj))))
    return real_delete(arr, obj, axis)


v2 = build(
    2,
    {
        "ns_search": 32,
        "search_cache_frac": 0.5,
        "search_optimizer": "none",
        "cache_frac": 0,
    },
)
gp2, fl2, os2 = with_gp(v2)
set_acq(v2, Cheap())
os2["cache"]["x_orig"] = np.empty((0, 2))
os2["cache"]["y_orig"] = np.empty(0)
asmod.np.delete = spy_delete
try:
    fl2, os2, _, gp2 = active_sample(
        gp2, 1, os2, fl2, v2.iteration_history, v2.vp, v2.options
    )
finally:
    asmod.np.delete = real_delete
print("  np.delete calls inside the step (arr shape, obj shape):", recorded)
print(
    "  acquired point identical to run (a):",
    bool(np.array_equal(fl2.X[fl2.Xn], acquired)),
)
print(
    "  search_cache identical to run (a):",
    bool(np.array_equal(np.asarray(os2["search_cache"]), cache)),
)

print(
    "\n=== (c) training inputs inside the search cache (repeat mechanism) ==="
)
v3 = build(
    2,
    {
        "ns_search": 32,
        "search_cache_frac": 0.5,
        "search_optimizer": "none",
        "cache_frac": 0,
        "specify_target_noise": True,
        "max_repeated_observations": 3,
    },
    target=noisy_target,
)
gp3, fl3, os3 = with_gp(v3)
set_acq(v3, Cheap())
os3["cache"]["x_orig"] = np.empty((0, 2))
os3["cache"]["y_orig"] = np.empty(0)
X_train = fl3.X[fl3.X_flag].copy()
fl3, os3, _, gp3 = active_sample(
    gp3, 1, os3, fl3, v3.iteration_history, v3.vp, v3.options
)
cache3 = np.asarray(os3["search_cache"])
hits = sum(int(np.any((cache3 == row).all(axis=1))) for row in X_train)
print(
    "  training rows:",
    X_train.shape[0],
    " of which present in the " "search cache:",
    hits,
    " (search cache shape",
    cache3.shape,
    ")",
)

print(
    "\n=== (d) a starting-cache row that comes back through the search cache ==="
)
v4 = build(
    2,
    {
        "ns_search": 8,
        "cache_frac": 0.5,
        "search_cache_frac": 0.5,
        "heavy_tail_search_frac": 0.0,
        "mvn_search_frac": 0.0,
        "hpd_search_frac": 0.0,
        "box_search_frac": 0.0,
        "search_optimizer": "none",
    },
)
gp4, fl4, os4 = with_gp(v4)
set_acq(v4, Cheap())
# A starting cache with stored values, all far from the origin except one.
os4["cache"]["x_orig"] = np.array(
    [[0.10, 0.10], [2.0, 2.0], [2.5, 2.5], [3.0, 3.0]]
)
os4["cache"]["y_orig"] = np.array([-11.0, -22.0, -33.0, -44.0])
_, idx_cache = asmod._get_search_points(8, os4, fl4, v4.vp, v4.options)
print("  idx_cache straight from _get_search_points:", idx_cache)
fl4, os4, _, gp4 = active_sample(
    gp4, 1, os4, fl4, v4.iteration_history, v4.vp, v4.options
)
print("  cache rows left:", os4["cache"]["x_orig"].shape[0])
cache4 = np.asarray(os4["search_cache"])
x_cached_tran = fl4.parameter_transformer(os4["cache"]["x_orig"])
in_sc = [int(np.any((cache4 == row).all(axis=1))) for row in x_cached_tran]
print("  starting-cache rows still present in the search cache:", in_sc)
# Next call with the direct cache route switched off, so that a starting
# cache row can only arrive through the search cache: its index is nan.
v4.options.__setitem__("cache_frac", 0.0, force=True)
v4.options.__setitem__("search_cache_frac", 1.0, force=True)
X2, idx_cache2 = asmod._get_search_points(8, os4, fl4, v4.vp, v4.options)
print("  idx_cache on the next call (cache_frac=0):", idx_cache2)
for j, row in enumerate(x_cached_tran):
    where = np.flatnonzero((X2 == row).all(axis=1))
    print(
        f"    cached row {j} appears at search-set rows {where.tolist()} "
        f"with idx_cache {idx_cache2[where].tolist()}"
    )
