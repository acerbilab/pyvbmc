"""P2i F10: the search-bound clip runs before the integer snap.

Settles: a candidate placed exactly on ``ub_search`` is moved past it by
``_real2int``; how far past; that the hard bounds still hold; and that the
expansion of the search bounds tests transformed-space distances.
"""
import importlib
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from _common import Cheap, banner, build, set_acq, with_gp  # noqa: E402

from pyvbmc.acquisition_functions import AbstractAcqFcn  # noqa: E402
from pyvbmc.vbmc.active_sample import active_sample  # noqa: E402

banner()
asmod = importlib.import_module("pyvbmc.vbmc.active_sample")

print("\n=== a candidate at ub_search snaps past it ===")
v = build(
    1,
    {"integer_vars": np.array([True])},
    lb=np.array([[-10.5]]),
    ub=np.array([[10.5]]),
)
pt = v.parameter_transformer
os_ = v.optim_state
ub_s = os_["ub_search"]
lb_s = os_["lb_search"]
print("  lb_search/ub_search (transformed):", lb_s, ub_s)
print("  in original coordinates:", pt.inverse(lb_s), pt.inverse(ub_s))
for edge, name in ((ub_s, "ub_search"), (lb_s, "lb_search")):
    cand = np.array([[float(edge[0, 0])]])
    snapped = AbstractAcqFcn._real2int(cand.copy(), pt, os_["integer_vars"])
    outside = (
        bool(snapped[0, 0] > ub_s[0, 0])
        if name == "ub_search"
        else bool(snapped[0, 0] < lb_s[0, 0])
    )
    print(
        f"  candidate at {name}: original {pt.inverse(cand)[0, 0]:.6f} -> "
        f"{pt.inverse(snapped)[0, 0]:.6f}; outside the search box: {outside}"
    )
    print(
        "    inside the hard bounds:",
        bool(
            pt.inverse(snapped)[0, 0] >= os_["lb_orig"][0, 0]
            and pt.inverse(snapped)[0, 0] <= os_["ub_orig"][0, 0]
        ),
    )

print(
    "\n=== the whole candidate set after _get_search_points then the snap ==="
)
v2 = build(
    2,
    {
        "integer_vars": np.array([True, False]),
        "ns_search": 512,
        "search_optimizer": "none",
    },
    lb=np.array([[-10.5, -np.inf]]),
    ub=np.array([[10.5, np.inf]]),
)
gp2, fl2, os2 = with_gp(v2, n=12)
X, _ = asmod._get_search_points(512, os2, fl2, v2.vp, v2.options)
inside_before = int(
    np.sum(np.all((X >= os2["lb_search"]) & (X <= os2["ub_search"]), axis=1))
)
Xs = AbstractAcqFcn._real2int(
    X.copy(), fl2.parameter_transformer, os2["integer_vars"]
)
inside_after = int(
    np.sum(np.all((Xs >= os2["lb_search"]) & (Xs <= os2["ub_search"]), axis=1))
)
print(
    "  candidates inside the search box before the snap:",
    inside_before,
    "/",
    X.shape[0],
)
print(
    "  candidates inside the search box after the snap :",
    inside_after,
    "/",
    Xs.shape[0],
)
worst = np.max(np.maximum(Xs - os2["ub_search"], os2["lb_search"] - Xs))
print(
    "  largest excursion past a search bound (transformed units):",
    float(worst),
)

print("\n=== the bound expansion tests transformed-space distances ===")
print(
    "  delta_search = 0.05*(ub_search - lb_search) =",
    0.05 * (os2["ub_search"] - os2["lb_search"]),
)
print("  one integer step near the centre, in transformed units:")
for x0 in (0.0, 5.0, 9.0):
    a = fl2.parameter_transformer(np.array([[x0, 0.0]]))[0, 0]
    b = fl2.parameter_transformer(np.array([[x0 + 1.0, 0.0]]))[0, 0]
    print(f"    {x0} -> {x0 + 1.0}: {b - a:.6f}")
