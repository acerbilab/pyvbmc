"""P2i F9: the two covariance normalizations and the literal 3 of the box.

Settles: (a) the numerical size of the difference between
``np.cov(X, rowvar=False)`` and ``np.cov(X, bias=True, rowvar=False)``;
(b) whether either branch is reachable at the shipped defaults, i.e. what
``hpd_search_frac`` is and whether ``lb_search``/``ub_search`` can be
non-finite; (c) what ``active_search_bound`` is and where it is used.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from _common import banner, build  # noqa: E402

banner()

print("\n=== (a) the two normalizations ===")
rng = np.random.default_rng(0)
X = rng.normal(size=(4, 2))
print("  np.cov(X, rowvar=False)            =\n", np.cov(X, rowvar=False))
print(
    "  np.cov(X, bias=True, rowvar=False) =\n",
    np.cov(X, bias=True, rowvar=False),
)
print(
    "  ratio of the [0,0] entries:",
    float(
        np.cov(X, rowvar=False)[0, 0]
        / np.cov(X, bias=True, rowvar=False)[0, 0]
    ),
    " (N/(N-1) =",
    4 / 3,
    ")",
)

print("\n=== (b) defaults and reachability ===")
for D, lb, ub in (
    (2, None, None),
    (2, np.array([[-10.0, -5.0]]), np.array([[10.0, 5.0]])),
    (2, np.array([[-10.0, -np.inf]]), np.array([[10.0, np.inf]])),
):
    v = build(D, {}, lb=lb, ub=ub)
    os_ = v.optim_state
    tag = (
        "unbounded"
        if lb is None
        else ("bounded" if np.all(np.isfinite(lb)) else "mixed")
    )
    print(f"  [{tag:9s}] lb_tran={os_['lb_tran']} ub_tran={os_['ub_tran']}")
    print(
        f"              lb_search={os_['lb_search']} "
        f"ub_search={os_['ub_search']}  finite="
        f"{bool(np.all(np.isfinite(os_['lb_search'])) and np.all(np.isfinite(os_['ub_search'])))}"
    )

v = build(2, {})
for name in (
    "hpd_search_frac",
    "hpd_frac",
    "box_search_frac",
    "heavy_tail_search_frac",
    "mvn_search_frac",
    "search_cache_frac",
    "cache_frac",
    "active_search_bound",
    "ns_search",
):
    print(f"  default {name:24s} = {v.options[name]}")

print("\n=== (c) the empty-HPD branch needs round(hpd_frac*N) == 0 ===")
from pyvbmc.stats import get_hpd  # noqa: E402

Xt = np.arange(12, dtype=float).reshape(6, 2)
yt = np.arange(6, dtype=float).reshape(6, 1)
for frac in (0.8 / 8, 0.01, 0.0):
    X_hpd, _, _, _ = get_hpd(Xt, yt, frac)
    print(f"  hpd_frac={frac!r:8s} -> X_hpd.shape = {X_hpd.shape}")
