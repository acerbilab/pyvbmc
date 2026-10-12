"""P4-6: the order of the two draws in the active-sampling step.

MATLAB prepares the importance samples (``activesample_vbmc.m:208-211``)
before it generates the search candidates (``:215-218``); PyVBMC
generates the candidates (``active_sample.py:379-381``) before the
importance samples (``:414-417``).  This script checks that nothing but
the random stream depends on the order: neither routine changes the GP,
the variational posterior or ``optim_state`` (beyond the key the call
site assigns), so swapping them changes only which draws each gets.
"""


import numpy as np
from wave4_P4_common import banner, load

banner()

from pyvbmc.acquisition_functions import AcqFcnIMIQR  # noqa: E402
from pyvbmc.vbmc.active_importance_sampling import (  # noqa: E402
    active_importance_sampling,
)
from pyvbmc.vbmc.active_sample import _get_search_points  # noqa: E402


def gp_fingerprint(gp):
    out = [gp.X.copy(), gp.y.copy()]
    if gp.s2 is not None:
        out.append(np.asarray(gp.s2).copy())
    for p in gp.posteriors:
        out += [p.hyp.copy(), p.alpha.copy(), p.L.copy(), p.sW.copy()]
    return out


def vp_fingerprint(vp):
    return [
        vp.w.copy(),
        vp.mu.copy(),
        vp.sigma.copy(),
        vp.lambd.copy(),
        np.array([vp.K]),
    ]


def same(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


st = load("rosenbrock_D2_noise1_viqr", seed=1)
vp, gp, opts, ostate, logger = (
    st["vp"],
    st["gp"],
    st["options"],
    st["optim_state"],
    st["logger"],
)
ostate.setdefault("cache", {"x_orig": np.zeros((0, gp.D))})
acq = AcqFcnIMIQR()

g0, v0 = gp_fingerprint(gp), vp_fingerprint(vp)
keys0 = set(ostate.keys())

vp.rng = np.random.default_rng(17)
out = active_importance_sampling(vp, gp, acq, opts)
print(
    "active_importance_sampling leaves the GP unchanged:",
    same(g0, gp_fingerprint(gp)),
)
print("... and the VP unchanged:", same(v0, vp_fingerprint(vp)))
print("... and adds no optim_state key:", set(ostate.keys()) == keys0)

vp.rng = np.random.default_rng(17)
Xs, idx = _get_search_points(opts["ns_search"], ostate, logger, vp, opts)
print(
    "_get_search_points leaves the GP unchanged:", same(g0, gp_fingerprint(gp))
)
print("... and the VP unchanged:", same(v0, vp_fingerprint(vp)))
print("... and adds no optim_state key:", set(ostate.keys()) == keys0)

# the order changes only which draws each gets
vp.rng = np.random.default_rng(23)
a1 = active_importance_sampling(vp, gp, acq, opts)
s1, _ = _get_search_points(opts["ns_search"], ostate, logger, vp, opts)
vp.rng = np.random.default_rng(23)
s2, _ = _get_search_points(opts["ns_search"], ostate, logger, vp, opts)
a2 = active_importance_sampling(vp, gp, acq, opts)
print()
print("search points identical across the two orders:", np.array_equal(s1, s2))
print(
    "importance samples identical across the two orders:",
    np.array_equal(a1["X"], a2["X"]),
)
print("(so only the stream differs, as expected)")
