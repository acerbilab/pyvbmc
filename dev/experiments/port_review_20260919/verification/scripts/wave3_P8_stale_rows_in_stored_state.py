"""Settles the reachability half of P8c F4 / P8i F5 from stored data.

``pyvbmc/testing/oracles/fixtures/corr_D5_warped.npz`` is a state captured
from a default-options run that reached one warp (``warping_count = 1``,
``last_warping = 15``) and holds 5 inactive rows out of 105.  Inactive rows
are produced only by the warm-up trim (``vbmc.py:2091-2106``), which runs
while the run is still in warm-up, and a warp needs ``not warmup``, so the
trim preceded the warp.  If ``warp_input`` had rewritten every filled row,
as ``misc/warp_input_vbmc.m:112-119`` does, the stored ``X`` of the inactive
rows would equal the transform of their ``X_orig``.  This script checks it.
"""

import json

import numpy as np

import pyvbmc
from pyvbmc.parameter_transformer import ParameterTransformer

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

root = (
    r"C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\testing\oracles\fixtures"
)
meta = json.load(open(root + r"\corr_D5_warped.json"))
z = np.load(root + r"\corr_D5_warped.npz", allow_pickle=False)

print(
    "warping_count =",
    meta["optim_state"]["warping_count"],
    " last_warping =",
    meta["optim_state"]["last_warping"],
    " warmup =",
    meta["optim_state"]["warmup"],
    " Xn =",
    meta["logger"]["Xn"],
)
pt_meta = meta["pt"]
print("transformer keys:", sorted(pt_meta))

D = int(pt_meta["D"]) if "D" in pt_meta else z["logger/X"].shape[1]
pt = ParameterTransformer(
    D,
    z["pt/lb_orig"],
    z["pt/ub_orig"],
    z["pt/plb_orig"] if "pt/plb_orig" in z.files else None,
    z["pt/pub_orig"] if "pt/pub_orig" in z.files else None,
    transform_type=pt_meta.get("transform_type", "probit"),
)
# install the stored warp exactly as the fixture holds it
for name in ("scale", "R_mat", "mu", "delta", "type"):
    key = "pt/" + name
    if key in z.files:
        setattr(pt, name, z[key])
print("scale =", getattr(pt, "scale", None))
print("mu =", pt.mu, " delta =", pt.delta)

flag = z["logger/X_flag"].astype(bool)
Xn = int(meta["logger"]["Xn"])
X = z["logger/X"]
X_orig = z["logger/X_orig"]
recomputed = pt(X_orig[: Xn + 1])
err = np.max(np.abs(recomputed - X[: Xn + 1]), axis=1)
act = flag[: Xn + 1]
print()
print(
    "rows filled:",
    Xn + 1,
    " active:",
    int(act.sum()),
    " inactive:",
    int((~act).sum()),
)
print(
    "max |stored X - transform(X_orig)| over ACTIVE rows  :", np.max(err[act])
)
print(
    "max |stored X - transform(X_orig)| over INACTIVE rows:", np.max(err[~act])
)
print()
for i in np.flatnonzero(~act):
    print(f"  row {i}: stored X = {X[i]}")
    print(f"          transform(X_orig) = {recomputed[i]}")
print()
print("the same for y:")
y = z["logger/y"]
y_orig = z["logger/y_orig"]
dy = pt.log_abs_det_jacobian(recomputed)
y_expected = y_orig[: Xn + 1].ravel() + dy
print(
    "  max |stored y - (y_orig + dy)| active  :",
    np.max(np.abs(y[: Xn + 1].ravel()[act] - y_expected[act])),
)
print(
    "  max |stored y - (y_orig + dy)| inactive:",
    np.max(np.abs(y[: Xn + 1].ravel()[~act] - y_expected[~act])),
)
