"""Settles how a default run reaches the nudge of P8c F1.

A transformed coordinate far enough out saturates the inverse, which clamps
the original-space value onto ``nextafter(bound)``.  That value is what the
function logger stores in ``X_orig``, and ``warp_input`` re-transforms every
stored ``X_orig`` at each warp (``whitening.py:216``).  This script follows
that composition for the default bounded transform and reports where the
nudge changes the result from MATLAB's +/-Inf.
"""

import numpy as np
from scipy.special import erfcinv

import pyvbmc
from pyvbmc.parameter_transformer import ParameterTransformer

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

for lb, ub in [
    (-5.0, 5.0),
    (0.0, 1.0),
    (1.0, 3.0),
    (-10.0, 10.0),
    (0.0, 3.0),
    (-1.0, 1.0),
]:
    pt = ParameterTransformer(
        1, np.array([[lb]]), np.array([[ub]]), transform_type="probit"
    )
    u_far = np.array([[60.0]])  # a saturating coordinate
    x_orig = pt.inverse(u_far)  # clamped onto the bound
    z = (x_orig[0, 0] - lb) / (ub - lb)
    with np.errstate(all="ignore"):
        py = pt(x_orig)[0, 0]
        matlab = -np.sqrt(2) * erfcinv(2 * z)
    print(
        f"  lb={lb:6}, ub={ub:6}: inverse(60) = {x_orig[0, 0]!r}"
        f"  (== ub: {x_orig[0, 0] == ub})"
    )
    print(f"      raw z = {z!r} -> PyVBMC {py!r}, MATLAB {matlab!r}")
print()
print(
    "The nudge changes the answer exactly where the clamped value is a"
    " distinct float from the bound but (x-lb)/(ub-lb) still rounds to"
    " 0 or 1, i.e. where the bound is far enough from zero relative to"
    " the box width."
)
