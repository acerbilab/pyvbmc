"""P3-2: the rounding convention of `_real2int` and whether a tie is reachable.

Settles: (a) that `np.around` (half to even) and MATLAB's `round`
(half away from zero, `misc/real2int_vbmc.m:7`) disagree on exact ties;
(b) whether an inverse-transformed coordinate of an integer variable can
equal a half-integer exactly, given that PyVBMC forces the hard bounds of
an integer variable to be finite and to sit at half-integers
(`vbmc.py:912-923`). The box midpoint is a half-integer whenever the number
of integer levels is even, and the probit / logit transforms send their
origin to it exactly.
"""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from wave4_P3_common import banner  # noqa: E402

banner()

from pyvbmc.acquisition_functions.abstract_acq_fcn import (  # noqa: E402
    AbstractAcqFcn,
)
from pyvbmc.parameter_transformer import ParameterTransformer  # noqa: E402


def matlab_round(v):
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


vals = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
print("value      ", vals)
print("np.around  ", np.around(vals))
print("MATLAB     ", matlab_round(vals))

# --- can the inverse transform land exactly on a half-integer? ----------
for ttype in ("probit", "logit"):
    for lb, ub in ((-0.5, 9.5), (-0.5, 10.5)):
        D = 1
        pt = ParameterTransformer(
            D,
            np.array([[lb]]),
            np.array([[ub]]),
            np.array([[lb + 1.0]]),
            np.array([[ub - 1.0]]),
            transform_type=ttype,
        )
        mid = 0.5 * (lb + ub)
        # The transformed value of the box midpoint, and its round trip.
        z_mid = pt(np.array([[mid]]))
        back = pt.inverse(z_mid)
        # The transformed origin of the transform (before mu/delta scaling
        # this is the point mapping to the middle of the box).
        z0 = np.array([[float(pt.mu[0])]])
        back0 = pt.inverse(z0)
        print(
            f"{ttype:7s} box [{lb},{ub}] mid={mid} "
            f"pt(mid)={z_mid.ravel()[0]!r} inverse->{back.ravel()[0]!r} "
            f"exact={back.ravel()[0] == mid}  "
            f"inverse(mu)={back0.ravel()[0]!r} "
            f"is_half={float(back0.ravel()[0]) % 1 == 0.5}"
        )
        if float(back0.ravel()[0]) % 1 == 0.5:
            iv = np.array([True])
            X = np.array(z0, dtype=float)
            snapped = AbstractAcqFcn._real2int(X.copy(), pt, iv)
            print(
                "         _real2int(transformed origin) -> orig ",
                pt.inverse(snapped).ravel()[0],
                " (MATLAB would give ",
                matlab_round(np.array([float(back0.ravel()[0])]))[0],
                ")",
            )

# --- how often does a draw hit a tie? ----------------------------------
# `_get_search_points` clips every candidate to the finite transformed
# search box (`active_sample.py:1072`), so the hard bounds themselves --
# which ARE half-integers -- are not reachable from the sieve.
rng = np.random.default_rng(0)
pt = ParameterTransformer(
    1,
    np.array([[-0.5]]),
    np.array([[9.5]]),
    np.array([[0.5]]),
    np.array([[8.5]]),
    transform_type="probit",
)
plb_t = pt(np.array([[0.5]]))
pub_t = pt(np.array([[8.5]]))
prange = pub_t - plb_t
lb_s = plb_t - 2 * prange
ub_s = pub_t + 2 * prange
z = rng.normal(size=(2_000_000, 1)) * 3.0
z_clip = np.minimum(np.maximum(z, lb_s), ub_s)
xo = pt.inverse(z_clip)
print(
    "search box in original coords:",
    pt.inverse(lb_s).ravel()[0],
    pt.inverse(ub_s).ravel()[0],
)
print(
    "clipped draws: exact half-integers in 2e6:",
    int(np.sum(xo % 1 == 0.5)),
)
xo_raw = pt.inverse(z)
print(
    "UNclipped draws (saturating at the hard bounds, which are "
    "half-integers): exact half-integers in 2e6:",
    int(np.sum(xo_raw % 1 == 0.5)),
    " -> np.around:",
    np.unique(np.around(xo_raw[xo_raw % 1 == 0.5])),
    " MATLAB:",
    np.unique(matlab_round(xo_raw[xo_raw % 1 == 0.5])),
)
