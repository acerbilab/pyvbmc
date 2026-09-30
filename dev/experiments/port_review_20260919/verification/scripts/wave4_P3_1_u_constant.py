"""P3-1: the `u` constant of VIQR/IMIQR.

Settles: (a) the numerical gap between `norm.ppf(0.75)` (the port) and the
literal 0.6745 (MATLAB `acq/acqviqr_vbmc.m:4`, `acq/acqimiqr_vbmc.m:4`);
(b) how far the two move the acquisition on a stored noisy state, and
whether the sieve minimizer moves. Everything else in the two formulas is
textually the same on both sides, so substituting `u` is exactly the
MATLAB-vs-port difference of these two acquisitions.
"""
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from wave4_P3_common import banner, load  # noqa: E402

banner()

u_py = norm.ppf(0.75)
u_ml = 0.6745
print(f"norm.ppf(0.75) = {u_py!r}")
print(f"literal        = {u_ml!r}")
print(f"abs diff       = {u_py - u_ml:.6e}   rel = {(u_py-u_ml)/u_ml:.6e}")
print(f"norm.cdf(0.6745) = {norm.cdf(u_ml):.10f}")

from pyvbmc.acquisition_functions import AcqFcnIMIQR, AcqFcnVIQR  # noqa: E402
from pyvbmc.testing.oracles._oracles import (  # noqa: E402
    prepare_gp_for_acq,
    prepare_importance_sampling,
)

SEED = 20260904
for cls in (AcqFcnVIQR, AcqFcnIMIQR):
    st = load("rosenbrock_D2_noise1_viqr", seed=SEED)
    gp, vp, logger, os_ = (
        st["gp"],
        st["vp"],
        st["logger"],
        st["optim_state"],
    )
    Xs = np.array(st["cand"]["Xs"], dtype=float)
    prepare_gp_for_acq(gp, logger, os_)
    acq = cls()
    prepare_importance_sampling(st, acq, SEED)
    a_py = acq(Xs.copy(), gp, vp, logger, os_)
    acq.u = u_ml  # the MATLAB literal; nothing else differs
    a_ml = acq(Xs.copy(), gp, vp, logger, os_)
    fin = np.isfinite(a_py) & np.isfinite(a_ml)
    d = np.abs(a_py - a_ml)[fin]
    rel = d / np.maximum(np.abs(a_ml[fin]), 1e-300)
    print(
        f"{cls.__name__:12s} N={Xs.shape[0]} Ns_gp={len(gp.posteriors)} "
        f"maxabs={d.max():.3e} maxrel={rel.max():.3e} "
        f"argmin py={np.argmin(a_py)} literal={np.argmin(a_ml)} "
        f"same_nonfinite={np.array_equal(np.isfinite(a_py), np.isfinite(a_ml))}"
    )
