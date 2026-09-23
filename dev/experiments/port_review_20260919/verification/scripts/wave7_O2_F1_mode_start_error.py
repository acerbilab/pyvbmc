"""F1: how far the unrefined start that mode() returns lies from the mode.

Random overlapping mixtures (K = 5) at D = 2, 4, 6, 10, built at scale 1
and at scale 1e-3 with the same shape (identity transformer, so the two
values of orig_flag optimize the same density). At scale 1 mode() refines;
at 1e-3 it returns its start. The distance is in units of the smallest
component SD sigma_k * lambda_d, and the density gap in nats.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
from wave7_O2_common import banner, build_vp, ref_logq_and_grad

banner()
warnings.simplefilter("ignore")


def true_mode(vp, u0):
    mu, s, lam, w = vp.mu, vp.sigma, vp.lambd.ravel(), vp.w
    c = np.min(s) * lam

    def f(y):
        lq, g = ref_logq_and_grad((y * c)[None, :], mu, s, lam, w)
        return -lq[0], -(g[0] * c)

    r = minimize(f, u0 / c, jac=True, method="BFGS", options={"gtol": 1e-10})
    return r.x * c, -r.fun


for D in (2, 4, 6, 10):
    rs = np.random.default_rng(100 + D)
    K = 5
    mu0 = rs.normal(0, 0.7, (D, K))
    sig = rs.uniform(0.7, 1.3, K)
    lam0 = rs.uniform(0.6, 1.4, D)
    w = rs.dirichlet(np.full(K, 3.0))
    for flag in (False, True):
        out = []
        for scale in (1.0, 1e-3):
            vp = build_vp(mu0 * scale, sig, lam0 * scale, w)
            m = vp.mode(orig_flag=flag)
            ut, lqt = true_mode(vp, m)
            c = np.min(vp.sigma) * vp.lambd.ravel()
            lq = ref_logq_and_grad(
                m[None, :], vp.mu, vp.sigma, vp.lambd, vp.w
            )[0][0]
            out.append(
                f"scale {scale:g}: |mode-true|/sd = "
                f"{np.max(np.abs(m - ut) / c):.2e}, "
                f"gap {lqt - lq:.1e} nats"
            )
        print(f"D={D:2d} orig_flag={flag!s:5s}  " + "; ".join(out), flush=True)
