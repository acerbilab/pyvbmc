"""O1 first question, variance branch at D = 2: J_sjk against Gauss-Hermite
quadrature (n x n nodes per component, all components' nodes passed to one
predict_full call), to confirm that the 1e-4 of the uniform-grid check in
O1_Q1_variance.py is discretization error of that grid."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from O1_Q1_variance import make_vp  # noqa: reuses the VP builder
from wave7_O1_common import banner, build_gp

from pyvbmc.vbmc.variational_optimization import _gp_log_joint

banner()
rng = np.random.default_rng(3)
for mean in ("zero", "const", "negquad"):
    gp = build_gp(2, N=15, Ns=2, seed=42, mean=mean, ln_sn=-2.0)
    vp = make_vp(2, 3, rng)
    J = _gp_log_joint(vp, gp, False, False, True, True, True)[6]
    s = vp.sigma.ravel()[None, :] * vp.lambd.ravel()[:, None]
    for n in (16, 24):
        t, wt = np.polynomial.hermite_e.hermegauss(n)
        wt = wt / wt.sum()
        T1, T2 = np.meshgrid(t, t, indexing="ij")
        W = np.outer(wt, wt).ravel()
        Xs = [
            np.column_stack(
                [
                    vp.mu[0, k] + s[0, k] * T1.ravel(),
                    vp.mu[1, k] + s[1, k] * T2.ravel(),
                ]
            )
            for k in range(3)
        ]
        _, C = gp.predict_full(np.vstack(Xs), add_noise=False)
        M = n * n
        worst = 0.0
        for sidx in range(2):
            Jr = np.array(
                [
                    [
                        W
                        @ C[j * M : (j + 1) * M, k * M : (k + 1) * M, sidx]
                        @ W
                        for k in range(3)
                    ]
                    for j in range(3)
                ]
            )
            worst = max(
                worst, np.max(np.abs(J[sidx] - Jr)) / np.max(np.abs(Jr))
            )
        print(
            f"{mean:7s} D=2 K=3 n={n}: max rel |J_sjk - GH| = {worst:.1e}",
            flush=True,
        )
