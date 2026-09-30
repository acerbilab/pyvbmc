"""O4, first question, the one noise feature PyVBMC does not build: the
rectified output-dependent term ([1,0,1] and [1,2,1]), through the log
likelihood, by finite differences, with points on both sides of the
threshold."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from wave7_O4_common import attach, banner, fd_grad, make_gp

banner()
rng = np.random.default_rng(12)
for noise in ((1, 0, 1), (1, 2, 1)):
    for D in (1, 2):
        N = 16
        X = rng.uniform(-1, 1, (N, D))
        y = (-2 * np.sum(X**2, 1) + 0.1 * rng.standard_normal(N)).reshape(
            -1, 1
        )
        s2 = (
            (0.05 + 0.1 * rng.uniform(size=N)).reshape(-1, 1)
            if noise[1]
            else None
        )
        gp = make_gp(D, "negquad", noise)
        attach(gp, X, y, s2)
        nb = gp.noise.get_bounds_info(gp.X, gp.y)["x0"]
        h = np.concatenate(
            [
                gp.covariance.get_bounds_info(gp.X, gp.y)["x0"],
                nb,
                gp.mean.get_bounds_info(gp.X, gp.y)["x0"],
            ]
        )
        cov_N = D + 1
        k = cov_N + 1 + (1 if noise[1] == 2 else 0)
        h[k] = np.median(
            y
        )  # threshold inside the range of y: both sides active
        h[k + 1] = np.log(0.5)
        h[cov_N] = np.log(0.05)
        _, g = gp.log_likelihood(h, compute_grad=True)
        fd = fd_grad(lambda hh: gp.log_likelihood(hh), h, 1e-5)
        scale = np.maximum(np.abs(fd), 1e-3 * np.max(np.abs(fd)))
        print(
            f"noise={noise} D={D}: per-comp err={np.max(np.abs(g - fd) / scale):.1e}; "
            f"points above threshold: {int(np.sum(y[:, 0] < h[k]))} below-threshold y rows "
            f"get the term"
        )
