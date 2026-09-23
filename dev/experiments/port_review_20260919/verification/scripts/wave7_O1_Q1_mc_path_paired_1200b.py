"""S = 1200 rerun on seeds 5000-6199 of O1_Q1_mc_path_paired.py. O1 first question, the Adam path, paired: per seed, the analytic dF of
_neg_elcbo with the Monte Carlo entropy minus the pathwise derivative of the
same seed's estimate (central differences, common random numbers). The
difference is the zero-mean term the reparameterization gradient drops;
z = its mean / its standard error over S seeds. Same state as
O1_Q1_mc_path.py."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wave7_O1_common import OPTS, banner, build_gp

from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import _neg_elcbo

banner()
D, K, Ns_ent, S = 2, 3, 60, 1200
gp = build_gp(D, N=20, Ns=2, seed=5)
bnd = VariationalPosterior(D, K).get_bounds(gp.X, OPTS, K)
rng = np.random.default_rng(1)
theta = np.concatenate(
    [
        rng.uniform(-1, 1, D * K),
        np.log([0.5, 0.7, 0.4]),
        np.log([1.1, 0.8]),
        [-3.0, 0.2, 0.0],
    ]
)
theta[0] = bnd["lb"][0] - 0.2
theta[D * K] = 1.9
t0 = time.time()
h = 1e-4
diffs = np.zeros((S, theta.size))
for s in range(S):

    def F(th):
        return _neg_elcbo(
            th,
            gp,
            VariationalPosterior(D, K, rng=s + 5000),
            0.0,
            Ns_ent,
            False,
            False,
            bnd,
        )[0]

    g = _neg_elcbo(
        theta,
        gp,
        VariationalPosterior(D, K, rng=s + 5000),
        0.0,
        Ns_ent,
        True,
        False,
        bnd,
    )[1]
    fd = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = h
        fd[i] = (
            -F(theta + 2 * e)
            + 8 * F(theta + e)
            - 8 * F(theta - e)
            + F(theta - 2 * e)
        ) / (12 * h)
    diffs[s] = g - fd
    if s % 400 == 399:
        print(f"  {s+1} seeds, {time.time()-t0:.0f} s", flush=True)
m, se = diffs.mean(0), diffs.std(0, ddof=1) / np.sqrt(S)
np.set_printoptions(precision=3, linewidth=150)
print("mean(analytic - pathwise):", m)
print("SE                       :", se)
print("z                        :", m / np.maximum(se, 1e-300))
print(
    "max|z| over mu/sigma/lambda entries:",
    np.max(np.abs((m / se)[: D * K + K + D])),
)
