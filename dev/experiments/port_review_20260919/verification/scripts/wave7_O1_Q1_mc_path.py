"""O1 first question, the Adam path: _neg_elcbo with the Monte Carlo entropy.
Mean analytic dF over seeds against central differences of the mean F over
the same seeds (common random numbers per seed), soft bounds and the weight
penalty active, D = 2, K = 3; z = difference / standard error of the mean
analytic gradient. Also dF_mc - dF_lb = -(dH_mc - dH_lb) exactly."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wave7_O1_common import OPTS, banner, build_gp

from pyvbmc.entropy import entlb_vbmc, entmc_vbmc
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import _neg_elcbo

banner()
D, K, Ns_ent, S = 2, 3, 60, 120
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


def F(th, seed):
    return _neg_elcbo(
        th,
        gp,
        VariationalPosterior(D, K, rng=seed),
        0.0,
        Ns_ent,
        False,
        False,
        bnd,
    )[0]


dFs = np.array(
    [
        _neg_elcbo(
            theta,
            gp,
            VariationalPosterior(D, K, rng=s),
            0.0,
            Ns_ent,
            True,
            False,
            bnd,
        )[1]
        for s in range(S)
    ]
)
vp = VariationalPosterior(D, K, rng=0)
_neg_elcbo(theta, gp, vp, 0.0, Ns_ent, False, False, bnd)
print("weights", vp.w.ravel(), "threshold", bnd["weight_threshold"])
mean_d, se = dFs.mean(0), dFs.std(0, ddof=1) / np.sqrt(S)
h = 1e-4
fd = np.zeros_like(theta)
for i in range(theta.size):
    e = np.zeros_like(theta)
    e[i] = h
    fd[i] = np.mean(
        [
            (
                -F(theta + 2 * e, s)
                + 8 * F(theta + e, s)
                - 8 * F(theta - e, s)
                + F(theta - 2 * e, s)
            )
            / (12 * h)
            for s in range(S)
        ]
    )
z = (mean_d - fd) / np.maximum(se, 1e-300)
np.set_printoptions(precision=4, linewidth=150)
print("mean analytic dF:", mean_d)
print("FD of mean F    :", fd)
print("SE              :", se)
print("z               :", z, " max|z| =", np.max(np.abs(z)))
# exact identity at one seed
v1 = VariationalPosterior(D, K, rng=3)
v2 = VariationalPosterior(D, K, rng=3)
dmc = _neg_elcbo(theta, gp, v1, 0.0, Ns_ent, True, False, bnd)[1]
dlb = _neg_elcbo(theta, gp, v2, 0.0, 0, True, False, bnd)[1]
v3 = VariationalPosterior(D, K, rng=3)
v3.set_parameters(theta)
flags = (True,) * 4
_, dHmc = entmc_vbmc(v3, Ns_ent, flags, True)
v4 = VariationalPosterior(D, K)
v4.set_parameters(theta)
_, dHlb = entlb_vbmc(v4, flags, True)
print(
    "max |(dF_mc - dF_lb) + (dH_mc - dH_lb)| =",
    np.max(np.abs((dmc - dlb) + (dHmc - dHlb))),
)
