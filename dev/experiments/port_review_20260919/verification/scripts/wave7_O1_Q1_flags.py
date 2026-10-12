"""O1: _neg_elcbo gradient (deterministic entropy, soft bounds active, weight
penalty active where the weights are optimized) under all 16 combinations of
the optimize_* flags, D = 3, K = 2, FD through a fresh posterior per
evaluation, frozen parameters held at fixed values, lambda of unit RMS.
The first component is wide enough that its log scale is above its soft
bound. The combinations with sigma frozen and lambda free are F1's; the others
must agree."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wave7_O1_common import OPTS, banner, build_gp, central_fd, matlab_negelcbo

from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import _neg_elcbo

banner()
D, K = 3, 2
gp = build_gp(D, N=24, Ns=2, seed=21)


def base_vp(flags):
    vp = VariationalPosterior(D, K, rng=0)
    vp.mu = np.array([[gp.X[:, 0].min() - 0.3, 0.4], [0.2, -0.5], [0.0, 0.3]])
    vp.sigma = np.array(
        [[6.0, 0.5]]
    )  # sigma_1*lambda_1 above the ln-scale bound
    lam = np.array([1.3, 0.9, 0.7])
    lam /= np.sqrt(np.mean(lam**2))
    vp.lambd = lam.reshape(-1, 1)
    vp.w = np.array([[0.05, 0.95]])
    vp.eta = np.log(vp.w)
    (
        vp.optimize_mu,
        vp.optimize_sigma,
        vp.optimize_lambd,
        vp.optimize_weights,
    ) = flags
    return vp


for flags in itertools.product((True, False), repeat=4):
    if not any(flags):
        continue
    vp0 = base_vp(flags)
    bnd = vp0.get_bounds(gp.X, OPTS, K)
    theta = vp0.get_parameters()
    f = lambda th: _neg_elcbo(
        th, gp, base_vp(flags), 0.0, 0, False, False, bnd
    )[0]
    dF = _neg_elcbo(theta, gp, base_vp(flags), 0.0, 0, True, False, bnd)[1]
    fd = central_fd(f, theta, 1e-5)
    e = np.max(np.abs(dF - fd)) / max(1.0, np.max(np.abs(fd)))
    Fm = matlab_negelcbo(theta, vp0, gp, bnd, False)
    Fp = f(theta)
    tag = "  <- F1 configuration" if (not flags[1] and flags[2]) else ""
    print(
        f"mu,sig,lam,w = {tuple(int(x) for x in flags)} | n={theta.size:2d} | "
        f"rel |dF - FD| = {e:.1e} | F py - F MATLAB transcr. = {Fp-Fm:+.2e}{tag}",
        flush=True,
    )
