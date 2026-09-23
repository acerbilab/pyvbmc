"""Test note: the entmc finite-difference tests compare the path gradient
with the derivative of the estimate at fixed draws. At finite Ns the two
differ by the score term at fixed samples, which is zero only in
expectation. Measured on the code: path gradient (dH from the call) against
the total derivative of the code's own H at fixed draws, by central
differences with common random numbers (jacobian_flag=False)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wave7_O2_common import banner

from pyvbmc.entropy import entmc_vbmc
from pyvbmc.variational_posterior import VariationalPosterior

banner()


def H_of(theta, D, K, Ns):
    vp = VariationalPosterior(D, K, rng=0)
    vp.mu = theta[: D * K].reshape((D, K), order="F")
    vp.sigma = theta[D * K : D * K + K].reshape(1, K)
    vp.lambd = theta[D * K + K : D * K + K + D].reshape(D, 1)
    vp.w = theta[D * K + K + D :].reshape(1, K)
    vp.eta = np.log(vp.w)
    return entmc_vbmc(vp, Ns, (True,) * 4, False, rng=42)


for D, K, Ns in [(3, 2, 40), (3, 2, 1000), (3, 2, 100000)]:
    rs = np.random.default_rng(1)
    th = np.concatenate(
        [
            rs.uniform(-1, 1, D * K),
            1 + 0.2 * rs.random(K),
            1 + 0.2 * rs.random(D),
            [0.55, 0.45],
        ]
    )
    _, dH = H_of(th, D, K, Ns)
    fd = np.zeros_like(th)
    for i in range(th.size):
        h = 1e-5
        tp, tm = th.copy(), th.copy()
        tp[i] += h
        tm[i] -= h
        fd[i] = (H_of(tp, D, K, Ns)[0] - H_of(tm, D, K, Ns)[0]) / (2 * h)
    n = D * K + K + D
    r = np.abs(dH[:n] - fd[:n]) / np.maximum(np.abs(fd[:n]), 1e-3)
    rw = np.max(np.abs(dH[n:] - fd[n:]) / np.abs(fd[n:]))
    print(
        f"D={D} K={K} Ns={Ns}: mu/sigma/lambda entries, relative "
        f"difference path vs fixed-draw derivative: median "
        f"{np.median(r):.1e}, max {np.max(r):.1e}; weight entries max "
        f"{rw:.1e}",
        flush=True,
    )
