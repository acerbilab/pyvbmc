"""Test note: the unasserted check_grad of test_entmc_vbmc_nonoverlapping_mixture
(test_entmc_vbmc.py:128). Recomputes the same comparison (same wrapper
logic, same check_grad helper, rtol=0.01) outside pytest and prints whether
it would pass if asserted, and the worst entries."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.differentiate import jacobian
from wave7_O2_common import banner

from pyvbmc.entropy import entmc_vbmc
from pyvbmc.testing import check_grad
from pyvbmc.variational_posterior import VariationalPosterior

banner()


def wrapper(theta, D, K, Ns, ret):
    vp = VariationalPosterior(D, K, rng=0)
    vp.mu = np.reshape(theta[: D * K], (D, K), "F")
    vp.sigma = theta[D * K : D * K + K]
    vp.lambd = theta[D * K + K : D * K + K + D]
    vp.w = theta[D * K + K + D :]
    flags = (ret != "H",) * 4
    H, dH = entmc_vbmc(vp, Ns, grad_flags=flags, jacobian_flag=False, rng=42)
    return H if ret == "H" else dH


Ns = 1e5
for D in (1, 2):
    for K in (2, 3):
        mu = np.stack([np.ones(D) * 10 * i for i in range(K)], 1)
        theta0 = np.concatenate(
            [mu.T.ravel(), np.ones(K), np.ones(D), np.ones(K) / K]
        )
        f = lambda t: wrapper(t, D, K, Ns, "H")
        g = lambda t: wrapper(t, D, K, Ns, "dH")
        ok = check_grad(f, g, theta0, rtol=0.01)
        an = g(theta0)
        num = jacobian(
            lambda x: np.apply_along_axis(f, 0, x), theta0, initial_step=1e-2
        ).df
        bad = np.abs(an - num) > 1e-6 + 0.01 * np.abs(num)
        print(
            f"D={D} K={K}: check_grad would return {ok}; entries outside "
            f"the band: {int(bad.sum())} of {an.size}; max |diff| "
            f"{np.max(np.abs(an - num)):.2e}",
            flush=True,
        )
