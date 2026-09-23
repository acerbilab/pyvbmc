"""O1 test-adequacy claims, recomputed on the shipped fixture (no test run):
(1) the weights at the starting points of the FD tests against the
    weight-penalty threshold 1/(4K) = 0.125;
(2) the per-entry tolerance of test_neg_elcbo_grad_fd_mc_entropy
    (np.allclose, rtol = atol = 1e-2) against the size of each entry;
(3) how far the eta block of that gradient can be scaled before the check
    fails."""
import sys
from pathlib import Path

import gpyreg as gpr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wave7_O1_common import banner

from pyvbmc.testing import check_grad
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import _neg_elcbo

banner()
B = Path("C:/Users/luigi/Documents/GitHub/pyvbmc-wave7/pyvbmc/testing/vbmc")
ld = lambda n: np.loadtxt(open(B / n, "rb"), delimiter=",")
D, K = 2, 2
OPTIONS = {
    "tol_con_loss": 0.01,
    "tol_weight": 1e-2,
    "weight_penalty": 0.1,
    "tol_length": 1e-6,
}


def raw_theta0(seed):
    rng = np.random.default_rng(seed)
    mu = ld("mu.txt") + 0.1 * rng.standard_normal((D, K))
    ln_sigma = np.log(0.5 + 0.5 * rng.random(K))
    ln_lambd = np.log(0.7 + 0.6 * rng.random(D))
    eta = 0.5 * rng.standard_normal(K)
    return np.concatenate([mu.ravel(order="F"), ln_sigma, ln_lambd, eta])


for seed in range(5):
    e = raw_theta0(seed)[-K:]
    w = np.exp(e - e.max())
    w /= w.sum()
    print(f"seed {seed}: w = {w}")

gp = gpr.GP(
    D=D,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.NegativeQuadratic(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
X = ld("X.txt")
gp.update(X_new=X, y_new=ld("y.txt").reshape(-1, 1), hyp=ld("hyp.txt"))
theta_bnd = VariationalPosterior(D, K).get_bounds(X, OPTIONS, K)
theta0 = raw_theta0(2)
theta0[0] = theta_bnd["lb"][0] - 0.5
theta0[D * K] = 1.0
theta0[-K:] -= 20.0
Ns = int(1e4)
f = lambda th: _neg_elcbo(
    th, gp, VariationalPosterior(D, K, rng=7), 0.0, Ns, False, False, theta_bnd
)[0]
g = lambda th: _neg_elcbo(
    th, gp, VariationalPosterior(D, K, rng=7), 0.0, Ns, True, False, theta_bnd
)[1]
dF = g(theta0)
np.set_printoptions(precision=4, linewidth=150)
print("analytic dF at the MC test point:", dF)
tol = 1e-2 + 1e-2 * np.abs(dF)
print("per-entry tolerance / |entry|    :", tol / np.abs(dF))
for scale in (1.10, 1.12, 1.15, 1.2, 1.5):

    def g_bad(th, s=scale):
        d = g(th).copy()
        d[-K:] *= s
        return d

    print(
        f"eta block scaled by {scale}: check_grad passes = {check_grad(f, g_bad, theta0, rtol=1e-2, atol=1e-2)}"
    )

# (4) the entropy's share of each entry at the MC test point, and how much
#     the entropy gradient alone may be off before the check fails
from pyvbmc.entropy import entmc_vbmc

vpH = VariationalPosterior(D, K, rng=7)
vpH.set_parameters(theta0)
_, dH = entmc_vbmc(vpH, Ns, (True,) * 4, True)
print("dH_mc at the MC test point      :", dH)
print("|dH| / per-entry tolerance      :", np.abs(dH) / tol)
for scale in (1.5, 2.0, 5.0, 10.0):

    def g_badH(th, s=scale):
        v = VariationalPosterior(D, K, rng=7)
        v.set_parameters(th)
        _, dh = entmc_vbmc(v, Ns, (True,) * 4, True)
        d = g(th).copy()
        d[-K:] -= (s - 1) * dh[-K:]
        return d

    print(
        f"entropy eta gradient scaled by {scale}: check_grad passes = "
        f"{check_grad(f, g_badH, theta0, rtol=1e-2, atol=1e-2)}"
    )
