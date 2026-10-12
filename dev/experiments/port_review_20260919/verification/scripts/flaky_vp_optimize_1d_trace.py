"""Where does a failing seed of `test_vp_optimize_1D_g_mixture` go wrong?

Replays the body of the test for the given seeds with `_sieve` and
`minimize_adam` wrapped, so that the random stream is the one of the plain
run, and prints: the best sieve candidates of each type with their sieve
ELBO, the starting point and the end point of each slow (Adam)
optimization, and the final posterior.

Usage: python -u flaky_vp_optimize_1d_trace.py 16 0
"""
import sys

import gpyreg as gpr
import numpy as np
from scipy.stats import norm

import pyvbmc.vbmc.variational_optimization as vo
from pyvbmc.testing.vbmc.test_variational_optimization import setup_options
from pyvbmc.variational_posterior import VariationalPosterior

seeds = [int(s) for s in sys.argv[1:]] or [16, 0]
D = 1
X = np.linspace(-5, 5, 200).reshape(-1, 1)
y = np.log(
    0.5 * norm.pdf(X, loc=-2, scale=1) + 0.5 * norm.pdf(X, loc=2, scale=1)
)

orig_sieve, orig_adam = vo._sieve, vo.minimize_adam


def describe(vp):
    return (
        f"mu {np.round(np.ravel(vp.mu), 2)} "
        f"width {np.round(np.ravel(vp.sigma * vp.lambd), 2)} "
        f"w {np.round(np.ravel(vp.w), 2)}"
    )


def theta_str(theta, K=2):
    theta = np.ravel(theta)
    return f"mu {np.round(theta[:K], 2)} ln_sigma {np.round(theta[K:2*K], 2)} rest {np.round(theta[2*K:], 2)}"


for seed in seeds:
    print(f"\n================ seed {seed} ================", flush=True)

    def sieve(*args, **kwargs):
        out = orig_sieve(*args, **kwargs)
        vp0_vec, vp0_type = out[0], out[1]
        print(
            f"sieve returned {len(vp0_vec)} candidates, ranked best first",
            flush=True,
        )
        for t in (1, 2, 3):
            idx = np.where(np.asarray(vp0_type) == t)[0]
            for j in idx[:2]:
                print(
                    f"  type {t} rank {j:3d}: {describe(vp0_vec[j])}",
                    flush=True,
                )
        return out

    def adam(f, x0, *args, **kwargs):
        start = np.array(x0, copy=True)
        res = orig_adam(f, x0, *args, **kwargs)
        x_end, y_end = res[0], res[1]
        print(f"  adam: start {theta_str(start)}", flush=True)
        print(
            f"        end   {theta_str(x_end)}  f_end {float(y_end):+.4f}",
            flush=True,
        )
        return res

    vo._sieve, vo.minimize_adam = sieve, adam
    try:
        np.random.seed(seed)
        gp = gpr.GP(
            D=D,
            covariance=gpr.covariance_functions.SquaredExponential(),
            mean=gpr.mean_functions.NegativeQuadratic(),
            noise=gpr.noise_functions.GaussianNoise(),
        )
        gp.fit(X, y)
        hyp = gp.get_hyperparameters(as_array=True)
        print(
            f"GP: {hyp.shape[0]} hyperparameter sample(s); first "
            f"{np.round(hyp[0], 3)}",
            flush=True,
        )
        vp = VariationalPosterior(D=D, K=2)
        print(f"initial vp: {describe(vp)}", flush=True)
        optim_state = {"warmup": True, "entropy_switch": False}
        options = setup_options(D, {})
        vp, _, _ = vo.optimize_vp(options, optim_state, vp, gp, 100, 2)
        print(
            f"final: {describe(vp)}  elbo {float(vp.stats['elbo']):+.5f} "
            f"elbo_sd {float(vp.stats['elbo_sd']):.5f}",
            flush=True,
        )
    finally:
        vo._sieve, vo.minimize_adam = orig_sieve, orig_adam
