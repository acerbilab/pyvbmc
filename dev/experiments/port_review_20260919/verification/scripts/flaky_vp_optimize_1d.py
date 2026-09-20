"""Spread of `test_vp_optimize_1D_g_mixture` over fixed seeds.

Replays the body of the test (GP fit to the log density of an equal
mixture of N(-2, 1) and N(2, 1) on 200 grid points, then `optimize_vp` with
100 fast and 2 slow optimizations from a K = 2 posterior) with the global
NumPy stream seeded, which is the stream the unseeded test draws from, and
records the two asserted quantities: |ELBO| (threshold 0.05) and the
moment-matched KL divergence to the true mixture (threshold 0.0015). The KL
uses the exact moments of the fitted posterior and of the target, so it has
no Monte Carlo noise of its own.

Usage: python -u flaky_vp_optimize_1d.py [n_seeds]
"""
import sys
import time

import gpyreg as gpr
import numpy as np
from scipy.stats import norm

from pyvbmc.stats import kl_div_mvn
from pyvbmc.testing.vbmc.test_variational_optimization import setup_options
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import optimize_vp

n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
D = 1
X = np.linspace(-5, 5, 200).reshape(-1, 1)
y = np.log(
    0.5 * norm.pdf(X, loc=-2, scale=1) + 0.5 * norm.pdf(X, loc=2, scale=1)
)
true_mu, true_sigma = np.array([[0.0]]), np.array([[np.sqrt(5.0)]])

rows = []
for seed in range(n_seeds):
    t0 = time.perf_counter()
    np.random.seed(seed)
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(),
    )
    gp.fit(X, y)
    vp = VariationalPosterior(D=D, K=2)
    optim_state = {"warmup": True, "entropy_switch": False}
    options = setup_options(D, {})
    vp, _, _ = optimize_vp(options, optim_state, vp, gp, 100, 2)
    mu, cov = vp.moments(orig_flag=True, cov_flag=True)
    kl = np.abs(kl_div_mvn(true_mu, true_sigma**2, mu, cov))
    kl_max = float(np.max(kl))
    elbo = float(vp.stats["elbo"])
    ok_elbo = abs(elbo) < 5e-2
    ok_kl = kl_max < 1.5e-3
    rows.append((seed, elbo, kl_max, ok_elbo, ok_kl))
    print(
        f"seed {seed:3d}  elbo {elbo:+.5f}  kl {kl_max:.6f}  K {vp.K}  "
        f"mu {np.round(np.ravel(vp.mu), 3)}  sigma*lambda "
        f"{np.round(np.ravel(vp.sigma * vp.lambd), 3)}  w "
        f"{np.round(np.ravel(vp.w), 3)}  "
        f"{'ok' if ok_elbo and ok_kl else 'FAIL ' + ('elbo ' if not ok_elbo else '') + ('kl' if not ok_kl else '')}"
        f"  [{time.perf_counter() - t0:.1f}s]",
        flush=True,
    )

a = np.array([(r[1], r[2]) for r in rows])
print(flush=True)
print(
    f"runs {len(rows)}; fail elbo {sum(not r[3] for r in rows)}; "
    f"fail kl {sum(not r[4] for r in rows)}",
    flush=True,
)
print(
    f"|elbo|: median {np.median(np.abs(a[:, 0])):.5f}  max "
    f"{np.max(np.abs(a[:, 0])):.5f}  (threshold 0.05)",
    flush=True,
)
print(
    f"kl    : median {np.median(a[:, 1]):.6f}  90% {np.quantile(a[:, 1], 0.9):.6f}  "
    f"max {np.max(a[:, 1]):.6f}  (threshold 0.0015)",
    flush=True,
)
