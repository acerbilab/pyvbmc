"""Per-component counts of SVBMC.sample(balance_flag=True)."""
import logging

import numpy as np

from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.svbmc import SVBMC
from pyvbmc.variational_posterior import VariationalPosterior

logging.getLogger("SVBMC").setLevel(logging.WARNING)


def make_vp(K, seed, weights):
    pt = ParameterTransformer(D=1)
    vp = VariationalPosterior(D=1, K=K, parameter_transformer=pt)
    r = np.random.default_rng(seed)
    # Far-apart, tight components, so the drawn point identifies its component.
    vp.mu = (np.arange(K, dtype=float) * 1000.0 + seed * 1e6).reshape(1, K)
    vp.sigma = np.full((1, K), 1e-3)
    vp.lambd = np.ones((1, 1))
    w = np.asarray(weights, dtype=float).reshape(1, K)
    vp.w = w / w.sum()
    vp.stats = {
        "stable": True,
        "elbo": 0.0,
        "elbo_sd": 0.01,
        "I_sk": np.zeros((1, K)),
        "J_sjk": np.zeros((1, K, K)),
        "uncertainty_handling_level": 0,
    }
    return vp


runs = [make_vp(3, 1, [0.5, 0.3, 0.2]), make_vp(4, 2, [0.4, 0.3, 0.2, 0.1])]
s = SVBMC(runs, M_min=2, seed=0, show_tips=False, noisy=False)
# Hand-set weights instead of optimizing.
s.w = np.array([[0.20, 0.13, 0.07, 0.25, 0.15, 0.12, 0.08]])

centers = np.concatenate([np.ravel(vp.mu) for vp in s.vp_list])
for n in (10, 37, 100, 1000):
    worst = 0.0
    for trial in range(200):
        X = s.sample(n, balance_flag=True).ravel()
        idx = np.argmin(np.abs(X[:, None] - centers[None, :]), axis=1)
        counts = np.bincount(idx, minlength=centers.size)
        worst = max(worst, np.max(np.abs(counts - n * s.w.ravel())))
    print(f"n={n:5d}  worst |count - n*w| over 200 draws = {worst:.3f}")
