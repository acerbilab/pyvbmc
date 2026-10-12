"""Three targeted probes: bounds mismatch end-to-end, re-optimization, shrinkage levels."""
import logging
import warnings

import numpy as np
import torch

from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.svbmc import SVBMC
from pyvbmc.svbmc._elbo_shrinkage import _two_level_shrinkage
from pyvbmc.variational_posterior import VariationalPosterior

logging.getLogger("SVBMC").setLevel(logging.WARNING)


def make_vp(lb, ub, K=2, seed=1, I=None, J=None):
    D = lb.shape[1]
    finite = np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))
    pt = ParameterTransformer(
        D=D,
        lb_orig=lb,
        ub_orig=ub,
        plb_orig=lb + 0.25 * (ub - lb) if finite else None,
        pub_orig=lb + 0.75 * (ub - lb) if finite else None,
    )
    vp = VariationalPosterior(D=D, K=K, parameter_transformer=pt)
    r = np.random.default_rng(seed)
    vp.mu = r.standard_normal((D, K)) * 0.5
    vp.sigma = np.full((1, K), 0.5)
    vp.lambd = np.ones((D, 1))
    vp.w = np.full((1, K), 1.0 / K)
    vp.stats = {
        "stable": True,
        "elbo": 0.0,
        "elbo_sd": 0.01,
        "I_sk": np.zeros((1, K)) if I is None else np.asarray(I, float),
        "J_sjk": np.zeros((1, K, K)) if J is None else np.asarray(J, float),
        "uncertainty_handling_level": 0,
    }
    vp.stats["elbo"] = float(np.mean(vp.stats["I_sk"]))
    return vp


print("=== 1. bounds mismatch, end to end ===")
a = make_vp(np.array([[-5.0]]), np.array([[5.0]]), seed=1)
b = make_vp(np.array([[-2.0]]), np.array([[2.0]]), seed=2)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    s = SVBMC([a, b], M_min=2, seed=0, show_tips=False)
    s.optimize(n_samples=3, max_steps=20, n_samples_final=4)
print(
    "elbo",
    s.elbo,
    "entropy",
    s.entropy,
    "w",
    np.round(s.w.ravel(), 4),
    "any nan in w:",
    bool(np.any(np.isnan(s.w))),
)

print()
print("=== 2. optimize() twice: second run starts from the first solution ===")
lb, ub = np.array([[-3.0]]), np.array([[3.0]])
runs = [
    make_vp(lb, ub, K=3, seed=1, I=[[0.0, 1.0, 2.0]]),
    make_vp(lb, ub, K=3, seed=2, I=[[3.0, 1.0, 0.5]]),
]
s1 = SVBMC(runs, M_min=2, seed=21, show_tips=False, noisy=False)
w0 = s1.w.copy()
s1.optimize(
    n_samples=8, max_steps=60, n_samples_final=32, version="posterior-only"
)
first = s1.w.copy()
s1.optimize(
    n_samples=8, max_steps=60, n_samples_final=32, version="posterior-only"
)
second = s1.w.copy()
print("initial per-run mass:", [w0.ravel()[0:3].sum(), w0.ravel()[3:6].sum()])
print("after 1st:", np.round(first.ravel(), 5))
print("after 2nd:", np.round(second.ravel(), 5))
ratio1 = first.ravel()[0:3] / first.ravel()[0:3].sum()
print(
    "within-run-0 relative weights after 1st (posterior-only should keep 1/3 each):",
    np.round(ratio1, 5),
)

s2 = SVBMC(runs, M_min=2, seed=21, show_tips=False, noisy=False)
s2.optimize(
    n_samples=8, max_steps=60, n_samples_final=32, version="all-weights"
)
f1 = s2.w.copy()
s2.optimize(
    n_samples=8, max_steps=60, n_samples_final=32, version="all-weights"
)
f2 = s2.w.copy()
print("all-weights, 1st:", np.round(f1.ravel(), 5))
print("all-weights, 2nd:", np.round(f2.ravel(), 5))

print()
print("=== 3. shrinkage: does the corrected run level equal `shifted`? ===")
# Asymmetric within-run configuration so that own @ within != own @ I.
I0 = np.array([0.0, 1.0, 9.0])
I1 = np.array([20.0, 21.0, 22.0])
C0 = np.diag([1.0, 4.0, 0.25])
C1 = np.diag([1.0, 1.0, 1.0])
own = [np.array([0.7, 0.2, 0.1]), np.array([0.2, 0.3, 0.5])]
sel = np.array([0.3, 0.1, 0.1, 0.2, 0.2, 0.1])
value, share = _two_level_shrinkage(
    [I0[None, :], I1[None, :]],
    [C0[None], C1[None]],
    [I0, I1],
    own,
    sel,
    0.0,
)


def within_of(I, C, ownm):
    mean = I.mean()
    spread = I.var(ddof=1)
    n = I.size
    noise = (np.trace(C) - C.sum() / n) / (n - 1)
    excess = max(spread - noise, 0.0)
    if excess > 0:
        return mean + excess * np.linalg.solve(
            excess * np.eye(n) + C, I - mean
        )
    return np.full(n, mean)


w0_ = within_of(I0, C0, own[0])
w1_ = within_of(I1, C1, own[1])
levels = np.array([own[0] @ I0, own[1] @ I1])
lv = np.array([own[0] @ C0 @ own[0], own[1] @ C1 @ own[1]])
run_mean = levels.mean()
run_spread = levels.var(ddof=1)
run_noise = (np.trace(np.diag(lv)) - np.diag(lv).sum() / 2) / 1
run_excess = max(run_spread - run_noise, 0.0)
shifted = run_mean + (run_excess / (run_excess + lv)) * (levels - run_mean)
corrected = np.concatenate(
    [w0_ + shifted[0] - levels[0], w1_ + shifted[1] - levels[1]]
)
print("estimate from module:", value)
print("estimate reproduced :", float(sel @ corrected))
print("target shifted levels :", np.round(shifted, 6))
print(
    "levels of corrected   :",
    np.round([own[0] @ corrected[:3], own[1] @ corrected[3:]], 6),
)
print(
    "own@within vs own@I   :",
    np.round([own[0] @ w0_, own[0] @ I0], 6),
    np.round([own[1] @ w1_, own[1] @ I1], 6),
)
