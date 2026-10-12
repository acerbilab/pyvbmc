"""O3 F5: the search box that warp_input computes after a rotoscaling warp,
against the exact image of the old box, as a function of D.

The real warp_input runs on constructed states: the default search box of
a fresh run (the plausible box plus two plausible widths on each side,
vbmc.py:1218-1228, `active_search_bound = 2`), and a correlated variational
posterior of a size typical after warm-up (component means spread by about
0.15 and scales of about 0.1 in the transformed space, where the plausible
box is [-0.5, 0.5]). Half the coordinates are bounded (probit), half not.

Reported per D, over 10 states each:
- extent: the new box's width per coordinate over the width of the exact
  bounding box of the affine image of the old box (the reviewer's figure);
- lost volume: the fraction of 1e5 fresh uniform points of the old box
  whose image falls outside the new box;
- the fraction of 1e5 draws of the warped posterior outside the new box;
- the fraction of the image of the old plausible box outside the new box;
- the smallest distance from the warped posterior's mean to a face of the
  new box, in units of the posterior's marginal SD along that coordinate.
"""
import contextlib
import io

import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.whitening import warp_gp_and_vp, warp_input

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpr.__file__)

options = {
    "warp_nonlinear": False,
    "warp_rotoscaling": True,
    "warp_roto_corr_thresh": 0.05,
    "warp_cov_reg": 0,
}


class _Owner:
    def __init__(self, optim_state, D):
        self.optim_state = optim_state
        self.D = D


def one_state(D, seed):
    rng = np.random.default_rng(seed)
    nb = D // 2
    lb = np.concatenate([np.full(nb, -3.0), np.full(D - nb, -np.inf)])[None]
    ub = np.concatenate([np.full(nb, 7.0), np.full(D - nb, np.inf)])[None]
    plb = np.full((1, D), -1.0)
    pub = np.full((1, D), 4.0)
    pt = ParameterTransformer(D, lb, ub, plb, pub, transform_type="probit")
    fl = FunctionLogger(
        lambda x: -0.5 * float(np.sum(np.atleast_2d(x) ** 2)),
        D,
        False,
        0,
        parameter_transformer=pt,
    )
    # a random correlation structure for the posterior
    B = rng.standard_normal((D, D))
    C = B @ B.T
    dC = np.sqrt(np.diag(C))
    L = np.linalg.cholesky(C / np.outer(dC, dC))
    for u in 0.15 * rng.standard_normal((2 * D + 5, D)) @ L.T:
        fl(u)
    K = 8
    vp = VariationalPosterior(
        D,
        K,
        x0=np.zeros((1, D)),
        parameter_transformer=pt,
        rng=np.random.default_rng(seed + 1),
    )
    vp.mu = (
        0.15 * rng.standard_normal((K, D)) @ L.T + 0.1 * rng.standard_normal(D)
    ).T
    vp.sigma = rng.uniform(0.07, 0.13, (1, K))
    lam = rng.uniform(0.6, 1.4, (D, 1))
    vp.lambd = lam / np.sqrt(np.sum(lam**2) / D)
    vp.w = np.full((1, K), 1.0 / K)
    vp.eta = np.log(vp.w)
    plb_t, pub_t = pt(plb), pt(pub)
    pr = pub_t - plb_t
    optim_state = {
        "N": fl.Xn + 1,
        "plb_orig": plb,
        "pub_orig": pub,
        "lb_search": plb_t - 2 * pr,
        "ub_search": pub_t + 2 * pr,
        "search_cache": None,
        "iter": 12,
        "warping_count": 0,
    }
    gp = gpr.GP(
        D,
        gpr.covariance_functions.SquaredExponential(),
        gpr.mean_functions.ConstantMean(),
        gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    hyp = np.concatenate([np.zeros(D), [0.0], [np.log(1e-3)], [0.0]])[None]
    gp.update(
        X_new=fl.X[fl.X_flag],
        y_new=fl.y[fl.X_flag],
        hyp=hyp,
        compute_posterior=True,
    )
    return pt, fl, vp, gp, optim_state, plb, pub


rng_check = np.random.default_rng(123)
print(
    f"{'D':>3} {'extent med':>10} {'extent min':>10} {'lost vol':>9} "
    f"{'VP out':>8} {'plaus out':>9} {'min margin (SD)':>15} "
    f"{'box half-width / VP SD, median':>30}"
)
for D in (2, 3, 5, 10, 15, 20):
    ext, lost, vpout, plout, margin, hw = [], [], [], [], [], []
    for rep in range(10):
        pt, fl, vp, gp, os_, plb, pub = one_state(D, 1000 * D + rep)
        old_lo, old_hi = os_["lb_search"][0], os_["ub_search"][0]
        pt_new, os_new, fl_new, _ = warp_input(vp, os_, fl, options)
        with contextlib.redirect_stdout(io.StringIO()):
            vp_new, _ = warp_gp_and_vp(pt_new, gp, vp, _Owner(os_new, D))

        def warpfun(u):
            return pt_new(pt.inverse(u))

        c0 = warpfun(np.zeros((1, D)))[0]
        A = np.column_stack(
            [warpfun(np.eye(D)[j : j + 1])[0] - c0 for j in range(D)]
        )
        cc, hh = 0.5 * (old_lo + old_hi), 0.5 * (old_hi - old_lo)
        exact_w = 2 * np.abs(A) @ hh
        lo, hi = os_new["lb_search"][0], os_new["ub_search"][0]
        ext.append((hi - lo) / exact_w)
        U = rng_check.random((100000, D)) * (old_hi - old_lo) + old_lo
        V = U @ A.T + c0
        lost.append(np.mean(np.any((V < lo) | (V > hi), axis=1)))
        Xs, _ = vp_new.sample(100000, orig_flag=False)
        vpout.append(np.mean(np.any((Xs < lo) | (Xs > hi), axis=1)))
        P = rng_check.random((100000, D)) * (pub - plb) + plb
        Pn = pt_new(P)
        plout.append(np.mean(np.any((Pn < lo) | (Pn > hi), axis=1)))
        m, S = vp_new.moments(orig_flag=False, cov_flag=True)
        sd = np.sqrt(np.diag(S))
        m = np.ravel(m)
        margin.append(np.min(np.minimum(m - lo, hi - m) / sd))
        hw.append(np.median(0.5 * (hi - lo) / sd))
    ext = np.concatenate(ext)
    print(
        f"{D:3d} {np.median(ext):10.3f} {ext.min():10.3f} "
        f"{np.mean(lost):9.4f} {np.max(vpout):8.1e} {np.max(plout):9.1e} "
        f"{np.min(margin):15.1f} {np.median(hw):30.1f}"
    )

# The sampled box against the unscented alternative MATLAB leaves under
# `if 0` (warp_input_vbmc.m:134-141), and the expected lost volume by order
# statistics: a fresh uniform point falls outside the range of 1000 others
# in one coordinate with probability 2/1001.
print("\norder-statistics estimate of the lost volume, 1-(1-2/1001)^D:")
for D in (2, 3, 5, 10, 15, 20):
    print(f"  D={D:2d}: {1 - (1 - 2 / 1001) ** D:.4f}")
