"""O3 Q3: spot checks of what a rotoscaling warp re-expresses, on a
bounded problem (the warp tests use unbounded transformers only).

State: the constructed D=3 state of the F2 script (two probit coordinates,
one unbounded), negative-quadratic GP, warp_roto_corr_thresh = 0 so that
nothing is thresholded. warpfun is affine, u' = A u + c.
Checked: A Sigma_vp A^T = I; the GP length scales equal
sqrt(A^2 ell^2) (marginal); x_m' = A x_m + c; omega' = sqrt(A^2 omega^2);
sigma' lambda' equals the marginal SD of each component exactly and
sum(lambda'^2) = D; the weights are unchanged; the search cache is mapped
exactly.
"""
import contextlib
import importlib.util
import io
import pathlib

import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc.whitening import warp_gp_and_vp, warp_input

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpr.__file__)

here = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "f2", here / "O3_F2_zero_mean_warp.py"
)
f2 = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(f2)

D = 3
pt, fl, vp, gp, os_ = f2.state("negquad", seed=5)
os_["search_cache"] = np.random.default_rng(1).standard_normal((4, D)) * 0.3
opts = dict(f2.options, warp_roto_corr_thresh=0)
pt_new, os_new, fl_new, _ = warp_input(vp, os_, fl, opts)
vp_new, hyp_w = warp_gp_and_vp(pt_new, gp, vp, f2._Owner(os_new))


def warpfun(u):
    return pt_new(pt.inverse(u))


c = warpfun(np.zeros((1, D)))[0]
A = np.column_stack([warpfun(np.eye(D)[j : j + 1])[0] - c for j in range(D)])
U = np.random.default_rng(2).standard_normal((50, D))
print(
    f"affine residual of warpfun: {np.max(np.abs(warpfun(U) - (U @ A.T + c))):.1e}"
)
_, S = vp.moments(orig_flag=False, cov_flag=True)
print(f"max |A Sigma A^T - I|: {np.max(np.abs(A @ S @ A.T - np.eye(D))):.1e}")
hyp = gp.posteriors[0].hyp
ell = np.exp(hyp[:D])
print(
    f"length scales: max rel |ell' - sqrt(A^2 ell^2)|: "
    f"{np.max(np.abs(np.exp(hyp_w[0, :D]) / np.sqrt((A**2) @ ell**2) - 1)):.1e}"
)
xm = hyp[D + 3 : 2 * D + 3]
om = np.exp(hyp[2 * D + 3 :])
print(
    f"x_m: max |x_m' - (A x_m + c)|: {np.max(np.abs(hyp_w[0, D + 3 : 2 * D + 3] - (A @ xm + c))):.1e}"
)
print(
    f"omega: max rel |omega' - sqrt(A^2 omega^2)|: "
    f"{np.max(np.abs(np.exp(hyp_w[0, 2 * D + 3 :]) / np.sqrt((A**2) @ om**2) - 1)):.1e}"
)
sl = (vp.lambd * vp.sigma).T
sl_exact = np.sqrt(sl**2 @ (A**2).T)
sl_new = (vp_new.lambd * vp_new.sigma).T
print(
    f"VP: max rel |sigma' lambda' - marginal SD|: {np.max(np.abs(sl_new / sl_exact - 1)):.1e}; "
    f"sum lambda'^2 = {np.sum(vp_new.lambd**2):.15f}"
)
print(
    f"VP means: max |mu' - (A mu + c)|: {np.max(np.abs(vp_new.mu.T - (vp.mu.T @ A.T + c))):.1e}; "
    f"weights: max |w' - w| {np.max(np.abs(vp_new.w - vp.w)):.1e}"
)
print(
    f"search cache: max |cache' - warpfun(cache)|: "
    f"{np.max(np.abs(os_new['search_cache'] - warpfun(os_['search_cache']))):.1e}"
)
print(
    f"new transformer: mu {pt_new.mu}, delta {pt_new.delta}, det R {np.linalg.det(pt_new.R_mat):.15f}"
)
