"""O3 F5, compounding: two successive warps with the real warp_input.

After the first warp the posterior is replaced by a new correlated one in
the warped space (unit-scale means spread, scales of about 0.5: the kind
of posterior the run fits between two warps), and the state is warped
again. Reported: the extent of the box after the second warp against the
exact image of the ORIGINAL box through both maps, the fraction of the
original box's volume lost, and the fraction of the second warped
posterior's mass outside the box.
"""
import contextlib
import importlib.util
import io
import pathlib

import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.whitening import warp_input

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpr.__file__)

here = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "f5", here / "O3_F5_search_box_after_warp.py"
)
f5 = importlib.util.module_from_spec(spec)
# Import the helpers only: the module body runs its table, so silence it.
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(f5)


def affine(fun, D):
    c0 = fun(np.zeros((1, D)))[0]
    A = np.column_stack([fun(np.eye(D)[j : j + 1])[0] - c0 for j in range(D)])
    return A, c0


rng = np.random.default_rng(7)
for D in (5, 10, 20):
    ext, lost, vpout = [], [], []
    for rep in range(5):
        pt, fl, vp, gp, os_, plb, pub = f5.one_state(D, 5000 * D + rep)
        lo0, hi0 = os_["lb_search"][0].copy(), os_["ub_search"][0].copy()
        pt1, os1, fl1, _ = warp_input(vp, os_, fl, f5.options)
        # a new correlated posterior in the warped space
        B = rng.standard_normal((D, D))
        C = B @ B.T
        dC = np.sqrt(np.diag(C))
        L = np.linalg.cholesky(C / np.outer(dC, dC))
        K = 8
        vp1 = VariationalPosterior(
            D,
            K,
            x0=np.zeros((1, D)),
            parameter_transformer=pt1,
            rng=np.random.default_rng(rep),
        )
        vp1.mu = (rng.standard_normal((K, D)) @ L.T).T
        vp1.sigma = rng.uniform(0.4, 0.6, (1, K))
        vp1.lambd = np.ones((D, 1))
        vp1.w = np.full((1, K), 1.0 / K)
        vp1.eta = np.log(vp1.w)
        os1["iter"] = 30
        pt2, os2, fl2, _ = warp_input(vp1, os1, fl1, f5.options)
        A, c0 = affine(lambda u: pt2(pt.inverse(u)), D)
        hh = 0.5 * (hi0 - lo0)
        exact_w = 2 * np.abs(A) @ hh
        lo, hi = os2["lb_search"][0], os2["ub_search"][0]
        ext.append((hi - lo) / exact_w)
        U = rng.random((100000, D)) * (hi0 - lo0) + lo0
        V = U @ A.T + c0
        lost.append(np.mean(np.any((V < lo) | (V > hi), axis=1)))
        # the second posterior, mapped into the twice-warped space exactly
        # (affine map of the samples of vp1)
        A12, c12 = affine(lambda u: pt2(pt1.inverse(u)), D)
        Xs, _ = vp1.sample(100000, orig_flag=False)
        Xs2 = Xs @ A12.T + c12
        vpout.append(np.mean(np.any((Xs2 < lo) | (Xs2 > hi), axis=1)))
    ext = np.concatenate(ext)
    print(
        f"D={D:2d}: extent after two warps vs exact image of the original "
        f"box: median {np.median(ext):.3f}, min {ext.min():.3f}; lost volume "
        f"{np.mean(lost):.4f}; posterior mass outside {np.max(vpout):.1e}"
    )
