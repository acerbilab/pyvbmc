"""O4, first question: the gradient of the fit objective (log likelihood plus
hyperprior) and of the log likelihood alone, by finite differences, on GPs
built by PyVBMC's `_gp_hyp` at uncertainty levels 0, 1 (the multiplier of the
recorded noise, `scale_user_provided`, with s2 = 1/n as FunctionLogger pools
repeats) and 2 (recorded noise added as is), each mean function, D = 1 and
3. The level-1 multiplier's derivative is also checked against a dense
evaluation of 0.5 * sum_i exp(h_mult) s2_i Q_ii written here."""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpyreg as gpr
import numpy as np
from wave7_O4_common import banner, fd_grad

from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import (
    _cov_identifier_to_covariance_function,
    _gp_hyp,
    _meanfun_name_to_mean_function,
)

banner()
warnings.simplefilter("ignore")


def comp_err(a, b):
    """Per-component error, relative to max(|b_i|, 1e-3 * max|b|)."""
    scale = np.maximum(np.abs(b), 1e-3 * np.max(np.abs(b)))
    return np.max(np.abs(a - b) / scale)


def dense_nlZ_parts(gp, h):
    """C, alpha, Q from dense linear algebra (SE ARD, noise [1,2,0])."""
    X, y, s2 = gp.X, gp.y, gp.s2
    N, D = X.shape
    ell = np.exp(h[:D])
    sf2 = np.exp(2 * h[D])
    Z = X / ell
    r2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, 2)
    K = sf2 * np.exp(-r2 / 2)
    sn2 = np.exp(2 * h[D + 1]) + np.exp(h[D + 2]) * s2[:, 0]
    C = K + np.diag(sn2)
    m = gp.mean.compute(h[D + 3 :], X).reshape(-1, 1)
    Ci = np.linalg.inv(C)
    alpha = Ci @ (y - m)
    Q = Ci - alpha @ alpha.T
    return Q, s2[:, 0]


worst = {}
for D in (1, 3):
    for level in (0, 1, 2):
        for meanfun in ("zero", "const", "negquad"):
            rng = np.random.default_rng(1000 + 10 * D + level)
            f = (
                (lambda x: (-0.5 * np.sum(x**2), 0.3))
                if level == 2
                else (lambda x: -0.5 * np.sum(x**2))
            )
            opts = {"gp_mean_fun": meanfun, "display": "off"}
            if level == 1:
                opts["uncertainty_handling"] = True
            if level == 2:
                opts["specify_target_noise"] = True
            vb = VBMC(
                f,
                np.zeros((1, D)),
                np.full((1, D), -np.inf),
                np.full((1, D), np.inf),
                -np.ones((1, D)),
                np.ones((1, D)),
                opts,
            )
            os_ = vb.optim_state
            nf = os_["gp_noise_fun"]
            gp = gpr.GP(
                D=D,
                covariance=_cov_identifier_to_covariance_function(
                    os_["gp_cov_fun"]
                ),
                mean=_meanfun_name_to_mean_function(os_["gp_mean_fun"]),
                noise=gpr.noise_functions.GaussianNoise(
                    constant_add=nf[0] == 1,
                    user_provided_add=nf[1] > 0,
                    scale_user_provided=nf[1] == 2,
                    rectified_linear_output_dependent_add=nf[2] == 1,
                ),
            )
            N = 18
            X = rng.uniform(-1, 1, (N, D))
            y = -0.5 * np.sum(X**2, 1) + (
                0.3 * rng.standard_normal(N) if level else 0.0
            )
            y = y.reshape(-1, 1)
            s2 = None
            if level == 1:
                s2 = (1.0 / rng.integers(1, 6, N)).reshape(-1, 1)
            elif level == 2:
                s2 = (0.05 + 0.2 * rng.uniform(size=N)).reshape(-1, 1) ** 2
            os_["N"], os_["stop_sampling"] = N, 0
            gp, hyp0, _ = _gp_hyp(
                os_, vb.options, os_["plb_tran"], os_["pub_tran"], gp, X, y
            )
            Xc, yc, s2c = gp._convert_shapes(X, y, s2)
            gp.X, gp.y, gp.s2 = Xc, yc, s2c
            gp.set_bounds(
                gp.get_recommended_bounds(gp.lower_bounds, gp.upper_bounds)
            )
            lb, ub = gp.lower_bounds, gp.upper_bounds
            # df of the Student's t priors: set explicitly to 3 by _gp_hyp
            e_obj = e_lik = e_mult = 0.0
            for k in range(3):
                h = hyp0 + 0.4 * rng.standard_normal(hyp0.size)
                h = np.clip(
                    h,
                    np.where(np.isfinite(lb), lb + 1e-2, -1e9),
                    np.where(np.isfinite(ub), ub - 1e-2, 1e9),
                )
                _, g = gp._GP__gp_obj_fun(h, True, False)
                fd = fd_grad(
                    lambda hh: gp._GP__gp_obj_fun(hh, False, False), h, 1e-4
                )
                e_obj = max(e_obj, comp_err(g, fd))
                _, gl = gp.log_likelihood(h, compute_grad=True)
                fdl = fd_grad(lambda hh: gp.log_likelihood(hh), h, 1e-4)
                e_lik = max(e_lik, comp_err(gl, fdl))
                if level == 1:
                    Q, s2v = dense_nlZ_parts(gp, h)
                    dmult = 0.5 * np.sum(np.exp(h[D + 2]) * s2v * np.diag(Q))
                    e_mult = max(
                        e_mult,
                        abs(-gl[D + 2] - dmult) / max(abs(dmult), 1e-12),
                    )
            extra = (
                f" mult vs dense formula rel={e_mult:.1e}"
                if level == 1
                else ""
            )
            print(
                f"D={D} L{level} noise={nf} {meanfun:7s} hyp_N={h.size:2d}: "
                f"obj per-comp err={e_obj:.1e}  lik per-comp err={e_lik:.1e}{extra}",
                flush=True,
            )

print(
    "\nRepeated training inputs (two exact duplicates), level-1 and level-2 noise, negquad, D=2"
)
for nf in ((1, 2, 0), (1, 1, 0), (1, 0, 0)):
    rng = np.random.default_rng(5)
    D, N = 2, 14
    X = rng.uniform(-1, 1, (N, D))
    X[1] = X[0]
    X[3] = X[2]
    y = (-0.5 * np.sum(X**2, 1) + 0.1 * rng.standard_normal(N)).reshape(
        -1, 1
    )
    s2 = (
        None
        if nf[1] == 0
        else (0.02 + 0.1 * rng.uniform(size=N)).reshape(-1, 1)
    )
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True,
            user_provided_add=nf[1] > 0,
            scale_user_provided=nf[1] == 2,
        ),
    )
    Xc, yc, s2c = gp._convert_shapes(X, y, s2)
    gp.X, gp.y, gp.s2 = Xc, yc, s2c
    gp.set_bounds(gp.get_recommended_bounds())
    info = gp.covariance.get_bounds_info(Xc, yc)["x0"]
    h = np.concatenate(
        [
            info,
            gp.noise.get_bounds_info(Xc, yc)["x0"],
            gp.mean.get_bounds_info(Xc, yc)["x0"],
        ]
    )
    h[D + 1] = np.log(0.05)
    h += 0.2 * rng.standard_normal(h.size)
    _, gl = gp.log_likelihood(h, compute_grad=True)
    fdl = fd_grad(lambda hh: gp.log_likelihood(hh), h, 1e-4)
    print(f"  noise={nf}: lik per-comp err={comp_err(gl, fdl):.1e}")
