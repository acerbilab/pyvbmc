"""P5-1 and P5-6: what bounds `_gp_hyp` actually installs on the GP.

Settles:
  * P5-1 (P5i F1 / P5c F2): whether the upper bound built from
    `upper_gp_length_factor` survives to `gp.upper_bounds`, or whether the
    SquaredExponential block that follows replaces the whole tuple with
    `(hpd LB, nan)`.
  * P5-6 (P5i F6 / P5c F3, noise and length-scale parts): whether
    `noise_log_scale`'s upper bound is `+inf` where gpyreg would recommend
    `log(max(y) - min(y))`, and whether the `-inf` lower length-scale bound
    written at line 370 survives the SquaredExponential block.
  * P5-7 (P5c F4): the hpd-derived covariance lower bounds against the
    full-training-set ones gpyreg would recommend.

Everything is one `_gp_hyp` call on a synthetic 12-point state; no fit.
"""

import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import _get_training_data, _gp_hyp

print("pyvbmc.__file__ =", pyvbmc.__file__)
print("gpyreg.__file__ =", gpr.__file__)

D = 2
N = 12
f = lambda x: -np.sum((np.atleast_2d(x) + 1.0) ** 2, axis=1)
x0 = np.zeros((1, D))
plb = np.full((1, D), -1.0)
pub = np.full((1, D), 1.0)


def build(options=None):
    vbmc = VBMC(f, x0, None, None, plb, pub, options or {})
    rng = np.random.default_rng(0)
    window = vbmc.optim_state["pub_tran"] - vbmc.optim_state["plb_tran"]
    Xs = window * rng.random((N, D)) + vbmc.optim_state["plb_tran"]
    ys = f(Xs)
    for i in range(N):
        vbmc.function_logger.X_flag[i] = True
        vbmc.function_logger.X[i] = Xs[i]
        vbmc.function_logger.y[i] = ys[i]
        if vbmc.function_logger.noise_flag:
            vbmc.function_logger.S[i] = 1
        vbmc.function_logger.fun_eval_time[i] = 1e-5
    vbmc.optim_state["N"] = N
    vbmc.optim_state["n_eff"] = N
    return vbmc


def run(factor):
    vbmc = build({"upper_gp_length_factor": factor} if factor else {})
    X, y, s2, _ = _get_training_data(vbmc.function_logger)
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp, hyp0, gp_s_N = _gp_hyp(
        vbmc.optim_state,
        vbmc.options,
        vbmc.optim_state["plb_tran"],
        vbmc.optim_state["pub_tran"],
        gp,
        X,
        y,
    )
    gp.X, gp.y = X, y
    installed = gp.get_bounds()
    # What gpyreg fills the nan entries with, on the FULL training set,
    # exactly as `GP.fit` does (`lower_bounds`/`upper_bounds` = "current").
    filled = gp.get_recommended_bounds(gp.lower_bounds, gp.upper_bounds)
    recommended = gp.get_recommended_bounds("recommended", "recommended")
    print(f"\n--- upper_gp_length_factor = {factor} ---")
    width = pub - plb
    if factor:
        print(
            "  option asks for length-scale UB =",
            np.log(
                factor
                * (vbmc.optim_state["pub_tran"] - vbmc.optim_state["plb_tran"])
            ).ravel(),
        )
    for key in (
        "covariance_log_lengthscale",
        "covariance_log_outputscale",
        "noise_log_scale",
        "mean_const",
    ):
        lo, hi = installed[key]
        flo, fhi = filled[key]
        rlo, rhi = recommended[key]
        print(f"  {key}")
        print(f"    installed  LB={np.atleast_1d(lo)}  UB={np.atleast_1d(hi)}")
        print(
            f"    after fill LB={np.atleast_1d(flo)}  UB={np.atleast_1d(fhi)}"
        )
        print(
            f"    gpyreg rec LB={np.atleast_1d(rlo)}  UB={np.atleast_1d(rhi)}"
        )
    print(
        "  y range: min=%.6f max=%.6f log(height)=%.6f"
        % (np.min(y), np.max(y), np.log(np.max(y) - np.min(y)))
    )


run(0)
run(3)
