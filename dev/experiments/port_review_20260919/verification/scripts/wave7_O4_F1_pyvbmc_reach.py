"""O4 F1, reach from PyVBMC: which bound pairs of a GP built by `_gp_hyp`
can come out equal once `fit` fills the NaN bounds with gpyreg's
recommendations, whether each such coordinate carries a prior, and whether
the gradient of the fit objective is then NaN. Degenerate training sets
included (tiny target range, equal targets, a shared input coordinate)."""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpyreg as gpr
import numpy as np
from wave7_O4_common import banner

from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import (
    _cov_identifier_to_covariance_function,
    _gp_hyp,
    _meanfun_name_to_mean_function,
)

banner()
warnings.simplefilter("ignore")


def names_of(gp):
    info = (
        gp.covariance.hyperparameter_info(gp.D)
        + gp.noise.hyperparameter_info()
        + gp.mean.hyperparameter_info(gp.D)
    )
    out = []
    for name, n in info:
        out += [f"{name}[{j}]" if n > 1 else name for j in range(n)]
    return out


def has_prior(gp):
    hp = gp.hyper_priors
    return (
        np.isfinite(hp["mu"]) | np.isfinite(hp["sigma"]) | np.isfinite(hp["a"])
    )


def build(level, meanfun, D):
    if level == 2:
        f = lambda x: (-0.5 * np.sum(x**2), 0.3)
    else:
        f = lambda x: -0.5 * np.sum(x**2)
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
    assert os_["uncertainty_handling_level"] == level
    nf = os_["gp_noise_fun"]
    noise = gpr.noise_functions.GaussianNoise(
        constant_add=nf[0] == 1,
        user_provided_add=nf[1] > 0,
        scale_user_provided=nf[1] == 2,
        rectified_linear_output_dependent_add=nf[2] == 1,
    )
    gp = gpr.GP(
        D=D,
        covariance=_cov_identifier_to_covariance_function(os_["gp_cov_fun"]),
        mean=_meanfun_name_to_mean_function(os_["gp_mean_fun"]),
        noise=noise,
    )
    return vb, gp, nf


def training_set(kind, D, level, rng):
    N = 12
    X = rng.uniform(-1, 1, (N, D))
    y = -0.5 * np.sum(X**2, 1)
    if kind == "tiny range 1e-8":
        y = -3 + 1e-8 * rng.uniform(size=N)
    elif kind == "range 1e-3":
        y = -3 + 1e-3 * rng.uniform(size=N)
    elif kind == "equal targets":
        y = np.full(N, -3.0)
    elif kind == "shared coordinate":
        X[:, 0] = 0.3
        y = -0.5 * np.sum(X**2, 1)
    s2 = None
    if level == 1:
        s2 = 1.0 / rng.integers(1, 4, N)
    elif level == 2:
        s2 = np.full(N, 0.09)
    return X, y.reshape(-1, 1), None if s2 is None else s2.reshape(-1, 1)


KINDS = (
    "regular",
    "tiny range 1e-8",
    "range 1e-3",
    "equal targets",
    "shared coordinate",
)
summary = {}
for kind in KINDS:
    for D in (1, 2):
        for level in (0, 1, 2):
            for meanfun in ("zero", "const", "negquad"):
                rng = np.random.default_rng(100 * D + 10 * level)
                vb, gp, nf = build(level, meanfun, D)
                X, y, s2 = training_set(kind, D, level, rng)
                os_ = vb.optim_state
                os_["N"] = X.shape[0]
                os_["stop_sampling"] = 0
                gp, hyp0, _ = _gp_hyp(
                    os_, vb.options, os_["plb_tran"], os_["pub_tran"], gp, X, y
                )
                Xc, yc, s2c = gp._convert_shapes(X, y, s2)
                gp.X, gp.y, gp.s2 = Xc, yc, s2c
                try:
                    gp.set_bounds(
                        gp.get_recommended_bounds(
                            gp.lower_bounds, gp.upper_bounds
                        )
                    )
                except ValueError as e:
                    print(
                        f"{kind:18s} D={D} L{level} {meanfun:7s}: ValueError {e}"
                    )
                    continue
                lb, ub = gp.lower_bounds, gp.upper_bounds
                fixed = np.where(lb == ub)[0]
                pri = has_prior(gp)
                nm = names_of(gp)
                desc = [
                    f"{nm[i]}=({lb[i]:.3g},{ub[i]:.3g}) prior={bool(pri[i])}"
                    for i in fixed
                ]
                # gradient of the fit objective at a point inside the box
                h = np.where(
                    np.isfinite(lb) & np.isfinite(ub),
                    0.5 * (lb + ub),
                    np.where(
                        np.isfinite(lb),
                        lb + 1.0,
                        np.where(np.isfinite(ub), ub - 1.0, 0.0),
                    ),
                )
                try:
                    _, g = gp._GP__gp_obj_fun(h, True, False)
                    nan_idx = [nm[i] for i in np.where(np.isnan(g))[0]]
                except Exception as e:  # noqa: BLE001
                    nan_idx = f"error {type(e).__name__}"
                key = (kind, bool(len(fixed)))
                line = (
                    f"{kind:18s} D={D} L{level} noise={nf} {meanfun:7s}: "
                    f"equal pairs: {desc if desc else 'none'}; NaN grad: {nan_idx}"
                )
                if desc or nan_idx:
                    print(line)
                summary.setdefault(kind, []).append(bool(desc))
    print(
        f"-- {kind}: configurations with an equal pair: "
        f"{sum(summary.get(kind, []))} of {len(summary.get(kind, []))}",
        flush=True,
    )

print(
    "\n== fit on a shared-coordinate training set, PyVBMC-built GP (L0, negquad, D=2) =="
)
rng = np.random.default_rng(7)
vb, gp, nf = build(0, "negquad", 2)
X, y, s2 = training_set("shared coordinate", 2, 0, rng)
os_ = vb.optim_state
os_["N"] = X.shape[0]
os_["stop_sampling"] = 0
gp, hyp0, _ = _gp_hyp(
    os_, vb.options, os_["plb_tran"], os_["pub_tran"], gp, X, y
)
try:
    out = gp.fit(
        X,
        y,
        None,
        options={"n_samples": 0, "init_N": 64, "opts_N": 1},
        rng=np.random.default_rng(0),
    )
    print("fit returned; optimizer:", out[1].nit, out[1].message)
except Exception as e:  # noqa: BLE001
    print(f"fit raised {type(e).__name__}: {e}")
print("bounds after the fill:")
for n_, a, b in zip(names_of(gp), gp.lower_bounds, gp.upper_bounds):
    print(f"   {n_:32s} ({a:.4g}, {b:.4g})")
