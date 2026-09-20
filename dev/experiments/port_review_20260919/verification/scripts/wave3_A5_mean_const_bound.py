"""Wave 3, A-5: the lower bound of `mean_const` and the space-filling design
of the hyperparameter fit; and, for A-6, the reviewer's 3-by-3 example.

MATLAB fills `LB_gp` and `UB_gp` with NaN and assigns single entries
(`misc/gptrain_vbmc.m:174-188`); `gplite_train.m:120-127` replaces what is
still NaN with the recommendation from the full training set, `min(y)` for
the constant of the negative quadratic mean (`gplite_meanfun.m:182`).
`_gp_hyp` writes the pair `(-inf, max(y_hpd) + delta_y)` for `mean_const`
(`gaussian_process_train.py:389`), and gpyreg fills only NaN
(`gaussian_process.py:1163-1175`), so the lower bound stays `-inf`.
`mean_const` has no hyperprior, and `f_min_fill.py:119-139` draws such a
coordinate from the mixture of uniforms over `[LB, PLB, PUB, UB]` only when
both bounds are finite, and from the plausible box alone otherwise
(`gplite/private/fminfill.m:73-82` reads the same).

Part 1 shows the installed bounds on a small state. Part 2 builds the design
of one fit twice, with the lower bound as PyVBMC installs it and as MATLAB's
convention would give it, and compares the `mean_const` column.
"""

import gpyreg as gpr
import numpy as np
from gpyreg.f_min_fill import f_min_fill

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import _get_training_data, _gp_hyp

print("pyvbmc:", pyvbmc.__file__, flush=True)
print("gpyreg:", gpr.__file__, flush=True)

D = 2
rng = np.random.default_rng(1)


def target(x):
    x = np.atleast_2d(x)
    return float(-0.5 * np.sum(x**2 / np.array([1.0, 4.0])))


vbmc = VBMC(
    target,
    np.zeros((1, D)),
    np.full((1, D), -10.0),
    np.full((1, D), 10.0),
    np.full((1, D), -3.0),
    np.full((1, D), 3.0),
    seed=2,
)
for p in vbmc.parameter_transformer(rng.uniform(-3, 3, size=(20, D))):
    vbmc.function_logger(p)
fl = vbmc.function_logger
vbmc.optim_state["N"] = fl.Xn + 1
vbmc.optim_state["n_eff"] = np.sum(fl.n_evals[fl.X_flag])
X, y, s2, _ = _get_training_data(fl)


def fresh_gp():
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp, hyp0, _ = _gp_hyp(
        vbmc.optim_state,
        vbmc.options,
        vbmc.optim_state["plb_tran"],
        vbmc.optim_state["pub_tran"],
        gp,
        X,
        y,
    )
    gp.X, gp.y = X, y
    return gp, hyp0


print("\n== Part 1: bounds after _gp_hyp ==", flush=True)
gp, hyp0 = fresh_gp()
b = gp.get_bounds()
for key in ("noise_log_scale", "mean_const", "covariance_log_lengthscale"):
    print(key, "->", b[key], flush=True)
rec = gp.get_recommended_bounds("recommended", "recommended")
print(
    "recommended (what a NaN entry becomes): noise_log_scale",
    rec["noise_log_scale"],
    " mean_const",
    rec["mean_const"],
    flush=True,
)
print("min(y) =", float(np.min(y)), " max(y) =", float(np.max(y)), flush=True)

print("\n== Part 2: the mean_const column of the design ==", flush=True)
n_cov = gp.covariance.hyperparameter_count(D)
n_noise = gp.noise.hyperparameter_count()
i_m0 = n_cov + n_noise  # mean_const is the first mean hyperparameter


def design(lower_m0):
    gp, hyp0 = fresh_gp()
    bounds = gp.get_bounds()
    bounds["mean_const"] = (lower_m0, bounds["mean_const"][1])
    gp.set_bounds(bounds)
    gp.set_bounds(gp.get_recommended_bounds(gp.lower_bounds, gp.upper_bounds))
    LB, UB = gp.lower_bounds, gp.upper_bounds
    infos = [
        gp.covariance.get_bounds_info(X, y),
        gp.noise.get_bounds_info(X, y),
        gp.mean.get_bounds_info(X, y),
    ]
    PLB = np.concatenate([i["PLB"] for i in infos])
    PUB = np.concatenate([i["PUB"] for i in infos])
    PLB = np.minimum(np.maximum(PLB, LB), UB)
    PUB = np.maximum(np.minimum(PUB, UB), LB)
    gp.hyper_priors["df"][np.isnan(gp.hyper_priors["df"])] = 7
    Xd, _ = f_min_fill(
        lambda h: 0.0,
        np.atleast_2d(hyp0),
        LB,
        UB,
        PLB,
        PUB,
        gp.hyper_priors,
        1024,
        "sobol",
        rng=7,
    )
    col = Xd[1:, i_m0]
    return LB[i_m0], UB[i_m0], PLB[i_m0], PUB[i_m0], col


for label, lower in (("PyVBMC (-inf)", -np.inf), ("MATLAB (NaN)", np.nan)):
    lb, ub, plb, pub, col = design(lower)
    print(
        f"{label}: LB={lb:.3f} PLB={plb:.3f} PUB={pub:.3f} UB={ub:.3f} | "
        f"design min={col.min():.3f} max={col.max():.3f} | below PLB: "
        f"{np.mean(col < plb):.3f}, above PUB: {np.mean(col > pub):.3f}",
        flush=True,
    )

print("\n== A-6: the reviewer's 3-by-3 example ==", flush=True)
C = np.array([[1, 0.72, 0.72], [0.72, 1, 0.04], [0.72, 0.04, 1.0]])
print("eigenvalues before:", np.linalg.eigvalsh(C), flush=True)
Ct = C.copy()
Ct[np.abs(C) <= 0.05] = 0
print("eigenvalues after the threshold:", np.linalg.eigvalsh(Ct), flush=True)
U, s, _ = np.linalg.svd(Ct)
W = np.diag(1 / np.sqrt(s + np.finfo(float).eps)) @ U.T
print("variances attained:", np.diag(W @ C @ W.T), flush=True)
