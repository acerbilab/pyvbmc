"""Wave 3, A-1: the GP noise function at uncertainty level 1.

Settles whether a run with ``uncertainty_handling=True`` and no
``specify_target_noise`` (level 1, "infer noise") gets the noise model that
``optim_state["gp_noise_fun"] = [1, 2, 0]`` names, which in MATLAB
(`gplite/gplite_noisefun.m:60-71`, `:177-194`) is a constant term plus a
scaled provided term, ``exp(2*h1) + exp(h2)*s2``, with two hyperparameters.

Part 1 builds the noise function the way `train_gp` does for the three
levels and beside it the one MATLAB's flags name. Part 2 runs `train_gp`
once on a small level-1 state built through `VBMC` and reports the fitted
model. No `optimize()` run.
"""

import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import train_gp

print("pyvbmc:", pyvbmc.__file__, flush=True)
print("gpyreg:", gpr.__file__, flush=True)


def as_train_gp(gp_noise_fun):
    """The translation of `gaussian_process_train.py:96-105`."""
    return gpr.noise_functions.GaussianNoise(
        constant_add=gp_noise_fun[0] == 1,
        user_provided_add=gp_noise_fun[1] == 1,
        scale_user_provided=gp_noise_fun[1] == 2,
        rectified_linear_output_dependent_add=gp_noise_fun[2] == 1,
    )


print("\n== Part 1: the noise function per level ==", flush=True)
X = np.zeros((3, 2))
y = np.array([[0.0], [1.0], [2.0]])
s2 = np.array([[1.0], [0.5], [0.25]])  # 1/n_evals for 1, 2, 4 evaluations
for level, flags in ((0, [1, 0, 0]), (1, [1, 2, 0]), (2, [1, 1, 0])):
    noise = as_train_gp(flags)
    n_hyp = noise.hyperparameter_count()
    hyp = np.array([np.log(0.1)] + [0.0] * (n_hyp - 1))
    sn2 = np.ravel(noise.compute(hyp, X, y, s2))
    print(
        f"level {level}: gp_noise_fun={flags} -> parameters="
        f"{noise.parameters.tolist()} names="
        f"{[n for n, _ in noise.hyperparameter_info()]} sn2={sn2}",
        flush=True,
    )

matlab_level1 = gpr.noise_functions.GaussianNoise(
    constant_add=True, user_provided_add=True, scale_user_provided=True
)
hyp = np.array([np.log(0.1), 0.0])
print(
    "what [1 2] names (MATLAB): parameters="
    f"{matlab_level1.parameters.tolist()} names="
    f"{[n for n, _ in matlab_level1.hyperparameter_info()]} sn2="
    f"{np.ravel(matlab_level1.compute(hyp, X, y, s2))}",
    flush=True,
)

print("\n== Part 2: train_gp on a level-1 state ==", flush=True)
D = 2
rng = np.random.default_rng(3)


def noisy_target(x):
    x = np.atleast_2d(x)
    return float(-0.5 * np.sum(x**2) + rng.normal())


vbmc = VBMC(
    noisy_target,
    np.zeros((1, D)),
    np.full((1, D), -10.0),
    np.full((1, D), 10.0),
    np.full((1, D), -3.0),
    np.full((1, D), 3.0),
    options={"uncertainty_handling": True},
    seed=11,
)
print(
    "uncertainty_handling_level:",
    vbmc.optim_state["uncertainty_handling_level"],
    " gp_noise_fun:",
    vbmc.optim_state["gp_noise_fun"],
    " logger.noise_flag:",
    vbmc.function_logger.noise_flag,
    flush=True,
)

# Twelve evaluations, three of them repeats of the first point, through
# the logger, as the initial design and active sampling make them.
pts = vbmc.parameter_transformer(rng.uniform(-3, 3, size=(9, D)))
for p in pts:
    vbmc.function_logger(p)
for _ in range(3):
    vbmc.function_logger(pts[0])
fl = vbmc.function_logger
print("n_evals:", np.ravel(fl.n_evals[fl.X_flag]), flush=True)
print("S (pooled SD):", np.round(np.ravel(fl.S[fl.X_flag]), 4), flush=True)

vbmc.optim_state["N"] = fl.Xn + 1
vbmc.optim_state["n_eff"] = np.sum(fl.n_evals[fl.X_flag])
gp, gp_s_N, sn2_hpd, hyp_dict = train_gp(
    {},
    vbmc.optim_state,
    fl,
    vbmc.iteration_history,
    vbmc.options,
    vbmc.optim_state["plb_tran"],
    vbmc.optim_state["pub_tran"],
    rng=5,
)
print("gp.noise.parameters:", gp.noise.parameters.tolist(), flush=True)
print(
    "noise hyperparameters:",
    [n for n, _ in gp.noise.hyperparameter_info()],
    flush=True,
)
n_cov = gp.covariance.hyperparameter_count(D)
n_noise = gp.noise.hyperparameter_count()
n_mean = gp.mean.hyperparameter_count(D)
print(
    f"hyperparameters: cov {n_cov} + noise {n_noise} + mean {n_mean} = "
    f"{n_cov + n_noise + n_mean}; MATLAB's [1 2] gives noise 2, total "
    f"{n_cov + 2 + n_mean}",
    flush=True,
)
print("s2 handed to the fit:", np.round(np.ravel(gp.s2), 4), flush=True)
priors = gp.get_priors()
print("prior names on the GP:", sorted(priors), flush=True)
print("noise_log_scale prior:", priors["noise_log_scale"], flush=True)
post = gp.posteriors[0]
sn2_rows = np.ravel(
    gp.noise.compute(post.hyp[n_cov : n_cov + n_noise], gp.X, gp.y, gp.s2)
)
print(
    "fitted noise variance per training row (one value = homoskedastic):",
    sn2_rows,
    flush=True,
)
