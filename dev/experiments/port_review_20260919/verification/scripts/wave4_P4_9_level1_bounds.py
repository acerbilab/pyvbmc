"""P4-9: the hyperparameter bounds a level-1 fit installs on the noise.

At uncertainty level 1 the recorded variance is the placeholder
``s2 == 1`` and the whole noise magnitude is carried by
``exp(h1) * 1``, whose multiplier ``noise_provided_log_multiplier`` gpyreg
bounds at ``[log 1e-3, log 1e3]``.  This script runs ``_gp_hyp`` as
``train_gp`` does at level 1 and prints the bounds it installs, to see
whether the constant term ``exp(2 h0)`` is free to absorb a noise
magnitude the multiplier cannot reach.
"""

import numpy as np
from wave4_P4_common import banner

banner()

import gpyreg as gpr  # noqa: E402

from pyvbmc.vbmc.gaussian_process_train import _gp_hyp  # noqa: E402
from pyvbmc.vbmc.options import Options  # noqa: E402

D = 2
rng = np.random.default_rng(0)
X = rng.standard_normal((14, D))
y = (-0.5 * np.sum(X**2, 1) + 3 * rng.standard_normal(14)).reshape(-1, 1)

gp = gpr.GP(
    D=D,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.NegativeQuadratic(),
    noise=gpr.noise_functions.GaussianNoise(
        constant_add=True, user_provided_add=True, scale_user_provided=True
    ),
)
gp.X, gp.y, gp.s2 = X, y, np.ones((14, 1))

import os  # noqa: E402

import pyvbmc  # noqa: E402

base = os.path.join(
    os.path.dirname(pyvbmc.__file__),
    "vbmc",
    "option_configs",
)
opts = Options(
    os.path.join(base, "basic_vbmc_options.ini"),
    evaluation_parameters={"D": D},
    user_options={},
)
opts.load_options_file(
    os.path.join(base, "advanced_vbmc_options.ini"),
    evaluation_parameters={"D": D},
)

optim_state = {
    "uncertainty_handling_level": 1,
    "gp_noise_fun": [1, 2, 0],
    "n_eff": 14,
    "iter": 3,
    "warmup": False,
    "delta": np.ones((1, D)) * 1e-6,
    "stop_sampling": 0,
    "N": 14,
    "vp_K": 2,
}
gp2, hyp0, gp_s_N = _gp_hyp(
    optim_state, opts, -np.ones((1, D)), np.ones((1, D)), gp, X, y
)
b = gp2.get_bounds()
print("bounds installed by _gp_hyp at level 1:")
for k in ("noise_log_scale", "noise_provided_log_multiplier"):
    print(f"  {k}: {b[k]}")
rec = gp2.noise.get_bounds_info(X, y)
print(
    "gpyreg's recommendation on the same data:",
    {"LB": rec["LB"], "UB": rec["UB"]},
)
print(
    "  height = max(y) - min(y) =",
    float(np.max(y) - np.min(y)),
    "-> log(height) =",
    float(np.log(np.max(y) - np.min(y))),
)
print()
print("priors:")
p = gp2.get_priors()
for k in ("noise_log_scale", "noise_provided_log_multiplier"):
    print(f"  {k}: {p[k]}")
print()
print(
    "so the total level-1 variance exp(2 h0) + exp(h1) can reach",
    f"{np.exp(2 * float(np.log(np.max(y) - np.min(y)))) + 1e3:.4g}",
    "even though the multiplier alone caps at 1e3 (SD 31.6).",
)
