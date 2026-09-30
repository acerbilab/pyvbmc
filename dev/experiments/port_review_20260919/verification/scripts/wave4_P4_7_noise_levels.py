"""P4-7/8/9/10: the observation noise inside and around the slice.

Settles, at uncertainty-handling levels 0, 1 and 2:

* what ``gp.predict(x, add_noise=True)`` adds at a test point with no
  ``s2_star`` -- the density the IMIQR/VIQR MCMC target
  (``is_log_full``) is built from;
* that every other GP call of the slice takes the latent variance;
* what ``S**2 * n_evals`` (``active_sample.py:337-339``) is at a
  repeated input whose supplied SDs differ;
* the hard bounds gpyreg installs on ``noise_provided_log_multiplier``
  and whether the multiplier scales the variance or the SD;
* the ``sn2_new`` trace of ``active_sample.py:326-351`` and the
  ``s2new`` handed to ``GP.update`` at ``:710-811``.
"""

import numpy as np
from wave4_P4_common import banner

banner()

import gpyreg as gpr  # noqa: E402

from pyvbmc.acquisition_functions import AcqFcnIMIQR, AcqFcnVIQR  # noqa: E402
from pyvbmc.function_logger import FunctionLogger  # noqa: E402
from pyvbmc.parameter_transformer import ParameterTransformer  # noqa: E402

D = 2
rng = np.random.default_rng(0)
X = rng.standard_normal((10, D))
y = (-0.5 * np.sum(X**2, 1)).reshape(-1, 1)
mean_hyp = [0.3] + [0.0] * D + [0.0] * D
cov_hyp = [0.0, 0.0, np.log(2.0)]


def noise_fun(gp_noise_fun):
    """Exactly what train_gp:97-106 builds from optim_state['gp_noise_fun']."""
    return gpr.noise_functions.GaussianNoise(
        constant_add=gp_noise_fun[0] == 1,
        user_provided_add=gp_noise_fun[1] > 0,
        scale_user_provided=gp_noise_fun[1] == 2,
        rectified_linear_output_dependent_add=gp_noise_fun[2] == 1,
    )


LEVELS = {
    0: ([1, 0, 0], None, cov_hyp + [np.log(0.1)] + mean_hyp),
    1: (
        [1, 2, 0],
        np.ones((10, 1)),
        cov_hyp + [np.log(np.sqrt(1e-5)), np.log(3.0)] + mean_hyp,
    ),
    2: (
        [1, 1, 0],
        np.full((10, 1), 0.4),
        cov_hyp + [np.log(np.sqrt(1e-5))] + mean_hyp,
    ),
}

Xt = rng.standard_normal((4, D))
print("--- (c) what add_noise=True adds at a test point with no s2_star ---")
for lvl, (params, s2, hyp) in LEVELS.items():
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=noise_fun(params),
    )
    h = np.array(hyp).reshape(1, -1)
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=h, compute_posterior=True)
    _, f_s2 = gp.predict(Xt)
    _, y_s2 = gp.predict(Xt, add_noise=True)
    added = (y_s2 - f_s2).ravel()
    noise_N = gp.noise.hyperparameter_count()
    hyp_noise = gp.posteriors[0].hyp[3 : 3 + noise_N]
    per_obs = gp.noise.compute(
        hyp_noise, X[:1], y[:1], None if s2 is None else s2[:1]
    )
    print(
        f"  level {lvl}: noise_N={noise_N}  add_noise adds {added[0]:.6g} "
        f"(same at every test point: {np.allclose(added, added[0])}); "
        f"variance of one training observation = {np.ravel(per_obs)[0]:.6g}"
    )
    # everything else in the slice
    acqI, acqV = AcqFcnIMIQR(), AcqFcnVIQR()
    x1 = Xt[:1]
    full = acqI.is_log_full(x1, gp=gp)
    fm, fs2 = gp.predict(x1, separate_samples=True)
    latent = acqI.is_log_base(x1, f_mu=fm, f_s2=fs2) + acqI.is_log_added(
        f_mu=fm, f_s2=fs2
    )
    print(
        f"           IMIQR is_log_full(gp=) = {np.ravel(full)[0]:.8f}; "
        f"same from the latent f_s2 = {np.ravel(latent)[0]:.8f}; "
        f"difference = {np.ravel(full)[0] - np.ravel(latent)[0]:.3e}"
    )

print()
print("--- (a)/(b) the logger trace at levels 1 and 2 ---")
pt = ParameterTransformer(D=D)


def make_logger(noise_flag):
    return FunctionLogger(
        fun=lambda x: (0.0, 1.0),
        D=D,
        noise_flag=noise_flag,
        uncertainty_handling_level=2 if noise_flag else 0,
    )


# level 1: the logger records fsd = 1 on a first evaluation
fl1 = FunctionLogger(
    fun=lambda x: -0.5 * np.sum(x**2),
    D=D,
    noise_flag=True,
    uncertainty_handling_level=1,
)
x0 = np.zeros(D)
for k in range(4):
    fl1(x0)
    print(
        f"  level 1, after {k + 1} evals: S = {fl1.S[0, 0]:.8f}, "
        f"n_evals = {fl1.n_evals[0, 0]}, "
        f"S**2 * n_evals = {fl1.S[0, 0] ** 2 * fl1.n_evals[0, 0]:.10f}"
    )

# level 2 with unequal supplied SDs at one input
sds = [1.0, 2.0, 4.0]
it = iter(sds)
fl2 = FunctionLogger(
    fun=lambda x: (-0.5 * np.sum(x**2), next(it)),
    D=D,
    noise_flag=True,
    uncertainty_handling_level=2,
)
for k, sd in enumerate(sds):
    fl2(x0)
    v = fl2.S[0, 0] ** 2 * fl2.n_evals[0, 0]
    print(
        f"  level 2, after {k + 1} evals (SDs {sds[: k + 1]}): "
        f"S**2 * n_evals = {v:.6f}"
    )
vs = np.array(sds) ** 2
print(
    f"    harmonic mean of {list(vs)} = {len(vs) / np.sum(1 / vs):.6f}; "
    f"arithmetic mean = {vs.mean():.6f}; last observation = {vs[-1]:.6f}"
)

print()
print("--- (9) the noise multiplier's bounds and what it scales ---")
noise = noise_fun([1, 2, 0])
info = noise.get_bounds_info(X, y)
print("  gpyreg LB/UB/PLB/PUB/x0 for the two noise hyperparameters:")
for k in ("LB", "UB", "PLB", "PUB", "x0"):
    print(f"    {k}: {np.asarray(info[k])}")
print(
    "  multiplier bounds in variance terms:",
    (np.exp(info["LB"][1]), np.exp(info["UB"][1])),
)
print("  in SD terms:", (np.exp(info["LB"][1] / 2), np.exp(info["UB"][1] / 2)))
h1 = np.array([np.log(np.sqrt(1e-5)), np.log(7.0)])
print(
    "  compute(h=[.., log 7], s2=1) =",
    np.ravel(noise.compute(h1, X[:1], y[:1], np.array([[1.0]])))[0],
    "-> the multiplier multiplies the VARIANCE s2",
)
