"""The scenario shared by the VIQR and IMIQR definition tests, and the
``optim_state`` and options set-up that ``active_sample`` performs before
an importance-sampling acquisition is evaluated."""

import os

import gpyreg as gpr
import numpy as np
import scipy.stats as sps

from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.options import Options


def prepare_optim_state(gp, s2):
    """What ``active_sample`` stores before the acquisition is evaluated."""
    D = gp.D
    optim_state = {
        "lb_eps_orig": -np.inf,
        "ub_eps_orig": np.inf,
    }
    Ns_gp = len(gp.posteriors)
    ln_ell = np.zeros((D, Ns_gp))
    for s in range(Ns_gp):
        ln_ell[:, s] = gp.posteriors[s].hyp[:D]
    optim_state["gp_length_scale"] = np.exp(ln_ell.mean(1))
    gp.temporary_data["X_rescaled"] = gp.X / optim_state["gp_length_scale"]
    sn2new = np.zeros((gp.X.shape[0], Ns_gp))

    cov_N = gp.covariance.hyperparameter_count(gp.D)
    noise_N = gp.noise.hyperparameter_count()
    for s in range(Ns_gp):
        hyp_noise = gp.posteriors[s].hyp[cov_N : cov_N + noise_N]
        sn2new[:, s] = gp.noise.compute(hyp_noise, gp.X, gp.y, s2).reshape(
            -1,
        )
    gp.temporary_data["sn2_new"] = sn2new.mean(1)
    return optim_state


def options(D, mcmc_samples):
    """The default options with the MCMC sample count of the importance
    sampler overridden, names validated."""
    pyvbmc_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "vbmc",
        )
    )
    basic_path = pyvbmc_path + "/option_configs/basic_vbmc_options.ini"
    vbmc_options = Options(
        basic_path,
        evaluation_parameters={"D": D},
        user_options={"active_importance_sampling_mcmc_samples": mcmc_samples},
    )
    advanced_path = pyvbmc_path + "/option_configs/advanced_vbmc_options.ini"
    vbmc_options.load_options_file(
        advanced_path,
        evaluation_parameters={"D": D},
    )
    vbmc_options.validate_option_names([basic_path, advanced_path])
    return vbmc_options


def look_ahead_scenario(seed=0):
    """A standard-normal target in ``D = 2`` observed, without noise in the
    values, on a 17 x 17 grid on [-5, 5]^2 with every other point dropped
    on half of the plane, the observation-noise variance being 1e-3 where
    ``x_1 < 0`` and ``||x|| + 1e-3`` elsewhere; a GP with fixed
    hyperparameters (length scales e, signal sd e^0.5, a constant noise sd
    of e^-10 on top of the user-provided variances) whose negative-quadratic
    mean *is* the target's log density, so that the posterior mean equals
    the target everywhere and the density IMIQR integrates against,
    exp(f_mu), is the VP; a standard-normal single-component VP whose
    generator is seeded; five candidates on the diagonal of [-5, 5]^2.

    Returns ``gp, s2, vp, X_eval, u`` with ``u`` the z-score of the 0.75
    quantile. The predictive sd is small on the low-noise half-plane and
    of order one on the other, which is what makes the importance
    weights of IMIQR (1/sinh of it) uneven.
    """
    D = 2
    epsilon = 1e-3

    def ltarget(theta):
        ll = sps.multivariate_normal(
            mean=np.zeros((D,)), cov=np.eye(D)
        ).logpdf(theta)
        if theta[0] < 0:
            return ll, epsilon
        return ll, np.linalg.norm(theta) + epsilon

    M = 17
    x1 = x2 = np.linspace(-5, 5, M)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.vstack([X1.ravel(), X2.ravel()]).T
    for i in range(len(X), len(X) // 2, -1):
        if i % 2 == 0:
            X = np.delete(X, i, 0)
    lls = np.array([ltarget(x) for x in X])
    y = lls[:, 0].reshape(-1, 1)
    s2 = lls[:, 1].reshape(-1, 1)

    hyp = np.array(
        [
            [
                # Covariance
                1.0,
                1.0,  # log ell
                1.0,  # log sf2
                # Noise
                -10.0,  # log std. dev. of the constant noise
                # Mean
                -(D / 2) * np.log(2 * np.pi),  # MVN mode
                0.0,
                0.0,  # Mode location
                0.0,
                0.0,  # log scale
            ]
        ]
    )
    # The noise function must carry the hyperparameter the array holds:
    # without ``constant_add`` the mean function reads shifted entries and
    # is not the target (the case until 2026-09-10).
    gp = gpr.GP(
        D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=True
        ),
    )
    assert hyp.shape[1] == (
        gp.covariance.hyperparameter_count(D)
        + gp.noise.hyperparameter_count()
        + gp.mean.hyperparameter_count(D)
    )
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp)

    vp = VariationalPosterior(D, 1)
    vp.mu = np.zeros((D, 1))
    vp.sigma = np.ones((1, 1))
    vp.rng = np.random.default_rng(seed)

    N_eval = 5
    X_eval = np.tile(np.linspace(-5, 5, N_eval).reshape((N_eval, 1)), (1, 2))
    u = sps.norm.ppf(0.75)
    return gp, s2, vp, X_eval, u
