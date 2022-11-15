import os

import gpyreg as gpr
import numpy as np
import scipy.stats as sps

from pyvbmc.acquisition_functions import AcqFcnVIQR
from pyvbmc.acquisition_functions.utilities import string_to_acq
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc import active_importance_sampling
from pyvbmc.vbmc.options import Options


def test_acq_info():
    acqf = AcqFcnVIQR()
    assert acqf.acq_info["importance_sampling"]
    assert not acqf.acq_info["importance_sampling_vp"]
    assert acqf.acq_info["variational_importance_sampling"]
    assert acqf.acq_info["log_flag"]
    assert np.isclose(sps.norm.cdf(acqf.u), 0.75)

    # Test handling of string input for SearchAcqFcn:
    acqf2 = string_to_acq("AcqFcnVIQR")
    acqf3 = string_to_acq("AcqFcnVIQR()")
    assert acqf.u == acqf2.u == acqf3.u

    acqf4 = AcqFcnVIQR(quantile=0.666)
    acqf5 = string_to_acq("AcqFcnVIQR(quantile=0.666)")
    acqf6 = string_to_acq("AcqFcnVIQR(0.666)")
    assert acqf4.u == acqf5.u == acqf6.u
    assert np.isclose(sps.norm.cdf(acqf4.u), 0.666)


def test_simple__call__():
    D = 2
    epsilon = 1e-6

    def ltarget(theta):  # Standard MVN with s2 est. propto dist. from origin
        return (
            sps.multivariate_normal(mean=np.zeros((D,)), cov=np.eye(D)).logpdf(
                theta
            )
            + np.sqrt(np.linalg.norm(theta)) * np.random.normal(),
            np.linalg.norm(theta) + epsilon,
        )

    # Train GP on "dummy" points far away from I.S. points.
    # This way the predictive variance at I.S. points will approximate the
    # GP prior variance.
    X = np.array([[50.0, -50.0], [-50.0, 50.0]])
    lls = np.array([ltarget(x) for x in X])
    y = lls[:, 0].reshape(-1, 1)
    s2 = lls[:, 1].reshape(-1, 1)

    # Choose a GP with a short lengthscale,
    hyp = np.array(
        [
            [
                # Covariance
                -4.0,
                -4.0,  # log ell
                1.0,  # log sf2
                # Noise
                0.0,  # log std. dev. of noise
                # Mean
                -(D / 2) * np.log(2 * np.pi),  # MVN mode
                0.0,
                0.0,  # Mode location
                0.0,
                0.0,  # log scale
            ]
        ]
    )
    gp = gpr.GP(
        D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=True
        ),
    )
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp)
    # gp.plot(lb=np.array([-5.0, -5.0]), ub=np.array([5.0, 5.0]))

    # Acquisition function evaluation point:
    X_eval = np.zeros((1, D))

    vp = VariationalPosterior(D, 1)  # VP with one component
    vp.mu = np.zeros((D, 1))
    vp.sigma = np.ones((1, 1))  # VP is standard normal

    ## Setup acquisition function and necessary preliminaries:

    acqviqr = AcqFcnVIQR()
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

    # load basic and advanced options and validate the names
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
        user_options={"active_importance_sampling_mcmc_samples": 100},
    )
    advanced_path = pyvbmc_path + "/option_configs/advanced_vbmc_options.ini"
    vbmc_options.load_options_file(
        advanced_path,
        evaluation_parameters={"D": D},
    )
    vbmc_options.validate_option_names([basic_path, advanced_path])

    optim_state["active_importance_sampling"] = active_importance_sampling(
        vp, gp, acqviqr, vbmc_options
    )

    # Test VIQR Acquisition Function Values:
    # Should be close to log(sinh(0.6745 * e)), because tau^2 approx= 0,
    # so s_pred^2 approx= f_s2. -log(2) correction is due to constant factor.
    result = acqviqr(
        X_eval[0], gp, vp, function_logger=None, optim_state=optim_state
    ) - np.log(2)
    # Re-normalize:
    result += -np.log(optim_state["active_importance_sampling"]["X"].shape[0])
    u = sps.norm.ppf(0.75)
    # print(result)
    # print(np.log(np.sinh(u * np.exp(1))))
    assert np.isclose(np.log(np.sinh(u * np.exp(1))), result, atol=1e-4)


def test_complex__call__():
    D = 2
    epsilon = 1e-3

    def ltarget(theta):  # Standard MVN with s2 est. propto dist. from origin
        ll = sps.multivariate_normal(
            mean=np.zeros((D,)), cov=np.eye(D)
        ).logpdf(theta)
        if theta[0] < 0:
            return ll, epsilon
        else:
            return ll, np.linalg.norm(theta) + epsilon

    # GP training data
    M = 17  # Number of training points = M^2
    x1 = x2 = np.linspace(-5, 5, M)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.vstack([X1.ravel(), X2.ravel()]).T
    # Delete every other point on half of the plane, to create some variation
    # in the expected posterior covariance, but leave the points dense enough
    # that _estimate_observation_noise() is accurate:
    for i in range(len(X), len(X) // 2, -1):
        if i % 2 == 0:
            X = np.delete(X, i, 0)
    # X = np.array([[1.0, 1.0], [1.0, 0.0], [1.0, -1.0]])
    lls = np.array([ltarget(x) for x in X])
    y = lls[:, 0].reshape(-1, 1)
    s2 = lls[:, 1].reshape(-1, 1)

    # Fixed GP hyperparameters
    hyp = np.array(
        [
            [
                # Covariance
                1.0,
                1.0,  # log ell
                1.0,  # log sf2
                # Noise
                -10.0,  # log std. dev. of noise
                # Mean
                -(D / 2) * np.log(2 * np.pi),  # MVN mode
                0.0,
                0.0,  # Mode location
                0.0,
                0.0,  # log scale
            ]
        ]
    )
    gp = gpr.GP(
        D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(user_provided_add=True),
    )
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp)
    # gp.plot(lb=np.array([-5.0, -5.0]), ub=np.array([5.0, 5.0]))

    ## Setup grid approximation of VIQR/IMIQR:

    def s_xsi_new(theta, theta_new):
        __, cov = gp.predict_full(np.vstack([theta, theta_new]))
        c_xsi2_t_tn = np.mean(cov, axis=2)[0, 1]
        # __, cov = gp.predict_full(np.atleast_2d(theta_new))
        # c_xsi2_tn_tn = np.mean(cov, axis=2)[0, 0]
        __, c_xsi2_tn_tn = gp.predict(np.atleast_2d(theta_new))
        c_xsi2_tn_tn = c_xsi2_tn_tn[0, 0]
        __, s_xsi2 = gp.predict(np.atleast_2d(theta))
        s_xsi2 = s_xsi2[0, 0]
        ret = s_xsi2 - c_xsi2_t_tn**2 / (
            c_xsi2_tn_tn + np.linalg.norm(theta_new) + epsilon
        )
        return np.sqrt(max(ret, 0.0))

    u = sps.norm.ppf(0.75)
    vp = VariationalPosterior(D, 1)  # VP with one component
    vp.mu = np.zeros((D, 1))
    vp.sigma = np.ones((1, 1))  # VP is standard normal

    def viqr_integrand(theta, theta_new):
        return 2 * vp.pdf(theta) * np.sinh(u * s_xsi_new(theta, theta_new))

    M = 60
    t1 = t2 = np.linspace(-30, 30, M)
    T1, T2 = np.meshgrid(t1, t2)
    thetas = np.vstack([T1.ravel(), T2.ravel()]).T

    # Acquisition function evaluation points:
    N_eval = 5
    X_eval = np.arange(-5, N_eval * D - 5).reshape(N_eval, D)
    X_eval = np.tile(np.linspace(-5, 5, N_eval).reshape((N_eval, 1)), (1, 2))

    # VIQR (IMIQR) values by grid approximationL
    viqrs = np.zeros((N_eval, M**2))
    for i in range(N_eval):
        x = X_eval[i, :]
        v_int = np.array(
            [viqr_integrand(theta, np.atleast_2d(x)) for theta in thetas]
        )
        viqrs[i, :] = v_int.reshape((M**2,))
    # Rough approximation for missing tails of grid:
    corrections = np.array(
        [
            sps.multivariate_normal.pdf(theta, mean=np.zeros((D,)))
            for theta in thetas
        ]
    )
    correction = np.sum(corrections * (60 / M) ** 2)
    viqr_grid = (
        np.sum(viqrs * (60 / M) ** 2, axis=1)  # Grid approx. of expectation
        / correction
    )

    ## Setup acquisition function and necessary preliminaries:

    acqviqr = AcqFcnVIQR()
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

    # load basic and advanced options and validate the names
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
        user_options={"active_importance_sampling_mcmc_samples": 8000},
    )
    advanced_path = pyvbmc_path + "/option_configs/advanced_vbmc_options.ini"
    vbmc_options.load_options_file(
        advanced_path,
        evaluation_parameters={"D": D},
    )
    vbmc_options.validate_option_names([basic_path, advanced_path])

    optim_state["active_importance_sampling"] = active_importance_sampling(
        vp, gp, acqviqr, vbmc_options
    )

    # VIQR Acquisition Function Values:
    log_result = acqviqr(
        X_eval, gp, vp, function_logger=None, optim_state=optim_state
    )
    # Re-normalize:
    log_result += -np.log(
        optim_state["active_importance_sampling"]["X"].shape[0]
    )
    result = np.exp(log_result).reshape((N_eval,))
    # print(result)
    # print(viqr_grid)
    assert np.allclose(viqr_grid, result, rtol=0.03)
    bias = np.mean(result - viqr_grid)
    assert np.allclose(viqr_grid + bias, result, rtol=0.01)
