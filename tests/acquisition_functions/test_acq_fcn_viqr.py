import pytest
import gpyreg as gpr
import numpy as np
import scipy as sp
import scipy.stats as sps
import os

from pyvbmc.acquisition_functions import AcqFcnVIQR
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.acquisition_functions.active_importance_sampling_vbmc import active_importance_sampling_vbmc
from pyvbmc.vbmc.options import Options

def test_acq_info():
    acqf = AcqFcnVIQR()
    assert acqf.importance_sampling
    assert not acqf.importance_sampling_vp
    assert acqf.variational_importance_sampling
    assert acqf.log_flag
    assert np.isclose(sps.norm.cdf(acqf.u), 0.75)


def test__call__():
    D = 2


    def ltarget(theta):  # Standard MVN with s2 est. propto dist. from origin
        return sps.multivariate_normal(
            mean=np.zeros((D,)),
            cov=np.eye(D)
        ).logpdf(theta), 1 * np.linalg.norm(theta)
        # ).logpdf(theta) + np.sqrt(np.linalg.norm(theta)) * np.random.normal(), 1 * np.linalg.norm(theta)

    # Fit a GP to a large number of training points from the target
    M = 21  # Number of training points = M^2
    x1 = x2 = np.linspace(-5, 5, M)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.vstack([X1.ravel(), X2.ravel()]).T
    lls = np.array([ltarget(x) for x in X])
    # lls = np.zeros(lls.shape)
    y = lls[:, 0].reshape(-1, 1)
    s2 = lls[:, 1].reshape(-1, 1)
    # gp.fit(X.reshape(-1, D), y, s2)

    hyp = np.array([[
        # Covariance
        0.0, 0.0,  # log ell
        1.0,  # log sf2
        # Noise
        -10.0,  # log std. dev. of noise
        # Mean
        -(D/2) * np.log(2 * np.pi),  # MVN mode
        0.0, 0.0,  # Mode location
        0.0, 0.0,  # log scale
    ]])
    gp = gpr.GP(
        D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True,
            user_provided_add=True
        ),
    )
    gp.update(
        X_new=X,
        y_new=y,
        s2_new=s2,
        hyp=hyp
    )
    gp.plot(lb=np.array([-10.0, -10.0]), ub=np.array([10.0, 10.0]))

    # Acquisition function evaluation points:
    N_eval = 5
    X_eval = np.arange(-5, N_eval * D - 5).reshape(N_eval, D)
    lls_eval = np.array([ltarget(x) for x in X_eval])
    y_eval = lls_eval[:, 0].reshape(-1, 1)
    s2_eval = lls_eval[:, 0].reshape(-1, 1)
    f_mu, f_s2 = gp.predict(x_star=X_eval, separate_samples=True)

    ## Setup grid approximation of VIQR/IMIQR:

    def s_xsi_new(theta, theta_new):
        __, cov = gp.predict_full(np.vstack([theta, theta_new]))
        c_xsi2_t_tn = np.mean(cov, axis=2)[0, 1]
        # __, cov = gp.predict_full(np.atleast_2d(theta_new))
        # c_xsi2_tn_tn = np.mean(cov, axis=2)[0, 0]
        __, c_xsi2_tn_tn = gp.predict(np.atleast_2d(theta_new))
        c_xsi2_tn_tn = c_xsi2_tn_tn[0,0]
        __, s_xsi2 = gp.predict(np.atleast_2d(theta), add_noise=True)
        s_xsi2 = s_xsi2[0,0]
        # print(c_xsi2_t_tn / (c_xsi2_tn_tn + np.linalg.norm(theta_new)))
        s_xsi2 = np.linalg.norm(theta)
        c_xsi2_tn_tn = np.linalg.norm(theta_new)
        # print(s_xsi2 - c_xsi2_t_tn / (c_xsi2_tn_tn + np.linalg.norm(theta_new)))
        # print(np.linalg.norm(theta_new))
        return s_xsi2 - c_xsi2_t_tn / (c_xsi2_tn_tn + np.linalg.norm(theta_new))

    u = sps.norm.ppf(0.75)
    vp = VariationalPosterior(D, 1)  # VP with one component
    vp.mu = np.zeros((D, 1))
    vp.sigma = np.ones((1, 1))  # VP is standard normal
    def viqr_integrand(theta, theta_new):
        return vp.pdf(theta) * np.sinh(u * s_xsi_new(theta, theta_new))
    M = 10
    t1 = t2 = np.linspace(-20, 20, M)
    T1, T2 = np.meshgrid(t1, t2)
    thetas = np.vstack([T1.ravel(), T2.ravel()]).T

    # VIQR (IMIQR) values by grid approximationL
    viqrs = np.array([viqr_integrand(theta, X_eval[0]) for theta in thetas])
    print(sum(viqrs) * (40/M)**2)

    ## Setup acquisition function and necessary preliminaries:

    acqviqr = AcqFcnVIQR()
    optim_state = {
        "lb_eps_orig" : -np.inf,
        "ub_eps_orig" : np.inf,
    }
    Ns_gp = len(gp.posteriors)
    ln_ell = np.zeros((D, Ns_gp))
    for s in range(Ns_gp):
        ln_ell[:, s] = gp.posteriors[s].hyp[:D]
    optim_state["gp_length_scale"] = np.exp(ln_ell.mean(1))
    gp.temporary_data["X_rescaled"] = (
        gp.X / optim_state["gp_length_scale"]
    )
    sn2new = np.zeros((gp.X.shape[0], Ns_gp))

    cov_N = gp.covariance.hyperparameter_count(gp.D)
    noise_N = gp.noise.hyperparameter_count()
    for s in range(Ns_gp):
        hyp_noise = gp.posteriors[s].hyp[cov_N : cov_N + noise_N]
        sn2new[:, s] = gp.noise.compute(hyp_noise, gp.X, gp.y, s2).reshape(-1,)
    gp.temporary_data["sn2_new"] = sn2new.mean(1)

    # load basic and advanced options and validate the names
    pyvbmc_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'pyvbmc', 'vbmc'))
    basic_path = pyvbmc_path + "/option_configs/basic_vbmc_options.ini"
    vbmc_options = Options(
        basic_path,
        evaluation_parameters={"D": D},
        user_options={"activeimportancesamplingmcmcsamples" : 100}
    )
    advanced_path = (
        pyvbmc_path + "/option_configs/advanced_vbmc_options.ini"
    )
    vbmc_options.load_options_file(
        advanced_path,
        evaluation_parameters={"D": D},
    )
    vbmc_options.validate_option_names([basic_path, advanced_path])

    optim_state["active_importance_sampling"] = active_importance_sampling_vbmc(vp, gp, acqviqr, vbmc_options)

    # VIQR Acquisition Function Values:
    print(np.exp(acqviqr(X_eval[0], gp, vp, function_logger=None, optim_state=optim_state)))
    fmu, fs2 = gp.predict(x_star=np.array([[8.0, 8.0]]))
    print(fs2)

@pytest.mark.skip
def test__call__2():
    D=3
    gp = gpr.GP(D,
                covariance=gpr.covariance_functions.SquaredExponential(),
                mean=gpr.mean_functions.NegativeQuadratic(),
                noise=gpr.noise_functions.GaussianNoise(constant_add=True)
            )
    N=2
    gp.X = np.array([[-1.0, 0.0, 1.0],[2.0, 1.0, -3.0]])
    gp.y = np.array([[1.0],[-1.5]])
    hyp=np.array([
        2.0, 2.0, 2.0,  # log ell, cov
        1.5, # log sf
        1.8,  # log sn, noise
        1.1, # m0, mean
        1.2, 1.3, 1.4, # x_m
        -1.5, 0.0, 1.5 # log omega
                ])
    gp.posteriors = np.array([gpr.gaussian_process.Posterior(
        hyp=hyp,
        alpha=np.array([[1.0, 2.0]]).T,
        sW=np.array([[0.5, 1.5]]).T,
        L=np.eye(N),
        sn2_mult=None,
        Lchol=True
    )])
    vp = VariationalPosterior(D)
    vp.mu = np.array([
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0]
    ])

    acqf = AcqFcnVIQR()
    optim_state = {
        "gp_length_scale" : 2.0,
        "active_importance_sampling" : {
            "Xa" : np.arange(1,10).reshape((3, 3)) - 4.5,
            "Kax_mat" : np.eye(3,2).reshape((3, 2, 1)),
            "f_s2a" : np.arange(1,4).reshape(3, 1),
            "ln_w" : np.arange(1,4).reshape(1, 3) - 0.5
        },
        "lb_eps_orig" : -np.inf,
        "ub_eps_orig" : np.inf
    }
    sn2_new = np.zeros((1,1))
    sn2_new[:, 0] = gp.noise.compute(hyp[4:5], gp.X, gp.y, s2=None).reshape(-1,)
    gp.temporary_data["sn2_new"] = sn2_new.mean(1)
    gp.temporary_data["X_rescaled"] = gp.X / optim_state["gp_length_scale"]
    result = acqf(np.arange(1, 7).reshape(2, 3), gp, vp, None, optim_state)
    print(result)
