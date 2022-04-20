import pytest
import gpyreg as gpr
import numpy as np
import scipy.stats as st
from pyvbmc.acquisition_functions import AcqFcnVIQR
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior


def test_acq_info():
    acqf = AcqFcnVIQR()
    assert acqf.importance_sampling
    assert not acqf.importance_sampling_vp
    assert acqf.variational_importance_sampling
    assert acqf.log_flag
    assert np.isclose(st.norm.cdf(acqf.u), 0.75)

def test__call__():
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
    acqf(np.arange(1, 7).reshape(2, 3), gp, vp, None, optim_state)


@pytest.mark.skip
def test__call__2(mocker):
    acqf = AcqFcnVIQR()
    M = 3
    Xs = np.ones((M, 3))

    mocker.patch(
        "gpyreg.GP.predict",
        return_value=(np.ones((M, 2)) * 3, np.ones((M, 2))),
    )

    mocker.patch(
        "pyvbmc.variational_posterior.VariationalPosterior.pdf",
        return_value=np.ones((M, 1)),
    )

    optim_state = dict()
    optim_state["integervars"] = None
    optim_state["variance_regularized_acq_fcn"] = False
    optim_state["gp_length_scale"] = np.exp(np.mean(np.ones((3, 2)), axis=1)).T

    # no constraints for test
    optim_state["lb_eps_orig"] = -np.inf
    optim_state["ub_eps_orig"] = np.inf

    vp = VariationalPosterior(3)
    function_logger = FunctionLogger(np.sum, 3, False, 0)
    function_logger(np.ones(3))

    gp = gpr.GP(
        D=3,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    gp.temporary_data["sn2_new"] = np.ones((M, 2))
    gp.temporary_data["X_rescaled"] = Xs / optim_state["gp_length_scale"]

    acq = acqf(Xs, gp, vp, function_logger, optim_state)

    assert acq.shape == (M,)
    assert np.all(acq == -0.5)


@pytest.mark.skip
def test_complex__call__(mocker):
    acqf = AcqFcnNoisy()
    M = 4
    Xs = np.arange(-3, 9).reshape(M, 3) / 10

    mocker.patch(
        "gpyreg.GP.predict",
        return_value=(np.arange(0, 2*M).reshape((M, 2), order="F")/13,
                      np.arange(0, 2*M).reshape((M, 2), order="F")/23 ),
    )

    mocker.patch(
        "pyvbmc.variational_posterior.VariationalPosterior.pdf",
        return_value=np.arange(0, M).reshape((M, 1), order="F") * np.exp(-1)
    )

    optim_state = dict()
    optim_state["integervars"] = None
    optim_state["variance_regularized_acq_fcn"] = False
    optim_state["gp_length_scale"] = np.exp(np.mean(np.arange(0, 3*2).reshape((3, 2), order="F"), axis=1)).T

    # no constraints for test
    optim_state["lb_eps_orig"] = -np.inf
    optim_state["ub_eps_orig"] = np.inf

    vp = VariationalPosterior(3)
    fun = lambda x: (np.sum(x), np.abs(0.5*np.sum(x)))
    function_logger = FunctionLogger(fun, 3, True, 2)
    function_logger(-0.5*np.ones(3))
    function_logger(np.ones(3))

    gp = gpr.GP(
        D=3,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    gp.temporary_data["sn2_new"] = np.exp(-1) *\
        np.arange(1, M*2+1).reshape((M, 2), order="F")
    gp.temporary_data["X_rescaled"] = function_logger.X[
        ~np.isnan(function_logger.X).all(axis=1)
    ] / optim_state["gp_length_scale"]

    acq = acqf(Xs, gp, vp, function_logger, optim_state)

    assert acq.shape == (M,)
    acq_MATLAB = np.array([
        -4.64016986306700e-311,
        -0.00133615097976628,
        -0.00254878290767088,
        -0.00565418068738646
    ])
    assert np.allclose(acq, acq_MATLAB)
