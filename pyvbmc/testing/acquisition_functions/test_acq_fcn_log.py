import gpyreg as gpr
import numpy as np

from pyvbmc.acquisition_functions import AcqFcn, AcqFcnLog
from pyvbmc.acquisition_functions.utilities import string_to_acq
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior


def test_acq_info():
    acqf = AcqFcnLog()
    assert isinstance(acqf.acq_info, dict)
    assert isinstance(acqf.get_info(), dict)
    assert acqf.acq_info.get("log_flag")
    assert not acqf.acq_info.get("compute_var_log_joint")

    # Test handling of string input for SearchAcqFcn:
    acqf2 = AcqFcnLog()
    acqf2 = string_to_acq("AcqFcnLog")
    acqf3 = string_to_acq("AcqFcnLog()")
    assert type(acqf) == type(acqf2) == type(acqf3)


def test__call__(mocker):
    acqf = AcqFcnLog()
    M = 3
    Xs = np.ones((M, 3))

    mocker.patch(
        "gpyreg.GP.predict",
        return_value=(np.ones((M, 2)) * 2, np.ones((M, 2))),
    )

    mocker.patch(
        "pyvbmc.variational_posterior.VariationalPosterior.pdf",
        return_value=np.zeros((M, 1)),
    )

    optim_state = dict()
    optim_state["integer_vars"] = None
    optim_state["variance_regularized_acq_fcn"] = False

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

    acq = acqf(Xs, gp, vp, function_logger, optim_state)

    assert acq.shape == (M,)
    assert np.all(acq == 1)


def test_acq_fcn_vs_acq_fcn_log(mocker):
    acqf = AcqFcn()
    acqf_log = AcqFcnLog()
    M = 3
    Xs = np.random.normal(size=(M, 2))

    optim_state = dict()
    optim_state["integer_vars"] = None
    optim_state["variance_regularized_acq_fcn"] = False

    # no constraints for test
    optim_state["lb_eps_orig"] = -np.inf
    optim_state["ub_eps_orig"] = np.inf

    # Set up VP
    vp = VariationalPosterior(2)
    vp.mu = np.array([[-1.0, -1.0], [1.0, 1.0]])
    vp.sigma = np.ones((1, 2))
    function_logger = FunctionLogger(
        lambda t: vp.pdf(t, log_flag=True), 2, False, 0
    )
    function_logger(np.ones(2))

    # Set up GP
    gp = gpr.GP(
        D=2,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
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
                -(3 / 2) * np.log(2 * np.pi),  # MVN mode
                0.0,
                0.0,  # Mode location
                0.0,
                0.0,  # log scale
            ]
        ]
    )
    gp.update(X_new=Xs, y_new=vp.pdf(Xs, log_flag=True), hyp=hyp)

    acq = acqf(Xs, gp, vp, function_logger, optim_state)
    log_acq = acqf_log(Xs, gp, vp, function_logger, optim_state)

    assert np.allclose(-np.log(-acq), log_acq)
