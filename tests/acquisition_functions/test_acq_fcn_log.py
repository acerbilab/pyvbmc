import gpyreg as gpr
import numpy as np
from pyvbmc.acquisition_functions import AcqFcnLog
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior


def test_acq_info():
    acqf = AcqFcnLog()
    assert isinstance(acqf.acq_info, dict)
    assert isinstance(acqf.get_info(), dict)
    assert acqf.acq_info.get("log_flag")
    assert not acqf.acq_info.get("compute_varlogjoint")


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
        return_value=np.ones((M, 1)),
    )

    optim_state = dict()
    optim_state["integervars"] = None
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

    assert acq.shape == (M, )
    assert np.all(acq == 1)