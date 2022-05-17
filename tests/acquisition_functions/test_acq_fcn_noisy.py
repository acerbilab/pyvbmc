import gpyreg as gpr
import numpy as np
from pyvbmc.acquisition_functions import AcqFcnNoisy
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior


def test_acq_info():
    acqf = AcqFcnNoisy()
    assert isinstance(acqf.acq_info, dict)
    assert isinstance(acqf.get_info(), dict)
    assert not acqf.acq_info.get("log_flag")
    assert not acqf.acq_info.get("compute_varlogjoint")


def test__call__(mocker):
    acqf = AcqFcnNoisy()
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


def test_complex__call__(mocker):
    acqf = AcqFcnNoisy()
    M = 4
    Xs = np.arange(-3, 9).reshape(M, 3) / 10

    mocker.patch(
        "gpyreg.GP.predict",
        return_value=(
            np.arange(0, 2 * M).reshape((M, 2), order="F") / 13,
            np.arange(0, 2 * M).reshape((M, 2), order="F") / 23,
        ),
    )

    mocker.patch(
        "pyvbmc.variational_posterior.VariationalPosterior.pdf",
        return_value=np.arange(0, M).reshape((M, 1), order="F") * np.exp(-1),
    )

    optim_state = dict()
    optim_state["integervars"] = None
    optim_state["variance_regularized_acq_fcn"] = False
    optim_state["gp_length_scale"] = np.exp(
        np.mean(np.arange(0, 3 * 2).reshape((3, 2), order="F"), axis=1)
    ).T

    # no constraints for test
    optim_state["lb_eps_orig"] = -np.inf
    optim_state["ub_eps_orig"] = np.inf

    vp = VariationalPosterior(3)
    fun = lambda x: (np.sum(x), np.abs(0.5 * np.sum(x)))
    function_logger = FunctionLogger(fun, 3, True, 2)
    function_logger(-0.5 * np.ones(3))
    function_logger(np.ones(3))

    gp = gpr.GP(
        D=3,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    gp.temporary_data["sn2_new"] = np.exp(-1) * np.arange(
        1, M * 2 + 1
    ).reshape((M, 2), order="F")
    gp.temporary_data["X_rescaled"] = (
        function_logger.X[~np.isnan(function_logger.X).all(axis=1)]
        / optim_state["gp_length_scale"]
    )

    acq = acqf(Xs, gp, vp, function_logger, optim_state)

    assert acq.shape == (M,)
    acq_MATLAB = np.array(
        [
            -4.64016986306700e-311,
            -0.00133615097976628,
            -0.00254878290767088,
            -0.00565418068738646,
        ]
    )
    assert np.allclose(acq, acq_MATLAB)
