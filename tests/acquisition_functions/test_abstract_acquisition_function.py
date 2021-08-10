import sys

import gpyreg as gpr
import numpy as np
from pyvbmc.acquisition_functions import AbstractAcquisitionFunction
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior


def test_acq_info():
    class BasicAcqClass(AbstractAcquisitionFunction):
        def _compute_acquisition_function(
            self, Xs, vp, gp, optimState, f_mu, f_s2, f_bar, var_tot
        ):
            pass

    acq_fcn = BasicAcqClass()
    assert isinstance(acq_fcn.acq_info, dict)
    assert isinstance(acq_fcn.get_info(), dict)


def create_gp(D=3):
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    return gp


def test__call__simple(mocker):
    """
    Test only the main branches of the function.
    """

    class BasicAcqClass(AbstractAcquisitionFunction):
        def _compute_acquisition_function(
            self, Xs, vp, gp, optimState, f_mu, f_s2, f_bar, var_tot
        ):
            return np.ones(Xs.shape[0])

    M = 20
    Xs = np.ones((M, 3))

    mocker.patch(
        "gpyreg.GP.predict",
        return_value=(np.ones((M, 2)), np.zeros((M, 2))),
    )

    acq_fcn = BasicAcqClass()
    optim_state = dict()
    optim_state["integervars"] = None
    optim_state["variance_regularized_acq_fcn"] = False
    # no constraints for test
    optim_state["lb_eps_orig"] = -np.inf
    optim_state["ub_eps_orig"] = np.inf
    vp = VariationalPosterior(3)
    acq = acq_fcn(Xs, create_gp(3), vp, optim_state)

    assert np.all(acq == 1)
    assert acq.shape == (M,)


def test__call_constraints(mocker):
    """
    Test hard bound checking: discard points too close to bounds
    """

    class BasicAcqClass(AbstractAcquisitionFunction):
        def _compute_acquisition_function(
            self, Xs, vp, gp, optimState, f_mu, f_s2, f_bar, var_tot
        ):
            return np.ones(Xs.shape[0])

    M = 20
    Xs = np.ones((M, 3))

    mocker.patch(
        "gpyreg.GP.predict",
        return_value=(np.ones((M, 2)), np.zeros((M, 2))),
    )

    acq_fcn = BasicAcqClass()
    optim_state = dict()
    optim_state["integervars"] = None
    optim_state["variance_regularized_acq_fcn"] = False

    # set constraints for this test
    optim_state["lb_eps_orig"] = 1000
    optim_state["ub_eps_orig"] = 1001
    vp = VariationalPosterior(3)
    acq = acq_fcn(Xs, create_gp(3), vp, optim_state)

    assert acq.shape == (M,)
    assert np.all(acq == np.inf)


def test__call_quad(mocker):
    """
    Quadrature mean and variance for each hyperparameter sample by
    assigning vp.delta = np.ones(1, 2)
    """

    class BasicAcqClass(AbstractAcquisitionFunction):
        def _compute_acquisition_function(
            self, Xs, vp, gp, optimState, f_mu, f_s2, f_bar, var_tot
        ):
            return np.ones(Xs.shape[0])

    M = 1

    Xs = np.ones((M, 3))

    mocker.patch(
        "gpyreg.GP.quad",
        return_value=(np.ones((M, 1)), np.zeros((M, 1))),
    )

    acq_fcn = BasicAcqClass()
    optim_state = dict()
    optim_state["integervars"] = None
    optim_state["variance_regularized_acq_fcn"] = False
    # no constraints for test
    optim_state["lb_eps_orig"] = -np.inf
    optim_state["ub_eps_orig"] = np.inf
    vp = VariationalPosterior(3)
    # assign delta for test
    vp.delta = np.ones((1, 2))
    acq = acq_fcn(Xs, create_gp(3), vp, optim_state)

    assert acq.shape == (M,)
    assert np.all(acq == 1)


def test__call__regularization(mocker):
    """
    Test regularization (penalize points where GP uncertainty is below
    threshold).
    """

    class BasicAcqClass(AbstractAcquisitionFunction):
        def _compute_acquisition_function(
            self, Xs, vp, gp, optimState, f_mu, f_s2, f_bar, var_tot
        ):
            return np.ones(Xs.shape[0])

    M = 20
    Xs = np.ones((M, 3))

    mocker.patch(
        "gpyreg.GP.predict",
        return_value=(np.ones((M, 2)) * 5, np.ones((M, 2))),
    )

    acq_fcn = BasicAcqClass()
    optim_state = dict()
    optim_state["integervars"] = None
    optim_state["variance_regularized_acq_fcn"] = True
    optim_state["tol_gp_var"] = 2000

    # no constraints for test
    optim_state["lb_eps_orig"] = -np.inf
    optim_state["ub_eps_orig"] = np.inf
    vp = VariationalPosterior(3)

    # no logflag
    acq_fcn.acq_info["log_flag"] = False
    acq = acq_fcn(Xs, create_gp(3), vp, optim_state)
    assert acq.shape == (M,)
    assert np.allclose(acq, 0)

    # logflag
    acq_fcn.acq_info["log_flag"] = True
    acq = acq_fcn(Xs, create_gp(3), vp, optim_state)
    assert acq.shape == (M,)
    assert np.all(acq == 2000)


def test__call__real_min(mocker):
    """
    Test with compute_acquisition_function returning less than realmin.
    """

    realmin = sys.float_info.min

    class BasicAcqClass(AbstractAcquisitionFunction):
        def _compute_acquisition_function(
            self, Xs, vp, gp, optimState, f_mu, f_s2, f_bar, var_tot
        ):
            return -2 * np.ones(Xs.shape[0]) * realmin

    M = 20
    Xs = np.ones((M, 3))

    mocker.patch(
        "gpyreg.GP.predict",
        return_value=(np.ones((M, 2)), np.zeros((M, 2))),
    )

    acq_fcn = BasicAcqClass()
    optim_state = dict()
    optim_state["integervars"] = None
    optim_state["variance_regularized_acq_fcn"] = False
    # no constraints for test
    optim_state["lb_eps_orig"] = -np.inf
    optim_state["ub_eps_orig"] = np.inf
    vp = VariationalPosterior(3)
    acq = acq_fcn(Xs, create_gp(3), vp, optim_state)

    assert np.all(acq == realmin)
    assert acq.shape == (M,)


def test_real2int():
    """
    Test that real2int works correctly.
    """
    class BasicAcqClass(AbstractAcquisitionFunction):
        def _compute_acquisition_function(
            self, Xs, vp, gp, optimState, f_mu, f_s2, f_bar, var_tot
        ):
            pass
    acq_fcn = BasicAcqClass()
    D = 3
    parameter_transformer = ParameterTransformer(D)
    X = np.ones((10, D)) * 0.5
    integervars = np.array([True, False, False])
    X_after = acq_fcn._real2int(X, parameter_transformer, integervars)
    np.all(X_after[:, 0] == 0)
    np.all(X_after[:, 1] == 0.5)
    np.all(X_after[:, 2] == 0.5)

    integervars = np.array([False, False, False])
    X_after = acq_fcn._real2int(X, parameter_transformer, integervars)
    np.all(X_after== X) 
