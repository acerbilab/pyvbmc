import sys

import gpyreg as gpr
import numpy as np

from pyvbmc.acquisition_functions import AbstractAcqFcn
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior


def test_acq_info():
    class BasicAcqClass(AbstractAcqFcn):
        def _compute_acquisition_function(
            self,
            Xs,
            vp,
            gp,
            function_logger,
            optim_state,
            f_mu,
            f_s2,
            f_bar,
            var_tot,
        ):
            pass

    acq_fcn = BasicAcqClass()
    assert isinstance(acq_fcn.acq_info, dict)
    assert isinstance(acq_fcn.get_info(), dict)
    assert not acq_fcn.acq_info.get("log_flag")
    assert not acq_fcn.acq_info.get("compute_var_log_joint")


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

    class BasicAcqClass(AbstractAcqFcn):
        def _compute_acquisition_function(
            self,
            Xs,
            vp,
            gp,
            function_logger,
            optim_state,
            f_mu,
            f_s2,
            f_bar,
            var_tot,
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
    optim_state["integer_vars"] = None
    optim_state["variance_regularized_acq_fcn"] = False
    # no constraints for test
    optim_state["lb_eps_orig"] = -np.inf
    optim_state["ub_eps_orig"] = np.inf
    vp = VariationalPosterior(3)
    function_logger = FunctionLogger(lambda x: x, 3, False, 0)
    acq = acq_fcn(Xs, create_gp(3), vp, function_logger, optim_state)

    assert np.all(acq == 1)
    assert acq.shape == (M,)


def test__call_constraints(mocker):
    """
    Test hard bound checking: discard points too close to bounds
    """

    class BasicAcqClass(AbstractAcqFcn):
        def _compute_acquisition_function(
            self,
            Xs,
            vp,
            gp,
            function_logger,
            optim_state,
            f_mu,
            f_s2,
            f_bar,
            var_tot,
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
    optim_state["integer_vars"] = None
    optim_state["variance_regularized_acq_fcn"] = False

    # set constraints for this test
    optim_state["lb_eps_orig"] = 1000
    optim_state["ub_eps_orig"] = 1001
    vp = VariationalPosterior(3)
    function_logger = FunctionLogger(lambda x: x, 3, False, 0)
    acq = acq_fcn(Xs, create_gp(3), vp, function_logger, optim_state)

    assert acq.shape == (M,)
    assert np.all(acq == np.inf)


def test__call__regularization(mocker):
    """
    Test regularization (penalize points where GP uncertainty is below
    threshold).
    """

    class BasicAcqClass(AbstractAcqFcn):
        def _compute_acquisition_function(
            self,
            Xs,
            vp,
            gp,
            function_logger,
            optim_state,
            f_mu,
            f_s2,
            f_bar,
            var_tot,
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
    optim_state["integer_vars"] = None
    optim_state["variance_regularized_acq_fcn"] = True
    optim_state["tol_gp_var"] = 2000

    # no constraints for test
    optim_state["lb_eps_orig"] = -np.inf
    optim_state["ub_eps_orig"] = np.inf

    vp = VariationalPosterior(3)
    function_logger = FunctionLogger(lambda x: x, 3, False, 0)

    # no log_flag
    acq_fcn.acq_info["log_flag"] = False
    acq = acq_fcn(Xs, create_gp(3), vp, function_logger, optim_state)
    assert acq.shape == (M,)
    assert np.allclose(acq, 0)

    # log_flag
    acq_fcn.acq_info["log_flag"] = True
    acq = acq_fcn(Xs, create_gp(3), vp, function_logger, optim_state)
    assert acq.shape == (M,)
    assert np.all(acq == 2000)


def test__call__real_max(mocker):
    """
    Test with compute_acquisition_function returning less than realmin.
    """

    realmax = sys.float_info.max

    class BasicAcqClass(AbstractAcqFcn):
        def _compute_acquisition_function(
            self,
            Xs,
            vp,
            gp,
            function_logger,
            optim_state,
            f_mu,
            f_s2,
            f_bar,
            var_tot,
        ):
            return np.ones(Xs.shape[0]) * -realmax - 1

    M = 20
    Xs = np.ones((M, 3))

    mocker.patch(
        "gpyreg.GP.predict",
        return_value=(np.ones((M, 2)), np.zeros((M, 2))),
    )

    acq_fcn = BasicAcqClass()
    optim_state = dict()
    optim_state["integer_vars"] = None
    optim_state["variance_regularized_acq_fcn"] = False
    # no constraints for test
    optim_state["lb_eps_orig"] = -np.inf
    optim_state["ub_eps_orig"] = np.inf
    vp = VariationalPosterior(3)
    function_logger = FunctionLogger(lambda x: x, 3, False, 0)
    acq = acq_fcn(Xs, create_gp(3), vp, function_logger, optim_state)

    assert np.all(acq == -realmax)
    assert acq.shape == (M,)


def test_real2int():
    """
    Test that real2int works correctly.
    """

    class BasicAcqClass(AbstractAcqFcn):
        def _compute_acquisition_function(
            self,
            Xs,
            vp,
            gp,
            function_logger,
            optim_state,
            f_mu,
            f_s2,
            f_bar,
            var_tot,
        ):
            pass

    acq_fcn = BasicAcqClass()
    D = 3
    parameter_transformer = ParameterTransformer(D)
    X = np.ones((10, D)) * 0.5
    integer_vars = np.array([True, False, False])
    X_after = acq_fcn._real2int(X, parameter_transformer, integer_vars)
    np.all(X_after[:, 0] == 0)
    np.all(X_after[:, 1] == 0.5)
    np.all(X_after[:, 2] == 0.5)

    integer_vars = np.array([False, False, False])
    X_after = acq_fcn._real2int(X, parameter_transformer, integer_vars)
    np.all(X_after == X)


def test_sq_dist():
    """
    Test data has been crossvalidated with (original) VBMC in MATLAB.
    """

    class BasicAcqClass(AbstractAcqFcn):
        def _compute_acquisition_function(
            self,
            Xs,
            vp,
            gp,
            function_logger,
            optim_state,
            f_mu,
            f_s2,
            f_bar,
            var_tot,
        ):
            pass

    a = np.linspace((1, 11), (10, 20), 10)
    b = np.linspace((30, 40), (50, 60), 21)
    acqf = BasicAcqClass()
    c = acqf._sq_dist(a, b)
    assert c.shape == (10, 21)
    assert c[0, 0] == 1682
    assert c[0, 20] == 4802
    assert c[9, 0] == 800
    assert c[9, 20] == 3200
