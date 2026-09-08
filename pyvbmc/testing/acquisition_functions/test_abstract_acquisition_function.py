import sys
import warnings

import gpyreg as gpr
import numpy as np
import pytest

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


class FixedOutputAcq(AbstractAcqFcn):
    def __init__(self, output, log_flag=False):
        super().__init__()
        self.output = output
        self.acq_info["log_flag"] = log_flag

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
        return self.output


def call_fixed_acq(mocker, output, f_s2, optim_state, log_flag=False):
    M = f_s2.shape[0]
    mocker.patch(
        "gpyreg.GP.predict",
        return_value=(np.zeros_like(f_s2), f_s2),
    )
    state = {
        "integer_vars": None,
        "lb_eps_orig": -np.inf,
        "ub_eps_orig": np.inf,
        **optim_state,
    }
    vp = VariationalPosterior(3)
    logger = FunctionLogger(lambda x: x, 3, False, 0)
    return FixedOutputAcq(output, log_flag)(
        np.ones((M, 3)), create_gp(3), vp, logger, state
    )


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


@pytest.mark.parametrize(
    "output",
    [
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0, 2.0, 3.0]]),
        np.array([[1.0], [2.0], [3.0]]),
    ],
)
def test__call__normalizes_vector_shapes_before_masks(mocker, output):
    result = call_fixed_acq(
        mocker,
        output,
        np.ones((3, 1)),
        {"variance_regularized_acq_fcn": False},
    )
    assert result.shape == (3,)
    assert np.array_equal(result, [1.0, 2.0, 3.0])


def test__call__normalizes_row_before_bounds_mask(mocker):
    result = call_fixed_acq(
        mocker,
        np.array([[1.0, 2.0, 3.0]]),
        np.ones((3, 1)),
        {
            "variance_regularized_acq_fcn": False,
            "lb_eps_orig": 2.0,
            "ub_eps_orig": 3.0,
        },
    )
    assert result.shape == (3,)
    assert np.all(result == np.inf)


@pytest.mark.parametrize(
    "output",
    [
        np.ones((3, 2)),
        np.ones(2),
        np.array(1.0),
        np.ones((3, 1, 1)),
    ],
)
def test__call__rejects_invalid_acquisition_shapes(mocker, output):
    with pytest.raises(ValueError, match="one value per input point"):
        call_fixed_acq(
            mocker,
            output,
            np.ones((3, 1)),
            {"variance_regularized_acq_fcn": False},
        )


@pytest.mark.parametrize(
    ("log_flag", "output", "expected"),
    [
        (
            False,
            np.array([np.inf, 3.0, 4.0]),
            np.array([0.0, 3.0 * np.exp(-1.0), 4.0]),
        ),
        (
            True,
            np.array([-np.inf, 3.0, 4.0]),
            np.array([np.inf, 4.0, 4.0]),
        ),
    ],
)
def test__call__regularization_mixed_and_zero_variance(
    mocker, log_flag, output, expected
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = call_fixed_acq(
            mocker,
            output.reshape(1, -1),
            np.array([[0.0], [0.5], [2.0]]),
            {
                "variance_regularized_acq_fcn": True,
                "tol_gp_var": 1.0,
            },
            log_flag=log_flag,
        )
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"variance_regularized_acq_fcn": True}, np.exp(-1.0)),
        ({"variance_regularized_acqfcn": True}, np.exp(-1.0)),
        (
            {
                "variance_regularized_acq_fcn": False,
                "variance_regularized_acqfcn": True,
            },
            1.0,
        ),
        ({"variance_regularized_acqfcn": False}, 1.0),
        ({}, 1.0),
    ],
)
def test__call__regularization_key_compatibility(mocker, state, expected):
    state = {**state, "tol_gp_var": 1.0}
    result = call_fixed_acq(mocker, np.array([1.0]), np.array([[0.5]]), state)
    assert result.shape == (1,)
    assert np.allclose(result, expected)


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
