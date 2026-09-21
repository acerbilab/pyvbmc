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
    assert acq_fcn.acq_info["compute_var_log_joint"] is False


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


def test_context_hooks_preserve_old_acquisition_signature():
    """A subclass implementing only the established method receives the
    ordinary arguments even when its prediction hook supplies context."""

    marker = object()

    class OldSignatureAcq(AbstractAcqFcn):
        def __init__(self):
            super().__init__()
            self.arguments = None

        def _predict_with_context(self, Xs, gp):
            shape = (Xs.shape[0], 1)
            return np.zeros(shape), np.ones(shape), marker

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
            self.arguments = (
                Xs,
                vp,
                gp,
                function_logger,
                optim_state,
                f_mu,
                f_s2,
                f_bar,
                var_tot,
            )
            return np.arange(Xs.shape[0], dtype=np.float64)

    acq_fcn = OldSignatureAcq()
    vp = VariationalPosterior(3)
    gp = create_gp(3)
    logger = FunctionLogger(lambda x: x, 3, False, 0)
    optim_state = {
        "integer_vars": None,
        "lb_eps_orig": -np.inf,
        "ub_eps_orig": np.inf,
    }
    Xs = np.ones((3, 3))

    actual = acq_fcn(Xs, gp, vp, logger, optim_state)

    np.testing.assert_array_equal(actual, [0.0, 1.0, 2.0])
    assert len(acq_fcn.arguments) == 9
    for actual_arg, expected_arg in zip(
        acq_fcn.arguments[:5], (Xs, vp, gp, logger, optim_state)
    ):
        assert actual_arg is expected_arg
    assert all(argument is not marker for argument in acq_fcn.arguments)


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

    0.5 is a tie, which ``misc/real2int_vbmc.m:7`` sends away from zero.
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
    assert np.all(X_after[:, 0] == 1)
    assert np.all(X_after[:, 1] == 0.5)
    assert np.all(X_after[:, 2] == 0.5)

    integer_vars = np.array([False, False, False])
    X_after = acq_fcn._real2int(X, parameter_transformer, integer_vars)
    assert np.all(X_after == X)


def test_real2int_rounds_a_half_away_from_zero():
    """Every tie goes away from zero, as ``round`` does in
    ``misc/real2int_vbmc.m:7``."""

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
    D = 1
    parameter_transformer = ParameterTransformer(D)
    ties = np.array([[-2.5], [-1.5], [-0.5], [0.5], [1.5], [2.5]])
    expected = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    X_after = acq_fcn._real2int(
        ties.copy(), parameter_transformer, np.array([True])
    )
    assert np.array_equal(X_after, expected)

    # The neighbours of a tie go to the nearer integer. Adding a half to
    # the largest number below a half gives one in floating point, and to
    # an odd integer of 53 bits gives the even integer above it.
    below_half = np.nextafter(0.5, 0.0)
    odd = 2.0**52 + 1.0
    near = np.array([[below_half], [-below_half], [odd], [-odd], [0.0]])
    expected = np.array([[0.0], [0.0], [odd], [-odd], [0.0]])
    X_after = acq_fcn._real2int(
        near.copy(), parameter_transformer, np.array([True])
    )
    assert np.array_equal(X_after, expected)


def test_real2int_rounds_a_box_midpoint_away_from_zero():
    """The reachable tie, through the default probit transform.

    The hard bounds of an integer variable sit at half-integers, so the
    midpoint of a box with an even number of integer levels is itself a
    half-integer, and the transform maps it back and forth exactly.
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
    D = 2
    parameter_transformer = ParameterTransformer(
        D,
        np.array([[-0.5, -0.5]]),
        np.array([[9.5, 9.5]]),
        np.array([[0.0, 0.0]]),
        np.array([[9.0, 9.0]]),
    )
    midpoint = np.array([[4.5, 4.5]])
    X = parameter_transformer(midpoint)
    assert np.array_equal(parameter_transformer.inverse(X), midpoint)

    X_after = acq_fcn._real2int(
        X, parameter_transformer, np.array([True, False])
    )
    X_orig = parameter_transformer.inverse(X_after)
    assert X_orig[0, 0] == 5.0
    assert X_orig[0, 1] == 4.5


def test_real2int_single_point():
    """A single point is snapped in the shape it is given.

    ``misc/real2int_vbmc.m`` takes the row vector that the local search of
    ``private/activesample_vbmc.m:325`` hands it, and the search optimizers
    of ``active_sample`` return a one-dimensional point.
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
    integer_vars = np.array([True, False, False])
    x = np.array([1.2, 1.2, 1.2])
    x_after = acq_fcn._real2int(x, parameter_transformer, integer_vars)

    assert x_after.shape == (D,)
    assert x_after[0] == 1.0
    assert np.all(x_after[1:] == 1.2)
    # Snapped in place, as an array of points is.
    assert x_after is x
    assert x[0] == 1.0


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
