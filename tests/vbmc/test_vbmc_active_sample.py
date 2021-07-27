import numpy as np
import pytest
from pyvbmc.vbmc import VBMC
from pyvbmc.vbmc.active_sample import active_sample
import logging

fun = lambda x: np.sum(x + 2)


def create_vbmc(
    D: int,
    x0: float,
    lower_bounds: float,
    upper_bounds: float,
    plausible_lower_bounds: float,
    plausible_upper_bounds: float,
    user_options: dict = None,
):
    lb = np.ones((1, D)) * lower_bounds
    ub = np.ones((1, D)) * upper_bounds
    x0_array = np.ones((2, D)) * x0
    plb = np.ones((1, D)) * plausible_lower_bounds
    pub = np.ones((1, D)) * plausible_upper_bounds
    return VBMC(fun, x0_array, lb, ub, plb, pub, user_options)


def test_active_sample_initial_sample_no_y_values():
    """
    Test initial sample with provided_sample_count == sample_count and
    ys not available.
    """
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500)
    sample_count = 10
    x_orig = np.linspace((1, 11, 21), (10, 20, 30), sample_count)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(x_orig)
    y_orig = np.full(sample_count, np.nan)
    vbmc.optim_state["cache"]["y_orig"] = y_orig

    assert not np.all(np.isnan(vbmc.optim_state["cache"]["x_orig"][:10]))
    assert np.all(np.isnan(vbmc.optim_state["cache"]["y_orig"][:10]))

    function_logger, optim_state = active_sample(
        gp=None,
        sample_count=sample_count,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        parameter_transformer=vbmc.parameter_transformer,
        options=vbmc.options,
    )

    assert np.allclose(
        function_logger.x_orig[:10], x_orig, rtol=1e-12, atol=1e-14
    )
    assert np.allclose(
        np.ravel(function_logger.y_orig[:10]),
        [fun(x) for x in x_orig],
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.all(np.isnan(optim_state["cache"]["x_orig"][:10]))
    assert np.all(np.isnan(optim_state["cache"]["y_orig"][:10]))


def test_active_sample_initial_sample_y_values():
    """
    Test initial sample with provided_sample_count == sample_count and
    ys are available.
    """
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500)
    sample_count = 10
    x_orig = np.linspace((1, 11, 21), (10, 20, 30), sample_count)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(x_orig)
    y_orig = [fun(x) for x in x_orig]
    vbmc.optim_state["cache"]["y_orig"] = np.copy(y_orig)

    assert not np.all(np.isnan(vbmc.optim_state["cache"]["x_orig"][:10]))
    assert not np.all(np.isnan(vbmc.optim_state["cache"]["y_orig"][:10]))

    function_logger, optim_state = active_sample(
        gp=None,
        sample_count=sample_count,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        parameter_transformer=vbmc.parameter_transformer,
        options=vbmc.options,
    )

    assert np.allclose(
        function_logger.x_orig[:10], x_orig, rtol=1e-12, atol=1e-14
    )
    assert np.allclose(
        np.ravel(function_logger.y_orig[:10]),
        y_orig,
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.all(np.isnan(optim_state["cache"]["x_orig"][:10]))
    assert np.all(np.isnan(optim_state["cache"]["y_orig"][:10]))


def test_active_sample_initial_sample_plausible(mocker):
    """
    Test initial sample with provided_sample_count < sample_count and
    initdesign is plausible.
    """
    user_options = {"initdesign": "plausible"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    provided_sample_count = 10
    sample_count = provided_sample_count + 102
    x_orig = np.linspace((1, 11, 21), (10, 20, 30), provided_sample_count)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(x_orig)
    y_orig = [fun(x) for x in x_orig]
    vbmc.optim_state["cache"]["y_orig"] = np.copy(y_orig)

    assert not np.all(np.isnan(vbmc.optim_state["cache"]["x_orig"][:10]))
    assert not np.all(np.isnan(vbmc.optim_state["cache"]["y_orig"][:10]))

    # return a linespace so that random_Xs is with mean 0
    mocker.patch(
        "numpy.random.standard_normal",
        return_value=np.linspace(
            (-100, -100, -100),
            (100, 100, 100),
            sample_count - provided_sample_count,
        ),
    )
    function_logger, optim_state = active_sample(
        gp=None,
        sample_count=sample_count,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        parameter_transformer=vbmc.parameter_transformer,
        options=vbmc.options,
    )

    # provided samples
    assert np.allclose(
        function_logger.x_orig[:10], x_orig, rtol=1e-12, atol=1e-14
    )
    assert np.allclose(
        np.ravel(function_logger.y_orig[:10]),
        y_orig,
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.all(np.isnan(optim_state["cache"]["x_orig"][:10]))
    assert np.all(np.isnan(optim_state["cache"]["y_orig"][:10]))

    # new samples obtained by plausible sampling
    assert function_logger.Xn == sample_count - 1
    assert np.allclose(
        np.ravel(function_logger.y_orig[10:sample_count]),
        [fun(x) for x in function_logger.x_orig[10:sample_count]],
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.mean(function_logger.x_orig[10:sample_count]) == -500


def test_active_sample_initial_sample_narrow(mocker):
    """
    Test initial sample with provided_sample_count < sample_count and
    initdesign is plausible.
    """
    user_options = {"initdesign": "narrow"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    provided_sample_count = 10
    sample_count = provided_sample_count + 102
    x_orig = np.linspace((0, 0, 0), (10, 10, 10), provided_sample_count)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(x_orig)
    y_orig = [fun(x) for x in x_orig]
    vbmc.optim_state["cache"]["y_orig"] = np.copy(y_orig)

    assert not np.all(np.isnan(vbmc.optim_state["cache"]["x_orig"][:10]))
    assert not np.all(np.isnan(vbmc.optim_state["cache"]["y_orig"][:10]))

    # return a linespace so that random_Xs is with mean 0
    mocker.patch(
        "numpy.random.standard_normal",
        return_value=np.linspace(
            (-0.1, -0.1, -0.1),
            (0.1, 0.1, 0.1),
            sample_count - provided_sample_count,
        ),
    )
    function_logger, optim_state = active_sample(
        gp=None,
        sample_count=sample_count,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        parameter_transformer=vbmc.parameter_transformer,
        options=vbmc.options,
    )

    # provided samples
    assert np.allclose(
        function_logger.x_orig[:10], x_orig, rtol=1e-12, atol=1e-14
    )
    assert np.allclose(
        np.ravel(function_logger.y_orig[:10]),
        y_orig,
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.all(np.isnan(optim_state["cache"]["x_orig"][:10]))
    assert np.all(np.isnan(optim_state["cache"]["y_orig"][:10]))

    # new samples obtained by narrow sampling
    assert function_logger.Xn == sample_count - 1
    assert np.allclose(
        np.ravel(function_logger.y_orig[10:sample_count]),
        [fun(x) for x in function_logger.x_orig[10:sample_count]],
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.allclose(
        np.mean(function_logger.x_orig[10:sample_count]),
        -50,
        rtol=1e-12,
        atol=1e-14,
    )


def test_active_sample_initial_sample_unknown_initial_design():
    """
    Test initial sample with provided_sample_count < sample_count and
    initdesign is unknown.
    """
    user_options = {"initdesign": "unknown"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    sample_count = 10

    with pytest.raises(ValueError) as execinfo:
        active_sample(
            gp=None,
            sample_count=sample_count,
            optim_state=vbmc.optim_state,
            function_logger=vbmc.function_logger,
            parameter_transformer=vbmc.parameter_transformer,
            options=vbmc.options,
        )
    assert "Unknown initial design for VBMC" in execinfo.value.args[0]


def test_active_sample_logger():
    """
    Test logging levels for various options.
    """
    # iter which is INFO
    user_options = {"display": "iter"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    active_sample(
        gp=None,
        sample_count=1,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        parameter_transformer=vbmc.parameter_transformer,
        options=vbmc.options,
    )
    assert logging.getLogger("ActiveSample").getEffectiveLevel() == 20

    # off which is WARN
    user_options = {"display": "off"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    active_sample(
        gp=None,
        sample_count=1,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        parameter_transformer=vbmc.parameter_transformer,
        options=vbmc.options,
    )
    assert logging.getLogger("ActiveSample").getEffectiveLevel() == 30

    # full which is DEBUG
    user_options = {"display": "full"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    active_sample(
        gp=None,
        sample_count=1,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        parameter_transformer=vbmc.parameter_transformer,
        options=vbmc.options,
    )
    assert logging.getLogger("ActiveSample").getEffectiveLevel() == 10

    # anything else is also INFO
    user_options = {"display": "strange_option"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    active_sample(
        gp=None,
        sample_count=1,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        parameter_transformer=vbmc.parameter_transformer,
        options=vbmc.options,
    )
    assert logging.getLogger("ActiveSample").getEffectiveLevel() == 20


def test_active_sample_initial_sample_more_provided(caplog):
    """
    Test initial sample with provided_sample_count > sample_count
    """
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500)
    provided_sample_count = 100
    sample_count = provided_sample_count - 10
    x_orig = np.linspace((0, 0, 0), (10, 10, 10), provided_sample_count)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(x_orig)
    y_orig = [fun(x) for x in x_orig]
    vbmc.optim_state["cache"]["y_orig"] = np.copy(y_orig)

    assert not np.all(np.isnan(vbmc.optim_state["cache"]["x_orig"]))
    assert not np.all(np.isnan(vbmc.optim_state["cache"]["y_orig"]))
    caplog.set_level(logging.INFO)
    function_logger, optim_state = active_sample(
        gp=None,
        sample_count=sample_count,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        parameter_transformer=vbmc.parameter_transformer,
        options=vbmc.options,
    )

    logger_message = "More than sample_count = 90 initial points have been "
    logger_message += "provided, using only the first 90 points."
    assert caplog.record_tuples == [
        ("ActiveSample", logging.INFO, logger_message)
    ]

    assert np.allclose(
        function_logger.x_orig[:sample_count],
        x_orig[:sample_count],
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.allclose(
        np.ravel(function_logger.y_orig[:sample_count]),
        y_orig[:sample_count],
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.all(np.isnan(optim_state["cache"]["x_orig"][:sample_count]))
    assert np.all(np.isnan(optim_state["cache"]["y_orig"][:sample_count]))
    assert function_logger.Xn == sample_count - 1
