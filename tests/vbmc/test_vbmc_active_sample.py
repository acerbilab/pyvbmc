import logging

import numpy as np
import pytest

from pyvbmc.vbmc import VBMC, active_sample
from pyvbmc.stats import get_hpd
from pyvbmc.vbmc.active_sample import _get_search_points

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
    X_orig = np.linspace((1, 11, 21), (10, 20, 30), sample_count)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(X_orig)
    y_orig = np.full(sample_count, np.nan)
    vbmc.optim_state["cache"]["y_orig"] = y_orig

    assert not np.all(np.isnan(vbmc.optim_state["cache"]["x_orig"][:10]))
    assert np.all(np.isnan(vbmc.optim_state["cache"]["y_orig"][:10]))

    function_logger, optim_state = active_sample(
        gp=None,
        sample_count=sample_count,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )

    assert np.allclose(
        function_logger.X_orig[:10], X_orig, rtol=1e-12, atol=1e-14
    )
    assert np.allclose(
        np.ravel(function_logger.y_orig[:10]),
        [fun(x) for x in X_orig],
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
    X_orig = np.linspace((1, 11, 21), (10, 20, 30), sample_count)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(X_orig)
    y_orig = [fun(x) for x in X_orig]
    vbmc.optim_state["cache"]["y_orig"] = np.copy(y_orig)

    assert not np.all(np.isnan(vbmc.optim_state["cache"]["x_orig"][:10]))
    assert not np.all(np.isnan(vbmc.optim_state["cache"]["y_orig"][:10]))

    function_logger, optim_state = active_sample(
        gp=None,
        sample_count=sample_count,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )

    assert np.allclose(
        function_logger.X_orig[:10], X_orig, rtol=1e-12, atol=1e-14
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
    X_orig = np.linspace((1, 11, 21), (10, 20, 30), provided_sample_count)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(X_orig)
    y_orig = [fun(x) for x in X_orig]
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
        vp=vbmc.vp,
        options=vbmc.options,
    )

    # provided samples
    assert np.allclose(
        function_logger.X_orig[:10], X_orig, rtol=1e-12, atol=1e-14
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
        [fun(x) for x in function_logger.X_orig[10:sample_count]],
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.mean(function_logger.X_orig[10:sample_count]) == -500


def test_active_sample_initial_sample_narrow(mocker):
    """
    Test initial sample with provided_sample_count < sample_count and
    initdesign is plausible.
    """
    user_options = {"initdesign": "narrow"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    provided_sample_count = 10
    sample_count = provided_sample_count + 102
    X_orig = np.linspace((0, 0, 0), (10, 10, 10), provided_sample_count)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(X_orig)
    y_orig = [fun(x) for x in X_orig]
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
        vp=vbmc.vp,
        options=vbmc.options,
    )

    # provided samples
    assert np.allclose(
        function_logger.X_orig[:10], X_orig, rtol=1e-12, atol=1e-14
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
        [fun(x) for x in function_logger.X_orig[10:sample_count]],
    )
    assert np.allclose(
        np.mean(function_logger.X_orig[10:sample_count]),
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
            vp=vbmc.vp,
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
        vp=vbmc.vp,
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
        vp=vbmc.vp,
        options=vbmc.options,
    )

    # full which is DEBUG
    user_options = {"display": "full"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    active_sample(
        gp=None,
        sample_count=1,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
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
        vp=vbmc.vp,
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
    X_orig = np.linspace((0, 0, 0), (10, 10, 10), provided_sample_count)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(X_orig)
    y_orig = [fun(x) for x in X_orig]
    vbmc.optim_state["cache"]["y_orig"] = np.copy(y_orig)

    assert not np.all(np.isnan(vbmc.optim_state["cache"]["y_orig"]))
    caplog.set_level(logging.INFO)
    function_logger, optim_state = active_sample(
        gp=None,
        sample_count=sample_count,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )

    logger_message = "More than sample_count = 90 initial points have been "
    logger_message += "provided, using only the first 90 points."
    assert caplog.record_tuples == [
        ("ActiveSample", logging.INFO, logger_message)
    ]

    assert np.allclose(
        function_logger.X_orig[:sample_count],
        X_orig[:sample_count],
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


def test_get_search_points_all_cache():
    """
    Take all points from cache.
    """
    user_options = {"cachefrac": 1}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    number_of_points = 2
    x_orig = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(x_orig)

    # no search bounds for test
    vbmc.optim_state["LB_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["UB_search"] = np.full((1, 3), np.inf)
    search_X, idx_cache = _get_search_points(
        number_of_points=number_of_points,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )

    assert np.all(search_X == vbmc.parameter_transformer(x_orig[idx_cache]))
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    assert np.all(np.isin(idx_cache, np.arange(number_of_points)))


def test_get_search_points_all_search_cache():
    """
    Take all points from search cache.
    """
    user_options = {
        "cachefrac": 1,
        "searchcachefrac": 1,
        "heavytailsearchfrac": 0,
        "mvnsearchfrac": 0,
        "boxsearchfrac": 0,
        "hpdsearchfrac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    number_of_points = 2
    X = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.zeros(0)
    vbmc.optim_state["searchcache"] = np.copy(X)

    # no search bounds for test
    vbmc.optim_state["LB_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["UB_search"] = np.full((1, 3), np.inf)
    search_X, idx_cache = _get_search_points(
        number_of_points=number_of_points,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    assert np.all(search_X == X)
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    assert np.all(np.isnan(idx_cache))


def test_get_search_points_search_bounds():
    """
    Ensure that search bounds constrain the search points.
    """
    user_options = {
        "cachefrac": 1,
        "searchcachefrac": 0,
        "heavytailsearchfrac": 0,
        "mvnsearchfrac": 0,
        "boxsearchfrac": 0,
        "hpdsearchfrac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    number_of_points = 2
    X = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(X)

    # no search bounds for test
    vbmc.optim_state["LB_search"] = np.full((1, 3), 2)
    vbmc.optim_state["UB_search"] = np.full((1, 3), 4)
    search_X, idx_cache = _get_search_points(
        number_of_points=number_of_points,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    assert np.all(search_X >= 2)
    assert np.all(search_X <= 4)
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    assert np.all(np.isin(idx_cache, np.arange(number_of_points)))


def test_get_search_points_all_heavytailsearch():
    """
    Take all points from heavytail search.
    """
    user_options = {
        "cachefrac": 1,
        "searchcachefrac": 0,
        "heavytailsearchfrac": 1,
        "mvnsearchfrac": 0,
        "boxsearchfrac": 0,
        "hpdsearchfrac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    number_of_points = 2
    X = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.zeros(0)

    # no search bounds for test
    vbmc.optim_state["LB_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["UB_search"] = np.full((1, 3), np.inf)
    search_X, idx_cache = _get_search_points(
        number_of_points=number_of_points,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    assert np.all(np.isnan(idx_cache))


def test_get_search_points_all_mvn():
    """
    Take all points from mvn.
    """
    user_options = {
        "cachefrac": 1,
        "searchcachefrac": 0,
        "heavytailsearchfrac": 0,
        "mvnsearchfrac": 1,
        "boxsearchfrac": 0,
        "hpdsearchfrac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    number_of_points = 2
    X = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.zeros(0)

    # no search bounds for test
    vbmc.optim_state["LB_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["UB_search"] = np.full((1, 3), np.inf)
    search_X, idx_cache = _get_search_points(
        number_of_points=number_of_points,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    assert np.all(np.isnan(idx_cache))


def test_get_search_points_all_mvn_vp_sample():
    """
    Take all points from mvn vp sample.
    """
    user_options = {
        "cachefrac": 1,
        "searchcachefrac": 0,
        "heavytailsearchfrac": 0,
        "mvnsearchfrac": 0,
        "boxsearchfrac": 0,
        "hpdsearchfrac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    number_of_points = 2
    X = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.zeros(0)

    # no search bounds for test
    vbmc.optim_state["LB_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["UB_search"] = np.full((1, 3), np.inf)
    search_X, idx_cache = _get_search_points(
        number_of_points=number_of_points,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    assert np.all(np.isnan(idx_cache))


def test_get_search_points_all_box_search(mocker):
    """
    Take all points from box search.
    """
    user_options = {
        "cachefrac": 1,
        "searchcachefrac": 0,
        "heavytailsearchfrac": 0,
        "mvnsearchfrac": 0,
        "boxsearchfrac": 1,
        "hpdsearchfrac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    number_of_points = 2
    X = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.zeros(0)

    # record some samples in FunctionLogger
    for i in range(10):
        vbmc.function_logger(np.ones(3) * i)
    assert vbmc.function_logger.Xn == 9

    # return a linespace so that random samples are predicatable.
    random_values = np.linspace(
        (-100, -100, -100),
        (100, 100, 100),
        number_of_points,
    )
    mocker.patch(
        "numpy.random.standard_normal",
        return_value=random_values,
    )
    # infinite bounds
    vbmc.optim_state["LB_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["UB_search"] = np.full((1, 3), np.inf)
    search_X, idx_cache = _get_search_points(
        number_of_points=number_of_points,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    assert np.all(np.isnan(idx_cache))
    box_lb = -0.5 - 3
    box_ub = 0.5 + 3
    assert np.all(search_X == random_values * (box_ub - box_lb) + box_lb)

    # finite bounds
    vbmc.optim_state["LB_search"] = np.full((1, 3), -3000)
    vbmc.optim_state["UB_search"] = np.full((1, 3), 3000)
    search_X, idx_cache = _get_search_points(
        number_of_points=number_of_points,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    assert np.all(np.isnan(idx_cache))
    box_lb = -4.5
    box_ub = 13.5
    assert np.all(search_X == random_values * (box_ub - box_lb) + box_lb)


def test_get_search_points_all_hpd_search(mocker):
    """
    Take all points from hpd search.
    """
    user_options = {
        "cachefrac": 1,
        "searchcachefrac": 0,
        "heavytailsearchfrac": 0,
        "mvnsearchfrac": 0,
        "boxsearchfrac": 0,
        "hpdsearchfrac": 1,
        "hpdfrac": 0.8,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    number_of_points = 2
    X = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.zeros(0)

    # record some samples in FunctionLogger
    for i in range(10):
        vbmc.function_logger(np.ones(3) * i)
    assert vbmc.function_logger.Xn == 9

    # make sure that all samples are from hpd search (disable final vp.sample)
    mocker.patch(
        "pyvbmc.variational_posterior.VariationalPosterior.sample",
        return_value=np.zeros(0),
    )

    # no search bounds for test
    vbmc.optim_state["LB_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["UB_search"] = np.full((1, 3), np.inf)
    search_X, idx_cache = _get_search_points(
        number_of_points=number_of_points,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    assert np.all(np.isnan(idx_cache))


def test_get_search_points_all_hpd_search_empty_get_hpd(mocker):
    """
    Take all points from hpd search when when get_hpd returns an empty array.
    """

    user_options = {
        "cachefrac": 1,
        "searchcachefrac": 0,
        "heavytailsearchfrac": 0,
        "mvnsearchfrac": 0,
        "boxsearchfrac": 0,
        "hpdsearchfrac": 1,
        "hpdfrac": 0
        }
    
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    number_of_points = 2
    X = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.zeros(0)

    # record some samples in FunctionLogger
    for i in range(10):
        vbmc.function_logger(np.ones(3) * i)
    assert vbmc.function_logger.Xn == 9

    # make sure that all samples are from hpd search (disable final vp.sample)
    mocker.patch(
        "pyvbmc.variational_posterior.VariationalPosterior.sample",
        return_value=np.zeros(0),
    )

    # no search bounds for test
    vbmc.optim_state["LB_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["UB_search"] = np.full((1, 3), np.inf)
    search_X, idx_cache = _get_search_points(
        number_of_points=number_of_points,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    assert np.all(np.isnan(idx_cache))


def test_get_search_points_more_points_randomly_than_requested():
    """
    Test that ValueError is raised when options lead to more points sampled than
    requested.
    """
    user_options = {
        "cachefrac": 0,
        "searchcachefrac": 0,
        "heavytailsearchfrac": 1,
        "mvnsearchfrac":1,
        "boxsearchfrac": 1,
        "hpdsearchfrac": 1,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, user_options)
    number_of_points = 100
    vbmc.optim_state["cache"]["x_orig"] = np.zeros(0)

    # no search bounds for test
    vbmc.optim_state["LB_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["UB_search"] = np.full((1, 3), np.inf)

    with pytest.raises(ValueError):
        _get_search_points(
            number_of_points=number_of_points,
            optim_state=vbmc.optim_state,
            function_logger=vbmc.function_logger,
            vp=vbmc.vp,
            options=vbmc.options,
        )