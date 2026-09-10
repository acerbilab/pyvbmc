import copy
import importlib
import logging

import numpy as np
import pytest

from pyvbmc import VBMC
from pyvbmc.acquisition_functions import AbstractAcqFcn, AcqFcnEIG
from pyvbmc.stats import get_hpd
from pyvbmc.vbmc import active_sample
from pyvbmc.vbmc.active_sample import _get_search_points
from pyvbmc.vbmc.gaussian_process_train import train_gp

fun = lambda x: np.sum(x + 2)


def create_vbmc(
    D: int,
    x0: float,
    lower_bounds: float,
    upper_bounds: float,
    plausible_lower_bounds: float,
    plausible_upper_bounds: float,
    options: dict = None,
):
    lb = np.ones((1, D)) * lower_bounds
    ub = np.ones((1, D)) * upper_bounds
    x0_array = np.ones((2, D)) * x0
    plb = np.ones((1, D)) * plausible_lower_bounds
    pub = np.ones((1, D)) * plausible_upper_bounds
    return VBMC(fun, x0_array, lb, ub, plb, pub, options)


def test_active_uncertainty_sampling(mocker):
    def rosen(self, x, *args):
        x = np.atleast_2d(x)
        r = np.sum(
            100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2.0 + (1 - x[:, :-1]) ** 2,
            axis=1,
        )
        return np.log(r)

    D = 2
    LB = -np.full((1, D), np.inf)  # Lower bounds
    UB = np.full((1, D), np.inf)  # Upper bounds
    PLB = np.array([[-3.0, -3.0]])  # Plausible lower bounds
    PUB = np.array([[3.0, 3.0]])  # Plausible upper bounds
    x0 = np.zeros((1, D))
    options = {
        "active_sample_vp_update": True,
        "active_sample_gp_update": True,
    }
    vbmc = VBMC(fun, x0, LB, UB, PLB, PUB, options)
    mocker.patch("pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", rosen)
    N_init = 10
    function_logger, optim_state, _, _ = active_sample(
        gp=None,
        sample_count=N_init,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        iteration_history=vbmc.iteration_history,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    optim_state["N"] = function_logger.Xn + 1
    optim_state["n_eff"] = np.sum(
        function_logger.n_evals[function_logger.X_flag]
    )
    gp, _, _, hyp_dict = train_gp(
        {},
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.options,
        vbmc.plausible_lower_bounds,
        vbmc.plausible_upper_bounds,
    )
    optim_state["hyp_dict"] = hyp_dict
    optim_state["vp_repo"] = []
    sample_count = 2
    function_logger, _, _, _ = active_sample(
        gp,
        sample_count,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    # Minimum of Rosenbrock function
    assert np.allclose(
        function_logger.X[N_init : N_init + sample_count, :], 1, atol=0.001
    )
    return


def test_active_sample_rollback_preserves_live_transformer(mocker):
    active_sample_module = importlib.import_module("pyvbmc.vbmc.active_sample")
    vbmc = create_vbmc(1, 0, -np.inf, np.inf, -2, 2)
    transformer = vbmc.function_logger.parameter_transformer
    for key, value in {
        "active_sample_vp_update": True,
        "active_sample_gp_update": False,
        "search_optimizer": "none",
        "ns_search": 1,
    }.items():
        vbmc.options.__setitem__(key, value, force=True)
    vbmc.optim_state["iter"] = 0
    vbmc.optim_state["last_warmup"] = 0
    vbmc.optim_state["hyp_dict"] = {}

    acq = mocker.Mock()
    acq.acq_info = {}
    acq.return_value = np.array([0.0])
    vbmc.options.__setitem__("search_acq_fcn", [acq], force=True)
    mocker.patch.object(
        active_sample_module,
        "_get_search_points",
        side_effect=[
            (np.array([[0.1]]), np.array([np.nan])),
            (np.array([[0.2]]), np.array([np.nan])),
        ],
    )

    gp = mocker.Mock()
    gp.D = 1
    gp.X = np.array([[0.0]])
    gp.y = np.array([[0.0]])
    gp.posteriors = [mocker.Mock(hyp=np.zeros(2))]
    gp.covariance.hyperparameter_count.return_value = 1
    gp.noise.hyperparameter_count.return_value = 1
    gp.noise.compute.return_value = np.ones(1)
    gp.temporary_data = {}
    mocker.patch.object(active_sample_module, "reupdate_gp", return_value=gp)

    old_parameters = vbmc.vp.get_parameters().copy()
    updated_vp = copy.deepcopy(vbmc.vp)
    assert updated_vp.parameter_transformer is not transformer
    assert updated_vp.rng is vbmc.vp.rng
    updated_vp.parameter_transformer = transformer
    updated_vp.mu += 1.0
    updated_vp.stats = {"elbo": -1.0}
    mocker.patch.object(
        active_sample_module,
        "optimize_vp",
        return_value=(updated_vp, 0.0, 0),
    )
    old_score = mocker.patch.object(
        active_sample_module, "_neg_elcbo", return_value=(-0.0,)
    )

    _, _, returned_vp, _ = active_sample(
        gp,
        2,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    old_score.assert_called_once()
    assert np.array_equal(returned_vp.get_parameters(), old_parameters)
    assert returned_vp.parameter_transformer is transformer


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

    function_logger, optim_state, _, _ = active_sample(
        gp=None,
        sample_count=sample_count,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        iteration_history=vbmc.iteration_history,
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

    function_logger, optim_state, _, _ = active_sample(
        gp=None,
        sample_count=sample_count,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        iteration_history=vbmc.iteration_history,
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
    init_design is plausible.
    """
    options = {"init_design": "plausible"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    provided_sample_count = 10
    sample_count = provided_sample_count + 102
    X_orig = np.linspace((1, 11, 21), (10, 20, 30), provided_sample_count)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(X_orig)
    y_orig = [fun(x) for x in X_orig]
    vbmc.optim_state["cache"]["y_orig"] = np.copy(y_orig)

    assert not np.all(np.isnan(vbmc.optim_state["cache"]["x_orig"][:10]))
    assert not np.all(np.isnan(vbmc.optim_state["cache"]["y_orig"][:10]))

    # return a linespace so that random_Xs is with mean 0
    random_values = np.linspace(
        (-100, -100, -100),
        (100, 100, 100),
        sample_count - provided_sample_count,
    )
    mocker.patch.object(
        vbmc.vp, "_rng", mocker.Mock(**{"random.return_value": random_values})
    )
    function_logger, optim_state, _, _ = active_sample(
        gp=None,
        sample_count=sample_count,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        iteration_history=vbmc.iteration_history,
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
    init_design is plausible.
    """
    options = {"init_design": "narrow"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    provided_sample_count = 10
    sample_count = provided_sample_count + 102
    X_orig = np.linspace((0, 0, 0), (10, 10, 10), provided_sample_count)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(X_orig)
    y_orig = [fun(x) for x in X_orig]
    vbmc.optim_state["cache"]["y_orig"] = np.copy(y_orig)

    assert not np.all(np.isnan(vbmc.optim_state["cache"]["x_orig"][:10]))
    assert not np.all(np.isnan(vbmc.optim_state["cache"]["y_orig"][:10]))

    # return a linespace so that random_Xs is with mean 0
    random_values = np.linspace(
        (-0.1, -0.1, -0.1),
        (0.1, 0.1, 0.1),
        sample_count - provided_sample_count,
    )
    mocker.patch.object(
        vbmc.vp, "_rng", mocker.Mock(**{"random.return_value": random_values})
    )
    function_logger, optim_state, _, _ = active_sample(
        gp=None,
        sample_count=sample_count,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        iteration_history=vbmc.iteration_history,
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
    init_design is unknown.
    """
    options = {"init_design": "unknown"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    sample_count = 10

    with pytest.raises(ValueError) as execinfo:
        active_sample(
            gp=None,
            sample_count=sample_count,
            optim_state=vbmc.optim_state,
            function_logger=vbmc.function_logger,
            iteration_history=vbmc.iteration_history,
            vp=vbmc.vp,
            options=vbmc.options,
        )
    assert "Unknown initial design for VBMC" in execinfo.value.args[0]


def test_active_sample_logger():
    """
    Test logging levels for various options.
    """
    # iter which is INFO
    options = {"display": "iter"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    active_sample(
        gp=None,
        sample_count=1,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        iteration_history=vbmc.iteration_history,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    assert logging.getLogger("ActiveSample").getEffectiveLevel() == 20

    # off which is WARN
    options = {"display": "off"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    active_sample(
        gp=None,
        sample_count=1,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        iteration_history=vbmc.iteration_history,
        vp=vbmc.vp,
        options=vbmc.options,
    )

    # full which is DEBUG
    options = {"display": "full"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    active_sample(
        gp=None,
        sample_count=1,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        iteration_history=vbmc.iteration_history,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    assert logging.getLogger("ActiveSample").getEffectiveLevel() == 10

    # anything else is also INFO
    options = {"display": "strange_option"}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    active_sample(
        gp=None,
        sample_count=1,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        iteration_history=vbmc.iteration_history,
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
    function_logger, optim_state, _, _ = active_sample(
        gp=None,
        sample_count=sample_count,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        iteration_history=vbmc.iteration_history,
        vp=vbmc.vp,
        options=vbmc.options,
    )

    logger_message = "More than sample_count=90 initial points have been "
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


@pytest.mark.parametrize("D", [1, 2])
@pytest.mark.parametrize("noisy", [False, True])
def test_vectorized_initial_design_matches_scalar(D, noisy):
    scalar_calls = []
    vector_calls = []

    def scalar_target(x):
        scalar_calls.append(x.copy())
        value = np.sum(x**2 + 2)
        return (value, 0.5) if noisy else value

    def vector_target(x):
        vector_calls.append(x.copy())
        values = np.sum(x**2 + 2, axis=1)
        return (values, np.full(x.shape[0], 0.5)) if noisy else values

    x0 = np.vstack((np.zeros(D), np.full(D, 0.25)))
    bounds = np.full((1, D), np.inf)
    plausible = np.full((1, D), 1.0)
    common_options = {
        "f_vals": [2 * D, np.nan],
        "specify_target_noise": noisy,
    }
    scalar = VBMC(
        scalar_target,
        x0,
        -bounds,
        bounds,
        -plausible,
        plausible,
        options=common_options,
        seed=1234,
    )
    vector = VBMC(
        vector_target,
        x0,
        -bounds,
        bounds,
        -plausible,
        plausible,
        options={**common_options, "vectorized_target": True},
        seed=1234,
    )

    for vbmc in (scalar, vector):
        active_sample(
            gp=None,
            sample_count=5,
            optim_state=vbmc.optim_state,
            function_logger=vbmc.function_logger,
            iteration_history=vbmc.iteration_history,
            vp=vbmc.vp,
            options=vbmc.options,
        )

    scalar_logger = scalar.function_logger
    vector_logger = vector.function_logger
    for name in ("X", "X_orig", "y", "y_orig", "n_evals", "X_flag"):
        assert np.array_equal(
            getattr(scalar_logger, name),
            getattr(vector_logger, name),
            equal_nan=True,
        )
    if noisy:
        assert np.array_equal(scalar_logger.S, vector_logger.S, equal_nan=True)
    assert scalar_logger.func_count == vector_logger.func_count == 4
    assert scalar_logger.cache_count == vector_logger.cache_count == 1
    assert (
        scalar.vp.rng.bit_generator.state == vector.vp.rng.bit_generator.state
    )
    assert len(scalar_calls) == 4
    assert all(call.shape == (D,) for call in scalar_calls)
    assert len(vector_calls) == 1
    assert vector_calls[0].shape == (4, D)


def test_vectorized_initial_design_all_values_cached():
    calls = []

    def target(x):
        calls.append(x)
        return np.sum(x, axis=1)

    x0 = np.array([[0.0], [0.5], [1.0]])
    vbmc = VBMC(
        target,
        x0,
        np.array([[-np.inf]]),
        np.array([[np.inf]]),
        np.array([[-1.0]]),
        np.array([[1.0]]),
        options={"vectorized_target": True, "f_vals": [0.0, 0.5, 1.0]},
        seed=1234,
    )
    state_before = vbmc.vp.rng.bit_generator.state

    active_sample(
        gp=None,
        sample_count=3,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        iteration_history=vbmc.iteration_history,
        vp=vbmc.vp,
        options=vbmc.options,
    )

    assert calls == []
    assert vbmc.function_logger.func_count == 0
    assert vbmc.function_logger.cache_count == 3
    assert vbmc.vp.rng.bit_generator.state == state_before


def test_get_search_points_all_cache():
    """
    Take all points from cache.
    """
    options = {"cache_frac": 1}
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    number_of_points = 2
    x_orig = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(x_orig)

    # no search bounds for test
    vbmc.optim_state["lb_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["ub_search"] = np.full((1, 3), np.inf)
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
    options = {
        "cache_frac": 1,
        "search_cache_frac": 1,
        "heavy_tail_search_frac": 0,
        "mvn_search_frac": 0,
        "box_search_frac": 0,
        "hpd_search_frac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    number_of_points = 2
    X = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.zeros(0)
    vbmc.optim_state["search_cache"] = np.copy(X)

    # no search bounds for test
    vbmc.optim_state["lb_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["ub_search"] = np.full((1, 3), np.inf)
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
    options = {
        "cache_frac": 1,
        "search_cache_frac": 0,
        "heavy_tail_search_frac": 0,
        "mvn_search_frac": 0,
        "box_search_frac": 0,
        "hpd_search_frac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    number_of_points = 2
    X = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(X)

    # no search bounds for test
    vbmc.optim_state["lb_search"] = np.full((1, 3), 2)
    vbmc.optim_state["ub_search"] = np.full((1, 3), 4)
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
    options = {
        "cache_frac": 1,
        "search_cache_frac": 0,
        "heavy_tail_search_frac": 1,
        "mvn_search_frac": 0,
        "box_search_frac": 0,
        "hpd_search_frac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    number_of_points = 2
    X = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.zeros(0)

    # no search bounds for test
    vbmc.optim_state["lb_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["ub_search"] = np.full((1, 3), np.inf)
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
    options = {
        "cache_frac": 1,
        "search_cache_frac": 0,
        "heavy_tail_search_frac": 0,
        "mvn_search_frac": 1,
        "box_search_frac": 0,
        "hpd_search_frac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    number_of_points = 2
    X = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.zeros(0)

    # no search bounds for test
    vbmc.optim_state["lb_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["ub_search"] = np.full((1, 3), np.inf)
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
    options = {
        "cache_frac": 1,
        "search_cache_frac": 0,
        "heavy_tail_search_frac": 0,
        "mvn_search_frac": 0,
        "box_search_frac": 0,
        "hpd_search_frac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    number_of_points = 2
    X = np.linspace((0, 0, 0), (10, 10, 10), number_of_points)
    vbmc.optim_state["cache"]["x_orig"] = np.zeros(0)

    # no search bounds for test
    vbmc.optim_state["lb_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["ub_search"] = np.full((1, 3), np.inf)
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
    options = {
        "cache_frac": 1,
        "search_cache_frac": 0,
        "heavy_tail_search_frac": 0,
        "mvn_search_frac": 0,
        "box_search_frac": 1,
        "hpd_search_frac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
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
    mocker.patch.object(
        vbmc.vp,
        "_rng",
        mocker.Mock(**{"standard_normal.return_value": random_values}),
    )
    # infinite bounds
    vbmc.optim_state["lb_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["ub_search"] = np.full((1, 3), np.inf)
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
    vbmc.optim_state["lb_search"] = np.full((1, 3), -3000)
    vbmc.optim_state["ub_search"] = np.full((1, 3), 3000)
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
    options = {
        "cache_frac": 1,
        "search_cache_frac": 0,
        "heavy_tail_search_frac": 0,
        "mvn_search_frac": 0,
        "box_search_frac": 0,
        "hpd_search_frac": 1,
        "hpd_frac": 0.8,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
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
    vbmc.optim_state["lb_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["ub_search"] = np.full((1, 3), np.inf)
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

    options = {
        "cache_frac": 1,
        "search_cache_frac": 0,
        "heavy_tail_search_frac": 0,
        "mvn_search_frac": 0,
        "box_search_frac": 0,
        "hpd_search_frac": 1,
        "hpd_frac": 0,
    }

    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
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
    vbmc.optim_state["lb_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["ub_search"] = np.full((1, 3), np.inf)
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
    options = {
        "cache_frac": 0,
        "search_cache_frac": 0,
        "heavy_tail_search_frac": 1,
        "mvn_search_frac": 1,
        "box_search_frac": 1,
        "hpd_search_frac": 1,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    number_of_points = 100
    vbmc.optim_state["cache"]["x_orig"] = np.zeros(0)

    # record some samples in FunctionLogger
    for i in range(10):
        vbmc.function_logger(np.ones(3) * i)
    assert vbmc.function_logger.Xn == 9

    # no search bounds for test
    vbmc.optim_state["lb_search"] = np.full((1, 3), -np.inf)
    vbmc.optim_state["ub_search"] = np.full((1, 3), np.inf)

    with pytest.raises(ValueError) as execinfo:
        _get_search_points(
            number_of_points=number_of_points,
            optim_state=vbmc.optim_state,
            function_logger=vbmc.function_logger,
            vp=vbmc.vp,
            options=vbmc.options,
        )

    assert "A maximum of 100 points" in execinfo.value.args[0]


def test_repeated_observation_candidates(mocker):
    """With observation noise and ``max_repeated_observations > 0``, the
    training inputs are search candidates; a chosen repeat is pooled into
    its row by the logger instead of adding one, and the streak cap
    excludes the training inputs once reached."""
    D = 2
    rng = np.random.default_rng(0)

    def noisy_target(x):
        x = np.atleast_2d(x)
        return -0.5 * np.sum(x**2) + rng.normal(), 1.0

    vbmc = VBMC(
        noisy_target,
        np.zeros((1, D)),
        -np.full((1, D), np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -3.0),
        np.full((1, D), 3.0),
        {
            "specify_target_noise": True,
            "max_repeated_observations": 2,
            "search_optimizer": "none",
            "active_sample_gp_update": False,
            "active_sample_vp_update": False,
        },
    )

    # An acquisition that prefers the training inputs (exact rows).
    def prefer_training(self, Xs, gp, vp, function_logger, optim_state):
        Xs = np.atleast_2d(Xs)
        X_train = function_logger.X[function_logger.X_flag]
        is_train = np.array([np.any(np.all(X_train == x, axis=1)) for x in Xs])
        return np.where(is_train, 0.0, 1.0)

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__",
        prefer_training,
    )
    N_init = 10
    function_logger, optim_state, _, _ = active_sample(
        None,
        N_init,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    optim_state["N"] = function_logger.Xn + 1
    optim_state["n_eff"] = np.sum(
        function_logger.n_evals[function_logger.X_flag]
    )
    gp, _, _, hyp_dict = train_gp(
        {},
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.options,
        vbmc.plausible_lower_bounds,
        vbmc.plausible_upper_bounds,
    )
    optim_state["hyp_dict"] = hyp_dict
    Xn0 = function_logger.Xn
    evals0 = np.sum(function_logger.n_evals[function_logger.X_flag])

    # Two picks: both repeats, no new row, the streak reaches the cap.
    function_logger, optim_state, _, gp = active_sample(
        gp,
        2,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    assert function_logger.Xn == Xn0
    assert (
        np.sum(function_logger.n_evals[function_logger.X_flag]) == evals0 + 2
    )
    assert np.max(function_logger.n_evals[function_logger.X_flag]) >= 2
    assert optim_state["repeated_observations_streak"] == 2

    # At the cap the training inputs are not candidates: a new row, and the
    # streak resets.
    function_logger, optim_state, _, gp = active_sample(
        gp,
        1,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    assert function_logger.Xn == Xn0 + 1
    assert optim_state["repeated_observations_streak"] == 0


def test_repeated_observation_candidates_off_by_default(mocker):
    """``max_repeated_observations = 0``: the training inputs are not
    candidates even when the acquisition would prefer them."""
    D = 2
    rng = np.random.default_rng(0)

    def noisy_target(x):
        x = np.atleast_2d(x)
        return -0.5 * np.sum(x**2) + rng.normal(), 1.0

    vbmc = VBMC(
        noisy_target,
        np.zeros((1, D)),
        -np.full((1, D), np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -3.0),
        np.full((1, D), 3.0),
        {
            "specify_target_noise": True,
            "search_optimizer": "none",
            "active_sample_gp_update": False,
            "active_sample_vp_update": False,
        },
    )
    assert vbmc.options["max_repeated_observations"] == 0

    def prefer_training(self, Xs, gp, vp, function_logger, optim_state):
        Xs = np.atleast_2d(Xs)
        X_train = function_logger.X[function_logger.X_flag]
        is_train = np.array([np.any(np.all(X_train == x, axis=1)) for x in Xs])
        return np.where(is_train, 0.0, 1.0)

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__",
        prefer_training,
    )
    function_logger, optim_state, _, _ = active_sample(
        None,
        10,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    optim_state["N"] = function_logger.Xn + 1
    optim_state["n_eff"] = np.sum(
        function_logger.n_evals[function_logger.X_flag]
    )
    gp, _, _, hyp_dict = train_gp(
        {},
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.options,
        vbmc.plausible_lower_bounds,
        vbmc.plausible_upper_bounds,
    )
    optim_state["hyp_dict"] = hyp_dict
    Xn0 = function_logger.Xn
    function_logger, optim_state, _, _ = active_sample(
        gp,
        1,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    assert function_logger.Xn == Xn0 + 1
    assert optim_state["repeated_observations_streak"] == 0


def _noisy_run(
    mocker, user_options, acq=None, lower_bounds=None, upper_bounds=None
):
    """A noisy ``VBMC`` on a quadratic target with the given options, its
    initial design drawn and a GP trained; ``acq`` replaces the
    acquisition wrapper (``AbstractAcqFcn.__call__``) when given."""
    D = 2
    rng = np.random.default_rng(0)

    def noisy_target(x):
        x = np.atleast_2d(x)
        return -0.5 * np.sum(x**2) + rng.normal(), 1.0

    vbmc = VBMC(
        noisy_target,
        np.zeros((1, D)),
        -np.full((1, D), np.inf) if lower_bounds is None else lower_bounds,
        np.full((1, D), np.inf) if upper_bounds is None else upper_bounds,
        np.full((1, D), -3.0),
        np.full((1, D), 3.0),
        {
            "specify_target_noise": True,
            "active_sample_gp_update": False,
            "active_sample_vp_update": False,
            **user_options,
        },
    )
    if acq is not None:
        mocker.patch(
            "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", acq
        )
    function_logger, optim_state, _, _ = active_sample(
        None,
        10,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    optim_state["N"] = function_logger.Xn + 1
    optim_state["n_eff"] = np.sum(
        function_logger.n_evals[function_logger.X_flag]
    )
    gp, _, _, hyp_dict = train_gp(
        {},
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.options,
        vbmc.plausible_lower_bounds,
        vbmc.plausible_upper_bounds,
    )
    optim_state["hyp_dict"] = hyp_dict
    return vbmc, gp, function_logger, optim_state


def _prefer_training(self, Xs, gp, vp, function_logger, optim_state):
    """An acquisition that prefers the training inputs (exact rows)."""
    Xs = np.atleast_2d(Xs)
    X_train = function_logger.X[function_logger.X_flag]
    is_train = np.array([np.any(np.all(X_train == x, axis=1)) for x in Xs])
    return np.where(is_train, 0.0, 1.0)


def test_repeated_observation_is_exact_with_integer_vars(mocker):
    """With an integer variable the acquisition snaps its input in place
    and a row of the initial design is not snapped, so the candidate that
    scores as a repeat is a near-duplicate of the stored row; the
    evaluated point must be the stored row itself, pooled by the logger,
    not a new training input."""

    def snapped(X, function_logger, optim_state):
        return AbstractAcqFcn._real2int(
            np.array(X, dtype=float),
            function_logger.parameter_transformer,
            optim_state["integer_vars"],
        )

    def snapping_prefer_target(self, Xs, gp, vp, function_logger, optim_state):
        Xs = np.atleast_2d(Xs)
        Xs = AbstractAcqFcn._real2int(  # in place, as the wrapper does
            Xs,
            function_logger.parameter_transformer,
            optim_state["integer_vars"],
        )
        X_train = function_logger.X[function_logger.X_flag]
        X_train_snapped = snapped(X_train, function_logger, optim_state)
        unsnapped = np.any(X_train != X_train_snapped, axis=1)
        target = X_train_snapped[unsnapped][:1]
        return np.where(np.all(Xs == target, axis=1), 0.0, 1.0)

    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker,
        {
            "integer_vars": np.array([1, 0]),
            "max_repeated_observations": 2,
            "search_optimizer": "none",
        },
        snapping_prefer_target,
        lower_bounds=np.array([[-10.5, -np.inf]]),
        upper_bounds=np.array([[10.5, np.inf]]),
    )
    X_train = function_logger.X[function_logger.X_flag].copy()
    assert np.any(X_train != snapped(X_train, function_logger, optim_state))
    Xn0 = function_logger.Xn
    evals0 = np.sum(function_logger.n_evals[function_logger.X_flag])

    function_logger, optim_state, _, gp = active_sample(
        gp,
        1,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    assert function_logger.Xn == Xn0
    assert (
        np.sum(function_logger.n_evals[function_logger.X_flag]) == evals0 + 1
    )
    assert optim_state["repeated_observations_streak"] == 1
    assert np.array_equal(function_logger.X[function_logger.X_flag], X_train)


def test_repeated_observation_skips_search_optimizer(mocker):
    """A chosen repeat skips the local optimizer (which would move it off
    the stored row); once the cap excludes the training inputs the
    optimizer runs again."""
    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker,
        {"max_repeated_observations": 2, "search_optimizer": "cmaes"},
        _prefer_training,
    )
    Xn0 = function_logger.Xn
    fmin = mocker.patch(
        "cma.fmin", side_effect=AssertionError("optimizer ran on a repeat")
    )
    function_logger, optim_state, _, gp = active_sample(
        gp,
        2,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    assert function_logger.Xn == Xn0
    assert optim_state["repeated_observations_streak"] == 2
    assert fmin.call_count == 0

    # At the cap: a non-repeat, and the optimizer is called (its result
    # is worse, so the sieve point stays).
    fmin = mocker.patch(
        "cma.fmin", side_effect=lambda f, x0, *a, **k: (x0, np.inf)
    )
    function_logger, optim_state, _, gp = active_sample(
        gp,
        1,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    assert function_logger.Xn == Xn0 + 1
    assert fmin.call_count == 1


def test_ns_gp_max_active_caps_the_in_loop_refits(mocker):
    """``ns_gp_max_active`` caps the number of hyperparameter samples of
    the GP refits within active sampling, as ``ns_gp_max_warmup`` and
    ``ns_gp_max_main`` cap the main fit's, and leaves the main options
    alone."""
    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker, {"search_optimizer": "none"}, _prefer_training
    )
    vbmc.options.__setitem__("active_sample_gp_update", True, force=True)
    vbmc.options.__setitem__("ns_gp_max_active", 0, force=True)
    seen = []

    def recording_train_gp(
        hyp_dict, optim_state, logger, history, options, *a, **k
    ):
        seen.append(
            (
                options["ns_gp_max_warmup"],
                options["ns_gp_max_main"],
                options["ns_gp_max"],
            )
        )
        return train_gp(
            hyp_dict, optim_state, logger, history, options, *a, **k
        )

    mocker.patch("pyvbmc.vbmc.active_sample.train_gp", recording_train_gp)
    _, _, _, gp = active_sample(
        gp,
        2,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    assert len(seen) == 1  # one refit between the two points
    assert all(caps[:2] == (0, 0) for caps in seen)
    assert all(caps[2] == vbmc.options["ns_gp_max"] for caps in seen)
    assert len(gp.posteriors) == 1  # the MAP fit
    assert vbmc.options["ns_gp_max_warmup"] == 8
    assert vbmc.options["ns_gp_max_main"] == np.inf


def test_eig_through_active_sample(mocker):
    """The expected information gain runs through ``active_sample``: the
    ``compute_var_log_joint`` hook stores the variance of the expected
    log joint per hyperparameter sample and the covariance of the
    components' integrals, and points are acquired."""
    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker, {"search_optimizer": "none"}
    )
    vbmc.options.__setitem__(
        "search_acq_fcn", [AcqFcnEIG(components=True)], force=True
    )
    Xn0 = function_logger.Xn
    function_logger, optim_state, _, gp = active_sample(
        gp,
        2,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    Ns = len(gp.posteriors)
    K = vbmc.vp.K
    assert np.shape(optim_state["var_log_joint_samples"]) == (Ns,)
    assert np.shape(optim_state["cov_log_joint_components"]) == (Ns, K, K)
    assert np.all(np.isfinite(optim_state["cov_log_joint_components"]))
    assert function_logger.Xn == Xn0 + 2
