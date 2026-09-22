import copy
import importlib
import logging
import math

import cma
import gpyreg as gpr
import numpy as np
import pytest
import scipy.optimize

from pyvbmc import VBMC
from pyvbmc.acquisition_functions import AbstractAcqFcn
from pyvbmc.stats import get_hpd
from pyvbmc.timer import main_timer
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc import active_sample
from pyvbmc.vbmc.active_sample import _get_search_points
from pyvbmc.vbmc.gaussian_process_train import reupdate_gp, train_gp
from pyvbmc.whitening import warp_gp_and_vp, warp_input

# The module, for patching its names. Inside the package the name
# `pyvbmc.vbmc.active_sample` is the function the package exports, and a
# dotted patch target that goes through it does not resolve on Python 3.10.
_active_sample_module = importlib.import_module("pyvbmc.vbmc.active_sample")

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


def _cheap_acq(self, x, *args):
    """A quadratic standing in for an acquisition function."""
    return np.sum(np.atleast_2d(x) ** 2, axis=1)


def _state_with_gp(
    D: int,
    options: dict = None,
    seed: int = None,
    lower_bound: float = -np.inf,
    upper_bound: float = np.inf,
):
    """Build a VBMC instance and a GP trained on one initial design."""
    vbmc = create_vbmc(D, 0.0, lower_bound, upper_bound, -3, 3, options)
    if seed is not None:
        vbmc.vp.rng = np.random.default_rng(seed)
    function_logger, optim_state, _, _ = active_sample(
        gp=None,
        sample_count=10,
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
        rng=vbmc.vp.rng,
    )
    optim_state["hyp_dict"] = hyp_dict
    return vbmc, gp


def test_cmaes_search_starts_from_per_coordinate_step_sizes(mocker):
    """CMA-ES starts at the per-coordinate scales of the search covariance."""
    D = 2
    vbmc, gp = _state_with_gp(D)

    # An anisotropic variational posterior, so that the per-coordinate
    # standard deviations differ from their maximum.
    vp = vbmc.vp
    vp.mu = np.zeros((D, vp.K))
    vp.sigma = np.ones((1, vp.K))
    vp.lambd = np.array([[1.0], [0.05]])
    vp.w = np.full((1, vp.K), 1 / vp.K)
    _, Sigma = vp.moments(orig_flag=False, cov_flag=True)
    insigma = np.sqrt(np.diag(Sigma))
    assert insigma.max() / insigma.min() > 5

    captured = {}

    def fake_fmin(objective, x0, sigma0, options=None, **kwargs):
        captured["x0"] = np.asarray(x0, dtype=float)
        captured["sigma0"] = sigma0
        captured["options"] = dict(options)
        # A rejected search result: the sieve's point is kept.
        return np.asarray(x0, dtype=float), np.inf

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", _cheap_acq
    )
    mocker.patch("cma.fmin", side_effect=fake_fmin)

    active_sample(
        gp,
        1,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vp,
        vbmc.options,
    )

    assert captured["sigma0"] == pytest.approx(insigma.max())
    assert np.allclose(
        captured["options"]["CMA_stds"], insigma / insigma.max()
    )

    # Those arguments give an initial population whose per-coordinate
    # standard deviations are `insigma`. The captured search bounds are
    # left out of the check, since cma repairs the drawn points into them.
    rng = np.random.default_rng(42)
    es = cma.CMAEvolutionStrategy(
        captured["x0"],
        captured["sigma0"],
        {
            "verbose": -9,
            "CMA_stds": captured["options"]["CMA_stds"],
            "seed": np.nan,
            "randn": lambda *shape: rng.standard_normal(shape),
        },
    )
    population = np.asarray(es.ask(4000))
    assert np.allclose(population.std(axis=0), insigma, rtol=0.1)


def test_cmaes_search_runs_without_noise_handling(mocker):
    """The CMA-ES search spends one population per generation."""

    def rosen(self, x, *args):
        x = np.atleast_2d(x)
        return np.log(
            np.sum(
                100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2.0
                + (1 - x[:, :-1]) ** 2,
                axis=1,
            )
        )

    vbmc, gp = _state_with_gp(2, seed=20260919)
    real_fmin = cma.fmin
    captured = {}

    def fake_fmin(objective, x0, sigma0, options=None, **kwargs):
        captured["objective"] = objective
        captured["x0"] = np.asarray(x0, dtype=float)
        captured["sigma0"] = sigma0
        captured["options"] = dict(options)
        captured["kwargs"] = kwargs
        # A rejected search result: the sieve's point is kept.
        return np.asarray(x0, dtype=float), np.inf

    mocker.patch("pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", rosen)
    mocker.patch("cma.fmin", side_effect=fake_fmin)

    active_sample(
        gp,
        1,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    assert captured["kwargs"].get("noise_handler") is None

    # Run that search: one batched call of `popsize` points per
    # generation, and no re-evaluations on top of them.
    batch_sizes = []

    def counting(X):
        batch_sizes.append(
            1 if isinstance(X, np.ndarray) and X.ndim == 1 else len(X)
        )
        return captured["objective"](X)

    res = real_fmin(
        counting,
        captured["x0"],
        captured["sigma0"],
        options=captured["options"],
        parallel_objective=counting,
    )
    es = res[-2]
    generations = res[4]
    assert generations > 1
    assert [n for n in batch_sizes if n > 1] == [es.popsize] * generations


def test_search_bounds_fallback_is_one_bound_per_coordinate(mocker):
    """With a non-finite search bound the local search is bounded by the
    training inputs and the starting point, one bound per coordinate."""
    D = 2
    vbmc, gp = _state_with_gp(D)
    vbmc.optim_state["ub_search"][0, 0] = np.inf
    assert gp.X.shape[0] > 1

    captured = {}

    def fake_fmin(objective, x0, sigma0, options=None, **kwargs):
        captured["options"] = dict(options)
        # A rejected search result: the sieve's point is kept.
        return np.asarray(x0, dtype=float), np.inf

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", _cheap_acq
    )
    mocker.patch("cma.fmin", side_effect=fake_fmin)

    active_sample(
        gp,
        1,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    lb_search, ub_search = captured["options"]["bounds"]
    assert np.shape(lb_search) == (D,)
    assert np.shape(ub_search) == (D,)
    assert np.all(lb_search <= gp.X.min(0))
    assert np.all(ub_search >= gp.X.max(0))


def test_one_dimensional_search_is_bounded(mocker):
    """A one-dimensional acquisition is minimized over the search
    interval, within the search budget, and the acquired point is no
    worse than the best candidate of the search set."""
    vbmc, gp = _state_with_gp(1, seed=20260919)
    candidates = np.array([[0.7], [1.5], [-2.0]])
    captured = {}
    real_minimize_scalar = scipy.optimize.minimize_scalar

    def recording_minimize_scalar(fun, **kwargs):
        captured.update(kwargs)
        return real_minimize_scalar(fun, **kwargs)

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", _cheap_acq
    )
    mocker.patch.object(
        _active_sample_module,
        "_get_search_points",
        return_value=(candidates, np.full(len(candidates), np.nan)),
    )
    mocker.patch(
        "scipy.optimize.minimize_scalar",
        side_effect=recording_minimize_scalar,
    )
    optimizer_before = vbmc.options["search_optimizer"]

    function_logger, _, _, _ = active_sample(
        gp,
        1,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    lb, ub = captured["bounds"]
    assert captured["method"] == "bounded"
    assert lb < ub
    assert (
        captured["options"]["maxiter"] == vbmc.options["search_max_fun_evals"]
    )

    x_new = function_logger.X[function_logger.Xn]
    assert lb <= x_new <= ub
    # No worse than the best candidate of the search set.
    assert (
        np.sum(x_new**2) <= np.min(np.sum(candidates**2, axis=1)) + 1e-12
    )
    # The run's options are as they were.
    assert vbmc.options["search_optimizer"] == optimizer_before


def test_one_dimensional_search_is_skipped_without_a_local_optimizer(mocker):
    """``search_optimizer = "none"`` runs no local search, a problem of
    one variable included: the acquired point is the best candidate of the
    search set."""
    vbmc, gp = _state_with_gp(
        1, options={"search_optimizer": "none"}, seed=20260919
    )
    candidates = np.array([[0.7], [1.5], [-2.0]])

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", _cheap_acq
    )
    mocker.patch.object(
        _active_sample_module,
        "_get_search_points",
        return_value=(candidates, np.full(len(candidates), np.nan)),
    )
    mocker.patch(
        "scipy.optimize.minimize_scalar",
        side_effect=AssertionError("a local search ran"),
    )
    mocker.patch("cma.fmin", side_effect=AssertionError("a local search ran"))

    function_logger, _, _, _ = active_sample(
        gp,
        1,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    assert function_logger.X[function_logger.Xn] == candidates[0]


def test_a_search_optimizer_forced_into_a_built_instance_is_named(mocker):
    """Construction refuses every value but the two the option names, so
    the chain of local searches reaches its last branch only for a value
    written into the options of a built instance. It says which option
    carries the value and which two values it takes."""
    D = 2
    vbmc, gp = _state_with_gp(D, options={"ns_search": 4}, seed=20260921)
    vbmc.options.__setitem__("search_optimizer", "fmincon", force=True)

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", _cheap_acq
    )
    with pytest.raises(NotImplementedError) as execinfo:
        active_sample(
            gp,
            1,
            vbmc.optim_state,
            vbmc.function_logger,
            vbmc.iteration_history,
            vbmc.vp,
            vbmc.options,
        )

    message = execinfo.value.args[0]
    assert "search_optimizer" in message
    assert "'cmaes'" in message and "'none'" in message
    assert "fmincon" in message


def test_acquiring_a_cached_point_reuses_its_value(mocker):
    """A search point that comes from the cache is acquired with the
    value stored there, and leaves every cache array."""
    D = 2
    vbmc, gp = _state_with_gp(
        D,
        options={
            "ns_search": 1,
            "cache_frac": 1,
            "search_optimizer": "none",
        },
    )
    x_cached = np.array([[0.3, -0.2]])
    y_cached = np.array([12.5])
    vbmc.optim_state["cache"]["x_orig"] = np.copy(x_cached)
    vbmc.optim_state["cache"]["y_orig"] = np.copy(y_cached)
    vbmc.optim_state["cache"]["skip_logger"] = np.zeros(1, dtype=bool)
    func_count_before = vbmc.function_logger.func_count
    cache_count_before = vbmc.function_logger.cache_count

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", _cheap_acq
    )
    function_logger, optim_state, _, _ = active_sample(
        gp,
        1,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    # The stored value was used instead of a target call.
    assert function_logger.func_count == func_count_before
    assert function_logger.cache_count == cache_count_before + 1
    last = function_logger.Xn
    assert np.allclose(function_logger.X_orig[last], x_cached[0])
    assert np.allclose(function_logger.y_orig[last], y_cached[0])

    # The point is gone from the cache, which stays consistent.
    assert optim_state["cache"]["x_orig"].shape == (0, D)
    assert optim_state["cache"]["y_orig"].shape == (0,)
    assert optim_state["cache"]["skip_logger"].shape == (0,)


@pytest.mark.parametrize(
    "integer_vars, x_cached",
    [
        (np.array([True, False]), np.array([[2.0, 0.1]])),
        (np.array([True, True]), np.array([[2.0, -4.0]])),
    ],
)
def test_acquiring_a_cached_point_on_the_integer_grid_reuses_its_value(
    mocker, integer_vars, x_cached
):
    """The candidate that a cached starting point becomes has been
    through the transform, the snap to the integer grid and the inverse
    transform, and the stored value belongs to it only where the three
    left the point where it was. A point already on the grid is acquired
    with its value, without a call to the target."""
    D = 2
    vbmc, gp, _, _ = _integer_var_state(
        D,
        options={
            "ns_search": 1,
            "cache_frac": 1,
            "search_optimizer": "none",
            "integer_vars": integer_vars,
        },
    )
    y_cached = np.array([12.5])
    vbmc.optim_state["cache"]["x_orig"] = np.copy(x_cached)
    vbmc.optim_state["cache"]["y_orig"] = np.copy(y_cached)
    vbmc.optim_state["cache"]["skip_logger"] = np.zeros(1, dtype=bool)
    func_count_before = vbmc.function_logger.func_count
    cache_count_before = vbmc.function_logger.cache_count

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", _cheap_acq
    )
    function_logger, optim_state, _, _ = active_sample(
        gp,
        1,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    # The stored value was used instead of a target call.
    assert function_logger.func_count == func_count_before
    assert function_logger.cache_count == cache_count_before + 1
    last = function_logger.Xn
    assert np.allclose(function_logger.X_orig[last], x_cached[0])
    assert np.allclose(function_logger.y_orig[last], y_cached[0])

    # The point is gone from the cache, which stays consistent.
    assert optim_state["cache"]["x_orig"].shape == (0, D)
    assert optim_state["cache"]["y_orig"].shape == (0,)
    assert optim_state["cache"]["skip_logger"].shape == (0,)


def test_acquiring_a_cached_point_without_a_value_evaluates_it(mocker):
    """A search point that comes from the cache without a stored value is
    evaluated through the target and leaves the cache like one with a
    value, so it cannot be drawn and evaluated a second time."""
    D = 2
    vbmc, gp = _state_with_gp(
        D,
        options={
            "ns_search": 1,
            "cache_frac": 1,
            "search_optimizer": "none",
        },
    )
    x_cached = np.array([[0.3, -0.2]])
    vbmc.optim_state["cache"]["x_orig"] = np.copy(x_cached)
    vbmc.optim_state["cache"]["y_orig"] = np.array([np.nan])
    vbmc.optim_state["cache"]["skip_logger"] = np.zeros(1, dtype=bool)
    func_count_before = vbmc.function_logger.func_count
    cache_count_before = vbmc.function_logger.cache_count

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", _cheap_acq
    )
    function_logger, optim_state, _, _ = active_sample(
        gp,
        1,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    # The target was called once, for the cached point.
    assert function_logger.func_count == func_count_before + 1
    assert function_logger.cache_count == cache_count_before
    last = function_logger.Xn
    assert np.allclose(function_logger.X_orig[last], x_cached[0])
    assert np.allclose(function_logger.y_orig[last], fun(x_cached))

    # The point is gone from every cache array.
    assert optim_state["cache"]["x_orig"].shape == (0, D)
    assert optim_state["cache"]["y_orig"].shape == (0,)
    assert optim_state["cache"]["skip_logger"].shape == (0,)


def test_two_steps_with_a_search_cache(mocker):
    """The search cache holds the candidates of the previous step, so it
    is empty at the first step of a run and full from the second. Its
    share of the search set is taken from what the other fractions of the
    sieve leave, and both steps complete."""
    D = 2
    vbmc, gp = _state_with_gp(
        D,
        options={
            "ns_search": 32,
            "search_cache_frac": 0.25,
            "search_optimizer": "none",
        },
        seed=20260921,
    )
    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", _cheap_acq
    )
    function_logger, optim_state = vbmc.function_logger, vbmc.optim_state
    calls = function_logger.func_count
    assert np.size(optim_state["search_cache"]) == 0

    for step in range(2):
        function_logger, optim_state, _, gp = active_sample(
            gp,
            1,
            optim_state,
            function_logger,
            vbmc.iteration_history,
            vbmc.vp,
            vbmc.options,
        )
        assert function_logger.func_count == calls + step + 1
        assert np.shape(optim_state["search_cache"]) == (32, D)


def _acquire_one_cached_point(mocker, vbmc, gp, x_cached, y_cached):
    """One active-sampling step whose only candidate is the one row of the
    starting cache, with a value stored for it."""
    vbmc.optim_state["cache"]["x_orig"] = np.copy(x_cached)
    vbmc.optim_state["cache"]["y_orig"] = np.array([y_cached])
    vbmc.optim_state["cache"]["skip_logger"] = np.zeros(1, dtype=bool)
    calls_before = vbmc.function_logger.func_count

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", _cheap_acq
    )
    function_logger, optim_state, _, _ = active_sample(
        gp,
        1,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    last = function_logger.Xn
    # The target was called once, at the point that was recorded, and the
    # value recorded is the one it returned there.
    assert function_logger.func_count == calls_before + 1
    recorded = function_logger.X_orig[last]
    assert not np.allclose(recorded, x_cached[0])
    assert np.isclose(function_logger.y_orig[last], fun(recorded))
    assert not np.isclose(function_logger.y_orig[last], y_cached)

    # The row is gone from the cache, so it cannot be drawn again.
    assert optim_state["cache"]["x_orig"].shape[0] == 0
    assert optim_state["cache"]["y_orig"].shape == (0,)
    assert optim_state["cache"]["skip_logger"].shape == (0,)
    return recorded


def test_a_cached_point_the_search_box_moved_is_evaluated(mocker):
    """The sieve clips every candidate into the search box, so a cached
    starting point outside it is offered to the acquisition at a point the
    target has not been called at: the stored value does not belong to it,
    and the target is called there instead."""
    D = 2
    vbmc, gp = _state_with_gp(
        D,
        options={
            "ns_search": 1,
            "cache_frac": 1,
            "search_optimizer": "none",
        },
        seed=20260921,
    )
    x_cached = np.array([[50.0, 0.0]])
    parameter_transformer = vbmc.function_logger.parameter_transformer
    u_cached = parameter_transformer(x_cached)
    lb_search = np.copy(vbmc.optim_state["lb_search"])
    ub_search = np.copy(vbmc.optim_state["ub_search"])
    assert np.any(u_cached > ub_search)
    clipped = parameter_transformer.inverse(
        np.minimum(np.maximum(u_cached, lb_search), ub_search)
    )

    recorded = _acquire_one_cached_point(
        mocker, vbmc, gp, x_cached, fun(x_cached)
    )
    assert np.allclose(recorded, clipped[0])


def test_a_cached_point_the_integer_grid_moved_is_evaluated(mocker):
    """The acquisition snaps an integer coordinate to its grid, so a
    cached starting point off the grid is offered at a point the target
    has not been called at."""
    vbmc, gp, _, _ = _integer_var_state(
        options={
            "ns_search": 1,
            "cache_frac": 1,
            "search_optimizer": "none",
        }
    )
    x_cached = np.array([[2.4, 0.1]])

    recorded = _acquire_one_cached_point(
        mocker, vbmc, gp, x_cached, fun(x_cached)
    )
    assert recorded[0] == np.round(recorded[0])
    assert np.isclose(recorded[1], x_cached[0, 1])


def _warped_state_with_gp(D: int, options: dict = None, seed: int = None):
    """A state whose inference space a rotoscaling warp has rotated, as the
    main loop warps it: the initial design drawn and a GP trained on it, the
    warp computed from a variational posterior with correlated coordinates,
    and a GP trained again in the warped space."""
    vbmc, gp = _state_with_gp(D, options, seed=seed)
    vp = vbmc.vp
    K = vp.K
    vp.mu = np.outer(np.linspace(1.0, -0.6, D), np.linspace(-1.0, 1.0, K))
    vp.sigma = np.full((1, K), 0.3)
    vp.lambd = np.ones((D, 1))
    vp.w = np.full((1, K), 1 / K)
    parameter_transformer, optim_state, function_logger, _ = warp_input(
        vp, vbmc.optim_state, vbmc.function_logger, vbmc.options
    )
    vp, _ = warp_gp_and_vp(parameter_transformer, gp, vp, vbmc)
    vp.parameter_transformer = parameter_transformer
    vbmc.vp = vp
    vbmc.parameter_transformer = parameter_transformer
    vbmc.function_logger = function_logger
    vbmc.optim_state = optim_state
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
        optim_state["plb_tran"],
        optim_state["pub_tran"],
        rng=vp.rng,
    )
    optim_state["hyp_dict"] = hyp_dict
    return vbmc, gp


def test_a_cached_point_nothing_moved_keeps_its_value_after_a_warp(mocker):
    """Once a warp has rotated the inference space the transform is a
    matrix product, and a row of a product can round differently according
    to the rows computed with it: a cached point transformed alone need not
    equal, bit for bit, the row that the sieve made of it among the other
    cached points. A cached point that neither the clip into the search box
    nor the snap to the integer grid moved is still the point the cache
    holds, and it is acquired with its stored value, without a target call.
    A cached point that the clip moved is evaluated."""
    D = 3
    n_cache = 7
    vbmc, gp = _warped_state_with_gp(
        D,
        options={
            "ns_search": n_cache,
            "cache_frac": 1,
            "search_optimizer": "none",
        },
        seed=20260922,
    )
    function_logger = vbmc.function_logger
    optim_state = vbmc.optim_state
    parameter_transformer = function_logger.parameter_transformer
    assert parameter_transformer.R_mat is not None

    # Cached points inside the search box, with stored values that the
    # target would not return there.
    x_cached = np.random.default_rng(3).uniform(-1.5, 1.5, (n_cache, D))
    y_cached = 100.0 + np.arange(n_cache)
    u_cached = parameter_transformer(x_cached)
    assert np.all(u_cached > optim_state["lb_search"])
    assert np.all(u_cached < optim_state["ub_search"])
    optim_state["cache"]["x_orig"] = np.copy(x_cached)
    optim_state["cache"]["y_orig"] = np.copy(y_cached)
    optim_state["cache"]["skip_logger"] = np.zeros(n_cache, dtype=bool)

    def prefer_a_row_a_new_transform_moves(
        self, Xs, gp, vp, function_logger, optim_state
    ):
        """Every candidate is a cached point; the first one that differs
        from a transform of its point alone scores best, where the
        platform's matrix product gives one."""
        Xs = np.atleast_2d(Xs)
        alone = [
            parameter_transformer(x[None, :])[0]
            for x in optim_state["cache"]["x_orig"]
        ]
        return np.array(
            [
                0.0 if not any(np.array_equal(x, a) for a in alone) else 1.0
                for x in Xs
            ]
        )

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__",
        prefer_a_row_a_new_transform_moves,
    )
    calls_before = function_logger.func_count
    cache_count_before = function_logger.cache_count
    function_logger, optim_state, _, _ = active_sample(
        gp,
        1,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    # The stored value was used instead of a target call.
    assert function_logger.func_count == calls_before
    assert function_logger.cache_count == cache_count_before + 1
    last = function_logger.Xn
    acquired = int(
        np.argmin(np.sum((x_cached - function_logger.X_orig[last]) ** 2, 1))
    )
    assert np.allclose(function_logger.X_orig[last], x_cached[acquired])
    assert function_logger.y_orig[last] == y_cached[acquired]
    assert optim_state["cache"]["x_orig"].shape == (n_cache - 1, D)

    # A cached point that the clip moves into the search box is evaluated,
    # at the point the clip moved it to.
    x_far = np.array([[50.0, 0.0, 0.0]])
    u_far = parameter_transformer(x_far)
    lb_search = np.copy(optim_state["lb_search"])
    ub_search = np.copy(optim_state["ub_search"])
    assert np.any(u_far > ub_search) or np.any(u_far < lb_search)
    clipped = parameter_transformer.inverse(
        np.minimum(np.maximum(u_far, lb_search), ub_search)
    )
    vbmc.options.__setitem__("ns_search", 1, force=True)
    optim_state["cache"]["x_orig"] = np.copy(x_far)
    optim_state["cache"]["y_orig"] = np.array([100.0])
    optim_state["cache"]["skip_logger"] = np.zeros(1, dtype=bool)
    calls_before = function_logger.func_count
    function_logger, optim_state, _, _ = active_sample(
        gp,
        1,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    assert function_logger.func_count == calls_before + 1
    recorded = function_logger.X_orig[function_logger.Xn]
    assert np.allclose(recorded, clipped[0])
    assert np.isclose(
        function_logger.y_orig[function_logger.Xn], fun(recorded)
    )
    assert optim_state["cache"]["x_orig"].shape[0] == 0


def test_local_search_failure_keeps_the_best_candidate(mocker, caplog):
    """A local search that raises costs one acquisition, not the run."""
    D = 2
    vbmc, gp = _state_with_gp(D)
    candidates = np.array([[0.5, 0.5], [2.0, 2.0], [-3.0, 1.0]])
    Xn_before = vbmc.function_logger.Xn

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", _cheap_acq
    )
    mocker.patch.object(
        _active_sample_module,
        "_get_search_points",
        return_value=(candidates, np.full(len(candidates), np.nan)),
    )
    mocker.patch(
        "cma.fmin",
        side_effect=RuntimeError("no search today"),
    )
    caplog.set_level(logging.WARNING)

    function_logger, _, _, _ = active_sample(
        gp,
        1,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    # The best candidate of the search set under `_cheap_acq`.
    assert function_logger.Xn == Xn_before + 1
    assert np.allclose(function_logger.X[function_logger.Xn], candidates[0])
    assert "Active search failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "no search today" in caplog.text


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
    # The search stops by its own tolerance (`tolfun = 1e-2` on a log
    # acquisition), which on some streams halts it on the valley floor
    # short of the minimum. Every draw below comes from the run's
    # generator, the GP fit's included, so the seed fixes the stream, and
    # it is one on which the search reaches the minimum.
    vbmc = VBMC(fun, x0, LB, UB, PLB, PUB, options, seed=0)
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
        rng=vbmc.vp.rng,
    )
    optim_state["hyp_dict"] = hyp_dict
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
    logger_message += "provided, using the first 90 for the initial design "
    logger_message += "and keeping the remaining 10 in the cache."
    assert caplog.record_tuples == [
        ("ActiveSample", logging.INFO, logger_message)
    ]

    # The design consumed the first `sample_count` points with their
    # values.
    assert function_logger.Xn == sample_count - 1
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

    # The points it did not consume stay in the cache with their values.
    assert np.allclose(
        optim_state["cache"]["x_orig"],
        X_orig[sample_count:],
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.allclose(
        np.ravel(optim_state["cache"]["y_orig"]),
        np.asarray(y_orig)[sample_count:],
        rtol=1e-12,
        atol=1e-14,
    )


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
    # A cached value for the first starting point. ``f_vals`` carries no
    # noise for its values, so a target that provides its own noise cannot
    # take it, and that arm evaluates every point of the design.
    common_options = {"specify_target_noise": noisy}
    if not noisy:
        common_options["f_vals"] = [2 * D, np.nan]
    n_cached = 0 if noisy else 1
    n_evaluated = 5 - n_cached
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
    assert scalar_logger.func_count == vector_logger.func_count == n_evaluated
    assert scalar_logger.cache_count == vector_logger.cache_count == n_cached
    assert (
        scalar.vp.rng.bit_generator.state == vector.vp.rng.bit_generator.state
    )
    assert len(scalar_calls) == n_evaluated
    assert all(call.shape == (D,) for call in scalar_calls)
    assert len(vector_calls) == 1
    assert vector_calls[0].shape == (n_evaluated, D)


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

    # The cache holds exactly the requested share, so every point comes
    # from it.
    assert np.all(search_X == vbmc.parameter_transformer(x_orig[idx_cache]))
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    assert not np.any(np.isnan(idx_cache))
    assert len(np.unique(idx_cache)) == number_of_points
    assert np.all(np.isin(idx_cache, np.arange(number_of_points)))


def test_get_search_points_cache_share():
    """
    A cache larger than its share of the search set contributes that
    share, and the rest of the set is sampled.
    """
    cache_frac = 0.5
    options = {
        "cache_frac": cache_frac,
        "search_cache_frac": 0,
        "heavy_tail_search_frac": 0,
        "mvn_search_frac": 0,
        "box_search_frac": 0,
        "hpd_search_frac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    number_of_points = 10
    cache_size = 20
    x_orig = np.linspace((0, 0, 0), (10, 10, 10), cache_size)
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

    n_from_cache = math.ceil(number_of_points * cache_frac)
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    assert np.sum(~np.isnan(idx_cache)) == n_from_cache
    assert np.all(
        search_X[:n_from_cache]
        == vbmc.parameter_transformer(
            x_orig[idx_cache[:n_from_cache].astype(int)]
        )
    )


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


def test_get_search_points_empty_search_cache():
    """
    An empty search cache contributes no points, and the sieve is full.
    """
    options = {
        "cache_frac": 1,
        "search_cache_frac": 0.25,
        "heavy_tail_search_frac": 0,
        "mvn_search_frac": 0,
        "box_search_frac": 0,
        "hpd_search_frac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    number_of_points = 8
    # The state of the first active-sampling step: neither cache is filled.
    vbmc.optim_state["cache"]["x_orig"] = np.full((0, 3), np.nan)
    assert len(vbmc.optim_state["search_cache"]) == 0

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


def test_get_search_points_search_bounds():
    """
    Ensure that search bounds constrain the search points.
    """
    cache_frac = 0.5
    options = {
        "cache_frac": cache_frac,
        "search_cache_frac": 0,
        "heavy_tail_search_frac": 0,
        "mvn_search_frac": 0,
        "box_search_frac": 0,
        "hpd_search_frac": 0,
    }
    vbmc = create_vbmc(3, 3, -np.inf, np.inf, -500, 500, options)
    number_of_points = 4
    # More cache points than the share the sieve takes from it, so that
    # the set is a mix of cache points and sampled ones.
    X = np.linspace((0, 0, 0), (10, 10, 10), 5)
    vbmc.optim_state["cache"]["x_orig"] = np.copy(X)

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
    assert np.sum(~np.isnan(idx_cache)) == math.ceil(
        number_of_points * cache_frac
    )


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


@pytest.mark.parametrize(
    "search_lb, search_ub, box_lb, box_ub",
    [
        ([-10, -10], [10, 10], [-3, 0], [5, 8]),
        ([-2, 1], [4, 9], [-2, 1], [4, 8]),
        ([-np.inf, -np.inf], [np.inf, np.inf], [-3, 0], [3.5, 3.5]),
    ],
)
def test_get_search_points_all_box_search(
    search_lb, search_ub, box_lb, box_ub
):
    """
    Box candidates are uniform over the training box within search bounds.
    """
    options = {
        "cache_frac": 1,
        "search_cache_frac": 0,
        "heavy_tail_search_frac": 0,
        "mvn_search_frac": 0,
        "box_search_frac": 1,
        "hpd_search_frac": 0,
    }
    vbmc = VBMC(
        fun,
        np.zeros((1, 2)),
        np.full((1, 2), -np.inf),
        np.full((1, 2), np.inf),
        np.full((1, 2), -0.5),
        np.full((1, 2), 0.5),
        options=options,
        seed=123,
    )
    number_of_points = 8192
    vbmc.optim_state["cache"]["x_orig"] = np.empty((0, 2))
    vbmc.function_logger(np.array([-1.0, 2.0]))
    vbmc.function_logger(np.array([3.0, 6.0]))
    vbmc.function_logger(np.array([100.0, 100.0]))
    vbmc.function_logger.X_flag[2] = False
    vbmc.optim_state["lb_search"] = np.array([search_lb])
    vbmc.optim_state["ub_search"] = np.array([search_ub])
    search_X, idx_cache = _get_search_points(
        number_of_points=number_of_points,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    assert search_X.shape == (number_of_points, 2)
    assert idx_cache.shape == (number_of_points,)
    assert np.all(np.isnan(idx_cache))
    unit_X = (search_X - box_lb) / (np.array(box_ub) - box_lb)
    assert np.all((unit_X >= 0) & (unit_X < 1))
    np.testing.assert_allclose(unit_X.mean(axis=0), 0.5, atol=0.02)
    np.testing.assert_allclose(
        np.quantile(unit_X, [0.25, 0.5, 0.75], axis=0),
        [[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]],
        atol=0.03,
    )


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


class _SieveGenerator(np.random.Generator):
    """A generator that keeps the number of rows of every draw the sieve
    makes through it.

    The multivariate normals are those of the ``mvn`` source (one draw)
    and of the high-posterior-density source (one per fraction); the
    two-dimensional uniform draw is the box source. The variational
    posterior draws through methods of its own.
    """

    def __init__(self, bit_generator):
        super().__init__(bit_generator)
        self.mvn_sizes = []
        self.box_rows = []

    def multivariate_normal(self, mean, cov, size=None, *args, **kwargs):
        self.mvn_sizes.append(int(size))
        return super().multivariate_normal(mean, cov, size, *args, **kwargs)

    def random(self, size=None, *args, **kwargs):
        if isinstance(size, tuple) and len(size) == 2:
            self.box_rows.append(int(size[0]))
        return super().random(size, *args, **kwargs)


def _record_vp_draws(mocker):
    """Record the size and the degrees of freedom of every draw from the
    variational posterior, and return the list they go into."""
    calls = []
    original = VariationalPosterior.sample

    def recording(self, N, orig_flag=True, balance_flag=False, df=np.inf):
        calls.append((int(N), float(df)))
        return original(self, N, orig_flag, balance_flag, df)

    mocker.patch(
        "pyvbmc.variational_posterior.VariationalPosterior.sample", recording
    )
    return calls


def _sieve_state(
    options=None, D=3, n_cache=0, n_search_cache=0, seed=20260921
):
    """A ``VBMC`` whose state is ready for ``_get_search_points``: a
    starting cache of `n_cache` rows, a search cache of `n_search_cache`
    rows, ten training points and no search bounds."""
    vbmc = create_vbmc(D, 3, -np.inf, np.inf, -500, 500, options)
    vbmc.vp.rng = _SieveGenerator(np.random.PCG64(seed))
    vbmc.optim_state["cache"]["x_orig"] = np.linspace(
        (-1,) * D, (1,) * D, n_cache
    )
    vbmc.optim_state["search_cache"] = np.linspace(
        (2,) * D, (4,) * D, n_search_cache
    )
    for i in range(10):
        vbmc.function_logger(np.ones(D) * i)
    vbmc.optim_state["lb_search"] = np.full((1, D), -np.inf)
    vbmc.optim_state["ub_search"] = np.full((1, D), np.inf)
    return vbmc


def _search_points(vbmc, number_of_points, D=3):
    return _get_search_points(
        number_of_points=number_of_points,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        vp=vbmc.vp,
        options=vbmc.options,
    )


@pytest.mark.parametrize("number_of_points", range(1, 10))
def test_the_sieve_returns_the_number_of_points_it_is_asked_for(
    number_of_points,
):
    """The shipped fractions claim three quarters of the points to draw,
    and each share is rounded with a half going away from zero, so the
    shares can claim more than the whole: three shares of one point for
    two points to draw. The sieve returns the number asked for."""
    vbmc = _sieve_state()
    search_X, idx_cache = _search_points(vbmc, number_of_points)
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)


@pytest.mark.parametrize("n_cache", [0, 1, 2, 3])
def test_the_sieve_returns_the_points_asked_for_with_a_search_cache(n_cache):
    """A quarter for the search cache beside the shipped quarters claims
    the whole, and the starting cache leaves a number of points to draw
    that the four rounded shares can overshoot."""
    number_of_points = 16
    vbmc = _sieve_state(
        {"search_cache_frac": 0.25, "cache_frac": 1},
        n_cache=n_cache,
        n_search_cache=16,
    )
    search_X, idx_cache = _search_points(vbmc, number_of_points)
    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    assert np.sum(~np.isnan(idx_cache)) == n_cache


@pytest.mark.parametrize("number_of_points", [7, 8, 9, 10])
def test_two_halves_of_the_search_set_stay_within_it(number_of_points):
    """Two fractions of a half claim one point more than the whole for an
    odd count; the second source takes what the first left."""
    vbmc = _sieve_state(
        {
            "search_cache_frac": 0,
            "heavy_tail_search_frac": 0,
            "mvn_search_frac": 0,
            "hpd_search_frac": 0.5,
            "box_search_frac": 0.5,
        }
    )
    search_X, _ = _search_points(vbmc, number_of_points)
    assert search_X.shape == (number_of_points, 3)
    assert len(vbmc.vp.rng.box_rows) == 1
    assert vbmc.vp.rng.box_rows[0] == number_of_points // 2


def test_a_share_of_half_a_point_goes_away_from_zero(mocker):
    """The shipped fractions meet a tie whenever the starting cache
    leaves a number of points to draw that is 2 modulo 4: a quarter of
    ten points is two and a half, which ``private/activesample_vbmc.m``
    rounds to three (``:565-605``, MATLAB's ``round``). Three of the
    sources take three points each and the variational posterior draws
    the one they leave."""
    number_of_points = 16
    vbmc = _sieve_state({"cache_frac": 1}, n_cache=6)
    vp_draws = _record_vp_draws(mocker)

    search_X, idx_cache = _search_points(vbmc, number_of_points)

    assert search_X.shape == (number_of_points, 3)
    assert np.sum(~np.isnan(idx_cache)) == 6
    assert vp_draws == [(3, 3.0), (1, np.inf)]
    assert vbmc.vp.rng.mvn_sizes == [3]
    assert vbmc.vp.rng.box_rows == [3]


def test_each_source_draws_its_rounded_share(mocker):
    """Where the rounded shares leave room, every source draws the share
    its fraction gives it of the points to draw, and the variational
    posterior draws the points the five fractions leave."""
    number_of_points = 100
    vbmc = _sieve_state(
        {
            "search_cache_frac": 0.1,
            "heavy_tail_search_frac": 0.2,
            "mvn_search_frac": 0.15,
            "hpd_search_frac": 0.25,
            "box_search_frac": 0.2,
        },
        n_search_cache=40,
    )
    search_cache = np.copy(vbmc.optim_state["search_cache"])
    vp_draws = _record_vp_draws(mocker)

    search_X, _ = _search_points(vbmc, number_of_points)

    assert search_X.shape == (number_of_points, 3)
    # The search cache contributes its first rows, in order.
    assert np.array_equal(search_X[:10], search_cache[:10])
    # The heavy-tailed draw and the draw of the points the fractions left.
    assert vp_draws == [(20, 3.0), (10, np.inf)]
    # One multivariate normal for the mvn source, then one per
    # high-posterior-density fraction.
    assert vbmc.vp.rng.mvn_sizes[0] == 15
    assert sum(vbmc.vp.rng.mvn_sizes[1:]) == 25
    assert vbmc.vp.rng.box_rows == [20]


def test_fractions_that_claim_more_than_the_whole_leave_later_sources_out(
    mocker,
):
    """Fractions that claim more than the whole search set are refused at
    construction, so the sieve meets them only from a caller that writes
    them into the options of a built instance. The sources are served in
    turn until the points to draw run out."""
    vbmc = _sieve_state({"cache_frac": 0})
    for name in (
        "heavy_tail_search_frac",
        "mvn_search_frac",
        "box_search_frac",
        "hpd_search_frac",
    ):
        vbmc.options.__setitem__(name, 1, force=True)
    number_of_points = 100
    vp_draws = _record_vp_draws(mocker)

    search_X, idx_cache = _search_points(vbmc, number_of_points)

    assert search_X.shape == (number_of_points, 3)
    assert idx_cache.shape == (number_of_points,)
    # The heavy-tailed source, first in turn, takes every point.
    assert vp_draws == [(100, 3.0)]
    assert vbmc.vp.rng.mvn_sizes == []
    assert vbmc.vp.rng.box_rows == []


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
    mocker,
    user_options,
    acq=None,
    lower_bounds=None,
    upper_bounds=None,
    noise_sd=1.0,
    uncertainty_level=2,
):
    """A noisy ``VBMC`` on a quadratic target with the given options, its
    initial design drawn and a GP trained; ``acq`` replaces the
    acquisition wrapper (``AbstractAcqFcn.__call__``) when given.

    The noise of the target has the standard deviation ``noise_sd``. At
    uncertainty level 2 the target returns it with each value
    (``specify_target_noise``); at level 1 it returns the value alone and
    the run infers the noise (``uncertainty_handling``)."""
    D = 2
    rng = np.random.default_rng(0)

    def noisy_target(x):
        x = np.atleast_2d(x)
        value = -0.5 * np.sum(x**2) + noise_sd * rng.normal()
        if uncertainty_level == 2:
            return value, noise_sd
        return value

    if uncertainty_level == 2:
        noise_options = {"specify_target_noise": True}
    else:
        noise_options = {"uncertainty_handling": True}
    vbmc = VBMC(
        noisy_target,
        np.zeros((1, D)),
        -np.full((1, D), np.inf) if lower_bounds is None else lower_bounds,
        np.full((1, D), np.inf) if upper_bounds is None else upper_bounds,
        np.full((1, D), -3.0),
        np.full((1, D), 3.0),
        {
            **noise_options,
            "active_sample_gp_update": False,
            "active_sample_vp_update": False,
            **user_options,
        },
    )
    assert vbmc.optim_state["uncertainty_handling_level"] == uncertainty_level
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
            "integer_vars": np.array([True, False]),
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


def _integer_var_state(D=2, options=None, seed=20260920):
    """A ``VBMC`` whose first variable is an integer, with its initial
    design drawn and a GP trained on it."""
    user_options = {
        "integer_vars": np.array([True] + [False] * (D - 1)),
        "active_sample_gp_update": False,
        "active_sample_vp_update": False,
        **(options or {}),
    }
    vbmc, gp = _state_with_gp(
        D, user_options, seed=seed, lower_bound=-10.5, upper_bound=10.5
    )
    return vbmc, gp, vbmc.function_logger, vbmc.optim_state


def test_search_result_is_snapped_with_integer_vars(mocker):
    """The point that improves on the sieve is snapped to the integer grid.

    The search optimizers return a single point as a one-dimensional array,
    which ``private/activesample_vbmc.m:325`` passes to ``real2int_vbmc``
    before the point is evaluated.
    """
    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", _cheap_acq
    )
    vbmc, gp, function_logger, optim_state = _integer_var_state(
        options={"search_optimizer": "cmaes"}
    )

    # A search that improves on the sieve's best candidate and returns a
    # one-dimensional point away from the integer grid.
    found = {}

    def better_point(objective, x0, sigma0, options=None, **kwargs):
        x = np.asarray(x0, dtype=float) + 0.37
        found["x"] = x.copy()  # the returned array is snapped in place
        return x, -1.0

    mocker.patch("cma.fmin", side_effect=better_point)

    Xn0 = function_logger.Xn
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
    parameter_transformer = function_logger.parameter_transformer
    # The search result itself is off the grid, so the snapping is what
    # the assertion below tests.
    unsnapped = parameter_transformer.inverse(found["x"][None, :])
    assert unsnapped[0, 0] != np.round(unsnapped[0, 0])
    acquired = function_logger.X_orig[function_logger.Xn]
    assert acquired[0] == np.round(acquired[0])


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


def test_the_search_cache_holds_no_training_input(mocker):
    """With repeated observations on, the training inputs are offered to
    the acquisition of the step as candidates for a repeat. They stay out
    of the search cache: coming back from it at a later step, a training
    input would be an ordinary candidate, without the repeat flag that
    caps consecutive repeats and that keeps the point an exact repeat."""
    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker,
        {
            "ns_search": 32,
            "search_cache_frac": 0.25,
            "max_repeated_observations": 3,
            "search_optimizer": "none",
        },
        _cheap_acq,
    )
    X_train = function_logger.X[function_logger.X_flag].copy()
    assert X_train.shape[0] > 0

    function_logger, optim_state, _, gp = active_sample(
        gp,
        1,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    # The training inputs joined the candidates of the step, and the cache
    # holds the search set without them.
    search_cache = np.asarray(optim_state["search_cache"])
    assert search_cache.shape == (32, X_train.shape[1])
    for row in X_train:
        assert not np.any(np.all(search_cache == row, axis=1))


def test_candidate_noise_is_one_observation_at_uncertainty_level_one(mocker):
    """At uncertainty level 1 the noise the acquisitions read for a
    candidate is the variance of a single new observation,
    ``exp(2*h0) + exp(h1)``, averaged over the GP hyperparameter samples
    (``private/activesample_vbmc.m:159-174``). The function logger records
    a standard deviation of 1 for every evaluation and pools repeats by
    precision, so the noise function is evaluated at ``S**2 * n_evals``,
    which is 1 at every training point whatever its number of
    evaluations. An input evaluated twice therefore keeps that candidate
    noise while its own training variance is the pooled
    ``exp(2*h0) + exp(h1)/2``.
    """
    D = 2
    vbmc = create_vbmc(
        D,
        0.0,
        -np.inf,
        np.inf,
        -3,
        3,
        {
            "uncertainty_handling": True,
            "search_optimizer": "none",
            "active_sample_gp_update": False,
            "active_sample_vp_update": False,
        },
    )
    # The two starting points are the same input, so the logger pools the
    # second evaluation into the first one's row.
    function_logger, optim_state, _, _ = active_sample(
        gp=None,
        sample_count=8,
        optim_state=vbmc.optim_state,
        function_logger=vbmc.function_logger,
        iteration_history=vbmc.iteration_history,
        vp=vbmc.vp,
        options=vbmc.options,
    )
    assert optim_state["uncertainty_handling_level"] == 1
    n_evals = np.ravel(function_logger.n_evals[function_logger.X_flag])
    assert np.count_nonzero(n_evals == 2) == 1
    repeated = int(np.flatnonzero(n_evals == 2)[0])

    optim_state["N"] = function_logger.Xn + 1
    optim_state["n_eff"] = np.sum(n_evals)
    gp, _, _, hyp_dict = train_gp(
        {},
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.options,
        vbmc.plausible_lower_bounds,
        vbmc.plausible_upper_bounds,
        rng=vbmc.vp.rng,
    )
    optim_state["hyp_dict"] = hyp_dict

    seen = {}

    def recording_acq(self, Xs, gp, vp, function_logger, optim_state):
        seen["sn2_new"] = np.array(gp.temporary_data["sn2_new"], copy=True)
        seen["hyp"] = np.array([p.hyp for p in gp.posteriors])
        seen["cov_N"] = gp.covariance.hyperparameter_count(gp.D)
        seen["noise_N"] = gp.noise.hyperparameter_count()
        seen["noise"] = gp.noise
        seen["X"] = np.array(gp.X, copy=True)
        seen["y"] = np.array(gp.y, copy=True)
        seen["s2"] = np.array(gp.s2, copy=True)
        return np.sum(np.atleast_2d(Xs) ** 2, axis=1)

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", recording_acq
    )
    active_sample(
        gp,
        1,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    cov_N, noise_N = seen["cov_N"], seen["noise_N"]
    assert noise_N == 2  # the constant term and the multiplier
    h0 = seen["hyp"][:, cov_N]
    h1 = seen["hyp"][:, cov_N + 1]
    one_observation = np.exp(2 * h0) + np.exp(h1)
    assert np.allclose(seen["sn2_new"], np.mean(one_observation))

    # The GP's own noise at the repeated row is the pooled variance.
    training_noise = np.array(
        [
            np.ravel(
                seen["noise"].compute(
                    hyp[cov_N : cov_N + noise_N],
                    seen["X"],
                    seen["y"],
                    seen["s2"],
                )
            )
            for hyp in seen["hyp"]
        ]
    )
    assert seen["s2"][repeated] == pytest.approx(0.5)
    assert np.allclose(
        training_noise[:, repeated], np.exp(2 * h0) + np.exp(h1) / 2
    )
    assert not np.isclose(
        seen["sn2_new"][repeated], np.mean(training_noise[:, repeated])
    )


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
    active_sample_module = importlib.import_module("pyvbmc.vbmc.active_sample")
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

    mocker.patch.object(active_sample_module, "train_gp", recording_train_gp)
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


class _VarLogJointAcquisition(AbstractAcqFcn):
    """A flat acquisition that asks for the log-joint uncertainty."""

    def __init__(self):
        super().__init__()
        self.acq_info["compute_var_log_joint"] = True

    def _compute_acquisition_function(self, Xs, *args):
        return np.zeros(Xs.shape[0])


def test_compute_var_log_joint_hook_through_active_sample(mocker):
    """``compute_var_log_joint`` stores the variance of the expected log
    joint per hyperparameter sample and the covariance of the components'
    integrals before the search, and points are acquired."""
    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker, {"search_optimizer": "none"}
    )
    vbmc.options.__setitem__(
        "search_acq_fcn", [_VarLogJointAcquisition()], force=True
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
    # The variance is a scalar with a single hyperparameter sample.
    var_samples = np.atleast_1d(optim_state["var_log_joint_samples"])
    assert var_samples.shape == (Ns,)
    assert np.all(np.isfinite(var_samples))
    assert np.shape(optim_state["cov_log_joint_components"]) == (Ns, K, K)
    assert np.all(np.isfinite(optim_state["cov_log_joint_components"]))
    assert function_logger.Xn == Xn0 + 2


@pytest.mark.parametrize("uncertainty_level", [1, 2])
def test_noisy_fresh_point_takes_the_rank_one_gp_update(
    mocker, uncertainty_level
):
    """On a noisy target a first observation at a new input extends the GP
    posterior by rank one, with the noise variance of the observation,
    reaching the factors of a full recomputation. At uncertainty level 2
    the target gives the standard deviation of its noise, here 0.3, and
    the variance passed is its square; at level 1 the function logger
    records a standard deviation of 1 for every observation."""
    noise_sd = 0.3
    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker,
        {"max_repeated_observations": 0},
        acq=_cheap_acq,
        noise_sd=noise_sd,
        uncertainty_level=uncertainty_level,
    )
    Xn0 = function_logger.Xn

    # The step takes that branch, with the observation's noise variance,
    # instead of recomputing the posterior.
    update_spy = mocker.spy(gpr.GP, "update")
    reupdate_spy = mocker.patch.object(
        _active_sample_module, "reupdate_gp", wraps=reupdate_gp
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
    assert reupdate_spy.call_count == 0
    assert update_spy.call_count == 1
    first = Xn0 + 1
    assert function_logger.n_evals[first] == 1
    s2_new = update_spy.call_args.kwargs["s2_new"]
    assert np.allclose(s2_new, function_logger.S[first] ** 2)
    recorded_sd = noise_sd if uncertainty_level == 2 else 1.0
    assert np.allclose(s2_new, recorded_sd**2)

    # The rank-one extension by the last acquired point, which the GP has
    # not seen yet, agrees with recomputing the posterior from the whole
    # training set.
    last = function_logger.Xn
    assert function_logger.n_evals[last] == 1
    gp_rank_one = copy.deepcopy(gp)
    gp_rank_one.update(
        function_logger.X[[last]],
        function_logger.y[[last]],
        s2_new=function_logger.S[[last]] ** 2,
        compute_posterior=True,
    )
    gp_full = reupdate_gp(function_logger, copy.deepcopy(gp))
    assert gp_rank_one.X.shape == gp_full.X.shape
    assert np.allclose(gp_rank_one.s2, gp_full.s2, rtol=1e-9, atol=1e-10)
    for post_one, post_full in zip(gp_rank_one.posteriors, gp_full.posteriors):
        assert np.allclose(
            post_one.alpha, post_full.alpha, rtol=1e-9, atol=1e-10
        )
        assert np.allclose(post_one.sW, post_full.sW, rtol=1e-9, atol=1e-10)
        assert np.allclose(post_one.L, post_full.L, rtol=1e-9, atol=1e-10)


def test_in_loop_gp_posterior_update_leaves_no_timer_running(mocker):
    """The GP training timer runs only while the hyperparameters are
    refit, so a step that only recomputes the posterior leaves it stopped
    and charges nothing to GP training."""
    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker,
        {
            "search_optimizer": "none",
            "active_sample_vp_update": True,
            "active_sample_gp_update": False,
        },
        acq=_cheap_acq,
    )
    main_timer.reset()

    active_sample(
        gp,
        2,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    assert "gp_train" not in main_timer._start_times
    assert main_timer._durations.get("gp_train") is None
    # The step did run its in-loop variational update.
    assert main_timer._durations.get("variational_fit") is not None


def test_in_loop_variational_update_keeps_no_repository(mocker):
    """The in-loop variational update leaves no repository of variational
    parameters in the state."""
    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker,
        {"search_optimizer": "none", "active_sample_vp_update": True},
        acq=_cheap_acq,
    )
    optim_state.pop("vp_repo", None)
    vp_before = copy.deepcopy(vbmc.vp.get_parameters())

    _, optim_state, vp, _ = active_sample(
        gp,
        2,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    # The update ran, and nothing was collected.
    vp_after = vp.get_parameters()
    assert np.size(vp_after) != np.size(vp_before) or np.any(
        vp_after != vp_before
    )
    assert "vp_repo" not in optim_state


def test_active_sample_refreshes_n_eff(mocker):
    """Each acquisition refreshes the effective training-set count, which
    the in-loop GP refit reads, to the number of evaluations over the live
    rows of the function logger."""
    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker,
        {"search_optimizer": "none", "max_repeated_observations": 2},
        acq=_prefer_training,
    )
    seen = []

    def recording_get_search_points(number_of_points, state, logger, *a, **k):
        seen.append(
            (
                state["n_eff"],
                np.sum(logger.n_evals[logger.X_flag]),
                int(np.sum(logger.X_flag)),
            )
        )
        return _get_search_points(number_of_points, state, logger, *a, **k)

    mocker.patch.object(
        _active_sample_module,
        "_get_search_points",
        side_effect=recording_get_search_points,
    )
    active_sample(
        gp,
        3,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )
    assert len(seen) == 3
    for n_eff, n_evals_sum, n_rows in seen:
        assert np.ndim(n_eff) == 0
        assert n_eff == n_evals_sum
    # A repeated observation is pooled into its row, so the count is not
    # the number of rows.
    assert any(n_eff > n_rows for n_eff, _, n_rows in seen)


def test_in_loop_gp_refit_reads_the_counts_after_the_evaluation(mocker):
    """The GP refit between two acquisitions reads the number of training
    inputs and the number of evaluations over them, and it reads them as
    they stand after the evaluation just logged: MATLAB refreshes both
    after every logged evaluation (``misc/funlogger_vbmc.m:278-279``). A
    repeated observation raises the second count alone, a fresh point
    both."""
    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker,
        {
            "search_optimizer": "none",
            "max_repeated_observations": 1,
            "active_sample_gp_update": True,
        },
        acq=_prefer_training,
    )
    seen = []

    def recording_train_gp(
        hyp_dict, optim_state, logger, history, options, *a, **k
    ):
        seen.append(
            (
                optim_state["N"],
                optim_state["n_eff"],
                logger.Xn + 1,
                np.sum(logger.n_evals[logger.X_flag]),
            )
        )
        return train_gp(
            hyp_dict, optim_state, logger, history, options, *a, **k
        )

    mocker.patch.object(
        _active_sample_module, "train_gp", side_effect=recording_train_gp
    )
    N0 = function_logger.Xn + 1
    n_eff0 = np.sum(function_logger.n_evals[function_logger.X_flag])
    active_sample(
        gp,
        3,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    # One refit after each acquisition but the last: the first acquisition
    # repeats a training input, the second is a fresh point.
    assert [(N, n_eff) for N, n_eff, _, _ in seen] == [
        (N0, n_eff0 + 1),
        (N0 + 1, n_eff0 + 2),
    ]
    for N, n_eff, logger_N, logger_n_eff in seen:
        assert N == logger_N
        assert n_eff == logger_n_eff


def _record_in_loop_variational_update(mocker):
    """Stand in for the in-loop variational optimization and for the ELBO
    of the posterior from before the step, recording the number of fast
    optimizations and of entropy samples each is asked for. The updated
    posterior is moved and scores low, so the comparison with the old one
    runs and keeps the old one."""
    seen = {}

    def fake_optimize_vp(options, optim_state, vp, gp, fast_opts_N, **kwargs):
        seen["fast_opts_N"] = fast_opts_N
        moved = copy.deepcopy(vp)
        moved.mu = moved.mu + 0.1
        moved.stats = {"elbo": -1e9}
        return moved, 0.0, 0

    def fake_neg_elcbo(theta, gp, vp, beta, Ns, *args, **kwargs):
        seen["entropy_samples"] = Ns
        return (0.0,)

    mocker.patch.object(
        _active_sample_module, "optimize_vp", side_effect=fake_optimize_vp
    )
    mocker.patch.object(
        _active_sample_module, "_neg_elcbo", side_effect=fake_neg_elcbo
    )
    return seen


def test_a_scalar_count_option_reaches_the_in_loop_update(mocker):
    """``ns_elbo`` and ``ns_ent_fine_active`` may be numbers as well as
    functions of the number of components, as construction and
    ``Options.eval`` allow. The in-loop variational update reads both, and
    a number gives the count that the shipped function gives at the same
    number of components."""
    K = 2
    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker,
        {
            "search_optimizer": "none",
            "active_sample_vp_update": True,
            "k_warmup": K,
            "ns_elbo": 50 * K,
            "ns_ent_fine_active": 200 * K,
        },
        acq=_cheap_acq,
    )
    assert vbmc.vp.K == K
    seen = _record_in_loop_variational_update(mocker)

    active_sample(
        gp,
        2,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    # The shipped functions are 50 * K and 200 * K.
    assert seen["fast_opts_N"] == math.ceil(
        vbmc.options["ns_elbo_incr"] * 50 * K
    )
    assert seen["entropy_samples"] == math.ceil(200 * K / K)


def test_in_loop_comparison_scores_one_component_with_its_exact_entropy(
    mocker,
):
    """Where the in-loop variational update has moved the posterior, the
    posterior from before the step is scored again and kept if its ELBO is
    the higher. The reported ELBO of the moved posterior takes the exact
    entropy of a Gaussian when it has one component, and so does the score
    of a one-component posterior from before the step, so that both sides
    of the comparison use one estimator."""
    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker,
        {
            "search_optimizer": "none",
            "active_sample_vp_update": True,
            "k_warmup": 1,
        },
        acq=_cheap_acq,
    )
    vp0 = copy.deepcopy(vbmc.vp)
    assert vp0.K == 1

    def fake_optimize_vp(options, optim_state, vp, gp, fast_opts_N, **kwargs):
        moved = copy.deepcopy(vp)
        moved.mu = moved.mu + 0.1
        moved.stats = {"elbo": -1e9}
        return moved, 0.0, 0

    mocker.patch.object(
        _active_sample_module, "optimize_vp", side_effect=fake_optimize_vp
    )
    score = mocker.spy(_active_sample_module, "_neg_elcbo")

    active_sample(
        gp,
        2,
        optim_state,
        function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    # No entropy samples: the deterministic entropy, which is the exact
    # entropy of a single Gaussian.
    assert score.call_count == 1
    assert score.call_args.args[4] == 0
    entropy = score.spy_return[3]
    D = vp0.D
    exact = 0.5 * D * (1 + np.log(2 * np.pi)) + np.sum(
        np.log(vp0.sigma[0, 0] * vp0.lambd)
    )
    assert np.isclose(entropy, exact, rtol=1e-12, atol=1e-12)
