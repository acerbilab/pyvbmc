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
from pyvbmc.vbmc import active_sample
from pyvbmc.vbmc.active_sample import _get_search_points
from pyvbmc.vbmc.gaussian_process_train import reupdate_gp, train_gp

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


def _state_with_gp(D: int, options: dict = None, seed: int = None):
    """Build a VBMC instance and a GP trained on one initial design."""
    vbmc = create_vbmc(D, 0.0, -np.inf, np.inf, -3, 3, options)
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
    mocker.patch("pyvbmc.vbmc.active_sample.cma.fmin", side_effect=fake_fmin)

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
    mocker.patch("pyvbmc.vbmc.active_sample.cma.fmin", side_effect=fake_fmin)

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
    mocker.patch("pyvbmc.vbmc.active_sample.cma.fmin", side_effect=fake_fmin)

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
    mocker.patch(
        "pyvbmc.vbmc.active_sample._get_search_points",
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


def test_local_search_failure_keeps_the_best_candidate(mocker, caplog):
    """A local search that raises costs one acquisition, not the run."""
    D = 2
    vbmc, gp = _state_with_gp(D)
    candidates = np.array([[0.5, 0.5], [2.0, 2.0], [-3.0, 1.0]])
    Xn_before = vbmc.function_logger.Xn

    mocker.patch(
        "pyvbmc.acquisition_functions.AbstractAcqFcn.__call__", _cheap_acq
    )
    mocker.patch(
        "pyvbmc.vbmc.active_sample._get_search_points",
        return_value=(candidates, np.full(len(candidates), np.nan)),
    )
    mocker.patch(
        "pyvbmc.vbmc.active_sample.cma.fmin",
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
    # acquisition), which on about one stream in seven halts it on the
    # valley floor short of the minimum; the seed fixes one that reaches it.
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


def test_noisy_fresh_point_takes_the_rank_one_gp_update(mocker):
    """On a noisy target a first observation at a new input extends the GP
    posterior by rank one, reaching the factors of a full recomputation."""
    vbmc, gp, function_logger, optim_state = _noisy_run(
        mocker, {"max_repeated_observations": 0}, acq=_cheap_acq
    )

    # The step takes that branch, with the observation's noise variance,
    # instead of recomputing the posterior.
    update_spy = mocker.spy(gpr.GP, "update")
    reupdate_spy = mocker.patch(
        "pyvbmc.vbmc.active_sample.reupdate_gp", wraps=reupdate_gp
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
    assert update_spy.call_args.kwargs["s2_new"] is not None

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
    the in-loop updates read, to the number of evaluations over the live
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

    mocker.patch(
        "pyvbmc.vbmc.active_sample._get_search_points",
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
