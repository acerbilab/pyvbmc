"""Search policy checks without inference runs or target evaluations."""

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from noisy_acq_search import SearchConfig, _refine, select_candidate

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.testing.oracles._state import build_state, load_snapshot


class Quadratic:
    def __init__(self):
        self.rows = 0

    def score(self, points, **kwargs):
        points = np.atleast_2d(points)
        self.rows += len(points)
        return {
            "full_score": np.sum(points**2, axis=1),
            "valid": np.ones(len(points), dtype=bool),
        }


def small_state():
    return {
        "vp": SimpleNamespace(moments=lambda **_: (np.zeros(2), np.eye(2))),
        "optim_state": {
            "integer_vars": None,
            "lb_search": np.array([[-3.0, -3.0]]),
            "ub_search": np.array([[3.0, 3.0]]),
        },
    }


def test_refinement_finds_quadratic_minimum_and_counts_every_row():
    evaluator = Quadratic()
    points = np.array([[1.0, 2.0], [-2.0, -2.0]])
    scores = evaluator.score(points)["full_score"]
    chosen, details = _refine(
        small_state(),
        evaluator,
        points,
        scores,
        np.zeros(2, dtype=bool),
        SearchConfig(arm="S3"),
    )
    np.testing.assert_allclose(chosen, 0, atol=1e-5)
    assert details["accurate_candidate_rows"] == evaluator.rows
    assert len(details["local_runs"]) == 2
    assert details["local_iterations"] == sum(
        run["iterations"] for run in details["local_runs"]
    )
    assert all(
        run["iterations"] == run["solver_reported_iterations"]
        for run in details["local_runs"]
    )
    assert details["selected_shortlist_index"] is None


def test_refinement_budget_keeps_accurately_rescored_fallback():
    evaluator = Quadratic()
    points = np.array([[1.0, 2.0], [-2.0, -2.0]])
    scores = evaluator.score(points)["full_score"]
    config = SearchConfig(arm="S3", shortlist_size=2, max_candidate_rows=4)
    chosen, details = _refine(
        small_state(),
        evaluator,
        points,
        scores,
        np.zeros(2, dtype=bool),
        config,
    )
    np.testing.assert_array_equal(chosen, points[0])
    assert details["fallback_reason"] == "candidate_row_budget"
    assert evaluator.rows <= 4


def test_refinement_iteration_budget_is_cumulative_across_starts(monkeypatch):
    evaluator = Quadratic()
    points = np.array([[0.1, 0.1], [1.0, 1.0], [-1.0, -1.0], [2.0, -2.0]])
    scores = evaluator.score(points)["full_score"]
    iteration_limits = []

    def fake_minimize(fun, x0, *, callback, options, **kwargs):
        iteration_limits.append(options["maxiter"])
        iterations = min(30, options["maxiter"])
        for _ in range(iterations):
            callback(x0)
        return SimpleNamespace(
            x=x0, success=False, message="iteration limit", nit=iterations
        )

    monkeypatch.setattr("noisy_acq_search.minimize", fake_minimize)
    chosen, details = _refine(
        small_state(),
        evaluator,
        points,
        scores,
        np.zeros(4, dtype=bool),
        SearchConfig(arm="S3", shortlist_size=4),
    )

    np.testing.assert_array_equal(chosen, points[0])
    assert iteration_limits == [50, 20]
    assert details["local_iterations"] == 50
    assert [run["iterations"] for run in details["local_runs"]] == [30, 20]
    assert details["refinement_stop_reason"] == "iteration_budget"
    assert details["fallback_reason"] is None


def test_interrupted_solver_retains_completed_iteration_count(monkeypatch):
    evaluator = Quadratic()
    points = np.array([[0.1, 0.1], [1.0, 1.0], [-1.0, -1.0], [2.0, -2.0]])
    scores = evaluator.score(points)["full_score"]

    def fake_minimize(fun, x0, *, callback, **kwargs):
        callback(x0)
        callback(x0)
        fun(x0)
        raise AssertionError("the row budget should stop the objective")

    monkeypatch.setattr("noisy_acq_search.minimize", fake_minimize)
    chosen, details = _refine(
        small_state(),
        evaluator,
        points,
        scores,
        np.zeros(4, dtype=bool),
        SearchConfig(arm="S3", shortlist_size=4, max_candidate_rows=16),
    )

    np.testing.assert_array_equal(chosen, points[0])
    assert details["accurate_candidate_rows"] == evaluator.rows == 14
    assert details["local_iterations"] == 2
    assert len(details["local_runs"]) == 1
    assert details["local_runs"][0]["iterations"] == 2
    assert details["local_runs"][0]["status"] == "candidate_row_budget"
    assert details["fallback_reason"] == "candidate_row_budget"
    assert details["selected_accurate_score"] == scores[0]


def test_later_invalid_start_restores_original_rescored_candidate():
    class InvalidSecondStart(Quadratic):
        def score(self, points, **kwargs):
            result = super().score(points)
            result["valid"] &= np.atleast_2d(points)[:, 0] >= 0
            return result

    evaluator = InvalidSecondStart()
    points = np.array([[1.0, 2.0], [-2.0, -2.0]])
    scores = np.sum(points**2, axis=1)
    chosen, details = _refine(
        small_state(),
        evaluator,
        points,
        scores,
        np.zeros(2, dtype=bool),
        SearchConfig(arm="S3"),
    )
    np.testing.assert_array_equal(chosen, points[0])
    assert details["fallback_reason"] == "invalid_local_score"
    assert details["selected_shortlist_index"] == 0


@pytest.mark.parametrize("condition", ["integer", "repeat"])
def test_integer_and_repeat_winners_do_not_refine(condition):
    state = small_state()
    if condition == "integer":
        state["optim_state"]["integer_vars"] = np.array([True, False])
    evaluator = Quadratic()
    points = np.array([[1.0, 2.0]])
    scores = evaluator.score(points)["full_score"]
    chosen, detail = _refine(
        state,
        evaluator,
        points,
        scores,
        np.array([condition == "repeat"]),
        SearchConfig(arm="S2"),
    )
    np.testing.assert_array_equal(chosen, points[0])
    assert evaluator.rows == 1
    assert detail["fallback_reason"] in {
        "integer_variables",
        "selected_repeat",
    }


def test_source_controller_stops_before_target_and_isolates_rng():
    fixture = (
        Path(__file__).resolve().parents[2]
        / "pyvbmc/testing/oracles/fixtures/rosenbrock_D2_noise1_viqr"
    )
    state = build_state(load_snapshot(fixture), rng=np.random.default_rng(771))
    state["options"].__setitem__("search_optimizer", "none", force=True)
    rng_before = copy.deepcopy(state["vp"].rng.bit_generator.state)
    candidates_before = state["cand"]["Xs"].copy()
    global_before = np.random.get_state()
    result = select_candidate(
        state, SearchConfig(arm="S0"), search_seed=22, accurate_seed=23
    )
    np.testing.assert_array_equal(
        result["selected"][0], result["coarse_winner"]
    )
    assert result["coarse_candidate_rows"] == 8192
    assert not result["target_called"]
    assert state["vp"].rng.bit_generator.state == rng_before
    np.testing.assert_array_equal(state["cand"]["Xs"], candidates_before)
    assert all(
        np.array_equal(a, b)
        for a, b in zip(global_before, np.random.get_state())
    )
    first = select_candidate(
        state,
        SearchConfig(arm="S1", accurate_budget=32),
        search_seed=22,
        accurate_seed=23,
    )
    second = select_candidate(
        state,
        SearchConfig(arm="S1", accurate_budget=32),
        search_seed=22,
        accurate_seed=23,
    )
    np.testing.assert_array_equal(first["selected"], second["selected"])
    np.testing.assert_array_equal(
        first["coarse_candidates"], second["coarse_candidates"]
    )
    assert first["coarse_candidate_rows"] == 1024
    assert first["accurate_candidate_rows"] == 8
    assert not first["target_called"]


def test_qmc_config_is_frozen_to_the_production_search_and_96_nodes():
    SearchConfig(arm="S0", importance_qmc=True).validate()
    SearchConfig(
        arm="S0", importance_qmc=True, importance_qmc_order="index"
    ).validate()
    with pytest.raises(ValueError, match="S0"):
        SearchConfig(arm="S2", importance_qmc=True).validate()
    with pytest.raises(ValueError, match="96"):
        SearchConfig(
            arm="S0", importance_qmc=True, importance_qmc_samples=128
        ).validate()
    with pytest.raises(ValueError, match="order"):
        SearchConfig(
            arm="S0", importance_qmc=True, importance_qmc_order="random"
        ).validate()


def test_qmc_arms_replace_only_the_importance_nodes(monkeypatch):
    fixture = (
        Path(__file__).resolve().parents[2]
        / "pyvbmc/testing/oracles/fixtures/rosenbrock_D2_noise1_viqr"
    )
    state = build_state(load_snapshot(fixture), rng=np.random.default_rng(5))
    state["options"].__setitem__("search_optimizer", "none", force=True)
    importance = importlib.import_module(
        "pyvbmc.vbmc.active_importance_sampling"
    )
    orders = []
    original = importance.qmc_vp_nodes

    def recording(vp, N, rng, order=None):
        orders.append(None if order is None else np.array(order, copy=True))
        return original(vp, N, rng, order=order)

    monkeypatch.setattr(importance, "qmc_vp_nodes", recording)
    baseline = select_candidate(
        state, SearchConfig(arm="S0"), search_seed=22, accurate_seed=23
    )
    sorted_arm = select_candidate(
        state,
        SearchConfig(arm="S0", importance_qmc=True),
        search_seed=22,
        accurate_seed=23,
    )
    unsorted_arm = select_candidate(
        state,
        SearchConfig(
            arm="S0", importance_qmc=True, importance_qmc_order="index"
        ),
        search_seed=22,
        accurate_seed=23,
    )
    assert baseline["importance_node_count"] == 100
    assert baseline["coarse_nodes"].shape == (100, 2)
    assert sorted_arm["importance_node_count"] == 96
    assert unsorted_arm["importance_node_count"] == 96
    assert sorted_arm["coarse_nodes"].shape == (96, 2)
    # The sieve is drawn before the nodes from the shared search seed, so
    # every arm scores the same 8192 candidates.
    np.testing.assert_array_equal(
        baseline["coarse_candidates"], sorted_arm["coarse_candidates"]
    )
    np.testing.assert_array_equal(
        baseline["coarse_candidates"], unsorted_arm["coarse_candidates"]
    )
    assert not np.array_equal(
        baseline["coarse_nodes"][:96], sorted_arm["coarse_nodes"]
    )
    assert orders[0] is None
    np.testing.assert_array_equal(orders[1], np.arange(state["vp"].K))
    assert not sorted_arm["target_called"]
    # Nothing leaks out of the private copy: the captured options keep the
    # Monte Carlo draw and the production component order is restored.
    assert state["options"]["active_importance_sampling_qmc"] is False
    assert importance._QMC_COMPONENT_ORDER == "axis"


def test_s0_matches_direct_production_cmaes_selection(monkeypatch):
    fixture = (
        Path(__file__).resolve().parents[2]
        / "pyvbmc/testing/oracles/fixtures/rosenbrock_D2_noise1_viqr"
    )
    state = build_state(load_snapshot(fixture), rng=np.random.default_rng(91))
    state["options"].__setitem__("search_optimizer", "cmaes", force=True)
    state["options"].__setitem__("search_max_fun_evals", 30, force=True)
    experiment = select_candidate(
        state, SearchConfig(arm="S0"), search_seed=44, accurate_seed=45
    )
    direct = copy.deepcopy(state)
    direct["vp"].rng = np.random.default_rng(44)
    direct["logger"].parameter_transformer = direct["vp"].parameter_transformer

    class StopTarget(Exception):
        def __init__(self, x):
            self.x = np.array(x, copy=True).reshape(1, -1)

    def stop(self, x, *args, **kwargs):
        raise StopTarget(x)

    monkeypatch.setattr(FunctionLogger, "__call__", stop)
    monkeypatch.setattr(FunctionLogger, "add", stop)
    active = importlib.import_module("pyvbmc.vbmc.active_sample")
    timer_before = copy.deepcopy(active.timer.__dict__)
    try:
        with pytest.raises(StopTarget) as caught:
            active.active_sample(
                direct["gp"],
                1,
                direct["optim_state"],
                direct["logger"],
                {"r_index": [np.inf]},
                direct["vp"],
                direct["options"],
            )
        np.testing.assert_array_equal(experiment["selected"], caught.value.x)
        assert experiment["production_candidate_rows"] > 8192
    finally:
        active.timer.__dict__.clear()
        active.timer.__dict__.update(timer_before)
