"""Focused checks for the bounded E5 live-search policy."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from noisy_acq_inference_policy import InferencePolicy
from noisy_acq_search import SearchConfig, select_candidate

from pyvbmc.acquisition_functions import AcqFcnVIQR
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.testing.oracles._state import build_state, load_snapshot

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "pyvbmc/testing/oracles/fixtures/rosenbrock_D2_noise1_viqr"
)


class StopTarget(Exception):
    def __init__(self, x):
        self.x = np.asarray(x, dtype=np.float64).reshape(1, -1)


def _stop_target(self, x, *args, **kwargs):
    raise StopTarget(x)


def _run_one_selection(state, monkeypatch, policy=None):
    active = importlib.import_module("pyvbmc.vbmc.active_sample")
    monkeypatch.setattr(FunctionLogger, "__call__", _stop_target)
    monkeypatch.setattr(FunctionLogger, "add", _stop_target)
    context = policy if policy is not None else _NullContext()
    with context:
        with pytest.raises(StopTarget) as stopped:
            active.active_sample(
                state["gp"],
                1,
                state["optim_state"],
                state["logger"],
                {"r_index": [np.inf]},
                state["vp"],
                state["options"],
            )
    return stopped.value.x


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_s0_observation_replays_production_exactly(monkeypatch):
    observed = build_state(
        load_snapshot(FIXTURE), rng=np.random.default_rng(8101)
    )
    production = build_state(
        load_snapshot(FIXTURE), rng=np.random.default_rng(8101)
    )
    for state in (observed, production):
        state["options"].__setitem__("search_optimizer", "cmaes", force=True)
        state["options"].__setitem__("search_max_fun_evals", 30, force=True)

    policy = InferencePolicy("S0", 2000, "replay")
    selected = _run_one_selection(observed, monkeypatch, policy)
    baseline = _run_one_selection(production, monkeypatch)

    np.testing.assert_array_equal(selected, baseline)
    assert (
        observed["vp"].rng.bit_generator.state
        == production["vp"].rng.bit_generator.state
    )
    assert len(policy.records) == 1
    assert policy.records[0]["status"] == "complete"
    assert policy.records[0]["accurate_seed"] is None
    assert policy.summary()["selection_count"] == 1


def test_two_live_selections_observe_updated_gp_logger_and_rng():
    def target(x):
        return -float(np.sum(np.asarray(x) ** 2)), 0.1

    state = build_state(
        load_snapshot(FIXTURE),
        fun=target,
        rng=np.random.default_rng(8103),
    )
    options = state["options"]
    options.__setitem__("ns_search", 32, force=True)
    options.__setitem__("search_optimizer", "none", force=True)
    options.__setitem__("active_sample_gp_update", False, force=True)
    options.__setitem__("active_sample_vp_update", False, force=True)
    initial_rows = len(state["gp"].X)
    policy = InferencePolicy("S0", 2000, "two-selections")
    active = importlib.import_module("pyvbmc.vbmc.active_sample")

    with policy:
        logger, _, _, gp = active.active_sample(
            state["gp"],
            2,
            state["optim_state"],
            state["logger"],
            {"r_index": [np.inf]},
            state["vp"],
            options,
        )

    assert len(policy.records) == 2
    assert policy.records[0]["gp_training_rows"] == initial_rows
    assert policy.records[1]["gp_training_rows"] == len(gp.X)
    assert (
        policy.records[1]["logger_live_rows"]
        == policy.records[1]["gp_training_rows"]
    )
    assert int(np.count_nonzero(logger.X_flag)) >= len(gp.X)
    assert (
        policy.records[0]["vp_rng_after_sha256"]
        == policy.records[1]["vp_rng_before_sha256"]
    )


def test_s2_live_matches_frozen_controller_with_fresh_search_rng(monkeypatch):
    search_seed = 8102
    accurate_seed = 9102
    frozen_state = build_state(
        load_snapshot(FIXTURE), rng=np.random.default_rng(0)
    )
    expected = select_candidate(
        frozen_state,
        SearchConfig(arm="S2"),
        search_seed=search_seed,
        accurate_seed=accurate_seed,
        retain_panel=False,
    )
    live = build_state(load_snapshot(FIXTURE), rng=np.random.default_rng(0))
    # ``build_state`` consumes its constructor RNG while rebuilding the VP;
    # the frozen controller installs a fresh search RNG after that rebuild.
    live["vp"].rng = np.random.default_rng(search_seed)
    original_ns_search = live["options"]["ns_search"]
    policy = InferencePolicy(
        "S2", 2000, "parity", testing_seeds=[accurate_seed]
    )
    selected = _run_one_selection(live, monkeypatch, policy)

    np.testing.assert_array_equal(selected, expected["selected"])
    assert live["options"]["ns_search"] == original_ns_search
    assert policy.records[0]["accurate_seed"] == accurate_seed
    assert policy.records[0]["coarse_candidate_rows"] == 1024
    assert policy.records[0]["selected_sha256"]


class _Options(dict):
    def __setitem__(self, key, value, force=False):
        super().__setitem__(key, value)

    def eval(self, key, evaluation_parameters=None):
        return self[key]


def _callback_inputs():
    vp = SimpleNamespace(
        D=2,
        K=1,
        rng=np.random.default_rng(44),
    )
    logger = SimpleNamespace(
        X=np.array([[9.0, 9.0], [np.nan, np.nan]]),
        X_flag=np.array([True, False]),
    )
    options = _Options(
        ns_search=8192,
        search_acq_fcn=[AcqFcnVIQR()],
        active_importance_sampling_mcmc_samples=100,
    )
    optim_state = {
        "cache": {
            "x_orig": np.zeros((3, 2)),
            "y_orig": np.zeros(3),
        }
    }
    return vp, logger, options, optim_state


@pytest.mark.parametrize(
    ("scores", "cache_indices", "source", "cache_index", "repeat"),
    [
        ([0.0, 1.0, 2.0], [np.nan, 2.0, np.nan], "repeat", None, True),
        ([2.0, 0.0, 1.0], [np.nan, 2.0, np.nan], "cache", 2, False),
    ],
)
def test_s2_preserves_repeat_and_cache_selection(
    monkeypatch, scores, cache_indices, source, cache_index, repeat
):
    module = importlib.import_module("noisy_acq_inference_policy")
    vp, logger, options, optim_state = _callback_inputs()
    candidates = np.array([[8.0, 8.0], [1.0, 1.0], [2.0, 2.0]])

    monkeypatch.setattr(
        module,
        "make_rule",
        lambda *args, **kwargs: {
            "available": True,
            "method": "mc",
            "budget": 1,
            "seed": 1,
            "nodes": np.zeros((1, 2)),
            "weights": np.ones(1),
            "metadata": {"component_assignment": [0]},
        },
    )
    evaluator = SimpleNamespace(
        score=lambda points, diagnostics=False: {
            "valid": np.ones(len(points), dtype=bool),
            "full_score": np.arange(len(points), dtype=float),
        }
    )
    monkeypatch.setattr(module, "prepare_rule", lambda *args: evaluator)

    def choose_first(state, evaluator, points, values, repeated, config):
        return points[0], {
            "selected_shortlist_index": 0,
            "accurate_candidate_rows": len(points),
            "local_iterations": 0,
            "fallback_reason": "selected_repeat" if repeated[0] else None,
            "refinement_stop_reason": None,
            "selected_accurate_score": float(values[0]),
        }

    monkeypatch.setattr(module, "_refine", choose_first)
    gp = SimpleNamespace(D=2, X=np.zeros((1, 2)))
    policy = InferencePolicy("S2", 2000, "row-semantics")
    with policy:
        policy.start(
            gp=gp,
            vp=vp,
            function_logger=logger,
            optim_state=optim_state,
            options=options,
        )
        selected, selected_cache, selected_repeat = policy.select(
            candidates=candidates,
            coarse_scores=np.asarray(scores),
            cache_indices=np.asarray(cache_indices),
            n_train=1,
            gp=gp,
            vp=vp,
            function_logger=logger,
            optim_state=optim_state,
            options=options,
        )
        policy.finish(
            selected=selected,
            cache_index=selected_cache,
            repeat=selected_repeat,
        )

    record = policy.records[0]
    assert record["selected_source"] == source
    assert record["cache_index"] == cache_index
    assert record["repeat"] is repeat
    if repeat:
        np.testing.assert_array_equal(selected[0], logger.X[0])


def test_exception_restores_callback_options_and_retains_failure_record():
    active = importlib.import_module("pyvbmc.vbmc.active_sample")
    vp, logger, options, optim_state = _callback_inputs()
    policy = InferencePolicy("S2", 2000, "exception")
    with pytest.raises(RuntimeError, match="selection failed"):
        with policy:
            policy.start(
                gp=SimpleNamespace(D=2, X=np.zeros((1, 2))),
                vp=vp,
                function_logger=logger,
                optim_state=optim_state,
                options=options,
            )
            raise RuntimeError("selection failed")

    assert active._selection_policy_callback is None
    assert options["ns_search"] == 8192
    assert policy.records[0]["status"] == "failed"
    assert policy.records[0]["exception_type"] == "RuntimeError"


def test_nested_policy_context_is_rejected_and_outer_is_restored():
    active = importlib.import_module("pyvbmc.vbmc.active_sample")
    with InferencePolicy("S0", 2000, "outer"):
        with pytest.raises(RuntimeError, match="nested"):
            with InferencePolicy("S0", 2000, "inner"):
                pass
    assert active._selection_policy_callback is None


@pytest.mark.parametrize("mode", ["repeat", "cache"])
def test_core_seam_preserves_live_repeat_and_cache_consumption(mode):
    def target(x):
        return -float(np.sum(np.asarray(x) ** 2)), 0.1

    state = build_state(
        load_snapshot(FIXTURE),
        fun=target,
        rng=np.random.default_rng(8104),
    )
    options = state["options"]
    options.__setitem__("ns_search", 16, force=True)
    options.__setitem__("search_optimizer", "none", force=True)
    options.__setitem__("cache_frac", 1.0, force=True)
    options.__setitem__("max_repeated_observations", 3, force=True)
    logger = state["logger"]
    active = importlib.import_module("pyvbmc.vbmc.active_sample")
    state["optim_state"]["repeated_observations_streak"] = 0

    if mode == "cache":
        state["optim_state"]["cache"]["x_orig"] = np.array([[0.25, 0.5]])
        state["optim_state"]["cache"]["y_orig"] = np.array([-0.3125])
    previous_streak = state["optim_state"].get(
        "repeated_observations_streak", 0
    )
    first_live_index = int(np.flatnonzero(logger.X_flag)[0])
    previous_evaluations = logger.n_evals[first_live_index].item()

    class Override:
        def start(self, **kwargs):
            self.logger = kwargs["function_logger"]

        def select(self, **kwargs):
            if mode == "repeat":
                assert kwargs["n_train"] > 0
                row = self.logger.X[self.logger.X_flag][[0]].copy()
                return row, np.nan, True
            indices = np.asarray(kwargs["cache_indices"])
            row = int(np.flatnonzero(np.isfinite(indices))[0])
            return kwargs["candidates"][[row]].copy(), indices[row], False

        def finish(self, **kwargs):
            self.finished = kwargs

    callback = Override()
    assert active._selection_policy_callback is None
    active._selection_policy_callback = callback
    try:
        active.active_sample(
            state["gp"],
            1,
            state["optim_state"],
            logger,
            {"r_index": [np.inf]},
            state["vp"],
            options,
        )
    finally:
        active._selection_policy_callback = None

    assert callback.finished["repeat"] is (mode == "repeat")
    if mode == "repeat":
        assert (
            logger.n_evals[first_live_index].item() == previous_evaluations + 1
        )
        assert (
            state["optim_state"]["repeated_observations_streak"]
            == previous_streak + 1
        )
    else:
        assert state["optim_state"]["cache"]["x_orig"].shape == (0, 2)
        assert state["optim_state"]["cache"]["y_orig"].shape == (0,)
