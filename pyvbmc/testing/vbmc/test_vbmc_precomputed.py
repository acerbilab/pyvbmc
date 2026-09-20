import logging

import numpy as np
import pytest

from pyvbmc import VBMC
from pyvbmc.vbmc.active_sample import active_sample
from pyvbmc.vbmc.gaussian_process_train import _get_gp_training_options


def _vbmc(target, *, x0=None, options=None, **kwargs):
    if x0 is None:
        x0 = np.array([[0.0]])
    defaults = {
        "display": "off",
        "fun_eval_start": 4,
        "max_fun_evals": 20,
        "min_iter": 0,
    }
    if options is not None:
        defaults.update(options)
    return VBMC(
        target,
        x0,
        np.array([[-2.0]]),
        np.array([[2.0]]),
        np.array([[-1.0]]),
        np.array([[1.0]]),
        options=defaults,
        seed=12,
        **kwargs,
    )


def _nextafter(value, steps):
    result = float(value)
    for _ in range(steps):
        result = np.nextafter(result, np.inf)
    return result


def test_precomputed_exact_rows_are_copied_and_deduplicated():
    X = np.array([[0.0], [0.5], [0.5]], dtype=np.float32)
    y = np.array([0.0, -3e18, _nextafter(-3e18, 2)])
    vbmc = _vbmc(
        lambda x: (_ for _ in ()).throw(AssertionError("target called")),
        precomputed_evaluations=(X, y),
    )

    X.fill(1)
    y.fill(1)
    stored_X, stored_y = vbmc.precomputed_evaluations
    assert stored_X.dtype == stored_y.dtype == np.float64
    assert np.array_equal(stored_X[:, 0], [0.0, 0.5, 0.5])
    assert stored_y[1] == -3e18
    assert vbmc.precomputed_observation_count == 3
    assert vbmc.precomputed_location_count == 2
    assert vbmc.function_logger.cache_count == 2
    assert vbmc.function_logger.Xn == 1
    assert np.array_equal(
        vbmc.function_logger.n_evals[:2, 0], np.ones(2, dtype=int)
    )


def test_precomputed_exact_conflict_and_legacy_f_vals_are_rejected():
    observations = (np.array([[0.25], [0.25]]), np.array([1.0, 1.0 + 1e-12]))
    with pytest.raises(ValueError, match="Conflicting deterministic"):
        _vbmc(np.sum, precomputed_evaluations=observations)

    max_float = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="Conflicting deterministic"):
        _vbmc(
            np.sum,
            precomputed_evaluations=(
                np.array([[0.25], [0.25]]),
                np.array([max_float, -max_float]),
            ),
        )

    with pytest.raises(ValueError, match="legacy.*f_vals"):
        _vbmc(
            np.sum,
            options={"f_vals": [0.0]},
            precomputed_evaluations=(np.array([[0.0]]), np.array([0.0])),
        )


def test_precomputed_values_receive_separate_prior_once():
    prior_calls = []

    def prior(x):
        prior_calls.append(np.asarray(x).copy())
        return -2.0 * np.sum(np.asarray(x) ** 2)

    vbmc = _vbmc(
        lambda x: (_ for _ in ()).throw(AssertionError("target called")),
        log_prior=prior,
        precomputed_evaluations=(
            np.array([[0.0], [0.5]]),
            np.array([3.0, 4.0]),
        ),
    )

    assert len(prior_calls) == 2
    assert all(call.shape == (1,) for call in prior_calls)
    assert np.allclose(vbmc.function_logger.y_orig[:2, 0], [3.0, 3.5])


@pytest.mark.parametrize(
    "options,evaluations,message",
    [
        (
            {"specify_target_noise": True},
            (np.array([[0.0]]), np.array([0.0])),
            "must include positive y_sd",
        ),
        (
            {},
            (np.array([[0.0]]), np.array([0.0]), np.array([0.5])),
            "require.*specify_target_noise",
        ),
        (
            {"uncertainty_handling": True},
            (np.array([[0.0]]), np.array([0.0]), np.array([0.5])),
            "require.*specify_target_noise",
        ),
        (
            {"specify_target_noise": True},
            (np.array([[0.0]]), np.array([0.0]), np.array([0.0])),
            "must be positive",
        ),
    ],
)
def test_precomputed_noise_metadata_validation(options, evaluations, message):
    with pytest.raises(ValueError, match=message):
        _vbmc(np.sum, options=options, precomputed_evaluations=evaluations)


@pytest.mark.parametrize("provided_sd", [False, True])
def test_precomputed_noisy_repeats_pool_without_corrupting_original_values(
    provided_sd,
):
    X = np.array([[0.0], [0.0]])
    y = np.array([0.0, 1.0])
    if provided_sd:
        evaluations = (X, y, np.array([1.0, 2.0]))
        options = {"specify_target_noise": True}
        expected_y = 0.2
        expected_sd = 1 / np.sqrt(1 + 1 / 4)
    else:
        evaluations = (X, y)
        options = {"uncertainty_handling": True}
        expected_y = 0.5
        expected_sd = 1 / np.sqrt(2)

    vbmc = _vbmc(
        lambda x: (0.0, 1.0) if provided_sd else 0.0,
        options=options,
        precomputed_evaluations=evaluations,
    )

    logger = vbmc.function_logger
    assert logger.Xn == 0
    assert logger.n_evals[0, 0] == 2
    assert logger.y_orig[0, 0] == pytest.approx(expected_y)
    assert logger.S[0, 0] == pytest.approx(expected_sd)
    jacobian = vbmc.parameter_transformer.log_abs_det_jacobian(
        vbmc.parameter_transformer(X[:1])
    )[0]
    expected_transformed = expected_y + jacobian
    assert logger.y[0, 0] == pytest.approx(expected_transformed)


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_precomputed_extreme_finite_sds_pool_by_relative_precision(scale):
    vbmc = _vbmc(
        lambda x: (0.0, 1.0),
        options={"specify_target_noise": True},
        precomputed_evaluations=(
            np.array([[0.0], [0.0]]),
            np.array([0.0, 1.0]),
            np.array([scale, 2 * scale]),
        ),
    )

    assert vbmc.function_logger.y_orig[0, 0] == pytest.approx(0.2)
    pooled_ratio = vbmc.function_logger.S[0, 0] / scale
    assert pooled_ratio == pytest.approx(1 / np.sqrt(1.25))


def test_precomputed_points_keep_full_design_and_cover_x0_once():
    calls = []

    def target(x):
        calls.append(np.asarray(x).copy())
        return -np.sum(np.asarray(x) ** 2)

    vbmc = _vbmc(
        target,
        options={"max_fun_evals": 3},
        precomputed_evaluations=(
            np.array([[0.0], [1.5]]),
            np.array([0.0, -2.25]),
        ),
    )
    assert np.array_equal(vbmc.plausible_lower_bounds, [[-1.0]])
    assert np.array_equal(vbmc.plausible_upper_bounds, [[1.0]])
    assert vbmc.optim_state["max_fun_evals"] == 3

    active_sample(
        None,
        4,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    assert len(calls) == vbmc.function_logger.func_count == 3
    assert not any(np.array_equal(call, [0.0]) for call in calls)
    start_idx = np.flatnonzero(vbmc.function_logger.X_orig[:, 0] == 0.0)
    assert start_idx.size == 1
    assert vbmc.function_logger.n_evals[start_idx[0], 0] == 1
    assert vbmc.function_logger.Xn + 1 == 5


def test_cost_only_budget_counts_legacy_cached_start_in_initial_design():
    calls = []

    def target(x):
        calls.append(np.asarray(x).copy())
        return -np.sum(np.asarray(x) ** 2)

    vbmc = _vbmc(
        target,
        options={"max_fun_evals": 5, "f_vals": [0.0]},
        initialization_cost=2,
    )
    assert "skip_logger" not in vbmc.optim_state["cache"]

    active_sample(
        None,
        4,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    assert len(calls) == vbmc.function_logger.func_count == 3
    assert vbmc.function_logger.cache_count == 1
    assert vbmc.function_logger.Xn + 1 == 4


def test_vectorized_initial_design_omits_cached_x0_from_target_batch():
    calls = []

    def target(x):
        calls.append(np.asarray(x).copy())
        return -np.sum(np.asarray(x) ** 2, axis=1)

    vbmc = _vbmc(
        target,
        options={"max_fun_evals": 3, "vectorized_target": True},
        precomputed_evaluations=(np.array([[0.0]]), np.array([0.0])),
    )

    active_sample(
        None,
        4,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.vp,
        vbmc.options,
    )

    assert len(calls) == 1
    assert calls[0].shape == (3, 1)
    assert not np.any(calls[0] == 0.0)
    assert vbmc.function_logger.func_count == 3
    assert vbmc.function_logger.n_evals[0, 0] == 1
    assert vbmc.function_logger.Xn + 1 == 4


def test_optimize_requests_full_first_design_at_exact_fresh_budget(mocker):
    class StopBeforeSampling(Exception):
        pass

    requested = []

    def inspect_first_batch(
        gp, sample_count, optim_state, function_logger, history, vp, options
    ):
        requested.append(sample_count)
        raise StopBeforeSampling

    vbmc = _vbmc(
        np.sum,
        options={"max_fun_evals": 3},
        precomputed_evaluations=(np.array([[0.0]]), np.array([0.0])),
    )
    mocker.patch("pyvbmc.vbmc.vbmc.active_sample", inspect_first_batch)

    with pytest.raises(StopBeforeSampling):
        vbmc.optimize()

    assert requested == [4]


@pytest.mark.parametrize(
    "evaluations,message",
    [
        ((np.array([0.0]), np.array([0.0])), "shape"),
        ((np.array([[0.0]]), np.array([[0.0]])), "shape"),
        ((np.array([[2.0]]), np.array([0.0])), "strictly inside"),
        ((np.array([[np.nan]]), np.array([0.0])), "finite float64"),
        ((np.array([[0.0]]), np.array([np.inf])), "finite float64"),
    ],
)
def test_precomputed_shape_value_and_bounds_validation(evaluations, message):
    with pytest.raises(ValueError, match=message):
        _vbmc(np.sum, precomputed_evaluations=evaluations)


@pytest.mark.parametrize("cost", [-1, 1.5, True])
def test_initialization_cost_validation(cost):
    with pytest.raises(ValueError, match="nonnegative integer"):
        _vbmc(np.sum, initialization_cost=cost)


def test_charged_budget_caps_batches_and_overrides_minimums():
    vbmc = _vbmc(
        np.sum,
        options={"max_fun_evals": 7, "min_fun_evals": 100, "min_iter": 100},
        precomputed_evaluations=(np.array([[0.0]]), np.array([0.0])),
        initialization_cost=3,
    )
    assert vbmc._configured_max_fun_evals == 7
    assert vbmc._effective_max_fun_evals == 4
    assert vbmc._fresh_evaluations_for_batch(5) == 4
    vbmc.function_logger.func_count = 3
    assert vbmc._fresh_evaluations_for_batch(5) == 1

    vbmc.function_logger.func_count = 4
    vbmc.optim_state["iter"] = 0
    finished, message = vbmc._check_termination_conditions()
    assert finished
    assert "max_fun_evals" in message


def test_budget_result_and_summary_are_positive_cost_only(caplog):
    charged = _vbmc(
        np.sum,
        options={"display": "iter", "max_fun_evals": 7},
        precomputed_evaluations=(np.array([[0.0]]), np.array([0.0])),
        initialization_cost=3,
    )
    charged.function_logger.func_count = 4
    charged.optim_state["iter"] = 0
    charged.vp.stats = {"elbo": 0.0, "elbo_sd": 0.1}
    charged.iteration_history.record_iteration(
        {"n_eff": 5, "r_index": np.inf, "stable": False}, 0
    )

    result = charged._create_result_dict(0, "done", False)

    assert result["func_count"] == 4
    assert result["precomputed_observations"] == 1
    assert result["precomputed_locations"] == 1
    assert result["evaluation_budget"] == {
        "unit": "function_equivalent_evaluations",
        "limit": 7,
        "initialization": 3,
        "new_calls": 4,
        "used": 7,
    }
    with caplog.at_level(logging.WARNING):
        charged._log_evaluation_budget_summary()
    assert (
        "Function-equivalent budget: 7 / 7 "
        "(initialization 3 + VBMC 4)" in caplog.text
    )
    caplog.clear()
    charged.options.__setitem__("display", "off", force=True)
    charged._log_evaluation_budget_summary()
    assert "Function-equivalent budget" not in caplog.text

    uncharged = _vbmc(np.sum)
    uncharged.function_logger.func_count = 4
    uncharged.optim_state["iter"] = 0
    uncharged.vp.stats = {"elbo": 0.0, "elbo_sd": 0.1}
    uncharged.iteration_history.record_iteration(
        {"n_eff": 4, "r_index": np.inf, "stable": False}, 0
    )
    ordinary = uncharged._create_result_dict(0, "done", False)
    assert "evaluation_budget" not in ordinary
    caplog.clear()
    uncharged._log_evaluation_budget_summary()
    assert "Function-equivalent budget" not in caplog.text


def test_new_path_rejects_exhausted_or_too_short_initial_budget():
    with pytest.raises(ValueError, match="exhausted"):
        _vbmc(np.sum, options={"max_fun_evals": 2}, initialization_cost=2)
    with pytest.raises(ValueError, match="insufficient.*initial design"):
        _vbmc(np.sum, options={"max_fun_evals": 3}, initialization_cost=0)


def test_short_precomputed_budget_has_finite_gp_training_schedule(mocker):
    vbmc = _vbmc(
        np.sum,
        options={"max_fun_evals": 3},
        precomputed_evaluations=(np.array([[0.0]]), np.array([0.0])),
    )
    vbmc.optim_state.update(n_eff=4, iter=0)
    mocker.patch(
        "pyvbmc.vbmc.gaussian_process_train._get_hyp_cov",
        return_value=None,
    )

    training = _get_gp_training_options(
        vbmc.optim_state,
        vbmc.iteration_history,
        vbmc.options,
        {},
        gp_s_N=0,
    )

    assert np.isfinite(training["init_N"])
    assert training["init_N"] >= 9


def test_precomputed_observations_do_not_extrapolate_gp_schedule(mocker):
    vbmc = _vbmc(
        np.sum,
        precomputed_evaluations=(np.array([[0.0]]), np.array([0.0])),
    )
    vbmc.optim_state.update(n_eff=100, iter=0)
    mocker.patch(
        "pyvbmc.vbmc.gaussian_process_train._get_hyp_cov",
        return_value=None,
    )

    training = _get_gp_training_options(
        vbmc.optim_state,
        vbmc.iteration_history,
        vbmc.options,
        {},
        gp_s_N=0,
    )

    assert training["init_N"] == max(vbmc.options["gp_train_n_init_final"], 9)


def test_precomputed_budget_state_survives_save_load_and_override(tmp_path):
    vbmc = _vbmc(
        np.sum,
        options={"max_fun_evals": 9},
        precomputed_evaluations=(np.array([[0.0]]), np.array([0.0])),
        initialization_cost=5,
    )
    path = tmp_path / "precomputed"
    vbmc.save(path)

    loaded = VBMC.load(path)
    assert loaded.initialization_cost == 5
    assert loaded._configured_max_fun_evals == 9
    assert loaded._effective_max_fun_evals == 4
    assert loaded.function_logger.cache_count == 1
    assert loaded.function_logger.n_evals[0, 0] == 1

    extended = VBMC.load(path, new_options={"max_fun_evals": 12})
    assert extended._configured_max_fun_evals == 12
    assert extended._effective_max_fun_evals == 7
    assert extended.optim_state["max_fun_evals"] == 7
    assert extended.function_logger.cache_count == 1

    exact = VBMC.load(path, new_options={"max_fun_evals": 8})
    assert exact._effective_max_fun_evals == 3
    with pytest.raises(ValueError, match="insufficient.*initial design"):
        VBMC.load(path, new_options={"max_fun_evals": 7})


def test_old_save_budget_migration_defaults_to_zero(tmp_path):
    vbmc = _vbmc(np.sum)
    del vbmc.initialization_cost
    del vbmc._budget_active
    del vbmc._configured_max_fun_evals
    del vbmc._effective_max_fun_evals
    del vbmc.precomputed_evaluations
    del vbmc.precomputed_observation_count
    del vbmc.precomputed_location_count
    path = tmp_path / "old"
    vbmc.save(path)

    loaded = VBMC.load(path, new_options={"max_fun_evals": 30})
    assert loaded.initialization_cost == 0
    assert loaded._budget_active is False
    assert loaded._configured_max_fun_evals == 30
    assert loaded._effective_max_fun_evals == 30
    assert loaded.precomputed_evaluations is None
