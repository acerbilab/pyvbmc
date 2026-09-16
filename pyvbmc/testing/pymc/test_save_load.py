"""Persistence and copying of PyMC targets bound to VBMC."""

import copy

import dill
import numpy as np
import pytest

pm = pytest.importorskip("pymc")
pytest.importorskip("arviz_base")

from pyvbmc import VBMC, VariationalPosterior
from pyvbmc.pymc import PyMCTarget


def _initial_values(model):
    return {
        rv.name: copy.deepcopy(model.rvs_to_initial_values[rv])
        for rv in model.free_RVs
    }


def _posterior(target, seed=914):
    vp = VariationalPosterior(target.D, 2, x0=target.x0, rng=seed)
    vp.sigma[:] = 0.15
    return vp


@pytest.fixture(scope="module")
def lifecycle_target():
    covariate = np.array([-1.0, 0.0, 1.0])
    observed = np.array([-0.4, 0.3, 1.2])
    labels = ["left", "center", "right"]
    with pm.Model(coords={"observation": labels}) as model:
        x_data = pm.Data("x_data", covariate, dims="observation")
        y_data = pm.Data("y_data", observed, dims="observation")
        beta = pm.Normal("beta", 0.0, 2.0, initval=np.asarray(0.25))
        offset = pm.Normal("offset", 0.0, 2.0, initval="support_point")
        mu = pm.Deterministic("mu", offset + beta * x_data, dims="observation")
        pm.Normal("y", mu, 0.7, observed=y_data, dims="observation")

    target = PyMCTarget(
        model,
        start={"beta": 0.25, "offset": 0.0},
        plausible_bounds={
            "beta": (-1.5, 1.5),
            "offset": (-1.5, 1.5),
        },
        seed=913,
    )
    points = np.vstack(
        (
            target.x0,
            0.75 * target.x0 + 0.25 * target.plb,
            0.75 * target.x0 + 0.25 * target.pub,
        )
    )
    values = np.array([target.log_joint(point) for point in points])
    snapshot_initial = _initial_values(target._snapshot)

    changed_covariate = np.array([2.0, -1.0, 0.5])
    changed_observed = np.array([1.1, -0.2, 0.8])
    with model:
        pm.set_data(
            {"x_data": changed_covariate, "y_data": changed_observed},
            coords={"observation": ["new-left", "new-center", "new-right"]},
        )

    return {
        "target": target,
        "points": points,
        "values": values,
        "changed_covariate": changed_covariate,
        "snapshot_initial": snapshot_initial,
    }


def _assert_density(target, points, expected):
    evaluate = getattr(target, "log_joint", target)
    actual = np.array([evaluate(point) for point in points])
    np.testing.assert_array_equal(actual, expected)


def _assert_nondefault_initial_values(target, expected):
    actual = _initial_values(target._snapshot)
    assert actual["offset"] == expected["offset"] == "support_point"
    np.testing.assert_array_equal(actual["beta"], expected["beta"])


def test_vbmc_target_save_load_preserves_density_budget_and_public_workflow(
    lifecycle_target, tmp_path
):
    target = lifecycle_target["target"]
    vbmc = VBMC(
        target,
        options={"display": "off", "max_fun_evals": 60},
        seed=915,
    )
    logger_before = {
        "func_count": vbmc.function_logger.func_count,
        "cache_count": vbmc.function_logger.cache_count,
        "Xn": vbmc.function_logger.Xn,
        "X_flag": vbmc.function_logger.X_flag.copy(),
        "X_orig": vbmc.function_logger.X_orig.copy(),
        "y_orig": vbmc.function_logger.y_orig.copy(),
        "n_evals": vbmc.function_logger.n_evals.copy(),
    }
    budget_before = (
        vbmc.initialization_cost,
        vbmc._configured_max_fun_evals,
        vbmc._effective_max_fun_evals,
    )

    path = tmp_path / "pymc_target"
    vbmc.save(path)
    loaded = VBMC.load(path)

    assert loaded.target is loaded.log_joint.__self__
    assert loaded.target is loaded.function_logger.fun.__self__
    _assert_density(
        loaded.function_logger.fun,
        lifecycle_target["points"],
        lifecycle_target["values"],
    )
    assert (
        loaded.initialization_cost,
        loaded._configured_max_fun_evals,
        loaded._effective_max_fun_evals,
    ) == budget_before
    assert loaded.precomputed_observation_count == len(
        target.setup_evaluations[1]
    )
    assert loaded.precomputed_location_count == len(
        np.unique(target.setup_evaluations[0], axis=0)
    )
    for actual, expected in zip(
        loaded.precomputed_evaluations, target.setup_evaluations
    ):
        np.testing.assert_array_equal(actual, expected)
    assert loaded.optim_state["initialization_cost"] == target.setup_cost
    assert loaded.optim_state["max_fun_evals"] == 60 - target.setup_cost
    assert (
        loaded.function_logger.func_count == logger_before["func_count"] == 0
    )
    assert loaded.function_logger.cache_count == logger_before["cache_count"]
    assert loaded.function_logger.Xn == logger_before["Xn"]
    for name in ("X_flag", "X_orig", "y_orig", "n_evals"):
        np.testing.assert_array_equal(
            getattr(loaded.function_logger, name), logger_before[name]
        )

    loaded.vp = _posterior(loaded.target)
    data = loaded.target.to_arviz(loaded.vp, 12)
    predictive = pm.sample_posterior_predictive(
        data,
        model=loaded.target.model,
        var_names=["mu", "y"],
        random_seed=916,
        progressbar=False,
    )["posterior_predictive"]
    posterior = data["posterior"]
    expected_mu = (
        posterior.offset.values[..., None]
        + posterior.beta.values[..., None]
        * lifecycle_target["changed_covariate"]
    )
    np.testing.assert_allclose(
        predictive.mu.values, expected_mu, rtol=0, atol=1e-12
    )
    assert predictive.y.shape == (1, 12, 3)


def test_target_deepcopy_and_dill_round_trip(lifecycle_target):
    target = lifecycle_target["target"]
    copied = copy.deepcopy(target)
    restored = dill.loads(dill.dumps(target, recurse=True))

    assert copied is not target
    assert copied.model is not target.model
    assert restored is not target
    assert restored.model is not target.model
    for candidate in (copied, restored):
        _assert_density(
            candidate,
            lifecycle_target["points"],
            lifecycle_target["values"],
        )
        _assert_nondefault_initial_values(
            candidate, lifecycle_target["snapshot_initial"]
        )

    data = copied.to_arviz(_posterior(copied, seed=917), 5)
    prediction = pm.sample_posterior_predictive(
        data,
        model=copied.model,
        var_names=["mu"],
        random_seed=918,
        progressbar=False,
    )["posterior_predictive"]
    posterior = data["posterior"]
    expected_mu = (
        posterior.offset.values[..., None]
        + posterior.beta.values[..., None]
        * lifecycle_target["changed_covariate"]
    )
    np.testing.assert_allclose(
        prediction.mu.values, expected_mu, rtol=0, atol=1e-12
    )
