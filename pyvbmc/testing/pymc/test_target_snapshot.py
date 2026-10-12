"""Immutability of a fully constructed PyMC target snapshot."""

import copy

import numpy as np
import pytest

pm = pytest.importorskip("pymc")
pytest.importorskip("arviz_base")
import pytensor

from pyvbmc import VariationalPosterior
from pyvbmc.pymc import PyMCTarget
from pyvbmc.pymc._snapshot import _numeric_shared_inputs


def _posterior(target, seed=921):
    vp = VariationalPosterior(target.D, 2, x0=target.x0, rng=seed)
    vp.sigma[:] = 0.12
    return vp


def _target_state(target):
    return {
        "names": copy.deepcopy(target.names),
        "shapes": copy.deepcopy(target.shapes),
        "sizes": copy.deepcopy(target.sizes),
        "value_names": copy.deepcopy(target.value_names),
        "coordinate_names": copy.deepcopy(target.coordinate_names),
        "kept": copy.deepcopy(target.kept),
        "support": {
            name: tuple(np.array(bound, copy=True) for bound in bounds)
            for name, bounds in target.support.items()
        },
        "dims": copy.deepcopy(target.dims),
        "coords": {
            name: np.array(values, copy=True)
            for name, values in target.coords.items()
        },
        "x0": target.x0.copy(),
        "lb": target.lb.copy(),
        "ub": target.ub.copy(),
        "plb": target.plb.copy(),
        "pub": target.pub.copy(),
        "setup": tuple(value.copy() for value in target.setup_evaluations),
        "setup_budget": target.setup_budget,
        "setup_cost": target.setup_cost,
        "plausible_info": copy.deepcopy(target.plausible_info),
    }


def _assert_target_state(target, expected):
    for name in (
        "names",
        "shapes",
        "sizes",
        "value_names",
        "coordinate_names",
        "kept",
        "dims",
        "setup_budget",
        "setup_cost",
        "plausible_info",
    ):
        assert getattr(target, name) == expected[name]
    for name, bounds in expected["support"].items():
        for actual, reference in zip(target.support[name], bounds):
            np.testing.assert_array_equal(actual, reference)
    assert set(target.coords) == set(expected["coords"])
    for name, values in expected["coords"].items():
        np.testing.assert_array_equal(target.coords[name], values)
    for name in ("x0", "lb", "ub", "plb", "pub"):
        np.testing.assert_array_equal(getattr(target, name), expected[name])
    for actual, reference in zip(target.setup_evaluations, expected["setup"]):
        np.testing.assert_array_equal(actual, reference)


@pytest.fixture(scope="module")
def snapshot_case():
    covariate = np.array([-1.0, 1.0])
    observed = np.array([-0.2, 1.1])
    registered_upper = np.asarray(3.0)
    labels = ["a", "b"]
    raw_scale = pytensor.shared(np.asarray(1.0), name="raw_scale")
    raw_upper = pytensor.shared(np.asarray(2.0), name="raw_upper")

    with pm.Model(coords={"observation": labels}) as model:
        covariate_data = pm.Data("covariate", covariate, dims="observation")
        observed_data = pm.Data("observed_data", observed, dims="observation")
        support_data = pm.Data("registered_upper", registered_upper)
        x = pm.Uniform(
            "x", lower=0.0, upper=support_data, initval=np.asarray(0.7)
        )
        z = pm.Uniform(
            "z", lower=0.0, upper=raw_upper, initval=np.asarray(0.8)
        )
        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=raw_scale,
            dims="observation",
            initval="support_point",
        )
        mu = pm.Deterministic(
            "mu", x + z + beta * covariate_data, dims="observation"
        )
        pm.Normal(
            "y",
            mu=mu,
            sigma=raw_scale,
            observed=observed_data,
            dims="observation",
        )

    kwargs = {
        "start": {"x": 0.7, "z": 0.8, "beta": np.zeros(2)},
        "plausible_bounds": {
            "x": (0.2, 2.5),
            "z": (0.2, 1.6),
            "beta": (-np.ones(2), np.ones(2)),
        },
        "seed": 919,
    }
    target = PyMCTarget(model, **kwargs)
    points = np.vstack(
        (
            target.x0,
            0.75 * target.x0 + 0.25 * target.plb,
            0.75 * target.x0 + 0.25 * target.pub,
        )
    )
    density = np.array([target.log_joint(point) for point in points])
    mappings = target.to_model_variables(points)
    state = _target_state(target)
    export = target.to_arviz(_posterior(target), 7)["posterior"]
    export_values = {name: export[name].values.copy() for name in target.names}
    export_layout = {
        name: (export[name].dims, export[name].shape) for name in target.names
    }
    export_attrs = dict(export.attrs)

    covariate[:] = [50.0, 60.0]
    observed[:] = [70.0, 80.0]
    registered_upper[...] = 30.0
    labels[:] = ["caller-a", "caller-b"]
    raw_scale.set_value(np.asarray(1.8))
    raw_upper.set_value(np.asarray(5.0))
    changed_covariate = np.array([2.0, -0.5])
    changed_observed = np.array([1.5, -1.0])
    changed_labels = ["new-a", "new-b"]
    with model:
        pm.set_data(
            {
                "covariate": changed_covariate,
                "observed_data": changed_observed,
                "registered_upper": np.asarray(4.0),
            },
            coords={"observation": changed_labels},
        )

    fresh = PyMCTarget(model, **kwargs)
    return {
        "target": target,
        "fresh": fresh,
        "points": points,
        "density": density,
        "mappings": mappings,
        "state": state,
        "export_values": export_values,
        "export_layout": export_layout,
        "export_attrs": export_attrs,
        "changed_covariate": changed_covariate,
        "changed_labels": changed_labels,
    }


def test_target_snapshot_is_immutable_after_every_source_changes(
    snapshot_case,
):
    target = snapshot_case["target"]
    _assert_target_state(target, snapshot_case["state"])
    actual = np.array(
        [target.log_joint(point) for point in snapshot_case["points"]]
    )
    np.testing.assert_allclose(
        actual, snapshot_case["density"], rtol=0, atol=1e-12
    )
    mapped = target.to_model_variables(snapshot_case["points"])
    for name, expected in snapshot_case["mappings"].items():
        np.testing.assert_array_equal(mapped[name], expected)
    recovered = np.vstack(
        [
            target.from_model_variables(
                {name: value[row] for name, value in mapped.items()}
            )
            for row in range(len(snapshot_case["points"]))
        ]
    )
    np.testing.assert_array_equal(recovered, snapshot_case["points"])
    assert not _numeric_shared_inputs(target._snapshot)

    initial = {
        rv.name: target._snapshot.rvs_to_initial_values[rv]
        for rv in target._snapshot.free_RVs
    }
    np.testing.assert_array_equal(initial["x"], np.asarray(0.7))
    np.testing.assert_array_equal(initial["z"], np.asarray(0.8))
    assert initial["beta"] == "support_point"

    export = target.to_arviz(_posterior(target), 7)["posterior"]
    for name, expected in snapshot_case["export_values"].items():
        np.testing.assert_array_equal(export[name].values, expected)
        assert (export[name].dims, export[name].shape) == snapshot_case[
            "export_layout"
        ][name]
    # ArviZ timestamps each export; inference metadata remains fixed.
    assert {
        key: value
        for key, value in export.attrs.items()
        if key != "created_at"
    } == {
        key: value
        for key, value in snapshot_case["export_attrs"].items()
        if key != "created_at"
    }
    np.testing.assert_array_equal(export.coords["observation"], ["a", "b"])


def test_fresh_target_reflects_changed_inputs_while_prediction_model_is_live(
    snapshot_case,
):
    target = snapshot_case["target"]
    fresh = snapshot_case["fresh"]
    assert fresh.support["x"][1] == pytest.approx(4.0)
    assert fresh.support["z"][1] == pytest.approx(5.0)
    np.testing.assert_array_equal(
        fresh.coords["observation"], snapshot_case["changed_labels"]
    )
    fresh_density = np.array(
        [fresh.log_joint(point) for point in snapshot_case["points"]]
    )
    assert not np.allclose(fresh_density, snapshot_case["density"])
    assert not np.array_equal(
        fresh.setup_evaluations[1], target.setup_evaluations[1]
    )

    data = target.to_arviz(_posterior(target, seed=922), 6)
    deterministics = pm.compute_deterministics(
        data,
        model=target.model,
        var_names=["mu"],
        progressbar=False,
    )
    posterior = data["posterior"]
    expected_mu = (
        posterior.x.values[..., None]
        + posterior.z.values[..., None]
        + posterior.beta.values * snapshot_case["changed_covariate"]
    )
    np.testing.assert_allclose(
        deterministics.mu.values, expected_mu, rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(
        deterministics.coords["observation"],
        snapshot_case["changed_labels"],
    )

    frozen_again = np.array(
        [target.log_joint(point) for point in snapshot_case["points"]]
    )
    np.testing.assert_allclose(
        frozen_again, snapshot_case["density"], rtol=0, atol=1e-12
    )
