"""Density, layout, and coordinate maps of :class:`PyMCTarget`."""

import copy

import numpy as np
import pytest

pm = pytest.importorskip("pymc")
pytest.importorskip("arviz_base")

from pyvbmc import VariationalPosterior
from pyvbmc.pymc import PyMCTarget
from pyvbmc.pymc._compat import closed_form
from pyvbmc.testing._dtype import assert_float64
from pyvbmc.testing.pymc.models import (
    accepted_models,
    mixed_order_model,
    nonfinite_mode_model,
)


@pytest.fixture(scope="module")
def accepted_targets():
    return [
        (spec, PyMCTarget(spec["model"], seed=710 + index))
        for index, spec in enumerate(accepted_models())
    ]


def _model_point(target, x):
    values = target.to_model_variables(np.asarray(x)[None])
    return {name: value[0] for name, value in values.items()}


def test_accepted_layout_and_support(accepted_targets):
    by_name = {spec["name"]: target for spec, target in accepted_targets}

    scalar = by_name["scalar"]
    assert scalar.names == ["mu"]
    assert scalar.shapes == [()]
    assert scalar.sizes == [1]
    assert scalar.value_names == ["mu"]
    assert scalar.coordinate_names == ["mu"]
    assert scalar.kept == {}

    positive = by_name["positive"]
    assert positive.names == ["sigma"]
    assert positive.value_names == ["sigma_log__"]
    assert positive.coordinate_names == ["sigma_log__"]
    assert positive.kept == {"sigma": "LogTransform"}
    np.testing.assert_array_equal(positive.support["sigma"][0], 0.0)
    assert np.isposinf(positive.support["sigma"][1])

    vector = by_name["vector"]
    assert vector.names == ["beta", "sigma"]
    assert vector.shapes == [(2,), ()]
    assert vector.sizes == [2, 1]
    assert vector.value_names == ["beta", "sigma_log__"]
    assert vector.coordinate_names == [
        "beta[intercept]",
        "beta[slope]",
        "sigma_log__",
    ]
    assert vector.dims == {"beta": ("coef",), "sigma": ()}
    np.testing.assert_array_equal(
        vector.coords["coef"], ["intercept", "slope"]
    )

    bounded = by_name["bounded"]
    assert bounded.names == ["p", "u"]
    assert bounded.value_names == ["p", "u"]
    assert bounded.coordinate_names == ["p", "u[0]", "u[1]"]
    assert bounded.kept == {}
    np.testing.assert_array_equal(bounded.lb, [[0.0, -1.0, 0.0]])
    np.testing.assert_array_equal(bounded.ub, [[1.0, 2.0, 1.0]])

    one_sided = by_name["one_sided"]
    assert one_sided.names == ["t", "v"]
    assert set(one_sided.kept) == {"t", "v"}
    assert all(name.endswith("_interval__") for name in one_sided.value_names)

    for spec, target in accepted_targets:
        assert target.model is spec["model"]
        assert target.D == sum(target.sizes)
        assert (
            target.x0.shape
            == target.lb.shape
            == target.ub.shape
            == (
                1,
                target.D,
            )
        )
        assert len(target.coordinate_names) == target.D
        assert_float64(
            {
                "x0": target.x0,
                "lb": target.lb,
                "ub": target.ub,
                "plb": target.plb,
                "pub": target.pub,
            },
            path=spec["name"],
            min_leaves=5,
        )


def test_source_free_variable_order_survives_partial_reconstruction():
    model = mixed_order_model()
    target = PyMCTarget(
        model,
        start={
            "beta": np.array([0.0, 0.1]),
            "sigma": 1.0,
            "t": 1.5,
            "p": 0.4,
        },
        plausible_bounds={
            "beta": (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
            "sigma": (0.2, 2.0),
            "t": (1.1, 3.0),
            "p": (0.1, 0.9),
        },
        setup_budget=1,
        seed=714,
    )
    assert target.names == ["beta", "sigma", "t", "p"]
    assert target.value_names[:2] == ["beta", "sigma_log__"]
    assert target.value_names[2].startswith("t_")
    assert target.value_names[2].endswith("__")
    assert target.value_names[3] == "p"
    assert target.coordinate_names[:2] == [
        "beta[intercept]",
        "beta[slope]",
    ]
    assert target.model is model


def test_densities_match_hand_written_references(accepted_targets):
    rng = np.random.default_rng(20260916)
    for spec, target in accepted_targets:
        X = rng.uniform(target.plb, target.pub, size=(25, target.D))
        delta = np.array(
            [
                target.log_joint_no_jacobian(x)
                - spec["hand"](_model_point(target, x))
                for x in X
            ]
        )
        offset = np.mean(delta)
        assert abs(offset) <= 1e-6, spec["name"]
        assert np.max(np.abs(delta - offset)) <= 1e-8, spec["name"]

        kept_blocks = [
            np.arange(target._offsets[i], target._offsets[i + 1])
            for i, name in enumerate(target.names)
            if name in target.kept
        ]
        kept_indices = (
            np.concatenate(kept_blocks).astype(int)
            if kept_blocks
            else np.empty(0, dtype=int)
        )
        expected = (
            np.sum(X[:, kept_indices], axis=1)
            if kept_indices.size
            else np.zeros(len(X))
        )
        actual = np.array(
            [target.log_joint(x) - target.log_joint_no_jacobian(x) for x in X]
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
        assert target.log_joint(X[0:1]) == pytest.approx(
            target.log_joint(X[0])
        )


def test_hard_bound_behavior(accepted_targets):
    for _, target in accepted_targets:
        for coordinate in np.flatnonzero(np.isfinite(target.lb.ravel())):
            point = target.x0.ravel().copy()
            point[coordinate] = target.lb.ravel()[coordinate]
            assert target.log_joint(point) == -np.inf
            point[coordinate] -= 1.0
            assert target.log_joint(point) == -np.inf
            point[coordinate] = np.nextafter(
                target.lb.ravel()[coordinate], np.inf
            )
            assert np.isfinite(target.log_joint(point))
        for coordinate in np.flatnonzero(np.isfinite(target.ub.ravel())):
            point = target.x0.ravel().copy()
            point[coordinate] = target.ub.ravel()[coordinate]
            assert target.log_joint(point) == -np.inf
            point[coordinate] += 1.0
            assert target.log_joint(point) == -np.inf
            point[coordinate] = np.nextafter(
                target.ub.ravel()[coordinate], -np.inf
            )
            assert np.isfinite(target.log_joint(point))


def test_flatten_and_model_coordinate_round_trips(accepted_targets):
    for _, target in accepted_targets:
        x = target.x0.ravel()
        np.testing.assert_array_equal(target.flatten(target.unflatten(x)), x)
        model_values = target.to_model_variables(np.vstack((x, x + 0.1)))
        np.testing.assert_allclose(
            target.from_model_variables(
                {name: value[0] for name, value in model_values.items()}
            ),
            x,
            rtol=0,
            atol=1e-12,
        )
        for i, name in enumerate(target.names):
            if name not in target.kept:
                continue
            block = x[target._offsets[i] : target._offsets[i + 1]].reshape(
                target.shapes[i]
            )
            expected = closed_form(
                target._maps[name]["kind"],
                block,
                *target.support[name],
            )
            np.testing.assert_allclose(
                model_values[name][0], expected, rtol=0, atol=1e-12
            )


def test_fixed_flat_coordinate_oracles_and_export(accepted_targets):
    by_name = {spec["name"]: target for spec, target in accepted_targets}

    vector = by_name["vector"]
    vector_flat = np.array([[1.25, -2.5, np.log(0.7)]])
    vector_values = vector.to_model_variables(vector_flat)
    np.testing.assert_array_equal(vector_values["beta"], [[1.25, -2.5]])
    np.testing.assert_allclose(
        vector_values["sigma"], [0.7], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        vector.from_model_variables(
            {"beta": np.array([1.25, -2.5]), "sigma": 0.7}
        ),
        vector_flat[0],
        rtol=0,
        atol=1e-15,
    )

    bounded = by_name["bounded"]
    bounded_flat = np.array([[0.25, -0.75, 0.6]])
    bounded_values = bounded.to_model_variables(bounded_flat)
    np.testing.assert_array_equal(bounded_values["p"], [0.25])
    np.testing.assert_array_equal(bounded_values["u"], [[-0.75, 0.6]])
    np.testing.assert_array_equal(
        bounded.from_model_variables({"p": 0.25, "u": np.array([-0.75, 0.6])}),
        bounded_flat[0],
    )

    one_sided = by_name["one_sided"]
    one_sided_flat = np.array([[np.log(0.7), np.log(0.4)]])
    one_sided_values = one_sided.to_model_variables(one_sided_flat)
    np.testing.assert_allclose(
        one_sided_values["t"], [1.7], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        one_sided_values["v"], [-0.9], rtol=0, atol=1e-15
    )

    class FixedPosterior:
        D = 3

        @staticmethod
        def sample(n_samples, orig_flag=True):
            assert n_samples == 1
            assert orig_flag
            return bounded_flat.copy(), None

    posterior = bounded.to_arviz(FixedPosterior(), 1)["posterior"]
    np.testing.assert_array_equal(posterior.p.values, [[0.25]])
    np.testing.assert_array_equal(posterior.u.values, [[[-0.75, 0.6]]])


def test_mapping_validation_and_string_table(accepted_targets):
    target = accepted_targets[2][1]
    with pytest.raises(ValueError, match="values.*beta"):
        target.flatten({"sigma_log__": np.asarray(0.0)})
    with pytest.raises(ValueError, match="shape"):
        target.flatten(
            {"beta": np.zeros((1, 2)), "sigma_log__": np.asarray(0.0)}
        )
    with pytest.raises(ValueError, match="shape"):
        target.unflatten(np.zeros(target.D + 1))
    with pytest.raises(ValueError, match="point.*sigma"):
        target.from_model_variables({"beta": np.zeros(2)})
    table = str(target)
    assert "variable | shape | VBMC coordinates" in table
    assert "beta[intercept]" in table
    assert "sigma_log__" in table


@pytest.mark.parametrize("value", [1000.0, -1000.0])
def test_model_mapping_rejects_nonfinite_or_boundary_transform_output(
    accepted_targets, value
):
    positive = next(
        target
        for spec, target in accepted_targets
        if spec["name"] == "positive"
    )
    with pytest.raises(
        ValueError,
        match=r"Mapped model values for 'sigma'.*strictly inside",
    ):
        positive.to_model_variables(np.array([[value]]))


def test_export_mapping_failure_restores_posterior_rng(
    accepted_targets, monkeypatch
):
    positive = next(
        target
        for spec, target in accepted_targets
        if spec["name"] == "positive"
    )
    vp = VariationalPosterior(positive.D, rng=715)
    state = copy.deepcopy(vp.rng.bit_generator.state)

    def extreme_sample(n_samples, orig_flag=True):
        vp.rng.standard_normal()
        return np.full((n_samples, 1), 1000.0), None

    monkeypatch.setattr(vp, "sample", extreme_sample)
    with pytest.raises(ValueError, match="Mapped model values for 'sigma'"):
        positive.to_arviz(vp, 3)
    assert vp.rng.bit_generator.state == state


@pytest.mark.parametrize("unnamed_size", [2, 3])
def test_generated_export_dimension_avoids_declared_names(unnamed_size):
    with pm.Model(coords={"a_dim_0": ["u", "v"]}) as model:
        pm.Normal("a", shape=unnamed_size)
        pm.Normal("b", dims="a_dim_0")
    target = PyMCTarget(
        model,
        start={"a": np.zeros(unnamed_size), "b": np.zeros(2)},
        plausible_bounds={
            "a": (-np.ones(unnamed_size), np.ones(unnamed_size)),
            "b": (-np.ones(2), np.ones(2)),
        },
        setup_budget=1,
        seed=716,
    )
    assert target.dims["a"] == ("a_dim_0_1",)
    assert target.dims["b"] == ("a_dim_0",)

    class FixedPosterior:
        D = unnamed_size + 2

        @staticmethod
        def sample(n_samples, orig_flag=True):
            return np.zeros((n_samples, unnamed_size + 2)), None

    posterior = target.to_arviz(FixedPosterior(), 2)["posterior"]
    assert posterior.a.dims == ("chain", "draw", "a_dim_0_1")
    assert posterior.b.dims == ("chain", "draw", "a_dim_0")
    np.testing.assert_array_equal(
        posterior.coords["a_dim_0_1"], np.arange(unnamed_size)
    )
    np.testing.assert_array_equal(posterior.coords["a_dim_0"], ["u", "v"])


def test_nonfinite_mode_names_model_variable():
    with pytest.raises(ValueError, match=r"Mode search failed.*x="):
        PyMCTarget(nonfinite_mode_model(), seed=33)
