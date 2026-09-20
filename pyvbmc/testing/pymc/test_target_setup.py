"""Target-level setup routes, diagnostics, and budget accounting."""

import copy

import numpy as np
import pytest

pm = pytest.importorskip("pymc")
pytest.importorskip("arviz_base")

from pyvbmc import VBMC
from pyvbmc.pymc import PyMCTarget, UnsupportedModel
from pyvbmc.pymc import _target as target_module
from pyvbmc.testing.pymc.models import (
    bounded_model,
    no_gradient_model,
    scalar_model,
    schools_model,
    stochastic_initial_model,
    undefined_gradient_model,
    undefined_hessian_model,
    vector_model,
)


@pytest.fixture(scope="module")
def scalar_target():
    spec = scalar_model(np.random.default_rng(801))
    return spec, PyMCTarget(spec["model"], seed=0)


@pytest.fixture(scope="module")
def diagnostic_targets():
    bounded = bounded_model(np.random.default_rng(802))
    return {
        "bounded": PyMCTarget(bounded["model"], seed=0),
        "centered": PyMCTarget(schools_model(True), seed=0),
        "noncentered": PyMCTarget(schools_model(False), seed=0),
        "no_gradient": PyMCTarget(no_gradient_model(), seed=0),
    }


def _assert_accounting(target, *, atol=1e-8):
    info = target.plausible_info
    X, y = target.setup_evaluations
    assert target.setup_cost == info["n_evaluations"]
    assert target.setup_cost == (info["n_target_calls"] + info["hessian_cost"])
    assert info["hessian_cost"] == target.D * info["n_hessian_calls"]
    assert target.setup_cost <= target.setup_budget
    assert X.shape == (len(y), target.D)
    assert len(y) <= info["n_target_calls"]
    for point, value in zip(X, y):
        assert target.log_joint(point) == pytest.approx(
            value, rel=1e-12, abs=atol
        )


def test_scalar_laplace_box_and_exact_accounting(scalar_target):
    spec, target = scalar_target
    expected = np.array(
        [
            spec["truth_mean"] - 3 * spec["truth_sd"],
            spec["truth_mean"] + 3 * spec["truth_sd"],
        ]
    )
    np.testing.assert_allclose(
        [target.plb.item(), target.pub.item()], expected, rtol=1e-3
    )
    info = target.plausible_info
    assert set(info) == {
        "route",
        "start",
        "mode",
        "n_evaluations",
        "n_target_calls",
        "n_hessian_calls",
        "hessian_cost",
        "cap_reached",
        "start_moved",
        "curvature",
        "location_outside_prior",
        "clipped",
    }
    assert info["route"] == "laplace"
    assert info["start"] == "mode"
    assert set(info["mode"]) == {"mu"}
    assert info["n_hessian_calls"] == 1
    assert not info["cap_reached"]
    assert info["start_moved"] == []
    assert info["curvature"] == []
    assert info["location_outside_prior"] == []
    assert info["clipped"] == []
    assert target.setup_budget == 25
    _assert_accounting(target)
    assert len(target.setup_evaluations[0]) + 1 == target.setup_cost


def test_setup_diagnostic_routes_and_warnings(diagnostic_targets, caplog):
    for target in diagnostic_targets.values():
        _assert_accounting(target)

    bounded = diagnostic_targets["bounded"]
    assert bounded.plausible_info["route"] == "laplace"
    assert set(bounded.plausible_info["curvature"]) == set(
        bounded.coordinate_names
    )
    assert {"u[0]", "u[1]"} <= set(bounded.plausible_info["clipped"])

    centered = diagnostic_targets["centered"]
    assert centered.plausible_info["cap_reached"]
    assert set(centered.plausible_info["curvature"]) == set(
        centered.coordinate_names
    )
    assert centered.plausible_info["location_outside_prior"] == ["tau_log__"]
    assert centered.x0[0, 1] == pytest.approx(
        centered.from_model_variables(centered.plausible_info["mode"])[1]
    )

    noncentered = diagnostic_targets["noncentered"]
    assert not noncentered.plausible_info["cap_reached"]
    assert noncentered.plausible_info["curvature"] == []
    assert noncentered.plausible_info["location_outside_prior"] == []

    no_gradient = diagnostic_targets["no_gradient"]
    assert no_gradient.plausible_info["route"] == "prior"
    assert no_gradient.plausible_info["start"] == "initial"
    assert no_gradient.setup_cost == 1
    assert no_gradient.plausible_info["n_hessian_calls"] == 0
    assert np.isfinite(no_gradient.log_joint(no_gradient.x0))

    caplog.clear()
    no_gradient._report_setup_warnings()
    centered._report_setup_warnings()
    messages = [record.getMessage() for record in caplog.records]
    assert any("gradient is unavailable" in message for message in messages)
    assert any("setup call cap" in message for message in messages)
    assert any("unusable curvature" in message for message in messages)
    assert any("outside the prior location" in message for message in messages)


def test_absent_scalar_gradient_takes_the_prior_route(caplog):
    caplog.clear()
    target = PyMCTarget(undefined_gradient_model(), seed=14)
    assert target.plausible_info["route"] == "prior"
    assert target.plausible_info["start"] == "initial"
    assert target.plausible_info["n_hessian_calls"] == 0
    assert np.isfinite(target.log_joint(target.x0))
    _assert_accounting(target)
    messages = [record.getMessage() for record in caplog.records]
    assert any("gradient is unavailable" in message for message in messages)


def test_prior_route_warning_names_the_starting_point(caplog):
    caplog.clear()
    automatic = PyMCTarget(undefined_gradient_model(), seed=16)
    automatic_messages = [record.getMessage() for record in caplog.records]
    caplog.clear()
    supplied = PyMCTarget(
        undefined_gradient_model(), start={"x": 0.5}, seed=16
    )
    supplied_messages = [record.getMessage() for record in caplog.records]

    assert automatic.plausible_info["start"] == "initial"
    assert supplied.plausible_info["start"] == "user"
    assert automatic.plausible_info["route"] == "prior"
    assert supplied.plausible_info["route"] == "prior"
    assert any(
        "gradient is unavailable; using the model's initial point" in message
        for message in automatic_messages
    )
    assert any(
        "gradient is unavailable; using the supplied starting point" in message
        for message in supplied_messages
    )


def test_absent_scalar_hessian_keeps_the_gradient_and_prior_widths():
    target = PyMCTarget(undefined_hessian_model(), seed=15)
    assert target.plausible_info["route"] == "laplace"
    assert target.plausible_info["start"] == "mode"
    assert target.plausible_info["n_hessian_calls"] == 0
    assert target.plausible_info["hessian_cost"] == 0
    assert set(target.plausible_info["curvature"]) == set(
        target.coordinate_names
    )
    assert np.isfinite(target.log_joint(target.x0))
    _assert_accounting(target)


def test_fully_explicit_and_mixed_setup_routes(monkeypatch):
    spec = vector_model(np.random.default_rng(803))
    start = {"beta": np.array([0.1, -0.2]), "sigma": 1.0}
    bounds = {
        "beta": (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        "sigma": (0.5, 2.0),
    }

    def no_search(*args, **kwargs):
        raise AssertionError("an explicit start must skip the mode search")

    monkeypatch.setattr(
        "pyvbmc.pymc._target._plausible.search_mode", no_search
    )
    fully_explicit = PyMCTarget(
        spec["model"],
        start=start,
        plausible_bounds=bounds,
        setup_budget=1,
        seed=4,
    )
    expected_start = fully_explicit.from_model_variables(start)
    np.testing.assert_allclose(fully_explicit.x0.ravel(), expected_start)
    np.testing.assert_allclose(
        fully_explicit.plb.ravel(), [-1.0, -1.0, np.log(0.5)]
    )
    np.testing.assert_allclose(
        fully_explicit.pub.ravel(), [1.0, 1.0, np.log(2.0)]
    )
    assert fully_explicit.plausible_info["route"] == "explicit"
    assert fully_explicit.plausible_info["start"] == "user"
    assert fully_explicit.plausible_info["mode"] is None
    assert fully_explicit.setup_cost == 1

    start_only = PyMCTarget(spec["model"], start=start, seed=4)
    assert start_only.plausible_info["route"] == "laplace"
    assert start_only.plausible_info["start"] == "user"
    assert start_only.plausible_info["mode"] is None
    assert start_only.setup_cost == start_only.D + 1
    assert start_only.plausible_info["location_outside_prior"] == []


def test_plausible_bounds_only_still_searches_without_hessian():
    spec = vector_model(np.random.default_rng(804))
    bounds = {
        "beta": (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        "sigma": (0.5, 2.0),
    }
    target = PyMCTarget(
        spec["model"], plausible_bounds=bounds, setup_budget=5, seed=5
    )
    assert target.plausible_info["route"] == "explicit"
    assert target.plausible_info["start"] == "mode"
    assert target.plausible_info["n_hessian_calls"] == 0
    assert target.plausible_info["hessian_cost"] == 0
    assert target.plausible_info["n_target_calls"] <= 5
    _assert_accounting(target)


def test_unsupported_model_inside_mode_search_keeps_its_type(monkeypatch):
    detail = (
        "compiled log-density gradient is float32; PyMC targets require "
        "float64 free variables, value variables, and density derivatives."
    )
    original = target_module._as_float64

    def raise_on_gradient(value, label):
        if label == "compiled log-density gradient":
            raise UnsupportedModel(detail)
        return original(value, label)

    monkeypatch.setattr(target_module, "_as_float64", raise_on_gradient)
    spec = scalar_model(np.random.default_rng(8052))
    with pytest.raises(UnsupportedModel) as raised:
        PyMCTarget(spec["model"], seed=62)
    assert str(raised.value) == detail


def test_small_cap_truncates_search_and_preserves_reserves():
    spec = scalar_model(np.random.default_rng(805))
    target = PyMCTarget(spec["model"], setup_budget=3, seed=6)
    assert target.plausible_info["cap_reached"]
    assert target.plausible_info["n_target_calls"] <= 2
    assert target.plausible_info["hessian_cost"] == 1
    assert target.setup_cost <= 3
    _assert_accounting(target)


def test_rejected_search_attempt_is_charged_but_not_reused(monkeypatch):
    def search_with_rejected_attempt(
        value_and_grad, x_start, lower, upper, max_calls
    ):
        value, _ = value_and_grad(x_start)
        return (
            np.array(x_start, copy=True),
            np.asarray(x_start, dtype=np.float64).reshape(1, -1),
            np.asarray([value], dtype=np.float64),
            2,
            True,
        )

    monkeypatch.setattr(
        "pyvbmc.pymc._target._plausible.search_mode",
        search_with_rejected_attempt,
    )
    spec = scalar_model(np.random.default_rng(8051))
    target = PyMCTarget(spec["model"], setup_budget=4, seed=61)
    assert target.plausible_info["n_target_calls"] == 2
    assert target.plausible_info["hessian_cost"] == target.D
    assert target.setup_cost == 2 + target.D
    assert target.plausible_info["cap_reached"]
    assert len(target.setup_evaluations[1]) == 1


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, np.bool_(False)])
def test_invalid_setup_budget(bad):
    spec = scalar_model(np.random.default_rng(806))
    with pytest.raises(ValueError, match="positive integer"):
        PyMCTarget(spec["model"], setup_budget=bad, seed=7)


def test_route_minimum_budgets():
    spec = scalar_model(np.random.default_rng(807))
    with pytest.raises(ValueError, match="at least 3"):
        PyMCTarget(spec["model"], setup_budget=2, seed=8)
    with pytest.raises(ValueError, match="at least 2"):
        PyMCTarget(spec["model"], start={"mu": 0.0}, setup_budget=1, seed=8)
    with pytest.raises(ValueError, match="at least 2"):
        PyMCTarget(
            spec["model"],
            plausible_bounds={"mu": (-1.0, 1.0)},
            setup_budget=1,
            seed=8,
        )


def test_explicit_bounds_and_start_validation_names_variable():
    spec = vector_model(np.random.default_rng(808))
    with pytest.raises(ValueError, match=r"missing model variable 'sigma'"):
        PyMCTarget(
            spec["model"],
            start={"beta": np.zeros(2)},
            seed=9,
        )
    with pytest.raises(ValueError, match=r"unknown model variable 'extra'"):
        PyMCTarget(
            spec["model"],
            start={"beta": np.zeros(2), "sigma": 1.0, "extra": 0.0},
            seed=9,
        )
    with pytest.raises(ValueError, match=r"sigma.*strictly inside"):
        PyMCTarget(
            spec["model"],
            start={"beta": np.zeros(2), "sigma": 0.0},
            seed=9,
        )


def test_near_boundary_explicit_values_match_core_normalization():
    with pm.Model() as model:
        pm.Uniform("p", np.float64(0.0), np.float64(1.0))
    target = PyMCTarget(
        model,
        start={"p": 0.0015},
        plausible_bounds={"p": (0.0012, 0.0018)},
        setup_budget=1,
        seed=10,
    )
    np.testing.assert_array_equal(target.x0, [[0.0015]])
    np.testing.assert_array_equal(target.plb, [[0.0012]])
    np.testing.assert_array_equal(target.pub, [[0.0018]])


def test_boundary_mode_is_moved_and_vbmc_accepts_bounds(caplog):
    with pm.Model() as model:
        pm.Beta("p", 1.0, 3.0)
    target = PyMCTarget(model, seed=11)
    assert target.plausible_info["start_moved"] == ["p"]
    assert np.isfinite(target.log_joint(target.x0))
    caplog.clear()
    VBMC(
        target.log_joint,
        target.x0,
        target.lb,
        target.ub,
        target.plb,
        target.pub,
        options={"display": "off"},
        seed=12,
    )
    assert not [
        record for record in caplog.records if record.name == "VBMC_init"
    ]


def test_prior_initial_strategy_is_seeded_without_global_rng_use():
    first_model = stochastic_initial_model()
    second_model = stochastic_initial_model()
    global_before = copy.deepcopy(np.random.get_state())
    first = PyMCTarget(
        first_model,
        plausible_bounds={"x": (-2.0, 2.0)},
        seed=13,
    )
    second = PyMCTarget(
        second_model,
        plausible_bounds={"x": (-2.0, 2.0)},
        seed=13,
    )
    global_after = np.random.get_state()
    assert first._initial_seed == second._initial_seed
    np.testing.assert_array_equal(first.x0, second.x0)
    np.testing.assert_array_equal(
        first.setup_evaluations[0], second.setup_evaluations[0]
    )
    assert all(
        np.array_equal(left, right)
        for left, right in zip(global_before, global_after)
    )
