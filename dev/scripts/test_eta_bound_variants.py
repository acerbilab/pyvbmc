"""Contract checks for the bounded Phase 6 eta objective variants."""

import copy
import importlib.util
import sys
from pathlib import Path

import gpyreg as gpr
import numpy as np
import pytest

from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc import variational_optimization as production

SCRIPTS = Path(__file__).resolve().parent


def _load_variants():
    path = SCRIPTS / "eta_bound_variants.py"
    name = "eta_bound_variants"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous
    return module


variants = _load_variants()


def _theta_and_bounds(D=2, K=2):
    vp = VariationalPosterior(D, K, rng=123)
    theta = np.concatenate(
        (
            np.array([-2.4, 2.3, 0.1, -0.2]),
            np.array([1.5, -0.1]),
            np.array([0.0, 0.1]),
            np.array([1.0, -7.0]),
        )
    )
    bounds = {
        "lb": np.concatenate(
            (np.full(D * K, -2.0), np.full(D * K, -2.0), np.full(K, -6.0))
        ),
        "ub": np.concatenate(
            (np.full(D * K, 2.0), np.full(D * K, 1.0), np.zeros(K))
        ),
        "tol_con": 0.1,
        "weight_threshold": 0.1,
        "weight_penalty": 0.25,
    }
    return vp, theta, bounds


def _gradient_size(vp, grad_flags):
    return (
        vp.D * vp.K * bool(grad_flags[0])
        + vp.K * bool(grad_flags[1])
        + vp.D * bool(grad_flags[2])
        + vp.K * bool(grad_flags[3])
    )


def _install_zero_objective(monkeypatch, module, *, mc):
    def gp_log_joint(
        vp,
        gp,
        grad_flags,
        avg_flag=True,
        jacobian_flag=True,
        compute_var=False,
        separate_K=False,
    ):
        gradient = np.zeros(_gradient_size(vp, grad_flags))
        if separate_K:
            return 0.0, gradient, 0.0, None, 0.0, np.zeros((1, vp.K)), None
        return 0.0, gradient, 0.0, None, 0.0

    def deterministic_entropy(vp, grad_flags, jacobian_flag):
        return 0.0, np.zeros(_gradient_size(vp, grad_flags))

    def mc_entropy(vp, Ns, grad_flags, jacobian_flag):
        # A fresh identically seeded VP per evaluation gives common draws.
        value = float(np.mean(vp.rng.standard_normal(Ns)))
        return value, np.zeros(_gradient_size(vp, grad_flags))

    monkeypatch.setattr(module, "_gp_log_joint", gp_log_joint)
    monkeypatch.setattr(module, "entlb_vbmc", deterministic_entropy)
    monkeypatch.setattr(module, "entmc_vbmc", mc_entropy)
    return 64 if mc else 0


def _central_difference(function, theta, step=1e-6):
    gradient = np.empty(theta.size)
    for index in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (function(plus) - function(minus)) / (2 * step)
    return gradient


@pytest.mark.parametrize("arm", variants.ARMS)
def test_variants_are_isolated_module_clones(arm):
    module = variants.get_variant(arm)
    assert module is variants.get_variant(variants.ARM_LABELS[arm])
    assert module is not production
    assert module._neg_elcbo.__globals__ is module.__dict__
    assert module._sieve.__globals__ is module.__dict__
    assert module.optimize_vp.__globals__ is module.__dict__
    assert module.ETA_BOUND_ARM == arm


@pytest.mark.parametrize("arm", ("A", "B"))
@pytest.mark.parametrize(
    "mc", (False, True), ids=("deterministic", "common_mc")
)
@pytest.mark.parametrize("eta_shift", (-2.0, 0.0, 2.0))
def test_isolated_penalty_gradient_and_no_mutation(
    monkeypatch, arm, mc, eta_shift
):
    module = variants.get_variant(arm)
    Ns = _install_zero_objective(monkeypatch, module, mc=mc)
    _, theta, bounds = _theta_and_bounds()
    theta[-2:] += eta_shift

    def evaluate(value, compute_grad):
        argument = value.copy()
        vp = VariationalPosterior(2, 2, rng=991)
        result = module._neg_elcbo(
            argument,
            object(),
            vp,
            0.0,
            Ns,
            compute_grad,
            False,
            bounds,
        )
        assert np.array_equal(argument, value)
        return result

    analytic = evaluate(theta, True)[1]
    numerical = _central_difference(
        lambda value: evaluate(value, False)[0], theta
    )
    assert np.allclose(analytic, numerical, rtol=2e-5, atol=2e-6)


def test_eta_bounds_and_common_shift_contracts():
    vp, theta, bounds = _theta_and_bounds()
    diagnostics = {
        arm: variants.penalty_terms(theta, bounds, vp, arm)
        for arm in variants.ARMS
    }

    assert diagnostics["A"]["eta_loss"] == 0.0
    assert not np.any(diagnostics["A"]["bound_gradient"][-vp.K :])
    assert diagnostics["B"]["eta_lower_violation"][1] > 0.0
    assert diagnostics["B"]["eta_upper_violation"][0] > 0.0
    assert diagnostics["C"]["eta_lower_violation"][1] > 0.0
    assert not np.any(diagnostics["C"]["eta_upper_violation"])

    shifted = theta.copy()
    shifted[-vp.K :] += 2.0
    shifted_diagnostics = {
        arm: variants.penalty_terms(shifted, bounds, vp, arm)
        for arm in variants.ARMS
    }
    assert (
        shifted_diagnostics["A"]["bound_loss"]
        == diagnostics["A"]["bound_loss"]
    )
    assert (
        shifted_diagnostics["C"]["bound_loss"]
        == diagnostics["C"]["bound_loss"]
    )
    assert (
        shifted_diagnostics["B"]["bound_loss"]
        != diagnostics["B"]["bound_loss"]
    )

    for arm in variants.ARMS:
        assert shifted_diagnostics[arm]["small_weight_loss"] == pytest.approx(
            diagnostics[arm]["small_weight_loss"], abs=0.0
        )
    assert len({d["small_weight_loss"] for d in diagnostics.values()}) == 1
    for left, right in zip(variants.ARMS[:-1], variants.ARMS[1:]):
        assert np.array_equal(
            diagnostics[left]["small_weight_gradient"],
            diagnostics[right]["small_weight_gradient"],
        )


def test_penalty_diagnostics_are_observational():
    vp, theta, bounds = _theta_and_bounds()
    theta_before = theta.copy()
    bounds_before = copy.deepcopy(bounds)
    vp_arrays_before = {
        key: value.copy()
        for key, value in vp.__dict__.items()
        if isinstance(value, np.ndarray)
    }
    rng_before = copy.deepcopy(vp.rng.bit_generator.state)

    for arm in variants.ARMS:
        variants.penalty_terms(theta, bounds, vp, arm)

    assert np.array_equal(theta, theta_before)
    assert bounds.keys() == bounds_before.keys()
    for key, value in bounds.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, bounds_before[key])
        else:
            assert value == bounds_before[key]
    assert vp_arrays_before.keys() <= vp.__dict__.keys()
    for key, value in vp_arrays_before.items():
        assert np.array_equal(vp.__dict__[key], value)
    assert vp.rng.bit_generator.state == rng_before


@pytest.mark.parametrize("arm", variants.ARMS)
def test_penalty_diagnostics_match_arm_objective(monkeypatch, arm):
    module = variants.get_variant(arm)
    _install_zero_objective(monkeypatch, module, mc=False)
    vp, theta, bounds = _theta_and_bounds()
    diagnostic = variants.penalty_terms(theta, bounds, vp, arm)

    with_bounds = module._neg_elcbo(
        theta.copy(),
        object(),
        VariationalPosterior(2, 2, rng=44),
        0.0,
        0,
        True,
        False,
        bounds,
    )
    without_bounds = module._neg_elcbo(
        theta.copy(),
        object(),
        VariationalPosterior(2, 2, rng=44),
        0.0,
        0,
        True,
        False,
        None,
    )
    expected_loss = diagnostic["bound_loss"] + diagnostic["small_weight_loss"]
    expected_gradient = (
        diagnostic["bound_gradient"] + diagnostic["small_weight_gradient"]
    )
    assert with_bounds[0] - without_bounds[0] == pytest.approx(expected_loss)
    assert np.allclose(with_bounds[1] - without_bounds[1], expected_gradient)


def test_c_diagnostic_corrects_relative_max_chain_rule():
    vp, theta, bounds = _theta_and_bounds()
    diagnostic = variants.penalty_terms(theta, bounds, vp, "C")
    assert diagnostic["correct_gradient_defined"]
    assert diagnostic["eta_max_tie_count"] == 1
    assert np.sum(diagnostic["bound_gradient"][-vp.K :]) != 0.0
    assert np.sum(
        diagnostic["correct_bound_gradient"][-vp.K :]
    ) == pytest.approx(0.0, abs=1e-12)

    numerical = _central_difference(
        lambda value: variants.penalty_terms(value, bounds, vp, "C")[
            "bound_loss"
        ],
        theta,
    )
    assert np.allclose(
        diagnostic["correct_bound_gradient"], numerical, rtol=2e-5, atol=2e-6
    )
    assert not np.allclose(diagnostic["bound_gradient"], numerical)


def _fixture_gp():
    data = Path(production.__file__).resolve().parents[1] / "testing" / "vbmc"
    load = lambda name: np.loadtxt(data / name, delimiter=",")
    gp = gpr.GP(
        D=2,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp.update(
        X_new=load("X.txt"),
        y_new=load("y.txt").reshape(-1, 1),
        hyp=load("hyp.txt")[0:2],
    )
    return gp


@pytest.mark.parametrize("arm", ("A", "B"))
@pytest.mark.parametrize(
    "Ns, rtol, atol",
    ((0, 3e-5, 3e-6), (128, 8e-2, 8e-2)),
    ids=("deterministic", "common_mc"),
)
@pytest.mark.parametrize("eta_shift", (-2.0, 0.0, 2.0))
def test_full_objective_gradient_with_active_eta_bounds_and_shifts(
    arm, Ns, rtol, atol, eta_shift
):
    module = variants.get_variant(arm)
    gp = _fixture_gp()
    _, theta, bounds = _theta_and_bounds()
    theta[-2:] += eta_shift

    def evaluate(value, compute_grad):
        argument = value.copy()
        vp = VariationalPosterior(2, 2, rng=1776)
        result = module._neg_elcbo(
            argument,
            gp,
            vp,
            0.0,
            Ns,
            compute_grad,
            False,
            bounds,
        )
        assert np.array_equal(argument, value)
        return result

    analytic = evaluate(theta, True)[1]
    numerical = _central_difference(
        lambda value: evaluate(value, False)[0], theta
    )
    assert np.allclose(analytic, numerical, rtol=rtol, atol=atol)


def _assert_results_exact(left, right):
    assert len(left) == len(right)
    for left_item, right_item in zip(left, right):
        if left_item is None or right_item is None:
            assert left_item is right_item
        else:
            assert np.array_equal(left_item, right_item)


def test_c_exactly_preserves_production_result_rng_and_mutation():
    gp = _fixture_gp()
    vp, theta, bounds = _theta_and_bounds()
    theta_production = theta.copy()
    theta_variant = theta.copy()
    vp_production = VariationalPosterior(2, 2, rng=8128)
    vp_variant = VariationalPosterior(2, 2, rng=8128)

    production_result = production._neg_elcbo(
        theta_production, gp, vp_production, 0.0, 128, True, False, bounds
    )
    variant_result = variants.get_variant("C")._neg_elcbo(
        theta_variant, gp, vp_variant, 0.0, 128, True, False, bounds
    )

    _assert_results_exact(production_result, variant_result)
    assert np.array_equal(theta_production, theta_variant)
    assert not np.array_equal(theta_production, theta)
    assert theta_variant[-vp.K :].max() == 0.0
    assert (
        vp_production.rng.bit_generator.state
        == vp_variant.rng.bit_generator.state
    )


def test_k1_and_weights_disabled_are_supported():
    vp = VariationalPosterior(1, 1, rng=7)
    theta = np.array([0.0, 0.0, 0.0, 3.0])
    bounds = {
        "lb": np.array([-1.0, -1.0, -5.0]),
        "ub": np.array([1.0, 1.0, 0.0]),
        "tol_con": 0.1,
        "weight_threshold": 0.25,
        "weight_penalty": 0.5,
    }
    for arm in variants.ARMS:
        diagnostic = variants.penalty_terms(theta, bounds, vp, arm)
        assert diagnostic["weights"] == pytest.approx(np.array([1.0]))
        assert diagnostic["small_weight_gradient"][-1] == 0.0

    vp.optimize_weights = False
    no_weight_theta = theta[:-1]
    no_weight_bounds = {
        "lb": np.array([-1.0, -1.0]),
        "ub": np.array([1.0, 1.0]),
        "tol_con": 0.1,
    }
    for arm in variants.ARMS:
        diagnostic = variants.penalty_terms(
            no_weight_theta, no_weight_bounds, vp, arm
        )
        assert diagnostic["raw_eta"].size == 0
        assert diagnostic["eta_violation_count"] == 0
        assert diagnostic["small_weight_loss"] == 0.0
