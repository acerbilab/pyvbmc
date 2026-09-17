"""Focused checks for fixed-node noisy-acquisition quadrature."""

import copy
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from dev.scripts.noisy_acq_quadrature import (
    _component_counts,
    _log_weighted_residual,
    _normal_from_unit,
    _stable_log_reduction,
    baseline_gain_resolution,
    judge_paired,
    make_rule,
    practical_band,
    prepare_rule,
    raw_loss_assessment,
)
from pyvbmc.acquisition_functions import AcqFcnVIQR
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.testing.acquisition_functions.test_acq_fcn_viqr_losses import (
    _brute_force,
    _setup,
)
from pyvbmc.testing.acquisition_functions.test_viqr_kernel_reuse import (
    _as_non_cholesky,
)


def _fake_vp(weights=(0.6, 0.3, 0.1)):
    weights = np.asarray(weights, dtype=np.float64)
    components = weights.size
    means = np.vstack(
        [
            np.linspace(-1.0, 1.0, components),
            np.linspace(0.5, -0.5, components),
        ]
    )
    return SimpleNamespace(
        w=weights[None, :],
        mu=means,
        sigma=np.linspace(0.4, 0.8, components)[None, :],
        lambd=np.array([[1.0], [1.5]]),
        rng=np.random.default_rng(991),
    )


def _state(gp, vp, optim_state):
    return {
        "pt": vp.parameter_transformer,
        "vp": vp,
        "gp": gp,
        "logger": None,
        "optim_state": optim_state,
        "options": None,
        "candidates": None,
        "ref": None,
        "meta": {},
    }


def _active_rule(optim_state, weights=None):
    nodes = optim_state["active_importance_sampling"]["X"]
    if weights is None:
        weights = np.ones(nodes.shape[0], dtype=np.float64)
    return {
        "method": "captured_mc",
        "budget": nodes.shape[0],
        "seed": None,
        "available": True,
        "nodes": nodes.copy(),
        "weights": np.asarray(weights, dtype=np.float64),
        "metadata": {},
    }


def test_component_allocation_uses_power_of_two_blocks_and_fixed_ties():
    weights = np.array([0.6, 0.3, 0.1])
    np.testing.assert_array_equal(_component_counts(weights, 8), [4, 2, 2])
    np.testing.assert_array_equal(_component_counts(weights, 9), [4, 4, 1])
    np.testing.assert_array_equal(_component_counts(weights, 10), [4, 4, 2])
    assert _component_counts(weights, 2) is None

    tied = _component_counts(np.array([0.5, 0.5]), 3)
    np.testing.assert_array_equal(tied, [2, 1])


@pytest.mark.parametrize("method", ["stratified_mc", "stratified_rqmc"])
def test_stratified_rules_share_allocation_and_explicit_weights(method):
    vp = _fake_vp()
    rule = make_rule(vp, method, budget=9, seed=12)
    assert rule["available"]
    assert rule["nodes"].dtype == np.float64
    assert rule["weights"].dtype == np.float64
    assert rule["metadata"]["component_counts"] == [4, 4, 1]
    assert rule["metadata"]["actual_nodes"] == 9
    assert rule["metadata"]["unused_budget"] == 0
    assignment = np.asarray(rule["metadata"]["component_assignment"])
    for component, count in enumerate([4, 4, 1]):
        selected = rule["weights"][assignment == component]
        np.testing.assert_allclose(
            selected,
            np.full(count, vp.w[0, component] / np.sum(vp.w) / count),
            rtol=4 * np.finfo(np.float64).eps,
            atol=0,
        )
    assert rule["metadata"]["scramble"] == (method == "stratified_rqmc")
    assert rule["metadata"]["seed_sequence"]["entropy"] == 12
    assert len(rule["metadata"]["component_seeds"]) == 3


def test_rule_unavailable_below_positive_component_count():
    rule = make_rule(_fake_vp(), "stratified_rqmc", budget=2, seed=3)
    assert not rule["available"]
    assert rule["nodes"].shape == (0, 2)
    assert rule["metadata"]["reason"] == (
        "budget_below_positive_component_count"
    )
    assert rule["metadata"]["actual_nodes"] == 0


def test_zero_weight_components_are_not_allocated():
    vp = _fake_vp((0.75, 0.0, 0.25))
    rule = make_rule(vp, "stratified_mc", budget=4, seed=2)
    assert rule["metadata"]["component_counts"][1] == 0
    assert 1 not in rule["metadata"]["component_assignment"]
    assert np.sum(rule["weights"]) == pytest.approx(1.0)


def test_tiny_weight_narrow_component_is_retained_with_its_full_mass():
    vp = _fake_vp((1.0 - 1e-12, 1e-12))
    vp.mu[:, 1] = np.array([1e4, -1e4])
    vp.sigma[0, 1] = 1e-12
    rule = make_rule(vp, "stratified_rqmc", budget=3, seed=81)
    assignment = np.asarray(rule["metadata"]["component_assignment"])
    narrow = assignment == 1
    assert rule["metadata"]["component_counts"] == [2, 1]
    assert np.count_nonzero(narrow) == 1
    assert np.sum(rule["weights"][narrow]) == pytest.approx(
        1e-12, rel=4 * np.finfo(np.float64).eps
    )
    assert np.all(np.isfinite(rule["nodes"][narrow]))
    np.testing.assert_allclose(
        rule["nodes"][narrow][0], vp.mu[:, 1], rtol=0, atol=5e-11
    )


def test_inverse_normal_machine_endpoints_stay_finite():
    transformed = _normal_from_unit(np.array([[0.0, 0.5, 1.0]]))
    assert np.all(np.isfinite(transformed))
    assert transformed[0, 0] < 0 < transformed[0, 2]
    assert transformed[0, 1] == 0


@pytest.mark.parametrize("method", ["mc", "stratified_mc", "stratified_rqmc"])
def test_rule_reproducibility_and_vp_rng_isolation(method):
    vp = _fake_vp()
    rng_before = copy.deepcopy(vp.rng.bit_generator.state)
    first = make_rule(vp, method, budget=128, seed=72)
    # Constructing another arm cannot affect this arm's private stream.
    make_rule(vp, "stratified_rqmc", budget=64, seed=99)
    second = make_rule(vp, method, budget=128, seed=72)
    np.testing.assert_array_equal(first["nodes"], second["nodes"])
    np.testing.assert_array_equal(first["weights"], second["weights"])
    assert first["metadata"] == second["metadata"]
    assert vp.rng.bit_generator.state == rng_before


def test_rqmc_weighted_moments_match_gaussian_mixture():
    vp = _fake_vp((0.7, 0.3))
    rule = make_rule(vp, "stratified_rqmc", budget=2048, seed=313)
    nodes, weights = rule["nodes"], rule["weights"]
    mean = nodes.T @ weights
    expected_mean = vp.mu @ vp.w.reshape(-1)
    np.testing.assert_allclose(mean, expected_mean, atol=8e-3, rtol=0)

    centered = nodes - mean
    covariance = (centered.T * weights) @ centered
    component_covariance = np.zeros((2, 2))
    for component, component_weight in enumerate(vp.w.reshape(-1)):
        scale = vp.sigma[0, component] * vp.lambd.reshape(-1)
        offset = vp.mu[:, component] - expected_mean
        component_covariance += component_weight * (
            np.diag(scale**2) + np.outer(offset, offset)
        )
    np.testing.assert_allclose(
        covariance, component_covariance, atol=2e-2, rtol=0
    )


@pytest.mark.parametrize("ns_gp", [1, 2])
@pytest.mark.parametrize("factor", ["cholesky", "inverse"])
def test_equal_weight_full_score_matches_public_viqr(ns_gp, factor):
    gp, vp, optim_state, candidates = _setup(ns_gp, Na=32)
    if factor == "inverse":
        _as_non_cholesky(gp, optim_state)
    rule = _active_rule(optim_state)
    evaluator = prepare_rule(_state(gp, vp, optim_state), rule)
    candidate_copy = candidates.copy()
    actual = evaluator.score(candidates)
    public = AcqFcnVIQR()(candidate_copy, gp, vp, None, optim_state)
    np.testing.assert_allclose(
        actual["full_score"],
        public - np.log(rule["nodes"].shape[0]),
        rtol=2e-12,
        atol=2e-13,
    )
    np.testing.assert_array_equal(candidates, candidate_copy)
    assert actual["full_score"].shape == (candidates.shape[0],)
    assert np.all(actual["valid"])
    assert evaluator.metadata["factor_representations"] == [factor] * ns_gp


def test_full_score_preserves_integer_penalty_and_bound_wrappers():
    gp, vp, optim_state, _ = _setup(2, Na=32)
    optim_state["integer_vars"] = np.array([True, False])
    optim_state["variance_regularized_acq_fcn"] = True
    optim_state["tol_gp_var"] = 100.0
    optim_state["lb_eps_orig"] = np.array([-1.0, -1.0])
    optim_state["ub_eps_orig"] = np.array([1.0, 1.0])
    candidates = np.array([[0.49, 0.2], [2.0, 0.0]])
    original = candidates.copy()
    rule = _active_rule(optim_state)
    evaluator = prepare_rule(_state(gp, vp, optim_state), rule)
    actual = evaluator.score(candidates)
    public = AcqFcnVIQR()(candidates.copy(), gp, vp, None, optim_state)
    np.testing.assert_allclose(
        actual["full_score"][:1],
        public[:1] - np.log(rule["nodes"].shape[0]),
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.isinf(actual["full_score"][1])
    assert actual["failure_reason"][1] == "outside_hard_bounds"
    assert actual["penalty"][0] > 0
    assert actual["X_used"][0, 0] == 0
    np.testing.assert_array_equal(candidates, original)


def test_full_score_parity_with_bounded_rotoscale_transformer():
    gp, vp, optim_state, _ = _setup(2, Na=32)
    lower = np.full((1, 2), -2.0)
    upper = np.full((1, 2), 3.0)
    transformer = ParameterTransformer(
        2, lower, upper, transform_type="probit"
    )
    transformer.mu = np.array([-0.2, 0.3])
    transformer.delta = np.array([0.7, 1.2])
    transformer.scale = np.array([0.8, 1.4])
    transformer.R_mat = np.array([[0.0, -1.0], [1.0, 0.0]])
    vp.parameter_transformer = transformer
    optim_state["lb_eps_orig"] = np.array([-1.5, -1.5])
    optim_state["ub_eps_orig"] = np.array([2.5, 2.5])
    candidates = transformer(np.array([[0.0, 0.0], [-1.99, 0.0]]))
    rule = _active_rule(optim_state)
    evaluator = prepare_rule(_state(gp, vp, optim_state), rule)
    actual = evaluator.score(candidates)
    public = AcqFcnVIQR()(candidates.copy(), gp, vp, None, optim_state)
    np.testing.assert_allclose(
        actual["full_score"][:1],
        public[:1] - np.log(rule["nodes"].shape[0]),
        rtol=2e-12,
        atol=2e-13,
    )
    assert np.isinf(public[1])
    assert np.isinf(actual["full_score"][1])
    assert actual["failure_reason"][1] == "outside_hard_bounds"
    np.testing.assert_allclose(
        transformer.inverse(actual["X_used"]),
        np.array([[0.0, 0.0], [-1.99, 0.0]]),
        rtol=0,
        atol=2e-14,
    )


def test_explicit_weight_residual_and_reduction_match_direct_sums():
    gp, vp, optim_state, candidates = _setup(2, Na=16)
    weights = np.arange(1, 17, dtype=np.float64)
    weights /= np.sum(weights)
    evaluator = prepare_rule(
        _state(gp, vp, optim_state), _active_rule(optim_state, weights)
    )
    result = evaluator.score(candidates, full=False)
    current_variance, tau2 = _brute_force(
        AcqFcnVIQR(), gp, optim_state, candidates
    )
    u = AcqFcnVIQR().u
    posterior_sd = np.sqrt(
        np.maximum(current_variance[None, :, :] - tau2, 0.0)
    )
    residual_by_hyper = np.sum(
        weights[None, :, None] * 2 * np.sinh(u * posterior_sd), axis=1
    )
    expected_residual = np.mean(residual_by_hyper, axis=1)
    current_sd = np.sqrt(current_variance)[None, :, :]
    reduction_by_hyper = np.sum(
        weights[None, :, None]
        * 2
        * (np.sinh(u * current_sd) - np.sinh(u * posterior_sd)),
        axis=1,
    )
    expected_reduction = np.mean(reduction_by_hyper, axis=1)
    np.testing.assert_allclose(
        result["residual"], expected_residual, rtol=2e-9, atol=0
    )
    np.testing.assert_allclose(
        result["reduction"], expected_reduction, rtol=2e-8, atol=1e-15
    )


def test_weighted_residual_uses_direct_and_overflow_safe_paths():
    weights = np.array([0.25, 0.75])
    log_weights = np.log(weights)
    scaled_sd = np.array([[0.1, 0.3], [710.0, 709.0]])
    actual = _log_weighted_residual(scaled_sd, weights, log_weights)
    expected_direct = np.log(2 * np.sum(weights * np.sinh(scaled_sd[0])))
    assert actual[0] == pytest.approx(expected_direct, rel=3e-16)
    expected_large = np.logaddexp(
        scaled_sd[1, 0]
        + np.log1p(-np.exp(-2 * scaled_sd[1, 0]))
        + log_weights[0],
        scaled_sd[1, 1]
        + np.log1p(-np.exp(-2 * scaled_sd[1, 1]))
        + log_weights[1],
    )
    assert np.isfinite(actual[1])
    assert actual[1] == pytest.approx(expected_large, rel=3e-16)


def test_stable_reduction_matches_high_precision_cancellation_case():
    acquisition = AcqFcnVIQR()
    tau2 = np.array([[1e-20]])
    log_reduction = acquisition._log_iqr_reduction(
        tau2, np.array([1.0]), np.array([0.0])
    )[0] + np.log(2.0)
    assert (
        2
        * (
            np.sinh(acquisition.u)
            - np.sinh(acquisition.u * np.sqrt(1.0 - tau2[0, 0]))
        )
        == 0
    )
    with localcontext() as context:
        context.prec = 80
        one = Decimal(1)
        u = Decimal.from_float(acquisition.u)
        tau = Decimal.from_float(tau2[0, 0])
        current_argument = u
        posterior_argument = u * (one - tau).sqrt()

        def sinh(value):
            return (value.exp() - (-value).exp()) / Decimal(2)

        expected = Decimal(2) * (
            sinh(current_argument) - sinh(posterior_argument)
        )
    assert log_reduction == pytest.approx(
        np.log(float(expected)), rel=2e-15, abs=0
    )


def test_stable_reduction_respects_source_variance_clipping():
    acquisition = AcqFcnVIQR()
    source_variance = np.array([1.0])
    actual = _stable_log_reduction(
        acquisition,
        np.array([[2.0]]),
        source_variance,
        np.array([0.0]),
    )[0]
    expected = np.log(2 * np.sinh(acquisition.u))
    assert actual == pytest.approx(expected, rel=2e-15, abs=0)


@pytest.mark.parametrize("factor", ["cholesky", "inverse"])
def test_fixed_cache_matches_captured_mc_cache_and_is_deterministic(factor):
    gp, vp, optim_state, candidates = _setup(2, Na=32)
    if factor == "inverse":
        _as_non_cholesky(gp, optim_state)
    active_before = {
        key: value.copy()
        for key, value in optim_state["active_importance_sampling"].items()
        if isinstance(value, np.ndarray)
    }
    rng_before = copy.deepcopy(vp.rng.bit_generator.state)
    state = _state(gp, vp, optim_state)
    rule = _active_rule(optim_state)
    first = prepare_rule(state, rule)
    second = prepare_rule(state, rule)
    np.testing.assert_allclose(
        first.cache.f_s2,
        optim_state["active_importance_sampling"]["f_s2"],
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        first.cache.K_Xa_X,
        optim_state["active_importance_sampling"]["K_Xa_X"],
        rtol=1e-14,
        atol=0,
    )
    np.testing.assert_allclose(
        first.cache.C_tmp,
        optim_state["active_importance_sampling"]["C_tmp"],
        rtol=2e-13,
        atol=0,
    )
    np.testing.assert_array_equal(first.cache.f_s2, second.cache.f_s2)
    np.testing.assert_array_equal(first.cache.K_Xa_X, second.cache.K_Xa_X)
    np.testing.assert_array_equal(first.cache.C_tmp, second.cache.C_tmp)
    first.score(candidates)
    assert vp.rng.bit_generator.state == rng_before
    for key, expected in active_before.items():
        np.testing.assert_array_equal(
            optim_state["active_importance_sampling"][key], expected
        )


def test_streamed_cache_matches_dense_cache_with_small_node_blocks():
    gp, vp, optim_state, candidates = _setup(2, Na=32)
    state = _state(gp, vp, optim_state)
    dense_rule = _active_rule(optim_state)
    streamed_rule = _active_rule(optim_state)
    streamed_rule["metadata"] = {
        "cache_max_bytes": 1,
        "node_block_size": 7,
        "candidate_chunk_size": 3,
    }
    dense = prepare_rule(state, dense_rule)
    streamed = prepare_rule(state, streamed_rule)
    dense_result = dense.score(candidates)
    streamed_result = streamed.score(candidates)
    assert streamed.cache.K_Xa_X is None
    assert streamed.cache.C_tmp is None
    assert streamed.metadata["streamed_node_cache"]
    assert streamed.metadata["node_block_size"] == 7
    for key in ("full_score", "log_residual", "log_reduction", "reduction"):
        np.testing.assert_allclose(
            streamed_result[key],
            dense_result[key],
            rtol=5e-13,
            atol=2e-14,
        )


def test_zero_reduction_is_reported_without_subtracting_residuals():
    gp, vp, optim_state, _ = _setup(1, Na=16)
    evaluator = prepare_rule(
        _state(gp, vp, optim_state), _active_rule(optim_state)
    )
    result = evaluator.score(np.array([[1e6, -1e6]]), full=False)
    assert result["log_reduction"][0] == -np.inf
    assert result["reduction"][0] == 0
    assert np.isfinite(result["log_residual"][0])
    assert result["valid"][0]


def test_score_can_skip_reduction_diagnostics(monkeypatch):
    gp, vp, optim_state, candidates = _setup(2, Na=16)
    evaluator = prepare_rule(
        _state(gp, vp, optim_state), _active_rule(optim_state)
    )
    expected = evaluator.score(candidates)

    def reject_reduction(*args, **kwargs):
        raise AssertionError("reduction diagnostic was evaluated")

    monkeypatch.setattr(
        evaluator._acquisition, "_log_iqr_reduction", reject_reduction
    )
    timed = evaluator.score(candidates, diagnostics=False)
    np.testing.assert_array_equal(timed["full_score"], expected["full_score"])
    np.testing.assert_array_equal(
        timed["log_residual"], expected["log_residual"]
    )
    assert timed["log_reduction"] is None
    assert timed["reduction"] is None
    assert not timed["metadata"]["diagnostics"]


def test_invalid_inputs_have_explicit_failures():
    gp, vp, optim_state, candidates = _setup(1, Na=8)
    rule = _active_rule(optim_state)
    bad_rule = dict(rule)
    bad_rule["weights"] = rule["weights"].copy()
    bad_rule["weights"][0] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        prepare_rule(_state(gp, vp, optim_state), bad_rule)

    unavailable = make_rule(_fake_vp(), "stratified_mc", budget=2, seed=8)
    with pytest.raises(ValueError, match="rule is unavailable"):
        prepare_rule(_state(gp, vp, optim_state), unavailable)

    evaluator = prepare_rule(_state(gp, vp, optim_state), rule)
    result = evaluator.score(np.vstack([candidates[0], [np.nan, 0.0]]))
    assert result["valid"].tolist() == [True, False]
    assert result["failure_reason"][1] == "nonfinite_candidate"
    assert np.isinf(result["full_score"][1])

    gp.temporary_data["sn2_new"] = np.full(gp.X.shape[0], -1e6)
    invalid_numeric = evaluator.score(candidates[:1])
    assert not invalid_numeric["valid"][0]
    assert invalid_numeric["failure_reason"][0] == (
        "invalid_weighted_residual"
    )
    assert np.isnan(invalid_numeric["log_residual"][0])


def test_practical_band_uses_floor_when_baseline_gain_is_unresolved():
    resolved = practical_band(3.0, 2.0)
    assert resolved["eps_F"] == pytest.approx(0.01)
    unresolved = practical_band(3.0, 2.0, baseline_gain_resolved=False)
    assert unresolved["eps_F"] == pytest.approx(np.log1p(1e-6))
    assert not unresolved["baseline_gain_resolved"]


def test_paired_judge_requires_consecutive_budgets_and_records_inputs():
    reference = np.zeros(8)
    arm = np.array([-0.11, -0.10, -0.09, -0.105] * 2)
    first = judge_paired(
        arm,
        reference,
        0.01,
        replicate_seeds=list(range(8)),
        pooled_difference=-0.1,
    )
    assert first["interval_classification"] == "beneficial"
    assert first["budget_classification"] == "beneficial"
    assert first["classification"] == "unresolved"
    assert first["reason"] == "needs_consecutive_budget"
    assert first["replicate_seeds"] == list(range(8))
    assert first["t_critical"] == pytest.approx(2.364624251)

    second = judge_paired(
        arm * 0.99,
        reference,
        0.01,
        previous=first,
        pooled_difference=-0.1,
    )
    assert second["classification"] == "beneficial"
    assert second["reason"] is None


def test_paired_judge_handles_ties_precision_and_pooled_disagreement():
    reference = np.zeros(8)
    tie = judge_paired(
        np.array([-1e-4, 1e-4] * 4),
        reference,
        0.01,
        previous="practical_tie",
    )
    assert tie["classification"] == "practical_tie"

    noisy_direction = judge_paired(
        np.array([-0.18, -0.02] * 4),
        reference,
        0.01,
        previous="beneficial",
    )
    assert noisy_direction["interval_classification"] == "beneficial"
    assert noisy_direction["classification"] == "unresolved"
    assert noisy_direction["reason"] == "directional_precision"

    disagreement = judge_paired(
        np.full(8, -0.1),
        reference,
        0.01,
        previous="beneficial",
        pooled_difference=0.1,
    )
    assert disagreement["pooled_classification"] == "harmful"
    assert disagreement["budget_classification"] == "unresolved"
    assert disagreement["classification"] == "unresolved"
    assert disagreement["reason"] == "pooled_mean_log_disagreement"

    after_disagreement = judge_paired(
        np.full(8, -0.1),
        reference,
        0.01,
        previous=disagreement,
        pooled_difference=-0.1,
    )
    assert after_disagreement["budget_classification"] == "beneficial"
    assert after_disagreement["classification"] == "unresolved"
    assert after_disagreement["reason"] == "budget_stability"


def test_paired_judge_rejects_nonfinite_or_unpaired_samples():
    with pytest.raises(ValueError, match="same shape"):
        judge_paired(np.zeros(3), np.zeros(2), 0.1)
    unresolved = judge_paired(np.array([np.nan, 0.0]), np.zeros(2), 0.1)
    assert unresolved["classification"] == "unresolved"
    assert unresolved["reason"] == "insufficient_or_nonfinite_pairs"


def test_baseline_gain_resolution_uses_relative_floor_and_eight_pairs():
    references = np.zeros(8)
    resolved = baseline_gain_resolution(np.full(8, np.log(2e-12)), references)
    assert resolved["resolved"]
    assert resolved["classification"] == "resolved_positive"
    assert resolved["ci"][0] > resolved["relative_gain_floor"]

    at_floor = baseline_gain_resolution(np.full(8, np.log(5e-13)), references)
    assert not at_floor["resolved"]
    assert at_floor["reason"] == "gain_at_numerical_floor"
    wrong_count = baseline_gain_resolution(
        np.full(7, np.log(1e-3)), np.zeros(7)
    )
    assert wrong_count["reason"] == "requires_eight_replicates"


def test_raw_loss_assessment_is_scaled_stable_and_budget_stable():
    references = np.full(8, 710.0)
    baseline = np.full(8, 700.0)
    arm = baseline + np.log(0.8)
    first = raw_loss_assessment(arm, baseline, references)
    assert first["baseline_gain"]["resolved"]
    assert first["interval_classification"] == "material_loss"
    assert first["budget_classification"] == "material_loss"
    assert first["classification"] == "unresolved"
    assert first["reason"] == "needs_consecutive_budget"
    assert first["arm_to_baseline_ratio"] == pytest.approx(0.8)
    assert np.all(np.isfinite(first["scaled_paired_differences"]))

    second = raw_loss_assessment(arm, baseline, references, previous=first)
    assert second["classification"] == "material_loss"
    assert second["ci"][1] < 0

    safe = raw_loss_assessment(
        baseline,
        baseline,
        references,
        previous="no_material_loss",
    )
    assert safe["classification"] == "no_material_loss"
    assert safe["ci"][0] >= 0

    blocked_previous = dict(first)
    blocked_previous["budget_classification"] = "unresolved"
    blocked = raw_loss_assessment(
        arm, baseline, references, previous=blocked_previous
    )
    assert blocked["interval_classification"] == "material_loss"
    assert blocked["budget_classification"] == "material_loss"
    assert blocked["classification"] == "unresolved"
    assert blocked["reason"] == "budget_stability"


def test_raw_loss_assessment_omits_ratio_when_baseline_is_unresolved():
    references = np.zeros(8)
    unresolved = raw_loss_assessment(
        np.full(8, np.log(1e-13)),
        np.full(8, np.log(1e-13)),
        references,
    )
    assert unresolved["classification"] == "unresolved"
    assert unresolved["reason"] == "baseline_gain_unresolved"
    assert unresolved["arm_to_baseline_ratio"] is None
    assert unresolved["scaled_paired_differences"] is None
