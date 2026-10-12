"""Static-sized checks for the bounded S-VBMC NumPy prototype."""

from types import SimpleNamespace

import numpy as np
import pytest

from dev.scripts import svbmc_numpy_core as core


class _LowerBoundedTransform:
    """Positive-domain transform with a nonconstant Jacobian."""

    def __init__(self, shift=0.0):
        self.shift = np.float64(shift)
        self.calls = {"forward": 0, "inverse": 0, "jacobian": 0}

    def __call__(self, x):
        self.calls["forward"] += 1
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        return np.log(x) + self.shift

    def inverse(self, u):
        self.calls["inverse"] += 1
        u = np.asarray(u, dtype=np.float64).reshape(-1, 1)
        return np.exp(u - self.shift)

    def log_abs_det_jacobian(self, u):
        self.calls["jacobian"] += 1
        u = np.asarray(u, dtype=np.float64).reshape(-1, 1)
        return (u[:, 0] - self.shift).astype(np.float64)


def _vp(k, offset, elbo, weights, shift):
    return SimpleNamespace(
        mu=(np.arange(k, dtype=np.float64)[None, :] * 0.3 + offset),
        sigma=np.linspace(0.7, 1.1, k, dtype=np.float64)[None, :],
        lambd=np.ones((1, 1), dtype=np.float64),
        w=np.asarray(weights, dtype=np.float64).reshape(1, k),
        parameter_transformer=_LowerBoundedTransform(shift),
        stats={"elbo": np.float64(elbo)},
    )


def _stacker(testing=False):
    vp1 = _vp(1, -0.3, -1.0, [1.0], 0.2)
    vp2 = _vp(3, 0.4, 0.7, [0.2, 0.3, 0.5], -0.15)
    base = np.concatenate((vp1.w.reshape(-1), vp2.w.reshape(-1)))
    base /= base.sum()
    return SimpleNamespace(
        vp_list=[vp1, vp2],
        D=1,
        K=[1, 3],
        M=2,
        w=base.reshape(1, -1),
        I=np.array([[0.4, -0.7, 1.1, 0.2]], dtype=np.float64),
        individual_elbos=[vp1.stats["elbo"], vp2.stats["elbo"]],
        I_corrected=[],
        E_corrected=np.zeros(2, dtype=np.float64),
        testing=testing,
        _svbmc_random_seed=17,
    )


def _fixed_problem(counts=(1, 3, 2), n_samples=4):
    counts = np.asarray(counts, dtype=np.int64)
    k_total = int(counts.sum())
    sample_count = k_total * n_samples
    row = np.linspace(-1.3, 1.7, sample_count)[:, None]
    column = np.linspace(-0.8, 0.9, k_total)[None, :]
    logq = -0.5 * np.square(row - column) + 0.07 * row * column
    integrals = np.linspace(-0.4, 1.2, k_total, dtype=np.float64)
    base = np.linspace(1.0, 2.5, k_total, dtype=np.float64)
    base /= base.sum()
    return counts, n_samples, logq, integrals, base


def _central_difference(function, x, step=1e-6):
    x = np.asarray(x, dtype=np.float64)
    gradient = np.empty_like(x)
    for j in range(x.size):
        delta = np.zeros_like(x)
        delta[j] = step
        gradient[j] = (function(x + delta) - function(x - delta)) / (
            2.0 * step
        )
    return gradient


def test_optimized_preparation_matches_reference_for_bounded_d1_single_draw():
    stacker = _stacker()
    np.random.seed(1234)
    expected_logq, expected_j = core.prepare_entropy_reference(stacker, 1)
    np.random.seed(1234)
    actual_logq, actual_j = core.prepare_entropy(stacker, 1, chunk_rows=2)
    np.testing.assert_allclose(actual_logq, expected_logq, rtol=0, atol=1e-15)
    np.testing.assert_allclose(actual_j, expected_j, rtol=0, atol=1e-15)
    assert actual_logq.dtype == np.float64
    assert actual_j.dtype == np.float64


def test_optimized_preparation_reuses_each_posterior_transform():
    stacker = _stacker(testing=True)
    sample_count = sum(stacker.K) * 3
    core.prepare_entropy(stacker, 3, chunk_rows=sample_count)
    for vp in stacker.vp_list:
        assert vp.parameter_transformer.calls == {
            "forward": 1,
            "inverse": 1,
            "jacobian": 2,
        }


def test_direct_unnormalized_weight_gradient_includes_both_entropy_terms():
    _, n_samples, logq, integrals, _ = _fixed_problem()
    weights = np.array([0.7, 1.4, 0.3, 2.1, 0.9, 0.4])
    elbo, _, gradient = core.objective_weights(
        weights, logq, integrals, n_samples
    )
    numerical = _central_difference(
        lambda value: core.objective_weights(
            value, logq, integrals, n_samples
        )[0],
        weights,
    )
    assert np.isfinite(elbo)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-7)


def test_all_weight_logit_gradient_at_moderate_weights():
    counts, n_samples, logq, integrals, base = _fixed_problem()
    logits = np.array([-0.8, 0.3, 1.1, -0.2, 0.6, -1.0])

    def evaluate(value):
        return core.objective_logits(
            value,
            base,
            counts,
            "all-weights",
            logq,
            integrals,
            n_samples,
        )[0]

    gradient = core.objective_logits(
        logits,
        base,
        counts,
        "all-weights",
        logq,
        integrals,
        n_samples,
    )[2]
    numerical = _central_difference(evaluate, logits)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize(
    ("version", "logits"),
    [
        ("all-weights", [32.0, -31.0, 7.0, -12.0, 1.0, 4.0]),
        ("posterior-only", [28.0, -27.0, 3.0]),
    ],
)
def test_logit_gradients_at_extreme_values_and_gauge_invariance(
    version, logits
):
    counts, n_samples, logq, integrals, base = _fixed_problem()
    logits = np.asarray(logits, dtype=np.float64)

    def evaluate(value):
        return core.objective_logits(
            value,
            base,
            counts,
            version,
            logq,
            integrals,
            n_samples,
        )

    elbo, entropy, gradient, weights = evaluate(logits)
    numerical = _central_difference(lambda value: evaluate(value)[0], logits)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-7)
    shifted = evaluate(logits + 917.0)
    np.testing.assert_allclose(shifted[0], elbo, rtol=0, atol=2e-14)
    np.testing.assert_allclose(shifted[1], entropy, rtol=0, atol=2e-14)
    np.testing.assert_allclose(shifted[3], weights, rtol=0, atol=2e-14)
    assert abs(np.sum(gradient)) < 2e-14
    assert weights.dtype == np.float64


def test_posterior_only_uses_uneven_counts_and_base_component_weights():
    counts, n_samples, logq, integrals, base = _fixed_problem()
    logits = np.array([-0.7, 0.2, 1.4], dtype=np.float64)
    _, _, gradient, weights = core.objective_logits(
        logits,
        base,
        counts,
        "posterior-only",
        logq,
        integrals,
        n_samples,
    )
    expected = np.exp(np.repeat(logits, counts)) * base
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected, rtol=2e-15, atol=0)
    numerical = _central_difference(
        lambda value: core.objective_logits(
            value,
            base,
            counts,
            "posterior-only",
            logq,
            integrals,
            n_samples,
        )[0],
        logits,
    )
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-7)


def test_adam_uses_bias_correction_and_epsilon_after_sqrt():
    parameters = np.array([0.5, -1.0], dtype=np.float64)
    gradient = np.array([0.2, -0.4], dtype=np.float64)
    zeros = np.zeros(2, dtype=np.float64)
    updated, first, second = core._adam_step(
        parameters, gradient, zeros, zeros, 1, 0.1
    )
    expected = parameters - 0.1 * gradient / (np.abs(gradient) + 1e-8)
    np.testing.assert_allclose(updated, expected, rtol=0, atol=2e-16)
    np.testing.assert_allclose(first, 0.1 * gradient, rtol=2e-15, atol=0)
    np.testing.assert_allclose(second, 0.001 * gradient**2, rtol=1e-15)


@pytest.mark.parametrize("version", ["all-weights", "posterior-only"])
def test_optimize_is_float64_json_traceable_and_caches_first_draw(
    monkeypatch, version
):
    stacker = _stacker()
    k_total = sum(stacker.K)
    n_samples = 2
    logq = np.zeros((k_total * n_samples, k_total), dtype=np.float64)
    corrections = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    monkeypatch.setattr(
        core,
        "prepare_entropy",
        lambda unused_stacker, unused_n: (logq.copy(), corrections.copy()),
    )
    # Equal densities and integrals equal to their mean give a constant
    # objective, so upstream stopping occurs after the initial best plus five
    # non-improvements, with no update on the stopping evaluation.
    stacker.I = corrections.reshape(1, -1)
    result = core.optimize_numpy(
        stacker,
        n_samples=n_samples,
        max_steps=20,
        version=version,
        trace=True,
    )
    assert result["steps"] == 6
    assert result["stopped"] is True
    assert result["best_iteration"] == 0
    assert result["weights"].dtype == np.float64
    np.testing.assert_allclose(result["weights"].sum(), 1.0, rtol=0, atol=0)
    np.testing.assert_array_equal(
        stacker.I_corrected, np.zeros((1, k_total), dtype=np.float64)
    )
    assert len(result["trace"]["weights"]) == result["steps"]
    assert isinstance(result["trace"]["weights"][0], list)
    assert set(result["timings"]) == {"preparation", "objective", "optimizer"}
