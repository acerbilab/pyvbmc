"""Algebra and numerical boundaries of the S-VBMC shrinkage report."""

import numpy as np
import pytest

from pyvbmc.svbmc._elbo_shrinkage import _two_level_shrinkage


def estimate(
    means,
    covariances,
    *,
    own=None,
    selected=None,
    entropy=0.0,
    samples=None,
):
    """Call the private estimator with compact run-level inputs."""
    means = [np.asarray(x, dtype=np.float64) for x in means]
    if samples is None:
        samples = [x[None, :] for x in means]
    if own is None:
        own = [np.full(x.size, 1.0 / x.size) for x in means]
    if selected is None:
        n = sum(x.size for x in means)
        selected = np.full(n, 1.0 / n)
    return _two_level_shrinkage(
        samples,
        [np.asarray(C, dtype=np.float64)[None, :, :] for C in covariances],
        means,
        own,
        selected,
        entropy,
    )


def test_full_covariance_changes_within_run_shrinkage():
    diagonal, diagonal_share = estimate(
        [[0.0, 2.0]],
        [np.eye(2)],
        selected=[0.75, 0.25],
        entropy=1.0,
    )
    correlated, correlated_share = estimate(
        [[0.0, 2.0]],
        [[[1.0, 0.5], [0.5, 1.0]]],
        selected=[0.75, 0.25],
        entropy=1.0,
    )

    assert diagonal == pytest.approx(1.75)
    assert correlated == pytest.approx(1.625)
    assert diagonal_share == pytest.approx(0.5)
    assert correlated_share == pytest.approx(0.25)


def test_between_hyperparameter_covariance_and_ns_one():
    samples = [np.array([[-1.0, 3.0], [1.0, 1.0]])]
    with_between, share = estimate(
        [[0.0, 2.0]],
        [np.zeros((2, 2))],
        selected=[0.75, 0.25],
        samples=samples,
    )
    ns_one, ns_one_share = estimate(
        [[0.0, 2.0]],
        [np.zeros((2, 2))],
        selected=[0.75, 0.25],
    )

    assert with_between == pytest.approx(1.0)
    assert share == pytest.approx(2.0)
    assert ns_one == pytest.approx(0.5)
    assert ns_one_share == pytest.approx(0.0)


def test_unequal_original_weights_drive_between_run_levels():
    means = [np.array([0.0, 2.0]), np.array([10.0, 14.0])]
    covariances = [np.ones((2, 2)), np.full((2, 2), 4.0)]
    own = [np.array([0.75, 0.25]), np.array([0.25, 0.75])]
    selected = np.array([0.4, 0.1, 0.2, 0.3])

    actual, share = estimate(
        means, covariances, own=own, selected=selected, entropy=0.75
    )

    levels = np.array([0.5, 13.0])
    variances = np.array([1.0, 4.0])
    spread = np.var(levels, ddof=1)
    excess = spread - np.mean(variances)
    factors = excess / (excess + variances)
    shifted = levels.mean() + factors * (levels - levels.mean())
    corrected = np.concatenate(
        [means[m] + shifted[m] - levels[m] for m in range(2)]
    )
    expected = selected @ corrected + 0.75

    assert actual == pytest.approx(expected)
    assert share == pytest.approx(0.0)


def test_single_run_single_component_and_zero_covariance_are_unchanged():
    one_run, _ = estimate(
        [[0.0, 2.0]],
        [np.eye(2)],
        selected=[0.5, 0.5],
        entropy=3.0,
    )
    one_component, share = estimate(
        [[2.0], [8.0]],
        [[[1.0]], [[1.0]]],
        selected=[0.75, 0.25],
    )
    zero_covariance, zero_share = estimate(
        [[0.0, 2.0], [6.0, 10.0]],
        [np.zeros((2, 2)), np.zeros((2, 2))],
        own=[[0.8, 0.2], [0.25, 0.75]],
        selected=[0.1, 0.2, 0.3, 0.4],
    )

    assert one_run == pytest.approx(4.0)
    # Run-level variance one gives excess variance 17 and factors 17/18.
    shifted = 5.0 + (17.0 / 18.0) * (np.array([2.0, 8.0]) - 5.0)
    assert one_component == pytest.approx(np.dot([0.75, 0.25], shifted))
    assert share is None
    assert zero_covariance == pytest.approx(
        np.dot([0.1, 0.2, 0.3, 0.4], [0.0, 2.0, 6.0, 10.0])
    )
    assert zero_share == pytest.approx(0.0)


def test_zero_covariance_stays_exact_when_spread_underflows():
    selected = np.array([0.75, 0.25])
    means = np.array([0.0, 1e-200])
    value, share = estimate([means], [np.zeros((2, 2))], selected=selected)
    assert value == float(selected @ means)
    assert share is None


def test_zero_excess_variance_and_identical_levels_are_finite():
    value, share = estimate(
        [[1.0, 1.0], [1.0, 1.0]],
        [np.eye(2), np.eye(2)],
        selected=[0.1, 0.2, 0.3, 0.4],
    )
    assert value == pytest.approx(1.0)
    assert share is None


def test_noise_share_renormalizes_over_positive_selected_mass():
    value, share = estimate(
        [[0.0, 2.0], [4.0, 8.0]],
        [np.eye(2), 2.0 * np.eye(2)],
        selected=[0.0, 0.0, 0.25, 0.75],
    )
    assert np.isfinite(value)
    # The first run has share 1/2 but zero selected mass. The second has
    # spread 8 and noise 2.
    assert share == pytest.approx(0.25)


def test_no_positive_spread_has_no_noise_share_but_keeps_estimate():
    value, share = estimate([[2.0, 2.0]], [np.eye(2)], selected=[0.25, 0.75])
    assert value == pytest.approx(2.0)
    assert share is None


def test_indefinite_but_solvable_covariance_is_eligible():
    value, share = estimate(
        [[0.0, 2.0]],
        [[[1.0, 2.0], [2.0, 1.0]]],
        own=[[0.5, 0.5]],
        selected=[0.75, 0.25],
    )
    assert value == pytest.approx(0.25)
    assert share == pytest.approx(-0.5)


@pytest.mark.parametrize(
    "means,covariance,warning",
    [
        (
            [[0.0, 2.0]],
            [[[0.0, -1.0], [-1.0, 0.0]]],
            "singular within-run linear solve.*retained run 0",
        ),
        (
            [[0.0, 2.0]],
            [[[-1.0, 0.0], [0.0, -1.0]]],
            "negative run-level variance.*retained run 0",
        ),
    ],
)
def test_undefined_numerics_warn_and_return_none(means, covariance, warning):
    with pytest.warns(RuntimeWarning, match=warning):
        result = estimate(means, covariance)
    assert result == (None, None)


def test_finite_extreme_inputs_that_overflow_return_none():
    samples = [np.array([[1e308, -1e308], [-1e308, 1e308]])]
    with pytest.warns(RuntimeWarning, match="nonfinite component covariance"):
        result = estimate(
            [[0.0, 1.0]],
            [np.zeros((2, 2))],
            samples=samples,
        )
    assert result == (None, None)
