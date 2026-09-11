"""Checks of the population assessment's statistics (run by explicit path)."""

import json

import analyze_population_run as analysis
import numpy as np
import pytest
from scipy import stats


def _permutation(delta):
    nonzero = delta[delta != 0]
    return stats.wilcoxon(
        nonzero,
        alternative="two-sided",
        method=stats.PermutationMethod(n_resamples=np.inf),
    )


@pytest.mark.parametrize("trial", range(30))
def test_exact_signed_rank_matches_scipy_enumeration_with_ties(trial):
    rng = np.random.default_rng(trial)
    size = rng.integers(3, 17)
    # Small integers force ties among absolute differences and zeros.
    delta = rng.integers(-4, 5, size).astype(float)
    if (delta != 0).sum() < 2:
        pytest.skip("fewer than two nonzero differences")
    statistic, pvalue = analysis.exact_signed_rank(delta)
    expected = _permutation(delta)
    assert statistic == expected.statistic
    assert pvalue == expected.pvalue


def test_exact_signed_rank_matches_scipy_exact_beyond_enumeration():
    delta = np.random.default_rng(7).normal(size=30)
    statistic, pvalue = analysis.exact_signed_rank(delta)
    expected = stats.wilcoxon(delta, alternative="two-sided", method="exact")
    assert statistic == expected.statistic
    assert pvalue == pytest.approx(expected.pvalue, rel=1e-12)


def test_exact_signed_rank_degenerate_and_bounds():
    assert analysis.exact_signed_rank([0.0, 0.0, 2.0]) == (0.0, 1.0)
    statistic, pvalue = analysis.exact_signed_rank([1.0, 2.0])
    assert (statistic, pvalue) == (0.0, 0.5)
    statistic, pvalue = analysis.exact_signed_rank(np.arange(1, 9.0))
    assert statistic == 0.0
    assert pvalue == 2 / 2**8
    with pytest.raises(ValueError):
        analysis.exact_signed_rank(np.arange(1, 64.0))


def test_paired_tests_family_and_holm():
    rows = []
    for label, deltas in (("a", [1, 2, 3, 4, 5, 6]), ("b", [1, -1, 2, -2])):
        for seed, d in enumerate(deltas):
            rows.append(
                {
                    "label": label,
                    "delta": {k: float(d) for k in analysis.QUALITY},
                    "old_usable": d > 0,
                    "new_usable": False,
                }
            )
    tests = analysis.paired_tests(rows, metrics=analysis.QUALITY)
    assert len(tests) == 8
    by_key = {(t["label"], t["metric"]): t for t in tests}
    assert by_key[("a", "gskl")]["pvalue"] == 2 / 2**6
    assert by_key[("a", "usable")]["losses"] == 6
    assert by_key[("a", "usable")]["pvalue"] == 2 / 2**6
    assert by_key[("b", "gskl")]["pvalue"] == 1.0
    assert all("holm_adjusted_pvalue" in t for t in tests)
    assert not any(t["holm_rejected"] for t in tests)
    json.dumps(tests)  # plain Python floats and bools only
    plain = analysis.paired_tests(rows, metrics=analysis.QUALITY, adjust=False)
    assert not any("holm_rejected" in t for t in plain)


def test_merge_populations_concatenates_labels():
    first = {"x": {"seeds": [0], "rows": [{"elbo_err": 1.0}], "fails": 0}}
    second = {
        "x": {"seeds": [1], "rows": [{"elbo_err": 2.0}], "fails": 1},
        "y": {"seeds": [0], "rows": [{"elbo_err": 3.0}], "fails": 0},
    }
    merged = analysis.merge_populations([first, second])
    assert merged["x"]["seeds"] == [0, 1]
    assert merged["x"]["fails"] == 1
    np.testing.assert_array_equal(merged["x"]["elbo_err"], [1.0, 2.0])
    assert np.isnan(merged["y"]["gskl"]).all()
