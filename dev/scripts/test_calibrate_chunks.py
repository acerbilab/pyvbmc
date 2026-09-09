"""Focused checks for the developer-only chunk calibration harness."""

import copy
from pathlib import Path

import numpy as np
import pytest

from dev.scripts import calibrate_chunks as calibration


def _fake_pdf(self, x):
    step = max(1, 2**15 // max(1, self.K * self.D))
    return x[:step]


def _fake_entropy(vp, Ns):
    budget = 123
    return budget + Ns


def test_source_clones_match_default_code_and_keep_globals_private():
    vp = calibration._make_vp(3, 4, 10)
    x = np.random.default_rng(11).normal(size=(31, 3))
    raw_pdf = calibration.inspect.unwrap(calibration.VariationalPosterior.pdf)
    pdf_clone = calibration.make_pdf_clone(calibration.DEFAULT_BUDGET)
    expected_pdf = raw_pdf(vp, x, orig_flag=False, grad_flag=True)
    actual_pdf = pdf_clone(vp, x, orig_flag=False, grad_flag=True)
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(actual_pdf, expected_pdf, strict=True)
    )

    expected_rng = np.random.default_rng(12)
    actual_rng = np.random.default_rng(12)
    expected_entropy = calibration.entmc_vbmc(vp, 18, rng=expected_rng)
    entropy_clone = calibration.make_entropy_clone(calibration.DEFAULT_BUDGET)
    actual_entropy = entropy_clone(vp, 18, rng=actual_rng)
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(
            actual_entropy, expected_entropy, strict=True
        )
    )
    assert calibration._same_state(
        actual_rng.bit_generator.state, expected_rng.bit_generator.state
    )
    assert entropy_clone.__globals__ is not calibration.entmc_vbmc.__globals__
    assert (
        entropy_clone.__globals__["_MAX_TENSOR_ELEMENTS"]
        == calibration.DEFAULT_BUDGET
    )


def test_source_guards_fail_closed_and_reject_invalid_budgets():
    with pytest.raises(calibration.SourceGuardError):
        calibration.make_pdf_clone(100, _fake_pdf)
    with pytest.raises(calibration.SourceGuardError):
        calibration.make_entropy_clone(100, _fake_entropy)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            calibration.make_pdf_clone(bad)


def test_effective_chunk_deduplication_prefers_default_representative():
    tiny = calibration.Workload("pdf", "tiny", 2, 2, 3)
    aliases, representatives = calibration._representatives(
        tiny, calibration.CANDIDATE_BUDGETS
    )
    assert representatives == (calibration.DEFAULT_BUDGET,)
    assert set(aliases.values()) == {calibration.DEFAULT_BUDGET}

    assert calibration.effective_pdf_signature(100, 3, 7, 11) == (4,)
    assert calibration.effective_entropy_signature(2600, 3, 7, 40) == (
        3,
        40,
        40,
    )
    assert calibration.effective_entropy_signature(500, 3, 7, 40) == (
        1,
        23,
        40,
    )


def _selection_case(candidate_times, default_times, complete=True):
    return {
        "complete": complete,
        "budget_aliases": {
            str(2**14): 2**14,
            str(2**15): 2**15,
            str(2**16): 2**16,
            str(2**17): 2**17,
            str(2**18): 2**18,
        },
        "round_seconds": {
            str(2**14): candidate_times,
            str(2**15): default_times,
            str(2**16): default_times,
            str(2**17): default_times,
            str(2**18): default_times,
        },
    }


def test_selection_is_conservative_and_incomplete_campaign_defaults():
    fast = _selection_case([0.7, 0.72, 0.71], [1.0, 1.02, 0.99])
    selection = calibration.select_budget([fast])
    assert selection["accepted"]
    assert selection["budget"] == 2**14

    noisy = _selection_case([0.94, 1.03, 0.96], [1.0, 1.0, 1.0])
    assert calibration.select_budget([noisy])["budget"] == 2**16
    assert (
        calibration.select_budget([_selection_case([], [], False)])["budget"]
        == 2**16
    )


def test_numerical_validation_covers_partial_chunks_flags_and_rng_contracts():
    validation = calibration.numerical_validation()
    assert validation["pass"]
    assert validation["pdf_state_unchanged"]
    assert validation["entropy_state_unchanged"]
    assert validation["numpy_global_rng_unchanged"]
    variants = {check["variant"] for check in validation["checks"]}
    assert "gradient_partial_rows" in variants
    assert "all_grad_jacobian_partial_components" in variants
    assert "all_grad_raw_partial_samples" in variants
    assert "sigma_only_partial_samples" in variants
    assert "value_only_default_clone" in variants
    assert all(check["float64"] for check in validation["checks"])
    assert all(
        check.get("rng_advancement_exact", True)
        for check in validation["checks"]
    )


def test_problem_construction_does_not_consume_numpy_global_rng():
    before = copy.deepcopy(np.random.get_state())
    workload = calibration.Workload("pdf", "small", 2, 3, 5)
    calibration._problem(workload)
    after = np.random.get_state()
    assert calibration._legacy_random_state_equal(before, after)


@pytest.mark.parametrize("count", [4, 5])
def test_timing_order_balances_every_position_in_each_complete_cycle(count):
    items = tuple(range(count))
    for cycle in range(2):
        orders = [
            calibration._balanced_order(items, cycle * count + index)
            for index in range(count)
        ]
        for position in range(count):
            assert sorted(order[position] for order in orders) == list(items)


def test_incomplete_heldout_is_not_a_passing_default_control():
    assert not calibration._heldout_accepts(
        {"complete": False}, calibration.DEFAULT_BUDGET
    )["pass"]


def test_expired_calibration_records_defaults_without_timing(
    monkeypatch, tmp_path
):
    args = calibration._parse_args(
        [
            "--calibration-seconds",
            "1e-12",
            "--output",
            str(tmp_path / "x.json"),
        ]
    )

    def unexpected_timing(*args, **kwargs):
        raise AssertionError("expired campaign must not begin timing")

    monkeypatch.setattr(calibration, "_measure_case", unexpected_timing)
    report = calibration.run_campaign(args)
    assert report["status"] == "partial_defaulted"
    assert report["campaign_numpy_global_rng_unchanged"]
    for kernel in report["selection"].values():
        for setting in kernel["by_regime"].values():
            assert setting["budget"] == calibration.DEFAULT_BUDGET
            assert not setting["accepted"]


def test_cli_defaults_keep_campaign_bounded_and_counts_explicit():
    args = calibration._parse_args(
        ["--quick", "--output", str(Path("ignored-calibration.json"))]
    )
    assert args.deadline == 20.0
    assert args.calibration_seconds == 12.0
    assert args.case_seconds == 3.0
    full = calibration._workloads(False)
    counts = {workload.name: workload.count for workload in full["entropy"]}
    assert counts == {
        "adam": int(np.ceil(100 * 20 ** (-1 / 3))),
        "boost": int(np.ceil(200 * 50 ** (-1 / 3))),
        "boost_high_d": int(np.ceil(200 * 50 ** (-1 / 3))),
        "fine": 4096,
        "fine_high_d": 4096,
        "active": 200,
    }
