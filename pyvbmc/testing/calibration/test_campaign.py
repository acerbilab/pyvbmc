"""Focused scheduler and numerical tests for the calibration campaign."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pyvbmc.calibration import _campaign as campaign


class _FakeClock:
    def __init__(self, call_seconds):
        self.now = 0.0
        self.call_seconds = call_seconds

    def __call__(self):
        return self.now

    def advance_call(self):
        self.now += self.call_seconds


def test_kernel_timer_is_independent_of_coarse_watchdog_clock():
    timer = _FakeClock(0.00002)
    watchdog = campaign._Watchdog(lambda: 100.0, 400.0, timer=timer)
    _, elapsed = watchdog.call("short kernel", timer.advance_call)
    assert elapsed == 0.00002


def test_unresolved_duration_is_incomplete_not_a_speedup():
    watchdog = campaign._Watchdog(lambda: 100.0, 400.0)
    with pytest.raises(campaign._IncompleteCoverage, match="timer did not"):
        watchdog.call("unresolved", lambda: None)


def _fake_kernels(clock):
    def make_vp(D, K, *, rng, transformer=None):
        return SimpleNamespace(
            D=D, K=K, rng=rng, parameter_transformer=transformer
        )

    def pdf(vp, x, *, orig_flag, log_flag, grad_flag, budget):
        del orig_flag, budget
        clock.advance_call()
        value = np.full((x.shape[0], 1), -0.25 if log_flag else 0.75)
        if grad_flag:
            return value, np.zeros((x.shape[0], vp.D), dtype=np.float64)
        return value

    def public_pdf(vp, x, *, orig_flag, log_flag, grad_flag, rng=None):
        del rng
        return pdf(
            vp,
            x,
            orig_flag=orig_flag,
            log_flag=log_flag,
            grad_flag=grad_flag,
            budget=campaign.DEFAULT_BUDGET,
        )

    def entropy(vp, Ns, *, grad_flags, jacobian_flag, rng, budget):
        del jacobian_flag, budget
        clock.advance_call()
        even = int(np.ceil(Ns / 2)) * 2
        rng.standard_normal((vp.K, even // 2, vp.D))
        size = (
            vp.D * vp.K * bool(grad_flags[0])
            + vp.K * bool(grad_flags[1])
            + vp.D * bool(grad_flags[2])
            + vp.K * bool(grad_flags[3])
        )
        return np.float64(1.25), np.zeros(size, dtype=np.float64)

    def public_entropy(vp, Ns, *, grad_flags, jacobian_flag, rng, budget=None):
        del budget
        return entropy(
            vp,
            Ns,
            grad_flags=grad_flags,
            jacobian_flag=jacobian_flag,
            rng=rng,
            budget=campaign.DEFAULT_BUDGET,
        )

    return campaign._KernelAPI(
        make_vp,
        pdf,
        public_pdf,
        entropy,
        public_entropy,
        lambda D: SimpleNamespace(D=D),
    )


def _small_recipe():
    return (
        campaign._Workload("pdf", "pdf", 2, 3, 5),
        campaign._Workload(
            "entropy_grad", "grad", 2, 3, 6, (True, True, True, True)
        ),
        campaign._Workload(
            "entropy_value", "value", 2, 3, 8, (False, False, False, False)
        ),
    )


def _timing_workload(name, candidate, default, control=None, aliases=None):
    aliases = aliases or {
        str(budget): budget for budget in campaign.CANDIDATE_BUDGETS
    }
    times = {
        str(budget): [1.0] * campaign.DISCOVERY_ROUNDS
        for budget in campaign.CANDIDATE_BUDGETS
    }
    times[str(2**14)] = list(candidate)
    times[str(campaign.DEFAULT_BUDGET)] = list(default)
    return {
        "name": name,
        "aliases": aliases,
        "round_seconds": times,
        "default_control_seconds": list(control or default),
    }


def test_balanced_orders_cover_every_position_before_reversing_cycles():
    items = campaign.CANDIDATE_BUDGETS
    orders = campaign._balanced_orders(items, 10)
    for position in range(len(items)):
        assert sorted(order[position] for order in orders[:5]) == sorted(items)
        assert sorted(order[position] for order in orders[5:]) == sorted(items)
    assert orders[0] == items
    assert orders[5] == tuple(reversed(items))


def test_collapsed_alias_orders_balance_each_aligned_round():
    aliases = {
        2**14: 2**14,
        2**15: 2**15,
        2**16: 2**16,
        2**17: 2**16,
        2**18: 2**16,
    }
    for round_index in range(campaign.DISCOVERY_ROUNDS):
        orders = campaign._aligned_timing_orders(aliases, round_index)
        assert len(orders) == 3
        representatives = sorted(set(aliases.values()))
        for position in range(3):
            assert (
                sorted(order[position] for order in orders) == representatives
            )


def test_heldout_balances_positions_and_every_pair_direction():
    arms = (0, 1, 2, 3)
    orders = campaign._heldout_orders(campaign.HELDOUT_ROUNDS)
    for position in range(4):
        assert sorted(order[position] for order in orders) == sorted(arms)
    for left in arms:
        for right in arms:
            if left < right:
                assert (
                    sum(
                        order.index(left) < order.index(right)
                        for order in orders
                    )
                    == 2
                )


def test_discovery_control_alternates_around_candidate_rounds(monkeypatch):
    checkpoints = []

    def observe(watchdog, checkpoint, *args):
        checkpoints.append(checkpoint)
        return 1.0

    monkeypatch.setattr(campaign, "_timed_invoke", observe)
    workload = campaign._Workload("pdf", "sieve", 4, 20, 8192)
    campaign._measure_discovery_group(
        "pdf", (workload,), {"sieve": ()}, None, None
    )
    for round_index in range(campaign.DISCOVERY_ROUNDS):
        observations = [
            name
            for name in checkpoints
            if name.startswith(f"discovery:sieve:{round_index}:")
        ]
        expected_position = 0 if round_index % 2 == 0 else -1
        assert observations[expected_position].endswith(":control")


def test_group_selection_uses_aligned_geometric_rounds_and_tie_break():
    quick = [0.75] * campaign.DISCOVERY_ROUNDS
    workloads = [
        _timing_workload("one", quick, [1.0] * 5),
        _timing_workload("two", quick, [1.0] * 5),
    ]
    valid = {budget: True for budget in campaign.CANDIDATE_BUDGETS}
    result = campaign._select_group(workloads, valid, 5)
    assert result["accepted"]
    assert result["budget"] == 2**14
    assert result["candidates"][str(2**14)]["median_group_speedup"] == 4 / 3


def test_selection_rejects_workload_regression_controls_aliases_and_gaps():
    valid = {budget: True for budget in campaign.CANDIDATE_BUDGETS}
    regression = [
        _timing_workload("fast", [0.5] * 5, [1.0] * 5),
        _timing_workload("slow", [1.1] * 5, [1.0] * 5),
    ]
    assert campaign._select_group(regression, valid, 5)["budget"] == 2**16

    noisy_control = [
        _timing_workload("noise", [0.7] * 5, [1.0] * 5, control=[1.1] * 5)
    ]
    assert campaign._select_group(noisy_control, valid, 5)["budget"] == 2**16

    aliases = {str(budget): 2**16 for budget in campaign.CANDIDATE_BUDGETS}
    equivalent = [
        _timing_workload("alias", [0.5] * 5, [1.0] * 5, aliases=aliases)
    ]
    assert campaign._select_group(equivalent, valid, 5)["budget"] == 2**16

    incomplete = [_timing_workload("gap", [0.7] * 4, [1.0] * 5)]
    assert campaign._select_group(incomplete, valid, 5)["budget"] == 2**16


def test_output_comparison_rejects_broadcastable_shape_mismatch():
    actual = np.ones((2, 1), dtype=np.float64)
    expected = np.ones((2,), dtype=np.float64)
    result = campaign._output_comparison(actual, expected, exact=False)
    assert not result["pass"]
    assert not result["shapes_equal"]


def test_heldout_rejects_unconfirmed_gain_and_bad_same_budget_control():
    discovery = [_timing_workload("one", [0.7] * 5, [1.0] * 5)]
    heldout = [
        {
            "name": "one",
            "default_seconds": [1.0] * 4,
            "selected_seconds": [0.95] * 4,
            "default_control_seconds": [1.0] * 4,
            "selected_control_seconds": [0.95] * 4,
        }
    ]
    result = campaign._validate_heldout(discovery, heldout, 2**14)
    assert not result["pass"]
    assert result["accepted_budget"] == campaign.DEFAULT_BUDGET

    heldout[0]["selected_seconds"] = [0.7] * 4
    heldout[0]["selected_control_seconds"] = [0.8] * 4
    result = campaign._validate_heldout(discovery, heldout, 2**14)
    assert not result["pass"]


def test_fake_campaign_completes_after_thirty_second_estimate():
    clock = _FakeClock(0.3)
    events = []
    result = campaign.run_campaign(
        progress=events.append,
        deadline=300.0,
        _clock=clock,
        _kernel_api=_fake_kernels(clock),
        _recipe=_small_recipe(),
    )
    assert result["status"] == "complete"
    assert result["report"]["timings_seconds"]["total"] > 30.0
    assert result["settings"] == campaign._defaults()
    assert events[-1]["stage"] == "complete"
    assert {"pdf", "entropy_grad", "entropy_value"} <= {
        event["stage"] for event in events
    }
    json.dumps(result["report"], allow_nan=False)
    from pyvbmc.calibration._cache import _validate_json_value

    _validate_json_value(result["report"])
    for setting, summary in result["report"]["summary"].items():
        assert setting in campaign.SETTING_GROUPS.values()
        assert summary["selected"] == campaign.DEFAULT_BUDGET


def test_watchdog_stops_only_after_call_and_returns_no_partial_winner():
    clock = _FakeClock(1.0)
    result = campaign.run_campaign(
        deadline=20.0,
        _clock=clock,
        _kernel_api=_fake_kernels(clock),
        _recipe=_small_recipe(),
    )
    assert result["status"] == "incomplete"
    assert result["settings"] == campaign._defaults()
    assert result["report"]["watchdog"]["stopped"]
    assert result["report"]["watchdog"]["maximum_kernel_call_seconds"] == 1.0


def test_incomplete_recipe_returns_defaults_without_partial_settings():
    clock = _FakeClock(0.01)
    result = campaign.run_campaign(
        deadline=300.0,
        _clock=clock,
        _kernel_api=_fake_kernels(clock),
        _recipe=(campaign._Workload("pdf", "only", 2, 3, 5),),
    )
    assert result["status"] == "incomplete"
    assert result["settings"] == campaign._defaults()
    assert "all setting groups" in result["report"]["reason"]


def test_mutating_baseline_kernel_invalidates_campaign():
    clock = _FakeClock(0.01)
    kernels = _fake_kernels(clock)
    original_pdf = kernels.pdf

    def mutating_pdf(vp, x, **kwargs):
        x[0, 0] += 1.0
        return original_pdf(vp, x, **kwargs)

    def mutating_public_pdf(vp, x, **kwargs):
        kwargs.pop("rng", None)
        return mutating_pdf(vp, x, budget=campaign.DEFAULT_BUDGET, **kwargs)

    kernels = kernels._replace(
        pdf=mutating_pdf, public_pdf=mutating_public_pdf
    )
    result = campaign.run_campaign(
        deadline=300.0,
        _clock=clock,
        _kernel_api=kernels,
        _recipe=_small_recipe(),
    )
    assert result["status"] == "invalid"
    assert result["settings"] == campaign._defaults()


def test_workspace_estimate_bounds_measured_incremental_peaks():
    """The analytical bound covers the retained allocator diagnostics."""
    workloads = {workload.name: workload for workload in campaign._workloads()}
    # Incremental tracemalloc peaks with each workload preallocated, recorded
    # in dev/scripts/runs/calibration_integration_20260909/.
    measured = (
        ("small_value", 2**16, 24_200),
        ("sieve_gradient", 2**16, 2_568_064),
        ("boost_d4_k50", 2**18, 7_512_096),
        ("boost_d15_k50", 2**18, 6_861_480),
        ("active_d4_k20", 2**16, 2_076_168),
        ("fine_d15_k26", 2**18, 25_567_888),
    )
    for name, budget, traced_peak in measured:
        estimate = campaign._workspace_estimate(workloads[name], budget)
        assert estimate["bytes"] >= traced_peak
        assert "25%" in estimate["label"]

    assert all(
        campaign._workspace_estimate(workload, budget)["within_limit"]
        for workload in workloads.values()
        for budget in campaign.CANDIDATE_BUDGETS
    )


def test_small_real_helpers_preserve_numerics_rng_and_state():
    kernels = campaign._load_kernel_api()
    watchdog = campaign._Watchdog(
        lambda: 0.0, 300.0, timer=campaign.time.perf_counter
    )
    workload = campaign._Workload(
        "entropy_grad", "small_real", 2, 3, 10, (True, True, True, True)
    )
    problem = campaign._make_problem(workload, kernels, 123)
    result = campaign._validate_workload(workload, problem, kernels, watchdog)
    assert result["pass"]
    assert result["inputs_vp_and_vp_rng_unchanged"]
    assert all(check["float64"] for check in result["checks"])
    assert all(check["rng_advancement_exact"] for check in result["checks"])
