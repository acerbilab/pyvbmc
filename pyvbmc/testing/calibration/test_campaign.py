"""Focused scheduler and numerical tests for the calibration campaign."""

import json
import math
from statistics import median
from types import SimpleNamespace

import numpy as np
import pytest

from pyvbmc.calibration import _cache
from pyvbmc.calibration import _campaign as campaign
from pyvbmc.calibration.profile import DEFAULT_CHUNK_ELEMENTS
from pyvbmc.testing.calibration.test_api import identity
from pyvbmc.vbmc import Options


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


def test_regression_veto_admits_exactly_the_named_workload_slowdown():
    assert 1 / campaign._MAX_WORKLOAD_SLOWDOWN == campaign._CONTROL_LOW

    def gate(workload_median):
        rounds = campaign.DISCOVERY_ROUNDS
        ratios = {
            "complete": True,
            "affected_workloads": ["fast", "slow"],
            "per_workload": {
                "fast": [2.0] * rounds,
                "slow": [workload_median] * rounds,
            },
            "rounds": [1.5] * rounds,
        }
        return campaign._gate_ratios(ratios, rounds)

    admitted = 1 / campaign._MAX_WORKLOAD_SLOWDOWN
    assert gate(admitted)["regression_veto_pass"]
    assert gate(admitted)["pass"]

    vetoed = math.nextafter(admitted, 0.0)
    assert not gate(vetoed)["regression_veto_pass"]
    assert not gate(vetoed)["pass"]


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


def test_summary_speedup_describes_the_selected_setting():
    rounds = [0.9, 1.3, 1.35, 1.4]

    def report_with(validation):
        return {
            "groups": {
                group: {
                    "setting": setting,
                    "discovery_selection": {
                        "budget": 2**14,
                        "accepted": True,
                        "reason": "discovery gates passed",
                    },
                    "heldout_validation": validation,
                }
                for group, setting in campaign.SETTING_GROUPS.items()
            }
        }

    rejected = report_with(
        {
            "pass": False,
            "accepted_budget": campaign.DEFAULT_BUDGET,
            "reason": "held-out gates failed",
            "group_speedups": list(rounds),
        }
    )
    assert median(rounds) > 1.0
    summary = campaign._summary(rejected, campaign._defaults(), "complete")
    for setting in campaign.SETTING_GROUPS.values():
        assert summary[setting]["selected"] == campaign.DEFAULT_BUDGET
        assert summary[setting]["reason"] == "held-out gates failed"
        assert summary[setting]["heldout_speedup"] is None

    accepted = report_with(
        {
            "pass": True,
            "accepted_budget": 2**14,
            "reason": "held-out gates passed",
            "group_speedups": list(rounds),
        }
    )
    settings = {
        setting: 2**14 for setting in campaign.SETTING_GROUPS.values()
    }
    summary = campaign._summary(accepted, settings, "complete")
    for setting in campaign.SETTING_GROUPS.values():
        assert summary[setting]["selected"] == 2**14
        assert summary[setting]["heldout_speedup"] == median(rounds)


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
    # in dev/scripts/runs/calibration_integration_20260909/ and, for
    # sieve_value and the value-only active_d4_k20, in
    # dev/scripts/runs/calibration_recipe_v2_20260923/.
    measured = (
        ("small_value", 2**16, 24_200),
        ("sieve_value", 2**16, 2_181_680),
        ("boost_d4_k50", 2**18, 7_512_096),
        ("boost_d15_k50", 2**18, 6_861_480),
        ("active_d4_k20", 2**16, 1_384_376),
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


# The options that set the total number of samples of the Monte Carlo
# entropy, by whether the calls that use them request gradients. The
# package's one call of `entmc_vbmc` is in `_neg_elcbo`, which requests
# gradients when its caller does:
# - with gradients, the stochastic optimization of `optimize_vp` draws the
#   count of `ns_ent`, which `VBMC.final_boost` replaces with `ns_ent_boost`
#   and `active_sample` with `ns_ent_active`;
# - value-only, `_eval_full_elcbo` draws the count of `ns_ent_fine`, which
#   `active_sample` replaces with `ns_ent_fine_active`, and `active_sample`
#   draws `ns_ent_fine_active` to compare the posteriors before and after
#   its update.
# The remaining count options (`ns_ent_fast` and its variants,
# `ns_ent_fine_boost`) default to no samples, which selects the
# deterministic entropy, or to one of the options above.
_ENTROPY_COUNT_OPTIONS = {
    True: ("ns_ent", "ns_ent_boost", "ns_ent_active"),
    False: ("ns_ent_fine", "ns_ent_fine_active"),
}


def _shipped_options(D):
    options = Options("option_configs/basic_vbmc_options.ini", {"D": D})
    options.load_options_file(
        "option_configs/advanced_vbmc_options.ini", {"D": D}
    )
    return options


def test_entropy_workloads_time_calls_the_package_makes():
    """Each entropy workload has the sample count and gradient use of a call.

    The package draws ``ceil(total / K)`` samples per component, ``total``
    being a count option evaluated at ``K``, and a call that requests any
    gradient runs at the gradient budget, a value-only call at the value
    budget.
    """
    entropy = [
        workload
        for workload in campaign._workloads()
        if workload.group.startswith("entropy")
    ]
    assert entropy
    for workload in entropy:
        with_gradients = any(workload.grad_flags)
        options = _shipped_options(workload.D)
        counts = {
            name: math.ceil(options.eval(name, {"K": workload.K}) / workload.K)
            for name in _ENTROPY_COUNT_OPTIONS[with_gradients]
        }
        expected_group = "entropy_grad" if with_gradients else "entropy_value"
        assert workload.group == expected_group, workload.name
        assert workload.count in counts.values(), (
            f"{workload.name} draws {workload.count} samples per component "
            f"{'with' if with_gradients else 'without'} gradients; the "
            f"package's counts for such calls at K={workload.K} are {counts}"
        )


def test_density_workloads_time_value_only_calls():
    """The density workloads time calls without gradients.

    The package requests the gradient of the density only in
    ``VariationalPosterior.mode`` in the transformed space, which no step of
    the algorithm calls. Its optimizer evaluates the density at one point at
    a time, and one point has one block layout at every candidate budget,
    so no setting changes how that call runs. Its screen of starting points
    evaluates the density on a draw of 100,000 points and keeps only the
    values.
    """
    density = [
        workload
        for workload in campaign._workloads()
        if workload.group == "pdf"
    ]
    assert density
    for workload in density:
        assert workload.grad_flags is None, workload.name

    for workload in density:
        one_point = campaign._Workload(
            "pdf",
            f"{workload.name}_one_point_gradient",
            workload.D,
            workload.K,
            1,
            (True, True, True, True),
        )
        aliases = campaign._layout_aliases(one_point)
        assert set(aliases) == set(campaign.CANDIDATE_BUDGETS)
        assert set(aliases.values()) == {campaign.DEFAULT_BUDGET}


def test_record_validator_agrees_with_the_shipped_recipe():
    """The recipe tables and constants of ``_cache`` match the campaign's."""
    recipe = campaign._workloads()
    groups = {}
    for workload in recipe:
        groups.setdefault(workload.group, []).append(workload.name)
    assert {
        group: tuple(names) for group, names in groups.items()
    } == _cache._GROUP_WORKLOADS
    assert _cache._WORKLOAD_SHAPES == {
        workload.name: (
            workload.D,
            workload.K,
            workload.count,
            campaign._effective_count(workload),
        )
        for workload in recipe
    }
    assert _cache._GROUP_SETTINGS == campaign.SETTING_GROUPS
    assert set(_cache._SETTING_NAMES) == set(campaign.SETTING_GROUPS.values())
    assert _cache.CANDIDATE_BUDGETS == frozenset(campaign.CANDIDATE_BUDGETS)
    assert DEFAULT_CHUNK_ELEMENTS == campaign.DEFAULT_BUDGET

    # The validator writes the round counts, the default budget and the
    # recipe version as literals: a complete campaign of the shipped recipe,
    # timed on instant stand-in kernels, has to make a valid record.
    clock = _FakeClock(0.001)
    result = campaign.run_campaign(
        deadline=300.0, _clock=clock, _kernel_api=_fake_kernels(clock)
    )
    assert result["status"] == "complete", result["report"].get("reason")
    _cache.make_record(
        identity=identity(),
        settings=result["settings"],
        report=result["report"],
        elapsed_seconds=1.0,
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
    assert all(check["exact"] for check in result["checks"])
    assert all(check["rng_advancement_exact"] for check in result["checks"])


def test_a_budget_that_moves_the_last_bit_is_not_accepted():
    """A kernel that is not chunk-independent invalidates its budgets."""
    clock = _FakeClock(0.001)
    kernels = _fake_kernels(clock)
    fake_entropy = kernels.entropy

    def drifting_entropy(vp, Ns, *, grad_flags, jacobian_flag, rng, budget):
        H, dH = fake_entropy(
            vp,
            Ns,
            grad_flags=grad_flags,
            jacobian_flag=jacobian_flag,
            rng=rng,
            budget=budget,
        )
        if budget != campaign.DEFAULT_BUDGET:
            H = np.nextafter(H, np.inf)
        return H, dH

    kernels = kernels._replace(entropy=drifting_entropy)
    watchdog = campaign._Watchdog(
        lambda: 0.0, 300.0, timer=campaign.time.perf_counter
    )
    workload = campaign._Workload(
        "entropy_grad", "drifting", 2, 3, 10, (True, True, True, True)
    )
    problem = campaign._make_problem(workload, kernels, 123)
    result = campaign._validate_workload(workload, problem, kernels, watchdog)
    assert not result["pass"]
    # The drift is far inside the tolerance a loose comparison would allow.
    assert all(check["within_tolerance"] for check in result["checks"])
    assert result["valid_budgets"][str(campaign.DEFAULT_BUDGET)]
    assert not any(
        valid
        for budget, valid in result["valid_budgets"].items()
        if int(budget) != campaign.DEFAULT_BUDGET
    )
