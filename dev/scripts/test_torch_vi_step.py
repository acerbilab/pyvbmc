"""Focused gates for the complete Stage 4 variational-fit adapter.

This developer test module is outside default pytest discovery.  The main
experiment process owns execution because the complete-fit checks are compute
bearing; ordinary package CI does not collect them.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

try:
    from torch_vi_step import (
        _RecordingRNG,
        _Trace,
        minimize_adam_tensor,
        rescore_candidates,
        run_numpy_control,
        run_step,
        source_hashes,
    )
except ModuleNotFoundError:
    from dev.scripts.torch_vi_step import (
        _RecordingRNG,
        _Trace,
        minimize_adam_tensor,
        rescore_candidates,
        run_numpy_control,
        run_step,
        source_hashes,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "pyvbmc" / "testing" / "oracles" / "fixtures"


def _oracle_state(name: str, seed: int = 1701):
    from pyvbmc.testing.oracles._state import build_state, load_snapshot

    return build_state(load_snapshot(FIXTURES / name), rng=seed)


def _assert_vp_equal(left, right, atol=0.0, rtol=0.0):
    assert left.K == right.K
    for name in ("mu", "sigma", "lambd", "w", "eta"):
        np.testing.assert_allclose(
            getattr(left, name), getattr(right, name), atol=atol, rtol=rtol
        )
    assert left.optimize_mu == right.optimize_mu
    assert left.optimize_sigma == right.optimize_sigma
    assert left.optimize_lambd == right.optimize_lambd
    assert left.optimize_weights == right.optimize_weights


def test_source_hash_guard_matches_current_control_flow():
    observed = source_hashes(check=True)
    assert set(observed) == {
        "optimize_vp",
        "_eval_full_elcbo",
        "_sieve",
        "_vb_init",
        "minimize_adam",
        "final_boost",
    }


def test_recording_rng_is_exact_and_reconstructable():
    trace = _Trace("numpy", "cpu")
    trace.phase = "adam"
    wrapped = _RecordingRNG(np.random.default_rng(91), trace)
    first = wrapped.standard_normal((2, 3, 4))
    second = wrapped.standard_normal((2, 3, 4))

    events = [
        event
        for event in trace.events
        if event["kind"] == "rng_standard_normal"
    ]
    assert events[0]["storage"] == "actual"
    assert events[1]["storage"] == "reconstructable"
    replay = np.random.default_rng()
    replay.bit_generator.state = copy.deepcopy(
        events[1]["generator_state_before"]
    )
    reconstructed = replay.standard_normal(events[1]["shape"])
    np.testing.assert_array_equal(reconstructed, second)
    assert events[1]["value_hash"] == events[0][
        "value_hash"
    ] or not np.array_equal(first, second)


def test_tensor_adam_exact_fixed_updates_and_trace_association():
    torch = pytest.importorskip("torch")
    from pyvbmc.vbmc.minimize_adam import minimize_adam

    x0 = np.array([-0.7, 1.3], dtype=np.float64)

    def numpy_fun(x):
        return np.sum((x - 0.25) ** 2), 2 * (x - 0.25)

    def tensor_fun(x):
        return torch.sum((x - 0.25) ** 2), 2 * (x - 0.25)

    expected = minimize_adam(
        numpy_fun,
        x0.copy(),
        max_iter=100,
        use_early_stopping=False,
        master_min=0.0007,
        master_max=0.043,
        master_decay=200,
    )
    trace = _Trace("torch", "cpu")
    trace.phase = "adam"
    trace.start_id = 0
    observed = minimize_adam_tensor(
        tensor_fun,
        x0.copy(),
        max_iter=100,
        use_early_stopping=False,
        master_min=0.0007,
        master_max=0.043,
        master_decay=200,
        device="cpu",
        trace=trace,
    )

    np.testing.assert_allclose(
        observed[0].numpy(), expected[0], rtol=0, atol=2e-15
    )
    np.testing.assert_allclose(
        observed[1].numpy(), expected[1], rtol=0, atol=2e-15
    )
    np.testing.assert_allclose(
        observed[2].numpy(), expected[2], rtol=0, atol=2e-15
    )
    np.testing.assert_allclose(observed[3], expected[3], rtol=0, atol=2e-15)
    assert observed[4] == expected[4] == 100
    # Production records f(x_i) beside the already updated x_{i+1}.
    assert expected[3][0] == pytest.approx(numpy_fun(x0)[0])
    assert not np.array_equal(expected[2][:, 0], x0)


def test_tensor_adam_matches_early_stop_and_clipping():
    torch = pytest.importorskip("torch")
    from pyvbmc.vbmc.minimize_adam import minimize_adam

    x0 = np.array([2.0, -2.0])
    lb = np.array([-0.5, -0.5])
    ub = np.array([0.5, 0.5])

    def np_flat(x):
        return 1.0, np.zeros_like(x)

    def torch_flat(x):
        return torch.ones((), dtype=x.dtype), torch.zeros_like(x)

    expected = minimize_adam(np_flat, x0.copy(), lb=lb, ub=ub)
    observed = minimize_adam_tensor(
        torch_flat, x0.copy(), lb=lb, ub=ub, device="cpu"
    )
    assert expected[4] == observed[4] == 40
    np.testing.assert_array_equal(observed[0].numpy(), expected[0])
    np.testing.assert_array_equal(observed[2].numpy(), expected[2])
    assert np.all(observed[2].numpy() <= ub[:, None])
    assert np.all(observed[2].numpy() >= lb[:, None])


@pytest.mark.parametrize("name", ["normal_D2_K1", "normal_D2_warmup"])
def test_numpy_adapter_is_exact_at_current_caller_budgets(name):
    source = _oracle_state(name, seed=99)
    expected = run_numpy_control(source, seed=1701)
    observed = run_step(source, backend="numpy", seed=1701)

    _assert_vp_equal(observed["vp"], expected["vp"])
    assert observed["var_ss"] == expected["var_ss"]
    assert observed["pruned"] == expected["pruned"]
    for key in expected["vp"].stats:
        np.testing.assert_equal(
            observed["vp"].stats[key], expected["vp"].stats[key]
        )
    assert observed["input_unchanged"]
    assert observed["result"]["complete"]
    assert observed["result"]["dispatch_counts"]["fine"] > 0
    if name.endswith("K1"):
        assert observed["result"]["scipy_runs"][0]["method"] == (
            "BFGS (SciPy default)"
        )


def test_numpy_boost_adapter_is_exact_at_current_caller_budgets():
    source = _oracle_state("cigar_D4_largeK", seed=99)
    expected = run_numpy_control(source, seed=1701, boost=True)
    observed = run_step(source, backend="numpy", seed=1701, boost=True)

    _assert_vp_equal(observed["vp"], expected["vp"])
    assert set(observed["vp"].stats) == set(expected["vp"].stats)
    for key in expected["vp"].stats:
        np.testing.assert_equal(
            observed["vp"].stats[key], expected["vp"].stats[key]
        )
    assert (
        observed["result"]["boost"]["changed"] == expected["boost"]["changed"]
    )
    assert observed["result"]["boost"]["elbo"] == expected["boost"]["elbo"]
    assert (
        observed["result"]["boost"]["elbo_sd"] == expected["boost"]["elbo_sd"]
    )


def test_torch_fixed_100_complete_fit_has_all_boundaries():
    pytest.importorskip("torch")
    source = _oracle_state("normal_D2_warmup", seed=99)
    state = {
        **source,
        "fast_opts_N": 1,
        "slow_opts_N": 1,
        "K": source["vp"].K,
    }
    result = run_step(
        state,
        backend="torch",
        device="cpu",
        seed=1701,
        fixed_updates=100,
    )

    assert result["result"]["complete"]
    assert result["result"]["iterations_by_start"] == [100]
    assert result["result"]["stopping_reasons"] == ["max_iter"]
    assert result["result"]["dispatch_counts"]["sieve"] > 0
    assert result["result"]["dispatch_counts"]["adam"] == 100
    assert result["result"]["dispatch_counts"]["fine"] > 0
    assert result["result"]["transformer_identity"]
    kinds = {candidate["kind"] for candidate in result["candidates"]}
    assert {"sieve_ranked", "selected_start", "endpoint", "final"} <= kinds
    rescored = rescore_candidates(result)
    assert len(rescored["seeds"]) == 5
    assert all(len(streams) == 5 for streams in rescored["scores"].values())


def test_boost_wrapper_preserves_guard_and_reports_transformer_identity():
    pytest.importorskip("torch")
    source = _oracle_state("cigar_D4_largeK", seed=99)
    result = run_step(
        source,
        backend="torch",
        device="cpu",
        seed=1701,
        boost=True,
        fixed_updates=100,
    )

    assert result["result"]["complete"]
    assert result["result"]["boost"] is not None
    assert result["result"]["K"] >= source["options"].get(
        "min_final_components"
    )
    assert result["result"]["pruned"] == 0
    assert not result["result"]["transformer_identity"]
    assert result["result"]["transformer_identity_note"] is not None
    assert result["result"]["dispatch_counts"]["adam"] == 100
