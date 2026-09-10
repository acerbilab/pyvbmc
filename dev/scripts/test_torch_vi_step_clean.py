"""Focused equivalence gates for the trace-disabled Stage 4 adapter."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

try:
    from torch_vi_step import run_step
except ModuleNotFoundError:
    from dev.scripts.torch_vi_step import run_step


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "pyvbmc" / "testing" / "oracles" / "fixtures"


def _oracle_state(name: str, seed: int = 1701):
    from pyvbmc.testing.oracles._state import build_state, load_snapshot

    return build_state(load_snapshot(FIXTURES / name), rng=seed)


def _assert_vp_equal(left, right):
    assert left.K == right.K
    for name in ("mu", "sigma", "lambd", "w", "eta"):
        np.testing.assert_array_equal(
            getattr(left, name), getattr(right, name)
        )
    assert set(left.stats) == set(right.stats)
    for key in left.stats:
        np.testing.assert_array_equal(left.stats[key], right.stats[key])


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("fixture", ["normal_D2_warmup", "normal_D2_K1"])
def test_trace_disabled_matches_traced_complete_fit(backend, fixture):
    if backend == "torch":
        pytest.importorskip("torch")
    source = _oracle_state(fixture, seed=99)
    original = copy.deepcopy(source["vp"])

    traced = run_step(
        source,
        backend=backend,
        device="cpu",
        seed=1701,
        trace_enabled=True,
    )
    clean = run_step(
        source,
        backend=backend,
        device="cpu",
        seed=1701,
        trace_enabled=False,
    )

    _assert_vp_equal(clean["vp"], traced["vp"])
    for name in ("mu", "sigma", "lambd", "w", "eta"):
        np.testing.assert_array_equal(
            getattr(source["vp"], name), getattr(original, name)
        )
    np.testing.assert_array_equal(clean["var_ss"], traced["var_ss"])
    assert clean["pruned"] == traced["pruned"]
    assert clean["result"]["budgets"] == traced["result"]["budgets"]
    assert (
        clean["result"]["iterations_by_start"]
        == traced["result"]["iterations_by_start"]
    )
    assert (
        clean["result"]["stopping_reasons"]
        == traced["result"]["stopping_reasons"]
    )
    assert clean["result"]["dispatch"]["objective_backend"] == backend
    assert all(
        clean["result"]["dispatch"][key]
        for key in ("sieve", "optimizer", "fine_scoring", "pruning")
    )


def test_trace_disabled_return_is_thin_and_timing_is_synchronized():
    source = _oracle_state("normal_D2_K1", seed=99)
    result = run_step(
        source,
        backend="numpy",
        device="cpu",
        seed=1701,
        trace_enabled=False,
    )

    assert set(result) == {
        "vp",
        "var_ss",
        "pruned",
        "backend",
        "device",
        "boost",
        "fixed_updates",
        "trace_enabled",
        "result",
        "timing",
        "source_hashes",
    }
    assert result["trace_enabled"] is False
    assert "trace" not in result
    assert "candidates" not in result
    assert "_context" not in result
    assert result["timing"]["total"] >= result["timing"]["setup"]
    assert result["timing"]["sync_count"] == 0


def test_trace_enabled_requires_boolean():
    with pytest.raises(TypeError, match="trace_enabled must be boolean"):
        run_step({}, trace_enabled="no")
