"""Focused lifecycle tests for the Stage 4 ``torch.compile`` adapter.

These tests use a fake compiler.  The bounded toolchain and objective smokes
exercise actual Inductor compilation in fresh benchmark workers.
"""

from __future__ import annotations

import types

import pytest

try:
    import torch_vi_compile as compile_adapter
except ModuleNotFoundError:
    from dev.scripts import torch_vi_compile as compile_adapter


class _FakeTorch:
    __version__ = "test"

    def __init__(self):
        self.requests = []

    def compile(self, function, **kwargs):
        self.requests.append((function, kwargs))

        def compiled(*args, **call_kwargs):
            return function(*args, **call_kwargs) + 1

        return compiled


@pytest.fixture
def fixed_evidence(monkeypatch):
    snapshots = []

    def snapshot(_torch=None):
        value = {"dynamo_counters": {"stats": {"calls": len(snapshots)}}}
        snapshots.append(value)
        return value

    monkeypatch.setattr(compile_adapter, "snapshot_compile_evidence", snapshot)
    return snapshots


def test_compiled_forward_is_scoped_reusable_and_counted(fixed_evidence):
    def forward(value, *_args):
        return value * 2

    core = types.SimpleNamespace(_neg_elcbo_forward=forward)
    torch = _FakeTorch()
    session = compile_adapter.CompiledForward(
        core,
        torch_module=torch,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
    )

    assert core._neg_elcbo_forward is forward
    assert torch.requests == [
        (
            forward,
            {"backend": "inductor", "fullgraph": True, "dynamic": False},
        )
    ]
    with session.install():
        assert core._neg_elcbo_forward(3) == 7
    assert core._neg_elcbo_forward is forward
    with session.install():
        assert core._neg_elcbo_forward(4) == 9

    with session.install():
        arguments = (3, None, None, 0.0, 4096, True, None, True, None)
        assert core._neg_elcbo_forward(*arguments) == 6

    evidence = session.evidence()
    assert session.call_count == 3
    assert session.compiled_call_count == 2
    assert session.eager_full_variance_call_count == 1
    assert evidence["call_count"] == 3
    assert evidence["compiled_call_count"] == 2
    assert evidence["eager_full_variance_call_count"] == 1
    assert evidence["installed"] is False
    assert evidence["timeline"][0]["label"] == "created"
    assert (
        evidence["delta_since_creation"]["dynamo_counters"]["stats"]["calls"]
        > 0
    )


def test_install_restores_after_failure_and_rejects_conflict(fixed_evidence):
    def forward(value):
        return value

    core = types.SimpleNamespace(_neg_elcbo_forward=forward)
    session = compile_adapter.CompiledForward(core, torch_module=_FakeTorch())

    with pytest.raises(RuntimeError, match="worker failed"):
        with session.install():
            raise RuntimeError("worker failed")
    assert core._neg_elcbo_forward is forward

    core._neg_elcbo_forward = lambda value: -value
    with pytest.raises(
        RuntimeError, match="changed after session construction"
    ):
        with session.install():
            pass


def test_compile_options_and_evidence_delta(fixed_evidence):
    core = types.SimpleNamespace(_neg_elcbo_forward=lambda value: value)
    with pytest.raises(ValueError, match="mode or options"):
        compile_adapter.CompiledForward(
            core,
            torch_module=_FakeTorch(),
            mode="reduce-overhead",
            options={"max_autotune": True},
        )

    before = {"a": 2, "nested": {"count": 5}, "label": "old"}
    after = {"a": 7, "nested": {"count": 4}, "label": "new"}
    assert compile_adapter.evidence_delta(before, after) == {
        "a": 5,
        "nested": {"count": -1},
        "label": "new",
    }
