"""The dtype-canary helpers on synthetic object graphs: they must find
what they exist to find, stop where they are meant to stop, and fail
when they find nothing."""

import logging
import types

import numpy as np
import pytest

from pyvbmc.testing import (
    assert_float64,
    assert_manifest_float64,
    iter_dtype_leaves,
    non_float64_leaves,
)


class _Node:
    """An instance of a ``pyvbmc`` class: entered through ``__dict__``."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class _Tensor:
    """A stand-in for a tensor of another backend: a ``dtype`` that is
    not NumPy's."""

    dtype = "torch.float32"


def test_offenders_are_found_through_every_container():
    inner = np.empty((2,), dtype=object)
    inner[0] = np.zeros((3,), dtype=np.float32)
    inner[1] = np.ones((2, 2))
    graph = {
        "a": [_Node(inner=inner, scalar=np.float32(1.5))],
        "b": (np.zeros((2,), dtype=np.complex128),),
        "ok": np.zeros((4,)),
        "mask": np.ones((4,), dtype=bool),
        "count": np.arange(3, dtype=np.int64),
        "wide": np.float64(2.0),
        "c": np.complex128(1j),
        "s": {np.float32(0.5)},
        "placeholder": 0,
    }
    offenders, n_leaves = non_float64_leaves(graph, "g")
    assert dict(offenders) == {
        "g['a'][0].inner[0]": "float32",
        "g['a'][0].scalar": "float32",
        "g['b'][0]": "complex128",
        "g['c']": "complex128",
        "g['s'][0]": "float32",
    }
    # The float64 array and scalar, the bool mask, the int64 count and
    # the five offenders are leaves; the Python int is not.
    assert n_leaves == 10
    with pytest.raises(AssertionError, match=r"g\['b'\]\[0\]  <complex128>"):
        assert_float64(graph, "g")


def test_foreign_dtype_is_an_offender():
    offenders, n_leaves = non_float64_leaves(_Node(t=_Tensor()), "n")
    assert n_leaves == 1
    assert offenders == [("n.t", "'torch.float32'")]


def test_walk_stops_at_objects_of_other_packages():
    """A logger or a plain namespace holding a float32 array is a
    boundary, not a finding."""
    other = types.SimpleNamespace(buf=np.zeros((2,), dtype=np.float32))
    log = logging.Logger("dtype-canary-test")
    log.buf = np.zeros((2,), dtype=np.float32)
    graph = _Node(other=other, log=log, own=np.zeros((2,)))
    offenders, n_leaves = non_float64_leaves(graph, "n")
    assert offenders == []
    assert n_leaves == 1


def test_cycles_and_shared_objects_are_visited_once():
    node = _Node(x=np.zeros((2,)))
    node.me = node
    shared = np.zeros((3,))
    graph = [node, node, {"s": shared, "t": shared}, (shared,)]
    graph.append(graph)
    leaves = list(iter_dtype_leaves(graph, "g"))
    assert len(leaves) == 2
    assert assert_float64(graph, "g", min_leaves=2) == 2


def test_empty_walk_fails_on_the_floor():
    with pytest.raises(AssertionError, match="0 dtype leaves"):
        assert_float64({"n": 3, "s": "x"}, "empty")
    with pytest.raises(AssertionError, match="fewer than the 5"):
        assert_float64({"a": np.zeros(2)}, "small", min_leaves=5)


def test_manifest_requires_float64_arrays():
    assert_manifest_float64({"a": np.zeros((2,)), "b": np.ones((1, 1))})
    with pytest.raises(AssertionError) as err:
        assert_manifest_float64(
            {
                "fine": np.zeros((2,)),
                "narrow": np.zeros((2,), dtype=np.float32),
                "integer": np.arange(2, dtype=np.int64),
                "scalar": 1.0,
                "missing": None,
            }
        )
    text = str(err.value)
    for line in (
        "narrow  <float32>",
        "integer  <int64>",
        "scalar  <float>",
        "missing  <NoneType>",
    ):
        assert line in text
    assert "fine" not in text
