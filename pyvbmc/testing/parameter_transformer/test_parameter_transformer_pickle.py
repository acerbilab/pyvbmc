"""Checks on what a pickled or copied ``ParameterTransformer`` holds.

The bounded transforms of a transformer are functions built from its
``bounded_types``. A pickle must not carry them: dill would store them by
value, as bytecode of the Python version that writes the file, and a saved
variational posterior could then not be used under another Python version.
"""

import copy
import pickle

import dill
import numpy as np
import pytest

from pyvbmc.parameter_transformer import ParameterTransformer

D = 3
TRANSFORM_TYPES = ["logit", "probit", "student4"]


def build(transform_type, rotate=False):
    """A transformer with two bounded variables and an unbounded one."""
    rotation = scale = None
    if rotate:
        angle = 0.3
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        scale = np.array([0.5, 2.0, 1.5])
    return ParameterTransformer(
        D,
        np.array([[0.0, -np.inf, -2.0]]),
        np.array([[10.0, np.inf, 3.0]]),
        np.array([[2.0, -1.0, -1.0]]),
        np.array([[6.0, 1.0, 1.0]]),
        scale=scale,
        rotation_matrix=rotation,
        transform_type=transform_type,
    )


def points():
    """Points strictly inside the hard bounds of ``build``."""
    rng = np.random.default_rng(5)
    return np.column_stack(
        [
            rng.uniform(0.5, 9.5, 20),
            rng.normal(0.0, 2.0, 20),
            rng.uniform(-1.9, 2.9, 20),
        ]
    )


def assert_same_maps(first, second):
    x = points()
    u = first(x)
    assert np.array_equal(u, second(x))
    assert np.array_equal(first.inverse(u), second.inverse(u))
    assert np.array_equal(
        first.log_abs_det_jacobian(u), second.log_abs_det_jacobian(u)
    )


@pytest.mark.parametrize("transform_type", TRANSFORM_TYPES)
def test_a_pickled_transformer_holds_no_function(transform_type):
    """Neither dill nor the standard pickle stores a function: dill marks
    one stored by value with its ``_create_function``, and the standard
    pickle refuses a function defined inside another."""
    transformer = build(transform_type)

    assert b"_create_function" not in dill.dumps(transformer)
    assert b"_create_code" not in dill.dumps(transformer)
    pickle.dumps(transformer)


@pytest.mark.parametrize("transform_type", TRANSFORM_TYPES)
@pytest.mark.parametrize("rotate", [False, True])
def test_a_restored_transformer_is_the_same_map(transform_type, rotate):
    """A transformer that went through a pickle or a copy maps points as
    the one it came from, in both directions and in its Jacobian."""
    transformer = build(transform_type, rotate=rotate)

    for restored in (
        dill.loads(dill.dumps(transformer)),
        pickle.loads(pickle.dumps(transformer)),
        copy.deepcopy(transformer),
        copy.copy(transformer),
    ):
        assert restored == transformer
        assert_same_maps(transformer, restored)


def test_functions_stored_by_an_earlier_version_are_dropped():
    """Files written by versions that pickled the bounded transforms still
    hold them. They may be bytecode of another Python version, so restoring
    such a state rebuilds the transforms and never calls the stored ones."""

    def stored_by_another_version(*args):
        raise AssertionError("a function from the file was called")

    transformer = build("probit")
    state = dict(transformer.__dict__)
    state["_bounded_transforms"] = {
        t: {
            "direct": stored_by_another_version,
            "inverse": stored_by_another_version,
            "jacobian": stored_by_another_version,
        }
        for t in transformer.bounded_types
    }
    restored = ParameterTransformer.__new__(ParameterTransformer)
    restored.__setstate__(state)

    assert_same_maps(transformer, restored)
    # The state handed in is the caller's and stays as it was.
    assert "_bounded_transforms" in state
