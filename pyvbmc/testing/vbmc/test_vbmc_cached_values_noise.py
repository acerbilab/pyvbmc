"""Cached starting values and user-provided target noise."""

import numpy as np
import pytest

from pyvbmc import VBMC

D = 2
noisy_target = lambda x: (np.sum(x + 2), 0.5)
noiseless_target = lambda x: np.sum(x + 2)


def _vbmc(target, options, precomputed_evaluations=None):
    x0 = np.vstack((np.zeros(D), np.full(D, 0.25)))
    return VBMC(
        target,
        x0,
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options=options,
        precomputed_evaluations=precomputed_evaluations,
    )


def test_cached_values_are_refused_with_user_provided_noise():
    """``f_vals`` gives the value of a starting point and has no channel
    for its noise, so it cannot describe an observation of a target whose
    noise the user provides. The message names the argument that can."""
    with pytest.raises(ValueError) as err:
        _vbmc(
            noisy_target,
            {"f_vals": [4.0, np.nan], "specify_target_noise": True},
        )
    assert "precomputed_evaluations" in str(err.value)


def test_no_cached_value_is_nothing_to_refuse():
    """A NaN in ``f_vals`` marks a starting point that is still to be
    evaluated, so an ``f_vals`` of NaN alone supplies no value, and there
    is none whose noise is missing."""
    vbmc = _vbmc(
        noisy_target,
        {"f_vals": [np.nan, np.nan], "specify_target_noise": True},
    )
    assert not vbmc.optim_state["cache_active"]


def test_cached_values_are_accepted_without_user_provided_noise():
    """Without user-provided noise the option keeps working."""
    vbmc = _vbmc(noiseless_target, {"f_vals": [4.0, np.nan]})
    assert np.array_equal(
        vbmc.optim_state["cache"]["y_orig"], [4.0, np.nan], equal_nan=True
    )


def test_user_provided_noise_takes_its_observations_precomputed():
    """The argument the message names accepts the same observation with
    its noise."""
    vbmc = _vbmc(
        noisy_target,
        {"specify_target_noise": True},
        precomputed_evaluations=(
            np.zeros((1, D)),
            np.array([4.0]),
            np.array([0.25]),
        ),
    )
    assert vbmc.function_logger.y_orig[0] == 4.0
    assert vbmc.function_logger.S[0] == 0.25
